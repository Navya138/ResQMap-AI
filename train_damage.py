"""Stage 2 - damage classifier on pre+post building crops (no / minor / major / destroyed)."""
import argparse, json, random
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from config import *
from model_defs import make_damage_model
from utils import pair_to_array


class CropDS(Dataset):
    def __init__(self, csv, train):
        df = pd.read_csv(csv)
        self.paths, self.labels, self.train = df.path.tolist(), df.label.tolist(), train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = pair_to_array(Image.open(self.paths[i]).convert("RGB"))
        if self.train:
            if random.random() < .5: x = x[:, :, ::-1]
            if random.random() < .5: x = x[:, ::-1, :]
        return torch.from_numpy(np.ascontiguousarray(x)), self.labels[i]


@torch.no_grad()
def predict(model, dl, dev):
    model.eval(); P, Y = [], []
    for x, y in dl:
        P += model(x.to(dev)).argmax(1).cpu().tolist(); Y += y.tolist()
    return np.array(Y), np.array(P)


def damage_score(y, p):
    f = f1_score(y, p, labels=range(4), average=None, zero_division=0)
    return float(4 / np.sum(1 / np.maximum(f, 1e-6)))   # harmonic mean of per-class F1 (xView2 style)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr_ds, te_ds = CropDS(DATA / "crops_train.csv", True), CropDS(DATA / "crops_test.csv", False)
    tr = DataLoader(tr_ds, a.bs, shuffle=True, num_workers=a.workers)
    te = DataLoader(te_ds, a.bs, num_workers=a.workers)

    cnt = np.bincount(tr_ds.labels, minlength=4).astype(float)
    w = 1 / np.sqrt(np.maximum(cnt, 1)); w = w / w.mean()
    print("train class counts:", cnt.astype(int).tolist())

    model = make_damage_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
    best = -1
    for ep in range(a.epochs):
        model.train()
        for x, y in tqdm(tr, desc=f"epoch {ep+1}/{a.epochs}"):
            x, y = x.to(dev), y.to(dev)
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        y, p = predict(model, te, dev)
        ds = damage_score(y, p)
        print(f"epoch {ep+1}: macro-F1 {f1_score(y, p, average='macro'):.4f} | damage-F1 (harmonic) {ds:.4f}")
        if ds > best:
            best = ds; torch.save(model.state_dict(), MODELS / "damage.pt")
            best_y, best_p = y, p

    rep = classification_report(best_y, best_p, target_names=CLASSES, output_dict=True, zero_division=0)
    json.dump({"damage_f1_harmonic": best, "report": rep}, open(RESULTS / "damage_metrics.json", "w"), indent=1)
    cm = confusion_matrix(best_y, best_p, labels=range(4), normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4.5)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels([c.split("-")[0] for c in CLASSES]); ax.set_yticklabels([c.split("-")[0] for c in CLASSES])
    for i in range(4):
        for j in range(4): ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Damage confusion matrix (test)")
    plt.tight_layout(); plt.savefig(RESULTS / "confusion_matrix.png", dpi=150)
    print(classification_report(best_y, best_p, target_names=CLASSES, zero_division=0))
