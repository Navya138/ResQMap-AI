"""Stage 1 - building footprint segmentation (U-Net/ResNet34) on the PRE-disaster image."""
import argparse, json, random
import numpy as np, torch
import segmentation_models_pytorch as smp
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from config import *
from utils import mask_from_label, MEAN3, STD3


class SegDS(Dataset):
    def __init__(self, recs, train, crop=512):
        self.recs, self.train, self.crop = recs, train, crop

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = np.asarray(Image.open(r["pre"]).convert("RGB"))
        m = mask_from_label(r["label"], img.shape[0])
        if self.train:
            c = self.crop
            y, x = random.randint(0, img.shape[0] - c), random.randint(0, img.shape[1] - c)
            img, m = img[y:y + c, x:x + c], m[y:y + c, x:x + c]
            if random.random() < .5: img, m = img[:, ::-1], m[:, ::-1]
            if random.random() < .5: img, m = img[::-1], m[::-1]
        x = ((img.astype(np.float32) / 255 - MEAN3) / STD3).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(m[None]).astype(np.float32))


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval(); inter = union = 0
    for x, y in dl:
        p = (torch.sigmoid(model(x.to(dev))) > .5).float().cpu()
        inter += (p * y).sum().item(); union += ((p + y) > 0).float().sum().item()
    return inter / max(union, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_train", type=int, default=1500, help="images used for training (speed)")
    ap.add_argument("--max_test", type=int, default=200)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sp = json.load(open(DATA / "splits.json"))
    random.Random(SEED).shuffle(sp["train"]); random.Random(SEED).shuffle(sp["test"])
    tr = DataLoader(SegDS(sp["train"][:a.max_train], True), a.bs, shuffle=True, num_workers=a.workers, drop_last=True)
    te = DataLoader(SegDS(sp["test"][:a.max_test], False), 2, num_workers=a.workers)

    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), 3e-4)
    dice, bce = smp.losses.DiceLoss("binary"), torch.nn.BCEWithLogitsLoss()
    amp = dev == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best = 0
    for ep in range(a.epochs):
        model.train()
        for x, y in tqdm(tr, desc=f"epoch {ep+1}/{a.epochs}"):
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev, enabled=amp):
                out = model(x); loss = dice(out, y) + bce(out, y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        iou = evaluate(model, te, dev)
        print(f"epoch {ep+1}: test IoU {iou:.4f}")
        if iou > best:
            best = iou; torch.save(model.state_dict(), MODELS / "seg.pt")
    json.dump({"building_iou_test": best}, open(RESULTS / "seg_metrics.json", "w"))
