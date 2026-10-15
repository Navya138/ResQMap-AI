# ResQMap AI - from disaster images to response priorities

## ONE-CLICK RUN (Windows)
Double-click **RUN_ME.bat**. It installs the packages, then:
- if an `xbd` folder exists (real dataset) it uses that;
- otherwise it **generates a synthetic xBD-style dataset** (`make_synthetic.py`) so the whole pipeline still runs;
- then it does the 80:20 split, trains the damage model on CPU, builds the demo scenes, and opens the app.

IMPORTANT: results from the synthetic data only prove the pipeline works. They are NOT real disaster results. Say so in your presentation, and swap in real xBD images if you can get even a small part of it.

Pipeline: **pre/post satellite tile -> building footprints (U-Net) -> damage class per building (ResNet18 on pre+post crop) -> zone clustering + priority score -> map + assessment queue (Streamlit)**

## 1. Setup
```bash
pip install -r requirements.txt
```
(`pip install -r requirements-seg.txt` only if you want to train the building-finder `train_seg.py`.)
Download xBD / xView2 (free registration at xview2.org) and unzip so you have `<root>/<tier>/images` and `<root>/<tier>/labels`.
Tip: for a hackathon, use only 2-3 disaster events (e.g. hurricane-michael, palu-tsunami, santa-rosa-wildfire).

## 2. Prepare data (80:20 split)
```bash
python prepare_data.py --root /path/to/xbd --max_per_class 8000
```
- Pools all labelled tiers, then splits **80% train / 20% test per disaster event** (seed 42) -> `data/splits.json`
- Cuts pre+post building crops -> `data/crops_train.csv`, `data/crops_test.csv`
- Train crops are class-capped (speed + imbalance); test cap is 25% of that. For final reported numbers, rerun with `--max_per_class 0`.

## 3. Train
```bash
python train_seg.py --epochs 8        # building footprints   -> models/seg.pt, results/seg_metrics.json
python train_damage.py --epochs 10    # damage classifier     -> models/damage.pt, results/damage_metrics.json, confusion_matrix.png
```
Use a GPU (Colab/Kaggle is fine). On CPU, lower `--max_train` and `--epochs`.

## 4. Precompute demo scenes (from the held-out 20%)
```bash
python infer.py --n 4
python infer.py --n 4 --use_gt     # safety net: ground-truth footprints, predicted damage
```

## 5. Run the demo
```bash
streamlit run app.py
```

## Priority score
`score = 100 x (w_sev x norm(sum(severity x confidence)) + w_den x norm(damaged buildings per hectare) + w_fac x facility_proximity)`
Severity weights: minor 1, major 3, destroyed 5 (in `config.py`). Zones with >40% low-confidence buildings are flagged "needs verification". Upload a CSV of hospitals/schools (`lon,lat,name`) to add context.

## Honest limitations (put these on a slide)
- Minor vs major damage is the hardest distinction; the dataset is heavily imbalanced toward no-damage.
- Neighbouring tiles of one scene can land in both train and test, so scores are optimistic. For a stricter test, hold out a whole event.
- Output is decision support for human inspectors, not an automatic verdict.
