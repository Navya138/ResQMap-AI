from pathlib import Path

SEED = 42
TEST_RATIO = 0.2                      # 80:20 train/test split
CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
CLS2ID = {c: i for i, c in enumerate(CLASSES)}
SEV_WEIGHT = {"no-damage": 0, "minor-damage": 1, "major-damage": 3, "destroyed": 5}

CROP = 128                            # building crop size (pre and post each CROP x CROP)
PAD = 8                               # context pixels around each building
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DATA = Path("data")
MODELS = Path("models")
RESULTS = Path("results")
DEMO = Path("demo")
for p in (DATA, MODELS, RESULTS, DEMO):
    p.mkdir(exist_ok=True)
