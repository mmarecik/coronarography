import torch
import os

BATCH_SIZE = 2
RESIZE_TO = 512
NUM_EPOCHS = 3
NUM_WORKERS = 4

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")

DATA_DIR = "./data"
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
TRAIN_ANNOTATION = os.path.join(DATA_DIR, "train_labels.csv")
TEST_ANNOTATION = os.path.join(DATA_DIR, "test_labels.csv")

CLASSES = [
    '__background__', 'Stenosis'
]
NUM_CLASSES = len(CLASSES)

OUT_DIR = './outputs'

NUM_SAMPLES_TO_VISUALIZE = 9