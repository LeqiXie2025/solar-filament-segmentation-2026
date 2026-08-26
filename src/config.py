from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(r"C:\Users\love5\Desktop\solar-filament-segmentation-2026-master\data\raw")

TRAIN_IMG_DIR = DATA_DIR / "train_images"
TEST_IMG_DIR = DATA_DIR / "test_images"
ANNOTATION_PATH = DATA_DIR / "train_annotations.json"

MODEL_DIR = PROJECT_ROOT / "models"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

IMAGE_SIZE = 768
BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
VAL_FRACTION = 0.2
SEED = 42

IN_CHANNELS = 3
BASE_CHANNELS = 32

PRED_THRESHOLD = 0.45
MIN_COMPONENT_AREA = 80
INFER_TILE_SIZE = 768
INFER_STRIDE = 512
