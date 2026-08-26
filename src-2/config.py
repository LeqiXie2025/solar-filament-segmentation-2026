# src/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(r"C:\Users\love5\Desktop\solar-filament-segmentation-2026-master\data\raw")

TRAIN_IMG_DIR = DATA_DIR / "train_images"
TEST_IMG_DIR = DATA_DIR / "test_images"
ANNOTATION_PATH = DATA_DIR / "train_annotations.json"

MODEL_DIR = PROJECT_ROOT / "models"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

# ===== 训练配置（目标 0.85+） =====
IMAGE_SIZE = 512
BATCH_SIZE = 2
EPOCHS = 60
LEARNING_RATE = 1.5e-4           # 关键：回到 2e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
VAL_FRACTION = 0.2
SEED = 42

IN_CHANNELS = 3
BASE_CHANNELS = 32

# ===== 推理配置 =====
PRED_THRESHOLD = 0.30
MIN_COMPONENT_AREA = 50
INFER_TILE_SIZE = 512
INFER_STRIDE = 384