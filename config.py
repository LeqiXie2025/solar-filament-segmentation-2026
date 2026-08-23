# config.py
# 所有配置参数集中管理

import os

PROJECT_ROOT = r'C:\Users\Love5\Desktop\filament-segmentation-2026'
DATA_DIR = r'C:\Users\love5\Desktop\filament-segmentation-2026\data'

# ============ 路径配置 ============
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train_images')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'test_images')
ANNO_PATH = os.path.join(DATA_DIR, 'train_annotations.json')
CHECKPOINT_DIR = './checkpoints'
SUBMISSION_PATH = './submission.csv'

# ============ 训练配置 ============
IMAGE_SIZE = 512              # 训练时resize的尺寸（内存不够就调小）
BATCH_SIZE = 4                # 根据GPU显存调整（T4建议4-8）
EPOCHS = 20                   # 训练轮数
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# ============ 模型配置 ============
BASE_CHANNELS = 32            # U-Net基础通道数
NUM_CLASSES = 1               # 二分类（日珥/背景）

# ============ 推理配置 ============
PRED_THRESHOLD = 0.5          # 概率转二值掩膜的阈值
MIN_AREA = 100                # 过滤掉小于该面积的噪点实例

# ============ 设备 ============
DEVICE = 'cpu'  # 'cpu'/'cuda'
