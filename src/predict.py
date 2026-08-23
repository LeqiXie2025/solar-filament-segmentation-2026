# src/predict.py
# 推理脚本

import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import *
from src.dataset import radial_flattening
from src.model import SimpleEdgeAttNet
from src.utils import encode_rle, mask_to_instances


def predict():
    # 1. 加载模型
    model = SimpleEdgeAttNet(in_channels=1, out_channels=1, base_channels=BASE_CHANNELS).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth')))
    model.eval()

    # 2. 获取测试集文件
    test_files = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.jpeg')]

    # 3. 推理
    submission_rows = []

    for filename in tqdm(test_files, desc="Predicting"):
        img_path = os.path.join(TEST_IMG_DIR, filename)

        # 读取并预处理
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        image = radial_flattening(image)

        # 转为Tensor (1, 1, H, W)
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

        # 预测
        with torch.no_grad():
            pred = model(image_tensor).squeeze().cpu().numpy()
            binary = (pred > PRED_THRESHOLD).astype(np.uint8)

        # 拆分为独立实例（连通域分析）
        instances = mask_to_instances(binary, min_area=MIN_AREA)

        # 编码为RLE
        file_id = filename.split('.')[0]
        for i, instance in enumerate(instances, 1):
            rle_str = encode_rle(instance)
            submission_rows.append({
                'filament_id': f"{file_id}_{i}",
                'segmentation_rle': rle_str
            })

    # 4. 保存提交文件
    df = pd.DataFrame(submission_rows)
    df.to_csv(SUBMISSION_PATH, index=False)
    print(f"✅ Submission saved to {SUBMISSION_PATH}")

    # 统计信息
    print(f"Total predicted filaments: {len(df)}")


if __name__ == '__main__':
    predict()
