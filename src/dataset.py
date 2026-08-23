# src/dataset.py
# 数据加载 + 预处理

import os
import cv2
import json
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from pycocotools import mask as cocomask
from config import TRAIN_IMG_DIR


def radial_flattening(image):
    """
    径向亮度校正，解决临边昏暗问题
    Args:
        image: np.ndarray, shape (H, W), dtype=float32, 范围[0,1]
    Returns:
        corrected: np.ndarray, shape (H, W), dtype=float32, 范围[0,1]
    """
    h, w = image.shape
    cy, cx = h // 2, w // 2
    r_max = min(h, w) // 2

    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    disk_mask = dist <= r_max

    # 按径向距离分bin，计算平均亮度
    bins = 100
    bin_edges = np.linspace(0, r_max, bins + 1)
    bin_indices = np.digitize(dist[disk_mask], bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins - 1)

    radial_means = np.zeros(bins)
    for i in range(bins):
        mask = bin_indices == i
        if mask.any():
            radial_means[i] = np.mean(image[disk_mask][mask])

    # 构建径向轮廓并校正
    radial_profile = np.zeros_like(image)
    for i in range(bins):
        mask = (dist[disk_mask] >= i * r_max / bins) & (dist[disk_mask] < (i + 1) * r_max / bins)
        if mask.any():
            radial_profile[disk_mask][mask] = radial_means[i]

    radial_profile = np.clip(radial_profile, 0.01, 1)
    corrected = np.clip(image / radial_profile, 0, 1)

    # CLAHE增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    corrected = (corrected * 255).astype(np.uint8)
    corrected = clahe.apply(corrected)
    corrected = corrected * disk_mask.astype(np.uint8)

    return corrected.astype(np.float32) / 255.0


def load_annotations(ann_file):
    """加载COCO格式的标注文件"""
    with open(ann_file, 'r') as f:
        data = json.load(f)

    img_info = {img['id']: img for img in data['images']}
    img_masks = {}

    for ann in data['annotations']:
        img_id = ann['image_id']
        seg = ann['segmentation']

        if isinstance(seg, list):
            rle = cocomask.frPyObjects(seg, img_info[img_id]['height'], img_info[img_id]['width'])
            mask = cocomask.decode(rle)
        else:
            mask = cocomask.decode(seg)

        if img_id not in img_masks:
            img_masks[img_id] = []
        img_masks[img_id].append(mask.squeeze())

    return img_info, img_masks


class FilamentDataset(Dataset):
    def __init__(self, img_ids, img_info, img_masks, image_size=512, augment=False):
        self.img_ids = img_ids
        self.img_info = img_info
        self.img_masks = img_masks
        self.image_size = image_size
        self.augment = augment

        # 基础变换
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            ToTensorV2(),
        ])

        # 数据增强（只在训练时启用）
        if augment:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.img_info[img_id]
        img_path = os.path.join(TRAIN_IMG_DIR, info['file_name'])

        # ===== 读取图片（带容错） =====
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # 如果读不到，尝试其他后缀
        if image is None:
            alt_path = img_path.replace('.jpeg', '.jpg').replace('.JPG', '.jpg')
            image = cv2.imread(alt_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                img_path = alt_path

        # 如果还是读不到，返回空图并跳过
        if image is None:
            print(f"⚠️ 跳过无法读取的图片: {img_path}")
            dummy_img = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
            dummy_mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
            return dummy_img, dummy_mask, info['file_name']

        image = image.astype(np.float32) / 255.0

        # ===== 预处理 =====
        image = radial_flattening(image)

        # ===== 生成组合掩膜 =====
        masks = self.img_masks.get(img_id, [])
        combined_mask = np.zeros_like(image)
        if masks:
            combined_mask = np.clip(np.sum(masks, axis=0), 0, 1)

        # ===== 应用变换 =====
        transformed = self.transform(image=image, mask=combined_mask)
        image = transformed['image'].float()
        mask = transformed['mask'].float().unsqueeze(0)

        return image, mask, info['file_name']