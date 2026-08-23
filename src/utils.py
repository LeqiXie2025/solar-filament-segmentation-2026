# src/utils.py
# 工具函数：RLE编码/解码、后处理、指标计算

import numpy as np
import cv2
from pycocotools import mask as cocomask


def encode_rle(mask):
    """
    将二值掩膜编码为RLE字符串
    Args:
        mask: np.ndarray, shape (H, W), dtype=uint8, 值为0或1
    Returns:
        rle_str: str, RLE编码字符串
    """
    # pycocotools要求数组是Fortran顺序
    rle = cocomask.encode(np.asfortranarray(mask))
    # rle['counts'] 可能是bytes，需要解码
    if isinstance(rle['counts'], bytes):
        return rle['counts'].decode('utf-8')
    return rle['counts']


def decode_rle(rle_str, shape=(2048, 2048)):
    """
    将RLE字符串解码为二值掩膜
    Args:
        rle_str: str, RLE编码字符串
        shape: tuple, (height, width)
    Returns:
        mask: np.ndarray, shape (H, W), dtype=uint8
    """
    rle = {'size': shape, 'counts': rle_str.encode('utf-8')}
    return cocomask.decode(rle)


def mask_to_instances(binary_mask, min_area=100):
    """
    将二值掩膜拆分为独立的实例（使用连通域分析）
    Args:
        binary_mask: np.ndarray, shape (H, W), dtype=uint8
        min_area: int, 过滤掉小于该面积的实例
    Returns:
        instances: list of np.ndarray, 每个元素是一个实例的掩膜
    """
    num_labels, labels = cv2.connectedComponents(binary_mask)
    instances = []
    for i in range(1, num_labels):
        instance = (labels == i).astype(np.uint8)
        if instance.sum() >= min_area:
            instances.append(instance)
    return instances


def dice_score(pred, target, smooth=1e-6):
    """
    计算Dice系数
    Args:
        pred: np.ndarray, 预测掩膜
        target: np.ndarray, 真实掩膜
    Returns:
        float: Dice系数
    """
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()
    intersection = (pred & target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
