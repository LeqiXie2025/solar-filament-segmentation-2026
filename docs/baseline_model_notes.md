# 当前 baseline 模型说明

这版代码的目标不是先追求复杂，而是先建立一条能复现、能用 GPU 跑、能生成 Kaggle 提交文件的稳定主线。

## 现在用了什么思路

模型是一个 PyTorch U-Net。它把太阳图像当成语义分割问题：输入一张太阳 H-alpha 图，输出每个像素是不是 filament。

输入不是单通道灰度图，而是 3 个通道：

- 原始灰度图；
- 经过太阳盘径向亮度校正和 CLAHE 增强后的图；
- black-hat 形态学增强图，用来突出较暗、较细的结构。

这样做的原因是 filament 通常是暗色细长结构，普通灰度图里容易被太阳盘整体亮度变化盖住。

## 相比队友分支修了什么

- 去掉了写死在队友电脑上的绝对路径，改成项目内相对路径。
- 自动使用 CUDA；你的电脑已验证可以使用 `NVIDIA GeForce RTX 5070 Laptop GPU`。
- 修复了径向亮度校正里没有真正写回数组的问题。
- 没有在 `forward()` 里动态创建神经网络层，避免参数不训练、权重无法稳定保存的问题。
- 训练使用 `BCEWithLogitsLoss + DiceLoss`，比单独 BCE 更适合小面积细目标。
- 验证集按文件名时间顺序切分，减少随机切分导致的虚高分。
- 推理使用滑窗，在 2048x2048 原图上分块预测，再拼回完整 mask。
- 提交文件输出为官方需要的 `filament_id,segmentation_rle` 两列。

## 下一步怎么优化

第一优先级是下载完整训练数据并跑通第一版：

```powershell
D:\python3.10.11\python.exe -m src.train --epochs 20 --image-size 768 --batch-size 2
D:\python3.10.11\python.exe -m src.predict --output submissions/submission_unet_baseline.csv
```

跑出第一版本地验证分数后，再做这些实验：

- 调整阈值：例如 `0.35, 0.40, 0.45, 0.50, 0.55`。
- 调整 patch 大小：`512, 768, 1024`。
- 改用更强骨干：U-Net++ / ResNet34 / EfficientNet。
- 增加 5-fold 时间分组验证。
- 加 TTA：水平翻转、垂直翻转、90 度旋转。
- 后处理只保守过滤小噪点，不用 leaderboard 猜答案。

## 当前限制

本地还没有完整官方训练数据，所以当前只完成了代码和 GPU smoke test，暂时还没有真实训练分数。
