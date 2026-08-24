# Solar Filament Segmentation 2026 AI Handoff

更新日期：2026-08-24

这份文档给任何新接手的 AI 或队友看，用来快速理解：我们在做什么比赛、最后交付什么、代码目前跑到哪一步、当前模型的主要问题是什么、下一步应该怎么继续训练。

## 1. 比赛是什么

比赛链接：https://www.kaggle.com/competitions/filament-segmentation-2026

这是一个太阳图像分割比赛。输入是一张张太阳 H-alpha 观测图，目标是在图里找出太阳暗条，也就是 filament，并给出它们所在的像素区域。

最简单的理解：

- 图片里有太阳圆盘和一些细长、暗色、不规则的结构。
- 模型要把这些结构圈出来。
- 圈得越接近官方隐藏答案，分数越高。

这不是普通分类题，不是回答“有没有暗条”，而是像素级分割题。模型最后输出的不是一个类别，而是一堆 mask，再编码成 Kaggle 要求的 RLE 字符串。

## 2. 最后要提交什么

Kaggle 最终提交的是一个 CSV 文件，不是 GitHub 分支，也不是模型权重。

当前预测脚本会生成：

```text
submissions/*.csv
```

CSV 结构是：

```text
filament_id,segmentation_rle
```

含义：

- `filament_id`：某张测试图里的第几个预测实例，例如 `xxx_1`、`xxx_2`。
- `segmentation_rle`：这个实例 mask 的游程编码。Kaggle 用它还原出模型预测的像素区域。

GitHub 用来存代码、文档、实验记录。Kaggle 用来上传最终预测 CSV。

## 3. 评分方式

比赛看的是预测 mask 和官方隐藏 mask 的重合程度，核心思想接近 Dice。

小白版解释：

- 你圈出的区域和标准答案重合越多，越好。
- 你漏掉真实暗条，会扣分。
- 你多圈了背景，也会扣分。
- 只把面积调大或调小不是真正提升模型，可能公开榜偶然变好，但私榜会掉。

当前我们已经决定不刻意迎合 public LB，而是优先训练更稳的模型。

## 4. 当前仓库位置

本地路径：

```text
C:\Users\xlq20\Documents\申请\solar-filament-segmentation-2026
```

主要目录：

- `src/`：训练、预测、数据集、mask 编码、模型工厂。
- `data/raw/`：Kaggle 官方数据，本地使用，不建议提交到 Git。
- `models/`：训练好的模型权重，本地使用，不建议提交到 Git。
- `submissions/`：生成的 Kaggle 提交 CSV，本地使用，默认不进 Git。
- `docs/`：比赛说明、交接文档、计划。
- `experiments/`：实验记录。

## 5. 当前代码能力

目前代码已经支持：

- PyTorch GPU 训练。
- 读取 COCO 风格训练标注。
- 把标注转成二值 mask。
- patch 训练，优先抽到有暗条的区域。
- U-Net / U-Net++ / DeepLabV3+ 等 `segmentation_models_pytorch` 模型。
- 滑窗预测整张测试图。
- 简单 TTA：水平翻转、水平+垂直翻转。
- 后处理：连通区域、形态学 close/openclose、最小面积过滤。
- 输出 Kaggle 需要的 CSV。

## 6. 已确认环境

本机可以用 GPU：

```text
NVIDIA GeForce RTX 5070 Laptop GPU
```

常用 Python：

```powershell
D:\python3.10.11\python.exe
```

安装依赖：

```powershell
D:\python3.10.11\python.exe -m pip install -r requirements.txt
```

## 7. 当前已知结果

### 7.1 Public LB

用户反馈：

- 之前有一次提交大约是 `0.28`。
- 后来一次提交变成 `0.27`，比前一次低 `0.01`。

这说明降低阈值、放出更多小区域的路线不一定稳定。它可能增加召回，但也容易带来更多误检。

### 7.2 本地验证

之前的 U-Net ResNet34 在旧验证方式上大约到过：

```text
val_dice ≈ 0.825
```

但旧验证切分可能偏乐观。现在训练改成了更稳的随机验证集：

```text
--split random
```

当前正在跑的稳定微调任务：

```powershell
D:\python3.10.11\python.exe -m src.train `
  --arch smp_unet `
  --encoder-name resnet34 `
  --epochs 12 `
  --image-size 512 `
  --batch-size 4 `
  --samples-per-epoch 5000 `
  --split random `
  --positive-patch-prob 0.75 `
  --lr 0.00008 `
  --resume models\smp_unet_resnet34_512_e5.pth `
  --output models\smp_unet_resnet34_512_stable_finetune_e12.pth `
  --patience 4 `
  --grad-clip 1.0
```

目前已观察到：

```text
epoch 1: loss=0.19470, val_dice=0.74907, lr=8.00e-05, saved
epoch 2: loss=0.18066, val_dice=0.68145, lr=8.00e-05
epoch 3: loss=0.17188, val_dice=0.74770, lr=8.00e-05
epoch 4: loss=0.16696, val_dice=0.65190, lr=4.00e-05
epoch 5: loss=0.15966, val_dice=0.74899, lr=4.00e-05, early stop
```

解释：

- 训练 loss 一直下降，说明模型在训练集上学得更好。
- 但验证 Dice 波动很大，且第 4 轮明显下降，说明继续微调同一个 U-Net 有过拟合风险。
- 第 5 轮几乎追平第 1 轮，但没有超过，因此 early stop 是合理的。
- 当前保存的最好 checkpoint 是第 1 轮的 `models\smp_unet_resnet34_512_stable_finetune_e12.pth`，best_val_dice 为 `0.74907`。

## 8. 当前核心问题

### 问题 1：模型会过拟合

训练 loss 越来越低，但验证分数没有同步变好。这说明模型可能在记训练图的细节，而不是学到更通用的暗条规律。

处理方向：

- 使用更严格的随机验证集。
- 开启 early stopping。
- 降低学习率。
- 不盲目训练很多 epoch。
- 保留验证集最好的 checkpoint，而不是最后一个 checkpoint。

### 问题 2：暗条形状细、长、断裂

太阳暗条不是规则物体，常常细长、弯曲、断断续续。模型可能会：

- 只学到最明显的粗暗条。
- 漏掉细暗条。
- 把噪声、太阳纹理误判成暗条。
- 把一个暗条切成多个实例，或者把多个暗条连成一个实例。

处理方向：

- 增强模型结构多样性，例如 U-Net++、DeepLabV3+。
- 做更稳的增强，而不是只调提交阈值。
- 后续可以看验证图可视化，判断是假阳性多还是漏检多。

### 问题 3：本地验证和 Kaggle LB 不是同一件事

Kaggle public LB 只看测试集的一部分。为了冲 public LB，可以调阈值、面积过滤、实例切分方式，但这容易过拟合 public LB。

现在用户明确要求：

```text
focus on 模型训练，不用刻意迎合 LB
```

所以后续优化优先级应该是：

1. 本地验证更稳。
2. 模型泛化更好。
3. 最后再做少量合理后处理。

不要做根据 public LB 猜答案排列、硬调提交文件、伪造规则内外边界的操作。

## 9. 当前应该怎么继续

### 第一步：结束当前 U-Net 稳定微调

如果后续 epoch 不能超过 `0.74907`，不要继续长时间训练这条线。保留第 1 轮 checkpoint 即可。

判断标准：

- 如果验证 Dice 连续几轮不超过当前最好值，停止。
- 不因为 training loss 下降就继续训练。

### 第二步：训练不同结构的模型

推荐下一条线：DeepLabV3+ ResNet34。

命令：

```powershell
D:\python3.10.11\python.exe -m src.train `
  --arch smp_deeplabv3plus `
  --encoder-name resnet34 `
  --encoder-weights imagenet `
  --epochs 10 `
  --image-size 512 `
  --batch-size 4 `
  --samples-per-epoch 5000 `
  --split random `
  --positive-patch-prob 0.75 `
  --lr 0.00015 `
  --output models\smp_deeplabv3plus_resnet34_512_random_e10.pth `
  --patience 4 `
  --grad-clip 1.0
```

如果显存不够，把 `--batch-size 4` 改成 `--batch-size 2`。

### 第三步：训练/保留 U-Net++ 作为多样性模型

已经跑过 U-Net++ ResNet34：

```text
best_val_dice=0.81896
```

这个结果在旧切分上接近 U-Net，但没有明显超过。它暂时不一定单独提交，但可以作为后续 ensemble 的候选。

### 第四步：增加训练侧验证质量

建议后续给 `src/train.py` 加一个验证阈值扫描，例如每个 epoch 同时看：

```text
0.35, 0.40, 0.45, 0.50, 0.55
```

原因：

- 现在验证固定用 `config.PRED_THRESHOLD=0.45`。
- 有时模型本身变好了，但最佳阈值变了，固定阈值会误导判断。
- 阈值扫描是训练评估改进，不是 public LB 投机。

### 第五步：做可视化误差分析

建议输出几张验证集图片：

- 原图。
- 真实 mask。
- 预测概率图。
- 二值预测 mask。
- 假阳性和漏检区域。

要回答的问题：

- 是漏检多，还是误检多？
- 是边界错，还是整条暗条没找到？
- 是小暗条错得多，还是大暗条错得多？
- 是太阳边缘错得多，还是中心区域错得多？

这一步比盲目调阈值更有价值。

## 10. 推荐运行顺序

### 训练

先继续或复现实验：

```powershell
D:\python3.10.11\python.exe -m src.train `
  --arch smp_unet `
  --encoder-name resnet34 `
  --epochs 12 `
  --image-size 512 `
  --batch-size 4 `
  --samples-per-epoch 5000 `
  --split random `
  --positive-patch-prob 0.75 `
  --lr 0.00008 `
  --resume models\smp_unet_resnet34_512_e5.pth `
  --output models\smp_unet_resnet34_512_stable_finetune_e12.pth `
  --patience 4 `
  --grad-clip 1.0
```

如果 U-Net 不再提升，跑 DeepLabV3+：

```powershell
D:\python3.10.11\python.exe -m src.train `
  --arch smp_deeplabv3plus `
  --encoder-name resnet34 `
  --encoder-weights imagenet `
  --epochs 10 `
  --image-size 512 `
  --batch-size 4 `
  --samples-per-epoch 5000 `
  --split random `
  --positive-patch-prob 0.75 `
  --lr 0.00015 `
  --output models\smp_deeplabv3plus_resnet34_512_random_e10.pth `
  --patience 4 `
  --grad-clip 1.0
```

### 预测提交

生成一个合理、不极端的提交文件：

```powershell
D:\python3.10.11\python.exe -m src.predict `
  --checkpoint models\smp_unet_resnet34_512_stable_finetune_e12.pth `
  --output submissions\submission_stable_model.csv `
  --tile-size 512 `
  --stride 256 `
  --batch-size 8 `
  --threshold 0.50 `
  --min-area 750 `
  --morphology close `
  --instance-method connected `
  --tta hvflip
```

注意：这只是生成 Kaggle CSV，不代表一定提升 LB。现在主线目标是模型更稳。

## 11. 不建议继续做的事

不建议为了 public LB 做这些事：

- 根据 public LB 反复猜测试集标签。
- 大量提交只为了试出隐藏数据分布。
- 只调阈值和面积过滤，忽略模型本身。
- 把 public LB 高的别人的提交硬混进来但没有理解来源。
- 修改评分规则、数据格式或用不符合比赛规则的数据。

这些做法可能短期涨 public LB，但私榜容易掉，也不利于真正学会这道题。

## 12. 给下一个 AI 的直接任务

接手后先做这几件事：

1. 检查当前训练进程是否还在跑。
2. 记录每个 epoch 的 `loss`、`val_dice`、`lr`。
3. 如果当前 U-Net 微调没有超过 `0.74907`，停止该训练，保留最佳 checkpoint。
4. 启动 DeepLabV3+ 或另一个结构，优先寻找低相关、泛化更好的模型。
5. 给训练脚本加验证阈值扫描，避免固定 `0.45` 误判模型。
6. 做验证集可视化误差分析，再决定该增强数据、换 loss、换模型，还是调整后处理。

一句话总结：

当前项目已经能完整训练和提交，但主要瓶颈不是“不会生成 submission”，而是模型泛化还不稳。现在应该用更严格的本地验证、早停、多结构模型和误差可视化来提升真实分割能力，而不是继续追 public LB 的偶然涨分。
