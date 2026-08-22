# 文件用途说明

这个文档用来解释当前仓库里每个文件和目录的作用。现在这个仓库只是比赛规划和文档仓库，还没有放训练代码。等我们完全看懂官方数据结构之后，再开始加 PyTorch 训练代码。

## 根目录

### `README.md`

项目首页。

这里应该放：

- 比赛名称和 Kaggle 官方链接；
- 当前项目状态；
- 这个仓库主要存放什么；
- 不要提交 Kaggle 原始数据的提醒；
- 仓库目录地图。

这里不要放：

- 很长的实验细节；
- 原始数据；
- 模型代码；
- 私密账号、密钥、`kaggle.json`。

### `.gitignore`

防误传文件。

它现在会忽略：

- `data/raw/` 里的 Kaggle 官方原始数据；
- `data/samples/` 里的本地样例图；
- `submissions/` 里的 Kaggle 提交文件；
- 模型权重和大型生成文件；
- notebook 缓存和 Python 缓存；
- `.env`、`kaggle.json` 等密钥文件。

除非我们认真确认过比赛规则，否则不要随便删除这些忽略规则。

## `docs/`

主要文档目录。这里放稳定的说明、计划和复盘。

### `docs/competition_overview.md`

赛题白话说明。

这里应该放：

- 官方比赛链接；
- 输入是什么；
- 输出是什么；
- Dice 分数怎么理解；
- 太阳暗条分割为什么难；
- 第一版模型建议。

这个文件适合发给队友，或者给其他 AI 看，让对方快速理解我们做的是什么比赛。

### `docs/data_guide.md`

数据说明和检查指南。

这里应该放：

- 官方数据本地保存在哪里；
- 数据里有哪些文件夹和文件类型；
- 图片数量、mask 数量、尺寸、对齐关系怎么检查；
- submission 格式怎么理解；
- EDA 过程中发现的问题。

等完整数据下载好之后，需要把里面的数据清单表补全。

### `docs/project_structure.md`

仓库结构说明。

这里应该放：

- 每个目录以后应该放什么；
- 哪些目录只在本地使用；
- 未来代码、实验记录、报告、提交文件分别放哪里；
- 队友协作时的文件放置规则。

这个文件相当于仓库的“放东西规则”。

### `docs/weekly_plan.md`

按周执行计划。

这里应该放：

- 每周目标；
- 报名、EDA、baseline、验证、优化、后处理、最终提交分别要做什么；
- 每周应该交付什么东西。

如果进度变化，或者某一阶段已经完成，就更新这个文件。

### `docs/download_status.md`

官方数据下载状态记录。

这里应该放：

- Kaggle 比赛 slug；
- 账号是否已经报名；
- 哪些官方文件已经下载到本地；
- 完整数据包是否下载成功；
- 后续继续下载要用的命令。

这个文件只记录本地数据状态，不上传原始数据本身。

### `docs/file_purpose_guide.md`

就是当前这个文件。

这里应该放：

- 当前每个文件和目录的用途；
- 每个文件以后应该写什么；
- 每个文件不应该放什么。

这个文件适合给队友作为入门说明。

## `data/`

本地数据目录。大部分内容默认不上传 GitHub。

### `data/raw/`

官方 Kaggle 原始数据目录。

本地应该放：

- 下载下来的比赛压缩包；
- 解压后的训练图片；
- 解压后的训练 mask 或标注；
- 官方测试图片；
- sample submission 文件；
- 官方 metadata 文件。

默认不要把这些原始文件提交到 GitHub。

### `data/raw/README.md`

说明官方原始数据应该放在 `data/raw/`。

这里应该放：

- Kaggle 下载命令；
- 原始数据不要通过 GitHub 再分发的提醒。

### `data/raw/.gitkeep`

占位文件。

因为空文件夹不会被 Git 记录，所以用它保留 `data/raw/` 这个目录。

一般不用改它。

### `data/raw/file_listing.csv`

本地保存的 Kaggle 官方文件清单。

用途：

- 在完整数据包下载前，先看官方数据目录长什么样；
- 记录官方文件路径和大小；
- 帮助我们判断 train/test/mask/sample submission 在哪里。

这个文件被 Git 忽略，没有上传到 GitHub。

### `data/samples/`

本地样例图目录。

本地应该放：

- 少量官方图片预览；
- image + mask 叠加图；
- prediction + mask 对比图；
- 失败案例截图。

默认不要提交这些图，除非我们确认比赛规则允许公开展示。

### `data/samples/README.md`

说明 `data/samples/` 是干什么的。

这里应该放：

- 可以放哪些样例图；
- 样例图默认不上传 GitHub 的提醒。

### `data/samples/.gitkeep`

占位文件。

它让 GitHub 上能看到 `data/samples/` 这个目录。

### `data/samples/*.jpeg`

本地下载的官方样例图。

当前本地已有：

- `20110120105534Ch.jpeg`
- `20110130110334Ch.jpeg`
- `20110214175414Mh.jpeg`

这些图片用于检查图片格式、尺寸和读取方式。它们没有上传到 GitHub。

## `experiments/`

实验记录目录。

### `experiments/experiments.csv`

主实验表。

以后每次认真训练或提交，都要在这里加一行。

当前表头包括：

- 实验编号；
- 日期；
- 模型；
- 输入尺寸；
- loss；
- 数据增强；
- 本地 validation Dice；
- Kaggle public score；
- submission 文件名；
- git commit；
- 备注。

这个表非常重要，因为它能把每个分数对应到具体代码版本和实验配置。

## `reports/`

分析报告目录。

以后这里应该放：

- EDA 总结；
- 数据质量问题；
- 模型对比分析；
- 失败案例分析；
- 最终方法说明；
- 最终提交说明。

当前状态：

- 目录存在，但还没有提交具体报告文件。

## `submissions/`

本地 Kaggle 提交文件目录。

本地应该放：

- 生成的 submission CSV；
- 最终候选提交文件；
- 临时验证提交文件。

默认不要把这些提交文件上传 GitHub。每次真正提交 Kaggle，只需要在 `experiments/experiments.csv` 里记录文件名、分数和备注。

### `submissions/README.md`

说明提交文件如何管理。

这里应该放：

- submission 文件只在本地保存的提醒；
- 每次真实提交都要记录到实验表的要求。

### `submissions/.gitkeep`

占位文件。

它让 GitHub 上能看到 `submissions/` 这个目录。

## 以后开始写代码时再新增的文件

现在先不要急着加这些文件。等我们下载并看懂数据结构后，再开始写：

- `requirements.txt`：训练和推理需要安装的 Python 包；
- `src/dataset.py`：读取图片和 mask；
- `src/train.py`：训练模型；
- `src/infer.py`：生成测试集预测和 submission；
- `src/metrics.py`：Dice 等指标；
- `src/postprocess.py`：阈值、连通域、小噪点清理；
- `configs/*.yaml`：模型配置、训练参数；
- `notebooks/01_eda.ipynb`：数据探索 notebook。

新增这些文件后，也要同步更新这个说明文档，让队友知道每个文件怎么用。
