# 太阳暗条实例分割：0.38 发布版

这个分支只保留 **0.38 模型的训练、验证、测试集导出，以及它们实际依赖的辅助代码**。无关历史实验和过时规划文档已移除；核心训练/验证/导出源码、冻结参数、证据与三份发布 CSV 未改动；补回原快照遗漏的矩形裁剪依赖函数，已有函数实现保持不变。

- 分支：`0.38`。仓库默认分支 `master` 是旧版，请不要混用。
- 方法：YOLO 普通框检测 → 扩边裁剪 → U-Net/ResNet34 掩码精修 → 固定形状融合与旋转框候选组装 → 边界修正 → RLE CSV。
- 本版改进：训练多边形标签采用 COCO 栅格化，不改变既有推理参数。
- Public **0.38 来自用户实际回分反馈**；本仓库没有独立读取平台成绩。141 张本地验证图已用于开发，不能当独立 OOF。
- 发布 CSV：`submissions/submission_r25_coco_target_20261004.csv`，180 图、177 图非空、1227 实例行。
- CSV SHA256：`105668985c949fa330dd1bc22992206ad9e0f70088f67d90e7ea216714fe1a3c`。
- 主精修权重 SHA256：`cf053df4739e8d91504b293e2165b7dbb1c1acd7bf4126087c047689cdcd4162`。

> **先说明复现边界：** GitHub 有代码、配置、历史验证证据与 CSV，但没有官方图片、衍生训练数据和大模型权重。仅 `git clone` 不能直接开始预测。下面提供“加载发布权重重生成 CSV”和“重新训练并验证”两条路线。它们都需要完整资产；尚不能凭公开原始数据从零还原所有上游权重的训练来源。

## 一、现在每个目录有什么

| 目录/文件 | 内容和用途 |
|---|---|
| `README.md` | 本清单及完整 Windows 本地运行步骤 |
| `RELEASE_038.md` | 发布说明入口，指向本指南 |
| `config.py` | 本地数据路径及通用默认参数；**0.38 固定参数以实验配置和导出脚本为准** |
| `requirements.txt` | 当前本地核查环境的直接依赖版本，不是原训练时保存的完整锁文件 |
| `scripts/check_release.py` | 无需安装深度学习库即可检查资产、SHA256及缺失文件；可从已有本地项目复制缺失资产 |
| `docs/cleanup_038.json` | 删除清单与递归依赖保留清单；旧文件可从 Git 历史恢复 |
| `data/raw/` | 官方图片和标注的本地放置位置；Git 只保存说明与占位文件 |
| `src/` | 下表列出的必要模型、裁剪、推理、组装和评估依赖 |
| `experiments/` | 仅保留发布流程实际调用的辅助脚本和必要证据 |
| `experiments/aspect_refiner_r25_20260922/` | 原始训练划分；运行时还需放初始权重 `square_control.pth` |
| `experiments/route_recovery_20261004/r25_proposal/` | 标签解码检查、训练支持审计、180 图测试清单 |
| `experiments/route_recovery_20261004/r25_proposal/repair_01/` | **0.38 的训练/验证/导出入口、固定抽样顺序、参数、验证门和交付证据** |
| `submissions/` | 发布 CSV 和两份旋转框辅助预测 CSV；`c030/c038` 是候选配置，不是对应的 LB 分数 |

需要在本地补齐的 `models/`、`data/clean/` 等目录不含 GitHub 权重或数据；资产检查工具会按相对路径创建它们。

### `src/` 保留文件

| 文件 | 作用 |
|---|---|
| `__init__.py` | Python 包标记 |
| `dataset.py`、`preprocessing.py` | 图像读取、数据和预处理 |
| `crop_refiner_utils.py` | 局部训练记录、扩边裁剪、输入通道与坐标处理 |
| `model.py`、`model_factory.py` | 网络结构及按权重元数据创建网络 |
| `losses.py` | BCE/IoU 等训练目标 |
| `masks.py` | 二值掩码后处理 |
| `predict_detect_refine.py` | 检测/精修辅助、检测 NMS 兼容处理 |
| `predict_detect_refine_multiscale.py` | 翻转 TTA 精修、掩码去重与重叠像素分配 |
| `predict_obb_refine.py`、`obb_utils.py` | 旋转框精修与几何辅助 |
| `assemble_obb_gate_submission.py` | 普通框与旋转框候选的固定组装逻辑 |
| `frozen_mask_quality.py` | RLE 编解码和计分基础函数 |
| `gt_dedup.py`、`instance_quality.py` | 被保留辅助模块导入的标注/实例处理依赖 |
| `metrics.py`、`validation.py`、`predict.py` | 被上述模块递归导入的评估/预测依赖 |
| `proposal_classifier.py` | 权重加载辅助脚本导入的模型依赖；不是本版新增分类器训练实验 |

### `experiments/` 根目录保留文件

| 文件 | 保留理由 |
|---|---|
| `aspect_refiner_r25.py` | 训练与验证调用其中的裁剪、坐标还原和概率预测函数 |
| `evaluate_hbb_adaptation.py` | 提供本版使用的精修权重加载函数 |
| `evaluate_proposal_adaptation.py` | 提供辅助预测 CSV 解码函数 |
| `mask_correction_r17.py` | 提供边界修正网络结构与推理函数；实际加载另一个冻结权重路径，名称保留以兼容历史权重 |
| `export_r25_probe.py` | 发布配置中登记并校验的原导出源码；不是推荐的新运行入口 |

这些名字虽然来自历史实验，但仍是当前运行依赖，不能随意删。**运行下面三个发布入口即可，不需要逐个执行上述辅助脚本。**

### `repair_01/` 文件清单

| 文件 | 用途 |
|---|---|
| `train_paired_rasterizer.py` | 顺序训练原栅格化控制组和 COCO 栅格化候选组，各 6 epoch |
| `validate_paired_rasterizer.py` | 固定 141 图完整流程验证，按每个原始 COCO 标注记录分别计分 |
| `export_test_after_gate.py` | 检查验证门/源文件/权重/图像身份，预测 180 图并校验 CSV |
| `preregistration.json` | 冻结划分、源文件 SHA、训练和导出参数 |
| `sampler_schedule.json` | 两组共享的 3072 次抽样顺序 |
| `validation_gate.json` | 历史开发验证结果和导出条件 |
| `decoder_fixture_audit.json` | 标签栅格化测试记录 |
| `test_export_preflight.json`、`delivery.json` | 历史测试导出检查和最终文件校验记录 |
| `public_feedback.json` | 用户 Public 回分反馈 |
| `release_manifest.json` | 完整外部资产、代码和 SHA256 清单；辨别哪些文件已在 Git 内 |

## 二、准备 Windows 环境

以下命令在 **PowerShell** 执行。建议新建一个干净运行目录，不在正在训练的项目里操作。

```powershell
git clone --branch 0.38 --single-branch https://github.com/LeqiXie2025/solar-filament-segmentation-2026.git D:\solar038
Set-Location D:\solar038
py -3.10 -m venv .venv
$py = 'D:\solar038\.venv\Scripts\python.exe'
& $py -m pip install --upgrade pip
& $py -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r requirements.txt
& $py -m pip check
& $py -c "import torch; print(torch.__version__); print('CUDA:',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

需要 Python 3.10、可用 NVIDIA GPU 和兼容 CUDA 12.8 构建的驱动。最后应显示 `CUDA: True`。CUDA wheel 通常自带运行库，不需要为了这些命令另外编译 CUDA Toolkit。如果没有 `py` 启动器，使用本机 Python 3.10 的绝对路径创建虚拟环境。

依赖版本来自本次维护时已有运行环境，不声称与原训练环境全部字节一致。显存、设备和库版本不同可能影响速度或数值；不能保证重新训练得到同一权重 SHA 或同一 Public 成绩。

## 三、把数据和权重补齐

所有路径相对于 `D:\solar038`。官方数据自行从已获访问权的 Kaggle 比赛下载；不要用其他队友生成的猜测标注替代。

```text
data/raw/train_images/                         原训练图片
data/raw/test_images/                          完整180张测试图片
data/raw/train_annotations.json                官方原标注
data/clean/train_annotations_consensus.json     已冻结的共识训练标注
data/clean/hbb_oof_r4_proposals.json             已冻结的OOF训练框
models/yolo/yolo11s_boxes_1280_e25/weights/best.pt
models/crop_refiner_resnet34_512_consensus_ft.pth
experiments/aspect_refiner_r25_20260922/square_control.pth
experiments/mask_correction_r22_20260918/corrector.pth
experiments/pipeline_diagnosis_r5/detector_boxes.json
experiments/r25_probe_20260924/valpred.csv
experiments/valpred_obb_refine_tta_c030.csv
experiments/valpred_obb_refine_tta_c038.csv
submissions/submission_obb_gate_extras_pq04502_20260901.csv
experiments/route_recovery_20261004/r25_proposal/repair_01/candidate_coco_epoch6.pth
```

最后的候选权重用于直接重生成 0.38 预测；重新训练路线会自行生成它。验证两组还需要 `control_cv2_epoch6.pth`，可由训练入口生成。Git 中已包含两份测试旋转框 CSV，无需重新跑该分支。

**有原本本地项目时，推荐按 SHA 复制，不手动猜版本：**

```powershell
# 先只读检查原项目是否包含完整导出资产
& $py scripts/check_release.py --mode export --asset-root 'C:\Users\xlq20\Documents\申请\solar-filament-segmentation-2026'

# 把缺失且身份核对通过的资产复制到当前clone；不覆盖已有文件
& $py scripts/check_release.py --mode export --copy-from 'C:\Users\xlq20\Documents\申请\solar-filament-segmentation-2026'

# 当前clone再次检查，应返回PASS和退出码0
& $py scripts/check_release.py --mode export
```

复制可能包含很多图片和权重，先保证目标盘空间足够。工具报 `MISSING` 时补对应资产；报 `SHA256_MISMATCH` 时找正确版本，不要删除校验或改成任意文件。`--asset-root` 只用于只读检查，不会让推理自动去那个目录读取；真正运行仍需在本 clone 的相对路径准备文件。

如果队友只有 GitHub，需向项目持有人索取上述衍生资料和权重资产包；仓库不提供它们的虚构下载链接。完整 SHA 清单在 `release_manifest.json`。

## 四、路线A：加载发布权重，重新生成测试CSV（推荐）

这条路线 **不重新训练**，目标是重放 0.38 发布推理。运行前执行资产检查，确保 PASS。

```powershell
Set-Location D:\solar038
$py = 'D:\solar038\.venv\Scripts\python.exe'
& $py scripts/check_release.py --mode export
if ($LASTEXITCODE -ne 0) { throw '资产检查未通过，停止导出' }
& $py -u experiments/route_recovery_20261004/r25_proposal/repair_01/export_test_after_gate.py
```

程序检查历史验证门、已登记输入 SHA、主权重元数据/张量身份、180 张测试图清单和文件 SHA；随后实际运行检测、精修、组装和修正，不读取 test 答案。

输出：

- `submissions/submission_r25_coco_target_20261004_v2.csv` 或下一个可用版本号。发布 CSV 已存在，所以通常不会覆盖原文件；Downloads 同名文件也影响版本号。
- 同时复制到当前 Windows 用户的 `Downloads`。
- `repair_01/delivery.json` 记录实际路径、SHA、180 图覆盖、RLE/尺寸/非重叠、重建检查结果。
- `repair_01/test_inference_checkpoint.jsonl` 保存逐图进度；中断后用同一命令续跑，不手工拼 CSV。

成功应看到 180 张图全部处理完成、CSV 和校验记录生成。历史结果是 177 图非空、3 图空、1227 实例行；实际重放需要核对输出，不能只看文件存在。比较 SHA：

```powershell
Get-FileHash .\submissions\submission_r25_coco_target_20261004.csv -Algorithm SHA256
# 把下面名称改成delivery.json记载的实际新文件
Get-FileHash .\submissions\submission_r25_coco_target_20261004_v2.csv -Algorithm SHA256
```

字节一致才叫相同 CSV；不同则先排查依赖环境和输出，不能直接声称是同一发布结果。程序不会上传 Kaggle。

## 五、路线B：重新训练控制组和候选组，再验证、导出

请使用 **另一个干净 clone**，不要在路线A已经放入候选权重或积累测试缓存的目录重训，否则会触发“不覆盖最终权重”保护。

1. 按第二节新建例如 `D:\solar038_retrain` 和虚拟环境；下面 `$py` 指向这个环境。
2. 只复制训练前资产，不复制已有最终候选/控制权重：

```powershell
Set-Location D:\solar038_retrain
$py = 'D:\solar038_retrain\.venv\Scripts\python.exe'
& $py scripts/check_release.py --mode train --copy-from 'C:\Users\xlq20\Documents\申请\solar-filament-segmentation-2026'
& $py scripts/check_release.py --mode train
if ($LASTEXITCODE -ne 0) { throw '训练资产不完整' }
& $py -u experiments/route_recovery_20261004/r25_proposal/repair_01/train_paired_rasterizer.py
```

固定训练：566 fit 图、141 验证图；两组同初始精修权重、同 3072 抽样、同 seed，每组 6 epoch × 512 样本，batch 1、累积 4；IoU+BCE，AdamW，decoder LR `3e-5`、encoder layer4 LR `3e-6`，其余 encoder 冻结，所有 BatchNorm 冻结。唯一对照变量为多边形标签栅格化方式。两组顺序运行，不能另开进程争用 GPU。

训练产物在 `repair_01/`：`control_cv2_epoch6.pth`、`candidate_coco_epoch6.pth`、各组 resume 权重、历史损失日志和 `state.json`。中断可用同一命令从已有 epoch resume；若最终权重已存在，脚本拒绝重新覆盖。不要将训练完成等同于提高成绩。

3. 补齐验证资产并运行完整验证：

```powershell
& $py scripts/check_release.py --mode validate --copy-from 'C:\Users\xlq20\Documents\申请\solar-filament-segmentation-2026'
& $py scripts/check_release.py --mode validate
if ($LASTEXITCODE -ne 0) { throw '验证资产不完整' }
& $py -u experiments/route_recovery_20261004/r25_proposal/repair_01/validate_paired_rasterizer.py
```

验证会重新写当前运行的 `validation_gate.json`、两组预测 CSV、逐图计分 CSV 和缓存；Git 仍保存历史发布证据。它按原始 COCO 标注记录分别分组，COCO 解码，严格 IoU > 0.5 后汇总，不能把同名图片不同标注记录扁平合并。

通过条件：候选相对控制 PQ ≥ `0.002`，日期簇 bootstrap 95% 下界 > 0；候选相对冻结原基线 PQ/TP 不下降、FP 增幅 ≤ 5%。这是开发验证，不是独立确认。

4. **只有本轮验证门通过才导出**：先按路线A准备180测试图，然后运行 `check_release.py --mode export` 和 `export_test_after_gate.py`。检查工具使用本轮验证门记录的候选 SHA；不会假定重新训练的权重应与历史候选字节一致。验证不通过时保留结果，不修改门或伪造历史成绩。

## 六、常见失败与处理

| 情况 | 处理 |
|---|---|
| `ModuleNotFoundError` | 用上述 `$py -m pip ...` 安装依赖，确保运行和安装是同一个虚拟环境 |
| CUDA不可用 | 检查 NVIDIA 驱动、CUDA版 torch 和 `torch.cuda.is_available()`；原入口没有CPU训练替代 |
| 缺权重/标注或SHA不符 | 看 `check_release.py` 输出，补正确资产；不能用另一个实验的同名文件替代 |
| CUDA out of memory | 先关闭其他GPU任务；发布参数固定，改batch/分辨率属于另一次运行，不能未经校验宣称精确复现 |
| `FileExistsError` 最终权重已存在 | 训练用新的干净clone；不要删除发布模型或把旧输出当本轮训练 |
| 导出验证门不通过 | 查看本轮开发指标；不绕过门、不把失败候选叫0.38版本 |
| 新CSV与发布SHA不同 | 核对软件环境、资产与模型身份，再看掩码差异；不靠文件名判断同一版本 |

## 七、维护边界

入口检查发现旧快照遗漏 `crop_bounds_from_bbox_rect`，已从实际生成0.38的本地依赖补回；详见清理记录。

本次清理只更新 `0.38` 发布分支；后台概率融合等新实验仍在另一工作目录，未混入此发布模型。69 个无关历史文件的删除清单及保留依赖在 `docs/cleanup_038.json`。它们仍可从清理前提交 `cc21ffb` 查看，不是删除全部 Git 历史。

本指南已验证代码依赖、资产哈希检查和静态入口；本次整理不重新占用 GPU 完整训练或重跑180图。不得把维护检查说成新LB回分或新精度提升。
