# Native16 GLIGEN

[English](README.md) | 中文

面向**边界框条件控制、保持辐射测量意义的 16-bit 热红外扩散模型**的可复现训练与推理项目。

本仓库是从原实验工程中提取的独立代码副本，包含 Native16 分支、配置契约、测试和命令行入口；运行时不导入原实验仓库。

> 代码包本身是独立的。模型权重、GLIGEN 基础模型和热红外数据集均作为外部、可版本化资产管理，本仓库不重新分发这些文件。

## 项目范围

包含：

- 原生 `uint16` TIFF 读取、辐射测量归一化和无损 TIFF 输出；
- 单通道 Native16 VAE 适配与 latent affine 校准；
- radiometric latent bridge、输出校准和窗口感知的辐射传输；
- GLIGEN 文本/边界框条件控制、尺度感知 instance fusion、BoxDiff guidance 和模型栈加载；
- ROI counterfactual 训练目标、分布式训练、可恢复 delta checkpoint 和来源校验；
- 固定的 `native16_roi_cf` 配方、父模型栈清单、烟雾测试和 TIFF 完整性校验。

以下内容有意排除：仅支持 RGB8 的路径、检测器训练/推理、数据集构建器、历史实验脚本、数据集、基础模型和 checkpoint 二进制文件。

## 方法概览

```text
uint16 TIFF + 辐射测量 profile
  -> 归一化单通道热红外张量
  -> Native16 VAE / latent affine / radiometric bridge
  -> 带文本 + 归一化边界框的 GLIGEN 去噪器
       + style UNet + grounding + RWT-D adapter 模型栈
       + 尺度感知 instance fusion
  -> 辐射输出校准 + ROI 感知传输
  -> uint16 TIFF、预览图、框选叠加图和 metadata.json
```

训练使用扩散噪声预测以及 `configs/recipes/native16_roi_cf.yaml` 中启用的辅助损失：instance-core、teacher distillation 和 counterfactual grounding。每个保存的 delta 都记录父模型栈指纹，因此不会无提示地加载到不同父模型栈上。

## 目录结构

```text
configs/
  recipes/native16_roi_cf.yaml           # 可复现训练/推理配方
  model_stacks/native16_roi_cf_init.yaml # 有序父 delta + SHA-256
  local/server.env.example               # 主机路径配置模板
src/native16_gligen/
  train.py                               # 分布式 Native16 训练
  sample_native16.py                     # Native16 uint16 TIFF 生成
  model_stack.py                         # 有序 delta 加载与 key contract
  thermal16.py                           # 辐射测量张量/TIFF 编解码
  native16_vae.py                        # 单通道 VAE 适配
  radiometric_*.py                       # bridge、校准和传输
  instance_fusion.py                     # ROI/尺度感知 GLIGEN fusion
scripts/
  train.sh                               # torchrun 训练入口
  generate.sh                            # 推理入口
  copy_parent_artifacts.sh               # 一次性父资产迁移
examples/native16_sample_requests.json   # 请求格式示例
tests/                                   # 单元测试和配置契约测试
```

## 与模型解耦的热红外视频 Pipeline

`src/dwm/pipelines/` 提供完整的训练/推理生命周期，但不内置生产模型。
`ThermalVideoBatch.from_loader` 将当前 loader 的
`vae_images`、`box_condition_images`、嵌套标注、`box_image_sizes` 和 `pts`
适配为统一的 `[B,T,V,...]` 契约。loader 声明秒或毫秒时间单位后，`pts`
会统一为秒；未声明单位的旧 batch 按秒解释。像素框仍与原始
`[height, width]` 成对保留。

后续模型只需实现 `ThermalVideoModel`，并在 `forward_video` 中接收完整的
`[B,T,V]` clip。Pipeline 负责扩散训练、按有效单元加权的梯度累积、分布式
归约、classifier-free guidance、预览/评估回调和更新边界 checkpoint。
编解码及 TIFF/视频写出由注入的模型和 `VideoWriter` 负责；本仓库有意不实现
新的具体模型。

```python
from dwm.common import load_pipeline_from_config

pipeline, config = load_pipeline_from_config(
    "/path/to/pipeline.yaml",
    model=new_video_model,
    optimizer=optimizer,
)
```

配置中的 `pipeline._class_name` 指向
`dwm.pipelines.ThermalVideoPipeline`；运行时依赖通过参数注入，保持模型接口与
数据集编排分离。

安装项目后，还提供以下 CLI：

```text
native16-train
native16-generate
native16-verify-tiffs
native16-verify-parents
```

## 安装

使用 Python 3.10 或更高版本，并先安装与目标 CUDA 驱动匹配的 PyTorch。本项目不固定 `torch` 版本，因为正确的 PyTorch wheel 取决于 CUDA 驱动和运行平台。

```bash
conda create -n native16-gligen python=3.10 -y
conda activate native16-gligen
# 根据目标 CUDA 运行时选择对应的 PyTorch wheel 源。
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e '.[train,eval,test]'
cp configs/local/server.env.example configs/local/server.env
```

在 `configs/local/server.env` 中设置绝对路径：

```bash
export NATIVE16_MODEL_ROOT=/srv/models
export NATIVE16_DATA_ROOT=/srv/datasets
export NATIVE16_ARTIFACT_ROOT=/srv/native16-gligen-artifacts
export NATIVE16_PARENT_ARTIFACT_ROOT=/srv/native16-parent-artifacts
export NATIVE16_PY=/path/to/venv/bin/python
export NATIVE16_TORCHRUN=/path/to/venv/bin/torchrun
export NATIVE16_TRAIN_GPUS=8
export NATIVE16_EVAL_GPUS="0 1"
```

`NATIVE16_ARTIFACT_ROOT` 仅用于可写输出；`NATIVE16_PARENT_ARTIFACT_ROOT` 是不可变父 delta 的只读资产目录。将两者分开可以避免训练输出覆盖推理所需的父资产。

## 外部资产要求

### 基础模型

`NATIVE16_MODEL_ROOT/gligen-1-4-generation-text-box` 必须是完整的 Diffusers GLIGEN SD1.4 pipeline，并包含 `vae/config.json`。代码会拒绝不是 Diffusers 基础模型目录的路径。

### 父模型资产

仓库内的模型栈清单固定了三个有序 UNet delta；配方还要求一个 radiometric bridge 和一个 style adapter：

| 作用 | 必需的相对路径 |
|---|---|
| Style UNet | `native16_ir_style_pilot1000_aligned_7gpu/checkpoints/final.pt` |
| Grounding | `native16_grounding_gentle_full_pilot300_v4/checkpoints/step_00000200.pt` |
| Grounding adapter | `native16_rwtd_pilot200_v1/checkpoints/step_00000150.pt` |
| Radiometric bridge | `native16_radiometric_bridge_stage1_v1/checkpoint-008000.pt` |
| Style adapter | `native16_style_adapter_affine_cont100_v5/final.pt` |

`configs/model_stacks/native16_roi_cf_init.yaml` 保存三个模型栈层的 SHA-256 摘要和 key contract。除非创建新的审计清单，否则不要替换 checkpoint。

如需让部署完全脱离原父工程的 artifact 目录，显式迁移五个资产：

```bash
scripts/copy_parent_artifacts.sh \
  /path/to/legacy-artifacts \
  /srv/native16-parent-artifacts

export NATIVE16_PARENT_ARTIFACT_ROOT=/srv/native16-parent-artifacts
native16-verify-parents --cfg configs/recipes/native16_roi_cf.yaml
```

迁移脚本拒绝覆盖已存在的目标目录；在文件系统支持时使用 copy-on-write reflink。脚本不会修改源资产。style checkpoint 可能较大，运行前请确认磁盘空间。

### 数据集契约

当前配方面向 `FLIR_ADAS_v2` Native16 训练清单：

```text
${NATIVE16_DATA_ROOT}/infrared_uav/detector_ir16_v1/
${NATIVE16_DATA_ROOT}/infrared_uav/metadata_full_v1/flir_ir16_train.jsonl
${NATIVE16_DATA_ROOT}/infrared_uav/detector_ir16_v1/radiometric_profile.json
```

每条 JSONL 记录必须提供配方选择的 `thermal16` 字段、有效边界框，以及以下类别标签之一：

```text
person, bike, car, motor, bus, truck, other_vehicle
```

数据集、元数据及其使用权由使用者自行负责；本仓库不包含图像或标注文件。

## 可复现训练

先验证配置并执行单进程 dry-run：

```bash
source configs/local/server.env
export PYTHONPATH="$PWD/src"
"${NATIVE16_PY}" -m native16_gligen.train \
  --cfg configs/recipes/native16_roi_cf.yaml \
  --dry-run
```

启动配置好的分布式训练：

```bash
scripts/train.sh configs/recipes/native16_roi_cf.yaml
```

不修改配方即可运行有上限的 pilot：

```bash
scripts/train.sh configs/recipes/native16_roi_cf.yaml \
  --set train.max_steps=300 \
  --set train.scheduler_total_steps=300
```

每个训练 delta 只属于一个确定的有序父模型栈。恢复训练时使用配方中的 `train.resume`，或显式指定：

```bash
--set train.resume=/absolute/path/to/checkpoint.pt
```

不要通过历史 checkpoint 参数替换模型栈层。

## Native16 推理

请求文件是 JSON 列表。每条请求需要唯一的 `name`、文本 prompt、归一化 `xyxy` 边界框，以及数量匹配的 labels/phrases。参见 `examples/native16_sample_requests.json`。

```bash
# 只检查解析后的路径和配置，不加载模型。
scripts/generate.sh \
  --requests examples/native16_sample_requests.json \
  --output "$NATIVE16_ARTIFACT_ROOT/native16_samples/smoke" \
  --dry-run

# 从固定父模型栈生成 Native16 TIFF。
scripts/generate.sh \
  --requests examples/native16_sample_requests.json \
  --output "$NATIVE16_ARTIFACT_ROOT/native16_samples/baseline" \
  --seed 20260812 \
  --guidance-scale 10.0 \
  --no-previews

# 在同一份已审计父模型栈上评估新训练的 delta。
scripts/generate.sh \
  --requests examples/native16_sample_requests.json \
  --checkpoint-layer /absolute/path/to/step_00000300.pt \
  --output "$NATIVE16_ARTIFACT_ROOT/native16_samples/step300"
```

成功生成后会得到：

- `images16/*.tiff`
- 可选的 `previews/*.png`
- 可选的 `overlays/*.png`
- `metadata.json`

`metadata.json` 记录解析后的父模型栈指纹、新增 checkpoint layer、VAE/bridge/校准来源、随机种子、边界框、短语和输出范围。

验证生成目录是否为每条请求生成了一个 TIFF：

```bash
native16-verify-tiffs \
  --requests examples/native16_sample_requests.json \
  --image-dir "$NATIVE16_ARTIFACT_ROOT/native16_samples/baseline/images16"
```

## 验证

以下快速契约测试不需要模型权重或数据集：

```bash
pytest -q
python -m compileall -q src tests
bash -n scripts/*.sh
```

生产运行前，还应执行上面的训练和推理 dry-run。报告推理结果时，应将 `metadata.json` 与对应的生成 TIFF 一起保留。

## 可复现性规则

1. 每次实验都保存 `configs/recipes/native16_roi_cf.yaml`、解析后的环境、父模型栈清单以及生成的 `resolved_config.yaml`。
2. 将父模型栈的 SHA-256 和 fingerprint 视为不可变信息。任何模型层变化都需要新的模型栈清单和新的实验标识。
3. Native16 路径不得将源 TIFF 转换为 PNG/JPEG。模型输入是辐射测量单通道数据，输出为 `uint16` TIFF。
4. 推理请求使用归一化 `xyxy` 边界框。请求名经过文件名清理后仍必须唯一。
5. 基础模型、父模型资产、输入数据和可写输出必须放在不同目录。

## 上游组件与许可证

本项目使用公开的 GLIGEN 和 Diffusers 生态组件。除本仓库的 MIT License 外，还必须遵守所下载基础模型、数据集和依赖项各自的许可证及模型使用条款。
