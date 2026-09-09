# 当前热红外视频 Pipeline 说明

## 1. 一句话理解

> Pipeline 是训练和生成过程的“总控流程”，不是具体的热红外视频模型。

它负责把 DataLoader 提供的视频 clip 整理好，然后安排：

- 训练时如何加噪、调用模型、计算损失和更新参数；
- 推理时如何从随机 latent 开始逐步去噪并生成视频；
- 如何做 CFG、梯度累积、分布式 loss 统计和 checkpoint；
- 如何把生成结果交给预览、评估和文件写出模块。

具体的 VAE、时序网络、文本/bbox 编码器和 scheduler 由外部注入的 `ThermalVideoModel` 提供。

当前代码位于：

```text
src/dwm/pipelines/
├── contracts.py       # 输入、模型、结果、writer 和 evaluator 的接口
├── objectives.py      # 默认扩散损失
└── thermal_video.py   # Pipeline 的训练和推理流程
```

**当前状态：** Pipeline 的通用编排逻辑已经实现，但仓库中还没有接入生产级 `ThermalVideoModel`。现有的 `native16_gligen.train` 是 Native16 GLIGEN 图像训练路径，不是这套视频 Pipeline 的实际生产训练入口。

---

## 2. Pipeline 的输入和输出

### 2.1 输入 batch

DataLoader 给 Pipeline 的 batch 是一个字典，主要包含：

```text
vae_images              视频图像
box_condition_images    bbox 条件图像，可选
boxes                   原始 bbox，可选
labels                  bbox 类别，可选
track_ids               目标跟踪 ID，可选
bbox_available          是否存在可用 bbox 标注
condition_valid         bbox condition 当前是否有效
frame_valid             当前单元是否参与 loss，可选
pts                     时间戳
fps                     帧率
box_image_sizes         bbox 图像尺寸
```

图像的统一形状是：

```text
[B, T, V, C, H, W]
```

例如：

```text
[2, 4, 1, 1, 256, 256]
```

表示：2 个视频 clip，每个 clip 4 帧、1 个视角、单通道、分辨率 256×256。

这里的 `T` 是视频帧数，不是秒数。

### 2.2 输出结果

推理完成后，Pipeline 返回 `GenerationResult`：

```text
result.latents     最终视频 latent
result.frames      解码后的视频 tensor，可选
result.batch       标准化后的输入 batch
result.num_steps   去噪步数
result.metadata    生成过程的元数据
```

`frames` 是否能被解码出来，由具体模型的 `decode_video()` 决定。

---

## 3. 第一步：把 DataLoader 的数据整理成统一格式

入口是：

```python
ThermalVideoBatch.from_loader(batch)
```

不同数据集的字段名称、时间单位和标注情况可能不同。这个适配层负责把它们整理成 Pipeline 能理解的统一对象。

它主要做以下工作：

1. 兼容不同字段名称，例如 `images` 和 `vae_images`；
2. 检查图像与 bbox condition 是否具有一致的 `[B,T,V]` 结构；
3. 检查图像是否在 `[0,1]`，并拒绝 NaN/Inf；
4. 保留变长的 bbox、类别和 track ID，不强行填充成固定目标数量；
5. 把毫秒时间戳转换成秒；
6. 把 tensor 和嵌套 tensor 移动到模型所在设备；
7. 保留 bbox 的 pixel 坐标，具体归一化由模型负责；
8. 记录无框帧和无效条件帧的状态。

可以把这一步理解为：

> 先把来自不同数据集的数据整理成同一种语言，再交给模型。

如果输入的 shape 或数值不正确，Pipeline 会直接报错，而不是继续产生难以定位的结果。

---

## 4. 训练流程

训练入口是：

```python
pipeline.train_step(batch)
```

完整过程如下：

```text
DataLoader batch
      ↓
ThermalVideoBatch.from_loader()
      ↓
视频图像编码成 latent
      ↓
生成噪声并采样时间步
      ↓
给 latent 加噪
      ↓
准备文本、bbox 等条件
      ↓
模型联合处理整个 [B,T,V] 视频
      ↓
计算扩散损失
      ↓
分布式统计和反向传播
      ↓
梯度累积
      ↓
更新模型参数
```

### 4.1 编码视频

Pipeline 调用：

```python
latents = model.encode_video(prepared_batch)
```

模型负责把视频从像素空间变成 latent 空间。Pipeline 不规定模型使用 2D VAE、Temporal VAE 还是其他编码器，只要求 latent 保留：

```text
[B, T, V, ...]
```

### 4.2 加噪

Pipeline 生成与 latent 同形状的随机噪声，然后调用：

```python
timesteps = model.sample_timesteps(...)
noisy_latents = model.add_noise(latents, noise, timesteps)
target = model.training_target(latents, noise, timesteps)
```

不同扩散模型可能预测不同目标，例如：

- 噪声 `epsilon`；
- velocity；
- 干净 latent；
- 其他扩散参数化目标。

这些差异由模型决定，Pipeline 不写死。

### 4.3 准备条件

Pipeline 调用：

```python
conditions = model.prepare_conditions(
    prepared_batch,
    training=True,
    generator=generator,
)
```

模型可以在这里处理：

- 文本 prompt；
- bbox condition image；
- bbox 和类别标签；
- 时间戳、帧率；
- 相机或其他辅助条件。

Pipeline 只要求返回 `ConditionBundle`，其中至少有：

```python
conditions.conditional
```

Pipeline 本身不直接实现文本编码和 bbox 编码。

### 4.4 联合处理视频

Pipeline 调用：

```python
prediction = model.forward_video(
    noisy_latents,
    timesteps,
    conditions.conditional,
    prepared_batch,
)
```

模型应该一次看到完整的视频 clip，而不是把每一帧完全独立处理。这样模型才有机会学习：

- 目标在连续帧中的运动；
- 视频帧之间的时序一致性；
- 多视角之间的关系；
- bbox 条件随时间的变化。

Pipeline 会检查 prediction 和 target 的 shape 是否一致。

### 4.5 计算损失

默认使用：

```python
MaskedDiffusionMSE
```

它先计算每个 `[B,T,V]` 单元的误差，再根据有效 mask 求平均：

```text
loss = 有效单元的误差总和 / 有效单元数量
```

这里必须区分两个字段：

- `condition_valid`：bbox 条件是否存在、是否可用；
- `frame_valid`：该视频单元是否参与扩散 loss。

默认损失直接使用 `frame_valid`。它不会自动把 `condition_valid=False` 转成 loss mask，因为 joint 训练中无 bbox 图像仍然可能用于风格和图像学习。如果需要跳过某些单元，需要由数据层提供 `frame_valid`，或者使用自定义 `VideoObjective`。

### 4.6 反向传播和参数更新

计算 loss 后，Pipeline 负责：

1. 汇总不同 GPU/rank 上的有效单元数量；
2. 对 loss 做全局归一化；
3. 执行反向传播；
4. 根据 `accumulation_steps` 决定是否更新参数；
5. 使用 AMP scaler（如果启用）；
6. 裁剪梯度（如果配置）；
7. 更新 optimizer；
8. 更新 learning-rate scheduler；
9. 清空梯度。

### 4.7 梯度累积

例如：

```python
accumulation_steps = 4
```

表示连续 4 个 micro batch 才更新一次模型参数。前 3 个 batch 只累积梯度。

如果训练结束时还有未完成的累积组，可以调用：

```python
pipeline.flush()
```

Pipeline 会按照已经累积的真实有效单元完成最后一次更新，不直接丢弃这些梯度。

---

## 5. 推理流程

推理入口是：

```python
result = pipeline.inference_pipeline(
    batch,
    num_steps=30,
    guidance_scale=7.5,
)
```

完整过程如下：

```text
输入文本、bbox 或其他条件
      ↓
确定视频 latent 的形状
      ↓
生成随机 latent
      ↓
准备条件
      ↓
循环执行多步 denoising
      ↓
得到干净的视频 latent
      ↓
解码为视频 tensor
```

### 5.1 确定 latent 形状

Pipeline 调用：

```python
shape = model.latent_shape(prepared_batch)
```

模型返回：

```text
[B, T, V, C_latent, H_latent, W_latent]
```

Pipeline 会检查前三个维度是否仍然是当前视频的 `[B,T,V]`。

### 5.2 初始化噪声

如果没有提供 `initial_latents`，Pipeline 使用随机种子生成初始 latent。

如果提供了 `initial_latents`，Pipeline 会先检查 shape，再从指定 latent 开始生成。这便于：

- 复现实验；
- 用同一随机起点比较不同条件；
- 后续扩展参考帧或长视频生成。

### 5.3 准备条件

推理时再次调用：

```python
model.prepare_conditions(
    prepared_batch,
    training=False,
    generator=generator,
)
```

如果 `guidance_scale != 1`，模型还必须提供：

```python
conditions.unconditional
```

否则 Pipeline 会报错。

### 5.4 逐步去噪

Pipeline 从模型获得指定数量的推理时间步，然后重复执行：

```python
model_output = model.forward_video(...)
latents = model.scheduler_step(...)
```

每一步都会去掉一部分噪声，最终得到视频 latent。

### 5.5 CFG

当使用 classifier-free guidance 时，Pipeline 同时得到：

- 无条件预测；
- 有条件预测。

然后按照下面的方式组合：

```text
最终预测 = 无条件预测
         + guidance_scale ×（有条件预测 - 无条件预测）
```

直观上，就是把文本和 bbox 对生成结果的影响增强一些。

### 5.6 解码

推理结束后：

- `output_type="latent"`：只返回 latent；
- `output_type="tensor"`：调用 `model.decode_video()` 返回视频 tensor。

Pipeline 不规定最终文件必须是 MP4、PNG 还是 16-bit TIFF。

---

## 6. Preview 和 Evaluation

### 6.1 Preview

```python
pipeline.preview_pipeline(...)
```

Preview 实际上做两件事：

```text
执行推理
  ↓
把结果交给 VideoWriter
```

`VideoWriter` 可以根据热红外任务需要写出：

- MP4 预览视频；
- 16-bit TIFF 序列；
- bbox 叠加图；
- 带辐射测量信息的 metadata。

Pipeline 不会擅自把热红外结果压缩成普通 8-bit RGB。

### 6.2 Evaluation

```python
pipeline.evaluate_pipeline(
    dataloader,
    evaluator,
    num_steps=30,
)
```

流程是：

```text
遍历验证集
  ↓
为每个 batch 生成视频
  ↓
调用 evaluator(result, batch)
  ↓
收集数值指标
  ↓
计算平均结果
```

`VideoEvaluator` 可以实现：

- thermal FID/KID；
- bbox adherence；
- text-box binding；
- radiometric consistency；
- temporal flicker；
- 下游检测器 mAP。

当前 Pipeline 只负责调用和汇总 evaluator，不内置热红外领域指标。跨 GPU 的专用指标合并需要由 evaluator 或外层评估程序负责。

---

## 7. Checkpoint 和恢复训练

接口是：

```python
pipeline.save_checkpoint(path)
pipeline.load_checkpoint(path)
```

保存内容包括：

- 模型参数；
- optimizer 状态；
- learning-rate scheduler 状态；
- AMP scaler 状态；
- epoch、micro step、optimizer step；
- Pipeline 随机数生成器状态；
- PyTorch CPU/CUDA 随机状态；
- 自定义 metadata。

保存前必须处于 optimizer 更新边界。如果还有未完成的梯度累积，应先调用：

```python
pipeline.flush()
```

这样恢复训练时，不仅模型参数能够恢复，训练计数器和随机数状态也能恢复。

---

## 8. Pipeline 和模型的分工

可以把 Pipeline 和模型理解为“总控系统”和“发动机”：

```text
ThermalVideoPipeline
├── 整理和检查 batch
├── 安排训练步骤
├── 安排推理步骤
├── 管理 loss 和梯度
├── 管理 CFG
├── 管理 preview/evaluation
└── 管理 checkpoint

ThermalVideoModel
├── 编码视频
├── 解码视频
├── 定义 latent shape
├── 定义加噪和训练目标
├── 编码文本/bbox 等条件
├── 对完整视频做时序前向
└── 执行 scheduler 更新
```

这样未来更换 VAE、UNet、DiT 或热红外条件编码器时，不需要重写整个训练和推理流程。

---

## 9. 当前已实现和未实现的内容

| 功能 | 状态 |
|---|---|
| 统一 `[B,T,V,...]` 视频 batch | 已实现 |
| 输入 shape、范围和有限性检查 | 已实现 |
| 秒/毫秒时间戳统一 | 已实现 |
| 通用视频扩散训练流程 | 已实现 |
| 有效单元 loss mask | 已实现，使用 `frame_valid` |
| 梯度累积 | 已实现 |
| AMP、梯度裁剪、optimizer/scheduler 更新 | 已实现 |
| 分布式 loss 归约 | 已实现 |
| 联合视频 latent 推理 | 已实现 |
| classifier-free guidance | 已实现 |
| latent 或 tensor 输出 | 已实现 |
| Preview 回调 | 已实现 |
| Evaluation 回调 | 已实现 |
| 模型、优化器和随机状态恢复 | 已实现 |
| 具体热红外视频模型 | 尚未接入 |
| 文本和 bbox 的具体编码 | 由具体模型实现 |
| 16-bit TIFF/视频写出 | 需要注入 `VideoWriter` |
| 热红外专用评估指标 | 需要注入 `VideoEvaluator` |
| 长视频自回归生成 | 当前未实现 |
| Streaming/FIFO 在线生成 | 当前未实现 |
| OpenDWM 的 3D box、HD map、depth、action 条件 | 当前未实现 |
| 完整训练 epoch loop | Pipeline 只提供 `train_step`，由外层程序驱动 |

`VariableVideoBatchSampler` 位于 `src/dwm/utils/sampler.py`，属于数据加载层，不是 Pipeline 核心。它负责按照配置选择分辨率、帧数和每卡 batch size。

---

## 10. 对外讲解版本

### 30 秒版本

> 这套 Pipeline 是热红外视频模型的通用总控层。它先把不同数据集返回的内容统一成 `[B,T,V]` 视频 batch，然后在训练时完成视频编码、加噪、条件准备、联合时空前向、扩散 loss、梯度累积和参数更新；在推理时从随机 latent 开始，经过多步 denoising 和 CFG，最后解码成视频。Pipeline 还负责 preview、evaluation、分布式 loss 统计和 checkpoint，而具体的 VAE、时序网络、文本/bbox 编码和 16-bit 输出由注入的模型与 writer 完成。

### 进一步解释

> 它不是某一个具体的 UNet，也不是单独的数据加载器，而是把数据、模型、损失、优化器和输出模块串起来的流程控制器。这样做的好处是，后面更换不同的视频模型或热红外编码方式时，只需要实现 `ThermalVideoModel`，不用重新实现训练、推理、梯度累积和 checkpoint 逻辑。

---

## 11. 相关代码

```text
src/dwm/pipelines/contracts.py
  ThermalVideoBatch
  ThermalVideoModel
  ConditionBundle
  GenerationResult
  VideoWriter
  VideoEvaluator

src/dwm/pipelines/objectives.py
  MaskedDiffusionMSE
  VideoObjective

src/dwm/pipelines/thermal_video.py
  ThermalVideoPipeline.train_step()
  ThermalVideoPipeline.inference_pipeline()
  ThermalVideoPipeline.preview_pipeline()
  ThermalVideoPipeline.evaluate_pipeline()
  ThermalVideoPipeline.save_checkpoint()
  ThermalVideoPipeline.load_checkpoint()

src/dwm/common.py
  load_pipeline_from_config()

src/dwm/utils/sampler.py
  VariableVideoBatchSampler

tests/test_dwm_pipelines.py
  Pipeline contract and behavior tests
```
