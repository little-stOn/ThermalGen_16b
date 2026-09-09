# DataLoader Batch 可视化检测说明

## 1. 这部分代码是做什么的

这部分代码不是目标检测模型，也不是生成模型。它的作用是：

> 把训练时 DataLoader 真正返回的 batch 取出来，先检查数据结构是否正确，再把 batch 中的视频 clip 画出来，帮助确认帧顺序、视频连续性、bbox 和 bbox condition 是否对应。

核心脚本是：

```text
scripts/prepare/visualize_dataset_batch.py
```

它模拟训练代码中的这一句：

```python
batch = next(iter(dataloader))
```

然后对这个真实 batch 做检查和可视化。因此，它检查的不是底层文件“看起来是否存在”，而是：

```text
数据集
  ↓
Dataset
  ↓
DataLoader
  ↓
collate_fn
  ↓
训练时真正拿到的 batch
  ↓
可视化检测脚本
```

这可以发现很多只有在 DataLoader 组 batch 后才会出现的问题，例如：

- 帧顺序错乱；
- 一个 clip 中混入了不同视频；
- 不同视角被拼到一起；
- bbox 和 bbox condition 数量不一致；
- 时间戳倒退；
- 图像和条件图像的维度不匹配。

---

## 2. 先理解几个基本概念

### batch

一次从 DataLoader 中取出来的一批数据。

例如：

```text
batch size = 2
```

表示一次取出 2 个视频 sample。

### sample

batch 中的一个独立视频 clip。

例如一个 sample 有 4 帧：

```text
sample 0 = frame 0, frame 1, frame 2, frame 3
```

### frame

视频中的一帧图像。

### view

一个视角或一个相机的数据。当前多数据集配置主要使用单个 thermal view，因此通常：

```text
V = 1
```

### bbox condition

把 bbox 画在黑色 RGB 图像上形成的条件图像。它不是原始热红外图像，而是给后续条件模型使用的空间提示。

### contact sheet

把多张图像排列到一张大图中，方便一次查看多帧或多个 sample。

### GIF

把一个 sample 的连续帧按顺序播放，用来直观看视频是否连续。

---

## 3. 整个可视化检测流程

脚本的整体流程是：

```text
读取命令行参数
      ↓
使用同一套配置工厂创建 DataLoader
      ↓
从 DataLoader 取出一个 batch
      ↓
检查 batch 的结构和内容
      ↓
逐个 sample、逐帧生成可视化图
      ↓
生成 sample contact sheet 和 GIF
      ↓
把整个 batch 的 sample contact sheet 再拼起来
      ↓
保存 report.json
```

代码中的主要入口是：

```python
main()
```

每一个 batch 的主要入口是：

```python
inspect_batch(batch, batch_index, output_dir, ...)
```

`inspect_batch()` 做两件事：

```text
validate_batch(batch)
      ↓
render_sample(...)
```

也就是说：

> 先检查，检查通过后才渲染；检查失败就立即报错，不会生成一个看似正常的结果来掩盖问题。

---

## 4. 第一步：使用真实 DataLoader

脚本支持两种方式选择数据：

```bash
--task bbox
--task style
--task joint
```

或者直接指定配置文件：

```bash
--cfg configs/datasets/xxx.yaml
```

使用 `--task` 时，脚本调用：

```python
load_task_dataloader(task)
```

因此它和训练时使用的是同一套 DataLoader 配置，不会另外写一套读取逻辑。

例如：

```bash
PYTHONPATH=src \
python scripts/prepare/visualize_dataset_batch.py \
  --task joint \
  --batches 2 \
  --max-samples-per-batch 0 \
  --output-dir outputs/joint_visual
```

参数含义：

- `--task joint`：使用 `multi_joint.yaml`；
- `--batches 2`：检查 2 个 batch；
- `--max-samples-per-batch 0`：每个 batch 中的所有 sample 都渲染；
- `--output-dir`：保存结果的目录。

脚本本身不会重新访问底层数据集索引，而是：

```python
iterator = iter(loader)
batch = next(iterator)
```

因此可视化内容就是训练循环实际收到的内容。

---

## 5. 第二步：检查 batch 的基本结构

### 5.1 batch 必须是字典

脚本首先检查：

```python
isinstance(batch, dict)
```

因为当前 Pipeline 约定 DataLoader 返回字典。

### 5.2 必须存在图像

图像字段可以是：

```text
vae_images
```

或者：

```text
images
```

如果两个字段都没有，脚本直接报错。

### 5.3 检查图像 shape

图像必须是 6 维：

```text
[B, T, V, C, H, W]
```

例如：

```text
[2, 4, 1, 1, 256, 256]
```

表示：

- `B=2`：两个 sample；
- `T=4`：每个 sample 四帧；
- `V=1`：一个视角；
- `C=1`：单通道热红外图像；
- `H=W=256`：空间分辨率 256×256。

同时检查：

- `B`、`T`、`V` 不能为 0；
- 图像 tensor 不能含 NaN；
- 图像 tensor 不能含 Inf。

### 5.4 检查 bbox condition

如果 batch 中存在：

```text
box_condition_images
```

则它也必须有 6 维，并且前三个维度必须与图像一致：

```text
images:              [B,T,V,C,H,W]
box_condition_images:[B,T,V,C,H,W]
```

例如：

```text
images:               [2,4,1,1,256,256]
box_condition_images: [2,4,1,3,256,256]
```

这里图像是单通道，而 bbox condition 是 3 通道 RGB 条件图，这是允许的；但 `B,T,V` 必须对应。

### 5.5 检查 mask 和时间戳

以下字段如果存在，都必须对应同样的 `[B,T,V]` 结构：

```text
bbox_available
condition_valid
pts
```

并且不能包含 NaN 或 Inf。

---

## 6. 第三步：检查 sample 是否真的是连续视频

完成整体 shape 检查后，脚本会逐个 sample、逐个 view、逐帧检查。

假设：

```text
B=2, T=4, V=1
```

脚本会检查：

```text
sample 0, view 0, frame 0 → frame 1 → frame 2 → frame 3
sample 1, view 0, frame 0 → frame 1 → frame 2 → frame 3
```

### 6.1 一个 clip 不能改变数据集

每个 sample 的所有帧必须来自同一个 dataset。

错误示例：

```text
第 0 帧来自 FLIR
第 1 帧来自 ZUT
```

这种情况会直接报错。

### 6.2 一个 clip 不能改变 sequence

每个 sample 的所有帧必须来自同一个视频序列。

错误示例：

```text
scene_001/frame_001
scene_002/frame_002
```

这说明 batch 组装时把两个视频拼成了一个 clip。

### 6.3 一个 view 内不能改变相机

如果一个 sample 有多个 view，那么同一个 view 位置上的所有时间帧必须属于同一视角。

### 6.4 frame ID 必须递增

脚本会解析 `sample_ids` 中的 frame ID，并检查帧顺序：

```text
frame 001 < frame 002 < frame 003 < frame 004
```

如果出现：

```text
frame 003 → frame 001
```

就会报错。

### 6.5 `pts` 必须单调不下降

如果 batch 中有时间戳 `pts`，脚本会检查：

```text
pts[0] <= pts[1] <= pts[2] <= ...
```

时间戳可以相等，但不能倒退。

---

## 7. 第四步：检查 bbox 和 condition 是否一致

对于每一帧、每一个 view，脚本会统计 bbox 数量，并结合：

```text
bbox_available
condition_valid
boxes
```

进行一致性检查。

### 7.1 没有 bbox 时

如果：

```text
condition_valid = False
```

则当前帧不能被标记为有有效 bbox condition。

### 7.2 有效 bbox 时

如果：

```text
condition_valid = True
```

那么通常必须满足：

```text
bbox_available = True
bbox 数量 > 0
bbox_condition 中确实存在框的像素信号
```

如果 mask 说这一帧有框，但条件图实际上是一张全黑图，脚本会报错。

### 7.3 记录每个 sample 的统计

每个 sample 最终会记录：

```json
{
  "batch_index": 0,
  "dataset": "...",
  "sequence": "...",
  "frames": 4,
  "views": 1,
  "bbox_frames": 4,
  "bbox_count": 6
}
```

其中：

- `frames`：这个 sample 有多少帧；
- `views`：这个 sample 有多少视角；
- `bbox_frames`：有有效 bbox 的帧数；
- `bbox_count`：所有帧中 bbox 的总数。

---

## 8. 第五步：把单通道热红外图像变成可看的图片

原始热红外图像可能是：

- PIL 图像；
- PyTorch tensor；
- NumPy array；
- 单通道灰度图；
- `[C,H,W]` 或 `[H,W,C]` 格式。

`_image_from_value()` 会把它们转换成普通 RGB 图片用于显示。

处理过程是：

```text
读取 tensor/PIL/NumPy
      ↓
转换成 [H,W] 或 [H,W,C]
      ↓
把数值缩放到 [0,1]
      ↓
可选 1%/99% 对比度拉伸
      ↓
单通道应用颜色映射
      ↓
转换成 8-bit RGB
      ↓
可选放大显示
```

默认配置：

```text
color_map = inferno
display_scale = 2
auto_contrast = True
```

### 重要说明

这个转换只作用于可视化图片：

```text
不会修改 batch 中的 vae_images
不会修改原始数据集文件
不会改变训练输入数值
```

输出的 PNG 和 GIF 是为了方便人眼查看，通常是 8-bit RGB；这不代表原始热红外数据已经变成了 8-bit。

支持的颜色映射：

```text
gray
inferno
turbo
```

可以通过参数切换：

```bash
--color-map gray
```

---

## 9. 第六步：绘制每一帧

每一帧会生成一个画布。

### style 模式

style 模式没有 bbox condition，因此主要显示：

```text
热红外图像的可视化结果
```

### bbox/joint 模式

如果当前 sample 中存在有效 bbox，画布会包含三个 panel：

```text
[image (contrast RGB)] [bbox condition] [image + bbox]
```

分别表示：

1. 原始热红外图像经过显示变换后的结果；
2. bbox 条件图像；
3. 将 bbox condition 叠加到原图上的结果。

每个 view 的标题还会显示：

```text
view=0 bbox=2 valid=True dataset:sequence:frame:view
```

顶部标题显示：

```text
dataset | sequence | t=0 | pts=0.0000
```

这样可以同时确认：

- 当前画面来自哪个数据集；
- 当前属于哪个 sequence；
- 当前是第几帧；
- 当前时间戳是多少；
- 当前有多少个 bbox；
- bbox condition 是否有效。

---

## 10. 第七步：生成 sample contact sheet 和 GIF

对于一个 sample，脚本会遍历它的所有时间帧：

```python
for time_index in range(time_steps):
    render_one_frame(...)
```

然后把这些帧横向拼接：

```text
frame 0 | frame 1 | frame 2 | frame 3
```

生成文件：

```text
batch_0000_sample_00_contact.png
```

如果没有指定：

```bash
--no-gif
```

还会把这些帧保存成 GIF：

```text
batch_0000_sample_00.gif
```

默认每帧持续：

```text
180 ms
```

可以修改：

```bash
--gif-duration-ms 250
```

看 contact sheet 主要检查：

- 相邻帧是否来自同一个视频；
- 目标是否连续移动；
- bbox 是否随帧变化；
- 是否出现突然跳帧；
- 是否出现图像和 condition 错位。

看 GIF 主要检查：

- 视频是否播放连续；
- 画面是否发生异常跳变；
- bbox 条件是否跟随目标；
- 是否把多个 sequence 错误拼接到一起。

---

## 11. 第八步：生成整个 batch 的总览图

一个 batch 中可能有多个 sample。脚本会把每个 sample 的 contact sheet 纵向拼接成一张图：

```text
sample 0 的 contact sheet
sample 1 的 contact sheet
sample 2 的 contact sheet
```

文件名是：

```text
batch_0000_contact.png
```

这张图可以快速回答：

> 这一次从 DataLoader 取出的整批数据，整体是不是都正常？

---

## 12. 第九步：保存 `report.json`

所有 batch 检查完成后，脚本会保存：

```text
report.json
```

其中包含：

```json
{
  "batches_requested": 2,
  "batches_rendered": 2,
  "output_dir": "...",
  "batches": [
    {
      "batch_index": 0,
      "validation": {
        "image_key": "vae_images",
        "shape": [2, 4, 1, 1, 256, 256],
        "condition_present": true,
        "boxes_present": true,
        "samples": []
      },
      "batch_contact": ".../batch_0000_contact.png",
      "rendered": []
    }
  ]
}
```

`report.json` 主要用于记录：

- 请求检查了多少个 batch；
- 实际成功检查了多少个 batch；
- 每个 batch 的 shape；
- 是否存在 bbox condition；
- 是否存在 boxes；
- 每个 sample 的数据集和 sequence；
- bbox 帧数和 bbox 总数；
- 生成了哪些可视化文件。

如果 DataLoader 没有产生任何 batch，脚本会直接报错：

```text
DataLoader produced no batches
```

---

## 13. 如何放进训练循环

`inspect_batch()` 可以直接放到训练循环中，在真正训练前检查少量 batch：

```python
from pathlib import Path

from scripts.prepare.visualize_dataset_batch import inspect_batch

for step, batch in enumerate(training_dataloader):
    if step < 2:
        inspect_batch(
            batch,
            batch_index=step,
            output_dir=Path("outputs/train_visual"),
            max_samples_per_batch=0,
        )

    # 后面才进入真正的训练逻辑
    training_step(batch)
```

实际含义是：

```text
取 batch
  ↓
检查并可视化
  ↓
检查通过
  ↓
送入模型训练
```

如果检查失败，训练会在进入模型前停止，避免错误数据继续传播。

---

## 14. 这部分已经验证了什么

可视化检测代码能够验证：

- DataLoader 能否产出 batch；
- batch 是否是字典；
- 图像字段是否存在；
- 图像 shape 是否为 `[B,T,V,C,H,W]`；
- bbox condition shape 是否匹配；
- sample 是否来自同一个 dataset；
- sample 是否来自同一个 sequence；
- view 是否稳定；
- frame ID 是否递增；
- `pts` 是否单调；
- bbox mask 和 bbox 数量是否一致；
- 有效 condition 是否真的画出了 bbox；
- 每个 sample 是否能还原成连续视频 clip；
- 整个 batch 是否可以生成可读的总览图。

## 15. 这部分没有验证什么

它不会验证：

- 模型能否正确生成视频；
- VAE 编码和解码是否正确；
- denoiser 是否有效；
- loss 是否收敛；
- 文本语义是否正确；
- bbox 类别标签的语义是否正确；
- 生成结果质量；
- 检测模型的 mAP。

它解决的是更基础的问题：

> 在模型训练之前，确认送进模型的数据确实是正确、连续、结构一致的视频 batch。

---

## 16. 测试和真实数据检查的区别

项目中的：

```text
tests/test_visualize_dataset_batch.py
```

主要使用小型模拟 batch 测试：

- shape 校验；
- 时间戳校验；
- bbox 校验；
- contact sheet 写出；
- GIF 写出。

而命令行脚本：

```text
scripts/prepare/visualize_dataset_batch.py
```

可以使用真实数据集配置和真实 DataLoader 检查实际 batch。

两者的关系是：

```text
单元测试：验证可视化代码本身
真实数据运行：验证数据集到 batch 的实际结果
```

---

## 17. 可以怎样向别人讲解

> 这部分可视化代码不是在做目标检测，而是在检查训练数据。它使用和训练完全相同的 DataLoader，取出真实 batch 后，先检查 batch 的维度、sample ID、帧顺序、时间戳以及 bbox mask 是否一致。检查通过后，再把每个 sample 的连续帧画成 contact sheet 和 GIF，并把整个 batch 汇总成总览图。这样可以在数据进入模型之前，确认每个 sample 确实是来自同一个视频序列的连续 clip，同时确认 bbox condition 和原始图像是对应的。所有颜色映射和对比度拉伸只用于显示，不会修改真正送进模型的 tensor。

最简流程：

```text
真实 DataLoader
    ↓
取出 batch
    ↓
检查结构和时序
    ↓
检查 bbox 和 condition
    ↓
渲染每个 sample
    ↓
保存 PNG、GIF、report.json
```

---

## 18. 相关代码位置

```text
scripts/prepare/visualize_dataset_batch.py
  validate_batch()       # 检查 batch 结构和时序一致性
  render_sample()        # 渲染一个 sample 的所有帧
  render_batch_contact() # 汇总整个 batch
  inspect_batch()        # 检查并渲染一个 batch
  main()                 # 命令行入口

tests/test_visualize_dataset_batch.py
  可视化检测逻辑的单元测试

scripts/prepare/load_dataset_config.py
  独立检查 DataLoader 构造和 batch 吞吐
```
