# 数据加载完整说明

本文解释本项目从 Python 命令开始，到数据集配置、索引、单帧读取、bbox condition、Tensor 变换，最后进入 DataLoader batch 的完整过程。

目标：读完后能够回答三个问题：

1. 程序从哪个文件开始加载数据？
2. 配置文件如何决定使用哪个数据集、哪些子集和哪些处理步骤？
3. 一张图像和它的 bbox 是怎样进入最终 batch 的？

本文对应的代码范围：

```text
src/dwm/common.py
src/dwm/datasets/common.py
src/dwm/utils/sampler.py
src/dwm/datasets/butiv.py
src/dwm/datasets/flir.py
src/dwm/datasets/ltir_v1.py
src/dwm/datasets/zut_fir_adas.py
src/dwm/datasets/ms2.py
src/dwm/datasets/vivid_pp.py
src/dwm/datasets/lynred_mobility.py
src/dwm/datasets/tartanrgbt.py
configs/datasets/*.yaml
scripts/prepare/load_dataset_config.py
```

---

## 1. 先记住一句话

整个系统可以概括为：

> 配置文件选择 loader，loader 只建立“路径和标注索引”，真正取样时才读取图像；shared common 负责把不同格式变成统一的时间 clip，DatasetAdapter 最后把 PIL 图像变成模型需要的 Tensor。

也就是说，代码不是启动时把全部图像读进内存，而是分成两个阶段：

```text
初始化阶段：读取配置、扫描标注、建立路径索引、生成 clip 索引
取样阶段：读取当前 T×V 图像、处理 bbox、resize、ToTensor、组成 batch
```

---

## 2. 从哪里开始：调试入口文件

最适合人手演示的入口是：

```text
scripts/prepare/load_dataset_config.py
```

它不是某个数据集的 loader，而是一个通用调试器。核心流程在 `main()`：

```python
construction_start = time.perf_counter()

if args.loader:
    loader, _ = load_dataloader_from_config(args.cfg, overrides=args.set)
    dataset = loader.dataset
else:
    dataset, _ = load_object_from_config(args.cfg, args.key, args.set)

construction_seconds = time.perf_counter() - construction_start
sample = dataset[index]
```

这里有两个模式：

### 2.1 只加载 dataset

```bash
python scripts/prepare/load_dataset_config.py \
  --cfg configs/datasets/flir.yaml
```

这个模式用于检查：

- dataset 是否能构造
- dataset 长度是多少
- `dataset[0]` 是否能读取
- 每个字段是什么类型和 shape
- 首个 sample 中有多少 bbox

### 2.2 加载完整 DataLoader

```bash
python scripts/prepare/load_dataset_config.py \
  --cfg configs/datasets/flir.yaml \
  --loader \
  --batches 8
```

这个模式除了加载 dataset，还会：

1. 根据 config 构造 DataLoader
2. 读取指定数量的 batch
3. 统计 `samples_per_second`

因此它可以证明完整链路确实执行了，而不是只成功解析了 YAML。

---

## 3. 完整调用链

完整调用关系如下：

```mermaid
flowchart TD
    A[load_dataset_config.py] --> B[load_dataloader_from_config]
    B --> C[load_config]
    C --> D[展开环境变量和 overrides]
    D --> E[initialize_global_state]
    E --> F[create_instance_from_config dataset]
    F --> G[DatasetAdapter]
    G --> H[具体 MotionDataset]
    H --> I[数据集 *_common.py]
    I --> J[FrameRecord 索引]
    J --> K[BBoxMotionDataset 生成 clips]
    B --> L[实例化 torch DataLoader]
    L --> M[dataset[index]]
    M --> N[DatasetAdapter 取样和 transform]
    N --> O[DataLoader collation]
    O --> P[最终 batch]
```

下面逐步解释每一层。

---

## 4. 第一步：读取 YAML/JSON 配置

代码位置：

```text
src/dwm/common.py
```

入口函数：

```python
load_config(path, overrides=None)
```

它做四件事：

### 4.1 读取配置文件

```python
raw = yaml.safe_load(handle)
```

配置文件必须是一个 mapping，例如：

```yaml
schema_version: 1

global_state:
  flir_root: ${FLIR_ROOT}

dataset:
  _class_name: dwm.datasets.common.DatasetAdapter
```

### 4.2 展开环境变量

```yaml
global_state:
  flir_root: ${FLIR_ROOT}
```

如果运行前设置：

```bash
export FLIR_ROOT=/path/to/data/raw/flir
```

最终配置中的 `${FLIR_ROOT}` 会变成真实路径。

环境变量没有设置时，配置加载会立即失败，而不是等到真正读图时才报一个难以理解的路径错误。

### 4.3 处理 dotted overrides

例如：

```bash
--set \
  dataset.base_dataset.sequence_length=8 \
  dataset.base_dataset.bbox_policy=all \
  dataloader.num_workers=2
```

配置工厂会把它们解释为：

```yaml
dataset:
  base_dataset:
    sequence_length: 8
    bbox_policy: all

dataloader:
  num_workers: 2
```

override 不会修改 YAML 原文件，只修改本次运行内存中的副本。

### 4.4 返回普通字典

此时还没有实例化 dataset，也没有读取图像。此时只是得到配置字典。

---

## 5. 第二步：初始化 global_state

配置中的：

```yaml
global_state:
  flir_root: ${FLIR_ROOT}
```

会由：

```python
initialize_global_state(config)
```

注册到：

```python
dwm.common.global_state
```

之后 dataset 配置不需要再次硬编码路径，而是这样引用：

```yaml
dataset_root:
  _class_name: dwm.common.get_state
  key: flir_root
```

这一步的意义是：

- 路径只在配置顶部声明一次
- dataset 构造时可以复用同一个路径对象
- 多数据集配置中可以分别管理多个数据根目录
- 不把项目绝对路径写死在 Python loader 中

注意：`global_state` 只负责传递路径或共享对象，不负责扫描数据。

---

## 6. 第三步：递归实例化配置对象

配置中的：

```yaml
dataset:
  _class_name: dwm.datasets.common.DatasetAdapter
  base_dataset:
    _class_name: dwm.datasets.flir.MotionDataset
```

会被：

```python
create_instance_from_config(config["dataset"])
```

递归处理。

实例化顺序可以理解为：

```text
1. 先解析 DatasetAdapter
2. 发现它需要 base_dataset
3. 继续解析 flir.MotionDataset
4. 发现 dataset_root 是 get_state
5. 从 global_state 取出 FLIR_ROOT
6. 构造 FLIR MotionDataset
7. 构造 transform_list 中的 Resize、ToTensor、Compose
8. 最后构造 DatasetAdapter
```

因此配置不是一张静态参数表，而是一棵对象构造树。

例如：

```yaml
transform:
  _class_name: torchvision.transforms.Compose
  transforms:
    - _class_name: torchvision.transforms.Resize
      size: [256, 256]
    - _class_name: torchvision.transforms.ToTensor
```

等价于 Python：

```python
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.ToTensor(),
])
```

---

## 7. 第四步：具体数据集的 common parser 建立 FrameRecord

每个数据集分成两个文件：

```text
<dataset>.py
<dataset>_common.py
```

职责不同：

```text
<dataset>_common.py：理解原始目录和标注格式
<dataset>.py：接收必要参数，调用 parser，生成统一 dataset
```

parser 输出统一的：

```python
BBoxFrameRecord(
    sequence=...,       # 时间序列 ID
    frame_id=...,       # 帧 ID
    views=(...),        # 一个或多个视角
    timestamp=...,      # 可选时间戳
    metadata=...,       # 数据集特有信息
)
```

每个视角内部是：

```python
BBoxViewRecord(
    name="thermal",
    path=Path(...),
    boxes=(BBoxAnnotation(...), ...),
)
```

此时只保存：

```text
图像路径
帧 ID
序列 ID
bbox 坐标
类别
track_id
时间戳
```

还没有把图像像素读进内存。

---

## 8. 第五步：BBoxMotionDataset 生成时间 clip

代码位置：

```text
src/dwm/datasets/common.py
```

所有带 bbox 数据集都使用：

```python
BBoxMotionDataset
```

它负责把单帧记录变成时间 clip。

例如：

```yaml
sequence_length: 4
fps_stride_tuples: [[0.0, 1.0]]
```

含义是：

```text
每个 sample 包含连续 4 帧
clip 起点每隔 1 个源帧移动一次
```

它不会跨越不同的 `sequence`。

### 8.1 bbox_policy

```yaml
bbox_policy: any
min_box_count: 1
```

表示：

```text
一个 clip 内只要至少有一个 bbox，就保留该 clip
```

这适合 ZUT 等存在空帧的数据集，因为空帧仍然可以提供时间上下文。

如果配置为：

```yaml
bbox_policy: all
```

则表示：

```text
clip 的每一帧都必须至少有一个 bbox
```

### 8.2 为什么初始化阶段仍然可能比较慢

因为初始化阶段需要：

- 扫描目录
- 读取 COCO、XML、YOLO 或 JSONL 标注
- 检查路径
- 建立所有 FrameRecord
- 生成 clip 索引

但是它仍然不读取所有图像像素。

ZUT 和 MS2 的标注文件数量较多，因此额外提供 index cache：

```text
ZUT_ROOT/.dwm_cache/
MS2_ANNOTATION_ROOT/.dwm_cache/
```

cache 只保存路径和标注索引，不保存图像内容。

---

## 9. 第六步：真正访问 dataset[0]

当执行：

```python
sample = dataset[0]
```

由于外层是 `DatasetAdapter`，实际流程是：

```text
DatasetAdapter.__getitem__(0)
    ↓
base_dataset.__getitem__(0)
    ↓
读取当前 clip 的 T×V 图像
    ↓
处理 16-bit 强度
    ↓
生成 bbox condition
    ↓
返回 PIL 图像和原生 bbox
    ↓
DatasetAdapter 执行配置中的 transform
    ↓
递归 stack 时间维度和视角维度
    ↓
删除原始中间字段
    ↓
返回最终 sample
```

### 9.1 图像读取

`BBoxMotionDataset._read_image()` 只读取当前 sample 的图像：

```text
不是启动时读取所有图像
而是访问哪个 sample，就读取哪个 sample 的 T×V 张图
```

16-bit 图像默认会转成固定范围的 PIL `F` 图像：

```text
原始 uint16
    ↓
按 uint16_value_range 归一化
    ↓
float32 [0,1]
    ↓
PIL mode F
```

### 9.2 bbox condition

bbox 坐标已经在 parser 阶段统一为绝对像素 `xyxy`：

```text
[x1, y1, x2, y2]
```

shared base 根据这些坐标生成：

```text
bbox_condition_images
```

它是一张黑色背景、bbox 框线绘制在上面的 RGB PIL 图像。

注意：

```text
bbox_condition_images 是 2D bbox condition
不是 3D box
```

名字中特意不再使用 `3dbox_images`，避免误导。

### 9.3 Adapter transform

配置中：

```yaml
transform_list:
  - old_key: images
    new_key: vae_images
    transform:
      _class_name: torchvision.transforms.Compose
      transforms:
        - _class_name: torchvision.transforms.Resize
          size: [256, 256]
        - _class_name: torchvision.transforms.ToTensor
```

会把：

```text
[T][V] PIL images
```

递归变成：

```text
[T][V][C][H][W] Tensor
```

随后 `DatasetAdapter` 自动 stack，得到：

```text
vae_images: [T, V, C, H, W]
```

bbox condition 同样处理：

```text
bbox_condition_images -> box_condition_images
```

### 9.4 pop_list

配置中的：

```yaml
pop_list: [images, bbox_condition_images]
```

表示 transform 完成后删除原始 PIL 字段，只保留：

```text
vae_images
box_condition_images
```

这样可以避免 sample 同时保存 PIL 和 Tensor 两份图像。

---

## 10. 第七步：DataLoader 组成 batch

配置中的：

```yaml
dataloader:
  _class_name: torch.utils.data.DataLoader
  batch_size: 2
  shuffle: true
  num_workers: 0
  pin_memory: true
```

由：

```python
load_dataloader_from_config()
```

完成构造。

这个 helper 会先构造 dataset，然后把同一个 dataset 实例注入 DataLoader：

```text
dataset = create_instance_from_config(dataset_config)
loader = DataLoader(dataset=dataset, ...)
```

不会把 dataset 再构造一次。

### 10.1 为什么需要 CollateFnIgnoring

`boxes`、`labels`、`track_ids` 每帧数量不同，不能直接由默认 collate 强行堆叠。

所以配置使用：

```yaml
collate_fn:
  _class_name: dwm.datasets.common.CollateFnIgnoring
  keys: [boxes, labels, track_ids, sample_ids, sequence]
```

它的行为是：

```text
boxes/labels/track_ids/sample_ids/sequence：保留为 Python list
vae_images/box_condition_images/pts/fps/box_image_sizes：按形状组成 Tensor batch
pts_unit：保留为每个样本一个字符串的 Python list
```

最终典型形状：

```text
vae_images:
[B, T, V, C, H, W]

box_condition_images:
[B, T, V, 3, H, W]

box_image_sizes:
[B, T, V, 2]，最后一维为原始图像 [height, width]

pts:
[B, T, V]；BU-TIV 的毫秒值由 `ThermalVideoBatch` 适配为秒

fps:
[B]

例如当前配置：

```text
vae_images:           [2, 4, 1, 1, 256, 256]
box_condition_images: [2, 4, 1, 3, 256, 256]
```

维度解释：

```text
2：batch size
4：时间帧数 T
1：视角数 V
1/3：图像或 condition 通道数
256×256：空间分辨率
```

---

## 11. 每个数据集到底做了什么

## 11.1 BU-TIV

文件：

```text
src/dwm/datasets/butiv.py
src/dwm/datasets/butiv_common.py
configs/datasets/butiv.yaml
```

流程：

```text
XML + 图像目录
    ↓
butiv_common.py 解析 frame/object
    ↓
FrameRecord
    ↓
按 scene/view 对齐
    ↓
按 frame gap 切 segment
    ↓
MotionDataset 生成 clip
    ↓
读取 16-bit thermal 图像
    ↓
裁剪/校验 bbox
    ↓
生成 bbox_condition_images
```

需要注意：

- Marathon 默认一个 `CAM_FRONT` 视角
- Atrium 有 orange/red 两个同步视角
- `view_mode=single` 会把不同视角作为独立样本源
- `view_mode=multiview` 才会组成 `[T][V]`
- 时间戳是合成 ordinal timebase，不是真实采集时间
- XML 中只有矩形 bbox 才会进入 bbox 数据
- 点标注不会被强行转换成 bbox
- loader 不再接受 `fs`、`3dbox`、HD map、caption、sample_data 等参数

## 11.2 FLIR

文件：

```text
src/dwm/datasets/flir.py
src/dwm/datasets/flir_common.py
configs/datasets/flir.yaml
```

流程：

```text
COCO json
    ↓
image_id -> annotations 索引
    ↓
按 video_id 和 frame number 分组
    ↓
[x, y, width, height] -> [x1, y1, x2, y2]
    ↓
BBoxFrameRecord
```

默认配置读取：

```text
FLIR thermal train
single view
```

multiview 只对官方 video-test 有意义，因为 train/val 的 RGB 与 thermal 使用不同 video ID。

video-test pairing 使用：

```text
rgb_to_thermal_vid_map.json
```

不能简单按照排序后的文件位置进行配对。

## 11.3 LTIR v1

文件：

```text
src/dwm/datasets/ltir_v1.py
src/dwm/datasets/ltir_v1_common.py
configs/datasets/ltir_v1.yaml
```

目录结构类似：

```text
ltir_v1_0_8bit_16bit/
  16_car/
    00000001.png
    00000002.png
    groundtruth.txt
```

流程：

```text
每个序列目录
    ↓
数字帧名排序
    ↓
groundtruth.txt 按行对齐
    ↓
4 点或 8 点标注转 xyxy
    ↓
BBoxFrameRecord
```

需要注意：

- `mode=16bit` 是当前默认训练配置
- 每个序列只有一个 thermal 视角
- 图像数量和 groundtruth 行数不一致会立即报错
- 不能把 LTIR 当成多视角数据集

## 11.4 ZUT-FIR-ADAS

文件：

```text
src/dwm/datasets/zut_fir_adas.py
src/dwm/datasets/zut_fir_adas_common.py
configs/datasets/zut_fir_adas.yaml
```

目录结构：

```text
country/
  route/
    recording/
      16BitFrames/
      16BitTransformed/
      annotations/
```

标注是 YOLO 格式：

```text
class_id center_x center_y width height
```

parser 会根据图像尺寸转换成绝对像素坐标：

```text
YOLO normalized cx/cy/w/h
    ↓
absolute x1/y1/x2/y2
```

需要注意：

- recording 名称末尾 `_b` 表示 benchmark
- `train/val` 不使用 benchmark recording
- `test/benchmark` 只使用 `_b`
- `bbox_policy=any` 保留有 bbox 的 clip，同时允许内部出现空帧
- `bbox_policy=all` 要求每个时间帧都有 bbox
- TXT 数量非常大，因此使用 index cache
- `frame_directory=16BitFrames` 和 `16BitTransformed` 必须显式选择

## 11.5 MS2

文件：

```text
src/dwm/datasets/ms2.py
src/dwm/datasets/ms2_common.py
configs/datasets/ms2.yaml
```

MS2 原始数据没有 provider bbox。

当前 loader 读取已经生成的：

```text
records.jsonl
```

流程：

```text
生成的 records.jsonl
    ↓
accepted_statuses 过滤
    ↓
min_score 过滤
    ↓
class whitelist 过滤
    ↓
解析项目相对路径
    ↓
读取已经投影到 thermal 坐标系的 xyxy
    ↓
BBoxFrameRecord
```

需要注意：

- `MS2_ROOT` 指向原始图像
- `MS2_ANNOTATION_ROOT` 指向生成标注
- `records.jsonl` 中的 bbox 已经在 thermal 坐标系
- loader 不会再次执行 RGB 到 thermal 投影
- 不要把 raw MS2 目录直接当成有 bbox 的目录
- JSONL 解析结果有 index cache

## 11.6 VIVID++

文件：

```text
src/dwm/datasets/vivid_pp.py
src/dwm/datasets/vivid_pp_common.py
configs/datasets/vivid_pp.yaml
```

VIVID++ 的原始 ROS bag 没有 provider bbox；生成 artifact 由：

```text
frames.jsonl
detections.jsonl
images/
```

共同组成。一个 `MotionDataset` 通过 `annotation_mode` 选择三种路径：

```text
required
    只读生成 artifact，输出有框 clip

optional
    先以 raw bag 建立图像全集，再按 bag + thermal_sequence 叠加生成框
    没有匹配框的帧保留，condition_valid=False

none
    dataset_root 存在时从 raw bag 提取无框图像；
    annotation_root 存在时只把 frames.jsonl 当作无框图像索引
```

三种路径都复用同一套 `BBoxMotionDataset`。输出中的：

```text
bbox_available[T,V]   标记该视角是否有可用标注来源
condition_valid[T,V]  标记该视角当前帧是否真正有框
```

thermal bbox 优先级：

```text
bbox_thr_lidar_xyxy
bbox_thr_geometry_xyxy
bbox_thr_unexpanded_xyxy
bbox_thr_xyxy
```

multiview 时，RGB 使用 `bbox_rgb_xyxy`，并固定输出 `RGB, thermal` 两个视角。
raw bag 提取和 generated JSONL 索引都使用 `.dwm_cache`；`max_frames_per_bag`
是显式规模上限，`None` 才表示完整 bag。

需要注意：

- `VIVID_ANNOTATION_ROOT` 必须指向包含 `frames.jsonl` 的生成 artifact；
- `required` 需要同时存在 `detections.jsonl`，`none` 的 manifest-only 模式不需要；
- `VIVID_ROOT` 指向 raw `vivid_pp` 根目录，仅在 raw 模式或 optional 合并模式使用；
- `modality=thermal, view_mode=single` 是默认训练配置；
- raw bag 首次索引需要解码并落盘图像，首次构建较慢，后续命中 cache。

---

## 11.7 无 bbox 的 style 数据集

Lynred Mobility 使用 `metadata/metadata.csv` 索引
`8bits/qvga`、`8bits/vga`、`16bits/qvga` 或 `16bits/vga` 图像；
TartanRGBT 使用 day/run 下的左右 thermal rectified 序列。两者都明确
写入 `annotations_available=False`，不把无框图像伪造成框。

```text
annotation_mode=none      style：只输出 images
annotation_mode=optional  joint：输出黑 condition 和 False mask
```

它们和其他数据集共用 `BBoxMotionDataset` 的 clip、时间戳和 transform
逻辑；`max_frames_per_sequence` 只限制索引规模并保留源序列时间位置。
---

## 12. single 和 multi 的区别

本项目中有两种“multi”概念，必须区分。

### 12.1 多视角 multiview

表示同一个时间点有多个同步相机视角：

```text
[T][V]
```

当前真正支持多视角的主要是：

- BU-TIV Atrium
- FLIR video-test RGB/thermal
- VIVID++ generated RGB/thermal

### 12.2 多数据集 multi_bbox

配置文件：

```text
configs/datasets/multi_bbox.yaml
```

它使用 `torch.utils.data.ConcatDataset` 合并有真实或明确生成来源的
FLIR、LTIR v1、ZUT、MS2、VIVID++ thermal 单视角样本。每个子 loader
保持自己的 bbox 解析、过滤和时间采样规则。

### 12.3 多数据集 multi_style

配置文件：

```text
configs/datasets/multi_style.yaml
```

它仍然使用同一组 `<dataset>.py` 和 `common.py`，但把
`annotation_mode` 设为 `none`。BU-TIV、FLIR、LTIR v1、ZUT、MS2、
VIVID++ raw、Lynred Mobility、TartanRGBT 的图像都可进入同一个 style
入口；不读取 bbox sidecar，不生成 `bbox_condition_images`。

### 12.4 多数据集 multi_joint

配置文件：

```text
configs/datasets/multi_joint.yaml
```

有标注来源的子集使用 `optional`，无框来源也使用同一个 `optional` 分支；
因此每个 batch 的键集合一致。`bbox_condition_images` 对无框帧是黑图，
`condition_valid=False`，训练目标可以按 mask 跳过无效框监督。VIVID++
额外保留 generated-only `required` 入口，用于不匹配 raw 全集的稀疏
生成帧。

三个 multi 配置都统一为：

```text
thermal single view
T=4
image C=1
condition C=3  (style 模式省略 condition)
H=W=256
```

`max_frames_per_sequence`/`max_frames_per_bag` 是配置级规模上限，不改变
loader 的统一接口；删除上限或设为 `null` 才会索引完整序列或 bag。

## 13. 如何实际证明加载成功

### 13.1 只看一个 sample

```bash
FLIR_ROOT=/path/to/flir \
PYTHONPATH=src \
python scripts/prepare/load_dataset_config.py \
  --cfg configs/datasets/flir.yaml
```

重点看输出：

```text
length
construction_seconds
sample_seconds
bbox_count
sample 中的字段和 shape
```

### 13.2 测量 DataLoader 速度

```bash
FLIR_ROOT=/path/to/flir \
PYTHONPATH=src \
python scripts/prepare/load_dataset_config.py \
  --cfg configs/datasets/flir.yaml \
  --loader \
  --batches 8
```

输出中的：

```text
construction_seconds
sample_seconds
samples_per_second
```

分别表示：

```text
construction_seconds：索引和对象构造耗时
sample_seconds：一个 dataset sample 的实际读取和 transform 耗时
samples_per_second：连续读取 batch 的吞吐
```

### 13.3 测试 DatasetAdapter 动态索引

```bash
FLIR_ROOT=/path/to/flir \
PYTHONPATH=src \
python scripts/prepare/load_dataset_config.py \
  --cfg configs/datasets/flir.yaml \
  --loader \
  --dynamic-index 0-2-128-192 \
  --batches 2
```

这个索引表示：

```text
第 0 个 clip
随机截取 2 帧
动态 resize 到 128×192
```

如果输出：

```text
vae_images:           (2, 1, 1, 128, 192)
box_condition_images: (2, 1, 3, 128, 192)
```

就证明动态 temporal slicing、图像读取、condition transform 和 stack 都实际执行了。

### 13.4 可视化 batch 和检查视频连续性

脚本：

```text
scripts/prepare/visualize_dataset_batch.py
```

它和训练循环使用同一个 DataLoader 语义：每次调用 `next(iterator)` 得到一个
完整 batch，先校验 `[B,T,V]`、`sample_ids`、frame 顺序、`pts`、bbox mask
和 condition，再保存一个 batch contact sheet，以及每个 sample 的时间 contact
sheet、GIF 和 `report.json`。这里的 sample 是 batch 中的一个独立 T 帧视频 clip。

```bash
PYTHONPATH=src \
python scripts/prepare/visualize_dataset_batch.py \
  --task joint \
  --batches 2 \
  --max-samples-per-batch 0 \
  --output-dir outputs/joint_visual
```

默认 `--max-samples-per-batch 0` 会渲染完整 batch；如设置为正数，只渲染
batch 中前几个 sample。batch contact sheet 的文件名为：

```text
batch_0000_contact.png
```

`style` 模式只显示 `vae_images`；存在有效 bbox 的 `bbox`/`joint` clip
额外显示 bbox condition 和 image+bbox 叠加图。joint 中无框帧不会伪造
框，只保留原图并在报告中记录 `bbox_frames=0`。
单通道热红外默认先做 1%/99% percentile contrast stretch，再使用
`inferno` false-color 映射为 RGB；`--color-map gray|inferno|turbo` 可切换，
`--display-scale 2` 默认放大显示。该转换只作用于可视化文件，不改变 batch
中的 `vae_images` 数值；输出 PNG/GIF 为 8-bit RGB。

如果要在已有训练脚本中只检查前几个 batch，可直接复用 hook：

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
    # training_step(batch)
```

校验失败会立即抛出异常，避免把错位帧、跨序列 clip 或无效 bbox
condition 静默送入训练。

---

## 13.5 OpenDWM 风格的 `mix_config` 动态 batch

配置文件顶层存在非空 `mix_config` 时，`load_dataloader_from_config()` 会构造
`dwm.utils.sampler.VariableVideoBatchSampler`，并把它作为 DataLoader 的
`batch_sampler`。没有 `mix_config` 时保持现有固定 `batch_size`、`shuffle`
和普通 DataLoader 行为不变。

本项目的可运行示例是 `configs/datasets/butiv_mix.yaml`；它固定 `T=4`，
按 `0.6/0.3/0.1` 选择 `256/192/128` 方形分辨率，并使用每卡
`B=2/4/6`。原始 `butiv.yaml` 不包含 `mix_config`，行为不受影响。

格式与 OpenDWM 官方配置一致：

```yaml
mix_config:
  "256-448": [0.6, [[19, 2, 1]]]
  "176-304": [0.3, [[19, 4, 1]]]
  "144-256": [0.1, [[19, 6, 1]]]
```

含义为：

```text
"H-W": [分辨率权重, [[T, 每卡 batch size B, 该选项权重], ...]]
```

官方当前配置的实际用法是按概率改变 `H/W`，同时为不同分辨率配置不同的
每卡 `B`；同一份配置中的 `T` 通常保持不变。实现保留了协议对多个 `T`
选项的支持，但不会擅自改变现有数据集的时间长度。Sampler 产生
`index-T-H-W` 字符串索引，`DatasetAdapter` 再按同一批次的 `T/H/W`
进行时间裁剪和图像变换。

Sampler 的分布式行为与 OpenDWM 一致：每个 rank 每一步得到一个完整 batch，
不同 rank 可以使用不同分辨率和 `B`；索引分桶由 `seed + epoch` 确定，尾部
补齐不会产生空 batch。`DataLoader` 中的 `batch_size`、`shuffle`、
`sampler`、`drop_last` 会在启用该模式时交给 sampler 管理。

`load_dataset_config.py --batches N` 会根据实际取出的 batch 计算样本数，
因此不同分辨率使用不同 `B` 时，吞吐统计仍然准确。

---

## 14. 直接用 Python 调试

不依赖命令行脚本时，可以直接执行：

```python
import time

from dwm.common import load_dataloader_from_config

start = time.perf_counter()
loader, config = load_dataloader_from_config(
    "configs/datasets/flir.yaml"
)
constructed = time.perf_counter() - start

start = time.perf_counter()
sample = loader.dataset[0]
read_one = time.perf_counter() - start

start = time.perf_counter()
iterator = iter(loader)
for _ in range(8):
    batch = next(iterator)
batched = time.perf_counter() - start

print("dataset length:", len(loader.dataset))
print("construction seconds:", constructed)
print("one sample seconds:", read_one)
print("last batch image shape:", batch["vae_images"].shape)
print("last batch condition shape:", batch["box_condition_images"].shape)
print("8 batch seconds:", batched)
```

更推荐使用调试脚本，因为它会正确复用一个 iterator：

```python
iterator = iter(loader)
for _ in range(8):
    batch = next(iterator)
```

这样测出的吞吐更接近连续 DataLoader 读取。

---

## 15. 常见问题

### 15.1 `unresolved environment variable`

说明没有设置对应的环境变量，例如：

```bash
FLIR_ROOT=/path/to/flir
```

### 15.2 `root does not exist`

通常是路径层级错误。例如 loader 允许的根目录可能是：

```text
/path/data/raw/flir
/path/data/raw/flir/FLIR_ADAS_v2
```

但不能把 COCO 的某个 `data/` 子目录当成 dataset root。

### 15.3 annotation mode 与 bbox 缺失

`annotation_mode` 的含义固定为：

```text
required：没有可用标注的 clip 被过滤
optional：无标注帧保留，condition_valid=False
none：不读取标注并省略 bbox_condition_images
```

因此 MS2 raw、VIVID++ raw、Lynred Mobility、TartanRGBT 在 style 配置中
没有 bbox 是预期行为；joint 配置会保留它们的图像，但不会伪造框。
MS2/VIVID++ 需要 generated annotation root 时，必须显式配置对应环境变量。

### 15.4 batch 中 bbox 无法 stack

这是正常的数据特性：不同帧的 bbox 数量不同。

应让：

```text
boxes/labels/track_ids 保持 list
```

而不是强行堆成固定 Tensor。

### 15.5 ZUT 初始化较慢

ZUT 有大量 TXT 标注文件。第一次运行会扫描和解析，后续会使用：

```text
.dwm_cache
```

如果标注文件发生变化，删除对应 cache 后重新构建。

### 15.6 `num_workers=0` 是否代表最终训练必须使用 0

不是。

`num_workers=0` 主要用于：

- 调试
- 清晰测量单进程耗时
- 方便定位错误

确认正确后，可以用 override 测试：

```bash
--set dataloader.num_workers=2
```

但应重新测量，因为 worker 数量、磁盘带宽、图像解码和内存都会影响实际吞吐。

---

## 16. 给别人讲解时的 30 秒版本

可以这样介绍：

> 我们把每个数据集的原始格式解析层和通用采样层分开。`*_common.py` 负责理解该数据集的目录和标注格式，把它们统一成带路径、时间和 bbox 的 `BBoxFrameRecord`。具体的 `MotionDataset` 再根据 sequence length 和采样策略生成时间 clip。真正访问 sample 时，shared `BBoxMotionDataset` 才读取当前 T×V 图像，归一化 16-bit 数据并画 bbox condition。最后 `DatasetAdapter` 根据 YAML 中的 transform 把 PIL 图像 resize、ToTensor 并 stack 成 `[T,V,C,H,W]`，DataLoader 再加 batch 维得到 `[B,T,V,C,H,W]`。所有路径、数据集类型、子集和 transform 都由 config 决定，所以新增数据集时主要只需要实现它自己的 parser 和 MotionDataset，不需要复制整套 batch 逻辑。

---

## 17. 最小记忆清单

```text
Python 入口：scripts/prepare/load_dataset_config.py
配置工厂：src/dwm/common.py
通用 Adapter：src/dwm/datasets/common.py
通用 bbox clip：BBoxMotionDataset
具体格式解析：各数据集的 *_common.py
具体 loader：各数据集的 <dataset>.py
模型 Tensor transform：configs/datasets/*.yaml 的 DatasetAdapter
多数据集组合：configs/datasets/multi_bbox.yaml、multi_style.yaml、multi_joint.yaml
速度证明：--loader --batches 8
动态索引证明：--dynamic-index 0-2-128-192
统一框策略：annotation_mode=required/optional/none
有效性掩码：bbox_available、condition_valid
```
