# Pipeline 输出结果：人话说明

## 先记住一件事

`result` 不是一个单独的视频文件，而是 Pipeline 一次生成后返回的一个“结果包”。

这个结果包里同时放了：

- 模型内部的中间结果；
- 最终生成的视频；
- 这次生成使用的输入；
- 生成过程的参数和记录。

可以把它想成：

```text
result = 视频成品 + 制作过程记录 + 原始订单
```

---

## 1. `result.latents`：模型内部的“半成品数字稿”

`latent` 不是普通人能直接打开的视频，也不是一张图片。

模型不会直接在原始像素上一步一步生成视频，而是先把视频压缩成一组数字，然后在这组数字上完成去噪和生成。这个数字空间就叫 latent 空间。

因此：

```text
result.latents = 模型生成完成后的内部数字表示
```

它的作用是：

- 继续进行后续去噪；
- 交给 `decode_video()` 还原成视频；
- 保存或复用生成过程；
- 做 latent 层面的分析。

它通常不能直接用图片查看器打开。

类比：

> `latents` 像电影还没有冲印成胶片之前的数字底稿。

---

## 2. `result.frames`：真正的视频内容

`frames` 是 latent 经过模型解码后的结果，也就是视频帧的像素数据。

```text
result.frames = 生成的视频内容
```

它通常仍然是 PyTorch tensor，而不是 MP4 文件。可以理解为：

```text
[B, T, V, C, H, W]
```

例如：

```text
[2, 4, 1, 1, 256, 256]
```

表示 2 个视频、每个视频 4 帧、1 个视角、单通道、256×256。

它后续可以交给 `VideoWriter` 写成：

- MP4；
- PNG 序列；
- 16-bit TIFF 序列；
- 其他热红外格式。

`frames` 之所以写“可选”，是因为：

```python
output_type="latent"
```

时，Pipeline 只返回 latent，不进行视频解码；或者具体模型没有返回解码结果。

类比：

> `frames` 是已经冲印出来、可以播放或保存的视频画面。

---

## 3. `result.batch`：这次生成使用的“订单和条件”

`batch` 不是生成出来的视频，而是这次生成所使用的输入信息，并且已经被 Pipeline 整理成统一格式。

它可能包含：

- 输入图像；
- bbox 条件图像；
- bbox 坐标；
- 类别标签；
- 时间戳；
- 帧率；
- 哪些帧有有效标注；
- 哪些条件可以使用。

```text
result.batch = 模型这次按照什么要求生成
```

类比：

> `batch` 是客户下的订单，`frames` 是按照订单做出来的成品。

保留 `batch` 的好处是，评估生成结果时可以知道：

```text
这个视频是根据什么条件生成的？
```

---

## 4. `result.num_steps`：去噪次数

扩散模型不是一步就得到最终视频，而是从一团随机噪声开始，逐步去除噪声。

```python
result.num_steps = 30
```

表示这次生成执行了 30 次去噪更新。

它表示的是：

- 生成过程走了多少步；
- 不是视频有多少帧；
- 不是视频持续多少秒；
- 不是训练了多少轮。

一般来说：

- 步数少：生成更快；
- 步数多：通常有更多机会完善结果，但生成更慢。

类比：

> `num_steps` 像照片冲洗过程中经过了多少道处理工序。

---

## 5. `result.metadata`：这次生成的“实验记录卡”

`metadata` 是一个字典，用来记录这次生成的参数和说明，不是视频画面本身。

当前 Pipeline 会记录类似信息：

```python
{
    "shape": [...],
    "num_steps": 30,
    "guidance_scale": 7.5,
    "box_coordinate_space": "pixel",
}
```

它可以回答：

- 生成的视频/latent 是什么 shape？
- 使用了多少次去噪？
- CFG 强度是多少？
- bbox 使用的是什么坐标空间？

类比：

> `metadata` 是贴在成品旁边的实验记录卡，记录怎么做出来，但不是成品本身。

---

## 6. 一次完整生成的关系

```text
输入 batch
   ↓
模型根据 batch 准备条件
   ↓
随机噪声 latent
   ↓
经过 num_steps 次去噪
   ↓
最终 result.latents
   ↓
decode_video()
   ↓
result.frames
```

同时，Pipeline 把输入和过程信息一起打包：

```text
result.batch       使用了什么输入
result.latents     模型内部的最终数字稿
result.frames      解码后的视频帧
result.num_steps   去噪了多少次
result.metadata    这次生成的参数记录
```

## 7. 最简记忆版

```text
latents  = 模型内部的数字半成品
frames   = 可以还原成视频的画面
batch    = 这次生成使用的输入条件
steps    = 去噪次数
metadata = 生成过程的参数记录
```
