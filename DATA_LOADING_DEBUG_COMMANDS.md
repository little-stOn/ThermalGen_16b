# BU-TIV 数据加载调试指令

## 运行环境

```bash
cd /inspire/qb-ilm/project/wuliqifa/public/mayilin/ir_dwm_base
source .venv-butiv/bin/activate
export PYTHONPATH=src
export BUTIV_ROOT=/inspire/qb-ilm/project/wuliqifa/public/mayilin/ir_dwm_base/data/raw/bu_tiv
```

## VS Code 调试

```text
Ctrl+Shift+P → Developer: Reload Window       # 重载调试配置
Ctrl+Shift+D                                      # 打开运行和调试
Ctrl+Shift+P → Debug: Select and Start Debugging # 选择调试配置
F5                                                 # 启动或继续
F10                                                # 单步跳过
F11                                                # 单步进入
Shift+F5                                           # 停止调试
```

## Debug Console：断点 624

```python
(type(loader).__name__, type(loader.batch_sampler).__name__, len(loader), loader.batch_size)  # 查看 loader
loader.dataset                                                                            # 查看 dataset
loader.batch_sampler                                                                       # 查看 sampler
```

## Debug Console：断点 631

```python
batch.keys()                                                   # 查看字段
(tuple(batch["vae_images"].shape), tuple(batch["box_condition_images"].shape))  # 查看图像 shape
(len(batch["sample_ids"]), len(batch["sample_ids"][0]))       # 查看 B、T
batch["sample_ids"][0]                                        # 查看 sample ID
batch["bbox_available"]                                       # 查看 bbox mask
batch["condition_valid"]                                      # 查看 condition mask
batch["pts"][0]                                               # 查看时间戳
validate_batch(batch)                                          # 执行完整校验
```

## 完整可视化

```bash
python scripts/prepare/visualize_dataset_batch.py \
  --cfg configs/datasets/butiv_mix.yaml \
  --batches 1 \
  --max-samples-per-batch 0 \
  --output-dir outputs/debug_butiv_mix
```

```bash
python scripts/prepare/visualize_dataset_batch.py \
  --cfg configs/datasets/butiv.yaml \
  --batches 1 \
  --max-samples-per-batch 0 \
  --output-dir outputs/debug_butiv_fixed
```

## 加载速度

```bash
python scripts/prepare/load_dataset_config.py \
  --cfg configs/datasets/butiv_mix.yaml \
  --loader \
  --batches 20
```

```bash
python scripts/prepare/load_dataset_config.py \
  --cfg configs/datasets/butiv.yaml \
  --loader \
  --batches 20
```

```bash
python scripts/prepare/load_dataset_config.py \
  --cfg configs/datasets/butiv_mix.yaml \
  --loader \
  --batches 20 \
  --set dataloader.num_workers=4
```

## GPU 监控

```bash
nvidia-smi dmon -s u                                      # 连续查看双卡利用率
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv  # 查看单次状态
```
