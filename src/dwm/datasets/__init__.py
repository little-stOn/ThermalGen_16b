"""Local dataset loaders with a shared bbox sequence contract."""

__all__ = ["MotionDataset"]


def __getattr__(name: str):
    if name == "MotionDataset":
        from .butiv import MotionDataset

        return MotionDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# 文件讲解：__getattr__ 只保留历史的 BU-TIV MotionDataset 懒加载入口；
# FLIR、LTIR、ZUT、MS2、VIVID++ 使用各自模块中的 MotionDataset，配置工厂
# 通过完整 dotted path 导入，避免不同数据集的同名类互相覆盖。
