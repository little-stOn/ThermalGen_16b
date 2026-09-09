"""Local dataset adapter namespace."""

# 文件讲解：该包只提供 dwm 命名空间；配置工厂在 dwm.common，具体数据集
# loader 在 dwm.datasets。不要在这里放数据集扫描逻辑，以免导入包时触发
# 大规模索引或图像读取。
