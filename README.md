# 3D SCTransNet for Medical Image Segmentation
本项目是哈尔滨工业大学（威海）本科毕业论文《基于融合交叉注意力Transformer的3D医学图像小目标分割方法》的代码实现。
对原用于2D 红外小目标分割任务的SCTransNet（Yuan et al., TGRS 2024）进行3D扩展与改进以适配3D 医学图像分割。
原SCTransNet：[Github](https://github.com/xdFai/SCTransNet)，[Paper](https://ieeexplore.ieee.org/document/10486932)。
## 1.运行环境
操作系统：Ubuntu 22.04
PyTorch 2.8.0
Python版本：3.12
CUDA版本：12.8
在其它版本差异不大的环境下大概也能运行。
## 2.如何使用？
本模型利用**nnU-Net**([Github](https://github.com/MIC-DKFZ/nnUNet),[Paper](https://www.nature.com/articles/s41592-020-01008-z))进行训练和预测。因此要最简单的方法是置于nnU-Net框架之中。但这不是必要的，下面介绍两种方式。
### 2.1 直接使用
只需model文件夹内所有代码文件，在您的训练器中导入SCTransNetAdaptive.py中的SCTransNet即可。
```python
from SCTransNetAdaptive import SCTransNetAdaptive # 导入模块

# 创建模型
# 具体相关参数SCTransNetAdaptive.py中有详细的docstring介绍
network = SCTransNetAdaptive(config,img_size,scale_factors,in_channels,n_classes,vis,mode,deepsuper)
#然后network就是一个能直接进行前向传播的nn.Module
result = network(inputdata)
```
### 2.2 使用nnU-Net

