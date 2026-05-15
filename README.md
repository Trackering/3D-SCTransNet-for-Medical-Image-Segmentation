# 3D SCTransNet for Medical Image Segmentation
本项目是哈尔滨工业大学（威海）本科毕业论文《基于融合交叉注意力Transformer的3D医学图像小目标分割方法》的代码实现。
对原用于2D 红外小目标分割任务的**SCTransNet**（Yuan et al., TGRS 2024）进行3D扩展与改进以适配3D 医学图像分割。
原SCTransNet：[Github](https://github.com/xdFai/SCTransNet)，[Paper](https://ieeexplore.ieee.org/document/10486932)。
## 1.运行环境
操作系统：Ubuntu 22.04
PyTorch 2.8.0
Python版本：3.12
CUDA版本：12.8  
在其它版本差异不大的环境下大概也能运行。

## 2.如何使用？
本模型利用**nnU-Net**([Github](https://github.com/MIC-DKFZ/nnUNet),[Paper](https://www.nature.com/articles/s41592-020-01008-z))进行训练和预测。因此要最简单的方法是置于nnU-Net框架之中。但这不是必要的，下面介绍两种方式。**注意**：使用nnU-Net框架时，训练器默认设计为模型部分使用其提供的模型结构参数。

### 2.1 直接使用
只需model文件夹内所有代码文件，在您的训练器中导入SCTransNetAdaptive.py中的SCTransNetAdaptive即可。  
```python
from SCTransNetAdaptive import SCTransNetAdaptive # 导入模块

# 创建模型
# 具体相关参数SCTransNetAdaptive.py中有详细的docstring介绍
network = SCTransNetAdaptive(config,img_size,scale_factors,in_channels,n_classes,vis,mode,deepsuper)

# 然后network就是一个能直接进行前向传播的nn.Module
result = network(inputdata)
```

### 2.2 使用nnU-Net
您需要先安装**nnU-Net v2**，具体方法见其[Readme](https://github.com/MIC-DKFZ/nnUNet)。

首先将训练器nnUNetTrainerSCTransNet.py放置于"nnunetv2\training\nnUNetTrainer"文件夹内，然后把model文件夹也放置于此。  
然后运行nnUNetv2_train命令时使用`-tr`指定nnUNetTrainerSCTransNet作为训练器，例如:
```bash
nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainerSCTransNet
```
接着就是标准的nnU-Net训练流程。

而对于预测，您只需在执行`nnUNetv2_predict`命令时，指定`-tr nnUNetTrainerSCTransNet`即可。
