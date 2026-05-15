import os
from . import SCTransNetAdaptive as sna
from . import Blocks as bs
from .SCTransNetAdaptive import SCTransNetAdaptive as SCTransNet
import torch
import torch.nn as nn
from . import Utilities as uti
import json;
from thop import profile
from . import ConfigManager as cm

class NetworkBuilder:
    '''
    网络生成器，根据输入数据自动配置网络结构
    '''
    def __init__(self,img_size,in_channels,n_classes,configuration_manager,dim=3,mode='train',vis=False,deepsuper=True):
        '''
        初始化网络构建器
        
        :param self: 说明
        :param img_size: 输入图像大小，2D为二维元组，3D为三维元组
        :param in_channels: 输入通道数
        :param n_classes: 输出类别数
        :param dim: 图像维数
        :param mode: 工作模式
        :param vis: 是否可视化权重
        :param deepsuper: 是否开启深监督
        :param configuration_manager: nnU-Net提供的该数据集的配置信息
        '''
        self.img_size=img_size
        self.in_channels=in_channels
        self.n_classes=n_classes
        self.vis=vis
        self.mode=mode
        self.deepsuper=deepsuper
        self.configmanger=configuration_manager
        self.dim=dim#维度数

        self.config=cm.ConfigManager.GetConfig(self.configmanger)#取得配置
        self.scale_factors=self.compute_scale_factors_from_strides(self.config.strides)[1:]
        self.config.n_channels=in_channels
        bs.Config=self.config

    #注意：必须使用ModuleList或者Sequential，否则层不会被PyTorch注册！
    def Build(self):
        #先根据维度确定使用的模块
        self.SetModules()

        #构建模型。
        network=SCTransNet(self.config,self.img_size,self.scale_factors,self.in_channels,self.n_classes,self.vis,self.mode,self.deepsuper)
        #network=APAUNet(self.in_channels,self.n_classes,self.config.features_per_stage,self.config.strides)

        return network
    
    def SetModules(self):
        '''
        根据维度确定模块
        '''
        if self.dim==3:
            bs.Dimension=3
            bs.convclass=nn.Conv3d
            bs.batch_norm=nn.InstanceNorm3d
            bs.GroupNormGroups=8
            bs.convtransposexd=nn.ConvTranspose3d
            bs.unsample_mode='trilinear'
            bs.instance_norm=nn.InstanceNorm2d#注意，因为只有注意力图用到了这个，而注意力图只会是2D
            bs.max_pool=nn.MaxPool3d
            bs.avg_poolxd=uti.avg_pool_3d
            bs.rearrange_xd=uti.rearrange_3d
            bs.to_3d=uti.to_3d_3d
            bs.to_4d=uti.to_4d_3d
            bs.adaptiveAvgPool=nn.AdaptiveAvgPool3d
            bs.getscalefactor=uti.GetScaleFactor3d
            bs.rearrange_out=uti.rearrange_out_3d
        if self.dim==2:
            bs.Dimension=2
            bs.convclass=nn.Conv2d
            bs.batch_norm=nn.InstanceNorm2d
            bs.GroupNormGroups=8
            bs.convtransposexd=nn.ConvTranspose2d
            bs.unsample_mode='bilinear'
            bs.instance_norm=nn.InstanceNorm2d
            bs.max_pool=nn.MaxPool2d
            bs.avg_poolxd=uti.avg_pool_2d
            bs.rearrange_xd=uti.rearrange_2d
            bs.to_3d=uti.to_3d_2d
            bs.to_4d=uti.to_4d_2d
            bs.adaptiveAvgPool=nn.AdaptiveAvgPool2d
            bs.getscalefactor=uti.GetScaleFactor2d
            bs.rearrange_out=uti.rearrange_out_2d
    
    def compute_scale_factors_from_strides(self,strides):
        """
        根据 strides 列表计算每个阶段相对于原始尺寸的缩放因子。
        
        参数:
            strides (list of tuple): 每个阶段的步长，例如 [(1,1,1), (1,2,2), ...]
        
        返回:
            list of tuple: 每个阶段的缩放因子，长度与 strides 相同。
        """
        scale_factors = []
        current = [1.0] * len(strides[0])  # 初始化为全1
        for s in strides:
            current = [c / step for c, step in zip(current, s)]
            scale_factors.append(tuple(current))
        return scale_factors
        
if __name__ == '__main__':
    """测试代码：验证模型结构和计算量"""

    script_dir = os.path.dirname(__file__)  # 获取当前脚本所在目录
    file_path = os.path.join(script_dir, 'plans.json')
    with open(file_path, 'r') as f:#加载配置
        plans = json.load(f)['configurations']['3d_fullres']
    
    dim=3
    img_size=(512,512)
    inputs=torch.rand(1,1,512,512) # 创建随机输入
    
    if dim==3:
        img_size = (64, 160, 160)  # 非立方体示例
        inputs = torch.rand(1, 1,64, 160, 160)  
    nb=NetworkBuilder(img_size,1,16,plans,dim=dim) #创建模型构建器
    model = nb.Build()#创建模型

    print(f"\n开始前向传播...")
    print(f"输入形状: {inputs.shape}")
    print(f"输入尺寸: {img_size}")
    print(f"基础通道数: {nb.config.base_channel}")

    output = model(inputs)  # 前向传播
    
    # 打印模型结构
    print(f"\n模型结构概要:")
    print(model)
    
    # 使用thop计算FLOPs和参数量
    try:
        flops, params = profile(model, (inputs,))
        print("-" * 50)
        print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')  # 打印GFLOPs
        print('Params = ' + str(params / 1000 ** 2) + ' M')  # 打印百万参数
    except Exception as e:
        print(f"计算FLOPs时出错: {e}")
