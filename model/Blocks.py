from __future__ import absolute_import  # 确保导入的模块是绝对路径的，避免相对导入混淆
from __future__ import division         # 确保除法运算总是返回浮点数
from __future__ import print_function   # 确保print函数在Python2和3中行为一致

from torch.nn import Dropout  # PyTorch神经网络组件
from torch.nn import Conv3d, BatchNorm3d, MaxPool3d, AvgPool3d, InstanceNorm3d #使用3D算法
import torch.nn as nn     # PyTorch神经网络模块
import torch              # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch函数式接口
import numbers            # 数字类型检查
from .Utilities import avg_pool_3d,GetPatchNum,rearrange_3d
from . import Utilities as uti#2D/3D切换方法

###全局变量，用于切换2D或3D分割所需的相关模块
Dimension=3
'''图像维度'''

convclass=nn.Conv3d
'''卷积方法，可选用2D或3D。如nn.Conv3d'''

batch_norm=nn.InstanceNorm3d
'''批归一化方法，可选用2D或3D。如nn.BatchNorm3d'''

unsample_mode='trilinear'
'''线性插值上采样模式，可选用trilinead或者bilinear'''

instance_norm=InstanceNorm3d
'''实例归一化，可选用2D或3D。如nn.InstanceNorm3d'''

max_pool=MaxPool3d
'''最大池化方法，可选用2D或3D。如nn.MaxPool3d'''

avg_poolxd=avg_pool_3d
'''平均池化方法，可选用2D或3D。如avg_pool_3d'''

rearrange_xd=rearrange_3d
'''张量尺度重塑方法，可选用2D或3D。如rearrange_3d'''

to_3d=uti.to_3d_3d
'''重塑为3D张量，可选择是3D版本还是2D版本'''
to_4d=uti.to_4d_3d
'''重塑为4D张量，可选择是3D版本还是2D版本'''

adaptiveAvgPool=nn.AdaptiveMaxPool3d
'''全局平均池化，可选用2D或3D。如nn.AdaptiveAvgPool3d

**仅用于ECA_Layer**，目的是把所有体素压缩成通道信息，计算通道注意力
'''

getscalefactor=uti.GetScaleFactor3d
'''获取正确维度数的缩放因子'''

rearrange_out=uti.rearrange_out_3d
'''将Attention前向传播中的out_维度重塑方法，可选用2D或3D如rearrange_out_3d'''

GroupNormGroups=8
'''使用GroupNorm方法时的分组个数'''

def NormMethod(out_channels):
    '''
    归一化方法

    :param out_channels: 输出通道数
    '''
    if(batch_norm==nn.GroupNorm):
        return batch_norm(GroupNormGroups,out_channels)#此时用的是GroupNorm
    else:
        return batch_norm(out_channels)#此时以为用的是BatchNorm或者InstanceNorm

convtransposexd=nn.ConvTranspose3d
'''转置卷积方法，用于Reconstruct和UpBlock_attention中进行上采样'''
######

class Channel_Embeddings(nn.Module):
    """通道嵌入层：将特征图分割为patch并进行位置编码"""

    def __init__(self, config, patch_size, img_size, in_channels,target_embedd_size):
        '''
        初始化通道嵌入层
        
        :param self: self
        :param config: 配置字典
        :param patchsize: 图像的Patch Size，请根据2D或3D来确定图像维度，下同
        :param img_size: 图像的尺寸大小
        :param target_embedd_size: 嵌入后的图像尺寸大小
        :param in_channels: 输入通道数
        '''
        super().__init__()
        self.IsDebug=False

        # 计算patch数量
        ####TODO:这里要分2D或3D来计算n_patches!#Finished
        n_patches = GetPatchNum(img_size,patch_size)

        # 使用卷积实现patch embedding：通过卷积核大小和步长等于patch大小来实现
        self.patch_embeddings = convclass(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        
        # # 位置编码：可学习的位置参数，形状为(1, patch数量, 通道数)
        # self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels)) #？但似乎位置编码没有使用

        # #我打算使用位置编码
        # # 位置编码：形状 [1, in_channels, D, H, W]
        # self.position_embeddings = nn.Parameter(torch.zeros(1, in_channels, *target_embedd_size))
        # # 初始化（使用正态分布）
        # nn.init.trunc_normal_(self.position_embeddings, std=0.02)

        self.dropout = Dropout(config.transformer["embeddings_dropout_rate"])  # dropout层

    def forward(self, x):
        if x is None:  # 如果输入为空，直接返回
            return None
        
        x = self.patch_embeddings(x)  # 应用patch embedding

        # # 将位置编码加到特征上（注意：位置编码形状为 [1, C, D, H, W]）
        # x = x + self.position_embeddings

        # 注意：原代码中位置编码没有实际使用，这里直接返回patch embedding结果
        return x

class Reconstruct(nn.Module):
    """特征重建层：将处理后的特征上采样并重建到原始尺寸"""
    def __init__(self, in_channels, out_channels, kernel_size,target_size):
        '''
        初始化Reconstruct
        
        :param self: 说明
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核尺寸
        :param target_size: 重建后的目标尺寸，支持2维或3维元组
        '''
        super(Reconstruct, self).__init__()
        # 根据卷积核大小设置padding，保持空间尺寸
         # 3D卷积的padding计算
        if kernel_size == 3:
            padding = 1
        elif kernel_size == (3, 3, 3):
            padding = (1, 1, 1)
        else:
            padding = 0
        
        #这个采用插值，我觉得没有可学习的参数，不好，但是不得不用，因为缩放比例有时很大，用卷积计算量太大
        self.up=nn.Upsample(size=target_size, mode=unsample_mode, align_corners=True)#上采样
        # # 重建卷积层
        self.conv = convclass(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        # 转置卷积：输入通道 in_channels，输出 out_channels，核大小与步长设为 scale_factor
        # self.deconv = convtransposexd(
        #     in_channels, out_channels,
        #     kernel_size=scale_factor,  # 例如 (2,2,2)
        #     stride=scale_factor,
        #     padding=padding,
        #     output_padding=0
        # )
        self.norm = NormMethod(out_channels)  # 批归一化
        self.activation = nn.ReLU(inplace=True)   # ReLU激活函数
        
        self.target_size=target_size

    def forward(self, x):
        if x is None:  # 如果输入为空，直接返回
            return None

        # 线性插值上采样到目标尺寸
        x = self.up(x)

        # 卷积 + 归一化 + 激活
        out = self.conv(x)
        # out=self.deconv(x)#直接转置卷积
        out = self.norm(out)
        out = self.activation(out)
        return out

class eca_layer(nn.Module):
    """高效通道注意力模块"""
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        padding = k_size // 2  # 保持输出尺寸不变的padding
        self.avg_pool = adaptiveAvgPool(output_size=1)  # 2D/3D全局平均池化
        self.conv = nn.Sequential(
            # 1D卷积捕获通道间关系
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size, padding=padding, bias=False),
            nn.Sigmoid()  # Sigmoid激活生成注意力权重
        )
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        # 全局平均池化获取通道统计信息
        out = self.avg_pool(x)
        # 重塑为 [B, 1, C] 以适应1D卷积
        out = out.view(x.size(0), 1, x.size(1))
        # 1D卷积 + Sigmoid生成通道注意力权重
        out = self.conv(out)
        # 重塑回 [B, C, 1, 1, 1] 并与原始特征相乘
        if(Dimension==2):#这里没必要再外部传入模块了，因此内部确定。但是最好还是不要在forward里面用判断，这样编译时又会报警
            out = out.view(x.size(0), x.size(1), 1, 1)
        if(Dimension==3):
            out = out.view(x.size(0), x.size(1), 1, 1, 1)
        return out * x  # 通道注意力加权

class BiasFree_LayerNorm(nn.Module):
    """无偏置的LayerNorm变体"""
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        # 确保normalized_shape是元组格式
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1  # 确保是一维的

        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 可学习的缩放参数
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # 计算方差（无偏估计设为False）
        sigma = x.var(-1, keepdim=True, unbiased=False)
        # 归一化：x / sqrt(var + eps) * weight
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """带偏置的LayerNorm变体"""
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        # 确保normalized_shape是元组格式
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1  # 确保是一维的

        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 可学习的缩放参数
        self.bias = nn.Parameter(torch.zeros(normalized_shape))   # 可学习的偏置参数
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # 计算均值和方差
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        # 归一化：(x - mean) / sqrt(var + eps) * weight + bias
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm3d(nn.Module):
    """3D LayerNorm：处理5D/4D张量 [B, C, (D,) H, W] 的LayerNorm"""
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm3d, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        #d, h, w = x.shape[-Dimension:]  # 获取2D/3D空间维度
        b,c,d,h,w=uti.UnZipShape(x.shape)#获取形状
        # 转换为2D/3D -> LayerNorm -> 转换回4D/5D
        return to_4d(self.body(to_3d(x)), d, h, w)

# Complementary Feed-forward Network (CFN)
class FeedForward(nn.Module):
    """3D互补前馈网络"""
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        # 计算隐藏层维度：输入维度 * 扩展因子
        hidden_features = int(dim * ffn_expansion_factor)

        # 输入投影：将通道数扩展为原来的2倍（用于两个分支）
        self.project_in = convclass(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # 多尺度深度卷积：3x3和5x5卷积捕获不同感受野的空间信息
        # 注意：3D卷积核可以是 (3,3,3) 或 (5,5,5)
        self.dwconv3x3 = convclass(hidden_features, hidden_features, kernel_size=3, stride=1, 
                                  padding=1, groups=hidden_features, bias=bias)
        self.dwconv5x5 = convclass(hidden_features, hidden_features, kernel_size=5, stride=1, 
                                  padding=2, groups=hidden_features, bias=bias)
        self.relu3 = nn.ReLU()  # 3x3分支的激活函数
        self.relu5 = nn.ReLU()  # 5x5分支的激活函数
        
        # 输出投影：将通道数恢复为原始维度
        self.project_out = convclass(hidden_features * 2, dim, kernel_size=1, bias=bias)
        self.eca = eca_layer(dim)  # 高效通道注意力模块

    def forward(self, x):
        # 将输入投影到高维并分割为两个分支
        x_3, x_5 = self.project_in(x).chunk(2, dim=1)
        
        # 3x3分支：深度卷积 + ReLU
        x1_3 = self.relu3(self.dwconv3x3(x_3))
        # 5x5分支：深度卷积 + ReLU  
        x1_5 = self.relu5(self.dwconv5x5(x_5))
        
        # 合并两个分支的特征
        x = torch.cat([x1_3, x1_5], dim=1)
        # 投影回原始维度
        x = self.project_out(x)
        # 应用通道注意力
        x = self.eca(x)
        return x

def get_activation(activation_type):
    """根据字符串获取激活函数"""
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):  # 检查nn模块中是否有该激活函数
        return getattr(nn, activation_type)()  # 动态获取并实例化
    else:
        return nn.ReLU()  # 默认使用ReLU

class CBN(nn.Module):
    """3D卷积 + 批归一化 + 激活函数的组合模块"""
    def __init__(self, in_channels, out_channels,kernelsize, activation='ReLU'):
        super(CBN, self).__init__()

        # 自动计算 padding 以保持空间尺寸
        # 你问我为啥？因为有时kernel_size在某一维度为奇数（一般只可能为1），那么这时就会用padding去填充，导致尺寸发生变化
        # 所以得确保padding在kernel_size为奇数时为0
        if isinstance(kernelsize, int):
            padding = kernelsize // 2
        else:
            padding = tuple(k // 2 for k in kernelsize)
        
        self.conv = convclass(in_channels, out_channels, kernel_size=kernelsize, padding=padding)  # 卷积
        self.norm = NormMethod(out_channels)  # 批归一化
        self.activation = get_activation(activation)  # 激活函数

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

def _make_nConv(in_channels, out_channels, nb_Conv,kernelsize, activation='ReLU'):
    """
    创建包含多个卷积块的序列
    :param nb_conv: 卷积块数量
    """
    layers = []
    # 第一个卷积块
    layers.append(CBN(in_channels, out_channels,kernelsize, activation))

    # 额外的卷积块
    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels,kernelsize, activation))
    return nn.Sequential(*layers)  # 转换为Sequential

class UpBlock_attention(nn.Module):
    """带注意力的上采样块"""
    def __init__(self, x_channels, skip_channels, out_channels, nb_Conv,scale_factor,kernel_size, activation='ReLU'):
        '''
        初始化UpBlock_Attention
        :param x_channels: 输入通道数
        :param skip_channels: 跳跃通道数
        :param out_channels: 输出通道数
        :nb_Conv: 卷积块数量
        :scale_factor: 上采样缩放因子，为2维或3维元组，一般是对应层级的stride参数
        :kernel_size: 卷积核大小
        '''
        super().__init__()
        #上采样暂时不用Upsample，用转置卷积
        #self.up = nn.Upsample(scale_factor=scale_factor, mode=unsample_mode, align_corners=True)  # 根据scale_factor对不同维度上采样
        # 转置卷积：将深层特征上采样到跳跃连接尺寸，输出通道设为 skip_channels（以便拼接）。但是呢是skip_channel好像不对，我改成了x_channels
        self.up = convtransposexd(
            x_channels, x_channels,
            kernel_size=scale_factor,
            stride=scale_factor,
            padding=0,
            output_padding=0
        )
        # 关键修改：使用不同的通道数参数
        self.coatt = CCA(F_g=x_channels, F_x=skip_channels)  # 通道交叉注意力
        # 拼接后的通道数是 x_channels + skip_channels
        self.nConvs = _make_nConv(x_channels + skip_channels, out_channels, nb_Conv,kernel_size, activation)

        #保存下来用于外部查询
        self.x_channels=x_channels
        '''输入通道数——上采样前特征的通道数（来自深层）'''
        self.skip_channels=skip_channels
        '''跳跃通道数——跳跃连接特征的通道数（来自编码器对应层）'''
        self.out_channels=out_channels
        '''输出通道数——该解码器块的输出通道数（传递给更浅层）'''
        self.scale_factor=scale_factor
        '''缩放因子'''
        self.kernel_size = kernel_size
        '''卷积核大小'''

    def forward(self, x, skip_x):
        """前向传播：上采样 + 注意力融合 + 卷积处理"""
            
        up = self.up(x)  # 上采样

        # 原有的使用CCA进行通道维度重标定，但我要进行CCA消融实验
        skip_x_att = self.coatt(g=up, x=skip_x)  # 应用通道注意力到skip connection  
        cat = torch.cat([skip_x_att, up], dim=1)  # 沿通道维度拼接

        # cat=torch.cat([skip_x,up],dim=1)#最简单的跳跃连接拼接方式

        return self.nConvs(cat)  # 卷积处理
    
class Flatten(nn.Module):
    """展平层：将特征图展平为向量"""
    def forward(self, x):
        return x.view(x.size(0), -1)  # [B, C, H, W] -> [B, C*H*W]

class CCA(nn.Module):
    """通道交叉注意力模块：用于特征融合"""
    def __init__(self, F_g, F_x):
        super().__init__()
        
        self.IsDebug=False #是否输出调试信息
        if self.IsDebug:
            print(f"\nCCA初始化调试:")
            print(f"  - F_g: {F_g}")
            print(f"  - F_x: {F_x}")
        
        self.F_g = F_g  # 保存原始通道数
        self.F_x = F_x  # 保存原始通道数
        
        # 编码输入特征x的MLP
        self.mlp_x = nn.Sequential(
            Flatten(),  # 展平
            nn.Linear(F_x, F_x)  # 全连接层
        )
        # 编码引导特征g的MLP  
        self.mlp_g = nn.Sequential(
            Flatten(),  # 展平
            nn.Linear(F_g, F_x)  # 全连接层（输出维度与x相同）
        )
        self.relu = nn.ReLU(inplace=True)  # ReLU激活

    ###TODO:这里的avg_poolxd的参数也要改！Finished
    def forward(self, g, x):
        """前向传播：计算通道注意力"""

        # 对x进行全局平均池化并编码
        avg_pool_x = avg_poolxd(x)
        channel_att_x = self.mlp_x(avg_pool_x)
        
        # 对g进行全局平均池化并编码
        avg_pool_g = avg_poolxd(g)
        channel_att_g = self.mlp_g(avg_pool_g)
        
        # 计算平均通道注意力
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        
        # Sigmoid生成注意力权重并扩展到与x相同的形状
        scale = torch.sigmoid(channel_att_sum)
        for i in range(Dimension):#根据维度数确定unsqueeze的次数
            scale=scale.unsqueeze(i+2)
        scale=scale.expand_as(x)

        # 应用注意力权重
        x_after_channel = x * scale
        out = self.relu(x_after_channel)  # ReLU激活

        return out

class Res_block(nn.Module):
    """残差块：包含两个卷积层的残差连接"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()

        self.IsDebug=False

        # 第一个卷积层
        self.conv1 = convclass(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = NormMethod(out_channels)  # 归一化
        self.relu = nn.LeakyReLU(inplace=True)   # LeakyReLU激活
        
        # 第二个卷积层
        self.conv2 = convclass(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = NormMethod(out_channels)  # 归一化
        
        # 快捷连接：当输入输出维度不匹配时需要投影
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                convclass(in_channels, out_channels, kernel_size=1, stride=stride),  # 1x1卷积投影
                NormMethod(out_channels)
            )
        else:
            self.shortcut = None  # 维度匹配时直接使用恒等映射

    def forward(self, x):
        residual = x  # 保存输入用于残差连接
        
        if self.shortcut is not None:  # 如果需要投影
            residual = self.shortcut(x)
            
        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual  # 残差连接
        out = self.relu(out)  # 激活
        return out

class EncoderBlock(nn.Module):
    """编码器块：池化下采样 + 多个残差块"""
    def __init__(self, in_channels, out_channels, scale_factor, num_blocks=1,pool=max_pool):
        '''
        初始化EncoderBlock。注意，若指定scale_factor，将据其计算kersize
        
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param scale_factor: 下采样因子，元组形式（如(2,2,2)或(2,2)）
        :param num_blocks: 残差块数量
        :param pool: 池化方式，默认使用全局变量max_pool，即2D/3D最大池化
        :param kersize: 池化核大小，元组形式如(2,2,2)或(2,2)
        '''
        super(EncoderBlock, self).__init__()
        
        self.IsDebug=False
        
        # scale_factor 已经是元组形式
        self.scale_factor = scale_factor
        self.in_channels=in_channels#都保存下来方便外部查询
        '''输入通道数'''
        self.out_channels=out_channels
        '''输出通道数'''

        self.kernel_size = scale_factor#池化核列表
        # 计算池化核大小（缩放因子的倒数）
        # 注意：缩放因子小于1表示下采样，所以池化核大小是缩放因子的倒数
        self.downsample = pool(kernel_size=self.kernel_size, stride=scale_factor)

        # 创建残差块序列
        layers = []
        # 第一个残差块（可能改变通道数）
        layers.append(Res_block(in_channels, out_channels))
        # 额外的残差块（保持通道数不变）
        for _ in range(num_blocks - 1):
            layers.append(Res_block(out_channels, out_channels))
        
        self.res_blocks = nn.Sequential(*layers)
        
        if self.IsDebug:
            print(f"EncoderBlock初始化:")
            print(f"  in_channels: {in_channels}")
            print(f"  out_channels: {out_channels}")
            print(f"  scale_factor: {scale_factor}")
            print(f"  pool_kernel: {self.kernel_size if 'pool_kernel' in locals() else 'None'}")
    
    def forward(self, x):
        #为啥要先下采样？
        # 下采样
        x_down = self.downsample(x)        
        # 通过残差块
        return self.res_blocks(x_down)

        # #改成先通过残差块试试
        # output=self.res_blocks(x)
        # return self.downsample(output)
    
class StridedEncoderBlock(nn.Module):
    """
    使用步长卷积进行下采样的编码器块。

    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        stride (tuple): 步长元组，维度与空间维度一致，例如 (2,2,2) 或 (2,2)
        num_blocks (int): 后续残差块的数量，默认1
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, num_blocks=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_blocks = num_blocks

        # 步长卷积（同时完成下采样和通道变换）
        # padding 根据 kernel_size 自动计算以保证输出尺寸正确（通常 kernel=3, padding=1）
        padding = tuple(k // 2 for k in kernel_size)
        self.conv = convclass(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding
        )#卷积
        self.norm = NormMethod(out_channels)#批归一化
        self.act = nn.ReLU(inplace=True)#激活

        # 后续残差块（输入输出通道均为 out_channels）
        blocks = []
        for _ in range(num_blocks):
            blocks.append(Res_block(out_channels, out_channels))
        self.res_blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.res_blocks(x)
        return x

class EncoderBlockList(nn.Module):
    '''
    编码器列表，由一个储存编码器的列表生成SCTransNet所需的参数
    '''
    def __init__(self, Blocks:nn.ModuleList):
        '''
        初始化编码器列表
        
        :param Blocks: 储存编码器的列表 
        '''
        super().__init__()#可别忘了初始化

        self.Blocks=Blocks

    def __len__(self):
        return len(self.Blocks)
    
    def __getitem__(self, key):
        return self.Blocks[key]
    def forward(self,x):
        '''
        前向传播，返回值是xs(含d_x，即xs[-1])
        '''
        xs=[self.Blocks[0](x)]#第一个编码器

        for i in range(1,len(self.Blocks)):
            xs.append(self.Blocks[i](xs[i-1]))#注意d_x位于xs最后一个
        #d_x=self.Blocks[-1](xs[-1])#这个才是d7
        return xs