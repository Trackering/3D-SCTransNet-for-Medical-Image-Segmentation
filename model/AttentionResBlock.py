import torch.nn as nn     # PyTorch神经网络模块
import torch              # PyTorch深度学习框架
from . import Blocks as bs
import torch.nn.functional as F  # PyTorch函数式接口

class AttentionResBlock(nn.Module):
    """带注意力的多尺度空洞卷积残差块"""
    def __init__(self, in_channels, out_channels, 
                 dilation_rates=[1, 2, 4, 8], 
                 reduction_ratio=16):
        '''
        初始化注意力多尺度残差块
        
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param dilation_rates: 空洞率列表，控制多尺度感受野
        :param reduction_ratio: 通道注意力中的降维比例
        :param use_spatial_attention: 是否使用空间注意力
        :param use_channel_attention: 是否使用通道注意力
        '''
        super(AttentionResBlock, self).__init__()
        
        self.IsDebug = False
        self.dilation_rates = dilation_rates
        self.num_branches = len(dilation_rates)
        
        if self.IsDebug:
            print(f"\nAttentionResBlock 初始化:")
            print(f"  in_channels: {in_channels}")
            print(f"  out_channels: {out_channels}")
            print(f"  dilation_rates: {dilation_rates}")
            print(f"  num_branches: {self.num_branches}")
        
        # 1. 创建多尺度卷积分支
        self.branches = nn.ModuleList()
        branch_out_channels = out_channels // self.num_branches
        
        for i, dilation in enumerate(dilation_rates):
            # 计算padding以保持空间尺寸不变
            padding = dilation if dilation > 1 else 1
            
            branch = nn.Sequential(
                bs.convclass(in_channels, branch_out_channels, 
                         kernel_size=3, padding=padding, dilation=dilation),
                bs.batch_norm(branch_out_channels),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
        
        # 2. 通道注意力模块
        self.channel_attention = self._build_channel_attention(
            out_channels, reduction_ratio
        )
        
        # 3. 空间注意力模块
        self.spatial_attention = self._build_spatial_attention(out_channels)
        
        # 4. 特征融合卷积（用于调整通道数和融合特征）
        self.fusion_conv = nn.Sequential(
            bs.convclass(out_channels, out_channels, kernel_size=1),
            bs.batch_norm(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 5. 残差连接
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                bs.convclass(in_channels, out_channels, kernel_size=1),
                bs.batch_norm(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        # 池化模块
        self.pool=bs.adaptiveAvgPool(1)
    
    def _build_channel_attention(self, channels, reduction_ratio):
        """构建通道注意力模块"""
        return nn.Sequential(
            # # 全局平均池化
            # bs.adaptiveAvgPool(1),
            # # 两层全连接，中间有降维
            # nn.Conv1d(channels, channels // reduction_ratio, kernel_size=1),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(channels // reduction_ratio, channels, kernel_size=1),
            # nn.Sigmoid()
            
            # 全局平均池化已经在forward中调用，这里只需要后面的处理
            nn.Flatten(),  # 将[B, C, 1, 1, 1]展平为[B, C]
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )
    
    def _build_spatial_attention(self, channels):
        """构建空间注意力模块"""
        # 使用7x7卷积捕获较大的空间上下文
        kernel_size = 7
        padding = kernel_size // 2
        
        return nn.Sequential(
            bs.convclass(channels, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        前向传播
        
        流程：
        1. 并行多尺度特征提取
        2. 通道维度拼接
        3. 应用通道注意力（可选）
        4. 应用空间注意力（可选）
        5. 特征融合
        6. 残差连接
        """
        residual = self.shortcut(x)
        
        # 1. 并行多尺度特征提取
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        # 2. 沿通道维度拼接
        concat_features = torch.cat(branch_outputs, dim=1)
        
        # 3. 应用通道注意力

        # 获取通道注意力权重
        b, c, *spatial_dims = concat_features.shape
        
        # 全局平均池化获取通道统计
        channel_pooled = self.pool(concat_features)
        
        # 重塑为 [B, C, 1] 以适应1D卷积
        channel_pooled = channel_pooled.view(b, c, 1)
        
        # 计算通道注意力权重
        channel_weights = self.channel_attention(channel_pooled)
        
        # 重塑回原始空间维度
        for _ in range(bs.Dimension):
            channel_weights = channel_weights.unsqueeze(-1)
        
        # 应用通道注意力
        concat_features = concat_features * channel_weights.expand_as(concat_features)
        
        # 4. 应用空间注意力

        # 计算空间注意力权重
        spatial_weights = self.spatial_attention(concat_features)
        
        # 应用空间注意力
        concat_features = concat_features * spatial_weights
        
        # 5. 特征融合
        fused_features = self.fusion_conv(concat_features)
        
        # 6. 残差连接
        output = fused_features + residual
        output = nn.ReLU(inplace=True)(output)
        
        return output


class SEAttention(nn.Module):
    """Squeeze-and-Excitation注意力模块（可选）"""
    def __init__(self, channel, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = bs.adaptiveAvgPool(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, *spatial_dims = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        
        # 重塑为 [B, C, 1, 1, (1)]
        for _ in range(bs.Dimension):
            y = y.unsqueeze(-1)
        
        return x * y.expand_as(x)

class PyramidMultiScaleEncoder(nn.Module):
    """金字塔式多尺度编码器：使用AttentionResBlock进行多尺度特征提取"""
    def __init__(self, in_channels, out_channels, scale_factor,input_size,
                 pyramid_levels=4):
        '''
        初始化金字塔式多尺度编码器
        
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param scale_factor: 下采样因子
        :param pyramid_levels: 金字塔层数（多尺度分支数）
        :param input_size: 输入的特征图尺寸
        '''
        super(PyramidMultiScaleEncoder, self).__init__()
        
        self.IsDebug = False
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.pyramid_levels = pyramid_levels
        self.input_size = input_size  # 保存输入尺寸

        if self.IsDebug:
            print(f"\nPyramidMultiScaleEncoder 初始化:")
            print(f"  in_channels: {in_channels}")
            print(f"  out_channels: {out_channels}")
            print(f"  scale_factor: {scale_factor}")
            print(f"  pyramid_levels: {pyramid_levels}")
        
        # 1. 多尺度注意力残差块（核心特征提取）
        self.multi_scale_block = self._build_multi_scale_block(
            in_channels, out_channels
        )
        
         # 2. 金字塔池化模块 - 基于输入尺寸动态计算
        self.pyramid_pooling, self.pyramid_branch_channels = self._build_pyramid_pooling(out_channels)
        
        # 3. 特征融合卷积 - 根据实际分支数和通道数动态计算
        # 计算总输入通道数 = 原始特征通道数 + 所有池化分支通道数之和
        total_input_channels = out_channels + self.pyramid_branch_channels * len(self.pyramid_pooling)
        self.fusion_conv = nn.Sequential(
            bs.convclass(total_input_channels, out_channels, kernel_size=1),
            bs.batch_norm(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 4. 下采样层
        self.downsample = self._create_downsample(scale_factor)
        
        # 5. 残差连接（如果需要）
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                bs.convclass(in_channels, out_channels, kernel_size=1),
                bs.batch_norm(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def _build_multi_scale_block(self, in_channels, out_channels):
        """构建多尺度注意力残差块"""
        # 根据金字塔层数确定空洞率
        dilation_rates = []
        for i in range(self.pyramid_levels):
            dilation_rates.append(2 ** i)  # 1, 2, 4, 8, ...
        
        return AttentionResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            dilation_rates=dilation_rates
        )
    
    def _build_pyramid_pooling(self, channels):
        """构建金字塔池化模块 - 返回模块列表和每个分支的输出通道数"""
        if self.input_size is None:
            return nn.ModuleList(), 0
        
        # 确定维度和池化函数
        ndim = bs.Dimension
        dim_sizes = self.input_size
        pool_func = bs.adaptiveAvgPool
        
        # 计算候选池化大小
        candidates = [2, 3, 4, 6]
        pool_sizes = []
        
        # 筛选合适的池化大小
        for scale in candidates:
            if all(dim >= scale for dim in dim_sizes):
                pool_sizes.append(tuple([scale] * ndim))
        
        # 如果特征图太小，使用特征图尺寸的一半
        if not pool_sizes:
            scaled_sizes = tuple(max(2, dim // 2) for dim in dim_sizes)
            pool_sizes.append(scaled_sizes)
        
        # 限制最多使用3个池化分支
        pool_sizes = pool_sizes[:3]
        
        # 计算每个分支的输出通道数
        num_branches = len(pool_sizes)
        branch_channels = max(1, channels // num_branches) if num_branches > 0 else 0
        
        # 创建池化层
        pyramid_pooling = nn.ModuleList([
            nn.Sequential(
                pool_func(pool_size),
                bs.convclass(channels, branch_channels, kernel_size=1),
                bs.batch_norm(branch_channels),
                nn.ReLU(inplace=True)
            )
            for pool_size in pool_sizes
        ])
        
        return pyramid_pooling, branch_channels
    
    def _create_downsample(self, scale_factor):
        """创建下采样层"""
        # 检查是否需要下采样（缩放因子小于1）
        if isinstance(scale_factor, (int, float)):
            if scale_factor < 1.0:
                # 计算池化核大小
                kernel_size = int(1.0 / scale_factor)
                return bs.max_pool(kernel_size=kernel_size, stride=kernel_size)
            else:
                return nn.Identity()
        elif isinstance(scale_factor, tuple):
            # 元组形式的缩放因子
            if any(f < 1.0 for f in scale_factor):
                # 计算池化核大小
                kernel_size = []
                for f in scale_factor:
                    if f < 1.0 and f > 0:
                        kernel_size.append(int(1.0 / f))
                    else:
                        kernel_size.append(1)
                
                kernel_size = tuple(kernel_size)
                stride = kernel_size
                return bs.max_pool(kernel_size=kernel_size, stride=stride)
            else:
                return nn.Identity()
        else:
            return nn.Identity()
    
    def forward(self, x):
        """
        前向传播
        
        流程：
        1. 多尺度特征提取
        2. 金字塔池化（可选）
        3. 特征融合
        4. 残差连接
        5. 下采样
        """
        residual = self.shortcut(x)
        
        # 1. 多尺度特征提取
        multi_scale_features = self.multi_scale_block(x)
        
        # 2. 金字塔池化
        pyramid_features = [multi_scale_features]
        
        for pooling in self.pyramid_pooling:
            pooled = pooling(multi_scale_features)
            
            # 上采样到原始尺寸
            pooled = F.interpolate(pooled, size=multi_scale_features.shape[2:], 
                                    mode=bs.unsample_mode, align_corners=True)
            
            pyramid_features.append(pooled)
        
        # 拼接所有金字塔特征
        all_features = torch.cat(pyramid_features, dim=1)
        
        # 3. 特征融合
        fused_features = self.fusion_conv(all_features)
        
        # 4. 残差连接
        output = fused_features + residual
        output = nn.ReLU(inplace=True)(output)
        
        # 5. 下采样
        output = self.downsample(output)
        
        return output
