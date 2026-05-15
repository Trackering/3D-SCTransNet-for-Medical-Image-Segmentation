###用于提供一些模块式方法

import torch
import torch.nn.functional as F  # PyTorch函数式接口
from einops import rearrange #用于张量重塑
import torch.nn as nn

def avg_pool_3d(x):
    '''
    用于切换2D/3D平均池化
    
    :param x: 输入参数
    :return: 返回F.avg_pool3d
    :rtype: list
    '''
    return F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)), 
                                 stride=(x.size(2), x.size(3), x.size(4)))
def avg_pool_2d(x):
    '''
    用于切换2D/3D平均池化
    
    :param x: 输入参数
    :return: 返回F.avg_pool2d
    :rtype: list
    '''
    return F.avg_pool2d(x, (x.size(2), x.size(3)), 
                                 stride=(x.size(2), x.size(3)))

def GetPatchNum(img_size,patch_size):
        '''
        计算图像的patch数量
        
        :param img_size: 图像大小，可为二维或三维
        :return: patch数量
        :rtype: int
        '''
        n_patches = 1
        for img_dim, patch_dim in zip(img_size, patch_size):
            n_patches *= img_dim // patch_dim
        
        return n_patches

def rearrange_2d(x,heads):
     '''
     2D张量重塑
     
     :param x: 输入张量
     :param heads: 注意力头数
     '''
     return rearrange(x, 'b (head c) h w -> b head c (h w)', head=heads)
def rearrange_3d(x,heads):
     '''
     3D张量重塑
     
     :param x: 输入张量
     :param heads: 注意力头数
     '''
     return rearrange(x, 'b (head c) d h w -> b head c (d h w)',head=heads)

def rearrange_out_2d(x,d,h,w):
    '''
    用于Attention_org前向传播中out_的空间重塑为2d
    
    :param x: 说明
    :param d: 占位符，不会被使用
    :param h: 说明
    :param w: 说明
    '''
    return rearrange(x, 'b  c (h w) -> b c h w', h=h, w=w)
def rearrange_out_3d(x,d,h,w):
    '''
    用于Attention_org前向传播中out_的空间重塑为3d
    
    :param x: 说明
    :param d: 说明
    :param h: 说明
    :param w: 说明
    '''
    return rearrange(x, 'b c (d h w) -> b c d h w', d=d,h=h, w=w)

def to_3d_3d(x):
    """将5D张量 [B, C, D, H, W] 重塑为3D张量 [B, D*H*W, C]"""
    return rearrange(x, 'b c d h w -> b (d h w) c')
def to_4d_3d(x, d, h, w):
    """将3D张量 [B, D * H * W, C] 重塑回5D张量 [B, C, D, H, W]"""
    return rearrange(x, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)

def to_3d_2d(x):
    '''
    将4D张量重塑为3D张量
    '''
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d_2d(x,d, h, w):#为了统一调用，因此多了一个冗余参数d
    '''
    将3D张量重塑为4D张量
    '''
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class _safe_pool3d(nn.Module):
    """安全的3D池化，能够处理深度为1的情况"""

    def __init__(self,pool_kernel,stride):
        super().__init__()#一定要执行，否则会报错
        self.pool_kernel=pool_kernel
        self.stride=stride
    
    def forward(self,x):
        _, _, d, h, w = x.shape
        
        if d == 1:
            # 深度为1时，使用2D池化
            # 重塑为2D: [B, C, H, W] -> [B, C, 1, H, W]
            x_2d = x.squeeze(2)  # [B, C, H, W]
            # 2D池化
            x_2d_pooled = F.max_pool2d(x_2d, kernel_size=self.pool_kernel[1:], 
                                        stride=self.stride[1:])
            # 恢复为3D: [B, C, H', W'] -> [B, C, 1, H', W']
            x_pooled = x_2d_pooled.unsqueeze(2)
        else:
            # 正常3D池化
            x_pooled = F.max_pool3d(x, kernel_size=self.pool_kernel, 
                                    stride=self.stride)
        
        return x_pooled

def UpdatePatchSize(config, img_size, scale_factors):
    '''
    根据特征图尺寸调整patch大小，确保整除
    同时确保所有层级经过patch嵌入后的特征图尺寸相同（便于在ChannelTransformer中拼接）
    注意：会自动更新config中的patch_sizes

    :param config: 配置对象
    :param img_size: 原始输入尺寸
    :param scale_factors: 各层级相对于原始输入的缩放因子列表
    :return: 调整后的patch大小列表
    '''
    
    original_patch_sizes = config.patch_sizes.copy()
    patch_sizes = []
    
    # 计算特征图尺寸
    relative_scales = ComputeRelativeScales(scale_factors)
    feature_sizes = ComputeFeatureSizes(img_size, relative_scales)
    
    # 首先，确保所有层级的特征图都能被对应的patch整除
    for i, patch in enumerate(original_patch_sizes):
        if i >= len(feature_sizes):
            break
            
        feature_size = feature_sizes[i]
        img_dim = len(feature_size)
        
        # 调整patch大小不超过特征图各维度
        adjusted_patch = []
        for j in range(img_dim):
            # 对每个维度，取patch和当前维度大小的最小值
            patch_dim = patch if isinstance(patch, int) else patch[j % len(patch)]
            adjusted_dim = min(patch_dim, feature_size[j])
            
            # 确保patch_size能够整除当前维度大小
            while feature_size[j] % adjusted_dim != 0 and adjusted_dim > 1:
                adjusted_dim -= 1
            
            adjusted_patch.append(adjusted_dim)
        
        patch_sizes.append(tuple(adjusted_patch))
    
    # 现在，计算每个层级patch嵌入后的特征图尺寸
    embedded_sizes = []
    for i in range(len(patch_sizes)):
        if i >= len(feature_sizes):
            break
        feature_size = feature_sizes[i]
        patch_size = patch_sizes[i]
        
        # 计算patch嵌入后的尺寸
        embedded_size = tuple(
            feature_size[j] // patch_size[j] for j in range(len(feature_size))
        )
        embedded_sizes.append(embedded_size)
    
    # 找到所有层级中最小的嵌入后尺寸（作为目标尺寸）
    if embedded_sizes:
        target_embedded_size = embedded_sizes[-1]  # 使用最深层的尺寸作为目标
        
        if len(embedded_sizes) > 1:
            print(f"\n调整前各层级patch嵌入后尺寸: {embedded_sizes}")
            print(f"目标统一尺寸: {target_embedded_size}")
        
        # 调整前几层的patch大小，使其嵌入后尺寸与目标一致
        for i in range(len(patch_sizes) - 1):  # 不调整最后一层
            feature_size = feature_sizes[i]
            current_embedded_size = embedded_sizes[i]
            
            # 如果当前嵌入后尺寸与目标不一致，调整patch大小
            if current_embedded_size != target_embedded_size:
                new_patch = []
                for j in range(len(feature_size)):
                    # 计算需要的patch大小
                    required_patch = feature_size[j] // target_embedded_size[j]
                    
                    # 确保patch大小是整数且能整除
                    while feature_size[j] % required_patch != 0 and required_patch < feature_size[j]:
                        required_patch += 1
                    
                    # 确保不超过原始patch大小
                    original_patch = patch_sizes[i][j]
                    if required_patch > original_patch:
                        # 如果不能达到目标尺寸，使用能整除的最大值
                        for candidate in range(original_patch, 0, -1):
                            if feature_size[j] % candidate == 0:
                                required_patch = candidate
                                break
                    
                    new_patch.append(required_patch)
                
                # 更新patch大小
                patch_sizes[i] = tuple(new_patch)
                
                # 重新计算嵌入后尺寸
                new_embedded_size = tuple(
                    feature_sizes[i][j] // patch_sizes[i][j] for j in range(len(feature_size))
                )
                
                print(f"层级 {i}: 调整patch大小从 {original_patch_sizes[i]} 到 {patch_sizes[i]}")
                print(f"        嵌入后尺寸从 {embedded_sizes[i]} 到 {new_embedded_size}")
    
    # 最终验证和打印
    print(f"\n最终patch大小:")
    for i, patch_tuple in enumerate(patch_sizes):
        if i < len(feature_sizes):
            feature_size = feature_sizes[i]
            embedded_size = tuple(
                feature_size[j] // patch_tuple[j] for j in range(len(feature_size))
            )
            print(f"  层级 {i}: patch大小={patch_tuple}, 特征图尺寸={feature_size}, 嵌入后尺寸={embedded_size}")
    
    # 更新config
    config.patch_sizes = patch_sizes
    
    return patch_sizes

def CreatePatchSizes(basepatch,levels):

    '''
    创建Patch_Sizes列表，步长为basepatch
    
    :param basepatch: 步长
    :param levels: patch_sizes级别数
    '''
    patch_sizes=[]
    for i in range(levels):
        patch_sizes.append(basepatch*2**(i))
    patch_sizes.reverse()#让其从大到小排列
    return patch_sizes

def GetBestPatchSizes(features_size,best_size):
    '''
    获得最佳的PatchSizes。我们默认编码器特征图尺寸小于best_size的已经被上采样至best_size

    :param features_size: 编码器特征图尺寸列表
    :parma best_size: 最佳嵌入后尺寸
    '''
    patch_sizes=[]
    for feature_size in features_size:
        if feature_size[0]>=best_size[0]:
            patch_sizes.append(tuple(x // y for x, y in zip(feature_size, best_size)))   # 比best_size大的就整除
        else:
            patch_sizes.append((1,)*len(best_size))#小的就默认已经上采样至bestsize
    return patch_sizes

def GetScaleFactor3d(scale_factor):
    '''
    处理 scale_factor为3维数据
    
    :param scale_factor: 说明
    '''
    if isinstance(scale_factor, (int, float)):
        # 1.如果是数字，转换为3D元组
        return (float(scale_factor), float(scale_factor), float(scale_factor))
    elif isinstance(scale_factor, tuple):
        if len(scale_factor) == 2:
            # 2.如果是2D元组，添加深度维度（设为1，表示不改变深度）
            return (1.0, float(scale_factor[0]), float(scale_factor[1]))
        elif len(scale_factor) == 3:
            # 3.如果是3D元组，直接使用
            return tuple(float(s) for s in scale_factor)
        else:
            raise ValueError(f"scale_factor 必须是int, float, 或者长度为2或3的元组, 但实际上是 {scale_factor}")
    else:
        raise TypeError(f"scale_factor 必须是int, float, 或元组, 但实际上是 {type(scale_factor)}")

def GetScaleFactor2d(scale_factor):
    '''
    处理 scale_factor为2维数据
    
    :param scale_factor: 说明
    '''
    if isinstance(scale_factor, (int, float)):
        # 1.如果是数字，转换为3D元组
        return (float(scale_factor), float(scale_factor), float(scale_factor))
    elif isinstance(scale_factor, tuple):
        if len(scale_factor) == 2:
            # 2.如果是2D元组，添加深度维度（设为1，表示不改变深度）
            return (1.0, float(scale_factor[0]), float(scale_factor[1]))
        else:
            raise ValueError(f"scale_factor 必须是int, float, 或者长度为2的元组, 但实际上是 {scale_factor}")
    else:raise TypeError(f"scale_factor 必须是int, float, 或元组, 但实际上是 {type(scale_factor)}")

def GetScaleFactorForDecoder(scale_factors):
    '''
    根据下采样缩放因子列表确定下采样缩放因子列表
    注意：这个方法没有实际价值，因为我们传递给解码器时，其实应该是相邻的缩放倍数，而这个确定的是相对于原图的绝对倍数
    :param scale_factors: 下采样缩放因子列表
    '''
    dscales=[]
    for factor in scale_factors:
        if isinstance(factor, (int, float)):
            # 如果是标量，直接取倒数
            result = int(1.0 / float(factor))
        elif isinstance(factor, tuple):
            # 如果是元组，对每个元素取倒数
            result = tuple(int(1.0 / s) for s in factor)
        else:
            raise ValueError(f"因子类型不支持: {type(factor)}")
        dscales.append(result)
    
    dscales.reverse()  # 反转列表，使其从深到浅排列
    return dscales

def ComputeRelativeScales(scale_factors):
    '''
    根据下采样因子列表计算每个层级相对于原始输入的缩放
    注意：scale_factors 已经包含了每个层级相对于原始输入的缩放
    例如：[(1,0.5,0.5), (0.5,0.25,0.25), ...]
    
    :param scale_factors: 各层级相对于原始输入的缩放因子列表
    :return: 相对缩放列表（包含原始尺寸）
    '''
    # 第一个是原始尺寸 (1,1,1) 或 (1,1)
    if len(scale_factors[0]) == 2:
        relative_scales = [(1.0, 1.0)]
    else:
        relative_scales = [(1.0, 1.0, 1.0)]
    
    # 直接使用传入的缩放因子
    relative_scales.extend(scale_factors)
    
    return relative_scales

def ComputeDownsampleFactors(relative_scales):
    '''
    根据相对缩放因子列表计算相邻层级之间的下采样因子
    
    :param relative_scales: 相对缩放因子列表
    :return: 下采样因子列表
    '''
    downsample_factors = []
    
    for i in range(1, len(relative_scales)):
        prev_scale = relative_scales[i-1]
        curr_scale = relative_scales[i]
        
        # 计算相邻层级之间的缩放因子（逐元素除法）
        if len(prev_scale) == 2:
            factor = (curr_scale[0] / prev_scale[0],
                     curr_scale[1] / prev_scale[1])
        else:
            factor = (curr_scale[0] / prev_scale[0],
                     curr_scale[1] / prev_scale[1],
                     curr_scale[2] / prev_scale[2])
        
        downsample_factors.append(factor)
    
    return downsample_factors

def ComputeUpsampleFactors(downsample_factors):
    '''
    根据下采样因子列表计算上采样因子（反转并取倒数）
    
    :param downsample_factors: 下采样因子列表
    :return: 上采样因子列表
    '''
    upsample_factors = []
    
    # 反转下采样因子列表
    reversed_factors = list(reversed(downsample_factors))
    
    for factor in reversed_factors:
        # 计算上采样因子（逐元素取倒数）
        if len(factor) == 2:
            # 确保是整数
            upsample_factor = (1.0 / factor[0], 1.0 / factor[1])
        else:
            upsample_factor = (1.0 / factor[0], 1.0 / factor[1], 1.0 / factor[2])
        upsample_factors.append(upsample_factor)

    return upsample_factors

def ComputeFeatureSizes(img_size, relative_scales):
    '''
    根据原始输入尺寸和相对缩放因子计算每个层级的特征图尺寸
    
    :param img_size: 原始输入尺寸
    :param relative_scales: 相对缩放因子列表
    :return: 特征图尺寸列表
    '''
    feature_sizes = []
    img_dim = len(img_size)
    
    for scale in relative_scales:
        if img_dim == 2:
            h = int(img_size[0] * scale[0])
            w = int(img_size[1] * scale[1])
            feature_sizes.append((h, w))
        else:
            d = int(img_size[0] * scale[0])
            h = int(img_size[1] * scale[1])
            w = int(img_size[2] * scale[2])
            feature_sizes.append((d, h, w))
    
    return feature_sizes

def ComputeReconstructFactors(patch_sizes, feature_sizes):
    '''
    根据 patch 大小和特征图尺寸计算 Reconstruct 层的上采样因子
    
    :param patch_sizes: patch 大小列表（元组形式）
    :param feature_sizes: 特征图尺寸列表（元组形式）
    :return: Reconstruct 上采样因子列表（整数元组形式）
    '''
    reconstruct_factors = []
    for i, patch_size in enumerate(patch_sizes):
        feature_size = feature_sizes[i]
        
        # 确保 patch_size 是元组格式
        if isinstance(patch_size, int):
            patch_size = tuple([patch_size] * len(feature_size))
        
        # 计算每个维度的缩放因子，并转换为整数
        if len(feature_size) == 2:
            h_patched = feature_size[0] // patch_size[0]
            w_patched = feature_size[1] // patch_size[1]
            
            # 确保除数不为零，并转换为整数
            h_scale = int(feature_size[0] / max(h_patched, 1))
            w_scale = int(feature_size[1] / max(w_patched, 1))
            reconstruct_factors.append((h_scale, w_scale))
        else:
            d_patched = feature_size[0] // patch_size[0]
            h_patched = feature_size[1] // patch_size[1]
            w_patched = feature_size[2] // patch_size[2]
            
            # 确保除数不为零，并转换为整数
            d_scale = int(feature_size[0] / max(d_patched, 1))
            h_scale = int(feature_size[1] / max(h_patched, 1))
            w_scale = int(feature_size[2] / max(w_patched, 1))
            reconstruct_factors.append((d_scale, h_scale, w_scale))
    
    return reconstruct_factors

def UnZipShape(shape):
    '''
    将2d/3d形状统一转换为[B,C,D,H,W]的表示。当输入为2D时，深度置为-1。

    用于满足2d算法模块要求的D维度占位符

    :param shape: 要处理的形状
    '''
    dim=len(shape)
    if dim==4:#二维图像有4个信息
        #2d:b,c,h,w,#3d:b,c,d,h,w
        return [shape[0],shape[1],-1,shape[2],shape[3]]
    if dim==5:
        return shape
    
def GetEncoderChannelList(in_channels,num_decoder):
    '''
    获得最简单的编码器通道数列表
    '''
    return [in_channels * (2**i) for i in range(num_decoder)]

def GetBestEmbeddSize(features_size,minsize=1):
    """
    获取最佳的嵌入后图像尺寸，用于ChannelTransformer Patch嵌入操作
    返回最佳嵌入后尺寸和该尺寸位于的特征图列表中的索引位置

    用于从候选特征图尺寸列表 features_size 中选出最佳尺寸 bestsize。
    遍历列表，找到第一个在所有维度上都大于等于 minsize 的最小特征图尺寸。
    如果遍历完所有尺寸都没有满足条件的（即所有特征图都小于 minsize），
    则检查最大尺寸（即 features_size[0]）是否至少有一个维度小于 minsize，
    若成立则进入异常分支：当列表长度大于等于2时，选取倒数第二个尺寸（次大的）作为 bestsize，否则选取第一个尺寸（唯一的）。
    最终返回的 bestsize 将用于后续网络配置，确保尺寸不低于 minsize 或尽可能不太小。

    :param features_size: 特征图尺寸列表，按空间分辨率从小到大排列。
    :param minsize: 允许的最小尺寸，小于此尺寸，将会返回(minsize,)*len(features_size)
    """
    dim = len(features_size[0])          # 图像维度
    bestsize = (minsize,) * dim           # 默认值

    index=0
    for i in range(len(features_size)):
        size=features_size[i]
        if all(s >= minsize for s in size[2:]):#该特征图尺寸每一个维度都得大于minsize。用[2:]是为了排除前面的通道数啥的
            bestsize = size
            index=i
        else:
            break
    # 如果没找到，bestsize 保持为 (minsize,)*dim
    #以上代码实现了找到大于等于minsize的最小特征图尺寸

    if any(s<minsize for s in features_size[0][2:]):#要是最大特征图都比minsize小，我们就只能挑比较大的了
        if len(features_size)>=2:#如果特征图列表多，我们要倒数第2个
            bestsize= features_size[-2]
            index=len(features_size)-2
        else:
            bestsize= features_size[0]#否则就要尺寸最大的
            index=0

    return bestsize,index

def _getshapes(features:list[torch.Tensor]):
    '''
    获取指定列表中的图像尺寸。

    我受够在VS Code里面挨个点击查看了！
    '''
    shapeList:list[torch.Size]=[]
    for x in features:
        shapeList.append(x.shape)
    return shapeList
