import ml_collections     # 配置管理库，用于管理模型配置
from . import Utilities as uti
class ConfigManager:
    '''
    配置管理器
    '''
    def __init__(self,path:str):
        self.config=ml_collections.ConfigDict()# 创建配置字典

    @staticmethod
    def GetConfig(configmanager):
        """
        获取网络的配置参数
        :param configman: nnU-Net推荐的该数据集配置信息,JSON格式
        """
        configman=ml_collections.ConfigDict(configmanager)
        archargs=configman.architecture.arch_kwargs#nnU-Net推荐的模型参数
        config = ml_collections.ConfigDict()# 创建配置字典
        config.transformer = ml_collections.ConfigDict()  # 创建transformer子配置
        config.archargs=archargs#储存模型推荐参数
        config.n_stages=archargs.n_stages   #含输入层的编码器数量
        config.features_per_stage=archargs.features_per_stage#每一个编码器的输出特征通道数
        config.kernel_sizes=archargs.kernel_sizes#卷积核大小列表
        config.strides=archargs.strides#Stride参数列表
        config.n_conv_per_stage_decoder=archargs.n_conv_per_stage_decoder#解码器卷积次数

        if 'n_conv_per_stage' in archargs:
            config.n_conv_per_stage = archargs['n_conv_per_stage']#编码器卷积次数
        else:
            config.n_conv_per_stage = archargs.n_blocks_per_stage*2  # 当使用ResEnc模型时
            config.n_blocks_per_stage=archargs.n_blocks_per_stage
        
        config.img_size=configman.patch_size#图像尺寸

        config.KV_size = 240  # Key-Value的大小，等于Q1+Q2+Q3+Q4+...的通道数之和
        config.transformer.num_heads = 1  # SSCA注意力头数
        config.transformer.num_layers = 4  # SCTB层数量，一般等于解码器数量

        config.num_decoder=config.n_stages-1 #解码器数量
        config.basepatch=2#最小的patch，而patch_sizes将基于此生成。最新版本不用了，因为我们根据指定的嵌入后尺寸自动计算
        config.patch_sizes=uti.CreatePatchSizes(config.basepatch,config.num_decoder)#在最开始时为int列表，初始化后会根据输入图像的维度来确定是二/三维元组列表
        config.base_channel = archargs.features_per_stage[0]  # U-Net的基础通道数
        config.n_classes = 1      # 输出类别数
        config.batchnorm=archargs.norm_op#批归一化方式
        config.num_SCTB=config.n_stages#实际上传给SCTB的特征图个数，这个会根据嵌入后尺寸自动调整，仅用于查询

        # 多尺度编码器配置
        config.pyramid_levels = 4  # 金字塔层数
        config.dilation_sets = [  # 各层级的空洞率设置
            [1, 2, 3],           # 浅层：小空洞率
            [1, 2, 4, 8],        # 中层：中等空洞率
            [1, 3, 6, 12],       # 中深层：较大空洞率
            [1, 3, 6, 12, 18]    # 深层：大空洞率
        ]

        # ********** 当前版本未使用的参数 **********
        config.transformer.embeddings_dropout_rate = 0.1  # embedding层的dropout率
        config.transformer.attention_dropout_rate = 0.1   # 注意力层的dropout率
        config.transformer.dropout_rate = 0               # 总体dropout率
        return config