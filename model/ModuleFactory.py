from typing import Tuple,List,Optional
import torch.nn as nn
from . import Blocks as bs
from . import AttentionResBlock as arb

class ModuleBuilder:
    '''
    模块构建器，用于动态创建网络中的各种组件
    '''
    @staticmethod
    def Get_attn_norms(num,channel_num):
        '''
        获取注意力归一化层列表
        :param num: 需要创建的归一化层数量
        :param channel_num: 各层通道数列表
        :return: 注意力归一化层模块列表
        '''
        attn_norms=nn.ModuleList()
        for i in range(num):
            attn_norms.append(bs.LayerNorm3d(channel_num[i], LayerNorm_type='WithBias'))
        return attn_norms
    
    @staticmethod
    def Get_attn_norm(KV_size):
        '''
        获取单个注意力归一化层
        :param KV_size: Key-Value对的尺寸大小
        :return: 注意力归一化层
        '''
        return bs.LayerNorm3d(KV_size, LayerNorm_type='WithBias')
    
    @staticmethod
    def Get_mheads(num,channel_num,num_attention_heads):
        '''
        获取多头注意力头列表
        :param num: 注意力头数量
        :param channel_num: 各层通道数列表
        :param num_attention_heads: 每个头的注意力头数
        '''
        mheads=nn.ModuleList()
        for i in range(num):
            mheads.append(bs.convclass(channel_num[i], channel_num[i] * num_attention_heads, kernel_size=1, bias=False))
        return mheads
    
    @staticmethod
    def Get_qs(num,channel_num,num_attention_heads):
        '''
        获取查询向量生成器列表
        :param num: 查询向量生成器数量
        :param channel_num: 各层通道数列表
        :param num_attention_heads: 注意力头数
        :return: 查询向量生成器模块列表
        '''
        qs=nn.ModuleList()
        for i in range(num):
            qs.append(bs.convclass(channel_num[i] * num_attention_heads, channel_num[i] * num_attention_heads,
                                kernel_size=3, stride=1,padding=1,
                                groups=channel_num[i] * num_attention_heads // 2, bias=False))
        return qs
    
    @staticmethod
    def Get_project_outs(num,channel_num):
        '''
        获取投影输出层列表
        :param num: 投影层数量
        :param channel_num: 各层通道数列表
        :return: 投影输出层模块列表
        '''
        project_outs=nn.ModuleList()
        for i in range(num):
            project_outs.append(bs.convclass(channel_num[i], channel_num[i], kernel_size=1, bias=False))
        return project_outs
    
    @staticmethod
    def Get_ffn_norms(num,channel_num):
        '''
        获取前馈网络归一化层列表
        :param num: 归一化层数量
        :param channel_num: 各层通道数列表
        '''
        ffn_norms=nn.ModuleList()
        for i in range(num):
            ffn_norms.append(bs.LayerNorm3d(channel_num[i], LayerNorm_type='WithBias'))
        return ffn_norms
    
    @staticmethod
    def Get_ffns(num,channel_num):
        '''
        获取前馈网络模块列表
        :param num: 前馈网络模块数量
        :param channel_num: 各层通道数列表
        :return: 前馈网络模块列表
        '''
        ffns=nn.ModuleList()
        for i in range(num):
            ffns.append(bs.FeedForward(channel_num[i], ffn_expansion_factor=2.66, bias=False))
        return ffns
    
    @staticmethod
    def Get_embeddings(num,config,patchSize,feature_sizes,channel_num,target_embedd_size):
        '''
        获取嵌入层模块列表
        
        :param num: 嵌入层数量
        :param config: 配置参数对象
        :param patchSize: 各层patch大小列表
        :param feature_sizes: 特征图尺寸列表
        :param channel_num: 各层通道数列表
        :param target_embedd_size: 嵌入后的图像尺寸
        :return: 嵌入层模块列表
        '''
            
        embeddings=nn.ModuleList()
        for i in range(num):
            embeddings.append(bs.Channel_Embeddings(config, patchSize[i],#注意patchsize是2/3维元组列表
                                                img_size=feature_sizes[i], in_channels=channel_num[i],target_embedd_size=target_embedd_size))
        return embeddings   
    
    @staticmethod
    def Get_reconstructs(num,channel_num,features_size):
        '''
        获取重建层模块列表
        :param num: 重建层数量
        :param channel_num: 各层通道数列表
        :param features_size: 特征图尺寸列表，注意，只需要参与SCTB的特征图尺寸列表
        :return: 重建层模块列表
        '''
        recontructs=nn.ModuleList()
        for i in range(num):
            recontructs.append(bs.Reconstruct(channel_num[i], channel_num[i], kernel_size=1, 
                                        target_size=features_size[i]))
        return recontructs
    
    @staticmethod
    def Get_encoder_norms(num,channel_num):
        '''
        获取编码器归一化层列表
        :param num: 归一化层数量
        :param channel_num: 各层通道数列表
        :return: 编码器归一化层模块列表
        '''
        encoder_norms=nn.ModuleList()
        for i in range(num):
            encoder_norms.append(bs.LayerNorm3d(channel_num[i], LayerNorm_type='WithBias'))
        return encoder_norms
    
    @staticmethod
    def Get_make_layer(block, input_channels, output_channels, num_blocks=1):
        """构建包含多个残差块的层"""
        layers = []
        # 第一个块（可能改变通道数）
        layers.append(block(input_channels, output_channels))
        # 额外的块（保持通道数不变）
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)  # 转换为Sequential
    
    @staticmethod
    def Get_up_decoders(num_encoders,scale_factors,encoder_channels,kernel_sizes=None):
        '''
        获取上采样解码器模块列表

        规则：
        1. 解码器数量 = 编码器数量（num_encoders）
        2. 第i个解码器（从深到浅，i=0最深）：
           - x_channels: i==0时为编码器最底层输出，否则为上一个解码器的输出
           - skip_channels: 编码器层[-(i+2)]的输出
           - out_channels: 除最后一个解码器外，均为skip_channels的一半
        
        注意：编码器通道序列为 [in_channels, in_channels*2, ..., in_channels*(2^num_encoders)]

        :param num: 解码器数量
        :param scale_factors: 缩放尺寸列表，注意必须是上采样尺度列表
        :param encoder_channels: 编码器特征通道数列表
        '''
        # 生成编码器通道序列
        ###TODO:这里最好通过外部传参来确定，这样最准确
        # 生成编码器通道序列
        #encoder_channels = [in_channels * (2**i) for i in range(num_encoders + 1)]
        # 示例：in_channels=16, num_encoders=6
        # encoder_channels = [16, 32, 64, 128, 256, 512, 1024]

        if(kernel_sizes==None):#要是不指定，就自己创建
            kernel_sizes=[]
            for i in range(1,num_encoders):
                kernel_sizes.append(3)

        decoder_params = []
        
        # 第一个解码器（最深）
        x_channels = encoder_channels[-1]  # 1024
        skip_channels = encoder_channels[-2]  # 512
        out_channels = skip_channels // 2  # 256
        
        decoder_params.append((x_channels, skip_channels, out_channels))
        
        # 后续解码器
        for i in range(1, num_encoders):
            x_channels = out_channels  # 上一个解码器的输出
            
            # 跳跃连接层索引
            skip_idx = -(i + 2)
            skip_channels = encoder_channels[skip_idx]
            
            # 输出通道：最后一个解码器保持相同，其他减半
            if i == num_encoders - 1:  # 最后一个解码器（最浅）
                out_channels = skip_channels
            else:
                out_channels = skip_channels // 2
            
            decoder_params.append((x_channels, skip_channels, out_channels))
        
        up_decoders=nn.ModuleList()
        for i in range(len(decoder_params)):
            up_decoders.append(bs.UpBlock_attention(decoder_params[i][0],decoder_params[i][1],
                                                    decoder_params[i][2],nb_Conv=2,scale_factor=scale_factors[i],kernel_size=kernel_sizes[i]))
                
        return up_decoders
    
    @staticmethod
    def Get_down_encoders(num,in_channels,downsample_factors):
        '''
        生成编码器列表
        
        :param num: 数量
        :param in_channels: 第一层的输入通道数
        :param downsample_factors: 下采样因子
        '''
        inc = ModuleBuilder.Get_make_layer(bs.Res_block, in_channels, in_channels*2)  # 输入卷积，注意因为我们没必要在输入层缩放尺寸，所以使用make_layer
        encoders = nn.ModuleList([inc])#创建编码器列表

        #依次增加通道数
        for i in range(num):
            # 修正：正确的通道数计算
            input_channels = in_channels * (2 ** i)  # 16, 32, 64, 128, 256, ...
            output_channels = in_channels * (2 ** (i+1))  # 32, 64, 128, 256, 512, ...
            encoders.append(bs.EncoderBlock(input_channels,output_channels,downsample_factors[i],1))
        return bs.EncoderBlockList(encoders)
    
    @staticmethod
    def Get_up_decoders_from_config(features_per_stage, strides, n_conv_per_stage_decoder, kernel_sizes=None):
        """
        根据 nnU‑Net 配置生成解码器模块列表（使用 UpBlock_attention）。

        参数:
            features_per_stage (list): 每个阶段的输出通道数，长度 = n_stages
            strides (list): 每个阶段的步长（下采样因子），长度 = n_stages
            n_conv_per_stage_decoder (list): 解码器每个阶段的卷积层数，长度 = n_stages - 1
            kernel_sizes (list, optional): 每个阶段的卷积核大小，长度 = n_stages。
                                        如果为 None，则默认使用 3（或根据维度构造元组）。

        返回:
            nn.ModuleList: 包含从深到浅的解码器块（顺序与编码器对称）
        """
        n_stages = len(features_per_stage)
        # 解码器阶段数 = n_stages - 1
        num_decoders = n_stages - 1

        # 如果未提供 kernel_sizes，则构造默认值（全 3）
        if kernel_sizes is None:
            dim = bs.Dimension
            default_kernel = tuple([3] * dim)
            kernel_sizes = [default_kernel] * n_stages

        # 计算每个解码器块的上采样因子（从对应编码器阶段的 stride 获得）
        # 注意：strides 列表顺序与编码器一致（从浅到深），解码器从深到浅，因此需要反转
        # strides[1:] 是从阶段1开始的步长，反转后对应解码器从深到浅的上采样因子
        upsample_factors = list(reversed(strides[1:]))   # 列表长度 = num_decoders

        # 解码器的输入/输出通道数计算（对称于编码器）
        # 从深到浅，第 i 个解码器的参数：
        #   x_channels: 深层特征通道 = features_per_stage[-1]（最深层）如果是第一个解码器，否则为上一解码器的输出
        #   skip_channels: 跳跃连接通道 = features_per_stage[-(i+2)]  （对称编码器层）
        #   out_channels: 通常等于 skip_channels（PlainConvUNet 风格），也可自定义
        # 这里我们采用 out_channels = skip_channels（与跳跃连接相同）
        decoders = nn.ModuleList()
        for i in range(num_decoders):
            # 深层特征通道（x_channels）
            if i == 0:
                x_ch = features_per_stage[-1]          # 最深层特征
            else:
                x_ch = out_ch     #实际上第一次不会执行这行代码，实际运行不会报错的                      # 上一解码器的输出

            # 跳跃连接通道
            skip_ch = features_per_stage[-(i+2)]        # 从后往前数第 i+2 个（索引从 -1 开始）
            out_ch = skip_ch                             # 输出通道设为与跳跃连接相同

            # 卷积核大小：使用对应编码器阶段的 kernel_sizes（对称）
            # 解码器阶段 i 对应编码器阶段 n_stages-2-i（从浅到深为索引）
            encoder_idx = n_stages - 2 - i
            kernel = kernel_sizes[encoder_idx]

            # 卷积层数
            nb_conv = n_conv_per_stage_decoder[encoder_idx]

            # 上采样因子
            scale_factor = tuple(upsample_factors[i])

            # 构建解码器块
            decoder = bs.UpBlock_attention(
                x_channels=x_ch,
                skip_channels=skip_ch,
                out_channels=out_ch,
                kernel_size=kernel,
                scale_factor=scale_factor,
                nb_Conv=nb_conv
            )
            decoders.append(decoder)

        return decoders
        
    @staticmethod
    def Get_down_encoders_strided(n_stages,features_per_stage,kernel_sizes,strides,n_conv_per_stage,in_channels):
        """
        根据 nnU‑Net 配置生成步长卷积编码器列表。

        参数:
            features_per_stage (list): 每个阶段的输出通道数，长度 = n_stages
            kernel_sizes (list): 每个阶段的卷积核大小，长度 = n_stages
            strides (list): 每个阶段的步长，长度 = n_stages
            n_conv_per_stage (list): 每个阶段的卷积层数，长度 = n_stages

        返回:
            nn.ModuleList: 包含从阶段1到阶段 n_stages-1 的编码器块
        """
        inc = ModuleBuilder.Get_make_layer(bs.Res_block, in_channels, features_per_stage[0])  # 输入卷积，注意因为我们没必要在输入层缩放尺寸，所以使用make_layer
        encoders = nn.ModuleList([inc])
        
        # 阶段0由输入层（inc）处理，因此编码器从阶段1开始到最后一个阶段
        for i in range(1, n_stages):
            in_ch = features_per_stage[i-1]   # 前一阶段输出通道
            out_ch = features_per_stage[i]    # 当前阶段输出通道
            kernel = kernel_sizes[i]
            stride = strides[i]

            # 计算残差块数量：每个残差块包含2个卷积层，因此块数 = n_conv_per_stage[i] // 2
            # 如果 n_conv_per_stage[i] 为奇数，则取整除后至少保留1个块
            n_convs = n_conv_per_stage[i]
            num_blocks = max(1, n_convs // 2)

            encoders.append(
                bs.StridedEncoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel,
                    stride=stride,
                    num_blocks=num_blocks
                )
            )

        return bs.EncoderBlockList(encoders)

    @staticmethod
    def Get_pyramid_encoders(num, in_channels, downsample_factors, input_sizes,
                             pyramid_levels=4):
        '''
        获取金字塔多尺度编码器列表
        
        :param num: 编码器数量
        :param in_channels: 基础通道数
        :param downsample_factors: 下采样因子列表
        :param pyramid_levels: 金字塔层数
        :param use_attention: 是否使用注意力
        :param input_sizes: 输入的特征图尺寸列表
        :return: 金字塔多尺度编码器模块列表
        '''
        encoders = nn.ModuleList()
        
        for i in range(num):
            # 计算输入和输出通道数
            input_channels = in_channels * (2 ** i)  # 16, 32, 64, ...
            output_channels = in_channels * (2 ** (i + 1))  # 32, 64, 128, ...
            
            # 创建金字塔多尺度编码器
            encoder = arb.PyramidMultiScaleEncoder(
                in_channels=input_channels,
                out_channels=output_channels,
                scale_factor=downsample_factors[i],
                pyramid_levels=pyramid_levels,input_size=input_sizes[i]
            )
            encoders.append(encoder)
        
        return bs.EncoderBlockList(encoders) 
    
    @staticmethod
    def Get_attention_res_blocks(num, channel_num, dilation_sets=None):
        '''
        获取注意力残差块列表
        
        :param num: 块数量
        :param channel_num: 各层通道数列表
        :param dilation_sets: 空洞率集合列表，如果不提供则使用默认
        :return: 注意力残差块模块列表
        '''
        if dilation_sets is None:
            # 默认的空洞率设置：深层使用更大的空洞率
            dilation_sets = []
            for i in range(num):
                if i < num // 3:  # 浅层
                    dilation_sets.append([1, 2, 3])
                elif i < 2 * num // 3:  # 中层
                    dilation_sets.append([1, 2, 4, 8])
                else:  # 深层
                    dilation_sets.append([1, 3, 6, 12, 18])
        
        attention_blocks = nn.ModuleList()
        
        for i in range(num):
            block = arb.AttentionResBlock(
                in_channels=channel_num[i],
                out_channels=channel_num[i],
                dilation_rates=dilation_sets[i]
            )
            attention_blocks.append(block)
        
        return attention_blocks
    
    @staticmethod
    def _get_suitable_num_heads(channel: int, target_heads: int) -> int:
        """
        根据特征通道数 channel 和目标头数 target_heads，返回一个能整除 channel 且最接近 target_heads 的头数。
        如果 target_heads 本身能整除，则直接返回；否则在 target_heads 附近寻找最优值。
        """
        # 限制 head 数的合理范围（最小为1，最大不超过 channel）
        max_heads = channel  # 理论上最大头数等于 channel（每个头维度为1）
        # 但实际中头数通常较小，可设置一个上限，例如 32 或 64，避免头数过多
        # 这里根据常见 Swin 设计，头数一般不超过 32，因此可以限制：
        max_heads = min(channel, 32)   # 可选约束，可根据实际调整

        # 如果目标头数超出合理范围，先限制
        if target_heads > max_heads:
            target_heads = max_heads
        if target_heads < 1:
            target_heads = 1

        # 寻找能整除 channel 且最接近 target_heads 的数
        best_heads = target_heads
        min_diff = channel  # 初始化一个大数

        # 遍历所有可能的头数（从1到max_heads），寻找能整除且差值最小的
        for h in range(1, max_heads + 1):
            if channel % h == 0:
                diff = abs(h - target_heads)
                if diff < min_diff:
                    min_diff = diff
                    best_heads = h

        return best_heads
    
    @staticmethod
    def Get_upsamples(features_size,bestsize):
        '''
        获取上采样模块列表，用于ChannelTransformer中的特征图嵌入，防止嵌入后的尺寸太小，所以上采样

        :param features_size: 特征图尺寸列表
        :param bestsize: 期望的最佳尺寸
        '''
        upsamplelist=nn.ModuleList()
        for feature_size in features_size:
            if(feature_size[0]<bestsize[0]):
                upsample=nn.Upsample(size=bestsize,mode=bs.unsample_mode,align_corners=True)
                upsamplelist.append(upsample)
        return upsamplelist
    
    @staticmethod
    def Get_residual_encoders(input_channels:int,n_stages:int,features_per_satges,kernel_sizes,strides,n_blocks_per_stage):
        '''
        创建nnU-Net所使用的带残差连接的编码器

        :param input_channels: 输入图像的通道数
        :param n_stages: 编码器个数
        :param features_per_stage: 每个编码器输出的通道数。注意: If the block is BottleneckD, then this number is supposed to be the number of
        features AFTER the expansion (which is not coded implicitly in this repository)! See todo!
        :param conv_op: 卷积类型
        :param kernel_sizes: 每个编码器的卷积核尺寸列表
        :param strides: 每个编码器的步长列表
        :param n_blocks_per_stage: 每个编码器的残差块个数
        '''
        from ResidualEncoders import ResidualEncoder
        #注意，我们应当return_skips=True，因为我们就是要每个编码器的特征图进行跳跃连接
        return ResidualEncoder(input_channels,n_stages,features_per_satges,bs.convclass,kernel_sizes,strides,n_blocks_per_stage,return_skips=True)
