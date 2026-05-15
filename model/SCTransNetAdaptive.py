# @Author  : Shuai Yuan, Trailblazer
# @File    : SCTransNetAdaptive.py
# @Software: Microsoft Visual Studio Code and PyCharm
# 在SCTransNet3D基础上修改，使其网络深度可通过外部参数配置,并且能够选择2D还是3D

from __future__ import absolute_import  # 确保导入的模块是绝对路径的，避免相对导入混淆
from __future__ import division         # 确保除法运算总是返回浮点数
from __future__ import print_function   # 确保print函数在Python2和3中行为一致

import copy      # 深拷贝模块，用于复制复杂对象
import math      # 数学函数模块
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm  # PyTorch神经网络组件
from torch.nn.modules.utils import _pair,_triple  # 将参数转换为元组格式的工具函数
import torch.nn as nn     # PyTorch神经网络模块
import torch              # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch函数式接口
from einops import rearrange    # 张量重塑库，提供更直观的张量操作
import numbers            # 数字类型检查
from thop import profile  # 模型计算量和参数量分析工具
from . import Utilities as uti #2D/3D切换方法
from .ModuleFactory import ModuleBuilder as mb # 工厂类
from . import Blocks as bs # 模块类

# spatial-embedded Single-head Channel-cross Attention (SSCA)
class SpaEmbeddedCrossAttention(nn.Module):
    """空间嵌入的单头通道交叉注意力机制————SSCA"""
    def __init__(self, config, vis, channel_num,attention_head_num=1):
        '''
        初始化交叉注意力机制
        
        :param self: self
        :param config: 配置字典
        :param vis: 是否可视化注意力权重
        :param channel_num: 各层通道数
        :param attention_head_num: 注意力头数，默认为1
        :param mheads: 注意力头列表
        :param qs: 嵌入卷积列表
        '''

        feature_num=len(channel_num)#特征图数量，因为某些深层特征图可能不参与运算
        super(SpaEmbeddedCrossAttention, self).__init__()
        self.vis = vis                    # 是否可视化注意力权重
        self.KV_size = config.KV_size     # Key-Value的尺寸
        self.channel_num = channel_num    # 各层通道数列表
        self.num_attention_heads = attention_head_num      # 注意力头数（单头）
        self.psi = bs.instance_norm(self.num_attention_heads,affine=True)  # 3D实例归一化，用于注意力图
        self.softmax = Softmax(dim=3)     # 沿第3维进行softmax###注意，这个dim=3好像与维度无关？

        self.mheads=mb.Get_mheads(feature_num,channel_num,self.num_attention_heads)

        # 键值投影：为拼接的特征创建键值投影
        self.mheadk = bs.convclass(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadv = bs.convclass(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)

        self.qs=mb.Get_qs(feature_num,channel_num,self.num_attention_heads)#嵌入卷积列表

        # 键值的空间嵌入卷积
        self.k = bs.convclass(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads, 
                          kernel_size=3, stride=1, padding=1, 
                          groups=self.KV_size * self.num_attention_heads, bias=False)
        self.v = bs.convclass(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads, 
                          kernel_size=3, stride=1, padding=1, 
                          groups=self.KV_size * self.num_attention_heads, bias=False)

        # 输出投影：将注意力输出投影回原始通道数
        self.project_outs=mb.Get_project_outs(feature_num,channel_num)
       
    ###这里向量重塑也得根据维度数来选用不同的方法，我认为可以单独写一个方法来生成。#Finished
    def forward(self, embs:nn.ModuleList, emb_all):
        """前向传播：计算多尺度通道交叉注意力"""
        b, c, d, h, w =uti.UnZipShape(embs[0].shape)  # 获取batch大小、通道数、高度、宽度，深度
        Os=[]#最终输出
        # 生成3D键值向量
        k = self.k(self.mheadk(emb_all))
        v = self.v(self.mheadv(emb_all))
        k = bs.rearrange_xd(k,self.num_attention_heads)
        v = bs.rearrange_xd(v,self.num_attention_heads)
        k = torch.nn.functional.normalize(k, dim=-1) #对查询和键进行L2归一化，提高训练稳定性
        
        #挨个遍历
        for i in range(len(self.qs)):
            q=self.qs[i](self.mheads[i](embs[i]))# 生成3D查询向量：1x1卷积 + 3x3深度卷积进行空间嵌入
            q=bs.rearrange_xd(q,self.num_attention_heads) # 重塑张量维度：从 [B, C, D, H, W] 到 [B, head, C, D*H*W]
            q=torch.nn.functional.normalize(q,dim=-1)# 对查询和键进行L2归一化，提高训练稳定性
            attn=q @ k.transpose(-2,-1)/math.sqrt(self.KV_size)# 计算注意力分数：Q * K^T / sqrt(d_k)
            attention_prob=self.softmax(self.psi(attn))# 应用实例归一化和softmax得到注意力权重
            out=(attention_prob @ v)# 应用注意力权重到值向量：注意力权重 * V
            out_=out.mean(dim=1)# 合并多头注意力输出（单头情况下就是取平均）
            out_ =bs.rearrange_out(out_,d=d,h=h,w=w)# 重塑回2D/3D空间维度。如3D时[B, C, D*H*W] -> [B, C, D, H, W]
            Os.append(self.project_outs[i](out_))# 投影输出到原始通道维度
            
        weights = None  # 如果不需要可视化，返回None
        return Os, weights

#  Spatial-channel Cross Transformer Block (SCTB)
class SpaChannelCrossTranBlock(nn.Module):
    """空间-通道交叉Transformer模块"""
    def __init__(self, config, vis, channel_num):
        super(SpaChannelCrossTranBlock, self).__init__()

        # 各层级的注意力前归一化
        self.attn_norms=mb.Get_attn_norms(config.num_SCTB,channel_num)

        # 拼接特征的归一化
        self.attn_norm = mb.Get_attn_norm(config.KV_size)

        # 核心注意力模块
        self.attnblock = SpaEmbeddedCrossAttention(config, vis, channel_num,config.transformer.num_heads)#传入配置中的注意力头数
        # self.attnblock = CrossScaleSpatialAttention(
        #     in_channels_list=channel_num[:-1],#我们不使用最深层的特征图
        #     common_dim=128,          # 可调节，建议设为 128 或 256
        #     num_heads=4,
        #     target_scale="deepest"   # 使用最深层分辨率
        # )

        # 各层级的前馈网络前归一化
        self.ffn_norms=mb.Get_ffn_norms(config.num_SCTB,channel_num)

        # 互补前馈网络
        self.ffns=mb.Get_ffns(config.num_SCTB,channel_num)

        #门控值，为了关闭效果不好的融合后的特征图。参数初始为-0.847 → sigmoid后≈0.3
        #self.gate_scales = nn.ParameterList([nn.Parameter(torch.tensor([-0.847])) for _ in range(len(channel_num))])

    #使用SpaEmbeddCrossAttenion时
    def forward(self, embs):
        '''
        SCTB的前向传播
        
        :param self: 说明
        :param embs: 嵌入层列表
        :return: 经过SCTB层多尺度融合后的特征图
        :rtype: list
        '''
        
        # 保存原始输入用于残差连接
        orgs=[]
        embcat = []
        for i in range(len(embs)):
            orgs.append(embs[i])
            # 收集所有非空的特征图
            if embs[i] is not None:
                embcat.append(embs[i])    
        
        emb_all = torch.cat(embcat, dim=1) # 沿通道维度拼接所有特征
        emb_all = self.attn_norm(emb_all)  # 拼接特征的归一化

        cxs=[]#别忘了存到列表里面
        for i in range(len(embs)):
            cxs.append(self.attn_norms[i](embs[i]) if embs[i] is not None else None)# 注意力前的归一化

        # 空间嵌入的单头通道交叉注意力
        cxs_r, weights = self.attnblock(cxs, emb_all)
        
        xs=[]#最终输出结果
        for i in range(len(embs)):
            cx=orgs[i]+cxs_r[i] if embs[i] is not None else None# 注意力残差连接
            org=cx # 保存注意力输出用于前馈网络的残差连接
            x = self.ffn_norms[i](cx) if embs[i] is not None else None# 前馈网络前的归一化
            x = self.ffns[i](x) if embs[i] is not None else None# 互补前馈网络

            # #门控机制
            # x = x + org          # x 是经过前馈网络残差后的特征
            # # 计算增强部分并应用门控
            # enhanced = x - orgs[i]                     # 净增强量
            # gate = torch.sigmoid(self.gate_scales[i])  # 门控值
            # output = orgs[i] + gate * enhanced         # 残差融合
            # xs.append(output)
            x = x + org if embs[i] is not None else None# 前馈网络残差连接
            xs.append(x)

        return xs,weights

    #使用纯空间注意力时
    # def forward(self, embs):
    #     # embs: list of features from encoder stages
    #     # 1. 保存原始输入用于残差
    #     orgs = [emb if emb is not None else None for emb in embs]

    #     # 2. 跨尺度空间注意力
    #     attn_out = self.attnblock(embs)

    #     # 3. 残差连接
    #     xs = []
    #     for i, (x, org) in enumerate(zip(attn_out, orgs)):
    #         if x is not None and org is not None:
    #             xs.append(org + x)   # 残差
    #         else:
    #             xs.append(None)

    #     # 4. 前馈网络（CFN）部分（与原代码相同）
    #     orgs_ffn = xs.copy()
    #     for i in range(len(xs)):
    #         if xs[i] is not None:
    #             x = self.ffn_norms[i](xs[i])
    #             x = self.ffns[i](x)
    #             xs[i] = orgs_ffn[i] + x

    #     return xs, None   # 第二个返回值原本是 attention weights，这里置 None


class Encoder(nn.Module):
    """编码器：包含多个SCTB层的堆叠，注意不是下采样编码器，这个才是SCTB的核心"""
    def __init__(self, config, vis, channel_num):
        '''
        __init__ 的 Docstring
        
        :param self: 说明
        :param config: 配置文件
        :param vis: 是否注意力权重可视化
        :param channel_num: 通道数列表
        :param encoder_norms: 输出归一化层列表
        '''
        super(Encoder, self).__init__()
        self.vis = vis  # 是否可视化注意力权重
        self.layer = nn.ModuleList()  # 存储多个SCTB层
        
        # 输出归一化层
        self.encoder_norms=mb.Get_encoder_norms(config.num_SCTB,channel_num)

        # 创建多个SCTB层
        for _ in range(config.transformer["num_layers"]):
            layer = SpaChannelCrossTranBlock(config, vis, channel_num)
            #self.layer.append(copy.deepcopy(layer))  # 深拷贝确保参数独立。问题是为何要深拷贝？每次都是新创建的实例。
            self.layer.append(layer)

    def forward(self, embs):
        """前向传播：依次通过所有SCTB层"""
        ####超超超极大问题：我之间模块化的时候，弄错了。结果只使用最后一个SCTB，前面的中浅层根本没用，我是在检查门控梯度回传时发现的。
        # attn_weights = []  # 存储注意力权重（用于可视化）
        
        # # 逐层处理，遍历SCTB层列表
        # embs_r=[]#处理后的embs
        # for layer_block in self.layer:
        #     embs_r, weights = layer_block(embs)
        #     if self.vis:  # 如果需要可视化，保存注意力权重
        #         attn_weights.append(weights)
                
        # # 最终归一化
        # for i in range(len(embs_r)):
        #     embs_r[i]=self.encoder_norms[i](embs_r[i]) if embs_r[i] is not None else None
        # return embs_r, attn_weights

        attn_weights = []
        x = embs
        for layer_block in self.layer:
            x, weights = layer_block(x)   # 使用上一层的输出
            if self.vis:
                attn_weights.append(weights)
        # 归一化
        for i in range(len(x)):
            x[i] = self.encoder_norms[i](x[i]) if x[i] is not None else None
        return x, attn_weights

class ChannelTransformer(nn.Module):
    """通道注意力模块：完整的空间-通道交叉变换模块"""
    def __init__(self, config, vis, img_size,channel_num,features_size):
        '''
        初始化通道注意力。
        
        注意，分辨率低于最佳嵌入尺寸的编码器特征图将会被原封不动的返回
        
        :param self: 说明
        :param config: 配置字典
        :param vis: 是否可视化注意力权重
        :param img_size: 输入图像大小
        :param channel_num: 各通道数量列表
        :param features_size: 编码器特征图尺寸列表，应为参与SCTB的编码器尺寸列表
        '''
        super().__init__()

        #以下几行代码，作用是排除深层编码器输出的低分辨率特征图，不适合用来做多尺度融合
        self.target_embedd_size,num_SCTB=uti.GetBestEmbeddSize(features_size)#获得最佳的嵌入后尺寸
        self.num_SCTB=len(features_size)-1
        self.UseDeepFeature=True#是否使用深层特征图
        if self.UseDeepFeature ==False:
            features_size=features_size[0:num_SCTB+1]#不用深层的特征图了
            channel_num=channel_num[0:num_SCTB+1]#也不用深层的通道数了
            config.num_SCTB=num_SCTB+1#这意味着我们只使用这个层级以上的特征图，太深层不适合做多尺度融合，因为信息太少，还得迁就PatchSize
            self.num_SCTB=num_SCTB+1#是索引位置，是特征图个数-1

        config.KV_size=sum(channel_num)#自动计算KV_size
        config.transformer.num_layers= config.num_SCTB#更新SCTB层的数量，接受几个编码器的特征图，就弄几个SCTB，但注意，这两个可以不相等
        # 各层级的patch大小
        patchSize = uti.GetBestPatchSizes(features_size,self.target_embedd_size)#之所以是features_size[:-1]，是因为我们默认是不用最深层级的特征图的
        self.patch_sizes=patchSize#保存下来

        self.upsamples=mb.Get_upsamples(features_size,self.target_embedd_size)#上采样模块列表，用于上采样特征图小于最佳尺寸的图像

        # 多尺度通道嵌入层，用于把各种特征图转化为一系列相同尺寸的嵌入后小尺寸特征图，方便计算注意力（计算量小了）
        self.embeddings = mb.Get_embeddings(config.num_SCTB,config,patchSize,features_size,channel_num,self.target_embedd_size)
    
        # SCTB编码器（多尺度融合操作）
        self.encoder = Encoder(config, vis, channel_num)

        # 特征重建层：将处理后的特征上采样回原始尺寸 
        self.reconstructs = mb.Get_reconstructs(config.num_SCTB,channel_num,features_size)####TODO:这个因子需要修改！修改完了。
        self.gates = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(config.num_SCTB)])
        
    def forward(self, ens):
        """前向传播：完整的通道变换流程
        
        :param ens: 所有编码器传来的特征图列表
        """
        ens_copy=copy.copy(ens[0:self.num_SCTB+1])#我们先复制一份编码器传来的特征图，用于上采样，再送到SCTB，否则残差连接时尺寸不一样
        #这里解释以下为啥只取出我们需要的特征图，因为这样一来，我们就不需要在外部再特意提出某些特征图，提高代码可读性

        length=len(ens_copy)#需要的特征图列表元素数量
        #这一步是用来上采样深层编码器的小尺寸特征图。因为深层特征图尺寸太小，导致嵌入后的特征图尺寸不得不迁就它们，而这会极大地损失空间信息
        if len(self.upsamples)>0:#要是不需要上采样，我们就不执行了
            for i in range(length):
                if ens[i].shape[2]<self.target_embedd_size[0]:#如果某一层级的特征图尺寸小于期望的特征图尺寸，我们就对其进行上采样
                    index=length-1-i#获取对应的上采样模块索引
                    ens_copy[i]=self.upsamples[index](ens[i])#进行上采样
        
        embs=[]#嵌入后的特征图列表
        # 对每一个特征图进行patch嵌入，因为原特征图尺寸太大，计算量大，所以嵌入成一个小的图像
        for i in range(length):
            embs.append(self.embeddings[i](ens_copy[i]))

        # 通过SCTB编码器对嵌入后的图像进行特征变换
        encodeds, attn_weights = self.encoder(embs)

        xs=[]#最终结果
        for i in range(length):
            x_enhanced = self.reconstructs[i](encodeds[i]) if ens[i] is not None else None# 特征重建：上采样回原始尺寸
            x = x_enhanced*self.gates[i] + ens[i] if ens[i] is not None else None# 残差连接：变换后的特征 + 原始输入特征
            xs.append(x)

        xs=xs+ens[self.num_SCTB+1:]#加上没使用的特征图
        return xs, attn_weights

class SCTransNetAdaptive(nn.Module):
    """支持2D/3D图像分割的融合交叉注意力模型"""
    def __init__(self, config, img_size,scale_factors,n_channels=1, n_classes=1,
                 vis=False, mode='train', deepsuper=True):
        '''
        初始化SCTransNet模型

        :param config: 配置字典，包含模型超参数和结构参数
        :param n_channels: 输入图像的通道数
        :param n_classes: 分割输出的类别数
        :param img_size: 输入图像的3D尺寸（深度, 高度, 宽度）
        :param vis: 是否可视化注意力权重，默认为False
        :param mode: 运行模式，'train'或'test'，影响输出格式，默认为'train'
        :param deepsuper: 是否启用深度监督，用于多尺度训练，默认为True
        :param scale_factors: 下采样过程中的缩放因子列表
        :param configuration_manager: nnU-Net提供的该训练集的配置信息
        '''

        super().__init__()
        self.IsDebug=False #是否输出调试信息
        self.vis = vis          # 是否可视化注意力权重
        self.deepsuper = deepsuper  # 是否使用深度监督
        print('SCTransNet深监督状态：', deepsuper)  # 打印深度监督状态
        self.mode = mode        # 模式：train或test
        self.n_channels = n_channels  # 输入通道数
        self.n_classes = n_classes    # 输出类别数
        
        in_channels = config.base_channel  # 基础通道数，也就是输入编码器的输出通道数
        
        # 使用 Utilities 中的函数计算各种因子
        self.relative_scales = uti.ComputeRelativeScales(scale_factors)#相对缩放因子
        self.downsample_factors = uti.ComputeDownsampleFactors(self.relative_scales)#下采样缩放因子
        self.upsample_factors = uti.ComputeUpsampleFactors(self.downsample_factors)#上采样缩放因子
        self.feature_sizes = uti.ComputeFeatureSizes(img_size, self.relative_scales)#特征图尺寸

        self.patch_sizes = config.patch_sizes
        # 可学习的深监督权重，初始值均匀分布（和为1）
        # self.ds_weights = nn.Parameter(torch.ones(config.n_stages+1) / (config.n_stages+1))#为什么是n_stages+1？因为有所有的各层级输出，以及多尺度融合卷积

        # 编码器部分：逐渐下采样增加通道数
        #self.down_encoders=mb.Get_down_encoders(config.num_decoder,in_channels,self.downsample_factors)#生成编码器
        self.down_encoders=mb.Get_down_encoders_strided(config.n_stages,config.features_per_stage,config.kernel_sizes,config.strides,config.n_conv_per_stage,n_channels)
        #self.down_encoders=mb.Get_residual_encoders(n_channels,config.n_stages,config.features_per_stage,config.kernel_sizes,config.strides,config.n_blocks_per_stage)

        # 核心：空间-通道交叉变换
        channel_num = config.features_per_stage#使用nnU-Net推荐的每个层级特征通道数
        self.mtc = ChannelTransformer(config, vis, img_size,
                                     channel_num=channel_num,features_size=self.feature_sizes)

        # 解码器部分：逐渐上采样减少通道数
        #self.up_decoders=mb.Get_up_decoders(config.num_decoder,self.upsample_factors,channel_num)#一定要注意，索引位置靠前的解码器编号大
        self.up_decoders=mb.Get_up_decoders_from_config(config.features_per_stage,config.strides,config.n_conv_per_stage_decoder,config.kernel_sizes)
        # 输出卷积
        self.outc=bs.convclass(in_channels, n_classes, 1, 1)#这里的kernel以及stride可能需要修改。不需要

        # 深度监督：多个尺度的输出头
        self.gt_convs=nn.ModuleList()#一定要注意，索引位置靠前的解码器编号大
        if self.deepsuper:
            for i in range(config.num_decoder):#其实这里有可能会多生成一个输出头，大概是num_decoder不对，但也无实际影响
                self.gt_convs.append(nn.Sequential(bs.convclass(in_channels= self.up_decoders[i].x_channels,#强制与decoder的x通道数相同，这样才能在深监督时正常运行
                                                             out_channels= n_classes,kernel_size= 1)))#注意，这里gtconv1在最末尾
            self.outconv = bs.convclass((config.num_decoder+1)*n_classes, n_classes, 1)  # 多尺度特征融合卷积

    def forward(self, x):
        """前向传播：完整的SCTransNet推理流程"""
        
        # 编码器路径
        xs=self.down_encoders(x)#编码器各层输出

        # 保存原始编码器特征用于残差连接
        fs=xs.copy()

        # 空间-通道交叉变换：增强特征表示———送给SCTB
        xs, att_weights = self.mtc(xs)#在这之后，xs部分变为多尺度融合之后的特征，部分仍是编码器原输出

        #原作者在这里又做了一次残差连接，明明ChannelTransformer.forward末尾做了一次，还做？移除！
        # 残差连接：变换后的特征 + 原始特征
        for i in range(len(xs)):
            xs[i]=xs[i]+fs[i]

        # 解码器路径：特征融合与上采样
        ds=[xs[-1]]#因为最后一个特征图是U-Net里面的"Bridge"
        num_updecoders=len(self.up_decoders)
        for i in range(num_updecoders-1):#最后一个up_decoder单独用于out
            ds.append(self.up_decoders[i](ds[i],xs[num_updecoders-1-i]))
        out = self.outc(self.up_decoders[-1](ds[-1], xs[0]))  # 最终上采样并输出

        # 深度监督：返回多尺度预测融合
        if self.deepsuper and self.mode=="train":
            # 各层级的预测
            gt_s=[]
            for i in range(len(self.gt_convs)):
                gt_s.append(self.gt_convs[i](ds[i]))#这样一来gt_s=[7,6,5,4,3,2]
            gt_s.reverse()#逆转序列，以便于与nnN-Net期望的尺度顺序相匹配，即尺寸从大到小

            gt_s_up=[]#因为在通道维度拼接必须要求空间尺寸相同，所以先把所有特征图上采样
            for gt_out in gt_s:
                gt_s_up.append(F.interpolate(gt_out,out.shape[2:],mode=bs.unsample_mode,align_corners=True))
            # 最终输出 out 已经计算（最浅层，原图分辨率）
            # 拼接所有预测（通道维度），out 放在最前面（顺序可调，但需与后续损失计算匹配）
            cat = torch.cat([out] + gt_s_up, dim=1)       # 通道数 = (len(gt_s)+1) * n_classes

            # 融合输出
            d0 = self.outconv(cat)                     # [B, n_classes, D, H, W] 与原图同分辨率

            # 返回列表：最终输出 out（最浅）、融合输出 d0、各深监督预测（从深到浅）
            outlist = [out] + [d0] + gt_s
            # outlist=[out]+gt_s# 不使用多尺度融合深监督
            return outlist
        else:
            # 测试模式：返回 out 
            return out    # 通常返回最终输出
