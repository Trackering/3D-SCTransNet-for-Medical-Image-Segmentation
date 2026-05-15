# nnUNetTrainerSCTransNet.py
import warnings
# 忽略特定的警告
warnings.filterwarnings("ignore", 
                       message="Using a non-tuple sequence for multidimensional indexing is deprecated",
                       category=UserWarning)

import inspect
import torch
import numpy as np
import time
from typing import Tuple, Union, List
from torch import nn
from torch import autocast
from torch._dynamo import OptimizedModule

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule
import torch.distributed as dist
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context
from batchgenerators.utilities.file_and_folder_operations import join
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

from .model.SCTransNetAdaptive import SCTransNetAdaptive
from .model.NetworkBuilder import NetworkBuilder as ntb
from .funny.congra import main as congratulations#祝贺信息
from .loss.DC_Focal import DC_and_Focal_loss#自己写的DC+Focal损失函数
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from .LearnableDeepSupervisionWrapper import LearnableDeepSupervisionWrapper

class nnUNetTrainerSCTransNet(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        self.IsDebug=False#是否在调试状态
        self.hascongra=False#是否已经显示祝贺信息

        #对接nnUNet提供的数据信息
        self.num_classes = self.label_manager.num_segmentation_heads#分割类别数

        # 从 dataset_json 中获取类别名列表，按标签值排序，目的是输出Pseudo Dice时各器官名称一目了然
        labels = self.dataset_json['labels']
        # labels 示例：{'background': 0, 'liver': 1, 'tumor': 2, ...}
        self.class_names = [name for name, idx in sorted(labels.items(), key=lambda x: x[1])][1:]#去除背景，通常背景是第一个
        #self.class_names 存储了按标签顺序排列的类别名

        # 超参数
        self.initial_lr = 1e-3  #学习率
        self.weight_decay = 1e-4 #权重衰减
        self.gradient_clip_norm=10 #梯度剪裁
        self.num_epochs = 1000  #训练轮数
        self.save_every = 50  #每多少轮保存一次检查点
        self.plans=plans['configurations'][configuration]#获取对应训练集的配置字典

        # 根据patch_size确定数据维度
        if hasattr(self.configuration_manager, 'patch_size'):
            self.patch_size = self.configuration_manager.patch_size
            self.data_dim = len(self.patch_size)  # 输入数据的维度，2或3
        else:
            # 默认为3D
            self.patch_size = (128, 128, 128)
            self.data_dim = 3
        
        print(f"检测到 {self.data_dim}D 数据")
        print(f"图像尺寸(Patch Size): {self.patch_size}")
        
        self.img_size=self.patch_size#直接作为图像尺寸即可

        self.netbuilder=None
        print(f"使用 {'2D' if self.data_dim == 2 else '3D'} ")
        print(f"参数配置完成，图像尺寸: {self.img_size}")
        
    #是否启用编译
    def _do_i_compile(self):
        return True
        
    def initialize(self):
        if not self.was_initialized:
            #输出数据集基本信息
            print(f"多分类任务: 期望输出通道数 = {self.num_classes}")
            print(f"是否有忽略标签: {self.label_manager.has_ignore_label}")
            print(f"是否有区域: {self.label_manager.has_regions}")
            
            #先创建模型
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json)
            self.netbuilder=ntb(self.img_size,self.num_input_channels,self.num_classes,self.plans,dim=self.data_dim)
            self.network = self.build_network_architecture()

            # 获取深监督尺度
            print("进行一次虚拟前向传播，以获取深监督尺度")
            self.deep_supervision_scales = self._get_deep_supervision_scales()
            if self.deep_supervision_scales is not None:
                expected_scales = len(self.deep_supervision_scales)
                print(f"期望的深监督尺度数量: {expected_scales}")
                print(self.deep_supervision_scales)
            self.network=self.network.to(self.device)#千万不要忘了移动到目标设备，否则编译很容易出错！我弄了几个小时才发现！
            self._set_batch_size_and_oversample()
                       
            if self._do_i_compile():
                self.print_to_log_file('开启编译！')
                self.network = torch.compile(self.network,backend="inductor",
                dynamic=True,
                fullgraph=False,
                #mode="default",
                options={
                    "trace.enabled": False,#编译跟踪，这会极大地增加编译时间
                    "triton.cudagraphs": False,
                })  # 不要求完整图
            
            if self.is_ddp:#使用多GPU并行时
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])
            
            self.loss = self._build_loss()
            self.print_to_log_file(f"使用的损失函数类型：{self.loss._get_name()}")

            self.optimizer, self.lr_scheduler = self.configure_optimizers()#创建优化器，注意，如果有什么自定义的可学习参数，一定要在优化器创建之前创建

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            
            self.was_initialized = True
        else:
            raise RuntimeError("Trainer already initialized!")

    def build_network_architecture(self) -> nn.Module:
        print(f"构建网络，输入通道: {self.num_input_channels}，输入类别数：{self.label_manager.num_segmentation_heads}")
        
        network = self.netbuilder.Build()
        self.print_to_log_file(f"使用模型：{network._get_name()}")
        return network

    def _get_deep_supervision_scales(self):
        if not self.enable_deep_supervision:
            return None
        # 构建一个虚拟输入，形状与训练数据一致
        dummy_input = torch.randn(1, self.num_input_channels, *self.configuration_manager.patch_size)
        with torch.no_grad():
            outputs = self.network(dummy_input)  # outputs 是一个列表或元组
        if isinstance(outputs, (list, tuple)):
            # 对每个输出，计算空间尺寸相对于输入尺寸的比例
            input_spatial = np.array(self.configuration_manager.patch_size)
            scales = []
            for out in outputs:
                out_spatial = np.array(out.shape[2:])  # 假设输出格式为 (b, c, d, h, w)
                scale = out_spatial / input_spatial    # 计算各维度的缩放比例
                scales.append(scale.tolist())
        else:
            # 如果只有单个输出，按需处理
            scales = None
        return scales
    
    def set_deep_supervision_enabled(self, enabled: bool):
        pass
    
    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self.deep_supervision_scales

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val
    
    #优化器相关配置
    def configure_optimizers(self):
        # # 假设已有的优化器定义
        # optimizer = torch.optim.Adam(
        #     self.network.parameters(),
        #     lr=self.initial_lr,
        #     weight_decay=self.weight_decay
        # )
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)

        # --- 设置 Warmup ---
        warmup_epochs = 50  # 可根据需要调整，比如 5 或总 epoch 的 10%
        # 线性 Warmup：从 0.01 * initial_lr 上升到 initial_lr
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,   # 起始学习率倍数 (0.01 * initial_lr)
            end_factor=1.0,      # 结束时倍数 (1.0 * initial_lr)
            total_iters=warmup_epochs
        )

        # # --- 设置余弦退火（扣除 Warmup 的 epoch 数）---
        # cosine_scheduler = CosineAnnealingLR(
        #     optimizer,
        #     T_max=self.num_epochs - warmup_epochs,  # 剩余 epoch
        #     eta_min=1e-6
        # )
        pllr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs-warmup_epochs)

        # --- 串联两个调度器 ---
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, pllr_scheduler],
            milestones=[warmup_epochs]   # 在第 warmup_epochs 个 epoch 后切换调度器
        )

        return optimizer, lr_scheduler

    #因为我们修改损失函数，所以得重写_build_loss
    def _build_loss(self):
        if self.label_manager.has_regions:#有区域就用这个，大概是二分类吧
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
        #     loss=DC_and_Focal_loss(
        #     soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
        #                       'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
        #     focal_kwargs={'gamma': 2.0,          # 可根据需要调整
        #                   'alpha': None,         # 类别平衡权重，None 表示不使用
        #                   'reduction': 'mean'},  # 保持与 CE 一致的 reduction
        #     weight_focal=1,          # 对应原 weight_ce
        #     weight_dice=1,           # 对应原 weight_dice
        #     ignore_label=self.label_manager.ignore_label,
        #     dice_class=MemoryEfficientSoftDiceLoss
        # )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # 给每个输出赋予一个权重，该权重随着分辨率的降低而指数递减（每次除以2）；
        # 这使得更高分辨率的输出在损失中具有更大的权重。

        if self.enable_deep_supervision:
            deep_supervision_scales = self.deep_supervision_scales
            # 原来的固定硬编码权重
            # weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights = np.array([1.0,]*len(deep_supervision_scales))
            if self.is_ddp and not self._do_i_compile():
                # 有一个非常奇怪且愚蠢的交互问题：当 weights[-1] = 0时，DDP 会崩溃，并声称有未使用的参数。
                # 有趣的是，启用 torch.compile 后，这个崩溃就不发生了，真奇怪。
                # 总之，最简单的解决方法就是把这里的权重设置得非常低。
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            # 注意，nnUNet原设计，是模型在深监督时是不输出最浅层解码的特征图的。然后他们让weight[-1]又排除了第二浅的
            # 因此说“we dont't use the lowest 2 outpus”
            # 然而，我们一次性输出所有层级，那么就应该weight[-1],weight[-2]=0
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
    
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        
        data = data.to(self.device, non_blocking=True)
        
        # DICE+CE 损失期望特定格式的目标
        if isinstance(target, list):
            # 深度监督：有多个尺度的target
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            # 单尺度target
            target = target.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        #使用混合精度计算
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            outputs = self.network(data)
            # 计算 DICE+CE 损失
            l = self.loss(outputs, target)
                
        # 反向传播
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip_norm)
            self.optimizer.step()
        
        return {'loss': l.detach().cpu().numpy()}
    
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
    #重写训练结束后的事件，原因是在之后验证时，要求模型进入测试模式，输出一个张量，而不是多特征图的元组
    def on_train_end(self):
        self.network.mode='test'
        super().on_train_end()#执行父类方法
    
    #以下几个方法是为了解决PyTorch推荐的调度器使用顺序的
    def on_train_epoch_start(self):
        """在每个训练epoch开始时调用 - 修改以修复调度器警告"""
        self.network.train()
        
        # 注意：这里不再调用 lr_scheduler.step()，因为我们需要在优化器更新后调用
        
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        
        # 获取当前学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        self.print_to_log_file(f"当前学习率: {np.round(current_lr, decimals=5)}")
        
        # 记录学习率
        self.logger.log('lrs', current_lr, self.current_epoch)

    def on_train_epoch_end(self, train_outputs: List[dict]):
        """在每个训练epoch结束时调用 - 在这里更新学习率调度器"""
        # 首先计算和记录训练损失（如果有的话）
        if train_outputs:
            outputs = collate_outputs(train_outputs)
            
            if self.is_ddp:
                losses_tr = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(losses_tr, outputs['loss'])
                loss_here = np.vstack(losses_tr).mean()
            else:
                loss_here = np.mean(outputs['loss'])
            
            self.logger.log('train_losses', loss_here, self.current_epoch)
        
        # 在优化器更新后调用学习率调度器（修复警告）
        self.lr_scheduler.step()
        
        # 打印下一个epoch将使用的学习率
        next_lr = self.optimizer.param_groups[0]['lr']
        self.print_to_log_file(f"下一轮学习率将会是: {np.round(next_lr, decimals=5)}")
        
        for name, param in self.network.named_parameters():
            if param.grad is None:        
                self.print_to_log_file(f"警告！{name}的梯度为None！")

        # 打印可学习的深监督权重（如果启用深监督且模型包含该参数）
        if self.enable_deep_supervision:
            # 获取原始模型（处理 DDP 和 torch.compile 包装）
            net = self.network
            if isinstance(net, (DDP, OptimizedModule)):
                net = net.modules
            if hasattr(net, 'ds_weights'):
                with torch.no_grad():
                    # 对 logits 做 softmax 得到实际权重
                    w = torch.softmax(net.ds_weights, dim=0).cpu().numpy()
                self.print_to_log_file(f"深监督权重: {np.round(w, decimals=4)}")

    def on_epoch_end(self):
        """在验证结束后调用 - 调整调用顺序"""
        # 注意：学习率调度器已经在 on_train_epoch_end 中更新了
        # 这里只需要调用父类的其他逻辑
        self.logger.log('epoch_end_timestamps', time.time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))

        ###打印各个类别的Pseudo Dice
        # self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
        #                                     self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        # 获取当前 epoch 的 Dice 列表
        dice_list = self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]
        # 确保类别名列表与 Dice 列表长度一致
        assert len(self.class_names) == len(dice_list), \
            f"类别名数量 {len(self.class_names)} 与 Dice 数量 {len(dice_list)} 不匹配"
        # 格式化成 "类别名:dice" 字符串
        dice_str = ', '.join([f"{name}:{dice:.4f}" for name, dice in zip(self.class_names, dice_list)])
        self.print_to_log_file(f"Pseudo dice: {dice_str}")
        
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

         # 处理周期性检查点保存
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # 处理最佳检查点
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"太好了！新的最佳EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

            if(self.hascongra ==False and self._best_ema>0.85):#如果pseudo dice大于0.85显示祝贺信息
                congratulations(self._best_ema)
                self.hascongra=True
            #简直是搞笑，连模型都不稳定还祝贺什么？<-过去式了，现在我都搭建了一个完全自适应架构，我看超过0.85也不是不可能

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()
        
        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value
        
        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        
        if 'inference_allowed_mirroring_axes' in checkpoint.keys():
            self.inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes']
        
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if self.grad_scaler is not None and 'grad_scaler_state' in checkpoint:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
        
        self.print_to_log_file(f"Loaded checkpoint from epoch {self.current_epoch}")
