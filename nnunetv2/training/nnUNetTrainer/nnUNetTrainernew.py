from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

import torch
from torch.optim import AdamW
from torch import nn

# from .fixed_idcunet import IDCPlainConvUNet

# from .fixed_idcunet import IDCPlainConvUNet

# from nnUNet.fixed_idcunet import IDCPlainConvUNet


# from nnunet.fixed_idcunet import IDCPlainConvUNet
from typing import Tuple, Union, List


class nnUNetTrainernew(nnUNetTrainerNoDeepSupervision):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        # unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            # plans, configuration, fold, dataset_json, unpack_dataset, device
            plans,
            configuration,
            fold,
            dataset_json,
            device,
        )
        original_patch_size = self.configuration_manager.patch_size
        new_patch_size = [-1] * len(original_patch_size)
        for i in range(len(original_patch_size)):
            if (original_patch_size[i] / 2**5) < 1 or (
                (original_patch_size[i] / 2**5) % 1
            ) != 0:
                new_patch_size[i] = round(original_patch_size[i] / 2**5 + 0.5) * 2**5
            else:
                new_patch_size[i] = original_patch_size[i]
        self.configuration_manager.configuration["patch_size"] = new_patch_size
        self.print_to_log_file(
            "Patch size changed from {} to {}".format(
                original_patch_size, new_patch_size
            )
        )
        self.plans_manager.plans["configurations"][self.configuration_name][
            "patch_size"
        ] = new_patch_size

        self.grad_scaler = None
        self.initial_lr = 8e-4
        self.weight_decay = 0.01

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        torch.set_float32_matmul_precision("high")

        # model = IDCPlainConvUNet(
        #     input_channels=num_input_channels,
        #     n_stages=6,
        #     features_per_stage=[32, 64, 128, 256, 320, 320],
        #     conv_op=nn.Conv3d,
        #     kernel_sizes=[
        #         [3, 3, 3],
        #         [3, 3, 3],
        #         [3, 3, 3],
        #         [3, 3, 3],
        #         [3, 3, 3],
        #         [3, 3, 3],
        #     ],
        #     strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
        #     n_conv_per_stage=[2, 2, 2, 2, 2, 2],
        #     num_classes=num_output_channels,
        #     n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
        #     conv_bias=False,
        #     norm_op=nn.InstanceNorm3d,
        #     norm_op_kwargs={"eps": 1e-05},
        #     dropout_op=None,
        #     dropout_op_kwargs=None,
        #     nonlin=nn.LeakyReLU,
        #     nonlin_kwargs=None,
        #     deep_supervision=False,
        #     nonlin_first=False,
        # ).to("cuda")

        model = IDCPlainConvUNet(
            input_channels=num_input_channels,
            n_stages=5,
            features_per_stage=[32, 64, 128, 256,320],
            conv_op=nn.Conv3d,
            kernel_sizes=[
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2],[1, 2, 2]],
            n_conv_per_stage=[2, 2, 2, 2, 2],
            num_classes=num_output_channels,
            n_conv_per_stage_decoder=[2, 2, 2,2],
            conv_bias=False,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-05},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs=None,
            deep_supervision=False,
            nonlin_first=False,
            use_dual_branch=True,
        ).to("cuda")
        return model

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        output = self.network(data)
        l = self.loss(output, target)
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()

        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        output = self.network(data)
        del data
        l = self.loss(output, target)

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float32
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=mask
        )

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

        return {
            "loss": l.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }

    def configure_optimizers(self):

        optimizer = AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-5,
        )
        scheduler = PolyLRScheduler(
            optimizer, self.initial_lr, self.num_epochs, exponent=1.0
        )

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")

        return optimizer, scheduler

    def set_deep_supervision_enabled(self, enabled: bool):
        pass


from typing import Union, Type, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import (
    init_last_bn_before_add_to_0,
)
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import (
    maybe_convert_scalar_to_list,
    get_matching_pool_op,
)
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp


class IDCPlainConvUNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        use_dual_branch: bool = True,
    ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        use_dual_branch: if True, use dual-branch encoder with IDC and residual branches
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, (
            "n_conv_per_stage must have as many entries as we have "
            f"resolution stages. here: {n_stages}. "
            f"n_conv_per_stage: {n_conv_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            "n_conv_per_stage_decoder must have one less entries "
            f"as we have resolution stages. here: {n_stages} "
            f"stages, so it should have {n_stages - 1} entries. "
            f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        )
        
        if use_dual_branch:
            self.encoder = DualBranchEncoder(
                input_channels,
                n_stages,
                features_per_stage,
                conv_op,
                kernel_sizes,
                strides,
                n_conv_per_stage,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                return_skips=True,
                nonlin_first=nonlin_first,
            )
            print("-------------------------------------IDCDualBranchUNet------------------------------------")
        else:
            self.encoder = PlainConvEncoder(
                input_channels,
                n_stages,
                features_per_stage,
                conv_op,
                kernel_sizes,
                strides,
                n_conv_per_stage,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                return_skips=True,
                nonlin_first=nonlin_first,
            )
            print("-------------------------------------IDCPlainConvUNet------------------------------------")
            
        self.decoder = UNetDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision,
            nonlin_first=nonlin_first,
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        return self.encoder.compute_conv_feature_map_size(
            input_size
        ) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


class DualBranchEncoder(nn.Module):
    """双支路编码器：一个支路使用IDC卷积，另一个支路使用普通残差连接"""
    
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        nonlin_first: bool = False,
        pool: str = "conv",
    ):
        super().__init__()
        
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
            
        assert (
            len(kernel_sizes) == n_stages
        ), "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(n_conv_per_stage) == n_stages
        ), "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(features_per_stage) == n_stages
        ), "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, (
            "strides must have as many entries as we have resolution stages (n_stages). "
            "Important: first entry is recommended to be 1, else we run strided conv directly on the input"
        )

        # 初始输入处理层
        self.input_conv = ConvDropoutNormReLU(
            conv_op,
            input_channels,
            features_per_stage[0] // 2,  # 每个支路使用一半的特征通道
            kernel_sizes[0],
            1,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            nonlin_first,
        )

        # 构建双支路阶段
        stages = []
        for s in range(n_stages):
            # 计算当前阶段的输入通道数
            if s == 0:
                stage_input_channels = features_per_stage[0] // 2
            else:
                stage_input_channels = features_per_stage[s-1]
                
            stage_output_channels = features_per_stage[s]
            
            # 下采样处理
            stage_modules = []
            if pool == "max" or pool == "avg":
                if (
                    (isinstance(strides[s], int) and strides[s] != 1)
                    or isinstance(strides[s], (tuple, list))
                    and any([i != 1 for i in strides[s]])
                ):
                    stage_modules.append(
                        get_matching_pool_op(conv_op, pool_type=pool)(
                            kernel_size=strides[s], stride=strides[s]
                        )
                    )
                conv_stride = 1
            elif pool == "conv":
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            
            # 双支路模块
            dual_branch_module = DualBranchStage(
                n_conv_per_stage[s],
                conv_op,
                stage_input_channels,
                stage_output_channels,
                kernel_sizes[s],
                conv_stride,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first,
            )
            
            stage_modules.append(dual_branch_module)
            stages.append(nn.Sequential(*stage_modules))

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # 存储解码器需要的信息
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        x = self.input_conv(x)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = self.input_conv.compute_conv_feature_map_size(input_size)
        
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, "compute_conv_feature_map_size"):
                        output += sq.compute_conv_feature_map_size(input_size)
            else:
                if hasattr(self.stages[s], "compute_conv_feature_map_size"):
                    output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


class ResidualBranch(nn.Module):
    """残差连接支路，保持与IDCBranch相同的下采样方式"""
    
    def __init__(
        self,
        num_convs: int,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        nonlin_first: bool = False,
    ):
        super().__init__()
        
        # 主路径卷积层
        if not isinstance(output_channels, (tuple, list)):
            output_channels_list = [output_channels] * num_convs
        else:
            output_channels_list = output_channels
            
        convs = []
        current_input_channels = input_channels
        
        for i in range(num_convs):
            # 只在第一个卷积层进行下采样
            current_stride = stride if i == 0 else 1
            current_output_channels = output_channels_list[i]
            
            # 除了最后一层，其他层都使用激活函数
            use_nonlin = nonlin if i < num_convs - 1 else None
            use_nonlin_kwargs = nonlin_kwargs if i < num_convs - 1 else None
            
            conv_layer = ConvDropoutNormReLU(
                conv_op,
                current_input_channels,
                current_output_channels,
                kernel_size,
                current_stride,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                use_nonlin,
                use_nonlin_kwargs,
                nonlin_first,
            )
            convs.append(conv_layer)
            current_input_channels = current_output_channels
            
        self.convs = nn.Sequential(*convs)
        
        # 残差连接的快捷路径
        if input_channels != output_channels_list[-1] or stride != 1:
            self.shortcut = ConvDropoutNormReLU(
                conv_op,
                input_channels,
                output_channels_list[-1],
                1,  # 1x1卷积
                stride,  # 与主路径相同的下采样步长
                conv_bias,
                norm_op,
                norm_op_kwargs,
                None,  # 快捷路径不使用dropout
                None,
                None,  # 快捷路径不使用激活函数
                None,
                nonlin_first,
            )
        else:
            self.shortcut = nn.Identity()
            
        # 最终激活函数
        if nonlin is not None:
            self.final_activation = nonlin(**nonlin_kwargs if nonlin_kwargs else {})
        else:
            self.final_activation = nn.Identity()
            
        self.output_channels = output_channels_list[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, stride)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.convs(x)
        out = out + identity
        out = self.final_activation(out)
        return out

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        current_size = input_size
        
        # 主路径
        for i, conv in enumerate(self.convs):
            output += conv.compute_conv_feature_map_size(current_size)
            if i == 0:
                current_size = [i // j for i, j in zip(current_size, self.initial_stride)]
        
        # 快捷路径
        if hasattr(self.shortcut, 'compute_conv_feature_map_size'):
            output += self.shortcut.compute_conv_feature_map_size(input_size)
            
        return output


class IDCBranch(nn.Module):
    """IDC卷积支路，保持与ResidualBranch相同的下采样方式"""
    
    def __init__(
        self,
        num_convs: int,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        nonlin_first: bool = False,
    ):
        super().__init__()
        
        # 使用IDC卷积构建支路
        if not isinstance(output_channels, (tuple, list)):
            output_channels_list = [output_channels] * num_convs
        else:
            output_channels_list = output_channels
            
        convs = []
        current_input_channels = input_channels
        
        for i in range(num_convs):
            # 只在第一个卷积层进行下采样
            current_stride = stride if i == 0 else 1
            current_output_channels = output_channels_list[i]
            
            conv_layer = IDCConvDropoutNormReLU(
                conv_op,
                current_input_channels,
                current_output_channels,
                kernel_size,
                current_stride,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first,
            )
            convs.append(conv_layer)
            current_input_channels = current_output_channels
            
        self.convs = nn.Sequential(*convs)
        self.output_channels = output_channels_list[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, stride)

    def forward(self, x):
        return self.convs(x)

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        current_size = input_size
        
        for i, conv in enumerate(self.convs):
            output += conv.compute_conv_feature_map_size(current_size)
            if i == 0:
                current_size = [i // j for i, j in zip(current_size, self.initial_stride)]
                
        return output


class DualBranchStage(nn.Module):
    """双支路阶段模块，确保两个分支的下采样一致"""
    
    def __init__(
        self,
        num_convs: int,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        nonlin_first: bool = False,
    ):
        super().__init__()
        
        # 每个支路的输出通道数
        branch_channels = output_channels // 2
        
        # IDC支路
        self.idc_branch = IDCBranch(
            num_convs,
            conv_op,
            input_channels,
            branch_channels,
            kernel_size,
            stride,  # 使用相同的下采样步长
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            nonlin_first,
        )
        
        # 残差支路
        self.residual_branch = ResidualBranch(
            num_convs,
            conv_op,
            input_channels,
            branch_channels,
            kernel_size,
            stride,  # 使用相同的下采样步长
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            nonlin_first,
        )
        
        # 特征融合层
        self.fusion_conv = ConvDropoutNormReLU(
            conv_op,
            branch_channels * 2,
            output_channels,
            1,  # 1x1卷积用于融合
            1,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            nonlin_first,
        )
        
        self.output_channels = output_channels
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, stride)

    def forward(self, x):
        # IDC支路
        idc_out = self.idc_branch(x)
        
        # 残差支路
        residual_out = self.residual_branch(x)
        
        # 确保两个分支的输出尺寸相同
        if idc_out.shape[2:] != residual_out.shape[2:]:
            # 如果尺寸不匹配，进行插值调整
            residual_out = F.interpolate(
                residual_out, 
                size=idc_out.shape[2:], 
                mode='trilinear' if residual_out.ndim == 5 else 'bilinear',
                align_corners=True
            )
        
        # 特征融合
        fused = torch.cat([idc_out, residual_out], dim=1)
        out = self.fusion_conv(fused)
        
        return out
class IDCConvDropoutNormReLU(nn.Module):
    """使用IDC卷积的ConvDropoutNormReLU"""
    
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        nonlin_first: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []
        
        # 使用IDC卷积
        if input_channels == output_channels and all(s == 1 for s in stride):
            self.conv = IDConv(
                conv_op=conv_op,
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=[(i - 1) // 2 for i in kernel_size],
                bias=conv_bias,
            )
        else:
            # 对于通道数不匹配或需要下采样的情况，使用常规卷积
            self.conv = conv_op(
                input_channels,
                output_channels,
                kernel_size,
                stride,
                padding=[(i - 1) // 2 for i in kernel_size],
                dilation=1,
                bias=conv_bias,
            )
        ops.append(self.conv)

        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)

        if nonlin_first and (norm_op is not None and nonlin is not None):
            ops[-1], ops[-2] = ops[-2], ops[-1]

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x):
        return self.all_modules(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        output_size = [
            i // j for i, j in zip(input_size, self.stride)
        ]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)


class PlainConvEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        nonlin_first: bool = False,
        pool: str = "conv",
    ):

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert (
            len(kernel_sizes) == n_stages
        ), "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(n_conv_per_stage) == n_stages
        ), "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(features_per_stage) == n_stages
        ), "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, (
            "strides must have as many entries as we have resolution stages (n_stages). "
            "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        )

        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == "max" or pool == "avg":
                if (
                    (isinstance(strides[s], int) and strides[s] != 1)
                    or isinstance(strides[s], (tuple, list))
                    and any([i != 1 for i in strides[s]])
                ):
                    stage_modules.append(
                        get_matching_pool_op(conv_op, pool_type=pool)(
                            kernel_size=strides[s], stride=strides[s]
                        )
                    )
                conv_stride = 1
            elif pool == "conv":
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            stage_modules.append(
                StackedConvBlocks(
                    n_conv_per_stage[s],
                    conv_op,
                    input_channels,
                    features_per_stage[s],
                    kernel_sizes[s],
                    conv_stride,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                )
            )
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, "compute_conv_feature_map_size"):
                        output += self.stages[s][-1].compute_conv_feature_map_size(
                            input_size
                        )
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


class UNetDecoder(nn.Module):
    def __init__(
        self,
        encoder: Union[PlainConvEncoder, DualBranchEncoder],
        num_classes: int,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision,
        nonlin_first: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        conv_bias: bool = None,
    ):
        """
        This class needs the skips of the encoder as input in its forward.
        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, (
            "n_conv_per_stage must have as many entries as we have "
            "resolution stages - 1 (n_stages in encoder - 1), "
            "here: %d" % n_stages_encoder
        )

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = (
            encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        )
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = (
            encoder.dropout_op_kwargs
            if dropout_op_kwargs is None
            else dropout_op_kwargs
        )
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = (
            encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs
        )

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(
                transpconv_op(
                    input_features_below,
                    input_features_skip,
                    stride_for_transpconv,
                    stride_for_transpconv,
                    bias=conv_bias,
                )
            )
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(
                StackedConvBlocks(
                    n_conv_per_stage[s - 1],
                    encoder.conv_op,
                    2 * input_features_skip,
                    input_features_skip,
                    encoder.kernel_sizes[-(s + 1)],
                    1,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                )
            )

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(
                encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True)
            )

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            # 确保尺寸匹配
            skip = skips[-(s + 2)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear' if x.ndim == 5 else 'bilinear', align_corners=True)
            
            x = torch.cat((x, skip), 1)
            x = self.stages[s](x)
            
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        # 确保总是返回张量，不是列表
        if not self.deep_supervision:
            return seg_outputs[0]
        else:
            return seg_outputs

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append(
                [i // j for i, j in zip(input_size, self.encoder.strides[s])]
            )
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            # trans conv
            output += np.prod(
                [self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]],
                dtype=np.int64,
            )
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod(
                    [self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64
                )
        return output


class StackedConvBlocks(nn.Module):
    def __init__(
        self,
        num_convs: int,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: Union[int, List[int], Tuple[int, ...]],
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        initial_stride: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        nonlin_first: bool = False,
    ):
        """
        :param conv_op:
        :param num_convs:
        :param input_channels:
        :param output_channels: can be int or a list/tuple of int. If list/tuple are provided, each entry is for
        one conv. The length of the list/tuple must then naturally be num_convs
        :param kernel_size:
        :param initial_stride:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        """
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op,
                input_channels,
                output_channels[0],
                kernel_size,
                initial_stride,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first,
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op,
                    output_channels[i - 1],
                    output_channels[i],
                    kernel_size,
                    1,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                )
                for i in range(1, num_convs)
            ],
        )

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x):
        return self.convs(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output


class ConvDropoutNormReLU(nn.Module):
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        nonlin_first: bool = False,
    ):
        super(ConvDropoutNormReLU, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []
        if input_channels == output_channels:
            self.conv = IDConv(
                conv_op=conv_op,
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=[(i - 1) // 2 for i in kernel_size],
                bias=conv_bias,
            )
        else:
            self.conv = conv_op(
                input_channels,
                output_channels,
                kernel_size,
                stride,
                padding=[(i - 1) // 2 for i in kernel_size],
                dilation=1,
                bias=conv_bias,
            )
        ops.append(self.conv)

        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)

        if nonlin_first and (norm_op is not None and nonlin is not None):
            ops[-1], ops[-2] = ops[-2], ops[-1]

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x):
        return self.all_modules(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        output_size = [
            i // j for i, j in zip(input_size, self.stride)
        ]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)


class IDConv(nn.Module):
    def __init__(
        self,
        conv_op,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    ):
        super().__init__()
        if conv_op == torch.nn.modules.conv.Conv2d:
            self.conv = InceptionDWConv2d(in_channels, 3, band_kernel_size=11)
        elif conv_op == torch.nn.modules.conv.Conv3d:
            self.conv = InceptionDWConv3d(in_channels, 3, band_kernel_size=11)

    def forward(self, x):
        return self.conv(x)


class InceptionDWConv3d(nn.Module):
    """Inception depthwise convolution for 3D data"""

    def __init__(
        self, in_channels, cube_kernel_size=3, band_kernel_size=11, branch_ratio=0.125
    ):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hwd = nn.Conv3d(
            gc, gc, cube_kernel_size, padding=cube_kernel_size // 2, groups=gc
        )
        self.dwconv_wd = nn.Conv3d(
            gc,
            gc,
            kernel_size=(1, 1, band_kernel_size),
            padding=(0, 0, band_kernel_size // 2),
            groups=gc,
        )
        self.dwconv_hd = nn.Conv3d(
            gc,
            gc,
            kernel_size=(1, band_kernel_size, 1),
            padding=(0, band_kernel_size // 2, 0),
            groups=gc,
        )
        self.dwconv_hw = nn.Conv3d(
            gc,
            gc,
            kernel_size=(band_kernel_size, 1, 1),
            padding=(band_kernel_size // 2, 0, 0),
            groups=gc,
        )
        self.split_indexes = (in_channels - 4 * gc, gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hwd, x_wd, x_hd, x_hw = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (
                x_id,
                self.dwconv_hwd(x_hwd),
                self.dwconv_wd(x_wd),
                self.dwconv_hd(x_hd),
                self.dwconv_hw(x_hw),
            ),
            dim=1,
        )


class InceptionDWConv2d(nn.Module):
    """Inception depthwise convolution for 2D data"""

    def __init__(
        self, in_channels, cube_kernel_size=3, band_kernel_size=11, branch_ratio=0.25
    ):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hwd = nn.Conv2d(
            gc, gc, cube_kernel_size, padding=cube_kernel_size // 2, groups=gc
        )
        self.dwconv_wd = nn.Conv2d(
            gc,
            gc,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
            groups=gc,
        )
        self.dwconv_hd = nn.Conv2d(
            gc,
            gc,
            kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0),
            groups=gc,
        )
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hwd, x_wd, x_hd = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hwd(x_hwd), self.dwconv_wd(x_wd), self.dwconv_hd(x_hd)),
            dim=1,
        )
