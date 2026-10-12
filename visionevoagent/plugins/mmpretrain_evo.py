"""MMPretrain integration: ResNet_CIFAR with searchable module insertion + fusion neck.

Imported through `custom_imports` in the generated config, so the official training script
(tools/train.py) and recipe are used unchanged.
"""
from __future__ import annotations

import torch.nn as nn
from mmengine.model.weight_init import constant_init
from mmpretrain.models.backbones.resnet import Bottleneck
from mmpretrain.models.backbones.resnet_cifar import ResNet_CIFAR
from mmpretrain.registry import MODELS

from .modules import WeightedFusion, build_attention, replace_conv3x3

STAGES = {"early": (0, 1), "late": (2, 3), "all": (0, 1, 2, 3)}


@MODELS.register_module()
class EvoResNet_CIFAR(ResNet_CIFAR):
    """evo = dict(attention=..., conv=..., target_stage=...).

    attention: inserted inside every Bottleneck of the target stages, after norm3 and before the
               residual addition (SE-Net placement).
    conv:      replaces the 3x3 conv2 of every Bottleneck in the target stages.
    """

    def __init__(self, evo: dict, **kwargs):
        super().__init__(**kwargs)
        self.evo = dict(evo)
        stages = STAGES[evo.get("target_stage", "late")]
        for si in stages:
            layer = getattr(self, self.res_layers[si])
            for block in layer:
                if not isinstance(block, Bottleneck):
                    raise TypeError("EvoResNet_CIFAR only supports Bottleneck blocks (depth >= 50)")
                if evo.get("conv", "standard") != "standard":
                    block.conv2 = replace_conv3x3(block.conv2, evo["conv"])
                if evo.get("attention", "none") != "none":
                    norm3 = getattr(block, block.norm3_name)
                    attn = build_attention(evo["attention"], norm3.num_features)
                    setattr(block, block.norm3_name, nn.Sequential(norm3, attn))

    def init_weights(self):
        super().init_weights()
        # keep zero_init_residual identical to the baseline when norm3 was wrapped
        if self.zero_init_residual and not (isinstance(self.init_cfg, dict)
                                            and self.init_cfg.get("type") == "Pretrained"):
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    norm3 = getattr(m, m.norm3_name)
                    if isinstance(norm3, nn.Sequential):
                        constant_init(norm3[0], 0)


@MODELS.register_module()
class EvoWeightedFusionNeck(nn.Module):
    """Softmax-weighted fusion of the pooled outputs of the last stages (replaces GlobalAveragePooling)."""

    def __init__(self, in_channels=(1024, 2048), out_channels=2048):
        super().__init__()
        self.fusion = WeightedFusion(list(in_channels), out_channels)

    def init_weights(self):
        pass

    def forward(self, inputs):
        return (self.fusion(list(inputs)),)
