"""MMSegmentation integration (2026-11-06): DeepLabV3+ R-50-D8 with the module space of the classification
experiments, for transferring principles across tasks.

Same dimensions as the classification space, adapted to an ImageNet-pretrained backbone:
  attention     after norm3 of every Bottleneck in the target stages, before the residual addition (as in
                classification). Attached with a forward hook so that the parameter names of norm3 do not
                change and its pretrained weights still load.
  conv          dilated: the existing 3x3 conv2 keeps its weights and doubles its dilation (R-50-D8 already
                uses dilation 2 / 4 in stages 3 / 4, so "dilated" there means 4 / 8);
                depthwise: conv2 is replaced by depthwise 3x3 + pointwise 1x1 (same stride and dilation);
                these weights are new, the pretrained conv2 cannot be reused.
  target_stage  early = stages 1-2, late = stages 3-4, all = 1-4 (as in classification).
  fusion        weighted: learnable softmax weights on the two decoder inputs (upsampled ASPP output and the
                projected low-level feature) before they are concatenated; equal weights = the original head.

Imported through `custom_imports`, so MMSeg's tools/train.py and the official recipe are used unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from mmseg.models.backbones.resnet import Bottleneck, ResNetV1c
from mmseg.models.decode_heads.sep_aspp_head import DepthwiseSeparableASPPHead
from mmseg.models.utils import resize
from mmseg.registry import MODELS

from .modules import build_attention

STAGES = {"early": (0, 1), "late": (2, 3), "all": (0, 1, 2, 3)}


def _attach_attention(block: Bottleneck, name: str) -> None:
    norm3 = getattr(block, block.norm3_name)
    block.evo_attn = build_attention(name, norm3.num_features)   # registered as a submodule (trained, saved)
    norm3.register_forward_hook(lambda _m, _inp, out, blk=block: blk.evo_attn(out))


def _dilate(conv: nn.Conv2d) -> None:
    d = conv.dilation[0] * 2
    conv.dilation, conv.padding = (d, d), (d, d)


def _depthwise(conv: nn.Conv2d) -> nn.Module:
    cin, cout, d = conv.in_channels, conv.out_channels, conv.dilation[0]
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride=conv.stride, padding=d, dilation=d, groups=cin, bias=False),
        nn.Conv2d(cin, cout, 1, bias=False))


@MODELS.register_module()
class EvoResNetV1c(ResNetV1c):
    """ResNetV1c with evo = dict(attention=..., conv=..., target_stage=...)."""

    def __init__(self, evo: dict, **kwargs):
        super().__init__(**kwargs)
        self.evo = dict(evo)
        for si in STAGES[evo.get("target_stage", "late")]:
            for block in getattr(self, self.res_layers[si]):
                if not isinstance(block, Bottleneck):
                    raise TypeError("EvoResNetV1c only supports Bottleneck blocks (depth >= 50)")
                if evo.get("conv", "standard") == "dilated":
                    _dilate(block.conv2)
                elif evo.get("conv", "standard") == "depthwise":
                    block.conv2 = _depthwise(block.conv2)
                if evo.get("attention", "none") != "none":
                    _attach_attention(block, evo["attention"])


@MODELS.register_module()
class EvoDepthwiseSeparableASPPHead(DepthwiseSeparableASPPHead):
    """DeepLabV3+ head whose two decoder inputs are mixed with learnable softmax weights (x 2, so that equal
    weights reproduce the original head) before concatenation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fusion_w = nn.Parameter(torch.zeros(2))

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        aspp_outs = [resize(self.image_pool(x), size=x.size()[2:], mode='bilinear',
                            align_corners=self.align_corners)]
        aspp_outs.extend(self.aspp_modules(x))
        output = self.bottleneck(torch.cat(aspp_outs, dim=1))
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(input=output, size=c1_output.shape[2:], mode='bilinear',
                            align_corners=self.align_corners)
            w = 2 * torch.softmax(self.fusion_w, 0)
            output = torch.cat([w[0] * output, w[1] * c1_output], dim=1)
        output = self.sep_bottleneck(output)
        return self.cls_seg(output)
