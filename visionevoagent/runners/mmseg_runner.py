"""Segmentation runner (2026-11-06): DeepLabV3+ R-50-D8 on Cityscapes (MMSegmentation official recipe).

Same mechanics as MMPretrainRunner (one small config per candidate that inherits the official one, the
framework's tools/train.py in a subprocess, result.json cache); only the config and the metric differ.
Score = mIoU on Cityscapes val after training. The official recipe is 80k iterations at batch 8 (4 GPUs x 2);
here the total batch of 8 is kept on one GPU. The proxy shortens the iterations (poly LR over the shorter
schedule) and evaluates once at the end.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List

from ..search_space import Architecture, canonicalize
from .mmpretrain_runner import MMPretrainRunner


@dataclass
class MMSegRunnerConfig:
    mmseg_root: str                                        # checkout containing tools/train.py (+ data/ link)
    base_config: str = "configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024.py"
    work_root: str = "work_dirs/search_seg"
    python: str = sys.executable
    schedule: str = "proxy"                                # proxy | official
    iters: int = 8000                                      # proxy length (official: 80000)
    batch_size: int = 8                                    # total batch of the official recipe
    seed: int = 0
    amp: bool = True
    device: str = ""                                       # "" = auto; "cpu" for smoke tests
    num_workers: int = 4
    timeout_s: float = 12 * 3600
    keep_checkpoints: bool = False
    extra_cfg: Dict[str, object] = field(default_factory=dict)


class MMSegRunner(MMPretrainRunner):
    metric = "mIoU"
    PLUGIN = "visionevoagent.plugins.mmseg_evo"
    SCOPE = "mmseg"
    METRIC_KEY = "mIoU"

    def __init__(self, cfg: MMSegRunnerConfig):  # noqa: D107 -- the parent's __init__ expects another config
        self.cfg = cfg
        self.cfg.mmseg_root = os.path.abspath(os.path.expanduser(cfg.mmseg_root))
        self.cfg.work_root = os.path.abspath(os.path.expanduser(cfg.work_root))

    @property
    def root(self) -> str:
        return self.cfg.mmseg_root

    def build_config(self, arch: Architecture) -> str:
        c, a = self.cfg, canonicalize(arch)
        base = os.path.join(c.mmseg_root, c.base_config).replace("\\", "/")
        evo = {"attention": a["attention"], "conv": a["conv"], "target_stage": a["target_stage"]}
        head = ", decode_head=dict(type='EvoDepthwiseSeparableASPPHead')" if a["fusion"] == "weighted" else ""
        n = 80000 if c.schedule == "official" else c.iters
        lines: List[str] = [
            f"_base_ = [{base!r}]",
            "custom_imports = dict(imports=['visionevoagent.plugins.mmseg_evo'], allow_failed_imports=False)",
            f"model = dict(backbone=dict(type='EvoResNetV1c', evo={evo!r}){head})",
            f"train_cfg = dict(type='IterBasedTrainLoop', max_iters={n}, val_interval={n})",
            f"param_scheduler = [dict(type='PolyLR', eta_min=1e-4, power=0.9, begin=0, end={n}, by_epoch=False)]",
            f"train_dataloader = dict(batch_size={c.batch_size}, num_workers={c.num_workers})",
            f"randomness = dict(seed={c.seed}, deterministic=False)",
            f"default_hooks = dict(checkpoint=dict(by_epoch=False, interval={n}, max_keep_ckpts=1), "
            "logger=dict(interval=50))",
        ]
        if c.amp:
            lines.append("optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')")
        for k, v in c.extra_cfg.items():
            lines.append(f"{k} = {v!r}")
        return "\n".join(lines) + "\n"
