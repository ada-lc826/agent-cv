"""Classification runner: CIFAR-100 ResNet-50 (MMPretrain official recipe) under a proxy schedule.

For each candidate it writes a small config that inherits the official config, swaps in
EvoResNet_CIFAR (+ fusion neck), shortens the schedule, and runs tools/train.py in a subprocess.
A finished trial leaves result.json in its work dir, so re-running the same tag is free (resume).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..search_space import Architecture, all_architectures, architecture_signature, canonicalize
from .base import BaseRunner, TrialResult

log = logging.getLogger(__name__)

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class MMPretrainRunnerConfig:
    mmpretrain_root: str                                   # checkout containing tools/train.py
    base_config: str = "configs/resnet/resnet50_8xb16_cifar100.py"
    work_root: str = "work_dirs/search"
    python: str = sys.executable
    schedule: str = "proxy"                                # proxy | official (full recipe, for the proxy check)
    epochs: int = 30                                       # proxy schedule length
    warmup_epochs: int = 1
    batch_size: int = 128                                  # 8 GPUs x 16 in the official recipe
    lr: float = 0.1
    seed: int = 0
    amp: bool = False
    device: str = ""                                       # "" = auto; "cpu" for smoke tests
    data_root: str = ""                                    # "" = keep the base config's data_root
    train_subset: int = 0                                  # >0: use only the first N training images
    val_subset: int = 0
    num_workers: int = 4
    timeout_s: float = 6 * 3600
    keep_checkpoints: bool = False                         # proxy checkpoints are ~100 MB each
    extra_cfg: Dict[str, object] = field(default_factory=dict)


class MMPretrainRunner(BaseRunner):
    metric = "top1"

    def __init__(self, cfg: MMPretrainRunnerConfig):
        self.cfg = cfg
        self.cfg.mmpretrain_root = os.path.abspath(os.path.expanduser(cfg.mmpretrain_root))
        self.cfg.work_root = os.path.abspath(os.path.expanduser(cfg.work_root))

    # ------------------------------------------------------------------
    def build_config(self, arch: Architecture) -> str:
        c, a = self.cfg, canonicalize(arch)
        base = os.path.join(c.mmpretrain_root, c.base_config).replace("\\", "/")
        evo = {"attention": a["attention"], "conv": a["conv"], "target_stage": a["target_stage"]}
        lines: List[str] = [
            f"_base_ = [{base!r}]",
            "custom_imports = dict(imports=['visionevoagent.plugins.mmpretrain_evo'], allow_failed_imports=False)",
        ]
        if a["fusion"] == "weighted":
            lines.append(f"model = dict(backbone=dict(type='EvoResNet_CIFAR', evo={evo!r}, out_indices=(2, 3)), "
                         "neck=dict(_delete_=True, type='EvoWeightedFusionNeck', in_channels=(1024, 2048), "
                         "out_channels=2048))")
        else:
            lines.append(f"model = dict(backbone=dict(type='EvoResNet_CIFAR', evo={evo!r}))")
        w, e = c.warmup_epochs, c.epochs
        if c.schedule == "official":
            # The reproduction's full recipe: the base config's own epochs / LR schedule / optimizer
            # (CIFAR-100 ResNet-50: SGD lr 0.1, 200 epochs, multi-step), total batch kept on one GPU.
            lines += [
                f"randomness = dict(seed={c.seed}, deterministic=False)",
                "default_hooks = dict(checkpoint=dict(max_keep_ckpts=1, save_last=True), logger=dict(interval=50))",
                "auto_scale_lr = dict(enable=False)",
            ]
            if c.amp:
                lines.append("optim_wrapper = dict(type='AmpOptimWrapper')")
        else:
            lines += [
                f"train_cfg = dict(by_epoch=True, max_epochs={e}, val_interval={e})",
                f"optim_wrapper = dict({'type=' + repr('AmpOptimWrapper') + ', ' if c.amp else ''}"
                f"optimizer=dict(lr={c.lr}))",
                "param_scheduler = ["
                + (f"dict(type='LinearLR', start_factor=0.01, by_epoch=True, begin=0, end={w}, "
                   "convert_to_iter_based=True), " if w > 0 else "")
                + f"dict(type='CosineAnnealingLR', T_max={e - w}, eta_min=0.0, by_epoch=True, begin={w}, end={e}, "
                "convert_to_iter_based=True)]",
                f"randomness = dict(seed={c.seed}, deterministic=False)",
                f"default_hooks = dict(checkpoint=dict(interval={e}, max_keep_ckpts=1, save_last=True), "
                "logger=dict(interval=50))",
                "auto_scale_lr = dict(enable=False)",
            ]
        for split, n, bs in (("train", c.train_subset, c.batch_size), ("val", c.val_subset, c.batch_size)):
            ds = []
            if c.data_root:
                ds.append(f"data_root={c.data_root.replace(chr(92), '/')!r}")
            if n:
                ds.append(f"indices={n}")
            ds_str = f", dataset=dict({', '.join(ds)})" if ds else ""
            lines.append(f"{split}_dataloader = dict(batch_size={bs}, num_workers={c.num_workers}{ds_str})")
        if c.val_subset or c.data_root:
            lines.append("test_dataloader = val_dataloader")
        for k, v in c.extra_cfg.items():
            lines.append(f"{k} = {v!r}")
        return "\n".join(lines) + "\n"

    def _env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in (PKG_ROOT, self.cfg.mmpretrain_root,
                                                        env.get("PYTHONPATH", "")) if p)
        env.pop("MPLBACKEND", None)
        if self.cfg.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        return env

    # ------------------------------------------------------------------
    def run(self, arch: Architecture, tag: str) -> TrialResult:
        work_dir = os.path.join(self.cfg.work_root, tag)
        result_path = os.path.join(work_dir, "result.json")
        if os.path.exists(result_path):  # finished earlier (resume)
            with open(result_path, encoding="utf-8") as f:
                return TrialResult(**json.load(f))
        os.makedirs(work_dir, exist_ok=True)
        cfg_path = os.path.join(work_dir, "trial_config.py")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(self.build_config(arch))

        t0 = time.time()
        res = self._run_inner(cfg_path, work_dir)
        res.duration_s = round(time.time() - t0, 1)
        res.work_dir = work_dir
        if res.status == "ok":  # only cache successes; failures are retried after a fix
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(res.__dict__, f, indent=2)
            if not self.cfg.keep_checkpoints:
                for name in os.listdir(work_dir):
                    if name.endswith(".pth"):
                        os.remove(os.path.join(work_dir, name))
        return res

    def _run_inner(self, cfg_path: str, work_dir: str) -> TrialResult:
        try:
            params = self.count_params(cfg_path)
        except Exception as e:  # noqa: BLE001 -- runner must never raise
            log.warning("model build failed: %s", str(e)[-1000:])
            return TrialResult("failed", failure_reason="build_error")
        cmd = [self.cfg.python, os.path.join(self.cfg.mmpretrain_root, "tools", "train.py"), cfg_path,
               "--work-dir", work_dir]
        log_path = os.path.join(work_dir, "train_stdout.log")
        try:
            with open(log_path, "w", encoding="utf-8") as out:
                proc = subprocess.run(cmd, cwd=self.cfg.mmpretrain_root, env=self._env(), stdout=out,
                                      stderr=subprocess.STDOUT, timeout=self.cfg.timeout_s)
        except subprocess.TimeoutExpired:
            return TrialResult("failed", params=params, failure_reason="timeout")
        except OSError as e:
            return TrialResult("failed", params=params, failure_reason=f"crash: {e}")
        tail = _tail(log_path)
        if proc.returncode != 0:
            low = tail.lower()
            reason = "oom" if "out of memory" in low else ("nan" if "nan" in low and "loss" in low else "crash")
            return TrialResult("failed", params=params, failure_reason=reason)
        score = parse_top1(work_dir)
        if score is None:
            return TrialResult("failed", params=params, failure_reason="missing_result")
        if score != score:  # NaN
            return TrialResult("failed", params=params, failure_reason="nan")
        return TrialResult("ok", score=round(score, 2), params=params)

    def count_params(self, cfg_path: str) -> float:
        """Build the model in a subprocess (same env as training) and return parameters in millions."""
        code = ("import sys, json\n"
                "from mmengine.config import Config\n"
                "from mmengine.registry import init_default_scope\n"
                "import visionevoagent.plugins.mmpretrain_evo\n"
                "from mmpretrain.registry import MODELS\n"
                "init_default_scope('mmpretrain')\n"
                "m = MODELS.build(Config.fromfile(sys.argv[1]).model)\n"
                "print(json.dumps({'params': sum(p.numel() for p in m.parameters()) / 1e6}))\n")
        p = subprocess.run([self.cfg.python, "-c", code, cfg_path], cwd=self.cfg.mmpretrain_root,
                           env=self._env(), capture_output=True, text=True, timeout=600)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-2000:])
        return round(json.loads(p.stdout.strip().splitlines()[-1])["params"], 3)


    def param_table(self, task: str) -> Optional[Dict[str, float]]:
        """Build every architecture once (one subprocess, CPU) and cache the counts in
        <work_root>/param_table.json. Used for the Design Agent's cost table, never for scores."""
        cache = os.path.join(self.cfg.work_root, "param_table.json")
        if os.path.exists(cache):
            with open(cache, encoding="utf-8") as f:
                return json.load(f)
        cfg_dir = os.path.join(self.cfg.work_root, "_param_configs")
        os.makedirs(cfg_dir, exist_ok=True)
        paths = {}
        for a in all_architectures(task):
            sig = architecture_signature(a)
            paths[sig] = os.path.join(cfg_dir, f"{sig}.py")
            with open(paths[sig], "w", encoding="utf-8") as f:
                f.write(self.build_config(a))
        code = ("import sys, json\n"
                "from mmengine.config import Config\n"
                "from mmengine.registry import init_default_scope\n"
                "import visionevoagent.plugins.mmpretrain_evo\n"
                "from mmpretrain.registry import MODELS\n"
                "init_default_scope('mmpretrain')\n"
                "out = {}\n"
                "for sig, path in json.loads(sys.argv[1]).items():\n"
                "    m = MODELS.build(Config.fromfile(path).model)\n"
                "    out[sig] = round(sum(p.numel() for p in m.parameters()) / 1e6, 3)\n"
                "print(json.dumps(out))\n")
        env = self._env()
        env["CUDA_VISIBLE_DEVICES"] = ""  # counting only; never take a GPU from a running trial
        try:
            p = subprocess.run([self.cfg.python, "-c", code, json.dumps(paths)], cwd=self.cfg.mmpretrain_root,
                               env=env, capture_output=True, text=True, timeout=1800)
            if p.returncode != 0:
                raise RuntimeError(p.stderr[-2000:])
            table = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception as e:  # noqa: BLE001 -- the cost table is optional
            log.warning("param table unavailable, agents run without it: %s", str(e)[-500:])
            return None
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(table, f, indent=2)
        return table


def parse_top1(work_dir: str) -> Optional[float]:
    """Last accuracy/top1 written by mmengine to <work_dir>/<timestamp>/vis_data/scalars.json."""
    best: Optional[float] = None
    stamps = sorted(d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d)))
    for d in stamps:
        path = os.path.join(work_dir, d, "vis_data", "scalars.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "accuracy/top1" in rec:
                    best = float(rec["accuracy/top1"])
    return best


def _tail(path: str, n: int = 8000) -> str:
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - n))
        return f.read().decode("utf-8", "replace")
