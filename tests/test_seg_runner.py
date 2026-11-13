"""Segmentation runner (2026-11-06): generated configs, the runner factory, and the classification config
unchanged by the refactoring. Offline (no MMSeg needed; the plugin itself is tested on the instance).

Usage (repo root): python tests/test_seg_runner.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.getcwd())

from visionevoagent.runners.factory import make_runner  # noqa: E402
from visionevoagent.runners.mmpretrain_runner import MMPretrainRunner  # noqa: E402
from visionevoagent.runners.mmseg_runner import MMSegRunner  # noqa: E402
from visionevoagent.search_space import BASELINE, all_architectures  # noqa: E402

d = tempfile.mkdtemp()
seg = make_runner("segmentation", {"mmseg_root": d, "iters": 8000}, seed=1, work_root=os.path.join(d, "w"))
assert isinstance(seg, MMSegRunner) and seg.root == os.path.abspath(d) and seg.cfg.seed == 1
assert len(all_architectures("segmentation")) == 86

base = seg.build_config(dict(BASELINE))
assert "type='EvoResNetV1c'" in base and "'attention': 'none'" in base
assert "EvoDepthwiseSeparableASPPHead" not in base                     # fusion none -> original head
assert "max_iters=8000, val_interval=8000" in base and "end=8000" in base
assert "AmpOptimWrapper" in base and "seed=1" in base and "batch_size=8" in base

cfg = seg.build_config({"attention": "eca", "conv": "dilated", "fusion": "weighted", "target_stage": "late"})
assert "'conv': 'dilated'" in cfg and "decode_head=dict(type='EvoDepthwiseSeparableASPPHead')" in cfg

seg.cfg.schedule = "official"
assert "max_iters=80000" in seg.build_config(dict(BASELINE))
assert seg._build_prelude().count("mmseg") == 3                         # plugin, registry, scope

cls = make_runner("classification", {"mmpretrain_root": d, "epochs": 60, "amp": True}, seed=0,
                  work_root=os.path.join(d, "c"))
assert isinstance(cls, MMPretrainRunner) and not isinstance(cls, MMSegRunner)
c = cls.build_config({"attention": "eca", "conv": "standard", "fusion": "none", "target_stage": "late"})
# the exact lines the classification score table was built with
assert "custom_imports = dict(imports=['visionevoagent.plugins.mmpretrain_evo'], allow_failed_imports=False)" in c
assert "train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=60)" in c
assert "optim_wrapper = dict(type='AmpOptimWrapper', optimizer=dict(lr=0.1))" in c
assert "visionevoagent.plugins.mmpretrain_evo" in cls._build_prelude() and "init_default_scope('mmpretrain')" in cls._build_prelude()

# task wording and metric of the segmentation searches (2026-11-12)
import json  # noqa: E402
from visionevoagent import search_space as ss  # noqa: E402
from visionevoagent.runners.table_runner import TableRunner  # noqa: E402
ss.activate("segmentation")
assert "decoder" in ss.MODULE_NOTES["weighted"] and "pretrained" in ss.MODULE_NOTES["depthwise"]
assert "Cityscapes" in ss.space_note()
ss.activate("classification")
assert "last two stages" in ss.MODULE_NOTES["weighted"] and ss.space_note() == ""
tab = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "score_table_seg_i4000.json")
t = TableRunner(tab, 0, metric="mIoU")
assert t.metric == "mIoU" and len(t.rows) == 86 and TableRunner(tab, 0).metric == "top1"
assert t.run(dict(BASELINE), "b").score == 69.4
assert all(r["status"] == "ok" for r in json.load(open(tab))["seeds"]["0"].values())

try:
    make_runner("nb201_cifar10", {}, seed=0)
    raise AssertionError("NB201 has no training runner")
except ValueError:
    pass
print("seg runner: all checks passed")
