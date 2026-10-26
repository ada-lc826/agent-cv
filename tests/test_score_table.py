"""Score table (2026-10-26): TableRunner answers from the table, a missing entry is a setup error, and
build_score_table imports only earlier trainings with the same proxy schedule and reports conflicts.
Offline.

Usage (repo root): python tests/test_score_table.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "tools"))

from build_score_table import import_results  # noqa: E402
from visionevoagent.runners.table_runner import ScoreTableMissing, TableRunner  # noqa: E402
from visionevoagent.search_space import BASELINE, architecture_signature as sig  # noqa: E402

ECA = {**BASELINE, "attention": "eca"}
d = tempfile.mkdtemp()
table = {"meta": {}, "seeds": {"0": {sig(BASELINE): {"status": "ok", "score": 78.06, "params": 23.7, "duration_s": 900},
                                     sig(ECA): {"status": "failed", "failure_reason": "oom", "params": 23.7}}}}
path = os.path.join(d, "t.json")
json.dump(table, open(path, "w"))

r = TableRunner(path, 0)
res = r.run(BASELINE, "r000")
assert res.status == "ok" and res.score == 78.06 and res.duration_s == 900
assert r.run(ECA, "r001").failure_reason == "oom"
try:
    r.run({**BASELINE, "attention": "se"}, "r002")
    raise AssertionError("missing entry must raise")
except ScoreTableMissing:
    pass
assert r.param_table("classification") == {sig(BASELINE): 23.7, sig(ECA): 23.7}
try:
    TableRunner(path, 1)
    raise AssertionError("missing seed must raise")
except ScoreTableMissing:
    pass


def trial(name, seed, epochs, amp, score):
    t = os.path.join(d, "runs", name)
    os.makedirs(t)
    cfg = f"train_cfg = dict(by_epoch=True, max_epochs={epochs}, val_interval={epochs})\n"
    cfg += ("optim_wrapper = dict(type='AmpOptimWrapper', optimizer=dict(lr=0.1))\n" if amp else "")
    cfg += f"randomness = dict(deterministic=False, seed={seed})\n"  # the order mmengine writes back
    open(os.path.join(t, "trial_config.py"), "w").write(cfg)
    json.dump({"status": "ok", "score": score, "params": 23.7, "duration_s": 800}, open(os.path.join(t, "result.json"), "w"))


trial("r001_eca__standard__none__late", 0, 60, True, 78.52)        # imported
trial("zz_r003_eca__standard__none__late", 0, 60, True, 78.40)  # same key, different score -> warning
trial("r002_se__standard__none__late", 1, 60, True, 78.2)          # seed 1
trial("r004_cbam__standard__none__late", 0, 30, True, 76.0)        # 30-epoch proxy: skipped
trial("r005_coordatt__standard__none__late", 0, 60, False, 76.0)   # fp32: skipped
t2 = {"meta": {}, "seeds": {}}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    n = import_results([os.path.join(d, "runs", "*")], 60, True, t2)
assert n == 2, n
assert t2["seeds"]["0"][sig(ECA)]["score"] == 78.52 and t2["seeds"]["1"][sig({**BASELINE, "attention": "se"})]["score"] == 78.2
assert sig({**BASELINE, "attention": "cbam"}) not in t2["seeds"]["0"]
assert "WARNING" in buf.getvalue()
print("score table: all checks passed")
