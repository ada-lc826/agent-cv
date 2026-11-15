"""tools/paired_tests.py: exact Wilcoxon signed-rank against known values, Holm, bootstrap. Offline.

Usage (repo root): python tests/test_paired_tests.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "tools"))

from paired_tests import bootstrap_ci, holm, wilcoxon_exact  # noqa: E402

# all n differences in one direction: p = 2 / 2^n
p, n = wilcoxon_exact([1, 2, 3, 4, 5])
assert n == 5 and abs(p - 2 / 32) < 1e-12, p
p, n = wilcoxon_exact([-0.1] * 10)                      # all tied, all negative
assert n == 10 and abs(p - 2 / 1024) < 1e-12, p
# zeros are dropped
p, n = wilcoxon_exact([0, 0, 1, 2, 3])
assert n == 3 and abs(p - 2 / 8) < 1e-12, p
# symmetric differences: p = 1
p, _ = wilcoxon_exact([1, -1, 2, -2])
assert abs(p - 1.0) < 1e-12, p
# textbook example (n = 8, no ties): W+ = 3 -> two-sided p = 2 * P(W+ <= 3) = 2 * 5/256
p, _ = wilcoxon_exact([-1, -2, 3, -4, -5, -6, -7, -8])
assert abs(p - 2 * 5 / 256) < 1e-12, p
assert wilcoxon_exact([]) == (None, 0)
# Holm: sorted p * (m - k), monotone, None kept
assert holm([0.01, 0.04, None, 0.03]) == [0.03, 0.06, None, 0.06]
lo, hi = bootstrap_ci([1.0] * 7)
assert lo == hi == 1.0
lo, hi = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5])
assert 0.1 <= lo < 0.3 < hi <= 0.5
print("paired tests: all checks passed")
