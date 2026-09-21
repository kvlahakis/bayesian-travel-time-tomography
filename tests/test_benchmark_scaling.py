"""Sanity check for scripts/benchmark_scaling.py's reusable `benchmark_one`.

Loads the script as a module (via importlib, not a `sys.path`/`PYTHONPATH`
change) and calls its per-size measurement function directly at a tiny grid
size, checking the returned quantities are sane -- shapes/counts are exact
and checked; timings are not checked against a specific threshold, since
wall-clock time is hardware-dependent and this would be a flaky test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_scaling.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_scaling", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_one_returns_sane_measurements():
    benchmark_scaling = _load_benchmark_module()
    n = 6
    result = benchmark_scaling.benchmark_one(n)

    assert result["n"] == n
    assert result["n_cells"] == n**2
    assert result["m_rays"] == benchmark_scaling.NS * benchmark_scaling.NR
    assert result["A_nonzeros"] > 0
    assert 0.0 < result["A_density"] <= 1.0
    assert result["A_bytes"] > 0
    assert result["Cs_bytes"] > 0
    assert result["C_post_bytes"] > 0
    assert result["forward_matrix_time_s"] >= 0.0
    assert result["posterior_solve_time_s"] >= 0.0
