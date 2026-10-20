"""Latency regression gates. Absolute checks only run on the machine that recorded the baseline."""
import json

import pytest

from bench_host import bench
from common import BENCH_DIR, Variant, environment
from conftest import require

TOLERANCE = 1.20
BASELINE = BENCH_DIR / "baseline_host.json"


def test_int8_not_slower_than_fp32():
    fp32, int8 = Variant(320, "fp32"), Variant(320, "int8")
    require(fp32, int8)
    f, i = bench(fp32, 10, 60, 4), bench(int8, 10, 60, 4)
    assert i["p50_ms"] <= f["p50_ms"] * 1.15, f"int8 p50 {i['p50_ms']} ms vs fp32 {f['p50_ms']} ms"


def test_p95_within_baseline_tolerance():
    baseline = json.loads(BASELINE.read_text())
    env = environment()
    if (baseline["environment"]["machine"], baseline["environment"]["system"]) != (env["machine"], env["system"]):
        pytest.skip("baseline recorded on different hardware")
    for row in baseline["results"]:
        if row["resolution"] != 320:
            continue
        v = Variant(row["resolution"], row["precision"])
        require(v)
        now = bench(v, 10, 100, baseline["config"]["threads"])
        assert now["p95_ms"] <= row["p95_ms"] * TOLERANCE, f"{v.name}: p95 {now['p95_ms']} vs baseline {row['p95_ms']}"
