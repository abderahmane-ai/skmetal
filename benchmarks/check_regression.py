"""Run benchmarks and compare against stored baseline.
Exits with code 1 if any speedup regresses beyond threshold.

Usage:
    python benchmarks/check_regression.py                  # check Metal bridge
    python benchmarks/check_regression.py --mlx             # check MLX backends
    python benchmarks/check_regression.py --update          # update baseline
    python benchmarks/check_regression.py --mlx --update    # update MLX baseline
"""
import json
import sys
from pathlib import Path

from benchmark_suite import BENCHMARKS, benchmark

THRESHOLD = 0.80
use_mlx = "--mlx" in sys.argv
update = "--update" in sys.argv

if use_mlx:
    from skmetal.estimators._mlx_registry import has_mlx
    if not has_mlx():
        print("MLX not installed — skipping MLX regression check")
        sys.exit(0)
    from benchmark_mlx import MLX_BENCHMARKS as BENCHMARKS, benchmark  # noqa: F811

baseline_path = Path(__file__).parent / ("baseline_mlx.json" if use_mlx else "baseline.json")


def load_baseline():
    if not baseline_path.exists():
        return {}
    with open(baseline_path) as f:
        return {b["estimator"]: b for b in json.load(f)}


def main():
    baseline = load_baseline()
    results = []
    failed = False

    print(f"{'Estimator':<24} {'Speedup':>9} {'Baseline':>9} {'Ratio':>7}  Status")
    print("-" * 60)

    for name, est, df in BENCHMARKS:
        r = benchmark(name, est, df)
        results.append(r)
        speedup = r["speedup"]

        if name not in baseline:
            print(f"{name:<24} {speedup:>7.2f}x {'—':>9} {'—':>7}  NEW")
            continue

        base_speedup = baseline[name]["speedup"]
        ratio = speedup / base_speedup if base_speedup > 0 else 0.0
        status = "PASS" if ratio >= THRESHOLD else "FAIL"
        if ratio < THRESHOLD:
            failed = True
        print(f"{name:<24} {speedup:>7.2f}x {base_speedup:>7.2f}x {ratio:>6.0%}  {status}")

    if update:
        with open(baseline_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBaseline updated ({len(results)} estimators)")

    if failed:
        print("\nREGRESSION DETECTED: some estimators dropped below "
              f"{THRESHOLD:.0%} of baseline speedup.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
