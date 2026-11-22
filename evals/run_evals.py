"""Golden evals: the headline claims, checked against the seeded experiment."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_overfitting.core import run_experiment


def main() -> int:
    rep = run_experiment()
    sizes = rep["dev_sizes"]
    t = rep["thresholdout"]
    checks = {
        "gap_positive_small_dev": sizes["25"]["mean_gap"] > 0.02,
        "gap_shrinks_with_dev_size": sizes["25"]["mean_gap"] > sizes["100"]["mean_gap"] > sizes["400"]["mean_gap"],
        "gap_ci_excludes_zero_small_dev": sizes["25"]["ci95"][0] > 0,
        "thresholdout_shrinks_gap": t["thresholdout_mean_gap"] < t["plain_mean_gap"],
        "thresholdout_keeps_most_true_quality": t["thresholdout_mean_true_quality"] >= t["plain_mean_true_quality"] - 0.06,
        "queries_within_budget": t["mean_queries_used"] <= t["query_budget"],
    }
    report = {"checks": checks, "passed": sum(checks.values()), "total": len(checks), "experiment": rep}
    out = Path(__file__).with_name("eval_report.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "total": report["total"], **checks}, indent=2))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
