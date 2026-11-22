"""CLI: demo / report / check."""
from __future__ import annotations

import json
import sys

from .core import hill_climb, run_experiment, thresholdout_climb, true_quality


def _report() -> dict:
    return run_experiment()


def cmd_demo() -> int:
    r = hill_climb(0, 25)
    print(f"One climb, seed 0, dev set of 25 cases:")
    print(f"  knobs on: {[i for i, v in enumerate(r.config) if v]} (only 0, 1, 2 truly help)")
    print(f"  dev score {r.dev:.4f} vs held-out {r.heldout:.4f} vs true {r.true:.4f}")
    print(f"  gap {r.gap:+.4f} after {r.evaluations} dev evaluations")
    m = thresholdout_climb(0, 25)
    print(f"Same seed with Thresholdout: gap {m.gap:+.4f}, true {m.true:.4f}, "
          f"queries {m.queries_used}/200")
    rep = _report()
    for n, row in rep["dev_sizes"].items():
        print(f"  n_dev={n}: mean gap {row['mean_gap']:+.4f} CI95 {row['ci95']}")
    t = rep["thresholdout"]
    print(f"  Thresholdout at n_dev={t['n_dev']}: gap {t['plain_mean_gap']:+.4f} -> "
          f"{t['thresholdout_mean_gap']:+.4f}; true quality {t['plain_mean_true_quality']:.4f} -> "
          f"{t['thresholdout_mean_true_quality']:.4f}")
    return 0


def cmd_report() -> int:
    print(json.dumps(_report(), indent=2))
    return 0


def cmd_check() -> int:
    """Exit 0 when the headline claims hold on the seeded experiment, else 1."""
    rep = _report()
    sizes = rep["dev_sizes"]
    ok = sizes["25"]["mean_gap"] > sizes["400"]["mean_gap"] > 0
    t = rep["thresholdout"]
    ok = ok and t["thresholdout_mean_gap"] < t["plain_mean_gap"]
    print("PASS" if ok else "FAIL", json.dumps(rep))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return cmd_demo()
    return {"demo": cmd_demo, "report": cmd_report, "check": cmd_check}.get(argv[0], cmd_demo)()


if __name__ == "__main__":
    raise SystemExit(main())
