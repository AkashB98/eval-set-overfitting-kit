"""Core model.

A configuration is 20 binary knobs. True quality depends on only 3 of them
(knobs 0, 1, 2); the other 17 do nothing. A dev score is the true score plus
sampling noise whose size is set by the dev-set size. The noise is a
deterministic function of (seed, config), so re-scoring the same config on
the same dev set returns the same number -- exactly the reuse that lets a
greedy hill-climber mistake noise for improvement (the winner's curse).
Held-out scores use fresh noise draws, so the dev-vs-held-out gap that
appears is a natural result of the setup, not a planted effect.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

N_KNOBS = 20
REAL_KNOBS = (0, 1, 2)
BASE_QUALITY = 0.62
KNOB_GAIN = 0.06
CASE_SD = 0.49  # sd of one Bernoulli-ish case outcome near p=0.65


def true_quality(config: tuple[int, ...]) -> float:
    return BASE_QUALITY + KNOB_GAIN * sum(config[k] for k in REAL_KNOBS)


def _noise(seed: int, stream: str, config: tuple[int, ...], sd: float) -> float:
    digest = hashlib.sha256(f"{seed}|{stream}|{config}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return rng.gauss(0.0, sd)


def dev_score(seed: int, config: tuple[int, ...], n_dev: int) -> float:
    sd = CASE_SD / math.sqrt(n_dev)
    return true_quality(config) + _noise(seed, "dev", config, sd)


def heldout_score(seed: int, config: tuple[int, ...], n_hold: int = 5000) -> float:
    sd = CASE_SD / math.sqrt(n_hold)
    return true_quality(config) + _noise(seed, "hold", config, sd)


@dataclass
class ClimbResult:
    config: tuple[int, ...]
    dev: float
    heldout: float
    true: float
    evaluations: int
    queries_used: int = 0
    budget_exhausted: bool = False

    @property
    def gap(self) -> float:
        return self.dev - self.heldout


def hill_climb(seed: int, n_dev: int, passes: int = 3) -> ClimbResult:
    """Greedy keep/revert over knobs in a seeded order, scored on the dev set."""
    rng = random.Random(seed * 7919 + n_dev)
    config = [0] * N_KNOBS
    best = dev_score(seed, tuple(config), n_dev)
    evals = 1
    for _ in range(passes):
        order = list(range(N_KNOBS))
        rng.shuffle(order)
        for k in order:
            cand = list(config)
            cand[k] = 1 - cand[k]
            score = dev_score(seed, tuple(cand), n_dev)
            evals += 1
            if score > best:
                config, best = cand, score
    final = tuple(config)
    return ClimbResult(final, best, heldout_score(seed, final), true_quality(final), evals)


def thresholdout_climb(
    seed: int,
    n_dev: int,
    passes: int = 3,
    tau: float = 0.02,
    noise_sd: float = 0.01,
    query_budget: int = 200,
    n_hold: int = 2000,
) -> ClimbResult:
    """Hill-climb where a flip is kept only if a reusable holdout confirms it.

    Simplified Thresholdout (Dwork et al. 2015): each candidate flip spends
    one holdout query. The holdout difference is released with Gaussian
    noise, and the flip is accepted only when the noisy difference clears
    the threshold tau. Once the query budget is spent, no further flips are
    accepted -- the guarantee has run out, and the climb stops improving.
    The dev score is still tracked so the (smaller) gap can be reported.
    """
    rng = random.Random(seed * 7919 + n_dev)
    config = [0] * N_KNOBS
    queries = 0
    exhausted = False
    for _ in range(passes):
        order = list(range(N_KNOBS))
        rng.shuffle(order)
        for k in order:
            if queries >= query_budget:
                exhausted = True
                break
            cand = list(config)
            cand[k] = 1 - cand[k]
            diff = heldout_score(seed, tuple(cand), n_hold) - heldout_score(seed, tuple(config), n_hold)
            noisy = diff + rng.gauss(0.0, noise_sd)
            queries += 1
            if noisy > tau:
                config = cand
        if exhausted:
            break
    final = tuple(config)
    return ClimbResult(
        final,
        dev_score(seed, final, n_dev),
        heldout_score(seed, final),
        true_quality(final),
        queries,
        queries_used=queries,
        budget_exhausted=exhausted,
    )


def run_experiment(seeds: range = range(20), dev_sizes: tuple[int, ...] = (25, 100, 400)) -> dict:
    """Mean dev-vs-held-out gap (with 95% CI) per dev-set size, plus the
    Thresholdout comparison at the middle dev-set size."""
    out: dict = {"dev_sizes": {}}
    for n in dev_sizes:
        gaps, trues = [], []
        for s in seeds:
            r = hill_climb(s, n)
            gaps.append(r.gap)
            trues.append(r.true)
        mean = sum(gaps) / len(gaps)
        sd = math.sqrt(sum((g - mean) ** 2 for g in gaps) / (len(gaps) - 1)) if len(gaps) > 1 else 0.0
        half = 1.96 * sd / math.sqrt(len(gaps))
        out["dev_sizes"][str(n)] = {
            "mean_gap": round(mean, 4),
            "ci95": [round(mean - half, 4), round(mean + half, 4)],
            "mean_true_quality": round(sum(trues) / len(trues), 4),
        }
    mid = dev_sizes[len(dev_sizes) // 2]
    plain_gaps, mit_gaps, plain_true, mit_true, queries = [], [], [], [], []
    for s in seeds:
        p = hill_climb(s, mid)
        m = thresholdout_climb(s, mid)
        plain_gaps.append(p.gap)
        mit_gaps.append(m.gap)
        plain_true.append(p.true)
        mit_true.append(m.true)
        queries.append(m.queries_used)
    out["thresholdout"] = {
        "n_dev": mid,
        "plain_mean_gap": round(sum(plain_gaps) / len(plain_gaps), 4),
        "thresholdout_mean_gap": round(sum(mit_gaps) / len(mit_gaps), 4),
        "plain_mean_true_quality": round(sum(plain_true) / len(plain_true), 4),
        "thresholdout_mean_true_quality": round(sum(mit_true) / len(mit_true), 4),
        "mean_queries_used": round(sum(queries) / len(queries), 1),
        "query_budget": 200,
    }
    return out
