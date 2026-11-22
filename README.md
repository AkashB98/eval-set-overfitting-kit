# eval-set-overfitting-kit

**What this proves about Akash:** he can measure a failure mode most teams only hand-wave about — an eval set that gets *overfit* by the very iteration loop meant to improve the system — and he can show a principled mitigation working, with the costs stated honestly.

## The problem

Teams iterate on an agent by trying a change, scoring it on the dev eval set, and keeping it if the score goes up. Reuse that same small eval set for hundreds of keep/revert decisions and the loop starts fitting the set's sampling noise, not the system: the dev score climbs while real quality stalls. This is the **winner's curse** — plain-English: when you pick the best of many noisy scores, the winner's score is inflated by luck.

## What this kit does

- A configuration is 20 binary **knobs** — plain-English: on/off settings. True quality depends on only 3 of them; the other 17 do nothing.
- A dev score is the true score plus sampling noise sized by the dev-set size. The noise is a deterministic function of (seed, configuration), so re-scoring the same configuration on the same dev set returns the same number — exactly the reuse that creates the trap. Nothing about the gap is planted: it emerges from the generator.
- A greedy **hill-climb** — plain-English: flip one knob at a time, keep the flip if the dev score improves — runs 3 passes (61 dev evaluations).
- Held-out scores use fresh noise draws, so the **dev-vs-held-out gap** — plain-English: how much the dev score overstates real quality — is measured, not asserted.
- The mitigation is a simplified **Thresholdout** (Dwork et al., 2015) — plain-English: a reusable holdout that answers "is this change really better?" with a noisy check against a threshold, so the holdout itself does not get overfit. Each check spends one query from a fixed budget; when the budget runs out, the guarantee runs out and the climb stops accepting flips.

## Headline results (20 seeds, mean gap with 95% CI)

| Dev-set size | Mean dev-vs-held-out gap |
|---|---|
| 25 cases | +0.1929 [0.1691, 0.2166] |
| 100 cases | +0.1061 [0.0920, 0.1201] |
| 400 cases | +0.0420 [0.0373, 0.0467] |

The gap shrinks as the dev set grows — a test asserts this, so if it ever stops holding, the code has a bug.

Thresholdout at 100 dev cases: mean gap +0.1061 → −0.0023, and mean true quality 0.7010 → 0.8000 (the maximum), using a mean of 60 of 200 allowed holdout queries. Honest note: on a single seed at 25 dev cases the mitigated climb's final dev score can still read high (+0.1434 on seed 0) — the mitigation controls what gets *accepted*, and the headline claims are the 20-seed means, not any single run. Thresholdout also has a real cost beyond queries: a change whose true gain is smaller than the threshold tau (0.02) will usually be rejected even when it is real.

## POC framing

This is the acceptance-criteria problem from the customer's side: an eval suite that has been iterated against for months stops being evidence. The kit is a small, runnable demonstration of how to detect that (dev-vs-held-out gap, tracked as you iterate) and one published mitigation, with its budget and limits stated.

## Run it

```bash
python3 demo.py                      # headline demo
python3 -m eval_overfitting.cli report   # full experiment as JSON
python3 -m eval_overfitting.cli check    # exit 0 when the headline claims hold
python3 -m unittest discover -s tests
python3 evals/run_evals.py           # writes evals/eval_report.json (6/6 checks)
```

## Scope and limits

- Configurations are abstract on purpose — no RAG or agent-framework framing; the statistics are the point.
- The noise model is Gaussian with a per-case sd of 0.49 (the Bernoulli sd near p = 0.65); real eval noise is messier and often correlated across cases, which would make overfitting *worse*, not better.
- Thresholdout here is a simplified teaching version (Gaussian noise, fixed threshold), not the full adaptive-data-analysis machinery of Dwork et al.
- Other mitigations (a fresh holdout per round, nested validation) are future work, noted here rather than compared against.
- Motivation: AutoRAGBench (arXiv:2611.25582, verified) shows agents now iterate on RAG systems automatically — automated iteration reuses eval sets far faster than a human team does, which makes this failure mode more urgent, not less.
- All data is synthetic and generated in code. No personal data, no API keys, standard library only.

## Layout

- `eval_overfitting/core.py` — truth model, deterministic dev/held-out scoring, hill-climb, Thresholdout climb, experiment runner
- `eval_overfitting/cli.py` — demo / report / check
- `tests/test_core.py` — 8 tests, including the gap-shrinks-with-dev-size assertion
- `evals/run_evals.py` — 6 golden checks over the 20-seed experiment
