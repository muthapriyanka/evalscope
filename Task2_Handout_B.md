# Task 2 Handout B: Why This Matters And How To Use It

## What Changes

The pruned benchmarks turn a slow "run everything and wait" evaluation into a fast screening step. Sales and deployment teams can ask whether a candidate model is probably good enough for coding and long-context workloads before spending the time and cost of the full benchmark suite.

This does not replace the full benchmark. It decides when a full run is worth doing.

## How To Run

In the EvalScope fork:

```bash
evalscope eval --model <model> --datasets live_code_bench --output ./results_full/
evalscope eval --model <model> --datasets live_code_bench_pruned \
    --dataset-args '{"pruning_strategy": "calibrated_coverage", "prune_ratio": 0.1}' \
    --output ./results_pruned/
python -m evalscope_ext.tools.compare_runs --full ./results_full/ --pruned ./results_pruned/
```

Use `aa_lcr_pruned` the same way for long-context reasoning. Use `mmmu_pruned` when a prospect asks about multimodal readiness.
All three pruned datasets are registered inside EvalScope and use the same pruning adapter path.

## How To Interpret

If the pruned score matches the full score within tolerance, the smaller set is a good operational proxy for that model. If it disagrees, escalate to a larger ratio or a full run. For AA-LCR, remember the judge is itself noisy, so one or two samples can move a 10-sample score.

The multimodal probe is useful because it is not just random MMMU sampling. It intentionally stresses image encoder failure modes: OCR, charts, diagrams, dense visual text, and multi-image grounding. That gives customer teams a clearer answer to "will this model actually read our visual inputs?" instead of only "does it know the subject?"

For PMs, the value is a faster customer conversation: run a small, defensible probe today, decide whether the model is promising, and reserve expensive full evaluations for the candidates that pass the first screen.
