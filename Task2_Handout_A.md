# Task 2 Handout A: Why This Works

## Problem Framing

The customer question is not "what is the exact benchmark score?" It is "is this model good enough for our coding and long-context workload without waiting for the full suite?" I treated pruning as a decision-preserving compression problem: keep enough samples to preserve model ordering and go/no-go movement, while still covering the workload shapes that fail differently across models.

## Part A Approach

I implemented the pruner inside an EvalScope fork as first-class benchmark aliases:

- `live_code_bench_pruned`
- `aa_lcr_pruned`
- `mmmu_pruned`

The aliases share a reusable pruning adapter mixin and metadata builder, so the extension
is not a one-off wrapper for LiveCodeBench. AA-LCR and MMMU register through the same
EvalScope pattern and call the same pruning selector.

The core strategy is `calibrated_coverage`. It uses the shipped reference reviews only to learn sample behavior patterns, not to pick top-k easy or hard cases. For each sample, the three reference model outcomes form a behavior pattern such as `(pass, fail, pass)`. The pruner allocates the reduced set so these behavior-pattern marginals match the full benchmark as closely as possible, then chooses samples within each behavior bucket to maximize metadata coverage. For unseen dataset revisions, it falls back to deterministic metadata coverage.

At `prune_ratio=0.1`:

- LiveCodeBench v5: 315 samples -> 32 samples.
- AA-LCR: 100 samples -> 10 samples.

Validation on the shipped reference models:

| Benchmark | Full scores | Pruned scores | Mean abs error | Ranking preserved |
|---|---|---|---:|---|
| LiveCodeBench v5 | 0.765 / 0.629 / 0.619 | 0.750 / 0.625 / 0.625 | 0.008 | Yes |
| AA-LCR | 0.480 / 0.660 / 0.640 | 0.500 / 0.700 / 0.600 | 0.033 | Yes |

AA-LCR is noisier because the metric is LLM-judge accuracy and the pruned set is only 10 samples. I would treat a small AA-LCR delta as directional, not absolute, unless repeated judging or a larger ratio such as 0.2 confirms it.

## Part B: Multimodal Probe

For MMMU, I added working `mmmu_pruned` code as an image-encoder stress probe. The point is not generic MMMU ability; it is to surface encoder degradation cheaply. The strategy favors:

- OCR-dense tables, charts, sheet music, and document images.
- Diagrams, maps, chemical structures, and geometry-like images.
- Multi-image samples where cross-image grounding matters.
- Hard/open-response samples where perception errors are less likely to be hidden by answer choices.

Through a standard OpenAI-compatible interface, I would measure encoder quality by comparing normal-image accuracy against controlled perturbation slices: downsampled images, JPEG compression, cropped borders, contrast changes, and OCR-obscuring blur. A model whose text reasoning is strong but whose score collapses under these visual perturbations is likely encoder-limited.

## Assumptions And Extensions

Assumptions: shipped reference models are enough to identify useful disagreement regions; benchmark order is stable for the shipped EvalScope dataset versions; customer workloads map reasonably to LCB coding and AA-LCR long-context retrieval.

With more data, I would learn bucket weights from more model families and run bootstrap confidence intervals. With a live endpoint, I would adaptively add samples near the model's decision boundary. With more time, I would add repeated AA-LCR judging and a calibration report that recommends 10%, 20%, or full-run escalation based on uncertainty.
