from evalscope.api.registry import register_benchmark
from evalscope.benchmarks._pruning.adapter import PrunedBenchmarkMixin, build_pruned_meta
from evalscope.benchmarks.mmmu.mmmu_adapter import MMMUAdapter


@register_benchmark(
    build_pruned_meta(
        base_benchmark='mmmu',
        pruned_name='mmmu_pruned',
        pretty_name='MMMU Image-Encoder Probe',
        default_strategy='image_encoder_probe',
        description="""
## Overview

MMMU Pruned is a forward-looking multimodal probe for image-encoder degradation. It favors OCR-dense,
table/chart, diagram, multi-image, and hard/open-response samples while keeping subject and question-type
coverage across the full MMMU validation split.

## Evaluation Notes

- Default pruning strategy: `image_encoder_probe`
- Runtime parameters: `pruning_strategy`, `prune_ratio`, `prune_seed`
- This is intended as a cheap probe, not a replacement for a full MMMU run.
""",
    )
)
class PrunedMMMUAdapter(PrunedBenchmarkMixin, MMMUAdapter):
    """MMMU adapter with deterministic image-encoder stress pruning."""
