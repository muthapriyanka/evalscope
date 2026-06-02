from evalscope.api.registry import register_benchmark
from evalscope.benchmarks._pruning.adapter import PrunedBenchmarkMixin, build_pruned_meta
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter


@register_benchmark(
    build_pruned_meta(
        base_benchmark='aa_lcr',
        pruned_name='aa_lcr_pruned',
        pretty_name='AA-LCR Pruned',
        description="""
## Overview

AA-LCR Pruned is a deterministic compression of AA-LCR for quick long-context reasoning checks. The selector
keeps a small set that matches reference model pass/fail marginals while preserving document-count, token-length,
and question-shape coverage.

## Evaluation Notes

- Default pruning strategy: `calibrated_coverage`
- Runtime parameters: `pruning_strategy`, `prune_ratio`, `prune_seed`
- AA-LCR uses an LLM judge; small-set variance should be interpreted with that judge noise in mind.
""",
    )
)
class PrunedAALCRAdapter(PrunedBenchmarkMixin, AALCRAdapter):
    """AA-LCR adapter with deterministic sample pruning."""
