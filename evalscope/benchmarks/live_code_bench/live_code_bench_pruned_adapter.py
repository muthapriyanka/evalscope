from evalscope.api.registry import register_benchmark
from evalscope.benchmarks._pruning.adapter import PrunedBenchmarkMixin, build_pruned_meta
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter


@register_benchmark(
    build_pruned_meta(
        base_benchmark='live_code_bench',
        pruned_name='live_code_bench_pruned',
        pretty_name='Live-Code-Bench Pruned',
        subset_list=['v5'],
        description="""
## Overview

LiveCodeBench Pruned is a deterministic 10%-style compression of LiveCodeBench v5 for quick customer
go/no-go checks. The selector preserves reference model outcome-pattern marginals and then covers prompt
shape diversity, so it is not uniform random and not top-k easy/hard sampling.

## Evaluation Notes

- Default subset: `v5`
- Default pruning strategy: `calibrated_coverage`
- Runtime parameters: `pruning_strategy`, `prune_ratio`, `prune_seed`
""",
    )
)
class PrunedLiveCodeBenchAdapter(PrunedBenchmarkMixin, LiveCodeBenchAdapter):
    """LiveCodeBench adapter with deterministic sample pruning."""
