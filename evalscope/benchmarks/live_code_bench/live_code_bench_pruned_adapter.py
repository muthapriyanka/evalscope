import copy
from typing import Any

from evalscope.api.dataset import DatasetDict
from evalscope.api.registry import BENCHMARK_REGISTRY, register_benchmark
from evalscope.benchmarks._pruning import apply_pruning
from evalscope.benchmarks._pruning.selector import DEFAULT_PRUNING_STRATEGY, PRUNING_EXTRA_PARAMS
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter


def _build_pruned_meta():
    meta = copy.deepcopy(BENCHMARK_REGISTRY['live_code_bench'])
    meta.name = 'live_code_bench_pruned'
    meta.pretty_name = 'Live-Code-Bench Pruned'
    meta.subset_list = ['v5']
    meta.description = """
## Overview

LiveCodeBench Pruned is a deterministic 10%-style compression of LiveCodeBench v5 for quick customer
go/no-go checks. The selector preserves reference model outcome-pattern marginals and then covers prompt
shape diversity, so it is not uniform random and not top-k easy/hard sampling.

## Evaluation Notes

- Default subset: `v5`
- Default pruning strategy: `calibrated_coverage`
- Runtime parameters: `pruning_strategy`, `prune_ratio`, `prune_seed`
"""
    meta.extra_params = copy.deepcopy(meta.extra_params)
    meta.extra_params.update(copy.deepcopy(PRUNING_EXTRA_PARAMS))
    return meta


@register_benchmark(_build_pruned_meta())
class PrunedLiveCodeBenchAdapter(LiveCodeBenchAdapter):
    """LiveCodeBench adapter with deterministic sample pruning."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pruning_strategy = self.extra_params.get('pruning_strategy', DEFAULT_PRUNING_STRATEGY)
        self.prune_ratio = float(self.extra_params.get('prune_ratio', 0.1))
        self.prune_seed = int(self.extra_params.get('prune_seed', self.seed or 42))

    def load_dataset(self) -> DatasetDict:
        dataset_dict = super().load_dataset()
        return apply_pruning(
            dataset_dict=dataset_dict,
            benchmark_name=self.name,
            prune_ratio=self.prune_ratio,
            pruning_strategy=self.pruning_strategy,
            prune_seed=self.prune_seed,
        )
