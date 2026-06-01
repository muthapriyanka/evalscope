import copy
from typing import Any

from evalscope.api.dataset import DatasetDict
from evalscope.api.registry import BENCHMARK_REGISTRY, register_benchmark
from evalscope.benchmarks._pruning import apply_pruning
from evalscope.benchmarks._pruning.selector import DEFAULT_PRUNING_STRATEGY, PRUNING_EXTRA_PARAMS
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter


def _build_pruned_meta():
    meta = copy.deepcopy(BENCHMARK_REGISTRY['aa_lcr'])
    meta.name = 'aa_lcr_pruned'
    meta.pretty_name = 'AA-LCR Pruned'
    meta.description = """
## Overview

AA-LCR Pruned is a deterministic compression of AA-LCR for quick long-context reasoning checks. The selector
keeps a small set that matches reference model pass/fail marginals while preserving document-count, token-length,
and question-shape coverage.

## Evaluation Notes

- Default pruning strategy: `calibrated_coverage`
- Runtime parameters: `pruning_strategy`, `prune_ratio`, `prune_seed`
- AA-LCR uses an LLM judge; small-set variance should be interpreted with that judge noise in mind.
"""
    meta.extra_params = copy.deepcopy(meta.extra_params)
    meta.extra_params.update(copy.deepcopy(PRUNING_EXTRA_PARAMS))
    return meta


@register_benchmark(_build_pruned_meta())
class PrunedAALCRAdapter(AALCRAdapter):
    """AA-LCR adapter with deterministic sample pruning."""

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
