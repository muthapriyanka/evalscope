import copy
from typing import Any, Optional

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import DatasetDict
from evalscope.api.registry import BENCHMARK_REGISTRY
from evalscope.benchmarks._pruning.selector import (
    DEFAULT_PRUNING_STRATEGY,
    PRUNING_EXTRA_PARAMS,
    apply_pruning,
)


def build_pruned_meta(
    *,
    base_benchmark: str,
    pruned_name: str,
    pretty_name: str,
    description: str,
    default_strategy: str = DEFAULT_PRUNING_STRATEGY,
    subset_list: Optional[list[str]] = None,
) -> BenchmarkMeta:
    """Build benchmark metadata for a pruned alias of an existing benchmark."""
    meta = copy.deepcopy(BENCHMARK_REGISTRY[base_benchmark])
    meta.name = pruned_name
    meta.pretty_name = pretty_name
    meta.description = description
    if subset_list is not None:
        meta.subset_list = subset_list

    meta.extra_params = copy.deepcopy(meta.extra_params)
    meta.extra_params.update(copy.deepcopy(PRUNING_EXTRA_PARAMS))
    meta.extra_params['pruning_strategy']['value'] = default_strategy
    return meta


class PrunedBenchmarkMixin:
    """Reusable EvalScope adapter mixin for deterministic benchmark pruning."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pruning_strategy = self.extra_params.get(
            'pruning_strategy',
            DEFAULT_PRUNING_STRATEGY,
        )
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
