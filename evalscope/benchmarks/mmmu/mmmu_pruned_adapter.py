import copy
from typing import Any

from evalscope.api.dataset import DatasetDict
from evalscope.api.registry import BENCHMARK_REGISTRY, register_benchmark
from evalscope.benchmarks._pruning import apply_pruning
from evalscope.benchmarks._pruning.selector import PRUNING_EXTRA_PARAMS
from evalscope.benchmarks.mmmu.mmmu_adapter import MMMUAdapter


def _build_pruned_meta():
    meta = copy.deepcopy(BENCHMARK_REGISTRY['mmmu'])
    meta.name = 'mmmu_pruned'
    meta.pretty_name = 'MMMU Image-Encoder Probe'
    meta.description = """
## Overview

MMMU Pruned is a forward-looking multimodal probe for image-encoder degradation. It favors OCR-dense,
table/chart, diagram, multi-image, and hard/open-response samples while keeping subject and question-type
coverage across the full MMMU validation split.

## Evaluation Notes

- Default pruning strategy: `image_encoder_probe`
- Runtime parameters: `pruning_strategy`, `prune_ratio`, `prune_seed`
- This is intended as a cheap probe, not a replacement for a full MMMU run.
"""
    meta.extra_params = copy.deepcopy(meta.extra_params)
    meta.extra_params.update(copy.deepcopy(PRUNING_EXTRA_PARAMS))
    meta.extra_params['pruning_strategy']['value'] = 'image_encoder_probe'
    return meta


@register_benchmark(_build_pruned_meta())
class PrunedMMMUAdapter(MMMUAdapter):
    """MMMU adapter with deterministic image-encoder stress pruning."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pruning_strategy = self.extra_params.get('pruning_strategy', 'image_encoder_probe')
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
