from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.benchmarks._pruning.selector import apply_pruning


def _dataset(size: int) -> DatasetDict:
    samples = [
        Sample(
            input=f'problem {index}',
            id=index,
            group_id=index,
            metadata={
                'question_type': 'multiple-choice' if index % 2 else 'open',
                'topic_difficulty': ['Easy', 'Medium', 'Hard'][index % 3],
                'input_tokens': index * 1000,
            },
        )
        for index in range(size)
    ]
    return DatasetDict({'default': MemoryDataset(samples=samples, name='unit')})


def test_calibrated_manifest_is_used_for_aa_lcr() -> None:
    pruned = apply_pruning(
        dataset_dict=_dataset(100),
        benchmark_name='aa_lcr_pruned',
        prune_ratio=0.1,
        pruning_strategy='calibrated_coverage',
        prune_seed=42,
    )

    ids = [sample.id for sample in pruned['default']]
    assert ids == [71, 47, 60, 53, 12, 34, 14, 57, 69, 0]


def test_fallback_coverage_is_deterministic() -> None:
    first = apply_pruning(
        dataset_dict=_dataset(25),
        benchmark_name='future_benchmark_pruned',
        prune_ratio=0.2,
        pruning_strategy='calibrated_coverage',
        prune_seed=7,
    )
    second = apply_pruning(
        dataset_dict=_dataset(25),
        benchmark_name='future_benchmark_pruned',
        prune_ratio=0.2,
        pruning_strategy='calibrated_coverage',
        prune_seed=7,
    )

    assert [sample.id for sample in first['default']] == [sample.id for sample in second['default']]
    assert len(first['default']) == 5
