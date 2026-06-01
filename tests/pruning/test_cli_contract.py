from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


def test_flat_dataset_args_update_pruning_extra_params() -> None:
    config = TaskConfig(
        datasets=['live_code_bench_pruned'],
        dataset_args={
            'pruning_strategy': 'calibrated_coverage',
            'prune_ratio': 0.2,
        },
    )

    benchmark = get_benchmark('live_code_bench_pruned', config)

    assert benchmark.extra_params['pruning_strategy'] == 'calibrated_coverage'
    assert benchmark.extra_params['prune_ratio'] == 0.2


def test_nested_dataset_args_still_work() -> None:
    config = TaskConfig(
        datasets=['aa_lcr_pruned'],
        dataset_args={
            'aa_lcr_pruned': {
                'prune_ratio': 0.15,
                'extra_params': {
                    'prune_seed': 99,
                },
            }
        },
    )

    benchmark = get_benchmark('aa_lcr_pruned', config)

    assert benchmark.extra_params['prune_ratio'] == 0.15
    assert benchmark.extra_params['prune_seed'] == 99
