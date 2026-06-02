from evalscope.api.registry import get_benchmark
from evalscope.benchmarks._pruning import PrunedBenchmarkMixin
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


def test_pruned_benchmarks_use_shared_adapter_mixin() -> None:
    for dataset_name in ['live_code_bench_pruned', 'aa_lcr_pruned', 'mmmu_pruned']:
        benchmark = get_benchmark(dataset_name, TaskConfig(datasets=[dataset_name]))

        assert isinstance(benchmark, PrunedBenchmarkMixin)


def test_mmmu_pruned_defaults_to_image_encoder_probe() -> None:
    benchmark = get_benchmark('mmmu_pruned', TaskConfig(datasets=['mmmu_pruned']))

    assert benchmark.extra_params['pruning_strategy'] == 'image_encoder_probe'
