import json
from pathlib import Path

from evalscope_ext.tools.compare_runs import compare_runs


def _write_report(path: Path, dataset: str, score: float, num: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                'dataset_name': dataset,
                'model_name': 'model-a',
                'metrics': [
                    {
                        'name': 'acc',
                        'score': score,
                        'num': num,
                    }
                ],
            }
        ),
        encoding='utf-8',
    )


def test_compare_runs_pairs_pruned_dataset_name(tmp_path) -> None:
    _write_report(tmp_path / 'full' / 'reports' / 'model-a' / 'live_code_bench.json', 'live_code_bench', 0.75, 100)
    _write_report(
        tmp_path / 'pruned' / 'reports' / 'model-a' / 'live_code_bench_pruned.json',
        'live_code_bench_pruned',
        0.72,
        10,
    )

    rows = compare_runs(str(tmp_path / 'full'), str(tmp_path / 'pruned'))

    assert rows[0]['dataset'] == 'live_code_bench'
    assert rows[0]['delta'] == -0.03
    assert rows[0]['verdict'] == 'match'
