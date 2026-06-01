import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class RunReport:
    model: str
    dataset: str
    metric: str
    score: float
    num: int
    path: str


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare full and pruned EvalScope runs.')
    parser.add_argument('--full', required=True, help='Full EvalScope output directory.')
    parser.add_argument('--pruned', required=True, help='Pruned EvalScope output directory.')
    parser.add_argument('--tolerance', type=float, default=0.05, help='Maximum acceptable absolute score delta.')
    parser.add_argument('--json', action='store_true', help='Print JSON instead of a Markdown table.')
    args = parser.parse_args()

    rows = compare_runs(full_dir=args.full, pruned_dir=args.pruned, tolerance=args.tolerance)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(_format_markdown(rows))

    if not rows:
        raise SystemExit(2)


def compare_runs(full_dir: str, pruned_dir: str, tolerance: float = 0.05) -> List[Dict[str, object]]:
    """Compare reports from full and pruned output directories."""
    full_reports = _load_reports(full_dir)
    pruned_reports = _load_reports(pruned_dir)

    full_by_key = {(_base_dataset(report.dataset), report.model, report.metric): report for report in full_reports}
    rows: List[Dict[str, object]] = []
    for pruned in pruned_reports:
        key = (_base_dataset(pruned.dataset), pruned.model, pruned.metric)
        full = full_by_key.get(key)
        if full is None:
            continue

        delta = pruned.score - full.score
        rows.append(
            {
                'dataset': key[0],
                'model': pruned.model,
                'metric': pruned.metric,
                'full_score': round(full.score, 4),
                'pruned_score': round(pruned.score, 4),
                'delta': round(delta, 4),
                'full_samples': full.num,
                'pruned_samples': pruned.num,
                'prune_ratio': round(pruned.num / full.num, 4) if full.num else 0,
                'verdict': 'match' if abs(delta) <= tolerance else 'review',
                'full_report': full.path,
                'pruned_report': pruned.path,
            }
        )

    return sorted(rows, key=lambda row: (str(row['dataset']), str(row['model']), str(row['metric'])))


def _load_reports(output_dir: str) -> List[RunReport]:
    reports: List[RunReport] = []
    for path in _iter_json_files(output_dir):
        report = _try_read_report(path)
        if report is not None:
            reports.append(report)
    return reports


def _iter_json_files(output_dir: str) -> Iterable[str]:
    for root, _, files in os.walk(output_dir):
        for filename in files:
            if filename.endswith('.json'):
                yield os.path.join(root, filename)


def _try_read_report(path: str) -> Optional[RunReport]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if not {'dataset_name', 'model_name', 'metrics'} <= set(data):
        return None

    metrics = data.get('metrics') or []
    if not metrics:
        return None
    first_metric = metrics[0]
    return RunReport(
        model=str(data.get('model_name', 'unknown')),
        dataset=str(data.get('dataset_name', 'unknown')),
        metric=str(first_metric.get('name', 'score')),
        score=float(first_metric.get('score', data.get('score', 0.0))),
        num=int(first_metric.get('num', data.get('num', 0))),
        path=path,
    )


def _base_dataset(dataset: str) -> str:
    return dataset[:-7] if dataset.endswith('_pruned') else dataset


def _format_markdown(rows: List[Dict[str, object]]) -> str:
    if not rows:
        return 'No matching full/pruned reports found.'

    header = (
        '| Dataset | Model | Metric | Full | Pruned | Delta | Samples | Ratio | Verdict |\n'
        '|---|---|---:|---:|---:|---:|---:|---:|---|'
    )
    body = []
    for row in rows:
        body.append(
            '| {dataset} | {model} | {metric} | {full_score:.4f} | {pruned_score:.4f} | {delta:+.4f} | '
            '{pruned_samples}/{full_samples} | {prune_ratio:.2%} | {verdict} |'.format(**row)
        )
    return '\n'.join([header] + body)


if __name__ == '__main__':
    main()
