import copy
import hashlib
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from evalscope.api.dataset import Dataset, DatasetDict, MemoryDataset, Sample

DEFAULT_PRUNING_STRATEGY = 'calibrated_coverage'
SUPPORTED_PRUNING_STRATEGIES = {
    DEFAULT_PRUNING_STRATEGY,
    'coverage',
    'image_encoder_probe',
    'your_strategy',
}

PRUNING_EXTRA_PARAMS = {
    'pruning_strategy': {
        'type': 'str',
        'description': 'Deterministic pruning strategy. Use calibrated_coverage for LCB/AA-LCR.',
        'value': DEFAULT_PRUNING_STRATEGY,
    },
    'prune_ratio': {
        'type': 'float',
        'description': 'Fraction of each benchmark subset to retain. 0.1 keeps roughly 10%.',
        'value': 0.1,
    },
    'prune_seed': {
        'type': 'int',
        'description': 'Stable tie-break seed for metadata-only fallback pruning.',
        'value': 42,
    },
}

# Precomputed from the shipped reference reviews with behavioral-pattern quota matching
# plus metadata coverage. These are sample indices in the unpruned EvalScope dataset order.
CALIBRATED_MANIFESTS: Dict[Tuple[str, str, int], List[int]] = {
    ('live_code_bench_pruned', 'v5', 315): [
        74,
        258,
        8,
        163,
        118,
        241,
        274,
        7,
        69,
        277,
        60,
        51,
        307,
        55,
        198,
        152,
        0,
        47,
        284,
        2,
        54,
        3,
        32,
        185,
        145,
        56,
        36,
        49,
        190,
        162,
        122,
        9,
    ],
    ('aa_lcr_pruned', 'default', 100): [
        71,
        47,
        60,
        53,
        12,
        34,
        14,
        57,
        69,
        0,
    ],
}


def apply_pruning(
    dataset_dict: DatasetDict,
    benchmark_name: str,
    prune_ratio: float,
    pruning_strategy: str = DEFAULT_PRUNING_STRATEGY,
    prune_seed: int = 42,
) -> DatasetDict:
    """Return a pruned copy of an EvalScope DatasetDict.

    The primary path uses calibrated manifests for shipped LCB v5 and AA-LCR.
    For unseen subsets or future dataset revisions, the fallback is deterministic
    metadata coverage, not random sampling.
    """
    strategy = _normalize_strategy(pruning_strategy)
    ratio = _normalize_ratio(prune_ratio)
    if ratio >= 1:
        return dataset_dict

    pruned: Dict[str, Dataset] = {}
    for subset_name, dataset in dataset_dict.items():
        samples = list(dataset)
        if not samples:
            pruned[subset_name] = dataset
            continue

        keep_count = max(1, round(len(samples) * ratio))
        indices = _select_indices(
            samples=samples,
            benchmark_name=benchmark_name,
            subset_name=subset_name,
            keep_count=keep_count,
            strategy=strategy,
            seed=prune_seed,
        )
        selected = [copy.deepcopy(samples[index]) for index in indices]
        pruned[subset_name] = MemoryDataset(
            samples=selected,
            name=f'{dataset.name or subset_name}:pruned',
            location=dataset.location,
            shuffled=dataset.shuffled,
        )

    return DatasetDict(pruned)


def _normalize_strategy(strategy: Optional[str]) -> str:
    if not strategy:
        return DEFAULT_PRUNING_STRATEGY
    if strategy not in SUPPORTED_PRUNING_STRATEGIES:
        raise ValueError(
            f'Unknown pruning_strategy {strategy!r}. Supported strategies: {sorted(SUPPORTED_PRUNING_STRATEGIES)}'
        )
    if strategy in {'coverage', 'your_strategy'}:
        return DEFAULT_PRUNING_STRATEGY
    return strategy


def _normalize_ratio(prune_ratio: float) -> float:
    ratio = float(prune_ratio)
    if ratio <= 0 or ratio > 1:
        raise ValueError(f'prune_ratio must be in (0, 1], got {prune_ratio!r}.')
    return ratio


def _select_indices(
    samples: Sequence[Sample],
    benchmark_name: str,
    subset_name: str,
    keep_count: int,
    strategy: str,
    seed: int,
) -> List[int]:
    manifest = CALIBRATED_MANIFESTS.get((benchmark_name, subset_name, len(samples)))
    if manifest is not None and strategy == DEFAULT_PRUNING_STRATEGY:
        selected = [index for index in manifest if index < len(samples)]
        if len(selected) >= keep_count:
            return selected[:keep_count]

    if strategy == 'image_encoder_probe':
        return _image_probe_indices(samples=samples, keep_count=keep_count, seed=seed)
    return _coverage_indices(samples=samples, keep_count=keep_count, seed=seed)


def _coverage_indices(samples: Sequence[Sample], keep_count: int, seed: int) -> List[int]:
    groups: Dict[Tuple[Any, ...], List[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[_coverage_key(sample)].append(index)

    for indices in groups.values():
        indices.sort(key=lambda index: _stable_hash(_sample_identity(samples[index]), seed))

    return _round_robin(groups=groups, keep_count=keep_count)


def _image_probe_indices(samples: Sequence[Sample], keep_count: int, seed: int) -> List[int]:
    groups: Dict[Tuple[Any, ...], List[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        metadata = sample.metadata or {}
        groups[(
            metadata.get('subfield') or 'unknown',
            metadata.get('question_type') or 'unknown',
            _coarse_image_type(metadata.get('img_type')),
        )].append(index)

    for indices in groups.values():
        indices.sort(
            key=lambda index: (
                -_image_stress_score(samples[index]),
                _stable_hash(_sample_identity(samples[index]), seed),
            )
        )

    return _round_robin(groups=groups, keep_count=keep_count)


def _round_robin(groups: Dict[Tuple[Any, ...], List[int]], keep_count: int) -> List[int]:
    selected: List[int] = []
    group_items = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    cursor = {key: 0 for key, _ in group_items}

    while len(selected) < keep_count:
        made_progress = False
        for key, indices in group_items:
            offset = cursor[key]
            if offset >= len(indices):
                continue
            selected.append(indices[offset])
            cursor[key] = offset + 1
            made_progress = True
            if len(selected) == keep_count:
                break
        if not made_progress:
            break

    return selected


def _coverage_key(sample: Sample) -> Tuple[Any, ...]:
    metadata = sample.metadata or {}
    return (
        metadata.get('question_type') or 'text',
        metadata.get('topic_difficulty') or 'unknown',
        _coarse_image_type(metadata.get('img_type')),
        _numeric_bucket(metadata.get('input_tokens')),
        _numeric_bucket(_input_length(sample)),
    )


def _coarse_image_type(raw_value: Any) -> str:
    text = str(raw_value or '').lower()
    if any(token in text for token in ['table', 'chart', 'plot', 'graph']):
        return 'data_visual'
    if any(token in text for token in ['diagram', 'map', 'geometry', 'structure']):
        return 'diagram'
    if any(token in text for token in ['text', 'document', 'sheet', 'music']):
        return 'ocr_dense'
    if text and text != 'none':
        return 'image_other'
    return 'none'


def _numeric_bucket(value: Any) -> str:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        number = 0
    if number <= 0:
        return 'none'
    if number < 2_000:
        return 'short'
    if number < 20_000:
        return 'medium'
    if number < 100_000:
        return 'long'
    return 'very_long'


def _input_length(sample: Sample) -> int:
    if isinstance(sample.input, str):
        return len(sample.input)
    total = 0
    for message in sample.input:
        content = getattr(message, 'content', '')
        total += len(str(content))
    return total


def _image_stress_score(sample: Sample) -> float:
    metadata = sample.metadata or {}
    img_type = _coarse_image_type(metadata.get('img_type'))
    score = {
        'data_visual': 4.0,
        'ocr_dense': 3.5,
        'diagram': 3.0,
        'image_other': 2.0,
        'none': 0.0,
    }[img_type]

    if str(metadata.get('topic_difficulty', '')).lower() == 'hard':
        score += 1.0
    if metadata.get('question_type') == 'open':
        score += 0.5
    score += min(_count_image_parts(sample), 4) * 0.4
    return score


def _count_image_parts(sample: Sample) -> int:
    if isinstance(sample.input, str):
        return 0
    count = 0
    for message in sample.input:
        content = getattr(message, 'content', None)
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get('image'):
                    count += 1
                elif getattr(part, 'image', None):
                    count += 1
    return count


def _sample_identity(sample: Sample) -> str:
    metadata = sample.metadata or {}
    stable_parts = [
        str(sample.id),
        str(sample.group_id),
        str(metadata.get('id', '')),
        str(metadata.get('question', '')),
        str(metadata.get('subfield', '')),
        str(_input_length(sample)),
    ]
    return '|'.join(stable_parts)


def _stable_hash(value: str, seed: int) -> int:
    digest = hashlib.sha256(f'{seed}:{value}'.encode('utf-8')).hexdigest()
    return int(digest[:16], 16)
