from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FoldSplit:
    fold: int
    train: list[int]
    val: list[int]
    test: list[int]


def cluster_members(clusters: list[str]) -> dict[str, list[int]]:
    members: dict[str, list[int]] = dict()
    for position, cluster in enumerate(clusters):
        members.setdefault(cluster, []).append(position)
    return members


def balanced_partition(members: dict[str, list[int]], keys: list[str], bins: int, rng) -> list[list[str]]:
    priority = {key: float(value) for key, value in zip(keys, rng.random(len(keys)))}
    order = sorted(keys, key=lambda key: (-len(members[key]), priority[key]))
    buckets: list[list[str]] = [[] for _ in range(bins)]
    loads = [0] * bins
    for key in order:
        target = min(range(bins), key=lambda index: (loads[index], index))
        buckets[target].append(key)
        loads[target] += len(members[key])
    return buckets


def take_until(members: dict[str, list[int]], keys: list[str], target: int, rng) -> list[str]:
    shuffled = list(keys)
    rng.shuffle(shuffled)
    chosen: list[str] = []
    total = 0
    for key in shuffled:
        if total >= target:
            break
        chosen.append(key)
        total += len(members[key])
    return chosen


def nested_folds(clusters, folds: int = 5, seed: int = 0, val_fraction: float = 0.15) -> list[FoldSplit]:
    members = cluster_members(list(clusters))
    rng = np.random.default_rng(seed)
    keys = list(members)
    buckets = balanced_partition(members, keys, int(folds), rng)
    total = sum(len(value) for value in members.values())
    target = int(round(total * float(val_fraction)))
    splits: list[FoldSplit] = []
    for index in range(int(folds)):
        test_keys = set(buckets[index])
        remaining = [key for key in keys if key not in test_keys]
        val_keys = set(take_until(members, remaining, target, rng))
        train_keys = [key for key in remaining if key not in val_keys]
        test_rows = sorted(position for key in test_keys for position in members[key])
        val_rows = sorted(position for key in val_keys for position in members[key])
        train_rows = sorted(position for key in train_keys for position in members[key])
        splits.append(FoldSplit(index, train_rows, val_rows, test_rows))
    return splits


def split_summary(split: FoldSplit, total: int) -> dict[str, float]:
    denominator = float(max(total, 1))
    return {
        "train": len(split.train) / denominator,
        "val": len(split.val) / denominator,
        "test": len(split.test) / denominator,
    }


def assert_cluster_disjoint(split: FoldSplit, clusters: list[str]) -> None:
    train = {clusters[position] for position in split.train}
    val = {clusters[position] for position in split.val}
    test = {clusters[position] for position in split.test}
    overlap = (train & val) | (train & test) | (val & test)
    if overlap:
        raise ValueError(f"cluster leakage across folds: {sorted(overlap)[:5]}")
