from __future__ import annotations

import numpy as np
import torch

from calm.dataset import collate
from calm.device import DeviceSpec
from calm.metrics import retrieval_metrics


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def build_pools(indices, pool_size: int, max_pairs: int, rng) -> list[np.ndarray]:
    order = np.asarray(list(indices), dtype=np.int64)
    rng.shuffle(order)
    if max_pairs and int(max_pairs) > 0:
        order = order[: int(max_pairs)]
    width = max(2, int(pool_size))
    pools: list[np.ndarray] = []
    for start in range(0, order.size, width):
        block = order[start:start + width]
        if block.size == width:
            pools.append(block)
    if not pools and order.size >= 2:
        pools.append(order)
    return pools


def evaluate_pools(model, dataset, pools: list[np.ndarray], ks: tuple[int, ...], spec: DeviceSpec) -> dict:
    if not pools:
        return {"ab_to_ag": dict(), "ag_to_ab": dict(), "mean": dict(), "pairs": 0, "pools": 0, "pool_size": 0}
    was_training = model.training
    model.eval()
    totals: dict[str, dict[int, float]] = {"ab_to_ag": dict(), "ag_to_ab": dict(), "mean": dict()}
    weight = 0
    with torch.no_grad():
        for pool in pools:
            items = [dataset[int(position)] for position in pool]
            batch = move_batch(collate(items), spec.device)
            with spec.autocast():
                _antibody, _antigen, logits = model(batch)
            scores = retrieval_metrics(logits.float(), batch["antibody_label"], batch["antigen_label"], ks)
            for direction, values in scores.items():
                for k, value in values.items():
                    totals[direction][int(k)] = totals[direction].get(int(k), 0.0) + value * int(pool.size)
            weight += int(pool.size)
    if was_training:
        model.train()
    result: dict = {}
    for direction, values in totals.items():
        result[direction] = {int(k): value / float(weight) for k, value in values.items()}
    result["pairs"] = weight
    result["pools"] = len(pools)
    result["pool_size"] = int(pools[0].size)
    return result


def evaluate_indices(model, dataset, indices, pool_size: int, max_pairs: int, ks, spec: DeviceSpec, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pools = build_pools(indices, pool_size, max_pairs, rng)
    return evaluate_pools(model, dataset, pools, tuple(int(k) for k in ks), spec)
