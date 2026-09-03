from __future__ import annotations

import torch


def positive_mask(labels: torch.Tensor) -> torch.Tensor:
    return labels.unsqueeze(1) == labels.unsqueeze(0)


def recall_at_k(logits: torch.Tensor, positives: torch.Tensor, ks: tuple[int, ...]) -> dict[int, float]:
    pool = logits.shape[1]
    max_k = min(max(ks), pool)
    ranked = logits.float().topk(max_k, dim=1).indices
    hits = torch.gather(positives, 1, ranked)
    scores = dict()
    for k in ks:
        limit = min(k, max_k)
        scores[int(k)] = float(hits[:, :limit].any(dim=1).to(torch.float32).mean().item())
    return scores


def retrieval_metrics(
    logits: torch.Tensor,
    antibody_labels: torch.Tensor,
    antigen_labels: torch.Tensor,
    ks: tuple[int, ...],
) -> dict[str, dict[int, float]]:
    forward_scores = recall_at_k(logits, positive_mask(antigen_labels), ks)
    reverse_scores = recall_at_k(logits.t(), positive_mask(antibody_labels), ks)
    mean_scores = {k: 0.5 * (forward_scores[k] + reverse_scores[k]) for k in forward_scores}
    return {"ab_to_ag": forward_scores, "ag_to_ab": reverse_scores, "mean": mean_scores}


def random_baseline(pool_size: int, ks: tuple[int, ...]) -> dict[int, float]:
    return {int(k): min(1.0, float(k) / float(pool_size)) for k in ks}


def mean_and_ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    count = len(values)
    mean = sum(values) / count
    if count < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    stderr = (variance / count) ** 0.5
    return mean, 1.96 * stderr
