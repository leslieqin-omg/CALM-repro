from __future__ import annotations

import torch


def positive_matrix(labels: torch.Tensor) -> torch.Tensor:
    return (labels.unsqueeze(1) == labels.unsqueeze(0)).to(torch.float32)


def multi_positive_infonce(logits: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    log_prob = torch.log_softmax(logits.float(), dim=1)
    positive_count = positives.sum(dim=1).clamp(min=1.0)
    per_query = -(log_prob * positives).sum(dim=1) / positive_count
    return per_query.mean()


def symmetric_multi_positive_loss(
    logits: torch.Tensor,
    antibody_labels: torch.Tensor,
    antigen_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forward_positives = positive_matrix(antigen_labels)
    reverse_positives = positive_matrix(antibody_labels)
    pa_to_ep = multi_positive_infonce(logits, forward_positives)
    ep_to_pa = multi_positive_infonce(logits.t(), reverse_positives)
    total = 0.5 * (pa_to_ep + ep_to_pa)
    return total, pa_to_ep.detach(), ep_to_pa.detach()
