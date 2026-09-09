from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from torch.utils.data import Subset

from .config import EvalConfig, ModelConfig, TrainConfig
from .dataset import PairDataset, make_loader
from .device import DeviceSpec, needs_checkpoint, suggest_batch_size, suggest_token_chunk
from .evaluate import evaluate_indices, move_batch
from .losses import symmetric_multi_positive_loss
from .model import CALM


@dataclass
class FoldOutcome:
    fold: int
    best_epoch: int
    best_val: float
    history: list[dict] = field(default_factory=list)
    test: dict = field(default_factory=dict)
    sizes: dict = field(default_factory=dict)
    seconds: float = 0.0


def selection_metric(scores: dict, k: int) -> float:
    mean = scores.get("mean") or dict()
    if not mean:
        return 0.0
    if int(k) in mean:
        return float(mean[int(k)])
    return float(mean[max(mean)])


def snapshot(model: CALM) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_fold(
    dataset: PairDataset,
    split,
    model_config: ModelConfig,
    train_config: TrainConfig,
    eval_config: EvalConfig,
    spec: DeviceSpec,
    verbose: bool = True,
) -> tuple[CALM, FoldOutcome]:
    torch.manual_seed(int(train_config.seed) + int(split.fold))
    model = CALM(model_config, dataset.antibody_dim, dataset.antigen_dim).to(spec.device)
    model.configure_runtime(suggest_token_chunk(spec, train_config.token_chunk), needs_checkpoint(spec))
    batch_size = suggest_batch_size(spec, int(train_config.train_batch_size))
    generator = torch.Generator()
    generator.manual_seed(int(train_config.seed) + int(split.fold))
    loader = make_loader(
        Subset(dataset, list(split.train)),
        batch_size,
        True,
        False,
        generator,
        int(train_config.num_workers),
        spec.is_cuda,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.learning_rate),
        weight_decay=float(train_config.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(1, int(train_config.scheduler_t0)),
        T_mult=max(1, int(train_config.scheduler_t_mult)),
        eta_min=float(train_config.scheduler_eta_min),
    )
    use_scaler = spec.amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(spec.device.type, enabled=use_scaler)
    target_k = 10 if 10 in eval_config.ks else max(int(k) for k in eval_config.ks)
    best_state = snapshot(model)
    best_value = -1.0
    best_epoch = -1
    history: list[dict] = []
    steps = max(1, len(loader))
    started = time.time()
    for epoch in range(int(train_config.epochs)):
        model.train()
        running = 0.0
        for position, raw in enumerate(loader):
            batch = move_batch(raw, spec.device)
            optimizer.zero_grad(set_to_none=True)
            with spec.autocast():
                _antibody, _antigen, logits = model(batch)
            loss, _forward, _reverse = symmetric_multi_positive_loss(
                logits.float(),
                batch["antibody_label"],
                batch["antigen_label"],
            )
            scaler.scale(loss).backward()
            if float(train_config.grad_clip) > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch + position / steps)
            running += float(loss.detach().item())
        val_scores = evaluate_indices(
            model,
            dataset,
            list(split.val),
            eval_config.pool_size,
            0,
            eval_config.ks,
            spec,
            int(train_config.seed) + epoch,
        )
        value = selection_metric(val_scores, target_k)
        history.append(
            {
                "epoch": epoch,
                "loss": running / steps,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "temperature": float(model.temperature.detach().item()),
                "val": val_scores,
            }
        )
        if value > best_value:
            best_value = value
            best_epoch = epoch
            best_state = snapshot(model)
        if verbose and (epoch % 10 == 0 or epoch + 1 == int(train_config.epochs)):
            average = running / steps
            print(f"[fold {split.fold}] epoch {epoch} loss {average:.4f} val_R@{target_k} {value:.4f}", flush=True)
    model.load_state_dict(best_state)
    model.to(spec.device)
    test_scores = evaluate_indices(
        model,
        dataset,
        list(split.test),
        eval_config.pool_size,
        eval_config.test_pairs_per_fold,
        eval_config.ks,
        spec,
        int(train_config.seed) + 1000 + int(split.fold),
    )
    outcome = FoldOutcome(
        fold=int(split.fold),
        best_epoch=best_epoch,
        best_val=best_value,
        history=history,
        test=test_scores,
        sizes={"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        seconds=time.time() - started,
    )
    return model, outcome
