from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EmbeddingConfig:
    antibody_model: str = "antiberty"
    antigen_model: str = "esm2_t33_650M_UR50D"
    antibody_dim: int = 512
    antigen_dim: int = 1280
    max_variable_len: int = 141
    max_antigen_len: int = 550
    batch_size: int = 8


@dataclass
class ModelConfig:
    hidden_dim: int = 1024
    embed_dim: int = 512
    dropout: float = 0.1
    temperature: float = 0.2
    learnable_temperature: bool = True


@dataclass
class TrainConfig:
    epochs: int = 100
    train_batch_size: int = 1024
    eval_batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.2
    scheduler_t0: int = 10
    scheduler_t_mult: int = 2
    scheduler_eta_min: float = 0.000001
    grad_clip: float = 1.0
    token_chunk: int | None = None
    seed: int = 0
    num_workers: int = 4


@dataclass
class DataConfig:
    dataset: str = "data/calm_dataset.csv"
    embedding_dir: str = "artifacts/embeddings"
    mask_mode: str = "interface"
    cluster_on: str = "antigen"
    min_seq_id: float = 0.8
    subsample_min_seq_id: float = 0.8
    max_per_cluster: int = 25
    random_shuffle: bool = False
    parasail_merge: bool = True
    coverage: float = 0.4


@dataclass
class EvalConfig:
    pool_size: int = 256
    test_pairs_per_fold: int = 512
    ks: tuple[int, ...] = (1, 5, 10, 50, 100)

    def __post_init__(self) -> None:
        if isinstance(self.ks, str):
            import ast
            try:
                parsed = ast.literal_eval(self.ks)
                if isinstance(parsed, (list, tuple)):
                    self.ks = tuple(int(k) for k in parsed)
                else:
                    self.ks = (int(parsed),)
            except Exception:
                self.ks = tuple(int(k.strip()) for k in self.ks.strip("[]()").split(",") if k.strip())
        else:
            self.ks = tuple(int(k) for k in self.ks)


@dataclass
class ExperimentConfig:
    name: str = "ag80"
    label: str = "80%"
    folds: int = 5
    output_dir: str = "artifacts/runs"
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _sub(cls, payload: dict[str, Any], key: str):
    section = payload.get(key) or {}
    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in section.items() if k in field_names}
    return cls(**filtered)


def from_dict(payload: dict[str, Any]) -> ExperimentConfig:
    payload = dict(payload or {})
    name = payload.get("name", "ag80")
    return ExperimentConfig(
        name=name,
        label=payload.get("label", name),
        folds=int(payload.get("folds", 5)),
        output_dir=payload.get("output_dir", "artifacts/runs"),
        device=payload.get("device", "auto"),
        data=_sub(DataConfig, payload, "data"),
        model=_sub(ModelConfig, payload, "model"),
        train=_sub(TrainConfig, payload, "train"),
        embedding=_sub(EmbeddingConfig, payload, "embedding"),
        evaluation=_sub(EvalConfig, payload, "evaluation"),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    text = Path(path).read_text(encoding="utf-8")
    return from_dict(yaml.safe_load(text) or {})


def apply_overrides(config: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    payload = config.to_dict()
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        head, _, tail = key.partition(".")
        if tail and isinstance(payload.get(head), dict):
            payload[head][tail] = value
        else:
            payload[key] = value
    return from_dict(payload)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
