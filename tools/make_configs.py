from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calm.config import apply_overrides, from_dict, save_config

REGIMES = (
    ("ag40", "Ab-Ag 40%", "none", "antigen", 0.40, False),
    ("ag60", "Ab-Ag 60%", "none", "antigen", 0.60, False),
    ("ag80", "Ab-Ag 80%", "none", "antigen", 0.80, False),
    ("rs_full", "Ab-Ag RS", "none", "antigen", 0.80, True),
    ("pa_ep40", "Pa-Ep 40%", "interface", "antigen", 0.40, False),
    ("pa_ep60", "Pa-Ep 60%", "interface", "antigen", 0.60, False),
    ("pa_ep80", "Pa-Ep 80%", "interface", "antigen", 0.80, False),
    ("rs_masked", "Pa-Ep RS", "interface", "antigen", 0.80, True),
    ("uc90", "UC90", "interface", "antibody", 0.90, False),
    ("uc95", "UC95", "interface", "antibody", 0.95, False),
)

SMOKE_OVERRIDES = {
    "embedding.antibody_model": "hashed",
    "embedding.antigen_model": "hashed",
    "embedding.antibody_dim": 64,
    "embedding.antigen_dim": 96,
    "data.embedding_dir": "artifacts/smoke/embeddings",
    "data.max_per_cluster": 25,
    "output_dir": "artifacts/smoke/runs",
    "folds": 5,
    "train.epochs": 30,
    "train.train_batch_size": 64,
    "train.eval_batch_size": 64,
    "train.scheduler_t0": 5,
    "model.hidden_dim": 128,
    "model.embed_dim": 64,
    "evaluation.pool_size": 32,
    "evaluation.test_pairs_per_fold": 128,
    "device": "cpu",
}

RTX3060_OVERRIDES = {
    "train.train_batch_size": 512,
    "train.token_chunk": 65536,
    "embedding.batch_size": 4,
    "device": "cuda:0",
}

RTX4090_OVERRIDES = {
    "train.train_batch_size": 1024,
    "train.token_chunk": 262144,
    "embedding.batch_size": 8,
    "device": "cuda:1",
}


def base(name, label, mask_mode, cluster_on, min_seq_id, shuffle):
    config = from_dict(dict())
    return apply_overrides(
        config,
        {
            "name": name,
            "label": label,
            "data.mask_mode": mask_mode,
            "data.cluster_on": cluster_on,
            "data.min_seq_id": float(min_seq_id),
            "data.random_shuffle": bool(shuffle),
        },
    )


def main() -> int:
    root = ROOT / "configs"
    written = []
    for spec in REGIMES:
        config = base(*spec)
        path = root / (spec[0] + ".yaml")
        save_config(config, path)
        written.append(path)
        smoke = apply_overrides(config, dict(SMOKE_OVERRIDES, **{"name": spec[0], "label": spec[1] + " (smoke)"}))
        smoke_path = root / "smoke" / (spec[0] + ".yaml")
        save_config(smoke, smoke_path)
        written.append(smoke_path)
        for folder, overrides in (("rtx3060", RTX3060_OVERRIDES), ("rtx4090", RTX4090_OVERRIDES)):
            gpu = apply_overrides(config, overrides)
            gpu_path = root / folder / (spec[0] + ".yaml")
            save_config(gpu, gpu_path)
            written.append(gpu_path)
    for path in written:
        print(path.relative_to(ROOT))
    print(f"wrote {len(written)} config files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
