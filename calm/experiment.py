from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from calm import clustering, schema, splits
from calm.config import ExperimentConfig, save_config
from calm.dataset import PairDataset, build_records
from calm.device import describe, resolve_device, set_seed
from calm.embeddings import embed_dataset, load_stores
from calm.metrics import mean_and_ci, random_baseline
from calm.train import train_fold


def ensure_embeddings(config: ExperimentConfig, spec, log=print, overwrite: bool = False) -> dict:
    frame, report = schema.load_dataset(config.data.dataset, require_masks=False)
    log(f"[embed] validated source rows {report['output_rows']} of {report['input_rows']}")
    paths = embed_dataset(frame, config.embedding, spec, config.data.embedding_dir, overwrite=overwrite)
    log(f"[embed] antibody={paths['antibody']} antigen={paths['antigen']}")
    return paths


def prepare_frame(config: ExperimentConfig, workdir, log=print):
    masked = config.data.mask_mode == "interface"
    frame, report = schema.load_dataset(config.data.dataset, require_masks=masked)
    log(f"[data] {report}")
    if config.data.random_shuffle:
        frame = clustering.random_shuffle(frame, int(config.train.seed))
        log("[data] random-shuffle control: antigen block permuted across pairs")
    assignment = clustering.build_clusters(frame, config.data, workdir)
    frame, assignment = clustering.subsample(
        frame,
        assignment,
        int(config.data.max_per_cluster),
        int(config.train.seed),
    )
    for note in assignment.notes:
        log(f"[cluster] {note}")
    return frame, assignment, report


def aggregate(outcomes, ks) -> dict:
    summary: dict = {}
    for direction in ("ab_to_ag", "ag_to_ab", "mean"):
        summary[direction] = dict()
        for k in ks:
            values = [float(item.test[direction][int(k)]) for item in outcomes if item.test.get(direction)]
            mean, ci = mean_and_ci(values)
            summary[direction][str(int(k))] = {"mean": mean, "ci95": ci, "folds": values}
    return summary


def run_experiment(config: ExperimentConfig, log=print) -> dict:
    set_seed(int(config.train.seed))
    spec = resolve_device(config.device)
    log(f"[device] {describe(spec)}")
    output = Path(config.output_dir) / config.name
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "config.yaml")
    ensure_embeddings(config, spec, log)
    frame, assignment, report = prepare_frame(config, output / "cluster", log)
    antibody_store, antigen_store = load_stores(config.data.embedding_dir)
    records = build_records(frame, config.data.mask_mode)
    for record, cluster in zip(records, assignment.split_clusters):
        record.cluster_id = cluster
    dataset = PairDataset(records, antibody_store, antigen_store)
    clusters = list(assignment.split_clusters)
    folds = splits.nested_folds(clusters, int(config.folds), int(config.train.seed))
    outcomes = []
    for split in folds:
        splits.assert_cluster_disjoint(split, clusters)
        fractions = splits.split_summary(split, len(records))
        shares = " ".join(f"{key}={value:.2f}" for key, value in fractions.items())
        log(f"[fold {split.fold}] pairs train={len(split.train)} val={len(split.val)} test={len(split.test)} {shares}")
        _model, outcome = train_fold(dataset, split, config.model, config.train, config.evaluation, spec)
        outcomes.append(outcome)
        log(f"[fold {split.fold}] test {outcome.test.get('mean')}")
        payload = json.dumps(asdict(outcome), indent=2)
        (output / f"fold{split.fold}.json").write_text(payload, encoding="utf-8")
    ks = tuple(int(k) for k in config.evaluation.ks)
    baseline = random_baseline(int(config.evaluation.pool_size), ks)
    result = {
        "name": config.name,
        "label": config.label,
        "mask_mode": config.data.mask_mode,
        "cluster_on": config.data.cluster_on,
        "min_seq_id": float(config.data.min_seq_id),
        "random_shuffle": bool(config.data.random_shuffle),
        "pairs": len(records),
        "clusters": len(set(clusters)),
        "unique_antibody_labels": len({item.antibody_label for item in records}),
        "unique_antigen_labels": len({item.antigen_label for item in records}),
        "dataset_report": report,
        "cluster_backend": assignment.backend,
        "cluster_notes": assignment.notes,
        "device": describe(spec),
        "pool_size": int(config.evaluation.pool_size),
        "random_baseline": {str(k): value for k, value in baseline.items()},
        "folds": [asdict(item) for item in outcomes],
        "summary": aggregate(outcomes, ks),
    }
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"[done] {config.name} -> {output / 'results.json'}")
    return result
