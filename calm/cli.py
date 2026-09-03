from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from calm import schema
from calm.config import ExperimentConfig, apply_overrides, from_dict, load_config
from calm.version import PAPER_DOI, PAPER_TITLE, __version__


def _parse_value(raw: str):
    lowered = raw.strip()
    if lowered.lower() in ("true", "false"):
        return lowered.lower() == "true"
    if lowered.lower() in ("none", "null"):
        return None
    try:
        return int(lowered)
    except ValueError:
        pass
    try:
        return float(lowered)
    except ValueError:
        pass
    if lowered.startswith("[") and lowered.endswith("]"):
        inner = lowered[1:-1].strip()
        if inner == "":
            return []
        return [_parse_value(part) for part in inner.split(",")]
    return lowered


def _overrides(pairs) -> dict:
    out: dict = dict()
    for item in pairs or []:
        key, sep, value = str(item).partition("=")
        if not sep:
            raise SystemExit(f"invalid --set entry {item!r}; expected key=value")
        out[key.strip()] = _parse_value(value)
    return out


def _resolve_config(args) -> ExperimentConfig:
    if getattr(args, "config", None):
        if not Path(args.config).is_file():
            raise SystemExit(f"config file not found: {args.config}")
        config = load_config(args.config)
    else:
        config = from_dict(dict())
    overrides = _overrides(getattr(args, "set", None))
    if getattr(args, "device", None):
        overrides["device"] = args.device
    if getattr(args, "dataset", None):
        overrides["data.dataset"] = args.dataset
    if getattr(args, "output_dir", None):
        overrides["output_dir"] = args.output_dir
    if getattr(args, "epochs", None):
        overrides["train.epochs"] = int(args.epochs)
    if getattr(args, "folds", None):
        overrides["folds"] = int(args.folds)
    return apply_overrides(config, overrides)


def cmd_prepare(args) -> int:
    roots = args.structure_dir or ["data/splits_final"]
    if args.summary_csv:
        from calm.sabdab import prepare_from_summary

        table, report = prepare_from_summary(args.summary_csv, roots, args.output, limit=args.limit)
    else:
        from calm.prepare import prepare_dataset

        table, report = prepare_dataset(
            args.split_csv,
            roots,
            args.output,
            workers=args.workers,
            limit=args.limit,
        )
    target = Path(args.output)
    filtered, filter_report = schema.validate(table, require_masks=False)
    final = target.with_name(target.stem.replace("_raw", "") + target.suffix)
    if final == target:
        final = target.with_name(target.stem + "_filtered" + target.suffix)
    filtered.to_csv(final, index=False)
    report["filter"] = filter_report
    report["filtered_output"] = str(final)
    report_path = target.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[prepare] filtered dataset -> {final} ({len(filtered)} pairs)")
    print(f"[prepare] report -> {report_path}")
    return 0


def cmd_embed(args) -> int:
    from calm.device import describe, resolve_device
    from calm.experiment import ensure_embeddings

    config = _resolve_config(args)
    spec = resolve_device(config.device)
    print(f"[device] {describe(spec)}")
    ensure_embeddings(config, spec, print, overwrite=bool(args.overwrite))
    return 0


def cmd_cluster(args) -> int:
    from calm.experiment import prepare_frame

    config = _resolve_config(args)
    workdir = Path(config.output_dir) / config.name / "cluster"
    frame, assignment, report = prepare_frame(config, workdir, print)
    sizes: dict = {}
    for label in assignment.split_clusters:
        sizes[label] = sizes.get(label, 0) + 1
    payload = {
        "name": config.name,
        "pairs": int(len(frame)),
        "clusters": len(sizes),
        "largest_cluster": max(sizes.values()) if sizes else 0,
        "backend": assignment.backend,
        "notes": assignment.notes,
        "dataset_report": report,
    }
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def cmd_experiment(args) -> int:
    from calm.experiment import run_experiment

    config = _resolve_config(args)
    result = run_experiment(config, print)
    summary = result["summary"]["mean"]
    for k in sorted(summary, key=lambda item: int(item)):
        entry = summary[k]
        print(f"[result] {config.name} Recall@{k} = {entry['mean'] * 100:.2f}% +/- {entry['ci95'] * 100:.2f}")
    return 0


def cmd_train(args) -> int:
    return cmd_experiment(args)


def cmd_figures(args) -> int:
    from calm.figures import build_all

    outputs = build_all(args.runs, args.output)
    for item in outputs:
        print(f"[figures] {item}")
    return 0


def cmd_decoder_check(args) -> int:
    import torch

    from calm.decoder import TASKS, CalmDecoder, DecoderConfig, SequenceTokenizer, teacher_forcing

    device = torch.device("cpu")
    tokenizer = SequenceTokenizer()
    config = DecoderConfig(hidden_dim=64, layers=2, heads=4, ffn_dim=128, max_len=64)
    memory_dim = 32
    decoder = CalmDecoder(config, memory_dim, tokenizer.vocab_size).to(device)
    sequences = ["QVQLVQSGAEVKKPG", "EIVLTQSPATLSLSP"]
    memory = torch.randn(len(sequences), 12, memory_dim, device=device)
    for task in TASKS:
        inputs, targets = teacher_forcing(tokenizer, sequences, task, device)
        logits = decoder(inputs, memory)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=tokenizer.pad_id,
        )
        print(f"[decoder] task={task} inputs={tuple(inputs.shape)} logits={tuple(logits.shape)} loss={loss.item():.4f}")
        sampled = decoder.generate(memory[:1], task, tokenizer, max_len=16)
        decoded = tokenizer.decode(sampled[0][len(tokenizer.prefix_ids(task)) :])
        print(f"[decoder] task={task} tokens={tuple(sampled.shape)} greedy_sample={decoded!r}")
    params = sum(item.numel() for item in decoder.parameters())
    print(f"[decoder] parameters={params}")
    steps = int(getattr(args, "steps", 0) or 0)
    if steps > 0:
        task = TASKS[0]
        inputs, targets = teacher_forcing(tokenizer, sequences, task, device)
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.003, weight_decay=0.0)
        report_every = max(1, steps // 5)
        history = list()
        decoder.train()
        for step in range(steps):
            optimizer.zero_grad()
            logits = decoder(inputs, memory)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=tokenizer.pad_id,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            history.append(float(loss.item()))
            if step == 0 or (step + 1) % report_every == 0:
                print(f"[decoder] overfit step={step + 1}/{steps} loss={history[-1]:.4f}")
        for position, sequence in enumerate(sequences):
            sampled = decoder.generate(memory[position : position + 1], task, tokenizer, max_len=len(sequence) + 4)
            decoded = tokenizer.decode(sampled[0][len(tokenizer.prefix_ids(task)) :])
            matched = sum(1 for left, right in zip(decoded, sequence) if left == right)
            print(f"[decoder] overfit target={sequence!r} decoded={decoded!r} char_match={matched}/{len(sequence)}")
        print(f"[decoder] overfit loss {history[0]:.4f} -> {history[-1]:.4f}")
    return 0


def cmd_info(args) -> int:
    from calm.clustering import alignment_backend, mmseqs_available
    from calm.device import describe, resolve_device

    spec = resolve_device(getattr(args, "device", None) or "auto")
    print(f"calm-repro {__version__}")
    print(f"paper: {PAPER_TITLE}")
    print(f"doi: {PAPER_DOI}")
    print(f"device: {describe(spec)}")
    print(f"mmseqs2: {mmseqs_available()}")
    print(f"alignment backend: {alignment_backend()}")
    return 0


def _add_config_args(parser) -> None:
    parser.add_argument("--config", help="path to an experiment YAML file")
    parser.add_argument("--set", action="append", default=[], help="override a config field, e.g. data.min_seq_id=0.6")
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--dataset", help="override data.dataset")
    parser.add_argument("--output-dir", dest="output_dir", help="override output_dir")
    parser.add_argument("--epochs", type=int, help="override train.epochs")
    parser.add_argument("--folds", type=int, help="override folds")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="calm", description="CALM reproduction toolkit")
    parser.add_argument("--version", action="version", version=f"calm-repro {__version__}")
    sub = parser.add_subparsers(dest="command")

    prepare = sub.add_parser("prepare", help="build the CALM table from SAbDab structures")
    prepare.add_argument("--split-csv", default="data/splits_final/abag_split.csv")
    prepare.add_argument("--structure-dir", action="append", default=None)
    prepare.add_argument("--summary-csv", default=None)
    prepare.add_argument("--output", default="data/calm_dataset_raw.csv")
    prepare.add_argument("--workers", type=int, default=None)
    prepare.add_argument("--limit", type=int, default=0)
    prepare.set_defaults(func=cmd_prepare)

    embed = sub.add_parser("embed", help="cache frozen antibody and antigen embeddings")
    _add_config_args(embed)
    embed.add_argument("--overwrite", action="store_true")
    embed.set_defaults(func=cmd_embed)

    cluster = sub.add_parser("cluster", help="cluster and subsample without training")
    _add_config_args(cluster)
    cluster.set_defaults(func=cmd_cluster)

    train = sub.add_parser("train", help="alias of experiment")
    _add_config_args(train)
    train.set_defaults(func=cmd_train)

    experiment = sub.add_parser("experiment", help="run nested cross-validation for one regime")
    _add_config_args(experiment)
    experiment.set_defaults(func=cmd_experiment)

    figures = sub.add_parser("figures", help="render Figure 3 and Tables 2-4 from finished runs")
    figures.add_argument("--runs", default="artifacts/runs")
    figures.add_argument("--output", default="artifacts/report")
    figures.set_defaults(func=cmd_figures)

    decoder = sub.add_parser("decoder-check", help="drive the generative decoder module")
    decoder.add_argument("--steps", type=int, default=0)
    decoder.set_defaults(func=cmd_decoder_check)

    info = sub.add_parser("info", help="print environment and backend availability")
    info.add_argument("--device")
    info.set_defaults(func=cmd_info)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
