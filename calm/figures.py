from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

MASKED_COLOR = "#e07a2f"
UNMASKED_COLOR = "#2f6fb0"
RANDOM_HATCH = "//"

FIGURE_ROWS = (
    ("40%", "ag40", "pa_ep40"),
    ("60%", "ag60", "pa_ep60"),
    ("80%", "ag80", "pa_ep80"),
    ("RS", "rs_full", "rs_masked"),
)

TABLE_SPECS = (
    ("table2_full_length", "Table 2: full-length antibody-antigen retrieval", ("ag40", "ag60", "ag80", "rs_full")),
    ("table3_paratope_epitope", "Table 3: paratope-epitope masked retrieval", ("pa_ep40", "pa_ep60", "pa_ep80", "rs_masked")),
    ("table4_unseen_complex", "Table 4: unseen-complex regimes", ("uc90", "uc95")),
)


def load_runs(runs_dir) -> dict:
    root = Path(runs_dir)
    out: dict = dict()
    if not root.exists():
        return out
    for path in sorted(root.glob("*/results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[str(payload.get("name") or path.parent.name)] = payload
    return out


def _entry(payload, direction: str, k: int):
    summary = (payload.get("summary") or dict()).get(direction) or dict()
    item = summary.get(str(int(k)))
    if not item:
        return None
    return float(item.get("mean", 0.0)), float(item.get("ci95", 0.0))


def _percent(value) -> str:
    return f"{value * 100:.2f}"


def _cell(payload, direction: str, k: int) -> str:
    item = _entry(payload, direction, k)
    if item is None:
        return "n/a"
    return _percent(item[0]) + " +/- " + _percent(item[1])


def table_rows(runs: dict, names, ks) -> list:
    rows = list()
    for name in names:
        payload = runs.get(name)
        if payload is None:
            continue
        row = dict()
        row["setting"] = str(payload.get("label") or name)
        row["run"] = name
        row["mask_mode"] = str(payload.get("mask_mode"))
        row["cluster_on"] = str(payload.get("cluster_on"))
        row["pairs"] = int(payload.get("pairs", 0))
        row["clusters"] = int(payload.get("clusters", 0))
        for k in ks:
            row[f"recall_at_{k}"] = _cell(payload, "mean", k)
            row[f"ab_to_ag_at_{k}"] = _cell(payload, "ab_to_ag", k)
            row[f"ag_to_ab_at_{k}"] = _cell(payload, "ag_to_ab", k)
        baseline = payload.get("random_baseline") or dict()
        for k in ks:
            value = baseline.get(str(int(k)))
            row[f"theoretical_at_{k}"] = _percent(float(value)) if value is not None else "n/a"
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    columns = list(rows[0].keys())
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join('"' + str(row.get(column, "")) + '"' for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def markdown_table(title: str, rows: list, ks) -> str:
    if not rows:
        return "## " + title + "\n\nno finished runs found\n"
    header = ["Setting", "Pairs", "Clusters"]
    for k in ks:
        header.append(f"Recall@{k} (%)")
    for k in ks:
        header.append(f"Random@{k} (%)")
    out = ["## " + title, "", "| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        cells = [row["setting"], str(row["pairs"]), str(row["clusters"])]
        for k in ks:
            cells.append(row[f"recall_at_{k}"])
        for k in ks:
            cells.append(row[f"theoretical_at_{k}"])
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def figure3(runs: dict, path: Path, ks=(1, 5, 10)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(ks), figsize=(4.2 * len(ks), 4.0), sharex=True)
    if len(ks) == 1:
        axes = [axes]
    width = 0.36
    for column, k in enumerate(ks):
        axis = axes[column]
        labels = list()
        drawn = 0
        baseline_value = None
        for position, (label, unmasked_name, masked_name) in enumerate(FIGURE_ROWS):
            labels.append(label)
            for offset, (name, color) in enumerate(((unmasked_name, UNMASKED_COLOR), (masked_name, MASKED_COLOR))):
                payload = runs.get(name)
                if payload is None:
                    continue
                item = _entry(payload, "mean", k)
                if item is None:
                    continue
                baseline = (payload.get("random_baseline") or dict()).get(str(int(k)))
                if baseline is not None:
                    baseline_value = float(baseline) * 100.0
                centre = position + (offset - 0.5) * width
                hatch = RANDOM_HATCH if bool(payload.get("random_shuffle")) else None
                axis.bar(
                    centre,
                    item[0] * 100.0,
                    width=width,
                    color=color,
                    edgecolor="black",
                    linewidth=0.7,
                    hatch=hatch,
                    yerr=item[1] * 100.0,
                    capsize=3.0,
                )
                drawn += 1
        if baseline_value is not None:
            axis.axhline(baseline_value, linestyle="--", color="#555555", linewidth=1.0)
            axis.text(
                len(FIGURE_ROWS) - 0.5,
                baseline_value,
                f"k/256 = {baseline_value:.2f}%",
                fontsize=7,
                color="#555555",
                ha="right",
                va="bottom",
            )
        axis.set_xticks(range(len(FIGURE_ROWS)))
        axis.set_xticklabels(labels)
        axis.set_title(f"Recall@{k}")
        axis.set_xlabel("antigen sequence identity threshold")
        if column == 0:
            axis.set_ylabel("Recall (%)")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if drawn == 0:
            axis.text(0.5, 0.5, "no runs", transform=axis.transAxes, ha="center", va="center")
    handles = [
        Patch(facecolor=UNMASKED_COLOR, edgecolor="black", label="full-length Ab-Ag"),
        Patch(facecolor=MASKED_COLOR, edgecolor="black", label="masked paratope-epitope"),
        Patch(facecolor="white", edgecolor="black", hatch=RANDOM_HATCH, label="random-shuffle control"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    figure.suptitle("Figure 3 reproduction: bidirectional retrieval recall, 5-fold mean with 95% CI")
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def uc_figure(runs: dict, path: Path, ks=(1, 5, 10)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = ("uc90", "uc95")
    available = [name for name in names if name in runs]
    figure, axis = plt.subplots(figsize=(5.4, 4.0))
    if not available:
        axis.text(0.5, 0.5, "no unseen-complex runs", transform=axis.transAxes, ha="center", va="center")
    else:
        width = 0.8 / len(available)
        for offset, name in enumerate(available):
            payload = runs[name]
            values = list()
            errors = list()
            for k in ks:
                item = _entry(payload, "mean", k)
                values.append(0.0 if item is None else item[0] * 100.0)
                errors.append(0.0 if item is None else item[1] * 100.0)
            centres = [index + (offset - (len(available) - 1) / 2.0) * width for index in range(len(ks))]
            axis.bar(
                centres,
                values,
                width=width,
                yerr=errors,
                capsize=3.0,
                edgecolor="black",
                linewidth=0.7,
                label=str(payload.get("label") or name),
            )
        baseline = runs[available[0]].get("random_baseline") or dict()
        for index, k in enumerate(ks):
            value = baseline.get(str(int(k)))
            if value is None:
                continue
            axis.plot(
                [index - 0.45, index + 0.45],
                [float(value) * 100.0, float(value) * 100.0],
                linestyle="--",
                color="#555555",
                linewidth=1.0,
            )
        axis.legend(frameon=False)
    axis.set_xticks(range(len(ks)))
    axis.set_xticklabels([f"Recall@{k}" for k in ks])
    axis.set_ylabel("Recall (%)")
    axis.set_title("Table 4 reproduction: unseen-complex retrieval")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def build_all(runs_dir, output_dir, ks=(1, 5, 10)) -> list:
    runs = load_runs(runs_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = list()
    sections = list()
    for slug, title, names in TABLE_SPECS:
        rows = table_rows(runs, names, ks)
        written.append(str(write_csv(out / (slug + ".csv"), rows)))
        sections.append(markdown_table(title, rows, ks))
    report = out / "tables.md"
    header = "# CALM reproduction results\n\nrecall is the per-query Hit@k rate averaged over both retrieval directions and 5 folds\n"
    report.write_text(header + "\n" + "\n".join(sections), encoding="utf-8")
    written.append(str(report))
    written.append(str(figure3(runs, out / "figure3_recall.png", ks)))
    written.append(str(uc_figure(runs, out / "figure4_unseen_complex.png", ks)))
    index = out / "runs_index.json"
    payload = dict()
    for name, run in runs.items():
        payload[name] = {
            "label": run.get("label"),
            "mask_mode": run.get("mask_mode"),
            "cluster_on": run.get("cluster_on"),
            "min_seq_id": run.get("min_seq_id"),
            "random_shuffle": run.get("random_shuffle"),
            "pairs": run.get("pairs"),
            "clusters": run.get("clusters"),
            "summary": run.get("summary"),
            "random_baseline": run.get("random_baseline"),
        }
    index.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    written.append(str(index))
    return written
