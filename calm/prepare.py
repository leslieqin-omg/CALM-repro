from __future__ import annotations

import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pandas as pd

from .schema import SCHEMA_COLUMNS, pair_hash

CONTACT_CUTOFF = 5.0

SKIP_RESIDUES = ("HOH", "DOD", "UNK")

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
}

EXTRA_COLUMNS = (
    "Instance",
    "PDB_ID",
    "SAbDab_ID",
    "Antigen_Chain",
    "Antibody_Cluster_id",
    "SAbDab_Split",
    "Resolution",
    "Method",
)


def _text(value) -> str:
    if value is None:
        return ""
    out = str(value).strip()
    if out.lower() in ("nan", "none", "na", "<na>"):
        return ""
    return out


def _flags_to_mask(flags) -> str:
    return "".join("1" if flag else "0" for flag in flags)


def chain_records(model, chain_id: str):
    if chain_id == "" or chain_id not in model:
        return []
    out = []
    for residue in model[chain_id]:
        if residue.id[0] != " ":
            continue
        name = residue.get_resname().strip().upper()
        if name in SKIP_RESIDUES:
            continue
        atoms = [atom for atom in residue if _text(atom.element).upper() != "H"]
        if not atoms:
            continue
        out.append((THREE_TO_ONE.get(name, "X"), atoms))
    return out


def records_sequence(records) -> str:
    return "".join(item[0] for item in records)


def contact_flags(target, partners, cutoff: float):
    from Bio.PDB.NeighborSearch import NeighborSearch

    if not target:
        return []
    atoms = [atom for item in partners for atom in item[1]]
    if not atoms:
        return [False] * len(target)
    search = NeighborSearch(atoms)
    flags = []
    for item in target:
        hit = False
        for atom in item[1]:
            if search.search(atom.coord, cutoff):
                hit = True
                break
        flags.append(hit)
    return flags


def cdr_mask(sequence: str, segment: str) -> str:
    if sequence == "":
        return ""
    flags = [False] * len(sequence)
    if segment:
        start = sequence.find(segment)
        if start >= 0:
            for offset in range(len(segment)):
                flags[start + offset] = True
    return _flags_to_mask(flags)


def framework_mask(sequence: str, masks) -> str:
    if sequence == "":
        return ""
    flags = [True] * len(sequence)
    for mask in masks:
        if len(mask) != len(sequence):
            continue
        for index, symbol in enumerate(mask):
            if symbol == "1":
                flags[index] = False
    return _flags_to_mask(flags)


def annotate(job):
    from Bio.PDB.MMCIFParser import MMCIFParser

    path, heavy, light, antigen, vh_expect, vl_expect = job
    if path == "":
        return {"status": "structure_not_found"}
    parser = MMCIFParser(QUIET=True)
    try:
        model = parser.get_structure("s", path)[0]
    except Exception as error:
        return {"status": "parse_error:" + type(error).__name__}
    antigen_records = chain_records(model, antigen)
    if not antigen_records:
        return {"status": "missing_antigen_chain"}
    heavy_records = chain_records(model, heavy)
    light_records = chain_records(model, light)
    heavy_sequence = records_sequence(heavy_records)
    light_sequence = records_sequence(light_records)
    vh_records = []
    vl_records = []
    if vh_expect:
        start = heavy_sequence.find(vh_expect)
        if start < 0:
            return {"status": "vh_not_aligned"}
        vh_records = heavy_records[start : start + len(vh_expect)]
    if vl_expect:
        start = light_sequence.find(vl_expect)
        if start < 0:
            return {"status": "vl_not_aligned"}
        vl_records = light_records[start : start + len(vl_expect)]
    if not vh_records and not vl_records:
        return {"status": "no_antibody_chain"}
    antibody_records = vh_records + vl_records
    epitope = contact_flags(antigen_records, antibody_records, CONTACT_CUTOFF)
    vh_paratope = contact_flags(vh_records, antigen_records, CONTACT_CUTOFF)
    vl_paratope = contact_flags(vl_records, antigen_records, CONTACT_CUTOFF)
    return {
        "status": "ok",
        "Antibody_VH_AA": records_sequence(vh_records),
        "Antibody_VL_AA": records_sequence(vl_records),
        "Antigen_AA": records_sequence(antigen_records),
        "Antibody_VH_Paratope": _flags_to_mask(vh_paratope),
        "Antibody_VL_Paratope": _flags_to_mask(vl_paratope),
        "Antigen_Epitope": _flags_to_mask(epitope),
    }


def select_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"input_rows": int(len(frame))}
    data = frame.copy()
    data = data[data["holo"].astype(str).str.lower() == "true"]
    report["after_holo"] = int(len(data))
    data = data[data["agtypes"].map(_text) == "PROTEIN"]
    report["after_single_protein_antigen"] = int(len(data))
    heavy = data["Hchain"].map(_text)
    light = data["Lchain"].map(_text)
    data = data[(heavy.str.len() > 0) | (light.str.len() > 0)]
    report["after_antibody_chain"] = int(len(data))
    vh = data["VH_numerable_seq"].map(_text)
    vl = data["VL_numerable_seq"].map(_text)
    data = data[(vh.str.len() > 0) | (vl.str.len() > 0)]
    report["after_variable_sequence"] = int(len(data))
    return data.reset_index(drop=True), report


def _pool_context():
    methods = multiprocessing.get_all_start_methods()
    order = ("forkserver", "spawn") if sys.platform == "darwin" else ("fork", "forkserver", "spawn")
    for name in order:
        if name in methods:
            return multiprocessing.get_context(name)
    return None


def _iter_results(jobs, workers, log):
    if workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=_pool_context()) as pool:
                for result in pool.map(annotate, jobs, chunksize=16):
                    yield result
            return
        except (OSError, BrokenProcessPool) as error:
            log(f"[prepare] process pool unavailable ({type(error).__name__}); falling back to a single worker")
    for job in jobs:
        yield annotate(job)


def structure_roots(structure_dir) -> list:
    if isinstance(structure_dir, (str, Path)):
        items = [structure_dir]
    else:
        items = list(structure_dir)
    return [Path(item) for item in items if _text(item)]


def resolve_structure(instance: str, roots) -> tuple:
    entry = "_".join(instance.split("_")[:2])
    for root in roots:
        direct = root / (instance + ".cif")
        if direct.is_file():
            return str(direct), "instance"
        whole = root / entry / (entry + "_sabdab.cif")
        if whole.is_file():
            return str(whole), "entry"
    return "", "unresolved"


def build_rows(selected: pd.DataFrame, structure_dir, workers, log) -> tuple[list, dict]:
    roots = structure_roots(structure_dir)
    jobs = []
    sources = dict()
    for row in selected.itertuples():
        path, origin = resolve_structure(str(row.INSTANCE), roots)
        sources[origin] = sources.get(origin, 0) + 1
        jobs.append(
            (
                path,
                _text(row.Hchain),
                _text(row.Lchain),
                _text(row.agchains),
                _text(row.VH_numerable_seq).upper(),
                _text(row.VL_numerable_seq).upper(),
            )
        )
    log(f"[prepare] structure sources: {sources}")
    statuses: dict = {}
    rows = []
    source = list(selected.itertuples())
    count = 0
    for row, result in zip(source, _iter_results(jobs, workers, log)):
        count += 1
        if count % 500 == 0:
            log(f"[prepare] annotated {count}/{len(jobs)} structures")
        status = result.get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        if status != "ok":
            continue
        rows.append(_assemble(row, result))
    statuses_report = {f"status_{key}": int(value) for key, value in sorted(statuses.items())}
    for key, value in sorted(sources.items()):
        statuses_report["structure_" + key] = int(value)
    return rows, statuses_report


def _assemble(row, result) -> dict:
    vh = result["Antibody_VH_AA"]
    vl = result["Antibody_VL_AA"]
    antigen = result["Antigen_AA"]
    vh_cdr1 = cdr_mask(vh, _text(row.CDRH1).upper())
    vh_cdr2 = cdr_mask(vh, _text(row.CDRH2).upper())
    vh_cdr3 = cdr_mask(vh, _text(row.CDRH3).upper())
    vl_cdr1 = cdr_mask(vl, _text(row.CDRL1).upper())
    vl_cdr2 = cdr_mask(vl, _text(row.CDRL2).upper())
    vl_cdr3 = cdr_mask(vl, _text(row.CDRL3).upper())
    payload = {
        "Unique_AG_VH_VL_Hash": pair_hash(vh, vl, antigen),
        "Antibody_VH_ID": _text(getattr(row, "HEAVY_ID", "")),
        "Antibody_VL_ID": _text(getattr(row, "LIGHT_ID", "")),
        "Antigen_ID": _text(row.PDB_ID) + "_" + _text(row.agchains),
        "Antibody_VH_AA": vh,
        "Antibody_VL_AA": vl,
        "Antigen_AA": antigen,
        "Antibody_VH_Paratope": result["Antibody_VH_Paratope"],
        "Antibody_VL_Paratope": result["Antibody_VL_Paratope"],
        "Antigen_Epitope": result["Antigen_Epitope"],
        "VH_CDR1_Chothia_Mask": vh_cdr1,
        "VH_CDR2_Chothia_Mask": vh_cdr2,
        "VH_CDR3_Chothia_Mask": vh_cdr3,
        "VH_FW_Chothia_Mask": framework_mask(vh, (vh_cdr1, vh_cdr2, vh_cdr3)),
        "VL_CDR1_Chothia_Mask": vl_cdr1,
        "VL_CDR2_Chothia_Mask": vl_cdr2,
        "VL_CDR3_Chothia_Mask": vl_cdr3,
        "VL_FW_Chothia_Mask": framework_mask(vl, (vl_cdr1, vl_cdr2, vl_cdr3)),
        "Antigen_AA_Uniprot": "",
        "Antigen_UniProt_ID": "",
        "Antibody_ScFv_ID": "",
        "Antibody_ScFv_AA": "",
        "Binder_Status": "Y",
        "Antibody_Type": _text(getattr(row, "type", "")),
        "Cluster_id": _text(getattr(row, "agclusters", "")),
        "Instance": _text(row.INSTANCE),
        "PDB_ID": _text(row.PDB_ID),
        "SAbDab_ID": _text(getattr(row, "SABDAB_ID", "")),
        "Antigen_Chain": _text(row.agchains),
        "Antibody_Cluster_id": _text(getattr(row, "ab_cluster", "")),
        "SAbDab_Split": _text(getattr(row, "ab_ag_split", "")),
        "Resolution": _text(getattr(row, "resolution", "")),
        "Method": _text(getattr(row, "method", "")),
    }
    return payload


def prepare_dataset(split_csv, structure_dir, output, workers=None, limit=0, log=print):
    frame = pd.read_csv(split_csv, low_memory=False)
    selected, report = select_rows(frame)
    if limit and limit > 0:
        selected = selected.iloc[: int(limit)].reset_index(drop=True)
        report["limited_to"] = int(len(selected))
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    log(f"[prepare] annotating {len(selected)} structures with {workers} workers")
    rows, statuses = build_rows(selected, structure_dir, int(workers), log)
    report.update(statuses)
    columns = list(SCHEMA_COLUMNS) + list(EXTRA_COLUMNS)
    table = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
    before = len(table)
    table = table.drop_duplicates(subset=["Antibody_VH_AA", "Antibody_VL_AA", "Antigen_AA"]).reset_index(drop=True)
    report["dropped_duplicate_pairs"] = int(before - len(table))
    report["output_rows"] = int(len(table))
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(target, index=False)
    log(f"[prepare] wrote {len(table)} pairs to {target}")
    return table, report
