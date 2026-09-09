from __future__ import annotations

from pathlib import Path

import pandas as pd

from .prepare import (
    CONTACT_CUTOFF,
    EXTRA_COLUMNS,
    _flags_to_mask,
    _text,
    cdr_mask,
    chain_records,
    contact_flags,
    framework_mask,
    records_sequence,
    structure_roots,
)
from .schema import SCHEMA_COLUMNS, pair_hash

LIGHT_TYPES = ("K", "L")

_DOMAINS: dict = dict()


def select_summary_rows(frame: pd.DataFrame) -> tuple:
    report = {"input_rows": int(len(frame))}
    data = frame.copy()
    chains = data["antigen_chain"].map(_text)
    data = data[chains.str.len() > 0]
    report["after_antigen_chain"] = int(len(data))
    data = data[~data["antigen_chain"].map(_text).str.contains("|", regex=False)]
    report["after_single_antigen_chain"] = int(len(data))
    data = data[data["antigen_type"].map(_text).str.lower() == "protein"]
    report["after_protein_antigen"] = int(len(data))
    heavy = data["Hchain"].map(_text)
    light = data["Lchain"].map(_text)
    data = data[(heavy.str.len() > 0) | (light.str.len() > 0)]
    report["after_antibody_chain"] = int(len(data))
    return data.reset_index(drop=True), report


def chothia_domain(sequence: str):
    if sequence == "":
        return None
    if sequence in _DOMAINS:
        return _DOMAINS[sequence]
    from abnumber import Chain

    try:
        chain = Chain(sequence, scheme="chothia")
    except Exception:
        _DOMAINS[sequence] = None
        return None
    payload = (
        chain.seq.upper(),
        str(chain.chain_type).upper(),
        chain.cdr1_seq.upper(),
        chain.cdr2_seq.upper(),
        chain.cdr3_seq.upper(),
    )
    _DOMAINS[sequence] = payload
    return payload


def entry_path(pdb: str, roots) -> str:
    for root in roots:
        target = root / pdb / (pdb + "_sabdab.cif")
        if target.is_file():
            return str(target)
    return ""


def _cached_records(model, cache, chain_id: str):
    if chain_id not in cache:
        cache[chain_id] = chain_records(model, chain_id)
    return cache[chain_id]


def _locate(records, domain):
    if domain is None:
        return []
    sequence = records_sequence(records)
    start = sequence.find(domain[0])
    if start < 0:
        return []
    return records[start : start + len(domain[0])]


def annotate_pair(model, cache, heavy: str, light: str, antigen: str) -> dict:
    antigen_records = _cached_records(model, cache, antigen)
    if not antigen_records:
        return {"status": "missing_antigen_chain"}
    heavy_records = _cached_records(model, cache, heavy)
    light_records = _cached_records(model, cache, light)
    vh = chothia_domain(records_sequence(heavy_records))
    vl = chothia_domain(records_sequence(light_records))
    if vh is not None and vh[1] != "H":
        vh = None
    if vl is not None and vl[1] not in LIGHT_TYPES:
        vl = None
    vh_records = _locate(heavy_records, vh)
    vl_records = _locate(light_records, vl)
    if not vh_records and not vl_records:
        return {"status": "no_variable_domain"}
    if not vh_records:
        vh = None
    if not vl_records:
        vl = None
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
        "CDRH": ("", "", "") if vh is None else vh[2:5],
        "CDRL": ("", "", "") if vl is None else vl[2:5],
    }


def assemble(row, result: dict) -> dict:
    vh = result["Antibody_VH_AA"]
    vl = result["Antibody_VL_AA"]
    antigen = result["Antigen_AA"]
    cdrh = result["CDRH"]
    cdrl = result["CDRL"]
    vh_cdr1 = cdr_mask(vh, cdrh[0])
    vh_cdr2 = cdr_mask(vh, cdrh[1])
    vh_cdr3 = cdr_mask(vh, cdrh[2])
    vl_cdr1 = cdr_mask(vl, cdrl[0])
    vl_cdr2 = cdr_mask(vl, cdrl[1])
    vl_cdr3 = cdr_mask(vl, cdrl[2])
    return {
        "Unique_AG_VH_VL_Hash": pair_hash(vh, vl, antigen),
        "Antibody_VH_ID": _text(getattr(row, "HEAVY_ID", "")),
        "Antibody_VL_ID": _text(getattr(row, "LIGHT_ID", "")),
        "Antigen_ID": _text(row.PDB) + "_" + _text(row.antigen_chain),
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
        "Cluster_id": "",
        "Instance": _text(row.INSTANCE),
        "PDB_ID": _text(row.PDB),
        "SAbDab_ID": _text(getattr(row, "SABDAB_ID", "")),
        "Antigen_Chain": _text(row.antigen_chain),
        "Antibody_Cluster_id": "",
        "SAbDab_Split": "",
        "Resolution": _text(getattr(row, "resolution", "")),
        "Method": _text(getattr(row, "method", "")),
    }


def prepare_from_summary(summary_csv, structure_dir, output, limit=0, log=print) -> tuple:
    from Bio.PDB.MMCIFParser import MMCIFParser

    frame = pd.read_csv(summary_csv, low_memory=False)
    selected, report = select_summary_rows(frame)
    if limit and limit > 0:
        selected = selected.iloc[: int(limit)].reset_index(drop=True)
        report["limited_to"] = int(len(selected))
    roots = structure_roots(structure_dir)
    groups: dict = dict()
    for row in selected.itertuples():
        groups.setdefault(_text(row.PDB), []).append(row)
    log(f"[prepare] annotating {len(selected)} pairs across {len(groups)} PDB entries")
    parser = MMCIFParser(QUIET=True)
    statuses: dict = dict()
    rows = []
    done = 0
    for pdb in sorted(groups):
        items = groups[pdb]
        path = entry_path(pdb, roots)
        if path == "":
            statuses["structure_not_found"] = statuses.get("structure_not_found", 0) + len(items)
            done += len(items)
            continue
        try:
            model = parser.get_structure("s", path)[0]
        except Exception as error:
            key = "parse_error:" + type(error).__name__
            statuses[key] = statuses.get(key, 0) + len(items)
            done += len(items)
            continue
        cache: dict = dict()
        for row in items:
            result = annotate_pair(model, cache, _text(row.Hchain), _text(row.Lchain), _text(row.antigen_chain))
            status = result.get("status", "unknown")
            statuses[status] = statuses.get(status, 0) + 1
            if status == "ok":
                rows.append(assemble(row, result))
        done += len(items)
        if done % 500 < len(items):
            log(f"[prepare] annotated {done}/{len(selected)} pairs")
    for key, value in sorted(statuses.items()):
        report["status_" + key] = int(value)
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
