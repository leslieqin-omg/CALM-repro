from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "Antibody_VH_AA",
    "Antibody_VL_AA",
    "Antigen_AA",
)

SCHEMA_COLUMNS = (
    "Unique_AG_VH_VL_Hash",
    "Antibody_VH_ID",
    "Antibody_VL_ID",
    "Antigen_ID",
    "Antibody_VH_AA",
    "Antibody_VL_AA",
    "Antigen_AA",
    "Antibody_VH_Paratope",
    "Antibody_VL_Paratope",
    "Antigen_Epitope",
    "VH_CDR1_Chothia_Mask",
    "VH_CDR2_Chothia_Mask",
    "VH_CDR3_Chothia_Mask",
    "VH_FW_Chothia_Mask",
    "VL_CDR1_Chothia_Mask",
    "VL_CDR2_Chothia_Mask",
    "VL_CDR3_Chothia_Mask",
    "VL_FW_Chothia_Mask",
    "Antigen_AA_Uniprot",
    "Antigen_UniProt_ID",
    "Antibody_ScFv_ID",
    "Antibody_ScFv_AA",
    "Binder_Status",
    "Antibody_Type",
    "Cluster_id",
)

MAX_VARIABLE_LEN = 141
MIN_ANTIGEN_LEN = 19
MAX_ANTIGEN_LEN = 550


def read_table(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(target)
    if suffix in (".tsv", ".tab", ".txt"):
        return pd.read_csv(target, sep="\t", dtype=str, keep_default_na=False)
    if suffix == ".jsonl":
        return pd.read_json(target, lines=True, dtype=str)
    if suffix == ".json":
        return pd.read_json(target, dtype=str)
    return pd.read_csv(target, dtype=str, keep_default_na=False)


def _clean_sequence(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if text in ("NAN", "NONE", "NA", "-"):
        return ""
    return "".join(ch for ch in text if ch.isalpha())


def _clean_mask(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in ("nan", "none", "na", "-"):
        return ""
    return "".join(ch for ch in text if ch in "01")


def pair_hash(vh: str, vl: str, antigen: str) -> str:
    payload = "|".join((vh, vl, antigen)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"dataset is missing required CALM schema columns: {missing}")
    data = frame.copy()
    for column in SCHEMA_COLUMNS:
        if column not in data.columns:
            data[column] = ""
    for column in ("Antibody_VH_AA", "Antibody_VL_AA", "Antigen_AA", "Antigen_AA_Uniprot", "Antibody_ScFv_AA"):
        data[column] = data[column].map(_clean_sequence)
    for column in ("Antibody_VH_Paratope", "Antibody_VL_Paratope", "Antigen_Epitope"):
        data[column] = data[column].map(_clean_mask)
    data["Binder_Status"] = data["Binder_Status"].astype(str).str.strip().str.upper().replace("", "Y")
    data["Cluster_id"] = data["Cluster_id"].astype(str).str.strip()
    blank_hash = data["Unique_AG_VH_VL_Hash"].astype(str).str.strip() == ""
    computed = [pair_hash(r.Antibody_VH_AA, r.Antibody_VL_AA, r.Antigen_AA) for r in data.itertuples()]
    data.loc[blank_hash, "Unique_AG_VH_VL_Hash"] = pd.Series(computed, index=data.index)[blank_hash]
    return data.reset_index(drop=True)


def _mask_is_consistent(mask: str, sequence: str) -> bool:
    if sequence == "":
        return mask == ""
    return mask == "" or len(mask) == len(sequence)


def validate(frame: pd.DataFrame, require_masks: bool = False) -> tuple[pd.DataFrame, dict[str, int]]:
    data = normalize(frame)
    report = {"input_rows": int(len(data))}
    before = len(data)
    data = data[data["Binder_Status"] == "Y"]
    report["dropped_non_binder"] = int(before - len(data))
    before = len(data)
    data = data.drop_duplicates(subset=["Antibody_VH_AA", "Antibody_VL_AA", "Antigen_AA"])
    report["dropped_duplicate_pairs"] = int(before - len(data))
    before = len(data)
    has_chain = (data["Antibody_VH_AA"].str.len() > 0) | (data["Antibody_VL_AA"].str.len() > 0)
    data = data[has_chain]
    report["dropped_no_antibody_chain"] = int(before - len(data))
    before = len(data)
    ag_len = data["Antigen_AA"].str.len()
    data = data[(ag_len >= MIN_ANTIGEN_LEN) & (ag_len <= MAX_ANTIGEN_LEN)]
    report["dropped_antigen_length"] = int(before - len(data))
    before = len(data)
    vh_ok = data["Antibody_VH_AA"].str.len() <= MAX_VARIABLE_LEN
    vl_ok = data["Antibody_VL_AA"].str.len() <= MAX_VARIABLE_LEN
    data = data[vh_ok & vl_ok]
    report["dropped_variable_length"] = int(before - len(data))
    before = len(data)
    rows = list(data.itertuples())
    keep = [
        _mask_is_consistent(r.Antibody_VH_Paratope, r.Antibody_VH_AA)
        and _mask_is_consistent(r.Antibody_VL_Paratope, r.Antibody_VL_AA)
        and _mask_is_consistent(r.Antigen_Epitope, r.Antigen_AA)
        for r in rows
    ]
    data = data[pd.Series(keep, index=data.index)]
    report["dropped_mask_mismatch"] = int(before - len(data))
    if require_masks:
        before = len(data)
        epitope_ok = data["Antigen_Epitope"].str.contains("1", regex=False)
        paratope_ok = data["Antibody_VH_Paratope"].str.contains("1", regex=False)
        paratope_ok = paratope_ok | data["Antibody_VL_Paratope"].str.contains("1", regex=False)
        data = data[epitope_ok & paratope_ok]
        report["dropped_empty_interface"] = int(before - len(data))
    data = data.reset_index(drop=True)
    report["output_rows"] = int(len(data))
    report["unique_antibodies"] = int(pd.unique(data["Antibody_VH_AA"] + "|" + data["Antibody_VL_AA"]).size)
    report["unique_antigens"] = int(data["Antigen_AA"].nunique())
    return data, report


def load_dataset(path: str | Path, require_masks: bool = False) -> tuple[pd.DataFrame, dict[str, int]]:
    return validate(read_table(path), require_masks=require_masks)


def mask_to_array(mask: str, length: int) -> np.ndarray:
    if mask == "":
        return np.ones(length, dtype=bool)
    values = np.frombuffer(mask.encode("ascii"), dtype=np.uint8) == ord("1")
    if values.size == length:
        return values.copy()
    out = np.zeros(length, dtype=bool)
    limit = min(length, values.size)
    out[:limit] = values[:limit]
    return out


def unique_antibody_chains(frame: pd.DataFrame) -> list[str]:
    chains = set(frame["Antibody_VH_AA"]) | set(frame["Antibody_VL_AA"])
    chains.discard("")
    return sorted(chains)


def unique_antigens(frame: pd.DataFrame) -> list[str]:
    antigens = set(frame["Antigen_AA"])
    antigens.discard("")
    return sorted(antigens)
