from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from calm import schema
from calm.config import EmbeddingConfig
from calm.device import DeviceSpec

TOKENS_FILE = "tokens.f16"
INDEX_FILE = "index.json"


def sequence_key(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:32]


class RaggedEmbeddingWriter:
    def __init__(self, root: str | Path, dim: int, source: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dim = int(dim)
        self.source = source
        self.entries: dict[str, list[int]] = dict()
        self.cursor = 0
        self.handle = open(self.root / TOKENS_FILE, "wb")

    def add(self, sequence: str, array: np.ndarray) -> None:
        key = sequence_key(sequence)
        if key in self.entries:
            return
        block = np.ascontiguousarray(array, dtype=np.float16)
        if block.ndim != 2 or block.shape[1] != self.dim:
            raise ValueError(f"expected (L, {self.dim}) embedding, got {block.shape}")
        block.tofile(self.handle)
        self.entries[key] = [self.cursor, int(block.shape[0])]
        self.cursor += int(block.shape[0])

    def close(self) -> None:
        self.handle.close()
        meta = {"dim": self.dim, "source": self.source, "tokens": self.cursor, "entries": self.entries}
        (self.root / INDEX_FILE).write_text(json.dumps(meta), encoding="utf-8")

    def __enter__(self) -> "RaggedEmbeddingWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class RaggedEmbeddingStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        meta = json.loads((self.root / INDEX_FILE).read_text(encoding="utf-8"))
        self.dim = int(meta["dim"])
        self.source = str(meta.get("source", "unknown"))
        self.entries = {key: (int(value[0]), int(value[1])) for key, value in meta["entries"].items()}
        self._data: np.ndarray | None = None

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            flat = np.memmap(self.root / TOKENS_FILE, dtype=np.float16, mode="r")
            self._data = flat.reshape(-1, self.dim)
        return self._data

    def has(self, sequence: str) -> bool:
        return sequence_key(sequence) in self.entries

    def get(self, sequence: str) -> np.ndarray:
        start, length = self.entries[sequence_key(sequence)]
        return np.asarray(self.data[start:start + length])

    def __len__(self) -> int:
        return len(self.entries)


class HashedEncoder:
    def __init__(self, dim: int, salt: str) -> None:
        self.dim = int(dim)
        self.salt = salt
        self._table = self._build_table()

    def _build_table(self) -> np.ndarray:
        alphabet = "ACDEFGHIKLMNPQRSTVWYXBZUO"
        table = np.zeros((128, self.dim), dtype=np.float32)
        for residue in alphabet:
            seed = int(hashlib.sha256(f"{self.salt}:{residue}".encode("utf-8")).hexdigest()[:8], 16)
            table[ord(residue)] = np.random.default_rng(seed).normal(0.0, 1.0, self.dim)
        return table

    def encode(self, sequences: Sequence[str]) -> list[np.ndarray]:
        outputs = []
        for sequence in sequences:
            codes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8).astype(np.int64)
            base = self._table[codes]
            shifted = np.roll(base, 1, axis=0)
            shifted[0] = 0.0
            outputs.append((base + 0.3 * shifted).astype(np.float32))
        return outputs


class AntiBERTyEncoder:
    def __init__(self, device: torch.device) -> None:
        from antiberty import AntiBERTyRunner

        self.runner = AntiBERTyRunner(device=device)
        self.dim = 512

    def encode(self, sequences: Sequence[str]) -> list[np.ndarray]:
        with torch.no_grad():
            embeddings = self.runner.embed(list(sequences))
        outputs = []
        for sequence, tensor in zip(sequences, embeddings):
            array = tensor.detach().float().cpu().numpy()
            outputs.append(_strip_special(array, len(sequence)))
        return outputs


class Esm2Encoder:
    def __init__(self, device: torch.device, model_name: str) -> None:
        import esm

        loader = getattr(esm.pretrained, model_name)
        model, alphabet = loader()
        self.model = model.eval().to(device)
        self.batch_converter = alphabet.get_batch_converter()
        self.repr_layer = int(model.num_layers)
        self.dim = int(model.embed_dim)
        self.device = device

    def encode(self, sequences: Sequence[str]) -> list[np.ndarray]:
        batch = [(str(i), seq) for i, seq in enumerate(sequences)]
        _labels, _strs, tokens = self.batch_converter(batch)
        tokens = tokens.to(self.device)
        with torch.no_grad():
            out = self.model(tokens, repr_layers=[self.repr_layer], return_contacts=False)
        hidden = out["representations"][self.repr_layer].float().cpu().numpy()
        return [hidden[i, 1:len(seq) + 1] for i, seq in enumerate(sequences)]


def _strip_special(array: np.ndarray, length: int) -> np.ndarray:
    if array.shape[0] == length:
        return array
    if array.shape[0] >= length + 2:
        return array[1:length + 1]
    return array[:length]


def build_encoder(kind: str, device: torch.device, dim: int, salt: str) -> object:
    name = (kind or "").strip().lower()
    if name == "hashed":
        return HashedEncoder(dim, salt)
    if name == "antiberty":
        return AntiBERTyEncoder(device)
    if name.startswith("esm"):
        return Esm2Encoder(device, kind)
    raise ValueError(f"unknown encoder: {kind}")


def _batched(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def encode_to_store(
    sequences: Sequence[str],
    encoder: object,
    root: str | Path,
    dim: int,
    source: str,
    batch_size: int,
    progress: bool = True,
) -> Path:
    ordered = sorted(set(sequences), key=len)
    writer = RaggedEmbeddingWriter(root, dim, source)
    done = 0
    for chunk in _batched(ordered, max(1, batch_size)):
        arrays = encoder.encode(chunk)
        for sequence, array in zip(chunk, arrays):
            writer.add(sequence, array)
        done += len(chunk)
        if progress:
            print(f"[embed] {source}: {done}/{len(ordered)}", flush=True)
    writer.close()
    return Path(root)


def _covers(root: Path, sequences: Sequence[str]) -> bool:
    if not (root / INDEX_FILE).exists():
        return False
    store = RaggedEmbeddingStore(root)
    missing = sum(1 for sequence in set(sequences) if not store.has(sequence))
    if missing:
        print(f"[embed] {root.name}: stale cache, {missing} sequences missing, re-encoding", flush=True)
        return False
    return True


def embed_dataset(
    frame,
    config: EmbeddingConfig,
    spec: DeviceSpec,
    out_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, str]:
    root = Path(out_dir)
    antibody_root = root / "antibody"
    antigen_root = root / "antigen"
    chains = schema.unique_antibody_chains(frame)
    antigens = schema.unique_antigens(frame)
    if overwrite or not _covers(antibody_root, chains):
        encoder = build_encoder(config.antibody_model, spec.device, config.antibody_dim, "antibody")
        dim = int(getattr(encoder, "dim", config.antibody_dim))
        encode_to_store(chains, encoder, antibody_root, dim, config.antibody_model, config.batch_size)
    if overwrite or not _covers(antigen_root, antigens):
        encoder = build_encoder(config.antigen_model, spec.device, config.antigen_dim, "antigen")
        dim = int(getattr(encoder, "dim", config.antigen_dim))
        encode_to_store(antigens, encoder, antigen_root, dim, config.antigen_model, config.batch_size)
    return {"antibody": str(antibody_root), "antigen": str(antigen_root)}


def load_stores(out_dir: str | Path) -> tuple[RaggedEmbeddingStore, RaggedEmbeddingStore]:
    root = Path(out_dir)
    return RaggedEmbeddingStore(root / "antibody"), RaggedEmbeddingStore(root / "antigen")
