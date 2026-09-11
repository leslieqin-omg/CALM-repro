import importlib

__version__ = "1.0.0"
PAPER_TITLE = "Contrastive learning for antibody-antigen sequence-to-specificity prediction"
PAPER_DOI = "10.64898/2026.02.25.707916"

_SUBMODULES = (
    "schema",
    "config",
    "dataset",
    "model",
    "train",
    "evaluate",
    "experiment",
    "clustering",
    "embeddings",
    "figures",
    "losses",
    "metrics",
    "sabdab",
    "prepare",
    "splits",
)

__all__ = ["__version__", "PAPER_DOI", "PAPER_TITLE", *_SUBMODULES]


def __getattr__(name):
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
