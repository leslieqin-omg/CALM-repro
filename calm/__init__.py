__version__ = "1.0.0"
PAPER_TITLE = "Contrastive learning for antibody-antigen sequence-to-specificity prediction"
PAPER_DOI = "10.64898/2026.02.25.707916"

__all__ = ["__version__", "PAPER_DOI", "PAPER_TITLE"]

from . import (
    schema,
    config,
    dataset,
    model,
    train,
    evaluate,
    experiment,
    clustering,
    embeddings,
    figures,
    losses,
    metrics,
    sabdab,
    prepare,
    splits,
)

__all__ += [
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
]
