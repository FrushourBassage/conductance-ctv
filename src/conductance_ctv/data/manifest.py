"""The public corpora used for pretraining, and what they are used for.

Both entries below are the resources the manuscript cites for pretraining. They
supply appearance invariance and abdominal vascular labels; neither supplies any
evaluation data and neither carries a clinical variable of this study. The
release records their pinned versions, licences and sources so a reader can fetch
them, and never bundles them.

Ref: Data availability; Sec. 6.4 ("the partners for pretraining were taken from
public CT corpus"); references [69] and [120].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Concept DOI the manuscript prints next to release 2.0.1. It resolves to the
# newest release of the same record family, so it is recorded beside the
# version-pinned DOI rather than instead of it.
TOTALSEGMENTATOR_CONCEPT_DOI = "https://doi.org/10.5281/zenodo.6802613"

TOTALSEGMENTATOR_CLASSES_OF_INTEREST: tuple[str, ...] = (
    "inferior_vena_cava",
    "iliac_vena_left",
    "iliac_vena_right",
    "iliac_artery_left",
    "iliac_artery_right",
)

ABDOMENATLAS_CLASSES_OF_INTEREST: tuple[str, ...] = ("postcava", "aorta")


@dataclass(frozen=True)
class PublicCorpus:
    """A pretraining corpus with the provenance the release has to state."""

    name: str
    version: str
    licence: str
    source_url: str
    code_url: str
    role: str
    label_classes: tuple[str, ...]
    approximate_size_gb: float
    content_tokens: tuple[str, ...]
    alternate_urls: tuple[str, ...] = ()

    def record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "licence": self.licence,
            "source_url": self.source_url,
            "code_url": self.code_url,
            "role": self.role,
            "label_classes": list(self.label_classes),
            "approximate_size_gb": self.approximate_size_gb,
            "alternate_urls": list(self.alternate_urls),
        }


PRETRAINING_CORPORA: tuple[PublicCorpus, ...] = (
    PublicCorpus(
        name="TotalSegmentator CT dataset",
        version="2.0.1",
        licence="CC BY 4.0",
        source_url="https://doi.org/10.5281/zenodo.10047292",
        code_url="https://github.com/wasserth/TotalSegmentator",
        role="abdominal vascular labels for pretraining the segmentation backbone",
        label_classes=TOTALSEGMENTATOR_CLASSES_OF_INTEREST,
        approximate_size_gb=23.6,
        # The release archive of record 10047292 is this file; its name and size
        # were read from the Zenodo API, not from the landing page prose.
        content_tokens=("Totalsegmentator_dataset_v201", "segmentations", "anatomical"),
        # The DOI the manuscript prints redirects through a resolver that drops
        # the connection from some networks, so the record it resolves to is
        # recorded beside it and either may answer.
        alternate_urls=("https://zenodo.org/records/10047292", TOTALSEGMENTATOR_CONCEPT_DOI),
    ),
    PublicCorpus(
        name="AbdomenAtlas 1.0",
        version="1.0",
        licence="see the licence stated at the repository",
        source_url="https://github.com/MrGiovanni/AbdomenAtlas",
        code_url="https://github.com/MrGiovanni/AbdomenAtlas",
        role="abdominal CT volumes and organ labels for pretraining",
        label_classes=ABDOMENATLAS_CLASSES_OF_INTEREST,
        approximate_size_gb=328.0,
        content_tokens=("abdomenatlas",),
    ),
)


def corpus_by_name(name: str) -> PublicCorpus:
    for corpus in PRETRAINING_CORPORA:
        if corpus.name == name:
            return corpus
    raise KeyError(f"unknown corpus {name!r}")


def staging_plan(root: str | Path) -> dict[str, Path]:
    """Where each public corpus is expected once a site has fetched it."""
    base = Path(root)
    return {
        "totalsegmentator": base / "totalsegmentator_v201",
        "abdomenatlas": base / "abdomenatlas_1_0",
        "pretraining_manifest": base / "pretraining_manifest.json",
    }


def check_presence(root: str | Path) -> dict[str, bool]:
    """Report which corpora are actually on disk, without fetching anything."""
    plan = staging_plan(root)
    return {key: path.exists() for key, path in plan.items() if key != "pretraining_manifest"}


def label_index_map(corpus: PublicCorpus) -> dict[str, int]:
    """One-based label indices, in the order the corpus lists its classes."""
    return {name: index for index, name in enumerate(corpus.label_classes, start=1)}


def pretraining_manifest() -> dict[str, object]:
    """Machine-readable form of the pretraining provenance."""
    return {
        "corpora": [corpus.record() for corpus in PRETRAINING_CORPORA],
        "concept_doi_for_pinned_release": TOTALSEGMENTATOR_CONCEPT_DOI,
        "supplies_evaluation_data": False,
        "supplies_clinical_variables": False,
    }
