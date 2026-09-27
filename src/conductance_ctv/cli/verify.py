"""Verification entry point: run every check, then write the release artefacts.

The order is fixed. The claim map, the verification report and the plain-text
summary are written first, so that the integrity manifest digests a tree that
already contains them; the manifest is written last and excludes only itself. The
run finishes by re-hashing the live tree and diffing it against the shipped
manifest, so a stale digest cannot survive to the report.

Usage:
    python -m conductance_ctv.cli.verify
    python -m conductance_ctv.cli.verify --root . --skip-pipeline
    python -m conductance_ctv.cli.verify --probe-live

Ref: the verification pass described in the project context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from conductance_ctv.data.manifest import PRETRAINING_CORPORA
from conductance_ctv.utils.config import repo_root
from conductance_ctv.utils.logging_setup import configure_logging, get_logger
from conductance_ctv.verification.artefacts import (
    artefact_paths,
    verify_manifest,
    write_claim_map,
    write_dataset_urls,
    write_integrity_manifest,
    write_summary,
    write_verification_report,
)
from conductance_ctv.verification.checks import environment_record, report_lines, run_all_checks, summarise
from conductance_ctv.verification.claims import CLAIMS, DEVIATIONS, apply_status

LOGGER = get_logger("cli.verify")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conductance-ctv-verify",
        description="Execute the verification pass and write the release artefacts.",
    )
    parser.add_argument(
        "--root", type=Path, default=None, help="repository root; defaults to the directory holding configs/"
    )
    parser.add_argument("--skip-pipeline", action="store_true", help="skip the checks that execute the full pipeline")
    parser.add_argument(
        "--probe-live",
        action="store_true",
        help="re-probe the recorded dataset sources and rewrite the link list, instead of reading the shipped one",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def probe_dataset_urls() -> list[dict[str, str]]:
    """Check each declared source live and keep only the ones whose content matched.

    A link is listed only when the request returned a success status and the body
    contained a token that identifies the resource; a redirect to an unrelated
    page therefore drops out rather than being listed as verified.
    """
    from conductance_ctv.verification.probes import probe_any

    entries: list[dict[str, str]] = []
    for corpus in PRETRAINING_CORPORA:
        probe = probe_any((corpus.source_url, *corpus.alternate_urls), corpus.content_tokens)
        if not probe.verified:
            LOGGER.warning("probe for %s did not verify: %s", corpus.name, probe.note)
            continue
        entries.append(
            {
                "name": corpus.name,
                "version": corpus.version,
                "licence": corpus.licence,
                "role": corpus.role,
                "url": probe.url,
                "recorded_doi": corpus.source_url,
                "checked": probe.note,
            }
        )
    return entries


def run_verification(root: Path | None = None, skip_pipeline: bool = False, probe_live: bool = False) -> dict[str, Any]:
    """Execute the checks and write every artefact in the required order.

    The default pass is offline and deterministic: it reads the shipped link list
    and never contacts a source, so two runs on the same tree produce the same
    artefacts. ``probe_live`` refreshes the link list first, which is how the
    recorded sources are re-verified, and that run's evidence carries the
    outcome.
    """
    base = repo_root() if root is None else Path(root)
    LOGGER.info("verification root: %s", base)
    if probe_live:
        # The link list is an input to the checks, not a product of them, so it is
        # rewritten before they run; otherwise the check that reads it would be
        # scoring the previous run's file.
        write_dataset_urls(probe_dataset_urls(), base)
    results = run_all_checks(base, include_pipeline=not skip_pipeline, probe_live=probe_live)
    summary = summarise(results)
    statuses = {result.name: result.status for result in results}
    evidence = {result.name: result.evidence for result in results}
    claims = apply_status(CLAIMS, statuses, evidence)

    claim_map = {
        "paper": "Imaging Features and Clinical Significance of Iliac Vein Lesions Based on 3D-CTV Classification",
        "release": "conductance-ctv",
        "overall_status": summary["overall_status"],
        "code_status": summary["code_status"],
        "paper_status": summary["paper_status"],
        "claims": [claim.record() for claim in claims],
        "deviations": list(DEVIATIONS),
        "summary": {
            "claims": len(claims),
            "statuses": statuses,
            "unresolved": summary["outstanding"],
            "failures": summary["code_failures"] + summary["paper_failures"],
        },
    }
    report = {
        "release": "conductance-ctv",
        "overall_status": summary["overall_status"],
        "code_status": summary["code_status"],
        "paper_status": summary["paper_status"],
        "environment": environment_record(),
        "summary": summary,
        "checks": [result.record() for result in results],
        "first_pass": {
            "what": "the manuscript claim to code mapping",
            "artifact": "claim_to_code.json",
            "claims": len(claims),
            "verified": sum(1 for claim in claims if claim.status == "PASS"),
        },
        "second_pass": {
            "what": (
                "the execution pass over data reading, forward, loss, backward, parameter update and checkpoint save"
                " and load"
            ),
            "checks": [result.name for result in results if result.family == "code"],
            "executed": sum(1 for result in results if result.family == "code" and result.status == "PASS"),
        },
        "limitations": [
            "Cohort-level point values are NOT_RUN because the clinical cohort is not shareable.",
            "The headline training run is NOT_RUN at scale because no accelerator is present.",
            "Pretraining is NOT_RUN because the public corpora were not staged.",
            "The container build is NOT_RUN because no container runtime is installed.",
            "The dataset sources are read from the shipped link list; pass --probe-live to re-verify them.",
        ],
    }

    write_claim_map(claim_map, base)
    write_verification_report(report, base)
    write_summary(report_lines(results, summary), base)
    write_integrity_manifest(base)
    manifest_check = verify_manifest(base)

    paths = artefact_paths(base)
    returned: dict[str, Any] = {
        "overall_status": summary["overall_status"],
        "by_status": summary["by_status"],
        "artefacts": {key: str(path.relative_to(base)) for key, path in paths.items()},
        "manifest": manifest_check,
    }
    if not summary["code_failures"] and not summary["paper_failures"]:
        LOGGER.info("all executed checks passed; outstanding items are the ones this environment cannot produce")
    else:
        LOGGER.warning("failures: %s", summary["code_failures"] + summary["paper_failures"])
    return returned


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    arguments = build_parser().parse_args(argv)
    result = run_verification(arguments.root, arguments.skip_pipeline, arguments.probe_live)
    if not arguments.quiet:
        print(json.dumps(result, indent=2, default=str))
    return 0 if result["manifest"]["matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
