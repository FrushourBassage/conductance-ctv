"""The verification layer: claim mapping, artefact writers and the integrity manifest.

The writers are exercised against a temporary root so a test run never replaces
the shipped artefacts, and one test asserts that the shipped report is the
complete one rather than a partial runner's report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conductance_ctv.verification.artefacts import (
    CLAIM_MAP_NAME,
    MANIFEST_NAME,
    REPORT_NAME,
    SUMMARY_NAME,
    artefact_paths,
    build_manifest,
    verify_manifest,
    write_claim_map,
    write_dataset_urls,
    write_integrity_manifest,
    write_summary,
    write_verification_report,
)
from conductance_ctv.verification.checks import (
    BLOCKED,
    FAIL,
    NOT_RUN,
    PASS,
    PRESPECIFIED_CRITERIA,
    CheckResult,
    environment_record,
    report_lines,
    summarise,
)
from conductance_ctv.verification.claims import CLAIMS, DEVIATIONS, apply_status, claims_by_check


def test_claim_map_is_complete_and_pointed_at_real_files(repo_root: Path) -> None:
    assert len(CLAIMS) >= 25
    identifiers = [claim.identifier for claim in CLAIMS]
    assert len(set(identifiers)) == len(identifiers)
    for claim in CLAIMS:
        assert claim.paper_location
        assert claim.claim
        assert claim.method
        assert claim.check
        assert claim.files
        for relative in claim.files:
            assert (repo_root / relative).exists(), relative
    covered = {claim.check for claim in CLAIMS}
    grouped = claims_by_check()
    assert set(grouped) == covered


def test_deviations_are_documented() -> None:
    assert len(DEVIATIONS) >= 10
    for deviation in DEVIATIONS:
        assert deviation["paper_location"]
        assert deviation["departure"]
        assert deviation["reason"]
        assert deviation["impact"]


def test_apply_status_writes_check_outcomes_into_the_claims() -> None:
    updated = apply_status(
        CLAIMS,
        {"solver_rectangle_series": PASS, "manufactured_solution": NOT_RUN},
        {"solver_rectangle_series": {"rows": 3}},
    )
    by_identifier = {claim.identifier: claim for claim in updated}
    assert by_identifier["flow-problem"].status == PASS
    assert by_identifier["flow-problem"].evidence == {"rows": 3}
    unresolved = [claim for claim in updated if claim.status not in {PASS, NOT_RUN}]
    assert all(claim.status == NOT_RUN for claim in unresolved)


def test_prespecified_criteria_values() -> None:
    assert PRESPECIFIED_CRITERIA["external_macro_auroc_floor"] == 0.900
    assert PRESPECIFIED_CRITERIA["minimal_luminal_area_icc_floor"] == 0.900
    assert PRESPECIFIED_CRITERIA["management_relevant_change_floor"] == 0.100
    assert PRESPECIFIED_CRITERIA["reader_kappa_non_inferiority_margin"] == 0.050


def test_summarise_derives_the_overall_status() -> None:
    clean = [
        CheckResult("a", "paper", PASS, "ok", {}),
        CheckResult("b", "code", PASS, "ok", {}),
        CheckResult("c", "code", NOT_RUN, "not here", {}),
    ]
    summary = summarise(clean)
    assert summary["overall_status"] == "PARTIALLY_VERIFIED"
    assert summary["code_status"] == "VERIFIED"
    assert summary["outstanding"] == ["c"]
    failing = summarise([CheckResult("d", "code", FAIL, "broken", {})])
    assert failing["overall_status"] == "UNVERIFIED"
    assert failing["code_failures"] == ["d"]
    paper_only = summarise([CheckResult("e", "paper", FAIL, "mismatch", {})])
    assert paper_only["overall_status"] == "PARTIALLY_VERIFIED"
    assert paper_only["paper_failures"] == ["e"]
    complete = summarise([CheckResult("f", "code", PASS, "ok", {}), CheckResult("g", "paper", PASS, "ok", {})])
    assert complete["overall_status"] == "VERIFIED"
    blocked = summarise([CheckResult("h", "code", BLOCKED, "no docker", {})])
    assert blocked["overall_status"] == "PARTIALLY_VERIFIED"


def test_report_lines_render_every_check() -> None:
    results = [CheckResult("a", "paper", PASS, "ok", {}), CheckResult("b", "code", NOT_RUN, "absent", {})]
    lines = report_lines(results, summarise(results))
    joined = "\n".join(lines)
    assert "overall status" in joined
    assert "a" in joined and "b" in joined
    assert "NOT_RUN or BLOCKED" in joined


def test_environment_record_reports_the_versions() -> None:
    record = environment_record()
    assert record["python"]
    assert record["torch"]
    assert isinstance(record["cuda_available"], bool)


def test_artefact_writers_use_a_given_root(tmp_path: Path) -> None:
    write_claim_map({"claims": []}, tmp_path)
    write_verification_report({"overall_status": "PARTIALLY_VERIFIED"}, tmp_path)
    write_summary(["line one", "line two"], tmp_path)
    write_dataset_urls(
        [
            {
                "name": "corpus",
                "version": "1.0",
                "licence": "CC BY 4.0",
                "role": "pretraining",
                "url": "https://example.org/corpus",
                "checked": "HTTP 200",
            }
        ],
        tmp_path,
    )
    paths = artefact_paths(tmp_path)
    assert paths["claim_map"].name == CLAIM_MAP_NAME
    assert json.loads(paths["claim_map"].read_text(encoding="utf-8")) == {"claims": []}
    assert json.loads(paths["verification_report"].read_text(encoding="utf-8"))["overall_status"]
    assert paths["summary"].read_text(encoding="utf-8").endswith("\n")
    assert "url: https://example.org/corpus" in paths["dataset_urls"].read_text(encoding="utf-8")
    for name in (CLAIM_MAP_NAME, REPORT_NAME, SUMMARY_NAME):
        assert oct((tmp_path / name).stat().st_mode & 0o777) == "0o644"


def test_manifest_covers_the_tree_and_excludes_itself(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme\n", encoding="utf-8")
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "module.pyc").write_bytes(b"\x00")
    manifest = build_manifest(tmp_path)
    paths = {entry.path for entry in manifest.files}
    assert paths == {"README.md", "src/module.py"}
    assert manifest.digest
    second = build_manifest(tmp_path)
    assert second.digest == manifest.digest
    write_integrity_manifest(tmp_path)
    shipped = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert MANIFEST_NAME not in {entry["path"] for entry in shipped["files"]}
    check = verify_manifest(tmp_path)
    assert check["matches"]
    (tmp_path / "src" / "module.py").write_text("value = 2\n", encoding="utf-8")
    stale = verify_manifest(tmp_path)
    assert not stale["matches"]
    assert stale["changed"] == ["src/module.py"]


def test_manifest_detects_an_added_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    write_integrity_manifest(tmp_path)
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    check = verify_manifest(tmp_path)
    assert not check["matches"]
    assert check["added"] == ["b.txt"]


def test_shipped_report_is_the_complete_one(repo_root: Path) -> None:
    """A test run must never overwrite the shipped artefacts with a partial report."""
    report_path = repo_root / REPORT_NAME
    claim_path = repo_root / CLAIM_MAP_NAME
    if not report_path.exists() or not claim_path.exists():
        pytest.skip("the artefacts have not been written yet")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    claim_map = json.loads(claim_path.read_text(encoding="utf-8"))
    assert len(report["checks"]) >= 30
    assert report["summary"]["checks"] == len(report["checks"])
    assert len(claim_map["claims"]) >= 25
    assert len(claim_map["deviations"]) >= 10
    executed = {check["name"] for check in report["checks"] if check["status"] == PASS}
    assert "solver_rectangle_series" in executed
    assert "checkpoint_roundtrip" in executed
    assert "pipeline_tables" in executed
