"""Plain-text rendering of a table, a report section and a whole run.

Reports are written as text with a fixed column layout so they read the same in a
terminal, in a log and in a review file. Nothing here writes markdown, because
the release ships one README and no other markdown document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from conductance_ctv.evaluation.tables import TableResult
from conductance_ctv.utils.io import write_text

MISSING = "NOT_RUN"


def plain_value(value: Any) -> str:
    """A value as it will be printed, with no truncation and no padding."""
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return MISSING
        return f"{float(value):.4f}"
    return str(value)


def column_widths(columns: tuple[str, ...], rows: list[dict[str, Any]], floor: int = 8) -> tuple[int, ...]:
    """Per-column widths taken from the data, so a long value is never cut."""
    widths: list[int] = []
    for name in columns:
        widest = max((len(plain_value(row.get(name, MISSING))) for row in rows), default=0)
        widths.append(max(floor, len(name), widest))
    return tuple(widths)


def format_value(value: Any, width: int = 12) -> str:
    return f"{plain_value(value):>{width}s}"


def render_rows(columns: tuple[str, ...], rows: list[dict[str, Any]]) -> list[str]:
    widths = column_widths(columns, rows)
    header = "".join(name.rjust(width) for name, width in zip(columns, widths, strict=True))
    lines = ["-" * len(header), header, "-" * len(header)]
    for row in rows:
        lines.append(
            "".join(format_value(row.get(name, MISSING), width) for name, width in zip(columns, widths, strict=True))
        )
    lines.append("-" * len(header))
    return lines


def render_table(result: TableResult, width: int = 12) -> str:
    """Render one table with a header rule and its notes underneath."""
    del width
    lines: list[str] = []
    lines.append(f"Table {result.number}. {result.title}")
    lines.append(f"status: {result.status}")
    if result.units:
        units = ", ".join(f"{key}={value}" for key, value in sorted(result.units.items()))
        lines.append(f"units: {units}")
    lines.extend(render_rows(result.columns, result.rows))
    for note in result.notes:
        lines.append(f"note: {note}")
    lines.append("")
    return "\n".join(lines)


@dataclass
class ReportSection:
    """A titled block of text, a table, or a key-value summary."""

    title: str
    body: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)
    columns: tuple[str, ...] = ()

    def render(self, width: int = 12) -> str:
        del width
        lines = [f"## {self.title}"]
        if self.body:
            lines.extend(self.body.rstrip("\n").split("\n"))
        if self.rows and self.columns:
            lines.extend(render_rows(self.columns, self.rows))
        lines.append("")
        return "\n".join(lines)


@dataclass
class Report:
    """A complete run report: a title, sections and tables."""

    title: str
    sections: list[ReportSection] = field(default_factory=list)
    tables: list[TableResult] = field(default_factory=list)

    def body(self) -> str:
        lines = [self.title, "=" * len(self.title), ""]
        for section in self.sections:
            lines.append(section.render().rstrip("\n"))
            lines.append("")
        for table in self.tables:
            lines.append(render_table(table).rstrip("\n"))
            lines.append("")
        return "\n".join(lines).rstrip("\n") + "\n"

    def write(self, path: str | Path) -> Path:
        return write_text(path, self.body())


def summary_section(title: str, payload: dict[str, Any]) -> ReportSection:
    """A key-value section from a flat mapping."""
    rows = [{"key": key, "value": value} for key, value in payload.items()]
    return ReportSection(title=title, rows=rows, columns=("key", "value"))


def matrix_section(
    title: str, matrix: list[list[int]], labels: tuple[str, ...] = ("I", "II", "III", "IV")
) -> ReportSection:
    """A confusion matrix rendered as one row per reference type."""
    rows = [
        {
            "reference": labels[row],
            **{f"pred_{labels[column]}": int(matrix[row][column]) for column in range(len(labels))},
        }
        for row in range(len(matrix))
    ]
    columns = ("reference", *[f"pred_{label}" for label in labels])
    return ReportSection(title=title, rows=rows, columns=columns)


def comparison_section(title: str, rows: list[dict[str, Any]]) -> ReportSection:
    columns = ("model", "metric", "estimate", "ci_low", "ci_high", "unit")
    return ReportSection(title=title, rows=rows, columns=columns)


def intervals_to_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten metric-interval records into display rows."""
    rows: list[dict[str, Any]] = []
    for entry in entries:
        rows.append(
            {
                "model": entry.get("model", ""),
                "metric": entry.get("metric", ""),
                "estimate": entry.get("estimate"),
                "ci_low": entry.get("ci_low"),
                "ci_high": entry.get("ci_high"),
                "unit": entry.get("unit", "patient"),
            }
        )
    return rows
