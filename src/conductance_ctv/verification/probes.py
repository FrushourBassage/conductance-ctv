"""Live reachability probes for the sources the release records.

Both the verification report and the dataset link list have to say whether a
recorded source was reachable and whether the page that answered is the
resource it claims to be, and they have to give the same answer. The probe
therefore lives here once: a successful status is not enough, because a DOI that
resolves to an unrelated landing page would otherwise be listed as verified.

Ref: the Data availability statement; the dataset link list.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass

PROBE_TIMEOUT_SECONDS = 30
PROBE_ATTEMPTS = 3
PROBE_BACKOFF_SECONDS = 2.0
_PROBE_BYTES = 65536


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of one probe, with the reason a failure failed."""

    url: str
    status: int | None
    matched: bool
    note: str

    @property
    def verified(self) -> bool:
        return self.status == 200 and self.matched


def probe_source(url: str, tokens: tuple[str, ...] = ()) -> ProbeResult:
    """Fetch a source and test its body against the tokens that identify it.

    The transfer to the public corpora is throttled from some networks, so a
    probe is retried before it is reported as unreachable. A body that does not
    carry any of the expected tokens is a content mismatch, not a success.
    """
    last_note = "not attempted"
    for attempt in range(PROBE_ATTEMPTS):
        request = urllib.request.Request(url, headers={"User-Agent": "conductance-ctv/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT_SECONDS) as response:
                status = int(response.status)
                body = response.read(_PROBE_BYTES).decode("utf-8", errors="ignore").lower()
        except urllib.error.HTTPError as error:
            last_note = f"HTTP {error.code}"
        except Exception as error:  # noqa: BLE001  (an unreachable host is a finding, not a crash)
            last_note = type(error).__name__
        else:
            matched = any(token in body for token in tokens) if tokens else True
            if status != 200:
                last_note = f"HTTP {status}"
            elif not matched:
                last_note = "content did not match the expected resource"
            else:
                return ProbeResult(url, status, True, f"HTTP {status}, page content matched")
        if attempt + 1 < PROBE_ATTEMPTS:
            time.sleep(PROBE_BACKOFF_SECONDS * (attempt + 1))
    return ProbeResult(url, None, False, last_note)


def probe_any(urls: tuple[str, ...], tokens: tuple[str, ...] = ()) -> ProbeResult:
    """Probe the recorded URLs in order and return the first that verifies.

    A record is reachable through more than one address: the DOI a manuscript
    prints, the repository record it resolves to, and the project's own page. The
    source counts as verified when one of them answers with the expected content,
    and the result names the address that answered.
    """
    notes: list[str] = []
    for url in urls:
        result = probe_source(url, tokens)
        if result.verified:
            return result
        notes.append(f"{url}: {result.note}")
    return ProbeResult(urls[0], None, False, "; ".join(notes))
