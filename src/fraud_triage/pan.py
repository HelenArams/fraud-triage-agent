"""Luhn check and PAN scrubbing.

Anything that looks like a full card number (13-19 digits, optionally split by
spaces or dashes, passing Luhn) is replaced with a 6+4 mask. Used on the way in
(loader) and on the way out (guardrails over model text).
"""

from __future__ import annotations

import re

# 13-19 digits with optional single space/dash separators. The lookarounds stop
# us from matching the middle of a longer number.
_CANDIDATE = re.compile(r"(?<![\d])(?:\d[ -]?){12,18}\d(?![\d])")


def luhn_ok(digits: str) -> bool:
    if not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def mask(digits: str) -> str:
    return digits[:6] + "*" * (len(digits) - 10) + digits[-4:]


def scrub(text: str) -> tuple[str, int]:
    """Return (scrubbed text, number of PANs masked)."""
    count = 0

    def _sub(m: re.Match[str]) -> str:
        nonlocal count
        digits = re.sub(r"[ -]", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn_ok(digits):
            count += 1
            return mask(digits)
        return m.group(0)

    return _CANDIDATE.sub(_sub, text), count


def contains_pan(text: str) -> bool:
    return scrub(text)[1] > 0
