"""Classify recognized speech into commands used by the operator."""

from __future__ import annotations

import os
import re

from rapidfuzz import fuzz, process


_COMMANDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("call", ("openduck", "おーぷんだっく", "オープンダック")),
    ("seek", ("come on", "来て", "こっちきて")),
    ("bye", ("bye", "see you", "さよなら", "ばいばい", "またね")),
    ("stop", ("stop", "終わり", "止まれ", "そこまで")),
    ("hello", ("hello", "こんにちは", "こんちは", "うっす", "よっ")),
)
_COMMAND_BY_PHRASE: dict[str, str] = {
    phrase.casefold(): command
    for command, phrases in _COMMANDS
    for phrase in phrases
}
_PHRASES: tuple[str, ...] = tuple(_COMMAND_BY_PHRASE)
_MATCH_THRESHOLD = float(os.getenv("SEMANTIC_MATCH_THRESHOLD", "75"))


def _normalize(text: str) -> str:
    """Normalize case and punctuation while retaining Japanese characters."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def interpret(text: str) -> str | None:
    """Return the closest recognized command, or ``None`` for irrelevant speech."""
    normalized = _normalize(text)
    if not normalized:
        return None

    match = process.extractOne(
        normalized,
        _PHRASES,
        scorer=fuzz.partial_ratio,
        score_cutoff=_MATCH_THRESHOLD,
    )
    return _COMMAND_BY_PHRASE[match[0]] if match is not None else None