"""Fuzzy id-map suggestions for Understat / API-Football → FPL ``player_key``.

Candidates are scored and written for human review. High-confidence rows are
formatted for a PR into ``configs/*/id_maps.yaml``; nothing is auto-merged.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal, cast

import yaml

SourceName = Literal["understat", "api_football"]

# Common FPL / media nicknames → normalized tokens that should match the
# canonical full name (also normalized). Keys and values are already
# lowercase ASCII after ``normalize_name``.
_NICKNAMES: dict[str, frozenset[str]] = {
    "son": frozenset({"heung min son", "son heung min"}),
    "jota": frozenset({"diogo jota"}),
    "bruno": frozenset({"bruno fernandes"}),
    "richarlison": frozenset({"richarlison de andrade"}),
    "rodri": frozenset({"rodrigo hernandez", "rodrigo"}),
    "pedri": frozenset({"pedro gonzalez"}),
}

_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_WS = re.compile(r"\s+")

DEFAULT_MIN_SCORE = 0.75
DEFAULT_HIGH_CONFIDENCE = 0.92


@dataclass(frozen=True)
class SourcePlayer:
    """One unmapped upstream player awaiting an FPL ``player_key``."""

    source_id: int
    name: str
    team_key: int | None = None
    position: str | None = None
    dob: date | None = None


@dataclass(frozen=True)
class TargetPlayer:
    """One FPL player (element ``code`` = ``player_key``) to match against."""

    player_key: int
    web_name: str
    full_name: str
    team_key: int
    position: str
    dob: date | None = None


@dataclass(frozen=True)
class MatchCandidate:
    """One scored suggestion from a source id to an FPL ``player_key``."""

    source_id: int
    source_name: str
    player_key: int
    target_name: str
    score: float
    high_confidence: bool
    reason: str


def normalize_name(value: str) -> str:
    """Fold accents and punctuation so ``José`` matches ``Jose``.

    Args:
        value: Raw display name from any source.

    Returns:
        Lowercase ASCII alphanumerics with collapsed whitespace.
    """
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    cleaned = _NON_ALNUM.sub(" ", ascii_only.lower())
    return _WS.sub(" ", cleaned).strip()


def name_similarity(left: str, right: str) -> float:
    """Return a 0–1 similarity after normalization and nickname expansion.

    Args:
        left: Source-side name.
        right: FPL ``web_name`` or full name.

    Returns:
        Best ``SequenceMatcher`` ratio across raw and nickname-expanded forms.
    """
    a = normalize_name(left)
    b = normalize_name(right)
    if not a or not b:
        return 0.0
    best = SequenceMatcher(None, a, b).ratio()
    for form_a in _expanded_forms(a):
        for form_b in _expanded_forms(b):
            best = max(best, SequenceMatcher(None, form_a, form_b).ratio())
            if _token_subset_match(form_a, form_b):
                best = max(best, 0.9)
    return best


def suggest_matches(
    unmapped: list[SourcePlayer],
    targets: list[TargetPlayer],
    *,
    min_score: float = DEFAULT_MIN_SCORE,
    high_confidence: float = DEFAULT_HIGH_CONFIDENCE,
) -> list[MatchCandidate]:
    """Fuzzy-match unmapped source players onto FPL ``player_key``s.

    Same-name players at different clubs are disambiguated by ``team_key``:
    a name-only collision across clubs never reaches high confidence without
    a team match. Scores below ``min_score`` are never suggested.

    Args:
        unmapped: Upstream players missing from the id map.
        targets: FPL players for the season (``player_key`` grain).
        min_score: Floor for emitting a candidate (exclusive below).
        high_confidence: Score at or above this is flagged for a PR.

    Returns:
        Candidates sorted by score descending, then ``source_id``.
    """
    if min_score < 0.0 or min_score > 1.0:
        raise ValueError("min_score must be in [0, 1]")
    if high_confidence < min_score or high_confidence > 1.0:
        raise ValueError("high_confidence must be in [min_score, 1]")

    by_name: dict[str, list[TargetPlayer]] = {}
    for target in targets:
        for label in (target.web_name, target.full_name):
            key = normalize_name(label)
            if key:
                by_name.setdefault(key, []).append(target)

    out: list[MatchCandidate] = []
    for source in unmapped:
        best = _best_match(
            source,
            targets,
            by_name=by_name,
            min_score=min_score,
            high_confidence=high_confidence,
        )
        if best is not None:
            out.append(best)
    out.sort(key=lambda row: (-row.score, row.source_id))
    return out


def high_confidence_map_entries(
    candidates: list[MatchCandidate],
) -> dict[int, int]:
    """Return ``source_id → player_key`` for high-confidence candidates only.

    Args:
        candidates: Output of :func:`suggest_matches`.

    Returns:
        Mapping fragment suitable for merging into ``id_maps.yaml`` ``players``.
    """
    return {row.source_id: row.player_key for row in candidates if row.high_confidence}


def format_candidates_report(
    candidates: list[MatchCandidate],
    *,
    source: SourceName,
) -> str:
    """Format a human-readable suggestion report for the CLI.

    Args:
        candidates: Scored suggestions.
        source: Upstream label for the header.

    Returns:
        Multi-line text (empty candidates → a short notice).
    """
    if not candidates:
        return f"idmap suggest ({source}): no candidates above threshold"
    lines = [
        f"idmap suggest ({source}): {len(candidates)} candidate(s)",
        "  (high-confidence → PR; nothing is auto-merged)",
    ]
    for row in candidates:
        flag = "HIGH" if row.high_confidence else "review"
        lines.append(
            f"  [{flag}] {row.source_id} ({row.source_name}) → "
            f"{row.player_key} ({row.target_name}) "
            f"score={row.score:.3f} {row.reason}"
        )
    high = high_confidence_map_entries(candidates)
    if high:
        lines.append("high-confidence players map fragment:")
        for source_id, player_key in sorted(high.items()):
            lines.append(f"  {source_id}: {player_key}")
    return "\n".join(lines)


def write_candidates_yaml(
    path: Path,
    candidates: list[MatchCandidate],
    *,
    source: SourceName,
) -> None:
    """Write scored candidates (and a high-confidence map fragment) to YAML.

    Args:
        path: Destination file (created/overwritten).
        candidates: Scored suggestions.
        source: Upstream label recorded in the file.
    """
    payload = {
        "source": source,
        "note": (
            "Review before merging into configs/"
            f"{source}/id_maps.yaml; nothing is auto-merged."
        ),
        "candidates": [
            {
                "source_id": row.source_id,
                "source_name": row.source_name,
                "player_key": row.player_key,
                "target_name": row.target_name,
                "score": round(row.score, 4),
                "high_confidence": row.high_confidence,
                "reason": row.reason,
            }
            for row in candidates
        ],
        "high_confidence_players": {
            str(source_id): player_key
            for source_id, player_key in sorted(
                high_confidence_map_entries(candidates).items()
            )
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _expanded_forms(normalized: str) -> set[str]:
    forms = {normalized}
    for nick, expansions in _NICKNAMES.items():
        if normalized == nick or nick in normalized.split():
            forms.update(expansions)
            forms.add(nick)
        if normalized in expansions:
            forms.add(nick)
            forms.update(expansions)
    return forms


def _token_subset_match(left: str, right: str) -> bool:
    a = set(left.split())
    b = set(right.split())
    if not a or not b:
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return shorter <= longer and len(shorter) >= 1


def _best_match(
    source: SourcePlayer,
    targets: list[TargetPlayer],
    *,
    by_name: dict[str, list[TargetPlayer]],
    min_score: float,
    high_confidence: float,
) -> MatchCandidate | None:
    ranked: list[tuple[float, TargetPlayer, str]] = []
    for target in targets:
        score, reason = _score_pair(source, target)
        if score >= min_score:
            ranked.append((score, target, reason))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1].player_key))
    score, target, reason = ranked[0]

    # Same display name at two clubs: refuse high confidence unless team matches.
    norm = normalize_name(source.name)
    name_hits = by_name.get(norm, [])
    distinct_teams = {row.team_key for row in name_hits}
    if len(distinct_teams) > 1 and (
        source.team_key is None or source.team_key != target.team_key
    ):
        if score >= high_confidence:
            score = min(score, high_confidence - 0.01)
            reason = f"{reason}; same-name multi-club (needs team)"

    is_high = score >= high_confidence and (
        source.team_key is None
        or source.team_key == target.team_key
        or len(distinct_teams) <= 1
    )
    if score < min_score:
        return None
    return MatchCandidate(
        source_id=source.source_id,
        source_name=source.name,
        player_key=target.player_key,
        target_name=target.full_name or target.web_name,
        score=score,
        high_confidence=is_high,
        reason=reason,
    )


def _score_pair(source: SourcePlayer, target: TargetPlayer) -> tuple[float, str]:
    name_score = max(
        name_similarity(source.name, target.web_name),
        name_similarity(source.name, target.full_name),
    )
    parts = [f"name={name_score:.3f}"]
    score = name_score
    if source.team_key is not None and source.team_key == target.team_key:
        score = min(1.0, score + 0.05)
        parts.append("team")
    elif source.team_key is not None and target.team_key != source.team_key:
        # Exact name matches still need a club preference (same-name multi-club).
        score = max(0.0, score - 0.15)
        parts.append("team_mismatch")
    if (
        source.position is not None
        and source.position == target.position
        and source.position != ""
    ):
        score = min(1.0, score + 0.03)
        parts.append("pos")
    if source.dob is not None and target.dob is not None:
        if source.dob == target.dob:
            score = min(1.0, score + 0.05)
            parts.append("dob")
        else:
            score = max(0.0, score - 0.2)
            parts.append("dob_mismatch")
    return score, "+".join(parts)


def collect_understat_unmapped(
    raw_root: Path,
    season: str,
    *,
    player_keys: dict[int, int],
    team_keys: dict[int, int],
) -> list[SourcePlayer]:
    """Scan raw Understat matches for unmapped ``player_id``s with names.

    Args:
        raw_root: ``$FPLT_DATA_DIR/raw``.
        season: Season label.
        player_keys: Existing Understat → FPL player map.
        team_keys: Understat → FPL team map.

    Returns:
        Deduplicated unmapped source players (latest name wins).
    """
    from fplt.data.understat import iter_raw_matches, read_json_gz

    found: dict[int, SourcePlayer] = {}
    for path, _season, _match_id in iter_raw_matches(raw_root, [season]):
        payload = cast(dict[str, object], read_json_gz(path))
        for source_id, name, team_id in _understat_roster_players(payload):
            if source_id in player_keys:
                continue
            team_key = team_keys.get(team_id)
            found[source_id] = SourcePlayer(
                source_id=source_id,
                name=name,
                team_key=team_key,
            )
    return [found[key] for key in sorted(found)]


def collect_api_football_unmapped(
    raw_root: Path,
    season: str,
    *,
    player_keys: dict[int, int],
    team_keys: dict[int, int],
) -> list[SourcePlayer]:
    """Scan raw API-Football injuries for unmapped player ids with names.

    Args:
        raw_root: ``$FPLT_DATA_DIR/raw``.
        season: Season label.
        player_keys: Existing API → FPL player map.
        team_keys: API → FPL team map.

    Returns:
        Deduplicated unmapped source players.
    """
    from fplt.data.api_football import iter_raw_injuries, read_json_gz

    found: dict[int, SourcePlayer] = {}
    for path, _season in iter_raw_injuries(raw_root, [season]):
        payload = cast(dict[str, object], read_json_gz(path))
        response = payload.get("response")
        if not isinstance(response, list):
            continue
        for item in cast(list[object], response):
            if not isinstance(item, dict):
                continue
            row = cast(dict[str, object], item)
            player = row.get("player")
            team = row.get("team")
            if not isinstance(player, dict):
                continue
            player_obj = cast(dict[str, object], player)
            if "id" not in player_obj:
                continue
            source_id = int(cast(Any, player_obj["id"]))
            if source_id in player_keys:
                continue
            name = str(player_obj.get("name") or f"id:{source_id}")
            team_key: int | None = None
            if isinstance(team, dict) and "id" in team:
                api_team_id = int(cast(Any, cast(dict[str, object], team)["id"]))
                team_key = team_keys.get(api_team_id)
            found[source_id] = SourcePlayer(
                source_id=source_id,
                name=name,
                team_key=team_key,
            )
    return [found[key] for key in sorted(found)]


def _understat_roster_players(
    payload: dict[str, object],
) -> list[tuple[int, str, int]]:
    """Extract ``(player_id, name, team_id)`` from a raw Understat match."""
    rows: list[tuple[int, str, int]] = []
    rosters_raw = payload.get("rosters")
    if isinstance(rosters_raw, dict):
        rosters = cast(dict[str, object], rosters_raw)
        for side in ("h", "a"):
            side_raw: object = rosters.get(side)
            entries: object = {} if side_raw is None else side_raw
            values: list[object]
            if isinstance(entries, dict):
                values = list(cast(dict[object, object], entries).values())
            elif isinstance(entries, list):
                values = list(cast(list[object], entries))
            else:
                continue
            for item in values:
                player = cast(dict[str, Any], item)
                if "player_id" not in player:
                    continue
                source_id = int(player["player_id"])
                name = str(player.get("player") or f"id:{source_id}")
                team_id = int(player.get("team_id") or 0)
                rows.append((source_id, name, team_id))
        return rows
    # Already-normalised match shape used in some fixtures/tests.
    for side in ("h", "a"):
        entries = payload.get(side)
        if not isinstance(entries, list):
            continue
        team_field = "team_h" if side == "h" else "team_a"
        team_id = int(cast(Any, payload.get(team_field) or 0))
        for item in cast(list[object], entries):
            player = cast(dict[str, Any], item)
            source_id = int(player["id"])
            name = str(player.get("player") or player.get("name") or f"id:{source_id}")
            rows.append((source_id, name, team_id))
    return rows
