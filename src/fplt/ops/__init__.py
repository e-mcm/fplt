"""Season-rollover and ops helper commands (preflight, idmap suggest)."""

from fplt.ops.idmap import (
    MatchCandidate,
    SourceName,
    SourcePlayer,
    TargetPlayer,
    format_candidates_report,
    high_confidence_map_entries,
    normalize_name,
    suggest_matches,
    write_candidates_yaml,
)
from fplt.ops.preflight import (
    CheckResult,
    PreflightReport,
    run_preflight,
)

__all__ = [
    "CheckResult",
    "MatchCandidate",
    "PreflightReport",
    "SourceName",
    "SourcePlayer",
    "TargetPlayer",
    "format_candidates_report",
    "high_confidence_map_entries",
    "normalize_name",
    "run_preflight",
    "suggest_matches",
    "write_candidates_yaml",
]
