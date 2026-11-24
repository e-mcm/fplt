"""Resolved component-param defaults match pre-typing ``.get`` defaults."""

from __future__ import annotations

import pytest

from fplt.models.component_params import (
    LEGACY_GET_DEFAULTS,
    PARAMS_BY_KIND,
    parse_component_params,
    resolve_component_params,
)


@pytest.mark.parametrize("kind", sorted(PARAMS_BY_KIND))
def test_resolved_defaults_match_legacy_get(kind: str) -> None:
    """Empty params resolve to the same values today's ``.get`` used."""
    assert resolve_component_params(kind, {}) == LEGACY_GET_DEFAULTS[kind]


@pytest.mark.parametrize("kind", sorted(PARAMS_BY_KIND))
def test_unknown_param_is_forbidden(kind: str) -> None:
    """Typos fail instead of silently using the default."""
    model = PARAMS_BY_KIND[kind]
    with pytest.raises(ValueError, match="extra"):
        parse_component_params(model, {"not_a_real_param": 1})


def test_dixon_coles_closing_proxy_follows_1x2_strength() -> None:
    """Absent ``use_closing_proxy`` tracks ``market_1x2_strength > 0``."""
    off = resolve_component_params("dixon_coles", {})
    assert off["use_closing_proxy"] is False
    on = resolve_component_params("dixon_coles", {"market_1x2_strength": 0.5})
    assert on["use_closing_proxy"] is True
    forced = resolve_component_params(
        "dixon_coles",
        {"market_1x2_strength": 0.5, "use_closing_proxy": False},
    )
    assert forced["use_closing_proxy"] is False
