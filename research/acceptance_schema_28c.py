"""C1: canonical JSON schema for the 28c GA acceptance record.

Why this module exists
----------------------
``ga_28c_30m_3y.py`` used to emit ``acceptance.portfolio_sharpe`` -- a *boolean*
under a key named after a *metric*. Two defects followed:

1. The name read like a value, not a gate, so downstream readers could not tell
   a passing criterion from a reported quantity.
2. The regime gate demotes that criterion to a diagnostic (the gate decides on
   excess Sharpe, not on absolute OOS Sharpe). Keeping the old key name while
   demoting it would let an existing reader keep treating a demoted diagnostic
   as a live gate. Silent demotion is worse than removal.

So the key is *renamed and relocated*, not merely moved:
    acceptance.portfolio_sharpe
      -> diagnostics.oos_portfolio_sharpe_exceeds_legacy_bar

Migration policy
----------------
- **Writers** emit the new schema only. There is no dual-write and no
  deprecated alias, because a kept alias is exactly what R6 warns about.
- **Readers** go through :func:`read_acceptance`, which understands both
  layouts and never guesses: an artifact written before this change is
  reported as ``schema_version=1`` and ``legacy=True``, and its verdict is
  fail-closed ``fail`` unless an explicit new-schema verdict is present.
- An artifact with no verdict at all is ``fail`` with a reason. Absence is
  never read as success.
"""
from __future__ import annotations

SCHEMA_VERSION = 2

#: The only three values ``acceptance.verdict`` may take.
PASS = "pass"
FAIL = "fail"
INSUFFICIENT = "insufficient_regime_coverage"
VALID_VERDICTS = (PASS, FAIL, INSUFFICIENT)

#: Top-level key for the per-fold table. Deliberately NOT ``history`` and NOT
#: ``plot_walk_forward``: dashboard/visualizer.py:75-83 reads those names from a
#: dict expecting ``step``/``sharpe``/``max_dd`` keys and silently returns an
#: empty figure on mismatch, which would hide the fold table rather than show it.
FOLD_TABLE_KEY = "regime_gate"

LEGACY_KEY = "portfolio_sharpe"
LEGACY_PATH = "acceptance.portfolio_sharpe"
NEW_KEY = "oos_portfolio_sharpe_exceeds_legacy_bar"
NEW_PATH = f"diagnostics.{NEW_KEY}"


def make_acceptance(verdict, verdict_reason, *, min_positive_coins,
                    oos_mdd, oos_solvent, oos_portfolio_sharpe,
                    legacy_bar, criteria):
    """Build the canonical acceptance record.

    ``verdict`` is validated, not trusted. An unrecognised value degrades to
    ``fail`` rather than propagating, so a future writer cannot accidentally
    invent a passing state.
    """
    if verdict not in VALID_VERDICTS:
        verdict_reason = (
            f"unrecognised verdict {verdict!r}; fail-closed to {FAIL}"
        )
        verdict = FAIL
    if not verdict_reason:
        verdict_reason = f"no reason recorded; fail-closed to {FAIL}"
        verdict = FAIL
    return {
        "schema_version": SCHEMA_VERSION,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "min_positive_coins": int(min_positive_coins),
        "mdd": bool(oos_mdd),
        "solvent": bool(oos_solvent),
        "criteria": criteria,
    }


def derive_live_adopted(acceptance):
    """``live_adopted`` is *derived*, never a literal.

    A hardcoded ``False`` is fail-closed by accident. This makes it fail-closed
    by construction: adoption requires an explicit, validated ``pass``.
    """
    return acceptance.get("verdict") == PASS


def read_acceptance(doc):
    """Read an acceptance record from either schema generation.

    Returns ``(acceptance, diagnostics)``. Never raises on a missing verdict and
    never returns a passing verdict it did not find in the document.
    """
    if not isinstance(doc, dict):
        return (make_acceptance(FAIL, "artifact is not a JSON object; fail-closed",
                                min_positive_coins=0, oos_mdd=False,
                                oos_solvent=False, oos_portfolio_sharpe=0.0,
                                legacy_bar=0.0, criteria={}), {})

    acc = doc.get("acceptance")
    diagnostics = doc.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}

    if not isinstance(acc, dict):
        return (make_acceptance(
            FAIL, "no acceptance record present; fail-closed",
            min_positive_coins=0, oos_mdd=False, oos_solvent=False,
            oos_portfolio_sharpe=0.0, legacy_bar=0.0, criteria={}), diagnostics)

    # v2: verdict present and well-formed.
    if "verdict" in acc:
        acc = dict(acc)
        acc.setdefault("schema_version", SCHEMA_VERSION)
        return acc, diagnostics

    # v1: the legacy layout. Report it as legacy and fail closed.
    acc = dict(acc)
    acc["schema_version"] = 1
    acc["legacy"] = True
    if LEGACY_KEY in acc and NEW_KEY not in diagnostics:
        diagnostics[NEW_KEY] = acc.pop(LEGACY_KEY)
        diagnostics[NEW_KEY + "_source_path"] = LEGACY_PATH
    acc.setdefault("verdict", FAIL)
    acc.setdefault("verdict_reason",
                   f"artifact predates schema v{SCHEMA_VERSION}; no gate verdict "
                   f"was ever computed for it, so it is fail-closed. The demoted "
                   f"criterion is now at {NEW_PATH}.")
    return acc, diagnostics
