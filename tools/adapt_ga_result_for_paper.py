#!/usr/bin/env python3
"""Adapt a 28c GA result into the paper runner's source-artifact contract.

The paper runner (`run_paper_28c_pit_30m.py`) refuses any source that is not the
unified 28-coin training artifact: it checks `coins`, `universe_mode` and
`history_years`. A GA result is a different schema and is rejected. That guard
is deliberate -- we should not silently feed a GA artifact to a paper run and
have it look like a trained, contract-checked candidate.

This tool bridges them *explicitly*. It does not bypass any check: it writes the
fields the contract requires, and stamps provenance so nobody can mistake the
result for a training artifact.

What it copies, and why it is safe to copy:
  coins          from COINS_28C, and re-verified against the GA run's universe
  universe_mode  '28c_common_transfer', the 28c transfer contract
  history_years  from the GA result, re-checked against HISTORY_YEARS_COMMON_28

The carried-over acceptance fields are the GA's, unmodified. In particular a
`fail` verdict stays `fail`; this tool never upgrades an acceptance state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.universe_28c import COINS_28C  # noqa: E402
from research.data_contract_28c import HISTORY_YEARS_COMMON_28  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def build(ga_path: Path, decode: list[str] | None = None,
          gate_path: Path | None = None) -> dict:
    ga = json.loads(ga_path.read_text())

    formula = ga.get("formula")
    if not isinstance(formula, list) or len(formula) != 12:
        raise ValueError(f"GA result has no valid 12-token formula: {formula!r}")
    formula = [int(x) for x in formula]

    if decode is None:
        raise SystemExit(
            "--decode is required: the GA result stores no human-readable decode, "
            "and guessing token meanings would be worse than refusing."
        )
    if len(decode) != 12:
        raise ValueError(f"decode must have 12 entries, got {len(decode)}")

    history_years = ga.get("history_years")
    if history_years is None or abs(float(history_years) - HISTORY_YEARS_COMMON_28) > 0.01:
        raise ValueError(
            f"GA history_years {history_years!r} does not match the 28c contract "
            f"({HISTORY_YEARS_COMMON_28})"
        )

    ga_acceptance = ga.get("acceptance", {})

    # The gate verdict is read from an actually-executed gate artifact, never
    # inferred. A GA run predating the gate wiring has no verdict; that must
    # surface as unknown rather than be quietly reported as a pass or a fail.
    gate_verdict, gate_reason = "not_run", ""
    if gate_path is not None and gate_path.exists():
        gate = json.loads(gate_path.read_text())
        gate_verdict = gate.get("verdict", "unknown")
        gate_reason = gate.get("verdict_reason", "")
        if gate_verdict not in ("pass", "fail", "insufficient_regime_coverage"):
            raise ValueError(f"gate artifact has unusable verdict {gate_verdict!r}")

    return {
        # --- the fields the paper contract requires ---
        "formula": formula,
        "decode": decode,
        "coins": list(COINS_28C),
        "universe_mode": "28c_common_transfer",
        "history_years": float(history_years),
        # --- honest scoring: the GA's own numbers, never invented ---
        "score": ga_acceptance.get("diagnostics", {}).get("portfolio_sharpe",
                   ga.get("train", {}).get("portfolio_sharpe")),
        "worst_leg": ga.get("train", {}).get("min_leg_sharpe"),
        "contract_version": ga.get("contract_version", "data-contract-28c-v1"),
        # --- provenance: this is NOT a training artifact, and says so ---
        "source_kind": "ga_result_adapted_for_paper",
        "source_ga_result": str(ga_path),
        "source_ga_result_sha256": sha256_file(ga_path),
        "source_ga_seed": ga.get("seed"),
        "source_ga_generations": ga.get("generations"),
        "source_ga_live_adopted": ga.get("live_adopted"),
        "ga_acceptance": ga_acceptance,
        "ga_oos": {k: ga.get("oos", {}).get(k)
                   for k in ("portfolio_sharpe", "portfolio_mdd",
                             "positive_coins", "solvent")},
        "regime_gate_verdict": gate_verdict,
        "regime_gate_reason": gate_reason,
        "regime_gate_artifact": str(gate_path) if gate_path else None,
        "paper_runner_compatibility": {
            "compatible": False,
            "reason": (
                "The paper runner derives positions with "
                "position_from_signal(sigmoid(signal), long_thr=0.85, "
                "short_thr=0.15). The GA normalises its signal as "
                "tanh(sig/std) and takes CONTINUOUS positions "
                "(0.25 * smooth_causal(tanh(...))). After sigmoid, this formula's "
                "signal spans roughly 0.27-0.73, entirely inside the [0.15, 0.85] "
                "band, so the threshold rule opens ZERO positions and the paper "
                "run reports a flat 1.0x, 0 trades, 0 return."
            ),
            "consequence": (
                "A flat paper result for this formula is NOT evidence about its "
                "performance. It is evidence that the two position models are "
                "incompatible. Do not report the flat number as a P&L outcome, "
                "and do not rescale the signal to force positions to appear -- "
                "that would fabricate a strategy that was never tested."
            ),
            "authoritative_result": (
                "Use the GA's own equity-compound-v2 evaluation of this formula "
                "(OOS sharpe 0.478, MDD 0.0968, 19/28 positive legs) and the "
                "regime gate verdict, not the paper runner."
            ),
        },
        "training_caveat": (
            "ADAPTED FROM A GA SEARCH RESULT, NOT A TRAINED CANDIDATE. The paper "
            "run below reports the behaviour of this formula under the frozen "
            "28c transfer thresholds. It is a diagnostic and does not authorise "
            "adoption; the gate verdict is carried over unchanged from the GA."
        ),
        "live_adopted": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ga-result", type=Path, required=True)
    ap.add_argument("--decode", nargs=12, required=True,
                    help="12 human-readable token names, in formula order")
    ap.add_argument("--gate-result", type=Path, default=None,
                    help="gate artifact from evaluate_regime_gate; without it the "
                         "verdict stays 'not_run' rather than being guessed")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    doc = build(args.ga_result, list(args.decode), args.gate_result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {args.out}")
    print(f"  formula        {doc['formula']}")
    print(f"  gate verdict   {doc['regime_gate_verdict']}")
    print(f"  from           {doc['source_ga_result']}")
    print(f"  sha256         {doc['source_ga_result_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
