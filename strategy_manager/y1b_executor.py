"""Y1b shadow/live executor (E10 lock): signals -> perp gate -> broker.

Default OFF / paper: Y1B_LIVE_ENABLED != 1 means dry-run only (no orders).
Live requires ALL: Y1B_LIVE_ENABLED=1, PAPER_MODE unset, ASTER keys present,
deadman ok, STOP absent, perp gate pass. Size = Y1B_NOTIONAL_USDT / price,
capped by PERP_MAX_NOTIONAL_USDT. Start with testnet (ASTER_TESTNET=true).
"""
from __future__ import annotations
import os
from dataclasses import dataclass

from strategy_manager.config import LEV, RiskConfig
from strategy_manager.y1b_basket import latest_signals
from strategy_manager.risk import RiskEngine
from execution.brokers.base import Side

SYMBOLS = {"ETC": "ETCUSDT", "TRX": "TRXUSDT"}

def live_enabled() -> bool:
    return os.getenv("Y1B_LIVE_ENABLED", "").strip() == "1"

def paper_mode() -> bool:
    return os.getenv("PAPER_MODE", "").lower() in {"1", "true", "yes"}

@dataclass
class Plan:
    coin: str
    symbol: str
    want: float  # -1/0/+1
    side: Side | None
    size: float
    price: float
    gate_ok: bool
    reason: str

async def build_plans(broker, risk: RiskEngine | None = None,
                      notional: float | None = None) -> list[Plan]:
    risk = risk or RiskEngine()
    cfg = risk.risk_config if isinstance(risk.risk_config, RiskConfig) else RiskConfig()
    want_map = latest_signals()
    notion = float(notional or os.getenv("Y1B_NOTIONAL_USDT", "50"))
    plans: list[Plan] = []
    for coin, want in want_map.items():
        sym = SYMBOLS[coin]
        try:
            price = await broker.get_price(sym)
        except Exception as e:
            plans.append(Plan(coin, sym, want, None, 0.0, 0.0, False, f"price fail: {e}"))
            continue
        if want == 0 or price <= 0:
            plans.append(Plan(coin, sym, want, None, 0.0, price, True, "flat"))
            continue
        side = Side.BUY if want > 0 else Side.SELL
        size = min(notion, cfg.perp_max_notional_usdt) / price
        funding = None
        get_funding = broker.__dict__.get("get_funding_rate", None)
        if get_funding is None:
            get_funding = getattr(type(broker), "get_funding_rate", None)
            if isinstance(get_funding, property):
                get_funding = None
        if callable(get_funding):
            try:
                funding = await get_funding(sym)
            except Exception:
                funding = None
            try:
                funding = float(funding) if funding is not None else None
            except (TypeError, ValueError):
                funding = None
        ok, reason = risk.check_perp(sym, int(LEV), min(notion, cfg.perp_max_notional_usdt), funding)
        plans.append(Plan(coin, sym, want, side, size, price, ok, reason))
    return plans

def stop_requested(path: str | None = None) -> bool:
    p = path or os.getenv("STOP_SIGNAL_PATH", "STOP_SIGNAL")
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            return f.read().strip().upper() in {"", "STOP", "STOPPED"}
    except OSError:
        return True


async def preflight(broker, risk: RiskEngine) -> tuple[bool, str]:
    """Live safety checks: STOP absent + circuit closed + deadman ok."""
    if stop_requested():
        return False, "stop_signal"
    blocked, reason = risk.check_circuit()
    if blocked:
        return False, f"circuit:{reason}"
    try:
        ok = await broker.enable_deadman(60)
    except NotImplementedError:
        return True, ""
    except Exception as e:
        return False, f"deadman err: {e}"
    if ok is False:
        return False, "deadman FAILED"
    return True, ""


async def sync_positions(broker, plans: list[Plan], pm, live: bool):
    """Flatten venue positions whose want==0; report flips. Dry-run only reports."""
    actions = []
    for p in plans:
        try:
            vpos = await broker.get_position(p.symbol)
        except Exception as e:
            actions.append({"symbol": p.symbol, "note": f"pos query fail: {e}"})
            continue
        try:
            _size = float(vpos.size) if vpos is not None else 0.0
        except (TypeError, ValueError):
            _size = 0.0
        _side = getattr(vpos, "side", "LONG") if vpos is not None else "LONG"
        held = _size * (1 if _side == "LONG" else -1) if vpos else 0.0
        if p.want == 0 and held != 0:
            if not live:
                actions.append({"symbol": p.symbol, "dry_run_close": True, "held": held})
                continue
            side = Side.SELL if held > 0 else Side.BUY
            res = await broker.market_open(p.symbol, side, abs(held))
            actions.append({"symbol": p.symbol, "closed": res.ok, "oid": res.oid,
                            "reason": res.reason})
            if res.ok:
                pm.reconcile(p.symbol, 0.0, venue="aster")
        elif p.want != 0 and held != 0 and ((held > 0) != (p.want > 0)):
            actions.append({"symbol": p.symbol, "flip_needed": True, "held": held,
                            "want": p.want, "dry_run": not live})
            if live:
                side = Side.BUY if p.want > 0 else Side.SELL
                res = await broker.market_open(p.symbol, side, p.size + abs(held))
                actions.append({"symbol": p.symbol, "flipped": res.ok, "oid": res.oid,
                                "reason": res.reason})
    return actions


async def run_once(broker=None, risk: RiskEngine | None = None,
                   notional: float | None = None, dry_run: bool | None = None):
    """One Y1b cycle. Returns (plans, results, sync). dry_run default True unless live fully enabled."""
    from strategy_manager.portfolio import PortfolioManager
    risk = risk or RiskEngine()
    if broker is None:
        from execution.brokers.aster import AsterBroker
        broker = AsterBroker()
    live = live_enabled() and not paper_mode() and (dry_run is False)
    plans = await build_plans(broker, risk, notional)
    results = []
    pm = PortfolioManager(state_file=os.getenv("Y1B_STATE", "y1b_state.json"))
    if live:
        ok, reason = await preflight(broker, risk)
        if not ok:
            return plans, [{"blocked": True, "reason": reason}], []
    sync = await sync_positions(broker, plans, pm, live)
    for p in plans:
        if not live:
            results.append({"symbol": p.symbol, "dry_run": True, "want": p.want,
                            "size": round(p.size, 6), "price": p.price, "gate_ok": p.gate_ok})
            continue
        if not p.gate_ok or p.side is None:
            results.append({"symbol": p.symbol, "skipped": True, "reason": p.reason})
            continue
        try:
            vpos = await broker.get_position(p.symbol)
        except Exception:
            vpos = None
        if vpos is not None:
            try:
                _held = float(vpos.size)
            except (TypeError, ValueError):
                _held = 0.0
            _same = (_held > 0) == (p.want > 0) and _held != 0
            if _same and pm.has_sig(vpos.raw.get("oid", "") if isinstance(vpos.raw, dict) else "", venue="aster"):
                results.append({"symbol": p.symbol, "skipped": True, "reason": "idempotent-held"})
                continue
            if _same:
                results.append({"symbol": p.symbol, "skipped": True, "reason": "already-held"})
                continue
        try:
            await broker.set_leverage(p.symbol, int(LEV))
        except Exception as e:
            results.append({"symbol": p.symbol, "error": f"leverage: {e}"})
            continue
        res = await broker.market_open(p.symbol, p.side, p.size)
        results.append({"symbol": p.symbol, "ok": res.ok, "oid": res.oid,
                        "fill": res.fill_price, "reason": res.reason})
        if res.ok:
            try:
                vpos = await broker.get_position(p.symbol)
                amt = vpos.size if vpos else p.size
            except Exception:
                amt = p.size
            pm.add_position(p.symbol, p.coin, res.fill_price or p.price, amt, 0.0,
                            tx_sig=res.oid or None, venue="aster",
                            side="LONG" if p.want > 0 else "SHORT", leverage=float(LEV))
    return plans, results, sync
