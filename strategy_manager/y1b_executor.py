"""Y1b shadow/live executor (E10 lock): signals -> perp gate -> broker.

Default OFF / paper: Y1B_LIVE_ENABLED != 1 means dry-run only (no orders).
Live requires ALL: Y1B_LIVE_ENABLED=1, PAPER_MODE unset, venue keys present,
deadman ok, STOP absent, perp gate pass. Size = Y1B_NOTIONAL_USDT / price,
capped by PERP_MAX_NOTIONAL_USDT. Start with testnet/demo.
Venues: aster (default) | binance | bybit | okx via Y1B_VENUE.
"""
from __future__ import annotations
import os
from dataclasses import dataclass

from strategy_manager.config import LEV, RiskConfig
from strategy_manager.y1b_basket import basket_signals, latest_signals
from strategy_manager.risk import RiskEngine
from execution.brokers.base import Side

SYMBOLS = {"ETC": "ETCUSDT", "TRX": "TRXUSDT"}
SYMBOLS_5 = {"ETC": "ETCUSDT", "TRX": "TRXUSDT", "ATOM": "ATOMUSDT", "APT": "APTUSDT", "KAS": "KASUSDT"}


def active_symbols():
    import os as _os
    if _os.getenv("Y1B_TOP5", "").strip().lower() in {"1", "true", "yes"}:
        return dict(SYMBOLS_5)
    return dict(SYMBOLS)


def venue_name(broker) -> str:
    """Venue key for portfolio/reconcile, derived from broker (default aster)."""
    try:
        v = getattr(broker, "venue", None)
        val = getattr(v, "value", v)
        if isinstance(val, str) and val:
            return val.lower()
    except Exception:
        pass
    return "aster"


def make_broker(venue: str | None = None):
    """Factory for Y1b venues: aster (default) | binance | bybit | okx."""
    import os as _os
    name = (venue or _os.getenv("Y1B_VENUE", "aster")).strip().lower()
    if name == "binance":
        from execution.brokers.binance import BinanceBroker
        b = BinanceBroker()
        try:
            b.set_deadman_symbols(list(active_symbols().values()))
        except Exception:
            pass
        return b
    if name == "bybit":
        from execution.brokers.bybit import BybitBroker
        return BybitBroker()
    if name == "okx":
        from execution.brokers.okx import OkxBroker
        return OkxBroker()
    from execution.brokers.aster import AsterBroker
    return AsterBroker()

def live_enabled() -> bool:
    return os.getenv("Y1B_LIVE_ENABLED", "").strip() == "1"

def paper_mode() -> bool:
    return (os.getenv("PAPER_MODE", "") or "").lower() in {"1", "true", "yes"}


# === P0-2 swap gates (E14/E15). ALL default OFF. ===
# Y1B_HYST_EPS: sigmoid hysteresis band half-width (e.g. 0.1). When ON, a flip
#   needs |sg-0.5| >= eps on the new side, else want degrades to 0 (hold).
# Y1B_MIN_HOLD_BARS: min 4h bars to hold before a flip (e.g. 2). Uses
#   PortfolioManager entry_time; risk-reducing closes (want==0/flat) exempt.
# Y1B_COST_K: cost-aware flips. Expected edge per unit must exceed
#   k*(fee2x + slip_bp + funding) else degrade flip to hold (close-only if
#   gate fails follows existing close-only path).
def _swap_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def hyst_eps() -> float:
    return _swap_env_float("Y1B_HYST_EPS", 0.0)


def min_hold_bars() -> int:
    try:
        return max(int(float(os.getenv("Y1B_MIN_HOLD_BARS", "0"))), 0)
    except (TypeError, ValueError):
        return 0


def cost_k() -> float:
    return _swap_env_float("Y1B_COST_K", 0.0)


def swap_gates_on() -> bool:
    return hyst_eps() > 0 or min_hold_bars() > 0 or cost_k() > 0


def apply_swap_gates(want: float, sg: float | None, coin: str, pm=None,
                     edge_per_unit: float | None = None,
                     fee2x: float = 0.0008, slip_bp: float = 5.0,
                     funding: float | None = None) -> tuple[float, str]:
    """Degrade want->0 (hold) when a swap gate blocks. Returns (want, note).
    note '' means pass. Pure function (pm only read for entry_time)."""
    if want == 0:
        return want, ""
    eps = hyst_eps()
    if eps > 0 and sg is not None:
        try:
            dist = abs(float(sg) - 0.5)
        except (TypeError, ValueError):
            dist = 0.0
        if dist < eps:
            return 0.0, "hyst"
    mh = min_hold_bars()
    if mh > 0 and pm is not None:
        try:
            poss = getattr(pm, "positions", {}) or {}
            hit = None
            for _k, _pos in poss.items():
                try:
                    if getattr(_pos, "token_address", "") == coin or getattr(_pos, "symbol", "") == coin:
                        hit = _pos
                        break
                except Exception:
                    continue
            if hit is not None:
                import time as _t
                age_h = (_t.time() - float(getattr(hit, "entry_time", 0) or 0)) / 3600.0
                if 0 <= age_h < mh * 4.0:
                    return 0.0, "min-hold"
        except Exception:
            pass
    k = cost_k()
    if k > 0 and edge_per_unit is not None:
        try:
            slip = float(slip_bp) / 10000.0
            fund = abs(float(funding)) if funding is not None else 0.0005
            if float(edge_per_unit) < k * (float(fee2x) + slip + fund):
                return 0.0, "cost"
        except (TypeError, ValueError):
            pass
    return want, ""

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
    try:
        _bs = basket_signals()
        want_map = {c: (s[-1] if s else 0.0) for c, s in _bs["signals"].items()}
        _sg_map = dict(_bs.get("sg_last") or {})
    except Exception:
        want_map = latest_signals()
        _sg_map = {}
    try:
        notion = float(notional if notional is not None else os.getenv("Y1B_NOTIONAL_USDT", "50"))
    except (TypeError, ValueError):
        notion = 50.0
    if notion <= 0:
        _syms = active_symbols()
        return [Plan(coin, _syms.get(coin, coin), want, None, 0.0, 0.0, False, "bad-notional")
                for coin, want in want_map.items()]
    plans: list[Plan] = []
    _syms = active_symbols()
    for coin, want in want_map.items():
        sym = _syms.get(coin, coin)
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
        # P1-2 drawdown brake (env-gated, default OFF => scale 1.0).
        try:
            from strategy_manager.y1b_basket import brake_open_scale
            size = size * brake_open_scale(coin)
        except Exception:
            pass
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
        # P0-2 swap gates (env-gated, default OFF => pass-through).
        # Hyst needs sg of the new side; min-hold/cost need live pm -> applied
        # at flip time in sync/run loop via apply_swap_gates; here record sg.
        try:
            _sg = _sg_map.get(coin)
        except Exception:
            _sg = None
        if swap_gates_on() and want != 0 and _sg is not None and hyst_eps() > 0:
            try:
                if abs(float(_sg) - 0.5) < hyst_eps():
                    plans.append(Plan(coin, sym, 0.0, None, 0.0, price, True, "hyst"))
                    continue
            except (TypeError, ValueError):
                pass
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
    """Flatten venue positions whose want==0 (close-only allowlist, gate-exempt
    by design: risk-reducing); report flips. Gate-failed flips close-only.
    Dry-run only reports."""
    _venue = venue_name(broker)
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
                pm.reconcile(p.symbol, 0.0, venue=_venue)
        elif p.want != 0 and held != 0 and ((held > 0) != (p.want > 0)):
            actions.append({"symbol": p.symbol, "flip_needed": True, "held": held,
                            "want": p.want, "dry_run": not live})
            if live:
                # P0-2 swap gates on flips: min-hold + cost degrade to hold
                # (close-only already handled by want==0 path next cycle).
                if swap_gates_on():
                    _gw, _gn = apply_swap_gates(
                        p.want, None, p.coin, pm=pm,
                        edge_per_unit=None, funding=None)
                    # hyst already applied at plan time (needs sg); here only
                    # enforce min-hold; cost gate needs edge estimate -> skip
                    # live (frontier-tuned offline), record only.
                    if _gn == "min-hold":
                        actions.append({"symbol": p.symbol, "held": True,
                                        "reason": "min-hold", "held_size": held})
                        continue
                if not p.gate_ok:
                    # H1 fix: gate-failed flips degrade to close-only (risk-reducing),
                    # never open fresh size against a failed perp gate.
                    side = Side.SELL if held > 0 else Side.BUY
                    res = await broker.market_open(p.symbol, side, abs(held))
                    actions.append({"symbol": p.symbol, "closed_only_gate_fail": res.ok,
                                    "oid": res.oid, "reason": p.reason or res.reason})
                    if res.ok:
                        pm.reconcile(p.symbol, 0.0, venue=_venue)
                    continue
                side = Side.BUY if p.want > 0 else Side.SELL
                res = await broker.market_open(p.symbol, side, p.size + abs(held))
                actions.append({"symbol": p.symbol, "flipped": res.ok, "oid": res.oid,
                                "reason": res.reason})
    return actions


def decision_aligned(hour: int | None = None) -> bool:
    """4h decision gate: only UTC hour%4==0 may open/flip. Default OFF."""
    import datetime as _dt
    if (__import__("os").getenv("Y1B_DECISION_ALIGN", "") or "").lower() not in {"1", "true", "yes"}:
        return True
    h = hour if hour is not None else _dt.datetime.now(_dt.timezone.utc).hour
    return (h % 4) == 0


async def run_once(broker=None, risk: RiskEngine | None = None,
                   notional: float | None = None, dry_run: bool | None = None,
                   decision_only: bool = False, hour: int | None = None):
    """One Y1b cycle. Returns (plans, results, sync). dry_run default True unless live fully enabled.
    decision_only=True (or non-aligned hour with Y1B_DECISION_ALIGN=1): risk-only,
    sync closes allowed, no fresh opens (market_open zero-call)."""
    from strategy_manager.portfolio import PortfolioManager
    risk = risk or RiskEngine()
    if broker is None:
        broker = make_broker()
    _venue = venue_name(broker)
    live = live_enabled() and not paper_mode() and (dry_run is False)
    _risk_only = bool(decision_only) or not decision_aligned(hour)
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
        if _risk_only:
            # Risk-only cycle: report, never open fresh size.
            try:
                _v = await broker.get_position(p.symbol)
                _h = (_v.size if _v else 0.0)
            except Exception:
                _h = 0.0
            results.append({"symbol": p.symbol, "skipped": True,
                            "reason": "risk-only",
                            "want": p.want, "held": _h})
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
                _size = float(vpos.size)
            except (TypeError, ValueError):
                _size = 0.0
            _vside = getattr(vpos, "side", "LONG")
            _held = _size * (1 if _vside == "LONG" else -1)
            _same = (_held > 0) == (p.want > 0) and _held != 0
            if _same and pm.has_sig(vpos.raw.get("oid", "") if isinstance(vpos.raw, dict) else "", venue=_venue):
                results.append({"symbol": p.symbol, "skipped": True, "reason": "idempotent-held"})
                continue
            if _same:
                results.append({"symbol": p.symbol, "skipped": True, "reason": "already-held"})
                continue
        try:
            lev_ok = await broker.set_leverage(p.symbol, int(LEV))
        except Exception as e:
            results.append({"symbol": p.symbol, "error": f"leverage: {e}"})
            continue
        if lev_ok is False:
            results.append({"symbol": p.symbol, "skipped": True, "reason": "leverage-rejected"})
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
                            tx_sig=res.oid or None, venue=_venue,
                            side="LONG" if p.want > 0 else "SHORT", leverage=float(LEV))
    return plans, results, sync
