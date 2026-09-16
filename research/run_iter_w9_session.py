"""W9 session split (15m native, Top5): Asia/EU/US 8h per-coin sharpe/trades + FULL session-only arms.

Mirror research/run_weight_modes.py leg_net + quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1, static equal 0.2 weights,
aster perp lev2, fee=FEE fund=FUND. 15m native: cd/ts/vw x16, BPY=35040.
Legs/positions are computed on the FULL series (causal); sessions only segment
the resulting net/turn/pos series. Session by bar UTC hour:
Asia [0,8), EU [8,16), US [16,24).
Trades = entries of nonzero-pos blocks; session trades = entries in session.
Session-only arm = basket net/turn masked to session (0.0 outside);
arm trades = sum of leg session trades. Read-only, live untouched.
Output: results/iter_W9_session.json (verdict PENDING P0-3 FAIL, no adoption).
Partial JSON dumped after each coin unit (partial survives crash).
"""
import csv, datetime, json, math, os, pathlib, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from strategy_manager.config import FORMULA, LOCKED_ATOM, LOCKED_APT, LOCKED_ETC, LOCKED_KAS, LOCKED_TRX, LEV, FUND, FEE

assert LEV == 2.0, "LEV lock broken: %r" % LEV
assert FORMULA == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10], "FORMULA lock broken"

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
SPECS = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM, "APT": LOCKED_APT, "KAS": LOCKED_KAS}
W = {c: 0.2 for c in COINS}
SESSIONS = {"asia": (0, 8), "eu": (8, 16), "us": (16, 24)}
BPY = 35040.0
SCALE = 16
OUT = pathlib.Path("results/iter_W9_session.json")
LOG = pathlib.Path("logs/iter_W9_session.log")


def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")


def load15m(c):
    rows = list(csv.DictReader(open("data/data_1y/15m/%s.csv" % c)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])) for r in rows]


def common15m(coins):
    raw = {c: load15m(c) for c in coins}
    s = max(r[0][0] for r in raw.values()); e = min(r[-1][0] for r in raw.values())
    bars = {}; ts = None
    for c in coins:
        rr = [r for r in raw[c] if s <= r[0] <= e]
        bars[c] = [(r[1], r[2], r[3], r[4], r[5]) for r in rr]
        tts = [r[0] for r in rr]
        if ts is None:
            ts = tts
        else:
            assert tts == ts, "timestamp misalignment: %s" % c
    n = min(len(b) for b in bars.values())
    for c in coins:
        bars[c] = bars[c][:n]
    return bars, ts[:n]


def session_of(ts_ms):
    h = datetime.datetime.fromtimestamp(ts_ms / 1000, datetime.timezone.utc).hour
    if 0 <= h < 8:
        return "asia"
    if 8 <= h < 16:
        return "eu"
    return "us"


def build_sig(bars):
    n = len(bars)
    raw = {"open": torch.tensor([[b[0] for b in bars]]), "high": torch.tensor([[b[1] for b in bars]]), "low": torch.tensor([[b[2] for b in bars]]), "close": torch.tensor([[b[3] for b in bars]]), "volume": torch.tensor([[b[4] for b in bars]]), "liquidity": torch.full((1, n), 1e7), "fdv": torch.full((1, n), 1e8)}
    sig = StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets = [(bars[i + 1][3] - bars[i][3]) / bars[i][3] for i in range(n - 1)] + [0.0]
    return raw, torch.tensor([rets]), sig


def qmask(sig, q):
    if q is None or float(q) <= 0:
        return None
    a = sig.detach().float().abs().reshape(-1)
    k = max(1, int(len(a) * float(q)))
    thr = torch.topk(a, k).values.min()
    return (sig.detach().float().abs() >= thr).float()


def leg_net_pos(raw, rt, sig, spec, fee, fund, lev):
    bt = MemeBacktest(venue="aster", leverage=lev, short_enabled=True, funding_override=fund, fee_override=fee, long_th=spec["lth"], short_th=spec["sth"], cooldown_bars=int(spec["cd"]) * SCALE, bars_per_year=BPY, stop_loss=spec["sl"], time_stop=int(spec["ts"]) * SCALE, vol_target=spec["vt"], vol_window=int(spec["vw"]) * SCALE)
    sg = torch.sigmoid(sig); safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe; sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, spec["q"])
    if mk is not None:
        lp = lp * mk
    lp, sp = bt._apply_cooldown(lp, sp); lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt); lp, sp = lp * sc, sp * sc
    lp = lp.roll(1, dims=1); lp[:, 0] = 0; sp = sp.roll(1, dims=1); sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    gross = (lp - sp) * rt * bt.leverage
    tx = turn * bt.base_fee * bt.leverage
    fnd = (lp - sp) * bt.default_funding_rate * bt.leverage
    return (gross - tx - fnd)[0].tolist(), turn[0].tolist(), (lp - sp)[0].tolist()


def seg_idx(net, turn, idx, trades=None):
    s = [net[t] for t in idx]; n = len(s)
    m = sum(s) / n if n else 0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = 0.0; pk0 = -1e18; md = 0.0
    for x in s:
        cs += x; pk0 = max(pk0, cs); md = max(md, pk0 - cs)
    cum = sum(s)
    out = {"sharpe": round(sh, 3), "ann": round(cum / n * BPY, 4) if n else 0.0, "mdd": round(md, 4), "cum": round(cum, 4), "final_x": round(1.0 + cum, 4), "n": n, "turnover": round(sum(turn[t] for t in idx) / n, 6) if n else 0.0}
    if trades is not None:
        out["trades"] = int(trades)
    return out


def entries(pos, idx_set=None):
    es = [t for t in range(len(pos)) if abs(pos[t]) > 0.5 and (t == 0 or abs(pos[t - 1]) <= 0.5)]
    if idx_set is None:
        return es
    return [t for t in es if t in idx_set]


def dump(res):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))


def base_config(n):
    return {"engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; static equal 0.2 Top5; 15m native cd/ts/vw x16; legs on FULL series, sessions segment outputs only", "formula": list(FORMULA), "basket": {c: dict(SPECS[c]) for c in COINS}, "weights": dict(W), "sessions": {k: list(v) for k, v in SESSIONS.items()}, "session_def": "bar UTC hour: asia [0,8), eu [8,16), us [16,24)", "trades_def": "entries of nonzero-pos blocks on FULL pos series; session trades = entries in session; arm trades = sum of leg session trades", "venue": "aster", "lev": LEV, "fund": FUND, "fee": FEE, "grid": "15m", "grid_bars": n, "bpy": BPY, "scale": SCALE, "note": "session split only, read-only; live untouched (stays FULL-calendar equal-weight)"}


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_W9_session start\n")
    bars, ts = common15m(COINS)
    n = len(bars["ETC"])
    sess = [session_of(t) for t in ts]
    sidx = {s: [t for t in range(n) if sess[t] == s] for s in SESSIONS}
    scount = {s: len(sidx[s]) for s in SESSIONS}
    log("common 15m native n=%d counts=%s scale=x%d" % (n, scount, SCALE))
    mats = {c: build_sig(bars[c]) for c in COINS}
    legs = {}; turns = {}; poss = {}
    per_coin = {}
    res = {"config": base_config(n), "per_coin": per_coin, "arms": {}, "compare": {}, "session_counts": scount, "verdict": "PENDING", "decision": "KEEP_FULL_NO_SESSION_FILTER", "decision_note": "\u5f85\u5b9a\uff0c\u4e0d\u505asession\u8fc7\u6ee4\u3002session\u4ec5\u4e3a\u5bf9\u7167\uff0c\u4e0d\u91c7\u7528\uff0c\u4e0dlive\u300215m\u539f\u751f\u683c\u5b50\u3002", "partial": True}
    for c in COINS:
        raw, rt, sg = mats[c]
        legs[c], turns[c], poss[c] = leg_net_pos(raw, rt, sg, SPECS[c], FEE, FUND, LEV)
        ent = entries(poss[c])
        eset = set(ent)
        row = {"FULL": seg_idx(legs[c], turns[c], range(n), trades=len(ent))}
        for s in SESSIONS:
            sset = set(sidx[s])
            row[s] = seg_idx(legs[c], turns[c], sidx[s], trades=sum(1 for t in ent if t in sset))
        per_coin[c] = row
        f = row["FULL"]
        log("%s FULL sh=%.3f mdd=%.4f cum=%.4f tr=%d | asia sh=%.3f tr=%d | eu sh=%.3f tr=%d | us sh=%.3f tr=%d" % (c, f["sharpe"], f["mdd"], f["cum"], f["trades"], row["asia"]["sharpe"], row["asia"]["trades"], row["eu"]["sharpe"], row["eu"]["trades"], row["us"]["sharpe"], row["us"]["trades"]))
        res["done"] = list(per_coin.keys())
        dump(res)
        log("partial dump done=%s" % res["done"])
    net = [sum(legs[c][t] * W[c] for c in COINS) for t in range(n)]
    turn = [sum(turns[c][t] * W[c] for c in COINS) for t in range(n)]
    arms = {}
    for s in SESSIONS:
        inset = set(sidx[s])
        net_m = [net[t] if t in inset else 0.0 for t in range(n)]
        turn_m = [turn[t] if t in inset else 0.0 for t in range(n)]
        atr = sum(per_coin[c][s]["trades"] for c in COINS)
        arms[s] = {"FULL_calendar": seg_idx(net_m, turn_m, range(n), trades=atr), "subset": seg_idx(net, turn, sidx[s], trades=atr)}
        a = arms[s]
        log("arm %s cal sh=%.3f mdd=%.4f cum=%.4f | subset sh=%.3f mdd=%.4f cum=%.4f tr=%d" % (s, a["FULL_calendar"]["sharpe"], a["FULL_calendar"]["mdd"], a["FULL_calendar"]["cum"], a["subset"]["sharpe"], a["subset"]["mdd"], a["subset"]["cum"], atr))
    basket_full = seg_idx(net, turn, range(n), trades=sum(per_coin[c]["FULL"]["trades"] for c in COINS))
    arms["basket_FULL"] = basket_full
    log("basket FULL sh=%.3f mdd=%.4f cum=%.4f tr=%d" % (basket_full["sharpe"], basket_full["mdd"], basket_full["cum"], basket_full["trades"]))
    res["arms"] = arms
    res["compare"] = {
        "best_session_by_coin": {c: max(SESSIONS, key=lambda s: per_coin[c][s]["sharpe"]) for c in COINS},
        "arm_subset_sharpe": {s: arms[s]["subset"]["sharpe"] for s in SESSIONS},
        "arm_calendar_sharpe": {s: arms[s]["FULL_calendar"]["sharpe"] for s in SESSIONS},
        "arm_subset_cum": {s: arms[s]["subset"]["cum"] for s in SESSIONS},
        "arm_trades": {s: arms[s]["subset"]["trades"] for s in SESSIONS},
    }
    res["partial"] = False
    if "done" in res:
        del res["done"]
    dump(res)
    log("wrote %s verdict=%s decision=%s" % (OUT, res["verdict"], res["decision"]))


if __name__ == "__main__":
    main()
