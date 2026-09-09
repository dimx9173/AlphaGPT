"""R1 robustness: Q2 candidate [0,20,18,14,20,12,14,18,11,14,14,14] walk-forward + fee/param stress.
Coins BTC/SOL/ETC/TRX/DOGE, 4h 6570 bars. BOTH legs default (0.85/0.15/cd6) aster 2x fund 0.0005.
- 6-fold walk-forward (1095 bars/fold) per coin, cand vs baseline.
- Fee sweep [0.0004,0.0008,0.0012] on OOS H2+FULL per coin (cand + base).
- Param jitter lth x cd grid on BTC+ETC OOS-H2 (cand + base).
Output: backtest_R1_robust.json
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_q2 as Q
from model_core.config import ModelConfig
from model_core.backtest import MemeBacktest
from model_core.vm import StackVM

CAND = [0,20,18,14,20,12,14,18,11,14,14,14]
BASE = Q.BASELINE_FORMULA
COINS = ("BTC","SOL","ETC","TRX","DOGE")
LTH, STH, CD = 0.85, 0.15, 6
FEES = [0.0004, 0.0008, 0.0012]
LTHS = [0.83, 0.85, 0.88]
CDS = [3, 6, 12]
NFOLD = 6

def make_bt(lth=LTH, sth=STH, cd=CD, sl=None, fee=None, fund=0.0005, side="both"):
    kw = dict(venue="aster", leverage=2.0, long_th=lth, short_th=sth,
              cooldown_bars=cd, bars_per_year=Q.BARS_PER_YEAR, stop_loss=sl,
              funding_override=fund)
    if side == "long":
        kw.update(short_enabled=False)
    elif side == "short":
        kw.update(short_enabled=True, long_th=10.0)
    else:
        kw.update(short_enabled=True)
    if fee is not None:
        kw["fee_override"] = fee
    return MemeBacktest(**kw)

def fit_of(fit):
    return float(fit.item()) if hasattr(fit, "item") else float(fit)

def eval_slice(vm, formula, feats, tgt, raw, device, s, e, lth=LTH, sth=STH, cd=CD,
               sl=None, fee=None, side="both"):
    res = vm.execute(formula, feats.to(device))
    if res is None:
        return None
    def cut(t, _s=s, _e=e):
        return t[:, _s:_e] if t.dim() == 2 else t[:, :, _s:_e]
    tgt_d = tgt.to(device)
    rawd = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in raw.items()}
    bt = make_bt(lth, sth, cd, sl, fee, side=side)
    fit, cum = bt.evaluate(cut(res), {k: cut(v) for k, v in rawd.items()}, cut(tgt_d))
    m = dict(bt.last_metrics)
    wlen = e - s
    ann = float(cum) * (Q.BARS_PER_YEAR / wlen)
    return {"fitness": fit_of(fit), "sharpe": round(m["sharpe"], 3),
            "ann": round(ann, 4), "mdd": round(m["max_dd"], 4),
            "cum": round(float(cum), 4)}

def main():
    device = ModelConfig.DEVICE
    vm = StackVM()
    all_bars = Q.load_4h(COINS)
    report = {"candidate": CAND, "decoded_cand": Q.decode(CAND),
              "baseline": BASE, "decoded_base": Q.decode(BASE),
              "config": {"lth": LTH, "sth": STH, "cd": CD, "venue": "aster",
                         "leverage": 2.0, "funding": 0.0005, "side": "both",
                         "bars_per_year": Q.BARS_PER_YEAR},
              "per_coin": {}}
    for coin in COINS:
        bars = all_bars[coin]
        n = len(bars)
        f, t, r = Q.build_single(bars)
        T = f.shape[2] if f.dim() == 3 else f.shape[1]
        assert T == n == 6570, f"{coin} T={T} n={n}"
        c = {"bars": n}
        # ---- 1) walk-forward 6-fold ----
        fw = T // NFOLD
        wf = {"cand": [], "base": []}
        for k in range(NFOLD):
            s, e = k * fw, (k + 1) * fw if k < NFOLD - 1 else T
            rc = eval_slice(vm, CAND, f, t, r, device, s, e)
            rb = eval_slice(vm, BASE, f, t, r, device, s, e)
            rc["fold"] = rb["fold"] = k
            rc["bars"] = rb["bars"] = e - s
            wf["cand"].append(rc)
            wf["base"].append(rb)
        c["walkforward_6fold"] = wf
        pos_c = sum(1 for x in wf["cand"] if x["sharpe"] > 0)
        min_c = min(x["sharpe"] for x in wf["cand"])
        pos_b = sum(1 for x in wf["base"] if x["sharpe"] > 0)
        min_b = min(x["sharpe"] for x in wf["base"])
        c["wf_pass_cand"] = (pos_c >= 4 and min_c > -1.5)
        c["wf_summary"] = {"cand_pos": pos_c, "cand_min": min_c,
                           "base_pos": pos_b, "base_min": min_b}
        print(f"WF {coin} cand " + " ".join(str(x['sharpe']) for x in wf["cand"]) +
              f" pos={pos_c}/6 min={min_c} PASS={c['wf_pass_cand']}", flush=True)
        print(f"WF {coin} base " + " ".join(str(x['sharpe']) for x in wf["base"]) +
              f" pos={pos_b}/6 min={min_b}", flush=True)
        # ---- 2) fee sweep on OOS H2 + OOS FULL ----
        no = n - int(n * Q.TRAIN_FRAC)
        os_, oe = n - no, n
        h = no // 2
        bounds = {"FULL": (os_, oe), "H2": (os_ + h, oe)}
        fee = {"cand": {}, "base": {}}
        for seg, (s, e) in bounds.items():
            for fee_v in FEES:
                key = str(fee_v)
                rc = eval_slice(vm, CAND, f, t, r, device, s, e, fee=fee_v)
                rb = eval_slice(vm, BASE, f, t, r, device, s, e, fee=fee_v)
                fee["cand"].setdefault(seg, {})[key] = rc
                fee["base"].setdefault(seg, {})[key] = rb
        c["fee_sweep"] = fee
        for seg in ("H2", "FULL"):
            line = f"FEE {coin} {seg} cand " + " ".join(
                f"{k}:{fee['cand'][seg][k]['sharpe']}" for k in [str(x) for x in FEES])
            line += " | base " + " ".join(
                f"{k}:{fee['base'][seg][k]['sharpe']}" for k in [str(x) for x in FEES])
            print(line, flush=True)
        c["fee_pass_H2_2x"] = fee["cand"]["H2"][str(0.0008)]["sharpe"] > 1.0
        # ---- 3) param jitter on OOS H2 (BTC+ETC only) ----
        if coin in ("BTC", "ETC"):
            s, e = bounds["H2"]
            jit = {"cand": [], "base": []}
            for lth in LTHS:
                for cd in CDS:
                    rc = eval_slice(vm, CAND, f, t, r, device, s, e, lth=lth, cd=cd)
                    rb = eval_slice(vm, BASE, f, t, r, device, s, e, lth=lth, cd=cd)
                    rc.update(lth=lth, cd=cd)
                    rb.update(lth=lth, cd=cd)
                    jit["cand"].append(rc)
                    jit["base"].append(rb)
            c["jitter_H2"] = jit
            rng_c = max(x["sharpe"] for x in jit["cand"]) - min(x["sharpe"] for x in jit["cand"])
            rng_b = max(x["sharpe"] for x in jit["base"]) - min(x["sharpe"] for x in jit["base"])
            c["jitter_range_cand"] = round(rng_c, 3)
            c["jitter_range_base"] = round(rng_b, 3)
            c["jitter_pass_cand"] = rng_c < 1.5
            print(f"JIT {coin} H2 cand sharpes " +
                  " ".join(f"({x['lth']},{x['cd']})={x['sharpe']}" for x in jit["cand"]) +
                  f" range={rng_c:.3f} PASS={c['jitter_pass_cand']}", flush=True)
            print(f"JIT {coin} H2 base sharpes " +
                  " ".join(f"({x['lth']},{x['cd']})={x['sharpe']}" for x in jit["base"]) +
                  f" range={rng_b:.3f}", flush=True)
        # ---- verdict ----
        if coin in ("BTC", "ETC"):
            ok = c["wf_pass_cand"] and c["fee_pass_H2_2x"] and c["jitter_pass_cand"]
            near = (c["wf_pass_cand"] or c["fee_pass_H2_2x"]) and c["jitter_pass_cand"]
        else:
            ok = c["wf_pass_cand"] and c["fee_pass_H2_2x"]
            near = c["wf_pass_cand"] or c["fee_pass_H2_2x"]
        c["verdict"] = "PAPER-READY" if ok else ("WATCH" if near else "REJECT")
        print(f"VERDICT {coin}: {c['verdict']} wf={c['wf_pass_cand']} "
              f"fee2x={c['fee_pass_H2_2x']} " +
              (f"jit={c.get('jitter_pass_cand')}" if coin in ("BTC", "ETC") else ""), flush=True)
        report["per_coin"][coin] = c
    json.dump(report, open("results/backtest_R1_robust.json", "w"), indent=1)
    print("R1 DONE -> backtest_R1_robust.json", flush=True)

if __name__ == "__main__":
    main()
