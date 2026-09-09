import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.dirname(os.path.abspath(__file__)) else ".")
import pathlib as _pl
root = _pl.Path(__file__).parent.parent if _pl.Path(__file__).parent.name=="research" else _pl.Path(".")
os.chdir(str(root))

import csv, math, json, pathlib
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3,2,7,2,7,11,15,4,4,6,6,10]
COINS = ['ETC','TRX']
FUND = 0.0005
FEE_BASE = 0.0004
Q = 0.3
VT_NONE = None
VT_012 = 0.012
VW = 12
TS = 24
FULL_N = 6580
FOLD_N = FULL_N // 12

# 6 configs spec
# Y1b: both q0.3 vtNone (mirror AB A1)
# Z1:  both q0.3 vt0.012 (mirror AB A2)
# AA-H1: (0.10/15/9) -> ETC 0.88/0.10/15/None/24 TRX 0.85/0.10/9/0.05/24 vt0.012 q0.3 both
# AA-H2: (0.12/15/9)
# AA-FULLbest: (0.11/18/9)
# AB-B2: short vt0.012 q0.3 (mirror AB B2) both legs side short

CONFIGS = [
    {"id":"Y1b", "label":"Y1b_both_q03_vtNone", "sth":0.12, "etc_cd":18, "trx_cd":6, "side":"both", "vt":None, "vw":12,
     "spec":{"ETC":(0.88,0.12,18,None,24,"both",None,12), "TRX":(0.85,0.12,6,0.05,24,"both",None,12)}},
    {"id":"Z1", "label":"Z1_both_q03_vt012", "sth":0.12, "etc_cd":18, "trx_cd":6, "side":"both", "vt":0.012, "vw":12,
     "spec":{"ETC":(0.88,0.12,18,None,24,"both",0.012,12), "TRX":(0.85,0.12,6,0.05,24,"both",0.012,12)}},
    {"id":"AA-H1", "label":"AA-H1_sth0.10_15_9", "sth":0.10, "etc_cd":15, "trx_cd":9, "side":"both", "vt":0.012, "vw":12,
     "spec":{"ETC":(0.88,0.10,15,None,24,"both",0.012,12), "TRX":(0.85,0.10,9,0.05,24,"both",0.012,12)}},
    {"id":"AA-H2", "label":"AA-H2_sth0.12_15_9", "sth":0.12, "etc_cd":15, "trx_cd":9, "side":"both", "vt":0.012, "vw":12,
     "spec":{"ETC":(0.88,0.12,15,None,24,"both",0.012,12), "TRX":(0.85,0.12,9,0.05,24,"both",0.012,12)}},
    {"id":"AA-FULLbest", "label":"AA-FULLbest_sth0.11_18_9", "sth":0.11, "etc_cd":18, "trx_cd":9, "side":"both", "vt":0.012, "vw":12,
     "spec":{"ETC":(0.88,0.11,18,None,24,"both",0.012,12), "TRX":(0.85,0.11,9,0.05,24,"both",0.012,12)}},
    {"id":"AB-B2", "label":"AB-B2_short_q03_vt012", "sth":0.12, "etc_cd":18, "trx_cd":6, "side":"short", "vt":0.012, "vw":12,
     "spec":{"ETC":(0.88,0.12,18,None,24,"short",0.012,12), "TRX":(0.85,0.12,6,0.05,24,"short",0.012,12)}},
]

SEG_MAP = {"FULL":(0,6580), "H2":(6077,6570), "B":(6380,6580), "C":(6080,6580)}
FEE_SCAN = [0.0002, 0.0004, 0.0008, 0.0012]

def load_bars(coin):
    rows=list(csv.DictReader(open(f"data/data_15m_3y/{coin}.csv")))
    bars=[]
    for i in range(0,len(rows),16):
        blk=rows[i:i+16]
        if len(blk)<16: break
        bars.append((float(blk[0]["open"]), max(float(x["high"]) for x in blk), min(float(x["low"]) for x in blk), float(blk[-1]["close"]), sum(float(x["volume"]) for x in blk)))
    return bars

def build_mats(bars):
    n=len(bars)
    raw={"open":torch.tensor([[b[0] for b in bars]]),
         "high":torch.tensor([[b[1] for b in bars]]),
         "low":torch.tensor([[b[2] for b in bars]]),
         "close":torch.tensor([[b[3] for b in bars]]),
         "volume":torch.tensor([[b[4] for b in bars]]),
         "liquidity":torch.full((1,n), 1e7),
         "fdv":torch.full((1,n), 1e8)}
    sig=StackVM().execute(FORMULA, FeatureEngineer.compute_features(raw))
    rets=[(bars[i+1][3]-bars[i][3])/bars[i][3] for i in range(n-1)] + [0.0]
    return raw, torch.tensor([rets]), sig

def quantile_mask_long(sig,q):
    if q is None:
        return None
    a=sig.detach().float().abs().reshape(-1)
    k=max(1,int(len(a)*float(q)))
    thr=torch.topk(a,k).values.min()
    return (sig.detach().float().abs() >= thr).float()

def leg_series_with_pos(raw, rets_t, sig, lth, sth, cd, sl, ts, side, vt, vw, fee, fund, q):
    kw=dict(venue="aster", leverage=2.0, short_enabled=True, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl, time_stop=ts, vol_target=vt, vol_window=vw)
    kw["funding_override"]=FUND if fund is None else fund
    if fee is not None:
        kw["fee_override"]=fee
    bt=MemeBacktest(**kw)
    signal=torch.sigmoid(sig)
    is_safe=(raw["liquidity"]>bt.min_liq).float()
    lp=(signal>bt.long_th).float()*is_safe
    sp=(signal<bt.short_th).float()*is_safe
    mask=quantile_mask_long(sig,q)
    if mask is not None:
        lp=lp*mask
    lp,sp=bt._apply_cooldown(lp,sp)
    lp,sp=bt._apply_stops(lp,sp,rets_t)
    scale=bt._vol_scale(rets_t)
    lp = lp * scale
    sp = sp * scale
    if side=="long":
        sp=sp*0.0
    if side=="short":
        lp=lp*0.0
    lp=lp.roll(1,dims=1); lp[:,0]=0
    sp=sp.roll(1,dims=1); sp[:,0]=0
    turn=(lp - lp.roll(1,dims=1)).abs() + (sp - sp.roll(1,dims=1)).abs()
    turn_l=(lp - lp.roll(1,dims=1)).abs()
    turn_s=(sp - sp.roll(1,dims=1)).abs()
    tx=turn*(bt.base_fee+torch.clamp(bt.trade_size/(raw["liquidity"]+1e-9),0.0,0.05))
    gross=(lp-sp)*rets_t*bt.leverage
    fnd=(lp-sp)*bt.default_funding_rate*bt.leverage
    net=(gross - tx*bt.leverage - fnd)[0].tolist()
    pos=(lp-sp)[0].tolist()
    lp_list=lp[0].tolist()
    sp_list=sp[0].tolist()
    turn_list=turn[0].tolist()
    turn_l_list=turn_l[0].tolist()
    turn_s_list=turn_s[0].tolist()
    trades=sum(1 for t in range(len(pos)) if pos[t]!=0.0 and (t==0 or pos[t-1]==0.0))
    turnover=sum(turn_list)/len(turn_list) if turn_list else 0.0
    long_turnover=sum(turn_l_list)/len(turn_l_list) if turn_l_list else 0.0
    short_turnover=sum(turn_s_list)/len(turn_s_list) if turn_s_list else 0.0
    pos_rate=sum(1 for v in pos if v!=0)/len(pos) if pos else 0
    long_pos_rate=sum(1 for v in lp_list if v!=0)/len(lp_list) if lp_list else 0
    short_pos_rate=sum(1 for v in sp_list if v!=0)/len(sp_list) if sp_list else 0
    return {"net":net,"pos":pos,"lp":lp_list,"sp":sp_list,"trades":trades,"turnover":turnover,"long_turnover":long_turnover,"short_turnover":short_turnover,"pos_rate":pos_rate,"long_pos_rate":long_pos_rate,"short_pos_rate":short_pos_rate,"turn":turn_list}

def stats(ser, trades=0, turnover=0.0, pos_rate=0.0, extra=None):
    n=len(ser)
    mean=sum(ser)/n if n else 0
    var=sum((x-mean)**2 for x in ser)/max(n-1,1) if n>1 else 0
    sharpe=mean/math.sqrt(var)*math.sqrt(2190.0) if var>0 else 0.0
    cum=sum(ser)
    ann=cum/n*2190.0 if n else 0
    cs,peak,mdd=0.0,-1e18,0.0
    for x in ser:
        cs+=x
        peak=max(peak,cs)
        mdd=max(mdd,peak-cs)
    out={"sharpe":round(sharpe,3),"ann":round(ann,4),"mdd":round(mdd,4),"cum":round(cum,4),"n":n,"trades":trades,"turnover":round(turnover,6),"pos_rate":round(pos_rate,4)}
    if extra:
        out.update(extra)
    return out

def combo(nets, weights=None):
    m=min(len(s) for s in nets)
    k=len(nets)
    w=weights or [1.0/k]*k
    sw=sum(w)
    return [sum(nets[i][t]*w[i]/sw for i in range(k)) for t in range(m)]

def holding_dist(pos_list):
    # pos_list: combo position proxy? Actually use per-leg pos averaged? We compute per-leg then aggregate as持仓持续分布
    # For portfolio holding length, we combine per-leg pos to combo pos approximated by avg? Simpler: compute holding spans on combo pos = (lp-sp) averaged
    # But we have per-leg pos separately; here pos_list is combo pos (weighted avg of legs weighted equally)
    # We treat combo pos !=0 as holding
    spans=[]
    i=0
    n=len(pos_list)
    while i<n:
        if pos_list[i]!=0:
            j=i+1
            while j<n and pos_list[j]!=0:
                # need check if continuous; but time_stop caps at 24 so max should be 24 unless cooldown etc.
                # For combo, pos zero if both legs zero; gaps within combo count as break
                j+=1
            # need handle sign changes? pos !=0 but sign flip is new trade - split spans
            # So walk and split by sign
            k=i
            while k<j:
                sign = 1 if pos_list[k]>0 else -1 if pos_list[k]<0 else 0
                # find until sign change or gap
                l=k+1
                while l<j and pos_list[l]!=0 and ((pos_list[l]>0) == (sign>0)):
                    l+=1
                spans.append(l-k)
                k=l
            i=j
        else:
            i+=1
    total=len(spans)
    if total==0:
        return {"counts":{"1-3":0,"4-8":0,"9-24":0,"25+":0},"pct":{"1-3":0.0,"4-8":0.0,"9-24":0.0,"25+":0.0},"n_spans":0,"mean_bars":0.0,"max_bars":0,"spans":spans}
    counts={"1-3":0,"4-8":0,"9-24":0,"25+":0}
    for s in spans:
        if s<=3: counts["1-3"]+=1
        elif s<=8: counts["4-8"]+=1
        elif s<=24: counts["9-24"]+=1
        else: counts["25+"]+=1
    pct={k: round(v/total,4) for k,v in counts.items()}
    return {"counts":counts,"pct":pct,"n_spans":total,"mean_bars":round(sum(spans)/len(spans),2) if spans else 0.0,"max_bars":max(spans) if spans else 0,"spans":spans}

def per_leg_holding_dist(pos_list):
    spans=[]
    i=0; n=len(pos_list)
    while i<n:
        if pos_list[i]!=0:
            j=i+1
            while j<n and pos_list[j]!=0 and ((pos_list[j]>0)==(pos_list[i]>0)):
                j+=1
            spans.append(j-i)
            i=j
        else:
            i+=1
    total=len(spans)
    if total==0:
        return {"counts":{"1-3":0,"4-8":0,"9-24":0,"25+":0},"pct":{"1-3":0.0,"4-8":0.0,"9-24":0.0,"25+":0.0},"n_spans":0,"mean_bars":0.0,"max_bars":0}
    counts={"1-3":0,"4-8":0,"9-24":0,"25+":0}
    for s in spans:
        if s<=3: counts["1-3"]+=1
        elif s<=8: counts["4-8"]+=1
        elif s<=24: counts["9-24"]+=1
        else: counts["25+"]+=1
    pct={k: round(v/total,4) for k,v in counts.items()}
    return {"counts":counts,"pct":pct,"n_spans":total,"mean_bars":round(sum(spans)/len(spans),2),"max_bars":max(spans)}

def eval_combo(bars_dict, spec_dict, fee, fund, q=Q):
    legs=[]; metas=[]; pos_lists=[]
    for c in COINS:
        lth,sth,cd,sl,ts,side,vt,vw = spec_dict[c]
        raw,rt,sg = build_mats(bars_dict[c])
        res = leg_series_with_pos(raw,rt,sg,lth,sth,cd,sl,ts,side,vt,vw,fee,fund,q)
        legs.append(res["net"])
        pos_lists.append(res["pos"])
        metas.append(res)
    cb = combo(legs, [0.5,0.5])
    # combo pos as weighted avg position (for holding dist)
    combo_pos = [ (pos_lists[0][t]*0.5 + pos_lists[1][t]*0.5) for t in range(len(pos_lists[0])) ]
    # threshold combo pos !=0 counts
    avg_to = sum(m["turnover"] for m in metas)/len(metas)
    avg_pr = sum(m["pos_rate"] for m in metas)/len(metas)
    avg_lpr = sum(m["long_pos_rate"] for m in metas)/len(metas)
    avg_spr = sum(m["short_pos_rate"] for m in metas)/len(metas)
    avg_lto = sum(m["long_turnover"] for m in metas)/len(metas)
    avg_sto = sum(m["short_turnover"] for m in metas)/len(metas)
    s = stats(cb, trades=sum(m["trades"] for m in metas), turnover=avg_to, pos_rate=avg_pr, extra={"long_pos_rate":round(avg_lpr,4),"short_pos_rate":round(avg_spr,4),"long_turnover":round(avg_lto,6),"short_turnover":round(avg_sto,6),"trades_by":{COINS[i]:metas[i]["trades"] for i in range(len(COINS))}})
    s["combo_pos"]=combo_pos
    s["per_leg"]=metas
    return s, cb, combo_pos, metas

def fold_stats_from_cb(cb, n_fold=12):
    fold_n = len(cb)//n_fold
    folds=[]
    for fi in range(n_fold):
        a=fi*fold_n
        b=a+fold_n if fi<n_fold-1 else len(cb)
        seg=cb[a:b]
        folds.append(stats(seg))
    sharpes=[f["sharpe"] for f in folds]
    anns=[f["ann"] for f in folds]
    return {"folds_sharpe":[round(x,3) for x in sharpes],"folds_ann":[round(x,4) for x in anns],"mean_sharpe":round(sum(sharpes)/len(sharpes),3) if sharpes else 0,"mean_ann":round(sum(anns)/len(anns),4) if anns else 0, "folds":folds}

def main():
    logp = pathlib.Path("logs/e3.log")
    logp.parent.mkdir(parents=True, exist_ok=True)
    open(logp,"w").write("E3 start\n")
    def log(msg):
        print(msg, flush=True)
        with open(logp,"a") as f: f.write(msg+"\n")
    full={c: load_bars(c) for c in COINS}
    n=min(len(v) for v in full.values())
    log(f"full 4h bars n={n} FORMULA {FORMULA} COINS {COINS} fund {FUND} fee_base {FEE_BASE} Q {Q}")
    assert n==FULL_N, f"expected {FULL_N} got {n}"
    results={}
    verdicts={}
    worst_dist=None
    worst_key=None
    for cfg in CONFIGS:
        cid=cfg["id"]
        log(f"=== {cid} {cfg['label']} spec ETC{cfg['spec']['ETC']} TRX{cfg['spec']['TRX']} ===")
        # per segment at base fee
        seg_stats={}
        seg_hold={}
        combo_positions={}
        for segname,(a,b) in SEG_MAP.items():
            bars_seg={c: full[c][a:b] for c in COINS}
            s, cb, cpos, metas = eval_combo(bars_seg, cfg["spec"], fee=FEE_BASE, fund=FUND, q=Q)
            # holding dist for this segment
            hd = holding_dist(cpos)
            # also per-leg dist for diagnostics
            per_leg_hd = {COINS[i]: per_leg_holding_dist(metas[i]["pos"]) for i in range(len(COINS))}
            # strip combo_pos from s for json
            cpos_save = s.pop("combo_pos", None)
            per_leg_save = s.pop("per_leg", None)
            seg_stats[segname]=s
            seg_hold[segname]={"combo":hd,"per_leg":per_leg_hd}
            combo_positions[segname]=cpos  # keep for export
            log(f"  {segname} n={s['n']} sh={s['sharpe']} ann={s['ann']} mdd={s['mdd']} to={s['turnover']:.6f} lto={s['long_turnover']:.6f} sto={s['short_turnover']:.6f} pr={s['pos_rate']} lpr={s['long_pos_rate']} spr={s['short_pos_rate']} trades={s['trades']} hold mean={hd['mean_bars']} max={hd['max_bars']} dist={hd['counts']} pct={hd['pct']}")
            # track worst holding: highest 25+ pct on FULL
            if segname=="FULL":
                pct25 = hd["pct"]["25+"]
                if worst_dist is None or pct25 > worst_dist:
                    worst_dist=pct25
                    worst_key=(cid, segname, hd)
        # fee scan on FULL
        fee_curve=[]
        cb_full = None
        for fee in FEE_SCAN:
            bars_full={c: full[c] for c in COINS}
            s, cb, _, _ = eval_combo(bars_full, cfg["spec"], fee=fee, fund=FUND, q=Q)
            s.pop("combo_pos", None); s.pop("per_leg", None)
            fee_curve.append({"fee":fee,"sharpe":s["sharpe"],"ann":s["ann"],"mdd":s["mdd"],"cum":s["cum"],"turnover":s["turnover"]})
            log(f"  fee {fee:.4f} FULL sh={s['sharpe']} ann={s['ann']} mdd={s['mdd']} to={s['turnover']}")
            if fee==FEE_BASE:
                cb_full=cb
        # 12fold mean curves at each fee: mean sharpe/ann across 12 folds on FULL
        fold_fee_curves=[]
        for fee in FEE_SCAN:
            bars_full={c: full[c] for c in COINS}
            s, cb, _, _ = eval_combo(bars_full, cfg["spec"], fee=fee, fund=FUND, q=Q)
            fs = fold_stats_from_cb(cb, 12)
            fold_fee_curves.append({"fee":fee,"mean_sharpe":fs["mean_sharpe"],"mean_ann":fs["mean_ann"],"folds_sharpe":fs["folds_sharpe"],"folds_ann":fs["folds_ann"]})
            log(f"  12fold fee {fee:.4f} mean_sh={fs['mean_sharpe']} mean_ann={fs['mean_ann']}")
        # FULL sharpe at fee 0.0012, turnover, 25+ pct
        full_to = seg_stats["FULL"]["turnover"]
        full_25pct = seg_hold["FULL"]["combo"]["pct"]["25+"]
        # fee 0.0012 from fee_curve
        fee12_sh = next(x["sharpe"] for x in fee_curve if x["fee"]==0.0012)
        pass_fee = bool(fee12_sh > 1.5 and full_to < 0.16 and full_25pct < 0.10)
        verdicts[cid]={"fee12_sharpe":fee12_sh,"turnover":full_to,"pct25":full_25pct,"pass":pass_fee}
        log(f"  VERDICT {cid}: fee0.0012 sh={fee12_sh} to={full_to:.5f} 25+={full_25pct:.4f} => {'PASS' if pass_fee else 'FAIL'} (req sh>1.5 & to<0.16 & 25+<0.10)")
        results[cid]={"config":cfg,"segments":seg_stats,"holding":seg_hold,"fee_curve":fee_curve,"fold_fee_curve":fold_fee_curves,"verdict":verdicts[cid],"combo_positions_FULL":combo_positions["FULL"][:20]}  # sample

    out={
        "meta":{"formula":FORMULA,"coins":COINS,"full_n":FULL_N,"fold_n":FOLD_N,"fund":FUND,"fee_base":FEE_BASE,"fee_scan":FEE_SCAN,"segments":SEG_MAP,"Q":Q,"note":"E3 mirror run_ab/run_w1: quantile_mask_long abs top-q long-only, vol_scale clamp0.2-2 roll1 post-stops pre-roll, cooldown, stops time_stop24, tx fee+liquidity, funding, lev2"},
        "results": {k: {kk:vv for kk,vv in v.items() if kk!="combo_positions_FULL"} for k,v in results.items()},
        "verdicts": verdicts,
        "worst_holding":{"key":worst_key[0] if worst_key else None,"segment":worst_key[1] if worst_key else None,"dist":worst_key[2] if worst_key else None}
    }
    # also save full positions for one config? Already omitted large pos; include sample only
    pathlib.Path("results/backtest_E3.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"saved results/backtest_E3.json worst={worst_key} {worst_dist}")
    # export logs
    log("E3 done")

if __name__=="__main__":
    main()
