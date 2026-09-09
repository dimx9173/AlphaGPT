"""AC weight & diversification re-probe.
Locks: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], Y1b+Z1 vol optimal
  Z1: ETC 0.88/0.12/cd18/None/ts24 + TRX 0.85/0.12/cd6/0.05/ts24, vt0.012/w12, q0.3, 50/50, aster 2x fund0.0005 fee0.0004
  Y1b: same without vol

AC1: weight fine sweep wETC in [0.2,0.3,0.4,0.5,0.6,0.7,0.8] x 2 bases =14 rows. Each H2/B/C/FULL/12fold + fee2x B/C + corr.
AC2: risk-parity contrast: Z1 base compute H2 & FULL realized vol (std per-bar net) => w_rp = (1/vol_TRX)/(1/vol_ETC+1/vol_TRX) ??? actually wETC_rp = inv_vol_ETC / (inv_vol_ETC+inv_vol_TRX). Compare vs 50/50 (2 rows contrast).
AC3: third leg re-probe on Z1 base: candidates [AVAX,SHIB,DOGE,BTC,SOL,BCH] each 1 row weight 50/30/20 + another set 60/30/10 =>12 rows; each H2/B/C/FULL+fee2x+turnover; beats_both = H2 and FULL sharpe > dual.

Total 28 rows. Engine mirrors run_w2.py / run_z3.py.
Slices: H2(6077,6570) B(6380,6580) C(6080,6580) FULL(0,6580). For 12-fold: split FULL 6580 into 12 folds ~548 each.
"""
import os, sys
proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj not in sys.path:
    sys.path.insert(0, proj)
os.chdir(proj)
import csv, json, math, pathlib
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA=[3,2,7,2,7,11,15,4,4,6,6,10]
BASE_FEE=0.0004
FEE2X=0.0008
FUND=0.0005
BARS_PER_YEAR=2190.0
WEIGHTS_AC=[0.2,0.3,0.4,0.5,0.6,0.7,0.8]
SLICES={"H2":(6077,6570),"B":(6380,6580),"C":(6080,6580),"FULL":(0,6580)}
THIRD_CANDIDATES=["AVAX","SHIB","DOGE","BTC","SOL","BCH"]
THIRD_MIXES_AC={"50_30_20":[0.5,0.3,0.2],"60_30_10":[0.6,0.3,0.1]}

BASES_AC={
  "Z1": {"label":"Z1 (Y1b + vt0.012 vw12) q0.3 50/50",
         "ETC":(0.88,0.12,18,None,24,0.012,12),
         "TRX":(0.85,0.12,6,0.05,24,0.012,12),
         "q":0.3},
  "Y1b": {"label":"Y1b (ETC 0.88/0.12/cd18 None ts24 + TRX 0.85/0.12/cd6 sl0.05 ts24) q0.3 no vol",
         "ETC":(0.88,0.12,18,None,24,None,12),
         "TRX":(0.85,0.12,6,0.05,24,None,12),
         "q":0.3},
}
# third leg params reuse TRX spec
P_THIRD=(0.85,0.12,6,0.05,24,0.012,12)

def load_bars(coin):
    rows=list(csv.DictReader(open("data/data_15m_3y/"+coin+".csv")))
    bars=[]
    for i in range(0,len(rows),16):
        blk=rows[i:i+16]
        if len(blk)<16: break
        bars.append((float(blk[0]["open"]), max(float(x["high"]) for x in blk), min(float(x["low"]) for x in blk), float(blk[-1]["close"]), sum(float(x["volume"]) for x in blk)))
    return bars

def build_mats(bars):
    n=len(bars)
    raw={"open": torch.tensor([[b[0] for b in bars]]),
         "high": torch.tensor([[b[1] for b in bars]]),
         "low": torch.tensor([[b[2] for b in bars]]),
         "close": torch.tensor([[b[3] for b in bars]]),
         "volume": torch.tensor([[b[4] for b in bars]]),
         "liquidity": torch.full((1,n),1e7),
         "fdv": torch.full((1,n),1e8)}
    sig=StackVM().execute(FORMULA, FeatureEngineer.compute_features(raw))
    rets=[(bars[i+1][3]-bars[i][3])/bars[i][3] for i in range(n-1)]+[0.0]
    return raw, torch.tensor([rets]), sig

def quantile_mask_long(sig,q):
    if q is None: return None
    a=sig.detach().float().abs().reshape(-1)
    k=max(1,int(len(a)*float(q)))
    thr=torch.topk(a,k).values.min()
    return (sig.detach().float().abs()>=thr).float()

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, ts, fee=None, fund=None, q=0.3, vt=None, vw=12):
    kw=dict(venue="aster", leverage=2.0, short_enabled=True, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=BARS_PER_YEAR, stop_loss=sl, time_stop=ts, vol_target=vt, vol_window=vw)
    kw["funding_override"]=FUND if fund is None else fund
    if fee is not None: kw["fee_override"]=fee
    bt=MemeBacktest(**kw)
    signal=torch.sigmoid(sig)
    is_safe=(raw["liquidity"]>bt.min_liq).float()
    lp=(signal>bt.long_th).float()*is_safe
    sp=(signal<bt.short_th).float()*is_safe
    mask=quantile_mask_long(sig,q)
    if mask is not None: lp=lp*mask
    lp,sp=bt._apply_cooldown(lp,sp)
    lp,sp=bt._apply_stops(lp,sp,rets_t)
    scale=bt._vol_scale(rets_t)
    lp,sp=lp*scale, sp*scale
    lp=lp.roll(1,dims=1); lp[:,0]=0
    sp=sp.roll(1,dims=1); sp[:,0]=0
    turn=(lp-lp.roll(1,dims=1)).abs()+(sp-sp.roll(1,dims=1)).abs()
    tx=turn*(bt.base_fee + torch.clamp(bt.trade_size/(raw["liquidity"]+1e-9),0.0,0.05))
    gross=(lp-sp)*rets_t*bt.leverage
    fnd=(lp-sp)*bt.default_funding_rate*bt.leverage
    net=(gross - tx*bt.leverage - fnd)[0].tolist()
    turnl=turn[0].tolist()
    pos=(lp-sp)[0].tolist()
    trades=sum(1 for t in range(len(pos)) if pos[t]!=0.0 and (t==0 or pos[t-1]==0.0))
    turnover=sum(turnl)/len(turnl)
    return net,trades,turnover

def stats(ser,trades=0,turnover=0.0):
    n=len(ser)
    mean=sum(ser)/n
    var=sum((x-mean)**2 for x in ser)/max(n-1,1)
    sharpe=mean/math.sqrt(var)*math.sqrt(BARS_PER_YEAR) if var>0 else 0.0
    cum=sum(ser)
    ann=cum/n*BARS_PER_YEAR
    cs,peak,mdd=0.0,-1e18,0.0
    for x in ser:
        cs+=x; peak=max(peak,cs); mdd=max(mdd,peak-cs)
    return {"sharpe":round(sharpe,3),"ann":round(ann,4),"mdd":round(mdd,4),"cum":round(cum,4),"n":n,"trades":trades,"turnover":round(turnover,6)}

def combo(ser_list, weights):
    m=min(len(s) for s in ser_list)
    sw=sum(weights)
    return [sum(ser_list[i][t]*weights[i]/sw for i in range(len(ser_list))) for t in range(m)]

def corr(a,b):
    n=min(len(a),len(b))
    if n<2: return 0.0
    ma=sum(a[:n])/n; mb=sum(b[:n])/n
    num=sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    da=sum((a[i]-ma)**2 for i in range(n)); db=sum((b[i]-mb)**2 for i in range(n))
    den=math.sqrt(da*db)
    return round(num/den,3) if den>1e-12 else 0.0

def realized_vol(ser):
    n=len(ser)
    if n<2: return 0.0
    m=sum(ser)/n
    var=sum((x-m)**2 for x in ser)/max(n-1,1)
    return math.sqrt(var)

def main():
    logp=pathlib.Path("logs/ac.log")
    logp.parent.mkdir(parents=True, exist_ok=True)
    open(logp,"w").write("")
    def log(msg):
        print(msg,flush=True)
        with open(logp,"a") as f: f.write(msg+"\n")
    log("AC start: formula=%s fee=%.4f fund=%.4f weights=%s slices=%s"%(FORMULA,BASE_FEE,FUND,WEIGHTS_AC,SLICES))
    # Load all needed coins
    all_coins=list(set(["ETC","TRX"]+THIRD_CANDIDATES))
    full={c:load_bars(c) for c in all_coins}
    n=min(len(b) for b in full.values())
    log("full 4h bars n=%d min across %s"% (n, all_coins))
    for c in all_coins:
        log("  %s n=%d"%(c, len(full[c])))
    # Build mats per slice for ETC/TRX and third candidates
    seg_mats={}
    for seg_name,(a,b) in SLICES.items():
        seg_mats[seg_name]={c:build_mats(full[c][a:b]) for c in all_coins}
        log("seg %s [%d:%d] n=%d"%(seg_name,a,b,b-a))
    # also build FULL fold splits for 12-fold
    fold_n = 6580//12  # ~548
    folds=[(i*fold_n, (i+1)*fold_n if i<11 else 6580) for i in range(12)]
    log("12-fold splits: %s fold_n~%d"%(folds, fold_n))
    # Prebuild leg series cache per base/coin/slice/fee for speed? We'll compute on fly.

    results_ac1=[]
    results_ac2={}
    results_ac3=[]

    # ---------- AC1 weight fine sweep ----------
    log("== AC1 weight fine sweep ==")
    for bname in ["Z1","Y1b"]:
        bdef=BASES_AC[bname]
        log("-- base %s: %s --"%(bname,bdef["label"]))
        # cache 12-fold per weight? We'll compute per row.
        for w in WEIGHTS_AC:
            wETC=w; wTRX=1-w
            weights=[wETC,wTRX]
            row={"task":"AC1","base":bname,"wETC":round(wETC,3),"wTRX":round(wTRX,3),"weights":[round(wETC,3),round(wTRX,3)]}
            # evaluate core slices at base fee
            for seg_name in ["H2","B","C","FULL"]:
                legs,trs,tos=[],[],[]
                for c,wt in zip(["ETC","TRX"],weights):
                    raw,rt,sg=seg_mats[seg_name][c]
                    lth,sth,cd,sl,ts,vt,vw=bdef[c]
                    q=bdef["q"]
                    net,t,to=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=BASE_FEE,q=q,vt=vt,vw=vw)
                    legs.append(net); trs.append(t); tos.append(to)
                cb=combo(legs,weights)
                s=stats(cb,trades=sum(trs),turnover=sum(tos)/len(tos))
                row[seg_name]=s
                # for FULL also store corr
            # fee2x B/C
            for seg_name in ["B","C"]:
                legs,trs,tos=[],[],[]
                for c,wt in zip(["ETC","TRX"],weights):
                    raw,rt,sg=seg_mats[seg_name][c]
                    lth,sth,cd,sl,ts,vt,vw=bdef[c]
                    q=bdef["q"]
                    net,t,to=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=FEE2X,q=q,vt=vt,vw=vw)
                    legs.append(net); trs.append(t); tos.append(to)
                cb=combo(legs,weights)
                s=stats(cb,trades=sum(trs),turnover=sum(tos)/len(tos))
                row[seg_name+"_fee2x"]=s
                row["fee_decay_"+seg_name]=round(row[seg_name]["sharpe"]-s["sharpe"],3)
            # corr per slice
            for seg_name in ["H2","B","C","FULL"]:
                legs=[]
                for c in ["ETC","TRX"]:
                    raw,rt,sg=seg_mats[seg_name][c]
                    lth,sth,cd,sl,ts,vt,vw=bdef[c]
                    q=bdef["q"]
                    net,_,_=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=BASE_FEE,q=q,vt=vt,vw=vw)
                    legs.append(net)
                row["corr_"+seg_name]=corr(legs[0],legs[1])
            # 12-fold on FULL (base fee): for each fold compute sharpe
            fold_sharpes=[]
            for fa,fb in folds:
                legs=[]
                trs=[]; tos=[]
                for c in ["ETC","TRX"]:
                    raw,rt,sg=build_mats(full[c][fa:fb])
                    lth,sth,cd,sl,ts,vt,vw=bdef[c]
                    q=bdef["q"]
                    net,t,to=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=BASE_FEE,q=q,vt=vt,vw=vw)
                    legs.append(net)
                # need to ensure legs length trimmed? build_mats n=fold length
                cb=combo(legs,weights)
                s=stats(cb)
                fold_sharpes.append(s["sharpe"])
            row["fold12_sharpes"]=[round(x,3) for x in fold_sharpes]
            row["fold12_mean"]=round(sum(fold_sharpes)/len(fold_sharpes),3)
            row["fold12_median"]=round(sorted(fold_sharpes)[len(fold_sharpes)//2],3)
            row["fold12_pos_rate"]=round(sum(1 for x in fold_sharpes if x>0)/len(fold_sharpes),3)
            # log line
            log("AC1 %s wETC=%.1f H2(sh=%.3f ann=%.4f mdd=%.4f) B=%.3f(B2x=%.3f) C=%.3f(C2x=%.3f) FULL=%.3f corr_H2=%.3f fold12_mean=%.3f med=%.3f pos=%.2f"%(
                bname,wETC,row["H2"]["sharpe"],row["H2"]["ann"],row["H2"]["mdd"],
                row["B"]["sharpe"],row["B_fee2x"]["sharpe"],row["C"]["sharpe"],row["C_fee2x"]["sharpe"],row["FULL"]["sharpe"],row["corr_H2"],row["fold12_mean"],row["fold12_median"],row["fold12_pos_rate"]))
            results_ac1.append(row)

    # Per-base best
    per_base_best={}
    for bname in ["Z1","Y1b"]:
        rows=[r for r in results_ac1 if r["base"]==bname]
        ref=[r for r in rows if abs(r["wETC"]-0.5)<1e-9][0] if any(abs(r["wETC"]-0.5)<1e-9 for r in rows) else rows[3]
        best_BC=max(rows, key=lambda r: r["B"]["sharpe"]+r["C"]["sharpe"])
        best_H2=max(rows, key=lambda r: r["H2"]["sharpe"])
        best_FULL=max(rows, key=lambda r: r["FULL"]["sharpe"])
        best_fold=max(rows, key=lambda r: r["fold12_mean"])
        per_base_best[bname]={
            "ref_50_50":{"wETC":0.5,"H2":ref["H2"],"B":ref["B"],"C":ref["C"],"FULL":ref["FULL"],"fold12_mean":ref["fold12_mean"]},
            "best_BC":{"wETC":best_BC["wETC"],"H2":best_BC["H2"],"B":best_BC["B"],"C":best_BC["C"],"FULL":best_BC["FULL"],"fold12_mean":best_BC["fold12_mean"],"gain_BC_vs50":round((best_BC["B"]["sharpe"]+best_BC["C"]["sharpe"])-(ref["B"]["sharpe"]+ref["C"]["sharpe"]),3)},
            "best_H2":{"wETC":best_H2["wETC"],"H2":best_H2["H2"]},
            "best_FULL":{"wETC":best_FULL["wETC"],"FULL":best_FULL["FULL"]},
            "best_fold12":{"wETC":best_fold["wETC"],"fold12_mean":best_fold["fold12_mean"]},
        }
        log(">> %s ref50 H2=%.3f B=%.3f C=%.3f FULL=%.3f fold12_mean=%.3f"%(bname,ref["H2"]["sharpe"],ref["B"]["sharpe"],ref["C"]["sharpe"],ref["FULL"]["sharpe"],ref["fold12_mean"]))
        log(">> %s best_BC w=%.1f B+C=%.3f gain=%.3f"%(bname,best_BC["wETC"],best_BC["B"]["sharpe"]+best_BC["C"]["sharpe"],per_base_best[bname]["best_BC"]["gain_BC_vs50"]))
        log(">> %s best_H2 w=%.1f sh=%.3f"%(bname,best_H2["wETC"],best_H2["H2"]["sharpe"]))
        log(">> %s best_FULL w=%.1f sh=%.3f"%(bname,best_FULL["wETC"],best_FULL["FULL"]["sharpe"]))

    # ---------- AC2 risk parity ----------
    log("== AC2 risk parity contrast (Z1) ==")
    bdef=BASES_AC["Z1"]
    # For H2 and FULL, compute per-coin realized vol (std of per-bar net at base fee)
    rp_results=[]
    for seg_name in ["H2","FULL"]:
        # per coin leg net
        legs={}
        vols={}
        for c in ["ETC","TRX"]:
            raw,rt,sg=seg_mats[seg_name][c]
            lth,sth,cd,sl,ts,vt,vw=bdef[c]
            q=bdef["q"]
            net,_,_=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=BASE_FEE,q=q,vt=vt,vw=vw)
            legs[c]=net
            vols[c]=realized_vol(net)
            log("AC2 %s %s vol=%.6f n=%d"%(seg_name,c,vols[c],len(net)))
        inv_etc=1/vols["ETC"] if vols["ETC"]>1e-12 else 0
        inv_trx=1/vols["TRX"] if vols["TRX"]>1e-12 else 0
        wETC_rp= inv_etc/(inv_etc+inv_trx) if (inv_etc+inv_trx)>0 else 0.5
        wTRX_rp=1-wETC_rp
        log("AC2 %s rp wETC=%.4f wTRX=%.4f (vol ETC=%.6f TRX=%.6f)"%(seg_name,wETC_rp,wTRX_rp,vols["ETC"],vols["TRX"]))
        # evaluate combo at rp vs 50/50
        for label,weights in [("rp",[wETC_rp,wTRX_rp]),("50_50",[0.5,0.5])]:
            for eval_seg in ["H2","B","C","FULL"]:
                legs2=[]
                trs=[]; tos=[]
                for c in ["ETC","TRX"]:
                    raw,rt,sg=seg_mats[eval_seg][c]
                    lth,sth,cd,sl,ts,vt,vw=bdef[c]
                    q=bdef["q"]
                    net,t,to=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=BASE_FEE,q=q,vt=vt,vw=vw)
                    legs2.append(net); trs.append(t); tos.append(to)
                cb=combo(legs2,weights)
                s=stats(cb,trades=sum(trs),turnover=sum(tos)/len(tos))
                rp_results.append({"rp_seg":seg_name,"label":label,"wETC":round(weights[0],4),"wTRX":round(weights[1],4),"eval_seg":eval_seg,"stats":s})
                if seg_name==eval_seg:
                    # also log fee2x for B/C if eval_seg is B/C
                    pass
            # also fee2x B/C
            # log summary for this rp_seg+label
        # quick compare
        # We'll collect later for report
        results_ac2[seg_name]={"vol_ETC":round(vols["ETC"],6),"vol_TRX":round(vols["TRX"],6),"wETC_rp":round(wETC_rp,4),"wTRX_rp":round(wTRX_rp,4),"inv_vol_ETC":round(inv_etc,4),"inv_vol_TRX":round(inv_trx,4)}
        log("AC2 %s rp weights: %.4f / %.4f"%(seg_name,wETC_rp,wTRX_rp))
    # Add detailed rows into results_ac2
    results_ac2["detail_rows"]=rp_results
    # Summarize rp vs 50/50 differences for table
    for seg_name in ["H2","FULL"]:
        rp_w=results_ac2[seg_name]["wETC_rp"]
        log("AC2 %s rp vs 50/50 comparison:"%seg_name)
        for eval_seg in ["H2","B","C","FULL"]:
            rp_s=[r for r in rp_results if r["rp_seg"]==seg_name and r["label"]=="rp" and r["eval_seg"]==eval_seg][0]["stats"]
            eq_s=[r for r in rp_results if r["rp_seg"]==seg_name and r["label"]=="50_50" and r["eval_seg"]==eval_seg][0]["stats"]
            log("  %s: rp sh=%.3f ann=%.4f vs 50/50 sh=%.3f ann=%.4f diff=%.3f"%(
                eval_seg, rp_s["sharpe"], rp_s["ann"], eq_s["sharpe"], eq_s["ann"], round(rp_s["sharpe"]-eq_s["sharpe"],3)))

    # ---------- AC3 third leg re-probe ----------
    log("== AC3 third leg re-probe (Z1 base, 50/30/20 and 60/30/10) ==")
    # Need dual Z1 baseline for beats_both reference: use 50/50 from AC1 Z1 row w=0.5
    dual_rows=[r for r in results_ac1 if r["base"]=="Z1" and abs(r["wETC"]-0.5)<1e-9]
    dual_ref=dual_rows[0] if dual_rows else None
    if dual_ref:
        log("dual Z1 50/50 ref: H2=%.3f B=%.3f C=%.3f FULL=%.3f"%(dual_ref["H2"]["sharpe"],dual_ref["B"]["sharpe"],dual_ref["C"]["sharpe"],dual_ref["FULL"]["sharpe"]))
    for third in THIRD_CANDIDATES:
        for mix_name,weights in THIRD_MIXES_AC.items():
            # weights order ETC/TRX/third
            coins=["ETC","TRX",third]
            row={"task":"AC3","base":"Z1","third":third,"mix":mix_name,"weights":[round(x,4) for x in weights],"coins":coins}
            for seg_name in ["H2","B","C","FULL"]:
                legs,trs,tos=[],[],[]
                for c,wt in zip(coins,weights):
                    raw,rt,sg=seg_mats[seg_name][c]
                    if c=="ETC":
                        lth,sth,cd,sl,ts,vt,vw=BASES_AC["Z1"]["ETC"]
                        q=BASES_AC["Z1"]["q"]
                    elif c=="TRX":
                        lth,sth,cd,sl,ts,vt,vw=BASES_AC["Z1"]["TRX"]
                        q=BASES_AC["Z1"]["q"]
                    else:
                        lth,sth,cd,sl,ts,vt,vw=P_THIRD
                        q=BASES_AC["Z1"]["q"]
                    net,t,to=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=BASE_FEE,q=q,vt=vt,vw=vw)
                    legs.append(net); trs.append(t); tos.append(to)
                cb=combo(legs,weights)
                s=stats(cb,trades=sum(trs),turnover=sum(tos)/len(tos))
                row[seg_name]=s
            # fee2x B/C
            for seg_name in ["B","C"]:
                legs,trs,tos=[],[],[]
                for c,wt in zip(coins,weights):
                    raw,rt,sg=seg_mats[seg_name][c]
                    if c=="ETC":
                        lth,sth,cd,sl,ts,vt,vw=BASES_AC["Z1"]["ETC"]
                        q=BASES_AC["Z1"]["q"]
                    elif c=="TRX":
                        lth,sth,cd,sl,ts,vt,vw=BASES_AC["Z1"]["TRX"]
                        q=BASES_AC["Z1"]["q"]
                    else:
                        lth,sth,cd,sl,ts,vt,vw=P_THIRD
                        q=BASES_AC["Z1"]["q"]
                    net,t,to=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=BASE_FEE,q=q,vt=vt,vw=vw)
                    # override fee
                    # need fee2x: recompute with FEE2X
                    net,t,to=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=FEE2X,q=q,vt=vt,vw=vw)
                    legs.append(net); trs.append(t); tos.append(to)
                cb=combo(legs,weights)
                s=stats(cb,trades=sum(trs),turnover=sum(tos)/len(tos))
                row[seg_name+"_fee2x"]=s
                row["fee_decay_"+seg_name]=round(row[seg_name]["sharpe"]-s["sharpe"],3)
            # turnover already in stats; also store corr ETC_TRX
            for seg_name in ["H2","FULL"]:
                # corr of ETC vs TRX leg within triple (not involving third)
                legs=[]
                for c in ["ETC","TRX"]:
                    raw,rt,sg=seg_mats[seg_name][c]
                    if c=="ETC":
                        lth,sth,cd,sl,ts,vt,vw=BASES_AC["Z1"]["ETC"]
                    else:
                        lth,sth,cd,sl,ts,vt,vw=BASES_AC["Z1"]["TRX"]
                    net,_,_=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=BASE_FEE,q=BASES_AC["Z1"]["q"],vt=vt,vw=vw)
                    legs.append(net)
                row["corr_ETC_TRX_"+seg_name]=corr(legs[0],legs[1])
            # beats_both vs dual
            if dual_ref:
                beats_H2=row["H2"]["sharpe"]>dual_ref["H2"]["sharpe"]
                beats_FULL=row["FULL"]["sharpe"]>dual_ref["FULL"]["sharpe"]
                row["beats_H2"]=bool(beats_H2)
                row["beats_FULL"]=bool(beats_FULL)
                row["beats_both"]=bool(beats_H2 and beats_FULL)
                row["dual_H2"]=dual_ref["H2"]["sharpe"]
                row["dual_FULL"]=dual_ref["FULL"]["sharpe"]
                row["delta_H2"]=round(row["H2"]["sharpe"]-dual_ref["H2"]["sharpe"],3)
                row["delta_FULL"]=round(row["FULL"]["sharpe"]-dual_ref["FULL"]["sharpe"],3)
            else:
                row["beats_both"]=False
            results_ac3.append(row)
            log("AC3 %s %s H2(sh=%.3f d=%.3f) B=%.3f(B2x=%.3f) C=%.3f(C2x=%.3f) FULL(sh=%.3f d=%.3f) to_H2=%.4f beats_both=%s"%(
                third,mix_name,row["H2"]["sharpe"],row.get("delta_H2",0),row["B"]["sharpe"],row["B_fee2x"]["sharpe"],row["C"]["sharpe"],row["C_fee2x"]["sharpe"],row["FULL"]["sharpe"],row.get("delta_FULL",0),row["H2"]["turnover"],row["beats_both"]))

    # Summary for AC3
    n_beats=sum(1 for r in results_ac3 if r.get("beats_both"))
    log("AC3 beats_both count %d/12"%n_beats)
    if results_ac3:
        best_ac3_H2=max(results_ac3, key=lambda r: r["H2"]["sharpe"])
        best_ac3_FULL=max(results_ac3, key=lambda r: r["FULL"]["sharpe"])
        log("AC3 best H2: %s %s sh=%.3f"%(best_ac3_H2["third"],best_ac3_H2["mix"],best_ac3_H2["H2"]["sharpe"]))
        log("AC3 best FULL: %s %s sh=%.3f"%(best_ac3_FULL["third"],best_ac3_FULL["mix"],best_ac3_FULL["FULL"]["sharpe"]))

    # ---------- Save ----------
    out={
        "config":{"formula":FORMULA,"fee":BASE_FEE,"fee2x":FEE2X,"fund":FUND,"bases":BASES_AC,"weights_AC":WEIGHTS_AC,"slices":SLICES,"folds":folds,"third_candidates":THIRD_CANDIDATES,"third_mixes":THIRD_MIXES_AC,"p_third":list(P_THIRD)},
        "AC1_rows":results_ac1,
        "AC1_per_base_best":per_base_best,
        "AC2":results_ac2,
        "AC3_rows":results_ac3,
        "summary":{"AC1_n":len(results_ac1),"AC2_n":len(rp_results),"AC3_n":len(results_ac3),"total":len(results_ac1)+len(rp_results)+len(results_ac3),"n_beats_both":n_beats if results_ac3 else 0},
    }
    # Also build flattened rows for counting 28 per spec: 14 AC1 + 2 rp contrast (count as 2) +12 AC3 =28. We'll expose.
    pathlib.Path("results/backtest_AC.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    log("saved results/backtest_AC.json total=%d (AC1=%d AC2_detail=%d AC3=%d) beats_both=%d"%(out["summary"]["total"], len(results_ac1), len(rp_results), len(results_ac3), out["summary"]["n_beats_both"]))
    log("AC done")

if __name__=="__main__":
    main()
