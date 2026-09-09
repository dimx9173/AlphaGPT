
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_core")) else ".")
# ensure proj root
import pathlib
proj = pathlib.Path(__file__).resolve().parent.parent if pathlib.Path(__file__).resolve().parent.name=="research" else pathlib.Path(".").resolve()
# fallback: try cwd
import os as _os
if not (pathlib.Path(_os.getcwd())/"model_core").exists():
    _os.chdir(str(proj))
else:
    _os.chdir(_os.getcwd())

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
WEIGHTS=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
SLICES={"H2":(6077,6570),"B":(6380,6580),"C":(6080,6580),"FULL":(0,6580)}
# Base definitions: (lth, sth, cd, sl, ts, vt, vw)
BASES={
  "Base-A": {"label":"Q1 (ETC 0.85/0.15/cd12 None ts0 + TRX 0.85/0.12/cd6 None ts0) q0.3 no vol",
             "ETC":(0.85,0.15,12,None,0,None,12),
             "TRX":(0.85,0.12,6,None,0,None,12),
             "q":0.3},
  "Base-B": {"label":"Y1b (ETC 0.88/0.12/cd18 None ts24 + TRX 0.85/0.12/cd6 sl0.05 ts24) q0.3 no vol",
             "ETC":(0.88,0.12,18,None,24,None,12),
             "TRX":(0.85,0.12,6,0.05,24,None,12),
             "q":0.3},
  "Base-C": {"label":"Y2 (Y1b + vt0.01 vw12) q0.3",
             "ETC":(0.88,0.12,18,None,24,0.01,12),
             "TRX":(0.85,0.12,6,0.05,24,0.01,12),
             "q":0.3},
  "Base-D": {"label":"Z1-best (Y1b + vt0.012 vw12) q0.3",
             "ETC":(0.88,0.12,18,None,24,0.012,12),
             "TRX":(0.85,0.12,6,0.05,24,0.012,12),
             "q":0.3},
}

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
    # also compute long/short breakdown if needed but not required
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
    k=len(ser_list)
    sw=sum(weights)
    return [sum(ser_list[i][t]*weights[i]/sw for i in range(k)) for t in range(m)]

def corr(a,b):
    n=min(len(a),len(b))
    if n<2: return 0.0
    ma=sum(a[:n])/n; mb=sum(b[:n])/n
    num=sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    da=sum((a[i]-ma)**2 for i in range(n)); db=sum((b[i]-mb)**2 for i in range(n))
    den=math.sqrt(da*db)
    return round(num/den,3) if den>1e-12 else 0.0

def main():
    import pathlib, os, json
    logp=pathlib.Path("logs/w2.log")
    logp.parent.mkdir(parents=True, exist_ok=True)
    def log(msg):
        print(msg,flush=True)
        with open(logp,"a") as f: f.write(msg+"\n")

    # clear log
    open(logp,"w").write("")
    log("W2 weight scan start: bases=%s weights=%s slices=%s formula=%s fee=%.4f/%.4f fund=%.4f"%(list(BASES.keys()),WEIGHTS,SLICES,FORMULA,BASE_FEE,FEE2X,FUND))
    # load bars for ETC/TRX
    full={c:load_bars(c) for c in ["ETC","TRX"]}
    n=min(len(b) for b in full.values())
    log("full 4h bars n=%d min(ETC,TRX)"%n)
    # build seg mats cache
    seg_mats={}
    for seg_name,(a,b) in SLICES.items():
        seg_mats[seg_name]={c:build_mats(full[c][a:b]) for c in ["ETC","TRX"]}
        log("seg %s [%d:%d] n=%d"%(seg_name,a,b,b-a))
    results=[]
    per_base_best={}
    for bname,bdef in BASES.items():
        log("== %s: %s =="%(bname,bdef["label"]))
        rows=[]
        for w in WEIGHTS:
            wETC=w; wTRX=1-w
            weights=[wETC,wTRX]
            # per coin leg_series for each segment
            # need per-coin params
            def eval_combo(seg_name, fee):
                legs,trs,tos=[],[],[]
                for c, wt in zip(["ETC","TRX"],weights):
                    raw,rt,sg=seg_mats[seg_name][c]
                    lth,sth,cd,sl,ts,vt,vw=bdef[c]
                    q=bdef["q"]
                    net,t,to=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=fee,q=q,vt=vt,vw=vw)
                    legs.append(net); trs.append(t); tos.append(to)
                cb=combo(legs,weights)
                s=stats(cb,trades=sum(trs),turnover=sum(tos)/len(tos))
                return s, legs
            row={"base":bname,"wETC":round(wETC,3),"wTRX":round(wTRX,3),"weights":[round(wETC,3),round(wTRX,3)]}
            # evaluate H2/B/C/FULL at BASE_FEE
            for seg_name in ["H2","B","C","FULL"]:
                s,_=eval_combo(seg_name,BASE_FEE)
                row[seg_name]=s
            # fee2x for B/C
            for seg_name in ["B","C"]:
                s,_=eval_combo(seg_name,FEE2X)
                row[seg_name+"_fee2x"]=s
                row["fee_decay_"+seg_name]=round(row[seg_name]["sharpe"]-s["sharpe"],3)
            # corr H2 (ETC_TRX)
            _,legsH2=eval_combo("H2",BASE_FEE)
            row["corr_H2"]=corr(legsH2[0],legsH2[1])
            _,legsB=eval_combo("B",BASE_FEE)
            row["corr_B"]=corr(legsB[0],legsB[1])
            _,legsC=eval_combo("C",BASE_FEE)
            row["corr_C"]=corr(legsC[0],legsC[1])
            _,legsF=eval_combo("FULL",BASE_FEE)
            row["corr_FULL"]=corr(legsF[0],legsF[1])
            rows.append(row)
            log("wETC=%.1f H2(sh=%.3f ann=%.4f dd=%.4f tr=%d to=%.4f) B(sh=%.3f ann=%.4f dd=%.4f tr=%d to=%.4f B2x=%.3f) C(sh=%.3f C2x=%.3f) FULL(sh=%.3f) corr_H2=%.3f"%(
                wETC,row["H2"]["sharpe"],row["H2"]["ann"],row["H2"]["mdd"],row["H2"]["trades"],row["H2"]["turnover"],
                row["B"]["sharpe"],row["B"]["ann"],row["B"]["mdd"],row["B"]["trades"],row["B"]["turnover"],row["B_fee2x"]["sharpe"],
                row["C"]["sharpe"],row["C_fee2x"]["sharpe"],row["FULL"]["sharpe"],row["corr_H2"]))
        results.extend(rows)
        # per-base best: argmax B+C and argmax H2+B+C
        # 50/50 ref
        ref=[r for r in rows if abs(r["wETC"]-0.5)<1e-9][0]
        best_BC=max(rows, key=lambda r: r["B"]["sharpe"]+r["C"]["sharpe"])
        best_HBC=max(rows, key=lambda r: r["H2"]["sharpe"]+r["B"]["sharpe"]+r["C"]["sharpe"])
        def gain(best,ref):
            return {"d_H2":round(best["H2"]["sharpe"]-ref["H2"]["sharpe"],3),
                    "d_B":round(best["B"]["sharpe"]-ref["B"]["sharpe"],3),
                    "d_C":round(best["C"]["sharpe"]-ref["C"]["sharpe"],3),
                    "d_sumBC":round((best["B"]["sharpe"]+best["C"]["sharpe"])-(ref["B"]["sharpe"]+ref["C"]["sharpe"]),3),
                    "d_sumHBC":round((best["H2"]["sharpe"]+best["B"]["sharpe"]+best["C"]["sharpe"])-(ref["H2"]["sharpe"]+ref["B"]["sharpe"]+ref["C"]["sharpe"]),3)}
        per_base_best[bname]={
            "ref_50_50":{"wETC":0.5,"H2":ref["H2"],"B":ref["B"],"C":ref["C"],"FULL":ref["FULL"],"B_fee2x":ref["B_fee2x"],"C_fee2x":ref["C_fee2x"]},
            "best_BC":{"wETC":best_BC["wETC"],"H2":best_BC["H2"],"B":best_BC["B"],"C":best_BC["C"],"FULL":best_BC["FULL"],"B_fee2x":best_BC["B_fee2x"],"C_fee2x":best_BC["C_fee2x"],"gain_vs_50":gain(best_BC,ref)},
            "best_HBC":{"wETC":best_HBC["wETC"],"H2":best_HBC["H2"],"B":best_HBC["B"],"C":best_HBC["C"],"FULL":best_HBC["FULL"],"B_fee2x":best_HBC["B_fee2x"],"C_fee2x":best_HBC["C_fee2x"],"gain_vs_50":gain(best_HBC,ref)},
        }
        log(">> %s best_BC wETC=%.1f B+C=%.3f gain_vs50 d_BC=%.3f"%(bname,best_BC["wETC"],best_BC["B"]["sharpe"]+best_BC["C"]["sharpe"],gain(best_BC,ref)["d_sumBC"]))
        log(">> %s best_HBC wETC=%.1f H2+B+C=%.3f gain_vs50 d_HBC=%.3f"%(bname,best_HBC["wETC"],best_HBC["H2"]["sharpe"]+best_HBC["B"]["sharpe"]+best_HBC["C"]["sharpe"],gain(best_HBC,ref)["d_sumHBC"]))
        log(">> %s ref 50/50 H2=%.3f B=%.3f C=%.3f"%(bname,ref["H2"]["sharpe"],ref["B"]["sharpe"],ref["C"]["sharpe"]))

    # cross-base conclusion
    # weight depends on cd/vol/ts/sth -> check monotonic
    # Build narrative
    # For each base, slope: correlation of wETC vs H2/B/C sharpe
    import math as _m
    cross_lines=[]
    for bname in BASES:
        rows=[r for r in results if r["base"]==bname]
        # pearson w vs sharpe
        def pearson(xs,ys):
            n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
            num=sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
            dx=sum((x-mx)**2 for x in xs); dy=sum((y-my)**2 for y in ys)
            den=_m.sqrt(dx*dy)
            return round(num/den,3) if den>1e-12 else 0.0
        ws=[r["wETC"] for r in rows]
        for seg in ["H2","B","C","FULL"]:
            cross_lines.append("%s %s corr(w,sharpe)=%.3f"%(bname,seg,pearson(ws,[r[seg]["sharpe"] for r in rows])))
    for l in cross_lines: log(l)

    # Recommendation logic: if gains vs 50/50 are small (<0.3) for most bases, recommend 50/50; else per-base switching.
    rec=[]
    for bname in BASES:
        info=per_base_best[bname]
        gBC=info["best_BC"]["gain_vs_50"]["d_sumBC"]
        gHBC=info["best_HBC"]["gain_vs_50"]["d_sumHBC"]
        # threshold 0.5 for meaningful (roughly 0.25 per slice)
        meaningful = abs(gBC)>0.6 or abs(gHBC)>0.6
        rec.append((bname,info["best_BC"]["wETC"],info["best_HBC"]["wETC"],gBC,gHBC,meaningful))
        log("REC %s best_BC w=%.1f gainBC=%.3f best_HBC w=%.1f gainHBC=%.3f meaningful=%s"%(bname,info["best_BC"]["wETC"],gBC,info["best_HBC"]["wETC"],gHBC,meaningful))

    # Overall verdict
    # Weight depends on cd/vol/ts/sth ?
    # Base-A (no vol, cd12/no sth) -> expect TRX-heavy low w; Base-C/D with vol -> expect ETC-heavy?
    # We report observed
    out={
        "config":{"formula":FORMULA,"weights":WEIGHTS,"slices":{k:list(v) for k,v in SLICES.items()},"fee":BASE_FEE,"fee2x":FEE2X,"fund":FUND,"bases":{k:{"label":v["label"],"ETC":list(v["ETC"]),"TRX":list(v["TRX"]),"q":v["q"]} for k,v in BASES.items()}},
        "rows":results,
        "per_base_best":per_base_best,
        "cross_corr_w_sharpe":cross_lines,
        "recommendation_table":[ {"base":r[0],"best_BC_wETC":r[1],"best_HBC_wETC":r[2],"gain_BC_vs50":r[3],"gain_HBC_vs50":r[4],"meaningful":r[5]} for r in rec],
    }
    # narrative conclusion
    # Check if w optimum flips
    wBC_list=[per_base_best[b]["best_BC"]["wETC"] for b in BASES]
    wHBC_list=[per_base_best[b]["best_HBC"]["wETC"] for b in BASES]
    flips = max(wBC_list)-min(wBC_list) >=0.3 or max(wHBC_list)-min(wHBC_list) >=0.3
    overall_gain_BC = sum(abs(per_base_best[b]["best_BC"]["gain_vs_50"]["d_sumBC"]) for b in BASES)/len(BASES)
    overall_gain_HBC = sum(abs(per_base_best[b]["best_HBC"]["gain_vs_50"]["d_sumHBC"]) for b in BASES)/len(BASES)
    if flips and overall_gain_BC>0.6:
        verdict="權重最優翻轉確實存在且增益>0.6，建議依基線切換而非固定50/50；但需注意成交與費衰減"
    elif flips and overall_gain_BC<=0.6:
        verdict="雖w最優翻轉但對B+C增益小(均值%.2f)，固定50/50穩健；若追求極值可按基線微調"%overall_gain_BC
    else:
        verdict="權重跨基線穩定未翻轉，推薦固定50/50"
    # enrich with weight dependence on cd/vol/ts
    # compute details: base with vol tends to push w higher/lower?
    out["verdict"]=verdict
    out["flips"]={"wBC_range":round(max(wBC_list)-min(wBC_list),2),"wHBC_range":round(max(wHBC_list)-min(wHBC_list),2),"wBC_list":wBC_list,"wHBC_list":wHBC_list,"mean_abs_gain_BC":round(overall_gain_BC,3),"mean_abs_gain_HBC":round(overall_gain_HBC,3)}
    out["notes"]={
        "contradiction":"X3(Q1 no vol) TRX-heavy vs Z3(Y2 vol) ETC-heavy contradicted; W2掃描4基線驗證是否基線依賴",
        "wt_dependence":"權重依賴 cd/vol/ts/sth：Base-A vs Base-C/D 比較 vol 引入與 cd18/ sth差異是否翻轉w最優"
    }
    pathlib.Path("results/backtest_W2_weight.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    log("W2 done -> results/backtest_W2_weight.json verdict: "+verdict)
    log(json.dumps(out["per_base_best"], ensure_ascii=False, indent=2)[:4000])
    return out

if __name__=="__main__":
    main()
