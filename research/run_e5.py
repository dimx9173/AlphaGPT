"""E5 dynamic weights & volatility timing — v2 per-slice engine.
Mirrors run_ac.py: per-slice build_mats + leg_series (recomputes quantile/top-k per slice).
Each dynamic scheme recomputed on that slice's per-bar nets; turnover_dynamic per slice.
This makes STATIC exactly match AC Z1 50/50 numbers (H2 5.013, B 7.042 etc).
"""
import os, sys, csv, json, math, pathlib
proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj not in sys.path:
    sys.path.insert(0, proj)
os.chdir(proj)
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA=[3,2,7,2,7,11,15,4,4,6,6,10]
BASE_FEE=0.0004
FUND=0.0005
BARS_PER_YEAR=2190.0
SLICES={"H2":(6077,6570),"B":(6380,6580),"C":(6080,6580),"FULL":(0,6580)}
BASE_Z1={
  "ETC":(0.88,0.12,18,None,24,0.012,12),
  "TRX":(0.85,0.12,6,0.05,24,0.012,12),
  "q":0.3,
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
    sig=StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
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
    return net, trades, turnover, turnl

def stats(ser,trades=0,turnover=0.0):
    n=len(ser)
    if n==0:
        return {"sharpe":0.0,"ann":0.0,"mdd":0.0,"cum":0.0,"n":0,"trades":trades,"turnover":round(turnover,6)}
    mean=sum(ser)/n
    var=sum((x-mean)**2 for x in ser)/max(n-1,1)
    sharpe=mean/math.sqrt(var)*math.sqrt(BARS_PER_YEAR) if var>0 else 0.0
    cum=sum(ser)
    ann=cum/n*BARS_PER_YEAR
    cs,peak,mdd=0.0,-1e18,0.0
    for x in ser:
        cs+=x; peak=max(peak,cs); mdd=max(mdd,peak-cs)
    return {"sharpe":round(sharpe,3),"ann":round(ann,4),"mdd":round(mdd,4),"cum":round(cum,4),"n":n,"trades":trades,"turnover":round(turnover,6)}

def rolling_sharpe(ser, window):
    n=len(ser); out=[0.0]*n
    for t in range(window, n):
        w=ser[t-window:t]
        m=sum(w)/window
        var=sum((x-m)**2 for x in w)/max(window-1,1)
        s=m/math.sqrt(var)*math.sqrt(BARS_PER_YEAR) if var>1e-12 else 0.0
        out[t]=s
    return out

def rolling_vol(ser, window):
    n=len(ser); out=[1e-9]*n
    for t in range(window, n):
        w=ser[t-window:t]
        m=sum(w)/window
        var=sum((x-m)**2 for x in w)/max(window-1,1)
        out[t]=math.sqrt(var) if var>1e-12 else 1e-9
    first=None
    for t in range(n):
        if out[t]>1e-9+1e-12:
            first=out[t]; break
    if first is None: first=0.02
    for t in range(n):
        if out[t]<1e-8: out[t]=first
    return out

def rolling_cum(ser, window):
    n=len(ser); out=[0.0]*n
    cumsum=[0.0]*(n+1)
    for i in range(n): cumsum[i+1]=cumsum[i]+ser[i]
    for t in range(window, n):
        out[t]=cumsum[t]-cumsum[t-window]
    return out

def dynamic_weights(net_etc, net_trx, scheme, params=None):
    params=params or {}; n=len(net_etc)
    w_etc=[0.5]*n; w_trx=[0.5]*n
    if scheme=="A":
        win=params.get("win",200)
        se=rolling_sharpe(net_etc, win); st=rolling_sharpe(net_trx, win)
        for t in range(n):
            if t<win: w_etc[t]=0.5; w_trx[t]=0.5
            else:
                pe=max(se[t],0.0); pt=max(st[t],0.0); s=pe+pt
                if s<1e-12: w_etc[t]=0.5; w_trx[t]=0.5
                else: w_etc[t]=pe/s; w_trx[t]=pt/s
    elif scheme=="B":
        win=params.get("win",200); vol_win=params.get("vol_win",20)
        ve=rolling_vol(net_etc, vol_win); vt=rolling_vol(net_trx, vol_win)
        for t in range(n):
            if t<win: w_etc[t]=0.5; w_trx[t]=0.5
            else:
                ie=1.0/max(ve[t],1e-9); it=1.0/max(vt[t],1e-9); s=ie+it
                w_etc[t]=ie/s; w_trx[t]=it/s
    elif scheme=="C":
        win=params.get("win",200)
        se=rolling_sharpe(net_etc, win); st=rolling_sharpe(net_trx, win)
        for t in range(n):
            if t<win: w_etc[t]=0.5; w_trx[t]=0.5
            else:
                ne=se[t]<0; nt=st[t]<0
                if ne and nt: w_etc[t]=0.5; w_trx[t]=0.5
                elif ne: w_etc[t]=0.0; w_trx[t]=1.0
                elif nt: w_etc[t]=1.0; w_trx[t]=0.0
                else: w_etc[t]=0.5; w_trx[t]=0.5
    elif scheme=="D":
        win=params.get("win",100)
        ce=rolling_cum(net_etc, win); ct=rolling_cum(net_trx, win)
        for t in range(n):
            if t<win: w_etc[t]=0.5; w_trx[t]=0.5
            else:
                if ce[t] > ct[t]: w_etc[t]=0.6; w_trx[t]=0.4
                else: w_etc[t]=0.4; w_trx[t]=0.6
    else: raise ValueError(scheme)
    return w_etc, w_trx

def combo_dynamic(nets, w_list):
    n=len(nets[0])
    return [nets[0][t]*w_list[0][t] + nets[1][t]*w_list[1][t] for t in range(n)]

def turnover_dynamic(turn_etc, turn_trx, w_etc, w_trx):
    n=len(turn_etc)
    out=[0.0]*n
    for t in range(n):
        base=w_etc[t]*turn_etc[t] + w_trx[t]*turn_trx[t]
        dw=abs(w_etc[t]-w_etc[t-1]) if t>0 else 0.0
        out[t]=base+dw*0.5
    return sum(out)/len(out) if out else 0.0

def weight_flip_rate(w_etc, a, b, th=0.02):
    seg=w_etc[a:b] if b>a else w_etc
    if len(seg)<2: return 0.0
    flips=sum(1 for i in range(1,len(seg)) if abs(seg[i]-seg[i-1])>th)
    return round(flips/len(seg),4)

def main():
    logp=pathlib.Path("logs/e5.log")
    logp.parent.mkdir(parents=True, exist_ok=True)
    open(logp,"w").write("")
    def log(msg):
        print(msg,flush=True)
        with open(logp,"a") as f: f.write(msg+"\n")
    log("E5 v2 per-slice: FORMULA=%s fee=%.4f fund=%.4f" % (FORMULA, BASE_FEE, FUND))
    full_bars={"ETC": load_bars("ETC"), "TRX": load_bars("TRX")}
    n=min(len(b) for b in full_bars.values())
    log("full 4h bars n=%d ETC=%d TRX=%d" % (n, len(full_bars["ETC"]), len(full_bars["TRX"])))

    fold_n=6580//12
    folds=[(i*fold_n, (i+1)*fold_n if i<11 else 6580) for i in range(12)]
    log("folds %s" % folds)

    # Per-slice mats+legs
    slice_data={}
    for seg_name,(a,b) in SLICES.items():
        sd={}
        for c in ["ETC","TRX"]:
            raw,rt,sg=build_mats(full_bars[c][a:b])
            lth,sth,cd,sl,ts,vt,vw=BASE_Z1[c]; q=BASE_Z1["q"]
            net,tr,to,turnl=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=BASE_FEE,q=q,vt=vt,vw=vw)
            sd[c]=(net,tr,to,turnl)
            log("slice %s leg %s n=%d net_len=%d trades=%d to=%.6f mean=%.6f" % (seg_name,c,b-a,len(net),tr,to,sum(net)/len(net) if net else 0))
        slice_data[seg_name]=sd

    # Also build per-fold legs for 12fold dynamic: for each fold, recompute legs on that fold's bars
    fold_slice_data=[]
    for fa,fb in folds:
        fd={}
        for c in ["ETC","TRX"]:
            raw,rt,sg=build_mats(full_bars[c][fa:fb])
            lth,sth,cd,sl,ts,vt,vw=BASE_Z1[c]; q=BASE_Z1["q"]
            net,tr,to,turnl=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=BASE_FEE,q=q,vt=vt,vw=vw)
            fd[c]=(net,turnl)
        fold_slice_data.append(fd)

    scheme_defs={
      "STATIC_50_50": None,
      "DYN_A_sharpe200": ("A", {"win":200}),
      "DYN_B_invvol200": ("B", {"win":200,"vol_win":20}),
      "DYN_C_timing200": ("C", {"win":200}),
      "DYN_D_momo100": ("D", {"win":100}),
    }

    results={}
    for k,v in scheme_defs.items():
        row={"scheme":k}
        for seg_name,(a,b) in SLICES.items():
            net_etc,_,_,turn_etc=slice_data[seg_name]["ETC"]
            net_trx,_,_,turn_trx=slice_data[seg_name]["TRX"]
            nn=len(net_etc)
            if v is None:
                w_etc=[0.5]*nn; w_trx=[0.5]*nn
                cb=combo_dynamic([net_etc,net_trx],[w_etc,w_trx])
                # for static, use average turnover matching AC (avg of leg to)
                _,_,to_e,_=slice_data[seg_name]["ETC"]; _,_,to_t,_=slice_data[seg_name]["TRX"]
                to_seg=(to_e+to_t)/2
                # but store precise? Use avg
            else:
                sch,par=v
                # need window sanity: if nn < win, weights stay 0.5
                w_etc,w_trx=dynamic_weights(net_etc, net_trx, sch, par)
                cb=combo_dynamic([net_etc,net_trx],[w_etc,w_trx])
                to_seg=turnover_dynamic(turn_etc, turn_trx, w_etc, w_trx)
                row["wETC_mean_"+seg_name]=round(sum(w_etc)/max(len(w_etc),1),4)
                if len(w_etc)>1:
                    m=row["wETC_mean_"+seg_name]
                    var=sum((x-m)**2 for x in w_etc)/max(len(w_etc)-1,1)
                    row["wETC_std_"+seg_name]=round(math.sqrt(var),4)
                else:
                    row["wETC_std_"+seg_name]=0.0
                row["flip_rate_"+seg_name]=weight_flip_rate(w_etc,0,len(w_etc),th=0.02)
                row["weight_drift_"+seg_name]=round(sum(abs(w_etc[t]-w_etc[t-1]) for t in range(1,len(w_etc)))/max(len(w_etc),1),6)
            if v is None:
                # fill weight stats for static too
                row["wETC_mean_"+seg_name]=0.5
                row["wETC_std_"+seg_name]=0.0
                row["flip_rate_"+seg_name]=0.0
                row["weight_drift_"+seg_name]=0.0
                # cb already
            else:
                pass
            # stats
            s=stats(cb, trades=0, turnover=0.0)
            # override turnover with dynamic turnover or static avg
            if v is None:
                s["turnover"]=round(to_seg,6)
            else:
                s["turnover"]=round(to_seg,6)
            row[seg_name]=s
            if v is not None:
                # sample
                row["wETC_sample_"+seg_name]=[round(w_etc[i],3) for i in range(0,len(w_etc), max(1,len(w_etc)//6))][:7]
        # 12fold for this scheme: compute per-fold dynamic separately (each fold independent)
        fold_sharpes=[]
        for idx,(fa,fb) in enumerate(folds):
            net_etc,_=fold_slice_data[idx]["ETC"]; _,_=fold_slice_data[idx]["TRX"]
            # unpack correctly
            net_etc=fold_slice_data[idx]["ETC"][0]; turn_etc=fold_slice_data[idx]["ETC"][1]
            net_trx=fold_slice_data[idx]["TRX"][0]; turn_trx=fold_slice_data[idx]["TRX"][1]
            if v is None:
                cb=[(net_etc[t]+net_trx[t])/2 for t in range(len(net_etc))]
            else:
                sch,par=v
                # for folds with n~548, window 200 still feasible; for small folds use same logic
                w_etc,w_trx=dynamic_weights(net_etc, net_trx, sch, par)
                cb=combo_dynamic([net_etc,net_trx],[w_etc,w_trx])
            fold_sharpes.append(stats(cb)["sharpe"])
        row["fold12_sharpes"]=[round(x,3) for x in fold_sharpes]
        row["fold12_mean"]=round(sum(fold_sharpes)/len(fold_sharpes),3)
        row["fold12_median"]=round(sorted(fold_sharpes)[len(fold_sharpes)//2],3)
        row["fold12_pos_rate"]=round(sum(1 for x in fold_sharpes if x>0)/len(fold_sharpes),3)
        # turnover FULL dyn already in row["FULL"]["turnover"]
        row["turnover_FULL_dyn"]=row["FULL"]["turnover"]
        results[k]=row
        log("E5 %s FULL(sh=%.3f ann=%.4f mdd=%.4f to=%.6f) H2=%.3f B=%.3f C=%.3f fold12(mean=%.3f med=%.3f pos=%.2f) wETC_FULL=%.3f std=%.3f flip_H2=%.4f flip_FULL=%.4f drift_FULL=%.6f" % (
            k, row["FULL"]["sharpe"], row["FULL"]["ann"], row["FULL"]["mdd"], row["turnover_FULL_dyn"],
            row["H2"]["sharpe"], row["B"]["sharpe"], row["C"]["sharpe"], row["fold12_mean"], row["fold12_median"], row["fold12_pos_rate"],
            row["wETC_mean_FULL"], row["wETC_std_FULL"], row["flip_rate_H2"], row["flip_rate_FULL"], row["weight_drift_FULL"]))

    base=results["STATIC_50_50"]
    comp=[]
    for k in ["DYN_A_sharpe200","DYN_B_invvol200","DYN_C_timing200","DYN_D_momo100"]:
        r=results[k]
        d={
          "scheme":k,
          "delta_sharpe_H2": round(r["H2"]["sharpe"]-base["H2"]["sharpe"],3),
          "delta_sharpe_B": round(r["B"]["sharpe"]-base["B"]["sharpe"],3),
          "delta_sharpe_C": round(r["C"]["sharpe"]-base["C"]["sharpe"],3),
          "delta_sharpe_FULL": round(r["FULL"]["sharpe"]-base["FULL"]["sharpe"],3),
          "delta_ann_H2": round(r["H2"]["ann"]-base["H2"]["ann"],4),
          "delta_ann_FULL": round(r["FULL"]["ann"]-base["FULL"]["ann"],4),
          "delta_fold12_mean": round(r["fold12_mean"]-base["fold12_mean"],3),
          "delta_fold12_median": round(r["fold12_median"]-base["fold12_median"],3),
          "turnover_FULL": r["turnover_FULL_dyn"],
          "turnover_static_FULL": base["turnover_FULL_dyn"],
          "turnover_increment_FULL": round(r["turnover_FULL_dyn"]-base["turnover_FULL_dyn"],6),
          "turnover_H2": r["H2"]["turnover"],
          "turnover_H2_static": base["H2"]["turnover"],
          "turnover_increment_H2": round(r["H2"]["turnover"]-base["H2"]["turnover"],6),
          "dual_win_H2_FULL": bool(r["H2"]["sharpe"]>base["H2"]["sharpe"] and r["FULL"]["sharpe"]>base["FULL"]["sharpe"]),
          "flip_rate_FULL": r["flip_rate_FULL"],
          "flip_rate_H2": r["flip_rate_H2"],
          "weight_drift_FULL": r["weight_drift_FULL"],
          "weight_drift_H2": r["weight_drift_H2"],
          "wETC_mean_FULL": r["wETC_mean_FULL"],
          "wETC_mean_H2": r["wETC_mean_H2"],
        }
        comp.append(d)
        log("COMPARE %s dH2=%+.3f dFULL=%+.3f dB=%+.3f dC=%+.3f dFoldMean=%+.3f to_inc_FULL=%+.6f to_inc_H2=%+.6f dual_win=%s" % (
            k, d["delta_sharpe_H2"], d["delta_sharpe_FULL"], d["delta_sharpe_B"], d["delta_sharpe_C"], d["delta_fold12_mean"], d["turnover_increment_FULL"], d["turnover_increment_H2"], d["dual_win_H2_FULL"]))
    adopt=None
    for d in comp:
        if d["dual_win_H2_FULL"] and d["turnover_increment_FULL"]<0.015:
            adopt=d["scheme"]; break
    decision = adopt if adopt else "KEEP_STATIC_50_50"
    reason = ("adopt %s dual-win H2+FULL with to_inc=%.6f<0.015" % (adopt, [x for x in comp if x["scheme"]==adopt][0]["turnover_increment_FULL"]) if adopt else "no scheme dual-wins H2+FULL with to_inc<0.015; reject all dynamics")
    log("DECISION: %s -- %s" % (decision, reason))
    for d in comp:
        log("  %s: H2 %.3f->%.3f (d%+.3f) FULL %.3f->%.3f (d%+.3f) to %.6f->%.6f inc %.6f dual_win=%s ok=%s" % (
            d["scheme"], base["H2"]["sharpe"], results[d["scheme"]]["H2"]["sharpe"], d["delta_sharpe_H2"],
            base["FULL"]["sharpe"], results[d["scheme"]]["FULL"]["sharpe"], d["delta_sharpe_FULL"],
            d["turnover_static_FULL"], d["turnover_FULL"], d["turnover_increment_FULL"], d["dual_win_H2_FULL"], bool(d["dual_win_H2_FULL"] and d["turnover_increment_FULL"]<0.015)))
    out={
      "config": {"formula":FORMULA,"fee":BASE_FEE,"fund":FUND,"slices":SLICES,"folds":folds,"base_Z1":{"ETC":list(BASE_Z1["ETC"]),"TRX":list(BASE_Z1["TRX"]),"q":BASE_Z1["q"]},
                 "schemes": {"STATIC_50_50":"static 0.5/0.5","DYN_A_sharpe200":"rolling 200bar sharpe-weighted (warmup 50/50)","DYN_B_invvol200":"rolling 200bar 1/vol20 weighted","DYN_C_timing200":"rolling 200bar sharpe<0 -> single leg","DYN_D_momo100":"rolling 100bar cum 0.6/0.4"}},
      "results": results,
      "comparison": comp,
      "baseline_static": {"H2":base["H2"],"B":base["B"],"C":base["C"],"FULL":base["FULL"],"fold12_mean":base["fold12_mean"],"fold12_median":base["fold12_median"],"turnover_FULL":base["turnover_FULL_dyn"],"fold12_sharpes":base["fold12_sharpes"]},
      "decision": {"adopt":decision, "reason":reason, "threshold_turnover_increment":0.015, "criterion":"dual-win H2 & FULL sharpe vs 50/50 and to_inc<0.015"},
    }
    pathlib.Path("results/backtest_E5.json").parent.mkdir(parents=True, exist_ok=True)
    open("results/backtest_E5.json","w").write(json.dumps(out, indent=2))
    log("Wrote results/backtest_E5.json")
    return out

if __name__=="__main__":
    main()
