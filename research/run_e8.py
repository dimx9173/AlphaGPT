"""E8 跨幣與 regime 診斷 — 單幣 H2 排名挑第三腿壓力 + SMA50 趨勢分桶
任務：
  引擎鏡像 run_ad/run_ac，單幣參數復用 TRX-spec (0.85/0.12/cd6/0.05/ts24) q0.3 vtNone both
  對全 29 幣跑單幣 H2/B/C sharpe 排名 (各段 both-leg, q0.3)，取 H2 top5 與 B top5 與 C top5 併集 ~10-12 幣
  對每幣試以 50/25/25 (ETC/TRX/third) 併入雙腿在 H2/B/C/FULL 測 beats_both (H2 & FULL 雙勝)，turnover 與 fee2x 一併報告
  另做 regime 分桶：以 FULL 持倉期的 50bar SMA 趨勢（close 50bar 均線斜率）把 FULL 拆為 up/flat/down 三桶，各桶內對 Y1b 雙腿算 sharpe/ann/占比
  判定：併集 12 幣中 0/12 beats_both 則再確 REJECT 第三腿；regime 某桶 sharpe<0 標記弱 regime
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

BASE_FORMULA=[3,2,7,2,7,11,15,4,4,6,6,10]
BASE_FEE=0.0004
FEE2X=0.0008
FUND=0.0005
BARS_PER_YEAR=2190.0
Q=0.3
SEGS={"H1":(5584,6077),"H2":(6077,6570),"B":(6380,6580),"C":(6080,6580),"FULL":(0,6580)}
# TRX-spec single + Y1b dual spec (vtNone)
TRX_SPEC=(0.85,0.12,6,0.05,24,None,12)  # vt None => no vol scale
DUAL_SPEC={
  "ETC":(0.88,0.12,18,None,24,None,12),
  "TRX":(0.85,0.12,6,0.05,24,None,12),
}
ALL_COINS=["ADA","APT","ARB","ASTER","ATOM","AVAX","BCH","BNB","BTC","DOGE","DOT","ETC","ETH","HBAR","ICP","KAS","LINK","LTC","NEAR","PEPE","POL","RENDER","SHIB","SOL","SUI","TRX","UNI","XLM","XRP"]

def load_bars(coin):
    p=f"data/data_15m_3y/{coin}.csv"
    rows=list(csv.DictReader(open(p)))
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
    sig=StackVM().execute(BASE_FORMULA, FeatureEngineer.compute_features(raw))
    rets=[(bars[i+1][3]-bars[i][3])/bars[i][3] for i in range(n-1)]+[0.0]
    return raw, torch.tensor([rets]), sig

def quantile_mask_long(sig,q):
    if q is None: return None
    a=sig.detach().float().abs().reshape(-1)
    k=max(1,int(len(a)*float(q)))
    thr=torch.topk(a,k).values.min()
    return (sig.detach().float().abs()>=thr).float()

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, ts, fee=None, fund=None, q=Q, vt=None, vw=12, side="both"):
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
    # when vt None scale is 1.0 scalar; handle
    if isinstance(scale, float):
        pass
    else:
        lp,sp=lp*scale, sp*scale
    if side=="long":
        sp=sp*0.0
    if side=="short":
        lp=lp*0.0
    lp=lp.roll(1,dims=1); lp[:,0]=0
    sp=sp.roll(1,dims=1); sp[:,0]=0
    turn=(lp-lp.roll(1,dims=1)).abs()+(sp-sp.roll(1,dims=1)).abs()
    tx=turn*(bt.base_fee + torch.clamp(bt.trade_size/(raw["liquidity"]+1e-9),0.0,0.05))
    gross=(lp-sp)*rets_t*bt.leverage
    fnd=(lp-sp)*bt.default_funding_rate*bt.leverage
    net=(gross - tx*bt.leverage - fnd)[0].tolist()
    pos=(lp-sp)[0].tolist()
    turnl=turn[0].tolist()
    trades=sum(1 for t in range(len(pos)) if pos[t]!=0.0 and (t==0 or pos[t-1]==0.0))
    turnover=sum(turnl)/len(turnl) if turnl else 0.0
    return net,trades,turnover,pos

def stats(ser,trades=0,turnover=0.0):
    n=len(ser)
    if n==0:
        return {"sharpe":0.0,"ann":0.0,"mdd":0.0,"cum":0.0,"n":0,"trades":trades,"turnover":round(turnover,6)}
    mean=sum(ser)/n
    var=sum((x-mean)**2 for x in ser)/max(n-1,1) if n>1 else 0.0
    sharpe=mean/math.sqrt(var)*math.sqrt(BARS_PER_YEAR) if var>0 else 0.0
    cum=sum(ser)
    ann=cum/n*BARS_PER_YEAR if n else 0.0
    cs,peak,mdd=0.0,-1e18,0.0
    for x in ser:
        cs+=x; peak=max(peak,cs); mdd=max(mdd,peak-cs)
    return {"sharpe":round(sharpe,3),"ann":round(ann,4),"mdd":round(mdd,4),"cum":round(cum,4),"n":n,"trades":trades,"turnover":round(turnover,6)}

def combo(ser_list, weights):
    m=min(len(s) for s in ser_list)
    sw=sum(weights)
    return [sum(ser_list[i][t]*weights[i]/sw for i in range(len(ser_list))) for t in range(m)]

def per_coin_seg_stats(bars_seg, spec_tuple, fee=BASE_FEE):
    raw,rt,sg=build_mats(bars_seg)
    lth,sth,cd,sl,ts,vt,vw=spec_tuple
    net,t,to,_=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=fee,q=Q,vt=vt,vw=vw,side="both")
    s=stats(net,trades=t,turnover=to)
    # fee2x and side split
    net2,t2,to2,_=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=FEE2X,q=Q,vt=vt,vw=vw,side="both")
    s2=stats(net2,trades=t2,turnover=to2)
    # side
    sides={}
    for sd in ["both","long","short"]:
        net_sd,tsd,tosd,_=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=fee,q=Q,vt=vt,vw=vw,side=sd)
        sides[sd]=stats(net_sd,trades=tsd,turnover=tosd)
    return s,s2,sides,net

def main():
    logp=pathlib.Path("logs/e8.log")
    logp.parent.mkdir(parents=True, exist_ok=True)
    open(logp,"w").write("")
    def log(msg):
        print(msg,flush=True)
        with open(logp,"a") as f: f.write(msg+"\n")
    log("E8 start formula=%s TRX-spec=%s DUAL=%s q=%.1f vtNone fee=%.4f fund=%.4f segs=%s"% (BASE_FORMULA, TRX_SPEC, DUAL_SPEC, Q, BASE_FEE, FUND, SEGS))
    # load all coins full bars (for per-seg slicing)
    full={c: load_bars(c) for c in ALL_COINS}
    for c in ALL_COINS:
        log("  %s n=%d"%(c,len(full[c])))
    # ---------- single-coin stats per segment ----------
    single_results={}
    for coin in ALL_COINS:
        # use TRX-spec for every coin single
        spec=TRX_SPEC
        row={"coin":coin,"spec":list(spec),"q":Q}
        # need bars per seg; handle short coins like ASTER(2111),POL,RENDER,KAS — segs FULL etc require min length; skip or pad
        # Strategy: for segs where b > n, truncate to available. Report n actually used.
        for seg_name,(a,b) in SEGS.items():
            n=len(full[coin])
            aa=min(a,n); bb=min(b,n)
            if aa>=bb:
                # no data for this seg (e.g. ASTER H2 at 6077 >2111)
                row[seg_name]={"sharpe":0.0,"ann":0.0,"mdd":0.0,"cum":0.0,"n":0,"trades":0,"turnover":0.0,"note":"no_data"}
                row[seg_name+"_fee2x"]={"sharpe":0.0,"ann":0.0,"mdd":0.0,"cum":0.0,"n":0,"trades":0,"turnover":0.0,"note":"no_data"}
                row[seg_name+"_side"]= {"both":{"sharpe":0.0},"long":{"sharpe":0.0},"short":{"sharpe":0.0}}
                row[seg_name+"_n"]=0
                continue
            seg_bars=full[coin][aa:bb]
            s,s2,sides,net=per_coin_seg_stats(seg_bars, spec, fee=BASE_FEE)
            row[seg_name]=s
            row[seg_name+"_fee2x"]=s2
            row[seg_name+"_side"]=sides
            row[seg_name+"_n"]=bb-aa
            row[seg_name+"_net"]=net  # keep for triple? remove later for json size? keep
        single_results[coin]=row
        log("single %s H2(sh=%.3f n=%d) B=%.3f C=%.3f FULL=%.3f fee2x_H2=%.3f trH2=%d toH2=%.4f"%(
            coin, row["H2"]["sharpe"], row.get("H2_n",0), row["B"]["sharpe"], row["C"]["sharpe"], row["FULL"]["sharpe"], row["H2_fee2x"]["sharpe"], row["H2"]["trades"], row["H2"]["turnover"]))
    # ranking H2/B/C by sharpe (desc), among coins with n>0 for that seg
    def top5(seg):
        items=[(c, single_results[c][seg]["sharpe"]) for c in ALL_COINS if single_results[c].get(seg+"_n",0)>0]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:5]
    h2_top=top5("H2")
    b_top=top5("B")
    c_top=top5("C")
    log("H2 top5: %s"% h2_top)
    log("B top5: %s"% b_top)
    log("C top5: %s"% c_top)
    union=list(dict.fromkeys([c for c,_ in h2_top]+[c for c,_ in b_top]+[c for c,_ in c_top]))
    log("union %s n=%d"% (union,len(union)))
    # ranking tables for report
    def sorted_all(seg):
        items=[(c, single_results[c][seg]["sharpe"], single_results[c][seg]["ann"], single_results[c][seg]["mdd"], single_results[c][seg]["trades"], single_results[c].get(seg+"_n",0)) for c in ALL_COINS]
        items.sort(key=lambda x: x[1], reverse=True)
        return items

    # ---------- dual baseline Y1b (ETC+TRX) per seg ----------
    dual_stats={}
    for seg_name,(a,b) in SEGS.items():
        # dual uses min(ETC,TRX) =6580, so slice valid
        legs=[]; trs=[]; tos=[]
        for c in ["ETC","TRX"]:
            spec=DUAL_SPEC[c]
            seg_bars=full[c][a:b]
            s,s2,sides,net=per_coin_seg_stats(seg_bars, spec, fee=BASE_FEE)
            # but for combo we need net series, not stats; reutilize net
            legs.append(net); trs.append(s["trades"]); tos.append(s["turnover"])
        cb=combo(legs,[0.5,0.5])
        ds=stats(cb,trades=sum(trs),turnover=sum(tos)/len(tos) if tos else 0.0)
        # fee2x
        legs2=[]; trs2=[]; tos2=[]
        for c in ["ETC","TRX"]:
            spec=DUAL_SPEC[c]
            raw,rt,sg=build_mats(full[c][a:b])
            lth,sth,cd,sl,ts,vt,vw=spec
            net2,t2,to2,_=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=FEE2X,q=Q,vt=vt,vw=vw,side="both")
            legs2.append(net2); trs2.append(t2); tos2.append(to2)
        cb2=combo(legs2,[0.5,0.5])
        ds2=stats(cb2,trades=sum(trs2),turnover=sum(tos2)/len(tos2) if tos2 else 0.0)
        dual_stats[seg_name]=ds
        dual_stats[seg_name+"_fee2x"]=ds2
        log("dual %s sh=%.3f fee2x=%.3f tr=%d to=%.4f FULL? %s"% (seg_name, ds["sharpe"], ds2["sharpe"], ds["trades"], ds["turnover"], seg_name=="FULL"))

    # ---------- triple pressure 50/25/25 for union ----------
    triple_rows=[]
    for third in union:
        weights=[0.5,0.25,0.25]  # ETC/TRX/third
        coins=["ETC","TRX",third]
        row={"third":third,"weights":weights,"coins":coins}
        # for each seg compute combo
        for seg_name,(a,b) in SEGS.items():
            n_third=len(full[third])
            # dual segs always valid; for third, if no data for seg, skip stats as 0
            # align length: combo truncates to min len
            legs=[]; trs=[]; tos=[]
            # ETC
            spec=DUAL_SPEC["ETC"]
            raw,rt,sg=build_mats(full["ETC"][a:b])
            lth,sth,cd,sl,ts,vt,vw=spec
            net_etc,t_etc,to_etc,_=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=BASE_FEE,q=Q,vt=vt,vw=vw,side="both")
            legs.append(net_etc); trs.append(t_etc); tos.append(to_etc)
            # TRX
            spec=DUAL_SPEC["TRX"]
            raw,rt,sg=build_mats(full["TRX"][a:b])
            lth,sth,cd,sl,ts,vt,vw=spec
            net_trx,t_trx,to_trx,_=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=BASE_FEE,q=Q,vt=vt,vw=vw,side="both")
            legs.append(net_trx); trs.append(t_trx); tos.append(to_trx)
            # third
            aa=min(a,n_third); bb=min(b,n_third)
            if aa>=bb:
                # no data for third in this seg: treat leg as zeros of length matching dual min (pad)
                m=min(len(net_etc),len(net_trx))
                net_third=[0.0]*m
                t_third=0; to_third=0.0
            else:
                spec_third=TRX_SPEC
                raw,rt,sg=build_mats(full[third][aa:bb])
                lth,sth,cd,sl,ts,vt,vw=spec_third
                net_third,t_third,to_third,_=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=BASE_FEE,q=Q,vt=vt,vw=vw,side="both")
                # if third shorter, combo will trim
            legs.append(net_third); trs.append(t_third); tos.append(to_third)
            cb=combo(legs,weights)
            s=stats(cb,trades=sum(trs),turnover=sum(tos)/len(tos) if tos else 0.0)
            row[seg_name]=s
            # fee2x
            # recompute with fee2x
            legs2=[]; trs2=[]; tos2=[]
            # ETC fee2x
            spec=DUAL_SPEC["ETC"]
            raw,rt,sg=build_mats(full["ETC"][a:b])
            lth,sth,cd,sl,ts,vt,vw=spec
            net,t,to,_=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=FEE2X,q=Q,vt=vt,vw=vw,side="both")
            legs2.append(net); trs2.append(t); tos2.append(to)
            # TRX fee2x
            spec=DUAL_SPEC["TRX"]
            raw,rt,sg=build_mats(full["TRX"][a:b])
            lth,sth,cd,sl,ts,vt,vw=spec
            net,t,to,_=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=FEE2X,q=Q,vt=vt,vw=vw,side="both")
            legs2.append(net); trs2.append(t); tos2.append(to)
            # third fee2x
            if aa>=bb:
                m=min(len(legs2[0]),len(legs2[1]))
                net=[0.0]*m; t=0; to=0.0
            else:
                spec_third=TRX_SPEC
                raw,rt,sg=build_mats(full[third][aa:bb])
                lth,sth,cd,sl,ts,vt,vw=spec_third
                net,t,to,_=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=FEE2X,q=Q,vt=vt,vw=vw,side="both")
            legs2.append(net); trs2.append(t); tos2.append(to)
            cb2=combo(legs2,weights)
            s2=stats(cb2,trades=sum(trs2),turnover=sum(tos2)/len(tos2) if tos2 else 0.0)
            row[seg_name+"_fee2x"]=s2
            row["fee_decay_"+seg_name]=round(s["sharpe"]-s2["sharpe"],3)
        # corr in H2 and FULL between ETC/TRX legs (reuse dual nets) and third vs dual?
        # beats_both H2 & FULL double win
        beats_H2=row["H2"]["sharpe"]>dual_stats["H2"]["sharpe"]
        beats_FULL=row["FULL"]["sharpe"]>dual_stats["FULL"]["sharpe"]
        row["beats_H2"]=bool(beats_H2)
        row["beats_FULL"]=bool(beats_FULL)
        row["beats_both"]=bool(beats_H2 and beats_FULL)
        row["dual_H2"]=dual_stats["H2"]["sharpe"]
        row["dual_FULL"]=dual_stats["FULL"]["sharpe"]
        row["delta_H2"]=round(row["H2"]["sharpe"]-dual_stats["H2"]["sharpe"],3)
        row["delta_FULL"]=round(row["FULL"]["sharpe"]-dual_stats["FULL"]["sharpe"],3)
        triple_rows.append(row)
        log("triple %s 50/25/25 H2 sh=%.3f(d=%.3f fee2x=%.3f) B=%.3f C=%.3f FULL sh=%.3f(d=%.3f fee2x=%.3f) toH2=%.4f beats_both=%s"%(
            third, row["H2"]["sharpe"], row["delta_H2"], row["H2_fee2x"]["sharpe"], row["B"]["sharpe"], row["C"]["sharpe"], row["FULL"]["sharpe"], row["delta_FULL"], row["FULL_fee2x"]["sharpe"], row["H2"]["turnover"], row["beats_both"]))

    # beats summary
    n_beats=sum(1 for r in triple_rows if r["beats_both"])
    if triple_rows:
        best_H2=max(triple_rows, key=lambda r: r["H2"]["sharpe"])
        best_FULL=max(triple_rows, key=lambda r: r["FULL"]["sharpe"])
        best_both=[r for r in triple_rows if r["beats_both"]]
        log("triple union n=%d beats_both %d/12 best_H2 %s sh=%.3f best_FULL %s sh=%.3f"% (len(union), n_beats, best_H2["third"], best_H2["H2"]["sharpe"], best_FULL["third"], best_FULL["FULL"]["sharpe"]))
        if best_both:
            for r in sorted(best_both, key=lambda r: r["delta_FULL"], reverse=True):
                log("  beats_both %s dH2=%.3f dFULL=%.3f"% (r["third"], r["delta_H2"], r["delta_FULL"]))

    # ---------- regime buckets by SMA50 slope on FULL ----------
    # Use FULL close series of a reference coin? Task says 以 FULL 的持倉期按趨勢分 — 用 50bar SMA 趨勢（close 50bar 均線斜率）把 FULL 拆為 up/flat/down 三桶
    # Interpret as: compute SMA50 on BTC or ETC as market proxy? More neutral: compute per-duo combo PnL buckets based on ETC close SMA50 slope (ETC as representative), but task implies single market trend split.
    # We'll use BTC as broad market proxy SMA50; slope via diff. Bucket thresholds: tercile of slope distribution across FULL (6580) -> up/flat/down each ~33%. Alternatively std band; we choose tercile for balanced buckets.
    # Compute slope_t = SMA50[t]-SMA50[t-1]; classify per bar t.
    # Then for each bucket, slice the dual net series indices belonging to bucket and compute stats on that subseries (sharpe/ann/占比 bars)
    import numpy as np
    # pick BTC closes for trend unless BTC n mismatch; BTC is 6580 same as ETC/TRX -> good.
    ref_closes=np.array([b[3] for b in full["BTC"][:6580]], dtype=float)
    n_full=len(ref_closes)
    # SMA50
    sma=np.zeros(n_full)
    for t in range(n_full):
        a=max(0,t-49)
        sma[t]=ref_closes[a:t+1].mean()
    slope=np.zeros(n_full)
    slope[1:]=sma[1:]-sma[:-1]
    # tercile thresholds
    p33=np.percentile(slope,33)
    p67=np.percentile(slope,67)
    labels=["down","flat","up"]
    buckets={}
    for label in labels:
        if label=="down":
            idx=np.where(slope<=p33)[0].tolist()
        elif label=="flat":
            idx=np.where((slope>p33)&(slope<=p67))[0].tolist()
        else:
            idx=np.where(slope>p67)[0].tolist()
        buckets[label]=idx
        log("regime %s p_thresh %.2f-%.2f n=%d pct=%.2f"% (label, p33 if label!="up" else p67, p67 if label!="up" else float("inf"), len(idx), len(idx)/n_full))
    # For calibration also compute k=0.3 std flat band as alternative metric for report
    std=slope.std()
    flat_k=np.sum(np.abs(slope)<=0.3*std)
    log("regime alt 0.3std flat n=%d pct=%.2f th=%.2f"% (flat_k, flat_k/n_full, 0.3*std))
    # Now compute dual net FULL series aligned
    # Build FULL dual net once for regime slicing
    raw_etc,rt_etc,sg_etc=build_mats(full["ETC"][:6580])
    raw_trx,rt_trx,sg_trx=build_mats(full["TRX"][:6580])
    lth,sth,cd,sl,ts,vt,vw=DUAL_SPEC["ETC"]
    net_etc,_,_,_=leg_series(raw_etc,rt_etc,sg_etc,lth,sth,cd,sl,ts,fee=BASE_FEE,q=Q,vt=vt,vw=vw,side="both")
    lth,sth,cd,sl,ts,vt,vw=DUAL_SPEC["TRX"]
    net_trx,_,_,_=leg_series(raw_trx,rt_trx,sg_trx,lth,sth,cd,sl,ts,fee=BASE_FEE,q=Q,vt=vt,vw=vw,side="both")
    cb_full=combo([net_etc,net_trx],[0.5,0.5])
    # slice per bucket
    regime_stats={}
    weak_regime=[]
    for label,idx in buckets.items():
        sub=[cb_full[i] for i in idx] if idx else []
        n_sub=len(sub)
        pct= n_sub/n_full if n_full else 0
        s=stats(sub) if sub else {"sharpe":0.0,"ann":0.0,"mdd":0.0,"cum":0.0,"n":0}
        # adjust ann already per BPY but sliced; keep same
        s["pct_bars"]=round(pct,4)
        s["n"]=n_sub
        regime_stats[label]=s
        is_weak= s["sharpe"]<0
        if is_weak:
            weak_regime.append(label)
        log("regime %s n=%d pct=%.2f sharpe=%.3f ann=%.4f cum=%.4f mdd=%.4f weak=%s"% (label,n_sub,pct,s["sharpe"],s["ann"],s["cum"],s["mdd"], is_weak))
    # Also overall FULL dual for reference already in dual_stats
    # ---------- verdict ----------
    reject_third= (n_beats==0)
    # ---------- save ----------
    # Prepare ranked tables without heavy net arrays for json
    def strip_nets(d):
        nd={}
        for k,v in d.items():
            # skip net arrays
            if k.endswith("_net"):
                continue
            nd[k]=v
        return nd
    single_stripped={c: strip_nets(single_results[c]) for c in ALL_COINS}
    out={
        "config":{"formula":BASE_FORMULA,"trx_spec":list(TRX_SPEC),"dual_spec":{k:list(v) for k,v in DUAL_SPEC.items()},"q":Q,"fee":BASE_FEE,"fee2x":FEE2X,"fund":FUND,"segs":{k:list(v) for k,v in SEGS.items()},"sma_window":50,"regime_method":"SMA50 slope tercile (BTC proxy) p33=%.2f p67=%.2f"% (p33,p67),"triple_weights":[0.5,0.25,0.25]},
        "single_coin":{k: single_stripped[k] for k in ALL_COINS},
        "rankings":{"H2_top5":h2_top,"B_top5":b_top,"C_top5":c_top,"union":union,"H2_sorted":sorted_all("H2"),"B_sorted":sorted_all("B"),"C_sorted":sorted_all("C"),"FULL_sorted":sorted_all("FULL")},
        "dual":{seg: dual_stats[seg] for seg in ["H2","H2_fee2x","B","B_fee2x","C","C_fee2x","FULL","FULL_fee2x"]} ,
        "triple":{"rows":triple_rows,"n_union":len(union),"n_beats_both":n_beats,"reject_third":bool(reject_third)},
        "regime":{"slope_p33":round(float(p33),2),"slope_p67":round(float(p67),2),"slope_std":round(float(std),2),"alt_flat_0_3std_n":int(flat_k),"buckets":{k: regime_stats[k] for k in labels},"weak_regime":weak_regime},
        "verdict":{"reject_third":bool(reject_third),"weak_regime":weak_regime,"n_beats_both":n_beats,"union_size":len(union)},
        "summary":{"single_H2_top":h2_top,"triple_best_H2": {"third": best_H2["third"],"H2":best_H2["H2"],"delta_H2":best_H2["delta_H2"]} if triple_rows else None,"triple_best_FULL": {"third": best_FULL["third"],"FULL":best_FULL["FULL"],"delta_FULL":best_FULL["delta_FULL"]} if triple_rows else None,"regime_weak":weak_regime}
    }
    pathlib.Path("results/backtest_E8.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    log("saved results/backtest_E8.json union=%d beats_both %d/12 reject_third=%s weak=%s"% (len(union),n_beats,reject_third,weak_regime))
    log("E8 done")

if __name__=="__main__":
    main()
