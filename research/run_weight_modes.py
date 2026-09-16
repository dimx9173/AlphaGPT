"""P1-1 inverse-vol weighting contrast (E06/E07). Offline read-only."""
import csv, json, math, os, pathlib, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from strategy_manager.config import FORMULA, LOCKED_ATOM, LOCKED_APT, LOCKED_ETC, LOCKED_KAS, LOCKED_TRX, LEV, FUND, FEE, FEE2X
from strategy_manager.y1b_basket import realized_vol, invvol_weights, Y1B_VOL_WINDOW

COINS = ["ETC","TRX","ATOM","APT","KAS"]
SPECS = {"ETC":LOCKED_ETC,"TRX":LOCKED_TRX,"ATOM":LOCKED_ATOM,"APT":LOCKED_APT,"KAS":LOCKED_KAS}
BPY = 2190.0
H2_LEN = 493
VOL_W = 60
WMIN, WMAX = 0.10, 0.35
VOL_TGT = float(os.getenv("Y1B_VOL_TARGET","0.35"))
LEV_MIN, LEV_MAX = 0.25, 2.0
FEE_GRID = [(FEE,FUND),(FEE2X,FUND),(FEE,0.001),(FEE2X,0.001)]
OUT = pathlib.Path("results/weight_modes.json")
LOG = pathlib.Path("logs/weight_modes.log")

def log(m):
    print(m, flush=True)
    open(LOG,"a").write(m+"\n")

def load15(c):
    rows=list(csv.DictReader(open("data/data_15m_3y/%s.csv"%c)))
    return [(int(r["timestamp"]),float(r["open"]),float(r["high"]),float(r["low"]),float(r["close"]),float(r["volume"])) for r in rows]

def common4h(coins):
    raw={c:load15(c) for c in coins}
    s=max(r[0][0] for r in raw.values()); e=min(r[-1][0] for r in raw.values())
    bars={}; closes={}
    for c in coins:
        rr=[r for r in raw[c] if s<=r[0]<=e]
        b4=[]
        for i in range(len(rr)//16):
            blk=rr[i*16:(i+1)*16]
            b4.append((blk[0][1],max(r[2] for r in blk),min(r[3] for r in blk),blk[-1][4],sum(r[5] for r in blk)))
        bars[c]=b4; closes[c]=[b[3] for b in b4]
    n=min(len(b) for b in bars.values())
    for c in coins: bars[c]=bars[c][:n]; closes[c]=closes[c][:n]
    return bars, closes

def build_sig(bars):
    n=len(bars)
    raw={"open":torch.tensor([[b[0] for b in bars]]),"high":torch.tensor([[b[1] for b in bars]]),"low":torch.tensor([[b[2] for b in bars]]),"close":torch.tensor([[b[3] for b in bars]]),"volume":torch.tensor([[b[4] for b in bars]]),"liquidity":torch.full((1,n),1e7),"fdv":torch.full((1,n),1e8)}
    sig=StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets=[(bars[i+1][3]-bars[i][3])/bars[i][3] for i in range(n-1)]+[0.0]
    return raw, torch.tensor([rets]), sig

def qmask(sig,q):
    if q is None or float(q)<=0: return None
    a=sig.detach().float().abs().reshape(-1)
    k=max(1,int(len(a)*float(q)))
    thr=torch.topk(a,k).values.min()
    return (sig.detach().float().abs()>=thr).float()

def leg_net(raw,rt,sig,spec,fee,fund,lev_scale=1.0):
    bt=MemeBacktest(venue="aster",leverage=LEV,short_enabled=True,funding_override=fund,fee_override=fee,long_th=spec["lth"],short_th=spec["sth"],cooldown_bars=spec["cd"],bars_per_year=BPY,stop_loss=spec["sl"],time_stop=spec["ts"],vol_target=spec["vt"],vol_window=spec["vw"])
    sg=torch.sigmoid(sig); safe=(raw["liquidity"]>bt.min_liq).float()
    lp=(sg>bt.long_th).float()*safe; sp=(sg<bt.short_th).float()*safe
    mk=qmask(sig,spec["q"])
    if mk is not None: lp=lp*mk
    lp,sp=bt._apply_cooldown(lp,sp); lp,sp=bt._apply_stops(lp,sp,rt)
    sc=bt._vol_scale(rt); lp,sp=lp*sc,sp*sc
    lp=lp.roll(1,dims=1); lp[:,0]=0; sp=sp.roll(1,dims=1); sp[:,0]=0
    turn=(lp-lp.roll(1,dims=1)).abs()+(sp-sp.roll(1,dims=1)).abs()
    gross=(lp-sp)*rt*bt.leverage*lev_scale
    tx=turn*bt.base_fee*bt.leverage*lev_scale
    fnd=(lp-sp)*bt.default_funding_rate*bt.leverage*lev_scale
    return (gross-tx-fnd)[0].tolist(), turn[0].tolist(), (lp-sp)[0].tolist()

def sharpe(s):
    n=len(s)
    if n<2: return 0.0
    m=sum(s)/n; v=sum((x-m)**2 for x in s)/(n-1)
    return m/math.sqrt(v)*math.sqrt(BPY) if v>0 else 0.0

def seg(net,turn,a,b):
    s=net[a:b]; t=turn[a:b]; n=len(s); m=sum(s)/n if n else 0
    v=sum((x-m)**2 for x in s)/max(n-1,1) if n>1 else 0
    sh=m/math.sqrt(v)*math.sqrt(BPY) if v>0 else 0.0
    cs=pk=md=0.0; pk0=-1e18
    cs2=0.0
    for x in s:
        cs2+=x; pk0=max(pk0,cs2); md=max(md,pk0-cs2)
    return {"sharpe":round(sh,3),"ann":round(sum(s)/n*BPY,4) if n else 0.0,"mdd":round(md,4),"cum":round(sum(s),4),"n":n,"turnover":round(sum(t)/n,6) if n else 0.0}

def dd_contrib(net,legs,Ws,Ls):
    n=len(net)
    trace=[]; cs=0.0
    for x in net: cs+=x; trace.append(cs)
    pkv=trace[0]; pak=0; bd=0.0; ta=0; tb=0
    for i,v in enumerate(trace):
        if v>pkv: pkv=v; pak=i
        if pkv-v>bd: bd=pkv-v; ta=pak; tb=i
    out={}
    for c in COINS:
        out[c]=round(sum(legs[c][k]*Ws[k][c]*Ls[k] for k in range(ta,tb+1)),4)
    return {"dd_start":ta,"dd_end":tb,"dd_depth":round(bd,4),"by_coin":out}

def weights_series(closes,mode,target):
    n=len(closes[COINS[0]])
    W=[None]*n; LS=[1.0]*n; V={c:[] for c in COINS}
    if mode=="equal":
        w={c:0.2 for c in COINS}
        return [dict(w) for _ in range(n)],[1.0]*n,{"mode":"equal","lev_scale":1.0}
    vols_hist={c:[] for c in COINS}
    out=[]; lss=[]
    for t in range(n):
        vols={}
        for c in COINS:
            win=closes[c][max(0,t-VOL_W):t+1]
            vols[c]=realized_vol(win,VOL_W) if len(win)>=3 else 0.0
            vols_hist[c].append(round(vols[c],4))
        if t<VOL_W:
            w={c:0.2 for c in COINS}; ls=1.0
        else:
            w=invvol_weights(vols,WMIN,WMAX)
            ls=1.0
            if mode=="invvol_cap":
                m=min(VOL_W+1, t+1)
                mat=[[ (closes[c][t-m+1+k+1]-closes[c][t-m+1+k])/closes[c][t-m+1+k] for k in range(m-1)] for c in COINS]
                mm=m-1
                if mm>=20:
                    means=[sum(r)/mm for r in mat]; pv=0.0
                    for i in range(5):
                        for j in range(5):
                            cov=sum((mat[i][k]-means[i])*(mat[j][k]-means[j]) for k in range(mm))/(mm-1)
                            pv+=w[COINS[i]]*w[COINS[j]]*cov
                    pv_=max(pv,0.0)**0.5*math.sqrt(BPY)
                    ls=min(max(target/pv_ if pv_>1e-9 else 1.0,LEV_MIN),LEV_MAX)
        out.append({c:round(w[c],4) for c in COINS}); lss.append(round(ls,4))
    return out,lss,{"mode":mode}

def fold12(net,turn):
    n=len(net); fn=n//12; ss=[]
    for i in range(12):
        a=i*fn; b=a+fn if i<11 else n
        ss.append(seg(net,turn,a,b)["sharpe"])
    srt=sorted(ss)
    return {"sharpes":[round(x,3) for x in ss],"mean":round(sum(ss)/len(ss),3),"median":round(srt[len(srt)//2],3),"n_pos":sum(1 for x in ss if x>0)}

def main():
    LOG.parent.mkdir(parents=True,exist_ok=True); OUT.parent.mkdir(parents=True,exist_ok=True)
    open(LOG,"w").write("weight_modes start\n")
    bars,closes=common4h(COINS)
    n=len(bars["ETC"]); h2a=n-H2_LEN
    log("common 4h n=%d h2a=%d"% (n,h2a))
    mats={c:build_sig(bars[c]) for c in COINS}
    modes={}
    for mode in ["equal","invvol","invvol_cap"]:
        legs={}; turns={}
        for c in COINS:
            raw,rt,sg=mats[c]
            legs[c],turns[c],_=leg_net(raw,rt,sg,SPECS[c],FEE,FUND,1.0)
        Ws,Ls,_=weights_series(closes,mode,VOL_TGT)
        net=[sum(legs[c][t]*Ws[t][c]*Ls[t] for c in COINS) for t in range(n)]
        turn=[sum(turns[c][t]*Ws[t][c]*Ls[t] for c in COINS) for t in range(n)]
        full=seg(net,turn,0,n); h2=seg(net,turn,h2a,n)
        to_inc=round(full["turnover"]-0.0,6)
        dd=dd_contrib(net,legs,Ws,Ls)
        # fee2x four-grid on FULL+H2 sharpe
        grid=[]
        for fee,fund in FEE_GRID:
            l2={}; t2={}
            for c in COINS:
                raw,rt,sg=mats[c]
                l2[c],t2[c],_=leg_net(raw,rt,sg,SPECS[c],fee,fund,1.0)
            n2=[sum(l2[c][t]*Ws[t][c]*Ls[t] for c in COINS) for t in range(n)]
            tu2=[sum(t2[c][t]*Ws[t][c]*Ls[t] for c in COINS) for t in range(n)]
            grid.append({"fee":fee,"fund":fund,"FULL_sharpe":seg(n2,tu2,0,n)["sharpe"],"H2_sharpe":seg(n2,tu2,h2a,n)["sharpe"],"FULL_turnover":seg(n2,tu2,0,n)["turnover"]})
        f12=fold12(net,turn)
        modes[mode]={"FULL":full,"H2":h2,"dd":dd,"turnover_inc_vs_equal":None,"fee_grid":grid,"fold12":f12,
            "w_final":{c:Ws[-1][c] for c in COINS},"lev_final":Ls[-1],
            "w_mean":{c:round(sum(Ws[t][c] for t in range(n))/n,4) for c in COINS},
            "lev_mean":round(sum(Ls)/n,4)}
        log("%s FULL sh=%.3f dd=%.4f to=%.5f | H2 sh=%.3f | w=%s lev=%.3f | fold med=%.3f npos=%d/12"%(
            mode,full["sharpe"],full["mdd"],full["turnover"],h2["sharpe"],modes[mode]["w_final"],Ls[-1],f12["median"],f12["n_pos"]))
    base=modes["equal"]["FULL"]["turnover"]
    for m in modes: modes[m]["turnover_inc_vs_equal"]=round(modes[m]["FULL"]["turnover"]-base,6)
    A,B,C=modes["equal"],modes["invvol"],modes["invvol_cap"]
    b_win=bool(B["FULL"]["sharpe"]>A["FULL"]["sharpe"] and B["FULL"]["mdd"]<=0.8*A["FULL"]["mdd"])
    c_win=bool(C["FULL"]["sharpe"]>A["FULL"]["sharpe"] and C["FULL"]["mdd"]<=0.8*A["FULL"]["mdd"])
    verdict="PENDING_P03_FAIL"
    decision="KEEP_equal"
    note=("P0-3 permutation FAIL (per-coin p>=0.05) => P1-1 verdict PENDING; contrast only, "
          "default stays equal regardless of B/C outcome.")
    res={"config":{"engine":"mirror run_qsweep leg_net + quantile q0.3 long-only + cooldown + stops + roll1; weights dynamic trailing 60x4h realized-vol reciprocal, clip 0.10-0.35; invvol_cap rescales lev to vol_target","formula":list(FORMULA),"basket":{c:dict(SPECS[c]) for c in COINS},"vol_window":VOL_W,"clip":[WMIN,WMAX],"vol_target":VOL_TGT,"lev_clamp":[LEV_MIN,LEV_MAX],"venue":"aster","lev":LEV,"fund":FUND,"fee":FEE,"fee2x":FEE2X,"fee_grid":[{ "fee":f,"fund":d} for f,d in FEE_GRID],"grid_bars":n,"h2_len":H2_LEN,"note":"E10 FORMULA untouched; live default equal; offline read-only"},
        "modes":modes,
        "compare":{"B_beats_A":b_win,"C_beats_A":c_win,"rule":"sharpe+ AND maxDD-20%+ vs A"},
        "verdict":verdict,"decision":decision,"decision_note":note}
    OUT.write_text(json.dumps(res,indent=1,ensure_ascii=False))
    log("wrote %s verdict=%s decision=%s"% (OUT,verdict,decision))

if __name__=="__main__":
    main()
