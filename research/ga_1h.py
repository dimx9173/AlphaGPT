import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, math, random, time
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
import csv

FORMULA = [3,2,7,2,7,11,15,4,4,6,6,10]
BPY = 8760.0  # 1h bars per year

# Load 1h data directly
def load_1h_bars(coin):
    rows = list(csv.DictReader(open(f"data/data_1y/1h/{coin}.csv")))
    bars = []
    for r in rows:
        bars.append((float(r["open"]), float(r["high"]), float(r["low"]),
                     float(r["close"]), float(r["volume"])))
    return bars

# Preload data
print("Loading 1h data...", flush=True)
bars_etc = load_1h_bars("ETC")
bars_trx = load_1h_bars("TRX")
n_etc = len(bars_etc); n_trx = len(bars_trx)
print(f"ETC: {n_etc} bars, TRX: {n_trx} bars", flush=True)

# Build torch tensors for backtest
def build_torch_mats(bars):
    n = len(bars)
    return {"open":torch.tensor([[b[0] for b in bars]]),
            "high":torch.tensor([[b[1] for b in bars]]),
            "low":torch.tensor([[b[2] for b in bars]]),
            "close":torch.tensor([[b[3] for b in bars]]),
            "volume":torch.tensor([[b[4] for b in bars]]),
            "liquidity":torch.full((1,n),1e7), "fdv":torch.full((1,n),1e8)}

torch_mats_etc = build_torch_mats(bars_etc)
torch_mats_trx = build_torch_mats(bars_trx)
rets_etc = torch.tensor([[(bars_etc[i+1][3]-bars_etc[i][3])/bars_etc[i][3] for i in range(n_etc-1)] + [0.0]])
rets_trx = torch.tensor([[(bars_trx[i+1][3]-bars_trx[i][3])/bars_trx[i][3] for i in range(n_trx-1)] + [0.0]])

def eval_gene(gene):
    """Evaluate gene and return FULL sharpe (equal-weight ETC+TRX)"""
    etc_lth, etc_sth, etc_cd, trx_lth, trx_sth, trx_cd, lev, ts = gene

    # ETC backtest
    feat_etc = FeatureEngineer.compute_features(torch_mats_etc, use_advanced=False)
    sig_etc = StackVM(use_advanced=False).execute(FORMULA, feat_etc)
    bt_etc = MemeBacktest(venue="aster", leverage=lev, short_enabled=True, funding_override=0.0005,
                          long_th=etc_lth, short_th=etc_sth, cooldown_bars=etc_cd, bars_per_year=BPY, stop_loss=None)
    signal = torch.sigmoid(sig_etc)
    is_safe = (torch_mats_etc["liquidity"] > bt_etc.min_liq).float()
    lp = (signal > bt_etc.long_th).float() * is_safe
    sp = (signal < bt_etc.short_th).float() * is_safe
    lp, sp = bt_etc._apply_cooldown(lp, sp)
    lp, sp = bt_etc._apply_stops(lp, sp, rets_etc)
    lp = lp.roll(1, dims=1); lp[:, 0] = 0
    sp = sp.roll(1, dims=1); sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    tx = turn * (bt_etc.base_fee + torch.clamp(bt_etc.trade_size / (torch_mats_etc["liquidity"] + 1e-9), 0.0, 0.05))
    gross = (lp - sp) * rets_etc * bt_etc.leverage
    fnd = (lp - sp) * bt_etc.default_funding_rate * bt_etc.leverage
    net_etc = (gross - tx * bt_etc.leverage - fnd)[0].tolist()

    # TRX backtest
    feat_trx = FeatureEngineer.compute_features(torch_mats_trx, use_advanced=False)
    sig_trx = StackVM(use_advanced=False).execute(FORMULA, feat_trx)
    bt_trx = MemeBacktest(venue="aster", leverage=lev, short_enabled=True, funding_override=0.0005,
                          long_th=trx_lth, short_th=trx_sth, cooldown_bars=trx_cd, bars_per_year=BPY, stop_loss=0.05)
    signal = torch.sigmoid(sig_trx)
    is_safe = (torch_mats_trx["liquidity"] > bt_trx.min_liq).float()
    lp = (signal > bt_trx.long_th).float() * is_safe
    sp = (signal < bt_trx.short_th).float() * is_safe
    lp, sp = bt_trx._apply_cooldown(lp, sp)
    lp, sp = bt_trx._apply_stops(lp, sp, rets_trx)
    lp = lp.roll(1, dims=1); lp[:, 0] = 0
    sp = sp.roll(1, dims=1); sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    tx = turn * (bt_trx.base_fee + torch.clamp(bt_trx.trade_size / (torch_mats_trx["liquidity"] + 1e-9), 0.0, 0.05))
    gross = (lp - sp) * rets_trx * bt_trx.leverage
    fnd = (lp - sp) * bt_trx.default_funding_rate * bt_trx.leverage
    net_trx = (gross - tx * bt_trx.leverage - fnd)[0].tolist()

    # Equal weight combo
    m = min(len(net_etc), len(net_trx))
    combo = [(net_etc[i] + net_trx[i])/2 for i in range(m)]
    mean = sum(combo)/len(combo)
    var = sum((x-mean)**2 for x in combo)/max(len(combo)-1,1)
    sharpe = mean/math.sqrt(var)*math.sqrt(BPY) if var > 0 else 0.0
    return sharpe

def tournament_selection(pop, scores, k=3):
    """Select best individual from tournament"""
    selected = random.sample(list(zip(pop, scores)), k)
    return max(selected, key=lambda x: x[1])[0]

def crossover(p1, p2):
    """Uniform crossover"""
    child = []
    for i in range(len(p1)):
        child.append(p1[i] if random.random() < 0.5 else p2[i])
    return child

def mutate(gene, mutation_rate=0.2):
    """Gaussian mutation"""
    mutated = []
    for i, val in enumerate(gene):
        if random.random() < mutation_rate:
            if i == 2 or i == 5 or i == 6 or i == 7:  # discrete params
                if i == 2: val = random.choice([6, 12, 18, 24, 36, 48])
                elif i == 5: val = random.choice([3, 6, 12, 18])
                elif i == 6: val = random.choice([1.0, 1.5, 2.0, 2.5, 3.0])
                elif i == 7: val = random.choice([0, 6, 12, 24, 48])
            else:  # continuous params (lth, sth)
                val = max(0.05, min(0.95, val + random.gauss(0, 0.05)))
                val = round(val, 2)
        mutated.append(val)
    return mutated

# Run GA with 100 generations
POP_SIZE = 30
GENERATIONS = 100
MUTATION_RATE = 0.2
ELITE_SIZE = 3

print(f"Starting GA 1h: pop={POP_SIZE} gen={GENERATIONS} mutation={MUTATION_RATE}", flush=True)

# Initialize population
# Gene: [etc_lth, etc_sth, etc_cd, trx_lth, trx_sth, trx_cd, lev, ts]
pop = []
for _ in range(POP_SIZE):
    gene = [
        round(random.choice([0.85, 0.88, 0.90, 0.92]), 2),  # ETC lth
        round(random.choice([0.12, 0.15, 0.20]), 2),         # ETC sth
        random.choice([6, 12, 18, 24]),                      # ETC cd
        round(random.choice([0.85, 0.88, 0.90]), 2),         # TRX lth
        round(random.choice([0.12, 0.15, 0.20, 0.25]), 2),   # TRX sth
        random.choice([3, 6, 12, 18]),                       # TRX cd
        random.choice([1.0, 1.5, 2.0, 2.5, 3.0]),           # leverage
        random.choice([0, 6, 12, 24, 48]),                   # time_stop
    ]
    pop.append(gene)

best_score = -999
best_gene = None
history = []

for gen in range(GENERATIONS):
    t0 = time.time()
    scores = []
    for gene in pop:
        score = eval_gene(gene)
        scores.append(score)

    # Sort by score
    sorted_pop = sorted(zip(pop, scores), key=lambda x: x[1], reverse=True)
    pop, scores = zip(*sorted_pop)
    pop = list(pop); scores = list(scores)

    best_gen_score = scores[0]
    best_gen_gene = pop[0]

    if best_gen_score > best_score:
        best_score = best_gen_score
        best_gene = best_gen_gene

    history.append({"gen": gen, "best": best_gen_score, "avg": sum(scores)/len(scores)})

    t1 = time.time()
    print(f"Gen {gen}: best={best_gen_score:.3f} avg={sum(scores)/len(scores):.3f} ({t1-t0:.1f}s)", flush=True)

    # Create new population
    new_pop = []
    # Elitism: keep top ELITE_SIZE
    for i in range(ELITE_SIZE):
        new_pop.append(pop[i])

    # Fill rest with tournament selection + crossover + mutation
    while len(new_pop) < POP_SIZE:
        p1 = tournament_selection(pop, scores)
        p2 = tournament_selection(pop, scores)
        child = crossover(p1, p2)
        child = mutate(child, MUTATION_RATE)
        new_pop.append(child)

    pop = new_pop

# Results
print(f"\n=== GA 1h COMPLETE ===", flush=True)
print(f"Best FULL sharpe: {best_score:.3f}", flush=True)
print(f"Best gene: {best_gene}", flush=True)
params = best_gene
print(f"Params: ETC(lth={params[0]}, sth={params[1]}, cd={params[2]}) TRX(lth={params[3]}, sth={params[4]}, cd={params[5]}) lev={params[6]} ts={params[7]}", flush=True)

# Save results
out = {"best_score": best_score, "best_gene": best_gene, "history": history,
       "params": {"etc_lth": best_gene[0], "etc_sth": best_gene[1], "etc_cd": best_gene[2],
                  "trx_lth": best_gene[3], "trx_sth": best_gene[4], "trx_cd": best_gene[5],
                  "leverage": best_gene[6], "time_stop": best_gene[7]},
       "meta": {"pop_size": POP_SIZE, "generations": GENERATIONS, "mutation_rate": MUTATION_RATE,
                "elite_size": ELITE_SIZE, "formula": FORMULA, "bars_per_year": BPY,
                "timeframe": "1h", "coins": ["ETC", "TRX"]}}
open("results/ga_1h_best.json", "w").write(json.dumps(out, indent=1))
print(f"saved results/ga_1h_best.json", flush=True)
