import json
import os
import math
from datetime import datetime, timezone

try:
    import torch
except Exception:
    torch = None


def _to_list(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy().tolist() if x.numel() > 0 else []
    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass
    return list(x) if x is not None else []


def sharpe_ratio(returns, eps: float = 1e-9) -> float:
    vals = _to_list(returns)
    if not vals:
        return 0.0
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std < eps:
        return 0.0
    return float(mean / std * math.sqrt(252.0))


def max_drawdown(pnl_series) -> float:
    vals = _to_list(pnl_series)
    if not vals:
        return 0.0
    cumsum = []
    s = 0.0
    for v in vals:
        s += float(v)
        cumsum.append(s)
    peak = cumsum[0]
    max_dd = 0.0
    for v in cumsum:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def turnover_rate(turnover_tensor) -> float:
    if turnover_tensor is None:
        return 0.0
    vals = _to_list(turnover_tensor)
    if not vals:
        return 0.0
    if isinstance(vals, list) and vals and isinstance(vals[0], list):
        flat = [float(x) for row in vals for x in row]
        return float(sum(flat) / len(flat)) if flat else 0.0
    return float(sum(float(x) for x in vals) / len(vals)) if vals else 0.0


def compute_backtest_metrics(net_pnl, turnover=None) -> dict:
    flat = _to_list(net_pnl)
    if flat and isinstance(flat[0], list):
        flat = [float(x) for row in flat for x in row]
    else:
        flat = [float(x) for x in flat] if flat else []
    sharpe = sharpe_ratio(flat)
    mdd = max_drawdown(flat)
    to = turnover_rate(turnover) if turnover is not None else 0.0
    return {"sharpe": float(sharpe), "max_dd": float(mdd), "turnover": float(to)}


def append_metrics_jsonl(path: str, record: dict) -> None:
    record = dict(record)
    record.setdefault("ts", datetime.now(timezone.utc).isoformat())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_metrics_jsonl(path: str, limit: int = 500) -> list:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    if limit and len(out) > limit:
        return out[-limit:]
    return out


def summarize_training_history(history: dict) -> dict:
    if not history or not history.get("step"):
        return {"steps": 0, "latest_sharpe": None, "latest_max_dd": None, "latest_turnover": None}
    n = len(history["step"])
    return {
        "steps": n,
        "latest_sharpe": (history.get("sharpe") or [None])[-1],
        "latest_max_dd": (history.get("max_dd") or [None])[-1],
        "latest_turnover": (history.get("turnover") or [None])[-1],
        "best_score": (history.get("best_score") or [None])[-1],
        "avg_reward": (history.get("avg_reward") or [None])[-1],
    }
