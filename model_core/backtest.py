import math
import torch


class MemeBacktest:
    def __init__(self):
        self.trade_size = 1000.0
        self.min_liq = 500000.0
        self.base_fee = 0.0060
        self.last_metrics = {"sharpe": 0.0, "max_dd": 0.0, "turnover": 0.0, "sortino": 0.0}

    def _sharpe(self, flat):
        if not flat:
            return 0.0
        n = len(flat)
        mean = sum(flat) / n
        var = sum((v - mean) ** 2 for v in flat) / max(n - 1, 1)
        std = math.sqrt(var) if var > 0 else 0.0
        if std < 1e-9:
            return 0.0
        return float(mean / std * math.sqrt(252.0))

    def _sortino(self, flat, target: float = 0.0):
        if not flat:
            return 0.0
        n = len(flat)
        mean = sum(flat) / n
        downside = [float(v) for v in flat if float(v) < target]
        if not downside:
            return 0.0
        dn = len(downside)
        d_mean = sum(downside) / dn
        var = sum((v - d_mean) ** 2 for v in downside) / max(dn - 1, 1)
        std = math.sqrt(var) if var > 0 else 0.0
        if std < 1e-9:
            var2 = sum((v - target) ** 2 for v in downside) / max(dn, 1)
            std = math.sqrt(var2) if var2 > 0 else 0.0
            if std < 1e-9:
                return 0.0
        return float(mean / std * math.sqrt(252.0))

    def _max_dd(self, flat):
        if not flat:
            return 0.0
        cumsum = []
        s = 0.0
        for v in flat:
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

    def evaluate(self, factors, raw_data, target_ret):
        liquidity = raw_data['liquidity']
        signal = torch.sigmoid(factors)
        is_safe = (liquidity > self.min_liq).float()
        position = (signal > 0.85).float() * is_safe
        impact_slippage = self.trade_size / (liquidity + 1e-9)
        impact_slippage = torch.clamp(impact_slippage, 0.0, 0.05)
        total_slippage_one_way = self.base_fee + impact_slippage
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        turnover = torch.abs(position - prev_pos)
        tx_cost = turnover * total_slippage_one_way
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - tx_cost
        cum_ret = net_pnl.sum(dim=1)
        big_drawdowns = (net_pnl < -0.05).float().sum(dim=1)
        score = cum_ret - (big_drawdowns * 2.0)
        activity = position.sum(dim=1)
        score = torch.where(activity < 5, torch.tensor(-10.0, device=score.device), score)
        final_fitness = torch.median(score)
        try:
            flat = net_pnl.detach().float().cpu().numpy().reshape(-1).tolist()
            t_flat = turnover.detach().float().cpu().numpy().reshape(-1).tolist()
            self.last_metrics = {
                "sharpe": self._sharpe(flat),
                "max_dd": self._max_dd(flat),
                "turnover": float(sum(t_flat) / len(t_flat)) if t_flat else 0.0,
                "sortino": self._sortino(flat),
            }
        except Exception:
            pass
        return final_fitness, cum_ret.mean().item()

    def evaluate_walk_forward(self, factors, raw_data, target_ret, n_splits: int = 3):
        if n_splits < 1:
            n_splits = 1
        try:
            t = int(factors.shape[1]) if hasattr(factors, 'shape') and len(factors.shape) >= 2 else int(len(factors))
        except Exception:
            return {"oos_scores": [], "mean_oos": 0.0, "splits": []}
        if t < 2 or n_splits > t:
            n_splits = max(1, min(n_splits, t // 2)) if t >= 2 else 1
        oos_scores = []
        splits = []
        for i in range(n_splits):
            start = int(t * i / n_splits)
            end = int(t * (i + 1) / n_splits)
            if end <= start:
                continue
            if hasattr(factors, 'ndim') and factors.ndim == 2:
                f_slice = factors[:, start:end]
                tr_slice = target_ret[:, start:end] if hasattr(target_ret, 'ndim') and target_ret.ndim == 2 else target_ret
            else:
                f_slice = factors[start:end]
                tr_slice = target_ret[start:end]
            raw_slice = {}
            for k, v in raw_data.items():
                try:
                    if hasattr(v, 'shape') and len(v.shape) >= 2 and v.shape[1] == t:
                        raw_slice[k] = v[:, start:end]
                    elif hasattr(v, 'shape') and len(v.shape) == 1 and v.shape[0] == t:
                        raw_slice[k] = v[start:end]
                    else:
                        raw_slice[k] = v
                except Exception:
                    raw_slice[k] = v
            fit, cum = self.evaluate(f_slice, raw_slice, tr_slice)
            val = float(fit.item()) if isinstance(fit, torch.Tensor) else float(fit)
            oos_scores.append(val)
            splits.append({"split": i, "start": start, "end": end, "fitness": val, "cum_ret": float(cum), "metrics": dict(self.last_metrics)})
        mean_oos = float(sum(oos_scores) / len(oos_scores)) if oos_scores else 0.0
        return {"oos_scores": oos_scores, "mean_oos": mean_oos, "splits": splits}


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk-forward", action="store_true", dest="walk_forward")
    parser.add_argument("--n-splits", type=int, default=3, dest="n_splits")
    args = parser.parse_args()
    if args.walk_forward:
        torch.manual_seed(0)
        n, t = 4, 30
        factors = torch.randn(n, t)
        raw_data = {"liquidity": torch.abs(torch.randn(n, t)) * 1e6 + 1e6}
        target_ret = torch.randn(n, t) * 0.02
        bt = MemeBacktest()
        res = bt.evaluate_walk_forward(factors, raw_data, target_ret, n_splits=args.n_splits)
        print(json.dumps(res, indent=2))
    else:
        torch.manual_seed(0)
        n, t = 4, 30
        factors = torch.randn(n, t)
        raw_data = {"liquidity": torch.abs(torch.randn(n, t)) * 1e6 + 1e6}
        target_ret = torch.randn(n, t) * 0.02
        bt = MemeBacktest()
        fit, cum = bt.evaluate(factors, raw_data, target_ret)
        out = {"fitness": float(fit.item()) if isinstance(fit, torch.Tensor) else float(fit), "cum_ret": float(cum), "metrics": bt.last_metrics}
        print(json.dumps(out, indent=2))
