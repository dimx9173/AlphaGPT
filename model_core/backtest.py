import math
import torch

class MemeBacktest:
    def __init__(self):
        self.trade_size = 1000.0
        self.min_liq = 500000.0
        self.base_fee = 0.0060
        self.last_metrics = {"sharpe": 0.0, "max_dd": 0.0, "turnover": 0.0}

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
            }
        except Exception:
            pass
        return final_fitness, cum_ret.mean().item()