import math
VENUE_TAKER_FEE = {"solana": 0.0060, "hyperliquid": 0.00045, "aster": 0.00040}
import torch


class MemeBacktest:
    def __init__(self, venue: str = "solana", leverage: float = 1.0,
                 short_enabled: bool = False, fee_override: float | None = None,
                 funding_override: float | None = None,
                 long_th: float = 0.85, short_th: float = 0.15,
                 cooldown_bars: int = 0, bars_per_year: float = 2190.0,
                 stop_loss: float | None = None, time_stop: int = 0,
                 take_profit: float | None = None,
                 vol_target: float | None = None, vol_window: int = 24):
        """Backtest engine. venue/fee/funding only affect accounting; short_enabled
        adds short positions when signal < short_th. Defaults preserve Solana spot behavior.
        long_th/short_th: entry thresholds on sigmoid(signal). cooldown_bars: min hold.
        bars_per_year: annualization for Sharpe (4h=2190, 1h=8760, 15m=35040, daily=252)."""
        self.trade_size = 1000.0
        self.min_liq = 500000.0
        self.venue = (venue or "solana").lower()
        self.leverage = max(float(leverage or 1.0), 1.0)
        self.short_enabled = bool(short_enabled)
        self.long_th = float(long_th)
        self.short_th = float(short_th)
        self.quantile_filter = None  # set via set_quantile_filter(q): keep |logit| top-q fraction
        self.cooldown_bars = int(cooldown_bars)
        self.bars_per_year = float(bars_per_year)
        self.stop_loss = None if stop_loss is None else float(stop_loss)
        self.time_stop = int(time_stop)
        self.take_profit = None if take_profit is None else float(take_profit)
        self.vol_target = None if vol_target is None else float(vol_target)
        self.vol_window = int(vol_window)
        if fee_override is not None:
            self.base_fee = float(fee_override)
        else:
            self.base_fee = float(VENUE_TAKER_FEE.get(self.venue, 0.0060))
        self.default_funding_rate = (
            float(funding_override) if funding_override is not None else 0.0
        )
        self.last_metrics = {"sharpe": 0.0, "max_dd": 0.0, "turnover": 0.0, "sortino": 0.0,
                             "long_turnover": 0.0, "short_turnover": 0.0,
                             "funding_paid": 0.0, "venue": self.venue,
                             "leverage": self.leverage}

    def _sharpe(self, flat):
        if not flat:
            return 0.0
        n = len(flat)
        mean = sum(flat) / n
        var = sum((v - mean) ** 2 for v in flat) / max(n - 1, 1)
        std = math.sqrt(var) if var > 0 else 0.0
        if std < 1e-9:
            return 0.0
        return float(mean / std * math.sqrt(self.bars_per_year))

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
        return float(mean / std * math.sqrt(self.bars_per_year))

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

    def set_quantile_filter(self, q: float | None, side: str = "both"):
        """Only trade bars where |factor logit| is in top-q fraction (e.g. 0.10).
        side: 'both' | 'long' | 'short' (filter only that leg)."""
        self.quantile_filter = None if q is None else float(q)
        self.quantile_side = side

    def _apply_quantile(self, factors, long_sig, short_sig):
        if not self.quantile_filter:
            return long_sig, short_sig
        import torch as _t
        a = factors.detach().float().abs().reshape(-1)
        k = max(1, int(len(a) * self.quantile_filter))
        thr = _t.topk(a, k).values.min()
        mask = (factors.detach().float().abs() >= thr).float()
        side = getattr(self, "quantile_side", "both")
        if side == "long":
            return long_sig * mask, short_sig
        if side == "short":
            return long_sig, short_sig * mask
        return long_sig * mask, short_sig * mask

    def _vol_scale(self, target_ret):
        """Inverse-vol exposure scaling: scale = clamp(vol_target / trailing_vol, 0.2, 2.0).
        Returns scalar tensor broadcast over positions. None target -> 1.0."""
        if not self.vol_target:
            return 1.0
        r = target_ret.detach().float()
        if r.dim() == 1:
            r = r.unsqueeze(0)
        w = max(self.vol_window, 2)
        pad = r[:, :1].repeat(1, w - 1)
        rp = torch.cat([pad, r], dim=1)
        win = rp.unfold(1, w, 1)
        vol = win.std(dim=-1) + 1e-9
        scale = (self.vol_target / vol).clamp(0.2, 2.0)
        # align: scale[t] uses window ending at t -> shift +1 like execution
        scale = scale.roll(1, dims=1); scale[:, 0] = 1.0
        return scale

    def _apply_stops(self, long_pos, short_pos, target_ret):
        """Per-position stop-loss (adverse excursion) + time-stop (max hold).

        Walk each token row in python (fine for backtest scale): track entry,
        cumulative excursion since entry, bars held. Force flat when
        excursion <= -stop_loss or held >= time_stop (0 = disabled).
        """
        if not self.stop_loss and not self.time_stop and not self.take_profit:
            return long_pos, short_pos
        L = long_pos.detach().float().cpu().tolist()
        S = short_pos.detach().float().cpu().tolist()
        R = target_ret.detach().float().cpu().tolist()
        if not isinstance(R[0], list):
            R = [R]
            L = [L] if not isinstance(L[0], list) else L
            S = [S] if not isinstance(S[0], list) else S
        out_l, out_s = [], []
        for i in range(len(L)):
            rl, rs = [], []
            cur = 0.0  # +1 long, -1 short, 0 flat
            exc, held = 0.0, 0
            for t in range(len(L[i])):
                want = 1.0 if L[i][t] > 0.5 else (-1.0 if S[i][t] > 0.5 else 0.0)
                if want != cur:
                    cur, exc, held = want, 0.0, 0
                if cur != 0.0:
                    exc += cur * R[i][t] if t < len(R[i]) else 0.0
                    held += 1
                    kill = False
                    if self.stop_loss and exc <= -abs(self.stop_loss):
                        kill = True
                    if self.take_profit and exc >= abs(self.take_profit):
                        kill = True
                    if self.time_stop and held >= self.time_stop:
                        kill = True
                    if kill:
                        cur, exc, held = 0.0, 0.0, 0
                rl.append(1.0 if cur > 0 else 0.0)
                rs.append(1.0 if cur < 0 else 0.0)
            out_l.append(rl); out_s.append(rs)
        import torch as _t
        dev = long_pos.device
        return _t.tensor(out_l, device=dev), _t.tensor(out_s, device=dev)

    def _apply_cooldown(self, long_sig, short_sig):
        """Min-hold: after a flip, freeze position for cooldown_bars (per row, python loop)."""
        if self.cooldown_bars <= 0:
            return long_sig, short_sig
        L = long_sig.detach().float().cpu().tolist()
        S = short_sig.detach().float().cpu().tolist()
        n = len(L)
        out_l, out_s = [], []
        for i in range(n):
            rl, rs = [], []
            cur_l, cur_s, lock = 0.0, 0.0, 0
            for t in range(len(L[0])):
                if lock > 0:
                    lock -= 1
                else:
                    nl, ns = L[i][t], S[i][t]
                    if (nl, ns) != (cur_l, cur_s):
                        cur_l, cur_s, lock = nl, ns, self.cooldown_bars
                rl.append(cur_l); rs.append(cur_s)
            out_l.append(rl); out_s.append(rs)
        import torch as _t
        dev = long_sig.device
        return _t.tensor(out_l, device=dev), _t.tensor(out_s, device=dev)

    def evaluate(self, factors, raw_data, target_ret, funding_rate=None):
        """Evaluate factor PnL. venue/leverage/short/funding configurable via
        constructor; signature backward-compatible (funding_rate arg optional)."""
        liquidity = raw_data['liquidity']
        signal = torch.sigmoid(factors)
        is_safe = (liquidity > self.min_liq).float()
        long_sig = (signal > self.long_th).float() * is_safe
        short_sig = ((signal < self.short_th).float() * is_safe) if self.short_enabled else torch.zeros_like(long_sig)
        long_sig, short_sig = self._apply_quantile(factors, long_sig, short_sig)
        long_pos, short_pos = self._apply_cooldown(long_sig, short_sig)
        long_pos, short_pos = self._apply_stops(long_pos, short_pos, target_ret)
        exposure_scale = self._vol_scale(target_ret)
        long_pos, short_pos = long_pos * exposure_scale, short_pos * exposure_scale
        # execution lag: signal at bar t fills at NEXT bar open -> shift positions +1.
        # gross uses target[t] captured by pos[t-1]; first bar flat
        long_pos = long_pos.roll(1, dims=1); long_pos[:, 0] = 0
        short_pos = short_pos.roll(1, dims=1); short_pos[:, 0] = 0
        lev = self.leverage
        impact_slippage = self.trade_size / (liquidity + 1e-9)
        impact_slippage = torch.clamp(impact_slippage, 0.0, 0.05)
        total_slippage_one_way = self.base_fee + impact_slippage
        prev_long = torch.roll(long_pos, 1, dims=1)
        prev_long[:, 0] = 0
        prev_short = torch.roll(short_pos, 1, dims=1)
        prev_short[:, 0] = 0
        long_turn = torch.abs(long_pos - prev_long)
        short_turn = torch.abs(short_pos - prev_short)
        turnover = long_turn + short_turn
        tx_cost = turnover * total_slippage_one_way
        gross_pnl = (long_pos - short_pos) * target_ret * lev
        fund = raw_data.get('funding', None)
        if fund is None:
            fr = self.default_funding_rate if funding_rate is None else float(funding_rate)
            funding_cost = (long_pos - short_pos) * float(fr) * lev
            funding_paid_total = float(funding_cost.detach().float().sum().item())
        else:
            funding_cost = (long_pos - short_pos) * fund * lev
            try:
                funding_paid_total = float(funding_cost.detach().float().sum().item())
            except Exception:
                funding_paid_total = 0.0
        net_pnl = gross_pnl - tx_cost * lev - funding_cost
        cum_ret = net_pnl.sum(dim=1)
        dd_th = -0.05 * max(self.leverage, 1.0)  # scale w/ leverage so 2x isn't over-penalized
        big_drawdowns = (net_pnl < dd_th).float().sum(dim=1)
        score = cum_ret - (big_drawdowns * 0.5)
        activity = (long_pos + short_pos).sum(dim=1)
        score = torch.where(activity < 5, torch.tensor(-10.0, device=score.device), score)
        final_fitness = torch.median(score)
        try:
            flat = net_pnl.detach().float().cpu().numpy().reshape(-1).tolist()
            t_flat = turnover.detach().float().cpu().numpy().reshape(-1).tolist()
            lt_flat = long_turn.detach().float().cpu().numpy().reshape(-1).tolist()
            st_flat = short_turn.detach().float().cpu().numpy().reshape(-1).tolist()
            self.last_metrics = {
                "sharpe": self._sharpe(flat),
                "max_dd": self._max_dd(flat),
                "turnover": float(sum(t_flat) / len(t_flat)) if t_flat else 0.0,
                "sortino": self._sortino(flat),
                "long_turnover": float(sum(lt_flat) / len(lt_flat)) if lt_flat else 0.0,
                "short_turnover": float(sum(st_flat) / len(st_flat)) if st_flat else 0.0,
                "funding_paid": funding_paid_total,
                "venue": self.venue,
                "leverage": self.leverage,
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
