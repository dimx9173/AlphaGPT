import torch

@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)

@torch.jit.script
def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y

@torch.jit.script
def _op_jump(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    z = (x - mean) / std
    return torch.relu(z - 3.0)

@torch.jit.script
def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)

@torch.jit.script
def _op_zscore(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    z = (x - mean) / std
    return torch.nan_to_num(z, nan=0.0, posinf=5.0, neginf=-5.0)

@torch.jit.script
def _op_rank(x: torch.Tensor) -> torch.Tensor:
    r = torch.argsort(torch.argsort(x, dim=1), dim=1).float()
    n = float(x.shape[1])
    if n <= 1:
        return torch.zeros_like(x)
    normed = r / (n - 1.0)
    return torch.nan_to_num(normed, nan=0.5, posinf=1.0, neginf=0.0)

@torch.jit.script
def _op_ts_rank(x: torch.Tensor) -> torch.Tensor:
    B = x.shape[0]
    T = x.shape[1]
    if T < 2:
        return torch.zeros_like(x)
    d = 20
    if d > T:
        d = T
    pad = torch.zeros((B, d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    mean_w = windows.mean(dim=-1)
    std_w = windows.std(dim=-1) + 1e-6
    z = (x - mean_w) / std_w
    return torch.nan_to_num(torch.clamp(z, -5.0, 5.0), nan=0.0, posinf=5.0, neginf=-5.0)

@torch.jit.script
def _op_delta(x: torch.Tensor) -> torch.Tensor:
    d = _ts_delay(x, 1)
    res = x - d
    return torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)

def _op_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xm = x - x.mean(dim=1, keepdim=True)
    ym = y - y.mean(dim=1, keepdim=True)
    num = (xm * ym).mean(dim=1, keepdim=True)
    den = x.std(dim=1, keepdim=True) * y.std(dim=1, keepdim=True) + 1e-6
    corr = num / den
    corr = corr.expand_as(x)
    return torch.nan_to_num(torch.clamp(corr, -1.0, 1.0), nan=0.0, posinf=1.0, neginf=-1.0)

OPS_CONFIG = [
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6), 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', torch.abs, 1),
    ('SIGN', torch.sign, 1),
    ('GATE', _op_gate, 3),
    ('JUMP', _op_jump, 1),
    ('DECAY', _op_decay, 1),
    ('DELAY1', lambda x: _ts_delay(x, 1), 1),
    ('MAX3', lambda x: torch.max(x, torch.max(_ts_delay(x,1), _ts_delay(x,2))), 1),
    ('ZSCORE', _op_zscore, 1),
    ('RANK', _op_rank, 1),
    ('TS_RANK', _op_ts_rank, 1),
    ('DELTA', _op_delta, 1),
    ('CORR', _op_corr, 2),
]
