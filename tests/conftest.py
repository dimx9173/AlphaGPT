import sys
import types
import pytest
import torch


def _mock_solders():
    for name in [
        "solders",
        "solders.keypair",
        "solders.pubkey",
        "solders.transaction",
        "solana",
        "solana.rpc",
        "solana.rpc.api",
        "solana.rpc.async_api",
        "solana.rpc.commitment",
        "solana.rpc.types",
    ]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    for n in ["solana.rpc.commitment"]:
        m = sys.modules[n]
        if not hasattr(m, "Confirmed"):
            m.Confirmed = "confirmed"
    for n in ["solana.rpc.types"]:
        m = sys.modules[n]
        if not hasattr(m, "TokenAccountOpts"):
            m.TokenAccountOpts = type("TokenAccountOpts", (), {"__init__": lambda self, **kw: None})
        if not hasattr(m, "TxOpts"):
            m.TxOpts = type("TxOpts", (), {"__init__": lambda self, **kw: None})
    for n in ["solana.rpc.async_api"]:
        m = sys.modules[n]
        if not hasattr(m, "AsyncClient"):
            m.AsyncClient = type("AsyncClient", (), {"__init__": lambda self, *a, **kw: None})
    if "solders" in sys.modules and hasattr(sys.modules["solders"], "keypair"):
        pass
    try:
        from solders.transaction import VersionedTransaction  # noqa: F401
    except Exception:
        mod = sys.modules["solders.transaction"]
        class _Dummy:
            @staticmethod
            def from_bytes(b): return _Dummy()
            def to_bytes(self): return b""
            @staticmethod
            def populate(msg, sigs): return _Dummy()
            @property
            def message(self):
                class M:
                    def to_bytes(self): return b""
                return M()
        mod.VersionedTransaction = _Dummy
    try:
        from solders.keypair import Keypair  # noqa: F401
    except Exception:
        mod = sys.modules["solders.keypair"]
        class _KP:
            @staticmethod
            def from_bytes(b): return _KP()
            def pubkey(self): return None
            def sign_message(self, m): return b"\x00"*64
        mod.Keypair = _KP
    try:
        from solders.pubkey import Pubkey  # noqa: F401
    except Exception:
        mod = sys.modules["solders.pubkey"]
        mod.Pubkey = type("Pubkey", (), {"from_string": staticmethod(lambda s: None)})


_mock_solders()


@pytest.fixture
def sample_feat_tensor():
    torch.manual_seed(0)
    return torch.randn(4, 6, 32)


@pytest.fixture
def sample_raw_data():
    torch.manual_seed(1)
    n, t = 4, 32
    close = torch.abs(torch.randn(n, t)) + 10
    open_ = close * (0.98 + 0.04 * torch.rand(n, t))
    high = torch.maximum(close, open_) + torch.rand(n, t)
    low = torch.minimum(close, open_) - torch.rand(n, t) * 0.5
    low = torch.clamp(low, min=1.0)
    volume = torch.abs(torch.randn(n, t)) * 1e6 + 1e5
    liquidity = torch.abs(torch.randn(n, t)) * 1e6 + 1e6
    fdv = liquidity * (2 + torch.rand(n, t))
    return {
        "close": close,
        "open": open_,
        "high": high,
        "low": low,
        "volume": volume,
        "liquidity": liquidity,
        "fdv": fdv,
    }


@pytest.fixture
def tmp_state_file(tmp_path):
    return str(tmp_path / "portfolio_state.json")


@pytest.fixture
def mocked_env(monkeypatch):
    monkeypatch.setenv("DB_USER", "test")
    monkeypatch.setenv("DB_PASSWORD", "test")
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_NAME", "test")
    monkeypatch.setenv("WALLET_PRIVATE_KEY", "test_key")
    monkeypatch.setenv("HELIUS_RPC_URL", "http://localhost:9999")
    monkeypatch.setenv("JUPITER_API_KEY", "test")
    return True
