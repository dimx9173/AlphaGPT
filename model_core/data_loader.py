import os
import pandas as pd
import torch
import sqlalchemy
from .config import ModelConfig
from .factors import FeatureEngineer

class CryptoDataLoader:
    def __init__(self):
        self.engine = sqlalchemy.create_engine(ModelConfig.DB_URL)
        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None
        self.addresses = []
        self.feat_tensor_train = None
        self.feat_tensor_valid = None
        self.target_ret_train = None
        self.target_ret_valid = None
        self.train_idx = None
        self.valid_idx = None

    def load_data(self, limit_tokens=500, train_ratio=None):
        """Load OHLCV data, compute features and 2-step forward returns.

        Args:
            limit_tokens: max tokens to load ordered by tokens table.
            train_ratio: if set (e.g. 0.8), split on time dim T_train=int(T*train_ratio)
                and expose train/valid views without mutating the full tensors.
                When None (default) keeps legacy behavior unchanged.

        Side effects:
            Sets feat_tensor (N,F,T), target_ret (N,T), and when train_ratio
            is not None also feat_tensor_train/valid, target_ret_train/valid,
            train_idx/valid_idx slicing the time dimension.
        """
        print("Loading data from SQL...")
        top_query = f"""
        SELECT address FROM tokens 
        LIMIT {limit_tokens} 
        """
        self.addresses = pd.read_sql(top_query, self.engine)['address'].tolist()
        if not self.addresses: raise ValueError("No tokens found.")
        addr_str = "'" + "','".join(self.addresses) + "'"
        data_query = f"""
        SELECT time, address, open, high, low, close, volume, liquidity, fdv
        FROM ohlcv
        WHERE address IN ({addr_str})
        ORDER BY time ASC
        """
        df = pd.read_sql(data_query, self.engine)
        def to_tensor(col):
            pivot = df.pivot(index='time', columns='address', values=col)
            pivot = pivot.reindex(columns=self.addresses)
            pivot = pivot.fillna(method='ffill').fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)
        self.raw_data_cache = {
            'open': to_tensor('open'),
            'high': to_tensor('high'),
            'low': to_tensor('low'),
            'close': to_tensor('close'),
            'volume': to_tensor('volume'),
            'liquidity': to_tensor('liquidity'),
            'fdv': to_tensor('fdv')
        }
        flag = os.getenv("USE_ADVANCED", "0") == "1"
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache, use_advanced=flag)
        op = self.raw_data_cache['open']
        t1 = torch.roll(op, -1, dims=1)
        t2 = torch.roll(op, -2, dims=1)
        self.target_ret = torch.log(t2 / (t1 + 1e-9))
        self.target_ret[:, -2:] = 0.0
        print(f"Data Ready. Shape: {self.feat_tensor.shape}")
        if train_ratio is not None:
            T = self.feat_tensor.shape[2]
            T_train = int(T * float(train_ratio))
            if T > 1:
                T_train = max(1, min(T - 1, T_train))
            self.train_idx = torch.arange(T_train)
            self.valid_idx = torch.arange(T_train, T)
            self.feat_tensor_train = self.feat_tensor[:, :, :T_train]
            self.feat_tensor_valid = self.feat_tensor[:, :, T_train:]
            self.target_ret_train = self.target_ret[:, :T_train]
            self.target_ret_valid = self.target_ret[:, T_train:]
        else:
            self.train_idx = None
            self.valid_idx = None
            self.feat_tensor_train = None
            self.feat_tensor_valid = None
            self.target_ret_train = None
            self.target_ret_valid = None
