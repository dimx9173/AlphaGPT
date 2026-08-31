import json
import os
import pandas as pd
import sqlalchemy
from dotenv import load_dotenv
from solders.pubkey import Pubkey
from solana.rpc.api import Client

load_dotenv()

class DashboardService:
    def __init__(self):
        db_user = os.getenv("DB_USER", "postgres")
        db_pass = os.getenv("DB_PASSWORD", "password")
        db_host = os.getenv("DB_HOST", "localhost")
        db_name = os.getenv("DB_NAME", "crypto_quant")
        self.engine = sqlalchemy.create_engine(f"postgresql://{db_user}:{db_pass}@{db_host}:5432/{db_name}")
        rpc_url = os.getenv("QUICKNODE_RPC_URL", "https://api.mainnet-beta.solana.com")
        self.rpc = Client(rpc_url)
        self.wallet_addr = self._get_wallet_address()

    def _get_wallet_address(self):
        try:
            from execution.config import ExecutionConfig
            return ExecutionConfig.get_wallet_address()
        except (ValueError, TypeError, json.JSONDecodeError):
            return "Unknown"

    def get_wallet_balance(self):
        try:
            resp = self.rpc.get_balance(Pubkey.from_string(self.wallet_addr))
            return resp.value / 1e9
        except Exception as e:
            return 0.0

    def load_portfolio(self):
        try:
            with open("portfolio_state.json", "r") as f:
                data = json.load(f)
                if not data: return pd.DataFrame()
                
                df = pd.DataFrame(data.values())
                # 计算当前预估 PnL
                if 'highest_price' in df.columns and 'entry_price' in df.columns:
                    df['pnl_pct'] = (df['highest_price'] - df['entry_price']) / df['entry_price']
                return df
        except FileNotFoundError:
            return pd.DataFrame()

    def load_strategy_info(self):
        try:
            with open("best_meme_strategy.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"formula": "Not Trained Yet"}

    def get_market_overview(self, limit=50):
        query = f"""
        SELECT t.symbol, o.address, o.close, o.volume, o.liquidity, o.fdv, o.time
        FROM ohlcv o
        JOIN tokens t ON o.address = t.address
        WHERE o.time = (SELECT MAX(time) FROM ohlcv)
        ORDER BY o.liquidity DESC
        LIMIT {limit}
        """
        try:
            return pd.read_sql(query, self.engine)
        except Exception:
            return pd.DataFrame()
    
    def get_recent_logs(self, n=50):
        log_file = "strategy.log"
        if not os.path.exists(log_file): return []
        
        with open(log_file, "r") as f:
            lines = f.readlines()
            return lines[-n:]

    def get_training_metrics(self):
        for path in ("training_history.json", "metrics.jsonl"):
            if not os.path.exists(path):
                continue
            try:
                if path.endswith(".jsonl"):
                    rows = []
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line=line.strip()
                            if not line: continue
                            try: rows.append(json.loads(line))
                            except Exception: continue
                    if rows:
                        train_rows = [r for r in rows if r.get("kind")=="train_step"]
                        latest = train_rows[-1] if train_rows else rows[-1]
                        return {"source": path, "latest": latest, "history": rows[-200:]}
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        data=json.load(f)
                    if isinstance(data, dict) and data.get("step"):
                        def _last(k):
                            v=data.get(k)
                            return v[-1] if isinstance(v,list) and v else None
                        return {"source": path, "latest": {"step": data["step"][-1], "avg_reward": _last("avg_reward"), "best_score": _last("best_score"), "sharpe": _last("sharpe"), "max_dd": _last("max_dd"), "turnover": _last("turnover")}, "history": data}
            except Exception:
                continue
        return {"source": None, "latest": None, "history": None}

    def get_pipeline_status(self):
        info={"last_updated": None, "token_count": None, "ohlcv_count": None, "checkpoint_exists": os.path.exists("checkpoint.json") or os.path.exists("data_pipeline/checkpoint.json")}
        try:
            df=pd.read_sql("SELECT MAX(last_updated) as lu, COUNT(*) as cnt FROM tokens", self.engine)
            if not df.empty:
                info["last_updated"]=str(df.iloc[0]["lu"]) if pd.notna(df.iloc[0]["lu"]) else None
                info["token_count"]=int(df.iloc[0]["cnt"]) if pd.notna(df.iloc[0]["cnt"]) else 0
        except Exception:
            pass
        try:
            df2=pd.read_sql("SELECT COUNT(*) as c FROM ohlcv", self.engine)
            if not df2.empty:
                info["ohlcv_count"]=int(df2.iloc[0]["c"])
        except Exception:
            pass
        for p in ("metrics.jsonl","training_history.json","best_meme_strategy.json"):
            info[p]=os.path.exists(p)
        return info

if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", action="store_true", help="print training metrics json")
    ap.add_argument("--status", action="store_true", help="print pipeline status json")
    ap.add_argument("--jsonl", type=str, default=None, help="path to metrics.jsonl to dump")
    args = ap.parse_args()
    svc = DashboardService()
    if args.metrics:
        print(json.dumps(svc.get_training_metrics(), indent=2, default=str))
    elif args.status:
        print(json.dumps(svc.get_pipeline_status(), indent=2, default=str))
    elif args.jsonl:
        j = args.jsonl
        if os.path.exists(j):
            with open(j) as f: sys.stdout.write(f.read())
        else:
            print(f"not found: {j}", file=sys.stderr); sys.exit(1)
    else:
        print(json.dumps({"metrics": svc.get_training_metrics(), "status": svc.get_pipeline_status()}, indent=2, default=str))