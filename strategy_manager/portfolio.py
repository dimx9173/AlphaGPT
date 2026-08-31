import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, asdict
from typing import Dict
from loguru import logger

try:
    import fcntl as _fcntl  # type: ignore
    _HAS_FCNTL = True
except ImportError:
    _fcntl = None  # type: ignore
    _HAS_FCNTL = False

try:
    from filelock import FileLock as _FileLock  # type: ignore
    _HAS_FILELOCK = True
except ImportError:
    _FileLock = None  # type: ignore
    _HAS_FILELOCK = False

@dataclass
class Position:
    token_address: str
    symbol: str
    entry_price: float     # 入场价格 (USD 或 SOL)
    entry_time: float      # 入场时间戳
    amount_held: float     # 当前持仓数量 (Token Units)
    initial_cost_sol: float # 初始投入 SOL
    highest_price: float   #以此计算回撤
    is_moonbag: bool = False # 是否已经翻倍出本，剩下的让利润奔跑

class PortfolioManager:
    def __init__(self, state_file="portfolio_state.json"):
        self.state_file = state_file
        self.positions: Dict[str, Position] = {}
        self.load_state()

    def add_position(self, token, symbol, price, amount, cost_sol):
        self.positions[token] = Position(
            token_address=token,
            symbol=symbol,
            entry_price=price,
            entry_time=time.time(),
            amount_held=amount,
            initial_cost_sol=cost_sol,
            highest_price=price
        )
        self.save_state()
        logger.info(f"[+] Position Added: {symbol} @ {price}")

    def update_price(self, token, current_price):
        if token in self.positions:
            pos = self.positions[token]
            if current_price > pos.highest_price:
                pos.highest_price = current_price
            self.save_state()

    def update_holding(self, token, new_amount):
        if token in self.positions:
            self.positions[token].amount_held = new_amount
            if new_amount <= 0:
                del self.positions[token]
            self.save_state()

    def close_position(self, token):
        if token in self.positions:
            logger.info(f"[+] Position Closed: {self.positions[token].symbol}")
            del self.positions[token]
            self.save_state()

    def get_open_count(self):
        return len(self.positions)

    def save_state(self):
        data = {k: asdict(v) for k, v in self.positions.items()}
        abs_state = os.path.abspath(self.state_file)
        dir_name = os.path.dirname(abs_state) or "."
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception:
            pass
        lock_path = self.state_file + ".lock"
        lock_fd = None
        file_lock = None
        tmp_fd = None
        tmp_path = None
        try:
            if _HAS_FCNTL:
                lock_dir = os.path.dirname(os.path.abspath(lock_path)) or "."
                try:
                    os.makedirs(lock_dir, exist_ok=True)
                except Exception:
                    pass
                lock_fd = open(lock_path, "a")
                _fcntl.flock(lock_fd, _fcntl.LOCK_EX)  # type: ignore
            elif _HAS_FILELOCK:
                file_lock = _FileLock(lock_path)  # type: ignore
                file_lock.acquire()
            else:
                pass  # no file locking available; write remains atomic via tempfile+replace but without inter-process mutual exclusion

            if os.path.exists(self.state_file):
                try:
                    shutil.copy2(self.state_file, self.state_file + ".bak")
                except Exception:
                    pass

            tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_name)
            with os.fdopen(tmp_fd, "w") as f:
                tmp_fd = None
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            try:
                dir_fd = os.open(dir_name, os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except Exception:
                pass
            os.replace(tmp_path, self.state_file)
            tmp_path = None
            try:
                dir_fd = os.open(dir_name, os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except Exception:
                pass
        finally:
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except Exception:
                    pass
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            if _HAS_FCNTL and lock_fd is not None:
                try:
                    _fcntl.flock(lock_fd, _fcntl.LOCK_UN)  # type: ignore
                except Exception:
                    pass
                try:
                    lock_fd.close()
                except Exception:
                    pass
            elif _HAS_FILELOCK and file_lock is not None:
                try:
                    file_lock.release()
                except Exception:
                    pass

    def load_state(self):
        lock_path = self.state_file + ".lock"
        lock_fd = None
        file_lock = None
        try:
            if _HAS_FCNTL:
                lock_dir = os.path.dirname(os.path.abspath(lock_path)) or "."
                try:
                    os.makedirs(lock_dir, exist_ok=True)
                except Exception:
                    pass
                lock_fd = open(lock_path, "a")
                _fcntl.flock(lock_fd, _fcntl.LOCK_SH)  # type: ignore
            elif _HAS_FILELOCK and os.path.exists(self.state_file):
                file_lock = _FileLock(lock_path)  # type: ignore
                file_lock.acquire()
            else:
                pass  # no file locking available; read without lock but atomic replace guarantees no truncated JSON

            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.positions[k] = Position(**v)
            except FileNotFoundError:
                self.positions = {}
        finally:
            if _HAS_FCNTL and lock_fd is not None:
                try:
                    _fcntl.flock(lock_fd, _fcntl.LOCK_UN)  # type: ignore
                except Exception:
                    pass
                try:
                    lock_fd.close()
                except Exception:
                    pass
            elif _HAS_FILELOCK and file_lock is not None:
                try:
                    file_lock.release()
                except Exception:
                    pass
