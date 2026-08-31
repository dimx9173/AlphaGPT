import asyncio
import json
import os
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from loguru import logger
from .config import ExecutionConfig


class QuickNodeClient:
    def __init__(self, client=None):
        self.client = client or AsyncClient(ExecutionConfig.RPC_URL, commitment=Confirmed)

    async def get_balance(self):
        try:
            resp = await self.client.get_balance(ExecutionConfig.get_payer_keypair().pubkey())
            return resp.value / 1e9
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0

    async def get_token_balance(self, mint_address: str) -> int:
        try:
            from solders.pubkey import Pubkey
            from solana.rpc.types import TokenAccountOpts
            wallet_pubkey = Pubkey.from_string(ExecutionConfig.get_wallet_address())
            mint_pubkey = Pubkey.from_string(mint_address)
            opts = TokenAccountOpts(
                program_id=Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"),
                mint=mint_pubkey,
            )
            resp = await self.client.get_token_accounts_by_owner_json_parsed(wallet_pubkey, opts)
            total = 0
            if resp.value:
                for acc in resp.value:
                    amount_str = acc.account.data.parsed["info"]["tokenAmount"]["amount"]
                    total += int(amount_str)
            return total
        except Exception as e:
            logger.error(f"Failed to get token balance for {mint_address}: {e}")
            return 0

    async def simulate_transaction(self, txn) -> bool:
        try:
            resp = await self.client.simulate_transaction(txn)
            val = getattr(resp, "value", None)
            if val is not None:
                err = getattr(val, "err", None)
                if err is not None:
                    logger.error(f"Simulate failed: {err}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Simulate error: {e}")
            return False

    async def send_and_confirm(self, txn, max_retries=3):
        receipt_path = os.getenv("RECEIPT_PATH", "data/receipts.jsonl")
        sig_str = None
        try:
            try:
                bh = await self.client.get_latest_blockhash()
                _ = bh
            except Exception:
                pass
            if not await self.simulate_transaction(txn):
                logger.error("Transaction simulate failed, aborting send")
                return None
            signature = await self.client.send_transaction(txn, opts=None)
            logger.info(f"Transaction Sent: {signature.value}")
            sig_str = str(signature.value)
            for attempt in range(max_retries):
                try:
                    await self.client.confirm_transaction(signature.value)
                    logger.success(f"Transaction Confirmed: https://solscan.io/tx/{sig_str}")
                    break
                except Exception as ce:
                    if attempt == max_retries - 1:
                        raise ce
                    await asyncio.sleep(2)
            try:
                statuses = await self.client.get_signature_statuses([sig_str])
                _ = statuses
            except Exception:
                pass
            try:
                os.makedirs(os.path.dirname(os.path.abspath(receipt_path)) or ".", exist_ok=True)
                with open(receipt_path, "a") as f:
                    f.write(json.dumps({"sig": sig_str, "status": "confirmed"}) + "\n")
            except Exception:
                pass
            return sig_str
        except Exception as e:
            logger.error(f"Transaction Failed: {e}")
            if sig_str:
                try:
                    statuses = await self.client.get_signature_statuses([sig_str])
                    _ = statuses
                except Exception:
                    pass
            return None

    async def close(self):
        await self.client.close()
