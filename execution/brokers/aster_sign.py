"""Aster V3 EIP-712 signing module (P4 Phase 3).

V3 auth: every signed request carries signer + microsecond nonce + signature.
Procedure: urlencode(business params + signer + nonce) into EIP-712
Message.msg, sign with the agent private key. Domain:
name=AsterSignTransaction, version=1, chainId=1666,
verifyingContract=0x0000...0000.
"""
from __future__ import annotations

import threading
import time
import urllib.parse

_lock = threading.Lock()
_last_ms = 0
_counter = 0

DOMAIN_NAME = "AsterSignTransaction"
DOMAIN_VERSION = "1"
DOMAIN_CHAIN_ID = 1666
DOMAIN_CONTRACT = "0x0000000000000000000000000000000000000000"


def micro_nonce() -> str:
    """Monotonic microsecond nonce (unique per agent address)."""
    global _last_ms, _counter
    with _lock:
        ms = int(time.time() * 1000)
        if ms == _last_ms:
            _counter += 1
        else:
            _last_ms = ms
            _counter = 0
        return str(ms * 1_000_000 + _counter)


def build_typed_data(query_string: str) -> dict:
    return {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            "Message": [{"name": "msg", "type": "string"}],
        },
        "primaryType": "Message",
        "domain": {
            "name": DOMAIN_NAME,
            "version": DOMAIN_VERSION,
            "chainId": DOMAIN_CHAIN_ID,
            "verifyingContract": DOMAIN_CONTRACT,
        },
        "message": {"msg": query_string},
    }


def _encode_typed_data(typed: dict):
    import eth_account.messages as _msgs

    if hasattr(_msgs, "encode_structured_data"):  # eth-account < 0.13
        return _msgs.encode_structured_data(typed)
    # eth-account >= 0.13 renamed to encode_typed_data (full_message kw)
    return _msgs.encode_typed_data(full_message=typed)


def sign_query_string(query_string: str, private_key: str) -> str:
    from eth_account import Account

    typed = build_typed_data(query_string)
    signed = Account.sign_message(_encode_typed_data(typed), private_key=private_key)
    return signed.signature.hex()


class AsterSigner:
    """Holds V3 identities (user + signer) and signs request fields."""

    def __init__(self, user_address: str, signer_address: str, private_key: str):
        if not user_address or not signer_address or not private_key:
            raise ValueError("AsterSigner needs user_address, signer_address, private_key")
        self.user = user_address
        self.signer = signer_address
        self._priv = private_key

    def sign_fields(self, fields: dict | None = None) -> dict:
        """Return fields + signer + nonce + signature (ready for query/body)."""
        base = dict(fields or {})
        base["signer"] = self.signer
        base["nonce"] = micro_nonce()
        qs = urllib.parse.urlencode(base)
        base["signature"] = sign_query_string(qs, self._priv)
        return base

    def build_url(self, host: str, path: str, fields: dict | None = None) -> str:
        signed = self.sign_fields(fields)
        return f"{host}{path}?{urllib.parse.urlencode(signed)}"
