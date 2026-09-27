"""Grammar masks for 12-token AlphaGPT formulas (RPN validity)."""
from __future__ import annotations
import torch
from model_core.vocab import ADVANCED_VOCAB
from model_core.ops import OPS_CONFIG

FEATURE_COUNT=ADVANCED_VOCAB.feature_count
OP_ARITY={FEATURE_COUNT+i:int(cfg[2]) for i,cfg in enumerate(OPS_CONFIG)}
OP_TOKENS=set(OP_ARITY)
ALL_TOKENS=FEATURE_COUNT+len(OPS_CONFIG)

def _can_finish(depth:int, slots:int)->bool:
    if slots==0: return depth==1
    if depth>12: return False
    if depth<1: return False
    if depth>slots+1: return False
    if _can_finish(depth+1,slots-1): return True
    for arity in OP_ARITY.values():
        if depth>=arity and _can_finish(depth-arity+1,slots-1): return True
    return False

def valid_token_mask(depth:int, position:int, max_len:int=12)->torch.Tensor:
    slots=max_len-position-1
    mask=torch.zeros(ALL_TOKENS,dtype=torch.bool)
    for token in range(FEATURE_COUNT):
        if _can_finish(depth+1,slots): mask[token]=True
    for token,arity in OP_ARITY.items():
        if depth>=arity and _can_finish(depth-arity+1,slots): mask[token]=True
    return mask

def update_depth(depth:int, token:int)->int:
    if token<FEATURE_COUNT: return depth+1
    if token not in OP_ARITY: raise ValueError(f"unknown token {token}")
    if depth<OP_ARITY[token]: raise ValueError("stack underflow")
    return depth-OP_ARITY[token]+1

def is_valid(formula)->bool:
    depth=0
    try:
        for i,t in enumerate(formula):
            t=int(t)
            if not bool(valid_token_mask(depth,i,len(formula))[t]): return False
            depth=update_depth(depth,t)
    except Exception: return False
    return depth==1
