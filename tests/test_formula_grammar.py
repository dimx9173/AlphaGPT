import random
import torch
from research.formula_grammar import (
    FEATURE_COUNT, valid_token_mask, update_depth, is_valid,
)

def test_grammar_mask_sampled_sequences_are_valid():
    rng=random.Random(7)
    for _ in range(500):
        depth=0; seq=[]
        for pos in range(12):
            choices=torch.nonzero(valid_token_mask(depth,pos,12)).flatten().tolist()
            token=rng.choice(choices); seq.append(token)
            depth=update_depth(depth,token)
        assert depth == 1
        assert is_valid(seq)

def test_grammar_rejects_stack_underflow_and_nonfinal_stack():
    assert not is_valid([FEATURE_COUNT+2, 0])  # binary op with empty stack
    assert not is_valid([0] * 12)              # 12 values, no reducing op
