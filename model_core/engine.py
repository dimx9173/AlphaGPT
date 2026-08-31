import os
import json as _json
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
import json
from datetime import datetime, timezone
from loguru import logger

from .config import ModelConfig
from .data_loader import CryptoDataLoader
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import MemeBacktest
from .metrics import append_metrics_jsonl

class AlphaEngine:
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5, v_coef=None, e_coef=None, clip_norm=None):
        """
        Initialize AlphaGPT training engine.
        
        Args:
            use_lord_regularization: Enable Low-Rank Decay (LoRD) regularization
            lord_decay_rate: Strength of LoRD regularization
            lord_num_iterations: Number of Newton-Schulz iterations per step
        """
        self.loader = CryptoDataLoader()
        self.loader.load_data()
        
        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        
        self.v_coef = v_coef if v_coef is not None else ModelConfig.V_COEF
        self.e_coef = e_coef if e_coef is not None else ModelConfig.E_COEF
        self.clip_norm = clip_norm if clip_norm is not None else ModelConfig.CLIP_NORM
        
        # Standard optimizer
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        # Low-Rank Decay regularizer
        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["q_proj", "k_proj", "attention", "qk_norm"]
            )
            self.rank_monitor = StableRankMonitor(
                self.model,
                target_keywords=["q_proj", "k_proj"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None
        
        self.vm = StackVM()
        self.bt = MemeBacktest()
        
        self.best_score = -float('inf')
        self.best_formula = None
        self.training_history = {
            'step': [],
            'avg_reward': [],
            'best_score': [],
            'stable_rank': [],
            'sharpe': [],
            'max_dd': [],
            'turnover': [],
        }

    def train(self):
        print("🚀 Starting Meme Alpha Mining with LoRD Regularization..." if self.use_lord else "🚀 Starting Meme Alpha Mining...")
        if self.use_lord:
            print(f"   LoRD Regularization enabled")
            print(f"   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")
        
        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
        
        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            
            log_probs = []
            tokens_list = []
            values = []
            entropies = []
            
            for _ in range(ModelConfig.MAX_FORMULA_LEN):
                logits, value, _ = self.model(inp)
                dist = Categorical(logits=logits)
                action = dist.sample()
                
                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
                values.append(value.squeeze(-1))
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
            
            seqs = torch.stack(tokens_list, dim=1)
            
            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)
            
            for i in range(bs):
                formula = seqs[i].tolist()
                
                res = self.vm.execute(formula, self.loader.feat_tensor)
                
                if res is None:
                    rewards[i] = -5.0
                    continue
                
                if res.std() < 1e-4:
                    rewards[i] = -2.0
                    continue
                
                score, ret_val = self.bt.evaluate(res, self.loader.raw_data_cache, self.loader.target_ret)
                rewards[i] = score
                
                if score.item() > self.best_score:
                    self.best_score = score.item()
                    self.best_formula = formula
                    tqdm.write(f"[!] New King: Score {score:.2f} | Ret {ret_val:.2%} | Formula {formula}")
            
            # Normalize rewards
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            
            policy_loss = torch.stack([-lp * adv for lp in log_probs], dim=0).sum(dim=0).mean()

            value_pred = torch.stack(values, dim=0).mean(dim=0)
            value_loss = F.mse_loss(value_pred, rewards)

            entropy = torch.stack(entropies, dim=0).mean()

            loss = policy_loss + self.v_coef * value_loss - self.e_coef * entropy
            
            # Gradient step
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.opt.step()
            
            # Apply Low-Rank Decay regularization
            if self.use_lord:
                self.lord_opt.step()
            
            # Logging
            avg_reward = rewards.mean().item()
            m = getattr(self.bt, "last_metrics", {}) or {}
            sharpe = float(m.get("sharpe", 0.0))
            max_dd = float(m.get("max_dd", 0.0))
            turnover = float(m.get("turnover", 0.0))
            postfix_dict = {'AvgRew': f"{avg_reward:.3f}", 'BestScore': f"{self.best_score:.3f}", 'Sharpe': f"{sharpe:.2f}"}
            
            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict['Rank'] = f"{stable_rank:.2f}"
                self.training_history['stable_rank'].append(stable_rank)
            else:
                if self.use_lord:
                    last = self.training_history['stable_rank'][-1] if self.training_history['stable_rank'] else 0.0
                    self.training_history['stable_rank'].append(last)
            
            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_reward)
            self.training_history['best_score'].append(self.best_score)
            self.training_history['sharpe'].append(sharpe)
            self.training_history['max_dd'].append(max_dd)
            self.training_history['turnover'].append(turnover)
            try:
                logger.bind(step=step, sharpe=sharpe, max_dd=max_dd, turnover=turnover, best_score=self.best_score).info(
                    f"train step={step} avg_reward={avg_reward:.4f} sharpe={sharpe:.3f} max_dd={max_dd:.4f} turnover={turnover:.4f}"
                )
            except Exception:
                pass
            try:
                append_metrics_jsonl("metrics.jsonl", {
                    "kind": "train_step",
                    "step": step,
                    "avg_reward": avg_reward,
                    "best_score": self.best_score,
                    "sharpe": sharpe,
                    "max_dd": max_dd,
                    "turnover": turnover,
                })
            except Exception:
                pass
            
            pbar.set_postfix(postfix_dict)

        # Save best formula
        with open("best_meme_strategy.json", "w") as f:
            json.dump(self.best_formula, f)
        
        # Save training history
        import json as js
        with open("training_history.json", "w") as f:
            js.dump(self.training_history, f)
        
        print(f"\n✓ Training completed!")
        print(f"  Best score: {self.best_score:.4f}")
        print(f"  Best formula: {self.best_formula}")


if __name__ == "__main__":
    eng = AlphaEngine(use_lord_regularization=True)
    eng.train()