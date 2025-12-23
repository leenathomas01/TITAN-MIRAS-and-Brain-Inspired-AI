#!/usr/bin/env python3
"""
Minimal simulation of surprise-gated memory for the Dolphin Twin thought experiment.
"""
# titan_toy_extended.py
# 
# A minimal simulation of surprise-gated memory with:
# - Gradient-based surprise metric
# - Adaptive thresholding (gets "bored" of routine patterns)
# - Momentum-based memory updates
# - Counterfactual replay with rollback (Epistemic Firewall)
# - Passive decay (forgetting unused patterns)
#
# This demonstrates the core ideas from the Dolphin Memory Contract.

import torch
import torch.nn as nn
import numpy as np
import copy
from collections import deque
import math
import time

# Set seeds for reproducibility (optional, but helpful for comparing runs)
torch.manual_seed(42)
np.random.seed(42)

# -------------------------------
# Configuration (tweak these!)
# -------------------------------
CONFIG = {
    "input_dim": 32,
    "memory_dim": 32,
    "hidden_dim": 64,
    "momentum": 0.9,           # Surprise smoothing
    "base_lr": 0.01,           # Base learning rate
    "sigma_scale": 1.0,        # Sigmoid scaling for adaptive LR
    "initial_threshold_k": 1.5, # Dynamic threshold: mean + k*std
    "running_window": 100,     # Window for mean/std of surprise
    "replay_prompts": 8,       # Number of test prompts for stability check
    "merge_threshold_by_tier": {
        "Tier1": 0.5,          # Low-impact updates
        "Tier2": 1.0           # High-impact updates (require more scrutiny)
    },
    "max_steps": 40,           # Total simulation steps
    "memory_momentum": 0.9,    # Momentum for memory trajectory
    "decay_rate": 1e-4         # Passive decay when not surprised
}

# -------------------------------
# Neural Memory Block (Simple MLP)
# -------------------------------
class NeuralMemoryBlock(nn.Module):
    """
    A simple MLP that represents the 'memory' of the system.
    The memory_weights parameter is what gets updated during surprise events.
    """
    def __init__(self, input_dim, hidden_dim, memory_dim):
        super().__init__()
        self.memory_weights = nn.Parameter(torch.randn(hidden_dim, memory_dim) * 0.01)
        
        # Momentum buffer for memory trajectory (Gemini's patch)
        self.register_buffer('v_mem', torch.zeros_like(self.memory_weights))
        
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(memory_dim, input_dim)
        self.act = nn.GELU()

    def forward(self, x):
        """Forward pass: encode → interact with memory → decode"""
        h = self.act(self.encoder(x))
        mem_out = torch.matmul(h, self.memory_weights)
        out = self.decoder(mem_out)
        return out

    def apply_update(self, grads, lr, is_surprising=False):
        """
        Apply memory update with momentum and decay.
        
        - If surprising: Update memory using momentum
        - If not surprising: Apply passive decay
        """
        with torch.no_grad():
            if is_surprising:
                # Momentum-based update (trajectory, not reaction)
                self.v_mem = CONFIG["memory_momentum"] * self.v_mem + \
                            (1 - CONFIG["memory_momentum"]) * grads
                self.memory_weights -= lr * self.v_mem
            else:
                # Passive decay (pruning unused connections)
                self.memory_weights *= (1 - CONFIG["decay_rate"])

# -------------------------------
# Surprise Gating Mechanism
# -------------------------------
class SurpriseGating:
    """
    Computes gradient-based surprise and manages adaptive thresholds.
    """
    def __init__(self, model):
        self.model = model
        self.window = deque(maxlen=CONFIG["running_window"])
        self.prev_smoothed = 0.0

    def compute_grad_surprise(self, x, target, loss_fn):
        """
        Calculate surprise as gradient norm w.r.t. memory parameters.
        Returns: (surprise, loss_value, gradients)
        """
        pred = self.model(x)
        loss = loss_fn(pred, target)
        
        # Compute gradients only for memory weights
        grads = torch.autograd.grad(
            loss, 
            self.model.memory_weights, 
            retain_graph=False
        )[0]
        
        grad_norm = float(torch.norm(grads, p=2).detach().cpu().numpy())
        return grad_norm, float(loss.detach().cpu().numpy()), grads

    def update_running(self, s):
        """Update rolling statistics and smooth surprise with momentum"""
        self.window.append(s)
        mean = np.mean(self.window) if len(self.window) > 0 else 0.0
        std = np.std(self.window) if len(self.window) > 0 else 1.0
        
        # Momentum smoothing
        self.prev_smoothed = CONFIG["momentum"] * self.prev_smoothed + \
                            (1 - CONFIG["momentum"]) * s
        
        return self.prev_smoothed, mean, std

    def dynamic_threshold(self, mean, std):
        """Adaptive threshold: mean + k*std"""
        return mean + CONFIG["initial_threshold_k"] * (std + 1e-8)

# -------------------------------
# Loss Functions (MIRAS variants)
# -------------------------------
def yaad_loss(pred, target, delta=1.0):
    """YAAD: Huber loss (outlier-resistant)"""
    return nn.SmoothL1Loss(reduction='mean')(pred, target)

def moneta_loss(pred, target, p=1.5):
    """MONETA: Generalized Lp norm"""
    err = torch.abs(pred - target)
    return torch.mean(err ** p)

# -------------------------------
# Merge Controller (Epistemic Firewall)
# -------------------------------
class MergeController:
    """
    Decides whether memory proposals should be accepted.
    Uses counterfactual replay to detect catastrophic forgetting.
    """
    def __init__(self):
        self.audit = []

    def score_proposal(self, surprise, offline_confidence, structure_score):
        """
        Compute acceptance score from:
        - Surprise (normalized)
        - Offline confidence (from validation)
        - Structure score (coherence metric)
        """
        s_norm = 1 - math.exp(-surprise / (1.0 + 1e-6))
        
        # Weighted combination
        w_s, w_c, w_m = 0.6, 0.25, 0.15
        return w_s * s_norm + w_c * offline_confidence + w_m * structure_score

    def counterfactual_replay(self, snapshot_model, current_model, test_prompts):
        """
        Compare baseline (snapshot) vs new (current) model on test prompts.
        Returns: delta (positive = degradation)
        """
        base_scores = []
        new_scores = []
        
        with torch.no_grad():
            for x, t in test_prompts:
                # Baseline
                pred_base = snapshot_model(x)
                loss_base = nn.MSELoss()(pred_base, t)
                base_scores.append(float(loss_base))
                
                # New (after update)
                pred_new = current_model(x)
                loss_new = nn.MSELoss()(pred_new, t)
                new_scores.append(float(loss_new))
        
        base_mean = np.mean(base_scores)
        new_mean = np.mean(new_scores)
        
        return new_mean - base_mean  # Positive = degradation

    def accept_or_reject(self, proposal, snapshot_model, current_model, 
                        tier="Tier1", test_prompts=None):
        """
        Main decision logic:
        1. Score the proposal
        2. Check tier-specific threshold
        3. For Tier2: Run counterfactual replay
        4. Decide: accept, reject, or rollback
        """
        score = self.score_proposal(
            proposal["surprise"],
            proposal["offline_confidence"],
            proposal["structure_score"]
        )
        
        min_req = CONFIG["merge_threshold_by_tier"].get(tier, 0.5)
        decision = {"accepted": False, "score": score, "reason": None}
        
        # Threshold check
        if score < min_req:
            decision["reason"] = "score_below_threshold"
            return decision
        
        # Stability check for high-impact updates
        if tier == "Tier2" and test_prompts:
            delta = self.counterfactual_replay(snapshot_model, current_model, test_prompts)
            
            # Hippocampal Veto: reject if degradation too high
            if delta > 1e-3:
                decision["reason"] = f"replay_degradation:{delta:.6f}"
                return decision
        
        # Accept
        decision["accepted"] = True
        decision["reason"] = "accepted"
        
        # Audit log
        self.audit.append({
            "time": time.time(),
            "proposal": proposal,
            "score": score,
            "tier": tier
        })
        
        return decision

# -------------------------------
# Main Simulation
# -------------------------------
def run_simulation():
    """
    3-phase simulation:
    1. Routine (Pattern A) → system gets bored
    2. Anomaly (Pattern B) → surprise spike
    3. Return to A → stability test
    """
    print("=" * 70)
    print("DOLPHIN TWIN SIMULATION: Surprise-Gated Memory")
    print("=" * 70)
    
    # Initialize
    model = NeuralMemoryBlock(
        CONFIG["input_dim"], 
        CONFIG["hidden_dim"], 
        CONFIG["memory_dim"]
    )
    surprise_agent = SurpriseGating(model)
    merge_controller = MergeController()
    
    # Loss function (can toggle between MSE and YAAD)
    loss_fn = nn.MSELoss()
    
    # Create test patterns
    pattern_A = torch.randn(1, CONFIG["input_dim"])
    pattern_B = torch.randn(1, CONFIG["input_dim"]) * 4.0  # More distinct
    
    # Test prompts for replay (variations of Pattern A)
    test_prompts = [
        (pattern_A + 0.05 * torch.randn_like(pattern_A), pattern_A) 
        for _ in range(CONFIG["replay_prompts"])
    ]
    
    print("\nPhase 1: Routine exposure to Pattern A (Steps 0-19)")
    print("Phase 2: Anomaly - Pattern B appears (Steps 20-24)")
    print("Phase 3: Return to Pattern A (Steps 25+)")
    print("=" * 70)
    print()
    
    # Main loop
    for step in range(CONFIG["max_steps"]):
        # Determine current pattern
        if step < 20:
            # Phase 1: Routine
            inp = pattern_A + 0.05 * torch.randn_like(pattern_A)
            target = inp.clone()
        elif step < 25:
            # Phase 2: Anomaly
            inp = pattern_B.clone()
            target = inp.clone()
        else:
            # Phase 3: Return
            inp = pattern_A.clone()
            target = inp.clone()
        
        # Choose loss (demonstrate toggling between MIRAS variants)
        if step % 10 < 5:
            used_loss_fn = loss_fn
            loss_tag = "MSE"
        else:
            used_loss_fn = lambda pred, tgt: yaad_loss(pred, tgt, delta=1.0)
            loss_tag = "YAAD"
        
        # Compute surprise
        surprise_t, loss_val, grads = surprise_agent.compute_grad_surprise(
            inp, target, used_loss_fn
        )
        
        # Update running stats
        smoothed, mean, std = surprise_agent.update_running(surprise_t)
        threshold = surprise_agent.dynamic_threshold(mean, std)
        
        # Decision: Is this surprising enough to update?
        is_surprising = smoothed > threshold
        
        # Snapshot BEFORE update (Grok's fix)
        snapshot_model = copy.deepcopy(model)
        
        # Apply update
        if is_surprising:
            # Adaptive learning rate based on surprise magnitude
            sigmoid_scale = 1 / (1 + math.exp(-(smoothed - threshold) / 
                                              (CONFIG["sigma_scale"] + 1e-9)))
            applied_lr = CONFIG["base_lr"] * sigmoid_scale
        else:
            applied_lr = 0.0
        
        model.apply_update(grads, applied_lr, is_surprising=is_surprising)
        
        # Create proposal for merge controller
        proposal = {
            "surprise": smoothed,
            "offline_confidence": 0.8 if is_surprising else 0.2,
            "structure_score": 0.6,
            "ops": {"update": "memory_weights_delta"}
        }
        
        # Determine tier (high surprise → Tier2 = high-impact updates requiring stability replay)
        tier_choice = "Tier2" if smoothed > threshold * 2 else "Tier1"
        
        # Merge decision
        decision = merge_controller.accept_or_reject(
            proposal, 
            snapshot_model, 
            model,
            tier=tier_choice,
            test_prompts=test_prompts
        )
        
        # Rollback if rejected (Gemini's patch)
        if not decision['accepted']:
            with torch.no_grad():
                model.memory_weights.copy_(snapshot_model.memory_weights)
            action_str = "ROLLBACK"
        else:
            action_str = "UPDATE" if is_surprising else "SKIP"
        
        # Log
        print(f"Step {step:02d} | Loss({loss_tag})={loss_val:.4f} | "
              f"Surprise={smoothed:.4f} | Threshold={threshold:.4f} | "
              f"Action={action_str:8s} lr={applied_lr:.4f} | "
              f"Tier={tier_choice} | Merge={decision['reason']}")
    
    # Summary
    print("\n" + "=" * 70)
    print("AUDIT LOG SUMMARY")
    print("=" * 70)
    for i, entry in enumerate(merge_controller.audit[:5]):  # Show first 5
        print(f"Entry {i}: score={entry['score']:.3f} tier={entry['tier']}")
    print(f"... (Total: {len(merge_controller.audit)} entries)")
    print("\nSimulation complete! 🐬")

if __name__ == "__main__":
    run_simulation()
