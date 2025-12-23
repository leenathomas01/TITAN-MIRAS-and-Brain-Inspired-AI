# Dolphin Memory Contract

## Purpose

This document defines the rules governing memory formation, retention, and forgetting in the Dolphin Twin thought experiments.

It's not an implementation guide, benchmark report, or claim of performance. Instead, it's a **behavioral contract**—a set of rules that describe how memory is *allowed* to change over time in a cognitive system designed for long-running interaction.

These ideas are inspired by recent work on neural memory and test-time learning, but are presented here as exploratory scaffolding rather than a finalized model.

---

## Core Principle

> **Surprise earns memory.**  
> **Stability allows memory to survive.**  
> **Neglect causes memory to fade.**

Everything in this contract follows from that rule.

---

## The Three Checkpoints

Memory evolution passes through three stages:

### 1. Surprise Gate (Online)

**What happens:** New input is evaluated for "surprise"—how unexpected it is given current memory.

**Decision logic:**
- If surprise is **below threshold** → Input is processed but not memorized
- If surprise is **above threshold** → A "Memory Proposal" is generated

**Why this matters:** Prevents memory bloat from routine, repetitive information. The system learns to get "bored" of patterns it already knows.

---

### 2. Epistemic Firewall (Reflective)

**What happens:** The Memory Proposal is tested against a stable baseline using **counterfactual replay**.

**Decision logic:**
- Run the proposal through a test suite of known patterns
- Measure if the update causes unacceptable "drift" (forgetting old capabilities)
- If drift is too high → **Hippocampal Veto** triggers a rollback

**Why this matters:** Protects long-term identity (retained competencies and previously reinforced patterns). No matter how surprising new information is, it can't erase what the system already knows without permission.

---

### 3. Pruning Field (Continuous)

**What happens:** During low-surprise cycles (routine operation), memory undergoes **passive decay**.

**Decision logic:**
- Weak, unused connections gradually fade
- Only trajectories reinforced by consistent surprise survive long-term

**Why this matters:** Prevents memory saturation. Forgetting is intentional, continuous, and necessary for capacity management.

---

## Detailed Rules

### Rule 1: Surprise Gating (Earning Memory)

Memory updates are **not** triggered by:
- Frequency
- Repetition  
- Volume

They **are** triggered by:
- Surprise (in practice, approximated via gradient magnitude—see Technical Appendix)

**Mechanism:**
- Inputs that are well-predicted → processed but not memorized
- Inputs that violate expectations → generate Memory Proposals
- Thresholds are adaptive (the system can become "bored")

**Result:** Memory capacity is reserved for salient structure, not noise.

---

### Rule 2: Epistemic Firewall (Identity Protection)

All Memory Proposals are **provisional** until validated.

**Process:**
1. Snapshot current memory state
2. Apply proposed update
3. Run counterfactual replay on test prompts
4. Measure degradation of previously learned capabilities
5. If degradation exceeds threshold → **rollback** to snapshot

**The Hippocampal Veto:** No proposal, however surprising, can overwrite core coherence without passing stability checks.

**Result:** Long-term identity is protected from catastrophic forgetting.

---

### Rule 3: Memory Trajectory (Momentum)

Memory updates are not isolated reactions—they accumulate directionally.

**Mechanism:**
- Updates use **momentum** (like in optimization)
- Single events cannot dominate memory state
- Learning is modeled as a trajectory through memory space

**Result:** Reduces oscillation, dampens overreaction, favors persistent patterns over transient spikes.

---

### Rule 4: Retention and Decay (Forgetting)

Forgetting is **intentional, continuous, and necessary**.

**Mechanism:**
- In the absence of surprise → passive decay
- Weak, unused connections gradually fade
- Capacity is reclaimed without explicit deletion

**Result:** Long-term memory reflects reinforced structure, not historical accumulation. The system doesn't become a "data landfill."

---

### Rule 5: Online vs. Offline Cognition

The Dolphin Twin separates cognition into complementary modes:

| Mode | Priority | Speed |
|------|----------|-------|
| **Online** | Responsiveness, provisional adaptation | Fast |
| **Offline** | Consolidation, noise resistance, structural coherence | Slow |

**Critical constraint:** Offline processes may *propose* memory changes but **cannot directly mutate core memory** without passing the Epistemic Firewall.

**Result:** Rapid interaction without sacrificing long-term stability.

---

## Non-Goals

This system does **not** aim to:
- ❌ Memorize all data
- ❌ Optimize benchmark performance  
- ❌ Replicate biological cognition
- ❌ Serve as a production architecture

It exists to explore how memory *might* be governed, not how fast it can grow.

---

## Status

This contract describes an **evolving thought experiment**.

It is intentionally:
- Incomplete
- Provisional  
- Open-ended

Its purpose is to invite better implementations elsewhere.

---

## 📑 Technical Appendix (Mathematical Invariants)

This appendix formalizes the mathematical rules used in the accompanying toy simulations.

### A.1 Surprise Metric

Surprise is defined as the norm of the gradient of the loss with respect to neural memory parameters θ_m:

```
S_t = ||∇_{θ_m} L(x_t)||_2
```

To reduce jitter, surprise is smoothed using momentum coefficient β (typically 0.9):

```
S̃_t = β S̃_{t-1} + (1 - β) S_t
```

---

### A.2 Adaptive Gating

The threshold for memory acquisition evolves dynamically based on the running mean μ_t and standard deviation σ_t of surprise over a window:

```
τ_t = μ_t + k · σ_t
```

where k is the **salience multiplier** (e.g., k=1.5).

A memory proposal is generated only if:

```
S̃_t > τ_t
```

---

### A.3 Memory Trajectory (Momentum)

Memory updates follow a momentum-based trajectory to ensure stability:

```
v_t = γ v_{t-1} + (1 - γ) ΔM_t
M_t = M_{t-1} - η v_t
```

where:
- γ = memory momentum (e.g., 0.9)
- η = learning rate
- ΔM_t = proposed memory delta

---

### A.4 Passive Decay

In the absence of surprise, memory undergoes continuous decay to simulate pruning:

```
M_t = (1 - λ) M_{t-1}
```

where λ is the decay constant (e.g., 1e-4).

---

### A.5 Stability Check (Counterfactual Replay)

For high-impact updates, **epistemic drift** is estimated via replay on test prompts:

```
Δ = E[L_new] - E[L_baseline]
```

The **Hippocampal Veto** is triggered if:

```
Δ > ε
```

where ε is the maximum allowable degradation of core identity.

---

**Note:** A minimal simulation exercising these invariants is provided in `titan_toy_extended.py`.

---

## Closing Note

This contract intentionally sits between philosophy and code.

It is meant to be:
- Read
- Questioned
- Adapted
- Discarded if better ideas emerge

**That is its success condition.**
