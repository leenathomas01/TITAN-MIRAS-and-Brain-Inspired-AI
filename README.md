Note: This repo is work in progress and will be updated sporadically during my downtime.

# TITAN, MIRAS and Brain Inspired AI
A thought experiment that builds on recent advancements in sequence modeling, remixing them with iterative thought experiments involving self-reflective loops and agent-based hypothesis evolution.

---
## References:
- Original Blog Post: [Titans + MIRAS: Helping AI have long-term memory](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)
- Titans Paper: [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)
- MIRAS Paper: [arXiv:2504.13173](https://arxiv.org/abs/2504.13173)
---

## TITAN and MIRAS

### Titans: Learning New Context On-the-Fly

**Core Idea:** Titans introduces a novel neural long-term memory module that mimics the human brain's separation of short-term and long-term memory. Unlike fixed-size vectors or matrices in RNNs, it uses a deep multi-layer perceptron (MLP) to summarize and retain high volumes of information.

Key Mechanism: The "surprise metric" (gradient-based error signal) detects unexpected inputs, triggering selective updates to permanent memory. This prioritizes anomalous or important events (e.g., a sudden "banana peel" in a financial report) while forgetting routine data via adaptive weight decay.

Architecture (e.g., MAC Variant): Compresses past data into a summary passed to attention, allowing decisions on what to retain. It incorporates momentum for coherent updates and maintains linear scaling for efficiency.

Benefits: Handles sequences up to 2M+ tokens, outperforming models like Mamba-2 and GPT-4 on long-context recall (e.g., BABILong benchmark) with faster inference.

### MIRAS: A Unified Framework for Sequence Modeling

**Core Idea:** MIRAS views sequence models as associative memory modules optimized via gradient-based methods, transcending mean squared error (MSE) for more robust objectives.

Four Design Pillars:
- Memory Architecture: Structures like vectors, matrices, or deep MLPs (as in Titans).
- Attentional Bias: Internal learning objective that prioritizes what the model optimizes.
- Retention Gate: Regularization (e.g., forgetting mechanisms) to balance new learning with past knowledge retention.
- Memory Algorithm: Optimization process for updates.

Variants:
- YAAD: Uses Huber loss for outlier resistance, gentle on major errors.
- MONETA: Applies generalized norms for stricter penalties, investigating attention vs. forgetting.
- MEMORA: Employs KL-divergence for probabilistic stability in updates.

Benefits: Outperforms baselines on perplexity, accuracy (e.g., HellaSwag, PIQA), and noisy tasks. Enables non-Euclidean objectives for richer designs, validated on DNA modeling and time-series forecasting.

Together, Titans and MIRAS enable real-time memory adaptation, combining RNN speed with Transformer accuracy for long-context AI.

---

## New Feature: Dolphin Twin Framework

Dolphin Twin is our brain-inspired AI thought experiment, designed as a "twin" system with online (real-time, fast) and offline (batch, reflective) components. 
It draws from emergent cognition ideas, such as multi-agent debates and self-reflective loops, to evolve hypotheses or "beliefs" over time.

---

## Summary

This thought experiment remixes Titans' selective memorization and MIRAS' framework with Dolphin Twin's agent-based structure for a hybrid system. This creates a "dolphin twin" that evolves cognition on-the-fly, such as in conversational AI or multi-agent debates where surprises trigger hypothesis remixes.

**Integration Approach:**
Use Titans' surprise metric to gate updates in online Dolphin, while mapping MIRAS variants to offline agents for robust proposal generation. The merge controller ensures safe incorporation, preventing catastrophic interference.

**Innovations:**
- Agents "surprise" each other in debates, updating shared memory to remix hypotheses.
- Continual learning in conversations, prioritizing unique user patterns.
- Counterfactual replays for conflict resolution, maintaining behavioral stability.

---

## Potential Applications

- **Conversational AI**: Personalized memory for ongoing dialogues, adapting to user styles.
- **Multi-Agent Simulations**: Evolving collective intelligence through surprise-driven debates.
- **Continual Learning Systems**: Real-time adaptation in dynamic environments like robotics or data streams.
- **Research Prototypes**: Toy PyTorch implementations for calibrating dynamics before scaling.

---

## TLDR; version
This thought experiment pushes the boundaries of AI memory and cognition by blending Titans/MIRAS' efficient, surprise-based updates with Dolphin Twin's brain-inspired agents. It opens avenues for scalable, emergent systems that learn like humans—selectively and continuously. Next steps include prototyping the merge policy and exploring integrations with tools.
