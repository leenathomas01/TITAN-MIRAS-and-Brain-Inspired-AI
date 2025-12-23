Note: This repo is work in progress and will be updated sporadically during my downtime.

# TITAN, MIRAS and Brain Inspired AI
A thought experiment and research prototype exploring emergent AI cognition. It is not merely a wrapper for existing models, but **a distinct cognitive architecture** designed to orchestrate short-term responsiveness and long-term reflection.
By integrating Google Research's Titans (neural memory) and MIRAS (sequence optimization) as subsystems, this thought experiment attempts to replicate the "Fast vs. Slow" thinking found in biological brains, enabling AI that learns, forgets, and evolves beliefs on the fly.

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

Traditional AI models are static snapshots. Dolphin Twin is designed to be a living system composed of two distinct modes of operation, inspired by the dual-process theory of cognition (System 1 vs. System 2) and the echolocation (probing) of dolphins.

#### 1. The Online Dolphin ("Fast System")
Role: Handles real-time interaction, intuition, and immediate context.
Mechanism: Uses the Titans architecture to manage a massive, efficient context window.
Behavior: It relies on a "Surprise Metric" to decide in the moment what is worth remembering. If an input is predictable, it is processed and discarded. If it is "surprising" (high gradient error), it is gated into long-term memory.

#### 2. The Offline Dolphin ("Slow System")
Role: Handles reflection, consolidation, and hypothesis evolution.
Mechanism: A multi-agent system powered by MIRAS optimization variants.
Behavior: While the Online system rests (or in parallel), Offline agents "dream" or debate over recent memories. They replay past events, prune noise, and restructure the memory architecture to be more efficient for the future.

Dolphin Twin orchestrates these two cutting-edge technologies to achieve its cognitive loop.

#### Titans: The Memory Module
Used by the Online System for real-time retention.
Function: Replaces fixed-size buffers with a deep Neural Memory MLP.
Surprise Metric: A gradient-based signal detects anomalies.
Example: A sudden "banana peel" in a financial report triggers a memory update, while routine data is allowed to decay.
Benefit: Allows the system to handle 2M+ token contexts efficiently, prioritizing "important" data over "recent" data.

#### MIRAS: The Optimization Framework
Used by Offline Agents to structure beliefs.
Function: Provides the mathematical rules for how memory is updated, moving beyond simple Mean Squared Error (MSE).
Agent Personas (Mapped to MIRAS Variants):
Dolphin-Cleaner (YAAD Variant): Uses Huber loss to identify and prune outliers (noise) from memory.
Dolphin-Structuralizer (MONETA Variant): Uses generalized norms to merge similar concepts and strictly enforce forgetting of irrelevant data.
Dolphin-Historian (MEMORA Variant): Uses KL-divergence to ensure new information doesn't catastrophically overwrite stable, long-held beliefs

---


## Potential Applications

- **Conversational AI**: Personalized memory for ongoing dialogues, adapting to user styles.
- **Multi-Agent Simulations**: Evolving collective intelligence through surprise-driven debates.
- **Continual Learning Systems**: Real-time adaptation in dynamic environments like robotics or data streams.
- **Research Prototypes**: Toy PyTorch implementations for calibrating dynamics before scaling.

---

## TLDR; version
This thought experiment pushes the boundaries of AI memory and cognition by blending Titans/MIRAS' efficient, surprise-based updates with Dolphin Twin's brain-inspired agents. It opens avenues for scalable, emergent systems that learn like humans—selectively and continuously. Next steps include prototyping the merge policy and exploring integrations with tools.
