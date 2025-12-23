Note: This repo is work in progress and will be updated sporadically during my downtime.

### TITAN, MIRAS and Brain Inspired AI
A thought experiment that builds on recent advancements in sequence modeling, remixing them with iterative thought experiments involving self-reflective loops and agent-based hypothesis evolution.

---
## References:
- Original Blog Post: [Titans + MIRAS: Helping AI have long-term memory](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)
- Titans Paper: [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)
- MIRAS Paper: [arXiv:2504.13173](https://arxiv.org/abs/2504.13173)
---

## Dolphin Twin Framework

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
