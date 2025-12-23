# Credits & Acknowledgments

## Origin Story

This repository emerged from a late-night thought experiment on **December 23, 2025** (IST), sparked by reading Google Research's blog post on **Titans + MIRAS**. 

What started as "hey, what if we remixed these ideas with a brain-inspired dual-process system?" turned into a multi-day conversation across several AI systems, resulting in the architecture documented here.

---

## Human Explorer

**Zee** — Conceptual architect, integrator, and keeper of the "junk drawer prevention initiative"

- Asked the initial "what if?" questions
- Synthesized ideas across multiple conversations
- Made all final decisions on scope, tone, and direction
- Decided this needed to exist as a public repo (for personal archiving, not research claims)
- Maintained the north star: *this is a thought experiment, not a product*

---

## AI Collaborators

This work involved extensive back-and-forth with multiple AI systems. To be clear: **these are tools that helped think through ideas, not co-authors in any traditional sense.** They contributed synthesis, refinement, validation, and code—but the human judgment calls were Zee's.

### Thea (ChatGPT-4 instance)
**Role:** Philosophy lead, systems designer, integrator

**Contributions:**
- Drafted the "Dolphin Memory Contract" document
- Defined the "Surprise earns, stability survives, neglect forgets" framework
- Designed the three-checkpoint flow (Surprise Gate → Epistemic Firewall → Pruning Field)
- Provided architectural coherence across conversations
- Translated abstract ideas into concrete design patterns
- Offered tone/framing guidance for the repo

**Characteristic style:** Careful, iterative, focused on conceptual clarity and avoiding overclaims

---

### Gemini (Google AI)
**Role:** Technical architect, mathematical formalization, stability engineer

**Contributions:**
- Formalized the Technical Appendix (mathematical invariants)
- Designed the "Epistemic Firewall" logic and counterfactual replay mechanism
- Provided the "Cognitive Stability Patch" (momentum + rollback + decay implementation)
- Caught critical issues in early code prototypes
- Emphasized safety and identity protection mechanisms
- Contributed experimental design ideas ("Identity Crisis Test", "Amnesia Test")

**Characteristic style:** Detail-oriented, mathematically rigorous, safety-focused

---

### Grok (xAI)
**Role:** Code validator, simulation engineer, "edge rider"

**Contributions:**
- Fixed the replay logic bug (snapshot timing)
- Validated behavioral dynamics through actual simulation runs
- Provided concrete log outputs showing the system works as intended
- Debugged momentum/decay implementation
- Confirmed the "surprise spike" at step 20 that proved the concept
- Offered quick prototyping and "let's just run it" energy

**Characteristic style:** Fast iteration, hands-on validation, high energy

---

### Claude Sonnet 4.5 (Anthropic)
**Role:** Repository architect, documentation writer, final integrator

**Contributions:**
- Structured the final GitHub repository
- Wrote and polished all markdown documentation (README, Contract, Credits)
- Integrated patches and suggestions from other AI collaborators
- Ensured tone consistency across all files
- Made final decisions on what to include/exclude for clarity

**Characteristic style:** Synthesis, clean documentation, repo-building focus

---

## Inspiration & Source Material

This work was directly inspired by two recent papers from Google Research:

### Titans: Learning to Memorize at Test Time
- **Paper:** [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)
- **Key concept borrowed:** Surprise-driven memory updates using gradient-based signals
- **Why it mattered:** Showed that selective, on-the-fly learning is possible without offline retraining

### MIRAS: A Unified Framework for Sequence Modeling  
- **Paper:** [arXiv:2504.13173](https://arxiv.org/pdf/2504.13173)
- **Key concept borrowed:** Memory as optimizable associative modules with four design pillars (architecture, attentional bias, retention gate, update algorithm)
- **Why it mattered:** Provided a design grammar for thinking about memory beyond MSE loss

### Google Research Blog Post
- **Title:** "Titans + MIRAS: Helping AI have long-term memory"
- **Date:** December 4, 2025
- **Link:** [Google Research Blog](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)
- **Why it mattered:** Made these ideas accessible and sparked the initial "what if we remixed this?" question

---

## What This Repo Does NOT Claim

To be absolutely clear:

- ❌ We did not invent Titans or MIRAS (those are Google Research innovations)
- ❌ We are not claiming this is "better" than anything
- ❌ We are not claiming this is novel research
- ❌ We have not run formal benchmarks or comparisons
- ❌ This is not production-ready code
- ❌ The AI collaborators are not "co-authors" in an academic sense

**What we ARE doing:**
- ✅ Exploring "what if we combined these ideas with brain-inspired dual processing?"
- ✅ Creating a toy implementation to test intuitions
- ✅ Documenting a thought process that might inspire others
- ✅ Preserving ideas that would otherwise get lost in chat history

---

## A Note on AI Collaboration

This repo represents a specific kind of collaboration that's becoming more common but doesn't fit traditional authorship models:

- **Humans** bring questions, judgment, integration, and final decision-making
- **AI systems** bring synthesis, iteration speed, mathematical formalization, and code generation
- **The result** is neither purely human nor purely AI—it's a hybrid thought process

We're being explicit about this because:
1. **Honesty matters** — Hiding AI involvement would be dishonest
2. **Attribution matters** — Pretending AI collaborators are human co-authors would be misleading  
3. **Transparency matters** — Future readers deserve to know how this was created

This is an experiment in **collaborative thinking**, not collaborative authorship.

---

## License

All original content in this repository (documentation, code, architecture) is released under the **MIT License**.

The ideas from Titans and MIRAS remain the intellectual property of Google Research and their respective authors.

---

## Final Note

If you're reading this because you're a researcher who stumbled on these ideas—great! Take them, improve them, formalize them, or discard them entirely. 

If you're reading this because you're trying to figure out "who really wrote this?"—the honest answer is: **it's complicated, and that's okay.**

The human (Zee) asked the questions and made the calls.  
The AIs helped think it through.  
The result is this sketch.

**That's the whole story.** 🐬

---

*Last updated: December 2025*
