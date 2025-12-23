# TITAN, MIRAS and Dolphin Twin: Thought Experiments on Surprise-Gated Memory

> **Status:** Personal thought experiment • Not production code • Intentionally incomplete

## What This Is

A collection of late-night explorations on how AI memory *could* work, inspired by ideas from:
- Google's **Titans** (surprise-driven learning)
- Google's **MIRAS** (unified memory framework)  
- Brain-inspired dual processing (online reflexes + offline reflection)

I call it "Dolphin Twin" because dolphins use echolocation to probe their environment, and this system uses **surprise** to decide what's worth remembering.

**This is not:**
- ❌ A research paper
- ❌ A working AI system
- ❌ A claim of novelty
- ❌ Production-ready code

**This is:**
- ✅ A personal sketch of "what if memory worked like this?"
- ✅ Math + code + philosophy mixed together
- ✅ Something I didn't want to lose in my chat history junk drawer

---

## Core Idea (One Sentence)

> **Surprise earns memory. Stability allows it to survive. Neglect causes it to fade.**

Everything else follows from that rule.

---

## The Setup

### Traditional Problem
Most AI models either:
1. Forget everything after a conversation ends (no long-term memory)
2. Try to memorize everything (memory bloat, no priorities)

### The Dolphin Twin Approach
**Split cognition into two modes:**

| Mode | What It Does | Speed |
|------|--------------|-------|
| **Online Dolphin** | Reacts in real-time, learns from surprises | Fast |
| **Offline Dolphins** | Reflect on memories, propose stable updates | Slow, careful |
| **Merge Controller** | Decides which updates survive (with veto power) | Gatekeeper |

**Key insight from Titans:** Use gradient-based "surprise" to decide what matters, not frequency or recency.

---

## What's In This Repo

```
/
├── README.md                        ← You are here
├── dolphin_memory_contract.md       ← The rules (philosophy + math)
├── titan_toy_extended.py            ← Minimal working simulation
└── CREDITS.md                       ← Who helped think this through
```

*See [CREDITS.md](CREDITS.md) for collaborators and muses.*

---

## Quick Start (For the Curious)

```bash
# Clone this repo
git clone https://github.com/leenathomas01/TITAN-MIRAS-and-Dolphin-Twin.git
cd TITAN-MIRAS-and-Dolphin-Twin

# Install dependencies (just PyTorch and NumPy)
pip install torch numpy

# Run the toy simulation
python titan_toy_extended.py
```

### What You'll See

The simulation runs through **3 phases**:

1. **Routine (Steps 0-19):** Model sees "Pattern A" repeatedly  
   → Surprise drops, system gets "bored," stops updating memory

2. **Anomaly (Steps 20-24):** Suddenly sees "Pattern B"  
   → Surprise **spikes**, triggers high-priority memory updates

3. **Return (Steps 25+):** Back to "Pattern A"  
   → Tests if the system can reconcile old and new without forgetting

**Look for:**
- `Surprise` metric shooting up at step 20
- `Action=UPDATE` when surprised vs `Action=SKIP` when bored
- `Merge=accepted` or `Merge=REJECT` based on stability checks

---

## Why This Matters (To Me)

I kept having conversations about memory, learning, and forgetting across different AI chats. Ideas would emerge, then vanish. This repo is my way of saying:

**"Here's a snapshot of how I was thinking about this problem on December 23, 2025."**

If someone stumbles on this and it sparks a better idea elsewhere—great. If not, at least I won't lose these notes again.

---

## What's Next (Maybe)

Possible extensions I might explore:
- [ ] Add MEMORA variant (KL-divergence gates for probabilistic stability)
- [ ] Test on real sequence data (wiki articles, conversations)
- [ ] Visualize the "surprise pulse" over time
- [ ] Build an actual multi-agent debate system where agents surprise each other

Or maybe none of that. This might just stay a sketch. That's okay too.

---

## License

MIT License - Take it, remix it, forget about it. Whatever works.

---

## One Last Thing

If you read this and thought "this person has no idea what they're doing"—you're probably right! I'm not a researcher. I'm just someone who reads papers at 10pm and goes "hmm, what if we mashed these ideas together?"

If that bothers you, this repo isn't for you.  
If that excites you, welcome to the thought lab. ☕🐬
