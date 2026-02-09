# Quick Start Guide

**Get up and running in 5 minutes**

---

## What This System Does

This is a **drilling intelligence system** that watches your drilling parameters and tells you:
- What's happening downhole (even if you can't see it)
- When something's about to go wrong
- What to change to fix it

Think of it as having a very experienced hand watching the data 24/7.

---

## Starting the System

### Step 1: Power Up
```
The system runs on a ruggedized field computer.
Turn it on like any computer - press the power button.
Wait for the dashboard to appear (about 2 minutes).
```

### Step 2: What You'll See

```
┌─────────────────────────────────────────────────────────────┐
│  DRILLING INTELLIGENCE PLATFORM                    [STATUS] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  CURRENT    │  │  SYSTEM     │  │                     │ │
│  │  REGIME     │  │  SAYS...    │  │   3D STATE SPACE    │ │
│  │             │  │             │  │                     │ │
│  │  [NORMAL]   │  │  "All good" │  │   (spinning dots)   │ │
│  │             │  │             │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                                             │
│  WOB: ████████░░ 25 klbs    RPM: ██████░░░░ 120            │
│  TQ:  ██████░░░░ 8,500 ft-lb                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## The 4 Things You Need to Know

### 1. The Regime Box (Most Important)

This tells you the current drilling state:

| Color | Regime | What It Means | What To Do |
|-------|--------|---------------|------------|
| **GREEN** | NORMAL | Everything's fine | Keep drilling |
| **YELLOW** | CAUTION | Something's changing | Watch closely |
| **ORANGE** | DYSFUNCTION | Problem detected | Follow recommendation |
| **RED** | CRITICAL | Serious issue | Act immediately |

### 2. The Recommendation Box

When something's wrong, the system tells you what to do in plain English:

```
"Stick-slip detected. Reduce WOB by 5 klbs or increase RPM by 10."
```

**Just follow what it says.** The system knows drilling.

### 3. The Confidence Number

Shows how sure the system is:

| Confidence | Meaning |
|------------|---------|
| **85%+** | Very confident - act on it |
| **70-85%** | Fairly confident - consider it |
| **Below 70%** | Not sure yet - keep watching |

### 4. The 3D Display

Those spinning dots show your drilling path through "state space." You don't need to understand the math - just know:
- **Tight cluster** = Stable drilling
- **Spread out** = Variable conditions
- **Loops/spirals** = Oscillating problem (stick-slip, whirl)

---

## What To Do When...

### The Box Goes YELLOW
1. Note what changed (depth, formation, parameters)
2. Watch for 2-3 minutes
3. If it stays yellow, prepare to adjust

### The Box Goes ORANGE
1. Read the recommendation
2. Make the suggested adjustment
3. Wait 30 seconds to see effect
4. If no improvement, call supervisor

### The Box Goes RED
1. **Stop increasing WOB immediately**
2. Read the recommendation
3. Call supervisor while making adjustment
4. Log the event

---

## Common Recommendations & Actions

| System Says | What It Means | What To Do |
|-------------|---------------|------------|
| "Add stabilizer" | BHA whipping around | Next trip, add a stab |
| "Reduce WOB 5 klbs" | Too much weight causing stick-slip | Back off weight |
| "Increase RPM 10" | Need more rotation to break stick-slip | Speed up |
| "Reduce motor bend" | Side force causing whirl | Next trip, less bend |
| "Check flow rate" | Hydraulics might be off | Verify pump rate |

---

## End of Shift

1. Note the current regime and any recommendations
2. Tell the oncoming driller what the system showed
3. Log any ORANGE or RED events

---

## Need Help?

- **Supervisor**: For any RED events or if unsure
- **Tech support**: If system isn't updating (frozen display)
- **This guide**: Posted by the station

---

**Remember**: The system is a tool to help you, not replace your judgment. If something doesn't look right, trust your gut and call for help.
