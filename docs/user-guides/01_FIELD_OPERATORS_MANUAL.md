# Field Operator's Manual

**Complete Guide to Day-to-Day Operations**

Version 1.0 | For Drillers, Toolpushers, and MWD Hands

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Screen Layout](#screen-layout)
3. [Understanding Regimes](#understanding-regimes)
4. [Reading the Displays](#reading-the-displays)
5. [Acting on Recommendations](#acting-on-recommendations)
6. [Data Entry](#data-entry)
7. [Mission Files](#mission-files)
8. [Shift Handover](#shift-handover)
9. [Troubleshooting](#troubleshooting)

---

## System Overview

### What This System Does

The Drilling Monitoring Platform continuously analyzes your drilling data and:

1. **Flags potential issues** - Multi-parameter analysis may indicate developing problems
2. **Classifies the regime** - Not just "vibration high" but "stick-slip pattern detected"
3. **Suggests adjustments** - Actionable steps you can consider
4. **Adapts to your well** - Calibrates as it processes more data

### How It Works (Simple Version)

```
Your drilling parameters → System analyzes the SHAPE of the data → Detects patterns → Tells you what's happening
```

Traditional systems: "Vibration = 2.5g" (threshold alarm)
This system: "Data is making loops in WOB-Torque space = stick-slip developing"

The goal: identify developing patterns that may precede tool damage.

---

## Screen Layout

### Main Dashboard

```
┌─────────────────────────────────────────────────────────────────────┐
│  DRILLING MONITORING PLATFORM              [Well: SMITH 14-2H]   │
├──────────────────┬──────────────────────────────────────────────────┤
│                  │                                                  │
│   REGIME         │   3D STATE SPACE                                 │
│   ┌──────────┐   │   ┌────────────────────────────────────────┐    │
│   │          │   │   │                                        │    │
│   │  NORMAL  │   │   │      * *  *                            │    │
│   │          │   │   │    *   * *  *                          │    │
│   │ ●●●●○    │   │   │   *  *   *    *                        │    │
│   │ 82% conf │   │   │      trajectory                        │    │
│   └──────────┘   │   │                                        │    │
│                  │   └────────────────────────────────────────┘    │
│   RECOMMENDATION │                                                  │
│   ┌──────────┐   │   PARAMETERS                                    │
│   │          │   │   ┌────────────────────────────────────────┐    │
│   │ All      │   │   │ WOB:  ████████░░░░░░░░  25.2 klbs     │    │
│   │ systems  │   │   │ RPM:  ██████░░░░░░░░░░  118 rpm       │    │
│   │ nominal  │   │   │ TQ:   █████████░░░░░░░  12,450 ft-lb  │    │
│   │          │   │   │ ROP:  ███████░░░░░░░░░  87 ft/hr      │    │
│   └──────────┘   │   │ SPP:  ██████████░░░░░░  3,240 psi     │    │
│                  │   └────────────────────────────────────────┘    │
│                  │                                                  │
│   BETTI NUMBERS  │   DEPTH: 12,847 ft MD | 10,234 ft TVD           │
│   β₀: 1  β₁: 0   │   FORMATION: Wolfcamp A                         │
│                  │                                                  │
└──────────────────┴──────────────────────────────────────────────────┘
```

### Panel Descriptions

| Panel | Purpose | What to Watch |
|-------|---------|---------------|
| **REGIME** | Current drilling state | Color and text |
| **RECOMMENDATION** | What to do | Read when not green |
| **3D STATE SPACE** | Visual of drilling | Shape of point cloud |
| **BETTI NUMBERS** | Technical indicators | β₁ > 0 means oscillation |
| **PARAMETERS** | Live drilling data | Trends, not just values |
| **DEPTH/FORMATION** | Where you are | Context for decisions |

---

## Understanding Regimes

### The Five Drilling Regimes

| Regime | Color | Description | Typical Cause |
|--------|-------|-------------|---------------|
| **NORMAL** | Green | Efficient drilling | Good parameters |
| **STICK-SLIP** | Orange | Torsional oscillation | High WOB, low RPM |
| **WHIRL** | Orange | Lateral vibration | BHA eccentricity |
| **BIT BOUNCE** | Orange | Axial vibration | Hard stringers |
| **CRITICAL** | Red | Multiple issues or severe | Various |

### Regime Transitions

The system shows when you're moving between regimes:

```
NORMAL ───────────────────────────────────────▶ STICK-SLIP
        ↑                                            ↓
        │    (system may flag it HERE)                │
        │                                            │
     You don't                               Damage starts
     notice yet                              happening
```

**Key Point**: Act when it says CAUTION, not when it says CRITICAL.

---

## Reading the Displays

### The 3D State Space

This shows your drilling parameters in 3D. Don't worry about the math - look for patterns:

| What You See | What It Means | Example |
|--------------|---------------|---------|
| **Tight ball of points** | Stable drilling | Normal operations |
| **Stretched cloud** | Variable conditions | Changing formation |
| **Loop or spiral** | Oscillating problem | Stick-slip or whirl |
| **Two separate clusters** | Jumping between states | Transition zone |

### Betti Numbers (β)

These are just counts of shapes in your data:

| Number | What It Counts | Drilling Meaning |
|--------|----------------|------------------|
| **β₀** | Clusters | 1 = stable, 2+ = jumping between modes |
| **β₁** | Loops | 0 = no oscillation, 1+ = vibration/cycling |

**Simple rule**: β₀ = 1 and β₁ = 0 is good. Anything else, pay attention.

### Confidence Level

```
●●●●● 90%+   Very high confidence - act immediately
●●●●○ 75-90% High confidence - strongly recommend
●●●○○ 60-75% Moderate - watch closely
●●○○○ 45-60% Low - more data needed
●○○○○ <45%   Uncertain - system learning
```

---

## Acting on Recommendations

### Recommendation Types

#### Immediate Actions (Do Now)

| Recommendation | What To Do | Expected Result |
|----------------|------------|-----------------|
| "Reduce WOB by X klbs" | Back off weight | Torque stabilizes in 30-60 sec |
| "Increase RPM by X" | Speed up rotation | Stick-slip breaks in 1-2 min |
| "Reduce RPM by X" | Slow down rotation | Whirl dampens in 30-60 sec |
| "Pick up off bottom" | Pull up 5 ft | Immediate stop, reset |

#### Trip-Required Actions (Next Trip)

| Recommendation | What To Do | Why |
|----------------|------------|-----|
| "Add stabilizer" | Add stab next BHA | Centralizes to reduce whirl |
| "Reduce motor bend" | Change bend setting | Reduces side force |
| "Change bit type" | Different bit design | Better formation match |
| "Add shock sub" | Install shock absorber | Dampens axial vibration |

### Decision Flow

```
Recommendation appears
        │
        ▼
Is it an IMMEDIATE action?
        │
   ┌────┴────┐
   │         │
  YES        NO
   │         │
   ▼         ▼
Make the    Log it for
adjustment  next trip
   │         │
   ▼         │
Watch for   Tell supervisor
30-60 sec   and relief
   │
   ▼
Better? ─────────────────────────────────────▶ Log result
   │
   NO
   │
   ▼
Try next suggestion or call supervisor
```

---

## Data Entry

### Required Inputs

The system needs some information from you:

| Input | When | How |
|-------|------|-----|
| **Formation tops** | When you cross | Enter via Formation panel |
| **BHA changes** | After every trip | BHA Builder screen |
| **Bit changes** | When changed | Equipment panel |
| **Problems noted** | When they occur | Notes field |

### Entering a Formation Top

1. Click "Formations" button
2. Click "+ Add Formation"
3. Enter:
   - Depth (MD)
   - Formation name
   - Lithology type (dropdown)
4. Click "Save"

### Entering BHA Components

1. Click "BHA Builder"
2. Drag components from library to assembly
3. Enter specifications for each:
   - OD, ID, Length
   - Stabilizer gauge (if applicable)
   - Motor bend (if applicable)
4. Click "Set Active BHA"

---

## Mission Files

### What's a Mission File?

A mission file saves everything about your well:
- All drilling data
- Formation tops you entered
- BHA configurations
- System recommendations history
- Notes

### Saving Your Work

**Auto-save**: System saves every 5 minutes automatically

**Manual save**:
1. Click "File" menu
2. Click "Save Mission"
3. Choose location (default is fine)
4. Click "Save"

### Loading a Previous Well

1. Click "File" menu
2. Click "Open Mission"
3. Browse to the .mission file
4. Click "Open"

### Exporting for Reports

1. Click "File" menu
2. Click "Export"
3. Choose format:
   - PDF (for printing)
   - CSV (for spreadsheets)
   - JSON (for other systems)

---

## Shift Handover

### End of Your Shift

1. **Screenshot the current state** (or note it)
2. **Check for pending recommendations**
3. **Review any trip-required actions**
4. **Note any formation changes**

### Handover Checklist

```
□ Current regime: _____________
□ Confidence level: ____%
□ Any active recommendations: Y / N
   If yes: ____________________
□ BHA changes needed next trip: Y / N
   If yes: ____________________
□ Unusual events this shift: Y / N
   If yes: ____________________
□ Current depth: _______ ft
□ Current formation: ___________
```

### Verbal Handover Should Include

1. "System is showing [REGIME] with [X]% confidence"
2. "We had [events] at [depths]"
3. "The system recommended [X] and we [did/didn't] act on it because [reason]"
4. "Next trip, we should [recommendation]"

---

## Troubleshooting

### Common Issues

| Problem | Likely Cause | Fix |
|---------|--------------|-----|
| Screen frozen | Display lag | Wait 30 sec, or refresh (F5) |
| No data updating | Connection lost | Check network cable |
| Confidence stays low | Not enough data | Normal for first few stands |
| Wrong formation showing | Needs manual entry | Enter correct formation |
| Recommendations seem wrong | Calibration needed | Call tech support |

### When to Call Tech Support

- System doesn't start
- Data hasn't updated for more than 5 minutes
- Error messages appear
- Recommendations consistently don't match observed behavior

### When to Call Supervisor

- Any RED regime
- Recommendations you don't understand
- System and your gut don't agree
- Before making non-standard adjustments

---

## Safety Reminders

1. **This system advises, you decide** - It's a tool, not your replacement
2. **Trust your experience** - If something doesn't feel right, investigate
3. **When in doubt, back off** - Reduce parameters until you understand
4. **Document everything** - Notes help future crews
5. **Never ignore RED** - Always respond to critical alerts

---

## Quick Reference Card

### Regime Colors
- **GREEN** = Good
- **YELLOW** = Watch
- **ORANGE** = Act
- **RED** = Emergency

### Key Betti Numbers
- **β₀ = 1, β₁ = 0** = Normal
- **β₁ ≥ 1** = Oscillation detected

### Response Times
- Stick-slip fix: 30-60 seconds
- Whirl fix: 30-60 seconds
- Formation change adjustment: 2-5 minutes

### Emergency Contacts
- Tech Support: _______________
- Supervisor: _______________
- Company Man: _______________
