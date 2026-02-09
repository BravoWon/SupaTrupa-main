# Supervisor's Reference

**For Toolpushers, Company Men, and Drilling Supervisors**

Technical depth for experienced hands who need to understand the "why"

---

## Executive Summary

This system applies **topological data analysis (TDA)** to drilling data to:
1. Detect drilling dysfunction 30-120 seconds before traditional threshold alarms
2. Identify specific dysfunction type (stick-slip vs whirl vs bit bounce)
3. Provide physics-based remediation recommendations
4. Adapt to each well's specific characteristics

**Bottom line**: Earlier detection + specific diagnosis = less NPT, less tool damage, better hole.

---

## How It Differs From Traditional Monitoring

### Traditional (MSE, RTO Systems)

```
Raw Data → Fixed Thresholds → Alarm

Problem: Reactive. Damage starts before alarm triggers.
```

| System Type | Detection Method | Limitation |
|-------------|------------------|------------|
| MSE Monitoring | Energy per volume > threshold | Single scalar, can't distinguish dysfunction types |
| Vibration Alarms | Acceleration > threshold | Late detection, MWD bandwidth limits resolution |
| RTO Systems | Multi-parameter thresholds | Still reactive, requires manual tuning |

### This System (Topological Approach)

```
Raw Data → State Space Embedding → Persistent Homology → Regime Classification → Recommendation
```

**Key difference**: Analyzes the **shape** of data in multi-dimensional space, not individual parameter values.

| Advantage | Explanation |
|-----------|-------------|
| Earlier detection | Topology changes before thresholds breach |
| Specific diagnosis | Different regimes have distinct topological signatures |
| Adaptive | Learns each well's "normal" topology |
| Physics-based | Recommendations derive from understanding of BHA mechanics |

---

## Technical Foundation

### State Space Representation

Drilling parameters form a high-dimensional state space:
- WOB, RPM, Torque, SPP, Flow Rate, ROP, ...

The system embeds recent data (30-sample sliding window) as points in this space.

### Persistent Homology

TDA technique that identifies topological features (clusters, loops, voids) that persist across multiple scales:

| Topological Feature | Mathematical Name | Drilling Interpretation |
|---------------------|-------------------|------------------------|
| Disconnected groups | H₀ (0-dimensional homology) | Multi-modal operation, regime jumping |
| Loops/cycles | H₁ (1-dimensional homology) | Oscillatory behavior (stick-slip, whirl) |
| Voids | H₂ (2-dimensional homology) | Avoided states (instability regions) |

### Betti Numbers

Count of features at each dimension:
- **β₀ = 1**: Single connected cluster (stable)
- **β₀ > 1**: Multiple clusters (jumping between modes)
- **β₁ = 0**: No loops (non-oscillatory)
- **β₁ > 0**: Loops present (cyclic behavior detected)

### Regime Classification

Topological signatures map to drilling regimes:

| Regime | β₀ | β₁ | Additional Signatures |
|--------|----|----|----------------------|
| Normal | 1 | 0 | Compact point cloud |
| Stick-slip | 1 | 1-3 | Loop in WOB-Torque plane |
| Whirl | 1 | 1-2 | Loop with higher frequency signature |
| Bit bounce | 1-2 | 0-1 | Axial separation in point cloud |
| Transition | 2 | 0-1 | Cloud elongation, potential bifurcation |

---

## System Confidence Calibration

### Confidence Calculation

Confidence reflects:
1. **Pattern clarity**: How distinct is the topological signature?
2. **Historical match**: How similar to previously seen patterns?
3. **Stability**: How long has this pattern persisted?
4. **Data quality**: Are all inputs present and reasonable?

### Confidence Thresholds

| Confidence | Interpretation | Recommended Action |
|------------|----------------|-------------------|
| 90%+ | Very high certainty | Act immediately |
| 80-90% | High certainty | Strongly recommend action |
| 70-80% | Moderate certainty | Crew should consider |
| 60-70% | Low certainty | Inform only, watch |
| <60% | Uncertain | System still learning |

### Factors That Lower Confidence

- First few stands in new formation
- Unusual parameter combinations
- Mixed/transitional states
- Sensor noise or dropouts
- Operations the system hasn't seen before

---

## BHA Optimization Logic ("Les Logic")

### The Riemannian Metric

The system uses a curved metric on drilling state space:

```
ds² = dt² + (dd/ROP)²
```

Where:
- dt = time increment
- dd = depth increment
- ROP = rate of penetration

**Interpretation**: Gaining depth when ROP is low "costs" more than when ROP is high.

### Geodesics and Optimal Paths

The system calculates geodesics (shortest paths in curved space) to determine:
- Optimal parameter trajectories
- Cost of proposed changes
- Expected improvement

### Recommendation Derivation

Each recommendation derives from:
1. Current regime identification
2. Physical model of BHA behavior
3. Geodesic from current state to target state
4. Historical success rate of similar recommendations

---

## Regime-Specific Guidance

### Stick-Slip

**Physics**: Torsional resonance in drill string. Bit grabs, torque builds, bit releases and overspeeds, cycle repeats (typically 0.1-0.5 Hz).

**Why surface looks OK**: Surface torque shows averaged behavior. Downhole peak torque can be 2-3x surface average.

**Standard remediation**:

| Action | Effect | Success Rate |
|--------|--------|--------------|
| Reduce WOB 5-10 klbs | Reduces bit-rock engagement | 70% |
| Increase RPM 10-20 | Raises resonant frequency | 65% |
| Reduce WOB + Increase RPM | Combined effect | 85% |

**BHA changes (next trip)**:
- Reduce motor bend angle (0.5° increment)
- Add near-bit stabilizer
- Consider different bit design

### Whirl

**Physics**: Lateral instability causing BHA to orbit wellbore. Forward whirl (BHA rolls along wall), backward whirl (BHA bounces along wall).

**Why surface looks OK**: Lateral motion averages out in torque/WOB readings. Only high-frequency MWD catches it.

**Damage pattern**: Stabilizer gauge wear, motor housing erosion, hole enlargement.

**Standard remediation**:

| Action | Effect | Success Rate |
|--------|--------|--------------|
| Reduce RPM 10-20 | Reduces centrifugal force | 60% |
| Reduce WOB 5 klbs | Reduces side force | 50% |

**BHA changes (next trip)** (usually required):
- Add stabilizer(s) - increases stiffness
- Check stabilizer gauge - replace if worn
- Consider packed BHA design

### Bit Bounce

**Physics**: Axial resonance, often tri-lobed pattern from PDC bit interaction with hard stringers.

**Standard remediation**:

| Action | Effect | Success Rate |
|--------|--------|--------------|
| Smooth WOB application | Reduces impact loading | 55% |
| Adjust flow rate | Changes hydraulic dampening | 45% |
| Pick up, re-engage | Resets system | 70% |

**BHA changes**:
- Add shock sub
- Consider roller cone in hard formations

---

## When System and Experience Conflict

### Trust the System When:

- Fresh eyes on a pattern you've normalized
- Early-stage detection (before you'd notice)
- Pattern matches known failure modes
- High confidence (>80%)

### Trust Experience When:

- System confidence is low
- You have specific knowledge system doesn't (e.g., planned formation change)
- Operational factors not in data (e.g., mud issues, equipment problems)
- Something "feels wrong"

### Best Practice

Document disagreements and outcomes. If system was right, note it. If experience was right, that's valuable training data.

---

## Integration with Existing Workflows

### WITSML/WITS Compatibility

System accepts standard oilfield data formats:
- WITS (legacy)
- WITSML 1.4.1.1
- WITSML 2.0
- Custom adapters available

### Recommended Data Feeds

| Data Type | Minimum Rate | Ideal Rate | Source |
|-----------|--------------|------------|--------|
| WOB | 1 Hz | 10 Hz | EDR/WITS |
| RPM | 1 Hz | 10 Hz | VFD/WITS |
| Torque | 1 Hz | 10 Hz | Top drive |
| SPP | 1 Hz | 5 Hz | Manifold |
| Flow | 1 Hz | 5 Hz | Flow meter |
| ROP | Derived | Derived | Calculated |
| MWD Surveys | Per stand | Per stand | MWD |
| MWD Vibration | If available | If available | MWD |

### Morning Report Integration

System can export:
- Regime time breakdown (% in each regime)
- Dysfunction events (time, depth, type, action taken)
- Parameter statistics by formation
- Recommendations log

---

## ROI Considerations

### Quantifiable Benefits

| Benefit | Typical Value | How Measured |
|---------|---------------|--------------|
| Avoided whirl damage | $100K-500K per event | Tool inspection reports |
| Reduced trips | $50K-200K per avoided trip | Trip count vs offset |
| Improved ROP | 5-15% improvement | ft/day vs offset |
| Reduced NPT | 5-10% reduction | NPT hours |

### Soft Benefits

- Crew confidence in drilling decisions
- Better shift handovers
- Historical data for future wells
- Training aid for green hands

---

## Crew Training Recommendations

### For Drillers

1. Start with Quick Start Guide (30 min)
2. Watch for regime changes during normal drilling
3. Practice: When you see ORANGE, read recommendation but verify your instinct first
4. Build trust over 2-3 wells

### For MWD Hands

1. Concepts Guide for technical understanding
2. Correlate downhole vibration with system detections
3. Help calibrate formation entries

### For Supervisors

1. This document
2. Review weekly: System recommendations vs outcomes
3. Debrief crew on significant events

---

## Emergency Procedures

### System Indicates CRITICAL (Red)

1. Driller: Reduce WOB immediately, read recommendation
2. Notify supervisor
3. Follow recommendation while monitoring
4. Log event details

### System Fails / No Data

1. Continue with conventional drilling practices
2. Note time of failure
3. Contact tech support
4. Increase manual monitoring

### System and Sensors Disagree

1. Trust sensors for immediate safety (kick indicators, etc.)
2. If system shows CRITICAL but sensors nominal, investigate carefully
3. If sensors alarm but system shows NORMAL, trust sensors and notify tech support

---

## Contact and Support

### Technical Support
- Phone: _______________
- Email: _______________
- Hours: _______________

### Training Requests
- Contact: _______________

### Feature Requests / Feedback
- Send to: _______________

---

## Appendix: Algorithm Details

### Persistent Homology Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Window size | 30 samples | 20-60 | Larger = more stable, slower response |
| Max dimension | 1 | 0-2 | Higher = more features, more compute |
| Persistence threshold | 0.1 | 0.05-0.2 | Filters noise |

### Regime Detection Thresholds

Internally tuned, but conceptually:

| Metric | Stable | Transition | Dysfunction |
|--------|--------|------------|-------------|
| β₀ | 1 | 1-2 | >1 sustained |
| β₁ | 0 | 0-1 | >0 sustained |
| Cloud compactness | <0.1 | 0.1-0.3 | >0.3 |
| Variance trend | Decreasing | Flat | Increasing |

### Hardware Requirements

- CPU: Modern multi-core (4+ cores)
- RAM: 8GB minimum, 16GB recommended
- Storage: 50GB for mission files
- Network: Ethernet to WITS/WITSML source
