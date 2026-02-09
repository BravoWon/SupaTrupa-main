# Glossary of Terms

**From Field to Algorithm - All the Terms You'll Encounter**

---

## Drilling Terms

| Term | Definition | Also Known As |
|------|------------|---------------|
| **BHA** | Bottom Hole Assembly - the drill string components near the bit | Bottomhole assembly |
| **Bit** | The cutting tool at the bottom of the string | Drill bit |
| **Bit bounce** | Axial vibration where the bit bounces up and down | Axial vibration |
| **Collar** | Heavy-wall pipe providing weight for drilling | Drill collar |
| **Directional drilling** | Drilling at angles other than vertical | Horizontal drilling (when horizontal) |
| **Formation** | The rock layer you're drilling through | - |
| **Formation top** | The depth where a new formation begins | Top |
| **Kick** | Unplanned entry of formation fluid into wellbore | Well kick |
| **LAS file** | Log ASCII Standard - standard file format for well log data | .las |
| **MD** | Measured Depth - distance along the wellbore path | - |
| **Mnemonic** | Short code identifying a log curve in a LAS file (e.g., GR, RHOB, DEPT) | Curve name |
| **Motor** | Mud motor that converts fluid flow to bit rotation | PDM, positive displacement motor |
| **Motor bend** | The angle of the bent housing in the motor | Bend angle, bent sub angle |
| **MSE** | Mechanical Specific Energy - energy per volume of rock | - |
| **MWD** | Measurement While Drilling - downhole sensors | - |
| **NPT** | Non-Productive Time - time not making hole | - |
| **PDC** | Polycrystalline Diamond Compact - type of drill bit | - |
| **ROP** | Rate of Penetration - drilling speed (ft/hr) | Penetration rate |
| **RPM** | Rotations Per Minute - rotary speed | Rotary speed |
| **SPP** | Standpipe Pressure - pump pressure | Pump pressure |
| **Stabilizer** | Tool that centers the BHA in the wellbore | Stab |
| **Stand** | Length of drill pipe (about 90 ft typically) | Joint (single), stand (triple) |
| **Stick-slip** | Torsional oscillation of the drill string | Torsional vibration |
| **Survey** | Directional measurement (inclination, azimuth) | MWD survey |
| **Top drive** | Surface motor that rotates the drill string | - |
| **Torque** | Rotational force on the drill string (ft-lbs) | TQ |
| **Trip** | Pulling out and running back in the hole | Round trip |
| **TVD** | True Vertical Depth - vertical distance from surface | - |
| **Whirl** | Lateral vibration where BHA orbits in the hole | Lateral vibration |
| **WITSML** | Standard protocol for real-time drilling data | - |
| **WOB** | Weight on Bit - downward force on bit (klbs) | Weight |

---

## System Terms

| Term | Definition | Plain English |
|------|------------|---------------|
| **Activity:State** | A distinct operating regime with its own rules | Like weather states - clear, stormy, etc. |
| **Advisory Engine** | System that computes regime-specific parameter prescriptions with correlation-aware ordering | The "what should I do?" engine |
| **Attractor** | A state toward which a dynamical system tends to evolve | Where the drilling behavior "settles" |
| **Attribution** | Per-feature breakdown of how much each TDA dimension contributes to the current regime classification | "Which features are driving this regime?" |
| **Betti number** | Count of topological features (β₀=clusters, β₁=loops) | Counts shapes in the data |
| **Betti curve** | Plot of Betti numbers across filtration values showing how topology changes with scale | Shape evolution across scales |
| **Calibration** | Period when system learns what's normal for this well | Learning time |
| **Change detection** | Monitoring windowed TDA signatures for abrupt shifts that signal a regime transition | Pattern shift alert |
| **Channel preset** | Named subset of drilling parameters for the Parameter Resonance Network (e.g., Co Man, DD, MWD) | Parameter filter |
| **ConditionState** | A single data point with all parameters | One snapshot of drilling |
| **Confidence** | How sure the system is about its assessment (%) | Certainty level |
| **Confidence band** | Upper and lower bounds around a forecast based on historical variance | Forecast uncertainty range |
| **Correlation dimension** | Fractal measure of how a system fills its state space; low = simple, high = complex | Complexity measure |
| **Curvature field** | Map of parameter sensitivity across drilling state space; high curvature = small changes have large effects | Sensitivity map |
| **Dashboard summary** | Aggregated view combining regime, topology, dynamics, and advisory data into a single status | At-a-glance status |
| **Delay embedding** | Reconstructing hidden dynamics from a single time series using time-delayed copies (Takens' theorem) | Hidden dynamics |
| **Field atlas** | Collection of registered wells with their topological fingerprints for cross-well comparison | Field-wide well library |
| **Fingerprint** | 10-dimensional TDA signature that uniquely characterizes a regime's topological structure | Regime signature |
| **Forecast** | Extrapolation of TDA signatures forward in time using weighted regression | Predicted trajectory |
| **Geodesic** | Optimal path through state space | Best route from A to B |
| **Manifold** | The mathematical surface data lives on | The "data landscape" |
| **Metric tensor** | How distance is measured in state space | The "ruler" for the data landscape |
| **Mission file** | Saved file containing all well data and settings | Save file (.mission) |
| **Parameter Resonance Network** | Force-directed graph showing correlation strengths between drilling channels | Parameter interaction map |
| **Pattern search** | Nearest-neighbor search across registered wells using topological distance | "Find similar wells" |
| **Persistent homology** | TDA technique for finding stable shapes in data | Finding real patterns vs noise |
| **Point cloud** | Collection of data points in multi-dimensional space | All your data points together |
| **RQA** | Recurrence Quantification Analysis - measures how often a system revisits similar states (recurrence rate, determinism, laminarity, trapping time) | Behavioral consistency metrics |
| **Regime** | Current operating state (Normal, Stick-slip, etc.) | Drilling mode |
| **Regime strip** | Colored bar showing regime classification at each depth window along the wellbore | Regime evolution display |
| **Riemannian** | Type of geometry where space is curved | Curved space math |
| **Risk assessment** | Evaluation of regime risk, change magnitude risk, and correlation risk for a proposed parameter change | "How dangerous is this move?" |
| **SANS** | Symbolic Abstract Neural Search - expert system | The decision-making part |
| **Shadow tensor** | Delay-coordinate embedding that reveals hidden dynamical structure invisible in raw parameter space | Hidden state reconstruction |
| **Sliding window analyzer** | Interactive tool for stepping through LAS depth windows with per-window regime classification and TDA | Depth-by-depth analysis |
| **State space** | Multi-dimensional space of all parameters | All parameters at once |
| **TDA** | Topological Data Analysis | Shape-based data analysis |
| **Threshold** | Fixed limit that triggers alarm (traditional method) | Alarm limit |
| **Topology** | Study of shapes and how they connect | Shape math |
| **Transition probability** | Softmax-based likelihood of moving to each possible regime given current trajectory | "Where are we heading?" |
| **Well comparison** | Pairwise analysis of topological distance, feature deltas, and regime similarity between two wells | Side-by-side well analysis |
| **Window** | Recent data samples used for analysis | Last 30 seconds of data |
| **Windowed signature** | TDA features computed over a sliding window, capturing how topology evolves over time | Time-sliced topology |

---

## Attractor Types

| System Term | Operator Term | What It Means | Typical Behavior |
|-------------|---------------|---------------|------------------|
| **fixed_point** | Stable | System converges to a single steady state | Smooth, consistent drilling |
| **limit_cycle** | Cyclic | System oscillates in a repeating pattern | Regular torsional or axial oscillation |
| **strange_attractor** | Complex | Deterministic chaos - structured but never repeating | Unpredictable but patterned dysfunction |
| **quasi_periodic** | Multi-Cycle | Multiple overlapping cycles that don't fully synchronize | Combined vibration modes |
| **stochastic** | Noisy | Dominated by random noise, no clear structure | Formation-driven randomness |
| **transient** | Transitioning | System is moving between attractor types | Regime change in progress |

---

## Operator-Language Translation

The UI uses drilling operator terms instead of mathematical notation:

| Math Term | Operator Term | Where You'll See It |
|-----------|---------------|---------------------|
| Beta-zero (β₀) | Drilling Zones | KPI Cards, Betti Timeline, Forecast |
| Beta-one (β₁) | Coupling Loops | KPI Cards, Betti Timeline, Forecast |
| Persistence Entropy | Stability | Fingerprint radar, Attribution bars |
| Lyapunov Exponent | Predictability Index | Dynamics tab, Dashboard |
| Ricci Scalar | Parameter Sensitivity | Sensitivity tab heatmap |
| Geodesic | Optimal Path | Sensitivity overlay, Advisory |
| Persistence Barcode | Feature Lifetime Chart | CTS panel |
| Topological Change | Pattern Shift | Change Detector footer bar |
| Delay Embedding | Hidden Dynamics | Dynamics tab 3D scatter |
| Topological Heatmap | Feature Evolution Map | CTS center panel |
| Trust Calibration | Automation Readiness | CTS right panel gauge |
| Recurrence Rate | Behavioral Consistency | Dashboard, Dynamics tab |

---

## Regime Terms

| Term | Color | What's Happening | What To Do |
|------|-------|------------------|------------|
| **Normal** | Green | Efficient drilling | Continue |
| **Optimal** | Green | Best possible drilling performance | Maintain parameters |
| **Darcy Flow** | Green | Normal fluid flow through formation | Continue |
| **Non-Darcy Flow** | Yellow | Abnormal fluid flow regime | Monitor flow rates |
| **Stick-slip** | Orange | Torsional oscillation | Reduce WOB or increase RPM |
| **Whirl** | Yellow | Lateral vibration | Reduce RPM, add stab next trip |
| **Bit bounce** | Yellow | Axial vibration | Smooth WOB, adjust flow |
| **Packoff** | Orange | Hole packing around BHA | Increase flow, work pipe |
| **Turbulent** | Orange | Turbulent annular flow | Reduce flow rate |
| **Multiphase** | Orange | Mixed fluid phases in wellbore | Monitor returns |
| **Formation change** | Yellow | Transitioning between rock types | Adjust parameters for new formation |
| **Washout** | Red | Hole enlargement from erosion | Reduce flow, pull off bottom |
| **Lost circulation** | Red | Drilling fluid lost to formation | Pump LCM, reduce mud weight |
| **Kick** | Red | Formation fluid entering wellbore | Shut in well, follow kill procedure |
| **Transition** | Yellow | Between regimes | Watch closely |
| **Unknown** | Yellow | Regime not yet classified | Gather more data |

---

## Mathematical Terms (Simplified)

| Term | What It Is | Why We Use It |
|------|------------|---------------|
| **β₀ (beta-zero)** | Count of separate clusters | 1 = stable, 2+ = jumping between modes |
| **β₁ (beta-one)** | Count of loops in data | 0 = smooth, 1+ = oscillating |
| **Filtration** | Connecting nearby points at increasing distances | Finding structure at different scales |
| **H₀, H₁** | Homology groups (0-dim, 1-dim) | Formal version of β₀, β₁ |
| **Lyapunov exponent** | Rate at which nearby trajectories diverge | Positive = chaotic, negative = stable |
| **Persistence** | How long a feature lasts across scales | Long-lived = real, short-lived = noise |
| **Ricci scalar** | Single number summarizing curvature at a point | High = sensitive region, low = stable region |
| **Simplex** | Basic building block (point, line, triangle) | How we build shapes from data |
| **Takens' theorem** | Proves you can reconstruct a system's dynamics from a single time series with delays | Why delay embedding works |
| **Vietoris-Rips** | Method of connecting points into shapes | The specific way we build structure |

---

## Units

| Parameter | Typical Units | Range |
|-----------|---------------|-------|
| WOB | klbs (thousand pounds) | 5-50 klbs typical |
| RPM | rpm (rotations per minute) | 50-200 rpm typical |
| Torque | ft-lbs (foot-pounds) | 5,000-30,000 ft-lbs |
| ROP | ft/hr (feet per hour) | 10-200 ft/hr |
| SPP | psi (pounds per square inch) | 2,000-5,000 psi |
| Flow | gpm (gallons per minute) | 300-800 gpm |
| Depth | ft (feet) | 0-25,000+ ft |
| Motor bend | degrees (°) | 0-3° typically |
| Stabilizer gauge | inches | 8.5" to 12.25" typical |
| Vibration | g (multiples of gravity) | 0-5g typical |

---

## Abbreviations

| Abbreviation | Full Form |
|--------------|-----------|
| API | American Petroleum Institute |
| BHA | Bottom Hole Assembly |
| CTS | Cognitive Topological System (the main operator interface) |
| DD | Directional Driller |
| EDR | Electronic Drilling Recorder |
| ECD | Equivalent Circulating Density |
| GPS | Gallons Per Stroke |
| HSE | Health, Safety, Environment |
| LAS | Log ASCII Standard |
| LWD | Logging While Drilling |
| MD | Measured Depth |
| MoE | Mixture of Experts |
| MSE | Mechanical Specific Energy |
| MWD | Measurement While Drilling |
| NPT | Non-Productive Time |
| PDC | Polycrystalline Diamond Compact |
| PDM | Positive Displacement Motor |
| PRN | Parameter Resonance Network |
| ROP | Rate of Penetration |
| RPM | Rotations Per Minute |
| RQA | Recurrence Quantification Analysis |
| RSS | Rotary Steerable System |
| SPP | Standpipe Pressure |
| TDA | Topological Data Analysis |
| TVD | True Vertical Depth |
| UCS | Unconfined Compressive Strength |
| WOB | Weight on Bit |

---

## Pronunciations

For terms that might be unfamiliar:

| Term | Pronunciation |
|------|---------------|
| Betti | "BET-ee" |
| Geodesic | "jee-oh-DESS-ik" |
| Homology | "hoe-MOLL-oh-jee" |
| Lyapunov | "lee-ah-POO-nov" |
| Manifold | "MAN-ih-fold" |
| Ricci | "REE-chee" |
| Riemannian | "ree-MAH-nee-an" |
| Takens | "TAH-kens" |
| Topology | "toh-POLL-oh-jee" |
| Vietoris-Rips | "vee-eh-TOR-iss rips" |
| WITSML | "WITS-em-ell" |

---

## See Also

- **00_QUICK_START.md** - Get up and running
- **01_FIELD_OPERATORS_MANUAL.md** - Day-to-day operations
- **02_CONCEPTS_GUIDE.md** - How it works (plain English)
- **03_SUPERVISORS_REFERENCE.md** - Technical depth
- **04_TROUBLESHOOTING_FAQ.md** - Problem solving
- **LES_LOGIC_BHA_OPTIMIZATION.md** - Full technical documentation
- **TEST_PLAN.md** - Comprehensive test procedures
