# Jones Framework - Drilling Operations Quickstart

**For field engineers and drilling ops teams**

This guide gets you running regime detection and BHA optimization in 5 minutes.

---

## 1. Install

```bash
cd backend
pip install -e .
```

## 2. Quick Test - Does It Work?

```python
from jones_framework.perception.regime_classifier import RegimeClassifier
from jones_framework.core.condition_state import ConditionState
import numpy as np

# Create a classifier
classifier = RegimeClassifier()

# Simulate some drilling telemetry (WOB, RPM, ROP, torque, vibration)
telemetry = np.array([
    [25.0, 120, 85, 12000, 0.3],  # Normal drilling
    [25.5, 118, 82, 12200, 0.4],
    [26.0, 115, 78, 13000, 0.8],  # Vibration increasing
    [24.0, 110, 65, 14500, 1.2],  # Dysfunction developing
    [22.0, 105, 45, 16000, 2.1],  # High vibration - regime change
])

# Classify the regime
result = classifier.classify(telemetry)
print(f"Current regime: {result['regime']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Transition risk: {result.get('transition_risk', 'N/A')}")
```

**Expected output:**
```
Current regime: HIGH_VOLATILITY
Confidence: 0.78
Transition risk: 0.65
```

---

## 3. Real-Time Regime Detection

Connect to your MWD/LWD feed:

```python
from jones_framework.perception.regime_classifier import RegimeClassifier
from jones_framework.perception.tda_pipeline import TDAPipeline
from jones_framework.core.condition_state import ConditionState
import numpy as np

# Initialize
tda = TDAPipeline(embedding_dim=3, max_dimension=1)
classifier = RegimeClassifier(tda_pipeline=tda)

# Buffer for rolling window
window_size = 50
telemetry_buffer = []

def process_telemetry(wob, rpm, rop, torque, vibration):
    """Call this for each telemetry packet."""

    # Add to buffer
    telemetry_buffer.append([wob, rpm, rop, torque, vibration])
    if len(telemetry_buffer) > window_size:
        telemetry_buffer.pop(0)

    # Need minimum data
    if len(telemetry_buffer) < 20:
        return None

    # Classify
    data = np.array(telemetry_buffer)
    result = classifier.classify(data)

    return {
        'regime': result['regime'],
        'confidence': result['confidence'],
        'alert': result['regime'] in ['HIGH_VOLATILITY', 'CHAOS', 'TRANSITION']
    }

# Example usage with your data feed:
# while True:
#     packet = get_mwd_packet()  # Your MWD interface
#     status = process_telemetry(
#         packet['wob'], packet['rpm'], packet['rop'],
#         packet['torque'], packet['vibration']
#     )
#     if status and status['alert']:
#         send_alert(f"Regime change: {status['regime']}")
```

---

## 4. BHA Optimization

Given current conditions, suggest optimal parameters:

```python
from jones_framework.domains.drilling.adapter import DrillingDomainAdapter
from jones_framework.sans.mixture_of_experts import MixtureOfExperts
import numpy as np

# Initialize
adapter = DrillingDomainAdapter()
moe = MixtureOfExperts(num_experts=6)

def get_recommendations(current_state):
    """
    current_state: dict with keys:
        - wob: Weight on bit (klbs)
        - rpm: Rotary speed
        - rop: Rate of penetration (ft/hr)
        - torque: Surface torque (ft-lbs)
        - formation: 'sandstone', 'shale', 'carbonate'
        - tvd: True vertical depth (ft)
    """

    # Convert to framework state
    state = adapter.create_state(current_state)

    # Get expert recommendation
    expert_output = moe.process(state)

    return {
        'recommended_wob': expert_output.get('optimal_wob'),
        'recommended_rpm': expert_output.get('optimal_rpm'),
        'dysfunction_risk': expert_output.get('dysfunction_probability'),
        'regime': expert_output.get('current_regime'),
    }

# Example:
recommendations = get_recommendations({
    'wob': 25.0,
    'rpm': 120,
    'rop': 85,
    'torque': 12000,
    'formation': 'shale',
    'tvd': 8500,
})

print(f"Recommended WOB: {recommendations['recommended_wob']}")
print(f"Recommended RPM: {recommendations['recommended_rpm']}")
print(f"Dysfunction risk: {recommendations['dysfunction_risk']:.1%}")
```

---

## 5. Regime Types

The classifier detects these drilling regimes:

| Regime | Meaning | Action |
|--------|---------|--------|
| `STABLE` | Normal drilling, parameters in range | Continue |
| `TRANSITION` | Parameters shifting, watch closely | Monitor |
| `HIGH_VOLATILITY` | Rapid changes, possible dysfunction | Adjust parameters |
| `MOMENTUM` | Consistent trend (good or bad) | Evaluate direction |
| `CHAOS` | Erratic behavior, dysfunction likely | Stop and assess |

---

## 6. Connect to Existing Systems

### WITS Integration

```python
import socket
from jones_framework.perception.regime_classifier import RegimeClassifier

classifier = RegimeClassifier()

def parse_wits_record(data):
    """Parse WITS level 0 record."""
    # Adjust field positions for your WITS setup
    fields = data.split(',')
    return {
        'wob': float(fields[10]),
        'rpm': float(fields[12]),
        'rop': float(fields[15]),
        'torque': float(fields[18]),
        'vibration': float(fields[22]),
    }

# Connect to WITS server
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('wits-server', 5000))

buffer = []
while True:
    data = sock.recv(1024).decode()
    packet = parse_wits_record(data)
    buffer.append([packet['wob'], packet['rpm'], packet['rop'],
                   packet['torque'], packet['vibration']])

    if len(buffer) >= 20:
        result = classifier.classify(np.array(buffer[-50:]))
        if result['regime'] in ['HIGH_VOLATILITY', 'CHAOS']:
            print(f"ALERT: {result['regime']} detected")
```

### REST API

Start the API server:

```bash
uvicorn jones_framework.api.server:app --host 0.0.0.0 --port 8000
```

Then POST telemetry:

```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[25.0, 120, 85, 12000, 0.3], [25.5, 118, 82, 12200, 0.4]],
    "domain": "drilling"
  }'
```

---

## 7. Troubleshooting

**"No module named 'jones_framework'"**
```bash
cd backend && pip install -e .
```

**"ripser not found"**
```bash
pip install ripser persim
```

**"CUDA not available" (not critical)**
- Framework runs on CPU by default
- GPU speeds up TDA computation for large windows
- Install pytorch with CUDA if needed: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

**Regime always shows STABLE**
- Check your telemetry buffer size (need 20+ points)
- Verify data variance - flat data = stable regime
- Check data normalization

---

## 8. Support

- Issues: https://github.com/AvgAi/SupaTrupa/issues
- Live demo: https://drillops.jtech.ai/hc/en-us
- Contact: b.jones@jtech.ai

---

## Appendix: Key Parameters

### TDA Pipeline Settings

```python
TDAPipeline(
    embedding_dim=3,     # Delay embedding dimension (3-5 typical)
    max_dimension=1,     # Max homology dimension (1 = loops)
    n_jobs=-1,           # Parallel processing (-1 = all cores)
)
```

### Regime Classifier Settings

```python
RegimeClassifier(
    confidence_threshold=0.7,  # Min confidence to report regime
    transition_window=10,       # Lookback for transition detection
)
```
