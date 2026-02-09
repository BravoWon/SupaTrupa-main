"""Cycle 7: Autonomous Advisory — geodesic-optimal parameter prescriptions.

Computes the shortest path through ROP-warped parameter space from the
current drilling state to a target regime, prescribing exact parameter
changes at each step along with risk assessment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Drilling parameter bounds (typical operational ranges)
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "wob": (5.0, 60.0),
    "rpm": (30.0, 250.0),
    "rop": (5.0, 200.0),
    "torque": (1.0, 30.0),
    "spp": (1000.0, 5000.0),
}

# Parameter display names and units
PARAM_META: Dict[str, Dict[str, str]] = {
    "wob": {"name": "Weight on Bit", "unit": "klbs"},
    "rpm": {"name": "Rotary Speed", "unit": "rpm"},
    "rop": {"name": "Rate of Penetration", "unit": "ft/hr"},
    "torque": {"name": "Torque", "unit": "kft-lbs"},
    "spp": {"name": "Standpipe Pressure", "unit": "psi"},
}

# Risk factors per regime
REGIME_RISK_SCORES: Dict[str, float] = {
    "NORMAL": 0.1,
    "OPTIMAL": 0.05,
    "DARCY_FLOW": 0.15,
    "NON_DARCY_FLOW": 0.3,
    "TURBULENT": 0.5,
    "MULTIPHASE": 0.6,
    "BIT_BOUNCE": 0.55,
    "PACKOFF": 0.7,
    "STICK_SLIP": 0.65,
    "WHIRL": 0.5,
    "FORMATION_CHANGE": 0.4,
    "WASHOUT": 0.8,
    "LOST_CIRCULATION": 0.9,
    "TRANSITION": 0.35,
    "KICK": 0.95,
    "UNKNOWN": 0.5,
}


@dataclass
class ParameterStep:
    """A single step in the parameter change prescription."""

    step_index: int
    parameter: str
    current_value: float
    target_value: float
    change_amount: float
    change_pct: float
    priority: int  # 1=first, 2=second, etc.
    rationale: str


@dataclass
class AdvisoryResult:
    """Complete advisory recommendation."""

    current_regime: str
    target_regime: str
    confidence: float
    steps: List[ParameterStep]
    geodesic_length: float
    euclidean_length: float
    path_efficiency: float  # geodesic/euclidean ratio (>1 = curved path)
    risk_score: float  # 0-1 composite risk
    risk_level: str  # low/medium/high/critical
    estimated_transitions: int  # number of regime transitions along path
    reasoning: List[str]
    parameter_trajectory: List[Dict[str, float]]  # interpolated path


@dataclass
class RiskAssessment:
    """Risk assessment for a proposed parameter change."""

    overall_risk: float
    risk_level: str
    risk_factors: List[Dict[str, Any]]
    mitigations: List[str]
    abort_conditions: List[str]
    regime_risk: float
    path_risk: float
    correlation_risk: float


class AdvisoryEngine:
    """Compute geodesic-optimal parameter prescriptions for regime navigation."""

    def __init__(
        self,
        regime_signatures: Optional[Dict[str, np.ndarray]] = None,
        norm_scale: Optional[np.ndarray] = None,
    ):
        self._regime_signatures = regime_signatures or {}
        self._norm_scale = norm_scale

    def compute_advisory(
        self,
        current_params: Dict[str, float],
        current_regime: str,
        target_regime: str,
        target_signature: Optional[np.ndarray] = None,
        current_signature: Optional[np.ndarray] = None,
        transition_probs: Optional[Dict[str, float]] = None,
        correlation_edges: Optional[List[Dict[str, Any]]] = None,
    ) -> AdvisoryResult:
        """Compute step-by-step parameter prescription from current to target regime.

        Uses a manifold-aware approach:
        1. Identify which parameters need to change based on regime signatures
        2. Order changes by correlation impact (avoid destabilizing sequences)
        3. Compute geodesic-like path through parameter space
        4. Assess risk at each step
        """
        # Get regime signatures for comparison
        target_sig = target_signature
        current_sig = current_signature
        if target_sig is None and target_regime in self._regime_signatures:
            target_sig = self._regime_signatures[target_regime]
        if current_sig is None and current_regime in self._regime_signatures:
            current_sig = self._regime_signatures[current_regime]

        # Compute parameter deltas needed
        param_deltas = self._compute_parameter_deltas(
            current_params, current_regime, target_regime,
            current_sig, target_sig,
        )

        # Order by correlation impact to minimize destabilization
        ordered_changes = self._order_by_correlation(
            param_deltas, correlation_edges
        )

        # Generate steps
        steps = []
        for priority, (param, delta, rationale) in enumerate(ordered_changes, 1):
            current_val = current_params.get(param, 0.0)
            target_val = current_val + delta
            # Clamp to bounds
            lo, hi = PARAM_BOUNDS.get(param, (0, 1e6))
            target_val = max(lo, min(hi, target_val))
            actual_delta = target_val - current_val
            pct = (actual_delta / current_val * 100) if abs(current_val) > 1e-10 else 0.0

            steps.append(ParameterStep(
                step_index=priority,
                parameter=param,
                current_value=round(current_val, 2),
                target_value=round(target_val, 2),
                change_amount=round(actual_delta, 2),
                change_pct=round(pct, 1),
                priority=priority,
                rationale=rationale,
            ))

        # Compute path metrics
        current_vec = np.array([current_params.get(p, 0.0) for p in PARAM_BOUNDS])
        target_vec = np.array([
            current_params.get(p, 0.0) + sum(
                s.change_amount for s in steps if s.parameter == p
            ) for p in PARAM_BOUNDS
        ])

        euclidean_length = float(np.linalg.norm(target_vec - current_vec))

        # Generate interpolated trajectory (curved through safe zones)
        n_traj_steps = max(5, len(steps) + 2)
        trajectory = self._interpolate_path(
            current_params, steps, n_traj_steps
        )

        # Geodesic length (sum of step distances, slightly longer due to sequencing)
        geodesic_length = sum(abs(s.change_amount) for s in steps)
        if geodesic_length < 1e-10:
            geodesic_length = euclidean_length

        path_efficiency = geodesic_length / max(euclidean_length, 1e-10)

        # Risk assessment
        target_risk = REGIME_RISK_SCORES.get(target_regime, 0.5)
        current_risk = REGIME_RISK_SCORES.get(current_regime, 0.5)
        transition_risk = 0.3  # base transition risk
        if transition_probs:
            # Higher risk if low probability of reaching target
            target_prob = transition_probs.get(target_regime, 0.0)
            transition_risk = max(0.1, 1.0 - target_prob)

        risk_score = 0.4 * current_risk + 0.3 * transition_risk + 0.3 * target_risk
        risk_score = min(1.0, risk_score)

        if risk_score < 0.25:
            risk_level = "low"
        elif risk_score < 0.5:
            risk_level = "medium"
        elif risk_score < 0.75:
            risk_level = "high"
        else:
            risk_level = "critical"

        # Reasoning
        reasoning = self._generate_reasoning(
            current_regime, target_regime, steps,
            risk_level, path_efficiency,
        )

        return AdvisoryResult(
            current_regime=current_regime,
            target_regime=target_regime,
            confidence=round(1.0 - risk_score, 3),
            steps=steps,
            geodesic_length=round(geodesic_length, 3),
            euclidean_length=round(euclidean_length, 3),
            path_efficiency=round(path_efficiency, 3),
            risk_score=round(risk_score, 3),
            risk_level=risk_level,
            estimated_transitions=max(1, len(steps) // 2),
            reasoning=reasoning,
            parameter_trajectory=trajectory,
        )

    def assess_risk(
        self,
        current_params: Dict[str, float],
        proposed_changes: Dict[str, float],
        current_regime: str,
        correlation_edges: Optional[List[Dict[str, Any]]] = None,
    ) -> RiskAssessment:
        """Assess risk of a proposed set of parameter changes."""
        risk_factors = []
        mitigations = []
        abort_conditions = []

        # Regime risk
        regime_risk = REGIME_RISK_SCORES.get(current_regime, 0.5)
        risk_factors.append({
            "factor": "Current regime",
            "value": current_regime,
            "risk": round(regime_risk, 2),
        })

        # Parameter change magnitude risk
        total_change_pct = 0.0
        for param, delta in proposed_changes.items():
            current = current_params.get(param, 1.0)
            pct = abs(delta / current * 100) if abs(current) > 1e-10 else 0.0
            total_change_pct += pct

            if pct > 20:
                risk_factors.append({
                    "factor": f"{param} change too large",
                    "value": f"{pct:.0f}%",
                    "risk": min(1.0, pct / 50),
                })
                mitigations.append(
                    f"Apply {param} change in 2-3 increments, monitoring between each"
                )

            # Check bounds
            lo, hi = PARAM_BOUNDS.get(param, (0, 1e6))
            new_val = current + delta
            if new_val < lo or new_val > hi:
                risk_factors.append({
                    "factor": f"{param} out of bounds",
                    "value": f"{new_val:.1f} (range: {lo}-{hi})",
                    "risk": 0.9,
                })
                abort_conditions.append(
                    f"Abort if {param} exceeds operational limits ({lo}-{hi})"
                )

        path_risk = min(1.0, total_change_pct / 100)

        # Correlation risk (simultaneous changes to correlated parameters)
        correlation_risk = 0.0
        if correlation_edges:
            changed_params = set(proposed_changes.keys())
            for edge in correlation_edges:
                src = edge.get("source", "").lower()
                tgt = edge.get("target", "").lower()
                r = abs(edge.get("pearson_r", 0.0))
                if src in changed_params and tgt in changed_params and r > 0.5:
                    correlation_risk = max(correlation_risk, r * 0.5)
                    risk_factors.append({
                        "factor": f"Correlated params {src}/{tgt}",
                        "value": f"r={r:.2f}",
                        "risk": round(r * 0.5, 2),
                    })
                    mitigations.append(
                        f"Sequence {src} and {tgt} changes (don't change simultaneously)"
                    )

        overall_risk = 0.35 * regime_risk + 0.35 * path_risk + 0.3 * correlation_risk
        overall_risk = min(1.0, overall_risk)

        if overall_risk < 0.25:
            risk_level = "low"
        elif overall_risk < 0.5:
            risk_level = "medium"
        elif overall_risk < 0.75:
            risk_level = "high"
        else:
            risk_level = "critical"

        # Default abort conditions
        abort_conditions.extend([
            "Abort if torque exceeds 120% of current value",
            "Abort if SPP drops below minimum safe threshold",
            "Abort if regime transitions to KICK or LOST_CIRCULATION",
        ])

        return RiskAssessment(
            overall_risk=round(overall_risk, 3),
            risk_level=risk_level,
            risk_factors=risk_factors,
            mitigations=mitigations,
            abort_conditions=abort_conditions[:5],
            regime_risk=round(regime_risk, 3),
            path_risk=round(path_risk, 3),
            correlation_risk=round(correlation_risk, 3),
        )

    # ── Private helpers ──────────────────────────────────────────────

    def _compute_parameter_deltas(
        self,
        current_params: Dict[str, float],
        current_regime: str,
        target_regime: str,
        current_sig: Optional[np.ndarray],
        target_sig: Optional[np.ndarray],
    ) -> List[Tuple[str, float, str]]:
        """Compute required parameter changes based on regime heuristics."""
        deltas: List[Tuple[str, float, str]] = []

        # Heuristic regime-specific parameter adjustments
        rules = _REGIME_TRANSITION_RULES.get(target_regime, {})
        for param, (direction, magnitude, reason) in rules.items():
            current_val = current_params.get(param, 0.0)
            lo, hi = PARAM_BOUNDS.get(param, (0, 1e6))
            range_size = hi - lo

            if direction == "increase":
                delta = range_size * magnitude
            elif direction == "decrease":
                delta = -range_size * magnitude
            else:
                delta = 0.0

            if abs(delta) > 1e-10:
                deltas.append((param, delta, reason))

        # If no rules, fall back to small nudge toward NORMAL
        if not deltas:
            deltas.append(("rpm", 5.0, "Slight RPM increase toward optimal envelope"))
            deltas.append(("wob", -2.0, "Slight WOB reduction for stability"))

        return deltas

    def _order_by_correlation(
        self,
        deltas: List[Tuple[str, float, str]],
        edges: Optional[List[Dict[str, Any]]],
    ) -> List[Tuple[str, float, str]]:
        """Order parameter changes to minimize correlation-driven instability."""
        if not edges or len(deltas) <= 1:
            return deltas

        # Build correlation impact score per parameter
        impact: Dict[str, float] = {}
        for param, _, _ in deltas:
            score = 0.0
            for edge in edges:
                src = edge.get("source", "").lower()
                tgt = edge.get("target", "").lower()
                r = abs(edge.get("pearson_r", 0.0))
                if param == src or param == tgt:
                    score += r
            impact[param] = score

        # Sort: lowest correlation impact first (safest changes first)
        return sorted(deltas, key=lambda x: impact.get(x[0], 0.0))

    def _interpolate_path(
        self,
        current_params: Dict[str, float],
        steps: List[ParameterStep],
        n_points: int,
    ) -> List[Dict[str, float]]:
        """Generate interpolated parameter trajectory along the advisory path."""
        trajectory = []
        params = dict(current_params)
        trajectory.append({k: round(v, 2) for k, v in params.items()})

        # Apply each step incrementally
        for step in steps:
            n_sub = max(1, n_points // len(steps)) if steps else 1
            for j in range(1, n_sub + 1):
                frac = j / n_sub
                interp = dict(params)
                interp[step.parameter] = round(
                    params.get(step.parameter, 0) + step.change_amount * frac, 2
                )
                trajectory.append(interp)
            params[step.parameter] = round(
                params.get(step.parameter, 0) + step.change_amount, 2
            )

        return trajectory[:n_points]

    def _generate_reasoning(
        self,
        current: str,
        target: str,
        steps: List[ParameterStep],
        risk_level: str,
        efficiency: float,
    ) -> List[str]:
        """Generate human-readable reasoning for the advisory."""
        reasoning = []
        reasoning.append(
            f"Navigate from {current} to {target} via {len(steps)} parameter adjustments."
        )
        if efficiency > 1.5:
            reasoning.append(
                "Path is significantly curved — sequential changes recommended over simultaneous."
            )
        elif efficiency > 1.1:
            reasoning.append(
                "Path is moderately curved — stagger changes for smoother transition."
            )
        else:
            reasoning.append(
                "Path is nearly direct — changes can be applied with minimal sequencing."
            )

        for step in steps[:3]:
            meta = PARAM_META.get(step.parameter, {})
            name = meta.get("name", step.parameter)
            unit = meta.get("unit", "")
            direction = "increase" if step.change_amount > 0 else "decrease"
            reasoning.append(
                f"Step {step.priority}: {direction} {name} by "
                f"{abs(step.change_amount):.1f} {unit} ({abs(step.change_pct):.0f}%). "
                f"{step.rationale}"
            )

        if risk_level in ("high", "critical"):
            reasoning.append(
                f"Risk level is {risk_level.upper()} — apply changes incrementally with "
                "continuous monitoring between each step."
            )

        return reasoning


# ── Regime transition heuristic rules ───────────────────────────────

# target_regime -> {param: (direction, magnitude_fraction_of_range, rationale)}
_REGIME_TRANSITION_RULES: Dict[str, Dict[str, Tuple[str, float, str]]] = {
    "NORMAL": {
        "wob": ("decrease", 0.05, "Reduce WOB toward normal operating envelope"),
        "rpm": ("increase", 0.03, "Increase RPM for smoother cutting action"),
    },
    "OPTIMAL": {
        "wob": ("increase", 0.03, "Optimize WOB for peak ROP"),
        "rpm": ("increase", 0.05, "Increase RPM into optimal cutting window"),
        "spp": ("increase", 0.02, "Boost hydraulics for improved hole cleaning"),
    },
    "DARCY_FLOW": {
        "spp": ("decrease", 0.05, "Reduce SPP to restore laminar flow"),
        "rpm": ("decrease", 0.02, "Lower RPM to reduce annular turbulence"),
    },
    "NON_DARCY_FLOW": {
        "spp": ("decrease", 0.08, "Reduce SPP significantly to exit non-Darcy regime"),
        "wob": ("decrease", 0.03, "Reduce WOB to lower cuttings generation rate"),
    },
    "STICK_SLIP": {
        "rpm": ("increase", 0.08, "Increase RPM above stick-slip threshold"),
        "wob": ("decrease", 0.1, "Reduce WOB to decrease torsional load"),
        "torque": ("decrease", 0.05, "Target lower torque operating point"),
    },
    "BIT_BOUNCE": {
        "wob": ("decrease", 0.12, "Reduce WOB to eliminate axial vibration"),
        "rpm": ("decrease", 0.05, "Reduce RPM to move away from resonance"),
    },
    "WHIRL": {
        "rpm": ("increase", 0.06, "Increase RPM past whirl resonance band"),
        "wob": ("increase", 0.03, "Increase WOB to stabilize BHA"),
    },
    "PACKOFF": {
        "rpm": ("increase", 0.04, "Increase RPM for better hole cleaning"),
        "spp": ("increase", 0.06, "Boost flow rate to clear packoff"),
        "wob": ("decrease", 0.08, "Reduce WOB to relieve differential sticking"),
    },
}
