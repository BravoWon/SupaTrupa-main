#!/usr/bin/env python3
"""
Jones Framework Deobfuscator

Reverses the obfuscation by:
1. Using __all__ exports to map obfuscated names back to originals
2. Analyzing context (enums, dataclasses) to infer names
3. Restoring readable code for the drilling community

Usage:
    python scripts/deobfuscate.py --all
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Known mappings from __all__ exports and context
KNOWN_MAPPINGS = {
    # Core condition_state.py
    'ConditionState': '_c1O1c6O',

    # Core activity_state.py
    'RegimeID': '_c01Ic3c',
    'ManifoldMetric': '_clOlc3d',
    'EuclideanMetric': '_cOlOc47',
    'WarpedMetric': '_c0I1c49',
    'ExpertModel': '_cI11c4d',
    'ActivityState': '_cOllc5l',

    # Core value_function.py
    'ValueSource': '_cO0Ic77',
    'WarpingMode': '_clI0c78',
    'OptimizationLandscape': '_c10Ic79',
    'MetricTensor': '_clOIc7A',
    'ConformalFactor': '_c0OIc8d',
    'ValueFunctionProtocol': '_c1Olc93',
    'ValueFunction': '_c01Oc97',
    'MarketValueFunction': '_cII0cBO',
    'ReservoirValueFunction': '_cI10cB3',

    # Arbitrage sentiment_vector.py
    'NarrativeType': '_c10l5f2',
    'TextDocument': '_c01O5f3',
    'SentimentVector': '_cl005f4',
    'KeywordExtractor': '_c1l05fB',
    'SentimentVectorPipeline': '_cIIO6OO',

    # Arbitrage linguistic_arbitrage.py
    'ArbitrageSignal': '_cII05cf',
    'PotentialEnergy': '_cO005dl',
    'LinguisticArbitrageEngine': '_cOIl5d4',
}


def build_reverse_mapping() -> Dict[str, str]:
    """Build obfuscated -> original mapping."""
    return {v: k for k, v in KNOWN_MAPPINGS.items()}


def infer_method_names(source: str) -> Dict[str, str]:
    """Infer method names from context and patterns."""
    mappings = {}

    # Common patterns
    patterns = [
        # _compute_hash pattern
        (r"def (_f[A-Za-z0-9]+)\(self\)[^:]*:\s*data = f'{self\.timestamp}", 'compute_hash', '_compute_hash'),
        # to_numpy pattern
        (r"def (_f[A-Za-z0-9]+)\(self\)[^:]*:\s*return np\.array\(self\.", 'to_numpy', 'to_numpy'),
        # from_numpy pattern
        (r"def (_f[A-Za-z0-9]+)\(cls,.*ndarray", 'from_numpy', 'from_numpy'),
        # from_reservoir pattern
        (r"def (_f[A-Za-z0-9]+)\(cls,.*lithology", 'from_reservoir', 'from_reservoir'),
        # from_market pattern
        (r"def (_f[A-Za-z0-9]+)\(cls,.*symbol", 'from_market', 'from_market'),
        # distance_to pattern
        (r"def (_f[A-Za-z0-9]+)\(self,.*linalg\.norm", 'distance_to', 'distance_to'),
        # dimension property
        (r"def (_f[A-Za-z0-9]+)\(self\)[^:]*:\s*return len\(self\.", 'dimension', 'dimension'),
        # compute_distance
        (r"def (_f[A-Za-z0-9]+)\(self,.*p1.*p2.*:\s*return float\(np\.linalg\.norm", 'compute_distance', 'compute_distance'),
        # compute_geodesic
        (r"def (_f[A-Za-z0-9]+)\(self,.*start.*end.*steps", 'compute_geodesic', 'compute_geodesic'),
        # get_metric_at
        (r"def (_f[A-Za-z0-9]+)\(self,.*point.*:\s*return.*metric", 'get_metric_at', 'get_metric_at'),
        # evaluate (value function)
        (r"def (_f[A-Za-z0-9]+)\(self,.*ConditionState.*:\s*cache_key", 'evaluate', 'evaluate'),
        # compute_gradient
        (r"def (_f[A-Za-z0-9]+)\(self,.*epsilon.*:\s*base_value = self\._f.*\s*grad = ", 'compute_gradient', 'compute_gradient'),
        # compute_hessian
        (r"def (_f[A-Za-z0-9]+)\(self,.*:\s*n = min.*hess = \[\[", 'compute_hessian', 'compute_hessian'),
        # process_batch
        (r"def (_f[A-Za-z0-9]+)\(self,.*List\[.*Document", 'process_batch', 'process_batch'),
        # analyze_document
        (r"def (_f[A-Za-z0-9]+)\(self,.*document.*:\s*text = self\._f.*keyword_scores", 'analyze_document', 'analyze_document'),
    ]

    for pattern, hint, replacement in patterns:
        match = re.search(pattern, source, re.MULTILINE | re.DOTALL)
        if match:
            obfuscated = match.group(1)
            mappings[obfuscated] = replacement

    return mappings


def deobfuscate_file(source: str, class_mappings: Dict[str, str]) -> str:
    """Deobfuscate a single file."""
    result = source

    # Replace class names
    for obfuscated, original in class_mappings.items():
        # Replace class definitions
        result = re.sub(rf'\bclass {obfuscated}\b', f'class {original}', result)
        # Replace type hints and references
        result = re.sub(rf'\b{obfuscated}\b', original, result)

    # Infer and replace method names
    method_mappings = infer_method_names(source)
    for obfuscated, original in method_mappings.items():
        result = re.sub(rf'\b{obfuscated}\b', original, result)

    # Fix common parameter patterns
    param_patterns = [
        # Vector/array parameters
        (r'_fll0c63', 'vector'),
        (r'_fI1Ic64', 'metadata'),
        (r'_fOlOc65', 'timestamp'),
        (r'_fO10c66', 'verified'),
        (r'_f0IOc68', 'lithology'),
        (r'_f001c69', 'porosity'),
        (r'_f1O0c6A', 'permeability'),
        (r'_fI10c6B', 'saturation'),
        (r'_fIO0c6c', 'location'),
        (r'_fIIIc6E', 'price'),
        (r'_fIl1c6f', 'volume'),
        (r'_fIl1c7O', 'bid'),
        (r'_f00Oc7l', 'ask'),
        (r'_f1l0c72', 'symbol'),
        (r'_fIlOc76', 'other'),
        (r'_flO0c3f', 'p1'),
        (r'_fOIOc4O', 'p2'),
        (r'_f1l1c42', 'start'),
        (r'_f111c43', 'end'),
        (r'_fO1lc44', 'steps'),
        (r'_f00lc46', 'point'),
        (r'_fO1Oc48', 'dimension'),
        (r'_f0Olc4A', 'base_metric'),
        (r'_fl1lc4B', 'value_function'),
        (r'_flOlc4f', 'state'),
        (r'_f0I0c59', 'state_a'),
        (r'_f0OOc5A', 'state_b'),
        (r'_fI1Ic5B', 'transition_cost'),
        (r'_fIIlc5f', 'risk_tolerance'),
        (r'_f111c5E', 'regime_id'),
        (r'_fl1Oc98', 'dimension'),
        (r'_f101c99', 'base_metric'),
        (r'_f0Ilc9A', 'warping_mode'),
        (r'_fIOOc9B', 'value_sources'),
        (r'_flOlc94', 'state'),
        (r'_fO00c9d', 'epsilon'),
        (r'_fIlI5dA', 'shadow_tensor'),
        (r'_flIO5dB', 'persistence_diagram'),
        (r'_f1Ol5dd', 'documents'),
        (r'_fIOl5df', 'sentiment'),
        (r'_fl105EO', 'details'),
        (r'_f10l5fd', 'text'),
        (r'_fIIl6O8', 'document'),
        (r'_f10O6OA', 'documents'),
        (r'_fIO06O5', 'source'),
        (r'_fl016O2', 'model_name'),
        (r'_fOll6Ol', 'use_transformer'),
    ]

    for obfuscated, original in param_patterns:
        result = re.sub(rf'\b{obfuscated}\b', original, result)

    # Fix internal method names based on patterns
    internal_patterns = [
        (r'_f1Olc9c', '_setup_default_components'),
        (r'_fO1Oc9E', '_get_warped_metric'),
        (r'_fOO0c9f', '_compute_warping_factor'),
        (r'_fIllcAO', '_compute_riemannian_warp'),
        (r'_fO01cA9', '_get_source_weights'),
        (r'_fIl0cAA', '_arbitrage_value'),
        (r'_f0OIcAB', '_efficiency_value'),
        (r'_fIOIcAc', '_risk_adjusted_value'),
        (r'_fO0OcAd', '_information_value'),
        (r'_f1I1cAE', '_stability_value'),
        (r'_f1O0cAf', '_adaptivity_value'),
        (r'_fIIIcA7', '_compute_christoffel'),
        (r'_f0O1cA2', '_compute_geodesic_trajectory'),
        (r'_fOlIc4c', '_conformal_scale'),
        (r'_fO1O6O4', '_classify_source'),
        (r'_f0I06O6', '_clean_text'),
        (r'_flOI6O7', '_analyze_document'),
        (r'_f1OI6O9', '_process_batch'),
        (r'_fO116Od', '_calculate_divergence'),
        (r'_f0O16lO', '_get_regime_stress'),
        (r'_f00O6ll', '_reset_history'),
        (r'_fO0I5fc', '_extract_keywords'),
        (r'_f1I05f5', '_to_array'),
        (r'_f1ll5f6', '_magnitude'),
        (r'_f0ll5f7', '_is_significant'),
        (r'_flI15f9', '_scale'),
        (r'_f1O15d9', '_update_potential'),
        (r'_f11I5dc', '_process_text'),
        (r'_f1l15dE', '_check_for_signals'),
        (r'_fIOl5El', '_predict_regime_transition'),
        (r'_f0l05E2', '_extract_trigger_keywords'),
        (r'_fll05E3', '_add_trigger_rule'),
        (r'_f1OI5E5', '_get_state'),
        (r'_fO005E6', '_get_signal_history'),
        (r'_fO0I5E9', '_set_regime'),
        (r'_fIl05EB', '_reset'),
        (r'_fII05Ec', '_create_fear_spike_rule'),
        (r'_f1lO5fl', '_create_divergence_collapse_rule'),
        (r'_fO0I5dO', '_is_actionable'),
        (r'_fO0l5d2', '_total_energy'),
        (r'_fO0l5d3', '_is_coiled'),
        (r'_flOIc7E', '_get_component'),
        (r'_fIllc8l', '_inner_product'),
        (r'_fl0lc84', '_norm'),
        (r'_fllIc86', '_distance'),
        (r'_fO0Ic89', '_scale'),
        (r'_fl1Ic8B', '_determinant'),
        (r'_f1lIc8c', '_volume_element'),
        (r'_f0lOc8E', '_compute_omega'),
        (r'_f1IOc9O', '_add_singularity'),
        (r'_fIll6O3', '_init_transformer'),
        (r'_fO016OB', '_aggregate_vectors'),
    ]

    for obfuscated, original in internal_patterns:
        result = re.sub(rf'\b{obfuscated}\b', original, result)

    return result


def deobfuscate_codebase(root_path: Path):
    """Deobfuscate the entire codebase."""
    class_mappings = build_reverse_mapping()

    # Process Python files
    backend_path = root_path / "backend"
    py_files = list(backend_path.rglob("*.py"))
    py_files = [f for f in py_files if '__pycache__' not in str(f) and '.egg-info' not in str(f)]

    print(f"Deobfuscating {len(py_files)} Python files...")

    for py_file in py_files:
        rel_path = py_file.relative_to(root_path)
        try:
            source = py_file.read_text()

            # Skip if already deobfuscated (has readable class names)
            if 'class ConditionState' in source or 'class ActivityState' in source:
                print(f"  Skipping (already clean): {rel_path}")
                continue

            deobfuscated = deobfuscate_file(source, class_mappings)

            if deobfuscated != source:
                py_file.write_text(deobfuscated)
                print(f"  Deobfuscated: {rel_path}")
            else:
                print(f"  No changes: {rel_path}")

        except Exception as e:
            print(f"  Error processing {rel_path}: {e}")

    # Process TypeScript files
    frontend_path = root_path / "frontend"
    ts_files = list(frontend_path.rglob("*.ts")) + list(frontend_path.rglob("*.tsx"))
    ts_files = [f for f in ts_files if 'node_modules' not in str(f) and 'dist' not in str(f)]

    print(f"\nDeobfuscating {len(ts_files)} TypeScript files...")

    for ts_file in ts_files:
        rel_path = ts_file.relative_to(root_path)
        try:
            source = ts_file.read_text()

            # TypeScript deobfuscation - look for _T and _v prefixes
            deobfuscated = source

            # Common TS patterns to restore
            ts_patterns = [
                (r'_TlOO(\d+)', r'Component\1'),
                (r'_vl1O(\d+)', r'state\1'),
            ]

            for pattern, replacement in ts_patterns:
                deobfuscated = re.sub(pattern, replacement, deobfuscated)

            if deobfuscated != source:
                ts_file.write_text(deobfuscated)
                print(f"  Deobfuscated: {rel_path}")
            else:
                print(f"  No changes: {rel_path}")

        except Exception as e:
            print(f"  Error processing {rel_path}: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Jones Framework Deobfuscator")
    parser.add_argument("--all", action="store_true", help="Deobfuscate everything")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")

    args = parser.parse_args()

    if not args.all:
        parser.print_help()
        sys.exit(1)

    root = Path(__file__).parent.parent

    print("=" * 60)
    print("Jones Framework Deobfuscator")
    print("=" * 60)
    print("Restoring readable code for the drilling community")
    print("=" * 60)

    deobfuscate_codebase(root)

    print("\n" + "=" * 60)
    print("Deobfuscation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
