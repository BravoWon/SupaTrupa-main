#!/usr/bin/env python3
"""
Add public API aliases for obfuscated class names.
This script adds alias statements at the end of Python files.
"""
import os
from pathlib import Path

# Base path for the backend
BASE_PATH = Path(__file__).parent.parent / "backend" / "jones_framework"

# Mapping of files to their obfuscated -> public name mappings
ALIAS_MAPPINGS = {
    "perception/metric_warper.py": [
        ("_c0l1AlE", "ValueFunction"),
        ("_cI0lA22", "VolatilityFunction"),
        ("_c0O0A25", "DrillabilityFunction"),
        ("_cOllA2B", "CompositeValueFunction"),
        ("_clIlA2d", "MetricWarper"),
    ],
    "sans/continuity_guard.py": [
        ("_cOlOlc8", "SafetyLevel"),
        ("_cl0Ilc9", "ValidationResult"),
        ("_cIIOlcA", "ContinuityGuard"),
    ],
    "arbitrage/sentiment_vector.py": [],  # Will check
    "arbitrage/linguistic_arbitrage.py": [],  # Will check
    "trading/__init__.py": [],  # Re-exports
    "trading/execution/order_manager.py": [
        ("_cOI0l34", "OrderSide"),
        ("_c0IIl35", "OrderType"),
        ("_cl10l36", "TimeInForce"),
        ("_clO1l37", "OrderStatus"),
        ("_c001l38", "RouteStrategy"),
        ("_clI1l39", "Order"),
        ("_cO1ll3E", "Fill"),
        ("_cO1ll4l", "Position"),
        ("_c1lOl42", "RiskLimits"),
        ("_cllIl4c", "RiskManager"),
        ("_cOl0l4d", "OrderManager"),
    ],
    "strategies/alpha.py": [
        ("_c100l", "StrategyType"),
        ("_c0OO2", "StrategyState"),
        ("_c1l19", "Strategy"),
        ("_cI0I2l", "MomentumStrategy"),
        ("_cO1022", "MeanReversionStrategy"),
        ("_c0II27", "StrategyManager"),
    ],
    "execution/algorithms.py": [
        ("_cl01cB6", "AlgorithmType"),
        ("_clOIcd9", "TWAPAlgorithm"),
        ("_c0OIcdA", "VWAPAlgorithm"),
        ("_cO00cf4", "ExecutionManager"),
    ],
    "simulation/paper_trading.py": [
        ("_cIlOdOf", "SimulationMode"),
        ("_cOO0d2d", "PaperTradingEngine"),
        ("_c00Od42", "BacktestEngine"),
        ("_clIld47", "MonteCarloSimulator"),
        ("_cIlld57", "StressTestSimulator"),
    ],
    "events/sourcing.py": [
        ("_cI0l3cl", "EventType"),
        ("_c0lI3c3", "Event"),
        ("_c0l13d8", "EventStore"),
        ("_c00l3fB", "AuditLogger"),
        ("_c01l4lf", "EventSourcingService"),
    ],
    "api/server.py": [],  # Already has public names
    "config/settings.py": [
        ("_cO007Of", "Environment"),
        ("_c1I0726", "ConfigManager"),
        ("_c1OO743", "FeatureFlagManager"),
        ("_cI0I753", "SecretManager"),
    ],
    "cache/store.py": [
        ("_c0ll9A6", "EvictionPolicy"),
        ("_c0009B7", "MemoryCache"),
        ("_c00O9c9", "QueryCache"),
        ("_cIl19E7", "MarketDataCache"),
    ],
    "scheduler/jobs.py": [
        ("_cI103B", "JobStatus"),
        ("_cIll3f", "Job"),
        ("_c11O64", "Scheduler"),
        ("_clIl86", "RateLimiter"),
        ("_c10192", "TradingScheduler"),
    ],
    "data/connectors/base.py": [
        ("_c0l1688", "DataConnector"),
    ],
    "ml/models.py": [],  # Will check
    "portfolio/optimizer.py": [
        ("_c1I0784", "PortfolioOptimizer"),
        ("_c11078f", "MeanVarianceOptimizer"),
        ("_c11O794", "RiskParityOptimizer"),
        ("_c01O79c", "Rebalancer"),
    ],
    "monitoring/observability.py": [
        ("_cIlIA56", "MetricType"),
        ("_c1I1A68", "Counter"),
        ("_cIlIA6d", "Gauge"),
        ("_cO1lA6f", "Histogram"),
        ("_cI1OA79", "MetricsRegistry"),
        ("_c0O0A7B", "AlertManager"),
        ("_clI0AA5", "MonitoringService"),
    ],
    "risk/engine.py": [
        ("_c1I03B9", "RiskEngine"),
        ("_cIl0387", "RiskMetrics"),
    ],
    "features/store.py": [
        ("_cl1l465", "FeatureStore"),
    ],
    "streaming/core.py": [
        ("_c0OO666", "StreamProcessor"),
    ],
}


def add_aliases_to_file(filepath: Path, aliases: list):
    """Add alias statements to the end of a Python file."""
    if not aliases:
        print(f"Skipping {filepath} - no aliases defined")
        return

    if not filepath.exists():
        print(f"File not found: {filepath}")
        return

    content = filepath.read_text(encoding='utf-8')

    # Check if aliases already exist
    if "# Public API aliases" in content:
        print(f"Aliases already exist in {filepath}")
        return

    # Build alias block
    alias_lines = ["\n\n# Public API aliases for obfuscated classes"]
    for obfuscated, public in aliases:
        alias_lines.append(f"{public} = {obfuscated}")

    new_content = content.rstrip() + "\n".join(alias_lines) + "\n"
    filepath.write_text(new_content, encoding='utf-8')
    print(f"Added {len(aliases)} aliases to {filepath}")


def main():
    for rel_path, aliases in ALIAS_MAPPINGS.items():
        filepath = BASE_PATH / rel_path
        add_aliases_to_file(filepath, aliases)


if __name__ == "__main__":
    main()
