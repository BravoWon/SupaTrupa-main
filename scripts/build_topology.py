#!/usr/bin/env python3
"""
Topology-Aware Build System
===========================

Maps component dependencies across the manifold and generates
optimized build configurations based on available resources.

This system treats the codebase as a topological space where:
- Components are nodes
- Dependencies are edges
- Build order follows geodesics through the dependency manifold
- Resource allocation is optimized via spectral embedding

Usage:
    python build_topology.py --analyze          # Analyze component topology
    python build_topology.py --plan core        # Plan core-only build
    python build_topology.py --plan full        # Plan full build
    python build_topology.py --optimize 4GB     # Optimize for memory constraint
    python build_topology.py --visualize        # Output DOT graph
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
from pathlib import Path
import os

# =============================================================================
# Build Compartments - Divisible Yet Connected
# =============================================================================

class Compartment(Enum):
    """Build compartments representing divisible subsystems."""
    CORE = "core"
    TDA = "tda"
    SANS = "sans"
    ML = "ml"
    API = "api"
    UI = "ui"
    DEV = "dev"


@dataclass
class ComponentSpec:
    """Specification for a buildable component."""
    name: str
    compartment: Compartment
    dependencies: List[str] = field(default_factory=list)
    python_packages: List[str] = field(default_factory=list)
    npm_packages: List[str] = field(default_factory=list)
    estimated_size_mb: float = 0.0
    requires_gpu: bool = False
    optional: bool = False


@dataclass
class BuildPlan:
    """Optimized build plan for a target configuration."""
    target: str
    components: List[ComponentSpec]
    total_size_mb: float
    install_order: List[str]
    parallelizable_groups: List[List[str]]
    resource_requirements: Dict[str, any]


# =============================================================================
# Component Registry - The Build Manifold
# =============================================================================

COMPONENT_REGISTRY: Dict[str, ComponentSpec] = {
    # CORE Compartment - Foundation
    "Tensor": ComponentSpec(
        name="Tensor",
        compartment=Compartment.CORE,
        dependencies=[],
        python_packages=["numpy>=1.21.0"],
        estimated_size_mb=5.0
    ),
    "ConditionState": ComponentSpec(
        name="ConditionState",
        compartment=Compartment.CORE,
        dependencies=["Tensor"],
        python_packages=[],
        estimated_size_mb=0.5
    ),
    "ActivityState": ComponentSpec(
        name="ActivityState",
        compartment=Compartment.CORE,
        dependencies=["ConditionState"],
        python_packages=[],
        estimated_size_mb=0.5
    ),
    "ManifoldBridge": ComponentSpec(
        name="ManifoldBridge",
        compartment=Compartment.CORE,
        dependencies=["Tensor", "ConditionState"],
        python_packages=["scipy>=1.7.0"],
        estimated_size_mb=15.0
    ),
    "ShadowTensor": ComponentSpec(
        name="ShadowTensor",
        compartment=Compartment.CORE,
        dependencies=["Tensor", "ConditionState"],
        python_packages=[],
        estimated_size_mb=1.0
    ),

    # TDA Compartment - Topological Data Analysis
    "TDAPipeline": ComponentSpec(
        name="TDAPipeline",
        compartment=Compartment.TDA,
        dependencies=["Tensor", "ShadowTensor"],
        python_packages=["ripser>=0.6.0", "persim>=0.3.0"],
        estimated_size_mb=25.0,
        optional=True
    ),
    "PersistenceLandscape": ComponentSpec(
        name="PersistenceLandscape",
        compartment=Compartment.TDA,
        dependencies=["TDAPipeline"],
        python_packages=[],
        estimated_size_mb=0.5
    ),
    "PersistenceSilhouette": ComponentSpec(
        name="PersistenceSilhouette",
        compartment=Compartment.TDA,
        dependencies=["TDAPipeline"],
        python_packages=[],
        estimated_size_mb=0.5
    ),
    "PersistenceImage": ComponentSpec(
        name="PersistenceImage",
        compartment=Compartment.TDA,
        dependencies=["TDAPipeline"],
        python_packages=[],
        estimated_size_mb=0.5
    ),
    "StreamingTDA": ComponentSpec(
        name="StreamingTDA",
        compartment=Compartment.TDA,
        dependencies=["TDAPipeline", "PersistenceLandscape"],
        python_packages=[],
        estimated_size_mb=1.0
    ),
    "RegimeClassifier": ComponentSpec(
        name="RegimeClassifier",
        compartment=Compartment.TDA,
        dependencies=["TDAPipeline", "ActivityState"],
        python_packages=["scikit-learn>=1.0.0"],
        estimated_size_mb=20.0
    ),

    # SANS Compartment - Specialized Adaptive Neural Systems
    "MixtureOfExperts": ComponentSpec(
        name="MixtureOfExperts",
        compartment=Compartment.SANS,
        dependencies=["ActivityState", "RegimeClassifier"],
        python_packages=[],
        estimated_size_mb=2.0
    ),
    "LoRAAdapter": ComponentSpec(
        name="LoRAAdapter",
        compartment=Compartment.SANS,
        dependencies=["Tensor"],
        python_packages=[],
        estimated_size_mb=1.0
    ),
    "ContinuityGuard": ComponentSpec(
        name="ContinuityGuard",
        compartment=Compartment.SANS,
        dependencies=["ActivityState", "MixtureOfExperts"],
        python_packages=[],
        estimated_size_mb=0.5
    ),

    # ML Compartment - Machine Learning (Optional Heavy)
    "InferenceEngine": ComponentSpec(
        name="InferenceEngine",
        compartment=Compartment.ML,
        dependencies=["MixtureOfExperts", "LoRAAdapter"],
        python_packages=["torch>=2.0.0"],
        estimated_size_mb=800.0,
        requires_gpu=True,
        optional=True
    ),
    "Transformers": ComponentSpec(
        name="Transformers",
        compartment=Compartment.ML,
        dependencies=["InferenceEngine"],
        python_packages=["transformers>=4.25.0"],
        estimated_size_mb=500.0,
        requires_gpu=True,
        optional=True
    ),

    # API Compartment - Server Components
    "APIServer": ComponentSpec(
        name="APIServer",
        compartment=Compartment.API,
        dependencies=["ManifoldBridge", "MixtureOfExperts"],
        python_packages=["fastapi>=0.104.0", "uvicorn[standard]>=0.24.0"],
        estimated_size_mb=30.0
    ),
    "WebSocketManager": ComponentSpec(
        name="WebSocketManager",
        compartment=Compartment.API,
        dependencies=["APIServer"],
        python_packages=["websockets>=12.0"],
        estimated_size_mb=5.0
    ),

    # UI Compartment - Frontend
    "ReactApp": ComponentSpec(
        name="ReactApp",
        compartment=Compartment.UI,
        dependencies=[],
        npm_packages=["react", "react-dom", "@tanstack/react-query"],
        estimated_size_mb=50.0
    ),
    "ThreeJS": ComponentSpec(
        name="ThreeJS",
        compartment=Compartment.UI,
        dependencies=["ReactApp"],
        npm_packages=["three", "@react-three/fiber", "@react-three/drei"],
        estimated_size_mb=30.0
    ),
    "Recharts": ComponentSpec(
        name="Recharts",
        compartment=Compartment.UI,
        dependencies=["ReactApp"],
        npm_packages=["recharts"],
        estimated_size_mb=10.0
    ),

    # DEV Compartment - Development Tools
    "TestFramework": ComponentSpec(
        name="TestFramework",
        compartment=Compartment.DEV,
        dependencies=["ManifoldBridge"],
        python_packages=["pytest>=7.0.0", "pytest-cov>=4.0.0"],
        estimated_size_mb=15.0,
        optional=True
    ),
    "Linting": ComponentSpec(
        name="Linting",
        compartment=Compartment.DEV,
        dependencies=[],
        python_packages=["black>=23.0.0", "mypy>=1.0.0"],
        estimated_size_mb=20.0,
        optional=True
    ),
}


# =============================================================================
# Build Profiles - Predefined Configurations
# =============================================================================

BUILD_PROFILES: Dict[str, List[Compartment]] = {
    "core": [Compartment.CORE],
    "tda": [Compartment.CORE, Compartment.TDA],
    "sans": [Compartment.CORE, Compartment.TDA, Compartment.SANS],
    "api": [Compartment.CORE, Compartment.TDA, Compartment.SANS, Compartment.API],
    "ml": [Compartment.CORE, Compartment.TDA, Compartment.SANS, Compartment.ML],
    "ui": [Compartment.UI],
    "full": [Compartment.CORE, Compartment.TDA, Compartment.SANS, Compartment.API, Compartment.UI],
    "dev": [Compartment.CORE, Compartment.TDA, Compartment.SANS, Compartment.API, Compartment.UI, Compartment.DEV],
    "ml-full": [Compartment.CORE, Compartment.TDA, Compartment.SANS, Compartment.ML, Compartment.API, Compartment.UI, Compartment.DEV],
}


# =============================================================================
# Topology Analyzer
# =============================================================================

class TopologyAnalyzer:
    """Analyzes the component dependency topology."""

    def __init__(self, registry: Dict[str, ComponentSpec]):
        self.registry = registry
        self._adjacency: Dict[str, Set[str]] = {}
        self._reverse_adjacency: Dict[str, Set[str]] = {}
        self._build_adjacency()

    def _build_adjacency(self):
        """Build adjacency lists from component dependencies."""
        for name, spec in self.registry.items():
            self._adjacency[name] = set(spec.dependencies)
            self._reverse_adjacency.setdefault(name, set())
            for dep in spec.dependencies:
                self._reverse_adjacency.setdefault(dep, set()).add(name)

    def get_compartment_components(self, compartment: Compartment) -> List[str]:
        """Get all components in a compartment."""
        return [name for name, spec in self.registry.items()
                if spec.compartment == compartment]

    def get_transitive_dependencies(self, component: str) -> Set[str]:
        """Get all transitive dependencies of a component."""
        visited = set()
        stack = [component]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            if current in self._adjacency:
                stack.extend(self._adjacency[current])
        visited.discard(component)
        return visited

    def topological_sort(self, components: Set[str]) -> List[str]:
        """Return components in dependency order (topological sort)."""
        in_degree = {c: 0 for c in components}
        for c in components:
            for dep in self._adjacency.get(c, set()):
                if dep in components:
                    in_degree[c] += 1

        queue = [c for c in components if in_degree[c] == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)
            for dependent in self._reverse_adjacency.get(current, set()):
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        return result

    def find_parallelizable_groups(self, ordered: List[str]) -> List[List[str]]:
        """Group components that can be installed in parallel."""
        groups = []
        remaining = set(ordered)
        installed = set()

        while remaining:
            # Find all components whose dependencies are satisfied
            ready = []
            for c in remaining:
                deps = self._adjacency.get(c, set())
                if deps.issubset(installed):
                    ready.append(c)

            if not ready:
                # Break cycle by taking first remaining
                ready = [next(iter(remaining))]

            groups.append(ready)
            installed.update(ready)
            remaining -= set(ready)

        return groups

    def compute_betti_numbers(self, components: Set[str]) -> Tuple[int, int]:
        """
        Compute Betti numbers for the component subgraph.

        β₀ = number of connected components
        β₁ = number of cycles (edges - vertices + components)
        """
        # Find connected components
        visited = set()
        num_components = 0

        def dfs(node, component_set):
            if node in visited or node not in components:
                return
            visited.add(node)
            component_set.add(node)
            for neighbor in self._adjacency.get(node, set()):
                dfs(neighbor, component_set)
            for neighbor in self._reverse_adjacency.get(node, set()):
                dfs(neighbor, component_set)

        for c in components:
            if c not in visited:
                comp_set = set()
                dfs(c, comp_set)
                num_components += 1

        # Count edges within the subgraph
        num_edges = sum(
            len(self._adjacency.get(c, set()) & components)
            for c in components
        )

        beta_0 = num_components
        beta_1 = num_edges - len(components) + num_components

        return beta_0, beta_1

    def to_dot(self, components: Optional[Set[str]] = None) -> str:
        """Generate DOT format graph visualization."""
        if components is None:
            components = set(self.registry.keys())

        lines = ["digraph BuildTopology {"]
        lines.append("  rankdir=BT;")
        lines.append("  node [shape=box];")

        # Color by compartment
        colors = {
            Compartment.CORE: "#4A90D9",
            Compartment.TDA: "#50C878",
            Compartment.SANS: "#FFD700",
            Compartment.ML: "#FF6B6B",
            Compartment.API: "#9B59B6",
            Compartment.UI: "#E67E22",
            Compartment.DEV: "#95A5A6",
        }

        for name in components:
            spec = self.registry.get(name)
            if spec:
                color = colors.get(spec.compartment, "#CCCCCC")
                lines.append(f'  "{name}" [fillcolor="{color}", style=filled];')

        for name in components:
            for dep in self._adjacency.get(name, set()):
                if dep in components:
                    lines.append(f'  "{dep}" -> "{name}";')

        lines.append("}")
        return "\n".join(lines)


# =============================================================================
# Build Planner
# =============================================================================

class BuildPlanner:
    """Plans optimized builds based on target and constraints."""

    def __init__(self, registry: Dict[str, ComponentSpec]):
        self.registry = registry
        self.analyzer = TopologyAnalyzer(registry)

    def plan_build(self, profile: str, memory_limit_mb: Optional[float] = None,
                   exclude_gpu: bool = False) -> BuildPlan:
        """Generate an optimized build plan."""
        compartments = BUILD_PROFILES.get(profile, BUILD_PROFILES["full"])

        # Collect components from selected compartments
        components = set()
        for comp in compartments:
            components.update(self.analyzer.get_compartment_components(comp))

        # Add transitive dependencies
        all_deps = set()
        for c in components:
            all_deps.update(self.analyzer.get_transitive_dependencies(c))
        components.update(all_deps)

        # Filter by constraints
        if exclude_gpu:
            components = {c for c in components
                         if not self.registry[c].requires_gpu}

        if memory_limit_mb:
            components = self._fit_to_memory(components, memory_limit_mb)

        # Sort and plan
        ordered = self.analyzer.topological_sort(components)
        parallel_groups = self.analyzer.find_parallelizable_groups(ordered)

        # Calculate totals
        total_size = sum(self.registry[c].estimated_size_mb for c in components)

        # Get component specs
        specs = [self.registry[c] for c in ordered]

        # Resource requirements
        beta_0, beta_1 = self.analyzer.compute_betti_numbers(components)
        resources = {
            "memory_mb": total_size * 1.5,  # Account for runtime overhead
            "components": len(components),
            "parallel_groups": len(parallel_groups),
            "betti_0": beta_0,
            "betti_1": beta_1,
        }

        return BuildPlan(
            target=profile,
            components=specs,
            total_size_mb=total_size,
            install_order=ordered,
            parallelizable_groups=parallel_groups,
            resource_requirements=resources
        )

    def _fit_to_memory(self, components: Set[str], limit_mb: float) -> Set[str]:
        """Remove optional components to fit memory constraint."""
        result = set(components)
        total = sum(self.registry[c].estimated_size_mb for c in result)

        if total <= limit_mb:
            return result

        # Remove optional components by size (largest first)
        optional = sorted(
            [c for c in result if self.registry[c].optional],
            key=lambda c: self.registry[c].estimated_size_mb,
            reverse=True
        )

        for c in optional:
            if total <= limit_mb:
                break
            result.discard(c)
            total -= self.registry[c].estimated_size_mb

        return result


# =============================================================================
# Output Formatters
# =============================================================================

def format_plan_text(plan: BuildPlan) -> str:
    """Format build plan as human-readable text."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"BUILD PLAN: {plan.target.upper()}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total Components: {len(plan.components)}")
    lines.append(f"Estimated Size: {plan.total_size_mb:.1f} MB")
    lines.append(f"Parallel Groups: {len(plan.parallelizable_groups)}")
    lines.append(f"Topology B0={plan.resource_requirements['betti_0']}, B1={plan.resource_requirements['betti_1']}")
    lines.append("")

    lines.append("Install Order:")
    lines.append("-" * 40)
    for i, group in enumerate(plan.parallelizable_groups, 1):
        if len(group) > 1:
            lines.append(f"  Phase {i} (parallel): {', '.join(group)}")
        else:
            lines.append(f"  Phase {i}: {group[0]}")

    lines.append("")
    lines.append("Python Packages:")
    lines.append("-" * 40)
    packages = set()
    for spec in plan.components:
        packages.update(spec.python_packages)
    for pkg in sorted(packages):
        lines.append(f"  - {pkg}")

    if any(spec.npm_packages for spec in plan.components):
        lines.append("")
        lines.append("NPM Packages:")
        lines.append("-" * 40)
        npm_packages = set()
        for spec in plan.components:
            npm_packages.update(spec.npm_packages)
        for pkg in sorted(npm_packages):
            lines.append(f"  - {pkg}")

    return "\n".join(lines)


def format_plan_json(plan: BuildPlan) -> str:
    """Format build plan as JSON."""
    return json.dumps({
        "target": plan.target,
        "total_size_mb": plan.total_size_mb,
        "install_order": plan.install_order,
        "parallelizable_groups": plan.parallelizable_groups,
        "resource_requirements": plan.resource_requirements,
        "python_packages": list(set(
            pkg for spec in plan.components for pkg in spec.python_packages
        )),
        "npm_packages": list(set(
            pkg for spec in plan.components for pkg in spec.npm_packages
        )),
    }, indent=2)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Topology-Aware Build System for Jones Framework"
    )
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze component topology")
    parser.add_argument("--plan", type=str, metavar="PROFILE",
                       help="Generate build plan (core, tda, api, full, etc.)")
    parser.add_argument("--optimize", type=str, metavar="MEMORY",
                       help="Optimize for memory constraint (e.g., 4GB)")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Exclude GPU-requiring components")
    parser.add_argument("--visualize", action="store_true",
                       help="Output DOT graph")
    parser.add_argument("--json", action="store_true",
                       help="Output as JSON")
    parser.add_argument("--list-profiles", action="store_true",
                       help="List available build profiles")

    args = parser.parse_args()

    analyzer = TopologyAnalyzer(COMPONENT_REGISTRY)
    planner = BuildPlanner(COMPONENT_REGISTRY)

    if args.list_profiles:
        print("Available Build Profiles:")
        print("-" * 40)
        for name, compartments in BUILD_PROFILES.items():
            comp_names = ", ".join(c.value for c in compartments)
            print(f"  {name:12} -> {comp_names}")
        return

    if args.analyze:
        print("Component Topology Analysis")
        print("=" * 60)
        for comp in Compartment:
            components = analyzer.get_compartment_components(comp)
            if components:
                print(f"\n{comp.value.upper()} ({len(components)} components):")
                for c in components:
                    spec = COMPONENT_REGISTRY[c]
                    deps = ", ".join(spec.dependencies) if spec.dependencies else "none"
                    gpu = " [GPU]" if spec.requires_gpu else ""
                    opt = " [optional]" if spec.optional else ""
                    print(f"  - {c}: deps=[{deps}] {spec.estimated_size_mb}MB{gpu}{opt}")
        return

    if args.visualize:
        print(analyzer.to_dot())
        return

    if args.plan:
        memory_limit = None
        if args.optimize:
            # Parse memory string like "4GB" or "2048MB"
            mem_str = args.optimize.upper()
            if mem_str.endswith("GB"):
                memory_limit = float(mem_str[:-2]) * 1024
            elif mem_str.endswith("MB"):
                memory_limit = float(mem_str[:-2])
            else:
                memory_limit = float(mem_str)

        plan = planner.plan_build(
            args.plan,
            memory_limit_mb=memory_limit,
            exclude_gpu=args.no_gpu
        )

        if args.json:
            print(format_plan_json(plan))
        else:
            print(format_plan_text(plan))
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
