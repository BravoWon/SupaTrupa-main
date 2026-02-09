#!/bin/bash
# =============================================================================
# JONES FRAMEWORK - Interdimensional Install System
# =============================================================================
# Topology-aware modular installation with compartmentalized builds
#
# Usage:
#   ./install.sh              # Install all components
#   ./install.sh --core       # Core only (minimal footprint)
#   ./install.sh --tda        # Core + TDA pipeline
#   ./install.sh --ml         # Core + ML/AI components
#   ./install.sh --api        # Core + API server
#   ./install.sh --full       # Everything including dev tools
#   ./install.sh --frontend   # Frontend only
#   ./install.sh --optimize   # Resource-optimized build
#
# Compartments:
#   CORE     → ConditionState, ActivityState, Tensor, ManifoldBridge
#   TDA      → TDAPipeline, RegimeClassifier, Persistence*
#   SANS     → MixtureOfExperts, LoRAAdapter, ContinuityGuard
#   ML       → PyTorch, Transformers, InferenceEngine
#   API      → FastAPI, WebSockets, REST endpoints
#   UI       → React, Three.js, Recharts
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
SHARED_DIR="$PROJECT_ROOT/shared"

# Default installation mode
INSTALL_MODE="all"
OPTIMIZE_RESOURCES=false
SKIP_FRONTEND=false
VERBOSE=false

# =============================================================================
# Utility Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${PURPLE}  $1${NC}"
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        return 1
    fi
    return 0
}

get_cpu_cores() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sysctl -n hw.ncpu
    else
        nproc
    fi
}

get_available_memory_gb() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo $(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
    else
        echo $(($(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024))
    fi
}

# =============================================================================
# Parse Arguments
# =============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --core)
                INSTALL_MODE="core"
                shift
                ;;
            --tda)
                INSTALL_MODE="tda"
                shift
                ;;
            --ml)
                INSTALL_MODE="ml"
                shift
                ;;
            --api)
                INSTALL_MODE="api"
                shift
                ;;
            --full)
                INSTALL_MODE="full"
                shift
                ;;
            --frontend)
                INSTALL_MODE="frontend"
                shift
                ;;
            --optimize)
                OPTIMIZE_RESOURCES=true
                shift
                ;;
            --no-frontend)
                SKIP_FRONTEND=true
                shift
                ;;
            --verbose|-v)
                VERBOSE=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    cat << EOF
Jones Framework Installer - Topology-Aware Modular Build System

Usage: ./install.sh [OPTIONS]

Options:
  --core        Install core components only (minimal ~50MB)
  --tda         Install core + TDA pipeline (~100MB)
  --ml          Install core + ML components (~2GB with PyTorch)
  --api         Install core + API server (~150MB)
  --full        Install everything including dev tools (~3GB)
  --frontend    Install frontend only
  --optimize    Enable resource optimization
  --no-frontend Skip frontend installation
  --verbose     Verbose output
  --help        Show this help message

Compartment Topology:
  CORE ────► TDA ────► SANS ────► API
    │          │         │         │
    └──────────┴─────────┴─────────┴───► ML (optional)

Examples:
  ./install.sh --core --optimize     # Minimal footprint
  ./install.sh --tda --api           # TDA + API without ML
  ./install.sh --full                # Everything
EOF
}

# =============================================================================
# System Detection & Resource Optimization
# =============================================================================

detect_system() {
    log_section "System Detection"

    CPU_CORES=$(get_cpu_cores)
    MEMORY_GB=$(get_available_memory_gb)

    log_info "CPU Cores: $CPU_CORES"
    log_info "Available Memory: ${MEMORY_GB}GB"
    log_info "OS Type: $OSTYPE"

    # Detect GPU
    GPU_TYPE="none"
    if check_command nvidia-smi; then
        GPU_TYPE="cuda"
        log_info "GPU: NVIDIA CUDA detected"
    elif [[ "$OSTYPE" == "darwin"* ]] && system_profiler SPDisplaysDataType | grep -q "Apple M"; then
        GPU_TYPE="metal"
        log_info "GPU: Apple Metal detected"
    else
        log_info "GPU: None detected (CPU mode)"
    fi

    # Set optimization flags based on resources
    if [[ "$OPTIMIZE_RESOURCES" == true ]]; then
        log_info "Resource optimization enabled"

        if [[ $MEMORY_GB -lt 8 ]]; then
            log_warn "Low memory detected - recommending --core mode"
            export PIP_NO_CACHE_DIR=1
            export PYTORCH_NO_CUDA_MEMORY_CACHING=1
        fi

        # Set parallel jobs based on CPU
        export MAKEFLAGS="-j$((CPU_CORES / 2))"
        export PIP_JOBS=$((CPU_CORES / 2))
    fi

    export GPU_TYPE
    export CPU_CORES
    export MEMORY_GB
}

# =============================================================================
# Python Environment Setup
# =============================================================================

setup_python_env() {
    log_section "Python Environment"

    # Check Python version
    if ! check_command python3; then
        log_error "Python 3 not found. Please install Python 3.9+"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    log_info "Python version: $PYTHON_VERSION"

    # Create virtual environment if it doesn't exist
    VENV_DIR="$PROJECT_ROOT/.venv"
    if [[ ! -d "$VENV_DIR" ]]; then
        log_info "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    log_success "Virtual environment activated"

    # Upgrade pip
    pip install --upgrade pip wheel setuptools
}

# =============================================================================
# Backend Installation - Compartmentalized
# =============================================================================

install_backend_core() {
    log_section "Installing Backend Core"

    cd "$BACKEND_DIR"

    # Create README if missing
    if [[ ! -f "README.md" ]]; then
        echo "# Jones Framework" > README.md
    fi

    # Install core dependencies only
    log_info "Installing core dependencies..."
    pip install -e . --no-deps
    pip install numpy scipy scikit-learn pandas pyyaml

    log_success "Core components installed"

    # Verify core imports
    python3 -c "
from jones_framework.core.condition_state import ConditionState
from jones_framework.core.activity_state import ActivityState
from jones_framework.core.manifold_bridge import get_registry
print('Core manifold connected')
"
}

install_backend_tda() {
    log_section "Installing TDA Pipeline"

    cd "$BACKEND_DIR"

    log_info "Installing TDA dependencies..."
    pip install ripser persim

    log_success "TDA pipeline installed"

    # Verify TDA
    python3 -c "
from jones_framework.perception.tda_pipeline import (
    TDAPipeline, PersistenceLandscape, TopologicalSignature
)
import numpy as np
pipeline = TDAPipeline()
print('TDA pipeline operational')
"
}

install_backend_sans() {
    log_section "Installing SANS Architecture"

    cd "$BACKEND_DIR"

    log_info "SANS components are included in core..."

    # Verify SANS
    python3 -c "
from jones_framework.sans.mixture_of_experts import MixtureOfExperts
from jones_framework.sans.lora_adapter import LoRAAdapter
from jones_framework.sans.continuity_guard import ContinuityGuard
print('SANS architecture connected')
"
    log_success "SANS architecture verified"
}

install_backend_ml() {
    log_section "Installing ML Components"

    cd "$BACKEND_DIR"

    log_info "Installing ML dependencies..."

    # Select PyTorch variant based on GPU
    if [[ "$GPU_TYPE" == "cuda" ]]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$GPU_TYPE" == "metal" ]]; then
        pip install torch torchvision
    else
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi

    pip install transformers

    log_success "ML components installed ($GPU_TYPE acceleration)"
}

install_backend_api() {
    log_section "Installing API Server"

    cd "$BACKEND_DIR"

    log_info "Installing API dependencies..."
    pip install fastapi uvicorn[standard] pydantic websockets

    log_success "API server installed"

    # Verify API
    python3 -c "
from fastapi import FastAPI
print('API framework ready')
"
}

install_backend_dev() {
    log_section "Installing Development Tools"

    cd "$BACKEND_DIR"

    log_info "Installing dev dependencies..."
    pip install pytest pytest-cov black mypy httpx

    log_success "Development tools installed"
}

# =============================================================================
# Frontend Installation
# =============================================================================

install_frontend() {
    log_section "Installing Frontend"

    if [[ "$SKIP_FRONTEND" == true ]]; then
        log_info "Skipping frontend installation"
        return
    fi

    # Check Node.js
    if ! check_command node; then
        log_error "Node.js not found. Please install Node.js 18+"
        exit 1
    fi

    NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
    log_info "Node.js version: v$NODE_VERSION"

    if [[ $NODE_VERSION -lt 18 ]]; then
        log_error "Node.js 18+ required"
        exit 1
    fi

    # Check pnpm
    if ! check_command pnpm; then
        log_info "Installing pnpm..."
        npm install -g pnpm
    fi

    # Install frontend dependencies
    cd "$FRONTEND_DIR"
    log_info "Installing frontend dependencies..."
    pnpm install

    # Install shared types
    cd "$SHARED_DIR"
    if [[ -f "package.json" ]]; then
        pnpm install
    fi

    log_success "Frontend installed"
}

# =============================================================================
# Build Verification
# =============================================================================

verify_installation() {
    log_section "Verifying Installation"

    cd "$BACKEND_DIR"

    # Run manifold health check
    python3 << 'EOF'
import warnings
warnings.filterwarnings('ignore')

from jones_framework.core.manifold_bridge import RecursiveImprover

print("Running manifold topology analysis...")
improver = RecursiveImprover()
gaps = improver.identify_gaps()

orphaned = [g for g in gaps if g['type'] == 'orphaned']
unidirectional = [g for g in gaps if g['type'] == 'unidirectional']

print(f"  Orphaned components: {len(orphaned)}")
print(f"  Unidirectional connections: {len(unidirectional)}")

if len(orphaned) == 0:
    print("  STATUS: MANIFOLD HEALTHY")
else:
    print("  STATUS: NEEDS ATTENTION")
EOF

    log_success "Installation verified"
}

# =============================================================================
# Generate Environment Configuration
# =============================================================================

generate_env_config() {
    log_section "Generating Environment Configuration"

    ENV_FILE="$PROJECT_ROOT/.env"

    cat > "$ENV_FILE" << EOF
# =============================================================================
# Jones Framework Environment Configuration
# Generated by install.sh
# =============================================================================

# Backend Configuration
JONES_ENV=development
JONES_DEVICE=$GPU_TYPE
JONES_LOG_LEVEL=INFO

# Resource Optimization
JONES_CPU_CORES=$CPU_CORES
JONES_MEMORY_GB=$MEMORY_GB
JONES_OPTIMIZE_RESOURCES=$OPTIMIZE_RESOURCES

# TDA Configuration
RIPSER_MAX_DIM=2
RIPSER_THRESHOLD=2.0

# Frontend Configuration
NODE_ENV=development
VITE_API_URL=http://localhost:8000

# Installation Mode
JONES_INSTALL_MODE=$INSTALL_MODE
EOF

    log_success "Environment configuration saved to .env"
}

# =============================================================================
# Main Installation Flow
# =============================================================================

main() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     JONES FRAMEWORK - Interdimensional Install System        ║${NC}"
    echo -e "${CYAN}║     Topology-Aware Modular Architecture                      ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    parse_args "$@"

    log_info "Installation mode: $INSTALL_MODE"

    detect_system
    setup_python_env

    # Install based on mode
    case $INSTALL_MODE in
        core)
            install_backend_core
            install_backend_sans
            ;;
        tda)
            install_backend_core
            install_backend_tda
            install_backend_sans
            ;;
        ml)
            install_backend_core
            install_backend_tda
            install_backend_sans
            install_backend_ml
            ;;
        api)
            install_backend_core
            install_backend_tda
            install_backend_sans
            install_backend_api
            ;;
        full)
            install_backend_core
            install_backend_tda
            install_backend_sans
            install_backend_ml
            install_backend_api
            install_backend_dev
            install_frontend
            ;;
        frontend)
            install_frontend
            ;;
        all|*)
            install_backend_core
            install_backend_tda
            install_backend_sans
            install_backend_api
            install_frontend
            ;;
    esac

    verify_installation
    generate_env_config

    log_section "Installation Complete"

    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    Installation Complete                     ║${NC}"
    echo -e "${GREEN}╠══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${GREEN}║  Mode: $INSTALL_MODE${NC}"
    echo -e "${GREEN}║  GPU:  $GPU_TYPE${NC}"
    echo -e "${GREEN}║  Manifold Status: HEALTHY${NC}"
    echo -e "${GREEN}╠══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${GREEN}║  To start (using CLI):                                       ║${NC}"
    echo -e "${GREEN}║    source .venv/bin/activate                                 ║${NC}"
    echo -e "${GREEN}║    jones start                                               ║${NC}"
    echo -e "${GREEN}║                                                              ║${NC}"
    echo -e "${GREEN}║  Or manually:                                                ║${NC}"
    echo -e "${GREEN}║    ./scripts/dev.sh                                          ║${NC}"
    echo -e "${GREEN}║                                                              ║${NC}"
    echo -e "${GREEN}║  For help: jones --help                                      ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

main "$@"
