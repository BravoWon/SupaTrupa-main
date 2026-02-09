"""CLI constants and user-friendly messages."""

from enum import Enum
from typing import Dict, List, NamedTuple


class InstallMode(Enum):
    """Installation modes with different feature sets."""
    SIMPLE = "simple"
    STANDARD = "standard"
    FULL = "full"


class InstallModeInfo(NamedTuple):
    """Information about an installation mode."""
    name: str
    description: str
    size: str
    time: str
    features: List[str]


INSTALL_MODES: Dict[InstallMode, InstallModeInfo] = {
    InstallMode.SIMPLE: InstallModeInfo(
        name="Simple Mode",
        description="Just the basics - easy to use, smaller download",
        size="~100 MB",
        time="~2 minutes",
        features=["Core framework", "Basic API"],
    ),
    InstallMode.STANDARD: InstallModeInfo(
        name="Standard Mode (Recommended)",
        description="Backend API + Web interface",
        size="~300 MB",
        time="~5 minutes",
        features=["Core framework", "Full API", "Web interface", "TDA Pipeline"],
    ),
    InstallMode.FULL: InstallModeInfo(
        name="Full Mode",
        description="Everything including AI/ML components",
        size="~3 GB",
        time="~15 minutes",
        features=["Core framework", "Full API", "Web interface", "TDA Pipeline", "ML models", "All adapters"],
    ),
}


# Default ports
DEFAULT_BACKEND_PORT = 8000
DEFAULT_FRONTEND_PORT = 5173

# Service names
SERVICE_BACKEND = "backend"
SERVICE_FRONTEND = "frontend"

# Required Python version
MIN_PYTHON_VERSION = (3, 10)

# Required disk space in bytes
MIN_DISK_SPACE = {
    InstallMode.SIMPLE: 100 * 1024 * 1024,      # 100 MB
    InstallMode.STANDARD: 300 * 1024 * 1024,    # 300 MB
    InstallMode.FULL: 3 * 1024 * 1024 * 1024,   # 3 GB
}


# User-friendly error messages
ERROR_MESSAGES = {
    "python_version": (
        "Python {found} is installed, but version {required}+ is needed.\n"
        "\n"
        "To fix:\n"
        "  1. Download Python from https://python.org/downloads\n"
        "  2. Install it (make sure to check 'Add to PATH')\n"
        "  3. Restart your terminal\n"
        "  4. Try again"
    ),
    "node_missing": (
        "Node.js is not installed on your computer.\n"
        "\n"
        "To fix:\n"
        "  1. Download Node.js from https://nodejs.org\n"
        "  2. Choose the LTS (Long Term Support) version\n"
        "  3. Run the installer\n"
        "  4. Restart your terminal\n"
        "  5. Try again"
    ),
    "pnpm_missing": (
        "pnpm (a package manager) is not installed.\n"
        "\n"
        "To fix, run this command:\n"
        "  npm install -g pnpm\n"
        "\n"
        "Then try again."
    ),
    "port_in_use": (
        "Port {port} is already being used by another program.\n"
        "\n"
        "To fix:\n"
        "  1. Run: jones doctor --fix\n"
        "  - OR -\n"
        "  2. Close the program using port {port}\n"
        "  3. Try again"
    ),
    "disk_space": (
        "Not enough disk space. Need {required}, but only {available} available.\n"
        "\n"
        "To fix:\n"
        "  1. Free up some disk space\n"
        "  2. Or choose a smaller installation mode"
    ),
    "venv_missing": (
        "The virtual environment is not set up.\n"
        "\n"
        "To fix, run:\n"
        "  jones install\n"
        "\n"
        "This will set up everything automatically."
    ),
    "backend_not_running": (
        "The backend server is not running.\n"
        "\n"
        "To start it, run:\n"
        "  jones start"
    ),
    "frontend_not_running": (
        "The web interface is not running.\n"
        "\n"
        "To start it, run:\n"
        "  jones start"
    ),
}


# Menu options for interactive mode
MAIN_MENU_OPTIONS = [
    ("1", "Quick Start", "Get everything running with defaults"),
    ("2", "Install", "Set up the framework on your computer"),
    ("3", "Start", "Launch the framework services"),
    ("4", "Status", "Check what's running"),
    ("5", "Stop", "Shut down services"),
    ("6", "Doctor", "Diagnose problems"),
    ("7", "Help", "Learn more"),
    ("8", "Exit", "Close this program"),
]
