"""
Jones Framework CLI Package

Unified command-line interface for the Jones Framework with:
- Interactive menu system for all skill levels
- Installation wizard with guided setup
- Service management (start/stop/status)
- Troubleshooting diagnostics (doctor command)
- Cross-platform support (Windows, macOS, Linux)
"""

from jones_framework.cli.main import app, main

__all__ = ["app", "main"]
