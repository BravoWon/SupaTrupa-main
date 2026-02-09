#!/usr/bin/env python3
"""
Fix obfuscated component exports.
Renames exported functions to match their expected names based on filename.
"""

import re
from pathlib import Path

COMPONENTS_DIR = Path(__file__).parent.parent / "frontend" / "client" / "src" / "components"
PAGES_DIR = Path(__file__).parent.parent / "frontend" / "client" / "src" / "pages"
HOOKS_DIR = Path(__file__).parent.parent / "frontend" / "client" / "src" / "hooks"

def get_expected_name(filepath: Path) -> str:
    """Get expected export name from filename."""
    name = filepath.stem
    # Convert to PascalCase if needed
    if '-' in name or '_' in name:
        parts = re.split(r'[-_]', name)
        name = ''.join(p.capitalize() for p in parts)
    return name

def fix_component(filepath: Path) -> bool:
    """Fix a single component file."""
    content = filepath.read_text()
    original = content
    expected_name = get_expected_name(filepath)

    # Pattern 1: export function _Txxx (named export)
    match = re.search(r'export\s+function\s+(_T[A-Za-z0-9_]+)\s*\(', content)
    if match:
        obfuscated = match.group(1)
        if obfuscated != expected_name:
            content = re.sub(rf'\b{re.escape(obfuscated)}\b', expected_name, content)
            if content != original:
                filepath.write_text(content)
                print(f"  Fixed: {filepath.name}: {obfuscated} -> {expected_name}")
                return True

    # Pattern 2: export default function _Txxx
    match = re.search(r'export\s+default\s+function\s+(_T[A-Za-z0-9_]+)\s*\(', content)
    if match:
        obfuscated = match.group(1)
        if obfuscated != expected_name:
            content = re.sub(rf'\b{re.escape(obfuscated)}\b', expected_name, content)
            if content != original:
                filepath.write_text(content)
                print(f"  Fixed: {filepath.name}: {obfuscated} -> {expected_name}")
                return True

    return False

def main():
    fixed = 0

    print(f"Scanning {COMPONENTS_DIR}")
    for filepath in COMPONENTS_DIR.glob("*.tsx"):
        # Skip ui directory (already fixed)
        if fix_component(filepath):
            fixed += 1

    print(f"\nScanning {PAGES_DIR}")
    for filepath in PAGES_DIR.glob("*.tsx"):
        if fix_component(filepath):
            fixed += 1

    print(f"\nScanning {HOOKS_DIR}")
    for filepath in HOOKS_DIR.glob("*.ts"):
        if fix_component(filepath):
            fixed += 1

    print(f"\nFixed {fixed} files")

if __name__ == "__main__":
    main()
