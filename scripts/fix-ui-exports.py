#!/usr/bin/env python3
"""
Fix obfuscated function names in UI components.
Matches function definitions with their exports.
"""

import re
import os
from pathlib import Path

UI_DIR = Path(__file__).parent.parent / "frontend" / "client" / "src" / "components" / "ui"

def fix_file(filepath: Path) -> bool:
    """Fix a single file by matching obfuscated names to exports."""
    content = filepath.read_text()
    original = content

    # Find all exports at the end of the file
    export_match = re.search(r'export\s*\{([^}]+)\}', content)
    if not export_match:
        return False

    export_names = [name.strip() for name in export_match.group(1).split(',')]
    export_names = [name for name in export_names if name]

    # Find all function definitions with obfuscated names
    func_pattern = r'(function|const)\s+(_T[A-Za-z0-9]+)\s*[=(]'
    funcs = list(re.finditer(func_pattern, content))

    if len(funcs) != len(export_names):
        print(f"  Warning: {filepath.name} has {len(funcs)} functions but {len(export_names)} exports")
        # Try to match by order anyway

    # Build replacement map
    replacements = {}
    for i, func_match in enumerate(funcs):
        if i < len(export_names):
            obfuscated = func_match.group(2)
            clean_name = export_names[i]
            if obfuscated != clean_name:
                replacements[obfuscated] = clean_name

    # Apply replacements (whole word only)
    for old, new in replacements.items():
        content = re.sub(rf'\b{re.escape(old)}\b', new, content)

    if content != original:
        filepath.write_text(content)
        print(f"  Fixed: {filepath.name} ({len(replacements)} replacements)")
        return True
    return False

def main():
    if not UI_DIR.exists():
        print(f"UI directory not found: {UI_DIR}")
        return

    print(f"Scanning {UI_DIR}")
    fixed = 0

    for filepath in UI_DIR.glob("*.tsx"):
        if fix_file(filepath):
            fixed += 1

    print(f"\nFixed {fixed} files")

if __name__ == "__main__":
    main()
