#!/usr/bin/env python3
"""
Fix obfuscated function/variable names in TypeScript/TSX files.
Handles multiple patterns of obfuscation.
"""

import re
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent / "frontend" / "client" / "src"

def fix_file(filepath: Path) -> int:
    """Fix a single file by matching obfuscated names to their usage context."""
    content = filepath.read_text()
    original = content
    replacements = 0

    # Pattern 1: Function exports - match obfuscated function names to exports
    export_match = re.search(r'export\s*\{([^}]+)\}', content)
    if export_match:
        export_names = [name.strip().split(' as ')[-1].strip() for name in export_match.group(1).split(',')]
        export_names = [name for name in export_names if name]

        # Find function definitions
        func_pattern = r'(function|const)\s+(_T[A-Za-z0-9_]+)\s*[=(]'
        funcs = list(re.finditer(func_pattern, content))

        for i, func_match in enumerate(funcs):
            if i < len(export_names):
                obfuscated = func_match.group(2)
                clean_name = export_names[i]
                if obfuscated != clean_name and obfuscated.startswith('_T'):
                    content = re.sub(rf'\b{re.escape(obfuscated)}\b', clean_name, content)
                    replacements += 1

    # Pattern 2: Default exports - match function name to 'export default'
    default_match = re.search(r'export\s+default\s+(\w+)', content)
    if default_match:
        exported_name = default_match.group(1)
        # Find function that should be the default export
        func_match = re.search(rf'(function|const)\s+(_T[A-Za-z0-9_]+).*?(?=export\s+default)', content, re.DOTALL)
        if func_match:
            obfuscated = func_match.group(2)
            if obfuscated != exported_name and obfuscated.startswith('_T'):
                content = re.sub(rf'\b{re.escape(obfuscated)}\b', exported_name, content)
                replacements += 1

    # Pattern 3: Interface/type names - match to their usage
    # Find interfaces with obfuscated names
    interface_matches = re.findall(r'interface\s+(_T[A-Za-z0-9_]+)\s*\{', content)
    for obfuscated in interface_matches:
        # Try to infer the real name from Props/State pattern
        if 'Props' in content and re.search(rf'{obfuscated}.*children|{obfuscated}.*React', content):
            content = re.sub(rf'\b{re.escape(obfuscated)}\b', 'Props', content)
            replacements += 1
        elif 'State' in content and re.search(rf'{obfuscated}.*hasError|{obfuscated}.*error', content):
            content = re.sub(rf'\b{re.escape(obfuscated)}\b', 'State', content)
            replacements += 1

    # Pattern 4: Variable assignments with obfuscated names used immediately after
    # e.g., const _vABC = value; then uses 'realName' in next line
    var_pattern = r'const\s+(_v[A-Za-z0-9_]+)\s*=\s*([^;]+);'
    for match in re.finditer(var_pattern, content):
        obfuscated = match.group(1)
        # Check if there's a clean variable name used right after that references this
        # This is complex - skip for now

    if content != original:
        filepath.write_text(content)
        return replacements
    return 0

def main():
    print(f"Scanning {BASE_DIR}")
    fixed = 0
    total_replacements = 0

    # Process all tsx and ts files
    for filepath in BASE_DIR.rglob("*.tsx"):
        reps = fix_file(filepath)
        if reps:
            print(f"  Fixed: {filepath.relative_to(BASE_DIR)} ({reps} replacements)")
            fixed += 1
            total_replacements += reps

    for filepath in BASE_DIR.rglob("*.ts"):
        if filepath.suffix == '.ts' and not str(filepath).endswith('.d.ts'):
            reps = fix_file(filepath)
            if reps:
                print(f"  Fixed: {filepath.relative_to(BASE_DIR)} ({reps} replacements)")
                fixed += 1
                total_replacements += reps

    print(f"\nFixed {fixed} files with {total_replacements} total replacements")

if __name__ == "__main__":
    main()
