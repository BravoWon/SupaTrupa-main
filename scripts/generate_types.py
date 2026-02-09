#!/usr/bin/env python3
"""
Type Generation Script: Python → TypeScript

Generates TypeScript type definitions from Python dataclasses in the jones_framework.
This ensures type safety and consistency between frontend and backend.

Usage:
    python scripts/generate_types.py [--output shared/types/generated.ts]
"""

import ast
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
import argparse


@dataclass
class TypeMapping:
    """Mapping from Python types to TypeScript types."""
    python_type: str
    typescript_type: str
    is_optional: bool = False
    is_array: bool = False


# Standard Python → TypeScript type mappings
TYPE_MAPPINGS: Dict[str, str] = {
    'str': 'string',
    'int': 'number',
    'float': 'number',
    'bool': 'boolean',
    'None': 'null',
    'NoneType': 'null',
    'Any': 'unknown',
    'dict': 'Record<string, unknown>',
    'Dict': 'Record<string, unknown>',
    'list': 'unknown[]',
    'List': 'unknown[]',
    'tuple': 'unknown[]',
    'Tuple': 'unknown[]',
    'set': 'Set<unknown>',
    'Set': 'Set<unknown>',
    'bytes': 'Uint8Array',
    'datetime': 'string',  # ISO format
    'date': 'string',
    'time': 'string',
    'timedelta': 'number',  # seconds
    'Decimal': 'string',
    'UUID': 'string',
    'np.ndarray': 'number[]',
    'ndarray': 'number[]',
    'Tensor': 'number[] | number[][]',
}


@dataclass
class ExtractedEnum:
    """Extracted enum definition."""
    name: str
    members: List[Tuple[str, Any]]
    docstring: Optional[str] = None


@dataclass
class ExtractedField:
    """Extracted field from a dataclass."""
    name: str
    type_annotation: str
    default: Optional[str] = None
    is_optional: bool = False
    docstring: Optional[str] = None


@dataclass
class ExtractedDataclass:
    """Extracted dataclass definition."""
    name: str
    fields: List[ExtractedField]
    docstring: Optional[str] = None
    base_classes: List[str] = field(default_factory=list)


class PythonTypeExtractor(ast.NodeVisitor):
    """AST visitor to extract type definitions from Python source."""

    def __init__(self):
        self.enums: List[ExtractedEnum] = []
        self.dataclasses: List[ExtractedDataclass] = []
        self._current_docstring: Optional[str] = None

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition to extract enums and dataclasses."""
        # Check for @dataclass decorator
        is_dataclass = any(
            (isinstance(d, ast.Name) and d.id == 'dataclass') or
            (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'dataclass')
            for d in node.decorator_list
        )

        # Check if it's an Enum
        base_names = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(base.attr)

        is_enum = 'Enum' in base_names or any(b.endswith('Enum') for b in base_names)

        # Skip obfuscated names
        if node.name.startswith('_c') or node.name.startswith('_f'):
            return

        # Get docstring
        docstring = ast.get_docstring(node)

        if is_enum:
            self._extract_enum(node, docstring)
        elif is_dataclass:
            self._extract_dataclass(node, docstring, base_names)

        self.generic_visit(node)

    def _extract_enum(self, node: ast.ClassDef, docstring: Optional[str]):
        """Extract enum members."""
        members = []
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        # Get the value
                        if isinstance(item.value, ast.Constant):
                            members.append((target.id, item.value.value))
                        elif isinstance(item.value, ast.Call):
                            # auto() or similar
                            members.append((target.id, target.id.lower()))

        if members:
            self.enums.append(ExtractedEnum(
                name=node.name,
                members=members,
                docstring=docstring
            ))

    def _extract_dataclass(self, node: ast.ClassDef, docstring: Optional[str], base_names: List[str]):
        """Extract dataclass fields."""
        fields = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                field_name = item.target.id
                type_str = self._annotation_to_string(item.annotation)

                # Check for default value
                default = None
                is_optional = 'Optional' in type_str or type_str.endswith('| None')

                if item.value is not None:
                    if isinstance(item.value, ast.Constant):
                        default = repr(item.value.value)
                    elif isinstance(item.value, ast.Call):
                        # field(default_factory=...) or similar
                        if isinstance(item.value.func, ast.Name) and item.value.func.id == 'field':
                            default = 'field(...)'
                        else:
                            default = 'callable'

                fields.append(ExtractedField(
                    name=field_name,
                    type_annotation=type_str,
                    default=default,
                    is_optional=is_optional or default is not None
                ))

        if fields:
            self.dataclasses.append(ExtractedDataclass(
                name=node.name,
                fields=fields,
                docstring=docstring,
                base_classes=base_names
            ))

    def _annotation_to_string(self, annotation) -> str:
        """Convert AST annotation to string representation."""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        elif isinstance(annotation, ast.Subscript):
            base = self._annotation_to_string(annotation.value)
            if isinstance(annotation.slice, ast.Tuple):
                args = ', '.join(self._annotation_to_string(e) for e in annotation.slice.elts)
            else:
                args = self._annotation_to_string(annotation.slice)
            return f'{base}[{args}]'
        elif isinstance(annotation, ast.Attribute):
            return f'{self._annotation_to_string(annotation.value)}.{annotation.attr}'
        elif isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            # Union types with | syntax
            left = self._annotation_to_string(annotation.left)
            right = self._annotation_to_string(annotation.right)
            return f'{left} | {right}'
        elif isinstance(annotation, ast.Tuple):
            return ', '.join(self._annotation_to_string(e) for e in annotation.elts)
        else:
            return 'unknown'


class TypeScriptGenerator:
    """Generates TypeScript types from extracted Python definitions."""

    def __init__(self):
        self.generated_types: Set[str] = set()

    def convert_type(self, python_type: str) -> str:
        """Convert a Python type annotation to TypeScript."""
        # Handle Optional
        if python_type.startswith('Optional[') and python_type.endswith(']'):
            inner = python_type[9:-1]
            return f'{self.convert_type(inner)} | null'

        # Handle Union (both Union[...] and | syntax)
        if python_type.startswith('Union[') and python_type.endswith(']'):
            inner = python_type[6:-1]
            parts = self._split_union_parts(inner)
            return ' | '.join(self.convert_type(p.strip()) for p in parts)

        if ' | ' in python_type:
            parts = python_type.split(' | ')
            return ' | '.join(self.convert_type(p.strip()) for p in parts)

        # Handle List
        if python_type.startswith('List[') and python_type.endswith(']'):
            inner = python_type[5:-1]
            return f'{self.convert_type(inner)}[]'

        # Handle Dict
        if python_type.startswith('Dict[') and python_type.endswith(']'):
            inner = python_type[5:-1]
            parts = self._split_dict_parts(inner)
            if len(parts) == 2:
                key_type = self.convert_type(parts[0].strip())
                val_type = self.convert_type(parts[1].strip())
                return f'Record<{key_type}, {val_type}>'
            return 'Record<string, unknown>'

        # Handle Tuple
        if python_type.startswith('Tuple[') and python_type.endswith(']'):
            inner = python_type[6:-1]
            parts = self._split_tuple_parts(inner)
            ts_types = [self.convert_type(p.strip()) for p in parts]
            return f'[{", ".join(ts_types)}]'

        # Handle Set
        if python_type.startswith('Set[') and python_type.endswith(']'):
            inner = python_type[4:-1]
            return f'Set<{self.convert_type(inner)}>'

        # Handle Callable (simplified)
        if python_type.startswith('Callable'):
            return '(...args: unknown[]) => unknown'

        # Direct mapping
        if python_type in TYPE_MAPPINGS:
            return TYPE_MAPPINGS[python_type]

        # Assume it's a reference to another type
        return python_type

    def _split_union_parts(self, inner: str) -> List[str]:
        """Split union parts accounting for nested generics."""
        parts = []
        depth = 0
        current = []
        for char in inner:
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
            elif char == ',' and depth == 0:
                parts.append(''.join(current).strip())
                current = []
                continue
            current.append(char)
        if current:
            parts.append(''.join(current).strip())
        return parts

    def _split_dict_parts(self, inner: str) -> List[str]:
        """Split Dict[K, V] parts."""
        return self._split_union_parts(inner)

    def _split_tuple_parts(self, inner: str) -> List[str]:
        """Split Tuple parts."""
        return self._split_union_parts(inner)

    def generate_enum(self, enum: ExtractedEnum) -> str:
        """Generate TypeScript type for an enum."""
        lines = []
        if enum.docstring:
            lines.append(f'/**\n * {enum.docstring}\n */')

        # Generate as union type (more flexible in TS)
        values = []
        for name, value in enum.members:
            if isinstance(value, str):
                values.append(f"'{value}'")
            else:
                values.append(f"'{name}'")

        lines.append(f"export type {enum.name} = {' | '.join(values)};")
        return '\n'.join(lines)

    def generate_interface(self, dc: ExtractedDataclass) -> str:
        """Generate TypeScript interface for a dataclass."""
        lines = []
        if dc.docstring:
            doc_lines = dc.docstring.split('\n')
            lines.append('/**')
            for dl in doc_lines:
                lines.append(f' * {dl}')
            lines.append(' */')

        extends = ''
        if dc.base_classes and dc.base_classes[0] not in ('ABC', 'Generic'):
            valid_bases = [b for b in dc.base_classes if not b.startswith('_')]
            if valid_bases:
                extends = f' extends {", ".join(valid_bases)}'

        lines.append(f'export interface {dc.name}{extends} {{')

        for fld in dc.fields:
            ts_type = self.convert_type(fld.type_annotation)
            optional = '?' if fld.is_optional else ''
            lines.append(f'  {fld.name}{optional}: {ts_type};')

        lines.append('}')
        return '\n'.join(lines)

    def generate(self, enums: List[ExtractedEnum], dataclasses: List[ExtractedDataclass]) -> str:
        """Generate complete TypeScript file."""
        lines = [
            '/**',
            ' * Auto-generated TypeScript types from Python jones_framework',
            ' * ',
            ' * DO NOT EDIT MANUALLY - Run `python scripts/generate_types.py` to regenerate',
            f' * Generated from {len(dataclasses)} dataclasses and {len(enums)} enums',
            ' */',
            '',
        ]

        # Generate enums first
        if enums:
            lines.append('// =============================================================================')
            lines.append('// Enums')
            lines.append('// =============================================================================')
            lines.append('')
            for enum in enums:
                lines.append(self.generate_enum(enum))
                lines.append('')

        # Generate interfaces
        if dataclasses:
            lines.append('// =============================================================================')
            lines.append('// Interfaces')
            lines.append('// =============================================================================')
            lines.append('')
            for dc in dataclasses:
                lines.append(self.generate_interface(dc))
                lines.append('')

        return '\n'.join(lines)


def extract_from_file(filepath: Path) -> Tuple[List[ExtractedEnum], List[ExtractedDataclass]]:
    """Extract types from a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source)
        extractor = PythonTypeExtractor()
        extractor.visit(tree)
        return extractor.enums, extractor.dataclasses
    except Exception as e:
        print(f"Warning: Failed to parse {filepath}: {e}", file=sys.stderr)
        return [], []


def find_python_files(root: Path, patterns: List[str] = None) -> List[Path]:
    """Find Python files to process."""
    if patterns is None:
        patterns = [
            'core/condition_state.py',
            'core/activity_state.py',
            'core/shadow_tensor.py',
            'perception/tda_pipeline.py',
            'perception/regime_classifier.py',
            'sans/mixture_of_experts.py',
            'sans/lora_adapter.py',
        ]

    files = []
    for pattern in patterns:
        full_path = root / pattern
        if full_path.exists():
            files.append(full_path)
    return files


def main():
    parser = argparse.ArgumentParser(description='Generate TypeScript types from Python')
    parser.add_argument('--output', '-o', default='shared/types/generated.ts',
                        help='Output TypeScript file')
    parser.add_argument('--backend', '-b', default='backend/jones_framework',
                        help='Backend source directory')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    backend_dir = project_root / args.backend

    if not backend_dir.exists():
        print(f"Error: Backend directory not found: {backend_dir}", file=sys.stderr)
        sys.exit(1)

    # Extract types from files
    all_enums: List[ExtractedEnum] = []
    all_dataclasses: List[ExtractedDataclass] = []

    files = find_python_files(backend_dir)

    if args.verbose:
        print(f"Processing {len(files)} files...")

    for filepath in files:
        if args.verbose:
            print(f"  Extracting from {filepath.name}...")
        enums, dataclasses = extract_from_file(filepath)
        all_enums.extend(enums)
        all_dataclasses.extend(dataclasses)

    if args.verbose:
        print(f"Found {len(all_enums)} enums and {len(all_dataclasses)} dataclasses")

    # Generate TypeScript
    generator = TypeScriptGenerator()
    output = generator.generate(all_enums, all_dataclasses)

    # Write output
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)

    print(f"Generated {output_path}")
    print(f"  - {len(all_enums)} enums")
    print(f"  - {len(all_dataclasses)} interfaces")


if __name__ == '__main__':
    main()
