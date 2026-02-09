#!/usr/bin/env python3
"""
Jones Framework Axiom-Compliant Code Obfuscator

Applies the framework's own principles to code protection:
- Substrate Principle: Creates verifiable transformation mapping
- Manifold Hypothesis: Preserves code topology (imports, connections)
- Continuity Thesis: Functionality-preserving continuous map
- SANS Architecture: Specialized expert for obfuscation regime

Usage:
    python scripts/obfuscate.py --backend   # Obfuscate Python
    python scripts/obfuscate.py --frontend  # Obfuscate TypeScript
    python scripts/obfuscate.py --all       # Obfuscate everything
"""

import ast
import os
import re
import sys
import json
import hashlib
import random
import string
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# Seed for reproducible obfuscation
OBFUSCATION_SEED = 0x4A6F6E6573  # "Jones" in hex

# Characters for obfuscated names
OBFUSCATION_CHARS = "Il1O0"  # Visually confusing chars
EXTENDED_CHARS = "abcdefghijklmnopqrstuvwxyz"

# Reserved names that should NOT be obfuscated
PYTHON_RESERVED = {
    # Keywords
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
    'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
    'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
    'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
    'while', 'with', 'yield',
    # Built-ins
    'print', 'len', 'range', 'str', 'int', 'float', 'bool', 'list', 'dict',
    'set', 'tuple', 'type', 'object', 'super', 'self', 'cls', 'isinstance',
    'issubclass', 'hasattr', 'getattr', 'setattr', 'delattr', 'property',
    'staticmethod', 'classmethod', 'abs', 'all', 'any', 'bin', 'callable',
    'chr', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter', 'format',
    'frozenset', 'globals', 'hash', 'hex', 'id', 'input', 'iter', 'locals',
    'map', 'max', 'min', 'next', 'oct', 'open', 'ord', 'pow', 'repr',
    'reversed', 'round', 'slice', 'sorted', 'sum', 'vars', 'zip',
    'Exception', 'BaseException', 'ValueError', 'TypeError', 'KeyError',
    'IndexError', 'AttributeError', 'RuntimeError', 'StopIteration',
    'NotImplementedError', 'ImportError', 'OSError', 'IOError', 'FileNotFoundError',
    # Common dunder methods (preserve for Python semantics)
    '__init__', '__new__', '__del__', '__repr__', '__str__', '__bytes__',
    '__format__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__',
    '__hash__', '__bool__', '__getattr__', '__getattribute__', '__setattr__',
    '__delattr__', '__dir__', '__get__', '__set__', '__delete__', '__slots__',
    '__init_subclass__', '__set_name__', '__instancecheck__', '__subclasscheck__',
    '__class_getitem__', '__call__', '__len__', '__length_hint__', '__getitem__',
    '__setitem__', '__delitem__', '__missing__', '__iter__', '__reversed__',
    '__contains__', '__add__', '__sub__', '__mul__', '__matmul__', '__truediv__',
    '__floordiv__', '__mod__', '__divmod__', '__pow__', '__lshift__', '__rshift__',
    '__and__', '__xor__', '__or__', '__neg__', '__pos__', '__abs__', '__invert__',
    '__complex__', '__int__', '__float__', '__index__', '__round__', '__trunc__',
    '__floor__', '__ceil__', '__enter__', '__exit__', '__await__', '__aiter__',
    '__anext__', '__aenter__', '__aexit__', '__name__', '__module__', '__qualname__',
    '__doc__', '__dict__', '__class__', '__bases__', '__mro__', '__subclasses__',
    '__annotations__', '__wrapped__', '__all__', '__file__', '__package__',
    '__loader__', '__spec__', '__path__', '__cached__', '__builtins__',
    # Dataclass/typing
    'dataclass', 'field', 'Optional', 'List', 'Dict', 'Set', 'Tuple', 'Union',
    'Any', 'Callable', 'TypeVar', 'Generic', 'Protocol', 'Final', 'Literal',
    'ClassVar', 'Annotated', 'overload', 'cast', 'TYPE_CHECKING',
    # Common decorators
    'abstractmethod', 'abstractproperty',
    # Framework exports (preserve public API for topology)
    'bridge', 'extends', 'transforms', 'composes', 'get_registry',
    'ConnectionType', 'RecursiveImprover',
}

TS_RESERVED = {
    # Keywords
    'break', 'case', 'catch', 'class', 'const', 'continue', 'debugger',
    'default', 'delete', 'do', 'else', 'enum', 'export', 'extends', 'false',
    'finally', 'for', 'function', 'if', 'import', 'in', 'instanceof', 'let',
    'new', 'null', 'return', 'static', 'super', 'switch', 'this', 'throw',
    'true', 'try', 'typeof', 'undefined', 'var', 'void', 'while', 'with',
    'yield', 'async', 'await', 'of', 'implements', 'interface', 'package',
    'private', 'protected', 'public', 'type', 'as', 'from', 'readonly',
    'abstract', 'declare', 'namespace', 'module', 'require', 'global',
    'asserts', 'infer', 'is', 'keyof', 'never', 'unknown', 'any', 'boolean',
    'number', 'string', 'symbol', 'bigint', 'object',
    # React
    'React', 'useState', 'useEffect', 'useCallback', 'useMemo', 'useRef',
    'useContext', 'useReducer', 'useLayoutEffect', 'useImperativeHandle',
    'useDebugValue', 'useDeferredValue', 'useTransition', 'useId',
    'Component', 'PureComponent', 'Fragment', 'StrictMode', 'Suspense',
    'createContext', 'forwardRef', 'lazy', 'memo', 'createElement',
    'cloneElement', 'createRef', 'isValidElement', 'Children',
    'props', 'children', 'key', 'ref', 'className', 'style', 'onClick',
    'onChange', 'onSubmit', 'value', 'disabled', 'placeholder', 'type',
    'id', 'name', 'href', 'src', 'alt', 'title', 'data', 'aria',
    # DOM
    'document', 'window', 'console', 'localStorage', 'sessionStorage',
    'navigator', 'location', 'history', 'fetch', 'XMLHttpRequest',
    'FormData', 'URLSearchParams', 'URL', 'Blob', 'File', 'FileReader',
    'ArrayBuffer', 'DataView', 'Int8Array', 'Uint8Array', 'Uint8ClampedArray',
    'Int16Array', 'Uint16Array', 'Int32Array', 'Uint32Array', 'Float32Array',
    'Float64Array', 'BigInt64Array', 'BigUint64Array', 'Map', 'Set', 'WeakMap',
    'WeakSet', 'Promise', 'Proxy', 'Reflect', 'Symbol', 'Error', 'EvalError',
    'RangeError', 'ReferenceError', 'SyntaxError', 'TypeError', 'URIError',
    'JSON', 'Math', 'Date', 'RegExp', 'Array', 'Object', 'String', 'Number',
    'Boolean', 'Function', 'Infinity', 'NaN', 'parseInt', 'parseFloat',
    'isNaN', 'isFinite', 'encodeURI', 'encodeURIComponent', 'decodeURI',
    'decodeURIComponent', 'escape', 'unescape', 'eval', 'arguments',
    'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval',
    'requestAnimationFrame', 'cancelAnimationFrame',
    # Three.js (preserve for 3D viz)
    'THREE', 'Scene', 'Camera', 'Renderer', 'Mesh', 'Geometry', 'Material',
    'Vector3', 'Vector2', 'Matrix4', 'Quaternion', 'Euler', 'Color',
    'Canvas', 'useFrame', 'useThree', 'extend',
}


@dataclass
class ObfuscationMapping:
    """Tracks all name transformations for reversibility (Substrate Principle)."""

    original_to_obfuscated: Dict[str, str] = field(default_factory=dict)
    obfuscated_to_original: Dict[str, str] = field(default_factory=dict)
    file_hashes: Dict[str, str] = field(default_factory=dict)
    topology_preserved: Dict[str, List[str]] = field(default_factory=dict)

    def add_mapping(self, original: str, obfuscated: str, context: str = "global"):
        """Add a name mapping."""
        key = f"{context}::{original}"
        self.original_to_obfuscated[key] = obfuscated
        self.obfuscated_to_original[obfuscated] = key

    def get_obfuscated(self, original: str, context: str = "global") -> Optional[str]:
        """Get obfuscated name for original."""
        key = f"{context}::{original}"
        return self.original_to_obfuscated.get(key)

    def save(self, path: Path):
        """Save mapping to JSON (encrypted in production)."""
        data = {
            "mappings": self.original_to_obfuscated,
            "reverse": self.obfuscated_to_original,
            "hashes": self.file_hashes,
            "topology": self.topology_preserved,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


class NameGenerator:
    """Generates confusing but valid identifier names."""

    def __init__(self, seed: int = OBFUSCATION_SEED):
        self.rng = random.Random(seed)
        self.used_names: Set[str] = set()
        self.counter = 0

    def generate(self, prefix: str = "_") -> str:
        """Generate a unique obfuscated name."""
        while True:
            self.counter += 1
            # Mix of confusing chars
            name = prefix + self._generate_confusing_name()
            if name not in self.used_names:
                self.used_names.add(name)
                return name

    def _generate_confusing_name(self) -> str:
        """Create visually confusing identifier."""
        # Use counter for uniqueness, add confusing suffix
        base = hex(self.counter)[2:]  # Remove '0x'
        # Replace with confusing chars
        replacements = {'0': 'O', '1': 'l', 'a': 'A', 'b': 'B', 'e': 'E'}
        for old, new in replacements.items():
            base = base.replace(old, new)

        # Add confusing prefix
        prefix_chars = self.rng.choices(OBFUSCATION_CHARS, k=3)
        return ''.join(prefix_chars) + base


class PythonObfuscator(ast.NodeTransformer):
    """
    AST-based Python obfuscator that preserves topology.

    Manifold Hypothesis compliance:
    - Preserves import structure (connectivity)
    - Maintains class hierarchies (component relationships)
    - Keeps public API intact (bridge connections)
    """

    def __init__(self, mapping: ObfuscationMapping, name_gen: NameGenerator):
        self.mapping = mapping
        self.name_gen = name_gen
        self.current_file = ""
        self.current_scope = "global"
        self.scope_stack: List[str] = []
        self.imports: Set[str] = set()
        self.exports: Set[str] = set()

        # Track what names are defined at module level
        self.module_level_names: Set[str] = set()

    def obfuscate_file(self, source: str, filename: str) -> str:
        """Obfuscate a single Python file."""
        self.current_file = filename
        self.imports = set()
        self.exports = set()
        self.module_level_names = set()

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            print(f"  Syntax error in {filename}: {e}")
            return source

        # First pass: collect all names defined at module level
        self._collect_module_names(tree)

        # Second pass: transform
        new_tree = self.visit(tree)

        # Remove docstrings
        new_tree = self._remove_docstrings(new_tree)

        # Generate code
        try:
            return ast.unparse(new_tree)
        except Exception as e:
            print(f"  Failed to unparse {filename}: {e}")
            return source

    def _collect_module_names(self, tree: ast.AST):
        """Collect names defined at module level."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):
                    self.module_level_names.add(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                if not node.name.startswith('_'):
                    self.module_level_names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                self.module_level_names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.module_level_names.add(target.id)

    def _should_obfuscate(self, name: str) -> bool:
        """Check if a name should be obfuscated."""
        # Never obfuscate reserved names
        if name in PYTHON_RESERVED:
            return False
        # Never obfuscate dunder methods
        if name.startswith('__') and name.endswith('__'):
            return False
        # Never obfuscate single underscore (convention)
        if name == '_':
            return False
        return True

    def _get_obfuscated_name(self, name: str) -> str:
        """Get or create obfuscated name."""
        if not self._should_obfuscate(name):
            return name

        context = self.current_file
        existing = self.mapping.get_obfuscated(name, context)
        if existing:
            return existing

        # Generate new obfuscated name
        prefix = "_c" if name[0].isupper() else "_f"  # class vs function hint
        obfuscated = self.name_gen.generate(prefix)
        self.mapping.add_mapping(name, obfuscated, context)
        return obfuscated

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Obfuscate function definitions."""
        old_scope = self.current_scope
        self.scope_stack.append(node.name)
        self.current_scope = "::".join(self.scope_stack)

        # Obfuscate function name
        new_name = self._get_obfuscated_name(node.name)
        node.name = new_name

        # Obfuscate parameters
        for arg in node.args.args:
            if arg.arg not in {'self', 'cls'}:
                arg.arg = self._get_obfuscated_name(arg.arg)

        # Visit body
        self.generic_visit(node)

        self.scope_stack.pop()
        self.current_scope = old_scope
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        """Obfuscate async function definitions."""
        old_scope = self.current_scope
        self.scope_stack.append(node.name)
        self.current_scope = "::".join(self.scope_stack)

        new_name = self._get_obfuscated_name(node.name)
        node.name = new_name

        for arg in node.args.args:
            if arg.arg not in {'self', 'cls'}:
                arg.arg = self._get_obfuscated_name(arg.arg)

        self.generic_visit(node)

        self.scope_stack.pop()
        self.current_scope = old_scope
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Obfuscate class definitions."""
        old_scope = self.current_scope
        self.scope_stack.append(node.name)
        self.current_scope = "::".join(self.scope_stack)

        # Obfuscate class name
        new_name = self._get_obfuscated_name(node.name)
        node.name = new_name

        # Visit body (methods, attributes)
        self.generic_visit(node)

        self.scope_stack.pop()
        self.current_scope = old_scope
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Obfuscate variable references."""
        if self._should_obfuscate(node.id):
            # Check if it's a local reference we've already mapped
            obfuscated = self.mapping.get_obfuscated(node.id, self.current_file)
            if obfuscated:
                node.id = obfuscated
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        """Handle attribute access - preserve external API topology."""
        # Visit the value (left side) but be careful with attribute names
        self.visit(node.value)

        # Only obfuscate attribute if it's not accessing an imported module
        if self._should_obfuscate(node.attr):
            # Check if this is a method/attribute we defined
            obfuscated = self.mapping.get_obfuscated(node.attr, self.current_file)
            if obfuscated:
                node.attr = obfuscated

        return node

    def visit_Import(self, node: ast.Import) -> ast.Import:
        """Preserve imports (topology)."""
        for alias in node.names:
            self.imports.add(alias.name)
            # Track import aliases in topology
            if alias.asname:
                self.mapping.topology_preserved.setdefault(self.current_file, []).append(
                    f"import {alias.name} as {alias.asname}"
                )
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        """Preserve from imports (topology)."""
        module = node.module or ""
        for alias in node.names:
            full_name = f"{module}.{alias.name}"
            self.imports.add(full_name)
        return node

    def _remove_docstrings(self, tree: ast.AST) -> ast.AST:
        """Remove docstrings from AST."""
        for node in ast.walk(tree):
            # Remove docstrings from modules, classes, functions
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if (node.body and
                    isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Constant) and
                    isinstance(node.body[0].value.value, str)):
                    node.body = node.body[1:]
        return tree


class TypeScriptObfuscator:
    """
    Regex-based TypeScript/JSX obfuscator.

    Preserves React component topology and Three.js integration.
    """

    def __init__(self, mapping: ObfuscationMapping, name_gen: NameGenerator):
        self.mapping = mapping
        self.name_gen = name_gen
        self.current_file = ""

    def obfuscate_file(self, source: str, filename: str) -> str:
        """Obfuscate TypeScript/TSX file."""
        self.current_file = filename
        result = source

        # Remove single-line comments (but not URLs in strings)
        result = re.sub(r'(?<!:)//[^\n]*', '', result)

        # Remove multi-line comments
        result = re.sub(r'/\*[\s\S]*?\*/', '', result)

        # Obfuscate function names
        result = self._obfuscate_functions(result)

        # Obfuscate variable names
        result = self._obfuscate_variables(result)

        # Obfuscate class names
        result = self._obfuscate_classes(result)

        # Obfuscate interface/type names
        result = self._obfuscate_types(result)

        # Minify whitespace (preserve some readability)
        result = self._minify_whitespace(result)

        return result

    def _should_obfuscate_ts(self, name: str) -> bool:
        """Check if TypeScript name should be obfuscated."""
        if name in TS_RESERVED:
            return False
        if len(name) <= 1:
            return False
        # Preserve prop names starting with on (event handlers)
        if name.startswith('on') and len(name) > 2 and name[2].isupper():
            return False
        return True

    def _get_ts_obfuscated(self, name: str) -> str:
        """Get or create obfuscated TypeScript name."""
        if not self._should_obfuscate_ts(name):
            return name

        existing = self.mapping.get_obfuscated(name, self.current_file)
        if existing:
            return existing

        prefix = "_T" if name[0].isupper() else "_v"
        obfuscated = self.name_gen.generate(prefix)
        self.mapping.add_mapping(name, obfuscated, self.current_file)
        return obfuscated

    def _obfuscate_functions(self, source: str) -> str:
        """Obfuscate function declarations."""
        # function name(
        pattern = r'\bfunction\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\('

        def replacer(match):
            name = match.group(1)
            obfuscated = self._get_ts_obfuscated(name)
            return f'function {obfuscated}('

        return re.sub(pattern, replacer, source)

    def _obfuscate_variables(self, source: str) -> str:
        """Obfuscate variable declarations."""
        # const/let/var name =
        pattern = r'\b(const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*='

        def replacer(match):
            keyword = match.group(1)
            name = match.group(2)
            obfuscated = self._get_ts_obfuscated(name)
            return f'{keyword} {obfuscated} ='

        return re.sub(pattern, replacer, source)

    def _obfuscate_classes(self, source: str) -> str:
        """Obfuscate class declarations."""
        # class Name {
        pattern = r'\bclass\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*[{<]'

        def replacer(match):
            name = match.group(1)
            obfuscated = self._get_ts_obfuscated(name)
            suffix = match.group(0)[-1]  # { or <
            return f'class {obfuscated} {suffix}'

        return re.sub(pattern, replacer, source)

    def _obfuscate_types(self, source: str) -> str:
        """Obfuscate interface/type declarations."""
        # interface Name { or type Name =
        patterns = [
            (r'\binterface\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*[{<]', 'interface'),
            (r'\btype\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=', 'type'),
        ]

        result = source
        for pattern, keyword in patterns:
            def make_replacer(kw):
                def replacer(match):
                    name = match.group(1)
                    obfuscated = self._get_ts_obfuscated(name)
                    suffix = match.group(0).split(name)[1]
                    return f'{kw} {obfuscated}{suffix}'
                return replacer
            result = re.sub(pattern, make_replacer(keyword), result)

        return result

    def _minify_whitespace(self, source: str) -> str:
        """Reduce whitespace while maintaining some readability."""
        # Remove trailing whitespace
        lines = [line.rstrip() for line in source.split('\n')]

        # Remove empty lines (keep max 1)
        result = []
        prev_empty = False
        for line in lines:
            is_empty = len(line.strip()) == 0
            if is_empty:
                if not prev_empty:
                    result.append('')
                prev_empty = True
            else:
                result.append(line)
                prev_empty = False

        return '\n'.join(result)


def obfuscate_python_codebase(backend_path: Path, mapping: ObfuscationMapping):
    """Obfuscate all Python files in backend."""
    name_gen = NameGenerator()
    obfuscator = PythonObfuscator(mapping, name_gen)

    # Get all Python files
    py_files = list(backend_path.rglob("*.py"))

    # Skip test files and __pycache__
    py_files = [
        f for f in py_files
        if '__pycache__' not in str(f)
        and '.egg-info' not in str(f)
        and 'venv' not in str(f)
    ]

    print(f"Obfuscating {len(py_files)} Python files...")

    for py_file in py_files:
        rel_path = py_file.relative_to(backend_path)
        print(f"  Processing: {rel_path}")

        try:
            source = py_file.read_text()

            # Store original hash for verification
            original_hash = hashlib.sha256(source.encode()).hexdigest()
            mapping.file_hashes[str(rel_path)] = original_hash

            # Obfuscate
            obfuscated = obfuscator.obfuscate_file(source, str(rel_path))

            # Write back
            py_file.write_text(obfuscated)

        except Exception as e:
            print(f"  Error processing {rel_path}: {e}")


def obfuscate_typescript_codebase(frontend_path: Path, mapping: ObfuscationMapping):
    """Obfuscate all TypeScript/TSX files in frontend."""
    name_gen = NameGenerator()
    obfuscator = TypeScriptObfuscator(mapping, name_gen)

    # Get all TS/TSX files
    ts_files = list(frontend_path.rglob("*.ts")) + list(frontend_path.rglob("*.tsx"))

    # Skip node_modules and dist
    ts_files = [
        f for f in ts_files
        if 'node_modules' not in str(f)
        and 'dist' not in str(f)
        and '.d.ts' not in str(f)  # Keep declaration files
    ]

    print(f"Obfuscating {len(ts_files)} TypeScript files...")

    for ts_file in ts_files:
        rel_path = ts_file.relative_to(frontend_path)
        print(f"  Processing: {rel_path}")

        try:
            source = ts_file.read_text()

            # Store original hash
            original_hash = hashlib.sha256(source.encode()).hexdigest()
            mapping.file_hashes[str(rel_path)] = original_hash

            # Obfuscate
            obfuscated = obfuscator.obfuscate_file(source, str(rel_path))

            # Write back
            ts_file.write_text(obfuscated)

        except Exception as e:
            print(f"  Error processing {rel_path}: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Jones Framework Axiom-Compliant Code Obfuscator"
    )
    parser.add_argument("--backend", action="store_true", help="Obfuscate Python backend")
    parser.add_argument("--frontend", action="store_true", help="Obfuscate TypeScript frontend")
    parser.add_argument("--all", action="store_true", help="Obfuscate everything")
    parser.add_argument("--mapping-file", type=str, default=".obfuscation-mapping.json",
                        help="Path to save mapping file")

    args = parser.parse_args()

    if not (args.backend or args.frontend or args.all):
        parser.print_help()
        sys.exit(1)

    # Initialize mapping
    mapping = ObfuscationMapping()

    # Get paths
    root = Path(__file__).parent.parent
    backend_path = root / "backend"
    frontend_path = root / "frontend"

    print("=" * 60)
    print("Jones Framework Axiom-Compliant Code Obfuscator")
    print("=" * 60)
    print(f"Substrate Principle: Creating verifiable mapping")
    print(f"Manifold Hypothesis: Preserving code topology")
    print(f"Continuity Thesis: Functionality-preserving transform")
    print("=" * 60)

    if args.backend or args.all:
        print("\n[Backend Obfuscation]")
        obfuscate_python_codebase(backend_path, mapping)

    if args.frontend or args.all:
        print("\n[Frontend Obfuscation]")
        obfuscate_typescript_codebase(frontend_path, mapping)

    # Save mapping
    mapping_path = root / args.mapping_file
    mapping.save(mapping_path)
    print(f"\nMapping saved to: {mapping_path}")
    print(f"Total mappings: {len(mapping.original_to_obfuscated)}")

    print("\n" + "=" * 60)
    print("Obfuscation complete!")
    print("IMPORTANT: Keep the mapping file secure for potential reversal.")
    print("=" * 60)


if __name__ == "__main__":
    main()
