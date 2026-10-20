"""Relocate the two reviewed notebooks' inputs without editing their frozen code.

Only exact, hash-reviewed input literals are mapped. The mapping belongs to the
verification harness and is installed only in its dedicated execution kernel.
"""
import ast
from pathlib import Path


def reviewed_input_mapping(code_cells, current_input):
    """Find the reviewed first cell's single literal ``input_path = Path(...)``."""
    tree = ast.parse(code_cells[0])
    candidates = [
        node.value.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "input_path" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id == "Path"
        and len(node.value.args) == 1 and not node.value.keywords
        and isinstance(node.value.args[0], ast.Constant)
        and isinstance(node.value.args[0].value, str)
    ]
    if len(candidates) != 1:
        raise ValueError("Reviewed loading cell must declare exactly one literal input path")
    current = Path(current_input).resolve(strict=True)
    original = candidates[0].replace("\\", "/")
    if original.rsplit("/", 1)[-1] != current.name:
        raise ValueError("Relocated input must have the same reviewed dataset filename")
    return {original: str(current)}


def path_mapping_source(mapping):
    """Return an explicit harness cell; retain the original pathlib.Path type.

    Mapping reads/stat calls rather than rewriting generated source also keeps
    the existing source-code hash gate and scientific reference checks intact.
    The ordinary file-open audit still sees and checks the actual target path.
    """
    return (
        "# TEST HARNESS: relocate only reviewed input file accesses.\n"
        "from pathlib import Path as _RelocatedPath\n"
        f"_test_input_mapping = {mapping!r}\n"
        "_test_original_path_open = _RelocatedPath.open\n"
        "_test_original_path_stat = _RelocatedPath.stat\n"
        "def _test_mapped_path(path):\n"
        "    key = str(path).replace('\\\\', '/')\n"
        "    return _RelocatedPath(_test_input_mapping[key]) if key in _test_input_mapping else path\n"
        "def _test_relocated_open(path, *args, **kwargs):\n"
        "    return _test_original_path_open(_test_mapped_path(path), *args, **kwargs)\n"
        "def _test_relocated_stat(path, *args, **kwargs):\n"
        "    return _test_original_path_stat(_test_mapped_path(path), *args, **kwargs)\n"
        "_RelocatedPath.open = _test_relocated_open\n"
        "_RelocatedPath.stat = _test_relocated_stat\n"
    )
