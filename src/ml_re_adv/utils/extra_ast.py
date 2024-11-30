import ast
from typing import Set


def get_called_functions(node: ast.AST) -> Set[str]:
    """
    Extracts all function names that are called in the form x.y.z(...) from the AST of the function.

    Args:
        node (ast.AST): The AST of the function source code.

    Returns:
        Set[str]: A set of function names that are called in the form x.y.z(...).
    """
    called_functions = set()

    # Recursively traverse the AST and collect all function calls
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            func_name = ""
            curr = n.func
            while isinstance(curr, ast.Attribute):
                func_name = curr.attr + ("." if func_name else "") + func_name
                curr = curr.value
            if isinstance(curr, ast.Name):
                func_name = curr.id + ("." if func_name else "") + func_name
            if func_name:
                called_functions.add(func_name)

    return called_functions
