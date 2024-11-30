# %%
import ast
import inspect
from collections import defaultdict
from typing import Callable, Collection, Optional, Set

from ml_re_adv.utils.extra_ast import get_called_functions


class DocstringRemover(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)
        return self.generic_visit(node)


class TypeAnnotationRemover(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        # Remove type annotations from function arguments
        for arg in node.args.args:
            arg.annotation = None
        if node.returns:
            node.returns = None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        # Remove type annotations from variable assignments
        node.annotation = None
        return self.generic_visit(node)


def getsource_and_imports_recursively(
    f: Callable,
    visited: Optional[Set[str]] = None,
    include_comments: bool = True,
    include_type_annotations: bool = True,
    import_modules: Collection[str] = (),
) -> tuple[str, dict[str, set[str]]]:
    """
    Recursively retrieves the source code of a function `f` and the source code of all functions
    that are called by `f` in the form x.y.z(...).

    Args:
        f (Callable): The function to retrieve the source code for.
        visited (Optional[Set[str]]): A set of function names that have already been visited
                                      to avoid processing the same function multiple times.

    Returns:
        str: The concatenated source code of the function and all recursively called functions.
    """
    if visited is None:
        visited = set()

    # Get source code of the function
    imports = defaultdict(set)
    if import_modules:
        module = inspect.getmodule(f)
        if module and module.__name__ in import_modules:
            imports[module.__name__].add(f.__name__)
            return "", imports
    try:
        source_code = inspect.getsource(f)
    except OSError:
        return ""  # Skip if we can't get the source (e.g., built-in functions)

    # Add function name to the visited set
    visited.add(f.__name__)

    # Parse the function's source code into an AST
    parsed_code = ast.parse(source_code)

    # Get all function names that are called as x.y.z(...)
    called_functions = get_called_functions(parsed_code)

    if not include_comments:
        # Remove docstrings only from FunctionDef nodes
        parsed_code = DocstringRemover().visit(parsed_code)
        if not include_type_annotations:
            # Remove type annotations from FunctionDef and AnnAssign nodes
            parsed_code = TypeAnnotationRemover().visit(parsed_code)
        ast.fix_missing_locations(parsed_code)
        source_code = ast.unparse(parsed_code)
    else:
        assert (
            include_type_annotations
        ), "Cannot include type annotations without including comments"

    result = source_code

    # Recursively get the source code of called functions
    for func_name in called_functions:
        try:
            # Resolve the function object from the current scope
            func = eval(func_name, f.__globals__)
            if inspect.isfunction(func) and func.__name__ not in visited:
                fn_body, fn_imports = getsource_and_imports_recursively(
                    func,
                    visited,
                    include_comments=include_comments,
                    import_modules=import_modules,
                    include_type_annotations=include_type_annotations,
                )
                if fn_body:
                    result += "\n\n" + fn_body
                for module, names in fn_imports.items():
                    imports[module].update(names)
        except (NameError, AttributeError):
            # Skip if the function cannot be resolved (e.g., if it doesn't exist in the current scope)
            continue

    return result, imports


def getsource_recursively(
    f: Callable,
    *,
    visited: Optional[Set[str]] = None,
    include_comments: bool = True,
    include_type_annotations: bool = True,
    import_modules: Collection[str] = (),
) -> str:
    """
    Recursively retrieves the source code of a function `f` and the source code of all functions
    that are called by `f` in the form x.y.z(...).

    Args:
        f (Callable): The function to retrieve the source code for.
        visited (Optional[Set[str]]): A set of function names that have already been visited
                                      to avoid processing the same function multiple times.

    Returns:
        str: The concatenated source code of the function and all recursively called functions.
    """
    result, imports = getsource_and_imports_recursively(
        f,
        visited=visited,
        include_comments=include_comments,
        include_type_annotations=include_type_annotations,
        import_modules=import_modules,
    )
    if imports:
        result = "\n" + result
        for module, names in imports.items():
            result = f"from {module} import {', '.join(names)}\n" + result
        result = f"import {', '.join(imports)}\n\n{result}"
    return result


# %%
