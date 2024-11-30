import argparse
import io
import logging
import pickle
import re
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from logging import Logger
from pathlib import Path

# from types import CodeType
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    overload,
)

import dill

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt

except ImportError:
    plt = None
    logger.warning("Matplotlib not available")
try:
    from PIL import Image

    ImageType = Image.Image
except ImportError:
    Image = None
    ImageType = Any
    logger.warning("Pillow not available")

# from _typeshed import ReadableBuffer


T = TypeVar("T")
A = TypeVar("A")
B = TypeVar("B")
E = TypeVar("E")
_exec = exec

file_path = Path(__file__)
cache_dir = file_path.parent / ".cache"
base_path = cache_dir / "exec"


state_persistance_logger = logging.getLogger(f"{__name__}.state_persistance")
### State utils

state_logger = logging.getLogger(f"{__name__}.state")
state_logger.setLevel(logging.INFO)


@overload
def make_state(
    default: T, *, name: Optional[str] = None, logger: Logger = state_logger
) -> Tuple[Callable[[T], None], Callable[[], T]]: ...


@overload
def make_state(
    default: None = None, *, name: Optional[str] = None, logger: Logger = state_logger
) -> Tuple[Callable[[T], None], Callable[[], Optional[T]]]: ...


def make_state(
    default: Optional[T] = None,
    *,
    name: Optional[str] = None,
    logger: Logger = state_logger,
) -> Tuple[Callable[[T], None], Callable[[], Optional[T]]]:
    """Returns a setter and getter for a new state variable.

    Args:
        default (T, optional): default value of state. Defaults to None.

    Returns:
        Tuple[Callable[[T], None], Callable[[], Optional[T]]]: setter and getter for state

        If the default value is passed explicitly and is not None, the getter will return a non-optional value.
    """
    state: Optional[T] = default

    def set_state(new_state: T) -> None:
        nonlocal state
        logger.debug("Updating %s state: %s => %s", name, state, new_state)
        state = new_state

    def get_state() -> Optional[T]:
        return state

    return set_state, get_state


### End State utils

# units = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}

# Alternative unit definitions, notably used by Windows:
# units = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40}
units = {
    "b": 1,
    "kb": 1000,
    "kib": 1024,
    "mib": 1024**2,
    "mb": 1000**2,
    "gib": 1024**3,
    "gb": 1000**3,
    "tib": 1024**4,
    "tb": 1000**4,
}


def parse_size(size: str):
    size = size.strip()
    match = re.search(r"([0-9.]+)\s*([a-zA-Z]+)", size)
    if not match:
        raise ValueError(f"Invalid size format: {size}")
    number, unit = match.groups()
    return int(float(number) * units[unit.lower()])


class ExecResults(TypedDict, total=False):
    stdout: str
    stderr: str
    output: Optional[Any]
    tb: Optional[str]
    images: list[ImageType]
    state_loading_error: Optional[Exception]
    state_saving_error: Optional[Exception]
    start: float
    end: float
    duration: float


### State persistance utils


def get_state_file(
    session_id: Optional[str] = None, *, state_dir: str | Path = base_path
) -> Path:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    if session_id is None:
        return state_dir / "state.pkl"
    return state_dir / f"state_{session_id}.pkl"


def save_state(
    state: dict[str, Any],
    session_id: Optional[str] = None,
    *,
    results: Optional[ExecResults] = None,
    max_file_bytes: Optional[int | str] = None,
    state_dir: str | Path = base_path,
):
    max_file_size = max_file_bytes
    if isinstance(max_file_bytes, str):
        max_file_bytes = parse_size(max_file_bytes)
    # avoid corrupting state with errors by writing to a temporary file first
    state_file = get_state_file(session_id, state_dir=state_dir)
    temp_file = state_file.with_suffix(".tmp")
    try:
        with open(temp_file, "wb") as f:
            dill.dump(state, f)
    except OSError as e:
        if results is not None:
            results["state_saving_error"] = e
        temp_file.unlink(missing_ok=True)
        return
    if max_file_bytes and temp_file.stat().st_size > max_file_bytes:
        if results is not None:
            results["state_saving_error"] = ValueError(
                f"Temporary file size ({temp_file.stat().st_size}) exceeds the maximum allowed size ({max_file_bytes}B ({max_file_size}))."
            )
        temp_file.unlink(missing_ok=True)
        return
    temp_file.replace(state_file)
    return state_file


def load_state(
    session_id: Optional[str] = None,
    *,
    results: Optional[ExecResults] = None,
    state_dir: str | Path = base_path,
) -> dict[str, Any]:
    state_file = get_state_file(session_id, state_dir=state_dir)
    if state_file.exists():
        try:
            with open(state_file, "rb") as f:
                return dill.load(f)
        except pickle.UnpicklingError as e:
            if results is not None:
                results["state_loading_error"] = e
    return {}


### End State persistance utils


def make_custom_show(generated_images: Optional[list[ImageType]] = None):
    if generated_images is None:
        generated_images = []

    def custom_show():
        # Create a bytes buffer to store the image
        buf = io.BytesIO()

        # Save the current figure to the buffer as PNG
        plt.savefig(buf, format="png")

        # Seek to the start of the buffer
        buf.seek(0)

        # Read the image from the buffer using PIL and store it in the list
        img = Image.open(buf)
        generated_images.append(img)

        # Clear the figure after saving it
        plt.close()

    return generated_images, custom_show


def capture_stdout_stderr(
    func: Callable[..., T], *args, reraise: bool = False, **kwargs
) -> Tuple[Union[Tuple[T, None], Tuple[None, Exception]], Tuple[str, str]]:
    """
    Capture stdout and stderr from an exec call.
    """
    stdout = io.StringIO()
    stderr = io.StringIO()

    result_err: Union[Tuple[T, None], Tuple[None, Exception]]
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            result = func(*args, **kwargs)
            result_err = (result, None)
        except Exception as e:
            result_err = (None, e)
            if reraise:
                raise

    return result_err, (stdout.getvalue(), stderr.getvalue())


def capture_exec_plt_show(
    source: str,
    globals: dict[str, Any] | None = None,
    locals: Mapping[str, object] | None = None,
    *,
    exec: Callable[
        [str, dict[str, Any] | None, Mapping[str, object] | None], T
    ] = _exec,
    **kwargs,
) -> Tuple[T, list[ImageType]]:
    generated_images, custom_show = make_custom_show()
    _exec("import matplotlib.pyplot as plt", globals, locals)
    ext_globals = globals or {}
    ext_locals = locals or {}
    orig_showg = globals["plt"].show if globals and "plt" in globals else None
    orig_showl = locals["plt"].show if locals and "plt" in locals else None
    assert "plt" in ext_globals or "plt" in ext_locals, (
        "No plt in globals or locals",
        ext_globals.keys(),
        ext_locals.keys(),
    )
    if globals and "plt" in globals:
        globals["plt"].show = custom_show
    if locals and "plt" in locals:
        locals["plt"].show = custom_show
    try:
        result = exec(source, globals, locals, **kwargs)
        return result, generated_images
    finally:
        if globals and "plt" in globals:
            globals["plt"].show = orig_showg
        if locals and "plt" in locals:
            locals["plt"].show = orig_showl


def is_syntax_error(source: str) -> bool:
    try:
        compile(source, "<string>", "exec")
    except SyntaxError:
        return True
    except Exception:
        pass
    return False


def capture_exec_output(
    source,
    globals: dict[str, Any] | None = None,
    locals: Mapping[str, object] | None = None,
    *,
    outputs_name: str = "_OUTPUTS",
    exec: Callable[
        [str, dict[str, Any] | None, Mapping[str, object] | None], T
    ] = _exec,
    **kwargs,
) -> Tuple[T, Any]:
    assert isinstance(source, str), type(source)
    orig_source = source
    lines = source.split("\n")
    if (
        not lines[-1].startswith(" ")
        and not lines[-1].startswith("print")
        and not lines[-1].startswith(")")
    ):
        final_line = f"{outputs_name}.append(({lines[-1]}))"
        if not is_syntax_error(final_line):
            lines[-1] = final_line
    if globals is None:
        globals = {}
    globals.setdefault(outputs_name, [])
    n = len(globals[outputs_name])
    result = exec("\n".join(lines), globals, locals, **kwargs)

    return result, (
        globals[outputs_name][n] if n < len(globals[outputs_name]) else None
    )


def capture_exec_exception(
    source,
    globals: dict[str, Any] | None = None,
    locals: Mapping[str, object] | None = None,
    *,
    errors_name: str = "_ERRORS",
    indent: str = "    ",
    exec: Callable[
        [str, dict[str, Any] | None, Mapping[str, object] | None], T
    ] = _exec,
    preformat_exn: Callable[[str], str] = lambda e: "",
    format_exn: Callable[[str], str] = lambda e: e,
    format_exn_external: Callable[[Exception], E] = lambda e: e,
    **kwargs,
) -> Tuple[Optional[T], Optional[Exception] | E]:
    assert isinstance(source, str), type(source)
    if globals is None:
        globals = {}
    globals.setdefault(errors_name, [])
    n = len(globals[errors_name])
    lines = source.split("\n")
    lines = [f"{indent}{line}" for line in lines]
    source = "\n".join(lines)
    source = rf"""try:
{source}
except Exception as e:
    {preformat_exn('e')}
    {errors_name}.append({format_exn('e')})
"""
    try:
        result = exec(source, globals, locals, **kwargs)
        return result, (
            globals[errors_name][n] if n < len(globals[errors_name]) else None
        )
    except Exception as e:
        return None, format_exn_external(e)


def sourcelines_with_context(
    lines: Sequence[str], error_line: int, context_lines: int = 3
):
    """Prints the source code around the error line with context."""
    start_line = max(error_line - context_lines - 1, 0)
    end_line = min(error_line + context_lines, len(lines))
    result = []
    for i in range(start_line, end_line):
        prefix = ">>> " if i == error_line - 1 else "    "
        result.append(f"{prefix}{i + 1}: {lines[i].rstrip()}")
    return "\n".join(result)


def source_with_context(file_name: Path | str, error_line: int, context_lines: int = 3):
    """Prints the source code around the error line with context."""
    try:
        with open(file_name, mode="r", encoding="utf-8") as f:
            return sourcelines_with_context(
                f.readlines(), error_line, context_lines=context_lines
            )
    except FileNotFoundError:
        return f"File '{file_name}' not found."


def capture_exec_traceback(
    source,
    globals: dict[str, Any] | None = None,
    locals: Mapping[str, object] | None = None,
    *,
    errors_name: str = "_ERRORS",
    indent: str = "    ",
    exec: Callable[[...], A] = _exec,
    split_output: Callable[[A], Tuple[str, B]],
    **kwargs,
) -> Tuple[B, Optional[str]]:
    result, e = capture_exec_exception(
        source,
        globals=globals,
        locals=locals,
        errors_name=errors_name,
        indent=indent,
        exec=exec,
        # preformat_exn=lambda e: "import traceback",
        # format_exn=lambda e: "traceback.format_exc()",
        **kwargs,
    )
    final_source, output = split_output(result)
    if e is None:
        return output, None
    # global exn
    # exn = e

    # # Get the full traceback information
    # tb = e.__traceback__

    # # Extract file name and line number from the last traceback frame
    # while tb.tb_next:
    #     tb = tb.tb_next

    # error_file = tb.tb_frame.f_code.co_filename
    # error_line = tb.tb_lineno

    tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
    tb_extended_lines = []
    sourcelines = final_source.split("\n")
    for line in tb_lines:
        tb_extended_lines.append(line)
        if line.lstrip().startswith('File "<string>", line '):
            error_line = int(
                line.lstrip()[len('File "<string>", line ') :].split(",")[0]
            )

            tb_extended_lines.append(
                sourcelines_with_context(sourcelines, error_line, context_lines=3)
                + "\n"
            )

    return output, "".join(tb_extended_lines)


def capture_exec_persist_state(
    source,
    globals: Optional[dict[str, Any]] = None,
    locals: Mapping[str, object] | None = None,
    *,
    exec: Callable[[str, dict[str, Any] | None, Mapping[str, Any] | None], T] = _exec,
    prev_session_id: Optional[str] = None,
    session_id: Optional[str] = None,
    results: Optional[ExecResults] = None,
    max_file_bytes: Optional[int | str] = None,
    state_dir: str | Path = base_path,
    **kwargs,
) -> T:
    if globals is None:
        globals = {}
    elif globals:
        raise ValueError("Cannot persist state with globals set; value:", globals)
    globals.update(
        load_state(prev_session_id or session_id, results=results, state_dir=state_dir)
    )

    try:
        return exec(source, globals, locals, **kwargs)
    finally:
        save_state(
            globals,
            session_id,
            results=results,
            max_file_bytes=max_file_bytes,
            state_dir=state_dir,
        )


def capture_exec(
    source,
    globals: dict[str, Any] | None = None,
    locals: Mapping[str, object] | None = None,
    *,
    exec: Callable[
        [str, dict[str, Any] | None, Mapping[str, object] | None], T
    ] = _exec,
    persist_state: bool = False,
    capture_stdouterr: bool = True,
    capture_tb: bool = True,
    capture_images: bool = True,
    capture_output: bool = True,
    max_file_bytes: Optional[int | str] = None,
    state_dir: str | Path = base_path,
    **kwargs,
) -> Tuple[
    Optional[T],
    ExecResults,
]:

    results: ExecResults = {}

    def exec0(source, globals, locals, **kwargs):
        results["start"] = time.time()
        try:
            return exec(source, globals, locals, **kwargs)
        finally:
            results["end"] = time.time()
            results["duration"] = results["end"] - results["start"]

    save_source, get_source = make_state(default="")

    def exec1(source, globals, locals, **kwargs):
        save_source(source)
        if not capture_stdouterr:
            return exec0(source, globals, locals, **kwargs)
        (result, _exn), (results["stdout"], results["stderr"]) = capture_stdout_stderr(
            exec0, source, globals=globals, locals=locals, reraise=True, **kwargs
        )
        logger.debug(
            "After:\n======\n%s\n======\nGlobals: %s\nLocals: %s",
            source,
            (globals or {}).keys(),
            (locals or {}).keys(),
        )
        return result

    def exec2(source, globals, locals, **kwargs):
        if not capture_tb:
            return exec1(source, globals=globals, locals=locals, **kwargs)
        result, results["tb"] = capture_exec_traceback(
            source,
            globals=globals,
            locals=locals,
            exec=exec1,
            split_output=lambda x: (get_source(), x),
            **kwargs,
        )
        return result

    def exec3(source, globals, locals, **kwargs):
        if not capture_images:
            return exec2(source, globals=globals, locals=locals, **kwargs)
        result, results["images"] = capture_exec_plt_show(
            source, globals=globals, locals=locals, exec=exec2, **kwargs
        )

    def exec4(source, globals, locals, **kwargs):
        if not capture_output:
            return exec3(source, globals=globals, locals=locals, **kwargs)
        result, results["output"] = capture_exec_output(
            source, globals=globals, locals=locals, exec=exec3, **kwargs
        )
        return result

    def exec5(source, globals, locals, **kwargs):
        if not persist_state:
            return exec4(source, globals=globals, locals=locals, **kwargs)
        return capture_exec_persist_state(
            source,
            globals=globals,
            locals=locals,
            exec=exec4,
            results=results,
            max_file_bytes=max_file_bytes,
            state_dir=state_dir,
            **kwargs,
        )

    res = exec5(source, globals=globals, locals=locals, **kwargs)
    return res, results


def capture_sequential_exec(
    sources: list,
    globals: dict[str, Any] | None = None,
    locals: Mapping[str, object] | None = None,
    *,
    exec: Callable[
        [str, dict[str, Any] | None, Mapping[str, object] | None], T
    ] = _exec,
    stop_on_error: bool = True,
    **kwargs,
) -> list[
    Tuple[
        Optional[T],
        ExecResults,
    ]
]:
    results = []
    for source in sources:
        results.append(
            capture_exec(source, globals=globals, locals=locals, exec=exec, **kwargs)
        )
        _result, values = results[-1]
        if values.get("tb") and stop_on_error:
            break
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an interactive Python session with state persistence."
    )
    parser.add_argument(
        "--id", type=str, default=None, help="Session ID for state differentiation"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=argparse.FileType(mode="wb"),
        default=None,
        help="output file to save the output to",
    )
    parser.add_argument(
        "--code",
        "-c",
        type=str,
        help="program passed in as string (terminates option list)",
    )
    parser.add_argument(
        "--max-file-size",
        type=str,
        default=None,
        help="Max file size for state persistance",
    )
    parser.add_argument(
        "--state-dir",
        type=str,
        default=base_path,
        help="Directory to store the state files",
    )
    args = parser.parse_args()

    session_id = args.id

    state_persistance_logger.debug("Starting session with ID: %s", session_id)

    capture_pipes: bool = args.output is not None

    code = args.code or sys.stdin.read()

    result, values = capture_exec(
        code,
        persist_state=True,
        session_id=session_id,
        capture_images=capture_pipes,
        capture_stdouterr=capture_pipes,
        max_file_bytes=args.max_file_size,
        state_dir=args.state_dir,
    )

    if args.output:
        pickle.dump((result, values), args.output)
    else:
        if values.get("tb"):
            print(values["tb"], file=sys.stderr)
        if values.get("output"):
            print(values["output"], file=sys.stdout)

    state_persistance_logger.debug("Ending session with ID: %s", session_id)
