import logging
import pickle
import shelve
from pathlib import Path
from typing import Any, Optional, Tuple

from inspect_ai.tool import (
    Content,
    ContentImage,
    ContentText,
    Tool,
    ToolError,
    tool,
)
from inspect_ai.util import sandbox, store

from ml_re_adv.shared_assets.exec import ExecResults, get_state_file
from ml_re_adv.shared_assets import path as shared_assets_path
from ml_re_adv.utils.images import image_to_base64

logger = logging.getLogger(__name__)

_exec_file_local = shared_assets_path / "shared_assets" / "exec.py"
assert _exec_file_local.exists(), _exec_file_local
_exec_file_contents_bytes = _exec_file_local.read_bytes()
_sandbox_exec_cache_dir = shared_assets_path / "shared_assets" / ".cache" / "sandbox_exec"


async def python_exec_in_sandbox(
    code: str,
    timeout: int | None = None,
    user: str | None = None,
    *,
    exec_file_remote: str = "/tmp/exec.py",
    session_id: Optional[str] = None,
    max_file_size: Optional[str] = None,
    cache: bool = False,
    local_cache_dir: str | Path = _sandbox_exec_cache_dir,
    remote_cache_dir: str | Path = "/tmp/.cache/sandbox_exec",
    name: Optional[str] = None,
) -> ExecResults:
    """
    Use the python function to execute Python code.

    Args:
      code (str): The python code to execute.
      timeout (int | None): Timeout (in seconds) for command.
      user (str | None): User to execute commands as.
      exec_file_remote (str, *optional*): The path to the location in the sandbox where we will copy the exec file.
      max_file_size (str, *optional*): The maximum file size for the saved state.
      cache (bool, optional): Whether to cache the results and state of the execution locally.
      local_cache_dir (str | Path, optional): The directory to use for caching the results and state of the execution.
      name (str, optional): The name of the execution context.

    Returns:
      The output of the command.
    """
    remote_cache_dir = Path(remote_cache_dir)
    local_cache_dir = Path(local_cache_dir)
    cache_map = store().get("python_exec_in_sandbox", {})
    store().set("python_exec_in_sandbox", cache_map)
    cache_key = (name, session_id)
    cache_file = local_cache_dir / str(
        cache_map.setdefault(cache_key, "initial-state.pkl")
    )
    key = str((user, timeout, code)) if cache else code
    name_descr = f" ({name})" if name is not None else ""
    if cache and cache_file.exists():
        try:
            with shelve.open(cache_file) as db:
                if key in db:
                    try:
                        cache_map[cache_key] = db[key]["next"]
                        state_bytes = Path(db[key]["state"]).read_bytes()
                        results = pickle.loads(db[key]["results"])
                        remote_state_file = get_state_file(
                            session_id=session_id, state_dir=remote_cache_dir
                        )
                        logger.debug("Updating sandbox state (%s) from cache", name)
                        await sandbox(name=name).write_file(
                            str(remote_state_file),
                            state_bytes,
                        )
                        return results
                    except Exception as e:
                        logger.error(
                            "Error reading cache file %s [%s] (%s): %s",
                            cache_file,
                            cache_key,
                            key,
                            e,
                        )
                        del db[key]
        except Exception as e:
            logger.error(
                "Error opening cache file %s [%s]: %s", cache_file, cache_key, e
            )
            cache_file.unlink(missing_ok=True)

    # "0" if the file exists and is executable, "2" if it exists but is not executable, "1" if it does not exist
    file_exists = await sandbox(name=name).exec(
        cmd=[
            "bash",
            "-c",
            f"{{ test -x {exec_file_remote} && echo 0; }} || {{ test -f {exec_file_remote} && echo 2; }} || echo 1",
        ],
        user=user,
    )
    if not file_exists.success:
        raise ToolError(
            f"Failed to check if exec file exists: {file_exists.stderr} (stdout: {file_exists.stdout!r})"
        )
    match file_exists.stdout.strip(" \n"):
        case "0":
            pass
        case "2":
            await sandbox(name=name).exec(
                cmd=["chmod", "+x", exec_file_remote], user=user
            )
        case "1":
            await sandbox(name=name).write_file(
                exec_file_remote, _exec_file_contents_bytes
            )
            await sandbox(name=name).exec(
                cmd=["chmod", "+x", exec_file_remote], user=user
            )
        case _:
            raise RuntimeError(
                f"Unexpected file_exists value: {file_exists!r} (for {exec_file_remote})"
            )

    session_id_args = ["--id", session_id] if session_id else []
    # remote_output_file_result = await sandbox(name=name).exec(cmd=["mktemp"], user=user)
    # assert remote_output_file_result.success, (remote_output_file_result, user)
    # remote_output_file = remote_output_file_result.stdout.strip()
    remote_output_file = f"/tmp/output-{session_id}-{hash(code)}.pkl"

    cache_args = ["--state-dir", str(remote_cache_dir)] if cache else []
    remote_state_file = get_state_file(
        session_id=session_id, state_dir=remote_cache_dir
    )

    logger.debug("Executing code in sandbox%s:\n```python\n%s\n```\n", name_descr, code)
    max_file_size_cmd = ["--max-file-size", max_file_size] if max_file_size else []
    exec_result = await sandbox(name=name).exec(
        cmd=[
            "python3",
            exec_file_remote,
            *session_id_args,
            "-o",
            remote_output_file,
            *cache_args,
            *max_file_size_cmd,
        ],
        input=code,
        timeout=timeout,
        user=user,
    )

    output_contents = await sandbox(name=name).read_file(remote_output_file, text=False)
    try:
        _result, results = pickle.loads(output_contents)
    except (pickle.UnpicklingError, EOFError) as e:
        raise ToolError(
            f"Internal error unpickling results: {e}\nExec result: {exec_result}\nCode: {code}\nOutput: {output_contents!r}"
        ) from e
    logger.debug("Exec result: %s", results)

    await sandbox(name=name).exec(cmd=["rm", remote_output_file], user=user)

    if cache:
        local_state_file = (
            local_cache_dir
            / f"state-{session_id}-{hash(Path(cache_map[cache_key]).stem)}-{hash(key)}.pkl"
        )
        local_state_cache_file = local_state_file.with_suffix(".cache.pkl")
        try:
            state_bytes = await sandbox(name=name).read_file(
                str(remote_state_file), text=False
            )
        except Exception as e:
            logger.error("Error reading remote sandbox state: %s", e)
            logger.info(
                await sandbox(name=name).exec(cmd=["find", str(remote_cache_dir)])
            )
            return results
        try:
            assert not local_state_file.exists(), local_state_file
            local_state_file.parent.mkdir(parents=True, exist_ok=True)
            local_state_file.write_bytes(state_bytes)
        except Exception as e:
            logger.error("Error writing local sandbox state: %s", e)
            return results
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with shelve.open(cache_file) as db:
                cache_map[cache_key] = str(local_state_cache_file)
                db[key] = {
                    "state": str(local_state_file),
                    "results": results,
                    "next": cache_map[cache_key],
                }
        except Exception as e:
            logger.error("Error caching sandbox state: %s", e)
            return results
        logger.debug("Cached sandbox state to %s", local_state_file)

    return results


def exec_results_to_messages(results: ExecResults) -> list[Content]:
    result_messages: list[Content] = []
    images = results.get("images")
    if images:
        result_messages.extend(
            ContentImage(image=f"data:image/png;base64,{image_to_base64(image)}")
            for image in images
        )
    if images is not None:
        del results["images"]
    keys = ["output", "stdout", "stderr", "tb"]
    keys += [key for key in results if key not in keys]
    for key in keys:
        contents = results.get(key)
        if contents is not None and contents != "":
            result_messages.append(
                ContentText(
                    text=(
                        f"{key}: {results.get(key)}"
                        if "\n" not in str(results.get(key))
                        else f"{key}:\n```\n{results.get(key)}\n```\n"
                    ),
                )
            )
            del results[key]

    return result_messages


@tool
def python_exec(
    timeout: int | None = None,
    user: str | None = None,
    *,
    exec_file_remote: str = "/tmp/exec.py",
    session_id: Optional[str] = None,
    max_file_size: Optional[str] = None,
    cache: bool = False,
    local_cache_dir: str | Path = _sandbox_exec_cache_dir,
    remote_cache_dir: str | Path = "/tmp/.cache/sandbox_exec",
    name: Optional[str] = None,
    **kwargs,
) -> Tool:
    """Python code execution tool.

    Execute Python code using a sandbox environment (e.g. "docker").

    Args:
      timeout (int | None): Timeout (in seconds) for command.
      user (str | None): User to execute commands as.
      exec_file_remote (str, *optional*): The path to the location in the sandbox where we will copy the exec file.
      max_file_size (str, *optional*): The maximum file size for the saved state.
      cache (bool, optional): Whether to cache the results and state of the execution locally.
      local_cache_dir (str | Path, optional): The directory to use for caching the results and state of the execution.
      name (str, optional): The name of the execution context.

    Returns:
      String with command output (stdout) or command error (stderr).
    """

    async def execute(code: str) -> str | list[Content]:
        """
        Use the python function to execute Python code.

        Args:
          code (str): The python code to execute.

        Returns:
          The output of the command.
        """

        results = await python_exec_in_sandbox(
            code=code,
            timeout=timeout,
            user=user,
            exec_file_remote=exec_file_remote,
            session_id=session_id,
            max_file_size=max_file_size,
            cache=cache,
            local_cache_dir=local_cache_dir,
            remote_cache_dir=remote_cache_dir,
            name=name,
            **kwargs,
        )

        return exec_results_to_messages(results)

    return execute


async def python_exec_code_func_in_sandbox(
    code: str, function_name: str, **kwargs
) -> Tuple[Any, list[Content]]:
    """
    Execute a Python function or retrieve a variable from the provided code in a sandbox environment.

    Args:
        code (str): The Python code to execute.
        function_name (str): The name of the function or variable to retrieve from the executed code.
        timeout (int | None, optional): Timeout (in seconds) for the command execution.
        user (str | None, optional): User to execute commands as.
        exec_file_remote (str, optional): The path to the location in the sandbox where the exec file will be copied.
        session_id (Optional[str], optional): Session identifier for the execution context.
        name (Optional[str], optional): The name of the execution context.

    Returns:
        Tuple[list[Content], Any]: A tuple containing a list of messages generated during execution and the result of the function or variable retrieval.
    """
    messages = []
    if code.strip(" \r\n"):
        results = await python_exec_in_sandbox(code=code, **kwargs)
        messages.extend(exec_results_to_messages(results))
    function_call = (
        function_name
        if "(" in function_name and function_name.strip().endswith(")")
        else f"({function_name.strip()})() if hasattr({function_name.strip()}, '__call__') else ({function_name.strip()})"
    )
    results = await python_exec_in_sandbox(code=function_call, **kwargs)
    result = results.get("output")
    if result is not None:
        del results["output"]
    messages.extend(exec_results_to_messages(results))
    return result, messages
