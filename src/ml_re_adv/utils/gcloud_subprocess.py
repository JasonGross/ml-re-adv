import asyncio
import atexit
import json
import logging
import os
import shlex
import socket
from asyncio.subprocess import Process
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence, Tuple, TypeVar, Union, overload

from dotenv import load_dotenv
from inspect_ai.util import ExecResult, subprocess

logger = logging.getLogger(__name__)

T = TypeVar("T", str, bytes)

load_dotenv()

DEFAULT_INSTANCE_NAME = "GCLOUD_INSTANCE"
DEFAULT_ZONE_NAME = "GCLOUD_ZONE"

DEFAULT_TPU_ARGS = ("tpus", "tpu-vm")
DEFAULT_INSTANCE = os.getenv(DEFAULT_INSTANCE_NAME)
DEFAULT_ZONE = os.getenv(DEFAULT_ZONE_NAME)


def cleanup_all_tunnels():
    cleanup_gcloud_tunneling_nowait()


atexit.register(cleanup_all_tunnels)


@overload
# type: ignore
async def gcloud_subprocess(
    args: str | list[str],
    text: Literal[True] = True,
    input: str | bytes | memoryview | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] = {},
    capture_output: bool = True,
    timeout: int | None = None,
    *,
    tpu_args: Sequence[str] = DEFAULT_TPU_ARGS,
    instance: str = DEFAULT_INSTANCE,
    zone: Optional[str] = DEFAULT_ZONE,
) -> ExecResult[str]: ...


@overload
async def gcloud_subprocess(
    args: str | list[str],
    text: Literal[False] = False,
    input: str | bytes | memoryview | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] = {},
    capture_output: bool = True,
    timeout: int | None = None,
    *,
    tpu_args: Sequence[str] = ("tpus", "tpu-vm"),
    instance: str = DEFAULT_INSTANCE,
    zone: Optional[str] = DEFAULT_ZONE,
) -> ExecResult[bytes]: ...


async def gcloud_subprocess(
    args: str | list[str],
    text: bool = True,
    input: str | bytes | memoryview | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] = {},
    capture_output: bool = True,
    timeout: int | None = None,
    *,
    tpu_args: Sequence[str] = ("tpus", "tpu-vm"),
    instance: str = DEFAULT_INSTANCE,
    zone: Optional[str] = DEFAULT_ZONE,
) -> Union[ExecResult[str], ExecResult[bytes]]:
    """Execute and wait for a subprocess on Google Cloud.

    Convenience method for solvers, scorers, and tools to launch
    subprocesses. Automatically enforces a limit on concurrent
    subprocesses (defaulting to os.cpu_count() but controllable
    via the `max_subprocesses` eval config option).

    Args:
       args (str | list[str]): Command and arguments to execute.
       text (bool): Return stdout and stderr as text (defaults to True)
       input (str | bytes | memoryview | None): Optional stdin
          for subprocess.
       cwd (str | Path | None): Switch to directory for execution.
       env (dict[str, str]): Additional environment variables.
       capture_output (bool): Capture stderr and stdout into ExecResult
         (if False, then output is redirected to parent stderr/stdout)
       timeout (int | None): Timeout. If the timeout expires then
         a `TimeoutError` will be raised.
       tpus_args (Sequence[str]): Arguments to pass to `gcloud compute`, for example, `tpus tpus-vm`.
       instance (str): Instance name.
       zone (str | None): Zone name.

    Returns:
       Subprocess result (text or binary depending on `text` param)

    Raises:
       TimeoutError: If the specified `timeout` expires.
    """
    zone_cmd = ["--zone", zone] if zone else []
    command_str = shlex.join(args) if isinstance(args, list) else args
    if cwd is not None:
        command_str = f"mkdir -p {str(cwd)}; cd {str(cwd)}; {command_str}"
    cmd = [
        "gcloud",
        "compute",
        *tpu_args,
        "ssh",
        instance,
        *zone_cmd,
        f"--command={command_str}",
    ]
    logger.debug("Running command: %s", shlex.join(cmd))
    return await subprocess(
        cmd,
        text=text,
        input=input,
        env=env,
        capture_output=capture_output,
        timeout=timeout,
    )


async def gcloud_scp(
    *files: Path | str,
    instance: str = DEFAULT_INSTANCE,
    remote_dir: Path | str = Path("~"),
    create_remote_dir: bool = True,
    tpu_args: Sequence[str] = ("tpus", "tpu-vm"),
    zone: Optional[str] = DEFAULT_ZONE,
) -> Tuple[ExecResult[str], Optional[ExecResult[str]]]:
    """
    Copy files to a Google Cloud instance using scp.

    This function uses `gcloud compute scp` to securely copy files to a specified
    Google Cloud instance. It supports specifying the instance, remote directory,
    TPU arguments, and zone.

    Args:
        files (Path | str): Files to be copied to the remote instance.
        instance (str): The name of the Google Cloud instance. Defaults to `DEFAULT_INSTANCE`.
        remote_dir (Path | str): The remote directory where files will be copied. Defaults to `~/`.
        tpu_args (Sequence[str]): Arguments to pass to `gcloud compute`, for example, `tpus tpus-vm`.
        zone (Optional[str]): The zone of the Google Cloud instance. Defaults to `DEFAULT_ZONE`.

    Returns:
        ExecResult[str]: The result of the SCP command execution.
        Optional[ExecResult[str]]: The result of the remote directory creation command execution.
    """
    zone_cmd = ["--zone", zone] if zone else []
    result_mkdir = None
    if create_remote_dir:
        cmd = [
            "gcloud",
            "compute",
            *tpu_args,
            "ssh",
            instance,
            *zone_cmd,
            f"--command=mkdir -p {str(remote_dir)}",
        ]
        logger.debug("Running command: %s", shlex.join(cmd))
        result_mkdir = await subprocess(cmd, text=True, capture_output=True)
    cmd = [
        "gcloud",
        "compute",
        *tpu_args,
        "scp",
        *map(str, files),
        f"{instance}:{str(Path(remote_dir))}/",
        *zone_cmd,
    ]
    logger.debug("Running command: %s", shlex.join(cmd))
    return await subprocess(cmd, text=True, capture_output=True), result_mkdir


async def get_external_ip(
    *,
    tpu_args: Sequence[str] = ("tpus", "tpu-vm"),
    instance: str = DEFAULT_INSTANCE,
    zone: Optional[str] = DEFAULT_ZONE,
) -> str:
    """
    Get the external IP of a Google Cloud instance.
    """
    zone_cmd = ["--zone", zone] if zone else []
    cmd = [
        "gcloud",
        "compute",
        *(tpu_args if tpu_args else ["instances"]),
        "list",
        "--format=json",
        *zone_cmd,
    ]
    result = await subprocess(cmd, text=True, capture_output=True)
    if not result.success:
        raise RuntimeError(f"Failed to get instances: {result.stderr}")
    try:
        instances_json = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to decode JSON: {result.stdout}") from e

    # Select the instance that matches the GCLOUD_INSTANCE
    gcloud_instance = next(
        (i for i in instances_json if i["name"].split("/")[-1] == instance), None
    )
    if not gcloud_instance:
        raise RuntimeError(
            f"No instance found for {instance} amongst {', '.join(i['name'] for i in instances_json)}"
        )

    # Get internal and external IPs from the instance's JSON
    external_ip = (
        gcloud_instance.get("networkEndpoints", [{}])[0]
        .get("accessConfig", {})
        .get("externalIp", "")
    )

    return external_ip


DOCKER_HOST_PORT = 2375


async def get_docker_host_for(
    instance: str = DEFAULT_INSTANCE,
    zone: Optional[str] = DEFAULT_ZONE,
    *,
    tpu_args: Sequence[str] = ("tpus", "tpu-vm"),
) -> str:
    ip = await get_external_ip(tpu_args=tpu_args, instance=instance, zone=zone)
    return f"tcp://{ip}:{DOCKER_HOST_PORT}"


_tunneling_processes: Dict[Tuple[str, Optional[str]], Tuple[int, Process]] = {}


async def _find_unused_port(start_port: int) -> int:
    port = start_port
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("localhost", port))
            sock.close()
            return port
        except OSError:
            await asyncio.sleep(0)  # Yield control to event loop
        finally:
            sock.close()
        port += 1


async def setup_gcloud_tunneling(
    instance: str = DEFAULT_INSTANCE,
    zone: Optional[str] = DEFAULT_ZONE,
    *,
    tpu_args: Sequence[str] = ("tpus", "tpu-vm"),
    start_port: int = DOCKER_HOST_PORT,
) -> Tuple[int, Process]:
    key = (instance, zone)
    if key in _tunneling_processes:
        port, proc = _tunneling_processes[key]
        if proc.returncode is None:  # Process is still running
            return port, proc
        else:
            del _tunneling_processes[key]  # Remove dead process

    port = await _find_unused_port(start_port)
    zone_cmd = ["--zone", zone] if zone else []
    cmd = [
        "gcloud",
        "compute",
        *tpu_args,
        "ssh",
        instance,
        *zone_cmd,
        "--",
        "-L",
        f"{port}:localhost:{DOCKER_HOST_PORT}",
        "-N",
        # "-f",
    ]
    logger.debug("Running command: %s", shlex.join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    # Wait for the SSH connection to be established
    while True:
        if await is_port_open(port):
            break
        if proc.returncode is not None:
            stderr = None
            if proc.stderr is not None:
                stderr = await proc.stderr.read()
                stderr = stderr.decode()
            raise RuntimeError(f"SSH process exited unexpectedly: {stderr}")
        await asyncio.sleep(0.1)

    _tunneling_processes[key] = (port, proc)
    return (port, proc)


async def is_port_open(port: int) -> bool:
    try:
        _, writer = await asyncio.open_connection("localhost", port)
        writer.close()
        await writer.wait_closed()
        return True
    except (OSError, asyncio.TimeoutError):
        # logger.debug("Port %d is not open yet: %s", port, str(e))
        return False
    except Exception as e:
        logger.error("Unexpected error when checking port %d: %s", port, str(e))
        raise e


def cleanup_gcloud_tunneling_nowait(
    instance: Optional[str] = None, zone: Optional[str] = None
):
    if instance:
        key = (instance, zone)
        if key in _tunneling_processes:
            _port, proc = _tunneling_processes[key]
            if proc.returncode is None:  # Process is still running
                proc.terminate()
    else:
        for key in _tunneling_processes:
            cleanup_gcloud_tunneling_nowait(*key)


async def cleanup_gcloud_tunneling(
    instance: Optional[str] = None, zone: Optional[str] = None
):
    if instance:
        key = (instance, zone)
        if key in _tunneling_processes:
            _port, proc = _tunneling_processes[key]
            if proc.returncode is None:  # Process is still running
                proc.terminate()
                await proc.wait()
            del _tunneling_processes[key]
    else:
        await asyncio.gather(
            *[
                cleanup_gcloud_tunneling(instance, zone)
                for instance, zone in _tunneling_processes
            ]
        )


def gcloud_tunneling_still_active(
    instance: str = DEFAULT_INSTANCE,
    zone: Optional[str] = DEFAULT_ZONE,
    *,
    cleanup: bool = True,
) -> bool:
    key = (instance, zone)
    if key not in _tunneling_processes:
        return False
    port, proc = _tunneling_processes[key]
    if proc.returncode is None:
        return True
    if cleanup:
        logger.info(
            "Removing dead tunneling process for %s (port %d) (stderr: %s) (return code: %s)",
            key,
            port,
            proc.stderr,
            proc.returncode,
        )
        del _tunneling_processes[key]
    return False


def get_gcloud_tunneling_processes() -> Dict[Tuple[str, str], Tuple[int, Process]]:
    active_processes = {}
    for key, (port, proc) in _tunneling_processes.items():
        if gcloud_tunneling_still_active(*key):
            active_processes[key] = (port, proc)
    return active_processes


async def get_gcloud_tunneling_port(
    instance: str = DEFAULT_INSTANCE, zone: Optional[str] = DEFAULT_ZONE
) -> Optional[int]:
    key = (instance, zone)
    if gcloud_tunneling_still_active(*key):
        port, _proc = _tunneling_processes[key]
        return port
    return None
