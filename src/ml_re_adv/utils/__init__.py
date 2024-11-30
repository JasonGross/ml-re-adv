# %%
import os
from pathlib import Path
from typing import Callable, Optional

package_dir = Path(__file__).parent.parent
# %%


def getenv_or_input(variable: str, input: Callable[[str], str] = input) -> str:
    result = os.getenv(variable)
    while not result:
        result = input(variable)
    os.environ[variable] = result
    return result


def walk_to_dict(
    *dirs: Path,
    target_directory: Optional[Path] = None,
    is_text: Callable[[Path], bool] = lambda p: False,
    include_unknown: bool = False,
    read_files: bool = True,
) -> dict[Path, str | bytes | None | Path]:
    """
    Walk through a directory and return a dictionary of the files and directories.
    """
    results: dict[Path, str | bytes | None | Path] = {}
    for dirpath in dirs:
        for p in dirpath.iterdir():
            target = (
                target_directory / p.relative_to(dirpath) if target_directory else p
            )
            if p.is_dir():
                results |= walk_to_dict(
                    p,
                    target_directory=target,
                    is_text=is_text,
                    include_unknown=include_unknown,
                    read_files=read_files,
                )
            elif p.is_file():
                if read_files:
                    if is_text(p):
                        results[target] = p.read_text()
                    else:
                        results[target] = p.read_bytes()
                else:
                    results[target] = p
            elif include_unknown:
                results[target] = None
    return results
