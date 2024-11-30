# %%
from functools import cache
from pathlib import Path

path = Path(__file__).parent

DOCKERFILE_PATH = path / "Dockerfile"
DOCKER_COMPOSE_PATH = path / "docker-compose.yml"


# %%
@cache
def get_installed_pip_packages(dockerfile: str | Path = DOCKERFILE_PATH) -> set[str]:
    """Get the list of installed pip packages from a Dockerfile.

    Args:
      dockerfile (str | Path): The path to the Dockerfile.

    Returns:
      A list of installed pip packages.
    """
    contents = Path(dockerfile).read_text(encoding="utf-8").split("\n")
    contents = [line for line in contents if line.startswith("RUN pip install")]
    contents = [line.split("RUN pip install ")[1] for line in contents]
    contents = [line.split(" &&")[0] for line in contents]
    packages = set(
        pkg.strip()
        for line in contents
        for pkg in line.split(" ")
        if pkg.strip() and not pkg.strip().startswith("--")
    )
    return packages


# %%
