"""Resolve project locations from src/configs/paths.yaml.

Public paths live inside this repository. Restricted paths (MIMIC, HCUP, and anything trained on them)
live outside it, at the folder named by the EDDM_RESTRICTED_ROOT environment variable. That variable is set
only in your own Terminal, so a script started by an automated agent, which lacks it, fails here with a
clear message instead of finding restricted data.

    from src.tools.paths import public_path, restricted_path
    public_path("results", "phase2")          # <repo>/results/phase2
    restricted_path("data", "mimic-iv-ed")    # $EDDM_RESTRICTED_ROOT/data/mimic-iv-ed
"""
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "src" / "configs" / "paths.yaml"
ENV_VAR = "EDDM_RESTRICTED_ROOT"


class RestrictedRootNotSet(RuntimeError):
    """Raised when a restricted path is requested and EDDM_RESTRICTED_ROOT is not set."""


class RestrictedRootInsideProject(RuntimeError):
    """Raised when EDDM_RESTRICTED_ROOT points inside this repository, where agents and git can reach it."""


def _config():
    with open(CONFIG_FILE, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def public_path(*parts):
    """A location inside this repository."""
    return (PROJECT_ROOT / _config()["public_root"]).resolve().joinpath(*parts)


def restricted_path(*parts):
    """A location in the restricted folder outside this repository."""
    if not os.environ.get(ENV_VAR):
        raise RestrictedRootNotSet(
            f"{ENV_VAR} is not set. Restricted data (MIMIC, HCUP) and models trained on it live outside this "
            f"project, in the folder named by {ENV_VAR}. It is set in ~/.zshrc on the project Mac and is "
            "available only in your own Terminal. Automated agents never receive it."
        )
    root = Path(os.path.expandvars(_config()["restricted_root"])).expanduser().resolve()
    if root == PROJECT_ROOT or PROJECT_ROOT in root.parents:
        raise RestrictedRootInsideProject(
            f"{ENV_VAR} points to {root}, which is inside the project folder {PROJECT_ROOT}. Restricted data "
            "must live outside the project, where no agent, git, or sync tool can reach it. Move it and reset "
            f"{ENV_VAR}."
        )
    return root.joinpath(*parts)


if __name__ == "__main__":
    print(f"public root: {public_path()}")
    try:
        print(f"restricted root: {restricted_path()}")
    except (RestrictedRootNotSet, RestrictedRootInsideProject) as error:
        print(f"restricted root: not available ({error})")
