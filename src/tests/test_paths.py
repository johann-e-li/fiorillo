"""Tests for src/tools/paths.py: the restricted root must be set and must lie outside the project."""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC.parent))

from src.tools import paths  # noqa: E402


def test_unset_variable_raises(monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    with pytest.raises(paths.RestrictedRootNotSet):
        paths.restricted_path("data")


def test_root_inside_project_raises(monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(paths.PROJECT_ROOT / "data" / "restricted"))
    with pytest.raises(paths.RestrictedRootInsideProject):
        paths.restricted_path("data")


def test_root_outside_project_returns_joined_path(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    assert paths.restricted_path("data", "x") == tmp_path.resolve() / "data" / "x"
