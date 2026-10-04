"""Tests for the pre-commit data guard (src/tools/check_no_data.py)."""
import subprocess
import sys
from pathlib import Path

GUARD = Path(__file__).resolve().parents[1] / "tools" / "check_no_data.py"
TWO_MB = 2 * 1024 * 1024


def make_repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def stage_file(repo, relative_path, size_bytes):
    target = repo / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"0" * size_bytes)
    subprocess.run(["git", "-C", str(repo), "add", relative_path], check=True)


def run_guard(repo):
    return subprocess.run([sys.executable, str(GUARD)], cwd=repo, capture_output=True, text=True)


def test_two_megabyte_file_under_src_is_blocked(tmp_path):
    repo = make_repo(tmp_path)
    stage_file(repo, "src/fake_large_file.txt", TWO_MB)
    result = run_guard(repo)
    assert result.returncode == 1
    assert "src/fake_large_file.txt" in result.stderr


def test_fixture_under_25_megabytes_is_allowed(tmp_path):
    repo = make_repo(tmp_path)
    stage_file(repo, "src/tests/fixtures/sample_fixture.csv", TWO_MB)
    result = run_guard(repo)
    assert result.returncode == 0, result.stderr


def test_small_results_table_is_allowed(tmp_path):
    repo = make_repo(tmp_path)
    stage_file(repo, "results/practice/table.csv", 200 * 1024)
    result = run_guard(repo)
    assert result.returncode == 0, result.stderr


def test_two_megabyte_results_table_is_blocked(tmp_path):
    repo = make_repo(tmp_path)
    stage_file(repo, "results/practice/table.csv", TWO_MB)
    result = run_guard(repo)
    assert result.returncode == 1
    assert "results/practice/table.csv" in result.stderr
