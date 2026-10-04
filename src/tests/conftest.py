"""Added to the public copy; this file is not in my working repository.

Two tests read things this copy does not carry: test_calfam_command_unchanged reads a file at a commit of the working
repository's history, and test_pod_upload_list_prints_step_9_uploads checks the pod script's upload list against the
working repository's local data folders and results files. Both are skipped here, with that reason. Every other test
runs as written.
"""

import pytest

SKIP = {
    "test_calfam_command_unchanged":
        "reads a file at a commit of the working repository's history, which this copy does not carry",
    "test_pod_upload_list_prints_step_9_uploads":
        "checks the pod script's upload list against the working repository's local data and results files, "
        "which this copy does not carry",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        reason = SKIP.get(item.name)
        if reason is not None and item.path.name == "test_session9_eval.py":
            item.add_marker(pytest.mark.skip(reason=reason))
