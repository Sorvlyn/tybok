"""Shared plumbing for the checks under ``tests/``.

Every check is a standalone script that owns its own model-specific work; this package owns
only what all of them need:

``level``            the ``quick`` / ``full`` split
``rows``             the child-process -> driver result protocol (one ``ROW`` JSON line)
``process``          spawning one process per job, echoing output, collecting rows
``cli``              the shared command line surface (``--level`` / ``--device`` / ...)
                     and the checkpoint lookup (``TYBOK_CHECKPOINT_<MODEL>`` / ``checkpoints.env``)
``report``           declarative tables, the final ``RESULT:`` line, the exit code
``overlap_matrix``   the 4-mode overlap matrix driver shared by pi05 and smolvla

Run ``python tests/<model>/<check>.py --help`` for the flags of a single check, and
``--list-jobs`` to see exactly which jobs the current ``--level`` will run.
"""

from __future__ import annotations

from .cli import (
    CHECKPOINT_ENV_PREFIX,
    ENV_FILE,
    ModelSpec,
    add_engine_arguments,
    add_level_arguments,
    check_description,
    checkpoint_from_env,
    cuda_available,
    job_keys,
    load_env_file,
    missing_environment,
    print_job_list,
)
from .level import Job, Level, jobs_for_level
from .process import PYTHON, REPO_ROOT, ChildResult, format_command, run_child, run_jobs
from .report import Column, finish, print_table, skip
from .rows import (
    RESULT_PREFIX,
    ROW_PREFIX,
    RowResult,
    RowStatus,
    emit_row,
    parse_result_line,
    parse_row_line,
)

__all__ = [
    "CHECKPOINT_ENV_PREFIX",
    "Column",
    "ChildResult",
    "ENV_FILE",
    "Job",
    "Level",
    "ModelSpec",
    "PYTHON",
    "REPO_ROOT",
    "ROW_PREFIX",
    "RESULT_PREFIX",
    "RowResult",
    "RowStatus",
    "add_engine_arguments",
    "add_level_arguments",
    "check_description",
    "checkpoint_from_env",
    "cuda_available",
    "emit_row",
    "finish",
    "format_command",
    "job_keys",
    "jobs_for_level",
    "load_env_file",
    "missing_environment",
    "parse_result_line",
    "parse_row_line",
    "print_job_list",
    "print_table",
    "run_child",
    "run_jobs",
    "skip",
]
