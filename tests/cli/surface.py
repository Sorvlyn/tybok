"""CLI surface guard for ``python -m tybok``.

Every other check calls ``create_engine`` directly, so **none of them parses a command line**: a
dropped flag, a renamed ``dest``, a changed default or a broken subcommand dispatch is invisible
to the whole suite and would only surface when someone deploys. This check locks that surface.

  1. ``surface``     the parser fingerprint of every subcommand -- option strings and their order,
                     ``dest``, default, type, choices, nargs, metavar and action kind, plus each
                     ``prog`` and one-line command help -- must equal ``tests/cli/expected.json``.
  2. ``invariants``  the properties that keep the golden meaningful: every ``_Flag`` definition is
                     attached to some command, no command declares a flag twice, every option and
                     command has a non-empty English help, and every command has a handler bound
                     for ``main`` to dispatch to.
  3. ``wiring``      every flag a parent hands to a child is accepted by the child parser it
                     targets. ``serve`` starts its children through the CLI (``python -m tybok
                     worker`` / ``gateway``) while the ``worker`` / ``gateway`` subcommands call
                     those modules in-process, so there are four parent/child pairs and two
                     different child parsers. Nothing else in the suite can see this coupling.
  4. ``smoke``       ``python -m tybok models`` lists what the installation ships -- and, where
                     the inference stack is installed, that set equals the registry's, so a backend
                     that fails to import fails this guard (only a missing third-party dependency
                     is a skip) -- ``--help`` renders for every command, an unknown subcommand exits
                     2, and a missing required option exits 2 naming that option.
  5. ``detects``     the comparison itself: a mutated fingerprint (dropped option, changed
                     default, swapped order, renamed flag, blanked help, dropped command) must be
                     reported, so a green ``surface`` cannot mean "the guard compares nothing".

Both levels run all five jobs; the whole check is seconds of CPU work and needs no GPU, no
compilation and no checkpoint. It is also the one part of the suite that runs on a machine with no
inference stack at all (no numpy / torch / aiohttp): the CLI and the backend registration are lazy
precisely so this guard can gate a bare runner, and ``smoke`` reports which of its two registry
comparisons the environment allows.

An *intentional* CLI change fails ``surface`` on purpose: regenerate the golden and read the diff
it produces::

    python tests/cli/surface.py --update-expected

Usage::

    cd tybok
    python tests/cli/surface.py
    python tests/cli/surface.py --list-jobs
    python tests/cli/surface.py --job-index 0
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import io
import json
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# Allow ``python tests/cli/surface.py`` without ``pip install -e .``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._common import (  # noqa: E402 - after the path bootstrap
    PYTHON,
    REPO_ROOT,
    Column,
    RowResult,
    RowStatus,
    add_level_arguments,
    check_description,
    emit_row,
    finish,
    job_keys,
    jobs_for_level,
    print_job_list,
    print_table,
    run_jobs,
)

__all__ = ["main"]

EXPECTED_PATH = Path(__file__).with_name("expected.json")

COLUMNS = (
    Column("guard", "key", "<"),
    Column("checked", "checked", "<"),
    Column("result", "status", "<"),
)

#: CJK / full-width characters -- help text in this repo is English (see tests/README.md).
_CJK = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")

#: A flag as the parent writes it into a child command line (``--model``, ``--gpu-ipc``, ...).
_FLAG = re.compile(r"^--[a-z0-9][a-z0-9-]*$")


class Kind(str, Enum):
    """The five guards of this check."""

    SURFACE = "surface"
    INVARIANTS = "invariants"
    WIRING = "wiring"
    SMOKE = "smoke"
    DETECTS = "detects"


@dataclass(frozen=True)
class CliJob:
    """One guard, run in its own process like every other job in this suite."""

    kind: Kind

    @property
    def quick(self) -> bool:
        return True

    @property
    def key(self) -> str:
        return self.kind.value


JOBS: tuple[CliJob, ...] = tuple(CliJob(kind) for kind in Kind)


# --------------------------------------------------------------------------- #
# The surface: what the golden locks
# --------------------------------------------------------------------------- #
def _subparsers(parser: argparse.ArgumentParser) -> argparse.Action:
    """The ``add_subparsers`` action, which owns the command -> subparser mapping.

    ``argparse`` has no public introspection API, so this (and the ``_actions`` reads below) reach
    for the documented-in-practice private attributes; a stable golden over them is the point.
    """
    return next(a for a in parser._actions if a.dest == "command" and getattr(a, "choices", None))


def _command_helps(group: argparse.Action) -> dict[str, str]:
    """``{command: one-line help}`` as the top level prints it (the pseudo-actions argparse made)."""
    return {pseudo.dest: (pseudo.help or "") for pseudo in group._choices_actions}


def _action_kind(action: argparse.Action) -> str | None:
    """The recorded ``action``, or ``None`` for argparse's plain ``store``."""
    known = {
        "_StoreAction": None,
        "_StoreTrueAction": "store_true",
        "_StoreFalseAction": "store_false",
        "_CountAction": "count",
        "_AppendAction": "append",
    }
    name = type(action).__name__
    return known[name] if name in known else name.lstrip("_")


def _option(action: argparse.Action) -> dict[str, Any]:
    """One option's surface, with everything argparse derives left out (keeps the golden short)."""
    record: dict[str, Any] = {"flags": list(action.option_strings)}
    if not action.option_strings:  # a positional argument
        record["positional"] = action.dest
    if action.required:
        record["required"] = True
    if action.default is not None and not (action.nargs == 0 and action.default is False):
        record["default"] = action.default
    if action.type is not None:
        record["type"] = getattr(action.type, "__name__", str(action.type))
    if action.choices is not None:
        record["choices"] = list(action.choices)
    if action.nargs is not None and action.nargs != 0:
        record["nargs"] = action.nargs
    if action.metavar is not None:
        record["metavar"] = action.metavar
    kind = _action_kind(action)
    if kind is not None:
        record["action"] = kind
    derived = action.option_strings[0].lstrip("-").replace("-", "_") if action.option_strings else action.dest
    if action.dest != derived:
        record["dest"] = action.dest
    return record


def _fingerprint_of(parser: argparse.ArgumentParser) -> dict[str, Any]:
    """The observable surface of ``parser``, in the shape the golden stores."""
    group = _subparsers(parser)
    helps = _command_helps(group)
    return {
        "prog": parser.prog,
        "description": parser.description or "",
        "commands": list(group.choices),
        "subcommands": {
            name: {
                "prog": sub.prog,
                "help": helps.get(name, ""),
                "options": [_option(action) for action in sub._actions if action.dest != "help"],
            }
            for name, sub in group.choices.items()
        },
    }


def _fingerprint() -> dict[str, Any]:
    """The live CLI's surface."""
    from tybok.__main__ import _build_parser

    return _fingerprint_of(_build_parser())


def _compare_commands(name: str, want: dict[str, Any], got: dict[str, Any]) -> list[str]:
    """Differences for one command: prog, help, then its options."""
    problems: list[str] = []
    for field in ("prog", "help"):
        if want.get(field) != got.get(field):
            problems.append(f"{name}: {field} {want.get(field)!r} -> {got.get(field)!r}")

    want_options, got_options = want.get("options", []), got.get("options", [])
    want_flags = [option.get("flags") for option in want_options]
    got_flags = [option.get("flags") for option in got_options]
    if want_flags != got_flags:
        if sorted(map(repr, want_flags)) == sorted(map(repr, got_flags)):
            return problems + [f"{name}: the same options came in a different order"]
        for flags in want_flags:
            if flags not in got_flags:
                problems.append(f"{name}: option {flags} removed")
        for flags in got_flags:
            if flags not in want_flags:
                problems.append(f"{name}: option {flags} added")
        return problems

    for want_option, got_option in zip(want_options, got_options):
        if want_option != got_option:
            problems.append(
                f"{name}: option {want_option.get('flags')} changed: "
                f"{json.dumps(want_option)} -> {json.dumps(got_option)}"
            )
    return problems


def _compare(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Differences between two fingerprints, as readable lines (empty means equal)."""
    problems: list[str] = []
    for field in ("prog", "description"):
        if expected.get(field) != actual.get(field):
            problems.append(f"{field}: {expected.get(field)!r} -> {actual.get(field)!r}")
    if expected.get("commands") != actual.get("commands"):
        problems.append(f"commands: {expected.get('commands')} -> {actual.get('commands')}")

    want_subs, got_subs = expected.get("subcommands", {}), actual.get("subcommands", {})
    for name in dict.fromkeys([*want_subs, *got_subs]):
        if name not in got_subs:
            problems.append(f"{name}: command removed")
        elif name not in want_subs:
            problems.append(f"{name}: command added")
        else:
            problems += _compare_commands(name, want_subs[name], got_subs[name])
    return problems


def _dump_expected(fingerprint: dict[str, Any]) -> str:
    """The golden, formatted so one option is one line (a diff then shows exactly what moved)."""
    lines = [
        "{",
        '    "generated_by": "python tests/cli/surface.py --update-expected",',
        f'    "prog": {json.dumps(fingerprint["prog"])},',
        f'    "description": {json.dumps(fingerprint["description"])},',
        f'    "commands": {json.dumps(fingerprint["commands"])},',
        '    "subcommands": {',
    ]
    blocks: list[str] = []
    for name, command in fingerprint["subcommands"].items():
        block = [
            f"        {json.dumps(name)}: {{",
            f'            "prog": {json.dumps(command["prog"])},',
            f'            "help": {json.dumps(command["help"])},',
        ]
        options = command["options"]
        if options:
            body = ",\n".join(f"                {json.dumps(option)}" for option in options)
            block += ['            "options": [', body, "            ]"]
        else:
            block.append('            "options": []')
        block.append("        }")
        blocks.append("\n".join(block))
    lines += [",\n".join(blocks), "    }", "}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Guard 1: the surface
# --------------------------------------------------------------------------- #
def _run_surface() -> tuple[str, list[str]]:
    """The live fingerprint must equal the golden."""
    if not EXPECTED_PATH.exists():
        return "no golden", [f"{EXPECTED_PATH.name} is missing: run `python tests/cli/surface.py --update-expected`"]
    try:
        expected = json.loads(EXPECTED_PATH.read_text())
    except json.JSONDecodeError as error:
        return "unreadable golden", [
            f"{EXPECTED_PATH.name} is not valid JSON ({error}): regenerate it with "
            "`python tests/cli/surface.py --update-expected`"
        ]
    actual = _fingerprint()
    counted = sum(len(command["options"]) for command in actual["subcommands"].values())
    problems = _compare(expected, actual)
    if problems:
        return f"{counted} options", problems
    return f"{counted} options in {len(actual['commands'])} commands", []


# --------------------------------------------------------------------------- #
# Guard 2: the invariants that keep the golden meaningful
# --------------------------------------------------------------------------- #
def _run_invariants() -> tuple[str, list[str]]:
    """No dead definition, no duplicate flag, no non-English help, handler bound.

    Missing help text is counted, not failed: the CLI shipped with a dozen bare ``--socket`` /
    ``--device`` / ``--compile`` style flags, so requiring help here would turn this guard into an
    unrelated documentation rewrite. The count keeps the state visible.
    """
    from tybok import __main__ as cli

    failures: list[str] = []
    parser = cli._build_parser()
    group = _subparsers(parser)
    helps = _command_helps(group)
    without_help = 0

    declared = [value for value in vars(cli).values() if isinstance(value, cli._Flag)]
    attached = {id(option) for command in cli._COMMANDS.values() for option in command.options}
    for flag in declared:
        if id(flag) not in attached:
            failures.append(f"{flag.flags[0]}: defined but attached to no command")

    for name, sub in group.choices.items():
        spellings = [spelling for action in sub._actions if action.dest != "help" for spelling in action.option_strings]
        for spelling, count in Counter(spellings).items():
            if count > 1:
                failures.append(f"{name}: {spelling} is declared {count} times")
        for action in sub._actions:
            if action.dest == "help":
                continue
            label = action.option_strings[0] if action.option_strings else action.dest
            if not (action.help or "").strip():
                without_help += 1
            elif _CJK.search(action.help) or _CJK.search(str(action.metavar or "")):
                failures.append(f"{name}: {label} has non-English help text")
        if sub.get_default("func") is None:
            failures.append(f"{name}: no handler bound, so main() cannot dispatch it")
        if _CJK.search(helps.get(name, "")):
            failures.append(f"{name}: the command help is not English")

    for name in cli._COMMANDS:
        if name not in group.choices:
            failures.append(f"{name}: declared in _COMMANDS but has no subparser")
    return (
        f"{len(declared)} definitions, {len(group.choices)} commands, {without_help} options without help",
        failures,
    )


# --------------------------------------------------------------------------- #
# Guard 3: what the parent hands the child
# --------------------------------------------------------------------------- #
def _assigned_value(function: ast.FunctionDef, variable: str) -> ast.AST | None:
    """The value assigned to ``variable`` inside ``function`` (a child command line)."""
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == variable for target in node.targets
        ):
            return node.value
    return None


def _call_argument(function: ast.FunctionDef, callee: str) -> ast.AST | None:
    """The first argument of the ``callee(...)`` call inside ``function`` (its argv list)."""
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == callee and node.args:
            return node.args[0]
    return None


def _flag_literals(node: ast.AST) -> set[str]:
    """Every ``--flag`` string literal inside ``node``."""
    return {
        item.value
        for item in ast.walk(node)
        if isinstance(item, ast.Constant) and isinstance(item.value, str) and _FLAG.match(item.value)
    }


def _parent_flags() -> dict[tuple[str, str], set[str]]:
    """``{(parent command, child command): flags the parent hands it}``, read from the parent."""
    source = (REPO_ROOT / "tybok" / "__main__.py").read_text()
    functions = {n.name: n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)}
    found: dict[tuple[str, str], set[str]] = {}

    serve = functions.get("_cmd_serve")
    if serve is not None:
        for variable, child in (("worker_cmd", "worker"), ("gateway_cmd", "gateway")):
            node = _assigned_value(serve, variable)
            if node is not None:
                found[("serve", child)] = _flag_literals(node)

    for function_name, callee, child in (
        ("_import_worker", "worker_main", "worker"),
        ("_import_gateway", "gateway_main", "gateway"),
    ):
        function = functions.get(function_name)
        if function is None:
            continue
        node = _call_argument(function, callee)
        if node is not None:
            found[(child, child)] = _flag_literals(node)
    return found


def _module_option_strings(module: str, entry: str) -> set[str]:
    """The option strings of ``tybok.worker`` / ``tybok.gateway``, captured as its main() parses.

    Those modules build their own parser inside ``main``; calling it with ``--help`` reaches
    ``parse_args`` and exits, so intercepting ``parse_args`` is how a caller can see it.
    """
    captured: list[argparse.ArgumentParser] = []
    original = argparse.ArgumentParser.parse_args

    def spy(parser: argparse.ArgumentParser, args: Any = None, namespace: Any = None) -> Any:
        captured.append(parser)
        return original(parser, args, namespace)

    argparse.ArgumentParser.parse_args = spy  # type: ignore[method-assign]
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                getattr(importlib.import_module(module), entry)(["--help"])
            except SystemExit:
                pass
    finally:
        argparse.ArgumentParser.parse_args = original  # type: ignore[method-assign]
    if not captured:
        raise RuntimeError(f"{module}.{entry} never reached parse_args")
    return {s for action in captured[-1]._actions for s in action.option_strings}


def _run_wiring() -> tuple[str, list[str]]:
    """Every flag the parent writes for a child must be accepted by that child's parser."""
    failures: list[str] = []
    from tybok.__main__ import _build_parser

    group = _subparsers(_build_parser())
    cli_strings = {
        name: {s for action in sub._actions for s in action.option_strings} for name, sub in group.choices.items()
    }
    module_strings = {
        "worker": _module_option_strings("tybok.worker", "main"),
        "gateway": _module_option_strings("tybok.gateway", "main"),
    }
    handed = _parent_flags()

    # serve starts its children through the CLI; the worker/gateway subcommands call the modules.
    targets = (
        ("serve", "worker", cli_strings["worker"]),
        ("serve", "gateway", cli_strings["gateway"]),
        ("worker", "worker", module_strings["worker"]),
        ("gateway", "gateway", module_strings["gateway"]),
    )
    for parent, child, accepted in targets:
        flags = handed.get((parent, child))
        if flags is None:
            failures.append(
                f"{parent} -> {child}: could not locate the child command line in "
                f"tybok/__main__.py (this check needs updating)"
            )
            continue
        for spelling in sorted(flags - accepted):
            failures.append(f"{parent} -> {child}: {spelling} is not accepted by the child parser")
    return f"{len(targets)} parent/child command lines", failures


# --------------------------------------------------------------------------- #
# Guard 4: the CLI runs
# --------------------------------------------------------------------------- #
def _tail(text: str, limit: int = 200) -> str:
    """The last non-empty line, trimmed (a CLI failure usually explains itself there)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][-limit:] if lines else ""


def _run_smoke() -> tuple[str, list[str]]:
    """``models``, ``--help``, an unknown subcommand and a missing required option."""
    from tybok.__main__ import _build_parser
    from tybok.registry import available, shipped_models

    failures: list[str] = []

    def run(*argv: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [PYTHON, "-m", "tybok", *argv],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )

    listed = run("models")
    if listed.returncode != 0:
        failures.append(f"`tybok models` exited {listed.returncode}: {_tail(listed.stderr)}")
    else:
        shown = {
            line.strip().lstrip("-").strip() for line in listed.stdout.splitlines() if line.strip().startswith("-")
        }
        shipped = set(shipped_models())
        if shown != shipped:
            failures.append(f"`tybok models` printed {sorted(shown)}, the installation ships {sorted(shipped)}")

    # The listing is deliberately dependency-free (it names what is shipped, not what loaded),
    # so compare it against the registry only where the backends can actually be imported: there
    # every shipped backend must register, under exactly its own key. Only a *missing third-party*
    # dependency is an excuse to skip -- a backend that cannot import for any other reason (a
    # renamed symbol, a missing module of our own) is a regression, and this guard is the only
    # place in the suite that imports all backends, so it has to say so rather than skip.
    registry_check = "not loadable"
    try:
        registered = set(available())
    except ImportError as error:
        absent = getattr(error, "name", "") or ""
        if isinstance(error, ModuleNotFoundError) and not absent.startswith("tybok"):
            registry_check = f"skipped, backends not importable ({error})"
        else:
            registry_check = "failed"
            failures.append(f"the backends do not import: {error}")
    else:
        registry_check = "matches"
        if registered != set(shipped_models()):
            failures.append(
                f"the registry holds {sorted(registered)}, the installation ships {sorted(shipped_models())}"
            )

    commands = list(_subparsers(_build_parser()).choices)
    for command in commands:
        helped = run(command, "--help")
        if helped.returncode != 0 or "usage:" not in helped.stdout:
            failures.append(f"`tybok {command} --help` exited {helped.returncode} without a usage")

    unknown = run("bogus")
    if unknown.returncode != 2:
        failures.append(f"an unknown subcommand exited {unknown.returncode}, expected 2")

    missing = run("worker")
    if missing.returncode != 2 or "--model" not in missing.stderr + missing.stdout:
        failures.append(
            f"`tybok worker` without --model exited {missing.returncode} without naming the "
            f"option: {_tail(missing.stderr)}"
        )

    return (
        f"{len(commands)} commands, {len(commands) + 3} invocations, registry {registry_check}",
        failures,
    )


# --------------------------------------------------------------------------- #
# Guard 5: the comparison itself is not vacuous
# --------------------------------------------------------------------------- #
def _synthetic_fingerprint() -> dict[str, Any]:
    """A miniature CLI with the real one's shape, to prove ``_compare`` reports changes."""
    parser = argparse.ArgumentParser(prog="synthetic", description="synthetic CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker", help="start the worker")
    worker.add_argument("--model", required=True, help="checkpoint dir")
    worker.add_argument("--device", default="auto", help="torch device")
    worker.add_argument("--graph", action="store_true", help="capture a CUDA graph")
    worker.set_defaults(func=lambda args: None)
    models = sub.add_parser("models", help="list backends")
    models.set_defaults(func=lambda args: None)
    return _fingerprint_of(parser)


def _run_detects() -> tuple[str, list[str]]:
    """Each mutation of the fingerprint must come back as a reported difference."""
    failures: list[str] = []
    base = _synthetic_fingerprint()
    mutations: dict[str, Callable[[dict[str, Any]], None]] = {
        "a dropped option": lambda f: f["subcommands"]["worker"]["options"].pop(1),
        "a changed default": lambda f: f["subcommands"]["worker"]["options"][1].update(default="cpu"),
        "a swapped option order": lambda f: f["subcommands"]["worker"]["options"].reverse(),
        "a renamed flag": lambda f: f["subcommands"]["worker"]["options"][0].update(flags=["--model-file"]),
        "a blanked command help": lambda f: f["subcommands"]["worker"].update(help=""),
        "a dropped command": lambda f: f["subcommands"].pop("models"),
        "a changed description": lambda f: f.update(description="something else"),
    }
    for label, mutate in mutations.items():
        changed = deepcopy(base)
        mutate(changed)
        if not _compare(changed, base):
            failures.append(f"{label} was not reported")
    if _compare(base, deepcopy(base)):
        failures.append("an unchanged fingerprint was reported as different")
    return f"{len(mutations)} mutations, all reported", failures


_RUNNERS: dict[Kind, Callable[[], tuple[str, list[str]]]] = {
    Kind.SURFACE: _run_surface,
    Kind.INVARIANTS: _run_invariants,
    Kind.WIRING: _run_wiring,
    Kind.SMOKE: _run_smoke,
    Kind.DETECTS: _run_detects,
}


def _run_job(job: CliJob) -> RowResult:
    """Child side: run one guard."""
    checked, failures = _RUNNERS[job.kind]()
    metrics: dict[str, Any] = {"checked": checked, "problems": len(failures)}
    if failures:
        return RowResult(job.key, RowStatus.FAIL, "; ".join(failures), job={"guard": job.key}, metrics=metrics)
    return RowResult(job.key, RowStatus.PASS, job={"guard": job.key}, metrics=metrics)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=check_description(__doc__))
    add_level_arguments(parser)
    parser.add_argument(
        "--update-expected",
        action="store_true",
        help="rewrite tests/cli/expected.json from the current CLI, then exit",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI surface guard (driver), or one job of it (``--job-index``)."""
    args = _parse_args(argv)
    if args.update_expected:
        fingerprint = _fingerprint()
        text = _dump_expected(fingerprint)
        json.loads(text)  # the golden must parse before it is written
        EXPECTED_PATH.write_text(text)
        counted = sum(len(command["options"]) for command in fingerprint["subcommands"].values())
        print(f"wrote {EXPECTED_PATH} ({counted} options); review the diff")
        return 0

    jobs = jobs_for_level(JOBS, args.level)
    if args.list_jobs:
        print_job_list(job_keys(jobs))
        return 0
    if args.job_index is not None:
        emit_row(_run_job(jobs[args.job_index]))
        return 0

    print(f"\n=== tybok CLI surface guard ({args.level.value} level) ===")
    print("no GPU, no compilation, no checkpoint: the command line of `python -m tybok`")
    rows = run_jobs(Path(__file__), job_keys(jobs), ["--level", args.level.value])
    print()
    print_table(COLUMNS, rows)
    print()
    print("NOTE: the regression checks never parse a command line -- this is the only guard on the")
    print("      CLI surface, and `detects` proves the comparison is not vacuous.")
    return finish(rows, pass_message="cli: surface, invariants, wiring and smoke all held")


if __name__ == "__main__":
    sys.exit(main())
