"""Geometry table: declares each fused kernel's geometry per arch and reconciles it against the
set actually compiled into the binary.

Geometry is compile-time: shape constants are ``constexpr`` and GEMM tile parameters are
template arguments, so "changing geometry" = **editing sources + recompiling**, not a runtime
switch. This table can only pin down the geometry of the current build: after recompiling with
changed macros, run ``check()`` and any mismatch between the table and the binary reports
**which field** differs. The self-report is provided by the kernel itself
(``<kernel>_geom()``), and instantiation and self-report use the same set of macros (two-level
stringization), so the self-report cannot come from a different set than what was actually
compiled in.

The numbers in the table are **declarations** (not copied from the binary each time):
``--bootstrap`` is only for one-time bootstrapping; to add an arch, add an entry to
``GEOMETRY``, and if there is no entry it honestly reports "no such arch in the table".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KernelGeometry:
    """Geometry of one kernel: shape constants + tile parameters for each GEMM phase."""

    shapes: dict[str, int]
    tiles: dict[str, tuple[int, ...]]


# ---- Geometry table (**declared** values, reconciled with the binary via `check()`) ----
GEOMETRY: dict[tuple[int, int], dict[str, KernelGeometry]] = {
    (8, 9): {
        'tmt5.ffn': KernelGeometry(
            shapes={'M': 128, 'H': 4096, 'F': 10240, 'NT': 256, 'SMEM': 98304, 'GRID': 66},
            tiles={
                'F_PHASE_2': (128, 64, 128, 4, 4, 2, 4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0),
                'F_PHASE_4': (128, 64, 128, 4, 4, 2, 4, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 2),
            },
        ),
        'tmt5.attn': KernelGeometry(
            shapes={'M': 128, 'S': 128, 'H': 4096, 'NH': 64, 'HD': 64, 'NT': 256, 'SMEM': 98304, 'GRID': 66},
            tiles={
                'S_PHASE_2': (128, 64, 128, 4, 4, 2, 4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0),
                'S_PHASE_5': (128, 64, 128, 4, 4, 2, 4, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 2),
            },
        ),
        'adit.attn_self': KernelGeometry(
            shapes={'M': 32, 'H': 1024, 'N': 9216, 'H3': 3072, 'HEADS': 24, 'BN_QKV': 96, 'BN_O': 32, 'BK_Q': 64, 'BK_O': 128, 'STAGES_Q': 3, 'STAGES_O': 5, 'NT': 256, 'GRID': 96},
            tiles={
            },
        ),
        'adit.attn_cross': KernelGeometry(
            shapes={'M': 32, 'H': 1024, 'NQ': 3072, 'NO': 1024, 'HEADS': 24, 'BN_A': 64, 'BK_A': 128, 'STAGES': 3, 'NT': 256, 'GRID': 48},
            tiles={
            },
        ),
        'adit.ffn': KernelGeometry(
            shapes={'M': 32, 'H': 1024, 'F': 4096, 'BNU': 64, 'BN1': 32, 'BKU': 128, 'BKD': 128, 'STAGES': 3, 'NT': 256, 'GRID0': 64, 'SLAB1': 32},
            tiles={
            },
        ),
        'vdit.attn_self': KernelGeometry(
            shapes={'M': 120, 'H': 3072, 'N': 9216, 'NH': 24, 'D': 128, 'NT': 256, 'SMEM': 49152, 'GRID': 132},
            tiles={
                'S_PHASE_2': (64, 64, 128, 3, 3, 4, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0),
                'S_PHASE_6': (64, 64, 128, 3, 3, 4, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1),
            },
        ),
        'vdit.attn_cross': KernelGeometry(
            shapes={'M': 120, 'C': 129, 'H': 3072, 'NKV': 6144, 'NT': 256, 'SMEM': 49152, 'GRID': 132},
            tiles={
                'C_PHASE_2_2G': (64, 64, 128, 3, 3, 4, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0),
                'C_PHASE_6': (64, 64, 128, 3, 3, 4, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1),
            },
        ),
        'vdit.ffn': KernelGeometry(
            shapes={'M': 120, 'H': 3072, 'F': 14336, 'NT': 256, 'SMEM': 49152, 'GRID': 132},
            tiles={
                'F_PHASE_2': (64, 64, 128, 3, 3, 4, 2, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0),
                'F_PHASE_4': (64, 48, 128, 3, 3, 4, 2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 2, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1),
            },
        ),
    },
}  # fmt: skip


@dataclass(frozen=True)
class Reporter:
    """Self-report entry point for one kernel.

    ``py`` is the Python-side call name (the bindings' ``m.def``), and ``csym`` is the
    ``extern "C"`` symbol name in ``.cu/.cpp`` (used for source scanning); the two are **not
    necessarily the same** (e.g. vdit.ffn's py name ``ffn_geom`` and C symbol ``vdit_ffn_geom``).
    """

    ext: str
    py: str
    csym: str


# each kernel -> the extension name + function name for its self-report (matching m.def / extern "C" in the bindings)
# Geometry table: declared values reconciled against the binary; the literal carries '# fmt: skip'.
_REPORTERS: dict[str, Reporter] = {
    "tmt5.ffn":          Reporter("tmt5", "ffn_geom", "tmt5_ffn_geom"),
    "tmt5.attn":         Reporter("tmt5", "attn_geom", "tmt5_attn_geom"),
    "adit.attn_self":  Reporter("action", "attn_self_geom", "adit_attn_self_geom"),
    "adit.attn_cross": Reporter("action", "adit_attn_cross_geom", "adit_attn_cross_geom"),
    "adit.ffn":        Reporter("action", "ffn_geom", "adit_ffn_geom"),
    "vdit.attn_self":  Reporter("video", "attn_self_geom", "vdit_attn_self_geom"),
    "vdit.attn_cross": Reporter("video", "attn_cross_geom", "vdit_attn_cross_geom"),
    "vdit.ffn":        Reporter("video", "ffn_geom", "vdit_ffn_geom"),
}  # fmt: skip


def _parse(report: str) -> KernelGeometry:
    """Parse a self-report of the form `<k>=<v>,... | <phase>=<tok>, <tok>, ...`."""
    shapes: dict[str, int] = {}
    tiles: dict[str, tuple[int, ...]] = {}
    for part in report.split(" | "):
        name, _, value = part.partition("=")
        name, value = name.strip(), value.strip()
        if "=" in value:
            # shape group: "M=128,H=4096,..." (comma-separated k=v). name was split off by
            # partition and must be reassembled; the shape group also contains commas, so the two
            # kinds of part cannot be told apart by "has a comma or not"
            for kv in (name + "=" + value).split(","):
                k, _, v = kv.partition("=")
                shapes[k.strip()] = int(v)
        elif "," in value:  # tile list (positional parameters, no names)
            tiles[name] = tuple(int(t) for t in value.split(",") if t.strip())
        else:  # just a single shape constant
            shapes[name] = int(value)
    return KernelGeometry(shapes=shapes, tiles=tiles)


def reported(kernel: str, device: Any = None) -> KernelGeometry:
    """Read the kernel's geometry from the **binary** (calls `<kernel>_geom()`)."""
    from . import registry as reg

    rep = _REPORTERS[kernel]
    ext = reg._loader(rep.ext)()  # noqa: SLF001 - same loader cache
    return _parse(getattr(ext, rep.py)())


def declared(kernel: str, arch: tuple[int, int] | None = None) -> KernelGeometry | None:
    """Read the geometry for this arch from the **table**; returns None if the arch is absent (honestly reports "not declared")."""
    arch = _arch() if arch is None else arch
    return GEOMETRY.get(arch, {}).get(kernel)


def _arch(device: Any = None) -> tuple[int, int]:
    import torch

    # (major, minor); torch returns a hashable 2-tuple and the value is used as the table key.
    return torch.cuda.get_device_capability(device)


def check(device: Any = None) -> list[str]:
    """Compare "table" against "binary" field by field. Returns the list of problems (empty = all green)."""
    arch = _arch(device)
    table = GEOMETRY.get(arch)
    if table is None:
        return [
            f"no sm_{arch[0]}{arch[1]} in the geometry table (declared: "
            + ", ".join(f"sm_{a}{b}" for a, b in GEOMETRY)
            + ")"
        ]
    problems: list[str] = []
    for kernel in _REPORTERS:
        got = reported(kernel, device)
        want = table.get(kernel)
        if want is None:
            problems.append(f"{kernel}: no entry in the table (the binary reports {got.shapes})")
            continue
        for k in sorted(set(got.shapes) | set(want.shapes)):
            g, w = got.shapes.get(k), want.shapes.get(k)
            if g != w:
                problems.append(f"{kernel}.{k}: table={w} binary={g}")
        for k in sorted(set(got.tiles) | set(want.tiles)):
            g, w = got.tiles.get(k), want.tiles.get(k)
            if g != w:
                problems.append(f"{kernel}.{k}: table={_fmt(w)} binary={_fmt(g)}")
    return problems


def _fmt(t: tuple[int, ...] | None) -> str:
    return "missing" if t is None else "(" + ", ".join(str(x) for x in t) + ")"


def geometry_plan(device: Any = None) -> str:
    """Print the geometry table and the reconciliation status of each kernel."""
    arch = _arch(device)
    lines = [f"geometry table sm_{arch[0]}{arch[1]} ({len(_REPORTERS)} kernels)"]
    for kernel in _REPORTERS:
        d = declared(kernel, arch)
        r = reported(kernel, device)
        mark = "OK " if d and d == r else "**MISMATCH**" if d else "not in table"
        lines.append(f"  [{mark}] {kernel}")
        lines.append(f"       shapes {r.shapes}")
        for name, toks in r.tiles.items():
            lines.append(f"       {name:<5} {toks}")
    return "\n".join(lines)


def bootstrap() -> str:
    """Generate the table literal from the current binary (for one-time bootstrapping; afterwards the table is the declaration, kept consistent via ``check()``)."""
    out = ["GEOMETRY: dict[tuple[int, int], dict[str, KernelGeometry]] = {", f"    {_arch()!r}: {{"]
    for kernel in _REPORTERS:
        g = reported(kernel)
        out.append(f"        {kernel!r}: KernelGeometry(")
        out.append(f"            shapes={g.shapes!r},")
        out.append("            tiles={")
        for name, toks in g.tiles.items():
            out.append(f"                {name!r}: {tuple(toks)!r},")
        out.append("            },")
        out.append("        ),")
    out += ["    },", "}"]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if "--bootstrap" in argv:
        print(bootstrap())
        return 0
    problems = check()
    if problems:
        print("geometry table and binary disagree:")
        for p in problems:
            print("  -", p)
        print()
    print(geometry_plan())
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
