#!/usr/bin/env python3
"""Compatibility shim for legacy workflows (`python setup.py install`, `pip install -e .`).

All project metadata (name, version, dependencies, extras, entry points) is
declared in ``pyproject.toml`` ([project] table); this file only delegates to
setuptools so both modern (PEP 517/621) and legacy tooling work.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
