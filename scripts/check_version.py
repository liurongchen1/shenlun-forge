#!/usr/bin/env python3
"""Guard against version drift.

The published distribution version used to be hard-coded in two places and
drifted apart (`__version__ == "0.2.0"` while PyPI served `0.2.4.1`). This
script fails if that can happen again.

Run it in CI and before every release:

    python scripts/check_version.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _fail(message: str, errors: list) -> None:
    errors.append(message)


def main() -> int:
    errors: list = []

    sys.path.insert(0, str(ROOT))
    try:
        from shenlun_forge._version import __version__ as source_version
    except ImportError as exc:  # pragma: no cover - only on a broken checkout
        print(f"ERROR: cannot import shenlun_forge._version: {exc}")
        return 1

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    # Only look inside the [project] table — [tool.setuptools.dynamic] also
    # contains a "version =" line, and matching that would be a false alarm.
    project_match = re.search(
        r"^\[project\](.*?)(?=^\[|\Z)", pyproject, re.MULTILINE | re.DOTALL
    )
    project_section = project_match.group(1) if project_match else ""

    # 1. The version must come from _version.py, not from a literal.
    if re.search(r'^version\s*=', project_section, re.MULTILINE):
        _fail('pyproject.toml declares a static "version"; remove it and keep '
              'dynamic = ["version"]', errors)
    if 'version = {attr = "shenlun_forge._version.__version__"}' not in pyproject:
        _fail("pyproject.toml does not read the version from "
              "shenlun_forge/_version.py", errors)

    # 2. The package must import the version rather than redefining it.
    init = (ROOT / "shenlun_forge" / "__init__.py").read_text(encoding="utf-8")
    if "from ._version import __version__" not in init:
        _fail('shenlun_forge/__init__.py must do "from ._version import '
              '__version__"', errors)
    if re.search(r'^__version__\s*=', init, re.MULTILINE):
        _fail("shenlun_forge/__init__.py hard-codes __version__ again", errors)

    # 3. If the distribution is installed, its metadata must agree.
    try:
        from importlib.metadata import version as dist_version

        installed = dist_version("shenlun-forge")
    except Exception:
        installed = None

    if installed is not None and installed != source_version:
        _fail(f"installed distribution is {installed} but the source says "
              f"{source_version}; reinstall with pip install -e .", errors)

    if errors:
        print("Version check FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"Version check OK: {source_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
