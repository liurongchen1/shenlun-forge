#!/usr/bin/env bash
#
# Build release artifacts locally. This script never uploads anything.
#
#   ./scripts/release.sh              # verify + build sdist and wheel
#   twine upload dist/*               # only when you decide to publish
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
DIST="$ROOT/dist"

echo "==> Checking version consistency"
"$PYTHON" scripts/check_version.py

if [ -n "$(git status --porcelain)" ]; then
  echo "!! Working tree is dirty — the artifacts would not match any commit." >&2
  echo "!! Commit your changes, then run this again." >&2
  exit 1
fi

if ! "$PYTHON" -c "import build" >/dev/null 2>&1; then
  echo "==> Installing build tooling"
  "$PYTHON" -m pip install --upgrade build twine
fi

if [ -d "$DIST" ]; then
  echo "==> Removing previous builds from $DIST"
  rm -rf "$DIST"
fi

echo "==> Building sdist and wheel"
"$PYTHON" -m build

echo "==> Verifying metadata"
"$PYTHON" -m twine check "$DIST"/*

VERSION="$("$PYTHON" -c "from shenlun_forge._version import __version__; print(__version__)")"

echo
echo "Version: $VERSION"
echo "Artifacts:"
ls -1 "$DIST"

if find "$DIST" -name '*.tar.gz' -print -quit | grep -q .; then
  echo "sdist: present"
else
  echo "sdist: MISSING — PyPI needs it for source installs" >&2
  exit 1
fi

echo
echo "Nothing was uploaded. To publish:"
echo "  twine upload $DIST/*"
