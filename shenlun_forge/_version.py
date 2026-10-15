"""Single source of truth for the package version.

``pyproject.toml`` reads this via ``[tool.setuptools.dynamic]`` and the
package reads it at import time, so the CLI and the installed distribution
can never disagree again.
"""

__version__ = "0.4.0"
