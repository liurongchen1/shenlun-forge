"""Strategy registry — centralized strategy name → class mapping.

Avoids hardcoded if/elif chains in ShenlunForge.create().
New strategies only need to register here.
"""

from __future__ import annotations
from typing import Type, Dict


def get_strategy_class(name: str):
    """Get a strategy class by name.
    
    Returns the class (not an instance) so the caller can construct it
    with appropriate arguments.
    
    Raises ValueError if strategy name is unknown.
    """
    registry = _build_registry()
    if name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown strategy: '{name}'. Available strategies: {available}")
    return registry[name]


def list_strategies() -> list[str]:
    """Return list of available strategy names."""
    return sorted(_build_registry().keys())


def _build_registry() -> Dict[str, type]:
    """Build the strategy registry. Lazy-imports to avoid circular imports."""
    from .compressor import ShenlunForge
    from .strategies.simple import SimpleStrategy
    from .strategies.hybrid import HCSCompressor

    return {
        "acs": ShenlunForge,       # Full 4-step ACS pipeline
        "simple": SimpleStrategy,      # Rule-based, no LLM needed
        "hcs": HCSCompressor,    # ACS + verbatim code augmentation
    }
