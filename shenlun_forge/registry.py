"""Strategy registry — centralized strategy name → class mapping.

Avoids hardcoded if/elif chains in ShenlunForge.create().
New strategies only need to register here.
"""

from __future__ import annotations
from typing import Type, Dict


# Strategies added at runtime via register_strategy(). Kept at module level so
# every call to _build_registry() sees them.
_EXTRA_STRATEGIES: Dict[str, type] = {}


def register_strategy(name: str, strategy_cls: type) -> None:
    """Register a custom strategy under ``name``.

    After registering, the name works with ``ShenlunForge.create(strategy=...)``,
    appears in ``list_strategies()`` and is accepted by the CLI:

        from shenlun_forge import register_strategy

        class MyStrategy:
            async def compress(self, messages, current_query="", max_tokens=4000, **kwargs):
                ...

        register_strategy("mine", MyStrategy)
        forge = ShenlunForge.create(strategy="mine")

    The class must implement ``async def compress(messages, current_query,
    max_tokens, **kwargs) -> CompressionResult``. If it also takes
    ``model``/``api_key``/``base_url`` constructor arguments it will receive
    them from ``create()``.
    """
    _EXTRA_STRATEGIES[name] = strategy_cls


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

    registry = {
        "acs": ShenlunForge,       # Full 4-step ACS pipeline
        "simple": SimpleStrategy,      # Rule-based, no LLM needed
        "hcs": HCSCompressor,    # ACS + verbatim code augmentation
    }
    registry.update(_EXTRA_STRATEGIES)
    return registry
