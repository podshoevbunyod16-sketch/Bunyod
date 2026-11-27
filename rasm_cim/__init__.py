"""RASM-CIM research prototype v0.1. See experiments.py for audit results.

v0.17 note: the prototype's modules need numpy, but this package is also the
entry point for ``python -m rasm_cim`` (v0.17 CLI), and the shipped promise is
"stdlib only, works on Termux".  Eagerly importing numpy here made the CLI fail
on any machine without numpy — so the prototype's names are loaded on first use
(PEP 562) instead.  The public API is unchanged::

    from rasm_cim import RNS            # works, loads numpy only now
"""

__version__ = "0.1.0"

#: name -> (module, attribute) — loaded on first access, never at import time
_LAZY = {
    "RNS": (".rns", "RNS"),
    "IntegerFPE": (".fpe", "IntegerFPE"),
    "ResidueFPE": (".fpe", "ResidueFPE"),
    "AssociativeMemory": (".cam", "AssociativeMemory"),
    "DiagonalSSM": (".state", "DiagonalSSM"),
    "Resonator": (".resonator", "Resonator"),
    "RASMCIM": (".system", "RASMCIM"),
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}"
                             ) from None
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value          # cache, so the next access is a dict hit
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals()) + __all__))
