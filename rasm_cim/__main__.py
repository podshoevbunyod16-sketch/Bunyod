"""``python -m rasm_cim`` — the unified entry point (added in v0.17).

The frozen v0.1 research package had no ``__main__``, so ``python -m rasm_cim``
failed with "No module named rasm_cim.__main__" while every other version had
its own entry point.  This file adds the missing dispatcher and delegates to
the v0.17 CLI, which is a superset of the v0.16 one.  Nothing else in the
frozen package is touched.
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from rasm_cim_v17.cli import main as v17_main
    return v17_main(argv)


if __name__ == "__main__":
    sys.exit(main())
