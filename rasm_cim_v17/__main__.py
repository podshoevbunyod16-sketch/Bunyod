"""``python -m rasm_cim_v17`` -> the v0.17 CLI."""
from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
