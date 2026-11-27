"""Session export / show / reset helpers (v0.13)."""
from __future__ import annotations

import json
import os
from typing import Optional

from .context import ContextStore


def export_session(ctx: ContextStore, out_path: Optional[str] = None) -> dict:
    data = ctx.export_session()
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    return data


def show_context(ctx: ContextStore) -> dict:
    return ctx.summary()


def reset_context(ctx: ContextStore) -> None:
    ctx.reset()
