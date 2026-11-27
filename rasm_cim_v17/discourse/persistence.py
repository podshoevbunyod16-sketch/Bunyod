"""Сохранение состояния диалога v0.17 между запусками CLI.

Проблема, которую это решает: `rasm-cim say "…"` — отдельный процесс. Ядро
v0.16 сохраняет граф и контекст (`kb.json`, `context.json`), но состояние
v0.17 — явный фокус исправления (`v17_explicit_focus`) и память упоминаний
(`v17_memory`) — жило только в памяти процесса. Из-за этого в CLI:

* «Нет, я про Анну» в одном вызове и «Она врач?» в следующем не связывались
  (исправление терялось между процессами);
* повторные упоминания не накапливались.

Файл `v17_state.json` лежит рядом с базой и пишется после каждого ответа,
только если состояние изменилось. Внутри — обычный JSON без путей и секретов.
"""
from __future__ import annotations

import json
import os

from .memory import DialogueMemory

STATE_FILE = "v17_state.json"


def state_path(home: str | os.PathLike) -> str:
    return os.path.join(str(home), STATE_FILE)


def save_state(dm, home: str | os.PathLike) -> bool:
    """Записать фокус и память диалога. Возвращает True, если файл обновлён."""
    memory = getattr(dm, "v17_memory", None)
    payload = {
        "version": 1,
        "explicit_focus": getattr(dm, "v17_explicit_focus", {}) or {},
        "mentioned": sorted(getattr(dm, "v17_mentioned", ()) or ()),
        "memory": memory.to_dict() if memory is not None else None,
    }
    path = state_path(home)
    try:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                if handle.read() == text:
                    return False
        os.makedirs(str(home), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    except OSError:
        return False
    return True


def load_state(dm, home: str | os.PathLike) -> bool:
    """Восстановить состояние в менеджер. Возвращает True, если что-то загружено."""
    path = state_path(home)
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    focus = payload.get("explicit_focus") or {}
    if isinstance(focus, dict):
        dm.v17_explicit_focus = {k: v for k, v in focus.items()
                                 if k in ("entity_id", "role", "turn")}
    mentioned = payload.get("mentioned") or []
    if mentioned:
        dm.v17_mentioned = {int(x) for x in mentioned}
    memory = payload.get("memory")
    if isinstance(memory, dict):
        try:
            dm.v17_memory = DialogueMemory.from_dict(memory)
        except (TypeError, ValueError):
            pass
    return True
