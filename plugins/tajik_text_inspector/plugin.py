"""Built-in ModelForge plugin: Tajik & Russian Cyrillic corpus inspector."""

from typing import Any, Dict, List

TAJIK_CHARS = set("ғӣқӯҳҷҒӢҚӮҲҶ")
CYRILLIC_RANGE = range(0x0400, 0x0500)


def run(inputs: Dict[str, Any]) -> Dict[str, Any]:
    text = str(inputs.get("text", ""))
    tajik_count = sum(1 for c in text if c in TAJIK_CHARS)
    has_cyrillic = any(ord(c) in CYRILLIC_RANGE for c in text)
    has_latin = any("a" <= c.lower() <= "z" for c in text)

    scripts: List[str] = []
    if has_cyrillic:
        scripts.append("cyrillic")
    if tajik_count > 0:
        scripts.append("tajik_cyrillic")
    if has_latin:
        scripts.append("latin")

    return {
        "char_count": len(text),
        "word_count": len(text.split()),
        "tajik_specific_count": tajik_count,
        "detected_scripts": scripts,
    }
