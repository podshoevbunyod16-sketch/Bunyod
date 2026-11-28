"""Real Character & Byte/Unicode BPE tokenizers supporting Russian, Tajik & Latin scripts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
TAJIK_EXTRA_CHARS = "ғӣқӯҳҷҒӢҚӮҲҶ"
RUSSIAN_CHARS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"


class BaseTokenizer:
    tokenizer_type: str = "base"

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        raw = json.dumps(payload, ensure_ascii=False, indent=2)
        path.write_text(raw, encoding="utf-8")
        return path


class CharTokenizer(BaseTokenizer):
    """Unicode character-level tokenizer with guaranteed Cyrillic + Tajik roundtrip."""

    tokenizer_type = "char"

    def __init__(self, vocab: List[str]) -> None:
        deduped: List[str] = []
        seen = set()
        for tok in SPECIAL_TOKENS + list(vocab):
            if tok not in seen:
                deduped.append(tok)
                seen.add(tok)
        self.id_to_token = deduped
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    @classmethod
    def train(cls, texts: Iterable[str], include_multilingual_alphabet: bool = True) -> "CharTokenizer":
        chars = set()
        if include_multilingual_alphabet:
            chars.update(" \n\t.,!?;:-()\"'0123456789")
            chars.update("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
            chars.update(RUSSIAN_CHARS)
            chars.update(TAJIK_EXTRA_CHARS)
        for txt in texts:
            chars.update(txt)
        return cls(sorted(chars))

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in text:
            ids.append(self.token_to_id.get(ch, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        out: List[str] = []
        for idx in ids:
            i = int(idx)
            if skip_special and i in special_ids:
                continue
            if 0 <= i < len(self.id_to_token):
                out.append(self.id_to_token[i])
        return "".join(out)

    def to_dict(self) -> Dict[str, Any]:
        raw_vocab = json.dumps(self.id_to_token, ensure_ascii=False).encode("utf-8")
        return {
            "tokenizer_type": self.tokenizer_type,
            "version": 1,
            "vocab_size": self.vocab_size,
            "sha256": hashlib.sha256(raw_vocab).hexdigest(),
            "vocab": self.id_to_token,
        }


class UnicodeBPETokenizer(BaseTokenizer):
    """
    Subword Byte-Pair Encoding (BPE) tokenizer operating over Unicode characters + word-boundary
    marker `▁`, preserving Russian, Tajik (`ғ,ӣ,қ,ӯ,ҳ,ҷ`), and Latin text cleanly.
    """

    tokenizer_type = "bpe"
    SPACE_MARK = "▁"

    def __init__(self, vocab: List[str], merges: List[Tuple[str, str]]) -> None:
        deduped: List[str] = []
        seen = set()
        for tok in SPECIAL_TOKENS + list(vocab):
            if tok not in seen:
                deduped.append(tok)
                seen.add(tok)
        self.id_to_token = deduped
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        self.merges = [(a, b) for a, b in merges]
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        target_vocab_size: int = 384,
        include_multilingual_alphabet: bool = True,
    ) -> "UnicodeBPETokenizer":
        corpus = "\n".join(texts)
        base_chars = set(corpus.replace(" ", cls.SPACE_MARK))
        base_chars.add(cls.SPACE_MARK)
        if include_multilingual_alphabet:
            base_chars.update("\n\t.,!?;:-()\"'0123456789")
            base_chars.update("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
            base_chars.update(RUSSIAN_CHARS)
            base_chars.update(TAJIK_EXTRA_CHARS)

        vocab = SPECIAL_TOKENS + sorted(base_chars)
        vocab_set = set(vocab)

        # Pre-segment into whitespace-prefixed words so BPE merges stay within words
        word_counts: Counter[Tuple[str, ...]] = Counter()
        normalized = corpus.replace(" ", f" {cls.SPACE_MARK}")
        for piece in normalized.split(" "):
            if not piece:
                continue
            word_counts[tuple(piece)] += 1

        merges: List[Tuple[str, str]] = []
        max_merges = max(0, target_vocab_size - len(vocab))

        for _ in range(max_merges):
            pair_freq: Counter[Tuple[str, str]] = Counter()
            for symbols, freq in word_counts.items():
                for i in range(len(symbols) - 1):
                    pair_freq[(symbols[i], symbols[i + 1])] += freq
            if not pair_freq:
                break
            best_pair, best_count = pair_freq.most_common(1)[0]
            if best_count < 2:
                break
            merged_tok = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            if merged_tok not in vocab_set:
                vocab.append(merged_tok)
                vocab_set.add(merged_tok)

            new_word_counts: Counter[Tuple[str, ...]] = Counter()
            a, b = best_pair
            for symbols, freq in word_counts.items():
                new_syms: List[str] = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                        new_syms.append(merged_tok)
                        i += 2
                    else:
                        new_syms.append(symbols[i])
                        i += 1
                new_word_counts[tuple(new_syms)] += freq
            word_counts = new_word_counts

        return cls(vocab=vocab, merges=merges)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def _bpe_word(self, word: str) -> List[str]:
        if not word:
            return []
        symbols = list(word)
        while len(symbols) > 1:
            candidates = []
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair in self.merge_ranks:
                    candidates.append((self.merge_ranks[pair], i, pair))
            if not candidates:
                break
            _, _, best_pair = min(candidates, key=lambda x: x[0])
            a, b = best_pair
            merged = a + b
            next_syms: List[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    next_syms.append(merged)
                    i += 2
                else:
                    next_syms.append(symbols[i])
                    i += 1
            symbols = next_syms
        return symbols

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        if text:
            marked = text.replace(" ", f"\x00{self.SPACE_MARK}")
            pieces = marked.split("\x00")
            for piece in pieces:
                for tok in self._bpe_word(piece):
                    ids.append(self.token_to_id.get(tok, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        pieces: List[str] = []
        for idx in ids:
            i = int(idx)
            if skip_special and i in special_ids:
                continue
            if 0 <= i < len(self.id_to_token):
                pieces.append(self.id_to_token[i])
        return "".join(pieces).replace(self.SPACE_MARK, " ")

    def to_dict(self) -> Dict[str, Any]:
        raw = json.dumps({"vocab": self.id_to_token, "merges": self.merges}, ensure_ascii=False).encode("utf-8")
        return {
            "tokenizer_type": self.tokenizer_type,
            "version": 1,
            "vocab_size": self.vocab_size,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "vocab": self.id_to_token,
            "merges": [[a, b] for a, b in self.merges],
        }


def train_tokenizer(
    texts: Iterable[str],
    tokenizer_type: str = "bpe",
    vocab_size: int = 384,
) -> BaseTokenizer:
    kind = tokenizer_type.strip().lower()
    if kind == "char":
        return CharTokenizer.train(texts)
    if kind in ("bpe", "subword", "token"):
        return UnicodeBPETokenizer.train(texts, target_vocab_size=vocab_size)
    raise ValueError(f"Unsupported tokenizer_type '{tokenizer_type}'. Choose 'char' or 'bpe'.")


def load_tokenizer(path: Path) -> BaseTokenizer:
    data = json.loads(path.read_text(encoding="utf-8"))
    kind = data.get("tokenizer_type", "char")
    if kind == "char":
        return CharTokenizer(vocab=data["vocab"])
    if kind == "bpe":
        merges = [(m[0], m[1]) for m in data.get("merges", [])]
        return UnicodeBPETokenizer(vocab=data["vocab"], merges=merges)
    raise ValueError(f"Unknown serialized tokenizer_type '{kind}'.")


def verify_multilingual_unicode(tokenizer: BaseTokenizer) -> Dict[str, Any]:
    """Verify lossless roundtrip across Russian, Tajik Cyrillic (ғ,ӣ,қ,ӯ,ҳ,ҷ) & English."""
    samples = {
        "russian": "Привет, мир! Компактная языковая модель обучается локально.",
        "tajik": "Салом ҷаҳон! Забони тоҷикӣ ҳарфҳои хос дорад: ғ, ӣ, қ, ӯ, ҳ, ҷ.",
        "english": "Hello world! Cross-platform ModelForge AI runs on ARM64 and x64.",
    }
    report: Dict[str, Any] = {"all_passed": True, "languages": {}}
    for lang, sample in samples.items():
        ids = tokenizer.encode(sample, add_bos=True, add_eos=True)
        decoded = tokenizer.decode(ids, skip_special=True)
        has_unk = tokenizer.unk_id in ids
        exact = decoded == sample and not has_unk
        if not exact:
            report["all_passed"] = False
        report["languages"][lang] = {
            "exact_roundtrip": exact,
            "token_count": len(ids),
            "unk_tokens": sum(1 for x in ids if x == tokenizer.unk_id),
        }
    return report
