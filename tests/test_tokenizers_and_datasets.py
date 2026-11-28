"""Tests for multilingual Russian/Tajik/English tokenizers and dataset validation & leakage checks."""

from __future__ import annotations

import json
from pathlib import Path

from modelforge.datasets.manager import DatasetManager
from modelforge.storage.artifacts import ArtifactStore
from modelforge.storage.database import StorageDatabase
from modelforge.tokenizers.tokenizer import load_tokenizer, train_tokenizer, verify_multilingual_unicode


def test_char_and_bpe_tokenizers_russian_tajik_roundtrip(tmp_path: Path):
    corpus = [
        "Привет мир! Языковая модель работает локально.",
        "Салом ҷаҳон! Забони тоҷикӣ ҳарфҳои ғ, ӣ, қ, ӯ, ҳ, ҷ дорад.",
        "ModelForge AI cross-platform runtime on ARM64 and x64.",
    ]
    for kind in ("char", "bpe"):
        tok = train_tokenizer(corpus, tokenizer_type=kind, vocab_size=260)
        rep = verify_multilingual_unicode(tok)
        assert rep["all_passed"] is True, rep

        saved = tok.save(tmp_path / f"tok_{kind}.json")
        reloaded = load_tokenizer(saved)
        tajik_phrase = "Дар Хуҷанд ва Душанбе забони тоҷикӣ: ғӣқӯҳҷ!"
        ids = reloaded.encode(tajik_phrase, add_bos=True, add_eos=True)
        assert reloaded.decode(ids, skip_special=True) == tajik_phrase


def test_dataset_formats_dedup_empty_and_no_leakage(tmp_path: Path):
    db = StorageDatabase(tmp_path / "mf.db")
    art = ArtifactStore(tmp_path / "store")
    dm = DatasetManager(db, art)

    raw_txt = "\n".join(
        [
            "Первая запись для обучения модели.",
            "",
            "   ",
            "Первая запись для обучения модели.",  # exact duplicate
            "Вторая уникальная запись на русском языке.",
            "Салом! Сабти сеюм ба забони тоҷикӣ.",
            "Fourth unique English record for split testing.",
            "Fifth distinct record for validation split.",
            "Sixth distinct record for test split.",
        ]
    )
    ds = dm.ingest_dataset(
        project_id="proj-1",
        name="multilingual-clean",
        raw_content=raw_txt,
        fmt="txt",
        license_name="",
        deduplicate=True,
    )
    val = ds["validation"]
    assert val["valid_utf8"] is True
    assert val["empty_records_removed"] == 2
    assert val["duplicate_records_removed"] == 1
    assert val["clean_records"] == 6
    assert val["has_leakage"] is False
    assert val["license_warning"] != ""

    # JSONL & CSV formats
    jsonl_content = "\n".join(
        [
            json.dumps({"prompt": "Савол", "completion": "Ҷавоб"}, ensure_ascii=False),
            json.dumps({"prompt": "Вопрос", "completion": "Ответ"}, ensure_ascii=False),
            json.dumps({"prompt": "Question", "completion": "Answer"}),
        ]
    )
    ds_jsonl = dm.ingest_dataset("proj-1", "qa-jsonl", jsonl_content, fmt="jsonl", license_name="MIT")
    assert ds_jsonl["record_count"] == 3

    csv_content = "text,lang\nПривет мир,ru\nСалом ҷаҳон,tg\nHello world,en\n"
    ds_csv = dm.ingest_dataset("proj-1", "qa-csv", csv_content, fmt="csv", license_name="CC-BY-4.0")
    assert ds_csv["record_count"] == 3
