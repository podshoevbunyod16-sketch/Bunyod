"""Dataset importer, validator (encoding, empty, dedup, license, split leakage) & splitter."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from modelforge.storage.artifacts import ArtifactStore
from modelforge.storage.database import StorageDatabase


@dataclass
class DatasetValidationReport:
    valid_utf8: bool
    raw_records: int
    empty_records_removed: int
    duplicate_records_removed: int
    clean_records: int
    total_chars: int
    license: str
    license_warning: str
    split_counts: Dict[str, int]
    split_leakage_count: int
    has_leakage: bool
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_record(text: str) -> str:
    return " ".join(text.strip().lower().split())


class DatasetManager:
    """
    Specification Section 4 & 9:
    - Import TXT, Markdown, JSON, JSONL, CSV.
    - Validate encoding, empty entries, duplicates, license metadata, and train/val/test leakage.
    """

    def __init__(self, db: StorageDatabase, artifacts: ArtifactStore) -> None:
        self._db = db
        self._artifacts = artifacts

    @staticmethod
    def _parse_raw_bytes(raw: bytes, fmt: str) -> tuple[bool, List[str]]:
        try:
            text = raw.decode("utf-8")
            valid_utf8 = True
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
            valid_utf8 = False

        fmt_low = fmt.lower().lstrip(".")
        if fmt_low in ("txt", "text"):
            blocks = [ln for ln in text.splitlines()]
            return valid_utf8, blocks
        if fmt_low in ("md", "markdown"):
            # Split Markdown by paragraphs / double newlines
            blocks = [blk.strip() for blk in text.split("\n\n")]
            return valid_utf8, blocks
        if fmt_low == "jsonl":
            records: List[str] = []
            for ln in text.splitlines():
                if not ln.strip():
                    continue
                obj = json.loads(ln)
                if isinstance(obj, str):
                    records.append(obj)
                elif isinstance(obj, dict):
                    val = obj.get("text") or obj.get("content") or obj.get("prompt") or json.dumps(obj, ensure_ascii=False)
                    if "completion" in obj or "response" in obj:
                        ans = obj.get("completion") or obj.get("response") or ""
                        val = f"{val}\n{ans}"
                    records.append(str(val))
            return valid_utf8, records
        if fmt_low == "json":
            parsed = json.loads(text)
            if isinstance(parsed, list):
                out = []
                for item in parsed:
                    if isinstance(item, str):
                        out.append(item)
                    elif isinstance(item, dict):
                        out.append(str(item.get("text") or item.get("content") or json.dumps(item, ensure_ascii=False)))
                return valid_utf8, out
            if isinstance(parsed, dict):
                items = parsed.get("records") or parsed.get("texts") or parsed.get("data") or [json.dumps(parsed, ensure_ascii=False)]
                return valid_utf8, [str(x.get("text", x) if isinstance(x, dict) else x) for x in items]
        if fmt_low == "csv":
            reader = csv.DictReader(io.StringIO(text))
            out_csv: List[str] = []
            for row in reader:
                val = row.get("text") or row.get("content") or " | ".join(f"{k}: {v}" for k, v in row.items() if v)
                out_csv.append(str(val))
            return valid_utf8, out_csv
        raise ValueError(f"Unsupported dataset format '{fmt}'. Supported: txt, md, json, jsonl, csv.")

    def ingest_dataset(
        self,
        project_id: str,
        name: str,
        raw_content: bytes | str,
        fmt: str = "txt",
        source_path: str = "inline",
        license_name: str = "unspecified",
        deduplicate: bool = True,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        raw_bytes = raw_content.encode("utf-8") if isinstance(raw_content, str) else raw_content
        valid_utf8, raw_records = self._parse_raw_bytes(raw_bytes, fmt)

        non_empty: List[str] = []
        empty_removed = 0
        for r in raw_records:
            cleaned = r.strip()
            if not cleaned:
                empty_removed += 1
            else:
                non_empty.append(cleaned)

        deduped: List[str] = []
        seen_norm: Set[str] = set()
        dup_removed = 0
        for r in non_empty:
            norm = _normalize_record(r)
            if deduplicate and norm in seen_norm:
                dup_removed += 1
                continue
            seen_norm.add(norm)
            deduped.append(r)

        if not deduped:
            raise ValueError("Dataset contains 0 non-empty records after cleaning.")

        rng = random.Random(seed)
        shuffled = list(deduped)
        rng.shuffle(shuffled)

        n = len(shuffled)
        if n >= 3:
            n_train = max(1, int(round(n * split_ratios[0])))
            n_val = max(1, int(round(n * split_ratios[1])))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
            train_recs = shuffled[:n_train]
            val_recs = shuffled[n_train : n_train + n_val]
            test_recs = shuffled[n_train + n_val :]
        elif n == 2:
            train_recs, val_recs, test_recs = [shuffled[0]], [shuffled[1]], []
        else:
            train_recs, val_recs, test_recs = shuffled, [], []

        # Check leakage across splits using normalized record fingerprints
        train_set = {_normalize_record(x) for x in train_recs}
        val_set = {_normalize_record(x) for x in val_recs}
        test_set = {_normalize_record(x) for x in test_recs}
        leakage = len(train_set & val_set) + len(train_set & test_set) + len(val_set & test_set)

        warnings: List[str] = []
        if not valid_utf8:
            warnings.append("Input contained invalid UTF-8 sequences that were replaced.")
        lic_warn = ""
        if license_name.strip().lower() in ("", "unspecified", "unknown"):
            lic_warn = "Dataset license is unspecified; verify redistribution rights before publishing."
            warnings.append(lic_warn)
        if leakage > 0:
            warnings.append(f"Detected {leakage} overlapping normalized records across splits.")

        report = DatasetValidationReport(
            valid_utf8=valid_utf8,
            raw_records=len(raw_records),
            empty_records_removed=empty_removed,
            duplicate_records_removed=dup_removed,
            clean_records=len(deduped),
            total_chars=sum(len(x) for x in deduped),
            license=license_name,
            license_warning=lic_warn,
            split_counts={"train": len(train_recs), "val": len(val_recs), "test": len(test_recs)},
            split_leakage_count=leakage,
            has_leakage=leakage > 0,
            warnings=warnings,
        )

        ds_id = f"ds-{secrets.token_hex(5)}"
        stored_dir = self._artifacts.datasets_dir / ds_id
        stored_dir.mkdir(parents=True, exist_ok=True)
        stored_jsonl = stored_dir / "clean_dataset.jsonl"
        payload_lines = [json.dumps({"text": r}, ensure_ascii=False) for r in deduped]
        stored_bytes = ("\n".join(payload_lines) + "\n").encode("utf-8")
        stored_jsonl.write_bytes(stored_bytes)
        sha = hashlib.sha256(stored_bytes).hexdigest()

        splits_payload = {"train": train_recs, "val": val_recs, "test": test_recs}
        (stored_dir / "splits.json").write_text(
            json.dumps(splits_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        now = time.time()
        record = {
            "id": ds_id,
            "project_id": project_id,
            "name": name,
            "format": fmt.lower().lstrip("."),
            "source_path": source_path,
            "stored_path": str(stored_jsonl),
            "sha256": sha,
            "license": license_name,
            "record_count": len(deduped),
            "validation": report.to_dict(),
            "splits": {"train": len(train_recs), "val": len(val_recs), "test": len(test_recs)},
            "tags": tags or [],
            "created_at": now,
        }
        self._db.upsert_row(
            "datasets",
            {
                "id": ds_id,
                "project_id": project_id,
                "name": name,
                "format": record["format"],
                "source_path": source_path,
                "stored_path": str(stored_jsonl),
                "sha256": sha,
                "license": license_name,
                "record_count": len(deduped),
                "validation_json": json.dumps(report.to_dict(), ensure_ascii=False),
                "splits_json": json.dumps(record["splits"], ensure_ascii=False),
                "tags_json": json.dumps(tags or [], ensure_ascii=False),
                "created_at": now,
            },
        )
        return record

    def ingest_file(
        self,
        project_id: str,
        file_path: Path,
        name: Optional[str] = None,
        license_name: str = "unspecified",
        seed: int = 42,
    ) -> Dict[str, Any]:
        fmt = file_path.suffix.lstrip(".") or "txt"
        return self.ingest_dataset(
            project_id=project_id,
            name=name or file_path.stem,
            raw_content=file_path.read_bytes(),
            fmt=fmt,
            source_path=str(file_path),
            license_name=license_name,
            seed=seed,
        )

    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        row = self._db.query_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        if not row:
            raise KeyError(f"Dataset not found: {dataset_id}")
        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "name": row["name"],
            "format": row["format"],
            "source_path": row["source_path"],
            "stored_path": row["stored_path"],
            "sha256": row["sha256"],
            "license": row["license"],
            "record_count": row["record_count"],
            "validation": json.loads(row["validation_json"] or "{}"),
            "splits": json.loads(row["splits_json"] or "{}"),
            "tags": json.loads(row["tags_json"] or "[]"),
            "created_at": row["created_at"],
        }

    def load_splits(self, dataset_id: str) -> Dict[str, List[str]]:
        ds = self.get_dataset(dataset_id)
        splits_file = Path(ds["stored_path"]).parent / "splits.json"
        return json.loads(splits_file.read_text(encoding="utf-8"))

    def list_datasets(self, project_id: str = "") -> List[Dict[str, Any]]:
        if project_id:
            rows = self._db.query_all(
                "SELECT id FROM datasets WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            )
        else:
            rows = self._db.query_all("SELECT id FROM datasets ORDER BY created_at DESC")
        return [self.get_dataset(r["id"]) for r in rows]
