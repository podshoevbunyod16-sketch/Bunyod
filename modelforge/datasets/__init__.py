"""Dataset ingestion (TXT, MD, JSON, JSONL, CSV), validation, deduplication & leakage checking."""

from modelforge.datasets.manager import DatasetManager, DatasetValidationReport

__all__ = ["DatasetManager", "DatasetValidationReport"]
