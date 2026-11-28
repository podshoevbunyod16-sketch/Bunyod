"""Tests for real training weight mutation, checkpoint integrity, resume, transfer learning & GGUF/Safetensors export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from modelforge.core.runtime import ModelForgeCore
from modelforge.inference.engine import GenerationConfig
from modelforge.inference.gguf import inspect_gguf_v3
from modelforge.training.checkpoints import load_checkpoint
from modelforge.training.engine import TrainingRunConfig


def test_real_training_mutates_weights_reduces_loss_and_generates(tmp_path: Path):
    core = ModelForgeCore(home_dir=tmp_path / "mf_home")
    try:
        proj = core.projects.create_project("RU-TJ Model Test")
        mdl = core.models.create_from_prompt(
            project_id=proj.id,
            prompt="Создай компактную языковую модель для русского и таджикского текста с экспериментальной архитектурой",
        )
        assert mdl["architecture"] == "hybrid_attention_ssm"

        train_res = core.training.train(
            model_id=mdl["id"],
            config=TrainingRunConfig(epochs=6, batch_size=4, learning_rate=6e-3, max_seq_len=48, seed=42),
        )
        m = train_res["metrics"]
        assert m["weights_changed"] is True
        assert m["weight_l2_delta"] > 0.01
        assert m["final_train_loss"] < m["initial_train_loss"]

        # Reload checkpoint and run streaming text generation
        ckpt_path = train_res["checkpoint"]["checkpoint_path"]
        loaded = load_checkpoint(ckpt_path)
        assert loaded["manifest"]["weights_sha256"] == train_res["checkpoint"]["sha256"]

        gen = core.inference.generate(
            model_id=mdl["id"],
            prompt="Салом ҷаҳон! Привет",
            config=GenerationConfig(max_tokens=20, temperature=0.7, seed=42),
        )
        assert gen["tokens_generated"] > 0
        assert len(gen["generated_text"]) > 0

        # Transfer learning freezes embedding & prefix block while updating head
        before_transfer = load_checkpoint(ckpt_path)["weights"]
        transfer_res = core.training.train(
            model_id=mdl["id"],
            config=TrainingRunConfig(epochs=2, batch_size=4, learning_rate=4e-3, mode="transfer", freeze_layers=1),
        )
        after_transfer = load_checkpoint(transfer_res["checkpoint"]["checkpoint_path"])["weights"]
        assert "tok_emb" in transfer_res["metrics"]["frozen_parameters"]
        np.testing.assert_array_equal(before_transfer["tok_emb"], after_transfer["tok_emb"])
        assert not np.array_equal(before_transfer["lm_head"], after_transfer["lm_head"])

        # Export GGUF v3 & Safetensors and verify binary structure
        gguf_info = core.registry.export_model(mdl["id"], "gguf")
        inspected = inspect_gguf_v3(Path(gguf_info["path"]))
        assert inspected["valid_gguf"] is True
        assert inspected["version"] == 3
        assert inspected["tensor_count"] == len(after_transfer)
        assert inspected["metadata"]["general.architecture"] == "hybrid_attention_ssm"

        st_info = core.registry.export_model(mdl["id"], "safetensors")
        assert Path(st_info["path"]).exists()
        assert st_info["size_bytes"] > 100

        # Corrupting checkpoint weights triggers SHA-256 tamper error
        weights_npz = Path(ckpt_path) / "weights.npz"
        tampered = dict(before_transfer)
        tampered["lm_head"] = tampered["lm_head"] + 1.0
        np.savez_compressed(str(weights_npz), **tampered)
        with pytest.raises(ValueError, match="SHA-256"):
            load_checkpoint(ckpt_path)
    finally:
        core.shutdown()
