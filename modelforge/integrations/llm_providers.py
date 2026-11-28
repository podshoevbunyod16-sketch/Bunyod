"""Unified LLM Provider Manager (Local, OpenAI-compatible HTTP, optional LiteLLM) without silent fallback."""

from __future__ import annotations

import json
import secrets
import time
from typing import Any, Callable, Dict, List, Optional

from modelforge.inference.engine import GenerationConfig, InferenceEngine
from modelforge.model_builder.builder import ModelBuilder
from modelforge.security.permissions import PermissionDeniedError
from modelforge.security.secrets import SecretManager, redact_secrets
from modelforge.storage.database import StorageDatabase


class LLMProviderManager:
    """
    Specification Section 11:
    - Local models, OpenAI-compatible endpoints, and optional LiteLLM behind one interface.
    - Connection test & model listing where API supports it.
    - Explicit model choice & transparent routing policy.
    - NEVER switches provider silently when cost, privacy, or local/external boundary changes.
    - Stores API keys in SecretManager; surfaces HTTP 429 quota/rate-limit and network failures clearly.
    """

    def __init__(
        self,
        db: StorageDatabase,
        secrets_mgr: SecretManager,
        model_builder: ModelBuilder,
        inference_engine: InferenceEngine,
        offline_getter: Callable[[], bool] = lambda: False,
    ) -> None:
        self._db = db
        self._secrets = secrets_mgr
        self._models = model_builder
        self._inference = inference_engine
        self._offline_getter = offline_getter
        self._ensure_builtin_local_provider()

    def _ensure_builtin_local_provider(self) -> None:
        existing = self._db.query_one("SELECT id FROM llm_providers WHERE id = 'provider-local'")
        if not existing:
            self._db.upsert_row(
                "llm_providers",
                {
                    "id": "provider-local",
                    "name": "Local ModelForge Runtime",
                    "provider_type": "local",
                    "base_url": "local://modelforge",
                    "default_model": "auto-latest-local",
                    "secret_key_name": "",
                    "is_paid": 0,
                    "is_external": 0,
                    "enabled": 1,
                    "metadata_json": json.dumps({"privacy": "on_device", "cost": "free"}),
                    "created_at": time.time(),
                },
            )

    def register_provider(
        self,
        name: str,
        provider_type: str,  # local | openai_compatible | litellm
        base_url: str = "",
        default_model: str = "",
        api_key: Optional[str] = None,
        is_paid: bool = False,
    ) -> Dict[str, Any]:
        ptype = provider_type.strip().lower()
        if ptype not in ("local", "openai_compatible", "litellm"):
            raise ValueError("provider_type must be 'local', 'openai_compatible', or 'litellm'.")
        pid = f"prov-{secrets.token_hex(4)}"
        secret_name = ""
        if api_key:
            secret_name = f"llm_{pid}_api_key"
            self._secrets.set_secret(secret_name, api_key)

        is_external = ptype != "local" and not base_url.startswith(("http://127.0.0.1", "http://localhost"))
        now = time.time()
        self._db.upsert_row(
            "llm_providers",
            {
                "id": pid,
                "name": name,
                "provider_type": ptype,
                "base_url": base_url.rstrip("/"),
                "default_model": default_model,
                "secret_key_name": secret_name,
                "is_paid": int(is_paid),
                "is_external": int(is_external),
                "enabled": 1,
                "metadata_json": json.dumps(
                    {
                        "privacy": "external_api" if is_external else "local",
                        "cost": "paid" if is_paid else "free_or_self_hosted",
                    }
                ),
                "created_at": now,
            },
        )
        return self.get_provider(pid)

    def get_provider(self, provider_id: str) -> Dict[str, Any]:
        row = self._db.query_one("SELECT * FROM llm_providers WHERE id = ?", (provider_id,))
        if not row:
            raise KeyError(f"LLM Provider not found: {provider_id}")
        return {
            "id": row["id"],
            "name": row["name"],
            "provider_type": row["provider_type"],
            "base_url": row["base_url"],
            "default_model": row["default_model"],
            "has_api_key": bool(row["secret_key_name"] and self._secrets.get_secret(row["secret_key_name"])),
            "is_paid": bool(row["is_paid"]),
            "is_external": bool(row["is_external"]),
            "enabled": bool(row["enabled"]),
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "created_at": row["created_at"],
        }

    def list_providers(self) -> List[Dict[str, Any]]:
        rows = self._db.query_all("SELECT id FROM llm_providers ORDER BY created_at ASC")
        return [self.get_provider(r["id"]) for r in rows]

    def test_provider(self, provider_id: str) -> Dict[str, Any]:
        prov = self.get_provider(provider_id)
        if prov["provider_type"] == "local":
            local_models = self._models.list_models()
            return {
                "ok": True,
                "provider_id": provider_id,
                "models": [m["id"] for m in local_models],
                "routing_policy": "strict_explicit_selection_no_silent_fallback",
            }

        if self._offline_getter() and prov["is_external"]:
            raise PermissionDeniedError("External LLM provider probe is blocked in Offline Mode.")

        row = self._db.query_one("SELECT secret_key_name FROM llm_providers WHERE id = ?", (provider_id,))
        api_key = self._secrets.get_secret(row["secret_key_name"]) if row and row["secret_key_name"] else None

        if prov["provider_type"] == "litellm":
            try:
                import litellm  # type: ignore  # noqa: F401

                return {"ok": True, "provider_id": provider_id, "engine": "litellm"}
            except ImportError:
                return {"ok": False, "provider_id": provider_id, "error": "Optional package `litellm` is not installed."}

        try:
            import httpx

            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            with httpx.Client(timeout=8.0) as client:
                resp = client.get(f"{prov['base_url']}/models", headers=headers)
            if resp.status_code == 200:
                body = resp.json()
                models = [m.get("id", "") for m in body.get("data", []) if isinstance(m, dict)]
                return {"ok": True, "provider_id": provider_id, "models": models}
            if resp.status_code == 429:
                return {
                    "ok": False,
                    "provider_id": provider_id,
                    "error_type": "quota_or_rate_limit",
                    "error": "HTTP 429: Provider quota or rate limit exceeded.",
                }
            return {
                "ok": False,
                "provider_id": provider_id,
                "error_type": "http_error",
                "error": f"HTTP {resp.status_code}: {redact_secrets(resp.text[:300])}",
            }
        except Exception as exc:
            return {
                "ok": False,
                "provider_id": provider_id,
                "error_type": "network_error",
                "error": redact_secrets(f"{type(exc).__name__}: {exc}"),
            }

    def complete(
        self,
        prompt: str,
        provider_id: str = "provider-local",
        model_id: str = "",
        max_tokens: int = 48,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Explicit provider completion. Never switches from local to paid/external or vice versa silently.
        """
        prov = self.get_provider(provider_id)
        if not prov["enabled"]:
            raise PermissionDeniedError(f"Provider '{prov['name']}' ({provider_id}) is disabled.")
        if self._offline_getter() and prov["is_external"]:
            raise PermissionDeniedError(
                f"Provider '{prov['name']}' requires network access, which is blocked in Offline Mode. "
                "No silent fallback performed."
            )

        if prov["provider_type"] == "local":
            target_model_id = model_id
            if not target_model_id:
                models = self._models.list_models()
                if not models:
                    return {
                        "provider_id": provider_id,
                        "mode": "local_assistant",
                        "text": (
                            "Локальных обученных моделей пока нет в реестре. "
                            "Создайте модель через Model Studio или `modelforge create model`."
                        ),
                    }
                target_model_id = models[0]["id"]
            gen = self._inference.generate(
                model_id=target_model_id,
                prompt=prompt,
                config=GenerationConfig(max_tokens=max_tokens, temperature=temperature),
            )
            return {
                "provider_id": provider_id,
                "model_id": target_model_id,
                "privacy": "on_device",
                "text": gen.get("generated_text", ""),
                "full_text": gen.get("full_text", ""),
                "metrics": {
                    "tokens_per_sec": gen.get("tokens_per_sec", 0.0),
                    "latency_ms": gen.get("total_duration_ms", 0.0),
                },
            }

        row = self._db.query_one("SELECT secret_key_name FROM llm_providers WHERE id = ?", (provider_id,))
        api_key = self._secrets.get_secret(row["secret_key_name"]) if row and row["secret_key_name"] else None
        chosen_model = model_id or prov["default_model"]
        if not chosen_model:
            raise ValueError("Explicit model_id or provider default_model is required.")

        if prov["provider_type"] == "litellm":
            import litellm  # type: ignore

            resp = litellm.completion(
                model=chosen_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                api_key=api_key,
            )
            return {
                "provider_id": provider_id,
                "model_id": chosen_model,
                "privacy": prov["metadata"].get("privacy", "external_api"),
                "text": resp.choices[0].message.content,
            }

        import httpx

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": chosen_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(f"{prov['base_url']}/chat/completions", headers=headers, json=payload)
        if resp.status_code == 429:
            raise RuntimeError("Quota/Rate-limit error (HTTP 429) from LLM provider.")
        if resp.status_code >= 400:
            raise RuntimeError(f"Provider returned HTTP {resp.status_code}: {redact_secrets(resp.text[:300])}")
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return {
            "provider_id": provider_id,
            "model_id": chosen_model,
            "privacy": prov["metadata"].get("privacy", "external_api"),
            "text": content,
        }
