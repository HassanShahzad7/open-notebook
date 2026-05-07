#!/usr/bin/env python3
"""
One-shot helper: registers all SAP AI Core models in Open Notebook,
then auto-assigns default chat / embedding / transformation models.

Prerequisites:
  1. aicore_proxy.py is running on http://127.0.0.1:8001
  2. Open Notebook API is running on http://127.0.0.1:5055
  3. The SAP AI Core openai_compatible credential already exists in Settings → API Keys

Usage:
    uv run setup_aicore_models.py
"""

import asyncio
import sys
from typing import Optional

import httpx

PROXY_URL = "http://127.0.0.1:8001/v1"
API_URL = "http://127.0.0.1:5055/api"
# Default Open Notebook password — change if you set OPEN_NOTEBOOK_PASSWORD
API_PASSWORD = "open-notebook-change-me"

EMBEDDING_PATTERNS = ["text-embedding", "embedding", "embed"]
TTS_PATTERNS = ["tts", "text-to-speech"]
STT_PATTERNS = ["whisper", "stt", "speech-to-text"]


def classify(model_id: str) -> str:
    n = model_id.lower()
    if any(p in n for p in EMBEDDING_PATTERNS):
        return "embedding"
    if any(p in n for p in TTS_PATTERNS):
        return "text_to_speech"
    if any(p in n for p in STT_PATTERNS):
        return "speech_to_text"
    return "language"


async def main() -> None:
    headers = {"Authorization": f"Bearer {API_PASSWORD}"}

    async with httpx.AsyncClient(timeout=30) as c:
        # ── 1. Fetch models from proxy ──────────────────────────────────────
        print("Fetching models from SAP AI Core proxy …")
        try:
            r = await c.get(
                f"{PROXY_URL}/models",
                headers={"Authorization": "Bearer aicore"},
            )
            r.raise_for_status()
        except Exception as e:
            print(f"ERROR: Could not reach proxy at {PROXY_URL}: {e}")
            print("Make sure aicore_proxy.py is running:  uv run aicore_proxy.py")
            sys.exit(1)

        proxy_models = r.json().get("data", [])
        print(f"  Found {len(proxy_models)} models in proxy")
        if not proxy_models:
            print("ERROR: Proxy returned no models. Check proxy logs.")
            sys.exit(1)

        # ── 2. Find openai_compatible credential ───────────────────────────
        print("\nLooking up SAP AI Core credential in Open Notebook …")
        try:
            r = await c.get(f"{API_URL}/credentials", headers=headers)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                print(f"ERROR: Auth failed. Is API_PASSWORD correct? (current: '{API_PASSWORD}')")
                print("Set OPEN_NOTEBOOK_PASSWORD in .env if you changed it.")
            else:
                print(f"ERROR: API returned {e.response.status_code}: {e.response.text[:200]}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Could not reach Open Notebook API at {API_URL}: {e}")
            print("Make sure the API is running:  uv run --env-file .env run_api.py")
            sys.exit(1)

        creds = r.json()
        aicore_cred: Optional[dict] = None
        for cred in creds:
            if cred.get("provider") == "openai_compatible":
                aicore_cred = cred
                break

        if not aicore_cred:
            print("  Not found — creating credential automatically …")
            r = await c.post(
                f"{API_URL}/credentials",
                json={
                    "name": "SAP AI Core",
                    "provider": "openai_compatible",
                    "modalities": ["language", "embedding"],
                    "api_key": "aicore",
                    "base_url": "http://127.0.0.1:8001/v1",
                },
                headers=headers,
            )
            if not r.is_success:
                print(f"ERROR creating credential: {r.status_code}: {r.text[:400]}")
                sys.exit(1)
            aicore_cred = r.json()
            print(f"  Created credential: {aicore_cred.get('name')} (id={aicore_cred.get('id')})")

        cred_id = aicore_cred["id"]
        print(f"  Using credential: {aicore_cred.get('name', 'unknown')} (id={cred_id})")

        # ── 3. Register models with auto-classification ────────────────────
        classified = [
            {
                "name": m["id"],
                "provider": "openai_compatible",
                "model_type": classify(m["id"]),
            }
            for m in proxy_models
        ]

        print("\nModels to register:")
        for m in classified:
            print(f"  {m['name']:55s} → {m['model_type']}")

        r = await c.post(
            f"{API_URL}/credentials/{cred_id}/register-models",
            json={"models": classified},
            headers=headers,
        )
        if not r.is_success:
            print(f"\nERROR registering models: {r.status_code}: {r.text[:400]}")
            sys.exit(1)

        result = r.json()
        print(f"\nRegistration complete: {result.get('created', 0)} new, {result.get('existing', 0)} already existed")

        # ── 4. Find the IDs of the SAP AI Core language + embedding models ───
        print("\nLooking up registered SAP AI Core model IDs …")
        r = await c.get(f"{API_URL}/models", headers=headers)
        all_models = r.json() if r.is_success else []

        # Pick best language model (prefer gpt-4o, else first language model)
        lang_model_id = None
        embed_model_id = None
        prefer_chat = ["gpt-4o", "gpt-4.1", "anthropic--claude-4.6-sonnet", "mistralai--mistral-large-instruct"]
        aicore_lang = [m for m in all_models if m.get("provider") == "openai_compatible" and m.get("type") == "language"]
        aicore_embed = [m for m in all_models if m.get("provider") == "openai_compatible" and m.get("type") == "embedding"]

        for preferred in prefer_chat:
            match = next((m for m in aicore_lang if m.get("name") == preferred), None)
            if match:
                lang_model_id = match["id"]
                break
        if not lang_model_id and aicore_lang:
            lang_model_id = aicore_lang[0]["id"]

        if aicore_embed:
            embed_model_id = aicore_embed[0]["id"]

        print(f"  Chat/transformation model: {lang_model_id}")
        print(f"  Embedding model:           {embed_model_id}")

        # ── 5. Force-set the critical defaults (overrides stale OpenAI values) ─
        print("\nSetting default models …")
        defaults_payload = {}
        if lang_model_id:
            defaults_payload["default_chat_model"] = lang_model_id
            defaults_payload["default_transformation_model"] = lang_model_id
            defaults_payload["large_context_model"] = lang_model_id
            defaults_payload["default_tools_model"] = lang_model_id
        if embed_model_id:
            defaults_payload["default_embedding_model"] = embed_model_id

        r = await c.put(f"{API_URL}/models/defaults", json=defaults_payload, headers=headers)
        if not r.is_success:
            print(f"WARNING: setting defaults failed: {r.status_code}: {r.text[:400]}")
        else:
            d = r.json()
            print(f"  default_chat_model:           {d.get('default_chat_model')}")
            print(f"  default_transformation_model: {d.get('default_transformation_model')}")
            print(f"  default_embedding_model:      {d.get('default_embedding_model')}")

        print("\nDone! You can now use Open Notebook with SAP AI Core.")
        print("If files are still stuck, start the background worker:")
        print("  uv run --env-file .env surreal-commands-worker --import-modules commands")


if __name__ == "__main__":
    asyncio.run(main())
