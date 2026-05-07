#!/usr/bin/env python3
"""
SAP AI Core → OpenAI-compatible local proxy.

Routes requests to the correct backend based on model prefix:
  - OpenAI / Mistral / Meta / IBM  → gen_ai_hub native OpenAI client (/chat/completions)
  - anthropic-- / amazon--         → Amazon Bedrock Converse API (direct to deployment URL)
  - gemini-- / google--            → Vertex AI generateContent API (direct to deployment URL)

Reads credentials from aicore_key.json automatically.

Usage:
    uv run aicore_proxy.py

Then in Open Notebook (Settings → API Keys → Add Credential):
  Provider:  OpenAI-Compatible
  Base URL:  http://127.0.0.1:8002/v1
  API Key:   aicore   (any non-empty string)
"""

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

load_dotenv(Path(__file__).parent / ".env")

_key_file = Path(__file__).parent / "aicore_key.json"
if _key_file.exists() and not os.environ.get("AICORE_SERVICE_KEY_FILE"):
    os.environ["AICORE_SERVICE_KEY_FILE"] = str(_key_file)

from gen_ai_hub.proxy.native.openai import chat  # noqa: E402

try:
    from gen_ai_hub.proxy.native.openai import embeddings as _oai_embeddings
except ImportError:
    _oai_embeddings = None

try:
    from gen_ai_hub.proxy.gen_ai_hub_proxy import GenAIHubProxyClient
    _proxy_client = GenAIHubProxyClient()
except Exception as e:
    print(f"[aicore] proxy client init error: {e}")
    _proxy_client = None

app = FastAPI(title="SAP AI Core Proxy", version="3.0")


# ---------------------------------------------------------------------------
# Model routing helpers
# ---------------------------------------------------------------------------

def _is_bedrock_model(model_name: str) -> bool:
    return model_name.startswith("anthropic--") or model_name.startswith("amazon--")


def _is_vertex_model(model_name: str) -> bool:
    return model_name.startswith("gemini") or model_name.startswith("google--")


def _get_deployment_info(model_name: str) -> tuple[str, dict]:
    if _proxy_client is None:
        raise RuntimeError("AI Core proxy client not initialized")
    deployment = _proxy_client.select_deployment(model_name=model_name)
    headers = dict(_proxy_client.request_header)
    headers["Content-Type"] = "application/json"
    return deployment.url, headers


# ---------------------------------------------------------------------------
# Bedrock (Converse API) — anthropic-- / amazon--
# ---------------------------------------------------------------------------

def _openai_to_converse(messages: list, body: dict) -> dict:
    system_blocks = []
    converse_msgs = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict) and "text" in c
            )
        else:
            text = str(content)

        if role == "system":
            system_blocks.append({"text": text})
        else:
            converse_msgs.append({
                "role": "assistant" if role == "assistant" else "user",
                "content": [{"text": text}],
            })

    payload: dict[str, Any] = {
        "messages": converse_msgs,
        "inferenceConfig": {
            "maxTokens": max(body.get("max_tokens") or 0, 8192),
            "temperature": body.get("temperature", 0),
        },
    }
    if system_blocks:
        payload["system"] = system_blocks
    return payload


def _converse_to_openai(response: dict, model_name: str) -> dict:
    try:
        content = response["output"]["message"]["content"][0]["text"]
    except (KeyError, IndexError):
        content = str(response)
    usage = response.get("usage", {})
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": usage.get("inputTokens", 0),
            "completion_tokens": usage.get("outputTokens", 0),
            "total_tokens": usage.get("totalTokens", 0),
        },
    }


async def _call_bedrock(model_name: str, body: dict) -> dict:
    url, headers = await asyncio.to_thread(_get_deployment_info, model_name)
    url = url.rstrip("/") + "/converse"
    payload = _openai_to_converse(body.get("messages", []), body)
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, f"AI Core Bedrock error: {resp.text[:500]}")
        return resp.json()


# ---------------------------------------------------------------------------
# Vertex AI (generateContent) — gemini-- / google--
# ---------------------------------------------------------------------------

def _openai_to_vertex(messages: list, body: dict) -> dict:
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict) and "text" in c
            )
        else:
            text = str(content)

        if role == "system":
            contents.append({"role": "user", "parts": [{"text": f"[System]: {text}"}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text}]})
        else:
            contents.append({"role": "user", "parts": [{"text": text}]})

    return {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max(body.get("max_tokens") or 0, 8192),
            "temperature": body.get("temperature", 0),
        },
    }


def _vertex_to_openai(response: dict, model_name: str) -> dict:
    try:
        parts = response["candidates"][0]["content"]["parts"]
        content = "".join(p.get("text", "") for p in parts if "text" in p)
    except (KeyError, IndexError):
        content = ""
    usage = response.get("usageMetadata", {})
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
        },
    }


async def _call_vertex(model_name: str, body: dict) -> dict:
    url, headers = await asyncio.to_thread(_get_deployment_info, model_name)
    url = url.rstrip("/") + f"/models/{model_name}:generateContent"
    payload = _openai_to_vertex(body.get("messages", []), body)
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, f"AI Core Vertex error: {resp.text[:500]}")
        return resp.json()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _list_model_names() -> list[str]:
    if _proxy_client is None:
        return []
    try:
        names = []
        for d in _proxy_client.deployments:
            name = getattr(d, "model_name", None) or getattr(d, "name", None)
            if name:
                names.append(name)
        return names
    except Exception as e:
        print(f"[aicore] deployment list error: {e}")
        return []


@app.get("/v1/models")
async def list_models():
    names = await asyncio.to_thread(_list_model_names)
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": 0, "owned_by": "sap-ai-core"}
            for name in names
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model_name: str = body.pop("model", "")
    print(f"[aicore] chat request: model={model_name!r} max_tokens={body.get('max_tokens')} keys={list(body.keys())}")

    try:
        if _is_bedrock_model(model_name):
            raw = await _call_bedrock(model_name, body)
            return JSONResponse(_converse_to_openai(raw, model_name))

        elif _is_vertex_model(model_name):
            raw = await _call_vertex(model_name, body)
            return JSONResponse(_vertex_to_openai(raw, model_name))

        else:
            # OpenAI-compatible: gpt-*, mistralai--, meta--, ibm--, etc.
            body["model_name"] = model_name
            def _call():
                return chat.completions.create(**body)
            result = await asyncio.to_thread(_call)
            return JSONResponse(result.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        print(f"[aicore] chat error: {e}")
        raise HTTPException(500, str(e))


@app.post("/v1/embeddings")
async def embeddings_endpoint(request: Request):
    if _oai_embeddings is None:
        raise HTTPException(500, "gen_ai_hub embeddings not available")

    body = await request.json()
    model_name: str = body.get("model", "")
    input_data = body.get("input", "")

    def _call():
        return _oai_embeddings.create(model_name=model_name, input=input_data)

    try:
        result = await asyncio.to_thread(_call)
        return JSONResponse(result.model_dump())
    except Exception as e:
        print(f"[aicore] embed error: {e}")
        raise HTTPException(500, str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    print("Starting SAP AI Core proxy on http://127.0.0.1:8002")
    print(f"Service key: {os.environ.get('AICORE_SERVICE_KEY_FILE', 'not set')}")
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
