"""Thinker backends — where the heavy cognition happens.

Supports OpenRouter (remote, free tier), Ollama (local), and
HuggingFace (local, instrumentable).
"""
from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import requests

log = logging.getLogger(__name__)


class ThinkerBackend(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str:
        ...


class OpenRouterBackend(ThinkerBackend):
    """Qwen3.6 or any OpenRouter model."""

    def __init__(
        self,
        model: str = "qwen/qwen3.6",
        api_key: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def chat(self, messages: list[dict], **kwargs) -> str:
        if not self.api_key:
            return "[!] No OpenRouter API key. Set OPENROUTER_API_KEY env var."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://demian.local",
            "X-Title": "Demian",
        }
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        resp = requests.post(self.url, headers=headers, json=body, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


class OllamaBackend(ThinkerBackend):
    """Local Ollama model."""

    def __init__(
        self,
        model: str = "qwen2.5:14b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: list[dict], **kwargs) -> str:
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": kwargs.get("max_tokens", self.max_tokens),
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class HFBackend(ThinkerBackend):
    """Local HuggingFace model — can be instrumented like proprioceptor."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model.eval()
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: list[dict], **kwargs) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            do_sample=True,
        )
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )


def create_backend(backend_type: str = "openrouter", **kwargs) -> ThinkerBackend:
    if backend_type == "openrouter":
        return OpenRouterBackend(**kwargs)
    elif backend_type == "ollama":
        return OllamaBackend(**kwargs)
    elif backend_type == "hf":
        return HFBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend_type}")
