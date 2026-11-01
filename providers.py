"""Thin wrappers around the OpenAI (ChatGPT) and Anthropic (Claude) APIs.

Each provider exposes:
  - list_models()  -> list of model IDs the API key can use
  - stream(system, prompt) -> generator yielding text chunks
"""

from __future__ import annotations

from typing import Iterator

import anthropic
import openai

# Fallback lists, used only if the live model listing fails.
DEFAULT_OPENAI_MODELS = ["gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4o", "o3", "o4-mini"]
DEFAULT_CLAUDE_MODELS = [
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
    "claude-fable-5-1",
]

# Claude models that accept server-side refusal fallbacks ("default" routing).
_CLAUDE_FALLBACK_MODELS = {
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
}

# OpenAI model IDs that are not text chat models.
_OPENAI_EXCLUDE = (
    "audio", "realtime", "tts", "transcribe", "image", "search",
    "embedding", "moderation", "whisper", "dall-e", "instruct",
)


class ProviderError(RuntimeError):
    pass


class ChatGPT:
    name = "ChatGPT"
    avatar = "🟢"

    def __init__(self, api_key: str, model: str = DEFAULT_OPENAI_MODELS[0]):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def list_models(self) -> list[str]:
        ids = [m.id for m in self.client.models.list()]
        chat = [
            i for i in ids
            if i.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-"))
            and not any(x in i for x in _OPENAI_EXCLUDE)
        ]
        return sorted(chat, reverse=True)

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=True
            )
            for chunk in resp:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.BadRequestError as e:
            # Some models / orgs don't allow streaming: fall back to one-shot.
            if "stream" not in str(e).lower():
                raise ProviderError(f"ChatGPT error: {e}") from e
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            yield resp.choices[0].message.content or ""
        except openai.OpenAIError as e:
            raise ProviderError(f"ChatGPT error: {e}") from e


class Claude:
    name = "Claude"
    avatar = "🟠"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_CLAUDE_MODELS[0],
        effort: str | None = None,
        refusal_fallback: bool = True,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.effort = effort
        self.refusal_fallback = refusal_fallback

    def list_models(self) -> list[str]:
        return [m.id for m in self.client.models.list(limit=100)]

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        kwargs: dict = dict(
            model=self.model,
            max_tokens=32000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}

        use_fallback = self.refusal_fallback and self.model in _CLAUDE_FALLBACK_MODELS

        def open_stream():
            if use_fallback:
                return self.client.beta.messages.stream(
                    betas=["server-side-fallback-2026-07-01"],
                    extra_body={"fallbacks": "default"},
                    **kwargs,
                )
            return self.client.messages.stream(**kwargs)

        try:
            try:
                ctx = open_stream().__enter__()
            except anthropic.BadRequestError as e:
                # Older models have smaller output caps: retry with a lower limit.
                if "max_tokens" not in str(e):
                    raise
                kwargs["max_tokens"] = 8192
                ctx = open_stream().__enter__()
            try:
                yield from ctx.text_stream
                final = ctx.get_final_message()
            finally:
                ctx.close()
        except anthropic.APIError as e:
            raise ProviderError(f"Claude error: {e}") from e

        if final.stop_reason == "refusal":
            yield "\n\n_(Claude declined to answer this request.)_"
        elif final.stop_reason == "max_tokens":
            yield "\n\n_(Response truncated: hit the max token limit.)_"
