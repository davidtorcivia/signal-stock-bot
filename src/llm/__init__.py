"""LLM integration: OpenAI-compatible client + per-user conversation history."""

from .client import LLMClient, LLMDisabled, LLMError, LLMNotConfigured
from .history import ConversationHistory
from .reactor import EmojiReactor

__all__ = [
    "LLMClient",
    "LLMDisabled",
    "LLMError",
    "LLMNotConfigured",
    "ConversationHistory",
    "EmojiReactor",
]
