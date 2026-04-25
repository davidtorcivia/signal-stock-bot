"""LLM integration: OpenAI-compatible client + per-user conversation history."""

from .client import LLMClient, LLMDisabled, LLMError, LLMNotConfigured
from .history import ConversationHistory

__all__ = [
    "LLMClient",
    "LLMDisabled",
    "LLMError",
    "LLMNotConfigured",
    "ConversationHistory",
]
