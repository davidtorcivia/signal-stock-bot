"""Inline enrichment of message text (tweet expansion, etc.) for LLM context."""

from .twitter import TwitterExpander

__all__ = ["TwitterExpander"]
