"""
Common English words to ignore during symbol extraction.
Prevents false positives like "CAN" (Can), "KEY" (KeyCorp), "ALL" (Allstate).
"""

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "from", "in", "of", "on", "to", "with",
    "vs", "versus", "via", "compare", "compared", # Comparison words
    "rsi", "macd", "sma", "ema", "bb", "bollinger", # TA words (avoid ticker confusion)
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "can", "could", "should", "would", "may", "might", "must",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "her", "its", "our", "their",
    "what", "which", "who", "whom", "whose", "where", "why", "how",
    "this", "that", "these", "those",
    "all", "any", "some", "no", "not", "none",
    "up", "down", "out", "over", "under", "again", "further",
    "now", "here", "there", "top", "best", "worst", 
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", # Numbers as words
    "eleven", "twelve", "hundred", "thousand", "million", "billion",
    "buy", "sell", "hold", "short", "long", "trade", "chart", "price",
    "stock", "share", "market", "value", "graph", "plot", "show", "get",
    "see", "look", "time", "day", "week", "month", "year",
    "high", "low", "open", "close", "volume", "cap", "about",
    "news", "info", "data", "real", "today", "yesterday", "tomorrow",
    "since", "from", "until", "last", "next", "between", # Date range words
    "list", "watch", "track", "alert", "change", "move", "action",
    "go", "run", "set", "add", "remove", "delete", "clear",
    "want", "need", "like", "use", "give", "take", "make", "put", "try", # Common verbs
    "tell", "say", "ask", "know", "think", "find", "search",
    "please", "thanks", "thank", "hello", "hi", "hey",
}
