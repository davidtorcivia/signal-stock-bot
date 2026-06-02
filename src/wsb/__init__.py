"""
WSB daily digest feature.

A scheduled pipeline that scrapes r/wallstreetbets (via a self-hosted Redlib
frontend), compiles a structured digest of the day's top posts, comments, and
most-mentioned tickers, hands it to the deep-think model for analysis (with a
live price/news cross-check), publishes a static daily read, and posts a teaser
+ link to a Signal group.

Module layout:
  redlib    — Redlib HTML scraper (RedditPost/RedditComment + RedlibSource).
  digest    — ticker tally + WSBDigest dataclass + compile_wsb_digest().
  store     — wsb_digests SQLite table (one canonical row per day).
  site      — static page + index + og:image generation.
  service   — orchestrator (crawl -> digest -> deep-think -> persist -> render).
"""
