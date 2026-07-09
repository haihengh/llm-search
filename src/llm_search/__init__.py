"""LLM Search — Give your local LLM internet search capability.

A middleware that sits between chat clients and LM Studio, intercepting
tool-call requests for web_search and executing them against a search
provider (SearXNG by default — self-hosted, no API keys).
"""

__version__ = "0.2.0"
