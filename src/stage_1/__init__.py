"""
Stage 1: Presence Filter

Gathers general company intelligence via web search and predicts
research priority for subsequent deep research stages.

Components:
    1. website.py    — Website health checks
    2. tavily.py     — Tavily web search
    3. classifier.py — GPT-5-nano research priority classification
    4. pipeline.py   — Orchestrator combining all three steps
"""

from .website import WebsiteStatus, check_website, check_websites_batch
from .tavily import SearchSnippet, TavilySearchResult, search_tavily, build_search_query
from .classifier import PresenceAssessment, classify_company
from .pipeline import Stage1Result, run_stage_1

__all__ = [
    "WebsiteStatus", "check_website", "check_websites_batch",
    "SearchSnippet", "TavilySearchResult", "search_tavily", "build_search_query",
    "PresenceAssessment", "classify_company",
    "Stage1Result", "run_stage_1",
]
