"""
Tavily Web Search

Searches Tavily for general company information using a name-only query.
The search is deliberately name-only so it doubles as a "findability test" —
if a company can't be found by name alone, the expensive deep research
API won't find it either.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

import httpx

from ..config import PROCESSING, APIKeys
from ..common.retry import async_retry


@dataclass
class SearchSnippet:
    """A single search result snippet."""
    title: str
    url: str
    content: str
    score: float = 0.0


@dataclass
class TavilySearchResult:
    """
    Result from a Tavily search for general company information.
    
    Attributes:
        company_name: The company we searched for.
        query: The exact query used.
        snippets: List of relevant search result snippets.
        result_count: Number of results returned (indicator of online presence).
        error: Error message if search failed.
    """
    company_name: str
    query: str
    snippets: list[SearchSnippet]
    result_count: int
    answer: Optional[str] = None  # Tavily's LLM-generated summary of the search
    raw_response: Optional[dict] = None
    error: Optional[str] = None


def build_search_query(
    company_name: str,
    homepage_url: Optional[str] = None,
    company_description: Optional[str] = None
) -> str:
    """
    Build a natural language search query for company information.
    
    Uses a conversational format that works well with Tavily's search API
    and avoids overly restrictive constraints (like site:) that can cause
    zero results for lesser-known companies.
    
    Example output:
    - "provide me information on the company Blue Dot"
    
    Args:
        company_name: Name of the company.
        homepage_url: Not used (kept for API compatibility).
        company_description: Not used (kept for API compatibility).
    
    Returns:
        Natural language search query string.
    """
    return f"Find information on the company {company_name}"


async def search_tavily(
    company_name: str,
    homepage_url: Optional[str] = None,
    company_description: Optional[str] = None,
    api_key: Optional[str] = None,
    max_results: int = 5,
    client: Optional[httpx.AsyncClient] = None,
) -> TavilySearchResult:
    """
    Search Tavily for general company information.
    
    This is a controlled, single search per company to ensure
    predictable costs (~$0.02 per search with advanced depth).
    
    Args:
        company_name: Name of the company to search for.
        homepage_url: Company's website (for domain constraint).
        company_description: Crunchbase description (for disambiguation).
        api_key: Tavily API key. Uses credentials folder or env var if not provided.
        max_results: Maximum number of results to return.
        client: Optional shared httpx.AsyncClient for connection pooling.
    
    Returns:
        TavilySearchResult with general company information snippets.
    """
    if not api_key:
        keys = APIKeys()
        api_key = keys.tavily
    
    if not api_key:
        return TavilySearchResult(
            company_name=company_name,
            query="",
            snippets=[],
            result_count=0,
            error="Tavily API key not set. Add to credentials/tavily_api_key.txt"
        )
    
    query = build_search_query(company_name, homepage_url, company_description)
    
    async def _do_search(c: httpx.AsyncClient) -> TavilySearchResult:
        """Raw search call — raises on retryable errors so retry wrapper can handle them."""
        response = await c.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "advanced",  # Highest relevance, returns reranked chunks
                "max_results": max_results,
                "chunks_per_source": 3,  # Multiple snippets per source for richer context
                "include_answer": "advanced",  # Detailed LLM-generated summary to guide GPT classification
                "include_raw_content": False,  # Not needed for Stage 1 filtering
            }
        )
        response.raise_for_status()
        data = response.json()
        
        # Parse results into snippets
        snippets = []
        for result in data.get("results", []):
            snippet = SearchSnippet(
                title=result.get("title", ""),
                url=result.get("url", ""),
                content=result.get("content", ""),
                score=result.get("score", 0.0),
            )
            snippets.append(snippet)
        
        return TavilySearchResult(
            company_name=company_name,
            query=query,
            snippets=snippets,
            result_count=len(snippets),
            answer=data.get("answer") or None,
            raw_response=data,
        )
    
    async def _search_with_retry(c: httpx.AsyncClient) -> TavilySearchResult:
        """Wrap the search with retry logic, catch permanent failures."""
        try:
            return await async_retry(
                _do_search, c,
                max_retries=PROCESSING.max_retries,
                delay_base=PROCESSING.retry_delay_base,
                operation_name=f"tavily_search({company_name})",
            )
        except httpx.HTTPStatusError as e:
            return TavilySearchResult(
                company_name=company_name,
                query=query,
                snippets=[],
                result_count=0,
                error=f"HTTP {e.response.status_code}: {e.response.text[:100]}"
            )
        except Exception as e:
            return TavilySearchResult(
                company_name=company_name,
                query=query,
                snippets=[],
                result_count=0,
                error=f"{type(e).__name__}: {str(e)[:100]}"
            )
    
    if client is not None:
        return await _search_with_retry(client)
    
    # Fallback: create a one-off client (backward compat for simple scripts)
    async with httpx.AsyncClient(timeout=PROCESSING.tavily_timeout) as new_client:
        return await _search_with_retry(new_client)


async def search_tavily_batch(
    companies: list[tuple[str, Optional[str], Optional[str]]],  # (name, homepage_url, description)
    api_key: Optional[str] = None,
    max_concurrent: int = 5  # Tavily rate limits are stricter
) -> list[TavilySearchResult]:
    """
    Search Tavily for multiple companies concurrently.
    
    Args:
        companies: List of (company_name, homepage_url, description) tuples.
        api_key: Tavily API key.
        max_concurrent: Maximum simultaneous requests (default 5 for Tavily).
    
    Returns:
        List of TavilySearchResult in same order as input.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def search_with_semaphore(name: str, url: Optional[str], desc: Optional[str]) -> TavilySearchResult:
        async with semaphore:
            return await search_tavily(name, url, desc, api_key)
    
    tasks = [search_with_semaphore(name, url, desc) for name, url, desc in companies]
    return await asyncio.gather(*tasks)


# ─────────────────────────────────────────────────────────────────────────────
# SYNCHRONOUS WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def search_tavily_sync(
    company_name: str,
    homepage_url: Optional[str] = None,
    company_description: Optional[str] = None,
    api_key: Optional[str] = None
) -> TavilySearchResult:
    """Synchronous wrapper for search_tavily."""
    return asyncio.run(search_tavily(company_name, homepage_url, company_description, api_key))


def search_tavily_batch_sync(
    companies: list[tuple[str, Optional[str], Optional[str]]],
    api_key: Optional[str] = None
) -> list[TavilySearchResult]:
    """Synchronous wrapper for search_tavily_batch."""
    return asyncio.run(search_tavily_batch(companies, api_key))
