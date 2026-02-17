"""
Stage 1 Pipeline Orchestrator

Combines website check, Tavily search, and GPT classification into
a single run_stage_1() function. Used by the test script; production
runners (run_tavily_pass.py / run_gpt_pass.py) call the components
directly for better decoupling.
"""

import asyncio
import json as json_module
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import IO, Optional

import httpx

from .website import WebsiteStatus, check_website
from .tavily import TavilySearchResult, search_tavily
from .classifier import PresenceAssessment, classify_company


@dataclass
class Stage1Result:
    """
    Complete result from Stage 1 presence filter.
    
    Combines website check, search, and classification into one result.
    """
    company_name: str
    company_id: int
    
    # Component results
    website_status: WebsiteStatus
    search_result: TavilySearchResult
    assessment: PresenceAssessment
    
    # Summary for easy access
    presence_score: int
    research_priority_score: int  # 0-5 (0=not researchable, 3+=worth deep research)
    
    # Cost tracking
    estimated_cost: float = 0.011  # $0.01 Tavily + $0.001 GPT-5-nano
    
    @property
    def should_deep_research(self) -> bool:
        """Should this company proceed to deep research? (score >= 3)"""
        return self.research_priority_score >= 3
    
    @property
    def is_high_priority(self) -> bool:
        """Is this a high priority company for deep research? (score >= 4)"""
        return self.research_priority_score >= 4


def _build_tavily_log_entry(
    result: Stage1Result,
    company_description: Optional[str] = None,
    homepage_url: Optional[str] = None,
) -> dict:
    """
    Build a JSONL entry for the Tavily log file.
    
    Contains: company metadata, website check, and full Tavily raw response.
    """
    ws = result.website_status
    sr = result.search_result
    
    return {
        "company_id": result.company_id,
        "company_name": result.company_name,
        "company_description": company_description,
        "homepage_url": homepage_url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "website_check": {
            "url": ws.url,
            "is_alive": ws.is_alive,
            "status_code": ws.status_code,
            "final_url": ws.final_url,
            "is_redirect": ws.is_redirect,
            "error": ws.error,
        },
        "tavily": {
            "query": sr.query,
            "result_count": sr.result_count,
            "raw_response": sr.raw_response,
        },
    }


def _build_gpt_log_entry(
    result: Stage1Result,
) -> dict:
    """
    Build a JSONL entry for the GPT log file.
    
    Contains only: company ID/name and GPT output (parsed result).
    Input data (Tavily results, prompts) lives in the Tavily log file.
    """
    assess = result.assessment
    
    return {
        "company_id": result.company_id,
        "company_name": result.company_name,
        "online_presence_score": assess.online_presence_score,
        "research_priority_score": assess.research_priority_score,
        "reasoning": assess.reasoning,
        "error": assess.error,
    }


async def run_stage_1(
    company_id: int,
    company_name: str,
    homepage_url: Optional[str],
    company_description: Optional[str],
    long_description: Optional[str] = None,
    tavily_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    tavily_log_file: Optional[IO] = None,
    gpt_log_file: Optional[IO] = None,
    http_client: Optional[httpx.AsyncClient] = None,
    tavily_client: Optional[httpx.AsyncClient] = None,
    openai_client: Optional[httpx.AsyncClient] = None,
) -> Stage1Result:
    """
    Run the complete Stage 1 presence filter for a single company.
    
    This orchestrates:
    1. Website health check
    2. Tavily search for general company information (name-only query)
    3. GPT-5-nano analysis comparing search results against Crunchbase profile
    
    Args:
        company_id: Unique identifier (rcid from Crunchbase).
        company_name: Company name.
        homepage_url: Company website URL.
        company_description: Crunchbase short description.
        long_description: Crunchbase long description (if available).
        tavily_api_key: Tavily API key.
        openai_api_key: OpenAI API key.
        tavily_log_file: Optional file handle for Tavily JSONL log.
        gpt_log_file: Optional file handle for GPT JSONL log.
        http_client: Optional shared httpx client for website checks.
        tavily_client: Optional shared httpx client for Tavily API.
        openai_client: Optional shared httpx client for OpenAI API.
    
    Returns:
        Stage1Result with all component results and research priority score.
    """
    # Steps 1 & 2: Website check and Tavily search are independent — run in parallel
    website_status, search_result = await asyncio.gather(
        check_website(homepage_url or "", client=http_client),
        search_tavily(
            company_name,
            homepage_url,
            company_description,
            api_key=tavily_api_key,
            client=tavily_client,
        ),
    )
    
    # Step 3: Analyze and assign research priority (GPT compares search vs Crunchbase profile)
    assessment = await classify_company(
        company_name,
        company_description,
        website_status,
        search_result,
        api_key=openai_api_key,
        homepage_url=homepage_url,
        long_description=long_description,
        client=openai_client,
    )
    
    result = Stage1Result(
        company_name=company_name,
        company_id=company_id,
        website_status=website_status,
        search_result=search_result,
        assessment=assessment,
        presence_score=assessment.online_presence_score,
        research_priority_score=assessment.research_priority_score,
    )
    
    # Write JSONL log entries (separate files for Tavily and GPT)
    if tavily_log_file is not None:
        entry = _build_tavily_log_entry(result, company_description, homepage_url)
        tavily_log_file.write(json_module.dumps(entry) + "\n")
        tavily_log_file.flush()
    
    if gpt_log_file is not None:
        entry = _build_gpt_log_entry(result)
        gpt_log_file.write(json_module.dumps(entry) + "\n")
        gpt_log_file.flush()
    
    return result


def run_stage_1_sync(
    company_id: int,
    company_name: str,
    homepage_url: Optional[str],
    company_description: Optional[str],
    long_description: Optional[str] = None,
    tavily_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    tavily_log_file: Optional[IO] = None,
    gpt_log_file: Optional[IO] = None
) -> Stage1Result:
    """Synchronous wrapper for run_stage_1."""
    return asyncio.run(run_stage_1(
        company_id, company_name, homepage_url, company_description,
        long_description, tavily_api_key, openai_api_key,
        tavily_log_file, gpt_log_file
    ))
