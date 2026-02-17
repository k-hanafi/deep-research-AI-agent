"""
GPT-5-nano Research Priority Classifier

Compares Tavily search results against the Crunchbase profile to assess:
1. Online presence (1-10 scale)
2. Research priority (0-5 scale) — whether an expensive deep research
   API call is worth the cost for this company.
"""

import json as json_module
from dataclasses import dataclass
from typing import Optional

import httpx

from ..config import PROCESSING, APIKeys, PROMPTS_DIR
from ..common.retry import async_retry
from .website import WebsiteStatus
from .tavily import TavilySearchResult


@dataclass
class PresenceAssessment:
    """
    GPT-5-nano's assessment of a company's profile and research priority.
    
    This is the output of Stage 1 and determines routing to subsequent stages.
    Includes raw API data for logging/auditing.
    """
    company_name: str
    online_presence_score: int  # 1-10
    research_priority_score: int  # 0-5 (0=not researchable, 5=definitely yields GenAI findings)
    reasoning: str  # Brief explanation for consistency assessment and priority judgment
    error: Optional[str] = None
    
    # Raw data for logging — not used for routing, but preserved for auditing
    user_prompt: Optional[str] = None  # The exact prompt sent to GPT
    raw_gpt_response: Optional[dict] = None  # Full OpenAI API response


def _load_system_prompt() -> str:
    """
    Load the classifier system prompt from the prompts file.
    
    Returns:
        The system prompt string.
    """
    prompt_file = PROMPTS_DIR / "stage_1_classifier.txt"
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    content = prompt_file.read_text()
    
    # Extract only the SYSTEM PROMPT section (between "## SYSTEM PROMPT" and "---")
    lines = content.split('\n')
    in_system_prompt = False
    system_prompt_lines = []
    
    for line in lines:
        if line.strip() == "## SYSTEM PROMPT":
            in_system_prompt = True
            continue
        if in_system_prompt and line.strip() == "---":
            break
        if in_system_prompt:
            system_prompt_lines.append(line)
    
    return '\n'.join(system_prompt_lines).strip()


# Load system prompt from file (cached at module load time)
CLASSIFIER_SYSTEM_PROMPT = _load_system_prompt()


def _build_classifier_prompt(
    company_name: str,
    company_description: Optional[str],
    website_status: WebsiteStatus,
    search_result: TavilySearchResult,
    homepage_url: Optional[str] = None,
    long_description: Optional[str] = None,
) -> str:
    """
    Build the user prompt for the classifier.
    
    Separates the Crunchbase profile (ground truth) from the search tool
    output so GPT can assess consistency between them.
    """
    # --- CRUNCHBASE PROFILE section ---
    profile_lines = [
        "CRUNCHBASE PROFILE:",
        f"Company: {company_name}",
        f"Description: {company_description or 'None'}",
    ]
    if long_description and long_description.strip():
        profile_lines.append(f"About: {long_description.strip()}")
    profile_lines.append(f"Website: {homepage_url or 'None'}")
    
    # --- SEARCH TOOL OUTPUT section ---
    if website_status.is_alive:
        website_info = f"ACTIVE ({website_status.status_code})"
    else:
        website_info = f"DOWN ({website_status.error})"
    
    if search_result.error:
        search_info = f"Search error: {search_result.error}"
    elif search_result.result_count == 0:
        search_info = "No search results found."
    else:
        # Include ALL snippets with FULL content
        snippets = []
        for i, snippet in enumerate(search_result.snippets, 1):
            snippets.append(f"{i}. {snippet.title}\n   URL: {snippet.url}\n   {snippet.content}")
        search_info = "\n\n".join(snippets)
    
    search_lines = [
        "SEARCH TOOL OUTPUT:",
        f'The following results were retrieved by searching only the company name "{company_name}".',
        f"Website check: {website_info}",
    ]
    
    # Include Tavily's LLM-generated summary if available
    if search_result.answer:
        search_lines.append(f"Search summary: {search_result.answer}")
    
    search_lines.extend([
        f"Results ({search_result.result_count} found):",
        "",
        search_info,
    ])
    
    prompt = "\n".join(profile_lines) + "\n\n" + "\n".join(search_lines)
    return prompt


async def classify_company(
    company_name: str,
    company_description: Optional[str],
    website_status: WebsiteStatus,
    search_result: TavilySearchResult,
    api_key: Optional[str] = None,
    homepage_url: Optional[str] = None,
    long_description: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> PresenceAssessment:
    """
    Use GPT-5-nano to assess a company's profile and predict research priority.
    
    Args:
        company_name: Name of the company.
        company_description: Crunchbase short description (if available).
        website_status: Result from check_website().
        search_result: Result from search_tavily().
        api_key: OpenAI API key. Uses env var if not provided.
        homepage_url: Company homepage URL from Crunchbase (for profile context).
        long_description: Crunchbase long description (if available).
        client: Optional shared httpx.AsyncClient for connection pooling.
    
    Returns:
        PresenceAssessment with online_presence_score and research_priority_score.
    """
    if not api_key:
        keys = APIKeys()
        api_key = keys.openai
    
    if not api_key:
        return PresenceAssessment(
            company_name=company_name,
            online_presence_score=0,
            research_priority_score=0,
            reasoning="",
            error="OpenAI API key not set. Add to credentials/openai_api_key.txt"
        )
    
    user_prompt = _build_classifier_prompt(
        company_name, company_description, website_status, search_result,
        homepage_url=homepage_url, long_description=long_description,
    )
    
    async def _do_classify(c: httpx.AsyncClient) -> PresenceAssessment:
        """Raw classify call — raises on retryable errors so retry wrapper can handle them."""
        response = await c.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-5-nano",
                "messages": [
                    {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            }
        )
        response.raise_for_status()
        data = response.json()
        
        # Parse the response
        content = data["choices"][0]["message"]["content"]
        result = json_module.loads(content)
        
        return PresenceAssessment(
            company_name=company_name,
            online_presence_score=int(result.get("online_presence_score", 0)),
            research_priority_score=int(result.get("research_priority_score", 0)),
            reasoning=result.get("reasoning", ""),
            user_prompt=user_prompt,
            raw_gpt_response=data,
        )
    
    async def _classify_with_retry(c: httpx.AsyncClient) -> PresenceAssessment:
        """Wrap the classify call with retry logic, catch permanent failures."""
        try:
            return await async_retry(
                _do_classify, c,
                max_retries=PROCESSING.max_retries,
                delay_base=PROCESSING.retry_delay_base,
                operation_name=f"gpt_classify({company_name})",
            )
        except httpx.HTTPStatusError as e:
            return PresenceAssessment(
                company_name=company_name,
                online_presence_score=0,
                research_priority_score=0,
                reasoning="",
                error=f"HTTP {e.response.status_code}: {e.response.text[:100]}",
                user_prompt=user_prompt,
            )
        except json_module.JSONDecodeError as e:
            return PresenceAssessment(
                company_name=company_name,
                online_presence_score=0,
                research_priority_score=0,
                reasoning="",
                error=f"Failed to parse GPT response: {str(e)}",
                user_prompt=user_prompt,
            )
        except Exception as e:
            return PresenceAssessment(
                company_name=company_name,
                online_presence_score=0,
                research_priority_score=0,
                reasoning="",
                error=f"{type(e).__name__}: {str(e)[:100]}",
                user_prompt=user_prompt,
            )
    
    if client is not None:
        return await _classify_with_retry(client)
    
    # Fallback: create a one-off client (backward compat for simple scripts)
    async with httpx.AsyncClient(timeout=PROCESSING.openai_timeout) as new_client:
        return await _classify_with_retry(new_client)


def classify_company_sync(
    company_name: str,
    company_description: Optional[str],
    website_status: WebsiteStatus,
    search_result: TavilySearchResult,
    api_key: Optional[str] = None,
    homepage_url: Optional[str] = None,
    long_description: Optional[str] = None,
) -> PresenceAssessment:
    """Synchronous wrapper for classify_company (creates its own client)."""
    return asyncio.run(classify_company(
        company_name, company_description, website_status, search_result, api_key,
        homepage_url=homepage_url, long_description=long_description,
    ))


# Need asyncio for the sync wrapper
import asyncio
