"""
Perplexity Sonar API Client

Unified client for all Perplexity research tiers:
- Sonar Base: Quick checks (Stage 2A)
- Sonar Pro: Deeper research (Stage 2B)  
- Deep Research: Comprehensive investigation (Stage 3)

"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import httpx

from ..config import PROCESSING, COSTS, PROMPTS_DIR, APIKeys


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts folder."""
    prompt_path = PROMPTS_DIR / filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text()


# ─────────────────────────────────────────────────────────────────────────────
# MODELS & TIERS
# ─────────────────────────────────────────────────────────────────────────────

class SonarModel(str, Enum):
    """Available Perplexity Sonar models."""
    # Base tier - fast and cheap
    SONAR_SMALL = "llama-3.1-sonar-small-128k-online"
    
    # Pro tier - more capable
    SONAR_LARGE = "llama-3.1-sonar-large-128k-online"
    
    # Huge tier - most capable (for deep research)
    SONAR_HUGE = "llama-3.1-sonar-huge-128k-online"


@dataclass
class Citation:
    """A source citation from Perplexity."""
    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None


@dataclass
class GenAIFinding:
    """
    A single GenAI adoption finding from research.
    
    This is the structured output format that maps directly to CSV rows.
    """
    tool_name: Optional[str] = None  # ChatGPT, Claude, Copilot, etc.
    use_case: Optional[str] = None   # customer_support, code_generation, etc.
    business_function: Optional[str] = None  # Engineering, Marketing, HR, etc.
    evidence_summary: Optional[str] = None
    source_url: Optional[str] = None
    source_type: Optional[str] = None  # company_blog, news, job_posting, etc.
    confidence: float = 0.0


@dataclass
class ResearchResult:
    """
    Parsed research result from Perplexity.
    
    Contains structured findings ready for CSV export.
    """
    genai_adoption_found: bool = False
    findings: list[GenAIFinding] = field(default_factory=list)
    no_finding_reason: Optional[str] = None  # "insufficient_information" | "no_evidence"
    recommend_escalation: bool = False  # Should this company be escalated to next stage?
    escalation_reason: Optional[str] = None  # Why escalation is recommended
    raw_response: str = ""
    parse_error: Optional[str] = None
    
    @classmethod
    def from_json(cls, json_str: str) -> "ResearchResult":
        """
        Parse a JSON response into a ResearchResult.
        
        Handles malformed JSON gracefully.
        """
        import json
        
        try:
            # Try to extract JSON from the response (might have extra text)
            json_str = json_str.strip()
            
            # Find JSON object boundaries
            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            
            if start == -1 or end == 0:
                return cls(
                    raw_response=json_str,
                    parse_error="No JSON object found in response"
                )
            
            json_content = json_str[start:end]
            data = json.loads(json_content)
            
            # Parse findings
            findings = []
            for f in data.get("findings", []):
                findings.append(GenAIFinding(
                    tool_name=f.get("tool_name"),
                    use_case=f.get("use_case"),
                    business_function=f.get("business_function"),
                    evidence_summary=f.get("evidence_summary"),
                    source_url=f.get("source_url"),
                    source_type=f.get("source_type"),
                    confidence=float(f.get("confidence", 0.0)),
                ))
            
            return cls(
                genai_adoption_found=bool(data.get("genai_adoption_found", False)),
                findings=findings,
                no_finding_reason=data.get("no_finding_reason"),
                recommend_escalation=bool(data.get("recommend_escalation", False)),
                escalation_reason=data.get("escalation_reason"),
                raw_response=json_str,
            )
            
        except json.JSONDecodeError as e:
            return cls(
                raw_response=json_str,
                parse_error=f"JSON parse error: {str(e)}"
            )
        except Exception as e:
            return cls(
                raw_response=json_str,
                parse_error=f"Parse error: {str(e)}"
            )


@dataclass 
class PerplexityResponse:
    """
    Response from Perplexity Sonar API.
    
    Attributes:
        content: The generated response text.
        citations: List of source URLs/titles that support the response.
        model: Which model was used.
        usage: Token usage for cost tracking.
        error: Error message if request failed.
        parsed: Structured research result (if successfully parsed).
    """
    content: str
    citations: list[Citation] = field(default_factory=list)
    model: str = ""
    usage: dict = field(default_factory=dict)
    error: Optional[str] = None
    parsed: Optional[ResearchResult] = None
    
    def __post_init__(self):
        """Auto-parse content into structured result."""
        if self.content and not self.error and not self.parsed:
            self.parsed = ResearchResult.from_json(self.content)
    
    @property
    def has_citations(self) -> bool:
        """Did the response include any citations?"""
        return len(self.citations) > 0
    
    @property
    def citation_urls(self) -> list[str]:
        """Extract just the URLs from citations."""
        return [c.url for c in self.citations]
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.usage.get("total_tokens", 0)
    
    @property
    def findings(self) -> list[GenAIFinding]:
        """Get parsed findings (empty list if parse failed)."""
        if self.parsed:
            return self.parsed.findings
        return []
    
    @property
    def adoption_found(self) -> bool:
        """Did research find GenAI adoption?"""
        if self.parsed:
            return self.parsed.genai_adoption_found
        return False
    
    @property
    def recommend_escalation(self) -> bool:
        """Should this company be escalated to the next research stage?"""
        if self.parsed:
            return self.parsed.recommend_escalation
        return False
    
    @property
    def escalation_reason(self) -> Optional[str]:
        """Why escalation is recommended (if applicable)."""
        if self.parsed:
            return self.parsed.escalation_reason
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS FOR GENAI ADOPTION RESEARCH
# ─────────────────────────────────────────────────────────────────────────────

def build_sonar_base_prompt(
    company_name: str,
    presence_score: int = 0,
    ai_signals_summary: Optional[str] = None,
    context: Optional[str] = None
) -> str:
    """
    Build prompt for Stage 2A: Quick check with Sonar Base.
    
    Loads template from: prompts/stage_2a_quick_check.txt
    
    Args:
        company_name: Name of the company to research.
        presence_score: Online presence score from Stage 1 (0-100).
        ai_signals_summary: Summary of AI signals found in Stage 1.
        context: Optional context (e.g., Crunchbase description).
    """
    template = _load_prompt("stage_2a_quick_check.txt")
    
    context_section = f"\nCompany description: {context}" if context else ""
    signals_summary = ai_signals_summary if ai_signals_summary else "None found in initial screening"
    
    return template.format(
        company_name=company_name,
        presence_score=presence_score,
        ai_signals_summary=signals_summary,
        context_section=context_section
    )


def build_sonar_pro_prompt(
    company_name: str, 
    homepage_url: Optional[str] = None,
    initial_signals: Optional[list[str]] = None,
    presence_score: int = 0,
    escalation_reason: Optional[str] = None,
    context: Optional[str] = None
) -> str:
    """
    Build prompt for Stage 2B: Deeper research with Sonar Pro.
    
    Loads template from: prompts/stage_2b_deep_check.txt
    
    Args:
        company_name: Name of the company to research.
        homepage_url: Company website for context.
        initial_signals: AI signals found in Stage 1.
        presence_score: Online presence score from Stage 1 (0-100).
        escalation_reason: Why this company was escalated to Stage 2B.
        context: Optional context (e.g., Crunchbase description).
    """
    template = _load_prompt("stage_2b_deep_check.txt")
    
    url_hint = f" (website: {homepage_url})" if homepage_url else ""
    
    if initial_signals:
        signals_section = "\n".join(f"- {s}" for s in initial_signals)
    else:
        signals_section = "- General AI adoption signals detected"
    
    escalation = escalation_reason if escalation_reason else "Promising signals found in initial screening"
    context_section = f"\n\nCompany description: {context}" if context else ""
    
    return template.format(
        company_name=company_name,
        url_hint=url_hint,
        signals_section=signals_section,
        presence_score=presence_score,
        escalation_reason=escalation,
        context_section=context_section
    )


def build_deep_research_prompt(
    company_name: str,
    homepage_url: Optional[str] = None,
    previous_findings: Optional[str] = None,
    context: Optional[str] = None
) -> str:
    """
    Build prompt for Stage 3: Deep Research.
    
    Loads template from: prompts/stage_3_deep_research.txt
    
    Args:
        company_name: Name of the company to research.
        homepage_url: Company website.
        previous_findings: What earlier stages found (to verify and expand).
        context: Company background information.
    """
    template = _load_prompt("stage_3_deep_research.txt")
    
    url_hint = f" (website: {homepage_url})" if homepage_url else ""
    
    # Format previous findings for the template
    if previous_findings:
        previous_section = previous_findings
    else:
        previous_section = "Initial stages found promising AI adoption signals but lacked specific details."
    
    context_section = f"\n\nCompany background: {context}" if context else ""
    
    return template.format(
        company_name=company_name,
        url_hint=url_hint,
        previous_section=previous_section,
        context_section=context_section
    )


# ─────────────────────────────────────────────────────────────────────────────
# API CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class PerplexityClient:
    """
    Async client for Perplexity Sonar API.
    
    Supports all three research tiers with appropriate prompts
    and cost tracking.
    """
    
    BASE_URL = "https://api.perplexity.ai/chat/completions"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the client.
        
        Args:
            api_key: Perplexity API key. Uses credentials folder or env var if not provided.
        """
        if api_key:
            self.api_key = api_key
        else:
            keys = APIKeys()
            self.api_key = keys.perplexity
    
    def _validate_api_key(self) -> Optional[str]:
        """Check if API key is set, return error message if not."""
        if not self.api_key:
            return "Perplexity API key not set. Add to credentials/perplexity_api_key.txt"
        return None
    
    async def _make_request(
        self,
        model: SonarModel,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024
    ) -> PerplexityResponse:
        """
        Make a request to the Perplexity API.
        
        Args:
            model: Which Sonar model to use.
            system_prompt: System message for the model.
            user_prompt: User message (the research query).
            temperature: Sampling temperature (lower = more focused).
            max_tokens: Maximum response length.
        
        Returns:
            PerplexityResponse with content and citations.
        """
        error = self._validate_api_key()
        if error:
            return PerplexityResponse(content="", error=error)
        
        async with httpx.AsyncClient(timeout=PROCESSING.openai_timeout) as client:
            try:
                response = await client.post(
                    self.BASE_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model.value,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "return_citations": True,
                        "search_recency_filter": "month",  # Prefer recent results
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract content
                content = data["choices"][0]["message"]["content"]
                
                # Extract citations
                citations = []
                raw_citations = data.get("citations", [])
                for cite in raw_citations:
                    if isinstance(cite, str):
                        citations.append(Citation(url=cite))
                    elif isinstance(cite, dict):
                        citations.append(Citation(
                            url=cite.get("url", ""),
                            title=cite.get("title"),
                            snippet=cite.get("snippet"),
                        ))
                
                return PerplexityResponse(
                    content=content,
                    citations=citations,
                    model=model.value,
                    usage=data.get("usage", {}),
                )
                
            except httpx.HTTPStatusError as e:
                return PerplexityResponse(
                    content="",
                    error=f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                )
            except Exception as e:
                return PerplexityResponse(
                    content="",
                    error=f"{type(e).__name__}: {str(e)[:200]}"
                )
    
    # ─────────────────────────────────────────────────────────────────────────
    # TIER-SPECIFIC METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    async def quick_check(
        self,
        company_name: str,
        presence_score: int = 0,
        ai_signals_summary: Optional[str] = None,
        context: Optional[str] = None
    ) -> PerplexityResponse:
        """
        Stage 2A: Quick check with Sonar Base.
        
        Fast, cheap check for any GenAI adoption signals.
        Cost: ~$0.02
        
        Args:
            company_name: Company to research.
            presence_score: Online presence score from Stage 1 (0-100).
            ai_signals_summary: Summary of AI signals found in Stage 1.
            context: Optional context (e.g., Crunchbase description).
        
        Returns:
            PerplexityResponse with findings.
        """
        system_prompt = (
            "You are a research assistant investigating whether companies "
            "use generative AI tools in their business operations. "
            "Be concise and factual. Cite your sources. Respond with valid JSON only."
        )
        
        user_prompt = build_sonar_base_prompt(
            company_name=company_name,
            presence_score=presence_score,
            ai_signals_summary=ai_signals_summary,
            context=context
        )
        
        return await self._make_request(
            model=SonarModel.SONAR_SMALL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=768,  # Increased for new prompt structure
        )
    
    async def deep_check(
        self,
        company_name: str,
        homepage_url: Optional[str] = None,
        initial_signals: Optional[list[str]] = None,
        presence_score: int = 0,
        escalation_reason: Optional[str] = None,
        context: Optional[str] = None
    ) -> PerplexityResponse:
        """
        Stage 2B: Deeper research with Sonar Pro.
        
        More thorough investigation with better source coverage.
        Cost: ~$0.05-0.10
        
        Args:
            company_name: Company to research.
            homepage_url: Company website for context.
            initial_signals: AI signals found in Stage 1.
            presence_score: Online presence score from Stage 1 (0-100).
            escalation_reason: Why this company was escalated to Stage 2B.
            context: Optional additional context.
        
        Returns:
            PerplexityResponse with detailed findings and citations.
        """
        system_prompt = (
            "You are a thorough research assistant investigating generative AI "
            "adoption in business operations. Find specific, verifiable evidence "
            "with citations. Distinguish between companies that USE AI tools "
            "versus companies that BUILD AI products. Respond with valid JSON only."
        )
        
        user_prompt = build_sonar_pro_prompt(
            company_name=company_name,
            homepage_url=homepage_url,
            initial_signals=initial_signals,
            presence_score=presence_score,
            escalation_reason=escalation_reason,
            context=context
        )
        
        return await self._make_request(
            model=SonarModel.SONAR_LARGE,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=1280,  # Increased for new prompt structure
        )
    
    async def deep_research(
        self,
        company_name: str,
        homepage_url: Optional[str] = None,
        previous_findings: Optional[str] = None,
        context: Optional[str] = None
    ) -> PerplexityResponse:
        """
        Stage 3: Comprehensive deep research with Sonar Huge.
        
        Most thorough investigation for high-value targets.
        Cost: ~$0.41-1.19
        
        Args:
            company_name: Company to research.
            homepage_url: Company website.
            previous_findings: What earlier stages found.
            context: Company background information.
        
        Returns:
            PerplexityResponse with comprehensive findings.
        """
        system_prompt = (
            "You are an expert research analyst conducting comprehensive due diligence "
            "on a company's adoption of generative AI tools for academic research. "
            "Investigate thoroughly, verify claims with multiple sources, and provide "
            "detailed findings with confidence levels. Focus on INTERNAL AI tool usage, "
            "not AI products they sell. Respond with valid JSON only."
        )
        
        user_prompt = build_deep_research_prompt(
            company_name=company_name,
            homepage_url=homepage_url,
            previous_findings=previous_findings,
            context=context
        )
        
        return await self._make_request(
            model=SonarModel.SONAR_HUGE,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=2560,  # Increased for comprehensive output
        )


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

# Module-level client instance
_client: Optional[PerplexityClient] = None


def get_client(api_key: Optional[str] = None) -> PerplexityClient:
    """Get or create a Perplexity client instance."""
    global _client
    if _client is None or api_key:
        _client = PerplexityClient(api_key)
    return _client


async def quick_check(
    company_name: str,
    presence_score: int = 0,
    ai_signals_summary: Optional[str] = None,
    context: Optional[str] = None
) -> PerplexityResponse:
    """Stage 2A: Quick check (async)."""
    return await get_client().quick_check(
        company_name=company_name,
        presence_score=presence_score,
        ai_signals_summary=ai_signals_summary,
        context=context
    )


async def deep_check(
    company_name: str,
    homepage_url: Optional[str] = None,
    initial_signals: Optional[list[str]] = None,
    presence_score: int = 0,
    escalation_reason: Optional[str] = None,
    context: Optional[str] = None
) -> PerplexityResponse:
    """Stage 2B: Deep check (async)."""
    return await get_client().deep_check(
        company_name=company_name,
        homepage_url=homepage_url,
        initial_signals=initial_signals,
        presence_score=presence_score,
        escalation_reason=escalation_reason,
        context=context
    )


async def deep_research(
    company_name: str,
    homepage_url: Optional[str] = None,
    previous_findings: Optional[str] = None,
    context: Optional[str] = None
) -> PerplexityResponse:
    """Stage 3: Deep research (async)."""
    return await get_client().deep_research(
        company_name=company_name,
        homepage_url=homepage_url,
        previous_findings=previous_findings,
        context=context
    )


# Sync wrappers
def quick_check_sync(
    company_name: str,
    presence_score: int = 0,
    ai_signals_summary: Optional[str] = None,
    context: Optional[str] = None
) -> PerplexityResponse:
    """Stage 2A: Quick check (sync)."""
    return asyncio.run(quick_check(
        company_name=company_name,
        presence_score=presence_score,
        ai_signals_summary=ai_signals_summary,
        context=context
    ))


def deep_check_sync(
    company_name: str,
    homepage_url: Optional[str] = None,
    initial_signals: Optional[list[str]] = None,
    presence_score: int = 0,
    escalation_reason: Optional[str] = None,
    context: Optional[str] = None
) -> PerplexityResponse:
    """Stage 2B: Deep check (sync)."""
    return asyncio.run(deep_check(
        company_name=company_name,
        homepage_url=homepage_url,
        initial_signals=initial_signals,
        presence_score=presence_score,
        escalation_reason=escalation_reason,
        context=context
    ))


def deep_research_sync(
    company_name: str,
    homepage_url: Optional[str] = None,
    previous_findings: Optional[str] = None,
    context: Optional[str] = None
) -> PerplexityResponse:
    """Stage 3: Deep research (sync)."""
    return asyncio.run(deep_research(
        company_name=company_name,
        homepage_url=homepage_url,
        previous_findings=previous_findings,
        context=context
    ))
