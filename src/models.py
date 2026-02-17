"""
Data Models

Type-safe structures for companies, findings, and pipeline state.
Using Pydantic for validation and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, HttpUrl


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class ResearchStage(str, Enum):
    """Which stage of the pipeline processed this company."""
    STAGE_1 = "1"
    STAGE_2A = "2A"
    STAGE_2B = "2B"
    STAGE_3 = "3"


class PresenceTier(str, Enum):
    """Online presence classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FindingType(str, Enum):
    """Categories of GenAI adoption evidence."""
    # Customer-facing
    CHATGPT_CUSTOMER_SUPPORT = "chatgpt_customer_support"
    AI_CHATBOT_GENERAL = "ai_chatbot_general"

    # Engineering & Development
    COPILOT_ENGINEERING = "copilot_engineering"
    AI_CODE_GENERATION = "ai_code_generation"
    AI_TESTING_QA = "ai_testing_qa"

    # Marketing & Content
    AI_CONTENT_GENERATION = "ai_content_generation"
    AI_MARKETING_AUTOMATION = "ai_marketing_automation"
    AI_COPYWRITING = "ai_copywriting"

    # Operations
    AI_DOCUMENT_PROCESSING = "ai_document_processing"
    AI_DATA_ANALYSIS = "ai_data_analysis"
    AI_WORKFLOW_AUTOMATION = "ai_workflow_automation"

    # HR & Recruiting
    AI_RECRUITING_SCREENING = "ai_recruiting_screening"
    AI_HR_OPERATIONS = "ai_hr_operations"

    # Sales
    AI_SALES_OUTREACH = "ai_sales_outreach"
    AI_LEAD_SCORING = "ai_lead_scoring"

    # Other
    AI_INTERNAL_TOOLS_GENERAL = "ai_internal_tools_general"
    AI_ADOPTION_UNSPECIFIED = "ai_adoption_unspecified"

    # Negative
    NO_ADOPTION_FOUND = "no_adoption_found"
    INSUFFICIENT_INFORMATION = "insufficient_information"


# ─────────────────────────────────────────────────────────────────────────────
# INPUT MODELS
# ─────────────────────────────────────────────────────────────────────────────

class Company(BaseModel):
    """
    A startup from the Crunchbase dataset.

    This is our input — one row from the CSV.
    """
    rcid: int
    org_uuid: str
    name: str
    cb_url: HttpUrl
    homepage_url: Optional[HttpUrl] = None
    short_description: Optional[str] = None
    category_list: Optional[str] = None
    category_groups_list: Optional[str] = None
    created_date: Optional[str] = None
    founded_date: Optional[str] = None
    description: Optional[str] = None

    @property
    def categories(self) -> list[str]:
        """Parse category_list into a list."""
        if not self.category_list:
            return []
        return [c.strip() for c in self.category_list.split(",")]

    @property
    def category_groups(self) -> list[str]:
        """Parse category_groups_list into a list."""
        if not self.category_groups_list:
            return []
        return [c.strip() for c in self.category_groups_list.split(",")]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

class PresenceFilterResult(BaseModel):
    """
    Output from Stage 1: Presence Filter.

    Determines the routing for subsequent stages.
    """
    company_id: int
    company_name: str

    # Website health check
    homepage_alive: bool = False
    homepage_status_code: Optional[int] = None

    # Presence assessment
    online_presence_score: int = Field(ge=0, le=100)
    presence_tier: PresenceTier

    # AI signal detection
    ai_mentions_found: bool = False
    ai_mentions_snippets: list[str] = Field(default_factory=list)

    # Routing decision
    next_stage: ResearchStage
    reasoning: str

    # Cost tracking
    cost_incurred: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# FINDING MODELS
# ─────────────────────────────────────────────────────────────────────────────

class Finding(BaseModel):
    """
    A single piece of evidence about GenAI adoption.

    One company can have multiple findings.
    """
    company_id: int
    company_name: str

    finding_type: FindingType
    description: str
    source_url: Optional[HttpUrl] = None
    source_title: Optional[str] = None

    confidence_score: float = Field(ge=0.0, le=1.0)
    research_stage: ResearchStage

    # Metadata
    found_at: datetime = Field(default_factory=datetime.utcnow)
    raw_evidence: Optional[str] = None  # Original text from source


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STATE
# ─────────────────────────────────────────────────────────────────────────────

class CompanyResult(BaseModel):
    """
    Complete result for a single company after pipeline processing.

    This is what gets written to the final output.
    """
    company_id: int
    company_name: str
    homepage_url: Optional[str] = None

    # Stage 1 results
    presence_score: Optional[int] = None
    presence_tier: Optional[PresenceTier] = None
    ai_mentions_in_initial_search: bool = False

    # Final results
    max_stage_reached: ResearchStage
    findings: list[Finding] = Field(default_factory=list)
    total_cost: float = 0.0

    # Processing metadata
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = None

    @property
    def has_adoption_evidence(self) -> bool:
        """Did we find any positive adoption evidence?"""
        return any(
            f.finding_type not in [
                FindingType.NO_ADOPTION_FOUND,
                FindingType.INSUFFICIENT_INFORMATION
            ]
            for f in self.findings
        )

    @property
    def highest_confidence(self) -> float:
        """Maximum confidence score across all findings."""
        if not self.findings:
            return 0.0
        return max(f.confidence_score for f in self.findings)


class PipelineCheckpoint(BaseModel):
    """
    Checkpoint for resuming interrupted pipeline runs.
    """
    run_id: str
    started_at: datetime
    last_updated: datetime

    total_companies: int
    processed_count: int
    last_processed_id: int

    total_cost_so_far: float
    findings_count: int

    # Stage distribution
    stage_1_only: int = 0
    stage_2a_count: int = 0
    stage_2b_count: int = 0
    stage_3_count: int = 0

    @property
    def progress_pct(self) -> float:
        """Completion percentage."""
        if self.total_companies == 0:
            return 0.0
        return (self.processed_count / self.total_companies) * 100

    @property
    def avg_cost_per_company(self) -> float:
        """Running average cost."""
        if self.processed_count == 0:
            return 0.0
        return self.total_cost_so_far / self.processed_count
