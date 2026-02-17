"""
Configuration & Constants

All magic numbers, thresholds, and settings live here.
Single source of truth for the entire pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
import os


# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "crunchbase_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CREDENTIALS_DIR = PROJECT_ROOT / "credentials"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Stage-specific output directories
STAGE1_OUTPUT_DIR = OUTPUT_DIR / "stage1"
STAGE1_TAVILY_DIR = STAGE1_OUTPUT_DIR / "tavily"
STAGE1_GPT_DIR = STAGE1_OUTPUT_DIR / "gpt"
STAGE2_OUTPUT_DIR = OUTPUT_DIR / "stage2"
STAGE3_OUTPUT_DIR = OUTPUT_DIR / "stage3"

# Ensure directories exist
for dir_path in [OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR,
                 STAGE1_OUTPUT_DIR, STAGE1_TAVILY_DIR, STAGE1_GPT_DIR,
                 STAGE2_OUTPUT_DIR, STAGE3_OUTPUT_DIR]:
    dir_path.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CREDENTIAL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _load_credential(filename: str) -> str:
    """
    Load an API key from a credential file.
    
    Falls back to environment variable if file doesn't exist.
    
    Args:
        filename: Name of the credential file (e.g., 'tavily_api_key.txt')
    
    Returns:
        The API key string, or empty string if not found.
    """
    # Try loading from file first
    cred_path = CREDENTIALS_DIR / filename
    if cred_path.exists():
        content = cred_path.read_text().strip()
        # Skip comment lines and empty lines
        lines = [line.strip() for line in content.split('\n') 
                 if line.strip() and not line.strip().startswith('#')]
        if lines:
            return lines[0]  # Return first non-comment line
    
    # Fall back to environment variable
    env_var = filename.replace('_api_key.txt', '').upper() + '_API_KEY'
    return os.getenv(env_var, "")


# ─────────────────────────────────────────────────────────────────────────────
# API KEYS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class APIKeys:
    """
    API credentials loaded from credentials folder or environment variables.
    
    Priority:
    1. credentials/<service>_api_key.txt file
    2. Environment variable (e.g., TAVILY_API_KEY)
    """
    tavily: str = ""
    openai: str = ""
    perplexity: str = ""
    
    def __post_init__(self):
        """Load credentials if not already set."""
        if not self.tavily:
            object.__setattr__(self, 'tavily', _load_credential('tavily_api_key.txt'))
        if not self.openai:
            object.__setattr__(self, 'openai', _load_credential('openai_api_key.txt'))
        if not self.perplexity:
            object.__setattr__(self, 'perplexity', _load_credential('perplexity_api_key.txt'))

    def validate(self) -> list[str]:
        """Return list of missing API keys."""
        missing = []
        if not self.tavily:
            missing.append("tavily (credentials/tavily_api_key.txt)")
        if not self.openai:
            missing.append("openai (credentials/openai_api_key.txt)")
        if not self.perplexity:
            missing.append("perplexity (credentials/perplexity_api_key.txt)")
        return missing
    
    def status(self) -> dict[str, bool]:
        """Return status of each API key."""
        return {
            "tavily": bool(self.tavily),
            "openai": bool(self.openai),
            "perplexity": bool(self.perplexity),
        }


# ─────────────────────────────────────────────────────────────────────────────
# COST TRACKING
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CostEstimates:
    """
    Per-query cost estimates for budget tracking.
    Values in USD.
    
    Note: These are conservative estimates. Actual costs depend on:
    - API pricing tier (volume discounts available)
    - Token usage (varies by company description length)
    - Search result complexity
    
    Last updated: February 2026
    """
    # Stage 1: Tavily search + GPT-5-nano interpretation
    # Tavily: ~$0.02/search (advanced depth, 2 credits)
    # GPT-5-nano: ~$0.05/1M input tokens, ~$0.40/1M output tokens
    tavily_search: float = 0.02       # Advanced depth = 2 credits
    gpt5_nano_call: float = 0.0002    # ~800 input + ~600 completion (incl. reasoning)
    stage_1_total: float = 0.020      # Tavily dominates the cost

    # Stage 2A: Perplexity Sonar Base (llama-3.1-sonar-small)
    # $0.20/1M input, $0.20/1M output + $5/1000 searches
    sonar_base: float = 0.02

    # Stage 2B: Perplexity Sonar Pro (llama-3.1-sonar-large)  
    # $1/1M input, $1/1M output + $5/1000 searches
    sonar_pro_min: float = 0.05
    sonar_pro_max: float = 0.10

    # Stage 3: Perplexity Deep Research
    # Significantly more expensive - multiple searches + reasoning
    deep_research_min: float = 0.41
    deep_research_max: float = 1.19

    # Budget constraints
    target_avg_per_company: float = 0.10
    hard_budget_limit_total: float = 5000.0  # $5k ceiling for 44k companies


# ─────────────────────────────────────────────────────────────────────────────
# ESCALATION THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EscalationThresholds:
    """
    Decision thresholds for routing companies through the pipeline.
    
    Research priority score (0-5) determines routing:
    - 0-2: No deep research (not worth it)
    - 3-5: Proceed to deep research
    """
    # Research priority score threshold for deep research
    deep_research_min_score: int = 3  # Score >= 3 triggers deep research


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProcessingConfig:
    """Settings for batch processing and rate limiting."""
    # Concurrency
    max_concurrent_requests: int = 10
    batch_size: int = 100

    # Timeouts (seconds)
    http_timeout: float = 10.0       # Website health checks
    tavily_timeout: float = 30.0     # Tavily API calls
    openai_timeout: float = 120.0    # GPT-5-nano needs headroom for reasoning tokens

    # Rate limiting (requests per minute) — set to 95% of actual limits for safety
    tavily_rpm: int = 950       # Actual limit: 1000 RPM
    openai_rpm: int = 28500     # Actual limit: 30,000 RPM (gpt-5-nano)
    perplexity_rpm: int = 60

    # Checkpointing
    checkpoint_every: int = 100  # Save progress every N companies

    # Retry policy
    max_retries: int = 3
    retry_delay_base: float = 1.0  # Exponential backoff base


# ─────────────────────────────────────────────────────────────────────────────
# FINDING TYPES
# ─────────────────────────────────────────────────────────────────────────────

FINDING_TYPES = [
    # Customer-facing
    "chatgpt_customer_support",
    "ai_chatbot_general",

    # Engineering & Development
    "copilot_engineering",
    "ai_code_generation",
    "ai_testing_qa",

    # Marketing & Content
    "ai_content_generation",
    "ai_marketing_automation",
    "ai_copywriting",

    # Operations
    "ai_document_processing",
    "ai_data_analysis",
    "ai_workflow_automation",

    # HR & Recruiting
    "ai_recruiting_screening",
    "ai_hr_operations",

    # Sales
    "ai_sales_outreach",
    "ai_lead_scoring",

    # Other
    "ai_internal_tools_general",
    "ai_adoption_unspecified",

    # Negative
    "no_adoption_found",
    "insufficient_information",
]


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT INSTANCES
# ─────────────────────────────────────────────────────────────────────────────

# Create singleton instances for easy import
COSTS = CostEstimates()
THRESHOLDS = EscalationThresholds()
PROCESSING = ProcessingConfig()


def load_api_keys() -> APIKeys:
    """Load and validate API keys from environment."""
    keys = APIKeys()
    missing = keys.validate()
    if missing:
        print(f"⚠️  Missing API keys: {', '.join(missing)}")
        print("   Set them in .env or export as environment variables.")
    return keys
