"""
Output Writer

Generates the final CSV dataset from pipeline results.

Schema:
- Every company gets at least one row
- Companies with multiple findings get multiple rows
- No findings = one row with genai_adoption_found=False
"""

import csv
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator

from ..config import OUTPUT_DIR
from ..stage_2.perplexity_client import GenAIFinding, ResearchResult


# ─────────────────────────────────────────────────────────────────────────────
# CSV SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    # Company identifiers
    "company_id",
    "company_name",
    "homepage_url",
    "industry",
    "founded_year",
    
    # Research metadata
    "research_stage_reached",
    "online_presence_score",
    "website_alive",
    "research_cost_usd",
    "researched_at",
    
    # Finding data (one per row)
    "finding_id",
    "genai_adoption_found",
    "adoption_confidence",
    "tool_name",
    "use_case",
    "business_function",
    "evidence_summary",
    "source_url",
    "source_type",
    "no_finding_reason",
]


@dataclass
class CSVRow:
    """
    A single row in the output CSV.
    
    Maps directly to CSV_COLUMNS.
    """
    # Company identifiers
    company_id: int
    company_name: str
    homepage_url: Optional[str] = None
    industry: Optional[str] = None
    founded_year: Optional[int] = None
    
    # Research metadata
    research_stage_reached: str = ""
    online_presence_score: Optional[int] = None
    website_alive: Optional[bool] = None
    research_cost_usd: float = 0.0
    researched_at: Optional[str] = None
    
    # Finding data
    finding_id: int = 1
    genai_adoption_found: bool = False
    adoption_confidence: float = 0.0
    tool_name: Optional[str] = None
    use_case: Optional[str] = None
    business_function: Optional[str] = None
    evidence_summary: Optional[str] = None
    source_url: Optional[str] = None
    source_type: Optional[str] = None
    no_finding_reason: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for CSV writing."""
        return asdict(self)


@dataclass
class CompanyResearchResult:
    """
    Complete research result for a single company.
    
    Contains all metadata plus findings from the pipeline.
    """
    company_id: int
    company_name: str
    homepage_url: Optional[str] = None
    industry: Optional[str] = None
    founded_year: Optional[int] = None
    
    research_stage_reached: str = "1"
    online_presence_score: Optional[int] = None
    website_alive: Optional[bool] = None
    research_cost_usd: float = 0.0
    
    # The parsed research result
    result: Optional[ResearchResult] = None
    
    def to_csv_rows(self) -> list[CSVRow]:
        """
        Expand this result into CSV rows.
        
        - If findings exist: one row per finding
        - If no findings: one row with genai_adoption_found=False
        """
        now = datetime.utcnow().isoformat()
        
        # Base row data (shared across all rows for this company)
        base = {
            "company_id": self.company_id,
            "company_name": self.company_name,
            "homepage_url": self.homepage_url,
            "industry": self.industry,
            "founded_year": self.founded_year,
            "research_stage_reached": self.research_stage_reached,
            "online_presence_score": self.online_presence_score,
            "website_alive": self.website_alive,
            "research_cost_usd": self.research_cost_usd,
            "researched_at": now,
        }
        
        rows = []
        
        if self.result and self.result.findings:
            # One row per finding
            for i, finding in enumerate(self.result.findings, start=1):
                rows.append(CSVRow(
                    **base,
                    finding_id=i,
                    genai_adoption_found=True,
                    adoption_confidence=finding.confidence,
                    tool_name=finding.tool_name,
                    use_case=finding.use_case,
                    business_function=finding.business_function,
                    evidence_summary=finding.evidence_summary,
                    source_url=finding.source_url,
                    source_type=finding.source_type,
                    no_finding_reason=None,
                ))
        else:
            # No findings - single row
            no_finding_reason = None
            if self.result:
                no_finding_reason = self.result.no_finding_reason
            
            rows.append(CSVRow(
                **base,
                finding_id=1,
                genai_adoption_found=False,
                adoption_confidence=0.0,
                tool_name=None,
                use_case=None,
                business_function=None,
                evidence_summary=None,
                source_url=None,
                source_type=None,
                no_finding_reason=no_finding_reason or "no_evidence",
            ))
        
        return rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV WRITER
# ─────────────────────────────────────────────────────────────────────────────

class OutputWriter:
    """
    Writes research results to CSV files.
    
    Supports streaming writes for large datasets.
    """
    
    def __init__(self, output_path: Optional[Path] = None):
        """
        Initialize the writer.
        
        Args:
            output_path: Path to output CSV file. Defaults to outputs/genai_adoption_findings.csv
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_DIR / f"genai_adoption_findings_{timestamp}.csv"
        
        self.output_path = output_path
        self._file = None
        self._writer = None
        self._row_count = 0
        self._company_count = 0
    
    def __enter__(self) -> "OutputWriter":
        """Open file for writing."""
        self._file = open(self.output_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close file."""
        if self._file:
            self._file.close()
    
    def write_result(self, result: CompanyResearchResult) -> int:
        """
        Write a company's research result to CSV.
        
        Args:
            result: The company research result to write.
        
        Returns:
            Number of rows written (1 for no findings, N for N findings).
        """
        rows = result.to_csv_rows()
        
        for row in rows:
            self._writer.writerow(row.to_dict())
            self._row_count += 1
        
        self._company_count += 1
        return len(rows)
    
    def write_results(self, results: Iterator[CompanyResearchResult]) -> tuple[int, int]:
        """
        Write multiple results.
        
        Args:
            results: Iterator of company research results.
        
        Returns:
            Tuple of (companies_written, rows_written).
        """
        for result in results:
            self.write_result(result)
        
        return self._company_count, self._row_count
    
    @property
    def row_count(self) -> int:
        """Total rows written."""
        return self._row_count
    
    @property
    def company_count(self) -> int:
        """Total companies written."""
        return self._company_count


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def write_results_to_csv(
    results: list[CompanyResearchResult],
    output_path: Optional[Path] = None
) -> Path:
    """
    Write a list of results to CSV.
    
    Args:
        results: List of company research results.
        output_path: Path to output file (auto-generated if not provided).
    
    Returns:
        Path to the written CSV file.
    """
    with OutputWriter(output_path) as writer:
        writer.write_results(iter(results))
        print(f"Wrote {writer.company_count} companies ({writer.row_count} rows) to {writer.output_path}")
    
    return writer.output_path


def create_sample_csv(output_path: Optional[Path] = None) -> Path:
    """
    Create a sample CSV with mock data for testing.
    
    Args:
        output_path: Path to output file.
    
    Returns:
        Path to the written CSV file.
    """
    from ..stage_2.perplexity_client import GenAIFinding, ResearchResult
    
    # Sample data
    samples = [
        # Company with multiple findings
        CompanyResearchResult(
            company_id=12345,
            company_name="TechCorp Inc",
            homepage_url="https://techcorp.com",
            industry="Software",
            founded_year=2020,
            research_stage_reached="2B",
            online_presence_score=75,
            website_alive=True,
            research_cost_usd=0.08,
            result=ResearchResult(
                genai_adoption_found=True,
                findings=[
                    GenAIFinding(
                        tool_name="ChatGPT",
                        use_case="customer_support",
                        business_function="Customer Service",
                        evidence_summary="Uses ChatGPT for automated ticket responses",
                        source_url="https://techcorp.com/blog/ai-support",
                        source_type="company_blog",
                        confidence=0.85,
                    ),
                    GenAIFinding(
                        tool_name="Copilot",
                        use_case="code_generation",
                        business_function="Engineering",
                        evidence_summary="Engineering team adopted GitHub Copilot",
                        source_url="https://linkedin.com/in/techcorp-cto",
                        source_type="linkedin",
                        confidence=0.7,
                    ),
                ],
            ),
        ),
        # Company with one finding
        CompanyResearchResult(
            company_id=67890,
            company_name="MarketingPro",
            homepage_url="https://marketingpro.io",
            industry="Marketing",
            founded_year=2019,
            research_stage_reached="2A",
            online_presence_score=60,
            website_alive=True,
            research_cost_usd=0.02,
            result=ResearchResult(
                genai_adoption_found=True,
                findings=[
                    GenAIFinding(
                        tool_name="ChatGPT",
                        use_case="content_creation",
                        business_function="Marketing",
                        evidence_summary="Uses ChatGPT for blog content drafts",
                        source_url="https://news.example.com/marketingpro-ai",
                        source_type="news",
                        confidence=0.6,
                    ),
                ],
            ),
        ),
        # Company with no findings
        CompanyResearchResult(
            company_id=11111,
            company_name="Traditional Corp",
            homepage_url="https://traditional.com",
            industry="Manufacturing",
            founded_year=2015,
            research_stage_reached="2A",
            online_presence_score=40,
            website_alive=True,
            research_cost_usd=0.02,
            result=ResearchResult(
                genai_adoption_found=False,
                findings=[],
                no_finding_reason="no_evidence",
            ),
        ),
        # Company with insufficient information
        CompanyResearchResult(
            company_id=22222,
            company_name="Ghost Startup",
            homepage_url=None,
            industry="Unknown",
            founded_year=2022,
            research_stage_reached="1",
            online_presence_score=10,
            website_alive=False,
            research_cost_usd=0.011,
            result=ResearchResult(
                genai_adoption_found=False,
                findings=[],
                no_finding_reason="insufficient_information",
            ),
        ),
    ]
    
    return write_results_to_csv(samples, output_path)
