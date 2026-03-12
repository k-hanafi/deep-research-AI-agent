"""
Production deep-research runner for Stage 2.

Processes companies from the Stage 2 input dataset using the Perplexity
Agent API (deep-research preset). Results accumulate in a single master
JSONL file across runs; a master CSV is regenerated after each run.

Features:
- Async concurrency with configurable parallelism
- Three-state Ctrl+C: PAUSE -> RESUME (Enter) or STOP (Ctrl+C again)
- Incremental CSV: rows appended after every API call
- Resume-safe: skips already-processed companies on restart
- Budget cap: stops new calls when cumulative spend reaches the limit

Usage:
    # Statistical test: 200 per priority
    python -m src.stage_2.production_agent_runner \\
        --sample-size 200 --priorities 5 4 \\
        --concurrency 5 --budget-cap 200

    # Scale to all remaining companies
    python -m src.stage_2.production_agent_runner \\
        --sample-size 0 --priorities 5 4 \\
        --concurrency 10 --budget-cap 1500

    # Dry run (no API calls)
    python -m src.stage_2.production_agent_runner \\
        --sample-size 200 --priorities 5 4 --dry-run
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from perplexity import AsyncPerplexity, AuthenticationError, RateLimitError, InternalServerError, APITimeoutError
from perplexity.types.output_item import MessageOutputItem, SearchResultsOutputItem

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    STAGE2_OUTPUT_DIR, STAGE2_RUNS_DIR,
    STAGE2_MASTER_JSONL, STAGE2_MASTER_CSV,
    PROMPTS_DIR, APIKeys,
)
from src.common import AsyncRateLimiter, AsyncJSONLWriter

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DATASET_PATH = STAGE2_OUTPUT_DIR / "stage2_input_dataset.jsonl"
PROMPT_FILE = PROMPTS_DIR / "stage_2_perplexity_prompt.txt"

DEFAULT_SEED = 2026
DEFAULT_TIMEOUT = 300.0
RPM_LIMIT = 150  # Agent API Tier 1: 150 req/min, 3 QPS
QPS_LIMIT = 3    # Agent API Tier 1: 3 queries per second
MAX_RETRIES = 3
RETRY_BASE_DELAY = 5.0  # seconds; doubles each retry (5s, 10s, 20s)
AUTH_RETRY_DELAY = 30.0  # seconds; for quota reload waits (30s, 60s, 120s)
DEFAULT_PRESET = "deep-research"

logger = logging.getLogger("production_agent_runner")

RETRYABLE_EXCEPTIONS = (RateLimitError, InternalServerError, APITimeoutError)
QUOTA_EXCEPTIONS = (AuthenticationError,)  # retried with longer backoff, fatal if exhausted


# ─────────────────────────────────────────────────────────────────────────────
# QPS LIMITER — enforces max queries per second across concurrent tasks
# ─────────────────────────────────────────────────────────────────────────────

class QPSLimiter:
    """Ensures a minimum interval between requests to respect QPS limits."""

    def __init__(self, qps: int):
        self._interval = 1.0 / qps
        self._lock = asyncio.Lock()
        self._last_request = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Company:
    rcid: int
    name: str
    homepage_url: Optional[str]
    short_description: Optional[str]
    research_priority_score: int
    online_presence_score: int
    category_list: Optional[str]


@dataclass
class ResearchResult:
    rcid: int
    company_name: str
    homepage_url: Optional[str]
    short_description: Optional[str]
    preset: str
    priority: int
    run_id: str
    genai_adoption_found: Optional[bool] = None
    findings_count: int = 0
    findings: list[dict] = field(default_factory=list)
    no_finding_reason: Optional[str] = None
    no_finding_analysis: Optional[str] = None
    cost_usd: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    search_results_count: int = 0
    response_id: Optional[str] = None
    model_used: Optional[str] = None
    response_status: Optional[str] = None
    citations: list[str] = field(default_factory=list)
    error: Optional[str] = None
    fatal: bool = False
    raw_content_preview: Optional[str] = None
    duration_seconds: float = 0.0
    timestamp: str = ""


RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "genai_research_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "company_id": {"type": "integer"},
                "company_name": {"type": "string"},
                "genai_adoption_found": {"type": "boolean"},
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "finding_id": {"type": "integer"},
                            "AI_tool_used": {"type": "string"},
                            "use_case": {"type": "string"},
                            "business_function": {"type": "string"},
                            "evidence_description": {"type": "string"},
                            "source_url": {"type": "string"},
                            "source_type": {"type": "string"},
                        },
                        "required": [
                            "finding_id", "AI_tool_used", "use_case",
                            "business_function", "evidence_description",
                            "source_url", "source_type",
                        ],
                        "additionalProperties": False,
                    },
                },
                "no_finding_reason": {
                    "type": ["string", "null"],
                },
            },
            "required": [
                "company_id", "company_name", "genai_adoption_found",
                "findings", "no_finding_reason",
            ],
            "additionalProperties": False,
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADING & SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> list[Company]:
    companies: list[Company] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            companies.append(Company(
                rcid=rec["rcid"],
                name=rec["name"],
                homepage_url=rec.get("homepage_url"),
                short_description=rec.get("short_description"),
                research_priority_score=rec["research_priority_score"],
                online_presence_score=rec.get("online_presence_score", 0),
                category_list=rec.get("category_list"),
            ))
    return companies


def sample_companies(
    companies: list[Company],
    priority: int,
    sample_size: int,
    seed: int,
) -> list[Company]:
    """Return a deterministic sample for a priority level.

    sample_size=0 means return all companies at this priority.
    """
    filtered = [c for c in companies if c.research_priority_score == priority]
    if not filtered:
        logger.warning("No companies with priority=%d found", priority)
        return []
    if sample_size == 0 or sample_size >= len(filtered):
        return filtered
    rng = random.Random(seed)
    return rng.sample(filtered, sample_size)


def load_completed_rcids(master_path: Path, preset: str) -> set[int]:
    """Scan the master JSONL for rcids already processed successfully."""
    completed: set[int] = set()
    if not master_path.exists():
        return completed
    with open(master_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("preset") == preset and not rec.get("error"):
                    completed.add(rec["rcid"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDING
# ─────────────────────────────────────────────────────────────────────────────

_prompt_template: Optional[str] = None


def get_prompt_template() -> str:
    global _prompt_template
    if _prompt_template is None:
        _prompt_template = PROMPT_FILE.read_text()
    return _prompt_template


def build_prompt(company: Company) -> str:
    return get_prompt_template().format(
        company_id=company.rcid,
        company_name=company.name,
        homepage_url=company.homepage_url or "N/A",
        short_description=company.short_description or "N/A",
    )


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE PARSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_fallback(output: list) -> str:
    """Walk MessageOutputItem content parts to find any text."""
    texts: list[str] = []
    for item in output:
        if isinstance(item, MessageOutputItem):
            for part in (item.content or []):
                text = getattr(part, "text", None)
                if text:
                    texts.append(text)
    return "".join(texts)


def _extract_json_from_text(text: str) -> str:
    """Find the outermost JSON object in text using brace-depth counting."""
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC SINGLE-COMPANY RESEARCH CALL
# ─────────────────────────────────────────────────────────────────────────────

async def _call_api_with_retry(
    client: AsyncPerplexity,
    company: Company,
    preset: str,
    max_steps: Optional[int] = None,
):
    """Call the Perplexity API with exponential backoff on transient errors.

    AuthenticationError (quota exceeded) uses a longer backoff to allow
    auto-reload to replenish credits. If retries exhaust, it re-raises
    so the caller can mark the result as fatal.
    """
    kwargs: dict = dict(
        preset=preset,
        input=build_prompt(company),
        response_format=RESPONSE_SCHEMA,
        timeout=DEFAULT_TIMEOUT,
    )
    if max_steps is not None:
        kwargs["max_steps"] = max_steps

    for attempt in range(MAX_RETRIES + 1):
        try:
            return await client.responses.create(**kwargs)
        except QUOTA_EXCEPTIONS as e:
            if attempt < MAX_RETRIES:
                delay = AUTH_RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    "  %s (rcid=%d): quota exceeded on attempt %d/%d — "
                    "waiting %.0fs for credit reload...",
                    company.name, company.rcid,
                    attempt + 1, MAX_RETRIES + 1, delay,
                )
                await asyncio.sleep(delay)
            else:
                raise
        except RETRYABLE_EXCEPTIONS as e:
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "  %s (rcid=%d): %s on attempt %d/%d, retrying in %.0fs",
                    company.name, company.rcid,
                    type(e).__name__, attempt + 1, MAX_RETRIES + 1, delay,
                )
                await asyncio.sleep(delay)
            else:
                raise


async def research_company(
    client: AsyncPerplexity,
    company: Company,
    preset: str,
    run_id: str,
    max_steps: Optional[int] = None,
) -> ResearchResult:
    start = time.monotonic()
    result = ResearchResult(
        rcid=company.rcid,
        company_name=company.name,
        homepage_url=company.homepage_url,
        short_description=company.short_description,
        preset=preset,
        priority=company.research_priority_score,
        run_id=run_id,
        timestamp=datetime.now().astimezone().isoformat(),
    )

    content = ""
    try:
        response = await _call_api_with_retry(client, company, preset, max_steps=max_steps)

        result.response_id = response.id
        result.model_used = response.model
        result.response_status = response.status

        if response.usage:
            result.input_tokens = response.usage.input_tokens
            result.output_tokens = response.usage.output_tokens
            result.total_tokens = response.usage.total_tokens
            if response.usage.cost:
                result.cost_usd = response.usage.cost.total_cost

        for item in response.output:
            if isinstance(item, SearchResultsOutputItem):
                for sr in (item.results or []):
                    result.citations.append(sr.url)
        result.search_results_count = len(result.citations)

        if response.status == "failed":
            err = response.error
            error_detail = f"{err.type}: {err.message}" if err else "unknown"
            raise ValueError(f"Response failed: {error_detail}")

        content = response.output_text
        if not content:
            content = _extract_text_fallback(response.output)

        content = (content or "").strip()
        if not content:
            output_types = [type(item).__name__ for item in response.output]
            raise ValueError(
                f"Empty response content (model={response.model}, "
                f"status={response.status}, output_types={output_types})"
            )

        result.raw_content_preview = content[:500]
        parsed = json.loads(content)

        result.genai_adoption_found = parsed.get("genai_adoption_found", False)
        result.findings = parsed.get("findings") or []
        result.findings_count = len(result.findings)
        result.no_finding_reason = parsed.get("no_finding_reason")
        result.no_finding_analysis = parsed.get("no_finding_analysis")

    except json.JSONDecodeError as e:
        result.error = f"JSON parse error: {e}"
        logger.debug(
            "Content that failed JSON parsing (%d chars): %.200s",
            len(content), content,
        )
    except QUOTA_EXCEPTIONS as e:
        result.error = f"{type(e).__name__}: {str(e)[:500]}"
        result.fatal = True
    except Exception as e:
        result.error = f"{type(e).__name__}: {str(e)[:500]}"

    result.duration_seconds = round(time.monotonic() - start, 2)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# RUN CONTROLLER — three-state Ctrl+C: RUNNING -> PAUSED -> STOPPING
# ─────────────────────────────────────────────────────────────────────────────

class RunController:
    """Manages pause / resume / stop lifecycle via Ctrl+C signals."""

    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"

    def __init__(self) -> None:
        self._state: str = self.RUNNING
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._in_flight: int = 0
        self._in_flight_lock = asyncio.Lock()

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, _signum: int, _frame) -> None:
        if self._state == self.RUNNING:
            self._state = self.PAUSED
            if self._loop:
                self._loop.call_soon_threadsafe(self._pause_event.clear)
            print(
                "\n"
                "  ┌─────────────────────────────────────────────────┐\n"
                "  │  PAUSED — finishing in-flight calls...          │\n"
                "  │  Press Enter to RESUME, or Ctrl+C to STOP      │\n"
                "  └─────────────────────────────────────────────────┘"
            )
        elif self._state == self.PAUSED:
            self._state = self.STOPPING
            if self._loop:
                self._loop.call_soon_threadsafe(self._pause_event.set)
            print(
                "\n"
                "  ┌─────────────────────────────────────────────────┐\n"
                "  │  STOPPING — will exit after summary             │\n"
                "  └─────────────────────────────────────────────────┘"
            )
        else:
            print("\n  Forcing exit.")
            os._exit(1)

    def resume(self) -> None:
        if self._state == self.PAUSED:
            self._state = self.RUNNING
            self._pause_event.set()
            print(
                "  ┌─────────────────────────────────────────────────┐\n"
                "  │  RESUMED                                        │\n"
                "  └─────────────────────────────────────────────────┘"
            )

    async def wait_if_paused(self) -> None:
        """Block until unpaused. Returns immediately if running or stopping."""
        await self._pause_event.wait()

    async def track_in_flight(self, delta: int) -> None:
        async with self._in_flight_lock:
            self._in_flight += delta

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def should_stop(self) -> bool:
        return self._state == self.STOPPING

    @property
    def is_paused(self) -> bool:
        return self._state == self.PAUSED

    @property
    def is_running(self) -> bool:
        return self._state == self.RUNNING


async def stdin_listener(controller: RunController) -> None:
    """Background task: reads Enter key to resume from pause."""
    loop = asyncio.get_event_loop()
    while not controller.should_stop:
        if controller.is_paused:
            try:
                await loop.run_in_executor(None, sys.stdin.readline)
                if controller.is_paused:
                    controller.resume()
            except (EOFError, OSError):
                break
        else:
            await asyncio.sleep(0.3)


# ─────────────────────────────────────────────────────────────────────────────
# PROGRESS TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class ProgressTracker:
    def __init__(self, total: int):
        self.total = total
        self.processed = 0
        self.errors = 0
        self.found = 0
        self.not_found = 0
        self.total_findings = 0
        self.total_cost = 0.0
        self.costs: list[float] = []
        self.durations: list[float] = []
        self._start = time.monotonic()
        self._lock = asyncio.Lock()

    async def record(self, result: ResearchResult) -> None:
        async with self._lock:
            self.processed += 1
            self.durations.append(result.duration_seconds)
            if result.error:
                self.errors += 1
            else:
                if result.genai_adoption_found:
                    self.found += 1
                else:
                    self.not_found += 1
                self.total_findings += result.findings_count
                cost = result.cost_usd or 0.0
                self.total_cost += cost
                self.costs.append(cost)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    @property
    def successful(self) -> int:
        return self.processed - self.errors

    @property
    def hit_rate(self) -> float:
        return (100 * self.total_findings / self.processed) if self.processed else 0.0

    @property
    def avg_cost(self) -> float:
        return (self.total_cost / len(self.costs)) if self.costs else 0.0

    @property
    def eta_seconds(self) -> float:
        if self.processed == 0:
            return 0.0
        return (self.elapsed / self.processed) * (self.total - self.processed)

    def status_line(self) -> str:
        eta_min = self.eta_seconds / 60
        parts = [
            f"[{self.processed}/{self.total}]",
            f"${self.total_cost:.2f} spent",
            f"avg ${self.avg_cost:.4f}/call",
            f"{self.hit_rate:.1f}% findings/startup",
            f"{self.total_findings} findings",
        ]
        if self.errors:
            parts.append(f"{self.errors} errors")
        parts.append(f"ETA {eta_min:.0f}min")
        return " | ".join(parts)

    def summary_dict(self) -> dict:
        return {
            "total_requested": self.total,
            "processed": self.processed,
            "successful": self.successful,
            "errors": self.errors,
            "companies_with_findings": self.found,
            "companies_without_findings": self.not_found,
            "total_findings": self.total_findings,
            "total_cost_usd": round(self.total_cost, 4),
            "avg_cost_usd": round(self.avg_cost, 4),
            "min_cost_usd": round(min(self.costs), 4) if self.costs else 0,
            "max_cost_usd": round(max(self.costs), 4) if self.costs else 0,
            "hit_rate_pct": round(self.hit_rate, 1),
            "avg_duration_s": round(sum(self.durations) / len(self.durations), 1) if self.durations else 0,
            "elapsed_seconds": round(self.elapsed, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# INCREMENTAL CSV WRITER
# ─────────────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "company_id", "company_name", "homepage_url", "short_description",
    "research_priority_score",
    "preset", "genai_adoption_found", "no_finding_reason", "no_finding_analysis", "finding_count",
    "finding_id", "AI_tool_used", "use_case", "business_function",
    "evidence_description", "source_url", "source_type",
    "cost_usd", "input_tokens", "output_tokens", "total_tokens",
    "search_results_count", "response_id", "response_status",
    "run_id", "error",
]


class CSVAppender:
    """Async-safe incremental CSV writer.

    Rebuilds from the master JSONL on init (to handle restarts cleanly),
    then appends new rows after each API call.
    """

    def __init__(self, csv_path: Path, master_jsonl: Path):
        self.csv_path = csv_path
        self.master_jsonl = master_jsonl
        self._lock = asyncio.Lock()

    def rebuild_from_jsonl(self) -> int:
        """Regenerate the entire CSV from the master JSONL. Returns rows written."""
        return _write_csv_from_jsonl(self.master_jsonl, self.csv_path)

    async def append_result(self, rec: dict) -> None:
        """Append CSV rows for a single research result (one row per finding)."""
        rows = _result_to_csv_rows(rec)
        async with self._lock:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                for row in rows:
                    writer.writerow(row)


def _result_to_csv_rows(rec: dict) -> list[dict]:
    """Convert a single JSONL record dict to one or more CSV row dicts."""
    base = {
        "company_id": rec.get("rcid"),
        "company_name": rec.get("company_name"),
        "homepage_url": rec.get("homepage_url") or "",
        "short_description": rec.get("short_description") or "",
        "research_priority_score": rec.get("priority"),
        "preset": rec.get("preset"),
        "genai_adoption_found": rec.get("genai_adoption_found"),
        "no_finding_reason": rec.get("no_finding_reason") or "",
        "no_finding_analysis": rec.get("no_finding_analysis") or "",
        "finding_count": rec.get("findings_count", 0),
        "cost_usd": rec.get("cost_usd") or "",
        "input_tokens": rec.get("input_tokens") or "",
        "output_tokens": rec.get("output_tokens") or "",
        "total_tokens": rec.get("total_tokens") or "",
        "search_results_count": rec.get("search_results_count", 0),
        "response_id": rec.get("response_id") or "",
        "response_status": rec.get("response_status") or "",
        "run_id": rec.get("run_id") or "",
        "error": rec.get("error") or "",
    }

    findings = rec.get("findings", [])
    if not findings:
        return [{
            **base,
            "finding_id": "", "AI_tool_used": "",
            "use_case": "", "business_function": "",
            "evidence_description": "",
            "source_url": "", "source_type": "",
        }]

    rows = []
    for f in findings:
        rows.append({
            **base,
            "finding_id": f.get("finding_id"),
            "AI_tool_used": f.get("AI_tool_used", ""),
            "use_case": f.get("use_case", ""),
            "business_function": f.get("business_function", ""),
            "evidence_description": f.get("evidence_description", ""),
            "source_url": f.get("source_url", ""),
            "source_type": f.get("source_type", ""),
        })
    return rows


def _deduplicate_jsonl(jsonl_path: Path) -> list[dict]:
    """Read the master JSONL and keep only the best record per (rcid, preset).

    For each duplicate, prefer the successful record (no error). If both
    succeeded or both failed, keep the later one (last in file order).
    """
    best: dict[tuple[int, str], dict] = {}
    if not jsonl_path.exists():
        return []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec.get("rcid"), rec.get("preset"))
            prev = best.get(key)
            if prev is None:
                best[key] = rec
            elif prev.get("error") and not rec.get("error"):
                best[key] = rec
            elif not prev.get("error") and rec.get("error"):
                pass  # keep the existing success
            else:
                best[key] = rec  # same status — keep the later one
    return list(best.values())


def _write_csv_from_jsonl(jsonl_path: Path, csv_path: Path) -> int:
    """Build the full CSV from the master JSONL (deduplicated). Returns rows written."""
    records = _deduplicate_jsonl(jsonl_path)
    rows_written = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for rec in records:
            for row in _result_to_csv_rows(rec):
                writer.writerow(row)
                rows_written += 1
    return rows_written


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def print_banner(args: argparse.Namespace) -> None:
    w = 62
    print("\n" + "=" * w)
    print("  PRODUCTION DEEP RESEARCH RUNNER")
    print("-" * w)
    print(f"  Preset:      {args.preset}")
    print(f"  Max steps:   {args.max_steps or 'preset default (10)'}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Budget cap:  ${args.budget_cap:.0f}")
    print(f"  Seed:        {args.seed}")
    print(f"  Master JSONL: {STAGE2_MASTER_JSONL}")
    print(f"  Master CSV:   {STAGE2_MASTER_CSV}")
    print("=" * w)


def print_result(result: ResearchResult, tracker: ProgressTracker) -> None:
    """Print a detailed per-company result to the terminal."""
    idx = tracker.processed
    total = tracker.total

    if result.error:
        print(
            f"\n[{idx}/{total}] {result.company_name}  (rcid={result.rcid}, p={result.priority})\n"
            f"        ERROR: {result.error}\n"
            f"        time={result.duration_seconds:.1f}s"
        )
    elif result.genai_adoption_found:
        tools = []
        for f in result.findings:
            tool = f.get("AI_tool_used", "")
            if tool and tool not in tools:
                tools.append(tool)
        tools_str = ", ".join(tools) if tools else "—"
        cost_str = f"${result.cost_usd:.4f}" if result.cost_usd else "N/A"
        print(
            f"\n[{idx}/{total}] {result.company_name}  (rcid={result.rcid}, p={result.priority})\n"
            f"        FOUND  findings={result.findings_count}  cost={cost_str}  time={result.duration_seconds:.1f}s\n"
            f"        Tools: {tools_str}"
        )
    else:
        cost_str = f"${result.cost_usd:.4f}" if result.cost_usd else "N/A"
        reason = result.no_finding_reason or "—"
        print(
            f"\n[{idx}/{total}] {result.company_name}  (rcid={result.rcid}, p={result.priority})\n"
            f"        NONE   cost={cost_str}  time={result.duration_seconds:.1f}s  reason={reason}"
        )


def print_stats_bar(tracker: ProgressTracker) -> None:
    """Print a compact running-stats bar."""
    print(f"        --- {tracker.status_line()} ---")


def print_summary(
    overall: ProgressTracker,
    priority_trackers: dict[int, ProgressTracker],
) -> None:
    w = 62
    print("\n" + "=" * w)
    print("  RUN SUMMARY")
    print("=" * w)

    for priority in sorted(priority_trackers.keys(), reverse=True):
        pt = priority_trackers[priority]
        s = pt.summary_dict()
        if s["processed"] == 0:
            continue
        print(f"\n  Priority {priority}:")
        print(f"    Processed:  {s['processed']}  ({s['errors']} errors)")
        print(f"    Hit rate:   {s['hit_rate_pct']}%  ({s['total_findings']} findings / {s['processed']} startups)")
        print(f"    Findings:   {s['total_findings']}  ({s['companies_with_findings']} companies with findings)")
        print(f"    Cost:       ${s['total_cost_usd']:.2f}  (avg ${s['avg_cost_usd']:.4f}  min ${s['min_cost_usd']:.4f}  max ${s['max_cost_usd']:.4f})")
        print(f"    Avg time:   {s['avg_duration_s']:.1f}s per call")

    s = overall.summary_dict()
    print(f"\n  Overall:")
    print(f"    Processed:  {s['processed']}  ({s['errors']} errors)")
    print(f"    Hit rate:   {s['hit_rate_pct']}%  ({s['total_findings']} findings / {s['processed']} startups)")
    print(f"    Findings:   {s['total_findings']}")
    print(f"    Cost:       ${s['total_cost_usd']:.2f}  (avg ${s['avg_cost_usd']:.4f}  min ${s['min_cost_usd']:.4f}  max ${s['max_cost_usd']:.4f})")
    print(f"    Avg time:   {s['avg_duration_s']:.1f}s per call")
    print(f"    Wall clock: {s['elapsed_seconds']:.0f}s  ({s['elapsed_seconds']/60:.1f}min)")

    if s["successful"] > 0 and s["total_cost_usd"] > 0:
        projected = int(4000 / s["avg_cost_usd"])
        full_cost = 9420 * s["avg_cost_usd"]
        print(f"\n  Projections:")
        print(f"    Full 9,420 companies: ~${full_cost:.0f}")
        print(f"    Companies within $4k: ~{projected:,}")

    print("=" * w)


def write_run_meta(
    run_dir: Path,
    args: argparse.Namespace,
    tracker: ProgressTracker,
) -> None:
    meta = {
        "run_id": run_dir.name,
        "started_at": datetime.now().astimezone().isoformat(),
        "args": {
            "sample_size": args.sample_size,
            "priorities": args.priorities,
            "preset": args.preset,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "concurrency": args.concurrency,
            "budget_cap": args.budget_cap,
        },
        "results": tracker.summary_dict(),
    }
    meta_path = run_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    logger.info("Run metadata: %s", meta_path)


# ─────────────────────────────────────────────────────────────────────────────
# RUN DIRECTORY
# ─────────────────────────────────────────────────────────────────────────────

def create_run_dir(label: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{label}_{timestamp}"
    run_dir = STAGE2_RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    latest = STAGE2_RUNS_DIR / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(run_name)

    return run_dir


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING (file only — console output is handled by print functions)
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(run_dir: Path, verbose: bool = False) -> None:
    console_level = logging.DEBUG if verbose else logging.WARNING

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))

    log_file = run_dir / "run.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logger.setLevel(logging.DEBUG)
    logger.addHandler(console)
    logger.addHandler(fh)

    logger.info("Run directory: %s", run_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production deep-research runner for Stage 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Statistical test: 200 per priority
  python -m src.stage_2.production_agent_runner \\
      --sample-size 200 --priorities 5 4 --concurrency 5 --budget-cap 200

  # Scale to all remaining
  python -m src.stage_2.production_agent_runner \\
      --sample-size 0 --priorities 5 4 --concurrency 10 --budget-cap 1500

  # Dry run
  python -m src.stage_2.production_agent_runner --sample-size 200 --dry-run
""",
    )
    parser.add_argument(
        "--sample-size", type=int, default=200,
        help="Companies to sample per priority level. 0 = all. (default: 200)",
    )
    parser.add_argument(
        "--priorities", nargs="+", type=int, default=[5, 4],
        help="Priority levels to process (default: 5 4)",
    )
    parser.add_argument(
        "--preset", default=DEFAULT_PRESET,
        help=f"Perplexity Agent preset (default: {DEFAULT_PRESET})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for deterministic sampling (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Max concurrent API calls (default: 5)",
    )
    parser.add_argument(
        "--budget-cap", type=float, default=200.0,
        help="Stop after spending this many USD in this run (default: 200)",
    )
    parser.add_argument(
        "--dataset", type=Path, default=DATASET_PATH,
        help=f"Path to input dataset JSONL (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override preset max_steps (deep-research default: 10). Lower = fewer reasoning iterations = cheaper.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate setup and show what would be processed, without calling the API",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging to console (always written to run.log)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def async_main(args: argparse.Namespace) -> None:
    run_dir = create_run_dir("run")
    setup_logging(run_dir, args.verbose)

    print_banner(args)

    # ── Load dataset ──
    print(f"\nLoading dataset from {args.dataset}")
    all_companies = load_dataset(args.dataset)
    logger.info("Loaded %d companies", len(all_companies))
    print(f"  Loaded {len(all_companies):,} companies")

    # ── Resume: find what's already done ──
    completed = load_completed_rcids(STAGE2_MASTER_JSONL, args.preset)
    if completed:
        print(f"  Already completed: {len(completed):,} (preset={args.preset})")
    logger.info("Completed rcids in master: %d", len(completed))

    # ── Build work queues per priority ──
    work_queue: list[Company] = []
    priority_queues: dict[int, list[Company]] = {}

    for priority in args.priorities:
        sample = sample_companies(all_companies, priority, args.sample_size, args.seed)
        remaining = [c for c in sample if c.rcid not in completed]
        priority_queues[priority] = remaining
        work_queue.extend(remaining)
        done = len(sample) - len(remaining)
        print(f"  Priority {priority}: {len(remaining)} to process  ({len(sample)} sampled, {done} already done)")
        logger.info("Priority %d: %d sampled, %d done, %d remaining", priority, len(sample), done, len(remaining))

    if not work_queue:
        print(f"\nAll requested companies already processed. Nothing to do.")
        print(f"  Master JSONL: {STAGE2_MASTER_JSONL}")
        print(f"  Master CSV:   {STAGE2_MASTER_CSV}")
        return

    est_cost = len(work_queue) * 0.33
    print(f"\n  Total: {len(work_queue)} companies to process")
    print(f"  Estimated cost: ~${est_cost:.0f}  (at ~$0.33/call)")
    print(f"  Run directory:  {run_dir}")

    # ── Dry run? ──
    if args.dry_run:
        print(f"\n--- DRY RUN (no API calls) ---")
        for priority in sorted(priority_queues.keys(), reverse=True):
            q = priority_queues[priority]
            print(f"\n  Priority {priority} — {len(q)} companies:")
            for c in q[:10]:
                print(f"    {c.name}  (rcid={c.rcid})")
            if len(q) > 10:
                print(f"    ... and {len(q) - 10} more")
        print()
        return

    # ── Init API client ──
    keys = APIKeys()
    if not keys.perplexity:
        print("\n  ERROR: Perplexity API key not found.")
        print("  Set in credentials/perplexity_api_key.txt or PERPLEXITY_API_KEY env var.")
        sys.exit(1)

    client = AsyncPerplexity(api_key=keys.perplexity)

    # ── Rebuild CSV from existing JSONL (clean state for incremental append) ──
    csv_appender = CSVAppender(STAGE2_MASTER_CSV, STAGE2_MASTER_JSONL)
    existing_rows = csv_appender.rebuild_from_jsonl()
    if existing_rows:
        print(f"\n  Master CSV rebuilt: {existing_rows} existing rows")
    logger.info("CSV rebuilt from JSONL: %d rows", existing_rows)

    # ── Set up controller ──
    controller = RunController()
    controller.attach(asyncio.get_event_loop())

    stdin_task = asyncio.create_task(stdin_listener(controller))

    # ── Process ──
    overall_tracker = ProgressTracker(total=len(work_queue))
    priority_trackers: dict[int, ProgressTracker] = {}
    budget_exceeded = False
    fatal_error_hit = False

    print(
        "\n" + "-" * 62 + "\n"
        "  Ctrl+C = PAUSE  |  Enter = RESUME  |  Ctrl+C again = STOP\n"
        + "-" * 62
    )

    try:
        async with AsyncJSONLWriter(STAGE2_MASTER_JSONL) as jsonl_writer:
            for priority in sorted(priority_queues.keys(), reverse=True):
                queue = priority_queues[priority]
                if not queue or controller.should_stop or budget_exceeded or fatal_error_hit:
                    continue

                print(f"\n{'=' * 62}")
                print(f"  Processing Priority {priority}  ({len(queue)} companies)")
                print(f"{'=' * 62}")

                pt = ProgressTracker(total=len(queue))
                priority_trackers[priority] = pt

                sem = asyncio.Semaphore(args.concurrency)
                rate_limiter = AsyncRateLimiter(rpm=RPM_LIMIT, name="perplexity")
                qps_limiter = QPSLimiter(qps=QPS_LIMIT)

                async def process_one(
                    company: Company,
                    _pt: ProgressTracker = pt,
                ) -> None:
                    nonlocal budget_exceeded, fatal_error_hit

                    await controller.wait_if_paused()
                    if controller.should_stop or budget_exceeded or fatal_error_hit:
                        return

                    async with sem:
                        await controller.wait_if_paused()
                        if controller.should_stop or budget_exceeded or fatal_error_hit:
                            return

                        await rate_limiter.acquire()
                        await qps_limiter.acquire()
                        await controller.track_in_flight(1)

                        logger.info(
                            "START %s (rcid=%d, p=%d)",
                            company.name, company.rcid, company.research_priority_score,
                        )
                        result = await research_company(
                            client, company, args.preset, run_dir.name,
                            max_steps=args.max_steps,
                        )

                        await controller.track_in_flight(-1)

                    # Write to master JSONL
                    result_dict = asdict(result)
                    await jsonl_writer.write(result_dict)

                    # Append to CSV immediately
                    await csv_appender.append_result(result_dict)

                    # Update trackers
                    await _pt.record(result)
                    await overall_tracker.record(result)

                    # Log to file (always detailed)
                    logger.info(
                        "DONE %s (rcid=%d): found=%s findings=%d cost=%s time=%.1fs error=%s",
                        company.name, company.rcid,
                        result.genai_adoption_found, result.findings_count,
                        f"${result.cost_usd:.4f}" if result.cost_usd else "N/A",
                        result.duration_seconds,
                        result.error or "none",
                    )

                    # Terminal output
                    print_result(result, overall_tracker)
                    print_stats_bar(overall_tracker)

                    # Fatal error check (quota exceeded, bad API key, etc.)
                    if result.fatal:
                        fatal_error_hit = True
                        print(
                            f"\n  FATAL ERROR — halting all dispatches:\n"
                            f"  {result.error}"
                        )
                        logger.error("Fatal error, stopping run: %s", result.error)
                        return

                    # Budget check
                    if overall_tracker.total_cost >= args.budget_cap:
                        budget_exceeded = True
                        print(
                            f"\n  BUDGET CAP REACHED: ${overall_tracker.total_cost:.2f} "
                            f">= ${args.budget_cap:.2f}"
                        )
                        logger.warning(
                            "Budget cap reached: $%.2f >= $%.2f",
                            overall_tracker.total_cost, args.budget_cap,
                        )

                tasks = [asyncio.create_task(process_one(c)) for c in queue]
                await asyncio.gather(*tasks, return_exceptions=True)

                if controller.should_stop or budget_exceeded or fatal_error_hit:
                    break

    finally:
        stdin_task.cancel()
        try:
            await stdin_task
        except asyncio.CancelledError:
            pass
        await client.close()

    # ── Summary ──
    print_summary(overall_tracker, priority_trackers)
    write_run_meta(run_dir, args, overall_tracker)

    print(f"\n  Master JSONL: {STAGE2_MASTER_JSONL}")
    print(f"  Master CSV:   {STAGE2_MASTER_CSV}")
    print(f"  Run log:      {run_dir / 'run.log'}")
    print(f"  Run metadata: {run_dir / 'run_meta.json'}")
    print()


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
