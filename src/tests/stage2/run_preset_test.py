"""
Preset-configurable test runner for Stage 2 deep research.

Runs Perplexity Agent API with configurable presets (deep-research,
advanced-deep-research) on sampled companies from the Stage 2 input
dataset. Supports resume, graceful shutdown, per-call cost tracking,
and automatic CSV export.

Usage:
    # Pilot: 5 companies per priority, both presets
    python -m src.tests.stage2.run_preset_test \
        --presets deep-research advanced-deep-research \
        --sample-size 5 --priorities 5 4 --phase pilot

    # Statistical: 100 priority=5 companies, single preset
    python -m src.tests.stage2.run_preset_test \
        --presets deep-research \
        --sample-size 100 --priorities 5 --phase statistical
"""

import argparse
import csv
import json
import logging
import random
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from perplexity import Perplexity
from perplexity.types.output_item import MessageOutputItem, SearchResultsOutputItem

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config import STAGE2_OUTPUT_DIR, STAGE2_TEST_RUNS_DIR, PROMPTS_DIR, LOG_DIR, APIKeys


DATASET_PATH = STAGE2_OUTPUT_DIR / "stage2_input_dataset.jsonl"
PROMPT_FILE = PROMPTS_DIR / "stage_2_perplexity_prompt.txt"

DEFAULT_SEED = 2026
DEFAULT_TIMEOUT = 300.0
RPM_LIMIT = 50          # Agent API Tier 0: 50 req/min, 1 QPS
INTER_CALL_DELAY = 1.0  # 1s between calls; each call takes 30-60s so RPM is never a concern

# Presets where response_format causes 500 errors; rely on prompt for JSON.
_SCHEMA_INCOMPATIBLE_PRESETS = {"advanced-deep-research"}

logger = logging.getLogger("run_preset_test")


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
class TestResult:
    rcid: int
    company_name: str
    homepage_url: Optional[str]
    short_description: Optional[str]
    preset: str
    priority: int
    genai_adoption_found: Optional[bool] = None
    findings_count: int = 0
    findings: list[dict] = field(default_factory=list)
    no_finding_reason: Optional[str] = None
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
    raw_content_preview: Optional[str] = None
    duration_seconds: float = 0.0
    timestamp: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# JSON SCHEMA — enforces structured output from the Perplexity Agent API
# ─────────────────────────────────────────────────────────────────────────────

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
    filtered = [c for c in companies if c.research_priority_score == priority]
    if not filtered:
        logger.warning("No companies with priority=%d found", priority)
        return []
    if sample_size >= len(filtered):
        return filtered
    rng = random.Random(seed)
    return rng.sample(filtered, sample_size)


def load_completed_rcids(output_path: Path, preset: str) -> set[int]:
    """Load rcids successfully processed for a given preset.

    Only counts records where error is null/absent — failed calls are
    eligible for retry on resume.
    """
    completed: set[int] = set()
    if not output_path.exists():
        return completed
    with open(output_path) as f:
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
# SINGLE COMPANY RESEARCH CALL (synchronous — SDK handles HTTP)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_fallback(output: list) -> str:
    """Try to extract text content when output_text is empty.

    Some model+preset combos may structure output differently. This walks
    all MessageOutputItem content parts to find any text, regardless of
    the content part type label.
    """
    texts: list[str] = []
    for item in output:
        if isinstance(item, MessageOutputItem):
            for part in (item.content or []):
                text = getattr(part, "text", None)
                if text:
                    texts.append(text)
    return "".join(texts)


def _extract_json_from_text(text: str) -> str:
    """Find the outermost JSON object in text that may contain prose.

    Models without response_format enforcement may wrap JSON in markdown
    fences or surrounding commentary. This extracts the first complete
    { … } block using brace-depth counting so we don't need regex for
    nested objects.
    """
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


def research_company(
    client: Perplexity,
    company: Company,
    preset: str,
) -> TestResult:
    start = time.monotonic()
    result = TestResult(
        rcid=company.rcid,
        company_name=company.name,
        homepage_url=company.homepage_url,
        short_description=company.short_description,
        preset=preset,
        priority=company.research_priority_score,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )

    try:
        create_kwargs: dict = dict(
            preset=preset,
            input=build_prompt(company),
            timeout=DEFAULT_TIMEOUT,
        )
        if preset in _SCHEMA_INCOMPATIBLE_PRESETS:
            create_kwargs["instructions"] = (
                "Your final answer must be ONLY a valid JSON object matching "
                "the schema described in the user prompt. Do not include any "
                "text, markdown fences, or commentary outside the JSON object."
            )
        else:
            create_kwargs["response_format"] = RESPONSE_SCHEMA

        response = client.responses.create(**create_kwargs)

        # ── Always capture metadata, usage, and citations first ──
        # These must be recorded even when content parsing fails,
        # so we can track costs on wasted API calls.
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

        # ── Check for API-level failure ──
        if response.status == "failed":
            err = response.error
            error_detail = f"{err.type}: {err.message}" if err else "unknown"
            raise ValueError(f"Response failed: {error_detail}")

        # ── Extract text content ──
        content = response.output_text
        if not content:
            content = _extract_text_fallback(response.output)
            if content:
                logger.debug(
                    "output_text empty; fallback extracted %d chars", len(content),
                )

        content = (content or "").strip()

        if not content:
            output_types = [type(item).__name__ for item in response.output]
            logger.warning(
                "Empty response for %s (model=%s, status=%s, types=%s)",
                company.name, response.model, response.status, output_types,
            )
            raise ValueError(
                f"Empty response content (model={response.model}, "
                f"status={response.status}, output_types={output_types})"
            )

        result.raw_content_preview = content[:500]

        # ── Parse JSON ──
        # Presets without response_format may wrap JSON in prose/fences.
        json_text = (
            _extract_json_from_text(content)
            if preset in _SCHEMA_INCOMPATIBLE_PRESETS
            else content
        )
        parsed = json.loads(json_text)

        result.genai_adoption_found = parsed.get("genai_adoption_found", False)
        result.findings = parsed.get("findings") or []
        result.findings_count = len(result.findings)
        result.no_finding_reason = parsed.get("no_finding_reason")

    except json.JSONDecodeError as e:
        result.error = f"JSON parse error: {e}"
        logger.debug(
            "Content that failed JSON parsing (%d chars): %.200s",
            len(content), content,
        )
    except Exception as e:
        result.error = f"{type(e).__name__}: {str(e)[:500]}"

    result.duration_seconds = round(time.monotonic() - start, 2)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────

class GracefulShutdown:
    def __init__(self):
        self.should_stop = False
        signal.signal(signal.SIGINT, self._handler)

    def _handler(self, _signum, _frame):
        if self.should_stop:
            logger.warning("Second SIGINT — forcing exit")
            sys.exit(1)
        logger.warning("SIGINT received — finishing current request then stopping")
        self.should_stop = True


# ─────────────────────────────────────────────────────────────────────────────
# JSONL WRITER (synchronous, append-mode for resume safety)
# ─────────────────────────────────────────────────────────────────────────────

def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


# ─────────────────────────────────────────────────────────────────────────────
# BATCH EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    companies: list[Company],
    preset: str,
    client: Perplexity,
    output_path: Path,
    shutdown: GracefulShutdown,
) -> list[TestResult]:
    """Process a batch of companies sequentially with a single preset."""
    results: list[TestResult] = []

    completed = load_completed_rcids(output_path, preset)
    remaining = [c for c in companies if c.rcid not in completed]

    if completed:
        logger.info(
            "Resuming: %d done, %d remaining for preset=%s",
            len(completed), len(remaining), preset,
        )
    if not remaining:
        logger.info("All companies already processed for preset=%s", preset)
        return results

    total_cost = 0.0

    for i, company in enumerate(remaining, 1):
        if shutdown.should_stop:
            logger.info(
                "Graceful shutdown — stopped after %d/%d",
                i - 1, len(remaining),
            )
            break

        logger.info(
            "[%s] %d/%d  %s  (rcid=%d, p=%d)",
            preset, i, len(remaining),
            company.name, company.rcid,
            company.research_priority_score,
        )

        result = research_company(
            client=client,
            company=company,
            preset=preset,
        )

        append_jsonl(output_path, asdict(result))
        results.append(result)

        if result.error:
            logger.error("  ERROR: %s", result.error)
        else:
            cost_str = f"${result.cost_usd:.4f}" if result.cost_usd else "N/A"
            total_cost += result.cost_usd or 0.0
            logger.info(
                "  found=%s  findings=%d  cost=%s  time=%.1fs",
                result.genai_adoption_found, result.findings_count,
                cost_str, result.duration_seconds,
            )
            logger.info(
                "  running total: $%.4f across %d companies",
                total_cost, len(results),
            )

        if i < len(remaining) and not shutdown.should_stop:
            time.sleep(INTER_CALL_DELAY)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# RUN DIRECTORY — one folder per invocation, all outputs co-located
# ─────────────────────────────────────────────────────────────────────────────

PRESET_SHORT = {
    "deep-research": "deep",
    "advanced-deep-research": "adv",
}


def create_run_dir(phase: str) -> Path:
    """Create a timestamped run directory and update the `latest` symlink.

    Layout:
        outputs/stage2/test_runs/
        ├── pilot_20260227_152049/     <- this run
        │   ├── deep_p5.jsonl
        │   ├── adv_p4.jsonl
        │   ├── results.csv
        │   └── run.log
        └── latest -> pilot_20260227_152049
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{phase}_{timestamp}"
    run_dir = STAGE2_TEST_RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    latest = STAGE2_TEST_RUNS_DIR / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(run_name)

    return run_dir


def build_output_path(run_dir: Path, preset: str, priority: str | int) -> Path:
    short = PRESET_SHORT.get(preset, preset.replace("-", ""))
    return run_dir / f"{short}_p{priority}.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# CSV EXPORT — one row per finding, 22 columns
# ─────────────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    # Company context
    "company_id", "company_name", "homepage_url", "short_description",
    "research_priority_score",
    # Research result
    "preset", "genai_adoption_found", "no_finding_reason", "finding_count",
    # Finding detail
    "finding_id", "AI_tool_used", "use_case", "business_function",
    "evidence_description", "source_url", "source_type",
    # Cost and diagnostics
    "cost_usd", "input_tokens", "output_tokens", "total_tokens",
    "search_results_count", "response_id", "response_status", "error",
]


def export_csv(jsonl_paths: list[Path], csv_path: Path) -> int:
    """Read all JSONL result files and write a flattened CSV (one row per finding).

    Companies with no findings produce a single row with blank finding columns.
    Returns the number of rows written.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for jsonl_path in jsonl_paths:
            if not jsonl_path.exists():
                continue
            with open(jsonl_path) as jf:
                for line in jf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    base = {
                        "company_id": rec.get("rcid"),
                        "company_name": rec.get("company_name"),
                        "homepage_url": rec.get("homepage_url") or "",
                        "short_description": rec.get("short_description") or "",
                        "research_priority_score": rec.get("priority"),
                        "preset": rec.get("preset"),
                        "genai_adoption_found": rec.get("genai_adoption_found"),
                        "no_finding_reason": rec.get("no_finding_reason") or "",
                        "finding_count": rec.get("findings_count", 0),
                        "cost_usd": rec.get("cost_usd") or "",
                        "input_tokens": rec.get("input_tokens") or "",
                        "output_tokens": rec.get("output_tokens") or "",
                        "total_tokens": rec.get("total_tokens") or "",
                        "search_results_count": rec.get("search_results_count", 0),
                        "response_id": rec.get("response_id") or "",
                        "response_status": rec.get("response_status") or "",
                        "error": rec.get("error") or "",
                    }

                    findings = rec.get("findings", [])
                    if not findings:
                        row = {**base, "finding_id": "", "AI_tool_used": "",
                               "use_case": "", "business_function": "",
                               "evidence_description": "",
                               "source_url": "", "source_type": ""}
                        writer.writerow(row)
                        rows_written += 1
                    else:
                        for finding in findings:
                            row = {
                                **base,
                                "finding_id": finding.get("finding_id"),
                                "AI_tool_used": finding.get("AI_tool_used", ""),
                                "use_case": finding.get("use_case", ""),
                                "business_function": finding.get("business_function", ""),
                                "evidence_description": finding.get("evidence_description", ""),
                                "source_url": finding.get("source_url", ""),
                                "source_type": finding.get("source_type", ""),
                            }
                            writer.writerow(row)
                            rows_written += 1

    return rows_written


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: dict[str, list[TestResult]]) -> None:
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    grand_cost = 0.0
    grand_calls = 0
    grand_errors = 0

    for label, results in all_results.items():
        if not results:
            continue

        costs = [r.cost_usd for r in results if r.cost_usd is not None]
        errors = [r for r in results if r.error]
        grand_errors += len(errors)
        non_error = len(results) - len(errors)
        found = [r for r in results if r.genai_adoption_found]
        total_findings = sum(r.findings_count for r in results)
        durations = [r.duration_seconds for r in results]

        print(f"\n{label}")
        print(f"  Companies processed: {len(results)}")
        print(f"  Errors: {len(errors)} ({100 * len(errors) / max(len(results), 1):.0f}%)")

        if costs:
            total_cost = sum(costs)
            grand_cost += total_cost
            avg = total_cost / len(costs)
            print(
                f"  Cost — total: ${total_cost:.4f}  avg: ${avg:.4f}  "
                f"min: ${min(costs):.4f}  max: ${max(costs):.4f}"
            )

        if non_error:
            pct = 100 * len(found) / non_error
            avg_findings = total_findings / non_error
            print(f"  Adoption found: {len(found)}/{non_error} ({pct:.0f}%)")
            print(f"  Total findings: {total_findings} (avg {avg_findings:.1f}/company)")

        if durations:
            avg_dur = sum(durations) / len(durations)
            print(
                f"  Duration — avg: {avg_dur:.1f}s  "
                f"min: {min(durations):.1f}s  max: {max(durations):.1f}s"
            )

        grand_calls += len(results)

    grand_successful = grand_calls - grand_errors
    print(f"\nGrand total: ${grand_cost:.4f} across {grand_calls} API calls ({grand_errors} errors)")

    if grand_successful and grand_cost > 0:
        avg_all = grand_cost / grand_successful
        budget = 4000.0
        projected = int(budget / avg_all)
        print(f"Projected capacity at this avg (${avg_all:.4f}/call, errors excluded): ~{projected:,} companies for $4k")

    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(run_dir: Path, verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))

    log_file = run_dir / "run.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logger.setLevel(logging.DEBUG)
    logger.addHandler(console)
    logger.addHandler(fh)

    logger.info("Run directory: %s", run_dir)
    logger.info("Logging to %s", log_file)


# ─────────────────────────────────────────────────────────────────────────────
# CLI & MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 2 deep research with configurable presets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pilot: 5 companies per priority, both presets
  python -m src.tests.stage2.run_preset_test \\
      --presets deep-research advanced-deep-research \\
      --sample-size 5 --priorities 5 4 --phase pilot

  # Statistical: 100 priority=5 companies, single preset
  python -m src.tests.stage2.run_preset_test \\
      --presets deep-research \\
      --sample-size 100 --priorities 5 --phase statistical

  # Target specific companies by rcid (bypasses sampling)
  python -m src.tests.stage2.run_preset_test \\
      --rcids 510536 59639 \\
      --presets deep-research advanced-deep-research --phase pilot
""",
    )
    parser.add_argument(
        "--presets", nargs="+", default=["deep-research"],
        help="Perplexity Agent presets to test (default: deep-research)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=5,
        help="Companies to sample per priority level (default: 5)",
    )
    parser.add_argument(
        "--priorities", nargs="+", type=int, default=[5],
        help="Priority levels to test (default: 5). Use '5 4' for both.",
    )
    parser.add_argument(
        "--phase", default="pilot",
        help="Test phase label for output filenames (default: pilot)",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for sampling (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--dataset", type=Path, default=DATASET_PATH,
        help=f"Path to input dataset JSONL (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--rcids", nargs="+", type=int, default=None,
        help="Target specific company rcids (bypasses --sample-size/--priorities)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = create_run_dir(args.phase)
    setup_logging(run_dir, args.verbose)

    keys = APIKeys()
    if not keys.perplexity:
        logger.error("Perplexity API key not found. Set in credentials/perplexity_api_key.txt")
        sys.exit(1)

    logger.info("Loading dataset from %s", args.dataset)
    all_companies = load_dataset(args.dataset)
    logger.info("Loaded %d companies", len(all_companies))

    client = Perplexity(api_key=keys.perplexity)
    shutdown = GracefulShutdown()
    all_results: dict[str, list[TestResult]] = {}
    all_jsonl_paths: list[Path] = []

    if args.rcids:
        rcid_set = set(args.rcids)
        targeted = [c for c in all_companies if c.rcid in rcid_set]
        missing = rcid_set - {c.rcid for c in targeted}
        if missing:
            logger.warning("rcids not found in dataset: %s", missing)
        if not targeted:
            logger.error("No matching companies found for --rcids")
            sys.exit(1)
        logger.info("Targeted %d companies by rcid", len(targeted))
        priority_groups = [("targeted", targeted)]
    else:
        priority_groups = []
        for priority in args.priorities:
            sample = sample_companies(all_companies, priority, args.sample_size, args.seed)
            logger.info(
                "Sampled %d companies for priority=%d (seed=%d)",
                len(sample), priority, args.seed,
            )
            if sample:
                priority_groups.append((str(priority), sample))

    for priority_label, sample in priority_groups:
        for preset in args.presets:
            if shutdown.should_stop:
                break

            output_path = build_output_path(run_dir, preset, priority_label)
            all_jsonl_paths.append(output_path)
            label = f"{preset} / priority={priority_label}"
            logger.info(
                "\n── %s  (%d companies) → %s",
                label, len(sample), output_path.name,
            )

            results = run_batch(
                companies=sample,
                preset=preset,
                client=client,
                output_path=output_path,
                shutdown=shutdown,
            )
            all_results[label] = results

        if shutdown.should_stop:
            break

    print_summary(all_results)

    existing_paths = [p for p in all_jsonl_paths if p.exists()]
    if existing_paths:
        csv_path = run_dir / "results.csv"
        rows = export_csv(existing_paths, csv_path)
        logger.info("CSV exported: %d rows → %s", rows, csv_path)
        print(f"\nCSV: {rows} rows written to {csv_path}")
        print(f"Run directory: {run_dir}")
        print(f"Latest shortcut: {STAGE2_TEST_RUNS_DIR / 'latest' / 'results.csv'}")

    client.close()


if __name__ == "__main__":
    main()
