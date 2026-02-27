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
from perplexity.types.output_item import SearchResultsOutputItem

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config import STAGE2_OUTPUT_DIR, PROMPTS_DIR, LOG_DIR, APIKeys


DATASET_PATH = STAGE2_OUTPUT_DIR / "stage2_input_dataset.jsonl"
PROMPT_FILE = PROMPTS_DIR / "stage_2_perplexity_prompt.txt"

DEFAULT_SEED = 2026
DEFAULT_TIMEOUT = 300.0
RPM_LIMIT = 5
INTER_CALL_DELAY = 60.0 / RPM_LIMIT  # 12s between calls to stay under 5 RPM

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
    citations: list[str] = field(default_factory=list)
    error: Optional[str] = None
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
                            "use_case": {"type": "string"},
                            "business_function": {"type": "string"},
                            "evidence_description": {"type": "string"},
                            "source_url": {"type": "string"},
                            "source_type": {"type": "string"},
                        },
                        "required": [
                            "finding_id", "use_case", "business_function",
                            "evidence_description", "source_url", "source_type",
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
    """Load rcids already processed for a given preset from an existing output file."""
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
                if rec.get("preset") == preset:
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

def research_company(
    client: Perplexity,
    company: Company,
    preset: str,
) -> TestResult:
    start = time.monotonic()
    result = TestResult(
        rcid=company.rcid,
        company_name=company.name,
        preset=preset,
        priority=company.research_priority_score,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )

    try:
        response = client.responses.create(
            preset=preset,
            input=build_prompt(company),
            response_format=RESPONSE_SCHEMA,
            timeout=DEFAULT_TIMEOUT,
        )

        result.response_id = response.id
        result.model_used = response.model

        parsed = json.loads(response.output_text)

        result.genai_adoption_found = parsed.get("genai_adoption_found", False)
        result.findings = parsed.get("findings", [])
        result.findings_count = len(result.findings)
        result.no_finding_reason = parsed.get("no_finding_reason")

        if response.usage:
            result.input_tokens = response.usage.input_tokens
            result.output_tokens = response.usage.output_tokens
            result.total_tokens = response.usage.total_tokens
            if response.usage.cost:
                result.cost_usd = response.usage.cost.total_cost

        search_result_count = 0
        for item in response.output:
            if isinstance(item, SearchResultsOutputItem):
                search_result_count += len(item.results)
                for sr in item.results:
                    result.citations.append(sr.url)
        result.search_results_count = search_result_count

    except json.JSONDecodeError as e:
        result.error = f"JSON parse error: {e}"
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
# OUTPUT PATH NAMING
# ─────────────────────────────────────────────────────────────────────────────

PRESET_SHORT = {
    "deep-research": "deep",
    "advanced-deep-research": "adv",
}


def build_output_path(phase: str, preset: str, priority: int) -> Path:
    date_str = datetime.now().strftime("%Y%m%d")
    short = PRESET_SHORT.get(preset, preset.replace("-", ""))
    filename = f"{phase}_{short}_p{priority}_{date_str}.jsonl"
    return STAGE2_OUTPUT_DIR / filename


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
    "finding_id", "use_case", "business_function", "evidence_description",
    "source_url", "source_type",
    # Cost and diagnostics
    "cost_usd", "input_tokens", "output_tokens", "total_tokens",
    "search_results_count", "response_id", "error",
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
                        "homepage_url": "",
                        "short_description": "",
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
                        "error": rec.get("error") or "",
                    }

                    findings = rec.get("findings", [])
                    if not findings:
                        row = {**base, "finding_id": "", "use_case": "",
                               "business_function": "", "evidence_description": "",
                               "source_url": "", "source_type": ""}
                        writer.writerow(row)
                        rows_written += 1
                    else:
                        for finding in findings:
                            row = {
                                **base,
                                "finding_id": finding.get("finding_id"),
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

    for label, results in all_results.items():
        if not results:
            continue

        costs = [r.cost_usd for r in results if r.cost_usd is not None]
        errors = [r for r in results if r.error]
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

    print(f"\nGrand total: ${grand_cost:.4f} across {grand_calls} API calls")

    if grand_calls and grand_cost > 0:
        avg_all = grand_cost / grand_calls
        budget = 4000.0
        projected = int(budget / avg_all)
        print(f"Projected capacity at this avg (${avg_all:.4f}/call): ~{projected:,} companies for $4k")

    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(phase: str, verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))

    log_file = LOG_DIR / f"preset_test_{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logger.setLevel(logging.DEBUG)
    logger.addHandler(console)
    logger.addHandler(fh)

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
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.phase, args.verbose)

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

    for priority in args.priorities:
        sample = sample_companies(all_companies, priority, args.sample_size, args.seed)
        logger.info(
            "Sampled %d companies for priority=%d (seed=%d)",
            len(sample), priority, args.seed,
        )
        if not sample:
            continue

        for preset in args.presets:
            if shutdown.should_stop:
                break

            output_path = build_output_path(args.phase, preset, priority)
            all_jsonl_paths.append(output_path)
            label = f"{preset} / priority={priority}"
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

    # Auto-generate CSV from all JSONL outputs
    existing_paths = [p for p in all_jsonl_paths if p.exists()]
    if existing_paths:
        date_str = datetime.now().strftime("%Y%m%d")
        csv_path = STAGE2_OUTPUT_DIR / f"{args.phase}_{date_str}_results.csv"
        rows = export_csv(existing_paths, csv_path)
        logger.info("CSV exported: %d rows → %s", rows, csv_path)
        print(f"\nCSV: {rows} rows written to {csv_path}")

    client.close()


if __name__ == "__main__":
    main()
