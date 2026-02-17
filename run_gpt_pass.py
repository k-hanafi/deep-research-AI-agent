"""
GPT Pass — Classify companies using GPT-5-nano against Tavily results.

Reads the Tavily JSONL (output of run_tavily_pass.py), reconstructs the
data structures, and runs GPT classification. Results are written to a
separate JSONL file.

This is intentionally decoupled from the Tavily pass so you can:
- Re-run GPT with different prompts without re-running Tavily
- Compare prompt versions side by side
- Iterate on the classifier without incurring Tavily costs

Output format (one JSON object per line):
{
    "rcid": 12345,
    "name": "Acme Corp",
    "online_presence_score": 7,
    "research_priority_score": 4,
    "reasoning": "...",
    "error": null
}

Usage:
    python run_gpt_pass.py                           # Use default tavily_results.jsonl
    python run_gpt_pass.py --input custom.jsonl      # Custom Tavily JSONL
    python run_gpt_pass.py --tag v2                  # Tag output as gpt_v2.jsonl
    python run_gpt_pass.py --concurrency 50          # Override concurrency limit
    python run_gpt_pass.py --retry-errors             # Re-classify previously failed companies
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    PROCESSING, STAGE1_OUTPUT_DIR, STAGE1_GPT_DIR, LOG_DIR,
    APIKeys,
)
from src.common import AsyncJSONLWriter, AsyncRateLimiter
from src.stage_1 import (
    WebsiteStatus,
    TavilySearchResult,
    SearchSnippet,
    classify_company,
)


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure dual logging: stdout + persistent log file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"gpt_pass_{timestamp}.log"

    logger = logging.getLogger("gpt_pass")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

TAVILY_JSONL = STAGE1_OUTPUT_DIR / "tavily_results.jsonl"


def load_tavily_records(jsonl_path: Path) -> list[dict]:
    """Load all records from the Tavily JSONL file."""
    records = []
    with open(jsonl_path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping malformed line {line_no}: {e}")
    return records


def load_existing_records(jsonl_path: Path) -> tuple[set[int], set[int]]:
    """
    Scan existing GPT JSONL and return (successful_rcids, error_rcids).

    A record is considered an error if its 'error' field is non-null.
    """
    successful = set()
    errored = set()
    if not jsonl_path.exists():
        return successful, errored
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rcid = int(obj["rcid"])
                if obj.get("error"):
                    errored.add(rcid)
                else:
                    successful.add(rcid)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return successful, errored


def reconstruct_website_status(record: dict) -> WebsiteStatus:
    """Rebuild WebsiteStatus from a Tavily JSONL record."""
    wc = record.get("website_check", {})
    return WebsiteStatus(
        url=wc.get("url", ""),
        is_alive=wc.get("is_alive", False),
        status_code=wc.get("status_code"),
        final_url=wc.get("final_url"),
        error=wc.get("error"),
    )


def reconstruct_tavily_result(record: dict) -> TavilySearchResult:
    """Rebuild TavilySearchResult from a Tavily JSONL record."""
    tv = record.get("tavily", {})
    raw = tv.get("raw_response", {})

    snippets = []
    for r in (raw or {}).get("results", []):
        snippets.append(SearchSnippet(
            title=r.get("title", ""),
            url=r.get("url", ""),
            content=r.get("content", ""),
            score=r.get("score", 0.0),
        ))

    # Answer may come from the explicit field or from raw_response (backward compat)
    answer = tv.get("answer") or (raw or {}).get("answer") or None

    return TavilySearchResult(
        company_name=record.get("name", ""),
        query=tv.get("query", ""),
        snippets=snippets,
        result_count=tv.get("result_count", 0),
        answer=answer,
        raw_response=raw or None,
        error=tv.get("error"),
    )


def build_gpt_record(
    rcid: int,
    name: str,
    assessment,
) -> dict:
    """Build the GPT JSONL record."""
    return {
        "rcid": rcid,
        "name": name,
        "online_presence_score": assessment.online_presence_score,
        "research_priority_score": assessment.research_priority_score,
        "reasoning": assessment.reasoning,
        "error": assessment.error,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────

class GracefulShutdown:
    """Signal handler for clean Ctrl+C shutdown."""

    def __init__(self, logger: logging.Logger):
        self.shutdown_requested = False
        self.force_exit = False
        self.logger = logger

    def handler(self, signum, frame):
        if self.shutdown_requested:
            self.logger.warning("Force shutdown requested. Exiting immediately.")
            self.force_exit = True
            sys.exit(1)
        self.shutdown_requested = True
        self.logger.warning(
            "Graceful shutdown requested (Ctrl+C). "
            "Finishing current batch... Press Ctrl+C again to force exit."
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Run GPT classification pass on Tavily results")
    parser.add_argument("--input", type=str, default=None, help="Path to Tavily JSONL")
    parser.add_argument("--tag", type=str, default=None,
                        help="Version tag for output file (e.g. 'v2' -> gpt_v2.jsonl)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N records")
    parser.add_argument("--concurrency", type=int, default=50,
                        help="Max concurrent requests (default: 50, GPT-5-nano is fast)")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Re-classify companies that previously failed with errors")
    args = parser.parse_args()

    tavily_path = Path(args.input) if args.input else TAVILY_JSONL
    if not tavily_path.exists():
        print(f"ERROR: Tavily JSONL not found: {tavily_path}")
        print("  Run run_tavily_pass.py first to generate Tavily results.")
        sys.exit(1)

    # Output path
    if args.tag:
        output_name = f"gpt_{args.tag}.jsonl"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"gpt_{timestamp}.jsonl"
    output_path = STAGE1_GPT_DIR / output_name

    logger = setup_logging(LOG_DIR)

    # Graceful shutdown handler
    shutdown = GracefulShutdown(logger)
    signal.signal(signal.SIGINT, shutdown.handler)

    # Load API keys
    keys = APIKeys()
    if not keys.openai:
        logger.error("OpenAI API key not found. Add to credentials/openai_api_key.txt")
        sys.exit(1)

    # Load Tavily records
    logger.info(f"Loading Tavily results from {tavily_path}...")
    records = load_tavily_records(tavily_path)
    logger.info(f"  Total Tavily records: {len(records)}")

    # Skip Tavily records that have errors (no point classifying them)
    records = [r for r in records if not r.get("tavily", {}).get("error")]
    logger.info(f"  Tavily records with valid search results: {len(records)}")

    if args.limit:
        records = records[:args.limit]
        logger.info(f"  Limited to first {args.limit}")

    # Check existing progress
    successful, errored = load_existing_records(output_path)
    if successful:
        logger.info(f"  Already classified successfully: {len(successful)}")
    if errored:
        logger.info(f"  Previously failed with errors: {len(errored)}")

    # Determine which records to process
    if args.retry_errors:
        skip = successful
        logger.info(f"  --retry-errors: will re-classify {len(errored)} failed companies")
    else:
        skip = successful | errored

    remaining = [r for r in records if int(r["rcid"]) not in skip]
    logger.info(f"  Remaining to classify: {len(remaining)}")

    if not remaining:
        logger.info("\nAll records already classified. Nothing to do.")
        return

    # Cost estimate (GPT-5-nano is very cheap)
    cost = len(remaining) * 0.0002
    logger.info(f"\nEstimated GPT cost: ${cost:.3f} ({len(remaining)} calls x $0.0002)")
    logger.info(f"Output: {output_path}")
    logger.info(f"Concurrency: {args.concurrency}")

    input("\nPress Enter to start (or Ctrl+C to cancel)...")

    # Create shared OpenAI client
    import httpx
    openai_client = httpx.AsyncClient(timeout=PROCESSING.openai_timeout)

    # Rate limiter + concurrency semaphore
    openai_limiter = AsyncRateLimiter(rpm=PROCESSING.openai_rpm, name="openai")
    semaphore = asyncio.Semaphore(args.concurrency)

    start_time = time.time()
    processed = 0
    errors = 0

    # Score distribution tracking
    score_dist = {i: 0 for i in range(6)}

    async with AsyncJSONLWriter(output_path) as writer:

        async def classify_one(record: dict, idx: int) -> None:
            nonlocal processed, errors
            rcid = int(record["rcid"])
            name = record["name"]

            # Reconstruct data from Tavily JSONL
            website_status = reconstruct_website_status(record)
            search_result = reconstruct_tavily_result(record)

            async with semaphore:
                await openai_limiter.acquire()

                try:
                    assessment = await classify_company(
                        company_name=name,
                        company_description=record.get("short_description"),
                        website_status=website_status,
                        search_result=search_result,
                        api_key=keys.openai,
                        homepage_url=record.get("homepage_url"),
                        long_description=record.get("description"),
                        client=openai_client,
                    )

                    gpt_record = build_gpt_record(rcid, name, assessment)
                    await writer.write(gpt_record)
                    processed += 1

                    score = assessment.research_priority_score
                    if 0 <= score <= 5:
                        score_dist[score] += 1
                    logger.info(
                        f"  [{idx}/{len(remaining)}] {name}: "
                        f"P{score} (presence={assessment.online_presence_score}) "
                        f"— {assessment.reasoning[:50]}"
                    )

                except Exception as e:
                    errors += 1
                    error_record = {
                        "rcid": rcid, "name": name,
                        "online_presence_score": 0, "research_priority_score": 0,
                        "reasoning": "", "error": f"{type(e).__name__}: {str(e)[:200]}",
                    }
                    await writer.write(error_record)
                    logger.error(f"  [{idx}/{len(remaining)}] {name}: ERROR — {e}")

        try:
            # Process in batches for progress reporting
            batch_size = 200
            for batch_start in range(0, len(remaining), batch_size):
                if shutdown.shutdown_requested:
                    logger.warning("Shutdown: stopping after current batch completes.")
                    break

                batch = remaining[batch_start:batch_start + batch_size]
                tasks = [
                    classify_one(record, batch_start + j + 1)
                    for j, record in enumerate(batch)
                ]
                await asyncio.gather(*tasks)

                # Progress report
                elapsed = time.time() - start_time
                done = batch_start + len(batch)
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - done) / rate if rate > 0 else 0
                logger.info(
                    f"\n  --- Progress: {done}/{len(remaining)} | "
                    f"{rate:.1f}/sec | ETA: {eta/60:.0f}min | "
                    f"Errors: {errors} | "
                    f"Rate limiter: {openai_limiter.stats} ---\n"
                )

        finally:
            await openai_client.aclose()

    elapsed = time.time() - start_time

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"  GPT pass complete")
    logger.info(f"  Processed: {processed}/{len(remaining)} ({errors} errors)")
    logger.info(f"  Time: {elapsed:.1f}s ({processed/elapsed:.1f}/sec)" if elapsed > 0 else "")
    logger.info(f"  Output: {output_path}")

    logger.info(f"\n  Score distribution:")
    labels = {
        5: "5 (definitely yields)",
        4: "4 (potentially yields)",
        3: "3 (worth a shot)",
        2: "2 (not worth it)",
        1: "1 (mostly unrelated)",
        0: "0 (not researchable)",
    }
    for score in [5, 4, 3, 2, 1, 0]:
        count = score_dist[score]
        pct = count / processed * 100 if processed else 0
        bar = "█" * int(pct / 2)
        logger.info(f"    {labels[score]:25} {count:4} ({pct:5.1f}%) {bar}")

    deep = sum(score_dist[s] for s in [3, 4, 5])
    if processed:
        logger.info(f"\n  Deep research candidates (>= 3): {deep}/{processed} "
                    f"({deep/processed*100:.0f}%)")

    if errors:
        logger.info(f"\n  Tip: run with --retry-errors to re-classify {errors} failed companies")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
