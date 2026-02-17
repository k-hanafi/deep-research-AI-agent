"""
Tavily Pass — Production runner for Stage 1 web search.

Reads the Crunchbase CSV, runs website checks + Tavily searches,
and writes results incrementally to a JSONL file.

Checkpointing is implicit: the JSONL file IS the checkpoint.
On resume, existing rcids in the JSONL are skipped automatically.

Output format (one JSON object per line in tavily_results.jsonl):
{
    "rcid": 12345,
    "name": "Acme Corp",
    "short_description": "...",
    "description": "...",
    "homepage_url": "https://...",
    "website_check": { "url": "...", "is_alive": true, ... },
    "tavily": { "query": "...", "result_count": 5, "raw_response": {...} },
    "timestamp": "2026-02-05T..."
}

Usage:
    python run_tavily_pass.py                    # Process all 44K companies
    python run_tavily_pass.py --limit 100        # First 100 only (testing)
    python run_tavily_pass.py --concurrency 25   # Override concurrency limit
    python run_tavily_pass.py --retry-errors      # Re-process previously failed companies
"""

import argparse
import asyncio
import csv
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    PROCESSING, STAGE1_OUTPUT_DIR, LOG_DIR,
    APIKeys, DATA_DIR,
)
from src.common import AsyncJSONLWriter, AsyncRateLimiter
from src.stage_1 import (
    check_website,
    search_tavily,
    WebsiteStatus,
    TavilySearchResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path) -> logging.Logger:
    """
    Configure dual logging: stdout + persistent log file.

    The log file persists across runs and is timestamped, so each
    invocation creates a new log file for easy debugging.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"tavily_pass_{timestamp}.log"

    logger = logging.getLogger("tavily_pass")
    logger.setLevel(logging.INFO)

    # File handler — captures everything for post-mortem debugging
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # Console handler — concise progress output
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FILE
# ─────────────────────────────────────────────────────────────────────────────

TAVILY_JSONL = STAGE1_OUTPUT_DIR / "tavily_results.jsonl"
DATA_FILE = DATA_DIR / "44k_crunchbase_startups.csv"


def load_existing_records(jsonl_path: Path) -> tuple[set[int], set[int]]:
    """
    Scan existing JSONL and return (successful_rcids, error_rcids).

    A record is considered an error if its tavily.error field is non-null.
    This distinction lets --retry-errors re-process failed companies.
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
                tavily_error = obj.get("tavily", {}).get("error")
                if tavily_error:
                    errored.add(rcid)
                else:
                    successful.add(rcid)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return successful, errored


def load_csv_companies(csv_path: Path, limit: int | None = None) -> list[dict]:
    """Load companies from the Crunchbase CSV."""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if limit is not None:
        rows = rows[:limit]
    return rows


def build_tavily_record(
    company: dict,
    website_status: WebsiteStatus,
    search_result: TavilySearchResult,
) -> dict:
    """Build the JSONL record for one company."""
    ws = website_status
    sr = search_result
    return {
        "rcid": int(company["rcid"]),
        "name": company["name"],
        "short_description": company.get("short_description") or None,
        "description": company.get("description") or None,
        "homepage_url": company.get("homepage_url") or None,
        "website_check": {
            "url": ws.url,
            "is_alive": ws.is_alive,
            "status_code": ws.status_code,
            "final_url": ws.final_url,
            "error": ws.error,
        },
        "tavily": {
            "query": sr.query,
            "answer": sr.answer,
            "result_count": sr.result_count,
            "raw_response": sr.raw_response,
            "error": sr.error,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def process_company(
    company: dict,
    tavily_api_key: str,
    http_client,
    tavily_client,
) -> dict:
    """Run website check + Tavily search for a single company, return JSONL record."""
    name = company["name"]
    homepage = company.get("homepage_url") or ""

    # Website check and Tavily search are independent — run in parallel
    website_status, search_result = await asyncio.gather(
        check_website(homepage, client=http_client),
        search_tavily(
            company_name=name,
            homepage_url=homepage or None,
            company_description=company.get("short_description") or None,
            api_key=tavily_api_key,
            client=tavily_client,
        ),
    )

    return build_tavily_record(company, website_status, search_result)


# ─────────────────────────────────────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────

class GracefulShutdown:
    """
    Signal handler for clean Ctrl+C shutdown.

    On first Ctrl+C: sets a flag so the main loop stops dispatching
    new batches and finishes the current in-flight batch.
    On second Ctrl+C: forces immediate exit.
    """

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
    parser = argparse.ArgumentParser(description="Run Tavily pass on Crunchbase dataset")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N companies")
    parser.add_argument("--output", type=str, default=None, help="Custom output JSONL path")
    parser.add_argument("--concurrency", type=int, default=25,
                        help="Max concurrent requests (default: 25)")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Re-process companies that previously failed with errors")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else TAVILY_JSONL
    logger = setup_logging(LOG_DIR)

    # Graceful shutdown handler
    shutdown = GracefulShutdown(logger)
    signal.signal(signal.SIGINT, shutdown.handler)

    # Load API keys
    keys = APIKeys()
    if not keys.tavily:
        logger.error("Tavily API key not found. Add to credentials/tavily_api_key.txt")
        sys.exit(1)

    # Load dataset
    logger.info(f"Loading dataset from {DATA_FILE}...")
    companies = load_csv_companies(DATA_FILE, limit=args.limit)
    total = len(companies)
    logger.info(f"  Total companies in scope: {total}")

    # Check existing progress (implicit checkpoint)
    successful, errored = load_existing_records(output_path)
    if successful:
        logger.info(f"  Already processed successfully: {len(successful)}")
    if errored:
        logger.info(f"  Previously failed with errors: {len(errored)}")

    # Determine which companies to process
    if args.retry_errors:
        # Process: not yet seen + previously errored
        skip = successful  # only skip successes
        logger.info(f"  --retry-errors: will re-process {len(errored)} failed companies")
    else:
        # Process: not yet seen (skip both successes and errors)
        skip = successful | errored

    remaining = [c for c in companies if int(c["rcid"]) not in skip]
    logger.info(f"  Remaining to process: {len(remaining)}")

    if not remaining:
        logger.info("\nAll companies already processed. Nothing to do.")
        return

    # Cost estimate
    cost = len(remaining) * 0.02  # ~$0.02/company (Tavily advanced)
    logger.info(f"\nEstimated Tavily cost: ${cost:.2f} ({len(remaining)} searches x $0.02)")
    logger.info(f"Output: {output_path}")
    logger.info(f"Concurrency: {args.concurrency}")

    input("\nPress Enter to start (or Ctrl+C to cancel)...")

    # Create shared HTTP clients
    import httpx
    http_client = httpx.AsyncClient(
        timeout=PROCESSING.http_timeout,
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0; +https://ubc.ca)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    tavily_client = httpx.AsyncClient(timeout=PROCESSING.tavily_timeout)

    # Rate limiter + concurrency semaphore
    tavily_limiter = AsyncRateLimiter(rpm=PROCESSING.tavily_rpm, name="tavily")
    semaphore = asyncio.Semaphore(args.concurrency)

    start_time = time.time()
    processed = 0
    errors = 0

    async with AsyncJSONLWriter(output_path) as writer:

        async def process_one(company: dict, idx: int) -> None:
            nonlocal processed, errors
            name = company["name"]

            async with semaphore:
                # Acquire rate limit slot before the Tavily call
                await tavily_limiter.acquire()

                try:
                    record = await process_company(
                        company, keys.tavily, http_client, tavily_client,
                    )
                    await writer.write(record)
                    processed += 1

                    result_count = record["tavily"]["result_count"]
                    tavily_err = record["tavily"].get("error")
                    alive = "alive" if record["website_check"]["is_alive"] else "dead"

                    if tavily_err:
                        errors += 1
                        logger.info(f"  [{idx}/{len(remaining)}] {name}: TAVILY ERROR — {tavily_err}")
                    else:
                        logger.info(f"  [{idx}/{len(remaining)}] {name}: {alive}, {result_count} results")

                except Exception as e:
                    errors += 1
                    # Write an error record so this company is tracked in the JSONL
                    error_record = {
                        "rcid": int(company["rcid"]),
                        "name": name,
                        "short_description": company.get("short_description") or None,
                        "description": company.get("description") or None,
                        "homepage_url": company.get("homepage_url") or None,
                        "website_check": {"url": "", "is_alive": False, "error": "skipped"},
                        "tavily": {
                            "query": "", "result_count": 0, "raw_response": None,
                            "error": f"{type(e).__name__}: {str(e)[:200]}",
                        },
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await writer.write(error_record)
                    logger.error(f"  [{idx}/{len(remaining)}] {name}: ERROR — {e}")

        try:
            # Process in batches for memory safety and progress reporting
            batch_size = 200
            for batch_start in range(0, len(remaining), batch_size):
                # Check for graceful shutdown between batches
                if shutdown.shutdown_requested:
                    logger.warning("Shutdown: stopping after current batch completes.")
                    break

                batch = remaining[batch_start:batch_start + batch_size]
                tasks = [
                    process_one(company, batch_start + j + 1)
                    for j, company in enumerate(batch)
                ]
                await asyncio.gather(*tasks)

                # Progress report after each batch
                elapsed = time.time() - start_time
                done = batch_start + len(batch)
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - done) / rate if rate > 0 else 0
                logger.info(
                    f"\n  --- Progress: {done}/{len(remaining)} | "
                    f"{rate:.1f} companies/sec | "
                    f"ETA: {eta/60:.0f}min | "
                    f"Errors: {errors} | "
                    f"Rate limiter: {tavily_limiter.stats} ---\n"
                )

        finally:
            await http_client.aclose()
            await tavily_client.aclose()

    elapsed = time.time() - start_time

    # Final summary
    successful_final, errored_final = load_existing_records(output_path)
    logger.info(f"\n{'='*60}")
    logger.info(f"  Tavily pass complete")
    logger.info(f"  Processed this run: {processed} ({errors} errors)")
    logger.info(f"  Time: {elapsed:.1f}s ({processed/elapsed:.1f} companies/sec)" if elapsed > 0 else "")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Total in JSONL: {len(successful_final) + len(errored_final)}/{total} "
                f"({len(successful_final)} ok, {len(errored_final)} errors)")
    if errored_final:
        logger.info(f"  Tip: run with --retry-errors to re-process {len(errored_final)} failed companies")
    logger.info(f"  Rate limiter stats: {tavily_limiter.stats}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
