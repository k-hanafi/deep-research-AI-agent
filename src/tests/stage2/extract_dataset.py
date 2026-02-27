"""
Extract priority 4+5 companies into a clean Stage 2 input dataset.

Joins GPT scoring results (JSONL) with Crunchbase metadata (CSV),
filters to research_priority_score >= 4, and writes a unified JSONL
ready for preset testing.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.config import DATA_DIR, STAGE1_GPT_DIR, STAGE2_OUTPUT_DIR


OUTPUT_FILE = STAGE2_OUTPUT_DIR / "stage2_input_dataset.jsonl"

GPT_RESULTS_FILE = STAGE1_GPT_DIR / "gpt_full_44k.jsonl"
CRUNCHBASE_CSV = DATA_DIR / "44k_crunchbase_startups.csv"

KEEP_FIELDS = [
    "rcid",
    "name",
    "homepage_url",
    "short_description",
    "research_priority_score",
    "online_presence_score",
    "category_list",
]


def load_gpt_results(path: Path, min_priority: int = 4) -> dict[int, dict]:
    """Load GPT scoring JSONL and filter to priority >= min_priority.

    Returns a dict keyed by rcid with scoring fields.
    """
    results: dict[int, dict] = {}
    errors = 0

    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            score = record.get("research_priority_score")
            if score is None or score < min_priority:
                continue

            rcid = record.get("rcid")
            if rcid is None:
                continue

            results[int(rcid)] = {
                "research_priority_score": score,
                "online_presence_score": record.get("online_presence_score", 0),
            }

    if errors:
        print(f"  Warning: {errors} malformed lines skipped in {path.name}")

    return results


def load_crunchbase_metadata(path: Path, rcids: set[int]) -> dict[int, dict]:
    """Load Crunchbase CSV and extract metadata for the given rcids."""
    df = pd.read_csv(path, usecols=["rcid", "name", "homepage_url", "short_description", "category_list"])
    df = df[df["rcid"].isin(rcids)]
    df = df.where(pd.notna(df), None)

    metadata: dict[int, dict] = {}
    for _, row in df.iterrows():
        metadata[int(row["rcid"])] = {
            "name": row["name"],
            "homepage_url": row["homepage_url"],
            "short_description": row["short_description"],
            "category_list": row["category_list"],
        }

    return metadata


def merge_and_write(
    gpt_results: dict[int, dict],
    crunchbase: dict[int, dict],
    output_path: Path,
) -> int:
    """Join GPT scores with Crunchbase metadata and write JSONL.

    Returns the number of records written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_metadata = 0

    # Sort by priority desc, then by rcid for deterministic output
    sorted_rcids = sorted(
        gpt_results.keys(),
        key=lambda r: (-gpt_results[r]["research_priority_score"], r),
    )

    with open(output_path, "w") as f:
        for rcid in sorted_rcids:
            cb = crunchbase.get(rcid)
            if cb is None:
                missing_metadata += 1
                continue

            record = {
                "rcid": rcid,
                "name": cb["name"],
                "homepage_url": cb["homepage_url"],
                "short_description": cb["short_description"],
                "research_priority_score": gpt_results[rcid]["research_priority_score"],
                "online_presence_score": gpt_results[rcid]["online_presence_score"],
                "category_list": cb["category_list"],
            }
            f.write(json.dumps(record) + "\n")
            written += 1

    if missing_metadata:
        print(f"  Warning: {missing_metadata} rcids had no Crunchbase metadata (skipped)")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract priority 4+5 companies into Stage 2 input dataset"
    )
    parser.add_argument(
        "--min-priority",
        type=int,
        default=4,
        help="Minimum research_priority_score to include (default: 4)",
    )
    parser.add_argument(
        "--gpt-results",
        type=Path,
        default=GPT_RESULTS_FILE,
        help=f"Path to GPT scoring JSONL (default: {GPT_RESULTS_FILE})",
    )
    parser.add_argument(
        "--crunchbase-csv",
        type=Path,
        default=CRUNCHBASE_CSV,
        help=f"Path to Crunchbase CSV (default: {CRUNCHBASE_CSV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Output JSONL path (default: {OUTPUT_FILE})",
    )
    args = parser.parse_args()

    print(f"Loading GPT results from {args.gpt_results} ...")
    gpt_results = load_gpt_results(args.gpt_results, min_priority=args.min_priority)
    print(f"  {len(gpt_results)} companies with priority >= {args.min_priority}")

    priority_counts: dict[int, int] = {}
    for r in gpt_results.values():
        s = r["research_priority_score"]
        priority_counts[s] = priority_counts.get(s, 0) + 1
    for score in sorted(priority_counts.keys(), reverse=True):
        print(f"    priority={score}: {priority_counts[score]}")

    print(f"Loading Crunchbase metadata from {args.crunchbase_csv} ...")
    crunchbase = load_crunchbase_metadata(args.crunchbase_csv, set(gpt_results.keys()))
    print(f"  {len(crunchbase)} matching Crunchbase records loaded")

    print(f"Writing merged dataset to {args.output} ...")
    written = merge_and_write(gpt_results, crunchbase, args.output)

    print(f"\nDone. {written} companies written to {args.output}")
    print(f"  File size: {args.output.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
