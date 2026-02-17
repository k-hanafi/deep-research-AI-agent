"""
Convert GPT classification JSONL results to an Excel-friendly CSV.

Enriches each GPT record with Crunchbase metadata (description, categories,
homepage URL) so the CSV is self-contained and ready for inspection or sharing.

Usage:
    python convert_gpt_to_csv.py                              # auto-detect latest gpt_*.jsonl
    python convert_gpt_to_csv.py --input gpt_test_1k.jsonl    # specific file
    python convert_gpt_to_csv.py --output results.csv         # custom output name
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.config import STAGE1_GPT_DIR, STAGE1_OUTPUT_DIR, DATA_DIR

# ─────────────────────────────────────────────────────────────────────────────
# UNICODE SANITIZATION (same as convert_tavily_to_csv.py)
# ─────────────────────────────────────────────────────────────────────────────

_UNICODE_REPLACEMENTS = {
    "\u202f": " ",   # narrow no-break space
    "\u00a0": " ",   # non-breaking space
    "\u2011": "-",   # non-breaking hyphen
    "\u2010": "-",   # hyphen
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u2026": "...", # ellipsis
    "\u00b7": ".",   # middle dot
}
_UNICODE_PATTERN = re.compile("|".join(re.escape(k) for k in _UNICODE_REPLACEMENTS))


def sanitize_text(text: str) -> str:
    """Replace common Unicode characters with ASCII equivalents."""
    if not text:
        return text
    return _UNICODE_PATTERN.sub(lambda m: _UNICODE_REPLACEMENTS[m.group()], text)


# ─────────────────────────────────────────────────────────────────────────────
# CRUNCHBASE ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_crunchbase_lookup() -> dict[int, dict]:
    """Load Crunchbase CSV into a dict keyed by rcid for fast lookup."""
    csv_path = DATA_DIR / "44k_crunchbase_startups.csv"
    lookup = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rcid = int(row["rcid"])
            except (ValueError, KeyError):
                continue
            lookup[rcid] = {
                "short_description": row.get("short_description") or "",
                "category_list": row.get("category_list") or "",
                "category_groups_list": row.get("category_groups_list") or "",
                "homepage_url": row.get("homepage_url") or "",
                "founded_date": row.get("founded_date") or "",
            }
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# CSV CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "rcid",
    "name",
    "short_description",
    "category_list",
    "category_groups_list",
    "homepage_url",
    "founded_date",
    "online_presence_score",
    "research_priority_score",
    "reasoning",
    "error",
]


def find_latest_gpt_jsonl() -> Path:
    """Find the most recently modified gpt_*.jsonl file."""
    files = sorted(STAGE1_GPT_DIR.glob("gpt_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        print(f"ERROR: No gpt_*.jsonl files found in {STAGE1_GPT_DIR}")
        sys.exit(1)
    return files[-1]


def convert(input_path: Path, output_path: Path) -> int:
    """Read GPT JSONL, enrich with Crunchbase, write CSV. Returns row count."""
    print(f"Loading Crunchbase data for enrichment...")
    cb_lookup = load_crunchbase_lookup()

    rows_written = 0

    with open(input_path, "r") as fin, \
         open(output_path, "w", newline="", encoding="utf-8") as fout:

        writer = csv.DictWriter(fout, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping malformed line {line_no}: {e}")
                continue

            rcid = int(record["rcid"])
            cb = cb_lookup.get(rcid, {})

            row = {
                "rcid": rcid,
                "name": record.get("name", ""),
                "short_description": cb.get("short_description", ""),
                "category_list": cb.get("category_list", ""),
                "category_groups_list": cb.get("category_groups_list", ""),
                "homepage_url": cb.get("homepage_url", ""),
                "founded_date": cb.get("founded_date", ""),
                "online_presence_score": record.get("online_presence_score", ""),
                "research_priority_score": record.get("research_priority_score", ""),
                "reasoning": record.get("reasoning", ""),
                "error": record.get("error", ""),
            }

            # Sanitize all string fields
            row = {k: sanitize_text(v) if isinstance(v, str) else v for k, v in row.items()}

            writer.writerow(row)
            rows_written += 1

    return rows_written


def main():
    parser = argparse.ArgumentParser(description="Convert GPT JSONL to CSV")
    parser.add_argument("--input", type=str, default=None,
                        help="GPT JSONL file (default: latest in outputs/stage1/gpt/)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: same name as input with .csv)")
    args = parser.parse_args()

    # Resolve input
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = STAGE1_GPT_DIR / input_path
    else:
        input_path = find_latest_gpt_jsonl()

    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    # Resolve output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".csv")

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    rows = convert(input_path, output_path)
    print(f"Done. Wrote {rows} rows to {output_path}")


if __name__ == "__main__":
    main()
