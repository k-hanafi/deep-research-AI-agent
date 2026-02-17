"""
Convert Tavily JSONL results to an Excel-friendly CSV for manual inspection.

Flattens each JSONL record into one row with company info, website check
summary, and up to 5 search result columns (title, url, snippet).

Usage:
    python convert_tavily_to_csv.py                          # default input/output
    python convert_tavily_to_csv.py --input custom.jsonl     # custom input
    python convert_tavily_to_csv.py --output results.csv     # custom output
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.config import STAGE1_OUTPUT_DIR

# Defaults
DEFAULT_INPUT = STAGE1_OUTPUT_DIR / "tavily_results.jsonl"
DEFAULT_OUTPUT = STAGE1_OUTPUT_DIR / "tavily_results.csv"

MAX_RESULTS = 5
SNIPPET_MAX_CHARS = None  # No truncation — show full Tavily content

# Unicode → ASCII replacements for Excel compatibility
_UNICODE_REPLACEMENTS = {
    "\u202f": " ",   # narrow no-break space → regular space
    "\u00a0": " ",   # non-breaking space → regular space
    "\u2011": "-",   # non-breaking hyphen → regular hyphen
    "\u2010": "-",   # hyphen → regular hyphen
    "\u2013": "-",   # en dash → hyphen
    "\u2014": "-",   # em dash → hyphen
    "\u2018": "'",   # left single quote → apostrophe
    "\u2019": "'",   # right single quote → apostrophe
    "\u201c": '"',   # left double quote → straight quote
    "\u201d": '"',   # right double quote → straight quote
    "\u2026": "...", # ellipsis → three dots
    "\u00b7": ".",   # middle dot → period
}
_UNICODE_PATTERN = re.compile("|".join(re.escape(k) for k in _UNICODE_REPLACEMENTS))


def sanitize_text(text: str) -> str:
    """Replace common Unicode characters with ASCII equivalents for Excel."""
    if not text:
        return text
    return _UNICODE_PATTERN.sub(lambda m: _UNICODE_REPLACEMENTS[m.group()], text)


def build_csv_columns() -> list[str]:
    """Build the flat CSV column list."""
    cols = [
        "rcid",
        "name",
        "short_description",
        "homepage_url",
        "website_alive",
        "website_status_code",
        "website_error",
        "tavily_query",
        "tavily_answer",
        "tavily_result_count",
        "tavily_error",
    ]
    for i in range(1, MAX_RESULTS + 1):
        cols.append(f"result_{i}_title")
        cols.append(f"result_{i}_url")
        cols.append(f"result_{i}_snippet")
    cols.append("timestamp")
    return cols


def flatten_record(record: dict) -> dict:
    """Flatten one JSONL record into a flat dict matching CSV columns."""
    wc = record.get("website_check", {})
    tv = record.get("tavily", {})
    raw = tv.get("raw_response") or {}

    row = {
        "rcid": record.get("rcid"),
        "name": record.get("name"),
        "short_description": record.get("short_description") or "",
        "homepage_url": record.get("homepage_url") or "",
        "website_alive": wc.get("is_alive", False),
        "website_status_code": wc.get("status_code") or "",
        "website_error": wc.get("error") or "",
        "tavily_query": tv.get("query") or "",
        "tavily_answer": tv.get("answer") or raw.get("answer") or "",
        "tavily_result_count": tv.get("result_count", 0),
        "tavily_error": tv.get("error") or "",
        "timestamp": record.get("timestamp") or "",
    }

    # Flatten search results (up to MAX_RESULTS)
    results = raw.get("results", [])
    for i in range(1, MAX_RESULTS + 1):
        if i <= len(results):
            r = results[i - 1]
            content = r.get("content") or ""
            row[f"result_{i}_title"] = r.get("title") or ""
            row[f"result_{i}_url"] = r.get("url") or ""
            row[f"result_{i}_snippet"] = content
        else:
            row[f"result_{i}_title"] = ""
            row[f"result_{i}_url"] = ""
            row[f"result_{i}_snippet"] = ""

    # Sanitize all string values for Excel compatibility
    return {k: sanitize_text(v) if isinstance(v, str) else v for k, v in row.items()}


def convert(input_path: Path, output_path: Path) -> int:
    """
    Read JSONL, write CSV. Returns number of rows written.
    """
    columns = build_csv_columns()
    rows_written = 0

    with open(input_path, "r") as fin, \
         open(output_path, "w", newline="", encoding="utf-8") as fout:

        writer = csv.DictWriter(fout, fieldnames=columns)
        writer.writeheader()

        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                row = flatten_record(record)
                writer.writerow(row)
                rows_written += 1
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping malformed line {line_no}: {e}")
            except Exception as e:
                print(f"  Warning: error on line {line_no}: {e}")

    return rows_written


def main():
    parser = argparse.ArgumentParser(
        description="Convert Tavily JSONL results to CSV for Excel inspection"
    )
    parser.add_argument("--input", type=str, default=None,
                        help=f"Input JSONL path (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=str, default=None,
                        help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else DEFAULT_INPUT
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    rows = convert(input_path, output_path)

    print(f"Done. Wrote {rows} rows to {output_path}")


if __name__ == "__main__":
    main()
