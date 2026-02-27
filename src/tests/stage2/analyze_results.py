"""
Analyze Stage 2 test results for cost, findings rate, and budget projection.

Reads one or more test output JSONL files (produced by run_preset_test.py)
and reports:
  - Cost summary: avg/median/min/max per company, by preset and priority.
  - Findings rate: adoption %, avg findings per company.
  - Error rate.
  - Budget projection: how many companies can be researched for the budget.
  - Sample findings for manual quality review.

Usage:
    python -m src.tests.stage2.analyze_results outputs/stage2/pilot_*.jsonl
    python -m src.tests.stage2.analyze_results results.jsonl --budget 4000 --csv summary.csv
"""

import argparse
import csv
import json
import statistics
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResultRecord:
    """One JSONL record from a test run."""
    rcid: int
    company_name: str
    research_priority_score: int
    preset: str
    genai_adoption_found: bool
    findings: list[dict]
    no_finding_reason: Optional[str]
    cost: Optional[float]
    duration_seconds: Optional[float]
    error: Optional[str]

    @classmethod
    def from_dict(cls, d: dict) -> "ResultRecord":
        return cls(
            rcid=int(d.get("rcid", d.get("company_id", 0))),
            company_name=d.get("company_name", d.get("name", "")),
            research_priority_score=int(d.get("research_priority_score", 0)),
            preset=d.get("preset", "unknown"),
            genai_adoption_found=bool(d.get("genai_adoption_found", False)),
            findings=d.get("findings", []),
            no_finding_reason=d.get("no_finding_reason"),
            cost=d.get("cost"),
            duration_seconds=d.get("duration_seconds"),
            error=d.get("error"),
        )


@dataclass
class GroupStats:
    """Aggregated statistics for a group of results."""
    label: str
    count: int = 0
    errors: int = 0
    adoption_found: int = 0
    total_findings: int = 0
    costs: list[float] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return self.count - self.errors

    @property
    def error_rate(self) -> float:
        return self.errors / self.count if self.count else 0.0

    @property
    def adoption_rate(self) -> float:
        return self.adoption_found / self.success_count if self.success_count else 0.0

    @property
    def avg_findings(self) -> float:
        return self.total_findings / self.success_count if self.success_count else 0.0

    @property
    def cost_avg(self) -> float:
        return statistics.mean(self.costs) if self.costs else 0.0

    @property
    def cost_median(self) -> float:
        return statistics.median(self.costs) if self.costs else 0.0

    @property
    def cost_min(self) -> float:
        return min(self.costs) if self.costs else 0.0

    @property
    def cost_max(self) -> float:
        return max(self.costs) if self.costs else 0.0

    @property
    def cost_total(self) -> float:
        return sum(self.costs)

    @property
    def cost_stdev(self) -> float:
        return statistics.stdev(self.costs) if len(self.costs) >= 2 else 0.0

    @property
    def duration_avg(self) -> float:
        return statistics.mean(self.durations) if self.durations else 0.0

    def add(self, record: ResultRecord) -> None:
        self.count += 1
        if record.error:
            self.errors += 1
            return
        self.adoption_found += int(record.genai_adoption_found)
        self.total_findings += len(record.findings)
        if record.cost is not None:
            self.costs.append(record.cost)
        if record.duration_seconds is not None:
            self.durations.append(record.duration_seconds)


# ─────────────────────────────────────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_results(paths: list[Path]) -> list[ResultRecord]:
    """Load and parse all JSONL result files."""
    records: list[ResultRecord] = []
    parse_errors = 0

    for path in paths:
        if not path.exists():
            print(f"  Warning: {path} not found, skipping")
            continue
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(ResultRecord.from_dict(json.loads(line)))
                except (json.JSONDecodeError, Exception):
                    parse_errors += 1

    if parse_errors:
        print(f"  Warning: {parse_errors} unparseable lines skipped")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(records: list[ResultRecord]) -> dict[str, GroupStats]:
    """Compute statistics grouped by (preset, priority) and overall.

    Returns a dict of label -> GroupStats for:
      - "overall"
      - each unique preset
      - each unique "preset | priority=N" combination
    """
    overall = GroupStats(label="overall")
    by_preset: dict[str, GroupStats] = defaultdict(lambda: GroupStats(label=""))
    by_combo: dict[str, GroupStats] = defaultdict(lambda: GroupStats(label=""))

    for r in records:
        overall.add(r)

        preset_key = r.preset
        if preset_key not in by_preset:
            by_preset[preset_key] = GroupStats(label=preset_key)
        by_preset[preset_key].add(r)

        combo_key = f"{r.preset} | priority={r.research_priority_score}"
        if combo_key not in by_combo:
            by_combo[combo_key] = GroupStats(label=combo_key)
        by_combo[combo_key].add(r)

    groups: dict[str, GroupStats] = {"overall": overall}
    for k, v in sorted(by_preset.items()):
        groups[k] = v
    for k, v in sorted(by_combo.items()):
        groups[k] = v

    return groups


def format_cost_table(groups: dict[str, GroupStats]) -> str:
    """Format cost summary as an aligned text table."""
    header = f"{'Group':<45} {'N':>5} {'Err%':>6} {'Avg$':>8} {'Med$':>8} {'Min$':>8} {'Max$':>8} {'Std$':>8} {'Total$':>9}"
    sep = "─" * len(header)
    lines = [sep, header, sep]

    for label, g in groups.items():
        if not g.count:
            continue
        lines.append(
            f"{label:<45} {g.count:>5} {g.error_rate:>5.1%} "
            f"{g.cost_avg:>8.4f} {g.cost_median:>8.4f} {g.cost_min:>8.4f} "
            f"{g.cost_max:>8.4f} {g.cost_stdev:>8.4f} {g.cost_total:>9.2f}"
        )

    lines.append(sep)
    return "\n".join(lines)


def format_findings_table(groups: dict[str, GroupStats]) -> str:
    """Format findings rate summary as an aligned text table."""
    header = f"{'Group':<45} {'OK':>5} {'Found':>6} {'Rate':>7} {'Avg#':>6} {'Total#':>7}"
    sep = "─" * len(header)
    lines = [sep, header, sep]

    for label, g in groups.items():
        if not g.count:
            continue
        lines.append(
            f"{label:<45} {g.success_count:>5} {g.adoption_found:>6} "
            f"{g.adoption_rate:>6.1%} {g.avg_findings:>6.2f} {g.total_findings:>7}"
        )

    lines.append(sep)
    return "\n".join(lines)


def format_budget_projection(
    groups: dict[str, GroupStats],
    budget: float,
    priority_5_count: int,
    priority_4_count: int,
) -> str:
    """Project how many companies can be researched within the budget."""
    lines: list[str] = []

    preset_groups = {k: v for k, v in groups.items() if k != "overall" and "|" not in k}

    for label, g in preset_groups.items():
        if not g.costs:
            continue

        avg = g.cost_avg
        med = g.cost_median
        if avg <= 0:
            continue

        can_research_avg = int(budget / avg)
        can_research_med = int(budget / med) if med > 0 else 0

        lines.append(f"  {label}:")
        lines.append(f"    At avg ${avg:.4f}/company: ~{can_research_avg:,} companies (${budget:,.0f} budget)")
        lines.append(f"    At med ${med:.4f}/company: ~{can_research_med:,} companies")
        lines.append(f"    Priority=5 only ({priority_5_count:,}): ${priority_5_count * avg:,.2f}")
        lines.append(f"    Priority=4+5   ({priority_5_count + priority_4_count:,}): ${(priority_5_count + priority_4_count) * avg:,.2f}")

        if (priority_5_count + priority_4_count) * avg <= budget:
            lines.append(f"    --> Full coverage of priority 4+5 is within budget")
        elif priority_5_count * avg <= budget:
            remaining = budget - priority_5_count * avg
            p4_possible = int(remaining / avg)
            lines.append(f"    --> Priority=5 full + ~{p4_possible:,} of priority=4 within budget")
        else:
            lines.append(f"    --> Budget insufficient for full priority=5 coverage")

        lines.append("")

    if not lines:
        lines.append("  No cost data available for budget projection.")

    return "\n".join(lines)


def format_sample_findings(records: list[ResultRecord], n: int = 10) -> str:
    """Pick up to n sample findings for manual quality review.

    Selects findings from distinct companies, prioritizing those with
    the most findings for richer examples.
    """
    candidates: list[tuple[ResultRecord, dict]] = []
    for r in records:
        if r.error or not r.findings:
            continue
        for f in r.findings:
            candidates.append((r, f))

    seen_companies: set[int] = set()
    selected: list[tuple[ResultRecord, dict]] = []

    candidates.sort(key=lambda x: len(x[0].findings), reverse=True)
    for r, f in candidates:
        if r.rcid in seen_companies:
            continue
        seen_companies.add(r.rcid)
        selected.append((r, f))
        if len(selected) >= n:
            break

    if not selected:
        return "  No findings available for review."

    lines: list[str] = []
    for i, (r, f) in enumerate(selected, 1):
        use_case = f.get("use_case", "N/A")
        biz_fn = f.get("business_function", "N/A")
        evidence = f.get("evidence_description", f.get("evidence_summary", "N/A"))
        source = f.get("source_url", "N/A")
        source_type = f.get("source_type", "")

        lines.append(f"  [{i}] {r.company_name} (rcid={r.rcid}, preset={r.preset}, priority={r.research_priority_score})")
        lines.append(f"      Use case: {use_case}")
        lines.append(f"      Function: {biz_fn}")
        wrapped = textwrap.fill(str(evidence), width=90, initial_indent="      Evidence: ", subsequent_indent="               ")
        lines.append(wrapped)
        lines.append(f"      Source:   {source}" + (f" ({source_type})" if source_type else ""))
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_summary_csv(groups: dict[str, GroupStats], output_path: Path) -> None:
    """Write group-level summary statistics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "group", "count", "errors", "error_rate",
        "adoption_found", "adoption_rate", "total_findings", "avg_findings",
        "cost_avg", "cost_median", "cost_min", "cost_max", "cost_stdev", "cost_total",
        "duration_avg",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, g in groups.items():
            writer.writerow({
                "group": label,
                "count": g.count,
                "errors": g.errors,
                "error_rate": round(g.error_rate, 4),
                "adoption_found": g.adoption_found,
                "adoption_rate": round(g.adoption_rate, 4),
                "total_findings": g.total_findings,
                "avg_findings": round(g.avg_findings, 3),
                "cost_avg": round(g.cost_avg, 6),
                "cost_median": round(g.cost_median, 6),
                "cost_min": round(g.cost_min, 6),
                "cost_max": round(g.cost_max, 6),
                "cost_stdev": round(g.cost_stdev, 6),
                "cost_total": round(g.cost_total, 4),
                "duration_avg": round(g.duration_avg, 2),
            })

    print(f"\nSummary CSV written to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Stage 2 test results: cost, findings rate, budget projection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m src.tests.stage2.analyze_results outputs/stage2/pilot_*.jsonl
              python -m src.tests.stage2.analyze_results results.jsonl --budget 4000 --csv summary.csv
              python -m src.tests.stage2.analyze_results outputs/stage2/*.jsonl --sample-count 15
        """),
    )
    parser.add_argument(
        "files",
        type=Path,
        nargs="+",
        help="One or more test output JSONL files to analyze",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=4000.0,
        help="Total budget in USD for scaling projection (default: 4000)",
    )
    parser.add_argument(
        "--priority-5-count",
        type=int,
        default=4400,
        help="Total priority=5 companies in the full dataset (default: 4400)",
    )
    parser.add_argument(
        "--priority-4-count",
        type=int,
        default=5000,
        help="Total priority=4 companies in the full dataset (default: 5000)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write summary statistics to a CSV file",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=10,
        help="Number of sample findings to display (default: 10)",
    )
    args = parser.parse_args()

    # Load
    print(f"Loading results from {len(args.files)} file(s)...")
    records = load_results(args.files)
    if not records:
        print("No records loaded. Check file paths.")
        sys.exit(1)
    print(f"  {len(records)} records loaded\n")

    # Compute
    groups = compute_stats(records)

    # Report: Cost
    print("=" * 80)
    print("COST SUMMARY")
    print("=" * 80)
    print(format_cost_table(groups))
    print()

    # Report: Findings rate
    print("=" * 80)
    print("FINDINGS RATE")
    print("=" * 80)
    print(format_findings_table(groups))
    print()

    # Report: Budget projection
    print("=" * 80)
    print(f"BUDGET PROJECTION (${args.budget:,.0f})")
    print("=" * 80)
    print(format_budget_projection(groups, args.budget, args.priority_5_count, args.priority_4_count))

    # Report: Sample findings
    print("=" * 80)
    print(f"SAMPLE FINDINGS ({args.sample_count} examples)")
    print("=" * 80)
    print(format_sample_findings(records, n=args.sample_count))

    # Optional CSV
    if args.csv:
        write_summary_csv(groups, args.csv)


if __name__ == "__main__":
    main()
