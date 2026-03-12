# Coding Instructions

General development guidelines for AI-assisted coding in this project.

## Python Style

- Use type hints on all function signatures and dataclass fields.
- Prefer `dataclasses` over plain dicts for structured data.
- Use `pathlib.Path` over `os.path` for all file operations.
- Imports: stdlib → third-party → local, separated by blank lines.

## Project Structure

- **Library code** lives in `src/` (organized by stage: `src/stage_1/`, `src/stage_2/`, `src/common/`).
- **Runners / scripts** live at project root or in `src/tests/`.
- **Prompts** live in `prompts/`.
- **Configuration** is centralized in `src/config.py` — all paths, thresholds, and cost estimates are defined there.
- **Credentials** are loaded from `credentials/` files (gitignored) with env-var fallback. Never hardcode keys.

## Data Conventions

- **JSONL** for incremental writes and intermediate results (one JSON object per line, append-friendly).
- **CSV** for final deliverables and EDA-ready datasets.
- **Reproducible seeds**: always expose a `--seed` CLI arg (default `2026`) for any sampling.
- Output files go under `outputs/` (gitignored). Stage-specific subdirs: `outputs/stage1/`, `outputs/stage2/`.

## Error Handling

- Graceful failures with error fields in result objects (see `ResearchResult.error` pattern in `src/stage_2/perplexity_client.py`).
- Never let one bad API call crash a batch run. Catch exceptions, record the error, continue.
- Log errors with enough context to diagnose without re-running (company ID, error type, truncated message).

## API Call Discipline

- **Before every API run**: review code for bugs, null-safety issues, and response-parsing edge cases. API credits are expensive and non-refundable. Catch problems in code review, not in production.
- Test with a single-call dry run or mock when possible before launching a batch.
- Treat API fields as potentially null even when SDK types say otherwise — always use defensive access (`or []`, `getattr(..., None)`, etc.).

## CLI & Scripts

- Use `argparse` for all CLI scripts.
- Support resume: check for already-processed IDs in the output file before re-processing.
- Support graceful shutdown via `SIGINT` handler — finish current item, then stop.
- Log to both stdout (concise) and a timestamped log file (detailed).

## Git Conventions

The commit history is a narrative — someone reading it should be able to follow the evolution of the project and understand the value added and the reasoning behind each step.

### Title Line

- Imperative mood ("Add budget-aware preset routing" not "Added routing").
- Describe the **value delivered**, not the mechanical change. The title should answer "what does the project gain from this commit?"
- Good: "Add structured JSON schema to enforce research output quality"
- Bad: "Update perplexity_client.py"
- Good: "Split test runner into configurable preset framework for A/B cost experiments"
- Bad: "Refactor run_stage2_test.py into src/tests/"

### Commit Body

- Explain **why** this change was made — the reasoning, the trade-off, the constraint that motivated it.
- Reference the bigger picture when relevant (budget constraints, pipeline stage, research methodology).
- Keep it concise but substantive — 1–3 sentences is typical.

### Commit Frequency

- Commit at meaningful milestones, not after every file edit.
- Each commit should represent a coherent unit of progress that someone could review independently.

### Other Rules

- No Cursor co-author trail.
- No `--amend` unless HEAD is ours and unpushed.
- Never commit credentials, API keys, or large data files.

## Dependencies

- Pin versions in `requirements.txt`.
- Keep dependencies minimal — prefer stdlib when possible.
