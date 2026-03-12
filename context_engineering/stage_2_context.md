# Stage 2 Context

Full context for building and running Stage 2 of the GenAI adoption research pipeline.

## Stage 1 Output Summary

Stage 1 scored ~44k US startups from Crunchbase using Tavily web search + GPT-5-nano classification.

- **Input**: `crunchbase_data/44k_crunchbase_startups.csv` — the full Crunchbase dataset.
- **Output**: `outputs/stage1/gpt/gpt_full_44k.jsonl` — one JSONL record per company with:
  - `rcid`: Crunchbase company ID (integer, primary key).
  - `name`: Company name.
  - `research_priority_score`: 0–5 integer. Higher = more likely to have GenAI adoption evidence.
  - `online_presence_score`: 0–5 integer. Higher = more findable online.
  - `error`: null if processed successfully.
- **Distribution**: ~4.4k companies scored priority=5, ~5k scored priority=4.

## The New Dataset (Stage 2 Input)

Priority 4+5 companies (~9.4k total) are the deep-research candidates.

- Extracted from Stage 1 results joined with Crunchbase metadata.
- Written to `outputs/stage2/stage2_input_dataset.jsonl`.
- Fields per record: `rcid`, `name`, `homepage_url`, `short_description`, `research_priority_score`, `online_presence_score`, `category_list`.

## Overarching Goal

Produce an **EDA-ready CSV** where each row is one finding of a startup using GenAI internally.

- Maximize findings count within a **$4k USD** budget.
- Academic rigor: every finding must have a verifiable `source_url`.
- Critical distinction: we want evidence of companies **using** GenAI tools for internal operations, **not** companies that sell/build AI products.

## Budget Constraint Analysis

- $4k total budget for all Stage 2 deep research.
- At ~$0.10/company average: can research ~40k companies (full coverage possible).
- At ~$0.50/company average: can research ~8k (would need to prioritize).
- At ~$1.00/company average: can research ~4k (priority=5 only).
- **Testing must determine actual cost/company for each preset to make this decision.**

## Perplexity Agent API

The deep research is performed via Perplexity's Agent API (`perplexity` Python SDK).

### Presets

| Preset | Model | Description |
|---|---|---|
| `deep-research` | GPT-5.2 | ~10 reasoning steps, web search |
| `advanced-deep-research` | Claude Opus 4.6 | ~10 reasoning steps, web search |

### Key API Details

- **Structured output**: `response_format` with JSON Schema enforces output structure.
- **Built-in tools**: `web_search` and `fetch_url` are used automatically by the agent.
- **Cost tracking**: `usage.cost.total_cost` on each response gives exact USD cost.
- **Client class**: `Stage2Client` in `src/stage_2/perplexity_client.py` wraps all of this.

### Response Schema

Each API call returns a structured JSON with:
- `company_id`, `company_name`: identifiers.
- `genai_adoption_found`: boolean.
- `findings`: array of finding objects (use_case, business_function, evidence_description, source_url, source_type).
- `no_finding_reason`: `"no_evidence"` | `"insufficient_information"` | null.

## Existing Code

| File | What It Does |
|---|---|
| `src/stage_2/perplexity_client.py` | `Stage2Client`, `AgentTier`, `ResearchResult`, JSON schema, response parsing. Ready to use. |
| `src/config.py` | All paths (`STAGE2_OUTPUT_DIR`, `STAGE1_GPT_DIR`, `DATA_DIR`), `APIKeys`, `CostEstimates`. |
| `src/common/jsonl_writer.py` | `AsyncJSONLWriter` for incremental file writes. |
| `run_stage2_test.py` | Root-level test runner with data loading, sampling, batch execution, resume, graceful shutdown. To be refactored into `src/tests/stage2/run_preset_test.py`. |
| `prompts/stage_2_perplexity_prompt.txt` | The unified research prompt with USE vs SELL distinction, few-shot examples, and output schema. Finalized. |

## Testing Plan

### Phase 1 — Pilot (est. $1–5)

- 5 companies from priority=5, 5 from priority=4.
- Run both `deep-research` and `advanced-deep-research` on each (4 batches, 20 total API calls).
- Goal: validate pipeline end-to-end, confirm costs are in expected range, check output schema compliance.

### Phase 2 — Statistical (est. $10–60)

- 50–100 companies from priority=5, 50–100 from priority=4.
- Run the more promising preset(s) based on Phase 1 results.
- Goal: statistically meaningful cost/company and findings rate data.

### Phase 3 — Analysis

- Run `analyze_results.py` on Phase 2 outputs.
- Key decisions: Is `advanced-deep-research` worth the cost premium? Should we scale on priority=4? What's the projected total cost? Do we need custom model configs?

### Phase 4 — Scale

- Execute on the full dataset within budget using the chosen configuration.
- Output: final EDA-ready CSV.
