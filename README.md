# Mixpanel Spark AI — Braintrust Demo

Seeds a Braintrust project with realistic Spark AI data: prompts, scorers, golden datasets, experiments, and production traces.

Spark AI is Mixpanel's NL-to-analytics agent — users ask questions in plain English and Spark translates them into Mixpanel reports (Insights, Funnels, Retention, Flows).

## What gets created

| Resource | Details |
|---|---|
| **2 prompts** | Structured (query-focused) vs Conversational (insight-driven) |
| **3 scorers** | Query correctness, Schema adherence, Response quality |
| **1 online scoring rule** | Runs all 3 scorers on production traces at 100% sampling |
| **4 facets** | Report type, Query complexity, Failure mode, Customer vertical |
| **1 topic automation** | Auto-clusters traces by Sentiment, Task, Issues, Failure mode |
| **1 golden dataset** | ~34 curated test cases across all report types + edge cases |
| **4 experiments** | Prompt A vs B, gpt-4.1-mini vs gpt-4.1-nano (with scores) |
| **500 traces** | Realistic Spark AI queries across 4 customer verticals |
| **3 derived datasets** | Production samples, Flagged for review, Playground scenarios |

## Setup

```bash
pip install -r requirements.txt
```

Set environment variables:

```bash
export BRAINTRUST_API_KEY="your-braintrust-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

Optionally set a custom API URL (defaults to `https://api.braintrust.dev`):

```bash
export BRAINTRUST_API_URL="https://your-instance.braintrust.dev"
```

## Usage

Full seed (everything):

```bash
python seed_spark_ai.py --project "Mixpanel Spark AI" --count 500
```

Prompts, scorers, and dataset only (no OpenAI calls needed):

```bash
python seed_spark_ai.py --project "Mixpanel Spark AI" --skip-traces --skip-experiments
```

Experiments only (requires OpenAI key, ~136 API calls):

```bash
python seed_spark_ai.py --project "Mixpanel Spark AI" --skip-traces
```

Traces only (requires OpenAI key, assumes prompts/scorers exist):

```bash
python seed_spark_ai.py --project "Mixpanel Spark AI" --traces-only --count 500
```

## Options

| Flag | Description |
|---|---|
| `--project` | Braintrust project name (default: "Mixpanel Spark AI") |
| `--count` | Number of traces to generate (default: 500) |
| `--parallelism` | Concurrent trace generation limit (default: 50) |
| `--skip-traces` | Skip trace generation |
| `--skip-experiments` | Skip A/B experiments |
| `--traces-only` | Only generate traces, skip everything else |

## Cost estimate

| Operation | OpenAI API calls | Estimated cost |
|---|---|---|
| Experiments (4 runs x 34 rows) | ~136 | ~$0.50 |
| Single-turn traces (460) | ~460 | ~$1.50 |
| Multi-turn traces (40 x ~3 turns) | ~120 | ~$0.40 |
| **Total** | **~716** | **~$2.40** |
