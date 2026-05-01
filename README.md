# Braintrust demo seed generator

Spin up a fully populated Braintrust demo project — prompts, scorers, facets, golden dataset, experiments, and 500 production-style traces — for **any customer's AI product**, by passing in a company name and website. The generator researches the product, synthesizes domain-specific data, and emits a runnable seed script plus a customer-specific requirements-mapping doc.

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set keys
export OPENAI_API_KEY="sk-..."
export BRAINTRUST_API_KEY="..."
# Optional: point at a non-prod Braintrust instance
export BRAINTRUST_API_URL="https://api.braintrust.dev"

# 3. Generate the seed script for a customer
#    Args: <Company name> <Website URL>
python -m generator "Zendesk" "zendesk.com"
#    → writes generated/seed_zendesk.py
#    → writes generated/Braintrust × Zendesk - Requirements Mapping.md
#    → caches each LLM step to generated/.cache/zendesk/ for cheap re-runs

# 4. Run the generated script against your Braintrust org
python generated/seed_zendesk.py --project "Zendesk Answer Bot" --count 500
```

That's it. The new project will appear in Braintrust with everything wired up.

### Optional generator flags

| Flag | Description |
|---|---|
| `--context "Focus on Answer Bot"` | Extra hint for the research phase |
| `--research-file path/to.json` | Skip web research, reuse a saved research report |
| `--output-dir custom/` | Where to write the generated script (default: `generated/`) |
| `--skip-requirements-doc` | Don't generate the markdown requirements doc |
| `--no-cache` | Disable per-step LLM caching (default: cache enabled) |
| `--dry-run` | Run research + synthesis + validation, but don't write files |
| `--verbose` | Print intermediate output |

### Optional flags on the generated seed script

| Flag | Description |
|---|---|
| `--project` | Braintrust project name |
| `--count` | Number of traces to generate (default: 500) |
| `--parallelism` | Concurrent trace generation limit (default: 50) |
| `--skip-traces` | Skip trace generation |
| `--skip-experiments` | Skip A/B experiments |
| `--traces-only` | Only generate traces, skip everything else |

### Resuming after a failure

Each of the 10 LLM synthesis steps is cached under `generated/.cache/<customer>/`. If step 9 fails, fix the issue, re-run, and the prior 8 steps reuse cached output. To force one step to re-run, delete just that file (e.g. `rm generated/.cache/zendesk/06_scorers.json`).

---

## What the generated script creates in Braintrust

| Resource | Details |
|---|---|
| **2 prompts** | Two style variants (e.g. structured vs conversational) with version metadata |
| **3 LLM scorers + 2 code scorers** | Domain-specific quality scorers + grounding/schema check + thread coherence |
| **1 online scoring rule** | Runs all scorers on production traces at 100% sampling |
| **4 custom facets** | Auto-classify traces (feature mode, complexity, failure mode, vertical) |
| **1 topic automation** | Auto-clusters traces by sentiment/task/issues/failure mode |
| **1 golden dataset** | ~30+ curated test cases across all feature modes + edge cases |
| **4 experiments** | Prompt A vs B, plus gpt-5-mini vs gpt-5-nano model comparison |
| **500 traces** | Realistic single-turn + multi-turn conversations across customer verticals |
| **3 derived datasets** | Production samples, flagged-for-review, playground scenarios |

---

## Cost estimate (per generated demo)

| Phase | Calls | Cost |
|---|---|---|
| Research (web search + synthesis) | ~4 | ~$0.05 |
| Synthesis (10 steps) | 10 | ~$0.10 |
| Requirements-mapping doc | 1 | ~$0.02 |
| **Generation total** | **~15** | **~$0.20** |
| Experiments (4 runs × ~30 rows) | ~120 | ~$0.40 |
| Single-turn traces (~460) | ~460 | ~$1.50 |
| Multi-turn traces (40 × ~3 turns) | ~120 | ~$0.40 |
| **Run total** | **~700** | **~$2.30** |

Caching makes regeneration of the same customer essentially free.

---

## Project layout

```
.
├── README.md
├── requirements.txt
├── generator/
│   ├── __main__.py                               # `python -m generator`
│   ├── cli.py                                    # argument parsing + orchestration
│   ├── research.py                               # phase 1: web research
│   ├── synthesize.py                             # phase 2: 10 LLM calls (cached)
│   ├── assemble.py                               # phase 3: substitute into template
│   ├── validate.py                               # phase 4: structural checks
│   ├── requirements_doc.py                       # phase 5: customer requirements doc
│   ├── models.py
│   ├── prompts/                                  # one .txt per synthesis step
│   └── template/seed_template.py                 # the generated-script skeleton
└── generated/                                    # output directory
    ├── .cache/<customer>/                        # per-step LLM caches (gitignored)
    └── seed_<customer>.py
```
