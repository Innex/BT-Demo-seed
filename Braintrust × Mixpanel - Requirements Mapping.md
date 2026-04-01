# Braintrust Capabilities Mapped to Mixpanel Spark AI

## What is Spark AI?

Spark AI is Mixpanel's generative AI analytics agent. It translates natural language questions into Mixpanel reports — Insights, Funnels, Retention, and Flows — so anyone in an organization can query product, marketing, and revenue data without SQL or knowing the Mixpanel query builder.

**How it works**: A user types a question (e.g., "Show me conversion from Sign Up to first purchase by channel") → Spark receives the project's event taxonomy (event names, property names — not the underlying data) → an OpenAI model translates the question into Mixpanel query parameters → Mixpanel executes the query and renders a chart. Users can iterate with follow-up prompts to refine breakdowns, time ranges, and filters.

**Why correctness matters**: Spark AI users are making business decisions — changing pricing, allocating marketing spend, prioritizing features — based on the reports Spark generates. A wrong event selection, incorrect aggregation (total events vs unique users), or hallucinated property silently produces a chart that looks right but answers the wrong question. The cost of undetected errors is bad business decisions at scale.

---

## Security & Compliance

| Requirement | Braintrust Capability |
|---|---|
| **SOC 2 Type II** | Braintrust is SOC 2 Type II certified. [Docs: Security](https://www.braintrust.dev/docs/reference/security) |
| **Zero-retention / zero-training** | Customer data is never used to train any models. Braintrust acts as a proxy — data flows through but is never retained by upstream providers. Mirrors Mixpanel's own OpenAI zero-retention agreement. [Docs: AI Proxy](https://www.braintrust.dev/docs/guides/proxy) |
| **Encryption** | Data encrypted at rest (AES-256) and in transit (TLS 1.2+). |
| **RBAC / Separation of Duties** | Role-based access control at org and project level. API keys and service tokens with scoped permissions allow the Spark AI team to experiment while restricting production prompt deployments. [Docs: Access control](https://www.braintrust.dev/docs/admin/access-control) |
| **Self-hosted / VPC** | Available as a self-hosted deployment via Helm chart into your VPC, or as an isolated cloud instance. [Docs: Self-hosting](https://www.braintrust.dev/docs/admin/self-hosting) |
| **Data residency (EU)** | Braintrust supports EU-resident deployments. Important for Mixpanel's EU customers where Spark AI is disabled by default due to data residency constraints. |

---

## Scenario 1: Evaluating Spark AI Query Correctness

**The core challenge**: Spark translates natural language → Mixpanel query parameters. Every translation decision has a correctness dimension:

| Decision | What can go wrong |
|---|---|
| Report type selection | User asks about drop-off but Spark builds Insights instead of Funnels |
| Event selection | Spark picks "Session Started" when the user meant "Login" |
| Aggregation type | "How many users" → should be unique users, but Spark uses total events (inflated numbers) |
| Property breakdowns | User asks "by country" but Spark breaks down by a different property |
| Time range | "Last quarter" → Spark interprets as last 90 days instead of calendar Q |
| Filter application | User specifies "enterprise customers" but filter isn't applied |
| Schema adherence | Spark references an event or property that doesn't exist in the project |

**Versioned prompts managed outside application code**
Spark's NL-to-query system prompt — the instructions that tell the model how to interpret questions and map them to Mixpanel query parameters — can be versioned in Braintrust with full edit history and diffs. Prompt changes are tested and promoted through environments (staging → production) without code deploys. At runtime, `braintrust.load_prompt()` fetches the active version.
- [Docs: Prompts](https://www.braintrust.dev/docs/guides/prompts)
- [Docs: Playground](https://www.braintrust.dev/docs/evaluate/playgrounds)

**Custom evaluation metrics for NL-to-query accuracy**
Braintrust supports custom scorers as LLM-as-a-judge prompts (built directly in the UI, no code required) or as Python/TypeScript functions. Spark-specific scorers:
- **Query correctness**: Did Spark select the right report type, events, aggregation, time range, and filters for the question?
- **Schema adherence**: Did Spark only reference events and properties that exist in the customer's project? (Critical because Mixpanel recommends "removing duplicate event & property data and giving this data simple, distinguishable names" — taxonomy quality directly impacts Spark accuracy.)
- **Response quality**: Is the natural language response helpful, clear, and actionable for a non-technical user?
- [Docs: Write scorers](https://www.braintrust.dev/docs/evaluate/write-scorers)
- [Docs: Autoevals library](https://github.com/braintrustdata/autoevals)

**Side-by-side prompt comparison on a test dataset**
Experiments run a prompt variant against a golden dataset of (question → expected query) pairs. When the Spark team changes the system prompt — e.g., adding instructions for handling ambiguous questions, improving aggregation selection, or adapting to new report features — multiple experiments can be compared side-by-side showing exactly where version A outperforms version B across query types (Insights, Funnels, Retention, Flows) and complexity levels.
- [Docs: Run evaluations](https://www.braintrust.dev/docs/evaluate/run-evaluations)
- [Docs: Datasets](https://www.braintrust.dev/docs/evaluate/datasets)

---

## Scenario 2: Model Upgrades & Regression Testing

**The problem**: Spark was originally built on GPT-3.5 Turbo. Upgrading to newer models (GPT-4+, or evaluating alternatives like Claude) risks regressions — a model that's generally smarter might handle specific Mixpanel query patterns worse than the previous one.

**Golden dataset as regression suite**
A curated dataset of (NL question → correct Mixpanel query) pairs acts as the regression test suite. Each row includes the project schema (event names, property names) so the evaluation tests schema-grounded query generation, not just general language understanding. Rows cover:
- All 4 report types (Insights, Funnels, Retention, Flows)
- Simple, moderate, and complex queries
- Multi-turn follow-ups ("Now break that down by platform")
- Ambiguous queries that require clarification
- Schema mismatch queries (user asks about events that don't exist)
- Edge cases across customer verticals (B2B SaaS, Ecommerce, Fintech, Media)

**Model comparison experiments**
Run the same golden dataset against different model configurations. Braintrust produces per-row and aggregate metrics showing cost/quality/latency tradeoffs. Identify the cheapest model that meets correctness thresholds for each query complexity tier.
- [Docs: Run evaluations](https://www.braintrust.dev/docs/evaluate/run-evaluations)

**CI/CD deployment gating**
Experiments produce structured summaries. CI pipelines assert thresholds (e.g., "query correctness must be > 0.90 across all report types") and block deployment if any scorer fails. Prompt environments (staging → production) provide a controlled promotion workflow — a new Spark prompt must pass evaluation before any customer sees it.
- [Docs: CI/CD integration](https://www.braintrust.dev/docs/evaluate/ci-cd)
- [Docs: Environments](https://www.braintrust.dev/docs/guides/prompts#environments)

---

## Scenario 3: Production Observability for Spark AI

**End-to-end trace visibility**
Braintrust captures the full Spark AI pipeline as a multi-step trace:

```
spark-ai-query (root span)
├── Schema lookup (tool span)
│   Input: customer project ID
│   Output: event names, property names, property types
│
├── NL-to-query translation (LLM span — auto-captured by wrap_openai)
│   Input: system prompt + schema context + user question + conversation history
│   Output: natural language response + generated query parameters
│
└── Query validation (tool span)
    Input: generated query parameters
    Output: validation result (events exist? properties valid? aggregation reasonable?)
```

Each span includes input/output, latency, token counts, and metadata. `wrap_openai` captures the LLM span automatically with zero code changes.
- [Docs: Tracing](https://www.braintrust.dev/docs/guides/tracing)
- [Docs: Logging](https://www.braintrust.dev/docs/guides/logging)

**Cost and latency tracking**
Every Spark request's LLM span captures token usage (prompt, completion, cached), latency, and estimated cost. Dashboards slice by model, prompt version, report type, customer tier, query complexity, or customer vertical. BTQL enables ad-hoc analytics: "What's the P95 latency for complex funnel queries on enterprise accounts?"
- [Docs: Dashboards](https://www.braintrust.dev/docs/observe/dashboards)
- [Docs: SQL reference](https://www.braintrust.dev/docs/reference/sql)

**Quality drift detection**
Online scoring rules run Spark's scorers (query correctness, schema adherence) automatically on production requests at configurable sampling rates. Topic automations cluster traces by report type, query complexity, failure mode (wrong event, wrong aggregation, schema hallucination), and customer vertical. Log automations trigger Slack/webhook alerts when accuracy distributions shift.
- [Docs: Online scoring](https://www.braintrust.dev/docs/observe/score-online)
- [Docs: Automations](https://www.braintrust.dev/docs/admin/projects#automations)

---

## Scenario 4: Handling Follow-up Queries & Multi-turn Conversations

Spark AI supports iterative refinement — users ask a question, then follow up to add breakdowns, change time ranges, or drill deeper. Multi-turn accuracy is harder than single-turn because the model must:
1. Maintain context from the previous query
2. Correctly interpret relative modifications ("break that down by platform", "what about just enterprise users?")
3. Modify the existing query without breaking what was already correct

**Multi-turn golden dataset rows**
Test cases include conversation histories so evaluations measure whether Spark correctly handles follow-up modifications without regressing the original query. Example:
- Turn 1: "Show me daily active users over the last 30 days" → Insights report
- Turn 2: "Can you break that down by platform?" → Add platform breakdown to existing report
- Turn 3: "What about just enterprise users?" → Add plan_type filter without losing the platform breakdown

**Thread-level scoring**
Braintrust's `{{thread}}` variable captures the full conversation across LLM spans. Scorers evaluate the entire multi-turn sequence end-to-end, not just individual responses.
- [Docs: Write scorers](https://www.braintrust.dev/docs/evaluate/write-scorers)

---

## Scenario 5: Human-in-the-Loop for Spark AI Quality

**Surfacing problematic queries for review**
Production traces where Spark generates incorrect or low-confidence queries are filtered and routed to human reviewers. Reviewers can:
- See the user's question, the generated query, the schema context, and the validation result
- Score the trace (correct / partially correct / incorrect)
- Correct the query and add the corrected pair to the golden dataset

**Continuous improvement loop**
Every human correction becomes a new regression test case. The next Spark prompt experiment automatically includes it, creating a flywheel: production error → human correction → dataset update → re-evaluation → prompt refinement → deploy.
- [Docs: Human review](https://www.braintrust.dev/docs/annotate/human-review)
- [Docs: Datasets](https://www.braintrust.dev/docs/evaluate/datasets)

**Taxonomy-driven quality analysis**
Spark accuracy depends heavily on customer data taxonomy quality. Braintrust traces tagged with customer vertical and schema complexity reveal which types of project schemas cause the most Spark errors — informing both Spark improvements and customer-facing taxonomy guidance.

---

## Scenario 6: Complement to Mixpanel's "User as Eval" Philosophy

Mixpanel published ["The ultimate AI eval is your user"](https://mixpanel.com/blog/ai-evals-product-analytics/) — arguing that behavioral signals (retention, engagement, feature adoption) are the best long-term metrics for AI quality.

Braintrust complements this perfectly:

| Layer | Tool | What it measures | When |
|---|---|---|---|
| **Pre-production** | Braintrust | Is the Spark query correct before users see it? | Before deploy |
| **Post-production** | Mixpanel | Do users trust and re-engage with Spark results? | After deploy |

Braintrust catches the errors that Mixpanel's behavioral analytics can only detect indirectly. A user who gets a wrong chart from Spark won't file a bug — they'll just stop using Spark. By the time Mixpanel's retention metrics show the drop, the damage is done. Braintrust's pre-production eval layer catches these errors before any user sees them.

The combined workflow:
1. **Braintrust** evaluates query correctness offline → gates deployment
2. **Braintrust** monitors production quality online → detects drift
3. **Mixpanel** measures Spark engagement, retention, feature adoption → validates real-world impact
4. Low-engagement patterns in Mixpanel trigger investigation in Braintrust traces → find the root cause
5. Corrections flow back into Braintrust golden datasets → next eval cycle improves

---

## Data Portability

| Requirement | Braintrust Capability |
|---|---|
| **Full API coverage** | 100% API parity with the UI. Every operation — prompts, datasets, experiments, logs, scoring rules — is available via REST API and SDKs. |
| **Export capabilities** | Datasets, prompt histories, and evaluation logs export via API in JSON. Automated BTQL exports to S3 in JSONL or Parquet format. Ad-hoc queries via `bt sql`. [Docs: API reference](https://www.braintrust.dev/docs/reference/api) |
| **MCP support** | Braintrust provides an MCP server for agentic integration — relevant as Mixpanel also exposes an MCP server for external LLM clients. [Docs: MCP](https://www.braintrust.dev/docs/guides/mcp) |
