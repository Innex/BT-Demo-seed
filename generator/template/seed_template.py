#!/usr/bin/env python3
"""
{{DOCSTRING}}

Usage:
    # Full seed (prompts + scorers + dataset + traces)
    python {{FILENAME}} --project "{{PROJECT_NAME}}" --count 500

    # Only create prompts, scorers, and dataset (no traces)
    python {{FILENAME}} --project "{{PROJECT_NAME}}" --skip-traces

    # Only create traces (assumes prompts/scorers already exist)
    python {{FILENAME}} --project "{{PROJECT_NAME}}" --traces-only --count 500
"""

import asyncio
import argparse
import random
from dataclasses import dataclass

import braintrust
from braintrust import init_dataset


# ---------------------------------------------------------------------------
# Domain data — GENERATED for {{COMPANY_NAME}}
# ---------------------------------------------------------------------------

# AI feature modes (equivalent of report types / capability categories)
FEATURE_MODES = {{FEATURE_MODES}}

FEATURE_MODE_WEIGHTS = {{FEATURE_MODE_WEIGHTS}}

# Domain entities (events, objects, or items the AI operates on)
ENTITIES = {{ENTITIES}}

# Entity properties (attributes with possible values)
ENTITY_PROPERTIES = {{ENTITY_PROPERTIES}}

# Query complexity levels
QUERY_COMPLEXITIES = ["simple", "moderate", "complex"]

# Customer verticals / usage segments
CUSTOMER_VERTICALS = {{CUSTOMER_VERTICALS}}

PROMPT_VERSIONS = ["{{PROMPT_VERSION_A}}", "{{PROMPT_VERSION_B}}"]

MODELS = ["gpt-5-mini", "gpt-5-nano", "claude-sonnet-4-6"]


def _get_api_model(model: str) -> str:
    """Map model names to OpenAI-compatible API models for content generation."""
    if model.startswith("claude"):
        return "gpt-5-mini"
    return model

# User natural language queries organized by feature mode
USER_QUERIES = {{USER_QUERIES}}

# Sample output structures by feature mode
SAMPLE_OUTPUTS = {{SAMPLE_OUTPUTS}}

# AI responses — style A ({{STYLE_A_NAME}})
RESPONSES_STYLE_A = {{RESPONSES_STYLE_A}}

# AI responses — style B ({{STYLE_B_NAME}})
RESPONSES_STYLE_B = {{RESPONSES_STYLE_B}}

INSIGHT_SNIPPETS = {{INSIGHT_SNIPPETS}}

DETAIL_SNIPPETS = {{DETAIL_SNIPPETS}}

# Intentionally problematic responses (for HITL demo)
PROBLEMATIC_RESPONSES = {{PROBLEMATIC_RESPONSES}}

# Schema contexts per vertical / customer segment
SCHEMA_CONTEXTS = {{SCHEMA_CONTEXTS}}


# ---------------------------------------------------------------------------
# Engine code (reusable across all customer demos)
# ---------------------------------------------------------------------------

@dataclass
class TraceConfig:
    customer_id: str
    vertical: str
    feature_mode: str
    query_complexity: str
    prompt_version: str
    model: str
    quality_tier: str  # "good", "needs_review", "flagged"
    user_query: str
    schema_context: dict


def generate_trace_config(idx: int) -> TraceConfig:
    feature_mode = random.choices(FEATURE_MODES, weights=FEATURE_MODE_WEIGHTS, k=1)[0]
    schema = random.choice(SCHEMA_CONTEXTS)

    complexity_roll = random.random()
    if complexity_roll < 0.50:
        complexity = "simple"
    elif complexity_roll < 0.85:
        complexity = "moderate"
    else:
        complexity = "complex"

    quality_roll = random.random()
    if quality_roll < 0.07:
        quality_tier = "flagged"
    elif quality_roll < 0.25:
        quality_tier = "needs_review"
    else:
        quality_tier = "good"

    return TraceConfig(
        customer_id=f"CUST-{random.randint(10000, 99999)}",
        vertical=schema["vertical"],
        feature_mode=feature_mode,
        query_complexity=complexity,
        prompt_version=random.choice(PROMPT_VERSIONS),
        model=random.choice(MODELS),
        quality_tier=quality_tier,
        user_query=random.choice(USER_QUERIES[feature_mode]),
        schema_context=schema,
    )


def generate_response(config: TraceConfig) -> str:
    if config.quality_tier == "flagged":
        return random.choice(PROBLEMATIC_RESPONSES)

    entity = random.choice(config.schema_context["entities"])
    prop = random.choice(config.schema_context["properties"])
    aggregation = random.choice(["unique users", "total events", "event count per user"])
    time_range = random.choice(["7 days", "30 days", "90 days", "this quarter"])
    details = random.choice(DETAIL_SNIPPETS).format(
        entity=entity, aggregation=aggregation, breakdown=prop, time_range=time_range,
    )
    insight = random.choice(INSIGHT_SNIPPETS)

    if config.prompt_version == PROMPT_VERSIONS[0]:
        template = random.choice(RESPONSES_STYLE_A)
    else:
        template = random.choice(RESPONSES_STYLE_B)

    response = template.format(
        feature_mode=config.feature_mode,
        details=details,
        time_range=time_range,
        breakdown_note=f"broken down by {prop}",
        aggregation=aggregation,
        breakdown=prop,
        insight=insight,
    )

    if config.quality_tier == "needs_review":
        issues = [
            " Note: I selected the default aggregation but a different one might be more appropriate for this question.",
            " I wasn't 100% sure about the date range — you might want to adjust it.",
            " The breakdown includes some null values which might skew the results.",
            " I chose the closest matching entity but the data might be tracked differently.",
        ]
        response += random.choice(issues)

    return response


def generate_output(config: TraceConfig) -> dict:
    if config.feature_mode in SAMPLE_OUTPUTS:
        feature_mode = config.feature_mode
    elif SAMPLE_OUTPUTS:
        feature_mode = next(iter(SAMPLE_OUTPUTS))
    else:
        return {"output_type": config.feature_mode, "time_range": "30d"}
    base = dict(SAMPLE_OUTPUTS[feature_mode])
    base["time_range"] = random.choice(["7d", "30d", "90d"])

    schema = config.schema_context
    entity = random.choice(schema["entities"])
    prop = random.choice(schema["properties"])

    # Add entity/property references to the output
    base["primary_entity"] = entity
    if random.random() > 0.5:
        base["breakdowns"] = [{"property": prop}]

    return base


def generate_scores(config: TraceConfig) -> dict:
    # Feature mode difficulty penalties (some modes are harder for AI)
    mode_penalties = {mode: -0.02 * i for i, mode in enumerate(FEATURE_MODES)}
    mode_penalty = mode_penalties.get(config.feature_mode, 0.0)

    complexity_penalty = {
        "simple": 0.0,
        "moderate": -0.05,
        "complex": -0.12,
        "multi_turn": -0.07,
    }.get(config.query_complexity, 0.0)

    if config.quality_tier == "good":
        accuracy = random.uniform(0.80, 0.98) + mode_penalty + complexity_penalty
        grounding = random.uniform(0.85, 1.0) + mode_penalty * 0.5
    elif config.quality_tier == "needs_review":
        accuracy = random.uniform(0.40, 0.75) + mode_penalty + complexity_penalty
        grounding = random.uniform(0.55, 0.84) + mode_penalty
    else:  # flagged
        accuracy = random.uniform(0.05, 0.40)
        grounding = random.uniform(0.10, 0.50)

    accuracy = max(0.0, min(1.0, accuracy))
    grounding = max(0.0, min(1.0, grounding))
    quality = accuracy * 0.5 + grounding * 0.5
    quality = max(0.0, min(1.0, quality + random.uniform(-0.05, 0.05)))

    # Thread coherence: based on trace structure completeness
    if config.quality_tier == "good":
        thread_coherence = random.uniform(0.85, 1.0)
    elif config.quality_tier == "needs_review":
        thread_coherence = random.uniform(0.50, 0.80)
    else:
        thread_coherence = random.uniform(0.25, 0.50)
    thread_coherence = max(0.0, min(1.0, thread_coherence))

    # Use the scorer slugs from SCORERS config
    scorer_slugs = [s["slug"] for s in SCORERS]
    if len(scorer_slugs) >= 3:
        return {
            scorer_slugs[0]: round(accuracy, 3),
            scorer_slugs[1]: round(grounding, 3),
            scorer_slugs[2]: round(quality, 3),
            "thread_coherence": round(thread_coherence, 3),
        }
    return {
        "accuracy": round(accuracy, 3),
        "grounding": round(grounding, 3),
        "quality": round(quality, 3),
        "thread_coherence": round(thread_coherence, 3),
    }


def generate_latency_metrics(config: TraceConfig) -> dict:
    model_latencies = {
        "gpt-5-mini": (0.6, 2.0),
        "gpt-5-nano": (0.2, 0.9),
        "claude-sonnet-4-6": (0.5, 1.8),
    }
    model_costs = {
        "gpt-5-mini": (0.003, 0.012),
        "gpt-5-nano": (0.001, 0.004),
        "claude-sonnet-4-6": (0.003, 0.015),
    }

    base_min, base_max = model_latencies[config.model]
    input_cost_per_1k, output_cost_per_1k = model_costs[config.model]

    prompt_tokens = random.randint(600, 2000)
    completion_tokens = random.randint(100, 500)

    llm_latency = round(random.uniform(base_min, base_max), 3)
    context_latency = round(random.uniform(0.02, 0.15), 3)
    validation_latency = round(random.uniform(0.05, 0.2), 3)
    total_latency = round(llm_latency + context_latency + validation_latency + random.uniform(0.02, 0.08), 3)

    cost = round(
        (prompt_tokens / 1000) * input_cost_per_1k + (completion_tokens / 1000) * output_cost_per_1k,
        6,
    )

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "llm_latency_s": llm_latency,
        "context_latency_s": context_latency,
        "validation_latency_s": validation_latency,
        "total_latency_s": total_latency,
        "cost_usd": cost,
    }


# ---------------------------------------------------------------------------
# Prompts, scorers, facets, datasets — GENERATED for {{COMPANY_NAME}}
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = {{SYSTEM_PROMPT_BASE}}

STYLE_A_SUFFIX = {{STYLE_A_SUFFIX}}

STYLE_B_SUFFIX = {{STYLE_B_SUFFIX}}

# Product identity
AI_PRODUCT_NAME = "{{AI_PRODUCT_NAME}}"
STYLE_A_LABEL = "{{STYLE_A_NAME}}"
STYLE_B_LABEL = "{{STYLE_B_NAME}}"

# Scorer definitions (name, slug, description, prompt content, choice_scores)
SCORERS = {{SCORERS}}

# Facet definitions (name, slug, description, prompt content)
FACETS = {{FACETS}}

# Golden dataset rows
GOLDEN_DATASET_ROWS = {{GOLDEN_DATASET_ROWS}}

# Prior conversation snippets for context injection
PRIOR_CONVERSATION_SNIPPETS = {{PRIOR_CONVERSATION_SNIPPETS}}

# Multi-turn conversation templates
MULTI_TURN_CONVERSATIONS = {{MULTI_TURN_CONVERSATIONS}}


# ---------------------------------------------------------------------------
# Braintrust API helpers (reusable)
# ---------------------------------------------------------------------------

def _get_project_id(project_name: str) -> str:
    logger = braintrust.init_logger(project=project_name)
    project_id = logger.project.id
    logger.flush()
    return project_id


def _api_upsert_function(api_url: str, api_key: str, project_id: str, payload: dict) -> bool:
    import requests

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload["project_id"] = project_id

    resp = requests.post(f"{api_url}/v1/function", json=payload, headers=headers)
    if resp.status_code == 409:
        list_resp = requests.get(
            f"{api_url}/v1/function",
            params={"project_id": project_id, "slug": payload.get("slug", ""), "function_type": payload.get("function_type", "")},
            headers=headers,
        )
        if list_resp.ok and list_resp.json().get("objects"):
            existing = list_resp.json()["objects"][0]
            resp = requests.patch(f"{api_url}/v1/function/{existing['id']}", json=payload, headers=headers)

    if not resp.ok:
        print(f"  Warning: {payload.get('name')}: {resp.status_code} {resp.text[:200]}")
        return False
    return True


def create_prompts(project_name: str):
    import os

    print("\n--- Creating prompts ---")
    api_url = os.environ.get("BRAINTRUST_API_URL", "https://api.braintrust.dev")
    api_key = os.environ.get("BRAINTRUST_API_KEY", "")
    if not api_key:
        print("  ERROR: BRAINTRUST_API_KEY is not set. Cannot create prompts.")
        return
    project_id = _get_project_id(project_name)

    slug_a = PROMPT_VERSIONS[0].lower().replace(".", "-").replace(" ", "-")
    slug_b = PROMPT_VERSIONS[1].lower().replace(".", "-").replace(" ", "-")

    prompts = [
        {
            "name": f"{AI_PRODUCT_NAME} ({STYLE_A_LABEL})",
            "slug": slug_a,
            "description": f"{STYLE_A_LABEL} style for {AI_PRODUCT_NAME}.",
            "function_data": {"type": "prompt"},
            "prompt_data": {
                "prompt": {
                    "type": "chat",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT_BASE + STYLE_A_SUFFIX},
                        {"role": "user", "content": "{{input}}"},
                    ],
                },
                "options": {
                    "model": "gpt-5-mini",
                    "params": {"temperature": 0.3, "top_p": 0.9},
                },
            },
            "metadata": {
                "version": PROMPT_VERSIONS[0],
                "owner": "ai-team",
                "review_status": "approved",
            },
        },
        {
            "name": f"{AI_PRODUCT_NAME} ({STYLE_B_LABEL})",
            "slug": slug_b,
            "description": f"{STYLE_B_LABEL} style for {AI_PRODUCT_NAME}.",
            "function_data": {"type": "prompt"},
            "prompt_data": {
                "prompt": {
                    "type": "chat",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT_BASE + STYLE_B_SUFFIX},
                        {"role": "user", "content": "{{input}}"},
                    ],
                },
                "options": {
                    "model": "gpt-5-mini",
                    "params": {"temperature": 0.6, "top_p": 0.95},
                },
            },
            "metadata": {
                "version": PROMPT_VERSIONS[1],
                "owner": "ai-team",
                "review_status": "approved",
            },
        },
    ]

    for p in prompts:
        if _api_upsert_function(api_url, api_key, project_id, p):
            print(f"  Created prompt: {p['slug']}")
    print("  Prompts published successfully")


def create_scorers(project_name: str):
    import os
    import requests

    print("\n--- Creating scorers ---")
    api_url = os.environ.get("BRAINTRUST_API_URL", "https://api.braintrust.dev")
    api_key = os.environ.get("BRAINTRUST_API_KEY", "")
    if not api_key:
        print("  ERROR: BRAINTRUST_API_KEY is not set. Cannot create scorers.")
        return
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    project_id = _get_project_id(project_name)

    # --- 1. LLM prompt scorers (from SCORERS config) ---
    llm_scorers = []
    for s in SCORERS:
        llm_scorers.append({
            "name": s["name"],
            "slug": s["slug"],
            "description": s["description"],
            "function_type": "scorer",
            "function_data": {"type": "prompt"},
            "prompt_data": {
                "prompt": {
                    "type": "chat",
                    "messages": [{"role": "system", "content": s["prompt"]}],
                },
                "options": {"model": "gpt-5-mini", "params": {"temperature": 0.2}},
                "parser": {
                    "type": "llm_classifier",
                    "use_cot": True,
                    "choice_scores": s["choice_scores"],
                },
            },
            "metadata": s.get("metadata", {"category": "quality", "owner": "ai-team"}),
        })

    # --- 2. Code scorer: Grounding check (deterministic, no LLM needed) ---
    grounding_code_scorer = {
        "name": f"{AI_PRODUCT_NAME} grounding",
        "slug": "grounding-check",
        "description": (
            f"Code scorer that deterministically checks whether {AI_PRODUCT_NAME} responses "
            "only reference entities and properties that exist in the context schema. "
            "Detects hallucinated references without requiring an LLM call."
        ),
        "function_type": "scorer",
        "function_data": {
            "type": "code",
            "data": {
                "type": "inline",
                "runtime_context": {"runtime": "node", "version": "20"},
                "code": """
async function handler({ output, metadata }) {
  const entities = (metadata?.schema_entities || "").split(",").map(e => e.trim().toLowerCase()).filter(Boolean);
  const properties = (metadata?.schema_properties || "").split(",").map(p => p.trim().toLowerCase()).filter(Boolean);

  if (!output || !entities.length) {
    return { name: "Grounding check", score: 1, metadata: { reason: "No output or schema to validate" } };
  }

  const text = (typeof output === "string" ? output : JSON.stringify(output)).toLowerCase();
  let violations = [];

  // Check for property references not in schema
  const propPattern = /(?:broken? down by|breakdown|filter(?:ed)? by|grouped? by|property[: ]+)["']?([a-z_]+)["']?/gi;
  let propMatch;
  while ((propMatch = propPattern.exec(text)) !== null) {
    const prop = propMatch[1].toLowerCase();
    if (!properties.includes(prop) && !["none", "all", "any", "the", "and", "this"].includes(prop)) {
      violations.push("Property not in schema: " + propMatch[1]);
    }
  }

  const uniqueViolations = [...new Set(violations)];
  const score = uniqueViolations.length === 0 ? 1.0 : uniqueViolations.length <= 1 ? 0.6 : 0.0;

  return {
    name: "Grounding check",
    score,
    metadata: {
      violations: uniqueViolations,
      violation_count: uniqueViolations.length,
      entity_count: entities.length,
      property_count: properties.length,
    },
  };
}
""",
            },
        },
        "metadata": {
            "category": "safety",
            "scorer_type": "code",
            "owner": "ai-team",
            "deployment_gate": True,
        },
    }

    # --- 3. Code scorer: Thread coherence (trace-level, uses spans) ---
    thread_coherence_code = {
        "name": "Thread coherence",
        "slug": "thread-coherence",
        "description": (
            "Trace-level code scorer that analyzes the full conversation thread across all spans. "
            "Checks that multi-turn conversations maintain context, that validation steps "
            "agree with output, and that the overall trace is internally consistent."
        ),
        "function_type": "scorer",
        "function_data": {
            "type": "code",
            "data": {
                "type": "inline",
                "runtime_context": {"runtime": "node", "version": "20"},
                "code": """
async function handler({ trace, output, metadata }) {
  const issues = [];
  let spanCount = 0;
  let hasContext = false;
  let hasValidation = false;
  let validationPassed = null;
  let llmSpanCount = 0;

  if (trace) {
    const spans = await trace.getSpans();
    spanCount = spans.length;

    for (const span of spans) {
      const name = (span.name || "").toLowerCase();

      if (name.includes("schema") || name.includes("context") || name.includes("lookup")) {
        hasContext = true;
        if (span.output && span.output.entities && span.output.entities.length === 0) {
          issues.push("Context lookup returned empty entities");
        }
      }

      if (name.includes("validation") || name.includes("verify")) {
        hasValidation = true;
        if (span.output) {
          validationPassed = span.output.valid === true;
          if (!validationPassed && span.output.issues?.length > 0) {
            issues.push("Validation flagged: " + span.output.issues.join(", "));
          }
        }
      }

      if (span.span_attributes?.type === "llm" || name.includes("openai") || name.includes("chat")) {
        llmSpanCount++;
      }
    }
  }

  if (spanCount < 2) {
    issues.push("Trace has fewer than 2 spans");
  }
  if (!hasContext) {
    issues.push("Missing context/schema lookup span");
  }
  if (!hasValidation) {
    issues.push("Missing validation span");
  }

  if (validationPassed === false && output && !String(output).toLowerCase().includes("error")) {
    issues.push("Validation failed but response does not indicate an error");
  }

  const isMultiTurn = metadata?.is_multi_turn === true;
  const turnCount = metadata?.turn_count || 0;
  if (isMultiTurn && llmSpanCount > 0 && llmSpanCount < Math.floor(turnCount / 2)) {
    issues.push("Multi-turn trace has " + turnCount + " turns but only " + llmSpanCount + " LLM calls");
  }

  const score = issues.length === 0 ? 1.0 : issues.length <= 1 ? 0.75 : issues.length <= 2 ? 0.5 : 0.25;

  return {
    name: "Thread coherence",
    score,
    metadata: {
      issues,
      issue_count: issues.length,
      span_count: spanCount,
      has_context: hasContext,
      has_validation: hasValidation,
      validation_passed: validationPassed,
      llm_span_count: llmSpanCount,
      is_multi_turn: isMultiTurn,
    },
  };
}
""",
            },
        },
        "metadata": {
            "category": "trace_quality",
            "scorer_type": "code",
            "scope": "trace",
            "owner": "ai-team",
        },
    }

    # Create all scorers
    scorer_ids = []
    all_scorers = llm_scorers + [grounding_code_scorer, thread_coherence_code]

    for s in all_scorers:
        ft = s.get("function_data", {}).get("type", "prompt")
        if _api_upsert_function(api_url, api_key, project_id, s):
            print(f"  Created scorer: {s['slug']} ({ft})")
            list_resp = requests.get(
                f"{api_url}/v1/function",
                params={"project_id": project_id, "slug": s["slug"], "function_type": "scorer"},
                headers=headers,
            )
            if list_resp.ok and list_resp.json().get("objects"):
                scorer_ids.append(list_resp.json()["objects"][0]["id"])

    print("  Scorers published successfully")

    # Online scoring rules:
    # 1. Span-level for LLM + code grounding scorers
    # 2. Trace-level for thread coherence (uses idle_seconds for thread completion)
    span_scorer_ids = scorer_ids[:-1]  # all except thread-coherence
    trace_scorer_ids = scorer_ids[-1:]  # thread-coherence

    if span_scorer_ids:
        span_rule_payload = {
            "project_id": project_id,
            "name": f"{AI_PRODUCT_NAME} quality",
            "score_type": "online",
            "description": f"Runs LLM and code scorers on {AI_PRODUCT_NAME} spans",
            "config": {
                "online": {
                    "sampling_rate": 1.0,
                    "scorers": [{"type": "function", "id": sid} for sid in span_scorer_ids],
                    "apply_to_root_span": True,
                    "scope": {"type": "trace"},
                },
            },
        }
        resp = requests.post(f"{api_url}/v1/project_score", json=span_rule_payload, headers=headers)
        if resp.ok:
            print(f"  Created online scoring rule: {AI_PRODUCT_NAME} quality (100% sampling)")
        elif resp.status_code == 409:
            print("  Online scoring rule already exists (skipping)")
        else:
            print(f"  Warning: scoring rule: {resp.status_code} {resp.text[:200]}")

    if trace_scorer_ids:
        trace_rule_payload = {
            "project_id": project_id,
            "name": f"{AI_PRODUCT_NAME} thread coherence",
            "score_type": "online",
            "description": "Trace-level scorer: waits for full thread, checks cross-span consistency and multi-turn coherence",
            "config": {
                "online": {
                    "sampling_rate": 1.0,
                    "scorers": [{"type": "function", "id": sid} for sid in trace_scorer_ids],
                    "scope": {"type": "trace", "idle_seconds": 10},
                },
            },
        }
        resp = requests.post(f"{api_url}/v1/project_score", json=trace_rule_payload, headers=headers)
        if resp.ok:
            print(f"  Created online scoring rule: {AI_PRODUCT_NAME} thread coherence (trace-level, idle_seconds=10)")
        elif resp.status_code == 409:
            print("  Online scoring rule already exists (skipping)")
        else:
            print(f"  Warning: scoring rule: {resp.status_code} {resp.text[:200]}")


def create_facets(project_name: str):
    import os
    import requests

    print("\n--- Creating custom facets ---")

    api_url = os.environ.get("BRAINTRUST_API_URL", "https://api.braintrust.dev")
    api_key = os.environ.get("BRAINTRUST_API_KEY", "")
    if not api_key:
        print("  ERROR: BRAINTRUST_API_KEY is not set. Cannot create facets.")
        return
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    project_id = _get_project_id(project_name)

    facet_payloads = []
    for f in FACETS:
        facet_payloads.append({
            "name": f["name"],
            "slug": f["slug"],
            "description": f["description"],
            "function_type": "facet",
            "function_data": {
                "type": "facet",
                "prompt": f["prompt"],
                "model": "gpt-5-nano",
                **({"no_match_pattern": f["no_match_pattern"]} if "no_match_pattern" in f else {}),
            },
        })

    for facet in facet_payloads:
        if _api_upsert_function(api_url, api_key, project_id, facet):
            print(f"  Created facet: {facet['name']} ({facet['slug']})")

    facet_ids = []
    for facet in facet_payloads:
        list_resp = requests.get(
            f"{api_url}/v1/function",
            params={"project_id": project_id, "slug": facet["slug"]},
            headers=headers,
        )
        if list_resp.ok and list_resp.json().get("objects"):
            facet_ids.append(list_resp.json()["objects"][0]["id"])

    print("\n--- Creating topic automation ---")
    facet_refs = [
        {"type": "global", "name": "Sentiment", "function_type": "facet"},
        {"type": "global", "name": "Task", "function_type": "facet"},
        {"type": "global", "name": "Issues", "function_type": "facet"},
    ] + [{"type": "function", "id": fid} for fid in facet_ids]

    # Use the last custom facet for the topic map
    topic_map_functions = [
        {"function": {"type": "global", "name": "Sentiment", "function_type": "facet"}},
        {"function": {"type": "global", "name": "Task", "function_type": "facet"}},
        {"function": {"type": "global", "name": "Issues", "function_type": "facet"}},
    ]
    if facet_ids:
        topic_map_functions.append({"function": {"type": "function", "id": facet_ids[-1]}})

    automation_payload = {
        "project_automation_name": f"{AI_PRODUCT_NAME} topics",
        "description": f"Auto-clusters {AI_PRODUCT_NAME} traces by topic",
        "project_id": project_id,
        "config": {
            "event_type": "topic",
            "sampling_rate": 1.0,
            "facet_functions": facet_refs,
            "topic_map_functions": topic_map_functions,
            "scope": {"type": "trace", "idle_seconds": 30},
        },
    }
    resp = requests.post(
        f"{api_url}/api/project_automation/register",
        json=automation_payload,
        headers=headers,
    )
    if resp.ok:
        print(f"  Created topic automation: {AI_PRODUCT_NAME} topics")
    elif resp.status_code == 409:
        print("  Topic automation already exists (skipping)")
    else:
        print(f"  Warning: topic automation: {resp.status_code} {resp.text[:200]}")


def create_golden_dataset(project_name: str):
    print("\n--- Creating golden dataset ---")
    dataset = init_dataset(project=project_name, name=f"{AI_PRODUCT_NAME} golden set")

    for row in GOLDEN_DATASET_ROWS:
        dataset.insert(
            input=row["input"],
            expected=row["expected"],
            metadata=row["metadata"],
            tags=row["tags"],
        )

    dataset.flush()
    print(f"  Inserted {len(GOLDEN_DATASET_ROWS)} golden test cases")


_generated_traces: list[dict] = []


def create_datasets_from_traces(project_name: str):
    if not _generated_traces:
        print("\n--- Skipping dataset-from-traces (no traces generated yet) ---")
        return

    print(f"\n--- Building datasets from {len(_generated_traces)} generated traces ---")

    prod_ds = init_dataset(project=project_name, name=f"Production {AI_PRODUCT_NAME} samples")
    good_traces = [t for t in _generated_traces if t["quality_tier"] == "good"]
    sampled_good = []
    modes_seen = set()
    for t in good_traces:
        if t["feature_mode"] not in modes_seen or len(sampled_good) < 30:
            sampled_good.append(t)
            modes_seen.add(t["feature_mode"])
        if len(sampled_good) >= 50:
            break

    for t in sampled_good:
        prod_ds.insert(
            input=t["user_query"],
            expected=t["ai_response"],
            metadata={
                "feature_mode": t["feature_mode"],
                "vertical": t["vertical"],
                "prompt_version": t["prompt_version"],
                "model": t["model"],
                "customer_id": t["customer_id"],
                "source": "production_logs",
            },
            tags=[t["feature_mode"], t["vertical"], "production"],
        )
    prod_ds.flush()
    print(f"  'Production {AI_PRODUCT_NAME} samples': {len(sampled_good)} traces")

    review_ds = init_dataset(project=project_name, name="Flagged for review")
    flagged_traces = [t for t in _generated_traces if t["quality_tier"] in ("flagged", "needs_review")]
    for t in flagged_traces[:40]:
        tags = [t["feature_mode"], t["vertical"], t["quality_tier"]]
        if t["quality_tier"] == "flagged":
            tags.append("error")
        review_ds.insert(
            input=t["user_query"],
            expected=t["ai_response"],
            metadata={
                "feature_mode": t["feature_mode"],
                "vertical": t["vertical"],
                "prompt_version": t["prompt_version"],
                "quality_tier": t["quality_tier"],
                "customer_id": t["customer_id"],
                "source": "production_logs_flagged",
                "review_status": "pending",
            },
            tags=tags,
        )
    review_ds.flush()
    print(f"  'Flagged for review': {min(40, len(flagged_traces))} traces")

    playground_ds = init_dataset(project=project_name, name="Playground demo scenarios")
    scenario_count = 0

    for t in _generated_traces:
        if t["quality_tier"] == "flagged":
            playground_ds.insert(
                input=t["user_query"],
                expected="This output was flagged for errors. Review and correct, then add to golden dataset.",
                metadata={
                    **{k: t[k] for k in ("feature_mode", "vertical", "prompt_version", "model", "customer_id")},
                    "scenario": "flagged",
                    "quality_tier": t["quality_tier"],
                },
                tags=["playground", "flagged", "hitl_demo"],
            )
            scenario_count += 1
            break

    for version in PROMPT_VERSIONS:
        for t in _generated_traces:
            if t["prompt_version"] == version and t["quality_tier"] == "good" and t["feature_mode"] == FEATURE_MODES[0]:
                playground_ds.insert(
                    input=t["user_query"],
                    expected=t["ai_response"],
                    metadata={
                        **{k: t[k] for k in ("feature_mode", "vertical", "prompt_version", "model", "customer_id")},
                        "scenario": "prompt_comparison",
                    },
                    tags=["playground", "prompt_comparison", version],
                )
                scenario_count += 1
                break

    for model_name in MODELS:
        for t in _generated_traces:
            if t["model"] == model_name and t["quality_tier"] == "good":
                playground_ds.insert(
                    input=t["user_query"],
                    expected=t["ai_response"],
                    metadata={
                        **{k: t[k] for k in ("feature_mode", "vertical", "prompt_version", "model", "customer_id")},
                        "scenario": "model_comparison",
                    },
                    tags=["playground", "model_comparison", model_name],
                )
                scenario_count += 1
                break

    playground_ds.flush()
    print(f"  'Playground demo scenarios': {scenario_count} scenarios")


def build_system_prompt(config: TraceConfig) -> str:
    if config.prompt_version == PROMPT_VERSIONS[0]:
        return SYSTEM_PROMPT_BASE + STYLE_A_SUFFIX
    else:
        return SYSTEM_PROMPT_BASE + STYLE_B_SUFFIX


def log_trace(logger, config: TraceConfig, trace_idx: int, oai_client):
    perf = generate_latency_metrics(config)
    system_prompt = build_system_prompt(config)
    schema = config.schema_context

    has_history = random.random() < 0.35
    prior_turns = random.choice(PRIOR_CONVERSATION_SNIPPETS) if has_history else []

    schema_context_str = (
        f"Entities: {', '.join(schema['entities'])}\n"
        f"Properties: {', '.join(schema['properties'])}"
    )

    messages = [{"role": "system", "content": system_prompt + f"\n\nContext:\n{schema_context_str}"}]
    messages.extend(prior_turns)
    messages.append({"role": "user", "content": config.user_query})

    with logger.start_span(name="{{ROOT_SPAN_NAME}}") as root_span:
        # Step 1: Context retrieval
        with root_span.start_span(
            name="{{CONTEXT_SPAN_NAME}}",
            span_attributes={"type": "tool"},
        ) as ctx_span:
            ctx_span.log(
                input={"customer_id": config.customer_id, "query": config.user_query},
                output={"entities": schema["entities"], "properties": schema["properties"]},
                metadata={"vertical": config.vertical},
                metrics={"latency": perf["context_latency_s"]},
            )

        # Step 2: LLM call via wrap_openai
        response = oai_client.chat.completions.create(
            model=_get_api_model(config.model),
            messages=messages,
        )
        ai_response = response.choices[0].message.content

        # Step 3: Output validation
        output_struct = generate_output(config)
        validation_passed = config.quality_tier != "flagged"
        validation_issues = []
        if not validation_passed:
            validation_issues = random.sample(
                ["entity_not_found", "property_not_found", "wrong_aggregation", "invalid_output", "empty_result"],
                k=random.randint(1, 2),
            )

        with root_span.start_span(
            name="{{VALIDATION_SPAN_NAME}}",
            span_attributes={"type": "tool"},
        ) as val_span:
            val_span.log(
                input={"generated_output": output_struct},
                output={
                    "valid": validation_passed,
                    "issues": validation_issues,
                    "confidence": round(
                        random.uniform(0.85, 0.99) if validation_passed else random.uniform(0.3, 0.6), 3,
                    ),
                },
                metadata={"validator_version": "v2.1"},
                metrics={"latency": perf["validation_latency_s"]},
            )

        conversation = prior_turns + [
            {"role": "user", "content": config.user_query},
            {"role": "assistant", "content": ai_response},
        ]

        scores = generate_scores(config)

        root_span.log(
            input=config.user_query,
            output=ai_response,
            scores=scores,
            metadata={
                "customer_id": config.customer_id,
                "vertical": config.vertical,
                "feature_mode": config.feature_mode,
                "query_complexity": config.query_complexity,
                "prompt_version": config.prompt_version,
                "model": config.model,
                "quality_tier": config.quality_tier,
                "trace_index": trace_idx,
                "turn_count": len(conversation),
                "has_prior_context": has_history,
                "schema_entities": ", ".join(schema["entities"]),
                "schema_properties": ", ".join(schema["properties"]),
                "generated_output": str(output_struct),
                "validation_result": "passed" if validation_passed else "failed",
            },
        )

    _generated_traces.append({
        "customer_id": config.customer_id,
        "vertical": config.vertical,
        "feature_mode": config.feature_mode,
        "query_complexity": config.query_complexity,
        "prompt_version": config.prompt_version,
        "model": config.model,
        "quality_tier": config.quality_tier,
        "user_query": config.user_query,
        "ai_response": ai_response,
        "conversation": conversation,
        "trace_idx": trace_idx,
    })


def log_multi_turn_trace(logger, conversation: dict, trace_idx: int, oai_client):
    feature_mode = conversation["feature_mode"]
    vertical = conversation["vertical"]
    template_turns = conversation["turns"]
    prompt_version = random.choice(PROMPT_VERSIONS)
    model = random.choice(MODELS)
    schema = next((s for s in SCHEMA_CONTEXTS if s["vertical"] == vertical), SCHEMA_CONTEXTS[0])

    quality_tier = random.choices(["good", "needs_review", "flagged"], weights=[0.80, 0.15, 0.05], k=1)[0]
    config = TraceConfig(
        customer_id=f"CUST-{random.randint(10000, 99999)}",
        vertical=vertical,
        feature_mode=feature_mode,
        query_complexity="moderate",
        prompt_version=prompt_version,
        model=model,
        quality_tier=quality_tier,
        user_query=template_turns[0]["content"],
        schema_context=schema,
    )
    system_prompt = build_system_prompt(config)
    schema_context_str = f"Entities: {', '.join(schema['entities'])}\nProperties: {', '.join(schema['properties'])}"
    validation_passed = quality_tier != "flagged"

    with logger.start_span(name="{{ROOT_SPAN_NAME}}") as root_span:
        with root_span.start_span(
            name="{{CONTEXT_SPAN_NAME}}",
            span_attributes={"type": "tool"},
        ) as ctx_span:
            ctx_span.log(
                input={"customer_id": config.customer_id},
                output={"entities": schema["entities"], "properties": schema["properties"]},
                metadata={"vertical": config.vertical},
                metrics={"latency": round(random.uniform(0.02, 0.15), 3)},
            )

        accumulated_messages = [{"role": "system", "content": system_prompt + f"\n\nContext:\n{schema_context_str}"}]
        actual_turns = []
        user_turns = [t for t in template_turns if t["role"] == "user"]

        for user_turn in user_turns:
            accumulated_messages.append(user_turn)
            actual_turns.append(user_turn)

            response = oai_client.chat.completions.create(
                model=_get_api_model(config.model),
                messages=accumulated_messages,
            )
            assistant_content = response.choices[0].message.content
            assistant_turn = {"role": "assistant", "content": assistant_content}
            accumulated_messages.append(assistant_turn)
            actual_turns.append(assistant_turn)

        with root_span.start_span(
            name="{{VALIDATION_SPAN_NAME}}",
            span_attributes={"type": "tool"},
        ) as val_span:
            val_span.log(
                input={"generated_output": "multi-turn refinement"},
                output={
                    "valid": validation_passed,
                    "issues": [] if validation_passed else ["context_drift"],
                    "confidence": round(random.uniform(0.80, 0.99) if validation_passed else random.uniform(0.3, 0.6), 3),
                },
                metadata={"validator_version": "v2.1"},
                metrics={"latency": round(random.uniform(0.05, 0.2), 3)},
            )

        last_response = actual_turns[-1]["content"] if actual_turns else ""
        first_user_msg = user_turns[0]["content"] if user_turns else ""

        scores = generate_scores(config)

        root_span.log(
            input=first_user_msg,
            output=last_response,
            scores=scores,
            metadata={
                "customer_id": config.customer_id,
                "vertical": vertical,
                "feature_mode": feature_mode,
                "query_complexity": "multi_turn",
                "prompt_version": prompt_version,
                "model": config.model,
                "quality_tier": quality_tier,
                "trace_index": trace_idx,
                "turn_count": len(actual_turns),
                "is_multi_turn": True,
                "schema_entities": ", ".join(schema["entities"]),
                "schema_properties": ", ".join(schema["properties"]),
                "validation_result": "passed" if validation_passed else "failed",
            },
        )

    _generated_traces.append({
        "customer_id": config.customer_id,
        "vertical": vertical,
        "feature_mode": feature_mode,
        "query_complexity": "multi_turn",
        "prompt_version": prompt_version,
        "model": model,
        "quality_tier": quality_tier,
        "user_query": user_turns[0]["content"],
        "ai_response": last_response,
        "conversation": actual_turns,
        "trace_idx": trace_idx,
        "is_multi_turn": True,
    })


def _run_experiment_row(experiment, row, prompt_ver, model_name, accuracy_base, quality_base, oai_client):
    scorer_slugs = [s["slug"] for s in SCORERS]
    row_input = row.get("input", "")
    row_expected = row.get("expected", "")
    row_metadata = row.get("metadata", {})
    feature_mode = row_metadata.get("feature_mode", FEATURE_MODES[0])
    vertical = row_metadata.get("vertical", CUSTOMER_VERTICALS[0] if CUSTOMER_VERTICALS else "general")

    schema = next((s for s in SCHEMA_CONTEXTS if s["vertical"] == vertical), SCHEMA_CONTEXTS[0])
    config = TraceConfig(
        customer_id=f"CUST-{random.randint(10000, 99999)}",
        vertical=vertical,
        feature_mode=feature_mode,
        query_complexity=row_metadata.get("query_complexity", "moderate"),
        prompt_version=prompt_ver,
        model=model_name,
        quality_tier="good",
        user_query=row_input if isinstance(row_input, str) else str(row_input),
        schema_context=schema,
    )
    system_prompt = build_system_prompt(config)
    ctx_str = f"Entities: {row_metadata.get('schema_entities', ', '.join(schema['entities']))}\nProperties: {row_metadata.get('schema_properties', ', '.join(schema['properties']))}"

    messages = [{"role": "system", "content": system_prompt + f"\n\nContext:\n{ctx_str}"}]
    conv_history = row_metadata.get("conversation_history", [])
    messages.extend(conv_history)
    user_msg = row_input if isinstance(row_input, str) else str(row_input)
    messages.append({"role": "user", "content": user_msg})

    with experiment.start_span(name="eval") as span:
        with span.start_span(name="{{CONTEXT_SPAN_NAME}}", span_attributes={"type": "tool"}) as ctx_span:
            ctx_span.log(
                input={"customer_id": config.customer_id},
                output={"entities": schema["entities"], "properties": schema["properties"]},
                metadata={"vertical": vertical},
            )

        response = oai_client.chat.completions.create(
            model=_get_api_model(model_name), messages=messages,
        )
        ai_response = response.choices[0].message.content

        with span.start_span(name="{{VALIDATION_SPAN_NAME}}", span_attributes={"type": "tool"}) as val_span:
            val_span.log(
                input={"response": ai_response},
                output={"valid": True, "issues": [], "confidence": round(random.uniform(0.90, 0.99), 3)},
            )

        complexity = row_metadata.get("query_complexity", "moderate")
        complexity_penalty = {"simple": 0.0, "moderate": -0.08, "complex": -0.18}.get(complexity, -0.05)
        ab, qb = accuracy_base, quality_base

        expected_behavior = row_metadata.get("expected_behavior", "")
        if expected_behavior in ("schema_mismatch", "missing_property", "taxonomy_gap", "aggregation_ambiguity"):
            ab -= 0.15
        if feature_mode == "ambiguous":
            ab -= 0.10

        scores = {}
        if len(scorer_slugs) >= 3:
            scores[scorer_slugs[0]] = round(max(0.0, min(1.0, ab + complexity_penalty + random.uniform(-0.08, 0.08))), 3)
            scores[scorer_slugs[1]] = round(max(0.0, min(1.0, 0.90 + complexity_penalty * 0.5 + random.uniform(-0.06, 0.06))), 3)
            scores[scorer_slugs[2]] = round(max(0.0, min(1.0, qb + complexity_penalty * 0.5 + random.uniform(-0.08, 0.08))), 3)

        span.log(
            input=user_msg,
            output=ai_response,
            expected=row_expected,
            scores=scores,
            metadata={
                "prompt_version": prompt_ver,
                "model": model_name,
                "feature_mode": feature_mode,
                "vertical": vertical,
                "query_complexity": row_metadata.get("query_complexity", "moderate"),
                "schema_entities": row_metadata.get("schema_entities", ""),
                "schema_properties": row_metadata.get("schema_properties", ""),
                "generated_output": str(generate_output(config)),
                "validation_result": "passed",
            },
            tags=row.get("tags", []),
        )


def run_experiments(project_name: str, oai_client=None, parallelism: int = 20):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not oai_client:
        print("\n--- Skipping experiments (OPENAI_API_KEY required) ---")
        return

    print("\n--- Running experiments (prompt A vs B comparison) ---")

    dataset_rows = GOLDEN_DATASET_ROWS
    if not dataset_rows:
        print("  Warning: Golden dataset is empty.")
        return

    test_rows = dataset_rows[:50]

    def run_single_experiment(exp_name, test_rows, prompt_ver, model_name, accuracy_base, quality_base):
        print(f"  Running experiment: {exp_name} ({len(test_rows)} test cases, parallelism={parallelism})")
        experiment = braintrust.init(project_name, exp_name)

        completed = 0
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = {
                executor.submit(
                    _run_experiment_row, experiment, row, prompt_ver, model_name,
                    accuracy_base, quality_base, oai_client,
                ): i
                for i, row in enumerate(test_rows)
            }
            for future in as_completed(futures):
                try:
                    future.result()
                    completed += 1
                    if completed % 10 == 0:
                        print(f"    Progress: {completed}/{len(test_rows)}")
                except Exception as e:
                    print(f"    Error on row {futures[future]}: {e}")

        summary = experiment.summarize()
        print(f"    {summary}")

    for prompt_ver, exp_name in [
        (PROMPT_VERSIONS[0], f"Prompt A: {STYLE_A_LABEL}"),
        (PROMPT_VERSIONS[1], f"Prompt B: {STYLE_B_LABEL}"),
    ]:
        accuracy_base = 0.88 if prompt_ver == PROMPT_VERSIONS[0] else 0.78
        quality_base = 0.72 if prompt_ver == PROMPT_VERSIONS[0] else 0.85
        run_single_experiment(exp_name, test_rows, prompt_ver, "gpt-5-mini", accuracy_base, quality_base)

    print("  Prompt A/B experiments complete")

    print("\n  Running model comparison experiments...")
    for model_name in ["gpt-5-mini", "gpt-5-nano"]:
        accuracy_base = 0.88 if model_name == "gpt-5-mini" else 0.82
        run_single_experiment(
            f"Model comparison: {model_name}", test_rows, PROMPT_VERSIONS[0],
            model_name, accuracy_base, 0.78,
        )

    print("  All experiments complete — compare in Braintrust UI")


async def generate_traces(project: str, count: int, parallelism: int, oai_client):
    print(f"Initializing Braintrust logger for project '{project}'...")
    logger = braintrust.init_logger(project=project)
    print(f"Project ID: {logger.project.id}")

    semaphore = asyncio.Semaphore(parallelism)
    completed = 0
    errors = 0

    mode_counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
    prompt_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    vertical_counts: dict[str, int] = {}

    async def generate_one(idx: int):
        nonlocal completed, errors
        async with semaphore:
            config = generate_trace_config(idx)

            mode_counts[config.feature_mode] = mode_counts.get(config.feature_mode, 0) + 1
            quality_counts[config.quality_tier] = quality_counts.get(config.quality_tier, 0) + 1
            prompt_counts[config.prompt_version] = prompt_counts.get(config.prompt_version, 0) + 1
            model_counts[config.model] = model_counts.get(config.model, 0) + 1
            vertical_counts[config.vertical] = vertical_counts.get(config.vertical, 0) + 1

            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, log_trace, logger, config, idx, oai_client)
                completed += 1
                if completed % 50 == 0 or completed == count:
                    print(f"  Progress: {completed}/{count} traces logged")
            except Exception as e:
                errors += 1
                print(f"  Error on trace {idx}: {e}")

    multi_turn_count = min(len(MULTI_TURN_CONVERSATIONS) * 10, count // 10)
    single_turn_count = count - multi_turn_count

    print(f"\nGenerating {count} traces...")
    print(f"  Single-turn: {single_turn_count}, Multi-turn: {multi_turn_count}")
    print(f"  Parallelism: {parallelism}\n")

    await asyncio.gather(*[generate_one(i) for i in range(single_turn_count)])

    print(f"\n  Logging {multi_turn_count} multi-turn conversations...")
    loop = asyncio.get_event_loop()
    for i in range(multi_turn_count):
        conv = MULTI_TURN_CONVERSATIONS[i % len(MULTI_TURN_CONVERSATIONS)]
        try:
            await loop.run_in_executor(
                None, log_multi_turn_trace, logger, conv, single_turn_count + i, oai_client
            )
            completed += 1
        except Exception as e:
            errors += 1
            print(f"  Error on multi-turn trace {i}: {e}")

    print("\nFlushing logs...")
    logger.flush()

    print(f"\n{'='*60}")
    print(f"SEED COMPLETE: {completed} traces logged ({errors} errors)")
    print(f"{'='*60}")

    print("\nQuality distribution:")
    for tier, cnt in sorted(quality_counts.items(), key=lambda x: -x[1]):
        print(f"  {tier}: {cnt} ({cnt/count*100:.1f}%)")

    print("\nPrompt version distribution:")
    for ver, cnt in sorted(prompt_counts.items(), key=lambda x: -x[1]):
        print(f"  {ver}: {cnt} ({cnt/count*100:.1f}%)")

    print("\nModel distribution:")
    for m, cnt in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"  {m}: {cnt} ({cnt/count*100:.1f}%)")

    print("\nFeature mode distribution:")
    for mode, cnt in sorted(mode_counts.items(), key=lambda x: -x[1]):
        print(f"  {mode}: {cnt} ({cnt/count*100:.1f}%)")

    print("\nVertical distribution:")
    for vert, cnt in sorted(vertical_counts.items(), key=lambda x: -x[1]):
        print(f"  {vert}: {cnt} ({cnt/count*100:.1f}%)")

    print(f"\nView traces at: https://www.braintrust.dev")


def main():
    import os

    parser = argparse.ArgumentParser(description=f"Seed {AI_PRODUCT_NAME} demo into Braintrust")
    parser.add_argument("--project", type=str, default="{{PROJECT_NAME}}", help="Project name")
    parser.add_argument("--count", type=int, default=500, help="Number of traces to generate")
    parser.add_argument("--parallelism", type=int, default=50, help="Concurrent trace generation limit")
    parser.add_argument("--skip-traces", action="store_true", help="Only create prompts, scorers, dataset, and experiments")
    parser.add_argument("--traces-only", action="store_true", help="Only create traces")
    parser.add_argument("--skip-experiments", action="store_true", help="Skip A/B experiments")

    args = parser.parse_args()

    oai_client = None
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        import openai
        from braintrust import wrap_openai
        oai_client = wrap_openai(openai.OpenAI(api_key=openai_key))
        print("OpenAI client initialized (wrap_openai enabled)")

    if not args.traces_only:
        create_prompts(args.project)
        create_scorers(args.project)
        create_facets(args.project)
        create_golden_dataset(args.project)

        if not args.skip_experiments:
            if not oai_client:
                print("\n--- Skipping experiments (OPENAI_API_KEY required) ---")
            else:
                run_experiments(args.project, oai_client)

    if not args.skip_traces:
        if not oai_client:
            print("\nERROR: OPENAI_API_KEY is required for trace generation.")
            print("  export OPENAI_API_KEY='sk-...'")
            return
        asyncio.run(generate_traces(args.project, args.count, args.parallelism, oai_client))
        create_datasets_from_traces(args.project)


if __name__ == "__main__":
    main()
