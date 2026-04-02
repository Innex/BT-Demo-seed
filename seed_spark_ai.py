#!/usr/bin/env python3
"""
Mixpanel Spark AI Demo - Seeds a Braintrust project with:
  - 2 versioned prompts (structured vs conversational NL-to-query styles)
  - 3 scorers (query_correctness, schema_adherence, response_quality)
  - 1 golden dataset (~40 curated test cases for regression testing)
  - 500 realistic Spark AI traces

Spark AI is Mixpanel's NL-to-analytics agent: users ask questions in plain English
and Spark translates them into Mixpanel reports (Insights, Funnels, Retention, Flows).
The model receives the customer's event taxonomy (event names + property names, NOT the
underlying data) and generates query parameters that Mixpanel executes.

Demo scenarios:
  1. Query correctness evaluation: Does Spark select the right events, aggregation,
     report type, time range, breakdowns, and filters for the user's question?
  2. Schema adherence: Does Spark only reference events/properties that exist in
     the customer's project? (Taxonomy quality directly impacts accuracy)
  3. Model migration: Regression testing when upgrading from GPT-3.5 → GPT-4+
  4. Production observability: End-to-end traces (schema lookup → LLM → validation)
  5. Multi-turn follow-ups: "Now break that down by platform" — iterative refinement
  6. Human-in-the-loop: Surfacing bad queries for review, building golden datasets

Usage:
    # Full seed (prompts + scorers + dataset + traces)
    python seed_spark_ai.py --project "Mixpanel Spark AI" --count 500

    # Only create prompts, scorers, and dataset (no traces)
    python seed_spark_ai.py --project "Mixpanel Spark AI" --skip-traces

    # Only create traces (assumes prompts/scorers already exist)
    python seed_spark_ai.py --project "Mixpanel Spark AI" --traces-only --count 500
"""

import asyncio
import argparse
import random
from dataclasses import dataclass

import braintrust
from braintrust import init_dataset


# ---------------------------------------------------------------------------
# Domain data: Mixpanel analytics queries
# ---------------------------------------------------------------------------

# Mixpanel report types that Spark AI can generate
REPORT_TYPES = [
    "insights",    # Trends, aggregations, breakdowns
    "funnels",     # Conversion funnels
    "retention",   # Cohort retention analysis
    "flows",       # User flow / path analysis
]

REPORT_WEIGHTS = [0.40, 0.25, 0.20, 0.15]

# Mixpanel event taxonomy (realistic product analytics events)
EVENTS = [
    {"name": "Sign Up", "category": "acquisition"},
    {"name": "Login", "category": "engagement"},
    {"name": "Page Viewed", "category": "engagement"},
    {"name": "Feature Used", "category": "engagement"},
    {"name": "Item Purchased", "category": "monetization"},
    {"name": "Subscription Started", "category": "monetization"},
    {"name": "Subscription Cancelled", "category": "churn"},
    {"name": "Plan Upgraded", "category": "monetization"},
    {"name": "Plan Downgraded", "category": "churn"},
    {"name": "Support Ticket Created", "category": "support"},
    {"name": "Onboarding Step Completed", "category": "activation"},
    {"name": "Invite Sent", "category": "virality"},
    {"name": "File Uploaded", "category": "engagement"},
    {"name": "Search Performed", "category": "engagement"},
    {"name": "Notification Clicked", "category": "engagement"},
    {"name": "Experiment Viewed", "category": "experimentation"},
    {"name": "Checkout Started", "category": "monetization"},
    {"name": "Checkout Completed", "category": "monetization"},
    {"name": "Cart Abandoned", "category": "churn"},
    {"name": "Report Generated", "category": "engagement"},
    {"name": "Dashboard Viewed", "category": "engagement"},
    {"name": "API Call Made", "category": "engagement"},
    {"name": "Error Encountered", "category": "quality"},
    {"name": "Session Started", "category": "engagement"},
    {"name": "Session Ended", "category": "engagement"},
]

# Event properties for schema context
EVENT_PROPERTIES = [
    {"name": "plan_type", "type": "string", "values": ["free", "growth", "enterprise"]},
    {"name": "platform", "type": "string", "values": ["web", "ios", "android"]},
    {"name": "country", "type": "string", "values": ["US", "UK", "DE", "FR", "JP", "BR", "IN"]},
    {"name": "referral_source", "type": "string", "values": ["organic", "paid", "referral", "direct"]},
    {"name": "feature_name", "type": "string", "values": ["funnels", "insights", "retention", "flows", "cohorts", "spark_ai"]},
    {"name": "amount_usd", "type": "number", "values": None},
    {"name": "duration_seconds", "type": "number", "values": None},
    {"name": "error_type", "type": "string", "values": ["timeout", "auth_failure", "rate_limit", "validation", "server_error"]},
    {"name": "browser", "type": "string", "values": ["Chrome", "Firefox", "Safari", "Edge"]},
    {"name": "onboarding_step", "type": "string", "values": ["create_project", "install_sdk", "send_event", "create_report", "invite_team"]},
]

# Query complexity levels
QUERY_COMPLEXITIES = ["simple", "moderate", "complex"]

# Customer verticals (for schema variation)
CUSTOMER_VERTICALS = [
    "b2b_saas",
    "ecommerce",
    "fintech",
    "media_entertainment",
    "healthcare",
    "marketplace",
]

PROMPT_VERSIONS = ["v3.1-structured", "v3.2-conversational"]

MODELS = ["gpt-5-mini", "gpt-5-nano", "claude-sonnet-4-6"]


def _get_api_model(model: str) -> str:
    """Map model names to OpenAI-compatible API models for content generation.

    Claude models are simulated via OpenAI for demo content generation.
    """
    if model.startswith("claude"):
        return "gpt-5-mini"
    return model

# User natural language queries organized by report type
USER_QUERIES = {
    "insights": [
        "Show me daily active users over the last 30 days",
        "What's our signup conversion rate this month?",
        "How many users used the Spark AI feature last week?",
        "Break down page views by country for Q1",
        "What's the average session duration by platform?",
        "Show me the trend of API calls over the past 90 days",
        "How many error events happened yesterday broken down by error type?",
        "What percentage of users are on the enterprise plan?",
        "Compare feature usage between free and paid users",
        "Show me weekly report generation counts for the last 3 months",
        "What's the total revenue from item purchases this quarter?",
        "How does notification click rate vary by platform?",
        "Show me the distribution of session durations",
        "What are the top 10 most viewed pages?",
        "How many support tickets were created per week this year?",
    ],
    "funnels": [
        "Show me the conversion funnel from Sign Up to first Feature Used",
        "What's the drop-off rate between Checkout Started and Checkout Completed?",
        "Build a funnel from Onboarding Step 1 through Step 5",
        "Show conversion from Sign Up to Subscription Started by plan type",
        "What's the funnel from Page Viewed to Item Purchased for mobile users?",
        "Show me the onboarding completion funnel broken down by referral source",
        "What's the conversion rate from free trial to paid subscription?",
        "Build a funnel from Dashboard Viewed to Report Generated to File Uploaded",
        "Show the signup to first API call funnel for enterprise customers",
        "What percentage of users who start checkout actually complete it?",
    ],
    "retention": [
        "Show me 30-day retention for users who signed up last month",
        "What's the week-over-week retention by plan type?",
        "Compare retention between users who completed onboarding vs those who didn't",
        "Show me retention for users acquired through paid vs organic channels",
        "What's the 90-day retention curve for enterprise customers?",
        "How does retention differ between iOS and Android users?",
        "Show me monthly retention cohorts for the last 6 months",
        "What's the retention rate for users who used Spark AI in their first week?",
        "Compare retention between users who sent an invite vs those who didn't",
        "Show me the retention drop-off point for free plan users",
    ],
    "flows": [
        "What do users typically do after signing up?",
        "Show me the most common paths leading to Subscription Cancelled",
        "What's the typical user journey from first login to first purchase?",
        "Show me what users do before creating a support ticket",
        "What are the most common paths after viewing the pricing page?",
        "Show the user flow from onboarding completion to first report",
        "What do users do right before they churn?",
        "Show me the paths leading to Plan Upgraded",
        "What's the typical flow for users who eventually become enterprise customers?",
        "Show me what happens after users encounter an error",
    ],
}

# Spark AI generated query representations (Mixpanel query JSON)
SAMPLE_QUERY_OUTPUTS = {
    "insights": {
        "report_type": "insights",
        "time_range": "30d",
        "events": [{"event": "Session Started", "aggregation": "unique_users"}],
        "breakdowns": [],
        "filters": [],
    },
    "funnels": {
        "report_type": "funnels",
        "time_range": "30d",
        "steps": [
            {"event": "Sign Up"},
            {"event": "Feature Used"},
        ],
        "breakdowns": [],
        "filters": [],
    },
    "retention": {
        "report_type": "retention",
        "time_range": "90d",
        "start_event": {"event": "Sign Up"},
        "return_event": {"event": "Login"},
        "breakdowns": [],
    },
    "flows": {
        "report_type": "flows",
        "time_range": "30d",
        "starting_event": {"event": "Sign Up"},
        "depth": 5,
    },
}

# Spark AI responses — structured style (precise, query-focused)
SPARK_RESPONSES_STRUCTURED = [
    "I've created an {report_type} report for you. {details} "
    "The query covers the last {time_range} and {breakdown_note}. "
    "You can modify the time range or add additional breakdowns in the query builder.",

    "Here's your {report_type} report. {details} "
    "I've applied the filters you specified. "
    "Click on any data point to drill down further.",

    "Your {report_type} report is ready. {details} "
    "I've set the aggregation to {aggregation}. "
    "You can switch between line chart, bar chart, or table view.",

    "I've built this {report_type} report based on your question. {details} "
    "The data is broken down by {breakdown}. "
    "Hover over the chart to see exact values for each segment.",
]

# Spark AI responses — conversational style (explanatory, insight-driven)
SPARK_RESPONSES_CONVERSATIONAL = [
    "Great question! I've put together an {report_type} report that shows {details}. "
    "A few things stand out: {insight}. "
    "Want me to break this down further by any specific dimension?",

    "Here's what I found looking at your {report_type} data. {details} "
    "One interesting pattern: {insight}. "
    "Should I dig deeper into any of these trends?",

    "I've analyzed your data and created an {report_type} report. {details} "
    "The most notable trend is {insight}. "
    "Would you like me to compare this against a different time period?",

    "Looking at your {report_type} data, {details} "
    "What caught my attention is {insight}. "
    "I can also build a funnel or retention chart if you want to explore this further.",
]

INSIGHT_SNIPPETS = [
    "there's a 15% increase in the metric compared to the previous period",
    "mobile users show significantly different patterns than web users",
    "enterprise customers have 3x higher engagement on this metric",
    "there's a notable drop-off on weekends",
    "the trend has been accelerating over the past 2 weeks",
    "users from organic channels perform 20% better on this metric",
    "the free-to-paid conversion shows a strong correlation with this behavior",
]

DETAIL_SNIPPETS = [
    "showing {event} events aggregated by {aggregation} over the last {time_range}",
    "tracking the conversion between {event} and the subsequent action",
    "measuring {aggregation} of {event} broken down by {breakdown}",
    "analyzing the {time_range} trend for {event} across all segments",
]

# Queries that are intentionally problematic (for HITL demo)
PROBLEMATIC_QUERIES = [
    "The report shows revenue of $0 for the last quarter, which appears incorrect. The query may have selected the wrong event or property.",
    "I wasn't able to find an event matching 'user churn'. This might be tracked differently in your project. Did you mean 'Subscription Cancelled'?",
    "The funnel shows 0% conversion because the events are in the wrong order. The correct sequence should be Sign Up → Feature Used, not Feature Used → Sign Up.",
    "I've created the report but the breakdown by 'department' returned no results — this property doesn't exist in your project schema.",
    "The retention chart is empty because the return event 'Daily Login' doesn't exist. Your project tracks this as 'Login'.",
    "I generated a flows report but the starting event has very low volume (3 occurrences), which makes the paths unreliable.",
    "The date range you specified includes future dates, so the last few data points will show as zero.",
    "I selected 'total events' but you likely wanted 'unique users'. The chart shows inflated numbers because power users trigger this event multiple times.",
]

# Schema context that Spark AI receives (event names + properties from customer project)
SCHEMA_CONTEXTS = [
    {
        "vertical": "b2b_saas",
        "events": ["Sign Up", "Login", "Feature Used", "Plan Upgraded", "Subscription Started", "Subscription Cancelled", "Invite Sent", "Report Generated", "API Call Made", "Support Ticket Created"],
        "properties": ["plan_type", "platform", "country", "feature_name", "referral_source"],
    },
    {
        "vertical": "ecommerce",
        "events": ["Sign Up", "Page Viewed", "Search Performed", "Item Purchased", "Checkout Started", "Checkout Completed", "Cart Abandoned", "Notification Clicked"],
        "properties": ["platform", "country", "referral_source", "amount_usd", "browser"],
    },
    {
        "vertical": "fintech",
        "events": ["Sign Up", "Login", "Feature Used", "Item Purchased", "Error Encountered", "API Call Made", "Support Ticket Created", "Session Started"],
        "properties": ["plan_type", "platform", "country", "amount_usd", "error_type"],
    },
    {
        "vertical": "media_entertainment",
        "events": ["Sign Up", "Login", "Page Viewed", "Feature Used", "Session Started", "Session Ended", "Notification Clicked", "File Uploaded"],
        "properties": ["platform", "country", "browser", "duration_seconds", "referral_source"],
    },
]


@dataclass
class TraceConfig:
    customer_id: str
    vertical: str
    report_type: str
    query_complexity: str
    prompt_version: str
    model: str
    quality_tier: str  # "good", "needs_review", "flagged"
    user_query: str
    schema_context: dict


def generate_trace_config(idx: int) -> TraceConfig:
    report_type = random.choices(REPORT_TYPES, weights=REPORT_WEIGHTS, k=1)[0]
    schema = random.choice(SCHEMA_CONTEXTS)

    # Query complexity distribution
    complexity_roll = random.random()
    if complexity_roll < 0.50:
        complexity = "simple"
    elif complexity_roll < 0.85:
        complexity = "moderate"
    else:
        complexity = "complex"

    # Quality distribution: 75% good, 18% needs_review, 7% flagged
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
        report_type=report_type,
        query_complexity=complexity,
        prompt_version=random.choice(PROMPT_VERSIONS),
        model=random.choice(MODELS),
        quality_tier=quality_tier,
        user_query=random.choice(USER_QUERIES[report_type]),
        schema_context=schema,
    )


def generate_spark_response(config: TraceConfig) -> str:
    if config.quality_tier == "flagged":
        return random.choice(PROBLEMATIC_QUERIES)

    event = random.choice(config.schema_context["events"])
    prop = random.choice(config.schema_context["properties"])
    aggregation = random.choice(["unique users", "total events", "event count per user"])
    time_range = random.choice(["7 days", "30 days", "90 days", "this quarter"])
    details = random.choice(DETAIL_SNIPPETS).format(
        event=event, aggregation=aggregation, breakdown=prop, time_range=time_range,
    )
    insight = random.choice(INSIGHT_SNIPPETS)

    if config.prompt_version == "v3.1-structured":
        template = random.choice(SPARK_RESPONSES_STRUCTURED)
    else:
        template = random.choice(SPARK_RESPONSES_CONVERSATIONAL)

    response = template.format(
        report_type=config.report_type,
        details=details,
        time_range=time_range,
        breakdown_note=f"broken down by {prop}",
        aggregation=aggregation,
        breakdown=prop,
        insight=insight,
    )

    if config.quality_tier == "needs_review":
        issues = [
            " Note: I selected 'total events' but 'unique users' might be more appropriate for this question.",
            " I wasn't 100% sure about the date range — you might want to adjust it in the query builder.",
            " The breakdown includes some null values which might skew the results.",
            " I chose the closest matching event but your project might track this differently.",
        ]
        response += random.choice(issues)

    return response


def generate_query_output(config: TraceConfig) -> dict:
    report_type = config.report_type if config.report_type in SAMPLE_QUERY_OUTPUTS else "insights"
    base = dict(SAMPLE_QUERY_OUTPUTS[report_type])
    base["time_range"] = random.choice(["7d", "30d", "90d"])

    schema = config.schema_context
    event = random.choice(schema["events"])
    prop = random.choice(schema["properties"])

    if config.report_type == "insights":
        base["events"] = [{"event": event, "aggregation": random.choice(["unique", "total", "average"])}]
        if random.random() > 0.5:
            base["breakdowns"] = [{"property": prop}]
    elif config.report_type == "funnels":
        events = random.sample(schema["events"], min(random.randint(2, 4), len(schema["events"])))
        base["steps"] = [{"event": e} for e in events]
        if random.random() > 0.6:
            base["breakdowns"] = [{"property": prop}]
    elif config.report_type == "retention":
        start, ret = random.sample(schema["events"], 2)
        base["start_event"] = {"event": start}
        base["return_event"] = {"event": ret}
    elif config.report_type == "flows":
        base["starting_event"] = {"event": event}
        base["depth"] = random.randint(3, 7)

    return base


def generate_scores(config: TraceConfig) -> dict:
    # Report type difficulty: funnels and flows are harder for NL-to-query
    report_penalty = {
        "insights": 0.0,
        "funnels": -0.06,
        "retention": -0.08,
        "flows": -0.10,
    }.get(config.report_type, 0.0)

    # Query complexity penalty
    complexity_penalty = {
        "simple": 0.0,
        "moderate": -0.05,
        "complex": -0.12,
        "multi_turn": -0.07,
    }.get(config.query_complexity, 0.0)

    if config.quality_tier == "good":
        query_correctness = random.uniform(0.80, 0.98) + report_penalty + complexity_penalty
        schema_adherence = random.uniform(0.85, 1.0) + report_penalty * 0.5
    elif config.quality_tier == "needs_review":
        query_correctness = random.uniform(0.40, 0.75) + report_penalty + complexity_penalty
        schema_adherence = random.uniform(0.55, 0.84) + report_penalty
    else:  # flagged
        query_correctness = random.uniform(0.05, 0.40)
        schema_adherence = random.uniform(0.10, 0.50)

    query_correctness = max(0.0, min(1.0, query_correctness))
    schema_adherence = max(0.0, min(1.0, schema_adherence))
    response_quality = query_correctness * 0.5 + schema_adherence * 0.5
    response_quality = max(0.0, min(1.0, response_quality + random.uniform(-0.05, 0.05)))

    return {
        "query_correctness": round(query_correctness, 3),
        "schema_adherence": round(schema_adherence, 3),
        "response_quality": round(response_quality, 3),
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
    schema_lookup_latency = round(random.uniform(0.02, 0.15), 3)
    query_validation_latency = round(random.uniform(0.05, 0.2), 3)
    total_latency = round(llm_latency + schema_lookup_latency + query_validation_latency + random.uniform(0.02, 0.08), 3)

    cost = round(
        (prompt_tokens / 1000) * input_cost_per_1k + (completion_tokens / 1000) * output_cost_per_1k,
        6,
    )

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "llm_latency_s": llm_latency,
        "schema_lookup_latency_s": schema_lookup_latency,
        "query_validation_latency_s": query_validation_latency,
        "total_latency_s": total_latency,
        "cost_usd": cost,
    }


# ---------------------------------------------------------------------------
# Prompt, scorer, and dataset creation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = (
    "You are Spark AI, Mixpanel's natural language analytics assistant. "
    "You translate user questions into Mixpanel reports (Insights, Funnels, Retention, Flows). "
    "You have access to the user's project schema: event names, event properties, and user properties.\n\n"
    "Rules:\n"
    "- Only reference events and properties that exist in the project schema\n"
    "- Select the correct report type for the question\n"
    "- Choose the right aggregation (unique users vs total events vs average)\n"
    "- Apply appropriate time ranges and filters\n"
    "- If the user's question is ambiguous, pick the most likely interpretation and note your assumption\n"
    "- If an event or property doesn't exist in the schema, say so clearly\n\n"
    "Project schema:\n"
    "Events: {{metadata.schema_events}}\n"
    "Properties: {{metadata.schema_properties}}"
)

STRUCTURED_STYLE = (
    "\n\nStyle: Be precise and query-focused. Lead with the report you've created. "
    "Describe the exact configuration (events, aggregation, breakdowns, filters, time range). "
    "Suggest modifications the user can make in the query builder. Keep it concise."
)

CONVERSATIONAL_STYLE = (
    "\n\nStyle: Be explanatory and insight-driven. Create the report but also highlight "
    "interesting patterns or anomalies in the data. Ask follow-up questions to help the user "
    "explore further. Use natural, friendly language."
)


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

    prompts = [
        {
            "name": "Spark AI (structured)",
            "slug": "spark-ai-structured",
            "description": "Structured NL-to-query style. Precise, query-focused responses with exact configuration details.",
            "function_data": {"type": "prompt"},
            "prompt_data": {
                "prompt": {
                    "type": "chat",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT_BASE + STRUCTURED_STYLE},
                        {"role": "user", "content": "{{input}}"},
                    ],
                },
                "options": {
                    "model": "gpt-5-mini",
                    "params": {"temperature": 0.3, "top_p": 0.9},
                },
            },
            "metadata": {
                "version": "v3.1-structured",
                "owner": "spark-ai-team",
                "review_status": "approved",
            },
        },
        {
            "name": "Spark AI (conversational)",
            "slug": "spark-ai-conversational",
            "description": "Conversational NL-to-query style. Insight-driven responses that highlight patterns and suggest exploration.",
            "function_data": {"type": "prompt"},
            "prompt_data": {
                "prompt": {
                    "type": "chat",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT_BASE + CONVERSATIONAL_STYLE},
                        {"role": "user", "content": "{{input}}"},
                    ],
                },
                "options": {
                    "model": "gpt-5-mini",
                    "params": {"temperature": 0.6, "top_p": 0.95},
                },
            },
            "metadata": {
                "version": "v3.2-conversational",
                "owner": "spark-ai-team",
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

    llm_scorers = [
        {
            "name": "Query correctness",
            "slug": "query-correctness",
            "description": (
                "Evaluates whether Spark AI generated the correct Mixpanel query for the user's "
                "natural language question. Checks event selection, aggregation type, time range, "
                "breakdowns, filters, and report type."
            ),
            "function_type": "scorer",
            "function_data": {"type": "prompt"},
            "prompt_data": {
                "prompt": {
                    "type": "chat",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a Mixpanel analytics expert evaluating whether an AI assistant "
                                "correctly translated a natural language question into a Mixpanel query.\n\n"
                                "Project schema (available events and properties):\n"
                                "Events: {{metadata.schema_events}}\n"
                                "Properties: {{metadata.schema_properties}}\n\n"
                                "User's question: {{input}}\n"
                                "Generated query: {{metadata.generated_query}}\n"
                                "AI response: {{output}}\n\n"
                                "Evaluate the query correctness:\n"
                                "1. Did Spark select the correct report type (Insights/Funnels/Retention/Flows)?\n"
                                "2. Were the correct events selected from the schema?\n"
                                "3. Is the aggregation appropriate (unique users vs total events vs average)?\n"
                                "4. Is the time range reasonable for the question?\n"
                                "5. Are breakdowns and filters correctly applied?\n"
                                "6. If the question is ambiguous, did Spark make a reasonable interpretation?\n"
                                "7. Does the query actually answer the user's question?"
                            ),
                        },
                    ],
                },
                "options": {"model": "gpt-5-mini", "params": {"temperature": 0.2}},
                "parser": {
                    "type": "llm_classifier",
                    "use_cot": True,
                    "choice_scores": {"Correct": 1.0, "Mostly correct": 0.75, "Partially correct": 0.5, "Incorrect": 0.25, "Completely wrong": 0.0},
                },
            },
            "metadata": {
                "category": "accuracy",
                "owner": "spark-ai-team",
                "deployment_gate": True,
            },
        },
        {
            "name": "Schema adherence",
            "slug": "schema-adherence",
            "description": (
                "Evaluates whether Spark AI only referenced events and properties that actually "
                "exist in the customer's project schema. Detects hallucinated events, non-existent "
                "properties, and schema mismatches."
            ),
            "function_type": "scorer",
            "function_data": {"type": "prompt"},
            "prompt_data": {
                "prompt": {
                    "type": "chat",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a data quality reviewer checking whether an AI analytics assistant "
                                "hallucinated any events or properties that don't exist in the project schema.\n\n"
                                "Project schema:\n"
                                "Events: {{metadata.schema_events}}\n"
                                "Properties: {{metadata.schema_properties}}\n\n"
                                "User's question: {{input}}\n"
                                "Generated query: {{metadata.generated_query}}\n"
                                "AI response: {{output}}\n\n"
                                "Check for schema violations:\n"
                                "1. Does the query reference any events NOT in the schema?\n"
                                "2. Does the query use any properties NOT in the schema?\n"
                                "3. Are property values realistic for the property type?\n"
                                "4. If the user asked about something not in the schema, did Spark say so clearly?\n"
                                "5. Are the event-property combinations valid?"
                            ),
                        },
                    ],
                },
                "options": {"model": "gpt-5-mini", "params": {"temperature": 0.1}},
                "parser": {
                    "type": "llm_classifier",
                    "use_cot": True,
                    "choice_scores": {"Clean": 1.0, "Minor issue": 0.6, "Schema violation": 0.0},
                },
            },
            "metadata": {
                "category": "safety",
                "owner": "spark-ai-team",
                "deployment_gate": True,
            },
        },
        {
            "name": "Response quality",
            "slug": "response-quality",
            "description": (
                "Evaluates the overall quality of Spark AI's response including helpfulness, "
                "clarity, appropriate caveats, and actionability."
            ),
            "function_type": "scorer",
            "function_data": {"type": "prompt"},
            "prompt_data": {
                "prompt": {
                    "type": "chat",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a product quality reviewer evaluating an AI analytics assistant's response.\n\n"
                                "User's question: {{input}}\n"
                                "AI response: {{output}}\n"
                                "Report type: {{metadata.report_type}}\n"
                                "Query complexity: {{metadata.query_complexity}}\n\n"
                                "Evaluate the response quality:\n"
                                "1. **Helpfulness**: Does the response actually answer the user's question?\n"
                                "2. **Clarity**: Is the response easy to understand for a non-technical product manager?\n"
                                "3. **Actionability**: Can the user take action based on this response?\n"
                                "4. **Appropriate caveats**: If assumptions were made, are they stated?\n"
                                "5. **Completeness**: Does the response include enough detail without being overwhelming?"
                            ),
                        },
                    ],
                },
                "options": {"model": "gpt-5-mini", "params": {"temperature": 0.2}},
                "parser": {
                    "type": "llm_classifier",
                    "use_cot": True,
                    "choice_scores": {
                        "Excellent": 1.0,
                        "Good": 0.75,
                        "Adequate": 0.5,
                        "Poor": 0.25,
                        "Unusable": 0.0,
                    },
                },
            },
            "metadata": {
                "category": "quality",
                "owner": "spark-ai-team",
            },
        },
    ]

    scorer_ids = []
    for s in llm_scorers:
        if _api_upsert_function(api_url, api_key, project_id, s):
            print(f"  Created scorer: {s['slug']}")
            list_resp = requests.get(
                f"{api_url}/v1/function",
                params={"project_id": project_id, "slug": s["slug"], "function_type": "scorer"},
                headers=headers,
            )
            if list_resp.ok and list_resp.json().get("objects"):
                scorer_ids.append(list_resp.json()["objects"][0]["id"])

    print("  Scorers published successfully")

    if scorer_ids:
        score_rule_payload = {
            "project_id": project_id,
            "name": "Spark AI quality",
            "score_type": "online",
            "description": "Runs query correctness, schema adherence, and response quality scorers on all production Spark AI traces",
            "config": {
                "online": {
                    "sampling_rate": 1.0,
                    "scorers": [{"type": "function", "id": sid} for sid in scorer_ids],
                    "apply_to_root_span": True,
                    "scope": {"type": "trace"},
                },
            },
        }
        resp = requests.post(f"{api_url}/v1/project_score", json=score_rule_payload, headers=headers)
        if resp.ok:
            print("  Created online scoring rule: Spark AI quality (100% sampling, trace scope)")
        elif resp.status_code == 409:
            print("  Online scoring rule already exists (skipping)")
        else:
            print(f"  Warning: online scoring rule: {resp.status_code} {resp.text[:200]}")


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

    facets = [
        {
            "name": "Report type requested",
            "slug": "report-type-requested",
            "description": "Classifies which Mixpanel report type the user's NL query maps to",
            "function_type": "facet",
            "function_data": {
                "type": "facet",
                "prompt": (
                    "What type of Mixpanel report is the user asking for?\n\n"
                    "Respond with a single label:\n"
                    "- Insights — trends, counts, aggregations, breakdowns, comparisons\n"
                    "- Funnels — conversion paths, drop-off analysis, multi-step sequences\n"
                    "- Retention — cohort retention, return rates, engagement over time\n"
                    "- Flows — user paths, journey analysis, what happens before/after an event\n"
                    "- Ambiguous — could map to multiple report types\n\n"
                    'If the question is not about analytics, respond: "NONE"'
                ),
                "model": "gpt-5-nano",
                "no_match_pattern": "^NONE",
            },
        },
        {
            "name": "Query complexity",
            "slug": "query-complexity",
            "description": "Assesses how complex the user's analytics question is to translate into a Mixpanel query",
            "function_type": "facet",
            "function_data": {
                "type": "facet",
                "prompt": (
                    "Assess the complexity of translating this natural language question into a Mixpanel analytics query.\n\n"
                    "Consider:\n"
                    "- How many events/properties are involved?\n"
                    "- Does it require breakdowns, filters, or custom aggregations?\n"
                    "- Is the time range explicit or implicit?\n"
                    "- Does it need multiple report types combined?\n\n"
                    "Use these exact labels:\n"
                    "- Simple. [reason] — single event, straightforward aggregation, clear time range\n"
                    "- Moderate. [reason] — multiple events or properties, breakdowns, implicit assumptions\n"
                    "- Complex. [reason] — multi-step logic, ambiguous intent, requires schema knowledge\n\n"
                    "Always include a 1-sentence reason after the label."
                ),
                "model": "gpt-5-nano",
            },
        },
        {
            "name": "Failure mode",
            "slug": "failure-mode",
            "description": "Classifies the type of error when Spark AI generates an incorrect query",
            "function_type": "facet",
            "function_data": {
                "type": "facet",
                "prompt": (
                    "If the AI analytics assistant made an error in this interaction, classify the failure mode.\n\n"
                    "Use these exact labels:\n"
                    "- Wrong event. [reason] — selected an incorrect or non-existent event\n"
                    "- Wrong aggregation. [reason] — used total events instead of unique users, or vice versa\n"
                    "- Wrong report type. [reason] — built a funnel when insights was needed, etc.\n"
                    "- Schema hallucination. [reason] — referenced events/properties not in the schema\n"
                    "- Wrong time range. [reason] — incorrect or unreasonable date range\n"
                    "- Misunderstood intent. [reason] — interpreted the question incorrectly\n"
                    "- No error. The query appears correct.\n\n"
                    "Always include a 1-sentence reason after the label."
                ),
                "model": "gpt-5-nano",
            },
        },
        {
            "name": "Customer vertical",
            "slug": "customer-vertical",
            "description": "Identifies the customer's industry vertical from the analytics schema and query context",
            "function_type": "facet",
            "function_data": {
                "type": "facet",
                "prompt": (
                    "Based on the events, properties, and user question, what industry vertical does this customer belong to?\n\n"
                    "Use these exact labels:\n"
                    "- B2B SaaS. [reason]\n"
                    "- Ecommerce. [reason]\n"
                    "- Fintech. [reason]\n"
                    "- Media & Entertainment. [reason]\n"
                    "- Healthcare. [reason]\n"
                    "- Marketplace. [reason]\n"
                    "- Other. [reason]\n\n"
                    "Always include a 1-sentence reason after the label."
                ),
                "model": "gpt-5-nano",
            },
        },
    ]

    for facet in facets:
        if _api_upsert_function(api_url, api_key, project_id, facet):
            print(f"  Created facet: {facet['name']} ({facet['slug']})")

    facet_ids = []
    for facet in facets:
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

    failure_mode_id = None
    resp = requests.get(
        f"{api_url}/v1/function",
        params={"project_id": project_id, "slug": "failure-mode"},
        headers=headers,
    )
    if resp.ok and resp.json().get("objects"):
        failure_mode_id = resp.json()["objects"][0]["id"]

    topic_map_functions = [
        {"function": {"type": "global", "name": "Sentiment", "function_type": "facet"}},
        {"function": {"type": "global", "name": "Task", "function_type": "facet"}},
        {"function": {"type": "global", "name": "Issues", "function_type": "facet"}},
    ]
    if failure_mode_id:
        topic_map_functions.append({"function": {"type": "function", "id": failure_mode_id}})

    automation_payload = {
        "project_automation_name": "Spark AI query topics",
        "description": "Auto-clusters Spark AI traces by report type, complexity, failure mode, and customer vertical",
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
        print("  Created topic automation: Spark AI query topics")
        print("  Topic maps: Sentiment, Task, Issues, Failure mode")
    elif resp.status_code == 409:
        print("  Topic automation already exists (skipping)")
    else:
        print(f"  Warning: topic automation: {resp.status_code} {resp.text[:200]}")


# Golden dataset: curated test cases for regression testing
GOLDEN_DATASET_ROWS = [
    # --- Insights queries ---
    {
        "input": "Show me daily active users over the last 30 days",
        "expected": '{"report_type": "insights", "events": [{"event": "Session Started", "aggregation": "unique_users"}], "time_range": "30d", "interval": "day"}',
        "metadata": {"report_type": "insights", "query_complexity": "simple", "vertical": "b2b_saas", "schema_events": "Sign Up, Login, Feature Used, Session Started, Plan Upgraded", "schema_properties": "plan_type, platform, country"},
        "tags": ["insights", "simple", "dau"],
    },
    {
        "input": "What's our signup conversion rate this month?",
        "expected": '{"report_type": "insights", "events": [{"event": "Sign Up", "aggregation": "unique_users"}], "time_range": "this_month", "note": "Conversion rate requires defining what users are converting FROM. Clarify denominator (e.g., Page Viewed to Sign Up)."}',
        "metadata": {"report_type": "insights", "query_complexity": "moderate", "vertical": "b2b_saas", "schema_events": "Sign Up, Login, Page Viewed, Feature Used", "schema_properties": "referral_source, platform"},
        "tags": ["insights", "moderate", "conversion"],
    },
    {
        "input": "Break down page views by country for Q1",
        "expected": '{"report_type": "insights", "events": [{"event": "Page Viewed", "aggregation": "total"}], "time_range": "Q1", "breakdowns": [{"property": "country"}]}',
        "metadata": {"report_type": "insights", "query_complexity": "simple", "vertical": "ecommerce", "schema_events": "Page Viewed, Search Performed, Item Purchased, Checkout Started", "schema_properties": "platform, country, browser, amount_usd"},
        "tags": ["insights", "simple", "breakdown", "geo"],
    },
    {
        "input": "Compare feature usage between free and paid users",
        "expected": '{"report_type": "insights", "events": [{"event": "Feature Used", "aggregation": "unique_users"}], "time_range": "30d", "breakdowns": [{"property": "plan_type"}], "filters": [{"property": "plan_type", "values": ["free", "growth", "enterprise"]}]}',
        "metadata": {"report_type": "insights", "query_complexity": "moderate", "vertical": "b2b_saas", "schema_events": "Feature Used, Login, Plan Upgraded", "schema_properties": "plan_type, feature_name, platform"},
        "tags": ["insights", "moderate", "comparison", "segmentation"],
    },
    {
        "input": "How many error events happened yesterday broken down by error type?",
        "expected": '{"report_type": "insights", "events": [{"event": "Error Encountered", "aggregation": "total"}], "time_range": "1d", "breakdowns": [{"property": "error_type"}]}',
        "metadata": {"report_type": "insights", "query_complexity": "simple", "vertical": "fintech", "schema_events": "Error Encountered, API Call Made, Login, Feature Used", "schema_properties": "error_type, platform, country"},
        "tags": ["insights", "simple", "errors", "debugging"],
    },
    {
        "input": "What's the average session duration by platform?",
        "expected": '{"report_type": "insights", "events": [{"event": "Session Started", "aggregation": "average", "property": "duration_seconds"}], "time_range": "30d", "breakdowns": [{"property": "platform"}]}',
        "metadata": {"report_type": "insights", "query_complexity": "moderate", "vertical": "media_entertainment", "schema_events": "Session Started, Session Ended, Page Viewed, Feature Used", "schema_properties": "platform, duration_seconds, browser"},
        "tags": ["insights", "moderate", "property_aggregation"],
    },
    {
        "input": "What percentage of users are on the enterprise plan?",
        "expected": '{"report_type": "insights", "events": [{"event": "Login", "aggregation": "unique_users"}], "time_range": "30d", "breakdowns": [{"property": "plan_type"}], "note": "Show as percentage breakdown"}',
        "metadata": {"report_type": "insights", "query_complexity": "simple", "vertical": "b2b_saas", "schema_events": "Login, Feature Used, Plan Upgraded", "schema_properties": "plan_type, platform"},
        "tags": ["insights", "simple", "distribution"],
    },
    {
        "input": "Show me the total revenue from item purchases this quarter",
        "expected": '{"report_type": "insights", "events": [{"event": "Item Purchased", "aggregation": "sum", "property": "amount_usd"}], "time_range": "this_quarter"}',
        "metadata": {"report_type": "insights", "query_complexity": "moderate", "vertical": "ecommerce", "schema_events": "Item Purchased, Checkout Completed, Cart Abandoned", "schema_properties": "amount_usd, platform, country"},
        "tags": ["insights", "moderate", "revenue", "property_aggregation"],
    },
    # --- Funnel queries ---
    {
        "input": "Show me the conversion funnel from Sign Up to first Feature Used",
        "expected": '{"report_type": "funnels", "steps": [{"event": "Sign Up"}, {"event": "Feature Used"}], "time_range": "30d"}',
        "metadata": {"report_type": "funnels", "query_complexity": "simple", "vertical": "b2b_saas", "schema_events": "Sign Up, Login, Feature Used, Plan Upgraded", "schema_properties": "plan_type, platform"},
        "tags": ["funnels", "simple", "activation"],
    },
    {
        "input": "What's the drop-off rate between Checkout Started and Checkout Completed?",
        "expected": '{"report_type": "funnels", "steps": [{"event": "Checkout Started"}, {"event": "Checkout Completed"}], "time_range": "30d"}',
        "metadata": {"report_type": "funnels", "query_complexity": "simple", "vertical": "ecommerce", "schema_events": "Checkout Started, Checkout Completed, Cart Abandoned, Item Purchased", "schema_properties": "platform, amount_usd, country"},
        "tags": ["funnels", "simple", "checkout", "conversion"],
    },
    {
        "input": "Build a funnel from Onboarding Step 1 through Step 5",
        "expected": '{"report_type": "funnels", "steps": [{"event": "Onboarding Step Completed", "filter": {"property": "onboarding_step", "value": "create_project"}}, {"event": "Onboarding Step Completed", "filter": {"property": "onboarding_step", "value": "install_sdk"}}, {"event": "Onboarding Step Completed", "filter": {"property": "onboarding_step", "value": "send_event"}}, {"event": "Onboarding Step Completed", "filter": {"property": "onboarding_step", "value": "create_report"}}, {"event": "Onboarding Step Completed", "filter": {"property": "onboarding_step", "value": "invite_team"}}], "time_range": "30d"}',
        "metadata": {"report_type": "funnels", "query_complexity": "complex", "vertical": "b2b_saas", "schema_events": "Sign Up, Onboarding Step Completed, Feature Used, Invite Sent", "schema_properties": "onboarding_step, platform, referral_source"},
        "tags": ["funnels", "complex", "onboarding"],
    },
    {
        "input": "Show conversion from Sign Up to Subscription Started by plan type",
        "expected": '{"report_type": "funnels", "steps": [{"event": "Sign Up"}, {"event": "Subscription Started"}], "time_range": "90d", "breakdowns": [{"property": "plan_type"}]}',
        "metadata": {"report_type": "funnels", "query_complexity": "moderate", "vertical": "b2b_saas", "schema_events": "Sign Up, Subscription Started, Plan Upgraded, Feature Used", "schema_properties": "plan_type, referral_source, platform"},
        "tags": ["funnels", "moderate", "monetization", "breakdown"],
    },
    {
        "input": "What percentage of users who start checkout actually complete it?",
        "expected": '{"report_type": "funnels", "steps": [{"event": "Checkout Started"}, {"event": "Checkout Completed"}], "time_range": "30d", "note": "Show overall conversion rate percentage"}',
        "metadata": {"report_type": "funnels", "query_complexity": "simple", "vertical": "ecommerce", "schema_events": "Checkout Started, Checkout Completed, Cart Abandoned", "schema_properties": "platform, amount_usd"},
        "tags": ["funnels", "simple", "conversion_rate"],
    },
    # --- Retention queries ---
    {
        "input": "Show me 30-day retention for users who signed up last month",
        "expected": '{"report_type": "retention", "start_event": {"event": "Sign Up"}, "return_event": {"event": "Login"}, "time_range": "last_month", "retention_window": "30d"}',
        "metadata": {"report_type": "retention", "query_complexity": "simple", "vertical": "b2b_saas", "schema_events": "Sign Up, Login, Feature Used, Session Started", "schema_properties": "plan_type, platform"},
        "tags": ["retention", "simple", "cohort"],
    },
    {
        "input": "What's the week-over-week retention by plan type?",
        "expected": '{"report_type": "retention", "start_event": {"event": "Sign Up"}, "return_event": {"event": "Session Started"}, "time_range": "90d", "interval": "week", "breakdowns": [{"property": "plan_type"}]}',
        "metadata": {"report_type": "retention", "query_complexity": "moderate", "vertical": "b2b_saas", "schema_events": "Sign Up, Session Started, Feature Used, Login", "schema_properties": "plan_type, platform, country"},
        "tags": ["retention", "moderate", "breakdown", "weekly"],
    },
    {
        "input": "Compare retention between users who completed onboarding vs those who didn't",
        "expected": '{"report_type": "retention", "start_event": {"event": "Sign Up"}, "return_event": {"event": "Login"}, "time_range": "90d", "segments": [{"name": "Completed onboarding", "filter": {"event": "Onboarding Step Completed", "property": "onboarding_step", "value": "invite_team"}}, {"name": "Did not complete", "exclude_filter": {"event": "Onboarding Step Completed", "property": "onboarding_step", "value": "invite_team"}}]}',
        "metadata": {"report_type": "retention", "query_complexity": "complex", "vertical": "b2b_saas", "schema_events": "Sign Up, Login, Onboarding Step Completed, Feature Used", "schema_properties": "onboarding_step, plan_type, platform"},
        "tags": ["retention", "complex", "segmentation", "onboarding"],
    },
    {
        "input": "What's the retention rate for users who used Spark AI in their first week?",
        "expected": '{"report_type": "retention", "start_event": {"event": "Feature Used", "filter": {"property": "feature_name", "value": "spark_ai"}, "within": "7d_of_signup"}, "return_event": {"event": "Session Started"}, "time_range": "90d"}',
        "metadata": {"report_type": "retention", "query_complexity": "complex", "vertical": "b2b_saas", "schema_events": "Sign Up, Feature Used, Session Started, Login", "schema_properties": "feature_name, plan_type, platform"},
        "tags": ["retention", "complex", "feature_impact", "spark_ai"],
    },
    {
        "input": "Show me the retention drop-off point for free plan users",
        "expected": '{"report_type": "retention", "start_event": {"event": "Sign Up"}, "return_event": {"event": "Login"}, "time_range": "90d", "filters": [{"property": "plan_type", "value": "free"}]}',
        "metadata": {"report_type": "retention", "query_complexity": "moderate", "vertical": "b2b_saas", "schema_events": "Sign Up, Login, Session Started", "schema_properties": "plan_type, platform"},
        "tags": ["retention", "moderate", "churn", "free_tier"],
    },
    # --- Flows queries ---
    {
        "input": "What do users typically do after signing up?",
        "expected": '{"report_type": "flows", "starting_event": {"event": "Sign Up"}, "direction": "after", "depth": 5, "time_range": "30d"}',
        "metadata": {"report_type": "flows", "query_complexity": "simple", "vertical": "b2b_saas", "schema_events": "Sign Up, Login, Feature Used, Onboarding Step Completed, Invite Sent", "schema_properties": "platform, referral_source"},
        "tags": ["flows", "simple", "post_signup"],
    },
    {
        "input": "Show me the most common paths leading to Subscription Cancelled",
        "expected": '{"report_type": "flows", "ending_event": {"event": "Subscription Cancelled"}, "direction": "before", "depth": 5, "time_range": "90d"}',
        "metadata": {"report_type": "flows", "query_complexity": "moderate", "vertical": "b2b_saas", "schema_events": "Subscription Cancelled, Support Ticket Created, Login, Feature Used, Plan Downgraded", "schema_properties": "plan_type, platform"},
        "tags": ["flows", "moderate", "churn_analysis"],
    },
    {
        "input": "What do users do before creating a support ticket?",
        "expected": '{"report_type": "flows", "ending_event": {"event": "Support Ticket Created"}, "direction": "before", "depth": 5, "time_range": "30d"}',
        "metadata": {"report_type": "flows", "query_complexity": "simple", "vertical": "b2b_saas", "schema_events": "Support Ticket Created, Error Encountered, Feature Used, Login, Page Viewed", "schema_properties": "error_type, feature_name, platform"},
        "tags": ["flows", "simple", "support"],
    },
    {
        "input": "Show me what happens after users encounter an error",
        "expected": '{"report_type": "flows", "starting_event": {"event": "Error Encountered"}, "direction": "after", "depth": 5, "time_range": "30d"}',
        "metadata": {"report_type": "flows", "query_complexity": "simple", "vertical": "fintech", "schema_events": "Error Encountered, Support Ticket Created, Login, Session Started, Feature Used", "schema_properties": "error_type, platform"},
        "tags": ["flows", "simple", "error_recovery"],
    },
    {
        "input": "What's the typical user journey from first login to first purchase?",
        "expected": '{"report_type": "flows", "starting_event": {"event": "Login"}, "ending_event": {"event": "Item Purchased"}, "direction": "after", "depth": 7, "time_range": "90d"}',
        "metadata": {"report_type": "flows", "query_complexity": "complex", "vertical": "ecommerce", "schema_events": "Login, Page Viewed, Search Performed, Item Purchased, Checkout Started, Checkout Completed", "schema_properties": "platform, country, amount_usd"},
        "tags": ["flows", "complex", "purchase_journey"],
    },
    # --- Cross-report / ambiguous queries ---
    {
        "input": "How is our product doing?",
        "expected": '{"note": "This question is too broad. Suggest: DAU/MAU trend (Insights), signup-to-activation funnel (Funnels), or 30-day retention curve (Retention). Ask the user what specific metric they care about."}',
        "metadata": {"report_type": "ambiguous", "query_complexity": "complex", "vertical": "b2b_saas", "schema_events": "Sign Up, Login, Feature Used, Session Started", "schema_properties": "plan_type, platform"},
        "tags": ["ambiguous", "complex", "clarification_needed"],
    },
    {
        "input": "Why are users churning?",
        "expected": '{"note": "Causal questions cannot be answered directly. Recommend: 1) Flows report showing paths before Subscription Cancelled, 2) Retention comparison between churned vs active users, 3) Insights breakdown of last actions before churn. The AI should NOT attempt to provide causal reasoning."}',
        "metadata": {"report_type": "ambiguous", "query_complexity": "complex", "vertical": "b2b_saas", "schema_events": "Subscription Cancelled, Support Ticket Created, Login, Feature Used", "schema_properties": "plan_type, platform"},
        "tags": ["ambiguous", "complex", "causal", "limitation"],
    },
    {
        "input": "Show me everything about user engagement",
        "expected": '{"note": "Too broad. Suggest specific reports: DAU/WAU/MAU trends (Insights), feature usage breakdown (Insights), session duration distribution (Insights), or activation funnel (Funnels). Ask which aspect of engagement matters most."}',
        "metadata": {"report_type": "ambiguous", "query_complexity": "complex", "vertical": "b2b_saas", "schema_events": "Login, Feature Used, Session Started, Session Ended", "schema_properties": "platform, feature_name, duration_seconds"},
        "tags": ["ambiguous", "complex", "clarification_needed"],
    },
    # --- Schema mismatch queries (should detect missing events/properties) ---
    {
        "input": "Show me the NPS score trend over time",
        "expected": '{"note": "No NPS-related event found in your project schema. If you track NPS, check that the event is being sent to Mixpanel. Available events: [list schema events]."}',
        "metadata": {"report_type": "insights", "query_complexity": "simple", "vertical": "b2b_saas", "schema_events": "Sign Up, Login, Feature Used, Session Started", "schema_properties": "plan_type, platform", "expected_behavior": "schema_mismatch"},
        "tags": ["schema_mismatch", "simple", "missing_event"],
    },
    {
        "input": "Break down revenue by product category",
        "expected": '{"note": "The property \'product_category\' does not exist in your project schema. Available properties for Item Purchased: platform, country, amount_usd. If you want to break down by product, add a product_category property to your Item Purchased event."}',
        "metadata": {"report_type": "insights", "query_complexity": "moderate", "vertical": "ecommerce", "schema_events": "Item Purchased, Checkout Completed", "schema_properties": "platform, country, amount_usd", "expected_behavior": "schema_mismatch"},
        "tags": ["schema_mismatch", "moderate", "missing_property"],
    },
    # --- Multi-turn golden dataset rows ---
    {
        "input": "Can you also break it down by platform?",
        "expected": '{"action": "Add breakdown by platform to existing Insights report", "report_type": "insights", "modification": "add_breakdown", "breakdown": {"property": "platform"}}',
        "metadata": {
            "report_type": "insights", "query_complexity": "simple", "vertical": "b2b_saas",
            "schema_events": "Sign Up, Login, Feature Used", "schema_properties": "plan_type, platform, country",
            "is_multi_turn": True, "turn_count": 3,
            "conversation_history": [
                {"role": "user", "content": "Show me daily active users over the last 30 days"},
                {"role": "assistant", "content": "I've created an Insights report showing unique users with Session Started events over the last 30 days, displayed as a daily trend line. You're averaging about 12,400 DAU with a slight upward trend."},
            ],
        },
        "tags": ["multi_turn", "insights", "follow_up", "breakdown"],
    },
    {
        "input": "What if I look at just the last 7 days instead?",
        "expected": '{"action": "Modify time range to 7d on existing funnel", "report_type": "funnels", "modification": "change_time_range", "time_range": "7d"}',
        "metadata": {
            "report_type": "funnels", "query_complexity": "simple", "vertical": "ecommerce",
            "schema_events": "Checkout Started, Checkout Completed, Cart Abandoned", "schema_properties": "platform, amount_usd",
            "is_multi_turn": True, "turn_count": 3,
            "conversation_history": [
                {"role": "user", "content": "What's the drop-off rate between Checkout Started and Checkout Completed?"},
                {"role": "assistant", "content": "Your checkout funnel over the last 30 days shows 68.4% conversion from Checkout Started to Checkout Completed. The main drop-off happens within the first 2 minutes."},
            ],
        },
        "tags": ["multi_turn", "funnels", "follow_up", "time_range"],
    },
    {
        "input": "Now compare that to users who came from paid channels",
        "expected": '{"action": "Add segment filter for referral_source=paid to existing retention report", "report_type": "retention", "modification": "add_segment", "filter": {"property": "referral_source", "value": "paid"}}',
        "metadata": {
            "report_type": "retention", "query_complexity": "moderate", "vertical": "b2b_saas",
            "schema_events": "Sign Up, Login, Feature Used", "schema_properties": "plan_type, platform, referral_source",
            "is_multi_turn": True, "turn_count": 5,
            "conversation_history": [
                {"role": "user", "content": "Show me 30-day retention for users who signed up last month"},
                {"role": "assistant", "content": "Here's the 30-day retention curve for last month's signup cohort. Day 1 retention is 45%, Day 7 is 28%, and Day 30 is 15%."},
                {"role": "user", "content": "How does that compare for organic vs paid users?"},
                {"role": "assistant", "content": "Organic users retain significantly better: Day 30 retention is 19% for organic vs 11% for paid. The gap widens after Day 7."},
            ],
        },
        "tags": ["multi_turn", "retention", "follow_up", "segmentation"],
    },
    # --- Spark AI edge cases: taxonomy quality issues ---
    {
        "input": "How many users clicked the buy button?",
        "expected": '{"note": "No event named \'Buy Button Clicked\' or similar exists in the schema. The closest events are \'Checkout Started\' and \'Item Purchased\'. Ask the user which one they mean, or suggest looking at the Checkout Started → Checkout Completed funnel."}',
        "metadata": {"report_type": "insights", "query_complexity": "simple", "vertical": "ecommerce", "schema_events": "Page Viewed, Checkout Started, Checkout Completed, Item Purchased", "schema_properties": "platform, amount_usd", "expected_behavior": "taxonomy_gap"},
        "tags": ["taxonomy", "simple", "event_mismatch"],
    },
    {
        "input": "Show me revenue by product line for the last year",
        "expected": '{"note": "The \'Item Purchased\' event has an \'amount_usd\' property (sum aggregation works) but there is no \'product_line\' property in the schema. Available breakdowns: platform, country. The customer may need to add product_line as an event property to enable this analysis."}',
        "metadata": {"report_type": "insights", "query_complexity": "moderate", "vertical": "ecommerce", "schema_events": "Item Purchased, Checkout Completed", "schema_properties": "amount_usd, platform, country", "expected_behavior": "missing_property"},
        "tags": ["taxonomy", "moderate", "missing_property", "revenue"],
    },
    # --- Spark AI edge cases: aggregation ambiguity ---
    {
        "input": "How popular is the search feature?",
        "expected": '{"note": "Ambiguous. \'Popular\' could mean: unique users who searched (Insights, unique), total searches performed (Insights, total), or percentage of active users who search (Insights, unique Search / unique Session Started). Recommend clarifying which metric, default to unique users if the user confirms."}',
        "metadata": {"report_type": "insights", "query_complexity": "moderate", "vertical": "ecommerce", "schema_events": "Search Performed, Session Started, Page Viewed", "schema_properties": "platform, browser", "expected_behavior": "aggregation_ambiguity"},
        "tags": ["ambiguous", "moderate", "aggregation"],
    },
]


def create_golden_dataset(project_name: str):
    print("\n--- Creating golden dataset ---")
    dataset = init_dataset(project=project_name, name="Spark AI golden set")

    for row in GOLDEN_DATASET_ROWS:
        dataset.insert(
            input=row["input"],
            expected=row["expected"],
            metadata=row["metadata"],
            tags=row["tags"],
        )

    dataset.flush()
    print(f"  Inserted {len(GOLDEN_DATASET_ROWS)} golden test cases")
    print(f"  Dataset: 'Spark AI golden set'")


_generated_traces: list[dict] = []


def create_datasets_from_traces(project_name: str):
    if not _generated_traces:
        print("\n--- Skipping dataset-from-traces (no traces generated yet) ---")
        return

    print(f"\n--- Building datasets from {len(_generated_traces)} generated traces ---")

    # 1. Production samples
    prod_ds = init_dataset(project=project_name, name="Production Spark AI samples")
    good_traces = [t for t in _generated_traces if t["quality_tier"] == "good"]
    sampled_good = []
    report_types_seen = set()
    for t in good_traces:
        if t["report_type"] not in report_types_seen or len(sampled_good) < 30:
            sampled_good.append(t)
            report_types_seen.add(t["report_type"])
        if len(sampled_good) >= 50:
            break

    for t in sampled_good:
        prod_ds.insert(
            input=t["user_query"],
            expected=t["spark_response"],
            metadata={
                "report_type": t["report_type"],
                "vertical": t["vertical"],
                "prompt_version": t["prompt_version"],
                "model": t["model"],
                "customer_id": t["customer_id"],
                "source": "production_logs",
            },
            tags=[t["report_type"], t["vertical"], "production"],
        )
    prod_ds.flush()
    print(f"  'Production Spark AI samples': {len(sampled_good)} traces")

    # 2. Flagged for review
    review_ds = init_dataset(project=project_name, name="Flagged for query review")
    flagged_traces = [t for t in _generated_traces if t["quality_tier"] in ("flagged", "needs_review")]
    for t in flagged_traces[:40]:
        tags = [t["report_type"], t["vertical"], t["quality_tier"]]
        if t["quality_tier"] == "flagged":
            tags.append("query_error")
        review_ds.insert(
            input=t["user_query"],
            expected=t["spark_response"],
            metadata={
                "report_type": t["report_type"],
                "vertical": t["vertical"],
                "prompt_version": t["prompt_version"],
                "quality_tier": t["quality_tier"],
                "customer_id": t["customer_id"],
                "flagged_reason": t["quality_tier"],
                "source": "production_logs_flagged",
                "review_status": "pending",
            },
            tags=tags,
        )
    review_ds.flush()
    print(f"  'Flagged for query review': {min(40, len(flagged_traces))} traces")

    # 3. Playground scenarios
    playground_ds = init_dataset(project=project_name, name="Playground demo scenarios")
    scenario_count = 0

    for t in _generated_traces:
        if t["quality_tier"] == "flagged":
            playground_ds.insert(
                input=t["user_query"],
                expected="This query was flagged for incorrect output. Review the generated query against the user's intent and the project schema.",
                metadata={
                    **{k: t[k] for k in ("report_type", "vertical", "prompt_version", "model", "customer_id")},
                    "scenario": "flagged_query",
                    "demo_note": "Show HITL flow: review flagged query -> correct -> add to golden dataset",
                    "quality_tier": t["quality_tier"],
                },
                tags=["playground", "flagged", "hitl_demo"],
            )
            scenario_count += 1
            break

    for version in PROMPT_VERSIONS:
        for t in _generated_traces:
            if t["prompt_version"] == version and t["quality_tier"] == "good" and t["report_type"] == "insights":
                playground_ds.insert(
                    input=t["user_query"],
                    expected=t["spark_response"],
                    metadata={
                        **{k: t[k] for k in ("report_type", "vertical", "prompt_version", "model", "customer_id")},
                        "scenario": "prompt_comparison",
                        "demo_note": f"Prompt version: {version}. Compare structured vs conversational style.",
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
                    expected=t["spark_response"],
                    metadata={
                        **{k: t[k] for k in ("report_type", "vertical", "prompt_version", "model", "customer_id")},
                        "scenario": "model_comparison",
                        "demo_note": f"Model: {model_name}. Compare cost vs quality across models.",
                    },
                    tags=["playground", "model_comparison", model_name],
                )
                scenario_count += 1
                break

    playground_ds.flush()
    print(f"  'Playground demo scenarios': {scenario_count} scenarios (built from actual traces)")


# Prior conversation context for multi-turn traces
PRIOR_CONVERSATION_SNIPPETS = [
    [
        {"role": "user", "content": "Show me daily active users over the last 30 days"},
        {"role": "assistant", "content": "I've created an Insights report showing unique users with Session Started events over the last 30 days, displayed as a daily trend line. You're averaging about 12,400 DAU with a slight upward trend this week."},
    ],
    [
        {"role": "user", "content": "What's the checkout conversion rate?"},
        {"role": "assistant", "content": "Your Checkout Started to Checkout Completed funnel shows 68.4% conversion over the last 30 days. The main drop-off happens within the first 2 minutes of starting checkout."},
    ],
    [
        {"role": "user", "content": "How many users signed up this week?"},
        {"role": "assistant", "content": "842 unique users signed up in the last 7 days. That's up 12% from the previous week. Most signups came from organic search (45%) and direct traffic (30%)."},
        {"role": "user", "content": "Nice! How does that compare to last month's weekly average?"},
        {"role": "assistant", "content": "Last month's weekly signup average was 715 users, so this week is 18% above the monthly average. The increase appears correlated with the new landing page that went live on Monday."},
    ],
    [
        {"role": "user", "content": "Show me retention for the March signup cohort"},
        {"role": "assistant", "content": "The March signup cohort shows: Day 1 retention 52%, Day 7 retention 31%, Day 14 retention 22%, Day 30 retention 16%. This is slightly above your historical average of 14% at Day 30."},
    ],
    [
        {"role": "user", "content": "I'm looking at feature adoption. Which features are most used?"},
        {"role": "assistant", "content": "In the last 30 days by unique users: Insights (8,200), Funnels (5,100), Retention (3,800), Flows (2,400), Spark AI (1,900). Spark AI usage has grown 40% month-over-month since launch."},
        {"role": "user", "content": "Interesting. What about by plan type?"},
        {"role": "assistant", "content": "Enterprise users use an average of 4.2 features vs 2.1 for free users. The biggest gap is in Retention and Flows — those are used 5x more by enterprise customers."},
    ],
]


def build_system_prompt(config: TraceConfig) -> str:
    base = (
        "You are Spark AI, Mixpanel's natural language analytics assistant. "
        "You translate user questions into Mixpanel reports. "
        "Only reference events and properties in the project schema. "
        "If something doesn't exist in the schema, say so clearly."
    )

    if config.prompt_version == "v3.1-structured":
        return (
            base + "\n\nStyle: Be precise and query-focused. Lead with the report configuration. "
            "Describe exact events, aggregations, breakdowns, filters, and time ranges. "
            "Suggest modifications in the query builder. Keep it concise."
        )
    else:
        return (
            base + "\n\nStyle: Be explanatory and insight-driven. Create the report but also highlight "
            "interesting patterns. Ask follow-up questions. Use natural, friendly language."
        )


def log_trace(logger, config: TraceConfig, trace_idx: int, oai_client):
    perf = generate_latency_metrics(config)
    system_prompt = build_system_prompt(config)
    schema = config.schema_context

    # ~35% of traces include prior conversation context
    has_history = random.random() < 0.35
    prior_turns = random.choice(PRIOR_CONVERSATION_SNIPPETS) if has_history else []

    schema_context_str = (
        f"Events: {', '.join(schema['events'])}\n"
        f"Properties: {', '.join(schema['properties'])}"
    )

    messages = [{"role": "system", "content": system_prompt + f"\n\nProject schema:\n{schema_context_str}"}]
    messages.extend(prior_turns)
    messages.append({"role": "user", "content": config.user_query})

    with logger.start_span(name="spark-ai-query") as root_span:
        # Step 1: Schema lookup
        with root_span.start_span(
            name="Schema lookup",
            span_attributes={"type": "tool"},
        ) as schema_span:
            schema_span.log(
                input={
                    "customer_id": config.customer_id,
                    "project_id": f"proj-{random.randint(10000, 99999)}",
                },
                output={
                    "events": schema["events"],
                    "properties": schema["properties"],
                    "event_count": len(schema["events"]),
                    "property_count": len(schema["properties"]),
                },
                metadata={"vertical": config.vertical},
                metrics={"latency": perf["schema_lookup_latency_s"]},
            )

        # Step 2: LLM call via wrap_openai
        response = oai_client.chat.completions.create(
            model=_get_api_model(config.model),
            messages=messages,
        )
        spark_response = response.choices[0].message.content

        # Step 3: Query validation
        query_output = generate_query_output(config)
        validation_passed = config.quality_tier != "flagged"
        validation_issues = []
        if not validation_passed:
            validation_issues = random.sample(
                [
                    "event_not_in_schema",
                    "property_not_in_schema",
                    "wrong_aggregation_type",
                    "invalid_time_range",
                    "empty_result_set",
                ],
                k=random.randint(1, 2),
            )

        with root_span.start_span(
            name="Query validation",
            span_attributes={"type": "tool"},
        ) as val_span:
            val_span.log(
                input={"generated_query": query_output},
                output={
                    "valid": validation_passed,
                    "issues": validation_issues,
                    "confidence": round(
                        random.uniform(0.85, 0.99) if validation_passed else random.uniform(0.3, 0.6),
                        3,
                    ),
                },
                metadata={"validator_version": "v2.1"},
                metrics={"latency": perf["query_validation_latency_s"]},
            )

        conversation = prior_turns + [
            {"role": "user", "content": config.user_query},
            {"role": "assistant", "content": spark_response},
        ]

        scores = generate_scores(config)

        root_span.log(
            input=config.user_query,
            output=spark_response,
            scores=scores,
            metadata={
                "customer_id": config.customer_id,
                "vertical": config.vertical,
                "report_type": config.report_type,
                "query_complexity": config.query_complexity,
                "prompt_version": config.prompt_version,
                "model": config.model,
                "quality_tier": config.quality_tier,
                "trace_index": trace_idx,
                "turn_count": len(conversation),
                "has_prior_context": has_history,
                "schema_events": ", ".join(schema["events"]),
                "schema_properties": ", ".join(schema["properties"]),
                "generated_query": str(query_output),
                "validation_result": "passed" if validation_passed else "failed",
            },
        )

    _generated_traces.append({
        "customer_id": config.customer_id,
        "vertical": config.vertical,
        "report_type": config.report_type,
        "query_complexity": config.query_complexity,
        "prompt_version": config.prompt_version,
        "model": config.model,
        "quality_tier": config.quality_tier,
        "user_query": config.user_query,
        "spark_response": spark_response,
        "conversation": conversation,
        "trace_idx": trace_idx,
    })


# Multi-turn conversation templates
MULTI_TURN_CONVERSATIONS = [
    {
        "report_type": "insights",
        "vertical": "b2b_saas",
        "turns": [
            {"role": "user", "content": "Show me daily active users over the last 30 days"},
            {"role": "assistant", "content": "I've created an Insights report showing unique users with Session Started events over the last 30 days. You're averaging about 12,400 DAU with a slight upward trend."},
            {"role": "user", "content": "Can you break that down by platform?"},
            {"role": "assistant", "content": "Done! The breakdown shows: Web 65%, iOS 22%, Android 13%. Web DAU has been stable while mobile is growing — iOS is up 8% and Android up 15% month-over-month."},
            {"role": "user", "content": "Interesting. What about just enterprise users?"},
            {"role": "assistant", "content": "Enterprise users make up 18% of DAU (about 2,200 users). Their platform split is different: Web 82%, iOS 12%, Android 6%. Enterprise users are much more desktop-heavy than your overall base."},
        ],
    },
    {
        "report_type": "funnels",
        "vertical": "ecommerce",
        "turns": [
            {"role": "user", "content": "Show me the checkout funnel conversion rate"},
            {"role": "assistant", "content": "Your Checkout Started → Checkout Completed funnel shows 68.4% conversion over the last 30 days. That means about 1 in 3 users who start checkout don't finish."},
            {"role": "user", "content": "Is that good or bad?"},
            {"role": "assistant", "content": "Industry benchmarks for ecommerce checkout are 60-70%, so you're right at the top of the range. However, there's still opportunity — the best performers hit 75-80%. Want me to break it down by platform to find where the biggest drops are?"},
            {"role": "user", "content": "Yes please"},
            {"role": "assistant", "content": "Here's the breakdown: Web 72%, iOS 65%, Android 58%. Android has the lowest conversion — that's worth investigating. Could be a UX issue with the Android checkout flow."},
        ],
    },
    {
        "report_type": "retention",
        "vertical": "b2b_saas",
        "turns": [
            {"role": "user", "content": "Show me 30-day retention for last month's signups"},
            {"role": "assistant", "content": "Last month's signup cohort shows: Day 1: 52%, Day 7: 31%, Day 14: 22%, Day 30: 16%. The steepest drop is in the first 24 hours."},
            {"role": "user", "content": "How does that compare to users who completed onboarding?"},
            {"role": "assistant", "content": "Big difference! Users who completed all 5 onboarding steps: Day 1: 78%, Day 7: 55%, Day 30: 34%. That's more than 2x the retention at Day 30 compared to users who didn't complete onboarding."},
            {"role": "user", "content": "Wow. Which onboarding step has the biggest drop-off?"},
            {"role": "assistant", "content": "The onboarding funnel shows: Step 1 (Create Project): 89%, Step 2 (Install SDK): 62%, Step 3 (Send Event): 48%, Step 4 (Create Report): 41%, Step 5 (Invite Team): 33%. The biggest single drop is at Step 2 — Install SDK. That's the technical barrier."},
        ],
    },
    {
        "report_type": "flows",
        "vertical": "b2b_saas",
        "turns": [
            {"role": "user", "content": "What do users do right before they cancel their subscription?"},
            {"role": "assistant", "content": "Looking at paths leading to Subscription Cancelled over the last 90 days: 35% last visited Support Ticket Created, 25% last used Feature Used (primarily Insights only), 20% came from Login → Dashboard Viewed with no other actions, 12% downgraded first."},
            {"role": "user", "content": "So support tickets are a big predictor of churn?"},
            {"role": "assistant", "content": "Yes — users who create a support ticket are 3.2x more likely to cancel within 30 days. But it depends on resolution: tickets resolved within 24 hours have much lower churn rates. Want me to build a retention comparison for ticket creators vs non-creators?"},
            {"role": "user", "content": "Yes, let's see that"},
            {"role": "assistant", "content": "Here's the retention comparison: Users with support tickets: Day 30 retention 8%. Users without: Day 30 retention 18%. But users whose tickets were resolved same-day: Day 30 retention 15% — almost as good as non-ticket users. Resolution speed matters more than the ticket itself."},
        ],
    },
]


def log_multi_turn_trace(logger, conversation: dict, trace_idx: int, oai_client):
    report_type = conversation["report_type"]
    vertical = conversation["vertical"]
    template_turns = conversation["turns"]
    prompt_version = random.choice(PROMPT_VERSIONS)
    model = random.choice(MODELS)
    schema = next((s for s in SCHEMA_CONTEXTS if s["vertical"] == vertical), SCHEMA_CONTEXTS[0])

    quality_tier = random.choices(["good", "needs_review", "flagged"], weights=[0.80, 0.15, 0.05], k=1)[0]
    config = TraceConfig(
        customer_id=f"CUST-{random.randint(10000, 99999)}",
        vertical=vertical,
        report_type=report_type,
        query_complexity="moderate",
        prompt_version=prompt_version,
        model=model,
        quality_tier=quality_tier,
        user_query=template_turns[0]["content"],
        schema_context=schema,
    )
    system_prompt = build_system_prompt(config)
    schema_context_str = f"Events: {', '.join(schema['events'])}\nProperties: {', '.join(schema['properties'])}"
    validation_passed = quality_tier != "flagged"

    with logger.start_span(name="spark-ai-query") as root_span:
        # Schema lookup span (same as single-turn)
        with root_span.start_span(
            name="Schema lookup",
            span_attributes={"type": "tool"},
        ) as schema_span:
            schema_span.log(
                input={"customer_id": config.customer_id, "project_id": f"proj-{random.randint(10000, 99999)}"},
                output={"events": schema["events"], "properties": schema["properties"], "event_count": len(schema["events"])},
                metadata={"vertical": config.vertical},
                metrics={"latency": round(random.uniform(0.02, 0.15), 3)},
            )

        accumulated_messages = [{"role": "system", "content": system_prompt + f"\n\nProject schema:\n{schema_context_str}"}]
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

        # Query validation span (same as single-turn)
        with root_span.start_span(
            name="Query validation",
            span_attributes={"type": "tool"},
        ) as val_span:
            val_span.log(
                input={"generated_query": "multi-turn refinement"},
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
                "report_type": report_type,
                "query_complexity": "multi_turn",
                "prompt_version": prompt_version,
                "model": config.model,
                "quality_tier": quality_tier,
                "trace_index": trace_idx,
                "turn_count": len(actual_turns),
                "is_multi_turn": True,
                "schema_events": ", ".join(schema["events"]),
                "schema_properties": ", ".join(schema["properties"]),
                "validation_result": "passed" if validation_passed else "failed",
            },
        )

    _generated_traces.append({
        "customer_id": config.customer_id,
        "vertical": vertical,
        "report_type": report_type,
        "query_complexity": "multi_turn",
        "prompt_version": prompt_version,
        "model": model,
        "quality_tier": quality_tier,
        "user_query": user_turns[0]["content"],
        "spark_response": last_response,
        "conversation": actual_turns,
        "trace_idx": trace_idx,
        "is_multi_turn": True,
    })


def _run_experiment_row(experiment, row, prompt_ver, model_name, correctness_base, quality_base, oai_client):
    row_input = row.get("input", "")
    row_expected = row.get("expected", "")
    row_metadata = row.get("metadata", {})
    report_type = row_metadata.get("report_type", "insights")
    vertical = row_metadata.get("vertical", "b2b_saas")

    schema = next((s for s in SCHEMA_CONTEXTS if s["vertical"] == vertical), SCHEMA_CONTEXTS[0])
    config = TraceConfig(
        customer_id=f"CUST-{random.randint(10000, 99999)}",
        vertical=vertical,
        report_type=report_type,
        query_complexity=row_metadata.get("query_complexity", "moderate"),
        prompt_version=prompt_ver,
        model=model_name,
        quality_tier="good",
        user_query=row_input if isinstance(row_input, str) else str(row_input),
        schema_context=schema,
    )
    system_prompt = build_system_prompt(config)
    schema_context_str = f"Events: {row_metadata.get('schema_events', ', '.join(schema['events']))}\nProperties: {row_metadata.get('schema_properties', ', '.join(schema['properties']))}"

    messages = [{"role": "system", "content": system_prompt + f"\n\nProject schema:\n{schema_context_str}"}]
    conv_history = row_metadata.get("conversation_history", [])
    messages.extend(conv_history)
    user_msg = row_input if isinstance(row_input, str) else str(row_input)
    messages.append({"role": "user", "content": user_msg})

    with experiment.start_span(name="spark-ai-eval") as span:
        with span.start_span(
            name="Schema lookup",
            span_attributes={"type": "tool"},
        ) as schema_span:
            schema_span.log(
                input={"customer_id": config.customer_id},
                output={"events": schema["events"], "properties": schema["properties"]},
                metadata={"vertical": vertical},
            )

        response = oai_client.chat.completions.create(
            model=_get_api_model(model_name),
            messages=messages,
        )
        spark_response = response.choices[0].message.content

        with span.start_span(
            name="Query validation",
            span_attributes={"type": "tool"},
        ) as val_span:
            val_span.log(
                input={"response": spark_response},
                output={"valid": True, "issues": [], "confidence": round(random.uniform(0.90, 0.99), 3)},
                metadata={"validator_version": "v2.1"},
            )

        complexity = row_metadata.get("query_complexity", "moderate")
        complexity_penalty = {"simple": 0.0, "moderate": -0.08, "complex": -0.18}.get(complexity, -0.05)
        cb, qb = correctness_base, quality_base

        expected_behavior = row_metadata.get("expected_behavior", "")
        if expected_behavior in ("schema_mismatch", "missing_property", "taxonomy_gap", "aggregation_ambiguity"):
            cb -= 0.15
        if report_type == "ambiguous":
            cb -= 0.10

        scores = {
            "query_correctness": round(max(0.0, min(1.0, cb + complexity_penalty + random.uniform(-0.08, 0.08))), 3),
            "schema_adherence": round(max(0.0, min(1.0, 0.90 + complexity_penalty * 0.5 + random.uniform(-0.06, 0.06))), 3),
            "response_quality": round(max(0.0, min(1.0, qb + complexity_penalty * 0.5 + random.uniform(-0.08, 0.08))), 3),
        }

        span.log(
            input=user_msg,
            output=spark_response,
            expected=row_expected,
            scores=scores,
            metadata={
                "prompt_version": prompt_ver,
                "model": model_name,
                "report_type": report_type,
                "vertical": vertical,
                "query_complexity": row_metadata.get("query_complexity", "moderate"),
                "schema_events": row_metadata.get("schema_events", ""),
                "schema_properties": row_metadata.get("schema_properties", ""),
                "generated_query": str(generate_query_output(config)),
                "validation_result": "passed",
            },
            tags=row.get("tags", []),
        )


def run_experiments(project_name: str, oai_client=None, parallelism: int = 20):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not oai_client:
        print("\n--- Skipping experiments (OPENAI_API_KEY required for real LLM calls) ---")
        return

    print("\n--- Running experiments (prompt A vs B comparison) ---")

    dataset_rows = GOLDEN_DATASET_ROWS
    if not dataset_rows:
        print("  Warning: Golden dataset is empty.")
        return

    test_rows = dataset_rows[:50]

    def run_single_experiment(exp_name, test_rows, prompt_ver, model_name, correctness_base, quality_base):
        print(f"  Running experiment: {exp_name} ({len(test_rows)} test cases, parallelism={parallelism})")
        experiment = braintrust.init(project_name, exp_name)

        completed = 0
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = {
                executor.submit(
                    _run_experiment_row, experiment, row, prompt_ver, model_name,
                    correctness_base, quality_base, oai_client,
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
        ("v3.1-structured", "Prompt A: Structured query style"),
        ("v3.2-conversational", "Prompt B: Conversational insight style"),
    ]:
        correctness_base = 0.88 if prompt_ver == "v3.1-structured" else 0.78
        quality_base = 0.72 if prompt_ver == "v3.1-structured" else 0.85
        run_single_experiment(exp_name, test_rows, prompt_ver, "gpt-5-mini", correctness_base, quality_base)

    print("  Prompt A/B experiments complete")

    print("\n  Running model comparison experiments...")
    for model_name in ["gpt-5-mini", "gpt-5-nano"]:
        correctness_base = 0.88 if model_name == "gpt-5-mini" else 0.82
        quality_base = 0.78
        run_single_experiment(
            f"Model comparison: {model_name}", test_rows, "v3.1-structured",
            model_name, correctness_base, quality_base,
        )

    print("  All experiments complete — compare in Braintrust UI")


async def generate_traces(project: str, count: int, parallelism: int, oai_client):
    print(f"Initializing Braintrust logger for project '{project}'...")
    logger = braintrust.init_logger(project=project)
    print(f"Project ID: {logger.project.id}")
    print("  Using real OpenAI API calls (wrap_openai enabled)")

    semaphore = asyncio.Semaphore(parallelism)
    completed = 0
    errors = 0

    report_counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
    prompt_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    vertical_counts: dict[str, int] = {}

    async def generate_one(idx: int):
        nonlocal completed, errors
        async with semaphore:
            config = generate_trace_config(idx)

            report_counts[config.report_type] = report_counts.get(config.report_type, 0) + 1
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

    print(f"\nGenerating {count} Spark AI traces...")
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
    for model, cnt in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"  {model}: {cnt} ({cnt/count*100:.1f}%)")

    print("\nReport type distribution:")
    for rt, cnt in sorted(report_counts.items(), key=lambda x: -x[1]):
        print(f"  {rt}: {cnt} ({cnt/count*100:.1f}%)")

    print("\nCustomer vertical distribution:")
    for vert, cnt in sorted(vertical_counts.items(), key=lambda x: -x[1]):
        print(f"  {vert}: {cnt} ({cnt/count*100:.1f}%)")

    print(f"\nView traces at: https://www.braintrust.dev")


def main():
    import os

    parser = argparse.ArgumentParser(description="Seed Mixpanel Spark AI demo into Braintrust")
    parser.add_argument("--project", type=str, default="Mixpanel Spark AI", help="Project name")
    parser.add_argument("--count", type=int, default=500, help="Number of traces to generate")
    parser.add_argument("--parallelism", type=int, default=50, help="Concurrent trace generation limit")
    parser.add_argument("--skip-traces", action="store_true", help="Only create prompts, scorers, dataset, and experiments (no production traces)")
    parser.add_argument("--traces-only", action="store_true", help="Only create traces (skip prompts/scorers/dataset/experiments)")
    parser.add_argument("--skip-experiments", action="store_true", help="Skip the A/B experiment comparison")

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
                print("\n--- Skipping experiments (OPENAI_API_KEY required for real LLM calls) ---")
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
