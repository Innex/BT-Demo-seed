"""Phase 2: Generate all 13 domain-specific data structures via LLM calls."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Callable, Optional

from .models import ResearchReport, CustomerData

PROMPTS_DIR = Path(__file__).parent / "prompts"

DEFAULT_MODEL = "gpt-5-mini"


def _call_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
    )
    return resp.output_text


def _parse_json(text: str, retry_prompt_context: str = "") -> dict:
    """Parse JSON from LLM response, handling code fences and retrying on failure."""
    json_str = text.strip()
    if json_str.startswith("```"):
        json_str = json_str.split("\n", 1)[1]
        json_str = json_str.rsplit("```", 1)[0]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        if not retry_prompt_context:
            raise
        retry_prompt = f"The following response was not valid JSON. Fix it and return ONLY valid JSON:\n\n{text}"
        retry_result = _call_llm(retry_prompt)
        json_str = retry_result.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("\n", 1)[1]
            json_str = json_str.rsplit("```", 1)[0]
        return json.loads(json_str)


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def _normalize_weights(weights, n: int) -> list[float]:
    """Coerce LLM-produced weights into a valid probability distribution.

    The LLM occasionally returns weights that don't sum to 1.0, are the wrong
    shape (dict instead of list), or contain non-numeric values. random.choices
    only accepts a numeric list — fix everything here so trace generation
    never crashes mid-run, and so volumes match the LLM's stated intent.
    """
    if n <= 0:
        return []
    uniform = [1.0 / n] * n
    if not isinstance(weights, list) or len(weights) != n:
        return uniform
    try:
        floats = [float(w) for w in weights]
    except (TypeError, ValueError):
        return uniform
    total = sum(floats)
    if total <= 0 or not math.isfinite(total):
        return uniform
    return [w / total for w in floats]


def _compose_project_name(company: str, ai_product: str) -> str:
    """Combine company + product without duplicating the company prefix.

    The LLM sometimes returns ai_product_name with the company already in it
    ("Salesforce Einstein and Agentforce"). Joining naively yields
    "Salesforce Salesforce Einstein and Agentforce".
    """
    if not ai_product:
        return company
    if not company:
        return ai_product
    if ai_product.lower().startswith(company.lower()):
        return ai_product
    return f"{company} {ai_product}"


def _checkpointed(cache_dir: Path | None, step: str, fn: Callable[[], dict]) -> dict:
    """Run fn() once and cache its JSON result by step name.

    On subsequent runs (after a partial failure), reuse the cached output and
    skip the LLM call. Delete a single step file to force a re-run of just
    that step.
    """
    if cache_dir is None:
        return fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{step}.json"
    if cache_file.exists():
        print(f"    [cached] {step}")
        return json.loads(cache_file.read_text())
    result = fn()
    cache_file.write_text(json.dumps(result, indent=2))
    return result


def _report_fields(report: ResearchReport) -> dict:
    """Common template fields from the research report."""
    return {
        "company_name": report.company_name,
        "ai_product_name": report.ai_product_name,
        "ai_product_description": report.ai_product_description,
        "product_domain": report.product_domain,
        "input_type": report.input_type,
        "output_type": report.output_type,
    }


def generate_feature_modes(report: ResearchReport) -> dict:
    """Generate AI feature modes, weights, and customer verticals."""
    prompt = _load_prompt("feature_modes.txt").format(**_report_fields(report))
    result = _call_llm(prompt)
    return _parse_json(result, retry_prompt_context="feature_modes")


def generate_entities(report: ResearchReport) -> dict:
    """Generate domain entities and properties."""
    prompt = _load_prompt("event_taxonomy.txt").format(**_report_fields(report))
    result = _call_llm(prompt)
    return _parse_json(result, retry_prompt_context="entities")


def generate_user_queries(report: ResearchReport, feature_modes: list[str], entities: list[dict]) -> dict:
    """Generate user queries organized by feature mode."""
    entities_summary = ", ".join(e["name"] for e in entities[:15])
    prompt = _load_prompt("user_queries.txt").format(
        **_report_fields(report),
        feature_modes=json.dumps(feature_modes),
        entities_summary=entities_summary,
    )
    result = _call_llm(prompt)
    return _parse_json(result, retry_prompt_context="user_queries")


def generate_response_templates(report: ResearchReport) -> dict:
    """Generate response templates, snippets, problematic responses, sample outputs."""
    prompt = _load_prompt("response_templates.txt").format(**_report_fields(report))
    result = _call_llm(prompt)
    return _parse_json(result, retry_prompt_context="response_templates")


def generate_system_prompts(report: ResearchReport, feature_modes: list[str]) -> dict:
    """Generate system prompts and style variants."""
    prompt = _load_prompt("system_prompts.txt").format(
        **_report_fields(report),
        feature_modes=json.dumps(feature_modes),
        evaluation_dimensions=json.dumps(report.evaluation_dimensions),
    )
    result = _call_llm(prompt)
    return _parse_json(result, retry_prompt_context="system_prompts")


def generate_scorers(report: ResearchReport) -> dict:
    """Generate 3 scorer definitions."""
    prompt = _load_prompt("scorer_prompts.txt").format(
        **_report_fields(report),
        evaluation_dimensions=json.dumps(report.evaluation_dimensions),
        failure_modes=json.dumps(report.failure_modes),
    )
    result = _call_llm(prompt)
    return _parse_json(result, retry_prompt_context="scorers")


def generate_facets(report: ResearchReport, feature_modes: list[str]) -> dict:
    """Generate 4 facet definitions."""
    prompt = _load_prompt("facet_prompts.txt").format(
        **_report_fields(report),
        feature_modes=json.dumps(feature_modes),
        failure_modes=json.dumps(report.failure_modes),
    )
    result = _call_llm(prompt)
    return _parse_json(result, retry_prompt_context="facets")


def generate_schema_contexts(
    report: ResearchReport,
    customer_verticals: list[str],
    entities: list[dict],
    properties: list[dict],
) -> dict:
    """Generate schema contexts per vertical."""
    prompt = _load_prompt("schema_contexts.txt").format(
        company_name=report.company_name,
        ai_product_name=report.ai_product_name,
        product_domain=report.product_domain,
        customer_verticals=json.dumps(customer_verticals),
        entities_summary=json.dumps([e["name"] for e in entities]),
        properties_summary=json.dumps([p["name"] for p in properties]),
    )
    result = _call_llm(prompt)
    return _parse_json(result, retry_prompt_context="schema_contexts")


def generate_golden_dataset(
    report: ResearchReport,
    feature_modes: list[str],
    entities: list[dict],
    properties: list[dict],
    schema_contexts: list[dict],
) -> dict:
    """Generate golden dataset rows."""
    prompt = _load_prompt("golden_dataset.txt").format(
        **_report_fields(report),
        feature_modes=json.dumps(feature_modes),
        entities_summary=json.dumps([e["name"] for e in entities[:15]]),
        properties_summary=json.dumps([p["name"] for p in properties]),
        schema_contexts_summary=json.dumps(schema_contexts[:2]),
    )
    result = _call_llm(prompt)
    return _parse_json(result, retry_prompt_context="golden_dataset")


def generate_multi_turn(report: ResearchReport, feature_modes: list[str]) -> dict:
    """Generate multi-turn conversations and prior conversation snippets."""
    prompt = _load_prompt("multi_turn.txt").format(
        **_report_fields(report),
        feature_modes=json.dumps(feature_modes),
    )
    result = _call_llm(prompt)
    return _parse_json(result, retry_prompt_context="multi_turn")


def synthesize_all(
    report: ResearchReport,
    verbose: bool = False,
    cache_dir: Path | None = None,
) -> CustomerData:
    """Run all 10 generation steps and return a complete CustomerData.

    If `cache_dir` is provided, each step's JSON output is cached to disk and
    reused on subsequent runs. This makes a partial failure cheap to resume —
    only the failing step (and steps after it) re-run.
    """

    print(f"\n--- Synthesizing data for {report.ai_product_name} ---")
    if cache_dir is not None:
        print(f"  Cache: {cache_dir}")
    data = CustomerData()
    data.company_name = report.company_name
    data.ai_product_name = report.ai_product_name
    data.project_name = _compose_project_name(report.company_name, report.ai_product_name)
    data.product_domain = report.product_domain

    # Step 1: Feature modes + verticals
    print("  [1/10] Generating feature modes...")
    fm = _checkpointed(cache_dir, "01_feature_modes", lambda: generate_feature_modes(report))
    data.ai_feature_modes = fm["feature_modes"]
    data.ai_feature_mode_weights = _normalize_weights(
        fm.get("feature_mode_weights", []), len(data.ai_feature_modes)
    )
    data.schema_contexts = []  # placeholder until step 8
    verticals = fm.get("customer_verticals", report.customer_verticals or ["general"])
    if verbose:
        print(f"    Modes: {data.ai_feature_modes}")

    # Step 2: Entities + properties
    print("  [2/10] Generating entities and properties...")
    ent = _checkpointed(cache_dir, "02_entities", lambda: generate_entities(report))
    data.domain_entities = ent["entities"]
    data.entity_properties = ent["entity_properties"]
    if verbose:
        print(f"    {len(data.domain_entities)} entities, {len(data.entity_properties)} properties")

    # Step 3: User queries
    print("  [3/10] Generating user queries...")
    uq = _checkpointed(
        cache_dir,
        "03_user_queries",
        lambda: generate_user_queries(report, data.ai_feature_modes, data.domain_entities),
    )
    data.user_queries = uq.get("user_queries", uq)
    if verbose:
        total_queries = sum(len(v) for v in data.user_queries.values())
        print(f"    {total_queries} queries across {len(data.user_queries)} modes")

    # Step 4: Response templates + snippets + sample outputs
    print("  [4/10] Generating response templates...")
    rt = _checkpointed(cache_dir, "04_response_templates", lambda: generate_response_templates(report))
    data.responses_structured = rt.get("responses_style_a", [])
    data.responses_conversational = rt.get("responses_style_b", [])
    data.insight_snippets = rt.get("insight_snippets", [])
    data.detail_snippets = rt.get("detail_snippets", [])
    data.problematic_responses = rt.get("problematic_responses", [])

    # Step 5: System prompts
    print("  [5/10] Generating system prompts...")
    sp = _checkpointed(
        cache_dir,
        "05_system_prompts",
        lambda: generate_system_prompts(report, data.ai_feature_modes),
    )
    data.system_prompt_base = sp["system_prompt_base"]
    data.style_a_name = sp.get("style_a_name", "structured")
    data.style_a_suffix = sp["style_a_suffix"]
    data.style_b_name = sp.get("style_b_name", "conversational")
    data.style_b_suffix = sp["style_b_suffix"]
    data.prompt_version_a = f"v1.0-{data.style_a_name}"
    data.prompt_version_b = f"v1.0-{data.style_b_name}"

    # Step 6: Scorers
    print("  [6/10] Generating scorers...")
    sc = _checkpointed(cache_dir, "06_scorers", lambda: generate_scorers(report))
    data.scorers = sc["scorers"]

    # Step 7: Facets
    print("  [7/10] Generating facets...")
    fc = _checkpointed(cache_dir, "07_facets", lambda: generate_facets(report, data.ai_feature_modes))
    data.facets = fc["facets"]

    # Step 8: Schema contexts
    print("  [8/10] Generating schema contexts...")
    ctx = _checkpointed(
        cache_dir,
        "08_schema_contexts",
        lambda: generate_schema_contexts(report, verticals, data.domain_entities, data.entity_properties),
    )
    data.schema_contexts = ctx["schema_contexts"]

    # Step 9: Golden dataset
    print("  [9/10] Generating golden dataset...")
    gd = _checkpointed(
        cache_dir,
        "09_golden_dataset",
        lambda: generate_golden_dataset(
            report, data.ai_feature_modes, data.domain_entities, data.entity_properties, data.schema_contexts
        ),
    )
    data.golden_dataset_rows = gd["golden_dataset_rows"]
    if verbose:
        print(f"    {len(data.golden_dataset_rows)} test cases")

    # Step 10: Multi-turn conversations
    print("  [10/10] Generating multi-turn conversations...")
    mt = _checkpointed(
        cache_dir,
        "10_multi_turn",
        lambda: generate_multi_turn(report, data.ai_feature_modes),
    )
    data.multi_turn_conversations = mt["multi_turn_conversations"]
    data.prior_conversation_snippets = mt["prior_conversation_snippets"]

    # Set span names based on pipeline steps
    steps = report.pipeline_steps
    if len(steps) >= 1:
        data.context_span_name = steps[0].replace("_", " ").title()
    if len(steps) >= 3:
        data.validation_span_name = steps[-1].replace("_", " ").title()
    data.root_span_name = report.ai_product_name.lower().replace(" ", "-") + "-query"

    # Store sample_outputs on the data object for assembly
    data._sample_outputs = rt.get("sample_outputs", {})

    print(f"\n  Synthesis complete: {len(data.golden_dataset_rows)} golden rows, "
          f"{len(data.ai_feature_modes)} modes, {len(data.domain_entities)} entities")

    return data
