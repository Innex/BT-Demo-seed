"""Phase 1: Deep research about a customer's AI product via web search + LLM."""

import json
import os
from pathlib import Path

from .models import ResearchReport

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _call_llm(prompt: str, model: str = "gpt-4.1-mini") -> str:
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def _web_search(query: str) -> str:
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.responses.create(
        model="gpt-4.1-mini",
        tools=[{"type": "web_search_preview"}],
        input=query,
    )
    # Extract text from response output items
    texts = []
    for item in resp.output:
        if hasattr(item, "text"):
            texts.append(item.text)
        elif hasattr(item, "content"):
            for block in item.content:
                if hasattr(block, "text"):
                    texts.append(block.text)
    return "\n".join(texts) if texts else str(resp.output)


def research_customer(name: str, url: str, context: str = "", verbose: bool = False) -> ResearchReport:
    """Research a customer's AI product by querying the web and synthesizing results."""

    print(f"\n--- Researching {name} ({url}) ---")

    # Run 3 targeted research queries
    queries = [
        f"What AI features does {name} ({url}) offer? What is their main AI product? How does it work technically?",
        f"{name} AI product architecture: what inputs does it take, what outputs does it produce, what LLM provider do they use? Site: {url}",
        f"{name} AI evaluation challenges: what can go wrong with their AI, what are the failure modes, how do they measure quality? Site: {url}",
    ]
    if context:
        queries[0] = f"{context}. What AI features does {name} ({url}) offer? How does it work?"

    research_notes = []
    for i, query in enumerate(queries, 1):
        print(f"  Research query {i}/{len(queries)}...")
        try:
            result = _web_search(query)
            research_notes.append(f"--- Query {i}: {query} ---\n{result}")
            if verbose:
                print(f"    Got {len(result)} chars")
        except Exception as e:
            print(f"    Warning: web search failed: {e}")
            # Fall back to LLM knowledge
            result = _call_llm(f"Tell me everything you know about {name}'s AI features and products. {context}")
            research_notes.append(f"--- Fallback LLM knowledge ---\n{result}")

    raw_notes = "\n\n".join(research_notes)

    # Synthesize structured report from raw research
    print("  Synthesizing research report...")
    synthesis_prompt = (PROMPTS_DIR / "research_synthesis.txt").read_text().format(
        company_name=name,
        website_url=url,
        additional_context=context,
        research_notes=raw_notes,
    )

    synthesis_result = _call_llm(synthesis_prompt, model="gpt-4.1-mini")

    # Parse JSON from response (handle markdown code fences)
    json_str = synthesis_result.strip()
    if json_str.startswith("```"):
        json_str = json_str.split("\n", 1)[1]
        json_str = json_str.rsplit("```", 1)[0]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        print(f"  Warning: Failed to parse synthesis JSON, retrying...")
        retry_prompt = f"The following response was not valid JSON. Fix it and return ONLY valid JSON:\n\n{synthesis_result}"
        retry_result = _call_llm(retry_prompt)
        json_str = retry_result.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("\n", 1)[1]
            json_str = json_str.rsplit("```", 1)[0]
        data = json.loads(json_str)

    report = ResearchReport(
        company_name=name,
        website_url=url,
        additional_context=context,
        ai_product_name=data.get("ai_product_name", f"{name} AI"),
        ai_product_description=data.get("ai_product_description", ""),
        product_domain=data.get("product_domain", "other"),
        input_type=data.get("input_type", "natural language"),
        output_type=data.get("output_type", "AI-generated response"),
        pipeline_steps=data.get("pipeline_steps", ["context retrieval", "generation", "validation"]),
        domain_entities=data.get("domain_entities", []),
        entity_properties=data.get("entity_properties", []),
        user_personas=data.get("user_personas", []),
        customer_verticals=data.get("customer_verticals", []),
        evaluation_dimensions=data.get("evaluation_dimensions", []),
        failure_modes=data.get("failure_modes", []),
        use_cases=data.get("use_cases", []),
        raw_research_notes=raw_notes,
    )

    print(f"  Discovered: {report.ai_product_name} ({report.product_domain})")
    print(f"  Input: {report.input_type} → Output: {report.output_type}")
    print(f"  {len(report.domain_entities)} entities, {len(report.evaluation_dimensions)} eval dimensions")

    return report


def save_research(report: ResearchReport, path: Path):
    """Save research report as JSON for reuse with --research-file."""
    import dataclasses
    path.write_text(json.dumps(dataclasses.asdict(report), indent=2))
    print(f"  Research saved to {path}")


def load_research(path: Path) -> ResearchReport:
    """Load a previously saved research report."""
    data = json.loads(path.read_text())
    return ResearchReport(**data)
