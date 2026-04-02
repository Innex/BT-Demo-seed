"""Data models that flow between generator phases."""

from dataclasses import dataclass, field


@dataclass
class ResearchReport:
    """Output of Phase 1: structured research about a customer's AI product."""

    company_name: str
    website_url: str
    additional_context: str = ""

    # Discovered AI product info
    ai_product_name: str = ""
    ai_product_description: str = ""
    product_domain: str = ""  # e.g., "analytics", "customer-support", "code-generation"

    # What the AI does
    input_type: str = ""  # What users provide (NL query, code, document, etc.)
    output_type: str = ""  # What the AI produces (report, code, answer, etc.)
    pipeline_steps: list[str] = field(default_factory=list)  # e.g., ["context retrieval", "generation", "validation"]

    # Domain details
    domain_entities: list[str] = field(default_factory=list)  # e.g., events, tickets, files
    entity_properties: list[str] = field(default_factory=list)  # e.g., plan_type, priority
    user_personas: list[str] = field(default_factory=list)  # Who uses this AI feature
    customer_verticals: list[str] = field(default_factory=list)  # e.g., b2b_saas, ecommerce

    # AI evaluation dimensions
    evaluation_dimensions: list[str] = field(default_factory=list)  # What "correctness" means
    failure_modes: list[str] = field(default_factory=list)  # What can go wrong

    # Use cases discovered
    use_cases: list[dict] = field(default_factory=list)

    # Raw research for reference
    raw_research_notes: str = ""


@dataclass
class CustomerData:
    """Output of Phase 2: all 13 domain-specific data structures for the seed script."""

    # 1. AI feature modes (equivalent of Mixpanel's report types)
    ai_feature_modes: list[str] = field(default_factory=list)
    ai_feature_mode_weights: list[float] = field(default_factory=list)

    # 2. Domain entities (equivalent of Mixpanel's events)
    domain_entities: list[dict] = field(default_factory=list)
    # Each: {"name": "...", "category": "..."}

    # 3. Entity properties (equivalent of Mixpanel's event properties)
    entity_properties: list[dict] = field(default_factory=list)
    # Each: {"name": "...", "type": "string"|"number", "values": [...] or None}

    # 4. User queries organized by feature mode
    user_queries: dict[str, list[str]] = field(default_factory=dict)

    # 5. Response templates (two styles)
    responses_structured: list[str] = field(default_factory=list)
    responses_conversational: list[str] = field(default_factory=list)
    insight_snippets: list[str] = field(default_factory=list)
    detail_snippets: list[str] = field(default_factory=list)

    # 6. Problematic responses (intentionally bad for HITL demo)
    problematic_responses: list[str] = field(default_factory=list)

    # 7. Schema contexts (customer-vertical-specific subsets)
    schema_contexts: list[dict] = field(default_factory=list)
    # Each: {"vertical": "...", "entities": [...], "properties": [...]}

    # 8. Golden dataset rows
    golden_dataset_rows: list[dict] = field(default_factory=list)
    # Each: {"input": "...", "expected": "...", "metadata": {...}, "tags": [...]}

    # 9. Multi-turn conversation templates
    multi_turn_conversations: list[dict] = field(default_factory=list)
    # Each: {"feature_mode": "...", "vertical": "...", "turns": [...]}

    # 10. Prior conversation snippets
    prior_conversation_snippets: list[list[dict]] = field(default_factory=list)

    # 11. System prompts
    system_prompt_base: str = ""
    style_a_name: str = "structured"
    style_a_suffix: str = ""
    style_b_name: str = "conversational"
    style_b_suffix: str = ""

    # 12. Scorer definitions (3 scorers)
    scorers: list[dict] = field(default_factory=list)
    # Each: {"name": "...", "slug": "...", "description": "...", "prompt": "...", "choices": {...}}

    # 13. Facet definitions (4 facets)
    facets: list[dict] = field(default_factory=list)
    # Each: {"name": "...", "slug": "...", "description": "...", "prompt": "..."}

    # Metadata
    project_name: str = ""
    ai_product_name: str = ""
    company_name: str = ""
    product_domain: str = ""

    # Pipeline span names (generalized from Schema lookup / Query validation)
    context_span_name: str = "Context retrieval"
    validation_span_name: str = "Output validation"
    root_span_name: str = "ai-query"

    # Prompt version labels
    prompt_version_a: str = "v1.0-structured"
    prompt_version_b: str = "v1.0-conversational"
