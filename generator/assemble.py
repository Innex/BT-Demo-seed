"""Phase 3: Assemble generated data into a runnable seed script."""

import json
import re
import textwrap
from pathlib import Path

from .models import CustomerData

TEMPLATE_PATH = Path(__file__).parent / "template" / "seed_template.py"


def _format_python_literal(obj) -> str:
    """Format a Python object as a readable literal string for embedding in source code."""
    if isinstance(obj, str):
        # Use repr for strings, but prefer double quotes
        r = repr(obj)
        if r.startswith("'") and '"' not in obj:
            r = '"' + r[1:-1] + '"'
        return r

    if isinstance(obj, (int, float, bool, type(None))):
        return repr(obj)

    if isinstance(obj, list):
        if not obj:
            return "[]"
        if all(isinstance(x, str) for x in obj) and len(obj) <= 6:
            items = ", ".join(_format_python_literal(x) for x in obj)
            line = f"[{items}]"
            if len(line) < 100:
                return line
        items = ",\n".join("    " + _format_python_literal(x) for x in obj)
        return f"[\n{items},\n]"

    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = ",\n".join(
            f"    {_format_python_literal(k)}: {_format_python_literal(v)}"
            for k, v in obj.items()
        )
        return f"{{\n{items},\n}}"

    return repr(obj)


def _format_multiline_string(s: str) -> str:
    """Format a long string as a valid Python string literal."""
    # repr() handles all escaping correctly (newlines, quotes, etc.)
    return repr(s)


def assemble(data: CustomerData, output_path: Path) -> Path:
    """Assemble the seed script by substituting placeholders in the template."""

    print(f"\n--- Assembling seed script ---")

    template = TEMPLATE_PATH.read_text()

    slug = data.company_name.lower().replace(" ", "_").replace("-", "_")
    filename = f"seed_{slug}.py"

    docstring = (
        f"{data.company_name} {data.ai_product_name} Demo - Seeds a Braintrust project with:\\n"
        f"  - 2 versioned prompts ({data.style_a_name} vs {data.style_b_name})\\n"
        f"  - {len(data.scorers)} scorers\\n"
        f"  - 1 golden dataset (~{len(data.golden_dataset_rows)} curated test cases)\\n"
        f"  - 500 realistic {data.ai_product_name} traces"
    )

    # Build the replacement map
    replacements = {
        "DOCSTRING": docstring,
        "FILENAME": filename,
        "PROJECT_NAME": data.project_name,
        "COMPANY_NAME": data.company_name,
        "AI_PRODUCT_NAME": data.ai_product_name,
        "STYLE_A_NAME": data.style_a_name,
        "STYLE_B_NAME": data.style_b_name,
        "PROMPT_VERSION_A": data.prompt_version_a,
        "PROMPT_VERSION_B": data.prompt_version_b,
        "ROOT_SPAN_NAME": data.root_span_name,
        "CONTEXT_SPAN_NAME": data.context_span_name,
        "VALIDATION_SPAN_NAME": data.validation_span_name,

        # Data structures (need Python literal formatting)
        "FEATURE_MODES": _format_python_literal(data.ai_feature_modes),
        "FEATURE_MODE_WEIGHTS": _format_python_literal(data.ai_feature_mode_weights),
        "ENTITIES": _format_python_literal(data.domain_entities),
        "ENTITY_PROPERTIES": _format_python_literal(data.entity_properties),
        "CUSTOMER_VERTICALS": _format_python_literal([sc["vertical"] for sc in data.schema_contexts]),
        "USER_QUERIES": _format_python_literal(data.user_queries),
        "SAMPLE_OUTPUTS": _format_python_literal(getattr(data, "_sample_outputs", {})),
        "RESPONSES_STYLE_A": _format_python_literal(data.responses_structured),
        "RESPONSES_STYLE_B": _format_python_literal(data.responses_conversational),
        "INSIGHT_SNIPPETS": _format_python_literal(data.insight_snippets),
        "DETAIL_SNIPPETS": _format_python_literal(data.detail_snippets),
        "PROBLEMATIC_RESPONSES": _format_python_literal(data.problematic_responses),
        "SCHEMA_CONTEXTS": _format_python_literal(data.schema_contexts),
        "SYSTEM_PROMPT_BASE": _format_multiline_string(data.system_prompt_base),
        "STYLE_A_SUFFIX": _format_multiline_string(data.style_a_suffix),
        "STYLE_B_SUFFIX": _format_multiline_string(data.style_b_suffix),
        "SCORERS": _format_python_literal(data.scorers),
        "FACETS": _format_python_literal(data.facets),
        "GOLDEN_DATASET_ROWS": _format_python_literal(data.golden_dataset_rows),
        "PRIOR_CONVERSATION_SNIPPETS": _format_python_literal(data.prior_conversation_snippets),
        "MULTI_TURN_CONVERSATIONS": _format_python_literal(data.multi_turn_conversations),
    }

    # Perform substitutions
    result = template
    for key, value in replacements.items():
        placeholder = "{{" + key + "}}"
        result = result.replace(placeholder, str(value))

    # Check for any remaining placeholders
    remaining = re.findall(r"\{\{[A-Z_]+\}\}", result)
    if remaining:
        unique = set(remaining)
        print(f"  Warning: {len(unique)} unresolved placeholders: {unique}")

    # Write output
    output_file = output_path / filename
    output_file.write_text(result)
    print(f"  Generated: {output_file}")

    return output_file
