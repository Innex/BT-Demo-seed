"""Phase 4: Validate the generated seed script and data."""

import py_compile
import tempfile
from pathlib import Path

from .models import CustomerData


def validate(data: CustomerData, script_path: Path) -> bool:
    """Run structural validation checks. Returns True if all pass."""

    print(f"\n--- Validating generated output ---")
    errors = []
    warnings = []

    # 1. Syntax check
    try:
        py_compile.compile(str(script_path), doraise=True)
        print("  [PASS] Python syntax valid")
    except py_compile.PyCompileError as e:
        errors.append(f"Syntax error: {e}")
        print(f"  [FAIL] Python syntax: {e}")

    # 2. Data completeness
    checks = [
        ("ai_feature_modes", data.ai_feature_modes, 2),
        ("domain_entities", data.domain_entities, 5),
        ("entity_properties", data.entity_properties, 3),
        ("user_queries", data.user_queries, 1),
        ("responses_structured", data.responses_structured, 2),
        ("responses_conversational", data.responses_conversational, 2),
        ("problematic_responses", data.problematic_responses, 3),
        ("schema_contexts", data.schema_contexts, 2),
        ("golden_dataset_rows", data.golden_dataset_rows, 10),
        ("multi_turn_conversations", data.multi_turn_conversations, 2),
        ("prior_conversation_snippets", data.prior_conversation_snippets, 2),
        ("scorers", data.scorers, 3),
        ("facets", data.facets, 3),
    ]

    for name, value, min_count in checks:
        count = len(value) if isinstance(value, (list, dict)) else 0
        if count >= min_count:
            print(f"  [PASS] {name}: {count} items")
        elif count > 0:
            warnings.append(f"{name}: only {count} items (expected >= {min_count})")
            print(f"  [WARN] {name}: {count} items (expected >= {min_count})")
        else:
            errors.append(f"{name}: empty")
            print(f"  [FAIL] {name}: empty")

    # 3. Cross-reference: user queries cover all feature modes
    for mode in data.ai_feature_modes:
        if mode not in data.user_queries:
            warnings.append(f"No user queries for feature mode '{mode}'")
            print(f"  [WARN] No user queries for mode '{mode}'")
        elif len(data.user_queries[mode]) < 5:
            warnings.append(f"Only {len(data.user_queries[mode])} queries for mode '{mode}'")

    # 4. Schema contexts reference valid entities/properties
    entity_names = {e["name"] for e in data.domain_entities}
    prop_names = {p["name"] for p in data.entity_properties}
    for ctx in data.schema_contexts:
        for ent in ctx.get("entities", []):
            if ent not in entity_names:
                warnings.append(f"Schema context '{ctx['vertical']}' references unknown entity '{ent}'")

    # 5. Golden dataset coverage
    modes_in_golden = {row.get("metadata", {}).get("feature_mode") for row in data.golden_dataset_rows}
    for mode in data.ai_feature_modes:
        if mode not in modes_in_golden:
            warnings.append(f"Golden dataset has no rows for feature mode '{mode}'")
            print(f"  [WARN] Golden dataset missing mode '{mode}'")

    # 6. Scorer slugs exist
    for scorer in data.scorers:
        if not scorer.get("slug"):
            errors.append(f"Scorer '{scorer.get('name')}' missing slug")
        if not scorer.get("prompt"):
            errors.append(f"Scorer '{scorer.get('name')}' missing prompt")

    # 7. System prompt not empty
    if not data.system_prompt_base:
        errors.append("system_prompt_base is empty")
    if not data.style_a_suffix:
        errors.append("style_a_suffix is empty")

    # Summary
    print(f"\n  Validation: {len(errors)} errors, {len(warnings)} warnings")
    if errors:
        print("  ERRORS:")
        for e in errors:
            print(f"    - {e}")
    if warnings and len(warnings) <= 10:
        print("  WARNINGS:")
        for w in warnings:
            print(f"    - {w}")
    elif warnings:
        print(f"  WARNINGS: {len(warnings)} (showing first 5)")
        for w in warnings[:5]:
            print(f"    - {w}")

    return len(errors) == 0
