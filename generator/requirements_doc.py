"""Generate the Braintrust × Customer requirements mapping document."""

import json
import os
from pathlib import Path

from .models import ResearchReport

PROMPTS_DIR = Path(__file__).parent / "prompts"


def generate_requirements_doc(report: ResearchReport, output_path: Path) -> Path:
    """Generate a requirements mapping markdown document."""
    import openai

    print(f"\n--- Generating requirements mapping doc ---")

    prompt_template = (PROMPTS_DIR / "requirements_mapping.txt").read_text()
    prompt = prompt_template.format(
        company_name=report.company_name,
        ai_product_name=report.ai_product_name,
        ai_product_description=report.ai_product_description,
        product_domain=report.product_domain,
        input_type=report.input_type,
        output_type=report.output_type,
        evaluation_dimensions=json.dumps(report.evaluation_dimensions),
        failure_modes=json.dumps(report.failure_modes),
        use_cases=json.dumps(report.use_cases),
        pipeline_steps=json.dumps(report.pipeline_steps),
    )

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    content = resp.choices[0].message.content

    # Strip any code fences the LLM might have added
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("```", 1)[0]

    doc_path = output_path / f"Braintrust × {report.company_name} - Requirements Mapping.md"
    doc_path.write_text(content)
    print(f"  Generated: {doc_path}")

    return doc_path
