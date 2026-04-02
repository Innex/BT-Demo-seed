"""CLI entry point: orchestrates research → synthesis → assembly → validation."""

import argparse
import sys
import time
from pathlib import Path

from .research import research_customer, save_research, load_research
from .synthesize import synthesize_all
from .assemble import assemble
from .validate import validate
from .requirements_doc import generate_requirements_doc


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Braintrust demo seed script for any customer's AI product",
        usage="python -m generator CUSTOMER_NAME WEBSITE_URL [options]",
    )
    parser.add_argument("customer", type=str, help="Customer name (e.g., 'Zendesk')")
    parser.add_argument("url", type=str, help="Customer website URL (e.g., 'zendesk.com')")
    parser.add_argument("--context", type=str, default="", help="Additional context about their AI (e.g., 'Focus on Answer Bot')")
    parser.add_argument("--output-dir", type=str, default="generated", help="Output directory (default: generated/)")
    parser.add_argument("--research-file", type=str, default=None, help="Path to pre-existing research JSON (skip web research)")
    parser.add_argument("--verbose", action="store_true", help="Show intermediate output")
    parser.add_argument("--dry-run", action="store_true", help="Generate and validate but don't write files")
    parser.add_argument("--skip-requirements-doc", action="store_true", help="Skip requirements mapping doc generation")

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print(f"{'='*60}")
    print(f"  Braintrust Demo Generator")
    print(f"  Customer: {args.customer}")
    print(f"  Website: {args.url}")
    if args.context:
        print(f"  Context: {args.context}")
    print(f"{'='*60}")

    # Phase 1: Research
    if args.research_file:
        print(f"\n  Loading research from {args.research_file}")
        report = load_research(Path(args.research_file))
    else:
        report = research_customer(args.customer, args.url, args.context, verbose=args.verbose)
        # Save research for reuse
        research_path = output_path / f"research_{args.customer.lower().replace(' ', '_')}.json"
        save_research(report, research_path)

    # Phase 2: Synthesize
    data = synthesize_all(report, verbose=args.verbose)

    if args.dry_run:
        print("\n  [DRY RUN] Skipping file generation")
        elapsed = time.time() - start_time
        print(f"\n  Completed in {elapsed:.1f}s")
        return

    # Phase 3: Assemble
    script_path = assemble(data, output_path)

    # Phase 4: Validate
    is_valid = validate(data, script_path)

    # Phase 5: Requirements doc
    if not args.skip_requirements_doc:
        generate_requirements_doc(report, output_path)

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*60}")
    print(f"  GENERATION COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"\n  Output directory: {output_path}/")
    print(f"  Seed script: {script_path.name}")
    print(f"  Valid: {'YES' if is_valid else 'NO — check errors above'}")
    print(f"\n  Next steps:")
    print(f"    1. Review the generated script: {script_path}")
    print(f"    2. Set environment variables:")
    print(f"       export BRAINTRUST_API_KEY='...'")
    print(f"       export OPENAI_API_KEY='...'")
    print(f"    3. Run: python {script_path} --project \"{data.project_name}\" --count 500")
    print()
