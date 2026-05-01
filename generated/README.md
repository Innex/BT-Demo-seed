# Generated outputs

This directory is the destination for files written by `python -m generator …`. Its contents are gitignored — each run produces customer-specific artifacts that don't belong in version control.

After a successful generator run, you'll find:

- `seed_<customer>.py` — the runnable seed script
- `Braintrust × <Customer> - Requirements Mapping.md` — the requirements doc
- `research_<customer>.json` — the research report (cached for re-runs via `--research-file`)
- `.cache/<customer>/` — per-step LLM caches that make re-runs free
