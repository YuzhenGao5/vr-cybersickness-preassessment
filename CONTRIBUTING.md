# Continuing the research prototype

Start with `docs/PROJECT_REPORT_2026-10-15.md` and `docs/AI_HANDOFF.md`. The project has no project-wide
open-source license yet, so contact the repository owner before reusing or redistributing original project code.

For a change intended to be merged:

1. Keep parser evidence, learned decisions, engineered safety, and risk scores as separate concepts.
2. Do not grant interaction authority from an LLM response or from code reflection alone.
3. Add a negative test for every safety or fail-closed behavior.
4. Preserve data provenance (`rule_bootstrap`, `model_rollout`, `public_human`, `unity_human`).
5. Split evaluation by participant and project/scene family as appropriate; never tune on a held-out fold.
6. Keep raw data, participant/session tables, OOF rows, checkpoints, logs, builds and EXEs out of Git.
7. Run `python scripts/check_release_package.py` and the test commands in `docs/REPRODUCIBILITY.md`.
8. Add a new dated aggregate manifest when claims or metrics change; do not silently replace old evidence.

A pull request or handoff should state:

- what claim/capability changed;
- source and dataset versions;
- train/validation/test grouping and leakage controls;
- commands and environment;
- positive and negative evidence;
- failures and limitations;
- affected runtime/model hashes.
