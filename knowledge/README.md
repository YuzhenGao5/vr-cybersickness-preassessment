# Unity semantic parser knowledge sources

P1 RAG is deliberately optional. Place licensed, attributable text sources in:

- `unity_docs/` for Unity documentation;
- `project_scripts/` for project-specific code excerpts that may be indexed;
- `cybersickness_rules/` for reviewed risk definitions and design guidance.

Run `python scripts/build_knowledge_base.py --embedding-model <local-model-directory>`
only after installing compatible `llama-index` and Hugging Face embedding
packages and reviewing the source licences. Remote/default embedding services
are disabled so project source is not uploaded implicitly. The script records
`results/knowledge_base_status.json` and does not create an empty or fabricated
index when the dependency or source documents are missing.
