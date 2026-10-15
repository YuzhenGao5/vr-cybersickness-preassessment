#!/usr/bin/env python3
"""Optional P1 RAG index builder for Unity and cybersickness documents.

This command is intentionally separate from the P0 parser.  If llama-index is
not installed, it records a structured ``not_run`` status and exits without
creating a pretend index.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from unity_parser import ensure_allowed_artifact_path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_ROOT = REPO_ROOT / "knowledge"
DEFAULT_STATUS = REPO_ROOT / "results" / "knowledge_base_status.json"


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), **payload}
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_directories(root: Path) -> list[Path]:
    return [
        root / "unity_docs",
        root / "project_scripts",
        root / "cybersickness_rules",
    ]


def build_knowledge_base(root: Path, status_path: Path, embedding_model: Path | None) -> int:
    root = ensure_allowed_artifact_path(root)
    status_path = ensure_allowed_artifact_path(status_path)
    if importlib.util.find_spec("llama_index") is None:
        write_status(status_path, {
            "status": "not_run",
            "reason": "llama-index is not installed; P1 indexing remains optional and was not fabricated.",
            "knowledge_root": str(root),
        })
        print("llama-index is unavailable; recorded status and stopped.", file=sys.stderr)
        return 2

    if embedding_model is None or not embedding_model.exists():
        write_status(status_path, {
            "status": "not_run",
            "reason": "A local embedding model path is required; remote/default embedding services are disabled.",
            "knowledge_root": str(root),
            "embedding_model": str(embedding_model) if embedding_model else None,
        })
        print("A local --embedding-model path is required; recorded status and stopped.", file=sys.stderr)
        return 2

    # Imports remain deferred so the P0 environment does not require RAG packages.
    try:
        from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as exc:
        write_status(status_path, {
            "status": "not_run",
            "reason": f"Required local LlamaIndex components are unavailable: {exc}",
            "knowledge_root": str(root),
        })
        print("Local LlamaIndex embedding components are unavailable; recorded status and stopped.", file=sys.stderr)
        return 2

    sources = [path for path in source_directories(root) if path.exists()]
    documents = []
    source_counts: dict[str, int] = {}
    for source in sources:
        loaded = SimpleDirectoryReader(str(source), recursive=True).load_data()
        documents.extend(loaded)
        source_counts[str(source)] = len(loaded)
    if not documents:
        write_status(status_path, {
            "status": "not_run",
            "reason": "No source documents were found; no empty index was created.",
            "knowledge_root": str(root),
            "searched_directories": [str(path) for path in source_directories(root)],
        })
        print("No knowledge documents found; recorded status and stopped.", file=sys.stderr)
        return 2

    output = root / "index"
    if output.exists() and any(output.iterdir()):
        write_status(status_path, {
            "status": "not_run",
            "reason": "The target index directory is not empty; existing index files were preserved.",
            "knowledge_root": str(root),
            "index_path": str(output),
        })
        print("Existing index is non-empty; preserved it and stopped.", file=sys.stderr)
        return 2
    try:
        Settings.llm = None
        Settings.embed_model = HuggingFaceEmbedding(model_name=str(embedding_model.resolve()))
        index = VectorStoreIndex.from_documents(documents)
        output.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(output))
        write_status(status_path, {
            "status": "success",
            "knowledge_root": str(root),
            "index_path": str(output),
            "embedding_backend": "local_huggingface",
            "embedding_model": str(embedding_model.resolve()),
            "document_count": len(documents),
            "source_counts": source_counts,
        })
    except Exception as exc:
        write_status(status_path, {
            "status": "failed",
            "reason": f"Local index construction failed: {type(exc).__name__}: {exc}",
            "knowledge_root": str(root),
            "embedding_model": str(embedding_model.resolve()),
        })
        print(f"Local index construction failed: {exc}", file=sys.stderr)
        return 1
    print(f"Knowledge index created from {len(documents)} documents: {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-root", default=str(DEFAULT_KNOWLEDGE_ROOT))
    parser.add_argument("--status", default=str(DEFAULT_STATUS))
    parser.add_argument(
        "--embedding-model",
        help="Existing local Hugging Face embedding model directory; remote defaults are disabled",
    )
    args = parser.parse_args(argv)
    try:
        return build_knowledge_base(
            Path(args.knowledge_root).expanduser().resolve(),
            Path(args.status).expanduser().resolve(),
            Path(args.embedding_model).expanduser().resolve() if args.embedding_model else None,
        )
    except Exception as exc:
        print(f"Knowledge-base build failed before indexing: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
