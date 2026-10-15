#!/usr/bin/env python3
"""Interactively refine Unity semantic parsing with natural-language hints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from unity_parser import (
    DEFAULT_PROMPT_PATH,
    DEFAULT_STATUS_PATH,
    UnityProjectParser,
    ensure_allowed_artifact_path,
    write_json,
    write_status,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "unity_parse_result.json"


class LLMUnavailable(RuntimeError):
    """The optional LLM backend cannot be started in this environment."""


def make_llm_client(use_llm: bool) -> Any:
    if not use_llm:
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        raise LLMUnavailable("OPENAI_API_KEY is not configured")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMUnavailable("the openai Python package is not installed") from exc
    return OpenAI()


def interactive_parse(
    project_path: str | Path,
    *,
    use_llm: bool = False,
    output_path: str | Path = DEFAULT_OUTPUT,
    prompt_path: str | Path = DEFAULT_PROMPT_PATH,
    model: str = "gpt-4o-mini",
    initial_instructions: list[str] | None = None,
) -> dict[str, Any]:
    client = make_llm_client(use_llm)
    parser = UnityProjectParser(project_path, client, prompt_path=prompt_path, model=model)

    print("=" * 60)
    print("Unity 项目语义解析专家")
    print("=" * 60)
    print(f"项目路径: {Path(project_path).resolve()}")
    print(f"解析后端: {'LLM' if use_llm else '本地规则'}")
    print("可输入多条指令；输入 done 结束并执行。")
    print("示例：重点关注快速移动区域 / 忽略装饰物 / 风险阈值调低")

    instructions = [item.strip() for item in (initial_instructions or []) if item.strip()]
    while True:
        try:
            user_input = input("\n你的指令: ").strip()
        except EOFError:
            break
        if user_input.casefold() == "done":
            break
        if user_input:
            instructions.append(user_input)
            print(f"[记录] {user_input}")

    result = parser.parse("\n".join(instructions), use_llm=use_llm)
    output = ensure_allowed_artifact_path(Path(output_path))
    if output == DEFAULT_STATUS_PATH.resolve():
        raise RuntimeError("semantic output and status paths must be different")
    preexisting_output = output.exists()
    write_json(output, result)
    write_status({
        "status": "partial" if result["analysis_scope"]["parse_warnings"] else "success",
        "backend": result["analysis_scope"]["backend"],
        "schema_version": result["schema_version"],
        "schema_valid": True,
        "output": str(output),
        "file_count": len(result["analysis_scope"]["files_received"]),
        "synthetic_input": result["analysis_scope"]["synthetic_input"],
        "output_written": True,
        "preexisting_output_detected": preexisting_output,
    }, DEFAULT_STATUS_PATH)
    print("\n" + json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n已保存: {output}")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", help="Unity project or evidence-directory path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Structured JSON output")
    parser.add_argument("--instruction", action="append", default=[], help="Preload an instruction; repeatable")
    parser.add_argument("--use-llm", action="store_true", help="Explicitly request the OpenAI backend")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT_PATH))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        output = ensure_allowed_artifact_path(Path(args.output))
    except Exception as exc:
        print(f"解析未开始：{exc}", file=sys.stderr)
        return 1
    try:
        interactive_parse(
            args.project,
            use_llm=args.use_llm,
            output_path=args.output,
            prompt_path=args.prompt,
            model=args.model,
            initial_instructions=args.instruction,
        )
    except LLMUnavailable as exc:
        write_status({
            "status": "not_run",
            "backend": "llm",
            "reason": str(exc),
            "output": str(output),
            "output_written": False,
            "preexisting_output_detected": output.exists(),
        }, DEFAULT_STATUS_PATH)
        print(f"解析未完成：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        write_status({
            "status": "failed",
            "backend": "llm" if args.use_llm else "rules",
            "reason": str(exc),
            "output": str(output),
            "output_written": False,
            "preexisting_output_detected": output.exists(),
        }, DEFAULT_STATUS_PATH)
        print(f"解析失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
