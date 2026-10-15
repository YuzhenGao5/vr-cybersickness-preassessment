#!/usr/bin/env python3
"""Run the data-free source test gate and expose failures as CI annotations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TEST_MODULES = (
    "scripts.test_unity_parser",
    "scripts.test_compile_scene_agent_contract",
    "scripts.test_generic_scene_agent_collision",
    "scripts.test_generic_scene_agent_interaction_lock",
    "scripts.test_scene_agent_terminal_acceptance",
)


def github_escape(value: str) -> str:
    """Escape multiline text for a GitHub Actions workflow command."""

    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        return 0

    for test, traceback in (*result.failures, *result.errors):
        message = f"{test.id()}\n{traceback}"
        print(f"::error::{github_escape(message)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
