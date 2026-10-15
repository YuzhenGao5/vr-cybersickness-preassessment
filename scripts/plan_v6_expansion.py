#!/usr/bin/env python3
"""Plan expansion to at least five local sessions for each selected VR.net game."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


GAMES = [
    "Pirate_Life",
    "Army_Men",
    "City_Parkour",
    "Creature_Feature",
    "Earth_Gym",
    "Kowloon",
    "American_Idol",
    "Horizon_Boxing",
    "Titanic_Simulation",
    "Slash_RPG",
    "Canyon_Runners",
]


def natural_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))


def local_sessions(root: Path, game: str) -> set[str]:
    sessions: set[str] = set()
    candidate_roots = [
        root / "data/raw/vrnet/sampled_csv" / game,
        root / "data/raw/vrnet/v5_sessions" / game,
        root / "data/raw/vrnet/v6_sessions" / game,
    ]
    for candidate in candidate_roots:
        if candidate.is_dir():
            sessions.update(path.parent.name for path in candidate.glob("*/data_file.csv"))
    return sessions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-log", type=Path, default=Path("data/raw/vrnet/main_log.csv"))
    parser.add_argument("--main-file", type=Path, default=Path("data/raw/vrnet/main_file.csv"))
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("results/vrnet_zip_member_inventory.csv.gz"),
    )
    parser.add_argument("--target-per-game", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/session_expansion_plan.md"))
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=Path("results/session_expansion_selection_v6.csv"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    log = pd.read_csv(args.main_log, encoding="utf-8-sig")
    games = pd.read_csv(args.main_file, encoding="utf-8-sig").set_index("game_name")
    log["game_name"] = log["name"].astype(str).str.split("_", n=2).str[2]
    inventory = pd.read_csv(args.inventory) if args.inventory.is_file() else pd.DataFrame()

    report_rows: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    for game in GAMES:
        game_log = log[log["game_name"].eq(game)].copy()
        scene = game_log[(game_log["scene_obj"] == 1) & (game_log["scene_cam"] == 1)].copy()
        labelled = scene[scene["MS_rating"] == 1].copy()
        local = local_sessions(root, game)
        pending_scene = sorted(set(scene["name"].astype(str)) - local, key=natural_key)
        pending_labelled = sorted(set(labelled["name"].astype(str)) - local, key=natural_key)

        # Prefer smaller uncompressed CSVs when a local ZIP member inventory is available.
        size_lookup: dict[str, int] = {}
        if not inventory.empty:
            members = inventory[
                inventory["archive"].eq(f"{game}.zip")
                & inventory["member_path"].astype(str).str.endswith("/data_file.csv")
            ].copy()
            if not members.empty:
                members["session_name"] = members["member_path"].astype(str).str.split("/").str[-2]
                size_lookup = dict(
                    zip(
                        members["session_name"],
                        pd.to_numeric(members["uncompressed_bytes"]).astype("int64"),
                    )
                )
                pending_labelled.sort(key=lambda name: (size_lookup.get(name, 10**30), natural_key(name)))

        needed = max(0, args.target_per_game - len(local))
        chosen = pending_labelled[:needed]
        for session in chosen:
            selections.append(
                {
                    "game_name": game,
                    "session_name": session,
                    "scene_obj": 1,
                    "scene_cam": 1,
                    "MS_rating": 1,
                    "selection_reason": (
                        "smallest known data_file.csv among labelled scene sessions"
                        if session in size_lookup
                        else "natural session order among labelled scene sessions"
                    ),
                    "known_uncompressed_bytes": size_lookup.get(session, pd.NA),
                }
            )
        report_rows.append(
            {
                "game": game,
                "total": int(games.loc[game, "no_of_sessions"]),
                "local": len(local),
                "scene": len(scene),
                "labelled_scene": len(labelled),
                "needed": needed,
                "chosen": chosen,
                "pending_scene": pending_scene,
                "local_names": sorted(local, key=natural_key),
            }
        )

    selection = pd.DataFrame(selections)
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(args.selection_output, index=False, encoding="utf-8-sig")

    lines = [
        "# VR.net v6 session 扩展计划",
        "",
        f"目标：每个游戏至少 {args.target_per_game} 个本地 session。只选择 `scene_obj=1`、`scene_cam=1` 且 `MS_rating=1` 的 session 进入训练扩展。",
        "",
        "| 游戏 | 索引总 session | 本地已有 | scene 合格 | scene+评分合格 | 需新增 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report_rows:
        lines.append(
            f"| {row['game']} | {row['total']} | {row['local']} | {row['scene']} | "
            f"{row['labelled_scene']} | {row['needed']} |"
        )
    lines.extend(["", "## 各游戏明细", ""])
    for row in report_rows:
        lines.extend(
            [
                f"### {row['game']}",
                "",
                f"- 已下载：{', '.join(f'`{name}`' for name in row['local_names']) or '无'}",
                f"- 本轮选择：{', '.join(f'`{name}`' for name in row['chosen']) or '无需下载'}",
                f"- 其余待下载 scene session：{', '.join(f'`{name}`' for name in row['pending_scene'] if name not in row['chosen']) or '无'}",
                "",
            ]
        )
    lines.extend(
        [
            "## 选择说明",
            "",
            "- Pirate_Life 已有 11 个 session，因此跳过下载。",
            "- 其他游戏以最终达到 5 个本地 session 为目标，每个选择 4 个新增 session。",
            "- 若本地已有 ZIP member inventory，则优先选择未压缩 CSV 较小的 session；否则按 session 编号稳定排序。",
            "- scene 合格但没有 `MS_rating` 的 session 不进入本轮训练选择，因为无法生成冻结的二分类标签。",
            f"- 机器可读选择：`{args.selection_output.as_posix()}`，共 {len(selection)} 个新增 session。",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Planned {len(selection)} new sessions across {selection['game_name'].nunique() if not selection.empty else 0} games")
    print(f"Wrote {args.output} and {args.selection_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
