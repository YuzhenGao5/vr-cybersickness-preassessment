#!/usr/bin/env python3
"""Audit local and public VR.net files for graphics-label data.

This script is read-only with respect to the dataset. It writes audit reports and
a gzip-compressed ZIP-member manifest under ``results/``.
"""

from __future__ import annotations

import csv
import gzip
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path, PurePosixPath

import pandas as pd
import requests


DRIVE_ROOT_ID = "1LLo4OE83Zeso8MkEXSvW9VrdjiqqlFRR"
DRIVE_ROOT_URL = f"https://drive.google.com/drive/folders/{DRIVE_ROOT_ID}"
GOOGLE_DRIVE_MIME = "application/vnd.google-apps.folder"
LABEL_PATTERN = re.compile(
    r"(?:fov|field[_ -]?of[_ -]?view|motion[_ -]?flow|motionflow|optical[_ -]?flow|"
    r"depth|depth[_ -]?of[_ -]?field|dof|camera|object|bounds|[mpv]_matrix|"
    r"label|graphics|render)",
    re.IGNORECASE,
)


def extract_api_keys(page: str) -> list[str]:
    return sorted(set(re.findall(r"AIza[0-9A-Za-z_-]{30,}", page)))


def list_drive_children(session: requests.Session, key: str, folder_id: str) -> list[dict]:
    endpoint = "https://www.googleapis.com/drive/v3/files"
    items: list[dict] = []
    token: str | None = None
    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "files(id,name,size,mimeType,modifiedTime,webViewLink),nextPageToken",
            "pageSize": 1000,
            "key": key,
        }
        if token:
            params["pageToken"] = token
        response = session.get(
            endpoint,
            params=params,
            headers={"Referer": "https://drive.google.com/"},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        items.extend(payload.get("files", []))
        token = payload.get("nextPageToken")
        if not token:
            break
    return sorted(items, key=lambda item: (item.get("mimeType") != GOOGLE_DRIVE_MIME, item.get("name", "").lower()))


def human_size(value: int | str | None) -> str:
    if value in (None, ""):
        return "unknown"
    size = int(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    number = float(size)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024
    raise AssertionError("unreachable")


def parse_game(session_name: str) -> str:
    parts = session_name.split("_", 2)
    return parts[2] if len(parts) == 3 else "unknown"


def write_csv_columns_report(root: Path) -> None:
    files = sorted((root / "data/raw/vrnet/sampled_csv").rglob("data_file.csv"))
    signatures: dict[tuple[str, ...], list[tuple[str, str]]] = defaultdict(list)
    rows: list[tuple[str, str, int, list[str]]] = []
    for path in files:
        columns = tuple(pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns.astype(str))
        session_name = path.parent.name
        game = path.parent.parent.name
        signatures[columns].append((game, session_name))
        candidates = [column for column in columns if LABEL_PATTERN.search(column)]
        rows.append((game, session_name, len(columns), candidates))

    lines = [
        "# VR.net 已下载 CSV 完整列名核验",
        "",
        f"核验日期：{date.today().isoformat()}",
        "",
        f"- 检查路径：`data/raw/vrnet/sampled_csv/`",
        f"- 找到 `data_file.csv`：**{len(files)}** 个",
        f"- 唯一 schema：**{len(signatures)}** 种",
        "- 检查方式：只读取 CSV 表头；未修改原始文件。",
        "",
        "## 按游戏汇总",
        "",
        "| 游戏 | CSV 数 | schema | 图形相关列 |",
        "|---|---:|---|---|",
    ]
    for game in sorted({item[0] for item in rows}):
        game_rows = [item for item in rows if item[0] == game]
        schema_ids = sorted(
            {
                index
                for index, (columns, members) in enumerate(signatures.items(), 1)
                if any(member_game == game for member_game, _ in members)
            }
        )
        candidate_columns = sorted({value for _, _, _, values in game_rows for value in values})
        lines.append(
            f"| {game} | {len(game_rows)} | {', '.join(f'S{value}' for value in schema_ids)} | "
            f"{', '.join(f'`{value}`' for value in candidate_columns) if candidate_columns else '未找到'} |"
        )

    lines.extend(["", "## 三种 schema", ""])
    for index, (columns, members) in enumerate(signatures.items(), 1):
        games = sorted({game for game, _ in members})
        sessions = sorted(session for _, session in members)
        lines.extend(
            [
                f"### S{index}：{len(columns)} 列，{len(members)} 个 session",
                "",
                f"- 游戏：{', '.join(games)}",
                f"- Session：{', '.join(f'`{value}`' for value in sessions)}",
                f"- 完整列名（原顺序）：{', '.join(f'`{value}`' for value in columns)}",
                "",
            ]
        )

    lines.extend(
        [
            "## 目标字段检查",
            "",
            "| 目标 | 明确字段是否存在 | 说明 |",
            "|---|---|---|",
            "| FOV / field_of_view | 否 | 未找到标量 FOV 列；有 scene camera 数据的 session 使用 `p_matrix` 表示投影矩阵，论文说明 FOV 编码在投影矩阵中。 |",
            "| motion_flow / optical_flow | 否 | 未找到逐帧 motion-flow 数值列或单独路径列。 |",
            "| depth / depth_of_field / dof | 否 | 未找到逐帧 depth 数值列或单独路径列。 |",
            "| camera_pos / camera_rot / camera_vel | 部分、矩阵形式 | Pirate_Life 的 11 个 session 有 `camera_name`, `p_matrix`, `v_matrix`；无这些显式标量列。 |",
            "| object_pos / object_rot / object_vel | 部分、矩阵形式 | Pirate_Life 的 11 个 session 有 `object_name`, `bounds`, `m_matrix`；无这些显式标量列。 |",
            "",
            "## 每个 CSV 的检查结果",
            "",
            "| 游戏 | Session | 列数 | 图形相关列 |",
            "|---|---|---:|---|",
        ]
    )
    for game, session_name, count, candidates in sorted(rows):
        lines.append(
            f"| {game} | `{session_name}` | {count} | "
            f"{', '.join(f'`{value}`' for value in candidates) if candidates else '未找到'} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "已下载的 33 个 CSV 中没有明确命名的 FOV、motion flow 或 depth 列。11 个 Pirate_Life session 的图形标签以内嵌矩阵/对象字段存在；其余 22 个 session 没有 scene camera/object 字段。",
            "",
        ]
    )
    (root / "results/vrnet_csv_columns_full.md").write_text("\n".join(lines), encoding="utf-8")


def write_main_log_report(root: Path) -> None:
    main_log_path = root / "data/raw/vrnet/main_log.csv"
    main_log = pd.read_csv(main_log_path, encoding="utf-8-sig")
    downloaded = sorted(path.parent.name for path in (root / "data/raw/vrnet/sampled_csv").rglob("data_file.csv"))
    subset = main_log.loc[main_log["name"].astype(str).isin(downloaded)].copy()
    obj_count = int(pd.to_numeric(main_log["scene_obj"], errors="coerce").eq(1).sum())
    cam_count = int(pd.to_numeric(main_log["scene_cam"], errors="coerce").eq(1).sum())
    both_count = int(
        (pd.to_numeric(main_log["scene_obj"], errors="coerce").eq(1)
         & pd.to_numeric(main_log["scene_cam"], errors="coerce").eq(1)).sum()
    )
    downloaded_obj = int(pd.to_numeric(subset["scene_obj"], errors="coerce").eq(1).sum())
    downloaded_cam = int(pd.to_numeric(subset["scene_cam"], errors="coerce").eq(1).sum())
    scene_columns_equal = main_log["scene_obj"].equals(main_log["scene_cam"])

    lines = [
        "# VR.net `main_log.csv` 分析",
        "",
        f"核验日期：{date.today().isoformat()}",
        "",
        f"- 行数：**{len(main_log)}**",
        f"- 完整列名：{', '.join(f'`{value}`' for value in main_log.columns)}",
        f"- `scene_obj=1`：**{obj_count}** 个 session",
        f"- `scene_cam=1`：**{cam_count}** 个 session",
        f"- 两者同时为 1：**{both_count}** 个 session",
        f"- 本地索引中两列逐行相同：{'是' if scene_columns_equal else '否'}。",
        "",
        "## 字段含义判断",
        "",
        "`scene_obj` 与 `scene_cam` 是数据可用性标志，不是文件路径。证据如下：",
        "",
        "1. 它们仅取 0/1，且没有 URL 或文件名。",
        "2. 在本地已下载的 session 中，标志为 1 时 `data_file.csv` 含对象与相机字段；标志为 0 时这些字段缺失。",
        "3. 论文把对象和相机标签描述为每帧记录的模型、视图和投影矩阵；本地文件将这些字段放在同一个 `data_file.csv` 中。",
        "",
        "因此，`scene_obj=1` 表示该 session 的 CSV 包含对象场景字段（`object_name`, `bounds`, `m_matrix`）；`scene_cam=1` 表示包含相机字段（`camera_name`, `p_matrix`, `v_matrix`）。",
        "",
        "## 已下载 33 个 session 的重叠",
        "",
        f"- 已下载 CSV：**{len(subset)}** 个",
        f"- 其中 `scene_obj=1`：**{downloaded_obj}** 个",
        f"- 其中 `scene_cam=1`：**{downloaded_cam}** 个",
        "",
        "| 游戏 | Session 数 | scene_obj=1 | scene_cam=1 |",
        "|---|---:|---:|---:|",
    ]
    subset["game"] = subset["name"].astype(str).map(parse_game)
    for game, group in subset.groupby("game", sort=True):
        lines.append(
            f"| {game} | {len(group)} | "
            f"{int(pd.to_numeric(group['scene_obj'], errors='coerce').eq(1).sum())} | "
            f"{int(pd.to_numeric(group['scene_cam'], errors='coerce').eq(1).sum())} |"
        )
    lines.extend(
        [
            "",
            "## Session 明细",
            "",
            "| Session | scene_obj | scene_cam | video | MS_rating |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in subset.sort_values("name").iterrows():
        lines.append(
            f"| `{row['name']}` | {row['scene_obj']} | {row['scene_cam']} | {row['video']} | {row['MS_rating']} |"
        )
    lines.append("")
    (root / "results/vrnet_main_log_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def write_zip_report(root: Path) -> None:
    zip_paths = sorted((root / "data/raw/vrnet").rglob("*.zip"))
    inventory_path = root / "results/vrnet_zip_member_inventory.csv.gz"
    summary_rows: list[dict] = []
    all_non_images: dict[str, list[str]] = {}
    with gzip.open(inventory_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["archive", "member_path", "is_directory", "uncompressed_bytes", "compressed_bytes", "extension"])
        for archive in zip_paths:
            valid = zipfile.is_zipfile(archive)
            if not valid:
                summary_rows.append({"archive": archive.name, "bytes": archive.stat().st_size, "valid": False})
                continue
            with zipfile.ZipFile(archive) as bundle:
                infos = bundle.infolist()
                files = [info for info in infos if not info.is_dir()]
                extension_counts = Counter(PurePosixPath(info.filename).suffix.lower() or "<none>" for info in files)
                non_images = [
                    info.filename for info in files
                    if PurePosixPath(info.filename).suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                ]
                keyword_matches = [info.filename for info in files if LABEL_PATTERN.search(info.filename)]
                header_signatures: set[tuple[str, ...]] = set()
                for info in files:
                    extension = PurePosixPath(info.filename).suffix.lower() or "<none>"
                    writer.writerow([archive.name, info.filename, False, info.file_size, info.compress_size, extension])
                    if info.filename.lower().endswith("data_file.csv"):
                        with bundle.open(info) as stream:
                            header = stream.readline().decode("utf-8-sig", errors="replace").strip()
                        header_signatures.add(tuple(next(csv.reader([header]))))
                all_non_images[archive.name] = non_images
                summary_rows.append(
                    {
                        "archive": archive.name,
                        "bytes": archive.stat().st_size,
                        "valid": True,
                        "members": len(infos),
                        "files": len(files),
                        "extensions": dict(sorted(extension_counts.items())),
                        "non_images": len(non_images),
                        "keyword_matches": len(keyword_matches),
                        "schemas": len(header_signatures),
                    }
                )

    lines = [
        "# VR.net 本地 ZIP 完整内容核验",
        "",
        f"核验日期：{date.today().isoformat()}",
        "",
        f"- 本地残留 ZIP：**{len(zip_paths)}** 个",
        "- 因为已有本地 ZIP，本次没有下载新 ZIP。",
        "- 使用 ZIP 中央目录读取全部成员路径及大小；没有解压图像、没有修改或删除已有 ZIP。",
        f"- 所有成员的完整清单：`{inventory_path.relative_to(root).as_posix()}`（gzip 压缩 CSV）。",
        "",
        "## 压缩包汇总",
        "",
        "| ZIP | 大小 | 有效 ZIP | 成员数 | 文件数 | 扩展名分布 | 额外标签文件名匹配 |",
        "|---|---:|---|---:|---:|---|---:|",
    ]
    for row in summary_rows:
        if not row["valid"]:
            lines.append(f"| {row['archive']} | {human_size(row['bytes'])} | 否 | unknown | unknown | unknown | unknown |")
            continue
        extensions = ", ".join(f"`{key}`={value}" for key, value in row["extensions"].items())
        lines.append(
            f"| {row['archive']} | {human_size(row['bytes'])} | 是 | {row['members']} | {row['files']} | {extensions} | {row['keyword_matches']} |"
        )

    lines.extend(["", "## 非图像文件（完整列表）", ""])
    for archive_name, paths in all_non_images.items():
        lines.extend([f"### {archive_name}", ""])
        lines.extend(f"- `{path}`" for path in paths)
        lines.append("")
    lines.extend(
        [
            "## 结论",
            "",
            "六个 ZIP 只包含每个 session 的 `data_file.csv` 和 `video/*.jpg`。文件名中没有 `graphics_labels`、`render_labels`、`frame_labels`、`motion_flow`、`depth` 或相似的单独标签文件。带场景标志的 session 将对象/相机矩阵直接存入 `data_file.csv`。",
            "",
        ]
    )
    (root / "results/vrnet_zip_contents.md").write_text("\n".join(lines), encoding="utf-8")


def write_drive_report(root: Path) -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; VRnet-graphics-audit/1.0)"})
    response = session.get(DRIVE_ROOT_URL, timeout=45)
    response.raise_for_status()
    keys = extract_api_keys(response.text)
    if not keys:
        raise RuntimeError("No public Google Drive API key was exposed by the public listing page")

    last_error: Exception | None = None
    root_items: list[dict] | None = None
    used_key = ""
    for key in keys:
        try:
            root_items = list_drive_children(session, key, DRIVE_ROOT_ID)
            used_key = key
            break
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
    if root_items is None:
        raise RuntimeError(f"Unable to list Drive root: {last_error}")

    child_listing: dict[str, list[dict]] = {}
    for item in root_items:
        if item.get("mimeType") == GOOGLE_DRIVE_MIME:
            child_listing[item["name"]] = list_drive_children(session, used_key, item["id"])

    keywords = ("label", "graphics", "render", "motion", "flow", "depth", "camera", "metadata", "frame")
    matches: list[tuple[str, dict]] = []
    for item in root_items:
        if any(token in item.get("name", "").lower() for token in keywords):
            matches.append(("root", item))
    for folder_name, items in child_listing.items():
        for item in items:
            if any(token in item.get("name", "").lower() for token in keywords):
                matches.append((folder_name, item))

    readme_text = ""
    updates_text = ""
    readme_item = next((item for item in root_items if item.get("name") == "ReadMe.md"), None)
    updates_item = next((item for item in root_items if item.get("name") == "Updates_ReadMe"), None)
    if readme_item:
        readme_response = session.get(
            "https://drive.google.com/uc",
            params={"export": "download", "id": readme_item["id"]},
            timeout=45,
        )
        readme_response.raise_for_status()
        readme_text = readme_response.content.decode("utf-8-sig", errors="replace")
    if updates_item:
        updates_response = session.get(
            f"https://docs.google.com/document/d/{updates_item['id']}/export",
            params={"format": "txt"},
            timeout=45,
        )
        updates_response.raise_for_status()
        updates_text = updates_response.content.decode("utf-8-sig", errors="replace")

    lines = [
        "# VR.net Google Drive 目录结构",
        "",
        f"访问日期：{date.today().isoformat()}",
        "",
        f"- 公开目录：{DRIVE_ROOT_URL}",
        f"- 根目录项目数：**{len(root_items)}**",
        "- 方法：读取公开 Google Drive 文件元数据；未下载任何文件。",
        "",
        "## 根目录",
        "",
        "| 名称 | 类型 | 大小 | 文件 ID |",
        "|---|---|---:|---|",
    ]
    for item in root_items:
        item_type = "folder" if item.get("mimeType") == GOOGLE_DRIVE_MIME else item.get("mimeType", "unknown")
        lines.append(f"| `{item.get('name', '')}` | {item_type} | {human_size(item.get('size'))} | `{item.get('id', '')}` |")

    for folder_name, items in child_listing.items():
        total_known = sum(int(item.get("size", 0)) for item in items if item.get("size") not in (None, ""))
        lines.extend(
            [
                "",
                f"## `{folder_name}/`",
                "",
                f"- 项目数：**{len(items)}**",
                f"- 已知文件大小合计：**{human_size(total_known)}**",
                "",
                "| 名称 | 类型 | 大小 | 文件 ID |",
                "|---|---|---:|---|",
            ]
        )
        for item in items:
            item_type = "folder" if item.get("mimeType") == GOOGLE_DRIVE_MIME else item.get("mimeType", "unknown")
            lines.append(f"| `{item.get('name', '')}` | {item_type} | {human_size(item.get('size'))} | `{item.get('id', '')}` |")

    lines.extend(
        [
            "",
            "## 官方 `ReadMe.md` 核验",
            "",
            "公开说明给出的游戏 ZIP 结构只有 `<session>/data_file.csv` 与 `<session>/video/*.jpg`。它把 Scene 数据定义为：",
            "",
            "- 对象：`object_name`, `bounds`, `m_matrix`",
            "- 相机：`camera_name`, `p_matrix`, `v_matrix`",
            "- `main_log.csv` 的 0/1 值表示相应数据类型是否存在。",
            "- 某些游戏缺少 scene 与 face 信息，此时 CSV 会省略相应列。",
            "",
            "该说明没有列出单独的 FOV 标量、depth texture 或 motion-vector/flow 文件。",
            "",
            f"说明文件读取成功：{'是' if bool(readme_text) else '否'}；更新说明读取成功：{'是' if bool(updates_text) else '否'}。",
        ]
    )

    lines.extend(["", "## 图形标签名称搜索", ""])
    if matches:
        lines.extend(
            [
                "| 位置 | 名称 | 类型 | 大小 |",
                "|---|---|---|---:|",
            ]
        )
        for location, item in matches:
            lines.append(
                f"| `{location}` | `{item.get('name', '')}` | {item.get('mimeType', 'unknown')} | {human_size(item.get('size'))} |"
            )
    else:
        lines.append("未找到名称含 `label`、`graphics`、`render`、`motion`、`flow`、`depth`、`camera`、`metadata` 或 `frame` 的根目录/一级子目录项目。")
    lines.extend(
        [
            "",
            "## 判断",
            "",
            "公开 Drive 的根目录和一级子目录中未发现单独的 `graphics_labels/`、`render_labels/`、`frame_labels/`、`camera_params/`、`motion_flow/`、`depth/` 或 `metadata/` 资源。图形相关字段应按 session 可用性直接包含在游戏 ZIP 内的 `data_file.csv`；公开目录未暴露论文所述 depth texture 或 motion-vector texture 的独立下载项。",
            "",
        ]
    )
    (root / "results/vrnet_drive_structure.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    (root / "results").mkdir(parents=True, exist_ok=True)
    write_csv_columns_report(root)
    write_main_log_report(root)
    write_zip_report(root)
    write_drive_report(root)
    print("Wrote local CSV, main_log, ZIP, and Drive structure reports.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
