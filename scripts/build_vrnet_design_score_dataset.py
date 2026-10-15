"""Audit local VR.net sources and export classified, report-level raw data.

No model fitting, downloads, resampling, score propagation, or source edits.
The exported scene columns are runtime graphics candidates, not certified
independent design settings. Every original rating row is retained.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import logging
import math
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(2_147_483_647)
PAPER = "https://arxiv.org/html/2306.03381v1"
DRIVE = "https://drive.google.com/drive/folders/1LLo4OE83Zeso8MkEXSvW9VrdjiqqlFRR"
UNITY = "https://docs.unity3d.com/2022.3/Documentation/ScriptReference/"
POSE = ["head_dir", "head_pos", "head_vel", "head_angvel",
        "left_eye_dir", "left_eye_pos", "left_eye_vel", "left_eye_angvel",
        "right_eye_dir", "right_eye_pos", "right_eye_vel", "right_eye_angvel"]
GAZE = ["left_eye", "right_eye", "confidence", "is_valid"]
CONTROL = ["ConnectedControllerTypes", "Buttons", "Touches", "NearTouches",
           "IndexTrigger", "HandTrigger", "Thumbstick"]
FACE = ["exp_weights", "exp_confidence", "exp_validity"]
SCENE = ["object_name", "bounds", "m_matrix", "p_matrix"]
MIXED = ["camera_name", "v_matrix"]
INTERACTION = POSE + GAZE + CONTROL + FACE
META = ["frame", "timestamp", "video"]
NATIVE_ROLES = ["camera", "object", "light", "pose", "control"]
KEYS = ["event_id", "dataset_family", "session_key", "session_id", "game_name",
        "participant_id", "participant_id_source", "source_file", "source_sha256",
        "source_csv_record", "report_frame_raw", "report_timestamp_raw",
        "timestamp_unit_interpretation", "timestamp_quality", "score_raw",
        "score_value", "score_numeric_valid", "rating_source", "rating_source_semantics",
        "repeated_record_key", "strict_design_numeric_fields_verified"]
SCENE_COLS = ["scene__" + x for x in SCENE] + ["mixed__" + x for x in MIXED]
SCENE_COLS += ["mixed__native_camera_raw_json", "scene__native_object_raw_json",
               "scene__native_light_raw_json"]
INT_COLS = ["interaction__" + x for x in INTERACTION]
INT_COLS += ["interaction__native_pose_raw_json", "interaction__native_control_raw_json"]
OTHER = ["asset__video_raw", "field_alignment_provenance_json",
         "has_scene_graphics_snapshot", "has_interaction_snapshot",
         "join_policy", "pure_design_status", "training_ready"]


def present(value):
    return value is not None and str(value).strip().lower() not in {
        "", "none", "null", "nan", "[]", "{}", "()"}


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def clock(value):
    """Interpret observed epoch magnitude, without certifying exporter units."""
    raw = str(value).strip()
    value = number(raw)
    if value is None:
        return None, "unknown", "missing_or_invalid"
    if "e" in raw.lower():
        return None, "unknown", "rounded_scientific_not_aligned"
    if abs(value) >= 1e14:
        return value / 1e6, "epoch_microseconds_inferred", "integer_epoch_microseconds"
    if abs(value) >= 1e11:
        return value / 1e3, "epoch_milliseconds_inferred", "integer_epoch_milliseconds"
    return None, "unknown", "clock_unit_unverified_not_aligned"


def sha_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def open_source(source):
    if not source.get("archive"):
        return source["path"].open("r", encoding="utf-8-sig", newline="")
    archive = zipfile.ZipFile(source["archive"])
    stream = io.TextIOWrapper(archive.open(source["member"]), encoding="utf-8-sig", newline="")
    # Keep archive lifetime tied to the returned stream.
    stream._vrnet_archive = archive
    return stream


def classify(field, family="modern"):
    base = dict(field=field, dataset_family=family, source_url=PAPER,
                is_interaction_parameter="no", pure_design_status="not_applicable",
                strict_design_input_allowed="no", unit="unknown")
    if family in {"native_pose", "native_control"} and field in {"device_id", "packetNum"}:
        return dict(base, category="device_or_packet_metadata", meaning="设备编号/数据包编号；不是动作量或设计参数")
    if family == "native_pose" and field not in {"framecounter", "timestamp"}:
        return dict(base, category="interaction_tracked_device_pose", is_interaction_parameter="yes",
                    meaning="追踪设备位姿/运动；具体HMD或控制器身份待确认", pure_design_status="no")
    if field in POSE or (family == "native_pose" and field not in {"framecounter", "timestamp"}):
        return dict(base, category="interaction_pose", is_interaction_parameter="yes",
                    meaning="头显/眼相关位姿或运动遥测；不等同人工导航设置", pure_design_status="no")
    if field in GAZE:
        return dict(base, category="interaction_gaze", is_interaction_parameter="yes",
                    meaning="眼动数据或其质量标志；发布README归入Gaze", pure_design_status="no", source_url=DRIVE)
    if field in CONTROL or family == "native_control" and field not in {"framecounter", "timestamp"}:
        return dict(base, category="interaction_control", is_interaction_parameter="yes",
                    meaning="控制器输入或设备连接状态", pure_design_status="no")
    if field in FACE:
        return dict(base, category="interaction_face", is_interaction_parameter="yes",
                    meaning="表情权重或追踪质量，不是情绪或晕动症真值", pure_design_status="no", source_url=DRIVE)
    if field == "MS_rating" or family == "native_score" and field in {"score", "rating_source"}:
        return dict(base, category="subjective_rating", meaning="自报评分/报告来源；auto/manual含义未认证")
    if field in {"v_matrix", "view"}:
        return dict(base, category="mixed_runtime_camera", pure_design_status="unverified",
                    meaning="总相机view矩阵，可能合成头动、人工导航和脚本运动",
                    source_url=UNITY + "Camera-worldToCameraMatrix.html")
    if field in {"p_matrix", "projection"}:
        return dict(base, category="scene_render_projection", pure_design_status="unverified",
                    meaning="当前投影；可能由XR设备/SDK决定，不等同作者FOV预设",
                    source_url=UNITY + "Camera-projectionMatrix.html")
    if field in {"m_matrix", "localToWorldMatrix"}:
        return dict(base, category="scene_runtime_transform", pure_design_status="unverified",
                    meaning="当前对象/光源变换；不提供用户作用与动画/初始设计的来源分离",
                    source_url=UNITY + "Renderer-localToWorldMatrix.html")
    if field in {"bounds", "boundingBox"}:
        return dict(base, category="scene_runtime_geometry", pure_design_status="unverified",
                    meaning="当前世界包围盒；不等同原始模型尺寸或空间密度",
                    source_url=UNITY + "Renderer-bounds.html")
    if field in {"object_name", "camera_name", "name"}:
        return dict(base, category="scene_metadata", pure_design_status="metadata_only",
                    meaning="开发者命名/当前记录对象名称；不是稳定实例ID或数值设计设置")
    if family == "native_light" and field not in {"timestamp", "framecounter"}:
        return dict(base, category="scene_runtime_light", pure_design_status="unverified",
                    meaning="当前光源属性；未证明静态预设/用户独立；没有light type或active状态",
                    source_url=UNITY + "Light.html")
    if field in {"timestamp", "frame", "framecounter", "video"}:
        return dict(base, category="alignment_or_asset", meaning="原始记录时间/帧索引/图像引用，不作设计输入")
    return dict(base, category="unknown", meaning="未认证语义，保留原字段，不作设计输入")


def discover(root, log):
    raw = root / "data/raw/vrnet"
    sources = []
    for path in sorted(raw.rglob("data_file.csv")):
        if "__MACOSX" in path.parts:
            continue
        sources.append(dict(path=path, locator=str(path.resolve()), session_id=path.parent.name,
                            family="modern", size_bytes=path.stat().st_size))
    for path in sorted((raw / "samples").glob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            members = [i for i in archive.infolist() if i.filename.endswith("/data_file.csv")
                       and "__MACOSX" not in i.filename]
            log.info("archive listing %s: %s CSV entries (no extraction)", path.name, len(members))
            for member in members:
                sources.append(dict(archive=path, member=member.filename,
                                    locator=str(path.resolve()) + "::" + member.filename,
                                    session_id=Path(member.filename).parent.name,
                                    family="modern", size_bytes=member.file_size))
    native = raw / "rollercoaster_raw"
    for path in sorted(native.rglob("*.csv")):
        if "__MACOSX" in path.parts or ".ipynb_checkpoints" in path.parts:
            continue
        if path.name != "voice.csv" and "SelfReportData" not in path.parts:
            continue
        match = re.search(r"VRLOG-[\w]+", str(path.name if path.name != "voice.csv" else path.parent.name))
        if not match:
            continue
        sources.append(dict(path=path, locator=str(path.resolve()), session_id=match.group(0),
                            family="native", size_bytes=path.stat().st_size))
    groups = defaultdict(list)
    for index, source in enumerate(sources, 1):
        try:
            if source.get("archive"):
                with zipfile.ZipFile(source["archive"]) as archive, archive.open(source["member"]) as stream:
                    source["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            else:
                source["sha256"] = sha_file(source["path"])
            source["status"] = "hashed"
            groups[(source["family"], source["session_id"], source["sha256"])].append(source)
        except Exception as exc:
            source.update(status="failed", error=repr(exc), sha256="")
            log.exception("source hashing failed %s", source["locator"])
        if index % 10 == 0:
            log.info("hashed %s/%s local CSV sources", index, len(sources))
    canonical = []
    variants = Counter((key[0], key[1]) for key in groups)
    for (family, session, digest), copies in groups.items():
        copies.sort(key=lambda s: (bool(s.get("archive")), s["locator"]))
        chosen = copies[0]
        for copy in copies:
            copy["canonical_locator"] = chosen["locator"]
            copy["status"] = "canonical" if copy is chosen else "byte_identical_alias"
        chosen["session_key"] = family + ":" + session
        chosen["multiple_source_versions"] = variants[(family, session)] > 1
        if chosen["multiple_source_versions"]:
            chosen["session_key"] += ":" + digest[:12]
        chosen["aliases"] = [s["locator"] for s in copies[1:]]
        canonical.append(chosen)
    return sources, canonical


def modern_identity(session):
    parts = session.split("_", 2)
    if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
        return parts[2], "modern:" + parts[0], "official_release_README_filename_convention"
    return "unknown", "unknown", "unknown"


def new_event(source, record, raw_time, rating, frame="", channel="unknown"):
    t, unit, quality = clock(raw_time)
    if source["family"] == "modern":
        game, participant, pid_source = modern_identity(source["session_id"])
    else:
        # Use a standalone filename alias to recover recording metadata.
        file = next((Path(x) for x in [source["locator"]] + source["aliases"]
                     if "SelfReportData" in x), None)
        game, participant, pid_source = "unknown", "unknown", "unknown"
        if file:
            m = re.match(r"(?:P)?(\d+)\s+(VRLOG-\w+)\s+(.+)\.csv$", file.name)
            if m:
                participant, game = "native:" + m.group(1), m.group(3)
                pid_source = "standalone_selfreport_filename"
    out = dict(event_id=source["session_key"] + ":record:" + str(record),
               dataset_family=source["family"], session_key=source["session_key"],
               session_id=source["session_id"], game_name=game, participant_id=participant,
               participant_id_source=pid_source, source_file=source["locator"],
               source_sha256=source["sha256"], source_csv_record=record,
               report_frame_raw=frame, report_timestamp_raw=raw_time,
               timestamp_unit_interpretation=unit, timestamp_quality=quality,
               score_raw=rating, score_value=number(rating), score_numeric_valid=number(rating) is not None,
               rating_source=channel, rating_source_semantics="unverified",
               repeated_record_key=False, strict_design_numeric_fields_verified=0,
               pure_design_status="no_numeric_design_setting_certified",
               training_ready=False, join_policy="latest_prior_record_within_1s;no_future;no_score_fill",
               field_alignment_provenance_json="{}", _time=t, _provenance={})
    return out


def scan_modern(source, log, field_stats):
    latest = {}
    rows = 0
    nonempty = Counter()
    events = []
    seen = Counter()
    first_time = last_time = None
    backwards = 0
    with open_source(source) as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        for rows, row in enumerate(reader, 1):
            if None in row:
                raise ValueError(f"extra CSV columns at logical record {rows}")
            t, _, _ = clock(row.get("timestamp", ""))
            if t is not None:
                if last_time is not None and t < last_time:
                    backwards += 1
                    latest.clear()  # Do not propagate observations across clock resets.
                first_time = t if first_time is None else min(t, first_time)
                last_time = t
            for field, value in row.items():
                stat = field_stats[("modern", field)]
                stat["rows"] += 1
                if present(value):
                    nonempty[field] += 1
                    stat["nonempty"] += 1
                    if not stat.get("example"):
                        stat["example"] = str(value)[:160]
                        stat["source"] = source["locator"]
                    if field in SCENE + MIXED + INTERACTION + ["video"] and t is not None:
                        latest[field] = (value, t, rows, row.get("frame", ""))
            if not present(row.get("MS_rating")):
                continue
            event = new_event(source, rows, row.get("timestamp", ""), row["MS_rating"], row.get("frame", ""))
            key = (event["report_timestamp_raw"], event["score_raw"], event["rating_source"])
            event["repeated_record_key"] = seen[key] > 0
            seen[key] += 1
            if t is not None:
                for field, (value, observed_time, observed_row, frame) in latest.items():
                    age = t - observed_time
                    if -1e-8 <= age <= 1.0 + 1e-8:
                        prefix = "scene__" if field in SCENE else "mixed__" if field in MIXED else "asset__" if field == "video" else "interaction__"
                        dest = prefix + ("video_raw" if field == "video" else field)
                        event[dest] = value
                        event["_provenance"][dest] = dict(source_file=source["locator"],
                            source_csv_record=observed_row, frame_raw=frame, age_seconds=round(age, 9))
            events.append(event)
    log.info("%s: %s CSV rows, %s recorded ratings", source["session_key"], rows, len(events))
    game, pid, pid_source = modern_identity(source["session_id"])
    return events, dict(session_key=source["session_key"], dataset_family="modern", session_id=source["session_id"],
                        game_name=game, participant_id=pid, participant_id_source=pid_source, csv_rows=rows,
                        raw_columns_json=json.dumps(fields), nonempty_by_field_json=json.dumps(nonempty),
                        timestamp_backwards_count=backwards,
                        observed_duration_seconds=last_time-first_time if last_time is not None else None)


def native_snapshots(root, source, events, field_stats, log):
    directory = next((p for p in (root / "data/raw/vrnet/rollercoaster_raw/derived_csv").glob("*")
                      if source["session_id"] in p.name), None)
    if not directory:
        return []
    pairs = sorted((e["_time"], i) for i, e in enumerate(events) if e["_time"] is not None)
    report_times = [p[0] for p in pairs]
    sources = []
    for role in NATIVE_ROLES:
        path = directory / (role + ".csv")
        if not path.is_file():
            continue
        digest = sha_file(path)
        snapshots = {}
        row_count = 0
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = [f for f in reader.fieldnames or [] if f]
            for row_count, row in enumerate(reader, 1):
                for field in fields:
                    stat = field_stats[("native_" + role, field)]
                    stat["rows"] += 1
                    if present(row.get(field)):
                        stat["nonempty"] += 1
                        if not stat.get("example"):
                            stat["example"], stat["source"] = str(row[field])[:160], str(path.resolve())
                t, _, _ = clock(row.get("timestamp", ""))
                if t is None:
                    continue
                a, b = bisect.bisect_left(report_times, t), bisect.bisect_right(report_times, t + 1.0)
                for k in range(a, b):
                    _, i = pairs[k]
                    old = snapshots.get(i)
                    if old is None or t > old[0]:
                        snapshots[i] = (t, [dict(source_csv_record=row_count, **{f: row.get(f, "") for f in fields})])
                    elif t == old[0]:
                        old[1].append(dict(source_csv_record=row_count, **{f: row.get(f, "") for f in fields}))
        dest = ("mixed__" if role == "camera" else "scene__" if role in {"object", "light"}
                else "interaction__") + "native_" + role + "_raw_json"
        for i, (t, rows) in snapshots.items():
            events[i][dest] = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
            events[i]["_provenance"][dest] = dict(source_file=str(path.resolve()), source_sha256=digest,
                                                  recorded_rows=len(rows), age_seconds=round(events[i]["_time"]-t, 9))
        log.info("%s %s: %s rows, %s reports have prior snapshot", source["session_id"], role, row_count, len(snapshots))
        sources.append(dict(locator=str(path.resolve()), family="native_" + role, session_id=source["session_id"],
                            sha256=digest, size_bytes=path.stat().st_size, status="native_parameter_source",
                            canonical_locator=str(path.resolve())))
    return sources


def scan_native(root, source, log, field_stats):
    events = []
    seen = Counter()
    with open_source(source) as stream:
        for record, fields in enumerate(csv.reader(stream), 1):
            if not fields or all(not present(f) for f in fields):
                continue
            if len(fields) != 3:
                raise ValueError(f"native score logical record {record} has {len(fields)} columns")
            raw_time, rating, channel = (f.strip() for f in fields)
            event = new_event(source, record, raw_time, rating, channel=channel)
            key = (raw_time, rating, channel)
            event["repeated_record_key"] = seen[key] > 0
            seen[key] += 1
            events.append(event)
            for field, value in zip(["timestamp", "score", "rating_source"], [raw_time, rating, channel]):
                stat = field_stats[("native_score", field)]
                stat["rows"] += 1
                stat["nonempty"] += int(present(value))
                if not stat.get("example"):
                    stat["example"], stat["source"] = value, source["locator"]
    parameter_sources = native_snapshots(root, source, events, field_stats, log)
    first = events[0] if events else {}
    log.info("%s: %s original score records", source["session_key"], len(events))
    return events, dict(session_key=source["session_key"], dataset_family="native", session_id=source["session_id"],
                        game_name=first.get("game_name", "unknown"), participant_id=first.get("participant_id", "unknown"),
                        participant_id_source=first.get("participant_id_source", "unknown"), csv_rows=len(events),
                        raw_columns_json=json.dumps(["timestamp", "score", "rating_source"]),
                        nonempty_by_field_json="{}", timestamp_backwards_count="unknown", observed_duration_seconds=None), parameter_sources


def document_dataset(out, root):
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    with (out / "session_inventory.csv").open(encoding="utf-8-sig", newline="") as stream:
        sessions = list(csv.DictReader(stream))
    with (out / "field_dictionary.csv").open(encoding="utf-8-sig", newline="") as stream:
        dictionary = list(csv.DictReader(stream))
    game_counts = defaultdict(lambda: Counter())
    for s in sessions:
        if s["scan_status"] == "success":
            game_counts[(s["dataset_family"], s["game_name"])]["sessions"] += 1
            game_counts[(s["dataset_family"], s["game_name"])]["ratings"] += int(s["rating_record_count"])
            game_counts[(s["dataset_family"], s["game_name"])]["graphics"] += int(s["reports_with_graphics_snapshot"])
    lines = ["# VR.net 场景参数、交互参数与原始评分数据集整理", "", "核验日期：2026-10-10（Pacific/Auckland）。", "",
        "## 结论", "",
        "已整理本地能实际读取的数据，包含场景图形候选字段、用户交互字段与原始评分记录。新版保留MS_rating；原始包auto/manual来源含义未认证，不把全部记录笼统视作独立用户自报。",
        "**没有找到可严格认证为完全独立于用户行为的数值设计设置。** 图形字段来自渲染引擎，但记录的是运行时状态；它们不自动等于设计预设。",
        "因此 `feature_groups.json` 的 `strict_design_features` 为空，所有报告行的 `training_ready=False`；这是纯设计参数证据不足，不能解释为所有场景无风险。", "",
        "## 检查范围与数量", "",
        f"- 本地新版公开索引：{summary['global_index_games']} 个游戏、{summary['global_index_sessions']} 个 session。仅索引完整，不代表全量原始数据已下载或核验。",
        f"- 索引中有场景对象、场景相机、自报评分三种模态的 session：{summary['global_index_scene_and_rating']}（资源标记，非评分记录数）。",
        f"- 实际核验新版原始 CSV：{summary['local_modern_unique_sessions']} 个 session、{summary['local_modern_games']} 个游戏。",
        f"- 原始发布包：{summary['local_native_unique_sessions']} 个评分 session；其中5个过山车记录另有camera/object/light/pose/control派生表。",
        f"- 总计保留原始评分记录 {summary['recorded_rating_rows']} 条。它们是原始报告行，不等于独立样本或参与者数。",
        f"- 按 SHA-256 排除字节相同文件副本 {summary['byte_identical_aliases']} 份。不同时间的相同评分全部保留；同记录键重复仅标记，不删除。",
        f"- 扫描失败数：{len(summary['source_scan_failures'])}；不同内容的同session版本数：{len(summary['multiple_version_sessions'])}。", "",
        "| 原始来源 | 评分记录 | 评分实际分布 | 有最近前置图形观测（≤1秒） | 有最近前置交互观测（≤1秒） |",
        "|---|---:|---|---:|---:|"]
    for family, data in summary["families"].items():
        lines.append(f"| {family} | {data['recorded_rating_rows']} | {json.dumps(data['rating_distribution'], ensure_ascii=False)} | {data['reports_with_graphics_snapshot']} | {data['reports_with_interaction_snapshot']} |")
    lines += ["", "新版与原始包保留不同命名空间；没有证明跨版本参与者对应，也没有合并量表编码。", "",
        "## 数据里具体包含什么", "",
        "| 分类 | 实际字段/内容 | 能否作为纯设计输入 |",
        "|---|---|---|",
        "| 场景对象图形 | object_name、bounds、m_matrix；原始包name、boundingBox、localToWorldMatrix | 候选；当前对象状态可能受交互、动画和可见性筛选影响 |",
        "| 渲染投影 | p_matrix / projection | 候选；含XR投影，不能自动称开发者设置的FOV |",
        "| 混合相机运动 | camera_name、v_matrix / name、view | 不直接启用；可能合成用户头动与人工导航 |",
        "| 原始包光源属性 | intensity、range、spotAngle、颜色、colorTemperature、bounceIntensity、shadow参数和变换 | 候选；是运行时光源状态，非已证明静态预设 |",
        "| 用户交互：位姿 | head_dir/pos/vel/angvel、左右眼相关位姿；原始包deviceToAbsoluteTracking/velocity/angularVelocity/device_id | 标为interaction；纯场景模型排除 |",
        "| 用户交互：眼动 | left_eye、right_eye、confidence、is_valid | 标为interaction；质量列不当作行为量 |",
        "| 用户交互：控制器 | Buttons、Touches、NearTouches、IndexTrigger、HandTrigger、Thumbstick、ConnectedControllerTypes；原始包buttons/axis字段 | 标为interaction；连接状态不当作设计参数 |",
        "| 用户交互：表情 | exp_weights、exp_confidence、exp_validity（只在部分新版schema存在） | 标为interaction；不当作情绪或晕动真值 |",
        "| 评分记录 | 新版自报MS_rating；原始包timestamp,score,auto/manual | 候选监督目标；auto/manual含义未认证；没有改分、填充或重映射 |",
        "| 元信息 | source、game、participant、session、frame、timestamp、video引用、哈希、对齐来源 | 用于追溯与分组，不直接作设计输入 |", "",
        "## 缺少或尚未认证的设计信息", "",
        "没有找到明确的作者移动速度上限、加速度曲线、预设路径、导航方式设置、参考框架设置、粒子发射率、三角面数、预设体验时长。",
        "原始过山车 light.csv 补充了光源属性；没有 light type/active 状态，不能直接把行数当动态光源总数。",
        "不能把session实际时长称预设时长；不能把渲染记录条目数称场景GameObject总数或空间密度。没有在本轮反算FOV、速度/加速度或推断米制。", "",
        "## 数据组织与标注规则", "",
        "- `vrnet_score_parameter_dataset.csv`：合并主表，一行对应canonical评分文件中的一个原始报告记录。",
        "- `scene_graphics_at_reports.csv`：场景图形候选及混合相机字段，无用户交互列；附同一评分记录和元信息。",
        "- `interaction_at_reports.csv`：交互字段统一以`interaction__`前缀标注；附同一报告评分。",
        "- `score_events.csv`：评分、来源、身份、原始时间戳与记录序号。",
        "- `field_dictionary.csv`：每个原始字段的类别、交互标记、纯设计状态、单位状态、实测非空数、来源URL与本地例子。",
        "- `session_inventory.csv`：实际CSV行数、列名、评分分布、来源、副本、版本与扫描状态。",
        "- `global_index_inventory.csv`：全部索引条目及本地实际覆盖；0/1列仅为模态可用性标记。",
        "- `source_catalog.csv`：所有检查来源的SHA-256、文件大小、canonical/alias状态与错误。", "",
        "矩阵、向量和对象列表保存原始字符串，不伪造标量设计值。纯设计认证字段目前为0。",
        "空单元格代表该报告没有满足联结要求的实测值，不代表数值0。现代字段按其最近的已记录过去值联结，最多1秒；不同字段有各自来源记录。原始包按每类表最近的过去时间戳保存该时刻记录组；不同组不保证同帧。",
        "每个字段/组在 `field_alignment_provenance_json` 中记录来源、记录序号/时间差。正确名称为评分记录级最近前置原始观测联结表，不保证同一时刻完整快照；它不是3/30/60秒窗口聚合训练表。没有向未来匹配，也没有沿用评分。",
        "`has_any_*_observation`仅表示至少一个字段有记录；`scene_numeric_observation_present`和`interaction_value_observation_present`另外区分矩阵/数值与名称/连接状态。现代object/camera/gaze组的`*_group_same_source_record`为真时，才允许跨字段联合解码。",
        "同源标志不证明列表长度匹配、对象身份稳定或眼动有效，仍需进一步检查。native的这些现代组标志为False表示不适用，不表示其组内联结失败。",
        "native raw_json保留作者导出CSV的字段原字符串；四位小数等导出精度不能称原始二进制日志的无损精度。", "",
        "## 时间、身份和评分限制", "",
        "现代与native运动表按13位epoch毫秒解释；原始评分按16位epoch微秒解释，并保留单位推断状态。`1.68325E+15`一类时间已丢精度，不还原为精确微秒、不配参数。",
        "modern参与者ID依照官方README目录约定解析，并加modern命名空间；native ID取评分文件名，加native命名空间。跨版本身份对应未确认。",
        "auto/manual语义未认证，两种来源分别保留。不能把全部记录直接视为相互独立且可靠的人体标签。评分的观察范围不等同量表正式上下界。", "",
        "## 每个游戏的本地覆盖", "",
        "| 来源版本 | 游戏 | Session/版本 | 原始评分行 | 有最近前置图形观测评分行 |", "|---|---|---:|---:|---:|"]
    for (family, game), counts in sorted(game_counts.items()):
        lines.append(f"| {family} | {game} | {counts['sessions']} | {counts['ratings']} | {counts['graphics']} |")
    lines += ["", "## 来源与复现", "",
        f"- 原论文：[VR.net v1]({PAPER})，§3.2/3.3为字段类型依据，§4.3为自报预测任务依据。论文版本规模不能代替当前公开索引规模。",
        f"- 发布索引与README：[公开Drive]({DRIVE})；本轮读取本地已保存索引和 `results/rollercoaster_sources/ReadMe.md`，没有重新下载全量或声称在线目录重新核验。",
        f"- 引擎字段语义：[view矩阵]({UNITY}Camera-worldToCameraMatrix.html)、[projection]({UNITY}Camera-projectionMatrix.html)、[bounds]({UNITY}Renderer-bounds.html)、[Light]({UNITY}Light.html)。",
        "- 当前原始CSV全部只读。6个本地ZIP仅在原位读取CSV成员，没有解压、删除或新增下载。没有训练模型，没有Git commit。",
        "- 复现：使用bundled Python运行 `scripts/build_vrnet_design_score_dataset.py --project-root <project> --output-dir <output>`。实测日志为 `audit_log.txt`。", ""]
    content = "\n".join(lines)
    (out / "README.md").write_text(content, encoding="utf-8")
    (root / "results/vrnet_design_score_dataset_summary.md").write_text(content, encoding="utf-8")


def finalize_dataset(out, root):
    """Add independent coverage/coherence flags to the exported observations."""
    with (out / "vrnet_score_parameter_dataset.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    flags = ["has_any_graphics_observation", "has_any_interaction_observation",
             "scene_numeric_observation_present", "interaction_value_observation_present",
             "object_group_same_source_record", "camera_group_same_source_record", "gaze_group_same_source_record"]
    for row in rows:
        row["has_any_graphics_observation"] = row.pop("has_scene_graphics_snapshot", row.get("has_any_graphics_observation", "False"))
        row["has_any_interaction_observation"] = row.pop("has_interaction_snapshot", row.get("has_any_interaction_observation", "False"))
        row["scene_numeric_observation_present"] = any(present(row.get(c)) for c in [
            "scene__bounds", "scene__m_matrix", "scene__p_matrix", "mixed__v_matrix",
            "mixed__native_camera_raw_json", "scene__native_object_raw_json", "scene__native_light_raw_json"])
        row["interaction_value_observation_present"] = any(present(row.get(c)) for c in INT_COLS if c != "interaction__ConnectedControllerTypes")
        provenance = json.loads(row["field_alignment_provenance_json"])
        for name, columns in {
            "object": ["scene__object_name", "scene__bounds", "scene__m_matrix"],
            "camera": ["mixed__camera_name", "scene__p_matrix", "mixed__v_matrix"],
            "gaze": ["interaction__left_eye", "interaction__right_eye", "interaction__confidence", "interaction__is_valid"]}.items():
            entries = [provenance.get(c, {}) for c in columns]
            row[name + "_group_same_source_record"] = row["dataset_family"] == "modern" and all(present(row.get(c)) for c in columns) and \
                all(e.get("source_csv_record") is not None for e in entries) and len({(e.get("source_file"), e.get("source_csv_record")) for e in entries}) == 1
    other = [c for c in OTHER if c not in {"has_scene_graphics_snapshot", "has_interaction_snapshot"}]
    write_csv(out / "vrnet_score_parameter_dataset.csv", rows, KEYS + SCENE_COLS + INT_COLS + flags + other)
    write_csv(out / "scene_graphics_at_reports.csv", rows, KEYS + SCENE_COLS + flags + ["field_alignment_provenance_json", "pure_design_status", "join_policy", "training_ready"])
    write_csv(out / "interaction_at_reports.csv", rows, KEYS + INT_COLS + flags + ["field_alignment_provenance_json", "join_policy", "training_ready"])
    with (out / "field_dictionary.csv").open(encoding="utf-8-sig", newline="") as stream:
        dictionary = list(csv.DictReader(stream))
    for item in dictionary:
        if item["dataset_family"].startswith("native_"):
            item.update(classify(item["field"], item["dataset_family"]))
    write_csv(out / "field_dictionary.csv", dictionary)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    summary["numerical_and_coherent_coverage"] = {family: {flag: sum(r["dataset_family"] == family and r[flag] in {True, "True"} for r in rows)
        for flag in flags[2:]} for family in ["modern", "native"]}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    document_dataset(out, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/vrnet_design_score_2026-10-10"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("vrnet_design_score")
    log.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    for handler in [logging.FileHandler(out / "audit_log.txt", encoding="utf-8"), logging.StreamHandler(sys.stdout)]:
        handler.setFormatter(formatter)
        log.addHandler(handler)
    begin = time.time()
    sources, canonical = discover(root, log)
    stats = defaultdict(lambda: dict(rows=0, nonempty=0))
    events, sessions = [], []
    for source in canonical:
        try:
            if source["family"] == "modern":
                rows, info = scan_modern(source, log, stats)
            else:
                rows, info, parameters = scan_native(root, source, log, stats)
                sources.extend(parameters)
            for row in rows:
                row["has_scene_graphics_snapshot"] = any(present(row.get(c)) for c in SCENE_COLS)
                row["has_interaction_snapshot"] = any(present(row.get(c)) for c in INT_COLS)
                row["field_alignment_provenance_json"] = json.dumps(row.pop("_provenance"), ensure_ascii=False, separators=(",", ":"))
                row.pop("_time", None)
            info.update(source_file=source["locator"], source_sha256=source["sha256"],
                        aliases_json=json.dumps(source["aliases"]), multiple_source_versions=source["multiple_source_versions"],
                        rating_record_count=len(rows), numeric_rating_count=sum(r["score_numeric_valid"] for r in rows),
                        rating_frequencies_json=json.dumps(dict(Counter(str(r["score_value"]) for r in rows if r["score_numeric_valid"]))),
                        repeated_record_keys=sum(r["repeated_record_key"] for r in rows),
                        reports_with_graphics_snapshot=sum(r["has_scene_graphics_snapshot"] for r in rows),
                        reports_with_interaction_snapshot=sum(r["has_interaction_snapshot"] for r in rows), scan_status="success")
            events.extend(rows)
            sessions.append(info)
            source["scan_status"] = "success"
        except Exception as exc:
            source["scan_status"] = "failed"
            source["error"] = repr(exc)
            sessions.append(dict(session_key=source["session_key"], dataset_family=source["family"],
                                 session_id=source["session_id"], source_file=source["locator"], scan_status="failed", error=repr(exc)))
            log.exception("source scan failed %s", source["locator"])
    index_path = root / "data/raw/vrnet/main_log.csv"
    with index_path.open(encoding="utf-8-sig", newline="") as stream:
        global_index = list(csv.DictReader(stream))
    modern_sessions = [s for s in sessions if s["dataset_family"] == "modern" and s["scan_status"] == "success"]
    verified_local = {s["session_id"]: s for s in modern_sessions}
    for row in global_index:
        game, pid, pid_source = modern_identity(row["name"])
        row.update(game_name=game, participant_id=pid, participant_id_source=pid_source,
                   local_raw_csv_verified=row["name"] in verified_local,
                   index_scene_and_rating_available=all(row.get(f) == "1" for f in ["scene_obj", "scene_cam", "MS_rating"]),
                   index_flags_are_scores=False, source_file=str(index_path.resolve()), source_url=DRIVE)
    dictionary = []
    for (family, field), stat in sorted(stats.items()):
        item = classify(field, family)
        item.update(observed_rows=stat["rows"], nonempty_rows=stat["nonempty"],
                    example_truncated=stat.get("example", ""), local_source_example=stat.get("source", ""))
        dictionary.append(item)
    # Desired design settings not present in the checked export schemas.
    for field in ["authored_locomotion_speed", "authored_acceleration_profile", "scripted_camera_path",
                  "particle_emission_rate", "triangle_count", "expected_duration_setting",
                  "navigation_mode_setting", "reference_frame_setting", "dynamic_light_count_setting"]:
        dictionary.append(dict(field=field, dataset_family="desired_design_setting", category="missing_design_setting",
            is_interaction_parameter="no", pure_design_status="not_observed", strict_design_input_allowed="no",
            unit="unknown", meaning="未找到明确作者设计设置字段；不由运行时数值或列名猜测生成",
            source_url=PAPER, observed_rows=0, nonempty_rows=0, example_truncated="", local_source_example="checked_local_sources"))
    catalog = [{k: s.get(k, "") for k in ["locator", "family", "session_id", "size_bytes", "sha256",
               "status", "canonical_locator", "scan_status", "error"]} for s in sources]
    write_csv(out / "source_catalog.csv", catalog)
    write_csv(out / "session_inventory.csv", sessions)
    write_csv(out / "global_index_inventory.csv", global_index)
    write_csv(out / "field_dictionary.csv", dictionary)
    write_csv(out / "score_events.csv", events, KEYS)
    write_csv(out / "scene_graphics_at_reports.csv", events, KEYS + SCENE_COLS + ["field_alignment_provenance_json", "has_scene_graphics_snapshot", "pure_design_status"])
    write_csv(out / "interaction_at_reports.csv", events, KEYS + INT_COLS + ["field_alignment_provenance_json", "has_interaction_snapshot"])
    write_csv(out / "vrnet_score_parameter_dataset.csv", events, KEYS + SCENE_COLS + INT_COLS + OTHER)
    summary = dict(
        date="2026-10-10", scope="local raw CSVs + local ZIP CSV entries + local original score/derived CSVs; not full remote dataset",
        global_index_sessions=len(global_index), global_index_games=len({r["game_name"] for r in global_index}),
        global_index_flags={k:sum(r.get(k)=="1" for r in global_index) for k in ["pose", "gaze", "control", "scene_obj", "scene_cam", "video", "MS_rating", "face"]},
        global_index_scene_and_rating=sum(r["index_scene_and_rating_available"] for r in global_index),
        local_modern_unique_sessions=len({s["session_id"] for s in modern_sessions}),
        local_modern_session_versions=len(modern_sessions), local_modern_games=len({s["game_name"] for s in modern_sessions}),
        local_native_unique_sessions=len({s["session_id"] for s in sessions if s["dataset_family"]=="native" and s["scan_status"]=="success"}),
        source_scan_failures=[s["source_file"] for s in sessions if s["scan_status"]=="failed"],
        byte_identical_aliases=sum(s.get("status")=="byte_identical_alias" for s in sources),
        multiple_version_sessions=[s["session_key"] for s in sessions if s.get("multiple_source_versions")],
        recorded_rating_rows=len(events), strict_design_verified_numeric_fields=0,
        families={family:dict(recorded_rating_rows=sum(r["dataset_family"]==family for r in events),
            numeric_rating_rows=sum(r["dataset_family"]==family and r["score_numeric_valid"] for r in events),
            rating_distribution=dict(Counter(str(r["score_value"]) for r in events if r["dataset_family"]==family and r["score_numeric_valid"])),
            source_distribution=dict(Counter(r["rating_source"] for r in events if r["dataset_family"]==family)),
            repeated_record_keys=sum(r["dataset_family"]==family and r["repeated_record_key"] for r in events),
            reports_with_graphics_snapshot=sum(r["dataset_family"]==family and r["has_scene_graphics_snapshot"] for r in events),
            reports_with_interaction_snapshot=sum(r["dataset_family"]==family and r["has_interaction_snapshot"] for r in events),
            imprecise_timestamp_rows=sum(r["dataset_family"]==family and "not_aligned" in r["timestamp_quality"] for r in events))
            for family in ["modern", "native"]}, elapsed_seconds=round(time.time()-begin, 3))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    groups = dict(strict_design_features=[],
                  scene_graphics_candidates=SCENE_COLS, interaction_features=INT_COLS,
                  target="score_value", excluded_predictor_metadata=KEYS,
                  note="empty strict_design_features is a real evidence limitation; graphics != user-independent authored settings")
    (out / "feature_groups.json").write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
    finalize_dataset(out, root)
    log.info("finished: %s recorded ratings; %s modern sessions; %s native sessions; %s failures",
             len(events), summary["local_modern_unique_sessions"], summary["local_native_unique_sessions"], len(summary["source_scan_failures"]))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
