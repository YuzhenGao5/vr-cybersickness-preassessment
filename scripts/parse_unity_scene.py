#!/usr/bin/env python3
"""Extract auditable scene metadata from Unity text-serialized .unity/.prefab files.

Unity's text serializer is a tagged, multi-document YAML subset. This parser
removes Unity's document tags and parses each object with PyYAML. It reports
unknown when the source file does not contain enough evidence; it does not
silently treat missing serialization as a negative finding.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only without declared dependency
    raise SystemExit("PyYAML is required. Install project requirements first.") from exc


DOC_HEADER = re.compile(r"(?m)^---\s+!u!(\d+)\s+&(-?\d+)(?:\s+stripped)?\s*$")
UNKNOWN = "unknown"
OUTPUT_FIELDS = [
    "scene_name", "scene_path", "scene_format", "camera_count", "camera_fov",
    "camera_fov_all", "camera_near_clip", "camera_far_clip",
    "particle_system_count", "particle_emission_rate", "dynamic_light_count",
    "light_component_count", "light_type_distribution", "object_count",
    "transform_depth_max", "triangle_count", "mesh_filter_count",
    "mesh_renderer_count", "has_reference_frame", "has_horizon",
    "has_virtual_nose", "reference_frame_evidence", "controllability",
    "controllability_evidence", "expected_duration",
    "expected_duration_unit", "parser_notes",
]


def as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def scalar_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def ref_id(value: Any) -> str | None:
    if isinstance(value, dict) and "fileID" in value:
        return str(value["fileID"])
    return None


def read_unity_documents(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    text = path.read_text(encoding="utf-8-sig")
    headers = list(DOC_HEADER.finditer(text))
    if not headers:
        raise ValueError("No Unity YAML document headers found (expected --- !u!<class> &<fileID>).")

    docs: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, header in enumerate(headers):
        start = header.end()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        body = text[start:end].strip()
        try:
            parsed = yaml.safe_load(body)
            if not isinstance(parsed, dict):
                raise ValueError("document root is not a mapping")
        except Exception as exc:
            errors.append(f"fileID={header.group(2)} classID={header.group(1)}: {exc}")
            continue
        docs.append({
            "class_id": int(header.group(1)),
            "file_id": str(header.group(2)),
            "root": parsed,
        })
    if not docs:
        raise ValueError("All Unity documents failed YAML parsing: " + "; ".join(errors))
    return docs, errors


def root_component(doc: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    root = as_mapping(doc.get("root"))
    if not root:
        return "", {}
    name = next(iter(root))
    return str(name), as_mapping(root.get(name))


def median_or_unknown(values: list[float]) -> float | str:
    return round(statistics.median(values), 6) if values else UNKNOWN


def curve_mean(value: Any) -> float | None:
    """Read a constant or constant-range serialized MinMaxCurve summary."""
    if isinstance(value, (int, float)):
        return float(value)
    item = as_mapping(value)
    for key in ("scalar", "constant", "m_Constant"):
        number = scalar_number(item.get(key))
        if number is not None:
            return number
    low = scalar_number(item.get("minScalar", item.get("constantMin", item.get("m_ConstantMin"))))
    high = scalar_number(item.get("maxScalar", item.get("constantMax", item.get("m_ConstantMax"))))
    if low is not None and high is not None:
        return (low + high) / 2.0
    return None


def get_transform_depths(docs: list[dict[str, Any]]) -> tuple[dict[str, int], int | str]:
    transform_to_gameobject: dict[str, str] = {}
    parent_transform: dict[str, str | None] = {}
    for doc in docs:
        name, data = root_component(doc)
        if name not in ("Transform", "RectTransform"):
            continue
        transform_id = doc["file_id"]
        gameobject_id = ref_id(data.get("m_GameObject"))
        if gameobject_id is not None:
            transform_to_gameobject[transform_id] = gameobject_id
        parent_transform[transform_id] = ref_id(data.get("m_Father"))

    depth_by_go: dict[str, int] = {}
    for transform_id, gameobject_id in transform_to_gameobject.items():
        depth, current, seen = 0, transform_id, set()
        while current and current not in seen:
            seen.add(current)
            parent = parent_transform.get(current)
            if not parent or parent == "0":
                break
            if parent not in transform_to_gameobject:
                break
            depth += 1
            current = parent
        depth_by_go[gameobject_id] = depth
    return depth_by_go, max(depth_by_go.values()) if depth_by_go else UNKNOWN


def hierarchy_gameobjects(docs: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    gameobjects: dict[str, dict[str, Any]] = {}
    transform_to_gameobject: dict[str, str] = {}
    for doc in docs:
        name, data = root_component(doc)
        if name == "GameObject":
            gameobjects[doc["file_id"]] = data
        elif name in ("Transform", "RectTransform"):
            go = ref_id(data.get("m_GameObject"))
            if go:
                transform_to_gameobject[doc["file_id"]] = go
    return gameobjects, transform_to_gameobject


def is_nonzero_reference(value: Any) -> bool:
    identifier = ref_id(value)
    if identifier is None:
        return False
    try:
        return int(identifier) != 0
    except ValueError:
        return identifier not in ("", "None")


def infer_reference_frame(
    docs: list[dict[str, Any]], gameobjects: dict[str, dict[str, Any]],
    transform_to_gameobject: dict[str, str],
) -> tuple[str, str, str, list[str]]:
    names: dict[str, str] = {
        go_id: str(data.get("m_Name", "")) for go_id, data in gameobjects.items()
    }
    static_ids: set[str] = set()
    for go_id, data in gameobjects.items():
        flags = scalar_number(data.get("m_StaticEditorFlags"))
        if flags is not None and flags != 0:
            static_ids.add(go_id)

    skybox = False
    for doc in docs:
        component, data = root_component(doc)
        if component == "RenderSettings" and is_nonzero_reference(data.get("m_SkyboxMaterial")):
            skybox = True
            break

    anchors: list[str] = []
    anchor_words = ("ground", "floor", "terrain", "grid", "horizon", "landmark", "reference frame", "world anchor")
    ground_like = False
    explicit_horizon = False
    for go_id, name in names.items():
        lowered = name.casefold()
        has_anchor_word = any(word in lowered for word in anchor_words)
        if has_anchor_word and (go_id in static_ids or "skybox" in lowered or "horizon" in lowered):
            anchors.append(f"GameObject '{name}'（静态标记或显式锚点名称）")
        if any(word in lowered for word in ("ground", "floor", "terrain")):
            ground_like = True
        if "horizon" in lowered:
            explicit_horizon = True

    if skybox:
        anchors.append("RenderSettings 引用了非空 Skybox Material")
    has_frame = "true" if anchors else UNKNOWN
    horizon = "true" if explicit_horizon or (skybox and ground_like) else UNKNOWN

    # A virtual nose is positive only if a nose-named object is a descendant
    # of a GameObject carrying a serialized Camera component.
    camera_go_ids: set[str] = set()
    for doc in docs:
        component, data = root_component(doc)
        if component == "Camera":
            owner = ref_id(data.get("m_GameObject"))
            if owner:
                camera_go_ids.add(owner)
    parent_by_go: dict[str, str | None] = {}
    for doc in docs:
        component, data = root_component(doc)
        if component in ("Transform", "RectTransform"):
            go = ref_id(data.get("m_GameObject"))
            parent_transform_id = ref_id(data.get("m_Father"))
            if go:
                parent_by_go[go] = transform_to_gameobject.get(parent_transform_id or "")
    nose_evidence = UNKNOWN
    for go_id, name in names.items():
        if "nose" not in name.casefold():
            continue
        current = parent_by_go.get(go_id)
        visited: set[str] = set()
        while current and current not in visited:
            if current in camera_go_ids:
                nose_evidence = "true"
                anchors.append(f"Camera 子对象 '{name}'（virtual nose 名称启发式）")
                break
            visited.add(current)
            current = parent_by_go.get(current)
        if nose_evidence == "true":
            break
    return has_frame, horizon, nose_evidence, anchors


def infer_controllability(docs: list[dict[str, Any]]) -> tuple[str, str]:
    script_text = "\n".join(
        json.dumps(doc.get("root", {}), ensure_ascii=False, default=str)
        for doc in docs if root_component(doc)[0] == "MonoBehaviour"
    )
    known_active = (
        "ContinuousMoveProvider", "ActionBasedContinuousMoveProvider", "DynamicMoveProvider",
        "TeleportationProvider", "SnapTurnProvider", "ContinuousTurnProvider",
        "LocomotionProvider", "CharacterControllerDriver", "StarterAssetsInputs",
        "PlayerInput", "XRController",
    )
    hits = [token for token in known_active if token.casefold() in script_text.casefold()]
    if hits:
        return "active", "; ".join(hits)
    return UNKNOWN, "未发现可识别的 locomotion/XR 控制脚本；外部 GUID 脚本未反编译。"


def triangle_count_from_mesh_docs(docs: list[dict[str, Any]]) -> tuple[int | str, str]:
    """Estimate triangles only when serialized mesh topology is inline/resolvable."""
    components = [root_component(doc) for doc in docs]
    mesh_filters = [data for name, data in components if name == "MeshFilter"]
    mesh_docs = {doc["file_id"]: root_component(doc)[1] for doc in docs if root_component(doc)[0] == "Mesh"}
    if not mesh_filters:
        return 0, "文件中没有 MeshFilter"

    total = 0
    for mesh_filter in mesh_filters:
        mesh_ref = ref_id(mesh_filter.get("m_Mesh"))
        if not mesh_ref or mesh_ref == "0" or mesh_ref not in mesh_docs:
            return UNKNOWN, "MeshFilter 引用的 Mesh 数据不在当前文件内；不根据 GUID/built-in fileID 猜三角形数"
        mesh = mesh_docs[mesh_ref]
        explicit = mesh.get("m_Triangles", mesh.get("triangles"))
        if isinstance(explicit, list):
            total += len(explicit) // 3
            continue
        submeshes = mesh.get("m_SubMeshes", mesh.get("subMeshes"))
        if isinstance(submeshes, list):
            for submesh in submeshes:
                item = as_mapping(submesh)
                index_count = scalar_number(item.get("indexCount"))
                topology = scalar_number(item.get("topology", 0))
                if index_count is None or topology != 0:
                    return UNKNOWN, "Mesh 子网格拓扑无法可靠解释为三角形索引"
                total += int(index_count) // 3
            continue
        return UNKNOWN, "Mesh 文档没有可解析的三角形索引/子网格拓扑"
    return total, "由当前文件内 Mesh 三角形索引估算（每 3 个 triangle indices 为 1 个三角形）"


def extract_features(path: Path, expected_duration_min: float | None = None) -> tuple[dict[str, Any], list[str]]:
    docs, yaml_errors = read_unity_documents(path)
    components = [root_component(doc) for doc in docs]
    gameobjects = {doc["file_id"]: data for doc in docs if root_component(doc)[0] == "GameObject" for data in [root_component(doc)[1]]}
    notes: list[str] = []
    if yaml_errors:
        notes.append(f"部分 YAML 对象解析失败，已忽略 {len(yaml_errors)} 个文档")
    if any(name == "PrefabInstance" for name, _ in components):
        notes.append("场景中的外部 PrefabInstance 未递归展开；只统计本文件内序列化对象")

    cameras: list[dict[str, Any]] = [data for name, data in components if name == "Camera"]
    enabled_cameras = [camera for camera in cameras if scalar_number(camera.get("m_Enabled", 1)) != 0]
    fovs, nears, fars = [], [], []
    for camera in enabled_cameras:
        for source, target in (("field of view", fovs), ("m_FieldOfView", fovs), ("near clip plane", nears),
                               ("m_NearClipPlane", nears), ("far clip plane", fars), ("m_FarClipPlane", fars)):
            number = scalar_number(camera.get(source))
            if number is not None:
                target.append(number)
    # Multiple enabled cameras are summarized by the median; all FOV values are also retained.
    fov_summary = median_or_unknown(fovs)
    fov_all = json.dumps([round(x, 6) for x in fovs], ensure_ascii=False)

    particle_systems = [data for name, data in components if name == "ParticleSystem"]
    rates: list[float] = []
    rate_unknown = False
    for particle in particle_systems:
        emission = as_mapping(particle.get("EmissionModule"))
        if scalar_number(emission.get("enabled", 1)) == 0:
            rates.append(0.0)
            continue
        rate = curve_mean(emission.get("rateOverTime"))
        multiplier = scalar_number(emission.get("rateOverTimeMultiplier"))
        if rate is None:
            rate_unknown = True
        else:
            rates.append(rate * (multiplier if multiplier is not None else 1.0))
    emission_total: float | int | str
    if not particle_systems:
        emission_total = 0
    elif rate_unknown or len(rates) != len(particle_systems):
        emission_total = UNKNOWN
    else:
        emission_total = round(sum(rates), 6)

    lights = [data for name, data in components if name == "Light"]
    light_counts = {"Directional": 0, "Point": 0, "Spot": 0, "Other": 0, "Unknown": 0}
    light_enum = {0: "Spot", 1: "Directional", 2: "Point"}
    for light in lights:
        raw_type = light.get("m_Type")
        num_type = scalar_number(raw_type)
        if num_type is not None:
            label = light_enum.get(int(num_type), "Other")
        elif isinstance(raw_type, str) and raw_type.casefold() in ("directional", "point", "spot"):
            label = raw_type.title()
        else:
            label = "Unknown"
        light_counts[label] += 1

    gameobjects, transform_to_gameobject = hierarchy_gameobjects(docs)
    depths, max_depth = get_transform_depths(docs)
    has_frame, has_horizon, has_nose, anchor_evidence = infer_reference_frame(docs, gameobjects, transform_to_gameobject)
    control, control_evidence = infer_controllability(docs)
    triangles, triangle_method = triangle_count_from_mesh_docs(docs)
    mesh_filter_count = sum(1 for name, _ in components if name == "MeshFilter")
    if triangle_method:
        notes.append("triangle_count: " + triangle_method)
    if not cameras:
        notes.append("未找到 Camera 组件")
    if not lights:
        notes.append("未找到 Light 组件")
    notes.append("dynamic_light_count 按 Light 组件数统计；不能据此判断灯光是否运行时动态/动画")
    notes.append("camera_fov/clip 为启用 Camera 对应值的中位数；完整 FOV 列表见 camera_fov_all")
    notes.append("particle_emission_rate 为所有 ParticleSystem 的 rateOverTime 常量/范围中点之和；不含 bursts、rateOverDistance 或运行时脚本修改")

    duration: float | str = expected_duration_min if expected_duration_min is not None else UNKNOWN
    row = {
        "scene_name": path.stem,
        "scene_path": str(path.resolve()),
        "scene_format": path.suffix.lower().lstrip("."),
        "camera_count": len(cameras),
        "camera_fov": fov_summary,
        "camera_fov_all": fov_all,
        "camera_near_clip": median_or_unknown(nears),
        "camera_far_clip": median_or_unknown(fars),
        "particle_system_count": len(particle_systems),
        "particle_emission_rate": emission_total,
        "dynamic_light_count": len(lights),
        "light_component_count": len(lights),
        "light_type_distribution": json.dumps(light_counts, ensure_ascii=False, sort_keys=True),
        "object_count": len(gameobjects),
        "transform_depth_max": max_depth,
        "triangle_count": triangles,
        "mesh_filter_count": mesh_filter_count,
        "mesh_renderer_count": sum(1 for name, _ in components if name == "MeshRenderer"),
        "has_reference_frame": has_frame,
        "has_horizon": has_horizon,
        "has_virtual_nose": has_nose,
        "reference_frame_evidence": "; ".join(anchor_evidence) if anchor_evidence else UNKNOWN,
        "controllability": control,
        "controllability_evidence": control_evidence,
        "expected_duration": duration,
        "expected_duration_unit": "minutes" if expected_duration_min is not None else UNKNOWN,
        "parser_notes": " | ".join(notes),
    }
    return row, yaml_errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, help="Unity text-serialized .unity or .prefab path")
    parser.add_argument("--output", required=True, help="Output CSV path (one row per scene)")
    parser.add_argument("--log", default=None, help="Log path; default is <project>/results/unity_scene_parse_log.txt")
    parser.add_argument("--expected-duration-min", type=float, default=None,
                        help="Optional user-provided duration in minutes; omitted means unknown")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    log_path = Path(args.log) if args.log else project_root / "results" / "unity_scene_parse_log.txt"
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_path), level=logging.INFO, encoding="utf-8",
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z",
        force=True,
    )
    scene = Path(args.scene)
    if not scene.is_absolute():
        scene = Path.cwd() / scene
    output = Path(args.output)
    if not output.is_absolute():
        output = Path.cwd() / output
    if scene.suffix.lower() not in (".unity", ".prefab"):
        logging.error("拒绝不支持的扩展名: %s", scene.suffix)
        print("ERROR: --scene must point to a .unity or .prefab file", file=sys.stderr)
        return 2
    try:
        logging.info("开始解析 scene=%s output=%s", scene, output)
        if not scene.is_file():
            raise FileNotFoundError(f"scene 文件不存在: {scene}")
        row, yaml_errors = extract_features(scene, args.expected_duration_min)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)
        for error in yaml_errors:
            logging.warning("部分文档跳过: %s", error)
        logging.info("解析成功 scene=%s fields=%d output=%s", scene, len(row), output)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        logging.exception("解析失败 scene=%s error=%s", scene, exc)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
