#!/usr/bin/env python3
"""
人物与车辆移动速度限制规则模块（Magic Director Stage）
=========================================================
职责：
1. 统一工程单位核查（1 世界单位 = 1m；若工程不一致则按比例换算，不自动缩放已有场景）。
2. 提供人物 6 档 + 车辆 3 档常用速度参考范围与默认值（并非物理极限，仅作自动生成默认基准与提示依据）。
3. 基于与前端预览/后端导出完全一致的三维世界路径曲线求值，采样计算整条轨迹弧长、瞬时峰值速度、转弯半径、
   前进/倒车方向（reverse 显式标记）、车轮滚动角与显式切镜/传送标记。
4. 区分随车乘员（仅限车内局部相对位移，不被车辆世界车速触发步行超限）与人物挂接道具/枪械（跟随父节点不独立限速脱离）。
5. 区分“自动生成/重算动画（校验并平滑修正或报构图冲突）”与“已有手工动画（仅检测警告，不擅自覆盖，批准修正后支持撤销与例外原因持久化）”。
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Dict, List, Optional, Tuple


WORLD_METERS_PER_UNIT: float = 1.0

# 人物与车辆常用参考速度范围及默认值（m/s；车辆另带 km/h = m/s * 3.6）
SPEED_LIMIT_TABLE: Dict[str, Dict[str, Any]] = {
    "char_slow_walk": {
        "label": "人物慢走",
        "kind": "character",
        "min_mps": 0.5,
        "max_mps": 0.9,
        "default_mps": 0.7,
        "default_accel_s": 0.35,
        "default_decel_s": 0.35,
    },
    "char_walk": {
        "label": "人物正常走",
        "kind": "character",
        "min_mps": 1.0,
        "max_mps": 1.5,
        "default_mps": 1.2,
        "default_accel_s": 0.40,
        "default_decel_s": 0.40,
    },
    "char_fast_walk": {
        "label": "人物快走",
        "kind": "character",
        "min_mps": 1.5,
        "max_mps": 2.0,
        "default_mps": 1.7,
        "default_accel_s": 0.45,
        "default_decel_s": 0.45,
    },
    "char_jog": {
        "label": "人物慢跑",
        "kind": "character",
        "min_mps": 2.0,
        "max_mps": 3.0,
        "default_mps": 2.5,
        "default_accel_s": 0.50,
        "default_decel_s": 0.50,
    },
    "char_run": {
        "label": "人物正常跑",
        "kind": "character",
        "min_mps": 3.0,
        "max_mps": 5.0,
        "default_mps": 4.0,
        "default_accel_s": 0.55,
        "default_decel_s": 0.55,
    },
    "char_sprint": {
        "label": "人物冲刺",
        "kind": "character",
        "min_mps": 5.0,
        "max_mps": 8.0,
        "default_mps": 6.0,
        "default_accel_s": 0.65,
        "default_decel_s": 0.60,
    },
    "veh_snow_crawl": {
        "label": "车辆雪地/林间缓行",
        "kind": "vehicle",
        "min_mps": 1.0,
        "max_mps": 3.0,
        "confirmed_road_max_mps": 4.0,  # 路况允许时确认可用 2–4 m/s
        "default_mps": 2.0,
        "default_accel_s": 0.85,
        "default_decel_s": 0.90,
        "turn_ref_radius_m": 9.0,
    },
    "veh_low_speed": {
        "label": "车辆低速行驶",
        "kind": "vehicle",
        "min_mps": 3.0,
        "max_mps": 8.0,
        "default_mps": 5.0,
        "default_accel_s": 1.10,
        "default_decel_s": 1.10,
        "turn_ref_radius_m": 12.0,
    },
    "veh_road": {
        "label": "车辆普通道路行驶",
        "kind": "vehicle",
        "min_mps": 8.0,
        "max_mps": 16.7,
        "default_mps": 12.0,
        "default_accel_s": 1.80,
        "default_decel_s": 1.60,
        "turn_ref_radius_m": 22.0,
    },
}

# 小队雪林穿行推荐一致慢走范围与微错峰起步参数
SQUAD_SNOW_FOREST_POLICY: Dict[str, Any] = {
    "motion_type": "char_slow_walk",
    "min_mps": 0.7,
    "max_mps": 1.0,
    "default_mps": 0.85,
    "stagger_step_s": 0.22,
}

# 人物肢体运动速度参考范围（1.70–1.72m 标准关节人偶局部参考系，仅用于校验与防抽搐，不干预球头人偶）
LIMB_SPEED_LIMIT_TABLE: Dict[str, Dict[str, Any]] = {
    "idle": {
        "label": "静止呼吸 / 微调",
        "hand_mps": (0.03, 0.35, 0.15),
        "foot_swing_mps": (0.03, 0.40, 0.15),
        "pelvis_y_mps": (0.02, 0.25, 0.10),
        "joint_deg_s": 75.0,
        "head_deg_s": 85.0,
        "torso_twist_deg_s": 70.0,
        "spine_deg_s": 50.0,
    },
    "calm": {
        "label": "日常缓动 / 坐立",
        "hand_mps": (0.35, 1.30, 0.80),
        "foot_swing_mps": (0.60, 2.40, 1.50),
        "pelvis_y_mps": (0.25, 0.70, 0.45),
        "joint_deg_s": 190.0,
        "head_deg_s": 130.0,
        "torso_twist_deg_s": 110.0,
        "spine_deg_s": 80.0,
    },
    "tactical_walk": {
        "label": "行进 / 戒备战术",
        "hand_mps": (0.60, 2.60, 1.45),
        "foot_swing_mps": (1.20, 3.80, 2.40),
        "pelvis_y_mps": (0.40, 1.20, 0.75),
        "joint_deg_s": 340.0,
        "head_deg_s": 240.0,
        "torso_twist_deg_s": 190.0,
        "spine_deg_s": 145.0,
    },
    "action_combat": {
        "label": "格斗 / 闪避 / 爆发动作",
        "hand_mps": (1.50, 8.20, 4.00),
        "foot_swing_mps": (2.20, 9.00, 5.20),
        "pelvis_y_mps": (0.65, 1.80, 1.15),
        "joint_deg_s": 780.0,
        "head_deg_s": 450.0,
        "torso_twist_deg_s": 360.0,
        "spine_deg_s": 260.0,
    },
}


# 可配置的运动学平滑参考参数（不等同于真实车辆动力学仿真）
DEFAULT_MOTION_DYNAMICS_CONFIG: Dict[str, Any] = {
    "tangent_align_threshold_mps": 0.08,
    "short_hop_distance_m": 1.8,
    "turn_min_factor": 0.55,
    "teleport_dt_threshold_s": 0.03,
    "teleport_jump_m": 0.35,
    "cruise_check_min_dur_s": 1.6,
    "is_full_vehicle_dynamics": False,
}

CONFLICT_RESOLUTION_OPTIONS: List[Dict[str, str]] = [
    {"id": "extend_duration", "label": "延长该段时间"},
    {"id": "raise_speed_limit", "label": "确认提高速度上限"},
    {"id": "adjust_path_or_target", "label": "允许调整路径或目标位置"},
]


def mps_to_kmh(v_mps: float) -> float:
    return float(v_mps) * 3.6


def resolve_scene_meters_per_unit(scene: Optional[Dict[str, Any]]) -> Tuple[float, bool]:
    """
    确认工程单位：默认 1 世界单位 = 1m。
    若 scene 显式指定不同的 meters_per_unit / unit_scale_m，则返回换算系数但不自动缩放已有场景。
    """
    if not isinstance(scene, dict):
        return WORLD_METERS_PER_UNIT, False
    raw = scene.get("meters_per_unit")
    if raw is None:
        raw = scene.get("unit_scale_m")
    if raw is None and isinstance(scene.get("units"), dict):
        raw = scene["units"].get("meters_per_unit")
    if raw is None:
        return WORLD_METERS_PER_UNIT, False
    try:
        val = float(raw)
        if not math.isfinite(val) or val <= 1e-6:
            return WORLD_METERS_PER_UNIT, False
        return val, abs(val - 1.0) > 1e-4
    except (TypeError, ValueError):
        return WORLD_METERS_PER_UNIT, False


def _eval_easing_py(alpha: float, mode: str = "linear") -> float:
    u = max(0.0, min(1.0, float(alpha)))
    m = str(mode or "linear").lower().strip()
    if m in ("hold", "step", "constant"):
        return 0.0 if u < 0.999999 else 1.0
    if m == "ease_in":
        return u * u
    if m == "ease_out":
        inv = 1.0 - u
        return 1.0 - inv * inv
    if m in ("ease_in_out", "smoothstep", "smooth"):
        return u * u * (3.0 - 2.0 * u)
    return u


def _normalize_interp_mode(mode: Any, default_mode: str = "linear") -> str:
    m = str(mode or default_mode).lower().strip()
    if m in ("continuous", "hermite", "continuous_smooth", "catmull_rom"):
        return "continuous"
    if m in ("ease_in_out", "smoothstep", "smooth"):
        return "ease_in_out"
    if m in ("hold", "step", "constant"):
        return "hold"
    if m == "ease_in":
        return "ease_in"
    if m == "ease_out":
        return "ease_out"
    return "linear"


def _eval_hermite_channel_py(
    kfs: List[Dict[str, Any]],
    times: List[float],
    vals: List[float],
    modes: List[str],
    seg_idx: int,
    u: float,
) -> float:
    n = len(vals)
    i = max(0, min(n - 2, int(seg_idx)))
    u_c = max(0.0, min(1.0, float(u)))
    y0 = float(vals[i])
    y1 = float(vals[i + 1])
    hi = float(times[i + 1]) - float(times[i])
    mode_i = modes[i]
    if hi <= 1e-4 or abs(y1 - y0) <= 1e-7:
        return y0 if mode_i == "hold" and u_c < 0.999999 else (y0 * (1.0 - u_c) + y1 * u_c)
    if mode_i != "continuous":
        s = _eval_easing_py(u_c, mode_i)
        return y0 * (1.0 - s) + y1 * s

    def _endpoint_slope(idx: int) -> float:
        if idx <= 0 or idx >= n - 1:
            return 0.0
        if bool(kfs[idx].get("stop", False)):
            return 0.0
        if modes[idx - 1] == "hold" or modes[idx] == "hold":
            return 0.0
        h_prev = float(times[idx]) - float(times[idx - 1])
        h_next = float(times[idx + 1]) - float(times[idx])
        if h_prev <= 1e-4 or h_next <= 1e-4:
            return 0.0
        d_prev = (float(vals[idx]) - float(vals[idx - 1])) / h_prev
        d_next = (float(vals[idx + 1]) - float(vals[idx])) / h_next
        if d_prev * d_next <= 0.0:
            return 0.0
        denom = h_prev + h_next
        slope = ((2.0 * h_next + h_prev) * d_prev + h_prev * d_next) / denom if h_prev < h_next else (
            (h_next * d_prev + (2.0 * h_prev + h_next) * d_next) / denom
        )
        # Fritsch-Carlson / Steffen 单调限幅，防止曲线中间无故反冲超调
        p = ((d_prev * h_next) + (d_next * h_prev)) / denom
        if d_prev > 0.0 and d_next > 0.0:
            slope = min(abs(p), 2.0 * min(abs(d_prev), abs(d_next)))
        elif d_prev < 0.0 and d_next < 0.0:
            slope = -min(abs(p), 2.0 * min(abs(d_prev), abs(d_next)))
        return slope

    m0 = _endpoint_slope(i)
    m1 = _endpoint_slope(i + 1)
    u2 = u_c * u_c
    u3 = u2 * u_c
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u_c
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    return h00 * y0 + h10 * (hi * m0) + h01 * y1 + h11 * (hi * m1)


def is_vehicle_object(obj: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(obj, dict):
        return False
    otype = str(obj.get("type") or "").lower()
    if otype == "uaz469":
        return True
    role = str(obj.get("arch_role") or "").lower()
    if role == "vehicle":
        return True
    oid = str(obj.get("id") or "").lower()
    name = str(obj.get("name") or "")
    if oid.startswith(("veh_", "car_", "jeep_", "van_", "truck_", "uaz")):
        return True
    parts = obj.get("parts")
    if isinstance(parts, list) and any(str(p.get("id") or "").startswith("wheel_") for p in parts if isinstance(p, dict)):
        return True
    if any(k in name for k in ("吉普", "越野车", "面包车", "卡车", "汽车", "轿车", "皮卡", "装甲车", "Jeep", "UAZ", "Van", "Truck")):
        return True
    return False


def is_prop_bound_active(obj: Optional[Dict[str, Any]], t: Optional[float] = None) -> bool:
    if not isinstance(obj, dict) or obj.get("type") == "character":
        return False
    pb = obj.get("parent_binding")
    if isinstance(pb, dict) and pb.get("enabled") is not False and str(pb.get("target_id") or "").strip():
        if t is None:
            return True
        st = pb.get("start_t")
        rt = pb.get("release_t")
        if st is not None and float(t) < float(st) - 1e-6:
            return False
        if rt is not None and float(t) >= float(rt) - 1e-6:
            return False
        return True
    if obj.get("bound_char_id") and obj.get("follow_bound_char", True) is not False:
        return True
    return False


def is_mounted_vehicle_rider(obj: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(obj, dict) or obj.get("type") != "character":
        return False
    return bool(obj.get("mounted_vehicle_id") and obj.get("follow_mounted_vehicle") is not False)


def infer_motion_category(
    obj: Optional[Dict[str, Any]],
    scene: Optional[Dict[str, Any]] = None,
    is_camera: bool = False,
    terrain_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    识别对象运动类型与对应推荐速度范围（并非物理极限）。
    返回: category_key, label, kind, min_mps, max_mps, effective_max_mps, default_mps, exception_reason, ...
    """
    if is_camera:
        return {
            "category": "camera",
            "label": "摄影机运镜",
            "kind": "camera",
            "min_mps": 0.0,
            "max_mps": None,
            "effective_max_mps": None,
            "default_mps": None,
            "min_kmh": 0.0,
            "max_kmh": None,
            "effective_max_kmh": None,
            "exception_reason": "",
            "confirmed_snow_road_4mps": False,
        }

    obj = obj or {}
    otype = str(obj.get("type") or "").lower()
    raw_exc = obj.get("speed_exception") if isinstance(obj.get("speed_exception"), dict) else {}
    explicit_mt = str(
        obj.get("motion_type")
        or raw_exc.get("motion_type")
        or obj.get("speed_profile")
        or ""
    ).strip()

    scene_text = " ".join([
        str((scene or {}).get("name") or ""),
        str((scene or {}).get("title") or ""),
        str(terrain_hint or (scene or {}).get("terrain") or ""),
        str(obj.get("terrain") or ""),
    ]).lower()
    obj_text = " ".join([
        str(obj.get("name") or ""),
        str(obj.get("id") or ""),
        str(obj.get("pose_name") or ""),
        str(obj.get("motion_hint") or ""),
    ]).lower()
    has_snow_forest = any(k in scene_text or k in obj_text for k in ("雪", "冰", "雪地", "雪林", "林间", "针叶林", "taiga", "snow", "forest", "forest_style"))

    is_veh = is_vehicle_object(obj)
    cat = explicit_mt if explicit_mt in SPEED_LIMIT_TABLE else ""

    if not cat:
        if otype == "character":
            if any(k in obj_text for k in ("冲刺", "百米", "sprint")):
                cat = "char_sprint"
            elif any(k in obj_text for k in ("逃跑", "狂奔", "追逐", "奔跑", "run", "chase", "fugitive", "hunter")):
                cat = "char_run"
            elif any(k in obj_text for k in ("慢跑", "小跑", "jog")):
                cat = "char_jog"
            elif any(k in obj_text for k in ("快走", "急行", "fast_walk")):
                cat = "char_fast_walk"
            elif any(k in obj_text for k in ("慢走", "潜行", "小队", "雪林", "squad", "slow_walk")) or (has_snow_forest and "squad" in obj_text):
                cat = "char_slow_walk"
            else:
                cat = "char_walk"
        elif is_veh:
            if any(k in obj_text for k in ("公路", "大路", "国道", "road", "highway")):
                cat = "veh_road"
            elif any(k in obj_text for k in ("低速", "院外道路", "营地路", "low_speed")):
                cat = "veh_low_speed"
            elif has_snow_forest or any(k in obj_text for k in ("雪", "林间", "uaz", "吉普", "面包车", "jeep", "van")):
                cat = "veh_snow_crawl"
            else:
                cat = "veh_low_speed"
        else:
            # 非车辆静态/辅助道具或独立道具（如被搬运木箱、烟团等）不做人车速度强制分类，除非显式指定 motion_type
            return {
                "category": "prop",
                "label": "普通道具",
                "kind": "prop",
                "min_mps": 0.0,
                "max_mps": None,
                "effective_max_mps": None,
                "default_mps": None,
                "min_kmh": 0.0,
                "max_kmh": None,
                "effective_max_kmh": None,
                "exception_reason": str(raw_exc.get("reason") or ""),
                "confirmed_snow_road_4mps": False,
            }

    spec = SPEED_LIMIT_TABLE[cat]
    eff_max = float(spec["max_mps"])
    confirmed_snow_road_4mps = False
    if cat == "veh_snow_crawl" and (
        bool(obj.get("allow_snow_4mps"))
        or bool((scene or {}).get("allow_snow_jeep_4mps"))
        or bool(raw_exc.get("allow_snow_4mps"))
    ):
        eff_max = max(eff_max, float(spec.get("confirmed_road_max_mps", 4.0)))
        confirmed_snow_road_4mps = True

    exc_max = raw_exc.get("max_mps")
    exc_reason = str(raw_exc.get("reason") or "").strip()
    if exc_max is not None:
        try:
            exc_val = float(exc_max)
            if math.isfinite(exc_val) and exc_val > eff_max:
                eff_max = exc_val
        except (TypeError, ValueError):
            pass

    return {
        "category": cat,
        "label": spec["label"],
        "kind": spec["kind"],
        "min_mps": float(spec["min_mps"]),
        "max_mps": float(spec["max_mps"]),
        "effective_max_mps": round(eff_max, 3),
        "default_mps": float(spec["default_mps"]),
        "min_kmh": round(mps_to_kmh(spec["min_mps"]), 1),
        "max_kmh": round(mps_to_kmh(spec["max_mps"]), 1),
        "effective_max_kmh": round(mps_to_kmh(eff_max), 1),
        "default_accel_s": float(spec["default_accel_s"]),
        "default_decel_s": float(spec["default_decel_s"]),
        "turn_ref_radius_m": float(spec.get("turn_ref_radius_m", 6.0)),
        "exception_reason": exc_reason,
        "confirmed_snow_road_4mps": confirmed_snow_road_4mps,
    }


def _dedup_sorted_kfs(raw_kfs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    raw_sorted = sorted([k for k in (raw_kfs or []) if isinstance(k, dict)], key=lambda x: float(x.get("t", 0.0)))
    kfs: List[Dict[str, Any]] = []
    for k in raw_sorted:
        kt = float(k.get("t", 0.0))
        if kfs and abs(kt - float(kfs[-1].get("t", 0.0))) < 1e-5:
            kfs[-1] = k
        else:
            kfs.append(k)
    return kfs


def eval_track_state_raw(
    obj: Dict[str, Any],
    t: float,
    is_camera: bool = False,
    scene: Optional[Dict[str, Any]] = None,
    continuity_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    纯关键帧轨道位置/朝向求值（不含父挂点覆盖）：
    - 若摄影机开启 camera_speed_curve 或物体/角色开启 motion_protection，统一走实际导出与播放相同的连续性求值器；
    - 未开启保护的旧轨道保持原有 Hermite/easing 兼容。
    """
    if is_camera:
        cc = obj.get("camera_speed_curve")
        if isinstance(cc, dict) and cc.get("enabled"):
            try:
                from director_engine import evaluate_camera_state
                st = evaluate_camera_state(obj, t)
                return {"pos": list(st["pos"]), "rot": [0.0, 0.0, float(st.get("roll_deg", 0.0))]}
            except Exception:
                pass
    else:
        mp = obj.get("motion_protection")
        if (
            isinstance(mp, dict)
            and mp.get("enabled")
            and str(mp.get("mode") or "smooth_variable").lower().strip() != "manual"
            and len(obj.get("keyframes") or []) >= 2
        ):
            try:
                from motion_continuity import eval_track_continuity_at_time
                pl = continuity_plan
                if pl is None:
                    try:
                        from director_engine import _get_cached_track_continuity_plan
                        pl = _get_cached_track_continuity_plan(obj, scene)
                    except Exception:
                        from motion_continuity import build_track_continuity_plan
                        pl = build_track_continuity_plan(obj, scene)
                if isinstance(pl, dict) and pl.get("enabled"):
                    ev = eval_track_continuity_at_time(pl, t)
                    base_rot_loc = [float(x) for x in (obj.get("rot") or [0.0, 0.0, 0.0])[:3]]
                    return {
                        "pos": list(ev["pos"]),
                        "rot": [base_rot_loc[0], float(ev["rot_y"]), base_rot_loc[2]],
                        "world_pos": list(ev["world_pos"]) if "world_pos" in ev else None,
                        "world_rot_y": float(ev["world_rot_y"]) if "world_rot_y" in ev else None,
                    }
            except Exception:
                pass

    raw_kfs = obj.get("camera_keyframes") if is_camera else obj.get("keyframes")
    kfs = _dedup_sorted_kfs(raw_kfs or [])
    base_pos = [float(x) for x in (obj.get("pos") or [0.0, 0.0, 0.0])[:3]]
    base_rot = [float(x) for x in (obj.get("rot") or [0.0, 0.0, 0.0])[:3]]
    if not kfs:
        return {"pos": base_pos, "rot": base_rot}

    def _kp(k: Dict[str, Any]) -> List[float]:
        p = k.get("pos")
        return [float(p[0]), float(p[1]), float(p[2])] if isinstance(p, (list, tuple)) and len(p) >= 3 else list(base_pos)

    def _kr(k: Dict[str, Any]) -> List[float]:
        r = k.get("rot")
        if isinstance(r, (list, tuple)) and len(r) >= 3:
            return [float(r[0]), float(r[1]), float(r[2])]
        if k.get("rot_y") is not None:
            return [base_rot[0], float(k["rot_y"]), base_rot[2]]
        return list(base_rot)

    if len(kfs) == 1 or t <= float(kfs[0].get("t", 0.0)):
        return {"pos": _kp(kfs[0]), "rot": _kr(kfs[0])}
    if t >= float(kfs[-1].get("t", 0.0)):
        return {"pos": _kp(kfs[-1]), "rot": _kr(kfs[-1])}

    is_char = (not is_camera) and (str(obj.get("type") or "") == "character")
    default_mode = "linear" if is_char else ("continuous" if is_camera else "linear")
    times = [float(k.get("t", 0.0)) for k in kfs]
    modes_pos = [_normalize_interp_mode(k.get("interp_pos") or k.get("interp") or default_mode, default_mode) for k in kfs]
    modes_rot = [_normalize_interp_mode(k.get("interp_rot") or k.get("interp") or default_mode, default_mode) for k in kfs]
    px = [_kp(k)[0] for k in kfs]
    py = [_kp(k)[1] for k in kfs]
    pz = [_kp(k)[2] for k in kfs]

    unwrapped_rots: List[List[float]] = [_kr(kfs[0])]
    for i in range(1, len(kfs)):
        prev_r = unwrapped_rots[-1]
        cur_r = _kr(kfs[i])
        unwrapped_rots.append([
            prev_r[j] + (((cur_r[j] - prev_r[j]) % 360.0 + 540.0) % 360.0 - 180.0)
            for j in range(3)
        ])
    ry = [r[1] for r in unwrapped_rots]

    for i in range(len(kfs) - 1):
        t0, t1 = times[i], times[i + 1]
        if t0 <= t <= t1:
            dt = t1 - t0
            alpha = (t - t0) / dt if dt > 1e-6 else 0.0
            pos_out = [
                _eval_hermite_channel_py(kfs, times, px, modes_pos, i, alpha),
                _eval_hermite_channel_py(kfs, times, py, modes_pos, i, alpha),
                _eval_hermite_channel_py(kfs, times, pz, modes_pos, i, alpha),
            ]
            ry_out = _eval_hermite_channel_py(kfs, times, ry, modes_rot, i, alpha)
            return {"pos": pos_out, "rot": [base_rot[0], ry_out, base_rot[2]]}

    return {"pos": _kp(kfs[-1]), "rot": _kr(kfs[-1])}


def detect_wheel_radius_from_object(obj: Dict[str, Any]) -> Optional[float]:
    parts = obj.get("parts") or []
    if isinstance(parts, list):
        radii = []
        for p in parts:
            if not isinstance(p, dict):
                continue
            pid = str(p.get("id") or "").lower()
            if pid.startswith("wheel_") or p.get("is_wheel") is True:
                sz = p.get("size") or []
                if len(sz) >= 3:
                    # 车轮柱体沿 X 旋转 90° 后 [dia_y, width, dia_z]
                    radii.append(0.5 * max(float(sz[0]), float(sz[2])))
        if radii:
            return sum(radii) / len(radii)
    if str(obj.get("type") or "").lower() == "uaz469":
        # UAZ469 默认轮径 0.70m -> radius 0.35m，按尺寸 Y 比例微调
        sy = float((obj.get("size") or [1.78, 2.02, 4.02])[1] or 2.02)
        return 0.35 * (sy / 2.02)
    return None


def _resolve_track_source(obj: Dict[str, Any], scene: Optional[Dict[str, Any]] = None) -> str:
    ms = str(obj.get("motion_source") or "").lower().strip()
    if ms in ("auto", "manual"):
        return ms
    if obj.get("manual_keyframes") is True:
        return "manual"
    if str(obj.get("repro_source") or "").strip():
        return "auto"
    if isinstance(scene, dict) and str(scene.get("repro_source") or "").strip():
        return "auto"
    # 保守默认：未标记 repro_source / motion_source="auto" 的现有关键帧均按手工轨道对待（仅检测提示，不擅自覆盖）
    return "manual"


def _compute_turn_radius_3pt(p0: List[float], p1: List[float], p2: List[float]) -> Optional[float]:
    """根据三点水平面 XZ 弧圆半径（m）估算转弯半径；近似直线返回 None。"""
    ax, az = p1[0] - p0[0], p1[2] - p0[2]
    bx, bz = p2[0] - p1[0], p2[2] - p1[2]
    la = math.hypot(ax, az)
    lb = math.hypot(bx, bz)
    if la < 0.08 or lb < 0.08:
        return None
    cross = abs(ax * bz - az * bx)
    lc = math.hypot(p2[0] - p0[0], p2[2] - p0[2])
    if cross < 1e-4 or lc < 0.08:
        return None
    radius = (la * lb * lc) / (2.0 * cross)
    return radius if radius < 200.0 else None


def analyze_track_speed_profile(
    obj: Dict[str, Any],
    scene: Optional[Dict[str, Any]] = None,
    is_camera: bool = False,
    samples_per_sec: int = 32,
) -> Dict[str, Any]:
    """
    沿最终插值后的整条 3D 世界曲线采样分析轨迹速度：
    - 使用实际 3D 世界弧长 / 时间（含 meters_per_unit 单位换算）
    - 识别曲线内部峰值超速（不仅看关键帧平均速度）
    - 区分起停/等待/短距离平滑低峰 vs 稳定巡航段下限
    - 识别乘车随动、人物挂接附件、倒车 reverse、显式切镜/传送与未标记瞬移
    - 为车辆额外提供 km/h 及理论车轮滚动累计角度
    """
    mpu, unit_mismatch = resolve_scene_meters_per_unit(scene)
    obj = obj or {}
    oid = "__camera__" if is_camera else str(obj.get("id") or "obj")
    oname = "摄影机" if is_camera else str(obj.get("name") or oid)
    cat_info = infer_motion_category(obj, scene, is_camera=is_camera)
    source = _resolve_track_source(obj, scene)

    raw_kfs = obj.get("camera_keyframes") if is_camera else obj.get("keyframes")
    kfs = _dedup_sorted_kfs(raw_kfs or [])

    is_bound = (not is_camera) and is_prop_bound_active(obj)
    is_mounted = (not is_camera) and is_mounted_vehicle_rider(obj)
    vehicle_parent_world_speed_mps = 0.0
    if is_mounted and isinstance(scene, dict):
        mvid = str(obj.get("mounted_vehicle_id") or "")
        for other in (scene.get("objects") or []):
            if str(other.get("id") or "") == mvid:
                v_rep = analyze_track_speed_profile(other, scene=scene, is_camera=False, samples_per_sec=12)
                vehicle_parent_world_speed_mps = float(v_rep.get("peak_speed_mps") or 0.0)
                break

    if is_bound:
        return {
            "obj_id": oid,
            "obj_name": oname,
            "motion_source": source,
            "is_bound_attachment": True,
            "is_mounted_rider": False,
            "meters_per_unit": mpu,
            "unit_mismatch": unit_mismatch,
            "category": cat_info,
            "status": "exempt_bound",
            "violations": [],
            "warnings": [],
            "total_arc_m": 0.0,
            "total_duration_s": 0.0,
            "overall_avg_mps": 0.0,
            "peak_speed_mps": 0.0,
            "peak_speed_kmh": 0.0,
            "peak_time_s": 0.0,
            "segments": [],
            "direction": "forward",
            "note": "已挂接绑定到人物，按绑定变换跟随求值，不独立限速以免脱离人物。",
        }

    # 检查随车乘员是否仅做车内局部偏移（|x|<1.6, |z|<2.2），此时仅限车内相对运动，不将车辆世界位移算作人物步行速度
    mounted_local_only = False
    if is_mounted:
        if not kfs:
            mounted_local_only = True
        else:
            mounted_local_only = all(
                abs(float((k.get("pos") or [0, 0, 0])[0])) < 1.6 and abs(float((k.get("pos") or [0, 0, 0])[2])) < 2.2
                for k in kfs
            )

    if len(kfs) < 2:
        return {
            "obj_id": oid,
            "obj_name": oname,
            "motion_source": source,
            "is_bound_attachment": False,
            "is_mounted_rider": is_mounted,
            "mounted_parent_world_peak_mps": round(vehicle_parent_world_speed_mps, 3),
            "meters_per_unit": mpu,
            "unit_mismatch": unit_mismatch,
            "category": cat_info,
            "status": "ok",
            "violations": [],
            "warnings": [],
            "total_arc_m": 0.0,
            "total_duration_s": 0.0,
            "overall_avg_mps": 0.0,
            "peak_speed_mps": 0.0,
            "peak_speed_kmh": 0.0,
            "peak_time_s": 0.0,
            "segments": [],
            "direction": "forward",
            "note": "随车乘员（车内局部姿态求值，车辆世界位移不计入步行速度限制）" if is_mounted else "",
        }

    eff_max = cat_info.get("effective_max_mps")
    rec_min = float(cat_info.get("min_mps") or 0.0)
    is_veh = (cat_info.get("kind") == "vehicle")
    wheel_radius = detect_wheel_radius_from_object(obj) if is_veh else None

    segments_out: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    total_arc_m = 0.0
    total_signed_arc_m = 0.0
    global_peak_v = 0.0
    global_peak_t = float(kfs[0].get("t", 0.0))
    any_reverse = False

    # 如果摄影机启用了独立 camera_speed_curve，优先按 director_engine 解算出的真实时刻采样摄影机轨迹
    cam_eval_fn = None
    if is_camera and isinstance(scene, dict) and isinstance(scene.get("camera_speed_curve"), dict) and scene["camera_speed_curve"].get("enabled"):
        try:
            from director_engine import _eval_camera_at_time_exact
            cam_eval_fn = lambda tt: _eval_camera_at_time_exact(scene, tt)
        except Exception:
            cam_eval_fn = None

    has_wp_sp = False
    obj_map_sp = None
    _eval_wp_sp = None
    mp_plan_sp = None
    if not is_camera:
        mp_cfg = obj.get("motion_protection")
        if (
            isinstance(mp_cfg, dict)
            and mp_cfg.get("enabled")
            and str(mp_cfg.get("mode") or "smooth_variable").lower().strip() != "manual"
            and len(kfs) >= 2
        ):
            try:
                from director_engine import _get_cached_track_continuity_plan
                mp_plan_sp = _get_cached_track_continuity_plan(obj, scene)
            except Exception:
                try:
                    from motion_continuity import build_track_continuity_plan
                    mp_plan_sp = build_track_continuity_plan(obj, scene)
                except Exception:
                    mp_plan_sp = None
    if (not is_camera) and (not mounted_local_only):
        try:
            from motion_continuity import _has_object_world_prefix, _eval_waypoint_world_transform_mc
            has_wp_sp = _has_object_world_prefix(obj, scene)
            if has_wp_sp:
                _eval_wp_sp = _eval_waypoint_world_transform_mc
                if isinstance(scene, dict):
                    obj_map_sp = {
                        str(o.get("id")): o
                        for o in (scene.get("objects") or [])
                        if isinstance(o, dict) and o.get("id")
                    }
        except Exception:
            has_wp_sp = False

    def _kf_rot_y(k: Dict[str, Any]) -> float:
        r = k.get("rot")
        if isinstance(r, (list, tuple)) and len(r) >= 2:
            return float(r[1] or 0.0)
        return float(k.get("rot_y") or 0.0)

    for i in range(len(kfs) - 1):
        k0, k1 = kfs[i], kfs[i + 1]
        t0, t1 = float(k0.get("t", 0.0)), float(k1.get("t", 0.0))
        dt = t1 - t0
        p0_raw = [float(x) for x in (k0.get("pos") or [0.0, 0.0, 0.0])[:3]]
        p1_raw = [float(x) for x in (k1.get("pos") or [0.0, 0.0, 0.0])[:3]]
        if has_wp_sp and _eval_wp_sp is not None:
            p0_w, _ = _eval_wp_sp(obj, p0_raw, _kf_rot_y(k0), t0, scene=scene, obj_map=obj_map_sp)
            p1_w, _ = _eval_wp_sp(obj, p1_raw, _kf_rot_y(k1), t1, scene=scene, obj_map=obj_map_sp)
            chord_m = math.dist(p0_w, p1_w) * mpu
        else:
            chord_m = math.dist(p0_raw, p1_raw) * mpu

        interp_m = _normalize_interp_mode(k0.get("interp_pos") or k0.get("interp") or ("continuous" if is_camera else "linear"))
        explicit_cut = bool(
            k0.get("cut") or k1.get("cut") or k0.get("teleport") or k1.get("teleport") or interp_m == "hold"
        )
        tiny_dt_jump = (
            dt <= DEFAULT_MOTION_DYNAMICS_CONFIG["teleport_dt_threshold_s"]
            and chord_m >= DEFAULT_MOTION_DYNAMICS_CONFIG["teleport_jump_m"]
        )

        if explicit_cut or tiny_dt_jump:
            seg_rec = {
                "index": i,
                "t0": round(t0, 3),
                "t1": round(t1, 3),
                "dt": round(max(0.0, dt), 3),
                "arc_distance_m": round(chord_m, 3),
                "chord_distance_m": round(chord_m, 3),
                "avg_speed_mps": 0.0 if explicit_cut else round(chord_m / max(1e-4, dt), 3),
                "peak_speed_mps": 0.0,
                "peak_speed_kmh": 0.0,
                "peak_time_s": round(t1, 3),
                "interp_pos": interp_m,
                "is_cut_or_teleport": True,
                "explicit_cut": explicit_cut,
                "direction": "reverse" if (k0.get("reverse") or k1.get("reverse")) else "forward",
            }
            segments_out.append(seg_rec)
            if tiny_dt_jump and not explicit_cut:
                msg_t = {
                    "type": "unmarked_teleport",
                    "segment_index": i,
                    "t0": round(t0, 3),
                    "t1": round(t1, 3),
                    "distance_m": round(chord_m, 3),
                    "message": f"{oname} 在 {t0:.2f}s→{t1:.2f}s 发生 {chord_m:.2f}m 未标记瞬间跳变（需显式标记 cut/teleport）",
                }
                (warnings if source == "manual" else violations).append(msg_t)
            continue

        if dt <= 1e-5:
            continue

        n_steps = max(20, min(160, int(math.ceil(dt * max(16, samples_per_sec)))))
        pts_m: List[List[float]] = []
        ts_list: List[float] = []
        yaws_deg: List[float] = []

        for s in range(n_steps + 1):
            ts = t0 + (dt * s) / n_steps
            if cam_eval_fn is not None:
                ev = cam_eval_fn(ts)
                ry_val = 0.0
                p_eval = ev["pos"]
            else:
                ev = eval_track_state_raw(obj, ts, is_camera=is_camera, scene=scene, continuity_plan=mp_plan_sp)
                ry_val = float((ev.get("rot") or [0.0, 0.0, 0.0])[1])
                if has_wp_sp and ev.get("world_pos") is not None:
                    p_eval = ev["world_pos"]
                    if ev.get("world_rot_y") is not None:
                        ry_val = float(ev["world_rot_y"])
                elif has_wp_sp and _eval_wp_sp is not None:
                    p_eval, ry_val = _eval_wp_sp(obj, ev["pos"], ry_val, ts, scene=scene, obj_map=obj_map_sp)
                else:
                    p_eval = ev["pos"]
            p_m = [float(p_eval[k]) * mpu for k in range(3)]
            pts_m.append(p_m)
            ts_list.append(ts)
            yaws_deg.append(ry_val)

        seg_arc_m = 0.0
        seg_signed_arc_m = 0.0
        seg_peak_v = 0.0
        seg_peak_t = t0
        rev_votes = 0
        fwd_votes = 0
        sample_speeds: List[Tuple[float, float]] = []
        min_turn_r: Optional[float] = None

        for s in range(n_steps):
            dp = [pts_m[s + 1][k] - pts_m[s][k] for k in range(3)]
            d_arc = math.sqrt(dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2])
            step_dt = ts_list[s + 1] - ts_list[s]
            inst_v = d_arc / max(1e-6, step_dt)
            seg_arc_m += d_arc

            # 检查车体局部前方 (-Z rotated by rot_y) 与位移水平分量夹角
            signed_step = d_arc
            d_horiz = math.hypot(dp[0], dp[2])
            if is_veh and d_horiz > 1e-4:
                yaw_mid = math.radians(0.5 * (yaws_deg[s] + yaws_deg[s + 1]))
                fwd_x = -math.sin(yaw_mid)
                fwd_z = -math.cos(yaw_mid)
                dot_fwd = (dp[0] * fwd_x + dp[2] * fwd_z) / d_horiz
                if bool(k0.get("reverse")) or (dot_fwd < -0.55 and not obj.get("align_tangent_y")):
                    signed_step = -d_arc
                    rev_votes += 1
                else:
                    fwd_votes += 1
            seg_signed_arc_m += signed_step

            t_mid = 0.5 * (ts_list[s] + ts_list[s + 1])
            sample_speeds.append((t_mid, inst_v))
            if inst_v > seg_peak_v:
                seg_peak_v = inst_v
                seg_peak_t = t_mid

            if s >= 1:
                r_turn = _compute_turn_radius_3pt(pts_m[s - 1], pts_m[s], pts_m[s + 1])
                if r_turn is not None and (min_turn_r is None or r_turn < min_turn_r):
                    min_turn_r = r_turn

        seg_avg_v = seg_arc_m / dt
        is_reverse = bool(k0.get("reverse") or (is_veh and rev_votes > fwd_votes and seg_arc_m > 0.15))
        if is_reverse:
            any_reverse = True

        total_arc_m += seg_arc_m
        total_signed_arc_m += (-abs(seg_arc_m) if is_reverse else abs(seg_signed_arc_m))
        if seg_peak_v > global_peak_v:
            global_peak_v = seg_peak_v
            global_peak_t = seg_peak_t

        # 提取巡航中段最小速度（跳过前/后加减速缓变区）
        cruise_speeds = [
            v for tm, v in sample_speeds
            if (tm - t0) >= min(0.65, dt * 0.25) and (t1 - tm) >= min(0.65, dt * 0.25)
        ]
        cruise_min_v = min(cruise_speeds) if cruise_speeds else seg_avg_v

        seg_rec: Dict[str, Any] = {
            "index": i,
            "t0": round(t0, 3),
            "t1": round(t1, 3),
            "dt": round(dt, 3),
            "arc_distance_m": round(seg_arc_m, 3),
            "chord_distance_m": round(chord_m, 3),
            "avg_speed_mps": round(seg_avg_v, 3),
            "avg_speed_kmh": round(mps_to_kmh(seg_avg_v), 1),
            "peak_speed_mps": round(seg_peak_v, 3),
            "peak_speed_kmh": round(mps_to_kmh(seg_peak_v), 1),
            "peak_time_s": round(seg_peak_t, 3),
            "cruise_min_speed_mps": round(cruise_min_v, 3),
            "min_turn_radius_m": round(min_turn_r, 2) if min_turn_r is not None else None,
            "interp_pos": interp_m,
            "is_cut_or_teleport": False,
            "direction": "reverse" if is_reverse else "forward",
            "reverse": is_reverse,
        }
        segments_out.append(seg_rec)

        if is_veh and is_reverse and not bool(k0.get("reverse")):
            warnings.append({
                "type": "implicit_reverse_motion",
                "segment_index": i,
                "t0": round(t0, 3),
                "t1": round(t1, 3),
                "message": f"{oname} 第#{i+1}段 ({t0:.2f}s–{t1:.2f}s) 行进方向与车头朝向相反，建议显式标记 reverse=True 区分倒车",
            })

        # 对受限运动类型检查上限与稳定巡航下限
        if eff_max is not None and seg_arc_m > 0.05:
            # 1) 上限沿整条插值曲线采样检查（捕捉平均速度正常但曲线中段/Hermite峰值超限）
            overspeed_tol = eff_max * 1.03 + 0.05
            if seg_peak_v > overspeed_tol:
                is_internal_only = seg_avg_v <= overspeed_tol
                kmh_note = f" ({mps_to_kmh(seg_peak_v):.1f} km/h > {mps_to_kmh(eff_max):.1f} km/h)" if is_veh else ""
                avg_kmh_note = f" ({mps_to_kmh(seg_avg_v):.1f} km/h)" if is_veh else ""
                if is_internal_only:
                    msg = (
                        f"{oname} [{cat_info['label']}] 第#{i+1}段 ({t0:.2f}s–{t1:.2f}s)："
                        f"段均速 {seg_avg_v:.2f}m/s{avg_kmh_note} 在范围内，但曲线内部在 {seg_peak_t:.2f}s 处峰值达到 "
                        f"{seg_peak_v:.2f}m/s{kmh_note}，超出上限 {eff_max:.2f}m/s"
                    )
                else:
                    msg = (
                        f"{oname} [{cat_info['label']}] 第#{i+1}段 ({t0:.2f}s–{t1:.2f}s)："
                        f"曲线峰值 {seg_peak_v:.2f}m/s{kmh_note}（均速 {seg_avg_v:.2f}m/s），"
                        f"超出参考范围 [{rec_min:.1f}–{eff_max:.1f} m/s]"
                    )
                issue = {
                    "type": "internal_curve_overspeed" if is_internal_only else "segment_overspeed",
                    "segment_index": i,
                    "t0": round(t0, 3),
                    "t1": round(t1, 3),
                    "peak_time_s": round(seg_peak_t, 3),
                    "arc_distance_m": round(seg_arc_m, 3),
                    "avg_speed_mps": round(seg_avg_v, 3),
                    "peak_speed_mps": round(seg_peak_v, 3),
                    "allowed_max_mps": round(eff_max, 3),
                    "category": cat_info["category"],
                    "category_label": cat_info["label"],
                    "internal_curve_only": is_internal_only,
                    "message": msg,
                }
                if source == "manual":
                    warnings.append(issue)
                else:
                    violations.append(issue)

            # 2) 下限仅用于较长距离的稳定运动段（短距移步或起停/等待允许 0 到下限之间平滑低峰）
            is_short_hop = seg_arc_m < max(DEFAULT_MOTION_DYNAMICS_CONFIG["short_hop_distance_m"], float(cat_info.get("default_mps") or 1.2) * 1.6)
            has_stop_endpoint = bool(k0.get("stop") or k1.get("stop"))
            if (
                not is_short_hop
                and not has_stop_endpoint
                and not bool(obj.get("is_idle_or_waiting"))
                and dt >= DEFAULT_MOTION_DYNAMICS_CONFIG["cruise_check_min_dur_s"]
                and seg_arc_m >= 2.0
                and cruise_min_v > 0.08
                and cruise_min_v < rec_min * 0.72
                and seg_avg_v < rec_min * 0.85
            ):
                warnings.append({
                    "type": "cruise_below_typical_min",
                    "segment_index": i,
                    "t0": round(t0, 3),
                    "t1": round(t1, 3),
                    "avg_speed_mps": round(seg_avg_v, 3),
                    "cruise_min_speed_mps": round(cruise_min_v, 3),
                    "recommended_min_mps": rec_min,
                    "message": f"{oname} [{cat_info['label']}] 稳定段 ({t0:.2f}s–{t1:.2f}s) 均速 {seg_avg_v:.2f}m/s 偏慢（参考下限 {rec_min:.1f}m/s）",
                })

    total_dur_s = max(0.0, float(kfs[-1].get("t", 0.0)) - float(kfs[0].get("t", 0.0)))
    overall_avg = (total_arc_m / total_dur_s) if total_dur_s > 1e-5 else 0.0

    status = "ok"
    if violations:
        status = "violation"
    elif warnings:
        status = "manual_warning" if source == "manual" else "warning"

    wheel_summary: Optional[Dict[str, Any]] = None
    if wheel_radius and wheel_radius > 1e-4:
        total_revs = abs(total_signed_arc_m) / (2.0 * math.pi * wheel_radius)
        wheel_summary = {
            "wheel_radius_m": round(wheel_radius, 3),
            "signed_arc_m": round(total_signed_arc_m, 3),
            "total_roll_deg": round(-math.degrees(total_signed_arc_m / wheel_radius), 1),
            "total_revolutions": round(total_revs, 2),
        }

    return {
        "obj_id": oid,
        "obj_name": oname,
        "motion_source": source,
        "is_bound_attachment": False,
        "is_mounted_rider": is_mounted,
        "mounted_local_only": mounted_local_only,
        "mounted_parent_world_peak_mps": round(vehicle_parent_world_speed_mps, 3),
        "meters_per_unit": mpu,
        "unit_mismatch": unit_mismatch,
        "category": cat_info,
        "status": status,
        "can_apply_fix": bool(violations or any(w.get("type") in ("segment_overspeed", "internal_curve_overspeed") for w in warnings)),
        "undoable": True,
        "violations": violations,
        "warnings": warnings,
        "total_arc_m": round(total_arc_m, 3),
        "total_duration_s": round(total_dur_s, 3),
        "overall_avg_mps": round(overall_avg, 3),
        "overall_avg_kmh": round(mps_to_kmh(overall_avg), 1),
        "peak_speed_mps": round(global_peak_v, 3),
        "peak_speed_kmh": round(mps_to_kmh(global_peak_v), 1),
        "peak_time_s": round(global_peak_t, 3),
        "direction": "reverse" if any_reverse else "forward",
        "wheel_summary": wheel_summary,
        "segments": segments_out,
    }


def compute_wheel_roll_deg_at_time(
    obj: Dict[str, Any],
    t: float,
    samples_per_sec: int = 24,
    scene: Optional[Dict[str, Any]] = None,
) -> float:
    """根据物体实际 3D 行进弧长与轮半径计算 t 时刻累计车轮滚动角（度），倒车回转，供前端与后端渲染同步使用。"""
    if not is_vehicle_object(obj):
        return 0.0
    radius = detect_wheel_radius_from_object(obj)
    if not radius or radius <= 1e-4:
        return 0.0
    kfs = _dedup_sorted_kfs(obj.get("keyframes") or [])
    if len(kfs) < 2:
        return 0.0
    t0 = float(kfs[0].get("t", 0.0))
    t_last = float(kfs[-1].get("t", 0.0))
    tc = max(t0, min(t_last, float(t)))
    if tc <= t0 + 1e-5:
        return 0.0

    has_wp_wr = False
    obj_map_wr = None
    _eval_wp_wr = None
    mp_plan_wr = None
    mp_wr_cfg = obj.get("motion_protection")
    if (
        isinstance(mp_wr_cfg, dict)
        and mp_wr_cfg.get("enabled")
        and str(mp_wr_cfg.get("mode") or "smooth_variable").lower().strip() != "manual"
        and len(kfs) >= 2
    ):
        try:
            from director_engine import _get_cached_track_continuity_plan
            mp_plan_wr = _get_cached_track_continuity_plan(obj, scene)
        except Exception:
            try:
                from motion_continuity import build_track_continuity_plan
                mp_plan_wr = build_track_continuity_plan(obj, scene)
            except Exception:
                mp_plan_wr = None
    try:
        from motion_continuity import _has_object_world_prefix, _eval_waypoint_world_transform_mc
        has_wp_wr = _has_object_world_prefix(obj, scene)
        if has_wp_wr:
            _eval_wp_wr = _eval_waypoint_world_transform_mc
            if isinstance(scene, dict):
                obj_map_wr = {
                    str(o.get("id")): o
                    for o in (scene.get("objects") or [])
                    if isinstance(o, dict) and o.get("id")
                }
    except Exception:
        has_wp_wr = False

    def _eval_wr_world(tq: float) -> Tuple[List[float], float]:
        ev = eval_track_state_raw(obj, tq, is_camera=False, scene=scene, continuity_plan=mp_plan_wr)
        ry_w = float((ev.get("rot") or [0.0, 0.0, 0.0])[1])
        if has_wp_wr and ev.get("world_pos") is not None:
            pw = ev["world_pos"]
            if ev.get("world_rot_y") is not None:
                ry_w = float(ev["world_rot_y"])
            return pw, ry_w
        if has_wp_wr and _eval_wp_wr is not None:
            pw, ry_w = _eval_wp_wr(obj, ev["pos"], ry_w, tq, scene=scene, obj_map=obj_map_wr)
            return pw, ry_w
        return list(ev["pos"]), ry_w

    # 按关键帧分段采样，确保 reverse 段与连续弧长与 analyze_track_speed_profile 严格一致
    signed_arc = 0.0
    for i in range(len(kfs) - 1):
        ka, kb = kfs[i], kfs[i + 1]
        ta, tb = float(ka.get("t", 0.0)), float(kb.get("t", 0.0))
        if tc <= ta:
            break
        end_t = min(tc, tb)
        span = end_t - ta
        if span <= 1e-6:
            continue
        interp_m = _normalize_interp_mode(ka.get("interp_pos") or ka.get("interp") or "continuous")
        if ka.get("cut") or ka.get("teleport") or interp_m == "hold":
            continue
        n = max(4, min(64, int(math.ceil(span * samples_per_sec))))
        prev_p, _ = _eval_wr_world(ta)
        for s in range(1, n + 1):
            ts = ta + (span * s) / n
            cur_p, cur_ry = _eval_wr_world(ts)
            dx = cur_p[0] - prev_p[0]
            dy = cur_p[1] - prev_p[1]
            dz = cur_p[2] - prev_p[2]
            step_d = math.sqrt(dx * dx + dy * dy + dz * dz)
            sign = -1.0 if bool(ka.get("reverse")) else 1.0
            if not ka.get("reverse") and not obj.get("align_tangent_y") and math.hypot(dx, dz) > 1e-4:
                yaw_r = math.radians(cur_ry)
                dot_f = dx * (-math.sin(yaw_r)) + dz * (-math.cos(yaw_r))
                if dot_f < -0.55 * math.hypot(dx, dz):
                    sign = -1.0
            signed_arc += sign * step_d
            prev_p = cur_p
    return -math.degrees(signed_arc / radius)


def check_locked_composition_conflicts(
    distance_m: float,
    duration_s: float,
    obj_id: str,
    obj_name: str,
    motion_type: str = "char_walk",
    t_start: float = 0.0,
    t_end: Optional[float] = None,
    has_start_stop_ramp: bool = True,
    surface: str = "default",
    custom_max_mps: Optional[float] = None,
    has_start_ramp: Optional[bool] = None,
    has_stop_ramp: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """
    检查锁定的关键构图区间 [t_start, t_end] 内“D 米距离 / T 秒时长”是否可在推荐速度范围内完成。
    若不能满足，明确展示冲突对象、路程、可用时长、所需均速/峰值及 3 个可选方案，绝不暗改锁定构图。
    """
    spec = SPEED_LIMIT_TABLE.get(motion_type, SPEED_LIMIT_TABLE["char_walk"])
    max_mps = float(custom_max_mps) if custom_max_mps is not None else float(spec["max_mps"])
    d = max(0.0, float(distance_m))
    t_avail = max(1e-4, float(duration_s))
    if d <= 1e-4:
        return None

    req_avg = d / t_avail
    # 考虑平滑起停所需的加速/减速时间（不同路况和动作可配；途经不停顿关键帧只计实际启停端）
    accel_s = float(spec["default_accel_s"])
    decel_s = float(spec["default_decel_s"])
    if surface == "snow" and spec["kind"] == "vehicle":
        accel_s *= 1.15
        decel_s *= 1.15
    use_start = has_start_stop_ramp if has_start_ramp is None else bool(has_start_ramp)
    use_stop = has_start_stop_ramp if has_stop_ramp is None else bool(has_stop_ramp)
    raw_ramp = (accel_s if use_start else 0.0) + (decel_s if use_stop else 0.0)
    ramp_total = min(t_avail * 0.72, raw_ramp) if raw_ramp > 0.0 else 0.0
    # 平滑梯形速度剖面下：D = v_peak * (T - 0.5 * ramp_total)
    eff_t = max(0.25 * t_avail, t_avail - 0.5 * ramp_total)
    req_peak = d / eff_t

    if req_peak <= max_mps * 1.02:
        return None

    comp_end = round(t_start + t_avail, 3) if t_end is None else round(float(t_end), 3)
    suggested_dur = round((d / max(0.2, max_mps * 0.88)) + (0.5 * ramp_total if ramp_total > 0.0 else 0.0), 2)
    suggested_dist = round(max_mps * 0.90 * eff_t, 2)
    suggested_limit = round(math.ceil(req_peak * 10.0) / 10.0, 1)

    return {
        "conflict": True,
        "obj_id": str(obj_id),
        "obj_name": str(obj_name),
        "motion_type": motion_type,
        "motion_label": spec["label"],
        "kind": spec["kind"],
        "distance_m": round(d, 3),
        "duration_s": round(t_avail, 3),
        "required_avg_mps": round(req_avg, 3),
        "required_peak_mps": round(req_peak, 3),
        "required_avg_kmh": round(mps_to_kmh(req_avg), 1),
        "required_peak_kmh": round(mps_to_kmh(req_peak), 1),
        "allowed_max_mps": round(max_mps, 3),
        "allowed_max_kmh": round(mps_to_kmh(max_mps), 1),
        "conflicting_comp_times": [round(float(t_start), 3), comp_end],
        "message": (
            f"构图时空冲突：{obj_name} ({spec['label']}) 需在 [{t_start:.2f}s–{comp_end:.2f}s]（可用 {t_avail:.2f}s）"
            f"行进 {d:.2f}m，所需均速 {req_avg:.2f}m/s / 含起停峰值 {req_peak:.2f}m/s，"
            f"超过建议上限 {max_mps:.1f}m/s"
        ),
        "options": [
            {
                "id": "extend_duration",
                "label": "延长该段时间",
                "suggested_duration_s": suggested_dur,
                "detail": f"将区间时长从 {t_avail:.2f}s 延长至约 {suggested_dur:.2f}s",
            },
            {
                "id": "raise_speed_limit",
                "label": "确认提高速度上限",
                "suggested_max_mps": suggested_limit,
                "detail": f"将 {obj_name} 速度上限例外提升至 {suggested_limit:.1f}m/s 并记录原因",
            },
            {
                "id": "adjust_path_or_target",
                "label": "允许调整路径或目标位置",
                "suggested_max_distance_m": suggested_dist,
                "detail": f"在原时长内将路程缩短至 ≤ {suggested_dist:.2f}m",
            },
        ],
    }


def apply_squad_uniform_pace(
    members: List[Dict[str, Any]],
    speed_mps: Optional[float] = None,
    stagger_step_s: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    小队雪林穿行规范化：
    - 统一小队行走速度在 0.7–1.0 m/s（默认 0.85 m/s），禁止给成员随机分配不同速度导致散队；
    - 保留合理间距，仅用轻微错峰起步延迟（如 0.22s）表达跟随。
    """
    target_v = float(speed_mps if speed_mps is not None else SQUAD_SNOW_FOREST_POLICY["default_mps"])
    target_v = max(SQUAD_SNOW_FOREST_POLICY["min_mps"], min(SQUAD_SNOW_FOREST_POLICY["max_mps"], target_v))
    step_s = float(stagger_step_s if stagger_step_s is not None else SQUAD_SNOW_FOREST_POLICY["stagger_step_s"])
    out: List[Dict[str, Any]] = []
    for idx, mem in enumerate(members):
        m = copy.deepcopy(mem)
        m["motion_type"] = "char_slow_walk"
        m["squad_uniform_speed_mps"] = round(target_v, 3)
        kfs = _dedup_sorted_kfs(m.get("keyframes") or [])
        if len(kfs) >= 2:
            delay = round(idx * step_s, 3)
            # 按统一目标速率与错峰偏移重算各段时刻，保持各成员行进速度严格一致
            anchor_t = float(kfs[0].get("t", 0.0)) + delay
            new_kfs = []
            cur_t = anchor_t
            k0 = copy.deepcopy(kfs[0])
            k0["t"] = round(cur_t, 3)
            new_kfs.append(k0)
            for j in range(1, len(kfs)):
                prev_p = kfs[j - 1].get("pos") or [0.0, 0.0, 0.0]
                cur_p = kfs[j].get("pos") or [0.0, 0.0, 0.0]
                seg_d = math.dist(prev_p[:3], cur_p[:3])
                seg_dt = (seg_d / target_v) if seg_d > 1e-4 else max(0.05, float(kfs[j]["t"]) - float(kfs[j - 1]["t"]))
                cur_t += seg_dt
                kj = copy.deepcopy(kfs[j])
                kj["t"] = round(cur_t, 3)
                new_kfs.append(kj)
            m["keyframes"] = new_kfs
        out.append(m)
    return out


def apply_speed_limit_fix_to_track(
    obj: Dict[str, Any],
    scene: Optional[Dict[str, Any]] = None,
    mode: str = "auto_curve_or_extend",
    exception_reason: Optional[str] = None,
    custom_max_mps: Optional[float] = None,
) -> Dict[str, Any]:
    """
    修正单条轨道以符合速度规范（支持平滑曲线调整、延长时间、或保存例外原因；不突破每 10s<=5 帧约束）。
    mode:
      - "curve_only": 优先尝试把导致内部过峰的超调/陡峭 ease 改为平顺 linear/continuous；不修改总时长或目的地。
      - "extend_duration" / "auto_curve_or_extend": 若平滑曲线仍超限且未锁定外部构图，则按所需时长等比例放宽该段耗时。
      - "raise_speed_limit": 记录用户批准的速度例外原因（speed_exception），不改写关键帧位置与时刻。
      - "adjust_path_or_target": 仅当显式选择时按上限速度回缩位移终点。
    """
    res_obj = copy.deepcopy(obj)
    if is_prop_bound_active(res_obj):
        return {"obj": res_obj, "changed": False, "action": "exempt_bound", "report": analyze_track_speed_profile(res_obj, scene)}

    report_before = analyze_track_speed_profile(res_obj, scene)
    cat_info = report_before["category"]
    eff_max = float(custom_max_mps) if custom_max_mps is not None else float(cat_info.get("effective_max_mps") or 0.0)

    if mode == "raise_speed_limit":
        peak_v = max(float(report_before.get("peak_speed_mps") or 0.0), eff_max)
        new_max = round(max(eff_max, math.ceil(peak_v * 10.0) / 10.0), 2)
        res_obj["speed_exception"] = {
            "max_mps": new_max,
            "reason": str(exception_reason or "剧情/特殊动作确认超限").strip(),
            "motion_type": cat_info["category"],
            "approved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return {"obj": res_obj, "changed": True, "action": "raise_speed_limit", "report": analyze_track_speed_profile(res_obj, scene)}

    kfs = _dedup_sorted_kfs(res_obj.get("keyframes") or [])
    if len(kfs) < 2 or eff_max <= 0.0:
        return {"obj": res_obj, "changed": False, "action": "noop", "report": report_before}

    changed = False
    mpu, _ = resolve_scene_meters_per_unit(scene)

    # Step 1: 先尝试仅平滑内部曲线（不改时间与终点），解决“段均速正常但曲线内部脉冲超速”
    for seg in report_before.get("segments") or []:
        if seg.get("is_cut_or_teleport"):
            continue
        if float(seg.get("peak_speed_mps") or 0.0) > eff_max * 1.03 + 0.05 and float(seg.get("avg_speed_mps") or 0.0) <= eff_max * 1.01:
            idx = int(seg["index"])
            if 0 <= idx < len(kfs) - 1:
                # 若 ease_in_out/ease_in 造成中段斜率达到 1.5x–2.0x，改为平稳 linear（仍保持起停外端点平滑）
                kfs[idx]["interp"] = "linear"
                kfs[idx]["interp_pos"] = "linear"
                changed = True

    if changed:
        res_obj["keyframes"] = kfs
        report_mid = analyze_track_speed_profile(res_obj, scene)
        if report_mid["status"] in ("ok", "exempt_bound", "warning") and not any(
            v.get("type") in ("segment_overspeed", "internal_curve_overspeed") for v in report_mid.get("violations", []) + report_mid.get("warnings", [])
        ):
            return {"obj": res_obj, "changed": True, "action": "curve_smoothed", "report": report_mid}
    else:
        report_mid = report_before

    if mode == "curve_only":
        return {"obj": res_obj, "changed": changed, "action": "curve_only", "report": report_mid}

    if mode == "adjust_path_or_target":
        # 按段回缩超出上限的步长（保留起点，仅收缩超出速度上限的路程增量）
        target_cruise = min(eff_max * 0.92, float(cat_info.get("default_mps") or eff_max * 0.85))
        new_kfs = [copy.deepcopy(kfs[0])]
        cum_shift = [0.0, 0.0, 0.0]
        for i in range(1, len(kfs)):
            k_cur = copy.deepcopy(kfs[i])
            prev_orig = kfs[i - 1].get("pos") or [0.0, 0.0, 0.0]
            cur_orig = kfs[i].get("pos") or [0.0, 0.0, 0.0]
            dx_m = (float(cur_orig[0]) - float(prev_orig[0])) * mpu
            dy_m = (float(cur_orig[1]) - float(prev_orig[1])) * mpu
            dz_m = (float(cur_orig[2]) - float(prev_orig[2])) * mpu
            dt = max(1e-4, float(kfs[i]["t"]) - float(kfs[i - 1]["t"]))
            seg_d_m = math.sqrt(dx_m * dx_m + dy_m * dy_m + dz_m * dz_m)
            max_d_m = target_cruise * dt
            if seg_d_m > max_d_m + 1e-4:
                scale = max_d_m / seg_d_m
                cum_shift[0] += (dx_m * (scale - 1.0)) / mpu
                cum_shift[1] += (dy_m * (scale - 1.0)) / mpu
                cum_shift[2] += (dz_m * (scale - 1.0)) / mpu
                changed = True
            k_cur["pos"] = [
                round(float(cur_orig[0]) + cum_shift[0], 4),
                round(float(cur_orig[1]) + cum_shift[1], 4),
                round(float(cur_orig[2]) + cum_shift[2], 4),
            ]
            new_kfs.append(k_cur)
        res_obj["keyframes"] = new_kfs
        return {"obj": res_obj, "changed": changed, "action": "adjusted_path", "report": analyze_track_speed_profile(res_obj, scene)}

    # Step 2: 延长超速段耗时（默认留 8% 余量确保整条曲线峰值也落入约束内）
    new_kfs = [copy.deepcopy(kfs[0])]
    time_shift = 0.0
    for seg in report_mid.get("segments") or []:
        idx = int(seg["index"])
        k_next = copy.deepcopy(kfs[idx + 1])
        dt_orig = float(kfs[idx + 1]["t"]) - float(kfs[idx]["t"])
        peak_v = float(seg.get("peak_speed_mps") or 0.0)
        if not seg.get("is_cut_or_teleport") and peak_v > eff_max * 1.03 + 0.05:
            ratio = max(1.0, peak_v / max(0.2, eff_max * 0.92))
            dt_new = round(dt_orig * ratio, 3)
            time_shift += (dt_new - dt_orig)
            changed = True
        k_next["t"] = round(float(kfs[idx + 1]["t"]) + time_shift, 3)
        new_kfs.append(k_next)
    res_obj["keyframes"] = new_kfs
    return {"obj": res_obj, "changed": changed, "action": "extended_duration", "report": analyze_track_speed_profile(res_obj, scene)}


def _vec_angle_deg(v0: List[float], v1: List[float]) -> float:
    n0 = math.sqrt(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2])
    n1 = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])
    if n0 < 1e-6 or n1 < 1e-6:
        return 0.0
    d = (v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2]) / (n0 * n1)
    d = max(-1.0, min(1.0, d))
    return math.degrees(math.acos(d))


def analyze_character_limb_speed_profile(
    obj: Dict[str, Any],
    scene: Optional[Dict[str, Any]] = None,
    samples_per_sec: int = 24,
) -> Dict[str, Any]:
    """
    评估关节人偶（mannequin）局部肢体线速度、肩/肘/髋/膝关节角速度、视线转头角速度及胸廓扭转角速度：
    - 严格跳过球头人偶（pawn），绝不把人物整体行走/乘车世界位移计入肢体速度
    - 沿实际插值 + Soft IK 求解轨迹采样峰值（而非只看关键帧平均）
    """
    oid = str(obj.get("id") or "char")
    oname = str(obj.get("name") or oid)
    scene_style = str((scene or {}).get("global_body_style") or "").strip() if isinstance(scene, dict) else ""
    body_style = str(obj.get("body_style") or scene_style or "mannequin").strip()
    if body_style == "pawn":
        return {"obj_id": oid, "obj_name": oname, "body_style": "pawn", "status": "exempt_pawn", "warnings": [], "violations": []}

    kfs = _dedup_sorted_kfs(obj.get("keyframes") or [])
    if len(kfs) < 2:
        return {"obj_id": oid, "obj_name": oname, "body_style": body_style, "status": "ok", "warnings": [], "violations": []}

    try:
        from director_engine import interpolate_character_at_time, solve_mannequin_ik
    except Exception:
        return {"obj_id": oid, "obj_name": oname, "body_style": body_style, "status": "skipped", "warnings": [], "violations": []}

    mpu, _ = resolve_scene_meters_per_unit(scene)
    profile_key = str(obj.get("limb_speed_profile") or "").strip()
    if profile_key not in LIMB_SPEED_LIMIT_TABLE:
        text_hint = " ".join([str(obj.get("name") or ""), str(obj.get("pose_name") or ""), str((scene or {}).get("name") or "")]).lower()
        if any(w in text_hint for w in ("打", "格斗", "拳", "踢", "fight", "combat", "martial", "dodge", "kick", "punch")):
            profile_key = "action_combat"
        elif any(w in text_hint for w in ("枪", "瞄", "跑", "走", "rifle", "walk", "run", "tactical")):
            profile_key = "tactical_walk"
        else:
            profile_key = "calm"
    lim = LIMB_SPEED_LIMIT_TABLE[profile_key]
    exc = obj.get("limb_speed_exception") if isinstance(obj.get("limb_speed_exception"), dict) else {}
    max_hand = float(exc.get("hand_max_mps") or lim["hand_mps"][1])
    max_foot = float(exc.get("foot_max_mps") or lim["foot_swing_mps"][1])
    max_pelvis_y = float(exc.get("pelvis_y_max_mps") or lim["pelvis_y_mps"][1])
    max_joint_deg_s = float(exc.get("joint_max_deg_s") or lim["joint_deg_s"])
    max_head_deg_s = float(exc.get("head_max_deg_s") or lim["head_deg_s"])
    max_twist_deg_s = float(exc.get("torso_twist_max_deg_s") or lim["torso_twist_deg_s"])

    peaks = {
        "hand_L_mps": 0.0,
        "hand_R_mps": 0.0,
        "foot_L_mps": 0.0,
        "foot_R_mps": 0.0,
        "pelvis_y_mps": 0.0,
        "joint_deg_s": 0.0,
        "head_deg_s": 0.0,
        "torso_twist_deg_s": 0.0,
    }
    peak_times = {k: 0.0 for k in peaks}
    warnings: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []

    def _sample_local_ik(ts: float) -> Dict[str, Any]:
        st = interpolate_character_at_time(obj, ts, scene=scene)
        # Solve in local space (pos=[0,0,0], rot=[0,0,0]) so root travel never pollutes limb speeds
        local_st = dict(st, pos=[0.0, 0.0, 0.0], rot=[0.0, 0.0, 0.0])
        ik = solve_mannequin_ik(local_st)
        j = ik["joints"]
        a = ik["axes"]
        return {
            "wrist_L_rel": [float(j["wrist_L"][k] - j["shoulder_L"][k]) * mpu for k in range(3)],
            "wrist_R_rel": [float(j["wrist_R"][k] - j["shoulder_R"][k]) * mpu for k in range(3)],
            "ankle_L_rel": [float(j["ankle_L"][k] - j["hip_L"][k]) * mpu for k in range(3)],
            "ankle_R_rel": [float(j["ankle_R"][k] - j["hip_R"][k]) * mpu for k in range(3)],
            "uarm_L": [float(j["elbow_L"][k] - j["shoulder_L"][k]) for k in range(3)],
            "farm_L": [float(j["wrist_L"][k] - j["elbow_L"][k]) for k in range(3)],
            "uarm_R": [float(j["elbow_R"][k] - j["shoulder_R"][k]) for k in range(3)],
            "farm_R": [float(j["wrist_R"][k] - j["elbow_R"][k]) for k in range(3)],
            "thigh_L": [float(j["knee_L"][k] - j["hip_L"][k]) for k in range(3)],
            "calf_L": [float(j["ankle_L"][k] - j["knee_L"][k]) for k in range(3)],
            "thigh_R": [float(j["knee_R"][k] - j["hip_R"][k]) for k in range(3)],
            "calf_R": [float(j["ankle_R"][k] - j["knee_R"][k]) for k in range(3)],
            "look_dir": [float(a["look_dir"][k]) for k in range(3)],
            "pelvis_y": float(j["pelvis"][1]) * mpu,
            "torso_twist": float(st.get("torso_twist", 0.0) or 0.0),
        }

    for i in range(len(kfs) - 1):
        k0, k1 = kfs[i], kfs[i + 1]
        t0, t1 = float(k0.get("t", 0.0)), float(k1.get("t", 0.0))
        dt = t1 - t0
        mode_p = _normalize_interp_mode(k0.get("interp_pose") or k0.get("interp") or "linear")
        if k0.get("cut") or k1.get("cut") or k0.get("teleport") or k1.get("teleport") or mode_p == "hold" or dt <= 1e-4:
            continue
        n_s = max(8, min(96, int(math.ceil(dt * max(16, samples_per_sec)))))
        prev_s = _sample_local_ik(t0)
        for s_idx in range(1, n_s + 1):
            ts = t0 + (dt * s_idx) / n_s
            step_dt = dt / n_s
            cur_s = _sample_local_ik(ts)
            t_mid = ts - 0.5 * step_dt
            for pk, rk in (("hand_L_mps", "wrist_L_rel"), ("hand_R_mps", "wrist_R_rel"), ("foot_L_mps", "ankle_L_rel"), ("foot_R_mps", "ankle_R_rel")):
                v = math.dist(prev_s[rk], cur_s[rk]) / step_dt
                if v > peaks[pk]:
                    peaks[pk] = v
                    peak_times[pk] = t_mid
            pv_y = abs(cur_s["pelvis_y"] - prev_s["pelvis_y"]) / step_dt
            if pv_y > peaks["pelvis_y_mps"]:
                peaks["pelvis_y_mps"] = pv_y
                peak_times["pelvis_y_mps"] = t_mid
            hd_v = _vec_angle_deg(prev_s["look_dir"], cur_s["look_dir"]) / step_dt
            if hd_v > peaks["head_deg_s"]:
                peaks["head_deg_s"] = hd_v
                peak_times["head_deg_s"] = t_mid
            tw_v = abs(cur_s["torso_twist"] - prev_s["torso_twist"]) / step_dt
            if tw_v > peaks["torso_twist_deg_s"]:
                peaks["torso_twist_deg_s"] = tw_v
                peak_times["torso_twist_deg_s"] = t_mid
            j_ang = max(
                _vec_angle_deg(prev_s["uarm_L"], cur_s["uarm_L"]),
                _vec_angle_deg(prev_s["farm_L"], cur_s["farm_L"]),
                _vec_angle_deg(prev_s["uarm_R"], cur_s["uarm_R"]),
                _vec_angle_deg(prev_s["farm_R"], cur_s["farm_R"]),
                _vec_angle_deg(prev_s["thigh_L"], cur_s["thigh_L"]),
                _vec_angle_deg(prev_s["calf_L"], cur_s["calf_L"]),
                _vec_angle_deg(prev_s["thigh_R"], cur_s["thigh_R"]),
                _vec_angle_deg(prev_s["calf_R"], cur_s["calf_R"]),
            ) / step_dt
            if j_ang > peaks["joint_deg_s"]:
                peaks["joint_deg_s"] = j_ang
                peak_times["joint_deg_s"] = t_mid
            prev_s = cur_s

    checks = [
        ("hand_L_mps", max_hand, "m/s", "左手腕相对肩速"),
        ("hand_R_mps", max_hand, "m/s", "右手腕相对肩速"),
        ("foot_L_mps", max_foot, "m/s", "左脚踝相对髋速"),
        ("foot_R_mps", max_foot, "m/s", "右脚踝相对髋速"),
        ("pelvis_y_mps", max_pelvis_y, "m/s", "骨盆重心升降速"),
        ("joint_deg_s", max_joint_deg_s, "°/s", "肢段最大角速度"),
        ("head_deg_s", max_head_deg_s, "°/s", "头部转视角速度"),
        ("torso_twist_deg_s", max_twist_deg_s, "°/s", "胸廓扭转角速度"),
    ]
    source = _resolve_track_source(obj, scene)
    for pk, limit_v, unit, desc in checks:
        pv = peaks[pk]
        if pv > limit_v * 1.05 + (0.05 if unit == "m/s" else 8.0):
            rec = {
                "type": "limb_overspeed",
                "channel": pk,
                "peak_value": round(pv, 2),
                "allowed_max": round(limit_v, 2),
                "unit": unit,
                "peak_time_s": round(peak_times[pk], 3),
                "message": f"{oname} [{lim['label']}] {desc} 在 {peak_times[pk]:.2f}s 峰值达 {pv:.2f}{unit}，超过建议上限 {limit_v:.1f}{unit}",
            }
            (warnings if source == "manual" else violations).append(rec)

    status = "violation" if violations else ("warning" if warnings else "ok")
    return {
        "obj_id": oid,
        "obj_name": oname,
        "body_style": body_style,
        "profile": profile_key,
        "profile_label": lim["label"],
        "status": status,
        "peaks": {k: round(v, 2) for k, v in peaks.items()},
        "peak_times": {k: round(v, 3) for k, v in peak_times.items()},
        "violations": violations,
        "warnings": warnings,
    }


def validate_scene_speed_policy(scene: Dict[str, Any]) -> Dict[str, Any]:
    """整镜批量验证：不修改原 scene，仅返回单位信息、全轨报告、违例/警告列表与构图冲突。"""
    if not isinstance(scene, dict):
        return {"ok": True, "unit_scale_m": 1.0, "unit_mismatch": False, "reports": [], "violations": [], "warnings": [], "conflicts": []}

    mpu, unit_mismatch = resolve_scene_meters_per_unit(scene)
    reports: List[Dict[str, Any]] = []
    all_violations: List[Dict[str, Any]] = []
    all_warnings: List[Dict[str, Any]] = []
    all_conflicts: List[Dict[str, Any]] = []

    export_plan = scene.get("export_plan") if isinstance(scene.get("export_plan"), dict) else {}
    locked_comps = sorted([float(c["t"]) for c in (export_plan.get("compositions") or []) if isinstance(c, dict) and c.get("t") is not None])

    limb_reports: List[Dict[str, Any]] = []
    for obj in (scene.get("objects") or []):
        if not isinstance(obj, dict):
            continue
        rep = analyze_track_speed_profile(obj, scene=scene, is_camera=False)
        reports.append(rep)
        for v in rep.get("violations") or []:
            all_violations.append({"obj_id": rep["obj_id"], "obj_name": rep["obj_name"], **v})
        for w in rep.get("warnings") or []:
            all_warnings.append({"obj_id": rep["obj_id"], "obj_name": rep["obj_name"], **w})

        if obj.get("type") == "character":
            l_rep = analyze_character_limb_speed_profile(obj, scene=scene)
            limb_reports.append(l_rep)
            for lv in l_rep.get("violations") or []:
                all_violations.append({"obj_id": l_rep["obj_id"], "obj_name": l_rep["obj_name"], **lv})
            for lw in l_rep.get("warnings") or []:
                all_warnings.append({"obj_id": l_rep["obj_id"], "obj_name": l_rep["obj_name"], **lw})

        cat = rep.get("category") or {}
        if rep.get("motion_source") == "auto" and cat.get("effective_max_mps") and rep.get("segments"):
            kfs = obj.get("keyframes") if isinstance(obj.get("keyframes"), list) else []
            for seg in rep["segments"]:
                if seg.get("is_cut_or_teleport"):
                    continue
                idx = int(seg.get("index", 0))
                prev_kf = kfs[idx] if 0 <= idx < len(kfs) and isinstance(kfs[idx], dict) else {}
                next_kf = kfs[idx + 1] if 0 <= idx + 1 < len(kfs) and isinstance(kfs[idx + 1], dict) else {}
                has_start_ramp = (idx == 0) or bool(prev_kf.get("stop"))
                has_stop_ramp = bool(next_kf.get("stop"))
                t0, t1 = float(seg["t0"]), float(seg["t1"])
                if any(abs(ct - t1) <= 0.15 for ct in locked_comps):
                    conf = check_locked_composition_conflicts(
                        distance_m=float(seg.get("arc_distance_m") or 0.0),
                        duration_s=float(seg.get("dt") or 0.1),
                        obj_id=rep["obj_id"],
                        obj_name=rep["obj_name"],
                        motion_type=cat.get("category", "char_walk"),
                        t_start=t0,
                        t_end=t1,
                        has_start_stop_ramp=bool(has_start_ramp or has_stop_ramp),
                        has_start_ramp=has_start_ramp,
                        has_stop_ramp=has_stop_ramp,
                        custom_max_mps=float(cat["effective_max_mps"]),
                    )
                    if conf:
                        all_conflicts.append(conf)

    # 摄影机也进入观测报告（不做限速报错）
    if isinstance(scene.get("camera_keyframes"), list) and len(scene["camera_keyframes"]) >= 2:
        cam_rep = analyze_track_speed_profile(scene, scene=scene, is_camera=True)
        reports.append(cam_rep)

    continuity_diag = {"ok": True, "conflicts": [], "warnings": []}
    try:
        from motion_continuity import check_scene_continuity_diagnostics
        continuity_diag = check_scene_continuity_diagnostics(scene)
        for cc in continuity_diag.get("conflicts") or []:
            all_conflicts.append(cc)
        for cw in continuity_diag.get("warnings") or []:
            all_warnings.append(cw)
    except Exception:
        pass

    return {
        "ok": len(all_violations) == 0 and len(all_conflicts) == 0,
        "meters_per_unit": mpu,
        "unit_mismatch": unit_mismatch,
        "unit_notice": f"已按工程单位 1 单位 = {mpu:.3f}m 换算速度（不自动缩放场景）" if unit_mismatch else "工程单位确认：1 世界单位 = 1m",
        "reports": reports,
        "limb_reports": limb_reports,
        "violations": all_violations,
        "warnings": all_warnings,
        "conflicts": all_conflicts,
        "continuity": continuity_diag,
    }
