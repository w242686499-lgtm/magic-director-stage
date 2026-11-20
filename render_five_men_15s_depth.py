#!/usr/bin/env python3
"""
render_five_men_15s_depth.py
15 秒五人一镜到底深度图视频 & 动作与移焦预览视频生成器 (1280x720, 24fps, 360 帧, 无音轨)

核心特性：
1. 严格遵循同一个按秒求值的 evaluateAtTime(t) 驱动五个人物（鲍里斯 + 后方四人）、简化步枪、摄影机三维推进路径及辅助预览移焦。
2. 逐像素真实相机空间深度 (Camera-Space Z-Buffer)：
   - 对球体、圆柱体、胶囊体、有向长方体及三维地面进行精确解析射线求交 (Analytical Ray-Primitive Z-Buffer)；
   - 全片使用固定近远深度映射范围 [NEAR_DEPTH=0.45m, FAR_DEPTH=14.00m]，近亮远暗（枪口最亮 -> 鲍里斯头部次亮 -> 后方四人依次更暗 -> 远景最暗），绝无逐帧归一化呼吸或前景过曝饱和；
   - 主深度图不包含任何文字、朝向短线、操控 UI、灯光明暗、阴影或移焦模糊。
3. 辅助动作与移焦预览：
   - 同一三维场景与机位，鲍里斯头前绘制醒目的朝向短线（展示 4-6s 看向画面最右侧、6-10s 从画面最右侧缓慢扫向画面中心）；
   - 10-14s 摄影机沿三维空间真实前推（固定焦距/FOV），焦点从鲍里斯脸部平滑移至步枪枪口（脸部由清晰变柔，枪口由柔变清晰，后景始终淡化）。
"""

import json
import math
import os
import subprocess
import time
from typing import Any, Dict, List, Tuple

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


WIDTH = 1280
HEIGHT = 720
FPS = 24
TOTAL_FRAMES = 360  # i = 0..359, t = i / 24, max t = 14.9583s

# 全片固定相机空间深度映射范围 (单位：米)
# 推进结尾时：枪口 z_cam ≈ 0.52m (最亮 ~248)，鲍里斯头 z_cam ≈ 1.50m (次亮 ~218)，后方四人 3.5m..7.3m (168..88)，背景最暗
NEAR_DEPTH = 0.32
FAR_DEPTH = 14.00


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if abs(edge1 - edge0) < 1e-9:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(edge0: float, edge1: float, x: float) -> float:
    if abs(edge1 - edge0) < 1e-9:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def rot_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def rot_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def rot_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _int_smoothstep(a: float, b: float, t: float) -> float:
    """smoothstep(a, b, tau) 从 0 到 t 的解析定积分，O(1) 精确无误差。"""
    if t <= a:
        return 0.0
    span = b - a
    if t < b:
        u = (t - a) / span
        return span * (u * u * u - 0.5 * u * u * u * u)
    return 0.5 * span + (t - b)


def integrate_walk_profile(
    t: float,
    z_start: float,
    t_start: float,
    t_steady: float,
    t_decel_start: float,
    t_stop: float,
    v_max: float,
    step_freq: float,
    phase_offset: float,
) -> Tuple[float, float, float]:
    """
    O(1) 解析积分速度曲线：
    由于加速段 [t_start, t_steady] 与减速段 [t_decel_start, t_stop] 互不重叠 (t_steady < t_decel_start)，
    v(tau) = v_max * (smoothstep(t_start, t_steady, tau) - smoothstep(t_decel_start, t_stop, tau))，
    因此位移 z(t) 与步态相位 phase(t) 均可由解析原函数直接求出，位置与速度 100% 连续光滑。
    """
    if t <= t_start:
        return z_start, 0.0, phase_offset
    tc = min(t, t_stop)
    up_v = smoothstep(t_start, t_steady, tc)
    down_v = smoothstep(t_decel_start, t_stop, tc)
    v_curr = v_max * max(0.0, up_v - down_v) if t < t_stop else 0.0

    # 归一化速度积分 int_0^tc (v(tau)/v_max) d_tau
    norm_dist = _int_smoothstep(t_start, t_steady, tc) - _int_smoothstep(
        t_decel_start, t_stop, tc
    )
    z = z_start + v_max * norm_dist
    phase = phase_offset + norm_dist * step_freq * 2.0 * math.pi
    return z, v_curr, phase


def evaluate_at_time(t: float) -> Dict[str, Any]:
    """
    核心时间求值函数 evaluateAtTime(t)，驱动全片 15 秒一镜到底：
      - 0–4s：走近（鲍里斯低头向摄影机走近，后方四人错开跟进，双腿交替摆动、手臂配合、身体起伏）
      - 4–6s：停步抬头（鲍里斯平滑减速停下并抬头，朝画面最右侧看；后方四人先后停下，步态幅度随速度自然归零）
      - 6–10s：巡视（鲍里斯头部与视线从画面最右侧缓慢扫向画面中心，步枪小幅同向转动；后方四人轻微转头等待）
      - 10–14s：推进与移焦（摄影机沿三维路径平滑前推，固定焦距产生真实透视视差，枪口相对头部更快变大并占据右侧近景；辅助预览焦点从鲍里斯脸部移至枪口）
      - 14–15s：结尾保持（摄影机与主体停稳，枪口在右侧前景、头部在枪后、四人在更深处）
    """
    t = clamp(float(t), 0.0, 15.0)

    # =========================================================================
    # 1. 摄影机轨迹 (Camera Trajectory)
    # 相机位于 +Z 朝 -Z 方向看，Up = (0, 1, 0)：
    # 因此世界坐标 +X 对应【摄影机画面右侧 (Screen Right)】，-X 对应【画面左侧 (Screen Left)】！
    # =========================================================================
    push_u = smootherstep(10.0, 14.0, t)
    cam_pos = np.array(
        [
            lerp(0.00, 0.08, push_u),
            lerp(1.48, 1.44, push_u),
            lerp(5.25, 3.60, push_u),
        ],
        dtype=np.float32,
    )
    cam_target = np.array(
        [
            lerp(0.00, 0.02, push_u),
            lerp(1.36, 1.42, push_u),
            lerp(1.50, 2.05, push_u),
        ],
        dtype=np.float32,
    )
    cam_fov_deg = 30.0  # 固定焦距，绝不通过缩放画布或变焦假冒推进

    # =========================================================================
    # 2. 鲍里斯 (Boris - 最前方领队，持步枪)
    # =========================================================================
    boris_vmax = 0.48
    boris_z, boris_v, boris_phase = integrate_walk_profile(
        t,
        z_start=0.20,
        t_start=0.0,
        t_steady=0.65,
        t_decel_start=3.65,
        t_stop=5.25,
        v_max=boris_vmax,
        step_freq=1.65,
        phase_offset=0.0,
    )
    boris_amp = clamp(boris_v / boris_vmax, 0.0, 1.0)
    boris_bob = boris_amp * 0.028 * abs(math.sin(boris_phase))
    boris_sway_x = boris_amp * 0.018 * math.cos(boris_phase)
    boris_pos = np.array([-0.08 + boris_sway_x, boris_bob, boris_z], dtype=np.float32)

    # 鲍里斯头部姿态：
    # 0-4s：低头 (pitch = -0.42 rad ≈ -24°)
    # 4-6s：停步同时抬头 (pitch -> +0.02 rad)，并转头朝【画面最右侧 (+X)】看
    # 6-10s：视线从【画面最右侧 (+X)】缓慢扫向【画面中心 (X≈0)】
    # 10-15s：保持注视画面中心偏前
    raise_head_u = smootherstep(4.0, 5.8, t)
    look_right_u = smootherstep(4.1, 5.9, t)
    scan_center_u = smootherstep(6.0, 10.0, t)

    head_pitch = lerp(-0.42, 0.02, raise_head_u)
    yaw_at_right = 0.58  # 朝向画面最右侧 (+X)
    yaw_at_center = 0.02  # 回到画面中心
    head_yaw = look_right_u * lerp(yaw_at_right, yaw_at_center, scan_center_u)

    # 步枪姿态（绑定在鲍里斯上身与双手）：
    # 0-4s：低姿持枪随步伐微摆
    # 4-6s：随停步抬头举枪指向右侧前方
    # 6-10s：随视线从右向中做克制的小幅同向转动（+0.32 rad -> +0.20 rad），
    #         保证结尾推进后枪口完整落在画面右侧近景内，而鲍里斯头部位于枪身左后方
    rifle_ready_u = smootherstep(4.0, 5.8, t)
    rifle_yaw_walk = 0.16 + boris_amp * 0.04 * math.sin(boris_phase)
    rifle_yaw_aim = lerp(0.32, 0.20, scan_center_u)
    rifle_yaw = lerp(rifle_yaw_walk, rifle_yaw_aim, rifle_ready_u)
    rifle_pitch = lerp(-0.18, 0.025, rifle_ready_u)

    boris_data = {
        "id": "boris",
        "name": "鲍里斯 (Boris)",
        "is_boris": True,
        "pos": boris_pos,
        "walk_amp": boris_amp,
        "walk_phase": boris_phase,
        "body_yaw": lerp(0.0, 0.05, rifle_ready_u),
        "head_yaw": head_yaw,
        "head_pitch": head_pitch,
        "rifle_yaw": rifle_yaw,
        "rifle_pitch": rifle_pitch,
        "rifle_ready_u": rifle_ready_u,
    }

    # =========================================================================
    # 3. 后方四人 (Four Followers - 错开横向车道与前后纵深，全片始终五人同框清晰可见)
    # =========================================================================
    follower_specs = [
        {
            "id": "follower_1",
            "name": "队员 A (左外近)",
            "x": -1.24,
            "z0": -1.65,
            "t_start": 0.20,
            "t_steady": 0.85,
            "t_decel": 3.85,
            "t_stop": 5.45,
            "vmax": 0.45,
            "freq": 1.58,
            "phase0": 0.9,
            "scan_yaw": 0.14,
        },
        {
            "id": "follower_2",
            "name": "队员 B (右内近)",
            "x": 0.64,
            "z0": -2.75,
            "t_start": 0.38,
            "t_steady": 1.05,
            "t_decel": 4.05,
            "t_stop": 5.65,
            "vmax": 0.44,
            "freq": 1.54,
            "phase0": 2.1,
            "scan_yaw": -0.12,
        },
        {
            "id": "follower_3",
            "name": "队员 C (左内远)",
            "x": -0.92,
            "z0": -4.05,
            "t_start": 0.52,
            "t_steady": 1.20,
            "t_decel": 4.20,
            "t_stop": 5.80,
            "vmax": 0.43,
            "freq": 1.50,
            "phase0": 3.5,
            "scan_yaw": 0.10,
        },
        {
            "id": "follower_4",
            "name": "队员 D (右外远)",
            "x": 1.38,
            "z0": -5.35,
            "t_start": 0.68,
            "t_steady": 1.35,
            "t_decel": 4.30,
            "t_stop": 5.95,
            "vmax": 0.42,
            "freq": 1.48,
            "phase0": 4.7,
            "scan_yaw": -0.15,
        },
    ]

    characters = [boris_data]
    for spec in follower_specs:
        fz, fv, fphase = integrate_walk_profile(
            t,
            z_start=spec["z0"],
            t_start=spec["t_start"],
            t_steady=spec["t_steady"],
            t_decel_start=spec["t_decel"],
            t_stop=spec["t_stop"],
            v_max=spec["vmax"],
            step_freq=spec["freq"],
            phase_offset=spec["phase0"],
        )
        famp = clamp(fv / spec["vmax"], 0.0, 1.0)
        fbob = famp * 0.024 * abs(math.sin(fphase))
        fsway = famp * 0.015 * math.cos(fphase)
        wait_turn = smootherstep(5.5, 7.5, t) * (1.0 - 0.4 * smootherstep(8.5, 10.5, t))
        fh_yaw = wait_turn * spec["scan_yaw"]
        fh_pitch = lerp(-0.18, 0.0, smootherstep(4.2, 5.8, t))

        characters.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "is_boris": False,
                "pos": np.array([spec["x"] + fsway, fbob, fz], dtype=np.float32),
                "walk_amp": famp,
                "walk_phase": fphase,
                "body_yaw": 0.0,
                "head_yaw": fh_yaw,
                "head_pitch": fh_pitch,
            }
        )

    geom = build_scene_primitives(characters)
    boris_head_center = geom["boris_head_center"]
    boris_head_dir = geom["boris_head_dir"]
    rifle_muzzle_pos = geom["rifle_muzzle_pos"]

    cam_forward = normalize(cam_target - cam_pos)
    head_view_z = float(np.dot(boris_head_center - cam_pos, cam_forward))
    muzzle_view_z = float(np.dot(rifle_muzzle_pos - cam_pos, cam_forward))

    focus_shift_u = smootherstep(10.0, 14.0, t)
    focus_depth_z = lerp(head_view_z, muzzle_view_z, focus_shift_u)

    stage_name = "0–4s 走近"
    if 4.0 <= t < 6.0:
        stage_name = "4–6s 停步抬头（看画面最右侧）"
    elif 6.0 <= t < 10.0:
        stage_name = "6–10s 巡视（从画面最右侧扫向中心）"
    elif 10.0 <= t < 14.0:
        stage_name = "10–14s 推进与移焦（脸部→枪口）"
    elif t >= 14.0:
        stage_name = "14–15s 结尾保持"

    return {
        "t": t,
        "stage_name": stage_name,
        "camera": {
            "pos": cam_pos,
            "target": cam_target,
            "fov_deg": cam_fov_deg,
        },
        "characters": characters,
        "primitives": geom["primitives"],
        "boris_head_center": boris_head_center,
        "boris_head_dir": boris_head_dir,
        "rifle_muzzle_pos": rifle_muzzle_pos,
        "rifle_breech_pos": geom["rifle_breech_pos"],
        "head_view_z": head_view_z,
        "muzzle_view_z": muzzle_view_z,
        "focus_depth_z": focus_depth_z,
        "focus_shift_u": focus_shift_u,
    }


def build_scene_primitives(characters: List[Dict[str, Any]]) -> Dict[str, Any]:
    primitives: List[Dict[str, Any]] = []
    boris_head_center = np.array([0.0, 1.5, 2.0], dtype=np.float32)
    boris_head_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    rifle_muzzle_pos = np.array([0.2, 1.4, 2.9], dtype=np.float32)
    rifle_breech_pos = np.array([-0.1, 1.35, 2.1], dtype=np.float32)

    for ch in characters:
        pos = ch["pos"]
        amp = ch["walk_amp"]
        phase = ch["walk_phase"]
        body_yaw = ch["body_yaw"]
        R_body = rot_y(body_yaw)
        is_boris = ch["is_boris"]
        role_tag = "boris" if is_boris else "follower"

        hip_center = pos + np.array([0.0, 0.90, 0.0], dtype=np.float32)
        torso_center = pos + np.array([0.0, 1.18, 0.0], dtype=np.float32)
        shoulder_center = pos + np.array([0.0, 1.38, 0.0], dtype=np.float32)
        head_center = pos + np.array([0.0, 1.56, 0.0], dtype=np.float32)

        # 1. 躯干 + 骨盆
        primitives.append(
            {
                "type": "box",
                "center": torso_center,
                "rot": R_body,
                "half_extents": np.array([0.19, 0.24, 0.12], dtype=np.float32),
                "role": role_tag,
                "part": "torso",
            }
        )
        primitives.append(
            {
                "type": "box",
                "center": hip_center,
                "rot": R_body,
                "half_extents": np.array([0.16, 0.09, 0.11], dtype=np.float32),
                "role": role_tag,
                "part": "pelvis",
            }
        )

        # 2. 球头 + 皮帽
        R_head = rot_y(body_yaw + ch["head_yaw"]) @ rot_x(-ch["head_pitch"])
        primitives.append(
            {
                "type": "sphere",
                "center": head_center,
                "radius": 0.122 if is_boris else 0.116,
                "role": role_tag,
                "part": "head",
            }
        )
        hat_center = head_center + R_head @ np.array([0.0, 0.062, 0.0], dtype=np.float32)
        primitives.append(
            {
                "type": "box",
                "center": hat_center,
                "rot": R_head,
                "half_extents": np.array([0.128, 0.058, 0.130], dtype=np.float32),
                "role": role_tag,
                "part": "hat",
            }
        )

        if is_boris:
            boris_head_center = head_center
            boris_head_dir = normalize(R_head @ np.array([0.0, 0.0, 1.0], dtype=np.float32))

        # 3. 双腿交替摆动
        leg_swing_l = amp * 0.46 * math.sin(phase)
        leg_swing_r = -amp * 0.46 * math.sin(phase)
        knee_bend_l = amp * 0.32 * max(0.0, -math.cos(phase))
        knee_bend_r = amp * 0.32 * max(0.0, math.cos(phase))

        for side_sign, hip_angle, knee_angle in [
            (1.0, leg_swing_l, knee_bend_l),
            (-1.0, leg_swing_r, knee_bend_r),
        ]:
            hip_joint = hip_center + R_body @ np.array([side_sign * 0.10, -0.02, 0.0], dtype=np.float32)
            R_thigh = R_body @ rot_x(-hip_angle)
            knee_joint = hip_joint + R_thigh @ np.array([0.0, -0.42, 0.0], dtype=np.float32)
            R_calf = R_thigh @ rot_x(knee_angle)
            ankle_joint = knee_joint + R_calf @ np.array([0.0, -0.42, 0.0], dtype=np.float32)

            primitives.append(
                {
                    "type": "capsule",
                    "p0": hip_joint,
                    "p1": knee_joint,
                    "radius": 0.068,
                    "role": role_tag,
                    "part": "limb",
                }
            )
            primitives.append(
                {
                    "type": "capsule",
                    "p0": knee_joint,
                    "p1": ankle_joint,
                    "radius": 0.056,
                    "role": role_tag,
                    "part": "limb",
                }
            )

        # 4. 双臂与鲍里斯步枪绑定
        if not is_boris:
            arm_swing_l = -amp * 0.28 * math.sin(phase)
            arm_swing_r = amp * 0.28 * math.sin(phase)
            for side_sign, sh_angle in [(1.0, arm_swing_l), (-1.0, arm_swing_r)]:
                sh_joint = shoulder_center + R_body @ np.array([side_sign * 0.22, -0.02, 0.0], dtype=np.float32)
                R_uarm = R_body @ rot_x(-sh_angle) @ rot_z(side_sign * 0.08)
                elbow = sh_joint + R_uarm @ np.array([0.0, -0.28, 0.04], dtype=np.float32)
                R_farm = R_uarm @ rot_x(-0.35)
                hand = elbow + R_farm @ np.array([0.0, -0.25, 0.08], dtype=np.float32)
                primitives.append(
                    {
                        "type": "capsule",
                        "p0": sh_joint,
                        "p1": elbow,
                        "radius": 0.052,
                        "role": role_tag,
                        "part": "limb",
                    }
                )
                primitives.append(
                    {
                        "type": "capsule",
                        "p0": elbow,
                        "p1": hand,
                        "radius": 0.045,
                        "role": role_tag,
                        "part": "limb",
                    }
                )
        else:
            # 鲍里斯简化步枪：少量长方体和圆柱拼接枪托、枪身、枪管、枪口箍圈与准星
            R_rifle = rot_y(ch["rifle_yaw"]) @ rot_x(-ch["rifle_pitch"])
            ready_u = ch["rifle_ready_u"]
            rifle_pivot = shoulder_center + np.array(
                [
                    lerp(-0.04, 0.02, ready_u),
                    lerp(-0.14, -0.04, ready_u),
                    lerp(0.10, 0.14, ready_u),
                ],
                dtype=np.float32,
            )
            rifle_dir = normalize(R_rifle @ np.array([0.0, 0.0, 1.0], dtype=np.float32))
            rifle_up = normalize(R_rifle @ np.array([0.0, 1.0, 0.0], dtype=np.float32))

            # (a) 枪托 (窄长方体 Stock)
            stock_center = rifle_pivot + rifle_dir * 0.12 - rifle_up * 0.02
            primitives.append(
                {
                    "type": "box",
                    "center": stock_center,
                    "rot": R_rifle,
                    "half_extents": np.array([0.022, 0.042, 0.14], dtype=np.float32),
                    "role": "rifle",
                    "part": "rifle_stock",
                }
            )
            # (b) 木质护木与机匣枪身 (窄长方体 Receiver & Fore-end)
            body_center = rifle_pivot + rifle_dir * 0.38 - rifle_up * 0.008
            primitives.append(
                {
                    "type": "box",
                    "center": body_center,
                    "rot": R_rifle,
                    "half_extents": np.array([0.025, 0.032, 0.20], dtype=np.float32),
                    "role": "rifle",
                    "part": "rifle_body",
                }
            )
            # (c) 主枪管 (圆柱体 Barrel)
            barrel_p0 = rifle_pivot + rifle_dir * 0.45 + rifle_up * 0.012
            barrel_p1 = rifle_pivot + rifle_dir * 0.90 + rifle_up * 0.012
            primitives.append(
                {
                    "type": "capsule",
                    "p0": barrel_p0,
                    "p1": barrel_p1,
                    "radius": 0.026,
                    "role": "rifle",
                    "part": "rifle_barrel",
                }
            )
            # (d) 枪口厚重箍圈 (加粗圆柱体 Muzzle Collar，对应参考图右侧近景的大枪口)
            muzzle_p0 = rifle_pivot + rifle_dir * 0.78 + rifle_up * 0.012
            muzzle_p1 = rifle_pivot + rifle_dir * 0.94 + rifle_up * 0.012
            primitives.append(
                {
                    "type": "capsule",
                    "p0": muzzle_p0,
                    "p1": muzzle_p1,
                    "radius": 0.044,
                    "role": "rifle",
                    "part": "rifle_muzzle",
                }
            )
            # (e) 枪口上方准星 (长方体 Front Sight)
            sight_center = rifle_pivot + rifle_dir * 0.87 + rifle_up * 0.068
            primitives.append(
                {
                    "type": "box",
                    "center": sight_center,
                    "rot": R_rifle,
                    "half_extents": np.array([0.009, 0.026, 0.020], dtype=np.float32),
                    "role": "rifle",
                    "part": "rifle_sight",
                }
            )

            rifle_breech_pos = rifle_pivot
            rifle_muzzle_pos = muzzle_p1

            # 鲍里斯双臂紧握步枪（随手部与上身稳定绑定，不悬空）
            r_sh = shoulder_center + R_body @ np.array([-0.21, -0.03, 0.0], dtype=np.float32)
            r_hand = rifle_pivot + rifle_dir * 0.18 - rifle_up * 0.03
            r_elbow = 0.5 * (r_sh + r_hand) + np.array([-0.08, -0.12, -0.02], dtype=np.float32)

            l_sh = shoulder_center + R_body @ np.array([0.21, -0.03, 0.0], dtype=np.float32)
            l_hand = rifle_pivot + rifle_dir * 0.46 - rifle_up * 0.02
            l_elbow = 0.5 * (l_sh + l_hand) + np.array([0.08, -0.14, 0.02], dtype=np.float32)

            for sh_p, el_p, ha_p in [(r_sh, r_elbow, r_hand), (l_sh, l_elbow, l_hand)]:
                primitives.append(
                    {
                        "type": "capsule",
                        "p0": sh_p,
                        "p1": el_p,
                        "radius": 0.054,
                        "role": "boris",
                        "part": "limb",
                    }
                )
                primitives.append(
                    {
                        "type": "capsule",
                        "p0": el_p,
                        "p1": ha_p,
                        "radius": 0.048,
                        "role": "boris",
                        "part": "limb",
                    }
                )
                primitives.append(
                    {
                        "type": "sphere",
                        "center": ha_p,
                        "radius": 0.052,
                        "role": "boris",
                        "part": "hand",
                    }
                )

    tree_positions = [
        (-2.6, -2.5, 0.18),
        (2.7, -3.2, 0.19),
        (-2.2, -5.8, 0.21),
        (2.5, -6.5, 0.20),
        (-3.2, -8.5, 0.24),
        (0.2, -9.2, 0.22),
        (3.1, -9.8, 0.25),
        (-1.6, -11.5, 0.26),
        (1.8, -12.0, 0.26),
    ]
    for tx, tz, trad in tree_positions:
        primitives.append(
            {
                "type": "capsule",
                "p0": np.array([tx, -0.1, tz], dtype=np.float32),
                "p1": np.array([tx, 4.5, tz], dtype=np.float32),
                "radius": trad,
                "role": "tree",
                "part": "trunk",
            }
        )

    return {
        "primitives": primitives,
        "boris_head_center": boris_head_center,
        "boris_head_dir": boris_head_dir,
        "rifle_muzzle_pos": rifle_muzzle_pos,
        "rifle_breech_pos": rifle_breech_pos,
    }


class FastAnalyticalRenderer:
    def __init__(self, width: int = WIDTH, height: int = HEIGHT, fov_deg: float = 30.0):
        self.w = width
        self.h = height
        self.aspect = float(width) / float(height)
        self.tan_half_fov = math.tan(math.radians(fov_deg) * 0.5)
        xs = (np.arange(width, dtype=np.float32) + 0.5) / float(width) * 2.0 - 1.0
        ys = 1.0 - (np.arange(height, dtype=np.float32) + 0.5) / float(height) * 2.0
        grid_x, grid_y = np.meshgrid(xs, ys)
        self.vx_full = grid_x * (self.tan_half_fov * self.aspect)
        self.vy_full = grid_y * self.tan_half_fov
        self.vv_full = self.vx_full * self.vx_full + self.vy_full * self.vy_full + 1.0

    def render_frame_pair(
        self, state: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        cam = state["camera"]
        cam_pos = cam["pos"]
        cam_target = cam["target"]
        tan_half_fov = self.tan_half_fov
        vx_full = self.vx_full
        vy_full = self.vy_full
        vv_full = self.vv_full

        forward = normalize(cam_target - cam_pos)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = normalize(np.cross(forward, world_up))
        up = normalize(np.cross(right, forward))
        R_cam_axes = np.stack([right, up, forward], axis=0)

        z_buf = np.full((self.h, self.w), FAR_DEPTH + 10.0, dtype=np.float32)
        role_buf = np.zeros((self.h, self.w), dtype=np.uint8)
        shade_buf = np.full((self.h, self.w), 0.15, dtype=np.float32)

        # 1. 三维地面 y = 0 求交
        dy_world = vx_full * right[1] + vy_full * up[1] + forward[1]
        valid_ground = dy_world < -1e-4
        z_ground = np.where(
            valid_ground,
            -cam_pos[1] / np.where(valid_ground, dy_world, -1.0),
            FAR_DEPTH + 10.0,
        )
        mask_g = valid_ground & (z_ground > 0.15) & (z_ground < FAR_DEPTH)
        z_buf[mask_g] = z_ground[mask_g]
        role_buf[mask_g] = 1

        def to_cam_point(p_w: np.ndarray) -> np.ndarray:
            return R_cam_axes @ (p_w - cam_pos)

        def project_bbox(c_cam: np.ndarray, bound_r: float) -> Tuple[int, int, int, int]:
            zc = float(c_cam[2])
            if zc + bound_r <= 0.15:
                return 0, 0, 0, 0
            z_near = max(0.15, zc - bound_r)
            cx_ndc = float(c_cam[0]) / (zc * tan_half_fov * self.aspect)
            cy_ndc = float(c_cam[1]) / (zc * tan_half_fov)
            rx_ndc = (bound_r * 1.06) / (z_near * tan_half_fov * self.aspect)
            ry_ndc = (bound_r * 1.06) / (z_near * tan_half_fov)
            x0 = max(0, int(math.floor(((cx_ndc - rx_ndc) * 0.5 + 0.5) * self.w)) - 1)
            x1 = min(self.w, int(math.ceil(((cx_ndc + rx_ndc) * 0.5 + 0.5) * self.w)) + 1)
            y0 = max(0, int(math.floor((0.5 - (cy_ndc + ry_ndc) * 0.5) * self.h)) - 1)
            y1 = min(self.h, int(math.ceil((0.5 - (cy_ndc - ry_ndc) * 0.5) * self.h)) + 1)
            return x0, x1, y0, y1

        role_map = {"ground": 1, "tree": 2, "follower": 3, "boris": 4, "rifle": 5}

        for prim in state["primitives"]:
            ptype = prim["type"]
            r_code = role_map.get(prim["role"], 3)

            if ptype == "sphere":
                c = to_cam_point(prim["center"])
                rad = float(prim["radius"])
                x0, x1, y0, y1 = project_bbox(c, rad)
                if x0 >= x1 or y0 >= y1:
                    continue
                vx = vx_full[y0:y1, x0:x1]
                vy = vy_full[y0:y1, x0:x1]
                vv = vv_full[y0:y1, x0:x1]
                b_half = -(vx * c[0] + vy * c[1] + c[2])
                c_term = float(np.dot(c, c)) - rad * rad
                disc = b_half * b_half - vv * c_term
                hit = disc >= 0.0
                if not np.any(hit):
                    continue
                z_hit = (-b_half - np.sqrt(np.maximum(disc, 0.0))) / vv
                valid = hit & (z_hit > 0.15) & (z_hit < z_buf[y0:y1, x0:x1])
                if np.any(valid):
                    z_sub = z_buf[y0:y1, x0:x1]
                    r_sub = role_buf[y0:y1, x0:x1]
                    s_sub = shade_buf[y0:y1, x0:x1]
                    z_sub[valid] = z_hit[valid]
                    r_sub[valid] = r_code
                    ny = (vy * z_hit - c[1]) / rad
                    nz = -(z_hit - c[2]) / rad
                    s_sub[valid] = np.clip(0.35 + 0.45 * nz[valid] + 0.20 * ny[valid], 0.15, 1.0)

            elif ptype == "capsule":
                a_pt = to_cam_point(prim["p0"])
                b_pt = to_cam_point(prim["p1"])
                rad = float(prim["radius"])
                mid = 0.5 * (a_pt + b_pt)
                ba = b_pt - a_pt
                seg_len = float(np.linalg.norm(ba))
                x0, x1, y0, y1 = project_bbox(mid, 0.5 * seg_len + rad)
                if x0 >= x1 or y0 >= y1:
                    continue
                vx = vx_full[y0:y1, x0:x1]
                vy = vy_full[y0:y1, x0:x1]
                vv = vv_full[y0:y1, x0:x1]
                ba_dot_ba = seg_len * seg_len + 1e-8
                v_dot_a = vx * a_pt[0] + vy * a_pt[1] + a_pt[2]
                v_dot_ba = vx * ba[0] + vy * ba[1] + ba[2]
                a_dot_ba = float(np.dot(a_pt, ba))

                denom = np.maximum(vv * ba_dot_ba - v_dot_ba * v_dot_ba, 1e-6)
                z_est = (v_dot_a * ba_dot_ba - v_dot_ba * a_dot_ba) / denom
                u_est = np.clip((z_est * v_dot_ba - a_dot_ba) / ba_dot_ba, 0.0, 1.0)
                cx = a_pt[0] + u_est * ba[0]
                cy = a_pt[1] + u_est * ba[1]
                cz = a_pt[2] + u_est * ba[2]

                b_half = -(vx * cx + vy * cy + cz)
                c_term = cx * cx + cy * cy + cz * cz - rad * rad
                disc = b_half * b_half - vv * c_term
                hit = disc >= 0.0
                if not np.any(hit):
                    continue
                z_hit = (-b_half - np.sqrt(np.maximum(disc, 0.0))) / vv
                valid = hit & (z_hit > 0.15) & (z_hit < z_buf[y0:y1, x0:x1])
                if np.any(valid):
                    z_sub = z_buf[y0:y1, x0:x1]
                    r_sub = role_buf[y0:y1, x0:x1]
                    s_sub = shade_buf[y0:y1, x0:x1]
                    z_sub[valid] = z_hit[valid]
                    r_sub[valid] = r_code
                    nz = -(z_hit - cz) / rad
                    s_sub[valid] = np.clip(0.32 + 0.52 * nz[valid], 0.15, 1.0)

            elif ptype == "box":
                c = to_cam_point(prim["center"])
                R_c = R_cam_axes @ prim["rot"]
                he = prim["half_extents"]
                bound_r = float(np.linalg.norm(he))
                x0, x1, y0, y1 = project_bbox(c, bound_r)
                if x0 >= x1 or y0 >= y1:
                    continue
                vx = vx_full[y0:y1, x0:x1]
                vy = vy_full[y0:y1, x0:x1]

                o_loc = -(R_c.T @ c)
                dx = R_c[0, 0] * vx + R_c[1, 0] * vy + R_c[2, 0]
                dy = R_c[0, 1] * vx + R_c[1, 1] * vy + R_c[2, 1]
                dz = R_c[0, 2] * vx + R_c[1, 2] * vy + R_c[2, 2]

                inv_dx = 1.0 / np.where(np.abs(dx) > 1e-6, dx, 1e-6)
                inv_dy = 1.0 / np.where(np.abs(dy) > 1e-6, dy, 1e-6)
                inv_dz = 1.0 / np.where(np.abs(dz) > 1e-6, dz, 1e-6)

                tx1 = (-he[0] - o_loc[0]) * inv_dx
                tx2 = (he[0] - o_loc[0]) * inv_dx
                tmin = np.minimum(tx1, tx2)
                tmax = np.maximum(tx1, tx2)

                ty1 = (-he[1] - o_loc[1]) * inv_dy
                ty2 = (he[1] - o_loc[1]) * inv_dy
                tmin = np.maximum(tmin, np.minimum(ty1, ty2))
                tmax = np.minimum(tmax, np.maximum(ty1, ty2))

                tz1 = (-he[2] - o_loc[2]) * inv_dz
                tz2 = (he[2] - o_loc[2]) * inv_dz
                tmin = np.maximum(tmin, np.minimum(tz1, tz2))
                tmax = np.minimum(tmax, np.maximum(tz1, tz2))

                valid = (tmax >= np.maximum(tmin, 0.15)) & (tmin > 0.15) & (tmin < z_buf[y0:y1, x0:x1])
                if np.any(valid):
                    z_sub = z_buf[y0:y1, x0:x1]
                    r_sub = role_buf[y0:y1, x0:x1]
                    s_sub = shade_buf[y0:y1, x0:x1]
                    z_sub[valid] = tmin[valid]
                    r_sub[valid] = r_code
                    s_sub[valid] = 0.68 if r_code == 5 else 0.58

        # 3. 主交付：纯净灰度深度图
        t_norm = np.clip((z_buf - NEAR_DEPTH) / (FAR_DEPTH - NEAR_DEPTH), 0.0, 1.0)
        depth_val = np.where(z_buf <= FAR_DEPTH, np.power(1.0 - t_norm, 1.35), 0.02)
        depth_gray_u8 = np.clip(depth_val * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)

        # 4. 辅助交付：动作与移焦预览图
        preview_rgb = np.empty((self.h, self.w, 3), dtype=np.float32)
        bg_col = np.array([0.11, 0.13, 0.16], dtype=np.float32)
        preview_rgb[:] = bg_col

        palette = {
            1: np.array([0.22, 0.25, 0.30], dtype=np.float32),
            2: np.array([0.18, 0.22, 0.26], dtype=np.float32),
            3: np.array([0.48, 0.56, 0.66], dtype=np.float32),
            4: np.array([0.74, 0.80, 0.88], dtype=np.float32),
            5: np.array([0.88, 0.72, 0.46], dtype=np.float32),
        }
        for code, base_c in palette.items():
            m = role_buf == code
            if np.any(m):
                sh = shade_buf[m][:, None]
                preview_rgb[m] = base_c[None, :] * (0.45 + 0.55 * sh)

        dz_x = np.abs(z_buf[:, 1:] - z_buf[:, :-1])
        dz_y = np.abs(z_buf[1:, :] - z_buf[:-1, :])
        edge_mask = np.zeros((self.h, self.w), dtype=np.float32)
        edge_mask[:, :-1] = np.maximum(edge_mask[:, :-1], np.clip(dz_x * 3.0, 0.0, 1.0))
        edge_mask[:-1, :] = np.maximum(edge_mask[:-1, :], np.clip(dz_y * 3.0, 0.0, 1.0))
        preview_rgb = preview_rgb * (1.0 - 0.35 * edge_mask[:, :, None]) + 0.35 * edge_mask[
            :, :, None
        ] * np.array([0.92, 0.96, 1.0], dtype=np.float32)

        # 快速 1/4 分辨率高斯模糊用于表现 10-14s 焦点从鲍里斯脸部移至枪口
        focus_z = state["focus_depth_z"]
        sharp_u8 = np.clip(preview_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        pil_small = Image.fromarray(sharp_u8).resize((self.w // 4, self.h // 4), Image.BILINEAR)
        pil_blur = pil_small.filter(ImageFilter.GaussianBlur(radius=1.8)).resize(
            (self.w, self.h), Image.BILINEAR
        )
        blur_rgb = np.asarray(pil_blur, dtype=np.float32) / 255.0

        dof_half_width = 0.25 + (1.0 - state["focus_shift_u"]) * 0.25
        coc = np.clip((np.abs(z_buf - focus_z) - 0.10) / dof_half_width, 0.0, 1.0)
        far_fade = np.clip((z_buf - 2.6) / 9.5, 0.0, 0.78)
        blend_blur = np.maximum(coc * 0.88, far_fade)
        preview_focused = (
            preview_rgb * (1.0 - blend_blur[:, :, None]) + blur_rgb * blend_blur[:, :, None]
        )
        preview_focused = preview_focused * (1.0 - 0.32 * far_fade[:, :, None])

        preview_pil = Image.fromarray(
            np.clip(preview_focused * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        )
        draw = ImageDraw.Draw(preview_pil)

        def project_world_to_screen(pt_w: np.ndarray) -> Tuple[float, float, float]:
            c_pt = to_cam_point(pt_w)
            if c_pt[2] <= 0.15:
                return -999.0, -999.0, float(c_pt[2])
            nx = float(c_pt[0]) / (float(c_pt[2]) * tan_half_fov * self.aspect)
            ny = float(c_pt[1]) / (float(c_pt[2]) * tan_half_fov)
            sx = (nx * 0.5 + 0.5) * self.w
            sy = (0.5 - ny * 0.5) * self.h
            return sx, sy, float(c_pt[2])

        hc = state["boris_head_center"]
        hd = state["boris_head_dir"]
        line_p0 = hc + hd * 0.13
        line_p1 = hc + hd * 0.55
        x_a, y_a, z_a = project_world_to_screen(line_p0)
        x_b, y_b, z_b = project_world_to_screen(line_p1)
        if z_a > 0.2 and z_b > 0.2:
            draw.line([(x_a, y_a), (x_b, y_b)], fill=(91, 157, 255), width=4)
            draw.ellipse([x_b - 5, y_b - 5, x_b + 5, y_b + 5], fill=(255, 210, 90))

        return depth_gray_u8, np.asarray(preview_pil, dtype=np.uint8)


def open_ffmpeg_writer(
    output_path: str, width: int = WIDTH, height: int = HEIGHT, fps: int = FPS
) -> subprocess.Popen:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cmd = [
        ffmpeg_exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def generate_videos() -> Dict[str, Any]:
    t0 = time.perf_counter()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    export_dir = os.path.join(base_dir, "exports")
    os.makedirs(export_dir, exist_ok=True)

    depth_mp4 = os.path.join(export_dir, "five_men_15s_depth_map_1280x720_24fps.mp4")
    preview_mp4 = os.path.join(
        export_dir, "five_men_15s_action_focus_preview_1280x720_24fps.mp4"
    )

    renderer = FastAnalyticalRenderer(width=WIDTH, height=HEIGHT)
    proc_depth = open_ffmpeg_writer(depth_mp4, WIDTH, HEIGHT, FPS)
    proc_prev = open_ffmpeg_writer(preview_mp4, WIDTH, HEIGHT, FPS)

    for i in range(TOTAL_FRAMES):
        t_sec = float(i) / float(FPS)
        state = evaluate_at_time(t_sec)
        depth_u8, prev_rgb = renderer.render_frame_pair(state)
        depth_rgb = np.repeat(depth_u8[:, :, None], 3, axis=2)
        proc_depth.stdin.write(depth_rgb.tobytes())
        proc_prev.stdin.write(prev_rgb.tobytes())

    proc_depth.stdin.close()
    proc_prev.stdin.close()
    err_d = proc_depth.stderr.read().decode("utf-8", errors="ignore")
    err_p = proc_prev.stderr.read().decode("utf-8", errors="ignore")
    proc_depth.wait()
    proc_prev.wait()
    if proc_depth.returncode != 0:
        raise RuntimeError(f"Depth MP4 encode failed: {err_d}")
    if proc_prev.returncode != 0:
        raise RuntimeError(f"Preview MP4 encode failed: {err_p}")

    elapsed = time.perf_counter() - t0
    res = {
        "depth_mp4": depth_mp4,
        "preview_mp4": preview_mp4,
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "frames": TOTAL_FRAMES,
        "last_sample_t": round((TOTAL_FRAMES - 1) / float(FPS), 4),
        "elapsed_sec": round(elapsed, 2),
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


if __name__ == "__main__":
    generate_videos()
