# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy",
#     "pillow",
#     "imageio-ffmpeg",
# ]
# ///
"""
Magic Director Stage 3D Engine (director_engine.py)
核心设计参考【魔法人形师 (Magic Poser)】+【Blender Python API】：
1. 点拉式 IK 控制球 (Point & Drag Inverse Kinematics)：
   - 无需繁琐逐根旋转骨骼！直接拖拽或用 Python 指定 8 个核心控制球的世界/局部坐标：
     * 腰部重心球 (pelvis) —— 往下拉即可自动屈膝下蹲/坐下（支持脚底图钉锁定 Pin Feet）
     * 左/右手腕控制球 (hand_L, hand_R) —— 拖动即自动解算大臂与小臂弯曲
     * 左/右脚踝控制球 (foot_L, foot_R) —— 拖动即自动解算大腿与小腿弯曲
     * 胸腔/颈部球 (chest) 与 头部朝向球 (head_target)
     * 手肘/膝盖极向微调球 (elbow_L/R_pole, knee_L/R_pole)
2. 素体木偶人白模 (Articulated Mannequin) + OpenPose 骨骼双模显示。
3. 彩色基础几何体建造 (长方体 box, 圆锥体 cone, 三角柱 prism, 圆柱 cylinder)。
4. 上传参考图一键提取 3D 关键点并驱动魔法人形师 IK 控制球。
5. Python + NumPy + Pillow + imageio-ffmpeg 极速软光栅管线，1 秒导出 MP4 白模/骨骼视频。
"""

import base64
import copy
import io
import json
import math
import multiprocessing
import os
import re
import subprocess
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import imageio_ffmpeg
from keyframe_policy import sparse_generated_scene
from motion_speed_policy import (
    analyze_track_speed_profile,
    compute_wheel_roll_deg_at_time,
    validate_scene_speed_policy,
)
from motion_continuity import (
    DEFAULT_PROTECTION_LIMITS,
    build_track_continuity_plan,
    compute_path_curvature_samples,
    eval_segments_at_time,
    eval_track_continuity_at_time,
    quantize_time_to_frame,
    solve_continuous_velocity_profile,
)
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ============================================================================
# 1. 魔法人形师 (Magic Poser) 标准素体解剖尺寸与预设 IK 姿态库
# 角色局部空间原点 [0, 0, 0] 在角色脚底地面中心 (X右, Y上, Z前)
# 标准身高: 1.72m
#   - 大腿长 L_thigh = 0.43m, 小腿长 L_calf = 0.42m (腿全长 0.85m)
#   - 躯干长 L_spine = 0.46m (骨盆到胸腔颈根)
#   - 大臂长 L_upper_arm = 0.28m, 小臂长 L_forearm = 0.26m (臂全长 0.54m)
# ============================================================================

MANNEQUIN_SPEC = {
    "thigh_len": 0.43,
    "calf_len": 0.42,
    "spine_len": 0.46,
    "upper_arm_len": 0.28,
    "forearm_len": 0.26,
    "hip_half_width": 0.105,
    "shoulder_half_width": 0.195,
    "head_radius": 0.115,
}

# Neutral standing body bounds (X width, Y height, Z depth), measured from
# the Three.js body meshes only; exclude labels, controls and outline shells.
CHARACTER_NEUTRAL_DIMENSIONS = {
    "mannequin": (0.6020078202985382, 1.5932627991782906, 0.29229958169663256),
    "pawn": (0.41999998688697815, 1.6699900248630026, 0.41999998688697815),
}

# 魔法人形师 IK 控制球预设库 (所有坐标均为相对于角色脚底原点的局部米制坐标 [x, y, z])
PRESET_IK_POSES: Dict[str, Dict[str, Any]] = {
    "sitting": {
        "name": "标准坐姿 (Sitting on Chair)",
        "targets": {
            "pelvis": [0.0, 0.48, -0.04],
            "chest": [0.0, 0.93, -0.01],
            "head_target": [0.0, 1.18, 0.65],
            "hand_L": [0.16, 0.54, 0.24],
            "hand_R": [-0.16, 0.54, 0.24],
            "foot_L": [0.13, 0.04, 0.36],
            "foot_R": [-0.13, 0.04, 0.36],
            "knee_L_pole": [0.15, 0.52, 0.55],
            "knee_R_pole": [-0.15, 0.52, 0.55],
            "elbow_L_pole": [0.36, 0.66, -0.15],
            "elbow_R_pole": [-0.36, 0.66, -0.15],
        },
        "pins": {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False},
    },
    "sitting_one_leg_forward": {
        "name": "坐地单腿朝前 (Ground Sit One-Leg Forward)",
        "targets": {
            "pelvis": [0.0, 0.14, -0.08],
            "chest": [0.0, 0.57, -0.10],
            "head_target": [0.0, 0.84, 0.65],
            "hand_L": [0.17, 0.44, 0.28],
            "hand_R": [-0.24, 0.05, -0.16],
            "foot_L": [0.15, 0.04, 0.34],
            "foot_R": [-0.14, 0.06, 0.72],
            "knee_L_pole": [0.17, 0.52, 0.42],
            "knee_R_pole": [-0.14, 0.24, 0.42],
            "elbow_L_pole": [0.38, 0.42, -0.02],
            "elbow_R_pole": [-0.38, 0.32, -0.26],
        },
        "pins": {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": True},
    },
    "sitting_relaxed": {
        "name": "二郎腿慵懒坐姿 (Crossed-Leg Sit)",
        "targets": {
            "pelvis": [0.0, 0.47, -0.05],
            "chest": [-0.03, 0.90, -0.14],
            "head_target": [0.08, 1.15, 0.60],
            "hand_L": [0.32, 0.48, -0.10],
            "hand_R": [-0.08, 0.62, 0.22],
            "foot_L": [0.12, 0.04, 0.38],
            "foot_R": [0.18, 0.42, 0.32],
            "knee_L_pole": [0.12, 0.50, 0.55],
            "knee_R_pole": [-0.05, 0.68, 0.50],
            "elbow_L_pole": [0.40, 0.68, -0.25],
            "elbow_R_pole": [-0.35, 0.70, -0.10],
        },
        "pins": {"foot_L": True, "foot_R": False, "hand_L": True, "hand_R": False},
    },
    "sitting_think": {
        "name": "托腮沉思坐姿 (Thinking Pose)",
        "targets": {
            "pelvis": [0.0, 0.48, -0.05],
            "chest": [0.0, 0.89, 0.10],
            "head_target": [0.0, 1.06, 0.65],
            "hand_L": [0.06, 1.04, 0.16],
            "hand_R": [-0.14, 0.56, 0.24],
            "foot_L": [0.14, 0.04, 0.35],
            "foot_R": [-0.14, 0.04, 0.35],
            "knee_L_pole": [0.16, 0.55, 0.55],
            "knee_R_pole": [-0.16, 0.55, 0.55],
            "elbow_L_pole": [0.16, 0.56, 0.32],
            "elbow_R_pole": [-0.35, 0.68, -0.05],
        },
        "pins": {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False},
    },
    "standing": {
        "name": "站姿 (Standing)",
        "targets": {
            "pelvis": [0.0, 0.86, 0.0],
            "chest": [0.0, 1.32, 0.0],
            "head_target": [0.0, 1.56, 0.70],
            "hand_L": [0.22, 0.78, 0.01],
            "hand_R": [-0.22, 0.78, 0.01],
            "foot_L": [0.09, 0.04, 0.0],
            "foot_R": [-0.09, 0.04, 0.0],
            "knee_L_pole": [0.09, 0.46, 0.40],
            "knee_R_pole": [-0.09, 0.46, 0.40],
            "elbow_L_pole": [0.32, 1.05, -0.20],
            "elbow_R_pole": [-0.32, 1.05, -0.20],
        },
        "pins": {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False},
    },
    "lying": {
        "name": "躺下姿势 (Lying Down)",
        "targets": {
            "pelvis": [0.0, 0.12, 0.0],
            "chest": [0.0, 0.14, -0.46],
            "head_target": [0.0, 0.90, -0.58],
            "hand_L": [0.26, 0.10, -0.02],
            "hand_R": [-0.26, 0.10, -0.02],
            "foot_L": [0.12, 0.10, 0.82],
            "foot_R": [-0.12, 0.10, 0.82],
            "knee_L_pole": [0.12, 0.45, 0.42],
            "knee_R_pole": [-0.12, 0.45, 0.42],
            "elbow_L_pole": [0.38, 0.08, -0.26],
            "elbow_R_pole": [-0.38, 0.08, -0.26],
        },
        "pins": {"foot_L": False, "foot_R": False, "hand_L": False, "hand_R": False},
    },
    "tpose": {
        "name": "T-Pose (标准绑定姿)",
        "targets": {
            "pelvis": [0.0, 0.87, 0.0],
            "chest": [0.0, 1.33, 0.0],
            "head_target": [0.0, 1.56, 0.70],
            "hand_L": [0.72, 1.31, 0.0],
            "hand_R": [-0.72, 1.31, 0.0],
            "foot_L": [0.11, 0.04, 0.0],
            "foot_R": [-0.11, 0.04, 0.0],
            "knee_L_pole": [0.11, 0.46, 0.40],
            "knee_R_pole": [-0.11, 0.46, 0.40],
            "elbow_L_pole": [0.45, 1.31, -0.35],
            "elbow_R_pole": [-0.45, 1.31, -0.35],
        },
        "pins": {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False},
    },
    "running": {
        "name": "冲刺奔跑 (Dynamic Run)",
        "targets": {
            "pelvis": [0.0, 0.76, 0.05],
            "chest": [0.0, 1.18, 0.22],
            "head_target": [0.0, 1.42, 0.90],
            "hand_L": [0.22, 1.12, -0.22],
            "hand_R": [-0.20, 1.18, 0.36],
            "foot_L": [0.12, 0.16, 0.38],
            "foot_R": [-0.12, 0.28, -0.36],
            "knee_L_pole": [0.12, 0.55, 0.65],
            "knee_R_pole": [-0.12, 0.42, 0.20],
            "elbow_L_pole": [0.38, 1.05, -0.45],
            "elbow_R_pole": [-0.38, 0.96, -0.05],
        },
        "pins": {"foot_L": False, "foot_R": False, "hand_L": False, "hand_R": False},
    },
    "fighting": {
        "name": "格斗抱架 (Fighting Guard)",
        "targets": {
            "pelvis": [0.0, 0.74, 0.0],
            "chest": [0.02, 1.18, 0.08],
            "head_target": [0.0, 1.40, 0.80],
            "hand_L": [0.14, 1.28, 0.32],
            "hand_R": [-0.10, 1.22, 0.22],
            "foot_L": [0.16, 0.04, 0.24],
            "foot_R": [-0.18, 0.04, -0.22],
            "knee_L_pole": [0.22, 0.45, 0.55],
            "knee_R_pole": [-0.22, 0.42, 0.20],
            "elbow_L_pole": [0.34, 0.92, 0.05],
            "elbow_R_pole": [-0.32, 0.90, -0.05],
        },
        "pins": {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False},
    },
    "kneeling": {
        "name": "单膝跪地 (Hero Kneel)",
        "targets": {
            "pelvis": [0.0, 0.46, 0.02],
            "chest": [0.0, 0.90, 0.12],
            "head_target": [0.0, 1.15, 0.75],
            "hand_L": [0.15, 0.54, 0.32],
            "hand_R": [-0.22, 0.06, 0.18],
            "foot_L": [0.14, 0.04, 0.36],
            "foot_R": [-0.14, 0.06, -0.34],
            "knee_L_pole": [0.16, 0.52, 0.60],
            "knee_R_pole": [-0.14, 0.04, 0.12],
            "elbow_L_pole": [0.36, 0.72, -0.05],
            "elbow_R_pole": [-0.36, 0.55, -0.05],
        },
        "pins": {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": True},
    },
    "lying_supine": {
        "name": "仰面平躺 (Supine Lying)",
        "targets": {
            "pelvis": [0.0, 0.11, 0.0],
            "chest": [0.0, 0.12, -0.45],
            "head_target": [-0.08, 0.85, -0.52],
            "hand_L": [0.34, 0.08, -0.05],
            "hand_R": [-0.36, 0.08, 0.04],
            "foot_L": [0.17, 0.10, 0.82],
            "foot_R": [-0.19, 0.10, 0.82],
            "knee_L_pole": [0.19, 0.45, 0.42],
            "knee_R_pole": [-0.21, 0.45, 0.42],
            "elbow_L_pole": [0.42, 0.16, -0.26],
            "elbow_R_pole": [-0.42, 0.16, -0.24],
            "shoulder_L_offset": [0.0, 0.0, 0.0],
            "shoulder_R_offset": [0.0, 0.0, 0.0],
        },
        "pins": {"foot_L": True, "foot_R": True, "hand_L": True, "hand_R": True},
    },
    # ========================================================================
    # 拿枪姿势形态库（6 种标准解剖学持枪姿态：+X 为左半身，-X 为右半身，+Z 为正前）
    # 配合 compute_gun_transform_from_hands 自动将枪托抵入右肩窝、右手握机匣握把、左手托前护木
    # ========================================================================
    "rifle_aim_cheek": {
        "name": "抵肩贴腮瞄准 (Cheek-Weld Rifle Aim)",
        "torso_twist": -16.0,
        "targets": {
            "pelvis": [0.0, 0.85, 0.0],
            "chest": [0.0, 1.30, 0.03],
            "head_target": [0.04, 1.38, 0.96],
            "hand_R": [-0.08, 1.29, 0.21],
            "hand_L": [-0.03, 1.28, 0.47],
            "foot_L": [0.11, 0.04, 0.08],
            "foot_R": [-0.12, 0.04, -0.08],
            "knee_L_pole": [0.12, 0.46, 0.42],
            "knee_R_pole": [-0.12, 0.46, 0.36],
            "elbow_R_pole": [-0.32, 1.04, 0.05],
            "elbow_L_pole": [0.14, 1.00, 0.26],
            "shoulder_L_offset": [0.0, 0.0, 0.03],
            "shoulder_R_offset": [0.0, 0.01, -0.02],
        },
        "pins": {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False},
    },
    "rifle_aim_tree_cover": {
        "name": "树后探身瞄准 (Tree-Cover Lean Aim)",
        "torso_twist": -14.0,
        "targets": {
            "pelvis": [0.0, 0.83, -0.02],
            "chest": [-0.02, 1.28, 0.06],
            "head_target": [0.02, 1.36, 0.98],
            "hand_R": [-0.08, 1.28, 0.23],
            "hand_L": [-0.03, 1.29, 0.49],
            "foot_L": [0.12, 0.04, 0.10],
            "foot_R": [-0.12, 0.04, -0.08],
            "knee_L_pole": [0.13, 0.45, 0.44],
            "knee_R_pole": [-0.13, 0.45, 0.36],
            "elbow_R_pole": [-0.32, 1.02, 0.06],
            "elbow_L_pole": [0.14, 0.98, 0.28],
            "shoulder_L_offset": [0.0, 0.0, 0.03],
            "shoulder_R_offset": [0.0, 0.01, -0.02],
        },
        "pins": {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False},
    },
    "rifle_low_ready": {
        "name": "胸前低姿戒备 (Low-Ready Alert Carry)",
        "torso_twist": -10.0,
        "targets": {
            "pelvis": [0.0, 0.85, 0.0],
            "chest": [0.0, 1.30, 0.03],
            "head_target": [0.0, 1.50, 0.85],
            "hand_R": [-0.13, 1.10, 0.20],
            "hand_L": [0.05, 1.02, 0.40],
            "foot_L": [0.10, 0.04, 0.06],
            "foot_R": [-0.11, 0.04, -0.06],
            "knee_L_pole": [0.11, 0.46, 0.40],
            "knee_R_pole": [-0.11, 0.46, 0.38],
            "elbow_R_pole": [-0.34, 0.92, 0.02],
            "elbow_L_pole": [0.24, 0.88, 0.18],
            "shoulder_L_offset": [0.0, 0.0, 0.02],
            "shoulder_R_offset": [0.0, 0.0, -0.01],
        },
        "pins": {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False},
    },
    "rifle_crouch_aim": {
        "name": "战术半蹲瞄准 (Tactical Crouch Aim)",
        "torso_twist": -15.0,
        "targets": {
            "pelvis": [0.0, 0.68, -0.04],
            "chest": [0.0, 1.11, 0.08],
            "head_target": [0.03, 1.20, 0.95],
            "hand_R": [-0.09, 1.07, 0.25],
            "hand_L": [-0.04, 1.08, 0.50],
            "foot_L": [0.12, 0.04, 0.14],
            "foot_R": [-0.12, 0.04, -0.10],
            "knee_L_pole": [0.14, 0.44, 0.52],
            "knee_R_pole": [-0.14, 0.42, 0.38],
            "elbow_R_pole": [-0.32, 0.82, 0.08],
            "elbow_L_pole": [0.16, 0.78, 0.28],
            "shoulder_L_offset": [0.0, 0.0, 0.03],
            "shoulder_R_offset": [0.0, 0.01, -0.02],
        },
        "pins": {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False},
    },
    "rifle_flinch_duck": {
        "name": "受惊缩肩收枪 (Flinch & Duck Low)",
        "torso_twist": -8.0,
        "targets": {
            "pelvis": [0.0, 0.74, -0.03],
            "chest": [0.0, 1.14, 0.12],
            "head_target": [0.0, 1.16, 0.78],
            "hand_R": [-0.12, 0.98, 0.22],
            "hand_L": [0.08, 0.94, 0.36],
            "foot_L": [0.12, 0.04, 0.08],
            "foot_R": [-0.12, 0.04, -0.06],
            "knee_L_pole": [0.14, 0.45, 0.46],
            "knee_R_pole": [-0.14, 0.45, 0.40],
            "elbow_R_pole": [-0.32, 0.82, 0.04],
            "elbow_L_pole": [0.24, 0.80, 0.16],
            "shoulder_L_offset": [0.0, 0.04, 0.03],
            "shoulder_R_offset": [0.0, 0.04, 0.02],
        },
        "pins": {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False},
    },
    "rifle_patrol_carry": {
        "name": "双手横抱持枪 (Patrol Cross-Chest Carry)",
        "torso_twist": -6.0,
        "targets": {
            "pelvis": [0.0, 0.86, 0.0],
            "chest": [0.0, 1.31, 0.02],
            "head_target": [0.0, 1.54, 0.82],
            "hand_R": [-0.15, 1.02, 0.19],
            "hand_L": [0.11, 1.05, 0.32],
            "foot_L": [0.10, 0.04, 0.04],
            "foot_R": [-0.10, 0.04, -0.04],
            "knee_L_pole": [0.11, 0.46, 0.40],
            "knee_R_pole": [-0.11, 0.46, 0.40],
            "elbow_R_pole": [-0.34, 0.88, -0.02],
            "elbow_L_pole": [0.28, 0.88, 0.12],
            "shoulder_L_offset": [0.0, 0.0, 0.01],
            "shoulder_R_offset": [0.0, 0.0, 0.0],
        },
        "pins": {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False},
    },
}


# ============================================================================
# 2. 魔法人形师双骨解析 IK 求解器 (Analytical Two-Bone IK Solver)
# ============================================================================

def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return v / n


def solve_two_bone_ik(
    root_pos: np.ndarray,
    target_pos: np.ndarray,
    pole_pos: np.ndarray,
    len1: float,
    len2: float,
    soft_ratio: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    给定根关节 root_pos(如髋/肩)、末端目标球 target_pos(如脚踝/手腕)、极向球 pole_pos(如膝盖/手肘朝向)，
    使用余弦定理精确解算中间关节 mid_pos(膝盖/手肘) 和真实末端 end_pos 的 3D 坐标。
    soft_ratio > 0 时开启 C1 闭式 Soft IK 阻尼缓冲（消除关节接近伸直瞬间的弹跳）。
    """
    diff = target_pos - root_pos
    dist = float(np.linalg.norm(diff))
    max_reach = (len1 + len2) - 1e-4
    min_reach = abs(len1 - len2) + 1e-4

    forward = _normalize(diff) if dist > 1e-6 else np.array([0.0, -1.0, 0.0], dtype=np.float64)
    soft_active = False
    if soft_ratio > 1e-5:
        d_soft = float(soft_ratio) * (len1 + len2)
        d_thresh = max(min_reach, max_reach - d_soft)
        if dist > d_thresh:
            eff_dist = d_thresh + d_soft * (1.0 - math.exp(-(dist - d_thresh) / max(1e-5, d_soft)))
            clamped_dist = min(max_reach, max(min_reach, eff_dist))
            soft_active = True
        else:
            clamped_dist = max(min_reach, dist)
    else:
        clamped_dist = min(max_reach, max(min_reach, dist))

    # 余弦定理求根关节弯曲角 alpha
    cos_alpha = (len1 * len1 + clamped_dist * clamped_dist - len2 * len2) / (2.0 * len1 * clamped_dist)
    cos_alpha = min(1.0, max(-1.0, cos_alpha))
    sin_alpha = math.sqrt(max(0.0, 1.0 - cos_alpha * cos_alpha))

    # 计算极向平面正交方向 bend_dir
    pole_vec = pole_pos - root_pos
    proj = pole_vec - np.dot(pole_vec, forward) * forward
    if np.linalg.norm(proj) < 1e-5:
        fallback = np.array([0.0, 0.0, 1.0]) if abs(forward[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        proj = fallback - np.dot(fallback, forward) * forward
    bend_dir = _normalize(proj)

    mid_pos = root_pos + forward * (len1 * cos_alpha) + bend_dir * (len1 * sin_alpha)
    end_pos = root_pos + forward * clamped_dist if (soft_active or dist > max_reach or dist < min_reach) else target_pos.copy()
    return mid_pos, end_pos


def _get_char_scale_xyz(char_dict: Dict[str, Any]) -> Tuple[float, float, float]:
    sc_raw = char_dict.get("scale", [1.0, 1.0, 1.0])
    if isinstance(sc_raw, (list, tuple)) and len(sc_raw) >= 1:
        sx = max(0.05, float(sc_raw[0] or 1.0))
        sy = max(0.05, float((sc_raw[1] if len(sc_raw) > 1 else sc_raw[0]) or 1.0))
        sz = max(0.05, float((sc_raw[2] if len(sc_raw) > 2 else sc_raw[0]) or 1.0))
        return sx, sy, sz
    s = max(0.05, float(sc_raw or 1.0))
    return s, s, s


def _compute_ergonomic_knee_pole(
    p_hip: np.ndarray,
    p_foot: np.ndarray,
    p_raw_pole: np.ndarray,
    side_L: np.ndarray,
    spine_dir: np.ndarray,
    is_left: bool,
) -> np.ndarray:
    """
    人体工学膝关节极向约束：
    保证无论站立、坐姿、前抬腿或侧抬腿，膝盖始终朝前上方（spine_dir + char_fwd）屈曲，
    大腿自然抬起、小腿自然下垂（knee.y >= ankle.y），严禁膝关节向下/向后反折或小腿倒翘。
    """
    leg_diff = p_foot - p_hip
    leg_len = float(np.linalg.norm(leg_diff))
    leg_dir = leg_diff / leg_len if leg_len > 1e-5 else -spine_dir
    char_fwd = _normalize(np.cross(side_L, spine_dir))
    s_sign = 1.0 if is_left else -1.0
    side_out = side_L * s_sign

    x = float(np.dot(leg_dir, side_out))
    y = float(np.dot(leg_dir, spine_dir))
    z = float(np.dot(leg_dir, char_fwd))
    up_w = 0.82
    fwd_w = 0.62
    if y < -0.65 and z < -0.15:
        up_w = -0.20
        fwd_w = 0.92
    elif y > 0.15 and z > 0.15:
        up_w = 0.85
        fwd_w = -0.45

    ideal_raw = (
        spine_dir * up_w
        + char_fwd * fwd_w
        + side_out * (0.14 * abs(x))
    )
    ideal_raw = ideal_raw - float(np.dot(ideal_raw, leg_dir)) * leg_dir
    if float(np.linalg.norm(ideal_raw)) < 1e-4:
        ant_dir = char_fwd
    else:
        ant_dir = _normalize(ideal_raw)

    lat_raw = np.cross(leg_dir, ant_dir)
    lat_dir = _normalize(lat_raw) if float(np.linalg.norm(lat_raw)) > 1e-4 else side_out

    pole_vec = p_raw_pole - p_hip
    proj = pole_vec - float(np.dot(pole_vec, leg_dir)) * leg_dir
    ant_comp = float(np.dot(proj, ant_dir))
    lat_comp = float(np.dot(proj, lat_dir))

    if ant_comp <= 0.08 or abs(lat_comp) > 0.35 * ant_comp:
        had_pos = ant_comp > 0.08
        ant_comp = max(0.35, ant_comp)
        lat_comp = max(-0.28 * ant_comp, min(0.28 * ant_comp, lat_comp if had_pos else 0.0))

    return p_hip + ant_dir * ant_comp + lat_dir * lat_comp


def _compute_ergonomic_elbow_pole(
    p_shoulder: np.ndarray,
    p_hand: np.ndarray,
    p_raw_pole: np.ndarray,
    side_chest: np.ndarray,
    spine_dir: np.ndarray,
    is_left: bool,
) -> np.ndarray:
    """
    人体工学肘关节极向约束（仅用于关节人偶自然模式）：
    防止手肘极向与肩-腕向量共线奇点跳变，以及手臂下垂/前伸时手肘向内穿入肋骨躯干。
    """
    arm_diff = p_hand - p_shoulder
    arm_len = float(np.linalg.norm(arm_diff))
    arm_dir = arm_diff / arm_len if arm_len > 1e-5 else -spine_dir
    chest_fwd = _normalize(np.cross(side_chest, spine_dir))
    side_out = side_chest * (1.0 if is_left else -1.0)

    nat_raw = -spine_dir * 0.72 + side_out * 0.68 - chest_fwd * 0.36
    nat_proj = nat_raw - float(np.dot(nat_raw, arm_dir)) * arm_dir
    if float(np.linalg.norm(nat_proj)) < 1e-4:
        nat_proj = side_out - float(np.dot(side_out, arm_dir)) * arm_dir
    nat_dir = _normalize(nat_proj) if float(np.linalg.norm(nat_proj)) > 1e-5 else side_out

    pole_vec = p_raw_pole - p_shoulder
    proj = pole_vec - float(np.dot(pole_vec, arm_dir)) * arm_dir
    proj_len = float(np.linalg.norm(proj))

    # 1) C1 光滑共线防奇点混合（proj_len=0.12 处平滑过渡到 0）
    if proj_len < 0.12:
        t_col = max(0.0, min(1.0, proj_len / 0.12))
        w_col = 1.0 - t_col * t_col * (3.0 - 2.0 * t_col)
        proj = proj + nat_dir * (0.20 * w_col)

    # 2) C1 光滑防穿肋外推（随手臂外展/上举平滑衰减到 0，无任何阶跃阈值）
    def _smooth_hi(edge_lo: float, edge_hi: float, val: float) -> float:
        t = max(0.0, min(1.0, (edge_hi - val) / max(1e-5, edge_hi - edge_lo)))
        return t * t * (3.0 - 2.0 * t)

    arm_up = float(np.dot(arm_dir, spine_dir))
    arm_out = float(np.dot(arm_dir, side_out))
    w_arm = _smooth_hi(0.30, 0.72, arm_up) * _smooth_hi(0.30, 0.72, arm_out)
    inward_excess = max(0.0, -float(np.dot(proj, side_out)) - 0.08)
    if w_arm > 1e-4 and inward_excess > 1e-5:
        out_perp = side_out - float(np.dot(side_out, arm_dir)) * arm_dir
        out_dir = _normalize(out_perp) if float(np.linalg.norm(out_perp)) > 1e-4 else nat_dir
        proj = proj + _normalize(out_dir * 0.65 + nat_dir * 0.35) * (w_arm * inward_excess * 1.15)

    return p_shoulder + proj


def _clamp_shoulder_offset(raw: Any) -> np.ndarray:
    if not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) < 3:
        return np.zeros(3, dtype=np.float64)
    dx = max(-0.04, min(0.04, float(raw[0] or 0.0)))
    dy = max(-0.06, min(0.075, float(raw[1] or 0.0)))
    dz = max(-0.10, min(0.10, float(raw[2] or 0.0)))
    return np.array([dx, dy, dz], dtype=np.float64)


def solve_mannequin_ik(char_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Solve anatomy in local space, then apply the same 3-axis XYZ Euler + scale affine transform as Three.js."""
    canonical = dict(char_dict, pos=[0, 0, 0], rot=[0, 0, 0], scale=[1, 1, 1])
    result = _solve_mannequin_ik_base(canonical)
    sx, sy, sz = _get_char_scale_xyz(char_dict)
    rot_arr = char_dict.get("rot", [0, 0, 0]) or [0, 0, 0]
    rx = math.radians(float(rot_arr[0] if len(rot_arr) > 0 else 0.0))
    ry = math.radians(float(rot_arr[1] if len(rot_arr) > 1 else 0.0))
    rz = math.radians(float(rot_arr[2] if len(rot_arr) > 2 else 0.0))
    cx, sx_r = math.cos(rx), math.sin(rx)
    cy, sy_r = math.cos(ry), math.sin(ry)
    cz, sz_r = math.cos(rz), math.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx_r], [0.0, sx_r, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0.0, sy_r], [0.0, 1.0, 0.0], [-sy_r, 0.0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz_r, 0.0], [sz_r, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    R_xyz = Rx @ Ry @ Rz
    affine = R_xyz @ np.diag([sx, sy, sz])
    origin = np.asarray(char_dict.get("pos", [0, 0, 0]), dtype=np.float64)
    result["joints"] = {key: affine @ point + origin for key, point in result["joints"].items()}
    for key in ("spine_dir", "side_L", "side_chest", "look_dir", "head_left", "head_up"):
        if key in result["axes"]:
            result["axes"][key] = _normalize(affine @ result["axes"][key])
    result["axes"].update(scale=sy, sx=sx, sy=sy, sz=sz)
    return result


def compute_gun_transform_from_hands(char_dict: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    根据角色解算后的双手手腕世界坐标 (wrist_R 后握把 / wrist_L 前护木) 与右肩窝 (shoulder_R)，
    精确计算步枪的世界空间位置 pos 与欧拉角 rot (度)：
    - 步枪局部坐标系：z=-0.20 为枪托底板（抵入右肩窝），z=+0.02 为右手握把，z=+0.26 为左手前护木，z=+0.70 为枪口。
    - 枪管方向严格沿 wrist_R -> wrist_L 方向，枪身稳稳落在双手掌心上方 1.8cm 处，绝不脱手或穿模。
    """
    ik = solve_mannequin_ik(char_dict)
    joints = ik["joints"]
    w_r = np.asarray(joints["wrist_R"], dtype=np.float64)
    w_l = np.asarray(joints["wrist_L"], dtype=np.float64)
    sh_r = np.asarray(joints["shoulder_R"], dtype=np.float64)

    fwd_vec = w_l - w_r
    dist_hands = float(np.linalg.norm(fwd_vec))
    if dist_hands < 0.05:
        fwd = np.asarray(ik["axes"].get("look_dir", [0.0, 0.0, 1.0]), dtype=np.float64)
        fwd = _normalize(fwd)
    else:
        fwd = fwd_vec / dist_hands

    # 微调使枪托后端 (w_r - 0.20*fwd) 自然贴近右肩前窝，同时保持右手在 z=+0.02、左手在 z=+0.26
    grip_pt = w_r + np.array([0.0, 0.018, 0.0], dtype=np.float64)
    gun_origin = grip_pt - fwd * 0.02
    gun_pos = [round(float(gun_origin[0]), 4), round(float(gun_origin[1]), 4), round(float(gun_origin[2]), 4)]
    gun_rot = [round(float(x), 3) for x in _dir_to_euler_xyz_deg(fwd.tolist())]
    return {"pos": gun_pos, "rot": gun_rot, "forward": [round(float(x), 4) for x in fwd.tolist()]}


def _solve_mannequin_ik_base(char_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据角色当前的 IK 控制球 targets 解算全身所有关节的 3D 世界坐标。
    与前端 Three.js 中的 solveMannequinIK 保持 100% 相同的数学解算逻辑。
    """
    base_pos = np.asarray(char_dict.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
    rot_y_deg = float(char_dict.get("rot", [0.0, 0.0, 0.0])[1])
    sx, sy, sz = _get_char_scale_xyz(char_dict)
    scale = sy

    ry = math.radians(rot_y_deg)
    cy, s_y = math.cos(ry), math.sin(ry)
    R_root = np.array([[cy, 0, s_y], [0, 1, 0], [-s_y, 0, cy]], dtype=np.float64)

    def to_world(local_pt: List[float]) -> np.ndarray:
        v = np.array([float(local_pt[0]) * sx, float(local_pt[1]) * sy, float(local_pt[2]) * sz], dtype=np.float64)
        return base_pos + (R_root @ v)

    targets = char_dict.get("targets", PRESET_IK_POSES["sitting"]["targets"])

    p_pelvis = to_world(targets.get("pelvis", [0.0, 0.48, -0.04]))
    p_chest_raw = to_world(targets.get("chest", [0.0, 0.93, -0.01]))
    p_head_target = to_world(targets.get("head_target", [0.0, 1.18, 0.65]))

    # 锁定脊柱长度
    spine_dir = _normalize(p_chest_raw - p_pelvis)
    if float(np.linalg.norm(spine_dir)) < 0.1:
        spine_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    p_neck = p_pelvis + spine_dir * (MANNEQUIN_SPEC["spine_len"] * sy)

    # 计算躯干横向右轴 (right) 与前轴 (forward)
    char_forward = R_root @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    torso_right = _normalize(np.cross(spine_dir, char_forward))
    if np.linalg.norm(torso_right) < 1e-5:
        torso_right = R_root @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    # 注意: 右手系中 X_left = +X, X_right = -X，保持与预设一致 (+X 为角色左半身，-X 为角色右半身)
    side_L = R_root @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    side_L = _normalize(side_L - np.dot(side_L, spine_dir) * spine_dir)

    # 躯干转身（胸廓相对骨盆绕脊柱轴转动，角度限制 ±45°）：作为上身基础朝向，与双肩偏移叠加
    torso_twist_deg = max(-45.0, min(45.0, float(char_dict.get("torso_twist", 0.0) or 0.0)))
    if abs(torso_twist_deg) > 1e-6:
        th = math.radians(torso_twist_deg)
        k = spine_dir
        side_twist = (
            side_L * math.cos(th)
            + np.cross(k, side_L) * math.sin(th)
            + k * float(np.dot(k, side_L)) * (1.0 - math.cos(th))
        )
        side_twist = _normalize(side_twist)
    else:
        side_twist = side_L
    fwd_twist = _normalize(np.cross(side_twist, spine_dir))

    # 左右肩膀自身偏移（耸肩/沉肩 dy、含胸/展肩 dz、小幅横移 dx），双肩前后错位自动带动胸廓朝向
    use_natural_ik = (str(char_dict.get("body_style") or "mannequin") != "pawn") and (char_dict.get("natural_limb_motion") is not False)
    off_L = _clamp_shoulder_offset(targets.get("shoulder_L_offset"))
    off_R = _clamp_shoulder_offset(targets.get("shoulder_R_offset"))
    sh_w = MANNEQUIN_SPEC["shoulder_half_width"] * sx
    hip_w = MANNEQUIN_SPEC["hip_half_width"] * sx
    p_sternum = p_neck - spine_dir * (0.04 * sy)
    p_hand_L_w = to_world(targets.get("hand_L", [0.16, 0.54, 0.24]))
    p_hand_R_w = to_world(targets.get("hand_R", [-0.16, 0.54, 0.24]))
    if use_natural_ik:
        if "shoulder_L_offset" not in targets:
            sh_base_L = p_sternum + side_twist * sh_w
            rel_L = p_hand_L_w - sh_base_L
            rf_L = max(0.0, float(np.dot(rel_L, fwd_twist)) / max(1e-4, sz) - 0.22)
            ru_L = max(0.0, float(np.dot(rel_L, spine_dir)) / max(1e-4, sy) + 0.04)
            off_L = np.array([0.0, min(0.042, 0.105 * ru_L), min(0.052, 0.135 * rf_L)], dtype=np.float64)
        if "shoulder_R_offset" not in targets:
            sh_base_R = p_sternum - side_twist * sh_w
            rel_R = p_hand_R_w - sh_base_R
            rf_R = max(0.0, float(np.dot(rel_R, fwd_twist)) / max(1e-4, sz) - 0.22)
            ru_R = max(0.0, float(np.dot(rel_R, spine_dir)) / max(1e-4, sy) + 0.04)
            off_R = np.array([0.0, min(0.042, 0.105 * ru_R), min(0.052, 0.135 * rf_R)], dtype=np.float64)
    p_shoulder_L = (
        p_sternum
        + spine_dir * (off_L[1] * sy)
        + side_twist * (sh_w + off_L[0] * sx)
        + fwd_twist * (off_L[2] * sz)
    )
    p_shoulder_R = (
        p_sternum
        + spine_dir * (off_R[1] * sy)
        + side_twist * (-sh_w + off_R[0] * sx)
        + fwd_twist * (off_R[2] * sz)
    )

    sh_span = p_shoulder_L - p_shoulder_R
    sh_span_horiz = sh_span - np.dot(sh_span, spine_dir) * spine_dir
    if float(np.linalg.norm(sh_span_horiz)) > 1e-5:
        side_chest = _normalize(sh_span_horiz)
    else:
        side_chest = side_twist

    p_hip_L = p_pelvis + side_L * hip_w
    p_hip_R = p_pelvis - side_L * hip_w

    # 头部中心、面部正交基底与双眼中心（视线起点）
    p_head_center = p_neck + spine_dir * (0.14 * sy)
    raw_look = p_head_target - p_head_center
    if float(np.linalg.norm(raw_look)) < 1e-5:
        look_dir = _normalize(np.cross(side_chest, spine_dir))
    else:
        look_dir = _normalize(raw_look)
    head_left = np.cross(spine_dir, look_dir)
    if float(np.linalg.norm(head_left)) < 1e-4:
        head_left = side_chest
    else:
        head_left = _normalize(head_left)
    head_up = _normalize(np.cross(look_dir, head_left))
    p_eyes_center = p_head_center + look_dir * (0.110 * sz)

    # 双臂与双腿双骨 IK 解算（关节人偶启用 Soft IK + 肘/膝防反折；球头人偶保持原版零改动）
    l_ua = MANNEQUIN_SPEC["upper_arm_len"] * sy
    l_fa = MANNEQUIN_SPEC["forearm_len"] * sy
    p_hand_L = p_hand_L_w
    p_hand_R = p_hand_R_w
    p_elb_raw_L = to_world(targets.get("elbow_L_pole", [0.36, 0.66, -0.15]))
    p_elb_raw_R = to_world(targets.get("elbow_R_pole", [-0.36, 0.66, -0.15]))
    if use_natural_ik:
        p_elb_pole_L = _compute_ergonomic_elbow_pole(p_shoulder_L, p_hand_L, p_elb_raw_L, side_chest, spine_dir, True)
        p_elb_pole_R = _compute_ergonomic_elbow_pole(p_shoulder_R, p_hand_R, p_elb_raw_R, side_chest, spine_dir, False)
        arm_soft = 0.042
        leg_soft = 0.0025
    else:
        p_elb_pole_L = p_elb_raw_L
        p_elb_pole_R = p_elb_raw_R
        arm_soft = 0.0
        leg_soft = 0.0

    p_elbow_L, p_wrist_L = solve_two_bone_ik(p_shoulder_L, p_hand_L, p_elb_pole_L, l_ua, l_fa, soft_ratio=arm_soft)
    p_elbow_R, p_wrist_R = solve_two_bone_ik(p_shoulder_R, p_hand_R, p_elb_pole_R, l_ua, l_fa, soft_ratio=arm_soft)

    # 双腿双骨 IK 解算（带人体工学膝关节防反折约束与自然脚踝朝向）
    l_th = MANNEQUIN_SPEC["thigh_len"] * sy
    l_ca = MANNEQUIN_SPEC["calf_len"] * sy
    p_foot_target_L = to_world(targets.get("foot_L", [0.13, 0.04, 0.36]))
    p_foot_target_R = to_world(targets.get("foot_R", [-0.13, 0.04, 0.36]))
    p_knee_pole_L = _compute_ergonomic_knee_pole(
        p_hip_L,
        p_foot_target_L,
        to_world(targets.get("knee_L_pole", [0.15, 0.52, 0.55])),
        side_L,
        spine_dir,
        True,
    )
    p_knee_pole_R = _compute_ergonomic_knee_pole(
        p_hip_R,
        p_foot_target_R,
        to_world(targets.get("knee_R_pole", [-0.15, 0.52, 0.55])),
        side_L,
        spine_dir,
        False,
    )
    p_knee_L, p_ankle_L = solve_two_bone_ik(
        p_hip_L,
        p_foot_target_L,
        p_knee_pole_L,
        l_th,
        l_ca,
        soft_ratio=leg_soft,
    )
    p_knee_R, p_ankle_R = solve_two_bone_ik(
        p_hip_R,
        p_foot_target_R,
        p_knee_pole_R,
        l_th,
        l_ca,
        soft_ratio=leg_soft,
    )

    # 脚尖朝前延伸：随小腿方向与前向自然过渡，保持近似 90° 踝关节人体工学夹角
    base_foot_fwd = R_root @ np.array([0.0, -0.02, 1.0], dtype=np.float64)
    base_foot_fwd = _normalize(base_foot_fwd)
    calf_dir_L = _normalize(p_ankle_L - p_knee_L)
    calf_dir_R = _normalize(p_ankle_R - p_knee_R)
    nat_foot_L = np.cross(calf_dir_L, side_L)
    nat_foot_R = np.cross(calf_dir_R, side_L)
    foot_dir_L = _normalize((_normalize(nat_foot_L) if float(np.linalg.norm(nat_foot_L)) > 1e-4 else base_foot_fwd) * 0.65 + base_foot_fwd * 0.35)
    foot_dir_R = _normalize((_normalize(nat_foot_R) if float(np.linalg.norm(nat_foot_R)) > 1e-4 else base_foot_fwd) * 0.65 + base_foot_fwd * 0.35)
    foot_len = math.hypot(0.02 * sy, 0.13 * sz)
    p_toe_L = p_ankle_L + foot_dir_L * foot_len
    p_toe_R = p_ankle_R + foot_dir_R * foot_len

    return {
        "joints": {
            "pelvis": p_pelvis,
            "neck": p_neck,
            "sternum": p_sternum,
            "head_center": p_head_center,
            "eyes_center": p_eyes_center,
            "head_target": p_head_target,
            "shoulder_L": p_shoulder_L,
            "elbow_L": p_elbow_L,
            "wrist_L": p_wrist_L,
            "shoulder_R": p_shoulder_R,
            "elbow_R": p_elbow_R,
            "wrist_R": p_wrist_R,
            "hip_L": p_hip_L,
            "knee_L": p_knee_L,
            "ankle_L": p_ankle_L,
            "toe_L": p_toe_L,
            "hip_R": p_hip_R,
            "knee_R": p_knee_R,
            "ankle_R": p_ankle_R,
            "toe_R": p_toe_R,
        },
        "axes": {
            "spine_dir": spine_dir,
            "side_L": side_L,
            "side_chest": side_chest,
            "torso_twist": torso_twist_deg,
            "shoulder_L_offset": off_L,
            "shoulder_R_offset": off_R,
            "look_dir": look_dir,
            "head_left": head_left,
            "head_up": head_up,
            "scale": scale,
            "sx": sx,
            "sy": sy,
            "sz": sz,
        },
    }


# ============================================================================
# 3. 像 Blender bpy + 魔法人形师一样的 Python SDK (Stage / CharacterObj / PrimitiveObj)
# ============================================================================

class CharacterObj:
    """魔法人形师 3D 素体木偶角色 (支持点拉式 IK 坐标控制、预设姿态与位移/朝向关键帧)"""

    def __init__(
        self,
        char_id: str,
        name: str,
        pos: List[float],
        rot_y: float = 0.0,
        scale: float = 1.0,
        color: str = "#e2e8f0",
        body_style: str = "mannequin",
    ):
        self.id = char_id
        self.name = name
        self.type = "character"
        self.body_style = body_style if body_style in ("mannequin", "pawn") else "mannequin"
        self.pos = [float(pos[0]), float(pos[1]), float(pos[2])]
        self.rot = [0.0, float(rot_y), 0.0]
        self.scale = [float(scale), float(scale), float(scale)]
        self.color = color
        self.pose_name = "sitting"
        self.targets = {k: list(v) for k, v in PRESET_IK_POSES["sitting"]["targets"].items()}
        self.pins = dict(PRESET_IK_POSES["sitting"]["pins"])
        self.keyframes: List[Dict[str, Any]] = []
        self.extra: Dict[str, Any] = {}

    def add_keyframe(
        self,
        t: float,
        pos: Optional[List[float]] = None,
        rot_y: Optional[float] = None,
        look_at: Optional[List[float]] = None,
        targets: Optional[Dict[str, List[float]]] = None,
        torso_twist: Optional[float] = None,
        pose: Optional[str] = None,
        pose_name: Optional[str] = None,
        interp: str = "linear",
        interp_pos: Optional[str] = None,
        interp_rot: Optional[str] = None,
        interp_pose: Optional[str] = None,
        walk_steps: Optional[int] = None,
        lead_foot: Optional[str] = None,
        stop: bool = False,
        cut: bool = False,
        teleport: bool = False,
    ):
        """为角色添加时间轴位移/转身/转头/全身IK姿态关键帧（支持独立设置位移/朝向/姿态插值曲线与落脚不滑步步数）"""
        base_interp = str(interp or "linear")
        ry = float(rot_y if rot_y is not None else self.rot[1])
        kf: Dict[str, Any] = {
            "t": float(t),
            "pos": [float(x) for x in (pos if pos is not None else self.pos)],
            "rot": [0.0, ry, 0.0],
            "rot_y": ry,
            "interp": base_interp,
            "interp_pos": str(interp_pos or base_interp),
            "interp_rot": str(interp_rot or base_interp),
            "interp_pose": str(interp_pose or base_interp),
        }
        if stop:
            kf["stop"] = True
        if cut:
            kf["cut"] = True
        if teleport:
            kf["teleport"] = True
        chosen_pose = (pose_name or pose or "").strip().lower()
        if chosen_pose and chosen_pose in PRESET_IK_POSES:
            kf["pose_name"] = chosen_pose
            if targets is None:
                targets = PRESET_IK_POSES[chosen_pose]["targets"]
        if look_at is not None:
            kf["look_at"] = [float(x) for x in look_at]
        if targets is not None:
            kf["targets"] = {k: [float(x) for x in v] for k, v in targets.items()}
            if "look_at" not in kf and "head_target" in kf["targets"]:
                kf["look_at"] = list(kf["targets"]["head_target"])
        if torso_twist is not None:
            kf["torso_twist"] = float(torso_twist)
        if walk_steps is not None and int(walk_steps) > 0:
            kf["walk_steps"] = int(walk_steps)
        if lead_foot in ("L", "R"):
            kf["lead_foot"] = str(lead_foot)
        self.keyframes = [k for k in self.keyframes if abs(k["t"] - float(t)) > 1e-3]
        self.keyframes.append(kf)
        self.keyframes.sort(key=lambda x: x["t"])
        return self

    def set_pose(self, pose_name: str):
        """一键切换魔法人形师预设姿势:
        'sitting'(标准坐姿), 'sitting_relaxed'(二郎腿坐姿), 'sitting_think'(托腮思考),
        'standing'(自然站立), 'tpose', 'running'(奔跑), 'fighting'(格斗), 'kneeling'(单膝跪地), 'lying_supine'(仰面平躺)
        """
        key = pose_name.lower().strip()
        alias = {
            "sit": "sitting",
            "坐姿": "sitting",
            "二郎腿": "sitting_relaxed",
            "思考": "sitting_think",
            "托腮": "sitting_think",
            "stand": "standing",
            "站立": "standing",
            "run": "running",
            "奔跑": "running",
            "fight": "fighting",
            "格斗": "fighting",
            "kneel": "kneeling",
            "跪姿": "kneeling",
            "lie": "lying_supine",
            "仰卧": "lying_supine",
            "平躺": "lying_supine",
        }
        key = alias.get(key, key)
        if key in PRESET_IK_POSES:
            self.pose_name = key
            self.targets = {k: list(v) for k, v in PRESET_IK_POSES[key]["targets"].items()}
            self.pins = dict(PRESET_IK_POSES[key]["pins"])
        return self

    def ik(self, **kwargs):
        """
        魔法人形师点拉式 IK 接口！直接传入任意控制球的目标坐标 [x, y, z]：
        可用参数:
          - pelvis: 腰部重心球 [x, y, z] (例如高度降到 0.48 即坐下)
          - chest: 胸腔球 [x, y, z]
          - head_target / look_at: 视线看向的目标点 [x, y, z]
          - hand_L, hand_R: 左/右手腕控制球 [x, y, z]
          - foot_L, foot_R: 左/右脚踝控制球 [x, y, z]
          - knee_L_pole, knee_R_pole: 左右膝盖朝向极向球 [x, y, z]
          - elbow_L_pole, elbow_R_pole: 左右手肘朝向极向球 [x, y, z]
        """
        for k, v in kwargs.items():
            key = "head_target" if k == "look_at" else k
            if (key in self.targets or key in ("shoulder_L_offset", "shoulder_R_offset")) and isinstance(v, (list, tuple)) and len(v) == 3:
                self.targets[key] = [float(v[0]), float(v[1]), float(v[2])]
        return self

    def pin(self, foot_L: Optional[bool] = None, foot_R: Optional[bool] = None, hand_L: Optional[bool] = None, hand_R: Optional[bool] = None):
        """魔法人形师【图钉锁定 (Pin)】功能：锁住手脚后拉动腰部即可自动屈膝下蹲"""
        if foot_L is not None:
            self.pins["foot_L"] = bool(foot_L)
        if foot_R is not None:
            self.pins["foot_R"] = bool(foot_R)
        if hand_L is not None:
            self.pins["hand_L"] = bool(hand_L)
        if hand_R is not None:
            self.pins["hand_R"] = bool(hand_R)
        return self

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "name": self.name,
            "type": "character",
            "body_style": getattr(self, "body_style", "mannequin"),
            "pos": self.pos,
            "rot": self.rot,
            "scale": self.scale,
            "color": self.color,
            "pose_name": self.pose_name,
            "targets": self.targets,
            "pins": self.pins,
            "keyframes": self.keyframes,
        }
        if getattr(self, "torso_twist", 0.0):
            d["torso_twist"] = float(getattr(self, "torso_twist", 0.0))
        if getattr(self, "mounted_vehicle_id", None):
            d["mounted_vehicle_id"] = self.mounted_vehicle_id
        if getattr(self, "seat_offset", None):
            d["seat_offset"] = self.seat_offset
        if getattr(self, "seat_rot_offset", None) is not None:
            d["seat_rot_offset"] = float(self.seat_rot_offset)
        if getattr(self, "pawn_visor", False):
            d["pawn_visor"] = True
        if getattr(self, "foot_ground_clamp", None) is not None:
            d["foot_ground_clamp"] = bool(self.foot_ground_clamp)
        if getattr(self, "locked", None) is not None:
            d["locked"] = bool(self.locked)
        if getattr(self, "hidden", None) is not None:
            d["hidden"] = bool(self.hidden)
        if getattr(self, "natural_limb_motion", None) is not None:
            d["natural_limb_motion"] = bool(self.natural_limb_motion)
        if getattr(self, "extra", None):
            d.update(self.extra)
        if abs(float(getattr(self, "torso_twist", 0.0) or 0.0)) > 1e-6:
            d["torso_twist"] = float(self.torso_twist)
        return d


class PrimitiveObj:
    """基础彩色几何体: box(长方体), cone(圆锥/四棱锥), prism(三角柱), cylinder(圆柱), wall(墙体), group(组合体), uaz469(越野车)"""

    def __init__(
        self,
        obj_id: str,
        name: str,
        shape: str,
        pos: List[float],
        size: List[float],
        color: str = "#3b82f6",
        rot: Optional[List[float]] = None,
    ):
        self.id = obj_id
        self.name = name
        self.type = shape
        self.pos = [float(pos[0]), float(pos[1]), float(pos[2])]
        self.size = [float(size[0]), float(size[1]), float(size[2])]
        self.rot = [float(r) for r in (rot or [0.0, 0.0, 0.0])]
        self.color = color
        self.locked = False if shape == "uaz469" else None
        self.keyframes: List[Dict[str, Any]] = []
        self.extra: Dict[str, Any] = {}

    def add_keyframe(
        self,
        t: float,
        pos: List[float],
        rot: Optional[List[float]] = None,
        rot_y: Optional[float] = None,
        size: Optional[List[float]] = None,
        interp: str = "linear",
        interp_pos: Optional[str] = None,
        interp_rot: Optional[str] = None,
        interp_size: Optional[str] = None,
        stop: bool = False,
        coord_space: Optional[str] = None,
        reverse: bool = False,
        cut: bool = False,
        teleport: bool = False,
    ):
        base_interp = str(interp or "linear")
        if rot is not None and len(rot) >= 3:
            r_vec = [float(rot[0]), float(rot[1]), float(rot[2])]
        elif rot_y is not None:
            r_vec = [float(self.rot[0]), float(rot_y), float(self.rot[2])]
        else:
            r_vec = list(self.rot)
        s_vec = (
            [float(size[0]), float(size[1]), float(size[2])]
            if (size is not None and len(size) >= 3)
            else list(self.size)
        )
        kf: Dict[str, Any] = {
            "t": float(t),
            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
            "rot": r_vec,
            "rot_y": float(r_vec[1]),
            "size": s_vec,
            "interp": base_interp,
            "interp_pos": str(interp_pos or base_interp),
            "interp_rot": str(interp_rot or base_interp),
            "interp_size": str(interp_size or base_interp),
        }
        if stop:
            kf["stop"] = True
        if coord_space:
            kf["coord_space"] = str(coord_space)
        if reverse:
            kf["reverse"] = True
        if cut:
            kf["cut"] = True
        if teleport:
            kf["teleport"] = True
        self.keyframes = [k for k in self.keyframes if abs(float(k.get("t", 0.0)) - float(t)) > 1e-3]
        self.keyframes.append(kf)
        self.keyframes.sort(key=lambda x: float(x["t"]))
        if len(self.keyframes) == 1 and coord_space != "local":
            self.pos = list(kf["pos"])
            self.rot = list(kf["rot"])
            self.size = list(kf["size"])
        return self

    def bind_to_character(
        self,
        target_id: str,
        socket: str = "hand_R",
        local_pos: Optional[List[float]] = None,
        local_rot: Optional[List[float]] = None,
        local_scale: Optional[List[float]] = None,
        two_handed: bool = False,
        secondary_grip_local: Optional[List[float]] = None,
    ):
        """将当前道具/枪械绑定到目标人物指定挂点 (hand_R / hand_L / chest / back / root)"""
        is_gun = (self.extra.get("arch_role") == "gun") or str(self.id).startswith(("rifle_", "gun_"))
        grip_pos = list(self.extra.get("grip_offset_pos") or ([0.0, -0.018, 0.02] if is_gun else [0.0, 0.0, 0.0]))
        grip_rot = list(self.extra.get("grip_offset_rot") or [0.0, 0.0, 0.0])
        l_pos = [float(x) for x in (local_pos if local_pos is not None else [-grip_pos[0], -grip_pos[1], -grip_pos[2]])]
        l_rot = [float(x) for x in (local_rot if local_rot is not None else [0.0, 0.0, 0.0])]
        l_scale = [float(x) for x in (local_scale if local_scale is not None else [1.0, 1.0, 1.0])]
        sec_hand = "hand_L" if socket == "hand_R" else ("hand_R" if socket == "hand_L" else None)
        self.extra["parent_binding"] = {
            "enabled": True,
            "target_id": str(target_id),
            "socket": str(socket or "hand_R"),
            "local_pos": [round(x, 4) for x in l_pos],
            "local_rot": [round(x, 2) for x in l_rot],
            "local_scale": [round(x, 4) for x in l_scale],
            "grip_offset_pos": [round(float(x), 4) for x in grip_pos],
            "grip_offset_rot": [round(float(x), 2) for x in grip_rot],
            "two_handed": bool(two_handed),
            "secondary_hand": sec_hand,
            "secondary_grip_local": [round(float(x), 4) for x in (secondary_grip_local or [0.0, -0.018, 0.26])],
        }
        self.extra["bound_char_id"] = str(target_id)
        self.extra["follow_bound_char"] = True
        return self

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "pos": self.pos,
            "size": self.size,
            "rot": self.rot,
            "color": self.color,
        }
        if self.locked is not None:
            d["locked"] = self.locked
        if self.keyframes:
            d["keyframes"] = copy.deepcopy(self.keyframes)
        d.update(self.extra)
        return d


def focal_mm_to_fov(focal_mm: float, sensor_height_mm: float = 24.0) -> float:
    """将 35mm 全画幅等效焦距 (mm) 转换为垂直视场角 FOV (degrees)"""
    f = max(8.0, min(300.0, float(focal_mm)))
    return round(math.degrees(2.0 * math.atan(sensor_height_mm / (2.0 * f))), 2)


def fov_to_focal_mm(fov_deg: float, sensor_height_mm: float = 24.0) -> float:
    """将垂直视场角 FOV (degrees) 转换为 35mm 全画幅等效焦距 (mm)"""
    fov = max(4.0, min(130.0, float(fov_deg)))
    return round(sensor_height_mm / (2.0 * math.tan(math.radians(fov) * 0.5)), 1)


# ============================================================================
# 3.5 SU 极简彩色预演集中配置与语义转换器 (su_minimal)
# ============================================================================

FORCE_HIDE_EDGES: bool = True  # 全局强制无线条：禁用所有模型黑色结构边线

SU_MINIMAL_CONFIG: Dict[str, Any] = {
    "style_name": "su_minimal",
    "default_render_mode": "su_minimal",
    "max_parts_per_category": {
        "tree": 1,        # 树只能用单个低面数锥体表示，不加树干/树枝/积雪/贴图
        "gun": 3,         # 枪只用 3 个基础体（枪托方块 + 机匣方块 + 枪管圆柱）
        "character": 7,   # 球头 + 躯干 + 4简单四肢 + 1头前短线（无五官/帽子/护目镜/关节球）
        "vehicle": 6,     # 车身 + 座舱 + 4必要车轮
        "prop": 4,        # 普通道具最多 4 个基础体
    },
    "character_palette": [
        "#D97A36",  # 0: 暖橙 (鲍里斯 / 领队)
        "#5B8DBE",  # 1: 柔蓝 (后方队员1)
        "#63A675",  # 2: 柔绿 (后方队员2)
        "#9378B8",  # 3: 浅紫 (后方队员3)
        "#C8A24B",  # 4: 土黄 (后方队员4)
        "#C46D5E",  # 5: 柔砖红 (扩展人物)
        "#4E9296",  # 6: 灰青 (扩展人物)
        "#A87B56",  # 7: 暖褐 (扩展人物)
    ],
    "known_character_colors": {
        "char_boris": "#D97A36",
        "char_f1": "#5B8DBE",
        "char_left_rear": "#5B8DBE",
        "char_f2": "#63A675",
        "char_mid_rear": "#63A675",
        "char_f3": "#9378B8",
        "char_right_rear": "#9378B8",
        "char_f4": "#C8A24B",
        "char_deep_rear": "#C8A24B",
    },
    "tree_palette": [
        "#5E8C6A",  # 柔绿1
        "#537E5F",  # 柔绿2
        "#689674",  # 柔绿3
        "#4B7356",  # 柔绿4
    ],
    "gun_palette": [
        "#4A5568",  # 独立深蓝灰钢色（与持枪人物暖橙明显区分）
        "#3E4C5E",  # 深铁蓝
        "#5C6470",  # 铅钢灰
        "#6B5B52",  # 暖胡桃钢褐
        "#485E56",  # 暗军绿钢
        "#5D536B",  # 暗紫钢灰
    ],
    "vehicle_palette": [
        "#6B7F6E",  # 军灰绿
        "#5C768D",  # 灰石蓝
    ],
    "prop_palette": [
        "#9A8B7A",
        "#7E8B9B",
        "#A69580",
        "#879484",
    ],
    "ground_color": "#D8DCE2",  # 地面浅灰
    "road_color": "#B8C0CA",    # 道路略深
    "sky_color": "#E6EBF2",     # 柔和浅灰蓝背景
    "head_line_color": "#1E242B",
    "depth_export_enabled": False,  # 取消深度图导出入口与灰度视频生成
}


def _stable_id_hash(obj_id: str) -> int:
    h = hashlib.md5(str(obj_id or "obj").encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def get_su_minimal_identity_color(obj_id: str, category: str = "prop", index_hint: Optional[int] = None) -> str:
    """按稳定对象 ID 与类别分配持久身份色，播放、重新生成、保存重开与导出均不随机改变。"""
    oid = str(obj_id or "")
    cat = str(category or "prop").lower()
    if cat == "character" or oid.startswith("char_"):
        if oid in SU_MINIMAL_CONFIG["known_character_colors"]:
            return SU_MINIMAL_CONFIG["known_character_colors"][oid]
        pal = SU_MINIMAL_CONFIG["character_palette"]
        idx = index_hint if index_hint is not None else (_stable_id_hash(oid) % len(pal))
        return pal[idx % len(pal)]
    if cat == "tree" or oid.startswith("tree_") or oid.startswith("env_tree_"):
        pal = SU_MINIMAL_CONFIG["tree_palette"]
        idx = index_hint if index_hint is not None else (_stable_id_hash(oid) % len(pal))
        return pal[idx % len(pal)]
    if cat == "gun" or oid.startswith("rifle_") or oid.startswith("gun_"):
        pal = SU_MINIMAL_CONFIG["gun_palette"]
        idx = index_hint if index_hint is not None else (_stable_id_hash(oid) % len(pal))
        return pal[idx % len(pal)]
    if cat in ("vehicle", "uaz469") or oid.startswith("uaz_"):
        pal = SU_MINIMAL_CONFIG["vehicle_palette"]
        idx = index_hint if index_hint is not None else (_stable_id_hash(oid) % len(pal))
        return pal[idx % len(pal)]
    if cat == "road" or oid.startswith("env_road_"):
        return SU_MINIMAL_CONFIG["road_color"]
    if cat in ("ground", "floor") or oid.startswith("env_ground_"):
        return SU_MINIMAL_CONFIG["ground_color"]
    pal = SU_MINIMAL_CONFIG["prop_palette"]
    idx = index_hint if index_hint is not None else (_stable_id_hash(oid) % len(pal))
    return pal[idx % len(pal)]


def make_su_minimal_gun_parts(color: str = "#4A5568") -> List[Dict[str, Any]]:
    """生成 SU 极简枪械的 3 个基础体：枪托方块 + 机匣方块 + 枪管圆柱（不建准星/扳机/内膛/护木刻线）"""
    return [
        {
            "id": "gun_stock",
            "name": "枪托方块",
            "shape": "box",
            "pos": [0.0, -0.018, -0.08],
            "rot": [0.0, 0.0, 0.0],
            "size": [0.050, 0.088, 0.24],
            "color": color,
        },
        {
            "id": "gun_receiver",
            "name": "枪身机匣方块",
            "shape": "box",
            "pos": [0.0, 0.0, 0.16],
            "rot": [0.0, 0.0, 0.0],
            "size": [0.056, 0.082, 0.26],
            "color": color,
        },
        {
            "id": "gun_barrel",
            "name": "枪管圆柱",
            "shape": "cylinder",
            "pos": [0.0, 0.012, 0.48],
            "rot": [90.0, 0.0, 0.0],
            "size": [0.052, 0.44, 0.052],
            "color": color,
        },
    ]


def convert_scene_to_su_minimal(scene_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    将任意已有或新生成的三维场景按语义转换为 SU 极简代理体（su_minimal）：
    1. 树 -> 单个低面数圆锥体（锥尖朝上、底面落地 y=0），移除树干/多层树冠/积雪，使用柔绿色系；
    2. 人物 -> 极简球头人偶 + 头前朝向短线，移除帽子/独眼罩/护目镜/五官/关节球，分配稳定身份色；
    3. 枪 -> 3 个基础体（枪托方块 + 枪身方块 + 枪管圆柱），移除准星/箍环/内膛孔，使用独立深钢灰身份色；
    4. 路面/地形 -> 移除车辙、倒木、碎雪堆等散落装饰，保留平整地面与道路；
    5. 保留全部世界变换、稳定对象 ID、关键帧轨迹与手部绑定关系。
    """
    sc = copy.deepcopy(scene_dict)
    sc["render_mode"] = "su_minimal"
    sc["visual_style"] = "su_minimal"
    env = sc.get("environment") or {}
    env["sky"] = SU_MINIMAL_CONFIG["sky_color"]
    env["ground"] = SU_MINIMAL_CONFIG["ground_color"]
    env["fog_color"] = SU_MINIMAL_CONFIG["sky_color"]
    env["show_grid"] = False
    sc["environment"] = env

    new_objects: List[Dict[str, Any]] = []
    char_idx = 0
    tree_idx = 0
    for obj in sc.get("objects", []):
        oid = str(obj.get("id", ""))
        otype = str(obj.get("type", "box"))
        role = str(obj.get("arch_role", ""))

        # 移除帽子、独眼罩、车辙、碎雪堆、倒木等复杂装饰物
        if (
            oid.startswith("hat_")
            or "eyepatch" in oid
            or oid.startswith("env_rut_")
            or oid.startswith("env_snowbank_")
            or oid.startswith("env_fallen_log_")
        ):
            continue

        if otype == "character":
            cur_c_col = str(obj.get("color") or "").strip()
            if not cur_c_col or cur_c_col.upper() in ("#E2E8F0", "#FFFFFF", "#EAEEF4", "#D4D6D9"):
                obj["color"] = get_su_minimal_identity_color(oid, "character", index_hint=char_idx)
            char_idx += 1
            obj.pop("pawn_visor", None)
            obj["su_minimal"] = True
            obj["head_dir_line"] = True
            new_objects.append(obj)
            continue

        # 树：强制转换为单个圆锥体（锥尖朝上，底面落地 y=0）
        if oid.startswith("tree_") or oid.startswith("env_tree_") or role == "tree":
            pos = list(obj.get("pos", [0.0, 0.0, 0.0]))
            size = list(obj.get("size", [1.4, 5.2, 1.4]))
            cone_w = max(0.28, min(2.4, float(size[0])))
            cone_h = max(2.8, min(7.0, float(size[1])))
            t_col = get_su_minimal_identity_color(oid, "tree", index_hint=tree_idx)
            tree_idx += 1
            cone_obj: Dict[str, Any] = {
                "id": oid,
                "name": obj.get("name", oid),
                "type": "cone",
                "arch_role": "tree",
                "pos": [round(float(pos[0]), 3), round(cone_h * 0.5, 3), round(float(pos[2]), 3)],
                "rot": [0.0, 0.0, 0.0],
                "size": [round(cone_w, 3), round(cone_h, 3), round(cone_w, 3)],
                "color": t_col,
            }
            if obj.get("keyframes"):
                kfs = []
                for kf in obj["keyframes"]:
                    kf_c = dict(kf)
                    kp = list(kf_c.get("pos", pos))
                    kf_c["pos"] = [round(float(kp[0]), 3), round(cone_h * 0.5, 3), round(float(kp[2]), 3)]
                    kf_c["size"] = [round(cone_w, 3), round(cone_h, 3), round(cone_w, 3)]
                    kfs.append(kf_c)
                cone_obj["keyframes"] = kfs
            new_objects.append(cone_obj)
            continue

        # 枪械：强制收敛为 3 个基础体（枪托 + 枪身 + 枪管），不建准星/扳机/内膛
        if oid.startswith("rifle_") or oid.startswith("gun_") or role == "gun":
            cur_g_col = str(obj.get("color") or "").strip()
            if not cur_g_col or cur_g_col.upper() in ("#D4D6D9", "#E2E3E5", "#FFFFFF", "#EAEEF4", "#3B82F6"):
                g_col = get_su_minimal_identity_color(oid, "gun")
            else:
                g_col = cur_g_col
            obj["type"] = "group"
            obj["arch_role"] = "gun"
            obj["color"] = g_col
            obj["parts"] = make_su_minimal_gun_parts(g_col)
            obj.setdefault("grip_offset_pos", [0.0, -0.018, 0.02])
            obj.setdefault("grip_offset_rot", [0.0, 0.0, 0.0])
            new_objects.append(obj)
            continue

        # 道路 / 地面
        if oid.startswith("env_road_") or role == "road":
            obj["color"] = SU_MINIMAL_CONFIG["road_color"]
            new_objects.append(obj)
            continue

        if otype == "uaz469":
            obj["color"] = get_su_minimal_identity_color(oid, "vehicle")
            new_objects.append(obj)
            continue

        # 其他组合体：限制部件数 <= 4 并统一身份色，避免逐部件花哨随机色
        p_col = obj.get("color")
        if not p_col or p_col.upper() in ("#D4D6D9", "#E2E3E5", "#FFFFFF", "#EAEEF4"):
            p_col = get_su_minimal_identity_color(oid, "prop")
        obj["color"] = p_col
        if isinstance(obj.get("parts"), list):
            max_p = SU_MINIMAL_CONFIG["max_parts_per_category"]["prop"]
            trimmed_parts = []
            for pt in obj["parts"][:max_p]:
                pt_c = dict(pt)
                pt_c["color"] = p_col
                trimmed_parts.append(pt_c)
            obj["parts"] = trimmed_parts
        new_objects.append(obj)

    sc["objects"] = new_objects
    return sc


class Stage:
    """导演台场景总控制器 (默认采用 SU 极简彩色预演规则 su_minimal)"""

    def __init__(self):
        self.objects: List[Any] = []
        self.camera: Dict[str, Any] = {
            "pos": [2.2, 1.50, 2.75],
            "target": [0.0, 0.72, 0.10],
            "focal_mm": 35.0,
            "fov": focal_mm_to_fov(35.0),
        }
        self.camera_keyframes: List[Dict[str, Any]] = []
        self.duration: float = 4.0
        self.fps: int = 24
        self.render_mode: str = SU_MINIMAL_CONFIG["default_render_mode"]  # 默认 'su_minimal'
        self.visual_style: Optional[str] = SU_MINIMAL_CONFIG["style_name"]  # 默认 'su_minimal'
        self._id_counter = 1
        self._last_exported_video: Optional[str] = None
        # 场景环境（天空 / 地面 / 雾）；默认启用 SU 极简浅灰地面与柔和天空
        self.environment: Optional[Dict[str, Any]] = {
            "preset": "su_minimal",
            "sky": SU_MINIMAL_CONFIG["sky_color"],
            "ground": SU_MINIMAL_CONFIG["ground_color"],
            "fog_color": SU_MINIMAL_CONFIG["sky_color"],
            "fog": False,
            "show_ground": True,
            "show_grid": False,
        }
        self.ref_video: Optional[Dict[str, Any]] = None
        self.repro_state: Optional[Dict[str, Any]] = None
        self.point_cloud_helper: Optional[Dict[str, Any]] = None
        self.constraints: List[Dict[str, Any]] = []
        self.camera_speed_curve: Optional[Dict[str, Any]] = None
        self.camera_rig: Optional[Dict[str, Any]] = None
        self.meters_per_unit: float = 1.0
        self.allow_snow_jeep_4mps: Optional[bool] = None
        self.hide_edges: bool = True  # 强制无线条

    def _next_id(self, prefix: str) -> str:
        cid = f"{prefix}_{self._id_counter}"
        self._id_counter += 1
        return cid

    def clear(self):
        self.objects.clear()
        self.camera_keyframes.clear()
        self.constraints.clear()
        self.camera_speed_curve = None
        self.camera_rig = None
        self._id_counter = 1
        return self

    def add_box(
        self,
        name: str = "长方体",
        pos: List[float] = [0.0, 0.24, 0.0],
        size: List[float] = [0.48, 0.48, 0.48],
        color: str = "#D4D6D9",
        rot: Optional[List[float]] = None,
        obj_id: Optional[str] = None,
        **extra,
    ) -> PrimitiveObj:
        obj = PrimitiveObj(obj_id or self._next_id("box"), name, "box", pos, size, color, rot)
        obj.extra.update(extra)
        self.objects.append(obj)
        return obj

    def add_cone(
        self,
        name: str = "圆锥体",
        pos: List[float] = [-1.1, 0.45, 0.2],
        size: List[float] = [0.52, 0.90, 0.52],
        color: str = "#D4D6D9",
        rot: Optional[List[float]] = None,
        obj_id: Optional[str] = None,
        **extra,
    ) -> PrimitiveObj:
        obj = PrimitiveObj(obj_id or self._next_id("cone"), name, "cone", pos, size, color, rot)
        obj.extra.update(extra)
        self.objects.append(obj)
        return obj

    def add_prism(
        self,
        name: str = "三角柱",
        pos: List[float] = [1.1, 0.35, 0.2],
        size: List[float] = [0.68, 0.70, 0.80],
        color: str = "#D4D6D9",
        rot: Optional[List[float]] = None,
        obj_id: Optional[str] = None,
        **extra,
    ) -> PrimitiveObj:
        obj = PrimitiveObj(obj_id or self._next_id("prism"), name, "prism", pos, size, color, rot)
        obj.extra.update(extra)
        self.objects.append(obj)
        return obj

    def add_cylinder(
        self,
        name: str = "圆柱体",
        pos: List[float] = [0.0, 0.4, -1.0],
        size: List[float] = [0.5, 0.8, 0.5],
        color: str = "#D4D6D9",
        rot: Optional[List[float]] = None,
        obj_id: Optional[str] = None,
        **extra,
    ) -> PrimitiveObj:
        obj = PrimitiveObj(obj_id or self._next_id("cyl"), name, "cylinder", pos, size, color, rot)
        obj.extra.update(extra)
        self.objects.append(obj)
        return obj

    def add_sphere(
        self,
        name: str = "球体",
        pos: List[float] = [0.0, 0.3, 0.0],
        size: List[float] = [0.6, 0.6, 0.6],
        color: str = "#D4D6D9",
        rot: Optional[List[float]] = None,
        obj_id: Optional[str] = None,
        **extra,
    ) -> PrimitiveObj:
        obj = PrimitiveObj(obj_id or self._next_id("sphere"), name, "sphere", pos, size, color, rot)
        obj.extra.update(extra)
        self.objects.append(obj)
        return obj

    def add_plane(
        self,
        name: str = "平面",
        pos: List[float] = [0.0, 0.01, 0.0],
        size: List[float] = [2.0, 0.02, 2.0],
        color: str = "#D4D6D9",
        rot: Optional[List[float]] = None,
        obj_id: Optional[str] = None,
        **extra,
    ) -> PrimitiveObj:
        obj = PrimitiveObj(obj_id or self._next_id("plane"), name, "plane", pos, size, color, rot)
        obj.extra.update(extra)
        self.objects.append(obj)
        return obj

    def add_wall(
        self,
        name: str = "墙体",
        pos: List[float] = [0.0, 1.2, -2.0],
        size: List[float] = [3.6, 2.4, 0.16],
        color: str = "#DEDFE1",
        rot: Optional[List[float]] = None,
        openings: Optional[List[Dict[str, Any]]] = None,
        obj_id: Optional[str] = None,
        **extra,
    ) -> PrimitiveObj:
        obj = PrimitiveObj(obj_id or self._next_id("wall"), name, "wall", pos, size, color, rot)
        if openings is not None:
            obj.extra["openings"] = copy.deepcopy(openings)
        obj.extra.update(extra)
        self.objects.append(obj)
        return obj

    def add_group(
        self,
        name: str = "组合结构",
        pos: List[float] = [0.0, 0.0, 0.0],
        size: List[float] = [1.0, 1.0, 1.0],
        color: str = "#D4D6D9",
        rot: Optional[List[float]] = None,
        parts: Optional[List[Dict[str, Any]]] = None,
        obj_id: Optional[str] = None,
        **extra,
    ) -> PrimitiveObj:
        obj = PrimitiveObj(obj_id or self._next_id("grp"), name, "group", pos, size, color, rot)
        obj.extra["parts"] = copy.deepcopy(parts or [])
        obj.extra.update(extra)
        self.objects.append(obj)
        return obj

    def add_scene_group(
        self,
        name: str = "布景编组",
        pos: List[float] = [0.0, 0.0, 0.0],
        rot: Optional[List[float]] = None,
        size: Optional[List[float]] = None,
        color: str = "#60A5FA",
        obj_id: Optional[str] = None,
        **extra,
    ) -> PrimitiveObj:
        obj = PrimitiveObj(
            obj_id or self._next_id("sgrp"),
            name,
            "scene_group",
            pos,
            size or [1.0, 1.0, 1.0],
            color,
            rot or [0.0, 0.0, 0.0],
        )
        obj.extra.update(extra)
        self.objects.append(obj)
        return obj

    def add_uaz469(
        self,
        name: str = "UAZ-469 越野车",
        pos: List[float] = [-1.85, 1.01, -0.95],
        size: List[float] = [1.78, 2.02, 4.02],
        color: str = "#4F5D3B",
        rot: Optional[List[float]] = None,
        obj_id: Optional[str] = None,
        **extra,
    ) -> PrimitiveObj:
        """添加苏联经典 UAZ-469 军用吉普越野车 (标准尺寸: 宽1.78m x 高2.02m x 长4.02m)"""
        obj = PrimitiveObj(obj_id or self._next_id("uaz"), name, "uaz469", pos, size, color, rot or [0.0, 32.0, 0.0])
        obj.extra.update(extra)
        self.objects.append(obj)
        return obj

    def add_tree_cone(
        self,
        name: str = "单锥体树",
        x: float = 0.0,
        z: float = -2.0,
        width: float = 1.4,
        height: float = 5.2,
        color: Optional[str] = None,
        obj_id: Optional[str] = None,
    ) -> PrimitiveObj:
        """SU 极简规范：每棵树严格使用单个低面数圆锥体（锥尖朝上、底面落地 y=0），使用柔绿色系"""
        tid = obj_id or self._next_id("tree")
        t_idx = sum(1 for o in self.objects if getattr(o, "id", "").startswith("tree_"))
        col = color or get_su_minimal_identity_color(tid, "tree", index_hint=t_idx)
        obj = PrimitiveObj(tid, name, "cone", [float(x), round(float(height) * 0.5, 3), float(z)], [float(width), float(height), float(width)], col, [0.0, 0.0, 0.0])
        obj.extra["arch_role"] = "tree"
        self.objects.append(obj)
        return obj

    def add_gun_minimal(
        self,
        name: str = "极简步枪",
        pos: List[float] = [0.0, 1.25, 0.0],
        rot: Optional[List[float]] = None,
        color: Optional[str] = None,
        obj_id: Optional[str] = None,
        bound_char_id: Optional[str] = None,
        socket: str = "hand_R",
        bind_mode: str = "snap_grip",
        local_pos: Optional[List[float]] = None,
        local_rot: Optional[List[float]] = None,
        local_scale: Optional[List[float]] = None,
        two_handed: bool = False,
        **extra,
    ) -> PrimitiveObj:
        """SU 极简规范：枪仅由 3 个基础体（枪托方块 + 机匣方块 + 枪管圆柱）构成，独立深钢灰身份色"""
        gid = obj_id or self._next_id("rifle")
        col = color or get_su_minimal_identity_color(gid, "gun")
        obj = PrimitiveObj(gid, name, "group", pos, [0.08, 0.12, 0.72], col, rot or [0.0, 0.0, 0.0])
        obj.extra["arch_role"] = "gun"
        obj.extra["parts"] = make_su_minimal_gun_parts(col)
        obj.extra["grip_offset_pos"] = [0.0, -0.018, 0.02]
        obj.extra["grip_offset_rot"] = [0.0, 0.0, 0.0]
        if bound_char_id:
            obj.bind_to_character(
                target_id=bound_char_id,
                socket=socket,
                local_pos=list(local_pos) if local_pos is not None else [0.0, 0.018, -0.02],
                local_rot=list(local_rot) if local_rot is not None else [0.0, 0.0, 0.0],
                local_scale=list(local_scale) if local_scale is not None else [1.0, 1.0, 1.0],
                two_handed=two_handed,
            )
        if extra:
            obj.extra.update(extra)
        self.objects.append(obj)
        return obj

    def add_character(
        self,
        name: str = "魔法人形偶_01",
        pos: List[float] = [0.0, 0.0, 0.0],
        rot_y: float = 0.0,
        pose: str = "sitting",
        scale: float = 1.0,
        color: str = "#e2e8f0",
        body_style: str = "mannequin",
        char_id: Optional[str] = None,
        **extra,
    ) -> CharacterObj:
        # Public scale=1 means a neutral standing height of 1.70 metres.
        cid = char_id or self._next_id("char")
        c_idx = sum(1 for o in self.objects if getattr(o, "type", "") == "character")
        if not color or color.lower() in ("#e2e8f0", "#ffffff", "#eaeef4", "#d4d6d9"):
            color = get_su_minimal_identity_color(cid, "character", index_hint=c_idx)
        reference_height = CHARACTER_NEUTRAL_DIMENSIONS.get(body_style, CHARACTER_NEUTRAL_DIMENSIONS["mannequin"])[1]
        char = CharacterObj(cid, name, pos, rot_y, float(scale) * 1.70 / reference_height, color=color, body_style=body_style)
        char.set_pose(pose)
        char.extra["head_dir_line"] = True
        if extra:
            char.extra.update(extra)
        self.objects.append(char)
        return char

    def add_pawn(
        self,
        name: str = "球头人偶_01",
        pos: List[float] = [0.0, 0.0, 0.0],
        rot_y: float = 0.0,
        pose: str = "standing",
        scale: float = 1.0,
        color: str = "#D98178",
        char_id: Optional[str] = None,
        **extra,
    ) -> CharacterObj:
        """添加球头人偶角色 (body_style='pawn')"""
        return self.add_character(name=name, pos=pos, rot_y=rot_y, pose=pose, scale=scale, color=color, body_style="pawn", char_id=char_id, **extra)

    def set_camera(
        self,
        pos: List[float],
        target: List[float],
        fov: Optional[float] = None,
        focal_mm: Optional[float] = None,
        aspect_ratio: Optional[str] = None,
        export_width: Optional[int] = None,
        export_height: Optional[int] = None,
        roll_deg: Optional[float] = None,
    ):
        if focal_mm is not None:
            f_mm = float(focal_mm)
            fov_val = focal_mm_to_fov(f_mm)
        elif fov is not None:
            fov_val = float(fov)
            f_mm = fov_to_focal_mm(fov_val)
        else:
            f_mm = 35.0
            fov_val = focal_mm_to_fov(35.0)

        prev_aspect = self.camera.get("aspect_ratio", "2.39:1") if isinstance(self.camera, dict) else "2.39:1"
        prev_ew = self.camera.get("export_width") if isinstance(self.camera, dict) else None
        prev_eh = self.camera.get("export_height") if isinstance(self.camera, dict) else None
        prev_roll = float(self.camera.get("roll_deg", 0.0) or 0.0) if isinstance(self.camera, dict) else 0.0
        self.camera = {
            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
            "target": [float(target[0]), float(target[1]), float(target[2])],
            "focal_mm": round(f_mm, 1),
            "fov": round(fov_val, 2),
            "aspect_ratio": aspect_ratio or prev_aspect,
        }
        r_val = float(roll_deg) if roll_deg is not None else prev_roll
        if abs(r_val) > 1e-4:
            self.camera["roll_deg"] = round(r_val, 3)
        ew = export_width if export_width is not None else prev_ew
        eh = export_height if export_height is not None else prev_eh
        if ew and eh:
            self.camera["export_width"] = int(ew)
            self.camera["export_height"] = int(eh)
        return self

    def set_camera_rig(
        self,
        rail_mode: Optional[str] = None,
        handheld_preset: Optional[str] = None,
        intensity: Optional[float] = None,
        seed: Optional[int] = None,
        lock_roll: Optional[bool] = None,
        **kwargs: Any,
    ):
        """设置摄影机轨道约束与手持微扰参数（camera_rig v3）。"""
        cur = copy.deepcopy(self.camera_rig) if isinstance(self.camera_rig, dict) else {}
        if rail_mode is not None:
            cur["rail_mode"] = str(rail_mode)
        if lock_roll is not None:
            cur["lock_roll"] = bool(lock_roll)
        hh = dict(cur.get("handheld") or {})
        if handheld_preset is not None:
            hh["preset"] = str(handheld_preset)
            hh["enabled"] = str(handheld_preset) not in ("off", "dolly", "none", "")
        if intensity is not None:
            hh["intensity"] = float(intensity)
        if seed is not None:
            hh["seed"] = int(seed)
        for k, v in kwargs.items():
            if k in ("pos_amp_m", "rot_amp_deg", "roll_amp_deg", "freq_hz", "step_freq_hz", "speed_gain", "settle_ends", "enabled"):
                hh[k] = v
            else:
                cur[k] = v
        cur["handheld"] = hh
        self.camera_rig = normalize_camera_rig_config(cur)
        return self

    def add_camera_keyframe(
        self,
        t: float,
        pos: List[float],
        target: List[float],
        fov: Optional[float] = None,
        focal_mm: Optional[float] = None,
        interp: str = "linear",
        interp_pos: Optional[str] = None,
        interp_target: Optional[str] = None,
        interp_fov: Optional[str] = None,
        stop: Optional[bool] = None,
        has_pos: Optional[bool] = None,
        has_fov: Optional[bool] = None,
        roll_deg: Optional[float] = None,
    ):
        if fov is not None:
            fov_val = float(fov)
            f_mm = float(focal_mm) if focal_mm is not None else fov_to_focal_mm(fov_val)
        elif focal_mm is not None:
            f_mm = float(focal_mm)
            fov_val = focal_mm_to_fov(f_mm)
        else:
            f_mm = float(self.camera.get("focal_mm", 35.0))
            fov_val = focal_mm_to_fov(f_mm)

        base_interp = str(interp or "linear")
        kf: Dict[str, Any] = {
            "t": float(t),
            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
            "target": [float(target[0]), float(target[1]), float(target[2])],
            "focal_mm": round(f_mm, 1),
            "fov": round(fov_val, 2),
            "interp": base_interp,
            "interp_pos": str(interp_pos or base_interp),
            "interp_target": str(interp_target or base_interp),
            "interp_fov": str(interp_fov or base_interp),
        }
        if stop is not None:
            kf["stop"] = bool(stop)
        if has_pos is not None:
            kf["has_pos"] = bool(has_pos)
        if has_fov is not None:
            kf["has_fov"] = bool(has_fov)
        if roll_deg is not None and abs(float(roll_deg)) > 1e-4:
            kf["roll_deg"] = round(float(roll_deg), 3)
        self.camera_keyframes = [k for k in self.camera_keyframes if abs(k["t"] - t) > 1e-3]
        self.camera_keyframes.append(kf)
        self.camera_keyframes.sort(key=lambda x: x["t"])
        if len(self.camera_keyframes) == 1:
            self.set_camera(pos, target, fov=fov_val, focal_mm=f_mm, roll_deg=roll_deg)
        return self

    def export_video(
        self,
        filename: str = "magic_previz.mp4",
        duration: Optional[float] = None,
        fps: Optional[int] = None,
        mode: Optional[str] = None,
        width: int = 960,
        height: int = 540,
        export_confirmation: Optional[Dict[str, Any]] = None,
    ) -> str:
        if duration is not None:
            self.duration = float(duration)
        if fps is not None:
            self.fps = int(fps)
        if mode is not None:
            self.render_mode = mode

        scene_dict = self.to_dict()
        gate = check_export_confirmation_gate(scene_dict, export_confirmation)
        if not gate["confirmed"]:
            self._pending_export_confirmation = gate["summary"]
            self._last_exported_video = None
            return ""

        scene_dict["export_plan"] = gate["plan"]
        self.export_plan = copy.deepcopy(gate["plan"])
        export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
        os.makedirs(export_dir, exist_ok=True)
        out_path = os.path.join(export_dir, os.path.basename(filename))
        render_scene_to_mp4(scene_dict, out_path, width=width, height=height, export_confirmation=gate["plan"])
        self._last_exported_video = os.path.basename(filename)
        return out_path

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "camera": self.camera,
            "camera_keyframes": self.camera_keyframes,
            "duration": self.duration,
            "fps": self.fps,
            "render_mode": self.render_mode,
            "objects": [obj.to_dict() for obj in self.objects],
        }
        if self.visual_style:
            out["visual_style"] = self.visual_style
        if self.environment:
            out["environment"] = dict(self.environment)
        if self.ref_video:
            out["ref_video"] = copy.deepcopy(self.ref_video)
        if self.repro_state:
            out["repro_state"] = copy.deepcopy(self.repro_state)
        if self.point_cloud_helper:
            out["point_cloud_helper"] = copy.deepcopy(self.point_cloud_helper)
        if getattr(self, "constraints", None):
            out["constraints"] = copy.deepcopy(self.constraints)
        if getattr(self, "export_plan", None):
            out["export_plan"] = copy.deepcopy(self.export_plan)
        if getattr(self, "camera_speed_curve", None):
            out["camera_speed_curve"] = copy.deepcopy(self.camera_speed_curve)
        if getattr(self, "camera_rig", None):
            out["camera_rig"] = copy.deepcopy(self.camera_rig)
        if getattr(self, "meters_per_unit", None) is not None and abs(float(self.meters_per_unit) - 1.0) > 1e-6:
            out["meters_per_unit"] = float(self.meters_per_unit)
        if getattr(self, "allow_snow_jeep_4mps", None) is not None:
            out["allow_snow_jeep_4mps"] = bool(self.allow_snow_jeep_4mps)
        if FORCE_HIDE_EDGES:
            out["hide_edges"] = True
        elif getattr(self, "hide_edges", None) is not None:
            out["hide_edges"] = bool(self.hide_edges)
        if getattr(self, "contact_shadows", None) is not None:
            out["contact_shadows"] = bool(self.contact_shadows)
        if getattr(self, "global_body_style", None):
            out["global_body_style"] = str(self.global_body_style)
        return out


MAX_TRACK_KEYFRAMES: int = 10
MAX_PROJECT_DURATION: float = 30.0


def simplify_keyframes_to_limit(kfs: List[Dict[str, Any]], max_kfs: int = MAX_TRACK_KEYFRAMES) -> List[Dict[str, Any]]:
    """将单条轨道关键帧按关键拐点（Ramer-Douglas-Peucker 最大偏离度）精简至最多 max_kfs 帧，保留首尾、显隐跳变与停顿帧。"""
    if not isinstance(kfs, list) or len(kfs) <= max_kfs:
        return kfs
    sorted_kfs = sorted(kfs, key=lambda item: float(item.get("t", 0.0)))
    n = len(sorted_kfs)
    keep = {0, n - 1}
    for idx in range(1, n - 1):
        cur_k = sorted_kfs[idx]
        prev_k = sorted_kfs[idx - 1]
        if cur_k.get("stop"):
            keep.add(idx)
        if isinstance(cur_k.get("visible"), bool) and cur_k.get("visible") != prev_k.get("visible"):
            keep.add(idx)
    if len(keep) > max_kfs:
        mandatory = sorted(keep)
        trimmed = {0, n - 1}
        for j in range(1, max_kfs - 1):
            pick = mandatory[int(round(j * (len(mandatory) - 1) / float(max_kfs - 1)))]
            trimmed.add(pick)
        return [sorted_kfs[i] for i in sorted(trimmed)]

    def _kf_vec(k: Dict[str, Any]) -> List[float]:
        p = k.get("pos") or [0.0, 0.0, 0.0]
        tr = k.get("target") or [0.0, 0.0, 0.0]
        t_map = k.get("targets") or {}
        hr = t_map.get("hand_R") or [0.0, 0.0, 0.0]
        rot_v = k.get("rot_y")
        if rot_v is None and isinstance(k.get("rot"), list) and len(k["rot"]) >= 2:
            rot_v = k["rot"][1]
        return [
            float(p[0] if len(p) > 0 else 0.0),
            float(p[1] if len(p) > 1 else 0.0),
            float(p[2] if len(p) > 2 else 0.0),
            float(tr[0] if len(tr) > 0 else 0.0),
            float(tr[1] if len(tr) > 1 else 0.0),
            float(tr[2] if len(tr) > 2 else 0.0),
            float(rot_v or 0.0) * 0.03,
            float(k.get("fov") or 0.0) * 0.08,
            float(hr[0] if len(hr) > 0 else 0.0) * 2.0,
            float(hr[1] if len(hr) > 1 else 0.0) * 2.0,
            float(hr[2] if len(hr) > 2 else 0.0) * 2.0,
        ]

    vecs = [_kf_vec(k) for k in sorted_kfs]
    while len(keep) < max_kfs:
        ordered = sorted(keep)
        best_idx = -1
        best_d2 = -1.0
        for s in range(len(ordered) - 1):
            i0, i1 = ordered[s], ordered[s + 1]
            if i1 - i0 <= 1:
                continue
            t0 = float(sorted_kfs[i0].get("t", 0.0))
            t1 = float(sorted_kfs[i1].get("t", 0.0))
            v0, v1 = vecs[i0], vecs[i1]
            for m in range(i0 + 1, i1):
                tm = float(sorted_kfs[m].get("t", 0.0))
                u = (tm - t0) / (t1 - t0) if (t1 - t0) > 1e-6 else 0.5
                vm = vecs[m]
                d2 = sum((vm[c] - (v0[c] + (v1[c] - v0[c]) * u)) ** 2 for c in range(len(v0)))
                if d2 > best_d2:
                    best_d2 = d2
                    best_idx = m
        if best_idx < 0:
            break
        keep.add(best_idx)
    return [sorted_kfs[i] for i in sorted(keep)]


def enforce_scene_track_and_duration_limits(
    scene_dict: Dict[str, Any],
    max_kfs: int = MAX_TRACK_KEYFRAMES,
    max_dur: float = MAX_PROJECT_DURATION,
) -> Dict[str, Any]:
    """统一执行导演台全局规格：单镜头最大时长 <= 30s，单条轨道最多 <= 10 个关键帧。"""
    if not isinstance(scene_dict, dict):
        return scene_dict

    raw_dur = float(scene_dict.get("duration") or 10.0)
    if raw_dur > max_dur:
        scale = max_dur / raw_dur
        scene_dict["duration"] = max_dur
        for k in scene_dict.get("camera_keyframes") or []:
            k["t"] = round(float(k.get("t", 0.0)) * scale, 3)
        for obj in scene_dict.get("objects") or []:
            for k in obj.get("keyframes") or []:
                k["t"] = round(float(k.get("t", 0.0)) * scale, 3)
    if isinstance(scene_dict.get("camera_keyframes"), list) and len(scene_dict["camera_keyframes"]) > max_kfs:
        scene_dict["camera_keyframes"] = simplify_keyframes_to_limit(scene_dict["camera_keyframes"], max_kfs)
    for obj in scene_dict.get("objects") or []:
        if isinstance(obj.get("keyframes"), list) and len(obj["keyframes"]) > max_kfs:
            obj["keyframes"] = simplify_keyframes_to_limit(obj["keyframes"], max_kfs)
    return scene_dict



@sparse_generated_scene
def build_forest_reveal_scene() -> Dict[str, Any]:
    """
    复现参考视频《雪林四人揭示》(1680x720 ≈ 2.33:1，10.0s @ 24fps，35mm)：
    - 0.0s~1.5s：低机位仰拍，紧贴蹲坐蓝人右后方，护目镜朝右前方
    - 1.5s~2.6s：机位基本不动，镜头向右摇并放平，红人、绿人、岩板入画
    - 2.6s~10s：摄影机向右后方拉出抬高并逐渐减速；红人 3.0s 起从岩板前走向蓝人身旁，
      7.2s 到位，随后转身望向岩板后方；终帧为四人全景
    场景布局由参考片终帧 1680x720 实测反算：35mm、机位高 ≈1.06m、站立圆柱顶球人物总高 1.68m 作为比例尺。
    角色为圆柱体顶着球 (body_style="pawn") + 深色护目镜条 (pawn_visor)。
    """
    import random

    stage = Stage()
    stage.duration = 10.0
    stage.fps = 24
    stage.render_mode = "clay_only"
    stage.environment = {
        "preset": "snow_forest",
        "sky": "#CDD6E0",
        "ground": "#B9C3D0",
        "fog_color": "#BAC4D1",
        "fog_near": 5.0,
        "fog_far": 30.0,
        "fog_max": 0.80,
        "show_grid": False,
    }

    def cyl(name, pos, size, color, rot=None, **extra):
        o = stage.add_cylinder(name, pos=pos, size=size, color=color, rot=rot)
        o.extra.update(extra)
        return o

    # 1. 地面深色斑块（贴地薄片 ground_decal，先于立体物体绘制）
    cyl("地面浅灰斑块", [1.35, 0.004, -0.75], [6.2, 0.008, 5.0], "#5A5F67", ground_decal=True)
    cyl("地面深色斑块", [1.85, 0.010, -1.05], [5.0, 0.008, 4.1], "#33373D", ground_decal=True)
    cyl("地面深色斑块_左上", [0.70, 0.012, -2.05], [2.2, 0.008, 1.2], "#33373D", ground_decal=True)

    # 2. 前景与中景道具
    cyl("左前景粗树干", [-1.98, 8.0, 0.66], [0.72, 16.0, 0.72], "#3B3536")
    cyl("中景深棕树干", [1.10, 8.0, -6.80], [0.55, 16.0, 0.55], "#3E3A3B")
    cyl("中景黑色树桩", [-1.20, 0.26, -1.17], [0.32, 0.52, 0.32], "#24262A")
    cyl("斜卧枯木", [0.62, 0.15, -0.50], [0.30, 2.90, 0.30], "#2A2B2F", rot=[0.0, -114.0, 90.0])
    cyl("右侧横卧粗木", [2.11, 0.30, -0.26], [0.59, 1.10, 0.59], "#3A3B3F", rot=[0.0, 20.0, 90.0])
    cyl("右侧倾斜黑杆_左", [2.30, 1.39, -0.60], [0.18, 2.10, 0.18], "#1F2023", rot=[0.0, 0.0, -13.0])
    cyl("右侧倾斜黑杆_右", [2.88, 1.46, -0.90], [0.16, 2.13, 0.16], "#1F2023", rot=[0.0, 0.0, -4.3])

    # 3. 中景灰色岩板带（带倾角的楔形板块，围成弧形，距终帧机位约 7m）
    rocks = [
        ("岩板_01", [-0.80, 0.26, -3.05], [1.40, 0.50, 0.80], "#6A6A6C", [16.0, 24.0, -7.0]),
        ("岩板_02", [0.45, 0.24, -2.90], [1.30, 0.46, 0.85], "#4C4D50", [-14.0, -12.0, 9.0]),
        ("岩板_03", [1.75, 0.30, -3.20], [1.60, 0.58, 0.90], "#666769", [18.0, 14.0, -10.0]),
        ("岩板_04", [3.05, 0.34, -3.00], [1.50, 0.62, 0.95], "#58595C", [-15.0, -22.0, 12.0]),
        ("岩板_05", [4.30, 0.28, -2.60], [1.20, 0.52, 0.85], "#626366", [12.0, 36.0, -9.0]),
        ("岩板_06", [0.05, 0.20, -3.70], [1.20, 0.36, 0.70], "#6F7073", [-8.0, 40.0, 6.0]),
        ("岩台_绿人脚下", [0.12, 0.17, -3.60], [0.80, 0.34, 0.70], "#5E5F62", [0.0, 15.0, 0.0]),
        ("岩台_黄人脚下", [2.60, 0.20, -3.40], [0.80, 0.40, 0.70], "#626366", [0.0, -20.0, 0.0]),
    ]
    for rname, rpos, rsize, rcol, rrot in rocks:
        stage.add_box(rname, pos=rpos, size=rsize, color=rcol, rot=rrot)

    # 4. 背景林木：稀疏排布（参考片可见大量天空），距离雾由深到浅
    rng = random.Random(469)
    keep_clear = [(-1.98, 0.66, 1.1), (-1.53, 0.37, 1.0), (-0.89, 0.18, 0.9), (-0.10, -1.90, 0.9),
                  (-0.4, -0.7, 0.9), (0.62, -0.50, 1.6), (2.2, -0.4, 1.3), (0.12, -3.6, 0.8),
                  (2.60, -3.4, 0.8), (-0.86, -0.07, 0.6), (-0.75, 0.37, 0.5), (-2.37, -1.45, 0.9), (1.10, -6.80, 0.8),
                  (-1.20, -1.17, 0.6)]
    n_tree = 0
    z = -4.2
    while z > -28.0:
        x = -12.0 + rng.uniform(0.0, 1.8)
        while x < 15.0:
            if all((x - cx) ** 2 + (z - cz) ** 2 > r * r for cx, cz, r in keep_clear):
                d = rng.uniform(0.26, 0.44)
                shade = rng.choice(["#46474C", "#4F5156", "#55575C", "#4A4B50", "#3F4045"])
                n_tree += 1
                cyl(f"林木_{n_tree:02d}", [round(x, 2), 8.0, round(z, 2)], [round(d, 2), 16.0, round(d, 2)], shade)
            x += rng.uniform(2.3, 4.0)
        z -= rng.uniform(3.0, 4.6)

    # 5. 四名角色：圆柱体顶着球 + 护目镜条（参考片纯色：蓝/红/绿/黄）
    def pawn(cid, name, pos, rot_y, color, targets=None, keys=None):
        c = stage.add_character(name, pos=pos, rot_y=rot_y, pose="standing", color=color, body_style="pawn")
        c.id = cid
        c.pawn_visor = True
        if targets:
            c.targets.update({k: list(v) for k, v in targets.items()})
        for t, kp, kr in (keys or [(0.0, pos, rot_y), (10.0, pos, rot_y)]):
            c.add_keyframe(t, pos=kp, rot_y=kr)
        return c

    crouch = {
        "pelvis": [0.0, 0.32, 0.02], "chest": [0.0, 0.62, 0.02], "head_target": [0.0, 0.80, 0.70],
        "foot_L": [0.12, 0.04, 0.26], "foot_R": [-0.12, 0.04, 0.28],
        "knee_L_pole": [0.12, 0.40, 0.48], "knee_R_pole": [-0.12, 0.40, 0.48],
        "hand_L": [0.20, 0.40, 0.24], "hand_R": [-0.20, 0.40, 0.24],
    }
    # 蓝人开场随镜头微微转头（护目镜始终在画面右侧），4.0s 后定向朝右前方
    pawn("forest_blue", "蓝衣人_蹲姿", [-1.53, 0.0, 0.37], 146.0, "#2F7FE0", crouch, keys=[(0.0, [-1.53, 0.0, 0.37], 146.0), (1.5, [-1.53, 0.0, 0.37], 138.0), (2.0, [-1.53, 0.0, 0.37], 116.0), (2.3, [-1.53, 0.0, 0.37], 98.0), (2.6, [-1.53, 0.0, 0.37], 84.0), (4.0, [-1.53, 0.0, 0.37], 100.0), (10.0, [-1.53, 0.0, 0.37], 100.0)])
    # 蓝人前下方的楔形腿部（参考片中蹲姿身体前侧的三角块），朝向与蓝人一致
    stage.add_prism("蓝衣人_腿部楔形", pos=[-1.33, 0.19, 0.335], size=[0.40, 0.38, 0.30], color="#2F7FE0", rot=[0.0, 10.0, 0.0])
    # 红人：开场站在岩板前，3.0s 起走向蓝人身旁（面朝行进方向），7.2s 到位，随后转身望向岩板后方
    red_start, red_end = [-0.10, 0.0, -1.90], [-0.89, 0.0, 0.18]
    pawn("forest_red", "红衣人_走位", red_start, -30.0, "#D8463C", keys=[
        (0.0, red_start, -30.0),
        (3.0, red_start, -30.0),
        (7.2, red_end, -60.0),
        (9.2, red_end, -170.0),
        (10.0, red_end, -170.0),
    ])
    pawn("forest_green", "绿衣人_岩台", [0.12, 0.34, -3.60], -73.0, "#2FAE4F")
    pawn("forest_yellow", "黄衣人_岩台", [2.60, 0.40, -3.40], 80.0, "#E2B52A")

    # 6. 摄影机（35mm）：低机位仰拍贴近蓝人右后方 → 原地右摇放平 → 向右后方拉出抬高 → 减速定格
    F = 35.0
    stage.set_camera(
        pos=[-0.78, 0.53, 0.64],
        target=[-3.52, 1.45, -0.58],
        focal_mm=F,
        aspect_ratio="2.33:1",
        export_width=1120,
        export_height=480,
    )
    cam_keys = [
        (0.0, [-0.78, 0.53, 0.64], [-3.52, 1.45, -0.58]),
        (1.5, [-0.78, 0.55, 0.70], [-3.33, 1.30, -0.89]),
        (2.0, [-0.85, 0.59, 0.86], [-2.61, 1.01, -1.56]),
        (2.3, [-0.96, 0.63, 1.00], [-1.89, 0.84, -1.85]),
        (2.6, [-1.07, 0.66, 1.10], [-1.28, 0.71, -1.89]),
        (3.5, [-1.00, 0.76, 1.55], [-0.25, 0.95, -2.10]),
        (5.0, [-0.50, 0.92, 2.60], [-0.08, 0.97, -1.30]),
        (7.0, [-0.10, 1.03, 3.70], [0.00, 0.98, -0.30]),
        (10.0, [0.00, 1.06, 4.00], [0.00, 0.98, 0.00]),
    ]
    for t, cp, ct in cam_keys:
        stage.add_camera_keyframe(t, pos=cp, target=ct, focal_mm=F)
    return stage.to_dict()


@sparse_generated_scene
def build_clean_dual_character_scene() -> Dict[str, Any]:
    """
    还原参考图极简深色网格 + 纯色圆头人偶舞台：
    - 干净深灰黑地面网格 (1m = 100cm 标准刻度)
    - 人物 1 (#D98178) 与 人物 2 (#73A9D8) 并排自然站立，头顶悬浮带色条名称标签
    """
    stage = Stage()
    stage.duration = 6.0
    stage.fps = 24
    stage.render_mode = "clay_only"

    uaz_pos = [-0.35, 1.01, -0.20]
    uaz_rot_y = 28.0
    uaz_size = [1.78, 2.02, 4.02]
    uaz_obj = stage.add_uaz469(
        "UAZ-469 越野车",
        pos=uaz_pos,
        size=uaz_size,
        color="#434D38",
        rot=[0.0, uaz_rot_y, 0.0],
    )
    uaz_obj.id = "uaz_1"

    # 车内四个位置局部偏移：前排两独立座椅（驾驶+副驾）+ 后舱左右内轮拱（轮包：后左+后右对向而坐）
    wheel_well_targets = {
        "pelvis": [0.0, 0.36, -0.04],
        "chest": [0.0, 0.80, -0.02],
        "head_target": [0.0, 1.04, 0.55],
        "hand_L": [0.15, 0.46, 0.18],
        "hand_R": [-0.15, 0.46, 0.18],
        "foot_L": [0.11, 0.05, 0.32],
        "foot_R": [-0.11, 0.05, 0.32],
        "knee_L_pole": [0.12, 0.44, 0.24],
        "knee_R_pole": [-0.12, 0.44, 0.24],
        "elbow_L_pole": [0.26, 0.58, -0.10],
        "elbow_R_pole": [-0.26, 0.58, -0.10],
        "shoulder_L_offset": [0.0, 0.0, 0.0],
        "shoulder_R_offset": [0.0, 0.0, 0.0],
    }
    seat_specs = [
        ("人物 1 (驾驶)", [0.38, 0.57, 0.02], 0.0, "#D98178", True, False),
        ("人物 2 (副驾)", [-0.38, 0.57, 0.02], 0.0, "#73A9D8", False, False),
        ("人物 3 (后左)", [0.48, 0.56, -1.24], -78.0, "#79B69C", False, True),
        ("人物 4 (后右)", [-0.48, 0.56, -1.24], 78.0, "#D4B56C", False, True),
    ]
    rad_y = math.radians(uaz_rot_y)
    cos_y, sin_y = math.cos(rad_y), math.sin(rad_y)
    ground_base_y = uaz_pos[1] - uaz_size[1] * 0.5

    for idx, (cname, (lx, ly, lz), rot_off, ccol, is_driver, is_wheel_well) in enumerate(seat_specs, start=1):
        wx = round(uaz_pos[0] + lx * cos_y + lz * sin_y, 3)
        wy = round(ground_base_y + ly, 3)
        wz = round(uaz_pos[2] - lx * sin_y + lz * cos_y, 3)
        char_rot_y = round(uaz_rot_y + rot_off, 1)
        c_handle = stage.add_character(
            cname,
            pos=[wx, wy, wz],
            rot_y=char_rot_y,
            pose="sitting",
            color=ccol,
        )
        c_handle.id = f"char_{idx}"
        c_handle.mounted_vehicle_id = "uaz_1"
        c_handle.seat_offset = [lx, ly, lz]
        c_handle.seat_rot_offset = rot_off
        if is_wheel_well:
            c_handle.targets = json.loads(json.dumps(wheel_well_targets))
        elif is_driver:
            # 驾驶员双手握方向盘
            c_handle.targets["hand_L"] = [0.16, 0.68, 0.32]
            c_handle.targets["hand_R"] = [-0.16, 0.68, 0.32]
        c_handle.add_keyframe(0.0, pos=[wx, wy, wz], rot_y=char_rot_y)
        c_handle.add_keyframe(6.0, pos=[wx, wy, wz], rot_y=char_rot_y)

    stage.set_camera(pos=[2.65, 2.05, 3.35], target=[-0.25, 1.08, -0.25], focal_mm=35.0, aspect_ratio="2.39:1")
    stage.add_camera_keyframe(0.0, pos=[2.65, 2.05, 3.35], target=[-0.25, 1.08, -0.25], focal_mm=35.0)
    stage.add_camera_keyframe(6.0, pos=[2.25, 1.88, 2.85], target=[-0.25, 1.08, -0.25], focal_mm=35.0)
    return stage.to_dict()


@sparse_generated_scene
def build_scene64_cabin_scene(shot_mode: str = "shot1") -> Dict[str, Any]:
    """
    第 64 场：猎人原木小屋搏斗现场（SketchUp 灰白体块＋黑色细结构边线风格）
    - 净空尺寸：宽 3.80m × 深 4.40m × 高 2.45m（米制，墙厚 0.18m）
    - 门窗真实留空：南墙前门洞 (0.95m×1.92m) + 内开 78° 木门；北墙六格木窗洞 (0.86m×1.12m)
    - 独立可编辑组合家具：左后双人床+滑落被褥、床头双层搁板、窗下微斜书桌+煤油灯、翻倒方凳、
      右侧双门立式木衣柜、右侧矮木箱+锅桶、右前圆柱形铸铁火炉+敞开炉门+排烟管、散落劈柴与左墙挂绳猎具
    - 支持单面墙/屋顶独立显隐与智能剖切
    - 人物 1 (仰卧猎人)：身高 1.80m，保持角色身份色 (#5B7FA6)，斜卧于地面中央
    """
    focal_mm = 21.0
    fov_deg = round(math.degrees(2.0 * math.atan(15.0 / (2.0 * focal_mm))), 2)

    objects: List[Dict[str, Any]] = [
        # 1. 地面与纵向木板层 (中灰 #C4C6C8)
        {
            "id": "s64_floor",
            "name": "木屋地面 (纵向宽木板)",
            "type": "group",
            "arch_role": "floor",
            "pos": [0.0, 0.0, 0.0],
            "rot": [0.0, 0.0, 0.0],
            "size": [3.96, 0.08, 4.56],
            "color": "#C4C6C8",
            "locked": True,
            "parts": [
                {"id": "fl_base", "name": "地面基板 (3.8m×4.4m)", "shape": "box", "pos": [0.0, -0.04, 0.0], "size": [3.96, 0.08, 4.56], "rot": [0, 0, 0], "color": "#C4C6C8"},
                {"id": "fl_snow_1", "name": "门口吹入积雪层", "shape": "box", "pos": [-0.32, 0.008, 1.72], "size": [1.18, 0.016, 0.78], "rot": [0, 8, 0], "color": "#ECEEEF"},
                {"id": "fl_snow_2", "name": "门槛内侧残雪", "shape": "box", "pos": [-0.65, 0.010, 1.32], "size": [0.74, 0.014, 0.52], "rot": [0, -14, 0], "color": "#E6E8EA"},
            ],
        },
        # 2. 北墙·后窗墙（真留空窗洞 0.86m × 1.12m，窗台高 1.05m，浅灰 #DEDFE1）
        {
            "id": "s64_wall_north",
            "name": "北墙·后窗墙 (真留空六格窗)",
            "type": "group",
            "arch_role": "wall_north",
            "pos": [0.0, 0.0, -2.20],
            "rot": [0.0, 0.0, 0.0],
            "size": [3.96, 2.45, 0.18],
            "color": "#DEDFE1",
            "locked": True,
            "parts": [
                {"id": "wn_left", "name": "窗左墙段", "shape": "box", "pos": [-1.115, 1.225, 0.0], "size": [1.73, 2.45, 0.18], "rot": [0, 0, 0], "color": "#DEDFE1"},
                {"id": "wn_right", "name": "窗右墙段", "shape": "box", "pos": [1.295, 1.225, 0.0], "size": [1.37, 2.45, 0.18], "rot": [0, 0, 0], "color": "#DEDFE1"},
                {"id": "wn_sill_wall", "name": "窗下墙裙", "shape": "box", "pos": [0.18, 0.525, 0.0], "size": [0.86, 1.05, 0.18], "rot": [0, 0, 0], "color": "#DEDFE1"},
                {"id": "wn_lintel", "name": "窗上过梁", "shape": "box", "pos": [0.18, 2.31, 0.0], "size": [0.86, 0.28, 0.18], "rot": [0, 0, 0], "color": "#DEDFE1"},
                # 窗框与窗台板（真实厚度，中间 6 格完全留空）
                {"id": "wn_sill_board", "name": "窗台挑板", "shape": "box", "pos": [0.18, 1.04, 0.02], "size": [0.96, 0.04, 0.24], "rot": [0, 0, 0], "color": "#EAEBED"},
                {"id": "wn_frame_top", "name": "窗框上沿", "shape": "box", "pos": [0.18, 2.15, 0.0], "size": [0.92, 0.05, 0.20], "rot": [0, 0, 0], "color": "#D4D6D8"},
                {"id": "wn_frame_L", "name": "窗框左立柱", "shape": "box", "pos": [-0.24, 1.61, 0.0], "size": [0.05, 1.12, 0.20], "rot": [0, 0, 0], "color": "#D4D6D8"},
                {"id": "wn_frame_R", "name": "窗框右立柱", "shape": "box", "pos": [0.60, 1.61, 0.0], "size": [0.05, 1.12, 0.20], "rot": [0, 0, 0], "color": "#D4D6D8"},
                {"id": "wn_mullion_v", "name": "窗竖中梃", "shape": "box", "pos": [0.18, 1.61, 0.0], "size": [0.03, 1.08, 0.08], "rot": [0, 0, 0], "color": "#EAEBED"},
                {"id": "wn_mullion_h1", "name": "窗下横档", "shape": "box", "pos": [0.18, 1.41, 0.0], "size": [0.82, 0.028, 0.08], "rot": [0, 0, 0], "color": "#EAEBED"},
                {"id": "wn_mullion_h2", "name": "窗上横档", "shape": "box", "pos": [0.18, 1.79, 0.0], "size": [0.82, 0.028, 0.08], "rot": [0, 0, 0], "color": "#EAEBED"},
            ],
        },
        # 3. 南墙·前门墙（真留空门洞 1.12m × 1.96m，浅灰 #DEDFE1）
        {
            "id": "s64_wall_south",
            "name": "南墙·前门墙 (真留空门洞)",
            "type": "group",
            "arch_role": "wall_south",
            "pos": [0.0, 0.0, 2.20],
            "rot": [0.0, 0.0, 0.0],
            "size": [3.96, 2.45, 0.18],
            "color": "#DEDFE1",
            "locked": True,
            "parts": [
                {"id": "ws_left", "name": "门左墙段", "shape": "box", "pos": [-1.34, 1.225, 0.0], "size": [1.28, 2.45, 0.18], "rot": [0, 0, 0], "color": "#DEDFE1"},
                {"id": "ws_right", "name": "门右墙段", "shape": "box", "pos": [1.20, 1.225, 0.0], "size": [1.56, 2.45, 0.18], "rot": [0, 0, 0], "color": "#DEDFE1"},
                {"id": "ws_lintel", "name": "门上过梁", "shape": "box", "pos": [-0.14, 2.20, 0.0], "size": [1.12, 0.50, 0.18], "rot": [0, 0, 0], "color": "#DEDFE1"},
                {"id": "ws_jamb_L", "name": "左门框柱", "shape": "box", "pos": [-0.68, 0.98, 0.0], "size": [0.05, 1.96, 0.21], "rot": [0, 0, 0], "color": "#D2D4D6"},
                {"id": "ws_jamb_R", "name": "右门框柱", "shape": "box", "pos": [0.40, 0.98, 0.0], "size": [0.05, 1.96, 0.21], "rot": [0, 0, 0], "color": "#D2D4D6"},
                {"id": "ws_jamb_T", "name": "上门框横梁", "shape": "box", "pos": [-0.14, 1.96, 0.0], "size": [1.14, 0.05, 0.21], "rot": [0, 0, 0], "color": "#D2D4D6"},
                {"id": "ws_threshold", "name": "木门槛", "shape": "box", "pos": [-0.14, 0.02, 0.0], "size": [1.12, 0.04, 0.20], "rot": [0, 0, 0], "color": "#C8CACD"},
            ],
        },
        # 4. 前木门（独立物体，绕左门轴向屋内敞开 80°，具有真实板厚与穿带横方）
        {
            "id": "s64_door_front",
            "name": "前木门 (向内左开 80°)",
            "type": "group",
            "arch_role": "door",
            "pos": [-0.68, 0.0, 2.10],
            "rot": [0.0, 80.0, 0.0],
            "size": [0.92, 1.92, 0.08],
            "color": "#D6D8DA",
            "locked": False,
            "parts": [
                {"id": "dr_panel", "name": "门扇主板", "shape": "box", "pos": [0.46, 0.96, 0.0], "size": [0.92, 1.92, 0.055], "rot": [0, 0, 0], "color": "#D6D8DA"},
                {"id": "dr_batten_top", "name": "门上穿带横木", "shape": "box", "pos": [0.46, 1.52, 0.036], "size": [0.86, 0.11, 0.028], "rot": [0, 0, 0], "color": "#EAEBED"},
                {"id": "dr_batten_mid", "name": "门中穿带横木", "shape": "box", "pos": [0.46, 0.92, 0.036], "size": [0.86, 0.11, 0.028], "rot": [0, 0, 0], "color": "#EAEBED"},
                {"id": "dr_batten_bot", "name": "门下穿带横木", "shape": "box", "pos": [0.46, 0.34, 0.036], "size": [0.86, 0.11, 0.028], "rot": [0, 0, 0], "color": "#EAEBED"},
                {"id": "dr_handle", "name": "门拉手", "shape": "box", "pos": [0.80, 0.96, 0.065], "size": [0.03, 0.18, 0.05], "rot": [0, 0, 0], "color": "#BFC1C4"},
            ],
        },
        # 5. 西墙·左墙（干净浅灰实体墙面 #DCDDDD）
        {
            "id": "s64_wall_west",
            "name": "西墙·左墙",
            "type": "group",
            "arch_role": "wall_west",
            "pos": [-1.90, 0.0, 0.0],
            "rot": [0.0, 0.0, 0.0],
            "size": [0.18, 2.45, 4.40],
            "color": "#DCDDDD",
            "locked": True,
            "parts": [
                {"id": "ww_main", "name": "西墙主体", "shape": "box", "pos": [0.0, 1.225, 0.0], "size": [0.18, 2.45, 4.40], "rot": [0, 0, 0], "color": "#DCDDDD"},
            ],
        },
        # 6. 东墙·右墙（干净浅灰实体墙面 #DCDDDD）
        {
            "id": "s64_wall_east",
            "name": "东墙·右墙",
            "type": "group",
            "arch_role": "wall_east",
            "pos": [1.90, 0.0, 0.0],
            "rot": [0.0, 0.0, 0.0],
            "size": [0.18, 2.45, 4.40],
            "color": "#DCDDDD",
            "locked": True,
            "parts": [
                {"id": "we_main", "name": "东墙主体", "shape": "box", "pos": [0.0, 1.225, 0.0], "size": [0.18, 2.45, 4.40], "rot": [0, 0, 0], "color": "#DCDDDD"},
            ],
        },
        # 7. 木屋屋顶与室内横梁（支持一键隐藏屋顶俯视布光摆机位）
        {
            "id": "s64_roof",
            "name": "木屋屋顶与横梁",
            "type": "group",
            "arch_role": "roof",
            "pos": [0.0, 2.45, 0.0],
            "rot": [0.0, 0.0, 0.0],
            "size": [3.98, 0.16, 4.58],
            "color": "#E2E3E5",
            "locked": True,
            "parts": [
                {"id": "rf_slab", "name": "屋顶顶板", "shape": "box", "pos": [0.0, 0.06, 0.0], "size": [3.98, 0.12, 4.58], "rot": [0, 0, 0], "color": "#E2E3E5"},
                {"id": "rf_beam_1", "name": "后横梁", "shape": "box", "pos": [0.0, -0.06, -1.25], "size": [3.76, 0.12, 0.14], "rot": [0, 0, 0], "color": "#D2D4D6"},
                {"id": "rf_beam_2", "name": "中横梁", "shape": "box", "pos": [0.0, -0.06, 0.0], "size": [3.76, 0.12, 0.14], "rot": [0, 0, 0], "color": "#D2D4D6"},
                {"id": "rf_beam_3", "name": "前横梁", "shape": "box", "pos": [0.0, -0.06, 1.25], "size": [3.76, 0.12, 0.14], "rot": [0, 0, 0], "color": "#D2D4D6"},
            ],
        },
        # 8. 左后双人木床与凌乱垂落被褥（组合家具：宽 1.46m × 长 1.65m × 高 0.42m）
        {
            "id": "s64_bed",
            "name": "左后双人木床与被褥 (组合)",
            "type": "group",
            "pos": [-1.10, 0.0, -1.32],
            "rot": [0.0, 0.0, 0.0],
            "size": [1.46, 0.65, 1.65],
            "color": "#EAEBED",
            "locked": False,
            "parts": [
                {"id": "bed_frame", "name": "床架围框", "shape": "box", "pos": [0.0, 0.28, 0.0], "size": [1.46, 0.14, 1.64], "rot": [0, 0, 0], "color": "#D4D6D9"},
                {"id": "bed_headboard", "name": "床头挡板", "shape": "box", "pos": [0.0, 0.52, -0.79], "size": [1.46, 0.36, 0.06], "rot": [0, 0, 0], "color": "#D4D6D9"},
                {"id": "bed_leg_fl", "name": "左前床腿", "shape": "box", "pos": [-0.66, 0.14, 0.76], "size": [0.08, 0.28, 0.08], "rot": [0, 0, 0], "color": "#C8CACD"},
                {"id": "bed_leg_fr", "name": "右前床腿", "shape": "box", "pos": [0.66, 0.14, 0.76], "size": [0.08, 0.28, 0.08], "rot": [0, 0, 0], "color": "#C8CACD"},
                {"id": "bed_leg_bl", "name": "左后床腿", "shape": "box", "pos": [-0.66, 0.14, -0.76], "size": [0.08, 0.28, 0.08], "rot": [0, 0, 0], "color": "#C8CACD"},
                {"id": "bed_leg_br", "name": "右后床腿", "shape": "box", "pos": [0.66, 0.14, -0.76], "size": [0.08, 0.28, 0.08], "rot": [0, 0, 0], "color": "#C8CACD"},
                {"id": "bed_mattress", "name": "床垫主体", "shape": "box", "pos": [0.0, 0.42, 0.0], "size": [1.42, 0.16, 1.60], "rot": [0, 0, 0], "color": "#E6E7E9"},
                {"id": "bed_pillow_L", "name": "左枕头", "shape": "box", "pos": [-0.36, 0.54, -0.58], "size": [0.54, 0.11, 0.32], "rot": [6, 8, 0], "color": "#F2F3F5"},
                {"id": "bed_pillow_R", "name": "右枕头", "shape": "box", "pos": [0.32, 0.54, -0.56], "size": [0.54, 0.11, 0.32], "rot": [5, -10, 0], "color": "#F2F3F5"},
                {"id": "bed_quilt_main", "name": "床上蓬松厚被", "shape": "box", "pos": [0.06, 0.55, 0.08], "size": [1.34, 0.14, 1.14], "rot": [2, -6, -3], "color": "#ECEEEF"},
                {"id": "bed_quilt_fold1", "name": "被褥上层褶块", "shape": "box", "pos": [0.18, 0.62, 0.12], "size": [0.88, 0.08, 0.82], "rot": [3, 14, -4], "color": "#F0F1F3"},
                {"id": "bed_quilt_drape", "name": "向右侧滑落被角", "shape": "box", "pos": [0.72, 0.28, 0.26], "size": [0.36, 0.44, 0.78], "rot": [6, 10, -32], "color": "#EAEBED"},
                {"id": "bed_quilt_floor", "name": "拖地床单褶块", "shape": "box", "pos": [0.92, 0.06, 0.34], "size": [0.50, 0.10, 0.64], "rot": [0, 22, 0], "color": "#E4E6E8"},
            ],
        },
        # 9. 床头后墙双层木搁板与碗罐（组合）
        {
            "id": "s64_shelf_north",
            "name": "后墙双层木搁板与器皿 (组合)",
            "type": "group",
            "pos": [-1.05, 1.48, -2.00],
            "rot": [0.0, 0.0, 0.0],
            "size": [1.15, 0.52, 0.24],
            "color": "#E4E5E7",
            "locked": False,
            "parts": [
                {"id": "sh_board_low", "name": "下层搁板", "shape": "box", "pos": [0.0, 0.0, 0.0], "size": [1.15, 0.03, 0.22], "rot": [0, 0, 0], "color": "#E2E4E6"},
                {"id": "sh_board_high", "name": "上层搁板", "shape": "box", "pos": [0.0, 0.34, 0.0], "size": [1.15, 0.03, 0.22], "rot": [0, 0, 0], "color": "#E2E4E6"},
                {"id": "sh_brk_L", "name": "左立柱托架", "shape": "box", "pos": [-0.48, 0.17, -0.06], "size": [0.04, 0.48, 0.10], "rot": [0, 0, 0], "color": "#CFD1D4"},
                {"id": "sh_brk_R", "name": "右立柱托架", "shape": "box", "pos": [0.48, 0.17, -0.06], "size": [0.04, 0.48, 0.10], "rot": [0, 0, 0], "color": "#CFD1D4"},
                {"id": "sh_pot_1", "name": "下层陶罐", "shape": "cylinder", "pos": [-0.32, 0.09, 0.0], "size": [0.13, 0.15, 0.13], "rot": [0, 0, 0], "color": "#EFEFF1"},
                {"id": "sh_bowl_1", "name": "下层木碗", "shape": "cylinder", "pos": [-0.05, 0.06, 0.02], "size": [0.14, 0.08, 0.14], "rot": [0, 0, 0], "color": "#EAEBED"},
                {"id": "sh_pot_2", "name": "上层储物罐", "shape": "cylinder", "pos": [-0.25, 0.43, 0.0], "size": [0.12, 0.15, 0.12], "rot": [0, 0, 0], "color": "#EFEFF1"},
                {"id": "sh_cup_2", "name": "上层水杯", "shape": "cylinder", "pos": [0.18, 0.41, 0.0], "size": [0.09, 0.11, 0.09], "rot": [0, 0, 0], "color": "#EAEBED"},
            ],
        },
        # 10. 后窗下木书桌与煤油灯（组合：长 1.10m × 宽 0.68m × 高 0.76m，微斜 -12°）
        {
            "id": "s64_desk",
            "name": "窗下木书桌与煤油灯 (组合)",
            "type": "group",
            "pos": [0.54, 0.0, -1.52],
            "rot": [0.0, -12.0, 0.0],
            "size": [1.10, 1.05, 0.68],
            "color": "#EAEBED",
            "locked": False,
            "parts": [
                {"id": "dk_top", "name": "书桌桌面", "shape": "box", "pos": [0.0, 0.74, 0.0], "size": [1.10, 0.045, 0.68], "rot": [0, 0, 0], "color": "#EAEBED"},
                {"id": "dk_apron_f", "name": "前横枨裙板", "shape": "box", "pos": [0.0, 0.66, 0.28], "size": [0.98, 0.10, 0.03], "rot": [0, 0, 0], "color": "#D8DADD"},
                {"id": "dk_apron_b", "name": "后横枨裙板", "shape": "box", "pos": [0.0, 0.66, -0.28], "size": [0.98, 0.10, 0.03], "rot": [0, 0, 0], "color": "#D8DADD"},
                {"id": "dk_apron_l", "name": "左横枨裙板", "shape": "box", "pos": [-0.48, 0.66, 0.0], "size": [0.03, 0.10, 0.56], "rot": [0, 0, 0], "color": "#D8DADD"},
                {"id": "dk_apron_r", "name": "右横枨裙板", "shape": "box", "pos": [0.48, 0.66, 0.0], "size": [0.03, 0.10, 0.56], "rot": [0, 0, 0], "color": "#D8DADD"},
                {"id": "dk_leg_fl", "name": "左前桌腿", "shape": "box", "pos": [-0.48, 0.36, 0.28], "size": [0.06, 0.72, 0.06], "rot": [0, 0, 0], "color": "#D2D4D7"},
                {"id": "dk_leg_fr", "name": "右前桌腿", "shape": "box", "pos": [0.48, 0.36, 0.28], "size": [0.06, 0.72, 0.06], "rot": [0, 0, 0], "color": "#D2D4D7"},
                {"id": "dk_leg_bl", "name": "左后桌腿", "shape": "box", "pos": [-0.48, 0.36, -0.28], "size": [0.06, 0.72, 0.06], "rot": [0, 0, 0], "color": "#D2D4D7"},
                {"id": "dk_leg_br", "name": "右后桌腿", "shape": "box", "pos": [0.48, 0.36, -0.28], "size": [0.06, 0.72, 0.06], "rot": [0, 0, 0], "color": "#D2D4D7"},
                # 桌面道具：煤油提灯、散落纸张、木盒、搪瓷杯
                {"id": "dk_lamp_base", "name": "煤油灯底座", "shape": "cylinder", "pos": [0.08, 0.80, 0.04], "size": [0.14, 0.08, 0.14], "rot": [0, 0, 0], "color": "#CFD1D4"},
                {"id": "dk_lamp_Chimney", "name": "煤油灯罩", "shape": "cylinder", "pos": [0.08, 0.91, 0.04], "size": [0.10, 0.16, 0.10], "rot": [0, 0, 0], "color": "#F7F7F5"},
                {"id": "dk_lamp_cap", "name": "煤油灯顶盖", "shape": "cylinder", "pos": [0.08, 1.00, 0.04], "size": [0.12, 0.03, 0.12], "rot": [0, 0, 0], "color": "#CFD1D4"},
                {"id": "dk_paper_1", "name": "桌面图纸左", "shape": "box", "pos": [-0.22, 0.77, 0.06], "size": [0.34, 0.008, 0.26], "rot": [0, 16, 0], "color": "#F4F5F6"},
                {"id": "dk_paper_2", "name": "桌面信纸右", "shape": "box", "pos": [0.28, 0.77, 0.12], "size": [0.28, 0.008, 0.22], "rot": [0, -22, 0], "color": "#F4F5F6"},
                {"id": "dk_cup", "name": "桌面水杯", "shape": "cylinder", "pos": [-0.30, 0.81, -0.08], "size": [0.08, 0.09, 0.08], "rot": [0, 0, 0], "color": "#EFEFF1"},
                {"id": "dk_box", "name": "桌面木匣", "shape": "box", "pos": [0.34, 0.81, -0.14], "size": [0.22, 0.09, 0.16], "rot": [0, -8, 0], "color": "#D8DADD"},
            ],
        },
        # 11. 书桌下方翻倒的木方凳（组合）
        {
            "id": "s64_stool_fallen",
            "name": "翻倒的木方凳 (组合)",
            "type": "group",
            "pos": [0.52, 0.20, -1.08],
            "rot": [68.0, 32.0, -18.0],
            "size": [0.38, 0.44, 0.38],
            "color": "#E2E4E6",
            "locked": False,
            "parts": [
                {"id": "st_seat", "name": "方凳面板", "shape": "box", "pos": [0.0, 0.20, 0.0], "size": [0.36, 0.035, 0.36], "rot": [0, 0, 0], "color": "#E2E4E6"},
                {"id": "st_leg1", "name": "凳腿 1", "shape": "box", "pos": [-0.14, -0.01, -0.14], "size": [0.04, 0.40, 0.04], "rot": [0, 0, 0], "color": "#CFD1D4"},
                {"id": "st_leg2", "name": "凳腿 2", "shape": "box", "pos": [0.14, -0.01, -0.14], "size": [0.04, 0.40, 0.04], "rot": [0, 0, 0], "color": "#CFD1D4"},
                {"id": "st_leg3", "name": "凳腿 3", "shape": "box", "pos": [-0.14, -0.01, 0.14], "size": [0.04, 0.40, 0.04], "rot": [0, 0, 0], "color": "#CFD1D4"},
                {"id": "st_leg4", "name": "凳腿 4", "shape": "box", "pos": [0.14, -0.01, 0.14], "size": [0.04, 0.40, 0.04], "rot": [0, 0, 0], "color": "#CFD1D4"},
            ],
        },
        # 12. 后墙右侧钉挂地图图纸（组合）
        {
            "id": "s64_wall_maps",
            "name": "后墙钉挂地图与图纸 (组合)",
            "type": "group",
            "pos": [0.95, 1.48, -2.09],
            "rot": [0.0, 0.0, 0.0],
            "size": [0.64, 0.78, 0.03],
            "color": "#F1F2F4",
            "locked": False,
            "parts": [
                {"id": "mp_1", "name": "主大幅地图", "shape": "box", "pos": [-0.08, -0.10, 0.0], "size": [0.38, 0.46, 0.012], "rot": [0, 0, -2], "color": "#F1F2F4"},
                {"id": "mp_2", "name": "右上笔记纸", "shape": "box", "pos": [0.16, 0.18, 0.005], "size": [0.28, 0.34, 0.012], "rot": [0, 0, 4], "color": "#ECEEEF"},
                {"id": "mp_3", "name": "左上便签纸", "shape": "box", "pos": [-0.18, 0.24, 0.004], "size": [0.22, 0.26, 0.012], "rot": [0, 0, -5], "color": "#F4F5F6"},
            ],
        },
        # 13. 右侧双门立式木衣柜（组合：宽 0.90m × 深 0.62m × 高 1.88m）
        {
            "id": "s64_wardrobe",
            "name": "右侧双门立式木衣柜 (组合)",
            "type": "group",
            "pos": [1.46, 0.0, -1.02],
            "rot": [0.0, -90.0, 0.0],
            "size": [0.90, 1.88, 0.62],
            "color": "#E2E3E5",
            "locked": False,
            "parts": [
                {"id": "wd_body", "name": "衣柜主体箱", "shape": "box", "pos": [0.0, 0.92, 0.0], "size": [0.88, 1.80, 0.58], "rot": [0, 0, 0], "color": "#E2E3E5"},
                {"id": "wd_crown", "name": "衣柜顶冠檐板", "shape": "box", "pos": [0.0, 1.85, 0.01], "size": [0.94, 0.06, 0.63], "rot": [0, 0, 0], "color": "#D4D6D9"},
                {"id": "wd_door_L", "name": "左柜门面板", "shape": "box", "pos": [-0.215, 0.94, 0.30], "size": [0.41, 1.66, 0.025], "rot": [0, 0, 0], "color": "#EAEBED"},
                {"id": "wd_door_R", "name": "右柜门面板", "shape": "box", "pos": [0.215, 0.94, 0.30], "size": [0.41, 1.66, 0.025], "rot": [0, 0, 0], "color": "#EAEBED"},
                {"id": "wd_knob_L", "name": "左门圆拉手", "shape": "box", "pos": [-0.04, 0.94, 0.325], "size": [0.03, 0.04, 0.03], "rot": [0, 0, 0], "color": "#BFC1C4"},
                {"id": "wd_knob_R", "name": "右门圆拉手", "shape": "box", "pos": [0.04, 0.94, 0.325], "size": [0.03, 0.04, 0.03], "rot": [0, 0, 0], "color": "#BFC1C4"},
            ],
        },
        # 14. 右侧中段矮木箱与铁锅木桶（组合）
        {
            "id": "s64_chest_right",
            "name": "右侧储物木箱与锅桶 (组合)",
            "type": "group",
            "pos": [1.50, 0.0, -0.14],
            "rot": [0.0, -90.0, 0.0],
            "size": [0.76, 0.85, 0.52],
            "color": "#DCDEDF",
            "locked": False,
            "parts": [
                {"id": "ch_body", "name": "木箱箱体", "shape": "box", "pos": [0.0, 0.26, 0.0], "size": [0.74, 0.52, 0.50], "rot": [0, 0, 0], "color": "#DCDEDF"},
                {"id": "ch_lid", "name": "木箱盖板", "shape": "box", "pos": [0.0, 0.54, 0.0], "size": [0.78, 0.04, 0.54], "rot": [0, 0, 0], "color": "#EAEBED"},
                {"id": "ch_bucket", "name": "箱顶木桶", "shape": "cylinder", "pos": [-0.18, 0.70, -0.05], "size": [0.26, 0.28, 0.26], "rot": [0, 0, 0], "color": "#E4E6E8"},
                {"id": "ch_pan", "name": "箱顶平底锅", "shape": "cylinder", "pos": [0.16, 0.62, 0.04], "size": [0.24, 0.12, 0.24], "rot": [0, 0, 0], "color": "#CFD1D4"},
            ],
        },
        # 15. 右前圆柱形铸铁火炉与排烟管（组合：直径 0.52m，炉门向内敞开，带排烟管与顶部小锅）
        {
            "id": "s64_stove",
            "name": "右前铸铁圆火炉与烟囱 (组合)",
            "type": "group",
            "pos": [1.26, 0.0, 0.56],
            "rot": [0.0, 0.0, 0.0],
            "size": [0.62, 2.40, 0.62],
            "color": "#CFD1D4",
            "locked": False,
            "parts": [
                {"id": "stv_body", "name": "圆柱形炉身", "shape": "cylinder", "pos": [0.0, 0.49, 0.0], "size": [0.52, 0.74, 0.52], "rot": [0, 0, 0], "color": "#CFD1D4"},
                {"id": "stv_top_rim", "name": "炉顶盖盘", "shape": "cylinder", "pos": [0.0, 0.87, 0.0], "size": [0.55, 0.03, 0.55], "rot": [0, 0, 0], "color": "#DCDEDF"},
                {"id": "stv_leg1", "name": "炉腿前左", "shape": "box", "pos": [-0.16, 0.06, 0.16], "size": [0.05, 0.12, 0.05], "rot": [0, 0, 0], "color": "#BFC1C4"},
                {"id": "stv_leg2", "name": "炉腿前右", "shape": "box", "pos": [0.16, 0.06, 0.16], "size": [0.05, 0.12, 0.05], "rot": [0, 0, 0], "color": "#BFC1C4"},
                {"id": "stv_leg3", "name": "炉腿后左", "shape": "box", "pos": [-0.16, 0.06, -0.16], "size": [0.05, 0.12, 0.05], "rot": [0, 0, 0], "color": "#BFC1C4"},
                {"id": "stv_leg4", "name": "炉腿后右", "shape": "box", "pos": [0.16, 0.06, -0.16], "size": [0.05, 0.12, 0.05], "rot": [0, 0, 0], "color": "#BFC1C4"},
                {"id": "stv_firebox", "name": "炉膛火口内腔", "shape": "box", "pos": [-0.23, 0.44, 0.0], "size": [0.10, 0.22, 0.22], "rot": [0, 0, 0], "color": "#F7F7F5"},
                {"id": "stv_door", "name": "敞开的铸铁炉门", "shape": "box", "pos": [-0.28, 0.44, -0.22], "size": [0.22, 0.23, 0.03], "rot": [0, -25, 0], "color": "#D6D8DA"},
                {"id": "stv_pot", "name": "炉顶铁锅", "shape": "cylinder", "pos": [-0.04, 0.96, 0.02], "size": [0.23, 0.16, 0.23], "rot": [0, 0, 0], "color": "#D8DADD"},
                {"id": "stv_flue", "name": "竖向排烟铁管", "shape": "cylinder", "pos": [0.16, 1.62, -0.12], "size": [0.13, 1.56, 0.13], "rot": [0, 0, 0], "color": "#C8CACD"},
            ],
        },
        # 16. 火炉前散落劈柴与右门边整齐柴垛（组合：用楔形 prism 与长方体组合还原劈柴堆）
        {
            "id": "s64_firewood",
            "name": "火炉前散落劈柴与门边柴垛 (组合)",
            "type": "group",
            "pos": [1.12, 0.0, 1.15],
            "rot": [0.0, 0.0, 0.0],
            "size": [1.20, 0.85, 1.50],
            "color": "#EAEBED",
            "locked": False,
            "parts": [
                {"id": "fw_stack", "name": "门边靠墙整齐柴垛", "shape": "box", "pos": [0.36, 0.40, 0.62], "size": [0.56, 0.80, 0.42], "rot": [0, 0, 0], "color": "#DCDEDF"},
                {"id": "fw_stack_top1", "name": "柴垛顶层原木 1", "shape": "prism", "pos": [0.24, 0.84, 0.62], "size": [0.16, 0.10, 0.40], "rot": [0, 0, 0], "color": "#EAEBED"},
                {"id": "fw_stack_top2", "name": "柴垛顶层原木 2", "shape": "prism", "pos": [0.44, 0.84, 0.62], "size": [0.16, 0.10, 0.40], "rot": [0, 0, 0], "color": "#EAEBED"},
                {"id": "fw_log_1", "name": "散落劈柴 1", "shape": "prism", "pos": [-0.18, 0.06, -0.25], "size": [0.14, 0.11, 0.42], "rot": [0, 34, 0], "color": "#EAEBED"},
                {"id": "fw_log_2", "name": "散落劈柴 2", "shape": "box", "pos": [-0.05, 0.09, -0.08], "size": [0.13, 0.10, 0.44], "rot": [8, -28, 12], "color": "#E2E4E6"},
                {"id": "fw_log_3", "name": "散落劈柴 3", "shape": "prism", "pos": [-0.28, 0.05, 0.12], "size": [0.15, 0.11, 0.40], "rot": [0, 62, 0], "color": "#ECEEEF"},
                {"id": "fw_log_4", "name": "散落劈柴 4", "shape": "box", "pos": [0.08, 0.07, 0.22], "size": [0.14, 0.11, 0.45], "rot": [-6, 18, 0], "color": "#E2E4E6"},
                {"id": "fw_log_5", "name": "散落劈柴 5", "shape": "prism", "pos": [-0.14, 0.12, 0.36], "size": [0.15, 0.12, 0.44], "rot": [12, -42, 8], "color": "#EAEBED"},
                {"id": "fw_log_6", "name": "散落劈柴 6", "shape": "box", "pos": [-0.36, 0.05, -0.52], "size": [0.12, 0.09, 0.38], "rot": [0, -54, 0], "color": "#ECEEEF"},
                {"id": "fw_kettle", "name": "翻倒的铁水壶", "shape": "cylinder", "pos": [-0.42, 0.11, -0.82], "size": [0.22, 0.22, 0.22], "rot": [58, 24, 0], "color": "#D2D4D7"},
            ],
        },
        # 17. 左墙挂绳与地面猎具杂物堆（组合）
        {
            "id": "s64_left_gear",
            "name": "左墙挂绳与地面猎具堆 (组合)",
            "type": "group",
            "pos": [-1.42, 0.0, 0.42],
            "rot": [0.0, 0.0, 0.0],
            "size": [0.76, 1.85, 1.35],
            "color": "#DCDEDF",
            "locked": False,
            "parts": [
                {"id": "lg_rack", "name": "左墙木挂架横条", "shape": "box", "pos": [-0.36, 1.72, 0.0], "size": [0.06, 0.08, 1.10], "rot": [0, 0, 0], "color": "#CFD1D4"},
                {"id": "lg_rope_1", "name": "垂挂盘绳 1", "shape": "box", "pos": [-0.32, 1.32, -0.28], "size": [0.09, 0.68, 0.22], "rot": [0, 0, 4], "color": "#E6E8EA"},
                {"id": "lg_rope_2", "name": "垂挂盘绳 2", "shape": "box", "pos": [-0.31, 1.28, 0.02], "size": [0.10, 0.74, 0.24], "rot": [0, 0, 5], "color": "#E2E4E6"},
                {"id": "lg_satchel", "name": "悬挂猎人挎包", "shape": "box", "pos": [-0.28, 0.96, -0.14], "size": [0.14, 0.28, 0.26], "rot": [0, 8, 6], "color": "#D6D8DA"},
                {"id": "lg_crate", "name": "地面木箱", "shape": "box", "pos": [-0.12, 0.22, -0.36], "size": [0.46, 0.44, 0.48], "rot": [0, 12, 0], "color": "#D8DADD"},
                {"id": "lg_pile_1", "name": "地面厚衣物堆", "shape": "prism", "pos": [0.05, 0.18, 0.18], "size": [0.58, 0.34, 0.72], "rot": [0, -18, 0], "color": "#E2E4E6"},
                {"id": "lg_boot", "name": "门后地面猎靴", "shape": "box", "pos": [0.16, 0.12, 0.62], "size": [0.22, 0.24, 0.28], "rot": [0, 28, 0], "color": "#CFD1D4"},
            ],
        },
        # 18. 门外与窗外夜景背景板（提供门洞、窗洞真实空间纵深感）
        {
            "id": "s64_outdoor_backdrop",
            "name": "门外雪地与夜空背景体块",
            "type": "group",
            "arch_role": "backdrop",
            "pos": [0.0, 0.0, 0.0],
            "rot": [0.0, 0.0, 0.0],
            "size": [4.20, 2.50, 6.80],
            "color": "#9499A1",
            "locked": True,
            "parts": [
                {"id": "od_south_snow", "name": "门外雪坡地面", "shape": "box", "pos": [-0.15, -0.02, 3.15], "size": [2.80, 0.06, 1.80], "rot": [0, 0, 0], "color": "#9FA4AC"},
                {"id": "od_south_sky", "name": "门外深灰夜空背板", "shape": "box", "pos": [-0.15, 1.25, 3.95], "size": [2.80, 2.50, 0.08], "rot": [0, 0, 0], "color": "#7E848D"},
                {"id": "od_north_sky", "name": "窗外夜空背板", "shape": "box", "pos": [0.18, 1.55, -2.75], "size": [1.60, 1.60, 0.08], "rot": [0, 0, 0], "color": "#858B94"},
            ],
        },
        # 19. 核心角色：人物 1 (仰卧猎人) — 身高 1.80m，保持角色身份色 #5B7FA6，头朝左后床铺、脚朝右前火炉斜卧
        {
            "id": "char_cabin_1",
            "name": "人物 1 (仰卧猎人)",
            "type": "character",
            "body_style": "mannequin",
            "pos": [-0.04, 0.0, 0.06],
            "rot": [0.0, 42.0, 0.0],
            "scale": [1.06, 1.06, 1.06],
            "torso_twist": 0.0,
            "color": "#5B7FA6",
            "pose": "lying_supine",
            "foot_ground_clamp": False,
            "targets": copy.deepcopy(PRESET_IK_POSES["lying_supine"]["targets"]),
            "pins": {"foot_L": True, "foot_R": True, "hand_L": True, "hand_R": True},
            "keyframes": [
                {
                    "t": 0.0,
                    "pos": [-0.04, 0.0, 0.06],
                    "rot": [0.0, 42.0, 0.0],
                    "rot_y": 42.0,
                    "targets": copy.deepcopy(PRESET_IK_POSES["lying_supine"]["targets"]),
                },
                {
                    "t": 6.0,
                    "pos": [-0.04, 0.0, 0.06],
                    "rot": [0.0, 42.0, 0.0],
                    "rot_y": 42.0,
                    "targets": copy.deepcopy(PRESET_IK_POSES["lying_supine"]["targets"]),
                },
            ],
        },
    ]

    if shot_mode == "shot2":
        # 第 2 镜：书桌上方朝门口反打俯拍 (对应参考图 2，前景带书桌煤油灯与右下方木床)
        cam_pos = [0.16, 1.54, -2.12]
        cam_target = [-0.06, 0.34, 0.32]
    else:
        # 第 1 镜：门口向内正打全景 (对应参考图 1，左前景带敞开木门，完整呈现躺地猎人与全屋陈设)
        cam_pos = [-0.02, 1.18, 2.24]
        cam_target = [-0.02, 0.38, -0.45]

    return {
        "duration": 6.0,
        "fps": 24,
        "render_mode": "clay_only",
        "visual_style": "sketchup_massing",
        "camera": {
            "pos": list(cam_pos),
            "target": list(cam_target),
            "fov": fov_deg,
            "focal_mm": focal_mm,
            "aspect_ratio": "2.39:1",
        },
        "camera_keyframes": [
            {"t": 0.0, "pos": list(cam_pos), "target": list(cam_target), "fov": fov_deg, "focal_mm": focal_mm},
            {"t": 6.0, "pos": list(cam_pos), "target": list(cam_target), "fov": fov_deg, "focal_mm": focal_mm},
        ],
        "objects": objects,
    }


@sparse_generated_scene
def build_fireplace_cabin_su_scene(shot_mode: str = "shot1") -> Dict[str, Any]:
    """
    猎人木屋·壁炉与长餐桌大堂（SketchUp 灰白体块＋黑色细结构边线风格 `sketchup_massing` · 5人偶人体工学比例校准版）
    依据实拍群像参考图（5名猎人站/坐/俯身盛汤饮酒比例）精确校准三维建筑净空与家具尺度：
    - 校准后室内净空：长 6.30m (X: -3.15..+3.15) × 宽 4.00m (Z: -2.00..+2.00，南北墙严格对称闭合) × 墙高 2.38m（粗木横梁底净高 2.10m）
    - 西墙 (X=-3.15)：石砌壁炉（炉台高 0.22m、炉口高 1.18m、粗木挑台顶高 1.58m，与弯腰盛汤猎人头顶齐平）+ 左前矮石墩黑铁汤锅（锅沿高 0.58m）
    - 北墙 (Z=-2.00)：三扇四格木窗 (X=-1.45 / -0.06 / +1.48，窗台高 0.90m、窗上沿 1.82m) + 远端窗下木榻 + 中段矮木柜 + 右前工作木桌
    - 南墙 (Z=+2.00)：镶板木门（净高 1.88m，X=-0.22）+ 左前挂衣架与猎装 + 坐箱饮水矮木箱（高 0.38m）+ 西段三角撑壁架与靠墙矮箱凳
    - 室内偏北：2.26m×0.80m×0.75m 长木餐桌（Z=-0.74，左沿对齐炉口右缘，留出南侧 1.4m 宽木地板通道）+ 四周长木凳（高 0.44m）
    - 5 名纯色身份人偶（1.70m–1.73m）：左前坐箱喝水(#3282EB)、炉前弯腰盛汤(#F07D23)、桌远端仰饮(#37C35A)、桌中段喝汤(#EBC32D)、右前站立举杯(#E13C3C)
    - shot1：西南高机位斜向俯拍全景（共享5名猎人与校准木屋尺度）
    - shot2：东侧平视纵深正打壁炉与五猎人群像（1:1 匹配实拍比例校准参考图）
    """
    def _mk_char(
        cid: str,
        name: str,
        color: str,
        pos: List[float],
        rot_y: float,
        scale_val: float,
        targets: Dict[str, List[float]],
        pose_name: str = "custom",
    ) -> Dict[str, Any]:
        full_targets = {
            "pelvis": [0.0, 0.86, 0.0],
            "chest": [0.0, 1.32, 0.0],
            "head_target": [0.0, 1.56, 0.70],
            "hand_L": [0.22, 0.78, 0.01],
            "hand_R": [-0.22, 0.78, 0.01],
            "foot_L": [0.09, 0.04, 0.0],
            "foot_R": [-0.09, 0.04, -0.0],
            "knee_L_pole": [0.09, 0.46, 0.40],
            "knee_R_pole": [-0.09, 0.46, 0.40],
            "elbow_L_pole": [0.32, 1.05, -0.20],
            "elbow_R_pole": [-0.32, 1.05, -0.20],
            "shoulder_L_offset": [0.0, 0.0, 0.0],
            "shoulder_R_offset": [0.0, 0.0, 0.0],
        }
        full_targets.update(copy.deepcopy(targets))
        kf0 = {
            "t": 0.0,
            "pos": list(pos),
            "rot": [0.0, float(rot_y), 0.0],
            "rot_y": float(rot_y),
            "targets": copy.deepcopy(full_targets),
        }
        kf1 = copy.deepcopy(kf0)
        kf1["t"] = 6.0
        return {
            "id": cid,
            "name": name,
            "type": "character",
            "body_style": "mannequin",
            "pos": list(pos),
            "rot": [0.0, float(rot_y), 0.0],
            "scale": [scale_val, scale_val, scale_val],
            "torso_twist": 0.0,
            "color": color,
            "keep_color": True,
            "protected": True,
            "pose": pose_name,
            "foot_ground_clamp": False,
            "targets": full_targets,
            "pins": {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False},
            "keyframes": [kf0, kf1],
        }

    # 1. 地面与纵向宽木板拼缝层（校准后净尺寸 6.48m × 4.18m，与四面墙体严丝合缝）
    floor_parts: List[Dict[str, Any]] = [
        {"id": "fc_fl_base", "name": "木屋地面基板 (6.48m×4.18m)", "shape": "box", "pos": [0.0, -0.04, 0.0], "size": [6.48, 0.08, 4.18], "rot": [0, 0, 0], "color": "#BFC3C8"},
    ]
    plank_zs = [-1.80, -1.44, -1.08, -0.72, -0.36, 0.00, 0.36, 0.72, 1.08, 1.44, 1.80]
    for idx_p, zp in enumerate(plank_zs):
        col_p = "#B6BAC0" if idx_p % 2 == 0 else "#C3C7CC"
        floor_parts.append({
            "id": f"fc_fl_plank_{idx_p+1}",
            "name": f"宽木地板条_{idx_p+1}",
            "shape": "box",
            "pos": [0.0, 0.004, zp],
            "size": [6.30, 0.008, 0.34],
            "rot": [0, 0, 0],
            "color": col_p,
        })

    # 2. 北墙·三扇四格木窗墙（内壁 Z = -2.00，中心 Z = -2.09；窗中心 X = -1.45, -0.06, +1.48；窗台高 0.90m，窗上框高 1.82m）
    north_parts: List[Dict[str, Any]] = [
        {"id": "wn_skirt", "name": "北墙窗下墙裙", "shape": "box", "pos": [0.0, 0.45, 0.0], "size": [6.48, 0.90, 0.18], "rot": [0, 0, 0], "color": "#D6D8DC"},
        {"id": "wn_top_beam", "name": "北墙窗上过梁墙", "shape": "box", "pos": [0.0, 2.09, 0.0], "size": [6.48, 0.58, 0.18], "rot": [0, 0, 0], "color": "#D6D8DC"},
        # 四段窗间立墙段 (Y: 0.90..1.80, H=0.90)
        {"id": "wn_pier_1", "name": "北墙西端墙段", "shape": "box", "pos": [-2.54, 1.35, 0.0], "size": [1.40, 0.90, 0.18], "rot": [0, 0, 0], "color": "#DCDEE2"},
        {"id": "wn_pier_2", "name": "左中窗间墙段", "shape": "box", "pos": [-0.755, 1.35, 0.0], "size": [0.63, 0.90, 0.18], "rot": [0, 0, 0], "color": "#DCDEE2"},
        {"id": "wn_pier_3", "name": "中右窗间墙段", "shape": "box", "pos": [0.71, 1.35, 0.0], "size": [0.78, 0.90, 0.18], "rot": [0, 0, 0], "color": "#DCDEE2"},
        {"id": "wn_pier_4", "name": "北墙东端墙段", "shape": "box", "pos": [2.55, 1.35, 0.0], "size": [1.38, 0.90, 0.18], "rot": [0, 0, 0], "color": "#DCDEE2"},
        # 第1、2扇窗间小挂架与铁壶 (X = -0.755，对应参考图第1、2扇窗之间墙面小挂架)
        {"id": "wn_rack_b1", "name": "窗间小挂架上板", "shape": "box", "pos": [-0.755, 1.50, 0.13], "size": [0.26, 0.03, 0.12], "rot": [0, 0, 0], "color": "#C6C9CE"},
        {"id": "wn_rack_b2", "name": "窗间小挂架下板", "shape": "box", "pos": [-0.755, 1.22, 0.13], "size": [0.26, 0.03, 0.12], "rot": [0, 0, 0], "color": "#C6C9CE"},
        {"id": "wn_rack_kettle", "name": "窗间挂架小铁壶", "shape": "cylinder", "pos": [-0.755, 1.31, 0.14], "size": [0.11, 0.13, 0.11], "rot": [0, 0, 0], "color": "#8E939C"},
    ]
    for w_idx, wx in enumerate([-1.45, -0.06, 1.48]):
        north_parts.extend([
            {"id": f"win{w_idx+1}_pane", "name": f"第{w_idx+1}扇深灰夜窗玻璃", "shape": "box", "pos": [wx, 1.35, -0.03], "size": [0.76, 0.88, 0.04], "rot": [0, 0, 0], "color": "#383D46"},
            {"id": f"win{w_idx+1}_sill", "name": f"第{w_idx+1}扇窗台挑板", "shape": "box", "pos": [wx, 0.90, 0.05], "size": [0.92, 0.05, 0.24], "rot": [0, 0, 0], "color": "#EAECEF"},
            {"id": f"win{w_idx+1}_head", "name": f"第{w_idx+1}扇窗上框罩", "shape": "box", "pos": [wx, 1.82, 0.04], "size": [0.94, 0.08, 0.22], "rot": [0, 0, 0], "color": "#CFD2D7"},
            {"id": f"win{w_idx+1}_jamb_L", "name": f"第{w_idx+1}扇窗左外框", "shape": "box", "pos": [wx - 0.41, 1.35, 0.03], "size": [0.08, 0.92, 0.20], "rot": [0, 0, 0], "color": "#CFD2D7"},
            {"id": f"win{w_idx+1}_jamb_R", "name": f"第{w_idx+1}扇窗右外框", "shape": "box", "pos": [wx + 0.41, 1.35, 0.03], "size": [0.08, 0.92, 0.20], "rot": [0, 0, 0], "color": "#CFD2D7"},
            {"id": f"win{w_idx+1}_mul_v", "name": f"第{w_idx+1}扇窗竖中梃", "shape": "box", "pos": [wx, 1.35, 0.04], "size": [0.034, 0.88, 0.08], "rot": [0, 0, 0], "color": "#EAECEF"},
            {"id": f"win{w_idx+1}_mul_h", "name": f"第{w_idx+1}扇窗横中梃", "shape": "box", "pos": [wx, 1.35, 0.04], "size": [0.76, 0.034, 0.08], "rot": [0, 0, 0], "color": "#EAECEF"},
        ])

    # 3. 东墙大型四层储物木货架与木箱罐头（X = +2.80, Z = -0.35，第1镜全景可见）
    shelf_parts: List[Dict[str, Any]] = [
        {"id": "sh_post_fl", "name": "货架前左立柱", "shape": "box", "pos": [-0.24, 1.00, -1.16], "size": [0.07, 2.00, 0.07], "rot": [0, 0, 0], "color": "#C6C9CE"},
        {"id": "sh_post_ml", "name": "货架前中左立柱", "shape": "box", "pos": [-0.24, 1.00, -0.38], "size": [0.07, 2.00, 0.07], "rot": [0, 0, 0], "color": "#C6C9CE"},
        {"id": "sh_post_mr", "name": "货架前中右立柱", "shape": "box", "pos": [-0.24, 1.00, 0.40], "size": [0.07, 2.00, 0.07], "rot": [0, 0, 0], "color": "#C6C9CE"},
        {"id": "sh_post_fr", "name": "货架前右立柱", "shape": "box", "pos": [-0.24, 1.00, 1.16], "size": [0.07, 2.00, 0.07], "rot": [0, 0, 0], "color": "#C6C9CE"},
        {"id": "sh_post_bl", "name": "货架后左立柱", "shape": "box", "pos": [0.22, 1.00, -1.16], "size": [0.07, 2.00, 0.07], "rot": [0, 0, 0], "color": "#BFC2C7"},
        {"id": "sh_post_br", "name": "货架后右立柱", "shape": "box", "pos": [0.22, 1.00, 1.16], "size": [0.07, 2.00, 0.07], "rot": [0, 0, 0], "color": "#BFC2C7"},
        {"id": "sh_tier_1", "name": "货架一层底搁板", "shape": "box", "pos": [0.0, 0.40, 0.0], "size": [0.52, 0.04, 2.40], "rot": [0, 0, 0], "color": "#D4D7DC"},
        {"id": "sh_tier_2", "name": "货架二层中搁板", "shape": "box", "pos": [0.0, 0.90, 0.0], "size": [0.52, 0.04, 2.40], "rot": [0, 0, 0], "color": "#D4D7DC"},
        {"id": "sh_tier_3", "name": "货架三层高搁板", "shape": "box", "pos": [0.0, 1.38, 0.0], "size": [0.52, 0.04, 2.40], "rot": [0, 0, 0], "color": "#D4D7DC"},
        {"id": "sh_tier_4", "name": "货架四层顶搁板", "shape": "box", "pos": [0.0, 1.84, 0.0], "size": [0.52, 0.04, 2.40], "rot": [0, 0, 0], "color": "#D4D7DC"},
        {"id": "sh_crate_1", "name": "底层左木箱", "shape": "box", "pos": [-0.02, 0.59, -0.78], "size": [0.44, 0.32, 0.62], "rot": [0, 3, 0], "color": "#DCDEE2"},
        {"id": "sh_crate_2", "name": "底层中木箱", "shape": "box", "pos": [-0.04, 0.60, 0.02], "size": [0.46, 0.34, 0.66], "rot": [0, -2, 0], "color": "#E2E4E8"},
        {"id": "sh_crate_3", "name": "底层右木箱", "shape": "box", "pos": [-0.02, 0.58, 0.78], "size": [0.44, 0.30, 0.60], "rot": [0, 4, 0], "color": "#D8DADF"},
        {"id": "sh_floor_box1", "name": "架下地面左木箱", "shape": "box", "pos": [-0.08, 0.17, -0.46], "size": [0.44, 0.32, 0.54], "rot": [0, -8, 0], "color": "#CFD2D7"},
        {"id": "sh_floor_box2", "name": "架下地面右木箱", "shape": "box", "pos": [-0.06, 0.16, 0.62], "size": [0.46, 0.30, 0.58], "rot": [0, 6, 0], "color": "#CFD2D7"},
        {"id": "sh_sack_1", "name": "二层左麻袋体块", "shape": "box", "pos": [0.02, 1.07, -0.82], "size": [0.38, 0.28, 0.48], "rot": [4, 8, -3], "color": "#E6E8EC"},
        {"id": "sh_box_2m", "name": "二层中储物盒", "shape": "box", "pos": [-0.02, 1.05, -0.04], "size": [0.40, 0.24, 0.52], "rot": [0, -4, 0], "color": "#DCDFE4"},
        {"id": "sh_box_2r", "name": "二层右木盒", "shape": "box", "pos": [0.0, 1.07, 0.74], "size": [0.40, 0.28, 0.54], "rot": [0, 2, 0], "color": "#E2E4E8"},
        {"id": "sh_can_1", "name": "三层搪瓷罐_1", "shape": "cylinder", "pos": [-0.08, 1.50, -0.84], "size": [0.14, 0.20, 0.14], "rot": [0, 0, 0], "color": "#EAECEF"},
        {"id": "sh_can_2", "name": "三层搪瓷罐_2", "shape": "cylinder", "pos": [-0.06, 1.49, -0.58], "size": [0.13, 0.18, 0.13], "rot": [0, 0, 0], "color": "#DFE2E6"},
        {"id": "sh_can_3", "name": "三层储物罐_3", "shape": "cylinder", "pos": [-0.08, 1.51, 0.24], "size": [0.15, 0.22, 0.15], "rot": [0, 0, 0], "color": "#EAECEF"},
        {"id": "sh_can_4", "name": "三层搪瓷杯_4", "shape": "cylinder", "pos": [-0.10, 1.47, 0.56], "size": [0.11, 0.14, 0.11], "rot": [0, 0, 0], "color": "#F0F2F5"},
        {"id": "sh_can_5", "name": "三层搪瓷杯_5", "shape": "cylinder", "pos": [-0.10, 1.47, 0.74], "size": [0.11, 0.14, 0.11], "rot": [0, 0, 0], "color": "#F0F2F5"},
        {"id": "sh_top_crate", "name": "顶层木框架箱", "shape": "box", "pos": [0.0, 1.98, 0.16], "size": [0.42, 0.24, 0.62], "rot": [0, 0, 0], "color": "#D6D9DE"},
    ]

    objects: List[Dict[str, Any]] = [
        {
            "id": "fc_floor",
            "name": "木屋宽板地面 (校准 6.30m×4.00m)",
            "type": "group",
            "arch_role": "floor",
            "pos": [0.0, 0.0, 0.0],
            "rot": [0.0, 0.0, 0.0],
            "size": [6.48, 0.08, 4.18],
            "color": "#BFC3C8",
            "parts": floor_parts,
        },
        {
            "id": "fc_wall_north",
            "name": "北墙·三扇四格木窗墙 (窗顶高1.82m)",
            "type": "group",
            "arch_role": "wall_north",
            "pos": [0.0, 0.0, -2.09],
            "rot": [0.0, 0.0, 0.0],
            "size": [6.48, 2.38, 0.18],
            "color": "#DCDEE2",
            "parts": north_parts,
        },
        {
            "id": "fc_wall_west",
            "name": "西墙·壁炉背景山墙 (墙高2.38m)",
            "type": "group",
            "arch_role": "wall_west",
            "pos": [-3.24, 0.0, 0.0],
            "rot": [0.0, 0.0, 0.0],
            "size": [0.18, 2.38, 4.18],
            "color": "#D6D8DC",
            "parts": [
                {"id": "ww_body", "name": "西墙主体木墙", "shape": "box", "pos": [0.0, 1.19, 0.0], "size": [0.18, 2.38, 4.18], "rot": [0, 0, 0], "color": "#D6D8DC"},
                {"id": "ww_niche_s1", "name": "壁炉右壁架上板", "shape": "box", "pos": [0.24, 1.46, -1.28], "size": [0.32, 0.04, 0.88], "rot": [0, 0, 0], "color": "#C6C9CE"},
                {"id": "ww_niche_s2", "name": "壁炉右壁架中板", "shape": "box", "pos": [0.24, 1.08, -1.28], "size": [0.32, 0.04, 0.88], "rot": [0, 0, 0], "color": "#C6C9CE"},
            ],
        },
        {
            "id": "fc_fireplace",
            "name": "西墙壁炉、挑梁台与炉前汤锅墩 (组合)",
            "type": "group",
            "pos": [-2.72, 0.0, 0.06],
            "rot": [0.0, 0.0, 0.0],
            "size": [1.02, 2.38, 2.12],
            "color": "#CFD2D7",
            "locked": False,
            "parts": [
                {"id": "fp_hearth_base", "name": "壁炉抬高石砌炉台", "shape": "box", "pos": [0.10, 0.11, 0.0], "size": [0.98, 0.22, 2.08], "rot": [0, 0, 0], "color": "#C4C7CC"},
                {"id": "fp_hearth_curb", "name": "炉台前缘厚石条", "shape": "box", "pos": [0.54, 0.23, 0.0], "size": [0.12, 0.04, 2.06], "rot": [0, 0, 0], "color": "#D4D7DC"},
                {"id": "fp_pillar_L", "name": "壁炉北侧砖柱", "shape": "box", "pos": [-0.04, 0.74, -0.74], "size": [0.64, 1.06, 0.50], "rot": [0, 0, 0], "color": "#CFD2D7"},
                {"id": "fp_pillar_R", "name": "壁炉南侧砖柱", "shape": "box", "pos": [-0.04, 0.74, 0.74], "size": [0.64, 1.06, 0.50], "rot": [0, 0, 0], "color": "#CFD2D7"},
                {"id": "fp_back_wall", "name": "炉膛深色内壁", "shape": "box", "pos": [-0.30, 0.72, 0.0], "size": [0.14, 1.02, 0.98], "rot": [0, 0, 0], "color": "#34373D"},
                {"id": "fp_lintel", "name": "壁炉石过梁", "shape": "box", "pos": [-0.02, 1.34, 0.0], "size": [0.68, 0.24, 1.98], "rot": [0, 0, 0], "color": "#C6C9CE"},
                {"id": "fp_mantel", "name": "壁炉粗木挑梁台 (顶高1.58m)", "shape": "box", "pos": [0.06, 1.525, 0.0], "size": [0.80, 0.11, 2.12], "rot": [0, 0, 0], "color": "#E6E8EC"},
                {"id": "fp_chimney", "name": "壁炉上部烟道墙", "shape": "box", "pos": [-0.10, 1.98, 0.0], "size": [0.58, 0.80, 1.78], "rot": [0, 0, 0], "color": "#CFD2D7"},
                {"id": "fp_log_1", "name": "炉膛底劈柴_1", "shape": "cylinder", "pos": [0.20, 0.29, -0.04], "size": [0.12, 0.60, 0.12], "rot": [78, 18, 12], "color": "#585B62"},
                {"id": "fp_log_2", "name": "炉膛斜劈柴_2", "shape": "cylinder", "pos": [0.24, 0.33, 0.10], "size": [0.11, 0.56, 0.11], "rot": [72, -24, -15], "color": "#646870"},
                {"id": "fp_fire_core", "name": "炉火暖橙外焰", "shape": "cone", "pos": [0.22, 0.55, 0.02], "size": [0.48, 0.60, 0.44], "rot": [0, 15, -6], "color": "#EB8238", "keep_color": True},
                {"id": "fp_fire_inner", "name": "炉火明黄内焰", "shape": "cone", "pos": [0.28, 0.47, -0.01], "size": [0.32, 0.44, 0.30], "rot": [0, -10, 8], "color": "#F6B94D", "keep_color": True},
                {"id": "fp_stove_block", "name": "炉前左矮石墩", "shape": "box", "pos": [0.56, 0.17, 0.48], "size": [0.46, 0.34, 0.46], "rot": [0, 0, 0], "color": "#BFC3C9"},
                {"id": "fp_soup_pot", "name": "炉前黑铁大汤锅", "shape": "cylinder", "pos": [0.56, 0.46, 0.48], "size": [0.34, 0.24, 0.34], "rot": [0, 0, 0], "color": "#454950"},
                {"id": "fp_lantern", "name": "壁炉上台煤油灯", "shape": "cylinder", "pos": [0.18, 1.70, 0.52], "size": [0.14, 0.24, 0.14], "rot": [0, 0, 0], "color": "#E2E4E8"},
                {"id": "fp_bottle_top", "name": "壁炉上台玻璃瓶", "shape": "cylinder", "pos": [0.16, 1.68, -0.38], "size": [0.09, 0.20, 0.09], "rot": [0, 0, 0], "color": "#D8DADF"},
            ],
        },
        {
            "id": "fc_bucket_hearth",
            "name": "壁炉北角落地木水桶",
            "type": "cylinder",
            "pos": [-2.12, 0.19, -1.24],
            "rot": [0.0, 15.0, 0.0],
            "size": [0.32, 0.38, 0.32],
            "color": "#D8DADF",
            "locked": False,
        },
        {
            "id": "fc_window_bench",
            "name": "北墙西中段靠墙矮木箱柜组 (组合)",
            "type": "group",
            "pos": [-0.42, 0.0, -1.72],
            "rot": [0.0, 0.0, 0.0],
            "size": [2.40, 0.62, 0.48],
            "color": "#DCDFE4",
            "locked": False,
            "parts": [
                {"id": "wb_box", "name": "远窗下木长榻箱", "shape": "box", "pos": [-1.06, 0.21, 0.0], "size": [1.04, 0.42, 0.46], "rot": [0, 0, 0], "color": "#D2D5DA"},
                {"id": "wb_top", "name": "远窗下木长榻盖", "shape": "box", "pos": [-1.06, 0.43, 0.01], "size": [1.08, 0.04, 0.48], "rot": [0, 0, 0], "color": "#E4E6EA"},
                {"id": "wb_mid_chest", "name": "中窗下储物木箱柜", "shape": "box", "pos": [0.58, 0.27, 0.02], "size": [0.96, 0.54, 0.48], "rot": [0, 0, 0], "color": "#CFD2D7"},
                {"id": "wb_mid_lid", "name": "中窗下木箱柜顶板", "shape": "box", "pos": [0.58, 0.56, 0.03], "size": [1.00, 0.04, 0.50], "rot": [0, 0, 0], "color": "#E2E4E8"},
            ],
        },
        {
            "id": "fc_window_worktable",
            "name": "北墙东窗下工作木桌与瓶罐 (组合)",
            "type": "group",
            "pos": [1.68, 0.0, -1.70],
            "rot": [0.0, 0.0, 0.0],
            "size": [1.14, 0.88, 0.54],
            "color": "#E2E4E8",
            "locked": False,
            "parts": [
                {"id": "wt_top", "name": "工作桌厚木面板", "shape": "box", "pos": [0.0, 0.72, 0.0], "size": [1.14, 0.05, 0.54], "rot": [0, 0, 0], "color": "#EAECEF"},
                {"id": "wt_apron", "name": "桌下抽屉围框", "shape": "box", "pos": [0.0, 0.62, 0.0], "size": [1.04, 0.14, 0.46], "rot": [0, 0, 0], "color": "#D4D7DC"},
                {"id": "wt_shelf", "name": "桌下底层搁板", "shape": "box", "pos": [0.0, 0.22, 0.0], "size": [1.02, 0.04, 0.44], "rot": [0, 0, 0], "color": "#CFD2D7"},
                {"id": "wt_leg_fl", "name": "左前桌腿", "shape": "box", "pos": [-0.49, 0.36, 0.21], "size": [0.06, 0.72, 0.06], "rot": [0, 0, 0], "color": "#C8CBD0"},
                {"id": "wt_leg_fr", "name": "右前桌腿", "shape": "box", "pos": [0.49, 0.36, 0.21], "size": [0.06, 0.72, 0.06], "rot": [0, 0, 0], "color": "#C8CBD0"},
                {"id": "wt_leg_bl", "name": "左后桌腿", "shape": "box", "pos": [-0.49, 0.36, -0.21], "size": [0.06, 0.72, 0.06], "rot": [0, 0, 0], "color": "#C8CBD0"},
                {"id": "wt_leg_br", "name": "右后桌腿", "shape": "box", "pos": [0.49, 0.36, -0.21], "size": [0.06, 0.72, 0.06], "rot": [0, 0, 0], "color": "#C8CBD0"},
                {"id": "wt_bowl", "name": "桌面大木碗", "shape": "cylinder", "pos": [0.22, 0.79, 0.04], "size": [0.20, 0.08, 0.20], "rot": [0, 0, 0], "color": "#F2F4F7"},
            ],
        },
        {
            "id": "fc_dining_table",
            "name": "偏北长木餐桌与桌面餐具 (2.26m×0.80m×0.75m)",
            "type": "group",
            "pos": [-0.62, 0.0, -0.74],
            "rot": [0.0, 0.0, 0.0],
            "size": [2.26, 0.92, 0.80],
            "color": "#E6E8EC",
            "locked": False,
            "parts": [
                {"id": "dt_top", "name": "长餐桌厚木桌面", "shape": "box", "pos": [0.0, 0.73, 0.0], "size": [2.26, 0.05, 0.80], "rot": [0, 0, 0], "color": "#EAECEF"},
                {"id": "dt_apron_f", "name": "餐桌南裙板", "shape": "box", "pos": [0.0, 0.66, 0.33], "size": [2.02, 0.09, 0.04], "rot": [0, 0, 0], "color": "#D6D9DE"},
                {"id": "dt_apron_b", "name": "餐桌北裙板", "shape": "box", "pos": [0.0, 0.66, -0.33], "size": [2.02, 0.09, 0.04], "rot": [0, 0, 0], "color": "#D6D9DE"},
                {"id": "dt_leg_fl", "name": "餐桌西前腿", "shape": "box", "pos": [-0.98, 0.36, 0.32], "size": [0.075, 0.72, 0.075], "rot": [0, 0, 0], "color": "#CFD2D7"},
                {"id": "dt_leg_fr", "name": "餐桌东前腿", "shape": "box", "pos": [0.98, 0.36, 0.32], "size": [0.075, 0.72, 0.075], "rot": [0, 0, 0], "color": "#CFD2D7"},
                {"id": "dt_leg_bl", "name": "餐桌西后腿", "shape": "box", "pos": [-0.98, 0.36, -0.32], "size": [0.075, 0.72, 0.075], "rot": [0, 0, 0], "color": "#CFD2D7"},
                {"id": "dt_leg_br", "name": "餐桌东后腿", "shape": "box", "pos": [0.98, 0.36, -0.32], "size": [0.075, 0.72, 0.075], "rot": [0, 0, 0], "color": "#CFD2D7"},
                {"id": "dt_rail_L", "name": "餐桌西端下横撑", "shape": "box", "pos": [-0.98, 0.18, 0.0], "size": [0.06, 0.05, 0.64], "rot": [0, 0, 0], "color": "#C6C9CE"},
                {"id": "dt_rail_R", "name": "餐桌东端下横撑", "shape": "box", "pos": [0.98, 0.18, 0.0], "size": [0.06, 0.05, 0.64], "rot": [0, 0, 0], "color": "#C6C9CE"},
                {"id": "dt_bowl_h3", "name": "远端猎人3餐盘", "shape": "cylinder", "pos": [-0.84, 0.77, -0.06], "size": [0.22, 0.04, 0.22], "rot": [0, 0, 0], "color": "#F2F4F7"},
                {"id": "dt_plate_f1", "name": "中远浅餐盘组", "shape": "cylinder", "pos": [-0.38, 0.78, 0.16], "size": [0.24, 0.05, 0.24], "rot": [0, 0, 0], "color": "#F5F6F8"},
                {"id": "dt_bowl_h4", "name": "中段猎人4汤碗", "shape": "cylinder", "pos": [-0.10, 0.79, -0.16], "size": [0.19, 0.07, 0.19], "rot": [0, 0, 0], "color": "#F2F4F7"},
                {"id": "dt_bowl_mid", "name": "桌面中左小汤碗", "shape": "cylinder", "pos": [0.08, 0.785, 0.18], "size": [0.16, 0.06, 0.16], "rot": [0, 0, 0], "color": "#F2F4F7"},
                {"id": "dt_mug_mid", "name": "桌面近中搪瓷杯", "shape": "cylinder", "pos": [0.30, 0.80, 0.12], "size": [0.085, 0.095, 0.085], "rot": [0, 0, 0], "color": "#F5F6F8"},
                {"id": "dt_bread_board", "name": "近端木砧板", "shape": "box", "pos": [0.68, 0.765, 0.06], "size": [0.30, 0.025, 0.22], "rot": [0, -12, 0], "color": "#DCDFE4"},
                {"id": "dt_bread_loaf", "name": "近端黑麦圆面包", "shape": "box", "pos": [0.68, 0.815, 0.06], "size": [0.19, 0.085, 0.15], "rot": [0, -12, 0], "color": "#62666E"},
                {"id": "dt_bowl_fgR", "name": "近端右侧大陶碗", "shape": "cylinder", "pos": [0.84, 0.79, -0.18], "size": [0.20, 0.075, 0.20], "rot": [0, 0, 0], "color": "#F2F4F7"},
            ],
        },
        {
            "id": "fc_benches",
            "name": "餐桌两侧长条木凳组 (高0.44m)",
            "type": "group",
            "pos": [-0.62, 0.0, -0.74],
            "rot": [0.0, 0.0, 0.0],
            "size": [2.46, 0.45, 1.48],
            "color": "#DCDFE4",
            "locked": False,
            "parts": [
                {"id": "bn_fl_top", "name": "南侧远段长凳座板", "shape": "box", "pos": [-0.62, 0.43, 0.64], "size": [1.02, 0.045, 0.28], "rot": [0, 0, 0], "color": "#E6E8EC"},
                {"id": "bn_fl_legL", "name": "南侧远段长凳西腿", "shape": "box", "pos": [-0.96, 0.21, 0.64], "size": [0.05, 0.42, 0.24], "rot": [0, 0, -5], "color": "#C8CBD0"},
                {"id": "bn_fl_legR", "name": "南侧远段长凳东腿", "shape": "box", "pos": [-0.28, 0.21, 0.64], "size": [0.05, 0.42, 0.24], "rot": [0, 0, 5], "color": "#C8CBD0"},
                {"id": "bn_fr_top", "name": "南侧近段长凳座板", "shape": "box", "pos": [0.58, 0.43, 0.66], "size": [1.06, 0.045, 0.28], "rot": [0, 0, 0], "color": "#E2E4E8"},
                {"id": "bn_fr_legL", "name": "南侧近段长凳西腿", "shape": "box", "pos": [0.22, 0.21, 0.66], "size": [0.05, 0.42, 0.24], "rot": [0, 0, -5], "color": "#C8CBD0"},
                {"id": "bn_fr_legR", "name": "南侧近段长凳东腿", "shape": "box", "pos": [0.94, 0.21, 0.66], "size": [0.05, 0.42, 0.24], "rot": [0, 0, 5], "color": "#C8CBD0"},
                {"id": "bn_bl_top", "name": "北侧远段长凳座板", "shape": "box", "pos": [-0.58, 0.43, -0.52], "size": [1.18, 0.045, 0.30], "rot": [0, 0, 0], "color": "#D8DADF"},
                {"id": "bn_bl_legL", "name": "北侧远段长凳西腿", "shape": "box", "pos": [-1.00, 0.21, -0.52], "size": [0.05, 0.42, 0.24], "rot": [0, 0, 0], "color": "#C4C7CC"},
                {"id": "bn_bl_legR", "name": "北侧远段长凳东腿", "shape": "box", "pos": [-0.14, 0.21, -0.52], "size": [0.05, 0.42, 0.24], "rot": [0, 0, 0], "color": "#C4C7CC"},
                {"id": "bn_br_top", "name": "北侧近段长凳座板", "shape": "box", "pos": [0.66, 0.43, -0.56], "size": [0.98, 0.045, 0.28], "rot": [0, 0, 0], "color": "#D8DADF"},
                {"id": "bn_br_legL", "name": "北侧近段长凳西腿", "shape": "box", "pos": [0.32, 0.21, -0.56], "size": [0.05, 0.42, 0.24], "rot": [0, 0, 0], "color": "#C4C7CC"},
                {"id": "bn_br_legR", "name": "北侧近段长凳东腿", "shape": "box", "pos": [1.00, 0.21, -0.56], "size": [0.05, 0.42, 0.24], "rot": [0, 0, 0], "color": "#C4C7CC"},
            ],
        },
        {
            "id": "fc_south_seat_crate",
            "name": "南侧坐人矮木箱组 (高0.38m，双镜常显)",
            "type": "group",
            "pos": [0.38, 0.0, 1.52],
            "rot": [0.0, 0.0, 0.0],
            "size": [1.14, 0.40, 0.48],
            "color": "#D4D7DC",
            "locked": False,
            "parts": [
                {"id": "sc_main", "name": "猎人1坐姿木箱", "shape": "box", "pos": [0.0, 0.19, 0.0], "size": [0.50, 0.38, 0.44], "rot": [0, -4, 0], "color": "#D4D7DC"},
                {"id": "sc_fg_small", "name": "左前景角矮木箱", "shape": "box", "pos": [0.62, 0.14, 0.08], "size": [0.46, 0.28, 0.38], "rot": [0, 6, 0], "color": "#CFD2D7"},
            ],
        },
        {
            "id": "fc_east_shelf",
            "name": "东墙大型多层木储物架与木箱 (组合)",
            "type": "group",
            "pos": [2.80, 0.0, -0.35],
            "rot": [0.0, 0.0, 0.0],
            "size": [0.56, 2.12, 2.44],
            "color": "#D4D7DC",
            "locked": False,
            "visible": shot_mode != "shot2",
            "parts": shelf_parts,
        },
        {
            "id": "fc_wall_east",
            "name": "东墙·右侧储物墙与暗门洞",
            "type": "group",
            "arch_role": "wall_east",
            "pos": [3.24, 0.0, 0.0],
            "rot": [0.0, 0.0, 0.0],
            "size": [0.18, 2.38, 4.18],
            "color": "#D6D8DC",
            "visible": shot_mode != "shot2",
            "parts": [
                {"id": "we_north_seg", "name": "东墙北中段墙面", "shape": "box", "pos": [0.0, 1.19, -0.52], "size": [0.18, 2.38, 3.14], "rot": [0, 0, 0], "color": "#D6D8DC"},
                {"id": "we_door_lintel", "name": "东墙南端门上过梁", "shape": "box", "pos": [0.0, 2.14, 1.56], "size": [0.18, 0.48, 1.02], "rot": [0, 0, 0], "color": "#D6D8DC"},
                {"id": "we_door_frame_L", "name": "东墙门框立柱", "shape": "box", "pos": [-0.04, 0.95, 1.06], "size": [0.16, 1.90, 0.10], "rot": [0, 0, 0], "color": "#C6C9CE"},
                {"id": "we_door_dark", "name": "东墙暗门洞背景", "shape": "box", "pos": [0.08, 0.95, 1.56], "size": [0.06, 1.90, 0.96], "rot": [0, 0, 0], "color": "#34383E"},
            ],
        },
        {
            "id": "fc_wall_south",
            "name": "南墙·木门(净高1.88m)、挂衣架与西段壁架 (组合)",
            "type": "group",
            "arch_role": "wall_south",
            "pos": [0.0, 0.0, 2.09],
            "rot": [0.0, 0.0, 0.0],
            "size": [6.48, 2.38, 0.52],
            "color": "#D8DADE",
            "visible": shot_mode == "shot2",
            "parts": [
                {"id": "ws_main", "name": "南墙木墙主体", "shape": "box", "pos": [0.0, 1.19, 0.0], "size": [6.48, 2.38, 0.18], "rot": [0, 0, 0], "color": "#D8DADE"},
                {"id": "ws_door_panel", "name": "南墙内镶板木门 (净高1.88m)", "shape": "box", "pos": [-0.38, 0.94, -0.10], "size": [0.84, 1.88, 0.06], "rot": [0, 0, 0], "color": "#E2E4E8"},
                {"id": "ws_door_jambL", "name": "南门西框柱", "shape": "box", "pos": [-0.82, 0.98, -0.11], "size": [0.09, 1.96, 0.10], "rot": [0, 0, 0], "color": "#C8CBD0"},
                {"id": "ws_door_jambR", "name": "南门东框柱", "shape": "box", "pos": [0.06, 0.98, -0.11], "size": [0.09, 1.96, 0.10], "rot": [0, 0, 0], "color": "#C8CBD0"},
                {"id": "ws_door_lintel", "name": "南门上框梁", "shape": "box", "pos": [-0.38, 1.94, -0.11], "size": [0.98, 0.09, 0.10], "rot": [0, 0, 0], "color": "#C8CBD0"},
                {"id": "ws_coat_shelf", "name": "左前景墙面挂衣横架", "shape": "box", "pos": [0.88, 1.68, -0.20], "size": [1.22, 0.04, 0.24], "rot": [0, 0, 0], "color": "#D2D5DA"},
                {"id": "ws_coat_1", "name": "悬挂厚重猎人长外套_1", "shape": "box", "pos": [1.18, 1.16, -0.24], "size": [0.40, 0.98, 0.22], "rot": [4, -10, 6], "color": "#52565E"},
                {"id": "ws_coat_2", "name": "悬挂猎人披风外套_2", "shape": "box", "pos": [0.80, 1.20, -0.22], "size": [0.30, 0.88, 0.18], "rot": [-3, 8, -5], "color": "#646870"},
                {"id": "ws_rope_coil", "name": "门东侧悬挂盘绳", "shape": "cylinder", "pos": [0.40, 1.32, -0.16], "size": [0.22, 0.08, 0.22], "rot": [90, 0, 0], "color": "#BFC3C9"},
                {"id": "ws_bunk_board", "name": "南墙西段三角撑木搁板", "shape": "box", "pos": [-1.42, 1.36, -0.28], "size": [1.06, 0.055, 0.42], "rot": [0, 0, 0], "color": "#DCDFE4"},
                {"id": "ws_bunk_pillow", "name": "搁板斜靠麻布枕包", "shape": "box", "pos": [-1.10, 1.50, -0.26], "size": [0.34, 0.20, 0.28], "rot": [10, -14, -18], "color": "#ECEEF2"},
                {"id": "ws_bunk_braceL", "name": "搁板斜撑木条_1", "shape": "box", "pos": [-1.00, 1.14, -0.22], "size": [0.05, 0.44, 0.05], "rot": [36, 0, 0], "color": "#C4C7CC"},
                {"id": "ws_bunk_braceR", "name": "搁板斜撑木条_2", "shape": "box", "pos": [-1.76, 1.14, -0.22], "size": [0.05, 0.44, 0.05], "rot": [36, 0, 0], "color": "#C4C7CC"},
                {"id": "ws_low_bench", "name": "南墙西段靠墙矮木箱凳", "shape": "box", "pos": [-1.42, 0.22, -0.30], "size": [1.04, 0.44, 0.38], "rot": [0, 0, 0], "color": "#D4D7DC"},
            ],
        },
        {
            "id": "fc_roof_beams",
            "name": "木屋屋顶与粗木横梁系统 (梁底净高2.10m)",
            "type": "group",
            "arch_role": "roof",
            "pos": [0.0, 2.28, 0.0],
            "rot": [0.0, 0.0, 0.0],
            "size": [6.48, 0.26, 4.18],
            "color": "#CFD2D7",
            "parts": [
                {"id": "rf_ceiling_slab", "name": "屋顶木板顶棚 (高2.38m)", "shape": "box", "pos": [0.0, 0.14, 0.0], "size": [6.48, 0.08, 4.18], "rot": [0, 0, 0], "color": "#D4D7DC"},
                {"id": "rf_cross_1", "name": "西段粗木横梁_1", "shape": "box", "pos": [-1.92, -0.09, 0.0], "size": [0.20, 0.18, 4.00], "rot": [0, 0, 0], "color": "#9498A1"},
                {"id": "rf_cross_2", "name": "中西段粗木横梁_2", "shape": "box", "pos": [-0.62, -0.09, 0.0], "size": [0.20, 0.18, 4.00], "rot": [0, 0, 0], "color": "#9498A1"},
                {"id": "rf_cross_3", "name": "前景构图粗木横梁_3", "shape": "box", "pos": [0.66, -0.09, 0.0], "size": [0.22, 0.18, 4.00], "rot": [0, 0, 0], "color": "#888C95"},
                {"id": "rf_cross_4", "name": "东端粗木横梁_4", "shape": "box", "pos": [1.96, -0.09, 0.0], "size": [0.20, 0.18, 4.00], "rot": [0, 0, 0], "color": "#9498A1"},
            ],
        },
        {
            "id": "fc_cup_h1",
            "name": "猎人1手持水杯",
            "type": "cylinder",
            "pos": [0.55, 0.96, 1.26],
            "rot": [18.0, 32.0, -12.0],
            "size": [0.082, 0.105, 0.082],
            "color": "#E6E8EC",
            "locked": False,
        },
        {
            "id": "fc_ladle_h2",
            "name": "猎人2手持长柄汤勺",
            "type": "group",
            "pos": [-2.10, 0.78, 0.59],
            "rot": [0.0, 0.0, 0.0],
            "size": [0.16, 0.36, 0.16],
            "color": "#DCDFE4",
            "locked": False,
            "parts": [
                {"id": "hp_ladle_shaft", "name": "汤勺长柄", "shape": "cylinder", "pos": [0.01, 0.02, 0.01], "size": [0.026, 0.42, 0.026], "rot": [34, -34, 18], "color": "#DCDFE4"},
                {"id": "hp_ladle_bowl", "name": "汤勺头", "shape": "cylinder", "pos": [-0.06, -0.14, -0.05], "size": [0.09, 0.05, 0.09], "rot": [0, 0, 0], "color": "#D4D7DC"},
            ],
        },
        {
            "id": "fc_bottle_h3",
            "name": "猎人3手持酒瓶",
            "type": "cylinder",
            "pos": [-1.64, 1.15, -0.96],
            "rot": [62.0, -36.0, 10.0],
            "size": [0.075, 0.21, 0.075],
            "color": "#585C64",
            "locked": False,
        },
        {
            "id": "fc_mug_h5",
            "name": "猎人5手持大酒杯",
            "type": "cylinder",
            "pos": [1.38, 1.49, -1.09],
            "rot": [-22.0, 52.0, 16.0],
            "size": [0.115, 0.16, 0.115],
            "color": "#E6E8EC",
            "locked": False,
        },
        _mk_char(
            cid="char_fc_hunter1",
            name="猎人1·左前坐箱喝水",
            color="#3282EB",
            pos=[0.38, 0.0, 1.46],
            rot_y=148.0,
            scale_val=1.08,
            pose_name="sitting_drink",
            targets={
                "pelvis": [0.0, 0.40, -0.04],
                "chest": [0.0, 0.83, 0.11],
                "head_target": [0.04, 1.00, 0.65],
                "hand_R": [-0.04, 0.89, 0.24],
                "hand_L": [0.16, 0.50, 0.36],
                "foot_L": [0.15, 0.04, 0.40],
                "foot_R": [-0.14, 0.04, 0.36],
                "knee_L_pole": [0.18, 0.48, 0.58],
                "knee_R_pole": [-0.16, 0.48, 0.55],
                "elbow_R_pole": [-0.30, 0.66, 0.18],
                "elbow_L_pole": [0.34, 0.68, 0.06],
            },
        ),
        _mk_char(
            cid="char_fc_hunter2",
            name="猎人2·炉前弯腰盛汤",
            color="#F07D23",
            pos=[-1.88, 0.0, 0.94],
            rot_y=-146.0,
            scale_val=1.075,
            pose_name="standing_ladle",
            targets={
                "pelvis": [0.0, 0.83, -0.03],
                "chest": [0.0, 1.25, 0.14],
                "head_target": [0.0, 0.72, 0.56],
                "hand_R": [-0.08, 0.88, 0.35],
                "hand_L": [0.06, 0.78, 0.41],
                "foot_L": [0.10, 0.04, 0.05],
                "foot_R": [-0.10, 0.04, -0.05],
                "knee_L_pole": [0.11, 0.45, 0.40],
                "knee_R_pole": [-0.11, 0.45, 0.36],
                "elbow_R_pole": [-0.30, 1.02, 0.06],
                "elbow_L_pole": [0.28, 0.98, 0.08],
            },
        ),
        _mk_char(
            cid="char_fc_hunter3",
            name="猎人3·桌远端仰饮",
            color="#37C35A",
            pos=[-1.50, 0.0, -1.16],
            rot_y=-36.0,
            scale_val=1.08,
            pose_name="sitting_bottle",
            targets={
                "pelvis": [0.0, 0.45, -0.02],
                "chest": [0.0, 0.90, -0.01],
                "head_target": [0.0, 1.24, 0.58],
                "hand_R": [-0.05, 1.05, 0.20],
                "hand_L": [0.18, 0.72, 0.32],
                "foot_L": [0.13, 0.04, 0.34],
                "foot_R": [-0.13, 0.04, 0.32],
                "knee_L_pole": [0.15, 0.48, 0.54],
                "knee_R_pole": [-0.15, 0.48, 0.52],
                "elbow_R_pole": [-0.34, 0.86, 0.14],
                "elbow_L_pole": [0.34, 0.74, 0.05],
            },
        ),
        _mk_char(
            cid="char_fc_hunter4",
            name="猎人4·桌中段喝汤",
            color="#EBC32D",
            pos=[-0.72, 0.0, -1.30],
            rot_y=-12.0,
            scale_val=1.08,
            pose_name="sitting_soup",
            targets={
                "pelvis": [0.0, 0.45, -0.03],
                "chest": [0.0, 0.87, 0.12],
                "head_target": [-0.02, 0.74, 0.48],
                "hand_R": [-0.08, 0.82, 0.30],
                "hand_L": [0.15, 0.72, 0.32],
                "foot_L": [0.13, 0.04, 0.34],
                "foot_R": [-0.13, 0.04, 0.34],
                "knee_L_pole": [0.15, 0.48, 0.54],
                "knee_R_pole": [-0.15, 0.48, 0.54],
                "elbow_R_pole": [-0.30, 0.74, 0.10],
                "elbow_L_pole": [0.32, 0.74, 0.06],
            },
        ),
        _mk_char(
            cid="char_fc_hunter5",
            name="猎人5·右前站立举杯",
            color="#E13C3C",
            pos=[1.22, 0.0, -1.29],
            rot_y=52.0,
            scale_val=1.095,
            pose_name="standing_mug",
            targets={
                "pelvis": [0.0, 0.86, 0.0],
                "chest": [0.0, 1.32, 0.02],
                "head_target": [-0.04, 1.44, 0.68],
                "hand_R": [-0.06, 1.37, 0.22],
                "hand_L": [0.22, 0.78, 0.04],
                "foot_L": [0.10, 0.04, 0.02],
                "foot_R": [-0.10, 0.04, -0.02],
                "knee_L_pole": [0.10, 0.46, 0.40],
                "knee_R_pole": [-0.10, 0.46, 0.40],
                "elbow_R_pole": [-0.34, 1.12, 0.16],
                "elbow_L_pole": [0.32, 1.05, -0.14],
            },
        ),
    ]

    if shot_mode == "shot2":
        focal_mm = 24
        fov_deg = round(math.degrees(2.0 * math.atan(24.0 / (2.0 * focal_mm))), 2)
        cam_pos = [2.88, 1.23, -0.02]
        cam_target = [-2.85, 0.93, 0.02]
    else:
        focal_mm = 20
        fov_deg = round(math.degrees(2.0 * math.atan(24.0 / (2.0 * focal_mm))), 2)
        cam_pos = [-2.52, 1.94, 1.92]
        cam_target = [0.22, 0.72, -0.44]

    export_plan = {
        "confirmed": True,
        "quality_tier": "fast",
        "key_compositions": [
            {
                "t": 0.0,
                "label": "五人偶比例校准主构图（正中炉火·左坐箱与盛汤·右餐桌双人与前景七分景举杯）" if shot_mode == "shot2" else "西南高机位木屋全景（共享5人偶与校准空间）",
                "keep": "保留5名纯色猎人与SU木屋建筑家具，以1.70m–1.73m人体比例锁定梁底2.10m、壁炉台1.58m、门高1.88m、窗顶1.82m与4.00m×6.30m净空",
            }
        ],
        "area_budget_accepted": True,
        "confirmed_at": "2026-11-12T14:50:00+08:00",
    }

    return {
        "duration": 6.0,
        "fps": 24,
        "render_mode": "clay_only",
        "visual_style": "sketchup_massing",
        "export_plan": export_plan,
        "environment": {
            "sky": "#AEB5BF",
            "ground": "#9FA6B0",
            "fog": False,
            "show_grid": False,
        },
        "camera": {
            "pos": list(cam_pos),
            "target": list(cam_target),
            "fov": fov_deg,
            "focal_mm": focal_mm,
            "aspect_ratio": "2.39:1",
            "export_width": 1280,
            "export_height": 536,
        },
        "camera_keyframes": [
            {"t": 0.0, "pos": list(cam_pos), "target": list(cam_target), "fov": fov_deg, "focal_mm": focal_mm},
            {"t": 6.0, "pos": list(cam_pos), "target": list(cam_target), "fov": fov_deg, "focal_mm": focal_mm},
        ],
        "objects": objects,
    }

@sparse_generated_scene
def build_default_sitting_scene() -> Dict[str, Any]:
    """默认启动场景：加载第 64 场猎人木屋现场（SketchUp 灰白体块＋黑色细边线风格）"""
    return build_scene64_cabin_scene("shot1")


def scene_dict_to_python_code(scene: Dict[str, Any]) -> str:
    """将当前 3D 场景与魔法人形师 IK 控制球坐标逆向生成直观的 Python 脚本（100% 无损保留姿态关键帧、组合体、锁定与画幅设置）"""
    dur = round(float(scene.get("duration", 10.0)), 1)
    fps = int(scene.get("fps", 24) or 24)
    render_mode = scene.get("render_mode", "clay_only")
    visual_style = scene.get("visual_style", "sketchup_massing")
    aspect = scene.get("camera", {}).get("aspect_ratio", "2.39:1")
    lines = [
        "# 魔法人形师 x Blender 风格 Python 导演脚本",
        "stage.clear()",
        f"stage.duration = {dur}",
        f"stage.fps = {fps}",
        f"stage.render_mode = {render_mode!r}",
        f"stage.visual_style = {visual_style!r}",
    ]
    if scene.get("environment"):
        lines.append(f"stage.environment = {scene['environment']!r}")
    if scene.get("ref_video"):
        lines.append(f"stage.ref_video = {scene['ref_video']!r}")
    if scene.get("repro_state"):
        lines.append(f"stage.repro_state = {scene['repro_state']!r}")
    if scene.get("point_cloud_helper"):
        lines.append(f"stage.point_cloud_helper = {scene['point_cloud_helper']!r}")
    if scene.get("constraints"):
        lines.append(f"stage.constraints = {scene['constraints']!r}")
    if scene.get("camera_speed_curve"):
        lines.append(f"stage.camera_speed_curve = {scene['camera_speed_curve']!r}")
    if scene.get("camera_rig"):
        lines.append(f"stage.camera_rig = {scene['camera_rig']!r}")
    if scene.get("meters_per_unit") is not None and abs(float(scene.get("meters_per_unit", 1.0)) - 1.0) > 1e-6:
        lines.append(f"stage.meters_per_unit = {float(scene['meters_per_unit'])!r}")
    if scene.get("allow_snow_jeep_4mps") is not None:
        lines.append(f"stage.allow_snow_jeep_4mps = {bool(scene['allow_snow_jeep_4mps'])!r}")
    if FORCE_HIDE_EDGES or scene.get("hide_edges") is not None:
        lines.append(f"stage.hide_edges = {(True if FORCE_HIDE_EDGES else bool(scene['hide_edges']))!r}")
    if scene.get("contact_shadows") is not None:
        lines.append(f"stage.contact_shadows = {bool(scene['contact_shadows'])!r}")
    lines.extend([
        "",
        "# 1. 建造场景几何体 (长方体 box / 圆锥体 cone / 三角柱 prism / 圆柱 cylinder / 墙体 wall / 组合体 group)",
    ])
    objs = scene.get("objects", [])
    prim_idx = 0
    for obj in objs:
        otype = obj.get("type", "box")
        if otype == "character":
            continue
        name = obj.get("name", otype)
        oid = obj.get("id")
        pos = [round(float(x), 3) for x in obj.get("pos", [0, 0, 0])]
        rot = [round(float(x), 1) for x in obj.get("rot", [0, 0, 0])]
        color = obj.get("color", "#D4D6D9")
        size = [round(float(x), 3) for x in obj.get("size", [0.5, 0.5, 0.5])]
        obj_kfs = obj.get("keyframes") or []
        lhs = ""
        if obj_kfs or obj.get("parent_binding") or obj.get("bound_char_id"):
            prim_idx += 1
            var_p = f"obj_{prim_idx}"
            lhs = f"{var_p} = "

        extra_kwargs = []
        if oid:
            extra_kwargs.append(f"obj_id={oid!r}")
        for k_ex in (
            "arch_role",
            "locked",
            "hidden",
            "protected",
            "keep_color",
            "unlit",
            "flame_phase",
            "ground_decal",
            "repro_source",
            "group_base_size",
            "align_tangent_y",
            "motion_type",
            "speed_exception",
            "allow_snow_4mps",
            "motion_source",
            "parent_binding",
            "bound_char_id",
            "follow_bound_char",
            "grip_offset_pos",
            "grip_offset_rot",
            "parent_id",
            "layout_ref",
        ):
            if obj.get(k_ex) is not None:
                extra_kwargs.append(f"{k_ex}={obj[k_ex]!r}")
        extra_str = (", " + ", ".join(extra_kwargs)) if extra_kwargs else ""

        if otype == "scene_group":
            lines.append(f'{lhs}stage.add_scene_group({name!r}, pos={pos}, rot={rot}, size={size}, color={color!r}{extra_str})')
        elif otype in ("box", "cone", "prism", "cylinder", "sphere", "plane", "uaz469"):
            fn = {
                "box": "add_box",
                "cone": "add_cone",
                "prism": "add_prism",
                "cylinder": "add_cylinder",
                "sphere": "add_sphere",
                "plane": "add_plane",
                "uaz469": "add_uaz469",
            }[otype]
            if any(abs(r) > 0.05 for r in rot):
                lines.append(f'{lhs}stage.{fn}({name!r}, pos={pos}, size={size}, color={color!r}, rot={rot}{extra_str})')
            else:
                lines.append(f'{lhs}stage.{fn}({name!r}, pos={pos}, size={size}, color={color!r}{extra_str})')
        elif otype == "wall":
            openings = obj.get("openings", [])
            lines.append(f'{lhs}stage.add_wall({name!r}, pos={pos}, size={size}, color={color!r}, rot={rot}, openings={openings!r}{extra_str})')
        elif otype == "group":
            parts = obj.get("parts", [])
            lines.append(f'{lhs}stage.add_group({name!r}, pos={pos}, size={size}, color={color!r}, rot={rot}, parts={parts!r}{extra_str})')

        if obj_kfs and lhs:
            for okf in obj_kfs:
                kt = round(float(okf.get("t", 0.0)), 3)
                kpos = [round(float(x), 3) for x in okf.get("pos", pos)]
                if isinstance(okf.get("rot"), (list, tuple)) and len(okf["rot"]) >= 3:
                    krot = [round(float(r), 2) for r in okf["rot"][:3]]
                else:
                    ry_v = round(float(okf.get("rot_y", rot[1])), 2)
                    krot = [rot[0], ry_v, rot[2]]
                okf_args = [f"t={kt}", f"pos={kpos}", f"rot={krot}"]
                if isinstance(okf.get("size"), (list, tuple)) and len(okf["size"]) >= 3:
                    ksize = [round(float(s), 3) for s in okf["size"][:3]]
                    if ksize != size:
                        okf_args.append(f"size={ksize}")
                for ik_name in ("interp", "interp_pos", "interp_rot", "interp_size"):
                    if okf.get(ik_name):
                        okf_args.append(f"{ik_name}={okf[ik_name]!r}")
                if okf.get("stop"):
                    okf_args.append("stop=True")
                if okf.get("reverse"):
                    okf_args.append("reverse=True")
                if okf.get("cut"):
                    okf_args.append("cut=True")
                if okf.get("teleport"):
                    okf_args.append("teleport=True")
                if okf.get("coord_space"):
                    okf_args.append(f"coord_space={okf['coord_space']!r}")
                lines.append(f"{var_p}.add_keyframe({', '.join(okf_args)})")

    lines.append("")
    lines.append("# 2. 角色 (球头人偶 / 关节人偶) 与走位/姿态关键帧")
    char_idx = 0
    for obj in objs:
        if obj.get("type") == "character":
            char_idx += 1
            var = f"char_{char_idx}"
            cid = obj.get("id", f"char_{char_idx}")
            name = obj.get("name", f"人物_0{char_idx}")
            pos = [round(float(x), 3) for x in obj.get("pos", [0, 0, 0])]
            rot_y = round(float(obj.get("rot", [0, 0, 0])[1]), 1)
            pose_name = obj.get("pose_name", "sitting")
            color = obj.get("color", "#e2e8f0")
            body_style = obj.get("body_style", "mannequin")
            c_extra = [f"char_id={cid!r}"]
            for k_ex in ("pawn_visor", "foot_ground_clamp", "locked", "hidden", "protected", "garment", "ground_squat", "idle_breath_phase", "pawn_smooth_walk", "natural_limb_motion", "mounted_vehicle_id", "follow_mounted_vehicle", "seat_offset", "seat_rot_offset", "repro_source", "motion_type", "speed_exception", "motion_source", "parent_id", "layout_ref"):
                if obj.get(k_ex) is not None:
                    c_extra.append(f"{k_ex}={obj[k_ex]!r}")
            c_extra_str = ", " + ", ".join(c_extra)
            if body_style == "pawn":
                lines.append(f'{var} = stage.add_character({name!r}, pos={pos}, rot_y={rot_y}, pose={pose_name!r}, color={color!r}, body_style="pawn"{c_extra_str})')
            else:
                lines.append(f'{var} = stage.add_character({name!r}, pos={pos}, rot_y={rot_y}, pose={pose_name!r}, color={color!r}{c_extra_str})')
            lines.append(f"{var}.scale = {list(_get_char_scale_xyz(obj))!r}  # X宽、Y高、Z深的缩放，完整保留精度")
            if obj.get("pins"):
                lines.append(f"{var}.pins = {dict(obj['pins'])!r}")
            t = obj.get("targets", {})
            lines.append(f"{var}.ik(")
            for k in ["pelvis", "chest", "head_target", "shoulder_L_offset", "shoulder_R_offset", "hand_L", "hand_R", "foot_L", "foot_R", "knee_L_pole", "knee_R_pole", "elbow_L_pole", "elbow_R_pole"]:
                if k in t:
                    coords = [round(float(x), 3 if k.endswith("_offset") else 3) for x in t[k]]
                    lines.append(f"    {k}={coords},")
            lines.append(")")
            if abs(float(obj.get("torso_twist", 0.0) or 0.0)) > 1e-6:
                lines.append(f"{var}.torso_twist = {round(float(obj['torso_twist']), 1)}  # 胸廓相对骨盆转身角度 (°)")
            for ckf in obj.get("keyframes", []):
                kt = round(float(ckf.get("t", 0.0)), 2)
                kpos = [round(float(x), 3) for x in ckf.get("pos", pos)]
                kroty = round(float(ckf.get("rot_y") if ckf.get("rot_y") is not None else (ckf.get("rot", [0, rot_y, 0])[1] if isinstance(ckf.get("rot"), (list, tuple)) else rot_y)), 1)
                kf_args = [f"t={kt}", f"pos={kpos}", f"rot_y={kroty}"]
                if ckf.get("look_at") is not None:
                    kf_args.append(f"look_at={[round(float(x), 3) for x in ckf['look_at']]}")
                if ckf.get("targets"):
                    clean_targets = {
                        tk: [round(float(v), 3) for v in tv]
                        for tk, tv in ckf["targets"].items()
                        if isinstance(tv, (list, tuple)) and len(tv) >= 3
                    }
                    kf_args.append(f"targets={clean_targets!r}")
                if ckf.get("torso_twist") is not None:
                    kf_args.append(f"torso_twist={round(float(ckf['torso_twist']), 1)}")
                for ik_name in ("interp", "interp_pos", "interp_rot", "interp_pose"):
                    if ckf.get(ik_name) and ckf.get(ik_name) != "linear":
                        kf_args.append(f"{ik_name}={ckf[ik_name]!r}")
                if ckf.get("walk_steps"):
                    kf_args.append(f"walk_steps={int(ckf['walk_steps'])}")
                if ckf.get("lead_foot") in ("L", "R"):
                    kf_args.append(f"lead_foot={ckf['lead_foot']!r}")
                for flag_k in ("stop", "cut", "teleport"):
                    if ckf.get(flag_k):
                        kf_args.append(f"{flag_k}=True")
                lines.append(f"{var}.add_keyframe({', '.join(kf_args)})")

    lines.append("")
    lines.append(f'# 3. 摄影机画幅比 ({aspect}) 与镜头焦距运镜')
    cam = scene.get("camera", {})
    cpos0 = [round(float(x), 3) for x in cam.get("pos", [2.2, 1.5, 2.75])]
    ctar0 = [round(float(x), 3) for x in cam.get("target", [0, 0.72, 0.1])]
    f_mm0 = round(float(cam.get("focal_mm") or fov_to_focal_mm(float(cam.get("fov", 37.8)))), 1)
    cam_extra = ""
    if cam.get("export_width") and cam.get("export_height"):
        cam_extra = f", export_width={int(cam['export_width'])}, export_height={int(cam['export_height'])}"
    if scene.get("camera", {}).get("roll_deg") is not None and abs(float(scene["camera"].get("roll_deg", 0.0))) > 1e-4:
        cam_extra += f", roll_deg={round(float(scene['camera']['roll_deg']), 3)}"
    lines.append(f'stage.set_camera(pos={cpos0}, target={ctar0}, focal_mm={f_mm0}, aspect_ratio={aspect!r}{cam_extra})')
    kfs = scene.get("camera_keyframes", [])
    for kf in kfs:
        t = round(float(kf.get("t", 0.0)), 2)
        cpos = [round(float(x), 3) for x in kf.get("pos", cpos0)]
        ctar = [round(float(x), 3) for x in kf.get("target", ctar0)]
        f_mm = round(float(kf.get("focal_mm") or fov_to_focal_mm(float(kf.get("fov", 37.8)))), 1)
        ckf_args = [f"t={t}", f"pos={cpos}", f"target={ctar}", f"focal_mm={f_mm}"]
        if kf.get("fov") is not None:
            ckf_args.append(f"fov={round(float(kf['fov']), 2)}")
        if kf.get("roll_deg") is not None and abs(float(kf.get("roll_deg", 0.0))) > 1e-4:
            ckf_args.append(f"roll_deg={round(float(kf['roll_deg']), 3)}")
        for ik_name in ("interp", "interp_pos", "interp_target", "interp_fov"):
            if kf.get(ik_name) and kf.get(ik_name) != "linear":
                ckf_args.append(f"{ik_name}={kf[ik_name]!r}")
        if kf.get("stop") is True:
            ckf_args.append("stop=True")
        elif kf.get("stop") is False:
            ckf_args.append("stop=False")
        if kf.get("has_pos") is False:
            ckf_args.append("has_pos=False")
        if kf.get("has_fov") is False:
            ckf_args.append("has_fov=False")
        lines.append(f"stage.add_camera_keyframe({', '.join(ckf_args)})")

    lines.append("")
    lines.append("# 取消下方注释可在运行脚本时直接用 Python 极速导出 MP4 视频：")
    lines.append(f'# stage.export_video("forest_reveal.mp4", duration={dur}, fps={fps}, mode="clay_only")')
    return "\n".join(lines)


# ============================================================================
# 4. 纯 Python 极速 3D 软光栅渲染器 (魔法人形师素体白模 + 彩色几何体 + OpenPose)
# ============================================================================

def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    h = hex_str.lstrip("#")
    if len(h) == 6:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (180, 190, 205)


def euler_xyz_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rx @ Ry @ Rz


def make_transform(pos: List[float], rot_deg: List[float], scale: float = 1.0) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = euler_xyz_to_matrix(rot_deg[0], rot_deg[1], rot_deg[2]) * scale
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ones = np.ones((len(pts), 1), dtype=np.float64)
    hom = np.hstack([pts, ones])
    return (hom @ T.T)[:, :3]


def make_limb_prism_between(
    p0: np.ndarray,
    p1: np.ndarray,
    r0: float,
    r1: float,
    sides: int = 8,
) -> List[np.ndarray]:
    """在两个 3D 关节点 p0 -> p1 之间生成魔法人形师风格锥形圆润肢体棱柱面片"""
    diff = p1 - p0
    length = float(np.linalg.norm(diff))
    if length < 1e-5:
        return []
    axis = diff / length
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(np.dot(axis, up)) > 0.95:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = _normalize(np.cross(axis, up))
    v = _normalize(np.cross(axis, u))

    ring0, ring1 = [], []
    for i in range(sides):
        ang = (2.0 * math.pi * i) / sides
        dir_vec = math.cos(ang) * u + math.sin(ang) * v
        ring0.append(p0 + dir_vec * r0)
        ring1.append(p1 + dir_vec * r1)

    faces = []
    for i in range(sides):
        ni = (i + 1) % sides
        faces.append(np.array([ring0[i], ring0[ni], ring1[ni], ring1[i]], dtype=np.float64))
    faces.append(np.array(ring1, dtype=np.float64))
    faces.append(np.array(ring0[::-1], dtype=np.float64))
    return faces


def make_oriented_box_between(
    p0: np.ndarray,
    p1: np.ndarray,
    side_axis: np.ndarray,
    width: float,
    depth: float,
) -> List[np.ndarray]:
    """在 p0 -> p1 之间生成带朝向的魔法人形师胸腔/骨盆块面片"""
    up_axis = _normalize(p1 - p0)
    right_axis = _normalize(side_axis - np.dot(side_axis, up_axis) * up_axis)
    fwd_axis = _normalize(np.cross(right_axis, up_axis))
    hw, hd = width * 0.5, depth * 0.5

    def corner(p_base: np.ndarray, sx: float, sz: float) -> np.ndarray:
        return p_base + right_axis * (sx * hw) + fwd_axis * (sz * hd)

    v = [
        corner(p0, -1, -1),
        corner(p0, 1, -1),
        corner(p1, 1, -1),
        corner(p1, -1, -1),
        corner(p0, -1, 1),
        corner(p0, 1, 1),
        corner(p1, 1, 1),
        corner(p1, -1, 1),
    ]
    v_arr = np.asarray(v, dtype=np.float64)
    idx = [
        [4, 5, 6, 7],
        [1, 0, 3, 2],
        [5, 1, 2, 6],
        [0, 4, 7, 3],
        [7, 6, 2, 3],
        [0, 1, 5, 4],
    ]
    return [v_arr[i] for i in idx]


def make_box_faces_local(sx: float, sy: float, sz: float) -> List[np.ndarray]:
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    v = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ], dtype=np.float64)
    idx = [[4, 5, 6, 7], [1, 0, 3, 2], [5, 1, 2, 6], [0, 4, 7, 3], [7, 6, 2, 3], [0, 1, 5, 4]]
    return [v[i] for i in idx]


def make_cone_faces_local(sx: float, sy: float, sz: float, segments: int = 12) -> List[np.ndarray]:
    rx, hy, rz = sx * 0.5, sy * 0.5, sz * 0.5
    apex = np.array([0.0, hy, 0.0], dtype=np.float64)
    ring = [np.array([rx * math.cos(2 * math.pi * i / segments), -hy, rz * math.sin(2 * math.pi * i / segments)]) for i in range(segments)]
    faces = [np.array([ring[i], ring[(i + 1) % segments], apex]) for i in range(segments)]
    faces.append(np.array(ring[::-1]))
    return faces


def make_prism_faces_local(sx: float, sy: float, sz: float) -> List[np.ndarray]:
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    f0, f1, f2 = np.array([-hx, -hy, hz]), np.array([hx, -hy, hz]), np.array([0.0, hy, hz])
    b0, b1, b2 = np.array([-hx, -hy, -hz]), np.array([hx, -hy, -hz]), np.array([0.0, hy, -hz])
    return [
        np.array([f0, f1, f2]),
        np.array([b1, b0, b2]),
        np.array([f1, b1, b2, f2]),
        np.array([b0, f0, f2, b2]),
        np.array([b0, b1, f1, f0]),
    ]


def make_cylinder_faces_local(sx: float, sy: float, sz: float, segments: int = 12) -> List[np.ndarray]:
    rx, hy, rz = sx * 0.5, sy * 0.5, sz * 0.5
    top_r = [np.array([rx * math.cos(2 * math.pi * i / segments), hy, rz * math.sin(2 * math.pi * i / segments)]) for i in range(segments)]
    bot_r = [np.array([rx * math.cos(2 * math.pi * i / segments), -hy, rz * math.sin(2 * math.pi * i / segments)]) for i in range(segments)]
    faces = [np.array([bot_r[i], bot_r[(i + 1) % segments], top_r[(i + 1) % segments], top_r[i]]) for i in range(segments)]
    faces.append(np.array(top_r))
    faces.append(np.array(bot_r[::-1]))
    return faces


def make_uaz469_polygons_world(
    pos: List[float],
    rot: List[float],
    size: List[float],
    base_rgb: Tuple[int, int, int],
    sketchup_style: bool = True,
    edge_out: Optional[List[Tuple[np.ndarray, np.ndarray, float]]] = None,
    cam_pos: Optional[np.ndarray] = None,
    door_FR_deg: float = 0.0,
    door_FL_deg: float = 0.0,
) -> List[Tuple[np.ndarray, Tuple[int, int, int]]]:
    """生成 UAZ-469 越野车多边形面片（按参考图：四门封闭车身 + 帆布软顶 + 横格栅圆大灯 + 方向盘；
    与 Three.js buildUaz469VisualGroup 比例严格对齐；车窗玻璃在软光栅中留空，以便看见车内乘员；
    支持 SketchUp 灰白体块风格与结构细边线输出）"""
    sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
    T = make_transform(pos, rot, 1.0)
    if sketchup_style:
        lum = int(0.299 * base_rgb[0] + 0.587 * base_rgb[1] + 0.114 * base_rgb[2])
        if max(base_rgb) - min(base_rgb) <= 18:
            body_rgb = base_rgb
        else:
            v_b = max(198, min(226, lum + 52))
            body_rgb = (v_b - 2, v_b, v_b + 2)
        tarp_rgb = (max(176, body_rgb[0] - 18), max(178, body_rgb[1] - 18), max(182, body_rgb[2] - 16))
        dark_rgb = (98, 102, 108)
        tire_rgb = (76, 80, 86)
        rim_rgb = (138, 142, 148)
        seat_rgb = (164, 168, 174)
        lamp_rgb = (238, 240, 242)
        amber_rgb = (196, 198, 202)
        tail_rgb = (156, 160, 166)
    else:
        body_rgb = base_rgb
        tarp_rgb = (74, 81, 65)
        dark_rgb = (31, 34, 36)
        tire_rgb = (26, 27, 29)
        rim_rgb = (52, 49, 43)
        seat_rgb = (46, 42, 37)
        lamp_rgb = (233, 228, 207)
        amber_rgb = (208, 138, 46)
        tail_rgb = (181, 50, 42)

    out_polys: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []

    def _emit_box_edges(w_faces: List[np.ndarray]):
        if edge_out is None:
            return
        for wf in w_faces:
            n_pts = len(wf)
            if n_pts < 3:
                continue
            if cam_pos is not None:
                f_norm = np.cross(wf[1] - wf[0], wf[2] - wf[0])
                f_center = np.mean(wf, axis=0)
                if float(np.dot(f_norm, cam_pos - f_center)) < -0.05:
                    continue
            for i_e in range(n_pts):
                edge_out.append((wf[i_e], wf[(i_e + 1) % n_pts], -0.008))

    def _apply_door_rot(local_pts: np.ndarray, door_side: Optional[float]) -> np.ndarray:
        if door_side is None:
            return local_pts
        ang_deg = float(door_FR_deg if door_side < 0 else door_FL_deg)
        if abs(ang_deg) < 1e-4:
            return local_pts
        th = (-float(door_side)) * math.radians(ang_deg)
        c_d, s_d = math.cos(th), math.sin(th)
        R_d = np.array([[c_d, 0.0, s_d], [0.0, 1.0, 0.0], [-s_d, 0.0, c_d]], dtype=np.float64)
        hp = np.array([float(door_side) * 0.44 * sx, 0.0, 0.135 * sz], dtype=np.float64)
        return (local_pts - hp) @ R_d.T + hp

    def B(w: float, h: float, l: float, x: float, y: float, z: float, col: Tuple[int, int, int],
          rx: float = 0.0, pivot: Optional[Tuple[float, float, float]] = None, with_edges: bool = True,
          door_side: Optional[float] = None):
        """以包围盒比例添加长方体：w/x 乘 sx，h/y 乘 sy，l/z 乘 sz；rx 为绕 X 轴后倾角（弧度），绕 pivot 旋转"""
        center = np.array([x * sx, y * sy, z * sz], dtype=np.float64)
        c, s_ = math.cos(rx), math.sin(rx)
        R = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]], dtype=np.float64)
        pv = np.array([pivot[0] * sx, pivot[1] * sy, pivot[2] * sz], dtype=np.float64) if pivot else center
        w_faces: List[np.ndarray] = []
        for lf in make_box_faces_local(w * sx, h * sy, l * sz):
            local = lf + center
            if rx:
                local = (local - pv) @ R.T + pv
            local = _apply_door_rot(local, door_side)
            w_poly = transform_points(T, local)
            w_faces.append(w_poly)
            for sub_p in _subdivide_polygon_world(w_poly, max_edge=0.70):
                out_polys.append((sub_p, col))
        if with_edges and max(w * sx, h * sy, l * sz) >= 0.08:
            _emit_box_edges(w_faces)

    def cyl(p0_local, p1_local, r: float, col: Tuple[int, int, int], sides: int = 14):
        p0 = transform_points(T, np.array([p0_local], dtype=np.float64))[0]
        p1 = transform_points(T, np.array([p1_local], dtype=np.float64))[0]
        prism_faces = make_limb_prism_between(p0, p1, r, r, sides=sides)
        for f in prism_faces:
            out_polys.append((f, col))
        if edge_out is not None and len(prism_faces) >= 2:
            # 两个端面圆环轮廓边
            for cap in prism_faces[-2:]:
                nc = len(cap)
                for ic in range(nc):
                    edge_out.append((cap[ic], cap[(ic + 1) % nc], -0.008))

    def TRAP(w: float, y0: float, y1: float, zf0: float, zf1: float, zb0: float, zb1: float, cx: float, col: Tuple[int, int, int],
             door_side: Optional[float] = None):
        w_faces: List[np.ndarray] = []
        for lf in make_box_faces_local(1.0, 1.0, 1.0):
            local = np.empty_like(lf)
            for idx, (vx, vy, vz) in enumerate(lf):
                local[idx, 0] = (cx + vx * w) * sx
                local[idx, 1] = (y1 if vy > 0 else y0) * sy
                zf = zf1 if vy > 0 else zf0
                zb = zb1 if vy > 0 else zb0
                local[idx, 2] = (zf if vz > 0 else zb) * sz
            local = _apply_door_rot(local, door_side)
            w_poly = transform_points(T, local)
            w_faces.append(w_poly)
            for sub_p in _subdivide_polygon_world(w_poly, max_edge=0.70):
                out_polys.append((sub_p, col))
        _emit_box_edges(w_faces)

    SIDES = (-1.0, 1.0)

    # 1. 底盘双纵梁
    for s in SIDES:
        B(0.05, 0.06, 0.90, s * 0.28, -0.30, 0.0, dark_rgb)

    # 2. 车厢：地板、防火墙/仪表台、车门下半部、后轮上方侧板、尾门下半部
    B(0.84, 0.03, 0.585, 0.0, -0.215, -0.1525, dark_rgb)
    B(0.84, 0.31, 0.02, 0.0, -0.045, 0.15, dark_rgb)
    B(0.84, 0.05, 0.04, 0.0, 0.08, 0.13, dark_rgb)
    for s in SIDES:
        B(0.04, 0.40, 0.135, s * 0.43, -0.08, -0.1225, body_rgb)
        B(0.04, 0.40, 0.070, s * 0.43, -0.08, 0.170, body_rgb)
        B(0.04, 0.40, 0.190, s * 0.43, -0.08, 0.040, body_rgb, door_side=s)
        B(0.04, 0.21, 0.25, s * 0.43, 0.015, -0.315, body_rgb)
        B(0.04, 0.12, 0.06, s * 0.43, -0.15, -0.445, body_rgb)
        for z in (0.135, -0.275):
            B(0.004, 0.39, 0.004, s * 0.4515, -0.075, z, dark_rgb, with_edges=False)
        B(0.004, 0.39, 0.004, s * 0.4515, -0.075, -0.053, dark_rgb, with_edges=False, door_side=s)
        B(0.01, 0.012, 0.025, s * 0.453, 0.07, -0.035, dark_rgb, with_edges=False, door_side=s)
        B(0.01, 0.012, 0.025, s * 0.453, 0.07, -0.250, dark_rgb, with_edges=False)
    B(0.90, 0.355, 0.02, 0.0, -0.0825, -0.455, body_rgb)
    B(0.86, 0.04, 0.03, 0.0, 0.10, 0.145, body_rgb)

    # 3. 车头：发动机舱 + 平直引擎盖 + 分离式翼子板 + 格栅 + 圆大灯 + 转向灯
    B(0.60, 0.25, 0.315, 0.0, -0.065, 0.3125, body_rgb)
    B(0.62, 0.022, 0.315, 0.0, 0.068, 0.3125, body_rgb)
    for s in SIDES:
        B(0.18, 0.095, 0.315, s * 0.39, -0.0125, 0.3125, body_rgb)
        B(0.18, 0.13, 0.05, s * 0.39, -0.125, 0.445, body_rgb)
    B(0.60, 0.24, 0.02, 0.0, -0.07, 0.465, body_rgb)
    for y in (-0.02, -0.055, -0.09, -0.125):
        B(0.30, 0.016, 0.006, 0.0, y, 0.477, dark_rgb, with_edges=False)
    for s in SIDES:
        cyl([s * 0.235 * sx, -0.035 * sy, 0.470 * sz], [s * 0.235 * sx, -0.035 * sy, 0.485 * sz], 0.055 * sx, lamp_rgb, 14)
        B(0.05, 0.025, 0.006, s * 0.235, -0.12, 0.477, amber_rgb, with_edges=False)

    # 前后保险杠
    B(0.98, 0.05, 0.03, 0.0, -0.24, 0.49, dark_rgb)
    B(0.92, 0.05, 0.025, 0.0, -0.25, -0.485, dark_rgb)

    # 4. 前风挡框（后倾 0.15 rad，玻璃真实留空）
    ws_pivot = (0.0, 0.245, 0.14)
    B(0.88, 0.025, 0.02, 0.0, 0.245 + 0.1125, 0.14, body_rgb, rx=-0.15, pivot=ws_pivot)
    B(0.88, 0.025, 0.02, 0.0, 0.245 - 0.1125, 0.14, body_rgb, rx=-0.15, pivot=ws_pivot)
    for s in SIDES:
        B(0.03, 0.25, 0.02, s * 0.425, 0.245, 0.14, body_rgb, rx=-0.15, pivot=ws_pivot)
    B(0.02, 0.25, 0.02, 0.0, 0.245, 0.14, body_rgb, rx=-0.15, pivot=ws_pivot)
    for s in SIDES:
        B(0.06, 0.065, 0.006, s * 0.545, 0.17, 0.135, dark_rgb)

    # 5. 帆布软顶、四门侧窗与车后箱（前门窗 + B柱安全带 + 后门斜框分体窗 + 后箱梯形帆布侧壁 + 宽幅圆角长方形后窗 + 顶棚横向拱管，所有车窗真实留空）
    B(0.90, 0.04, 0.60, 0.0, 0.46, -0.165, tarp_rgb)
    B(0.80, 0.02, 0.58, 0.0, 0.49, -0.165, tarp_rgb)
    B(0.90, 0.085, 0.03, 0.0, 0.402, 0.122, tarp_rgb)
    for z in (-0.058, -0.220, -0.415):
        B(0.84, 0.016, 0.016, 0.0, 0.436, z, dark_rgb)
        for s in SIDES:
            B(0.016, 0.055, 0.016, s * 0.422, 0.405, z, dark_rgb)
    for s in SIDES:
        B(0.02, 0.07, 0.59, s * 0.44, 0.405, -0.165, tarp_rgb)
        B(0.016, 0.016, 0.56, s * 0.422, 0.372, -0.165, dark_rgb)
        B(0.026, 0.022, 0.163, s * 0.44, 0.360, -0.1365, body_rgb)
        B(0.026, 0.022, 0.177, s * 0.44, 0.360, 0.0335, body_rgb, door_side=s)
        B(0.028, 0.018, 0.233, s * 0.441, 0.125, -0.1715, body_rgb)
        B(0.006, 0.008, 0.233, s * 0.453, 0.116, -0.1715, dark_rgb, with_edges=False)
        B(0.028, 0.018, 0.007, s * 0.441, 0.125, 0.1385, body_rgb)
        B(0.006, 0.008, 0.007, s * 0.453, 0.116, 0.1385, dark_rgb, with_edges=False)
        B(0.028, 0.018, 0.190, s * 0.441, 0.125, 0.040, body_rgb, door_side=s)
        B(0.006, 0.008, 0.190, s * 0.453, 0.116, 0.040, dark_rgb, with_edges=False, door_side=s)
        TRAP(0.026, 0.134, 0.351, 0.142, 0.122, 0.126, 0.106, s * 0.44, body_rgb)
        TRAP(0.022, 0.134, 0.351, 0.134, 0.114, 0.120, 0.100, s * 0.44, body_rgb, door_side=s)
        B(0.028, 0.217, 0.026, s * 0.44, 0.2425, -0.062, body_rgb)
        B(0.005, 0.242, 0.004, s * 0.4525, 0.2425, -0.062, dark_rgb, with_edges=False)
        B(0.012, 0.24, 0.022, s * 0.420, 0.145, -0.058, dark_rgb)
        B(0.022, 0.217, 0.014, s * 0.44, 0.2425, -0.048, body_rgb, door_side=s)
        B(0.024, 0.217, 0.009, s * 0.440, 0.2425, 0.080, dark_rgb, door_side=s)
        B(0.026, 0.217, 0.010, s * 0.440, 0.2425, -0.186, dark_rgb)
        TRAP(0.026, 0.134, 0.351, -0.268, -0.200, -0.288, -0.220, s * 0.44, body_rgb)
        TRAP(0.028, 0.125, 0.365, -0.286, -0.218, -0.290, -0.222, s * 0.442, dark_rgb)
        TRAP(0.024, 0.118, 0.372, -0.288, -0.220, -0.458, -0.458, s * 0.442, tarp_rgb)
        TRAP(0.018, 0.115, 0.372, -0.282, -0.214, -0.295, -0.227, s * 0.422, dark_rgb)
        B(0.018, 0.26, 0.018, s * 0.422, 0.242, -0.442, dark_rgb)
        B(0.024, 0.022, 0.36, s * 0.418, 0.102, -0.265, dark_rgb)
        for bz in (-0.308, -0.352, -0.396, -0.440):
            B(0.008, 0.045, 0.009, s * 0.455, 0.132, bz, dark_rgb, with_edges=False)
    B(0.82, 0.28, 0.010, 0.0, -0.055, -0.446, dark_rgb)
    B(0.84, 0.018, 0.020, 0.0, 0.085, -0.444, dark_rgb)
    for tx in (-0.25, 0.0, 0.25):
        B(0.21, 0.20, 0.006, tx, -0.058, -0.440, body_rgb)
        B(0.18, 0.17, 0.006, tx, -0.058, -0.437, dark_rgb)
    B(0.90, 0.062, 0.022, 0.0, 0.115, -0.458, tarp_rgb)
    B(0.90, 0.092, 0.022, 0.0, 0.400, -0.458, tarp_rgb)
    for s in SIDES:
        B(0.225, 0.216, 0.022, s * 0.3375, 0.250, -0.458, tarp_rgb)
    B(0.46, 0.018, 0.026, 0.0, 0.356, -0.456, dark_rgb)
    B(0.46, 0.018, 0.026, 0.0, 0.144, -0.456, dark_rgb)
    for s in SIDES:
        B(0.020, 0.218, 0.026, s * 0.226, 0.250, -0.456, dark_rgb)
        for cy in (-1.0, 1.0):
            B(0.040, 0.036, 0.024, s * 0.208, 0.250 + cy * 0.088, -0.457, tarp_rgb, with_edges=False)

    # 6. 车内：前排两独立圆弧低靠背座椅 + 方向盘 + 中央低矮储物盒；后箱为宽幅纵向压楞地板 + 左右圆润内轮拱（轮包）
    for s in SIDES:
        z = 0.035
        B(0.30, 0.055, 0.115, s * 0.215, -0.105, z, seat_rgb)
        B(0.28, 0.195, 0.028, s * 0.215, -0.002, z - 0.045, seat_rgb)
        B(0.23, 0.032, 0.026, s * 0.215, 0.102, z - 0.045, seat_rgb)
    # 方向盘（位于左侧驾驶位 x=+0.21*sx, y=0.115*sy, z=0.10*sz，倾斜 -0.55 rad，与 Three.js 对齐）
    st_cx, st_cy, st_cz = 0.21 * sx, 0.115 * sy, 0.10 * sz
    st_r = 0.085 * sx
    st_rx = -0.55
    st_c, st_s = math.cos(st_rx), math.sin(st_rx)
    n_st = 14
    st_pts_local = []
    for i_st in range(n_st):
        ang = 2.0 * math.pi * i_st / n_st
        lx = st_cx + st_r * math.cos(ang)
        ly0 = st_r * math.sin(ang)
        ly = st_cy + ly0 * st_c
        lz = st_cz + ly0 * st_s
        st_pts_local.append([lx, ly, lz])
    for i_st in range(n_st):
        cyl(st_pts_local[i_st], st_pts_local[(i_st + 1) % n_st], 0.011 * sx, dark_rgb, sides=6)
    # 方向盘辐条与转向柱
    cyl([st_cx - st_r * 0.9, st_cy, st_cz], [st_cx + st_r * 0.9, st_cy, st_cz], 0.009 * sx, dark_rgb, sides=6)
    cyl([st_cx, st_cy, st_cz], [st_cx, 0.04 * sy, 0.145 * sz], 0.014 * sx, dark_rgb, sides=8)

    B(0.11, 0.065, 0.10, 0.0, -0.165, -0.01, dark_rgb)
    for rx in (-0.22, -0.176, -0.132, -0.088, -0.044, 0.0, 0.044, 0.088, 0.132, 0.176, 0.22):
        B(0.016, 0.008, 0.42, rx, -0.197, -0.240, rim_rgb, with_edges=False)
    for s in SIDES:
        B(0.175, 0.115, 0.22, s * 0.33, -0.140, -0.320, dark_rgb)
        B(0.155, 0.024, 0.20, s * 0.325, -0.075, -0.320, rim_rgb)

    # 7. 尾灯（移除遮挡后窗的高位外挂备胎）
    for s in SIDES:
        B(0.035, 0.06, 0.01, s * 0.40, -0.12, -0.468, tail_rgb, with_edges=False)

    # 8. 四个越野轮胎 + 钢圈
    r_wheel = 0.2 * sy
    w_wheel = 0.135 * sx
    wy = -0.5 * sy + r_wheel
    for s in SIDES:
        for z in (0.30, -0.29):
            wx = s * 0.405 * sx
            cyl([wx - w_wheel * 0.5, wy, z * sz], [wx + w_wheel * 0.5, wy, z * sz], r_wheel, tire_rgb, 16)
            cyl([wx - w_wheel * 0.51, wy, z * sz], [wx + w_wheel * 0.51, wy, z * sz], r_wheel * 0.6, rim_rgb, 12)
    return out_polys


def compute_camera_basis(pos: List[float], target: List[float], roll_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据机位、焦点与横滚角 roll_deg 计算正交右/上/前单位向量（与前端 Three.js 严格对齐）。"""
    forward = _normalize(np.asarray(target, dtype=np.float64) - np.asarray(pos, dtype=np.float64))
    if np.linalg.norm(forward) < 1e-6:
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(forward, world_up))) > 0.998:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right0 = _normalize(np.cross(forward, world_up))
    if np.linalg.norm(right0) < 1e-6:
        right0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    up0 = _normalize(np.cross(right0, forward))
    theta = math.radians(float(roll_deg or 0.0))
    if abs(theta) > 1e-7:
        ct, st = math.cos(theta), math.sin(theta)
        right = right0 * ct - up0 * st
        up = right0 * st + up0 * ct
    else:
        right, up = right0, up0
    return right, up, forward


class RenderCamera:
    def __init__(self, pos: List[float], target: List[float], fov_deg: float, width: int, height: int, roll_deg: float = 0.0):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64)
        self.fov_deg = float(fov_deg)
        self.roll_deg = float(roll_deg or 0.0)
        self.width = int(width)
        self.height = int(height)
        self.near = 0.15

        right, up, forward = compute_camera_basis(self.pos, self.target, self.roll_deg)
        self.right = right
        self.up = up
        self.forward = forward
        self._basis_mat = np.column_stack([right, up, forward])
        self.focal = (self.height * 0.5) / math.tan(math.radians(self.fov_deg) * 0.5)

    def world_to_cam(self, pts: np.ndarray) -> np.ndarray:
        return (pts - self.pos) @ self._basis_mat

    def project_point(self, pt_world: np.ndarray) -> Optional[Tuple[float, float, float]]:
        rel = pt_world - self.pos
        cz = float(np.dot(rel, self.forward))
        if cz < self.near:
            return None
        cx = float(np.dot(rel, self.right))
        cy = float(np.dot(rel, self.up))
        sx = (self.width * 0.5) + (cx / cz) * self.focal
        sy = (self.height * 0.5) - (cy / cz) * self.focal
        return sx, sy, cz


def clip_polygon_near(pts_cam: np.ndarray, near: float = 0.15) -> np.ndarray:
    n = len(pts_cam)
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)
    if float(pts_cam[:, 2].min()) >= near:
        return pts_cam
    out = []
    for i in range(n):
        cur, prev = pts_cam[i], pts_cam[i - 1]
        cur_in, prev_in = cur[2] >= near, prev[2] >= near
        if cur_in != prev_in:
            dz = cur[2] - prev[2]
            t = (near - prev[2]) / dz if abs(dz) > 1e-9 else 0.0
            inter = prev + t * (cur - prev)
            inter[2] = near
            out.append(inter)
        if cur_in:
            out.append(cur)
    return np.asarray(out, dtype=np.float64) if len(out) >= 3 else np.empty((0, 3), dtype=np.float64)


_SHADE_L_LEN = math.sqrt(0.45 * 0.45 + 0.80 * 0.80 + 0.42 * 0.42)
_SHADE_LX = 0.45 / _SHADE_L_LEN
_SHADE_LY = 0.80 / _SHADE_L_LEN
_SHADE_LZ = 0.42 / _SHADE_L_LEN


def shade_polygon(base_rgb: Tuple[int, int, int], pts_world: np.ndarray) -> Tuple[int, int, int]:
    if len(pts_world) < 3:
        return base_rgb
    p0, p1, p2 = pts_world[0], pts_world[1], pts_world[2]
    ux = float(p1[0] - p0[0])
    uy = float(p1[1] - p0[1])
    uz = float(p1[2] - p0[2])
    vx = float(p2[0] - p0[0])
    vy = float(p2[1] - p0[1])
    vz = float(p2[2] - p0[2])
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
    if n_len < 1e-9:
        return base_rgb
    diff = abs((nx * _SHADE_LX + ny * _SHADE_LY + nz * _SHADE_LZ) / n_len)
    factor = 0.54 + 0.46 * diff
    return (
        min(255, max(0, int(base_rgb[0] * factor))),
        min(255, max(0, int(base_rgb[1] * factor))),
        min(255, max(0, int(base_rgb[2] * factor))),
    )


def eval_easing(alpha: float, mode: str = "linear") -> float:
    """
    统一关键帧插值曲线评估器（前端预览与 Python 导出使用 100% 相同的插值规则）：
    - 'linear' (匀速): s = u，经过中间关键点绝不自动减速停顿
    - 'ease_in' (缓入): s = u^2
    - 'ease_out' (缓出): s = 1 - (1 - u)^2
    - 'ease_in_out' / 'smoothstep' (缓入缓出): s = u^2 * (3 - 2u)
    - 'hold' / 'step' (保持): s = 0 (直到下一关键帧才跳变)
    """
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
    # 默认 linear 匀速，避免连续行走经过中间关键点时自动停顿
    return u


def lerp_angle_deg(a0: float, a1: float, s: float) -> float:
    """最短路径角度插值（避免跨越 ±180° 时反向大转圈）"""
    delta = ((float(a1) - float(a0) + 180.0) % 360.0) - 180.0
    return float(a0) + delta * float(s)


def normalize_cam_interp_mode(mode: Any) -> str:
    """
    归一化摄影机关键帧插值模式名称（前端与 Python 保持完全一致）：
    - 'continuous' (连续平滑 Hermite): 考虑非均匀时间间隔的 C1 保单调三次 Hermite 曲线，经过中间关键帧不停车
    - 'ease_in_out' (逐段缓入缓出 Smooth): 每段独立使用 smoothstep，在每个关键帧处速度降为 0（用于显式停顿）
    - 'linear' (匀速), 'ease_in' (缓入), 'ease_out' (缓出), 'hold' (保持)
    """
    m = str(mode or "linear").lower().strip()
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


CAMERA_RIG_HANDHELD_PRESETS: Dict[str, Dict[str, Any]] = {
    "off": {
        "enabled": False,
        "pos_amp_m": 0.0,
        "rot_amp_deg": 0.0,
        "roll_amp_deg": 0.0,
        "freq_hz": 0.30,
        "step_freq_hz": 1.40,
        "speed_gain": 0.0,
        "aim_lag_sec": 0.0,
        "aim_overshoot": 0.0,
        "settle_ends": True,
    },
    "steadicam": {
        "enabled": True,
        "pos_amp_m": 0.008,
        "rot_amp_deg": 0.18,
        "roll_amp_deg": 0.28,
        "freq_hz": 0.28,
        "step_freq_hz": 1.20,
        "speed_gain": 0.25,
        "aim_lag_sec": 0.24,
        "aim_overshoot": 0.05,
        "settle_ends": True,
    },
    "handheld_breath": {
        "enabled": True,
        "pos_amp_m": 0.014,
        "rot_amp_deg": 0.38,
        "roll_amp_deg": 0.85,
        "freq_hz": 0.32,
        "step_freq_hz": 1.45,
        "speed_gain": 0.35,
        "aim_lag_sec": 0.12,
        "aim_overshoot": 0.08,
        "settle_ends": True,
    },
    "handheld_follow": {
        "enabled": True,
        "pos_amp_m": 0.028,
        "rot_amp_deg": 0.85,
        "roll_amp_deg": 1.65,
        "freq_hz": 0.40,
        "step_freq_hz": 1.85,
        "speed_gain": 0.85,
        "aim_lag_sec": 0.16,
        "aim_overshoot": 0.16,
        "settle_ends": True,
    },
}


def normalize_camera_rig_config(rig_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """规范化摄影机轨道与手持参数 (camera_rig v3)，前后端数值严格同构。"""
    raw = rig_cfg if isinstance(rig_cfg, dict) else {}
    rail_mode = str(raw.get("rail_mode") or "free_spline").strip().lower()
    if rail_mode not in ("free_spline", "linear_rail", "arc_rail"):
        rail_mode = "free_spline"
    spline_mode = str(raw.get("spline_mode") or "centripetal_3d").strip().lower()
    if spline_mode not in ("centripetal_3d", "legacy_steffen"):
        spline_mode = "centripetal_3d"

    arc_center_raw = raw.get("arc_center")
    arc_center = None
    if isinstance(arc_center_raw, (list, tuple)) and len(arc_center_raw) >= 3:
        arc_center = [float(arc_center_raw[0]), float(arc_center_raw[1]), float(arc_center_raw[2])]

    hh_raw = raw.get("handheld") if isinstance(raw.get("handheld"), dict) else {}
    preset = str(hh_raw.get("preset") or "off").strip().lower()
    if preset in ("dolly", "none", ""):
        preset = "off"
    base_hh = CAMERA_RIG_HANDHELD_PRESETS.get(preset, CAMERA_RIG_HANDHELD_PRESETS["handheld_breath"])

    hh_enabled = bool(hh_raw["enabled"]) if "enabled" in hh_raw else bool(base_hh["enabled"])
    if preset == "off" and hh_raw.get("enabled") is not True:
        hh_enabled = False

    def _fval(key: str, default_val: float, lo: float, hi: float) -> float:
        src = hh_raw.get(key)
        if src is None:
            src = default_val
        return max(lo, min(hi, float(src)))

    aim_lag_sec = max(0.0, min(0.8, float(raw["aim_lag_sec"] if raw.get("aim_lag_sec") is not None else (hh_raw.get("aim_lag_sec") if hh_raw.get("aim_lag_sec") is not None else (base_hh["aim_lag_sec"] if hh_enabled else 0.0)))))
    aim_overshoot = max(0.0, min(0.5, float(raw["aim_overshoot"] if raw.get("aim_overshoot") is not None else (hh_raw.get("aim_overshoot") if hh_raw.get("aim_overshoot") is not None else (base_hh["aim_overshoot"] if hh_enabled else 0.0)))))

    return {
        "version": 3,
        "rail_mode": rail_mode,
        "spline_mode": spline_mode,
        "arc_center": arc_center,
        "lock_arc_radius": bool(raw.get("lock_arc_radius", False)),
        "lock_roll": bool(raw.get("lock_roll", True)),
        "aim_lag_sec": aim_lag_sec,
        "aim_overshoot": aim_overshoot,
        "handheld": {
            "preset": preset,
            "enabled": hh_enabled,
            "seed": int(hh_raw.get("seed") if hh_raw.get("seed") is not None else 1075),
            "intensity": _fval("intensity", 1.0, 0.0, 3.0),
            "pos_amp_m": _fval("pos_amp_m", base_hh["pos_amp_m"], 0.0, 0.25),
            "rot_amp_deg": _fval("rot_amp_deg", base_hh["rot_amp_deg"], 0.0, 8.0),
            "roll_amp_deg": _fval("roll_amp_deg", base_hh["roll_amp_deg"], 0.0, 10.0),
            "freq_hz": _fval("freq_hz", base_hh["freq_hz"], 0.05, 4.0),
            "step_freq_hz": _fval("step_freq_hz", base_hh["step_freq_hz"], 0.2, 6.0),
            "speed_gain": _fval("speed_gain", base_hh["speed_gain"], 0.0, 3.0),
            "settle_ends": bool(hh_raw.get("settle_ends", base_hh["settle_ends"])),
        },
    }


def _eval_camera_hermite_channel(
    kfs: List[Dict[str, Any]],
    times: List[float],
    vals: List[float],
    modes: List[str],
    seg_idx: int,
    u: float,
    default_endpoint_stop: bool = False,
) -> float:
    """
    单通道摄影机关键帧插值（前端 evalCameraHermiteChannel 与 Python 100% 数值一致）：
    - 当当前段模式为 'continuous' 时，使用基于真实非均匀时间间隔 h_k = t_{k+1} - t_k 的三次 Hermite 插值；
    - 中间连续关键帧使用非均匀三点抛物线导数 + Steffen 保单调切线限制 (|v_k| <= min(2|d_{k-1}|, 2|d_k|))，
      确保位置与速度 C1 连续、经过中间关键帧绝不自动停车、且严格无过冲（不超出相邻关键值范围）；
    - 当 default_endpoint_stop=True 时，首尾关键帧默认平稳起止（除非显式 stop=False）。
    """
    n = len(vals)
    i = max(0, min(n - 2, int(seg_idx)))
    u_clamped = max(0.0, min(1.0, float(u)))
    y0 = float(vals[i])
    y1 = float(vals[i + 1])
    hi = float(times[i + 1]) - float(times[i])
    mode_i = modes[i]

    if hi <= 1e-4 or abs(y1 - y0) <= 1e-7:
        return y0 if (mode_i == "hold" and u_clamped < 0.999999) else (y0 if abs(y1 - y0) <= 1e-7 else (y0 if u_clamped < 0.5 else y1))

    if mode_i != "continuous":
        s = eval_easing(u_clamped, mode_i)
        return y0 * (1.0 - s) + y1 * s

    h: List[float] = []
    d: List[float] = []
    for k in range(n - 1):
        hk = float(times[k + 1]) - float(times[k])
        dy = float(vals[k + 1]) - float(vals[k])
        h.append(hk)
        if hk <= 1e-4 or abs(dy) <= 1e-7 or modes[k] == "hold":
            d.append(0.0)
        else:
            d.append(dy / hk)

    v: List[float] = [0.0] * n
    for k in range(1, n - 1):
        explicit_stop = (
            bool(kfs[k].get("stop") or kfs[k].get("pause"))
            or (modes[k - 1] in ("hold", "ease_out", "ease_in_out"))
            or (modes[k] in ("hold", "ease_in", "ease_in_out"))
        )
        if explicit_stop or d[k - 1] * d[k] <= 0.0:
            v[k] = 0.0
        else:
            hk_prev = h[k - 1]
            hk_next = h[k]
            p_prime = (hk_next * d[k - 1] + hk_prev * d[k]) / (hk_prev + hk_next)
            sgn = 1.0 if d[k] > 0.0 else -1.0
            v[k] = sgn * min(abs(p_prime), 2.0 * abs(d[k - 1]), 2.0 * abs(d[k]))

    if bool(kfs[0].get("stop") or kfs[0].get("pause")) or (default_endpoint_stop and kfs[0].get("stop") is not False) or abs(d[0]) <= 1e-7:
        v[0] = 0.0
    elif n <= 2:
        v[0] = d[0]
    else:
        sgn0 = 1.0 if d[0] > 0.0 else -1.0
        v[0] = sgn0 * max(0.0, min(abs(d[0]), 3.0 * abs(d[0]) - 2.0 * abs(v[1])))

    if bool(kfs[n - 1].get("stop") or kfs[n - 1].get("pause")) or (default_endpoint_stop and kfs[n - 1].get("stop") is not False) or abs(d[n - 2]) <= 1e-7:
        v[n - 1] = 0.0
    elif n <= 2:
        v[n - 1] = d[n - 2]
    else:
        sgn_n = 1.0 if d[n - 2] > 0.0 else -1.0
        v[n - 1] = sgn_n * max(0.0, min(abs(d[n - 2]), 3.0 * abs(d[n - 2]) - 2.0 * abs(v[n - 2])))

    m0 = v[i] * hi
    m1 = v[i + 1] * hi
    u2 = u_clamped * u_clamped
    u3 = u2 * u_clamped
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u_clamped
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    val = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1

    lo = min(y0, y1)
    hi_bound = max(y0, y1)
    return max(lo, min(hi_bound, val))


def _eval_camera_vec3_centripetal_hermite(
    kfs: List[Dict[str, Any]],
    times: List[float],
    vecs: List[List[float]],
    modes: List[str],
    seg_idx: int,
    u: float,
    spatial_only: bool = False,
) -> List[float]:
    """
    三维向量级向心三次 Hermite 轨道样条（消除分轴 Steffen 折角与前段急起）：
    - 在三维速度空间统一融合切线方向（避免 X/Y/Z 单轴到极值被单独归零破坏弧线），
      仅对向量模长做 1.65*min(|d_prev|, |d_next|) 防过冲等比限幅；
    - 整镜起止默认零初速/零末速平滑加减速（除非关键帧显式设置 stop=False）。
    """
    n = len(vecs)
    i = max(0, min(n - 2, int(seg_idx)))
    uc = max(0.0, min(1.0, float(u)))
    p0 = [float(vecs[i][0]), float(vecs[i][1]), float(vecs[i][2])]
    p1 = [float(vecs[i + 1][0]), float(vecs[i + 1][1]), float(vecs[i + 1][2])]
    hi = float(times[i + 1]) - float(times[i])
    dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    chord_i = math.hypot(dx, dy, dz)
    mode_i = modes[i]

    if hi <= 1e-4 or chord_i <= 1e-7:
        if mode_i == "hold" and uc < 0.999999:
            return p0
        if chord_i <= 1e-7:
            return p0
        return p0 if uc < 0.5 else p1

    if mode_i != "continuous":
        s = eval_easing(uc, mode_i)
        return [p0[0] + dx * s, p0[1] + dy * s, p0[2] + dz * s]

    h: List[float] = []
    chords: List[float] = []
    d_vec: List[List[float]] = []
    d_spd: List[float] = []
    for k in range(n - 1):
        hk = float(times[k + 1]) - float(times[k])
        vx = float(vecs[k + 1][0]) - float(vecs[k][0])
        vy = float(vecs[k + 1][1]) - float(vecs[k][1])
        vz = float(vecs[k + 1][2]) - float(vecs[k][2])
        lk = math.hypot(vx, vy, vz)
        h.append(hk)
        chords.append(lk)
        if hk <= 1e-4 or lk <= 1e-7 or modes[k] == "hold":
            d_vec.append([0.0, 0.0, 0.0])
            d_spd.append(0.0)
        else:
            inv_h = 1.0 / hk
            d_vec.append([vx * inv_h, vy * inv_h, vz * inv_h])
            d_spd.append(lk * inv_h)

    v: List[List[float]] = [[0.0, 0.0, 0.0] for _ in range(n)]
    for k in range(1, n - 1):
        explicit_stop = (
            False if spatial_only else (
                bool(kfs[k].get("stop") or kfs[k].get("pause"))
                or (modes[k - 1] in ("hold", "ease_out", "ease_in_out"))
                or (modes[k] in ("hold", "ease_in", "ease_in_out"))
            )
        )
        sp_p, sp_n = d_spd[k - 1], d_spd[k]
        if explicit_stop:
            continue
        if sp_p <= 1e-7 or sp_n <= 1e-7:
            if spatial_only:
                if sp_p > 1e-7:
                    v[k] = list(d_vec[k - 1])
                elif sp_n > 1e-7:
                    v[k] = list(d_vec[k])
            continue
        dp, dn = d_vec[k - 1], d_vec[k]
        cos_th = (dp[0] * dn[0] + dp[1] * dn[1] + dp[2] * dn[2]) / (sp_p * sp_n)
        if cos_th < -0.92:
            continue
        tau_p = math.sqrt(h[k - 1] * max(chords[k - 1], 1e-4))
        tau_n = math.sqrt(h[k] * max(chords[k], 1e-4))
        denom = tau_p + tau_n
        if denom <= 1e-8:
            continue
        vx = (tau_n * dp[0] + tau_p * dn[0]) / denom
        vy = (tau_n * dp[1] + tau_p * dn[1]) / denom
        vz = (tau_n * dp[2] + tau_p * dn[2]) / denom
        vm = math.hypot(vx, vy, vz)
        max_spd = 1.65 * min(sp_p, sp_n)
        if vm > max_spd and vm > 1e-8:
            sc = max_spd / vm
            vx, vy, vz = vx * sc, vy * sc, vz * sc
        v[k] = [vx, vy, vz]

    def _cap_vec(vec3: List[float], lim: float) -> List[float]:
        vm = math.hypot(vec3[0], vec3[1], vec3[2])
        if vm > lim and vm > 1e-8:
            sc = lim / vm
            return [vec3[0] * sc, vec3[1] * sc, vec3[2] * sc]
        return vec3

    if (spatial_only or kfs[0].get("stop") is False) and d_spd[0] > 1e-7 and (spatial_only or modes[0] == "continuous"):
        if n >= 3 and math.hypot(v[1][0], v[1][1], v[1][2]) > 1e-7:
            v[0] = _cap_vec([1.5 * d_vec[0][c] - 0.5 * v[1][c] for c in range(3)], 1.25 * d_spd[0])
        else:
            v[0] = list(d_vec[0])

    if (spatial_only or kfs[n - 1].get("stop") is False) and d_spd[n - 2] > 1e-7 and (spatial_only or modes[n - 2] == "continuous"):
        if n >= 3 and math.hypot(v[n - 2][0], v[n - 2][1], v[n - 2][2]) > 1e-7:
            v[n - 1] = _cap_vec([1.5 * d_vec[n - 2][c] - 0.5 * v[n - 2][c] for c in range(3)], 1.25 * d_spd[n - 2])
        else:
            v[n - 1] = list(d_vec[n - 2])

    u2 = uc * uc
    u3 = u2 * uc
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = (u3 - 2.0 * u2 + uc) * hi
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = (u3 - u2) * hi
    return [
        h00 * p0[c] + h10 * v[i][c] + h01 * p1[c] + h11 * v[i + 1][c]
        for c in range(3)
    ]


def _fit_arc_rail_center_xz(pos_kfs: List[Dict[str, Any]], rig_cfg: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """拟合圆弧轨道水平圆心 (cx, cz)。"""
    ac = rig_cfg.get("arc_center")
    if isinstance(ac, (list, tuple)) and len(ac) >= 3:
        return float(ac[0]), float(ac[2])
    n = len(pos_kfs)
    if n >= 3:
        p0 = pos_kfs[0]["pos"]
        pm = pos_kfs[n // 2]["pos"]
        p2 = pos_kfs[-1]["pos"]
        ax, az = float(p0[0]), float(p0[2])
        bx, bz = float(pm[0]), float(pm[2])
        cx, cz = float(p2[0]), float(p2[2])
        d_det = 2.0 * (ax * (bz - cz) + bx * (cz - az) + cx * (az - bz))
        if abs(d_det) > 1e-4:
            a2 = ax * ax + az * az
            b2 = bx * bx + bz * bz
            c2 = cx * cx + cz * cz
            ux = (a2 * (bz - cz) + b2 * (cz - az) + c2 * (az - bz)) / d_det
            uz = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d_det
            r = math.hypot(ax - ux, az - uz)
            if 0.15 <= r <= 180.0:
                return ux, uz
    # 回退到关键帧目标平均水平焦点作为环绕圆心
    txs = [float(k["target"][0]) for k in pos_kfs if isinstance(k.get("target"), (list, tuple)) and len(k["target"]) >= 3]
    tzs = [float(k["target"][2]) for k in pos_kfs if isinstance(k.get("target"), (list, tuple)) and len(k["target"]) >= 3]
    if txs and tzs:
        return sum(txs) / len(txs), sum(tzs) / len(tzs)
    return None


def _ensure_valid_camera_target(
    pos: List[float],
    tar: List[float],
    k0: Dict[str, Any],
    k1: Dict[str, Any],
    u: float,
    min_dist: float = 0.05,
) -> List[float]:
    """防止摄影机位置与注视目标过近 (< min_dist) 导致视线方向突变或退化"""
    dx = float(tar[0]) - float(pos[0])
    dy = float(tar[1]) - float(pos[1])
    dz = float(tar[2]) - float(pos[2])
    dist = math.hypot(dx, dy, dz)
    if dist >= min_dist:
        return tar

    def _unit_look(kf: Dict[str, Any]) -> List[float]:
        kp = kf.get("pos", [0.0, 0.0, 0.0])
        kt = kf.get("target", [0.0, 0.0, -1.0])
        vx, vy, vz = float(kt[0]) - float(kp[0]), float(kt[1]) - float(kp[1]), float(kt[2]) - float(kp[2])
        vl = math.hypot(vx, vy, vz)
        return [vx / vl, vy / vl, vz / vl] if vl > 1e-6 else [0.0, 0.0, -1.0]

    u0 = _unit_look(k0)
    u1 = _unit_look(k1)
    ux = u0[0] * (1.0 - u) + u1[0] * u
    uy = u0[1] * (1.0 - u) + u1[1] * u
    uz = u0[2] * (1.0 - u) + u1[2] * u
    ul = math.hypot(ux, uy, uz)
    if ul <= 1e-6:
        ux, uy, uz = 0.0, 0.0, -1.0
    else:
        ux, uy, uz = ux / ul, uy / ul, uz / ul
    return [float(pos[0]) + ux * min_dist, float(pos[1]) + uy * min_dist, float(pos[2]) + uz * min_dist]


def _cam_v3_dot(a: List[float], b: List[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def _cam_v3_norm(v: List[float], fallback: Tuple[float, float, float] = (0.0, 0.0, -1.0)) -> List[float]:
    l = math.hypot(v[0], v[1], v[2])
    if l > 1e-9:
        inv = 1.0 / l
        return [v[0] * inv, v[1] * inv, v[2] * inv]
    return [fallback[0], fallback[1], fallback[2]]


def _cam_ortho_unit(u: List[float]) -> List[float]:
    ref = [0.0, 1.0, 0.0] if abs(u[1]) < 0.9 else [1.0, 0.0, 0.0]
    return _cam_v3_norm([
        ref[1] * u[2] - ref[2] * u[1],
        ref[2] * u[0] - ref[0] * u[2],
        ref[0] * u[1] - ref[1] * u[0],
    ])


def _slerp_dir(u0: List[float], u1: List[float], t: float) -> List[float]:
    tc = max(0.0, min(1.0, float(t)))
    if tc <= 0.0:
        return [u0[0], u0[1], u0[2]]
    if tc >= 1.0:
        return [u1[0], u1[1], u1[2]]
    d = max(-1.0, min(1.0, _cam_v3_dot(u0, u1)))
    if d > 0.999999:
        return _cam_v3_norm([u0[j] + (u1[j] - u0[j]) * tc for j in range(3)], (u0[0], u0[1], u0[2]))
    if d < -0.999999:
        axis = _cam_ortho_unit(u0)
        ang = math.pi * tc
        ca, sa = math.cos(ang), math.sin(ang)
        return _cam_v3_norm([u0[j] * ca + axis[j] * sa for j in range(3)], (u0[0], u0[1], u0[2]))
    theta = math.acos(d)
    st = math.sin(theta)
    w0 = math.sin((1.0 - tc) * theta) / st
    w1 = math.sin(tc * theta) / st
    return _cam_v3_norm([u0[j] * w0 + u1[j] * w1 for j in range(3)], (u0[0], u0[1], u0[2]))


def _log_s2(u_base: List[float], u_other: List[float]) -> List[float]:
    d = max(-1.0, min(1.0, _cam_v3_dot(u_base, u_other)))
    if d > 0.99999999:
        return [0.0, 0.0, 0.0]
    if d < -0.999999:
        axis = _cam_ortho_unit(u_base)
        return [axis[0] * math.pi, axis[1] * math.pi, axis[2] * math.pi]
    vx = u_other[0] - d * u_base[0]
    vy = u_other[1] - d * u_base[1]
    vz = u_other[2] - d * u_base[2]
    nv = math.hypot(vx, vy, vz)
    if nv <= 1e-9:
        return [0.0, 0.0, 0.0]
    scale = math.acos(d) / nv
    return [vx * scale, vy * scale, vz * scale]


def _exp_s2(u_base: List[float], omega: List[float]) -> List[float]:
    ang = math.hypot(omega[0], omega[1], omega[2])
    if ang <= 1e-9:
        return [u_base[0], u_base[1], u_base[2]]
    ca = math.cos(ang)
    sa = math.sin(ang) / ang
    return _cam_v3_norm([u_base[j] * ca + omega[j] * sa for j in range(3)], (u_base[0], u_base[1], u_base[2]))


def _parallel_transport_s2(vec: List[float], u_from: List[float], u_to: List[float]) -> List[float]:
    c = max(-1.0, min(1.0, _cam_v3_dot(u_from, u_to)))
    if c > 0.9999999:
        proj = _cam_v3_dot(vec, u_to)
        return [vec[j] - proj * u_to[j] for j in range(3)]
    if c < -0.99999:
        proj = _cam_v3_dot(vec, u_to)
        orth = [vec[j] - proj * u_to[j] for j in range(3)]
        lo = math.hypot(orth[0], orth[1], orth[2])
        lv = math.hypot(vec[0], vec[1], vec[2])
        if lo > 1e-8 and lv > 1e-8:
            sc = lv / lo
            return [orth[0] * sc, orth[1] * sc, orth[2] * sc]
        ax = _cam_ortho_unit(u_to)
        return [ax[0] * lv, ax[1] * lv, ax[2] * lv]
    k = _cam_v3_dot(u_to, vec) / (1.0 + c)
    return [vec[j] - k * (u_from[j] + u_to[j]) for j in range(3)]


def _eval_camera_spherical_dir(
    kfs: List[Dict[str, Any]],
    times: List[float],
    dirs: List[List[float]],
    modes: List[str],
    seg_idx: int,
    u: float,
    default_endpoint_stop: bool = False,
) -> List[float]:
    """
    单位视线球面 S^2 向心三次 Bezier (De Casteljau Slerp) 插值：
    - 保留显式 hold/ease 语义，非 continuous 段直接按测地线 Slerp 缓动；
    - 避免 1D 欧拉 Steffen 在拐点/偏航逆转处归零卡顿，以及三维笛卡尔 target 插值导致的视距坍缩甩头。
    """
    n = len(dirs)
    if n <= 0:
        return [0.0, 0.0, -1.0]
    if n == 1:
        return list(dirs[0])
    i = max(0, min(n - 2, int(seg_idx)))
    uc = max(0.0, min(1.0, float(u)))
    u0, u1 = dirs[i], dirs[i + 1]
    hi = float(times[i + 1]) - float(times[i])
    mode_i = modes[i]
    ang_i = math.hypot(*_log_s2(u0, u1))

    if hi <= 1e-4 or ang_i <= 1e-7:
        if mode_i == "hold" and uc < 0.999999:
            return list(u0)
        if ang_i <= 1e-7:
            return list(u0)
        return list(u0 if uc < 0.5 else u1)

    if mode_i != "continuous":
        if mode_i == "hold":
            return list(u0 if uc < 0.999999 else u1)
        return _slerp_dir(u0, u1, eval_easing(uc, mode_i))

    h: List[float] = []
    angs: List[float] = []
    v_in: List[List[float]] = [[0.0, 0.0, 0.0] for _ in range(n)]
    v_out: List[List[float]] = [[0.0, 0.0, 0.0] for _ in range(n)]
    for k in range(n - 1):
        hk = float(times[k + 1]) - float(times[k])
        h.append(hk)
        log_f = _log_s2(dirs[k], dirs[k + 1])
        ak = math.hypot(log_f[0], log_f[1], log_f[2])
        angs.append(ak)
        if hk > 1e-4 and ak > 1e-7 and modes[k] != "hold":
            inv_h = 1.0 / hk
            v_out[k] = [log_f[j] * inv_h for j in range(3)]
            log_b = _log_s2(dirs[k + 1], dirs[k])
            v_in[k + 1] = [-log_b[j] * inv_h for j in range(3)]

    omega: List[List[float]] = [[0.0, 0.0, 0.0] for _ in range(n)]
    for k in range(1, n - 1):
        explicit_stop = (
            bool(kfs[k].get("stop") or kfs[k].get("pause"))
            or (modes[k - 1] in ("hold", "ease_out", "ease_in_out"))
            or (modes[k] in ("hold", "ease_in", "ease_in_out"))
        )
        if explicit_stop:
            continue
        vi, vo = v_in[k], v_out[k]
        si = math.hypot(vi[0], vi[1], vi[2])
        so = math.hypot(vo[0], vo[1], vo[2])
        if si <= 1e-7 or so <= 1e-7:
            continue
        cos_th = _cam_v3_dot(vi, vo) / (si * so)
        if cos_th < -0.92:
            continue
        tau_p = math.sqrt(h[k - 1] * max(angs[k - 1], 1e-4))
        tau_n = math.sqrt(h[k] * max(angs[k], 1e-4))
        denom = tau_p + tau_n
        if denom <= 1e-8:
            continue
        wk = [(tau_n * vi[j] + tau_p * vo[j]) / denom for j in range(3)]
        wm = math.hypot(wk[0], wk[1], wk[2])
        lim = 1.65 * min(si, so)
        if wm > lim and wm > 1e-8:
            sc = lim / wm
            wk = [wk[0] * sc, wk[1] * sc, wk[2] * sc]
        omega[k] = wk

    for end_i, nb_i, cv, cs, active in (
        (0, 1, v_out[0], angs[0] / max(h[0], 1e-4), modes[0] == "continuous" and h[0] > 1e-4 and angs[0] > 1e-7),
        (n - 1, n - 2, v_in[n - 1], angs[n - 2] / max(h[n - 2], 1e-4), modes[n - 2] == "continuous" and h[n - 2] > 1e-4 and angs[n - 2] > 1e-7),
    ):
        if not active:
            continue
        kf = kfs[end_i]
        if bool(kf.get("stop") or kf.get("pause")) or (default_endpoint_stop and kf.get("stop") is not False):
            continue
        nb_m = math.hypot(omega[nb_i][0], omega[nb_i][1], omega[nb_i][2])
        if n <= 2 or nb_m <= 1e-7:
            omega[end_i] = list(cv)
        else:
            tr = _parallel_transport_s2(omega[nb_i], dirs[nb_i], dirs[end_i])
            we = [1.5 * cv[j] - 0.5 * tr[j] for j in range(3)]
            if _cam_v3_dot(we, cv) <= 0.0:
                omega[end_i] = [0.0, 0.0, 0.0]
            else:
                wm = math.hypot(we[0], we[1], we[2])
                if wm > 1.25 * cs and wm > 1e-8:
                    sc = (1.25 * cs) / wm
                    we = [we[0] * sc, we[1] * sc, we[2] * sc]
                omega[end_i] = we

    h_third = hi / 3.0
    b0 = u0
    b1 = _exp_s2(u0, [omega[i][j] * h_third for j in range(3)])
    b2 = _exp_s2(u1, [-omega[i + 1][j] * h_third for j in range(3)])
    b3 = u1
    p01 = _slerp_dir(b0, b1, uc)
    p12 = _slerp_dir(b1, b2, uc)
    p23 = _slerp_dir(b2, b3, uc)
    return _slerp_dir(_slerp_dir(p01, p12, uc), _slerp_dir(p12, p23, uc), uc)


def make_sphere_faces_world(center: np.ndarray, radius: float, lat_steps: int = 6, lon_steps: int = 10) -> List[np.ndarray]:
    """生成 3D 球体面片（用于圆头人偶与圆柱+球棋子头部，保证 Python 渲染与 Three.js 一致）"""
    faces: List[np.ndarray] = []
    for i in range(lat_steps):
        th0 = math.pi * (-0.5 + float(i) / lat_steps)
        th1 = math.pi * (-0.5 + float(i + 1) / lat_steps)
        y0, r0 = math.sin(th0) * radius, math.cos(th0) * radius
        y1, r1 = math.sin(th1) * radius, math.cos(th1) * radius
        for j in range(lon_steps):
            ph0 = 2.0 * math.pi * float(j) / lon_steps
            ph1 = 2.0 * math.pi * float(j + 1) / lon_steps
            p00 = center + np.array([r0 * math.cos(ph0), y0, r0 * math.sin(ph0)], dtype=np.float64)
            p01 = center + np.array([r0 * math.cos(ph1), y0, r0 * math.sin(ph1)], dtype=np.float64)
            p11 = center + np.array([r1 * math.cos(ph1), y1, r1 * math.sin(ph1)], dtype=np.float64)
            p10 = center + np.array([r1 * math.cos(ph0), y1, r1 * math.sin(ph0)], dtype=np.float64)
            faces.append(np.array([p00, p01, p11, p10], dtype=np.float64))
    return faces


def _fov_to_focal_mm_exact(fov_deg: float, sensor_height_mm: float = 24.0) -> float:
    fov = max(4.0, min(130.0, float(fov_deg or 37.8)))
    return sensor_height_mm / (2.0 * math.tan(math.radians(fov) * 0.5))


def _focal_mm_to_fov_exact(focal_mm: float, sensor_height_mm: float = 24.0) -> float:
    f = max(4.0, min(400.0, float(focal_mm or 35.0)))
    return math.degrees(2.0 * math.atan(sensor_height_mm / (2.0 * f)))


def _get_kf_fov_value(kf: Dict[str, Any], default_fov: float = 37.8) -> float:
    if kf.get("fov") is not None:
        return max(4.0, min(130.0, float(kf["fov"])))
    if kf.get("focal_mm") is not None:
        return max(4.0, min(130.0, _focal_mm_to_fov_exact(float(kf["focal_mm"]))))
    return max(4.0, min(130.0, float(default_fov)))


def _get_kf_roll_value(kf: Dict[str, Any], default_roll: float = 0.0) -> float:
    if kf.get("roll_deg") is not None:
        return max(-180.0, min(180.0, float(kf["roll_deg"])))
    return max(-180.0, min(180.0, float(default_roll or 0.0)))


def interpolate_camera_base(
    scene: Dict[str, Any],
    t: float,
    continuous_dir: bool = False,
) -> Dict[str, Any]:
    """
    基础空间摄影机插值求值器（与前端 evalCameraBaseAtTau 100% 数值对齐）：
    返回 {"pos", "target", "fov", "roll_deg", "look_dir", "look_dist", "focal_mm_exact"}。
    - spline_mode == "centripetal_3d"（默认）：三维向量级向心三次 Hermite，消除分轴折角与前段急起；
    - rail_mode == "linear_rail"：首尾直线滑轨 1D 投影约束，无侧漂；
    - rail_mode == "arc_rail"：水平极坐标 (theta, r, y) 圆弧轨道插值，消除弦切内缩。
    """
    raw_kfs = scene.get("camera_keyframes", [])
    cam = scene.get("camera", {}) or {}
    default_pos = [float(x) for x in cam.get("pos", [2.2, 1.5, 2.75])[:3]]
    default_tar = [float(x) for x in cam.get("target", [0.0, 0.72, 0.1])[:3]]
    default_fov = _get_kf_fov_value(cam, 37.8)
    default_roll = float(cam.get("roll_deg", 0.0) or 0.0)
    rig_cfg = normalize_camera_rig_config(scene.get("camera_rig"))
    rail_mode = rig_cfg["rail_mode"]
    spline_mode = rig_cfg["spline_mode"]
    use_centripetal = spline_mode == "centripetal_3d"

    def _pack_result(
        pos_v: List[float],
        tar_v: List[float],
        fov_v: float,
        roll_v: float = default_roll,
        dir_override: Optional[List[float]] = None,
        dist_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        fv = max(4.0, min(130.0, float(fov_v)))
        rv = max(-180.0, min(180.0, float(roll_v or 0.0)))
        if dir_override is not None and dist_override is not None:
            ld = dir_override
            r_d = max(0.05, float(dist_override))
            tv = [pos_v[0] + ld[0] * r_d, pos_v[1] + ld[1] * r_d, pos_v[2] + ld[2] * r_d]
        else:
            dx = float(tar_v[0]) - float(pos_v[0])
            dy = float(tar_v[1]) - float(pos_v[1])
            dz = float(tar_v[2]) - float(pos_v[2])
            r_d = math.hypot(dx, dy, dz)
            if r_d > 1e-6:
                ld = [dx / r_d, dy / r_d, dz / r_d]
            else:
                ld = [0.0, 0.0, -1.0]
                r_d = 1.0
            tv = tar_v
        return {
            "pos": pos_v,
            "target": tv,
            "fov": fv,
            "roll_deg": rv,
            "look_dir": ld,
            "look_dist": r_d,
            "focal_mm_exact": _fov_to_focal_mm_exact(fv),
        }

    if not raw_kfs:
        return _pack_result(default_pos, default_tar, default_fov, default_roll)

    kfs = sorted(raw_kfs, key=lambda item: float(item.get("t", 0.0)))
    pos_kfs = [k for k in kfs if k.get("has_pos", True) is not False and k.get("pos") and k.get("target")]
    fov_kfs = [k for k in kfs if k.get("has_fov", True) is not False and (k.get("fov") is not None or k.get("focal_mm") is not None)]
    roll_kfs = [k for k in kfs if k.get("has_roll", True) is not False and k.get("roll_deg") is not None]

    # 1. 求值焦距 / FOV
    if not fov_kfs:
        out_fov = default_fov
    elif len(fov_kfs) == 1 or t <= float(fov_kfs[0].get("t", 0.0)):
        out_fov = _get_kf_fov_value(fov_kfs[0], default_fov)
    elif t >= float(fov_kfs[-1].get("t", 0.0)):
        out_fov = _get_kf_fov_value(fov_kfs[-1], default_fov)
    else:
        f_times = [float(k.get("t", 0.0)) for k in fov_kfs]
        modes_fov = [normalize_cam_interp_mode(k.get("interp_fov") or k.get("interp") or "continuous") for k in fov_kfs]
        fov_vals = [_get_kf_fov_value(k, default_fov) for k in fov_kfs]
        out_fov = fov_vals[-1]
        for i in range(len(fov_kfs) - 1):
            t0, t1 = f_times[i], f_times[i + 1]
            if t0 <= t <= t1:
                dt = t1 - t0
                alpha = (t - t0) / dt if dt > 1e-4 else 0.0
                out_fov = _eval_camera_hermite_channel(fov_kfs, f_times, fov_vals, modes_fov, i, alpha, default_endpoint_stop=False)
                break

    # 2. 求值横滚角 Roll (deg)
    if not roll_kfs:
        out_roll = default_roll
    elif len(roll_kfs) == 1 or t <= float(roll_kfs[0].get("t", 0.0)):
        out_roll = _get_kf_roll_value(roll_kfs[0], default_roll)
    elif t >= float(roll_kfs[-1].get("t", 0.0)):
        out_roll = _get_kf_roll_value(roll_kfs[-1], default_roll)
    else:
        r_times = [float(k.get("t", 0.0)) for k in roll_kfs]
        modes_roll = [normalize_cam_interp_mode(k.get("interp_roll") or k.get("interp_target") or k.get("interp") or "continuous") for k in roll_kfs]
        roll_vals = [_get_kf_roll_value(k, default_roll) for k in roll_kfs]
        out_roll = roll_vals[-1]
        for i in range(len(roll_kfs) - 1):
            t0, t1 = r_times[i], r_times[i + 1]
            if t0 <= t <= t1:
                dt = t1 - t0
                alpha = (t - t0) / dt if dt > 1e-4 else 0.0
                out_roll = _eval_camera_hermite_channel(roll_kfs, r_times, roll_vals, modes_roll, i, alpha, default_endpoint_stop=use_centripetal)
                break

    # 3. 求值机位与视线 (pos / target)
    if not pos_kfs:
        return _pack_result(default_pos, default_tar, out_fov, out_roll)
    if len(pos_kfs) == 1 or t <= float(pos_kfs[0].get("t", 0.0)):
        k = pos_kfs[0]
        p = [float(x) for x in k["pos"][:3]]
        tar = _ensure_valid_camera_target(p, [float(x) for x in k["target"][:3]], k, k, 0.0)
        return _pack_result(p, tar, out_fov, out_roll)
    if t >= float(pos_kfs[-1].get("t", 0.0)):
        k = pos_kfs[-1]
        p = [float(x) for x in k["pos"][:3]]
        tar = _ensure_valid_camera_target(p, [float(x) for x in k["target"][:3]], k, k, 1.0)
        return _pack_result(p, tar, out_fov, out_roll)

    times = [float(k.get("t", 0.0)) for k in pos_kfs]
    modes_pos = [normalize_cam_interp_mode(k.get("interp_pos") or k.get("interp") or "continuous") for k in pos_kfs]
    modes_tar = [normalize_cam_interp_mode(k.get("interp_target") or k.get("interp") or "continuous") for k in pos_kfs]
    pos_vecs = [[float(k["pos"][0]), float(k["pos"][1]), float(k["pos"][2])] for k in pos_kfs]
    pos_x = [v[0] for v in pos_vecs]
    pos_y = [v[1] for v in pos_vecs]
    pos_z = [v[2] for v in pos_vecs]

    geom_modes_pos = ["hold" if m == "hold" else "continuous" for m in modes_pos] if continuous_dir else modes_pos
    geom_modes_tar = ["hold" if m == "hold" else "continuous" for m in modes_tar] if continuous_dir else modes_tar
    geom_pos_kfs = (
        [{k_n: v_n for k_n, v_n in kf.items() if k_n not in ("stop", "pause")} for kf in pos_kfs]
        if continuous_dir else pos_kfs
    )

    def _eval_rail_pos(seg_idx: int, alpha_val: float) -> List[float]:
        active_kfs = geom_pos_kfs if continuous_dir else pos_kfs
        active_modes = geom_modes_pos if continuous_dir else modes_pos
        if rail_mode == "linear_rail":
            p_start = pos_vecs[0]
            p_end = pos_vecs[-1]
            ax = p_end[0] - p_start[0]
            ay = p_end[1] - p_start[1]
            az = p_end[2] - p_start[2]
            ax_len = math.hypot(ax, ay, az)
            if ax_len > 1e-5:
                u_ax = [ax / ax_len, ay / ax_len, az / ax_len]
                s_vals = [
                    (pv[0] - p_start[0]) * u_ax[0] + (pv[1] - p_start[1]) * u_ax[1] + (pv[2] - p_start[2]) * u_ax[2]
                    for pv in pos_vecs
                ]
                s_cur = _eval_camera_hermite_channel(
                    active_kfs, times, s_vals, active_modes, seg_idx, alpha_val,
                    default_endpoint_stop=not continuous_dir,
                )
                return [p_start[0] + u_ax[0] * s_cur, p_start[1] + u_ax[1] * s_cur, p_start[2] + u_ax[2] * s_cur]
        elif rail_mode == "arc_rail":
            center_xz = _fit_arc_rail_center_xz(pos_kfs, rig_cfg)
            if center_xz is not None:
                cx, cz = center_xz
                rads: List[float] = []
                thetas: List[float] = []
                valid_arc = True
                for pv in pos_vecs:
                    dx_c = pv[0] - cx
                    dz_c = pv[2] - cz
                    rc = math.hypot(dx_c, dz_c)
                    if rc < 0.05:
                        valid_arc = False
                        break
                    th = math.atan2(dz_c, dx_c)
                    if thetas:
                        while th - thetas[-1] > math.pi:
                            th -= 2.0 * math.pi
                        while th - thetas[-1] < -math.pi:
                            th += 2.0 * math.pi
                    rads.append(rc)
                    thetas.append(th)
                if valid_arc and thetas:
                    th_i = _eval_camera_hermite_channel(
                        active_kfs, times, thetas, active_modes, seg_idx, alpha_val,
                        default_endpoint_stop=not continuous_dir,
                    )
                    if rig_cfg.get("lock_arc_radius"):
                        r_i = rads[0]
                    else:
                        r_i = max(0.05, _eval_camera_hermite_channel(active_kfs, times, rads, active_modes, seg_idx, alpha_val, default_endpoint_stop=not continuous_dir))
                    y_i = _eval_camera_hermite_channel(active_kfs, times, pos_y, active_modes, seg_idx, alpha_val, default_endpoint_stop=not continuous_dir)
                    return [cx + r_i * math.cos(th_i), y_i, cz + r_i * math.sin(th_i)]

        if use_centripetal:
            return _eval_camera_vec3_centripetal_hermite(active_kfs, times, pos_vecs, active_modes, seg_idx, alpha_val, spatial_only=continuous_dir)
        return [
            _eval_camera_hermite_channel(active_kfs, times, pos_x, active_modes, seg_idx, alpha_val, default_endpoint_stop=False),
            _eval_camera_hermite_channel(active_kfs, times, pos_y, active_modes, seg_idx, alpha_val, default_endpoint_stop=False),
            _eval_camera_hermite_channel(active_kfs, times, pos_z, active_modes, seg_idx, alpha_val, default_endpoint_stop=False),
        ]

    # 统一采用球面 S^2 向心三次 Bezier 插值视线方向 + 单通道 Hermite 插值视距
    dir_kfs = geom_pos_kfs if continuous_dir else pos_kfs
    dir_modes = geom_modes_tar if continuous_dir else modes_tar
    look_dirs: List[List[float]] = []
    dists: List[float] = []
    for k in pos_kfs:
        kp = [float(k["pos"][0]), float(k["pos"][1]), float(k["pos"][2])]
        kt = _ensure_valid_camera_target(kp, [float(x) for x in k["target"][:3]], k, k, 0.0)
        vx, vy, vz = float(kt[0]) - kp[0], float(kt[1]) - kp[1], float(kt[2]) - kp[2]
        rl = max(0.05, math.hypot(vx, vy, vz))
        look_dirs.append([vx / rl, vy / rl, vz / rl])
        dists.append(rl)

    out_pos = [float(x) for x in pos_kfs[-1]["pos"][:3]]
    out_dir = list(look_dirs[-1])
    out_dist = float(dists[-1])
    for i in range(len(pos_kfs) - 1):
        t0, t1 = times[i], times[i + 1]
        if t0 <= t <= t1:
            dt = t1 - t0
            alpha = (t - t0) / dt if dt > 1e-4 else 0.0
            out_pos = _eval_rail_pos(i, alpha)
            out_dir = _eval_camera_spherical_dir(
                dir_kfs, times, look_dirs, dir_modes, i, alpha,
                default_endpoint_stop=not continuous_dir,
            )
            out_dist = max(0.05, _eval_camera_hermite_channel(
                dir_kfs, times, dists, dir_modes, i, alpha,
                default_endpoint_stop=not continuous_dir,
            ))
            break
    return _pack_result(out_pos, out_pos, out_fov, out_roll, dir_override=out_dir, dist_override=out_dist)


# ============================================================================
# 摄影机弧长空间路径与速度曲线积分引擎 (camera_speed_curve v1，前后端 100% 同构)
# ============================================================================

_SPEED_CURVE_SOLVE_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_SPEED_CURVE_CACHE_MAX = 12


def _build_camera_speed_curve_signature(scene: Dict[str, Any]) -> str:
    rig_norm = normalize_camera_rig_config(scene.get("camera_rig"))
    payload = {
        "dur": round(float(scene.get("duration") or 4.0), 4),
        "cam": scene.get("camera"),
        "kfs": scene.get("camera_keyframes"),
        "curve": scene.get("camera_speed_curve"),
        "rig_geom": {
            "rail_mode": rig_norm["rail_mode"],
            "spline_mode": rig_norm["spline_mode"],
            "arc_center": rig_norm["arc_center"],
            "lock_arc_radius": rig_norm["lock_arc_radius"],
        },
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def _pchip_monotone_slopes(xs: List[float], ys: List[float]) -> List[float]:
    n = len(xs)
    m = [0.0] * n
    if n < 2:
        return m
    d = [0.0] * (n - 1)
    h = [0.0] * (n - 1)
    for i in range(n - 1):
        hi = xs[i + 1] - xs[i]
        h[i] = hi
        d[i] = max(0.0, (ys[i + 1] - ys[i]) / hi) if hi > 1e-9 else 0.0
    for i in range(1, n - 1):
        if d[i - 1] <= 1e-9 or d[i] <= 1e-9:
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    if d[0] > 1e-9:
        m[0] = d[0] if n == 2 else min(2.0 * d[0], max(0.0, ((2.0 * h[0] + h[1]) * d[0] - h[0] * d[1]) / max(1e-8, h[0] + h[1])))
    if d[n - 2] > 1e-9:
        m[n - 1] = d[n - 2] if n == 2 else min(2.0 * d[n - 2], max(0.0, ((2.0 * h[n - 2] + h[n - 3]) * d[n - 2] - h[n - 2] * d[n - 3]) / max(1e-8, h[n - 2] + h[n - 3])))
    # Fritsch-Carlson strict monotonicity guarantee
    for i in range(n - 1):
        if d[i] <= 1e-9:
            m[i] = 0.0
            m[i + 1] = 0.0
        else:
            a = m[i] / d[i]
            b = m[i + 1] / d[i]
            s2 = a * a + b * b
            if s2 > 9.0:
                sc = 3.0 / math.sqrt(s2)
                m[i] = sc * a * d[i]
                m[i + 1] = sc * b * d[i]
    return m


def _detect_camera_arc_cusps(scene: Dict[str, Any], dur: float) -> Dict[str, List[float]]:
    """识别摄影机基础几何样条在移动关键帧处的零切线折返尖点 (ds/dtau == 0)，避免速度规划强给非零速度引起反算跳变。"""
    raw_kfs = scene.get("camera_keyframes") or []
    kfs_sorted = sorted(raw_kfs, key=lambda item: float(item.get("t", 0.0)))
    pos_kfs = [k for k in kfs_sorted if k.get("has_pos", True) is not False and k.get("pos") and k.get("target")]
    fov_kfs = [k for k in kfs_sorted if k.get("has_fov", True) is not False and (k.get("fov") is not None or k.get("focal_mm") is not None)]
    cam = scene.get("camera") or {}
    default_fov = _get_kf_fov_value(cam, 37.8)
    rig_cfg = normalize_camera_rig_config(scene.get("camera_rig"))
    rail_mode = rig_cfg["rail_mode"]
    pos_cusps: List[float] = []
    rot_cusps: List[float] = []
    zoom_cusps: List[float] = []

    if len(pos_kfs) >= 3:
        times = [float(k.get("t", 0.0)) for k in pos_kfs]
        modes_pos = [normalize_cam_interp_mode(k.get("interp_pos") or k.get("interp") or "continuous") for k in pos_kfs]
        modes_tar = [normalize_cam_interp_mode(k.get("interp_target") or k.get("interp") or "continuous") for k in pos_kfs]
        pts = [[float(k["pos"][0]), float(k["pos"][1]), float(k["pos"][2])] for k in pos_kfs]
        dirs = [
            _cam_v3_norm([float(k["target"][0]) - pts[i][0], float(k["target"][1]) - pts[i][1], float(k["target"][2]) - pts[i][2]])
            for i, k in enumerate(pos_kfs)
        ]

        s_vals_lin: Optional[List[float]] = None
        if rail_mode == "linear_rail":
            p_start, p_end = pts[0], pts[-1]
            ax, ay, az = p_end[0] - p_start[0], p_end[1] - p_start[1], p_end[2] - p_start[2]
            ax_len = math.hypot(ax, ay, az)
            if ax_len > 1e-5:
                u_ax = [ax / ax_len, ay / ax_len, az / ax_len]
                s_vals_lin = [(pv[0] - p_start[0]) * u_ax[0] + (pv[1] - p_start[1]) * u_ax[1] + (pv[2] - p_start[2]) * u_ax[2] for pv in pts]

        thetas_arc: Optional[List[float]] = None
        if rail_mode == "arc_rail":
            center_xz = _fit_arc_rail_center_xz(pos_kfs, rig_cfg)
            if center_xz is not None:
                cx, cz = center_xz
                thetas_tmp: List[float] = []
                valid_arc = True
                for pv in pts:
                    dxc, dzc = pv[0] - cx, pv[2] - cz
                    if math.hypot(dxc, dzc) < 0.05:
                        valid_arc = False
                        break
                    th = math.atan2(dzc, dxc)
                    if thetas_tmp:
                        while th - thetas_tmp[-1] > math.pi:
                            th -= 2.0 * math.pi
                        while th - thetas_tmp[-1] < -math.pi:
                            th += 2.0 * math.pi
                    thetas_tmp.append(th)
                if valid_arc:
                    thetas_arc = thetas_tmp

        for k in range(1, len(pos_kfs) - 1):
            tk = times[k]
            if tk <= 1e-3 or tk >= dur - 1e-3:
                continue
            h0 = tk - times[k - 1]
            h1 = times[k + 1] - tk
            if h0 <= 1e-4 or h1 <= 1e-4:
                continue
            if modes_pos[k - 1] != "hold" and modes_pos[k] != "hold":
                if s_vals_lin is not None:
                    ds0 = s_vals_lin[k] - s_vals_lin[k - 1]
                    ds1 = s_vals_lin[k + 1] - s_vals_lin[k]
                    if abs(ds0) > 1e-3 and abs(ds1) > 1e-3 and ds0 * ds1 < 0.0:
                        pos_cusps.append(tk)
                elif thetas_arc is not None:
                    dt0 = thetas_arc[k] - thetas_arc[k - 1]
                    dt1 = thetas_arc[k + 1] - thetas_arc[k]
                    if abs(dt0) > 1e-3 and abs(dt1) > 1e-3 and dt0 * dt1 < 0.0:
                        pos_cusps.append(tk)
                else:
                    d0 = [pts[k][j] - pts[k - 1][j] for j in range(3)]
                    d1 = [pts[k + 1][j] - pts[k][j] for j in range(3)]
                    l0, l1 = math.hypot(*d0), math.hypot(*d1)
                    if l0 > 1e-3 and l1 > 1e-3:
                        cos_p = _cam_v3_dot(d0, d1) / (l0 * l1)
                        if cos_p < -0.92:
                            pos_cusps.append(tk)

            if modes_tar[k - 1] != "hold" and modes_tar[k] != "hold":
                lb = _log_s2(dirs[k], dirs[k - 1])
                lf = _log_s2(dirs[k], dirs[k + 1])
                a0, a1 = math.hypot(*lb), math.hypot(*lf)
                if a0 > 1e-3 and a1 > 1e-3:
                    vi = [-lb[j] / h0 for j in range(3)]
                    vo = [lf[j] / h1 for j in range(3)]
                    si, so = math.hypot(*vi), math.hypot(*vo)
                    if si > 1e-7 and so > 1e-7:
                        cos_r = _cam_v3_dot(vi, vo) / (si * so)
                        if cos_r < -0.92:
                            rot_cusps.append(tk)

    if len(fov_kfs) >= 3:
        ftimes = [float(k.get("t", 0.0)) for k in fov_kfs]
        modes_fov = [normalize_cam_interp_mode(k.get("interp_fov") or k.get("interp") or "continuous") for k in fov_kfs]
        focals = [_fov_to_focal_mm_exact(_get_kf_fov_value(k, default_fov)) for k in fov_kfs]
        for k in range(1, len(fov_kfs) - 1):
            tk = ftimes[k]
            if tk <= 1e-3 or tk >= dur - 1e-3:
                continue
            if ftimes[k] - ftimes[k - 1] <= 1e-4 or ftimes[k + 1] - ftimes[k] <= 1e-4:
                continue
            if modes_fov[k - 1] == "continuous" and modes_fov[k] == "continuous":
                df0 = focals[k] - focals[k - 1]
                df1 = focals[k + 1] - focals[k]
                if abs(df0) > 0.05 and abs(df1) > 0.05 and df0 * df1 < 0.0:
                    zoom_cusps.append(tk)

    return {"pos": pos_cusps, "rot": rot_cusps, "zoom": zoom_cusps}


def build_camera_arc_length_table(scene: Dict[str, Any]) -> Dict[str, Any]:
    """采样摄影机基础空间曲线（120Hz 高分辨率 + 单调 PCHIP + 空间曲率）并构建累积弧长表。"""
    dur = max(0.5, float(scene.get("duration") or 4.0))
    n_grid = max(480, min(4800, int(round(dur * 120.0))))
    raw_taus = [dur * (float(j) / float(n_grid)) for j in range(n_grid + 1)]
    for kf in (scene.get("camera_keyframes") or []):
        kt = float(kf.get("t", 0.0))
        if 0.0 <= kt <= dur:
            raw_taus.append(kt)
    curve_cfg = scene.get("camera_speed_curve") or {}
    for comp in (curve_cfg.get("compositions") or []):
        pt = float(comp.get("path_tau", comp.get("t", 0.0)))
        if 0.0 <= pt <= dur:
            raw_taus.append(pt)
    raw_taus.extend([0.0, dur])
    raw_taus.sort()

    taus: List[float] = []
    for tv in raw_taus:
        if not taus or abs(tv - taus[-1]) > 1e-5:
            taus.append(tv)
    taus[0] = 0.0
    taus[-1] = dur

    samples = [interpolate_camera_base(scene, tv, continuous_dir=True) for tv in taus]
    s_pos = [0.0]
    s_rot = [0.0]
    s_zoom = [0.0]
    s_gen = [0.0]
    for j in range(1, len(taus)):
        p0, p1 = samples[j - 1]["pos"], samples[j]["pos"]
        d_pos = math.hypot(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        u0, u1 = samples[j - 1]["look_dir"], samples[j]["look_dir"]
        dot_u = max(-1.0, min(1.0, u0[0] * u1[0] + u0[1] * u1[1] + u0[2] * u1[2]))
        d_rot = math.degrees(math.acos(dot_u))
        d_zoom = abs(samples[j]["focal_mm_exact"] - samples[j - 1]["focal_mm_exact"])
        s_pos.append(s_pos[-1] + d_pos)
        s_rot.append(s_rot[-1] + d_rot)
        s_zoom.append(s_zoom[-1] + d_zoom)
        s_gen.append(s_gen[-1] + d_pos + 0.02 * d_rot + 0.01 * d_zoom)

    m_pos = _pchip_monotone_slopes(taus, s_pos)
    m_rot = _pchip_monotone_slopes(taus, s_rot)
    m_zoom = _pchip_monotone_slopes(taus, s_zoom)
    m_gen = _pchip_monotone_slopes(taus, s_gen)
    curv_pos = compute_path_curvature_samples(
        lambda tq: interpolate_camera_base(scene, tq, continuous_dir=True)["pos"],
        taus,
        s_pos,
    )

    return {
        "duration": dur,
        "taus": taus,
        "samples": samples,
        "s_pos": s_pos,
        "s_rot": s_rot,
        "s_zoom": s_zoom,
        "s_gen": s_gen,
        "m_pos": m_pos,
        "m_rot": m_rot,
        "m_zoom": m_zoom,
        "m_gen": m_gen,
        "curv_pos": curv_pos,
        "total_pos": s_pos[-1],
        "total_rot": s_rot[-1],
        "total_zoom": s_zoom[-1],
        "total_gen": s_gen[-1],
        "cusps": _detect_camera_arc_cusps(scene, dur),
    }


def _eval_arc_at_tau(arc_table: Dict[str, Any], channel: str, tau: float) -> float:
    taus = arc_table["taus"]
    arr = arc_table[f"s_{channel}"]
    m_arr = arc_table.get(f"m_{channel}")
    if tau <= taus[0]:
        return arr[0]
    if tau >= taus[-1]:
        return arr[-1]
    lo, hi = 0, len(taus) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if taus[mid] <= tau:
            lo = mid
        else:
            hi = mid
    dt = taus[hi] - taus[lo]
    u = (tau - taus[lo]) / dt if dt > 1e-8 else 0.0
    if m_arr and dt > 1e-8:
        u2 = u * u
        u3 = u2 * u
        h00 = 2.0 * u3 - 3.0 * u2 + 1.0
        h10 = (u3 - 2.0 * u2 + u) * dt
        h01 = -2.0 * u3 + 3.0 * u2
        h11 = (u3 - u2) * dt
        return h00 * arr[lo] + h10 * m_arr[lo] + h01 * arr[hi] + h11 * m_arr[hi]
    return arr[lo] + (arr[hi] - arr[lo]) * u


def _arc_to_tau_in_window(
    arc_table: Dict[str, Any],
    channel: str,
    s_target: float,
    tau_a: float,
    tau_b: float,
) -> float:
    taus = arc_table["taus"]
    arr = arc_table[f"s_{channel}"]
    m_arr = arc_table.get(f"m_{channel}")
    if tau_b <= tau_a + 1e-7:
        return tau_a
    sa = _eval_arc_at_tau(arc_table, channel, tau_a)
    sb = _eval_arc_at_tau(arc_table, channel, tau_b)
    if sb - sa <= 1e-6:
        return tau_a
    s_clamped = max(sa, min(sb, float(s_target)))
    if s_clamped <= sa + 1e-9:
        return tau_a
    if s_clamped >= sb - 1e-9:
        return tau_b

    lo, hi = 0, len(arr) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= s_clamped:
            lo = mid
        else:
            hi = mid
    while hi < len(arr) - 1 and arr[hi] - arr[lo] <= 1e-9 and taus[hi] < tau_b:
        lo = hi
        hi += 1
    ds = arr[hi] - arr[lo]
    dt = taus[hi] - taus[lo]
    if ds <= 1e-9 or dt <= 1e-9:
        return max(tau_a, min(tau_b, taus[lo]))
    u = max(0.0, min(1.0, (s_clamped - arr[lo]) / ds))
    if m_arr:
        s0, s1 = arr[lo], arr[hi]
        k0, k1 = m_arr[lo] * dt, m_arr[hi] * dt
        u_lo, u_hi = 0.0, 1.0
        for _ in range(5):
            u2 = u * u
            u3 = u2 * u
            val = (2.0 * u3 - 3.0 * u2 + 1.0) * s0 + (u3 - 2.0 * u2 + u) * k0 + (-2.0 * u3 + 3.0 * u2) * s1 + (u3 - u2) * k1
            err = val - s_clamped
            if abs(err) <= 1e-10:
                break
            if err < 0.0:
                u_lo = max(u_lo, u)
            else:
                u_hi = min(u_hi, u)
            dval = (6.0 * u2 - 6.0 * u) * s0 + (3.0 * u2 - 4.0 * u + 1.0) * k0 + (-6.0 * u2 + 6.0 * u) * s1 + (3.0 * u2 - 2.0 * u) * k1
            if abs(dval) > 1e-9:
                u_cand = u - err / dval
                u = u_cand if u_lo < u_cand < u_hi else 0.5 * (u_lo + u_hi)
            else:
                u = 0.5 * (u_lo + u_hi)
    raw_tau = taus[lo] + u * dt
    return max(tau_a, min(tau_b, raw_tau))


def _build_monotone_speed_segments(points: List[Dict[str, Any]], t_start: float, t_end: float) -> List[Dict[str, Any]]:
    """
    由控制点 [{t, v, hold_duration}] 构建非负 C1 分段速度模型（Steffen 保单调三次 Hermite + 精确零速停顿段）。
    满足正齐次性 F(λv) = λF(v) (λ >= 0)，且每段速度 v(t) >= 0、积分有精确解析式。
    """
    if t_end <= t_start + 1e-6:
        return []
    pts = [
        {
            "t": max(t_start, min(t_end, float(p.get("t", t_start)))),
            "v": max(0.0, float(p.get("v", 0.0))),
            "hold": max(0.0, float(p.get("hold_duration", p.get("hold", 0.0)) or 0.0)),
        }
        for p in (points or [])
    ]
    pts.sort(key=lambda x: x["t"])
    dedup: List[Dict[str, Any]] = []
    for p in pts:
        if dedup and abs(p["t"] - dedup[-1]["t"]) < 1e-4:
            dedup[-1]["v"] = p["v"]
            dedup[-1]["hold"] = max(dedup[-1]["hold"], p["hold"])
        else:
            dedup.append(p)
    if not dedup:
        dedup = [{"t": t_start, "v": 0.0, "hold": 0.0}, {"t": t_end, "v": 0.0, "hold": 0.0}]
    if dedup[0]["t"] > t_start + 1e-4:
        dedup.insert(0, {"t": t_start, "v": dedup[0]["v"], "hold": 0.0})
    else:
        dedup[0]["t"] = t_start
    if dedup[-1]["t"] < t_end - 1e-4:
        dedup.append({"t": t_end, "v": dedup[-1]["v"], "hold": 0.0})
    else:
        dedup[-1]["t"] = t_end

    # 将带停顿的控制点拆解为停顿段 + 连续运动子序列
    segments: List[Dict[str, Any]] = []
    run_nodes: List[Tuple[float, float]] = []

    def _flush_run(nodes: List[Tuple[float, float]]) -> None:
        m = len(nodes)
        if m < 2:
            return
        h_list = [nodes[k + 1][0] - nodes[k][0] for k in range(m - 1)]
        d_list = [
            (nodes[k + 1][1] - nodes[k][1]) / h_list[k] if h_list[k] > 1e-6 else 0.0
            for k in range(m - 1)
        ]
        tangents = [0.0] * m
        if m == 2:
            v0, v1 = nodes[0][1], nodes[1][1]
            if v0 <= 1e-7 or v1 <= 1e-7 or abs(v1 - v0) <= 1e-9:
                tangents[0] = 0.0
                tangents[1] = 0.0
            else:
                tangents[0] = d_list[0]
                tangents[1] = d_list[0]
        else:
            for k in range(1, m - 1):
                if nodes[k][1] <= 1e-7 or d_list[k - 1] * d_list[k] <= 0.0:
                    tangents[k] = 0.0
                else:
                    hp, hn = h_list[k - 1], h_list[k]
                    p_prime = (hn * d_list[k - 1] + hp * d_list[k]) / (hp + hn)
                    sgn = 1.0 if d_list[k] > 0.0 else -1.0
                    tangents[k] = sgn * min(abs(p_prime), 2.0 * abs(d_list[k - 1]), 2.0 * abs(d_list[k]))
            if nodes[0][1] <= 1e-7 or abs(d_list[0]) <= 1e-9:
                tangents[0] = 0.0
            else:
                sgn0 = 1.0 if d_list[0] > 0.0 else -1.0
                tangents[0] = sgn0 * max(0.0, min(abs(d_list[0]), 3.0 * abs(d_list[0]) - 2.0 * abs(tangents[1])))
            if nodes[-1][1] <= 1e-7 or abs(d_list[-1]) <= 1e-9:
                tangents[-1] = 0.0
            else:
                sgn_n = 1.0 if d_list[-1] > 0.0 else -1.0
                tangents[-1] = sgn_n * max(0.0, min(abs(d_list[-1]), 3.0 * abs(d_list[-1]) - 2.0 * abs(tangents[-2])))

        for k in range(m - 1):
            t0, v0 = nodes[k]
            t1, v1 = nodes[k + 1]
            hk = t1 - t0
            if hk <= 1e-6:
                continue
            m0, m1 = tangents[k], tangents[k + 1]
            area = max(0.0, 0.5 * hk * (v0 + v1) + (hk * hk / 12.0) * (m0 - m1))
            segments.append({
                "kind": "hermite",
                "t0": t0,
                "t1": t1,
                "v0": v0,
                "v1": v1,
                "m0": m0,
                "m1": m1,
                "area": area,
            })

    for idx, pt in enumerate(dedup):
        tp = pt["t"]
        hold_d = pt["hold"]
        if hold_d > 1e-4 and tp < t_end - 1e-4:
            next_lim = dedup[idx + 1]["t"] - 0.02 if idx + 1 < len(dedup) else t_end
            hold_end = min(t_end, max(tp, min(tp + hold_d, next_lim if next_lim > tp + 1e-4 else dedup[idx + 1]["t"])))
            run_nodes.append((tp, 0.0))
            _flush_run(run_nodes)
            if hold_end > tp + 1e-5:
                segments.append({
                    "kind": "pause",
                    "t0": tp,
                    "t1": hold_end,
                    "v0": 0.0,
                    "v1": 0.0,
                    "m0": 0.0,
                    "m1": 0.0,
                    "area": 0.0,
                })
            run_nodes = [(hold_end, 0.0)]
        else:
            if run_nodes and tp <= run_nodes[-1][0] + 1e-5:
                run_nodes[-1] = (run_nodes[-1][0], pt["v"])
            else:
                run_nodes.append((tp, pt["v"]))
    _flush_run(run_nodes)
    return segments


def _eval_speed_segments_at_time(segments: List[Dict[str, Any]], t: float) -> Tuple[float, float]:
    """求值分段速度曲线在时刻 t 的瞬时速度 v(t) 与累积积分距离 s(t)。"""
    if not segments:
        return 0.0, 0.0
    if t <= segments[0]["t0"]:
        return segments[0]["v0"], 0.0
    acc = 0.0
    for seg in segments:
        t0, t1 = seg["t0"], seg["t1"]
        if t >= t1:
            acc += seg["area"]
            continue
        if t <= t0:
            return seg["v0"], acc
        if seg["kind"] == "pause":
            return 0.0, acc
        h = t1 - t0
        u = max(0.0, min(1.0, (t - t0) / h)) if h > 1e-7 else 0.0
        u2 = u * u
        u3 = u2 * u
        u4 = u2 * u2
        v0, v1, m0, m1 = seg["v0"], seg["v1"], seg["m0"], seg["m1"]
        val = (2.0 * u3 - 3.0 * u2 + 1.0) * v0 + (u3 - 2.0 * u2 + u) * h * m0 + (-2.0 * u3 + 3.0 * u2) * v1 + (u3 - u2) * h * m1
        part = h * (
            (0.5 * u4 - u3 + u) * v0
            + (0.25 * u4 - (2.0 / 3.0) * u3 + 0.5 * u2) * h * m0
            + (-0.5 * u4 + u3) * v1
            + (0.25 * u4 - (1.0 / 3.0) * u3) * h * m1
        )
        return max(0.0, val), acc + max(0.0, min(seg["area"], part))
    return segments[-1]["v1"], acc


def _interp_point_speed_at_t(points: List[Dict[str, Any]], t_q: float) -> float:
    if not points:
        return 0.0
    if t_q <= float(points[0].get("t", 0.0)):
        return max(0.0, float(points[0].get("v", 0.0)))
    if t_q >= float(points[-1].get("t", 0.0)):
        return max(0.0, float(points[-1].get("v", 0.0)))
    for i in range(len(points) - 1):
        t0 = float(points[i].get("t", 0.0))
        t1 = float(points[i + 1].get("t", 0.0))
        if t0 <= t_q <= t1:
            v0 = max(0.0, float(points[i].get("v", 0.0)))
            v1 = max(0.0, float(points[i + 1].get("v", 0.0)))
            dt = t1 - t0
            u = (t_q - t0) / dt if dt > 1e-6 else 0.0
            return v0 + (v1 - v0) * u
    return max(0.0, float(points[-1].get("v", 0.0)))


def _generate_preset_interval_points(
    preset: str,
    t_a: float,
    t_b: float,
    delta_s: float,
    full_dur: float,
    existing_pauses: List[Dict[str, Any]],
    v_join_left: Optional[float] = None,
    v_join_right: Optional[float] = None,
) -> List[Dict[str, Any]]:
    dt = max(1e-4, t_b - t_a)
    if delta_s <= 1e-6:
        base = [{"t": t_a, "v": 0.0, "hold_duration": 0.0}, {"t": t_b, "v": 0.0, "hold_duration": 0.0}]
        return base

    is_shot_start = t_a <= 1e-4
    is_shot_end = t_b >= full_dur - 1e-4
    p_mode = str(preset or "continuous_push")

    if p_mode == "constant":
        pts = [
            {"t": t_a, "v": 1.0, "hold_duration": 0.0},
            {"t": round(t_a + 0.5 * dt, 4), "v": 1.0, "hold_duration": 0.0},
            {"t": t_b, "v": 1.0, "hold_duration": 0.0},
        ]
    elif p_mode == "ease_ends":
        r_in = min(0.22 * dt, 2.2) if is_shot_start else 0.0
        r_out = min(0.22 * dt, 2.2) if is_shot_end else 0.0
        pts = [{"t": t_a, "v": 0.0 if is_shot_start else 1.0, "hold_duration": 0.0}]
        if r_in > 0.04 and t_a + r_in < t_b - r_out - 0.04:
            pts.append({"t": round(t_a + r_in, 4), "v": 1.0, "hold_duration": 0.0})
        pts.append({"t": round(0.5 * (t_a + t_b), 4), "v": 1.0, "hold_duration": 0.0})
        if r_out > 0.04 and t_b - r_out > t_a + r_in + 0.04:
            pts.append({"t": round(t_b - r_out, 4), "v": 1.0, "hold_duration": 0.0})
        pts.append({"t": t_b, "v": 0.0 if is_shot_end else 1.0, "hold_duration": 0.0})
    else:
        # continuous_push (默认持续推进：起尾自然过渡，经过中间帧连续不停顿，尊重用户停顿)
        r_in = min(0.15 * dt, 1.8) if is_shot_start else 0.0
        r_out = min(0.25 * dt, 2.8) if is_shot_end else 0.0
        pts = [{"t": t_a, "v": 0.0 if is_shot_start else 1.0, "hold_duration": 0.0}]
        if r_in > 0.04 and t_a + r_in < t_b - r_out - 0.04:
            pts.append({"t": round(t_a + r_in, 4), "v": 1.0, "hold_duration": 0.0})
        pts.append({"t": round(t_a + 0.45 * dt, 4), "v": 1.0, "hold_duration": 0.0})
        if r_out > 0.04 and t_b - r_out > t_a + r_in + 0.04:
            pts.append({"t": round(t_b - r_out, 4), "v": 0.88, "hold_duration": 0.0})
        pts.append({"t": t_b, "v": 0.0 if is_shot_end else 1.0, "hold_duration": 0.0})

    for pz in (existing_pauses or []):
        pt_t = float(pz.get("t", -1.0))
        hd = max(0.0, float(pz.get("hold_duration", pz.get("hold", 0.0)) or 0.0))
        if hd > 1e-4 and t_a <= pt_t <= t_b - 0.02:
            pts.append({"t": pt_t, "v": 0.0, "hold_duration": hd})

    # 若指定了相邻锁定构图区间的连续衔接速率 (v_join_left / v_join_right)，保持边界连续并求解内部控制点满足积分 = delta_s
    if p_mode != "constant" and (v_join_left is not None or v_join_right is not None):
        pts.sort(key=lambda x: x["t"])
        v_l = 0.0 if is_shot_start else (float(v_join_left) if v_join_left is not None else None)
        v_r = 0.0 if is_shot_end else (float(v_join_right) if v_join_right is not None else None)
        interior_free = [
            p for idx_p, p in enumerate(pts)
            if 0 < idx_p < len(pts) - 1 and float(p.get("hold_duration", 0.0)) <= 1e-4
        ]
        if interior_free:
            def _area_with_scale(sc: float) -> float:
                trial = []
                for idx_p, p in enumerate(pts):
                    if float(p.get("hold_duration", 0.0)) > 1e-4:
                        vv = 0.0
                    elif idx_p == 0 and v_l is not None:
                        vv = v_l
                    elif idx_p == len(pts) - 1 and v_r is not None:
                        vv = v_r
                    else:
                        vv = p["v"] * sc
                    trial.append({"t": p["t"], "v": vv, "hold_duration": p.get("hold_duration", 0.0)})
                return sum(s["area"] for s in _build_monotone_speed_segments(trial, t_a, t_b))

            min_area = _area_with_scale(0.0)
            if min_area <= delta_s + 1e-7:
                lo, hi = 0.0, max(1.0, (delta_s / max(1e-6, dt)) * 2.5)
                while _area_with_scale(hi) < delta_s and hi < 2048.0:
                    hi *= 2.0
                for _ in range(28):
                    mid = 0.5 * (lo + hi)
                    if _area_with_scale(mid) < delta_s:
                        lo = mid
                    else:
                        hi = mid
                sc_star = 0.5 * (lo + hi)
                for idx_p, p in enumerate(pts):
                    if float(p.get("hold_duration", 0.0)) > 1e-4:
                        p["v"] = 0.0
                    elif idx_p == 0 and v_l is not None:
                        p["v"] = v_l
                    elif idx_p == len(pts) - 1 and v_r is not None:
                        p["v"] = v_r
                    else:
                        p["v"] = p["v"] * sc_star
                return pts

    segs = _build_monotone_speed_segments(pts, t_a, t_b)
    area = sum(s["area"] for s in segs)
    if area > 1e-8:
        scale = delta_s / area
        for p in pts:
            if float(p.get("hold_duration", 0.0)) <= 1e-4:
                p["v"] = p["v"] * scale
    else:
        v_const = delta_s / dt
        for p in pts:
            p["hold_duration"] = 0.0
            p["v"] = v_const
    return pts


def _slice_hermite_segments_window(
    segments: List[Dict[str, Any]],
    t_a: float,
    t_b: float,
    expected_ds: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """从全局连续段中精确切分出时间区间 [t_a, t_b] 的子段（利用 Hermite sub-division 保持零失真与积分和精确）。"""
    out: List[Dict[str, Any]] = []
    if t_b <= t_a + 1e-7:
        return out
    for seg in (segments or []):
        s0 = float(seg["t0"])
        s1 = float(seg["t1"])
        if s1 <= t_a + 1e-7 or s0 >= t_b - 1e-7:
            continue
        c0 = max(t_a, s0)
        c1 = min(t_b, s1)
        if c1 <= c0 + 1e-7:
            continue
        if seg.get("kind") == "pause" or (seg.get("area", 0.0) <= 1e-8 and abs(seg.get("v0", 0.0)) <= 1e-8 and abs(seg.get("v1", 0.0)) <= 1e-8):
            out.append({"kind": "pause", "t0": c0, "t1": c1, "v0": 0.0, "v1": 0.0, "m0": 0.0, "m1": 0.0, "area": 0.0})
            continue
        h = s1 - s0
        if c0 <= s0 + 1e-7 and c1 >= s1 - 1e-7:
            out.append(dict(seg))
            continue
        u_start = max(0.0, min(1.0, (c0 - s0) / h)) if h > 1e-7 else 0.0
        u_end = max(0.0, min(1.0, (c1 - s0) / h)) if h > 1e-7 else 1.0
        # 用全局 eval_segments_at_time 精确提取区间两端速度/导数/面积
        v_c0, s_c0, a_c0 = eval_segments_at_time([seg], c0)
        v_c1, s_c1, a_c1 = eval_segments_at_time([seg], c1)
        out.append({
            "kind": "hermite",
            "t0": c0,
            "t1": c1,
            "v0": max(0.0, v_c0),
            "v1": max(0.0, v_c1),
            "m0": a_c0,
            "m1": a_c1,
            "area": max(0.0, s_c1 - s_c0),
        })
    if not out:
        out.append({"kind": "pause", "t0": t_a, "t1": t_b, "v0": 0.0, "v1": 0.0, "m0": 0.0, "m1": 0.0, "area": 0.0})
    elif expected_ds is not None:
        cur_a = sum(float(s["area"]) for s in out)
        if expected_ds > 1e-6 and cur_a > 1e-8 and abs(cur_a - expected_ds) > 1e-5:
            fac = expected_ds / cur_a
            for s in out:
                s["v0"] *= fac
                s["v1"] *= fac
                s["m0"] *= fac
                s["m1"] *= fac
                s["area"] *= fac
    return out


def solve_camera_speed_curve_channel(
    arc_table: Dict[str, Any],
    curve_cfg: Dict[str, Any],
    channel: str = "pos",
    edited_time: Optional[float] = None,
    edited_speed: Optional[float] = None,
    baseline_samples: Optional[List[Tuple[float, float]]] = None,
    limits: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    联合求解单个通道 (pos / rot / zoom) 的连续速度模型与约束检查结果：
    - 默认走 motion_continuity 联合 C2 ADMM B 样条求解（相邻锁定边界共享速度与加速度，消除境界突变与急刹折角）；
    - constant / constant_speed 走 S5 平滑起停 + 严格恒速巡航；
    - 仅显式 manual 覆盖时回退传统分段控制。
    """
    dur = arc_table["duration"]
    total_len = float(arc_table[f"total_{channel}"])
    lock_comps = bool(curve_cfg.get("lock_compositions", True))
    chan_map = curve_cfg.get("channels") or {}
    chan_cfg = chan_map.get(channel) or {}
    preset = str(chan_cfg.get("preset") or curve_cfg.get("preset") or "continuous_push")
    mode_cfg = str(chan_cfg.get("mode") or curve_cfg.get("mode") or "").lower().strip()
    raw_pts = sorted(
        [dict(p) for p in (chan_cfg.get("points") or []) if isinstance(p, dict)],
        key=lambda x: float(x.get("t", 0.0)),
    )

    # 整理构图锚点 (含 0 与 duration；无论是否锁定构图，均识别静止平台起点/终点避免延长总时长后将停驻尾段扯长移动)
    anchors: List[Tuple[float, float]] = [(0.0, 0.0)]
    if lock_comps:
        for c in sorted((curve_cfg.get("compositions") or []), key=lambda x: float(x.get("t", 0.0))):
            if c.get("locked", True) is False:
                continue
            tc = max(0.0, min(dur, float(c.get("t", 0.0))))
            tau_c = max(0.0, min(dur, float(c.get("path_tau", tc))))
            if tc > anchors[-1][0] + 1e-3 and tc < dur - 1e-3:
                tau_c = max(anchors[-1][1], tau_c)
                anchors.append((tc, tau_c))
    anchors.append((dur, dur))

    taus_tbl = arc_table.get("taus") or []
    flat_tables_to_scan = []
    s_chan_tbl = arc_table.get(f"s_{channel}") or []
    if len(taus_tbl) == len(s_chan_tbl) and len(taus_tbl) >= 3 and total_len > 1e-5:
        flat_tables_to_scan.append(s_chan_tbl)
    is_linked_mode = curve_cfg.get("linked", True) is not False
    s_gen_tbl = arc_table.get("s_gen") or []
    if is_linked_mode and len(taus_tbl) == len(s_gen_tbl) and len(taus_tbl) >= 3 and float(arc_table.get("total_gen", 0.0) or 0.0) > 1e-5:
        flat_tables_to_scan.append(s_gen_tbl)
    flat_boundaries: List[float] = []
    for s_tbl in flat_tables_to_scan:
        flat_start: Optional[float] = None
        for k_f in range(len(s_tbl) - 1):
            if abs(s_tbl[k_f + 1] - s_tbl[k_f]) <= 1e-6:
                if flat_start is None:
                    flat_start = float(taus_tbl[k_f])
            else:
                if flat_start is not None:
                    flat_end = float(taus_tbl[k_f])
                    if flat_end - flat_start >= 0.05:
                        flat_boundaries.extend((flat_start, flat_end))
                    flat_start = None
        if flat_start is not None and dur - flat_start >= 0.05:
            flat_boundaries.append(flat_start)
    flat_boundaries.sort()
    for bd in flat_boundaries:
        if bd <= 1e-3 or bd >= dur - 1e-3:
            continue
        if any(abs(at - bd) <= 1e-3 or abs(atau - bd) <= 1e-3 for at, atau in anchors):
            continue
        ins_idx = -1
        for ai in range(1, len(anchors)):
            if anchors[ai - 1][0] + 1e-3 < bd < anchors[ai][0] - 1e-3:
                ins_idx = ai
                break
        if ins_idx > 0 and anchors[ins_idx - 1][1] <= bd + 1e-6 and bd <= anchors[ins_idx][1] + 1e-6:
            anchors.insert(ins_idx, (bd, bd))

    cusp_stop_times: List[float] = []
    if mode_cfg != "manual" and preset != "manual":
        chan_cusps = sorted(float(c) for c in ((arc_table.get("cusps") or {}).get(channel) or []))
        for tau_c in chan_cusps:
            if tau_c <= 1e-3 or tau_c >= dur - 1e-3:
                continue
            matched_anchor_t: Optional[float] = None
            for at, atau in anchors:
                if abs(atau - tau_c) <= 1e-3 and 1e-3 < at < dur - 1e-3:
                    matched_anchor_t = at
                    break
            if matched_anchor_t is not None:
                cusp_stop_times.append(matched_anchor_t)
                continue
            for m_idx in range(len(anchors) - 1):
                ta_m, tau_a_m = anchors[m_idx]
                tb_m, tau_b_m = anchors[m_idx + 1]
                if tau_a_m + 1e-3 < tau_c < tau_b_m - 1e-3 and tb_m - ta_m > 2e-2:
                    sa_m = _eval_arc_at_tau(arc_table, channel, tau_a_m)
                    sb_m = _eval_arc_at_tau(arc_table, channel, tau_b_m)
                    sc_m = _eval_arc_at_tau(arc_table, channel, tau_c)
                    ds_iv = max(0.0, sb_m - sa_m)
                    ds_sub = max(0.0, sc_m - sa_m)
                    if ds_sub > 1e-5 and (ds_iv - ds_sub) > 1e-5:
                        tc_m = ta_m + (tb_m - ta_m) * (ds_sub / ds_iv)
                        if ta_m + 1e-3 < tc_m < tb_m - 1e-3:
                            anchors.insert(m_idx + 1, (tc_m, tau_c))
                            cusp_stop_times.append(tc_m)
                    break

    n_iv = len(anchors) - 1
    iv_ds_list: List[float] = []
    iv_dt_list: List[float] = []
    iv_sa_list: List[float] = []
    iv_sb_list: List[float] = []
    for m in range(n_iv):
        ta_m, tau_a_m = anchors[m]
        tb_m, tau_b_m = anchors[m + 1]
        sa_m = _eval_arc_at_tau(arc_table, channel, tau_a_m)
        sb_m = _eval_arc_at_tau(arc_table, channel, tau_b_m)
        ds_m = max(0.0, sb_m - sa_m)
        dt_m = max(1e-4, tb_m - ta_m)
        iv_sa_list.append(sa_m)
        iv_sb_list.append(sb_m)
        iv_ds_list.append(ds_m)
        iv_dt_list.append(dt_m)

    # 若显式指定手动模式（高级手动/非连续覆盖），保留手工控制点分支
    if mode_cfg == "manual" or preset == "manual":
        intervals: List[Dict[str, Any]] = []
        conflicts: List[Dict[str, Any]] = []
        for m in range(n_iv):
            ta, tau_a = anchors[m]
            tb, tau_b = anchors[m + 1]
            sa, sb, ds, dt = iv_sa_list[m], iv_sb_list[m], iv_ds_list[m], iv_dt_list[m]
            v_left = _interp_point_speed_at_t(raw_pts, ta)
            v_right = _interp_point_speed_at_t(raw_pts, tb)
            iv_pts = [{"t": ta, "v": v_left, "hold_duration": 0.0}]
            for p in raw_pts:
                pt_t = float(p.get("t", 0.0))
                if abs(pt_t - ta) <= 1e-4:
                    iv_pts[0]["v"] = max(0.0, float(p.get("v", v_left)))
                    iv_pts[0]["hold_duration"] = max(0.0, float(p.get("hold_duration", p.get("hold", 0.0)) or 0.0))
                elif ta + 1e-4 < pt_t < tb - 1e-4:
                    iv_pts.append({
                        "t": pt_t,
                        "v": max(0.0, float(p.get("v", 0.0))),
                        "hold_duration": max(0.0, float(p.get("hold_duration", p.get("hold", 0.0)) or 0.0)),
                    })
            end_match = next((p for p in raw_pts if abs(float(p.get("t", 0.0)) - tb) <= 1e-4), None)
            iv_pts.append({
                "t": tb,
                "v": max(0.0, float(end_match.get("v", v_right))) if end_match else v_right,
                "hold_duration": 0.0,
            })
            if lock_comps:
                if ds <= 1e-6:
                    for p in iv_pts:
                        p["v"] = 0.0
                else:
                    segs_try = _build_monotone_speed_segments(iv_pts, ta, tb)
                    area_try = sum(s["area"] for s in segs_try)
                    if abs(area_try - ds) > 1e-4:
                        if area_try > 1e-8:
                            sc_f = ds / area_try
                            for p in iv_pts:
                                if float(p.get("hold_duration", 0.0)) <= 1e-4:
                                    p["v"] = p["v"] * sc_f
                        else:
                            v_c = ds / max(0.04, dt)
                            for p in iv_pts:
                                if float(p.get("hold_duration", 0.0)) <= 1e-4:
                                    p["v"] = v_c
            segs = _build_monotone_speed_segments(iv_pts, ta, tb)
            intervals.append({
                "t_a": ta, "t_b": tb, "tau_a": tau_a, "tau_b": tau_b,
                "s_a": sa, "s_b": sb, "delta_s": ds, "segments": segs, "points": iv_pts,
            })
        return {
            "channel": channel,
            "mode": "manual",
            "total_length": total_len,
            "lock_compositions": lock_comps,
            "intervals": intervals,
            "conflicts": conflicts,
            "segments": [s for iv in intervals for s in iv["segments"]],
            "clamped": False,
        }

    # 连续平滑 / 保持匀速 统一联合求解
    solve_mode = "constant_speed" if (mode_cfg == "constant_speed" or preset in ("constant", "constant_speed")) else "smooth_variable"
    chan_lim = dict(DEFAULT_PROTECTION_LIMITS["camera"])
    if isinstance(curve_cfg.get("limits"), dict):
        for kl, vl in curve_cfg["limits"].items():
            if vl is not None:
                chan_lim[kl] = float(vl)
    if limits:
        for kl, vl in limits.items():
            if vl is not None:
                chan_lim[kl] = float(vl)

    if channel == "rot":
        chan_lim["max_speed"] = chan_lim.get("max_ang_vel", 180.0)
        chan_lim["max_accel"] = chan_lim.get("max_ang_accel", 420.0)
        chan_lim["max_jerk"] = chan_lim["max_accel"] * 4.0
        curv_samples = None
    elif channel == "zoom":
        chan_lim["max_speed"] = chan_lim.get("max_zoom_speed", 120.0)
        chan_lim["max_accel"] = chan_lim.get("max_zoom_accel", 280.0)
        chan_lim["max_jerk"] = chan_lim["max_accel"] * 4.0
        curv_samples = None
    else:
        curv_samples = arc_table.get("curv_pos")

    # 若通道有自定义关键点超出默认上限且被显式 pin 或保留，提高兼容上界或按冲突检测处理
    pauses_only = [p for p in raw_pts if float(p.get("hold_duration", p.get("hold", 0.0)) or 0.0) > 1e-4 or bool(p.get("stop"))]
    if solve_mode == "constant_speed":
        pts_in = list(pauses_only)
    elif preset == "custom" and raw_pts:
        pts_in = list(raw_pts)
    else:
        pts_in = list(pauses_only)
    if cusp_stop_times:
        existing_holds = [
            (
                float(p.get("t", 0.0)),
                float(p.get("t", 0.0)) + max(0.0, float(p.get("hold_duration", p.get("hold", 0.0)) or 0.0)),
            )
            for p in pts_in
            if float(p.get("hold_duration", p.get("hold", 0.0)) or 0.0) > 1e-4
        ]
        for tc_stop in cusp_stop_times:
            if any(hs - 1e-3 <= tc_stop <= he + 1e-3 for hs, he in existing_holds):
                continue
            pts_in = [p for p in pts_in if abs(float(p.get("t", -1.0)) - tc_stop) > 1e-3]
            pts_in.append({"t": tc_stop, "v": 0.0, "stop": True})

    # 判断首尾停顿条件（若 custom/raw_pts 明确首尾给非零初速则不强压为零）
    bc0 = True
    bc1 = True
    if raw_pts and preset == "custom":
        first_p = next((p for p in raw_pts if abs(float(p.get("t", -1.0))) <= 0.03), None)
        last_p = next((p for p in raw_pts if abs(float(p.get("t", -1.0)) - dur) <= 0.03), None)
        if first_p and float(first_p.get("v", 0.0)) > 0.05 and not first_p.get("stop"):
            bc0 = False
        if last_p and float(last_p.get("v", 0.0)) > 0.05 and not last_p.get("stop"):
            bc1 = False

    ivs_in = [
        {
            "t_a": anchors[m][0],
            "t_b": anchors[m + 1][0],
            "tau_a": anchors[m][1],
            "tau_b": anchors[m + 1][1],
            "s_a": iv_sa_list[m],
            "s_b": iv_sb_list[m],
            "delta_s": iv_ds_list[m],
        }
        for m in range(n_iv)
    ]

    solved = solve_continuous_velocity_profile(
        duration=dur,
        intervals_in=ivs_in,
        points_in=pts_in,
        mode=solve_mode,
        limits=chan_lim,
        bc_start_stop=bc0,
        bc_end_stop=bc1,
        edited_time=edited_time,
        edited_speed=edited_speed,
        baseline_samples=baseline_samples,
        curvature_samples=curv_samples,
        lock_compositions=lock_comps,
    )

    intervals_out: List[Dict[str, Any]] = []
    for m in range(n_iv):
        ta, tau_a = anchors[m]
        tb, tau_b = anchors[m + 1]
        sa, sb, ds = iv_sa_list[m], iv_sb_list[m], iv_ds_list[m]
        iv_segs = _slice_hermite_segments_window(solved["segments"], ta, tb, expected_ds=ds if lock_comps else None)
        iv_pts = [{"t": ta, "v": round(eval_segments_at_time(iv_segs, ta)[0], 5), "hold_duration": 0.0}]
        for p in raw_pts:
            pt_t = float(p.get("t", 0.0))
            if ta + 1e-4 < pt_t < tb - 1e-4:
                iv_pts.append({
                    "t": pt_t,
                    "v": round(eval_segments_at_time(iv_segs, pt_t)[0], 5),
                    "hold_duration": float(p.get("hold_duration", p.get("hold", 0.0)) or 0.0),
                })
        iv_pts.append({"t": tb, "v": round(eval_segments_at_time(iv_segs, tb)[0], 5), "hold_duration": 0.0})
        intervals_out.append({
            "t_a": ta,
            "t_b": tb,
            "tau_a": tau_a,
            "tau_b": tau_b,
            "s_a": sa,
            "s_b": sb,
            "delta_s": ds,
            "segments": iv_segs,
            "points": iv_pts,
        })

    return {
        "channel": channel,
        "mode": solve_mode,
        "total_length": total_len,
        "lock_compositions": lock_comps,
        "intervals": intervals_out,
        "segments": solved["segments"],
        "conflicts": solved["conflicts"],
        "clamped": solved.get("clamped", False),
        "achieved_edit_speed": solved.get("achieved_edit_speed"),
        "affected_range": solved.get("affected_range"),
        "cruise_infos": solved.get("cruise_infos", []),
    }


def _eval_zero_ds_tau(
    arc_table: Dict[str, Any],
    intervals: List[Dict[str, Any]],
    idx: int,
    tc: float,
    ta: float,
    tb: float,
    tau_a: float,
    tau_b: float,
) -> float:
    """在主通道位移为 0 的滞留区间内（如定点摇镜），对衔接起停边界做 C1 平滑时间映射（避免 gen 线性化撕裂反转摇镜）。"""
    dt = max(1e-6, tb - ta)
    u = max(0.0, min(1.0, (tc - ta) / dt))
    has_ang_zoom = (
        abs(_eval_arc_at_tau(arc_table, "rot", tau_b) - _eval_arc_at_tau(arc_table, "rot", tau_a)) > 1e-4
        or abs(_eval_arc_at_tau(arc_table, "zoom", tau_b) - _eval_arc_at_tau(arc_table, "zoom", tau_a)) > 1e-4
    )
    if not has_ang_zoom or tau_b - tau_a <= 1e-6:
        return tau_a + u * (tau_b - tau_a)

    def _interval_is_still(iv_check: Dict[str, Any], boundary_t: float) -> bool:
        if iv_check["delta_s"] > 1e-6:
            vb, _ = _eval_speed_segments_at_time(iv_check["segments"], boundary_t)
            return vb <= 0.05
        t0_c, t1_c = iv_check["tau_a"], iv_check["tau_b"]
        return (
            abs(_eval_arc_at_tau(arc_table, "rot", t1_c) - _eval_arc_at_tau(arc_table, "rot", t0_c)) <= 1e-4
            and abs(_eval_arc_at_tau(arc_table, "zoom", t1_c) - _eval_arc_at_tau(arc_table, "zoom", t0_c)) <= 1e-4
        )

    stop_a = True if idx == 0 else _interval_is_still(intervals[idx - 1], ta)
    stop_b = True if idx == len(intervals) - 1 else _interval_is_still(intervals[idx + 1], tb)
    da = min(0.25, 0.45 / dt) if stop_a else 0.0
    db = min(0.25, 0.45 / dt) if stop_b else 0.0
    area = 1.0 - 0.5 * da - 0.5 * db
    if da + db <= 1e-6 or area <= 1e-6:
        return tau_a + u * (tau_b - tau_a)
    if u < da:
        s = u / da
        integ = da * (s * s * s - 0.5 * s * s * s * s)
    elif u <= 1.0 - db:
        integ = 0.5 * da + (u - da)
    else:
        s = (1.0 - u) / db
        tail = db * (s * s * s - 0.5 * s * s * s * s)
        integ = area - tail
    phi = max(0.0, min(1.0, integ / area))
    return tau_a + phi * (tau_b - tau_a)


def _eval_channel_tau_and_speed(
    arc_table: Dict[str, Any],
    solved_chan: Dict[str, Any],
    t: float,
    allow_gen_fallback: bool = False,
) -> Tuple[float, float, float]:
    """返回 (tau(t), actual_speed(t), traveled_arc(t))。"""
    dur = arc_table["duration"]
    tc = max(0.0, min(dur, float(t)))
    intervals = solved_chan["intervals"]
    if not intervals:
        return tc, 0.0, 0.0
    chan = solved_chan["channel"]
    for idx, iv in enumerate(intervals):
        if tc <= iv["t_b"] + 1e-7 or idx == len(intervals) - 1:
            ta, tb = iv["t_a"], iv["t_b"]
            tau_a, tau_b = iv["tau_a"], iv["tau_b"]
            sa, ds = iv["s_a"], iv["delta_s"]
            v_raw, local_s = _eval_speed_segments_at_time(iv["segments"], tc)
            if ds <= 1e-6:
                if allow_gen_fallback:
                    return _eval_zero_ds_tau(arc_table, intervals, idx, tc, ta, tb, tau_a, tau_b), 0.0, sa
                u_t = (tc - ta) / max(1e-6, tb - ta)
                return tau_a + u_t * (tau_b - tau_a), 0.0, sa

            if local_s >= ds - 1e-7 and not solved_chan["lock_compositions"] and local_s > ds + 1e-5:
                return tau_b, 0.0, sa + ds
            s_cur = sa + min(ds, max(0.0, local_s))
            tau_cur = _arc_to_tau_in_window(arc_table, chan, s_cur, tau_a, tau_b)
            return tau_cur, v_raw, s_cur
    return dur, 0.0, solved_chan["total_length"]


def _get_solved_camera_speed_bundle(scene: Dict[str, Any]) -> Dict[str, Any]:
    sig = _build_camera_speed_curve_signature(scene)
    cached = _SPEED_CURVE_SOLVE_CACHE.get(sig)
    if cached is not None:
        _SPEED_CURVE_SOLVE_CACHE.move_to_end(sig)
        return cached
    arc_table = build_camera_arc_length_table(scene)
    curve_cfg = scene.get("camera_speed_curve") or {}
    linked = bool(curve_cfg.get("linked", True))
    solved_pos = solve_camera_speed_curve_channel(arc_table, curve_cfg, "pos")
    solved_rot = solve_camera_speed_curve_channel(arc_table, curve_cfg, "rot")
    solved_zoom = solve_camera_speed_curve_channel(arc_table, curve_cfg, "zoom")
    bundle = {
        "arc_table": arc_table,
        "linked": linked,
        "pos": solved_pos,
        "rot": solved_rot,
        "zoom": solved_zoom,
    }
    _SPEED_CURVE_SOLVE_CACHE[sig] = bundle
    while len(_SPEED_CURVE_SOLVE_CACHE) > _SPEED_CURVE_CACHE_MAX:
        _SPEED_CURVE_SOLVE_CACHE.popitem(last=False)
    return bundle


def _eval_camera_rail_state(scene: Dict[str, Any], t: float) -> Dict[str, Any]:
    """求值未叠加手持与云台滞后之前的纯净轨道摄影机状态（兼容 camera_speed_curve 开关）。"""
    curve_cfg = scene.get("camera_speed_curve")
    if not isinstance(curve_cfg, dict) or not curve_cfg.get("enabled"):
        return interpolate_camera_base(scene, t, continuous_dir=False)

    bundle = _get_solved_camera_speed_bundle(scene)
    arc_table = bundle["arc_table"]
    if bundle["linked"]:
        if arc_table["total_pos"] > 1e-5:
            tau_shared, _, _ = _eval_channel_tau_and_speed(arc_table, bundle["pos"], t, allow_gen_fallback=True)
        elif arc_table["total_rot"] > 1e-4:
            tau_shared, _, _ = _eval_channel_tau_and_speed(arc_table, bundle["rot"], t, allow_gen_fallback=True)
        elif arc_table["total_zoom"] > 1e-4:
            tau_shared, _, _ = _eval_channel_tau_and_speed(arc_table, bundle["zoom"], t, allow_gen_fallback=True)
        else:
            tau_shared = max(0.0, min(arc_table["duration"], float(t)))
        return interpolate_camera_base(scene, tau_shared, continuous_dir=True)

    tau_p, _, _ = _eval_channel_tau_and_speed(arc_table, bundle["pos"], t, allow_gen_fallback=False)
    tau_r, _, _ = _eval_channel_tau_and_speed(arc_table, bundle["rot"], t, allow_gen_fallback=False)
    tau_z, _, _ = _eval_channel_tau_and_speed(arc_table, bundle["zoom"], t, allow_gen_fallback=False)
    ev_p = interpolate_camera_base(scene, tau_p, continuous_dir=True)
    ev_r = interpolate_camera_base(scene, tau_r, continuous_dir=True)
    ev_z = interpolate_camera_base(scene, tau_z, continuous_dir=True)
    pos_out = ev_p["pos"]
    ld = ev_r["look_dir"]
    rd = max(0.05, float(ev_r["look_dist"]))
    tar_out = [pos_out[0] + ld[0] * rd, pos_out[1] + ld[1] * rd, pos_out[2] + ld[2] * rd]
    return {
        "pos": pos_out,
        "target": tar_out,
        "fov": float(ev_z["fov"]),
        "roll_deg": float(ev_r.get("roll_deg", 0.0)),
        "look_dir": ld,
        "look_dist": rd,
        "focal_mm_exact": float(ev_z["focal_mm_exact"]),
    }


_HANDHELD_HARMONIC_RATIOS = (1.0, 1.6180339, 2.7182818, 0.6180339)
_HANDHELD_HARMONIC_WEIGHTS = (0.48, 0.25, 0.12, 0.15)
_HANDHELD_AXIS_FREQ_MULS = (1.00, 1.13, 0.87, 0.94, 1.07, 0.79)


def _splitmix32_phase(seed: int, index: int) -> float:
    x = (int(seed) + int(index) * 0x9E3779B9) & 0xFFFFFFFF
    x = ((x ^ (x >> 16)) * 0x85EBCA6B) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    return (float(x) / 4294967296.0) * (2.0 * math.pi)


def _smoothstep01(x: float) -> float:
    c = max(0.0, min(1.0, float(x)))
    return c * c * (3.0 - 2.0 * c)


def evaluate_handheld_rig_offset(
    rig_cfg: Dict[str, Any],
    t: float,
    dur: float,
    speed_mps: float = 0.0,
) -> Dict[str, float]:
    """解析式确定性手持微扰叠加（0 额外关键帧，前后端任意时刻拖帧 100% 一致）。"""
    hh = (rig_cfg or {}).get("handheld") or {}
    intensity = float(hh.get("intensity", 0.0) or 0.0)
    if not hh.get("enabled") or intensity <= 1e-6:
        return {
            "dx": 0.0, "dy": 0.0, "dz": 0.0,
            "dyaw_deg": 0.0, "dpitch_deg": 0.0, "droll_deg": 0.0,
            "env": 0.0, "move_blend": 0.0,
        }

    d_tot = max(0.2, float(dur or 4.0))
    tc = max(0.0, min(d_tot, float(t)))
    if hh.get("settle_ends", True):
        t_in = min(0.32, 0.18 * d_tot)
        t_out = min(0.45, 0.22 * d_tot)
        env = _smoothstep01(tc / max(1e-4, t_in)) * _smoothstep01((d_tot - tc) / max(1e-4, t_out))
    else:
        env = 1.0

    move_blend = _smoothstep01((float(speed_mps) - 0.15) / (1.40 - 0.15))
    vel_mul = 1.0 + float(hh.get("speed_gain", 0.0)) * min(2.0, float(speed_mps) / 1.5)
    scale_amp = intensity * vel_mul * env
    seed = int(hh.get("seed", 1075))
    freq_hz = float(hh.get("freq_hz", 0.32))
    step_freq_hz = float(hh.get("step_freq_hz", 1.45))

    waves = [0.0] * 6
    for ch in range(6):
        f_base = freq_hz * _HANDHELD_AXIS_FREQ_MULS[ch]
        drift = 0.0
        for h_idx in range(4):
            phi = _splitmix32_phase(seed, ch * 8 + h_idx)
            drift += _HANDHELD_HARMONIC_WEIGHTS[h_idx] * math.sin(
                2.0 * math.pi * (f_base * _HANDHELD_HARMONIC_RATIOS[h_idx]) * tc + phi
            )
        f_step = step_freq_hz * (0.5 if ch in (0, 3, 5) else 1.0)
        phi_s0 = _splitmix32_phase(seed, ch * 8 + 4)
        phi_s1 = _splitmix32_phase(seed, ch * 8 + 5)
        step_wave = (
            0.72 * math.sin(2.0 * math.pi * f_step * tc + phi_s0)
            + 0.28 * math.sin(2.0 * math.pi * (2.03 * f_step) * tc + phi_s1)
        )
        waves[ch] = (1.0 - 0.35 * move_blend) * drift + (0.65 * move_blend) * step_wave

    roll_locked = bool((rig_cfg or {}).get("lock_roll", True))
    pos_amp = float(hh.get("pos_amp_m", 0.0))
    rot_amp = float(hh.get("rot_amp_deg", 0.0))
    roll_amp = float(hh.get("roll_amp_deg", 0.0))
    return {
        "dx": pos_amp * 0.75 * waves[0] * scale_amp,
        "dy": pos_amp * 1.00 * waves[1] * scale_amp,
        "dz": pos_amp * 0.55 * waves[2] * scale_amp,
        "dyaw_deg": rot_amp * 1.00 * waves[3] * scale_amp,
        "dpitch_deg": rot_amp * 0.85 * waves[4] * scale_amp,
        "droll_deg": 0.0 if roll_locked else (roll_amp * 1.00 * waves[5] * scale_amp),
        "env": env,
        "move_blend": move_blend,
    }


def evaluate_camera_state(scene: Dict[str, Any], t: float) -> Dict[str, Any]:
    """
    摄影机完整求值管线 v3（轨道样条 + 速度曲线 + 云台滞后/制动过冲 + 确定性手持微扰 + Roll）：
    返回 {"pos", "target", "fov", "roll_deg", "look_dir", "look_dist", "focal_mm_exact", "rail_speed_mps", "handheld_offset"}。
    """
    dur = max(0.5, float(scene.get("duration") or 4.0))
    tc = max(0.0, min(dur, float(t)))
    base = _eval_camera_rail_state(scene, tc)
    rig_cfg = normalize_camera_rig_config(scene.get("camera_rig"))
    roll_locked = bool(rig_cfg.get("lock_roll", True))

    pos_base = [float(x) for x in base["pos"][:3]]
    fov_out = float(base["fov"])
    roll_base = 0.0 if roll_locked else float(base.get("roll_deg", 0.0))
    look_dir = [float(x) for x in base["look_dir"][:3]]
    look_dist = max(0.05, float(base["look_dist"]))

    t_lo = max(0.0, tc - 0.04)
    t_hi = min(dur, tc + 0.04)
    if t_hi - t_lo > 1e-4:
        p_lo = _eval_camera_rail_state(scene, t_lo)["pos"]
        p_hi = _eval_camera_rail_state(scene, t_hi)["pos"]
        speed_mps = math.hypot(p_hi[0] - p_lo[0], p_hi[1] - p_lo[1], p_hi[2] - p_lo[2]) / (t_hi - t_lo)
    else:
        speed_mps = 0.0

    aim_lag = float(rig_cfg.get("aim_lag_sec", 0.0))
    aim_ov = float(rig_cfg.get("aim_overshoot", 0.0))
    if aim_lag > 1e-5 and tc > 1e-5:
        settle_end_mul = 1.0
        if rig_cfg["handheld"].get("settle_ends", True):
            t_end_s = min(0.45, 0.22 * dur)
            settle_end_mul = _smoothstep01((dur - tc) / max(1e-4, t_end_s))
        taps = (0.0, 0.25, 0.55, 0.90, 1.35)
        wts = (0.34, 0.26, 0.19, 0.13, 0.08)
        sx, sy, sz = 0.0, 0.0, 0.0
        for k_t in range(5):
            tk = max(0.0, min(dur, tc - taps[k_t] * aim_lag))
            dk = _eval_camera_rail_state(scene, tk)["look_dir"]
            sx += float(dk[0]) * wts[k_t]
            sy += float(dk[1]) * wts[k_t]
            sz += float(dk[2]) * wts[k_t]
        sn = math.hypot(sx, sy, sz)
        lag_dir = [sx / sn, sy / sn, sz / sn] if sn > 1e-6 else list(look_dir)

        if aim_ov > 1e-5:
            dt_v = max(0.03, min(0.12, 0.35 * aim_lag + 0.04))
            d_m2 = _eval_camera_rail_state(scene, max(0.0, tc - 2.0 * dt_v))["look_dir"]
            d_m1 = _eval_camera_rail_state(scene, max(0.0, tc - 1.0 * dt_v))["look_dir"]
            raw_ov = [((d_m1[c] - d_m2[c]) - (look_dir[c] - d_m1[c])) / dt_v * aim_ov * 0.45 for c in range(3)]
            proj = raw_ov[0] * lag_dir[0] + raw_ov[1] * lag_dir[1] + raw_ov[2] * lag_dir[2]
            raw_ov = [raw_ov[c] - proj * lag_dir[c] for c in range(3)]
            ov_n = math.hypot(raw_ov[0], raw_ov[1], raw_ov[2])
            if ov_n > 1e-7:
                ov_cap = 0.08 * math.tanh(ov_n / 0.08)
                sc_ov = ov_cap / ov_n
                lag_dir = [lag_dir[c] + raw_ov[c] * sc_ov for c in range(3)]
                ln2 = math.hypot(lag_dir[0], lag_dir[1], lag_dir[2])
                if ln2 > 1e-6:
                    lag_dir = [lag_dir[0] / ln2, lag_dir[1] / ln2, lag_dir[2] / ln2]

        if settle_end_mul < 0.999999:
            bl = [look_dir[c] * (1.0 - settle_end_mul) + lag_dir[c] * settle_end_mul for c in range(3)]
            bn = math.hypot(bl[0], bl[1], bl[2])
            look_dir = [bl[0] / bn, bl[1] / bn, bl[2] / bn] if bn > 1e-6 else look_dir
        else:
            look_dir = lag_dir

    hh_off = evaluate_handheld_rig_offset(rig_cfg, tc, dur, speed_mps)
    tar_lag = [pos_base[c] + look_dir[c] * look_dist for c in range(3)]
    has_hh = (
        abs(hh_off["dx"]) > 1e-8
        or abs(hh_off["dy"]) > 1e-8
        or abs(hh_off["dz"]) > 1e-8
        or abs(hh_off["dyaw_deg"]) > 1e-8
        or abs(hh_off["dpitch_deg"]) > 1e-8
        or abs(hh_off["droll_deg"]) > 1e-8
    )
    if has_hh:
        right_v, up_v, fwd_v = compute_camera_basis(pos_base, tar_lag, roll_base)
        pos_out = [
            float(pos_base[c] + right_v[c] * hh_off["dx"] + up_v[c] * hh_off["dy"] + fwd_v[c] * hh_off["dz"])
            for c in range(3)
        ]
        yaw_r = math.radians(hh_off["dyaw_deg"])
        pitch_r = math.radians(hh_off["dpitch_deg"])
        cy, sy_r = math.cos(yaw_r), math.sin(yaw_r)
        cp, sp = math.cos(pitch_r), math.sin(pitch_r)
        f1 = fwd_v * cy + right_v * sy_r
        f2 = f1 * cp + up_v * sp
        fn = float(np.linalg.norm(f2))
        dir_final = [float(f2[0] / fn), float(f2[1] / fn), float(f2[2] / fn)] if fn > 1e-6 else list(look_dir)
        tar_out = [pos_out[c] + dir_final[c] * look_dist for c in range(3)]
        roll_out = 0.0 if roll_locked else (roll_base + hh_off["droll_deg"])
    else:
        pos_out = pos_base
        tar_out = tar_lag
        dir_final = look_dir
        roll_out = 0.0 if roll_locked else roll_base

    return {
        "pos": pos_out,
        "target": tar_out,
        "fov": fov_out,
        "roll_deg": roll_out,
        "look_dir": dir_final,
        "look_dist": look_dist,
        "focal_mm_exact": float(base["focal_mm_exact"]),
        "rail_speed_mps": speed_mps,
        "handheld_offset": hh_off,
    }


def _eval_camera_at_time_exact(scene: Dict[str, Any], t: float) -> Dict[str, Any]:
    return evaluate_camera_state(scene, t)


def interpolate_camera(scene: Dict[str, Any], t: float) -> Tuple[List[float], List[float], float]:
    """
    根据时间 t 插值摄影机位置 pos、焦点 target 与垂直视场角 fov。
    统一经由 evaluate_camera_state（轨道样条 + 速度曲线 + 云台滞后 + 确定性手持）。
    """
    st = evaluate_camera_state(scene, t)
    return st["pos"], st["target"], st["fov"]


def is_object_manually_visible(obj: Dict[str, Any]) -> bool:
    """成片手动可见性：仅读取持久字段 hidden / visible，绝不读取前端缓存 _kfVisible。"""
    if not isinstance(obj, dict):
        return True
    if obj.get("hidden") is True:
        return False
    if obj.get("visible") is False:
        return False
    return True


def eval_keyframe_step_visible(keyframes: Optional[List[Dict[str, Any]]], t: float) -> bool:
    """时间轴关键帧可见性阶跃求值（不作布尔渐变）：
    - 无显式布尔 visible 关键帧：返回 True；
    - 首帧前（t < 第一个含布尔 visible 关键帧的时间）：返回 True（尚未到达显式隐藏点）；
    - 帧间与末帧后：取 <= t 的最近一个显式布尔 visible 值（中间不含 visible 字段的关键帧保持上一段阶跃值）；
    - 纯函数无状态，正反向 scrub 结果完全一致。
    """
    if not isinstance(keyframes, list) or not keyframes:
        return True
    indexed_vis: List[Tuple[float, int, bool]] = []
    for idx, kf in enumerate(keyframes):
        if isinstance(kf, dict) and isinstance(kf.get("visible"), bool):
            indexed_vis.append((float(kf.get("t", 0.0)), idx, bool(kf["visible"])))
    if not indexed_vis:
        return True
    indexed_vis.sort(key=lambda item: (item[0], item[1]))
    cur_val: Optional[bool] = None
    t_val = float(t)
    for kt, _idx, kv in indexed_vis:
        if kt <= t_val + 1e-6:
            cur_val = kv
        else:
            break
    return True if cur_val is None else cur_val


def _apply_procedural_walk_gait(
    res: Dict[str, Any],
    k0: Dict[str, Any],
    k1: Dict[str, Any],
    r0: float,
    r1: float,
    s_pos: float,
) -> None:
    """在稀疏关键帧区间 [k0, k1] 上合成世界空间锁脚不滑步步态（先落脚再移重心、末步自然收步跟进）。"""
    n_steps = int(k0.get("walk_steps") or 0)
    if n_steps < 1:
        return
    if res.get("pawn_smooth_walk"):
        # Root interpolation remains authoritative. End envelope is C1 continuous.
        u = max(0.0, min(1.0, float(s_pos)))
        envelope = math.sin(math.pi * u) ** 2
        bob = 0.0035 * envelope * (math.sin(math.pi * n_steps * u) ** 2)
        sway = 0.002 * envelope * math.sin(math.pi * n_steps * u)
        for name in ("pelvis", "chest"):
            if name in res.get("targets", {}):
                res["targets"][name][1] += bob
        if "chest" in res.get("targets", {}):
            res["targets"]["chest"][0] += sway
        return
    lead = "R" if k0.get("lead_foot") == "R" else "L"
    sx, sy, sz = _get_char_scale_xyz(res)
    p0 = [float(x) for x in k0["pos"][:3]]
    p1 = [float(x) for x in k1["pos"][:3]]

    def _loc_to_w(pr: List[float], yaw_deg: float, loc: List[float]) -> List[float]:
        yr = math.radians(yaw_deg)
        cy, sy_a = math.cos(yr), math.sin(yr)
        lx, ly, lz = float(loc[0]) * sx, float(loc[1]) * sy, float(loc[2]) * sz
        return [pr[0] + lx * cy + lz * sy_a, pr[1] + ly, pr[2] - lx * sy_a + lz * cy]

    def _w_to_loc(pr: List[float], yaw_deg: float, wpt: List[float]) -> List[float]:
        yr = math.radians(yaw_deg)
        cy, sy_a = math.cos(yr), math.sin(yr)
        dx, dy, dz = wpt[0] - pr[0], wpt[1] - pr[1], wpt[2] - pr[2]
        return [
            (dx * cy - dz * sy_a) / max(1e-6, sx),
            dy / max(1e-6, sy),
            (dx * sy_a + dz * cy) / max(1e-6, sz),
        ]

    t0_map = k0.get("targets") or {}
    t1_map = k1.get("targets") or {}
    fL_0 = list(t0_map.get("foot_L") or [0.10, 0.04, 0.0])
    fR_0 = list(t0_map.get("foot_R") or [-0.10, 0.04, 0.0])
    fL_end = list(t1_map.get("foot_L") or [0.10, 0.04, 0.0])
    fR_end = list(t1_map.get("foot_R") or [-0.10, 0.04, 0.0])
    closing_stop = abs(float(fL_end[2]) - float(fR_end[2])) < 0.09 and n_steps >= 2

    cur_WL = _loc_to_w(p0, r0, fL_0)
    cur_WR = _loc_to_w(p0, r0, fR_0)
    WL_final = _loc_to_w(p1, r1, fL_end)
    WR_final = _loc_to_w(p1, r1, fR_end)

    steps_plan: List[Tuple[bool, List[float], List[float], List[float], float, float]] = []
    denom = float(max(1, n_steps - 1))
    for m in range(n_steps):
        swing_is_L = (m % 2 == 0) if lead == "L" else (m % 2 == 1)
        if m >= n_steps - 2:
            land_W = list(WL_final if swing_is_L else WR_final)
        else:
            sm = min(1.0, float(m + 1) / denom)
            pm = [p0[j] * (1.0 - sm) + p1[j] * sm for j in range(3)]
            ym = lerp_angle_deg(r0, r1, sm)
            lat = [0.105, 0.04, 0.0] if swing_is_L else [-0.105, 0.04, 0.0]
            land_W = _loc_to_w(pm, ym, lat)

        sr0 = 0.0 if m == 0 else (min(1.0, (n_steps - 0.32) / float(n_steps)) if (closing_stop and m == n_steps - 1) else float(m) / float(n_steps))
        sr1 = 1.0 if m == n_steps - 1 else (min(1.0, (n_steps - 0.32) / float(n_steps)) if (closing_stop and m == n_steps - 2) else float(m + 1) / float(n_steps))
        start_W = list(cur_WL if swing_is_L else cur_WR)
        stance_W = list(cur_WR if swing_is_L else cur_WL)
        steps_plan.append((swing_is_L, start_W, land_W, stance_W, sr0, sr1))
        if swing_is_L:
            cur_WL = land_W
        else:
            cur_WR = land_W

    u_tot = max(0.0, min(float(n_steps), float(s_pos) * float(n_steps)))
    m_idx = min(n_steps - 1, int(math.floor(u_tot)))
    u = max(0.0, min(1.0, u_tot - float(m_idx)))
    swing_is_L, start_W, land_W, stance_W, sr0, sr1 = steps_plan[m_idx]

    # 先落脚 (u=0..0.68 迈步落稳) -> 再移重心 (u=0.42..1.0 重心平稳过渡至前脚)
    if u < 0.68:
        us = u / 0.68
        ss = us * us * (3.0 - 2.0 * us)
        lift = 0.062 * sy * math.sin(math.pi * us)
        swing_W = [
            start_W[0] * (1.0 - ss) + land_W[0] * ss,
            start_W[1] * (1.0 - ss) + land_W[1] * ss + lift,
            start_W[2] * (1.0 - ss) + land_W[2] * ss,
        ]
    else:
        swing_W = list(land_W)

    if u < 0.42:
        wu = 0.26 * ((u / 0.42) ** 2)
    else:
        vu = (u - 0.42) / 0.58
        wu = 0.26 + 0.74 * (vu * vu * (3.0 - 2.0 * vu))
    s_root = sr0 + (sr1 - sr0) * wu
    res["pos"] = [p0[j] * (1.0 - s_root) + p1[j] * s_root for j in range(3)]

    cur_yaw = float(res["rot"][1])
    wL_now = swing_W if swing_is_L else stance_W
    wR_now = stance_W if swing_is_L else swing_W
    res["targets"]["foot_L"] = _w_to_loc(res["pos"], cur_yaw, wL_now)
    res["targets"]["foot_R"] = _w_to_loc(res["pos"], cur_yaw, wR_now)

    # 轻微重心起伏、支撑脚横向移重心与自然摆臂（兼容球头人偶与关节人偶）
    bob = (-0.016 * math.sin(math.pi * ((u - 0.56) / 0.44))) if u > 0.56 else (0.013 * math.sin(math.pi * (u / 0.56)))
    stance_sign = -1.0 if swing_is_L else 1.0
    lat_sway = stance_sign * (0.014 if (closing_stop and m_idx == n_steps - 1) else 0.024) * math.sin(math.pi * u)
    fwd_lean = 0.016 * math.sin(math.pi * max(0.0, min(1.0, float(s_pos))))
    if "pelvis" in res["targets"]:
        res["targets"]["pelvis"][1] = float(res["targets"]["pelvis"][1]) + bob
    if "chest" in res["targets"]:
        res["targets"]["chest"][0] = float(res["targets"]["chest"][0]) + lat_sway
        res["targets"]["chest"][1] = float(res["targets"]["chest"][1]) + bob
        res["targets"]["chest"][2] = float(res["targets"]["chest"][2]) + fwd_lean
    arm_sw = math.sin(math.pi * u) * (0.045 if (closing_stop and m_idx == n_steps - 1) else 0.075)
    sgn = 1.0 if swing_is_L else -1.0
    if "hand_R" in res["targets"]:
        res["targets"]["hand_R"][2] = float(res["targets"]["hand_R"][2]) + sgn * arm_sw
    if "hand_L" in res["targets"]:
        res["targets"]["hand_L"][2] = float(res["targets"]["hand_L"][2]) - sgn * arm_sw


def _apply_idle_breathing(res: Dict[str, Any], obj: Dict[str, Any], t: float) -> None:
    """不同步细微自然呼吸（兼容球头人偶与关节人偶，躯干微起伏与球头轻微呼吸律动）。"""
    ph = obj.get("idle_breath_phase")
    if ph is None:
        return
    w = math.sin(float(t) * 1.55 + float(ph))
    breath_y = 0.0055 * w
    breath_z = 0.0085 * w
    targs = res.get("targets") or {}
    if "pelvis" in targs:
        targs["pelvis"][1] = float(targs["pelvis"][1]) + breath_y * 0.55
    if "chest" in targs:
        targs["chest"][1] = float(targs["chest"][1]) + breath_y
        targs["chest"][2] = float(targs["chest"][2]) + breath_z
    if "head_target" in targs:
        targs["head_target"][1] = float(targs["head_target"][1]) + 0.014 * w
    if "shoulder_L_offset" not in targs:
        targs["shoulder_L_offset"] = [0.0, breath_y * 0.65, 0.0]
    else:
        targs["shoulder_L_offset"][1] = float(targs["shoulder_L_offset"][1]) + breath_y * 0.65
    if "shoulder_R_offset" not in targs:
        targs["shoulder_R_offset"] = [0.0, breath_y * 0.65, 0.0]
    else:
        targs["shoulder_R_offset"][1] = float(targs["shoulder_R_offset"][1]) + breath_y * 0.65


_TRACK_CONTINUITY_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_TRACK_CONTINUITY_CACHE_MAX = 64


def _get_cached_track_continuity_plan(obj: Dict[str, Any], scene: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    anc_chain = []
    if isinstance(scene, dict):
        obj_map = {
            str(o.get("id")): o
            for o in (scene.get("objects") or [])
            if isinstance(o, dict) and o.get("id")
        }
        cur_p = obj
        seen_anc = {str(obj.get("id") or "")}
        for _ in range(12):
            pid = str(cur_p.get("parent_id") or "").strip() if isinstance(cur_p, dict) else ""
            if not pid or pid in seen_anc:
                break
            seen_anc.add(pid)
            p_obj = obj_map.get(pid)
            if not p_obj:
                break
            anc_chain.append({
                "id": pid,
                "pos": p_obj.get("pos"),
                "rot": p_obj.get("rot"),
                "size": p_obj.get("size"),
                "lr": p_obj.get("layout_ref"),
                "kfs": p_obj.get("keyframes"),
            })
            cur_p = p_obj
    sig_payload = {
        "type": obj.get("type"),
        "motion_type": obj.get("motion_type"),
        "pos": obj.get("pos"),
        "rot": obj.get("rot"),
        "parent_id": obj.get("parent_id"),
        "layout_ref": obj.get("layout_ref"),
        "mounted_vehicle_id": obj.get("mounted_vehicle_id"),
        "follow_mounted_vehicle": obj.get("follow_mounted_vehicle"),
        "ancestors": anc_chain,
        "mp": obj.get("motion_protection"),
        "se": obj.get("speed_exception"),
        "kfs": obj.get("keyframes"),
        "dur": (scene or {}).get("duration") if scene else None,
    }
    sig = json.dumps(sig_payload, sort_keys=True, ensure_ascii=False, default=str)
    cached = _TRACK_CONTINUITY_CACHE.get(sig)
    if cached is not None:
        _TRACK_CONTINUITY_CACHE.move_to_end(sig)
        return cached
    plan = build_track_continuity_plan(obj, scene)
    _TRACK_CONTINUITY_CACHE[sig] = plan
    while len(_TRACK_CONTINUITY_CACHE) > _TRACK_CONTINUITY_CACHE_MAX:
        _TRACK_CONTINUITY_CACHE.popitem(last=False)
    return plan


def _phase_warp_c1(u: float, delta: float) -> float:
    """闭式 C1 错峰相位偏移：endpoints (u=0, 1) 值与导数完全保持不变，中段按 delta 前移或滞后。"""
    uc = max(0.0, min(1.0, float(u)))
    if abs(delta) < 1e-6 or uc <= 0.0 or uc >= 1.0:
        return uc
    um = 1.0 - uc
    return max(0.0, min(1.0, uc + float(delta) * 16.0 * uc * uc * um * um))


def _slerp_unit3(u0: np.ndarray, u1: np.ndarray, s: float, fallback_axis: np.ndarray) -> np.ndarray:
    """单位方向球面插值（180° 反向时自动沿解剖外侧摆动平面旋转，防止穿头/穿胸）。"""
    sc = max(0.0, min(1.0, float(s)))
    if sc <= 0.0:
        return u0
    if sc >= 1.0:
        return u1
    d = max(-1.0, min(1.0, float(np.dot(u0, u1))))
    if d > 0.9999:
        return _normalize(u0 * (1.0 - sc) + u1 * sc)
    if d < -0.96:
        ax = np.cross(u0, fallback_axis)
        if float(np.linalg.norm(ax)) < 1e-4:
            alt = np.array([0.0, 0.0, 1.0], dtype=np.float64) if abs(u0[2]) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
            ax = np.cross(u0, alt)
        ax = _normalize(ax)
        ang = sc * math.acos(d)
        ca, sa = math.cos(ang), math.sin(ang)
        return _normalize(u0 * ca + np.cross(ax, u0) * sa + ax * float(np.dot(ax, u0)) * (1.0 - ca))
    theta = math.acos(d)
    sin_t = math.sin(theta)
    w0 = math.sin((1.0 - sc) * theta) / sin_t
    w1 = math.sin(sc * theta) / sin_t
    return _normalize(u0 * w0 + u1 * w1)


def _eval_dir_progress_o1(
    kfs: List[Dict[str, Any]],
    times: List[float],
    dirs: List[np.ndarray],
    modes: List[str],
    seg_idx: int,
    u_warped: float,
) -> float:
    """O(1) 四点局部单调角速度 Hermite 进度求解器（连续过渡不急停，反转顶点平滑减速）。"""
    mode_i = modes[seg_idx]
    if mode_i == "hold":
        return 0.0 if u_warped < 0.999999 else 1.0
    if mode_i != "continuous":
        return eval_easing(u_warped, mode_i)

    n = len(dirs)
    i = seg_idx
    h_i = float(times[i + 1]) - float(times[i])
    theta_i = math.acos(max(-1.0, min(1.0, float(np.dot(dirs[i], dirs[i + 1])))))
    if theta_i < 1e-4 or h_i <= 1e-4:
        return u_warped

    w_i = theta_i / h_i
    m0 = 0.0
    if i > 0 and not (kfs[i].get("stop") or kfs[i].get("pause")) and modes[i - 1] not in ("hold", "ease_out", "ease_in_out"):
        h_p = float(times[i]) - float(times[i - 1])
        th_p = math.acos(max(-1.0, min(1.0, float(np.dot(dirs[i - 1], dirs[i])))))
        if th_p >= 1e-4 and h_p > 1e-4:
            if float(np.dot(dirs[i] - dirs[i - 1], dirs[i + 1] - dirs[i])) > 0.0:
                w_p = th_p / h_p
                v_mid = min((h_i * w_p + h_p * w_i) / (h_p + h_i), 2.0 * w_p, 2.0 * w_i)
                m0 = max(0.0, min(2.0, (v_mid * h_i) / theta_i))

    m1 = 0.0
    if i + 1 < n - 1 and not (kfs[i + 1].get("stop") or kfs[i + 1].get("pause")) and modes[i + 1] not in ("hold", "ease_in", "ease_in_out"):
        h_n = float(times[i + 2]) - float(times[i + 1])
        th_n = math.acos(max(-1.0, min(1.0, float(np.dot(dirs[i + 1], dirs[i + 2])))))
        if th_n >= 1e-4 and h_n > 1e-4:
            if float(np.dot(dirs[i + 1] - dirs[i], dirs[i + 2] - dirs[i + 1])) > 0.0:
                w_n = th_n / h_n
                v_mid = min((h_n * w_i + h_i * w_n) / (h_i + h_n), 2.0 * w_i, 2.0 * w_n)
                m1 = max(0.0, min(2.0, (v_mid * h_i) / theta_i))

    u = max(0.0, min(1.0, float(u_warped)))
    u2 = u * u
    u3 = u2 * u
    s = (u3 - 2.0 * u2 + u) * m0 + (-2.0 * u3 + 3.0 * u2) + (u3 - u2) * m1
    return max(0.0, min(1.0, s))


def _eval_mannequin_natural_pose_segment(
    res: Dict[str, Any],
    obj: Dict[str, Any],
    kfs: List[Dict[str, Any]],
    seg_idx: int,
    alpha: float,
    base_targets: Dict[str, List[float]],
) -> None:
    """
    仅针对关节人偶（body_style != 'pawn'）的 O(1) 闭式自然肢体插值：
    - 保持多帧 C1 单调连续与端点精准对齐
    - 躯干/视线领先、四肢滞后跟进的闭式错峰相位
    - 绕肩/髋/头的球面弧线插值 + 躯干避让 + 迈步抬脚弧线 + 自动锁骨协同
    """
    n = len(kfs)
    i = max(0, min(n - 2, int(seg_idx)))
    a = max(0.0, min(1.0, float(alpha)))
    k0, k1 = kfs[i], kfs[i + 1]
    times = [float(k.get("t", 0.0)) for k in kfs]
    modes_pose = [normalize_cam_interp_mode(k.get("interp_pose") or k.get("interp") or "linear") for k in kfs]
    mode_i = modes_pose[i]
    is_hold = (mode_i == "hold")
    if is_hold:
        s_snap = 0.0 if a < 0.999999 else 1.0
        a = s_snap

    pins = obj.get("pins") if isinstance(obj.get("pins"), dict) else {}
    stand_fb = PRESET_IK_POSES["standing"]["targets"]

    def _kf_pt(idx: int, key: str) -> np.ndarray:
        kf = kfs[idx]
        tmap = kf.get("targets") or {}
        if key == "head_target":
            raw = tmap.get("head_target") or kf.get("look_at") or base_targets.get("head_target") or stand_fb["head_target"]
        elif key in ("shoulder_L_offset", "shoulder_R_offset"):
            raw = tmap.get(key) or base_targets.get(key) or [0.0, 0.0, 0.0]
        else:
            raw = tmap.get(key) or base_targets.get(key) or stand_fb.get(key, [0.0, 0.0, 0.0])
        return np.array([float(raw[0]), float(raw[1]), float(raw[2])], dtype=np.float64)

    def _eval_scalar_list(vals: List[float], u_w: float) -> float:
        if is_hold:
            return vals[i] if a < 0.999999 else vals[i + 1]
        if mode_i == "continuous":
            return _eval_camera_hermite_channel(kfs, times, vals, modes_pose, i, u_w, default_endpoint_stop=True)
        s = eval_easing(u_w, mode_i)
        return vals[i] * (1.0 - s) + vals[i + 1] * s

    def _eval_vec_scalar(vecs: List[np.ndarray], u_w: float) -> np.ndarray:
        return np.array([
            _eval_scalar_list([float(v[c]) for v in vecs], u_w)
            for c in range(3)
        ], dtype=np.float64)

    u_pelvis = _phase_warp_c1(a, 0.05 if not is_hold else 0.0)
    u_chest = _phase_warp_c1(a, 0.07 if not is_hold else 0.0)
    u_head = _phase_warp_c1(a, 0.04 if not is_hold else 0.0)
    u_arm_L = _phase_warp_c1(a, 0.0 if (is_hold or pins.get("hand_L")) else -0.06)
    u_arm_R = _phase_warp_c1(a, 0.0 if (is_hold or pins.get("hand_R")) else -0.06)
    u_leg_L = _phase_warp_c1(a, 0.0 if (is_hold or pins.get("foot_L")) else -0.04)
    u_leg_R = _phase_warp_c1(a, 0.0 if (is_hold or pins.get("foot_R")) else -0.04)

    # 1. Pelvis & Torso Twist & Shoulder Offsets
    pelvis_list = [_kf_pt(k, "pelvis") for k in range(n)]
    p_pelvis = _eval_vec_scalar(pelvis_list, u_pelvis)
    res["targets"]["pelvis"] = [float(p_pelvis[0]), float(p_pelvis[1]), float(p_pelvis[2])]

    base_tw = float(obj.get("torso_twist", 0.0) or 0.0)
    tw_list = [max(-45.0, min(45.0, float(kfs[k].get("torso_twist", base_tw) or 0.0))) for k in range(n)]
    twist_cur = _eval_scalar_list(tw_list, u_chest)
    res["torso_twist"] = float(twist_cur)

    sh_off_L_list = [_kf_pt(k, "shoulder_L_offset") for k in range(n)]
    sh_off_R_list = [_kf_pt(k, "shoulder_R_offset") for k in range(n)]
    off_L_cur = _eval_vec_scalar(sh_off_L_list, u_chest)
    off_R_cur = _eval_vec_scalar(sh_off_R_list, u_chest)

    # 2. Chest (spherical around pelvis to preserve spine length)
    chest_raw_list = [_kf_pt(k, "chest") for k in range(n)]
    sp_vecs = [chest_raw_list[k] - pelvis_list[k] for k in range(n)]
    sp_lens = [max(0.15, float(np.linalg.norm(sv))) for sv in sp_vecs]
    sp_dirs = [sp_vecs[k] / sp_lens[k] if float(np.linalg.norm(sp_vecs[k])) > 1e-5 else np.array([0.0, 1.0, 0.0]) for k in range(n)]
    s_sp = _eval_dir_progress_o1(kfs, times, sp_dirs, modes_pose, i, u_chest)
    sp_dir_cur = _slerp_unit3(sp_dirs[i], sp_dirs[i + 1], s_sp, np.array([0.0, 0.0, 1.0]))
    sp_len_cur = _eval_scalar_list(sp_lens, u_chest)
    p_chest_cur = p_pelvis + sp_dir_cur * sp_len_cur
    res["targets"]["chest"] = [float(p_chest_cur[0]), float(p_chest_cur[1]), float(p_chest_cur[2])]

    # Helper to compute local shoulder & hip frames at any keyframe k or current state
    spine_len = MANNEQUIN_SPEC["spine_len"]
    sh_w = MANNEQUIN_SPEC["shoulder_half_width"]
    hip_w = MANNEQUIN_SPEC["hip_half_width"]

    def _frame_from_spine(pv: np.ndarray, sp_d: np.ndarray, tw_deg: float, off_l: np.ndarray, off_r: np.ndarray):
        s_l = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        s_l = _normalize(s_l - float(np.dot(s_l, sp_d)) * sp_d)
        if abs(tw_deg) > 1e-6:
            th = math.radians(tw_deg)
            ct, st = math.cos(th), math.sin(th)
            s_tw = _normalize(s_l * ct + np.cross(sp_d, s_l) * st + sp_d * float(np.dot(sp_d, s_l)) * (1.0 - ct))
        else:
            s_tw = s_l
        f_tw = _normalize(np.cross(s_tw, sp_d))
        stern = pv + sp_d * (spine_len - 0.04)
        sh_l = stern + sp_d * off_l[1] + s_tw * (sh_w + off_l[0]) + f_tw * off_l[2]
        sh_r = stern + sp_d * off_r[1] + s_tw * (-sh_w + off_r[0]) + f_tw * off_r[2]
        hp_l = pv + s_l * hip_w
        hp_r = pv - s_l * hip_w
        hc = pv + sp_d * (spine_len + 0.14)
        return sh_l, sh_r, hp_l, hp_r, hc, s_tw, f_tw

    frames_k = [_frame_from_spine(pelvis_list[k], sp_dirs[k], tw_list[k], sh_off_L_list[k], sh_off_R_list[k]) for k in range(n)]
    sh_L_cur, sh_R_cur, hp_L_cur, hp_R_cur, hc_cur, side_tw_cur, fwd_tw_cur = _frame_from_spine(
        p_pelvis, sp_dir_cur, twist_cur, off_L_cur, off_R_cur
    )

    # 3. Head Target (spherical gaze around head_center)
    ht_raw = [_kf_pt(k, "head_target") for k in range(n)]
    ht_vecs = [ht_raw[k] - frames_k[k][4] for k in range(n)]
    ht_lens = [max(0.25, float(np.linalg.norm(v))) for v in ht_vecs]
    ht_dirs = [ht_vecs[k] / float(np.linalg.norm(ht_vecs[k])) if float(np.linalg.norm(ht_vecs[k])) > 1e-5 else np.array([0.0, 0.0, 1.0]) for k in range(n)]
    s_hd = _eval_dir_progress_o1(kfs, times, ht_dirs, modes_pose, i, u_head)
    ht_dir_cur = _slerp_unit3(ht_dirs[i], ht_dirs[i + 1], s_hd, np.array([0.0, 1.0, 0.0]))
    ht_len_cur = _eval_scalar_list(ht_lens, u_head)
    ht_cur = hc_cur + ht_dir_cur * ht_len_cur
    res["targets"]["head_target"] = [float(ht_cur[0]), float(ht_cur[1]), float(ht_cur[2])]

    # 4. Arms (spherical around shoulder + torso avoidance + elbow pole + auto clavicle assist)
    torso_mid = p_pelvis + sp_dir_cur * 0.25

    def _in_torso_core(pt: np.ndarray, pv_k: np.ndarray, sp_k: np.ndarray, stw_k: np.ndarray, ftw_k: np.ndarray) -> bool:
        d = pt - (pv_k + sp_k * 0.25)
        lat = abs(float(np.dot(d, stw_k)))
        fwd = float(np.dot(d, ftw_k))
        vert = float(np.dot(d, sp_k))
        return lat < 0.165 and (-0.13 < fwd < 0.125) and (-0.23 < vert < 0.26)

    for side, is_left, sh_idx, u_arm in (("L", True, 0, u_arm_L), ("R", False, 1, u_arm_R)):
        hand_key = f"hand_{side}"
        pole_key = f"elbow_{side}_pole"
        sh_cur = sh_L_cur if is_left else sh_R_cur
        side_out = side_tw_cur * (1.0 if is_left else -1.0)

        h_raw = [_kf_pt(k, hand_key) for k in range(n)]
        r_vecs = [h_raw[k] - frames_k[k][sh_idx] for k in range(n)]
        r_lens = [max(0.12, float(np.linalg.norm(v))) for v in r_vecs]
        r_dirs = [r_vecs[k] / float(np.linalg.norm(r_vecs[k])) if float(np.linalg.norm(r_vecs[k])) > 1e-5 else np.array([0.0, -1.0, 0.0]) for k in range(n)]
        s_arm = _eval_dir_progress_o1(kfs, times, r_dirs, modes_pose, i, u_arm)
        dir_cur = _slerp_unit3(r_dirs[i], r_dirs[i + 1], s_arm, side_out)
        len_cur = _eval_scalar_list(r_lens, u_arm)
        hand_cur = sh_cur + dir_cur * len_cur

        # Smooth C1 torso collision pushout along lateral outward / forward arc (zero at boundary and endpoints)
        if not is_hold and (0.0 < a < 1.0):
            in0 = _in_torso_core(h_raw[i], pelvis_list[i], sp_dirs[i], frames_k[i][5], frames_k[i][6])
            in1 = _in_torso_core(h_raw[i + 1], pelvis_list[i + 1], sp_dirs[i + 1], frames_k[i + 1][5], frames_k[i + 1][6])
            if not in0 and not in1:
                rel_t = hand_cur - torso_mid
                lat = float(np.dot(rel_t, side_tw_cur))
                fwd = float(np.dot(rel_t, fwd_tw_cur))
                vert = float(np.dot(rel_t, sp_dir_cur))
                pen_lat = max(0.0, 0.175 - abs(lat))
                pen_fwd = max(0.0, min(0.13 - fwd, fwd + 0.14))
                pen_vert = max(0.0, min(0.27 - vert, vert + 0.24))
                if pen_lat > 0.0 and pen_fwd > 0.0 and pen_vert > 0.0:
                    w_arc = math.sin(math.pi * a) ** 2
                    pen_mag = min(pen_lat, pen_fwd, pen_vert) * w_arc
                    push_dir = side_out if (lat * (1.0 if is_left else -1.0) >= -0.05) else fwd_tw_cur
                    hand_cur = hand_cur + _normalize(push_dir * 0.78 + fwd_tw_cur * 0.42) * pen_mag

        res["targets"][hand_key] = [float(hand_cur[0]), float(hand_cur[1]), float(hand_cur[2])]

        # Elbow pole spherical follow around shoulder
        ep_raw = [_kf_pt(k, pole_key) for k in range(n)]
        ep_vecs = [ep_raw[k] - frames_k[k][sh_idx] for k in range(n)]
        ep_lens = [max(0.15, float(np.linalg.norm(v))) for v in ep_vecs]
        ep_dirs = [ep_vecs[k] / float(np.linalg.norm(ep_vecs[k])) if float(np.linalg.norm(ep_vecs[k])) > 1e-5 else _normalize(side_out - sp_dir_cur) for k in range(n)]
        ep_dir_cur = _slerp_unit3(ep_dirs[i], ep_dirs[i + 1], s_arm, side_out)
        ep_len_cur = _eval_scalar_list(ep_lens, u_arm)
        ep_cur = sh_cur + ep_dir_cur * ep_len_cur
        res["targets"][pole_key] = [float(ep_cur[0]), float(ep_cur[1]), float(ep_cur[2])]

        sh_key = f"shoulder_{side}_offset"
        t0_sh = (k0.get("targets") or {}).get(sh_key)
        t1_sh = (k1.get("targets") or {}).get(sh_key)
        if t0_sh is not None or t1_sh is not None:
            base_off = off_L_cur if is_left else off_R_cur
            res["targets"][sh_key] = [float(base_off[0]), float(base_off[1]), float(base_off[2])]
        else:
            res["targets"].pop(sh_key, None)

    # 5. Legs (spherical around hip + floor clamp + swing-vs-stance natural step arc lift)
    walk_active = bool((obj.get("pawn_smooth_walk") or obj.get("enable_walk_gait", False)) and k0.get("walk_steps"))
    fL0, fL1 = _kf_pt(i, "foot_L"), _kf_pt(i + 1, "foot_L")
    fR0, fR1 = _kf_pt(i, "foot_R"), _kf_pt(i + 1, "foot_R")
    dz_L, dz_R = float(fL1[2] - fL0[2]), float(fR1[2] - fR0[2])
    horiz_L = math.hypot(float(fL1[0] - fL0[0]), dz_L)
    horiz_R = math.hypot(float(fR1[0] - fR0[0]), dz_R)
    p0_w = k0.get("pos") or obj.get("pos") or [0.0, 0.0, 0.0]
    p1_w = k1.get("pos") or obj.get("pos") or [0.0, 0.0, 0.0]
    ry0 = float(k0.get("rot_y") if k0.get("rot_y") is not None else ((k0.get("rot") or [0, 0, 0])[1]))
    ry1 = float(k1.get("rot_y") if k1.get("rot_y") is not None else ((k1.get("rot") or [0, 0, 0])[1]))
    ry_mid = math.radians(lerp_angle_deg(ry0, ry1, 0.5))
    travel_fwd = (float(p1_w[0]) - float(p0_w[0])) * math.sin(ry_mid) + (float(p1_w[2]) - float(p0_w[2])) * math.cos(ry_mid)
    both_swap = (dz_L * dz_R < -0.02) and (horiz_L > 0.14) and (horiz_R > 0.14)

    for side, is_left, hp_idx, u_leg in (("L", True, 2, u_leg_L), ("R", False, 3, u_leg_R)):
        foot_key = f"foot_{side}"
        pole_key = f"knee_{side}_pole"
        hp_cur = hp_L_cur if is_left else hp_R_cur
        side_out = side_tw_cur * (1.0 if is_left else -1.0)

        f_raw = [_kf_pt(k, foot_key) for k in range(n)]
        l_vecs = [f_raw[k] - frames_k[k][hp_idx] for k in range(n)]
        l_lens = [max(0.20, float(np.linalg.norm(v))) for v in l_vecs]
        l_dirs = [l_vecs[k] / float(np.linalg.norm(l_vecs[k])) if float(np.linalg.norm(l_vecs[k])) > 1e-5 else np.array([0.0, -1.0, 0.0]) for k in range(n)]
        s_leg = _eval_dir_progress_o1(kfs, times, l_dirs, modes_pose, i, u_leg)
        ld_cur = _slerp_unit3(l_dirs[i], l_dirs[i + 1], s_leg, fwd_tw_cur)
        ll_cur = _eval_scalar_list(l_lens, u_leg)
        foot_cur = hp_cur + ld_cur * ll_cur

        f0, f1 = f_raw[i], f_raw[i + 1]
        min_fy = min(float(f0[1]), float(f1[1]))
        if min_fy >= 0.015 and foot_cur[1] < min_fy:
            foot_cur[1] = min_fy
        if not is_hold and not walk_active:
            dz_self = dz_L if is_left else dz_R
            horiz_step = horiz_L if is_left else horiz_R
            other_horiz = horiz_R if is_left else horiz_L
            if horiz_step > 0.12 and float(f0[1]) <= 0.18 and float(f1[1]) <= 0.18:
                is_swing_leg = (
                    ((dz_self > 0.0) if travel_fwd >= -0.05 else (dz_self < 0.0))
                    if both_swap
                    else (other_horiz < 0.10 or (dz_self > 0.0 if travel_fwd >= -0.05 else dz_self < 0.0))
                )
                if is_swing_leg:
                    max_lift = 0.135 if both_swap else 0.095
                    step_lift = min(max_lift, 0.20 * horiz_step) * (math.sin(math.pi * s_leg) ** 1.4)
                    foot_cur[1] = float(foot_cur[1]) + step_lift
        res["targets"][foot_key] = [float(foot_cur[0]), float(foot_cur[1]), float(foot_cur[2])]

        kp_raw = [_kf_pt(k, pole_key) for k in range(n)]
        kp_vecs = [kp_raw[k] - frames_k[k][hp_idx] for k in range(n)]
        kp_lens = [max(0.18, float(np.linalg.norm(v))) for v in kp_vecs]
        kp_dirs = [kp_vecs[k] / float(np.linalg.norm(kp_vecs[k])) if float(np.linalg.norm(kp_vecs[k])) > 1e-5 else _normalize(fwd_tw_cur) for k in range(n)]
        kp_dir_cur = _slerp_unit3(kp_dirs[i], kp_dirs[i + 1], s_leg, side_out)
        kp_len_cur = _eval_scalar_list(kp_lens, u_leg)
        kp_cur = hp_cur + kp_dir_cur * kp_len_cur
        res["targets"][pole_key] = [float(kp_cur[0]), float(kp_cur[1]), float(kp_cur[2])]


def interpolate_character_at_time(obj: Dict[str, Any], t: float, scene: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    根据时间 t 插值角色的位置 pos、朝向 rot[1]（最短路径插值）、头部视线、全身 IK 控制球 targets 与阶跃可见性 visible。
    若开启 motion_protection（默认平滑变速 / 保持匀速），改走统一连续性引擎（普通关键帧不停步、停顿点平滑归零）。
    """
    kfs = sorted(
        [k for k in (obj.get("keyframes") or []) if isinstance(k, dict)],
        key=lambda item: float(item.get("t", 0.0)),
    )
    res = dict(obj)
    res.pop("_kfVisible", None)
    scene_style = str((scene or {}).get("global_body_style") or "").strip() if isinstance(scene, dict) else ""
    eff_body_style = str(obj.get("body_style") or scene_style or "mannequin").strip()
    res["body_style"] = eff_body_style
    is_pawn_style = (eff_body_style == "pawn")
    use_natural_limb = (not is_pawn_style) and (obj.get("natural_limb_motion") is not False)
    manual_vis = is_object_manually_visible(obj)
    anim_vis = eval_keyframe_step_visible(kfs, t)
    res["manual_visible"] = manual_vis
    res["anim_visible"] = anim_vis
    res["visible"] = bool(manual_vis and anim_vis)
    res["pos"] = list(obj.get("pos", [0.0, 0.0, 0.0]))
    res["rot"] = list(obj.get("rot", [0.0, 0.0, 0.0]))
    res["targets"] = {k: list(v) for k, v in obj.get("targets", {}).items()}
    base_targets_snapshot = {k: list(v) for k, v in res["targets"].items()}
    if not kfs:
        _apply_idle_breathing(res, obj, t)
        return res

    def _kf_rot_y(kf: Dict[str, Any], default_ry: float) -> float:
        if "rot_y" in kf and kf["rot_y"] is not None:
            return float(kf["rot_y"])
        if isinstance(kf.get("rot"), (list, tuple)) and len(kf["rot"]) >= 2:
            return float(kf["rot"][1])
        return float(default_ry)

    mp = obj.get("motion_protection")
    if (
        len(kfs) >= 2
        and isinstance(mp, dict)
        and mp.get("enabled")
        and str(mp.get("mode") or "smooth_variable").lower().strip() != "manual"
    ):
        plan = _get_cached_track_continuity_plan(obj, scene)
        if plan.get("enabled"):
            mp_ev = eval_track_continuity_at_time(plan, t)
            res["pos"] = list(mp_ev["pos"])
            res["rot"][1] = float(mp_ev["rot_y"])
            res["motion_speed_mps"] = float(mp_ev["speed"])
            res["motion_accel_mps2"] = float(mp_ev["accel"])
            res["motion_conflicts"] = list(plan.get("conflicts") or [])
            i = max(0, min(len(kfs) - 2, int(mp_ev["seg_idx"])))
            k0, k1 = kfs[i], kfs[i + 1]
            dt = float(k1.get("t", 0.0)) - float(k0.get("t", 0.0))
            alpha_t = max(0.0, min(1.0, (float(t) - float(k0.get("t", 0.0))) / dt)) if dt > 1e-6 else 0.0
            s_pose = max(0.0, min(1.0, float(mp_ev["seg_alpha"]) if (t >= kfs[0]["t"] and t <= kfs[-1]["t"]) else alpha_t))
            if use_natural_limb:
                _eval_mannequin_natural_pose_segment(res, obj, kfs, i, alpha_t, base_targets_snapshot)
            else:
                if k0.get("torso_twist") is not None or k1.get("torso_twist") is not None:
                    tw0 = float(k0.get("torso_twist", obj.get("torso_twist", 0.0)) or 0.0)
                    tw1 = float(k1.get("torso_twist", obj.get("torso_twist", 0.0)) or 0.0)
                    res["torso_twist"] = tw0 * (1.0 - s_pose) + tw1 * s_pose
                if k0.get("targets") or k1.get("targets"):
                    t0_map = k0.get("targets") or {}
                    t1_map = k1.get("targets") or {}
                    all_keys = set(res["targets"].keys()) | set(t0_map.keys()) | set(t1_map.keys())
                    for tk in all_keys:
                        default_v = [0.0, 0.0, 0.0] if tk in ("shoulder_L_offset", "shoulder_R_offset") else res["targets"].get(tk)
                        v0 = t0_map.get(tk, default_v)
                        v1 = t1_map.get(tk, default_v)
                        if isinstance(v0, (list, tuple)) and isinstance(v1, (list, tuple)) and len(v0) >= 3 and len(v1) >= 3:
                            res["targets"][tk] = [
                                float(v0[j]) * (1.0 - s_pose) + float(v1[j]) * s_pose
                                for j in range(3)
                            ]
                if k0.get("look_at") and k1.get("look_at"):
                    res["targets"]["head_target"] = [
                        k0["look_at"][j] * (1.0 - s_pose) + k1["look_at"][j] * s_pose for j in range(3)
                    ]
            if (obj.get("pawn_smooth_walk") or obj.get("enable_walk_gait", False)) and k0.get("walk_steps"):
                r0 = _kf_rot_y(k0, res["rot"][1])
                r1 = _kf_rot_y(k1, res["rot"][1])
                _apply_procedural_walk_gait(res, k0, k1, r0, r1, s_pose)
            else:
                _apply_idle_breathing(res, obj, t)
            return res

    if len(kfs) == 1 or t <= kfs[0]["t"]:
        k = kfs[0]
        res["pos"] = list(k["pos"])
        res["rot"][1] = _kf_rot_y(k, res["rot"][1])
        if k.get("torso_twist") is not None:
            res["torso_twist"] = float(k["torso_twist"])
        if k.get("targets"):
            for tk, tv in k["targets"].items():
                res["targets"][tk] = list(tv)
        if k.get("look_at"):
            res["targets"]["head_target"] = list(k["look_at"])
        _apply_idle_breathing(res, obj, t)
        return res
    if t >= kfs[-1]["t"]:
        k = kfs[-1]
        res["pos"] = list(k["pos"])
        res["rot"][1] = _kf_rot_y(k, res["rot"][1])
        if k.get("torso_twist") is not None:
            res["torso_twist"] = float(k["torso_twist"])
        if k.get("targets"):
            for tk, tv in k["targets"].items():
                res["targets"][tk] = list(tv)
        if k.get("look_at"):
            res["targets"]["head_target"] = list(k["look_at"])
        _apply_idle_breathing(res, obj, t)
        return res

    times_c = [float(k.get("t", 0.0)) for k in kfs]
    modes_pos_c = [normalize_cam_interp_mode(k.get("interp_pos", k.get("interp", "linear"))) for k in kfs]
    modes_rot_c = [normalize_cam_interp_mode(k.get("interp_rot", k.get("interp", "linear"))) for k in kfs]
    pos_x_c = [float(k["pos"][0]) for k in kfs]
    pos_y_c = [float(k["pos"][1]) for k in kfs]
    pos_z_c = [float(k["pos"][2]) for k in kfs]
    unwrapped_ry_c = [_kf_rot_y(kfs[0], res["rot"][1])]
    for i in range(1, len(kfs)):
        prev_ry = unwrapped_ry_c[-1]
        cur_ry = _kf_rot_y(kfs[i], res["rot"][1])
        unwrapped_ry_c.append(prev_ry + (((cur_ry - prev_ry) % 360.0 + 540.0) % 360.0 - 180.0))

    for i in range(len(kfs) - 1):
        k0, k1 = kfs[i], kfs[i + 1]
        if k0["t"] <= t <= k1["t"]:
            dt = k1["t"] - k0["t"]
            alpha = (t - k0["t"]) / dt if dt > 1e-6 else 0.0
            base_m = str(k0.get("interp", "linear") or "linear")
            m_pos = modes_pos_c[i]
            m_rot = modes_rot_c[i]
            s_pos = eval_easing(alpha, k0.get("interp_pos", base_m))
            s_rot = eval_easing(alpha, k0.get("interp_rot", base_m))
            s_pose = eval_easing(alpha, k0.get("interp_pose", base_m))
            if m_pos == "continuous":
                res["pos"] = [
                    _eval_camera_hermite_channel(kfs, times_c, pos_x_c, modes_pos_c, i, alpha),
                    _eval_camera_hermite_channel(kfs, times_c, pos_y_c, modes_pos_c, i, alpha),
                    _eval_camera_hermite_channel(kfs, times_c, pos_z_c, modes_pos_c, i, alpha),
                ]
            else:
                res["pos"] = [k0["pos"][j] * (1.0 - s_pos) + k1["pos"][j] * s_pos for j in range(3)]
            r0 = _kf_rot_y(k0, res["rot"][1])
            r1 = _kf_rot_y(k1, res["rot"][1])
            if m_rot == "continuous":
                res["rot"][1] = _eval_camera_hermite_channel(kfs, times_c, unwrapped_ry_c, modes_rot_c, i, alpha)
            else:
                res["rot"][1] = lerp_angle_deg(r0, r1, s_rot)
            if use_natural_limb:
                _eval_mannequin_natural_pose_segment(res, obj, kfs, i, alpha, base_targets_snapshot)
            else:
                if k0.get("torso_twist") is not None or k1.get("torso_twist") is not None:
                    tw0 = float(k0.get("torso_twist", obj.get("torso_twist", 0.0)) or 0.0)
                    tw1 = float(k1.get("torso_twist", obj.get("torso_twist", 0.0)) or 0.0)
                    res["torso_twist"] = tw0 * (1.0 - s_pose) + tw1 * s_pose
                if k0.get("targets") or k1.get("targets"):
                    t0_map = k0.get("targets") or {}
                    t1_map = k1.get("targets") or {}
                    all_keys = set(res["targets"].keys()) | set(t0_map.keys()) | set(t1_map.keys())
                    for tk in all_keys:
                        default_v = [0.0, 0.0, 0.0] if tk in ("shoulder_L_offset", "shoulder_R_offset") else res["targets"].get(tk)
                        v0 = t0_map.get(tk, default_v)
                        v1 = t1_map.get(tk, default_v)
                        if isinstance(v0, (list, tuple)) and isinstance(v1, (list, tuple)) and len(v0) >= 3 and len(v1) >= 3:
                            res["targets"][tk] = [
                                float(v0[j]) * (1.0 - s_pose) + float(v1[j]) * s_pose
                                for j in range(3)
                            ]
                if k0.get("look_at") and k1.get("look_at"):
                    res["targets"]["head_target"] = [
                        k0["look_at"][j] * (1.0 - s_pose) + k1["look_at"][j] * s_pose for j in range(3)
                    ]
            if (obj.get("pawn_smooth_walk") or obj.get("enable_walk_gait", False)) and k0.get("walk_steps"):
                _apply_procedural_walk_gait(res, k0, k1, r0, r1, s_pos)
            else:
                _apply_idle_breathing(res, obj, t)
            return res
    return res


def interpolate_object_at_time(obj: Dict[str, Any], t: float, _skip_align_tangent: bool = False, scene: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """在时刻 t 插值非人物物体（车辆 uaz469、组合体 group、几何体 box 等）的位置 pos、三轴旋转 rot、尺寸 size 与阶跃可见性 visible。
    若开启 motion_protection，优先使用统一 C2 弧长连续性求解器（兼顾门/旋转件纯旋转 S5 缓和）；未开启时保留传统插值兼容。
    """
    raw_kfs = sorted(
        [k for k in (obj.get("keyframes", []) or []) if isinstance(k, dict)],
        key=lambda x: float(x.get("t", 0.0)),
    )
    res = dict(obj)
    res.pop("_kfVisible", None)
    manual_vis = is_object_manually_visible(obj)
    anim_vis = eval_keyframe_step_visible(raw_kfs, t)
    res["manual_visible"] = manual_vis
    res["anim_visible"] = anim_vis
    res["visible"] = bool(manual_vis and anim_vis)
    base_pos = [float(x) for x in (obj.get("pos") or [0.0, 0.0, 0.0])[:3]]
    base_rot = [float(x) for x in (obj.get("rot") or [0.0, 0.0, 0.0])[:3]]
    base_size = [float(x) for x in (obj.get("size") or [1.0, 1.0, 1.0])[:3]]
    res["pos"] = list(base_pos)
    res["rot"] = list(base_rot)
    res["size"] = list(base_size)
    if not raw_kfs:
        return res
    kfs: List[Dict[str, Any]] = []
    for k in raw_kfs:
        kt = float(k.get("t", 0.0))
        if kfs and abs(kt - float(kfs[-1].get("t", 0.0))) < 1e-5:
            kfs[-1] = k
        else:
            kfs.append(k)

    def _kf_pos(k: Dict[str, Any]) -> List[float]:
        p = k.get("pos")
        if isinstance(p, (list, tuple)) and len(p) >= 3:
            return [float(p[0]), float(p[1]), float(p[2])]
        return list(base_pos)

    def _kf_rot(k: Dict[str, Any]) -> List[float]:
        r = k.get("rot")
        if isinstance(r, (list, tuple)) and len(r) >= 3:
            return [float(r[0]), float(r[1]), float(r[2])]
        if "rot_y" in k and k.get("rot_y") is not None:
            return [base_rot[0], float(k["rot_y"]), base_rot[2]]
        return list(base_rot)

    def _kf_size(k: Dict[str, Any]) -> List[float]:
        s = k.get("size")
        if isinstance(s, (list, tuple)) and len(s) >= 3:
            return [float(s[0]), float(s[1]), float(s[2])]
        return list(base_size)

    def _apply_tangent_align(out_res: Dict[str, Any]) -> Dict[str, Any]:
        if _skip_align_tangent or not obj.get("align_tangent_y") or len(kfs) < 2:
            return out_res
        t_min = float(kfs[0].get("t", 0.0))
        t_max = float(kfs[-1].get("t", 0.0))
        eps = 0.04
        ta = max(t_min, float(t) - eps)
        tb = min(t_max, float(t) + eps)
        dt_span = tb - ta
        if dt_span <= 1e-4:
            return out_res
        ev_a = interpolate_object_at_time(obj, ta, _skip_align_tangent=True, scene=scene)
        ev_b = interpolate_object_at_time(obj, tb, _skip_align_tangent=True, scene=scene)
        dx = float(ev_b["pos"][0]) - float(ev_a["pos"][0])
        dz = float(ev_b["pos"][2]) - float(ev_a["pos"][2])
        dist = math.hypot(dx, dz)
        speed = dist / dt_span
        if speed <= 0.08 or dist <= 1e-4:
            return out_res
        # 检查当前段是否显式标记倒车 reverse，倒车时车头保持反向（不把车尾甩到前面）
        is_rev = False
        for seg_idx in range(len(kfs) - 1):
            if float(kfs[seg_idx].get("t", 0.0)) - 1e-5 <= float(t) <= float(kfs[seg_idx + 1].get("t", 0.0)) + 1e-5:
                is_rev = bool(kfs[seg_idx].get("reverse"))
                break
        tangent_deg = math.degrees(math.atan2(dx, dz) if is_rev else math.atan2(-dx, -dz))
        blend = max(0.0, min(1.0, (speed - 0.08) / 0.25))
        cur_yaw = float(out_res["rot"][1])
        diff = (((tangent_deg - cur_yaw) % 360.0 + 540.0) % 360.0) - 180.0
        out_res["rot"] = [float(out_res["rot"][0]), cur_yaw + diff * blend, float(out_res["rot"][2])]
        out_res["direction"] = "reverse" if is_rev else "forward"
        return out_res

    def _kf_door_fr(k: Dict[str, Any]) -> float:
        v = k.get("door_FR_deg")
        return float(v) if v is not None else float(obj.get("door_FR_deg") or 0.0)

    def _kf_door_fl(k: Dict[str, Any]) -> float:
        v = k.get("door_FL_deg")
        return float(v) if v is not None else float(obj.get("door_FL_deg") or 0.0)

    if len(kfs) == 1 or t <= float(kfs[0].get("t", 0.0)):
        k = kfs[0]
        res["pos"] = _kf_pos(k)
        res["rot"] = _kf_rot(k)
        res["size"] = _kf_size(k)
        res["door_FR_deg"] = _kf_door_fr(k)
        res["door_FL_deg"] = _kf_door_fl(k)
        return _apply_tangent_align(res)

    if t >= float(kfs[-1].get("t", 0.0)):
        k = kfs[-1]
        res["pos"] = _kf_pos(k)
        res["rot"] = _kf_rot(k)
        res["size"] = _kf_size(k)
        res["door_FR_deg"] = _kf_door_fr(k)
        res["door_FL_deg"] = _kf_door_fl(k)
        return _apply_tangent_align(res)

    mp = obj.get("motion_protection")
    if (
        isinstance(mp, dict)
        and mp.get("enabled")
        and str(mp.get("mode") or "smooth_variable").lower().strip() != "manual"
    ):
        plan = _get_cached_track_continuity_plan(obj, scene)
        if plan.get("enabled"):
            mp_ev = eval_track_continuity_at_time(plan, t)
            res["pos"] = list(mp_ev["pos"])
            seg_i = max(0, min(len(kfs) - 2, int(mp_ev["seg_idx"])))
            u_prog = max(0.0, min(1.0, float(mp_ev["seg_alpha"])))
            unwrapped_rots: List[List[float]] = [_kf_rot(kfs[0])]
            for idx_u in range(1, len(kfs)):
                prev_r = unwrapped_rots[-1]
                cur_r = _kf_rot(kfs[idx_u])
                unwrapped_rots.append([
                    prev_r[j] + (((cur_r[j] - prev_r[j]) % 360.0 + 540.0) % 360.0 - 180.0)
                    for j in range(3)
                ])
            r0, r1 = unwrapped_rots[seg_i], unwrapped_rots[seg_i + 1]
            s0, s1 = _kf_size(kfs[seg_i]), _kf_size(kfs[seg_i + 1])
            res["rot"] = [
                r0[0] + (r1[0] - r0[0]) * u_prog,
                float(mp_ev["rot_y"]),
                r0[2] + (r1[2] - r0[2]) * u_prog,
            ]
            res["size"] = [
                max(0.01, s0[0] + (s1[0] - s0[0]) * u_prog),
                max(0.01, s0[1] + (s1[1] - s0[1]) * u_prog),
                max(0.01, s0[2] + (s1[2] - s0[2]) * u_prog),
            ]
            res["motion_speed_mps"] = float(mp_ev["speed"])
            res["motion_accel_mps2"] = float(mp_ev["accel"])
            res["motion_conflicts"] = list(plan.get("conflicts") or [])
            return _apply_tangent_align(res)

    times = [float(k.get("t", 0.0)) for k in kfs]
    modes_pos = [normalize_cam_interp_mode(k.get("interp_pos") or k.get("interp") or "linear") for k in kfs]
    modes_rot = [normalize_cam_interp_mode(k.get("interp_rot") or k.get("interp") or "linear") for k in kfs]
    modes_size = [normalize_cam_interp_mode(k.get("interp_size") or k.get("interp") or "linear") for k in kfs]

    pos_x = [_kf_pos(k)[0] for k in kfs]
    pos_y = [_kf_pos(k)[1] for k in kfs]
    pos_z = [_kf_pos(k)[2] for k in kfs]

    # 构建连续展开角度序列（避免经过 ±180° 或 0/360° 时反向猛甩）
    unwrapped_rots: List[List[float]] = [_kf_rot(kfs[0])]
    for i in range(1, len(kfs)):
        prev_r = unwrapped_rots[-1]
        cur_r = _kf_rot(kfs[i])
        unwrapped_rots.append([
            prev_r[j] + (((cur_r[j] - prev_r[j]) % 360.0 + 540.0) % 360.0 - 180.0)
            for j in range(3)
        ])
    rot_x = [r[0] for r in unwrapped_rots]
    rot_y = [r[1] for r in unwrapped_rots]
    rot_z = [r[2] for r in unwrapped_rots]

    size_x = [_kf_size(k)[0] for k in kfs]
    size_y = [_kf_size(k)[1] for k in kfs]
    size_z = [_kf_size(k)[2] for k in kfs]
    doors_fr = [_kf_door_fr(k) for k in kfs]
    doors_fl = [_kf_door_fl(k) for k in kfs]

    for i in range(len(kfs) - 1):
        t0, t1 = times[i], times[i + 1]
        if t0 <= t <= t1:
            dt = t1 - t0
            alpha = (t - t0) / dt if dt > 1e-4 else 0.0
            res["pos"] = [
                _eval_camera_hermite_channel(kfs, times, pos_x, modes_pos, i, alpha),
                _eval_camera_hermite_channel(kfs, times, pos_y, modes_pos, i, alpha),
                _eval_camera_hermite_channel(kfs, times, pos_z, modes_pos, i, alpha),
            ]
            res["rot"] = [
                _eval_camera_hermite_channel(kfs, times, rot_x, modes_rot, i, alpha),
                _eval_camera_hermite_channel(kfs, times, rot_y, modes_rot, i, alpha),
                _eval_camera_hermite_channel(kfs, times, rot_z, modes_rot, i, alpha),
            ]
            res["size"] = [
                max(0.01, _eval_camera_hermite_channel(kfs, times, size_x, modes_size, i, alpha)),
                max(0.01, _eval_camera_hermite_channel(kfs, times, size_y, modes_size, i, alpha)),
                max(0.01, _eval_camera_hermite_channel(kfs, times, size_z, modes_size, i, alpha)),
            ]
            res["door_FR_deg"] = _eval_camera_hermite_channel(kfs, times, doors_fr, modes_rot, i, alpha)
            res["door_FL_deg"] = _eval_camera_hermite_channel(kfs, times, doors_fl, modes_rot, i, alpha)
            return _apply_tangent_align(res)

    k_last = kfs[-1]
    res["pos"] = _kf_pos(k_last)
    res["rot"] = _kf_rot(k_last)
    res["size"] = _kf_size(k_last)
    res["door_FR_deg"] = _kf_door_fr(k_last)
    res["door_FL_deg"] = _kf_door_fl(k_last)
    return _apply_tangent_align(res)


def compute_track_motion_stats(
    keyframes: List[Dict[str, Any]],
    is_camera: bool = False,
    obj: Optional[Dict[str, Any]] = None,
    scene: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """计算关键帧轨道的真实 3D 弧长、平均速度、曲线峰值速度（m/s 与车辆 km/h）及限速规范检查"""
    if not keyframes or len(keyframes) < 2:
        return {
            "segments": [],
            "total_distance_m": 0.0,
            "total_duration_s": 0.0,
            "overall_avg_speed_mps": 0.0,
            "peak_speed_mps": 0.0,
        }
    target_obj = copy.deepcopy(obj) if isinstance(obj, dict) else {}
    if is_camera:
        target_obj["camera_keyframes"] = keyframes
    else:
        target_obj["keyframes"] = keyframes
    profile = analyze_track_speed_profile(target_obj, scene=scene, is_camera=is_camera)
    kfs = sorted(keyframes, key=lambda x: float(x.get("t", 0.0)))
    segments = []
    for seg in profile.get("segments") or []:
        i = int(seg["index"])
        k0 = kfs[i] if i < len(kfs) else {}
        base_m = k0.get("interp", "linear")
        seg_info: Dict[str, Any] = {
            "index": i,
            "t0": seg["t0"],
            "t1": seg["t1"],
            "dt": seg["dt"],
            "distance_m": seg["arc_distance_m"],
            "chord_distance_m": seg["chord_distance_m"],
            "avg_speed_mps": seg["avg_speed_mps"],
            "avg_speed_kmh": seg.get("avg_speed_kmh", round(seg["avg_speed_mps"] * 3.6, 1)),
            "peak_speed_mps": seg["peak_speed_mps"],
            "peak_speed_kmh": seg.get("peak_speed_kmh", round(seg["peak_speed_mps"] * 3.6, 1)),
            "peak_time_s": seg.get("peak_time_s", seg["t1"]),
            "min_turn_radius_m": seg.get("min_turn_radius_m"),
            "direction": seg.get("direction", "forward"),
            "reverse": bool(seg.get("reverse", False)),
            "is_cut_or_teleport": bool(seg.get("is_cut_or_teleport", False)),
            "interp_pos": k0.get("interp_pos", base_m),
        }
        if is_camera:
            seg_info["interp_target"] = k0.get("interp_target", base_m)
            seg_info["interp_fov"] = k0.get("interp_fov", base_m)
        else:
            seg_info["interp_rot"] = k0.get("interp_rot", base_m)
            seg_info["interp_pose"] = k0.get("interp_pose", base_m)
        segments.append(seg_info)

    return {
        "segments": segments,
        "total_distance_m": profile["total_arc_m"],
        "total_duration_s": profile["total_duration_s"],
        "overall_avg_speed_mps": profile["overall_avg_mps"],
        "overall_avg_speed_kmh": profile.get("overall_avg_kmh", round(profile["overall_avg_mps"] * 3.6, 1)),
        "peak_speed_mps": profile["peak_speed_mps"],
        "peak_speed_kmh": profile.get("peak_speed_kmh", round(profile["peak_speed_mps"] * 3.6, 1)),
        "peak_time_s": profile.get("peak_time_s", 0.0),
        "direction": profile.get("direction", "forward"),
        "wheel_summary": profile.get("wheel_summary"),
        "speed_policy_report": profile,
    }


def transform_keyframes_timing(
    keyframes: List[Dict[str, Any]],
    scale_factor: float = 1.0,
    time_offset: float = 0.0,
    anchor_time: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """支持整段关键帧等比例加速/减速、平移时间，保持内部相对节奏"""
    if not keyframes:
        return []
    kfs = [dict(k) for k in sorted(keyframes, key=lambda x: float(x.get("t", 0.0)))]
    t_anchor = float(anchor_time) if anchor_time is not None else float(kfs[0].get("t", 0.0))
    factor = max(0.05, min(20.0, float(scale_factor)))
    offset = float(time_offset)
    for k in kfs:
        t_orig = float(k.get("t", 0.0))
        t_new = max(0.0, t_anchor + (t_orig - t_anchor) * factor + offset)
        k["t"] = round(t_new, 3)
    kfs.sort(key=lambda x: x["t"])
    return kfs


def _char_world_to_local(char_dict: Dict[str, Any], world_pt: np.ndarray) -> List[float]:
    base_pos = np.asarray(char_dict.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
    rot_y_deg = float(char_dict.get("rot", [0.0, 0.0, 0.0])[1])
    sx, sy, sz = _get_char_scale_xyz(char_dict)
    ry = math.radians(-rot_y_deg)
    cy, s_y = math.cos(ry), math.sin(ry)
    R_inv = np.array([[cy, 0, s_y], [0, 1, 0], [-s_y, 0, cy]], dtype=np.float64)
    local_v = R_inv @ (np.asarray(world_pt, dtype=np.float64) - base_pos)
    return [round(float(local_v[0] / sx), 4), round(float(local_v[1] / sy), 4), round(float(local_v[2] / sz), 4)]


def _char_local_to_world(char_dict: Dict[str, Any], local_pt: List[float]) -> np.ndarray:
    base_pos = np.asarray(char_dict.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
    rot_y_deg = float(char_dict.get("rot", [0.0, 0.0, 0.0])[1])
    sx, sy, sz = _get_char_scale_xyz(char_dict)
    ry = math.radians(rot_y_deg)
    cy, s_y = math.cos(ry), math.sin(ry)
    R_root = np.array([[cy, 0, s_y], [0, 1, 0], [-s_y, 0, cy]], dtype=np.float64)
    v = np.array([float(local_pt[0]) * sx, float(local_pt[1]) * sy, float(local_pt[2]) * sz], dtype=np.float64)
    return base_pos + (R_root @ v)


def get_target_character_part_world(
    tgt_char: Dict[str, Any],
    tgt_ik: Dict[str, Any],
    part_name: str,
    local_offset: Optional[List[float]] = None,
) -> np.ndarray:
    """获取目标角色指定部位（肩、前臂、手、胸、骨盆或局部点）叠加局部偏移后的 3D 世界坐标"""
    J = tgt_ik["joints"]
    pname = (part_name or "shoulder_L").strip()
    if pname == "forearm_L":
        base_w = 0.5 * (J["elbow_L"] + J["wrist_L"])
    elif pname == "forearm_R":
        base_w = 0.5 * (J["elbow_R"] + J["wrist_R"])
    elif pname in ("hand_L", "wrist_L"):
        base_w = J["wrist_L"].copy()
    elif pname in ("hand_R", "wrist_R"):
        base_w = J["wrist_R"].copy()
    elif pname in ("chest", "neck"):
        base_w = J["neck"].copy()
    elif pname == "local_point":
        base_w = np.asarray(tgt_char.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
    elif pname in J:
        base_w = J[pname].copy()
    else:
        base_w = J["shoulder_L"].copy()

    if local_offset and len(local_offset) >= 3:
        rot_y_deg = float(tgt_char.get("rot", [0.0, 0.0, 0.0])[1])
        sx, sy, sz = _get_char_scale_xyz(tgt_char)
        ry = math.radians(rot_y_deg)
        cy, s_y = math.cos(ry), math.sin(ry)
        R_tgt = np.array([[cy, 0, s_y], [0, 1, 0], [-s_y, 0, cy]], dtype=np.float64)
        off_v = np.array([float(local_offset[0]) * sx, float(local_offset[1]) * sy, float(local_offset[2]) * sz], dtype=np.float64)
        base_w = base_w + (R_tgt @ off_v)
    return base_w


def compute_constraint_effective_weight(
    constraint: Dict[str, Any],
    cur_frame: float,
) -> float:
    """根据起止帧、基础权重与渐入渐出帧数计算当前帧的生效权重 [0.0, 1.0]"""
    if not constraint.get("enabled", True):
        return 0.0
    base_w = max(0.0, min(1.0, float(constraint.get("weight", 1.0))))
    if base_w <= 1e-6:
        return 0.0
    s_frame = float(constraint.get("start_frame", 0))
    e_frame = float(constraint.get("end_frame", 999999))
    if cur_frame < s_frame or cur_frame > e_frame:
        return 0.0
    fade_in = max(0.0, float(constraint.get("fade_in_frames", 0)))
    fade_out = max(0.0, float(constraint.get("fade_out_frames", 0)))
    w_in = min(1.0, (cur_frame - s_frame) / fade_in) if fade_in > 1e-4 else 1.0
    w_out = min(1.0, (e_frame - cur_frame) / fade_out) if fade_out > 1e-4 else 1.0
    return round(base_w * min(w_in, w_out), 4)


def matrix_to_euler_xyz_deg(R: np.ndarray) -> List[float]:
    """将 3x3 旋转矩阵转换为与 euler_xyz_to_matrix / Three.js Euler('XYZ') 严格一致的欧拉角 (deg)"""
    m02 = max(-1.0, min(1.0, float(R[0, 2])))
    ry = math.asin(m02)
    if abs(m02) < 0.999999:
        rx = math.atan2(-float(R[1, 2]), float(R[2, 2]))
        rz = math.atan2(-float(R[0, 1]), float(R[0, 0]))
    else:
        rx = math.atan2(float(R[2, 1]), float(R[1, 1]))
        rz = 0.0
    return [round(math.degrees(rx), 3), round(math.degrees(ry), 3), round(math.degrees(rz), 3)]


def get_character_socket_transform(
    char_dict: Dict[str, Any],
    ik_res: Dict[str, Any],
    socket_name: str = "hand_R",
) -> Dict[str, Any]:
    """
    计算角色在当前姿态下指定挂点 (hand_R / hand_L / chest / back / root) 的 3D 世界位置与正交旋转基底：
    - +Z (f_socket): 挂点前向（持枪瞄准方向/胸前法向）
    - +Y (u_socket): 挂点上向
    - +X (s_socket): 挂点右向 (u_ref × f_socket)
    右手/左手挂点由 肩->腕 主轴(0.72) + 肘->腕 前臂方向(0.18) + 胸前朝向(0.10) 稳定合成，手臂接近伸直时绝不翻转。
    """
    J = ik_res["joints"]
    A = ik_res["axes"]
    sx, sy, sz = _get_char_scale_xyz(char_dict)
    spine_dir = _normalize(np.asarray(A.get("spine_dir", [0.0, 1.0, 0.0]), dtype=np.float64))
    side_chest = _normalize(np.asarray(A.get("side_chest", A.get("side_L", [1.0, 0.0, 0.0])), dtype=np.float64))
    chest_fwd = np.cross(side_chest, spine_dir)
    if float(np.linalg.norm(chest_fwd)) < 1e-5:
        yaw_r = math.radians(float(char_dict.get("rot", [0.0, 0.0, 0.0])[1]))
        chest_fwd = np.array([math.sin(yaw_r), 0.0, math.cos(yaw_r)], dtype=np.float64)
    else:
        chest_fwd = _normalize(chest_fwd)

    sock = str(socket_name or "hand_R").strip()
    if sock in ("hand_R", "wrist_R", "hand_L", "wrist_L"):
        is_r = sock in ("hand_R", "wrist_R")
        p_sh = np.asarray(J["shoulder_R" if is_r else "shoulder_L"], dtype=np.float64)
        p_el = np.asarray(J["elbow_R" if is_r else "elbow_L"], dtype=np.float64)
        p_wr = np.asarray(J["wrist_R" if is_r else "wrist_L"], dtype=np.float64)
        v_sh_wr = p_wr - p_sh
        dir_sh_wr = _normalize(v_sh_wr) if float(np.linalg.norm(v_sh_wr)) > 1e-4 else chest_fwd
        v_el_wr = p_wr - p_el
        dir_el_wr = _normalize(v_el_wr) if float(np.linalg.norm(v_el_wr)) > 1e-4 else chest_fwd
        raw_f = dir_sh_wr * 0.72 + dir_el_wr * 0.18 + chest_fwd * 0.10
        f_socket = _normalize(raw_f) if float(np.linalg.norm(raw_f)) > 1e-4 else chest_fwd
        dot_up = float(np.dot(f_socket, spine_dir))
        if abs(dot_up) > 0.90:
            sign_v = 1.0 if dot_up >= 0.0 else -1.0
            u_ref = _normalize(spine_dir * 0.35 - chest_fwd * (0.65 * sign_v))
        else:
            u_ref = spine_dir
        p_socket = p_wr.copy()
    elif sock == "chest":
        p_st = np.asarray(J.get("sternum", J["neck"]), dtype=np.float64)
        p_socket = p_st + chest_fwd * (0.11 * sz)
        f_socket = chest_fwd
        u_ref = spine_dir
    elif sock == "back":
        p_st = np.asarray(J.get("sternum", J["neck"]), dtype=np.float64)
        p_socket = p_st - chest_fwd * (0.12 * sz)
        f_socket = -chest_fwd
        u_ref = spine_dir
    else:
        p_socket = np.asarray(char_dict.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
        yaw_r = math.radians(float(char_dict.get("rot", [0.0, 0.0, 0.0])[1]))
        f_socket = np.array([math.sin(yaw_r), 0.0, math.cos(yaw_r)], dtype=np.float64)
        u_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    s_raw = np.cross(u_ref, f_socket)
    if float(np.linalg.norm(s_raw)) < 1e-5:
        s_socket = -side_chest
    else:
        s_socket = _normalize(s_raw)
    u_socket = _normalize(np.cross(f_socket, s_socket))
    R_socket = np.column_stack([s_socket, u_socket, f_socket])
    return {
        "pos": p_socket,
        "R": R_socket,
        "scale": float(sy),
        "sx": float(sx),
        "sy": float(sy),
        "sz": float(sz),
        "socket": sock,
    }


def socket_local_to_world(
    local_pos: List[float],
    local_rot: List[float],
    local_scale: Optional[List[float]],
    base_size: List[float],
    socket_tf: Dict[str, Any],
) -> Dict[str, Any]:
    """将挂点局部变换 (local_pos, local_rot, local_scale) 合成到世界空间"""
    p_sock = np.asarray(socket_tf["pos"], dtype=np.float64)
    R_sock = np.asarray(socket_tf["R"], dtype=np.float64)
    sy = max(1e-4, float(socket_tf.get("scale", 1.0)))
    l_pos = np.asarray((local_pos or [0.0, 0.0, 0.0])[:3], dtype=np.float64)
    l_rot = [float(x) for x in (local_rot or [0.0, 0.0, 0.0])[:3]]
    l_sc = np.asarray((local_scale or [1.0, 1.0, 1.0])[:3], dtype=np.float64)
    b_sz = np.asarray((base_size or [0.08, 0.12, 0.72])[:3], dtype=np.float64)

    R_local = euler_xyz_to_matrix(l_rot[0], l_rot[1], l_rot[2])
    R_world = R_sock @ R_local
    P_world = p_sock + (R_sock @ (l_pos * sy))
    rot_world = matrix_to_euler_xyz_deg(R_world)
    size_world = np.maximum(0.01, b_sz * l_sc * sy)
    render_scale = float(sy * float(l_sc[1] if len(l_sc) > 1 else 1.0))
    return {
        "pos": [round(float(x), 4) for x in P_world],
        "rot": rot_world,
        "size": [round(float(x), 4) for x in size_world],
        "render_scale": round(render_scale, 4),
        "R_world": R_world,
        "P_world": P_world,
    }


def world_to_socket_local(
    world_pos: List[float],
    world_rot: List[float],
    world_size: Optional[List[float]],
    base_size: Optional[List[float]],
    socket_tf: Dict[str, Any],
) -> Dict[str, Any]:
    """将世界空间变换逆解算为挂点局部坐标 (用于“保持当前位置绑定”与视口微调)"""
    p_sock = np.asarray(socket_tf["pos"], dtype=np.float64)
    R_sock = np.asarray(socket_tf["R"], dtype=np.float64)
    sy = max(1e-4, float(socket_tf.get("scale", 1.0)))
    w_pos = np.asarray((world_pos or [0.0, 0.0, 0.0])[:3], dtype=np.float64)
    w_rot = [float(x) for x in (world_rot or [0.0, 0.0, 0.0])[:3]]
    R_world = euler_xyz_to_matrix(w_rot[0], w_rot[1], w_rot[2])
    R_local = R_sock.T @ R_world
    local_rot = matrix_to_euler_xyz_deg(R_local)
    local_pos_v = (R_sock.T @ (w_pos - p_sock)) / sy
    b_sz = np.maximum(1e-4, np.asarray((base_size or world_size or [0.08, 0.12, 0.72])[:3], dtype=np.float64))
    w_sz = np.asarray((world_size or b_sz)[:3], dtype=np.float64)
    local_scale_v = (w_sz / b_sz) / sy
    return {
        "local_pos": [round(float(x), 4) for x in local_pos_v],
        "local_rot": local_rot,
        "local_scale": [round(float(x), 4) for x in local_scale_v],
    }


def _compose_layout_ref_matrix(layout_ref: Any) -> Optional[np.ndarray]:
    if not isinstance(layout_ref, dict):
        return None
    pos = [float(x) for x in (layout_ref.get("pos") or [0.0, 0.0, 0.0])[:3]]
    rot = [float(x) for x in (layout_ref.get("rot") or [0.0, 0.0, 0.0])[:3]]
    scale = [float(x) for x in (layout_ref.get("scale") or [1.0, 1.0, 1.0])[:3]]
    origin = [float(x) for x in (layout_ref.get("origin") or [0.0, 0.0, 0.0])[:3]]
    has_p = any(abs(v) > 1e-6 for v in pos)
    has_r = any(abs(v) > 1e-5 for v in rot)
    has_s = any(abs(v - 1.0) > 1e-5 for v in scale)
    if not (has_p or has_r or has_s):
        return None
    T_neg = make_transform([-origin[0], -origin[1], -origin[2]], [0.0, 0.0, 0.0], 1.0)
    T_scale = np.diag([scale[0], scale[1], scale[2], 1.0])
    T_rot = make_transform([origin[0] + pos[0], origin[1] + pos[1], origin[2] + pos[2]], rot, 1.0)
    return T_rot @ T_scale @ T_neg


def _decompose_prefix_matrix(M: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    c0, c1, c2 = M[:3, 0], M[:3, 1], M[:3, 2]
    sx = float(np.linalg.norm(c0)) or 1.0
    sy = float(np.linalg.norm(c1)) or 1.0
    sz = float(np.linalg.norm(c2)) or 1.0
    R = np.column_stack([c0 / sx, c1 / sy, c2 / sz])
    return R, [sx, sy, sz]


def _get_hierarchy_parent_world_transform(
    obj: Dict[str, Any],
    obj_map: Optional[Dict[str, Dict[str, Any]]],
    t: float,
    visited: Optional[set] = None,
) -> Tuple[np.ndarray, bool]:
    if not obj_map or not isinstance(obj, dict):
        return np.eye(4, dtype=np.float64), True
    pid = str(obj.get("parent_id") or "").strip()
    self_id = str(obj.get("id") or "")
    if not pid or pid == self_id or pid not in obj_map:
        return np.eye(4, dtype=np.float64), True
    seen = set(visited) if visited else {self_id}
    if pid in seen:
        return np.eye(4, dtype=np.float64), True
    seen.add(pid)
    parent = obj_map[pid]
    M_gp, gp_vis = _get_hierarchy_parent_world_transform(parent, obj_map, t, seen)
    p_local = interpolate_object_at_time(parent, t)
    p_vis = bool(gp_vis and p_local.get("visible", True))
    M_plocal = make_transform(p_local.get("pos", [0.0, 0.0, 0.0]), p_local.get("rot", [0.0, 0.0, 0.0]), 1.0)
    if str(parent.get("type") or "") == "scene_group":
        psz = [float(x) for x in (p_local.get("size") or [1.0, 1.0, 1.0])[:3]]
        M_plocal = M_plocal @ np.diag([psz[0], psz[1], psz[2], 1.0])
    M_preloc = _compose_layout_ref_matrix(parent.get("layout_ref"))
    if M_preloc is not None:
        M_plocal = M_preloc @ M_plocal
    return M_gp @ M_plocal, p_vis


def evaluate_object_at_time(
    obj: Dict[str, Any],
    t: float,
    char_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ik_map: Optional[Dict[str, Dict[str, Any]]] = None,
    all_char_ids: Optional[set] = None,
    obj_map: Optional[Dict[str, Dict[str, Any]]] = None,
    scene: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    在时刻 t 求值普通物体或绑定道具/枪械的世界变换与继承可见性：
    - 支持 parent_id 持久布景编组嵌套变换与祖先显隐继承；
    - 支持 layout_ref 布景参考系偏移（不破坏关键帧局部动画）；
    - 若存在有效 parent_binding (且 enabled != False)，默认挂接附件继承目标人物最终可见性（不回写修改道具自身 hidden/visible）；
    - 解绑物体保留自身独立可见性状态；防循环依赖；
    - 目标人物可见时按人物世界姿态挂点 × 道具局部偏移（或局部关键帧 coord_space='local'）计算真实 3D 世界变换。
    """
    if obj_map is None and isinstance(scene, dict):
        obj_map = {
            str(o.get("id")): o
            for o in (scene.get("objects") or [])
            if isinstance(o, dict) and o.get("id")
        }
    res = interpolate_object_at_time(obj, t, scene=scene)
    if str(obj.get("arch_role") or "") == "flame_volume":
        ph = float(obj.get("flame_phase", 0.0) or 0.0)
        sway_x = 4.6 * math.sin(float(t) * 3.1 + ph) + 2.2 * math.sin(float(t) * 5.3 + ph * 1.7)
        sway_z = 4.2 * math.cos(float(t) * 2.7 + ph * 1.3) + 1.8 * math.sin(float(t) * 4.8 + ph * 0.6)
        stretch_y = 1.0 + 0.075 * math.sin(float(t) * 3.8 + ph * 0.9) + 0.035 * math.cos(float(t) * 6.4 + ph * 2.1)
        breathe_xz = 1.0 - 0.035 * math.sin(float(t) * 3.8 + ph * 0.9)
        r_cur = list(res.get("rot") or [0.0, 0.0, 0.0])
        s_cur = list(res.get("size") or [0.3, 0.8, 0.3])
        p_cur = list(res.get("pos") or [0.0, 0.5, 0.0])
        old_sy = float(s_cur[1])
        new_sy = old_sy * stretch_y
        res["rot"] = [float(r_cur[0]) + sway_x, float(r_cur[1]), float(r_cur[2]) + sway_z]
        res["size"] = [float(s_cur[0]) * breathe_xz, new_sy, float(s_cur[2]) * breathe_xz]
        res["pos"] = [float(p_cur[0]) + 0.014 * math.sin(float(t) * 3.1 + ph), float(p_cur[1]) + 0.5 * (new_sy - old_sy), float(p_cur[2]) + 0.012 * math.cos(float(t) * 2.7 + ph)]

    def _apply_layout_and_hierarchy(out_res: Dict[str, Any]) -> Dict[str, Any]:
        M_reloc = _compose_layout_ref_matrix(obj.get("layout_ref"))
        M_parent, anc_vis = _get_hierarchy_parent_world_transform(obj, obj_map, t)
        if not anc_vis:
            out_res["ancestor_visible"] = False
            out_res["visible"] = False
        else:
            out_res.setdefault("ancestor_visible", True)
        if M_reloc is None and not obj.get("parent_id"):
            return out_res
        M_prefix = M_parent @ (M_reloc if M_reloc is not None else np.eye(4, dtype=np.float64))
        M_local = make_transform(out_res.get("pos", [0.0, 0.0, 0.0]), out_res.get("rot", [0.0, 0.0, 0.0]), 1.0)
        M_world = M_prefix @ M_local
        R_w, s_w = _decompose_prefix_matrix(M_world)
        base_sz = [float(x) for x in (out_res.get("size") or [1.0, 1.0, 1.0])[:3]]
        out_res["_local_size"] = list(base_sz)
        out_res["pos"] = [round(float(M_world[0, 3]), 5), round(float(M_world[1, 3]), 5), round(float(M_world[2, 3]), 5)]
        out_res["rot"] = matrix_to_euler_xyz_deg(R_w)
        out_res["size"] = [round(base_sz[0] * s_w[0], 5), round(base_sz[1] * s_w[1], 5), round(base_sz[2] * s_w[2], 5)]
        out_res["_M_world"] = M_world
        return out_res

    pb = obj.get("parent_binding")
    if not isinstance(pb, dict) or pb.get("enabled") is False:
        return _apply_layout_and_hierarchy(res)
    if pb.get("release_t") is not None and float(t) >= float(pb["release_t"]) - 1e-6:
        return _apply_layout_and_hierarchy(res)
    if pb.get("start_t") is not None and float(t) < float(pb["start_t"]) - 1e-6:
        return _apply_layout_and_hierarchy(res)
    tgt_id = str(pb.get("target_id") or "").strip()
    self_id = str(obj.get("id") or "")
    if not tgt_id or tgt_id == self_id:
        return _apply_layout_and_hierarchy(res)

    if all_char_ids is not None and tgt_id in all_char_ids:
        if not char_map or tgt_id not in char_map or char_map[tgt_id].get("visible") is False:
            res["ancestor_visible"] = False
            res["visible"] = False
            return res

    if not char_map or not ik_map or tgt_id not in char_map or tgt_id not in ik_map:
        return res

    tgt_char = char_map[tgt_id]
    if tgt_char.get("visible") is False:
        res["ancestor_visible"] = False
        res["visible"] = False
        return res
    res["ancestor_visible"] = True
    tgt_ik = ik_map[tgt_id]
    sock_name = str(pb.get("socket") or "hand_R")
    socket_tf = get_character_socket_transform(tgt_char, tgt_ik, sock_name)

    kfs = obj.get("keyframes") or []
    has_local_kfs = bool(kfs) and any(k.get("coord_space") == "local" for k in kfs if isinstance(k, dict))
    if has_local_kfs:
        l_pos = res.get("pos", pb.get("local_pos", [0.0, 0.018, -0.02]))
        l_rot = res.get("rot", pb.get("local_rot", [0.0, 0.0, 0.0]))
        base_sz = res.get("size", obj.get("size", [0.08, 0.12, 0.72]))
    else:
        l_pos = pb.get("local_pos", [0.0, 0.018, -0.02])
        l_rot = pb.get("local_rot", [0.0, 0.0, 0.0])
        base_sz = obj.get("size", [0.08, 0.12, 0.72])
    l_scale = pb.get("local_scale", [1.0, 1.0, 1.0])

    w_tf = socket_local_to_world(l_pos, l_rot, l_scale, base_sz, socket_tf)
    res["pos"] = w_tf["pos"]
    res["rot"] = w_tf["rot"]
    res["size"] = w_tf["size"]
    res["render_scale"] = w_tf["render_scale"]
    res["_R_world"] = w_tf["R_world"]
    res["_P_world"] = w_tf["P_world"]
    return res


def evaluate_scene_characters_at_time(
    scene: Dict[str, Any],
    t: float,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    在时刻 t 统一解算所有角色的：
    1. 关键帧分通道插值 (pos / rot / pose) 与阶跃可见性求值
    2. 脚掌接地防穿模 (foot_ground_clamp) 与 世界空间/局部空间锁脚 (foot_lock_mode: world / local)
    3. 手部接触绑定约束 (将角色 A 的手绑定到角色 B 的肩/前臂/手或局部点，支持起止帧、权重、渐入渐出与超距告警)
    返回: (char_dict_map, ik_res_map, reach_warnings)
    """
    fps = max(1.0, float(scene.get("fps", 24)))
    cur_frame = float(t) * fps
    char_map: Dict[str, Dict[str, Any]] = {}
    ik_map: Dict[str, Dict[str, Any]] = {}
    reach_warnings: List[Dict[str, Any]] = []
    all_char_ids = {
        str(o.get("id"))
        for o in scene.get("objects", [])
        if isinstance(o, dict) and o.get("type") == "character" and o.get("id")
    }

    # 0. 预解算所有带动画或载具物体的当前时刻变换（含编组 parent_id 与布局参考系 layout_ref），供乘员随车联动使用
    obj_map: Dict[str, Dict[str, Any]] = {
        str(o.get("id")): o
        for o in scene.get("objects", [])
        if isinstance(o, dict) and o.get("id")
    }
    vehicle_state_map: Dict[str, Dict[str, Any]] = {}
    for obj in scene.get("objects", []):
        if obj.get("type") not in ("character", "scene_group") and obj.get("id"):
            vehicle_state_map[obj["id"]] = evaluate_object_at_time(
                obj, t, all_char_ids=all_char_ids, obj_map=obj_map, scene=scene
            )

    # 1. 插值所有角色并执行随车联动、编组层级/布局变换、脚掌接地 & 世界空间锁脚（按求值后 visible 过滤）
    for obj in scene.get("objects", []):
        if obj.get("type") != "character":
            continue
        cid = obj.get("id", "")
        c_state = interpolate_character_at_time(obj, t, scene=scene)
        M_reloc_c = _compose_layout_ref_matrix(obj.get("layout_ref"))
        M_parent_c, anc_vis_c = _get_hierarchy_parent_world_transform(obj, obj_map, t)
        if not anc_vis_c:
            c_state["ancestor_visible"] = False
            c_state["visible"] = False
            continue
        if not c_state.get("visible", True):
            continue
        mounted_vid = c_state.get("mounted_vehicle_id")
        if mounted_vid and mounted_vid in vehicle_state_map and (c_state.get("follow_mounted_vehicle") is not False):
            veh_st = vehicle_state_map[mounted_vid]
            v_pos = veh_st.get("pos", [0.0, 1.01, 0.0])
            v_rot = veh_st.get("rot", [0.0, 0.0, 0.0])
            v_size = veh_st.get("size", [1.78, 2.02, 4.02])
            has_char_kfs = bool(obj.get("keyframes"))
            # 若乘员带有车内局部微调关键帧（如第 2 镜后排左乘员俯身靠近，局部坐标在 [-1.5, 1.5]m 内），叠加局部偏移；否则使用固定 seat_offset
            c_pos_raw = c_state.get("pos", [0.0, 0.42, 0.0])
            if has_char_kfs and abs(float(c_pos_raw[0])) < 1.6 and abs(float(c_pos_raw[2])) < 2.2:
                seat_off = c_pos_raw
            else:
                seat_off = c_state.get("seat_offset") or c_pos_raw
            ox, oy, oz = float(seat_off[0]), float(seat_off[1]), float(seat_off[2])
            v_yaw_deg = float(v_rot[1]) if len(v_rot) >= 2 else 0.0
            v_yaw_rad = math.radians(v_yaw_deg)
            cy_v, sy_v = math.cos(v_yaw_rad), math.sin(v_yaw_rad)
            v_base_y = float(v_pos[1]) - 0.5 * float(v_size[1])
            c_state["pos"] = [
                round(float(v_pos[0]) + ox * cy_v + oz * sy_v, 5),
                round(v_base_y + oy, 5),
                round(float(v_pos[2]) - ox * sy_v + oz * cy_v, 5),
            ]
            if has_char_kfs and abs(float(c_pos_raw[0])) < 1.6 and abs(float(c_pos_raw[2])) < 2.2:
                local_ry = float(c_state.get("rot", [0.0, 0.0, 0.0])[1])
            else:
                local_ry = float(c_state.get("seat_rot_offset", c_state.get("rot", [0.0, 0.0, 0.0])[1]))
            c_state["rot"] = [0.0, round(v_yaw_deg + local_ry, 4), 0.0]
        elif M_reloc_c is not None or obj.get("parent_id"):
            M_pref_c = M_parent_c @ (M_reloc_c if M_reloc_c is not None else np.eye(4, dtype=np.float64))
            M_loc_c = make_transform(c_state.get("pos", [0.0, 0.0, 0.0]), c_state.get("rot", [0.0, 0.0, 0.0]), 1.0)
            M_w_c = M_pref_c @ M_loc_c
            R_w_c, s_w_c = _decompose_prefix_matrix(M_w_c)
            e_w_c = matrix_to_euler_xyz_deg(R_w_c)
            c_state["pos"] = [round(float(M_w_c[0, 3]), 5), round(float(M_w_c[1, 3]), 5), round(float(M_w_c[2, 3]), 5)]
            c_state["rot"] = [0.0, round(float(e_w_c[1]), 4), 0.0]
            sc_orig = _get_char_scale_xyz(c_state)
            c_state["scale"] = [
                round(sc_orig[0] * s_w_c[0], 4),
                round(sc_orig[1] * s_w_c[1], 4),
                round(sc_orig[2] * s_w_c[2], 4),
            ]

        targets = c_state.get("targets", {})
        _, sy, _ = _get_char_scale_xyz(c_state)
        scale = sy

        # 区分世界空间锁脚 (world) 与角色局部空间锁脚 (local)
        lock_mode = c_state.get("foot_lock_mode", "local")
        world_pins = c_state.get("world_pinned_feet", {}) or {}
        lock_start = float(c_state.get("foot_lock_start_frame", 0))
        lock_end = float(c_state.get("foot_lock_end_frame", 999999))
        if lock_mode == "world" and (lock_start <= cur_frame <= lock_end):
            for fk in ("foot_L", "foot_R"):
                w_pt = world_pins.get(fk)
                if isinstance(w_pt, (list, tuple)) and len(w_pt) >= 3:
                    targets[fk] = _char_world_to_local(c_state, np.asarray(w_pt[:3], dtype=np.float64))

        # 脚掌接地约束 (防止脚底陷入地面 Y < 0，保持脚踝高度 >= 0.04m)
        if c_state.get("foot_ground_clamp", True):
            min_local_y = max(0.0, (0.04 * scale - float(c_state["pos"][1])) / scale)
            for fk in ("foot_L", "foot_R"):
                if fk in targets and isinstance(targets[fk], list) and len(targets[fk]) >= 3:
                    targets[fk] = list(targets[fk])
                    if targets[fk][1] < min_local_y:
                        targets[fk][1] = round(float(min_local_y), 4)

        c_state["targets"] = targets
        char_map[cid] = c_state
        ik_map[cid] = solve_mannequin_ik(c_state)

    # 2. 收集场景级与角色级手部接触约束
    all_constraints: List[Dict[str, Any]] = []
    for sc_c in scene.get("constraints", []) or []:
        if isinstance(sc_c, dict):
            all_constraints.append(sc_c)
    for cid, c_state in char_map.items():
        for ch_c in c_state.get("constraints", []) or []:
            if isinstance(ch_c, dict):
                c_copy = dict(ch_c)
                c_copy.setdefault("source_char_id", cid)
                all_constraints.append(c_copy)

    # 3. 逐条解算手部接触绑定并检测肢体最大可达距离
    for c_idx, c_item in enumerate(all_constraints):
        w_eff = compute_constraint_effective_weight(c_item, cur_frame)
        if w_eff <= 1e-5:
            continue
        src_id = c_item.get("source_char_id", "")
        tgt_id = c_item.get("target_char_id", "")
        src_hand = c_item.get("source_hand", "hand_R")
        if src_hand not in ("hand_L", "hand_R"):
            src_hand = "hand_R"
        tgt_part = c_item.get("target_part", "shoulder_L")
        local_offset = c_item.get("local_offset", [0.0, 0.0, 0.0])

        if src_id not in char_map or tgt_id not in char_map:
            continue
        src_char = char_map[src_id]
        tgt_char = char_map[tgt_id]
        src_ik = ik_map[src_id]
        tgt_ik = ik_map[tgt_id]

        # 计算目标角色部位的世界坐标
        bind_world = get_target_character_part_world(tgt_char, tgt_ik, tgt_part, local_offset)
        orig_local = src_char["targets"].get(src_hand, [-0.16, 0.54, 0.24] if src_hand == "hand_R" else [0.16, 0.54, 0.24])
        orig_world = _char_local_to_world(src_char, orig_local)
        desired_world = orig_world * (1.0 - w_eff) + bind_world * w_eff

        # 检查源角色肩膀到目标点的可达距离（绝不无限拉长手臂！）
        sh_key = "shoulder_L" if src_hand == "hand_L" else "shoulder_R"
        shoulder_world = src_ik["joints"][sh_key]
        _, sy_a, _ = _get_char_scale_xyz(src_char)
        scale_a = sy_a
        max_arm_reach = (MANNEQUIN_SPEC["upper_arm_len"] + MANNEQUIN_SPEC["forearm_len"]) * scale_a
        diff_vec = desired_world - shoulder_world
        dist = float(np.linalg.norm(diff_vec))

        if dist > max_arm_reach + 1e-3:
            dir_u = diff_vec / max(1e-6, dist)
            clamped_world = shoulder_world + dir_u * (max_arm_reach - 1e-4)
            exceed_m = dist - max_arm_reach
            reach_warnings.append({
                "constraint_id": c_item.get("id", f"c_{c_idx}"),
                "source_char_id": src_id,
                "source_char_name": src_char.get("name", src_id),
                "source_hand": src_hand,
                "target_char_id": tgt_id,
                "target_char_name": tgt_char.get("name", tgt_id),
                "target_part": tgt_part,
                "distance_m": round(dist, 3),
                "max_reach_m": round(max_arm_reach, 3),
                "exceed_m": round(exceed_m, 3),
                "weight_effective": w_eff,
                "frame": int(round(cur_frame)),
                "t": round(float(t), 3),
                "desired_world": [round(float(x), 4) for x in desired_world],
                "clamped_world": [round(float(x), 4) for x in clamped_world],
                "message": (
                    f"⚠️ 超出肢体可达距离：{src_char.get('name', src_id)}.{src_hand} -> "
                    f"{tgt_char.get('name', tgt_id)}.{tgt_part} 距离 {dist:.2f}m > 最大臂展 {max_arm_reach:.2f}m "
                    f"(超出 {exceed_m:.2f}m)，已限制在最大臂长，不拉长肢体"
                ),
            })
        else:
            clamped_world = desired_world

        src_char["targets"][src_hand] = _char_world_to_local(src_char, clamped_world)
        ik_map[src_id] = solve_mannequin_ik(src_char)

    # 4. 单向双手持枪解算 (主手驱动枪 -> 枪世界变换 -> 副手跟随枪上第二握持点，绝不反向递归驱动枪)
    for obj in scene.get("objects", []):
        if obj.get("type") == "character" or obj.get("is_helper") is True:
            continue
        pb = obj.get("parent_binding")
        if not isinstance(pb, dict) or pb.get("enabled") is False or not pb.get("two_handed"):
            continue
        if pb.get("release_t") is not None and float(t) >= float(pb["release_t"]) - 1e-6:
            continue
        if pb.get("start_t") is not None and float(t) < float(pb["start_t"]) - 1e-6:
            continue
        tgt_id = str(pb.get("target_id") or "").strip()
        sock_name = str(pb.get("socket") or "hand_R").strip()
        if not tgt_id or tgt_id not in char_map or sock_name not in ("hand_R", "hand_L"):
            continue
        tgt_char = char_map[tgt_id]
        tgt_ik = ik_map[tgt_id]
        sec_hand = str(pb.get("secondary_hand") or ("hand_L" if sock_name == "hand_R" else "hand_R"))
        if sec_hand not in ("hand_L", "hand_R") or sec_hand == sock_name:
            sec_hand = "hand_L" if sock_name == "hand_R" else "hand_R"
        eval_gun = evaluate_object_at_time(obj, t, char_map, ik_map, all_char_ids=all_char_ids)
        if not eval_gun.get("visible", True):
            continue
        R_gun = eval_gun.get("_R_world")
        P_gun = eval_gun.get("_P_world")
        if R_gun is None or P_gun is None:
            continue
        r_scale = float(eval_gun.get("render_scale", 1.0))
        sec_grip_local = np.asarray((pb.get("secondary_grip_local") or [0.0, -0.018, 0.26])[:3], dtype=np.float64)
        desired_sec_world = P_gun + (R_gun @ (sec_grip_local * r_scale))
        sh_key = "shoulder_L" if sec_hand == "hand_L" else "shoulder_R"
        shoulder_world = np.asarray(tgt_ik["joints"][sh_key], dtype=np.float64)
        _, sy_c, _ = _get_char_scale_xyz(tgt_char)
        max_arm_reach = (MANNEQUIN_SPEC["upper_arm_len"] + MANNEQUIN_SPEC["forearm_len"]) * sy_c * 0.995
        diff_vec = desired_sec_world - shoulder_world
        dist = float(np.linalg.norm(diff_vec))
        if dist > max_arm_reach + 1e-3:
            dir_u = diff_vec / max(1e-6, dist)
            clamped_sec_world = shoulder_world + dir_u * max_arm_reach
            exceed_m = dist - max_arm_reach
            reach_warnings.append({
                "constraint_id": f"two_handed_{obj.get('id', 'gun')}",
                "source_char_id": tgt_id,
                "source_char_name": tgt_char.get("name", tgt_id),
                "source_hand": sec_hand,
                "target_char_id": str(obj.get("id", "gun")),
                "target_char_name": obj.get("name", "枪械"),
                "target_part": "secondary_grip",
                "distance_m": round(dist, 3),
                "max_reach_m": round(max_arm_reach, 3),
                "exceed_m": round(exceed_m, 3),
                "weight_effective": 1.0,
                "frame": int(round(cur_frame)),
                "t": round(float(t), 3),
                "desired_world": [round(float(x), 4) for x in desired_sec_world],
                "clamped_world": [round(float(x), 4) for x in clamped_sec_world],
                "message": (
                    f"⚠️ 双手持枪副手超出可达距离：{tgt_char.get('name', tgt_id)}.{sec_hand} -> "
                    f"{obj.get('name', '枪械')}前护木 距离 {dist:.2f}m > 最大臂展 {max_arm_reach:.2f}m，已限制在最大臂长"
                ),
            })
        else:
            clamped_sec_world = desired_sec_world
        tgt_char["targets"][sec_hand] = _char_world_to_local(tgt_char, clamped_sec_world)
        ik_map[tgt_id] = solve_mannequin_ik(tgt_char)

    return char_map, ik_map, reach_warnings


def _subdivide_polygon_world(poly: np.ndarray, max_edge: float = 0.85) -> List[np.ndarray]:
    """对大尺寸四边形面片（如高大树干、地面、长墙）细分，保证画家算法遮挡排序与 Three.js 深度缓冲一致"""
    if len(poly) != 4:
        return [poly]
    p0, p1, p2, p3 = poly[0], poly[1], poly[2], poly[3]
    u_len = max(float(np.linalg.norm(p1 - p0)), float(np.linalg.norm(p2 - p3)))
    v_len = max(float(np.linalg.norm(p3 - p0)), float(np.linalg.norm(p2 - p1)))
    nu = max(1, min(6, int(math.ceil(u_len / max_edge))))
    nv = max(1, min(6, int(math.ceil(v_len / max_edge))))
    if nu == 1 and nv == 1:
        return [poly]
    sub_faces: List[np.ndarray] = []
    for iu in range(nu):
        u0, u1 = float(iu) / nu, float(iu + 1) / nu
        for iv in range(nv):
            v0, v1 = float(iv) / nv, float(iv + 1) / nv
            q00 = (1 - u0) * (1 - v0) * p0 + u0 * (1 - v0) * p1 + u0 * v0 * p2 + (1 - u0) * v0 * p3
            q10 = (1 - u1) * (1 - v0) * p0 + u1 * (1 - v0) * p1 + u1 * v0 * p2 + (1 - u1) * v0 * p3
            q11 = (1 - u1) * (1 - v1) * p0 + u1 * (1 - v1) * p1 + u1 * v1 * p2 + (1 - u1) * v1 * p3
            q01 = (1 - u0) * (1 - v1) * p0 + u0 * (1 - v1) * p1 + u0 * v1 * p2 + (1 - u0) * v1 * p3
            sub_faces.append(np.array([q00, q10, q11, q01], dtype=np.float64))
    return sub_faces


def render_abstract_block_frame(scene: Dict[str, Any], t: float, width: int = 960, height: int = 540) -> Image.Image:
    """
    方案 B (abstract_block)：无边线极简方块 + 角色色块渲染
    - 0 黑色线条（outline=None）
    - 树木（id 以 tree_ 开头）：简化为直立柱杆（0.44m × 7.5m × 0.44m），看不出松树锥体，仅提供前后遮挡与运动视差
    - 车辆（type == 'uaz469'）：远景为上下两个简单方盒（无车轮/车灯/窗框）；车内近景为极简底板、两侧低挡板与座椅长条块
    - 角色（type == 'character'）：上下两个方块（躯干方块 + 头部方块），保留角色身份色区分不同人物，无四肢细节
    """
    cam_state = evaluate_camera_state(scene, t)
    cam_pos = cam_state["pos"]
    cam_tar = cam_state["target"]
    fov = cam_state["fov"]
    cam_roll = cam_state.get("roll_deg", 0.0)
    cam_pos_np = np.array(cam_pos, dtype=float)
    cam_tar_np = np.array(cam_tar, dtype=float)

    char_map, ik_map, _ = evaluate_scene_characters_at_time(scene, t)
    polys: List[Tuple[np.ndarray, Tuple[int, int, int], str]] = []

    # 1. 地面与环境体块
    g_pts = np.array([
        [-55.0, -0.02, -75.0],
        [ 55.0, -0.02, -75.0],
        [ 55.0, -0.02,  30.0],
        [-55.0, -0.02,  30.0],
    ], dtype=float)
    polys.append((g_pts, (198, 204, 212), "ground"))

    obj_map_abs: Dict[str, Dict[str, Any]] = {
        str(o.get("id")): o
        for o in scene.get("objects", [])
        if isinstance(o, dict) and o.get("id")
    }
    all_char_ids_abs = {
        str(o.get("id"))
        for o in scene.get("objects", [])
        if isinstance(o, dict) and o.get("type") == "character" and o.get("id")
    }
    for obj in scene.get("objects", []):
        if obj.get("is_helper") is True:
            continue
        oid = str(obj.get("id", ""))
        otype = str(obj.get("type", "box"))
        if otype in ("character", "scene_group"):
            continue
        if otype == "uaz469":
            v_tf = evaluate_object_at_time(obj, t, char_map, ik_map, all_char_ids=all_char_ids_abs, obj_map=obj_map_abs, scene=scene)
            if not v_tf.get("visible", True):
                continue
            vpos = v_tf["pos"]
            vrot = v_tf["rot"]
            vsize = v_tf["size"]
            sx, sy, sz = float(vsize[0]), float(vsize[1]), float(vsize[2])
            ground_y = float(vpos[1]) - sy * 0.5
            T_veh = make_transform([vpos[0], ground_y, vpos[2]], vrot, 1.0)
            # 鲜明半透明车体色：主吉普为醒目翠绿/军青 (#289664)，后随车为醒目琥珀青蓝 (#287E9E)，配合 4 个深色车轮与车头引擎盖+车顶，一眼能认出是车且能透视车内 4 个彩色方块人
            body_rgb = (40, 150, 100) if oid == "uaz_lead" else (42, 126, 158)
            hood_rgb = (34, 134, 88) if oid == "uaz_lead" else (36, 112, 142)
            roof_rgb = (56, 168, 116) if oid == "uaz_lead" else (58, 144, 176)
            wheel_rgb = (36, 40, 46)
            seat_rgb = (90, 110, 102)

            def _add_box_front_faces(Tp: np.ndarray, bsx: float, bsy: float, bsz: float, col_rgb: Tuple[int, int, int], tag: str) -> None:
                b_center = transform_points(Tp, np.array([[0.0, 0.0, 0.0]], dtype=float))[0]
                for face in make_box_faces_local(bsx, bsy, bsz):
                    fw = transform_points(Tp, face)
                    fc = np.mean(fw, axis=0)
                    n_vec = np.cross(fw[1] - fw[0], fw[2] - fw[0])
                    if float(np.dot(n_vec, fc - b_center)) < 0.0:
                        n_vec = -n_vec
                    if float(np.dot(n_vec, cam_pos_np - fc)) > 1e-6:
                        polys.append((fw, col_rgb, tag))

            cam_dist_veh = float(np.linalg.norm(cam_pos_np - np.array([float(vpos[0]), float(vpos[1]), float(vpos[2])], dtype=float)))

            # 1) 4 个深色车轮块（不透明 0.92，一眼识别车辆姿态与接地）
            for wx in (-sx * 0.46, sx * 0.46):
                for wz in (sz * 0.30, -sz * 0.30):
                    Tw = T_veh @ make_transform([wx, 0.38, wz], [0, 0, 0], 1.0)
                    _add_box_front_faces(Tw, 0.26, 0.74, 0.76, wheel_rgb, "wheel")

            # 2) 车内底盘与前后排座椅（半透明 0.58）
            for lx, ly, lz, bsx, bsy, bsz, c in [
                (0.0, 0.26, -0.10, sx * 0.92, 0.14, sz * 0.88, hood_rgb),
                (-0.34, 0.54, -0.22, 0.42, 0.48, 0.12, seat_rgb),
                ( 0.34, 0.54, -0.22, 0.42, 0.48, 0.12, seat_rgb),
                ( 0.00, 0.56, -1.42, 1.14, 0.50, 0.14, seat_rgb),
            ]:
                Tp = T_veh @ make_transform([lx, ly, lz], [0, 0, 0], 1.0)
                _add_box_front_faces(Tp, bsx, bsy, bsz, c, "veh_frame")

            # 3) 醒目半透明车头引擎盖 (+Z 前方) + 半透明车身侧围 + 半透明车顶棚与立柱
            car_shell_parts = [
                # 前引擎盖 (突出的车头，当镜头从车头穿入前挡风玻璃时平滑淡出，避免近裁面闪烁)
                (0.0, 0.62, sz * 0.27, sx * 0.92, 0.54, sz * 0.38, hood_rgb, "veh_hood_lead" if oid == "uaz_lead" else "veh_glass"),
                # 左右两侧半高车门围板
                (-sx * 0.45, 0.60, -sz * 0.14, 0.10, 0.54, sz * 0.58, body_rgb, "veh_glass"),
                ( sx * 0.45, 0.60, -sz * 0.14, 0.10, 0.54, sz * 0.58, body_rgb, "veh_glass"),
                # 车尾围板
                (0.0, 0.62, -sz * 0.43, sx * 0.92, 0.58, 0.10, body_rgb, "veh_glass"),
                # 车顶盖平板
                (0.0, sy * 0.95, -sz * 0.14, sx * 0.90, 0.09, sz * 0.60, roof_rgb, "veh_glass"),
                # 四根极简车顶立柱
                (-sx * 0.43, sy * 0.68,  sz * 0.12, 0.08, sy * 0.48, 0.08, roof_rgb, "veh_glass"),
                ( sx * 0.43, sy * 0.68,  sz * 0.12, 0.08, sy * 0.48, 0.08, roof_rgb, "veh_glass"),
                (-sx * 0.43, sy * 0.68, -sz * 0.41, 0.08, sy * 0.48, 0.08, roof_rgb, "veh_glass"),
                ( sx * 0.43, sy * 0.68, -sz * 0.41, 0.08, sy * 0.48, 0.08, roof_rgb, "veh_glass"),
            ]
            for lx, ly, lz, bsx, bsy, bsz, c, tag in car_shell_parts:
                Tp = T_veh @ make_transform([lx, ly, lz], [0, 0, 0], 1.0)
                _add_box_front_faces(Tp, bsx, bsy, bsz, c, tag)
            continue

        o_tf = evaluate_object_at_time(obj, t, char_map, ik_map, all_char_ids=all_char_ids_abs, obj_map=obj_map_abs, scene=scene)
        if not o_tf.get("visible", True):
            continue
        opos = o_tf["pos"]
        orot = o_tf["rot"]
        osize = o_tf["size"]
        sx, sy, sz = float(osize[0]), float(osize[1]), float(osize[2])
        if oid.startswith("env_rut_"):
            continue
        elif oid.startswith("tree_") or oid.startswith("env_tree_"):
            pole_w = 0.44
            pole_h = 7.5
            T = make_transform([opos[0], pole_h * 0.5, opos[2]], [0, 0, 0], 1.0)
            for face in make_box_faces_local(pole_w, pole_h, pole_w):
                polys.append((transform_points(T, face), (136, 143, 154), "pole"))
        elif oid.startswith("env_road_"):
            T = make_transform(opos, orot, 1.0)
            for face in make_box_faces_local(sx, sy, sz):
                polys.append((transform_points(T, face), (156, 163, 174), "road"))
        elif oid.startswith("env_snowbank_"):
            T = make_transform([opos[0], 0.06, opos[2]], orot, 1.0)
            for face in make_box_faces_local(sx * 0.85, sy * 0.22, sz):
                polys.append((transform_points(T, face), (212, 217, 224), "slope"))
        elif oid.startswith("env_fallen_log_"):
            T = make_transform(opos, orot, 1.0)
            for face in make_box_faces_local(sx * 1.1, sy, sz * 1.1):
                polys.append((transform_points(T, face), (142, 148, 158), "pole"))
        else:
            T = make_transform(opos, orot, 1.0)
            col = hex_to_rgb(obj.get("color", "#D5D9DF"))
            for face in make_box_faces_local(sx, sy, sz):
                polys.append((transform_points(T, face), col, "prop"))

    # 2. 角色：上下两个极简方块 + 角色身份色
    for cid, cdict in char_map.items():
        if cdict.get("hidden") or cdict.get("visible") is False:
            continue
        cpos = cdict["pos"]
        crot = cdict.get("rot", [0, 0, 0])
        s_val = cdict.get("scale", 0.82)
        sc = float(s_val[0]) if isinstance(s_val, (list, tuple)) else float(s_val)
        rgb = hex_to_rgb(cdict.get("color", "#D4B56C"))
        targets = cdict.get("targets", {})
        chest_off = targets.get("chest", [0.0, 0.92, 0.0])
        lean_x = float(chest_off[0]) * 0.85

        T_char = make_transform(cpos, crot, 1.0)
        T_torso = T_char @ make_transform([lean_x * 0.6, 0.66 * sc, 0.0], [0, 0, 0], 1.0)
        for face in make_box_faces_local(0.36 * sc, 0.52 * sc, 0.24 * sc):
            polys.append((transform_points(T_torso, face), rgb, "char"))
        head_rgb = (min(255, int(rgb[0] * 1.06)), min(255, int(rgb[1] * 1.06)), min(255, int(rgb[2] * 1.06)))
        T_head = T_char @ make_transform([lean_x, 1.06 * sc, 0.02], [0, 0, 0], 1.0)
        for face in make_box_faces_local(0.24 * sc, 0.25 * sc, 0.24 * sc):
            polys.append((transform_points(T_head, face), head_rgb, "char"))

    right, up, forward = compute_camera_basis(cam_pos_np, cam_tar_np, cam_roll)

    aspect = float(width) / max(1.0, float(height))
    fov_rad = math.radians(fov)
    f_scale = 1.0 / math.tan(fov_rad * 0.5)

    env = scene.get("environment") or {}
    bg_color = hex_to_rgb(env.get("sky", "#D6DCE4")) if isinstance(env, dict) and env.get("sky") else (214, 220, 228)
    fog_near = float(env.get("fog_near", 10.0)) if isinstance(env, dict) else 10.0
    fog_far = float(env.get("fog_far", 48.0)) if isinstance(env, dict) else 48.0

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img, "RGBA")

    alpha_map = {
        "veh_glass": 102,      # ~40% 透明度：鲜明半透明车壳，清晰透出车内彩色方块人
        "veh_hood_lead": 102,  # 主车前引擎盖：远景清晰，近景穿入挡风玻璃时按距离平滑淡出
        "veh_frame": 148,      # ~58% 透明度：底盘与座椅
        "wheel": 235,          # ~92% 不透明：深色车轮
    }
    bg_layer_map = {
        "ground": 3,
        "slope": 2,
        "road": 1,
    }
    no_subdiv_kinds = {"veh_glass", "veh_hood_lead", "veh_frame", "wheel", "char"}

    render_list = []
    for pts_w, base_rgb, kind in polys:
        if kind in no_subdiv_kinds:
            sub_polys = [pts_w]
        else:
            sub_polys = _subdivide_polygon_world(pts_w, max_edge=1.6 if kind != "ground" else 5.0)
        base_alpha = alpha_map.get(kind, 255)
        bg_prio = bg_layer_map.get(kind, 0)
        for sp in sub_polys:
            rel = sp - cam_pos_np
            x_cam = rel @ right
            y_cam = rel @ up
            z_cam = rel @ forward
            pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
            clipped = clip_polygon_near(pts_cam, near=0.18)
            if len(clipped) < 3:
                continue
            z_mean = float(np.mean(clipped[:, 2]))
            if z_mean > 75.0:
                continue
            alpha_val = base_alpha
            if kind == "veh_hood_lead":
                # 当摄影机从车头上方穿入前挡风玻璃 (z_mean 从 2.2m 靠近到 0.9m) 时平滑淡出前引擎盖，避免近裁面切变闪烁
                u_fade = max(0.0, min(1.0, (z_mean - 0.90) / 1.30))
                alpha_val = int(round(base_alpha * (u_fade * u_fade * (3.0 - 2.0 * u_fade))))
                if alpha_val <= 2:
                    continue
            proj_pts = []
            for p in clipped:
                ndc_x = (p[0] * f_scale / aspect) / p[2]
                ndc_y = (p[1] * f_scale) / p[2]
                sx_px = (ndc_x + 1.0) * 0.5 * width
                sy_px = (1.0 - ndc_y) * 0.5 * height
                proj_pts.append((float(sx_px), float(sy_px)))
            shaded = shade_polygon(base_rgb, sp)
            fog_u = max(0.0, min(0.78, (z_mean - fog_near) / max(1.0, fog_far - fog_near)))
            fill_rgba = (
                int(round(shaded[0] * (1.0 - fog_u) + bg_color[0] * fog_u)),
                int(round(shaded[1] * (1.0 - fog_u) + bg_color[1] * fog_u)),
                int(round(shaded[2] * (1.0 - fog_u) + bg_color[2] * fog_u)),
                alpha_val,
            )
            render_list.append((bg_prio, z_mean, proj_pts, fill_rgba))

    render_list.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for _, _, proj_pts, fill_rgba in render_list:
        draw.polygon(proj_pts, fill=fill_rgba, outline=None)

    return img


def render_su_minimal_frame(scene: Dict[str, Any], t: float, width: int = 1280, height: int = 720) -> Image.Image:
    """
    SU 极简彩色预演渲染器 (su_minimal)：
    - 树：单个低面数圆锥体（锥尖朝上、底面落地 y=0），柔绿色系，无树干/树枝/积雪/圆环线；
    - 人物：极简球头人偶（球头 + 躯干 + 4简单四肢 + 1头前朝向短线），无五官/帽子/独眼罩/关节球；
      鲍里斯用暖橙 (#D97A36)，后方四人分别用柔蓝 (#5B8DBE)、柔绿 (#63A675)、浅紫 (#9378B8)、土黄 (#C8A24B)；
    - 枪：3 个基础体（枪托方块 + 机匣方块 + 枪管圆柱，枪口用圆柱前端面表示），独立深钢灰身份色 (#4A5568)；
    - 地面浅灰 (#D8DCE2)、道路略深 (#B8C0CA)，无地面密集网格、无三角边线与穿透虚线；
    - 支持 10–14s 推进移焦示意（focus_shift_10_14：焦点由鲍里斯脸部平滑移向枪口）。
    """
    sc = scene if scene.get("_su_minimal_normalized") else convert_scene_to_su_minimal(scene)
    sc["_su_minimal_normalized"] = True

    cam_state = evaluate_camera_state(sc, t)
    cam = RenderCamera(cam_state["pos"], cam_state["target"], cam_state["fov"], width, height, roll_deg=cam_state.get("roll_deg", 0.0))

    env = sc.get("environment") or {}
    sky_rgb = hex_to_rgb(env.get("sky", SU_MINIMAL_CONFIG["sky_color"]))
    ground_rgb = hex_to_rgb(env.get("ground", SU_MINIMAL_CONFIG["ground_color"]))

    # 分层收集多边形以支持 10-14s 从脸部移焦到枪口的清晰度过渡：
    # layer 0 = 背景层 (地面/道路/单锥体树/后方四人)
    # layer 1 = 中景焦点层 (鲍里斯主体 + 头前朝向短线)
    # layer 2 = 前景焦点层 (鲍里斯步枪)
    has_focus_shift = bool(sc.get("focus_shift_10_14", False))
    layer_polys: Dict[int, List[Tuple[np.ndarray, Tuple[int, int, int], bool]]] = {0: [], 1: [], 2: []}

    # 1. 地面与道路（大平面，不加网格线）
    g_pts = np.array(
        [[-60.0, 0.0, -80.0], [60.0, 0.0, -80.0], [60.0, 0.0, 40.0], [-60.0, 0.0, 40.0]],
        dtype=np.float64,
    )
    for sub_g in _subdivide_polygon_world(g_pts, max_edge=8.0):
        layer_polys[0].append((sub_g, ground_rgb, False))

    char_map, ik_map, _ = evaluate_scene_characters_at_time(sc, t)

    def _add_prim_to_layer(
        layer_id: int,
        shape: str,
        T_mat: np.ndarray,
        sx_p: float,
        sy_p: float,
        sz_p: float,
        rgb_p: Tuple[int, int, int],
        is_gun_barrel: bool = False,
    ) -> None:
        shp = (shape or "box").lower()
        if shp in ("box", "wall", "plane"):
            lfaces = make_box_faces_local(sx_p, max(0.02, sy_p) if shp == "plane" else sy_p, sz_p)
        elif shp == "sphere":
            raw_sf = make_sphere_faces_world(np.array([0.0, 0.0, 0.0], dtype=float), 0.5, lat_steps=7, lon_steps=12)
            sc_v = np.array([sx_p, sy_p, sz_p], dtype=float)
            lfaces = [f * sc_v for f in raw_sf]
        elif shp == "cone":
            # 单锥体树：低面数圆锥 (10边形底)，无分段环线
            lfaces = make_cone_faces_local(sx_p, sy_p, sz_p, segments=10)
        elif shp in ("prism", "wedge"):
            lfaces = make_prism_faces_local(sx_p, sy_p, sz_p)
        else:
            lfaces = make_cylinder_faces_local(sx_p, sy_p, sz_p, segments=12)

        for lf in lfaces:
            w_poly = transform_points(T_mat, lf)
            f_col = rgb_p
            if is_gun_barrel and len(lf) > 4:
                # 枪管圆柱的前端面（枪口端面）使用深色明度区分，不额外增加几何部件即可清晰读出枪口朝向
                if float(np.mean(lf[:, 1])) > 0.10:
                    f_col = (max(22, int(rgb_p[0] * 0.42)), max(26, int(rgb_p[1] * 0.42)), max(34, int(rgb_p[2] * 0.45)))
            layer_polys[layer_id].append((w_poly, f_col, True))

    obj_map_su: Dict[str, Dict[str, Any]] = {
        str(o.get("id")): o
        for o in sc.get("objects", [])
        if isinstance(o, dict) and o.get("id")
    }
    all_char_ids_su = {
        str(o.get("id"))
        for o in sc.get("objects", [])
        if isinstance(o, dict) and o.get("type") == "character" and o.get("id")
    }
    plan_mode_su = str((sc.get("export_plan") or {}).get("priority_mode") or "balanced").lower()
    vis_tree_cap_su = 300 if plan_mode_su == "fast" else (1000 if plan_mode_su == "composition" else 600)

    for idx_obj, obj in enumerate(sc.get("objects", [])):
        if obj.get("is_helper") is True:
            continue
        oid = str(obj.get("id", ""))
        otype = str(obj.get("type", "box"))
        if otype == "scene_group":
            continue
        role = str(obj.get("arch_role", ""))
        if plan_mode_su == "fast" and obj.get("is_decorative_rubble") is True and not obj.get("protected") and not obj.get("locked"):
            continue

        if otype == "uaz469":
            # 车辆仅保留车身、座舱、必要车轮（最多 6 个基础体）
            obj_t = evaluate_object_at_time(obj, t, char_map, ik_map, all_char_ids=all_char_ids_su, obj_map=obj_map_su, scene=scene)
            if not obj_t.get("visible", True):
                continue
            vpos = obj_t.get("pos", [0.0, 1.01, 0.0])
            vrot = obj_t.get("rot", [0.0, 0.0, 0.0])
            vsize = obj_t.get("size", [1.78, 2.02, 4.02])
            sx, sy, sz = float(vsize[0]), float(vsize[1]), float(vsize[2])
            v_rgb = hex_to_rgb(obj_t.get("color", get_su_minimal_identity_color(oid, "vehicle")))
            cabin_rgb = (min(255, int(v_rgb[0] * 1.12)), min(255, int(v_rgb[1] * 1.12)), min(255, int(v_rgb[2] * 1.12)))
            wheel_rgb = (52, 58, 66)
            T_veh = make_transform([vpos[0], float(vpos[1]) - sy * 0.5, vpos[2]], vrot, 1.0)
            # 1) 车身下体
            _add_prim_to_layer(0, "box", T_veh @ make_transform([0.0, sy * 0.34, 0.0], [0, 0, 0], 1.0), sx * 0.94, sy * 0.38, sz * 0.94, v_rgb)
            # 2) 座舱上体
            _add_prim_to_layer(0, "box", T_veh @ make_transform([0.0, sy * 0.68, -sz * 0.10], [0, 0, 0], 1.0), sx * 0.88, sy * 0.34, sz * 0.56, cabin_rgb)
            # 3-6) 四个简单车轮（随行进弧长按轮半径旋转滚动）
            wheel_roll_deg = compute_wheel_roll_deg_at_time(obj, t)
            for wx in (-sx * 0.46, sx * 0.46):
                for wz in (sz * 0.30, -sz * 0.30):
                    _add_prim_to_layer(0, "cylinder", T_veh @ make_transform([wx, 0.36, wz], [-wheel_roll_deg, 0, 90], 1.0), 0.70, 0.24, 0.70, wheel_rgb)
            continue

        if otype == "group" or isinstance(obj.get("parts"), list):
            obj_t = evaluate_object_at_time(obj, t, char_map, ik_map, all_char_ids=all_char_ids_su, obj_map=obj_map_su, scene=scene)
            if not obj_t.get("visible", True):
                continue
            pos = obj_t.get("pos", [0.0, 0.0, 0.0])
            rot = obj_t.get("rot", [0.0, 0.0, 0.0])
            r_scale = float(obj_t.get("render_scale", 1.0))
            if isinstance(obj_t.get("_M_world"), np.ndarray):
                T_parent = obj_t["_M_world"] @ np.diag([r_scale, r_scale, r_scale, 1.0])
                local_sz = obj_t.get("_local_size") or obj.get("size") or [1.0, 1.0, 1.0]
            else:
                T_parent = make_transform(pos, rot, r_scale)
                local_sz = obj_t.get("size", [1.0, 1.0, 1.0])
            reference = obj_t.get("group_base_size")
            if isinstance(reference, (list, tuple)) and len(reference) == 3:
                axes_scale = [float(local_sz[i]) / max(0.0001, float(reference[i])) for i in range(3)]
                T_parent = T_parent @ np.diag(axes_scale + [1.0])
            is_fg_gun = has_focus_shift and (oid == "rifle_boris")
            tgt_layer = 2 if is_fg_gun else 0
            base_hex = obj_t.get("color", get_su_minimal_identity_color(oid, role or "prop"))
            base_rgb = hex_to_rgb(base_hex)
            wheel_roll_deg = compute_wheel_roll_deg_at_time(obj, t)
            for part in obj_t.get("parts") or []:
                if part.get("visible") is False or part.get("hidden") is True:
                    continue
                p_shape = part.get("shape") or part.get("type") or "box"
                p_pos = part.get("pos", [0.0, 0.0, 0.0])
                p_rot = list(part.get("rot", [0.0, 0.0, 0.0]))
                p_id = str(part.get("id") or "").lower()
                if abs(wheel_roll_deg) > 1e-3 and (p_id.startswith("wheel_") or part.get("is_wheel") is True):
                    p_rot = [float(p_rot[0]) - wheel_roll_deg, float(p_rot[1]), float(p_rot[2])]
                p_size = part.get("size", [0.2, 0.2, 0.2])
                p_rgb = hex_to_rgb(part.get("color", base_hex)) if part.get("color") else base_rgb
                T_world = T_parent @ make_transform(p_pos, p_rot, 1.0)
                is_barrel = (role == "gun" or oid.startswith("rifle_")) and str(part.get("id", "")).endswith("barrel")
                _add_prim_to_layer(
                    tgt_layer, p_shape, T_world, float(p_size[0]), float(p_size[1]), float(p_size[2]), p_rgb, is_gun_barrel=is_barrel
                )
            if obj.get("procedural_forest") == "taiga_aerial_sea":
                for tx, ty, tz, tr, th, tc_rgb in get_taiga_aerial_proxy_subset(vis_tree_cap_su):
                    T_cone = T_parent @ make_transform([tx, ty, tz], [0.0, 0.0, 0.0], 1.0)
                    for lf in make_cone_faces_local(tr * 2.0, th, tr * 2.0, segments=6):
                        layer_polys[0].append((transform_points(T_cone, lf), tc_rgb, True))
            continue

        if otype in ("box", "wall", "plane", "sphere", "cone", "prism", "wedge", "cylinder"):
            obj_t = evaluate_object_at_time(obj, t, char_map, ik_map, all_char_ids=all_char_ids_su, obj_map=obj_map_su, scene=scene)
            if not obj_t.get("visible", True):
                continue
            pos = obj_t.get("pos", [0.0, 0.0, 0.0])
            rot = obj_t.get("rot", [0.0, 0.0, 0.0])
            size = obj_t.get("size", [0.5, 0.5, 0.5])
            rgb = hex_to_rgb(obj_t.get("color", get_su_minimal_identity_color(oid, role or otype)))
            if isinstance(obj_t.get("_M_world"), np.ndarray):
                loc_sz = obj_t.get("_local_size") or obj.get("size") or size
                T = obj_t["_M_world"]
                sx_r, sy_r, sz_r = float(loc_sz[0]), float(loc_sz[1]), float(loc_sz[2])
            else:
                T = make_transform(pos, rot, 1.0)
                sx_r, sy_r, sz_r = float(size[0]), float(size[1]), float(size[2])
            is_road = oid.startswith("env_road_") or role == "road"
            if is_road:
                for lf in make_box_faces_local(sx_r, max(0.015, sy_r), sz_r):
                    w_poly = transform_points(T, lf)
                    for sub_r in _subdivide_polygon_world(w_poly, max_edge=4.0):
                        layer_polys[0].append((sub_r, rgb, False))
            else:
                _add_prim_to_layer(0, otype, T, sx_r, sy_r, sz_r, rgb)
            continue

        if otype == "character":
            char_t = char_map.get(oid) if oid in char_map else interpolate_character_at_time(obj, t, scene=scene)
            if not char_t.get("visible", True):
                continue
            is_boris = has_focus_shift and (oid == "char_boris")
            tgt_layer = 1 if is_boris else 0
            c_hex = char_t.get("color", get_su_minimal_identity_color(oid, "character", index_hint=idx_obj))
            char_rgb = hex_to_rgb(c_hex)
            line_rgb = hex_to_rgb(SU_MINIMAL_CONFIG["head_line_color"])

            ik_local = _solve_mannequin_ik_base(dict(char_t, pos=[0, 0, 0], rot=[0, 0, 0], scale=[1, 1, 1]))
            J = ik_local["joints"]
            A = ik_local["axes"]
            sc_val = A["scale"]
            sx = A.get("sx", sc_val)
            sy = A.get("sy", sc_val)
            sz = A.get("sz", sc_val)
            limb_r = math.sqrt(sx * sz)

            local_polys: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []
            is_stat = bool(char_t.get("is_stationary") or char_t.get("body_style") == "pawn")

            if is_stat:
                # 静止极简球头人偶：1 个圆柱躯干 + 1 个球头 + 1 条头前短线（共 3 部件）
                p_bot = np.array([float(J["pelvis"][0]), 0.04 * sy, float(J["pelvis"][2])], dtype=np.float64)
                p_top = np.array([float(J["neck"][0]), float(J["neck"][1]) + 0.02 * sy, float(J["neck"][2])], dtype=np.float64)
                for f in make_limb_prism_between(p_bot, p_top, 0.19 * limb_r, 0.17 * limb_r, sides=10):
                    local_polys.append((f, char_rgb))
                head_c = J["head_center"]
                for f in make_sphere_faces_world(head_c, 0.125 * sy, lat_steps=6, lon_steps=12):
                    local_polys.append((f, char_rgb))
            else:
                # 行走/持枪极简球头人偶：1 球头 + 1 躯干柱体 + 4 简单四肢 + 1 头前朝向短线（共 7 部件，无五官/帽子/关节球/脚掌细节）
                # (1) 躯干 (pelvis -> neck 单一体块)
                p_torso_bot = J["pelvis"] - A["spine_dir"] * (0.06 * sy)
                p_torso_top = J["neck"] + A["spine_dir"] * (0.02 * sy)
                for f in make_oriented_box_between(p_torso_bot, p_torso_top, A.get("side_chest", A["side_L"]), 0.27 * sx, 0.16 * sz):
                    local_polys.append((f, char_rgb))
                # (2) 球头 (无五官、无帽子、无眼罩)
                head_c = J["head_center"]
                for f in make_sphere_faces_world(head_c, 0.120 * sy, lat_steps=6, lon_steps=12):
                    local_polys.append((f, char_rgb))
                # (3-6) 4 条简单四肢（左右臂、左右腿，随步态/持枪摆动，不加 14 个装饰关节球与脚尖）
                simple_limbs = [
                    ("shoulder_L", "elbow_L", "wrist_L", 0.044 * limb_r),
                    ("shoulder_R", "elbow_R", "wrist_R", 0.044 * limb_r),
                    ("hip_L", "knee_L", "ankle_L", 0.058 * limb_r),
                    ("hip_R", "knee_R", "ankle_R", 0.058 * limb_r),
                ]
                for j_start, j_mid, j_end, r_l in simple_limbs:
                    for f in make_limb_prism_between(J[j_start], J[j_mid], r_l, r_l * 0.90, sides=8):
                        local_polys.append((f, char_rgb))
                    for f in make_limb_prism_between(J[j_mid], J[j_end], r_l * 0.90, r_l * 0.80, sides=8):
                        local_polys.append((f, char_rgb))

            # (7) 头部朝向短线 (分段生成避免画家算法被球头侧面遮挡，清晰表示头的朝向)
            h_fwd = A.get("look_dir", np.array([0.0, 0.0, 1.0], dtype=np.float64))
            h_left = A.get("head_left", A.get("side_chest", A["side_L"]))
            for seg_i in range(4):
                r0_s = (0.122 + seg_i * 0.062) * sy
                r1_s = (0.122 + (seg_i + 1) * 0.062) * sy
                ls_i = J["head_center"] + h_fwd * r0_s
                le_i = J["head_center"] + h_fwd * r1_s
                for f in make_oriented_box_between(ls_i, le_i, h_left, 0.024 * sx, 0.024 * sy):
                    local_polys.append((f, line_rgb))

            sx_w, sy_w, sz_w = _get_char_scale_xyz(char_t)
            yaw = math.radians(float(char_t.get("rot", [0, 0, 0])[1]))
            c_y, s_y = math.cos(yaw), math.sin(yaw)
            affine = np.array([[c_y, 0.0, s_y], [0.0, 1.0, 0.0], [-s_y, 0.0, c_y]]) @ np.diag([sx_w, sy_w, sz_w])
            origin = np.asarray(char_t.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
            for f_loc, f_col in local_polys:
                layer_polys[tgt_layer].append((f_loc @ affine.T + origin, f_col, True))

    def _rasterize_polys_onto_image(
        polys_list: List[Tuple[np.ndarray, Tuple[int, int, int], bool]],
        base_img: Optional[Image.Image] = None,
    ) -> Image.Image:
        if base_img is not None:
            canvas = base_img
        else:
            canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        d_ctx = ImageDraw.Draw(canvas)
        items: List[Tuple[float, List[Tuple[float, float]], Tuple[int, int, int, int]]] = []
        for pts_w, rgb_c, do_shade in polys_list:
            pts_c = cam.world_to_cam(pts_w)
            if np.all(pts_c[:, 2] < cam.near):
                continue
            clipped = clip_polygon_near(pts_c, near=cam.near)
            if len(clipped) < 3:
                continue
            depth = float(np.mean(clipped[:, 2]))
            z = np.maximum(clipped[:, 2], cam.near)
            px = (cam.width * 0.5) + (clipped[:, 0] / z) * cam.focal
            py = (cam.height * 0.5) - (clipped[:, 1] / z) * cam.focal
            col = shade_polygon(rgb_c, pts_w) if do_shade else rgb_c
            items.append((depth, [(float(a), float(b)) for a, b in zip(px, py)], (col[0], col[1], col[2], 255)))
        items.sort(key=lambda x: x[0], reverse=True)
        for _, p2d, rgba in items:
            d_ctx.polygon(p2d, fill=rgba, outline=rgba)
        return canvas

    bg_base = Image.new("RGBA", (width, height), (sky_rgb[0], sky_rgb[1], sky_rgb[2], 255))
    if not has_focus_shift:
        all_p = layer_polys[0] + layer_polys[1] + layer_polys[2]
        out_rgba = _rasterize_polys_onto_image(all_p, base_img=bg_base)
        return out_rgba.convert("RGB")

    # 10-14s 推进与移焦：焦点从鲍里斯脸部 (layer 1) 平滑移至枪口 (layer 2)，后景 (layer 0) 始终柔和淡化
    u_focus = 0.0
    if t > 10.0:
        x_u = max(0.0, min(1.0, (t - 10.0) / 4.0))
        u_focus = x_u * x_u * (3.0 - 2.0 * x_u)

    img_bg = _rasterize_polys_onto_image(layer_polys[0], base_img=bg_base)
    blur_bg = 0.8 + 1.8 * u_focus
    if blur_bg > 0.15:
        img_bg = img_bg.filter(ImageFilter.GaussianBlur(radius=blur_bg))

    img_mid = _rasterize_polys_onto_image(layer_polys[1], base_img=None)
    blur_mid = 2.4 * u_focus
    if blur_mid > 0.15:
        img_mid = img_mid.filter(ImageFilter.GaussianBlur(radius=blur_mid))

    img_fg = _rasterize_polys_onto_image(layer_polys[2], base_img=None)
    blur_fg = 1.3 * (1.0 - u_focus)
    if blur_fg > 0.15:
        img_fg = img_fg.filter(ImageFilter.GaussianBlur(radius=blur_fg))

    comp = Image.alpha_composite(img_bg, img_mid)
    comp = Image.alpha_composite(comp, img_fg)
    return comp.convert("RGB")


def render_single_frame(scene: Dict[str, Any], t: float, width: int = 960, height: int = 540) -> Image.Image:
    mode = scene.get("render_mode", "su_minimal")
    if mode == "su_minimal" or scene.get("visual_style") == "su_minimal":
        return render_su_minimal_frame(scene, t, width=width, height=height)
    if mode == "abstract_block":
        return render_abstract_block_frame(scene, t, width=width, height=height)
    cam_state = evaluate_camera_state(scene, t)
    cpos = cam_state["pos"]
    cam = RenderCamera(cpos, cam_state["target"], cam_state["fov"], width, height, roll_deg=cam_state.get("roll_deg", 0.0))

    # 场景环境（天空/地面/雾）：与 Three.js applySceneEnvironment() 对齐；无 environment 时为默认深色网格舞台
    env = scene.get("environment") if mode != "pose_only" else None
    env = env if isinstance(env, dict) and env else None
    fog_rgb = hex_to_rgb(env.get("fog_color") or env.get("sky", "#CDD5DF")) if env else (0, 0, 0)
    fog_near = float(env.get("fog_near", 10.0)) if env else 0.0
    fog_far = float(env.get("fog_far", 45.0)) if env else 1.0
    fog_max = float(env.get("fog_max", 0.88)) if env else 0.0
    fog_on = bool(env) and env.get("fog", True) is not False

    def apply_fog(rgb: Tuple[int, int, int], depth: float) -> Tuple[int, int, int]:
        if not fog_on:
            return rgb
        f = max(0.0, min(1.0, (depth - fog_near) / max(1e-3, fog_far - fog_near))) * fog_max
        return (int(rgb[0] + (fog_rgb[0] - rgb[0]) * f),
                int(rgb[1] + (fog_rgb[1] - rgb[1]) * f),
                int(rgb[2] + (fog_rgb[2] - rgb[2]) * f))

    # 与 Three.js 视口底色 #1B1E23 = (27, 30, 35) 严格对齐
    bg_color = (8, 8, 10) if mode == "pose_only" else (27, 30, 35)
    if env:
        bg_color = hex_to_rgb(env.get("sky", "#CDD5DF"))
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    def _project_world_poly(pts_w: np.ndarray, base_rgb: Tuple[int, int, int], shade: bool):
        pts_c = cam.world_to_cam(pts_w)
        if np.all(pts_c[:, 2] < cam.near):
            return None
        clipped = clip_polygon_near(pts_c, near=cam.near)
        if len(clipped) < 3:
            return None
        depth = float(np.mean(clipped[:, 2]))
        z = np.maximum(clipped[:, 2], cam.near)
        px = (cam.width * 0.5) + (clipped[:, 0] / z) * cam.focal
        py = (cam.height * 0.5) - (clipped[:, 1] / z) * cam.focal
        col = shade_polygon(base_rgb, pts_w) if shade else base_rgb
        return (depth, [(float(a), float(b)) for a, b in zip(px, py)], apply_fog(col, depth))

    def draw_world_polys_far_to_near(polys: List[Tuple[np.ndarray, Tuple[int, int, int]]], shade: bool):
        items = []
        for pts_w, base_rgb in polys:
            proj = _project_world_poly(pts_w, base_rgb, shade=shade)
            if proj is not None:
                items.append(proj)
        items.sort(key=lambda x: x[0], reverse=True)
        for _, poly_2d, col in items:
            draw.polygon(poly_2d, fill=col, outline=col)

    # 1a. 环境地面：以摄影机为中心铺 3m 方格（远→近绘制，叠加距离雾），保证始终在物体之下
    if env and env.get("show_ground", True) is not False:
        g_rgb = hex_to_rgb(env.get("ground", "#D2D9E2"))
        tile, reach = 3.0, 60.0
        cx0 = math.floor(float(cpos[0]) / tile) * tile
        cz0 = math.floor(float(cpos[2]) / tile) * tile
        n_t = int(reach / tile)
        ground_polys = []
        for ix in range(-n_t, n_t):
            for iz in range(-n_t, n_t):
                x0, z0 = cx0 + ix * tile, cz0 + iz * tile
                ground_polys.append((np.array([[x0, 0.0, z0], [x0 + tile, 0.0, z0],
                                               [x0 + tile, 0.0, z0 + tile], [x0, 0.0, z0 + tile]], dtype=np.float64), g_rgb))
        draw_world_polys_far_to_near(ground_polys, shade=False)

    # 1b. 绘制与 Three.js GridHelper(24, 24, 0x383842, 0x2b2b33) 严格一致的 24m x 24m 地面网格
    if mode != "pose_only" and not (env and env.get("show_grid", False) is False):
        for g in range(-12, 13):
            col = (56, 56, 66) if g == 0 else (43, 43, 51)
            p_x0 = cam.project_point(np.array([float(g), 0.001, -12.0]))
            p_x1 = cam.project_point(np.array([float(g), 0.001, 12.0]))
            if p_x0 and p_x1:
                draw.line([(p_x0[0], p_x0[1]), (p_x1[0], p_x1[1])], fill=col, width=1)
            p_z0 = cam.project_point(np.array([-12.0, 0.001, float(g)]))
            p_z1 = cam.project_point(np.array([12.0, 0.001, float(g)]))
            if p_z0 and p_z1:
                draw.line([(p_z0[0], p_z0[1]), (p_z1[0], p_z1[1])], fill=col, width=1)

    # 2. 收集 3D 多边形 (几何体 + SketchUp 灰白体块与黑色细结构边线 + 魔法人形师素体木偶白模)
    polygons_world: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []
    unlit_polys: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []
    ground_layer_polys: List[Tuple[float, np.ndarray, Tuple[int, int, int]]] = []  # (elevation_y, pts_w, rgb) 按高程分层先绘
    decal_polys: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []  # 贴地薄片（地面阴影/水洼），先于立体物体绘制
    edge_segments_world: List[Tuple[np.ndarray, np.ndarray, float]] = []  # (pt0_w, pt1_w, ref_depth_bias)
    characters_ik: List[Dict[str, Any]] = []
    char_map, ik_map, _reach_warnings = evaluate_scene_characters_at_time(scene, t)
    hide_edges_flag = True if FORCE_HIDE_EDGES else bool(scene.get("hide_edges"))

    def _normalize_sketchup_gray_rgb(rgb: Tuple[int, int, int], role_or_type: str = "prop") -> Tuple[int, int, int]:
        """将非人物场景物体统一收敛为 SketchUp 风格无贴图灰白体块色，保留轻微明度差区分墙/地/道具"""
        lum = int(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
        max_c, min_c = max(rgb), min(rgb)
        if max_c - min_c <= 18:
            # 已经是灰白色系，直接保留其精细明度与冷暖微差
            return rgb
        # 将旧版高饱和颜色按类别映射为灰白明度阶梯
        rt = (role_or_type or "").lower()
        if "floor" in rt or "ground" in rt:
            v = max(175, min(205, lum))
            return (v - 2, v, v + 2)
        if "wall" in rt or "roof" in rt:
            v = max(210, min(230, lum + 45))
            return (v, v + 1, v + 2)
        v = max(205, min(238, lum + 55))
        return (v, v + 1, v + 2)

    def _is_flat_ground_slab(role_s: str, shape_s: str, T_mat: np.ndarray, sx_p: float, sy_p: float, sz_p: float) -> bool:
        if (shape_s or "box").lower() not in ("box", "wall", "plane"):
            return False
        if (role_s or "").lower() not in ("ground", "road", "sidewalk", "floor"):
            return False
        return (
            sy_p <= 0.25
            and max(sx_p, sz_p) >= 2.0
            and abs(float(T_mat[1, 1])) >= 0.995
            and (float(T_mat[1, 3]) + 0.5 * sy_p) <= 0.22
        )

    def _append_primitive_faces_and_edges(
        shape: str,
        T_mat: np.ndarray,
        sx_p: float,
        sy_p: float,
        sz_p: float,
        rgb_p: Tuple[int, int, int],
        is_decal: bool = False,
        is_unlit: bool = False,
        is_ground_layer: bool = False,
        skip_edges: bool = False,
    ) -> None:
        shp = (shape or "box").lower()
        if shp in ("box", "wall", "plane"):
            lfaces = make_box_faces_local(sx_p, max(0.02, sy_p) if shp == "plane" else sy_p, sz_p)
        elif shp == "sphere":
            raw_sfaces = make_sphere_faces_world(np.array([0.0, 0.0, 0.0], dtype=float), 0.5, lat_steps=8, lon_steps=14)
            scale_vec = np.array([sx_p, sy_p, sz_p], dtype=float)
            lfaces = [f * scale_vec for f in raw_sfaces]
        elif shp == "cone":
            lfaces = make_cone_faces_local(sx_p, sy_p, sz_p, segments=16)
        elif shp in ("prism", "wedge"):
            lfaces = make_prism_faces_local(sx_p, sy_p, sz_p)
        else:
            lfaces = make_cylinder_faces_local(sx_p, sy_p, sz_p, segments=16 if not is_decal else 28)

        target_list = decal_polys if is_decal else (unlit_polys if is_unlit else polygons_world)
        if is_ground_layer and not is_decal and not is_unlit:
            poly_max_edge = 18.0 if max(sx_p, sz_p) >= 80.0 else 8.0
            edge_max_step = 4.0
        else:
            poly_max_edge = max(1.6, 0.45 * max(sx_p, sy_p, sz_p))
            edge_max_step = max(0.85, 0.35 * max(sx_p, sy_p, sz_p))
        w_faces: List[np.ndarray] = []
        for lf in lfaces:
            w_poly = transform_points(T_mat, lf)
            w_faces.append(w_poly)
            for sub_p in _subdivide_polygon_world(w_poly, max_edge=poly_max_edge):
                if is_ground_layer and not is_decal and not is_unlit:
                    ground_layer_polys.append((float(np.mean(sub_p[:, 1])), sub_p, rgb_p))
                else:
                    target_list.append((sub_p, rgb_p))

        if is_decal or is_unlit or skip_edges or hide_edges_flag:
            return

        def _add_subdivided_edge(pt_a: np.ndarray, pt_b: np.ndarray, bias: float = -0.010, max_step: float = edge_max_step) -> None:
            vec = pt_b - pt_a
            dist = float(np.linalg.norm(vec))
            if dist < 1e-6:
                return
            steps = max(1, int(math.ceil(dist / max_step)))
            if steps == 1:
                edge_segments_world.append((pt_a, pt_b, bias))
                return
            for idx_seg in range(steps):
                t0 = idx_seg / steps
                t1 = (idx_seg + 1) / steps
                edge_segments_world.append((pt_a + vec * t0, pt_a + vec * t1, bias))

        # 收集结构转折边线（隐藏三角剖分线、共面内部边与圆柱侧面细分线）
        if shp in ("box", "wall", "prism", "wedge"):
            for wf in w_faces:
                n_pts = len(wf)
                if n_pts < 3:
                    continue
                f_norm = np.cross(wf[1] - wf[0], wf[2] - wf[0])
                f_center = np.mean(wf, axis=0)
                # 朝向摄影机或接近轮廓的面提取外框结构边
                if float(np.dot(f_norm, cam.pos - f_center)) >= -0.05:
                    for i_e in range(n_pts):
                        _add_subdivided_edge(wf[i_e], wf[(i_e + 1) % n_pts], -0.010)
        elif shp in ("cylinder", "cone"):
            # 圆柱/圆锥：仅绘制上下顶底圆环边 + 两侧外轮廓母线，隐藏侧面内部细分线
            side_normals: List[np.ndarray] = []
            side_faces: List[np.ndarray] = []
            for wf in w_faces:
                if len(wf) == 4:
                    fn = _normalize(np.cross(wf[1] - wf[0], wf[2] - wf[0]))
                    side_normals.append(fn)
                    side_faces.append(wf)
                    # 上下圆环端边直接添加弦段
                    edge_segments_world.append((wf[0], wf[1], -0.010))
                    edge_segments_world.append((wf[2], wf[3], -0.010))
                elif len(wf) == 3 and shp == "cone":
                    # 圆锥底面圆环边
                    edge_segments_world.append((wf[0], wf[1], -0.010))
            n_sides = len(side_faces)
            if n_sides >= 6:
                facing = [
                    float(np.dot(side_normals[idx_s], cam.pos - np.mean(side_faces[idx_s], axis=0)))
                    for idx_s in range(n_sides)
                ]
                for idx_s in range(n_sides):
                    nxt_s = (idx_s + 1) % n_sides
                    if (facing[idx_s] >= 0.0) != (facing[nxt_s] >= 0.0):
                        sf = side_faces[idx_s]
                        _add_subdivided_edge(sf[1], sf[2], -0.012)

    is_sketchup_massing = bool(scene.get("visual_style") == "sketchup_massing")
    all_char_ids_sf = {
        str(o.get("id"))
        for o in scene.get("objects", [])
        if isinstance(o, dict) and o.get("type") == "character" and o.get("id")
    }
    plan_mode_sf = str((scene.get("export_plan") or {}).get("priority_mode") or "balanced").lower()
    vis_tree_cap_sf = 300 if plan_mode_sf == "fast" else (1000 if plan_mode_sf == "composition" else 600)

    for obj in scene.get("objects", []):
        if obj.get("is_helper") is True:
            continue
        if plan_mode_sf == "fast" and obj.get("is_decorative_rubble") is True and not obj.get("protected") and not obj.get("locked"):
            continue
        otype = obj.get("type", "box")
        if otype == "uaz469":
            if mode == "pose_only":
                continue
            obj_t = evaluate_object_at_time(obj, t, char_map, ik_map, all_char_ids=all_char_ids_sf, scene=scene)
            if not obj_t.get("visible", True):
                continue
            pos = obj_t.get("pos", [0.0, 1.01, 0.0])
            rot = obj_t.get("rot", [0.0, 0.0, 0.0])
            size = obj_t.get("size", [1.78, 2.02, 4.02])
            rgb = hex_to_rgb(obj_t.get("color", "#D5D9DF" if is_sketchup_massing else "#4F5D3B"))
            polygons_world.extend(
                make_uaz469_polygons_world(
                    pos,
                    rot,
                    size,
                    rgb,
                    sketchup_style=is_sketchup_massing,
                    edge_out=edge_segments_world if (is_sketchup_massing and not hide_edges_flag) else None,
                    cam_pos=cam.pos,
                    door_FR_deg=float(obj_t.get("door_FR_deg") or 0.0),
                    door_FL_deg=float(obj_t.get("door_FL_deg") or 0.0),
                )
            )
        elif otype == "group" or isinstance(obj.get("parts"), list):
            if mode == "pose_only":
                continue
            obj_t = evaluate_object_at_time(obj, t, char_map, ik_map, all_char_ids=all_char_ids_sf, scene=scene)
            if not obj_t.get("visible", True):
                continue
            pos = obj_t.get("pos", [0.0, 0.0, 0.0])
            rot = obj_t.get("rot", [0.0, 0.0, 0.0])
            r_scale = float(obj_t.get("render_scale", 1.0))
            T_parent = make_transform(pos, rot, r_scale)
            reference = obj_t.get("group_base_size")
            if isinstance(reference, (list, tuple)) and len(reference) == 3:
                size_now = obj_t.get("size", reference)
                axes_scale = [float(size_now[i]) / max(0.0001, float(reference[i])) for i in range(3)]
                T_parent = T_parent @ np.diag(axes_scale + [1.0])
            role_tag = str(obj_t.get("arch_role") or otype)
            default_hex = obj_t.get("color", "#E2E3E5")
            wheel_roll_deg = compute_wheel_roll_deg_at_time(obj, t)
            for part in obj_t.get("parts") or []:
                if part.get("visible") is False or part.get("hidden") is True:
                    continue
                p_shape = part.get("shape") or part.get("type") or "box"
                p_pos = part.get("pos", [0.0, 0.0, 0.0])
                p_rot = list(part.get("rot", [0.0, 0.0, 0.0]))
                p_id = str(part.get("id") or "").lower()
                if abs(wheel_roll_deg) > 1e-3 and (p_id.startswith("wheel_") or part.get("is_wheel") is True):
                    p_rot = [float(p_rot[0]) - wheel_roll_deg, float(p_rot[1]), float(p_rot[2])]
                p_size = part.get("size", [0.2, 0.2, 0.2])
                raw_part_rgb = hex_to_rgb(part.get("color", default_hex))
                p_rgb = raw_part_rgb if (part.get("keep_color") or obj_t.get("keep_color")) else _normalize_sketchup_gray_rgb(raw_part_rgb, role_tag)
                T_part = make_transform(p_pos, p_rot, 1.0)
                T_world = T_parent @ T_part
                is_floor_decal = bool(part.get("ground_decal") or obj_t.get("ground_decal") or role_tag == "floor")
                is_part_unlit = bool(part.get("unlit") or obj_t.get("unlit") or role_tag == "flame_volume")
                is_part_ground_layer = (not is_floor_decal) and (not is_part_unlit) and _is_flat_ground_slab(
                    role_tag, p_shape, T_world, float(p_size[0]), float(p_size[1]), float(p_size[2])
                )
                _append_primitive_faces_and_edges(
                    p_shape, T_world, float(p_size[0]), float(p_size[1]), float(p_size[2]), p_rgb,
                    is_floor_decal, is_part_unlit,
                    is_ground_layer=is_part_ground_layer,
                    skip_edges=is_part_ground_layer and (role_tag.lower() == "ground"),
                )
            if obj.get("procedural_forest") == "taiga_aerial_sea":
                for tx, ty, tz, tr, th, tc_rgb in get_taiga_aerial_proxy_subset(vis_tree_cap_sf):
                    T_cone = T_parent @ make_transform([tx, ty, tz], [0.0, 0.0, 0.0], 1.0)
                    for lf in make_cone_faces_local(tr * 2.0, th, tr * 2.0, segments=6):
                        polygons_world.append((transform_points(T_cone, lf), tc_rgb))
        elif otype in ("box", "wall", "plane", "sphere", "cone", "prism", "wedge", "cylinder"):
            if mode == "pose_only":
                continue
            obj_t = evaluate_object_at_time(obj, t, char_map, ik_map, all_char_ids=all_char_ids_sf, scene=scene)
            if not obj_t.get("visible", True):
                continue
            pos = obj_t.get("pos", [0.0, 0.0, 0.0])
            rot = obj_t.get("rot", [0.0, 0.0, 0.0])
            size = obj_t.get("size", [0.5, 0.5, 0.5])
            role_tag = str(obj_t.get("arch_role") or otype)
            raw_rgb = hex_to_rgb(obj_t.get("color", "#E2E3E5"))
            rgb = raw_rgb if (obj_t.get("keep_color") or role_tag == "flame_volume") else _normalize_sketchup_gray_rgb(raw_rgb, role_tag)
            T = make_transform(pos, rot, 1.0)
            is_prim_decal = bool(obj_t.get("ground_decal"))
            is_prim_unlit = bool(obj_t.get("unlit") or role_tag == "flame_volume")
            is_prim_ground_layer = (not is_prim_decal) and (not is_prim_unlit) and _is_flat_ground_slab(
                role_tag, otype, T, float(size[0]), float(size[1]), float(size[2])
            )
            _append_primitive_faces_and_edges(
                otype, T, float(size[0]), float(size[1]), float(size[2]), rgb,
                is_prim_decal,
                is_prim_unlit,
                is_ground_layer=is_prim_ground_layer,
                skip_edges=is_prim_ground_layer and (role_tag.lower() == "ground"),
            )

        elif otype == "character":
            cid = obj.get("id", "")
            char_t = char_map.get(cid) if cid in char_map else interpolate_character_at_time(obj, t, scene=scene)
            if not char_t.get("visible", True):
                continue
            ik_res = ik_map.get(cid) or solve_mannequin_ik(char_t)
            characters_ik.append(ik_res)
            if mode in ("clay_and_pose", "clay_only"):
                # Build local geometry once, then stretch it on the character's
                # local axes. Scaling bone lengths by Y alone distorts X/Z edits.
                polygon_start = len(polygons_world)
                ik_res = _solve_mannequin_ik_base(dict(char_t, pos=[0, 0, 0], rot=[0, 0, 0], scale=[1, 1, 1]))
                J = ik_res["joints"]
                A = ik_res["axes"]
                sc = A["scale"]
                sx = A.get("sx", sc)
                sy = A.get("sy", sc)
                sz = A.get("sz", sc)
                limb_rad_scale = math.sqrt(sx * sz)
                base_c = hex_to_rgb(char_t.get("color", "#e2e8f0"))
                clay_main = base_c
                body_style = char_t.get("body_style") or scene.get("global_body_style") or "mannequin"

                if body_style == "pawn":
                    # 球头人偶：圆柱体身体 + 头顶球体形态（支持坐姿高度裁切、贴地深蹲与躯干前倾/侧倾/倒悬，与 Three.js pawnGroup 严格对齐）
                    spine_dir_y = float(A["spine_dir"][1])
                    if spine_dir_y > 0.5:
                        pelvis_y = float(J["pelvis"][1])
                        ankle_min_y = min(float(J["ankle_L"][1]), float(J["ankle_R"][1])) - 0.03 * sy
                        pose_nm = str(char_t.get("pose_name", "")).lower()
                        is_ground_squat = bool(char_t.get("ground_squat")) or pose_nm.startswith("squat") or pose_nm.startswith("crouch")
                        is_seated_pawn = (not is_ground_squat) and ((0.22 * sy < pelvis_y < 0.62 * sy) or pose_nm.startswith("sitting"))
                        if is_seated_pawn:
                            min_y = max(0.0, pelvis_y - 0.10 * sy)
                        else:
                            min_y = max(0.0, ankle_min_y)
                        top_y = float(J["neck"][1]) + 0.02 * sy
                        p_bot = np.array([float(J["pelvis"][0]), min_y, float(J["pelvis"][2])], dtype=np.float64)
                        p_top = np.array([float(J["neck"][0]), max(min_y + 0.36 * sy, top_y), float(J["neck"][2])], dtype=np.float64)
                        cyl_axis = _normalize(p_top - p_bot)
                        if float(np.linalg.norm(cyl_axis)) < 1e-6:
                            cyl_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                        head_c = p_top + cyl_axis * (0.175 * sy)
                    else:
                        feet_mid = 0.5 * (np.array(J["ankle_L"], dtype=np.float64) + np.array(J["ankle_R"], dtype=np.float64))
                        p_bot = feet_mid
                        p_top = np.array(J["neck"], dtype=np.float64)
                        if float(np.linalg.norm(p_top - p_bot)) < 0.45 * sy:
                            p_bot = p_top - A["spine_dir"] * (0.45 * sy)
                        head_c = p_top + A["spine_dir"] * (0.175 * sy)
                    for f in make_limb_prism_between(p_bot, p_top, 0.21 * limb_rad_scale, 0.18 * limb_rad_scale, sides=16):
                        polygons_world.append((f, clay_main))
                    for f in make_sphere_faces_world(head_c, 0.165 * sy, lat_steps=8, lon_steps=14):
                        polygons_world.append((f, clay_main))
                    if obj.get("pawn_visor"):
                        # 头部正前方深色护目镜条（朝向 = head_target 水平方向，与 Three.js pawnVisor 对齐）
                        fwd = np.array(J["head_target"], dtype=np.float64) - head_c
                        fwd[1] = 0.0
                        if np.linalg.norm(fwd) < 1e-6:
                            ry = math.radians(float(char_t.get("rot", [0, 0, 0])[1]))
                            fwd = np.array([math.sin(ry), 0.0, math.cos(ry)])
                        fwd = fwd / np.linalg.norm(fwd)
                        up = np.array([0.0, 1.0, 0.0])
                        side = np.cross(up, fwd)
                        vc = head_c + fwd * (0.150 * sx) + up * (0.015 * sy)
                        vh = 0.055 * sy
                        visor_rgb = (int(clay_main[0] * 0.32), int(clay_main[1] * 0.32), int(clay_main[2] * 0.32))
                        for f in make_oriented_box_between(vc - up * vh * 0.5, vc + up * vh * 0.5, side, 0.17 * sx, 0.07 * sz):
                            polygons_world.append((f, visor_rgb))
                    else:
                        eye_rgb = (max(18, int(clay_main[0] * 0.28)), max(18, int(clay_main[1] * 0.28)), max(18, int(clay_main[2] * 0.28)))
                        nose_rgb = (max(30, int(clay_main[0] * 0.82)), max(30, int(clay_main[1] * 0.82)), max(30, int(clay_main[2] * 0.82)))
                        h_fwd = A.get("look_dir", np.array([0.0, 0.0, 1.0], dtype=np.float64))
                        h_left = A.get("head_left", A.get("side_chest", A["side_L"]))
                        h_up = A.get("head_up", A["spine_dir"])
                        eye_L_c = head_c + h_fwd * (0.152 * sz) + h_left * (0.048 * sx)
                        eye_R_c = head_c + h_fwd * (0.152 * sz) - h_left * (0.048 * sx)
                        nose_c = head_c + h_fwd * (0.170 * sz) - h_up * (0.022 * sy)
                        for f in make_sphere_faces_world(eye_L_c, 0.018 * limb_rad_scale, lat_steps=4, lon_steps=8):
                            polygons_world.append((f, eye_rgb))
                        for f in make_sphere_faces_world(eye_R_c, 0.018 * limb_rad_scale, lat_steps=4, lon_steps=8):
                            polygons_world.append((f, eye_rgb))
                        for f in make_sphere_faces_world(nose_c, 0.020 * limb_rad_scale, lat_steps=4, lon_steps=8):
                            polygons_world.append((f, nose_rgb))
                else:
                    # 极简纯色圆头关节人偶（与 Three.js clayGroup 100% 几何对齐）
                    off_L = A.get("shoulder_L_offset", np.zeros(3, dtype=np.float64))
                    off_R = A.get("shoulder_R_offset", np.zeros(3, dtype=np.float64))
                    sh_mid_shift = (
                        A["spine_dir"] * (0.35 * 0.5 * float(off_L[1] + off_R[1]) * sy)
                        + np.cross(A.get("side_chest", A["side_L"]), A["spine_dir"]) * (0.35 * 0.5 * float(off_L[2] + off_R[2]) * sz)
                    )
                    mid_waist = J["pelvis"] + (J["neck"] - J["pelvis"]) * 0.35 + sh_mid_shift * 0.5
                    chest_top = J["neck"] + sh_mid_shift
                    for f in make_oriented_box_between(mid_waist, chest_top, A.get("side_chest", A["side_L"]), 0.28 * sx, 0.165 * sz):
                        polygons_world.append((f, clay_main))
                    pelvis_bot = J["pelvis"] - A["spine_dir"] * (0.07 * sy)
                    pelvis_top = J["pelvis"] + A["spine_dir"] * (0.07 * sy)
                    for f in make_oriented_box_between(pelvis_bot, pelvis_top, A["side_L"], 0.26 * sx, 0.16 * sz):
                        polygons_world.append((f, clay_main))
                    for f in make_sphere_faces_world(J["head_center"], 0.115 * sy, lat_steps=8, lon_steps=14):
                        polygons_world.append((f, clay_main))

                    # 头部极简五官：双眼 + 小鼻尖（跟随头部 3D 朝向旋转）
                    eye_rgb = (max(18, int(clay_main[0] * 0.28)), max(18, int(clay_main[1] * 0.28)), max(18, int(clay_main[2] * 0.28)))
                    nose_rgb = (max(30, int(clay_main[0] * 0.82)), max(30, int(clay_main[1] * 0.82)), max(30, int(clay_main[2] * 0.82)))
                    h_fwd = A.get("look_dir", np.array([0.0, 0.0, 1.0], dtype=np.float64))
                    h_left = A.get("head_left", A.get("side_chest", A["side_L"]))
                    h_up = A.get("head_up", A["spine_dir"])
                    eye_L_c = J["head_center"] + h_fwd * (0.106 * sz) + h_left * (0.036 * sx)
                    eye_R_c = J["head_center"] + h_fwd * (0.106 * sz) - h_left * (0.036 * sx)
                    nose_c = J["head_center"] + h_fwd * (0.120 * sz) - h_up * (0.018 * sy)
                    for f in make_sphere_faces_world(eye_L_c, 0.014 * limb_rad_scale, lat_steps=4, lon_steps=8):
                        polygons_world.append((f, eye_rgb))
                    for f in make_sphere_faces_world(eye_R_c, 0.014 * limb_rad_scale, lat_steps=4, lon_steps=8):
                        polygons_world.append((f, eye_rgb))
                    for f in make_sphere_faces_world(nose_c, 0.015 * limb_rad_scale, lat_steps=4, lon_steps=8):
                        polygons_world.append((f, nose_rgb))

                    # 与 Three.js CylinderGeometry(0.88, 0.76, 1.0, 16) + alignLimbMesh 严格一致（含锁骨连接段）
                    limb_specs = [
                        ("sternum", "shoulder_L", 0.042 * limb_rad_scale),
                        ("sternum", "shoulder_R", 0.042 * limb_rad_scale),
                        ("hip_L", "knee_L", 0.064 * limb_rad_scale),
                        ("knee_L", "ankle_L", 0.054 * limb_rad_scale),
                        ("ankle_L", "toe_L", 0.044 * limb_rad_scale),
                        ("hip_R", "knee_R", 0.064 * limb_rad_scale),
                        ("knee_R", "ankle_R", 0.054 * limb_rad_scale),
                        ("ankle_R", "toe_R", 0.044 * limb_rad_scale),
                        ("shoulder_L", "elbow_L", 0.046 * limb_rad_scale),
                        ("elbow_L", "wrist_L", 0.040 * limb_rad_scale),
                        ("shoulder_R", "elbow_R", 0.046 * limb_rad_scale),
                        ("elbow_R", "wrist_R", 0.040 * limb_rad_scale),
                    ]
                    for j0, j1, r_base in limb_specs:
                        for f in make_limb_prism_between(J[j0], J[j1], r_base * 0.76, r_base * 0.88, sides=12):
                            polygons_world.append((f, clay_main))

                    # 与 Three.js ud.jointMeshes 严格一致的 14 个关节球
                    joint_names = [
                        "pelvis", "neck",
                        "shoulder_L", "elbow_L", "wrist_L",
                        "shoulder_R", "elbow_R", "wrist_R",
                        "hip_L", "knee_L", "ankle_L",
                        "hip_R", "knee_R", "ankle_R",
                    ]
                    for jn in joint_names:
                        if jn in J:
                            jr = (0.056 if (jn == "pelvis" or jn.startswith("hip")) else 0.044) * limb_rad_scale
                            for f in make_sphere_faces_world(J[jn], jr, lat_steps=5, lon_steps=10):
                                polygons_world.append((f, clay_main))

                    garment_kind = str(char_t.get("garment") or "")
                    if garment_kind in ("winter_coat_long", "winter_parka"):
                        coat_rgb = clay_main
                        trim_rgb = (min(255, clay_main[0] + 6), min(255, clay_main[1] + 6), min(255, clay_main[2] + 6))
                        hand_rgb = (min(255, clay_main[0] + 9), min(255, clay_main[1] + 9), min(255, clay_main[2] + 9))
                        boot_rgb = (max(40, clay_main[0] - 20), max(40, clay_main[1] - 19), max(40, clay_main[2] - 18))
                        side_ch = A.get("side_chest", A["side_L"])
                        fwd_ch = _normalize(np.cross(side_ch, A["spine_dir"]))
                        c_state_v = (char_t.get("targets") or {}).get("coat_state") or [0.0, 0.0, 0.0]
                        hem_lag_z = float(c_state_v[0]) if len(c_state_v) >= 1 else 0.0
                        hem_sway_x = float(c_state_v[1]) if len(c_state_v) >= 2 else 0.0
                        grip_R = max(0.0, min(1.0, float(c_state_v[2]) if len(c_state_v) >= 3 else 0.0))

                        # (a) 冬季外套上身与厚围领轮廓
                        coat_waist = J["pelvis"] + (J["neck"] - J["pelvis"]) * 0.16
                        for f in make_oriented_box_between(coat_waist, chest_top + A["spine_dir"] * (0.015 * sy), side_ch, 0.34 * sx, 0.215 * sz):
                            polygons_world.append((f, coat_rgb))
                        col_bot = chest_top - A["spine_dir"] * (0.010 * sy)
                        col_top = chest_top + A["spine_dir"] * (0.052 * sy)
                        for f in make_oriented_box_between(col_bot, col_top, side_ch, 0.245 * sx, 0.195 * sz):
                            polygons_world.append((f, trim_rgb))

                        # (b) 冬季大衣衣摆（长款下垂至膝下并保留衣摆滞后；短款棉服止于胯下不遮挡深蹲膝盖）
                        def _add_frustum_skirt(pt_t: np.ndarray, pt_b: np.ndarray, wt: float, dt: float, wb: float, db: float, col_s: Tuple[int, int, int]) -> None:
                            s_ax = _normalize(A["side_L"])
                            f_ax = _normalize(np.cross(s_ax, np.array([0.0, 1.0, 0.0], dtype=np.float64)))
                            if float(np.linalg.norm(f_ax)) < 1e-6:
                                f_ax = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                            c_top = [
                                pt_t - s_ax * (wt * 0.5) - f_ax * (dt * 0.5),
                                pt_t + s_ax * (wt * 0.5) - f_ax * (dt * 0.5),
                                pt_t + s_ax * (wt * 0.5) + f_ax * (dt * 0.5),
                                pt_t - s_ax * (wt * 0.5) + f_ax * (dt * 0.5),
                            ]
                            c_bot = [
                                pt_b - s_ax * (wb * 0.5) - f_ax * (db * 0.5),
                                pt_b + s_ax * (wb * 0.5) - f_ax * (db * 0.5),
                                pt_b + s_ax * (wb * 0.5) + f_ax * (db * 0.5),
                                pt_b - s_ax * (wb * 0.5) + f_ax * (db * 0.5),
                            ]
                            for k_i in range(4):
                                k_n = (k_i + 1) % 4
                                polygons_world.append((np.array([c_top[k_i], c_top[k_n], c_bot[k_n], c_bot[k_i]], dtype=np.float64), col_s))
                            polygons_world.append((np.array([c_top[0], c_top[3], c_top[2], c_top[1]], dtype=np.float64), col_s))
                            polygons_world.append((np.array([c_bot[0], c_bot[1], c_bot[2], c_bot[3]], dtype=np.float64), col_s))

                        sk_top = J["pelvis"] + np.array([0.0, 0.08 * sy, 0.0], dtype=np.float64)
                        if garment_kind == "winter_coat_long":
                            mid_kn = 0.5 * (J["knee_L"] + J["knee_R"])
                            hem_y = max(0.36 * sy, float(J["pelvis"][1]) - 0.45 * sy)
                            sk_bot = np.array([
                                0.55 * float(J["pelvis"][0]) + 0.45 * float(mid_kn[0]) + hem_sway_x * sx,
                                hem_y,
                                0.50 * float(J["pelvis"][2]) + 0.50 * float(mid_kn[2]) + hem_lag_z * sz,
                            ], dtype=np.float64)
                            kn_span_x = abs(float(J["knee_L"][0]) - float(J["knee_R"][0]))
                            kn_span_z = abs(float(J["knee_L"][2]) - float(J["knee_R"][2]))
                            wb_dyn = max(0.43 * sx, kn_span_x + 0.22 * sx)
                            db_dyn = max(0.30 * sz, kn_span_z + 0.22 * sz)
                            _add_frustum_skirt(sk_top, sk_bot, 0.33 * sx, 0.22 * sz, wb_dyn, db_dyn, coat_rgb)
                        else:
                            sk_bot = J["pelvis"] - A["spine_dir"] * (0.14 * sy) + fwd_ch * (hem_lag_z * 0.4 * sz)
                            _add_frustum_skirt(sk_top, sk_bot, 0.32 * sx, 0.20 * sz, 0.35 * sx, 0.23 * sz, coat_rgb)

                        # (c) 冬季衣袖、袖口与可辨识手掌/拇指/手指抓握动作
                        for s_side in ("L", "R"):
                            j_sh = J[f"shoulder_{s_side}"]
                            j_el = J[f"elbow_{s_side}"]
                            j_wr = J[f"wrist_{s_side}"]
                            for f in make_limb_prism_between(j_sh, j_el, 0.056 * limb_rad_scale, 0.064 * limb_rad_scale, sides=12):
                                polygons_world.append((f, coat_rgb))
                            cuff_a = j_el + (j_wr - j_el) * 0.80
                            cuff_b = j_el + (j_wr - j_el) * 0.94
                            for f in make_limb_prism_between(j_el, cuff_b, 0.050 * limb_rad_scale, 0.055 * limb_rad_scale, sides=12):
                                polygons_world.append((f, coat_rgb))
                            for f in make_limb_prism_between(cuff_a, cuff_b, 0.055 * limb_rad_scale, 0.054 * limb_rad_scale, sides=12):
                                polygons_world.append((f, trim_rgb))

                            f_dir = _normalize(j_wr - j_el)
                            if float(np.linalg.norm(f_dir)) < 1e-6:
                                f_dir = np.array([0.0, -1.0, 0.0], dtype=np.float64)
                            inward = np.array([-1.0 if s_side == "L" else 1.0, 0.0, 0.0], dtype=np.float64)
                            h_side = _normalize(np.cross(f_dir, inward))
                            if float(np.linalg.norm(h_side)) < 1e-6:
                                h_side = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                            palm_n = _normalize(np.cross(h_side, f_dir))
                            palm_end = j_wr + f_dir * (0.056 * sy)
                            for f in make_oriented_box_between(j_wr, palm_end, h_side, 0.050 * sx, 0.026 * sz):
                                polygons_world.append((f, hand_rgb))

                            grip_v = grip_R if s_side == "R" else 0.15
                            # 拇指（稳定对握包拢 / 舒展）
                            th_sign = -1.0 if s_side == "R" else 1.0
                            th_base = j_wr + f_dir * (0.024 * sy) + h_side * (th_sign * 0.024 * sx)
                            th_mid = th_base + f_dir * (0.028 * sy) + h_side * (th_sign * (0.016 - 0.010 * grip_v) * sx) + palm_n * ((0.008 + 0.026 * grip_v) * sz)
                            th_tip = th_mid + f_dir * ((0.024 - 0.010 * grip_v) * sy) - h_side * (th_sign * 0.014 * grip_v * sx) + palm_n * ((0.004 + 0.018 * grip_v) * sz)
                            for f in make_limb_prism_between(th_base, th_mid, 0.010 * limb_rad_scale, 0.011 * limb_rad_scale, sides=8):
                                polygons_world.append((f, hand_rgb))
                            for f in make_limb_prism_between(th_mid, th_tip, 0.0085 * limb_rad_scale, 0.010 * limb_rad_scale, sides=8):
                                polygons_world.append((f, hand_rgb))
                            # 三组并拢手指（松开时依次舒展）
                            for fi, lat_u in enumerate((-0.016, 0.0, 0.016)):
                                g_f = max(0.0, min(1.0, grip_v * 1.15 - (2 - fi) * 0.08 * (1.0 - grip_v)))
                                kn_p = palm_end + h_side * (lat_u * sx)
                                ang1 = 0.22 + 1.05 * g_f
                                ang2 = 0.35 + 1.85 * g_f
                                d1 = _normalize(f_dir * math.cos(ang1) + palm_n * math.sin(ang1))
                                d2 = _normalize(f_dir * math.cos(ang2) + palm_n * math.sin(ang2))
                                p_mid = kn_p + d1 * (0.032 * sy)
                                p_tip = p_mid + d2 * (0.026 * sy)
                                for f in make_limb_prism_between(kn_p, p_mid, 0.0085 * limb_rad_scale, 0.0095 * limb_rad_scale, sides=8):
                                    polygons_world.append((f, hand_rgb))
                                for f in make_limb_prism_between(p_mid, p_tip, 0.0072 * limb_rad_scale, 0.0085 * limb_rad_scale, sides=8):
                                    polygons_world.append((f, hand_rgb))

                        # (d) 冬季厚底靴筒与鞋靴轮廓（确保脚底平稳贴合地面）
                        for l_side in ("L", "R"):
                            j_kn = J[f"knee_{l_side}"]
                            j_an = J[f"ankle_{l_side}"]
                            j_to = J[f"toe_{l_side}"]
                            boot_top = j_kn + (j_an - j_kn) * 0.46
                            for f in make_limb_prism_between(boot_top, j_an, 0.056 * limb_rad_scale, 0.062 * limb_rad_scale, sides=12):
                                polygons_world.append((f, boot_rgb))
                            f_boot = j_to - j_an
                            f_boot[1] = 0.0
                            if float(np.linalg.norm(f_boot)) < 1e-6:
                                f_boot = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                            else:
                                f_boot = _normalize(f_boot)
                            b_side = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float64), f_boot)
                            sole_y = max(0.030 * sy, float(j_an[1]) * 0.72)
                            heel_p = np.array([float(j_an[0]), sole_y, float(j_an[2])], dtype=np.float64) - f_boot * (0.055 * sz)
                            toe_p = np.array([float(j_an[0]), sole_y, float(j_an[2])], dtype=np.float64) + f_boot * (0.155 * sz)
                            for f in make_oriented_box_between(heel_p, toe_p, b_side, 0.092 * sx, 0.060 * sy):
                                polygons_world.append((f, boot_rgb))

                sx, sy, sz = _get_char_scale_xyz(char_t)
                yaw = math.radians(float(char_t.get("rot", [0, 0, 0])[1]))
                c, s = math.cos(yaw), math.sin(yaw)
                affine = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ np.diag([sx, sy, sz])
                origin = np.asarray(char_t.get("pos", [0, 0, 0]), dtype=np.float64)
                for index in range(polygon_start, len(polygons_world)):
                    face, color = polygons_world[index]
                    polygons_world[index] = (face @ affine.T + origin, color)

                # 动态地面接触阴影（随脚步与身体重心实时贴地移动，强化落地感、杜绝漂浮）
                if scene.get("contact_shadows"):
                    sh_boot_rgb = (118, 125, 134)
                    sh_body_rgb = (138, 145, 154)
                    pel_w = np.asarray(J["pelvis"], dtype=np.float64) @ affine.T + origin
                    T_sh_body = make_transform([float(pel_w[0]), 0.002, float(pel_w[2])], [0.0, 0.0, 0.0], 1.0)
                    _append_primitive_faces_and_edges("cylinder", T_sh_body, 0.62 * sx, 0.004, 0.56 * sz, sh_body_rgb, is_decal=True)
                    for l_side in ("L", "R"):
                        an_w = np.asarray(J[f"ankle_{l_side}"], dtype=np.float64) @ affine.T + origin
                        lift_h = max(0.0, float(an_w[1]) - 0.045 * sy)
                        r_foot_sh = max(0.14, 0.26 - lift_h * 0.9)
                        c_foot_sh = (
                            min(165, int(sh_boot_rgb[0] + lift_h * 190)),
                            min(172, int(sh_boot_rgb[1] + lift_h * 190)),
                            min(180, int(sh_boot_rgb[2] + lift_h * 190)),
                        )
                        T_sh_f = make_transform([float(an_w[0]), 0.003, float(an_w[2]) + 0.02], [0.0, 0.0, 0.0], 1.0)
                        _append_primitive_faces_and_edges("cylinder", T_sh_f, r_foot_sh * sx, 0.004, (r_foot_sh * 1.25) * sz, c_foot_sh, is_decal=True)

    # 3. 先绘制地面/路面高程分层与贴地薄片，再用画家算法光栅化绘制所有立体 3D 面片与 SketchUp 黑色细结构边线
    if ground_layer_polys:
        ground_layer_items = []
        for layer_y, pts_w, base_rgb in ground_layer_polys:
            proj = _project_world_poly(pts_w, base_rgb, shade=True)
            if proj is not None:
                depth_g, poly_2d_g, col_g = proj
                ground_layer_items.append((round(layer_y, 3), -depth_g, poly_2d_g, col_g))
        ground_layer_items.sort(key=lambda item: (item[0], item[1]))
        for _, _, poly_2d_g, col_g in ground_layer_items:
            draw.polygon(poly_2d_g, fill=col_g, outline=col_g)
    if decal_polys:
        draw_world_polys_far_to_near(decal_polys, shade=False)
    drawable: List[Tuple[float, str, Any, Tuple[int, int, int]]] = []
    for pts_w, base_rgb in polygons_world:
        pts_c = cam.world_to_cam(pts_w)
        if np.all(pts_c[:, 2] < cam.near):
            continue
        clipped = clip_polygon_near(pts_c, near=cam.near)
        if len(clipped) < 3:
            continue
        depth = float(np.mean(clipped[:, 2]))
        z = np.maximum(clipped[:, 2], cam.near)
        sx = (cam.width * 0.5) + (clipped[:, 0] / z) * cam.focal
        sy = (cam.height * 0.5) - (clipped[:, 1] / z) * cam.focal
        poly_2d = [(float(x), float(y)) for x, y in zip(sx, sy)]
        drawable.append((depth, "poly", poly_2d, apply_fog(shade_polygon(base_rgb, pts_w), depth)))

    light_dir_u = _normalize(np.array([0.45, 0.80, 0.42], dtype=np.float64))
    for pts_w, base_rgb in unlit_polys:
        pts_c = cam.world_to_cam(pts_w)
        if np.all(pts_c[:, 2] < cam.near):
            continue
        clipped = clip_polygon_near(pts_c, near=cam.near)
        if len(clipped) < 3:
            continue
        depth = float(np.mean(clipped[:, 2]))
        z = np.maximum(clipped[:, 2], cam.near)
        sx = (cam.width * 0.5) + (clipped[:, 0] / z) * cam.focal
        sy = (cam.height * 0.5) - (clipped[:, 1] / z) * cam.focal
        poly_2d = [(float(x), float(y)) for x, y in zip(sx, sy)]
        nrm = np.cross(pts_w[1] - pts_w[0], pts_w[2] - pts_w[0])
        nl = float(np.linalg.norm(nrm))
        fac = 0.88 + 0.12 * (abs(float(np.dot(nrm / nl, light_dir_u))) if nl > 1e-9 else 0.5)
        col_u = (min(255, int(base_rgb[0] * fac)), min(255, int(base_rgb[1] * fac)), min(255, int(base_rgb[2] * fac)))
        drawable.append((depth - 0.015, "poly", poly_2d, apply_fog(col_u, depth)))

    edge_rgb = (27, 29, 32)
    for pt0_w, pt1_w, depth_bias in edge_segments_world:
        seg_c = cam.world_to_cam(np.array([pt0_w, pt1_w], dtype=np.float64))
        c0, c1 = seg_c[0], seg_c[1]
        if c0[2] < cam.near and c1[2] < cam.near:
            continue
        if c0[2] < cam.near:
            dz = c1[2] - c0[2]
            t_clip = (cam.near - c0[2]) / dz if abs(dz) > 1e-9 else 0.0
            c0 = c0 + t_clip * (c1 - c0)
            c0[2] = cam.near
        elif c1[2] < cam.near:
            dz = c0[2] - c1[2]
            t_clip = (cam.near - c1[2]) / dz if abs(dz) > 1e-9 else 0.0
            c1 = c1 + t_clip * (c0 - c1)
            c1[2] = cam.near
        seg_depth = 0.5 * float(c0[2] + c1[2]) + float(depth_bias)
        x0 = (cam.width * 0.5) + (float(c0[0]) / max(cam.near, float(c0[2]))) * cam.focal
        y0 = (cam.height * 0.5) - (float(c0[1]) / max(cam.near, float(c0[2]))) * cam.focal
        x1 = (cam.width * 0.5) + (float(c1[0]) / max(cam.near, float(c1[2]))) * cam.focal
        y1 = (cam.height * 0.5) - (float(c1[1]) / max(cam.near, float(c1[2]))) * cam.focal
        drawable.append((seg_depth, "line", [(x0, y0), (x1, y1)], apply_fog(edge_rgb, max(cam.near, seg_depth))))

    drawable.sort(key=lambda x: x[0], reverse=True)
    for _, kind, geom_2d, col in drawable:
        if kind == "poly":
            draw.polygon(geom_2d, fill=col, outline=col)
        else:
            draw.line(geom_2d, fill=col, width=1)

    # 4. 绘制 OpenPose 骨骼线与关节球 (在 clay_and_pose 或 pose_only 模式下)
    if mode in ("clay_and_pose", "pose_only"):
        openpose_bones = [
            ("pelvis", "neck", (0, 180, 255), 5),
            ("neck", "head_center", (255, 0, 85), 5),
            ("neck", "shoulder_L", (85, 255, 0), 5),
            ("shoulder_L", "elbow_L", (102, 255, 0), 5),
            ("elbow_L", "wrist_L", (180, 255, 0), 4),
            ("neck", "shoulder_R", (255, 128, 0), 5),
            ("shoulder_R", "elbow_R", (255, 170, 0), 5),
            ("elbow_R", "wrist_R", (255, 230, 0), 4),
            ("pelvis", "hip_L", (0, 255, 128), 5),
            ("hip_L", "knee_L", (0, 255, 170), 6),
            ("knee_L", "ankle_L", (0, 255, 255), 5),
            ("pelvis", "hip_R", (0, 128, 255), 5),
            ("hip_R", "knee_R", (0, 85, 255), 6),
            ("knee_R", "ankle_R", (85, 0, 255), 5),
        ]
        for ik_res in characters_ik:
            J = ik_res["joints"]
            for j0, j1, bcol, bwidth in openpose_bones:
                p0 = cam.project_point(J[j0])
                p1 = cam.project_point(J[j1])
                if p0 and p1:
                    draw.line([(p0[0], p0[1]), (p1[0], p1[1])], fill=bcol, width=bwidth)
            for jname, jpt in J.items():
                if jname == "head_target":
                    continue
                proj = cam.project_point(jpt)
                if proj:
                    r = 4
                    draw.ellipse([proj[0] - r, proj[1] - r, proj[0] + r, proj[1] + r], fill=(255, 50, 75), outline=(255, 255, 255))

    return img


ASPECT_EXPORT_SIZES: Dict[str, Tuple[int, int]] = {
    "2.39:1": (1148, 480),
    "2.35:1": (1280, 544),
    "2.33:1": (1120, 480),
    "16:9": (960, 540),
    "1.85:1": (998, 540),
    "4:3": (720, 540),
    "1:1": (540, 540),
    "9:16": (540, 960),
}


def resolve_scene_export_dimensions(
    scene: Dict[str, Any],
    req_width: Optional[int] = None,
    req_height: Optional[int] = None,
) -> Tuple[int, int]:
    """
    统一解析摄影机预览框、截图与 MP4 视频导出的画幅分辨率（确保偶数宽高，兼容 H.264 yuv420p）：
    - 若摄影机配置了显式 export_width / export_height 或自定义比例（如参考片 1680x720 = 2.33:1），优先与摄影机画幅保持一致。
    """
    cam = scene.get("camera") or {}
    aspect_str = str(cam.get("aspect_ratio") or "2.39:1").strip()

    # 若调用方传入了非默认 (960x540) 的明确尺寸，则采用调用方尺寸并对齐到偶数
    if req_width and req_height and (int(req_width), int(req_height)) != (960, 540):
        w = max(64, (int(req_width) // 2) * 2)
        h = max(64, (int(req_height) // 2) * 2)
        return w, h

    if cam.get("export_width") and cam.get("export_height"):
        w = max(64, (int(cam["export_width"]) // 2) * 2)
        h = max(64, (int(cam["export_height"]) // 2) * 2)
        return w, h

    if aspect_str in ASPECT_EXPORT_SIZES:
        return ASPECT_EXPORT_SIZES[aspect_str]

    # 解析形如 "W:H" 或自定义数值比例
    if ":" in aspect_str:
        parts = aspect_str.split(":", 1)
        try:
            aw = float(parts[0])
            ah = float(parts[1])
            if aw > 0 and ah > 0:
                ratio = aw / ah
                if ratio >= 1.0:
                    base_h = 540 if ratio <= 2.0 else 480
                    base_w = max(64, (int(round(base_h * ratio)) // 2) * 2)
                    return base_w, base_h
                else:
                    base_w = 540
                    base_h = max(64, (int(round(base_w / ratio)) // 2) * 2)
                    return base_w, base_h
        except Exception:
            pass

    ref_vid = scene.get("ref_video") or {}
    if ref_vid.get("width") and ref_vid.get("height"):
        rw, rh = float(ref_vid["width"]), float(ref_vid["height"])
        if rw > 0 and rh > 0:
            ratio = rw / rh
            base_h = 480 if ratio > 2.0 else 540
            base_w = max(64, (int(round(base_h * ratio)) // 2) * 2)
            return base_w, base_h

    w = max(64, (int(req_width or 960) // 2) * 2)
    h = max(64, (int(req_height or 540) // 2) * 2)
    return w, h


AREA_DENSITY_BUDGET_PER_100M2: Dict[str, Dict[str, int]] = {
    "fast": {
        "near_mid_trees": 6,
        "normal_props": 4,
        "decor_rubble_snow": 0,
        "bg_mannequins": 2,
    },
    "balanced": {
        "near_mid_trees": 12,
        "normal_props": 8,
        "decor_rubble_snow": 4,
        "bg_mannequins": 4,
    },
    "composition": {
        "near_mid_trees": 20,
        "normal_props": 12,
        "decor_rubble_snow": 8,
        "bg_mannequins": 6,
    },
}

FAR_FOREST_BUDGET_PER_1000M2: Dict[str, int] = {
    "fast": 8,
    "balanced": 16,
    "composition": 24,
}

VISIBLE_PROXY_BUDGETS: Dict[str, Dict[str, int]] = {
    "fast": {"visible_trees": 300, "visible_bg_props": 80, "label": "极速"},
    "balanced": {"visible_trees": 600, "visible_bg_props": 160, "label": "均衡"},
    "composition": {"visible_trees": 1000, "visible_bg_props": 240, "label": "构图优先"},
}

_TAIGA_AERIAL_FULL_CACHE: Optional[List[Tuple[float, float, float, float, float, Tuple[int, int, int], int]]] = None


def get_taiga_aerial_full_specs() -> List[Tuple[float, float, float, float, float, Tuple[int, int, int], int]]:
    """与前端 static/index.html seed=108 完全同源的固定种子远景单锥树生成（保证预览与导出一致且不闪烁）。"""
    global _TAIGA_AERIAL_FULL_CACHE
    if _TAIGA_AERIAL_FULL_CACHE is not None:
        return _TAIGA_AERIAL_FULL_CACHE
    specs: List[Tuple[float, float, float, float, float, Tuple[int, int, int], int]] = []
    seed = 108

    def _rand() -> float:
        nonlocal seed
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        return seed / 4294967296.0

    for ri in range(13):
        r_ring = 14.5 + (ri / 12.0) * (72.0 - 14.5)
        n_trees = int(math.floor(r_ring * 1.35))
        for k in range(n_trees):
            ang = (k / float(max(1, n_trees))) * math.pi * 2.0 + (_rand() - 0.5) * 0.14
            rr = r_ring + (_rand() - 0.5) * 3.4
            tx = rr * math.sin(ang)
            tz_our = rr * math.cos(ang)
            if -3.5 < tx < 14.0 and -2.0 < tz_our < 22.0:
                continue
            th = 9.5 + _rand() * 4.0
            rsc = 0.88 + _rand() * 0.30
            col_rgb = (168, 188, 209) if _rand() > 0.38 else (122, 145, 171)
            rank_hash = ((len(specs) * 2654435761) ^ 0x9E3779B9) & 0xFFFFFFFF
            specs.append((tx, th * 0.52, -tz_our, 1.55 * rsc, th * 0.72, col_rgb, rank_hash))

    for zi in range(42):
        gz = -105.0 + (zi / 41.0) * (585.0 + 105.0)
        x_span = 110.0 + max(0.0, gz) * 0.54
        n_x = max(12, int(math.floor(x_span / 7.2)))
        for xi in range(n_x):
            gx = -x_span + (xi / float(max(1, n_x - 1))) * (x_span * 2.0)
            tx = gx + (_rand() - 0.5) * 4.6
            tz_our = gz + (_rand() - 0.5) * 4.6
            if math.hypot(tx, tz_our) < 70.0:
                continue
            if tz_our > 38.0:
                r_cx = 30.0 + 26.0 * math.sin((tz_our - 45.0) * 0.024)
                if abs(tx - r_cx) < 28.0:
                    continue
                if 115.0 < tz_our < 290.0:
                    t_cx = -95.0 + (tz_our - 125.0) * 0.47
                    if abs(tx - t_cx) < 20.0:
                        continue
            th = 7.8 + _rand() * 3.8
            rsc = 0.95 + _rand() * 0.52
            col_rgb = (168, 188, 209) if _rand() > 0.40 else (122, 145, 171)
            rank_hash = ((len(specs) * 2654435761) ^ 0x85EBCA6B) & 0xFFFFFFFF
            specs.append((tx, th * 0.45, -tz_our, 1.95 * rsc, th * 0.85, col_rgb, rank_hash))

    specs.sort(key=lambda item: item[6])
    _TAIGA_AERIAL_FULL_CACHE = specs
    return specs


def get_taiga_aerial_proxy_subset(max_count: int = 300) -> List[Tuple[float, float, float, float, float, Tuple[int, int, int]]]:
    specs = get_taiga_aerial_full_specs()
    cap = max(0, min(len(specs), int(max_count)))
    return [(x, y, z, r, h, c) for (x, y, z, r, h, c, _rk) in specs[:cap]]


def normalize_priority_mode(mode_raw: Optional[str], default: str = "fast") -> str:
    m = str(mode_raw or "").strip().lower()
    if m in ("fast", "极速", "speed"):
        return "fast"
    if m in ("balanced", "均衡", "medium"):
        return "balanced"
    if m in ("composition", "quality", "构图优先", "构图还原"):
        return "composition"
    return default


def parse_composition_spec_text(spec_text: str, duration: float) -> List[Dict[str, Any]]:
    """支持自然语言输入关键构图（例如：'第3秒低机位、第11秒伊万近景、最后全队前进'）。"""
    dur = max(0.5, float(duration or 4.0))
    if not spec_text or not str(spec_text).strip():
        return []
    parts = [p.strip() for p in re.split(r"[、，,；;\n]+", str(spec_text)) if p.strip()]
    out: List[Dict[str, Any]] = []
    for idx, seg in enumerate(parts):
        t_val: Optional[float] = None
        m_sec = re.search(r"第?\s*(\d+(?:\.\d+)?)\s*(?:秒|s|S)", seg)
        if m_sec:
            t_val = float(m_sec.group(1))
        elif any(kw in seg for kw in ("首", "开头", "起幅", "第一", "开始")):
            t_val = 0.0
        elif any(kw in seg for kw in ("最后", "结尾", "尾幅", "收尾", "末帧", "终点")):
            t_val = dur
        elif any(kw in seg for kw in ("中段", "中间", "转折", "中景")):
            t_val = round(dur * 0.5, 2)
        else:
            m_num = re.search(r"(\d+(?:\.\d+)?)", seg)
            if m_num:
                t_val = float(m_num.group(1))
        if t_val is None:
            t_val = round((idx / float(max(1, len(parts) - 1))) * dur, 2) if len(parts) > 1 else 0.0
        t_clamped = round(max(0.0, min(dur, t_val)), 2)
        note = re.sub(r"^第?\s*\d+(?:\.\d+)?\s*(?:秒|s|S)\s*[:：\-]?\s*", "", seg).strip() or seg
        out.append({
            "id": f"comp_nl_{idx + 1}",
            "t": t_clamped,
            "role": "custom",
            "note": note,
        })
    out.sort(key=lambda x: float(x["t"]))
    return out


def recommend_default_compositions(scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    """默认推荐首幅、动作转折、尾幅 3 个关键构图时间点（从摄影机/角色关键帧自动识别核心转折）。"""
    dur = max(0.5, float(scene.get("duration") or 4.0))
    cam_kfs = sorted(
        [k for k in (scene.get("camera_keyframes") or []) if isinstance(k, dict)],
        key=lambda k: float(k.get("t", 0.0)),
    )
    t_start = 0.0
    t_end = round(dur, 2)
    t_turn = round(dur * 0.5, 2)
    if len(cam_kfs) >= 3:
        best_score = -1.0
        for i in range(1, len(cam_kfs) - 1):
            kt = float(cam_kfs[i].get("t", 0.0))
            if kt <= 0.15 or kt >= dur - 0.15:
                continue
            p_prev = np.asarray((cam_kfs[i - 1].get("pos") or [0, 0, 0])[:3], dtype=float)
            p_cur = np.asarray((cam_kfs[i].get("pos") or [0, 0, 0])[:3], dtype=float)
            p_next = np.asarray((cam_kfs[i + 1].get("pos") or [0, 0, 0])[:3], dtype=float)
            v1 = p_cur - p_prev
            v2 = p_next - p_cur
            n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
            turn_score = (1.0 - float(np.dot(v1, v2)) / max(1e-6, n1 * n2)) if (n1 > 1e-3 and n2 > 1e-3) else 0.2
            mid_bias = 1.0 - abs(kt - dur * 0.5) / max(0.5, dur * 0.5)
            score = turn_score * 1.5 + mid_bias * 0.5
            if score > best_score:
                best_score = score
                t_turn = round(max(0.0, min(dur, kt)), 2)
    return [
        {"id": "comp_start", "t": t_start, "role": "start", "note": "首幅构图（起始空间与主体站位）"},
        {"id": "comp_turn", "t": t_turn, "role": "turn", "note": "中段动作/运镜转折构图"},
        {"id": "comp_end", "t": t_end, "role": "end", "note": "尾幅落幅构图（终点遮挡与视线关系）"},
    ]


def estimate_scene_area_and_visible_budget(scene: Dict[str, Any], priority_mode: str = "fast") -> Dict[str, Any]:
    """按 10m×10m (100㎡) 中心归属统计面积密度、整镜同时可见代理预算、保护项与全镜头运动走廊穿插检查。"""
    p_mode = normalize_priority_mode(priority_mode, default="fast")
    area_caps = AREA_DENSITY_BUDGET_PER_100M2[p_mode]
    vis_caps = VISIBLE_PROXY_BUDGETS[p_mode]
    far_cap_1000 = FAR_FOREST_BUDGET_PER_1000M2[p_mode]

    dur = max(0.5, float(scene.get("duration") or 4.0))
    objs = [o for o in (scene.get("objects") or []) if isinstance(o, dict) and not o.get("is_helper")]
    char_objs = [o for o in objs if o.get("type") == "character" and not o.get("is_background_crowd")]
    bound_prop_ids = {
        str(o.get("id"))
        for o in objs
        if isinstance(o.get("parent_binding"), dict)
        and o["parent_binding"].get("enabled") is not False
        and o["parent_binding"].get("target_id")
    }

    cell_counts: Dict[Tuple[int, int], Dict[str, int]] = {}
    protected_items: List[Dict[str, Any]] = []
    near_tree_total = 0
    far_tree_source_total = 0
    bg_prop_total = 0
    decor_rubble_total = 0

    for o in objs:
        oid = str(o.get("id") or "")
        oname = str(o.get("name") or oid)
        otype = str(o.get("type") or "box")
        role = str(o.get("arch_role") or "")
        is_bound = oid in bound_prop_ids
        is_char = otype == "character" and not o.get("is_background_crowd")
        is_prot = bool(
            is_char
            or is_bound
            or o.get("protected") is True
            or o.get("is_key_occlusion") is True
            or o.get("locked") is True
            or role in ("floor", "ground", "road", "wall", "roof", "ceiling", "gun", "vehicle")
            or otype == "uaz469"
        )
        if is_prot and o.get("procedural_forest") != "taiga_aerial_sea":
            protected_items.append({
                "id": oid,
                "name": oname,
                "reason": "剧情人物" if is_char else ("角色绑定附属物" if is_bound else ("手工锁定/关键保护" if (o.get("protected") or o.get("locked")) else "空间边界/主结构")),
            })

        if o.get("procedural_forest") == "taiga_aerial_sea":
            if is_object_manually_visible(o):
                far_tree_source_total += len(get_taiga_aerial_full_specs())
            continue

        parts = o.get("parts") if isinstance(o.get("parts"), list) else None
        if parts and ("forest" in oid or "tree" in oid or "trunk" in oid):
            base_px, _, base_pz = (o.get("pos") or [0.0, 0.0, 0.0])[:3]
            for pt in parts:
                if not isinstance(pt, dict) or pt.get("hidden") is True or pt.get("visible") is False:
                    continue
                p_pos = pt.get("pos") or [0.0, 0.0, 0.0]
                wx = float(base_px) + float(p_pos[0])
                wz = float(base_pz) + float(p_pos[2])
                cell = (int(math.floor(wx / 10.0)), int(math.floor(wz / 10.0)))
                bucket = cell_counts.setdefault(cell, {"near_mid_trees": 0, "normal_props": 0, "decor_rubble_snow": 0, "bg_mannequins": 0})
                bucket["near_mid_trees"] += 1
                near_tree_total += 1
            continue

        pos = o.get("pos") or [0.0, 0.0, 0.0]
        cell = (int(math.floor(float(pos[0]) / 10.0)), int(math.floor(float(pos[2]) / 10.0)))
        bucket = cell_counts.setdefault(cell, {"near_mid_trees": 0, "normal_props": 0, "decor_rubble_snow": 0, "bg_mannequins": 0})
        if otype == "cone" or "tree" in oid.lower() or "树" in oname:
            bucket["near_mid_trees"] += 1
            near_tree_total += 1
        elif o.get("is_background_crowd"):
            bucket["bg_mannequins"] += 1
        elif o.get("is_decorative_rubble"):
            bucket["decor_rubble_snow"] += 1
            decor_rubble_total += 1
        elif not is_char and not is_bound and role not in ("floor", "ground", "road", "wall", "roof", "ceiling"):
            bucket["normal_props"] += 1
            bg_prop_total += 1

    # 采样全镜头摄影机运动走廊与碰撞/穿树/穿人安全检查
    sample_steps = max(5, min(41, int(math.ceil(dur / 0.5)) + 1))
    corridor_cells = set()
    corridor_warnings: List[str] = []
    warned_keys = set()
    for s_idx in range(sample_steps):
        ts = round((s_idx / float(max(1, sample_steps - 1))) * dur, 3)
        cpos, ctar, _cfov = interpolate_camera(scene, ts)
        cx, cy, cz = float(cpos[0]), float(cpos[1]), float(cpos[2])
        for dx_c in (-1, 0, 1):
            for dz_c in (-1, 0, 1):
                corridor_cells.add((int(math.floor(cx / 10.0)) + dx_c, int(math.floor(cz / 10.0)) + dz_c))
                corridor_cells.add((int(math.floor(float(ctar[0]) / 10.0)) + dx_c, int(math.floor(float(ctar[2]) / 10.0)) + dz_c))
        if cy < 6.0:
            for co in char_objs:
                cst = interpolate_character_at_time(co, ts)
                if not cst.get("visible", True):
                    continue
                cp = cst.get("pos") or [0.0, 0.0, 0.0]
                d_xz = math.hypot(cx - float(cp[0]), cz - float(cp[2]))
                if d_xz < 0.28 and abs(cy - (float(cp[1]) + 0.9)) < 0.95:
                    wk = f"char:{co.get('id')}"
                    if wk not in warned_keys:
                        warned_keys.add(wk)
                        corridor_warnings.append(f"t={ts:.1f}s 摄影机距角色「{co.get('name', co.get('id'))}」仅 {d_xz:.2f}m，请注意近景穿插")

    active_cells_n = max(1, len(corridor_cells) or len(cell_counts))
    corridor_area_m2 = active_cells_n * 100
    max_cell_trees = max([v["near_mid_trees"] for v in cell_counts.values()], default=0)
    max_cell_props = max([v["normal_props"] for v in cell_counts.values()], default=0)

    effective_far_proxy = min(far_tree_source_total, max(0, vis_caps["visible_trees"] - near_tree_total)) if far_tree_source_total > 0 else 0
    effective_visible_trees = near_tree_total + effective_far_proxy
    effective_visible_props = bg_prop_total if p_mode != "fast" else max(0, bg_prop_total - decor_rubble_total)

    exceptions: List[str] = []
    if near_tree_total > vis_caps["visible_trees"]:
        exceptions.append(f"原工程关键/近中景树共 {near_tree_total} 棵，超出{vis_caps['label']}档可见预算 {vis_caps['visible_trees']}，已按非破坏规则保留原资产并仅收敛远景代理。")
    if max_cell_trees > area_caps["near_mid_trees"]:
        exceptions.append(f"局部 100㎡ 网格最高含 {max_cell_trees} 棵近景树（自动生成初始预算为 {area_caps['near_mid_trees']}/100㎡），现有工程树木保留不删。")

    return {
        "priority_mode": p_mode,
        "priority_label": vis_caps["label"],
        "corridor_cells": active_cells_n,
        "corridor_area_m2": corridor_area_m2,
        "area_caps_per_100m2": area_caps,
        "far_forest_cap_per_1000m2": far_cap_1000,
        "visible_proxy_budget": vis_caps,
        "current_counts": {
            "story_characters": len(char_objs),
            "bound_accessories": len(bound_prop_ids),
            "near_mid_trees": near_tree_total,
            "far_forest_source_cones": far_tree_source_total,
            "effective_visible_tree_proxies": effective_visible_trees,
            "effective_visible_bg_props": effective_visible_props,
            "max_trees_per_100m2_cell": max_cell_trees,
            "max_props_per_100m2_cell": max_cell_props,
        },
        "protected_count": len(protected_items),
        "protected_items": protected_items[:24],
        "corridor_warnings": corridor_warnings,
        "budget_exceptions": exceptions,
    }


def prepare_export_confirmation_summary(
    scene: Dict[str, Any],
    priority_mode: Optional[str] = None,
    compositions: Optional[List[Dict[str, Any]]] = None,
    composition_text: Optional[str] = None,
    render_previews: bool = False,
    preview_width: int = 320,
    preview_height: int = 144,
) -> Dict[str, Any]:
    """构建导出/复现前的一次性轻量确认摘要（带出已有 export_plan，支持低分辨率关键构图预览图）。"""
    existing_plan = scene.get("export_plan") if isinstance(scene.get("export_plan"), dict) else {}
    dur = max(0.5, float(scene.get("duration") or 4.0))
    p_mode = normalize_priority_mode(
        priority_mode or existing_plan.get("priority_mode"),
        default="fast",
    )
    if composition_text and str(composition_text).strip():
        comp_list = parse_composition_spec_text(str(composition_text), dur)
    elif isinstance(compositions, list) and len(compositions) > 0:
        comp_list = []
        for idx, c in enumerate(compositions):
            if isinstance(c, dict):
                ct = round(max(0.0, min(dur, float(c.get("t", 0.0)))), 2)
                comp_list.append({
                    "id": str(c.get("id") or f"comp_{idx + 1}"),
                    "t": ct,
                    "role": str(c.get("role") or "custom"),
                    "note": str(c.get("note") or f"第 {ct:.1f}s 关键构图"),
                })
        comp_list.sort(key=lambda x: float(x["t"]))
    elif isinstance(existing_plan.get("compositions"), list) and len(existing_plan["compositions"]) > 0:
        comp_list = copy.deepcopy(existing_plan["compositions"])
    else:
        comp_list = recommend_default_compositions(scene)

    budget_info = estimate_scene_area_and_visible_budget(scene, p_mode)
    previews: List[Dict[str, Any]] = []
    if render_previews:
        import io as _io
        sc_preview = copy.deepcopy(scene)
        sc_preview["export_plan"] = {"priority_mode": p_mode, "compositions": comp_list, "confirmed": True}
        pw, ph = resolve_scene_export_dimensions(sc_preview, preview_width, preview_height)
        scale_r = min(1.0, float(max(160, preview_width)) / float(max(1, pw)))
        pw_s = max(128, (int(round(pw * scale_r)) // 2) * 2)
        ph_s = max(72, (int(round(ph * scale_r)) // 2) * 2)
        for c_item in comp_list:
            ct = float(c_item.get("t", 0.0))
            im = render_single_frame(sc_preview, ct, width=pw_s, height=ph_s)
            buf = _io.BytesIO()
            im.save(buf, format="JPEG", quality=82)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            previews.append({
                "id": c_item.get("id"),
                "t": ct,
                "note": c_item.get("note", ""),
                "width": pw_s,
                "height": ph_s,
                "data_url": f"data:image/jpeg;base64,{b64}",
            })

    has_saved_confirmation = bool(existing_plan.get("confirmed") is True)
    speed_policy = validate_scene_speed_policy(scene)
    return {
        "has_saved_plan": has_saved_confirmation,
        "saved_confirmed_at": existing_plan.get("confirmed_at"),
        "priority_mode": p_mode,
        "priority_label": VISIBLE_PROXY_BUDGETS[p_mode]["label"],
        "compositions": comp_list,
        "budget": budget_info,
        "speed_policy": speed_policy,
        "previews": previews,
        "questions": [
            "1. 这段镜头最关键的是哪几个构图？（可点选首幅/转折/尾幅或直接填如“第3秒低机位、第11秒伊万近景、最后全队前进”）",
            "2. 优先极速还是构图还原？（默认推荐极速；已有场景保留原资产，仅调整派生代理）",
            "3. 是否沿用本次 100㎡ 与整镜同时可见代理预算？（可直接一键沿用）",
        ],
    }


def check_export_confirmation_gate(
    scene: Dict[str, Any],
    export_confirmation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """校验是否已获得用户显式导出确认（或显式点击“沿用上次确认并导出”）。绝不把默认建议或超时沉默当成已确认。"""
    conf = export_confirmation if isinstance(export_confirmation, dict) else {}
    saved_plan = scene.get("export_plan") if isinstance(scene.get("export_plan"), dict) else {}
    reuse_saved = bool(conf.get("reuse_saved_plan") is True and saved_plan.get("confirmed") is True)
    explicitly_confirmed = bool(conf.get("confirmed") is True or reuse_saved)

    summary = prepare_export_confirmation_summary(
        scene,
        priority_mode=conf.get("priority_mode") or saved_plan.get("priority_mode"),
        compositions=conf.get("compositions") or saved_plan.get("compositions"),
        composition_text=conf.get("composition_text"),
        render_previews=False,
    )
    if not explicitly_confirmed:
        return {"confirmed": False, "summary": summary, "plan": None}

    plan = {
        "confirmed": True,
        "confirmed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "priority_mode": summary["priority_mode"],
        "priority_label": summary["priority_label"],
        "compositions": summary["compositions"],
        "preserve_existing_scene": bool(conf.get("preserve_existing_scene", True)),
        "use_area_and_visible_budget": bool(conf.get("use_area_and_visible_budget", True)),
        "budget_snapshot": summary["budget"],
    }
    return {"confirmed": True, "summary": summary, "plan": plan}


_MP4_WORKER_SCENE: Optional[Dict[str, Any]] = None


def _mp4_init_worker(scene: Dict[str, Any]) -> None:
    global _MP4_WORKER_SCENE
    _MP4_WORKER_SCENE = scene


def _mp4_render_frame_bytes(job: Tuple[float, int, int]) -> bytes:
    t, width, height = job
    assert _MP4_WORKER_SCENE is not None
    return render_single_frame(_MP4_WORKER_SCENE, t, width=width, height=height).tobytes()


def render_scene_to_mp4(
    scene: Dict[str, Any],
    output_path: str,
    width: int = 960,
    height: int = 540,
    export_confirmation: Optional[Dict[str, Any]] = None,
    require_confirmation: bool = False,
) -> Dict[str, Any]:
    """
    按项目帧率 (fps) 与总帧数 (total_frames) 输出 MP4 视频：
    - 当 require_confirmation=True 时，强制核对关键构图与预算确认状态；
    - 严格按整数帧索引采样：t = min(duration, frame_index / fps)，最后一帧 (frame_index = total_frames - 1)
      的采样时间定义为 min(duration, (total_frames - 1) / fps)，保证相邻帧时间步长恒为 1 / fps；
    - 使用已确认的极简表示渲染，不在最后偷偷恢复几千棵未确认复杂树，同时保留原 scene 数据不破坏。
    """
    if require_confirmation:
        gate = check_export_confirmation_gate(scene, export_confirmation)
        if not gate["confirmed"]:
            return {
                "ok": False,
                "confirmation_required": True,
                "summary": gate["summary"],
            }
        render_scene = copy.copy(scene)
        render_scene["export_plan"] = gate["plan"]
    elif isinstance(export_confirmation, dict) and export_confirmation.get("priority_mode"):
        render_scene = copy.copy(scene)
        render_scene["export_plan"] = dict(export_confirmation)
    else:
        render_scene = scene

    t0 = time.time()
    width, height = resolve_scene_export_dimensions(render_scene, width, height)
    fps = int(render_scene.get("fps", 24))
    duration = float(render_scene.get("duration", 4.0))
    ref_total = (render_scene.get("ref_video") or {}).get("total_frames")
    explicit_frames = render_scene.get("total_frames") or (ref_total if ref_total and abs(float(ref_total) / float(fps) - duration) < 0.05 else None)
    total_frames = max(1, int(explicit_frames) if explicit_frames else int(round(fps * duration)))

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-preset", "ultrafast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        frame_jobs = [
            (min(duration, float(f_idx) / float(fps)) if total_frames > 1 else 0.0, width, height)
            for f_idx in range(total_frames)
        ]
        workers = min(6, max(1, (os.cpu_count() or 1) - 1), total_frames)
        used_parallel = False
        frames_written = 0
        if workers >= 2 and total_frames >= 16:
            try:
                ctx = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=ctx,
                    initializer=_mp4_init_worker,
                    initargs=(render_scene,),
                ) as executor:
                    for frame_bytes in executor.map(_mp4_render_frame_bytes, frame_jobs, chunksize=4):
                        proc.stdin.write(frame_bytes)
                        frames_written += 1
                used_parallel = True
            except Exception:
                if frames_written > 0:
                    raise
                used_parallel = False
        if not used_parallel:
            for t, w_f, h_f in frame_jobs:
                img = render_single_frame(render_scene, t, width=w_f, height=h_f)
                proc.stdin.write(img.tobytes())
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    companion_json_path = os.path.splitext(output_path)[0] + ".project.json"
    try:
        save_scene = {k: v for k, v in render_scene.items() if not str(k).startswith("_")}
        with open(companion_json_path, "w", encoding="utf-8") as jf:
            json.dump(save_scene, jf, ensure_ascii=False, indent=2)
    except Exception:
        companion_json_path = ""

    elapsed_ms = int((time.time() - t0) * 1000)
    return {
        "ok": True,
        "output_path": output_path,
        "project_snapshot_path": companion_json_path,
        "width": width,
        "height": height,
        "frames": total_frames,
        "fps": fps,
        "duration": duration,
        "elapsed_ms": elapsed_ms,
        "export_plan": render_scene.get("export_plan"),
    }


def render_5frame_parity_sheet(
    scene: Dict[str, Any],
    browser_frames: List[Dict[str, Any]],
    out_dir: str,
    width: int = 574,
    height: int = 240,
) -> Dict[str, Any]:
    """
    选取同一工程的 0%、25%、50%、75%、末帧(100%)，对比浏览器最终镜头与 Python 输出：
    - 校验画幅、焦距/FOV、机位、人物位置、关节姿态、遮挡与约束
    - 输出 5 行并排对照总表 PNG 及单帧并排图，返回定量差异报告
    """
    os.makedirs(out_dir, exist_ok=True)
    duration = max(0.01, float(scene.get("duration", 6.0)))
    fps = max(1.0, float(scene.get("fps", 24.0)))
    aspect_key = (scene.get("camera", {}) or {}).get("aspect_ratio", "2.39:1")
    ratios = [0.0, 0.25, 0.50, 0.75, 1.0]
    labels = ["0% (Start)", "25%", "50%", "75%", "100% (End)"]

    row_images: List[Image.Image] = []
    checkpoint_reports: List[Dict[str, Any]] = []
    ts_tag = int(time.time() * 1000)

    for idx, r in enumerate(ratios):
        t_val = round(duration * r, 4)
        frame_no = int(round(t_val * fps))
        bf_item = browser_frames[idx] if idx < len(browser_frames) and isinstance(browser_frames[idx], dict) else {}

        # 1. Python 渲染与解算数据
        py_img = render_single_frame(scene, t_val, width=width, height=height)
        cpos_py, ctar_py, cfov_py = interpolate_camera(scene, t_val)
        fmm_py = fov_to_focal_mm(cfov_py)
        char_map_py, ik_map_py, warnings_py = evaluate_scene_characters_at_time(scene, t_val)

        # 2. 解码浏览器画面
        b64_str = bf_item.get("data_url") or bf_item.get("image_b64") or ""
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        if b64_str:
            try:
                br_raw = Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")
                br_img = fit_image_to_aspect_box(br_raw, width, height, bg_rgb=(27, 30, 35))
            except Exception:
                br_img = py_img.copy()
        else:
            br_img = py_img.copy()

        # 3. 对比浏览器与 Python 的相机、人物与关节 3D 坐标误差
        br_cam = bf_item.get("camera") or {}
        cam_pos_err = 0.0
        cam_tar_err = 0.0
        fov_err = 0.0
        if isinstance(br_cam.get("pos"), list) and len(br_cam["pos"]) >= 3:
            cam_pos_err = float(np.linalg.norm(np.asarray(br_cam["pos"][:3], dtype=np.float64) - np.asarray(cpos_py, dtype=np.float64)))
        if isinstance(br_cam.get("target"), list) and len(br_cam["target"]) >= 3:
            cam_tar_err = float(np.linalg.norm(np.asarray(br_cam["target"][:3], dtype=np.float64) - np.asarray(ctar_py, dtype=np.float64)))
        if br_cam.get("fov") is not None:
            fov_err = abs(float(br_cam["fov"]) - float(cfov_py))

        br_chars = bf_item.get("characters") or {}
        max_char_pos_err = 0.0
        max_joint_err = 0.0
        for cid, c_py in char_map_py.items():
            c_br = br_chars.get(cid)
            if not isinstance(c_br, dict):
                continue
            if isinstance(c_br.get("pos"), list) and len(c_br["pos"]) >= 3:
                pe = float(np.linalg.norm(np.asarray(c_br["pos"][:3], dtype=np.float64) - np.asarray(c_py["pos"][:3], dtype=np.float64)))
                max_char_pos_err = max(max_char_pos_err, pe)
            j_br_map = c_br.get("joints") or {}
            j_py_map = ik_map_py.get(cid, {}).get("joints", {})
            for jn, j_py_vec in j_py_map.items():
                if jn in j_br_map and isinstance(j_br_map[jn], list) and len(j_br_map[jn]) >= 3:
                    je = float(np.linalg.norm(np.asarray(j_br_map[jn][:3], dtype=np.float64) - j_py_vec))
                    max_joint_err = max(max_joint_err, je)

        # 4. 构造并排对照单行图 (左: 浏览器最终镜头 WebGL | 右: Python 导出渲染)
        header_h = 34
        row_w = width * 2 + 12
        row_h = height + header_h + 6
        row_canvas = Image.new("RGB", (row_w, row_h), (20, 22, 25))
        d_row = ImageDraw.Draw(row_canvas)
        row_canvas.paste(br_img, (4, header_h))
        row_canvas.paste(py_img, (width + 8, header_h))
        d_row.line([(width + 6, header_h), (width + 6, row_h - 4)], fill=(91, 157, 255), width=2)

        tc_str = format_timecode_sec(t_val, fps)
        d_row.text(
            (10, 9),
            f"[{labels[idx]}] t={t_val:.2f}s (F:{frame_no}) | TC {tc_str} | Aspect {aspect_key} | f={fmm_py:.0f}mm (VFOV {cfov_py:.1f}deg)",
            fill=(231, 235, 240),
        )
        d_row.text(
            (width + 14, 9),
            f"Right: Python Export | dCam={cam_pos_err*1000:.2f}mm dPos={max_char_pos_err*1000:.2f}mm dJoint={max_joint_err*1000:.2f}mm",
            fill=(121, 182, 156),
        )
        # 标注角标
        d_row.rectangle([10, header_h + 6, 205, header_h + 26], fill=(20, 22, 25))
        d_row.text((16, header_h + 9), "Browser Final Camera", fill=(91, 157, 255))
        d_row.rectangle([width + 14, header_h + 6, width + 205, header_h + 26], fill=(20, 22, 25))
        d_row.text((width + 20, header_h + 9), "Python Export Render", fill=(121, 182, 156))

        single_name = f"parity_{int(r*100):03d}_{ts_tag}.png"
        single_path = os.path.join(out_dir, single_name)
        row_canvas.save(single_path, format="PNG")
        row_images.append(row_canvas)

        checkpoint_reports.append({
            "label": labels[idx],
            "ratio": r,
            "t": t_val,
            "frame": frame_no,
            "aspect_ratio": aspect_key,
            "focal_mm": round(fmm_py, 2),
            "fov_deg": round(cfov_py, 2),
            "cam_pos": [round(float(x), 4) for x in cpos_py],
            "cam_target": [round(float(x), 4) for x in ctar_py],
            "cam_pos_error_mm": round(cam_pos_err * 1000.0, 3),
            "cam_target_error_mm": round(cam_tar_err * 1000.0, 3),
            "fov_error_deg": round(fov_err, 4),
            "max_char_pos_error_mm": round(max_char_pos_err * 1000.0, 3),
            "max_joint_error_mm": round(max_joint_err * 1000.0, 3),
            "constraint_warnings": len(warnings_py),
            "image_url": f"/exports/comparisons/{single_name}",
            "image_path": single_path,
        })

    # 5. 拼接 5 帧总表 PNG
    top_banner_h = 46
    total_w = row_images[0].width
    total_h = top_banner_h + sum(im.height for im in row_images)
    sheet = Image.new("RGB", (total_w, total_h), (20, 22, 25))
    d_sheet = ImageDraw.Draw(sheet)
    d_sheet.text(
        (14, 14),
        f"Director Stage 5-Frame Parity Verification (0% / 25% / 50% / 75% / 100%) - Browser Preview vs Python Export ({width}x{height}, {aspect_key})",
        fill=(231, 235, 240),
    )
    cur_y = top_banner_h
    for im in row_images:
        sheet.paste(im, (0, cur_y))
        cur_y += im.height

    sheet_name = f"parity_5frames_{ts_tag}.png"
    sheet_path = os.path.join(out_dir, sheet_name)
    sheet.save(sheet_path, format="PNG")
    # 同时保存一份固定路径方便查看
    latest_path = os.path.join(out_dir, "parity_5frames_latest.png")
    sheet.save(latest_path, format="PNG")

    return {
        "sheet_path": sheet_path,
        "latest_path": latest_path,
        "sheet_url": f"/exports/comparisons/{sheet_name}",
        "latest_url": "/exports/comparisons/parity_5frames_latest.png",
        "checkpoints": checkpoint_reports,
        "all_passed": all(
            c["cam_pos_error_mm"] < 1.0 and c["max_char_pos_error_mm"] < 1.0 and c["max_joint_error_mm"] < 1.0
            for c in checkpoint_reports
        ),
    }



# ============================================================================
# 4.5 参考视频元数据探测、缩略图生成、双帧率时间映射与“原片＋白模”对照图导出
# ============================================================================

def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return max(1, a)


def format_aspect_ratio_label(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "16:9 (1.78:1)"
    ratio = float(width) / float(height)
    if abs(ratio - 1280.0 / 544.0) < 0.02:
        return f"2.35:1 宽银幕 ({ratio:.2f}:1, {width}×{height})"
    if abs(ratio - 2.39) < 0.08 or abs(ratio - 1680.0 / 720.0) < 0.04:
        return f"2.39:1 宽银幕 ({ratio:.2f}:1, {width}×{height})"
    if abs(ratio - 16.0 / 9.0) < 0.04:
        return f"16:9 ({ratio:.2f}:1, {width}×{height})"
    if abs(ratio - 9.0 / 16.0) < 0.04:
        return f"9:16 ({ratio:.2f}:1, {width}×{height})"
    if abs(ratio - 4.0 / 3.0) < 0.04:
        return f"4:3 ({ratio:.2f}:1, {width}×{height})"
    if abs(ratio - 1.0) < 0.04:
        return f"1:1 ({ratio:.2f}:1, {width}×{height})"
    g = _gcd(width, height)
    return f"{width // g}:{height // g} ({ratio:.2f}:1, {width}×{height})"


def map_aspect_ratio_key(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "2.39:1"
    ratio = float(width) / float(height)
    candidates = [
        ("2.39:1", 1148.0 / 480.0),
        ("2.35:1", 1280.0 / 544.0),
        ("16:9", 16.0 / 9.0),
        ("9:16", 9.0 / 16.0),
        ("4:3", 4.0 / 3.0),
        ("1:1", 1.0),
    ]
    best_key = min(candidates, key=lambda item: abs(ratio - item[1]))[0]
    return best_key


def probe_video_metadata(video_path: str) -> Dict[str, Any]:
    """使用内置 ffmpeg 读取参考视频的真实视频流尺寸、画幅、视频帧数、帧率与视频流时长（忽略音频尾部长度）"""
    import re
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"参考视频文件不存在: {video_path}")
    # 先测试本进程是否有读取权限（例如 macOS TCC 沙盒容器目录权限检测）
    with open(video_path, "rb") as f:
        f.read(64)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run([ffmpeg_exe, "-i", video_path], capture_output=True, text=True)
    stderr = proc.stderr or ""

    # 解析容器时长作为后备
    dur_sec = 10.0
    m_dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    if m_dur:
        hh, mm, ss = int(m_dur.group(1)), int(m_dur.group(2)), float(m_dur.group(3))
        dur_sec = round(hh * 3600.0 + mm * 60.0 + ss, 3)

    # 解析分辨率与帧率 Stream #0:0...: Video: ..., 1280x544, ..., 24 fps
    width, height = 1680, 720
    fps = 24.0
    for line in stderr.splitlines():
        if "Video:" in line:
            m_res = re.search(r",\s*(\d{2,5})x(\d{2,5})(?:[\s,\[])", line)
            if m_res:
                width, height = int(m_res.group(1)), int(m_res.group(2))
            m_fps = re.search(r"(\d+(?:\.\d+)?)\s*fps", line)
            if m_fps:
                fps = float(m_fps.group(1))
            else:
                m_tbr = re.search(r"(\d+(?:\.\d+)?)\s*tbr", line)
                if m_tbr:
                    fps = float(m_tbr.group(1))
            break

    # 使用 -map 0:v:0 -c copy -f null - 快速统计视频流的真实总帧数（不解码像素，耗时 < 20ms，排除音频尾部误差）
    total_frames = max(1, int(round(dur_sec * fps)))
    try:
        proc_v = subprocess.run(
            [ffmpeg_exe, "-v", "info", "-i", video_path, "-map", "0:v:0", "-c", "copy", "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        m_frames = re.findall(r"frame=\s*(\d+)", proc_v.stderr or "")
        if m_frames:
            v_frames = int(m_frames[-1])
            if v_frames > 0:
                total_frames = v_frames
                dur_sec = round(float(total_frames) / max(1.0, float(fps)), 3)
    except Exception:
        pass

    aspect_num = round(float(width) / max(1.0, float(height)), 4)
    return {
        "path": video_path,
        "filename": os.path.basename(video_path),
        "width": width,
        "height": height,
        "aspect_num": aspect_num,
        "aspect_label": format_aspect_ratio_label(width, height),
        "matched_aspect_key": map_aspect_ratio_key(width, height),
        "duration": dur_sec,
        "fps": round(fps, 3),
        "total_frames": total_frames,
    }


def extract_video_frame_pil(video_path: str, t_sec: float, target_w: Optional[int] = None, target_h: Optional[int] = None) -> Image.Image:
    """从本地参考视频精确提取指定秒数 t_sec 的帧画面"""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    t_clamped = max(0.0, float(t_sec))
    cmd = [
        ffmpeg_exe,
        "-ss", f"{t_clamped:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "png",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        # 若在视频最末尾越界，回退 0.08s 重试
        t_fallback = max(0.0, t_clamped - 0.08)
        cmd[2] = f"{t_fallback:.3f}"
        proc = subprocess.run(cmd, capture_output=True)
    if proc.stdout:
        img = Image.open(io.BytesIO(proc.stdout)).convert("RGB")
    else:
        img = Image.new("RGB", (target_w or 960, target_h or 540), (18, 20, 26))
    if target_w and target_h:
        img = fit_image_to_aspect_box(img, target_w, target_h)
    return img


def fit_image_to_aspect_box(img: Image.Image, box_w: int, box_h: int, bg_rgb: Tuple[int, int, int] = (14, 15, 19)) -> Image.Image:
    """按相同画幅等比例缩放至目标框内，绝不拉伸变形"""
    src_w, src_h = img.size
    if src_w <= 0 or src_h <= 0:
        return Image.new("RGB", (box_w, box_h), bg_rgb)
    if src_w == box_w and src_h == box_h:
        return img
    scale = min(float(box_w) / float(src_w), float(box_h) / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (box_w, box_h), bg_rgb)
    ox = (box_w - new_w) // 2
    oy = (box_h - new_h) // 2
    canvas.paste(resized, (ox, oy))
    return canvas


def extract_video_thumbnails(video_path: str, count: int = 11) -> List[Dict[str, Any]]:
    """自动生成参考视频等间距缩略图序列（返回 base64 DataURL、时间戳与对应参考帧号）"""
    meta = probe_video_metadata(video_path)
    dur = float(meta["duration"])
    fps = float(meta["fps"])
    thumbs: List[Dict[str, Any]] = []
    n = max(2, min(24, int(count)))
    for i in range(n):
        t_val = round((dur * i) / (n - 1), 2) if n > 1 else 0.0
        t_seek = min(max(0.0, dur - 0.05), t_val)
        frame_img = extract_video_frame_pil(video_path, t_seek)
        thumb_h = 68
        thumb_w = max(68, int(round(thumb_h * (meta["width"] / max(1, meta["height"])))))
        thumb_img = frame_img.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        thumb_img.save(buf, format="JPEG", quality=78)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        thumbs.append({
            "index": i,
            "t": t_val,
            "frame": int(round(t_val * fps)),
            "data_url": f"data:image/jpeg;base64,{b64}",
        })
    return thumbs


def format_timecode_sec(t_sec: float, fps: float = 24.0) -> str:
    t_pos = max(0.0, float(t_sec))
    total_int_sec = int(math.floor(t_pos + 1e-6))
    mm = (total_int_sec // 60) % 60
    ss = total_int_sec % 60
    ff = int(round((t_pos - total_int_sec) * max(1.0, float(fps)))) % max(1, int(round(fps)))
    return f"00:{mm:02d}:{ss:02d}:{ff:02d}"


def render_comparison_image(
    scene: Dict[str, Any],
    t_proj: float,
    video_path: Optional[str],
    out_path: str,
    compare_mode: str = "side_by_side",
    overlay_alpha: float = 0.5,
    wipe_ratio: float = 0.5,
) -> Dict[str, Any]:
    """
    渲染指定工程时间 t_proj 的“原片＋白模”对照图，严格按相同画幅等比例显示（不拉伸），
    并区分工程帧率与参考片帧率、支持起始时间偏移，在画面上标注时间码与帧号。
    """
    proj_fps = float(scene.get("fps", 24.0))
    proj_dur = float(scene.get("duration", 10.0))
    proj_total_frames = max(1, int(round(proj_dur * proj_fps)))
    proj_frame = max(0, min(proj_total_frames - 1, int(round(float(t_proj) * proj_fps))))
    t_proj = proj_frame / max(1.0, proj_fps)
    ref_cfg = scene.get("ref_video", {}) or {}
    time_offset = float(ref_cfg.get("time_offset", 0.0))

    # 读取参考片元数据与当前映射时间 t_ref
    ref_fps = float(ref_cfg.get("fps", 24.0))
    ref_w = int(ref_cfg.get("width", 1148))
    ref_h = int(ref_cfg.get("height", 480))
    ref_dur = float(ref_cfg.get("duration", proj_dur))
    ref_total_frames = max(1, int(ref_cfg.get("frame_count") or round(ref_dur * ref_fps)))
    if video_path and os.path.exists(video_path):
        try:
            meta = probe_video_metadata(video_path)
            ref_fps = float(meta["fps"])
            ref_w = int(meta["width"])
            ref_h = int(meta["height"])
            ref_dur = float(meta["duration"])
            ref_total_frames = max(1, int(meta.get("frame_count") or round(ref_dur * ref_fps)))
        except Exception:
            pass

    # 按时间映射计算参考片精确采样时刻 t_ref 与参考片帧号 ref_frame（末帧严格不越界）
    ref_frame = max(0, min(ref_total_frames - 1, int(round((float(t_proj) + time_offset) * ref_fps))))
    t_ref = ref_frame / max(1.0, ref_fps)
    t_ref_quantized = t_ref

    # 统一单侧面板像素宽高（严格等比例匹配参考片画幅，高度固定为 480px，宽度按比例计算）
    pane_h = 480
    aspect_num = float(ref_w) / max(1.0, float(ref_h))
    pane_w = max(320, int(round(pane_h * aspect_num)))
    if pane_w % 2 == 1:
        pane_w += 1

    # 1. 渲染右侧 3D 白模帧 (相同画幅尺寸 pane_w x pane_h)
    white_img = render_single_frame(scene, float(t_proj), width=pane_w, height=pane_h)

    # 2. 提取左侧原片参考视频帧 (相同画幅尺寸 pane_w x pane_h)
    if video_path and os.path.exists(video_path):
        ref_img = extract_video_frame_pil(video_path, t_ref_quantized, target_w=pane_w, target_h=pane_h)
    else:
        ref_img = Image.new("RGB", (pane_w, pane_h), (20, 22, 28))
        d_empty = ImageDraw.Draw(ref_img)
        d_empty.text((24, pane_h // 2 - 10), "No Local Reference Video Loaded", fill=(160, 170, 185))

    header_h = 44
    footer_h = 34
    cpos, ctar, cfov = interpolate_camera(scene, float(t_proj))
    f_mm = fov_to_focal_mm(cfov)
    proj_tc = format_timecode_sec(t_proj, proj_fps)
    ref_tc = format_timecode_sec(t_ref, ref_fps)

    if compare_mode == "overlay":
        alpha = max(0.0, min(1.0, float(overlay_alpha)))
        blended = Image.blend(white_img, ref_img, alpha)
        out_w = pane_w
        out_h = pane_h + header_h + footer_h
        comp = Image.new("RGB", (out_w, out_h), (15, 17, 22))
        comp.paste(blended, (0, header_h))
    elif compare_mode == "wipe":
        ratio = max(0.0, min(1.0, float(wipe_ratio)))
        split_x = int(round(pane_w * ratio))
        merged = white_img.copy()
        if split_x > 0:
            merged.paste(ref_img.crop((0, 0, split_x, pane_h)), (0, 0))
        d_m = ImageDraw.Draw(merged)
        d_m.line([(split_x, 0), (split_x, pane_h)], fill=(251, 191, 36), width=2)
        out_w = pane_w
        out_h = pane_h + header_h + footer_h
        comp = Image.new("RGB", (out_w, out_h), (15, 17, 22))
        comp.paste(merged, (0, header_h))
    else:
        # 默认并排对照 (Side-by-Side): 左原片 + 右白模，中间 6px 分隔线
        gap = 6
        out_w = pane_w * 2 + gap
        out_h = pane_h + header_h + footer_h
        comp = Image.new("RGB", (out_w, out_h), (15, 17, 22))
        comp.paste(ref_img, (0, header_h))
        comp.paste(white_img, (pane_w + gap, header_h))
        d_sep = ImageDraw.Draw(comp)
        d_sep.rectangle([pane_w, header_h, pane_w + gap - 1, header_h + pane_h], fill=(45, 52, 64))

    draw = ImageDraw.Draw(comp)
    # 顶部时间码与帧号标注栏
    draw.rectangle([0, 0, comp.width, header_h - 1], fill=(20, 23, 30))
    left_title = f"[REF VIDEO]  t={t_ref:.2f}s | TC {ref_tc} | Frame #{ref_frame} ({ref_fps:g}fps) | {ref_w}x{ref_h} | Offset {time_offset:+.2f}s"
    right_title = f"[3D BLOCKOUT]  t={float(t_proj):.2f}s | TC {proj_tc} | Frame #{proj_frame} ({proj_fps:g}fps) | f={f_mm:.1f}mm (VFOV {cfov:.1f} deg)"
    draw.text((14, 14), left_title, fill=(251, 191, 36))
    if compare_mode == "side_by_side":
        draw.text((pane_w + 18, 14), right_title, fill=(96, 165, 250))
    else:
        draw.text((comp.width // 2 + 10, 14), right_title, fill=(96, 165, 250))

    # 底部相机参数与角色坐标标注栏
    y_foot = header_h + pane_h + 9
    cam_summary = f"Cam Pos=[{cpos[0]:.2f}, {cpos[1]:.2f}, {cpos[2]:.2f}]m  Target=[{ctar[0]:.2f}, {ctar[1]:.2f}, {ctar[2]:.2f}]m  Aspect={pane_w}:{pane_h} ({aspect_num:.2f}:1)"
    draw.text((14, y_foot), cam_summary, fill=(180, 190, 205))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    comp.save(out_path, format="PNG")
    return {
        "output_path": out_path,
        "t_proj": round(float(t_proj), 3),
        "proj_frame": proj_frame,
        "proj_fps": proj_fps,
        "t_ref": round(t_ref, 3),
        "ref_frame": ref_frame,
        "ref_fps": ref_fps,
        "width": comp.width,
        "height": comp.height,
    }


def render_multi_time_comparisons(
    scene: Dict[str, Any],
    times: List[float],
    video_path: Optional[str],
    out_dir: str,
    prefix: str = "compare",
) -> Dict[str, Any]:
    """按需导出指定时间点（默认首帧＋末帧，末帧按实际最后一帧采样不越界）的并排对照图及汇总图"""
    os.makedirs(out_dir, exist_ok=True)
    proj_fps = max(1.0, float(scene.get("fps", 24.0)))
    proj_dur = max(0.1, float(scene.get("duration", 10.0)))
    last_frame_idx = max(0, int(round(proj_dur * proj_fps)) - 1)
    max_valid_t = last_frame_idx / proj_fps
    if not times:
        clean_times = [0.0, round(max_valid_t, 4)]
    else:
        clean_times = [round(max(0.0, min(max_valid_t, float(t))), 4) for t in times]

    items = []
    pil_rows: List[Image.Image] = []
    ts_tag = int(time.time() * 1000)
    for idx, t_val in enumerate(clean_times):
        t_str = f"{t_val:.1f}s".replace(".", "_")
        fname = f"{prefix}_t{t_str}_{ts_tag}.png"
        fpath = os.path.join(out_dir, fname)
        info = render_comparison_image(scene, t_val, video_path, fpath, compare_mode="side_by_side")
        info["filename"] = fname
        info["url"] = f"/exports/comparisons/{fname}"
        items.append(info)
        pil_rows.append(Image.open(fpath).convert("RGB"))

    # 拼接汇总对照大图 (Contact Sheet)
    sheet_path = ""
    sheet_url = ""
    if pil_rows:
        row_w, row_h = pil_rows[0].size
        gap_y = 8
        sheet_h = len(pil_rows) * row_h + max(0, len(pil_rows) - 1) * gap_y
        sheet_img = Image.new("RGB", (row_w, sheet_h), (10, 12, 16))
        for r_idx, r_im in enumerate(pil_rows):
            sheet_img.paste(r_im, (0, r_idx * (row_h + gap_y)))
        sheet_fname = f"{prefix}_sheet_{'_'.join(str(int(t)) if float(t).is_integer() else str(t) for t in clean_times)}s_{ts_tag}.png"
        sheet_path = os.path.join(out_dir, sheet_fname)
        sheet_img.save(sheet_path, format="PNG")
        sheet_url = f"/exports/comparisons/{sheet_fname}"

    return {
        "items": items,
        "sheet_path": sheet_path,
        "sheet_url": sheet_url,
        "times": clean_times,
    }


# ============================================================================
# 5. Python 图像转魔法人形师 IK 控制球坐标 (OpenPose 颜色检测 + 轮廓重心推断)
# ============================================================================

def analyze_image_to_ik_targets(image_b64: str, filename: str = "") -> Dict[str, Any]:
    """
    当用户上传参考图时（若前端 MediaPipe 3D Pose 未检测到人体关键点，转交后端检测是否为纯黑底 OpenPose 骨骼图）：
    严格遵守规范：
    - 绝不按文件名猜测姿态，绝不在识别失败时静默套用默认坐姿！
    - 明确返回识别来源 (source)、成功/失败状态 (status / recognized) 与原因 (reason)，允许用户在 UI 手动改姿态。
    """
    try:
        raw_b64 = image_b64.split(",", 1)[1] if "," in (image_b64 or "") else (image_b64 or "")
        if not raw_b64:
            return {
                "recognized": False,
                "status": "failed",
                "source": "python_openpose_color_detector",
                "reason": "上传的图像数据为空，未执行任何静默默认姿态覆盖。",
                "targets": None,
                "available_presets": list(PRESET_IK_POSES.keys()),
            }
        img_bytes = base64.b64decode(raw_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.asarray(pil_img, dtype=np.float32)
        h, w, _ = arr.shape

        # 检测是否为标准深色底 OpenPose 骨骼图 (背景超过 75% 像素亮度 < 35，且存在至少 4 种高饱和 OpenPose 骨骼色块)
        luminance = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        dark_ratio = float(np.mean(luminance < 35.0))
        if dark_ratio < 0.72:
            return {
                "recognized": False,
                "status": "failed",
                "source": "mediapipe_pose + python_openpose_detector",
                "reason": "前端 MediaPipe 未检出人体关键点，且后端检测该图不是标准黑底 OpenPose 骨骼图（已禁止按文件名或默认坐姿静默套用）。请使用下方手动姿态预设或 IK 控制球调整。",
                "targets": None,
                "available_presets": list(PRESET_IK_POSES.keys()),
            }

        # 检测 OpenPose 关键颜色区域的归一化重心:
        # 头颈红粉 (255, 0, 85), 左腕黄绿 (180, 255, 0), 右腕亮黄 (255, 230, 0), 左踝青色 (0, 255, 255), 右踝蓝紫 (85, 0, 255)
        def _find_color_centroid(target_rgb: Tuple[float, float, float], tol: float = 65.0) -> Optional[Tuple[float, float]]:
            diff = np.linalg.norm(arr - np.array(target_rgb, dtype=np.float32), axis=2)
            ys, xs = np.where(diff < tol)
            if len(xs) < 12:
                return None
            return float(np.mean(xs)) / max(1.0, float(w)), float(np.mean(ys)) / max(1.0, float(h))

        c_head = _find_color_centroid((255, 0, 85))
        c_pelvis = _find_color_centroid((0, 180, 255))
        c_hand_l = _find_color_centroid((180, 255, 0))
        c_hand_r = _find_color_centroid((255, 230, 0))
        c_foot_l = _find_color_centroid((0, 255, 255))
        c_foot_r = _find_color_centroid((85, 0, 255))

        found_count = sum(1 for c in (c_head, c_pelvis, c_hand_l, c_hand_r, c_foot_l, c_foot_r) if c is not None)
        if found_count < 4 or c_pelvis is None or c_head is None:
            return {
                "recognized": False,
                "status": "failed",
                "source": "python_openpose_color_detector",
                "reason": f"仅检出 {found_count}/6 个 OpenPose 关键色块，不足以重建 3D 姿态（已禁止按文件名或默认坐姿静默套用）。请手动选择姿态。",
                "targets": None,
                "available_presets": list(PRESET_IK_POSES.keys()),
            }

        # 由检测到的 2D 骨骼重心反推局部 3D IK 控制球坐标
        px, py = c_pelvis
        body_span_y = max(0.22, ((c_foot_l[1] if c_foot_l else 0.85) - c_head[1]))
        scale_2d_to_m = 1.50 / body_span_y

        def _uv_to_local(uv: Optional[Tuple[float, float]], default_xyz: List[float], z_val: float = 0.05) -> List[float]:
            if uv is None:
                return list(default_xyz)
            u, v = uv
            lx = -(u - px) * scale_2d_to_m * 0.85
            ly = max(0.04, 0.86 - (v - py) * scale_2d_to_m)
            return [round(float(lx), 3), round(float(ly), 3), round(float(z_val), 3)]

        base_t = {k: list(v) for k, v in PRESET_IK_POSES["standing"]["targets"].items()}
        base_t["pelvis"] = [0.0, round(max(0.42, min(0.92, 1.65 - py * 1.1)), 3), 0.0]
        base_t["chest"] = [0.0, round(base_t["pelvis"][1] + 0.45, 3), 0.02]
        base_t["head_target"] = _uv_to_local(c_head, base_t["head_target"], 0.65)
        base_t["hand_L"] = _uv_to_local(c_hand_l, base_t["hand_L"], 0.18)
        base_t["hand_R"] = _uv_to_local(c_hand_r, base_t["hand_R"], 0.18)
        base_t["foot_L"] = _uv_to_local(c_foot_l, base_t["foot_L"], 0.04)
        base_t["foot_R"] = _uv_to_local(c_foot_r, base_t["foot_R"], 0.04)

        return {
            "recognized": True,
            "status": "success",
            "pose_name": "openpose_detected",
            "targets": base_t,
            "pins": {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False},
            "source": "python_openpose_color_detector",
            "reason": f"成功从 OpenPose 骨骼图检出 {found_count} 组关节点色块并解算 3D IK 目标",
        }
    except Exception as exc:
        return {
            "recognized": False,
            "status": "failed",
            "source": "python_openpose_color_detector",
            "reason": f"图像解析异常 ({exc})，未执行任何静默默认姿态覆盖。",
            "targets": None,
            "available_presets": list(PRESET_IK_POSES.keys()),
        }


# ============================================================================
# 6. 视频提示词自动掰人偶 + 摆道具 + 设运镜引擎 (Prompt-to-3D-Pose & Stage)
# ============================================================================

def apply_semantic_ik_modifiers(char: CharacterObj, text: str):
    """
    根据自然语言提示词中的动作细节，对角色首先选定基底姿态，
    再叠加【魔法人形师 IK 控制球】的局部肢体微调（手部、腿部、躯干俯仰、头部视线）。
    """
    t = text.lower()

    # 1. 选择基底姿势 (Base Pose)
    if any(w in t for w in ["二郎腿", "翘腿", "瘫坐", "懒散坐"]):
        char.set_pose("sitting_relaxed")
    elif any(w in t for w in ["托腮", "思考", "沉思", "托着下巴"]) and any(w in t for w in ["坐", "椅", "桌", "凳"]):
        char.set_pose("sitting_think")
    elif any(w in t for w in ["坐", "椅", "凳", "沙发", "床沿", "sit"]):
        char.set_pose("sitting")
    elif any(w in t for w in ["跪", "单膝", "kneel"]):
        char.set_pose("kneeling")
    elif any(w in t for w in ["跑", "冲刺", "狂奔", "逃", "追", "run"]):
        char.set_pose("running")
    elif any(w in t for w in ["格斗", "战斗", "打架", "搏斗", "出拳", "挥拳", "格挡", "对峙", "fight"]):
        char.set_pose("fighting")
    else:
        char.set_pose("standing")

    targets = {k: list(v) for k, v in char.targets.items()}

    # 2. 特殊全身姿态覆盖（蹲下 / 趴下 / 躺倒）
    if any(w in t for w in ["蹲", "半蹲", "抱头蹲", "squat"]):
        targets["pelvis"] = [0.0, 0.42, -0.06]
        targets["chest"] = [0.0, 0.84, 0.14]
        targets["head_target"] = [0.0, 1.02, 0.65]
        targets["foot_L"] = [0.15, 0.04, 0.12]
        targets["foot_R"] = [-0.15, 0.04, 0.12]
        targets["knee_L_pole"] = [0.24, 0.48, 0.55]
        targets["knee_R_pole"] = [-0.24, 0.48, 0.55]
        targets["hand_L"] = [0.18, 0.52, 0.28]
        targets["hand_R"] = [-0.18, 0.52, 0.28]
    elif any(w in t for w in ["趴", "俯卧", "倒地", "躺"]):
        targets["pelvis"] = [0.0, 0.14, -0.10]
        targets["chest"] = [0.0, 0.18, 0.36]
        targets["head_target"] = [0.0, 0.28, 0.85]
        targets["foot_L"] = [0.14, 0.06, -0.82]
        targets["foot_R"] = [-0.14, 0.06, -0.82]
        targets["hand_L"] = [0.24, 0.05, 0.42]
        targets["hand_R"] = [-0.24, 0.05, 0.42]

    pelvis_y = targets["pelvis"][1]
    chest_y = targets["chest"][1]
    chest_z = targets["chest"][2]

    # 3. 躯干俯仰微调 (弯腰前倾 / 后仰)
    if any(w in t for w in ["弯腰", "前倾", "俯身", "鞠躬", "低姿"]):
        targets["chest"][1] = round(chest_y - 0.08, 3)
        targets["chest"][2] = round(chest_z + 0.22, 3)
        chest_y, chest_z = targets["chest"][1], targets["chest"][2]
    elif any(w in t for w in ["后仰", "往后靠"]):
        targets["chest"][2] = round(chest_z - 0.18, 3)
        chest_z = targets["chest"][2]

    # 4. 手部组合式 IK 微调
    if any(w in t for w in ["举起双手", "双手高举", "投降", "欢呼", "高举双臂", "伸懒腰"]):
        targets["hand_L"] = [0.26, round(chest_y + 0.38, 3), round(chest_z + 0.06, 3)]
        targets["hand_R"] = [-0.26, round(chest_y + 0.38, 3), round(chest_z + 0.06, 3)]
        targets["elbow_L_pole"] = [0.42, round(chest_y + 0.12, 3), -0.10]
        targets["elbow_R_pole"] = [-0.42, round(chest_y + 0.12, 3), -0.10]
    elif any(w in t for w in ["张开双臂", "摊手", "双手摊开", "拥抱"]):
        targets["hand_L"] = [0.58, round(chest_y - 0.04, 3), round(chest_z + 0.18, 3)]
        targets["hand_R"] = [-0.58, round(chest_y - 0.04, 3), round(chest_z + 0.18, 3)]
    elif any(w in t for w in ["抱头", "捂脸", "掩面", "抱头蹲", "崩溃"]):
        targets["hand_L"] = [0.11, round(chest_y + 0.18, 3), round(chest_z + 0.14, 3)]
        targets["hand_R"] = [-0.11, round(chest_y + 0.18, 3), round(chest_z + 0.14, 3)]
        targets["elbow_L_pole"] = [0.34, round(chest_y - 0.05, 3), 0.25]
        targets["elbow_R_pole"] = [-0.34, round(chest_y - 0.05, 3), 0.25]
    elif any(w in t for w in ["叉腰", "双手叉腰"]):
        targets["hand_L"] = [0.21, round(pelvis_y + 0.09, 3), 0.0]
        targets["hand_R"] = [-0.21, round(pelvis_y + 0.09, 3), 0.0]
        targets["elbow_L_pole"] = [0.45, round(pelvis_y + 0.22, 3), -0.12]
        targets["elbow_R_pole"] = [-0.45, round(pelvis_y + 0.22, 3), -0.12]
    elif any(w in t for w in ["抱胸", "双手抱胸", "交叉胸前"]):
        targets["hand_L"] = [-0.08, round(chest_y - 0.08, 3), round(chest_z + 0.20, 3)]
        targets["hand_R"] = [0.08, round(chest_y - 0.08, 3), round(chest_z + 0.20, 3)]
    elif any(w in t for w in ["背手", "背在身后"]):
        targets["hand_L"] = [0.08, round(pelvis_y + 0.05, 3), -0.16]
        targets["hand_R"] = [-0.08, round(pelvis_y + 0.05, 3), -0.16]
    elif any(w in t for w in ["格挡", "防御", "护住头"]):
        targets["hand_L"] = [0.13, round(chest_y + 0.14, 3), round(chest_z + 0.24, 3)]
        targets["hand_R"] = [-0.13, round(chest_y + 0.14, 3), round(chest_z + 0.24, 3)]
    elif any(w in t for w in ["出拳", "挥拳", "重拳", "直拳", "攻击"]):
        targets["hand_R"] = [-0.04, round(chest_y + 0.02, 3), round(chest_z + 0.48, 3)]
        targets["hand_L"] = [0.16, round(chest_y + 0.04, 3), round(chest_z + 0.20, 3)]
    elif any(w in t for w in ["指着", "指向", "伸手", "拔枪", "举枪", "瞄准"]):
        targets["hand_R"] = [-0.06, round(chest_y + 0.04, 3), round(chest_z + 0.50, 3)]
    elif any(w in t for w in ["撑地", "手撑"]):
        targets["hand_L"] = [0.18, 0.05, 0.28]

    # 单侧手微调
    if any(w in t for w in ["右手举", "举起右手", "右手挥", "挥右手"]):
        targets["hand_R"] = [-0.26, round(chest_y + 0.36, 3), round(chest_z + 0.14, 3)]
    if any(w in t for w in ["左手举", "举起左手", "左手挥", "挥左手"]):
        targets["hand_L"] = [0.26, round(chest_y + 0.36, 3), round(chest_z + 0.14, 3)]
    if any(w in t for w in ["右手托腮", "右手托下巴", "托着下巴"]):
        targets["hand_R"] = [-0.05, round(chest_y + 0.13, 3), round(chest_z + 0.16, 3)]

    # 5. 腿部组合式 IK 微调
    if any(w in t for w in ["踢", "飞踢", "踹", "高抬腿"]):
        targets["foot_R"] = [-0.08, 0.86, 0.52]
        targets["knee_R_pole"] = [-0.10, 0.95, 0.35]

    # 6. 头部视线微调 (head_target)
    head_y = chest_y + 0.24
    head_x = 0.0
    head_z = chest_z + 0.65
    if any(w in t for w in ["抬头", "仰望", "看天", "往上看", "向上看"]):
        head_y += 0.48
    elif any(w in t for w in ["低头", "俯视", "看地", "垂头", "往下看"]):
        head_y -= 0.45
    if any(w in t for w in ["看左", "向左看", "转头看左", "左顾"]):
        head_x = 0.60
    elif any(w in t for w in ["看右", "向右看", "转头看右", "右盼"]):
        head_x = -0.60
    targets["head_target"] = [round(head_x, 3), round(head_y, 3), round(head_z, 3)]

    char.ik(**targets)
    return char


def prompt_to_stage(prompt: str) -> Dict[str, Any]:
    """
    接收一段视频提示词（如“一个男人坐在椅子上右手托腮沉思，面前有张蓝色桌子，镜头从侧面缓慢推近特写”），
    自动生成包含角色 IK 姿态、彩色几何体道具和摄影机运镜关键帧的完整 3D Stage！
    """
    stage = Stage()
    stage.clear()
    p = prompt.strip()
    p_low = p.lower()

    is_two_people = any(w in p_low for w in ["两个人", "两人", "双人", "对峙", "搏斗", "对战", "打架", "一左一右", "一坐一站", "对话", "交谈", "追逐", "追击"])

    # 1. 创建角色与姿态
    if is_two_people:
        if "追" in p_low:
            # 追逐戏：前逃后追
            c1 = stage.add_character("逃跑者_01", pos=[0.0, 0.0, 0.65], rot_y=0.0, pose="running")
            apply_semantic_ik_modifiers(c1, "冲刺奔跑 前倾")
            c2 = stage.add_character("追击者_02", pos=[0.0, 0.0, -0.75], rot_y=0.0, pose="running")
            apply_semantic_ik_modifiers(c2, "冲刺奔跑 伸手")
            c2.color = "#fca5a5"
        elif "一坐一站" in p_low or ("坐" in p_low and "站" in p_low):
            stage.add_box("红色坐凳", pos=[-0.55, 0.22, 0.0], size=[0.46, 0.44, 0.46], color="#ef4444")
            c1 = stage.add_character("坐姿角色_01", pos=[-0.55, 0.0, 0.0], rot_y=65.0, pose="sitting")
            apply_semantic_ik_modifiers(c1, "坐在椅子上 抬头")
            c2 = stage.add_character("站立角色_02", pos=[0.55, 0.0, 0.0], rot_y=-90.0, pose="standing")
            apply_semantic_ik_modifiers(c2, p + " 低头")
            c2.color = "#93c5fd"
        else:
            # 双人面对面对峙 / 格斗 / 交谈
            c1 = stage.add_character("左侧角色_01", pos=[-0.52, 0.0, 0.0], rot_y=90.0)
            c2 = stage.add_character("右侧角色_02", pos=[0.52, 0.0, 0.0], rot_y=-90.0)
            c2.color = "#fca5a5"
            if any(w in p_low for w in ["格斗", "搏斗", "打架", "对战", "出拳", "攻击"]):
                apply_semantic_ik_modifiers(c1, "格斗 出拳 攻击")
                if "踢" in p_low:
                    apply_semantic_ik_modifiers(c1, "格斗 飞踢")
                apply_semantic_ik_modifiers(c2, "格斗 格挡 防御 后仰")
            else:
                apply_semantic_ik_modifiers(c1, p)
                apply_semantic_ik_modifiers(c2, p)
    else:
        # 单人主角
        hero = stage.add_character("魔法人形偶_01", pos=[0.0, 0.0, 0.0], rot_y=0.0)
        apply_semantic_ik_modifiers(hero, p)

        # 若为坐姿则自动在臀部下方生成坐凳
        if hero.pose_name.startswith("sitting"):
            stage.add_box("红色坐凳", pos=[0.0, 0.22, -0.05], size=[0.46, 0.44, 0.46], color="#ef4444")

    # 2. 根据提示词自动摆放彩色几何体道具 (长方体 / 圆锥体 / 三角柱 / 圆柱)
    if any(w in p_low for w in ["桌", "吧台", "台子", "电脑", "办公"]):
        stage.add_box("蓝色长桌", pos=[0.0, 0.35, 0.72], size=[1.15, 0.70, 0.46], color="#3b82f6")
    if any(w in p_low for w in ["树", "森林", "雪林", "针叶", "路障", "锥"]):
        stage.add_tree_cone("左侧针叶树锥", x=-1.45, z=-0.65, width=1.20, height=4.40, obj_id="tree_prompt_L")
        stage.add_tree_cone("右侧针叶树锥", x=1.45, z=-0.95, width=1.30, height=4.80, obj_id="tree_prompt_R")
    if any(w in p_low for w in ["碑", "墙", "门", "高塔", "大楼", "巨石", "掩体", "箱"]):
        stage.add_box("高大方碑", pos=[0.0, 0.95, 1.45], size=[0.85, 1.90, 0.35], color="#9A8B7A")
    if any(w in p_low for w in ["斜坡", "山坡", "屋顶", "三角", "悬崖", "岩石"]):
        stage.add_prism("三角斜坡", pos=[1.15, 0.35, 0.15], size=[0.75, 0.70, 0.90], color="#879484")
    if any(w in p_low for w in ["柱", "油桶", "立柱"]):
        stage.add_cylinder("圆柱体块", pos=[-1.10, 0.65, -0.25], size=[0.45, 1.30, 0.45], color="#7E8B9B")

    # 如果提示词没有提到任何特定道具且不是坐姿，默认放两个单锥体参照树丰富空间感
    if len(stage.objects) == (2 if is_two_people else 1):
        stage.add_tree_cone("左侧参照树锥", x=-1.35, z=-0.65, width=1.15, height=4.20, obj_id="tree_ref_L")
        stage.add_tree_cone("右侧参照树锥", x=1.35, z=-0.85, width=1.25, height=4.50, obj_id="tree_ref_R")

    # 3. 根据提示词自动计算摄影机景别与 0s -> 4s 运镜关键帧
    # 默认目标点高度
    first_char = next((o for o in stage.objects if o.type == "character"), None)
    focus_y = 0.72 if (first_char and first_char.pose_name.startswith("sitting")) else 0.98
    if "特写" in p_low or "近景" in p_low:
        focus_y += 0.18

    # 判断机位高度 (仰拍 vs 俯拍 vs 平视)
    if any(w in p_low for w in ["仰拍", "低角度", "低机位"]):
        cam_y0, cam_y1 = 0.42, 0.38
    elif any(w in p_low for w in ["俯拍", "高机位", "上帝视角", "鸟瞰"]):
        cam_y0, cam_y1 = 2.65, 2.35
    else:
        cam_y0, cam_y1 = focus_y + 0.45, focus_y + 0.35

    # 判断镜头焦距 (支持 16mm / 24mm / 35mm / 50mm / 85mm / 135mm 或 广角/长焦/特写)
    import re
    mm_match = re.search(r"(\d{2,3})\s*mm", p_low)
    if mm_match:
        base_mm = float(max(14, min(200, int(mm_match.group(1)))))
    elif any(w in p_low for w in ["超广角", "16mm"]):
        base_mm = 16.0
    elif any(w in p_low for w in ["广角", "大全景", "24mm"]):
        base_mm = 24.0
    elif any(w in p_low for w in ["长焦", "望远", "压缩感"]):
        base_mm = 85.0
    elif any(w in p_low for w in ["特写", "近景"]):
        base_mm = 50.0
    else:
        base_mm = 35.0

    # 判断运镜轨迹 (推近 / 拉远 / 环绕 / 侧面横移)
    if any(w in p_low for w in ["推", "推近", "推进", "特写"]):
        end_mm = min(135.0, max(base_mm, 50.0))
        stage.add_camera_keyframe(0.0, pos=[1.85, cam_y0, 2.85], target=[0.0, focus_y, 0.05], focal_mm=base_mm)
        stage.add_camera_keyframe(4.0, pos=[0.95, cam_y1, 1.65], target=[0.0, focus_y + 0.08, 0.05], focal_mm=end_mm)
        stage.set_camera(pos=[1.85, cam_y0, 2.85], target=[0.0, focus_y, 0.05], focal_mm=base_mm)
    elif any(w in p_low for w in ["拉", "拉远", "后退", "远景", "全景"]):
        stage.add_camera_keyframe(0.0, pos=[1.05, cam_y1, 1.75], target=[0.0, focus_y, 0.05], focal_mm=max(35.0, base_mm))
        stage.add_camera_keyframe(4.0, pos=[2.55, cam_y0 + 0.35, 3.45], target=[0.0, focus_y, 0.0], focal_mm=min(24.0, base_mm))
        stage.set_camera(pos=[1.05, cam_y1, 1.75], target=[0.0, focus_y, 0.05], focal_mm=max(35.0, base_mm))
    elif any(w in p_low for w in ["侧面", "横移", "跟拍", "侧拍"]):
        stage.add_camera_keyframe(0.0, pos=[2.55, cam_y0, 0.85], target=[0.0, focus_y, 0.10], focal_mm=base_mm)
        stage.add_camera_keyframe(4.0, pos=[2.35, cam_y1, -0.95], target=[0.0, focus_y, -0.10], focal_mm=base_mm)
        stage.set_camera(pos=[2.55, cam_y0, 0.85], target=[0.0, focus_y, 0.10], focal_mm=base_mm)
    else:
        # 默认电影感环绕运镜
        stage.add_camera_keyframe(0.0, pos=[2.15, cam_y0, 2.65], target=[0.0, focus_y, 0.05], focal_mm=base_mm)
        stage.add_camera_keyframe(4.0, pos=[-1.65, cam_y1, 2.35], target=[0.0, focus_y, 0.05], focal_mm=base_mm)
        stage.set_camera(pos=[2.15, cam_y0, 2.65], target=[0.0, focus_y, 0.05], focal_mm=base_mm)

    scene_dict = convert_scene_to_su_minimal(stage.to_dict())
    py_code = f"# ✨ 由视频提示词自动生成 (SU 极简彩色预演 su_minimal)：\"{p}\"\n" + scene_dict_to_python_code(scene_dict)
    return {
        "scene": scene_dict,
        "python_code": py_code,
        "prompt": p,
    }


# ============================================================================
# 7. 参考视频空间复现预演场次：雪林道路车队 (Shot 1) & 吉普车内四人聚焦后排 (Shot 2)
# ============================================================================

@sparse_generated_scene
def build_convoy_and_cabin_previz_shot(shot_mode: str = "shot1") -> Dict[str, Any]:
    """
    根据参考片 91d8e0a37359f9fd0afd7866042f57de.mp4 (1280x544, 24fps, 241帧 = 10.04s)
    构建可编辑极简白模动画：
    - shot_mode == "shot1": 第 1 镜：雪林道路与车队 (F000..F105, 106帧 = 4.42s)
    - shot_mode == "shot2": 第 2 镜：吉普车内四人（聚焦后排）(F106..F240, 135帧 = 5.62s)
    统一使用：
    - 人物：球头人偶 (body_style="pawn")，保留跨镜头稳定 ID 与身份色
    - 场景与车辆：SketchUp 灰白体块＋黑色细结构边 (visual_style="sketchup_massing")，真实留空车窗
    """
    stage = Stage()
    stage.clear()
    is_shot1 = (str(shot_mode).lower() not in ("shot2", "2"))
    stage.fps = 24
    stage.duration = 4.42 if is_shot1 else 5.62
    stage.render_mode = "abstract_block"
    stage.global_body_style = "pawn"
    stage.visual_style = "sketchup_massing"
    stage.environment = {
        "sky": "#D6DCE4" if is_shot1 else "#CCD3DC",
        "ground": "#C7CDD6",
        "fog": True,
        "fog_color": "#D6DCE4" if is_shot1 else "#CCD3DC",
        "fog_near": 12.0 if is_shot1 else 6.5,
        "fog_far": 52.0 if is_shot1 else 28.0,
        "fog_max": 0.82 if is_shot1 else 0.75,
        "show_ground": True,
        "show_grid": False,
    }

    # -------------------------------------------------------------------------
    # 1. 雪林弯曲道路、两侧雪坡与关键轮廓/遮挡林木（复用同一套空间资产）
    # -------------------------------------------------------------------------
    road_segments = [
        ("雪林弯曲道路_远段", [-8.20, 0.03, -32.00], [0.0, 14.0, 0.0], [3.60, 0.06, 13.00]),
        ("雪林弯曲道路_中远段", [-5.40, 0.03, -20.50], [0.0, 18.0, 0.0], [3.60, 0.06, 11.50]),
        ("雪林弯曲道路_中段", [-2.40, 0.03, -10.00], [0.0, 19.0, 0.0], [3.60, 0.06, 11.00]),
        ("雪林弯曲道路_近段", [0.20, 0.03, -0.50], [0.0, 14.0, 0.0], [3.70, 0.06, 10.00]),
        ("雪林弯曲道路_前段", [1.60, 0.03, 7.80], [0.0, 6.0, 0.0], [3.80, 0.06, 8.50]),
    ]
    for r_idx, (r_name, r_pos, r_rot, r_size) in enumerate(road_segments, 1):
        seg = stage.add_box(r_name, pos=r_pos, rot=r_rot, size=r_size, color="#B2B7C0")
        seg.id = f"env_road_seg_{r_idx}"
        seg.extra["arch_role"] = "floor"
        seg.extra["ground_decal"] = True

    # 车辙轨迹压痕（辅助表现道路弯曲走势与透视纵深）
    rut_segments = [
        ("左车辙_远段", [-6.00, 0.05, -20.50], [0.0, 18.0, 0.0], [0.30, 0.04, 11.00]),
        ("右车辙_远段", [-4.80, 0.05, -20.50], [0.0, 18.0, 0.0], [0.30, 0.04, 11.00]),
        ("左车辙_中段", [-3.00, 0.05, -10.00], [0.0, 19.0, 0.0], [0.30, 0.04, 10.40]),
        ("右车辙_中段", [-1.80, 0.05, -10.00], [0.0, 19.0, 0.0], [0.30, 0.04, 10.40]),
        ("左车辙_近段", [-0.40, 0.05, -0.50], [0.0, 14.0, 0.0], [0.30, 0.04, 9.40]),
        ("右车辙_近段", [0.80, 0.05, -0.50], [0.0, 14.0, 0.0], [0.30, 0.04, 9.40]),
    ]
    for u_idx, (u_name, u_pos, u_rot, u_size) in enumerate(rut_segments, 1):
        rut = stage.add_box(u_name, pos=u_pos, rot=u_rot, size=u_size, color="#989EA8")
        rut.id = f"env_rut_{u_idx}"
        rut.extra["arch_role"] = "floor"
        rut.extra["ground_decal"] = True

    # 道路两侧雪坡体块
    snowbanks = [
        ("左侧雪坡_远段", [-9.80, 0.42, -19.50], [0.0, 18.0, 0.0], [4.60, 0.84, 16.50], "#E1E5EC"),
        ("左侧雪坡_近段", [-4.80, 0.36, -3.00], [0.0, 15.0, 0.0], [4.20, 0.72, 17.00], "#E4E8EE"),
        ("右侧雪坡_远段", [-0.60, 0.44, -18.50], [0.0, 18.0, 0.0], [4.80, 0.88, 16.50], "#DFE3EA"),
        ("右侧雪坡_近段", [4.40, 0.38, -2.00], [0.0, 14.0, 0.0], [4.60, 0.76, 17.00], "#E2E6ED"),
    ]
    for b_idx, (b_name, b_pos, b_rot, b_size, b_col) in enumerate(snowbanks, 1):
        sb = stage.add_prism(b_name, pos=b_pos, rot=b_rot, size=b_size, color=b_col)
        sb.id = f"env_snowbank_{b_idx}"
        sb.extra["arch_role"] = "prop"

    # 倒木（使用 Rx=88° 让圆柱横卧在雪坡地面上，Ry 控制水平朝向，避免侵入车道）
    fallen_logs = [
        ("右侧雪地倒木_近景", [3.40, 0.24, -2.50], [88.0, -42.0, 0.0], [0.28, 3.60, 0.28], "#AEB4BD"),
        ("右侧雪地倒木_中景", [1.60, 0.26, -8.80], [88.0, -55.0, 0.0], [0.26, 3.20, 0.26], "#B2B8C1"),
        ("左侧雪地倒木_中景", [-4.50, 0.24, -4.80], [88.0, 36.0, 0.0], [0.28, 3.40, 0.28], "#B4BAC3"),
    ]
    for l_idx, (l_name, l_pos, l_rot, l_size, l_col) in enumerate(fallen_logs, 1):
        lg = stage.add_cylinder(l_name, pos=l_pos, rot=l_rot, size=l_size, color=l_col)
        lg.id = f"env_fallen_log_{l_idx}"
        lg.extra["arch_role"] = "prop"

    # 少量关键针叶树（仅保留影响轮廓、景深与前景遮挡关系的树木，每棵树为可编辑组合体：树干圆柱 + 双层树冠锥体）
    tree_specs = [
        ("右侧前景遮挡树", "tree_fg_right", [3.50, 0.0, 1.40], 1.15),
        ("右侧中景林木_01", "tree_mid_right_1", [3.60, 0.0, -5.50], 1.08),
        ("右侧中景林带_01b", "tree_mid_right_1b", [6.80, 0.0, -8.20], 1.18),
        ("右侧中远林木_02", "tree_far_right_2", [1.50, 0.0, -13.50], 1.15),
        ("右侧中远林带_02b", "tree_far_right_2b", [5.80, 0.0, -16.50], 1.22),
        ("右侧远景林木_03", "tree_far_right_3", [-0.60, 0.0, -21.50], 1.20),
        ("右侧远景林带_03b", "tree_far_right_3b", [4.20, 0.0, -24.50], 1.25),
        ("左侧前景林木_01", "tree_fg_left_1", [-4.20, 0.0, -1.80], 1.05),
        ("左侧中景林木_02", "tree_mid_left_2", [-6.50, 0.0, -9.20], 1.12),
        ("左侧中远林木_03", "tree_mid_left_3", [-9.00, 0.0, -17.50], 1.18),
        ("远景林带左树_04", "tree_far_center_4", [-11.50, 0.0, -27.00], 1.25),
    ]
    for t_name, t_id, t_pos, t_scale in tree_specs:
        trunk_h = round(4.80 * t_scale, 2)
        trunk_r = round(0.34 * t_scale, 2)
        c1_h = round(3.60 * t_scale, 2)
        c1_w = round(2.20 * t_scale, 2)
        c2_h = round(2.90 * t_scale, 2)
        c2_w = round(1.65 * t_scale, 2)
        tree_obj = PrimitiveObj(
            obj_id=t_id,
            name=t_name,
            shape="group",
            pos=t_pos,
            rot=[0.0, 0.0, 0.0],
            size=[c1_w, round(trunk_h + c2_h * 0.6, 2), c1_w],
            color="#C9CFD7",
        )
        tree_obj.extra = {
            "arch_role": "prop",
            "parts": [
                {
                    "id": f"{t_id}_trunk",
                    "name": "树干",
                    "shape": "cylinder",
                    "pos": [0.0, round(trunk_h * 0.5, 2), 0.0],
                    "rot": [0.0, 0.0, 0.0],
                    "size": [trunk_r, trunk_h, trunk_r],
                    "color": "#B6BCC4",
                },
                {
                    "id": f"{t_id}_canopy_low",
                    "name": "下层雪冠",
                    "shape": "cone",
                    "pos": [0.0, round(trunk_h * 0.58, 2), 0.0],
                    "rot": [0.0, 0.0, 0.0],
                    "size": [c1_w, c1_h, c1_w],
                    "color": "#CFD5DD",
                },
                {
                    "id": f"{t_id}_canopy_top",
                    "name": "上层雪冠",
                    "shape": "cone",
                    "pos": [0.0, round(trunk_h * 0.88, 2), 0.0],
                    "rot": [0.0, 0.0, 0.0],
                    "size": [c2_w, c2_h, c2_w],
                    "color": "#DCE1E8",
                },
            ],
        }
        stage.objects.append(tree_obj)

    # -------------------------------------------------------------------------
    # 2. 车队资产（主车 uaz_lead + 后随车 uaz_follow，跨镜头复用同一资产 ID）
    # -------------------------------------------------------------------------
    uaz_lead = PrimitiveObj(
        obj_id="uaz_lead",
        name="主吉普车 (UAZ-469 四人座舱)",
        shape="uaz469",
        pos=[-6.20, 1.01, -23.00] if is_shot1 else [0.0, 1.01, 0.0],
        rot=[0.0, 14.5, 0.0] if is_shot1 else [0.0, 0.0, 0.0],
        size=[1.78, 2.02, 4.02],
        color="#D5D9DF",
    )
    uaz_lead.locked = False
    if is_shot1:
        # 第 1 镜：主车真实沿弯曲雪道从左上方远景连续匀速驶向右下方前景（位置弧长与朝向切线连续协调，中间帧不停车）
        lead_kfs = [
            (0.00, [-6.20, 1.01, -23.00], [0.0, 14.5, 0.0]),
            (1.00, [-4.77, 1.01, -17.36], [0.0, 14.3, 0.0]),
            (2.00, [-3.37, 1.01, -11.71], [0.0, 13.8, 0.0]),
            (3.00, [-2.02, 1.01, -6.05], [0.0, 12.8, 0.0]),
            (3.75, [-1.06, 1.01, -1.79], [0.0, 11.4, 0.0]),
            (4.42, [0.00, 1.01, 2.10], [0.0, 10.0, 0.0]),
        ]
        for kt, kpos, krot in lead_kfs:
            uaz_lead.add_keyframe(kt, pos=kpos, rot=krot, interp="continuous")
    stage.objects.append(uaz_lead)

    uaz_follow = PrimitiveObj(
        obj_id="uaz_follow",
        name="后随车队车辆 (UAZ-469)",
        shape="uaz469",
        pos=[-8.40, 1.04, -34.00] if is_shot1 else [-0.45, 1.04, -12.50],
        rot=[0.0, 13.5, 0.0] if is_shot1 else [0.0, 5.0, 0.0],
        size=[1.82, 2.06, 4.10],
        color="#C8CDD5",
    )
    uaz_follow.locked = False
    if is_shot1:
        follow_kfs = [
            (0.00, [-8.40, 1.04, -34.00], [0.0, 13.5, 0.0]),
            (1.00, [-7.04, 1.04, -28.38], [0.0, 14.0, 0.0]),
            (2.00, [-5.64, 1.04, -22.76], [0.0, 14.5, 0.0]),
            (3.00, [-4.20, 1.04, -17.14], [0.0, 14.5, 0.0]),
            (4.42, [-2.10, 1.04, -9.20], [0.0, 13.8, 0.0]),
        ]
        for kt, kpos, krot in follow_kfs:
            uaz_follow.add_keyframe(kt, pos=kpos, rot=krot, interp="continuous")
    else:
        # 第 2 镜：透过后车窗留空开口可见后方雪道上的跟随车辆连续平稳跟进
        uaz_follow.add_keyframe(0.00, pos=[-0.45, 1.04, -12.50], rot=[0.0, 5.0, 0.0], interp="continuous")
        uaz_follow.add_keyframe(2.83, pos=[-0.35, 1.04, -11.54], rot=[0.0, 4.0, 0.0], interp="continuous")
        uaz_follow.add_keyframe(5.62, pos=[-0.25, 1.04, -10.60], rot=[0.0, 3.0, 0.0], interp="continuous")
    stage.objects.append(uaz_follow)

    # -------------------------------------------------------------------------
    # 3. 车内四人（统一使用“球头人偶” body_style="pawn"，保留四种身份色与跨镜头一致 ID）
    #    注意：摄影机从 +Z 往 -Z 拍车内时：
    #    - 画面右侧 (Screen Right) = 车辆左侧 (+X，方向盘侧，驾驶员)
    #    - 画面左侧 (Screen Left)  = 车辆右侧 (-X，副驾驶侧)
    # -------------------------------------------------------------------------
    # (1) 副驾驶（前排画面左，身份色：天蓝 #73A9D8）
    c_codriver = stage.add_character(
        "副驾驶 (前排左)",
        pos=[-0.43, 0.42, -0.06],
        rot_y=4.0,
        scale=0.90,
        color="#73A9D8",
        pose="sitting",
    )
    c_codriver.id = "char_codriver"
    c_codriver.body_style = "pawn"
    c_codriver.ik(
        pelvis=[0.0, 0.48, 0.0],
        chest=[0.02, 0.90, -0.02],
        head_target=[0.06, 1.12, 0.60],
        hand_L=[0.18, 0.56, 0.24],
        hand_R=[-0.18, 0.56, 0.24],
    )
    c_codriver.extra["mounted_vehicle_id"] = "uaz_lead"
    c_codriver.extra["follow_mounted_vehicle"] = True
    c_codriver.extra["seat_offset"] = [-0.43, 0.42, -0.06]
    c_codriver.extra["seat_rot_offset"] = 4.0
    if not is_shot1:
        c_codriver.add_keyframe(0.00, pos=[-0.43, 0.42, -0.06], rot_y=4.0, targets=c_codriver.targets, interp="smooth")
        c_codriver.add_keyframe(5.62, pos=[-0.43, 0.42, -0.06], rot_y=6.0, targets=c_codriver.targets, interp="smooth")

    # (2) 驾驶员（前排画面右/方向盘后，身份色：珊瑚红 #D98178）
    c_driver = stage.add_character(
        "驾驶员 (前排右)",
        pos=[0.43, 0.42, -0.06],
        rot_y=-4.0,
        scale=0.90,
        color="#D98178",
        pose="sitting",
    )
    c_driver.id = "char_driver"
    c_driver.body_style = "pawn"
    c_driver.ik(
        pelvis=[0.0, 0.48, 0.0],
        chest=[0.0, 0.90, -0.02],
        head_target=[0.0, 1.14, 0.65],
        hand_L=[0.16, 0.68, 0.34],
        hand_R=[-0.16, 0.68, 0.34],
    )
    c_driver.extra["mounted_vehicle_id"] = "uaz_lead"
    c_driver.extra["follow_mounted_vehicle"] = True
    c_driver.extra["seat_offset"] = [0.43, 0.42, -0.06]
    c_driver.extra["seat_rot_offset"] = -4.0
    if not is_shot1:
        c_driver.add_keyframe(0.00, pos=[0.43, 0.42, -0.06], rot_y=-4.0, targets=c_driver.targets, interp="smooth")
        c_driver.add_keyframe(5.62, pos=[0.43, 0.42, -0.06], rot_y=-4.0, targets=c_driver.targets, interp="smooth")

    # (3) 后排左乘员（画面左侧，身体向画面右侧后排同伴前倾、靠拢，身份色：青绿 #79B69C）
    c_rear_left = stage.add_character(
        "后排左乘员 (前倾靠拢)",
        pos=[-0.40, 0.34, -1.22],
        rot_y=50.0,
        scale=0.80,
        color="#79B69C",
        pose="sitting",
    )
    c_rear_left.id = "char_rear_left"
    c_rear_left.body_style = "pawn"
    base_rl_targets = {k: list(v) for k, v in c_rear_left.targets.items()}
    base_rl_targets["pelvis"] = [0.0, 0.48, 0.0]
    base_rl_targets["chest"] = [0.0, 0.88, 0.08]
    base_rl_targets["head_target"] = [0.0, 1.04, 0.62]
    c_rear_left.ik(**base_rl_targets)
    c_rear_left.extra["mounted_vehicle_id"] = "uaz_lead"
    c_rear_left.extra["follow_mounted_vehicle"] = True
    c_rear_left.extra["seat_offset"] = [-0.40, 0.34, -1.22]
    c_rear_left.extra["seat_rot_offset"] = 50.0
    if not is_shot1:
        # 第 2 镜：按原片时间节奏，后排左乘员在左半画幅内逐步向右前倾、俯身压低靠近后排右乘员（与右乘员保持清晰主次分离）
        rl_kf_specs = [
            (0.00, [-0.40, 0.34, -1.22], 50.0, [0.0, 0.88, 0.08], [0.0, 1.04, 0.62]),
            (1.83, [-0.39, 0.30, -1.22], 54.0, [0.0, 0.85, 0.11], [0.0, 0.98, 0.66]),
            (2.83, [-0.38, 0.25, -1.21], 58.0, [0.0, 0.82, 0.14], [0.0, 0.92, 0.70]),
            (4.33, [-0.37, 0.20, -1.20], 60.0, [0.0, 0.80, 0.15], [0.0, 0.86, 0.74]),
            (5.62, [-0.37, 0.14, -1.20], 62.0, [0.0, 0.78, 0.16], [0.0, 0.82, 0.76]),
        ]
        for kt, kpos, kry, kchest, khead in rl_kf_specs:
            t_map = {k: list(v) for k, v in base_rl_targets.items()}
            t_map["chest"] = kchest
            t_map["head_target"] = khead
            c_rear_left.add_keyframe(kt, pos=kpos, rot_y=kry, targets=t_map, interp="smooth")

    # (4) 后排右乘员（画面右侧主视觉焦点，端坐微侧头，身份色：暖黄 #D4B56C）
    c_rear_right = stage.add_character(
        "后排右乘员 (后排主焦点)",
        pos=[0.26, 0.34, -1.24],
        rot_y=-12.0,
        scale=0.82,
        color="#D4B56C",
        pose="sitting",
    )
    c_rear_right.id = "char_rear_right"
    c_rear_right.body_style = "pawn"
    base_rr_targets = {k: list(v) for k, v in c_rear_right.targets.items()}
    base_rr_targets["pelvis"] = [0.0, 0.48, 0.0]
    base_rr_targets["chest"] = [0.0, 0.92, 0.02]
    base_rr_targets["head_target"] = [0.0, 1.16, 0.65]
    c_rear_right.ik(**base_rr_targets)
    c_rear_right.extra["mounted_vehicle_id"] = "uaz_lead"
    c_rear_right.extra["follow_mounted_vehicle"] = True
    c_rear_right.extra["seat_offset"] = [0.26, 0.34, -1.24]
    c_rear_right.extra["seat_rot_offset"] = -12.0
    if not is_shot1:
        rr_kf_specs = [
            (0.00, [0.26, 0.34, -1.24], -12.0, [0.0, 0.92, 0.02], [0.0, 1.16, 0.65]),
            (2.83, [0.26, 0.34, -1.24], -10.0, [0.0, 0.91, 0.03], [-0.08, 1.12, 0.62]),
            (5.62, [0.26, 0.34, -1.24], -8.0, [-0.02, 0.90, 0.04], [-0.14, 1.06, 0.58]),
        ]
        for kt, kpos, kry, kchest, khead in rr_kf_specs:
            t_map = {k: list(v) for k, v in base_rr_targets.items()}
            t_map["chest"] = kchest
            t_map["head_target"] = khead
            c_rear_right.add_keyframe(kt, pos=kpos, rot_y=kry, targets=t_map, interp="smooth")

    # -------------------------------------------------------------------------
    # 4. 摄影机画幅 (1280×544, 2.35:1) 与分镜运镜关键帧
    # -------------------------------------------------------------------------
    stage.camera["aspect_ratio"] = "2.35:1"
    stage.camera["export_width"] = 1280
    stage.camera["export_height"] = 544

    if is_shot1:
        # 第 1 镜：高机位俯拍雪林弯道车队（与主车关键帧时间节奏同步对齐，消除末端速度突变与画面回弹）
        cam_kfs_1 = [
            (0.00, [-3.40, 9.80, 8.80], [3.80, 1.50, -14.00], 36.0),
            (1.00, [-2.68, 7.75, 7.42], [3.08, 1.38, -10.80], 35.0),
            (2.00, [-1.90, 5.45, 6.04], [2.32, 1.28, -7.30], 33.8),
            (3.00, [-1.06, 3.15, 4.82], [1.52, 1.24, -3.50], 32.0),
            (3.75, [-0.44, 1.92, 4.06], [0.76, 1.28, -0.75], 30.4),
            (4.42, [0.00, 1.48, 3.62], [0.00, 1.32, 0.90], 29.5),
        ]
        stage.set_camera(pos=cam_kfs_1[0][1], target=cam_kfs_1[0][2], fov=cam_kfs_1[0][3])
        for kt, cpos, ctar, cfov in cam_kfs_1:
            stage.add_camera_keyframe(kt, pos=cpos, target=ctar, fov=cfov, interp="continuous")
    else:
        # 第 2 镜：从前挡风玻璃外整体车内四人关系 -> 穿过前排两座间隙 -> 推至后排两人近景
        cam_kfs_2 = [
            (0.00, [0.00, 1.48, 1.52], [0.00, 1.32, -1.20], 30.0),
            (1.00, [0.00, 1.46, 1.18], [0.00, 1.30, -1.20], 30.0),
            (1.83, [0.00, 1.44, 0.88], [0.00, 1.28, -1.20], 29.5),
            (2.83, [0.00, 1.39, 0.54], [0.01, 1.24, -1.22], 28.5),
            (3.58, [0.00, 1.35, 0.34], [0.01, 1.21, -1.24], 27.5),
            (4.33, [0.00, 1.32, 0.20], [0.02, 1.20, -1.25], 26.5),
            (5.62, [0.05, 1.29, -0.04], [0.09, 1.19, -1.25], 24.5),
        ]
        stage.set_camera(pos=cam_kfs_2[0][1], target=cam_kfs_2[0][2], fov=cam_kfs_2[0][3])
        for kt, cpos, ctar, cfov in cam_kfs_2:
            stage.add_camera_keyframe(kt, pos=cpos, target=ctar, fov=cfov, interp="continuous")

    scene_dict = stage.to_dict()
    ref_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_videos")
    if is_shot1:
        ref_slice = os.path.join(ref_dir, "ref_convoy_shot1.mp4")
        scene_dict["ref_video"] = {
            "path": ref_slice,
            "filename": "ref_convoy_shot1.mp4",
            "url": "/ref_videos/ref_convoy_shot1.mp4",
            "width": 1280,
            "height": 544,
            "aspect_num": round(1280.0 / 544.0, 4),
            "aspect_label": "2.35:1 宽银幕 (2.35:1, 1280×544)",
            "matched_aspect_key": "2.35:1",
            "duration": 4.42,
            "fps": 24.0,
            "total_frames": 106,
            "time_offset": 0.0,
            "source_video": os.path.join(ref_dir, "91d8e0a37359f9fd0afd7866042f57de.mp4"),
            "source_frame_range": [0, 105],
        }
    else:
        ref_slice = os.path.join(ref_dir, "ref_convoy_shot2.mp4")
        scene_dict["ref_video"] = {
            "path": ref_slice,
            "filename": "ref_convoy_shot2.mp4",
            "url": "/ref_videos/ref_convoy_shot2.mp4",
            "width": 1280,
            "height": 544,
            "aspect_num": round(1280.0 / 544.0, 4),
            "aspect_label": "2.35:1 宽银幕 (2.35:1, 1280×544)",
            "matched_aspect_key": "2.35:1",
            "duration": 5.62,
            "fps": 24.0,
            "total_frames": 135,
            "time_offset": 0.0,
            "source_video": os.path.join(ref_dir, "91d8e0a37359f9fd0afd7866042f57de.mp4"),
            "source_frame_range": [106, 240],
        }
    return scene_dict



def evaluate_five_men_15s_at_time(t: float) -> Dict[str, Any]:
    """
    统一的按秒求值函数 evaluateAtTime(t)（0.0 <= t <= 15.0s），同时驱动五名角色、3部件极简步枪、摄影机与移焦参数：
    - 0–4s：鲍里斯低头朝摄影机走近 (z: -1.10 -> +1.40)，后方四人错开跟进，双腿交替摆动、身体轻微起伏；
    - 4–6s：鲍里斯平滑停步抬头，看向画面最右侧 (+X, yaw=+42°)，后方四人陆续停下，步态随速度归零；
    - 6–10s：鲍里斯头部与头前短线从画面最右侧 (+42°) 缓慢扫向画面中心 (+2°)，步枪小幅同向转动；
    - 10–14s：摄影机真实向前推进 (z: 3.85 -> 2.28)，枪口相对头部更快变大并进入右侧近景，焦点由脸部平滑移向枪口；
    - 14–15s：结尾保持（枪口占右侧前景、鲍里斯头部在枪后、后方四人在更深处）。
    """
    def _cl(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _ss(e0: float, e1: float, x: float) -> float:
        u = _cl((x - e0) / max(1e-6, e1 - e0), 0.0, 1.0)
        return u * u * (3.0 - 2.0 * u)

    def _sms(e0: float, e1: float, x: float) -> float:
        u = _cl((x - e0) / max(1e-6, e1 - e0), 0.0, 1.0)
        return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)

    def _lrp(a: float, b: float, u: float) -> float:
        return a + (b - a) * u

    def _walk_curve(t_val: float, t_start: float, t_stop: float) -> Tuple[float, float]:
        if t_val <= t_start:
            return 0.0, 0.0
        if t_val >= t_stop:
            return 1.0, 0.0
        dur = t_stop - t_start
        tau = t_val - t_start
        t_acc = min(0.45, dur * 0.15)
        t_dec = max(t_acc + 0.1, dur - 0.95)
        norm_area = 0.5 * t_acc + (t_dec - t_acc) + 0.5 * (dur - t_dec)
        if tau < t_acc:
            u = tau / t_acc
            env = u * u * (3.0 - 2.0 * u)
            area = 0.5 * tau * env
        elif tau < t_dec:
            env = 1.0
            area = 0.5 * t_acc + (tau - t_acc)
        else:
            u = (tau - t_dec) / (dur - t_dec)
            env = 1.0 - u * u * (3.0 - 2.0 * u)
            area = 0.5 * t_acc + (t_dec - t_acc) + (dur - t_dec) * (u - 0.5 * u * u * u * (2.0 - u))
        return _cl(area / max(1e-6, norm_area), 0.0, 1.0), env

    # 1. 鲍里斯 (char_boris, 暖橙 #D97A36)
    prog_b, env_b = _walk_curve(t, 0.0, 4.35)
    x_b = -0.06 + 0.012 * math.sin(2.0 * math.pi * 0.95 * t) * env_b
    z_b = -1.10 + 2.50 * prog_b  # -1.10m -> +1.40m (走近约四步)
    ph_b = 2.0 * math.pi * 0.95 * t
    str_b = math.sin(ph_b) * env_b
    lift_bL = max(0.0, math.cos(ph_b)) * 0.075 * env_b
    lift_bR = max(0.0, -math.cos(ph_b)) * 0.075 * env_b
    bob_b = abs(math.cos(ph_b)) * 0.028 * env_b

    # 头部朝向：0-4s 低头 (+15° = 0.26 rad, yaw=0)；4-6s 停步抬头看向画面最右侧 (+X, yaw=+42° = 0.73 rad)；
    # 6-10s 从画面最右侧缓慢扫向画面中心 (yaw -> +2° = 0.035 rad)
    u_look_up = _sms(3.85, 5.65, t)
    u_scan_center = _sms(6.00, 9.95, t)
    head_pitch_b = _lrp(0.26, -0.02, u_look_up)
    head_yaw_b = _lrp(0.0, 0.73, u_look_up) * (1.0 - u_scan_center) + 0.035 * u_scan_center

    # 步枪姿态：随鲍里斯手部与上身稳定绑定，6-10s 小幅同向转动并保持结尾枪口进入右侧前景
    gun_yaw_b = _lrp(0.26, 0.38, u_look_up) * (1.0 - u_scan_center) + 0.195 * u_scan_center
    gun_pitch_b = _lrp(0.16, -0.025, _sms(4.0, 9.8, t)) - 0.02 * str_b
    gun_y_b = _lrp(1.24, 1.385, _sms(4.2, 10.2, t)) + bob_b * 0.85
    gun_pos_b = [round(x_b + 0.02, 4), round(gun_y_b - 0.01, 4), round(z_b + 0.15, 4)]
    gun_rot_b = [round(math.degrees(gun_pitch_b), 2), round(math.degrees(gun_yaw_b), 2), 0.0]

    # 2. 后方四人（错开时间跟进、陆续停下、6-10s 轻微转头或等待）
    follower_specs = [
        ("char_f1", -1.22, -3.10, -0.60, 0.15, 4.65, 0.90, 0.9, 0.16),
        ("char_f2", -0.76, -4.20, -1.70, 0.35, 5.05, 0.86, 2.1, -0.18),
        ("char_f3",  0.94, -3.50, -1.00, 0.25, 4.85, 0.88, 1.5, -0.14),
        ("char_f4",  0.42, -5.10, -2.60, 0.45, 5.35, 0.84, 2.8, 0.15),
    ]
    followers_state = []
    for fid, fx0, fz0, fz1, t_s, t_e, freq, ph_off, wait_yaw in follower_specs:
        pf, env_f = _walk_curve(t, t_s, t_e)
        fz = fz0 + (fz1 - fz0) * pf
        ph_f = 2.0 * math.pi * freq * t + ph_off
        str_f = math.sin(ph_f) * env_f
        lift_fL = max(0.0, math.cos(ph_f)) * 0.065 * env_f
        lift_fR = max(0.0, -math.cos(ph_f)) * 0.065 * env_f
        bob_f = abs(math.cos(ph_f)) * 0.024 * env_f
        fyaw = wait_yaw * math.sin(_cl((t - 5.5) / 6.0, 0.0, 1.0) * math.pi)
        followers_state.append({
            "id": fid,
            "x": round(fx0, 4),
            "z": round(fz, 4),
            "env": env_f,
            "str": str_f,
            "liftL": lift_fL,
            "liftR": lift_fR,
            "bob": bob_f,
            "head_yaw": fyaw,
        })

    # 3. 摄影机路径与移焦参数 (10-14s 真实三维前推 + 焦点从脸部移至枪口)
    u_push = _sms(10.0, 14.0, t)
    cam_pos = [
        round(_lrp(0.00, 0.045, u_push), 4),
        round(_lrp(1.45, 1.435, u_push), 4),
        round(_lrp(3.85, 2.42, u_push), 4),
    ]
    cam_tar = [
        round(_lrp(0.00, 0.025, u_push), 4),
        round(_lrp(1.32, 1.405, u_push), 4),
        1.40,
    ]
    cam_fov = 32.0

    return {
        "t": t,
        "boris": {
            "x": round(x_b, 4),
            "z": round(z_b, 4),
            "env": env_b,
            "str": str_b,
            "liftL": lift_bL,
            "liftR": lift_bR,
            "bob": bob_b,
            "head_pitch": head_pitch_b,
            "head_yaw": head_yaw_b,
            "gun_y": gun_y_b,
            "gun_yaw": gun_yaw_b,
            "gun_pitch": gun_pitch_b,
            "gun_pos": gun_pos_b,
            "gun_rot": gun_rot_b,
        },
        "followers": followers_state,
        "camera": {
            "pos": cam_pos,
            "target": cam_tar,
            "fov": cam_fov,
            "focus_shift": u_push,
        },
    }


@sparse_generated_scene
def build_video5_five_men_15s_scene() -> Dict[str, Any]:
    """
    SU 极简彩色预演标准演示：五人、15 秒、一镜到底 (1280×720, 24fps, 360 帧)：
    - 默认采用 su_minimal 配置：
      * 树全部使用单个低面数圆锥体（锥尖朝上、底面落地 y=0），柔绿色系 (#5E8C6A 等)，无树干/积雪/圆环线；
      * 五名角色使用极简球头人偶 + 头前朝向短线，无五官/帽子/独眼罩/关节球，五人稳定身份色：
        鲍里斯暖橙 (#D97A36)，后方四人分别为柔蓝 (#5B8DBE)、柔绿 (#63A675)、浅紫 (#9378B8)、土黄 (#C8A24B)；
      * 步枪仅用 3 个基础体（枪托方块 + 机匣方块 + 枪管圆柱），独立深钢灰身份色 (#4A5568)；
      * 0–4s 走近 -> 4–6s 停步抬头看画面最右侧 (+X) -> 6–10s 头部与短线从最右侧扫向中心 -> 10–14s 摄影机真实前推与移焦 -> 14–15s 结尾保持。
    """
    stage = Stage()
    stage.clear()
    stage.fps = 24
    stage.duration = 15.00
    stage.render_mode = "su_minimal"
    stage.visual_style = "su_minimal"
    stage.environment = {
        "preset": "su_minimal",
        "sky": SU_MINIMAL_CONFIG["sky_color"],
        "ground": SU_MINIMAL_CONFIG["ground_color"],
        "fog_color": SU_MINIMAL_CONFIG["sky_color"],
        "fog": False,
        "show_ground": True,
        "show_grid": False,
    }

    # 1. 道路与单锥体树森林（近处保留影响构图的少量树，远处用稀疏单锥体阵列表达森林）
    road = stage.add_box("中央林间道路", pos=[0.0, 0.01, -4.5], size=[3.20, 0.02, 24.0], color=SU_MINIMAL_CONFIG["road_color"], obj_id="env_road_main")
    road.extra["arch_role"] = "road"

    cone_trees = [
        ("左侧前景树", "tree_01", -1.95,  0.80, 1.35, 4.8),
        ("右侧前景树", "tree_02",  1.88,  0.50, 1.40, 5.0),
        ("左侧中景树", "tree_03", -1.75, -1.80, 1.45, 5.2),
        ("右侧中景树", "tree_04",  1.72, -2.10, 1.45, 5.1),
        ("左侧后景树", "tree_05", -2.20, -4.50, 1.55, 5.4),
        ("右侧后景树", "tree_06",  2.15, -4.80, 1.50, 5.3),
        ("左侧远景树", "tree_07", -1.65, -7.40, 1.60, 5.6),
        ("右侧远景树", "tree_08",  1.60, -7.80, 1.60, 5.5),
        ("左外远景树", "tree_09", -3.40, -6.20, 1.70, 5.8),
        ("右外远景树", "tree_10",  3.35, -6.60, 1.70, 5.7),
        ("深处左林锥", "tree_11", -0.95, -10.50, 1.65, 5.8),
        ("深处右林锥", "tree_12",  0.88, -11.20, 1.70, 6.0),
    ]
    for t_name, t_id, tx, tz, tw, th in cone_trees:
        stage.add_tree_cone(name=t_name, x=tx, z=tz, width=tw, height=th, obj_id=t_id)

    # 2. 五名极简球头人偶 + 鲍里斯 3 部件极简步枪
    c_boris = stage.add_character("鲍里斯 (暖橙领队)", pos=[-0.06, 0.0, -1.10], rot_y=0.0, scale=1.00, color="#D97A36", pose="standing", char_id="char_boris")
    c_f1 = stage.add_character("左后队员 (柔蓝)", pos=[-1.05, 0.0, -3.10], rot_y=0.0, scale=0.98, color="#5B8DBE", pose="standing", char_id="char_f1")
    c_f2 = stage.add_character("中左队员 (柔绿)", pos=[-0.44, 0.0, -4.20], rot_y=0.0, scale=0.97, color="#63A675", pose="standing", char_id="char_f2")
    c_f3 = stage.add_character("右后队员 (浅紫)", pos=[0.92, 0.0, -3.50], rot_y=0.0, scale=0.98, color="#9378B8", pose="standing", char_id="char_f3")
    c_f4 = stage.add_character("纵深队员 (土黄)", pos=[0.36, 0.0, -5.10], rot_y=0.0, scale=0.96, color="#C8A24B", pose="standing", char_id="char_f4")
    f_chars = [c_f1, c_f2, c_f3, c_f4]

    rifle_boris = stage.add_gun_minimal("鲍里斯极简步枪 (3基础体)", pos=[-0.045, 1.23, -0.96], rot=[9.0, 15.0, 0.0], color="#4A5568", obj_id="rifle_boris")

    # 3. 采样关键帧 (由同一个 evaluate_five_men_15s_at_time(t) 驱动)
    sample_times = [round(i * 0.25, 2) for i in range(int(15.00 / 0.25) + 1)]
    cam_kfs = []
    for t in sample_times:
        st = evaluate_five_men_15s_at_time(t)
        b = st["boris"]
        hx_b = round(0.70 * math.sin(b["head_yaw"]) * math.cos(b["head_pitch"]), 3)
        hy_b = round(1.56 + b["bob"] - 0.70 * math.sin(b["head_pitch"]), 3)
        hz_b = round(0.70 * math.cos(b["head_yaw"]) * math.cos(b["head_pitch"]), 3)

        t_boris = {
            "pelvis": [0.0, round(0.86 + b["bob"], 3), 0.0],
            "chest": [0.0, round(1.32 + b["bob"], 3), round(0.02 * b["env"], 3)],
            "head_target": [hx_b, hy_b, hz_b],
            "hand_R": [round(-0.04 + 0.08 * math.sin(b["gun_yaw"]), 3), round(b["gun_y"] - 0.12, 3), 0.18],
            "hand_L": [round(-0.02 + 0.22 * math.sin(b["gun_yaw"]), 3), round(b["gun_y"] - 0.10, 3), 0.36],
            "foot_L": [0.10, round(0.04 + b["liftL"], 3), round(0.24 * b["str"], 3)],
            "foot_R": [-0.10, round(0.04 + b["liftR"], 3), round(-0.24 * b["str"], 3)],
            "knee_L_pole": [0.10, 0.48, 0.45],
            "knee_R_pole": [-0.10, 0.48, 0.45],
            "elbow_L_pole": [-0.10, round(b["gun_y"] - 0.26, 3), 0.20],
            "elbow_R_pole": [-0.16, round(b["gun_y"] - 0.30, 3), 0.10],
        }
        c_boris.add_keyframe(t, pos=[b["x"], 0.0, b["z"]], rot_y=0.0, targets=t_boris, interp="smooth")
        rifle_boris.add_keyframe(t, pos=b["gun_pos"], rot=b["gun_rot"], interp="continuous")

        for f_idx, f_st in enumerate(st["followers"]):
            fc = f_chars[f_idx]
            fyaw = f_st["head_yaw"]
            t_f = {
                "pelvis": [0.0, round(0.86 + f_st["bob"], 3), 0.0],
                "chest": [0.0, round(1.32 + f_st["bob"], 3), round(0.02 * f_st["env"], 3)],
                "head_target": [round(0.70 * math.sin(fyaw), 3), round(1.55 + f_st["bob"], 3), round(0.70 * math.cos(fyaw), 3)],
                "hand_L": [0.20, round(0.78 + f_st["bob"], 3), round(-0.16 * f_st["str"], 3)],
                "hand_R": [-0.20, round(0.78 + f_st["bob"], 3), round(0.16 * f_st["str"], 3)],
                "foot_L": [0.09, round(0.04 + f_st["liftL"], 3), round(0.20 * f_st["str"], 3)],
                "foot_R": [-0.09, round(0.04 + f_st["liftR"], 3), round(-0.20 * f_st["str"], 3)],
                "knee_L_pole": [0.09, 0.46, 0.42],
                "knee_R_pole": [-0.09, 0.46, 0.42],
                "elbow_L_pole": [0.28, 1.02, -0.16],
                "elbow_R_pole": [-0.28, 1.02, -0.16],
            }
            fc.add_keyframe(t, pos=[f_st["x"], 0.0, f_st["z"]], rot_y=0.0, targets=t_f, interp="smooth")

        cam_s = st["camera"]
        cam_kfs.append((t, cam_s["pos"], cam_s["target"], cam_s["fov"]))

    stage.camera["aspect_ratio"] = "16:9"
    stage.camera["export_width"] = 1280
    stage.camera["export_height"] = 720
    stage.set_camera(pos=cam_kfs[0][1], target=cam_kfs[0][2], fov=cam_kfs[0][3], aspect_ratio="16:9", export_width=1280, export_height=720)
    for kt, cpos, ctar, cfov in cam_kfs:
        stage.add_camera_keyframe(kt, pos=cpos, target=ctar, fov=cfov, interp="continuous")

    scene_dict = stage.to_dict()
    scene_dict["focus_shift_10_14"] = True
    return scene_dict



@sparse_generated_scene
def build_su_minimal_test_scene() -> Dict[str, Any]:
    """
    第二个小型验证测试场景：用于验证新建或转换任意场景时，默认自动沿用 su_minimal 规则
    （单锥体树、受控部件数、不同人物与道具稳定身份色）。
    """
    stage = Stage()
    stage.clear()
    stage.duration = 4.0
    stage.fps = 24
    stage.add_box("测试路面", pos=[0.0, 0.01, -2.0], size=[2.8, 0.02, 10.0], obj_id="env_road_test")
    stage.add_tree_cone("测试左松锥", x=-1.8, z=-1.5, width=1.3, height=4.6, obj_id="tree_test_L")
    stage.add_tree_cone("测试右松锥", x=1.8, z=-2.4, width=1.4, height=5.0, obj_id="tree_test_R")
    stage.add_uaz469("测试简模越野车", pos=[-1.6, 1.01, -4.8], rot=[0.0, 18.0, 0.0], obj_id="uaz_test")
    c1 = stage.add_character("测试领队A", pos=[-0.4, 0.0, -0.8], rot_y=15.0, pose="standing", char_id="char_test_a")
    c1.ik(hand_R=[-0.02, 1.16, 0.18], hand_L=[0.06, 1.18, 0.34], elbow_R_pole=[-0.18, 0.96, 0.08], elbow_L_pole=[0.18, 0.98, 0.12])
    c2 = stage.add_character("测试同伴B", pos=[0.5, 0.0, -1.6], rot_y=-12.0, pose="standing", char_id="char_test_b")
    g1 = stage.add_gun_minimal("测试步枪", pos=[-0.38, 1.24, -0.66], rot=[5.0, 15.0, 0.0], obj_id="rifle_test_a")
    c1.add_keyframe(0.0, pos=[-0.4, 0.0, -0.8], rot_y=15.0, targets=c1.targets)
    c1.add_keyframe(4.0, pos=[-0.3, 0.0, 0.2], rot_y=5.0, targets=c1.targets)
    c2.add_keyframe(0.0, pos=[0.5, 0.0, -1.6], rot_y=-12.0)
    c2.add_keyframe(4.0, pos=[0.4, 0.0, -0.7], rot_y=-5.0)
    g1.add_keyframe(0.0, pos=[-0.36, 1.25, -0.66], rot=[5.0, 15.0, 0.0])
    g1.add_keyframe(4.0, pos=[-0.26, 1.28, 0.34], rot=[2.0, 5.0, 0.0])
    stage.set_camera(pos=[0.0, 1.55, 2.80], target=[0.0, 1.20, -1.00], fov=34.0, aspect_ratio="16:9", export_width=1280, export_height=720)
    stage.add_camera_keyframe(0.0, pos=[0.0, 1.55, 2.80], target=[0.0, 1.20, -1.00], fov=34.0)
    stage.add_camera_keyframe(4.0, pos=[0.0, 1.48, 2.10], target=[0.0, 1.25, -0.60], fov=34.0)
    return convert_scene_to_su_minimal(stage.to_dict())


def _dir_to_euler_xyz_deg(fwd_vec: List[float], up_hint: List[float] = [0.0, 1.0, 0.0]) -> List[float]:
    """根据目标前向向量 (+Z 轴) 与上方向提示 (+Y 轴)，反解精确的 Rx @ Ry @ Rz 欧拉角 (度)，供 Python 与 Three.js 同步旋转枪械"""
    f = _normalize(np.asarray(fwd_vec, dtype=np.float64))
    u0 = _normalize(np.asarray(up_hint, dtype=np.float64))
    s = _normalize(np.cross(u0, f))
    u = _normalize(np.cross(f, s))
    # R columns are [s, u, f]
    R = np.column_stack([s, u, f])
    sy_val = max(-1.0, min(1.0, float(R[0, 2])))
    ry = math.asin(sy_val)
    if abs(sy_val) < 0.9999:
        rx = math.atan2(-float(R[1, 2]), float(R[2, 2]))
        rz = math.atan2(-float(R[0, 1]), float(R[0, 0]))
    else:
        rx = math.atan2(float(R[2, 1]), float(R[1, 1]))
        rz = 0.0
    return [round(math.degrees(rx), 2), round(math.degrees(ry), 2), round(math.degrees(rz), 2)]


def evaluate_video9_tree_cover_12s_at_time(t: float) -> Dict[str, Any]:
    """
    视频+9 复现（12.0 秒 · 24fps · 288 帧 · 992×432 · SU 极简彩色预演）：
    - 0.0–4.85s：鲍里斯站在右侧前景大树掩体左侧，双手举步枪抵肩朝画面最右侧（+X 方向）瞄准，枪管横穿前景树前方；
    - 4.85–5.15s：步枪击发产生短促后坐力震动；
    - 5.15–6.20s：鲍里斯迅速收枪至胸前下方，同时压低身体与头部下蹲躲入右侧大树掩体旁；
    - 6.20–12.0s：鲍里斯保持压低隐蔽姿态，缓慢抬头将视线重新投向画面右前方警觉观察；摄影机同步平滑 Dolly In 推成近景。
    """
    t = max(0.0, min(12.0, float(t)))

    def _cl(x: float, a: float, b: float) -> float:
        return max(a, min(b, x))

    def _sms(e0: float, e1: float, x: float) -> float:
        u = _cl((x - e0) / max(1e-6, e1 - e0), 0.0, 1.0)
        return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)

    def _lrp(a: float, b: float, u: float) -> float:
        return a + (b - a) * u

    # 1) 0–4.85s 呼吸微动 + 4.95s 击发后坐脉冲 + 5.10–6.15s 下蹲收枪 + 6.30–9.20s 抬头警觉看右
    breath = 0.006 * math.sin(t * 2.4)
    recoil = math.exp(-((t - 4.96) / 0.09) ** 2) if 4.80 <= t <= 5.25 else 0.0
    u_duck = _sms(5.05, 6.15, t)
    u_look_up = _sms(6.25, 9.20, t)

    # 鲍里斯根位置 (站在画左偏中 x=-0.18，右侧紧挨前景掩体树 x=+0.34)
    bx = _lrp(-0.18, -0.14, u_duck) - 0.032 * recoil
    by = _lrp(0.00, -0.20, u_duck) + breath * (1.0 - 0.5 * u_duck)
    bz = _lrp(0.00, 0.04, u_duck)
    rot_y = _lrp(14.0, 10.0, u_duck)

    # 胸廓高度与前倾 (局部坐标：站立时 neck 在 y≈1.08, head_center 在 y≈1.22)
    chest_x = _lrp(0.01, 0.03, u_duck) - 0.015 * recoil
    chest_y = _lrp(0.94, 0.88, u_duck)
    chest_z = _lrp(0.03, 0.09, u_duck)

    # 头部视线目标 (局部坐标：head_center 位于 [0, 1.22, 0.03])
    # 0–5s 沿枪管朝画面右侧 (+X) 瞄准；5.1–6.2s 下蹲低头看右下方；6.3–12s 抬头警觉看向画面右前方
    look_aim = [1.35, 1.24 + 0.03 * recoil, 0.55]
    look_duck_down = [0.52, 0.96, 0.78]
    look_alert_right = [0.92, 1.27, 0.72]
    if t <= 6.20:
        lx = _lrp(look_aim[0], look_duck_down[0], u_duck)
        ly = _lrp(look_aim[1], look_duck_down[1], u_duck)
        lz = _lrp(look_aim[2], look_duck_down[2], u_duck)
    else:
        lx = _lrp(look_duck_down[0], look_alert_right[0], u_look_up)
        ly = _lrp(look_duck_down[1], look_alert_right[1], u_look_up)
        lz = _lrp(look_duck_down[2], look_alert_right[2], u_look_up)

    # 双手局部 IK 目标（0–5s 双手举枪抵肩：右手握机匣、左手托护木；5.1–6.2s 收枪至胸前下方）
    hand_R_aim = [-0.04, 1.00, 0.22]
    hand_R_low = [-0.06, 0.84, 0.22]
    hand_L_aim = [0.22, 1.02, 0.26]
    hand_L_low = [0.16, 0.78, 0.26]

    hr = [
        round(_lrp(hand_R_aim[0], hand_R_low[0], u_duck) - 0.025 * recoil, 4),
        round(_lrp(hand_R_aim[1], hand_R_low[1], u_duck) + 0.015 * recoil, 4),
        round(_lrp(hand_R_aim[2], hand_R_low[2], u_duck), 4),
    ]
    hl = [
        round(_lrp(hand_L_aim[0], hand_L_low[0], u_duck) - 0.025 * recoil, 4),
        round(_lrp(hand_L_aim[1], hand_L_low[1], u_duck) + 0.022 * recoil, 4),
        round(_lrp(hand_L_aim[2], hand_L_low[2], u_duck), 4),
    ]

    # 步枪世界空间位置与朝向（0–5s 抵在肩窝 y=1.03 指向画面最右侧 +X，横穿右侧树干前方；5.1–6.2s 下压收回胸前 y=0.68）
    gun_pos_aim = [-0.06, 1.03, 0.24]
    gun_pos_low = [-0.06, 0.68, 0.26]
    gpos = [
        round(_lrp(gun_pos_aim[0], gun_pos_low[0], u_duck) - 0.042 * recoil, 4),
        round(_lrp(gun_pos_aim[1], gun_pos_low[1], u_duck) + 0.018 * recoil + breath, 4),
        round(_lrp(gun_pos_aim[2], gun_pos_low[2], u_duck), 4),
    ]
    gun_fwd_aim = [0.955, 0.065 + 0.045 * recoil, 0.285]
    gun_fwd_low = [0.810, -0.360, 0.460]
    gfwd = [
        _lrp(gun_fwd_aim[0], gun_fwd_low[0], u_duck),
        _lrp(gun_fwd_aim[1], gun_fwd_low[1], u_duck),
        _lrp(gun_fwd_aim[2], gun_fwd_low[2], u_duck),
    ]
    grot = _dir_to_euler_xyz_deg(gfwd)

    # 摄影机运镜：0–5s 中景稳拍 -> 5–6.2s 随鲍里斯下蹲微降 -> 6.2–12.0s 平滑物理前推 (Dolly In) 至近景
    u_cam_drop = _sms(5.00, 6.30, t)
    u_cam_dolly = _sms(5.60, 12.00, t)
    cam_pos = [
        round(_lrp(0.00, -0.06, u_cam_dolly), 4),
        round(_lrp(1.22, 1.03, u_cam_drop) + _lrp(0.0, -0.02, u_cam_dolly), 4),
        round(_lrp(2.15, 1.38, u_cam_dolly), 4),
    ]
    cam_tar = [
        round(_lrp(0.00, -0.09, u_cam_dolly), 4),
        round(_lrp(1.16, 0.96, u_cam_drop) + _lrp(0.0, 0.01, u_cam_dolly), 4),
        0.00,
    ]
    cam_fov = 27.0  # 50mm 恒定焦距

    return {
        "t": t,
        "boris": {
            "pos": [round(bx, 4), round(by, 4), round(bz, 4)],
            "rot_y": round(rot_y, 2),
            "look_at": [round(lx, 4), round(ly, 4), round(lz, 4)],
            "chest": [round(chest_x, 4), round(chest_y, 4), round(chest_z, 4)],
            "hand_R": hr,
            "hand_L": hl,
        },
        "gun": {
            "pos": gpos,
            "rot": grot,
        },
        "camera": {
            "pos": cam_pos,
            "target": cam_tar,
            "fov": cam_fov,
        },
    }


@sparse_generated_scene
def build_video9_tree_cover_12s_scene() -> Dict[str, Any]:
    """
    构建《视频+9》12 秒一镜到底 SU 极简彩色预演场景（992×432 · 24fps · 288 帧）：
    - 鲍里斯（暖橙 #D97A36）在右侧单锥体大树（#4B7356）左侧立姿举枪朝画右瞄准（0–5s）
    - 击发后迅速收枪下蹲隐蔽（5–6.2s）
    - 在树旁掩体后缓慢抬头警觉看向画右，摄影机同步 Dolly In 推至近景（6.2–12s）
    """
    stage = Stage()
    stage.clear()
    stage.duration = 12.0
    stage.fps = 24
    stage.render_mode = "su_minimal"
    stage.visual_style = "su_minimal"
    stage.environment["ground"] = SU_MINIMAL_CONFIG["ground_color"]
    stage.environment["sky"] = SU_MINIMAL_CONFIG["sky_color"]

    # 1. 林间浅灰道路底面
    stage.add_box(
        name="林间雪道",
        pos=[0.0, 0.008, -6.0],
        size=[4.2, 0.016, 28.0],
        color=SU_MINIMAL_CONFIG["road_color"],
        obj_id="env_road_main",
    )

    # 2. 右侧前景掩体树（单锥体树，紧贴鲍里斯屏幕右侧 x=+0.34，形成掩体遮挡与近景参照）
    # 以及后方纵深稀疏单锥体树林
    cone_trees = [
        ("tree_cover_fg", "右侧掩体近景松锥", 0.42, -0.22, 0.64, 4.6, "#4B7356"),
        ("tree_fg_left", "左侧边缘松锥", -1.25, -0.80, 0.85, 5.0, "#537E5F"),
        ("tree_mid_l1", "左中景松锥1", -0.82, -2.60, 0.96, 5.2, "#5E8C6A"),
        ("tree_mid_c1", "中景偏左松锥", -0.18, -3.80, 1.05, 5.5, "#537E5F"),
        ("tree_mid_r1", "右中景松锥1", 0.92, -2.40, 0.92, 5.0, "#689674"),
        ("tree_mid_r2", "右中景松锥2", 1.55, -3.90, 1.12, 5.6, "#5E8C6A"),
        ("tree_far_l1", "左远景松锥1", -1.95, -5.80, 1.30, 6.0, "#4B7356"),
        ("tree_far_c1", "中远景松锥1", 0.35, -6.40, 1.35, 6.2, "#5E8C6A"),
        ("tree_far_r1", "右远景松锥1", 1.85, -6.80, 1.40, 6.3, "#537E5F"),
        ("tree_deep_l", "纵深左松锥", -1.10, -9.20, 1.55, 6.8, "#689674"),
        ("tree_deep_c", "纵深中松锥", 0.65, -10.50, 1.65, 7.0, "#4B7356"),
        ("tree_deep_r", "纵深右松锥", 2.30, -11.40, 1.70, 7.2, "#5E8C6A"),
    ]
    for tid, tname, tx, tz, tw, th, tcol in cone_trees:
        stage.add_tree_cone(name=tname, x=tx, z=tz, width=tw, height=th, color=tcol, obj_id=tid)

    # 3. 鲍里斯（暖橙 #D97A36）与 3 部件极简步枪（独立深蓝灰钢色 #4A5568）
    s0 = evaluate_video9_tree_cover_12s_at_time(0.0)
    boris = stage.add_character(
        name="鲍里斯(暖橙领队)",
        pos=s0["boris"]["pos"],
        rot_y=s0["boris"]["rot_y"],
        pose="standing",
        color="#D97A36",
        char_id="char_boris",
    )
    rifle = stage.add_gun_minimal(
        name="鲍里斯步枪(3部件)",
        pos=s0["gun"]["pos"],
        rot=s0["gun"]["rot"],
        color="#4A5568",
        obj_id="rifle_boris",
    )

    sample_times = [round(i * 0.20, 2) for i in range(61)]  # 0.00s .. 12.00s 共 61 个采样关键帧
    # 在 4.85s–5.25s 加密采样以精确捕获开火后坐与快速收枪下蹲
    for extra_t in [4.85, 4.95, 5.05, 5.15, 5.30, 5.50, 5.70, 5.90, 6.10]:
        if extra_t not in sample_times:
            sample_times.append(extra_t)
    sample_times.sort()

    cam_kfs = []
    for kt in sample_times:
        st = evaluate_video9_tree_cover_12s_at_time(kt)
        b_st = st["boris"]
        t_b = {
            "pelvis": [0.0, 0.48, -0.02],
            "head_target": b_st["look_at"],
            "chest": b_st["chest"],
            "hand_R": b_st["hand_R"],
            "hand_L": b_st["hand_L"],
            "foot_R": [-0.12, 0.03, -0.04],
            "foot_L": [0.12, 0.03, 0.06],
            "knee_R_pole": [-0.14, 0.48, 0.45],
            "knee_L_pole": [0.14, 0.48, 0.45],
            "elbow_R_pole": [-0.32, 0.72, 0.08],
            "elbow_L_pole": [0.20, 0.68, 0.14],
        }
        if kt == 0.0:
            boris.targets = copy.deepcopy(t_b)
        boris.add_keyframe(
            kt,
            pos=b_st["pos"],
            rot_y=b_st["rot_y"],
            look_at=b_st["look_at"],
            targets=t_b,
            interp="smooth",
        )
        g_st = st["gun"]
        rifle.add_keyframe(kt, pos=g_st["pos"], rot=g_st["rot"], interp="smooth")
        c_st = st["camera"]
        cam_kfs.append((kt, c_st["pos"], c_st["target"], c_st["fov"]))

    stage.set_camera(
        pos=cam_kfs[0][1],
        target=cam_kfs[0][2],
        fov=cam_kfs[0][3],
        aspect_ratio="2.39:1",
        export_width=992,
        export_height=432,
    )
    for kt, cpos, ctar, cfov in cam_kfs:
        stage.add_camera_keyframe(kt, pos=cpos, target=ctar, fov=cfov, interp="continuous")

    scene_dict = convert_scene_to_su_minimal(stage.to_dict())
    ref_v9_path = "/Users/wangmu/.gemini/antigravity/scratch/director-stage/ref_videos/video9_ref_12s.mp4"
    scene_dict["ref_video"] = {
        "path": ref_v9_path,
        "filename": "video9_ref_12s.mp4",
        "fps": 24.0,
        "duration": 12.0,
        "width": 992,
        "height": 432,
        "time_offset": 0.0,
    }
    return scene_dict



def evaluate_video3_cabin_curtain_7s_at_time(t: float) -> Dict[str, Any]:
    """
    《视频_3》7.0 秒室内木屋背身撩帘观察 → 俯身钻入帘后暗门（992×432 · 24fps · 168 帧 · SU 极简彩色预演）：
    - 0.0–0.6s：角色背对摄影机站在画面中左（x=-0.16, z=-0.46, rot_y=170°），左手（画左）拉住左侧门帘边缘，头部从正前转向画面右侧（+X）观察架子；
    - 0.6–4.5s：保持背身侧头凝视画面右下方储物架，期间有细微呼吸与轻微点头观察（1.8–2.8s）；
    - 4.5–7.0s：转头回正并低头俯身，迈步向画面左前方的门帘后暗门走入（x: -0.16 -> -0.68, z: -0.46 -> -1.48），
      穿过 z=-0.92 的门帘平面后被左侧门帘自然遮挡直至仅露右侧衣摆边缘。
    """
    t = max(0.0, min(7.0, float(t)))

    def _cl(x: float, a: float, b: float) -> float:
        return max(a, min(b, x))

    def _sms(e0: float, e1: float, x: float) -> float:
        u = _cl((x - e0) / max(1e-6, e1 - e0), 0.0, 1.0)
        return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)

    def _lrp(a: float, b: float, u: float) -> float:
        return a + (b - a) * u

    u_turn_right = _sms(0.0, 0.55, t)
    u_enter = _sms(4.55, 6.75, t)
    u_duck = _sms(4.70, 6.20, t)
    walk_env = math.sin(_cl((t - 4.55) / 2.20, 0.0, 1.0) * math.pi)
    phase = (t - 4.55) * 4.8
    s_w = math.sin(phase) if t >= 4.55 else 0.0
    c_w = math.cos(phase) if t >= 4.55 else 0.0
    bob = abs(c_w) * 0.022 * walk_env
    breath = 0.005 * math.sin(t * 2.1) * (1.0 - walk_env)

    # 角色世界坐标：4.55s 前站在门帘前方 z=-0.46；4.55s–6.75s 向左前方跨入帘后 z=-1.48 (穿过 z=-0.92 的门帘被遮挡)
    px = _lrp(-0.15, -0.72, u_enter)
    py = _lrp(0.00, -0.18, u_duck) + bob + breath
    pz = _lrp(-0.46, -1.48, u_enter)
    rot_y = _lrp(168.0, 198.0, u_enter)

    # 局部坐标系下 (rot_y≈170° 背对相机)：局部 -X 对应世界 +X (画面右侧)，局部 +X 对应世界 -X (画面左侧)
    # 0.0s 正对后墙 -> 0.55s 转头看画面右侧 (-X_local = -0.88) -> 4.6s–5.5s 回正低头看左前暗门
    u_head_back = _sms(4.50, 5.45, t)
    nod = 0.04 * math.sin((t - 1.2) * 2.6) if 1.2 <= t <= 3.8 else 0.0
    hx_0 = _lrp(-0.22, -0.88, u_turn_right)
    hy_0 = _lrp(1.20, 1.17 + nod, u_turn_right)
    hz_0 = _lrp(0.85, 0.36, u_turn_right)

    hx = _lrp(hx_0, 0.12, u_head_back)
    hy = _lrp(hy_0, 0.96, u_head_back)
    hz = _lrp(hz_0, 0.82, u_head_back)

    # 左手 (画面左侧，局部 +X) 撩住门帘边缘；进入暗门时随身体前伸
    hl_x = _lrp(0.28, 0.22, u_enter)
    hl_y = _lrp(0.88, 0.80, u_enter)
    hl_z = _lrp(0.24, 0.32, u_enter)

    # 右手 (画面右侧，局部 -X) 自然垂放微屈
    hr_x = -0.22
    hr_y = _lrp(0.74, 0.70, u_enter)
    hr_z = 0.10

    str_L = s_w * 0.24 * walk_env
    str_R = -s_w * 0.24 * walk_env
    lift_L = max(0.0, s_w) * 0.07 * walk_env
    lift_R = max(0.0, -s_w) * 0.07 * walk_env

    # 门帘微动 (随左手撩动与钻入在 4.8–6.6s 轻微摆动)
    curtain_push = math.sin(_cl((t - 4.7) / 1.9, 0.0, 1.0) * math.pi)
    curtain_x = round(-0.74 - 0.03 * curtain_push, 4)

    return {
        "t": t,
        "char": {
            "pos": [round(px, 4), round(py, 4), round(pz, 4)],
            "rot_y": round(rot_y, 2),
            "look_at": [round(hx, 4), round(hy, 4), round(hz, 4)],
            "chest": [0.0, round(_lrp(0.93, 0.84, u_duck), 4), round(_lrp(0.02, 0.10, u_duck), 4)],
            "hand_L": [round(hl_x, 4), round(hl_y, 4), round(hl_z, 4)],
            "hand_R": [round(hr_x, 4), round(hr_y, 4), round(hr_z, 4)],
            "foot_L": [0.11, round(0.03 + lift_L, 4), round(str_L, 4)],
            "foot_R": [-0.11, round(0.03 + lift_R, 4), round(str_R, 4)],
        },
        "curtain_x": curtain_x,
        "camera": {
            "pos": [-0.08, 1.14, 0.88],
            "target": [-0.08, 1.10, -1.20],
            "fov": 30.0,
        },
    }


@sparse_generated_scene
def build_video3_cabin_curtain_7s_scene() -> Dict[str, Any]:
    """
    构建《视频_3》7.0 秒室内木屋背身撩帘观察 → 俯身钻入帘后暗门 SU 极简彩色预演场景（992×432 · 24fps · 168 帧）
    """
    stage = Stage()
    stage.clear()
    stage.duration = 7.0
    stage.fps = 24
    stage.render_mode = "su_minimal"
    stage.visual_style = "su_minimal"
    stage.environment["ground"] = "#8C939E"
    stage.environment["sky"] = "#4A525D"

    # 1. 木屋后墙（右侧墙板 + 左侧暗门洞深色背景）与顶梁
    stage.add_box("暗门深处背景", pos=[-0.55, 1.05, -1.95], size=[1.40, 2.10, 0.10], color="#282E36", obj_id="cabin_dark_doorway")
    stage.add_box("木屋右侧后墙", pos=[0.52, 1.05, -1.52], size=[1.45, 2.10, 0.10], color="#5B6470", obj_id="cabin_back_wall_r")
    stage.add_box("木屋门框立柱", pos=[-0.18, 1.05, -1.46], size=[0.12, 2.10, 0.14], color="#4D5560", obj_id="cabin_door_post")
    stage.add_box("木屋顶棚横梁", pos=[0.0, 1.72, -1.20], size=[2.80, 0.18, 1.20], color="#464E58", obj_id="cabin_ceiling_beam")

    # 2. 右侧两层储物木架与极简罐头/木箱体块（不同类别稳定身份色）
    stage.add_box("储物架上层板", pos=[0.55, 1.12, -1.36], size=[1.18, 0.04, 0.26], color="#7A6B58", obj_id="shelf_plank_top")
    stage.add_box("储物架下层板", pos=[0.55, 0.74, -1.36], size=[1.18, 0.04, 0.26], color="#7A6B58", obj_id="shelf_plank_bot")
    # 上层储物块（左侧扁木盒 + 中部圆柱罐 + 右侧木条箱）
    stage.add_box("上层木盒", pos=[0.16, 1.21, -1.35], size=[0.18, 0.14, 0.16], color="#9E8668", obj_id="shelf_box_top1")
    stage.add_cylinder("上层储物罐1", pos=[0.36, 1.20, -1.34], size=[0.09, 0.12, 0.09], color="#7E8E9E", obj_id="shelf_can_t1")
    stage.add_cylinder("上层储物罐2", pos=[0.49, 1.19, -1.35], size=[0.08, 0.10, 0.08], color="#6C7D8C", obj_id="shelf_can_t2")
    stage.add_Cylinder_or_box = stage.add_cylinder("上层储物罐3", pos=[0.62, 1.20, -1.34], size=[0.09, 0.12, 0.09], color="#8898A6", obj_id="shelf_can_t3")
    stage.add_box("上层右侧木箱", pos=[0.88, 1.24, -1.34], size=[0.26, 0.20, 0.20], color="#8C765C", obj_id="shelf_crate_top_r")
    # 下层储物罐阵列
    for idx_c, cx in enumerate([0.12, 0.25, 0.38, 0.52, 0.66, 0.82]):
        c_col = "#7E8E9E" if idx_c % 2 == 0 else "#6A7A89"
        stage.add_cylinder(f"下层罐头_{idx_c+1}", pos=[cx, 0.82, -1.34], size=[0.09, 0.12, 0.09], color=c_col, obj_id=f"shelf_can_b{idx_c+1}")

    # 3. 左侧悬挂门帘（位于 z=-0.92，角色在 0–4.5s 位于 z=-0.46 门帘前，4.6–7.0s 走入 z=-1.48 门帘后被自然遮挡！）
    curtain = stage.add_box(
        name="左侧门帘体块",
        pos=[-0.74, 1.02, -0.92],
        size=[0.96, 1.90, 0.06],
         rot=[0.0, 0.0, -4.0],
        color="#877862",
        obj_id="cabin_curtain_left",
    )

    # 4. 背身角色（暖橙身份色 #D97A36 + 头前短方向线 #1E242B）
    s0 = evaluate_video3_cabin_curtain_7s_at_time(0.0)
    man = stage.add_character(
        name="木屋男子(暖橙主角)",
        pos=s0["char"]["pos"],
        rot_y=s0["char"]["rot_y"],
        pose="standing",
        color="#D97A36",
        char_id="char_boris",
    )

    sample_times = [round(i * 0.20, 2) for i in range(36)]  # 0.00s .. 7.00s
    for extra_t in [0.15, 0.30, 0.50, 4.65, 4.90, 5.10, 5.30, 5.50, 5.70, 5.90, 6.10, 6.30, 6.50, 6.70]:
        if extra_t not in sample_times:
            sample_times.append(extra_t)
    sample_times.sort()

    cam_kfs = []
    for kt in sample_times:
        st = evaluate_video3_cabin_curtain_7s_at_time(kt)
        c_st = st["char"]
        t_c = {
            "pelvis": [0.0, 0.48, -0.02],
            "head_target": c_st["look_at"],
            "chest": c_st["chest"],
            "hand_L": c_st["hand_L"],
            "hand_R": c_st["hand_R"],
            "foot_L": c_st["foot_L"],
            "foot_R": c_st["foot_R"],
            "knee_L_pole": [0.13, 0.48, 0.42],
            "knee_R_pole": [-0.13, 0.48, 0.42],
            "elbow_L_pole": [0.34, 0.72, -0.08],
            "elbow_R_pole": [-0.32, 0.70, -0.08],
        }
        if kt == 0.0:
            man.targets = copy.deepcopy(t_c)
        man.add_keyframe(
            kt,
            pos=c_st["pos"],
            rot_y=c_st["rot_y"],
            look_at=c_st["look_at"],
            targets=t_c,
            interp="smooth",
        )
        curtain.add_keyframe(kt, pos=[st["curtain_x"], 1.02, -0.92], rot=[0.0, 0.0, -4.0], interp="smooth")
        cam_s = st["camera"]
        cam_kfs.append((kt, cam_s["pos"], cam_s["target"], cam_s["fov"]))

    stage.set_camera(
        pos=cam_kfs[0][1],
        target=cam_kfs[0][2],
        fov=cam_kfs[0][3],
        aspect_ratio="2.39:1",
        export_width=992,
        export_height=432,
    )
    for kt, cpos, ctar, cfov in cam_kfs:
        stage.add_camera_keyframe(kt, pos=cpos, target=ctar, fov=cfov, interp="continuous")

    scene_dict = convert_scene_to_su_minimal(stage.to_dict())
    ref_v3_path = "/Users/wangmu/.gemini/antigravity/scratch/director-stage/ref_videos/video3_ref.mp4"
    scene_dict["ref_video"] = {
        "path": ref_v3_path,
        "filename": "video3_ref.mp4",
        "fps": 24.0,
        "duration": 7.0,
        "width": 992,
        "height": 432,
        "time_offset": 0.0,
    }
    return scene_dict



def evaluate_seedance_snow_flinch_11s_at_time(t: float) -> Dict[str, Any]:
    """
    《雪林惊吓躲避｜Seedance 提示词》11.0 秒一镜到底精确复现（992×432 · 24fps · 264 帧 · SU 极简彩色预演）：
    - 0.0–6.0s：老人站在完好无损的树干左侧，已举枪抵肩贴腮朝画右（+X）瞄准；
      2.0–3.6s 枪口只做一次细小方向修正后稳住；老人全程不开枪，树干表面完整无弹坑；
      摄影机从 0.0s 起便以小幅度缓慢 Dolly In 推进。
    - t = 6.0s（明确锁定枪击节点）：画外射来一发子弹击中右侧近旁树干，树皮破裂形成弹坑，木屑向外飞散并按重力下落；
      老人听到枪击并察觉木屑，肩膀突然一缩，立即解除贴腮瞄准，朝摄影机方向小幅弯腰低头躲避（6.0–7.1s），步枪随上身降低至胸前较低位置。
    - 7.1–8.0s：老人保持略微弯腰，短暂停住，惊吓后的呼吸可见。
    - 8.0–10.0s：老人先缓缓抬起视线，再小幅转头看向画面右侧（+X）画外方向，步枪仍握在胸前低位，不再次举枪。
    - 10.0–11.0s：摄影机沿平滑浅弧推进至侧面 3/4 脸部特写（前景树干自然移向画面右缘，步枪位于画框下缘），结尾保持 3/4 构图与看画右视线，摄影机仍有极轻微前移。
    """
    t = max(0.0, min(11.0, float(t)))

    def _cl(x: float, a: float, b: float) -> float:
        return max(a, min(b, x))

    def _sms(e0: float, e1: float, x: float) -> float:
        u = _cl((x - e0) / max(1e-6, e1 - e0), 0.0, 1.0)
        return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)

    def _lrp(a: float, b: float, u: float) -> float:
        return a + (b - a) * u

    # 1) 0–6s 轻浅呼吸 + 2.0–3.6s 枪口一次细小修正
    breath_freq = 2.2 if t < 6.0 else 3.6  # 受惊后呼吸略急
    breath_amp = 0.004 if t < 6.0 else 0.007
    breath = breath_amp * math.sin(t * breath_freq)
    u_micro_adj = _sms(2.0, 3.3, t) * (1.0 - 0.35 * _sms(3.3, 4.4, t))

    # 2) t=6.0s 树干中弹受惊缩肩脉冲 + 6.05–7.15s 小幅弯腰低头躲避（只发生一次）
    flinch_shrug = math.exp(-((t - 6.12) / 0.14) ** 2) if 5.98 <= t <= 6.55 else 0.0
    u_duck = _sms(6.02, 7.15, t)
    # 7.15–7.95s 短暂停顿，7.95–9.85s 缓慢抬头并转头看画右
    u_lift_gaze = _sms(7.90, 9.40, t)
    u_turn_right = _sms(8.20, 9.95, t)

    # 老人根位置与微弯腰
    bx = _lrp(-0.18, -0.15, u_duck) - 0.018 * flinch_shrug
    by = _lrp(0.00, -0.19, u_duck) + breath
    bz = _lrp(0.00, 0.06, u_duck)  # 朝摄影机方向小幅弯腰前倾
    rot_y = _lrp(14.0, 10.0, u_duck)

    chest = [
        round(_lrp(0.01, 0.02, u_duck) - 0.012 * flinch_shrug, 4),
        round(_lrp(0.94, 0.86, u_duck) + 0.025 * flinch_shrug, 4),
        round(_lrp(0.03, 0.11, u_duck), 4),
    ]

    # 头部视线（0–6s 沿枪管盯画右 -> 6.0–7.15s 解除贴腮低头躲避 -> 7.15–7.95s 停顿 -> 8.0–10.0s 先抬视线再转头看画右）
    look_aim = [1.35, 1.24 + 0.015 * u_micro_adj, 0.55]
    look_duck = [0.28, 0.94, 0.85]
    look_end_right = [0.96, 1.26, 0.68]
    if t <= 7.50:
        lx = _lrp(look_aim[0], look_duck[0], u_duck)
        ly = _lrp(look_aim[1], look_duck[1], u_duck)
        lz = _lrp(look_aim[2], look_duck[2], u_duck)
    else:
        lx = _lrp(look_duck[0], look_end_right[0], u_turn_right)
        ly = _lrp(look_duck[1], look_end_right[1], u_lift_gaze)
        lz = _lrp(look_duck[2], look_end_right[2], u_turn_right)

    # 双手握枪 IK（0–6s 抵肩贴腮握持；6.0–7.15s 随上身降低至胸前低位，此后保持低位不再举枪）
    hr = [
        round(_lrp(-0.04, -0.06, u_duck), 4),
        round(_lrp(1.00, 0.80, u_duck) + 0.020 * flinch_shrug, 4),
        round(_lrp(0.22, 0.24, u_duck), 4),
    ]
    hl = [
        round(_lrp(0.22, 0.15, u_duck), 4),
        round(_lrp(1.02 + 0.012 * u_micro_adj, 0.75, u_duck) + 0.020 * flinch_shrug, 4),
        round(_lrp(0.26, 0.27, u_duck), 4),
    ]

    # 步枪世界空间位置与朝向（老人全程不开枪，无枪口火光；2–3.6s 微调一次，6.0s 后降低至画框下缘）
    gpos = [
        round(_lrp(-0.06, -0.07, u_duck), 4),
        round(_lrp(1.03 + 0.008 * u_micro_adj, 0.64, u_duck) + breath, 4),
        round(_lrp(0.24, 0.27, u_duck), 4),
    ]
    gfwd = [
        _lrp(0.955, 0.800, u_duck),
        _lrp(0.055 + 0.022 * u_micro_adj, -0.380, u_duck),
        _lrp(0.285 - 0.018 * u_micro_adj, 0.465, u_duck),
    ]
    grot = _dir_to_euler_xyz_deg(gfwd)

    # 3) 第 6.0 秒树干弹坑与飞散木屑物理轨迹（6.0s 前隐藏在地下 y=-5.0；6.0s 出现弹坑，木屑向外飞散并受重力下落，落尽后不再补充）
    if t < 6.0:
        crater_pos = [0.48, -5.0, 0.08]
        splinters = [
            [0.48, -5.0, 0.10],
            [0.48, -5.0, 0.10],
            [0.48, -5.0, 0.10],
        ]
    else:
        crater_pos = [0.48, 1.05, 0.08]  # 6.0s 起固定在右侧近景树干前表面的中弹破损处
        dt_s = t - 6.0
        # 3 块飞散木屑：初速度向外/向上/向前，受重力 g=2.8m/s^2 自然下落
        splinter_vels = [
            (0.36, 0.32, 0.25),
            (0.22, 0.16, 0.32),
            (0.45, -0.04, 0.20),
        ]
        splinters = []
        for vx_s, vy_s, vz_s in splinter_vels:
            sx_p = 0.48 + vx_s * dt_s
            sy_p = 1.05 + vy_s * dt_s - 0.5 * 2.8 * (dt_s ** 2)
            sz_p = 0.10 + vz_s * dt_s
            if sy_p < -0.2:
                sy_p = -5.0
            splinters.append([round(sx_p, 4), round(sy_p, 4), round(sz_p, 4)])

    # 4) 摄影机全程连续 Dolly In（前段 0–6s 幅度小保留中景腰部以上与步枪；6–10s 沿浅弧降低并接近脸部；10–11s 收束至侧面 3/4 特写且仍有极轻微前移）
    u_cam_pre = _sms(0.0, 6.0, t)
    u_cam_drop = _sms(6.0, 7.4, t)
    u_cam_main = _sms(6.0, 10.2, t)
    u_cam_tail = _sms(10.0, 11.0, t)

    # 沿浅弧向老人眼罩一侧的前侧方靠拢，前景树干自然移向画面右缘
    cam_x = 0.01 - 0.02 * u_cam_pre - 0.07 * u_cam_main - 0.012 * u_cam_tail
    cam_y = 1.22 - 0.01 * u_cam_pre - 0.16 * u_cam_drop - 0.03 * u_cam_main
    cam_z = 2.22 - 0.18 * u_cam_pre - 0.68 * u_cam_main - 0.06 * u_cam_tail

    tar_x = 0.01 - 0.015 * u_cam_pre - 0.10 * u_cam_main - 0.01 * u_cam_tail
    tar_y = 1.16 - 0.01 * u_cam_pre - 0.17 * u_cam_drop - 0.01 * u_cam_main
    tar_z = 0.04

    return {
        "t": t,
        "boris": {
            "pos": [round(bx, 4), round(by, 4), round(bz, 4)],
            "rot_y": round(rot_y, 2),
            "look_at": [round(lx, 4), round(ly, 4), round(lz, 4)],
            "chest": chest,
            "hand_R": hr,
            "hand_L": hl,
        },
        "gun": {
            "pos": gpos,
            "rot": grot,
        },
        "crater_pos": crater_pos,
        "splinters": splinters,
        "camera": {
            "pos": [round(cam_x, 4), round(cam_y, 4), round(cam_z, 4)],
            "target": [round(tar_x, 4), round(tar_y, 4), round(tar_z, 4)],
            "fov": 27.0,
        },
    }


@sparse_generated_scene
def build_seedance_snow_flinch_11s_scene() -> Dict[str, Any]:
    """
    构建《雪林惊吓躲避｜Seedance 提示词》11.0 秒一镜到底 SU 极简彩色预演场景（992×432 · 24fps · 264 帧）：
    - 0–6s：中景维持贴腮瞄准画右 + 枪口细微修正，树干完整无破损，摄影机小幅缓慢 Dolly In；
    - t=6.0s：明确锁定画外枪击节点，近旁树干出现弹坑、木屑飞散下落，老人缩肩弯腰低头躲避并收低步枪；
    - 7.1–11.0s：弯腰短暂停顿后缓慢抬头看向画右，摄影机沿浅弧推进至侧面 3/4 脸部特写。
    """
    stage = Stage()
    stage.clear()
    stage.duration = 11.0
    stage.fps = 24
    stage.render_mode = "su_minimal"
    stage.visual_style = "su_minimal"
    stage.environment["ground"] = SU_MINIMAL_CONFIG["ground_color"]
    stage.environment["sky"] = SU_MINIMAL_CONFIG["sky_color"]

    stage.add_box(
        name="林间雪道",
        pos=[0.0, 0.008, -6.0],
        size=[4.2, 0.016, 28.0],
        color=SU_MINIMAL_CONFIG["road_color"],
        obj_id="env_road_main",
    )

    cone_trees = [
        ("tree_cover_fg", "近旁中弹掩体松锥", 0.42, -0.22, 0.64, 4.6, "#4B7356"),
        ("tree_fg_left", "左侧边缘松锥", -1.25, -0.80, 0.85, 5.0, "#537E5F"),
        ("tree_mid_l1", "左中景松锥1", -0.82, -2.60, 0.96, 5.2, "#5E8C6A"),
        ("tree_mid_c1", "中景偏左松锥", -0.18, -3.80, 1.05, 5.5, "#537E5F"),
        ("tree_mid_r1", "右中景松锥1", 0.92, -2.40, 0.92, 5.0, "#689674"),
        ("tree_mid_r2", "右中景松锥2", 1.55, -3.90, 1.12, 5.6, "#5E8C6A"),
        ("tree_far_l1", "左远景松锥1", -1.95, -5.80, 1.30, 6.0, "#4B7356"),
        ("tree_far_c1", "中远景松锥1", 0.35, -6.40, 1.35, 6.2, "#5E8C6A"),
        ("tree_far_r1", "右远景松锥1", 1.85, -6.80, 1.40, 6.3, "#537E5F"),
        ("tree_deep_l", "纵深左松锥", -1.10, -9.20, 1.55, 6.8, "#689674"),
        ("tree_deep_c", "纵深中松锥", 0.65, -10.50, 1.65, 7.0, "#4B7356"),
        ("tree_deep_r", "纵深右松锥", 2.30, -11.40, 1.70, 7.2, "#5E8C6A"),
    ]
    for tid, tname, tx, tz, tw, th, tcol in cone_trees:
        stage.add_tree_cone(name=tname, x=tx, z=tz, width=tw, height=th, color=tcol, obj_id=tid)

    # 第 6 秒树干弹坑标记与 3 块飞散木屑（0–6s 隐藏在地下，6.0s 树干中弹瞬间触发）
    crater_box = stage.add_box(
        name="第6秒树干弹坑",
        pos=[0.45, -5.0, -0.02],
        size=[0.07, 0.11, 0.06],
        color="#B88352",
        obj_id="prop_tree_crater",
    )
    splinter_objs = [
        stage.add_box(f"飞散木屑_{idx_s+1}", pos=[0.46, -5.0, -0.02], size=[0.04, 0.03, 0.03], color="#C89B68", obj_id=f"prop_splinter_{idx_s+1}")
        for idx_s in range(3)
    ]

    s0 = evaluate_seedance_snow_flinch_11s_at_time(0.0)
    boris = stage.add_character(
        name="持枪老人(暖橙主角)",
        pos=s0["boris"]["pos"],
        rot_y=s0["boris"]["rot_y"],
        pose="standing",
        color="#D97A36",
        char_id="char_boris",
    )
    rifle = stage.add_gun_minimal(
        name="老人步枪(3部件)",
        pos=s0["gun"]["pos"],
        rot=s0["gun"]["rot"],
        color="#4A5568",
        obj_id="rifle_boris",
    )

    # 精简语义关键帧（仅保留动作转折与节奏节点，中间速度与过渡由引擎自动推算）：
    # 0.0s: 开场贴腮瞄准
    # 2.0s -> 2.8s -> 3.6s: 枪口细微修正并稳定
    # 6.0s: 画外枪击命中树干瞬间（瞄准截止）
    # 6.4s: 木屑向外飞散中段 / 缩肩急促下蹲中段
    # 7.15s: 受惊弯腰低头躲避到位（步枪降至胸前）
    # 7.95s: 略微弯腰短暂停顿结束，准备抬头
    # 9.0s: 先抬起视线过渡帧
    # 10.0s: 转头看向画右到位，推至侧面 3/4 特写
    # 11.0s: 结尾保持观察停顿与极轻微推进余速
    semantic_times = [0.0, 2.0, 2.8, 3.6, 6.0, 6.4, 7.15, 7.95, 9.0, 10.0, 11.0]

    cam_kfs = []
    for kt in semantic_times:
        st = evaluate_seedance_snow_flinch_11s_at_time(kt)
        b_st = st["boris"]
        t_b = {
            "pelvis": [0.0, 0.48, -0.02],
            "head_target": b_st["look_at"],
            "chest": b_st["chest"],
            "hand_R": b_st["hand_R"],
            "hand_L": b_st["hand_L"],
            "foot_R": [-0.12, 0.03, -0.04],
            "foot_L": [0.12, 0.03, 0.06],
            "knee_R_pole": [-0.14, 0.48, 0.45],
            "knee_L_pole": [0.14, 0.48, 0.45],
            "elbow_R_pole": [-0.32, 0.72, 0.08],
            "elbow_L_pole": [0.20, 0.68, 0.14],
        }
        if kt == 0.0:
            boris.targets = copy.deepcopy(t_b)
        boris.add_keyframe(
            kt,
            pos=b_st["pos"],
            rot_y=b_st["rot_y"],
            look_at=b_st["look_at"],
            targets=t_b,
            interp="smooth",
        )
        g_st = st["gun"]
        rifle.add_keyframe(kt, pos=g_st["pos"], rot=g_st["rot"], interp="smooth")
        c_st = st["camera"]
        cam_kfs.append((kt, c_st["pos"], c_st["target"], c_st["fov"]))

    # 弹坑与木屑仅在 [0.0s, 5.96s, 6.00s, 6.35s, 6.80s, 7.35s] 打稀疏关键帧
    crater_times = [0.0, 5.96, 6.00, 11.0]
    for kt in crater_times:
        st = evaluate_seedance_snow_flinch_11s_at_time(kt)
        crater_box.add_keyframe(kt, pos=st["crater_pos"], rot=[0.0, 0.0, 0.0], interp="hold" if kt < 6.0 else "linear")

    splinter_times = [0.0, 5.96, 6.00, 6.35, 6.80, 7.35, 7.50]
    for kt in splinter_times:
        st = evaluate_seedance_snow_flinch_11s_at_time(kt)
        for idx_s, sp_obj in enumerate(splinter_objs):
            sp_obj.add_keyframe(
                kt,
                pos=st["splinters"][idx_s],
                rot=[kt * 120.0, kt * 85.0, 0.0],
                interp="hold" if kt < 6.0 else "linear",
            )

    stage.set_camera(
        pos=cam_kfs[0][1],
        target=cam_kfs[0][2],
        fov=cam_kfs[0][3],
        aspect_ratio="2.39:1",
        export_width=992,
        export_height=432,
    )
    for kt, cpos, ctar, cfov in cam_kfs:
        stage.add_camera_keyframe(kt, pos=cpos, target=ctar, fov=cfov, interp="continuous")

    scene_dict = convert_scene_to_su_minimal(stage.to_dict())
    ref_v9_path = "/Users/wangmu/.gemini/antigravity/scratch/director-stage/ref_videos/video9_ref_12s.mp4"
    if os.path.isfile(ref_v9_path):
        scene_dict["ref_video"] = {
            "path": ref_v9_path,
            "filename": "video9_ref_12s.mp4",
            "fps": 24.0,
            "duration": 11.0,
            "width": 992,
            "height": 432,
            "time_offset": 0.0,
        }
    return scene_dict



def evaluate_wechat_two_hunters_aim_left_6s_at_time(t: float) -> Dict[str, Any]:
    """
    微信视频 e9d3818fdde08447e4109db8c7aca189.mp4（6.08 秒，992×432，24fps）：
    《雪林双人树后持枪向画左瞄准（前景右侧猎人贴腮特写 + 中景老猎人树后探枪）》
    - 全程遵循 SU 极简彩色预演规则（树单锥体、枪 3 基础体、球头人偶、稳定身份色）与精简关键帧规则。
    - 空间构图与遮挡关系：
      1) 右侧前景树锥 (`tree_fg_right`, x=+0.45, z=+0.18)：位于画面最右缘 (u≈0.86..0.98)，遮挡前景猎人右侧帽沿与右肩外侧。
      2) 前景持枪猎人 (`char_boris`, 暖橙 #D97A36, x=+0.22, z=0.00)：位于画面右侧 (头部 u≈0.70, v≈0.26)，贴腮握持步枪 (`rifle_boris`, #4A5568)，枪管向画面左下前方斜伸 (从 u≈0.68, v≈0.62 延伸至左下 u≈0.08, v≈0.92)，视线越过枪管紧盯画面左前方 (-X, +Z)。
      3) 左中景掩体树锥 (`tree_mid_left`, x=-0.16, z=-0.92)：竖直立于画面偏左位置 (u≈0.38)，形成前景与中景之间的标志性垂直分割。
      4) 中景持枪老猎人 (`char_hunter_mid`, 柔蓝 #4A78B8, x=-0.02, z=-1.28)：站在中景树锥右后方 (头部位于两树之间 u≈0.50, v≈0.28)，双手前伸托举步枪 (`rifle_mid`, #3B4452)，细长枪管从左中景树锥左侧水平探出指向画左 (-X，覆盖 u≈0.05..0.44, v≈0.36)。
      5) 摄影机：长焦压缩感 (FOV 25°)，0.0s–6.08s 极缓慢微幅向前推进 (Dolly In: z 从 1.54m 推至 1.44m)，两人随呼吸做极细微的枪口与视线微调。
    """
    tc = max(0.0, min(6.08, float(t)))
    def smoothstep(e0: float, e1: float, x: float) -> float:
        u = max(0.0, min(1.0, (x - e0) / max(1e-6, e1 - e0)))
        return u * u * (3.0 - 2.0 * u)
    def lerp(a: float, b: float, u: float) -> float:
        return a + (b - a) * u
    u_prog = smoothstep(0.0, 6.08, tc)

    # 细微呼吸与微调包络（支持在 4 个稀疏关键帧 0.0s / 2.0s / 4.2s / 6.08s 上精确采样并由引擎平滑插值）
    # 0.0s -> 2.0s: 轻微吸气抬枪 0.8cm；2.0s -> 4.2s: 呼气微沉 1.0cm；4.2s -> 6.08s: 眨眼微颌首并稳住
    if tc <= 2.0:
        b_wave = smoothstep(0.0, 2.0, tc)
    elif tc <= 4.2:
        b_wave = 1.0 - 1.3 * smoothstep(2.0, 4.2, tc)
    else:
        b_wave = -0.3 + 0.5 * smoothstep(4.2, 6.08, tc)

    nod_end = smoothstep(4.5, 5.4, tc) * (1.0 - 0.6 * smoothstep(5.4, 6.08, tc))

    # 1. 摄影机运镜：从后方人物（中景老猎人 char_hunter_mid）近景平滑拉出（Dolly Out）至现有双人错位尾帧
    #    标准站姿/探身持枪人物眼高约 1.44m，持枪高度约 1.24m：
    #    - t=0.0s：机位位于 z=-0.22（越过前景猎人左前方），单独构图后方老猎人树后抵肩探枪向画左瞄准的近景；
    #    - t=6.08s：机位向后拉出至 z=+1.46，前景右侧猎人（标准抵肩贴腮瞄准画左）与最右侧前景掩体树从画面右侧自然入画。
    cam_pos = [
        round(lerp(-0.100, -0.010, u_prog), 4),
        round(lerp(1.380, 1.375, u_prog), 4),
        round(lerp(-0.220, 1.460, u_prog), 4),
    ]
    cam_target = [
        round(lerp(-0.125, -0.030, u_prog), 4),
        round(lerp(1.355, 1.345, u_prog), 4),
        round(lerp(-1.280, -0.550, u_prog), 4),
    ]
    cam_fov = 25.0

    # 2. 前景猎人 (char_boris，右侧近景，使用拿枪姿势库 rifle_aim_cheek：抵肩贴腮向画左前方瞄准)
    base_fg_targets = copy.deepcopy(PRESET_IK_POSES["rifle_aim_cheek"]["targets"])
    fg_pos = [0.205, round(0.004 * b_wave, 4), 0.0]
    fg_rot_y = round(-52.0 + 0.8 * b_wave, 3)
    base_fg_targets["chest"] = [0.0, round(1.30 + 0.005 * b_wave, 4), 0.03]
    # 右手握机匣握把，左手托前护木（抵肩贴腮高度 y≈1.29m），枪管斜指画面左下方
    base_fg_targets["hand_R"] = [-0.08, round(1.29 + 0.004 * b_wave, 4), 0.21]
    base_fg_targets["hand_L"] = [-0.03, round(1.24 + 0.005 * b_wave, 4), 0.47]
    base_fg_targets["head_target"] = [
        -0.02,
        round(1.36 - 0.016 * nod_end + 0.004 * b_wave, 4),
        0.96,
    ]
    fg_char_tmp = {
        "pos": fg_pos,
        "rot": [0.0, fg_rot_y, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "torso_twist": PRESET_IK_POSES["rifle_aim_cheek"]["torso_twist"],
        "targets": base_fg_targets,
    }
    fg_gun_tf = compute_gun_transform_from_hands(fg_char_tmp)

    # 3. 中景老猎人 (char_hunter_mid，左中景树后使用拿枪姿势库 rifle_aim_tree_cover：探身抵肩向画左水平瞄准)
    mid_wave = -0.7 * b_wave
    base_mid_targets = copy.deepcopy(PRESET_IK_POSES["rifle_aim_tree_cover"]["targets"])
    mid_pos = [-0.02, round(0.004 * mid_wave, 4), -1.28]
    mid_rot_y = round(-82.0 + 0.6 * mid_wave, 3)
    base_mid_targets["chest"] = [-0.02, round(1.28 + 0.004 * mid_wave, 4), 0.06]
    base_mid_targets["hand_R"] = [-0.08, round(1.28 + 0.004 * mid_wave, 4), 0.23]
    base_mid_targets["hand_L"] = [-0.03, round(1.29 + 0.005 * mid_wave, 4), 0.49]
    base_mid_targets["head_target"] = [
        0.02,
        round(1.36 + 0.004 * mid_wave, 4),
        0.98,
    ]
    mid_char_tmp = {
        "pos": mid_pos,
        "rot": [0.0, mid_rot_y, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "torso_twist": PRESET_IK_POSES["rifle_aim_tree_cover"]["torso_twist"],
        "targets": base_mid_targets,
    }
    mid_gun_tf = compute_gun_transform_from_hands(mid_char_tmp)

    return {
        "camera": {"pos": cam_pos, "target": cam_target, "fov": cam_fov},
        "boris": {
            "pos": fg_pos,
            "rot_y": fg_rot_y,
            "torso_twist": PRESET_IK_POSES["rifle_aim_cheek"]["torso_twist"],
            "look_at": base_fg_targets["head_target"],
            "targets": base_fg_targets,
        },
        "gun_fg": {"pos": fg_gun_tf["pos"], "rot": fg_gun_tf["rot"]},
        "hunter_mid": {
            "pos": mid_pos,
            "rot_y": mid_rot_y,
            "torso_twist": PRESET_IK_POSES["rifle_aim_tree_cover"]["torso_twist"],
            "look_at": base_mid_targets["head_target"],
            "targets": base_mid_targets,
        },
        "gun_mid": {"pos": mid_gun_tf["pos"], "rot": mid_gun_tf["rot"]},
    }


@sparse_generated_scene
def build_wechat_two_hunters_aim_left_6s_scene() -> Dict[str, Any]:
    """
    构建微信视频 e9d3818fdde08447e4109db8c7aca189.mp4（6.08s，24fps，992×432）的 SU 极简彩色预演场景。
    严格采用精简语义关键帧（仅 4 个关键帧：0.0s, 2.0s, 4.2s, 6.08s），中间速度由引擎自动推算。
    """
    duration = 6.08
    stage = Stage()
    stage.clear()
    stage.duration = duration
    stage.fps = 24
    stage.render_mode = "su_minimal"
    stage.visual_style = "su_minimal"
    stage.environment["ground"] = SU_MINIMAL_CONFIG["ground_color"]
    stage.environment["sky"] = SU_MINIMAL_CONFIG["sky_color"]

    # 1. 浅灰雪地地面
    stage.add_box(
        name="雪林浅灰地面",
        pos=[0.0, -0.25, -4.0],
        size=[28.0, 0.5, 28.0],
        color="#D8DEE6",
        obj_id="env_snow_ground",
    )

    # 2. 森林单锥体树（2 棵核心前景/中景掩体树 + 远景稀疏森林锥体）
    cone_trees = [
        # 右侧前景掩体树：位于画面最右侧边缘 (u≈0.88..0.98)，紧贴前景猎人右侧而不遮挡脸部
        ("tree_fg_right", "右侧前景掩体松锥", 0.50, 0.08, 0.32, 6.5, "#4B7356"),
        # 左中景掩体树：位于画面偏左 (u≈0.32)，中景老猎人站在其右后方探枪瞄准画左，首帧近景不遮挡老猎人面部
        ("tree_mid_left", "左中景掩体松锥", -0.32, -1.16, 0.28, 6.8, "#537E5F"),
        # 远景稀疏背景松锥
        ("tree_bg_l1", "左远景松锥1", -1.45, -4.80, 0.95, 5.8, "#5E8C6A"),
        ("tree_bg_l2", "左远景松锥2", -0.85, -6.40, 1.10, 6.2, "#4B7356"),
        ("tree_bg_c1", "中远景松锥1", -0.28, -5.50, 0.92, 5.6, "#689674"),
        ("tree_bg_c2", "中远景松锥2", 0.32, -7.20, 1.15, 6.4, "#537E5F"),
        ("tree_bg_r1", "右远景松锥1", 1.10, -5.20, 1.05, 6.0, "#5E8C6A"),
        ("tree_deep_l", "纵深左松锥", -2.10, -9.00, 1.45, 6.8, "#537E5F"),
        ("tree_deep_c", "纵深中松锥", -0.55, -9.80, 1.55, 7.0, "#4B7356"),
    ]
    for tid, tname, tx, tz, tw, th, tcol in cone_trees:
        stage.add_tree_cone(name=tname, x=tx, z=tz, width=tw, height=th, color=tcol, obj_id=tid)

    s0 = evaluate_wechat_two_hunters_aim_left_6s_at_time(0.0)

    # 3. 前景右侧持枪猎人（暖橙 #D97A36）+ 前景步枪（#4A5568）
    boris = stage.add_character(
        name="前景右侧猎人(暖橙)",
        pos=s0["boris"]["pos"],
        rot_y=s0["boris"]["rot_y"],
        pose="rifle_aim_cheek",
        color="#D97A36",
        char_id="char_boris",
    )
    boris.torso_twist = s0["boris"].get("torso_twist", -22.0)
    rifle_fg = stage.add_gun_minimal(
        name="前景猎人步枪(3部件)",
        pos=s0["gun_fg"]["pos"],
        rot=s0["gun_fg"]["rot"],
        color="#4A5568",
        obj_id="rifle_boris",
    )
    rifle_fg.extra["bound_char_id"] = "char_boris"
    rifle_fg.extra["follow_bound_char"] = True

    # 4. 中景树后持枪老猎人（柔蓝 #4A78B8）+ 中景步枪（#3B4452）
    hunter_mid = stage.add_character(
        name="中景树后老猎人(柔蓝)",
        pos=s0["hunter_mid"]["pos"],
        rot_y=s0["hunter_mid"]["rot_y"],
        pose="rifle_aim_tree_cover",
        color="#4A78B8",
        char_id="char_hunter_mid",
    )
    hunter_mid.torso_twist = s0["hunter_mid"].get("torso_twist", -24.0)
    rifle_mid = stage.add_gun_minimal(
        name="中景老猎人步枪(3部件)",
        pos=s0["gun_mid"]["pos"],
        rot=s0["gun_mid"]["rot"],
        color="#3B4452",
        obj_id="rifle_mid",
    )
    rifle_mid.extra["bound_char_id"] = "char_hunter_mid"
    rifle_mid.extra["follow_bound_char"] = True

    # 5. 精简语义关键帧（仅 4 个节点：0.0s, 2.0s, 4.2s, 6.08s，中间速度由引擎自动推算）
    semantic_times = [0.0, 2.0, 4.2, 6.08]
    cam_kfs = []

    for kt in semantic_times:
        st = evaluate_wechat_two_hunters_aim_left_6s_at_time(kt)

        b_st = st["boris"]
        t_b = copy.deepcopy(b_st["targets"])
        if kt == 0.0:
            boris.targets = copy.deepcopy(t_b)
            boris.torso_twist = b_st.get("torso_twist", -22.0)
        boris.add_keyframe(
            kt,
            pos=b_st["pos"],
            rot_y=b_st["rot_y"],
            look_at=b_st["look_at"],
            targets=t_b,
            torso_twist=b_st.get("torso_twist", -22.0),
            interp="linear",
        )
        g_fg = st["gun_fg"]
        rifle_fg.add_keyframe(kt, pos=g_fg["pos"], rot=g_fg["rot"], interp="linear")

        m_st = st["hunter_mid"]
        t_m = copy.deepcopy(m_st["targets"])
        if kt == 0.0:
            hunter_mid.targets = copy.deepcopy(t_m)
            hunter_mid.torso_twist = m_st.get("torso_twist", -24.0)
        hunter_mid.add_keyframe(
            kt,
            pos=m_st["pos"],
            rot_y=m_st["rot_y"],
            look_at=m_st["look_at"],
            targets=t_m,
            torso_twist=m_st.get("torso_twist", -24.0),
            interp="linear",
        )
        g_mid = st["gun_mid"]
        rifle_mid.add_keyframe(kt, pos=g_mid["pos"], rot=g_mid["rot"], interp="linear")

        c_st = st["camera"]
        cam_kfs.append((kt, c_st["pos"], c_st["target"], c_st["fov"]))

    stage.set_camera(
        pos=cam_kfs[0][1],
        target=cam_kfs[0][2],
        fov=cam_kfs[0][3],
        aspect_ratio="2.39:1",
        export_width=992,
        export_height=432,
    )
    # 摄影机仅保留首尾 2 个关键帧（0.0s 后方老猎人近景 -> 6.08s 拉出现有双人错位尾帧），默认线性插值匀速推算
    for kt, cpos, ctar, cfov in [cam_kfs[0], cam_kfs[-1]]:
        stage.add_camera_keyframe(kt, pos=cpos, target=ctar, fov=cfov, interp="linear")

    scene_dict = convert_scene_to_su_minimal(stage.to_dict())
    ref_path = "/Users/wangmu/.gemini/antigravity/scratch/director-stage/ref_videos/wechat_e9d3_ref_6s.mp4"
    if os.path.isfile(ref_path):
        scene_dict["ref_video"] = {
            "path": ref_path,
            "filename": "wechat_e9d3_ref_6s.mp4",
            "fps": 24.0,
            "duration": duration,
            "width": 992,
            "height": 432,
            "time_offset": 0.0,
        }
    return scene_dict



@sparse_generated_scene
def build_scene08_snow_blood_tracking_scene(shot_mode: str = "shot1") -> Dict[str, Any]:
    """
    守护者 · 第 8 场 西伯利亚雪林追踪血迹（双镜头无缝衔接 2 × 20.00s @ 24fps, 1680×720 21:9 电影宽银幕）：
    - shot1 (20.00s, 480 帧): 高空俯瞰西伯利亚覆雪针叶林海、冰封蜿蜒河谷与远方连绵雪山 -> 顺滑穿过覆雪松冠垂直下潜 -> 轻柔静止在积雪渗透殷红血渍特写上。
    - shot2 (20.00s, 480 帧): 起幅与第 1 镜尾帧 100% 衔接 -> 蓝色尤里伸手捻起血雪 -> 低机位顺滑仰摇至尤里查看血雪并转头汇报“我们很近了” -> 连续 S 弧线升移右摇至右侧前景红色伊万与后景 5 人对峙 -> 伊万向画左前走出画露出正后方纵深第 6 人（青色） -> 马尔科夫与其余 5 名队员依次迈步向画左前走出画框，定格覆雪松林。
    - 8 人 8 色纯色身份色:
      * 蓝色 #3282EB: 尤里 (Yuri) —— 前景半蹲捻血雪汇报
      * 绿色 #37C35A: 马尔科夫 (Markov) —— 中景站立警戒并随队出画
      * 红色 #E13C3C: 伊万 (Ivan) —— 画右近景队长，率先向画左前出画
      * 黄色 #EBC32D / 橙色 #F07D23 / 紫色 #A54BE1 / 青色 #28CDD2 (第6人纵深) / 品红 #E14BA5: 5 名护林队员
    """
    is_shot2 = (shot_mode == "shot2")
    duration = 20.0
    fps = 24
    fov_deg = 32.0
    focal_mm = round(24.0 / (2.0 * math.tan(math.radians(fov_deg) * 0.5)))  # 42mm
    aspect_w, aspect_h = 1680, 720

    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _lerp(a: float, b: float, t_val: float) -> float:
        return a + (b - a) * t_val

    def _smootherstep(e0: float, e1: float, x: float) -> float:
        if abs(e1 - e0) < 1e-9:
            return 1.0 if x >= e1 else 0.0
        u = _clamp((x - e0) / (e1 - e0), 0.0, 1.0)
        return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)

    def _tilt_ease_c2(e0: float, e1: float, x: float) -> float:
        if abs(e1 - e0) < 1e-9:
            return 1.0 if x >= e1 else 0.0
        u = _clamp((x - e0) / (e1 - e0), 0.0, 1.0)
        om = 1.0 - u
        w = _clamp(u + 0.40 * u * om + 0.10 * u * om * om, 0.0, 1.0)
        return w * w * w * (w * (w * 6.0 - 15.0) + 10.0)

    def _walk_stride(t0: float, t1: float, t_val: float, accel_frac: float = 0.35) -> float:
        if t_val <= t0:
            return 0.0
        u = (t_val - t0) / max(t1 - t0, 1e-6)
        a = accel_frac
        vc = 1.0 / (1.0 - 0.5 * a)
        if u < a:
            return vc * ((u ** 3) / (a ** 2) - 0.5 * (u ** 4) / (a ** 3))
        return vc * (u - 0.5 * a)

    # 28.0s 摄影机空间 -> Three.js 世界坐标 (X_three = x_our, Z_three = -z_our)
    yaw28_rad = math.radians(24.80)
    cos28 = math.cos(yaw28_rad)
    sin28 = math.sin(yaw28_rad)
    tan_x28 = math.tan(math.radians(fov_deg * 0.5)) * (float(aspect_w) / float(aspect_h))

    def _cam28_to_three_xz(x_cam: float, z_cam: float):
        wx = 0.54 + x_cam * cos28 + z_cam * sin28
        wz_our = -0.16 - x_cam * sin28 + z_cam * cos28
        return round(wx, 4), round(-wz_our, 4)

    def _u28_to_three_xz(u_screen: float, z_cam: float):
        x_cam = (u_screen - 0.5) * 2.0 * tan_x28 * z_cam
        return _cam28_to_three_xz(x_cam, z_cam)

    def _is_past_left_frustum(x_cam: float, z_cam: float, radius: float) -> bool:
        if z_cam <= 0.14:
            return True
        return (x_cam + radius * 1.25) < (-z_cam * tan_x28 * 1.04)

    # 1. 雪原基底、殷红雪渗血迹贴花、近/背景松树树干群、高空针叶林海与冰封蜿蜒河谷远山
    seep_threads = [
        (-0.048, -0.030, 0.095, 0.060, "#B81018"),
        ( 0.024, -0.018, 0.088, 0.055, "#C8161E"),
        (-0.004,  0.020, 0.120, 0.044, "#D6242C"),
        ( 0.098, -0.016, 0.110, 0.046, "#DC3840"),
        (-0.138,  0.040, 0.085, 0.048, "#BE141C"),
        (-0.225,  0.080, 0.070, 0.040, "#CE222A"),
        (-0.330,  0.132, 0.080, 0.044, "#D8343C"),
        (-0.485,  0.215, 0.095, 0.050, "#E04A52"),
        (-0.680,  0.320, 0.110, 0.056, "#E66068"),
    ]
    blood_parts = []
    for b_idx, (bx_our, bz_our, sx_b, sz_b, b_col) in enumerate(seep_threads):
        blood_parts.append({
            "id": f"s08_blood_seep_{b_idx + 1}",
            "name": f"雪渗血丝_{b_idx + 1}",
            "shape": "cylinder",
            "ground_decal": True,
            "keep_color": True,
            "pos": [round(bx_our - (-0.15), 4), 0.002, round((-bz_our) - (-0.10), 4)],
            "size": [round(sx_b, 4), 0.003, round(sz_b, 4)],
            "rot": [0, 0, 0],
            "color": b_col,
        })

    # 近景林间空地围合松树 + 背景纵深松林树干 (共 43 棵)
    trunk_parts: List[Dict[str, Any]] = []
    aerial_parts: List[Dict[str, Any]] = []

    inner_clearing_trees = [
        (-2.25, -0.55, 11.5, 2.2, 1.15, 0.20, True),
        ( 2.35, -0.65, 11.5, 2.4, 1.10, 0.20, True),
        (-2.45,  2.60, 14.5, 1.6, 1.35, 0.22, False),
        (-1.65,  4.40, 16.5, 2.2, 1.25, 0.22, True),
        (-4.50,  0.60, 14.0, 2.2, 1.35, 0.23, False),
        (-3.80, -2.50, 12.5, 2.2, 1.30, 0.22, True),
        ( 0.10, -3.40, 12.0, 2.2, 1.25, 0.22, False),
        ( 3.80, -2.60, 12.5, 2.2, 1.30, 0.22, True),
        (-3.80,  4.20, 15.5, 1.8, 1.40, 0.23, True),
        (-2.35,  3.80, 15.5, 2.2, 1.20, 0.21, False),
    ]
    for idx_t, (tx_o, tz_o, th, cby, cr, tr, idark) in enumerate(inner_clearing_trees):
        tx3, tz3 = round(tx_o, 3), round(-tz_o, 3)
        trunk_h = round(th * 0.96 + 0.40, 3)
        trunk_parts.append({
            "id": f"s08_clr_trunk_{idx_t + 1}",
            "name": f"空地围合松干_{idx_t + 1}",
            "shape": "cylinder",
            "keep_color": True,
            "pos": [tx3, round(th * 0.48 - 0.20, 3), tz3],
            "size": [round(tr * 1.65, 3), trunk_h, round(tr * 1.65, 3)],
            "rot": [0, 0, 0],
            "color": "#33302E" if idark else "#45403D",
        })
        span_y = th - cby
        aerial_parts.extend([
            {
                "id": f"s08_clr_crown_b_{idx_t + 1}",
                "name": f"松冠底层_{idx_t + 1}",
                "shape": "cone",
                "keep_color": True,
                "pos": [tx3, round(cby + span_y * 0.24, 3), tz3],
                "size": [round(cr * 2.0, 3), round(span_y * 0.50, 3), round(cr * 2.0, 3)],
                "rot": [0, 0, 0],
                "color": "#2E4257",
            },
            {
                "id": f"s08_clr_crown_m_{idx_t + 1}",
                "name": f"松冠中层霜雪_{idx_t + 1}",
                "shape": "cone",
                "keep_color": True,
                "pos": [tx3, round(cby + span_y * 0.52, 3), tz3],
                "size": [round(cr * 1.64, 3), round(span_y * 0.44, 3), round(cr * 1.64, 3)],
                "rot": [0, 0, 0],
                "color": "#7A91AB",
            },
            {
                "id": f"s08_clr_crown_t_{idx_t + 1}",
                "name": f"松冠顶层覆雪_{idx_t + 1}",
                "shape": "cone",
                "keep_color": True,
                "pos": [tx3, round(cby + span_y * 0.80, 3), tz3],
                "size": [round(cr * 1.12, 3), round(span_y * 0.36, 3), round(cr * 1.12, 3)],
                "rot": [0, 0, 0],
                "color": "#A8BCD1",
            },
        ])

    bg_squad_u_z = [
        (-0.18,  9.50, 19.5, 0.23, True),  (-0.10, 11.20, 20.0, 0.24, False),
        (-0.03,  9.20, 19.5, 0.23, True),  ( 0.04, 10.80, 20.0, 0.24, True),
        ( 0.12,  9.40, 19.5, 0.22, False), ( 0.17, 11.40, 20.5, 0.25, True),
        ( 0.22, 11.00, 20.0, 0.23, False), ( 0.28, 12.20, 20.5, 0.24, True),
        ( 0.33, 11.20, 20.0, 0.22, False), ( 0.39, 11.80, 20.5, 0.24, True),
        ( 0.44, 12.60, 21.0, 0.24, False), ( 0.49, 11.40, 20.0, 0.23, True),
        ( 0.55, 11.80, 20.5, 0.24, False), ( 0.61, 11.20, 20.0, 0.23, True),
        ( 0.67, 12.00, 20.5, 0.24, False), ( 0.73, 11.00, 20.0, 0.23, True),
        ( 0.79, 10.40, 19.5, 0.23, False), ( 0.84,  9.20, 19.5, 0.24, True),
        ( 0.89, 11.20, 20.0, 0.23, False), ( 0.95,  9.60, 19.5, 0.24, True),
        ( 1.02, 10.80, 20.0, 0.23, False), (-0.14, 14.20, 21.0, 0.25, False),
        (-0.05, 15.00, 21.5, 0.26, True),  ( 0.06, 14.60, 21.0, 0.25, False),
        ( 0.15, 15.80, 21.5, 0.26, True),  ( 0.25, 14.80, 21.0, 0.25, False),
        ( 0.36, 15.60, 21.5, 0.26, True),  ( 0.47, 15.20, 21.0, 0.25, False),
        ( 0.58, 16.00, 21.5, 0.26, True),  ( 0.69, 15.00, 21.0, 0.25, False),
        ( 0.81, 14.60, 21.0, 0.25, True),  ( 0.92, 15.40, 21.0, 0.25, False),
        ( 1.04, 14.40, 20.5, 0.24, True),
    ]
    for idx_b, (u_tr, zc_tr, th, tr, idark) in enumerate(bg_squad_u_z):
        tx3, tz3 = _u28_to_three_xz(u_tr, zc_tr)
        trunk_parts.append({
            "id": f"s08_bg_trunk_{idx_b + 1}",
            "name": f"背景纵深松干_{idx_b + 1}",
            "shape": "cylinder",
            "keep_color": True,
            "pos": [tx3, round(th * 0.48 - 0.20, 3), tz3],
            "size": [round(tr * 1.65, 3), round(th * 0.96 + 0.40, 3), round(tr * 1.65, 3)],
            "rot": [0, 0, 0],
            "color": "#33302E" if idark else "#45403D",
        })
        span_y = th - 3.0
        aerial_parts.append({
            "id": f"s08_bg_crown_{idx_b + 1}",
            "name": f"背景松冠_{idx_b + 1}",
            "shape": "cone",
            "keep_color": True,
            "pos": [tx3, round(3.0 + span_y * 0.52, 3), tz3],
            "size": [2.60, round(span_y * 0.72, 3), 2.60],
            "rot": [0, 0, 0],
            "color": "#8EA3BA" if idx_b % 2 == 0 else "#A8BCD1",
        })

    # 蜿蜒冰封河谷 + 支流 + 远方连绵雪山 (Three.js: Z_three = -Z_our, rot_y = -yaw_deg)
    river_waypoints = [
        ( 32.0,  45.0, 22.0), ( 18.0,  95.0, 26.0), ( 46.0, 150.0, 30.0),
        ( 12.0, 215.0, 34.0), (-22.0, 280.0, 38.0), ( 35.0, 350.0, 42.0),
        ( 68.0, 430.0, 46.0), ( 22.0, 520.0, 52.0), ( 55.0, 620.0, 58.0),
    ]
    for r_i in range(len(river_waypoints) - 1):
        x0, z0, w0 = river_waypoints[r_i]
        x1, z1, w1 = river_waypoints[r_i + 1]
        dx, dz = x1 - x0, z1 - z0
        seg_len = round(math.hypot(dx, dz) + 10.0, 2)
        rot_y3 = round(-math.degrees(math.atan2(dx, dz)), 2)
        w_avg = 0.5 * (w0 + w1)
        y_st = r_i * 0.008
        cx_m, cz_m = round(0.5 * (x0 + x1), 2), round(-0.5 * (z0 + z1), 2)
        aerial_parts.append({
            "id": f"s08_river_snow_{r_i + 1}",
            "name": f"冰封河床覆雪段_{r_i + 1}",
            "shape": "box",
            "keep_color": True,
            "pos": [cx_m, round(0.05 + y_st, 3), cz_m],
            "size": [round(w_avg * 1.50, 2), 0.14, seg_len],
            "rot": [0, rot_y3, 0],
            "color": "#C4D3E5",
        })
        aerial_parts.append({
            "id": f"s08_river_ice_{r_i + 1}",
            "name": f"冰封河心蓝冰段_{r_i + 1}",
            "shape": "box",
            "keep_color": True,
            "pos": [round(cx_m + 3.0, 2), round(0.13 + y_st, 3), cz_m],
            "size": [round(w_avg * 0.55, 2), 0.18, seg_len],
            "rot": [0, rot_y3, 0],
            "color": "#5C7594",
        })

    trib_pts = [(-95.0, 125.0, 20.0), (-55.0, 195.0, 24.0), (-22.0, 280.0, 28.0)]
    for t_i in range(len(trib_pts) - 1):
        x0, z0, w0 = trib_pts[t_i]
        x1, z1, w1 = trib_pts[t_i + 1]
        dx, dz = x1 - x0, z1 - z0
        seg_len = round(math.hypot(dx, dz) + 8.0, 2)
        rot_y3 = round(-math.degrees(math.atan2(dx, dz)), 2)
        aerial_parts.append({
            "id": f"s08_river_trib_{t_i + 1}",
            "name": f"左翼支流雪河_{t_i + 1}",
            "shape": "box",
            "keep_color": True,
            "pos": [round(0.5 * (x0 + x1), 2), round(0.06 + t_i * 0.008, 3), round(-0.5 * (z0 + z1), 2)],
            "size": [round(0.5 * (w0 + w1), 2), 0.14, seg_len],
            "rot": [0, rot_y3, 0],
            "color": "#C4D3E5",
        })

    mtn_specs = [
        (-280.0, 680.0, 95.0, 18.0, False),
        (-160.0, 670.0, 85.0, 22.0, True),
        ( -40.0, 690.0, 90.0, 19.0, False),
        (  55.0, 660.0, 80.0, 26.0, True),
        ( 135.0, 645.0, 95.0, 44.0, True),
        ( 185.0, 655.0, 85.0, 38.0, True),
        ( 265.0, 670.0, 90.0, 30.0, True),
        ( 350.0, 680.0, 95.0, 25.0, False),
    ]
    for m_i, (mx, mz_o, mr, mh, has_cap) in enumerate(mtn_specs):
        aerial_parts.append({
            "id": f"s08_mtn_base_{m_i + 1}",
            "name": f"远景雪山岩峰_{m_i + 1}",
            "shape": "cone",
            "keep_color": True,
            "pos": [mx, round(mh * 0.5, 2), -mz_o],
            "size": [round(mr * 2.0, 1), mh, round(mr * 2.0, 1)],
            "rot": [0, 0, 0],
            "color": "#5C738F",
        })
        if has_cap:
            aerial_parts.append({
                "id": f"s08_mtn_cap_{m_i + 1}",
                "name": f"远景雪山白雪峰顶_{m_i + 1}",
                "shape": "cone",
                "keep_color": True,
                "pos": [mx, round(mh * 0.68, 2), -(mz_o - 2.0)],
                "size": [round(mr * 1.24, 1), round(mh * 0.55, 2), round(mr * 1.24, 1)],
                "rot": [0, 0, 0],
                "color": "#C2D2E6",
            })

    # 2. 构建 Stage 与 8 名角色
    stage = Stage()
    stage.clear()
    stage.fps = fps
    stage.duration = duration
    stage.render_mode = "su_minimal"
    stage.visual_style = "su_minimal"
    stage.environment = {
        "preset": "su_minimal",
        "sky": "#ADC2D6",
        "ground": "#BACAD8",
        "fog_color": "#ADC2D6",
        "fog": False,
        "show_ground": True,
        "show_grid": False,
    }

    snow_ground = stage.add_box(
        "西伯利亚雪原大地 (1600m)",
        pos=[0.0, -0.21, -220.0],
        size=[1600.0, 0.40, 1600.0],
        color="#BACAD8",
        obj_id="s08_snow_ground",
    )
    snow_ground.extra["arch_role"] = "ground"
    snow_ground.extra["keep_color"] = True

    blood_obj = stage.add_box(
        "积雪渗透殷红血渍 (贴花)",
        pos=[-0.15, 0.004, -0.10],
        size=[2.00, 0.01, 1.00],
        color="#B81018",
        obj_id="s08_blood_decal",
    )
    blood_obj.obj_type = "group"
    blood_obj.locked = False
    blood_obj.extra["keep_color"] = True
    blood_obj.extra["procedural_decal"] = "snow_blood_seepage"
    blood_obj.extra["parts"] = blood_parts

    trunks_obj = stage.add_box(
        "雪林近景与背景松林树干群 (43棵)",
        pos=[0.0, 0.0, 0.0],
        size=[24.0, 20.0, 24.0],
        color="#3D3936",
        obj_id="s08_forest_trunks",
    )
    trunks_obj.obj_type = "group"
    trunks_obj.locked = True
    trunks_obj.extra["keep_color"] = True
    trunks_obj.extra["parts"] = trunk_parts
    if is_shot2:
        trunks_obj.keyframes = [
            {"t": 0.0, "pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "visible": False, "interp": "step"},
            {"t": 3.05, "pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "visible": True, "interp": "step"},
        ]
    else:
        trunks_obj.keyframes = [
            {"t": 0.0, "pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "visible": True, "interp": "step"},
            {"t": 16.50, "pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "visible": False, "interp": "step"},
        ]

    aerial_obj = stage.add_box(
        "高空针叶林海·冰封河谷与远山",
        pos=[0.0, 0.0, 0.0],
        size=[400.0, 44.0, 700.0],
        color="#8EA3BA",
        obj_id="s08_aerial_taiga_sea",
    )
    aerial_obj.obj_type = "group"
    aerial_obj.locked = True
    aerial_obj.extra["keep_color"] = True
    aerial_obj.extra["procedural_forest"] = "taiga_aerial_sea"
    aerial_obj.extra["visible"] = not is_shot2
    aerial_obj.extra["parts"] = aerial_parts
    if not is_shot2:
        aerial_obj.keyframes = [
            {"t": 0.0, "pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "visible": True, "interp": "step"},
            {"t": 16.50, "pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "visible": False, "interp": "step"},
        ]

    # 8 名纯色身份色角色（镜头一默认隐藏，镜头二按精确时间线入画出画）
    c_yuri = stage.add_character(
        "尤里 Yuri (蓝·半蹲捻血雪)",
        pos=[-0.27, 0.0, -0.76],
        rot_y=16.0,
        pose="crouching",
        color="#3282EB",
        body_style="mannequin",
        char_id="char_s08_yuri",
    )
    c_yuri.extra["keep_color"] = True

    mk_x0, mk_z0 = _cam28_to_three_xz(-0.687, 3.310)
    c_markov = stage.add_character(
        "马尔科夫 Markov (绿·中景警戒)",
        pos=[mk_x0, 0.0, mk_z0],
        rot_y=-155.0,
        pose="standing",
        color="#37C35A",
        body_style="pawn",
        char_id="char_s08_markov",
    )
    c_markov.extra["keep_color"] = True

    iv_x0, iv_z0 = _cam28_to_three_xz(0.150, 0.860)
    c_ivan = stage.add_character(
        "伊万 Ivan (红·右前景队长)",
        pos=[iv_x0, 0.0, iv_z0],
        rot_y=-155.0,
        pose="standing",
        color="#E13C3C",
        body_style="pawn",
        char_id="char_s08_ivan",
    )
    c_ivan.extra["keep_color"] = True

    squad_defs = [
        ("char_s08_sq1", "护林队员 #1 (黄·左远)",   "#EBC32D", -3.606,  6.54, 14.00, 16.85, -1.95, 1.80,  3.05),
        ("char_s08_sq2", "护林队员 #2 (橙·左中深)", "#F07D23", -2.833,  7.99, 14.15, 17.65, -1.35, 1.00,  3.05),
        ("char_s08_sq3", "护林队员 #3 (紫·中景深)", "#A54BE1", -0.646, 10.27, 14.35, 19.20, -1.05, 0.65,  3.05),
        ("char_s08_sq4", "护林队员 #4 (青·第6人纵深)", "#28CDD2",  1.687,  9.00, 14.45, 18.80, -1.05, 0.65, 11.30),
        ("char_s08_sq5", "护林队员 #5 (品红·右景)", "#E14BA5",  3.655,  6.80, 14.20, 18.20, -1.05, 0.65,  3.05),
    ]
    squad_chars = []
    for cid, cname, ccol, xcs, zcs, tw0, tw1, xce, zce, t_vis in squad_defs:
        sx0, sz0 = _cam28_to_three_xz(xcs, zcs)
        sc_obj = stage.add_character(
            cname,
            pos=[sx0, 0.0, sz0],
            rot_y=-155.0,
            pose="standing",
            color=ccol,
            body_style="pawn",
            char_id=cid,
        )
        sc_obj.extra["keep_color"] = True
        squad_chars.append((sc_obj, xcs, zcs, tw0, tw1, xce, zce, t_vis))

    if not is_shot2:
        for ch in [c_yuri, c_markov, c_ivan] + [item[0] for item in squad_chars]:
            ch.extra["visible"] = False
    else:
        # 尤里 (蓝) 捻血雪 -> 仰摇起身 -> 转头向右上方汇报“我们很近了” (精简关键拐点 8 帧 <= 10)
        yuri_times = [0.0, 0.85, 2.30, 3.10, 5.20, 6.40, 7.60, 10.30]
        for yt in yuri_times:
            if yt < 0.85 or yt >= 10.30:
                c_yuri.keyframes.append({
                    "t": yt,
                    "pos": [-0.27, 0.0, -0.76],
                    "rot_y": 16.0,
                    "visible": False,
                    "interp": "step",
                })
                continue
            reach = _smootherstep(0.85, 2.30, yt)
            u_tilt = _tilt_ease_c2(3.10, 7.60, yt)
            look_iv = _smootherstep(6.35, 7.75, yt)
            # 0.85..3.10s 前倾使右手自然伸入血渍中心 (-0.025, 0.048, +0.018)；3.10..7.60s 随仰摇坐起端详血雪并转头看伊万
            px = round(_lerp(-0.18, -0.27, u_tilt), 4)
            pz = round(_lerp(-0.44, -0.76, u_tilt), 4)
            rot_y_val = round(14.0 + 38.0 * look_iv, 2)
            if yt < 3.10:
                wx_h = _lerp(-0.36, -0.025, reach)
                wy_h = _lerp( 0.11,  0.048, reach)
                wz_h = _lerp( 0.22,  0.018, reach)
            else:
                wx_h = _lerp(-0.025, -0.17, u_tilt)
                wy_h = _lerp( 0.048,  0.58, u_tilt ** 1.85)
                wz_h = _lerp( 0.018, -0.48, u_tilt ** 1.25)
            # 转入尤里局部坐标 (近似小角度 yaw)
            ry_rad = math.radians(rot_y_val)
            dx_w, dz_w = wx_h - px, wz_h - pz
            lx_h = round(dx_w * math.cos(ry_rad) - dz_w * math.sin(ry_rad), 4)
            lz_h = round(dx_w * math.sin(ry_rad) + dz_w * math.cos(ry_rad), 4)
            pelvis_y = round(_lerp(0.42, 0.52, u_tilt), 4)
            chest_y = round(_lerp(0.56, 0.82, u_tilt), 4)
            chest_z = round(_lerp(0.16, 0.06, u_tilt), 4)
            look_world = [
                round(_lerp(-0.04, iv_x0, look_iv), 3),
                round(_lerp(0.12 + 0.50 * u_tilt, 1.38, look_iv), 3),
                round(_lerp(0.02 - 0.35 * u_tilt, iv_z0, look_iv), 3),
            ]
            t_yuri = {
                "pelvis": [0.0, pelvis_y, 0.0],
                "chest": [0.0, chest_y, chest_z],
                "head_target": [round(0.32 * look_iv, 3), round(chest_y + 0.24, 3), 0.55],
                "hand_R": [lx_h, round(wy_h, 4), lz_h],
                "hand_L": [0.20, round(pelvis_y - 0.04, 3), 0.22],
                "foot_L": [0.12, 0.04, 0.12],
                "foot_R": [-0.12, 0.04, -0.06],
                "knee_L_pole": [0.16, 0.36, 0.48],
                "knee_R_pole": [-0.14, 0.16, 0.42],
                "elbow_L_pole": [0.30, round(chest_y - 0.14, 3), 0.02],
                "elbow_R_pole": [-0.28, round(wy_h + 0.14, 3), round(lz_h - 0.14, 3)],
            }
            c_yuri.keyframes.append({
                "t": yt,
                "pos": [px, 0.0, pz],
                "rot_y": rot_y_val,
                "look_at": look_world,
                "targets": t_yuri,
                "visible": True,
                "interp": "smooth",
            })

        # 伊万 (红) 6.50s 入画静止 -> 12.60s 从容向画左前迈步出画 (5 帧 <= 10)
        ivan_sample_ts = [0.0, 6.50, 12.60, 13.55, 14.40]
        for it in ivan_sample_ts:
            iv_s = _walk_stride(12.60, 14.45, it)
            ix_c = _lerp(0.150, -0.58, iv_s)
            iz_c = _lerp(0.860,  0.28, iv_s)
            vis_iv = (it >= 6.50) and (not _is_past_left_frustum(ix_c, iz_c, 0.250))
            ix_w, iz_w = _cam28_to_three_xz(ix_c, iz_c)
            c_ivan.keyframes.append({
                "t": it,
                "pos": [ix_w, 0.0, iz_w],
                "rot_y": -155.0,
                "visible": vis_iv,
                "interp": "linear",
            })

        # 马尔科夫 (绿) 3.05s 可见 -> 13.85s 跟随向画左前出画 (5 帧 <= 10)
        mk_sample_ts = [0.0, 3.05, 13.85, 15.05, 16.25]
        for mt in mk_sample_ts:
            mk_s = _walk_stride(13.85, 16.25, mt)
            mxc = _lerp(-0.687, -1.05, mk_s)
            mzc = _lerp( 3.310,  0.65, mk_s)
            vis_mk = (mt >= 3.05) and (not _is_past_left_frustum(mxc, mzc, 0.235))
            mx_w, mz_w = _cam28_to_three_xz(mxc, mzc)
            c_markov.keyframes.append({
                "t": mt,
                "pos": [mx_w, 0.0, mz_w],
                "rot_y": -155.0,
                "visible": vis_mk,
                "interp": "linear",
            })

        # 5 名护林队员依次迈步向画左前走出画框 (每条轨道 <= 6 帧 <= 10)
        for sc_obj, xcs, zcs, tw0, tw1, xce, zce, t_vis in squad_chars:
            t_mid = round(0.5 * (tw0 + min(20.0, tw1)), 2)
            t_end = round(min(20.0, tw1), 2)
            sq_ts = sorted({0.0, t_vis, tw0, t_mid, t_end, 20.0})[:6]
            for st_t in sq_ts:
                sw_s = _walk_stride(tw0, tw1, st_t)
                sxc = _lerp(xcs, xce, sw_s)
                szc = _lerp(zcs, zce, sw_s)
                vis_sq = (st_t >= t_vis) and (not _is_past_left_frustum(sxc, szc, 0.225))
                sx_w, sz_w = _cam28_to_three_xz(sxc, szc)
                sc_obj.keyframes.append({
                    "t": st_t,
                    "pos": [sx_w, 0.0, sz_w],
                    "rot_y": -155.0,
                    "visible": vis_sq,
                    "interp": "linear",
                })

    # 3. 摄影机关键拐点 (每镜 9 帧 <= 10 帧，保留全部叙事节奏与落幅定格)
    pose_blood = (0.00, 0.50, -0.04,  0.00, -86.00)
    pose_yuri  = (0.02, 0.34, -0.34,  0.00,  12.80)
    pose_ivan  = (0.54, 1.40, -0.16, 24.80,  -1.00)

    def _eval_cvals(t_sec: float):
        if not is_shot2:
            if t_sec < 17.60:
                u = _clamp(t_sec / 17.60, 0.0, 1.0)
                om = 1.0 - u
                s_y = 0.18 * (1.0 - (om ** 3) * (1.0 + 2.2 * u)) + 0.82 * _smootherstep(0.0, 1.0, u)
                gamma = 0.46
                y0_g, y1_g = 92.00 ** gamma, 0.50 ** gamma
                cy = (y0_g + (y1_g - y0_g) * s_y) ** (1.0 / gamma)
                s_z = 1.0 - (om ** 6) * (1.0 + 3.4 * u)
                cz_o = -125.00 + (-0.04 - (-125.00)) * s_z
                up = _clamp(t_sec / 13.40, 0.0, 1.0)
                omp = 1.0 - up
                s_p = 0.08 * (1.0 - (omp ** 3) * (1.0 + 2.2 * up)) + 0.92 * _smootherstep(0.0, 1.0, up)
                cpitch = -12.20 + (-86.00 - (-12.20)) * s_p
                return (0.00, cy, cz_o, 0.00, cpitch)
            return pose_blood
        u_t = _tilt_ease_c2(3.10, 7.60, t_sec)
        u_p = _smootherstep(6.60, 11.30, t_sec)
        return tuple(
            pose_blood[i] + (pose_yuri[i] - pose_blood[i]) * u_t + (pose_ivan[i] - pose_yuri[i]) * u_p
            for i in range(5)
        )

    def _cvals_to_pos_target(cvals):
        cx, cy, cz_o, cyaw_deg, cpitch_deg = cvals
        p_rad = math.radians(cpitch_deg)
        y_rad = math.radians(cyaw_deg)
        fx = math.cos(p_rad) * math.sin(y_rad)
        fy = math.sin(p_rad)
        fz = -math.cos(p_rad) * math.cos(y_rad)
        d_look = 0.50
        cpos = [round(cx, 4), round(cy, 4), round(-cz_o, 4)]
        ctar = [round(cx + fx * d_look, 4), round(cy + fy * d_look, 4), round(-cz_o + fz * d_look, 4)]
        return cpos, ctar

    if not is_shot2:
        cam_ts = [0.0, 2.80, 5.50, 8.50, 11.00, 13.80, 16.20, 17.60, 20.0]
    else:
        cam_ts = [0.0, 3.10, 4.60, 6.20, 7.60, 9.00, 10.30, 11.50, 20.0]

    cam_kfs = []
    for ct in cam_ts:
        cpos, ctar = _cvals_to_pos_target(_eval_cvals(ct))
        cam_kfs.append((ct, cpos, ctar))

    stage.set_camera(
        pos=cam_kfs[0][1],
        target=cam_kfs[0][2],
        fov=fov_deg,
        aspect_ratio="21:9",
        export_width=aspect_w,
        export_height=aspect_h,
    )
    stage.camera["focal_mm"] = focal_mm
    for ct, cpos, ctar in cam_kfs:
        stage.add_camera_keyframe(ct, pos=cpos, target=ctar, fov=fov_deg, interp="linear")

    scene_dict = stage.to_dict()
    scene_dict["_su_minimal_normalized"] = True
    scene_dict["camera"]["focal_mm"] = focal_mm
    scene_dict["camera"]["aspect_ratio"] = "21:9"
    scene_dict["camera"]["export_width"] = aspect_w
    scene_dict["camera"]["export_height"] = aspect_h

    base_dir = os.path.dirname(os.path.abspath(__file__))
    mp4_name = (
        "scene08_shot02_blood_to_squad_20s_blockout.mp4"
        if is_shot2
        else "scene08_shot01_aerial_to_blood_20s_blockout.mp4"
    )
    ref_mp4_path = os.path.join(base_dir, "ref_videos", mp4_name)
    if os.path.isfile(ref_mp4_path):
        scene_dict["ref_video"] = {
            "path": ref_mp4_path,
            "filename": mp4_name,
            "fps": 24.0,
            "duration": duration,
            "width": aspect_w,
            "height": aspect_h,
            "time_offset": 0.0,
        }
    return scene_dict



@sparse_generated_scene
def build_five_men_bonfire_15s_scene() -> Dict[str, Any]:
    """15秒五人围火 · 白模方块预演（4m×4m 整体火堆木架立方体 + 顶部横卧静止人形轮廓 + 谢尔盖添柴与丹尼尔投鹿皮松手解绑 + 丹尼尔真实视锥暂出左画重入 + 谢尔盖后蹲起单次转身 + 五人错峰庄严落幅）。"""
    duration = 15.0
    fps = 24
    focal_mm = 48.0
    fov_deg = focal_mm_to_fov(focal_mm)
    aspect_w, aspect_h = 1280, 544

    stage = Stage()
    stage.clear()
    stage.duration = duration
    stage.fps = fps
    stage.render_mode = "clay_only"
    stage.visual_style = "clay_blockout"
    stage.global_body_style = "mannequin"
    stage.hide_edges = True
    stage.contact_shadows = True
    stage.environment = {
        "preset": "clay_studio_bonfire",
        "sky": "#69717D",
        "ground": "#A9B1BC",
        "fog_color": "#737B87",
        "fog": True,
        "fog_near": 16.0,
        "fog_far": 48.0,
        "fog_max": 0.82,
        "show_ground": True,
        "show_grid": False,
    }

    # 1. 地面受光底块（单个低饱和浅灰扁平方块）
    stage.add_box(
        "火堆地面受光底块",
        pos=[0.0, 0.002, -0.95],
        size=[6.40, 0.004, 6.40],
        color="#B6BEC9",
        obj_id="env_bonfire_ground_pool",
        arch_role="floor",
        ground_decal=True,
        keep_color=True,
        protected=True,
    )

    # 2. 火堆与木架整体：单个 4m × 4m 长方体占位（底部贴地 Y=0.00m，顶面 Y=0.56m，不拆分横杆立柱木柴）
    stage.add_box(
        "中央火堆与木架整体立方体 (4m×4m)",
        pos=[0.0, 0.28, -0.95],
        size=[4.00, 0.56, 4.00],
        color="#AEB6C2",
        obj_id="prop_bonfire_crib",
        arch_role="prop",
        keep_color=True,
        protected=True,
    )

    # 3. 4m×4m 立方体顶部承托的横卧、完全静止人形轮廓（按参考图方向：头在右侧 +X、脚靴朝上在左侧 -X，紧贴 Y=0.56m 顶面，与五名活动人物分开计数）
    supine_color = "#5E6775"
    supine_parts = [
        {"id": "supine_head",   "shape": "sphere", "pos": [ 0.72, 0.115,  0.00], "size": [0.23, 0.23, 0.22], "rot": [0, 0, -10], "color": supine_color, "keep_color": True},
        {"id": "supine_chest",  "shape": "box",    "pos": [ 0.34, 0.115,  0.00], "size": [0.42, 0.23, 0.38], "rot": [0, 0,   3], "color": supine_color, "keep_color": True},
        {"id": "supine_pelvis", "shape": "box",    "pos": [ 0.02, 0.100,  0.00], "size": [0.28, 0.20, 0.34], "rot": [0, 0,   0], "color": supine_color, "keep_color": True},
        {"id": "supine_arm_f",  "shape": "box",    "pos": [ 0.22, 0.085,  0.23], "size": [0.56, 0.13, 0.11], "rot": [0, 4,   5], "color": supine_color, "keep_color": True},
        {"id": "supine_arm_b",  "shape": "box",    "pos": [ 0.22, 0.085, -0.23], "size": [0.56, 0.13, 0.11], "rot": [0, -4,  5], "color": supine_color, "keep_color": True},
        {"id": "supine_leg_f",  "shape": "box",    "pos": [-0.42, 0.088,  0.09], "size": [0.68, 0.16, 0.15], "rot": [0, -2, -2], "color": supine_color, "keep_color": True},
        {"id": "supine_leg_b",  "shape": "box",    "pos": [-0.42, 0.088, -0.09], "size": [0.68, 0.16, 0.15], "rot": [0,  2, -2], "color": supine_color, "keep_color": True},
        {"id": "supine_foot_f", "shape": "box",    "pos": [-0.79, 0.135,  0.10], "size": [0.12, 0.24, 0.12], "rot": [0,  0, 16], "color": supine_color, "keep_color": True},
        {"id": "supine_foot_b", "shape": "box",    "pos": [-0.77, 0.130, -0.10], "size": [0.12, 0.23, 0.12], "rot": [0,  0, 14], "color": supine_color, "keep_color": True},
    ]
    stage.add_group(
        "火堆顶部横卧静止人形轮廓",
        pos=[0.0, 0.56, -0.46],
        size=[1.72, 0.25, 0.58],
        color=supine_color,
        parts=supine_parts,
        obj_id="prop_supine_human_silhouette",
        arch_role="prop",
        keep_color=True,
        protected=True,
        static_silhouette=True,
    )

    # 4. 沿用现有火焰占位（起幅火焰特写为主体，镜头拉开后辨认出立方体顶部横卧人形轮廓，部分被火焰遮挡）
    fl_base = stage.add_sphere(
        "火堆底部浅色火核",
        pos=[0.0, 0.72, 0.06],
        size=[0.70, 0.34, 0.58],
        color="#F3F5F8",
        obj_id="flame_base_core",
        arch_role="flame_volume",
        keep_color=True,
        unlit=True,
        flame_phase=0.4,
        protected=True,
    )
    fl_base.add_keyframe(0.00, pos=[0.00, 0.71, 0.06], size=[0.68, 0.32, 0.56], interp="continuous")
    fl_base.add_keyframe(3.60, pos=[0.00, 0.73, 0.06], size=[0.74, 0.36, 0.60], interp="continuous")
    fl_base.add_keyframe(10.00, pos=[0.00, 0.75, 0.06], size=[0.78, 0.40, 0.64], interp="continuous")
    fl_base.add_keyframe(15.00, pos=[0.00, 0.74, 0.06], size=[0.76, 0.38, 0.62], interp="continuous")

    fl_main = stage.add_cone(
        "中央主焰浅色动态体积",
        pos=[0.01, 1.12, 0.05],
        size=[0.54, 1.12, 0.48],
        color="#F7F8FA",
        rot=[-2.0, 0.0, 1.5],
        obj_id="flame_main_spire",
        arch_role="flame_volume",
        keep_color=True,
        unlit=True,
        flame_phase=1.1,
        protected=True,
    )
    fl_main.add_keyframe(0.00, pos=[0.01, 1.06, 0.05], rot=[-2.5, 0.0, 1.5], size=[0.52, 1.00, 0.46], interp="continuous")
    fl_main.add_keyframe(3.60, pos=[0.00, 1.14, 0.05], rot=[-2.0, 0.0, -2.0], size=[0.56, 1.16, 0.50], interp="continuous")
    fl_main.add_keyframe(10.20, pos=[0.01, 1.24, 0.05], rot=[-2.5, 0.0, 1.5], size=[0.60, 1.36, 0.54], interp="continuous")
    fl_main.add_keyframe(15.00, pos=[0.00, 1.22, 0.05], rot=[-2.0, 0.0, 1.0], size=[0.58, 1.32, 0.52], interp="continuous")

    fl_left = stage.add_cone(
        "左前浅色副焰占位",
        pos=[-0.19, 0.90, 0.14],
        size=[0.32, 0.68, 0.30],
        color="#E9ECF1",
        rot=[4.0, 10.0, 7.0],
        obj_id="flame_left_tongue",
        arch_role="flame_volume",
        keep_color=True,
        unlit=True,
        flame_phase=2.5,
        protected=True,
    )
    fl_left.add_keyframe(0.00, pos=[-0.18, 0.88, 0.14], rot=[4.0, 10.0, 7.0], size=[0.30, 0.64, 0.28], interp="continuous")
    fl_left.add_keyframe(3.80, pos=[-0.19, 0.94, 0.14], rot=[4.5, 10.0, 6.0], size=[0.34, 0.76, 0.32], interp="continuous")
    fl_left.add_keyframe(10.00, pos=[-0.20, 0.99, 0.14], rot=[3.5, 11.0, 6.0], size=[0.36, 0.86, 0.34], interp="continuous")
    fl_left.add_keyframe(15.00, pos=[-0.19, 0.98, 0.14], rot=[4.0, 10.0, 7.0], size=[0.35, 0.84, 0.33], interp="continuous")

    fl_right = stage.add_cone(
        "右侧浅色副焰占位",
        pos=[0.20, 0.93, 0.04],
        size=[0.34, 0.74, 0.32],
        color="#E3E7EE",
        rot=[-3.5, -12.0, -8.0],
        obj_id="flame_right_tongue",
        arch_role="flame_volume",
        keep_color=True,
        unlit=True,
        flame_phase=4.2,
        protected=True,
    )
    fl_right.add_keyframe(0.00, pos=[0.19, 0.90, 0.04], rot=[-3.5, -12.0, -8.0], size=[0.32, 0.68, 0.30], interp="continuous")
    fl_right.add_keyframe(9.80, pos=[0.21, 1.02, 0.04], rot=[-3.0, -12.0, -7.0], size=[0.37, 0.92, 0.35], interp="continuous")
    fl_right.add_keyframe(15.00, pos=[0.20, 1.00, 0.04], rot=[-3.5, -12.0, -8.0], size=[0.36, 0.88, 0.34], interp="continuous")

    # 5. 五名活动人物（标准四肢可动 IK 关节人偶 body_style="mannequin"，沿用固定身份色、身高尺度与已确认的 15s 落点，精准还原参考图姿态；关闭步态算法，稀疏关键帧平滑插值 <= 5 帧/10s）
    def _ckf(
        t: float,
        pos: List[float],
        rot_y: float,
        pelvis: List[float],
        chest: List[float],
        head_target: List[float],
        hand_L: List[float],
        hand_R: List[float],
        foot_L: Optional[List[float]] = None,
        foot_R: Optional[List[float]] = None,
        knee_L_pole: Optional[List[float]] = None,
        knee_R_pole: Optional[List[float]] = None,
        elbow_L_pole: Optional[List[float]] = None,
        elbow_R_pole: Optional[List[float]] = None,
        shoulder_L_offset: Optional[List[float]] = None,
        shoulder_R_offset: Optional[List[float]] = None,
        interp: str = "ease_in_out",
    ) -> Dict[str, Any]:
        fL = foot_L if foot_L is not None else [0.10, 0.04, 0.00]
        fR = foot_R if foot_R is not None else [-0.10, 0.04, 0.00]
        kL = knee_L_pole if knee_L_pole is not None else [0.10, 0.46, 0.40]
        kR = knee_R_pole if knee_R_pole is not None else [-0.10, 0.46, 0.40]
        eL = elbow_L_pole if elbow_L_pole is not None else [0.32, 1.04, -0.18]
        eR = elbow_R_pole if elbow_R_pole is not None else [-0.32, 1.04, -0.18]
        sL = shoulder_L_offset if shoulder_L_offset is not None else [0.0, 0.0, 0.0]
        sR = shoulder_R_offset if shoulder_R_offset is not None else [0.0, 0.0, 0.0]
        return {
            "t": round(float(t), 3),
            "pos": [round(float(v), 4) for v in pos],
            "rot": [0.0, round(float(rot_y), 2), 0.0],
            "rot_y": round(float(rot_y), 2),
            "look_at": [round(float(v), 3) for v in head_target],
            "interp": interp,
            "interp_pos": interp,
            "interp_rot": interp,
            "interp_pose": interp,
            "targets": {
                "pelvis": [round(float(v), 4) for v in pelvis],
                "chest": [round(float(v), 4) for v in chest],
                "head_target": [round(float(v), 4) for v in head_target],
                "hand_L": [round(float(v), 4) for v in hand_L],
                "hand_R": [round(float(v), 4) for v in hand_R],
                "foot_L": [round(float(v), 4) for v in fL],
                "foot_R": [round(float(v), 4) for v in fR],
                "knee_L_pole": [round(float(v), 4) for v in kL],
                "knee_R_pole": [round(float(v), 4) for v in kR],
                "elbow_L_pole": [round(float(v), 4) for v in eL],
                "elbow_R_pole": [round(float(v), 4) for v in eR],
                "shoulder_L_offset": [round(float(v), 4) for v in sL],
                "shoulder_R_offset": [round(float(v), 4) for v in sR],
            },
        }

    # A · 橙色丹尼尔：左站双手持宽扁鹿皮 -> 2.15~3.05s 向火堆中央送出松手 -> 3.05~4.15s 收回空手停顿 -> 4.15~10.80s 沿左侧路线向镜头缓行，真实视锥暂出窄画左框后随扩幅自然重入 -> 落左前景 [-1.1722, 0, 5.1] 3/4背影朝火堆站定（对位参考图：挺直站立，左脚微后、右脚微前，双臂自然垂于体侧偏前）
    char_A_kfs = [
        _ckf(0.00, [-2.12, 0.0, 0.16], 96.0,
             pelvis=[0.0, 0.87, 0.01], chest=[0.0, 1.32, 0.06], head_target=[0.0, 1.30, 0.78],
             hand_L=[0.20, 0.92, 0.34], hand_R=[-0.20, 0.92, 0.34],
             foot_L=[0.11, 0.04, -0.02], foot_R=[-0.11, 0.04, 0.04],
             elbow_L_pole=[0.34, 1.02, -0.10], elbow_R_pole=[-0.34, 1.02, -0.10], interp="linear"),
        _ckf(2.15, [-2.12, 0.0, 0.16], 96.0,
             pelvis=[0.0, 0.86, 0.02], chest=[0.0, 1.30, 0.08], head_target=[0.0, 1.26, 0.80],
             hand_L=[0.19, 0.91, 0.38], hand_R=[-0.19, 0.91, 0.38],
             foot_L=[0.11, 0.04, -0.02], foot_R=[-0.11, 0.04, 0.04],
             elbow_L_pole=[0.34, 1.00, -0.06], elbow_R_pole=[-0.34, 1.00, -0.06], interp="ease_in_out"),
        _ckf(3.05, [-2.12, 0.0, 0.16], 95.0,
             pelvis=[0.0, 0.82, 0.06], chest=[0.0, 1.21, 0.22], head_target=[0.0, 1.08, 0.86],
             hand_L=[0.18, 0.84, 0.58], hand_R=[-0.18, 0.84, 0.58],
             foot_L=[0.11, 0.04, -0.04], foot_R=[-0.11, 0.04, 0.08],
             elbow_L_pole=[0.32, 0.96, 0.04], elbow_R_pole=[-0.32, 0.96, 0.04], interp="ease_in_out"),
        _ckf(4.15, [-2.14, 0.0, 0.18], 35.0,
             pelvis=[0.0, 0.885, 0.00], chest=[0.0, 1.345, 0.015], head_target=[0.0, 1.52, 0.72],
             hand_L=[0.22, 0.79, 0.04], hand_R=[-0.22, 0.79, 0.04],
             foot_L=[0.11, 0.04, 0.02], foot_R=[-0.11, 0.04, -0.02], interp="ease_in_out"),
        _ckf(6.65, [-2.12, 0.0, 3.65], 20.0,
             pelvis=[0.0, 0.885, 0.00], chest=[0.0, 1.345, 0.015], head_target=[0.0, 1.52, 0.72],
             hand_L=[0.22, 0.79, 0.04], hand_R=[-0.22, 0.79, 0.04],
             foot_L=[0.11, 0.04, -0.03], foot_R=[-0.11, 0.04, 0.04], interp="linear"),
        _ckf(10.80, [-1.1722, 0.0, 5.10], 157.5,
             pelvis=[0.0, 0.886, 0.00], chest=[0.0, 1.346, 0.02], head_target=[-0.08, 1.52, 0.74],
             hand_L=[0.21, 0.79, 0.04], hand_R=[-0.22, 0.78, 0.07],
             foot_L=[0.11, 0.04, -0.05], foot_R=[-0.11, 0.04, 0.06],
             elbow_L_pole=[0.32, 1.05, -0.16], elbow_R_pole=[-0.33, 1.05, -0.14], interp="ease_in_out"),
        _ckf(15.00, [-1.1722, 0.0, 5.10], 157.5,
             pelvis=[0.0, 0.886, 0.00], chest=[0.0, 1.346, 0.02], head_target=[-0.08, 1.50, 0.74],
             hand_L=[0.21, 0.79, 0.04], hand_R=[-0.22, 0.78, 0.07],
             foot_L=[0.11, 0.04, -0.05], foot_R=[-0.11, 0.04, 0.06],
             elbow_L_pole=[0.32, 1.05, -0.16], elbow_R_pole=[-0.33, 1.05, -0.14], interp="linear"),
    ]

    # B · 蓝色谢尔盖：右蹲持细长木柴 -> 0.00~1.85s 低低送入火堆松手 -> 2.80s 收回空手蹲稳 -> 待丹尼尔离开后 4.95~6.35s 移重心伸腿站起 -> 沿右侧外通道缓行 -> 10.10s 缩步停稳在右中景 [1.55, 0.0, 2.1827] -> 10.10~11.90s 单次缓转身正侧面看火（对位参考图：双腿挺直左侧身朝火，双臂自然垂于体侧偏前）
    char_B_kfs = [
        _ckf(0.00, [2.28, 0.0, -0.20], -92.0,
             pelvis=[0.0, 0.40, -0.04], chest=[0.0, 0.77, 0.10], head_target=[0.0, 0.86, 0.78],
             hand_L=[0.18, 0.48, 0.24], hand_R=[-0.10, 0.74, 0.42],
             foot_L=[0.13, 0.04, 0.04], foot_R=[-0.13, 0.04, 0.06],
             knee_L_pole=[0.15, 0.46, 0.56], knee_R_pole=[-0.15, 0.46, 0.56],
             elbow_L_pole=[0.34, 0.64, -0.06], elbow_R_pole=[-0.34, 0.72, 0.02], interp="ease_in_out"),
        _ckf(1.85, [2.26, 0.0, -0.20], -92.0,
             pelvis=[0.0, 0.39, 0.00], chest=[0.0, 0.75, 0.20], head_target=[0.0, 0.78, 0.84],
             hand_L=[0.18, 0.47, 0.26], hand_R=[-0.08, 0.73, 0.62],
             foot_L=[0.13, 0.04, 0.04], foot_R=[-0.13, 0.04, 0.08],
             knee_L_pole=[0.15, 0.46, 0.58], knee_R_pole=[-0.15, 0.46, 0.58],
             elbow_L_pole=[0.34, 0.64, -0.04], elbow_R_pole=[-0.32, 0.72, 0.12], interp="ease_in_out"),
        _ckf(2.80, [2.28, 0.0, -0.20], -92.0,
             pelvis=[0.0, 0.40, -0.04], chest=[0.0, 0.77, 0.08], head_target=[0.0, 0.88, 0.78],
             hand_L=[0.17, 0.48, 0.26], hand_R=[-0.17, 0.48, 0.26],
             foot_L=[0.13, 0.04, 0.04], foot_R=[-0.13, 0.04, 0.04],
             knee_L_pole=[0.15, 0.46, 0.56], knee_R_pole=[-0.15, 0.46, 0.56],
             elbow_L_pole=[0.34, 0.64, -0.06], elbow_R_pole=[-0.34, 0.64, -0.06], interp="linear"),
        _ckf(4.95, [2.30, 0.0, -0.18], -82.0,
             pelvis=[0.0, 0.50, 0.04], chest=[0.0, 0.90, 0.14], head_target=[0.0, 1.02, 0.78],
             hand_L=[0.20, 0.56, 0.18], hand_R=[-0.20, 0.56, 0.18],
             foot_L=[0.12, 0.04, 0.02], foot_R=[-0.12, 0.04, 0.04],
             knee_L_pole=[0.14, 0.48, 0.50], knee_R_pole=[-0.14, 0.48, 0.50],
             elbow_L_pole=[0.34, 0.74, -0.10], elbow_R_pole=[-0.34, 0.74, -0.10], interp="ease_in_out"),
        _ckf(6.35, [2.74, 0.0, 0.35], -26.0,
             pelvis=[0.0, 0.888, 0.00], chest=[0.0, 1.348, 0.015], head_target=[0.0, 1.53, 0.72],
             hand_L=[0.21, 0.80, 0.04], hand_R=[-0.21, 0.80, 0.04],
             foot_L=[0.10, 0.04, 0.02], foot_R=[-0.10, 0.04, -0.02], interp="ease_in_out"),
        _ckf(10.10, [1.55, 0.0, 2.1827], -38.0,
             pelvis=[0.0, 0.888, 0.00], chest=[0.0, 1.348, 0.015], head_target=[0.0, 1.53, 0.72],
             hand_L=[0.21, 0.80, 0.04], hand_R=[-0.21, 0.80, 0.04],
             foot_L=[0.10, 0.04, 0.02], foot_R=[-0.10, 0.04, -0.02], interp="ease_in_out"),
        _ckf(11.90, [1.55, 0.0, 2.1827], -114.0,
             pelvis=[0.0, 0.888, 0.00], chest=[0.0, 1.348, 0.01], head_target=[0.0, 1.54, 0.76],
             hand_L=[0.20, 0.80, 0.05], hand_R=[-0.20, 0.80, 0.04],
             foot_L=[0.10, 0.04, 0.03], foot_R=[-0.10, 0.04, -0.02],
             elbow_L_pole=[0.29, 1.06, -0.10], elbow_R_pole=[-0.29, 1.06, -0.10], interp="ease_in_out"),
        _ckf(15.00, [1.55, 0.0, 2.1827], -114.0,
             pelvis=[0.0, 0.888, 0.00], chest=[0.0, 1.348, 0.01], head_target=[0.0, 1.53, 0.76],
             hand_L=[0.20, 0.80, 0.05], hand_R=[-0.20, 0.80, 0.04],
             foot_L=[0.10, 0.04, 0.03], foot_R=[-0.10, 0.04, -0.02],
             elbow_L_pole=[0.29, 1.06, -0.10], elbow_R_pole=[-0.29, 1.06, -0.10], interp="linear"),
    ]

    # C · 绿色亚历山大：7.90s~10.95s 从镜头右后画外走入两步至 [1.5729, 0, 5.16] -> 10.95s~12.45s 双手缓慢收至腹前交握（对位参考图：右前景高大挺直3/4左后侧背影，双肘微屈、双手收于小腹前方，头微朝左看火）
    char_C_kfs = [
        _ckf(0.00, [1.96, 0.0, 6.58], -164.0,
             pelvis=[0.0, 0.888, 0.00], chest=[0.0, 1.348, 0.01], head_target=[0.08, 1.54, 0.72],
             hand_L=[0.22, 0.80, 0.02], hand_R=[-0.22, 0.80, 0.02],
             foot_L=[0.11, 0.04, 0.02], foot_R=[-0.11, 0.04, -0.02], interp="linear"),
        _ckf(7.90, [1.96, 0.0, 6.58], -164.0,
             pelvis=[0.0, 0.888, 0.00], chest=[0.0, 1.348, 0.01], head_target=[0.08, 1.54, 0.72],
             hand_L=[0.22, 0.80, 0.02], hand_R=[-0.22, 0.80, 0.02],
             foot_L=[0.11, 0.04, 0.02], foot_R=[-0.11, 0.04, -0.02], interp="ease_in_out"),
        _ckf(10.95, [1.5729, 0.0, 5.16], -164.0,
             pelvis=[0.0, 0.888, 0.00], chest=[0.0, 1.348, 0.01], head_target=[0.10, 1.53, 0.72],
             hand_L=[0.21, 0.80, 0.03], hand_R=[-0.21, 0.80, 0.03],
             foot_L=[0.11, 0.04, 0.04], foot_R=[-0.11, 0.04, -0.03], interp="ease_in_out"),
        _ckf(12.45, [1.5729, 0.0, 5.16], -164.0,
             pelvis=[0.0, 0.888, 0.00], chest=[0.0, 1.348, 0.015], head_target=[0.10, 1.52, 0.74],
             hand_L=[0.06, 0.85, 0.22], hand_R=[-0.06, 0.85, 0.22],
             foot_L=[0.11, 0.04, 0.04], foot_R=[-0.11, 0.04, -0.03],
             elbow_L_pole=[0.30, 1.04, -0.10], elbow_R_pole=[-0.30, 1.04, -0.10], interp="ease_in_out"),
        _ckf(15.00, [1.5729, 0.0, 5.16], -164.0,
             pelvis=[0.0, 0.888, 0.00], chest=[0.0, 1.348, 0.015], head_target=[0.10, 1.52, 0.74],
             hand_L=[0.06, 0.85, 0.22], hand_R=[-0.06, 0.85, 0.22],
             foot_L=[0.11, 0.04, 0.04], foot_R=[-0.11, 0.04, -0.03],
             elbow_L_pole=[0.30, 1.04, -0.10], elbow_R_pole=[-0.30, 1.04, -0.10], interp="linear"),
    ]

    # D · 紫色安德烈：8.85s~11.65s 从最左画外沿短路线进入左中景 [-3.6904, 0, 1.4147] -> 11.65s~13.25s 站稳后缓蹲（对位参考图：紧凑深蹲朝右看火，臀部贴近脚跟后上方，双膝前屈，小臂与手搭在膝盖前上方）
    char_D_kfs = [
        _ckf(0.00, [-4.85, 0.0, 1.05], 78.0,
             pelvis=[0.0, 0.888, 0.00], chest=[0.0, 1.348, 0.01], head_target=[0.0, 1.52, 0.72],
             hand_L=[0.21, 0.80, 0.02], hand_R=[-0.21, 0.80, 0.02], interp="linear"),
        _ckf(8.85, [-4.85, 0.0, 1.05], 78.0,
             pelvis=[0.0, 0.888, 0.00], chest=[0.0, 1.348, 0.01], head_target=[0.0, 1.52, 0.72],
             hand_L=[0.21, 0.80, 0.02], hand_R=[-0.21, 0.80, 0.02], interp="ease_in_out"),
        _ckf(11.65, [-3.6904, 0.0, 1.4147], 85.0,
             pelvis=[0.0, 0.888, 0.00], chest=[0.0, 1.348, 0.015], head_target=[0.0, 1.50, 0.72],
             hand_L=[0.21, 0.80, 0.02], hand_R=[-0.21, 0.80, 0.02], interp="ease_in_out"),
        _ckf(13.25, [-3.6904, 0.0, 1.4147], 85.0,
             pelvis=[0.0, 0.29, 0.03], chest=[0.0, 0.68, 0.18], head_target=[0.0, 0.83, 0.86],
             hand_L=[0.13, 0.42, 0.42], hand_R=[-0.12, 0.43, 0.44],
             foot_L=[0.13, 0.04, 0.10], foot_R=[-0.13, 0.04, 0.13],
             knee_L_pole=[0.15, 0.45, 0.66], knee_R_pole=[-0.15, 0.45, 0.68],
             elbow_L_pole=[0.30, 0.54, 0.12], elbow_R_pole=[-0.30, 0.54, 0.14], interp="ease_in_out"),
        _ckf(15.00, [-3.6904, 0.0, 1.4147], 85.0,
             pelvis=[0.0, 0.29, 0.03], chest=[0.0, 0.68, 0.18], head_target=[0.0, 0.83, 0.86],
             hand_L=[0.13, 0.42, 0.42], hand_R=[-0.12, 0.43, 0.44],
             foot_L=[0.13, 0.04, 0.10], foot_R=[-0.13, 0.04, 0.13],
             knee_L_pole=[0.15, 0.45, 0.66], knee_R_pole=[-0.15, 0.45, 0.68],
             elbow_L_pole=[0.30, 0.54, 0.12], elbow_R_pole=[-0.30, 0.54, 0.14], interp="linear"),
    ]

    # E · 黄色尼古拉：6.90s~8.55s 在火堆右后侧向火堆靠近一小步停在 [4.6, 0, -0.35] -> 8.55s~9.85s 轻微低头看向火堆（对位参考图：双腿挺直朝左前方火堆站立，微沉肩低头凝视火堆，双手自然垂在大腿前侧）
    char_E_kfs = [
        _ckf(0.00, [5.05, 0.0, -0.46], -82.0,
             pelvis=[0.0, 0.888, 0.00], chest=[0.0, 1.348, 0.01], head_target=[0.0, 1.52, 0.72],
             hand_L=[0.20, 0.80, 0.03], hand_R=[-0.20, 0.80, 0.03], interp="linear"),
        _ckf(6.90, [5.05, 0.0, -0.46], -82.0,
             pelvis=[0.0, 0.888, 0.00], chest=[0.0, 1.348, 0.01], head_target=[0.0, 1.52, 0.72],
             hand_L=[0.20, 0.80, 0.03], hand_R=[-0.20, 0.80, 0.03], interp="ease_in_out"),
        _ckf(8.55, [4.60, 0.0, -0.35], -82.0,
             pelvis=[0.0, 0.888, 0.00], chest=[0.0, 1.348, 0.02], head_target=[0.0, 1.48, 0.72],
             hand_L=[0.20, 0.79, 0.05], hand_R=[-0.20, 0.79, 0.05],
             foot_L=[0.10, 0.04, 0.03], foot_R=[-0.10, 0.04, -0.02], interp="ease_in_out"),
        _ckf(9.85, [4.60, 0.0, -0.35], -82.0,
             pelvis=[0.0, 0.886, 0.00], chest=[0.0, 1.332, 0.045], head_target=[0.0, 1.24, 0.76],
             hand_L=[0.19, 0.78, 0.09], hand_R=[-0.19, 0.78, 0.09],
             foot_L=[0.10, 0.04, 0.03], foot_R=[-0.10, 0.04, -0.02],
             elbow_L_pole=[0.29, 1.04, -0.10], elbow_R_pole=[-0.29, 1.04, -0.10], interp="ease_in_out"),
        _ckf(15.00, [4.60, 0.0, -0.35], -82.0,
             pelvis=[0.0, 0.886, 0.00], chest=[0.0, 1.332, 0.045], head_target=[0.0, 1.24, 0.76],
             hand_L=[0.19, 0.78, 0.09], hand_R=[-0.19, 0.78, 0.09],
             foot_L=[0.10, 0.04, 0.03], foot_R=[-0.10, 0.04, -0.02],
             elbow_L_pole=[0.29, 1.04, -0.10], elbow_R_pole=[-0.29, 1.04, -0.10], interp="linear"),
    ]

    # 将原 pawn 比例转换为等效身高尺度的 mannequin 比例（保持五人真实头顶米制身高不变）
    mannequin_scale_ratio = CHARACTER_NEUTRAL_DIMENSIONS["pawn"][1] / CHARACTER_NEUTRAL_DIMENSIONS["mannequin"][1]
    actor_specs = [
        {
            "id": "char_A",
            "name": "丹尼尔 Daniel (橙·投鹿皮后先离场重入左前景)",
            "pos": [-1.1722, 0.0, 5.10],
            "rot": [0.0, 157.5, 0.0],
            "scale": [round(1.0485092569002876 * mannequin_scale_ratio, 6)] * 3,
            "color": "#E68A32",
            "pose_name": "standing",
            "ground_squat": False,
            "keyframes": char_A_kfs,
        },
        {
            "id": "char_B",
            "name": "谢尔盖 Sergei (蓝·右蹲添柴后起身退右中景转身)",
            "pos": [1.55, 0.0, 2.1827],
            "rot": [0.0, -114.0, 0.0],
            "scale": [round(1.038329555376984 * mannequin_scale_ratio, 6)] * 3,
            "color": "#3282EB",
            "pose_name": "standing",
            "ground_squat": True,
            "keyframes": char_B_kfs,
        },
        {
            "id": "char_C",
            "name": "亚历山大 Alexander (绿·右后走入两步收手腹前)",
            "pos": [1.5729, 0.0, 5.16],
            "rot": [0.0, -164.0, 0.0],
            "scale": [round(1.0179701523303764 * mannequin_scale_ratio, 6)] * 3,
            "color": "#37B76D",
            "pose_name": "standing",
            "ground_squat": False,
            "keyframes": char_C_kfs,
        },
        {
            "id": "char_D",
            "name": "安德烈 Andrei (紫·最左入画至左中景缓蹲扶膝)",
            "pos": [-3.6904, 0.0, 1.4147],
            "rot": [0.0, 85.0, 0.0],
            "scale": [round(1.0077904508070727 * mannequin_scale_ratio, 6)] * 3,
            "color": "#A46AE0",
            "pose_name": "standing",
            "ground_squat": True,
            "keyframes": char_D_kfs,
        },
        {
            "id": "char_E",
            "name": "尼古拉 Nikolai (黄·右后远景近火半步微低头)",
            "pos": [4.60, 0.0, -0.35],
            "rot": [0.0, -82.0, 0.0],
            "scale": [round(0.9976107492837689 * mannequin_scale_ratio, 6)] * 3,
            "color": "#E2BC35",
            "pose_name": "standing",
            "ground_squat": False,
            "keyframes": char_E_kfs,
        },
    ]

    for spec in actor_specs:
        actor = stage.add_character(
            spec["name"],
            pos=spec["pos"],
            rot_y=spec["rot"][1],
            pose=spec["pose_name"],
            scale=1.0,
            color=spec["color"],
            body_style="mannequin",
            char_id=spec["id"],
        )
        actor.scale = copy.deepcopy(spec["scale"])
        actor.targets = copy.deepcopy(spec["keyframes"][-1]["targets"])
        actor.keyframes = copy.deepcopy(spec["keyframes"])
        actor.pins = {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False}
        actor.extra.update({
            "head_dir_line": True,
            "ground_squat": bool(spec["ground_squat"]),
            "enable_walk_gait": False,
            "pawn_smooth_walk": False,
            "keep_color": True,
            "protected": True,
        })

    # 7. 解算谢尔盖木柴 (t=1.85s) 与丹尼尔鹿皮 (t=3.05s) 松手时刻真实世界位姿，保证松手解绑 0.0mm 无缝连续下落
    tmp_scene = stage.to_dict()

    # 7a) 谢尔盖的细长木柴方块
    t_rel_wood = 1.85
    cmap_w, ikmap_w, _ = evaluate_scene_characters_at_time(tmp_scene, t_rel_wood)
    sock_B_R = get_character_socket_transform(cmap_w["char_B"], ikmap_w["char_B"], "hand_R")
    wood_base_size = [0.09, 0.07, 0.72]
    wood_local_pos = [0.0, -0.01, 0.28]
    wood_local_rot = [8.0, 0.0, 0.0]
    sc_B = float(sock_B_R["scale"])
    wood_local_scale = [round(1.0 / sc_B, 5), round(1.0 / sc_B, 5), round(1.0 / sc_B, 5)]
    wood_rel_tf = socket_local_to_world(wood_local_pos, wood_local_rot, wood_local_scale, wood_base_size, sock_B_R)
    w_pos_rel = wood_rel_tf["pos"]
    w_rot_rel = wood_rel_tf["rot"]
    w_sz_rel = wood_rel_tf["size"]
    w_pos_land = [round(w_pos_rel[0] - 0.08, 4), 0.595, round(w_pos_rel[2], 4)]
    w_rot_land = [0.0, round(w_rot_rel[1], 2), 0.0]

    firewood_box = stage.add_box(
        "谢尔盖的细长木柴",
        pos=w_pos_rel,
        size=wood_base_size,
        color="#8C837A",
        rot=w_rot_rel,
        obj_id="prop_sergei_firewood",
        arch_role="prop",
        keep_color=True,
        protected=True,
        semantic_label="细长木柴",
        parent_binding={
            "enabled": True,
            "target_id": "char_B",
            "socket": "hand_R",
            "local_pos": wood_local_pos,
            "local_rot": wood_local_rot,
            "local_scale": wood_local_scale,
            "release_t": t_rel_wood,
        },
    )
    firewood_box.add_keyframe(0.00, pos=w_pos_rel, rot=w_rot_rel, size=w_sz_rel, interp="linear")
    firewood_box.add_keyframe(t_rel_wood, pos=w_pos_rel, rot=w_rot_rel, size=w_sz_rel, interp="ease_in")
    firewood_box.add_keyframe(2.25, pos=w_pos_land, rot=w_rot_land, size=wood_base_size, interp="linear", stop=True)
    firewood_box.add_keyframe(15.00, pos=w_pos_land, rot=w_rot_land, size=wood_base_size, interp="linear")

    # 7b) 丹尼尔手中的最后一张鹿皮：薄、宽的扁平方块代理（名称与属性保留语义标注“鹿皮”，不叠加到摄影机画面）
    t_rel_hide = 3.05
    cmap_h, ikmap_h, _ = evaluate_scene_characters_at_time(tmp_scene, t_rel_hide)
    sock_A_R = get_character_socket_transform(cmap_h["char_A"], ikmap_h["char_A"], "hand_R")
    hide_base_size = [0.82, 0.04, 0.56]
    p_wr_L = np.asarray(ikmap_h["char_A"]["joints"]["wrist_L"], dtype=np.float64)
    p_wr_R = np.asarray(ikmap_h["char_A"]["joints"]["wrist_R"], dtype=np.float64)
    hands_mid = 0.5 * (p_wr_L + p_wr_R)
    desired_hide_rel_pos = [round(float(hands_mid[0]) + 0.36, 4), round(float(hands_mid[1]) - 0.04, 4), round(float(hands_mid[2]), 4)]
    desired_hide_rel_rot = [4.0, 8.0, -5.0]
    hide_local_dict = world_to_socket_local(desired_hide_rel_pos, desired_hide_rel_rot, hide_base_size, hide_base_size, sock_A_R)
    hide_rel_tf = socket_local_to_world(
        hide_local_dict["local_pos"],
        hide_local_dict["local_rot"],
        hide_local_dict["local_scale"],
        hide_base_size,
        sock_A_R,
    )
    h_pos_rel = hide_rel_tf["pos"]
    h_rot_rel = hide_rel_tf["rot"]
    h_sz_rel = hide_rel_tf["size"]
    h_pos_land = [round(h_pos_rel[0] + 0.28, 4), 0.58, round(h_pos_rel[2] - 0.08, 4)]
    h_rot_land = [0.0, 10.0, 0.0]

    deerskin_box = stage.add_box(
        "鹿皮",
        pos=h_pos_rel,
        size=hide_base_size,
        color="#C4AE92",
        rot=h_rot_rel,
        obj_id="prop_daniel_deerskin",
        arch_role="prop",
        keep_color=True,
        protected=True,
        semantic_label="鹿皮",
        prop_category="整张柔软毛皮占位",
        parent_binding={
            "enabled": True,
            "target_id": "char_A",
            "socket": "hand_R",
            "local_pos": hide_local_dict["local_pos"],
            "local_rot": hide_local_dict["local_rot"],
            "local_scale": hide_local_dict["local_scale"],
            "release_t": t_rel_hide,
        },
    )
    deerskin_box.add_keyframe(0.00, pos=h_pos_rel, rot=h_rot_rel, size=h_sz_rel, interp="linear")
    deerskin_box.add_keyframe(t_rel_hide, pos=h_pos_rel, rot=h_rot_rel, size=h_sz_rel, interp="ease_in")
    deerskin_box.add_keyframe(3.55, pos=h_pos_land, rot=h_rot_land, size=hide_base_size, interp="linear", stop=True)
    deerskin_box.add_keyframe(15.00, pos=h_pos_land, rot=h_rot_land, size=hide_base_size, interp="linear")

    # 8. 摄影机：严格保持原有四个关键点、固定 48mm 焦距、2.35:1 宽银幕与平缓后拉减速停稳曲线
    stage.set_camera(
        pos=[0.0, 0.62, 1.32],
        target=[0.0, 0.54, 0.0],
        focal_mm=focal_mm,
        aspect_ratio="2.35:1",
        export_width=aspect_w,
        export_height=aspect_h,
    )
    stage.add_camera_keyframe(0.00, pos=[0.0, 0.62, 1.32], target=[0.0, 0.54, 0.0], focal_mm=focal_mm, fov=fov_deg, interp="continuous")
    stage.add_camera_keyframe(4.50, pos=[0.0, 1.12, 4.42], target=[0.0, 1.00, 0.0], focal_mm=focal_mm, fov=fov_deg, interp="continuous")
    stage.add_camera_keyframe(13.40, pos=[0.0, 1.10, 10.02], target=[0.0, 0.91, 0.0], focal_mm=focal_mm, fov=fov_deg, interp="continuous")
    stage.add_camera_keyframe(15.00, pos=[0.0, 1.12, 10.42], target=[0.0, 0.92, 0.0], focal_mm=focal_mm, fov=fov_deg, interp="continuous", stop=True)

    scene_dict = stage.to_dict()
    scene_dict["export_plan"] = {
        "confirmed": True,
        "confirmed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "priority_mode": "fast",
        "priority_label": "极速",
        "material_style": "matte_blockout_previz",
        "compositions": [
            {"t": 0.0, "label": "火焰特写起幅（中央火焰占位为主体，4m×4m火堆顶面承托横卧静止人形轮廓）"},
            {"t": 1.85, "label": "右蹲蓝色谢尔盖低送细长木柴入火并松手落下"},
            {"t": 3.05, "label": "左站橙色丹尼尔向火堆中央送出宽扁鹿皮并松手落于人形轮廓附近燃烧区"},
            {"t": 6.65, "label": "丹尼尔先离开沿左侧路线自然暂出窄画左缘，谢尔盖随后移重心伸腿起身"},
            {"t": 10.10, "label": "丹尼尔随扩幅重入左前景，谢尔盖达右中景准备单次转身，黄绿紫三人错峰运动"},
            {"t": 15.0, "label": "五人庄严落幅（左前橙、右前绿收手腹前、右中蓝面火、左中紫蹲扶膝、右后远黄微低头）"},
        ],
        "preserve_existing_scene": True,
        "use_area_and_visible_budget": True,
    }
    return scene_dict
    return scene_dict

def build_video13_cabin_to_lake_8s_scene() -> Dict[str, Any]:
    """视频13 · 木屋门口走向冰湖 8.06秒【固定机位原地右摇 Pan Right】白模预演（遵守一物一方块规范）"""
    duration = 8.06
    fps = 24
    focal_mm = 50.0
    fov_deg = focal_mm_to_fov(focal_mm)
    aspect_w, aspect_h = 1280, 548

    stage = Stage()
    stage.clear()
    stage.duration = duration
    stage.fps = fps
    stage.render_mode = "clay_only"
    stage.visual_style = "clay_blockout"
    stage.global_body_style = "mannequin"
    stage.hide_edges = True
    stage.contact_shadows = True
    stage.environment = {
        "preset": "clay_studio_frozen_lake",
        "sky": "#4F5A68",
        "ground": "#939EA9",
        "fog_color": "#5C6877",
        "fog": True,
        "fog_near": 14.0,
        "fog_far": 52.0,
        "fog_max": 0.85,
        "show_ground": True,
        "show_grid": False,
    }

    # 1. 左侧木屋主建筑：单个外包围长方体（位于男主起幅后左方，随固定机位向右摇镜自然向画左扫出）
    stage.add_box(
        "左侧木屋整体占位块",
        pos=[-2.65, 1.45, -2.55],
        size=[4.20, 2.90, 4.40],
        color="#6E7682",
        rot=[0.0, 8.0, 0.0],
        obj_id="env_cabin_main_block",
        arch_role="wall",
        keep_color=True,
        protected=True,
    )

    # 2. 起幅左侧前景门框遮挡块（关键前景遮挡，供 0~1.08s 男主右手扶靠，随右摇迅速从画左出画）
    stage.add_box(
        "木屋门口近景门框遮挡块",
        pos=[-0.52, 1.25, -1.16],
        size=[0.16, 2.50, 0.24],
        color="#484F59",
        obj_id="env_doorframe_occluder",
        arch_role="wall",
        keep_color=True,
        protected=True,
    )

    # 3. 右后方木棚：单个外包围长方体（位于 -Z 侧，起幅在男主右后方露出，随摄影机原地向右摇镜而与木屋同步向画左扫出）
    stage.add_box(
        "右后方残破木棚占位块",
        pos=[1.45, 1.10, -5.60],
        size=[2.70, 2.20, 2.40],
        color="#7C8592",
        rot=[0.0, -14.0, 0.0],
        obj_id="env_right_shed_block",
        arch_role="prop",
        keep_color=True,
        protected=True,
    )

    # 4. 远方冰湖彼岸山脊林线：单个低矮宽长方体压在 +X 远方地平线上（固定机位摇至 +X 后静止稳定横贯画面）
    stage.add_box(
        "冰湖远岸地平线占位块",
        pos=[36.0, 0.50, -0.15],
        size=[2.40, 1.00, 78.0],
        color="#5B6573",
        obj_id="env_far_shoreline_block",
        arch_role="prop",
        keep_color=True,
        protected=True,
    )

    # 5. 男主（四肢可动关节人偶 mannequin，关闭步态算法，稀疏关键帧 <= 5帧/8s）
    def _mkf(
        t: float,
        pos: List[float],
        rot_y: float,
        pelvis: List[float],
        chest: List[float],
        head_target: List[float],
        hand_L: List[float],
        hand_R: List[float],
        foot_L: List[float],
        foot_R: List[float],
        elbow_L_pole: Optional[List[float]] = None,
        elbow_R_pole: Optional[List[float]] = None,
        interp: str = "ease_in_out",
    ) -> Dict[str, Any]:
        eL = elbow_L_pole if elbow_L_pole is not None else [0.31, 1.04, -0.14]
        eR = elbow_R_pole if elbow_R_pole is not None else [-0.31, 1.04, -0.14]
        return {
            "t": round(float(t), 3),
            "pos": [round(float(v), 4) for v in pos],
            "rot": [0.0, round(float(rot_y), 2), 0.0],
            "rot_y": round(float(rot_y), 2),
            "look_at": [round(float(v), 3) for v in head_target],
            "interp": interp,
            "interp_pos": interp,
            "interp_rot": interp,
            "interp_pose": interp,
            "targets": {
                "pelvis": [round(float(v), 4) for v in pelvis],
                "chest": [round(float(v), 4) for v in chest],
                "head_target": [round(float(v), 4) for v in head_target],
                "hand_L": [round(float(v), 4) for v in hand_L],
                "hand_R": [round(float(v), 4) for v in hand_R],
                "foot_L": [round(float(v), 4) for v in foot_L],
                "foot_R": [round(float(v), 4) for v in foot_R],
                "knee_L_pole": [0.10, 0.46, 0.40],
                "knee_R_pole": [-0.10, 0.46, 0.40],
                "elbow_L_pole": [round(float(v), 4) for v in eL],
                "elbow_R_pole": [round(float(v), 4) for v in eR],
                "shoulder_L_offset": [0.0, 0.0, 0.0],
                "shoulder_R_offset": [0.0, 0.0, 0.0],
            },
        }

    # 男主真实走位（绕过固定摄影机 [0, 1.54, 0] 前侧近处擦身走过，再沿 +X 轴走向冰湖深处）：
    # 0.00s~1.08s：站木屋门口 [-0.04, 0.0, -1.52]（距固定机位 1.52m），面朝镜头偏右 rot_y=18°，右手扶左侧门框
    # 1.08s~3.76s：松开右手，从固定镜头右前切线近处 [0.96, 0.0, -0.34]（距镜头仅 1.02m，自然形成大比例紧凑右侧轮廓近景）走过并转向冰湖（rot_y -> 85°）
    # 3.76s~5.37s：走上固定摄影机正前方的 +X 轴线 [1.46, 0.0, -0.02]（距镜头 1.46m，正对冰湖 rot_y=90°，自然切换为居中正后背影）
    # 5.37s~8.06s：摄影机完全不动，男主自己在冰面上继续向前缓步走远至 [2.05, 0.0, -0.02]（靠真实近大远小使头肩在画框中平缓变小）
    hero_kfs = [
        _mkf(
            0.00, [-0.04, 0.0, -1.52], 18.0,
            pelvis=[0.0, 0.888, 0.0], chest=[0.0, 1.348, 0.015], head_target=[0.24, 1.53, 0.80],
            hand_L=[0.22, 0.79, 0.04], hand_R=[-0.46, 1.04, 0.18],
            foot_L=[0.11, 0.04, 0.02], foot_R=[-0.11, 0.04, -0.02],
            elbow_R_pole=[-0.38, 0.96, -0.10], interp="linear",
        ),
        _mkf(
            1.08, [0.02, 0.0, -1.46], 28.0,
            pelvis=[0.0, 0.888, 0.0], chest=[0.0, 1.348, 0.02], head_target=[0.26, 1.53, 0.80],
            hand_L=[0.22, 0.79, 0.04], hand_R=[-0.38, 0.96, 0.14],
            foot_L=[0.11, 0.04, 0.03], foot_R=[-0.11, 0.04, -0.02],
            elbow_R_pole=[-0.36, 0.96, -0.10], interp="ease_in_out",
        ),
        _mkf(
            3.76, [0.96, 0.0, -0.34], 84.0,
            pelvis=[0.0, 0.886, 0.0], chest=[0.0, 1.346, 0.025], head_target=[0.04, 1.52, 0.86],
            hand_L=[0.21, 0.79, 0.05], hand_R=[-0.21, 0.79, 0.05],
            foot_L=[0.11, 0.04, 0.04], foot_R=[-0.11, 0.04, -0.04], interp="ease_in_out",
        ),
        _mkf(
            5.37, [1.46, 0.0, -0.02], 90.0,
            pelvis=[0.0, 0.888, 0.0], chest=[0.0, 1.348, 0.02], head_target=[0.0, 1.53, 0.86],
            hand_L=[0.21, 0.79, 0.04], hand_R=[-0.21, 0.79, 0.04],
            foot_L=[0.11, 0.04, 0.03], foot_R=[-0.11, 0.04, -0.03], interp="linear",
        ),
        _mkf(
            8.06, [2.05, 0.0, -0.02], 90.0,
            pelvis=[0.0, 0.888, 0.0], chest=[0.0, 1.348, 0.015], head_target=[0.0, 1.53, 0.86],
            hand_L=[0.21, 0.79, 0.03], hand_R=[-0.21, 0.79, 0.03],
            foot_L=[0.11, 0.04, 0.02], foot_R=[-0.11, 0.04, -0.02], interp="ease_in_out",
        ),
    ]

    mannequin_scale_ratio = CHARACTER_NEUTRAL_DIMENSIONS["pawn"][1] / CHARACTER_NEUTRAL_DIMENSIONS["mannequin"][1]
    hero = stage.add_character(
        "男主 (扶门走出经镜头前走向冰湖)",
        pos=[2.05, 0.0, -0.02],
        rot_y=90.0,
        pose="standing",
        scale=1.0,
        color="#D98236",
        body_style="mannequin",
        char_id="char_v13_hero",
    )
    hero.scale = [round(1.035 * mannequin_scale_ratio, 6)] * 3
    hero.targets = copy.deepcopy(hero_kfs[-1]["targets"])
    hero.keyframes = copy.deepcopy(hero_kfs)
    hero.pins = {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False}
    hero.extra.update({
        "head_dir_line": True,
        "enable_walk_gait": False,
        "pawn_smooth_walk": False,
        "keep_color": True,
        "protected": True,
    })

    # 6. 摄影机轨迹：三脚架固定机位 pos=[0.0, 1.53, 0.0] 全程零位移！纯靠水平向右摇镜（Pan Right ~92°）追随人物，5.37s 后完全锁死定镜
    fixed_cam_pos = [0.00, 1.53, 0.00]
    stage.set_camera(
        pos=fixed_cam_pos,
        target=[0.04, 1.49, -5.00],
        focal_mm=focal_mm,
        aspect_ratio="2.35:1",
        export_width=aspect_w,
        export_height=aspect_h,
    )
    # t=0.00s: 镜头朝向 -Z（木屋门口），左露门框、右后露木棚
    stage.add_camera_keyframe(0.00, pos=fixed_cam_pos, target=[0.04, 1.49, -5.00], focal_mm=focal_mm, fov=fov_deg, interp="continuous")
    # t=1.08s: 男主准备松手起步，镜头极微起步预备右摇
    stage.add_camera_keyframe(1.08, pos=fixed_cam_pos, target=[0.32, 1.50, -4.98], focal_mm=focal_mm, fov=fov_deg, interp="continuous")
    # t=3.76s: 固定机位向右摇至约 70°（追随正从镜头前 1.02m 擦身走过的男主右侧面轮廓）
    stage.add_camera_keyframe(3.76, pos=fixed_cam_pos, target=[4.72, 1.53, -1.68], focal_mm=focal_mm, fov=fov_deg, interp="continuous")
    # t=5.37s: 完成约 91° 原地右摇，正对 +X 冰湖与男主正后背影并平缓停稳
    stage.add_camera_keyframe(5.37, pos=fixed_cam_pos, target=[10.00, 1.51, -0.02], focal_mm=focal_mm, fov=fov_deg, interp="continuous", stop=True)
    # t=8.06s: 机位与角度完全静止保持，靠男主自己向冰湖走远呈现背影渐小
    stage.add_camera_keyframe(8.06, pos=fixed_cam_pos, target=[10.00, 1.51, -0.02], focal_mm=focal_mm, fov=fov_deg, interp="linear", stop=True)

    scene_dict = stage.to_dict()
    scene_dict["export_plan"] = {
        "confirmed": True,
        "confirmed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "priority_mode": "fast",
        "priority_label": "极速",
        "material_style": "matte_blockout_previz",
        "compositions": [
            {"t": 0.00, "label": "固定机位起幅：朝-Z拍木屋门口胸上中近景，右手扶左侧门框，右后露木棚"},
            {"t": 2.15, "label": "原地右摇前半程：男主松手向镜头右前切线走近，木屋木棚向画左扫出"},
            {"t": 3.76, "label": "原地右摇70°：男主从距固定镜头1.02m切线走过，自然形成紧凑右侧轮廓近景"},
            {"t": 5.37, "label": "原地右摇91°停稳：男主走上+X冰湖中轴线，形成居中正后双肩背影"},
            {"t": 8.06, "label": "定镜落幅：摄影机纹丝不动，男主继续朝冰湖缓步走远，背影自然变小"},
        ],
        "preserve_existing_scene": True,
        "use_area_and_visible_budget": True,
    }
    return scene_dict


def build_pat_shoulder_dolly_15s_scene() -> Dict[str, Any]:
    """冰湖岸边双人过肩拍肩 -> 前景擦镜出画+云台微左摇 -> 滑轨慢推男主特写 15.07秒白模预演（遵守一物一方块规范）"""
    duration = 15.07
    fps = 24
    focal_mm = 65.0
    fov_deg = focal_mm_to_fov(focal_mm)
    aspect_w, aspect_h = 1280, 544

    stage = Stage()
    stage.clear()
    stage.duration = duration
    stage.fps = fps
    stage.render_mode = "clay_only"
    stage.visual_style = "clay_blockout"
    stage.global_body_style = "mannequin"
    stage.hide_edges = True
    stage.contact_shadows = True
    stage.environment = {
        "preset": "clay_studio_frozen_lake",
        "sky": "#465261",
        "ground": "#8B97A4",
        "fog_color": "#556272",
        "fog": True,
        "fog_near": 16.0,
        "fog_far": 56.0,
        "fog_max": 0.85,
        "show_ground": True,
        "show_grid": False,
    }

    # 1. 远方冰湖彼岸雪岭地平线：单个低矮宽长方体（位于 -Z 远岸，压在男主耳际高度附近，检验滑轨慢推时近大远小视差）
    stage.add_box(
        "冰湖远岸山脊占位块",
        pos=[-4.00, 0.65, -42.00],
        size=[86.00, 1.30, 2.60],
        color="#5A6473",
        obj_id="env_far_shoreline_ridge",
        arch_role="prop",
        keep_color=True,
        protected=True,
    )

    # 2. 左前挂灯木柱定位块：单个立柱长方体（0~8.4s 在左画框外；8.4~10.1s 随云台微左摇短暂进入最左缘；10.1~15.07s 随滑轨慢推自然出左画框）
    stage.add_box(
        "左前挂灯木柱占位块",
        pos=[-0.95, 1.15, -1.45],
        size=[0.14, 2.30, 0.14],
        color="#6B5E53",
        obj_id="env_left_lantern_post_block",
        arch_role="prop",
        keep_color=True,
        protected=True,
    )

    def _mkf(
        t: float,
        pos: List[float],
        rot_y: float,
        pelvis: List[float],
        chest: List[float],
        head_target: List[float],
        hand_L: List[float],
        hand_R: List[float],
        foot_L: List[float],
        foot_R: List[float],
        elbow_L_pole: Optional[List[float]] = None,
        elbow_R_pole: Optional[List[float]] = None,
        interp: str = "ease_in_out",
    ) -> Dict[str, Any]:
        eL = elbow_L_pole if elbow_L_pole is not None else [0.31, 1.04, -0.14]
        eR = elbow_R_pole if elbow_R_pole is not None else [-0.31, 1.04, -0.14]
        return {
            "t": round(float(t), 3),
            "pos": [round(float(v), 4) for v in pos],
            "rot": [0.0, round(float(rot_y), 2), 0.0],
            "rot_y": round(float(rot_y), 2),
            "look_at": [round(float(v), 3) for v in head_target],
            "interp": interp,
            "interp_pos": interp,
            "interp_rot": interp,
            "interp_pose": interp,
            "targets": {
                "pelvis": [round(float(v), 4) for v in pelvis],
                "chest": [round(float(v), 4) for v in chest],
                "head_target": [round(float(v), 4) for v in head_target],
                "hand_L": [round(float(v), 4) for v in hand_L],
                "hand_R": [round(float(v), 4) for v in hand_R],
                "foot_L": [round(float(v), 4) for v in foot_L],
                "foot_R": [round(float(v), 4) for v in foot_R],
                "knee_L_pole": [0.10, 0.46, 0.40],
                "knee_R_pole": [-0.10, 0.46, 0.40],
                "elbow_L_pole": [round(float(v), 4) for v in eL],
                "elbow_R_pole": [round(float(v), 4) for v in eR],
                "shoulder_L_offset": [0.0, 0.0, 0.0],
                "shoulder_R_offset": [0.0, 0.0, 0.0],
            },
        }

    mannequin_scale_ratio = CHARACTER_NEUTRAL_DIMENSIONS["pawn"][1] / CHARACTER_NEUTRAL_DIMENSIONS["mannequin"][1]

    # 3. 左中景年轻男主（暖琥珀橙 #D98236，全程原地站立 [-0.32, 0.0, -1.66]，共 5 个关键帧 <= 5帧/10s）
    # 0.00s: 侧头看画左木屋方向；2.00s~8.20s: 转头看右前方军官倾听；10.20s: 目送军官向画左离开；15.07s: 缓慢回正微低头沉思
    young_kfs = [
        _mkf(
            0.00, [-0.32, 0.0, -1.66], 20.0,
            pelvis=[0.0, 0.888, 0.0], chest=[0.0, 1.348, 0.015], head_target=[-0.82, 1.54, 0.24],
            hand_L=[0.21, 0.79, 0.04], hand_R=[-0.21, 0.79, 0.04],
            foot_L=[0.11, 0.04, 0.02], foot_R=[-0.11, 0.04, -0.02], interp="ease_in_out",
        ),
        _mkf(
            2.00, [-0.32, 0.0, -1.66], 20.0,
            pelvis=[0.0, 0.888, 0.0], chest=[0.0, 1.348, 0.015], head_target=[0.34, 1.56, 0.72],
            hand_L=[0.21, 0.79, 0.04], hand_R=[-0.21, 0.79, 0.04],
            foot_L=[0.11, 0.04, 0.02], foot_R=[-0.11, 0.04, -0.02], interp="ease_in_out",
        ),
        _mkf(
            8.20, [-0.32, 0.0, -1.66], 18.0,
            pelvis=[0.0, 0.888, 0.0], chest=[0.0, 1.348, 0.015], head_target=[0.32, 1.56, 0.74],
            hand_L=[0.21, 0.79, 0.04], hand_R=[-0.21, 0.79, 0.04],
            foot_L=[0.11, 0.04, 0.02], foot_R=[-0.11, 0.04, -0.02], interp="ease_in_out",
        ),
        _mkf(
            10.20, [-0.32, 0.0, -1.66], 8.0,
            pelvis=[0.0, 0.888, 0.0], chest=[0.0, 1.348, 0.018], head_target=[-0.38, 1.53, 0.76],
            hand_L=[0.21, 0.79, 0.04], hand_R=[-0.21, 0.79, 0.04],
            foot_L=[0.11, 0.04, 0.02], foot_R=[-0.11, 0.04, -0.02], interp="ease_in_out",
        ),
        _mkf(
            15.07, [-0.32, 0.0, -1.66], -6.0,
            pelvis=[0.0, 0.888, 0.0], chest=[0.0, 1.344, 0.026], head_target=[-0.03, 1.24, 0.78],
            hand_L=[0.21, 0.79, 0.04], hand_R=[-0.21, 0.79, 0.04],
            foot_L=[0.11, 0.04, 0.02], foot_R=[-0.11, 0.04, -0.02], interp="ease_in_out",
        ),
    ]

    char_young = stage.add_character(
        "年轻男主 (倾听拍肩后沉思)",
        pos=[-0.32, 0.0, -1.66],
        rot_y=-6.0,
        pose="standing",
        scale=1.0,
        color="#D98236",
        body_style="mannequin",
        char_id="char_pat_young",
    )
    char_young.scale = [round(1.03 * mannequin_scale_ratio, 6)] * 3
    char_young.targets = copy.deepcopy(young_kfs[-1]["targets"])
    char_young.keyframes = copy.deepcopy(young_kfs)
    char_young.pins = {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False}
    char_young.extra.update({
        "head_dir_line": True,
        "enable_walk_gait": False,
        "pawn_smooth_walk": False,
        "keep_color": True,
        "protected": True,
    })

    # 4. 右前景年长军官（冷钢蓝 #4A78A8，任意 10s 窗口 <= 5 帧）
    # 0.00s~6.50s: 右前景左后过肩对谈；7.50s~8.35s: 伸手按拍男主左肩；9.04s: 转身从摄影机与男主之间擦镜横穿遮挡；10.20s: 出左画框
    older_kfs = [
        _mkf(
            0.00, [0.25, 0.0, -1.24], -128.0,
            pelvis=[0.0, 0.888, 0.0], chest=[0.0, 1.348, 0.015], head_target=[0.0, 1.54, 0.84],
            hand_L=[0.22, 0.79, 0.04], hand_R=[-0.21, 0.79, 0.05],
            foot_L=[0.11, 0.04, 0.02], foot_R=[-0.11, 0.04, -0.02], interp="ease_in_out",
        ),
        _mkf(
            6.50, [0.25, 0.0, -1.24], -122.0,
            pelvis=[0.0, 0.888, 0.0], chest=[0.0, 1.348, 0.015], head_target=[-0.04, 1.54, 0.84],
            hand_L=[0.22, 0.79, 0.04], hand_R=[-0.21, 0.82, 0.08],
            foot_L=[0.11, 0.04, 0.02], foot_R=[-0.11, 0.04, -0.02], interp="ease_in_out",
        ),
        _mkf(
            7.60, [0.24, 0.0, -1.23], -116.0,
            pelvis=[0.0, 0.888, 0.0], chest=[0.0, 1.350, 0.022], head_target=[-0.06, 1.53, 0.84],
            hand_L=[0.22, 0.79, 0.04], hand_R=[-0.21, 1.33, 0.44],
            foot_L=[0.11, 0.04, 0.02], foot_R=[-0.11, 0.04, -0.02],
            elbow_R_pole=[-0.34, 1.02, 0.18], interp="ease_in_out",
        ),
        _mkf(
            8.35, [0.22, 0.0, -1.21], -112.0,
            pelvis=[0.0, 0.888, 0.0], chest=[0.0, 1.348, 0.018], head_target=[-0.04, 1.53, 0.84],
            hand_L=[0.22, 0.79, 0.04], hand_R=[-0.11, 1.22, 0.43],
            foot_L=[0.11, 0.04, 0.02], foot_R=[-0.11, 0.04, -0.02],
            elbow_R_pole=[-0.30, 0.98, 0.18], interp="ease_in_out",
        ),
        _mkf(
            9.04, [-0.17, 0.0, -1.06], -92.0,
            pelvis=[0.0, 0.886, 0.0], chest=[0.0, 1.346, 0.020], head_target=[0.0, 1.53, 0.86],
            hand_L=[0.22, 0.79, 0.04], hand_R=[-0.21, 0.80, 0.06],
            foot_L=[0.11, 0.04, 0.03], foot_R=[-0.11, 0.04, -0.03], interp="linear",
        ),
        _mkf(
            10.20, [-1.45, 0.0, -1.06], -90.0,
            pelvis=[0.0, 0.888, 0.0], chest=[0.0, 1.348, 0.015], head_target=[0.0, 1.53, 0.86],
            hand_L=[0.22, 0.79, 0.04], hand_R=[-0.21, 0.79, 0.04],
            foot_L=[0.11, 0.04, 0.02], foot_R=[-0.11, 0.04, -0.02], interp="ease_in_out",
        ),
    ]

    char_older = stage.add_character(
        "年长军官 (过肩拍肩后擦镜出左画)",
        pos=[-1.45, 0.0, -1.06],
        rot_y=-90.0,
        pose="standing",
        scale=1.0,
        color="#4A78A8",
        body_style="mannequin",
        char_id="char_pat_older",
    )
    char_older.scale = [round(1.045 * mannequin_scale_ratio, 6)] * 3
    char_older.targets = copy.deepcopy(older_kfs[-1]["targets"])
    char_older.keyframes = copy.deepcopy(older_kfs)
    char_older.pins = {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False}
    char_older.extra.update({
        "head_dir_line": True,
        "enable_walk_gait": False,
        "pawn_smooth_walk": False,
        "keep_color": True,
        "protected": True,
    })

    # 5. 摄影机轨迹（共 4 个关键帧，严格三段物理复合运动）：
    # - 0.00s ~ 8.40s: 完全定机位（pos=[0.00, 1.55, 0.00], target=[0.00, 1.51, -5.00] 零视差锁死）
    # - 8.40s ~ 10.10s: 配合军官擦镜出画，云台平缓左摇约 9° 将男主移至画面中央（左缘短暂带入挂灯木柱）
    # - 10.10s ~ 15.07s: 沿光轴滑轨平滑慢推约 0.42m（Dolly In）至男主锁骨以上近景特写并稳停
    stage.set_camera(
        pos=[0.00, 1.55, 0.00],
        target=[0.00, 1.51, -5.00],
        focal_mm=focal_mm,
        aspect_ratio="2.35:1",
        export_width=aspect_w,
        export_height=aspect_h,
    )
    stage.add_camera_keyframe(0.00, pos=[0.00, 1.55, 0.00], target=[0.00, 1.51, -5.00], focal_mm=focal_mm, fov=fov_deg, interp="linear", stop=True)
    stage.add_camera_keyframe(8.40, pos=[0.00, 1.55, 0.00], target=[0.00, 1.51, -5.00], focal_mm=focal_mm, fov=fov_deg, interp="continuous", stop=True)
    stage.add_camera_keyframe(10.10, pos=[-0.02, 1.55, -0.05], target=[-0.80, 1.51, -5.00], focal_mm=focal_mm, fov=fov_deg, interp="continuous")
    stage.add_camera_keyframe(15.07, pos=[-0.09, 1.555, -0.47], target=[-0.97, 1.48, -5.00], focal_mm=focal_mm, fov=fov_deg, interp="continuous", stop=True)

    scene_dict = stage.to_dict()
    scene_dict["export_plan"] = {
        "confirmed": True,
        "confirmed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "priority_mode": "fast",
        "priority_label": "极速",
        "material_style": "matte_blockout_previz",
        "compositions": [
            {"t": 0.00, "label": "定机位起幅：过肩双人中近景，男主在左1/3看画左，军官在右前景左后过肩"},
            {"t": 2.00, "label": "定机位对视：男主转头回看右前方军官倾听，背景远岸零视差静止"},
            {"t": 7.80, "label": "定机位拍肩：军官抬手按拍男主左肩嘱托"},
            {"t": 9.04, "label": "前景擦镜+云台微左摇：军官转身贴镜横穿遮挡，镜头同步左摇重构图"},
            {"t": 10.10, "label": "左摇到位起推：军官出左画，男主居中目送，左缘短暂带入挂灯木柱"},
            {"t": 15.07, "label": "滑轨慢推落幅：沿光轴慢推0.42m至男主锁骨以上近景特写，男主回正微低头沉思"},
        ],
        "preserve_existing_scene": True,
        "use_area_and_visible_budget": True,
    }
    return scene_dict


def build_taiga_gunfight_chase_18p8s_scene() -> Dict[str, Any]:
    """雪林持枪追击 18.81秒 10镜连贯硬切+甩镜白模预演（方式一：单条时间轴阶跃切换，遵守一物一方块规范）"""
    duration = 18.81
    fps = 24
    aspect_w, aspect_h = 1280, 536

    stage = Stage()
    stage.clear()
    stage.duration = duration
    stage.fps = fps
    stage.render_mode = "clay_only"
    stage.visual_style = "clay_blockout"
    stage.global_body_style = "mannequin"
    stage.hide_edges = True
    stage.contact_shadows = True
    stage.environment = {
        "preset": "clay_studio_taiga_snow",
        "sky": "#6D7987",
        "ground": "#B4BEC8",
        "fog_color": "#7B8795",
        "fog": True,
        "fog_near": 12.0,
        "fog_far": 58.0,
        "fog_max": 0.82,
        "show_ground": True,
        "show_grid": False,
    }

    # ==================== 一物一方块：全走廊 8 个关键空间与遮挡占位方块 ====================
    # 1. 第1镜右前景冰岸雪石块
    stage.add_box(
        "冰岸右侧雪石块",
        pos=[1.75, 0.40, -2.55],
        size=[2.30, 0.80, 1.60],
        color="#8D97A3",
        obj_id="env_chase_shore_rock",
        arch_role="prop",
        keep_color=True,
        protected=True,
    )
    # 2. 远景针叶林岸背景墙块
    stage.add_box(
        "远景林岸背景块",
        pos=[0.00, 4.20, -64.00],
        size=[88.00, 8.40, 2.80],
        color="#556170",
        obj_id="env_chase_far_taiga",
        arch_role="prop",
        keep_color=True,
        protected=True,
    )
    # 3. 第2镜左前景擦镜暗树干块
    stage.add_box(
        "第2镜前景擦镜树干块",
        pos=[-0.38, 1.50, -4.35],
        size=[0.36, 3.00, 0.36],
        color="#48505B",
        obj_id="env_chase_fg_trunk",
        arch_role="prop",
        keep_color=True,
        protected=True,
    )
    # 4. 第3镜逃跑者掩体白桦树干块
    stage.add_box(
        "第3镜白桦掩体树干块",
        pos=[-0.08, 1.60, -10.80],
        size=[0.34, 3.20, 0.34],
        color="#C2CAD2",
        obj_id="env_chase_birch_trunk",
        arch_role="prop",
        keep_color=True,
        protected=True,
    )
    # 5 & 6. 第4镜林间雪道左右林墙块（单一长方体夹出雪道纵深走廊）
    stage.add_box(
        "雪道左侧林墙占位块",
        pos=[-3.30, 3.20, -19.50],
        size=[1.80, 6.40, 15.50],
        color="#5B6674",
        obj_id="env_chase_trail_left_wall",
        arch_role="prop",
        keep_color=True,
        protected=True,
    )
    stage.add_box(
        "雪道右侧林墙占位块",
        pos=[3.30, 3.20, -19.50],
        size=[1.80, 6.40, 15.50],
        color="#515C69",
        obj_id="env_chase_trail_right_wall",
        arch_role="prop",
        keep_color=True,
        protected=True,
    )
    # 7. 第6镜横倒雪木块（底部离地 0.75m 供两人先后滑钻无穿模通过）
    stage.add_box(
        "第6镜横倒巨木块",
        pos=[0.00, 0.92, -27.50],
        size=[3.60, 0.34, 0.36],
        rot=[0.0, 0.0, -3.0],
        color="#685E54",
        obj_id="env_chase_fallen_log",
        arch_role="prop",
        keep_color=True,
        protected=True,
    )
    # 8. 第8~10镜雪灌木丛遮挡块
    stage.add_box(
        "第8至10镜左侧雪灌木块",
        pos=[-1.08, 0.56, -35.20],
        size=[1.35, 1.12, 1.20],
        color="#7A8693",
        obj_id="env_chase_snow_bush",
        arch_role="prop",
        keep_color=True,
        protected=True,
    )

    def _mkf(
        t: float,
        pos: List[float],
        rot_y: float,
        pelvis: List[float],
        chest: List[float],
        head_target: List[float],
        hand_L: List[float],
        hand_R: List[float],
        foot_L: List[float],
        foot_R: List[float],
        elbow_L_pole: Optional[List[float]] = None,
        elbow_R_pole: Optional[List[float]] = None,
        interp: str = "linear",
    ) -> Dict[str, Any]:
        eL = elbow_L_pole if elbow_L_pole is not None else [0.32, 1.04, -0.14]
        eR = elbow_R_pole if elbow_R_pole is not None else [-0.32, 1.04, -0.14]
        return {
            "t": round(float(t), 3),
            "pos": [round(float(v), 4) for v in pos],
            "rot": [0.0, round(float(rot_y), 2), 0.0],
            "rot_y": round(float(rot_y), 2),
            "look_at": [round(float(v), 3) for v in head_target],
            "interp": interp,
            "interp_pos": interp,
            "interp_rot": interp,
            "interp_pose": interp,
            "targets": {
                "pelvis": [round(float(v), 4) for v in pelvis],
                "chest": [round(float(v), 4) for v in chest],
                "head_target": [round(float(v), 4) for v in head_target],
                "hand_L": [round(float(v), 4) for v in hand_L],
                "hand_R": [round(float(v), 4) for v in hand_R],
                "foot_L": [round(float(v), 4) for v in foot_L],
                "foot_R": [round(float(v), 4) for v in foot_R],
                "knee_L_pole": [0.12, 0.48, 0.42],
                "knee_R_pole": [-0.12, 0.48, 0.42],
                "elbow_L_pole": [round(float(v), 4) for v in eL],
                "elbow_R_pole": [round(float(v), 4) for v in eR],
                "shoulder_L_offset": [0.0, 0.0, 0.0],
                "shoulder_R_offset": [0.0, 0.0, 0.0],
            },
        }

    mannequin_scale_ratio = CHARACTER_NEUTRAL_DIMENSIONS["pawn"][1] / CHARACTER_NEUTRAL_DIMENSIONS["mannequin"][1]

    # ==================== 角色 A：追击者猎人（暖琥珀橙 #D98236 + 单方块步枪） ====================
    hunter_kfs = [
        # Shot 1 (0.00~1.88s): 踏冰从右向左奔跑横穿低机位前景
        _mkf(
            0.00, [1.55, 0.0, -2.15], -90.0,
            pelvis=[0.0, 0.85, 0.0], chest=[0.0, 1.29, 0.08], head_target=[0.0, 1.48, 0.85],
            hand_L=[0.12, 1.02, 0.36], hand_R=[-0.16, 0.98, 0.16],
            foot_L=[0.11, 0.14, 0.24], foot_R=[-0.11, 0.04, -0.22], interp="linear",
        ),
        _mkf(
            1.87, [-1.35, 0.0, -2.20], -90.0,
            pelvis=[0.0, 0.85, 0.0], chest=[0.0, 1.29, 0.08], head_target=[0.0, 1.48, 0.85],
            hand_L=[0.12, 1.02, 0.36], hand_R=[-0.16, 0.98, 0.16],
            foot_L=[0.11, 0.04, -0.20], foot_R=[-0.11, 0.14, 0.24], interp="step",
        ),
        # Shot 2 (1.88~4.52s): 从前景树后左奔 -> 急停正对镜头抵肩举枪瞄准
        _mkf(
            1.88, [0.52, 0.0, -5.50], -78.0,
            pelvis=[0.0, 0.86, 0.0], chest=[0.0, 1.31, 0.06], head_target=[0.0, 1.50, 0.85],
            hand_L=[0.10, 1.06, 0.38], hand_R=[-0.15, 1.02, 0.18],
            foot_L=[0.11, 0.04, 0.12], foot_R=[-0.11, 0.04, -0.12], interp="ease_in_out",
        ),
        _mkf(
            3.35, [-0.85, 0.0, -5.55], 0.0,
            pelvis=[0.0, 0.87, 0.0], chest=[0.0, 1.32, 0.05], head_target=[0.0, 1.44, 0.90],
            hand_L=[-0.05, 1.23, 0.52], hand_R=[-0.14, 1.22, 0.32],
            foot_L=[0.14, 0.04, 0.08], foot_R=[-0.14, 0.04, -0.08],
            elbow_L_pole=[0.24, 0.98, 0.20], elbow_R_pole=[-0.28, 1.00, 0.08], interp="ease_in_out",
        ),
        _mkf(
            4.51, [-0.85, 0.0, -5.55], 0.0,
            pelvis=[0.0, 0.87, 0.0], chest=[0.0, 1.32, 0.05], head_target=[0.0, 1.44, 0.90],
            hand_L=[-0.05, 1.24, 0.52], hand_R=[-0.14, 1.23, 0.32],
            foot_L=[0.14, 0.04, 0.08], foot_R=[-0.14, 0.04, -0.08],
            elbow_L_pole=[0.24, 0.98, 0.20], elbow_R_pole=[-0.28, 1.00, 0.08], interp="step",
        ),
        # Shot 4 (5.58~8.83s): 雪道深处迎着高速前推镜头全速冲刺至近景
        _mkf(
            5.58, [0.06, 0.0, -25.40], 0.0,
            pelvis=[0.0, 0.83, 0.0], chest=[0.0, 1.25, 0.12], head_target=[0.0, 1.44, 0.88],
            hand_L=[0.10, 1.04, 0.36], hand_R=[-0.16, 0.98, 0.16],
            foot_L=[0.12, 0.08, 0.18], foot_R=[-0.12, 0.04, -0.18], interp="linear",
        ),
        _mkf(
            8.82, [0.08, 0.0, -21.15], 4.0,
            pelvis=[0.0, 0.78, 0.0], chest=[0.0, 1.18, 0.16], head_target=[0.0, 1.38, 0.88],
            hand_L=[0.10, 1.04, 0.38], hand_R=[-0.16, 0.98, 0.18],
            foot_L=[0.12, 0.04, -0.16], foot_R=[-0.12, 0.10, 0.20], interp="step",
        ),
        # Shot 5 (8.83~9.75s): 右侧脸近景向右冲刺前扑
        _mkf(
            8.83, [-0.42, 0.0, -24.25], 78.0,
            pelvis=[0.0, 0.84, 0.0], chest=[0.0, 1.27, 0.10], head_target=[0.0, 1.46, 0.88],
            hand_L=[0.10, 1.05, 0.38], hand_R=[-0.15, 1.00, 0.18],
            foot_L=[0.11, 0.04, 0.14], foot_R=[-0.11, 0.04, -0.14], interp="linear",
        ),
        _mkf(
            9.74, [0.46, 0.0, -24.45], 82.0,
            pelvis=[0.0, 0.68, 0.0], chest=[0.0, 1.08, 0.18], head_target=[0.0, 1.24, 0.88],
            hand_L=[0.10, 0.92, 0.40], hand_R=[-0.15, 0.88, 0.20],
            foot_L=[0.11, 0.04, -0.14], foot_R=[-0.11, 0.08, 0.18], interp="step",
        ),
        # Shot 6 (11.05~12.25s): 紧随逃跑者之后从横倒巨木下方低姿俯身钻过
        _mkf(
            11.05, [0.08, 0.0, -28.45], 0.0,
            pelvis=[0.0, 0.32, 0.0], chest=[0.0, 0.46, 0.26], head_target=[0.0, 0.48, 0.92],
            hand_L=[0.18, 0.24, 0.38], hand_R=[-0.18, 0.24, 0.24],
            foot_L=[0.14, 0.04, -0.22], foot_R=[-0.14, 0.04, -0.06], interp="ease_in_out",
        ),
        _mkf(
            12.24, [0.06, 0.0, -27.05], 0.0,
            pelvis=[0.0, 0.35, 0.0], chest=[0.0, 0.50, 0.24], head_target=[0.0, 0.52, 0.92],
            hand_L=[0.16, 0.28, 0.38], hand_R=[-0.16, 0.28, 0.22],
            foot_L=[0.14, 0.04, -0.08], foot_R=[-0.14, 0.04, -0.20], interp="step",
        ),
        # Shot 7 (12.25~13.96s): 右侧脸紧迫凝视大特写
        _mkf(
            12.25, [0.10, 0.0, -30.20], 62.0,
            pelvis=[0.0, 0.888, 0.0], chest=[0.0, 1.348, 0.02], head_target=[0.04, 1.54, 0.88],
            hand_L=[0.10, 1.06, 0.36], hand_R=[-0.16, 1.00, 0.16],
            foot_L=[0.11, 0.04, 0.02], foot_R=[-0.11, 0.04, -0.02], interp="ease_in_out",
        ),
        _mkf(
            13.95, [0.11, 0.0, -30.20], 66.0,
            pelvis=[0.0, 0.888, 0.0], chest=[0.0, 1.348, 0.02], head_target=[0.06, 1.54, 0.88],
            hand_L=[0.10, 1.06, 0.36], hand_R=[-0.16, 1.00, 0.16],
            foot_L=[0.11, 0.04, 0.02], foot_R=[-0.11, 0.04, -0.02], interp="step",
        ),
        # Shot 9 (15.88~17.12s): 灌木丛后端枪瞄准开火
        _mkf(
            15.88, [0.85, 0.0, -38.40], -18.0,
            pelvis=[0.0, 0.84, 0.0], chest=[0.0, 1.28, 0.06], head_target=[0.0, 1.40, 0.90],
            hand_L=[-0.05, 1.21, 0.52], hand_R=[-0.14, 1.20, 0.32],
            foot_L=[0.14, 0.04, 0.08], foot_R=[-0.14, 0.04, -0.08],
            elbow_L_pole=[0.24, 0.96, 0.20], elbow_R_pole=[-0.28, 0.98, 0.08], interp="ease_in_out",
        ),
        _mkf(
            17.11, [0.86, 0.0, -38.35], -18.0,
            pelvis=[0.0, 0.84, 0.0], chest=[0.0, 1.28, 0.05], head_target=[0.0, 1.40, 0.90],
            hand_L=[-0.05, 1.22, 0.51], hand_R=[-0.14, 1.21, 0.31],
            foot_L=[0.14, 0.04, 0.08], foot_R=[-0.14, 0.04, -0.08],
            elbow_L_pole=[0.24, 0.96, 0.20], elbow_R_pole=[-0.28, 0.98, 0.08], interp="step",
        ),
        # Shot 10 (17.12~18.81s): 右后方纵深处持枪稳步逼近前景伏地的 B
        _mkf(
            17.12, [1.46, 0.0, -39.20], -20.0,
            pelvis=[0.0, 0.87, 0.0], chest=[0.0, 1.32, 0.04], head_target=[0.0, 1.44, 0.88],
            hand_L=[0.10, 1.08, 0.38], hand_R=[-0.16, 1.04, 0.18],
            foot_L=[0.12, 0.04, 0.08], foot_R=[-0.12, 0.04, -0.08], interp="linear",
        ),
        _mkf(
            18.81, [1.32, 0.0, -38.05], -22.0,
            pelvis=[0.0, 0.87, 0.0], chest=[0.0, 1.32, 0.04], head_target=[0.0, 1.42, 0.88],
            hand_L=[0.10, 1.08, 0.38], hand_R=[-0.16, 1.04, 0.18],
            foot_L=[0.12, 0.04, -0.06], foot_R=[-0.12, 0.04, 0.08], interp="ease_in_out",
        ),
    ]

    char_hunter = stage.add_character(
        "追击者猎人 A (暖橙)",
        pos=[1.32, 0.0, -38.05],
        rot_y=-22.0,
        pose="standing",
        scale=1.0,
        color="#D98236",
        body_style="mannequin",
        char_id="char_chase_hunter_A",
    )
    char_hunter.scale = [round(1.04 * mannequin_scale_ratio, 6)] * 3
    char_hunter.targets = copy.deepcopy(hunter_kfs[-1]["targets"])
    char_hunter.keyframes = copy.deepcopy(hunter_kfs)
    char_hunter.pins = {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False}
    char_hunter.extra.update({
        "head_dir_line": True,
        "enable_walk_gait": False,
        "pawn_smooth_walk": False,
        "keep_color": True,
        "protected": True,
    })

    rifle_A = stage.add_box(
        "追击者步枪块",
        pos=[1.20, 1.05, -37.80],
        size=[0.06, 0.10, 0.90],
        color="#3B3530",
        obj_id="rifle_chase_A",
        arch_role="gun",
        keep_color=True,
        protected=True,
    )
    rifle_A.bind_to_character("char_chase_hunter_A", socket="hand_R", local_pos=[0.0, 0.0, 0.24], local_rot=[0.0, 0.0, 0.0], two_handed=False)

    # ==================== 角色 B：逃跑者男子（冷钢蓝 #4A78A8 + 单方块步枪） ====================
    fugitive_kfs = [
        # Shot 1~2 期间退避在画左林墙后避免穿帮；Shot 3 (4.52~5.58s) 躲白桦树干后惊避向左侧逃离
        _mkf(
            0.00, [-2.20, 0.0, -11.40], -105.0,
            pelvis=[0.0, 0.72, 0.0], chest=[0.0, 1.12, 0.12], head_target=[0.0, 1.28, 0.85],
            hand_L=[0.12, 0.92, 0.34], hand_R=[-0.16, 0.88, 0.16],
            foot_L=[0.12, 0.04, 0.06], foot_R=[-0.12, 0.04, -0.06], interp="step",
        ),
        _mkf(
            4.52, [0.14, 0.0, -11.15], -105.0,
            pelvis=[0.0, 0.72, 0.0], chest=[0.0, 1.12, 0.12], head_target=[0.0, 1.28, 0.85],
            hand_L=[0.12, 0.92, 0.34], hand_R=[-0.16, 0.88, 0.16],
            foot_L=[0.12, 0.04, 0.06], foot_R=[-0.12, 0.04, -0.06], interp="ease_in_out",
        ),
        _mkf(
            5.57, [-0.58, 0.0, -11.30], -115.0,
            pelvis=[0.0, 0.68, 0.0], chest=[0.0, 1.06, 0.16], head_target=[0.0, 1.22, 0.85],
            hand_L=[0.12, 0.88, 0.34], hand_R=[-0.16, 0.84, 0.16],
            foot_L=[0.12, 0.04, -0.10], foot_R=[-0.12, 0.08, 0.14], interp="step",
        ),
        # Shot 6 (9.75~11.00s): 先从横倒巨木下方滑铲钻过并向画左深处撤出
        _mkf(
            9.75, [-0.10, 0.0, -27.15], 175.0,
            pelvis=[0.0, 0.34, 0.0], chest=[0.0, 0.48, 0.22], head_target=[0.0, 0.52, 0.90],
            hand_L=[0.16, 0.26, 0.36], hand_R=[-0.16, 0.26, 0.22],
            foot_L=[0.13, 0.04, -0.18], foot_R=[-0.13, 0.04, 0.06], interp="linear",
        ),
        _mkf(
            10.95, [-1.35, 0.0, -29.30], -145.0,
            pelvis=[0.0, 0.58, 0.0], chest=[0.0, 0.92, 0.18], head_target=[0.0, 1.06, 0.88],
            hand_L=[0.14, 0.72, 0.34], hand_R=[-0.16, 0.68, 0.18],
            foot_L=[0.13, 0.04, 0.10], foot_R=[-0.13, 0.04, -0.14], interp="step",
        ),
        # Shot 8 (13.96~15.88s): 撞入雪灌木丛跌倒后挣扎爬起向右逃
        _mkf(
            13.96, [-0.28, 0.0, -36.15], 38.0,
            pelvis=[0.0, 0.42, 0.0], chest=[0.0, 0.66, 0.20], head_target=[0.0, 0.74, 0.86],
            hand_L=[0.18, 0.36, 0.32], hand_R=[-0.18, 0.42, 0.24],
            foot_L=[0.14, 0.04, -0.08], foot_R=[-0.14, 0.04, 0.08], interp="ease_in_out",
        ),
        _mkf(
            15.87, [0.58, 0.0, -35.90], 78.0,
            pelvis=[0.0, 0.74, 0.0], chest=[0.0, 1.14, 0.14], head_target=[0.0, 1.28, 0.86],
            hand_L=[0.12, 0.92, 0.36], hand_R=[-0.16, 0.88, 0.18],
            foot_L=[0.12, 0.04, 0.12], foot_R=[-0.12, 0.04, -0.12], interp="step",
        ),
        # Shot 10 (17.12~18.81s): 前景扑倒喘息 -> 力竭低头伏地（右后方追击者逼近）
        _mkf(
            17.12, [-0.18, 0.0, -34.95], 10.0,
            pelvis=[0.0, 0.44, 0.0], chest=[0.0, 0.88, 0.12], head_target=[0.0, 0.98, 0.84],
            hand_L=[0.20, 0.56, 0.28], hand_R=[-0.22, 0.54, 0.22],
            foot_L=[0.15, 0.04, -0.16], foot_R=[-0.15, 0.04, -0.18], interp="ease_in_out",
        ),
        _mkf(
            18.81, [-0.12, 0.0, -34.88], 18.0,
            pelvis=[0.0, 0.32, 0.0], chest=[0.0, 0.52, 0.22], head_target=[0.14, 0.38, 0.78],
            hand_L=[0.20, 0.24, 0.30], hand_R=[-0.22, 0.26, 0.24],
            foot_L=[0.15, 0.04, -0.22], foot_R=[-0.15, 0.04, -0.20], interp="ease_in_out",
        ),
    ]

    char_fugitive = stage.add_character(
        "逃跑者 B (冷钢蓝)",
        pos=[-0.12, 0.0, -34.88],
        rot_y=18.0,
        pose="standing",
        scale=1.0,
        color="#4A78A8",
        body_style="mannequin",
        char_id="char_chase_fugitive_B",
    )
    char_fugitive.scale = [round(1.02 * mannequin_scale_ratio, 6)] * 3
    char_fugitive.targets = copy.deepcopy(fugitive_kfs[-1]["targets"])
    char_fugitive.keyframes = copy.deepcopy(fugitive_kfs)
    char_fugitive.pins = {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False}
    char_fugitive.extra.update({
        "head_dir_line": True,
        "enable_walk_gait": False,
        "pawn_smooth_walk": False,
        "keep_color": True,
        "protected": True,
    })

    rifle_B = stage.add_box(
        "逃跑者步枪块",
        pos=[-0.35, 0.30, -34.65],
        size=[0.06, 0.10, 0.86],
        color="#343940",
        obj_id="rifle_chase_B",
        arch_role="gun",
        keep_color=True,
        protected=True,
    )
    rifle_B.bind_to_character("char_chase_fugitive_B", socket="hand_R", local_pos=[0.0, 0.0, 0.22], local_rot=[0.0, 0.0, 0.0], two_handed=False)

    # ==================== 10 段摄影机轨迹（含 4.20~4.52s 极速左甩镜 + 阶跃硬切） ====================
    f35 = focal_mm_to_fov(35.0)
    f45 = focal_mm_to_fov(45.0)
    f50 = focal_mm_to_fov(50.0)
    f55 = focal_mm_to_fov(55.0)
    f75 = focal_mm_to_fov(75.0)

    stage.set_camera(
        pos=[0.00, 0.32, 0.00],
        target=[-0.20, 0.52, -5.00],
        focal_mm=35.0,
        aspect_ratio="2.39:1",
        export_width=aspect_w,
        export_height=aspect_h,
    )

    # Shot 1 (0.00 ~ 1.88s): 贴冰面低角度定机位
    stage.add_camera_keyframe(0.000, pos=[0.00, 0.32, 0.00], target=[-0.20, 0.52, -5.00], focal_mm=35.0, fov=f35, interp="step", stop=True)
    # Shot 2 (1.88 ~ 4.52s): 斯坦尼康侧跟 -> 正面瞄准 -> 4.20s起极速左甩镜
    stage.add_camera_keyframe(1.880, pos=[-0.16, 1.44, -3.55], target=[0.24, 1.42, -5.55], focal_mm=50.0, fov=f50, interp="continuous", stop=True)
    stage.add_camera_keyframe(3.350, pos=[-0.85, 1.46, -3.95], target=[-0.85, 1.44, -5.55], focal_mm=50.0, fov=f50, interp="continuous", stop=True)
    stage.add_camera_keyframe(4.200, pos=[-0.85, 1.47, -4.02], target=[-0.85, 1.45, -5.55], focal_mm=50.0, fov=f50, interp="linear", stop=True)
    stage.add_camera_keyframe(4.518, pos=[-0.92, 1.45, -4.02], target=[-3.60, 1.42, -4.40], focal_mm=50.0, fov=f50, interp="step", stop=True)
    # Shot 3 (4.52 ~ 5.58s): 甩镜急停落稳在白桦掩体中景
    stage.add_camera_keyframe(4.520, pos=[0.08, 1.34, -8.90], target=[0.38, 1.32, -10.80], focal_mm=50.0, fov=f50, interp="ease_out", stop=True)
    stage.add_camera_keyframe(4.780, pos=[0.00, 1.34, -8.90], target=[-0.16, 1.30, -10.80], focal_mm=50.0, fov=f50, interp="step", stop=True)
    # Shot 4 (5.58 ~ 8.83s): 雪道低机位高速前推（追击者迎面冲刺）
    stage.add_camera_keyframe(5.580, pos=[0.00, 0.66, -13.20], target=[0.05, 0.92, -22.00], focal_mm=35.0, fov=f35, interp="linear", stop=True)
    stage.add_camera_keyframe(8.828, pos=[0.04, 0.72, -19.55], target=[0.08, 1.04, -22.00], focal_mm=35.0, fov=f35, interp="step", stop=True)
    # Shot 5 (8.83 ~ 9.75s): 侧面跟移持枪右冲前扑近景
    stage.add_camera_keyframe(8.830, pos=[-0.38, 1.38, -22.80], target=[-0.38, 1.36, -24.30], focal_mm=55.0, fov=f55, interp="linear", stop=True)
    stage.add_camera_keyframe(9.748, pos=[0.42, 1.22, -22.80], target=[0.46, 1.14, -24.45], focal_mm=55.0, fov=f55, interp="step", stop=True)
    # Shot 6 (9.75 ~ 12.25s): 横倒巨木前低机位定镜微摇（B先钻出、A紧随钻入）
    stage.add_camera_keyframe(9.750, pos=[0.00, 0.56, -25.55], target=[-0.08, 0.62, -27.50], focal_mm=45.0, fov=f45, interp="ease_in_out", stop=True)
    stage.add_camera_keyframe(12.248, pos=[0.04, 0.54, -25.48], target=[0.06, 0.60, -27.50], focal_mm=45.0, fov=f45, interp="step", stop=True)
    # Shot 7 (12.25 ~ 13.96s): 追击者 A 右侧脸紧迫大特写
    stage.add_camera_keyframe(12.250, pos=[0.06, 1.54, -29.12], target=[0.10, 1.52, -30.20], focal_mm=75.0, fov=f75, interp="ease_in_out", stop=True)
    stage.add_camera_keyframe(13.958, pos=[0.08, 1.54, -29.18], target=[0.12, 1.52, -30.20], focal_mm=75.0, fov=f75, interp="step", stop=True)
    # Shot 8 (13.96 ~ 15.88s): 雪灌木丛中景（B跌倒挣扎爬起右逃）
    stage.add_camera_keyframe(13.960, pos=[0.00, 0.94, -33.25], target=[-0.24, 0.80, -36.10], focal_mm=45.0, fov=f45, interp="ease_in_out", stop=True)
    stage.add_camera_keyframe(15.878, pos=[0.06, 0.96, -33.25], target=[0.46, 0.88, -36.00], focal_mm=45.0, fov=f45, interp="step", stop=True)
    # Shot 9 (15.88 ~ 17.12s): 追击者 A 端枪瞄准击发中近景
    stage.add_camera_keyframe(15.880, pos=[0.56, 1.36, -36.55], target=[0.84, 1.36, -38.40], focal_mm=55.0, fov=f55, interp="ease_in_out", stop=True)
    stage.add_camera_keyframe(17.118, pos=[0.54, 1.35, -36.64], target=[0.85, 1.35, -38.35], focal_mm=55.0, fov=f55, interp="step", stop=True)
    # Shot 10 (17.12 ~ 18.81s): 双人纵深对峙落幅（前景 B 伏地 + 右后景 A 端枪逼近）
    stage.add_camera_keyframe(17.120, pos=[-0.06, 0.86, -33.35], target=[0.20, 0.92, -37.50], focal_mm=50.0, fov=f50, interp="ease_in_out", stop=True)
    stage.add_camera_keyframe(18.810, pos=[-0.04, 0.85, -33.42], target=[0.26, 0.90, -37.50], focal_mm=50.0, fov=f50, interp="linear", stop=True)

    scene_dict = stage.to_dict()
    scene_dict["export_plan"] = {
        "confirmed": True,
        "confirmed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "priority_mode": "fast",
        "priority_label": "极速",
        "material_style": "matte_blockout_previz",
        "compositions": [
            {"t": 1.00, "label": "第1镜：冰岸低角度定镜，猎人A踏冰向左奔跑"},
            {"t": 3.80, "label": "第2镜：斯坦尼康跟退至猎人A急停正对镜头抵肩瞄准"},
            {"t": 5.00, "label": "第3镜：左甩镜急停白桦树干，逃跑者B树后惊避左逃"},
            {"t": 8.40, "label": "第4镜：雪道高速前推，猎人A迎面冲刺逼近镜头"},
            {"t": 9.30, "label": "第5镜：侧面跟拍猎人A向右冲刺前扑"},
            {"t": 10.30, "label": "第6镜前段：横倒巨木前低机位，逃跑者B先从木下钻出"},
            {"t": 11.75, "label": "第6镜后段：猎人A紧随俯身从横倒巨木下方钻入"},
            {"t": 13.00, "label": "第7镜：猎人A右侧脸紧迫凝视大特写"},
            {"t": 15.00, "label": "第8镜：逃跑者B撞入雪灌木丛跌倒爬起"},
            {"t": 16.40, "label": "第9镜：猎人A灌木后端枪瞄准开火"},
            {"t": 18.70, "label": "第10镜：双人纵深落幅，前景B力竭伏地、右后方A端枪逼近"},
        ],
        "preserve_existing_scene": True,
        "use_area_and_visible_budget": True,
    }
    return scene_dict


def build_jeep_crane_reveal_8s_scene() -> Dict[str, Any]:
    """吉普启动穿门与起重机拉升一镜到底 8.08秒 白模预演（按基准鸟瞰图校准建筑比例位置，无窗低模，<=30树林体块）"""
    duration = 8.08
    fps = 24
    aspect_w, aspect_h = 1680, 720

    stage = Stage()
    stage.clear()
    stage.duration = duration
    stage.fps = fps
    stage.meters_per_unit = 1.0
    stage.allow_snow_jeep_4mps = True
    stage.render_mode = "clay_only"
    stage.visual_style = "clay_blockout"
    stage.global_body_style = "pawn"
    stage.hide_edges = True
    stage.contact_shadows = True
    stage.environment = {
        "preset": "clay_studio_snow_yard",
        "sky": "#9AA8B5",
        "ground": "#C8D0D8",
        "fog_color": "#A3B1BE",
        "fog": True,
        "fog_near": 68.0,
        "fog_far": 180.0,
        "fog_max": 0.36,
        "show_ground": True,
        "show_grid": False,
    }

    def _box(
        name: str,
        pos: List[float],
        size: List[float],
        color: str,
        obj_id: str,
        rot: Optional[List[float]] = None,
        arch_role: str = "prop",
        surface_zones: Optional[Dict[str, Any]] = None,
    ) -> PrimitiveObj:
        extra: Dict[str, Any] = {
            "arch_role": arch_role,
            "keep_color": True,
            "protected": True,
        }
        if surface_zones:
            extra["surface_zones"] = surface_zones
        return stage.add_box(
            name=name,
            pos=pos,
            size=size,
            color=color,
            rot=rot or [0.0, 0.0, 0.0],
            obj_id=obj_id,
            **extra,
        )

    def _part(
        pid: str,
        shape: str,
        pos: List[float],
        size: List[float],
        color: str,
        rot: Optional[List[float]] = None,
        surface_zones: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        p: Dict[str, Any] = {
            "id": pid,
            "shape": shape,
            "pos": pos,
            "size": size,
            "rot": rot or [0.0, 0.0, 0.0],
            "color": color,
            "keep_color": True,
        }
        if surface_zones:
            p["surface_zones"] = surface_zones
        return p

    def _wheel_parts(radius: float, width: float, ax_x: float, z_front: float, z_rear: float) -> List[Dict[str, Any]]:
        dia = round(radius * 2.0, 3)
        return [
            _part(pid, "cylinder", [sx, radius, sz], [dia, width, dia], "#1E2226", [0.0, 0.0, 90.0])
            for pid, sx, sz in (
                ("wheel_fl", -ax_x, z_front),
                ("wheel_fr", ax_x, z_front),
                ("wheel_rl", -ax_x, z_rear),
                ("wheel_rr", ax_x, z_rear),
            )
        ]

    def _add_gable_cabin(
        name: str,
        obj_id: str,
        pos: List[float],
        rot_y: float,
        width: float,
        wall_h: float,
        depth: float,
        ridge_extra: float,
        wall_col: str = "#6B5E53",
        roof_col_l: str = "#CBD3DC",
        roof_col_r: str = "#BCC5CF",
        is_ew_ridge: bool = False,
    ) -> None:
        """无窗双坡木屋（严格 3 个基础体：左右屋面板精确停于 X=0 屋脊接缝不互穿，加厚实心盖合无透空）"""
        lw = depth if is_ew_ridge else width
        ld = width if is_ew_ridge else depth
        ry = rot_y + 90.0 if is_ew_ridge else rot_y
        half_w = lw * 0.5
        theta = math.atan2(ridge_extra, half_w)
        slope_deg = round(math.degrees(theta), 2)
        eave_out = 0.26
        panel_w = round(math.hypot(half_w, ridge_extra) + eave_out, 3)
        half_len = panel_w * 0.5
        slab_t = max(0.48, round(ridge_extra * 0.56, 3))
        # 使 roof_l 外表面右上角严格停在屋脊线 (0, wall_h + ridge_extra)，绝不穿出对侧坡面
        surf_x = -half_len * math.cos(theta)
        surf_y = (wall_h + ridge_extra) - half_len * math.sin(theta)
        off_x = round(abs(surf_x + math.sin(theta) * (slab_t * 0.5)), 3)
        mid_y = round(surf_y - math.cos(theta) * (slab_t * 0.5), 3)
        stage.add_group(
            name,
            pos=pos,
            rot=[0.0, ry, 0.0],
            obj_id=obj_id,
            arch_role="building",
            keep_color=True,
            protected=True,
            parts=[
                _part("wall", "box", [0.0, round(wall_h * 0.5, 3), 0.0], [lw, wall_h, ld], wall_col),
                _part("roof_l", "box", [-off_x, mid_y, 0.0], [panel_w, slab_t, ld + 0.500], roof_col_l, [0.0, 0.0, slope_deg]),
                _part("roof_r", "box", [off_x, round(mid_y + 0.004, 3), 0.0], [panel_w, slab_t, ld + 0.494], roof_col_r, [0.0, 0.0, -slope_deg]),
            ],
        )

    def _add_shed_cabin(
        name: str,
        obj_id: str,
        pos: List[float],
        rot_y: float,
        width: float,
        wall_h: float,
        depth: float,
        slope_deg: float = 10.0,
        wall_col: str = "#685D53",
        roof_col: str = "#CBD4DC",
    ) -> None:
        """无窗单坡木棚（严格 2 个基础体：无窗墙体 + 单坡雪顶）"""
        stage.add_group(
            name,
            pos=pos,
            rot=[0.0, rot_y, 0.0],
            obj_id=obj_id,
            arch_role="building",
            keep_color=True,
            protected=True,
            parts=[
                _part("wall", "box", [0.0, round(wall_h * 0.5, 3), 0.0], [width, wall_h, depth], wall_col),
                _part("roof", "box", [0.0, round(wall_h + 0.24, 3), 0.0], [width + 0.45, 0.36, depth + 0.45], roof_col, [slope_deg, 0.0, 0.0]),
            ],
        )

    # ==================== 1. 左侧冰封大河延伸至地平线、泥雪院路与门外右拐雪道（各 1 个基础体） ====================
    _box("左侧冰封裂冰大河", [-53.0, 0.014, -410.0], [78.0, 0.028, 900.0], "#4C6173", "env_icy_river", rot=[0.0, 0.0, 0.0], arch_role="road")
    _box("远左雪原对岸雪坡", [-127.0, 0.018, -410.0], [70.0, 0.036, 900.0], "#D5DDE5", "env_river_west_bank", rot=[0.0, 0.0, 0.0], arch_role="road")

    _box("院内下段弯道泥雪路", [8.0, 0.015, -9.5], [3.2, 0.030, 18.0], "#4E5760", "env_road_yard_south", rot=[0.0, -6.0, 0.0], arch_role="road")
    _box("院内主干泥雪路", [9.0, 0.017, -26.5], [3.3, 0.034, 19.5], "#49525B", "env_road_yard_main", rot=[0.0, -2.5, 0.0], arch_role="road")
    _box("院内左岔路泥雪块", [5.2, 0.016, -20.2], [6.8, 0.032, 2.8], "#545D66", "env_road_yard_branch", rot=[0.0, 12.0, 0.0], arch_role="road")
    _box("后门出弯右拐外雪路", [17.8, 0.014, -40.2], [20.0, 0.028, 3.8], "#7C8894", "env_road_outside_turn", rot=[0.0, 8.0, 0.0], arch_role="road")
    _box("外围东侧雪地巡逻道", [22.2, 0.013, -15.0], [3.6, 0.026, 54.0], "#84909C", "env_road_outside_east", rot=[0.0, -3.0, 0.0], arch_role="road")

    # ==================== 2. 校准后无窗紧凑村落建筑（共11栋，严格 <=3 基础体且零窗户） ====================
    # 正前横屋（居中前景横置，略北收紧院落前后纵深）
    _add_gable_cabin(
        "正前横置原木屋", "arch_fg_center_cabin",
        pos=[3.60, 0.0, 3.50], rot_y=1.0, width=10.20, wall_h=2.40, depth=4.60, ridge_extra=1.40,
        wall_col="#695C50", roof_col_l="#CDD5DE", roof_col_r="#B9C3CD", is_ew_ridge=True,
    )
    # 左前高架双层棚（3 基础体：底层木屋 + 中间架空柱芯 + 雪顶盖板）
    stage.add_group(
        "左前河岸高架木棚",
        pos=[-9.50, 0.0, -4.50],
        rot=[0.0, 8.0, 0.0],
        obj_id="arch_fg_left_pavilion",
        arch_role="building",
        keep_color=True,
        protected=True,
        parts=[
            _part("base_room", "box", [0.0, 0.80, 0.0], [4.60, 1.60, 4.20], "#62574C"),
            _part("upper_frame", "box", [0.0, 2.15, 0.0], [3.90, 1.10, 3.60], "#4C443C"),
            _part("snow_cap", "box", [0.0, 2.86, 0.0], [5.10, 0.32, 4.70], "#CBD4DD", [4.0, 0.0, 0.0]),
        ],
    )
    # 左河岸瞭望哨塔（3 基础体：在主木屋左后肩外探出屋顶，齐冰河岸边）
    stage.add_group(
        "左河岸木制哨塔",
        pos=[-10.40, 0.0, -27.50],
        rot=[0.0, 10.0, 0.0],
        obj_id="arch_watchtower",
        arch_role="building",
        keep_color=True,
        protected=True,
        parts=[
            _part("tower_legs", "box", [0.0, 2.30, 0.0], [1.35, 4.60, 1.35], "#574D43"),
            _part("lookout_box", "box", [0.0, 5.30, 0.0], [2.20, 1.40, 2.20], "#6B5E53"),
            _part("tower_roof", "box", [0.0, 6.28, 0.0], [2.70, 0.42, 2.70], "#C8D1DA", [8.0, 0.0, 0.0]),
        ],
    )
    # 左后河岸木屋 & 远后储木区木屋
    _add_gable_cabin(
        "左后河岸木屋", "arch_left_back_cabin",
        pos=[-5.80, 0.0, -37.00], rot_y=5.0, width=4.60, wall_h=2.20, depth=5.20, ridge_extra=1.40,
        wall_col="#64594F", roof_col_l="#C6CFD8", roof_col_r="#B6C0CA", is_ew_ridge=False,
    )
    _add_gable_cabin(
        "远后储木区木屋", "arch_far_timber_cabin",
        pos=[2.20, 0.0, -52.00], rot_y=3.0, width=5.80, wall_h=2.30, depth=4.20, ridge_extra=1.40,
        wall_col="#62584E", roof_col_l="#C8D1DA", roof_col_r="#B6C0CA", is_ew_ridge=True,
    )
    # 中左大主木屋（严格按参考比例：檐口 ~3.0m、屋脊 ~5.2m，与右侧木屋形成约10m紧凑院落）
    _add_gable_cabin(
        "中左大主木屋", "arch_main_cabin",
        pos=[-1.40, 0.0, -24.00], rot_y=-2.0, width=6.20, wall_h=3.00, depth=13.00, ridge_extra=2.20,
        wall_col="#6E6054", roof_col_l="#D0D8E1", roof_col_r="#B8C2CC", is_ew_ridge=False,
    )
    # 后门左旁储物铁皮屋（与油罐紧挨在院门左侧）& 右排三栋木屋 + 右下门房（全部向内收拢）
    _add_shed_cabin(
        "后门旁储物小屋", "arch_back_gate_shed",
        pos=[3.60, 0.0, -38.50], rot_y=0.0, width=4.20, wall_h=2.20, depth=3.60,
        slope_deg=10.0, wall_col="#676A6E", roof_col="#C6CFD8",
    )
    _add_gable_cabin(
        "右后木屋", "arch_right_back_cabin",
        pos=[14.20, 0.0, -29.00], rot_y=-8.0, width=5.40, wall_h=2.35, depth=4.60, ridge_extra=1.45,
        wall_col="#65594E", roof_col_l="#C8D1DA", roof_col_r="#B6C0CA", is_ew_ridge=True,
    )
    _add_gable_cabin(
        "右中斜置木屋", "arch_right_mid_cabin",
        pos=[15.80, 0.0, -16.50], rot_y=-34.0, width=5.00, wall_h=2.45, depth=6.40, ridge_extra=1.65,
        wall_col="#685C51", roof_col_l="#CBD4DD", roof_col_r="#B8C2CC", is_ew_ridge=False,
    )
    _add_gable_cabin(
        "右前门边大木棚", "arch_right_fg_shed",
        pos=[16.20, 0.0, -2.50], rot_y=-32.0, width=4.40, wall_h=2.40, depth=5.00, ridge_extra=1.45,
        wall_col="#64584E", roof_col_l="#C7D0D9", roof_col_r="#B6C0CA", is_ew_ridge=False,
    )
    _add_shed_cabin(
        "右下角门房小棚", "arch_right_fg_booth",
        pos=[15.00, 0.0, 3.50], rot_y=-20.0, width=3.80, wall_h=2.40, depth=3.80,
        slope_deg=10.0, wall_col="#5F544A", roof_col="#CBD4DD",
    )

    # ==================== 3. 围栏（留出干净后门缺口 x=6.6..12.4）与原木堆/油罐（各 1 基础体） ====================
    _box("前左围栏墙段", [-4.60, 0.68, 0.40], [7.20, 1.36, 0.30], "#655E56", "arch_fence_fg_west", rot=[0.0, 12.0, 0.0], arch_role="prop")
    _box("后左栅栏墙块", [4.20, 0.66, -35.80], [4.80, 1.32, 0.30], "#655E56", "arch_fence_back_left", rot=[0.0, 2.0, 0.0], arch_role="prop")
    _box("后右栅栏墙块", [15.90, 0.66, -34.60], [7.00, 1.32, 0.30], "#676058", "arch_fence_back_right", rot=[0.0, -8.0, 0.0], arch_role="prop")
    _box("东侧外围栏上段", [19.80, 0.68, -22.50], [0.30, 1.36, 13.50], "#655E56", "arch_fence_east_01", rot=[0.0, 4.0, 0.0], arch_role="prop")
    _box("东侧外围栏下段", [19.40, 0.68, -7.00], [0.30, 1.36, 13.50], "#655E56", "arch_fence_east_02", rot=[0.0, 4.0, 0.0], arch_role="prop")

    _box("左前大原木垛", [-8.20, 0.42, 2.20], [4.20, 0.84, 2.80], "#756555", "prop_logs_fg_left", rot=[0.0, 8.0, 0.0], arch_role="prop")
    _box("前中立放木料垛", [-3.00, 0.40, 2.80], [2.20, 0.80, 3.40], "#786858", "prop_logs_fg_mid", rot=[0.0, 3.0, 0.0], arch_role="prop")
    _box("右中院原木垛", [13.60, 0.44, -9.20], [3.40, 0.88, 2.20], "#726252", "prop_logs_right_yard", rot=[0.0, -34.0, 0.0], arch_role="prop")
    _box("远后储木区大木堆", [7.60, 0.50, -50.00], [5.20, 1.00, 3.00], "#726252", "prop_logs_far_yard", rot=[0.0, 6.0, 0.0], arch_role="prop")
    stage.add_cylinder(
        name="后门左旁卧式油罐",
        pos=[3.80, 0.56, -35.80],
        size=[1.04, 2.40, 1.04],
        color="#586370",
        rot=[0.0, 0.0, 90.0],
        obj_id="prop_yard_fuel_tank",
        arch_role="prop",
        keep_color=True,
        protected=True,
    )

    # ==================== 4. 树木与远景林带（近中景 8 棵×2基础体=16 + 远景 5 个林带体块=5，总计 21<=30） ====================
    near_trees = [
        (-8.80, -45.50, 0.50, 3.00, 3.60, 8.20, "#4B6054"),
        (-6.20, -48.50, 0.48, 2.80, 3.20, 7.60, "#506559"),
        (9.50, -48.00, 0.52, 3.20, 3.80, 9.00, "#45594D"),
        (21.50, -44.50, 0.54, 3.60, 4.00, 10.00, "#41554A"),
        (24.50, -29.00, 0.56, 3.80, 4.20, 10.80, "#43574C"),
        (25.20, -15.50, 0.56, 3.80, 4.30, 11.20, "#465A4F"),
        (24.20, -2.50, 0.54, 3.60, 4.00, 10.20, "#42564B"),
        (21.50, 7.50, 0.52, 3.40, 3.80, 9.60, "#475B50"),
    ]
    for idx, (tx, tz, tdia, th_trunk, cdia, ch_crown, ccol) in enumerate(near_trees, start=1):
        stage.add_group(
            f"近中景松树_{idx:02d}",
            pos=[tx, 0.0, tz],
            obj_id=f"tree_near_{idx:02d}",
            arch_role="tree",
            keep_color=True,
            protected=True,
            parts=[
                _part("trunk", "cylinder", [0.0, round(th_trunk * 0.5, 3), 0.0], [tdia, th_trunk, tdia], "#4F4238"),
                _part("crown", "cone", [0.0, round(th_trunk * 0.85 + ch_crown * 0.5, 3), 0.0], [cdia, ch_crown, cdia], ccol),
            ],
        )

    forest_belts = [
        ("远景中轴深林带", [10.00, 4.20, -90.00], [56.00, 8.40, 12.00], [0.0, 2.0, 0.0], "#53665E", "forest_belt_horizon_center"),
        ("远景东北林带", [38.00, 5.20, -72.00], [40.00, 10.40, 16.00], [0.0, -14.0, 0.0], "#4E6159", "forest_belt_horizon_right"),
        ("门外北东深林带", [30.00, 4.80, -54.00], [26.00, 9.60, 14.00], [0.0, -18.0, 0.0], "#556860", "forest_belt_back_taiga"),
        ("右侧外围林带中段", [34.00, 5.40, -21.00], [14.00, 10.80, 40.00], [0.0, -6.0, 0.0], "#576A62", "forest_belt_east_mid"),
        ("右侧外围林带前段", [31.00, 5.40, 9.00], [14.00, 10.80, 26.00], [0.0, -8.0, 0.0], "#5A6D65", "forest_belt_east_fg"),
    ]
    for b_name, b_pos, b_size, b_rot, b_col, b_id in forest_belts:
        _box(b_name, b_pos, b_size, b_col, b_id, rot=b_rot, arch_role="forest_belt")

    # ==================== 5. 车辆（左面包车在视频原位、右吉普车顺右道穿门右拐，严格 <=6 基础体） ====================
    stage.add_group(
        "停靠面包车 UAZ-452",
        pos=[4.95, 0.0, -20.80],
        rot=[0.0, -64.0, 0.0],
        obj_id="veh_van_uaz452",
        arch_role="vehicle",
        keep_color=True,
        protected=True,
        parts=[
            _part("body", "box", [0.0, 0.78, 0.0], [1.94, 0.82, 4.36], "#5A6655"),
            _part(
                "cabin",
                "box",
                [0.0, 1.58, 0.02],
                [1.86, 0.86, 4.24],
                "#5A6655",
                surface_zones={
                    "py": "#616E5C",
                    "ny": "#4B5547",
                    "nz": {"fill": "#5A6655", "rects": [[0.08, 0.14, 0.84, 0.66, "#222A32"]]},
                    "pz": {"fill": "#5A6655", "rects": [[0.14, 0.16, 0.31, 0.50, "#222A32"], [0.55, 0.16, 0.31, 0.50, "#222A32"]]},
                    "px": {"fill": "#5A6655", "rects": [[0.08, 0.16, 0.24, 0.58, "#222A32"], [0.38, 0.16, 0.24, 0.58, "#222A32"], [0.68, 0.16, 0.24, 0.58, "#222A32"]]},
                    "nx": {"fill": "#5A6655", "rects": [[0.08, 0.16, 0.24, 0.58, "#222A32"], [0.38, 0.16, 0.24, 0.58, "#222A32"], [0.68, 0.16, 0.24, 0.58, "#222A32"]]},
                },
            ),
            *_wheel_parts(radius=0.37, width=0.26, ax_x=0.86, z_front=-1.34, z_rear=1.34),
        ],
    )

    # 主角军用越野吉普 UAZ-469（位于面包车右侧车道，开场半秒怠速喷尾气后按雪地道路确认上限 <=4.0 m/s 平滑穿后门右拐出村）
    jeep = stage.add_group(
        "主角越野吉普 UAZ-469",
        pos=[8.30, 0.0, -9.80],
        rot=[0.0, -4.0, 0.0],
        obj_id="veh_jeep_uaz469",
        arch_role="vehicle",
        align_tangent_y=True,
        keep_color=True,
        protected=True,
        parts=[
            _part(
                "body",
                "box",
                [0.0, 0.74, 0.0],
                [1.78, 0.72, 4.02],
                "#4A5941",
                surface_zones={
                    "nz": {"fill": "#4A5941", "rects": [[0.18, 0.22, 0.64, 0.44, "#23282D"], [0.04, 0.78, 0.92, 0.18, "#2C3238"]]},
                    "pz": {"fill": "#4A5941", "rects": [[0.08, 0.34, 0.14, 0.18, "#2A2E33"], [0.78, 0.34, 0.14, 0.18, "#2A2E33"], [0.04, 0.76, 0.92, 0.18, "#2C3238"]]},
                },
            ),
            _part(
                "cabin",
                "box",
                [0.0, 1.48, 0.48],
                [1.62, 0.84, 2.58],
                "#4A5941",
                surface_zones={
                    "py": "#4E5E44",
                    "ny": "#3E4B36",
                    "nz": {"fill": "#4A5941", "rects": [[0.09, 0.14, 0.82, 0.66, "#222A32"]]},
                    "pz": {"fill": "#4A5941", "rects": [[0.14, 0.16, 0.72, 0.58, "#222A32"]]},
                    "px": {"fill": "#4A5941", "rects": [[0.10, 0.14, 0.36, 0.64, "#222A32"], [0.54, 0.14, 0.36, 0.64, "#222A32"]]},
                    "nx": {"fill": "#4A5941", "rects": [[0.10, 0.14, 0.36, 0.64, "#222A32"], [0.54, 0.14, 0.36, 0.64, "#222A32"]]},
                },
            ),
            *_wheel_parts(radius=0.38, width=0.26, ax_x=0.80, z_front=-1.32, z_rear=1.18),
        ],
    )
    jeep.extra["motion_type"] = "veh_snow_crawl"
    jeep.extra["allow_snow_4mps"] = True
    jeep.extra["repro_source"] = "video_repro"
    jeep.extra["motion_source"] = "auto"
    jeep.add_keyframe(0.00, pos=[8.30, 0.0, -9.80], rot=[0.0, -4.0, 0.0], size=[1.0, 1.0, 1.0], interp="ease_in", interp_pos="ease_in", interp_rot="continuous")
    jeep.add_keyframe(0.50, pos=[8.32, 0.0, -10.10], rot=[0.0, -4.0, 0.0], size=[1.0, 1.0, 1.0], interp="continuous", interp_pos="continuous", interp_rot="continuous")
    jeep.add_keyframe(2.90, pos=[8.80, 0.0, -17.00], rot=[0.0, -3.5, 0.0], size=[1.0, 1.0, 1.0], interp="continuous", interp_pos="continuous", interp_rot="continuous")
    jeep.add_keyframe(5.55, pos=[9.30, 0.0, -26.80], rot=[0.0, -10.0, 0.0], size=[1.0, 1.0, 1.0], interp="linear", interp_pos="linear", interp_rot="continuous")
    jeep.add_keyframe(8.08, pos=[10.90, 0.0, -36.50], rot=[0.0, -38.0, 0.0], size=[1.0, 1.0, 1.0], interp="linear", interp_pos="linear", interp_rot="continuous", stop=False)

    # 尾气烟团（单一低分段球体 <=1：开场 0.00-0.50s 沿车尾后方 +Z 喷出可见白灰尾气团，随后在院中缓缓散尽）
    exhaust = stage.add_sphere(
        name="车尾排气管黑烟",
        pos=[8.32, 0.24, -7.72],
        size=[0.16, 0.14, 0.18],
        color="#3D4652",
        rot=[0.0, 0.0, 0.0],
        obj_id="fx_jeep_exhaust_smoke",
        arch_role="prop",
        keep_color=True,
        protected=True,
    )
    exhaust.extra["motion_source"] = "auto"
    exhaust.add_keyframe(0.00, pos=[8.32, 0.24, -7.72], size=[0.16, 0.14, 0.18], interp="ease_out")
    exhaust.add_keyframe(0.25, pos=[8.38, 0.31, -7.15], size=[0.68, 0.42, 0.78], interp="continuous")
    exhaust.add_keyframe(0.50, pos=[8.42, 0.40, -6.75], size=[0.92, 0.54, 1.05], interp="continuous")
    exhaust.add_keyframe(1.10, pos=[8.48, 0.52, -6.55], size=[0.46, 0.28, 0.50], interp="continuous")
    exhaust.add_keyframe(1.85, pos=[8.54, 0.62, -6.45], size=[0.01, 0.01, 0.01], interp="linear", stop=True)

    # 装车兵搬入左侧面包车尾箱的木箱（1基础体）与提包兵行李（1基础体）
    crate = _box(
        "装车木箱",
        pos=[2.79, 0.90, -19.75],
        size=[0.56, 0.42, 0.48],
        color="#9B8263",
        obj_id="prop_van_loading_crate",
        rot=[0.0, -64.0, 0.0],
        arch_role="prop",
    )
    crate.add_keyframe(0.00, pos=[2.79, 0.90, -19.75], rot=[0.0, -64.0, 0.0], interp="linear", stop=True)
    crate.add_keyframe(2.20, pos=[2.79, 0.90, -19.75], rot=[0.0, -64.0, 0.0], interp="ease_in_out", stop=True)
    crate.add_keyframe(4.80, pos=[3.51, 0.94, -20.10], rot=[0.0, -64.0, 0.0], interp="ease_in_out", stop=True)
    crate.add_keyframe(8.08, pos=[3.51, 0.94, -20.10], rot=[0.0, -64.0, 0.0], interp="linear", stop=True)

    bag = _box(
        "提包兵携带行李",
        pos=[5.35, 0.62, -18.52],
        size=[0.34, 0.42, 0.36],
        color="#82755A",
        obj_id="prop_soldier_duffle_bag",
        rot=[0.0, -28.0, 0.0],
        arch_role="prop",
    )
    bag.add_keyframe(0.00, pos=[5.35, 0.62, -18.52], rot=[0.0, -28.0, 0.0], interp="linear", stop=True)
    bag.add_keyframe(2.40, pos=[5.52, 0.62, -18.92], rot=[0.0, -28.0, 0.0], interp="continuous", stop=True)
    bag.add_keyframe(5.40, pos=[5.95, 0.62, -19.90], rot=[0.0, -28.0, 0.0], interp="continuous")
    bag.add_keyframe(8.08, pos=[6.12, 0.62, -20.30], rot=[0.0, -28.0, 0.0], interp="linear", stop=True)

    # ==================== 6. 两名球头人偶（不用走路算法；站在左侧面包车车尾与右侧旁，贴近视频相对站位） ====================
    def _pawn_stand_targets(lean_z: float = 0.0, drop_y: float = 0.0) -> Dict[str, List[float]]:
        t = copy.deepcopy(PRESET_IK_POSES["standing"]["targets"])
        t["pelvis"] = [0.0, 0.887 - drop_y, 0.0]
        t["chest"] = [0.0, 1.347 - drop_y * 1.4, lean_z]
        t["head_target"] = [0.0, 1.62 - drop_y * 1.5, 0.14 + lean_z * 1.25]
        return t

    pawn1 = stage.add_pawn(
        "装车士兵 (琥珀橙)",
        pos=[2.34, 0.0, -19.53],
        rot_y=116.0,
        pose="standing",
        scale=1.0,
        color="#D98236",
        char_id="char_soldier_loader",
    )
    pawn1.extra.update({"enable_walk_gait": False, "pawn_smooth_walk": False, "keep_color": True, "protected": True})
    pawn1.add_keyframe(0.00, pos=[2.34, 0.0, -19.53], rot_y=116.0, targets=_pawn_stand_targets(lean_z=0.34, drop_y=0.10), interp="linear")
    pawn1.add_keyframe(2.60, pos=[2.34, 0.0, -19.53], rot_y=116.0, targets=_pawn_stand_targets(lean_z=0.34, drop_y=0.10), interp="ease_in_out")
    pawn1.add_keyframe(4.80, pos=[2.52, 0.0, -19.62], rot_y=116.0, targets=_pawn_stand_targets(lean_z=0.40, drop_y=0.12), interp="ease_in_out")
    pawn1.add_keyframe(8.08, pos=[2.44, 0.0, -19.58], rot_y=116.0, targets=_pawn_stand_targets(lean_z=0.04, drop_y=0.0), interp="linear")

    pawn2 = stage.add_pawn(
        "提包士兵 (冷钢蓝)",
        pos=[5.25, 0.0, -18.25],
        rot_y=-28.0,
        pose="standing",
        scale=1.0,
        color="#4A78A8",
        char_id="char_soldier_carrier",
    )
    pawn2.extra.update({"enable_walk_gait": False, "pawn_smooth_walk": False, "keep_color": True, "protected": True})
    pawn2.add_keyframe(0.00, pos=[5.25, 0.0, -18.25], rot_y=-28.0, targets=_pawn_stand_targets(lean_z=0.08, drop_y=0.0), interp="linear")
    pawn2.add_keyframe(2.40, pos=[5.42, 0.0, -18.65], rot_y=-28.0, targets=_pawn_stand_targets(lean_z=0.10, drop_y=0.0), interp="continuous")
    pawn2.add_keyframe(5.40, pos=[5.85, 0.0, -19.63], rot_y=-28.0, targets=_pawn_stand_targets(lean_z=0.08, drop_y=0.0), interp="continuous")
    pawn2.add_keyframe(8.08, pos=[6.02, 0.0, -20.03], rot_y=-28.0, targets=_pawn_stand_targets(lean_z=0.04, drop_y=0.0), interp="ease_out")

    # ==================== 7. 丝滑低跟 -> 升起 -> 缓缓向后拉升至鸟瞰基准构图（全程无遮挡） ====================
    f32 = focal_mm_to_fov(32.0)
    f308 = focal_mm_to_fov(30.8)
    f292 = focal_mm_to_fov(29.2)
    f285 = focal_mm_to_fov(28.5)
    stage.set_camera(
        pos=[8.12, 0.88, -5.35],
        target=[8.38, 0.72, -14.20],
        focal_mm=32.0,
        fov=f32,
        aspect_ratio="2.33:1",
        export_width=aspect_w,
        export_height=aspect_h,
    )
    stage.add_camera_keyframe(0.00, pos=[8.12, 0.88, -5.35], target=[8.38, 0.72, -14.20], focal_mm=32.0, fov=f32, interp="continuous")
    stage.add_camera_keyframe(0.62, pos=[8.38, 0.98, -4.45], target=[8.42, 0.82, -15.50], focal_mm=32.0, fov=f32, interp="continuous")
    stage.add_camera_keyframe(2.35, pos=[10.05, 2.75, 0.40], target=[6.20, 1.35, -21.00], focal_mm=30.8, fov=f308, interp="continuous")
    stage.add_camera_keyframe(4.95, pos=[12.80, 8.80, 12.20], target=[3.60, 0.90, -21.50], focal_mm=29.2, fov=f292, interp="continuous")
    stage.add_camera_keyframe(8.08, pos=[15.40, 16.20, 26.00], target=[3.40, 0.40, -19.50], focal_mm=28.5, fov=f285, interp="continuous", stop=False)

    scene_dict = stage.to_dict()
    scene_dict["export_plan"] = {
        "confirmed": True,
        "confirmed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "priority_mode": "balanced",
        "priority_label": "均衡",
        "material_style": "matte_blockout_previz",
        "compositions": [
            {"t": 0.50, "label": "0.50s 尾气起幅：低机位缓停等待车尾喷出 0.5s 可见尾气"},
            {"t": 2.35, "label": "2.35s 缓起后右拉：沿东侧空廊向后右方平稳升起，完整带出院落主屋、面包车与装车士兵"},
            {"t": 4.95, "label": "4.95s 吊臂爬升：右后上方平稳推高，全景展开冰封河岸、哨塔与门前吉普直行"},
            {"t": 8.00, "label": "8.00s 高空落幅：左冰河、11栋无窗木屋、林带与吉普穿后门右拐出村落幅"},
        ],
        "preserve_existing_scene": True,
        "use_area_and_visible_budget": True,
    }
    return scene_dict
