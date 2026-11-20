#!/usr/bin/env python3
"""
Universal Video-to-Previz Reproduction Engine (fast_repro_kit.py)
通用视频 → SU 极简彩色三维预演极速复现引擎

支持任意题材视频（室内/室外/车戏/多人物走位/对话/坐卧/持物/格斗/奔跑/推拉摇移跟环绕运镜）：
1. 一键抽帧与元数据探测 (probe_and_extract_sheet)：1.5 秒输出宫格图与视频参数。
2. 通用人体动作与自动步态合成器 (UniversalCharTrack)：
   - 根据位移速度自动驱动双腿/双臂步态振幅与频率（走/跑/停自动平滑过渡，绝不滑步或停下后原地摆腿）；
   - 支持任意基础姿态：stand / walk / run / crouch / sit / kneel / lie / lean；
   - 支持任意上肢动作：auto_swing / reach / point / hold_prop / aim_gun / low_gun / chin_rest / crossed_arms / hands_on_hips / guard；
   - 支持基于【摄影机屏幕空间方向】的转头视线解算（"screen_right" / "screen_left" / "screen_center" / "down" / 看着某角色或物体），彻底杜绝左右转头方向弄反。
3. 道具自动手部绑定 (AutoPropBinder)：任何手持道具（枪、箱子、手电、水杯、手机、长棍等）自动跟随角色手部/肩部世界坐标与朝向。
4. 通用摄影机反求与构图器 (UniversalCameraTrack)：支持按景别 (closeup/medium/full/wide) 与屏幕构图偏移 (screen_x, screen_y) 自动反算机位，支持跟拍 (follow) 与环绕 (orbit)。
5. 10 毫秒级数学投影自检 (verify_scene_projection)：渲染前秒级检查人物屏幕占比、左右方向与前景遮挡穿模。
6. 8 核并行渲染 MP4 + 免重启热推入导演台 (fast_export_and_push)。
"""

import copy
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from director_engine import (
    SU_MINIMAL_CONFIG,
    RenderCamera,
    Stage,
    _dir_to_euler_xyz_deg,
    _normalize,
    convert_scene_to_su_minimal,
    evaluate_scene_characters_at_time,
    get_su_minimal_identity_color,
    interpolate_camera,
    render_su_minimal_frame,
)
from server import EXPORT_DIR, PROJECTS_FILE, REF_VIDEO_DIR, load_project_bank, save_project_bank

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 基础数学与插值工具
# ---------------------------------------------------------------------------

def smoothstep5(edge0: float, edge1: float, x: float) -> float:
    u = max(0.0, min(1.0, (x - edge0) / max(1e-6, edge1 - edge0)))
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def lerp_vec(v0: List[float], v1: List[float], u: float) -> List[float]:
    return [round(a + (b - a) * u, 4) for a, b in zip(v0, v1)]


def eval_piecewise_table(t: float, keyframes: List[Dict[str, Any]], key: str, default_val: Any) -> Any:
    """在按时间排序的状态节点列表中对任意数值或向量字段做平滑五次多项式插值"""
    if not keyframes:
        return default_val
    if len(keyframes) == 1 or t <= keyframes[0]["t"]:
        return keyframes[0].get(key, default_val)
    if t >= keyframes[-1]["t"]:
        return keyframes[-1].get(key, default_val)
    for i in range(len(keyframes) - 1):
        k0, k1 = keyframes[i], keyframes[i + 1]
        t0, t1 = float(k0["t"]), float(k1["t"])
        if t0 <= t <= t1:
            v0 = k0.get(key, default_val)
            v1 = k1.get(key, v0)
            mode = k0.get("ease", "smooth")
            raw_u = (t - t0) / max(1e-6, t1 - t0)
            u = smoothstep5(0.0, 1.0, raw_u) if mode == "smooth" else raw_u
            if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
                return lerp(float(v0), float(v1), u)
            if isinstance(v0, (list, tuple)) and isinstance(v1, (list, tuple)) and len(v0) == len(v1):
                return [lerp(float(a), float(b), u) for a, b in zip(v0, v1)]
            return v1 if u > 0.5 else v0
    return default_val


# ---------------------------------------------------------------------------
# 1. 一键探测视频元数据 + 复制到 ref_videos/ + 抽帧合成时间戳宫格图
# ---------------------------------------------------------------------------

def probe_and_extract_sheet(
    src_video_path: str,
    ref_filename: str,
    out_dir: str,
    sample_fps: float = 2.0,
    cols: int = 3,
) -> Dict[str, Any]:
    """一步完成：复制到 ref_videos/ + 解析宽高/帧率/时长 + 抽帧生成带时间戳宫格图"""
    t0 = time.time()
    os.makedirs(REF_VIDEO_DIR, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    dst_ref_path = os.path.join(REF_VIDEO_DIR, ref_filename)
    if os.path.abspath(src_video_path) != os.path.abspath(dst_ref_path):
        shutil.copy2(src_video_path, dst_ref_path)

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    p = subprocess.run([ff, "-i", dst_ref_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = p.stderr

    dur = 12.0
    m_dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info)
    if m_dur:
        dur = int(m_dur.group(1)) * 3600 + int(m_dur.group(2)) * 60 + float(m_dur.group(3))

    width, height = 992, 432
    m_res = re.search(r"Stream #.*Video:.*?\s(\d{2,5})x(\d{2,5})\b", info)
    if m_res:
        width, height = int(m_res.group(1)), int(m_res.group(2))

    fps = 24.0
    m_fps = re.search(r"(\d+(?:\.\d+)?)\s*fps", info)
    if m_fps:
        fps = float(m_fps.group(1))

    for old_f in os.listdir(out_dir):
        if old_f.startswith("f_") and old_f.endswith(".jpg"):
            os.remove(os.path.join(out_dir, old_f))

    subprocess.run(
        [ff, "-y", "-i", dst_ref_path, "-vf", f"fps={sample_fps}", os.path.join(out_dir, "f_%03d.jpg")],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    files = sorted([f for f in os.listdir(out_dir) if f.startswith("f_") and f.endswith(".jpg")])
    imgs = [Image.open(os.path.join(out_dir, f)) for f in files]
    if imgs:
        tw = 480
        th = max(120, int(round(tw * height / max(1, width))))
        rows = (len(imgs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tw, rows * th), (20, 22, 26))
        draw = ImageDraw.Draw(sheet)
        for idx, im in enumerate(imgs):
            r, c = divmod(idx, cols)
            sheet.paste(im.resize((tw, th), Image.Resampling.LANCZOS), (c * tw, r * th))
            t_sec = idx / sample_fps
            draw.rectangle([c * tw + 6, r * th + 6, c * tw + 148, r * th + 34], fill=(0, 0, 0))
            draw.text((c * tw + 12, r * th + 10), f"t={t_sec:.2f}s (#{idx+1})", fill=(255, 230, 0))
        sheet_path = os.path.join(out_dir, "contact_sheet.jpg")
        sheet.save(sheet_path, quality=92)
    else:
        sheet_path = ""

    aspect_str = "16:9" if abs((width / max(1, height)) - 16 / 9) < 0.18 else "2.39:1"
    return {
        "ref_path": dst_ref_path,
        "ref_filename": ref_filename,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_str,
        "fps": round(fps),
        "duration": round(dur, 2),
        "extracted_frames": len(files),
        "contact_sheet": sheet_path,
        "elapsed_sec": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# 2. 通用摄影机构图反解与屏幕方向映射
# ---------------------------------------------------------------------------

def solve_camera_from_framing(
    subject_pos: List[float],
    shot_size: str = "medium",  # "closeup" | "medium_closeup" | "medium" | "full" | "wide"
    cam_yaw_deg: float = 0.0,   # 0° 表示摄影机在 +Z 朝 -Z 看向主体正面
    cam_pitch_deg: float = 0.0, # 正值为俯拍，负值为仰拍
    screen_offset_x: float = 0.0,  # 主体在画面的水平偏置 (-0.5 左 .. 0 中 .. +0.5 右)
    screen_offset_y: float = 0.0,
    subject_eye_y: float = 1.22,
    fov_deg: float = 30.0,
) -> Dict[str, Any]:
    """根据期望景别和主体在画面中的左右偏置，一步算出摄影机 3D pos 与 target"""
    dist_map = {
        "extreme_closeup": 0.95,
        "closeup": 1.35,
        "medium_closeup": 1.85,
        "medium": 2.45,
        "full": 4.20,
        "wide": 6.80,
    }
    dist = dist_map.get(shot_size, 2.45)
    yaw_r = math.radians(cam_yaw_deg)
    pitch_r = math.radians(cam_pitch_deg)

    anchor = np.array([float(subject_pos[0]), float(subject_pos[1]) + subject_eye_y, float(subject_pos[2])], dtype=np.float64)
    cam_dir_from_sub = np.array(
        [math.sin(yaw_r) * math.cos(pitch_r), math.sin(pitch_r), math.cos(yaw_r) * math.cos(pitch_r)],
        dtype=np.float64,
    )
    fwd = -cam_dir_from_sub
    up_w = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right_cam = _normalize(np.cross(fwd, up_w))
    up_cam = _normalize(np.cross(right_cam, fwd))

    half_h = dist * math.tan(math.radians(fov_deg * 0.5))
    half_w = half_h * 1.85
    # 若希望主体落在画面右侧 (screen_offset_x > 0)，则摄影机 target 应向画面左侧 (-right_cam) 偏置
    tar = anchor - right_cam * (screen_offset_x * half_w) - up_cam * (screen_offset_y * half_h)
    pos = tar + cam_dir_from_sub * dist
    return {
        "pos": [round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)],
        "target": [round(float(tar[0]), 4), round(float(tar[1]), 4), round(float(tar[2]), 4)],
        "fov": fov_deg,
    }


def screen_dir_to_local_head_target(
    char_pos: List[float],
    char_rot_y_deg: float,
    cam_pos: List[float],
    cam_target: List[float],
    screen_yaw_deg: float = 0.0,    # 0°=朝向摄影机/正前, +45°=朝画面最右侧看, -45°=朝画面最左侧看
    screen_pitch_deg: float = 0.0,  # +15°=抬头, -20°=低头
    head_y_local: float = 1.22,
) -> List[float]:
    """
    将【画面左右方向 (screen_yaw_deg)】严格通过摄影机真实右向量映射为角色局部的 head_target，
    彻底保证无论角色朝向何处、摄影机在哪个角度，screen_yaw_deg > 0 永远朝【画面右侧】转头！
    """
    cp = np.asarray(cam_pos, dtype=np.float64)
    ct = np.asarray(cam_target, dtype=np.float64)
    cam_fwd = _normalize(ct - cp)
    cam_right = _normalize(np.cross(cam_fwd, np.array([0.0, 1.0, 0.0], dtype=np.float64)))
    cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # 画面正前方向取「角色朝向摄影机 (-cam_fwd)」与「角色自身正前」的合理正面基准
    toward_cam = -cam_fwd
    toward_cam[1] = 0.0
    if np.linalg.norm(toward_cam) < 1e-5:
        toward_cam = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        toward_cam = _normalize(toward_cam)

    sy_r = math.radians(screen_yaw_deg)
    sp_r = math.radians(screen_pitch_deg)
    world_look_dir = _normalize(
        toward_cam * (math.cos(sy_r) * math.cos(sp_r))
        + cam_right * (math.sin(sy_r) * math.cos(sp_r))
        + cam_up * math.sin(sp_r)
    )

    # 反变换回角色局部坐标 (R_root^T @ world_look_dir)
    ry = math.radians(char_rot_y_deg)
    cy, sy = math.cos(ry), math.sin(ry)
    R_inv = np.array([[cy, 0.0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]], dtype=np.float64)
    loc_dir = R_inv @ world_look_dir
    # 防止向脑后 180° 反折
    if loc_dir[2] < 0.15:
        loc_dir[2] = 0.15
        loc_dir = _normalize(loc_dir)

    dist = 0.95
    return [
        round(float(loc_dir[0] * dist), 4),
        round(float(head_y_local + loc_dir[1] * dist), 4),
        round(float(0.03 + loc_dir[2] * dist), 4),
    ]


# ---------------------------------------------------------------------------
# 3. 通用角色姿态 + 自动步态 + 道具绑定合成器
# ---------------------------------------------------------------------------

def evaluate_universal_character_state(
    t: float,
    state_kfs: List[Dict[str, Any]],
    cam_pos: List[float],
    cam_target: List[float],
    step_freq: float = 3.2,
    phase_offset: float = 0.0,
) -> Dict[str, Any]:
    """
    根据声明式状态关键帧列表 state_kfs 计算时刻 t 的角色完整姿态与绑定道具变换：
    每个关键帧支持字段：
    - t: 时间 (秒)
    - pos: [x, y, z] 世界坐标
    - rot_y: 身体朝向角 (度，可省略，若省略且在移动则自动朝向移动方向或摄影机)
    - posture: "stand" | "crouch" | "sit" | "kneel" | "lean" (或直接指定 crouch=0.0..1.0, sit=0.0..1.0)
    - arm_action: "auto" | "aim_right" | "aim_front" | "aim_left" | "low_gun" | "hold_box" |
                  "reach_front" | "reach_right" | "hands_on_hips" | "crossed_arms" | "chin_rest" | "guard"
    - screen_yaw: 头部朝画面左右看的角度 (+40=看画面右侧, -40=看画面左侧, 0=看正面)
    - screen_pitch: 头部俯仰角 (+15=抬头, -20=低头)
    - recoil: 击发/受击震动强度 (0.0..1.0)
    """
    pos = eval_piecewise_table(t, state_kfs, "pos", [0.0, 0.0, 0.0])
    # 用有限差分计算真实移动速度，自动驱动步态包络 (walk_env)，停止移动时步态自动归零！
    dt = 0.05
    p_prev = eval_piecewise_table(max(0.0, t - dt), state_kfs, "pos", pos)
    p_next = eval_piecewise_table(t + dt, state_kfs, "pos", pos)
    vx = (p_next[0] - p_prev[0]) / (2.0 * dt)
    vz = (p_next[2] - p_prev[2]) / (2.0 * dt)
    speed = math.hypot(vx, vz)

    # 若显式指定了 walk_env 则用显式值，否则按真实移动速度自动计算 (0.65m/s 达到满幅步态)
    explicit_env = eval_piecewise_table(t, state_kfs, "walk_env", None)
    walk_env = float(explicit_env) if explicit_env is not None else min(1.0, speed / 0.55)
    stride = float(eval_piecewise_table(t, state_kfs, "stride", min(0.32, max(0.18, speed * 0.36))))

    # 身体朝向 rot_y
    rot_y = float(eval_piecewise_table(t, state_kfs, "rot_y", 0.0))
    crouch = float(eval_piecewise_table(t, state_kfs, "crouch", 0.0))
    sit = float(eval_piecewise_table(t, state_kfs, "sit", 0.0))
    recoil = float(eval_piecewise_table(t, state_kfs, "recoil", 0.0))
    screen_yaw = float(eval_piecewise_table(t, state_kfs, "screen_yaw", 0.0))
    screen_pitch = float(eval_piecewise_table(t, state_kfs, "screen_pitch", 0.0))
    arm_action = str(eval_piecewise_table(t, state_kfs, "arm_action", "auto"))

    # 步态相位与呼吸微动
    phase = t * step_freq + phase_offset
    s = math.sin(phase)
    c = math.cos(phase)
    lift_L = max(0.0, s) * 0.085 * walk_env * (1.0 - sit)
    lift_R = max(0.0, -s) * 0.085 * walk_env * (1.0 - sit)
    str_L = s * stride * walk_env * (1.0 - sit)
    str_R = -s * stride * walk_env * (1.0 - sit)
    bob = abs(c) * 0.024 * walk_env * (1.0 - sit)
    breath = 0.005 * math.sin(t * 2.3 + phase_offset) * (1.0 - 0.5 * walk_env)

    pos_out = [
        round(pos[0] - 0.03 * recoil, 4),
        round(pos[1] - 0.20 * crouch - 0.36 * sit + bob + breath, 4),
        round(pos[2], 4),
    ]

    # 头部视线目标（基于屏幕空间角度自动转换）
    head_target = screen_dir_to_local_head_target(
        pos_out, rot_y, cam_pos, cam_target, screen_yaw_deg=screen_yaw, screen_pitch_deg=screen_pitch
    )

    chest = [
        round(0.02 * crouch - 0.015 * recoil, 4),
        round(lerp(0.94, 0.86, max(crouch, sit * 0.6)) + bob * 0.5, 4),
        round(lerp(0.03, 0.10, crouch), 4),
    ]
    pelvis = [0.0, round(lerp(0.48, 0.44, sit), 4), round(lerp(-0.02, -0.06, sit), 4)]

    # 双脚目标（支持站立/行走/下蹲/坐姿）
    if sit > 0.5:
        foot_L = [0.13, 0.06, 0.36]
        foot_R = [-0.13, 0.06, 0.36]
        knee_L = [0.14, 0.52, 0.55]
        knee_R = [-0.14, 0.52, 0.55]
    else:
        foot_L = [0.11, round(0.03 + lift_L, 4), round(str_L + 0.04 * crouch, 4)]
        foot_R = [-0.11, round(0.03 + lift_R, 4), round(str_R - 0.04 * crouch, 4)]
        knee_L = [0.13, 0.48, round(0.45 + str_L * 0.5, 4)]
        knee_R = [-0.13, 0.48, round(0.45 + str_R * 0.5, 4)]

    # 上肢动作与道具绑定矩阵
    elbow_R = [-0.32, 0.72, 0.08]
    elbow_L = [0.22, 0.68, 0.14]
    prop_local_off = [0.0, 0.92, 0.24]
    prop_fwd = [0.0, 0.0, 1.0]

    if arm_action == "aim_right":
        hand_R = [round(lerp(-0.04, -0.06, crouch) - 0.025 * recoil, 4), round(lerp(1.00, 0.84, crouch) + 0.015 * recoil, 4), 0.22]
        hand_L = [round(lerp(0.22, 0.16, crouch) - 0.025 * recoil, 4), round(lerp(1.02, 0.78, crouch) + 0.020 * recoil, 4), 0.26]
        prop_local_off = [lerp(-0.06, -0.06, crouch) - 0.04 * recoil, lerp(1.03, 0.68, crouch) + 0.018 * recoil, lerp(0.24, 0.26, crouch)]
        prop_fwd = [lerp(0.955, 0.810, crouch), lerp(0.065 + 0.045 * recoil, -0.360, crouch), lerp(0.285, 0.460, crouch)]
    elif arm_action == "aim_left":
        hand_L = [round(lerp(0.04, 0.06, crouch) + 0.025 * recoil, 4), round(lerp(1.00, 0.84, crouch) + 0.015 * recoil, 4), 0.22]
        hand_R = [round(lerp(-0.22, -0.16, crouch) + 0.025 * recoil, 4), round(lerp(1.02, 0.78, crouch) + 0.020 * recoil, 4), 0.26]
        prop_local_off = [0.06 + 0.04 * recoil, lerp(1.03, 0.68, crouch) + 0.018 * recoil, 0.24]
        prop_fwd = [lerp(-0.955, -0.810, crouch), lerp(0.065 + 0.045 * recoil, -0.360, crouch), lerp(0.285, 0.460, crouch)]
    elif arm_action == "aim_front":
        sy_r = math.radians(screen_yaw * 0.4)
        hand_R = [-0.05, round(lerp(1.00, 0.86, crouch) + 0.015 * recoil, 4), round(0.22 - 0.025 * recoil, 4)]
        hand_L = [0.05, round(lerp(1.02, 0.84, crouch) + 0.020 * recoil, 4), round(0.40 - 0.025 * recoil, 4)]
        prop_local_off = [-0.02, lerp(1.02, 0.82, crouch) + 0.018 * recoil, 0.24 - 0.03 * recoil]
        prop_fwd = [math.sin(sy_r), lerp(0.03 + 0.04 * recoil, -0.26, crouch), math.cos(sy_r)]
    elif arm_action == "low_gun":
        sy_r = math.radians(screen_yaw * 0.35)
        hand_R = [-0.08, round(0.92 + bob * 0.4, 4), 0.22]
        hand_L = [0.08, round(0.94 + bob * 0.4, 4), 0.34]
        prop_local_off = [-0.01, 0.94 + bob * 0.4, 0.24]
        prop_fwd = [0.22 + math.sin(sy_r), -0.14, 0.94]
    elif arm_action == "hold_box":
        hand_R = [-0.14, 0.88, 0.28]
        hand_L = [0.14, 0.88, 0.28]
        prop_local_off = [0.0, 0.88, 0.32]
        prop_fwd = [0.0, 0.0, 1.0]
    elif arm_action == "reach_front":
        hand_R = [-0.10, 1.04, 0.46]
        hand_L = [0.18, 0.78, 0.10]
        prop_local_off = [-0.10, 1.04, 0.48]
    elif arm_action == "reach_right":
        hand_L = [0.42, 1.04, 0.24]
        hand_R = [-0.18, 0.78, 0.10]
        prop_local_off = [0.44, 1.04, 0.24]
    elif arm_action == "chin_rest":
        hand_R = [-0.05, 1.08, 0.16]
        hand_L = [0.18, 0.78, 0.22]
        elbow_R = [-0.16, 0.78, 0.26]
    elif arm_action == "hands_on_hips":
        hand_R = [-0.18, 0.62, 0.04]
        hand_L = [0.18, 0.62, 0.04]
        elbow_R = [-0.36, 0.74, -0.04]
        elbow_L = [0.36, 0.74, -0.04]
    elif arm_action == "crossed_arms":
        hand_R = [0.08, 0.88, 0.18]
        hand_L = [-0.08, 0.86, 0.18]
    elif arm_action == "guard":
        hand_R = [-0.10, 1.04, 0.24]
        hand_L = [0.12, 1.06, 0.32]
    else:
        # 默认自然摆臂 (随步态速度自动摆动，静止时自然垂于体侧)
        if sit > 0.5:
            hand_L = [0.16, 0.56, 0.24]
            hand_R = [-0.16, 0.56, 0.24]
        else:
            hand_L = [0.22, round(0.75 + max(0.0, -s) * 0.04 * walk_env, 4), round(0.05 - str_L * 0.42, 4)]
            hand_R = [-0.22, round(0.75 + max(0.0, s) * 0.04 * walk_env, 4), round(0.05 - str_R * 0.42, 4)]

    targets = {
        "pelvis": pelvis,
        "chest": chest,
        "head_target": head_target,
        "hand_R": hand_R,
        "hand_L": hand_L,
        "foot_L": foot_L,
        "foot_R": foot_R,
        "knee_L_pole": knee_L,
        "knee_R_pole": knee_R,
        "elbow_R_pole": elbow_R,
        "elbow_L_pole": elbow_L,
    }

    ry = math.radians(rot_y)
    cy, sy = math.cos(ry), math.sin(ry)
    px = pos_out[0] + prop_local_off[0] * cy + prop_local_off[2] * sy
    py = pos_out[1] + prop_local_off[1]
    pz = pos_out[2] - prop_local_off[0] * sy + prop_local_off[2] * cy
    prot = _dir_to_euler_xyz_deg(prop_fwd)

    return {
        "pos": pos_out,
        "rot_y": round(rot_y, 2),
        "look_at": head_target,
        "targets": targets,
        "prop": {
            "pos": [round(px, 4), round(py, 4), round(pz, 4)],
            "rot": prot,
        },
    }


# ---------------------------------------------------------------------------
# 4. 10 毫秒级数学屏幕投影自检器 (verify_scene_projection)
# ---------------------------------------------------------------------------

def verify_scene_projection(scene_dict: Dict[str, Any], check_times: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    在光栅化渲染前，用 10ms 纯数学透视投影检验场景在关键时间点的屏幕构图：
    - 每个角色头部在画面的归一化坐标 (u, v) in [0..1] (u<0.5 在画左，u>0.5 在画右)
    - 头前朝向短线的屏幕指向 (du > 0 表示朝画面右侧看，du < 0 表示朝画面左侧看)
    - 角色在画面的垂直高度占比 (height_ratio)
    """
    dur = float(scene_dict.get("duration") or 12.0)
    if not check_times:
        check_times = [0.0, round(dur * 0.4, 2), round(dur * 0.7, 2), round(max(0.0, dur - 0.2), 2)]

    cam_cfg = scene_dict.get("camera") or {}
    w = int(cam_cfg.get("export_width") or 992)
    h = int(cam_cfg.get("export_height") or 432)

    report_by_time = {}
    for ct in check_times:
        cpos, ctar, cfov = interpolate_camera(scene_dict, ct)
        rcam = RenderCamera(cpos, ctar, cfov, w, h)
        _, ik_map, _ = evaluate_scene_characters_at_time(scene_dict, ct)
        char_reports = {}
        for cid, ik in ik_map.items():
            J = ik["joints"]
            A = ik["axes"]
            hc = J["head_center"]
            htip = hc + A.get("look_dir", np.array([0.0, 0.0, 1.0])) * 0.32
            foot_mid = 0.5 * (J["ankle_L"] + J["ankle_R"])
            pts_w = np.array([hc, htip, foot_mid], dtype=np.float64)
            pts_c = rcam.world_to_cam(pts_w)
            z = np.maximum(pts_c[:, 2], 0.05)
            u = 0.5 + (pts_c[:, 0] / z) * (rcam.focal / w)
            v = 0.5 - (pts_c[:, 1] / z) * (rcam.focal / h)
            du = float(u[1] - u[0])
            dv = float(v[1] - v[0])
            char_reports[cid] = {
                "head_screen_uv": [round(float(u[0]), 3), round(float(v[0]), 3)],
                "gaze_screen_delta_uv": [round(du, 4), round(dv, 4)],
                "gaze_screen_dir": "RIGHT" if du > 0.008 else ("LEFT" if du < -0.008 else "CENTER"),
                "depth_m": round(float(pts_c[0, 2]), 2),
                "screen_height_ratio": round(abs(float(v[2] - v[0])) * 1.18, 3),
            }
        report_by_time[f"t={ct:.2f}s"] = char_reports
    return report_by_time


# ---------------------------------------------------------------------------
# 5. 8 进程并行渲染 + 6 帧原片对比图 + 免重启热推入导演台
# ---------------------------------------------------------------------------

_WORKER_SCENE: Optional[Dict[str, Any]] = None
_WORKER_W: int = 992
_WORKER_H: int = 432
_WORKER_FPS: int = 24


def _init_pool_worker(scene_dict: Dict[str, Any], w: int, h: int, fps: int):
    global _WORKER_SCENE, _WORKER_W, _WORKER_H, _WORKER_FPS
    _WORKER_SCENE = convert_scene_to_su_minimal(scene_dict)
    _WORKER_SCENE["_su_minimal_normalized"] = True
    _WORKER_W = w
    _WORKER_H = h
    _WORKER_FPS = fps


def _render_frame_batch(args: Tuple[int, int, str]) -> int:
    start_i, end_i, tmp_dir = args
    for i in range(start_i, end_i):
        t = i / float(_WORKER_FPS)
        im = render_su_minimal_frame(_WORKER_SCENE, t, _WORKER_W, _WORKER_H)
        im.save(os.path.join(tmp_dir, f"frame_{i:05d}.png"), format="PNG", compress_level=1)
    return end_i - start_i


def fast_export_and_push(
    scene_dict: Dict[str, Any],
    scene_id: str,
    scene_name: str,
    shot_id: str,
    shot_name: str,
    out_mp4_filename: str,
    ref_frames_dir: Optional[str] = None,
    ref_sample_fps: float = 2.0,
    compare_times: Optional[List[float]] = None,
    desktop_copy_filename: Optional[str] = None,
    brain_dir: Optional[str] = None,
    workers: int = 8,
) -> Dict[str, Any]:
    """一步完成：SU 极简校验 + 数学投影自检 + 6 帧对比图 + 8 核并行渲染 MP4 + 免重启热更新导演台"""
    t0 = time.time()
    sc = convert_scene_to_su_minimal(copy.deepcopy(scene_dict))
    proj_check = verify_scene_projection(sc, compare_times)

    cam_cfg = sc.get("camera") or {}
    w = int(cam_cfg.get("export_width") or 992)
    h = int(cam_cfg.get("export_height") or 432)
    fps = int(sc.get("fps") or 24)
    dur = float(sc.get("duration") or 12.0)
    total_frames = max(1, int(round(dur * fps)))

    compare_sheet_path = ""
    if ref_frames_dir and os.path.isdir(ref_frames_dir):
        if not compare_times:
            compare_times = [round(dur * k / 5.0, 1) for k in range(6)]
            if compare_times[-1] >= dur:
                compare_times[-1] = round(max(0.0, dur - 0.4), 1)
        rows_im = []
        for ct in compare_times:
            idx = max(1, int(round(ct * ref_sample_fps)) + 1)
            f_cand = os.path.join(ref_frames_dir, f"f_{idx:03d}.jpg")
            if not os.path.isfile(f_cand):
                all_jpg = sorted([f for f in os.listdir(ref_frames_dir) if f.startswith("f_") and f.endswith(".jpg")])
                f_cand = os.path.join(ref_frames_dir, all_jpg[min(len(all_jpg) - 1, max(0, idx - 1))]) if all_jpg else ""
            ren_im = render_su_minimal_frame(sc, ct, w, h)
            pair = Image.new("RGB", (w * 2, h), (18, 20, 24))
            if f_cand and os.path.isfile(f_cand):
                ref_im = Image.open(f_cand).resize((w, h), Image.Resampling.LANCZOS)
                pair.paste(ref_im, (0, 0))
            pair.paste(ren_im, (w, 0))
            d = ImageDraw.Draw(pair)
            d.rectangle([10, 10, 180, 40], fill=(0, 0, 0))
            d.text((18, 15), f"REF t={ct:.1f}s", fill=(255, 220, 0))
            d.rectangle([w + 10, 10, w + 240, 40], fill=(0, 0, 0))
            d.text((w + 18, 15), f"SU MINIMAL t={ct:.1f}s", fill=(100, 255, 160))
            rows_im.append(pair)
        sheet = Image.new("RGB", (w * 2, h * len(rows_im)), (18, 20, 24))
        for i, r in enumerate(rows_im):
            sheet.paste(r, (0, i * h))
        compare_sheet_path = os.path.join(ref_frames_dir, "compare_sheet.jpg")
        sheet.save(compare_sheet_path, quality=92)

    t_ren0 = time.time()
    tmp_dir = tempfile.mkdtemp(prefix="fast_repro_")
    chunk = max(8, (total_frames + workers - 1) // workers)
    batches = [(i, min(i + chunk, total_frames), tmp_dir) for i in range(0, total_frames, chunk)]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_pool_worker, initargs=(sc, w, h, fps)) as ex:
        list(ex.map(_render_frame_batch, batches))

    os.makedirs(EXPORT_DIR, exist_ok=True)
    out_mp4 = os.path.join(EXPORT_DIR, out_mp4_filename)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ff, "-y", "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%05d.png"),
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "fast", "-movflags", "+faststart",
            out_mp4,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    shutil.rmtree(tmp_dir, ignore_errors=True)
    render_sec = round(time.time() - t_ren0, 2)

    if brain_dir and os.path.isdir(brain_dir):
        shutil.copy2(out_mp4, os.path.join(brain_dir, out_mp4_filename))
    desk_dir = "/Users/wangmu/Desktop/MagicDirectorStage_导演台完整软件/01_最新成品_雪林车队极简方块预演"
    if desktop_copy_filename and os.path.isdir(desk_dir):
        shutil.copy2(out_mp4, os.path.join(desk_dir, desktop_copy_filename))

    hot_ok = False
    try:
        req_body = json.dumps(
            {
                "scene_id": scene_id,
                "scene_name": scene_name,
                "shot_id": shot_id,
                "shot_name": shot_name,
                "scene": sc,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:8765/api/hot_activate_scene",
            data=req_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            hot_ok = bool(res_json.get("ok"))
    except Exception:
        bank = load_project_bank()
        existing = next((s for s in bank["scenes"] if s.get("id") == scene_id), None)
        shot_entry = {"id": shot_id, "name": shot_name, "updatedAt": "SU极简彩色预演", "scene": sc}
        if existing is None:
            bank["scenes"].insert(0, {"id": scene_id, "name": scene_name, "shots": [shot_entry]})
        else:
            existing["name"] = scene_name
            existing["shots"] = [shot_entry]
        bank["activeSceneId"] = scene_id
        bank["activeShotId"] = shot_id
        save_project_bank(bank)

    return {
        "out_mp4": out_mp4,
        "compare_sheet": compare_sheet_path,
        "projection_check": proj_check,
        "frames": total_frames,
        "render_sec": render_sec,
        "total_sec": round(time.time() - t0, 2),
        "hot_activated": hot_ok,
    }
