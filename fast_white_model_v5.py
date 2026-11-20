#!/usr/bin/env python3
"""
极速复现《视频+5+(2).mp4》的 15 秒一镜到底简单白模视频（Simple White-Model Video）
- 画幅与帧率完全对齐原片：992×432, 24.04fps (362 帧 / 15.06s)
- 采用高速三维多边形投影与深度排序光栅化（PIL ImageDraw + NumPy）
- 精确复刻原片空间站位与运镜（已严格校验屏幕左右方向 +X_world = 画面右侧）：
  * 5 名白模人物：前景中心鲍里斯（球头+毛帽白模+右眼独眼罩+躯干+四肢+简化步枪含分段枪管/箍环/准星/枪口孔），
    左侧中远景 2 名跟进队员（0-15s 全程可见于画面左侧背景），右侧中景 2 名跟进队员
  * 右侧近景标志性粗松树干与背景针叶林树干、雪坡地面透视网格
  * 0–4s：鲍里斯低头持枪向前走近，右侧与左侧队员错拍跟进
  * 4–6s：鲍里斯停步抬头看向画面最右侧（+X），步枪横举指向画面右侧
  * 6–10s：鲍里斯头部与视线从画面最右侧缓慢扫向画面中心，步枪转向画面右前侧对准镜头右前方
  * 10–14s：摄影机平滑推进至步枪枪口大特写（枪口、箍环与准星占据画面右侧近景，鲍里斯毛帽与独眼位于枪后偏左上方，左后方两名队员位于画面左侧远景），14–15.06s 稳住
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
from typing import List, Tuple

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw


FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
WIDTH = 992
HEIGHT = 432
FPS = 24.04
TOTAL_FRAMES = 362  # 362 / 24.04 = 15.06s

REF_VIDEO_PATH = "/Users/wangmu/.gemini/antigravity/brain/f4799dab-ece2-40b4-955d-6a8f185d371f/.user_uploaded/media_1794294246543_c2ff5dfc.mp4"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = clamp((x - edge0) / max(1e-6, edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(edge0: float, edge1: float, x: float) -> float:
    t = clamp((x - edge0) / max(1e-6, edge1 - edge0), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _int_smoothstep(a: float, b: float, x: float) -> float:
    if x <= a:
        return 0.0
    L = max(1e-6, b - a)
    if x >= b:
        return 0.5 * L
    u = (x - a) / L
    return L * (u * u * u - 0.5 * u * u * u * u)


def walk_envelope(t: float, t_start: float, t_full: float, t_decel: float, t_stop: float) -> float:
    if t <= t_start or t >= t_stop:
        return 0.0
    up = smoothstep(t_start, t_full, t)
    down = 1.0 - smoothstep(t_decel, t_stop, t)
    return up * down


def integrate_walk(t: float, t_start: float, t_full: float, t_decel: float, t_stop: float) -> float:
    t_clamped = clamp(t, t_start, t_stop)
    if t_clamped <= t_start:
        return 0.0
    p1 = _int_smoothstep(t_start, t_full, min(t_clamped, t_decel))
    if t_clamped <= t_decel:
        return p1
    p2 = (t_clamped - t_decel) - (
        _int_smoothstep(t_decel, t_stop, t_clamped) - _int_smoothstep(t_decel, t_stop, t_decel)
    )
    return p1 + p2


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


# Unit cube mesh (8 vertices, 6 quad faces with normals)
CUBE_VERTS = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    dtype=np.float32,
)

CUBE_FACES = [
    ([4, 5, 6, 7], np.array([0.0, 0.0, 1.0], dtype=np.float32)),   # +Z front
    ([1, 0, 3, 2], np.array([0.0, 0.0, -1.0], dtype=np.float32)),  # -Z back
    ([0, 4, 7, 3], np.array([-1.0, 0.0, 0.0], dtype=np.float32)),  # -X left
    ([5, 1, 2, 6], np.array([1.0, 0.0, 0.0], dtype=np.float32)),   # +X right
    ([7, 6, 2, 3], np.array([0.0, 1.0, 0.0], dtype=np.float32)),   # +Y top
    ([0, 1, 5, 4], np.array([0.0, -1.0, 0.0], dtype=np.float32)),  # -Y bottom
]


def make_cylinder_z(
    radius: float, length: float, segs: int = 16, z_rings: int = 5, front_cap: bool = True
) -> Tuple[np.ndarray, List[Tuple[List[int], np.ndarray]]]:
    """Create a subdivided cylinder along +Z from z=0 to z=length for accurate depth sorting in close-ups."""
    verts = []
    for r_idx in range(z_rings + 1):
        z = length * (r_idx / z_rings)
        for i in range(segs):
            ang = 2.0 * math.pi * i / segs
            verts.append([radius * math.cos(ang), radius * math.sin(ang), z])
    verts_arr = np.array(verts, dtype=np.float32)
    faces = []
    for r_idx in range(z_rings):
        base0 = r_idx * segs
        base1 = (r_idx + 1) * segs
        for i in range(segs):
            ni = (i + 1) % segs
            ang = 2.0 * math.pi * (i + 0.5) / segs
            normal = np.array([math.cos(ang), math.sin(ang), 0.0], dtype=np.float32)
            faces.append(([base0 + i, base0 + ni, base1 + ni, base1 + i], normal))
    if front_cap:
        top_base = z_rings * segs
        faces.append((list(range(top_base, top_base + segs)), np.array([0.0, 0.0, 1.0], dtype=np.float32)))
    return verts_arr, faces


def make_sphere_lowpoly(
    radius: float, lat_segs: int = 7, lon_segs: int = 12
) -> Tuple[np.ndarray, List[Tuple[List[int], np.ndarray]]]:
    """Create a clean low-poly sphere for the white-model puppet head."""
    verts = []
    for i in range(lat_segs + 1):
        lat = math.pi * (-0.5 + i / lat_segs)
        y = radius * math.sin(lat)
        r = radius * math.cos(lat)
        for j in range(lon_segs):
            lon = 2.0 * math.pi * j / lon_segs
            verts.append([r * math.cos(lon), y, r * math.sin(lon)])
    verts_arr = np.array(verts, dtype=np.float32)
    faces = []
    for i in range(lat_segs):
        for j in range(lon_segs):
            nj = (j + 1) % lon_segs
            v0 = i * lon_segs + j
            v1 = i * lon_segs + nj
            v2 = (i + 1) * lon_segs + nj
            v3 = (i + 1) * lon_segs + j
            center = (verts_arr[v0] + verts_arr[v1] + verts_arr[v2] + verts_arr[v3]) * 0.25
            n = center / max(1e-6, float(np.linalg.norm(center)))
            faces.append(([v0, v1, v2, v3], n.astype(np.float32)))
    return verts_arr, faces


HEAD_VERTS, HEAD_FACES = make_sphere_lowpoly(1.0, lat_segs=7, lon_segs=12)
BARREL_OPEN_VERTS, BARREL_OPEN_FACES = make_cylinder_z(1.0, 1.0, segs=16, z_rings=5, front_cap=False)
BARREL_CAP_VERTS, BARREL_CAP_FACES = make_cylinder_z(1.0, 1.0, segs=16, z_rings=4, front_cap=True)


class FastWhiteModelScene:
    def __init__(self) -> None:
        # 5 characters matching 视频+5+(2).mp4:
        # Note: Camera is at +Z looking toward -Z, so +X is SCREEN RIGHT and -X is SCREEN LEFT!
        self.specs = [
            {
                "name": "Boris",
                "x0": -0.06,
                "z0": -0.35,
                "dist": 2.95,  # 0-5s walks 2.95m forward and stops at z = 2.60
                "t_start": 0.0,
                "t_full": 0.45,
                "t_decel": 3.80,
                "t_stop": 4.95,
                "freq": 1.65,
                "phase": 0.0,
                "scale": 1.0,
                "shade": (236, 240, 246),
            },
            {
                "name": "Right_1",
                "x0": 0.44,
                "z0": -1.85,
                "dist": 2.75,  # stops at z = 0.90
                "t_start": 0.05,
                "t_full": 0.65,
                "t_decel": 4.20,
                "t_stop": 5.35,
                "freq": 1.55,
                "phase": 0.9,
                "scale": 0.98,
                "shade": (210, 216, 225),
            },
            {
                "name": "Right_2",
                "x0": 0.86,
                "z0": -3.20,
                "dist": 2.60,  # stops at z = -0.60
                "t_start": 0.15,
                "t_full": 0.85,
                "t_decel": 4.40,
                "t_stop": 5.65,
                "freq": 1.50,
                "phase": 2.1,
                "scale": 0.96,
                "shade": (195, 202, 212),
            },
            {
                "name": "Left_1",
                "x0": -1.42,
                "z0": -2.75,
                "dist": 2.15,  # stops at z = -0.60 (visible in left background at t=14s!)
                "t_start": 0.10,
                "t_full": 0.75,
                "t_decel": 4.10,
                "t_stop": 5.25,
                "freq": 1.45,
                "phase": 1.6,
                "scale": 0.96,
                "shade": (202, 210, 220),
            },
            {
                "name": "Left_2",
                "x0": -0.82,
                "z0": -3.95,
                "dist": 2.00,  # stops at z = -1.95 (visible in left-center background at t=14s!)
                "t_start": 0.20,
                "t_full": 0.90,
                "t_decel": 4.35,
                "t_stop": 5.55,
                "freq": 1.42,
                "phase": 2.8,
                "scale": 0.95,
                "shade": (188, 196, 208),
            },
        ]
        for s in self.specs:
            s["norm"] = max(1e-6, integrate_walk(15.0, s["t_start"], s["t_full"], s["t_decel"], s["t_stop"]))

        # Pine trees matching 视频+5+(2).mp4:
        # Includes the left foreground pass-by tree (-1.22, 3.45) that sweeps off the left edge in 0-5s,
        # and the prominent right-foreground tree trunk at x=+1.18, z=2.15
        self.trees = [
            (1.18, 2.15, 0.22, 5.5),    # Landmark right foreground trunk
            (-1.22, 3.45, 0.17, 5.2),   # Left foreground pass-by tree (sweeps across left in 0-5s!)
            (-1.62, 0.85, 0.17, 5.2),   # Left mid-near tree
            (-2.15, -1.80, 0.18, 5.0),
            (-1.65, -4.40, 0.16, 5.0),
            (-0.92, -6.10, 0.15, 5.0),
            (-0.18, -6.90, 0.18, 5.0),
            (0.45, -5.80, 0.14, 5.0),
            (1.55, -3.60, 0.17, 5.0),
            (1.85, -0.60, 0.18, 5.0),
            (2.10, -5.80, 0.19, 5.0),
        ]

    def evaluate_render_ables(self, t: float):
        """
        Return camera parameters and a list of 3D primitives (boxes, spheres, cylinders)
        at time t (0..15.06s).
        """
        # 0-5s: Camera actively tracks/dollies from (-0.42, 1.56, 7.10) to (0.02, 1.47, 4.95)
        # so the camera and spatial environment (trees, snowbanks, ground grid) move continuously in the first 5s!
        u_intro = smootherstep(0.0, 5.0, t)
        u_mid = smoothstep(5.0, 9.6, t)
        u_push = smootherstep(9.6, 13.8, t)

        base_cam_x = lerp(-0.42, 0.02, u_intro)
        base_cam_y = lerp(1.56, 1.47, u_intro) - 0.01 * u_mid
        base_cam_z = lerp(7.10, 4.95, u_intro) - 0.30 * u_mid

        base_look_x = lerp(-0.14, 0.02, u_intro)
        base_look_y = lerp(1.24, 1.38, u_intro) + 0.02 * u_mid
        base_look_z = lerp(-0.40, 1.70, u_intro) + 0.30 * u_mid

        cam_pos = np.array(
            [
                lerp(base_cam_x, 0.05, u_push),
                lerp(base_cam_y, 1.48, u_push),
                lerp(base_cam_z, 3.74, u_push),
            ],
            dtype=np.float32,
        )
        cam_look = np.array(
            [
                lerp(base_look_x, 0.01, u_push),
                lerp(base_look_y, 1.47, u_push),
                lerp(base_look_z, 2.60, u_push),
            ],
            dtype=np.float32,
        )
        fov_deg = 32.0

        boxes = []      # (center, half_size, rot_mat, base_rgb, outline_rgb)
        spheres = []    # (center, radius_vec3, rot_mat, base_rgb, outline_rgb)
        cylinders = []  # (p0, radius, length, rot_mat, base_rgb, outline_rgb, is_hollow_muzzle)

        # 1. Tree trunks
        for tx, tz, tr, th in self.trees:
            boxes.append(
                (
                    np.array([tx, th * 0.5 - 0.2, tz], dtype=np.float32),
                    np.array([tr, th * 0.5, tr], dtype=np.float32),
                    np.eye(3, dtype=np.float32),
                    (150, 158, 170),
                    (95, 102, 114),
                )
            )

        # 2. Five white-model characters
        for idx, s in enumerate(self.specs):
            w_env = walk_envelope(t, s["t_start"], s["t_full"], s["t_decel"], s["t_stop"])
            prog = integrate_walk(t, s["t_start"], s["t_full"], s["t_decel"], s["t_stop"]) / s["norm"]
            z_pos = s["z0"] + s["dist"] * prog
            x_pos = s["x0"] + 0.02 * math.sin(prog * math.pi) * (1.0 if idx % 2 == 0 else -1.0)

            phase = 2.0 * math.pi * s["freq"] * t + s["phase"]
            stride = math.sin(phase) * w_env
            bob = abs(math.cos(phase)) * 0.032 * w_env
            sway = math.sin(phase) * 0.022 * w_env

            sc = s["scale"]
            pelvis = np.array([x_pos + sway, 0.90 * sc + bob, z_pos], dtype=np.float32)

            if idx == 0:
                # Boris:
                # Remember: +X is Screen Right! And rot_y(+yaw) @ [0, 0, 1] = [+sin(yaw), 0, cos(yaw)]!
                # So POSITIVE yaw rotates Boris's head/rifle toward Screen Right (+X)!
                u_lift = smootherstep(3.6, 5.6, t)
                u_scan = smootherstep(6.0, 10.0, t)
                u_aim_hold = smootherstep(10.0, 13.5, t)

                # Head pitch:rot_x(+pitch) @ [0,0,1] = [0, -sin(pitch), cos(pitch)], so +pitch is looking DOWN!
                head_pitch = lerp(0.36, 0.02, u_lift) + 0.04 * u_aim_hold
                # 0-3.6s: slight yaw (+0.05); 3.6-5.6s: turns head to look Screen Right (+0.48 rad);
                # 6.0-10.0s: sweeps head from Screen Right (+0.48) toward Screen Center / rifle barrel (+0.10 rad)
                head_yaw = (1.0 - u_scan) * lerp(0.05, 0.48, u_lift) + u_scan * 0.10
                torso_pitch = lerp(0.10, 0.02, u_lift)
                torso_yaw = 0.22 * head_yaw
            else:
                u_SubLift = smootherstep(4.0, 5.8, t)
                u_SubScan = smootherstep(6.2, 10.2, t)
                head_pitch = lerp(0.20, 0.02, u_SubLift)
                head_yaw = lerp(0.0, 0.20, u_SubLift) * (1.0 - 0.65 * u_SubScan)
                torso_pitch = lerp(0.06, 0.0, u_SubLift)
                torso_yaw = 0.25 * head_yaw

            r_torso = rot_y(torso_yaw) @ rot_x(torso_pitch)
            torso_c = pelvis + r_torso @ np.array([0.0, 0.27 * sc, 0.0], dtype=np.float32)
            base_c = s["shade"]
            edge_c = (78, 86, 98)

            # Torso + Coat skirt block
            boxes.append(
                (
                    torso_c,
                    np.array([0.195 * sc, 0.27 * sc, 0.125 * sc], dtype=np.float32),
                    r_torso,
                    base_c,
                    edge_c,
                )
            )
            hem_c = pelvis + r_torso @ np.array([0.0, -0.04 * sc, 0.0], dtype=np.float32)
            boxes.append(
                (
                    hem_c,
                    np.array([0.205 * sc, 0.11 * sc, 0.135 * sc], dtype=np.float32),
                    r_torso,
                    tuple(max(0, c - 10) for c in base_c),
                    edge_c,
                )
            )
            collar_c = pelvis + r_torso @ np.array([0.0, 0.51 * sc, -0.01 * sc], dtype=np.float32)
            boxes.append(
                (
                    collar_c,
                    np.array([0.21 * sc, 0.048 * sc, 0.135 * sc], dtype=np.float32),
                    r_torso,
                    tuple(max(0, c - 6) for c in base_c),
                    edge_c,
                )
            )

            # Head (low-poly sphere) + Ushanka fur hat crown & earflaps
            r_head = rot_y(head_yaw) @ rot_x(head_pitch)
            head_c = pelvis + r_torso @ np.array([0.0, 0.69 * sc, 0.02 * sc], dtype=np.float32)
            spheres.append(
                (
                    head_c,
                    np.array([0.125 * sc, 0.135 * sc, 0.13 * sc], dtype=np.float32),
                    r_head,
                    (242, 245, 250) if idx == 0 else base_c,
                    edge_c,
                )
            )
            # Fur hat (Ushanka) crown cap sitting cleanly on top of the head sphere (no intersection!)
            hat_c = head_c + r_head @ np.array([0.0, 0.115 * sc, 0.0], dtype=np.float32)
            spheres.append(
                (
                    hat_c,
                    np.array([0.142 * sc, 0.075 * sc, 0.142 * sc], dtype=np.float32),
                    r_head,
                    (182, 190, 202) if idx == 0 else tuple(max(0, c - 22) for c in base_c),
                    edge_c,
                )
            )
            # Nose bridge block on front of face (+Z_head)
            nose_c = head_c + r_head @ np.array([0.0, -0.015 * sc, 0.132 * sc], dtype=np.float32)
            boxes.append(
                (
                    nose_c,
                    np.array([0.018 * sc, 0.028 * sc, 0.020 * sc], dtype=np.float32),
                    r_head,
                    (225, 230, 238),
                    edge_c,
                )
            )
            if idx == 0:
                # Boris's signature eyepatch on his RIGHT eye:
                # Since Boris faces +Z (toward camera), his right eye is on the SCREEN LEFT (-X_head)!
                patch_c = head_c + r_head @ np.array([-0.046 * sc, 0.010 * sc, 0.120 * sc], dtype=np.float32)
                boxes.append(
                    (
                        patch_c,
                        np.array([0.034 * sc, 0.024 * sc, 0.014 * sc], dtype=np.float32),
                        r_head,
                        (45, 50, 58),
                        (32, 36, 44),
                    )
                )

            # Legs
            for side, sgn in (("L", 1.0), ("R", -1.0)):
                hip = pelvis + np.array([sgn * 0.105 * sc, 0.0, 0.0], dtype=np.float32)
                s_leg = stride if side == "L" else -stride
                hip_ang = 0.48 * s_leg
                knee_ang = -0.42 * max(0.0, -s_leg) - 0.06 * w_env

                r_up = rot_x(hip_ang)
                up_len = 0.42 * sc
                up_c = hip + r_up @ np.array([0.0, -0.5 * up_len, 0.0], dtype=np.float32)
                knee = hip + r_up @ np.array([0.0, -up_len, 0.0], dtype=np.float32)
                boxes.append(
                    (
                        up_c,
                        np.array([0.072 * sc, 0.5 * up_len, 0.078 * sc], dtype=np.float32),
                        r_up,
                        tuple(max(0, c - 16) for c in base_c),
                        edge_c,
                    )
                )

                r_lo = rot_x(hip_ang + knee_ang)
                lo_len = 0.43 * sc
                lo_c = knee + r_lo @ np.array([0.0, -0.5 * lo_len, 0.0], dtype=np.float32)
                boxes.append(
                    (
                        lo_c,
                        np.array([0.062 * sc, 0.5 * lo_len, 0.068 * sc], dtype=np.float32),
                        r_lo,
                        tuple(max(0, c - 26) for c in base_c),
                        edge_c,
                    )
                )

            if idx == 0:
                # Boris's rifle & arms:
                # Positive yaw points the rifle toward Screen Right (+X_world)!
                u_raise = smootherstep(2.2, 5.0, t)
                u_aim_sweep = smootherstep(5.8, 10.2, t)

                rifle_pivot = pelvis + r_torso @ np.array(
                    [0.035 * sc, lerp(0.42, 0.52, u_raise) * sc, 0.14 * sc], dtype=np.float32
                )
                # At 0-5s: angled toward Screen Right (+0.38 rad);
                # At 6-10s: rotates slightly toward camera's right-foreground (+0.135 rad) so at t=10-15s
                # the muzzle sits squarely in the right-foreground of the frame!
                rifle_yaw = lerp(0.38, 0.135, u_aim_sweep)
                # Note: rot_x(+pitch) tilts +Z downward (-Y). At 0-2.5s rifle is pitched slightly down (+0.28),
                # then raises to near-horizontal (-0.025, slightly angled up toward camera) at 4-15s!
                rifle_pitch = lerp(0.28, -0.025, u_raise) - 0.02 * stride
                r_gun = rot_y(rifle_yaw) @ rot_x(rifle_pitch)

                # Wooden stock & receiver block (subdivided into 2 blocks along Z for clean sorting)
                for z_off, z_half in ((0.10, 0.11), (0.28, 0.10)):
                    stock_c = rifle_pivot + r_gun @ np.array([0.0, -0.016, z_off], dtype=np.float32)
                    boxes.append(
                        (
                            stock_c,
                            np.array([0.028, 0.048, z_half], dtype=np.float32),
                            r_gun,
                            (182, 188, 198),
                            (70, 78, 90),
                        )
                    )
                # Forend wooden handguard block
                guard_c = rifle_pivot + r_gun @ np.array([0.0, -0.008, 0.47], dtype=np.float32)
                boxes.append(
                    (
                        guard_c,
                        np.array([0.030, 0.042, 0.13], dtype=np.float32),
                        r_gun,
                        (192, 198, 208),
                        (70, 78, 90),
                    )
                )
                # Main cylindrical barrel
                barrel_p0 = rifle_pivot + r_gun @ np.array([0.0, 0.010, 0.32], dtype=np.float32)
                cylinders.append(
                    (
                        barrel_p0,
                        0.029,
                        0.38,
                        r_gun,
                        (208, 214, 224),
                        (65, 72, 84),
                        False,
                    )
                )
                # Signature heavy muzzle collar / band (prominent in v5_ref_12_14s.png!)
                collar_p0 = rifle_pivot + r_gun @ np.array([0.0, 0.010, 0.62], dtype=np.float32)
                cylinders.append(
                    (
                        collar_p0,
                        0.038,
                        0.070,
                        r_gun,
                        (222, 228, 236),
                        (58, 66, 78),
                        False,
                    )
                )
                # Muzzle tip cylinder with dark bore hole
                muzzle_p0 = rifle_pivot + r_gun @ np.array([0.0, 0.010, 0.69], dtype=np.float32)
                cylinders.append(
                    (
                        muzzle_p0,
                        0.034,
                        0.075,
                        r_gun,
                        (234, 240, 248),
                        (52, 60, 72),
                        True,  # render dark inner bore at front cap!
                    )
                )
                # Front sight post on top of muzzle collar (prominent silhouette in v5_ref_12_14s.png!)
                sight_c = rifle_pivot + r_gun @ np.array([0.0, 0.056, 0.655], dtype=np.float32)
                boxes.append(
                    (
                        sight_c,
                        np.array([0.008, 0.022, 0.018], dtype=np.float32),
                        r_gun,
                        (230, 236, 244),
                        (52, 60, 72),
                    )
                )

                # Boris's two arms gripping the rifle (gloved hands under handguard & grip)
                sh_r = pelvis + r_torso @ np.array([-0.20 * sc, 0.45 * sc, 0.0], dtype=np.float32)
                sh_l = pelvis + r_torso @ np.array([0.20 * sc, 0.45 * sc, 0.0], dtype=np.float32)
                grip_r = rifle_pivot + r_gun @ np.array([0.0, -0.05, 0.16], dtype=np.float32)
                grip_l = rifle_pivot + r_gun @ np.array([-0.01, -0.05, 0.42], dtype=np.float32)

                for sh, grip in ((sh_r, grip_r), (sh_l, grip_l)):
                    mid = 0.5 * (sh + grip) + np.array([0.0, -0.06, 0.0], dtype=np.float32)
                    for a_pt, b_pt in ((sh, mid), (mid, grip)):
                        seg_v = b_pt - a_pt
                        seg_l = max(1e-4, float(np.linalg.norm(seg_v)))
                        seg_c = 0.5 * (a_pt + b_pt)
                        u_dir = seg_v / seg_l
                        yaw_a = math.atan2(float(u_dir[0]), float(u_dir[2]))
                        pitch_a = -math.asin(clamp(float(u_dir[1]), -1.0, 1.0))
                        r_seg = rot_y(yaw_a) @ rot_x(pitch_a)
                        boxes.append(
                            (
                                seg_c,
                                np.array([0.052 * sc, 0.052 * sc, 0.5 * seg_l + 0.015], dtype=np.float32),
                                r_seg,
                                (202, 208, 218),
                                edge_c,
                            )
                        )
                    # Gloved hand block gripping underneath the rifle
                    boxes.append(
                        (
                            grip,
                            np.array([0.044 * sc, 0.042 * sc, 0.050 * sc], dtype=np.float32),
                            r_gun,
                            (172, 180, 190),
                            edge_c,
                        )
                    )
            else:
                # Followers' swinging arms
                for side, sgn in (("L", 1.0), ("R", -1.0)):
                    sh = pelvis + r_torso @ np.array([sgn * 0.22 * sc, 0.46 * sc, 0.0], dtype=np.float32)
                    a_sw = -stride if side == "L" else stride
                    r_arm = r_torso @ rot_x(0.28 * a_sw + 0.08)
                    arm_len = 0.54 * sc
                    arm_c = sh + r_arm @ np.array([0.0, -0.5 * arm_len, 0.0], dtype=np.float32)
                    boxes.append(
                        (
                            arm_c,
                            np.array([0.055 * sc, 0.5 * arm_len, 0.060 * sc], dtype=np.float32),
                            r_arm,
                            tuple(max(0, c - 12) for c in base_c),
                            edge_c,
                        )
                    )

        return cam_pos, cam_look, fov_deg, boxes, spheres, cylinders


def render_white_model_frame(scene: FastWhiteModelScene, t: float) -> np.ndarray:
    """Render one 992x432 frame using fast 3D polygon projection and painter's sorting."""
    cam_pos, cam_look, fov_deg, boxes, spheres, cylinders = scene.evaluate_render_ables(t)

    # Camera basis: right (+X_screen), up (+Y_screen), forward (+Z_cam into scene)
    fwd = cam_look - cam_pos
    fwd = fwd / max(1e-6, float(np.linalg.norm(fwd)))
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(fwd, world_up)
    right = right / max(1e-6, float(np.linalg.norm(right)))
    up = np.cross(right, fwd)
    up = up / max(1e-6, float(np.linalg.norm(up)))

    R_cam = np.stack([right, up, fwd], axis=0)  # 3x3: world -> cam
    fy = 0.5 * HEIGHT / math.tan(math.radians(fov_deg) * 0.5)
    fx = fy
    cx = 0.5 * WIDTH
    cy = 0.5 * HEIGHT

    img = Image.new("RGB", (WIDTH, HEIGHT), (188, 195, 205))
    draw = ImageDraw.Draw(img)

    # Sky / snow forest horizon band
    horizon_v = int(cy - fy * (0.0 - cam_pos[1]) / 12.0)
    horizon_v = clamp(horizon_v, 80, HEIGHT - 80)
    draw.rectangle([0, 0, WIDTH, int(horizon_v)], fill=(172, 180, 192))
    draw.rectangle([0, int(horizon_v), WIDTH, HEIGHT], fill=(216, 222, 230))

    # 3D perspective snow ground grid lines
    grid_color = (196, 203, 213)
    for gx in np.linspace(-3.5, 3.5, 15):
        p0_w = np.array([gx, 0.0, -7.0], dtype=np.float32)
        p1_w = np.array([gx, 0.0, 3.6], dtype=np.float32)
        p0_c = R_cam @ (p0_w - cam_pos)
        p1_c = R_cam @ (p1_w - cam_pos)
        if p0_c[2] > 0.20 and p1_c[2] > 0.20:
            u0 = cx + fx * (p0_c[0] / p0_c[2])
            v0 = cy - fy * (p0_c[1] / p0_c[2])
            u1 = cx + fx * (p1_c[0] / p1_c[2])
            v1 = cy - fy * (p1_c[1] / p1_c[2])
            draw.line([(u0, v0), (u1, v1)], fill=grid_color, width=1)

    for gz in np.linspace(-6.5, 3.5, 18):
        p0_w = np.array([-3.5, 0.0, gz], dtype=np.float32)
        p1_w = np.array([3.5, 0.0, gz], dtype=np.float32)
        p0_c = R_cam @ (p0_w - cam_pos)
        p1_c = R_cam @ (p1_w - cam_pos)
        if p0_c[2] > 0.20 and p1_c[2] > 0.20:
            u0 = cx + fx * (p0_c[0] / p0_c[2])
            v0 = cy - fy * (p0_c[1] / p0_c[2])
            u1 = cx + fx * (p1_c[0] / p1_c[2])
            v1 = cy - fy * (p1_c[1] / p1_c[2])
            draw.line([(u0, v0), (u1, v1)], fill=grid_color, width=1)

    # Studio key light for clean white-model shading
    light_dir = np.array([0.38, 0.78, 0.50], dtype=np.float32)
    light_dir = light_dir / float(np.linalg.norm(light_dir))

    polys = []  # (sort_dist, screen_pts, fill_rgb, outline_rgb)
    near_z = 0.16

    def add_mesh(verts_local: np.ndarray, faces, center: np.ndarray, r_mat: np.ndarray, base_rgb, outline_rgb):
        verts_w = (verts_local @ r_mat.T) + center
        verts_c = (verts_w - cam_pos) @ R_cam.T
        for idxs, n_local in faces:
            f_c = verts_c[idxs]
            if np.any(f_c[:, 2] <= near_z):
                continue
            centroid_c = np.mean(f_c, axis=0)
            n_w = r_mat @ n_local
            n_c = R_cam @ n_w
            if float(np.dot(n_c, centroid_c)) >= 0.0:
                continue
            ndl = max(0.0, float(np.dot(n_w, light_dir)))
            shade = 0.68 + 0.34 * ndl
            sort_dist = float(np.linalg.norm(centroid_c))
            fog = clamp(( float(centroid_c[2]) - 1.2) / 11.0, 0.0, 0.32)
            r = int(clamp(base_rgb[0] * shade * (1.0 - fog) + 185 * fog, 0, 255))
            g = int(clamp(base_rgb[1] * shade * (1.0 - fog) + 192 * fog, 0, 255))
            b = int(clamp(base_rgb[2] * shade * (1.0 - fog) + 204 * fog, 0, 255))

            us = cx + fx * (f_c[:, 0] / f_c[:, 2])
            vs = cy - fy * (f_c[:, 1] / f_c[:, 2])
            pts = list(zip(us.tolist(), vs.tolist()))
            polys.append((sort_dist, pts, (r, g, b), outline_rgb))

    # 1. Boxes
    for center, half_size, r_mat, base_rgb, outline_rgb in boxes:
        v_loc = CUBE_VERTS * (2.0 * half_size)
        add_mesh(v_loc, CUBE_FACES, center, r_mat, base_rgb, outline_rgb)

    # 2. Spheres (heads & fur hats)
    for center, rad_vec, r_mat, base_rgb, outline_rgb in spheres:
        v_loc = HEAD_VERTS * rad_vec
        add_mesh(v_loc, HEAD_FACES, center, r_mat, base_rgb, None)

    # 3. Cylinders (rifle barrel, collar, muzzle)
    for p0, radius, length, r_mat, base_rgb, outline_rgb, is_hollow in cylinders:
        v_loc = (BARREL_CAP_VERTS if is_hollow else BARREL_OPEN_VERTS).copy()
        c_faces = BARREL_CAP_FACES if is_hollow else BARREL_OPEN_FACES
        v_loc[:, 0] *= radius
        v_loc[:, 1] *= radius
        v_loc[:, 2] *= length
        add_mesh(v_loc, c_faces, p0, r_mat, base_rgb, outline_rgb)
        if is_hollow:
            segs = 16
            bore_r = radius * 0.60
            bore_verts = []
            for i in range(segs):
                ang = 2.0 * math.pi * i / segs
                bore_verts.append([bore_r * math.cos(ang), bore_r * math.sin(ang), length + 0.002])
            bore_v = np.array(bore_verts, dtype=np.float32)
            bore_w = (bore_v @ r_mat.T) + p0
            bore_c = (bore_w - cam_pos) @ R_cam.T
            if np.all(bore_c[:, 2] > near_z):
                us = cx + fx * (bore_c[:, 0] / bore_c[:, 2])
                vs = cy - fy * (bore_c[:, 1] / bore_c[:, 2])
                pts = list(zip(us.tolist(), vs.tolist()))
                centroid_c = np.mean(bore_c, axis=0)
                polys.append((float(np.linalg.norm(centroid_c)) - 0.015, pts, (24, 28, 34), (52, 60, 72)))

    # Sort back-to-front by 3D Euclidean distance to camera
    polys.sort(key=lambda item: item[0], reverse=True)

    for _, pts, fill_rgb, outline_rgb in polys:
        if outline_rgb is not None:
            draw.polygon(pts, fill=fill_rgb, outline=outline_rgb)
        else:
            draw.polygon(pts, fill=fill_rgb)

    return np.asarray(img, dtype=np.uint8)


def generate_fast_white_model_video(out_mp4: str) -> Tuple[float, str]:
    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)

    scene = FastWhiteModelScene()
    has_ref_audio = os.path.exists(REF_VIDEO_PATH)

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-pix_fmt",
        "rgb24",
        "-r",
        f"{FPS:.2f}",
        "-i",
        "-",
    ]
    if has_ref_audio:
        cmd += ["-i", REF_VIDEO_PATH, "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-shortest"]
    else:
        cmd += ["-an"]

    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        out_mp4,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    for i in range(TOTAL_FRAMES):
        t = i / FPS
        frame = render_white_model_frame(scene, t)
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    stderr_bytes = proc.stderr.read() if proc.stderr else b""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"FFmpeg failed ({rc}): {stderr_bytes.decode('utf-8', errors='ignore')}")

    elapsed = time.perf_counter() - t0
    return elapsed, out_mp4


if __name__ == "__main__":
    out_dir = "/Users/wangmu/.gemini/antigravity/scratch/director-stage/exports"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "video5_white_model_992x432_24fps.mp4")

    elapsed, final_path = generate_fast_white_model_video(out_path)
    print(f"DONE in {elapsed:.2f}s -> {final_path}")

    desktop_dir = "/Users/wangmu/Desktop/MagicDirectorStage_导演台完整软件/01_最新成品_雪林车队极简方块预演"
    if os.path.isdir(desktop_dir):
        shutil.copy2(final_path, os.path.join(desktop_dir, "视频5_简单白模极速复现_992x432_24fps.mp4"))
    artifact_copy = "/Users/wangmu/.gemini/antigravity/brain/f4799dab-ece2-40b4-955d-6a8f185d371f/video5_white_model_992x432_24fps.mp4"
    shutil.copy2(final_path, artifact_copy)
