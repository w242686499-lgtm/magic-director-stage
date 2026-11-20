#!/usr/bin/env python3
"""
Magic Director Stage - Python Backend Server (server.py)
提供 Web 导演台静态服务、Python 脚本实时执行 API、单图姿态解析 API 与极速白模 MP4 导出 API。
"""

import base64
import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
import threading
from typing import Optional, Tuple

from keyframe_policy import sparsify_scene
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from director_engine import (
    PRESET_IK_POSES,
    Stage,
    analyze_image_to_ik_targets,
    build_clean_dual_character_scene,
    build_convoy_and_cabin_previz_shot,
    build_default_sitting_scene,
    build_fireplace_cabin_su_scene,
    build_five_men_bonfire_15s_scene,
    build_forest_reveal_scene,
    build_scene08_snow_blood_tracking_scene,
    build_scene64_cabin_scene,
    build_video3_cabin_curtain_7s_scene,
    build_video5_five_men_15s_scene,
    build_video9_tree_cover_12s_scene,
    build_wechat_two_hunters_aim_left_6s_scene,
    compute_track_motion_stats,
    check_export_confirmation_gate,
    extract_video_thumbnails,
    prepare_export_confirmation_summary,
    probe_video_metadata,
    prompt_to_stage,
    render_5frame_parity_sheet,
    render_comparison_image,
    render_multi_time_comparisons,
    render_scene_to_mp4,
    resolve_scene_export_dimensions,
    scene_dict_to_python_code,
    transform_keyframes_timing,
)
from video_repro_service import (
    cancel_repro_job,
    detect_repro_environment,
    get_repro_job_status,
    prepare_video_repro_confirmation_proposal,
    solve_white_model_video_reproduction,
    start_repro_analysis_job,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
PLUGINS_DIR = os.path.join(STATIC_DIR, "plugins")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")
COMPARISON_DIR = os.path.join(EXPORT_DIR, "comparisons")
REF_VIDEO_DIR = os.path.join(BASE_DIR, "ref_videos")
PROJECTS_FILE = os.environ.get("DIRECTOR_PROJECTS_FILE") or os.path.join(BASE_DIR, "saved_projects.json")
PROJECT_IO_LOCK = threading.RLock()
_PROJECT_BANK_CACHE = {"mtime": -1.0, "bank": None, "json_bytes": None}
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(PLUGINS_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(COMPARISON_DIR, exist_ok=True)
os.makedirs(REF_VIDEO_DIR, exist_ok=True)

DEFAULT_REF_VIDEO_PATH = os.path.join(REF_VIDEO_DIR, "video3_ref.mp4")

_INIT_SCENE = build_five_men_bonfire_15s_scene()
LIVE_STATE = {
    "version": 53,
    "scene": _INIT_SCENE,
    "python_code": scene_dict_to_python_code(_INIT_SCENE),
    "last_prompt": "五人围火 · 球头人偶动作预演（15秒 · 2.35:1 · 全员运动）",
}


import math


def _stamp_camera_hud_on_pil(pil_img, scene: dict):
    """在后端兜底导出的截图左上角叠加半透明深灰角标：焦距 (mm) + 机位高度 (m)。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        cam = (scene or {}).get("camera") or {}
        fov_deg = float(cam.get("fov") or 37.8)
        focal_mm = int(round(float(cam.get("focal_mm") or (24.0 / (2.0 * math.tan(math.radians(max(4.0, min(130.0, fov_deg))) * 0.5))))))
        cam_pos = cam.get("pos") or [0.0, 1.5, 0.0]
        cam_h = float(cam_pos[1]) if len(cam_pos) > 1 else 1.5
        label = f"焦距 {focal_mm}mm  |  机位高度 {cam_h:.2f}m"

        base = pil_img.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        h = base.size[1]
        font_size = max(12, int(round(h * 0.025)))
        font = None
        for fp in (
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
        ):
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, font_size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), label, font=font)
        tw = max(40, bbox[2] - bbox[0])
        pad_x = max(10, int(round(h * 0.022)))
        pad_y = max(10, int(round(h * 0.022)))
        inner_h = max(9, int(round(font_size * 0.72)))
        box_h = max(24, int(round(font_size * 1.85)))
        box_w = tw + inner_h * 2
        radius = max(5, int(round(box_h * 0.24)))

        draw.rounded_rectangle(
            [pad_x, pad_y, pad_x + box_w, pad_y + box_h],
            radius=radius,
            fill=(15, 20, 28, 194),
            outline=(255, 255, 255, 56),
            width=1,
        )
        text_y = pad_y + (box_h - (bbox[3] - bbox[1])) * 0.5 - bbox[1]
        draw.text((pad_x + inner_h, text_y), label, font=font, fill=(243, 246, 250, 255))
        return Image.alpha_composite(base, overlay).convert("RGB")
    except Exception:
        return pil_img


def build_default_project_bank() -> dict:
    """构建默认的多场次 + 同场多镜头工程库（1场包含多个共享相同3D场景的镜头）。"""
    bonfire_shot1 = copy.deepcopy(build_five_men_bonfire_15s_scene())
    s08_shot1 = copy.deepcopy(build_scene08_snow_blood_tracking_scene("shot1"))
    s08_shot2 = copy.deepcopy(build_scene08_snow_blood_tracking_scene("shot2"))
    fc_shot1 = copy.deepcopy(build_fireplace_cabin_su_scene("shot1"))
    fc_shot2 = copy.deepcopy(build_fireplace_cabin_su_scene("shot2"))
    video3_shot1 = copy.deepcopy(build_video3_cabin_curtain_7s_scene())
    video9_shot1 = copy.deepcopy(build_video9_tree_cover_12s_scene())
    video5_shot1 = copy.deepcopy(build_video5_five_men_15s_scene())
    convoy_shot1 = copy.deepcopy(build_convoy_and_cabin_previz_shot("shot1"))
    convoy_shot2 = copy.deepcopy(build_convoy_and_cabin_previz_shot("shot2"))
    s64_shot1 = copy.deepcopy(build_scene64_cabin_scene("shot1"))
    s64_shot2 = copy.deepcopy(build_scene64_cabin_scene("shot2"))
    clean_shot1 = copy.deepcopy(build_clean_dual_character_scene())
    forest_base = build_forest_reveal_scene()
    s1_shot1 = copy.deepcopy(forest_base)

    # 红林场 · 第 2 镜：复用完全相同的红林与双人 3D 场景，但机位改为 50mm 侧面越肩跟拍
    s1_shot2 = copy.deepcopy(forest_base)
    s1_shot2["duration"] = 6.0
    s1_shot2["camera"] = {
        "pos": [-2.35, 1.42, 0.25],
        "target": [0.15, 1.15, -0.55],
        "fov": 27.0,
        "focal_mm": 50,
        "aspect_ratio": "2.39:1",
    }
    s1_shot2["camera_keyframes"] = [
        {"t": 0.0, "pos": [-2.35, 1.42, 0.95], "target": [-0.30, 1.18, 0.50], "fov": 27.0, "focal_mm": 50},
        {"t": 3.0, "pos": [-2.35, 1.42, 0.25], "target": [0.15, 1.15, -0.55], "fov": 27.0, "focal_mm": 50},
        {"t": 6.0, "pos": [-2.10, 1.48, -0.45], "target": [0.35, 1.12, -1.25], "fov": 27.0, "focal_mm": 50},
    ]

    # 红林场 · 第 3 镜：复用完全相同的红林与双人 3D 场景，机位改为 24mm 广角高空俯拍
    s1_shot3 = copy.deepcopy(forest_base)
    s1_shot3["duration"] = 6.0
    s1_shot3["camera"] = {
        "pos": [0.0, 5.80, 2.20],
        "target": [0.0, 0.60, -0.80],
        "fov": 53.1,
        "focal_mm": 24,
        "aspect_ratio": "2.39:1",
    }
    s1_shot3["camera_keyframes"] = [
        {"t": 0.0, "pos": [0.0, 6.20, 2.80], "target": [0.0, 0.60, -0.40], "fov": 53.1, "focal_mm": 24},
        {"t": 6.0, "pos": [0.0, 4.80, 1.20], "target": [0.0, 0.60, -1.20], "fov": 53.1, "focal_mm": 24},
    ]

    # 第 3 场 · 第 1 镜：桌前托腮沉思（独立场次）
    s2_stage = prompt_to_stage("坐在桌前右手托腮低头沉思，缓慢推镜头")["scene"]

    return {
        "activeSceneId": "scene_five_men_bonfire_15s",
        "activeShotId": "shot_bonfire_1",
        "scenes": [
            {
                "id": "scene_five_men_bonfire_15s",
                "name": "五人围火 · 球头人偶动作预演 (15s · 1280×544 · 一镜到底全员运动)",
                "shots": [
                    {
                        "id": "shot_bonfire_1",
                        "name": "第 1 镜：火焰特写后拉→A添柴起身左出画→B右退揭露D与E→五人四站一蹲落幅 (15.0s)",
                        "updatedAt": "球头人偶预演",
                        "scene": bonfire_shot1,
                    },
                ],
            },
            {
                "id": "scene_08_snow_blood_tracking",
                "name": "第 8 场：西伯利亚雪林追踪血迹 (2镜×20s · 1680×720 · 8人彩色白模)",
                "shots": [
                    {
                        "id": "shot_s08_1",
                        "name": "第 1 镜：高空针叶林海与冰封河谷俯冲下潜至雪地血迹特写 (20.0s)",
                        "updatedAt": "白模预演导入",
                        "scene": s08_shot1,
                    },
                    {
                        "id": "shot_s08_2",
                        "name": "第 2 镜：尤里捻血雪低机位仰摇→S弧右移至伊万与六人小队出画 (20.0s)",
                        "updatedAt": "白模预演导入",
                        "scene": s08_shot2,
                    },
                ],
            },
            {
                "id": "scene_fireplace_cabin_su",
                "name": "猎人木屋·壁炉与长餐桌大堂 (SU灰白体块+黑边线 · 2镜共享3D空间)",
                "shots": [
                    {
                        "id": "shot_fc_1",
                        "name": "第 1 镜：西南高机位斜向俯拍全景（对应参考图1：左壁炉+后三窗+右大货架+中央长餐桌）",
                        "updatedAt": "SU风格场景",
                        "scene": fc_shot1,
                    },
                    {
                        "id": "shot_fc_2",
                        "name": "第 2 镜：东侧平视纵深正打壁炉（对应参考图2：正前壁炉+右三窗与工作桌+左挂衣架与木门）",
                        "updatedAt": "SU风格场景",
                        "scene": fc_shot2,
                    },
                ],
            },
            {
                "id": "scene_video3_cabin_curtain_7s",
                "name": "视频3复现：7秒木屋背身撩帘看架→转身俯身钻入帘后暗门 (992×432 · 168帧)",
                "shots": [
                    {
                        "id": "shot_video3_1",
                        "name": "第 1 镜：0–0.6s转头看右架 → 0.6–4.5s凝视观察 → 4.5–7s俯身钻入帘后 (7.0s)",
                        "updatedAt": "SU极简彩色预演",
                        "scene": video3_shot1,
                    },
                ],
            },
            {
                "id": "scene_video9_tree_cover_12s",
                "name": "视频9复现：12秒树旁抵肩瞄准→击发收枪下蹲→抬头观察推近 (992×432 · 288帧)",
                "shots": [
                    {
                        "id": "shot_video9_1",
                        "name": "第 1 镜：0–5s抵肩瞄准画右 → 5–6.2s击发收枪下蹲 → 6.2–12s抬头观察+DollyIn (12.0s)",
                        "updatedAt": "SU极简彩色预演",
                        "scene": video9_shot1,
                    },
                ],
            },
            {
                "id": "scene_video5_five_men_15s",
                "name": "SU极简彩色预演：15秒五人一镜到底 (1280×720 · 360帧)",
                "shots": [
                    {
                        "id": "shot_video5_1",
                        "name": "第 1 镜：0–4s走近 → 4–6s停步抬头 → 6–10s巡视 → 10–14s推进移焦 → 14–15s停稳 (15.0s)",
                        "updatedAt": "SU极简彩色预演",
                        "scene": video5_shot1,
                    },
                ],
            },
            {
                "id": "scene_convoy_cabin_previz",
                "name": "雪林车队与吉普车厢预演 (1280×544 · 2镜10.04s)",
                "shots": [
                    {
                        "id": "shot_convoy_1",
                        "name": "第 1 镜：雪林道路与车队 (4.42s / 106帧)",
                        "updatedAt": "参考片复现",
                        "scene": convoy_shot1,
                    },
                    {
                        "id": "shot_convoy_2",
                        "name": "第 2 镜：吉普车内四人（聚焦后排 5.62s / 135帧）",
                        "updatedAt": "参考片复现",
                        "scene": convoy_shot2,
                    },
                ],
            },
            {
                "id": "scene_64",
                "name": "第 64 场：猎人木屋现场 (灰白体块风格)",
                "shots": [
                    {
                        "id": "shot_64_1",
                        "name": "第 1 镜：门口向内正打全景 (24mm)",
                        "updatedAt": "默认",
                        "scene": s64_shot1,
                    },
                    {
                        "id": "shot_64_2",
                        "name": "第 2 镜：书桌上方朝门口反打俯拍 (24mm)",
                        "updatedAt": "默认",
                        "scene": s64_shot2,
                    },
                ],
            },
            {
                "id": "scene_0",
                "name": "第 1 场：UAZ-469 吉普车四人舞台",
                "shots": [
                    {
                        "id": "shot_0_1",
                        "name": "第 1 镜：UAZ-469 吉普车四人坐姿 (35mm)",
                        "updatedAt": "默认",
                        "scene": clean_shot1,
                    },
                ],
            },
            {
                "id": "scene_1",
                "name": "第 2 场：雪林四人揭示",
                "shots": [
                    {
                        "id": "shot_1_1",
                        "name": "第 1 镜：贴背仰拍右摇拉出至四人全景 (参考片复现)",
                        "updatedAt": "默认",
                        "scene": s1_shot1,
                    },
                    {
                        "id": "shot_1_2",
                        "name": "第 2 镜：侧面 50mm 越肩跟拍",
                        "updatedAt": "默认",
                        "scene": s1_shot2,
                    },
                    {
                        "id": "shot_1_3",
                        "name": "第 3 镜：高空 24mm 俯拍红林全景",
                        "updatedAt": "默认",
                        "scene": s1_shot3,
                    },
                ],
            },
            {
                "id": "scene_2",
                "name": "第 3 场：室内桌前沉思",
                "shots": [
                    {
                        "id": "shot_2_1",
                        "name": "第 1 镜：桌前托腮缓慢推镜",
                        "updatedAt": "默认",
                        "scene": s2_stage,
                    },
                ],
            },
        ],
    }


def load_project_bank() -> dict:
    # Reading an existing project must never regenerate templates or write the file.
    with PROJECT_IO_LOCK:
        if os.path.exists(PROJECTS_FILE):
            mtime = os.path.getmtime(PROJECTS_FILE)
            if _PROJECT_BANK_CACHE["bank"] is not None and _PROJECT_BANK_CACHE["mtime"] == mtime:
                return _PROJECT_BANK_CACHE["bank"]
            with open(PROJECTS_FILE, "r", encoding="utf-8") as stream:
                data = json.load(stream)
            if not isinstance(data, dict) or not isinstance(data.get("scenes"), list):
                raise ValueError("工程文件格式无效；原文件保留，未用默认场景覆盖")
            _PROJECT_BANK_CACHE["mtime"] = mtime
            _PROJECT_BANK_CACHE["bank"] = data
            return data
        bank = build_default_project_bank()
        save_project_bank(bank)
        return bank


def save_project_bank(bank: dict) -> None:
    if not isinstance(bank, dict) or not isinstance(bank.get("scenes"), list):
        raise ValueError("Invalid project_bank")
    body = json.dumps(bank, ensure_ascii=False, indent=2)
    with PROJECT_IO_LOCK:
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=os.path.dirname(PROJECTS_FILE),
                                             prefix=".project-save-", suffix=".tmp", delete=False) as stream:
                temp_name = stream.name
                stream.write(body)
                stream.flush()
            os.replace(temp_name, PROJECTS_FILE)
            _PROJECT_BANK_CACHE["mtime"] = os.path.getmtime(PROJECTS_FILE)
            _PROJECT_BANK_CACHE["bank"] = bank
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)


def _sync_live_state_from_saved_bank() -> None:
    try:
        if not os.path.exists(PROJECTS_FILE):
            return
        bank = load_project_bank()
        scenes = bank.get("scenes") or []
        active_sc = next((s for s in scenes if s.get("id") == bank.get("activeSceneId")), None) or (scenes[0] if scenes else None)
        shots = (active_sc or {}).get("shots") or []
        active_sh = next((k for k in shots if k.get("id") == bank.get("activeShotId")), None) or (shots[0] if shots else None)
        if active_sh and isinstance(active_sh.get("scene"), dict):
            active_sh["scene"]["hide_edges"] = True
            LIVE_STATE["scene"] = active_sh["scene"]
            LIVE_STATE["python_code"] = scene_dict_to_python_code(active_sh["scene"])
    except Exception:
        pass


_sync_live_state_from_saved_bank()


def _find_ffmpeg_exe() -> Optional[str]:
    """定位可用 ffmpeg 可执行文件路径（优先系统 PATH，次选 imageio_ffmpeg 内置二进制）。"""
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        return ffmpeg_exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _sanitize_export_label(val: str, fallback: str, max_len: int = 42) -> str:
    """清洗场次/镜头文件名片段，确保跨平台路径安全。"""
    cleaned = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", str(val or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._ ")
    return cleaned[:max_len] if cleaned else fallback


def _resolve_active_scene_shot_labels() -> Tuple[str, str]:
    """从工程库解析当前激活场次与镜头名称，用于按场景归档至下载文件夹。"""
    try:
        bank = load_project_bank()
        scenes = bank.get("scenes") or []
        active_sc = next((s for s in scenes if s.get("id") == bank.get("activeSceneId")), None) or (scenes[0] if scenes else None)
        shots = (active_sc or {}).get("shots") or []
        active_sh = next((k for k in shots if k.get("id") == bank.get("activeShotId")), None) or (shots[0] if shots else None)
        sc_name = str((active_sc or {}).get("name") or (active_sc or {}).get("id") or "未命名场景").strip()
        sh_name = str((active_sh or {}).get("name") or (active_sh or {}).get("id") or "第1镜").strip()
        return sc_name or "未命名场景", sh_name or "第1镜"
    except Exception:
        return "未命名场景", "第1镜"


def _copy_file_unique(src_file: str, target_dir: str, stem: str, ext: str) -> Tuple[str, str]:
    """将文件复制到目标目录并自动避让重名文件，返回 (filename, dest_path)。"""
    os.makedirs(target_dir, exist_ok=True)
    filename = f"{stem}{ext}"
    dest_path = os.path.join(target_dir, filename)
    collision_idx = 1
    while os.path.exists(dest_path):
        filename = f"{stem}_{collision_idx}{ext}"
        dest_path = os.path.join(target_dir, filename)
        collision_idx += 1
    shutil.copy2(src_file, dest_path)
    try:
        os.chmod(dest_path, 0o644)
    except OSError:
        pass
    return filename, dest_path


def _save_file_to_downloads_scene_dir(
    src_file: str, scene_label: str, stem: str, ext: str
) -> Tuple[str, str]:
    """按场景名称创建 ~/Downloads/<场景名>/ 文件夹并保存不覆写副本，返回 (downloads_dir, downloads_path)。"""
    safe_scene_dir = _sanitize_export_label(scene_label, "未命名场景", max_len=64)
    downloads_dir = os.path.join(os.path.expanduser("~/Downloads"), safe_scene_dir)
    _, downloads_path = _copy_file_unique(src_file, downloads_dir, stem, ext)
    return downloads_dir, downloads_path


def _save_export_with_optional_desktop(
    src_file: str,
    base_stem: str,
    ext: str,
    save_to_desktop: bool,
    scene_label: str = "",
    save_to_downloads: bool = True,
):
    """写入 exports/ 目录并按场景文件夹保存到 ~/Downloads/<场景名>/，可选同步一份到桌面。"""
    filename, export_out_path = _copy_file_unique(src_file, EXPORT_DIR, base_stem, ext)
    final_stem = os.path.splitext(filename)[0]

    downloads_dir = ""
    downloads_out_path = ""
    if save_to_downloads:
        eff_scene = str(scene_label or "").strip()
        if not eff_scene or eff_scene == "场次":
            eff_scene, _ = _resolve_active_scene_shot_labels()
        downloads_dir, downloads_out_path = _save_file_to_downloads_scene_dir(
            export_out_path, eff_scene, final_stem, ext
        )

    desktop_out_path = ""
    if save_to_desktop:
        desktop_dir = os.path.expanduser("~/Desktop")
        _, desktop_out_path = _copy_file_unique(export_out_path, desktop_dir, final_stem, ext)

    file_size = os.path.getsize(export_out_path) if os.path.exists(export_out_path) else 0
    return filename, export_out_path, desktop_out_path, downloads_out_path, downloads_dir, file_size


class DirectorHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # 保持日志简洁
        pass

    def _safe_write_bytes(self, content: bytes) -> bool:
        try:
            self.wfile.write(content)
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html", "/five_men_15s_demo.html", "/su_minimal_15s_demo.html"):
            target_html = path.lstrip("/") if path in ("/five_men_15s_demo.html", "/su_minimal_15s_demo.html") else "index.html"
            index_path = os.path.join(STATIC_DIR, target_html)
            if not os.path.exists(index_path):
                self._send_json({"error": f"{target_html} not found"}, 404)
                return
            with open(index_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self._safe_write_bytes(content)
            return

        if path.startswith("/vendor/"):
            fname = os.path.basename(path)
            fpath = os.path.join(STATIC_DIR, "vendor", fname)
            if not os.path.exists(fpath):
                self._send_json({"error": f"Vendor file not found: {fname}"}, 404)
                return
            with open(fpath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return

        if path.startswith("/plugins/"):
            fname = os.path.basename(unquote(path))
            fpath = os.path.join(PLUGINS_DIR, fname)
            if not os.path.exists(fpath) or not os.path.isfile(fpath):
                self._send_json({"error": f"Plugin file not found: {fname}"}, 404)
                return
            with open(fpath, "rb") as f:
                content = f.read()
            ctype = "text/css; charset=utf-8" if fname.endswith(".css") else "application/javascript; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return

        if path == "/api/default_scene":
            self._send_json({
                "ok": True,
                "version": LIVE_STATE["version"],
                "scene": LIVE_STATE["scene"],
                "python_code": LIVE_STATE["python_code"],
                "presets": PRESET_IK_POSES,
                "project_bank": load_project_bank(),
            })
            return

        if path == "/api/projects":
            self._send_json({
                "ok": True,
                "project_bank": load_project_bank(),
            })
            return

        if path == "/api/repro/env_status":
            self._send_json({"ok": True, **detect_repro_environment()})
            return

        if path == "/api/repro/job_status":
            qs = parse_qs(parsed.query)
            job_id = str(qs.get("job_id", [""])[0]).strip()
            self._send_json(get_repro_job_status(job_id))
            return

        if path == "/api/live_state":
            qs = parse_qs(parsed.query)
            client_v = int(qs.get("version", ["0"])[0])
            if client_v < LIVE_STATE["version"]:
                self._send_json({
                    "ok": True,
                    "changed": True,
                    "version": LIVE_STATE["version"],
                    "scene": LIVE_STATE["scene"],
                    "python_code": LIVE_STATE["python_code"],
                    "last_prompt": LIVE_STATE["last_prompt"],
                    "project_bank": load_project_bank(),
                })
            else:
                self._send_json({"ok": True, "changed": False, "version": LIVE_STATE["version"]})
            return

        if path.startswith("/ref_videos/"):
            fname = os.path.basename(unquote(path))
            fpath = os.path.join(REF_VIDEO_DIR, fname)
            if not os.path.exists(fpath):
                self._send_json({"error": f"Reference video not found: {fname}"}, 404)
                return
            file_size = os.path.getsize(fpath)
            ctype = "video/quicktime" if fname.lower().endswith(".mov") else "video/mp4"
            range_hdr = self.headers.get("Range")
            if range_hdr and range_hdr.startswith("bytes="):
                m_rng = re.match(r"bytes=(\d+)-(\d*)", range_hdr)
                if m_rng:
                    start = int(m_rng.group(1))
                    end = int(m_rng.group(2)) if m_rng.group(2) else (file_size - 1)
                    end = min(file_size - 1, max(start, end))
                    chunk_len = (end - start) + 1
                    with open(fpath, "rb") as f:
                        f.seek(start)
                        chunk = f.read(chunk_len)
                    self.send_response(206)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(chunk)
                    return
            with open(fpath, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
            return

        if path.startswith("/exports/comparisons/"):
            fname = os.path.basename(unquote(path))
            fpath = os.path.join(COMPARISON_DIR, fname)
            if not os.path.exists(fpath):
                self._send_json({"error": f"Comparison image not found: {fname}"}, 404)
                return
            with open(fpath, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        if path.startswith("/exports/"):
            fname = os.path.basename(unquote(path))
            fpath = os.path.join(EXPORT_DIR, fname)
            if not os.path.exists(fpath):
                self._send_json({"error": "Video not found"}, 404)
                return
            with open(fpath, "rb") as f:
                data = f.read()
            v_ctype = "video/webm" if fname.lower().endswith(".webm") else "video/mp4"
            self.send_response(200)
            self.send_header("Content-Type", v_ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        self._send_json({"error": f"Not found: {path}"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/save_h264_mp4":
            t_mux_start = time.time()
            qs = parse_qs(parsed.query)
            fps_hint = max(1, min(120, int(float((qs.get("fps") or ["24"])[0]))))
            scene_label = (qs.get("scene_label") or [""])[0]
            shot_label = (qs.get("shot_label") or [""])[0]
            custom_filename = (qs.get("filename") or [""])[0].strip()
            save_to_downloads = (qs.get("save_to_downloads") or ["1"])[0].lower() in ("1", "true", "yes")
            save_to_desktop = (qs.get("save_to_desktop") or ["0"])[0].lower() in ("1", "true", "yes")
            active_sc_name, active_sh_name = _resolve_active_scene_shot_labels()
            eff_scene_label = scene_label.strip() if scene_label.strip() and scene_label.strip() != "场次" else active_sc_name
            eff_shot_label = shot_label.strip() if shot_label.strip() and shot_label.strip() != "镜头" else active_sh_name
            tmp_out_path = None
            try:
                content_len = int(self.headers.get("Content-Length", "0"))
                if content_len <= 128:
                    raise ValueError("H.264 比特流为空或过短")
                raw_h264 = self.rfile.read(content_len)
                if len(raw_h264) <= 128:
                    raise ValueError("未读取到完整 H.264 比特流")

                ffmpeg_exe = _find_ffmpeg_exe()
                if not ffmpeg_exe:
                    raise RuntimeError("未找到可用的 ffmpeg 可执行文件进行 MP4 无损封装")

                if custom_filename:
                    raw_stem = os.path.splitext(os.path.basename(custom_filename))[0]
                    base_stem = _sanitize_export_label(raw_stem, "极速导出", max_len=80)
                else:
                    safe_scene = _sanitize_export_label(eff_scene_label, "未命名场景")
                    safe_shot = _sanitize_export_label(eff_shot_label, "第1镜")
                    ms_part = int(time.time() * 1000) % 1000
                    ts_str = f"{time.strftime('%Y%m%d_%H%M%S')}_{ms_part:03d}"
                    base_stem = f"极速导出_{safe_scene}_{safe_shot}_{ts_str}"

                tmp_out_fd, tmp_out_path = tempfile.mkstemp(prefix="mds_fastmp4_", suffix=".mp4")
                os.close(tmp_out_fd)

                # 直接无损 remux（-c:v copy，不做任何二次像素编码），毫秒级生成兼容 QuickTime/剪映/微信的 faststart MP4
                cmd = [
                    ffmpeg_exe,
                    "-y",
                    "-fflags", "+genpts",
                    "-r", str(fps_hint),
                    "-f", "h264",
                    "-i", "pipe:0",
                    "-c:v", "copy",
                    "-movflags", "+faststart",
                    tmp_out_path,
                ]
                proc = subprocess.run(cmd, input=raw_h264, capture_output=True, timeout=30)
                if proc.returncode != 0 or not os.path.exists(tmp_out_path) or os.path.getsize(tmp_out_path) <= 256:
                    err_msg = (proc.stderr or b"").decode("utf-8", errors="ignore").strip()
                    raise RuntimeError(f"FFmpeg 无损封装 MP4 失败: {err_msg or proc.returncode}")

                filename, export_out_path, desktop_out_path, downloads_out_path, downloads_dir, file_size = _save_export_with_optional_desktop(
                    tmp_out_path,
                    base_stem,
                    ".mp4",
                    save_to_desktop,
                    scene_label=eff_scene_label,
                    save_to_downloads=save_to_downloads,
                )
                mux_ms = int((time.time() - t_mux_start) * 1000)
                self._send_json({
                    "ok": True,
                    "filename": filename,
                    "video_url": f"/exports/{filename}?t={int(time.time() * 1000)}",
                    "export_path": export_out_path,
                    "downloads_dir": downloads_dir,
                    "downloads_path": downloads_out_path,
                    "desktop_path": desktop_out_path,
                    "saved_path": downloads_out_path or desktop_out_path or export_out_path,
                    "mux_ms": mux_ms,
                    "file_size_bytes": file_size,
                })
            except Exception as e:
                self._send_json({"ok": False, "error": f"极速 H.264 封装保存失败: {e}"}, 200)
            finally:
                if tmp_out_path and os.path.exists(tmp_out_path):
                    try:
                        os.remove(tmp_out_path)
                    except Exception:
                        pass
            return

        try:
            payload = self._read_json()
        except Exception as e:
            self._send_json({"ok": False, "error": f"JSON 解析错误: {e}"}, 400)
            return

        if path == "/api/hot_activate_scene":
            scene = payload.get("scene")
            if not isinstance(scene, dict) or "objects" not in scene:
                self._send_json({"ok": False, "error": "Invalid scene dict"}, 400)
                return
            scene["hide_edges"] = True
            scene_id = str(payload.get("scene_id", "scene_fast_repro"))
            scene_name = str(payload.get("scene_name", "极速复现场景"))
            shot_id = str(payload.get("shot_id", "shot_fast_1"))
            shot_name = str(payload.get("shot_name", "第 1 镜：SU极简彩色预演"))
            LIVE_STATE["scene"] = scene
            LIVE_STATE["python_code"] = scene_dict_to_python_code(scene)
            LIVE_STATE["version"] = int(LIVE_STATE.get("version", 37)) + 1
            rv_path = (scene.get("ref_video") or {}).get("path")
            if rv_path and os.path.isfile(rv_path):
                LIVE_STATE["last_ref_video_path"] = rv_path
            bank = load_project_bank()
            existing = next((s for s in bank["scenes"] if s.get("id") == scene_id), None)
            shot_entry = {
                "id": shot_id,
                "name": shot_name,
                "updatedAt": "SU极简彩色预演",
                "scene": scene,
            }
            if existing is None:
                bank["scenes"].insert(0, {"id": scene_id, "name": scene_name, "shots": [shot_entry]})
            else:
                existing["name"] = scene_name
                existing["shots"] = [shot_entry]
            bank["activeSceneId"] = scene_id
            bank["activeShotId"] = shot_id
            save_project_bank(bank)
            self._send_json({"ok": True, "version": LIVE_STATE["version"], "activeSceneId": scene_id})
            return

        if path == "/api/keyframes/simplify":
            try:
                scene = payload.get("scene")
                if not isinstance(scene, dict) or not isinstance(scene.get("objects"), list):
                    raise ValueError("Invalid scene")
                sparse, stats = sparsify_scene(scene)
                self._send_json({"ok": True, "scene": sparse, "stats": stats})
            except Exception as error:
                self._send_json({"ok": False, "error": str(error)}, 400)
            return

        if path == "/api/projects":
            bank = payload.get("project_bank")
            if isinstance(bank, dict) and "scenes" in bank:
                try:
                    save_project_bank(bank)
                    _sync_live_state_from_saved_bank()
                    self._send_json({
                        "ok": True,
                        "saved_at": time.strftime("%H:%M:%S"),
                        "synced_live_state": True,
                    })
                except Exception as e:
                    self._send_json({"ok": False, "error": f"保存工程失败: {e}"}, 500)
            else:
                self._send_json({"ok": False, "error": "Invalid project_bank"}, 400)
            return

        if path == "/api/ref_video/load_path":
            req_path = str(payload.get("path", "")).strip()
            use_default = bool(payload.get("use_default", False))
            target_path = DEFAULT_REF_VIDEO_PATH if (use_default or not req_path) else req_path
            try:
                meta = probe_video_metadata(target_path)
                # 若文件不在 REF_VIDEO_DIR 内，复制到本机 REF_VIDEO_DIR 以便浏览器 <video> 流式播放
                safe_fname = re.sub(r"[^a-zA-Z0-9._-\u4e00-\u9fa5]", "_", os.path.basename(target_path))
                served_path = os.path.join(REF_VIDEO_DIR, safe_fname)
                if os.path.abspath(target_path) != os.path.abspath(served_path):
                    shutil.copy2(target_path, served_path)
                thumbs = extract_video_thumbnails(served_path, count=int(payload.get("thumb_count", 11)))
                LIVE_STATE["last_ref_video_path"] = served_path
                self._send_json({
                    "ok": True,
                    "video_url": f"/ref_videos/{safe_fname}?t={int(time.time()*1000)}",
                    "local_path": served_path,
                    "source_path": target_path,
                    "meta": meta,
                    "thumbnails": thumbs,
                })
            except PermissionError as pe:
                # 针对 macOS TCC 沙盒容器目录 (如 ~/Library/Containers/com.tencent.xinWeChat/...) 权限限制给出明确提示，并支持回退本机已缓存的 10s 参考片
                fallback_available = os.path.exists(DEFAULT_REF_VIDEO_PATH)
                self._send_json({
                    "ok": False,
                    "permission_denied": True,
                    "fallback_available": fallback_available,
                    "error": (
                        f"macOS 沙盒权限限制 (Operation not permitted)：后台进程无法直接读取微信容器目录 {target_path}。"
                        "请点击界面上的【📂 选择本地 MP4/MOV】按钮在浏览器文件选择器中直接选中该视频，或点击【🎞 加载内置10s参考片】！"
                    ),
                })
            except Exception as e:
                self._send_json({"ok": False, "error": str(e), "traceback": traceback.format_exc()})
            return

        if path == "/api/ref_video/upload":
            fname = os.path.basename(str(payload.get("filename", "ref_video.mp4")))
            safe_fname = re.sub(r"[^a-zA-Z0-9._-\u4e00-\u9fa5]", "_", fname)
            if not safe_fname.lower().endswith((".mp4", ".mov")):
                safe_fname += ".mp4"
            b64_data = str(payload.get("data_b64", ""))
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            try:
                raw_bytes = base64.b64decode(b64_data)
                served_path = os.path.join(REF_VIDEO_DIR, safe_fname)
                with open(served_path, "wb") as f:
                    f.write(raw_bytes)
                meta = probe_video_metadata(served_path)
                thumbs = extract_video_thumbnails(served_path, count=int(payload.get("thumb_count", 11)))
                LIVE_STATE["last_ref_video_path"] = served_path
                self._send_json({
                    "ok": True,
                    "video_url": f"/ref_videos/{safe_fname}?t={int(time.time()*1000)}",
                    "local_path": served_path,
                    "meta": meta,
                    "thumbnails": thumbs,
                })
            except Exception as e:
                self._send_json({"ok": False, "error": f"本地参考视频解析失败: {e}", "traceback": traceback.format_exc()})
            return

        if path == "/api/export_comparison":
            scene = payload.get("scene", {})
            t_proj = float(payload.get("t", 0.0))
            compare_mode = str(payload.get("compare_mode", "side_by_side"))
            overlay_alpha = float(payload.get("overlay_alpha", 0.5))
            wipe_ratio = float(payload.get("wipe_ratio", 0.5))
            canvas_data_url = str(payload.get("canvas_data_url", ""))
            ref_path = payload.get("ref_video_path") or LIVE_STATE.get("last_ref_video_path") or DEFAULT_REF_VIDEO_PATH

            ts_str = time.strftime("%Y%m%d_%H%M%S")
            t_tag = f"{t_proj:.2f}s".replace(".", "_")
            fname = f"compare_t{t_tag}_{ts_str}.png"
            out_path = os.path.join(COMPARISON_DIR, fname)

            try:
                if canvas_data_url and canvas_data_url.startswith("data:image/png;base64,"):
                    raw_png = base64.b64decode(canvas_data_url.split(",", 1)[1])
                    with open(out_path, "wb") as f:
                        f.write(raw_png)
                    proj_fps = float(scene.get("fps", 24.0))
                    ref_fps = float((scene.get("ref_video") or {}).get("fps", 24.0))
                    t_offset = float((scene.get("ref_video") or {}).get("time_offset", 0.0))
                    t_ref = max(0.0, t_proj + t_offset)
                    info = {
                        "output_path": out_path,
                        "t_proj": round(t_proj, 3),
                        "proj_frame": int(round(t_proj * proj_fps)),
                        "proj_fps": proj_fps,
                        "t_ref": round(t_ref, 3),
                        "ref_frame": int(round(t_ref * ref_fps)),
                        "ref_fps": ref_fps,
                    }
                else:
                    info = render_comparison_image(
                        scene,
                        t_proj,
                        ref_path if (ref_path and os.path.exists(ref_path)) else None,
                        out_path,
                        compare_mode=compare_mode,
                        overlay_alpha=overlay_alpha,
                        wipe_ratio=wipe_ratio,
                    )
                self._send_json({
                    "ok": True,
                    "url": f"/exports/comparisons/{fname}?t={int(time.time()*1000)}",
                    "filename": fname,
                    **info,
                })
            except Exception as e:
                self._send_json({"ok": False, "error": str(e), "traceback": traceback.format_exc()})
            return

        if path == "/api/export_multi_comparisons":
            scene = payload.get("scene", {})
            times_raw = payload.get("times", [0, 2, 4, 6, 8, 9])
            times = [float(x) for x in times_raw] if isinstance(times_raw, list) else [0.0, 2.0, 4.0, 6.0, 8.0, 9.0]
            ref_path = payload.get("ref_video_path") or LIVE_STATE.get("last_ref_video_path") or DEFAULT_REF_VIDEO_PATH
            client_frames = payload.get("client_frames")  # 可选：前端 Three.js 截取的各时间点对照图列表

            try:
                if isinstance(client_frames, list) and len(client_frames) > 0:
                    from PIL import Image
                    import io as _io
                    ts_tag = int(time.time() * 1000)
                    items = []
                    pil_rows = []
                    for item in client_frames:
                        t_val = float(item.get("t", 0.0))
                        d_url = str(item.get("data_url", ""))
                        if "," in d_url:
                            raw = base64.b64decode(d_url.split(",", 1)[1])
                            t_str = f"{t_val:.1f}s".replace(".", "_")
                            fn = f"compare_t{t_str}_{ts_tag}.png"
                            fp = os.path.join(COMPARISON_DIR, fn)
                            with open(fp, "wb") as f:
                                f.write(raw)
                            im = Image.open(_io.BytesIO(raw)).convert("RGB")
                            pil_rows.append(im)
                            items.append({
                                "t_proj": t_val,
                                "filename": fn,
                                "output_path": fp,
                                "url": f"/exports/comparisons/{fn}",
                                "width": im.width,
                                "height": im.height,
                            })
                    sheet_path = ""
                    sheet_url = ""
                    if pil_rows:
                        rw, rh = pil_rows[0].size
                        gap_y = 8
                        sh = len(pil_rows) * rh + max(0, len(pil_rows) - 1) * gap_y
                        sheet_im = Image.new("RGB", (rw, sh), (10, 12, 16))
                        for idx, rim in enumerate(pil_rows):
                            sheet_im.paste(rim, (0, idx * (rh + gap_y)))
                        s_fn = f"compare_sheet_{'_'.join(str(int(t)) if float(t).is_integer() else str(t) for t in times)}s_{ts_tag}.png"
                        sheet_path = os.path.join(COMPARISON_DIR, s_fn)
                        sheet_im.save(sheet_path, format="PNG")
                        sheet_url = f"/exports/comparisons/{s_fn}"
                    self._send_json({
                        "ok": True,
                        "items": items,
                        "sheet_path": sheet_path,
                        "sheet_url": sheet_url,
                        "times": times,
                    })
                else:
                    res = render_multi_time_comparisons(
                        scene,
                        times,
                        ref_path if (ref_path and os.path.exists(ref_path)) else None,
                        COMPARISON_DIR,
                    )
                    self._send_json({"ok": True, **res})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e), "traceback": traceback.format_exc()})
            return

        if path == "/api/track_motion_stats":
            is_cam = bool(payload.get("is_camera", False))
            obj = payload.get("obj") if isinstance(payload.get("obj"), dict) else None
            scene_ctx = payload.get("scene") if isinstance(payload.get("scene"), dict) else None
            kfs = payload.get("keyframes") or (obj.get("keyframes", []) if obj else [])
            stats = compute_track_motion_stats(kfs, is_camera=is_cam, obj=obj, scene=scene_ctx)
            self._send_json({"ok": True, **stats})
            return

        if path == "/api/scene_to_python":
            scene = payload.get("scene", {})
            code = scene_dict_to_python_code(scene)
            self._send_json({"ok": True, "python_code": code})
            return

        if path == "/api/prompt_to_pose":
            prompt = payload.get("prompt", "")
            if not prompt.strip():
                self._send_json({"ok": False, "error": "提示词不能为空"}, 200)
                return
            try:
                res = prompt_to_stage(prompt)
                LIVE_STATE["version"] += 1
                LIVE_STATE["scene"] = res["scene"]
                LIVE_STATE["python_code"] = res["python_code"]
                LIVE_STATE["last_prompt"] = res["prompt"]
                self._send_json({
                    "ok": True,
                    "version": LIVE_STATE["version"],
                    **res,
                })
            except Exception as e:
                self._send_json({"ok": False, "error": str(e), "traceback": traceback.format_exc()}, 200)
            return

        if path == "/api/repro/prepare_confirm":
            req_path = str(payload.get("video_path", "")).strip()
            target_path = req_path if (req_path and os.path.exists(req_path)) else (LIVE_STATE.get("last_ref_video_path") or DEFAULT_REF_VIDEO_PATH)
            existing_scene = payload.get("existing_scene") if isinstance(payload.get("existing_scene"), dict) else None
            priority_mode = str(payload.get("priority_mode") or "fast")
            composition_text = payload.get("composition_text")
            try:
                summary = prepare_video_repro_confirmation_proposal(
                    video_path=target_path,
                    existing_scene=existing_scene,
                    priority_mode=priority_mode,
                    composition_text=composition_text,
                )
                self._send_json({"ok": True, "summary": summary})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e), "traceback": traceback.format_exc()}, 200)
            return

        if path == "/api/repro/start_job":
            req_path = str(payload.get("video_path", "")).strip()
            target_path = req_path if (req_path and os.path.exists(req_path)) else (LIVE_STATE.get("last_ref_video_path") or DEFAULT_REF_VIDEO_PATH)
            calibration = payload.get("calibration") if isinstance(payload.get("calibration"), dict) else {}
            existing_scene = payload.get("existing_scene") if isinstance(payload.get("existing_scene"), dict) else None
            confirmation = payload.get("confirmation") if isinstance(payload.get("confirmation"), dict) else {}
            saved_plan = (existing_scene or {}).get("export_plan") if isinstance((existing_scene or {}).get("export_plan"), dict) else {}
            is_confirmed = bool(
                confirmation.get("confirmed") is True
                or (confirmation.get("reuse_saved_plan") is True and saved_plan.get("confirmed") is True)
            )
            if not is_confirmed:
                summary = prepare_video_repro_confirmation_proposal(
                    video_path=target_path,
                    existing_scene=existing_scene,
                    priority_mode=str(confirmation.get("priority_mode") or calibration.get("priority_mode") or "fast"),
                    composition_text=confirmation.get("composition_text"),
                )
                self._send_json({
                    "ok": False,
                    "confirmation_required": True,
                    "message": "开始自动生成白模场景前，请先确认关键构图、优先档位与面积/可见总量预算。",
                    "summary": summary,
                }, 200)
                return
            if confirmation.get("priority_mode") and "priority_mode" not in calibration:
                calibration = dict(calibration, priority_mode=confirmation.get("priority_mode"))
            try:
                job_info = start_repro_analysis_job(
                    video_path=target_path,
                    calibration=calibration,
                    existing_scene=existing_scene,
                )
                self._send_json(job_info)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e), "traceback": traceback.format_exc()}, 200)
            return

        if path == "/api/repro/cancel_job":
            job_id = str(payload.get("job_id", "")).strip()
            self._send_json(cancel_repro_job(job_id))
            return

        if path == "/api/repro/refit":
            req_path = str(payload.get("video_path", "")).strip()
            target_path = req_path if (req_path and os.path.exists(req_path)) else (LIVE_STATE.get("last_ref_video_path") or DEFAULT_REF_VIDEO_PATH)
            calibration = payload.get("calibration") if isinstance(payload.get("calibration"), dict) else {}
            existing_scene = payload.get("existing_scene") if isinstance(payload.get("existing_scene"), dict) else None
            confirmation = payload.get("confirmation") if isinstance(payload.get("confirmation"), dict) else {}
            saved_plan = (existing_scene or {}).get("export_plan") if isinstance((existing_scene or {}).get("export_plan"), dict) else {}
            is_confirmed = bool(
                confirmation.get("confirmed") is True
                or (confirmation.get("reuse_saved_plan") is True and saved_plan.get("confirmed") is True)
            )
            if not is_confirmed:
                summary = prepare_video_repro_confirmation_proposal(
                    video_path=target_path,
                    existing_scene=existing_scene,
                    priority_mode=str(confirmation.get("priority_mode") or calibration.get("priority_mode") or "fast"),
                    composition_text=confirmation.get("composition_text"),
                )
                self._send_json({
                    "ok": False,
                    "confirmation_required": True,
                    "message": "重算前请先确认关键构图、优先档位与预算配置。",
                    "summary": summary,
                }, 200)
                return
            if confirmation.get("priority_mode") and "priority_mode" not in calibration:
                calibration = dict(calibration, priority_mode=confirmation.get("priority_mode"))
            try:
                res = solve_white_model_video_reproduction(
                    video_path=target_path,
                    calibration=calibration,
                    existing_scene=existing_scene,
                )
                self._send_json({"ok": True, **res})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e), "traceback": traceback.format_exc()}, 200)
            return

        if path == "/api/run_python":
            code = payload.get("code", "")
            prev_scene = payload.get("prev_scene") if isinstance(payload.get("prev_scene"), dict) else None
            isolated_preview = bool(payload.get("isolated_preview", False))
            stage = Stage()
            if prev_scene:
                if prev_scene.get("ref_video"):
                    stage.ref_video = copy.deepcopy(prev_scene["ref_video"])
                if prev_scene.get("repro_state"):
                    stage.repro_state = copy.deepcopy(prev_scene["repro_state"])
                if prev_scene.get("point_cloud_helper"):
                    stage.point_cloud_helper = copy.deepcopy(prev_scene["point_cloud_helper"])
                if prev_scene.get("export_plan"):
                    stage.export_plan = copy.deepcopy(prev_scene["export_plan"])
                if prev_scene.get("camera_speed_curve"):
                    stage.camera_speed_curve = copy.deepcopy(prev_scene["camera_speed_curve"])
                if prev_scene.get("camera_rig"):
                    stage.camera_rig = copy.deepcopy(prev_scene["camera_rig"])
            # 注入安全且强大的内置环境
            env = {
                "stage": stage,
                "Stage": Stage,
                "PRESET_IK_POSES": PRESET_IK_POSES,
                "math": __import__("math"),
            }
            try:
                exec(code, env)
                scene_dict = stage.to_dict()
                scene_dict["hide_edges"] = True
                if not isolated_preview:
                    LIVE_STATE["version"] += 1
                    LIVE_STATE["scene"] = scene_dict
                    LIVE_STATE["python_code"] = code
                video_url = None
                if stage._last_exported_video:
                    video_url = f"/exports/{stage._last_exported_video}?t={int(__import__('time').time()*1000)}"
                pending_summary = getattr(stage, "_pending_export_confirmation", None)
                self._send_json({
                    "ok": True,
                    "version": LIVE_STATE["version"],
                    "scene": scene_dict,
                    "exported_video_url": video_url,
                    "confirmation_required": bool(pending_summary is not None),
                    "pending_export_summary": pending_summary,
                })
            except Exception as e:
                tb = traceback.format_exc()
                self._send_json({"ok": False, "error": str(e), "traceback": tb}, 200)
            return

        if path == "/api/verify_5frames_parity":
            scene = payload.get("scene", {})
            browser_frames = payload.get("browser_frames", [])
            width = int(payload.get("width") or 574)
            height = int(payload.get("height") or 240)
            try:
                res = render_5frame_parity_sheet(
                    scene,
                    browser_frames,
                    COMPARISON_DIR,
                    width=width,
                    height=height,
                )
                self._send_json({"ok": True, **res})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e), "traceback": traceback.format_exc()}, 200)
            return

        if path == "/api/ref_video/check_path":
            raw_path = str(payload.get("path", "")).strip()
            exists = bool(raw_path and os.path.exists(raw_path) and os.path.isfile(raw_path))
            self._send_json({
                "ok": True,
                "path": raw_path,
                "exists": exists,
            })
            return

        if path == "/api/export_prepare":
            scene = payload.get("scene", {})
            priority_mode = payload.get("priority_mode")
            compositions = payload.get("compositions") if isinstance(payload.get("compositions"), list) else None
            composition_text = payload.get("composition_text")
            render_previews = bool(payload.get("render_previews", True))
            preview_w = int(payload.get("preview_width") or 320)
            preview_h = int(payload.get("preview_height") or 144)
            try:
                t_prep0 = time.time()
                summary = prepare_export_confirmation_summary(
                    scene,
                    priority_mode=priority_mode,
                    compositions=compositions,
                    composition_text=composition_text,
                    render_previews=render_previews,
                    preview_width=preview_w,
                    preview_height=preview_h,
                )
                summary["prepare_elapsed_ms"] = int((time.time() - t_prep0) * 1000)
                self._send_json({"ok": True, "summary": summary})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e), "traceback": traceback.format_exc()}, 200)
            return

        if path == "/api/export_video":
            scene = payload.get("scene", {})
            width = int(payload.get("width", 960))
            height = int(payload.get("height", 540))
            export_confirmation = payload.get("export_confirmation") if isinstance(payload.get("export_confirmation"), dict) else None
            gate = check_export_confirmation_gate(scene, export_confirmation)
            if not gate["confirmed"]:
                self._send_json({
                    "ok": False,
                    "confirmation_required": True,
                    "message": "正式导出 MP4 前请先确认关键构图、极速/均衡/构图优先档位与可见预算。",
                    "summary": gate["summary"],
                }, 200)
                return

            active_sc_name, active_sh_name = _resolve_active_scene_shot_labels()
            raw_scene_name = str(
                payload.get("scene_name")
                or payload.get("scene_label")
                or scene.get("scene_name")
                or scene.get("name")
                or active_sc_name
            ).strip()
            raw_shot_name = str(
                payload.get("shot_name")
                or payload.get("shot_label")
                or scene.get("shot_name")
                or active_sh_name
            ).strip()
            if raw_scene_name in ("", "场次"):
                raw_scene_name = active_sc_name
            if raw_shot_name in ("", "镜头"):
                raw_shot_name = active_sh_name
            save_to_downloads = bool(payload.get("save_to_downloads", True))

            os.makedirs(EXPORT_DIR, exist_ok=True)
            raw_filename = str(payload.get("filename", "")).strip()
            ms_suffix = int(time.time() * 1000) % 1000
            ts_label = f"{time.strftime('%Y%m%d_%H%M%S')}_{ms_suffix:03d}"
            if (not raw_filename) or raw_filename.startswith("director-stage-"):
                safe_scene_stem = _sanitize_export_label(raw_scene_name, "未命名场景", max_len=48)
                safe_shot_stem = _sanitize_export_label(raw_shot_name, "第1镜", max_len=36)
                safe_name = f"{safe_scene_stem}_{safe_shot_stem}_{ts_label}.mp4"
            else:
                safe_name = os.path.basename(raw_filename)
                if not safe_name.endswith(".mp4"):
                    safe_name += ".mp4"
                # 始终避免覆盖已有成片：若同名文件已存在，自动附加唯一时间戳后缀
                candidate_path = os.path.join(EXPORT_DIR, safe_name)
                if os.path.exists(candidate_path):
                    stem = safe_name[:-4]
                    safe_name = f"{stem}_{ts_label}.mp4"
            out_path = os.path.join(EXPORT_DIR, safe_name)

            try:
                stats = render_scene_to_mp4(
                    scene,
                    out_path,
                    width=width,
                    height=height,
                    export_confirmation=gate["plan"],
                    require_confirmation=True,
                )
                downloads_dir = ""
                downloads_path = ""
                if save_to_downloads and os.path.exists(out_path):
                    downloads_dir, downloads_path = _save_file_to_downloads_scene_dir(
                        out_path, raw_scene_name, os.path.splitext(safe_name)[0], ".mp4"
                    )
                ts = int(time.time() * 1000)
                file_size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
                self._send_json({
                    "ok": True,
                    "video_url": f"/exports/{safe_name}?t={ts}",
                    "filename": safe_name,
                    "output_path": out_path,
                    "downloads_dir": downloads_dir,
                    "downloads_path": downloads_path,
                    "saved_path": downloads_path or out_path,
                    "scene_name": raw_scene_name,
                    "shot_name": raw_shot_name,
                    "project_snapshot_path": stats.get("project_snapshot_path", ""),
                    "file_size_bytes": file_size,
                    "elapsed_ms": stats["elapsed_ms"],
                    "frames": stats["frames"],
                    "fps": stats["fps"],
                    "duration": stats["duration"],
                    "export_plan": gate["plan"],
                })
            except Exception as e:
                self._send_json({"ok": False, "error": str(e), "traceback": traceback.format_exc()}, 200)
            return

        if path == "/api/save_desktop_screenshot":
            image_data_url = str(payload.get("image_data_url", "")).strip()
            scene_label = str(payload.get("scene_label") or payload.get("scene_id") or "场次").strip()
            shot_label = str(payload.get("shot_label") or payload.get("shot_id") or "镜头").strip()
            width = int(payload.get("width") or 1280)
            height = int(payload.get("height") or 720)
            scene = payload.get("scene")
            t_val = float(payload.get("t") or 0.0)

            try:
                desktop_dir = os.path.expanduser("~/Desktop")
                if not os.path.isdir(desktop_dir):
                    os.makedirs(desktop_dir, exist_ok=True)

                def _clean_part(val: str, fallback: str) -> str:
                    cleaned = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", val)
                    cleaned = re.sub(r"\s+", "_", cleaned).strip("._ ")
                    return cleaned[:48] if cleaned else fallback

                safe_scene = _clean_part(scene_label, "场次")
                safe_shot = _clean_part(shot_label, "镜头")
                ms_part = int(time.time() * 1000) % 1000
                ts_str = f"{time.strftime('%Y%m%d_%H%M%S')}_{ms_part:03d}"
                base_stem = f"{safe_scene}_{safe_shot}_{ts_str}"
                filename = f"{base_stem}.png"
                out_path = os.path.join(desktop_dir, filename)
                collision_idx = 1
                while os.path.exists(out_path):
                    filename = f"{base_stem}_{collision_idx}.png"
                    out_path = os.path.join(desktop_dir, filename)
                    collision_idx += 1

                if image_data_url and "," in image_data_url:
                    b64_part = image_data_url.split(",", 1)[1]
                    raw_png = base64.b64decode(b64_part)
                    if len(raw_png) < 64:
                        raise ValueError("截图数据为空或损坏")
                    with open(out_path, "wb") as f:
                        f.write(raw_png)
                elif isinstance(scene, dict):
                    pil_img = _stamp_camera_hud_on_pil(
                        render_scene_frame_pil(scene, t_val, width=width, height=height),
                        scene,
                    )
                    pil_img.save(out_path, format="PNG")
                else:
                    raise ValueError("未收到有效的摄影机画面数据")

                file_size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
                if file_size <= 0:
                    raise IOError("截图文件写入后大小为 0 字节")

                self._send_json({
                    "ok": True,
                    "filename": filename,
                    "output_path": out_path,
                    "width": width,
                    "height": height,
                    "file_size_bytes": file_size,
                })
            except Exception as e:
                self._send_json({
                    "ok": False,
                    "error": f"保存截图到桌面失败: {e}",
                }, 200)
            return

        if path == "/api/copy_screenshot_clipboard":
            image_data_url = str(payload.get("image_data_url") or "")
            width = int(payload.get("width") or 1280)
            height = int(payload.get("height") or 720)
            scene = payload.get("scene")
            t_val = float(payload.get("t") or 0.0)
            try:
                tmp_fd, tmp_png_path = tempfile.mkstemp(prefix="director_stage_clip_", suffix=".png")
                os.close(tmp_fd)
                try:
                    if image_data_url and "," in image_data_url:
                        b64_part = image_data_url.split(",", 1)[1]
                        raw_png = base64.b64decode(b64_part)
                        if len(raw_png) < 64:
                            raise ValueError("截图数据为空或损坏")
                        with open(tmp_png_path, "wb") as f:
                            f.write(raw_png)
                    elif isinstance(scene, dict):
                        pil_img = _stamp_camera_hud_on_pil(
                            render_scene_frame_pil(scene, t_val, width=width, height=height),
                            scene,
                        )
                        pil_img.save(tmp_png_path, format="PNG")
                    else:
                        raise ValueError("未收到有效的摄影机画面数据")

                    file_size = os.path.getsize(tmp_png_path) if os.path.exists(tmp_png_path) else 0
                    if file_size <= 0:
                        raise IOError("截图数据写入为空")

                    # 写入 macOS 系统剪贴板（支持 Cmd+V 直接粘贴到微信/飞书/预览/PS/Keynote 等任意应用）
                    apple_script = f'set the clipboard to (read (POSIX file "{tmp_png_path}") as «class PNGf»)'
                    proc = subprocess.run(["osascript", "-e", apple_script], capture_output=True, text=True, timeout=8)
                    if proc.returncode != 0:
                        raise RuntimeError(proc.stderr.strip() or "osascript 写入剪贴板返回非零状态")

                    self._send_json({
                        "ok": True,
                        "width": width,
                        "height": height,
                        "file_size_bytes": file_size,
                    })
                finally:
                    if os.path.exists(tmp_png_path):
                        try:
                            os.remove(tmp_png_path)
                        except Exception:
                            pass
            except Exception as e:
                self._send_json({
                    "ok": False,
                    "error": f"截图复制到剪贴板失败: {e}",
                }, 200)
            return

        if path == "/api/save_camera_recording":
            video_b64 = str(payload.get("video_base64") or "").strip()
            mime_type = str(payload.get("mime_type") or "video/webm").lower()
            scene_label = str(payload.get("scene_label") or "").strip()
            shot_label = str(payload.get("shot_label") or "").strip()
            save_to_downloads = bool(payload.get("save_to_downloads", True))
            save_to_desktop = bool(payload.get("save_to_desktop", False))
            fps_hint = max(12, min(60, int(payload.get("fps") or 24)))
            active_sc_name, active_sh_name = _resolve_active_scene_shot_labels()
            eff_scene_label = scene_label if (scene_label and scene_label != "场次") else active_sc_name
            eff_shot_label = shot_label if (shot_label and shot_label != "镜头") else active_sh_name

            try:
                if "," in video_b64:
                    video_b64 = video_b64.split(",", 1)[1]
                raw_bytes = base64.b64decode(video_b64) if video_b64 else b""
                if len(raw_bytes) < 128:
                    raise ValueError("录屏数据为空或录制时间过短")

                safe_scene = _sanitize_export_label(eff_scene_label, "未命名场景")
                safe_shot = _sanitize_export_label(eff_shot_label, "第1镜")
                ms_part = int(time.time() * 1000) % 1000
                ts_str = f"{time.strftime('%Y%m%d_%H%M%S')}_{ms_part:03d}"
                base_stem = f"录屏_{safe_scene}_{safe_shot}_{ts_str}"

                ffmpeg_exe = _find_ffmpeg_exe()

                in_ext = ".mp4" if "mp4" in mime_type else ".webm"
                tmp_in_fd, tmp_in_path = tempfile.mkstemp(prefix="mds_camrec_in_", suffix=in_ext)
                os.close(tmp_in_fd)
                tmp_out_fd, tmp_out_path = tempfile.mkstemp(prefix="mds_camrec_out_", suffix=".mp4")
                os.close(tmp_out_fd)

                final_ext = ".mp4"
                converted_mp4 = False
                try:
                    with open(tmp_in_path, "wb") as f:
                        f.write(raw_bytes)

                    if ffmpeg_exe:
                        # 转为通用 H.264 yuv420p MP4（偶数尺寸保护 + faststart），兼容 QuickTime / 剪映 / 微信 / 浏览器直接播放
                        cmd = [
                            ffmpeg_exe,
                            "-y",
                            "-i", tmp_in_path,
                            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                            "-r", str(fps_hint),
                            "-an",
                            "-c:v", "libx264",
                            "-pix_fmt", "yuv420p",
                            "-crf", "18",
                            "-preset", "ultrafast",
                            "-movflags", "+faststart",
                            tmp_out_path,
                        ]
                        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                        if proc.returncode == 0 and os.path.exists(tmp_out_path) and os.path.getsize(tmp_out_path) > 256:
                            converted_mp4 = True

                    if not converted_mp4:
                        final_ext = in_ext

                    src_file = tmp_out_path if converted_mp4 else tmp_in_path
                    filename, export_out_path, desktop_out_path, downloads_out_path, downloads_dir, file_size = _save_export_with_optional_desktop(
                        src_file,
                        base_stem,
                        final_ext,
                        save_to_desktop,
                        scene_label=eff_scene_label,
                        save_to_downloads=save_to_downloads,
                    )
                    self._send_json({
                        "ok": True,
                        "filename": filename,
                        "video_url": f"/exports/{filename}?t={int(time.time() * 1000)}",
                        "export_path": export_out_path,
                        "downloads_dir": downloads_dir,
                        "downloads_path": downloads_out_path,
                        "desktop_path": desktop_out_path,
                        "saved_path": downloads_out_path or desktop_out_path or export_out_path,
                        "converted_mp4": converted_mp4,
                        "file_size_bytes": file_size,
                    })
                finally:
                    for p_tmp in (tmp_in_path, tmp_out_path):
                        if p_tmp and os.path.exists(p_tmp):
                            try:
                                os.remove(p_tmp)
                            except Exception:
                                pass
            except Exception as e:
                self._send_json({
                    "ok": False,
                    "error": f"保存摄影机录屏失败: {e}",
                }, 200)
            return

        if path == "/api/validate_speed_policy":
            scene = payload.get("scene") if isinstance(payload.get("scene"), dict) else LIVE_STATE.get("scene")
            try:
                from motion_speed_policy import validate_scene_speed_policy

                report = validate_scene_speed_policy(scene or {})
                self._send_json({"ok": True, **report})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 200)
            return

        if path == "/api/apply_speed_policy_fix":
            scene = copy.deepcopy(payload.get("scene") if isinstance(payload.get("scene"), dict) else (LIVE_STATE.get("scene") or {}))
            obj_id = str(payload.get("obj_id") or "").strip()
            mode = str(payload.get("mode") or "auto_curve_or_extend").strip()
            reason = payload.get("exception_reason")
            custom_max = payload.get("custom_max_mps")
            try:
                from motion_speed_policy import apply_speed_limit_fix_to_track, validate_scene_speed_policy

                fixed_res = None
                new_objs = []
                for o in (scene.get("objects") or []):
                    if isinstance(o, dict) and str(o.get("id") or "") == obj_id:
                        fixed_res = apply_speed_limit_fix_to_track(
                            o,
                            scene=scene,
                            mode=mode,
                            exception_reason=reason,
                            custom_max_mps=float(custom_max) if custom_max is not None else None,
                        )
                        new_objs.append(fixed_res["obj"])
                    else:
                        new_objs.append(o)
                if fixed_res is None:
                    self._send_json({"ok": False, "error": f"未找到对象 {obj_id}"}, 200)
                    return
                scene["objects"] = new_objs
                report = validate_scene_speed_policy(scene)
                self._send_json({
                    "ok": True,
                    "changed": bool(fixed_res.get("changed")),
                    "action": fixed_res.get("action"),
                    "obj": fixed_res.get("obj"),
                    "obj_report": fixed_res.get("report"),
                    "scene": scene,
                    "speed_policy": report,
                })
            except Exception as e:
                self._send_json({"ok": False, "error": str(e), "traceback": traceback.format_exc()}, 200)
            return

        if path == "/api/continuity/check":
            scene = payload.get("scene") if isinstance(payload.get("scene"), dict) else (LIVE_STATE.get("scene") or {})
            try:
                from motion_continuity import check_scene_continuity_diagnostics
                diag = check_scene_continuity_diagnostics(scene)
                cam_conflicts = []
                cc_cfg = scene.get("camera_speed_curve")
                if isinstance(cc_cfg, dict) and cc_cfg.get("enabled") and len(scene.get("camera_keyframes") or []) >= 2:
                    from director_engine import _get_solved_camera_speed_bundle
                    bnd = _get_solved_camera_speed_bundle(scene)
                    for ch_name in ("pos", "rot", "zoom"):
                        for c_item in (bnd.get(ch_name, {}).get("conflicts") or []):
                            cam_conflicts.append({"obj_id": "camera", "obj_name": f"摄影机 ({ch_name})", **c_item})
                all_c = cam_conflicts + list(diag.get("conflicts") or [])
                self._send_json({
                    "ok": len(all_c) == 0,
                    "conflicts": all_c,
                    "warnings": list(diag.get("warnings") or []),
                })
            except Exception as e:
                self._send_json({"ok": False, "error": str(e), "traceback": traceback.format_exc()}, 200)
            return

        if path == "/api/continuity/apply":
            scene = copy.deepcopy(payload.get("scene") if isinstance(payload.get("scene"), dict) else (LIVE_STATE.get("scene") or {}))
            obj_id = payload.get("obj_id")
            mode = str(payload.get("mode") or "smooth_variable").strip()
            apply_all = bool(payload.get("apply_all", False))
            try:
                from motion_continuity import apply_continuity_protection_to_track, check_scene_continuity_diagnostics
                changed_ids = []
                if apply_all or obj_id == "camera":
                    if isinstance(scene.get("camera_speed_curve"), dict) or len(scene.get("camera_keyframes") or []) >= 2:
                        sc_cc = copy.deepcopy(scene.get("camera_speed_curve") or {})
                        sc_cc["enabled"] = (mode != "manual")
                        sc_cc["mode"] = mode
                        if mode == "constant_speed":
                            sc_cc["preset"] = "constant"
                        elif mode == "manual":
                            sc_cc["preset"] = "manual"
                        elif str(sc_cc.get("preset") or "") in ("constant", "manual", ""):
                            sc_cc["preset"] = "custom" if sc_cc.get("channels", {}).get("pos", {}).get("points") else "continuous_push"
                        scene["camera_speed_curve"] = sc_cc
                        changed_ids.append("camera")
                new_objs = []
                for o in (scene.get("objects") or []):
                    if isinstance(o, dict) and o.get("keyframes") and len(o.get("keyframes") or []) >= 2:
                        if apply_all or (obj_id and str(o.get("id") or "") == str(obj_id)):
                            new_objs.append(apply_continuity_protection_to_track(o, scene=scene, mode=mode))
                            changed_ids.append(str(o.get("id")))
                            continue
                    new_objs.append(o)
                scene["objects"] = new_objs
                diag = check_scene_continuity_diagnostics(scene)
                self._send_json({
                    "ok": True,
                    "changed_ids": changed_ids,
                    "scene": scene,
                    "continuity": diag,
                })
            except Exception as e:
                self._send_json({"ok": False, "error": str(e), "traceback": traceback.format_exc()}, 200)
            return

        if path == "/api/image_to_pose":
            img_b64 = payload.get("image", "")
            fname = payload.get("filename", "")
            try:
                res = analyze_image_to_ik_targets(img_b64, fname)
                self._send_json({"ok": True, **res})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 200)
            return

        self._send_json({"ok": False, "error": f"Unknown endpoint: {path}"}, 404)


def run_server(port: int = 8765):
    ThreadingHTTPServer.allow_reuse_address = True
    ThreadingHTTPServer.daemon_threads = True
    server = ThreadingHTTPServer(("0.0.0.0", port), DirectorHandler)
    print(f"🎬 Magic Director Stage Server running on LAN & Localhost (port {port})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    import sys
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    run_server(p)
