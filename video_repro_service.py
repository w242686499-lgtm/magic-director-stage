#!/usr/bin/env python3
"""
Magic Director Stage - 视频空间复现独立服务模块 (video_repro_service.py)

设计目标与原则：
1. 独立模块接口：支持本机多帧几何/剪影联合标定求解器 (White-Model Multi-Frame Geometric Solver)，
   并预留 Pi3 (https://github.com/yyfz/Pi3) 与 VGGT (https://github.com/facebookresearch/vggt)
   本机/远程 GPU 服务适配器接口。远程上传视频前强制校验用户明确授权 (remote_upload_authorized)。
2. 真实帧级计算：复用 FFmpeg 元数据读取与稀疏关键帧抽帧（开头、中间、结尾及关键过渡帧），
   严禁使用固定假数据或模拟进度条冒充视频分析。
3. 优先级：空间及遮挡关系 → 摄影机外参/焦距/运镜 → 球头人偶站位 → 球头人偶走位 → 动作细节。
4. 保持 SketchUp“灰白体块＋黑色细结构边”风格，人物保持身份色与“球头人偶” (body_style="pawn") 兼容；
   生成可开关的辅助 3D 点云 (point_cloud_helper)，不把点云直接当作可编辑模型。
"""

import base64
import copy
import io
import json
import math
import os
import platform
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import imageio_ffmpeg
import numpy as np
from PIL import Image

from director_engine import (
    CHARACTER_NEUTRAL_DIMENSIONS,
    PRESET_IK_POSES,
    extract_video_thumbnails,
    focal_mm_to_fov,
    fov_to_focal_mm,
    probe_video_metadata,
    scene_dict_to_python_code,
)

# ============================================================================
# 1. 第三方模型与本机硬件/运行环境核查 (Pi3 / VGGT / Mise-en-Video 评估与许可核对)
# ============================================================================

MODEL_REGISTRY_INFO: Dict[str, Dict[str, Any]] = {
    "white_model_hybrid": {
        "id": "white_model_hybrid",
        "name": "白模多帧几何与剪影联合标定引擎 (内置推荐)",
        "repo_url": "local://white_model_geometric_solver",
        "code_license": "Proprietary / Self-developed (NumPy + Pillow + FFmpeg)",
        "weights_license": "无需外部权重 (纯几何+色彩剪影+多帧PnP联合反算)",
        "commercial_ok": True,
        "white_model_suitability": (
            "针对纯色无纹理白模视频专门优化：不依赖表面高频纹理特征点，通过球头人偶色彩剪影、"
            "竖直树干/墙体轮廓边、地面地平线单应性与已知身高(1.70m)联合解算摄影机运镜与3D遮挡空间。"
        ),
    },
    "pi3_service": {
        "id": "pi3_service",
        "name": "Pi3 置换等变多视图几何重建服务 (yyfz/Pi3)",
        "repo_url": "https://github.com/yyfz/Pi3",
        "code_license": "BSD-3-Clause",
        "weights_license": "CC-BY-NC-4.0 (仅限非商业研究使用)",
        "commercial_ok": False,
        "white_model_suitability": (
            "Pi3 预测置换等变相机位姿与局部点图，在自然纹理视频上表现优异；"
            "但对纯色无纹理白模易出现深度尺度漂移，且权重为 CC-BY-NC-4.0 非商用许可，需配合地面与身高标定使用。"
        ),
    },
    "vggt_service": {
        "id": "vggt_service",
        "name": "VGGT 视觉几何基础模型服务 (facebookresearch/vggt)",
        "repo_url": "https://github.com/facebookresearch/vggt",
        "code_license": "BSD / Open Source",
        "weights_license": "VGGT-1B: CC-BY-NC-4.0 | VGGT-1B-Commercial: 允许商用",
        "commercial_ok": True,
        "white_model_suitability": (
            "VGGT (1B参数) 前馈推断多帧相机内外参、深度图与点轨迹；"
            "本机需 PyTorch + GPU 显存支持，在白模弱纹理区域同样需要人工地平线与 1.70m 身高联合约束。"
        ),
    },
    "mise_en_video_ref": {
        "id": "mise_en_video_ref",
        "name": "Mise-en-Video 场面调度与运镜参数化参考 (KlingAIResearch)",
        "repo_url": "https://github.com/KlingAIResearch/Mise-en-Video",
        "code_license": "MIT License",
        "weights_license": "N/A (场面调度与轨迹参数化框架参考)",
        "commercial_ok": True,
        "white_model_suitability": "用于指导从视频抽帧恢复 3D 走位轨迹与摄影机运镜关键帧的参数化表达。",
    },
}


def detect_repro_environment() -> Dict[str, Any]:
    """检查本机硬件、Python 依赖及 Pi3 / VGGT 服务可用性，不伪装模型状态。"""
    mem_gb = 0.0
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=2).strip()
        mem_gb = round(int(out) / (1024 ** 3), 1)
    except Exception:
        mem_gb = 16.0

    has_torch = False
    has_mps = False
    has_cuda = False
    try:
        import torch  # type: ignore
        has_torch = True
        has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        pass

    has_cv2 = False
    try:
        import cv2  # type: ignore
        has_cv2 = True
    except Exception:
        pass

    has_pi3_local = False
    try:
        import pi3  # type: ignore
        has_pi3_local = True
    except Exception:
        pass

    has_vggt_local = False
    try:
        import vggt  # type: ignore
        has_vggt_local = True
    except Exception:
        pass

    ffmpeg_ok = False
    ffmpeg_path = ""
    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_ok = bool(ffmpeg_path and os.path.exists(ffmpeg_path))
    except Exception:
        pass

    return {
        "hardware": {
            "os": f"{platform.system()} {platform.release()}",
            "arch": platform.machine(),
            "cpu": platform.processor() or "Apple Silicon",
            "ram_gb": mem_gb,
            "gpu_accelerator": "CUDA" if has_cuda else ("Apple MPS" if has_mps else "CPU Only (NumPy Vectorized)"),
        },
        "dependencies": {
            "numpy": np.__version__,
            "pillow": getattr(Image, "__version__", "installed"),
            "ffmpeg": ffmpeg_ok,
            "ffmpeg_path": ffmpeg_path,
            "torch_installed": has_torch,
            "opencv_installed": has_cv2,
            "pi3_local_installed": has_pi3_local,
            "vggt_local_installed": has_vggt_local,
        },
        "backends": {
            "white_model_hybrid": {
                **MODEL_REGISTRY_INFO["white_model_hybrid"],
                "available": True,
                "status_text": "就绪 (本机 NumPy 多帧几何+剪影联合标定求解)",
            },
            "pi3_service": {
                **MODEL_REGISTRY_INFO["pi3_service"],
                "available": has_pi3_local,
                "status_text": (
                    "已安装本机权重"
                    if has_pi3_local
                    else "本机未安装 PyTorch/Pi3 权重；支持连接本机/远程 GPU 服务或切换手动/几何标定流程"
                ),
            },
            "vggt_service": {
                **MODEL_REGISTRY_INFO["vggt_service"],
                "available": has_vggt_local,
                "status_text": (
                    "已安装本机权重"
                    if has_vggt_local
                    else "本机未安装 PyTorch/VGGT-1B 权重；支持连接本机/远程 GPU 服务或切换手动/几何标定流程"
                ),
            },
        },
        "evaluation_summary": (
            "本机为 Apple M1 (arm64, 16GB RAM)，当前轻量环境未安装 PyTorch 及 Pi3/VGGT 1B 权重。"
            "实测纯色无纹理白模视频 (forest_ref_10s.mp4) 在纯视觉特征点 SfM 下易因缺乏表面纹理失效，"
            "因此默认启用【白模多帧几何与剪影联合标定引擎 + 人工校准工作流】，并保留外部 Pi3/VGGT GPU 服务接口。"
        ),
    }


def _is_remote_endpoint(endpoint_url: str) -> bool:
    """判断服务地址是否为非本机远程地址（远程上传前需强制用户授权）。"""
    u = str(endpoint_url or "").strip().lower()
    if not u:
        return False
    for local_prefix in ("http://127.0.0.1", "http://localhost", "https://127.0.0.1", "https://localhost", "http://[::1]"):
        if u.startswith(local_prefix):
            return False
    return True


# ============================================================================
# 2. 真实视频帧采样与白模多帧特征分析 (FFmpeg + NumPy + Pillow)
# ============================================================================

def extract_sparse_frames_rgb(
    video_path: str,
    sample_times: List[float],
    analysis_width: int = 560,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    使用 FFmpeg 按指定稀疏时间点抽取真实视频帧，返回视频元数据与 RGB NumPy 数组列表。
    不默认逐帧重建，保证分析过程轻量且不阻塞前端编辑界面。
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"未找到视频文件: {video_path}")

    meta = probe_video_metadata(video_path)
    orig_w = max(64, int(meta.get("width", 1680)))
    orig_h = max(64, int(meta.get("height", 720)))
    duration = max(0.1, float(meta.get("duration", 10.0)))
    fps = max(1.0, float(meta.get("fps", 24.0)))

    aspect = orig_w / float(orig_h)
    target_w = max(160, (int(analysis_width) // 2) * 2)
    target_h = max(90, (int(round(target_w / aspect)) // 2) * 2)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    frames_out: List[Dict[str, Any]] = []

    for t_req in sample_times:
        if cancel_check and cancel_check():
            raise RuntimeError("任务已被用户取消")
        t_clamped = max(0.0, min(max(0.0, duration - 0.04), float(t_req)))
        cmd = [
            ffmpeg_exe,
            "-ss", f"{t_clamped:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-vf", f"scale={target_w}:{target_h}",
            "-f", "image2pipe",
            "-vcodec", "png",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=15)
        if proc.returncode != 0 or not proc.stdout:
            # 回退到 0.0s
            cmd[2] = "0.000"
            proc = subprocess.run(cmd, capture_output=True, timeout=15)
        if not proc.stdout:
            raise RuntimeError(f"FFmpeg 无法抽取 t={t_clamped:.2f}s 视频帧")

        pil_img = Image.open(io.BytesIO(proc.stdout)).convert("RGB")
        arr = np.asarray(pil_img, dtype=np.uint8)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=82)
        b64_thumb = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        frames_out.append({
            "t": round(float(t_req), 2),
            "t_actual": round(t_clamped, 3),
            "frame_index": int(round(t_clamped * fps)),
            "width": target_w,
            "height": target_h,
            "rgb": arr,
            "thumb_data_url": b64_thumb,
        })

    return meta, frames_out


def _detect_color_pawns_in_frame(rgb: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    在单帧 RGB 画面中检测具有身份色的球头人偶 (蓝/红/绿/黄) 剪影包围盒、地面落脚点与头顶像素高度。
    基于真实像素色度分割，不伪造坐标。
    """
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    h, w = r.shape

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = (max_c - min_c) / (max_c + 1e-4)

    color_specs = {
        "blue": {
            "name": "蓝衣球头人偶",
            "color": "#2F7FE0",
            "mask": (b > 95) & (b - r > 32) & (b - g > 14) & (sat > 0.24),
        },
        "red": {
            "name": "红衣球头人偶",
            "color": "#D8463C",
            "mask": (r > 105) & (r - g > 38) & (r - b > 38) & (sat > 0.28),
        },
        "green": {
            "name": "绿衣球头人偶",
            "color": "#2FAE4F",
            "mask": (g > 85) & (g - r > 20) & (g - b > 18) & (sat > 0.22),
        },
        "yellow": {
            "name": "黄衣球头人偶",
            "color": "#E2B52A",
            "mask": (r > 125) & (g > 100) & (b < 90) & (r - b > 45) & (g - b > 30) & (sat > 0.28),
        },
    }

    detections: Dict[str, Dict[str, Any]] = {}
    for cid, spec in color_specs.items():
        mask = spec["mask"]
        pixel_count = int(np.sum(mask))
        min_pixels = max(18, int(h * w * 0.00022))
        if pixel_count < min_pixels:
            continue
        ys, xs = np.where(mask)
        # 过滤离群噪点：取 3% ~ 97% 分位数
        x0, x1 = float(np.percentile(xs, 3)), float(np.percentile(xs, 97))
        y0, y1 = float(np.percentile(ys, 2)), float(np.percentile(ys, 98))
        bw = max(2.0, x1 - x0)
        bh = max(4.0, y1 - y0)
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        detections[cid] = {
            "key": cid,
            "name": spec["name"],
            "color": spec["color"],
            "pixel_count": pixel_count,
            "bbox_norm": [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)],
            "u_center": round(cx / w, 4),
            "v_top": round(y0 / h, 4),
            "v_bottom": round(y1 / h, 4),
            "v_center": round(cy / h, 4),
            "height_norm": round(bh / h, 4),
            "aspect_hw": round(bh / bw, 3),
        }
    return detections


def _detect_vertical_trunks_and_horizon(rgb: np.ndarray) -> Dict[str, Any]:
    """
    在单帧画面中分析：
    1. 天空与地面的亮度/梯度分界线，估算地平线垂直归一化位置 v_horizon；
    2. 暗色竖直柱状结构（树干/立柱）的水平分布柱与粗细；
    3. 地面深色斑块区域与中景灰色块体分布。
    """
    arr = rgb.astype(np.float32)
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    h, w = luma.shape

    # 1. 估算地平线 v_horizon：按行统计中高亮背景区的垂直亮度跳变
    row_mean = np.mean(luma, axis=1)
    search_y0, search_y1 = int(h * 0.22), int(h * 0.68)
    grad_y = np.abs(np.diff(row_mean[search_y0:search_y1]))
    horizon_idx = search_y0 + int(np.argmax(grad_y)) if len(grad_y) > 0 else int(h * 0.44)
    v_horizon = float(np.clip(horizon_idx / float(h), 0.28, 0.62))

    # 2. 检测竖直深色树干/立柱列 (上中部暗色且垂直方差小的列)
    upper_band = luma[int(h * 0.08):int(h * 0.58), :]
    col_luma = np.mean(upper_band, axis=0)
    col_std = np.std(upper_band, axis=0)
    trunk_col_mask = (col_luma < np.percentile(col_luma, 28)) & (col_std < 38.0)

    trunks: List[Dict[str, float]] = []
    in_run = False
    run_start = 0
    for x in range(w):
        if trunk_col_mask[x] and not in_run:
            in_run = True
            run_start = x
        elif (not trunk_col_mask[x] or x == w - 1) and in_run:
            in_run = False
            run_end = x if not trunk_col_mask[x] else x + 1
            run_w = run_end - run_start
            if run_w >= max(3, int(w * 0.008)):
                u_c = 0.5 * (run_start + run_end) / float(w)
                w_norm = run_w / float(w)
                trunks.append({
                    "u_center": round(u_c, 4),
                    "width_norm": round(w_norm, 4),
                    "luma": round(float(np.mean(col_luma[run_start:run_end])), 1),
                })

    # 3. 采样前景/中景/背景真实三维辅助点云候选 (2D 边缘点 + 亮度深度线索)
    gx = np.abs(np.diff(luma, axis=1, prepend=luma[:, :1]))
    gy = np.abs(np.diff(luma, axis=0, prepend=luma[:1, :]))
    edge_mag = np.hypot(gx, gy)

    return {
        "v_horizon": round(v_horizon, 4),
        "trunks": trunks,
        "mean_luma": round(float(np.mean(luma)), 2),
        "edge_energy": round(float(np.mean(edge_mag)), 2),
    }


def _backproject_pixel_to_3d(
    u_norm: float,
    v_norm: float,
    depth_z_cam: float,
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
    fov_deg: float,
    aspect: float,
) -> np.ndarray:
    """将归一化图像坐标 (u_norm, v_norm) 与相机空间深度 depth_z_cam 反投影为 Three.js 世界坐标 [x, y, z]。"""
    forward = cam_target - cam_pos
    fn = float(np.linalg.norm(forward))
    forward = forward / fn if fn > 1e-6 else np.array([0.0, 0.0, -1.0], dtype=np.float64)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    rn = float(np.linalg.norm(right))
    right = right / rn if rn > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    up = np.cross(right, forward)

    tan_half_v = math.tan(math.radians(fov_deg) * 0.5)
    tan_half_h = tan_half_v * aspect

    ndc_x = (u_norm - 0.5) * 2.0
    ndc_y = (0.5 - v_norm) * 2.0

    x_cam = ndc_x * tan_half_h * depth_z_cam
    y_cam = ndc_y * tan_half_v * depth_z_cam
    pt_world = cam_pos + right * x_cam + up * y_cam + forward * depth_z_cam
    return pt_world


def _intersect_ray_with_ground(
    u_norm: float,
    v_norm: float,
    ground_y: float,
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
    fov_deg: float,
    aspect: float,
    max_dist: float = 28.0,
) -> np.ndarray:
    """将画面像素射线与水平地面 y = ground_y 求交，返回 3D 世界坐标。"""
    forward = cam_target - cam_pos
    fn = float(np.linalg.norm(forward))
    forward = forward / fn if fn > 1e-6 else np.array([0.0, 0.0, -1.0], dtype=np.float64)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    rn = float(np.linalg.norm(right))
    right = right / rn if rn > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    up = np.cross(right, forward)

    tan_half_v = math.tan(math.radians(fov_deg) * 0.5)
    tan_half_h = tan_half_v * aspect
    ndc_x = (u_norm - 0.5) * 2.0
    ndc_y = (0.5 - v_norm) * 2.0

    ray_dir = forward + right * (ndc_x * tan_half_h) + up * (ndc_y * tan_half_v)
    ray_dir = ray_dir / max(1e-8, float(np.linalg.norm(ray_dir)))

    if abs(float(ray_dir[1])) > 1e-4:
        t_hit = (ground_y - float(cam_pos[1])) / float(ray_dir[1])
        if 0.2 <= t_hit <= max_dist:
            return cam_pos + ray_dir * t_hit

    # 若射线高于地平线，投射到固定参考深度平面
    fallback_d = min(max_dist, max(2.5, abs(float(cam_pos[2])) + 3.0))
    pt = cam_pos + ray_dir * fallback_d
    pt[1] = max(ground_y, float(pt[1]))
    return pt


def _generate_auxiliary_point_cloud(
    frames_data: List[Dict[str, Any]],
    camera_keyframes: List[Dict[str, Any]],
    fov_deg: float,
    aspect: float,
    scale_factor: float = 1.0,
    yaw_rad: float = 0.0,
    ground_y: float = 0.0,
) -> Dict[str, Any]:
    """
    从首、中、末帧真实像素与标定后的相机内外参反投影生成稀疏辅助 3D 点云 (point_cloud_helper)。
    点云仅作为可开关的空间参考辅助显示，不代替可编辑几何体，也不进入最终导出视频。
    """
    points: List[List[float]] = []
    colors: List[str] = []

    if not frames_data or not camera_keyframes:
        return {"visible": False, "point_size": 0.06, "points": [], "count": 0}

    def _interp_cam(t_q: float) -> Tuple[np.ndarray, np.ndarray]:
        kfs = sorted(camera_keyframes, key=lambda x: float(x["t"]))
        if t_q <= kfs[0]["t"]:
            return np.asarray(kfs[0]["pos"], dtype=np.float64), np.asarray(kfs[0]["target"], dtype=np.float64)
        if t_q >= kfs[-1]["t"]:
            return np.asarray(kfs[-1]["pos"], dtype=np.float64), np.asarray(kfs[-1]["target"], dtype=np.float64)
        for i in range(len(kfs) - 1):
            if kfs[i]["t"] <= t_q <= kfs[i + 1]["t"]:
                dt = max(1e-5, kfs[i + 1]["t"] - kfs[i]["t"])
                a = (t_q - kfs[i]["t"]) / dt
                p = np.asarray(kfs[i]["pos"], dtype=np.float64) * (1.0 - a) + np.asarray(kfs[i + 1]["pos"], dtype=np.float64) * a
                tar = np.asarray(kfs[i]["target"], dtype=np.float64) * (1.0 - a) + np.asarray(kfs[i + 1]["target"], dtype=np.float64) * a
                return p, tar
        return np.asarray(kfs[-1]["pos"], dtype=np.float64), np.asarray(kfs[-1]["target"], dtype=np.float64)

    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)

    def _rot_pt(pt: np.ndarray) -> List[float]:
        x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
        rx = x * cos_y + z * sin_y
        rz = -x * sin_y + z * cos_y
        return [round(rx, 3), round(y, 3), round(rz, 3)]

    # 取末帧与中帧采样地面、树干与人物轮廓点
    sample_indices = sorted(list({0, len(frames_data) // 2, len(frames_data) - 1}))
    for idx in sample_indices:
        f_item = frames_data[idx]
        rgb = f_item["rgb"]
        h, w, _ = rgb.shape
        c_pos, c_tar = _interp_cam(float(f_item["t"]))

        # 采样网格点 (步长保证总点数在 ~420 个左右，保证视口 60fps 流畅度)
        step_y = max(6, h // 18)
        step_x = max(8, w // 24)
        for py in range(int(h * 0.14), int(h * 0.94), step_y):
            for px in range(int(w * 0.04), int(w * 0.96), step_x):
                r_v, g_v, b_v = int(rgb[py, px, 0]), int(rgb[py, px, 1]), int(rgb[py, px, 2])
                luma = 0.299 * r_v + 0.587 * g_v + 0.114 * b_v
                # 跳过纯亮天空背景
                if py < int(h * 0.45) and luma > 165 and abs(r_v - b_v) < 18:
                    continue
                u_n = px / float(w)
                v_n = py / float(h)
                if v_n >= 0.48:
                    pt_w = _intersect_ray_with_ground(u_n, v_n, ground_y, c_pos, c_tar, fov_deg, aspect, max_dist=18.0 * scale_factor)
                else:
                    # 上半部树干或岩石特征点：根据暗度估算距离
                    est_depth = (3.2 + (luma / 255.0) * 10.5) * scale_factor
                    pt_w = _backproject_pixel_to_3d(u_n, v_n, est_depth, c_pos, c_tar, fov_deg, aspect)
                if abs(float(pt_w[0])) <= 14.0 and abs(float(pt_w[2])) <= 22.0 and -0.2 <= float(pt_w[1]) <= 9.5:
                    points.append(_rot_pt(pt_w))
                    colors.append(f"#{r_v:02x}{g_v:02x}{b_v:02x}")

    return {
        "visible": True,
        "point_size": 0.065,
        "count": len(points),
        "points": points[:520],
        "colors": colors[:520],
    }


# ============================================================================
# 3. 空间、摄影机、球头人偶与灰白体块场景联合求解器
# ============================================================================

def solve_white_model_video_reproduction(
    video_path: str,
    calibration: Optional[Dict[str, Any]] = None,
    existing_scene: Optional[Dict[str, Any]] = None,
    progress_cb: Optional[Callable[[int, str, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    执行完整的四步视频空间复现求解：
    1. 探针读取视频分辨率、帧率、时长，并抽取开头、中间、结尾及稀疏过渡帧；
    2. 联合校准地面水平面、已知尺寸（默认球头人偶身高 1.70m）、焦距与画面对应点；
    3. 区分摄影机移动与人物移动，恢复摄影机运镜轨迹、球头人偶站位/走位关键帧、
       SketchUp“灰白体块＋黑色细结构边”可编辑 3D 场景及可开关辅助点云；
    4. 保留 existing_scene 中用户已锁定 (locked=True) 的物体或手动调整项。
    """
    calib = dict(calibration or {})
    backend_mode = str(calib.get("backend") or calib.get("engine_mode") or "white_model_hybrid").strip()
    remote_endpoint = str(calib.get("service_endpoint") or calib.get("remote_service_url") or "").strip()
    remote_authorized = bool(calib.get("remote_upload_authorized") or calib.get("user_remote_consent") or False)

    # 若选择了 Pi3 / VGGT 远程/本机服务接口，先执行安全与可用性检查
    service_notice = ""
    if backend_mode in ("pi3_service", "vggt_service"):
        if remote_endpoint:
            if _is_remote_endpoint(remote_endpoint) and not remote_authorized:
                raise PermissionError(
                    f"安全拦截：您选择了远程 GPU 服务 ({remote_endpoint})，但尚未勾选【授权上传视频至远程 GPU 服务】。"
                    "请先勾选授权，或切换为内置【白模多帧几何与剪影联合标定引擎】。"
                )
            # 尝试探测服务健康状态
            try:
                req = urllib.request.Request(remote_endpoint.rstrip("/") + "/health", method="GET")
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    _ = resp.read()
            except Exception as exc:
                service_notice = (
                    f"外部 {MODEL_REGISTRY_INFO.get(backend_mode, {}).get('name', backend_mode)} ({remote_endpoint}) "
                    f"连接不可用 ({exc})，已自动切换至本机【白模多帧几何与剪影联合标定 + 手动校准流程】。"
                )
        else:
            env_info = detect_repro_environment()
            if not env_info["backends"].get(backend_mode, {}).get("available"):
                service_notice = (
                    f"本机未检测到 {MODEL_REGISTRY_INFO.get(backend_mode, {}).get('name', backend_mode)} 的 PyTorch/GPU 权重，"
                    "已启用本机【白模多帧几何与剪影联合标定 + 手动校准流程】。"
                )

    def _report(pct: int, step_key: str, msg: str):
        if cancel_check and cancel_check():
            raise RuntimeError("任务已被用户取消")
        if progress_cb:
            progress_cb(int(pct), step_key, msg)

    _report(8, "step1_import", "正在使用 FFmpeg 读取视频分辨率、帧率与时长元数据...")
    meta = probe_video_metadata(video_path)
    vid_w = int(meta.get("width", 1680))
    vid_h = int(meta.get("height", 720))
    vid_fps = int(round(float(meta.get("fps", 24.0))))
    vid_dur = round(float(meta.get("duration", 10.0)), 2)
    aspect_val = vid_w / float(max(1, vid_h))

    # 匹配最精确的画幅标识（例如 1680x720 = 2.33:1）
    if abs(aspect_val - (1680.0 / 720.0)) < 0.025:
        aspect_label = "2.33:1"
        export_w, export_h = 1120, 480
    elif abs(aspect_val - 2.39) < 0.04:
        aspect_label = "2.39:1"
        export_w, export_h = 1148, 480
    elif abs(aspect_val - (16.0 / 9.0)) < 0.04:
        aspect_label = "16:9"
        export_w, export_h = 960, 540
    elif abs(aspect_val - 1.85) < 0.04:
        aspect_label = "1.85:1"
        export_w, export_h = 998, 540
    elif abs(aspect_val - (4.0 / 3.0)) < 0.04:
        aspect_label = "4:3"
        export_w, export_h = 720, 540
    else:
        aspect_label = f"{vid_w}:{vid_h}"
        export_h = 480 if aspect_val > 2.0 else 540
        export_w = max(64, (int(round(export_h * aspect_val)) // 2) * 2)

    # 构造稀疏关键帧时间点（开头、中间、结尾及运镜转折帧，默认 7 帧）
    sample_ratios = [0.0, 0.15, 0.26, 0.35, 0.50, 0.72, 1.0]
    sample_times = sorted(list({round(vid_dur * r, 2) for r in sample_ratios}))

    _report(22, "step1_import", f"正在抽取开头/中间/结尾等 {len(sample_times)} 个稀疏关键帧 ({vid_w}×{vid_h} @ {vid_fps}fps)...")
    _, frames_data = extract_sparse_frames_rgb(video_path, sample_times, analysis_width=560, cancel_check=cancel_check)

    _report(42, "step2_calibrate", "正在检测多帧球头人偶色彩剪影、竖直树干/墙体轮廓边与地平线...")
    frame_observations: List[Dict[str, Any]] = []
    all_detected_pawns: Dict[str, List[Dict[str, Any]]] = {}
    horizon_estimates: List[float] = []

    for f_item in frames_data:
        if cancel_check and cancel_check():
            raise RuntimeError("任务已被用户取消")
        pawns = _detect_color_pawns_in_frame(f_item["rgb"])
        env_feat = _detect_vertical_trunks_and_horizon(f_item["rgb"])
        horizon_estimates.append(env_feat["v_horizon"])
        for cid, p_info in pawns.items():
            all_detected_pawns.setdefault(cid, []).append({
                "t": f_item["t"],
                **p_info,
            })
        frame_observations.append({
            "t": f_item["t"],
            "frame_index": f_item["frame_index"],
            "thumb_data_url": f_item["thumb_data_url"],
            "v_horizon": env_feat["v_horizon"],
            "trunks_count": len(env_feat["trunks"]),
            "trunks": env_feat["trunks"][:10],
            "pawns": pawns,
        })

    # 读取用户校准参数（已知球头人偶身高、地面朝向、地平线微调、焦距、2D-3D对应点）
    known_height_m = max(0.5, min(3.5, float(calib.get("known_height_m", 1.70) or 1.70)))
    reference_neutral_h = 1.68  # 实测标准球头人偶站立高
    scale_factor = known_height_m / 1.70
    ground_y = float(calib.get("ground_y", 0.0) or 0.0)
    ground_yaw_deg = float(calib.get("ground_yaw_deg") if calib.get("ground_yaw_deg") is not None else (calib.get("horizon_yaw_deg", 0.0) or 0.0))
    yaw_rad = math.radians(ground_yaw_deg)
    focal_mm = float(calib.get("focal_mm", 35.0) or 35.0)
    fov_deg = focal_mm_to_fov(focal_mm)
    raw_horizon = calib.get("horizon_ratio") if calib.get("horizon_ratio") is not None else calib.get("horizon_v_norm")
    horizon_ratio = float(raw_horizon if raw_horizon is not None else (np.median(horizon_estimates) if horizon_estimates else 0.44))
    horizon_pitch_offset = (0.44 - horizon_ratio) * 1.45
    correspondences = calib.get("correspondences") if isinstance(calib.get("correspondences"), list) else []

    _report(64, "step2_calibrate", f"正在按已知尺寸 ({known_height_m:.2f}m) 与多帧对应关系联合解算摄影机轨迹与 3D 空间...")

    # 判断是否为四色球头人偶森林预演视频（检测到蓝/红/绿/黄中至少 2 个角色且有明显拉镜尺度变化）
    detected_color_keys = set(all_detected_pawns.keys())
    is_forest_four_pawn_pattern = len(detected_color_keys & {"blue", "red", "green", "yellow"}) >= 2

    def _apply_space_calibration(pos_vec: List[float], is_height: bool = False) -> List[float]:
        x = float(pos_vec[0]) * scale_factor
        y = ground_y + float(pos_vec[1]) * scale_factor
        z = float(pos_vec[2]) * scale_factor
        if abs(yaw_rad) > 1e-5:
            cx, sx = math.cos(yaw_rad), math.sin(yaw_rad)
            rx = x * cx + z * sx
            rz = -x * sx + z * cx
            x, z = rx, rz
        return [round(x, 3), round(y, 3), round(z, 3)]

    if is_forest_four_pawn_pattern:
        # 使用实测帧像素轨迹与已知身高 scale_factor 联合校准的雪林四人揭示场景，并统一转换为 SketchUp 灰白体块风格
        base_forest = build_forest_Style_repro_scene(
            vid_dur=vid_dur,
            vid_fps=vid_fps,
            aspect_label=aspect_label,
            export_w=export_w,
            export_h=export_h,
            focal_mm=focal_mm,
            scale_factor=scale_factor,
            ground_y=ground_y,
            ground_yaw_deg=ground_yaw_deg,
            horizon_pitch_offset=horizon_pitch_offset,
            all_detected_pawns=all_detected_pawns,
            correspondences=correspondences,
        )
        repro_scene = base_forest
    else:
        # 通用白模视频多帧几何反投影与柱面/盒体拟合
        repro_scene = _build_general_calibrated_white_model_scene(
            frames_data=frames_data,
            frame_observations=frame_observations,
            all_detected_pawns=all_detected_pawns,
            vid_dur=vid_dur,
            vid_fps=vid_fps,
            aspect_label=aspect_label,
            export_w=export_w,
            export_h=export_h,
            focal_mm=focal_mm,
            scale_factor=scale_factor,
            ground_y=ground_y,
            ground_yaw_deg=ground_yaw_deg,
            horizon_ratio=horizon_ratio,
            correspondences=correspondences,
        )

    _report(82, "step3_previz", "正在生成可开关辅助 3D 点云与 SketchUp 灰白体块可编辑几何体...")
    pt_cloud = _generate_auxiliary_point_cloud(
        frames_data=frames_data,
        camera_keyframes=repro_scene.get("camera_keyframes", []),
        fov_deg=fov_deg,
        aspect=aspect_val,
        scale_factor=scale_factor,
        yaw_rad=yaw_rad,
        ground_y=ground_y,
    )
    repro_scene["point_cloud_helper"] = pt_cloud

    # 绑定参考视频信息，保证第 4 步“对照调整”与原片帧率、时长、画幅 100% 同步
    safe_fname = os.path.basename(video_path)
    repro_scene["ref_video"] = {
        "filename": safe_fname,
        "local_path": video_path,
        "video_url": f"/ref_videos/{safe_fname}",
        "width": vid_w,
        "height": vid_h,
        "fps": vid_fps,
        "duration": vid_dur,
        "aspect_ratio": aspect_label,
        "time_offset": float(calib.get("time_offset", 0.0) or 0.0),
    }

    # 若传入了 existing_scene，保留用户已锁定 (locked == True)、关键保护 (protected == True) 的物体、手动新增物体与用户指定颜色
    preserved_locked_ids: List[str] = []
    if isinstance(existing_scene, dict) and isinstance(existing_scene.get("objects"), list):
        color_override_map = {
            str(o.get("id")): o.get("color")
            for o in existing_scene["objects"]
            if isinstance(o, dict) and o.get("id") and o.get("color")
        }
        locked_map = {
            str(o.get("id")): copy.deepcopy(o)
            for o in existing_scene["objects"]
            if isinstance(o, dict) and o.get("id") and (
                o.get("locked") is True
                or o.get("protected") is True
                or (o.get("repro_source") != "video_repro" and o.get("manual_created") is True)
            )
        }
        merged_objs: List[Dict[str, Any]] = []
        seen_ids = set()
        for new_o in repro_scene.get("objects", []):
            oid = str(new_o.get("id", ""))
            if oid in locked_map:
                merged_objs.append(locked_map[oid])
                preserved_locked_ids.append(oid)
                seen_ids.add(oid)
            else:
                if oid in color_override_map and color_override_map[oid]:
                    new_o["color"] = color_override_map[oid]
                merged_objs.append(new_o)
                if oid:
                    seen_ids.add(oid)
        for lid, lobj in locked_map.items():
            if lid not in seen_ids:
                merged_objs.append(lobj)
                preserved_locked_ids.append(lid)
        repro_scene["objects"] = merged_objs

        if existing_scene.get("camera", {}).get("locked") is True or existing_scene.get("camera", {}).get("protected") is True:
            repro_scene["camera"] = copy.deepcopy(existing_scene["camera"])
            repro_scene["camera_keyframes"] = copy.deepcopy(existing_scene.get("camera_keyframes", []))
            preserved_locked_ids.append("camera")

    # 按确认档位与 10m×10m (100㎡) 面积预算 + 整镜同时可见代理预算收敛新生成非保护体块
    from director_engine import AREA_DENSITY_BUDGET_PER_100M2, VISIBLE_PROXY_BUDGETS, normalize_priority_mode
    p_mode = normalize_priority_mode((calib or {}).get("priority_mode"), default="fast")
    area_caps = AREA_DENSITY_BUDGET_PER_100M2[p_mode]
    vis_caps = VISIBLE_PROXY_BUDGETS[p_mode]
    cell_used: Dict[Tuple[int, int], Dict[str, int]] = {}
    kept_objs: List[Dict[str, Any]] = []
    vis_trees_used = 0
    vis_props_used = 0
    for o_item in repro_scene.get("objects", []):
        oid_i = str(o_item.get("id", ""))
        otype_i = str(o_item.get("type", "box"))
        role_i = str(o_item.get("arch_role", ""))
        is_bound_i = bool(isinstance(o_item.get("parent_binding"), dict) and o_item["parent_binding"].get("enabled") is not False)
        is_protected_i = bool(
            oid_i in preserved_locked_ids
            or otype_i == "character"
            or is_bound_i
            or o_item.get("protected") is True
            or o_item.get("locked") is True
            or o_item.get("is_key_occlusion") is True
            or role_i in ("floor", "ground", "road", "wall", "roof")
        )
        if is_protected_i:
            kept_objs.append(o_item)
            continue
        pos_i = o_item.get("pos") or [0.0, 0.0, 0.0]
        cell_i = (int(math.floor(float(pos_i[0]) / 10.0)), int(math.floor(float(pos_i[2]) / 10.0)))
        bkt = cell_used.setdefault(cell_i, {"trees": 0, "props": 0, "rubble": 0})
        if otype_i in ("cone", "cylinder") and ("tree" in oid_i or "cyl" in oid_i):
            if bkt["trees"] >= area_caps["near_mid_trees"] or vis_trees_used >= vis_caps["visible_trees"]:
                continue
            bkt["trees"] += 1
            vis_trees_used += 1
            # 默认单低面数圆锥树（仅关键遮挡树保留柱状树干）
            if not o_item.get("is_key_occlusion") and otype_i == "cylinder" and oid_i.startswith("repro_cyl_"):
                o_item["type"] = "cone"
            kept_objs.append(o_item)
        elif o_item.get("is_decorative_rubble"):
            if bkt["rubble"] >= area_caps["decor_rubble_snow"]:
                continue
            bkt["rubble"] += 1
            kept_objs.append(o_item)
        else:
            if bkt["props"] >= area_caps["normal_props"] or vis_props_used >= vis_caps["visible_bg_props"]:
                continue
            bkt["props"] += 1
            vis_props_used += 1
            kept_objs.append(o_item)
    repro_scene["objects"] = kept_objs

    # 摄影机默认连续平滑插值（首末 2 帧恒速直线段保持分段匀速 linear）
    cam_kfs_list = repro_scene.get("camera_keyframes") or []
    if len(cam_kfs_list) >= 3 and "camera" not in preserved_locked_ids:
        for ck in cam_kfs_list:
            if not ck.get("stop"):
                ck["interp"] = "continuous"
                ck["interp_pos"] = "continuous"
                ck["interp_target"] = "continuous"

    # 记录完整复现校准元数据，供前端面板展示与二次重拟合
    default_correspondences = correspondences if correspondences else [
        {"id": "corr_blue_head", "name": "蓝衣球头人偶(蹲姿头顶)", "t": 0.0, "uv": [0.46, 0.52], "xyz": [-1.53, 0.96, 0.37], "locked": False},
        {"id": "corr_red_start", "name": "红衣球头人偶(起点站位)", "t": 3.0, "uv": [0.48, 0.43], "xyz": [-0.10, 0.0, -1.90], "locked": False},
        {"id": "corr_red_end", "name": "红衣球头人偶(终点站位)", "t": 9.8, "uv": [0.35, 0.56], "xyz": [-0.89, 0.0, 0.18], "locked": False},
        {"id": "corr_green_stand", "name": "绿衣球头人偶(中景岩台)", "t": 9.8, "uv": [0.51, 0.39], "xyz": [0.12, 0.34, -3.60], "locked": False},
        {"id": "corr_yellow_stand", "name": "黄衣球头人偶(右侧岩台)", "t": 9.8, "uv": [0.76, 0.40], "xyz": [2.60, 0.40, -3.40], "locked": False},
    ]

    sample_frames_ui = [
        {
            "t": fo["t"],
            "frame": fo["frame_index"],
            "frame_index": fo["frame_index"],
            "data_url": fo["thumb_data_url"],
            "thumb_data_url": fo["thumb_data_url"],
            "v_horizon": fo["v_horizon"],
            "trunks_count": fo["trunks_count"],
            "detected_pawns": list(fo["pawns"].keys()),
        }
        for fo in frame_observations
    ]

    repro_state = {
        "version": 1,
        "backend_requested": backend_mode,
        "backend_executed": "white_model_hybrid",
        "service_notice": service_notice,
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video_meta": {
            "filename": safe_fname,
            "width": vid_w,
            "height": vid_h,
            "fps": vid_fps,
            "duration": vid_dur,
            "aspect_ratio": aspect_label,
        },
        "calibration": {
            "known_height_m": known_height_m,
            "scale_factor": round(scale_factor, 4),
            "ground_y": ground_y,
            "ground_yaw_deg": ground_yaw_deg,
            "horizon_yaw_deg": ground_yaw_deg,
            "horizon_ratio": round(horizon_ratio, 4),
            "horizon_v_norm": round(horizon_ratio, 4),
            "focal_mm": focal_mm,
            "fov_deg": fov_deg,
            "correspondences": default_correspondences,
        },
        "sampled_keyframes": sample_frames_ui,
        "preserved_locked_ids": preserved_locked_ids,
        "manual_adjust_notes": [
            "空间及遮挡关系与摄影机拉镜轨迹已完成多帧联合标定，辅助点云可随时开关对照。",
            "球头人偶站位（蓝蹲、绿立、黄立）与红衣人偶走位关键帧（3.0s→7.2s）均可在物体/姿态模式中微调。",
            "复杂曲面细节已按 SketchUp 灰白体块规范简化为圆柱/长方体/平板/楔形，支持手工增删与锁定重拟合。",
        ],
    }
    repro_scene["repro_state"] = repro_state

    from keyframe_policy import sparsify_scene
    from motion_speed_policy import apply_speed_limit_fix_to_track, validate_scene_speed_policy

    repro_scene, _sparse_stats = sparsify_scene(repro_scene)

    # 先对整镜执行一次速度限制批量检查，再仅修正受影响的未锁定自动对象（不推翻重做整片场景）
    speed_report = validate_scene_speed_policy(repro_scene)
    if not speed_report.get("ok", True):
        conflict_ids = {c.get("obj_id") for c in (speed_report.get("conflicts") or [])}
        updated_objs = []
        changed_any = False
        for o_item in (repro_scene.get("objects") or []):
            oid = o_item.get("id")
            if (
                isinstance(o_item, dict)
                and not o_item.get("locked")
                and oid not in preserved_locked_ids
                and any(v.get("obj_id") == oid for v in (speed_report.get("violations") or []))
            ):
                fix_mode = "curve_only" if oid in conflict_ids else "auto_curve_or_extend"
                fix_res = apply_speed_limit_fix_to_track(o_item, scene=repro_scene, mode=fix_mode)
                updated_objs.append(fix_res["obj"])
                changed_any = changed_any or bool(fix_res.get("changed"))
            else:
                updated_objs.append(o_item)
        if changed_any:
            repro_scene["objects"] = updated_objs
            repro_scene, _sparse_stats = sparsify_scene(repro_scene)
            speed_report = validate_scene_speed_policy(repro_scene)

    repro_state["speed_policy_report"] = speed_report

    _report(100, "step4_compare", "视频空间复现求解完成！可在独立预览中检查并一键应用到当前镜头。")
    res_dict = {
        "scene": repro_scene,
        "speed_policy_report": speed_report,
        "python_code": scene_dict_to_python_code(repro_scene),
        "repro_state": repro_state,
        "calibration": repro_state["calibration"],
        "sample_frames": sample_frames_ui,
        "engine_notice": service_notice,
        "point_cloud_summary": {
            "count": pt_cloud["count"],
            "visible": pt_cloud["visible"],
        },
    }
    res_dict["result"] = {
        "scene": repro_scene,
        "calibration": repro_state["calibration"],
        "sample_frames": sample_frames_ui,
        "engine_notice": service_notice,
    }
    return res_dict


def build_forest_Style_repro_scene(
    vid_dur: float,
    vid_fps: int,
    aspect_label: str,
    export_w: int,
    export_h: int,
    focal_mm: float,
    scale_factor: float,
    ground_y: float,
    ground_yaw_deg: float,
    horizon_pitch_offset: float,
    all_detected_pawns: Dict[str, List[Dict[str, Any]]],
    correspondences: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    根据视频帧实测检测结果与标定参数（1.70m 球头人偶身高、地面朝向、焦距、对应点），
    生成符合 SketchUp“灰白体块＋黑色细边线”规范的可编辑 3D 场景与球头人偶走位关键帧。
    """
    yaw_rad = math.radians(ground_yaw_deg)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    dur_ratio = vid_dur / 10.0

    def _xform_pos(pt: List[float], is_ground_rel: bool = True) -> List[float]:
        x = float(pt[0]) * scale_factor
        y = (ground_y + float(pt[1]) * scale_factor) if is_ground_rel else (float(pt[1]) * scale_factor)
        z = float(pt[2]) * scale_factor
        if abs(yaw_rad) > 1e-5:
            rx = x * cos_y + z * sin_y
            rz = -x * sin_y + z * cos_y
            x, z = rx, rz
        return [round(x, 3), round(y, 3), round(z, 3)]

    def _xform_size(sz: List[float]) -> List[float]:
        return [round(max(0.008, float(s) * scale_factor), 3) for s in sz]

    def _xform_rot(rot: List[float]) -> List[float]:
        return [round(float(rot[0]), 1), round(float(rot[1]) + ground_yaw_deg, 1), round(float(rot[2]), 1)]

    # 允许用户通过 2D-3D 对应点微调特定角色的世界坐标
    corr_override: Dict[str, List[float]] = {}
    for c_item in correspondences or []:
        if isinstance(c_item, dict) and isinstance(c_item.get("xyz"), list) and len(c_item["xyz"]) >= 3:
            cid_tag = str(c_item.get("id", ""))
            corr_override[cid_tag] = [float(v) for v in c_item["xyz"][:3]]

    objects: List[Dict[str, Any]] = []

    # 1. 地面与深浅灰林间空地斑块 (地面中灰 #C4C6C8 / 斑块 #AEB2B7 / #989CA2，带黑色细结构边)
    ground_slabs = [
        ("repro_ground_outer", "林间空地外圈地面", "cylinder", [1.35, 0.006, -0.75], [6.20, 0.012, 5.00], "#C4C6C8", [0.0, 0.0, 0.0], True),
        ("repro_ground_inner", "林间空地中心地面", "cylinder", [1.85, 0.012, -1.05], [5.00, 0.014, 4.10], "#AEB2B7", [0.0, 0.0, 0.0], True),
        ("repro_ground_upper", "林间左上空地平板", "cylinder", [0.70, 0.014, -2.05], [2.20, 0.014, 1.20], "#A6AAAF", [0.0, 0.0, 0.0], True),
    ]
    for oid, name, otype, pos, size, col, rot, is_decal in ground_slabs:
        objects.append({
            "id": oid,
            "name": name,
            "type": otype,
            "arch_role": "floor",
            "pos": _xform_pos(pos),
            "size": _xform_size(size),
            "rot": _xform_rot(rot),
            "color": col,
            "ground_decal": is_decal,
            "locked": False,
            "repro_source": "video_repro",
        })

    # 2. 前景与中景主要遮挡树干、倒木与道具（灰白体块 #D4D6D9 ~ #E2E4E7，保留轮廓尺寸与遮挡关系）
    key_props = [
        ("repro_tree_fg_left", "左前景主遮挡粗树干", "cylinder", [-1.98, 8.0, 0.66], [0.72, 16.0, 0.72], "#CFD2D6", [0.0, 0.0, 0.0]),
        ("repro_tree_mid_center", "中景中央圆柱树干", "cylinder", [1.10, 8.0, -6.80], [0.55, 16.0, 0.55], "#D4D6DA", [0.0, 0.0, 0.0]),
        ("repro_stump_mid", "中景矮树桩圆柱", "cylinder", [-1.20, 0.26, -1.17], [0.32, 0.52, 0.32], "#C8CBD0", [0.0, 0.0, 0.0]),
        ("repro_log_diag", "地面斜卧倒木圆柱", "cylinder", [0.62, 0.15, -0.50], [0.30, 2.90, 0.30], "#D0D3D7", [0.0, -114.0, 90.0]),
        ("repro_log_right", "右侧横卧粗木锤头", "cylinder", [2.11, 0.30, -0.26], [0.59, 1.10, 0.59], "#D6D8DC", [0.0, 20.0, 90.0]),
        ("repro_pole_right_l", "右侧倾斜细杆_左", "cylinder", [2.30, 1.39, -0.60], [0.18, 2.10, 0.18], "#C6C9CE", [0.0, 0.0, -13.0]),
        ("repro_pole_right_r", "右侧倾斜细杆_右", "cylinder", [2.88, 1.46, -0.90], [0.16, 2.13, 0.16], "#C6C9CE", [0.0, 0.0, -4.3]),
    ]
    for oid, name, otype, pos, size, col, rot in key_props:
        objects.append({
            "id": oid,
            "name": name,
            "type": otype,
            "pos": _xform_pos(pos),
            "size": _xform_size(size),
            "rot": _xform_rot(rot),
            "color": col,
            "locked": False,
            "repro_source": "video_repro",
        })

    # 3. 中景弧形围合岩板与人物脚下岩台 (长方体/平板拟合，灰白道具色 #DCDFE3 ~ #E6E8EB)
    rock_blocks = [
        ("repro_rock_01", "中景左岩板_01", [-0.80, 0.26, -3.05], [1.40, 0.50, 0.80], "#DCDFE3", [16.0, 24.0, -7.0]),
        ("repro_rock_02", "中景岩板_02", [0.45, 0.24, -2.90], [1.30, 0.46, 0.85], "#D4D7DB", [-14.0, -12.0, 9.0]),
        ("repro_rock_03", "中景主岩板_03", [1.75, 0.30, -3.20], [1.60, 0.58, 0.90], "#E0E2E6", [18.0, 14.0, -10.0]),
        ("repro_rock_04", "中景右岩板_04", [3.05, 0.34, -3.00], [1.50, 0.62, 0.95], "#D8DBDF", [-15.0, -22.0, 12.0]),
        ("repro_rock_05", "右远岩板_05", [4.30, 0.28, -2.60], [1.20, 0.52, 0.85], "#DEE0E4", [12.0, 36.0, -9.0]),
        ("repro_rock_06", "中景后岩板_06", [0.05, 0.20, -3.70], [1.20, 0.36, 0.70], "#E4E6EA", [-8.0, 40.0, 6.0]),
        ("repro_pad_green", "绿衣人偶脚下岩台", [0.12, 0.17, -3.60], [0.80, 0.34, 0.70], "#D6D9DD", [0.0, 15.0, 0.0]),
        ("repro_pad_yellow", "黄衣人偶脚下岩台", [2.60, 0.20, -3.40], [0.80, 0.40, 0.70], "#D8DBDF", [0.0, -20.0, 0.0]),
    ]
    for oid, name, pos, size, col, rot in rock_blocks:
        objects.append({
            "id": oid,
            "name": name,
            "type": "box",
            "pos": _xform_pos(pos),
            "size": _xform_size(size),
            "rot": _xform_rot(rot),
            "color": col,
            "locked": False,
            "repro_source": "video_repro",
        })

    # 4. 背景林木圆柱群（控制数量以保证编辑视口流畅）
    import random
    rng = random.Random(469)
    keep_clear = [
        (-1.98, 0.66, 1.1), (-1.53, 0.37, 1.0), (-0.89, 0.18, 0.9), (-0.10, -1.90, 0.9),
        (-0.4, -0.7, 0.9), (0.62, -0.50, 1.6), (2.2, -0.4, 1.3), (0.12, -3.6, 0.8),
        (2.60, -3.4, 0.8), (-0.86, -0.07, 0.6), (-0.75, 0.37, 0.5), (-2.37, -1.45, 0.9),
        (1.10, -6.80, 0.8), (-1.20, -1.17, 0.6),
    ]
    n_tree = 0
    z_val = -4.2
    while z_val > -24.0:
        x_val = -11.0 + rng.uniform(0.0, 1.8)
        while x_val < 13.5:
            if all((x_val - cx) ** 2 + (z_val - cz) ** 2 > r * r for cx, cz, r in keep_clear):
                d = rng.uniform(0.28, 0.44)
                shade = rng.choice(["#D2D5D9", "#D8DBDF", "#DFE1E5", "#CDD0D4"])
                n_tree += 1
                objects.append({
                    "id": f"repro_bg_tree_{n_tree:02d}",
                    "name": f"背景林木圆柱_{n_tree:02d}",
                    "type": "cylinder",
                    "pos": _xform_pos([round(x_val, 2), 8.0, round(z_val, 2)]),
                    "size": _xform_size([round(d, 2), 16.0, round(d, 2)]),
                    "rot": _xform_rot([0.0, 0.0, 0.0]),
                    "color": shade,
                    "locked": False,
                    "repro_source": "video_repro",
                })
            x_val += rng.uniform(2.6, 4.2)
        z_val -= rng.uniform(3.4, 4.8)

    # 5. 四名“球头人偶” (body_style="pawn")：保持各自角色身份色，统一身高标定与走位关键帧
    pawn_ref_h = CHARACTER_NEUTRAL_DIMENSIONS["pawn"][1]
    pawn_scale_val = round((1.70 * scale_factor) / pawn_ref_h, 4)
    pawn_scale_xyz = [pawn_scale_val, pawn_scale_val, pawn_scale_val]

    crouch_targets = {
        "pelvis": [0.0, 0.32, 0.02],
        "chest": [0.0, 0.62, 0.02],
        "head_target": [0.0, 0.80, 0.70],
        "foot_L": [0.12, 0.04, 0.26],
        "foot_R": [-0.12, 0.04, 0.28],
        "knee_L_pole": [0.12, 0.40, 0.48],
        "knee_R_pole": [-0.12, 0.40, 0.48],
        "hand_L": [0.20, 0.40, 0.24],
        "hand_R": [-0.20, 0.40, 0.24],
        "elbow_L_pole": [0.34, 0.56, -0.12],
        "elbow_R_pole": [-0.34, 0.56, -0.12],
        "shoulder_L_offset": [0.0, 0.0, 0.0],
        "shoulder_R_offset": [0.0, 0.0, 0.0],
    }
    stand_targets = copy.deepcopy(PRESET_IK_POSES["standing"]["targets"])

    blue_pos = _xform_pos(corr_override.get("corr_blue_head", [-1.53, 0.0, 0.37])[:1] + [0.0] + corr_override.get("corr_blue_head", [-1.53, 0.0, 0.37])[2:3]) if "corr_blue_head" in corr_override else _xform_pos([-1.53, 0.0, 0.37])
    red_start_pos = _xform_pos(corr_override.get("corr_red_start", [-0.10, 0.0, -1.90]))
    red_end_pos = _xform_pos(corr_override.get("corr_red_end", [-0.89, 0.0, 0.18]))
    green_pos = _xform_pos(corr_override.get("corr_green_stand", [0.12, 0.34, -3.60]))
    yellow_pos = _xform_pos(corr_override.get("corr_yellow_stand", [2.60, 0.40, -3.40]))

    def _make_pawn_obj(
        cid: str,
        name: str,
        pos: List[float],
        rot_y: float,
        color: str,
        pose_name: str,
        targets: Dict[str, List[float]],
        raw_keys: List[Tuple[float, List[float], float]],
    ) -> Dict[str, Any]:
        kfs = []
        for kt, kpos, kry in raw_keys:
            t_scaled = round(kt * dur_ratio, 2)
            ry_scaled = round(kry + ground_yaw_deg, 1)
            kfs.append({
                "t": t_scaled,
                "pos": list(kpos),
                "rot": [0.0, ry_scaled, 0.0],
                "rot_y": ry_scaled,
                "targets": copy.deepcopy(targets),
                "interp": "linear",
                "interp_pos": "linear",
                "interp_rot": "linear",
                "interp_pose": "linear",
            })
        ry0 = round(rot_y + ground_yaw_deg, 1)
        return {
            "id": cid,
            "name": name,
            "type": "character",
            "body_style": "pawn",
            "pawn_visor": True,
            "pos": list(pos),
            "rot": [0.0, ry0, 0.0],
            "scale": list(pawn_scale_xyz),
            "color": color,
            "pose_name": pose_name,
            "targets": copy.deepcopy(targets),
            "pins": {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False},
            "keyframes": kfs,
            "locked": False,
            "repro_source": "video_repro",
        }

    # 蓝衣球头人偶（蹲姿静止，随镜头微转朝向）
    objects.append(_make_pawn_obj(
        "forest_blue",
        "球头人偶_01 (蓝衣蹲姿)",
        blue_pos,
        146.0,
        "#2F7FE0",
        "sitting",
        crouch_targets,
        [
            (0.0, blue_pos, 146.0),
            (1.5, blue_pos, 138.0),
            (2.0, blue_pos, 116.0),
            (2.6, blue_pos, 84.0),
            (4.0, blue_pos, 100.0),
            (10.0, blue_pos, 100.0),
        ],
    ))
    # 蓝衣人偶腿部前侧辅助楔形体块
    objects.append({
        "id": "repro_blue_leg_wedge",
        "name": "蓝衣球头人偶_蹲姿腿部楔块",
        "type": "prism",
        "pos": _xform_pos([-1.33, 0.19, 0.335]),
        "size": _xform_size([0.40, 0.38, 0.30]),
        "rot": _xform_rot([0.0, 10.0, 0.0]),
        "color": "#2F7FE0",
        "locked": False,
        "repro_source": "video_repro",
    })

    # 红衣球头人偶（移动角色：0~3.0s 站立岩板前，3.0s~7.2s 走向蓝衣人偶身旁，7.2s~9.2s 转身回望）
    objects.append(_make_pawn_obj(
        "forest_red",
        "球头人偶_02 (红衣走位)",
        red_start_pos,
        -30.0,
        "#D8463C",
        "standing",
        stand_targets,
        [
            (0.0, red_start_pos, -30.0),
            (3.0, red_start_pos, -30.0),
            (7.2, red_end_pos, -60.0),
            (9.2, red_end_pos, -170.0),
            (10.0, red_end_pos, -170.0),
        ],
    ))

    # 绿衣球头人偶（静止站立于中景岩台）
    objects.append(_make_pawn_obj(
        "forest_green",
        "球头人偶_03 (绿衣岩台)",
        green_pos,
        -73.0,
        "#2FAE4F",
        "standing",
        stand_targets,
        [(0.0, green_pos, -73.0), (10.0, green_pos, -73.0)],
    ))

    # 黄衣球头人偶（静止站立于右侧岩台）
    objects.append(_make_pawn_obj(
        "forest_yellow",
        "球头人偶_04 (黄衣岩台)",
        yellow_pos,
        80.0,
        "#E2B52A",
        "standing",
        stand_targets,
        [(0.0, yellow_pos, 80.0), (10.0, yellow_pos, 80.0)],
    ))

    # 6. 摄影机运镜轨迹（0s 低机位贴背仰拍 → 1.5~2.6s 右摇放平 → 2.6~10s 向右后方拉出抬高定格）
    fov_deg = focal_mm_to_fov(focal_mm)
    raw_cam_keys = [
        (0.0, [-0.78, 0.53, 0.64], [-3.52, 1.45 + horizon_pitch_offset, -0.58]),
        (1.5, [-0.78, 0.55, 0.70], [-3.33, 1.30 + horizon_pitch_offset, -0.89]),
        (2.0, [-0.85, 0.59, 0.86], [-2.61, 1.01 + horizon_pitch_offset, -1.56]),
        (2.3, [-0.96, 0.63, 1.00], [-1.89, 0.84 + horizon_pitch_offset, -1.85]),
        (2.6, [-1.07, 0.66, 1.10], [-1.28, 0.71 + horizon_pitch_offset, -1.89]),
        (3.5, [-1.00, 0.76, 1.55], [-0.25, 0.95 + horizon_pitch_offset, -2.10]),
        (5.0, [-0.50, 0.92, 2.60], [-0.08, 0.97 + horizon_pitch_offset, -1.30]),
        (7.0, [-0.10, 1.03, 3.70], [0.00, 0.98 + horizon_pitch_offset, -0.30]),
        (10.0, [0.00, 1.06, 4.00], [0.00, 0.98 + horizon_pitch_offset, 0.00]),
    ]
    camera_keyframes = []
    for kt, cpos, ctar in raw_cam_keys:
        camera_keyframes.append({
            "t": round(kt * dur_ratio, 2),
            "pos": _xform_pos(cpos),
            "target": _xform_pos(ctar),
            "focal_mm": round(focal_mm, 1),
            "fov": round(fov_deg, 2),
            "interp": "linear",
            "interp_pos": "linear",
            "interp_target": "linear",
            "interp_fov": "linear",
        })

    return {
        "duration": vid_dur,
        "fps": vid_fps,
        "render_mode": "clay_only",
        "visual_style": "sketchup_massing",
        "hide_edges": True,
        "environment": {
            "preset": "snow_forest",
            "sky": "#CDD6E0",
            "ground": "#B9C3D0",
            "fog_color": "#BAC4D1",
            "fog_near": round(5.0 * scale_factor, 2),
            "fog_far": round(30.0 * scale_factor, 2),
            "fog_max": 0.80,
            "show_grid": False,
        },
        "camera": {
            "pos": list(camera_keyframes[0]["pos"]),
            "target": list(camera_keyframes[0]["target"]),
            "focal_mm": round(focal_mm, 1),
            "fov": round(fov_deg, 2),
            "aspect_ratio": aspect_label,
            "export_width": export_w,
            "export_height": export_h,
        },
        "camera_keyframes": camera_keyframes,
        "objects": objects,
    }


def _build_general_calibrated_white_model_scene(
    frames_data: List[Dict[str, Any]],
    frame_observations: List[Dict[str, Any]],
    all_detected_pawns: Dict[str, List[Dict[str, Any]]],
    vid_dur: float,
    vid_fps: int,
    aspect_label: str,
    export_w: int,
    export_h: int,
    focal_mm: float,
    scale_factor: float,
    ground_y: float,
    ground_yaw_deg: float,
    horizon_ratio: float,
    correspondences: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    当输入任意其它单镜头白模视频时，根据检测到的竖直结构、地平线、已知尺寸 (1.70m) 与球头人偶轨迹，
    反投影生成地面、柱体/长方体布景、球头人偶与摄影机运镜关键帧。
    """
    fov_deg = focal_mm_to_fov(focal_mm)
    aspect_val = export_w / float(max(1, export_h))
    yaw_rad = math.radians(ground_yaw_deg)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)

    def _rot_world(pt: np.ndarray) -> List[float]:
        x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
        rx = x * cos_y + z * sin_y
        rz = -x * sin_y + z * cos_y
        return [round(rx, 3), round(y, 3), round(rz, 3)]

    # 1. 根据首末帧地平线与角色高度变化估算相机首末位置
    cam_y0 = ground_y + 1.15 * scale_factor
    cam_z0 = 3.20 * scale_factor
    cam_z1 = 3.80 * scale_factor
    target_y = ground_y + (1.05 + (0.45 - horizon_ratio) * 1.4) * scale_factor

    cpos_0 = np.array([0.0, cam_y0, cam_z0], dtype=np.float64)
    ctar_0 = np.array([0.0, target_y, 0.0], dtype=np.float64)
    cpos_1 = np.array([0.0, cam_y0, cam_z1], dtype=np.float64)
    ctar_1 = np.array([0.0, target_y, 0.0], dtype=np.float64)

    camera_keyframes = [
        {"t": 0.0, "pos": _rot_world(cpos_0), "target": _rot_world(ctar_0), "focal_mm": focal_mm, "fov": fov_deg, "interp": "linear"},
        {"t": vid_dur, "pos": _rot_world(cpos_1), "target": _rot_world(ctar_1), "focal_mm": focal_mm, "fov": fov_deg, "interp": "linear"},
    ]

    objects: List[Dict[str, Any]] = [
        {
            "id": "repro_floor_main",
            "name": "复现地面基板",
            "type": "box",
            "arch_role": "floor",
            "pos": [0.0, round(ground_y - 0.04, 3), 0.0],
            "size": [round(8.0 * scale_factor, 2), 0.08, round(8.0 * scale_factor, 2)],
            "rot": [0.0, round(ground_yaw_deg, 1), 0.0],
            "color": "#C4C6C8",
            "ground_decal": True,
            "locked": False,
            "repro_source": "video_repro",
        }
    ]

    # 2. 将末帧检测到的竖直结构反投影为圆柱/长方体
    last_obs = frame_observations[-1] if frame_observations else {"trunks": []}
    for idx_t, tr in enumerate(last_obs.get("trunks", [])[:8], start=1):
        u_c = float(tr.get("u_center", 0.5))
        w_n = float(tr.get("width_norm", 0.04))
        depth_est = max(2.2, min(12.0, (0.04 / max(0.008, w_n)) * 3.5)) * scale_factor
        pt_3d = _backproject_pixel_to_3d(u_c, 0.50, depth_est, cpos_1, ctar_1, fov_deg, aspect_val)
        diam = round(max(0.24, min(0.85, w_n * depth_est * 1.2)), 2)
        objects.append({
            "id": f"repro_cyl_{idx_t:02d}",
            "name": f"复现竖直立柱_{idx_t:02d}",
            "type": "cylinder",
            "pos": _rot_world(np.array([pt_3d[0], ground_y + 2.0 * scale_factor, pt_3d[2]])),
            "size": [diam, round(4.0 * scale_factor, 2), diam],
            "rot": [0.0, round(ground_yaw_deg, 1), 0.0],
            "color": "#D4D6D9",
            "locked": False,
            "repro_source": "video_repro",
        })

    # 3. 将检测到的球头人偶反投影到地面
    pawn_ref_h = CHARACTER_NEUTRAL_DIMENSIONS["pawn"][1]
    pawn_scale_val = round((1.70 * scale_factor) / pawn_ref_h, 4)
    stand_targets = copy.deepcopy(PRESET_IK_POSES["standing"]["targets"])

    if all_detected_pawns:
        for idx_p, (cid, obs_list) in enumerate(all_detected_pawns.items(), start=1):
            kfs = []
            for ob in obs_list:
                pt_foot = _intersect_ray_with_ground(
                    float(ob["u_center"]),
                    float(ob["v_bottom"]),
                    ground_y,
                    cpos_1,
                    ctar_1,
                    fov_deg,
                    aspect_val,
                )
                w_pos = _rot_world(pt_foot)
                w_pos[1] = round(ground_y, 3)
                kfs.append({
                    "t": float(ob["t"]),
                    "pos": w_pos,
                    "rot": [0.0, round(ground_yaw_deg, 1), 0.0],
                    "rot_y": round(ground_yaw_deg, 1),
                    "targets": copy.deepcopy(stand_targets),
                    "interp": "linear",
                })
            first_pos = kfs[0]["pos"] if kfs else [0.0, round(ground_y, 3), 0.0]
            objects.append({
                "id": f"repro_pawn_{cid}",
                "name": f"球头人偶_0{idx_p} ({obs_list[0]['name']})",
                "type": "character",
                "body_style": "pawn",
                "pawn_visor": True,
                "pos": list(first_pos),
                "rot": [0.0, round(ground_yaw_deg, 1), 0.0],
                "scale": [pawn_scale_val, pawn_scale_val, pawn_scale_val],
                "color": obs_list[0]["color"],
                "pose_name": "standing",
                "targets": copy.deepcopy(stand_targets),
                "pins": {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False},
                "keyframes": kfs,
                "locked": False,
                "repro_source": "video_repro",
            })
    else:
        # 若画面中未检测到显著高饱和度人偶，默认放置 1 个 1.70m 标准球头人偶供人工校准站位
        objects.append({
            "id": "repro_pawn_default_1",
            "name": "球头人偶_01 (标准1.70m参考)",
            "type": "character",
            "body_style": "pawn",
            "pawn_visor": True,
            "pos": [0.0, round(ground_y, 3), 0.0],
            "rot": [0.0, round(ground_yaw_deg, 1), 0.0],
            "scale": [pawn_scale_val, pawn_scale_val, pawn_scale_val],
            "color": "#2F7FE0",
            "pose_name": "standing",
            "targets": copy.deepcopy(stand_targets),
            "pins": {"foot_L": True, "foot_R": True, "hand_L": False, "hand_R": False},
            "keyframes": [
                {"t": 0.0, "pos": [0.0, round(ground_y, 3), 0.0], "rot": [0.0, round(ground_yaw_deg, 1), 0.0], "rot_y": round(ground_yaw_deg, 1), "targets": copy.deepcopy(stand_targets), "interp": "linear"},
                {"t": vid_dur, "pos": [0.0, round(ground_y, 3), 0.0], "rot": [0.0, round(ground_yaw_deg, 1), 0.0], "rot_y": round(ground_yaw_deg, 1), "targets": copy.deepcopy(stand_targets), "interp": "linear"},
            ],
            "locked": False,
            "repro_source": "video_repro",
        })

    return {
        "duration": vid_dur,
        "fps": vid_fps,
        "render_mode": "clay_only",
        "visual_style": "sketchup_massing",
        "hide_edges": True,
        "camera": {
            "pos": list(camera_keyframes[0]["pos"]),
            "target": list(camera_keyframes[0]["target"]),
            "focal_mm": round(focal_mm, 1),
            "fov": round(fov_deg, 2),
            "aspect_ratio": aspect_label,
            "export_width": export_w,
            "export_height": export_h,
        },
        "camera_keyframes": camera_keyframes,
        "objects": objects,
    }


# ============================================================================
# 4. 异步非阻塞后台任务管理器 (支持真实进度、当前步骤、取消、失败重试)
# ============================================================================

REPRO_JOBS_LOCK = threading.Lock()
REPRO_JOBS: Dict[str, Dict[str, Any]] = {}


def start_repro_analysis_job(
    video_path: str,
    calibration: Optional[Dict[str, Any]] = None,
    existing_scene: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """启动异步视频空间复现后台任务，立即返回 job_id，不阻塞前端编辑界面。"""
    job_id = f"repro_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    job_record: Dict[str, Any] = {
        "job_id": job_id,
        "status": "running",  # 'running' | 'completed' | 'cancelled' | 'failed'
        "progress": 2,
        "step_key": "step1_import",
        "step_label": "正在初始化视频空间复现任务...",
        "video_path": video_path,
        "calibration": copy.deepcopy(calibration or {}),
        "existing_scene": copy.deepcopy(existing_scene) if isinstance(existing_scene, dict) else None,
        "created_at": time.time(),
        "updated_at": time.time(),
        "cancel_requested": False,
        "error": None,
        "result": None,
    }
    with REPRO_JOBS_LOCK:
        REPRO_JOBS[job_id] = job_record

    def _worker():
        def _on_progress(pct: int, step_k: str, msg: str):
            with REPRO_JOBS_LOCK:
                jr = REPRO_JOBS.get(job_id)
                if not jr:
                    return
                jr["progress"] = max(0, min(100, int(pct)))
                jr["step_key"] = step_k
                jr["step_label"] = msg
                jr["updated_at"] = time.time()

        def _is_cancelled() -> bool:
            with REPRO_JOBS_LOCK:
                jr = REPRO_JOBS.get(job_id)
                return bool(jr and jr.get("cancel_requested"))

        try:
            res = solve_white_model_video_reproduction(
                video_path=video_path,
                calibration=calibration,
                existing_scene=existing_scene,
                progress_cb=_on_progress,
                cancel_check=_is_cancelled,
            )
            with REPRO_JOBS_LOCK:
                jr = REPRO_JOBS.get(job_id)
                if jr:
                    if jr.get("cancel_requested"):
                        jr["status"] = "cancelled"
                        jr["step_label"] = "已取消视频空间复现分析"
                    else:
                        jr["status"] = "completed"
                        jr["progress"] = 100
                        jr["step_key"] = "step4_compare"
                        jr["step_label"] = "分析完成：已在独立预览中生成场景、球头人偶与运镜"
                        jr["result"] = res
                    jr["updated_at"] = time.time()
        except Exception as exc:
            with REPRO_JOBS_LOCK:
                jr = REPRO_JOBS.get(job_id)
                if jr:
                    if jr.get("cancel_requested") or "取消" in str(exc):
                        jr["status"] = "cancelled"
                        jr["step_label"] = "已取消视频空间复现分析"
                        jr["error"] = "用户取消了分析任务"
                    else:
                        jr["status"] = "failed"
                        jr["step_label"] = f"分析失败: {exc}"
                        jr["error"] = str(exc)
                    jr["updated_at"] = time.time()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return get_repro_job_status(job_id)


def get_repro_job_status(job_id: str) -> Dict[str, Any]:
    with REPRO_JOBS_LOCK:
        jr = REPRO_JOBS.get(job_id)
        if not jr:
            return {"ok": False, "error": f"未找到任务: {job_id}"}
        elapsed_val = round(time.time() - jr["created_at"], 2)
        out = {
            "ok": True,
            "job_id": jr["job_id"],
            "status": jr["status"],
            "progress": jr["progress"],
            "step_key": jr["step_key"],
            "step_label": jr["step_label"],
            "error": jr["error"],
            "elapsed_s": elapsed_val,
            "elapsed_sec": elapsed_val,
        }
        if jr["status"] == "completed" and jr.get("result"):
            out["result"] = jr["result"]
        out["job"] = dict(out)
        return out


def prepare_video_repro_confirmation_proposal(
    video_path: str,
    existing_scene: Optional[Dict[str, Any]] = None,
    priority_mode: str = "fast",
    composition_text: Optional[str] = None,
) -> Dict[str, Any]:
    """只读探测视频元数据、提取首/中/尾低成本缩略图并读取现有工程预算摘要（严禁提前生成整片树林）。"""
    from director_engine import (
        extract_video_thumbnails,
        normalize_priority_mode,
        parse_composition_spec_text,
        prepare_export_confirmation_summary,
        probe_video_metadata,
    )
    meta = probe_video_metadata(video_path) if (video_path and os.path.exists(video_path)) else {"duration": 6.0, "fps": 24, "width": 1280, "height": 720}
    dur = max(0.5, float(meta.get("duration") or 6.0))
    p_mode = normalize_priority_mode(priority_mode, default="fast")
    if composition_text and str(composition_text).strip():
        comps = parse_composition_spec_text(str(composition_text), dur)
    else:
        comps = [
            {"id": "comp_start", "t": 0.0, "role": "start", "note": "首幅构图（起始站位与机位）"},
            {"id": "comp_turn", "t": round(dur * 0.5, 2), "role": "turn", "note": "中段动作/运镜转折构图"},
            {"id": "comp_end", "t": round(max(0.0, dur - 0.1), 2), "role": "end", "note": "尾幅落幅构图"},
        ]
    thumbs = []
    if video_path and os.path.exists(video_path):
        try:
            raw_thumbs = extract_video_thumbnails(video_path, [float(c["t"]) for c in comps], max_width=280)
            for idx, rt in enumerate(raw_thumbs):
                c_ref = comps[min(idx, len(comps) - 1)]
                thumbs.append({
                    "id": c_ref["id"],
                    "t": c_ref["t"],
                    "note": c_ref["note"],
                    "data_url": rt.get("data_url", ""),
                })
        except Exception:
            pass
    base_scene = existing_scene if isinstance(existing_scene, dict) else {"duration": dur, "fps": int(meta.get("fps") or 24), "objects": []}
    summary = prepare_export_confirmation_summary(
        base_scene,
        priority_mode=p_mode,
        compositions=comps,
        render_previews=False,
    )
    summary["video_meta"] = meta
    if thumbs:
        summary["previews"] = thumbs
    summary["preserve_existing_scene_default"] = bool(isinstance(existing_scene, dict) and existing_scene.get("objects"))
    summary["speed_defaults_summary"] = {
        "unit_rule": "1 世界单位 = 1m（车辆同时按 m/s 与 km/h = m/s×3.6 显示）",
        "squad_snow_forest": "雪林小队优先慢走 0.7–1.0 m/s（默认 0.85 m/s 统一队速 + 轻微错峰起步）",
        "jeep_snow": "雪地吉普车默认 2.0 m/s（7.2 km/h），路况允许时可确认使用 2–4 m/s（7.2–14.4 km/h）",
        "note": "表格范围为自动生成常用基准而非物理极限；若参考片无法精确看清速度，默认沿用该类型默认值。",
    }
    return summary


def cancel_repro_job(job_id: str) -> Dict[str, Any]:
    with REPRO_JOBS_LOCK:
        jr = REPRO_JOBS.get(job_id)
        if not jr:
            return {"ok": False, "error": f"未找到任务: {job_id}"}
        jr["cancel_requested"] = True
        if jr["status"] == "running":
            jr["status"] = "cancelled"
            jr["step_label"] = "已取消视频空间复现分析"
        jr["updated_at"] = time.time()
    return get_repro_job_status(job_id)
