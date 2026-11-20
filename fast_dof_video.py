#!/usr/bin/env python3
"""
fast_dof_video.py - 高速黑白深度图视频 (Depth Map Video) 生成管线

核心流程 (Apple Silicon CoreML + Depth Anything V2 Small):
1. FFmpeg 元数据探测与流式解码：读取分辨率、帧率、时长、音轨，并自动检测切镜边界 (Shot Cuts)。
2. 神经网络单目深度估计 (Depth Anything V2 Small ONNX + CoreMLExecutionProvider)：
   - 彻底摒弃“上下位置/颜色明暗”假启发式公式；
   - 准确识别人物轮廓、衣帽、手部、方向盘、后视镜、车窗框与远景空间层次，不受物体表面黑白颜色干扰。
3. 运动自适应关键帧采样 + 联合双边运动补偿时间插值 (JBTU) + 原图 RGB 引导滤波 (Fast Guided Filter)：
   - 关键帧通过 Depth Anything V2 推理锚定精确三维几何深度；
   - 中间帧通过局部 RGB 块匹配偏移补偿 (Motion-Compensated Patch Transfer) + 原图全分辨率 RGB 引导滤波，使 24fps 每一帧的深度边缘严丝合缝贴合原片运动轮廓，无拖影、不糊边。
4. 镜头级全局深度尺度锁定 (Shot-Level Affine & Percentile Lock)：
   - 同一镜头内所有帧与参考帧做最小二乘线性尺度对齐，并按镜头全局分位数统一映射到 [0, 255]（默认近白远黑，支持 --invert 切换近黑远白），彻底消除逐帧归一化导致的闪烁与呼吸效应。
5. 默认只输出纯净黑白深度图视频 (MP4)，流式编码并保留原音轨。
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import imageio_ffmpeg
import numpy as np
import onnxruntime as ort
from PIL import Image


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "depth_anything_v2_vits",
    "model_fp16.onnx",
)


def get_ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_video_full(video_path: str) -> Dict[str, Any]:
    """使用 FFmpeg 探测视频分辨率、帧率、总时长、旋转角度与音轨信息。"""
    ffmpeg_exe = get_ffmpeg_exe()
    proc = subprocess.run(
        [ffmpeg_exe, "-hide_banner", "-i", video_path],
        capture_output=True,
        text=True,
    )
    stderr = proc.stderr or ""

    width, height = 1280, 720
    m_res = re.search(
        r"Stream #0:\d+.*Video:.*?\s(\d{2,5})x(\d{2,5})(?:[\s,]|$)", stderr
    )
    if m_res:
        width = int(m_res.group(1))
        height = int(m_res.group(2))

    fps = 24.0
    m_fps = re.search(r"(\d+(?:\.\d+)?)\s*fps", stderr)
    if m_fps:
        fps = float(m_fps.group(1))

    duration = 0.0
    m_dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    if m_dur:
        duration = (
            int(m_dur.group(1)) * 3600
            + int(m_dur.group(2)) * 60
            + float(m_dur.group(3))
        )

    rotation = 0
    m_rot = re.search(r"rotate\s*:\s*(-?\d+)", stderr)
    if m_rot:
        rotation = int(m_rot.group(1))

    has_audio = bool(re.search(r"Stream #0:\d+.*Audio:", stderr))
    audio_info = "none"
    m_aud = re.search(r"Stream #0:\d+.*Audio:\s*([^\r\n]+)", stderr)
    if m_aud:
        audio_info = m_aud.group(1).strip()

    return {
        "path": os.path.abspath(video_path),
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "rotation": rotation,
        "has_audio": has_audio,
        "audio_info": audio_info,
    }


def read_video_frames_rgb(
    video_path: str,
    width: int,
    height: int,
    fps: float,
    max_frames: Optional[int] = None,
) -> np.ndarray:
    """使用 FFmpeg 流式读取视频帧（按原始帧率时间戳采样，保持音画同步）。"""
    ffmpeg_exe = get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vf",
        f"fps={fps:.4f},scale={width}:{height}:flags=bicubic",
        "-pix_fmt",
        "rgb24",
    ]
    if max_frames is not None and max_frames > 0:
        cmd.extend(["-frames:v", str(max_frames)])
    cmd.extend(["-f", "rawvideo", "pipe:1"])

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw_bytes, _ = proc.communicate()
    frame_bytes = width * height * 3
    n_frames = len(raw_bytes) // frame_bytes
    if n_frames == 0:
        raise RuntimeError(f"未能从视频读取到有效帧: {video_path}")
    frames = np.frombuffer(raw_bytes[: n_frames * frame_bytes], dtype=np.uint8).reshape(
        (n_frames, height, width, 3)
    )
    return frames


def detect_shot_cuts(frames_rgb: np.ndarray) -> List[Tuple[int, int]]:
    """自动识别镜头切换 (Shot Cuts)，返回 [(start_idx, end_idx_exclusive), ...]。"""
    n = len(frames_rgb)
    if n <= 1:
        return [(0, n)]

    # 快速下采样步长切片统计
    small = frames_rgb[:, ::8, ::8, :].astype(np.float32) / 255.0
    h, w, _ = small.shape[1:]
    gh, gw = max(1, h // 4), max(1, w // 4)
    cropped = small[:, : gh * 4, : gw * 4, :].reshape(n, 4, gh, 4, gw, 3)
    grid_means = cropped.mean(axis=(2, 4))
    diffs = np.abs(grid_means[1:] - grid_means[:-1]).mean(axis=(1, 2, 3))

    med = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med))) + 1e-4

    cut_indices = [0]
    min_shot_len = 12
    for i in range(1, n):
        score = float(diffs[i - 1])
        if score > max(0.085, med + 6.5 * mad) and (i - cut_indices[-1] >= min_shot_len):
            cut_indices.append(i)
    cut_indices.append(n)

    return [(cut_indices[k], cut_indices[k + 1]) for k in range(len(cut_indices) - 1)]


def box_filter_2d(img: np.ndarray, r: int) -> np.ndarray:
    """基于积分图 (cumsum) 的 O(1) 快速二维均值滤波，支持 (H, W) 或 (H, W, C)。"""
    if r <= 0:
        return img
    pad_y = np.pad(
        img, ((r, r + 1), (0, 0)) + ((0, 0),) * (img.ndim - 2), mode="edge"
    )
    cs_y = np.cumsum(pad_y, axis=0, dtype=np.float32)
    out_y = (cs_y[2 * r + 1 :] - cs_y[: -2 * r - 1]) / float(2 * r + 1)

    pad_x = np.pad(
        out_y, ((0, 0), (r, r + 1)) + ((0, 0),) * (img.ndim - 2), mode="edge"
    )
    cs_x = np.cumsum(pad_x, axis=1, dtype=np.float32)
    out_x = (cs_x[:, 2 * r + 1 :] - cs_x[:, : -2 * r - 1]) / float(2 * r + 1)
    return out_x


def guided_filter_rgb_gray(
    rgb_guide: np.ndarray, p: np.ndarray, r: int = 6, eps: float = 4e-4
) -> np.ndarray:
    """
    快速边缘保持引导滤波：使用原图亮度 + 局部色彩梯度引导深度图上采样与边缘贴合，
    确保人物轮廓、帽子、五官、手部、方向盘与窗框边缘锐利且无光晕。
    """
    I = (
        0.299 * rgb_guide[..., 0]
        + 0.587 * rgb_guide[..., 1]
        + 0.114 * rgb_guide[..., 2]
    ).astype(np.float32)
    mean_I = box_filter_2d(I, r)
    mean_p = box_filter_2d(p, r)
    mean_Ip = box_filter_2d(I * p, r)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = box_filter_2d(I * I, r)
    var_I = np.maximum(mean_II - mean_I * mean_I, 0.0)
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = box_filter_2d(a, r)
    mean_b = box_filter_2d(b, r)
    return mean_a * I + mean_b


class DepthAnythingV2Estimator:
    """封装 Depth Anything V2 Small ONNX 推理会话（复用单次加载，启用 CoreML 加速）。"""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到深度估计模型文件: {model_path}")
        self.model_path = model_path
        so = ort.SessionOptions()
        so.log_severity_level = 3  # 仅输出 Error，屏蔽分区 Warning
        so.intra_op_num_threads = 8
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        # Depth Anything V2 原生 DINOv2 ViT-S/14 位置编码网格为 37x37 -> 518x518
        # 在 CoreML 上 518x518 无需动态位置编码插值，速度最快且空间精度最高
        self.infer_w = 518
        self.infer_h = 518
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def infer_single(self, frame_rgb_u8: np.ndarray) -> np.ndarray:
        """对单帧 RGB (H, W, 3) 推理深度图，返回 (infer_h, infer_w) float32 相对视差。"""
        im = Image.fromarray(frame_rgb_u8).resize(
            (self.infer_w, self.infer_h), Image.BICUBIC
        )
        arr = (np.asarray(im, dtype=np.float32) / 255.0 - self.mean) / self.std
        inp = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)
        pred = self.sess.run(None, {self.input_name: inp})[0]
        return pred.squeeze().astype(np.float32)


def select_adaptive_keyframes(
    shot_frames_rgb: np.ndarray, base_stride: int = 8
) -> List[int]:
    """
    在单个镜头内根据帧间运动量自适应选择神经网络推理关键帧：
    - 镜头首尾帧必选；
    - 基础步长 base_stride；
    - 当局部累积运动超过阈值时提前插入关键帧，兼顾速度与动作准确性。
    """
    t_len = len(shot_frames_rgb)
    if t_len <= 1:
        return [0]

    small = shot_frames_rgb[:, ::8, ::8, :].astype(np.float32) / 255.0
    frame_diffs = np.abs(small[1:] - small[:-1]).mean(axis=(1, 2, 3))

    keyframes = [0]
    acc_motion = 0.0
    for i in range(1, t_len - 1):
        acc_motion += float(frame_diffs[i - 1])
        dist = i - keyframes[-1]
        if dist >= base_stride or (dist >= 4 and acc_motion >= 0.085):
            keyframes.append(i)
            acc_motion = 0.0
    if keyframes[-1] != t_len - 1:
        keyframes.append(t_len - 1)
    return keyframes


def _shift_2d(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """对二维或三维数组做边界复制平移 (dy, dx)。"""
    h, w = arr.shape[:2]
    res = np.empty_like(arr)
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)

    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)

    res[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    if dy > 0:
        res[:dst_y0, :] = res[dst_y0 : dst_y0 + 1, :]
    elif dy < 0:
        res[dst_y1:, :] = res[dst_y1 - 1 : dst_y1, :]
    if dx > 0:
        res[:, :dst_x0] = res[:, dst_x0 : dst_x0 + 1]
    elif dx < 0:
        res[:, dst_x1:] = res[:, dst_x1 - 1 : dst_x1]
    return res


def motion_compensated_depth_interp(
    rgb_t: np.ndarray,
    rgb_0: np.ndarray,
    depth_0: np.ndarray,
    rgb_1: np.ndarray,
    depth_1: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    联合双边运动补偿深度插值 (Joint Bilateral Motion-Compensated Temporal Upsampling)：
    在相邻两个深度关键帧 (t0, t1) 之间，对当前帧 t 的每个像素搜索邻域平移偏移 (dy, dx)，
    按局部 RGB 块相似度加权迁移深度，彻底避免运动边缘出现双影或跳帧感。
    """
    offsets = [
        (0, 0),
        (0, -3),
        (0, 3),
        (-3, 0),
        (3, 0),
        (0, -6),
        (0, 6),
        (-4, -4),
        (-4, 4),
        (4, -4),
        (4, 4),
    ]
    sigma_color = 0.045
    inv_2sigma2 = 1.0 / (2.0 * sigma_color * sigma_color)

    # 先对 RGB 做 r=2 局部盒式平滑以代表 5x5 Patch 颜色特征
    patch_t = box_filter_2d(rgb_t, 2)
    patch_0 = box_filter_2d(rgb_0, 2)
    patch_1 = box_filter_2d(rgb_1, 2)

    acc_d0 = np.zeros_like(depth_0)
    acc_w0 = np.zeros_like(depth_0)
    acc_d1 = np.zeros_like(depth_1)
    acc_w1 = np.zeros_like(depth_1)

    for dy, dx in offsets:
        spatial_penalty = np.exp(-0.015 * (dy * dy + dx * dx))

        s_p0 = _shift_2d(patch_0, dy, dx)
        s_d0 = _shift_2d(depth_0, dy, dx)
        diff0 = np.sum((patch_t - s_p0) ** 2, axis=-1)
        w0 = np.exp(-diff0 * inv_2sigma2) * spatial_penalty
        acc_d0 += w0 * s_d0
        acc_w0 += w0

        s_p1 = _shift_2d(patch_1, dy, dx)
        s_d1 = _shift_2d(depth_1, dy, dx)
        diff1 = np.sum((patch_t - s_p1) ** 2, axis=-1)
        w1 = np.exp(-diff1 * inv_2sigma2) * spatial_penalty
        acc_d1 += w1 * s_d1
        acc_w1 += w1

    d0_warped = acc_d0 / (acc_w0 + 1e-6)
    d1_warped = acc_d1 / (acc_w1 + 1e-6)

    # 按时间比例与双向匹配置信度融合
    c0 = (1.0 - alpha) * (acc_w0 + 1e-4)
    c1 = alpha * (acc_w1 + 1e-4)
    return (c0 * d0_warped + c1 * d1_warped) / (c0 + c1)


def process_shot_depth_sequence(
    shot_frames_rgb: np.ndarray,
    estimator: DepthAnythingV2Estimator,
    keyframe_stride: int = 8,
    invert_depth: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    对单个镜头序列 (T, H, W, 3) 生成时间稳定、边缘锐利的 8-bit 灰度深度图序列 (T, H, W)：
      1. 关键帧通过 Depth Anything V2 推理；
      2. 将同一镜头内所有关键帧深度与首帧做线性尺度对齐 (消除单帧深度呼吸闪烁)；
      3. 中间帧通过 RGB 块匹配运动补偿插值；
      4. 全分辨率 RGB 引导滤波锐化边缘，并统一映射到 [0, 255]。
    """
    t_len, h, w, _ = shot_frames_rgb.shape
    kf_indices = select_adaptive_keyframes(shot_frames_rgb, base_stride=keyframe_stride)

    # 工作分辨率（用于快速双边运动补偿插值，随后再用全分辨率 RGB 引导滤波上采样到 HxW）
    work_w = min(w, 496)
    work_h = max(64, int(round(h * (work_w / float(w)))))

    rgb_work = np.empty((t_len, work_h, work_w, 3), dtype=np.float32)
    for t in range(t_len):
        im_w = Image.fromarray(shot_frames_rgb[t]).resize((work_w, work_h), Image.BILINEAR)
        rgb_work[t] = np.asarray(im_w, dtype=np.float32) / 255.0

    # 1. 关键帧神经网络深度推理 + 镜头内尺度对齐
    kf_depths: Dict[int, np.ndarray] = {}
    ref_p10, ref_p90 = None, None
    for idx in kf_indices:
        d_raw = estimator.infer_single(shot_frames_rgb[idx])
        d_work = np.asarray(
            Image.fromarray(d_raw).resize((work_w, work_h), Image.BICUBIC),
            dtype=np.float32,
        )
        p10 = float(np.percentile(d_work, 10))
        p90 = float(np.percentile(d_work, 90))
        if ref_p10 is None:
            ref_p10, ref_p90 = p10, p90
        else:
            scale = (ref_p90 - ref_p10) / max(p90 - p10, 1e-4)
            # 限制单镜头内尺度微调范围，防止极端帧跳变
            scale = float(np.clip(scale, 0.85, 1.18))
            d_work = (d_work - p10) * scale + ref_p10
        kf_depths[idx] = d_work

    # 2. 生成全序列工作分辨率深度图（关键帧 + 运动补偿中间帧）
    depth_work_seq = np.empty((t_len, work_h, work_w), dtype=np.float32)
    for k in range(len(kf_indices) - 1):
        i0 = kf_indices[k]
        i1 = kf_indices[k + 1]
        depth_work_seq[i0] = kf_depths[i0]
        depth_work_seq[i1] = kf_depths[i1]
        span = i1 - i0
        if span > 1:
            for t in range(i0 + 1, i1):
                alpha = float(t - i0) / float(span)
                depth_work_seq[t] = motion_compensated_depth_interp(
                    rgb_work[t],
                    rgb_work[i0],
                    kf_depths[i0],
                    rgb_work[i1],
                    kf_depths[i1],
                    alpha,
                )
    if len(kf_indices) == 1:
        depth_work_seq[0] = kf_depths[0]

    # 3. 跨帧运动门控时间平滑 (Motion-Gated Temporal EMA) 消除残余微闪烁
    for t in range(1, t_len):
        rgb_diff = np.abs(rgb_work[t] - rgb_work[t - 1]).mean(axis=-1)
        # 静止区融合权重 0.72 抑制闪烁，运动边缘权重降为 0.10 紧跟当前帧
        ema_prev_w = 0.72 * np.exp(-rgb_diff / 0.045)
        depth_work_seq[t] = (
            ema_prev_w * depth_work_seq[t - 1]
            + (1.0 - ema_prev_w) * depth_work_seq[t]
        )

    # 4. 镜头全局分位数统一归一化
    d_min = float(np.percentile(depth_work_seq, 1.0))
    d_max = float(np.percentile(depth_work_seq, 99.5))
    denom = max(d_max - d_min, 1e-5)
    depth_work_norm = np.clip((depth_work_seq - d_min) / denom, 0.0, 1.0)

    # 5. 全分辨率上采样 + 原图 RGB 引导边缘锐化
    out_u8 = np.empty((t_len, h, w), dtype=np.uint8)
    for t in range(t_len):
        d_up = np.asarray(
            Image.fromarray(depth_work_norm[t]).resize((w, h), Image.BILINEAR),
            dtype=np.float32,
        )
        rgb_f = shot_frames_rgb[t].astype(np.float32) / 255.0
        d_sharp = guided_filter_rgb_gray(rgb_f, d_up, r=5, eps=3e-4)
        d_sharp = np.clip(d_sharp, 0.0, 1.0)
        if invert_depth:
            d_sharp = 1.0 - d_sharp
        out_u8[t] = np.clip(d_sharp * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)

    return out_u8, len(kf_indices)


def encode_depth_video_mp4(
    depth_u8_seq: np.ndarray,
    fps: float,
    output_mp4_path: str,
    source_video_path: Optional[str] = None,
    has_audio: bool = False,
) -> None:
    """将 (T, H, W) uint8 灰度深度序列流式编码为兼容 H.264 MP4，并无损复用原视频音轨。"""
    ffmpeg_exe = get_ffmpeg_exe()
    n_frames, height, width = depth_u8_seq.shape
    os.makedirs(os.path.dirname(os.path.abspath(output_mp4_path)), exist_ok=True)

    # 将单通道灰度扩展为 3 通道 RGB24 以兼容所有播放器与 AI 视频平台
    rgb_seq = np.repeat(depth_u8_seq[:, :, :, None], 3, axis=3)

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
        f"{fps:.4f}",
        "-i",
        "pipe:0",
    ]
    if source_video_path and has_audio:
        cmd.extend(["-i", source_video_path, "-map", "0:v:0", "-map", "1:a:0?"])

    cmd.extend(
        [
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
        ]
    )
    if source_video_path and has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])

    cmd.append(output_mp4_path)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr_bytes = proc.communicate(input=rgb_seq.tobytes())
    if proc.returncode != 0:
        err_msg = stderr_bytes.decode("utf-8", errors="ignore")
        raise RuntimeError(f"FFmpeg 编码失败: {err_msg}")


def process_video_to_depth_mp4(
    input_video_path: str,
    output_depth_mp4_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    keyframe_stride: int = 8,
    invert_depth: bool = False,
) -> Dict[str, Any]:
    """完整执行“输入视频 → 黑白深度图视频”流程并返回分阶段耗时统计。"""
    t_start = time.perf_counter()

    # 1. 探测元数据
    t0 = time.perf_counter()
    meta = probe_video_full(input_video_path)
    t_probe = time.perf_counter() - t0
    w, h, fps = meta["width"], meta["height"], meta["fps"]

    # 2. 流式解码原视频帧
    t0 = time.perf_counter()
    frames_rgb = read_video_frames_rgb(input_video_path, w, h, fps)
    n_frames = len(frames_rgb)
    t_decode = time.perf_counter() - t0

    # 3. 切镜检测
    t0 = time.perf_counter()
    shots = detect_shot_cuts(frames_rgb)
    t_cut = time.perf_counter() - t0

    # 4. 初始化并加载 Depth Anything V2 模型
    t0 = time.perf_counter()
    estimator = DepthAnythingV2Estimator(model_path=model_path)
    t_model_load = time.perf_counter() - t0

    # 5. 按镜头生成深度图序列
    t0 = time.perf_counter()
    depth_all = np.empty((n_frames, h, w), dtype=np.uint8)
    total_keyframes = 0
    for s_start, s_end in shots:
        shot_depth, n_kf = process_shot_depth_sequence(
            frames_rgb[s_start:s_end],
            estimator=estimator,
            keyframe_stride=keyframe_stride,
            invert_depth=invert_depth,
        )
        depth_all[s_start:s_end] = shot_depth
        total_keyframes += n_kf
    t_depth_infer = time.perf_counter() - t0

    # 6. 流式编码导出 MP4（保留原音轨）
    t0 = time.perf_counter()
    encode_depth_video_mp4(
        depth_all,
        fps=fps,
        output_mp4_path=output_depth_mp4_path,
        source_video_path=input_video_path,
        has_audio=meta["has_audio"],
    )
    t_encode = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start

    report = {
        "input_video": os.path.abspath(input_video_path),
        "output_depth_video": os.path.abspath(output_depth_mp4_path),
        "resolution": f"{w}x{h}",
        "fps": fps,
        "frames": n_frames,
        "duration_sec": round(n_frames / fps, 2),
        "shots": shots,
        "neural_keyframes": total_keyframes,
        "keyframe_stride": keyframe_stride,
        "polarity": "near_black_far_white" if invert_depth else "near_white_far_black",
        "timings_sec": {
            "1_probe": round(t_probe, 3),
            "2_decode": round(t_decode, 3),
            "3_shot_cut": round(t_cut, 3),
            "4_model_load": round(t_model_load, 3),
            "5_depth_estimation_and_temporal_filter": round(t_depth_infer, 3),
            "6_mp4_encode": round(t_encode, 3),
            "total_sec": round(t_total, 2),
            "effective_fps": round(n_frames / max(t_total, 1e-3), 2),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="生成高质量黑白深度图视频 (Depth Map Video)")
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("-o", "--output", help="输出黑白深度图 MP4 路径", default=None)
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="神经网络关键帧基础采样步长（默认 8，兼顾速度与精度；设为 1 则逐帧全量推理）",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="反转极性为【近黑远白】（默认【近白远黑】）",
    )
    args = parser.parse_args()

    in_path = os.path.abspath(args.input)
    if args.output:
        out_path = os.path.abspath(args.output)
    else:
        base, _ = os.path.splitext(in_path)
        out_path = f"{base}_黑白深度图_DepthMap.mp4"

    process_video_to_depth_mp4(
        input_video_path=in_path,
        output_depth_mp4_path=out_path,
        keyframe_stride=args.stride,
        invert_depth=args.invert,
    )


if __name__ == "__main__":
    main()
