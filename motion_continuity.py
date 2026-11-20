#!/usr/bin/env python3
"""
摄影机、人物与可移动对象统一运动连续性保护求解模块（Magic Director Stage）
===========================================================================
核心原则：
1. 空间路径 p(s) 与时间推进 s(t) 分离：路径决定去哪里（严格保留用户选择轨道形态），
   时间推进决定每刻走多快、何时停、何处减速。
2. 普通中间关键帧默认“连续经过”而不自动停车；仅明确停顿/零速平台/镜头边界/动作结束才停车，
   进入与离开停顿时速度与加速度平滑归零（v=0, a=0），禁止非零斜率硬接或负速裁剪。
3. 两种明确模式：
   - constant_speed（保持匀速）：按真实三维弧长保持巡航段恒速（起停使用平滑 S5 过渡并显式标示），
     空间路径变动时尽量维持上次确认速度并重排未锁定中间时刻；
   - smooth_variable（平滑变速，普通编辑默认）：三阶 B 样条约束 QP（ADMM）联合求解相邻锁定区间，
     数学保证 v(t) >= 0、C2 连续（速度与加速度无折角/跳变）、有限加减速与 jerk、尽量局部化修正。
4. 不可行约束检查：时空冲突或急转向心加速度冲突时明确指出冲突区间与可操作方案，
   绝不静默解锁或用隐藏尖峰掩盖。
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple


# 三点 Gauss-Legendre 积分节点 [-1, 1] -> [0, 1]（精确积分至 5 次多项式，对 3 次 B 样条及导数积严格精确）
_GL3_U: Tuple[float, ...] = (
    0.5 * (1.0 - math.sqrt(3.0 / 5.0)),
    0.5,
    0.5 * (1.0 + math.sqrt(3.0 / 5.0)),
)
_GL3_W: Tuple[float, ...] = (5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0)


DEFAULT_PROTECTION_LIMITS: Dict[str, Dict[str, float]] = {
    "camera": {
        "max_speed": 25.0,
        "max_accel": 6.5,
        "max_jerk": 24.0,
        "max_centripetal_accel": 5.0,
        "max_ang_vel": 180.0,
        "max_ang_accel": 420.0,
        "max_zoom_speed": 120.0,
        "max_zoom_accel": 280.0,
        "corner_tolerance_m": 0.12,
    },
    "character": {
        "max_speed": 3.0,
        "max_accel": 5.2,
        "max_jerk": 20.0,
        "max_centripetal_accel": 3.2,
        "max_ang_vel": 360.0,
        "max_ang_accel": 900.0,
        "corner_tolerance_m": 0.10,
    },
    "vehicle": {
        "max_speed": 16.7,
        "max_accel": 3.5,
        "max_jerk": 9.0,
        "max_centripetal_accel": 3.2,
        "max_ang_vel": 120.0,
        "max_ang_accel": 240.0,
        "corner_tolerance_m": 0.25,
    },
    "prop": {
        "max_speed": 12.0,
        "max_accel": 8.0,
        "max_jerk": 32.0,
        "max_centripetal_accel": 6.0,
        "max_ang_vel": 240.0,
        "max_ang_accel": 600.0,
        "corner_tolerance_m": 0.08,
    },
}


def quantize_time_to_frame(t: float, fps: float = 24.0, duration: Optional[float] = None) -> Tuple[int, float]:
    """将时间 t 按项目实际帧率量化到整数帧。"""
    fps_val = max(1.0, float(fps or 24.0))
    frame_idx = int(round(max(0.0, float(t or 0.0)) * fps_val))
    if duration is not None and duration > 0.0:
        max_frame = max(0, int(round(float(duration) * fps_val)))
        frame_idx = max(0, min(max_frame, frame_idx))
        t_q = min(float(duration), float(frame_idx) / fps_val)
    else:
        t_q = float(frame_idx) / fps_val
    return frame_idx, round(t_q, 6)


def s5_step(u: float) -> float:
    """C2 光滑五次 S 形阶跃函数：S5(0)=S5'(0)=S5''(0)=0, S5(1)=1, S5'(1)=S5''(1)=0。"""
    uc = max(0.0, min(1.0, float(u)))
    u3 = uc * uc * uc
    return u3 * (10.0 + uc * (-15.0 + 6.0 * uc))


def _solve_linear_system(mat: List[List[float]], rhs: List[float]) -> Optional[List[float]]:
    """纯 Python 带列主元的高斯消元求解稠密线性方程组（n <= 90 时毫秒级）。"""
    n = len(rhs)
    if n == 0:
        return []
    a = [list(row) for row in mat]
    b = list(rhs)
    for col in range(n):
        piv = col
        max_v = abs(a[col][col])
        for r in range(col + 1, n):
            av = abs(a[r][col])
            if av > max_v:
                max_v = av
                piv = r
        if max_v <= 1e-14:
            a[col][col] += 1e-10
            piv = col
        if piv != col:
            a[col], a[piv] = a[piv], a[col]
            b[col], b[piv] = b[piv], b[col]
        inv_diag = 1.0 / a[col][col]
        row_c = a[col]
        for j in range(col, n):
            row_c[j] *= inv_diag
        b[col] *= inv_diag
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            if abs(factor) <= 1e-15:
                continue
            row_r = a[r]
            for j in range(col, n):
                row_r[j] -= factor * row_c[j]
            b[r] -= factor * b[col]
    return b


def _invert_or_factor_spd(mat: List[List[float]]) -> Optional[List[List[float]]]:
    """用 LDLT/Cholesky 思想分解对称矩阵（直接预求逆小矩阵 KKT，后续每次 ADMM 仅 O(n^2) 乘法）。"""
    n = len(mat)
    if n == 0:
        return []
    aug = [list(mat[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = col
        max_v = abs(aug[col][col])
        for r in range(col + 1, n):
            av = abs(aug[r][col])
            if av > max_v:
                max_v = av
                piv = r
        if max_v <= 1e-13:
            aug[col][col] += 1e-10
        elif piv != col:
            aug[col], aug[piv] = aug[piv], aug[col]
        inv_d = 1.0 / aug[col][col]
        rc = aug[col]
        for j in range(col, 2 * n):
            rc[j] *= inv_d
        for r in range(n):
            if r == col:
                continue
            f = aug[r][col]
            if abs(f) <= 1e-15:
                continue
            rr = aug[r]
            for j in range(col, 2 * n):
                rr[j] -= f * rc[j]
    return [row[n:] for row in aug]


def _bspline_basis_derivs(knots: List[float], t: float, max_deriv: int = 2) -> Tuple[int, List[List[float]]]:
    """
    计算 3 次 Clamped B 样条在 t 处的局部非零基函数及 0..max_deriv 阶导数。
    返回 (span_idx, ders)，其中 ders[d][r] 对应第 span_idx - 3 + r 个控制点的 d 阶导数（r = 0..3）。
    """
    p = 3
    m = len(knots) - 1
    n_ctrl = m - p
    t_start = knots[p]
    t_end = knots[n_ctrl]
    tc = max(t_start, min(t_end, float(t)))
    if tc >= t_end - 1e-12:
        span = n_ctrl - 1
        while span > p and knots[span] >= knots[span + 1] - 1e-12:
            span -= 1
    else:
        lo, hi = p, n_ctrl - 1
        span = (lo + hi) // 2
        while lo <= hi:
            mid = (lo + hi) // 2
            if tc < knots[mid]:
                hi = mid - 1
            elif tc >= knots[mid + 1]:
                lo = mid + 1
            else:
                span = mid
                break

    # Piegl & Tiller Algorithm A2.3
    ndu = [[0.0] * (p + 1) for _ in range(p + 1)]
    ndu[0][0] = 1.0
    left = [0.0] * (p + 1)
    right = [0.0] * (p + 1)
    for j in range(1, p + 1):
        left[j] = tc - knots[span + 1 - j]
        right[j] = knots[span + j] - tc
        saved = 0.0
        for r in range(j):
            ndu[j][r] = right[r + 1] + left[j - r]
            denom = ndu[j][r]
            temp = (ndu[r][j - 1] / denom) if abs(denom) > 1e-15 else 0.0
            ndu[r][j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j][j] = saved

    ders = [[0.0] * (p + 1) for _ in range(max_deriv + 1)]
    for j in range(p + 1):
        ders[0][j] = ndu[j][p]

    a = [[0.0] * (p + 1), [0.0] * (p + 1)]
    for r in range(p + 1):
        s1, s2 = 0, 1
        a[0][0] = 1.0
        for k in range(1, max_deriv + 1):
            d = 0.0
            rk = r - k
            pk = p - k
            if r >= k:
                denom = ndu[pk + 1][rk]
                a[s2][0] = (a[s1][0] / denom) if abs(denom) > 1e-15 else 0.0
                d = a[s2][0] * ndu[rk][pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = (k - 1) if (r - 1 <= pk) else (p - r)
            for j in range(j1, j2 + 1):
                denom = ndu[pk + 1][rk + j]
                a[s2][j] = ((a[s1][j] - a[s1][j - 1]) / denom) if abs(denom) > 1e-15 else 0.0
                d += a[s2][j] * ndu[rk + j][pk]
            if r <= pk:
                denom = ndu[pk + 1][r]
                a[s2][k] = (-a[s1][k - 1] / denom) if abs(denom) > 1e-15 else 0.0
                d += a[s2][k] * ndu[r][pk]
            ders[k][r] = d
            s1, s2 = s2, s1
    fac = float(p)
    for k in range(1, max_deriv + 1):
        for j in range(p + 1):
            ders[k][j] *= fac
        fac *= float(p - k)
    return span, ders


def _eval_bspline_at(knots: List[float], ctrl: List[float], t: float) -> Tuple[float, float, float]:
    """返回 (v(t), a(t)=v'(t), j(t)=v''(t))。"""
    span, ders = _bspline_basis_derivs(knots, t, max_deriv=2)
    v = a = j = 0.0
    for r in range(4):
        c = ctrl[span - 3 + r]
        v += ders[0][r] * c
        a += ders[1][r] * c
        j += ders[2][r] * c
    return v, a, j


def _bspline_to_hermite_segments(knots: List[float], ctrl: List[float]) -> List[Dict[str, Any]]:
    """
    将精确 3 次 B 样条按互异节点区间无损转为标准分段三次 Hermite（完全精确：同为 3 次多项式，且 v>=0、C2 连续）。
    """
    p = 3
    n_ctrl = len(ctrl)
    unique_breaks: List[float] = []
    for k in knots[p:n_ctrl + 1]:
        if not unique_breaks or k > unique_breaks[-1] + 1e-7:
            unique_breaks.append(float(k))
    segs: List[Dict[str, Any]] = []
    for i in range(len(unique_breaks) - 1):
        t0, t1 = unique_breaks[i], unique_breaks[i + 1]
        h = t1 - t0
        if h <= 1e-7:
            continue
        t_mid0 = t0 + 1e-6 * h
        t_mid1 = t1 - 1e-6 * h
        v0, m0, _ = _eval_bspline_at(knots, ctrl, t_mid0)
        v1, m1, _ = _eval_bspline_at(knots, ctrl, t_mid1)
        # 严格由内部四次精确积分，保证与现有 _eval_speed_segments_at_time 一致
        area = 0.0
        for uu, ww in zip(_GL3_U, _GL3_W):
            vv, _, _ = _eval_bspline_at(knots, ctrl, t0 + uu * h)
            area += ww * max(0.0, vv)
        area *= h
        segs.append({
            "kind": "hermite",
            "t0": t0,
            "t1": t1,
            "v0": max(0.0, v0),
            "v1": max(0.0, v1),
            "m0": m0,
            "m1": m1,
            "area": max(0.0, area),
        })
    return segs


def _build_span_knots(
    t_a: float,
    t_b: float,
    interval_bounds: List[float],
    point_times: List[float],
    target_interior_count: int = 24,
) -> List[float]:
    """为移动跨度 [t_a, t_b] 构建包含锁定边界与用户控制点的自适应 clamped 3 次节点向量。"""
    span_dur = max(1e-4, t_b - t_a)
    critical: List[float] = [t_a, t_b]
    for tb in interval_bounds:
        if t_a + 1e-4 < tb < t_b - 1e-4:
            critical.append(float(tb))
    for tp in point_times:
        if t_a + 1e-4 < tp < t_b - 1e-4:
            critical.append(float(tp))
    critical.sort()
    dedup_crit: List[float] = [critical[0]]
    for c in critical[1:]:
        if c - dedup_crit[-1] >= 2e-3:
            dedup_crit.append(c)
    if dedup_crit[-1] < t_b - 1e-4:
        dedup_crit.append(t_b)
    else:
        dedup_crit[-1] = t_b

    # 在每两个关键时间之间补充适当均匀内部节点（每段至少 3 个子段，总内节点控制在 14..42）
    n_sub_total = max(12, min(42, target_interior_count))
    step_target = span_dur / float(n_sub_total)
    all_nodes: List[float] = [t_a]
    for idx in range(len(dedup_crit) - 1):
        a0, a1 = dedup_crit[idx], dedup_crit[idx + 1]
        w = a1 - a0
        sub_n = max(3 if len(dedup_crit) <= 6 else 2, int(round(w / max(0.08, step_target))))
        sub_n = min(14, sub_n)
        for s in range(1, sub_n):
            all_nodes.append(a0 + (w * float(s)) / float(sub_n))
        all_nodes.append(a1)

    # 控制总控制点数量 <= 64 保证高速实时拖拽
    if len(all_nodes) > 58:
        # 优先保留 dedup_crit，按距离稀疏删除非关键网格点
        crit_set = set(round(x, 5) for x in dedup_crit)
        filtered = [all_nodes[0]]
        min_gap = span_dur / 46.0
        for x in all_nodes[1:-1]:
            is_c = round(x, 5) in crit_set
            if is_c or (x - filtered[-1] >= min_gap):
                if is_c and len(filtered) > 1 and (round(filtered[-1], 5) not in crit_set) and (x - filtered[-1] < 0.35 * min_gap):
                    filtered.pop()
                filtered.append(x)
        filtered.append(t_b)
        all_nodes = filtered

    knots = [t_a, t_a, t_a] + all_nodes + [t_b, t_b, t_b]
    return knots


def _max_trapezoid_distance(dt: float, v_max: float, a_max: float, v_left_zero: bool, v_right_zero: bool) -> float:
    """给定时间 dt、速度上限 v_max、加速度上限 a_max 与起止是否零速，解析上界最大可完成位移。"""
    if dt <= 1e-6 or v_max <= 1e-7:
        return 0.0
    if a_max <= 1e-7:
        return 0.0
    t_acc = (v_max / a_max) if v_left_zero else 0.0
    t_dec = (v_max / a_max) if v_right_zero else 0.0
    if t_acc + t_dec <= dt:
        return v_max * dt - 0.5 * v_max * (t_acc + t_dec)
    # 三角或单坡极限
    if v_left_zero and v_right_zero:
        return 0.25 * a_max * dt * dt
    return 0.5 * a_max * dt * dt


def _solve_moving_span_qp(
    t_a: float,
    t_b: float,
    sub_intervals: List[Dict[str, Any]],
    pref_points: List[Dict[str, Any]],
    limits: Dict[str, float],
    bc_left_stop: bool,
    bc_right_stop: bool,
    v_left_fixed: Optional[float] = None,
    v_right_fixed: Optional[float] = None,
    edited_time: Optional[float] = None,
    edited_speed: Optional[float] = None,
    baseline_samples: Optional[List[Tuple[float, float]]] = None,
    curvature_samples: Optional[List[Tuple[float, float]]] = None,
    flat_preference: bool = False,
) -> Dict[str, Any]:
    """
    在连续移动跨度 [t_a, t_b] 上用 3 次 B 样条 + ADMM QP 联合求解所有锁定子区间：
    - 严格满足 sum_{i} A[m,i] c_i = ds_m
    - 严格满足 0 <= c_i <= u_i (含曲率弯道安全上限)
    - 满足起止停顿零速且零加速度 (c0=c1=0 / c_{-1}=c_{-2}=0)
    - 满足 |v'(t)| <= a_max, |v''(t)| <= j_max（可行域内），不可行时准确返回冲突区间与解决建议
    """
    span_dur = max(1e-4, t_b - t_a)
    total_ds = sum(max(0.0, float(iv["delta_s"])) for iv in sub_intervals)
    if total_ds <= 1e-6:
        return {
            "segments": [{
                "kind": "hermite",
                "t0": t_a, "t1": t_b, "v0": 0.0, "v1": 0.0, "m0": 0.0, "m1": 0.0, "area": 0.0,
            }],
            "conflicts": [],
            "clamped": False,
            "achieved_edit_speed": 0.0 if edited_time is not None else None,
            "affected_range": None,
        }

    v_max = max(0.05, float(limits.get("max_speed", 25.0)))
    a_max = max(0.05, float(limits.get("max_accel", 6.5)))
    j_max = max(0.10, float(limits.get("max_jerk", 24.0)))
    an_max = max(0.05, float(limits.get("max_centripetal_accel", 5.0)))

    iv_bounds = [float(iv["t_a"]) for iv in sub_intervals] + [t_b]
    pt_times = [float(p["t"]) for p in pref_points if t_a + 1e-4 < float(p["t"]) < t_b - 1e-4]
    if edited_time is not None and t_a + 1e-4 < edited_time < t_b - 1e-4:
        pt_times.append(float(edited_time))

    knots = _build_span_knots(t_a, t_b, iv_bounds, pt_times, target_interior_count=max(16, min(38, int(round(span_dur * 3.0)))))
    n_ctrl = len(knots) - 4

    # 计算 Greville 横坐标以映射曲率极限与基线速度
    greville = [(knots[i + 1] + knots[i + 2] + knots[i + 3]) / 3.0 for i in range(n_ctrl)]

    # 预先构造每个锁定子区间的精确积分行 A_int[m][0..n_ctrl-1] 和平滑正则矩阵 Q_acc, Q_jerk
    n_iv = len(sub_intervals)
    A_int = [[0.0] * n_ctrl for _ in range(n_iv)]
    Q_acc = [[0.0] * n_ctrl for _ in range(n_ctrl)]
    Q_jerk = [[0.0] * n_ctrl for _ in range(n_ctrl)]

    # 将每个锁定子区间切分到原始唯一节点区间内做 3 点 GL（保证跨锁定边界与节点边界积分机器精度级精确）
    all_u = sorted(set(knots[3:n_ctrl + 1] + [float(iv["t_a"]) for iv in sub_intervals] + [t_b]))
    for seg_k in range(len(all_u) - 1):
        u0, u1 = all_u[seg_k], all_u[seg_k + 1]
        h = u1 - u0
        if h <= 1e-9:
            continue
        t_mid = 0.5 * (u0 + u1)
        m_owner = -1
        for m_idx, iv in enumerate(sub_intervals):
            if float(iv["t_a"]) - 1e-8 <= t_mid <= float(iv["t_b"]) + 1e-8:
                m_owner = m_idx
                break
        for g_u, g_w in zip(_GL3_U, _GL3_W):
            tq = u0 + g_u * h
            span, ders = _bspline_basis_derivs(knots, tq, max_deriv=2)
            wt = g_w * h
            i0 = span - 3
            for r in range(4):
                ci = i0 + r
                if m_owner >= 0:
                    A_int[m_owner][ci] += wt * ders[0][r]
                d1r = ders[1][r]
                d2r = ders[2][r]
                for c in range(4):
                    cj = i0 + c
                    Q_acc[ci][cj] += wt * d1r * ders[1][c]
                    Q_jerk[ci][cj] += wt * d2r * ders[2][c]

    # 物理可行性前置检验（检查各锁定子区间与曲率）
    conflicts: List[Dict[str, Any]] = []
    # 构建基线期望速度函数
    avg_span_v = total_ds / span_dur
    def _eval_baseline_v(tq: float) -> float:
        if baseline_samples:
            if tq <= baseline_samples[0][0]:
                return baseline_samples[0][1]
            if tq >= baseline_samples[-1][0]:
                return baseline_samples[-1][1]
            for j in range(len(baseline_samples) - 1):
                t0, v0 = baseline_samples[j]
                t1, v1 = baseline_samples[j + 1]
                if t0 <= tq <= t1:
                    f = (tq - t0) / max(1e-6, t1 - t0)
                    return v0 + (v1 - v0) * f
        if pref_points:
            pts_s = sorted(pref_points, key=lambda x: float(x["t"]))
            if tq <= float(pts_s[0]["t"]):
                return max(0.0, float(pts_s[0]["v"]))
            if tq >= float(pts_s[-1]["t"]):
                return max(0.0, float(pts_s[-1]["v"]))
            for j in range(len(pts_s) - 1):
                t0, v0 = float(pts_s[j]["t"]), max(0.0, float(pts_s[j]["v"]))
                t1, v1 = float(pts_s[j + 1]["t"]), max(0.0, float(pts_s[j + 1]["v"]))
                if t0 <= tq <= t1:
                    f = (tq - t0) / max(1e-6, t1 - t0)
                    return v0 + (v1 - v0) * f
        return avg_span_v

    # 控制点上限（纳入空间曲率离心加速度约束 v <= sqrt(a_n_max / kappa)）
    u_bounds = [v_max] * n_ctrl
    l_bounds = [0.0] * n_ctrl
    curv_cap = [v_max] * n_ctrl
    curv_time_map: List[Tuple[float, float]] = []
    if curvature_samples:
        # 计算每个子区间起点累计弧长（若调用方未显式传入 s_a 则顺序累加）
        iv_s_starts: List[float] = []
        running_s = float(sub_intervals[0].get("s_a", 0.0)) if sub_intervals else 0.0
        for iv_c in sub_intervals:
            sa_use = float(iv_c["s_a"]) if "s_a" in iv_c else running_s
            iv_s_starts.append(sa_use)
            running_s = sa_use + max(0.0, float(iv_c["delta_s"]))
        cur_iv_idx = 0
        for i in range(n_ctrl):
            tg = greville[i]
            while cur_iv_idx < n_iv - 1 and tg > float(sub_intervals[cur_iv_idx]["t_b"]) + 1e-6:
                cur_iv_idx += 1
            iv_c = sub_intervals[cur_iv_idx]
            dt_c = max(1e-5, float(iv_c["t_b"]) - float(iv_c["t_a"]))
            frac = max(0.0, min(1.0, (tg - float(iv_c["t_a"])) / dt_c))
            s_est = iv_s_starts[cur_iv_idx] + frac * float(iv_c["delta_s"])
            kappa_loc = 0.0
            for sk, kp in curvature_samples:
                if abs(sk - s_est) <= 0.45:
                    kappa_loc = max(kappa_loc, kp)
            if kappa_loc > 1e-4:
                v_safe = math.sqrt(an_max / kappa_loc)
                if v_safe < curv_cap[i]:
                    curv_cap[i] = max(0.05, v_safe)
                    curv_time_map.append((i, tg, curv_cap[i], float(iv_c["delta_s"]) / dt_c))
                if v_safe < u_bounds[i]:
                    u_bounds[i] = max(0.05, v_safe)

    # 边界速度与零加速度固定值
    fixed_idx: Dict[int, float] = {}
    if bc_left_stop:
        fixed_idx[0] = 0.0
        if n_ctrl >= 3:
            fixed_idx[1] = 0.0
    elif v_left_fixed is not None:
        fixed_idx[0] = max(0.0, min(v_max, float(v_left_fixed)))

    if bc_right_stop:
        fixed_idx[n_ctrl - 1] = 0.0
        if n_ctrl >= 3:
            fixed_idx[n_ctrl - 2] = 0.0
    elif v_right_fixed is not None:
        fixed_idx[n_ctrl - 1] = max(0.0, min(v_max, float(v_right_fixed)))

    for idx_f, val_f in fixed_idx.items():
        l_bounds[idx_f] = val_f
        u_bounds[idx_f] = val_f

    # 分区间前置物理极限 check
    for m_idx, iv in enumerate(sub_intervals):
        ta_m, tb_m = float(iv["t_a"]), float(iv["t_b"])
        dt_m = max(1e-4, tb_m - ta_m)
        ds_m = max(0.0, float(iv["delta_s"]))
        vz_l = bc_left_stop and (m_idx == 0)
        vz_r = bc_right_stop and (m_idx == n_iv - 1)
        # 检查曲率约束下的该区间有效速度上界
        iv_u_vals = [u_bounds[i] for i in range(n_ctrl) if ta_m - 1e-3 <= greville[i] <= tb_m + 1e-3]
        eff_iv_vmax = min(v_max, max(iv_u_vals) if iv_u_vals else v_max)
        # 保守物理上限（允许 B 样条内 C2 减速峰值留 3% 数值余量）
        cap_ds = _max_trapezoid_distance(dt_m, eff_iv_vmax, a_max, vz_l, vz_r)
        if ds_m > cap_ds + 0.02:
            req_avg = ds_m / dt_m
            reason_parts = []
            if req_avg > v_max:
                reason_parts.append(f"所需均速 {req_avg:.2f} > 速度上限 {v_max:.2f}")
            else:
                reason_parts.append(f"在当前加减速上限 ({a_max:.1f}m/s²) 与起停零速下最多行进 {cap_ds:.2f} < 目标 {ds_m:.2f}")
            req_dur_est = (ds_m / max(0.15, eff_iv_vmax * 0.85)) + (eff_iv_vmax / max(0.2, a_max)) * (0.5 if vz_l else 0.0 + 0.5 if vz_r else 0.0)
            conflicts.append({
                "type": "constraint_infeasible",
                "interval": [round(ta_m, 3), round(tb_m, 3)],
                "required_length": round(ds_m, 3),
                "max_feasible_length": round(cap_ds, 3),
                "required_avg_speed": round(req_avg, 3),
                "limit_speed": round(eff_iv_vmax, 3),
                "limit_accel": round(a_max, 3),
                "message": (
                    f"构图时空/运动学冲突 [{ta_m:.2f}s–{tb_m:.2f}s]：" + "；".join(reason_parts)
                ),
                "options": [
                    {"id": "extend_duration", "label": "延长时间", "suggested_duration_s": round(max(dt_m + 0.3, req_dur_est), 2), "detail": f"将该段耗时从 {dt_m:.2f}s 延长至约 {max(dt_m + 0.3, req_dur_est):.2f}s"},
                    {"id": "unlock_boundary", "label": "放宽到达时刻", "detail": "解锁中间构图锚点时刻，由系统按连续运动重排到达时机"},
                    {"id": "raise_limit", "label": "提高运动上限", "suggested_max_speed": round(max(v_max, req_avg * 1.25), 1), "detail": f"确认将速度/加速度上限提高至可行范围"},
                ],
            })

    # 采样 1 阶与 2 阶导数配点（每段中点与节点），用于 ADMM 限加速度/限加加速度
    colloc_ts: List[float] = []
    for seg_k in range(len(all_u) - 1):
        u0, u1 = all_u[seg_k], all_u[seg_k + 1]
        if u1 - u0 > 1e-5:
            colloc_ts.append(0.5 * (u0 + u1))
    # 确保不超过 ~64 个配点
    if len(colloc_ts) > 64:
        stride = max(2, len(colloc_ts) // 48)
        colloc_ts = colloc_ts[::stride]
    n_col = len(colloc_ts)
    D1 = [[0.0] * n_ctrl for _ in range(n_col)]
    D2 = [[0.0] * n_ctrl for _ in range(n_col)]
    for q_idx, tq in enumerate(colloc_ts):
        span, ders = _bspline_basis_derivs(knots, tq, max_deriv=2)
        i0 = span - 3
        for r in range(4):
            D1[q_idx][i0 + r] = ders[1][r]
            D2[q_idx][i0 + r] = ders[2][r]

    # 构造目标二次型：1/2 c^T H c - g^T c
    H = [[0.0] * n_ctrl for _ in range(n_ctrl)]
    g = [0.0] * n_ctrl

    # 1) 加速度与 Jerk 能量最小化（按尺度自适应标度）
    scale_v = max(0.3, avg_span_v)
    w_acc = (0.35 if flat_preference else 0.18) / (scale_v * scale_v)
    w_jerk = 0.035 / (scale_v * scale_v)
    for i in range(n_ctrl):
        for j in range(n_ctrl):
            H[i][j] += w_acc * Q_acc[i][j] + w_jerk * Q_jerk[i][j]

    # 2) 未编辑区域保留原速基线（随距离 editTime 高斯局部放开邻近区以自然联动平滑适应）
    sigma_loc = max(0.9, min(3.2, span_dur * 0.28))
    trailing_low_drag = (
        edited_time is not None
        and edited_speed is not None
        and bc_right_stop
        and float(edited_speed) <= max(0.18, avg_span_v * 0.25)
        and float(edited_time) >= t_b - max(3.2, span_dur * 0.35)
    )
    for k_i in range(len(all_u) - 1):
        u0, u1 = all_u[k_i], all_u[k_i + 1]
        h = u1 - u0
        if h <= 1e-7:
            continue
        for g_u, g_w in zip(_GL3_U, _GL3_W):
            tq = u0 + g_u * h
            v_b = max(0.0, _eval_baseline_v(tq))
            if edited_time is not None:
                te_f = float(edited_time)
                dist_e = abs(tq - te_f)
                # 邻域内放开以自动吸纳面积差与 C2 平滑过渡，邻域外紧锁原有节奏
                loc_factor = 1.0 - 0.88 * math.exp(-0.5 * (dist_e / sigma_loc) ** 2)
                w_loc = (0.04 + 1.8 * loc_factor) * (g_w * h)
                # 若用户在末段减速进入停顿区间把某点拉至近零速，抑制后半段基线“回拉反弹”
                if trailing_low_drag and tq >= te_f:
                    rem_frac = max(0.0, min(1.0, (t_b - tq) / max(0.15, t_b - te_f)))
                    v_b = min(v_b, max(0.0, float(edited_speed)) * rem_frac)
                    w_loc = max(w_loc, 6.0 * (g_w * h))
            else:
                w_loc = (1.6 if flat_preference else 0.35) * (g_w * h)
            span, ders = _bspline_basis_derivs(knots, tq, max_deriv=0)
            i0 = span - 3
            for r in range(4):
                br = ders[0][r]
                ci = i0 + r
                g[ci] += w_loc * v_b * br
                for c in range(4):
                    H[ci][i0 + c] += w_loc * br * ders[0][c]

    # 3) 用户显式控制点与拖拽意图匹配
    for pt in pref_points:
        tp = float(pt.get("t", 0.0))
        if tp < t_a - 1e-4 or tp > t_b + 1e-4:
            continue
        vp = max(0.0, float(pt.get("v", 0.0)))
        is_edit_pt = edited_time is not None and abs(tp - float(edited_time)) <= 1e-3
        if is_edit_pt and edited_speed is not None:
            vp = max(0.0, float(edited_speed))
            w_pt = 85.0
        else:
            if edited_time is not None:
                # 非当前拖动点：按距拖点距离保留（近处权重降低从而自然联动平滑）
                d_rel = abs(tp - float(edited_time))
                w_pt = 0.25 + 3.5 * (1.0 - math.exp(-0.5 * (d_rel / sigma_loc) ** 2))
                if trailing_low_drag and tp >= float(edited_time):
                    rem_frac = max(0.0, min(1.0, (t_b - tp) / max(0.15, t_b - float(edited_time))))
                    vp = min(vp, max(0.0, float(edited_speed)) * rem_frac)
                    w_pt = max(w_pt, 12.0)
            else:
                w_pt = 4.0
        if pt.get("pin_v"):
            w_pt = max(w_pt, 60.0)
        span, ders = _bspline_basis_derivs(knots, max(t_a, min(t_b, tp)), max_deriv=0)
        i0 = span - 3
        for r in range(4):
            br = ders[0][r]
            ci = i0 + r
            g[ci] += w_pt * vp * br
            for c in range(4):
                H[ci][i0 + c] += w_pt * br * ders[0][c]

    if edited_time is not None and t_a - 1e-4 <= float(edited_time) <= t_b + 1e-4 and edited_speed is not None:
        # 即使 edited_time 未出现在 pref_points 中也显式加上强拟合项
        if not any(abs(float(p.get("t", -99)) - float(edited_time)) <= 1e-3 for p in pref_points):
            span, ders = _bspline_basis_derivs(knots, max(t_a, min(t_b, float(edited_time))), max_deriv=0)
            i0 = span - 3
            vp = max(0.0, float(edited_speed))
            w_pt = 85.0
            for r in range(4):
                br = ders[0][r]
                ci = i0 + r
                g[ci] += w_pt * vp * br
                for c in range(4):
                    H[ci][i0 + c] += w_pt * br * ders[0][c]

    # ADMM 求解：
    # min 1/2 c^T H c - g^T c  s.t.  A_int c = ds, z_c = c ∈ [l, u], z_a = D1 c ∈ [-a_eff, a_eff], z_j = D2 c ∈ [-j_eff, j_eff]
    # 若前置可行性已探测冲突（硬界无法同时满足全部区间积分），在冲突求解分支略放宽盒约束/导数上限以便输出可视化预览并如实保留冲突标志
    has_hard_infeasible = len(conflicts) > 0
    a_eff = a_max if not has_hard_infeasible else max(a_max * 4.0, (total_ds / max(0.1, span_dur)) * 4.0)
    j_eff = j_max if not has_hard_infeasible else max(j_max * 6.0, a_eff * 6.0)
    u_eff = list(u_bounds)
    if has_hard_infeasible:
        # 按各自所在区间的平均速率给足求解空间以防无解，但对急弯控制点尽量收紧在较低上限内并保留 conflicts 警告
        for i in range(n_ctrl):
            tg = greville[i]
            iv_m = next((iv for iv in sub_intervals if float(iv["t_a"]) - 1e-4 <= tg <= float(iv["t_b"]) + 1e-4), sub_intervals[-1])
            req_v_iv = float(iv_m["delta_s"]) / max(0.05, float(iv_m["t_b"]) - float(iv_m["t_a"]))
            if curv_cap[i] < v_max - 1e-3:
                u_eff[i] = max(curv_cap[i], min(v_max, req_v_iv * 1.15))
            else:
                u_eff[i] = max(u_bounds[i], req_v_iv * 2.6, 0.5)
        for idx_f, val_f in fixed_idx.items():
            u_eff[idx_f] = val_f

    # 3b) 弯道曲率安全减速引导项（仅在冲突放宽分支且属于急弯减速区时生效，绝不对慢速镜头把轻缓弯道强拉到高速上限）
    if has_hard_infeasible:
        for idx_c, tg_c, v_safe_c, req_v_c in curv_time_map:
            if idx_c in fixed_idx:
                continue
            if not (v_safe_c < u_eff[idx_c] - 1e-3 and v_safe_c < max(0.25, req_v_c * 0.95)):
                continue
            span_c, ders_c = _bspline_basis_derivs(knots, max(t_a, min(t_b, tg_c)), max_deriv=0)
            i0_c = span_c - 3
            w_curv = 18.0
            for r in range(4):
                br = ders_c[0][r]
                ci = i0_c + r
                g[ci] += w_curv * v_safe_c * br
                for c in range(4):
                    H[ci][i0_c + c] += w_curv * br * ders_c[0][c]

    # 微弱对角正则保证绝对正定
    for i in range(n_ctrl):
        H[i][i] += 1e-5

    rho_c = 6.0
    rho_d1 = 1.2
    rho_d2 = 0.15

    # 构造 ADMM 增广 KKT 矩阵：[ H + rho_c I + rho_d1 D1^T D1 + rho_d2 D2^T D2, A_int^T ; A_int, 0 ]
    dim_kkt = n_ctrl + n_iv
    M_kkt = [[0.0] * dim_kkt for _ in range(dim_kkt)]
    for i in range(n_ctrl):
        for j in range(n_ctrl):
            v_ij = H[i][j] + (rho_c if i == j else 0.0)
            if rho_d1 > 0.0:
                s1 = 0.0
                for q in range(n_col):
                    s1 += D1[q][i] * D1[q][j]
                v_ij += rho_d1 * s1
            if rho_d2 > 0.0:
                s2 = 0.0
                for q in range(n_col):
                    s2 += D2[q][i] * D2[q][j]
                v_ij += rho_d2 * s2
            M_kkt[i][j] = v_ij
    for m_idx in range(n_iv):
        for i in range(n_ctrl):
            M_kkt[i][n_ctrl + m_idx] = A_int[m_idx][i]
            M_kkt[n_ctrl + m_idx][i] = A_int[m_idx][i]
        M_kkt[n_ctrl + m_idx][n_ctrl + m_idx] = -1e-11  # tiny saddle stabilizer

    K_inv = _invert_or_factor_spd(M_kkt)
    ds_targets = [max(0.0, float(iv["delta_s"])) for iv in sub_intervals]

    c_vec = [min(u_eff[i], max(l_bounds[i], avg_span_v)) for i in range(n_ctrl)]
    z_c = list(c_vec)
    y_c = [0.0] * n_ctrl
    z_a = [0.0] * n_col
    y_a = [0.0] * n_col
    z_j = [0.0] * n_col
    y_j = [0.0] * n_col

    rhs = [0.0] * dim_kkt
    for m_idx in range(n_iv):
        rhs[n_ctrl + m_idx] = ds_targets[m_idx]

    max_iters = 75
    for it in range(max_iters):
        for i in range(n_ctrl):
            rhs[i] = g[i] + rho_c * (z_c[i] - y_c[i])
        if n_col > 0:
            for q in range(n_col):
                va_q = rho_d1 * (z_a[q] - y_a[q])
                vj_q = rho_d2 * (z_j[q] - y_j[q])
                if abs(va_q) > 1e-15 or abs(vj_q) > 1e-15:
                    for i in range(n_ctrl):
                        rhs[i] += va_q * D1[q][i] + vj_q * D2[q][i]

        # x = K_inv @ rhs
        c_next = [0.0] * n_ctrl
        for i in range(n_ctrl):
            row_inv = K_inv[i]
            s_val = 0.0
            for j in range(dim_kkt):
                s_val += row_inv[j] * rhs[j]
            c_next[i] = s_val

        # z-updates & dual updates + residual check
        max_primal_res = 0.0
        max_dual_res = 0.0
        for i in range(n_ctrl):
            raw_zc = c_next[i] + y_c[i]
            new_zc = max(l_bounds[i], min(u_eff[i], raw_zc))
            pr = abs(c_next[i] - new_zc)
            dr = abs(new_zc - z_c[i])
            if pr > max_primal_res:
                max_primal_res = pr
            if dr > max_dual_res:
                max_dual_res = dr
            z_c[i] = new_zc
            y_c[i] += c_next[i] - new_zc

        for q in range(n_col):
            d1_c = sum(D1[q][i] * c_next[i] for i in range(n_ctrl))
            d2_c = sum(D2[q][i] * c_next[i] for i in range(n_ctrl))
            new_za = max(-a_eff, min(a_eff, d1_c + y_a[q]))
            new_zj = max(-j_eff, min(j_eff, d2_c + y_j[q]))
            max_primal_res = max(max_primal_res, abs(d1_c - new_za), abs(d2_c - new_zj))
            max_dual_res = max(max_dual_res, abs(new_za - z_a[q]), abs(new_zj - z_j[q]))
            z_a[q] = new_za
            y_a[q] += d1_c - new_za
            z_j[q] = new_zj
            y_j[q] += d2_c - new_zj

        c_vec = c_next
        if it >= 12 and max_primal_res <= 1e-4 and max_dual_res <= 1e-4:
            break

    # 使用精确投影后的 z_c 并微调严格非负且精确匹配每个锁定区间的积分（消掉任何 1e-4 残差）
    c_final = [max(l_bounds[i], min(u_eff[i], z_c[i])) for i in range(n_ctrl)]
    # 针对 A_int c_final = ds_targets 做零空间最小范数修正，同时保持非负与固定端点不变
    free_mask = [0.0 if (i in fixed_idx) else 1.0 for i in range(n_ctrl)]
    for _ in range(5):
        err_iv = [ds_targets[m_idx] - sum(A_int[m_idx][i] * c_final[i] for i in range(n_ctrl)) for m_idx in range(n_iv)]
        if max((abs(e) for e in err_iv), default=0.0) <= 1e-6:
            break
        # 求解 (A W A^T) lam = err_iv, dc = W A^T lam, W_ii = free_mask[i] * (c_final[i] > 1e-6 or ...)
        W = [
            free_mask[i] * (1.0 if (l_bounds[i] < u_eff[i] - 1e-6) else 0.0)
            for i in range(n_ctrl)
        ]
        M_iv = [[0.0] * n_iv for _ in range(n_iv)]
        for r in range(n_iv):
            for c in range(n_iv):
                s_val = 0.0
                for i in range(n_ctrl):
                    s_val += A_int[r][i] * W[i] * A_int[c][i]
                M_iv[r][c] = s_val
            M_iv[r][r] += 1e-12
        lam = _solve_linear_system(M_iv, err_iv)
        if lam is None:
            break
        for i in range(n_ctrl):
            if W[i] > 0.0:
                delta_ci = W[i] * sum(A_int[m_idx][i] * lam[m_idx] for m_idx in range(n_iv))
                c_final[i] = max(l_bounds[i], c_final[i] + delta_ci)

    # 检测拖动意图是否被可行域限幅，以及邻域影响范围
    achieved_edit_v: Optional[float] = None
    was_clamped = False
    if edited_time is not None and t_a - 1e-4 <= float(edited_time) <= t_b + 1e-4:
        achieved_edit_v, _, _ = _eval_bspline_at(knots, c_final, float(edited_time))
        achieved_edit_v = max(0.0, achieved_edit_v)
        if edited_speed is not None and abs(achieved_edit_v - float(edited_speed)) > max(0.03, 0.04 * max(0.5, float(edited_speed))):
            was_clamped = True

    affected_range: Optional[Tuple[float, float]] = None
    if edited_time is not None and baseline_samples:
        aff_lo, aff_hi = None, None
        for i in range(101):
            ts = t_a + (span_dur * i) / 100.0
            v_now, _, _ = _eval_bspline_at(knots, c_final, ts)
            v_old = _eval_baseline_v(ts)
            if abs(v_now - v_old) > max(0.03, 0.04 * max(0.3, v_old)):
                aff_lo = ts if aff_lo is None else min(aff_lo, ts)
                aff_hi = ts if aff_hi is None else max(aff_hi, ts)
        if aff_lo is not None and aff_hi is not None:
            affected_range = (round(aff_lo, 2), round(aff_hi, 2))

    # 若求解后自身检测到超限加速度或速度，且不在硬冲突里，补发冲突报告
    if has_hard_infeasible:
        pass
    else:
        for i in range(81):
            ts = t_a + (span_dur * i) / 80.0
            vv, aa, _ = _eval_bspline_at(knots, c_final, ts)
            if vv > v_max + 0.08 or abs(aa) > a_max + 0.15:
                iv_hit = next((iv for iv in sub_intervals if float(iv["t_a"]) - 1e-4 <= ts <= float(iv["t_b"]) + 1e-4), sub_intervals[0])
                tag = (round(float(iv_hit["t_a"]), 3), round(float(iv_hit["t_b"]), 3))
                if not any(tuple(c.get("interval", [])) == tag for c in conflicts):
                    conflicts.append({
                        "type": "kinematic_limit_exceeded",
                        "interval": list(tag),
                        "required_peak_speed": round(vv, 3),
                        "required_accel": round(abs(aa), 3),
                        "limit_speed": round(v_max, 3),
                        "limit_accel": round(a_max, 3),
                        "message": f"区间 [{tag[0]:.2f}s–{tag[1]:.2f}s] 约束冲突：峰值速度 {vv:.2f} / 加速度 {abs(aa):.2f} 超出上限（{v_max:.1f} / {a_max:.1f}）",
                        "options": [
                            {"id": "extend_duration", "label": "延长时间", "detail": "延长冲突区间的到达时间以降低所需加减速"},
                            {"id": "unlock_boundary", "label": "放宽到达时刻", "detail": "解锁相邻构图边界，由系统联合优化到达时刻"},
                            {"id": "raise_limit", "label": "提高运动上限", "detail": "确认提升速度/加速度上限并记录例外"},
                        ],
                    })

    segments = _bspline_to_hermite_segments(knots, c_final)
    return {
        "segments": segments,
        "conflicts": conflicts,
        "clamped": was_clamped,
        "achieved_edit_speed": round(achieved_edit_v, 5) if achieved_edit_v is not None else None,
        "affected_range": list(affected_range) if affected_range else None,
        "knots": knots,
        "ctrl": c_final,
    }


def _solve_constant_speed_span(
    t_a: float,
    t_b: float,
    sub_intervals: List[Dict[str, Any]],
    limits: Dict[str, float],
    bc_left_stop: bool,
    bc_right_stop: bool,
    curvature_samples: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """
    保持匀速模式求解：
    - 当全部锁定子区间平均速率一致（或仅单个连续移动区间）时：使用 S5 启停平滑过渡 + 严格恒定巡航段，
      明确标注启停缓冲区间，绝不虚假声称含启停的全程严格恒速。
    - 若相邻锁定子区间强行指定了彼此矛盾的平均速率，按冲突报告并做 C2 光滑桥接（绝不边界阶跃跳速）。
    """
    span_dur = max(1e-4, t_b - t_a)
    total_ds = sum(max(0.0, float(iv["delta_s"])) for iv in sub_intervals)
    if total_ds <= 1e-6:
        return {
            "segments": [{"kind": "hermite", "t0": t_a, "t1": t_b, "v0": 0.0, "v1": 0.0, "m0": 0.0, "m1": 0.0, "area": 0.0}],
            "conflicts": [],
            "cruise_speed": 0.0,
            "ramp_in_s": 0.0,
            "ramp_out_s": 0.0,
            "clamped": False,
        }

    v_max = max(0.05, float(limits.get("max_speed", 25.0)))
    a_max = max(0.05, float(limits.get("max_accel", 6.5)))
    an_max = max(0.05, float(limits.get("max_centripetal_accel", 5.0)))
    conflicts: List[Dict[str, Any]] = []

    iv_speeds = [
        max(0.0, float(iv["delta_s"])) / max(1e-4, float(iv["t_b"]) - float(iv["t_a"]))
        for iv in sub_intervals
    ]
    mean_v = total_ds / span_dur

    # 检查是否存在多个互不兼容速率的锁定子区间
    multi_locked_mismatch = False
    if len(sub_intervals) >= 2:
        # 考虑首末起停折算有效时长后再比对一致性
        sp_spread = max(iv_speeds) - min(iv_speeds)
        if sp_spread > max(0.04, 0.08 * max(0.2, mean_v)):
            multi_locked_mismatch = True

    if multi_locked_mismatch:
        # 用平板偏好的 B 样条求解连续 C2 过渡，同时明确报告保持匀速模式与锁定区间速率冲突
        pref_pts: List[Dict[str, Any]] = []
        for idx, iv in enumerate(sub_intervals):
            mid_t = 0.5 * (float(iv["t_a"]) + float(iv["t_b"]))
            pref_pts.append({"t": mid_t, "v": iv_speeds[idx]})
        qp_res = _solve_moving_span_qp(
            t_a, t_b, sub_intervals, pref_pts, limits, bc_left_stop, bc_right_stop,
            curvature_samples=curvature_samples, flat_preference=True,
        )
        conflicts = list(qp_res["conflicts"])
        conflicts.append({
            "type": "constant_speed_locked_conflict",
            "interval": [round(t_a, 3), round(t_b, 3)],
            "message": f"保持匀速模式下多段锁定构图所需平均速率不一致（{min(iv_speeds):.2f}–{max(iv_speeds):.2f}），已生成平滑过渡曲线（可解锁中间构图重排时间或切为平滑变速）",
            "options": [
                {"id": "reschedule_unlocked", "label": "按确认速度重排中间时刻", "detail": "解锁中间构图时间以保持恒速"},
                {"id": "switch_smooth_variable", "label": "切换为平滑变速", "detail": "保留各锁定到达时刻，由平滑变速连续过渡"},
            ],
        })
        return {
            "segments": qp_res["segments"],
            "conflicts": conflicts,
            "cruise_speed": mean_v,
            "ramp_in_s": 0.0,
            "ramp_out_s": 0.0,
            "clamped": False,
        }

    # 单区间或速率兼容的恒速巡航：构造 C2 光滑 S5 启停 + 严格恒速巡航段
    def _ramp_dur(active: bool) -> float:
        if not active:
            return 0.0
        v_est = mean_v * 1.15
        return max(min(0.18 * span_dur, 0.45), min(0.26 * span_dur, max(0.35, 1.9 * v_est / max(0.5, a_max))))

    r_in = _ramp_dur(bc_left_stop)
    r_out = _ramp_dur(bc_right_stop)
    if r_in + r_out > 0.72 * span_dur:
        sc_r = (0.72 * span_dur) / max(1e-5, r_in + r_out)
        r_in *= sc_r
        r_out *= sc_r

    eff_dt = max(1e-4, span_dur - 0.5 * (r_in + r_out))
    v_cruise = total_ds / eff_dt

    # 校验单区间内部速度、加速度上限及弯道向心加速度
    if v_cruise > v_max + 0.03:
        conflicts.append({
            "type": "constant_speed_over_limit",
            "interval": [round(t_a, 3), round(t_b, 3)],
            "required_speed": round(v_cruise, 3),
            "limit_speed": round(v_max, 3),
            "message": f"保持匀速所需巡航速度 {v_cruise:.2f} 超出上限 {v_max:.2f}",
            "options": [
                {"id": "extend_duration", "label": "延长时间", "suggested_duration_s": round(total_ds / max(0.1, v_max * 0.92) + 0.5 * (r_in + r_out), 2)},
                {"id": "raise_limit", "label": "提高运动上限", "suggested_max_speed": round(v_cruise * 1.15, 1)},
            ],
        })

    if curvature_samples:
        max_k = max((kp for _, kp in curvature_samples), default=0.0)
        if max_k > 1e-4 and (v_cruise * v_cruise * max_k) > an_max + 0.15:
            # 弯道处若恒定速率会超过向心加速度安全上限，转为带曲率限速的 C2 求解（安全过弯减速 + 冲突说明）
            qp_res = _solve_moving_span_qp(
                t_a, t_b, sub_intervals, [{"t": 0.5 * (t_a + t_b), "v": v_cruise}],
                limits, bc_left_stop, bc_right_stop,
                curvature_samples=curvature_samples, flat_preference=True,
            )
            conflicts.extend(qp_res["conflicts"])
            conflicts.append({
                "type": "corner_centripetal_conflict",
                "interval": [round(t_a, 3), round(t_b, 3)],
                "message": f"路径弯道向心加速度 ({v_cruise*v_cruise*max_k:.2f}m/s²) 超出安全上限 ({an_max:.1f}m/s²)，已在弯道处安全减速",
                "options": [
                    {"id": "switch_smooth_variable", "label": "切为平滑变速过弯", "detail": "允许弯道前自动降速、出弯回速"},
                    {"id": "extend_duration", "label": "放慢目标速率/延长时间", "detail": "降低巡航速度以匀速安全通过弯道"},
                ],
            })
            return {
                "segments": qp_res["segments"],
                "conflicts": conflicts,
                "cruise_speed": v_cruise,
                "ramp_in_s": r_in,
                "ramp_out_s": r_out,
                "clamped": False,
            }

    # 用精细 3 次 B 样条或 S5 分段生成 exact Hermite（S5 函数与多项式逼近或精确积分采样片完全一致）
    # 注意：五次多项式 S5 若拆成 6 个小 Hermite/B样条节点即可精确至 <1e-5 并维持 strict C2！
    # 而纯恒速巡航段本身就是 1 段直线 Hermite (v0=v1=v_cruise, m0=m1=0) -> 100.0000% 严格常数！
    segs: List[Dict[str, Any]] = []
    t_cur = t_a
    if r_in > 1e-4:
        n_sub = 6
        for s in range(n_sub):
            u0 = float(s) / n_sub
            u1 = float(s + 1) / n_sub
            ta_s = t_a + u0 * r_in
            tb_s = t_a + u1 * r_in
            hs = tb_s - ta_s
            v0 = v_cruise * s5_step(u0)
            v1 = v_cruise * s5_step(u1)
            dv0 = v_cruise * (30.0 * (u0 * (1.0 - u0)) ** 2) / r_in
            dv1 = v_cruise * (30.0 * (u1 * (1.0 - u1)) ** 2) / r_in
            # 精确求 S5 积分: int_0^u S5(x) dx = 2.5 u^4 - 3 u^5 + u^6
            def _int_s5(uu: float) -> float:
                u4 = uu ** 4
                return 2.5 * u4 - 3.0 * u4 * uu + u4 * uu * uu
            area_s = v_cruise * r_in * (_int_s5(u1) - _int_s5(u0))
            segs.append({
                "kind": "hermite", "t0": ta_s, "t1": tb_s,
                "v0": v0, "v1": v1, "m0": dv0, "m1": dv1, "area": area_s,
                "is_transition_ramp": True,
            })
        t_cur = t_a + r_in

    t_cruise_end = t_b - r_out
    if t_cruise_end > t_cur + 1e-5:
        h_c = t_cruise_end - t_cur
        segs.append({
            "kind": "hermite", "t0": t_cur, "t1": t_cruise_end,
            "v0": v_cruise, "v1": v_cruise, "m0": 0.0, "m1": 0.0,
            "area": v_cruise * h_c,
            "is_constant_cruise": True,
        })
        t_cur = t_cruise_end

    if r_out > 1e-4:
        n_sub = 6
        for s in range(n_sub):
            u0 = float(s) / n_sub
            u1 = float(s + 1) / n_sub
            ta_s = t_cur + u0 * r_out
            tb_s = t_cur + u1 * r_out
            v0 = v_cruise * (1.0 - s5_step(u0))
            v1 = v_cruise * (1.0 - s5_step(u1))
            dv0 = -v_cruise * (30.0 * (u0 * (1.0 - u0)) ** 2) / r_out
            dv1 = -v_cruise * (30.0 * (u1 * (1.0 - u1)) ** 2) / r_out
            def _int_s5_inv(uu: float) -> float:
                u4 = uu ** 4
                return uu - (2.5 * u4 - 3.0 * u4 * uu + u4 * uu * uu)
            area_s = v_cruise * r_out * (_int_s5_inv(u1) - _int_s5_inv(u0))
            segs.append({
                "kind": "hermite", "t0": ta_s, "t1": tb_s,
                "v0": max(0.0, v0), "v1": max(0.0, v1), "m0": dv0, "m1": dv1, "area": max(0.0, area_s),
                "is_transition_ramp": True,
            })

    # 微调保证严格总面积 = total_ds
    tot_area = sum(s["area"] for s in segs)
    if tot_area > 1e-8 and abs(tot_area - total_ds) > 1e-7:
        fac = total_ds / tot_area
        for s in segs:
            s["v0"] *= fac
            s["v1"] *= fac
            s["m0"] *= fac
            s["m1"] *= fac
            s["area"] *= fac
        v_cruise *= fac

    return {
        "segments": segs,
        "conflicts": conflicts,
        "cruise_speed": v_cruise,
        "ramp_in_s": r_in,
        "ramp_out_s": r_out,
        "clamped": False,
    }


def solve_continuous_velocity_profile(
    duration: float,
    intervals_in: List[Dict[str, Any]],
    points_in: Optional[List[Dict[str, Any]]] = None,
    mode: str = "smooth_variable",
    limits: Optional[Dict[str, float]] = None,
    bc_start_stop: bool = True,
    bc_end_stop: bool = True,
    edited_time: Optional[float] = None,
    edited_speed: Optional[float] = None,
    baseline_samples: Optional[List[Tuple[float, float]]] = None,
    curvature_samples: Optional[List[Tuple[float, float]]] = None,
    lock_compositions: bool = True,
) -> Dict[str, Any]:
    """
    统一联合求解多锁定区间（或单连续跨度）的速度曲线：
    1. 将全片划分为“静止/停顿段（v=0）”与若干个“连续移动跨度（由真正的停顿点/静止段切开，绝不因普通锁定边界切断！）”；
    2. 在每个连续移动跨度内部：
       - 对 constant_speed 使用 S5 平滑起停 + 严格恒速巡航；
       - 对 smooth_variable 使用带边界条件的高阶 B 样条约束 QP（ADMM）联合求解相邻所有锁定子区间，
         确保任意锁定边界左右速度与加速度连续、速度非负、每段面积严格等于 delta_s_m。
    """
    dur = max(0.1, float(duration))
    lim = dict(DEFAULT_PROTECTION_LIMITS["camera"])
    if limits:
        lim.update({k: float(v) for k, v in limits.items() if v is not None})

    if edited_time is not None and not baseline_samples:
        base_pre = solve_continuous_velocity_profile(
            duration=dur,
            intervals_in=intervals_in,
            points_in=points_in,
            mode=mode,
            limits=lim,
            bc_start_stop=bc_start_stop,
            bc_end_stop=bc_end_stop,
            edited_time=None,
            edited_speed=None,
            baseline_samples=None,
            curvature_samples=curvature_samples,
            lock_compositions=lock_compositions,
        )
        n_bs = max(60, min(200, int(round(dur * 10.0))))
        baseline_samples = [
            ((dur * k) / float(n_bs), eval_segments_at_time(base_pre["segments"], (dur * k) / float(n_bs))[0])
            for k in range(n_bs + 1)
        ]

    pts = sorted([dict(p) for p in (points_in or []) if isinstance(p, dict)], key=lambda x: float(x.get("t", 0.0)))
    # 提取显式停顿（hold_duration > 0 或 explicit stop 点）
    explicit_holds: List[Tuple[float, float]] = []
    explicit_stop_times: List[float] = []
    for p in pts:
        pt_t = max(0.0, min(dur, float(p.get("t", 0.0))))
        hd = max(0.0, float(p.get("hold_duration", p.get("hold", 0.0)) or 0.0))
        if hd > 1e-4 and pt_t < dur - 1e-4:
            explicit_holds.append((pt_t, min(dur, pt_t + hd)))
        elif bool(p.get("stop")) and 1e-3 < pt_t < dur - 1e-3:
            explicit_stop_times.append(pt_t)

    # 先将含 hold 的区间劈分成纯静止段与移动段；若某锁定区间 delta_s <= 1e-6 本身也是静止停顿平台
    base_ivs: List[Dict[str, Any]] = []
    for iv in intervals_in:
        ta = float(iv["t_a"])
        tb = float(iv["t_b"])
        if tb <= ta + 1e-6:
            continue
        ds = max(0.0, float(iv["delta_s"]))
        # 检查是否有落在此区间内部的显式 hold 窗口
        overlapping_holds = [
            (max(ta, hs), min(tb, he))
            for hs, he in explicit_holds
            if he > ta + 1e-4 and hs < tb - 1e-4
        ]
        # 同时检查显式中间停顿点（hold==0 但 stop=True 的内部停靠）
        split_stops = [
            st for st in explicit_stop_times
            if ta + 1e-3 < st < tb - 1e-3 and not any(hs - 1e-4 <= st <= he + 1e-4 for hs, he in overlapping_holds)
        ]

        if not overlapping_holds and not split_stops:
            base_ivs.append({
                "t_a": ta, "t_b": tb, "delta_s": ds,
                "s_a": float(iv.get("s_a", 0.0)),
                "tau_a": float(iv.get("tau_a", ta)), "tau_b": float(iv.get("tau_b", tb)),
                "is_pause": ds <= 1e-6,
                "stop_boundary_end": any(abs(tb - st) <= 1e-3 for st in explicit_stop_times),
                "stop_boundary_start": any(abs(ta - st) <= 1e-3 for st in explicit_stop_times),
                "locked_end": bool(iv.get("locked_end", True)),
            })
            continue

        # 带暂停切分（不擅自丢弃原始锁定面积总预算）
        sub_pts: List[float] = [ta]
        for hs, he in overlapping_holds:
            sub_pts.extend([hs, he])
        for st in split_stops:
            sub_pts.append(st)
        sub_pts.append(tb)
        sub_pts = sorted(set(round(x, 6) for x in sub_pts))
        move_pieces: List[Tuple[int, float]] = []
        raw_sub: List[Dict[str, Any]] = []
        for idx_s in range(len(sub_pts) - 1):
            sa_t, sb_t = sub_pts[idx_s], sub_pts[idx_s + 1]
            if sb_t - sa_t <= 1e-5:
                continue
            mid = 0.5 * (sa_t + sb_t)
            in_hold = any(hs - 1e-5 <= mid <= he + 1e-5 for hs, he in overlapping_holds)
            raw_sub.append({"t_a": sa_t, "t_b": sb_t, "is_pause": in_hold or ds <= 1e-6})
            if not in_hold and ds > 1e-6:
                move_pieces.append((len(raw_sub) - 1, sb_t - sa_t))
        tot_move_dt = sum(w for _, w in move_pieces)
        acc_s = float(iv.get("s_a", 0.0))
        acc_tau = float(iv.get("tau_a", ta))
        tau_span = float(iv.get("tau_b", tb)) - acc_tau
        for idx_r, sub in enumerate(raw_sub):
            dt_r = sub["t_b"] - sub["t_a"]
            if sub["is_pause"] or tot_move_dt <= 1e-6:
                ds_r = 0.0
                dtau_r = dt_r if ds <= 1e-6 else 0.0
            else:
                ds_r = ds * (dt_r / tot_move_dt)
                dtau_r = tau_span * (ds_r / ds) if ds > 1e-6 else 0.0
            base_ivs.append({
                "t_a": sub["t_a"], "t_b": sub["t_b"], "delta_s": ds_r,
                "s_a": acc_s, "tau_a": acc_tau, "tau_b": acc_tau + dtau_r,
                "is_pause": sub["is_pause"],
                "stop_boundary_end": any(abs(sub["t_b"] - st) <= 1e-3 for st in explicit_stop_times),
                "stop_boundary_start": any(abs(sub["t_a"] - st) <= 1e-3 for st in explicit_stop_times),
                "locked_end": bool(iv.get("locked_end", True)),
            })
            acc_s += ds_r
            acc_tau += dtau_r

    # 若未锁定构图时间（lock_compositions=False），把连续移动的邻接区间合并为一个自由全局跨度（但仍尊重停顿切片！）
    # 注意：即使 lock_compositions=True，相邻的连续移动区间也以同一跨度进入联合 B 样条 QP，同时把每个锁点 t_b 作为精确积分等式！
    # 因此无论是否锁定，只要没有遇到真正的停顿/零速平台/切镜，都不会切断跨边界的速度与加速度连续性！
    spans: List[Dict[str, Any]] = []
    cur_move_ivs: List[Dict[str, Any]] = []
    for idx_b, biv in enumerate(base_ivs):
        if biv["is_pause"]:
            if cur_move_ivs:
                spans.append({"type": "move", "intervals": cur_move_ivs})
                cur_move_ivs = []
            spans.append({"type": "pause", "intervals": [biv]})
        else:
            if cur_move_ivs and (biv.get("stop_boundary_start") or cur_move_ivs[-1].get("stop_boundary_end")):
                spans.append({"type": "move", "intervals": cur_move_ivs})
                cur_move_ivs = [biv]
            else:
                cur_move_ivs.append(biv)
    if cur_move_ivs:
        spans.append({"type": "move", "intervals": cur_move_ivs})

    all_segments: List[Dict[str, Any]] = []
    all_conflicts: List[Dict[str, Any]] = []
    any_clamped = False
    achieved_edit_v: Optional[float] = None
    affected_range: Optional[List[float]] = None
    cruise_infos: List[Dict[str, Any]] = []

    for sp_idx, sp in enumerate(spans):
        ivs = sp["intervals"]
        ta = float(ivs[0]["t_a"])
        tb = float(ivs[-1]["t_b"])
        if sp["type"] == "pause":
            all_segments.append({
                "kind": "pause", "t0": ta, "t1": tb,
                "v0": 0.0, "v1": 0.0, "m0": 0.0, "m1": 0.0, "area": 0.0,
            })
            continue

        left_is_pause_or_start = (sp_idx == 0 and bc_start_stop) or (sp_idx > 0 and (spans[sp_idx - 1]["type"] == "pause" or ivs[0].get("stop_boundary_start")))
        right_is_pause_or_end = (sp_idx == len(spans) - 1 and bc_end_stop) or (sp_idx < len(spans) - 1 and (spans[sp_idx + 1]["type"] == "pause" or ivs[-1].get("stop_boundary_end")))

        # 若未锁定中间锚点（lock_compositions=False），把本移动跨度内所有子区间合并成单一总距离等式，保留真实路径自由分配各时刻
        solve_ivs = ivs
        if not lock_compositions and len(ivs) > 1:
            solve_ivs = [{
                "t_a": ta, "t_b": tb,
                "delta_s": sum(x["delta_s"] for x in ivs),
                "s_a": ivs[0]["s_a"],
                "tau_a": ivs[0]["tau_a"], "tau_b": ivs[-1]["tau_b"],
            }]

        # 本跨度对应的局部曲率片段
        s_lo = float(ivs[0]["s_a"])
        s_hi = float(ivs[-1]["s_a"]) + float(ivs[-1]["delta_s"])
        local_kappa = None
        if curvature_samples:
            local_kappa = [(sk, kp) for sk, kp in curvature_samples if s_lo - 0.5 <= sk <= s_hi + 0.5]

        if mode == "constant_speed":
            res_sp = _solve_constant_speed_span(
                ta, tb, solve_ivs, lim,
                bc_left_stop=bool(left_is_pause_or_start),
                bc_right_stop=bool(right_is_pause_or_end),
                curvature_samples=local_kappa,
            )
            cruise_infos.append({
                "t_a": ta, "t_b": tb,
                "cruise_speed": res_sp["cruise_speed"],
                "ramp_in_s": res_sp["ramp_in_s"],
                "ramp_out_s": res_sp["ramp_out_s"],
            })
        else:
            # 过滤落在此移动跨度上的偏好点；若起止端点用户设置了显式非零初/末速且未标记 stop=True，则尊重之
            sp_pts = [p for p in pts if ta - 1e-4 <= float(p.get("t", 0.0)) <= tb + 1e-4 and float(p.get("hold_duration", 0.0) or 0.0) <= 1e-4]
            v_l_fixed = None
            v_r_fixed = None
            if not left_is_pause_or_start:
                p_start = next((p for p in sp_pts if abs(float(p["t"]) - ta) <= 1e-3), None)
                if p_start is not None and p_start.get("pin_v"):
                    v_l_fixed = float(p_start["v"])
            if not right_is_pause_or_end:
                p_end = next((p for p in sp_pts if abs(float(p["t"]) - tb) <= 1e-3), None)
                if p_end is not None and p_end.get("pin_v"):
                    v_r_fixed = float(p_end["v"])

            # 如果传入 edited_time 不在此移动跨度，不扰动本跨度
            ed_t = edited_time if (edited_time is not None and ta - 1e-3 <= edited_time <= tb + 1e-3) else None
            res_sp = _solve_moving_span_qp(
                ta, tb, solve_ivs, sp_pts, lim,
                bc_left_stop=bool(left_is_pause_or_start),
                bc_right_stop=bool(right_is_pause_or_end),
                v_left_fixed=v_l_fixed,
                v_right_fixed=v_r_fixed,
                edited_time=ed_t,
                edited_speed=edited_speed if ed_t is not None else None,
                baseline_samples=baseline_samples,
                curvature_samples=local_kappa,
            )
            if ed_t is not None:
                any_clamped = any_clamped or bool(res_sp.get("clamped"))
                achieved_edit_v = res_sp.get("achieved_edit_speed")
                affected_range = res_sp.get("affected_range")

        all_segments.extend(res_sp["segments"])
        all_conflicts.extend(res_sp.get("conflicts") or [])

    return {
        "segments": all_segments,
        "conflicts": all_conflicts,
        "clamped": any_clamped,
        "achieved_edit_speed": achieved_edit_v,
        "affected_range": affected_range,
        "cruise_infos": cruise_infos,
    }


def eval_segments_at_time(segments: List[Dict[str, Any]], t: float) -> Tuple[float, float, float]:
    """求值分段三次 Hermite/Pause 速度曲线：(v(t), s_traveled(t), a(t))。"""
    if not segments:
        return 0.0, 0.0, 0.0
    if t <= segments[0]["t0"]:
        s0 = segments[0]
        return float(s0.get("v0", 0.0)), 0.0, float(s0.get("m0", 0.0))
    acc = 0.0
    for seg in segments:
        t0, t1 = float(seg["t0"]), float(seg["t1"])
        if t >= t1:
            acc += float(seg.get("area", 0.0))
            continue
        if t <= t0:
            return float(seg.get("v0", 0.0)), acc, float(seg.get("m0", 0.0))
        if seg.get("kind") == "pause":
            return 0.0, acc, 0.0
        h = t1 - t0
        u = max(0.0, min(1.0, (t - t0) / h)) if h > 1e-8 else 0.0
        u2 = u * u
        u3 = u2 * u
        u4 = u2 * u2
        v0, v1 = float(seg["v0"]), float(seg["v1"])
        m0, m1 = float(seg["m0"]), float(seg["m1"])
        val = (
            (2.0 * u3 - 3.0 * u2 + 1.0) * v0
            + (u3 - 2.0 * u2 + u) * h * m0
            + (-2.0 * u3 + 3.0 * u2) * v1
            + (u3 - u2) * h * m1
        )
        # a(t) = dv/dt = (1/h) * dv/du
        accel = (
            (6.0 * u2 - 6.0 * u) * v0 / max(1e-8, h)
            + (3.0 * u2 - 4.0 * u + 1.0) * m0
            + (-6.0 * u2 + 6.0 * u) * v1 / max(1e-8, h)
            + (3.0 * u2 - 2.0 * u) * m1
        )
        part = h * (
            (0.5 * u4 - u3 + u) * v0
            + (0.25 * u4 - (2.0 / 3.0) * u3 + 0.5 * u2) * h * m0
            + (-0.5 * u4 + u3) * v1
            + (0.25 * u4 - (1.0 / 3.0) * u3) * h * m1
        )
        return max(0.0, val), acc + max(0.0, min(float(seg.get("area", 0.0)), part)), accel
    sl = segments[-1]
    return float(sl.get("v1", 0.0)), acc, float(sl.get("m1", 0.0))


def _eval_vec3_spatial_centripetal(
    vecs: List[List[float]],
    params: List[float],
    modes: List[str],
    stops: List[bool],
    seg_idx: int,
    u: float,
) -> List[float]:
    """
    纯几何空间向量级向心三次样条：仅由空间弦长驱动切线，不受时间分布和起停零速衰减影响，
    因此路点几何曲率纯由空间位置决定，且末端保持非零几何切线方向不发生 s->tau 奇异性。
    """
    n = len(vecs)
    i = max(0, min(n - 2, int(seg_idx)))
    uc = max(0.0, min(1.0, float(u)))
    p0 = [float(vecs[i][c]) for c in range(3)]
    p1 = [float(vecs[i + 1][c]) for c in range(3)]
    hi = float(params[i + 1]) - float(params[i])
    dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    chord_i = math.hypot(dx, dy, dz)
    mode_i = modes[i]
    if hi <= 1e-6 or chord_i <= 1e-7:
        return p0
    if mode_i == "linear" or mode_i == "hold":
        return [p0[c] + (p1[c] - p0[c]) * uc for c in range(3)]

    chords = []
    d_vec = []
    d_spd = []
    h_list = []
    for k in range(n - 1):
        hk = max(1e-6, float(params[k + 1]) - float(params[k]))
        vx = vecs[k + 1][0] - vecs[k][0]
        vy = vecs[k + 1][1] - vecs[k][1]
        vz = vecs[k + 1][2] - vecs[k][2]
        lk = math.hypot(vx, vy, vz)
        h_list.append(hk)
        chords.append(lk)
        if lk <= 1e-7 or modes[k] == "hold":
            d_vec.append([0.0, 0.0, 0.0])
            d_spd.append(0.0)
        else:
            d_vec.append([vx / hk, vy / hk, vz / hk])
            d_spd.append(lk / hk)

    v = [[0.0, 0.0, 0.0] for _ in range(n)]
    for k in range(1, n - 1):
        # 注意：若明确折角/急停且显式是线性夹角(linear -> linear)则断开；否则在连续点保持 C1 空间切线
        sp_p, sp_n = d_spd[k - 1], d_spd[k]
        if sp_p <= 1e-7 or sp_n <= 1e-7:
            continue
        if stops[k] and modes[k - 1] == "linear" and modes[k] == "linear":
            continue
        dp, dn = d_vec[k - 1], d_vec[k]
        cos_th = (dp[0] * dn[0] + dp[1] * dn[1] + dp[2] * dn[2]) / (sp_p * sp_n)
        if cos_th < -0.94:
            continue
        tp_w = math.sqrt(h_list[k - 1] * max(chords[k - 1], 1e-4))
        tn_w = math.sqrt(h_list[k] * max(chords[k], 1e-4))
        denom = tp_w + tn_w
        if denom <= 1e-8:
            continue
        vx = (tn_w * dp[0] + tp_w * dn[0]) / denom
        vy = (tn_w * dp[1] + tp_w * dn[1]) / denom
        vz = (tn_w * dp[2] + tp_w * dn[2]) / denom
        vm = math.hypot(vx, vy, vz)
        max_sp = 1.65 * min(sp_p, sp_n)
        if vm > max_sp and vm > 1e-8:
            sc = max_sp / vm
            vx, vy, vz = vx * sc, vy * sc, vz * sc
        v[k] = [vx, vy, vz]

    # 首末几何切线保持自然指向非零（不把空间斜率强行归零，防止速度curve叠加二次减速造成 s(tau) 扁平奇异）
    if d_spd[0] > 1e-7:
        if n >= 3 and math.hypot(v[1][0], v[1][1], v[1][2]) > 1e-7:
            t0v = [1.5 * d_vec[0][c] - 0.5 * v[1][c] for c in range(3)]
            tm = math.hypot(t0v[0], t0v[1], t0v[2])
            if tm > 1.25 * d_spd[0] and tm > 1e-8:
                sc = (1.25 * d_spd[0]) / tm
                t0v = [t0v[c] * sc for c in range(3)]
            v[0] = t0v
        else:
            v[0] = list(d_vec[0])

    if d_spd[n - 2] > 1e-7:
        if n >= 3 and math.hypot(v[n - 2][0], v[n - 2][1], v[n - 2][2]) > 1e-7:
            tnv = [1.5 * d_vec[n - 2][c] - 0.5 * v[n - 2][c] for c in range(3)]
            tm = math.hypot(tnv[0], tnv[1], tnv[2])
            if tm > 1.25 * d_spd[n - 2] and tm > 1e-8:
                sc = (1.25 * d_spd[n - 2]) / tm
                tnv = [tnv[c] * sc for c in range(3)]
            v[n - 1] = tnv
        else:
            v[n - 1] = list(d_vec[n - 2])

    u2 = uc * uc
    u3 = u2 * uc
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = (u3 - 2.0 * u2 + uc) * hi
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = (u3 - u2) * hi
    return [h00 * p0[c] + h10 * v[i][c] + h01 * p1[c] + h11 * v[i + 1][c] for c in range(3)]


def unwrap_angles_deg(angles: List[float]) -> List[float]:
    """对旋转角度列表沿最短路径展开，跨 ±180° 界不产生 360° 兜圈突变。"""
    if not angles:
        return []
    out = [float(angles[0])]
    for a in angles[1:]:
        v = float(a)
        while v - out[-1] > 180.0:
            v -= 360.0
        while v - out[-1] < -180.0:
            v += 360.0
        out.append(v)
    return out


def compute_path_curvature_samples(
    eval_pos_at_tau,
    taus: List[float],
    s_vals: List[float],
) -> List[Tuple[float, float]]:
    """采样路径曲率 kappa(s) = ||d^2 p / ds^2||，返回 [(s_i, kappa_i)]。"""
    n = len(taus)
    if n < 3:
        return []
    out: List[Tuple[float, float]] = []
    stride = max(1, n // 160)
    pts_cache = [eval_pos_at_tau(taus[i]) for i in range(0, n, stride)]
    s_sub = [s_vals[i] for i in range(0, n, stride)]
    for k in range(1, len(pts_cache) - 1):
        ds1 = s_sub[k] - s_sub[k - 1]
        ds2 = s_sub[k + 1] - s_sub[k]
        if ds1 <= 1e-4 or ds2 <= 1e-4:
            continue
        p0, p1, p2 = pts_cache[k - 1], pts_cache[k], pts_cache[k + 1]
        t1 = [(p1[c] - p0[c]) / ds1 for c in range(3)]
        t2 = [(p2[c] - p1[c]) / ds2 for c in range(3)]
        n1 = math.hypot(*t1)
        n2 = math.hypot(*t2)
        if n1 > 1e-5 and n2 > 1e-5:
            t1 = [c / n1 for c in t1]
            t2 = [c / n2 for c in t2]
            dk = math.hypot(t2[0] - t1[0], t2[1] - t1[1], t2[2] - t1[2])
            kappa = dk / (0.5 * (ds1 + ds2))
            if kappa > 1e-3:
                out.append((s_sub[k], kappa))
    return out


def _is_mounted_local_cabin_track(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict) or obj.get("type") != "character":
        return False
    if not obj.get("mounted_vehicle_id") or obj.get("follow_mounted_vehicle") is False:
        return False
    kfs = obj.get("keyframes") or []
    if not kfs:
        return True
    return all(
        abs(float((k.get("pos") or [0.0, 0.0, 0.0])[0])) < 1.6
        and abs(float((k.get("pos") or [0.0, 0.0, 0.0])[2])) < 2.2
        for k in kfs if isinstance(k, dict)
    )


def _has_object_world_prefix(obj: Dict[str, Any], scene: Optional[Dict[str, Any]] = None) -> bool:
    if not isinstance(obj, dict):
        return False
    lr = obj.get("layout_ref")
    if isinstance(lr, dict) and lr:
        return True
    pid = str(obj.get("parent_id") or "").strip()
    if not pid or pid == str(obj.get("id") or "") or not isinstance(scene, dict):
        return False
    objs = scene.get("objects") or []
    return any(isinstance(o, dict) and str(o.get("id") or "") == pid for o in objs)


def _get_object_world_prefix_matrix_mc(
    obj: Dict[str, Any],
    t: float,
    scene: Optional[Dict[str, Any]] = None,
    obj_map: Optional[Dict[str, Dict[str, Any]]] = None,
    visited: Optional[set] = None,
):
    import numpy as np
    from director_engine import (
        _compose_layout_ref_matrix,
        interpolate_object_at_time,
        make_transform,
    )
    M_reloc = _compose_layout_ref_matrix(obj.get("layout_ref") if isinstance(obj, dict) else None)
    M_reloc = M_reloc if M_reloc is not None else np.eye(4, dtype=np.float64)
    pid = str(obj.get("parent_id") or "").strip() if isinstance(obj, dict) else ""
    self_id = str(obj.get("id") or "") if isinstance(obj, dict) else ""
    if not pid or pid == self_id:
        return M_reloc
    if obj_map is None and isinstance(scene, dict):
        obj_map = {
            str(o.get("id")): o
            for o in (scene.get("objects") or [])
            if isinstance(o, dict) and o.get("id")
        }
    if not obj_map or pid not in obj_map:
        return M_reloc
    seen = set(visited) if visited else ({self_id} if self_id else set())
    if pid in seen:
        return M_reloc
    seen.add(pid)
    parent = obj_map[pid]
    M_gp = _get_object_world_prefix_matrix_mc(parent, t, scene=scene, obj_map=obj_map, visited=seen)
    p_local = interpolate_object_at_time(parent, float(t), scene=scene)
    M_plocal = make_transform(p_local.get("pos", [0.0, 0.0, 0.0]), p_local.get("rot", [0.0, 0.0, 0.0]), 1.0)
    if str(parent.get("type") or "") == "scene_group":
        psz = [float(x) for x in (p_local.get("size") or [1.0, 1.0, 1.0])[:3]]
        M_plocal = M_plocal @ np.diag([psz[0], psz[1], psz[2], 1.0])
    return M_gp @ M_plocal @ M_reloc


def _eval_waypoint_world_transform_mc(
    obj: Dict[str, Any],
    local_pos: List[float],
    local_rot_y: float,
    t: float,
    scene: Optional[Dict[str, Any]] = None,
    obj_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[float], float]:
    from director_engine import (
        _decompose_prefix_matrix,
        make_transform,
        matrix_to_euler_xyz_deg,
    )
    M_prefix = _get_object_world_prefix_matrix_mc(obj, t, scene=scene, obj_map=obj_map)
    M_loc = make_transform(local_pos, [0.0, float(local_rot_y), 0.0], 1.0)
    M_w = M_prefix @ M_loc
    R_w, _ = _decompose_prefix_matrix(M_w)
    e_w = matrix_to_euler_xyz_deg(R_w)
    return [float(M_w[0, 3]), float(M_w[1, 3]), float(M_w[2, 3])], float(e_w[1])


def _eval_obj_track_world_state_mc(
    obj: Dict[str, Any],
    t: float,
    scene: Optional[Dict[str, Any]] = None,
    obj_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    from director_engine import (
        evaluate_object_at_time,
        interpolate_character_at_time,
    )
    if obj_map is None and isinstance(scene, dict):
        obj_map = {
            str(o.get("id")): o
            for o in (scene.get("objects") or [])
            if isinstance(o, dict) and o.get("id")
        }
    if obj.get("type") == "character":
        c_st = interpolate_character_at_time(obj, float(t), scene=scene)
        mounted_vid = str(obj.get("mounted_vehicle_id") or "")
        if mounted_vid and obj_map and mounted_vid in obj_map and obj.get("follow_mounted_vehicle") is not False:
            veh_st = evaluate_object_at_time(obj_map[mounted_vid], float(t), obj_map=obj_map, scene=scene)
            v_pos = veh_st.get("pos", [0.0, 1.01, 0.0])
            v_rot = veh_st.get("rot", [0.0, 0.0, 0.0])
            v_size = veh_st.get("size", [1.78, 2.02, 4.02])
            has_kfs = bool(obj.get("keyframes"))
            c_pos_raw = c_st.get("pos", [0.0, 0.42, 0.0])
            if has_kfs and abs(float(c_pos_raw[0])) < 1.6 and abs(float(c_pos_raw[2])) < 2.2:
                seat_off = c_pos_raw
                local_ry = float((c_st.get("rot") or [0.0, 0.0, 0.0])[1])
            else:
                seat_off = obj.get("seat_offset") or c_pos_raw
                local_ry = float(obj.get("seat_rot_offset", (c_st.get("rot") or [0.0, 0.0, 0.0])[1]))
            ox, oy, oz = float(seat_off[0]), float(seat_off[1]), float(seat_off[2])
            v_yaw_deg = float(v_rot[1]) if len(v_rot) >= 2 else 0.0
            v_yaw_rad = math.radians(v_yaw_deg)
            cy_v, sy_v = math.cos(v_yaw_rad), math.sin(v_yaw_rad)
            v_base_y = float(v_pos[1]) - 0.5 * float(v_size[1])
            return {
                "pos": [
                    float(v_pos[0]) + ox * cy_v + oz * sy_v,
                    v_base_y + oy,
                    float(v_pos[2]) - ox * sy_v + oz * cy_v,
                ],
                "rot": [0.0, v_yaw_deg + local_ry, 0.0],
            }
        if _has_object_world_prefix(obj, scene):
            wp, wry = _eval_waypoint_world_transform_mc(
                obj,
                [float(x) for x in (c_st.get("pos") or [0.0, 0.0, 0.0])[:3]],
                float((c_st.get("rot") or [0.0, 0.0, 0.0])[1]),
                float(t),
                scene=scene,
                obj_map=obj_map,
            )
            return {"pos": wp, "rot": [0.0, wry, 0.0]}
        return {"pos": list(c_st.get("pos") or [0.0, 0.0, 0.0]), "rot": list(c_st.get("rot") or [0.0, 0.0, 0.0])}
    return evaluate_object_at_time(obj, float(t), obj_map=obj_map, scene=scene)


def build_track_continuity_plan(
    obj: Dict[str, Any],
    scene: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    为角色或可移动对象构建统一的“空间路径 p(s) + 连续时间推进 s(t) + 朝向连续映射”求解包。
    不更改原 obj；若尚未启用 motion_protection 且无保护标记则返回 enabled=False。
    """
    mp = obj.get("motion_protection") if isinstance(obj.get("motion_protection"), dict) else None
    if not mp or not mp.get("enabled"):
        return {"enabled": False}

    kfs = sorted(
        [dict(k) for k in (obj.get("keyframes") or []) if isinstance(k, dict)],
        key=lambda x: float(x.get("t", 0.0)),
    )
    if len(kfs) < 2:
        return {"enabled": False}

    mode = str(mp.get("mode") or "smooth_variable").lower().strip()
    if mode == "manual":
        return {"enabled": False, "mode": "manual"}

    kind = "character" if obj.get("type") == "character" else ("vehicle" if obj.get("type") in ("uaz469", "vehicle", "car", "truck") else "prop")
    base_lim = dict(DEFAULT_PROTECTION_LIMITS.get(kind, DEFAULT_PROTECTION_LIMITS["prop"]))
    if isinstance(obj.get("speed_exception"), dict) and obj["speed_exception"].get("max_mps"):
        base_lim["max_speed"] = max(base_lim["max_speed"], float(obj["speed_exception"]["max_mps"]))
    if isinstance(mp.get("limits"), dict):
        for k_lim, v_lim in mp["limits"].items():
            if v_lim is not None:
                base_lim[k_lim] = float(v_lim)

    # 若人物在已有镜头里按速度分类（如跑/慢跑），允许读取对应速度上限而不报错
    if kind == "character":
        try:
            from motion_speed_policy import SPEED_LIMIT_TABLE
            mt = str(obj.get("motion_type") or "").strip()
            if mt in SPEED_LIMIT_TABLE and not (isinstance(mp.get("limits"), dict) and mp["limits"].get("max_speed")):
                base_lim["max_speed"] = max(base_lim["max_speed"], float(SPEED_LIMIT_TABLE[mt]["max_mps"]))
        except Exception:
            pass

    dur = max(0.1, float((scene or {}).get("duration") or kfs[-1].get("t", 1.0) or 1.0))
    times = [float(k.get("t", 0.0)) for k in kfs]
    base_pos = [float(x) for x in (obj.get("pos") or [0.0, 0.0, 0.0])[:3]]
    vecs_local = [[float((k.get("pos") or base_pos)[c]) for c in range(3)] for k in kfs]
    # 注意：门或只旋转不位移的对象（如 prop_green_door），pos 弧长为 0 而 rot 变化明显
    rot_y_local_raw = [float((k.get("rot") or [0.0, float(k.get("rot_y", 0.0)), 0.0])[1] if len(k.get("rot") or []) >= 2 else k.get("rot_y", 0.0)) for k in kfs]
    rot_y_local_unwrapped = unwrap_angles_deg(rot_y_local_raw)

    is_mounted_local = _is_mounted_local_cabin_track(obj)
    has_world_prefix = (not is_mounted_local) and _has_object_world_prefix(obj, scene)
    obj_map = (
        {str(o.get("id")): o for o in (scene.get("objects") or []) if isinstance(o, dict) and o.get("id")}
        if isinstance(scene, dict)
        else None
    )
    has_animated_parent = False
    if has_world_prefix and obj_map:
        cur_p = obj
        seen_anc = {str(obj.get("id") or "")}
        for _ in range(16):
            pid = str(cur_p.get("parent_id") or "").strip()
            if not pid or pid in seen_anc:
                break
            seen_anc.add(pid)
            p_obj = obj_map.get(pid)
            if not p_obj:
                break
            if len(p_obj.get("keyframes") or []) >= 2:
                has_animated_parent = True
                break
            cur_p = p_obj

    if has_world_prefix:
        wp_pairs = [
            _eval_waypoint_world_transform_mc(obj, lp, rot_y_local_unwrapped[idx], times[idx], scene=scene, obj_map=obj_map)
            for idx, lp in enumerate(vecs_local)
        ]
        vecs_world = [pw[0] for pw in wp_pairs]
        rot_y_world_unwrapped = unwrap_angles_deg([pw[1] for pw in wp_pairs])
    else:
        vecs_world = vecs_local
        rot_y_world_unwrapped = rot_y_local_unwrapped

    vecs = vecs_world
    rot_y_unwrapped = rot_y_world_unwrapped

    # 检测区间属性（是否静止、切镜、或平滑路径）
    modes = []
    stops = [bool(k.get("stop") or k.get("pause")) for k in kfs]
    # 首末未显式声明 stop=False 时默认在零位移端或起止端起停
    for i in range(len(kfs) - 1):
        m_i = str(kfs[i].get("interp_pos") or kfs[i].get("interp") or "continuous").lower().strip()
        if kfs[i].get("cut") or kfs[i].get("teleport") or m_i == "hold":
            modes.append("hold")
        elif kfs[i].get("motion_override") == "manual":
            modes.append("linear")
        else:
            modes.append("continuous")
    modes.append("linear")

    # 判断对象是否主要是位置移动，还是原地纯旋转（如门动画），均按世界空间真实弦长/转角衡量
    pos_chords = [math.dist(vecs[i], vecs[i + 1]) for i in range(len(kfs) - 1)]
    rot_deltas = [abs(rot_y_unwrapped[i + 1] - rot_y_unwrapped[i]) for i in range(len(kfs) - 1)]
    total_pos_chord = sum(pos_chords)
    total_rot_delta = sum(rot_deltas)
    is_rot_primary = (total_pos_chord <= 1e-4 and total_rot_delta > 1e-3)

    def _eval_curve_on_vecs(vec_arr: List[List[float]], ry_arr: List[float], tau: float) -> Tuple[List[float], float]:
        tc = max(times[0], min(times[-1], float(tau)))
        for i in range(len(kfs) - 1):
            t0, t1 = times[i], times[i + 1]
            if tc <= t1 + 1e-9 or i == len(kfs) - 2:
                dt_i = t1 - t0
                u = max(0.0, min(1.0, (tc - t0) / dt_i)) if dt_i > 1e-6 else 0.0
                if modes[i] == "hold":
                    p = list(vec_arr[i] if u < 0.9999 else vec_arr[i + 1])
                    ry = ry_arr[i] if u < 0.9999 else ry_arr[i + 1]
                elif pos_chords[i] <= 1e-5:
                    p = list(vec_arr[i])
                    ry = ry_arr[i] + (ry_arr[i + 1] - ry_arr[i]) * u
                else:
                    # 若某关键帧为明确停顿后的转折步（比如到门停稳后转向再进门），用当前连续子段做样条避免停顿前后的折线被拉弯
                    p = _eval_vec3_spatial_centripetal(vec_arr, times, modes, stops, i, u)
                    ry = ry_arr[i] + (ry_arr[i + 1] - ry_arr[i]) * u
                return p, ry
        return list(vec_arr[-1]), ry_arr[-1]

    static_prefix_inv = None
    if has_world_prefix and not has_animated_parent:
        try:
            import numpy as np
            M_pref0 = _get_object_world_prefix_matrix_mc(obj, times[0], scene=scene, obj_map=obj_map)
            static_prefix_inv = np.linalg.inv(M_pref0)
        except Exception:
            static_prefix_inv = None

    def _eval_world_and_local_at_tau(tau: float) -> Tuple[List[float], float, List[float], float]:
        if not has_world_prefix:
            p, ry = _eval_curve_on_vecs(vecs_local, rot_y_local_unwrapped, tau)
            return p, ry, p, ry
        if not has_animated_parent and static_prefix_inv is not None:
            import numpy as np
            from director_engine import _decompose_prefix_matrix, make_transform, matrix_to_euler_xyz_deg
            wp, wry = _eval_curve_on_vecs(vecs_world, rot_y_world_unwrapped, tau)
            lp_h = static_prefix_inv @ np.array([wp[0], wp[1], wp[2], 1.0], dtype=np.float64)
            M_w_rot = make_transform(wp, [0.0, wry, 0.0], 1.0)
            M_loc_rec = static_prefix_inv @ M_w_rot
            R_loc, _ = _decompose_prefix_matrix(M_loc_rec)
            e_loc = matrix_to_euler_xyz_deg(R_loc)
            return wp, wry, [float(lp_h[0]), float(lp_h[1]), float(lp_h[2])], float(e_loc[1])
        lp, lry = _eval_curve_on_vecs(vecs_local, rot_y_local_unwrapped, tau)
        wp, wry = _eval_waypoint_world_transform_mc(obj, lp, lry, tau, scene=scene, obj_map=obj_map)
        return wp, wry, lp, lry

    def _eval_spatial_at_tau(tau: float) -> Tuple[List[float], float]:
        wp, wry, _, _ = _eval_world_and_local_at_tau(tau)
        return wp, wry

    # 高分辨率构建精确路径弧长查找表
    t_first, t_last = times[0], times[-1]
    span_t = max(1e-4, t_last - t_first)
    n_tbl = max(180, min(1600, int(math.ceil(span_t * 100.0))))
    tbl_tau = [t_first + (span_t * i) / float(n_tbl) for i in range(n_tbl + 1)]
    tbl_s = [0.0] * (n_tbl + 1)
    prev_p, prev_r = _eval_spatial_at_tau(tbl_tau[0])
    acc_s = 0.0
    for idx_s in range(1, n_tbl + 1):
        cur_p, cur_r = _eval_spatial_at_tau(tbl_tau[idx_s])
        step_s = abs(cur_r - prev_r) if is_rot_primary else math.dist(prev_p, cur_p)
        acc_s += step_s
        tbl_s[idx_s] = acc_s
        prev_p, prev_r = cur_p, cur_r

    def _s_at_tau(tq: float) -> float:
        if tq <= tbl_tau[0]:
            return 0.0
        if tq >= tbl_tau[-1]:
            return tbl_s[-1]
        pos_f = ((tq - t_first) / span_t) * n_tbl
        idx_l = max(0, min(n_tbl - 1, int(math.floor(pos_f))))
        frac = max(0.0, min(1.0, pos_f - idx_l))
        return tbl_s[idx_l] + (tbl_s[idx_l + 1] - tbl_s[idx_l]) * frac

    def _tau_at_s_window(sq: float, tau_lo: float, tau_hi: float) -> float:
        s_lo, s_hi = _s_at_tau(tau_lo), _s_at_tau(tau_hi)
        if s_hi - s_lo <= 1e-7:
            return tau_lo
        sc = max(s_lo, min(s_hi, sq))
        # 二分检索 tbl_s + 线性局部插值
        lo_i = max(0, int(math.floor(((tau_lo - t_first) / span_t) * n_tbl)))
        hi_i = min(n_tbl, int(math.ceil(((tau_hi - t_first) / span_t) * n_tbl)))
        while lo_i + 1 < hi_i:
            mid = (lo_i + hi_i) // 2
            if tbl_s[mid] < sc:
                lo_i = mid
            else:
                hi_i = mid
        s0, s1 = tbl_s[lo_i], tbl_s[min(n_tbl, lo_i + 1)]
        f = (sc - s0) / (s1 - s0) if (s1 - s0) > 1e-9 else 0.0
        t0, t1 = tbl_tau[lo_i], tbl_tau[min(n_tbl, lo_i + 1)]
        return max(tau_lo, min(tau_hi, t0 + f * (t1 - t0)))

    # 组织区间与明确停顿标记（区分普通经过点 vs 停顿点/零位移平台）
    ivs_solve: List[Dict[str, Any]] = []
    pts_solve: List[Dict[str, Any]] = []
    if t_first > 1e-4:
        ivs_solve.append({"t_a": 0.0, "t_b": t_first, "delta_s": 0.0, "s_a": 0.0, "tau_a": 0.0, "tau_b": t_first})
    for i in range(len(kfs) - 1):
        ta_i, tb_i = times[i], times[i + 1]
        sa_i, sb_i = _s_at_tau(ta_i), _s_at_tau(tb_i)
        ds_i = max(0.0, sb_i - sa_i)
        ivs_solve.append({
            "t_a": ta_i, "t_b": tb_i,
            "delta_s": 0.0 if (modes[i] == "hold" or ds_i <= 1e-5) else ds_i,
            "s_a": sa_i, "tau_a": ta_i, "tau_b": tb_i,
        })
        if stops[i] and 1e-3 < ta_i < dur - 1e-3:
            pts_solve.append({"t": ta_i, "v": 0.0, "stop": True})
    if stops[-1] and 1e-3 < t_last < dur - 1e-3:
        pts_solve.append({"t": t_last, "v": 0.0, "stop": True})
    if t_last < dur - 1e-4:
        ivs_solve.append({"t_a": t_last, "t_b": dur, "delta_s": 0.0, "s_a": tbl_s[-1], "tau_a": t_last, "tau_b": dur})

    curv_samples = None if is_rot_primary else compute_path_curvature_samples(lambda tq: _eval_spatial_at_tau(tq)[0], tbl_tau, tbl_s)
    solve_lim = dict(base_lim)
    if is_rot_primary:
        solve_lim["max_speed"] = base_lim.get("max_ang_vel", 240.0)
        solve_lim["max_accel"] = base_lim.get("max_ang_accel", 600.0)
        solve_lim["max_jerk"] = base_lim.get("max_ang_accel", 600.0) * 4.0

    # 首末点若存在移动，默认平滑起停除非 stop is False 且明确设速
    bc0 = kfs[0].get("stop") is not False
    bc1 = kfs[-1].get("stop") is not False
    solved = solve_continuous_velocity_profile(
        duration=dur,
        intervals_in=ivs_solve,
        points_in=pts_solve,
        mode=("constant_speed" if mode == "constant_speed" else "smooth_variable"),
        limits=solve_lim,
        bc_start_stop=bc0,
        bc_end_stop=bc1,
        curvature_samples=curv_samples,
        lock_compositions=True,
    )

    return {
        "enabled": True,
        "mode": mode,
        "is_rot_primary": is_rot_primary,
        "has_world_prefix": has_world_prefix,
        "kfs": kfs,
        "times": times,
        "vecs": vecs,
        "vecs_local": vecs_local,
        "modes": modes,
        "stops": stops,
        "rot_y_unwrapped": rot_y_unwrapped,
        "rot_y_local_unwrapped": rot_y_local_unwrapped,
        "pos_chords": pos_chords,
        "duration": dur,
        "t_first": t_first,
        "t_last": t_last,
        "tbl_tau": tbl_tau,
        "tbl_s": tbl_s,
        "total_s": tbl_s[-1],
        "intervals": ivs_solve,
        "segments": solved["segments"],
        "conflicts": solved["conflicts"],
        "_eval_spatial_at_tau": _eval_spatial_at_tau,
        "_eval_world_and_local_at_tau": _eval_world_and_local_at_tau,
        "_s_at_tau": _s_at_tau,
        "_tau_at_s_window": _tau_at_s_window,
    }


def eval_track_continuity_at_time(plan: Dict[str, Any], t: float) -> Dict[str, Any]:
    """根据已求得的连续性规划包，在时刻 t 求值 (pos, rot_y, speed, accel, s_cur, tau_cur, seg_idx, seg_alpha)。"""
    dur = float(plan["duration"])
    tc = max(0.0, min(dur, float(t)))
    v_cur, s_cur, a_cur = eval_segments_at_time(plan["segments"], tc)

    # 确定当前 t 落在哪个关键帧窗口 [tau_a, tau_b]，将 s_cur 精准反查为空间路径参数 tau
    kfs = plan["kfs"]
    times = plan["times"]
    tau_cur = max(times[0], min(times[-1], tc))
    seg_idx = 0
    seg_alpha = 0.0
    if tc <= times[0]:
        tau_cur = times[0]
        seg_idx = 0
        seg_alpha = 0.0
    elif tc >= times[-1]:
        tau_cur = times[-1]
        seg_idx = max(0, len(kfs) - 2)
        seg_alpha = 1.0
    else:
        # 寻找时间所属的锁定关键帧区间（若未越界则精确映射到同一窗口）
        for i in range(len(kfs) - 1):
            ta_i, tb_i = times[i], times[i + 1]
            if tc <= tb_i + 1e-7 or i == len(kfs) - 2:
                seg_idx = i
                sa_i = plan["_s_at_tau"](ta_i)
                sb_i = plan["_s_at_tau"](tb_i)
                if sb_i - sa_i <= 1e-6:
                    # 静止期或纯转体期：参数 tau 线性跟随时标 tc
                    dt_i = tb_i - ta_i
                    seg_alpha = max(0.0, min(1.0, (tc - ta_i) / dt_i)) if dt_i > 1e-6 else 0.0
                    tau_cur = ta_i + seg_alpha * dt_i
                else:
                    tau_cur = plan["_tau_at_s_window"](s_cur, ta_i, tb_i)
                    dt_i = tb_i - ta_i
                    seg_alpha = max(0.0, min(1.0, (tau_cur - ta_i) / dt_i)) if dt_i > 1e-6 else 0.0
                break

    ev_pair = plan["_eval_world_and_local_at_tau"](tau_cur) if "_eval_world_and_local_at_tau" in plan else None
    if ev_pair is not None:
        world_pos_out, world_rot_y_out, pos_out, rot_y_out = ev_pair
    else:
        pos_out, rot_y_out = plan["_eval_spatial_at_tau"](tau_cur)
        world_pos_out, world_rot_y_out = pos_out, rot_y_out
    # 检查是否有站立转身段（pos 不变而 rot_y 改变）：用五次 S5 缓和旋转起步与停顿，消除线性转身突变！
    if seg_idx < len(kfs) - 1 and plan["pos_chords"][seg_idx] <= 1e-5:
        ta_i, tb_i = times[seg_idx], times[seg_idx + 1]
        u_t = max(0.0, min(1.0, (tc - ta_i) / max(1e-6, tb_i - ta_i)))
        u_s = s5_step(u_t) if plan["modes"][seg_idx] != "hold" else (0.0 if u_t < 0.9999 else 1.0)
        l_ry_arr = plan.get("rot_y_local_unwrapped") or plan["rot_y_unwrapped"]
        w_ry_arr = plan["rot_y_unwrapped"]
        rot_y_out = l_ry_arr[seg_idx] + (l_ry_arr[seg_idx + 1] - l_ry_arr[seg_idx]) * u_s
        world_rot_y_out = w_ry_arr[seg_idx] + (w_ry_arr[seg_idx + 1] - w_ry_arr[seg_idx]) * u_s

    return {
        "pos": pos_out,
        "rot_y": rot_y_out,
        "world_pos": world_pos_out,
        "world_rot_y": world_rot_y_out,
        "speed": v_cur,
        "accel": a_cur,
        "s_cur": s_cur,
        "tau": tau_cur,
        "seg_idx": seg_idx,
        "seg_alpha": seg_alpha,
    }


def apply_continuity_protection_to_track(obj: Dict[str, Any], scene: Optional[Dict[str, Any]] = None, mode: str = "smooth_variable") -> Dict[str, Any]:
    """显式将轨道切换为启用运动连续性保护（保留原有显式停顿、切镜与角色自定义朝向，支持撤销）。"""
    out = copy.deepcopy(obj)
    mp = dict(out.get("motion_protection") or {})
    mp["enabled"] = (mode != "manual")
    mp["mode"] = mode if mode in ("constant_speed", "smooth_variable", "manual") else "smooth_variable"
    mp["respect_stops"] = True
    out["motion_protection"] = mp
    return out


def check_scene_continuity_diagnostics(scene: Dict[str, Any]) -> Dict[str, Any]:
    """检查整场摄影机、人物、道具（含互锁双人组与门动画开关门撞击风险）并返回诊断列表。"""
    if not isinstance(scene, dict):
        return {"ok": True, "conflicts": [], "warnings": []}

    conflicts: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    objs = scene.get("objects") or []

    # 1) 检查启用了保护的人物/对象是否存在时空/动力学冲突
    plans_by_id: Dict[str, Dict[str, Any]] = {}
    for o in objs:
        if not isinstance(o, dict):
            continue
        oid = str(o.get("id") or "")
        pl = build_track_continuity_plan(o, scene)
        if pl.get("enabled"):
            plans_by_id[oid] = pl
            for c in pl.get("conflicts") or []:
                conflicts.append({"obj_id": oid, "obj_name": o.get("name", oid), **c})

    # 2) 检查明确建立关联的组合（hand_bind / motion_group_id / follow_pair）的距离安全
    fps = max(1.0, float(scene.get("fps", 24.0)))
    dur = float(scene.get("duration", 6.0))
    pair_links: List[Tuple[str, str, float, float, float]] = []
    for c in (scene.get("constraints") or []):
        if isinstance(c, dict) and c.get("enabled", True) is not False and c.get("type") in ("hand_bind", "follow_pair"):
            s_id, t_id = str(c.get("source_id") or ""), str(c.get("target_id") or "")
            if s_id and t_id:
                pair_links.append((s_id, t_id, float(c.get("start_frame", 0)) / fps, float(c.get("end_frame", dur * fps)) / fps, float(c.get("max_separation_m", 1.35))))

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for o in objs:
        if isinstance(o, dict) and o.get("motion_group_id"):
            gid = str(o["motion_group_id"])
            groups.setdefault(gid, []).append(o)
    for gid, members in groups.items():
        for idx in range(len(members) - 1):
            pair_links.append((str(members[idx].get("id")), str(members[idx + 1].get("id")), 0.0, dur, float(members[idx].get("pair_max_separation_m", 1.6))))

    obj_map = {str(o.get("id")): o for o in objs if isinstance(o, dict) and o.get("id")}
    if pair_links:
        for id_a, id_b, t_s, t_e, max_sep in pair_links:
            oa, ob = obj_map.get(id_a), obj_map.get(id_b)
            if not oa or not ob:
                continue
            max_dist = 0.0
            max_t = t_s
            n_chk = max(8, min(60, int(round((t_e - t_s) * 4))))
            for k in range(n_chk + 1):
                tk = max(0.0, min(dur, t_s + (t_e - t_s) * k / max(1, n_chk)))
                pa = _eval_obj_track_world_state_mc(oa, tk, scene=scene, obj_map=obj_map)["pos"]
                pb = _eval_obj_track_world_state_mc(ob, tk, scene=scene, obj_map=obj_map)["pos"]
                d = math.dist(pa[:3], pb[:3])
                if d > max_dist:
                    max_dist = d
                    max_t = tk
            if max_dist > max_sep:
                warnings.append({
                    "type": "pair_separation_warning",
                    "pair": [id_a, id_b],
                    "time_s": round(max_t, 2),
                    "distance_m": round(max_dist, 2),
                    "max_m": round(max_sep, 2),
                    "message": f"关联角色 {oa.get('name', id_a)} 与 {ob.get('name', id_b)} 在 t={max_t:.2f}s 间距达 {max_dist:.2f}m（超过 {max_sep:.2f}m，建议同步推进或缩放位移）",
                })

    # 3) 检查人物穿过门动画附近时的关门碰撞时序（不自动改动门或人，只精准告警）
    doors = [o for o in objs if isinstance(o, dict) and ("door" in str(o.get("id", "")).lower() or "门" in str(o.get("name", ""))) and len(o.get("keyframes") or []) >= 2]
    chars = [o for o in objs if isinstance(o, dict) and o.get("type") == "character" and len(o.get("keyframes") or []) >= 2]
    if doors and chars:
        for dr in doors:
            dr_kfs = sorted(dr.get("keyframes") or [], key=lambda k: float(k.get("t", 0.0)))
            st_d0 = _eval_obj_track_world_state_mc(dr, float(dr_kfs[0].get("t", 0.0)), scene=scene, obj_map=obj_map)
            dr_closed_y = float((st_d0.get("rot") or [0, 0, 0])[1])
            for ch in chars:
                n_s = max(16, int(round(dur * 8)))
                for s in range(n_s + 1):
                    ts = (dur * s) / float(n_s)
                    st_d = _eval_obj_track_world_state_mc(dr, ts, scene=scene, obj_map=obj_map)
                    st_c = _eval_obj_track_world_state_mc(ch, ts, scene=scene, obj_map=obj_map)
                    dp = st_d.get("pos", [0, 0, 0])
                    cp = st_c.get("pos", [0, 0, 0])
                    dist_xz = math.hypot(cp[0] - dp[0], cp[2] - dp[2])
                    dy_rot = abs(float((st_d.get("rot") or [0, 0, 0])[1]) - dr_closed_y)
                    if dist_xz < 0.42 and dy_rot < 25.0:
                        warnings.append({
                            "type": "door_collision_timing",
                            "obj_id": str(ch.get("id")),
                            "door_id": str(dr.get("id")),
                            "time_s": round(ts, 2),
                            "message": f"时序注意：{ch.get('name', ch.get('id'))} 于 t={ts:.2f}s 接近【{dr.get('name', dr.get('id'))}】(<0.42m)，此时门尚未敞开（偏转 {dy_rot:.0f}°）",
                        })
                        break

    return {
        "ok": len(conflicts) == 0,
        "conflicts": conflicts,
        "warnings": warnings,
    }
