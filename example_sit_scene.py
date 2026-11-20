#!/usr/bin/env python3
"""
示例脚本：使用 Python (类似 Blender bpy + 魔法人形师 IK) 搭建坐姿导演台场景并极速导出 MP4 白模视频
运行方式：
  /Users/wangmu/.gemini/antigravity/scratch/scene75_blockout/.venv/bin/python3 example_sit_scene.py
"""

from director_engine import Stage

stage = Stage()

# 1. 添加不同颜色的基础几何体 (长方体 box / 圆锥体 cone / 三角柱 prism / 圆柱 cylinder)
stage.add_box("红色坐凳", pos=[0.0, 0.22, -0.05], size=[0.46, 0.44, 0.46], color="#ef4444")
stage.add_box("蓝色长桌", pos=[0.0, 0.35, 0.78], size=[1.15, 0.70, 0.46], color="#3b82f6")
stage.add_cone("黄色锥体", pos=[-1.15, 0.45, 0.20], size=[0.52, 0.90, 0.52], color="#eab308")
stage.add_prism("绿色三角柱", pos=[1.15, 0.35, 0.15], size=[0.68, 0.70, 0.80], color="#22c55e")

# 2. 添加魔法人形师素体偶，使用点拉式 IK 直接指定控制球 3D 坐标摆出坐姿
hero = stage.add_character("魔法人形偶_01", pos=[0.0, 0.0, 0.0], pose="sitting")
hero.ik(
    pelvis=[0.0, 0.48, -0.04],       # 腰部重心球降低到 0.48m 坐在红色方凳上
    chest=[0.0, 0.93, -0.01],        # 胸腔自然微倾
    head_target=[0.0, 1.18, 0.65],   # 视线看向正前方
    hand_L=[0.16, 0.54, 0.24],       # 左手搭在左大腿/桌面方向
    hand_R=[-0.16, 0.54, 0.24],      # 右手搭在右大腿/桌面方向
    foot_L=[0.13, 0.04, 0.36],       # 左脚踩稳地面
    foot_R=[-0.13, 0.04, 0.36],      # 右脚踩稳地面
)

# 3. 设置摄影机运镜关键帧 (0s -> 4s 环绕推进)
stage.add_camera_keyframe(t=0.0, pos=[2.2, 1.50, 2.75], target=[0.0, 0.72, 0.10], fov=48.0)
stage.add_camera_keyframe(t=4.0, pos=[-1.5, 1.35, 2.45], target=[0.0, 0.72, 0.10], fov=45.0)

# 4. 极速导出 MP4 白模+骨骼预演视频
out_mp4 = stage.export_video("sitting_previz_demo.mp4", duration=4.0, fps=24, mode="clay_and_pose")
print(f"✅ 已极速导出白模预演视频: {out_mp4}")
