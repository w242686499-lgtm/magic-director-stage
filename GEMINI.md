# Magic Director Stage 持久工作流规则

1. **白模复现与正式导出前强制读取 Skill 与三项轻量确认**：
   - 每次开始分析/生成白模视频或正式导出 MP4 前，必须先读取并遵守 `/Users/wangmu/.gemini/config/skills/director-stage-code-import/SKILL.md`。
   - 生成场景或正式导出前先完成一次三项确认（① 关键构图与核心遮挡/人物关系；② 极速/均衡/构图优先，默认极速；③ `10m×10m=100㎡` 面积预算与整镜同时可见代理预算 `300/80`、`600/160`、`1000/240`），已有设置带出一键确认，绝不把沉默或超时当同意；自动化/API 缺少确认时返回 `confirmation_required: true`。
2. **导入代码/白模脚本默认直导，不导出视频、不校准**：
   - 当用户发来 Python 白模脚本、场景代码或分镜编码要求导入时，严格按源码坐标、尺寸、焦距/FOV、画幅、颜色与关键帧导入工程库，默认不渲染 MP4、不跑浏览器截图校准，除非用户明确要求。
3. **非破坏性工程与关键帧规范**：
   - **严禁加载/保存时自动删减关键帧（已废除旧 10 帧/30 秒硬截断）**，严禁删树或覆盖用户 `locked` / `protected` 手工修改来掩盖性能问题。
   - 自动生成关键帧默认每轨任意滚动 10 秒最多 5 帧（连续平滑曲线）；手工关键帧精简仅限显式可执行且可撤销。
   - **默认不使用走路算法**：角色移动默认直接使用关键帧平滑插值（`enable_walk_gait=False`），不叠加程序化迈步/停顿/晃动步态算法，仅当用户明确要求开启走路算法时才启用。
4. **今后统一使用的方块预演规范（新增预演强制执行）**：
   - **既有场景保持现状**：已由用户手动调整完成的当前/既有场景原样保留，不重新替换、重建、简化或重排当前资产，不重置用户已调整的尺寸、位置、颜色、人物轨迹和摄影机参数。
   - **一物一方块（仅基础立方体/长方体 `box`）**：新增预演仅用于验证人物位置、运动路线、动作节奏、镜头构图和遮挡关系；家具、车辆、建筑、树木、木架、火焰、烟雾等均用**单个外包围长方体**表示，不拆分横杆/立柱/木柴/轮胎/门窗/枝叶/内部零件，不制作粒子、火舌、体积效果或复杂物理模拟（仅当用户明确要求区分多个独立活动部分或关键遮挡时才增加必要方块）。
   - **人物、材质、强制无线条与真实米制**：人物沿用用户选定代理形态和固定身份色，不加人体/服装/面部/配件细化；无贴图哑光纯色（场景/道具优先低饱和灰，关键占位物用简单颜色区分）；**强制使用无线条（`FORCE_HIDE_EDGES=True` / `hide_edges=True`，禁用所有模型黑色结构边线与网格线框）**；真实米制尺寸与属性面板数值严格对应实际几何尺寸，保存重开与播放后保持一致。
   - **调度重心、交付核查与默认录屏导出**：重点打磨路线、速度、起止时间、摄影机节奏和构图，未经要求不改已确认落幅与镜头；交付前核查人数身份、路线速度、出入画顺序、前后遮挡、落幅位置及尺寸生效；**以后凡是说“导出 / 导出视频 / 导出 MP4 / 录屏”，一律默认使用内置【摄影机画面录屏插件】（`static/plugins/camera_screen_recorder.js` + `/api/save_camera_recording`，页面「导出视频」按钮 / 快捷键 `6` / `window.__cameraScreenRecorder.startRecording({ mode: 'full_shot', hideHud: true, saveToDownloads: true, saveToDesktop: false })`）录制纯净完整画幅（无界面、无角标、强制无线条）并自动转码为 `H.264 MP4`，按场景名称建文件夹保存至 `~/Downloads/<场景名>/`（同时保留在 `exports/`）；仅当用户明确要求“后端离线渲染 / Python 渲染”时，才调用后端 `POST /api/export_video` / `render_scene_to_mp4` 并同样归档到 `~/Downloads/<场景名>/`**。
   - 修改 `static/index.html` 后，始终同步复制到根目录 `index.html`。
5. **极速零试错执行管线（严禁盲试接口、环境或串行空转）**：
   - **唯一指定 Python 环境**：始终直接使用 `/Users/wangmu/Desktop/MagicDirectorStage_导演台完整软件/.venv/bin/python`，禁止盲试 `playwright` 或其他缺少 `PIL` / 依赖的外部虚拟环境。
   - **一步热推与工程保存**：更新场景后直接 `POST http://127.0.0.1:8765/api/hot_activate_scene`（携带 `scene`、`scene_id`、`scene_name`、`shot_id`、`shot_name`），一次完成内存热更新与 `saved_projects.json` 原子持久化；严禁盲猜 `/api/scene`、`/api/set_scene` 等不存在的路由或盲解 `saved_projects.json`（顶层固定为 `{activeSceneId, activeShotId, scenes, characterColors}`）。
   - **一步 WebGL 纯净抽帧与 Mannequin 站姿基准**：截图预览直接通过 Chrome `--headless=new --use-angle=metal --enable-webgl` + CDP 调用 `window.__director.renderCleanCameraFrame({ hideHud: true })`；`mannequin` 直立站姿默认 `pelvis=[0, 0.888, 0]`、`chest=[0, 1.348, 0.015]`、`foot_L/R y=0.04`（局部 `+X=左`、`-X=右`、`+Z=前`；与 `pawn` 保持同真实身高时乘 `CHARACTER_NEUTRAL_DIMENSIONS["pawn"][1] / CHARACTER_NEUTRAL_DIMENSIONS["mannequin"][1]`），一次跑通不返工。
