# Magic Director Stage 3D 导演台（完整独立软件包）

## 一、快速启动方式
1. **方式 A（推荐）**：直接双击目录中的 **`Magic Director Stage.app`** 或 **`双击启动导演台.command`**，即可自动启动本地服务并在浏览器打开 `http://127.0.0.1:8765/`。
2. **方式 B（命令行启动）**：
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   .venv/bin/python3 server.py 8765
   ```

## 二、目录结构说明
- `Magic Director Stage.app` / `双击启动导演台.command`：macOS 一键启动器
- `server.py`：本地 Web 服务与 API 后端（工程保存、桌面截图写入、视频渲染导出）
- `director_engine.py`：Python 3D 姿态解算、UAZ-469/场景几何建模与离线视频渲染引擎
- `static/`：前端单页工作台（`index.html`）及本地离线 Three.js 依赖（`static/vendor/`，无需联网即可运行）
- `saved_projects.json`：已保存的全部场次与镜头工程库（默认第 1 场为 UAZ-469 吉普车四人舞台）
- `scenes_backup/`：独立导出的 UAZ-469 场景 JSON、Python 脚本、机位静帧 PNG 与 MP4 预演视频
- `ref_videos/`：本地参考视频目录
- `exports/`：视频与对照图渲染输出目录

## 三、核心快捷键
- `1`：浏览模式
- `2`：物体模式（配合 `G` 移动 / `R` 旋转 / `T` 缩放）
- `3`：姿态 IK 模式（直接拖拽双肩、胸廓、手腕、手肘、视线、骨盆、膝盖、脚踝控制球或文字标签）
- `4`：一键将当前右侧摄影机画面截图到剪贴板（可直接 Cmd+V 粘贴）
- `5`：一键将当前右侧摄影机画面按画幅保存 PNG 到桌面
- `0`：摄影机编辑模式
- `` ` ``（反引号键）：打开 / 关闭键盘快捷键面板
