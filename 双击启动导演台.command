#!/bin/bash
cd "$(dirname "$0")"
PORT=8765

# 优先选用已安装 numpy + PIL + imageio_ffmpeg 的 Python 解释器
PYTHON_BIN=""
for cand in \
  "$PWD/.venv/bin/python3" \
  "/Users/wangmu/.gemini/antigravity/scratch/scene75_blockout/.venv/bin/python3" \
  "$(which python3)" \
  "/usr/bin/python3"; do
  if [ -n "$cand" ] && [ -x "$cand" ]; then
    if "$cand" -c "import numpy, PIL, imageio_ffmpeg" >/dev/null 2>&1; then
      PYTHON_BIN="$cand"
      break
    fi
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "正在初始化本地 Python 虚拟环境 (.venv)..."
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  PYTHON_BIN="$PWD/.venv/bin/python3"
fi

if lsof -iTCP:$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo "🎬 检测到导演台服务已在端口 $PORT 运行，正在打开浏览器..."
  open "http://127.0.0.1:$PORT/"
  exit 0
fi

echo "🎬 正在启动 Magic Director Stage 3D 导演台 (端口 $PORT)..."
(sleep 1.2 && open "http://127.0.0.1:$PORT/") &
exec "$PYTHON_BIN" server.py $PORT
