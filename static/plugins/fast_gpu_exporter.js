/**
 * Magic Director Stage · 极速离线 GPU 直出插件 (Fast WebCodecs GPU Exporter)
 *
 * 原理（完全不录屏、不等实时播放时间）：
 * 1. 调用 prepareCleanCameraBatchSession() 锁定干净摄影机离屏视口，按精确整数帧 t = f / fps 离线步进渲染 Three.js WebGL。
 * 2. 通过浏览器原生 WebCodecs VideoEncoder（Apple Silicon VideoToolbox H.264 硬编，Annex B 裸流）直接读取 WebGL Canvas。
 * 3. 将原始 H.264 比特流以二进制形式直传 /api/save_h264_mp4，后端执行 FFmpeg 无损封装 (-c:v copy +faststart) 秒级落盘桌面与 exports/。
 */
(function () {
  "use strict";

  const PLUGIN_ID = "fast_gpu_exporter";

  function injectStyles() {
    if (document.getElementById("mdsFastGpuExporterStyle")) return;
    const style = document.createElement("style");
    style.id = "mdsFastGpuExporterStyle";
    style.textContent = `
      .btn-fast-gpu-export {
        background: rgba(16, 185, 129, 0.16) !important;
        border-color: #10B981 !important;
        color: #6EE7B7 !important;
        font-weight: 600 !important;
      }
      .btn-fast-gpu-export:hover {
        background: rgba(16, 185, 129, 0.28) !important;
        color: #ECFDF5 !important;
      }
      .btn-fast-gpu-export.busy {
        opacity: 0.78;
        cursor: wait !important;
        border-color: #60A5FA !important;
        color: #BFDBFE !important;
      }
    `;
    document.head.appendChild(style);
  }

  async function pickBestH264Config(width, height, fps, bitrate) {
    const levelHex = width * height <= 1920 * 1088 ? "28" : "33";
    // Baseline (42E0) 无 B 帧，DTS 与 PTS 严格单调一致；quality 模式可释放硬件编码吞吐（实测比 realtime 快 ~2.5x）
    const candidates = [
      { codec: `avc1.42E0${levelHex}`, hardwareAcceleration: "prefer-hardware", latencyMode: "quality" },
      { codec: `avc1.42E0${levelHex}`, hardwareAcceleration: "no-preference", latencyMode: "quality" },
      { codec: `avc1.4d40${levelHex}`, hardwareAcceleration: "prefer-hardware", latencyMode: "quality" },
      { codec: `avc1.42E0${levelHex}`, hardwareAcceleration: "prefer-hardware", latencyMode: "realtime" },
      { codec: `avc1.6400${levelHex}`, hardwareAcceleration: "prefer-hardware", latencyMode: "realtime" },
      { codec: `avc1.42E0${levelHex}`, hardwareAcceleration: "no-preference", latencyMode: "realtime" },
    ];
    for (const c of candidates) {
      const cfg = {
        codec: c.codec,
        width,
        height,
        bitrate,
        framerate: fps,
        bitrateMode: "variable",
        hardwareAcceleration: c.hardwareAcceleration,
        latencyMode: c.latencyMode,
        avc: { format: "annexb" },
      };
      try {
        const res = await VideoEncoder.isConfigSupported(cfg);
        if (res && res.supported) return res.config || cfg;
      } catch (_) {}
    }
    return null;
  }

  function mergeUint8Arrays(chunks, totalBytes) {
    const out = new Uint8Array(totalBytes);
    let offset = 0;
    for (let i = 0; i < chunks.length; i++) {
      out.set(chunks[i], offset);
      offset += chunks[i].byteLength;
    }
    return out;
  }

  function mountPlugin() {
    const D = window.__director;
    if (!D || window.__fastGpuExporter) return;

    injectStyles();

    const state = {
      exporting: false,
      lastResult: null,
    };

    function setBusyLabel(text) {
      const topBtn = document.getElementById("btnTopBarFastGpuExport");
      const menuText = document.getElementById("menuFastGpuExportText");
      const mainExportBtn = document.getElementById("btnExportVideo");
      if (topBtn) {
        topBtn.classList.toggle("busy", state.exporting);
        topBtn.textContent = text || "⚡ 极速直出";
      }
      if (menuText) {
        menuText.textContent = text || "⚡ 极速 GPU 离线直出 MP4";
      }
      if (mainExportBtn) {
        mainExportBtn.textContent = state.exporting ? (text || "⚡ 导出中…") : "导出视频";
      }
    }

    async function exportVideo(customOpts = {}) {
      if (state.exporting) return { ok: false, error: "正在进行极速导出" };
      if (!customOpts?.fromCameraRecorder && (window.__cameraScreenRecorder?.isRecording || window.__cameraScreenRecorder?.isSaving)) {
        return { ok: false, error: "当前正在进行摄影机录屏，请先等待结束" };
      }
      if (typeof D.prepareCleanCameraBatchSession !== "function") {
        return { ok: false, error: "导演台离线渲染会话接口未就绪" };
      }
      if (typeof VideoEncoder === "undefined" || typeof VideoFrame === "undefined") {
        const msg = "当前浏览器不支持 WebCodecs VideoEncoder，请改用快捷键 6 摄影机录屏";
        D.showCameraScreenshotToast?.(msg, true);
        return { ok: false, error: msg };
      }

      const hideHud = customOpts.hideHud !== undefined ? Boolean(customOpts.hideHud) : true;
      const saveToDownloads = customOpts.saveToDownloads !== undefined ? Boolean(customOpts.saveToDownloads) : true;
      const saveToDesktop = customOpts.saveToDesktop !== undefined ? Boolean(customOpts.saveToDesktop) : false;

      state.exporting = true;
      setBusyLabel("⚡ 初始化…");

      const tTotalStart = performance.now();
      let session = null;
      let encoder = null;
      let encoderErr = null;
      const chunks = [];
      let totalBytes = 0;

      try {
        session = D.prepareCleanCameraBatchSession({
          hideHud,
          fps: customOpts.fps,
        });
        const { exportW, exportH, fps, duration, totalFrames, sceneLabel, shotLabel, aspectKey } = session;
        const defaultBitrate = Math.max(4_000_000, Math.min(20_000_000, Math.round(exportW * exportH * fps * 0.35)));
        const bitrate = Math.max(1_000_000, Number(customOpts.bitrate) || defaultBitrate);

        const encCfg = await pickBestH264Config(exportW, exportH, fps, bitrate);
        if (!encCfg) {
          throw new Error(`浏览器 VideoEncoder 不支持 ${exportW}×${exportH}@${fps} H.264 Annex-B 硬编配置`);
        }

        encoder = new VideoEncoder({
          output(chunk) {
            const buf = new Uint8Array(chunk.byteLength);
            chunk.copyTo(buf);
            chunks.push(buf);
            totalBytes += buf.byteLength;
          },
          error(err) {
            encoderErr = err;
          },
        });
        encoder.configure(encCfg);

        const frameDurUs = Math.round(1_000_000 / fps);
        const gopFrames = Math.max(1, fps);
        const MAX_QUEUE_DEPTH = 24;
        const DRAIN_TO_DEPTH = 12;
        const waitForQueueDrain = () =>
          new Promise((resolve) => {
            if (!encoder || encoder.encodeQueueSize < DRAIN_TO_DEPTH || encoderErr) {
              resolve();
              return;
            }
            let settled = false;
            let timer = null;
            const finish = () => {
              if (settled) return;
              settled = true;
              if (timer !== null) clearTimeout(timer);
              encoder?.removeEventListener?.("dequeue", onDeq);
              resolve();
            };
            const onDeq = () => {
              if (!encoder || encoder.encodeQueueSize < DRAIN_TO_DEPTH || encoderErr) {
                finish();
              }
            };
            if (typeof encoder.addEventListener === "function") {
              encoder.addEventListener("dequeue", onDeq);
              timer = setTimeout(finish, 120);
            } else {
              setTimeout(resolve, 4);
            }
          });

        const yieldToUi = () => new Promise((resolve) => setTimeout(resolve, 0));
        let lastYieldMs = performance.now();

        for (let fIdx = 0; fIdx < totalFrames; fIdx++) {
          if (encoderErr) throw encoderErr;
          const tSec = totalFrames > 1 ? Math.min(duration, fIdx / fps) : 0.0;
          const srcCanvas = session.renderFrameAtTime(tSec);
          const tsUs = Math.round((fIdx * 1_000_000) / fps);
          const vf = new VideoFrame(srcCanvas, { timestamp: tsUs, duration: frameDurUs });
          try {
            encoder.encode(vf, { keyFrame: fIdx === 0 || fIdx % gopFrames === 0 });
          } finally {
            vf.close();
          }
          if (encoder.encodeQueueSize >= MAX_QUEUE_DEPTH) {
            const pct = Math.min(99, Math.round(((fIdx + 1) / totalFrames) * 100));
            setBusyLabel(`⚡ ${pct}%`);
            await waitForQueueDrain();
            lastYieldMs = performance.now();
          } else if (performance.now() - lastYieldMs >= 16) {
            const pct = Math.min(99, Math.round(((fIdx + 1) / totalFrames) * 100));
            setBusyLabel(`⚡ ${pct}%`);
            await yieldToUi();
            lastYieldMs = performance.now();
          }
        }

        if (encoderErr) throw encoderErr;
        setBusyLabel("⚡ 封装MP4…");
        await encoder.flush();
        if (encoderErr) throw encoderErr;

        const renderEncodeMs = Math.round(performance.now() - tTotalStart);
        const mergedH264 = mergeUint8Arrays(chunks, totalBytes);
        const q = new URLSearchParams({
          fps: String(fps),
          scene_label: sceneLabel,
          shot_label: shotLabel,
          save_to_downloads: saveToDownloads ? "1" : "0",
          save_to_desktop: saveToDesktop ? "1" : "0",
        });
        if (customOpts && typeof customOpts.filename === "string" && customOpts.filename.trim()) {
          q.set("filename", customOpts.filename.trim());
        }

        const resp = await fetch(`/api/save_h264_mp4?${q.toString()}`, {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream" },
          body: mergedH264,
        }).then((r) => r.json());

        if (!resp || !resp.ok) {
          throw new Error(resp?.error || "后端 H.264 无损封装 MP4 失败");
        }

        const totalMs = Math.round(performance.now() - tTotalStart);
        const resultObj = {
          ok: true,
          ...resp,
          width: exportW,
          height: exportH,
          aspect_ratio: aspectKey,
          frames: totalFrames,
          fps,
          duration,
          codec_used: encCfg.codec,
          hw_accel: encCfg.hardwareAcceleration,
          latency_mode: encCfg.latencyMode,
          render_encode_ms: renderEncodeMs,
          total_ms: totalMs,
        };
        state.lastResult = resultObj;
        const targetWhere = resp.downloads_path || resp.export_path || resp.desktop_path || resp.filename;
        D.showCameraScreenshotToast?.(
          `⚡ 极速 GPU 导出完成 (${totalFrames} 帧 ${exportW}×${exportH} · ${(totalMs / 1000).toFixed(2)}s): ${targetWhere}`,
          false
        );
        D.logStatus?.(
          `⚡ 极速 GPU 离线直出完成 (${(totalMs / 1000).toFixed(2)}s，含无损封装 ${resp.mux_ms}ms): ${targetWhere}`
        );
        if (customOpts.openModal && resp.video_url && typeof D.openVideoModal === "function") {
          const modalMeta = `⚡ 极速 GPU 离线直出 ${exportW}×${exportH} (${aspectKey}) | ${totalFrames} 帧 · ${fps} FPS · 耗时 ${(totalMs / 1000).toFixed(2)}s`;
          D.openVideoModal(resp.video_url, modalMeta, targetWhere);
        }
        return resultObj;
      } catch (err) {
        const errMsg = String(err?.message || err);
        D.showCameraScreenshotToast?.(`极速导出失败: ${errMsg}`, true);
        D.logStatus?.(`❌ 极速导出失败: ${errMsg}`, true);
        return { ok: false, error: errMsg };
      } finally {
        if (encoder) {
          try {
            if (encoder.state !== "closed") encoder.close();
          } catch (_) {}
        }
        if (session) {
          try { session.dispose(); } catch (_) {}
        }
        state.exporting = false;
        setBusyLabel("");
      }
    }

    // 1. 顶部栏精简：不在“导出视频”旁额外注入「⚡ 极速直出」按钮（保留「导出 → ⚡ 极速 GPU 离线直出 MP4」与快捷键 7）

    // 2. 顶部「导出」菜单入口
    const exportSubParent = document.getElementById("exportScreenshotParent");
    if (exportSubParent && exportSubParent.parentElement && !document.getElementById("btnMenuFastGpuExport")) {
      const menuItem = document.createElement("button");
      menuItem.className = "cascade-item";
      menuItem.id = "btnMenuFastGpuExport";
      menuItem.title = "不录屏 · WebGL + WebCodecs 硬件直出纯净 MP4（快捷键 7）";
      menuItem.innerHTML = `
        <span class="menu-row-left">
          <svg class="menu-icon" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
          <span id="menuFastGpuExportText">⚡ 极速 GPU 离线直出 MP4</span>
        </span>
        <span class="menu-shortcut">7</span>
      `;
      const recMenu = document.getElementById("btnMenuCamScreenRecord") || exportSubParent;
      exportSubParent.parentElement.insertBefore(menuItem, recMenu);
      menuItem.addEventListener("click", (e) => {
        e.stopPropagation();
        if (typeof D.closeAllTopMenus === "function") D.closeAllTopMenus();
        exportVideo();
      });
    }

    // 3. 快捷键面板与全局键位 7
    if (Array.isArray(D.APP_SHORTCUT_ACTIONS) && !D.APP_SHORTCUT_ACTIONS.some((a) => a.id === "fast_gpu_export")) {
      const recIdx = D.APP_SHORTCUT_ACTIONS.findIndex((a) => a.id === "cam_screen_record");
      const actionDef = {
        id: "fast_gpu_export",
        group: "视角与摄影机",
        label: "极速 GPU 离线直出 MP4 (不录屏)",
        kbd: "7",
        isActive: () => Boolean(state.exporting),
        run: () => exportVideo(),
      };
      if (recIdx >= 0) D.APP_SHORTCUT_ACTIONS.splice(recIdx + 1, 0, actionDef);
      else D.APP_SHORTCUT_ACTIONS.push(actionDef);
      if (typeof D.renderShortcutsDrawerItems === "function") {
        try { D.renderShortcutsDrawerItems(); } catch (_) {}
      }
    }

    window.addEventListener("keydown", (ev) => {
      if (ev.defaultPrevented || ev.metaKey || ev.ctrlKey || ev.altKey) return;
      const tag = document.activeElement?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || document.activeElement?.isContentEditable) {
        return;
      }
      if ((ev.key === "7" || ev.code === "Digit7") && !ev.repeat) {
        ev.preventDefault();
        exportVideo();
      }
    });

    const api = {
      id: PLUGIN_ID,
      get isExporting() { return state.exporting; },
      get lastResult() { return state.lastResult; },
      exportVideo,
    };
    window.__fastGpuExporter = api;
    D.fastGpuExporter = api;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mountPlugin);
  } else {
    mountPlugin();
  }
})();
