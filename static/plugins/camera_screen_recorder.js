/**
 * Magic Director Stage · 摄影机画面纯净录屏插件 (Camera Screen Recorder Plugin)
 *
 * 功能特点：
 * 1. 直接录制浏览器右侧摄影机预览画幅（保留完整宽高比，默认不带软件界面、不带 HUD 角标，也不触发后端导出三问）。
 * 2. 两种录屏模式：
 *    - 「录完整本镜（自动从头播到尾）」：回到起点 -> 开启录屏 -> 自动播放至末帧自动停录。
 *    - 「手动自由录屏」：随时点开始、边调机位/播放边录，再点停止结束。
 * 3. 录完自动通过本地后端转换为通用 H.264 MP4 并保存到桌面（同时保存在 exports/ 目录，若离线则自动回退浏览器本地下载）。
 * 4. 快捷键：数字键 `6` 一键开始 / 停止录屏。
 */
(function () {
  "use strict";

  const PLUGIN_ID = "camera_screen_recorder";

  function injectPluginStyles() {
    if (document.getElementById("mdsCamRecorderPluginStyle")) return;
    const style = document.createElement("style");
    style.id = "mdsCamRecorderPluginStyle";
    style.textContent = `
      .cam-record-group {
        position: relative;
        display: inline-flex;
        align-items: center;
        gap: 2px;
        pointer-events: auto;
      }
      .cam-record-group.is-floating-fallback {
        position: absolute;
        top: 7px;
        right: 46px;
        z-index: 26;
      }
      .cam-rec-btn {
        height: 22px;
        padding: 0 5px 0 3px;
        background: transparent;
        border: 1px solid transparent;
        color: #4ADE80;
        border-radius: 4px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        cursor: pointer;
        user-select: none;
        font-family: "SF Mono", "Menlo", "DIN Alternate", monospace;
        font-size: 12.5px;
        font-weight: 700;
        letter-spacing: 0.5px;
        white-space: nowrap;
        transition: background 0.12s ease, border-color 0.12s ease, color 0.12s ease;
      }
      .cam-rec-btn:hover {
        background: rgba(74, 222, 128, 0.12);
        border-color: rgba(74, 222, 128, 0.35);
      }
      .cam-rec-btn.recording,
      .cam-rec-btn.saving {
        background: rgba(239, 68, 68, 0.16);
        border-color: rgba(248, 113, 113, 0.55);
        color: #F87171;
      }
      .cam-rec-btn.saving {
        cursor: wait;
      }
      .cam-arri-dot {
        width: 9.5px;
        height: 9.5px;
        border-radius: 50%;
        background: #4ADE80;
        box-shadow: 0 0 8px rgba(74, 222, 128, 0.78);
        display: inline-block;
        flex-shrink: 0;
      }
      .cam-rec-btn.recording .cam-arri-dot,
      .cam-rec-btn.saving .cam-arri-dot {
        background: #EF4444;
        box-shadow: 0 0 9px rgba(239, 68, 68, 0.9);
        animation: mdsRecPulse 1s ease-in-out infinite;
      }
      @keyframes mdsRecPulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.45; transform: scale(0.88); }
      }
      .cam-rec-gear-btn {
        width: 16px;
        height: 22px;
        padding: 0;
        background: transparent;
        border: 1px solid transparent;
        color: #64748B;
        border-radius: 4px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        font-size: 10px;
        transition: background 0.12s ease, border-color 0.12s ease, color 0.12s ease;
      }
      .cam-rec-gear-btn:hover,
      .cam-rec-gear-btn.open {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(96, 165, 250, 0.45);
        color: #E2E8F0;
      }
      .cam-rec-popover {
        position: absolute;
        bottom: 36px;
        left: 8px;
        z-index: 38;
        width: 238px;
        background: rgba(20, 24, 30, 0.97);
        border: 1px solid #3B4452;
        border-radius: 8px;
        padding: 9px 10px;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.58);
        color: #E2E8F0;
        font-size: 11.5px;
        display: none;
        flex-direction: column;
        gap: 7px;
        pointer-events: auto;
      }
      .cam-rec-popover.open {
        display: flex;
      }
      .cam-rec-popover-title {
        font-size: 11px;
        font-weight: 700;
        color: #94A3B8;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #2D3542;
        padding-bottom: 5px;
      }
      .cam-rec-opt-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        cursor: pointer;
        user-select: none;
      }
      .cam-rec-opt-row input[type="checkbox"] {
        accent-color: #5B9DFF;
        cursor: pointer;
      }
      .cam-rec-select {
        background: #161A20;
        border: 1px solid #384150;
        color: #F1F5F9;
        border-radius: 4px;
        height: 22px;
        font-size: 11px;
        padding: 0 5px;
      }
      .cam-rec-hint {
        font-size: 10.5px;
        color: #8A94A6;
        line-height: 1.38;
        border-top: 1px solid #282F3B;
        padding-top: 5px;
      }
    `;
    document.head.appendChild(style);
  }

  function pickSupportedMimeType() {
    if (typeof MediaRecorder === "undefined") return "";
    const candidates = [
      "video/mp4;codecs=avc1.42E01E",
      "video/mp4",
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm",
    ];
    for (const m of candidates) {
      try {
        if (MediaRecorder.isTypeSupported(m)) return m;
      } catch (_) {}
    }
    return "";
  }

  function formatDurationSec(sec) {
    const s = Math.max(0, Number(sec) || 0);
    const mins = Math.floor(s / 60);
    const rem = s - mins * 60;
    return mins > 0 ? `${mins}:${rem.toFixed(1).padStart(4, "0")}` : `${rem.toFixed(1)}s`;
  }

  function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(String(reader.result || ""));
      reader.onerror = (e) => reject(e);
      reader.readAsDataURL(blob);
    });
  }

  function triggerBrowserDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  }

  function mountPlugin() {
    const D = window.__director;
    if (!D || !D.camPaneDom || window.__cameraScreenRecorder) return;

    injectPluginStyles();

    const state = {
      recording: false,
      saving: false,
      mode: "full_shot", // "full_shot" | "free"
      fpsMode: "60",     // "60" | "project"
      hideHud: true,     // 默认不带任何角标与界面，保留纯净摄影机画幅
      saveToDownloads: true,
      saveToDesktop: false,
      startedAtMs: 0,
      prevLoopState: false,
      captureCanvas: document.createElement("canvas"),
      stream: null,
      mediaRecorder: null,
      chunks: [],
      mimeType: "",
      lastFrameMeta: null,
      lastSavedResult: null,
      stopResolver: null,
      autoStopping: false,
      openModalOnFinish: false,
    };

    // 1. 底部 ARRI 监视器状态栏 STBY 录屏按钮组（嵌入 #arriStbySlot）
    const stbyHost = document.getElementById("arriStbySlot");
    const recGroup = document.createElement("div");
    recGroup.className = "cam-record-group" + (stbyHost ? "" : " is-floating-fallback");
    recGroup.id = "camRecordPluginGroup";
    recGroup.innerHTML = `
      <button class="cam-rec-btn" id="btnCamScreenRecord" title="STBY：点击录制纯净摄影机画面 (快捷键 6) · 不带软件界面、不走导出确认" aria-label="录屏">
        <span class="cam-arri-dot" aria-hidden="true"></span>
        <span id="camRecBtnLabel">STBY</span>
      </button>
      <button class="cam-rec-gear-btn" id="btnCamRecSettings" title="录屏模式与帧率设置" aria-label="录屏设置">▾</button>
    `;
    (stbyHost || D.camPaneDom).appendChild(recGroup);

    const popover = document.createElement("div");
    popover.className = "cam-rec-popover";
    popover.id = "camRecSettingsPopover";
    popover.innerHTML = `
      <div class="cam-rec-popover-title">
        <span>摄影机画面录屏插件</span>
        <span style="font-weight:500;color:#64748B;">快捷键 6</span>
      </div>
      <label class="cam-rec-opt-row" title="自动回到本镜起点播放至末帧并自动停录；或手动自由控制起止">
        <span>录制方式</span>
        <select class="cam-rec-select" id="selCamRecMode">
          <option value="full_shot" selected>完整本镜 (播完即停)</option>
          <option value="free">手动起止 (自由操作)</option>
        </select>
      </label>
      <label class="cam-rec-opt-row" title="60fps 消除高刷屏幕上的 24fps 步进顿挫感；也可按工程原始帧率导出">
        <span>导出帧率</span>
        <select class="cam-rec-select" id="selCamRecFps">
          <option value="60" selected>60 fps (丝滑流畅)</option>
          <option value="project">跟随工程帧率</option>
        </select>
      </label>
      <label class="cam-rec-opt-row" title="默认纯净无界面、无文字角标；勾选则在左上角附带焦距与机高">
        <span>叠加焦距/机高角标</span>
        <input type="checkbox" id="chkCamRecShowHud" />
      </label>
      <label class="cam-rec-opt-row" title="录完自动转码为 H.264 MP4 并按场景名归档到 macOS 下载文件夹">
        <span>按场景存到「下载」文件夹</span>
        <input type="checkbox" id="chkCamRecDownloads" checked />
      </label>
      <label class="cam-rec-opt-row" title="额外同步保存一份到 macOS 桌面">
        <span>同时另存一份到桌面</span>
        <input type="checkbox" id="chkCamRecDesktop" />
      </label>
      <div class="cam-rec-hint">
        已设为默认视频导出方式：直接录制右侧摄影机纯净画幅（无界面、强制无线条），自动转码 H.264 MP4 按场景名存至「下载」文件夹。
      </div>
    `;
    D.camPaneDom.appendChild(popover);

    // 2. 顶部「导出」菜单与模式栏快捷入口
    const exportSubParent = document.getElementById("exportScreenshotParent");
    if (exportSubParent && exportSubParent.parentElement && !document.getElementById("btnMenuCamScreenRecord")) {
      const menuItem = document.createElement("button");
      menuItem.className = "cascade-item";
      menuItem.id = "btnMenuCamScreenRecord";
      menuItem.title = "直接录制摄影机预览纯画面并保存为 MP4（不带软件界面，快捷键 6）";
      menuItem.innerHTML = `
        <span class="menu-row-left">
          <svg class="menu-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3.5" fill="currentColor"/></svg>
          <span id="menuCamRecText">摄影机录屏</span>
        </span>
        <span class="menu-shortcut">6</span>
      `;
      exportSubParent.parentElement.insertBefore(menuItem, exportSubParent);
      menuItem.addEventListener("click", (e) => {
        e.stopPropagation();
        if (typeof D.closeAllTopMenus === "function") D.closeAllTopMenus();
        toggleRecording();
      });
    }

    // 顶部栏精简：不在“导出视频”旁额外注入「● 录屏」按钮（保留「导出 → 摄影机录屏」、右视口组与快捷键 6）

    // 3. 注册进全局快捷键面板 APP_SHORTCUT_ACTIONS
    if (Array.isArray(D.APP_SHORTCUT_ACTIONS) && !D.APP_SHORTCUT_ACTIONS.some((a) => a.id === "cam_screen_record")) {
      const deskIdx = D.APP_SHORTCUT_ACTIONS.findIndex((a) => a.id === "cam_snap_desk");
      const actionDef = {
        id: "cam_screen_record",
        group: "视角与摄影机",
        label: "摄影机画面录屏 (起/止)",
        kbd: "6",
        isActive: () => Boolean(state.recording),
        run: () => toggleRecording(),
      };
      if (deskIdx >= 0) D.APP_SHORTCUT_ACTIONS.splice(deskIdx + 1, 0, actionDef);
      else D.APP_SHORTCUT_ACTIONS.push(actionDef);
      if (typeof D.renderShortcutsDrawerItems === "function") {
        try { D.renderShortcutsDrawerItems(); } catch (_) {}
      }
    }

    const btnRec = document.getElementById("btnCamScreenRecord");
    const btnRecLabel = document.getElementById("camRecBtnLabel");
    const btnGear = document.getElementById("btnCamRecSettings");
    const selMode = document.getElementById("selCamRecMode");
    const selFps = document.getElementById("selCamRecFps");
    const chkHud = document.getElementById("chkCamRecShowHud");
    const chkDownloads = document.getElementById("chkCamRecDownloads");
    const chkDesktop = document.getElementById("chkCamRecDesktop");

    function syncUI() {
      const topBarBtn = document.getElementById("btnTopBarCamRecord");
      const topBarLabel = document.getElementById("topBarCamRecLabel");
      const menuText = document.getElementById("menuCamRecText");
      const mainExportBtn = document.getElementById("btnExportVideo");

      if (state.saving) {
        btnRec.classList.remove("recording");
        btnRec.classList.add("saving");
        btnRec.title = "正在转码保存摄影机录屏…";
        btnRecLabel.textContent = "REC";
        if (topBarLabel) topBarLabel.textContent = "保存中…";
        if (menuText) menuText.textContent = "正在保存录屏…";
        if (mainExportBtn) mainExportBtn.textContent = "⏳ 保存录屏…";
        return;
      }

      btnRec.classList.remove("saving");
      if (state.recording) {
        const elapsed = (performance.now() - state.startedAtMs) / 1000;
        btnRec.classList.add("recording");
        btnRec.title = `正在录屏 (${formatDurationSec(elapsed)}) · 点击或按 6 停止并保存`;
        btnRecLabel.textContent = `REC ${formatDurationSec(elapsed)}`;
        if (topBarBtn) topBarBtn.classList.add("active");
        if (topBarLabel) topBarLabel.textContent = `停止 ${formatDurationSec(elapsed)}`;
        if (menuText) menuText.textContent = "停止摄影机录屏";
        if (mainExportBtn) mainExportBtn.textContent = `⏹ 停止 ${formatDurationSec(elapsed)}`;
      } else {
        btnRec.classList.remove("recording");
        btnRec.title = "STBY：点击录制纯净摄影机画面 (快捷键 6) · 不带软件界面、不走导出确认";
        btnRecLabel.textContent = "STBY";
        if (topBarBtn) topBarBtn.classList.remove("active");
        if (topBarLabel) topBarLabel.textContent = "录屏";
        if (menuText) menuText.textContent = "摄影机录屏";
        if (mainExportBtn) mainExportBtn.textContent = "导出视频";
      }
    }

    function pushCleanCameraFrameToStream() {
      if (!state.recording || !D.sceneData) return;
      const meta = D.renderCleanCameraFrame({
        hideHud: state.hideHud,
        targetCanvas: state.captureCanvas,
        skipDataUrl: true,
      });
      state.lastFrameMeta = meta;
      if (state.stream) {
        const vTrack = state.stream.getVideoTracks?.()[0];
        if (vTrack && typeof vTrack.requestFrame === "function") {
          try { vTrack.requestFrame(); } catch (_) {}
        }
      }
    }

    async function startRecording(customOpts = {}) {
      if (state.recording || state.saving) return { ok: false, error: "当前正在录制或保存中" };
      if (!D.sceneData) {
        D.showCameraScreenshotToast?.("录屏失败: 当前场景尚未初始化", true);
        return { ok: false, error: "当前场景尚未初始化" };
      }
      if (typeof MediaRecorder === "undefined" || typeof state.captureCanvas.captureStream !== "function") {
        D.showCameraScreenshotToast?.("当前浏览器不支持 MediaRecorder 视口录屏", true);
        return { ok: false, error: "浏览器不支持 MediaRecorder" };
      }

      if (customOpts && typeof customOpts.mode === "string") {
        state.mode = customOpts.mode;
        if (selMode) selMode.value = state.mode;
      }
      if (customOpts && typeof customOpts.hideHud === "boolean") {
        state.hideHud = customOpts.hideHud;
        if (chkHud) chkHud.checked = !state.hideHud;
      }
      if (customOpts && typeof customOpts.saveToDownloads === "boolean") {
        state.saveToDownloads = customOpts.saveToDownloads;
        if (chkDownloads) chkDownloads.checked = state.saveToDownloads;
      }
      if (customOpts && typeof customOpts.saveToDesktop === "boolean") {
        state.saveToDesktop = customOpts.saveToDesktop;
        if (chkDesktop) chkDesktop.checked = state.saveToDesktop;
      }
      state.openModalOnFinish = Boolean(customOpts && customOpts.openModal);

      popover.classList.remove("open");
      btnGear.classList.remove("open");

      const projectFps = Math.max(12, Math.min(60, Number(D.sceneData?.fps || 24)));
      const explicitFps = Number(customOpts?.fps);
      const fps = Number.isFinite(explicitFps) && explicitFps > 0
        ? Math.max(12, Math.min(60, Math.round(explicitFps)))
        : (state.fpsMode === "project" ? projectFps : 60);
      state.prevLoopState = Boolean(D.isLoopPlayback);

      if (state.mode === "full_shot") {
        // 完整本镜优先走确定性 WebCodecs 逐帧步进直出，彻底消除浏览器实时 MediaRecorder 掉帧/卡顿
        if (window.__fastGpuExporter && typeof window.__fastGpuExporter.exportVideo === "function") {
          try {
            state.saving = true;
            syncUI();
            const gpuRes = await window.__fastGpuExporter.exportVideo({
              fromCameraRecorder: true,
              fps,
              hideHud: state.hideHud,
              saveToDownloads: state.saveToDownloads,
              saveToDesktop: state.saveToDesktop,
            });
            if (gpuRes && gpuRes.ok) {
              state.lastSavedResult = gpuRes;
              if (state.openModalOnFinish && gpuRes.video_url && typeof D.openVideoModal === "function") {
                const modalMeta = `纯净画幅导出 ${gpuRes.width}×${gpuRes.height} | ${gpuRes.fps || fps} FPS | 零掉帧`;
                const targetWhere = gpuRes.downloads_path || gpuRes.desktop_path || gpuRes.export_path || gpuRes.filename;
                D.openVideoModal(gpuRes.video_url, modalMeta, targetWhere);
              }
              return gpuRes;
            }
          } catch (_) {
            // 若硬件编码不可用则静默回退实时 MediaRecorder 录屏
          } finally {
            state.saving = false;
            syncUI();
          }
        }

        D.setPlaybackState(false, "cam_rec_prepare");
        if (D.isLoopPlayback) {
          if (typeof D.toggleLoopPlaybackState === "function") D.toggleLoopPlaybackState();
          else D.isLoopPlayback = false;
        }
        D.jumpPlayheadToStart?.();
        D.applyObjectsAtTime?.(D.currentTime);
        D.applyCameraAtTime?.(D.currentTime);
      }

      // 先渲染首帧确定画布精确画幅宽高
      const firstMeta = D.renderCleanCameraFrame({
        hideHud: state.hideHud,
        targetCanvas: state.captureCanvas,
        skipDataUrl: true,
      });
      state.lastFrameMeta = firstMeta;

      const mimeType = pickSupportedMimeType();
      state.mimeType = mimeType || "video/webm";
      state.chunks = [];
      state.autoStopping = false;

      // 使用 captureStream(fps) 保证稳定帧率，并在每帧渲染后推送最新摄影机画面
      const stream = state.captureCanvas.captureStream(fps);
      state.stream = stream;

      const recOptions = { videoBitsPerSecond: 12_000_000 };
      if (mimeType) recOptions.mimeType = mimeType;

      let recorder;
      try {
        recorder = new MediaRecorder(stream, recOptions);
      } catch (_) {
        recorder = new MediaRecorder(stream);
      }
      state.mediaRecorder = recorder;

      recorder.ondataavailable = (ev) => {
        if (ev.data && ev.data.size > 0) {
          state.chunks.push(ev.data);
        }
      };

      recorder.onstop = async () => {
        const recordedMime = recorder.mimeType || state.mimeType || "video/webm";
        const blob = new Blob(state.chunks, { type: recordedMime });
        state.chunks = [];
        if (state.stream) {
          state.stream.getTracks().forEach((t) => t.stop());
          state.stream = null;
        }
        state.mediaRecorder = null;
        state.recording = false;
        state.saving = true;
        syncUI();

        let resultObj = null;
        try {
          const b64 = await blobToBase64(blob);
          const meta = state.lastFrameMeta || {};
          const resp = await fetch("/api/save_camera_recording", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              video_base64: b64,
              mime_type: recordedMime,
              scene_label: meta.sceneName || meta.sceneLabel || "场次",
              shot_label: meta.shotName || meta.shotLabel || "镜头",
              fps,
              save_to_downloads: state.saveToDownloads,
              save_to_desktop: state.saveToDesktop,
            }),
          }).then((r) => r.json());

          if (resp && resp.ok) {
            resultObj = {
              ok: true,
              ...resp,
              width: meta.exportW || state.captureCanvas.width,
              height: meta.exportH || state.captureCanvas.height,
            };
            state.lastSavedResult = resultObj;
            const targetWhere = resp.downloads_path || resp.desktop_path || resp.export_path || resp.filename;
            D.showCameraScreenshotToast?.(
              `已保存摄影机录屏 (${resultObj.width}×${resultObj.height}): ${targetWhere}`,
              false
            );
            D.logStatus?.(`🎥 摄影机画面录屏已保存: ${targetWhere}`);
            if (state.openModalOnFinish && resp.video_url && typeof D.openVideoModal === "function") {
              const modalMeta = `录屏导出 ${resultObj.width}×${resultObj.height} | ${fps} FPS | 无线条纯净画幅`;
              D.openVideoModal(resp.video_url, modalMeta, targetWhere);
            }
          } else {
            throw new Error(resp?.error || "后端保存失败");
          }
        } catch (err) {
          // 离线或后端异常时自动触发浏览器本地下载兜底
          const ext = recordedMime.includes("mp4") ? "mp4" : "webm";
          const fallbackName = `录屏_${Date.now()}.${ext}`;
          triggerBrowserDownload(blob, fallbackName);
          resultObj = {
            ok: true,
            fallback_download: true,
            filename: fallbackName,
            error: String(err?.message || err),
          };
          state.lastSavedResult = resultObj;
          D.showCameraScreenshotToast?.(`录屏已通过浏览器下载保存: ${fallbackName}`, false);
        } finally {
          if (state.mode === "full_shot" && state.prevLoopState && !D.isLoopPlayback) {
            if (typeof D.toggleLoopPlaybackState === "function") D.toggleLoopPlaybackState();
            else D.isLoopPlayback = true;
          }
          state.saving = false;
          syncUI();
          if (typeof state.stopResolver === "function") {
            const fn = state.stopResolver;
            state.stopResolver = null;
            fn(resultObj);
          }
        }
      };

      state.recording = true;
      state.startedAtMs = performance.now();
      recorder.start(200);
      pushCleanCameraFrameToStream();
      syncUI();

      if (state.mode === "full_shot") {
        D.setPlaybackState(true, "cam_rec_AutoPlay");
        D.showCameraScreenshotToast?.(
          `正在录制完整本镜 (${firstMeta.exportW}×${firstMeta.exportH} 纯净画幅)… 播完自动保存`,
          false
        );
      } else {
        D.showCameraScreenshotToast?.(
          `已开始摄影机录屏 (${firstMeta.exportW}×${firstMeta.exportH}) · 再按 6 或点击停止`,
          false
        );
      }

      return {
        ok: true,
        width: firstMeta.exportW,
        height: firstMeta.exportH,
        mode: state.mode,
      };
    }

    function stopRecording() {
      if (!state.recording || !state.mediaRecorder) {
        return Promise.resolve(state.lastSavedResult || { ok: false, error: "当前未在录屏" });
      }
      return new Promise((resolve) => {
        state.stopResolver = resolve;
        try {
          pushCleanCameraFrameToStream();
          if (state.mode === "full_shot" && D.isPlaying) {
            D.setPlaybackState(false, "cam_rec_stop");
          }
          if (state.mediaRecorder.state !== "inactive") {
            state.mediaRecorder.stop();
          } else {
            resolve(state.lastSavedResult);
          }
        } catch (e) {
          state.recording = false;
          syncUI();
          resolve({ ok: false, error: String(e?.message || e) });
        }
      });
    }

    function toggleRecording(customOpts) {
      if (state.saving) return Promise.resolve({ ok: false, error: "正在保存上一段录屏" });
      if (state.recording) return stopRecording();
      return startRecording(customOpts);
    }

    // 4. 挂载视口渲染前钩子：先抓取纯净离屏摄影机帧（避免 renderer.setSize 清空屏幕上左右视口画面），再让正常左右视口照常渲染显示！
    const prevBeforeHook = window.__onDirectorBeforeViewportRender;
    window.__onDirectorBeforeViewportRender = function (now) {
      if (typeof prevBeforeHook === "function") {
        try { prevBeforeHook(now); } catch (_) {}
      }
      if (!state.recording) return;
      pushCleanCameraFrameToStream();
      syncUI();

      if (state.mode === "full_shot" && !state.autoStopping) {
        const elapsedMs = performance.now() - state.startedAtMs;
        // 当时间轴自然播放到末帧自动停下（isPlaying 变为 false）且已至少录制 250ms，自动结束录屏并保存
        if (!D.isPlaying && elapsedMs > 250) {
          state.autoStopping = true;
          setTimeout(() => {
            if (state.recording) stopRecording();
          }, 120);
        }
      }
    };

    // 5. UI 交互事件与快捷键 6 绑定
    btnRec.addEventListener("pointerdown", (e) => e.stopPropagation());
    btnRec.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleRecording();
    });

    btnGear.addEventListener("pointerdown", (e) => e.stopPropagation());
    btnGear.addEventListener("click", (e) => {
      e.stopPropagation();
      const nextOpen = !popover.classList.contains("open");
      if (nextOpen) {
        ["arriLensPopover", "arriAspectPopover"].forEach((id) => document.getElementById(id)?.classList.remove("open"));
        ["btnArriLensPicker", "btnArriAspectPicker"].forEach((id) => document.getElementById(id)?.classList.remove("open"));
      }
      popover.classList.toggle("open", nextOpen);
      btnGear.classList.toggle("open", nextOpen);
    });

    popover.addEventListener("pointerdown", (e) => e.stopPropagation());
    popover.addEventListener("click", (e) => e.stopPropagation());

    selMode?.addEventListener("change", () => {
      state.mode = selMode.value === "free" ? "free" : "full_shot";
    });
    selFps?.addEventListener("change", () => {
      state.fpsMode = selFps.value === "project" ? "project" : "60";
    });
    chkHud?.addEventListener("change", () => {
      state.hideHud = !chkHud.checked;
    });
    chkDownloads?.addEventListener("change", () => {
      state.saveToDownloads = Boolean(chkDownloads.checked);
    });
    chkDesktop?.addEventListener("change", () => {
      state.saveToDesktop = Boolean(chkDesktop.checked);
    });

    document.addEventListener("pointerdown", (ev) => {
      if (!popover.classList.contains("open")) return;
      if (popover.contains(ev.target) || recGroup.contains(ev.target)) return;
      popover.classList.remove("open");
      btnGear.classList.remove("open");
    });

    window.addEventListener("keydown", (ev) => {
      if (ev.defaultPrevented || ev.metaKey || ev.ctrlKey || ev.altKey) return;
      const tag = document.activeElement?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || document.activeElement?.isContentEditable) {
        return;
      }
      if ((ev.key === "6" || ev.code === "Digit6") && !ev.repeat) {
        ev.preventDefault();
        toggleRecording();
      }
    });

    // 暴露给自动化测试与控制台调用的插件句柄
    const api = {
      id: PLUGIN_ID,
      get isRecording() { return state.recording; },
      get isSaving() { return state.saving; },
      get mode() { return state.mode; },
      set mode(v) {
        state.mode = v === "free" ? "free" : "full_shot";
        if (selMode) selMode.value = state.mode;
      },
      get fpsMode() { return state.fpsMode; },
      set fpsMode(v) {
        state.fpsMode = v === "project" ? "project" : "60";
        if (selFps) selFps.value = state.fpsMode;
      },
      get hideHud() { return state.hideHud; },
      set hideHud(v) {
        state.hideHud = Boolean(v);
        if (chkHud) chkHud.checked = !state.hideHud;
      },
      get lastSavedResult() { return state.lastSavedResult; },
      startRecording,
      stopRecording,
      toggleRecording,
    };
    window.__cameraScreenRecorder = api;
    D.cameraScreenRecorder = api;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mountPlugin);
  } else {
    mountPlugin();
  }
})();
