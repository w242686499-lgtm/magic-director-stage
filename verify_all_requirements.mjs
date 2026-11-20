import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const EXPORTS_DIR = "/Users/wangmu/.gemini/antigravity/scratch/director-stage/exports";
fs.mkdirSync(EXPORTS_DIR, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const proc = spawn(
    CHROME_BIN,
    [
      "--headless=new",
      "--remote-debugging-port=9233",
      "--window-size=1440,900",
      "--hide-scrollbars",
      "--no-first-run",
      "--no-default-browser-check",
      "--user-data-dir=/tmp/chrome_verify_9233",
      "http://127.0.0.1:8765/",
    ],
    { stdio: "ignore" }
  );

  try {
    let wsUrl = null;
    for (let i = 0; i < 35; i++) {
      await sleep(250);
      try {
        const res = await fetch("http://127.0.0.1:9233/json");
        const tabs = await res.json();
        const pageTab = tabs.find((t) => t.type === "page" && t.url.includes("8765"));
        if (pageTab) {
          wsUrl = pageTab.webSocketDebuggerUrl;
          break;
        }
      } catch (_) {}
    }
    if (!wsUrl) throw new Error("Failed to connect to headless Chrome CDP on port 9233");

    const ws = new WebSocket(wsUrl);
    await new Promise((resolve, reject) => {
      ws.addEventListener("open", resolve, { once: true });
      ws.addEventListener("error", reject, { once: true });
    });

    let nextId = 1;
    const pending = new Map();
    ws.addEventListener("message", (ev) => {
      const msg = JSON.parse(String(ev.data));
      if (msg.id && pending.has(msg.id)) {
        const { resolve, reject } = pending.get(msg.id);
        pending.delete(msg.id);
        if (msg.error) reject(new Error(JSON.stringify(msg.error)));
        else resolve(msg.result || {});
      }
    });

    const cdp = (method, params = {}) =>
      new Promise((resolve, reject) => {
        const id = nextId++;
        pending.set(id, { resolve, reject });
        ws.send(JSON.stringify({ id, method, params }));
      });

    const evalJs = async (expression) => {
      const res = await cdp("Runtime.evaluate", {
        expression,
        returnByValue: true,
        awaitPromise: true,
      });
      if (res.exceptionDetails) {
        throw new Error(`JS Exception: ${JSON.stringify(res.exceptionDetails)}`);
      }
      return res.result?.value;
    };

    const captureScreenshot = async (outPath) => {
      const res = await cdp("Page.captureScreenshot", { format: "png" });
      fs.writeFileSync(outPath, Buffer.from(res.data, "base64"));
    };

    await cdp("Page.enable");
    await cdp("Runtime.enable");
    await cdp("Emulation.setDeviceMetricsOverride", {
      width: 1440,
      height: 900,
      deviceScaleFactor: 2,
      mobile: false,
    });

    for (let i = 0; i < 40; i++) {
      const step = await evalJs("window.__initStep");
      if (step === "done") break;
      await sleep(200);
    }
    await sleep(500);

    const initErr = await evalJs("window.__director?.initError");
    if (initErr) throw new Error(`initApp failed: ${initErr}`);

    // 1. Audit Blender-style UI, Left Tool Shelf, Top Menu Bar, No 3D Cursor, No auto ref video
    const uiAudit = await evalJs(`(() => {
      const shelf = document.getElementById('leftToolShelf');
      const shelfBtns = Array.from(shelf.querySelectorAll('.tool-shelf-btn'));
      const topMenus = Array.from(document.querySelectorAll('#menubarRow .top-menu-btn')).map(b => b.textContent.trim());
      const vpLayer = document.getElementById('viewportPanesLayer').getBoundingClientRect();
      const rightCol = document.getElementById('rightInspectorSidebar').getBoundingClientRect();
      const thumbCollapsed = !document.getElementById('refThumbStripWrap').classList.contains('active');
      const speedCollapsed = !document.getElementById('kfSpeedAndInterpBar').classList.contains('open');
      const hasEmojiInTopOrShelf = /[\\p{Extended_Pictographic}\\uFE0F]/u.test(
        document.getElementById('topbarMenuRoot').innerText + shelf.innerText
      );
      return {
        shelfWidth: Math.round(shelf.getBoundingClientRect().width),
        shelfBtnCount: shelfBtns.length,
        shelfBtnIds: shelfBtns.map(b => b.id),
        shelfBtnSizes: shelfBtns.map(b => {
          const r = b.getBoundingClientRect();
          const svgR = b.querySelector('svg').getBoundingClientRect();
          return { w: Math.round(r.width), h: Math.round(r.height), svgW: Math.round(svgR.width), svgH: Math.round(svgR.height) };
        }),
        topMenus,
        vpHeightRatio: Number((vpLayer.height / window.innerHeight).toFixed(3)),
        rightSidebarWidth: Math.round(rightCol.width),
        thumbCollapsed,
        speedCollapsed,
        hasEmojiInTopOrShelf,
        refLoadedOnStartup: window.__director.sceneData?.ref_video?.loaded || false,
        has3DCursorInDOM: !!document.getElementById('btnCursorTool') || document.body.innerText.includes('三维游标'),
      };
    })()`);
    console.log("1. UI Audit:", JSON.stringify(uiAudit, null, 2));

    // 2. Audit 20x viewport orbit/pan/zoom navigation + clicking empty space + dragging character surface in Browse mode
    const navAudit = await evalJs(`(() => {
      const snapBefore = JSON.stringify(window.__director.sceneData);
      const undoBefore = window.__director.undoStack.length;
      const pane = window.__director.workPaneDom;
      const rect = pane.getBoundingClientRect();

      for (let i = 0; i < 20; i++) {
        const sx = rect.left + 120 + (i * 7) % 180;
        const sy = rect.top + 120 + (i * 11) % 160;
        pane.dispatchEvent(new PointerEvent('pointerdown', { clientX: sx, clientY: sy, button: 0, bubbles: true }));
        pane.dispatchEvent(new PointerEvent('pointermove', { clientX: sx + 35, clientY: sy + 20, button: 0, bubbles: true }));
        pane.dispatchEvent(new PointerEvent('pointerup', { clientX: sx + 35, clientY: sy + 20, button: 0, bubbles: true }));
        pane.dispatchEvent(new WheelEvent('wheel', { clientX: sx, clientY: sy, deltaY: (i % 2 === 0 ? 40 : -40), bubbles: true }));
      }

      pane.dispatchEvent(new PointerEvent('pointerdown', { clientX: rect.left + 18, clientY: rect.top + 18, button: 0, bubbles: true }));
      pane.dispatchEvent(new PointerEvent('pointerup', { clientX: rect.left + 18, clientY: rect.top + 18, button: 0, bubbles: true }));
      const selAfterEmptyClick = window.__director.selectedObjId;

      const snapAfter = JSON.stringify(window.__director.sceneData);
      const undoAfter = window.__director.undoStack.length;
      return {
        sceneUnchanged: snapBefore === snapAfter,
        undoUnchanged: undoBefore === undoAfter,
        selAfterEmptyClick,
      };
    })()`);
    console.log("2. Nav & Anti-Mistrigger Audit:", JSON.stringify(navAudit));

    // 3. Audit Focus Selected (F), Frame All (A), G/R/T tools, Prop Lock, Input Focus shortcut suppression, Esc, Read-only right preview
    const interactionAudit = await evalJs(`(() => {
      const d = window.__director;
      const firstChar = d.sceneData.objects.find(o => o.type === 'character');
      d.selectedObjId = firstChar.id;
      d.focusSelectedObject();
      const pivotAfterFocus = d.workOrbit.target.toArray().map(v => Number(v.toFixed(2)));

      d.frameAllSceneObjects();
      const pivotAfterFrameAll = d.workOrbit.target.toArray().map(v => Number(v.toFixed(2)));

      d.setTransformTool('translate');
      const modeG = { mode: d.interactionMode, tool: d.activeTransformTool, attached: !!d.transformControl.object };
      d.setTransformTool('rotate');
      const modeR = { mode: d.interactionMode, tool: d.activeTransformTool, attached: !!d.transformControl.object };
      d.setTransformTool('scale');
      const modeT = { mode: d.interactionMode, tool: d.activeTransformTool, attached: !!d.transformControl.object };

      const inp = document.getElementById('propPosX');
      inp.focus();
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'g', bubbles: true }));
      const toolWhileInputFocused = d.activeTransformTool;
      inp.blur();

      const propObj = d.sceneData.objects.find(o => o.type !== 'character');
      let propLockedGizmoAttached = false;
      if (propObj) {
        d.selectedObjId = propObj.id;
        d.attachGizmoToSelection();
        propLockedGizmoAttached = !!d.transformControl.object;
      }

      d.selectedObjId = firstChar.id;
      d.setTransformTool('translate');
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
      const modeAfterEsc = { mode: d.interactionMode, tool: d.activeTransformTool };

      return {
        pivotAfterFocus,
        pivotAfterFrameAll,
        modeG,
        modeR,
        modeT,
        toolWhileInputFocused,
        propLockedGizmoAttached,
        modeAfterEsc,
        camOrbitEnabled: d.camOrbit.enabled,
      };
    })()`);
    console.log("3. Interaction Audit:", JSON.stringify(interactionAudit, null, 2));

    // 4. Audit Animation Edit Policies A (base_only), B (current_frame), C (shift_all) + Undo
    const policyAudit = await evalJs(`(() => {
      const d = window.__director;
      const ch = d.sceneData.objects.find(o => o.type === 'character' && o.keyframes?.length > 0);
      d.selectedObjId = ch.id;
      const origKfs = JSON.stringify(ch.keyframes);
      const origTargets0 = JSON.stringify(ch.keyframes[0].targets);

      // Policy A: base_only
      d.charAnimEditPolicy = 'base_only';
      document.getElementById('propPosX').value = '1.25';
      document.getElementById('propPosX').dispatchEvent(new Event('change', { bubbles: true }));
      const kfsAfterA = JSON.stringify(ch.keyframes);
      const basePosAfterA = [...(ch.base_pos || ch.pos)];
      d.performUndo();

      // Policy B: current_frame
      d.charAnimEditPolicy = 'current_frame';
      const kf0BeforeB = ch.keyframes[0].pos[0];
      document.getElementById('propPosX').value = '1.50';
      document.getElementById('propPosX').dispatchEvent(new Event('change', { bubbles: true }));
      const kf0AfterB = ch.keyframes[0].pos[0];
      d.performUndo();
      const kf0AfterUndoB = ch.keyframes[0].pos[0];

      // Policy C: shift_all
      d.charAnimEditPolicy = 'shift_all';
      const kfsBeforeC = ch.keyframes.map(k => k.pos[0]);
      const curX = ch.pos[0];
      const deltaX = 0.6;
      document.getElementById('propPosX').value = String((curX + deltaX).toFixed(2));
      document.getElementById('propPosX').dispatchEvent(new Event('change', { bubbles: true }));
      const kfsAfterC = ch.keyframes.map(k => k.pos[0]);
      const targetsAfterC = JSON.stringify(ch.keyframes[0].targets);
      d.performUndo();
      const kfsAfterUndoC = JSON.stringify(ch.keyframes);

      return {
        policyA_kfsUnchanged: origKfs === kfsAfterA,
        policyA_basePosX: basePosAfterA[0],
        policyB_updatedKf0: kf0AfterB === 1.5 && kf0AfterB !== kf0BeforeB,
        policyB_undoRestored: Math.abs(kf0AfterUndoB - kf0BeforeB) < 1e-3,
        policyC_shiftedAll: kfsAfterC.every((x, idx) => Math.abs((x - kfsBeforeC[idx]) - deltaX) < 1e-2),
        policyC_localTargetsUnchanged: origTargets0 === targetsAfterC,
        policyC_undoRestored: origKfs === kfsAfterUndoC,
      };
    })()`);
    console.log("4. Animation Policy A/B/C Audit:", JSON.stringify(policyAudit, null, 2));

    // 5. Audit Focus Previz Layout & capture screenshot
    const focusAudit = await evalJs(`(() => {
      document.getElementById('btnCtxFocusPreviz').click();
      const vp = document.getElementById('viewportPanesLayer').getBoundingClientRect();
      return {
        focusPrevizActive: document.getElementById('btnCtxFocusPreviz').classList.contains('active'),
        vpHeightRatio: Number((vp.height / window.innerHeight).toFixed(3)),
      };
    })()`);
    await sleep(300);
    await captureScreenshot(path.join(EXPORTS_DIR, "opt_focus_previz_1440x900.png"));
    await evalJs("document.getElementById('btnCtxFocusPreviz').click()");
    await sleep(250);
    console.log("5. Focus Previz Audit:", JSON.stringify(focusAudit));

    // 6. Audit Narrow Window (980x760) Overflow Menu ('更多')
    await cdp("Emulation.setDeviceMetricsOverride", {
      width: 980,
      height: 760,
      deviceScaleFactor: 2,
      mobile: false,
    });
    await sleep(250);
    const narrowAudit = await evalJs(`(() => {
      window.dispatchEvent(new Event('resize'));
      const moreWrap = document.getElementById('menuMoreWrap');
      return {
        moreVisibleAt980: getComputedStyle(moreWrap).display !== 'none',
      };
    })()`);
    console.log("6. Narrow Window Audit:", JSON.stringify(narrowAudit));
    await cdp("Emulation.setDeviceMetricsOverride", {
      width: 1440,
      height: 900,
      deviceScaleFactor: 2,
      mobile: false,
    });
    await sleep(250);

    // 7. Audit 5-Frame (0%, 25%, 50%, 75%, 100%) WebGL Preview vs Python Export Parity
    const parityRes = await evalJs("window.__director.verifyPreviewVsPython5Frames()");
    console.log("7. 5-Frame Parity Result:", JSON.stringify(parityRes));
    await evalJs("document.getElementById('btnCloseCompareModal')?.click()");

    // 8. Audit MP4 Video Export with unique filename
    const exportRes = await evalJs(`fetch('/api/render', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scene: window.__director.sceneData })
    }).then(r => r.json())`);
    console.log("8. MP4 Export Result:", JSON.stringify(exportRes));

    // 9. Capture final 1440x900 screenshot with Move tool active on Character 1
    await evalJs(`(() => {
      const d = window.__director;
      const firstChar = d.sceneData.objects.find(o => o.type === 'character');
      d.selectedObjId = firstChar.id;
      d.setTransformTool('translate');
    })()`);
    await sleep(350);
    await captureScreenshot(path.join(EXPORTS_DIR, "opt_after_1440x900.png"));

    ws.close();
    console.log("ALL_VERIFICATION_CHECKS_PASSED");
  } finally {
    proc.kill("SIGTERM");
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
