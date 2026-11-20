import base64
import hashlib
import json
import os
import socket
import struct
import subprocess
import time
import urllib.parse
import urllib.request

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
EXPORTS_DIR = "/Users/wangmu/.gemini/antigravity/scratch/director-stage/exports"
os.makedirs(EXPORTS_DIR, exist_ok=True)


class SimpleWSClient:
    def __init__(self, ws_url):
        parsed = urllib.parse.urlparse(ws_url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 9233
        self.path = parsed.path or "/"
        self.sock = socket.create_connection((self.host, self.port), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        req = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(req.encode("ascii"))
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError("WebSocket handshake failed")
            buf += chunk
        self._leftover = buf.split(b"\r\n\r\n", 1)[1]
        self._next_id = 1

    def _recv_exact(self, n):
        while len(self._leftover) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RuntimeError("WebSocket closed unexpectedly")
            self._leftover += chunk
        out = self._leftover[:n]
        self._leftover = self._leftover[n:]
        return out

    def send_text(self, text):
        data = text.encode("utf-8")
        mask_key = os.urandom(4)
        header = bytearray([0x81])
        length = len(data)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        header.extend(mask_key)
        masked = bytes(b ^ mask_key[i % 4] for i, b in enumerate(data))
        self.sock.sendall(bytes(header) + masked)

    def recv_text(self):
        msg_bytes = bytearray()
        while True:
            b1, b2 = self._recv_exact(2)
            fin = (b1 & 0x80) != 0
            opcode = b1 & 0x0F
            masked = (b2 & 0x80) != 0
            length = b2 & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._recv_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._recv_exact(8))[0]
            mask_key = self._recv_exact(4) if masked else None
            payload = self._recv_exact(length)
            if masked and mask_key:
                payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
            if opcode == 0x8:
                raise RuntimeError("WebSocket closed by server")
            if opcode == 0x9:
                # Ping -> Pong
                continue
            msg_bytes.extend(payload)
            if fin:
                return msg_bytes.decode("utf-8", errors="replace")

    def cdp(self, method, params=None):
        msg_id = self._next_id
        self._next_id += 1
        self.send_text(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        while True:
            raw = self.recv_text()
            msg = json.loads(raw)
            if msg.get("id") == msg_id:
                if "error" in msg:
                    raise RuntimeError(f"CDP error in {method}: {msg['error']}")
                return msg.get("result", {})

    def eval_js(self, expr):
        res = self.cdp(
            "Runtime.evaluate",
            {"expression": expr, "returnByValue": True, "awaitPromise": True},
        )
        if "exceptionDetails" in res:
            raise RuntimeError(f"JS Exception: {res['exceptionDetails']}")
        return res.get("result", {}).get("value")

    def screenshot(self, out_path):
        res = self.cdp("Page.captureScreenshot", {"format": "png"})
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(res["data"]))

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


def main():
    proc = subprocess.Popen(
        [
            CHROME_BIN,
            "--headless=new",
            "--remote-debugging-port=9233",
            "--window-size=1440,900",
            "--hide-scrollbars",
            "--no-first-run",
            "--no-default-browser-check",
            "--user-data-dir=/tmp/chrome_verify_9233",
            "http://127.0.0.1:8765/",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        ws_url = None
        for _ in range(35):
            time.sleep(0.25)
            try:
                with urllib.request.urlopen("http://127.0.0.1:9233/json") as r:
                    tabs = json.loads(r.read().decode())
                    for t in tabs:
                        if t.get("type") == "page" and "8765" in t.get("url", ""):
                            ws_url = t["webSocketDebuggerUrl"]
                            break
                if ws_url:
                    break
            except Exception:
                pass
        assert ws_url, "Failed to connect to headless Chrome CDP"

        ws = SimpleWSClient(ws_url)
        ws.cdp("Page.enable")
        ws.cdp("Runtime.enable")
        ws.cdp(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1440, "height": 900, "deviceScaleFactor": 2, "mobile": False},
        )

        for _ in range(40):
            step = ws.eval_js("window.__initStep")
            if step == "done":
                break
            time.sleep(0.2)
        time.sleep(0.5)

        init_err = ws.eval_js("window.__director?.initError")
        assert not init_err, f"initError: {init_err}"

        # 1. Audit Blender-style UI, Left Tool Shelf, Top Menu Bar, No 3D Cursor, No auto ref video
        ui_audit = ws.eval_js(
            """(() => {
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
            })()"""
        )
        print("1. UI Audit:", json.dumps(ui_audit, ensure_ascii=False, indent=2))
        assert 44 <= ui_audit["shelfWidth"] <= 48, f"Expected shelf width 44-48px, got {ui_audit['shelfWidth']}"
        assert ui_audit["shelfBtnCount"] == 6, f"Expected 6 shelf tools, got {ui_audit['shelfBtnCount']}"
        assert ui_audit["vpHeightRatio"] >= 0.60, f"Viewport height ratio {ui_audit['vpHeightRatio']} < 0.60"
        assert ui_audit["thumbCollapsed"] and ui_audit["speedCollapsed"], "Low-freq bars must default to collapsed"
        assert not ui_audit["hasEmojiInTopOrShelf"], "Found Emoji in topbar or tool shelf"
        assert not ui_audit["has3DCursorInDOM"], "3D Cursor UI still present"
        assert not ui_audit["refLoadedOnStartup"], "Reference video should NOT auto-load on startup"

        # 2. Audit 20x viewport orbit/pan/zoom navigation + clicking empty space + dragging character surface in Browse mode
        nav_audit = ws.eval_js(
            """(() => {
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
            })()"""
        )
        print("2. Nav & Anti-Mistrigger Audit:", json.dumps(nav_audit, ensure_ascii=False))
        assert nav_audit["sceneUnchanged"] and nav_audit["undoUnchanged"], "Navigation mutated sceneData or undoStack!"
        assert nav_audit["selAfterEmptyClick"] != "__domain__", "Empty space click selected __domain__!"

        # 3. Audit Focus Selected (F), Frame All (A), G/R/T tools, Prop Lock, Input Focus shortcut suppression, Esc, Read-only right preview
        interaction_audit = ws.eval_js(
            """(() => {
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
            })()"""
        )
        print("3. Interaction Audit:", json.dumps(interaction_audit, ensure_ascii=False, indent=2))
        assert interaction_audit["modeG"]["tool"] == "translate" and interaction_audit["modeG"]["attached"]
        assert interaction_audit["modeR"]["tool"] == "rotate" and interaction_audit["modeR"]["attached"]
        assert interaction_audit["modeT"]["tool"] == "scale" and interaction_audit["modeT"]["attached"]
        assert interaction_audit["toolWhileInputFocused"] == "scale"
        assert not interaction_audit["propLockedGizmoAttached"]
        assert interaction_audit["modeAfterEsc"]["mode"] == "browse" and interaction_audit["modeAfterEsc"]["tool"] == "none"
        assert not interaction_audit["camOrbitEnabled"]

        # 4. Audit Animation Edit Policies A (base_only), B (current_frame), C (shift_all) + Undo
        policy_audit = ws.eval_js(
            """(() => {
              const d = window.__director;
              const ch0 = d.sceneData.objects.find(o => o.type === 'character' && o.keyframes?.length > 0);
              const chId = ch0.id;
              const getCh = () => d.sceneData.objects.find(o => o.id === chId);
              d.selectedObjId = chId;
              const origKfs = JSON.stringify(getCh().keyframes);
              const origTargets0 = JSON.stringify(getCh().keyframes[0].targets);

              // Policy A: base_only
              d.charAnimEditPolicy = 'base_only';
              document.getElementById('propPosX').value = '1.25';
              document.getElementById('propPosX').dispatchEvent(new Event('change', { bubbles: true }));
              const kfsAfterA = JSON.stringify(getCh().keyframes);
              const basePosAfterA = [...(getCh().base_pos || getCh().pos)];
              d.performUndo();

              // Policy B: current_frame
              d.charAnimEditPolicy = 'current_frame';
              const kf0BeforeB = getCh().keyframes[0].pos[0];
              document.getElementById('propPosX').value = '1.50';
              document.getElementById('propPosX').dispatchEvent(new Event('change', { bubbles: true }));
              const kf0AfterB = getCh().keyframes[0].pos[0];
              d.performUndo();
              const kf0AfterUndoB = getCh().keyframes[0].pos[0];

              // Policy C: shift_all
              d.charAnimEditPolicy = 'shift_all';
              const kfsBeforeC = getCh().keyframes.map(k => k.pos[0]);
              const curX = getCh().pos[0];
              const deltaX = 0.6;
              document.getElementById('propPosX').value = String((curX + deltaX).toFixed(2));
              document.getElementById('propPosX').dispatchEvent(new Event('change', { bubbles: true }));
              const kfsAfterC = getCh().keyframes.map(k => k.pos[0]);
              const targetsAfterC = JSON.stringify(getCh().keyframes[0].targets);
              d.performUndo();
              const kfsAfterUndoC = JSON.stringify(getCh().keyframes);
              d.charAnimEditPolicy = 'current_frame';
              document.getElementById('ctxAnimPolicySelect').value = 'current_frame';
              document.getElementById('ctxAnimPolicySelect').dispatchEvent(new Event('change', { bubbles: true }));

              return {
                policyA_kfsUnchanged: origKfs === kfsAfterA,
                policyA_basePosX: basePosAfterA[0],
                policyB_updatedKf0: kf0AfterB === 1.5 && kf0AfterB !== kf0BeforeB,
                policyB_undoRestored: Math.abs(kf0AfterUndoB - kf0BeforeB) < 1e-3,
                policyC_shiftedAll: kfsAfterC.every((x, idx) => Math.abs((x - kfsBeforeC[idx]) - deltaX) < 1e-2),
                policyC_localTargetsUnchanged: origTargets0 === targetsAfterC,
                policyC_undoRestored: origKfs === kfsAfterUndoC,
              };
            })()"""
        )
        print("4. Animation Policy A/B/C Audit:", json.dumps(policy_audit, ensure_ascii=False, indent=2))
        assert policy_audit["policyA_kfsUnchanged"]
        assert policy_audit["policyB_updatedKf0"] and policy_audit["policyB_undoRestored"]
        assert policy_audit["policyC_shiftedAll"] and policy_audit["policyC_localTargetsUnchanged"] and policy_audit["policyC_undoRestored"]

        # 5. Audit Focus Previz Layout & capture screenshot
        focus_audit = ws.eval_js(
            """(() => {
              document.getElementById('btnFocusPreviz').click();
              const vp = document.getElementById('viewportPanesLayer').getBoundingClientRect();
              return {
                focusPrevizActive: document.getElementById('btnFocusPreviz').classList.contains('active'),
                vpHeightRatio: Number((vp.height / window.innerHeight).toFixed(3)),
              };
            })()"""
        )
        time.sleep(0.35)
        ws.screenshot(os.path.join(EXPORTS_DIR, "opt_focus_previz_1440x900.png"))
        ws.eval_js("document.getElementById('btnFocusPreviz').click()")
        time.sleep(0.25)
        print("5. Focus Previz Audit:", json.dumps(focus_audit, ensure_ascii=False))
        assert focus_audit["vpHeightRatio"] >= 0.85

        # 6. Audit Narrow Window (980x760) Overflow Menu ('更多')
        ws.cdp(
            "Emulation.setDeviceMetricsOverride",
            {"width": 980, "height": 760, "deviceScaleFactor": 2, "mobile": False},
        )
        time.sleep(0.25)
        narrow_audit = ws.eval_js(
            """(() => {
              window.dispatchEvent(new Event('resize'));
              const moreWrap = document.getElementById('menuMoreWrap');
              return {
                moreVisibleAt980: getComputedStyle(moreWrap).display !== 'none',
              };
            })()"""
        )
        print("6. Narrow Window Audit:", json.dumps(narrow_audit, ensure_ascii=False))
        ws.cdp(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1440, "height": 900, "deviceScaleFactor": 2, "mobile": False},
        )
        time.sleep(0.25)

        # 7. Audit 5-Frame (0%, 25%, 50%, 75%, 100%) WebGL Preview vs Python Export Parity
        parity_res = ws.eval_js("window.__director.verifyPreviewVsPython5Frames()")
        print("7. 5-Frame Parity Result:", json.dumps(parity_res, ensure_ascii=False))
        assert parity_res and parity_res.get("ok")
        ws.eval_js("document.getElementById('btnCloseCompareModal')?.click()")

        # 8. Audit MP4 Video Export with unique filename
        export_res = ws.eval_js(
            """fetch('/api/export_video', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ scene: window.__director.sceneData })
            }).then(r => r.json())"""
        )
        print("8. MP4 Export Result:", json.dumps(export_res, ensure_ascii=False))
        assert export_res and export_res.get("ok") and os.path.exists(export_res.get("output_path", ""))

        # 9. Capture final 1440x900 screenshot with Move tool active on Character 1
        ws.eval_js(
            """(() => {
              const d = window.__director;
              const firstChar = d.sceneData.objects.find(o => o.type === 'character');
              d.selectedObjId = firstChar.id;
              d.setTransformTool('translate');
            })()"""
        )
        time.sleep(0.35)
        ws.screenshot(os.path.join(EXPORTS_DIR, "opt_after_1440x900.png"))
        ws.close()
        print("ALL_VERIFICATION_CHECKS_PASSED")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
