import argparse
import time
import sys
import subprocess
import cv2
import numpy as np
import mediapipe as mp
import os
import warnings
import platform
import threading
import queue
import tempfile
import textwrap
import shutil
from typing import Optional
import socket
import logging

# ------------------------ quiet optional logs (optional) ------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="google.protobuf.symbol_database",
    message=".*SymbolDatabase.GetPrototype\\(\\) is deprecated.*",
)
# -------------------------------------------------------------------------------

mp_fd = mp.solutions.face_detection

API_MAP = {
    "any":    cv2.CAP_ANY,
    "dshow":  cv2.CAP_DSHOW,
    "msmf":   cv2.CAP_MSMF,
    "ffmpeg": cv2.CAP_FFMPEG,
}

FOURCC_MAP = {
    "any":  None,
    "mjpg": "MJPG",
    "yuy2": "YUY2",
    "h264": "H264",
}

def open_cam(index, api_hint):
    cap = cv2.VideoCapture(index, api_hint)
    if cap.isOpened():
        return cap
    for name, api in API_MAP.items():
        if api == api_hint:
            continue
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            print(f"Opened camera index {index} with API {name}")
            return cap
    for idx in range(0, 10):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Opened camera index {idx} with API dshow (fallback)")
            return cap
    return None

def ema_smooth(prev_box, new_box, alpha=0.75):
    if prev_box is None:
        return new_box
    px, py, pw, ph = prev_box
    nx, ny, nw, nh = new_box
    sx = alpha * px + (1 - alpha) * nx
    sy = alpha * py + (1 - alpha) * ny
    sw = alpha * pw + (1 - alpha) * nw
    sh = alpha * ph + (1 - alpha) * nh
    return (sx, sy, sw, sh)

def clamp_box(box, W, H):
    x, y, w, h = box
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return (x, y, w, h)

def map_range(x, in_lo, in_hi, out_lo, out_hi):
    if in_hi == in_lo:
        return out_lo
    ratio = (x - in_lo) / (in_hi - in_lo)
    return out_lo + ratio * (out_hi - out_lo)

class PD1D:
    def __init__(self, kp=0.08, kd=0.18):
        self.kp = kp
        self.kd = kd
        self.prev_err = None
    def step(self, err, dt):
        d = 0.0 if self.prev_err is None or dt <= 0 else (err - self.prev_err) / dt
        self.prev_err = err
        return self.kp * err + self.kd * d

# ------------------- MotorConsole (clean new window, no file paths) -------------
class MotorConsole:
    """
    Opens a separate Windows console window to display motor output.
    If unavailable (non-Windows or failure), falls back to printing in the current console.
    """
    def __init__(self, enable_split: bool = False, queue_size: int = 256):
        self.enable_requested = enable_split
        self.proc: Optional[subprocess.Popen] = None
        self.stdin = None
        self.q: Optional[queue.Queue] = None
        self.writer_thread: Optional[threading.Thread] = None
        self.alive = False
        self.temp_dir: Optional[str] = None
        self.helper_path: Optional[str] = None
        self.queue_size = max(1, int(queue_size))

        if not self.enable_requested:
            return
        if platform.system() != "Windows":
            print("Split console only supported on Windows; using main console.")
            return

        try:
            self.temp_dir = tempfile.mkdtemp(prefix="motorconsole_")
            self.helper_path = os.path.join(self.temp_dir, "motor_console.py")
            helper_code = textwrap.dedent(
                """
                import sys
                try:
                    print("Motor console ready. Press Ctrl+C here to close.", flush=True)
                    for line in sys.stdin:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                except KeyboardInterrupt:
                    pass
                """
            ).strip() + "\n"
            with open(self.helper_path, "w", encoding="utf-8") as f:
                f.write(helper_code)

            python_cmd = self._resolve_console_python()
            CREATE_NEW_CONSOLE = subprocess.CREATE_NEW_CONSOLE
            self.proc = subprocess.Popen(
                [python_cmd, "-u", self.helper_path],
                stdin=subprocess.PIPE,
                creationflags=CREATE_NEW_CONSOLE,
                text=True,
                bufsize=1,
                cwd=self.temp_dir,
            )
            if not self.proc or not self.proc.stdin:
                raise RuntimeError("Child process failed to expose stdin")

            self.stdin = self.proc.stdin
            self.q = queue.Queue(maxsize=self.queue_size)
            self.alive = True

            def _writer():
                while self.alive:
                    if self.proc.poll() is not None:
                        self.alive = False
                        print("[motor] split console exited; using main console.")
                        return
                    try:
                        msg = self.q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    try:
                        self.stdin.write(msg)
                        self.stdin.flush()
                    except Exception as e:
                        self.alive = False
                        print(f"[motor] split console write failed ({e}); using main console.")
                        return

            self.writer_thread = threading.Thread(target=_writer, daemon=True)
            self.writer_thread.start()
            self.write("[motor] split console: OK\n")

        except Exception as e:
            print(f"Could not open split console ({e}). Using main console.")
            self._cleanup_child(handles_only=True)

    def _resolve_console_python(self) -> str:
        exe = sys.executable or "python"
        base = os.path.basename(exe).lower()
        if base == "pythonw.exe":
            py_launcher = shutil.which("py")
            if py_launcher:
                return f"{py_launcher} -3"
            candidate = os.path.join(os.path.dirname(exe), "python.exe")
            if os.path.exists(candidate):
                return candidate
            return exe
        return exe

    def write(self, text: str):
        if self.proc and self.alive and self.q:
            try:
                if self.q.full():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                self.q.put_nowait(text)
                return
            except Exception:
                pass
        print(text, end="")

    def close(self):
        self.alive = False
        if self.q:
            try:
                while not self.q.empty():
                    _ = self.q.get_nowait()
            except Exception:
                pass
        self._cleanup_child(handles_only=False)

    def _cleanup_child(self, handles_only: bool):
        try:
            if self.stdin:
                try:
                    self.stdin.close()
                except Exception:
                    pass
            if self.proc:
                for _ in range(5):
                    if self.proc.poll() is not None:
                        break
                    time.sleep(0.1)
                if self.proc.poll() is None:
                    try:
                        self.proc.terminate()
                    except Exception:
                        pass
        finally:
            if not handles_only:
                try:
                    if self.helper_path and os.path.exists(self.helper_path):
                        try:
                            os.remove(self.helper_path)
                        except Exception:
                            pass
                    if self.temp_dir and os.path.isdir(self.temp_dir):
                        try:
                            shutil.rmtree(self.temp_dir, ignore_errors=True)
                        except Exception:
                            pass
                except Exception:
                    pass

# ------------------- Optional threaded camera reader ----------------------------
class CamReader:
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.stopped = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stopped:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.ok, self.frame = True, f

    def get(self):
        with self.lock:
            return self.ok, None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        try:
            self.thread.join(timeout=0.5)
        except Exception:
            pass

class MJPEGStream:
    """Minimal thread-safe MJPEG broadcaster. Call publish(jpg_bytes) from main loop."""
    def __init__(self, fps=15):
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.frame = None
        self.min_interval = 1.0 / float(max(1, fps))
        self._last = 0.0

    def publish(self, jpg_bytes: bytes):
        now = time.time()
        with self.lock:
            self.frame = jpg_bytes
            if (now - self._last) >= self.min_interval:
                self._last = now
                self.cond.notify_all()

    def generator(self):
        boundary = b"--frame\r\n"
        header = b"Content-Type: image/jpeg\r\n\r\n"
        while True:
            with self.lock:
                if self.frame is None:
                    self.cond.wait(timeout=1.0)
                else:
                    self.cond.wait(timeout=1.0)
                data = self.frame
            if data:
                yield boundary + header + data + b"\r\n"

# ------------------- Web Remote (phone) -----------------------------------------
class WebRemote:
    """
    Lightweight Flask server in a background thread.
    Exposes manual control & scan for pan/tilt + optional camera MJPEG.
    """
    def __init__(self, port: int, has_tilt: bool):
        from flask import Flask, request, jsonify, Response
        self.Flask = Flask
        self.request = request
        self.jsonify = jsonify
        self.Response = Response

        self.port = port
        self.has_tilt = has_tilt

        # shared state
        self.lock = threading.Lock()
        self.manual = True           # always start in Manual
        self.scan = False
        self.scan_speed = 25.0       # deg/s
        self.scan_range = (-80.0, 80.0)
        self.scan_dir = 1.0
        self.t_pan = 0.0
        self.t_tilt = 10.0
        self.exit_requested = False
        self.streamer = MJPEGStream(fps=15)  # 10–20 fps is plenty for phone preview

        # figure out LAN IP once & print ONCE
        self.server_addr = f"http://{self._get_local_ip()}:{self.port}"

        self._build_app()

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _get_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def _build_app(self):
        app = self.Flask(__name__)

        # Silence Flask/Werkzeug per-request logs
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        INDEX_HTML = """
        <!doctype html>
        <html>
        <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
        <title>Eye Remote</title>
        <style>
          body { font-family: system-ui, sans-serif; margin: 16px; }
          .card { max-width: 520px; margin: 0 auto; padding: 16px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.15); }
          h1 { font-size: 1.25rem; margin: 0 0 12px; }
          .row { display:flex; gap:12px; align-items:center; margin: 12px 0; flex-wrap: wrap; }
          button { padding: 12px 16px; border-radius: 12px; border: 0; background:#222; color:#fff; font-weight:600; }
          button.secondary { background:#555; }
          button.warn { background:#a41212; }
          input[type=range] { width: 100%; }
          label { font-weight:600; }
          .small { font-size: 0.9rem; color:#555; }
          .toggle { padding: 8px 12px; border-radius: 10px; background:#004aad; color:#fff; }
        </style>
        </head>
        <body>
        <div class="card">
          <h1>Eye of Sauron — Remote</h1>

          <div class="row">
            <span>Mode:</span>
            <button id="modeBtn" class="toggle">Manual</button>
          </div>

          <div class="row">
            <label>Pan (°): <span id="panVal">0</span></label>
            <input id="pan" type="range" min="-90" max="90" step="1" value="0">
          </div>

          <div id="tiltBlock" style="display:none;">
            <div class="row">
              <label>Tilt (°): <span id="tiltVal">10</span></label>
              <input id="tilt" type="range" min="-5" max="35" step="1" value="10">
            </div>
          </div>

          <div class="row">
            <button id="centerBtn" class="secondary">Center</button>
            <button id="scanBtn" class="secondary">Start Scan</button>
            <button id="stopBtn" class="secondary">Stop</button>
            <button id="killBtn" class="warn">Kill</button>
          </div>

          <div class="row">
            <button id="camToggle" class="secondary">Show Camera</button>
        </div>
        <div class="row" id="camRow" style="display:none;">
            <img id="camImg" src="/stream" style="width:100%;border-radius:12px;box-shadow:0 4px 16px rgba(0,0,0,0.15);" />
        </div>
        <div class="row small">
            <div>Tip: Add to Home Screen for full‑screen remote.</div>
        </div>

        </div>

        <script>
        const tiltEnabled = {{TILT}};
        const pan = document.getElementById('pan');
        const panVal = document.getElementById('panVal');
        const tiltBlock = document.getElementById('tiltBlock');
        const tilt = document.getElementById('tilt');
        const tiltVal = document.getElementById('tiltVal');
        const scanBtn = document.getElementById('scanBtn');
        const modeBtn = document.getElementById('modeBtn');

        if (tiltEnabled) tiltBlock.style.display = 'block';

        function post(url, data) {
          return fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)}).then(r=>r.json());
        }
        function get(url) { return fetch(url).then(r=>r.json()); }

        pan.addEventListener('input', () => {
          panVal.textContent = pan.value;
          post('/api/set', { pan: Number(pan.value) });
        });
        if (tiltEnabled) {
          tilt.addEventListener('input', () => {
            tiltVal.textContent = tilt.value;
            post('/api/set', { tilt: Number(tilt.value) });
          });
        }

        document.getElementById('centerBtn').addEventListener('click', () => {
          pan.value = 0; panVal.textContent = '0';
          if (tiltEnabled) { tilt.value = 10; tiltVal.textContent = '10'; }
          post('/api/center', {});
        });

        let scanning = false;
        scanBtn.addEventListener('click', () => {
          scanning = !scanning;
          scanBtn.textContent = scanning ? 'Stop Scan' : 'Start Scan';
          post('/api/scan', { enabled: scanning });
        });

        document.getElementById('stopBtn').addEventListener('click', () => {
          scanning = false; scanBtn.textContent = 'Start Scan';
          post('/api/stop', {});
        });

        modeBtn.addEventListener('click', async () => {
          let s = await post('/api/mode/toggle', {});
          modeBtn.textContent = s.manual ? 'Manual' : 'Auto';
        });

        document.getElementById('killBtn').addEventListener('click', async () => {
          try {
            await post('/api/exit', {});
            document.getElementById('killBtn').textContent = 'Killing...';
            document.getElementById('killBtn').disabled = true;
          } catch (e) {}
        });

        // Pull initial status
        get('/api/status').then(s=>{
          pan.value = s.t_pan; panVal.textContent = s.t_pan;
          if (tiltEnabled) { tilt.value = s.t_tilt; tiltVal.textContent = s.t_tilt; }
          modeBtn.textContent = s.manual ? 'Manual' : 'Auto';
          scanning = s.scan; scanBtn.textContent = scanning ? 'Stop Scan' : 'Start Scan';
        });

        const camToggle = document.getElementById('camToggle');
        const camRow = document.getElementById('camRow');
        let camVisible = false;
        camToggle.addEventListener('click', () => {
            camVisible = !camVisible;
            camRow.style.display = camVisible ? 'block' : 'none';
            camToggle.textContent = camVisible ? 'Hide Camera' : 'Show Camera';
        });

        </script>
        </body>
        </html>
        """

        @app.route("/")
        def index():
            html = INDEX_HTML.replace("{{TILT}}", "true" if self.has_tilt else "false")
            return self.Response(html, mimetype="text/html")

        @app.get("/api/status")
        def api_status():
            with self.lock:
                return self.jsonify(
                    ok=True, manual=self.manual, scan=self.scan,
                    t_pan=self.t_pan, t_tilt=self.t_tilt,
                    exit=self.exit_requested
                )

        @app.post("/api/mode/toggle")
        def api_toggle_mode():
            with self.lock:
                self.manual = not self.manual
            return self.jsonify(ok=True, manual=self.manual)

        @app.post("/api/set")
        def api_set():
            data = self.request.get_json(force=True, silent=True) or {}
            with self.lock:
                if "pan" in data:
                    self.t_pan = float(data["pan"])
                if self.has_tilt and "tilt" in data:
                    self.t_tilt = float(data["tilt"])
            return self.jsonify(ok=True)

        @app.post("/api/center")
        def api_center():
            with self.lock:
                self.t_pan = 0.0
                if self.has_tilt:
                    self.t_tilt = 10.0
            return self.jsonify(ok=True)

        @app.post("/api/scan")
        def api_scan():
            data = self.request.get_json(force=True, silent=True) or {}
            enabled = bool(data.get("enabled", True))
            with self.lock:
                self.scan = enabled
            return self.jsonify(ok=True, enabled=self.scan)

        @app.post("/api/stop")
        def api_stop():
            with self.lock:
                self.scan = False
            return self.jsonify(ok=True)

        @app.post("/api/exit")
        def api_exit():
            with self.lock:
                self.exit_requested = True
            def _hard_kill():
                time.sleep(1.0)
                os._exit(0)
            threading.Thread(target=_hard_kill, daemon=True).start()
            return self.jsonify(ok=True)
        
        @app.get("/stream")
        def stream():
            return self.Response(self.streamer.generator(),
                                mimetype="multipart/x-mixed-replace; boundary=frame")

        self.app = app

    def _run(self):
        self.app.run(host="0.0.0.0", port=self.port, debug=False, threaded=True)

    def get_state(self):
        with self.lock:
            return dict(
                manual=self.manual, scan=self.scan,
                scan_speed=self.scan_speed, scan_range=self.scan_range,
                scan_dir=self.scan_dir, t_pan=self.t_pan, t_tilt=self.t_tilt,
                exit=self.exit_requested
            )

    def set_state(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)

# ----------------------- Xbox Controller wrapper --------------------------------
class XboxController:
    """
    Lightweight pygame-based reader for an Xbox (XInput) controller.
    Maps left stick: X -> pan, Y -> tilt.
    """
    def __init__(self, index=0, deadzone=0.15, expo=0.35, invert_x=False, invert_y=True):
        # import pygame lazily so the dependency is only required when used
        try:
            import pygame
        except ImportError as e:
            raise RuntimeError(
                "pygame is required for --xbox-control. Install with: pip install pygame"
            ) from e
        self.pygame = pygame
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() <= index:
            raise RuntimeError(f"No joystick at index {index}. Connected count: {pygame.joystick.get_count()}")

        self.js = pygame.joystick.Joystick(index)
        self.js.init()

        self.deadzone = float(deadzone)
        self.expo = float(expo)
        self.invert_x = bool(invert_x)
        self.invert_y = bool(invert_y)

        name = self.js.get_name()
        print(f"[xbox] Using controller: {name} (index {index})")

    def _shape(self, v):
        # deadzone
        if abs(v) < self.deadzone:
            return 0.0
        # re-scale post-deadzone to 0..1
        sign = 1.0 if v >= 0 else -1.0
        mag = (abs(v) - self.deadzone) / (1.0 - self.deadzone)
        mag = np.clip(mag, 0.0, 1.0)
        # expo curve for finer control near center
        shaped = sign * ( (1 - self.expo) * mag + self.expo * (mag ** 3) )
        return float(np.clip(shaped, -1.0, 1.0))

    def read_axes(self):
        # pump events to get fresh state
        self.pygame.event.pump()
        lx = self.js.get_axis(0)  # left stick X
        ly = self.js.get_axis(1)  # left stick Y
        if self.invert_x:
            lx = -lx
        if self.invert_y:
            ly = -ly
        return self._shape(lx), self._shape(ly)

# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Head tracking with motor commands + phone/web remote + optional Xbox control")
    ap.add_argument("--cam", type=int, default=0, help="Camera index")
    ap.add_argument("--api", type=str, default="dshow", choices=list(API_MAP.keys()))
    ap.add_argument("--flip", action="store_true", help="Mirror the preview")
    ap.add_argument("--show-fps", action="store_true")
    ap.add_argument("--smooth", type=float, default=0.60, help="EMA alpha (0..1)")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--model", type=int, default=1, choices=[0,1], help="MediaPipe model selection")

    # Motor PD & limits
    ap.add_argument("--pan-only", action="store_true")
    ap.add_argument("--center-pan", type=float, default=90.0)
    ap.add_argument("--center-tilt", type=float, default=90.0)
    ap.add_argument("--min-pan", type=float, default=20.0)
    ap.add_argument("--max-pan", type=float, default=160.0)
    ap.add_argument("--min-tilt", type=float, default=30.0)
    ap.add_argument("--max-tilt", type=float, default=150.0)
    ap.add_argument("--pan-kp", type=float, default=0.08)
    ap.add_argument("--pan-kd", type=float, default=0.18)
    ap.add_argument("--tilt-kp", type=float, default=0.10)
    ap.add_argument("--tilt-kd", type=float, default=0.22)
    ap.add_argument("--deadband-x", type=float, default=12.0)
    ap.add_argument("--deadband-y", type=float, default=12.0)
    ap.add_argument("--rate-limit", type=float, default=180.0, help="deg/sec per axis")
    ap.add_argument("--print-rate", type=float, default=25.0, help="updates per second")

    # split console
    ap.add_argument("--split-console", action="store_true", help="Open second console for motor output")

    # camera/FPS flags
    ap.add_argument("--fps", type=int, default=60, help="Request camera FPS")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--buffersize", type=int, default=1, help="CAP_PROP_BUFFERSIZE (lower = fresher frames)")
    ap.add_argument("--fourcc", type=str, default="mjpg", choices=list(FOURCC_MAP.keys()), help="Request camera pixel format")

    # threaded capture + detection downscale
    ap.add_argument("--threaded-capture", action="store_true", help="Read frames on a background thread")
    ap.add_argument("--detect-scale", type=float, default=1.0, help="Scale factor for detection (e.g., 0.5)")

    # NEW: no preview window
    ap.add_argument("--no-preview", action="store_true", help="Disable camera preview window")

    # NEW: phone/web remote
    ap.add_argument("--web-remote", action="store_true", help="Enable phone/web remote")
    ap.add_argument("--web-port", type=int, default=8080, help="Port for web remote (default 8080)")

    # NEW: allow running without opening a camera device
    ap.add_argument("--no-camera", action="store_true", help="Do not open any camera (headless or Xbox-only)")

    # NEW: Xbox controller options
    ap.add_argument("--xbox-control", action="store_true", help="Use Xbox controller (USB) to drive pan/tilt")
    ap.add_argument("--xbox-index", type=int, default=0, help="Joystick index (default 0)")
    ap.add_argument("--xbox-deadzone", type=float, default=0.15, help="Controller stick deadzone (0..1)")
    ap.add_argument("--xbox-expo", type=float, default=0.35, help="Expo curve (0=linear, higher=more fine center control)")
    ap.add_argument("--xbox-invert-x", action="store_true", help="Invert pan axis")
    ap.add_argument("--xbox-invert-y", action="store_true", help="Invert tilt axis (default behavior flips Y already)")

    args = ap.parse_args()

    # Optional Xbox controller
    xbox = None
    if args.xbox_control:
        try:
            xbox = XboxController(
                index=args.xbox_index,
                deadzone=args.xbox_deadzone,
                expo=args.xbox_expo,
                invert_x=args.xbox_invert_x,
                invert_y=not args.xbox_invert_y  # our wrapper default is invert_y=True; this flag flips that behavior
            )
        except Exception as e:
            print(f"[xbox] Failed to init controller: {e}")
            raise SystemExit(1)

    # Camera init (optional)
    cap = None
    if not args.no_camera:
        cap = open_cam(args.cam, API_MAP[args.api])
        if not cap:
            raise SystemExit("Could not open camera. If using OBS, Start Virtual Camera; if PS3 Eye, try --api dshow and your index.")

        # Request fast camera path
        if args.fourcc != "any":
            fourcc = cv2.VideoWriter_fourcc(*FOURCC_MAP[args.fourcc])
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, args.buffersize))

    detector = None
    if not args.no_camera:
        detector = mp_fd.FaceDetection(model_selection=args.model,
                                       min_detection_confidence=args.min_conf)

    prev_box = None
    prev_time = time.time()
    lost_counter = 0
    LOST_DECAY = 5

    pan_pd = PD1D(kp=args.pan_kp, kd=args.pan_kd)
    tilt_pd = PD1D(kp=args.tilt_kp, kd=args.tilt_kd)
    pan_angle = float(args.center_pan)
    tilt_angle = float(args.center_tilt)

    last_print = 0.0
    print_interval = 1.0 / max(1.0, args.print_rate)

    # only print when values change (rounded integer)
    last_printed_pan = None
    last_printed_tilt = None

    motor = MotorConsole(enable_split=args.split_console)

    # Optional threaded capture
    reader = CamReader(cap) if (cap is not None and args.threaded_capture) else None

    # Web remote (optional)
    remote = None
    if args.web_remote:
        has_tilt = not args.pan_only
        try:
            remote = WebRemote(port=args.web_port, has_tilt=has_tilt)
            if remote and hasattr(remote, "server_addr"):
                motor.write(f"[web] {remote.server_addr}\n")
        except Exception as e:
            print(f"[web] Failed to start remote: {e}")

    # Help text
    if args.no_preview:
        print("Headless mode: press Ctrl+C to stop.")
    else:
        print("Press Q to quit.")

    try:
        while True:
            # timing
            now = time.time()
            dt = max(1e-3, now - prev_time)
            prev_time = now

            # Acquire frame if camera present
            frame = None
            if reader:
                ok, frame = reader.get()
                if not ok or frame is None:
                    time.sleep(0.001)
            elif cap is not None:
                ok, frame = cap.read()
                if not ok:
                    ok, frame = cap.read()
                    if not ok:
                        print("Frame grab failed; continuing...")
                        time.sleep(0.005)

            # ------------- Remote state (manual/scan) -----------------
            manual = False
            scan_enabled = False
            t_pan = 0.0
            t_tilt = 10.0
            if remote:
                s = remote.get_state()
                manual = s["manual"]
                scan_enabled = s["scan"]
                t_pan = float(s["t_pan"])
                t_tilt = float(s["t_tilt"])

                # exit check from phone
                if s.get("exit"):
                    break

                # scan logic: bounce target within range
                if scan_enabled and manual:
                    lo, hi = s["scan_range"]
                    spd = s["scan_speed"]
                    t_pan += s["scan_dir"] * spd * dt
                    if t_pan > hi:
                        t_pan = hi
                        remote.set_state(scan_dir=-1.0)
                    if t_pan < lo:
                        t_pan = lo
                        remote.set_state(scan_dir=+1.0)
                    remote.set_state(t_pan=t_pan)

            # ------------- Xbox overrides targets (Manual) -------------
            # If xbox-control is active, we force manual mode and generate targets from sticks.
            if xbox is not None:
                manual = True
                lx, ly = xbox.read_axes()  # -1..+1 after shaping
                # Map to the same UI-friendly ranges as the web remote sliders:
                # pan slider is -90..+90, tilt slider is -5..+35
                t_pan = float(np.clip(lx * 90.0, -90.0, 90.0))
                if not args.pan_only:
                    t_tilt = float(np.clip(ly * 20.0 + 10.0, -5.0, 35.0))  # center around 10°, span ~40°

            # ----------------- Tracking / control ----------------------
            head_cx = head_cy = None

            if not manual:
                # AUTO: Face tracking if camera is available
                if frame is not None:
                    if args.flip:
                        frame = cv2.flip(frame, 1)

                    H, W = frame.shape[:2]
                    cx, cy = W / 2.0, H / 2.0

                    DS = float(args.detect_scale)
                    if DS <= 0.0 or DS > 1.0:
                        DS = 1.0

                    if DS != 1.0:
                        small = cv2.resize(frame, (0, 0), fx=DS, fy=DS, interpolation=cv2.INTER_LINEAR)
                        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                        det_W, det_H = small.shape[1], small.shape[0]
                    else:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        det_W, det_H = W, H

                    result = detector.process(rgb) if detector is not None else None

                    if result and result.detections:
                        det = max(result.detections, key=lambda d: d.score[0] if d.score else 0.0)
                        rel = det.location_data.relative_bounding_box
                        dx = int(rel.xmin * det_W); dy = int(rel.ymin * det_H)
                        dw = int(rel.width * det_W); dh = int(rel.height * det_H)
                        pad = int(0.08 * max(dw, dh))
                        dx -= pad; dy -= pad; dw += 2*pad; dh += 2*pad
                        dx = max(0, min(dx, det_W - 1))
                        dy = max(0, min(dy, det_H - 1))
                        dw = max(1, min(dw, det_W - dx))
                        dh = max(1, min(dh, det_H - dy))
                        if DS != 1.0:
                            x = int(dx / DS); y = int(dy / DS); w = int(dw / DS); h = int(dh / DS)
                        else:
                            x, y, w, h = dx, dy, dw, dh
                        x, y, w, h = clamp_box((x, y, w, h), W, H)
                        box_now = (x, y, w, h)
                        prev_box = ema_smooth(prev_box, box_now, alpha=args.smooth)
                        lost_counter = 0

                        if not args.no_preview:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 220, 40), 2)
                            if prev_box is not None:
                                px, py, pw, ph = map(int, prev_box)
                                cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 180, 0), 1)
                                head_cx, head_cy = px + pw / 2.0, py + ph / 2.0
                                cv2.circle(frame, (int(head_cx), int(head_cy)), 3, (255, 180, 0), -1)
                        else:
                            if prev_box is not None:
                                px, py, pw, ph = map(int, prev_box)
                                head_cx, head_cy = px + pw / 2.0, py + ph / 2.0
                    else:
                        lost_counter += 1
                        if prev_box is not None and lost_counter <= LOST_DECAY:
                            px, py, pw, ph = map(int, prev_box)
                            if not args.no_preview and frame is not None:
                                cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 200, 255), 1)
                            head_cx, head_cy = px + pw / 2.0, py + ph / 2.0
                        else:
                            prev_box = None

                    if head_cx is not None and head_cy is not None:
                        pan_err = (head_cx - cx)
                        tilt_err = (head_cy - cy)

                        if abs(pan_err) < args.deadband_x:
                            pan_err = 0.0
                        if abs(tilt_err) < args.deadband_y:
                            tilt_err = 0.0

                        pan_delta = pan_pd.step(pan_err, dt)
                        tilt_delta = tilt_pd.step(tilt_err, dt)

                        max_step = args.rate_limit * dt
                        pan_delta = float(np.clip(pan_delta, -max_step, max_step))
                        tilt_delta = float(np.clip(tilt_delta, -max_step, max_step))

                        pan_angle = float(np.clip(pan_angle + pan_delta, args.min_pan, args.max_pan))
                        tilt_angle = float(np.clip(tilt_angle + tilt_delta, args.min_tilt, args.max_tilt))

                if args.show_fps and not args.no_preview and frame is not None:
                    fps = 1.0 / dt if dt > 0 else 0.0
                    cv2.putText(frame, f"{fps:4.1f} FPS (AUTO)", (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            else:
                # MANUAL MODE:
                # If Xbox present, t_pan/t_tilt already set from sticks.
                # Otherwise, use web-remote sliders (if remote enabled).
                max_step = args.rate_limit * dt

                # Map pan: slider is -90..+90, servo expects min_pan..max_pan
                pan_target_servo = map_range(t_pan, -90.0, 90.0, args.min_pan, args.max_pan)
                pan_target_servo = float(np.clip(pan_target_servo, args.min_pan, args.max_pan))
                dpan = float(np.clip(pan_target_servo - pan_angle, -max_step, max_step))
                pan_angle = float(np.clip(pan_angle + dpan, args.min_pan, args.max_pan))

                if not args.pan_only:
                    # Tilt slider is -5..+35
                    tilt_target_servo = map_range(t_tilt, -5.0, 35.0, args.min_tilt, args.max_tilt)
                    tilt_target_servo = float(np.clip(tilt_target_servo, args.min_tilt, args.max_tilt))
                    dtilt = float(np.clip(tilt_target_servo - tilt_angle, -max_step, max_step))
                    tilt_angle = float(np.clip(tilt_angle + dtilt, args.min_tilt, args.max_tilt))

                if args.show_fps and not args.no_preview and frame is not None:
                    fps = 1.0 / dt if dt > 0 else 0.0
                    mode = "XBOX" if xbox is not None else "MANUAL"
                    cv2.putText(frame, f"{fps:4.1f} FPS ({mode})", (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # ------------- output motor commands (print only on change) -------------
            now2 = time.time()
            if (now2 - last_print) >= print_interval:
                last_print = now2
                pan_i = int(round(pan_angle))
                if args.pan_only:
                    if pan_i != last_printed_pan:
                        motor.write(f"PAN {pan_i}\n")
                        last_printed_pan = pan_i
                else:
                    tilt_i = int(round(tilt_angle))
                    if pan_i != last_printed_pan or tilt_i != last_printed_tilt:
                        motor.write(f"PAN {pan_i}  TILT {tilt_i}\n")
                        last_printed_pan = pan_i
                        last_printed_tilt = tilt_i

            # --- MJPEG publish to phone UI (only if we have frames) ---
            if remote and frame is not None:
                sf = frame
                h, w = sf.shape[:2]
                maxw = 640
                if w > maxw:
                    scale = maxw / float(w)
                    sf = cv2.resize(sf, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
                ok_jpg, jpg = cv2.imencode(".jpg", sf, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok_jpg:
                    remote.streamer.publish(jpg.tobytes())

            # Preview / headless
            if not args.no_preview and frame is not None:
                title = "Head Tracking + Motor Commands + Web Remote + Xbox"
                cv2.imshow(title, frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                    break
            else:
                time.sleep(0.001)

    finally:
        motor.close()
        if reader:
            reader.stop()
        if cap is not None:
            cap.release()
        if not args.no_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

if __name__ == "__main__":
    main()
