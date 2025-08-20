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
            # Helper script (avoids -c "..." so no commandline/filepath noise)
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

# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Head tracking with motor commands (split console + FPS tweaks)")
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

    args = ap.parse_args()

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

    motor = MotorConsole(enable_split=args.split_console)

    # Optional threaded capture
    reader = CamReader(cap) if args.threaded_capture else None

    print("Press Q to quit." if not args.no_preview else "Headless mode: press Ctrl+C to stop.")
    try:
        while True:
            if reader:
                ok, frame = reader.get()
                if not ok or frame is None:
                    time.sleep(0.001)
                    continue
            else:
                ok, frame = cap.read()
                if not ok:
                    ok, frame = cap.read()
                    if not ok:
                        print("Frame grab failed; continuing...")
                        continue

            if args.flip:
                frame = cv2.flip(frame, 1)

            now = time.time()
            dt = max(1e-3, now - prev_time)
            prev_time = now

            H, W = frame.shape[:2]
            cx, cy = W / 2.0, H / 2.0

            # Downscale for detection if requested
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

            result = detector.process(rgb)

            head_cx = head_cy = None

            if result and result.detections:
                det = max(result.detections, key=lambda d: d.score[0] if d.score else 0.0)
                rel = det.location_data.relative_bounding_box
                # Map relative box -> detection-space pixels
                dx = int(rel.xmin * det_W); dy = int(rel.ymin * det_H)
                dw = int(rel.width * det_W); dh = int(rel.height * det_H)
                # Pad in detection space
                pad = int(0.08 * max(dw, dh))
                dx -= pad; dy -= pad; dw += 2*pad; dh += 2*pad
                # Clamp in detection space
                dx = max(0, min(dx, det_W - 1))
                dy = max(0, min(dy, det_H - 1))
                dw = max(1, min(dw, det_W - dx))
                dh = max(1, min(dh, det_H - dy))
                # Scale to full-res frame space if needed
                if DS != 1.0:
                    x = int(dx / DS); y = int(dy / DS); w = int(dw / DS); h = int(dh / DS)
                else:
                    x, y, w, h = dx, dy, dw, dh

                x, y, w, h = clamp_box((x, y, w, h), W, H)
                box_now = (x, y, w, h)
                prev_box = ema_smooth(prev_box, box_now, alpha=args.smooth)
                lost_counter = 0

                # draw detection (only if preview)
                if not args.no_preview:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 220, 40), 2)
                    if prev_box is not None:
                        px, py, pw, ph = map(int, prev_box)
                        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 180, 0), 1)
                        head_cx, head_cy = px + pw / 2.0, py + ph / 2.0
                        cv2.circle(frame, (int(head_cx), int(head_cy)), 3, (255, 180, 0), -1)
                else:
                    px, py, pw, ph = map(int, prev_box)
                    head_cx, head_cy = px + pw / 2.0, py + ph / 2.0
            else:
                lost_counter += 1
                if prev_box is not None and lost_counter <= LOST_DECAY:
                    px, py, pw, ph = map(int, prev_box)
                    if not args.no_preview:
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

                # Use persistent PD controllers
                pan_delta = pan_pd.step(pan_err, dt)
                tilt_delta = tilt_pd.step(tilt_err, dt)

                max_step = args.rate_limit * dt
                pan_delta = float(np.clip(pan_delta, -max_step, max_step))
                tilt_delta = float(np.clip(tilt_delta, -max_step, max_step))

                pan_angle = float(np.clip(pan_angle + pan_delta, args.min_pan, args.max_pan))
                tilt_angle = float(np.clip(tilt_angle + tilt_delta, args.min_tilt, args.max_tilt))

                # crosshair & error vector (only if preview)
                if not args.no_preview:
                    cv2.drawMarker(frame, (int(cx), int(cy)), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
                    cv2.line(frame, (int(cx), int(cy)), (int(head_cx), int(head_cy)), (0, 180, 255), 1)

                # print to motor console at limited rate
                if (now - last_print) >= print_interval:
                    last_print = now
                    if args.pan_only:
                        motor.write(f"PAN {int(round(pan_angle))}\n")
                    else:
                        motor.write(f"PAN {int(round(pan_angle))}  TILT {int(round(tilt_angle))}\n")

            if args.show_fps and not args.no_preview:
                fps = 1.0 / dt if dt > 0 else 0.0
                cv2.putText(frame, f"{fps:4.1f} FPS", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Preview / headless
            if not args.no_preview:
                cv2.imshow("Head Tracking + Motor Commands", frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                    break
            else:
                # small sleep to avoid busy-wait in headless mode
                time.sleep(0.001)

    finally:
        motor.close()
        if reader:
            reader.stop()
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
