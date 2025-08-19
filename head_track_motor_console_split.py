import argparse
import time
import sys
import subprocess
import cv2
import numpy as np
import mediapipe as mp

# ------------------------ quiet optional logs (optional) ------------------------
import os, warnings
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
    "any":   cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf":  cv2.CAP_MSMF,
    "ffmpeg": cv2.CAP_FFMPEG
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

# ------------------- NEW: secondary console helper -------------------
class MotorConsole:
    def __init__(self, enable_split=False):
        self.proc = None
        self.enable_split = enable_split
        if not enable_split:
            return
        # Spawn a fresh console window that echoes stdin
        CREATE_NEW_CONSOLE = 0x00000010
        code = (
            "import sys\n"
            "print('Motor console ready. Press Ctrl+C here to close.', flush=True)\n"
            "for line in sys.stdin:\n"
            "    sys.stdout.write(line)\n"
            "    sys.stdout.flush()\n"
        )
        cmd = [sys.executable, "-u", "-c", code]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                creationflags=CREATE_NEW_CONSOLE,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            print(f"Could not open split console ({e}). Falling back to main console.")
            self.proc = None

    def write(self, text: str):
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.write(text)
                self.proc.stdin.flush()
                return
            except Exception:
                pass
        # fallback: print to current console
        print(text, end="")

    def close(self):
        try:
            if self.proc and self.proc.stdin:
                self.proc.stdin.close()
            if self.proc:
                self.proc.terminate()
        except Exception:
            pass

# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Head tracking with motor commands (split console option)")
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
    # NEW:
    ap.add_argument("--split-console", action="store_true", help="Open second console for motor output")
    args = ap.parse_args()

    cap = open_cam(args.cam, API_MAP[args.api])
    if not cap:
        raise SystemExit("Could not open camera. If using OBS, Start Virtual Camera; if PS3 Eye, try --api dshow and your index.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

    print("Press Q to quit.")
    try:
        while True:
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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)

            head_cx = head_cy = None

            if result and result.detections:
                det = max(result.detections, key=lambda d: d.score[0] if d.score else 0.0)
                rel = det.location_data.relative_bounding_box
                x = int(rel.xmin * W)
                y = int(rel.ymin * H)
                w = int(rel.width * W)
                h = int(rel.height * H)
                pad = int(0.08 * max(w, h))
                x -= pad; y -= pad; w += 2*pad; h += 2*pad
                x, y, w, h = clamp_box((x, y, w, h), W, H)
                box_now = (x, y, w, h)
                prev_box = ema_smooth(prev_box, box_now, alpha=args.smooth)
                lost_counter = 0

                # draw detection
                cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 220, 40), 2)
                if prev_box is not None:
                    px, py, pw, ph = map(int, prev_box)
                    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 180, 0), 1)
                    head_cx, head_cy = px + pw / 2.0, py + ph / 2.0
                    cv2.circle(frame, (int(head_cx), int(head_cy)), 3, (255, 180, 0), -1)
            else:
                lost_counter += 1
                if prev_box is not None and lost_counter <= LOST_DECAY:
                    px, py, pw, ph = map(int, prev_box)
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

                # crosshair & error vector
                cv2.drawMarker(frame, (int(cx), int(cy)), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
                cv2.line(frame, (int(cx), int(cy)), (int(head_cx), int(head_cy)), (0, 180, 255), 1)

                # print to motor console at limited rate
                if (now - last_print) >= print_interval:
                    last_print = now
                    if args.pan_only:
                        motor.write(f"PAN {int(round(pan_angle))}\n")
                    else:
                        motor.write(f"PAN {int(round(pan_angle))}  TILT {int(round(tilt_angle))}\n")

            if args.show_fps:
                fps = 1.0 / dt if dt > 0 else 0.0
                cv2.putText(frame, f"{fps:4.1f} FPS", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Head Tracking + Motor Commands", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break
    finally:
        motor.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
