# Head Tracking with Motor Commands (OBS / PS3 Eye)

This project uses [MediaPipe Face Detection](https://developers.google.com/mediapipe) and OpenCV to detect a person’s head from a webcam (OBS VirtualCam, PS3 Eye, or other camera).  
It outputs continuous **pan/tilt motor commands** to follow the detected head, simulating the “Eye of Sauron” style movement.  

By default, motor commands are printed to the console. With `--split-console`, a **second window** is opened that only shows motor commands.

---

## Requirements
- Python 3.8+
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)

---

## Running

Example (OBS Virtual Camera, index 6):

```bash
python head_track_split_console.py --api dshow --cam 6 --show-fps --model 1 --smooth 0.6 --split-console
```

Press **Q** in the preview window to quit.

---

## Command-Line Options

| Flag              | Default | Description |
|-------------------|---------|-------------|
| `--cam`           | `0`     | Camera index (e.g., `6` for OBS Virtual Camera). |
| `--api`           | `dshow` | Backend for capture: `any`, `dshow`, `msmf`, `ffmpeg`. |
| `--flip`          | off     | Mirror the preview (like a webcam selfie view). |
| `--show-fps`      | off     | Show FPS overlay on video. |
| `--smooth`        | `0.60`  | Exponential smoothing factor (0–1) for head box stability. |
| `--min-conf`      | `0.5`   | Minimum confidence for MediaPipe face detection. |
| `--model`         | `1`     | MediaPipe model: `0` = short-range, `1` = full-range. |
| `--pan-only`      | off     | Only output pan (no tilt). |
| `--center-pan`    | `90.0`  | Starting pan angle (degrees). |
| `--center-tilt`   | `90.0`  | Starting tilt angle (degrees). |
| `--min-pan`       | `20.0`  | Minimum pan angle. |
| `--max-pan`       | `160.0` | Maximum pan angle. |
| `--min-tilt`      | `30.0`  | Minimum tilt angle. |
| `--max-tilt`      | `150.0` | Maximum tilt angle. |
| `--pan-kp`        | `0.08`  | Pan proportional gain. |
| `--pan-kd`        | `0.18`  | Pan derivative gain. |
| `--tilt-kp`       | `0.10`  | Tilt proportional gain. |
| `--tilt-kd`       | `0.22`  | Tilt derivative gain. |
| `--deadband-x`    | `12.0`  | Deadband (pixels) in X (ignore tiny errors). |
| `--deadband-y`    | `12.0`  | Deadband (pixels) in Y. |
| `--rate-limit`    | `180.0` | Max motor speed in degrees/sec. |
| `--print-rate`    | `25.0`  | How often motor commands print (per second). |
| `--split-console` | off     | Open second console for motor output only. |

---

## Notes
- Works with **OBS Virtual Camera** (`--cam 6` on this system).  
- Works with **PS3 Eye** at high frame rates (try `--api dshow --cam 0`).  
- Motor commands are currently text only (`PAN <deg>  TILT <deg>`). Replace with [pyserial](https://pythonhosted.org/pyserial/) writes when connecting real hardware.

---
