# Head Tracking + Motor Console + Web Remote + Xbox Control

This project uses [MediaPipe Face Detection](https://developers.google.com/mediapipe) and OpenCV to detect a person’s head from a webcam, then outputs **pan/tilt motor commands** to follow the detected head.  

It can also be controlled manually via:  
- **Xbox controller** (wired, USB)  
- **Phone/Web remote** (sliders, scan mode, and live MJPEG stream)  

By default, motor commands are printed in the console. With `--split-console`, a **second window** opens just for motor output.

---

## Features
- **Head Tracking (AUTO mode)**  
  - Tracks faces with MediaPipe.  
  - Outputs servo pan/tilt angles.  
  - Configurable PID gains, deadbands, and rate limits.  

- **Manual Modes**  
  - **Xbox Controller:** Left stick X → Pan, Y → Tilt.  
  - **Web Remote:** Phone-friendly HTML UI with pan/tilt sliders, “Center” button, and “Scan” sweep mode.  

- **Web Remote**  
  - Lightweight Flask server.  
  - Access via `http://<your-lan-ip>:8080`.  
  - Live MJPEG camera preview.  

- **Motor Console**  
  - Opens in a second window (Windows only).  
  - Cleanly prints `PAN xx  TILT yy`.  

- **Flexible camera options**  
  - Works with OBS Virtual Camera, PS3 Eye, Iriun, or USB webcams.  
  - Supports DirectShow, MSMF, FFmpeg backends.  

---

## Requirements
- Python 3.10 or 3.12 (Mediapipe not yet available on 3.13)  
- Install dependencies:
  ```bash
  pip install opencv-python mediapipe numpy pygame flask
  ```

---

## Running

### Xbox Controller + Camera (default cam index 1, with preview):
```bash
python head_track_motor_console_split.py --xbox-control --cam 1 --api dshow --fourcc mjpg --fps 60 --width 640 --height 480 --split-console
```

### Head Tracking (AUTO) with OBS Virtual Camera (example cam 6):
```bash
python head_track_motor_console_split.py --cam 6 --api dshow --show-fps --model 1 --smooth 0.6 --split-console
```

### Headless Xbox Control (no camera, no preview):
```bash
python head_track_motor_console_split.py --xbox-control --no-camera --no-preview --split-console
```

### Web Remote + Phone Control:
```bash
python head_track_motor_console_split.py --web-remote --cam 1 --split-console
```

---

## Command-Line Options

| Flag               | Default | Description |
|--------------------|---------|-------------|
| `--cam`            | `0`     | Camera index. |
| `--api`            | `dshow` | Backend: `any`, `dshow`, `msmf`, `ffmpeg`. |
| `--fourcc`         | `mjpg`  | Pixel format: `any`, `mjpg`, `yuy2`, `h264`. |
| `--fps`            | `60`    | Target FPS. |
| `--width` / `--height` | `640x480` | Frame size. |
| `--flip`           | off     | Mirror preview. |
| `--show-fps`       | off     | Show FPS overlay. |
| `--smooth`         | `0.60`  | EMA smoothing factor for face box. |
| `--min-conf`       | `0.5`   | Minimum face detection confidence. |
| `--model`          | `1`     | MediaPipe model: `0` = short, `1` = full range. |
| `--pan-only`       | off     | Disable tilt axis. |
| `--center-pan`     | `90`    | Initial pan. |
| `--center-tilt`    | `90`    | Initial tilt. |
| `--min-pan` / `--max-pan` | `20 / 160` | Servo pan range. |
| `--min-tilt` / `--max-tilt` | `30 / 150` | Servo tilt range. |
| `--pan-kp/kd`      | `0.08 / 0.18` | Pan PID gains. |
| `--tilt-kp/kd`     | `0.10 / 0.22` | Tilt PID gains. |
| `--deadband-x/y`   | `12`    | Pixel deadband for small errors. |
| `--rate-limit`     | `180`   | Max slew rate (deg/s). |
| `--print-rate`     | `25`    | Motor update frequency. |
| `--split-console`  | off     | Open second console for motor output. |
| `--threaded-capture` | off   | Use background thread for camera capture. |
| `--detect-scale`   | `1.0`   | Downscale factor for detection (e.g. 0.5). |
| `--no-preview`     | off     | Disable camera preview window. |
| `--no-camera`      | off     | Run without opening a camera. |
| `--web-remote`     | off     | Enable phone/web remote. |
| `--web-port`       | `8080`  | Port for web remote. |
| **Xbox Options:** |||
| `--xbox-control`   | off     | Enable Xbox controller input. |
| `--xbox-index`     | `0`     | Joystick index. |
| `--xbox-deadzone`  | `0.15`  | Stick deadzone. |
| `--xbox-expo`      | `0.35`  | Expo curve (fine control). |
| `--xbox-invert-x`  | off     | Invert pan axis. |
| `--xbox-invert-y`  | off     | Invert tilt axis. |

---

## Notes
- Works with webcams, OBS VirtualCam, and PS3 Eye.  
- Web remote UI is mobile-friendly; “Add to Home Screen” for full-screen app feel.  
- Motor commands are text only (`PAN <deg>  TILT <deg>`). Replace with [pyserial](https://pythonhosted.org/pyserial/) writes to control real servos.  
