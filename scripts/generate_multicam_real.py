"""
scripts/generate_multicam_real.py
Extracts and generates 4 visually distinct, authentic CCTV camera streams
from real surveillance footage:
  - CAM_01: Highway Inbound (Wide angle overview)
  - CAM_02: Gate 2 West Checkpoint (Telephoto lane zoom, reverse angle, cool security grade)
  - CAM_03: East Commercial Walkway (Pedestrian corridor zoom, warm grade)
  - CAM_04: Metro Intersection (Center junction close-up, IR monochrome surveillance mode)
"""
import subprocess
import cv2
import numpy as np
from pathlib import Path
import imageio_ffmpeg

RAW_VIDEO = Path(r"C:\Users\Mohammed Anish\Downloads\cctv_footages\AQPAgfNBjYD-KvK-tl9wNFkaN44drzypy2h_FQjj0ajeV6O02NvQrFfZTzLf0iPuhruBbBE_Dm1fFZS6pvvAeQtQm8ZoTsvUouqVJJhPoQqmgA.mp4")
OUT_DIR = Path("data/videos")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

CAM_CONFIGS = [
    {
        "cam_id": "camera_01",
        "title": "CAM 01 - NORTH HIGHWAY OVERVIEW",
        "channel": "CH-01 [WIDE-ANGLE]",
        "start_sec": 0.0,
        "duration_sec": 18.0,
        "crop": (0, 0, 848, 478),
        "flip": False,
        "color_mode": "daylight",
    },
    {
        "cam_id": "camera_02",
        "title": "CAM 02 - GATE 2 WEST CHECKPOINT",
        "channel": "CH-02 [TELEPHOTO-REV]",
        "start_sec": 18.0,
        "duration_sec": 18.0,
        "crop": (0, 70, 480, 380),
        "flip": True,  # Flipped horizontally to simulate opposite traffic direction
        "color_mode": "cool_security",
    },
    {
        "cam_id": "camera_03",
        "title": "CAM 03 - EAST COMMERCIAL WALKWAY",
        "channel": "CH-03 [PEDESTRIAN-ZOOM]",
        "start_sec": 36.0,
        "duration_sec": 18.0,
        "crop": (350, 40, 848, 420),
        "flip": False,
        "color_mode": "warm_day",
    },
    {
        "cam_id": "camera_04",
        "title": "CAM 04 - METRO INTERSECTION",
        "channel": "CH-04 [IR-MONOCHROME]",
        "start_sec": 54.0,
        "duration_sec": 18.0,
        "crop": (170, 130, 700, 478),
        "flip": False,
        "color_mode": "mono_ir",
    },
]

TARGET_WIDTH = 848
TARGET_HEIGHT = 478


def draw_cctv_hud(frame, cam_cfg, frame_idx, fps):
    h, w = frame.shape[:2]
    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (15, 15, 15), -1)
    # Semi-transparent bottom bar
    cv2.rectangle(overlay, (0, h - 26), (w, h), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    # Red REC blinking dot
    elapsed_sec = frame_idx / fps
    total_sec = cam_cfg["start_sec"] + elapsed_sec
    mm = int(total_sec // 60)
    ss = int(total_sec % 60)
    ms = int((total_sec % 1) * 100)

    if int(elapsed_sec * 2) % 2 == 0:
        cv2.circle(frame, (18, 18), 5, (0, 0, 240), -1)
    else:
        cv2.circle(frame, (18, 18), 5, (80, 80, 80), -1)

    # Top Left Title
    cv2.putText(frame, cam_cfg["title"], (32, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)

    # Top Right Timestamp
    time_str = f"2026-10-09 08:{14 + mm:02d}:{ss:02d}.{ms:02d}"
    cv2.putText(frame, time_str, (w - 235, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 235, 150), 1, cv2.LINE_AA)

    # Bottom Left Mode & Bitrate
    cv2.putText(frame, f"{cam_cfg['channel']} | 15.0 FPS | 2048 KBPS", (15, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    # Bottom Right Status
    cv2.putText(frame, "LIVE STREAM OK", (w - 130, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 0), 1, cv2.LINE_AA)

    return frame


def process_camera(cam_cfg):
    cam_id = cam_cfg["cam_id"]
    out_mp4 = OUT_DIR / f"{cam_id}.mp4"
    print(f"\nProcessing {cam_id}: {cam_cfg['title']} -> {out_mp4}")

    cap = cv2.VideoCapture(str(RAW_VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    start_frame = int(cam_cfg["start_sec"] * fps)
    total_frames = int(cam_cfg["duration_sec"] * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Launch FFmpeg pipe
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_mp4)
    ]

    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    x1, y1, x2, y2 = cam_cfg["crop"]
    mode = cam_cfg["color_mode"]

    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            # Loop if needed
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, frame = cap.read()
            if not ret:
                break

        # 1. Spatial crop
        f_crop = frame[y1:y2, x1:x2]

        # 2. Resize to standard surveillance 848x478
        f_res = cv2.resize(f_crop, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # 3. Geometric Flip if configured (simulating opposing vantage point)
        if cam_cfg["flip"]:
            f_res = cv2.flip(f_res, 1)

        # 4. Color Grading
        if mode == "cool_security":
            b, g, r = cv2.split(f_res)
            b = cv2.add(b, 25)
            r = cv2.subtract(r, 12)
            f_res = cv2.merge([b, g, r])
        elif mode == "warm_day":
            b, g, r = cv2.split(f_res)
            r = cv2.add(r, 18)
            b = cv2.subtract(b, 15)
            f_res = cv2.merge([b, g, r])
        elif mode == "mono_ir":
            gray = cv2.cvtColor(f_res, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            f_res = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 5. Professional CCTV HUD Overlay
        f_hud = draw_cctv_hud(f_res, cam_cfg, idx, fps)

        # Write to pipe
        pipe.stdin.write(f_hud.tobytes())

    cap.release()
    pipe.stdin.close()
    pipe.wait()
    print(f"Successfully generated {out_mp4} ({total_frames} frames)")


def main():
    for cam_cfg in CAM_CONFIGS:
        process_camera(cam_cfg)
    print("\nALL 4 DISTINCT CCTV CAMERAS GENERATED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
