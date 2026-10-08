"""
scripts/generate_sequential_dataset.py
Generates 4 synchronized sequential CCTV surveillance videos (CAM_01 to CAM_04)
featuring the exact same distinctive Red Sedan traveling sequentially through:
  Cam A (CAM_01): North Security Gate Entrance (enters facility)
  Cam B (CAM_02): Central Avenue (drives down the main boulevard)
  Cam C (CAM_03): Urban Crossing Junction (navigates intersection)
  Cam D (CAM_04): South Perimeter Exit Gate (approaches exit barrier)
"""
import cv2
import numpy as np
from pathlib import Path

# Paths to generated assets
ARTIFACTS_DIR = Path(r"C:\Users\Mohammed Anish\.gemini\antigravity-ide\brain\d4671da7-41a1-4e1c-ba0e-eeaf8dd5395a")
CAR_IMG_PATH = ARTIFACTS_DIR / "target_red_car_1791488197278.jpg"

BG_PATHS = {
    "CAM_01": ARTIFACTS_DIR / "cctv_cam1_gate_1791488296617.jpg",
    "CAM_02": ARTIFACTS_DIR / "cctv_cam2_avenue_1791488349732.jpg",
    "CAM_03": ARTIFACTS_DIR / "cctv_cam3_crossing_1791488384991.jpg",
    "CAM_04": ARTIFACTS_DIR / "cctv_cam4_exit_1791488430389.jpg",
}

OUT_DIR = Path("data/videos")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_car_sprite(car_path: Path):
    """Extract car and alpha mask by removing pure white background."""
    car = cv2.imread(str(car_path))
    if car is None:
        raise FileNotFoundError(f"Cannot open car image: {car_path}")
    
    gray = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    
    # Clean mask with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # Extract tight bounding box of car
    pts = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(pts)
    car_cropped = car[y:y+h, x:x+w]
    mask_cropped = mask[y:y+h, x:x+w]
    return car_cropped, mask_cropped


def blend_sprite(bg_frame, sprite, mask, center_x, center_y, width, flip_horizontal=False):
    """Blends sprite onto background with smooth soft shadow and alpha channel."""
    orig_h, orig_w = sprite.shape[:2]
    target_w = int(width)
    target_h = int(orig_h * (target_w / orig_w))
    
    if target_w <= 10 or target_h <= 10:
        return bg_frame
    
    resized_sprite = cv2.resize(sprite, (target_w, target_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    if flip_horizontal:
        resized_sprite = cv2.flip(resized_sprite, 1)
        resized_mask = cv2.flip(resized_mask, 1)
        
    x1 = int(center_x - target_w // 2)
    y1 = int(center_y - target_h // 2)
    x2 = x1 + target_w
    y2 = y1 + target_h
    
    bg_h, bg_w = bg_frame.shape[:2]
    
    # Clamp bounds
    clip_x1 = max(0, x1)
    clip_y1 = max(0, y1)
    clip_x2 = min(bg_w, x2)
    clip_y2 = min(bg_h, y2)
    
    if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
        return bg_frame
    
    sp_x1 = clip_x1 - x1
    sp_y1 = clip_y1 - y1
    sp_x2 = sp_x1 + (clip_x2 - clip_x1)
    sp_y2 = sp_y1 + (clip_y2 - clip_y1)
    
    crop_sprite = resized_sprite[sp_y1:sp_y2, sp_x1:sp_x2]
    crop_mask = resized_mask[sp_y1:sp_y2, sp_x1:sp_x2]
    
    # 1. Subtle soft ground shadow under car
    shadow_y1 = min(bg_h, clip_y1 + int(target_h * 0.7))
    shadow_y2 = min(bg_h, clip_y2 + int(target_h * 0.2))
    if shadow_y2 > shadow_y1:
        shadow_roi = bg_frame[shadow_y1:shadow_y2, clip_x1:clip_x2]
        bg_frame[shadow_y1:shadow_y2, clip_x1:clip_x2] = (shadow_roi * 0.55).astype(np.uint8)
        
    # 2. Alpha blend car body
    alpha = (crop_mask / 255.0)[:, :, np.newaxis]
    roi = bg_frame[clip_y1:clip_y2, clip_x1:clip_x2]
    blended = (crop_sprite * alpha + roi * (1 - alpha)).astype(np.uint8)
    bg_frame[clip_y1:clip_y2, clip_x1:clip_x2] = blended
    return bg_frame


def generate_camera_video(cam_id: str, cam_name: str, bg_path: Path, sprite, mask, trajectory_fn, out_path: Path):
    bg_base = cv2.imread(str(bg_path))
    if bg_base is None:
        raise FileNotFoundError(f"Cannot load background: {bg_path}")
    
    # Ensure 1920x1080 resolution
    bg_base = cv2.resize(bg_base, (1920, 1080), interpolation=cv2.INTER_AREA)
    
    fps = 25.0
    duration_secs = 8.0
    total_frames = int(duration_secs * fps)
    
    # Open VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (1920, 1080))
    
    print(f"Rendering {cam_id} ({cam_name}) -> {out_path.name}...")
    
    for frame_idx in range(total_frames):
        t = frame_idx / fps  # timestamp in seconds
        frame = bg_base.copy()
        
        # Calculate car pose
        car_info = trajectory_fn(t)
        if car_info is not None:
            cx, cy, width, flip = car_info
            frame = blend_sprite(frame, sprite, mask, cx, cy, width, flip)
            
        # Add dynamic live surveillance timecode watermark
        timecode = f"2026-10-09 10:{int(t*10):02d}:{int(t%60):02d}.{int((t%1)*100):02d}"
        cv2.putText(frame, f"{cam_id} - {cam_name.upper()} | {timecode}", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Red REC indicator
        if (frame_idx // 15) % 2 == 0:
            cv2.circle(frame, (1850, 50), 12, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (1780, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
        out.write(frame)
        
    out.release()

    # Transcode to browser-native H.264 (avc1, yuv420p)
    try:
        import imageio_ffmpeg
        import subprocess
        import shutil
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        tmp_h264 = out_path.with_name("h264_" + out_path.name)
        subprocess.run([
            ffmpeg_exe, "-y", "-i", str(out_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-crf", "22",
            str(tmp_h264)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        shutil.move(str(tmp_h264), str(out_path))
    except Exception as e:
        print(f"  Note: H.264 post-transcode skipped: {e}")

    print(f"  Finished {cam_id}: {out_path} ({total_frames} frames, H.264)")


# ── Trajectories for each camera ──────────────────────────────────────────────

def cam1_trajectory(t):
    """Cam A: Enters gate from bottom-center (x=500, y=950), drives into facility (x=950, y=420)."""
    start_t, end_t = 1.0, 6.2
    if t < start_t or t > end_t:
        return None
    progress = (t - start_t) / (end_t - start_t)
    
    # Curved path past security booth
    cx = 520 + progress * 420
    cy = 960 - progress * 520
    # Perspective scaling: gets smaller as it travels into the distance
    width = 380 - progress * 240
    return cx, cy, width, False


def cam2_trajectory(t):
    """Cam B: Enters avenue from bottom (x=700, y=1020), drives forward down avenue (x=980, y=410)."""
    start_t, end_t = 1.2, 6.4
    if t < start_t or t > end_t:
        return None
    progress = (t - start_t) / (end_t - start_t)
    
    cx = 720 + progress * 250
    cy = 1040 - progress * 620
    width = 360 - progress * 250
    return cx, cy, width, False


def cam3_trajectory(t):
    """Cam C: Enters intersection from bottom-right (x=1200, y=980), crosses zebra (x=480, y=420)."""
    start_t, end_t = 1.0, 6.0
    if t < start_t or t > end_t:
        return None
    progress = (t - start_t) / (end_t - start_t)
    
    cx = 1250 - progress * 740
    cy = 980 - progress * 540
    width = 380 - progress * 230
    return cx, cy, width, True  # moving right to left


def cam4_trajectory(t):
    """Cam D: Enters from distance curved road (x=950, y=400), approaches exit gate (x=460, y=820)."""
    start_t, end_t = 1.5, 6.8
    if t < start_t or t > end_t:
        return None
    progress = (t - start_t) / (end_t - start_t)
    
    # Curves towards exit gate, getting larger
    cx = 950 - progress * 480
    cy = 410 + progress * 420
    width = 160 + progress * 240  # gets larger as it approaches camera
    return cx, cy, width, True


def main():
    print("=== Generating 4 Sequential CCTV Videos ===")
    sprite, mask = extract_car_sprite(CAR_IMG_PATH)
    print(f"Extracted car sprite: {sprite.shape}")
    
    cameras = [
        ("CAM_01", "North Entrance Gate", BG_PATHS["CAM_01"], cam1_trajectory, OUT_DIR / "camera_01.mp4"),
        ("CAM_02", "Central Avenue Boulevard", BG_PATHS["CAM_02"], cam2_trajectory, OUT_DIR / "camera_02.mp4"),
        ("CAM_03", "Urban Crossing Junction", BG_PATHS["CAM_03"], cam3_trajectory, OUT_DIR / "camera_03.mp4"),
        ("CAM_04", "South Perimeter Exit", BG_PATHS["CAM_04"], cam4_trajectory, OUT_DIR / "camera_04.mp4"),
    ]
    
    for cam_id, cam_name, bg_path, traj_fn, out_path in cameras:
        generate_camera_video(cam_id, cam_name, bg_path, sprite, mask, traj_fn, out_path)
        
    print("\n[SUCCESS] All 4 sequential videos created successfully in data/videos/!")


if __name__ == "__main__":
    main()
