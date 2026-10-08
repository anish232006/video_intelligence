"""
scripts/create_demo_videos.py — Generate lightweight synthetic CCTV test videos.

Creates:
- data/videos/camera_01.mp4 (Main Gate: includes a red vehicle moving across frame)
- data/videos/camera_02.mp4 (Parking Area: includes a white vehicle and a person)

These videos allow full offline demonstration and automated smoke testing without
requiring external copyrighted footage downloads.
"""
import os
import sys
from pathlib import Path
import numpy as np

# Ensure data/videos directory exists
VIDEOS_DIR = Path("data/videos")
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


def generate_cctv_video(output_path: Path, scenario: str, duration_sec: int = 6, fps: int = 10):
    """
    Generate a synthetic CCTV video with animated objects and timestamp watermark.
    Uses OpenCV VideoWriter.
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed yet. Please wait for dependencies.")
        return False

    width, height = 640, 480
    total_frames = duration_sec * fps
    
    # FourCC: use mp4v or XVID for broad compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        # Fallback to MJPG or avc1
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
    if not out.isOpened():
        print(f"Warning: Could not open VideoWriter for {output_path}")
        return False

    print(f"Generating synthetic video: {output_path.name} ({duration_sec}s, {fps}fps, scenario={scenario})...")
    
    for frame_idx in range(total_frames):
        # Base background: Asphalt road / security parking lot style
        frame = np.full((height, width, 3), 45, dtype=np.uint8)
        
        # Road markings
        cv2.line(frame, (0, 300), (width, 300), (120, 120, 120), 2)
        cv2.line(frame, (0, 440), (width, 440), (120, 120, 120), 2)
        for dash_x in range(0, width, 50):
            cv2.line(frame, (dash_x, 370), (dash_x + 25, 370), (220, 220, 220), 2)
            
        progress = frame_idx / total_frames
        
        if scenario == "camera_01_main_gate":
            # Scenario 1: A RED CAR drives from left to right along the road
            car_w, car_h = 130, 65
            car_x = int(-100 + progress * (width + 200))
            car_y = 330
            
            # Car body (Bright Red in BGR: B=20, G=20, R=220)
            cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), (20, 20, 220), -1)
            # Car roof/cabin
            cv2.rectangle(frame, (car_x + 25, car_y - 25), (car_x + car_w - 25, car_y), (15, 15, 180), -1)
            # Wheels (dark grey)
            cv2.circle(frame, (car_x + 25, car_y + car_h), 12, (10, 10, 10), -1)
            cv2.circle(frame, (car_x + car_w - 25, car_y + car_h), 12, (10, 10, 10), -1)
            # Headlights (yellow)
            cv2.circle(frame, (car_x + car_w, car_y + 15), 5, (0, 255, 255), -1)
            
            # Label
            cv2.putText(frame, "CAM_01 - MAIN GATE", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif scenario == "camera_02_parking":
            # Scenario 2: A WHITE CAR parked/moving, and a PERSON with backpack walking
            car_w, car_h = 120, 60
            car_x = 350
            car_y = 220
            # White car (BGR: 240, 240, 240)
            cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), (240, 240, 240), -1)
            cv2.rectangle(frame, (car_x + 20, car_y - 20), (car_x + car_w - 20, car_y), (200, 200, 200), -1)
            cv2.circle(frame, (car_x + 20, car_y + car_h), 10, (20, 20, 20), -1)
            cv2.circle(frame, (car_x + car_w - 20, car_y + car_h), 10, (20, 20, 20), -1)

            # Walking person from right to left
            person_x = int(580 - progress * 400)
            person_y = 140
            # Head
            cv2.circle(frame, (person_x, person_y), 10, (180, 150, 130), -1)
            # Torso (Blue shirt: BGR: 220, 50, 50)
            cv2.rectangle(frame, (person_x - 12, person_y + 10), (person_x + 12, person_y + 50), (220, 50, 50), -1)
            # Legs
            cv2.line(frame, (person_x - 6, person_y + 50), (person_x - 10, person_y + 85), (40, 40, 40), 4)
            cv2.line(frame, (person_x + 6, person_y + 50), (person_x + 10, person_y + 85), (40, 40, 40), 4)
            # Backpack on person's back (dark brown)
            cv2.rectangle(frame, (person_x + 10, person_y + 15), (person_x + 22, person_y + 40), (20, 50, 90), -1)
            
            cv2.putText(frame, "CAM_02 - PARKING AREA", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # Add realistic CCTV timecode watermark
        time_sec = frame_idx / fps
        time_str = f"2026-10-08 14:30:{time_sec:05.2f} REC [REC]"
        cv2.putText(frame, time_str, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        out.write(frame)
        
    out.release()
    print(f"Successfully generated: {output_path} ({total_frames} frames)")
    return True


def create_all_demo_videos():
    vid1 = VIDEOS_DIR / "camera_01.mp4"
    vid2 = VIDEOS_DIR / "camera_02.mp4"
    
    success1 = generate_cctv_video(vid1, "camera_01_main_gate", duration_sec=6, fps=10)
    success2 = generate_cctv_video(vid2, "camera_02_parking", duration_sec=6, fps=10)
    return success1 and success2


if __name__ == "__main__":
    create_all_demo_videos()
