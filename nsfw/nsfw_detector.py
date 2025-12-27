import os
import cv2
import uuid
import tempfile
from nudenet import NudeDetector

# ----------------------------
# Init model once (IMPORTANT)
# ----------------------------
detector = NudeDetector()

# ----------------------------
# NSFW policy
# ----------------------------
HARD_NSFW = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED"
}

THRESHOLD = 0.5
VIDEO_NSFW_FRAME_LIMIT = 3


# ----------------------------
# Image NSFW detection
# ----------------------------
def image_nsfw(image_path: str) -> bool:
    """
    Returns True if image is NSFW
    """
    try:
        detections = detector.detect(image_path)
        print("[NSFW][IMAGE] Detections:", detections)
    except Exception as e:
        print("[NSFW][IMAGE] Detection failed:", e)
        return False

    for d in detections:
        if d.get("class") in HARD_NSFW and d.get("score", 0) >= THRESHOLD:
            print("[NSFW][IMAGE] HARD NSFW detected")
            return True

    return False


# ----------------------------
# Video NSFW detection
# ----------------------------
def video_nsfw(video_path: str, skip_frames: int = 10) -> bool:
    """
    Returns True if video is NSFW.
    If VIDEO_NSFW_FRAME_LIMIT frames contain NSFW → True
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[NSFW][VIDEO] Failed to open video:", video_path)
        return False

    frame_count = 0
    nsfw_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip logic (optional, currently disabled)
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 1:
                continue

            # Create a unique temp file per frame (SAFE)
            with tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False
            ) as tmp:
                temp_path = tmp.name

            cv2.imwrite(temp_path, frame)

            try:
                detections = detector.detect(temp_path)
            except Exception as e:
                print("[NSFW][VIDEO] Detection error:", e)
                detections = []
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            for d in detections:
                if d.get("class") in HARD_NSFW and d.get("score", 0) >= THRESHOLD:
                    nsfw_frames += 1
                    print(
                        f"[NSFW][VIDEO] NSFW frame detected "
                        f"({nsfw_frames}/{VIDEO_NSFW_FRAME_LIMIT})"
                    )
                    break

            if nsfw_frames >= VIDEO_NSFW_FRAME_LIMIT:
                print("[NSFW][VIDEO] HARD NSFW video detected")
                return True

    finally:
        cap.release()

    return False


# ----------------------------
# Unified NSFW entry function
# ----------------------------
def is_nsfw(path: str) -> bool:
    """
    Detect NSFW for image or video
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()

    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    print(f"[NSFW] Checking file: {path}")

    if ext in image_exts:
        return image_nsfw(path)

    if ext in video_exts:
        return video_nsfw(path)

    raise ValueError(f"[NSFW] Unsupported file type: {ext}")
