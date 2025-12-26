import os
import cv2
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
        print("Detections:", detections)
    except Exception as e:
        print("Image detection failed:", e)
        return False

    for d in detections:
        if d["class"] in HARD_NSFW and d["score"] >= THRESHOLD:
            return True

    return False


# ----------------------------
# Video NSFW detection
# ----------------------------
def video_nsfw(video_path: str, skip_frames: int = 0) -> bool:
    """
    Returns True if video is NSFW.
    If 3 frames contain NSFW → True
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Failed to open video:", video_path)
        return False

    frame_count = 0
    nsfw_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip logic (disabled for now)
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 1:
            continue

        # Save frame temporarily in memory
        temp_path = "__temp_frame.jpg"
        cv2.imwrite(temp_path, frame)

        try:
            detections = detector.detect(temp_path)
        except Exception:
            continue

        for d in detections:
            if d["class"] in HARD_NSFW and d["score"] >= THRESHOLD:
                nsfw_frames += 1
                break

        if nsfw_frames >= VIDEO_NSFW_FRAME_LIMIT:
            cap.release()
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return True

    cap.release()
    if os.path.exists(temp_path):
        os.remove(temp_path)

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

    if ext in image_exts:
        return image_nsfw(path)

    if ext in video_exts:
        return video_nsfw(path)

    raise ValueError(f"Unsupported file type: {ext}")
