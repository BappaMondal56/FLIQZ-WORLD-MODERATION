import os
import cv2
import redis
import time
import json
import numpy as np
from pathlib import Path
from PIL import Image

from model import owl_model, owl_processor, DEVICE
from merged_owlvit_detector import run_merged_detection

from face_detect.minor_detect import is_minor
from meetup_detect.personal_details_detect import detect_personal_info
from violance_detect.violation_detect import is_violence_detected

from dynamic_update import dynamic_update
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, INPUT_QUEUE, REDIS_BRPOP_TIMEOUT


# =====================================================
# REDIS
# =====================================================
print("[INIT] Connecting to Redis...")
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True
)
print("[INIT] Redis connected")


# =====================================================
# PATH HANDLING
# =====================================================
POSSIBLE_BASE_PATHS = [
    "/var/www/html/admin.fliqzworld.com/public/storage",
    "/var/www/html/admin.fliqzworld.com/storage",
    "/var/www/html/admin.fliqzworld.com/public_html/storage",
    "D:/codex/bots/NSFW-DETECT-BOT/var/www/html/admin.fliqzworld.com/public/storage",
    "C:/CodeX/fliqz-world-media-bots"
]

def get_valid_base_path():
    for base in POSSIBLE_BASE_PATHS:
        if os.path.exists(base):
            print(f"[PATH] Using base path: {base}")
            return base
    print("[PATH] Using fallback base path")
    return POSSIBLE_BASE_PATHS[0]

SERVER_STORAGE_PATH = get_valid_base_path()

def normalize_file_path(original_file: str) -> str:
    print(f"[PATH] Normalizing file path: {original_file}")
    clean_path = (
        original_file.replace("\\", "/")
        .replace("//", "/")
        .strip()
        .lstrip("/")
    )

    for base in POSSIBLE_BASE_PATHS:
        full_path = os.path.join(base, clean_path).replace("\\", "/")
        if os.path.exists(full_path):
            print(f"[PATH] Resolved path: {full_path}")
            return full_path

    fallback = os.path.join(SERVER_STORAGE_PATH, clean_path).replace("\\", "/")
    print(f"[PATH] Using fallback path: {fallback}")
    return fallback


# =====================================================
# MEDIA TYPES
# =====================================================
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


# =====================================================
# IMAGE DETECTION
# =====================================================
def run_image_detection(image: Image.Image):
    print("[IMAGE] Running OWL on image")
    return run_merged_detection(
        image,
        owl_model,
        owl_processor,
        DEVICE
    )


# =====================================================
# STAGE 1: CHEAP KEYFRAME EXTRACTION
# =====================================================
def extract_candidate_frames(
    video_path: str,
    max_frames: int = 12,
    scene_threshold: float = 25.0
):
    print(f"[VIDEO] Opening video for keyframe extraction: {video_path}")
    cap = cv2.VideoCapture(video_path)

    prev_gray = None
    candidates = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            score = diff.mean()

            if score > scene_threshold:
                print(f"[KEYFRAME] Scene change at frame {frame_idx} (score={score:.2f})")
                candidates.append(frame)

        prev_gray = gray

        if len(candidates) >= max_frames:
            print("[KEYFRAME] Reached max candidate frames")
            break

    cap.release()

    # fallback
    if not candidates:
        print("[KEYFRAME] No scene changes detected, using fallback frame")
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            candidates.append(frame)
        cap.release()

    print(f"[KEYFRAME] Selected {len(candidates)} candidate frames")
    return candidates[:max_frames]


# =====================================================
# STAGE 2 + 3: OWL WITH VOTING
# =====================================================
def run_video_with_voting(
    video_path: str,
    min_hits: int = 3,
    min_ratio: float = 0.667  # 66.7%
):
    print("[VIDEO] Starting video moderation pipeline")
    frames = extract_candidate_frames(video_path)

    total_frames = len(frames)
    print(f"[VIDEO] Total frames checked: {total_frames}")

    label_hits = {
        "animal": 0,
        "das": 0,
        "nsfw": 0,
        "weapon": 0
    }

    for idx, frame in enumerate(frames):
        print(f"[OWL] Running OWL on frame {idx + 1}/{total_frames}")

        image = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        result = run_merged_detection(
            image,
            owl_model,
            owl_processor,
            DEVICE
        )

        print("[DEBUG] OWL raw result:", result)

        for label in label_hits:
            if result.get(label):
                label_hits[label] += 1
                print(f"[VOTE] {label} hit → {label_hits[label]}")

    # -------------------------------------------------
    # FINAL DECISION (ABSOLUTE HITS OR PERCENTAGE)
    # -------------------------------------------------
    label_final = {}

    for label, hits in label_hits.items():
        ratio = hits / total_frames if total_frames > 0 else 0

        label_final[label] = (
            hits >= min_hits or ratio >= min_ratio
        )

        print(
            f"[FINAL] {label.upper()} → "
            f"hits={hits}, ratio={ratio:.2%}, result={label_final[label]}"
        )

    print("[VIDEO] OWL voting completed")
    return label_final


# =====================================================
# PROCESS REDIS MESSAGE
# =====================================================
def process_redis(payload: dict):
    print("[WORKER] Processing new Redis payload")

    payload["table_name"] = payload.get("table")
    payload["primary_key"] = "id"
    payload["key_value"] = payload.get("id")

    data = payload.get("data", {})
    payload["file_path"] = data.get("file")

    if not payload["file_path"]:
        print("[WORKER] No file path found, skipping")
        return

    file_path = normalize_file_path(payload["file_path"])
    if not os.path.exists(file_path):
        print("[WORKER] File does not exist, skipping")
        return

    ext = Path(file_path).suffix.lower()
    print(f"[WORKER] Detected file type: {ext}")

    if ext in IMAGE_EXT:
        print("[WORKER] Image detected")
        image = Image.open(file_path).convert("RGB")
        merged = run_image_detection(image)

    elif ext in VIDEO_EXT:
        print("[WORKER] Video detected")
        merged = run_video_with_voting(file_path)

    else:
        print("[WORKER] Unsupported file type")
        return

    print("[WORKER] Running auxiliary detectors")

    try:
        minor_detected = is_minor(file_path)
        print(f"[AUX] Minor detected: {minor_detected}")
    except:
        minor_detected = False

    try:
        personal_info_detected = detect_personal_info(file_path)
        print(f"[AUX] PII detected: {personal_info_detected}")
    except:
        personal_info_detected = False

    try:
        violence_detected = is_violence_detected(file_path)
        print(f"[AUX] Violence detected: {violence_detected}")
    except:
        violence_detected = False

    print("[DB] Updating database")

    success, status = dynamic_update(
        payload=payload,
        animal_detected=merged.get("animal", False),
        das_detected=merged.get("das", False),
        minor_detected=minor_detected,
        personal_info_detected=personal_info_detected,
        nsfw_detected=merged.get("nsfw", False),
        violence_detected=violence_detected,
        weapon_detected=merged.get("weapon", False)
    )

    print(f"[DB] Update result: {status if success else 'FAILED'}")
    print("Animal Detected:", merged.get("animal", False))
    print("DAS Detected:", merged.get("das", False))
    print("Minor Detected:", minor_detected)
    print("Personal Info Detected:", personal_info_detected)
    print("NSFW Detected:", merged.get("nsfw", False))
    print("Violence Detected:", violence_detected)
    print("Weapon Detected:", merged.get("weapon", False))


# =====================================================
# WORKER LOOP
# =====================================================
def worker():
    print("[WORKER] Media Moderation Worker started")
    print(f"[WORKER] Listening on queue: {INPUT_QUEUE}")

    while True:
        try:
            item = r.brpop(INPUT_QUEUE, timeout=REDIS_BRPOP_TIMEOUT)
            if not item:
                time.sleep(0.1)
                continue

            _, message = item
            payload = json.loads(message)
            process_redis(payload)

        except Exception as e:
            print("[ERROR] Worker error:", e)
            time.sleep(1)


# =====================================================
# ENTRY
# =====================================================
if __name__ == "__main__":
    worker()
