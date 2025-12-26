import cv2
import os
import tempfile

# -----------------------------
# Face detection
# -----------------------------
def detect_faces(net, frame, conf_threshold=0.7):
    frameHeight, frameWidth = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        [104, 117, 123], True, False
    )

    net.setInput(blob)
    detections = net.forward()

    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])

    return faceBoxes


# -----------------------------
# Models & constants
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

faceProto = os.path.join(BASE_DIR, "opencv_face_detector.pbtxt")
faceModel = os.path.join(BASE_DIR, "opencv_face_detector_uint8.pb")
ageProto  = os.path.join(BASE_DIR, "age_deploy.prototxt")
ageModel  = os.path.join(BASE_DIR, "age_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = [
    '(0-2)', '(4-6)', '(8-12)', '(15-22)',
    '(23-32)', '(38-43)', '(48-58)', '(60-80)'
]

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet  = cv2.dnn.readNet(ageModel, ageProto)


# -----------------------------
# Normalize image
# -----------------------------
def normalize_to_jpg(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return temp_path


# -----------------------------
# Core frame-level minor check
# -----------------------------
def is_minor_frame(frame):
    faceBoxes = detect_faces(faceNet, frame)

    if not faceBoxes:
        return False

    padding = 20
    h, w = frame.shape[:2]

    for box in faceBoxes:
        x1, y1, x2, y2 = box

        face = frame[
            max(0, y1 - padding):min(y2 + padding, h - 1),
            max(0, x1 - padding):min(x2 + padding, w - 1)
        ]

        if face.size == 0:
            continue

        faceBlob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227),
            MODEL_MEAN_VALUES,
            swapRB=False
        )

        ageNet.setInput(faceBlob)
        agePreds = ageNet.forward()
        ageBucket = AGE_BUCKETS[agePreds[0].argmax()]

        print(f"Detected age bucket: {ageBucket}")

        if ageBucket in ['(0-2)', '(4-6)', '(8-12)']:
            return True

    return False


# -----------------------------
# Image minor detection
# -----------------------------
def is_minor_image(image_path):
    if not os.path.exists(image_path):
        return False

    img = cv2.imread(image_path)
    if img is None:
        return False

    normalized_path = normalize_to_jpg(img)

    try:
        frame = cv2.imread(normalized_path)
        if frame is None:
            return False

        return is_minor_frame(frame)

    finally:
        try:
            os.remove(normalized_path)
        except:
            pass


# -----------------------------
# Video minor detection
# -----------------------------
# def is_minor_video(video_path, frame_skip=15):
#     """
#     frame_skip=15 → check ~2 frames/sec for 30fps video
#     """
#     cap = cv2.VideoCapture(video_path)
#     frame_id = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_id % frame_skip == 0:
#             if is_minor_frame(frame):
#                 cap.release()
#                 return True

#         frame_id += 1

#     cap.release()
#     return False


def is_minor_video(video_path, frame_skip=15, min_percent=0.50, min_frames=10):
    """
     Hybrid rule:
    - OR condition
    - If >= min_frames (default 3) detect minor → True
    - OR if >= min_percent (default 50%) frames detect minor → True
    """

    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    checked_frames = 0
    minor_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_skip == 0:
            checked_frames += 1

            if is_minor_frame(frame):
                minor_frames += 1

                # # ✅ CONDITION 1: at least 3 frames
                # if minor_frames >= min_frames:
                #     cap.release()
                #     return True

        frame_id += 1

    cap.release()

    if checked_frames == 0:
        return False

    percent = minor_frames / checked_frames
    print(f"Minor frames: {minor_frames}/{checked_frames} ({percent:.2%})")

    # ✅ CONDITION 2: percentage ≥ 50%
    return percent >= min_percent



# -----------------------------
# Unified API
# -----------------------------
def is_minor(path):
    ext = os.path.splitext(path)[1].lower()

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}

    if ext in image_exts:
        return is_minor_image(path)

    if ext in video_exts:
        return is_minor_video(path)

    return False
