import re
import spacy
import platform
import pytesseract
from PIL import Image
import requests
from io import BytesIO
import os
import cv2
import numpy as np
from pyzbar.pyzbar import decode as qr_decode

# =========================================================
# Load NLP model (ONCE)
# =========================================================
nlp = spacy.load("en_core_web_sm")

# =========================================================
# Regex patterns
# =========================================================
email_pattern = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
)

url_pattern = re.compile(
    r'(https?:\/\/[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\/[^\s]*)?)'
)

phone_pattern = re.compile(
    r'\+?\d[\d\s\-()]{7,}\d'
)

PLATFORM_DOMAIN = "myvault-web.codextechnolife.com"

number_words = {
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety", "hundred", "thousand", "million", "billion"
}

# =========================================================
# Tesseract path
# =========================================================
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# =========================================================
# Helper checks
# =========================================================
def isEmail(text: str) -> bool:
    return bool(email_pattern.search(text))


def hasPhoneNumber(text: str) -> bool:
    return bool(phone_pattern.search(text))


def hasNumber(n) -> bool:
    if isinstance(n, int):
        return True
    if isinstance(n, str):
        if PLATFORM_DOMAIN in n:
            return False
        return any(ch.isdigit() for ch in n)
    return False


def hasNumberWords(text: str) -> bool:
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return any(word in number_words for word in words)


def hasForbiddenURL(text: str) -> bool:
    for match in url_pattern.finditer(text):
        url = match.group(0).rstrip('.,!?')
        if not re.search(r'\.[a-zA-Z]{2,}', url):
            continue
        if PLATFORM_DOMAIN not in url:
            return True
    return False


def hasAddress(text: str) -> bool:
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in {"GPE", "LOC", "FAC"}:
            return True
    return False


# =========================================================
# Core decision logic
# =========================================================
def isPersonalDetails(text: str) -> bool:
    return any([
        hasForbiddenURL(text),
        isEmail(text),
        hasPhoneNumber(text),
        hasNumberWords(text),
        hasNumber(text),
        hasAddress(text)
    ])


# =========================================================
# QR helpers
# =========================================================
def extract_qr_from_frame(frame) -> list[str]:
    try:
        decoded = qr_decode(frame)
        return [
            obj.data.decode("utf-8", errors="ignore")
            for obj in decoded
        ]
    except Exception:
        return []


# =========================================================
# OCR + QR (Image / URL)
# =========================================================
def extract_text_and_qr_from_file(file_path_or_url: str) -> tuple[str, list]:
    text = ""
    qr_payloads = []

    try:
        if file_path_or_url.startswith("http"):
            response = requests.get(file_path_or_url, timeout=10)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(file_path_or_url)

        text = pytesseract.image_to_string(img)

        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        qr_payloads = extract_qr_from_frame(frame)

    except Exception as e:
        print(f"OCR/QR error: {e}")

    return text, qr_payloads


# =========================================================
# OCR + QR (Video)
# =========================================================
def detect_personal_info_video(video_path, frame_skip=30) -> bool:
    """
    frame_skip=30 → ~1 frame/sec for 30fps video
    """
    if not os.path.exists(video_path):
        return False

    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_skip == 0:
            text = pytesseract.image_to_string(Image.fromarray(frame))
            qr_payloads = extract_qr_from_frame(frame)

            if text and isPersonalDetails(text):
                cap.release()
                return True

            for qr_text in qr_payloads:
                if isPersonalDetails(qr_text):
                    cap.release()
                    return True

        frame_id += 1

    cap.release()
    return False


# =========================================================
# PUBLIC API
# =========================================================
def detect_personal_info(data) -> bool:
    """
    Detect personal information in:
    - Images (OCR + QR)
    - Videos (OCR + QR)
    - URLs
    - Dict payloads
    """

    # -----------------------------
    # STRING INPUT
    # -----------------------------
    if isinstance(data, str):

        # Video
        if data.lower().endswith((
            ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"
        )):
            return detect_personal_info_video(data)

        # Image / URL
        text, qr_payloads = extract_text_and_qr_from_file(data)

        if isPersonalDetails(text):
            return True

        for qr_text in qr_payloads:
            if isPersonalDetails(qr_text):
                return True

        return False

    # -----------------------------
    # DICT INPUT
    # -----------------------------
    elif isinstance(data, dict):
        text_to_check = data.get("text", "")

        if "file" in data:
            file_path = data["file"]

            # Video
            if file_path.lower().endswith((
                ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"
            )):
                return detect_personal_info_video(file_path)

            # Image
            text, qr_payloads = extract_text_and_qr_from_file(file_path)
            text_to_check += " " + text

            for qr_text in qr_payloads:
                if isPersonalDetails(qr_text):
                    return True

        return isPersonalDetails(text_to_check)

    return False
