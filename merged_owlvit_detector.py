import torch

# =====================================================
# MERGED LABEL SET
# =====================================================

ANIMAL_LABELS = [
    "Dog", "Cat", "Cow", "Horse", "Goat", "Sheep", "Pig",
    "Elephant", "Tiger", "Lion", "Bear", "Deer", "Monkey",
    "Bird", "Snake", "Rabbit", "Fish", "Animal"
]

DAS_LABELS = [
    "Alcohol", "Alcohol bottle", "Beer", "Wine", "Liquor",
    "cigarette", "smoking", "cigar", "vape",
    "weed joint", "cannabis", "drug packet",
    "syringe", "injection", "needle",
    "Tablet", "Pill", "Capsule"
]

WEAPON_LABELS = [
    "knife", "pistol", "gun", "revolver", "rifle",
    "assault rifle", "AK47", "grenade", "weapon",
    "blade", "machete", "bazooka", "sniper rifle", "sword"
]

ALL_LABELS = (
    ANIMAL_LABELS +
    DAS_LABELS +
    WEAPON_LABELS
)

# =====================================================
# PER-CLASS THRESHOLDS (SOURCE OF TRUTH)
# =====================================================

# ---- ANIMAL ----
ANIMAL_THRESHOLDS = {
    "Animal": 0.50  # generic term stricter
}
DEFAULT_ANIMAL_THRESHOLD = 0.44

# ---- DAS ----
DAS_THRESHOLDS = {
    "Alcohol": 0.40,
    "Alcohol bottle": 0.45,
    "Beer": 0.38,
    "Wine": 0.45,
    "Liquor": 0.45,

    "cigarette": 0.50,
    "smoking": 0.50,
    "cigar": 0.50,
    "vape": 0.50,

    "weed joint": 0.50,
    "cannabis": 0.55,
    "drug packet": 0.65,

    "syringe": 0.48,
    "injection": 0.45,
    "needle": 0.50,

    "Tablet": 0.50,
    "Pill": 0.50,
    "Capsule": 0.40,
}
DEFAULT_DAS_THRESHOLD = 0.44


# ---- WEAPON ----
DEFAULT_WEAPON_THRESHOLD = 0.30


# =====================================================
# THRESHOLD RESOLVER (KEY LOGIC)
# =====================================================
def get_threshold(label: str) -> float:
    if label in ANIMAL_LABELS:
        return ANIMAL_THRESHOLDS.get(label, DEFAULT_ANIMAL_THRESHOLD)

    if label in DAS_LABELS:
        return DAS_THRESHOLDS.get(label, DEFAULT_DAS_THRESHOLD)

    if label in WEAPON_LABELS:
        return DEFAULT_WEAPON_THRESHOLD

    return 1.0  # safety: never trigger unknown labels


# =====================================================
# CORE MERGED DETECTOR
# =====================================================
def run_merged_detection(media, model, processor, device):
    """
    media: PIL.Image OR list[PIL.Image]
    """

    frames = media if isinstance(media, list) else [media]

    result = {
        "animal": False,
        "das": False,
        "weapon": False
    }

    for image in frames:
        inputs = processor(
            text=ALL_LABELS,
            images=image,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor(
            [image.size[::-1]]
        ).to(device)

        detections = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=0.25
        )[0]

        for score, label_idx in zip(
            detections["scores"],
            detections["labels"]
        ):
            label = ALL_LABELS[label_idx]
            score = float(score)

            threshold = get_threshold(label)
            if score < threshold:
                continue

            # -------- CATEGORY FLAGS --------
            if label in ANIMAL_LABELS:
                result["animal"] = True

            elif label in DAS_LABELS:
                result["das"] = True

            elif label in WEAPON_LABELS:
                result["weapon"] = True

        # 🔥 EARLY EXIT — only when all found
        if all(result.values()):
            break

    return result
