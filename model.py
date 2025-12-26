# models.py
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🚀 Loading OWL-V2 model once...")

owl_processor = Owlv2Processor.from_pretrained(
    "google/owlv2-base-patch16"
)

owl_model = Owlv2ForObjectDetection.from_pretrained(
    "google/owlv2-base-patch16"
).to(DEVICE).eval()

# # optional but recommended
# owl_model.half()

print(f"✅ OWL-V2 loaded on {DEVICE}")
