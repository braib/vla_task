"""
perception.py
=============
Robust Vision-Language grounding module.

Pipeline
--------
1. Prompt Parsing  → Gemini 3 Flash
2. Object Detection → Grounding DINO
3. Fallback         → Image centre if detection fails

Design Goals
------------
- Robust engineering-focused pipeline
- Gemini-based natural language parsing
- Grounding DINO only
- Rich DetectionResult object
- Bounding boxes + centroids
- PIL-based debug visualisation
- Strong failure handling
- Handles missing DINO weights gracefully
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from google import genai

# ─────────────────────────────────────────────────────────────────────────────
# Grounding DINO imports
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    from groundingdino.util.inference import load_model, predict
    import groundingdino.datasets.transforms as T
    GDINO_IMPORT_OK = True
except Exception:
    GDINO_IMPORT_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODEL = "gemini-3-flash-preview"

_GDINO_CONFIG  = "weights/GroundingDINO_SwinT_OGC.py"
_GDINO_WEIGHTS = "weights/groundingdino_swint_ogc.pth"

BOX_THRESHOLD  = 0.35
TEXT_THRESHOLD = 0.25

SYSTEM_PROMPT = (
    "You are a robotics perception assistant. "
    "Extract the object to pick and the destination object. "
    "Return ONLY valid JSON with keys: target and destination. "
    "Example: {\"target\": \"red cube\", \"destination\": \"blue bowl\"}"
)


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Client
# ─────────────────────────────────────────────────────────────────────────────

client = genai.Client()


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ObjectDescription:
    phrase: str = ""

    def grounding_text(self) -> str:
        return self.phrase.strip()

    def __repr__(self):
        return f"ObjectDescription(phrase={self.phrase!r})"


@dataclass
class DetectionResult:
    target_desc: ObjectDescription = field(default_factory=ObjectDescription)
    dest_desc: ObjectDescription = field(default_factory=ObjectDescription)

    target_centroid_px: Optional[tuple[int, int]] = None
    dest_centroid_px: Optional[tuple[int, int]] = None

    target_bbox: Optional[tuple[int, int, int, int]] = None
    dest_bbox: Optional[tuple[int, int, int, int]] = None

    method: str = "unknown"
    debug_image: Optional[np.ndarray] = None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Gemini Prompt Parsing
# ─────────────────────────────────────────────────────────────────────────────


def parse_prompt(prompt: str) -> tuple[ObjectDescription, ObjectDescription]:
    """
    Parse natural language prompt using Gemini.
    """

    full_prompt = f"{SYSTEM_PROMPT}\n\nInstruction: {prompt}"

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=full_prompt,
        )

        raw = response.text.strip()
        print(f"[perception] Gemini raw response: {raw}")

        # Remove markdown fences
        raw = re.sub(r"^```(?:json)?\\s*", "", raw)
        raw = re.sub(r"\\s*```$", "", raw)
        raw = raw.strip()

        # Convert single quotes → double quotes
        raw = raw.replace("'", '"')

        parsed = json.loads(raw)

        target = parsed.get("target", "").strip()
        dest = parsed.get("destination", "").strip()

        t = ObjectDescription(phrase=target)
        d = ObjectDescription(phrase=dest)

        print(f"[perception] Parsed target={t!r}  dest={d!r}")

        return t, d

    except Exception as e:
        print(f"[perception] Gemini parsing failed: {e}")

        # Strong fallback
        return (
            ObjectDescription(phrase=prompt),
            ObjectDescription(phrase="destination"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Grounding DINO
# ─────────────────────────────────────────────────────────────────────────────

_gdino_model = None
_gdino_available = None


def _load_gdino():
    """
    Load Grounding DINO safely.
    """

    global _gdino_model
    global _gdino_available

    if _gdino_available is False:
        return None

    if _gdino_model is not None:
        return _gdino_model

    if not GDINO_IMPORT_OK:
        print("[perception] Grounding DINO import failed")
        _gdino_available = False
        return None

    if not os.path.exists(_GDINO_CONFIG):
        print(f"[perception] Missing config: {_GDINO_CONFIG}")
        _gdino_available = False
        return None

    if not os.path.exists(_GDINO_WEIGHTS):
        print(f"[perception] Missing weights: {_GDINO_WEIGHTS}")
        _gdino_available = False
        return None

    try:
        print("[perception] Loading Grounding DINO...")

        _gdino_model = load_model(
            _GDINO_CONFIG,
            _GDINO_WEIGHTS,
        )

        print("[perception] Grounding DINO loaded ✓")

        _gdino_available = True
        return _gdino_model

    except Exception as e:
        print(f"[perception] Failed to load Grounding DINO: {e}")
        _gdino_available = False
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Image Transform
# ─────────────────────────────────────────────────────────────────────────────


def _rgb_to_tensor(rgb: np.ndarray):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    pil_img = Image.fromarray(rgb)
    tensor, _ = transform(pil_img, None)

    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Detection Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _box_to_pixels(box, W, H):
    cx, cy, bw, bh = box.tolist()

    x1 = int((cx - bw / 2) * W)
    y1 = int((cy - bh / 2) * H)
    x2 = int((cx + bw / 2) * W)
    y2 = int((cy + bh / 2) * H)

    return (x1, y1, x2, y2)



def _centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)



def _best_detection(boxes, logits, phrases, query, W, H):
    """
    Select best matching detection.
    """

    if len(boxes) == 0:
        return None

    query_words = query.lower().split()

    matched = []

    for i, phrase in enumerate(phrases):
        p = phrase.lower()

        if any(word in p for word in query_words):
            matched.append(i)

    candidates = matched if matched else list(range(len(boxes)))

    best_idx = max(candidates, key=lambda i: logits[i].item())

    bbox = _box_to_pixels(boxes[best_idx], W, H)

    print(
        f"[perception] '{query}' → bbox={bbox} "
        f"score={float(logits[best_idx]):.3f} "
        f"phrase='{phrases[best_idx]}'"
    )

    return bbox


# ─────────────────────────────────────────────────────────────────────────────
# PIL Debug Drawing
# ─────────────────────────────────────────────────────────────────────────────


def _draw_debug(rgb: np.ndarray, result: DetectionResult):
    """
    Draw bounding boxes + centroids.
    """

    img = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            18,
        )
    except Exception:
        font = ImageFont.load_default()

    objects = [
        (
            "TARGET",
            result.target_bbox,
            result.target_centroid_px,
            result.target_desc.grounding_text(),
            "red",
        ),
        (
            "DEST",
            result.dest_bbox,
            result.dest_centroid_px,
            result.dest_desc.grounding_text(),
            "lime",
        ),
    ]

    for label, bbox, centroid, text, colour in objects:

        if bbox:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
            draw.text((x1, y1 - 22), f"{label}: {text}", fill=colour, font=font)

        if centroid:
            cx, cy = centroid
            r = 6
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=colour)

    return np.array(img)


# ─────────────────────────────────────────────────────────────────────────────
# Main Detection API
# ─────────────────────────────────────────────────────────────────────────────


def detect_objects(
    rgb: np.ndarray,
    target_desc: ObjectDescription,
    dest_desc: ObjectDescription,
) -> DetectionResult:
    """
    Detect target + destination using Grounding DINO.

    Strong fallback:
    - If detection fails → image centre
    - If DINO unavailable → image centre
    """

    result = DetectionResult(
        target_desc=target_desc,
        dest_desc=dest_desc,
    )

    H, W = rgb.shape[:2]
    centre = (W // 2, H // 2)

    model = _load_gdino()

    # ── DINO unavailable ────────────────────────────────────────────────────
    if model is None:

        result.method = "image_center_fallback"

        result.target_centroid_px = centre
        result.dest_centroid_px = centre

        result.debug_image = _draw_debug(rgb, result)

        return result

    try:
        caption = (
            f"{target_desc.grounding_text()} . "
            f"{dest_desc.grounding_text()}"
        )

        print(f"[perception] DINO caption: '{caption}'")

        tensor = _rgb_to_tensor(rgb)

        boxes, logits, phrases = predict(
            model=model,
            image=tensor,
            caption=caption,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        t_box = _best_detection(
            boxes,
            logits,
            phrases,
            target_desc.grounding_text(),
            W,
            H,
        )

        d_box = _best_detection(
            boxes,
            logits,
            phrases,
            dest_desc.grounding_text(),
            W,
            H,
        )

        # ── Target ──────────────────────────────────────────────────────────
        if t_box:
            result.target_bbox = t_box
            result.target_centroid_px = _centroid(t_box)
        else:
            result.target_centroid_px = centre

        # ── Destination ─────────────────────────────────────────────────────
        if d_box:
            result.dest_bbox = d_box
            result.dest_centroid_px = _centroid(d_box)
        else:
            result.dest_centroid_px = centre

        result.method = "grounding_dino"

    except Exception as e:

        print(f"[perception] Detection failed: {e}")

        result.method = "image_center_fallback"

        result.target_centroid_px = centre
        result.dest_centroid_px = centre

    result.debug_image = _draw_debug(rgb, result)

    print(f"[perception] method = {result.method}")
    print(f"[perception] target = {result.target_centroid_px}")
    print(f"[perception] dest   = {result.dest_centroid_px}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline API
# ─────────────────────────────────────────────────────────────────────────────


def ground_prompt(
    prompt: str,
    rgb: np.ndarray,
) -> DetectionResult:
    """
    Full pipeline:

    prompt → Gemini parsing → Grounding DINO detection
    """

    target_desc, dest_desc = parse_prompt(prompt)

    result = detect_objects(
        rgb,
        target_desc,
        dest_desc,
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Self Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import cv2

    prompt = "Pick up the red cube and drop it into the blue bowl"

    rgb = cv2.cvtColor(
        cv2.imread("test.png"),
        cv2.COLOR_BGR2RGB,
    )

    result = ground_prompt(prompt, rgb)

    Image.fromarray(result.debug_image).save("debug_output.png")

    print(result)
