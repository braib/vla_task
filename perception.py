"""
perception.py
=============
Perception & grounding module.

Two-stage pipeline
------------------
Stage 1 — NLP parsing
    Extracts (target_description, destination_description) from free-form
    natural language using simple rule-based extraction + keyword mapping.
    No model required.

Stage 2 — Visual Grounding
    Uses Grounding DINO (local, no API key) to ground the text descriptions
    onto the RGB image and return bounding boxes.

    Falls back to colour-histogram centroid matching if Grounding DINO is
    unavailable (useful for CI / headless testing).

Public API
----------
    parse_prompt(prompt: str) -> (target_desc: str, dest_desc: str)

    detect_objects(
        rgb: np.ndarray,                # (H, W, 3) uint8
        target_desc: str,
        dest_desc: str,
    ) -> DetectionResult

    DetectionResult.target_centroid_px  -> (u, v) or None
    DetectionResult.dest_centroid_px    -> (u, v) or None
    DetectionResult.target_bbox         -> (x1,y1,x2,y2) or None
    DetectionResult.dest_bbox           -> (x1,y1,x2,y2) or None
    DetectionResult.debug_image         -> annotated RGB np.ndarray
"""

from __future__ import annotations
import re
import textwrap
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2


# ── colour palette for fallback matcher ───────────────────────────────────────
COLOUR_PALETTE = {
    "red":    ([0,   80,  80],  [10,  255, 255]),   # HSV lower/upper (wrap)
    "green":  ([40,  60,  60],  [85,  255, 255]),
    "blue":   ([100, 80,  80],  [130, 255, 255]),
    "yellow": ([20,  80,  80],  [35,  255, 255]),
    "orange": ([10,  80,  80],  [20,  255, 255]),
    "purple": ([130, 60,  60],  [155, 255, 255]),
    "pink":   ([155, 60,  60],  [175, 255, 255]),
    "white":  ([0,   0,  200],  [180,  30, 255]),
    "black":  ([0,   0,   0],   [180,  255, 30]),
}

# ── shape keywords used for disambiguation ─────────────────────────────────────
SHAPE_KEYWORDS = {
    "cube":        "cube",
    "block":       "cube",
    "box":         "cube",
    "square":      "cube",
    "bowl":        "bowl",
    "container":   "bowl",
    "dish":        "bowl",
    "plate":       "bowl",
    "cup":         "bowl",
    "cylinder":    "cylinder",
    "sphere":      "sphere",
    "ball":        "sphere",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ObjectDescription:
    """Parsed description of a single object."""
    colour: Optional[str]  = None
    shape:  Optional[str]  = None
    raw:    str            = ""

    def grounding_text(self) -> str:
        """Return a clean text query for the visual grounder."""
        parts = []
        if self.colour: parts.append(self.colour)
        if self.shape:  parts.append(self.shape)
        return " ".join(parts) if parts else self.raw

    def colour_keywords(self) -> list[str]:
        """Return colour aliases to try in fallback matcher."""
        return [self.colour] if self.colour else []


@dataclass
class DetectionResult:
    target_desc:       ObjectDescription = field(default_factory=ObjectDescription)
    dest_desc:         ObjectDescription = field(default_factory=ObjectDescription)
    target_centroid_px: Optional[tuple[int,int]] = None   # (u, v)
    dest_centroid_px:   Optional[tuple[int,int]] = None
    target_bbox:        Optional[tuple[int,int,int,int]] = None  # x1y1x2y2
    dest_bbox:          Optional[tuple[int,int,int,int]] = None
    target_score:       float = 0.0
    dest_score:         float = 0.0
    method:             str   = "unknown"
    debug_image:        Optional[np.ndarray] = None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: NLP parsing
# ─────────────────────────────────────────────────────────────────────────────

# Verbs signalling "pick" action
PICK_VERBS = r"(?:pick\s+up|grasp|grab|take|lift|get|fetch)"
# Verbs signalling "place" action
PLACE_VERBS = r"(?:place|put|drop|move|set|transfer|deposit|bring)"

# Prepositions for destination
DEST_PREPS = r"(?:in(?:to|side)?|on(?:to)?|at|to|inside)"

# Strip filler words
FILLER = r"(?:the|a|an|that|this|some|it)\s*"


def _extract_colour(text: str) -> Optional[str]:
    for colour in COLOUR_PALETTE:
        if colour in text.lower():
            return colour
    return None


def _extract_shape(text: str) -> Optional[str]:
    for kw, canonical in SHAPE_KEYWORDS.items():
        if kw in text.lower():
            return canonical
    return None


def _parse_fragment(fragment: str) -> ObjectDescription:
    fragment = fragment.strip()
    colour = _extract_colour(fragment)
    shape  = _extract_shape(fragment)
    return ObjectDescription(colour=colour, shape=shape, raw=fragment)


def parse_prompt(prompt: str) -> tuple[ObjectDescription, ObjectDescription]:
    """
    Parse a natural language pick-and-place command.

    Supports patterns like:
      "Pick up the red cube and put it in the blue bowl"
      "Grab the yellow block and drop it into the red bowl"
      "Move the green object to the blue container"
      "Take the cube that is yellow and place it into the bowl which is blue"

    Returns
    -------
    target_desc : ObjectDescription  — what to pick
    dest_desc   : ObjectDescription  — where to place
    """
    p = prompt.lower().strip()

    # ── Pattern 1: "pick up X and place Y into Z" ─────────────────────────
    # Matches: "pick up the red cube and put it in the blue bowl"
    pat1 = (
        rf"{PICK_VERBS}\s+{FILLER}?(.+?)"        # group 1: target phrase
        rf"\s+and\s+{PLACE_VERBS}\s+(?:it\s+)?{DEST_PREPS}\s+{FILLER}?(.+)"  # group 2: dest phrase
    )
    m = re.search(pat1, p)
    if m:
        return _parse_fragment(m.group(1)), _parse_fragment(m.group(2))

    # ── Pattern 2: "pick up X and place into Z" (no pronoun) ─────────────
    pat2 = (
        rf"{PICK_VERBS}\s+{FILLER}?(.+?)"
        rf"\s+and\s+{PLACE_VERBS}\s+{DEST_PREPS}\s+{FILLER}?(.+)"
    )
    m = re.search(pat2, p)
    if m:
        return _parse_fragment(m.group(1)), _parse_fragment(m.group(2))

    # ── Pattern 3: "move X to/into Z" ────────────────────────────────────
    pat3 = (
        rf"{PICK_VERBS}\s+{FILLER}?(.+?)\s+{DEST_PREPS}\s+{FILLER}?(.+)"
    )
    m = re.search(pat3, p)
    if m:
        return _parse_fragment(m.group(1)), _parse_fragment(m.group(2))

    # ── Pattern 4: "place X into Z" (no pick verb) ───────────────────────
    pat4 = (
        rf"{PLACE_VERBS}\s+{FILLER}?(.+?)\s+{DEST_PREPS}\s+{FILLER}?(.+)"
    )
    m = re.search(pat4, p)
    if m:
        return _parse_fragment(m.group(1)), _parse_fragment(m.group(2))

    # ── Fallback: split on "and" ──────────────────────────────────────────
    parts = re.split(r"\band\b", p, maxsplit=1)
    if len(parts) == 2:
        warnings.warn(f"[perception] Fuzzy parse of: '{prompt}'")
        return _parse_fragment(parts[0]), _parse_fragment(parts[1])

    # Last resort: return whole prompt as target, empty dest
    warnings.warn(f"[perception] Could not parse prompt: '{prompt}'")
    return _parse_fragment(prompt), ObjectDescription(raw="")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Visual Grounding
# ─────────────────────────────────────────────────────────────────────────────

def _try_grounding_dino(
    rgb: np.ndarray,
    queries: list[str],
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
) -> list[Optional[tuple[int,int,int,int]]]:
    """
    Run Grounding DINO on the image.

    Parameters
    ----------
    rgb : (H, W, 3) uint8
    queries : list of text strings, one per object to detect

    Returns
    -------
    list of (x1, y1, x2, y2) bounding boxes or None if not found.
    """
    try:
        from groundingdino.util.inference import load_model, predict
        from groundingdino.util import box_ops
        import torch
        import torchvision.transforms as T
    except ImportError:
        return [None] * len(queries)

    try:
        # Load model (cached after first call)
        if not hasattr(_try_grounding_dino, "_model"):
            import os
            # Try to find weights in common locations
            weight_candidates = [
                "groundingdino_swint_ogc.pth",
                os.path.expanduser("~/.cache/groundingdino/groundingdino_swint_ogc.pth"),
                "weights/groundingdino_swint_ogc.pth",
            ]
            cfg_candidates = [
                "GroundingDINO_SwinT_OGC.py",
                os.path.join(os.path.dirname(__file__), "weights/GroundingDINO_SwinT_OGC.py"),
            ]
            weight_path = next(
                (p for p in weight_candidates if os.path.exists(p)), None
            )
            cfg_path = next(
                (p for p in cfg_candidates if os.path.exists(p)), None
            )
            if weight_path is None or cfg_path is None:
                print("[perception] Grounding DINO weights/config not found, "
                      "falling back to colour matcher.")
                return [None] * len(queries)

            _try_grounding_dino._model = load_model(cfg_path, weight_path)

        model = _try_grounding_dino._model

        # Preprocess image
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(rgb)

        results = []
        H, W = rgb.shape[:2]

        for query in queries:
            boxes, logits, phrases = predict(
                model=model,
                image=img_tensor,
                caption=query,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            if len(boxes) == 0:
                results.append(None)
                continue

            # boxes are (cx, cy, w, h) normalised → convert to pixel x1y1x2y2
            best_idx = logits.argmax().item()
            cx, cy, bw, bh = boxes[best_idx].tolist()
            x1 = int((cx - bw/2) * W)
            y1 = int((cy - bh/2) * H)
            x2 = int((cx + bw/2) * W)
            y2 = int((cy + bh/2) * H)
            results.append((x1, y1, x2, y2))

        return results

    except Exception as e:
        warnings.warn(f"[perception] Grounding DINO error: {e}")
        return [None] * len(queries)


def _colour_fallback(
    rgb: np.ndarray,
    desc: ObjectDescription,
) -> Optional[tuple[int,int,int,int]]:
    """
    Colour-histogram-based fallback: segments by HSV colour, optionally
    filters by shape keyword (bowl vs cube) using aspect ratio.

    Returns (x1, y1, x2, y2) bounding box or None.
    """
    colour = desc.colour
    if colour is None:
        return None

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    lo_arr, hi_arr = COLOUR_PALETTE.get(colour, ([0,0,0],[180,255,255]))
    lo = np.array(lo_arr, dtype=np.uint8)
    hi = np.array(hi_arr, dtype=np.uint8)

    # Red wraps in HSV → two ranges
    if colour == "red":
        mask1 = cv2.inRange(hsv, lo, hi)
        mask2 = cv2.inRange(hsv, np.array([170,80,80]), np.array([180,255,255]))
        mask  = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, lo, hi)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter by shape keyword if present
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ar = w / max(h, 1)

        if desc.shape == "bowl":
            # Bowls appear circular → aspect ratio near 1, larger area
            if 0.6 < ar < 1.6 and area > 300:
                candidates.append((area, (x, y, x+w, y+h)))
        elif desc.shape == "cube":
            # Cubes appear square-ish
            if 0.5 < ar < 1.8:
                candidates.append((area, (x, y, x+w, y+h)))
        else:
            candidates.append((area, (x, y, x+w, y+h)))

    if not candidates:
        return None

    # Return the largest matching region
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _bbox_to_centroid(bbox: tuple[int,int,int,int]) -> tuple[int,int]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _draw_debug(
    rgb: np.ndarray,
    result: DetectionResult,
) -> np.ndarray:
    img = rgb.copy()
    colours_bgr = {"target": (0, 255, 0), "dest": (255, 128, 0)}

    for label, bbox, centroid in [
        ("target", result.target_bbox, result.target_centroid_px),
        ("dest",   result.dest_bbox,   result.dest_centroid_px),
    ]:
        colour = colours_bgr[label]
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1,y1), (x2,y2), colour, 2)
            tag = (result.target_desc if label=="target" else result.dest_desc).grounding_text()
            cv2.putText(img, f"{label}: {tag}", (x1, max(y1-6,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        if centroid is not None:
            cv2.circle(img, centroid, 5, colour, -1)

    method_label = f"method: {result.method}"
    cv2.putText(img, method_label, (8, img.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_objects(
    rgb: np.ndarray,
    target_desc: ObjectDescription,
    dest_desc: ObjectDescription,
) -> DetectionResult:
    """
    Detect target and destination objects in the RGB image.

    Tries Grounding DINO first, falls back to colour segmentation.

    Parameters
    ----------
    rgb         : (H, W, 3) uint8
    target_desc : ObjectDescription for the object to pick
    dest_desc   : ObjectDescription for the destination

    Returns
    -------
    DetectionResult with populated centroid_px and bbox fields.
    """
    result = DetectionResult(target_desc=target_desc, dest_desc=dest_desc)

    target_query = target_desc.grounding_text()
    dest_query   = dest_desc.grounding_text()

    print(f"[perception] Target query : '{target_query}'")
    print(f"[perception] Dest   query : '{dest_query}'")

    # ── Attempt Grounding DINO ─────────────────────────────────────────────
    gdino_boxes = _try_grounding_dino(rgb, [target_query, dest_query])
    target_box_gdino, dest_box_gdino = gdino_boxes

    used_gdino_target = False
    used_gdino_dest   = False

    if target_box_gdino is not None:
        result.target_bbox        = target_box_gdino
        result.target_centroid_px = _bbox_to_centroid(target_box_gdino)
        used_gdino_target = True

    if dest_box_gdino is not None:
        result.dest_bbox        = dest_box_gdino
        result.dest_centroid_px = _bbox_to_centroid(dest_box_gdino)
        used_gdino_dest = True

    # ── Fallback: colour matcher ───────────────────────────────────────────
    if not used_gdino_target:
        box = _colour_fallback(rgb, target_desc)
        if box is not None:
            result.target_bbox        = box
            result.target_centroid_px = _bbox_to_centroid(box)

    if not used_gdino_dest:
        box = _colour_fallback(rgb, dest_desc)
        if box is not None:
            result.dest_bbox        = box
            result.dest_centroid_px = _bbox_to_centroid(box)

    # ── Method label ──────────────────────────────────────────────────────
    if used_gdino_target or used_gdino_dest:
        result.method = "groundingdino+colour_fallback"
    else:
        result.method = "colour_segmentation"

    # ── Debug image ───────────────────────────────────────────────────────
    result.debug_image = _draw_debug(rgb, result)

    # ── Logging ───────────────────────────────────────────────────────────
    def _log(name, centroid, bbox):
        if centroid:
            print(f"[perception] {name}: centroid=({centroid[0]},{centroid[1]}) "
                  f"bbox={bbox}")
        else:
            print(f"[perception] {name}: NOT DETECTED")

    _log("Target", result.target_centroid_px, result.target_bbox)
    _log("Dest  ", result.dest_centroid_px,   result.dest_bbox)

    return result