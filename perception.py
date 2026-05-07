"""
perception.py
=============
Stage 1 — Prompt parsing  (spaCy NLP + keyword extraction)
Stage 2 — Visual grounding (Grounding DINO, primary)
           Colour segmentation (fallback when DINO unavailable)

Robustness design
-----------------
Parsing uses spaCy dependency parsing to extract noun chunks, then maps
colour/shape adjectives regardless of word order:
  "red cube"         → colour=red, shape=cube
  "cube that is red" → colour=red, shape=cube  
  "the yellow block" → colour=yellow, shape=cube
  "bowl which is blue" → colour=blue, shape=bowl

If spaCy is unavailable, falls back to exhaustive keyword scanning
across the full phrase — still handles adjective-noun reordering.

Grounding DINO takes the grounding_text() query and grounds it in the
image via cross-modal attention — it handles all synonyms and phrasings
without any threshold tuning.

Install
-------
  pip install groundingdino-py spacy
  python -m spacy download en_core_web_sm
  mkdir -p weights
  wget .../groundingdino_swint_ogc.pth -O weights/groundingdino_swint_ogc.pth
  wget .../GroundingDINO_SwinT_OGC.py   -O weights/GroundingDINO_SwinT_OGC.py
"""

from __future__ import annotations
import re, os, warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2

# ── Grounding DINO config ─────────────────────────────────────────────────────
GDINO_WEIGHT_CANDIDATES = [
    "weights/groundingdino_swint_ogc.pth",
    os.path.expanduser("~/weights/groundingdino_swint_ogc.pth"),
    os.path.expanduser("~/.cache/groundingdino/groundingdino_swint_ogc.pth"),
    "groundingdino_swint_ogc.pth",
]
GDINO_CONFIG_CANDIDATES = [
    "weights/GroundingDINO_SwinT_OGC.py",
    os.path.expanduser("~/weights/GroundingDINO_SwinT_OGC.py"),
    "GroundingDINO_SwinT_OGC.py",
]
GDINO_BOX_THRESHOLD  = 0.25
GDINO_TEXT_THRESHOLD = 0.20

# ── Colour / shape vocabularies ───────────────────────────────────────────────
COLOURS = {
    "red", "green", "blue", "yellow", "orange",
    "purple", "pink", "cyan", "white", "black", "grey", "gray",
}

SHAPE_MAP = {
    # cubes
    "cube": "cube", "block": "cube", "box": "cube",
    "square": "cube", "brick": "cube",
    # bowls
    "bowl": "bowl", "container": "bowl", "dish": "bowl",
    "plate": "bowl", "cup": "bowl", "bin": "bowl", "basket": "bowl",
    # cylinders
    "cylinder": "cylinder", "tube": "cylinder",
    # spheres
    "sphere": "sphere", "ball": "sphere",
}

# ── Colour fallback HSV palette ───────────────────────────────────────────────
COLOUR_HSV = {
    "red":    dict(h_lo=  0, h_hi= 10, s_min= 30, v_min= 20, v_max=244),
    "green":  dict(h_lo= 38, h_hi= 68, s_min= 60, v_min= 40, v_max=244),
    "blue":   dict(h_lo=100, h_hi=125, s_min= 30, v_min= 40, v_max=244),
    "yellow": dict(h_lo= 20, h_hi= 34, s_min=100, v_min= 80, v_max=244),
    "orange": dict(h_lo=  9, h_hi= 20, s_min=140, v_min= 80, v_max=244),
    "purple": dict(h_lo=125, h_hi=155, s_min= 60, v_min= 40, v_max=244),
    "pink":   dict(h_lo=150, h_hi=175, s_min= 50, v_min= 80, v_max=244),
    "cyan":   dict(h_lo= 80, h_hi=100, s_min= 80, v_min= 60, v_max=244),
}
MIN_CONTOUR_AREA = 60


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ObjectDescription:
    colour: Optional[str] = None
    shape:  Optional[str] = None
    raw:    str           = ""

    def grounding_text(self) -> str:
        """Clean text query for DINO — colour + shape if known, else raw."""
        parts = [p for p in [self.colour, self.shape] if p]
        return " ".join(parts) if parts else self.raw.strip()

    def __repr__(self):
        return f"ObjectDescription(colour={self.colour!r}, shape={self.shape!r}, raw={self.raw!r})"


@dataclass
class DetectionResult:
    target_desc:        ObjectDescription = field(default_factory=ObjectDescription)
    dest_desc:          ObjectDescription = field(default_factory=ObjectDescription)
    target_centroid_px: Optional[tuple[int,int]] = None
    dest_centroid_px:   Optional[tuple[int,int]] = None
    target_bbox:        Optional[tuple[int,int,int,int]] = None
    dest_bbox:          Optional[tuple[int,int,int,int]] = None
    method:             str = "unknown"
    debug_image:        Optional[np.ndarray] = None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Robust NLP Parsing
# ─────────────────────────────────────────────────────────────────────────────

def _extract_attributes(text: str) -> ObjectDescription:
    """
    Extract colour and shape from a noun phrase regardless of word order.

    Handles:
      "red cube"          → colour=red, shape=cube
      "cube that is red"  → colour=red, shape=cube
      "the yellow block"  → colour=yellow, shape=cube
      "something red"     → colour=red, shape=None
      "bowl which is blue"→ colour=blue, shape=bowl
    """
    t = text.lower().strip()
    colour = next((c for c in COLOURS if re.search(rf'\b{c}\b', t)), None)
    shape  = next((SHAPE_MAP[k] for k in SHAPE_MAP if re.search(rf'\b{k}\b', t)), None)
    return ObjectDescription(colour=colour, shape=shape, raw=text.strip())


def _spacy_parse(prompt: str) -> tuple[ObjectDescription, ObjectDescription] | None:
    """
    Use spaCy dependency parsing to split prompt into (target_phrase, dest_phrase).
    Returns None if spaCy is unavailable or parsing fails.
    """
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            return None

        doc = nlp(prompt)

        # Find the main pick verb and place verb
        pick_verbs  = {"pick", "grab", "grasp", "take", "lift", "get", "fetch"}
        place_verbs = {"place", "put", "drop", "move", "set", "transfer",
                       "deposit", "bring"}

        pick_obj_span  = None
        place_obj_span = None

        for token in doc:
            if token.lemma_.lower() in pick_verbs:
                # Direct object of pick verb = target
                for child in token.children:
                    if child.dep_ in ("dobj", "obj"):
                        pick_obj_span = doc[child.left_edge.i : child.right_edge.i + 1]
                        break
            if token.lemma_.lower() in place_verbs:
                # Prepositional object (into/in/onto) = destination
                for child in token.children:
                    if child.dep_ == "prep":
                        for pobj in child.children:
                            if pobj.dep_ == "pobj":
                                place_obj_span = doc[pobj.left_edge.i : pobj.right_edge.i + 1]
                                break

        if pick_obj_span and place_obj_span:
            return (
                _extract_attributes(pick_obj_span.text),
                _extract_attributes(place_obj_span.text),
            )
    except Exception:
        pass
    return None


def _regex_parse(prompt: str) -> tuple[ObjectDescription, ObjectDescription] | None:
    """
    Regex fallback: pattern-match common pick-and-place phrasings.
    Handles multi-word phrases and relative clauses.
    """
    p = prompt.lower().strip()

    PICK  = r"(?:pick\s+up|grab|grasp|take|lift|get|fetch|move|transfer)"
    PLACE = r"(?:place|put|drop|move|set|transfer|deposit|bring)"
    PREP  = r"(?:in(?:to|side)?|on(?:to)?|at|to|inside)"
    ART   = r"(?:the|a|an|that|this|some|it)\s*"

    patterns = [
        # "pick up X and place it into Y"
        rf"{PICK}\s+{ART}?(.+?)\s+and\s+{PLACE}\s+(?:it\s+)?{PREP}\s+{ART}?(.+?)(?:\s*$)",
        # "pick up X and place into Y"
        rf"{PICK}\s+{ART}?(.+?)\s+and\s+{PLACE}\s+{PREP}\s+{ART}?(.+?)(?:\s*$)",
        # "move X to Y"
        rf"{PICK}\s+{ART}?(.+?)\s+{PREP}\s+{ART}?(.+?)(?:\s*$)",
        # "place X into Y"
        rf"{PLACE}\s+{ART}?(.+?)\s+{PREP}\s+{ART}?(.+?)(?:\s*$)",
    ]
    for pat in patterns:
        m = re.search(pat, p)
        if m:
            return _extract_attributes(m.group(1)), _extract_attributes(m.group(2))
    return None


def _keyword_fallback(prompt: str) -> tuple[ObjectDescription, ObjectDescription]:
    """
    Last resort: split on 'and', scan each half for colour+shape.
    """
    warnings.warn(f"[perception] Using keyword fallback for: {prompt!r}")
    parts = re.split(r'\band\b', prompt.lower(), maxsplit=1)
    if len(parts) == 2:
        return _extract_attributes(parts[0]), _extract_attributes(parts[1])
    # If no 'and', try splitting on place prepositions
    m = re.search(r'\s+(?:in(?:to)?|on(?:to)?)\s+', prompt.lower())
    if m:
        t = prompt[:m.start()]
        d = prompt[m.end():]
        return _extract_attributes(t), _extract_attributes(d)
    return _extract_attributes(prompt), ObjectDescription(raw="")


def parse_prompt(prompt: str) -> tuple[ObjectDescription, ObjectDescription]:
    """
    Parse a natural language pick-and-place command into (target, destination).

    Tries in order:
      1. spaCy dependency parsing  (robust, handles complex syntax)
      2. Regex pattern matching    (fast, handles common phrasings)
      3. Keyword fallback          (splits on 'and' or prepositions)

    All paths use _extract_attributes() which finds colour/shape regardless
    of word order, so "cube that is yellow" == "yellow cube".
    """
    # Try spaCy first (most robust)
    result = _spacy_parse(prompt)
    if result:
        t, d = result
        if t.colour or t.shape:
            print(f"[perception] Parsed via spaCy: target={t!r}  dest={d!r}")
            return t, d

    # Regex fallback
    result = _regex_parse(prompt)
    if result:
        t, d = result
        if t.colour or t.shape:
            print(f"[perception] Parsed via regex: target={t!r}  dest={d!r}")
            return t, d

    # Last resort
    t, d = _keyword_fallback(prompt)
    print(f"[perception] Parsed via keywords: target={t!r}  dest={d!r}")
    return t, d


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2A: Grounding DINO (primary detector)
# ─────────────────────────────────────────────────────────────────────────────

_gdino_model    = None
_gdino_available: Optional[bool] = None


def _patch_transformers():
    """Patch transformers >= 4.36 which removed get_head_mask from BertModel."""
    try:
        import transformers.models.bert.modeling_bert as bm
        if not hasattr(bm.BertModel, 'get_head_mask'):
            def get_head_mask(self, hm, n, chunked=False):
                return [None] * n
            bm.BertModel.get_head_mask = get_head_mask
    except Exception:
        pass


def _load_gdino():
    global _gdino_model, _gdino_available
    if _gdino_available is False:
        return None
    if _gdino_model is not None:
        return _gdino_model

    try:
        from groundingdino.util.inference import load_model
    except ImportError:
        print("[perception] Grounding DINO not installed. Run: pip install groundingdino-py")
        _gdino_available = False
        return None

    weight = next((p for p in GDINO_WEIGHT_CANDIDATES if os.path.exists(p)), None)
    config = next((p for p in GDINO_CONFIG_CANDIDATES if os.path.exists(p)), None)

    if not weight or not config:
        print("[perception] Grounding DINO weights/config not found.")
        print("[perception]   mkdir -p weights")
        print("[perception]   wget .../groundingdino_swint_ogc.pth -O weights/groundingdino_swint_ogc.pth")
        print("[perception]   wget .../GroundingDINO_SwinT_OGC.py  -O weights/GroundingDINO_SwinT_OGC.py")
        _gdino_available = False
        return None

    _patch_transformers()
    try:
        print(f"[perception] Loading Grounding DINO from {weight} ...")
        _gdino_model = load_model(config, weight)
        _gdino_model.cpu()
        _gdino_model.eval()
        _gdino_available = True
        print("[perception] Grounding DINO loaded on CPU ✓")
        return _gdino_model
    except Exception as e:
        print(f"[perception] Failed to load Grounding DINO: {e}")
        _gdino_available = False
        return None


def _run_gdino(rgb: np.ndarray, queries: list[str]) -> list[Optional[tuple]]:
    model = _load_gdino()
    if model is None:
        return [None] * len(queries)
    try:
        from groundingdino.util.inference import predict
        import torchvision.transforms as T

        transform = T.Compose([
            T.ToPILImage(), T.Resize(800), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_t = transform(rgb)
        H, W  = rgb.shape[:2]

        results = []
        for query in queries:
            def _predict():
                try:
                    return predict(model=model, image=img_t, caption=query,
                                   box_threshold=GDINO_BOX_THRESHOLD,
                                   text_threshold=GDINO_TEXT_THRESHOLD,
                                   device="cpu")
                except TypeError:
                    return predict(model=model, image=img_t, caption=query,
                                   box_threshold=GDINO_BOX_THRESHOLD,
                                   text_threshold=GDINO_TEXT_THRESHOLD)

            try:
                boxes, logits, phrases = _predict()
            except RuntimeError as e:
                if "driver" in str(e).lower() or "cuda" in str(e).lower():
                    print("[perception] CUDA error — forcing CPU and retrying ...")
                    model.cpu()
                    boxes, logits, phrases = _predict()
                else:
                    raise

            if len(boxes) == 0:
                print(f"[perception] DINO: no detection for '{query}'")
                results.append(None)
                continue

            best  = logits.argmax().item()
            score = float(logits[best])
            cx, cy, bw, bh = boxes[best].tolist()
            x1 = max(0, int((cx - bw/2) * W))
            y1 = max(0, int((cy - bh/2) * H))
            x2 = min(W, int((cx + bw/2) * W))
            y2 = min(H, int((cy + bh/2) * H))
            print(f"[perception] DINO '{query}' → "
                  f"bbox=({x1},{y1},{x2},{y2}) score={score:.2f} phrase='{phrases[best]}'")
            results.append((x1, y1, x2, y2))
        return results

    except Exception as e:
        warnings.warn(f"[perception] DINO inference error: {e}")
        return [None] * len(queries)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2B: Colour segmentation fallback
# ─────────────────────────────────────────────────────────────────────────────

def _colour_mask(hsv, colour):
    cfg = COLOUR_HSV.get(colour)
    if cfg is None:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)
    lo = np.array([cfg["h_lo"], cfg["s_min"], cfg["v_min"]], dtype=np.uint8)
    hi = np.array([cfg["h_hi"], 255, cfg.get("v_max", 255)], dtype=np.uint8)
    if colour == "red":
        m1 = cv2.inRange(hsv, lo, hi)
        m2 = cv2.inRange(hsv,
                         np.array([168, cfg["s_min"], cfg["v_min"]], dtype=np.uint8),
                         np.array([180, 255, cfg.get("v_max",255)], dtype=np.uint8))
        return cv2.bitwise_or(m1, m2)
    return cv2.inRange(hsv, lo, hi)


def _best_bbox(mask, shape_hint=None, exclude=None):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if exclude:
        ex1,ey1,ex2,ey2 = exclude
        mask[ey1:ey2, ex1:ex2] = 0
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < MIN_CONTOUR_AREA: continue
        x,y,w,h = cv2.boundingRect(c)
        ar = w / max(h,1)
        if shape_hint=="bowl" and not (0.5<ar<2.0 and a>150): continue
        if shape_hint=="cube" and not (0.4<ar<2.5): continue
        cands.append((a,(x,y,x+w,y+h)))
    if not cands: return None
    return max(cands, key=lambda t: t[0])[1]


def _colour_fallback(rgb, target_desc, dest_desc):
    print("[perception] ⚠ Using colour segmentation fallback "
          "(install Grounding DINO for robust detection)")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    t_box = _best_bbox(_colour_mask(hsv, target_desc.colour), target_desc.shape) \
            if target_desc.colour else None
    excl  = t_box if target_desc.colour == dest_desc.colour else None
    d_box = _best_bbox(_colour_mask(hsv, dest_desc.colour), dest_desc.shape, excl) \
            if dest_desc.colour else None
    return t_box, d_box


# ─────────────────────────────────────────────────────────────────────────────
# Debug drawing
# ─────────────────────────────────────────────────────────────────────────────

def _draw_debug(rgb, result):
    img = rgb.copy()
    for label, bbox, centroid, bgr in [
        ("target", result.target_bbox, result.target_centroid_px, (0,255,0)),
        ("dest",   result.dest_bbox,   result.dest_centroid_px,   (255,128,0)),
    ]:
        desc = result.target_desc if label=="target" else result.dest_desc
        if bbox:
            x1,y1,x2,y2 = bbox
            cv2.rectangle(img,(x1,y1),(x2,y2),bgr,2)
            cv2.putText(img,f"{label}:{desc.grounding_text()}",
                        (x1,max(y1-6,12)),cv2.FONT_HERSHEY_SIMPLEX,0.55,bgr,2)
        if centroid:
            cv2.circle(img,centroid,6,bgr,-1)
    cv2.putText(img,f"[{result.method}]",(8,img.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)
    return img


def _centroid(bbox):
    x1,y1,x2,y2 = bbox
    return ((x1+x2)//2,(y1+y2)//2)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_objects(
    rgb: np.ndarray,
    target_desc: ObjectDescription,
    dest_desc:   ObjectDescription,
) -> DetectionResult:
    """
    Detect target and destination objects in the RGB image.
    Primary: Grounding DINO (open-vocab, handles any phrasing).
    Fallback: HSV colour segmentation.
    """
    result = DetectionResult(target_desc=target_desc, dest_desc=dest_desc)
    tq, dq = target_desc.grounding_text(), dest_desc.grounding_text()
    print(f"[perception] Target query: '{tq}'")
    print(f"[perception] Dest   query: '{dq}'")

    t_box, d_box = None, None

    # ── Grounding DINO ─────────────────────────────────────────────
    gdino = _run_gdino(rgb, [tq, dq])
    t_box, d_box = gdino[0], gdino[1]

    # ── ALWAYS run colour fallback for testing ────────────────────
    fallback_t_box, fallback_d_box = _colour_fallback(
        rgb,
        target_desc,
        dest_desc,
    )

    print("\n[TEST] ───────── COMPARISON ─────────")

    print(f"[TEST] DINO Target Box     : {t_box}")
    print(f"[TEST] DINO Dest Box       : {d_box}")

    print(f"[TEST] Fallback Target Box : {fallback_t_box}")
    print(f"[TEST] Fallback Dest Box   : {fallback_d_box}")

    print("[TEST] ─────────────────────────────────\n")

    # ── Actual pipeline logic remains unchanged ───────────────────
    if not t_box and not d_box:
        t_box, d_box = fallback_t_box, fallback_d_box
        result.method = "colour_segmentation_fallback"
    else:
        result.method = "grounding_dino"

    if t_box:
        result.target_bbox        = t_box
        result.target_centroid_px = _centroid(t_box)
    if d_box:
        result.dest_bbox          = d_box
        result.dest_centroid_px   = _centroid(d_box)

    result.debug_image = _draw_debug(rgb, result)

    for name, c, b in [("Target", result.target_centroid_px, result.target_bbox),
                        ("Dest  ", result.dest_centroid_px,   result.dest_bbox)]:
        if c: print(f"[perception] {name}: centroid=({c[0]},{c[1]}) bbox={b}")
        else: print(f"[perception] {name}: NOT DETECTED")

    return result