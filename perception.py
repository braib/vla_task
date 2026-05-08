import re, os, warnings, time, json
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import cv2

# CRITICAL: Force CPU and hide GPU to prevent driver crashes
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# --- Tuned Thresholds for Reliability ---
# Raising these values forces DINO to be more certain before returning a box.
GDINO_BOX_THRESHOLD  = 0.11  # Was 0.25; prevents "guessing" random spots
GDINO_TEXT_THRESHOLD = 0.08  # Was 0.20; ensures text-image alignment is strong

GDINO_WEIGHT_CANDIDATES = ["weights/groundingdino_swint_ogc.pth"]
GDINO_CONFIG_CANDIDATES = ["weights/GroundingDINO_SwinT_OGC.py"]

@dataclass
class ObjectDescription:
    colour: Optional[str] = None
    shape:  Optional[str] = None
    raw:    str           = ""

    def grounding_text(self) -> str:
            color = self.colour or ""
            shape = self.shape or ""
            main_desc = f"{color} {shape}".strip()
            
            if not main_desc:
                return self.raw.strip()

            # Keep it strictly to two aliases to avoid exhausting the text encoder
            if "cube" in main_desc.lower() or "square" in main_desc.lower():
                return f"{color} cube . {color} block"
                
            if "bowl" in main_desc.lower() or "circle" in main_desc.lower():
                return f"{color} bowl . {color} container"
            
            return main_desc

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
# Stage 1: Gemini Parsing (The "Brain")
# ─────────────────────────────────────────────────────────────────────────────

def parse_prompt(prompt: str) -> tuple[ObjectDescription, ObjectDescription]:
    """Primary entry for parsing, utilizing Gemini's semantic understanding."""
    result = _gemini_parse_v2(prompt)
    if result:
        return result
    
    parts = prompt.lower().split("and")
    t_raw = parts[0]
    d_raw = parts[1] if len(parts) > 1 else ""
    return ObjectDescription(raw=t_raw), ObjectDescription(raw=d_raw)

def _gemini_parse_v2(prompt: str) -> tuple[ObjectDescription, ObjectDescription]:
    from google import genai
    
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("[perception] Error: GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    
    # Note: If "gemini-3-flash-preview" throws a 404 error, change it to "gemini-2.0-flash"
    model_id = "gemini-3-flash-preview" 

    instr = (
        "You are a robotics perception unit. Convert natural language to JSON.\n"
        "Map metaphors to colors: red, green, blue, yellow, orange, purple, pink, cyan, white, black.\n"
        "Map shapes: cube, bowl, cylinder, sphere.\n"
        "Output JSON only: {\"target\": {\"colour\": \"...\", \"shape\": \"...\"}, "
        "\"destination\": {\"colour\": \"...\", \"shape\": \"...\"}}"
    )

    print("[perception] Sending prompt to Gemini API... (waiting for response)")
    
    # Catch rate limits (429) loudly so the terminal doesn't just freeze
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=f"{instr}\n\nCommand: {prompt}"
            )
            break # Success! Break out of the loop
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "exhausted" in error_str:
                print(f"[perception] ⚠ Rate limit hit. Waiting 20 seconds... (Attempt {attempt+1}/3)")
                time.sleep(60)
            else:
                # If it's a different error (like a bad API key or wrong model name), crash loudly
                raise RuntimeError(f" Gemini API Error: {e}")
    else:
        raise RuntimeError("[perception] Failed to get response from Gemini after 3 attempts.")

    text = re.sub(r"```json|```", "", response.text).strip()
    data = json.loads(text)


    t = ObjectDescription(colour=_get_val(data["target"], "colour"), 
                          shape=_get_val(data["target"], "shape"), raw=prompt)
    d = ObjectDescription(colour=_get_val(data["destination"], "colour"), 
                          shape=_get_val(data["destination"], "shape"), raw=prompt)
    
    return t, d

def _get_val(obj, key):
    val = obj.get(key)
    return val if val and str(val).lower() not in ("null", "none") else None

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Grounding DINO (The "Eyes")
# ─────────────────────────────────────────────────────────────────────────────

_gdino_model = None

def _patch_transformers():
    try:
        import transformers.models.bert.modeling_bert as bm
        if not hasattr(bm.BertModel, 'get_head_mask'):
            bm.BertModel.get_head_mask = lambda self, hm, n, chunked=False: [None] * n
    except Exception: pass

def _load_gdino():
    global _gdino_model
    if _gdino_model: return _gdino_model
    _patch_transformers()
    from groundingdino.util.inference import load_model
    config = next((p for p in GDINO_CONFIG_CANDIDATES if os.path.exists(p)), None)
    weight = next((p for p in GDINO_WEIGHT_CANDIDATES if os.path.exists(p)), None)
    
    _gdino_model = load_model(config, weight)
    _gdino_model.to("cpu")
    _gdino_model.eval()
    return _gdino_model

def _run_gdino(rgb: np.ndarray, queries: list[str]) -> list[Optional[tuple]]:
    model = _load_gdino()
    from groundingdino.util.inference import predict
    import torchvision.transforms as T
    
    transform = T.Compose([
        T.ToPILImage(), T.Resize(800), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_t = transform(rgb)
    H, W = rgb.shape[:2]
    results = []

    for query in queries:
        try:
    # ... inside _run_gdino ...
            boxes, logits, phrases = predict(
                model=model, image=img_t, caption=query,
                box_threshold=GDINO_BOX_THRESHOLD,
                text_threshold=GDINO_TEXT_THRESHOLD,
                device="cpu"
            )
            
            print(f"\n[DINO Debug] Query: '{query}'")
            if len(boxes) == 0:
                print("  -> No objects found above threshold.")
                results.append(None)
            else:
                # Print exactly what the model is "thinking" for every box it found
                for i in range(len(logits)):
                    print(f"  Found: '{phrases[i]}' (Confidence: {logits[i].item():.2f})")
                
                # Grab the best one
                best = logits.argmax().item()
                cx, cy, bw, bh = boxes[best].tolist()
                print(f"  -> Selected '{phrases[best]}' at confidence {logits[best].item():.2f}")
                
                results.append((
                    max(0, int((cx - bw/2) * W)), max(0, int((cy - bh/2) * H)),
                    min(W, int((cx + bw/2) * W)), min(H, int((cy + bw/2) * H))
                ))
        except Exception: results.append(None)
    return results

def detect_objects(rgb: np.ndarray, target_desc: ObjectDescription, dest_desc: ObjectDescription) -> DetectionResult:
    """Strict detection that rejects low-confidence matches."""
    result = DetectionResult(target_desc=target_desc, dest_desc=dest_desc)
    queries = [target_desc.grounding_text(), dest_desc.grounding_text()]
    
    print(f"[perception] Running DINO (CPU): {queries}")
    boxes = _run_gdino(rgb, queries)
    
    if boxes[0]:
        result.target_bbox = boxes[0]
        result.target_centroid_px = ((boxes[0][0]+boxes[0][2])//2, (boxes[0][1]+boxes[0][3])//2)
    
    if boxes[1]:
        result.dest_bbox = boxes[1]
        result.dest_centroid_px = ((boxes[1][0]+boxes[1][2])//2, (boxes[1][1]+boxes[1][3])//2)

    result.method = "grounding_dino"
    result.debug_image = _draw_debug(rgb, result)
    return result

def _draw_debug(rgb, result):
    img = rgb.copy()
    for label, bbox, centroid, color in [
        ("target", result.target_bbox, result.target_centroid_px, (0, 255, 0)),
        ("dest", result.dest_bbox, result.dest_centroid_px, (255, 0, 0))
    ]:
        if bbox:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.circle(img, centroid, 5, color, -1)
    return img