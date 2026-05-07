# SOLUTION.md — VLA Pick & Place Pipeline

## Architecture Overview

```
NL Prompt
    │
    ▼
┌──────────────────────────────────────────┐
│  perception.py  — Stage 1: NLP Parsing  │
│  parse_prompt()                          │
│  → target_desc, dest_desc               │
└──────────────────┬───────────────────────┘
                   │
    ┌──────────────▼───────────────────────┐
    │  sim_env.py — Camera API            │
    │  get_camera_image()                  │
    │  → rgb (H×W×3), depth (H×W), K      │
    └──────────────┬───────────────────────┘
                   │
    ┌──────────────▼───────────────────────┐
    │  perception.py — Stage 2: Grounding  │
    │  detect_objects()                    │
    │  → DetectionResult (centroids, bbox) │
    └──────────────┬───────────────────────┘
                   │
    ┌──────────────▼───────────────────────┐
    │  projection.py — 2D → 3D            │
    │  pixel_to_world()                    │
    │  → world XYZ (metres)               │
    └──────────────┬───────────────────────┘
                   │
    ┌──────────────▼───────────────────────┐
    │  robot_control.py — Sequencer        │
    │  RobotController.pick_and_place()    │
    │  → 6-step pick & place              │
    └──────────────┬───────────────────────┘
                   │
    ┌──────────────▼───────────────────────┐
    │  sim_env.py — Robot Control API      │
    │  move_to_pose() / set_gripper()      │
    │  → mink IK → data.ctrl → mj_step   │
    └──────────────────────────────────────┘
```

---

## VLM / AI Model Choices

### Vision-Language Grounding: Grounding DINO + Colour Fallback

**Why Grounding DINO?**
- Open-vocabulary detection: handles arbitrary noun phrases, adjective-noun
  combinations, and synonyms ("yellow block", "cube that is yellow",
  "golden coloured box") out of the box.
- No API key required — fully local inference.
- State-of-the-art zero-shot grounding on tabletop manipulation benchmarks.
- pip-installable (`groundingdino-py`).

**Why keep a colour-segmentation fallback?**
- The colour fallback requires zero dependencies beyond OpenCV and runs in
  milliseconds. It handles the most common tabletop cases (distinguishing
  objects by colour) reliably when DINO weights are not available.
- This makes the pipeline runnable without any GPU or model download.

**Why not GPT-4o or Claude Vision?**
- Requires API keys and network access; not suited for robotics deployment.
- Higher latency than local inference.
- Could be added as a third-tier fallback trivially.

### IK Solver: mink (differential IK on MuJoCo)

**Why mink?**
- Native MuJoCo integration: uses MuJoCo's own Jacobian computation
  (`mj_jacSite`) via its C extension — no URDF/Pinocchio import step.
- QP-based: joint limit constraints handled exactly, not heuristically.
- pip-installable, Apache-2.0 licensed.
- Supports `FrameTask` on a named site — perfect for tracking the Panda's
  `attachment_site` in SE3.

**Why not analytic IK?**
- Franka Panda is a 7-DOF redundant manipulator; analytic solutions require
  picking a redundancy resolution strategy manually.
- mink handles redundancy naturally through the QP's null-space behaviour.

---

## 2D-to-3D Coordinate Transformation

### Camera Setup
A single overhead camera is mounted 1.3 m above the table surface
(world position [0.5, 0.0, 1.72]), looking straight down (euler = [π, 0, 0]).
This gives:
- Minimal perspective distortion for flat tabletop objects.
- Depth ≈ camera height − object height for all points on the table, making
  the back-projection numerically stable.

### Intrinsic Matrix

From MuJoCo's pinhole camera model with `fovy` = 60°:

```
fy = (H/2) / tan(fovy/2)   =  (480/2) / tan(30°)  ≈  415.7
fx = fy × (W/H)             =  415.7 × (640/480)   ≈  554.3
cx = W/2 = 320,  cy = H/2 = 240

K = [[554.3,    0,  320],
     [   0,  415.7, 240],
     [   0,    0,    1 ]]
```

### Back-Projection

```
Z       = depth[v, u]              # metric depth from MuJoCo renderer
X_cam   = (u - cx) × Z / fx
Y_cam   = (v - cy) × Z / fy

# Camera → World (fixed known transform)
p_world = cam_pos + R_cam_to_world × [X_cam, Y_cam, Z_cam]
```

where `R_cam_to_world = diag(1, -1, -1)` for a pure downward-looking camera
(camera Y flipped, optical axis = -world Z).

### Depth Noise Mitigation

A 7×7 median filter on the depth patch around the centroid pixel suppresses
single-pixel outliers common in MuJoCo's z-buffer depth.

Additionally, `project_to_table()` snaps the inferred Z to the known table
height (0.42 m). This is valid because all objects rest on the table and their
centroid Z is well-modelled by `table_z + half_height`.

---

## Current Limitations

| Limitation | Description | Fix with More Time |
|---|---|---|
| Top-down grasp only | roll=pitch=yaw=0 throughout. Cannot grasp objects requiring wrist rotation. | Add grasp pose estimation (e.g., AnyGrasp, GraspNet). |
| Colour segmentation fragility | Fallback fails under strong shadows or similar-coloured objects. | Fine-tune DINO on tabletop domain; add depth-based clustering. |
| No grasp success detection | Pipeline does not verify the object was actually grasped (no tactile/force feedback). | Check gripper finger position after close; if still open, re-grasp. |
| Single camera only | Occlusions can hide objects from the overhead view. | Add wrist camera for close-range verification and secondary viewpoint. |
| Fixed table height assumption | `project_to_table` breaks for objects stacked on top of each other. | Use full depth map to estimate per-object height. |
| No collision avoidance in IK | mink's configuration limit is used but no obstacle avoidance. | Add `CollisionAvoidanceLimit` in mink for table / object geoms. |
| Gripper modelled simply | Panda finger dynamics are simplified; grasping small objects can slip. | Tune finger friction / use soft contacts; add weld constraint post-grasp. |

---

## Run Instructions

### Installation

```bash
# 1. Create environment (Python 3.10–3.13)
python -m venv vla_env
source vla_env/bin/activate       # Windows: vla_env\Scripts\activate

# 2. Install core dependencies
pip install -r requirements.txt

# 3. (Optional) Install Grounding DINO for richer language grounding
#    First install PyTorch matching your CUDA version, e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install groundingdino-py

#    Download weights:
mkdir -p weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
     -O weights/groundingdino_swint_ogc.pth
```

### Running

```bash
# Single prompt (with viewer)
python pipeline.py --prompt "Pick up the red cube and place it in the blue bowl"

# With random object placement
python pipeline.py --prompt "Grab the yellow block and drop it into the red bowl" --seed 42

# Run two demo prompts back-to-back
python pipeline.py --demo

# Headless (no viewer window, e.g. on a server)
python pipeline.py --prompt "Pick up the green cube and put it in the blue bowl" --no-render

# Save annotated debug image showing detections
python pipeline.py --prompt "Pick up the red cube and place it in the blue bowl"
# → debug_<timestamp>.png written to current directory
```

### Project Structure

```
vla_task/
├── sim_env.py          # MuJoCo environment, Camera API, Robot Control API
├── perception.py       # NLP parsing + Grounding DINO / colour segmentation
├── projection.py       # 2D pixel → 3D world coordinate math
├── robot_control.py    # 6-step pick-and-place sequencer
├── pipeline.py         # Entry point — wires all modules together
├── requirements.txt
└── SOLUTION.md
```
