"""
pipeline.py
===========
Entry point — wires perception → projection → control into a single callable.

Usage
-----
    python pipeline.py --prompt "Pick up the red cube and place it in the blue bowl"
    python pipeline.py --prompt "Grab the yellow block and drop it into the red bowl" --seed 42
    python pipeline.py --demo          # runs two hard-coded demo prompts sequentially
    python pipeline.py --no-render     # headless (no viewer window, useful for CI)
"""

from __future__ import annotations
import argparse
import sys
import time
import numpy as np
import cv2

# ── local modules ──────────────────────────────────────────────────────────────
from starter_code.sim_env      import SimEnv
from perception   import parse_prompt, detect_objects
from projection   import pixel_to_world, project_to_table, get_camera_extrinsics
from robot_control import RobotController


# ── table height in world frame (must match scene XML) ─────────────────────────
TABLE_Z = 0.42     # top surface of table (m)
BOWL_Z  = 0.425    # bowl centroid z (slightly above table)


def run_task(
    env:        SimEnv,
    controller: RobotController,
    prompt:     str,
    save_debug: bool = True,
) -> bool:
    """
    Run a single pick-and-place task described by `prompt`.

    Steps
    -----
    1. Parse the natural language command.
    2. Capture RGB + depth from overhead camera.
    3. Detect objects in the image.
    4. Back-project centroids to 3D world coordinates.
    5. Execute the pick-and-place sequence.

    Parameters
    ----------
    env        : SimEnv
    controller : RobotController
    prompt     : natural language command
    save_debug : if True, save an annotated image to debug_<timestamp>.png

    Returns
    -------
    True on success.
    """
    print("\n" + "═" * 60)
    print(f"  PROMPT: {prompt!r}")
    print("═" * 60)

    # ── 1. NLP Parsing ────────────────────────────────────────────────────
    target_desc, dest_desc = parse_prompt(prompt)
    print(f"[pipeline] Parsed target : {target_desc.grounding_text()!r}")
    print(f"[pipeline] Parsed dest   : {dest_desc.grounding_text()!r}")

    if not target_desc.colour and not target_desc.shape:
        print("[pipeline] ERROR: Could not extract target description from prompt.")
        return False
    if not dest_desc.colour and not dest_desc.shape:
        print("[pipeline] ERROR: Could not extract destination description from prompt.")
        return False

    # ── 2. Capture camera frame ───────────────────────────────────────────
    print("[pipeline] Capturing overhead camera frame ...")
    rgb, depth, K = env.get_camera_image()
    print(f"[pipeline] RGB: {rgb.shape}  Depth: {depth.shape}  "
          f"K[0,0]={K[0,0]:.1f} K[1,1]={K[1,1]:.1f}")

    # ── 3. Object detection / grounding ──────────────────────────────────
    print("[pipeline] Running perception ...")
    detection = detect_objects(rgb, target_desc, dest_desc)

    if save_debug and detection.debug_image is not None:
        ts    = int(time.time())
        fname = f"debug_{ts}.png"
        cv2.imwrite(fname, cv2.cvtColor(detection.debug_image, cv2.COLOR_RGB2BGR))
        print(f"[pipeline] Debug image saved → {fname}")

    if detection.target_centroid_px is None:
        print("[pipeline] ERROR: Target object not detected in image.")
        return False
    if detection.dest_centroid_px is None:
        print("[pipeline] ERROR: Destination not detected in image.")
        return False

    # ── 4. 2D → 3D back-projection ───────────────────────────────────────
    cam_pos, cam_R = get_camera_extrinsics()

    tu, tv = detection.target_centroid_px
    du, dv = detection.dest_centroid_px

    print(f"[pipeline] Target pixel  = ({tu}, {tv})")
    print(f"[pipeline] Dest   pixel  = ({du}, {dv})")

    target_world = pixel_to_world(tu, tv, depth, K, cam_pos, cam_R)
    dest_world   = pixel_to_world(du, dv, depth, K, cam_pos, cam_R)

    # Snap Z to known table height (compensates for depth noise on small objects)
    target_world = project_to_table(target_world, TABLE_Z)
    dest_world   = project_to_table(dest_world,   TABLE_Z)

    print(f"[pipeline] Target world  = {target_world}")
    print(f"[pipeline] Dest   world  = {dest_world}")

    # Safety check: are objects within the table workspace?
    for label, pos in [("target", target_world), ("dest", dest_world)]:
        if not (0.1 < pos[0] < 0.9 and -0.5 < pos[1] < 0.5):
            print(f"[pipeline] WARNING: {label} position {pos} looks out of workspace.")

    # ── 5. Execute pick-and-place ─────────────────────────────────────────
    print("[pipeline] Executing pick-and-place ...")
    success = controller.pick_and_place(target_world, dest_world)

    # Return arm home
    controller.home()

    return success


def main():
    parser = argparse.ArgumentParser(
        description="VLA Pipeline — Language-conditioned pick & place in MuJoCo"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help='Natural language command, e.g. "Pick up the red cube and place it in the blue bowl"'
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run two demo prompts sequentially"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for object placement"
    )
    parser.add_argument(
        "--no-render", dest="render", action="store_false", default=True,
        help="Disable interactive viewer (headless mode)"
    )
    parser.add_argument(
        "--no-debug-img", dest="save_debug", action="store_false", default=True,
        help="Do not save debug images"
    )
    args = parser.parse_args()

    if args.prompt is None and not args.demo:
        parser.print_help()
        print("\nExample:\n  python pipeline.py --prompt \"Pick up the red cube and place it in the blue bowl\"")
        sys.exit(0)

    # ── Initialise environment ─────────────────────────────────────────────
    print("[pipeline] Initialising MuJoCo environment ...")
    env        = SimEnv(render=args.render, random_seed=args.seed)
    controller = RobotController(env, verbose=True)

    try:
        if args.demo:
            prompts = [
                "Pick up the red cube and place it in the blue bowl",
                "Grab the yellow block and drop it into the red bowl",
            ]
            results = []
            for p in prompts:
                env.reset()
                env.step(100)
                ok = run_task(env, controller, p, save_debug=args.save_debug)
                results.append((p, ok))
                time.sleep(1.0)

            print("\n" + "═" * 60)
            print("DEMO RESULTS")
            print("═" * 60)
            for p, ok in results:
                status = "✓ SUCCESS" if ok else "✗ FAILED"
                print(f"  {status}: {p!r}")

        else:
            ok = run_task(env, controller, args.prompt, save_debug=args.save_debug)
            sys.exit(0 if ok else 1)

    finally:
        env.close()
        print("[pipeline] Environment closed.")


if __name__ == "__main__":
    main()