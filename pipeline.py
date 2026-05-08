"""
pipeline.py — VLA pick-and-place entry point.

Usage
-----
  python pipeline.py --prompt "Pick up the red cube and place it in the blue bowl"
  python pipeline.py --prompt "..." --record                # also saves demo.mp4
  python pipeline.py --prompt "..." --record --output my.mp4
  python pipeline.py --interactive                          # loop mode
  python pipeline.py --interactive --record                 # record whole session
  python pipeline.py --demo --record                        # two prompts + record
"""

import argparse, sys, time
import numpy as np
import cv2

from starter_code.sim_env import SimEnv
from perception import parse_prompt, detect_objects
from projection import estimate_object_from_depth, get_camera_extrinsics
from robot_control import RobotController

# ── Video recording ───────────────────────────────────────────────────────────
RECORD_W    = 1280
RECORD_H    = 720
RECORD_FPS  = 30
RECORD_SKIP = 3      # capture 1 frame every N physics steps

# ─────────────────────────────────────────────────────────────────────────────
# Video recorder — wraps MuJoCo renderer, captures frames on every step
# ─────────────────────────────────────────────────────────────────────────────

class VideoRecorder:
    """
    Captures frames from a dedicated cinematic MuJoCo renderer and writes
    to an mp4 file. Activated by passing --record on the command line.

    The cinematic camera is a fixed side-angle view showing the full arm
    and table — independent of the interactive viewer camera.
    """

    def __init__(self, model, output_path: str):
        import mujoco
        # Try requested resolution, fall back to 640x480 if framebuffer too small
        w, h = RECORD_W, RECORD_H
        try:
            self._renderer = mujoco.Renderer(model, height=h, width=w)
        except ValueError:
            w, h = 640, 480
            print(f"[recorder] ⚠ Framebuffer too small for {RECORD_W}×{RECORD_H}, "
                  f"falling back to {w}×{h}.")
            print("[recorder]   To enable HD recording add to sim_env.py scene XML:")
            print("[recorder]   <visual><global offwidth='1280' offheight='720'/></visual>")
            self._renderer = mujoco.Renderer(model, height=h, width=w)

        self._w, self._h = w, h
        self._writer    = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            RECORD_FPS,
            (w, h),
        )
        self._step_n    = 0
        self._label     = ""
        self._prompt = ""
        self._cam       = mujoco.MjvCamera()
        self._cam.lookat[:]  = [0.45, 0.05, 0.42]
        self._cam.distance   = 1.35
        self._cam.elevation  = -22
        self._cam.azimuth    = 155
        print(f"[recorder] Initialised - {output_path}  ({w}×{h} @ {RECORD_FPS}fps)")

    def set_label(self, label: str):
        self._label = label

    def set_prompt(self, prompt: str):
        self._prompt = prompt

    def capture(self, data):
        """Call after every physics step. Writes frame every RECORD_SKIP steps."""
        self._step_n += 1
        if self._step_n % RECORD_SKIP != 0:
            return

        self._renderer.update_scene(data, camera=self._cam)
        rgb = self._renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Existing bottom-left stage label
        if self._label:
            cv2.putText(
                bgr,
                self._label,
                (20, self._h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # New top-right prompt label
        if self._prompt:
            text = f"Prompt: {self._prompt}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.55
            thickness = 2

            text_size, _ = cv2.getTextSize(text, font, scale, thickness)
            text_w, text_h = text_size

            x = self._w - text_w - 20
            y = 35

            cv2.putText(
                bgr,
                text,
                (x, y),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        self._writer.write(bgr)

    def close(self):
        self._writer.release()
        self._renderer.close()
        print("[recorder] Video saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Recording-aware SimEnv subclass
# ─────────────────────────────────────────────────────────────────────────────

class RecordingSimEnv(SimEnv):
    """
    SimEnv that optionally captures a frame on every physics step.
    If no recorder is attached it behaves identically to SimEnv.
    """
    _recorder: VideoRecorder | None = None

    def attach_recorder(self, recorder: VideoRecorder):
        self._recorder = recorder

    def set_stage(self, label: str):
        """Update the overlay label shown in the recording."""
        if self._recorder:
            self._recorder.set_label(label)

    def step(self, n: int = 1):
        import mujoco
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            if self._recorder:
                self._recorder.capture(self.data)
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()


# ─────────────────────────────────────────────────────────────────────────────
# Core task runner
# ─────────────────────────────────────────────────────────────────────────────

def run_task(
    env:          RecordingSimEnv,
    controller:   RobotController,
    prompt:       str,
    save_debug:   bool = True,
) -> bool:

    print("\n" + "═" * 60)
    print(f"  PROMPT: {prompt!r}")
    print("═" * 60)

    env.set_stage(f"Prompt: {prompt}")

    if env._recorder:
        env._recorder.set_prompt(prompt)

    # ── 1. Parse ──────────────────────────────────────────────────────────
    target_desc, dest_desc = parse_prompt(prompt)
    print(f"[pipeline] Target : {target_desc.grounding_text()!r}")
    print(f"[pipeline] Dest   : {dest_desc.grounding_text()!r}")

    # ── 2. Photo pose + capture ───────────────────────────────────────────
    env.set_stage("Photo pose — scanning scene")
    print("[pipeline] Moving arm to photo pose ...")
    env.move_to_photo_pose()
    rgb, depth, K = env.get_camera_image()
    print(f"[pipeline] Camera frame: {rgb.shape}  mean={rgb.mean():.1f}")

    # ── 3. Detect ─────────────────────────────────────────────────────────
    env.set_stage("Running perception (Grounding DINO)...")
    print("[pipeline] Running perception ...")
    try:
        detection = detect_objects(rgb, target_desc, dest_desc)
    except Exception as e:
        print(f"[pipeline] ⚠ Perception error: {e}")
        from perception import DetectionResult
        detection = DetectionResult(target_desc=target_desc, dest_desc=dest_desc)

    if save_debug and detection.debug_image is not None:
        ts    = int(time.time())
        fname = f"debug_{ts}.png"
        cv2.imwrite(fname, cv2.cvtColor(detection.debug_image, cv2.COLOR_RGB2BGR))
        print(f"[pipeline] Debug image - {fname}")

    # ── 4. 2D -> 3D depth geometry ─────────────────────────────────────
    target_geom = dest_geom = None

    if not detection.target_centroid_px or not detection.dest_centroid_px:
        print("[pipeline] ❌ Detection failed. No hardcoded body/height fallback is used.")
        return False

    cam_pos, cam_R = get_camera_extrinsics()
    t_px = detection.target_centroid_px
    d_px = detection.dest_centroid_px

    try:
        target_geom = estimate_object_from_depth(
            t_px[0], t_px[1], depth, K, cam_pos, cam_R,
        )
        dest_geom = estimate_object_from_depth(
            d_px[0], d_px[1], depth, K, cam_pos, cam_R,
        )
    except ValueError as e:
        print(f"[pipeline] ❌ Depth geometry failed: {e}")
        print("[pipeline] No fallback to fixed cube/bowl/table heights is used.")
        return False

    print(
        "[pipeline] Target depth geometry: "
        f"top_xyz={target_geom.top_xyz.round(3)}, "
        f"support_z={target_geom.support_z:.3f}, "
        f"height={target_geom.height:.3f}"
    )
    print(
        "[pipeline] Dest depth geometry:   "
        f"top_xyz={dest_geom.top_xyz.round(3)}, "
        f"support_z={dest_geom.support_z:.3f}, "
        f"height={dest_geom.height:.3f}"
    )

    # ── 6. Patch controller to update stage label during execution ────────
    original_log = controller._log
    def labelled_log(msg):
        original_log(msg)
        # Update recording label when we enter a new stage
        if "[Stage" in msg or "Transit" in msg or "gripper" in msg:
            env.set_stage(msg.strip())
    controller._log = labelled_log

    # ── 7. Execute ────────────────────────────────────────────────────────
    print("[pipeline] Executing pick-and-place ...")
    success = controller.pick_and_place(target_geom, dest_geom)

    controller._log = original_log
    env.set_stage("Task complete — at home")
    print("[pipeline] Task complete. Robot at home, ready for next command.")
    return success


# ─────────────────────────────────────────────────────────────────────────────
# Interactive loop
# ─────────────────────────────────────────────────────────────────────────────

def interactive_loop(env, controller):
    print("\n[pipeline] Interactive mode. Type a prompt or 'quit' to exit.")
    print("[pipeline] Example: Pick up the green cube and place it in the red bowl\n")
    while True:
        try:
            prompt = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[pipeline] Interrupted.")
            break
        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("[pipeline] Exiting.")
            break
        run_task(env, controller, prompt)
        print("[pipeline] Ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VLA Pick & Place Pipeline")
    parser.add_argument("--prompt",      type=str,  default=None,
                        help="Natural language command")
    parser.add_argument("--demo",        action="store_true",
                        help="Run two scripted demo prompts")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive loop: type prompts one at a time")
    parser.add_argument("--record",      action="store_true",
                        help="Record execution to a video file")
    parser.add_argument("--output",      type=str,  default="demo.mp4",
                        help="Output video path (default: demo.mp4)")
    parser.add_argument("--seed",        type=int,  default=None,
                        help="Random seed for object placement")
    parser.add_argument("--no-render",   dest="render",
                        action="store_false", default=True)
    args = parser.parse_args()

    if not args.prompt and not args.demo and not args.interactive:
        parser.print_help()
        print('\nExamples:')
        print('  python pipeline.py --prompt "Pick up the red cube and place it in the blue bowl"')
        print('  python pipeline.py --prompt "..." --record')
        print('  python pipeline.py --interactive --record --output session.mp4')
        sys.exit(0)

    # ── Initialise environment ─────────────────────────────────────────────
    print("[pipeline] Initialising MuJoCo environment ...")
    env        = RecordingSimEnv(render=args.render, random_seed=args.seed)
    controller = RobotController(env, verbose=True)

    # ── Attach recorder if requested ───────────────────────────────────────
    recorder = None
    if args.record:
        recorder = VideoRecorder(env.model, args.output)
        env.attach_recorder(recorder)
        print(f"[pipeline] Recording enabled - {args.output}")

    try:
        if args.demo:
            prompts = [
                "Pick up the red cube and place it in the blue bowl",
                "Grab the yellow block and drop it into the red bowl",
            ]
            for p in prompts:
                env.reset()
                env.step(200)
                run_task(env, controller, p)
                # Hold final state for 2s in recording
                if recorder:
                    for _ in range(0, int(2.0/0.002), RECORD_SKIP):
                        env.step(RECORD_SKIP)
                time.sleep(0.5)
            print("\n[pipeline] Demo complete.")
            if not args.interactive:
                # Stay open briefly so viewer is visible
                env.step(500)

        elif args.prompt:
            run_task(env, controller, args.prompt)
            # After task: drop into interactive loop so window stays open
            print("\n[pipeline] Done. Enter another prompt, or 'quit' to exit.")
            interactive_loop(env, controller)

        if args.interactive:
            interactive_loop(env, controller)

    except KeyboardInterrupt:
        print("\n[pipeline] Interrupted.")
    finally:
        # Close env (viewer) BEFORE recorder renderer to avoid GLXBadWindow
        try:
            env.close()
            print("[pipeline] Environment closed.")
        except Exception:
            pass
        if recorder:
            recorder.close()


if __name__ == "__main__":
    main()