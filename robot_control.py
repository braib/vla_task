"""
robot_control.py
================
High-level action sequencer for pick-and-place.

Implements the standard 6-step sequence:
  1. hover above target
  2. descend to grasp height
  3. close gripper
  4. lift
  5. move to destination hover
  6. open gripper (release)

Also exposes lower-level primitives (hover, descend, lift, …) for composability.

Public API
----------
    RobotController(env: SimEnv)
    controller.pick_and_place(target_xyz, dest_xyz) -> bool
    controller.home()
"""

from __future__ import annotations
import time
import numpy as np

from starter_code.sim_env import SimEnv, HOVER_OFFSET


# ── tuning constants ───────────────────────────────────────────────────────────
GRASP_Z_OFFSET  = 0.01     # extra descent below the centroid (m) to ensure contact
PLACE_Z_OFFSET  = 0.04     # release height above destination centroid (m)
SETTLE_STEPS    = 300      # physics steps to let the gripper settle after open/close
PRE_GRASP_OPEN  = True     # open gripper before descending
LIFT_HEIGHT     = 0.25     # how high to lift above the grasp point (m)


class RobotController:
    """
    High-level controller wrapping SimEnv's move_to_pose / set_gripper.

    Parameters
    ----------
    env : SimEnv
        Initialised simulation environment.
    verbose : bool
        Print step-by-step progress.
    """

    def __init__(self, env: SimEnv, verbose: bool = True):
        self.env     = env
        self.verbose = verbose

    # ─────────────────────────────────────────────────────────────────────────
    # Primitives
    # ─────────────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self.verbose:
            print(f"[robot_control] {msg}")

    def home(self):
        """Return to the Panda home configuration."""
        self._log("Moving to home configuration ...")
        self.env.reset()
        self.env.step(200)

    def open_gripper(self):
        self._log("Gripper OPEN")
        self.env.set_gripper(open=True)

    def close_gripper(self):
        self._log("Gripper CLOSE")
        self.env.set_gripper(open=False)

    def move_to(
        self,
        xyz: np.ndarray,
        roll: float  = 0.0,
        pitch: float = 0.0,
        yaw: float   = 0.0,
        label: str   = "",
    ) -> bool:
        x, y, z = xyz
        self._log(
            f"move_to {'(' + label + ') ' if label else ''}"
            f"({x:.3f}, {y:.3f}, {z:.3f})"
        )
        ok = self.env.move_to_pose(x, y, z, roll, pitch, yaw)
        if not ok:
            self._log("  ⚠ IK did not fully converge (continuing anyway)")
        return ok

    # ─────────────────────────────────────────────────────────────────────────
    # Pick-and-place sequence
    # ─────────────────────────────────────────────────────────────────────────

    def pick_and_place(
        self,
        target_xyz: np.ndarray,
        dest_xyz:   np.ndarray,
        hover_offset: float = HOVER_OFFSET,
    ) -> bool:
        """
        Execute the full pick-and-place sequence.

        Parameters
        ----------
        target_xyz   : (3,) world position of the object to pick (centroid)
        dest_xyz     : (3,) world position of the destination (centroid)
        hover_offset : height above grasp / place point for approach (m)

        Returns
        -------
        True if all steps completed successfully.
        """
        tx, ty, tz = target_xyz
        dx, dy, dz = dest_xyz

        self._log("=" * 50)
        self._log(f"PICK target  = ({tx:.3f}, {ty:.3f}, {tz:.3f})")
        self._log(f"PLACE dest   = ({dx:.3f}, {dy:.3f}, {dz:.3f})")
        self._log("=" * 50)

        success = True

        # ── Step 0: Open gripper to start ────────────────────────────────
        self.open_gripper()

        # ── Step 1: Hover above target ───────────────────────────────────
        self._log("Step 1 / 6 — hover above target")
        ok = self.move_to(
            np.array([tx, ty, tz + hover_offset]),
            label="hover_above_target"
        )
        success &= ok

        # ── Step 2: Descend to grasp height ──────────────────────────────
        self._log("Step 2 / 6 — descend to grasp height")
        grasp_z = tz - GRASP_Z_OFFSET
        ok = self.move_to(
            np.array([tx, ty, grasp_z]),
            label="grasp_height"
        )
        success &= ok

        # Short settle
        self.env.step(SETTLE_STEPS)

        # ── Step 3: Close gripper ─────────────────────────────────────────
        self._log("Step 3 / 6 — close gripper")
        self.close_gripper()

        # ── Step 4: Lift ──────────────────────────────────────────────────
        self._log("Step 4 / 6 — lift object")
        ok = self.move_to(
            np.array([tx, ty, tz + LIFT_HEIGHT]),
            label="lift"
        )
        success &= ok

        # ── Step 5: Move to destination hover ────────────────────────────
        self._log("Step 5 / 6 — move to destination")
        ok = self.move_to(
            np.array([dx, dy, dz + hover_offset]),
            label="dest_hover"
        )
        success &= ok

        # ── Step 5b: Descend slightly over destination ────────────────────
        self._log("Step 5b — lower over destination")
        ok = self.move_to(
            np.array([dx, dy, dz + PLACE_Z_OFFSET]),
            label="dest_lower"
        )
        success &= ok

        # ── Step 6: Open gripper (release) ───────────────────────────────
        self._log("Step 6 / 6 — release object")
        self.open_gripper()

        # ── Step 7: Retreat ───────────────────────────────────────────────
        self._log("Retreat to hover height")
        self.move_to(
            np.array([dx, dy, dz + hover_offset]),
            label="retreat"
        )

        self._log("Pick-and-place COMPLETE" if success else
                  "Pick-and-place finished WITH WARNINGS")
        return success