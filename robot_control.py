"""
robot_control.py
================
Pick-and-place controller with fully depth-derived Z motion.

No cube/bowl/object height constants are used.  The controller receives
DepthObjectEstimate objects from projection.estimate_object_from_depth() and
uses:

  target.support_z -> local surface under picked object
  target.top_z     -> visible top surface of picked object
  target.height    -> measured object height above that support
  dest.top_z       -> visible destination top surface / rim / stack top

The grasp Z is adjusted using the gripper finger-pad height so the gripper
approaches near the object's side centre without driving into the surface.
"""

from __future__ import annotations
import numpy as np

# ── Robot/gripper geometry constants (metres) ────────────────────────────────
# These are NOT object-depth constants. They describe your physical gripper.
# Tune GRIPPER_FINGER_PAD_HEIGHT to the vertical size of the finger contact pad.
GRIPPER_FINGER_PAD_HEIGHT = 0.018
GRIPPER_SURFACE_CLEARANCE = 0.004
GRASP_HEIGHT_FRACTION     = 0.50   # desired grasp: middle of measured object height

# Motion safety/release constants (metres)
CLEARANCE                 = 0.15   # free-space travel margin above visible geometry
PLACE_CLEARANCE           = 0.010  # open gripper slightly above destination top
SETTLE_STEPS              = 200
GRIPPER_STEPS             = 500
TRAJ_N                    = 40


class RobotController:
    def __init__(self, env, verbose: bool = True):
        self.env     = env
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"[robot_control] {msg}")

    def _move(self, xyz, label=""):
        x, y, z = xyz
        self._log(f"  - ({label}) ({x:.3f}, {y:.3f}, {z:.3f})")
        ok = self.env.move_to_pose(x, y, z, 0., 0., 0.)
        if not ok:
            self._log("    ⚠ IK partial convergence")
        self.env.step(SETTLE_STEPS)
        return ok

    def _open_gripper(self):
        self._log("  gripper - OPEN")
        self.env.set_gripper(open=True)

    def _close_gripper(self):
        self._log("  gripper - CLOSE")
        self.env.set_gripper(open=False)
        self.env.step(GRIPPER_STEPS)

    def _compute_grasp_z(self, target_geom) -> float:
        """
        Side-grasp height from measured object geometry and gripper size.

        The desired TCP/EEF height is near the object's mid-height, but clipped
        so the vertical finger pad does not scrape the support plane or ride
        above the object top whenever the measured height allows that.
        """
        support_z = float(target_geom.support_z)
        top_z     = float(target_geom.top_z)
        height    = float(target_geom.height)

        raw_grasp_z = support_z + GRASP_HEIGHT_FRACTION * height
        half_pad = 0.5 * GRIPPER_FINGER_PAD_HEIGHT

        lower_safe = support_z + half_pad + GRIPPER_SURFACE_CLEARANCE
        upper_safe = top_z     - half_pad - GRIPPER_SURFACE_CLEARANCE

        if upper_safe >= lower_safe:
            grasp_z = float(np.clip(raw_grasp_z, lower_safe, upper_safe))
        else:
            # Object is shorter than the finger pad.  Do not invent a height;
            # use the measured mid-height and let the gripper contact the side.
            grasp_z = raw_grasp_z

        self._log(
            f"  Grasp-Z from depth+gripper: support_z={support_z:.3f}, "
            f"top_z={top_z:.3f}, height={height:.3f}, grasp_z={grasp_z:.3f}"
        )
        return grasp_z

    def _cartesian_transit(self, start_xyz: np.ndarray, end_xyz: np.ndarray, n: int = TRAJ_N):
        """Straight-line Cartesian transit at constant Z using SE3.interpolate."""
        import mink
        self._log(f"  Cartesian transit: {start_xyz.round(3)} - {end_xyz.round(3)}")
        self._log(f"  ({n} waypoints at z={start_xyz[2]:.3f})")

        R     = mink.SO3.identity()
        se3_s = mink.SE3.from_rotation_and_translation(R, start_xyz)
        se3_e = mink.SE3.from_rotation_and_translation(R, end_xyz)

        for alpha in np.linspace(0., 1., n):
            wp = se3_s.interpolate(se3_e, float(alpha)).translation()
            self.env.move_to_pose_fast(float(wp[0]), float(wp[1]), float(wp[2]), 0., 0., 0.)
            self.env.step(5)

        self.env.step(SETTLE_STEPS)
        self._log("  Transit complete.")

    def pick_and_place(self, target_geom, dest_geom) -> bool:
        """
        Full pick-and-place using only depth-derived object geometry.

        target_geom and dest_geom must be DepthObjectEstimate objects.
        If they are missing, this function raises an error instead of falling
        back to hardcoded object/body heights.
        """
        if target_geom is None or dest_geom is None:
            raise ValueError("Depth geometry is required. No fallback object heights are allowed.")

        tx, ty = map(float, target_geom.xy)
        dx, dy = map(float, dest_geom.xy)

        grasp_z = self._compute_grasp_z(target_geom)

        # During carry, the object bottom is below the TCP/EEF by this amount.
        carried_bottom_offset = grasp_z - float(target_geom.support_z)

        # Transit height must clear both the picked object's original top and
        # the destination/stack top while carrying the object.
        transit_z = max(
            float(target_geom.top_z) + CLEARANCE,
            float(dest_geom.top_z) + carried_bottom_offset + CLEARANCE,
        )

        # To place/stack: put the carried object's bottom just above dest top.
        drop_z = float(dest_geom.top_z) + carried_bottom_offset + PLACE_CLEARANCE

        self._log("=" * 64)
        self._log(f"PICK  xy=({tx:.3f}, {ty:.3f})  top_z={target_geom.top_z:.3f}")
        self._log(f"DEST  xy=({dx:.3f}, {dy:.3f})  top_z={dest_geom.top_z:.3f}")
        self._log(f"GRASP_Z={grasp_z:.3f}  DROP_Z={drop_z:.3f}  TRANSIT_Z={transit_z:.3f}")
        self._log("=" * 64)

        # Stage 0: open gripper
        self._log("\n[Stage 0] Open gripper")
        self._open_gripper()

        # Stage 1: pre-grasp above the target
        pre_grasp = np.array([tx, ty, transit_z], dtype=float)
        self._log(f"\n[Stage 1] Pre-grasp at depth-derived z={transit_z:.3f}")
        self._move(pre_grasp, "pre_grasp")

        # Stage 2: descend to depth+gripper adjusted side-grasp height
        grasp_xyz = np.array([tx, ty, grasp_z], dtype=float)
        self._log(f"\n[Stage 2] Grasp at depth+gripper adjusted z={grasp_z:.3f}")
        self._move(grasp_xyz, "grasp")

        # Stage 3: close gripper
        self._log("\n[Stage 3] Close gripper")
        self._close_gripper()

        # Stage 4: post-grasp rise
        post_grasp = np.array([tx, ty, transit_z], dtype=float)
        self._log(f"\n[Stage 4] Post-grasp — rise to z={transit_z:.3f}")
        self._move(post_grasp, "post_grasp")

        # Transit: Cartesian straight line at transit_z
        pre_drop = np.array([dx, dy, transit_z], dtype=float)
        self._log(f"\n[Transit] Cartesian straight line at z={transit_z:.3f}")
        self._cartesian_transit(post_grasp, pre_drop, n=TRAJ_N)

        # Stage 5: pre-drop
        self._log(f"\n[Stage 5] Pre-drop at ({dx:.3f}, {dy:.3f}, {transit_z:.3f})")
        self.env.step(SETTLE_STEPS)

        # Stage 6: descend to depth-derived stacking/place height
        drop_xyz = np.array([dx, dy, drop_z], dtype=float)
        self._log(f"\n[Stage 6] Drop at depth-derived z={drop_z:.3f}")
        self._move(drop_xyz, "drop")

        # Stage 7: open gripper
        self._log("\n[Stage 7] Open gripper — release")
        self._open_gripper()
        self.env.step(SETTLE_STEPS)

        # Stage 7b: retreat straight upward before homing
        post_drop = np.array([dx, dy, transit_z], dtype=float)
        self._log(f"\n[Stage 7b] Post-drop retreat — rise to z={transit_z:.3f}")
        self._move(post_drop, "post_drop")

        # Stage 8: return home via IK + control
        self._log("\n[Stage 8] Return to home via IK + control")
        self.env.return_home_ik()

        self._log("\n✓ Pick-and-place complete.")
        return True
