"""
robot_control.py
================
Pick-and-place controller with depth-driven waypoint heights.

Waypoint sequence
-----------------
  Stage 0: open gripper
  Stage 1: pre_grasp      → (tx, ty, TRANSIT_Z)   — computed from depth
  Stage 2: grasp          → (tx, ty, obj_z + GRASP_Z_ABOVE)
  Stage 3: close gripper  + settle
  Stage 4: post_grasp     → (tx, ty, TRANSIT_Z)   — same height as pre_grasp
            ~~~ Cartesian straight-line at TRANSIT_Z via SE3.interpolate ~~~
  Stage 5: pre_drop       → (dx, dy, TRANSIT_Z)   — same height throughout
  Stage 6: drop           → (dx, dy, dest_z + DROP_Z_ABOVE)
  Stage 7: open gripper
  Stage 8: return home    → via IK + PD control (not teleport)

TRANSIT_Z is computed from depth:
    obj_z   = cam_z - min_depth_in_patch    (top of object)
    TRANSIT_Z = obj_z + CLEARANCE           (safe travel height above object)

This means TRANSIT_Z adapts to the actual object height — no hardcoding.
"""

from __future__ import annotations
import numpy as np

# ── Geometry constants (in metres) ───────────────────────────────────────────
GRASP_Z_ABOVE  = 0.005    # EEF site target above obj centroid at grasp
                           # (IK undershoots ~8mm → effective contact at cube side)
DROP_Z_ABOVE   = 0.055    # EEF site target above dest centroid at release
CLEARANCE      = 0.15     # transit height above object top surface
SETTLE_STEPS   = 200      # physics steps to pause at each waypoint stop
GRIPPER_STEPS  = 500      # physics steps after gripper close (build friction)
TRAJ_N         = 40       # Cartesian waypoints in transit segment
DEPTH_PATCH_R  = 7        # pixel radius for min-depth sampling


class RobotController:
    def __init__(self, env, verbose: bool = True):
        self.env     = env
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"[robot_control] {msg}")

    def _move(self, xyz, label=""):
        x, y, z = xyz
        self._log(f"  → ({label}) ({x:.3f}, {y:.3f}, {z:.3f})")
        ok = self.env.move_to_pose(x, y, z, 0., 0., 0.)
        if not ok:
            self._log("    ⚠ IK partial convergence")
        self.env.step(SETTLE_STEPS)
        return ok

    def _open_gripper(self):
        self._log("  gripper → OPEN")
        self.env.set_gripper(open=True)

    def _close_gripper(self):
        self._log("  gripper → CLOSE")
        self.env.set_gripper(open=False)
        self.env.step(GRIPPER_STEPS)   # settle so fingers grip firmly

    def _compute_transit_z(
        self,
        target_u: int, target_v: int,
        dest_u:   int, dest_v:   int,
        depth:    np.ndarray,
    ) -> float:
        """
        Compute TRANSIT_Z from actual depth measurements.

        transit_z = max(obj_z_top, dest_z_top) + CLEARANCE

        Using minimum depth in a patch around each object's centroid pixel
        gives the top surface Z (closest point to camera = highest point).
        """
        target_z = self.env.get_object_z_from_depth(
            target_u, target_v, depth, patch_radius=DEPTH_PATCH_R)
        dest_z   = self.env.get_object_z_from_depth(
            dest_u, dest_v, depth, patch_radius=DEPTH_PATCH_R)

        # Use the higher of the two objects + clearance
        transit_z = max(target_z, dest_z) + CLEARANCE
        self._log(f"  Depth-derived: target_top_z={target_z:.3f}  "
                  f"dest_top_z={dest_z:.3f}  TRANSIT_Z={transit_z:.3f}")
        return transit_z

    def _cartesian_transit(
        self,
        start_xyz: np.ndarray,
        end_xyz:   np.ndarray,
        n: int = TRAJ_N,
    ):
        """Straight-line Cartesian transit at constant Z using SE3.interpolate."""
        import mink
        self._log(f"  Cartesian transit: {start_xyz.round(3)} → {end_xyz.round(3)}")
        self._log(f"  ({n} waypoints at z={start_xyz[2]:.3f})")

        R     = mink.SO3.identity()
        se3_s = mink.SE3.from_rotation_and_translation(R, start_xyz)
        se3_e = mink.SE3.from_rotation_and_translation(R, end_xyz)

        for alpha in np.linspace(0., 1., n):
            wp = se3_s.interpolate(se3_e, float(alpha)).translation()
            self.env.move_to_pose_fast(float(wp[0]), float(wp[1]), float(wp[2]),
                                       0., 0., 0.)
            self.env.step(5)

        self.env.step(SETTLE_STEPS)
        self._log("  Transit complete.")

    def pick_and_place(
        self,
        target_xyz:    np.ndarray,
        dest_xyz:      np.ndarray,
        target_pixel:  tuple[int,int] | None = None,
        dest_pixel:    tuple[int,int]   | None = None,
        depth:         np.ndarray       | None = None,
    ) -> bool:
        """
        Full 8-stage pick-and-place.

        Parameters
        ----------
        target_xyz   : (3,) world position of object to pick
        dest_xyz     : (3,) world position of destination
        target_pixel : (u,v) centroid pixel of target in camera image
        dest_pixel   : (u,v) centroid pixel of destination
        depth        : (H,W) metric depth map from overhead camera
                       If provided, TRANSIT_Z is computed from actual depth.
                       If None, falls back to obj_z + CLEARANCE.
        """
        tx, ty, tz = target_xyz
        dx, dy, dz = dest_xyz

        # ── Compute TRANSIT_Z ─────────────────────────────────────────────
        if depth is not None and target_pixel and dest_pixel:
            tu, tv = target_pixel
            du, dv = dest_pixel
            transit_z = self._compute_transit_z(tu, tv, du, dv, depth)
        else:
            # Fallback: use obj top + clearance (obj centroid + half_cube + clearance)
            transit_z = max(tz, dz) + 0.025 + CLEARANCE
            self._log(f"  No depth provided — using fallback TRANSIT_Z={transit_z:.3f}")

        self._log("=" * 57)
        self._log(f"PICK  target = ({tx:.3f}, {ty:.3f}, {tz:.3f})")
        self._log(f"PLACE dest   = ({dx:.3f}, {dy:.3f}, {dz:.3f})")
        self._log(f"TRANSIT_Z    = {transit_z:.3f} m  (depth-derived)")
        self._log("=" * 57)

        # Stage 0: open gripper
        self._log("\n[Stage 0] Open gripper")
        self._open_gripper()

        # Stage 1: pre-grasp
        pre_grasp = np.array([tx, ty, transit_z])
        self._log(f"\n[Stage 1] Pre-grasp at z={transit_z:.3f}")
        self._move(pre_grasp, "pre_grasp")

        # Stage 2: descend to grasp
        grasp_xyz = np.array([tx, ty, tz + GRASP_Z_ABOVE])
        self._log(f"\n[Stage 2] Grasp at z={grasp_xyz[2]:.3f}")
        self._move(grasp_xyz, "grasp")

        # Stage 3: close gripper
        self._log("\n[Stage 3] Close gripper")
        self._close_gripper()

        # Stage 4: post-grasp (rise to transit height)
        post_grasp = np.array([tx, ty, transit_z])
        self._log(f"\n[Stage 4] Post-grasp — rise to z={transit_z:.3f}")
        self._move(post_grasp, "post_grasp")

        # Transit: Cartesian straight line at transit_z
        pre_drop = np.array([dx, dy, transit_z])
        self._log(f"\n[Transit] Cartesian straight line at z={transit_z:.3f}")
        self._cartesian_transit(post_grasp, pre_drop, n=TRAJ_N)

        # Stage 5: pre-drop (already there after transit)
        self._log(f"\n[Stage 5] Pre-drop at ({dx:.3f}, {dy:.3f}, {transit_z:.3f})")
        self.env.step(SETTLE_STEPS)

        # Stage 6: descend to drop
        drop_xyz = np.array([dx, dy, dz + DROP_Z_ABOVE])
        self._log(f"\n[Stage 6] Drop at z={drop_xyz[2]:.3f}")
        self._move(drop_xyz, "drop")

        # Stage 7: open gripper
        self._log("\n[Stage 7] Open gripper — release")
        self._open_gripper()
        self.env.step(SETTLE_STEPS)

        # Stage 8: return home via IK + PD control
        self._log("\n[Stage 8] Return to home via IK + control")
        self.env.return_home_ik()

        self._log("\n✓ Pick-and-place complete.")
        return True