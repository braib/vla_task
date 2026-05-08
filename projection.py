"""
projection.py
=============
2D pixel -> 3D world coordinate back-projection plus depth-only object-height
estimation for pick-and-place.

Important principle
-------------------
No cube/bowl/table height is hardcoded here.  For every detected object, the
visible top surface is estimated from the depth map.  The local support surface
is estimated from an annulus around the detection pixel.  This allows objects
resting on the table or stacked on a visible support object to produce different
Z values automatically.

Physical limitation
-------------------
With one overhead depth image, a support plane hidden completely by the target
object cannot be observed.  In that case the estimator raises an error instead
of falling back to hardcoded heights.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Fixed camera pose (matches scene XML / SimEnv)
# ─────────────────────────────────────────────────────────────────────────────

# Overhead camera body position in world frame (metres)
_CAM_WORLD_POS = np.array([0.5, 0.0, 1.72])

# MuJoCo overhead camera convention used in this project.
_CAM_R = np.array([
    [ 1,  0,  0],   # camera X  -> world +X
    [ 0, -1,  0],   # camera Y  -> world -Y
    [ 0,  0, -1],   # camera Z  -> world -Z
], dtype=np.float64)


@dataclass(frozen=True)
class DepthObjectEstimate:
    """Depth-derived geometry for one detected object."""
    pixel: tuple[int, int]
    top_xyz: np.ndarray       # world point on visible top surface
    support_z: float          # local surface below/around object
    top_z: float              # visible top-surface world Z
    height: float             # top_z - support_z
    top_depth: float          # metric camera depth of top surface
    support_depth: float      # metric camera depth of support surface

    @property
    def xy(self) -> np.ndarray:
        return self.top_xyz[:2]


def get_camera_extrinsics() -> tuple[np.ndarray, np.ndarray]:
    """Return (cam_pos, cam_R) for the fixed overhead camera."""
    return _CAM_WORLD_POS.copy(), _CAM_R.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Core math
# ─────────────────────────────────────────────────────────────────────────────

def _valid_depth_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return values[np.isfinite(values) & (values > 0)]


def depth_at_pixel(
    depth: np.ndarray,
    u: int,
    v: int,
    kernel: int = 7,
) -> float:
    """Return a robust median depth estimate at pixel (u, v)."""
    H, W = depth.shape
    r = kernel // 2
    v0, v1 = max(0, v - r), min(H, v + r + 1)
    u0, u1 = max(0, u - r), min(W, u + r + 1)
    valid = _valid_depth_values(depth[v0:v1, u0:u1])
    if valid.size == 0:
        raise ValueError(f"No valid depth at pixel ({u}, {v})")
    return float(np.median(valid))


def pixel_to_camera(u: int, v: int, Z: float, K: np.ndarray) -> np.ndarray:
    """Back-project a pixel (u, v) at depth Z into the camera frame."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float64)


def camera_to_world(
    p_cam: np.ndarray,
    cam_pos: np.ndarray,
    cam_R: np.ndarray,
) -> np.ndarray:
    """Transform a point from camera frame to world frame."""
    return cam_pos + cam_R @ p_cam


def pixel_to_world_at_depth(
    u: int,
    v: int,
    depth_value: float,
    K: np.ndarray,
    cam_pos: np.ndarray | None = None,
    cam_R: np.ndarray | None = None,
) -> np.ndarray:
    """Back-project pixel (u, v) using a supplied metric depth value."""
    if cam_pos is None or cam_R is None:
        cam_pos, cam_R = get_camera_extrinsics()
    p_cam = pixel_to_camera(u, v, float(depth_value), K)
    return camera_to_world(p_cam, cam_pos, cam_R)


def pixel_to_world(
    u: int,
    v: int,
    depth: np.ndarray,
    K: np.ndarray,
    cam_pos: np.ndarray | None = None,
    cam_R: np.ndarray | None = None,
    depth_kernel: int = 7,
) -> np.ndarray:
    """Full pipeline: pixel (u, v) -> world XYZ using local median depth."""
    Z = depth_at_pixel(depth, u, v, depth_kernel)
    return pixel_to_world_at_depth(u, v, Z, K, cam_pos, cam_R)


# ─────────────────────────────────────────────────────────────────────────────
# Depth-derived object geometry
# ─────────────────────────────────────────────────────────────────────────────

def _world_z_from_depth_value(depth_value: float, cam_pos: np.ndarray, cam_R: np.ndarray) -> float:
    """
    Convert camera depth to world Z for the overhead camera.

    This is equivalent to back-projecting the centre ray for Z, but cheaper.
    The formula is valid for the fixed overhead camera orientation used here.
    """
    return float(cam_pos[2] + cam_R[2, 2] * depth_value)


def estimate_object_from_depth(
    u: int,
    v: int,
    depth: np.ndarray,
    K: np.ndarray,
    cam_pos: np.ndarray | None = None,
    cam_R: np.ndarray | None = None,
    top_radius: int = 6,
    support_inner_radius: int = 12,
    support_outer_radius: int = 32,
    top_percentile: float = 5.0,
    support_percentile: float = 70.0,
    min_visible_height: float = 0.003,
) -> DepthObjectEstimate:
    """
    Estimate top Z, local support Z, and object height from the depth image.

    top_z:
        Uses the closest valid depth values in a small disk around the detected
        centroid.  For an overhead depth camera, closest means highest surface.

    support_z:
        Uses a surrounding annulus.  This is the local surface the object sits
        on: table, bowl rim, or another visible object.  No table/cube/bowl Z is
        assumed.

    Raises
    ------
    ValueError if the top/support cannot be measured or if the object has no
    measurable height above its local support.  This is intentional: no fallback
    to fixed heights.
    """
    if cam_pos is None or cam_R is None:
        cam_pos, cam_R = get_camera_extrinsics()

    H, W = depth.shape
    u, v = int(round(u)), int(round(v))
    if not (0 <= u < W and 0 <= v < H):
        raise ValueError(f"Pixel ({u}, {v}) is outside depth image {W}x{H}")

    y0, y1 = max(0, v - support_outer_radius), min(H, v + support_outer_radius + 1)
    x0, x1 = max(0, u - support_outer_radius), min(W, u + support_outer_radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    rr = np.sqrt((xx - u) ** 2 + (yy - v) ** 2)
    patch = depth[y0:y1, x0:x1]

    top_mask = rr <= top_radius
    support_mask = (rr >= support_inner_radius) & (rr <= support_outer_radius)

    top_vals = _valid_depth_values(patch[top_mask])
    support_vals = _valid_depth_values(patch[support_mask])

    if top_vals.size < 5:
        raise ValueError(f"Not enough valid top-depth pixels near ({u}, {v})")
    if support_vals.size < 20:
        raise ValueError(f"Not enough valid support-depth pixels around ({u}, {v})")

    # Smaller depth = closer to camera = higher world Z.
    top_depth = float(np.percentile(top_vals, top_percentile))

    # Larger depth in the annulus corresponds to the lower local support plane.
    # 70th percentile is more robust than max, which can hit background/noise.
    support_depth = float(np.percentile(support_vals, support_percentile))

    top_z = _world_z_from_depth_value(top_depth, cam_pos, cam_R)
    support_z = _world_z_from_depth_value(support_depth, cam_pos, cam_R)
    height = top_z - support_z

    if height < min_visible_height:
        raise ValueError(
            f"Measured height too small at ({u}, {v}): {height:.4f} m. "
            "The support plane may be hidden or the detection may be wrong."
        )

    top_xyz = pixel_to_world_at_depth(u, v, top_depth, K, cam_pos, cam_R)
    return DepthObjectEstimate(
        pixel=(u, v),
        top_xyz=top_xyz,
        support_z=float(support_z),
        top_z=float(top_z),
        height=float(height),
        top_depth=top_depth,
        support_depth=support_depth,
    )
