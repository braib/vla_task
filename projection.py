"""
projection.py
=============
2D pixel → 3D world coordinate back-projection.

Uses the standard pinhole camera model:

    Z = depth[v, u]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

The result is in the camera frame. We then apply the fixed camera-to-world
transform (known from the MJCF: overhead cam at [0.5, 0, 1.72], looking
straight down) to get world-frame coordinates.

Public API
----------
    pixel_to_world(
        u, v        : int     — pixel coordinates (col, row)
        depth       : ndarray — (H, W) metric depth map in metres
        K           : ndarray — (3, 3) intrinsic matrix
        cam_pos     : ndarray — (3,) camera world position
        cam_R       : ndarray — (3, 3) camera rotation (cols = x,y,z axes in world)
    ) -> np.ndarray (3,)  world XYZ in metres

    get_camera_extrinsics() -> (cam_pos, cam_R)
        Returns the fixed overhead camera pose derived from the scene XML.

    depth_at_pixel(depth, u, v, kernel=5) -> float
        Returns a robust (median-filtered) depth estimate at pixel (u,v).
"""

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Fixed camera pose (matches scene XML / SimEnv)
# ─────────────────────────────────────────────────────────────────────────────

# Overhead camera body position in world frame (metres)
_CAM_WORLD_POS = np.array([0.5, 0.0, 1.72])

# Overhead camera looks straight down → euler (π, 0, 0) in MuJoCo convention.
# In MuJoCo, euler="π 0 0" means Rx(π): camera Z-axis points world -Z,
# camera X = world X, camera Y = world -Y.
#
# The resulting rotation matrix R_cam_to_world:
#   world_pt = cam_pos + R @ cam_pt
_CAM_R = np.array([
    [ 1,  0,  0],   # camera X  → world +X
    [ 0, -1,  0],   # camera Y  → world -Y  (image rows go down = world -Y)
    [ 0,  0, -1],   # camera Z  → world -Z  (optical axis points down)
], dtype=np.float64)


def get_camera_extrinsics() -> tuple[np.ndarray, np.ndarray]:
    """
    Return (cam_pos, cam_R) for the fixed overhead camera.

    cam_pos : (3,) world position of camera optical centre.
    cam_R   : (3, 3) rotation s.t. world_pt = cam_pos + cam_R @ cam_pt
    """
    return _CAM_WORLD_POS.copy(), _CAM_R.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Core math
# ─────────────────────────────────────────────────────────────────────────────

def depth_at_pixel(
    depth: np.ndarray,
    u: int,
    v: int,
    kernel: int = 7,
) -> float:
    """
    Return a robust depth estimate at pixel (u, v) using a small median filter.

    A kernel window avoids noise from a single bad pixel.

    Parameters
    ----------
    depth  : (H, W) float32 metric depth map
    u      : column (x-axis in image)
    v      : row    (y-axis in image)
    kernel : window half-size (total = 2*kernel+1)

    Returns
    -------
    median depth in metres (float)
    """
    H, W = depth.shape
    r = kernel // 2
    v0, v1 = max(0, v - r), min(H, v + r + 1)
    u0, u1 = max(0, u - r), min(W, u + r + 1)
    patch = depth[v0:v1, u0:u1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return float(depth[v, u])
    return float(np.median(valid))


def pixel_to_camera(
    u: int,
    v: int,
    Z: float,
    K: np.ndarray,
) -> np.ndarray:
    """
    Back-project a pixel (u, v) at depth Z into the camera frame.

    Parameters
    ----------
    u, v : pixel column and row
    Z    : metric depth (metres)
    K    : (3, 3) pinhole intrinsic matrix

    Returns
    -------
    (3,) point in camera frame [X_cam, Y_cam, Z_cam]
    """
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
    """
    Transform a point from camera frame to world frame.

    world_pt = cam_pos + cam_R @ p_cam

    Parameters
    ----------
    p_cam   : (3,) point in camera frame
    cam_pos : (3,) camera world position
    cam_R   : (3, 3) camera rotation matrix (cam → world)

    Returns
    -------
    (3,) point in world frame
    """
    return cam_pos + cam_R @ p_cam


def pixel_to_world(
    u: int,
    v: int,
    depth: np.ndarray,
    K: np.ndarray,
    cam_pos: np.ndarray | None = None,
    cam_R: np.ndarray   | None = None,
    depth_kernel: int = 7,
) -> np.ndarray:
    """
    Full pipeline: pixel (u, v) → world XYZ.

    Parameters
    ----------
    u, v         : pixel column and row
    depth        : (H, W) float32 metric depth map
    K            : (3, 3) pinhole intrinsics
    cam_pos      : (3,) camera world position. If None, uses fixed overhead cam.
    cam_R        : (3, 3) cam→world rotation.  If None, uses fixed overhead cam.
    depth_kernel : window size for robust depth estimate

    Returns
    -------
    (3,) world XYZ in metres
    """
    if cam_pos is None or cam_R is None:
        cam_pos, cam_R = get_camera_extrinsics()

    Z       = depth_at_pixel(depth, u, v, depth_kernel)
    p_cam   = pixel_to_camera(u, v, Z, K)
    p_world = camera_to_world(p_cam, cam_pos, cam_R)

    return p_world


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for the robot
# ─────────────────────────────────────────────────────────────────────────────

def project_to_table(
    world_xyz: np.ndarray,
    table_z: float = 0.42,
) -> np.ndarray:
    """
    Override the Z coordinate with the known table surface height.
    Useful when depth is noisy and we know objects rest on the table.

    Parameters
    ----------
    world_xyz : (3,)
    table_z   : height of table surface in world frame (metres)

    Returns
    -------
    (3,) with z replaced by table_z
    """
    out    = world_xyz.copy()
    out[2] = table_z
    return out