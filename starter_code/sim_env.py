"""
sim_env.py
==========
MuJoCo simulation environment for VLA pick-and-place.

Controller design
-----------------
MuJoCo position actuators with explicit kp/kd implement PD control:
    tau = kp * (q_des - q) - kd * qd

We use mink differential IK to compute desired joint angles (q_des)
at each control step, then write them to data.ctrl. The Panda actuators
in mujoco_menagerie are already configured as position controllers, so
writing to data.ctrl IS the PD reference signal.

Two IK modes
------------
move_to_pose()      — full convergence (300 IK iters), for waypoint stops
move_to_pose_fast() — light (20 IK iters), for dense Cartesian trajectory
                      sampling where we care about smooth motion, not exact
                      convergence at every point
"""

import os, re, tempfile, warnings
import numpy as np
import mujoco
import mujoco.viewer

# ── Camera ────────────────────────────────────────────────────────────────────
CAM_W, CAM_H = 640, 480
CAM_FOVY     = 60.0

# ── Simulation ────────────────────────────────────────────────────────────────
DT           = 0.002          # MuJoCo timestep
EEF_SITE     = "eef_site"     # site injected into Panda hand body

# ── Home & photo poses ────────────────────────────────────────────────────────
# From panda.xml keyframe: qpos and ctrl for home pose
HOME_QPOS  = np.array([0, 0, 0, -1.5708, 0, 1.5708, -0.7853, 0.04, 0.04])
HOME_CTRL  = np.array([0, 0, 0, -1.5708, 0, 1.5708, -0.7853, 255.0])
# ctrl[7]=255 = fingers OPEN (ctrlrange 0-255, gainprm=0.0157 → 0.04m)
# ctrl[7]=0   = fingers CLOSED

# Arm folded backward — verified: no links above table surface
PHOTO_QPOS = np.array([0, 0, 0, 0, 0, 1.57, 0, 0, 0])

# ── IK parameters ────────────────────────────────────────────────────────────
IK_DT        = 0.002          # IK integration step = 1 physics step
IK_ITERS     = 2000           # iterations for move_to_pose (waypoint stops)
IK_ITERS_FAST = 100           # iterations for move_to_pose_fast (trajectory)
IK_DAMPING   = 1e-3           # slightly higher damping for stability
IK_TOL       = 8e-3           # convergence tolerance (m)
IK_PHYS_PER_STEP = 1          # 1 physics step per IK iter — tightest coupling

# Panda velocity limits (rad/s) from spec sheet
PANDA_VEL_LIMITS = {
    "joint1": 2.175, "joint2": 2.175, "joint3": 2.175, "joint4": 2.175,
    "joint5": 2.610, "joint6": 2.610, "joint7": 2.610,
}

HOVER_OFFSET = 0.12   # kept for backward compat


# ─────────────────────────────────────────────────────────────────────────────
# Scene building
# ─────────────────────────────────────────────────────────────────────────────

def _inject_eef_site(panda_xml_path: str) -> str:
    """Inject EEF site into hand body and fix meshdir to absolute path."""
    import pathlib as _pl
    with open(panda_xml_path) as f:
        xml = f.read()
    asset_dir = str(_pl.Path(panda_xml_path).parent / "assets")
    xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{asset_dir}"', xml)
    site_tag = (f'\n      <site name="{EEF_SITE}" pos="0 0 0.105" '
                f'size="0.005" rgba="1 0 0 1"/>')
    xml = re.sub(r'(<body name="hand"[^>]*>)', r'\1' + site_tag, xml, count=1)
    return xml


def _build_scene(panda_xml_path: str, tmpdir: str) -> str:
    panda_content = _inject_eef_site(panda_xml_path)
    panda_out = os.path.join(tmpdir, "panda_patched.xml")
    with open(panda_out, "w") as f:
        f.write(panda_content)

    scene = f"""
<mujoco model="pick_and_place">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{DT}" gravity="0 0 -9.81" integrator="implicitfast"/>

  <visual>
    <headlight diffuse="0.3 0.3 0.3" ambient="0.4 0.4 0.4" specular="0 0 0"/>
    <rgba haze="0 0 0 0"/>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <default>
    <default class="object">
      <geom condim="4" friction="1.0 0.005 0.0001"
            solimp="0.99 0.99 0.01" solref="0.01 1"/>
    </default>
  </default>

  <include file="panda_patched.xml"/>

  <worldbody>
    <light name="lt1" pos="0.3 -0.3 1.8" dir=" 0.2  0.2 -1"
           diffuse="0.45 0.45 0.45" specular="0 0 0" castshadow="false"/>
    <light name="lt2" pos="0.7  0.3 1.8" dir="-0.2 -0.2 -1"
           diffuse="0.45 0.45 0.45" specular="0 0 0" castshadow="false"/>
    <light name="lt3" pos="0.5  0.0 0.9" dir="0    0   -1"
           diffuse="0.20 0.20 0.20" specular="0 0 0" castshadow="false"/>

    <geom name="vla_floor" type="plane" size="3 3 0.1" pos="0 0 0"
          rgba="0.6 0.6 0.6 1" contype="1" conaffinity="1"/>

    <body name="table" pos="0.5 0 0">
      <geom name="table_top" type="box" size="0.4 0.4 0.02"
            pos="0 0 0.4" rgba="0.55 0.37 0.18 1"
            contype="1" conaffinity="1"/>
      <geom type="cylinder" size="0.02 0.2" pos="-0.35 -0.35 0.2" rgba="0.45 0.3 0.15 1"/>
      <geom type="cylinder" size="0.02 0.2" pos=" 0.35 -0.35 0.2" rgba="0.45 0.3 0.15 1"/>
      <geom type="cylinder" size="0.02 0.2" pos="-0.35  0.35 0.2" rgba="0.45 0.3 0.15 1"/>
      <geom type="cylinder" size="0.02 0.2" pos=" 0.35  0.35 0.2" rgba="0.45 0.3 0.15 1"/>
    </body>

    <!-- Overhead camera: euler=0 → optical axis points straight down -->
    <body name="cam_body" pos="0.5 0 1.72">
      <camera name="overhead_cam" fovy="{CAM_FOVY}" pos="0 0 0" euler="0 0 0"/>
    </body>

    <body name="red_cube"    pos="0.38 -0.10 0.445">
      <freejoint/>
      <geom class="object" type="box" size="0.025 0.025 0.025"
            rgba="0.9 0.1 0.1 1" mass="0.05"/>
      <site name="red_cube"    size="0.002" rgba="1 0 0 0"/>
    </body>
    <body name="green_cube"  pos="0.55  0.05 0.445">
      <freejoint/>
      <geom class="object" type="box" size="0.025 0.025 0.025"
            rgba="0.1 0.8 0.1 1" mass="0.05"/>
      <site name="green_cube"  size="0.002" rgba="0 1 0 0"/>
    </body>
    <body name="yellow_cube" pos="0.62 -0.08 0.445">
      <freejoint/>
      <geom class="object" type="box" size="0.025 0.025 0.025"
            rgba="0.95 0.85 0.05 1" mass="0.05"/>
      <site name="yellow_cube" size="0.002" rgba="1 1 0 0"/>
    </body>
    <body name="blue_bowl"   pos="0.50  0.20 0.422">
      <freejoint/>
      <geom class="object" type="cylinder" size="0.055 0.005"
            pos="0 0 0"     rgba="0.1 0.2 0.95 1" mass="0.10"/>
      <geom class="object" type="cylinder" size="0.055 0.018"
            pos="0 0 0.013" rgba="0.1 0.2 0.95 0.25" mass="0.01"
            contype="0" conaffinity="0"/>
      <site name="blue_bowl"   size="0.002" rgba="0 0 1 0"/>
    </body>
    <body name="red_bowl"    pos="0.35  0.18 0.422">
      <freejoint/>
      <geom class="object" type="cylinder" size="0.055 0.005"
            pos="0 0 0"     rgba="0.9 0.15 0.15 1" mass="0.10"/>
      <geom class="object" type="cylinder" size="0.055 0.018"
            pos="0 0 0.013" rgba="0.9 0.15 0.15 0.25" mass="0.01"
            contype="0" conaffinity="0"/>
      <site name="red_bowl"    size="0.002" rgba="1 0 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""
    path = os.path.join(tmpdir, "scene.xml")
    with open(path, "w") as f:
        f.write(scene)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# SimEnv
# ─────────────────────────────────────────────────────────────────────────────

class SimEnv:
    def __init__(self, render: bool = True, random_seed=None):
        self.render  = render
        self._rng    = np.random.default_rng(random_seed)
        self._tmpdir = tempfile.mkdtemp()
        self._viewer = None

        try:
            from robot_descriptions import panda_mj_description
            panda_path = panda_mj_description.MJCF_PATH
        except ImportError:
            raise ImportError("Run: pip install robot_descriptions")

        scene_path = _build_scene(panda_path, self._tmpdir)
        self.model  = mujoco.MjModel.from_xml_path(scene_path)
        self.data   = mujoco.MjData(self.model)
        print(f"[SimEnv] nq={self.model.nq} nu={self.model.nu} "
              f"nsite={self.model.nsite}")

        # Find EEF site
        self._eef_id = -1
        for i in range(self.model.nsite):
            if mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_SITE, i) == EEF_SITE:
                self._eef_id = i
                break
        if self._eef_id < 0:
            warnings.warn("[SimEnv] EEF site not found — using hand body")
            self._hand_id = self.model.body("hand").id
        else:
            self._hand_id = None
            print(f"[SimEnv] EEF site found (id={self._eef_id})")

        if random_seed is not None:
            self._randomise_objects()

        self.reset()
        self._renderer = mujoco.Renderer(self.model, height=CAM_H, width=CAM_W)

        if render:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.cam.lookat[:] = [0.5, 0.05, 0.45]
            self._viewer.cam.distance  = 1.4
            self._viewer.cam.elevation = -28
            self._viewer.cam.azimuth   = 170

        # Pre-build mink limits (reused across IK calls)
        self._setup_ik()
        print("[SimEnv] Ready.")

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_ik(self):
        """Pre-build mink IK objects reused across calls."""
        import mink
        self._ik_cfg    = mink.Configuration(self.model)
        # Use only ConfigurationLimit — VelocityLimit was capping movement to
        # 0.00435 rad/step, causing arm to take 1000+ steps per waypoint
        self._ik_limits = [
            mink.ConfigurationLimit(self.model),
        ]
        # Top-down SO3: site z-axis = [0,0,-1] (pointing down).
        # Matches EEF site orientation at home: xmat ≈ diag(1,-1,-1)
        R_down = np.array([[1., 0., 0.],
                           [0.,-1., 0.],
                           [0., 0.,-1.]])
        self._R_down = mink.SO3.from_matrix(R_down)

    # ── Reset ─────────────────────────────────────────────────────────────────

    # Verified robot workspace bounds (IK reachable at grasp height z≈0.47)
    WORKSPACE_X = (0.26, 0.64)
    WORKSPACE_Y = (-0.28, 0.28)
    MIN_OBJECT_SEPARATION = 0.13   # metres, prevents objects overlapping

    def _randomise_objects(self):
        """
        Place objects randomly within the verified robot workspace.
        X=[0.26, 0.64], Y=[-0.28, 0.28] — IK-reachable region on table top.
        Uses rejection sampling to ensure minimum separation between objects.
        """
        z_map = {"blue_bowl": 0.422, "red_bowl": 0.422}
        placed = []
        for name in ["red_cube", "green_cube", "yellow_cube", "blue_bowl", "red_bowl"]:
            z = z_map.get(name, 0.445)
            for _ in range(500):   # rejection sampling attempts
                x = self._rng.uniform(*self.WORKSPACE_X)
                y = self._rng.uniform(*self.WORKSPACE_Y)
                pos = np.array([x, y, z])
                if all(np.linalg.norm(pos[:2] - p[:2]) > self.MIN_OBJECT_SEPARATION
                       for p in placed):
                    placed.append(pos)
                    break
            try:
                adr = self.model.jnt_qposadr[self.model.joint(name).id]
                self.data.qpos[adr:adr+3]   = pos
                self.data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
            except Exception:
                pass

    def reset(self):
        """Teleport arm to home and step physics to settle."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:9] = HOME_QPOS
        self.data.ctrl[:8]  = HOME_CTRL   # ctrl[7]=255 = fingers open
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # 1000 steps gives PD actuators enough time to drive arm to home from any pose
        for _ in range(1000):
            mujoco.mj_step(self.model, self.data)
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    # ── Stepping ─────────────────────────────────────────────────────────────

    def step(self, n: int = 1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
        try:
            self._renderer.close()
        except Exception:
            pass

    # ── Photo pose ────────────────────────────────────────────────────────────

    def return_home_ik(self):
        """
        Return arm to home configuration using IK + PD control (not teleport).
        Drives each joint toward HOME_QPOS by setting ctrl and stepping physics.
        This is the proper motion-controlled home return — no discontinuous jump.
        """
        print("[SimEnv] Returning to home via IK+control ...")
        # Set ctrl to home and step until arm converges
        self.data.ctrl[:7]  = HOME_CTRL[:7]   # arm joint targets
        self.data.ctrl[7]   = 255.0            # gripper open
        converged = False
        for i in range(3000):
            mujoco.mj_step(self.model, self.data)
            if self._viewer is not None and i % 50 == 0 and self._viewer.is_running():
                self._viewer.sync()
            # Check convergence: all arm joints within 10mrad of home
            q_err = np.abs(self.data.qpos[:7] - HOME_QPOS[:7])
            if q_err.max() < 0.01:
                converged = True
                break
        if converged:
            print(f"[SimEnv] Home reached (max_joint_err={q_err.max()*1000:.1f}mrad).")
        else:
            print(f"[SimEnv] Home: max_joint_err={q_err.max()*1000:.1f}mrad (not fully converged).")
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def move_to_photo_pose(self):
        """Retract arm to photo pose — no links above table."""
        self.data.qpos[:9] = PHOTO_QPOS
        self.data.ctrl[:7]  = PHOTO_QPOS[:7]   # arm joints
        self.data.ctrl[7]   = 255.0             # fingers open
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(1000):
            mujoco.mj_step(self.model, self.data)
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()
        print("[SimEnv] Arm at photo pose (clear of table).")

    # ── Camera API ────────────────────────────────────────────────────────────

    def get_camera_image(self):
        self._renderer.update_scene(self.data, camera="overhead_cam")
        rgb = self._renderer.render().copy()

        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera="overhead_cam")
        depth_raw = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()

        # MuJoCo >= 3.x Renderer returns metric depth directly (metres from camera).
        # No z-buffer conversion needed — raw values are already in metres.
        depth = depth_raw.astype(np.float32)

        K = self._intrinsics()
        return rgb, depth, K

    def _intrinsics(self):
        fovy_rad = np.deg2rad(CAM_FOVY)
        fy = (CAM_H / 2.0) / np.tan(fovy_rad / 2.0)
        fx = fy * CAM_W / CAM_H
        return np.array([[fx,0,CAM_W/2],[0,fy,CAM_H/2],[0,0,1]],
                        dtype=np.float64)

    # ── IK core ───────────────────────────────────────────────────────────────

    def _ik_step(
        self,
        x: float, y: float, z: float,
        max_iters: int,
    ) -> bool:
        """
        Run mink IK for up to max_iters steps targeting (x,y,z).

        PD control: mink computes q_des, written to data.ctrl[:7].
        MuJoCo position actuators implement tau=kp(q_des-q)+kd(0-qd).

        Orientation: identity SO3 target with small orientation_cost
        enforces top-down (roll=pitch=yaw=0) throughout.
        PostureTask pulls null-space toward home, keeping wrist stable.
        """
        import mink

        cfg = self._ik_cfg
        cfg.update(self.data.qpos)

        frame_name = EEF_SITE if self._eef_id >= 0 else "hand"
        frame_type = "site"  if self._eef_id >= 0 else "body"

        # EEF task: position + light orientation cost to maintain top-down
        eef_task = mink.FrameTask(
            frame_name=frame_name,
            frame_type=frame_type,
            position_cost=1.0,
            orientation_cost=0.1,   # small: keeps top-down without over-constraining
        )
        target = mink.SE3.from_rotation_and_translation(
            self._R_down, np.array([x, y, z])
        )
        eef_task.set_target(target)

        # Posture task: bias null-space toward home (stabilises wrist/orientation)
        # Must provide full nq-length vector; pad with current object positions
        posture_task = mink.PostureTask(self.model, cost=0.005)
        full_home = self.data.qpos.copy()   # keeps object freejoint positions
        full_home[:9] = HOME_QPOS           # overwrite only arm+finger joints
        posture_task.set_target(full_home)

        tasks = [eef_task, posture_task]
        converged = False

        # Strategy: sync cfg from actual qpos every step so IK always starts
        # from where the arm actually is. This prevents cfg from drifting ahead.
        # The arm tracks cfg.q via PD. With kp=4500 and dt=0.002 the arm moves
        # ~kp*dt^2 per step — fast enough to converge in ~200-400 steps.

        for i in range(max_iters):
            # Always sync from actual arm state
            cfg.update(self.data.qpos)

            try:
                vel = mink.solve_ik(
                    cfg, tasks, IK_DT, solver="daqp",
                    limits=self._ik_limits, damping=IK_DAMPING,
                )
            except Exception as e:
                warnings.warn(f"[SimEnv] IK: {e}")
                break

            # Integrate ONE step to get desired joint angles
            cfg.integrate_inplace(vel, IK_DT)

            # PD reference: write one-step-ahead desired joints
            self.data.ctrl[:7] = cfg.q[:7]
            self.step(IK_PHYS_PER_STEP)

            # Check convergence on actual EEF position
            if self._eef_id >= 0:
                actual_eef = self.data.site_xpos[self._eef_id]
                actual_err = np.linalg.norm(actual_eef - np.array([x, y, z]))
            else:
                actual_err = np.linalg.norm(
                    self.data.xpos[self._hand_id] - np.array([x, y, z]))

            if actual_err < IK_TOL:
                converged = True
                break

        return converged

    # ── Robot Control API ─────────────────────────────────────────────────────

    def move_to_pose(self, x, y, z, roll=0., pitch=0., yaw=0., **kw):
        """
        Move EEF to (x,y,z) with full convergence (IK_ITERS iterations).
        roll/pitch/yaw must be 0 — top-down grasp enforced.
        Returns True if converged within IK_TOL.
        """
        ok = self._ik_step(x, y, z, max_iters=IK_ITERS)
        if not ok:
            # Extra settle with current ctrl
            self.step(100)
        return ok

    def move_to_pose_fast(self, x, y, z, roll=0., pitch=0., yaw=0., **kw):
        """
        Light IK step (IK_ITERS_FAST iterations) for dense trajectory tracking.
        Prioritises smooth motion over exact convergence at each point.
        """
        self._ik_step(x, y, z, max_iters=IK_ITERS_FAST)

    def set_gripper(self, open: bool):
        """
        Open or close Panda fingers.
        actuator8 uses ctrlrange=[0,255]:
          255 → fingers open  (0.04 m each)
          0   → fingers closed
        Driven via 'split' tendon coupling both finger joints equally.
        """
        ctrl_val = 255.0 if open else 0.0
        self.data.ctrl[7] = ctrl_val
        self.step(300)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_object_z_from_depth(
        self,
        u: int, v: int,
        depth: np.ndarray,
        patch_radius: int = 7,
    ) -> float:
        """
        Return world-frame Z of the object top surface from the depth map.

        Uses minimum depth in a patch (closest point = top of object).
        depth is metric (metres from camera, MuJoCo 3.x format).

        world_z = cam_z - d_min  (camera looks straight down)
        """
        from projection import get_camera_extrinsics
        cam_pos, _ = get_camera_extrinsics()
        H, W = depth.shape
        v0, v1 = max(0, v-patch_radius), min(H, v+patch_radius+1)
        u0, u1 = max(0, u-patch_radius), min(W, u+patch_radius+1)
        patch = depth[v0:v1, u0:u1]
        # Closest point = minimum depth value = top surface of object
        d_min = float(patch[patch > 0].min()) if (patch > 0).any() else float(depth[v, u])
        world_z = cam_pos[2] - d_min
        return world_z

    def get_eef_position(self) -> np.ndarray:
        if self._eef_id >= 0:
            return self.data.site_xpos[self._eef_id].copy()
        return self.data.xpos[self._hand_id].copy()

    def get_object_position(self, name: str) -> np.ndarray:
        return self.data.xpos[self.model.body(name).id].copy()