"""
sim_env.py - MuJoCo environment for VLA pick-and-place
"""

import os, re, tempfile, warnings
import numpy as np
import mujoco
import mujoco.viewer

CAM_W, CAM_H = 640, 480
CAM_FOVY     = 60.0
DT           = 0.002
IK_ITERS     = 300
HOVER_OFFSET = 0.12
EEF_SITE     = "eef_site"
HOME_QPOS    = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])


def _inject_eef_site(panda_xml_path: str) -> str:
    import pathlib as _pl
    with open(panda_xml_path, "r") as f:
        xml = f.read()
    # Make meshdir absolute so the patched XML can live in /tmp
    asset_dir = str(_pl.Path(panda_xml_path).parent / "assets")
    xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{asset_dir}"', xml)
    site_tag = f'\n      <site name="{EEF_SITE}" pos="0 0 0.105" size="0.005" rgba="1 0 0 1"/>'
    xml = re.sub(r'(<body name="hand"[^>]*>)', r'\1' + site_tag, xml, count=1)
    return xml


def _build_scene(panda_xml_path: str, tmpdir: str) -> str:
    asset_dir = os.path.dirname(panda_xml_path)
    panda_content = _inject_eef_site(panda_xml_path)
    panda_out = os.path.join(tmpdir, "panda_patched.xml")
    with open(panda_out, "w") as f:
        f.write(panda_content)

    scene = f"""
<mujoco model="pick_and_place">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{DT}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <default class="object">
      <geom condim="4" friction="1.0 0.005 0.0001" solimp="0.99 0.99 0.01" solref="0.01 1"/>
    </default>
  </default>
  <include file="panda_patched.xml"/>
  <worldbody>
    <light name="lt_top" pos="0.5 0 2.5" dir="0 0 -1" diffuse="0.8 0.8 0.8" castshadow="false"/>
    <light name="lt_side" pos="-0.5 0.5 1.5" dir="1 -0.5 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom name="vla_floor" type="plane" size="3 3 0.1" pos="0 0 0" rgba="0.6 0.6 0.6 1" contype="1" conaffinity="1"/>
    <body name="table" pos="0.5 0 0">
      <geom name="table_top" type="box" size="0.4 0.4 0.02" pos="0 0 0.4" rgba="0.55 0.37 0.18 1" contype="1" conaffinity="1"/>
      <geom type="cylinder" size="0.02 0.2" pos="-0.35 -0.35 0.2" rgba="0.45 0.3 0.15 1"/>
      <geom type="cylinder" size="0.02 0.2" pos=" 0.35 -0.35 0.2" rgba="0.45 0.3 0.15 1"/>
      <geom type="cylinder" size="0.02 0.2" pos="-0.35  0.35 0.2" rgba="0.45 0.3 0.15 1"/>
      <geom type="cylinder" size="0.02 0.2" pos=" 0.35  0.35 0.2" rgba="0.45 0.3 0.15 1"/>
    </body>
    <body name="cam_body" pos="0.5 0 1.72">
      <camera name="overhead_cam" fovy="{CAM_FOVY}" pos="0 0 0" euler="3.14159 0 0"/>
    </body>
    <body name="red_cube" pos="0.38 -0.10 0.445">
      <freejoint/>
      <geom class="object" type="box" size="0.025 0.025 0.025" rgba="0.9 0.1 0.1 1" mass="0.05"/>
      <site name="red_cube" size="0.002" rgba="1 0 0 0"/>
    </body>
    <body name="green_cube" pos="0.55 0.05 0.445">
      <freejoint/>
      <geom class="object" type="box" size="0.025 0.025 0.025" rgba="0.1 0.8 0.1 1" mass="0.05"/>
      <site name="green_cube" size="0.002" rgba="0 1 0 0"/>
    </body>
    <body name="yellow_cube" pos="0.62 -0.08 0.445">
      <freejoint/>
      <geom class="object" type="box" size="0.025 0.025 0.025" rgba="0.95 0.85 0.05 1" mass="0.05"/>
      <site name="yellow_cube" size="0.002" rgba="1 1 0 0"/>
    </body>
    <body name="blue_bowl" pos="0.50 0.20 0.422">
      <freejoint/>
      <geom class="object" type="cylinder" size="0.055 0.005" pos="0 0 0" rgba="0.1 0.2 0.95 1" mass="0.10"/>
      <geom class="object" type="cylinder" size="0.055 0.018" pos="0 0 0.013" rgba="0.1 0.2 0.95 0.25" mass="0.01" contype="0" conaffinity="0"/>
      <site name="blue_bowl" size="0.002" rgba="0 0 1 0"/>
    </body>
    <body name="red_bowl" pos="0.35 0.18 0.422">
      <freejoint/>
      <geom class="object" type="cylinder" size="0.055 0.005" pos="0 0 0" rgba="0.9 0.15 0.15 1" mass="0.10"/>
      <geom class="object" type="cylinder" size="0.055 0.018" pos="0 0 0.013" rgba="0.9 0.15 0.15 0.25" mass="0.01" contype="0" conaffinity="0"/>
      <site name="red_bowl" size="0.002" rgba="1 0 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""
    path = os.path.join(tmpdir, "scene.xml")
    with open(path, "w") as f:
        f.write(scene)
    return path


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

        scene_path   = _build_scene(panda_path, self._tmpdir)
        self.model   = mujoco.MjModel.from_xml_path(scene_path)
        self.data    = mujoco.MjData(self.model)
        print(f"[SimEnv] nq={self.model.nq} nu={self.model.nu} nsite={self.model.nsite}")

        # Find EEF site
        self._eef_id = -1
        for i in range(self.model.nsite):
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i) == EEF_SITE:
                self._eef_id = i
                break
        if self._eef_id < 0:
            warnings.warn("[SimEnv] EEF site not found; using hand body")
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
            self._viewer.cam.lookat[:] = [0.5, 0.0, 0.42]
            self._viewer.cam.distance  = 1.5
            self._viewer.cam.elevation = -30
            self._viewer.cam.azimuth   = 180

        print("[SimEnv] Ready.")

    def _randomise_objects(self):
        z_map = {"blue_bowl": 0.422, "red_bowl": 0.422}
        placed = []
        for name in ["red_cube", "green_cube", "yellow_cube", "blue_bowl", "red_bowl"]:
            z = z_map.get(name, 0.445)
            for _ in range(300):
                x = self._rng.uniform(0.22, 0.72)
                y = self._rng.uniform(-0.30, 0.30)
                pos = np.array([x, y, z])
                if all(np.linalg.norm(pos[:2] - p[:2]) > 0.13 for p in placed):
                    placed.append(pos); break
            try:
                jnt_id   = self.model.joint(name).id
                adr      = self.model.jnt_qposadr[jnt_id]
                self.data.qpos[adr:adr+3]   = pos
                self.data.qpos[adr+3:adr+7] = [1,0,0,0]
            except Exception:
                pass

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        n = min(len(HOME_QPOS), self.model.nq)
        self.data.qpos[:n] = HOME_QPOS[:n]
        n_ctrl = min(8, self.model.nu)
        self.data.ctrl[:n_ctrl] = HOME_QPOS[:n_ctrl]
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            try: self._viewer.close()
            except: pass
        self._renderer.close()

    # ── Camera API ──────────────────────────────────────────────────────────

    def get_camera_image(self):
        self._renderer.update_scene(self.data, camera="overhead_cam")
        rgb = self._renderer.render().copy()

        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera="overhead_cam")
        depth_raw = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()

        extent = self.model.stat.extent
        znear  = self.model.vis.map.znear * extent
        zfar   = self.model.vis.map.zfar  * extent
        denom  = zfar - depth_raw * (zfar - znear)
        denom  = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        depth  = (znear * zfar / denom).astype(np.float32)

        K = self._intrinsics()
        return rgb, depth, K

    def _intrinsics(self):
        fovy_rad = np.deg2rad(CAM_FOVY)
        fy = (CAM_H / 2.0) / np.tan(fovy_rad / 2.0)
        fx = fy * CAM_W / CAM_H
        return np.array([[fx, 0, CAM_W/2], [0, fy, CAM_H/2], [0, 0, 1]], dtype=np.float64)

    # ── Robot Control API ───────────────────────────────────────────────────

    def move_to_pose(self, x, y, z, roll=0., pitch=0., yaw=0., **kw):
        import mink
        from scipy.spatial.transform import Rotation

        configuration = mink.Configuration(self.model)
        configuration.update(self.data.qpos)

        frame_type = "site" if self._eef_id >= 0 else "body"
        frame_name = EEF_SITE if self._eef_id >= 0 else "hand"

        eef_task = mink.FrameTask(
            frame_name=frame_name, frame_type=frame_type,
            position_cost=1.0, orientation_cost=0.3,
        )
        R_mat  = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
        target = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(R_mat), np.array([x, y, z])
        )
        eef_task.set_target(target)
        limits = [mink.ConfigurationLimit(self.model)]

        dt = DT * 20
        converged = False
        for _ in range(IK_ITERS):
            configuration.update(self.data.qpos)
            try:
                # vel = mink.solve_ik(configuration, [eef_task], dt, limits=limits, damping=1e-3)
                vel = mink.solve_ik(configuration, [eef_task], dt, solver="quadprog", limits=limits, damping=1e-3)
            except Exception as e:
                warnings.warn(f"[SimEnv] IK error: {e}"); break
            configuration.integrate_inplace(vel, dt)
            self.data.ctrl[:7] = configuration.q[:7]
            self.step(10)
            pos_err = np.linalg.norm(eef_task.compute_error(configuration)[:3])
            if pos_err < 8e-3:
                converged = True; break
        return converged

    def set_gripper(self, open: bool):
        pos = 0.04 if open else 0.001
        for i in range(7, self.model.nu):
            self.data.ctrl[i] = pos
        self.step(200)

    def get_eef_position(self):
        if self._eef_id >= 0:
            return self.data.site_xpos[self._eef_id].copy()
        return self.data.xpos[self._hand_id].copy()

    def get_object_position(self, name: str):
        return self.data.xpos[self.model.body(name).id].copy()