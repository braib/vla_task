import numpy as np
import pybullet as pb
import pybullet_data as pd
import time
import math


IMG_W, IMG_H = 640, 480
FOV          = 60          # degrees
NEAR, FAR    = 0.01, 10.0

CAM_EYE    = [0.0,  0.0,  3.0]   # position in world
CAM_TARGET = [0.0,  0.0,  0.0]   # looking at table center
CAM_UP     = [0.0,  1.0,  0.0]   # up vector

OBJECTS = [
    {"id": 1, "name": "red_cube",  "color": [1, 0, 0, 1], "shape": "box",  "length": 0.04, "width": 0.04, "height": 0.04 },
    {"id": 2, "name": "blue_bowl", "color": [0, 0, 1, 1], "shape": "bowl", "radius": 0.2, "height": 0.04 },
]

MAX_ARM_REACH = 0.8  # 0.89 m # 855 mm 
MIN_ARM_REACH = 0.2  # 0.15 m

TABLE_LENGTH = 3.0
TABLE_WIDTH  = 2.5
TABLE_HEIGHT = 0.6

PLANE_GROUND = ["plane.urdf",              [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
ROBOTIC_ARM  = ["franka_panda/panda.urdf", [0.0, 0.0, TABLE_HEIGHT], [0.0, 0.0, 0.0, 1.0]]

def sample_workspace_pose(max_reach= 0.8, min_reach=0.2, seed=None):

    rng = np.random.default_rng(seed)
    
    r_min_sq = min_reach ** 2
    r_max_sq = max_reach ** 2
    
    r  = np.sqrt(rng.uniform(r_min_sq, r_max_sq))
    theta = rng.uniform(0, 2 * np.pi)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return x, y

class SimEnv:
    def __init__(self, gui=True):
        self.gui = gui
        self.object_ids   = {}

        self.random_seed = 44
        np.random.seed(self.random_seed)

        self._connect()
        self._load_scene()
        self._spawn_robot()
        self._compute_intrinsics()
 
    def _connect(self):
        mode = pb.GUI if self.gui else pb.DIRECT
        self.client = pb.connect(mode)
        pb.resetSimulation()
        pb.setAdditionalSearchPath(pd.getDataPath())
        pb.setGravity(0, 0, -9.81)
        pb.setRealTimeSimulation(0)

    def _load_scene(self):
        self.ground_plane = pb.loadURDF(PLANE_GROUND[0], PLANE_GROUND[1], PLANE_GROUND[2])
        table_col = pb.createCollisionShape(
            pb.GEOM_BOX,
            halfExtents=[TABLE_LENGTH / 2, TABLE_WIDTH / 2, TABLE_HEIGHT / 2]
        )
        table_vis = pb.createVisualShape(
            pb.GEOM_BOX,
            halfExtents=[TABLE_LENGTH / 2, TABLE_WIDTH / 2, TABLE_HEIGHT / 2],
            rgbaColor=[0.7, 0.5, 0.3, 1]
        )
        self.table_id = pb.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_col,
            baseVisualShapeIndex=table_vis,
            basePosition=[0.0, 0.0, TABLE_HEIGHT / 2]
        )
        self._spawn_objects()
    
    def _spawn_objects(self):
        table_top_z = TABLE_HEIGHT
        quat = [0, 0, 0, 1]

        for obj in OBJECTS:
            if obj["shape"] == "box":
                fos = max(obj["length"], obj["width"])
                x, y = sample_workspace_pose(
                    max_reach=MAX_ARM_REACH - fos,
                    min_reach=max(MIN_ARM_REACH + fos, 0.0),
                    seed=self.random_seed + obj["id"]
                )
                col_id = pb.createCollisionShape(
                    pb.GEOM_BOX,
                    halfExtents=[obj["length"] / 2, obj["width"] / 2, obj["height"] / 2]
                )
                vis_id = pb.createVisualShape(
                    pb.GEOM_BOX,
                    halfExtents=[obj["length"] / 2, obj["width"] / 2, obj["height"] / 2],
                    rgbaColor=obj["color"]
                )
                body_id = pb.createMultiBody(
                    baseMass=0.1,
                    baseCollisionShapeIndex=col_id,
                    baseVisualShapeIndex=vis_id,
                    basePosition=[x, y, table_top_z + obj["height"] / 2],
                    baseOrientation=quat
                )

            elif obj["shape"] == "bowl":
                fos = obj["radius"]
                x, y = sample_workspace_pose(
                    max_reach=MAX_ARM_REACH - fos,
                    min_reach=max(MIN_ARM_REACH + fos, 0.0),
                    seed=self.random_seed + obj["id"]
                )
                col_id = pb.createCollisionShape(
                    pb.GEOM_CYLINDER, radius=obj["radius"], height=obj["height"]
                )
                vis_id = pb.createVisualShape(
                    pb.GEOM_CYLINDER, radius=obj["radius"], length=obj["height"],
                    rgbaColor=obj["color"]
                )
                body_id = pb.createMultiBody(
                    baseMass=0.1,
                    baseCollisionShapeIndex=col_id,
                    baseVisualShapeIndex=vis_id,
                    basePosition=[x, y, table_top_z + obj["height"] / 2],
                    baseOrientation=quat
                )
            else:
                continue

            self.object_ids[obj["name"]] = body_id
            for _ in range(100):
                pb.stepSimulation()

    def _spawn_robot(self):
        self.robotic_arm     = pb.loadURDF(ROBOTIC_ARM[0], ROBOTIC_ARM[1], ROBOTIC_ARM[2], useFixedBase=True)
        self.num_joints      = pb.getNumJoints(self.robotic_arm)

        self.end_effector_id = None
        for i in range(self.num_joints):
            info = pb.getJointInfo(self.robotic_arm, i)
            if info[12].decode() == "panda_hand":
                self.end_effector_id = i
                break
        if self.end_effector_id is None:
            self.end_effector_id = self.num_joints - 1  # fallback
        
        print("Endeffector Id: ", self.end_effector_id)

    def _compute_intrinsics(self):
        aspect           = IMG_W / IMG_H
        fov_v_rad        = 2 * math.atan(math.tan(math.radians(FOV / 2.0)) / aspect)
        fx               = (IMG_W / 2.0) / math.tan(math.radians(FOV / 2.0))
        fy               = (IMG_H / 2.0) / math.tan(fov_v_rad / 2.0)
        self.K = np.array([
            [fx,  0,  IMG_W / 2.0],
            [ 0, fy,  IMG_H / 2.0],
            [ 0,  0,  1          ]
        ], dtype=np.float64)


    def get_camera_data(self):

        view_matrix = pb.computeViewMatrix(CAM_EYE, CAM_TARGET, CAM_UP)
        proj_matrix = pb.computeProjectionMatrixFOV(
            fov=FOV, aspect=IMG_W / IMG_H,
            nearVal=NEAR, farVal=FAR
        )
        _, _, rgba, depth_buf, _ = pb.getCameraImage(
            width=IMG_W, height=IMG_H,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pb.ER_TINY_RENDERER
        )

        # Convert RGBA → RGB
        rgb = np.array(rgba, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[:, :, :3] # to drop alpha #  it controls transparency

        # Linearise depth buffer → real metric depth
        depth_raw = np.array(depth_buf, dtype=np.float32).reshape(IMG_H, IMG_W)
        depth = FAR * NEAR / (FAR - (FAR - NEAR) * depth_raw)

        return rgb, depth, self.K

    def move_to_pose(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        target_pos = [x, y, z]
        target_orn = pb.getQuaternionFromEuler([roll, pitch, yaw])

        joint_poses = pb.calculateInverseKinematics(
            self.robotic_arm,        # was self.robot_id
            self.end_effector_id,    # was self.ee_index
            target_pos,
            target_orn,
            maxNumIterations=200,
            residualThreshold=1e-5
        )

        for i in range(min(len(joint_poses), self.num_joints)):
            pb.setJointMotorControl2(
                bodyIndex=self.robotic_arm,
                jointIndex=i,
                controlMode=pb.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=500
            )

        for _ in range(240):
            pb.stepSimulation()
            if self.gui:
                time.sleep(1.0 / 240.0)

    def set_gripper(self, is_open: bool):
        state = "OPEN" if is_open else "CLOSED" 
        print(f"[Gripper] {state}")
        if not is_open:
            self._attach_object()
        else:
            self._detach_object()

        for _ in range(60):
            pb.stepSimulation()
            if self.gui:
                time.sleep(1.0 / 240.0)

    def _attach_object(self):
        link_state = pb.getLinkState(self.robotic_arm, self.end_effector_id)
        ee_pos = link_state[4]
        best_id, best_dist = None, float("inf")
        for name, body_id in self.object_ids.items():
            pos, _ = pb.getBasePositionAndOrientation(body_id)
            d = math.dist(ee_pos, pos)
            if d < best_dist:
                best_dist = d
                best_id   = body_id
                self._held_name = name

        if best_id is not None and best_dist < 0.1:
            self._constraint = pb.createConstraint(
                parentBodyUniqueId=self.robotic_arm,     # fixed
                parentLinkIndex=self.end_effector_id,    # fixed
                childBodyUniqueId=best_id,
                childLinkIndex=-1,
                jointType=pb.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0]
            )
            print(f"[Gripper] Grasped: {self._held_name}")
        else:
            self._constraint = None
            self._held_name  = None
            print("[Gripper] Nothing within reach to grasp.")

    def _detach_object(self):
        if hasattr(self, "_constraint") and self._constraint is not None:
            pb.removeConstraint(self._constraint)
            self._constraint = None
            print(f"[Gripper] Released: {getattr(self, '_held_name', '?')}")
            self._held_name = None

    def close(self):
        pb.disconnect()

def main():

    env = SimEnv(gui=True)
    rgb, depth, K = env.get_camera_data()
    time.sleep(100)
    env.close()

if __name__=="__main__":
    main()