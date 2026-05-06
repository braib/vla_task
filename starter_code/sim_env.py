import numpy as np
import pybullet as pb
import pybullet_data as pd
import time

OBJECTS = [
    {"id": 1, "name": "red_cube",  "color": [1, 0, 0, 1], "shape": "box",  "length": 0.04, "width": 0.04, "height": 0.04 },
    {"id": 2, "name": "blue_bowl", "color": [0, 0, 1, 1], "shape": "bowl", "radius": 0.2, "height": 0.04 },
]

MAX_ARM_REACH = 0.8  # 0.89 m # 855 mm 
MIN_ARM_REACH = 0.2  # 0.15 m

PLANE_GROUND = ["plane.urdf",               [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
TABLE_LENGTH = 3.0
TABLE_WIDTH  = 2.5
TABLE_HEIGHT = 0.6
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
        self.object_poses = {}

        self.random_seed = 1
        np.random.seed(self.random_seed)

        self._connect()

        self._load_scene()
        self._spawn_robot()
        # self._compute_intrinsics()
 
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
            halfExtents=[
                TABLE_LENGTH / 2,
                TABLE_WIDTH / 2,
                TABLE_HEIGHT / 2
            ]
        )

        table_vis = pb.createVisualShape(
            pb.GEOM_BOX,
            halfExtents=[
                TABLE_LENGTH / 2,
                TABLE_WIDTH / 2,
                TABLE_HEIGHT / 2
            ],
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
        spawned_poses = []  

        for obj in OBJECTS:

            if obj["shape"] == "box":

                fos = max(obj["length"], obj["width"])
                x, y = sample_workspace_pose(
                    max_reach = MAX_ARM_REACH - fos,
                    min_reach = max(MIN_ARM_REACH + fos, 0.0),
                    seed = self.random_seed + obj["id"]
                )

                length = obj["length"]
                width  = obj["width"]
                height = obj["height"]

                col_id = pb.createCollisionShape(
                    pb.GEOM_BOX,
                    halfExtents=[
                        length / 2,
                        width / 2,
                        height / 2
                    ]
                )

                vis_id = pb.createVisualShape(
                    pb.GEOM_BOX,
                    halfExtents=[
                        length / 2,
                        width / 2,
                        height / 2
                    ],
                    rgbaColor=obj["color"]
                )

                body_id = pb.createMultiBody(
                    baseMass=0.1,
                    baseCollisionShapeIndex=col_id,
                    baseVisualShapeIndex=vis_id,
                    basePosition=[
                        x,
                        y,
                        table_top_z + height / 2
                    ],
                    baseOrientation=quat
                )

            # ─────────────────────────────
            # BOWL OBJECT
            # ─────────────────────────────

            elif obj["shape"] == "bowl":
                fos = obj["radius"]
                x, y = sample_workspace_pose(
                    max_reach = MAX_ARM_REACH - fos,
                    min_reach = max(MIN_ARM_REACH + fos, 0.0),
                    seed = self.random_seed + obj["id"]
                )
                radius = obj["radius"]
                height = obj["height"]

                col_id = pb.createCollisionShape(
                    pb.GEOM_CYLINDER,
                    radius=radius,
                    height=height
                )

                vis_id = pb.createVisualShape(
                    pb.GEOM_CYLINDER,
                    radius=radius,
                    length=height,
                    rgbaColor=obj["color"]
                )

                body_id = pb.createMultiBody(
                    baseMass=0.1,
                    baseCollisionShapeIndex=col_id,
                    baseVisualShapeIndex=vis_id,
                    basePosition=[
                        x,
                        y,
                        table_top_z + height / 2
                    ],
                    baseOrientation=quat
                )

            else:
                continue

            self.object_ids[obj["name"]] = body_id

    def _spawn_robot(self):
        self.robotic_arm     = pb.loadURDF(ROBOTIC_ARM[0], ROBOTIC_ARM[1], ROBOTIC_ARM[2], useFixedBase=True)

        self.num_joints      = pb.getNumJoints(self.robotic_arm)
        self.end_effector_id = self.num_joints - 1




def main():

    env = SimEnv(gui=True)

    time.sleep(10)


if __name__=="__main__":
    main()