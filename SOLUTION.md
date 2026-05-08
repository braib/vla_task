# SOLUTION.md - VLA Pick & Place Pipeline

## 1. VLM / AI Model Choices

**Gemini Flash** is used for NLP parsing because tabletop commands often contain colour metaphors, synonyms, and indirect phrasing (e.g. "the cube with same color as apple", "transfer into container having color that rhymes with buffalo") that a regex parser cannot handle. Gemini maps arbitrary natural language to a clean `{colour, shape}` struct via a tightly constrained JSON prompt.

**Grounding DINO** (SwinT backbone) is used for visual grounding because the names or labels of the objects are not fixed - open-vocabulary detection handles arbitrary adjective-noun combinations like "yellow cube" or "red bowl" without any fine-tuning. It runs fully locally with no API key required.

**mink** is used for IK because the Franka Panda has 7 DOF, meaning there are infinitely many joint configurations for any given end-effector pose - making it impossible to solve analytically through geometry and trigonometry alone.

---

## 2. 2D-to-3D Coordinate Transformation

The pipeline uses standard **pinhole back-projection** implemented in `projection.py`.

### Step 1 - Robust depth at pixel
A 7×7 median filter is applied to the depth patch around the detected centroid pixel to suppress single-pixel z-buffer outliers common in MuJoCo's rendered depth maps:

```
Z = median( depth[v±3, u±3] )
```

### Step 2 - Pixel → Camera frame
Using the pinhole intrinsic matrix **K** derived from MuJoCo's `fovy=60°`:

```
fx = (H/2) / tan(fovy/2) × (W/H)  ≈ 554.3
fy = (H/2) / tan(fovy/2)           ≈ 415.7
cx = W/2 = 320,  cy = H/2 = 240

X_cam = (u - cx) × Z / fx
Y_cam = (v - cy) × Z / fy
Z_cam = Z
```

### Step 3 - Camera frame → World frame
The overhead camera sits at world position `[0.5, 0.0, 1.72]`, pointing straight down. Its fixed rotation matrix flips the Y axis and negates the optical axis to align with world coordinates:

```
R_cam_to_world = diag(1, -1, -1)

p_world = cam_pos + R_cam_to_world × [X_cam, Y_cam, Z_cam]
```

### Step 4 - Table-surface snap
Because all objects rest on the table (`z ≈ 0.42 m`), `project_to_table()` overwrites the inferred Z with the known table height, eliminating residual depth noise for the grasp target position.

---

## 3. Current Limitations & Improvements

1. **No grasp-success detection** - The pipeline does not verify the object is held after gripper close; if the object falls the pipeline does not re-attempt the pick. This could be addressed by using a VLM such as GPT-4.1 to detect failures by giving it context of the task, robot, and current camera view.

2. **Single overhead camera** - Occlusions or objects near the edge of the workspace are missed. A wrist-mounted camera would provide close-range verification, and a secondary viewpoint would resolve occlusions.

3. **Fixed table-height assumption** - `project_to_table` breaks when objects are stacked. The fix is to estimate per-object height from the full depth map point cloud rather than snapping to a constant Z.

4. **No obstacle avoidance in IK** - mink currently uses joint-limit constraints only. Adding `CollisionAvoidanceLimit` in mink for the table, object geometries, and arm self-collision would prevent unintended contacts.

5. **Simplified gripper dynamics** - Finger friction is not tuned, so small or smooth objects can slip. Tuning contact friction parameters in the MJCF and adding a post-grasp weld constraint would stabilise the held object during transport.

6. **Top-down grasp only** - Roll/pitch/yaw is fixed to zero throughout the pick sequence. Integrating a grasp-pose estimator such as AnyGrasp or GraspNet would enable 6-DOF grasp candidates predicted directly from the depth image.