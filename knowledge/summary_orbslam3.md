# ORB-SLAM3 (Campos, Elvira, Gómez, Montiel, Tardós, TRO 2021)

Paper: [arxiv:2007.11898](https://arxiv.org/abs/2007.11898) · IEEE TRO DOI 10.1109/TRO.2021.3075644

## Thesis of the paper

Modern visual SLAM is **MAP estimation** (bundle adjustment). Accuracy comes from using **all three kinds of data association**, not from a fancier frontend:

1. **Short-term** — last few seconds. Pure VO lives here and always drifts.
2. **Mid-term** — map elements still geometrically close (small drift). Reprojection + local BA. Zero drift in already-mapped areas. This is why ORB-SLAM beats VO-with-loops.
3. **Long-term** — place recognition regardless of drift: loop closure, map merge, relocalization.

ORB-SLAM3 adds a fourth: **multi-map data association** (Atlas).

IMU is the other headline (tight VI, fast MAP IMU init). **Out of scope for our first system** (camera-only video). We still need Atlas and the improved place recognizer.

## System components

Four concurrent roles (visual mode):

| Thread | Rate | Job |
|---|---|---|
| Tracking | camera | Pose vs **active map** by minimizing reprojection; keyframe decision; reloc or new map on loss |
| Local mapping | keyframe | Insert KF/points, cull, local (VI) BA on a covisibility window |
| Loop + merge | keyframe | Place recognition vs **whole Atlas**; loop if same map, merge if different |
| Full BA | after loop | Independent thread; does not block tracking |

**Atlas:** set of disconnected maps. One **active** map. Unique DBoW2 database across all maps.

On tracking loss: reloc against all Atlas maps; if that fails after a timeout, freeze the active map as non-active and **initialize a new active map**.

## Camera abstraction (worth copying)

Pipeline is agnostic of the camera model: projection, unprojection, Jacobians live in a camera module. Pinhole and Kannala–Brandt fisheye shipped.

Relocalization uses **MLPnP** on projection rays, not ePnP (which assumes pinhole).

**Non-rectified stereo / multi-camera:** do not require rectified images. A rigid body pose in $\mathrm{SE}(3)$ plus known extrinsics. Overlapping views give metric triangulation; non-overlapping regions behave as monocular. This is the right model for **nuScenes 6-cam** and **Waymo 5-cam**.

## Improved-recall place recognition

Classic DBoW2: temporal consistency (3 consecutive KFs) **then** geometry $\Rightarrow$ ~100% precision, 30–40% recall, delayed loops, duplicated regions in Atlas.

ORB-SLAM3:

1. Query top-3 DBoW2 candidates, excluding covisible KFs.
2. Build a **local window** around the match (covisible KFs + their points).
3. RANSAC $\mathrm{Sim}(3)$ or $\mathrm{SE}(3)$ via Horn on 3-point 3D–3D matches.
4. Guided matching + nonlinear refinement of bidirectional reprojection (Huber).
5. Verify in **three covisible keyframes already in the map** (not three future frames). If missing, wait for incoming KFs without requiring BoW to fire again.

Then densify mid-term matches in that local window. This is why stereo ORB-SLAM3 is more accurate than ORB-SLAM2 on EuRoC even without IMU.

## Map merging (welding)

If $K_a$ (active map $M_a$) matches $K_m$ in another map $M_m$:

1. Assemble a **welding window** (covisible neighborhoods of both KFs). Transform $M_a$ by $T_{ma}$ into $M_m$.
2. Fuse maps; duplicate points collapse onto $M_m$ points; new covisibility edges.
3. **Welding BA** on the window (gauge: $M_m$ KFs outside the window that still see local points are fixed). Tracking can reuse $M_m$ immediately.
4. Essential-graph pose-graph on the rest of the merged map, welding window **fixed**.

Loop closing is the same algorithm when both KFs already belong to the active map, then full BA.

## Visual-inertial (deferred)

State $\mathcal{S}_i=\{T_i,v_i,b^g_i,b^a_i\}$. IMU preintegration (Forster manifold). Residuals on $\Delta R,\Delta v,\Delta p$ plus visual reprojection through $T_{CB}T_i^{-1}$. Fast MAP init: 2 s vision-only $\to$ inertial-only MAP for scale/gravity/bias $\to$ joint VI BA. Not needed until we have IMU streams (nuScenes CAN/IMU, KITTI IMU, later).

## What we will *not* copy first

- IMU preintegration / VI BA / gravity checks
- Fisheye Kannala–Brandt (KITTI/Waymo/nuScenes cameras are calibrated pinhole-like; undistort first)
- Running four tightly coupled C++ threads on day one — Python can be multi-threaded later; correctness of the pipeline first

## What we *will* copy

Three-thread visual SLAM + Atlas + improved place recognition + abstract camera + covisibility / essential graph / spanning tree + stereo-or-multi-camera body pose. Dataset adapters feed a `Frame` stream. Evaluation is RMS ATE with $\mathrm{SE}(3)$ alignment (stereo/multi-cam) or $\mathrm{Sim}(3)$ (monocular).

## Related reading used alongside this paper

- Hartley & Zisserman, *Multiple View Geometry* — $F$, $E$, $H$, triangulation, BA.
- Scaramuzza VO lectures — sequential SfM = initialize, localize, extend, BA; pose-graph vs BA.
- Triggs et al., *Bundle Adjustment — A Modern Synthesis*.
- Galvez-López & Tardós, DBoW2 (TRO 2012).
- Strasdat, *Scale-aware pose-graph* / double-window optimization.
- Cadena et al., *Past, Present, and Future of SLAM* (TRO 2016) — short/mid/long-term association terminology ORB-SLAM3 extends.
