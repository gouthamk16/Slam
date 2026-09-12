# Visual ORB-SLAM3 analog — walkthrough plan

> This is the **teaching sequence** for writing slam2.0 together. You type the project code. The assistant already has a complete hidden reference so later phases are not invented mid-stream. Do not copy that reference into this repo.

**Goal:** Build a camera-only visual SLAM analogous to ORB-SLAM3 (tracking, local mapping, loop closing, Atlas) on a KITTI stereo stream, then plug nuScenes/Waymo in as `Frame` adapters.

**Architecture:** Keyframe feature SLAM. Estimator is Huber-weighted reprojection BA. Same ORB features for tracking, mapping, and place recognition. Three threads in *logic* (tracking / local mapping / loop); we may run them sequentially at first and split processes only when the math is honest.

**Tech stack:** Python 3.11+, NumPy, OpenCV, SciPy. Gauss–Newton / LM written by us for motion-only BA, local BA, and SE(3) pose-graph. No g2o/Ceres required for v1. IMU, SuperPoint, DROID, fisheye: later.

**Already done (not recoded in this plan):** KITTI odometry sequence 00 on disk at `datasets/kitti/` (4541 stereo pairs, `calib.txt`, `times.txt`, poses `00.txt`–`10.txt`).

**How each chat step works**

1. One engineer **thought** (why this exists now).
2. The **smallest code** that makes that thought real (about 5–20 lines you type).
3. A **run** (pytest node or a 10-line script) and what you should see.
4. Stop. No unused abstractions for later phases.

We do **not** finish files top-to-bottom. We add a function when the next frame would be wrong without it.

---

## End-state files (what we grow into, not what we create on day one)

| Path | Responsibility |
|---|---|
| `src/slam/data/frame.py` | `Frame`: id, stamp, left/right images, camera, pose, features |
| `src/slam/data/kitti.py` | KITTI odometry iterator + calib/pose parse |
| `src/slam/geometry/camera.py` | Pinhole + stereo rig: project, unproject, undistort, baseline depth |
| `src/slam/geometry/lie.py` | SE(3) exp/log, invert, transform points, left Jacobian |
| `src/slam/geometry/triangulate.py` | DLT, cheirality, parallax |
| `src/slam/geometry/align.py` | Horn SE(3), RANSAC, PnP |
| `src/slam/features/orb.py` | ORB pyramid extract |
| `src/slam/features/stereo.py` | Right-image match, close/far split |
| `src/slam/optimize/ba.py` | Huber, motion-only BA, local BA |
| `src/slam/optimize/pose_graph.py` | Essential-graph SE(3) pose-graph |
| `src/slam/map/structure.py` | MapPoint, KeyFrame, covisibility, spanning tree, Map, Atlas |
| `src/slam/tracking.py` | Motion model, associate local map, keyframe decision, reloc |
| `src/slam/mapping.py` | Insert KF, triangulate, cull, local BA |
| `src/slam/loop.py` | BoW, geometric verify, fuse, pose-graph, full BA merge |
| `src/slam/eval.py` | ATE / RPE, Sim(3) vs SE(3) alignment |
| `src/slam/system.py` | `SlamSystem.track(frame)` orchestration |
| `scripts/run_kitti.py` | Play sequence 00, dump trajectory |
| `tests/` | One test module per phase checkpoint |

Names can change. Behavior cannot.

---

## Phase 1 — Dataset stream

**Engineer thought:** I have KITTI. I need to loop frames.

**Checkpoint:** `python scripts/show_frame.py` (or a test) loads `datasets/kitti/sequences/00/image_0/000000.png` and the matching right image; shape `(376, 1241)`, `uint8`. Then iterate 10 frames with stamps from `times.txt`.

### Steps (thought order)

1. Package skeleton: `pyproject.toml`, `src/slam`, pytest, existing `.venv`.
2. Open stereo frame 0 from disk (paths only). No classes.
3. `Frame` dataclass: `frame_id`, `stamp`, `image_left`, `image_right`.
4. Parse `times.txt` → stamp per index.
5. `KittiSequence` that yields `Frame` for `image_0` / `image_1` by index. `__len__` = 4541 for seq 00.
6. Parse `poses/00.txt` (3×4 camera-to-world per line) → store `gt_T_cw` on the frame as 4×4 **world-to-camera** (invert KITTI’s T_wc). Needed later for ATE; attach now while the file is in front of us.
7. Parse `calib.txt` **just enough to hang on the frame:** `P0` → `fx, fy, cx, cy`; `P1[0,3]` → `baseline = -P1[0,3]/P1[0,0]`. Store numbers on `Frame.camera`. **Do not** write `project()` yet.

**Explicitly not this phase:** ORB, undistort, SE(3) exp/log, BA, map.

**Coverage:** KITTI gray stereo stream, timestamps, GT pose file, rectified calib numbers.

---

## Phase 2 — Camera as geometry (because the next line of code needs rays)

**Engineer thought:** Pixels are measurements. I need π and π⁻¹.

**Checkpoint:** Project a point on the optical axis to `(cx, cy)`. Stereo: known 3D point → `(u_L, v, u_R)` → recover depth `fx * b / (u_L - u_R)` within 1e-6.

### Steps

1. `PinholeCamera.project` / `unproject` / `in_image`.
2. `StereoRig.project_stereo`, `depth_from_disparity`, `triangulate_stereo`.
3. Undistort points if `dist` is non-zero (KITTI odometry images are already rectified; keep the hook, default zeros).
4. SE(3) `exp` / `log` / `invert` / `transform_points` — introduced here because `T_cw` is about to mean something. Left perturbation `xi = [ω, v]`, `T ← exp(ξ) T`.

**Explicitly not this phase:** F/E/H RANSAC, PnP, bundle adjustment. Those appear when tracking/init need them.

**Coverage:** Pinhole, stereo baseline scale, Lie algebra for poses.

---

## Phase 3 — Frontend (ORB + stereo match)

**Engineer thought:** I need the same features for tracking, mapping, and place recognition.

**Checkpoint:** On KITTI 00 frame 0, extract ~1000–2000 ORB; a large fraction of left features get a right `u_R` on the same row, positive disparity. Draw matches for one frame.

### Steps

1. Grayscale is already loaded; color conversion if a later dataset is RGB.
2. ORB: 8 levels, scale 1.2, grid so features are not all on one building. KITTI resolution → target 2000 corners.
3. Stereo match: same-row search, Hamming + ratio test, disparity in `(0.5, max_disp]`.
4. Close vs far: close if depth `< 40 × baseline` (~21.6 m on KITTI). Tag each keypoint. Highway sequences will need this for keyframe policy.

**Explicitly not this phase:** Data association to a map, pose.

**Coverage:** ORB-SLAM2 stereo keypoints `x_s = (u_L, v, u_R)` and monocular leftovers.

---

## Phase 4 — Visual odometry (tracking a map that is only “recent”)

**Engineer thought:** Frame 0 can build a metric map from stereo. Frame 1 must find that map in the image and refine pose.

**Checkpoint:** Run 50–100 frames of seq 00. Trajectory is smooth, not identity. ATE vs GT is finite (will still drift — no loops yet). RPE on short windows is the real VO score.

### Steps

1. **Bootstrap:** first stereo frame at `T = I`; triangulate close stereo points → MapPoints.
2. Constant-velocity motion model: `T_rel = T_last @ inv(T_prev)`, predict `T_pred = T_rel @ T_last`.
3. Associate: project map points, search in a window, Hamming if descriptors exist.
4. **Motion-only BA:** pose only, Huber reprojection (2-vector mono or 3-vector stereo).
5. If too few matches: widen search; if still lost, mark lost (Atlas comes in Phase 7; for now fail loudly).
6. Keyframe decision (VO version): min frames since last KF; tracked count; visual change vs reference KF; extra insert if close-point count drops (`τ_t ≈ 100`, `τ_c ≈ 70`).
7. PnP+RANSAC relocalization against the current map (needed before Atlas).
8. Two-view monocular init (H vs F, Hartley–Zisserman) **only if** we later run mono; KITTI stereo does not need it. Keep as a named later bullet, not in the KITTI path.

**Geometry pulled in here:** motion-only BA Jacobians (`J_π @ [-hat(X_c) | I]`), OpenCV `solvePnPRansac` / MLPnP if we abstract the camera.

**Explicitly not this phase:** Local BA of structure, covisibility graph, loop, BoW.

**Coverage:** Short-term data association + pose at camera rate.

---

## Phase 5 — Local mapping (the map becomes SLAM, not VO)

**Engineer thought:** Intermediate frames only localize. Structure is estimated from keyframes.

**Checkpoint:** Same 100 frames, now with local BA. Map point count stays bounded (culling). RPE better than Phase 4. Covisibility neighbors of the latest KF are non-empty.

### Steps

1. MapPoint: `Xw`, observations `{kf_id: kp_index}`, representative descriptor, viewing direction, `dmin/dmax`, found/visible counts.
2. KeyFrame: `T_cw`, keypoints, descriptors, `u_right`, map point ids.
3. Covisibility graph: edge if ≥ 15 shared points; weight = count.
4. Spanning tree: new KF links to the KF with most shared points.
5. On new KF: update graph; BoW vector (can compute now, query in Phase 6).
6. Recent point culling: must be found in >25% of predicted frames and seen in ≥ 3 KFs (relax to 2 on tiny tests).
7. Triangulate unmatched ORB vs covisible KFs: epipolar, positive depth, parallax, scale consistency. Stereo pairs triangulate immediately.
8. **Local BA:** optimize current KF + covisible KFs + their points; other KFs that see those points are **fixed**. Huber.
9. Keyframe culling: drop if 90% of points already seen in ≥ 3 KFs at same or finer scale.

**Coverage:** Mid-term data association; ORB-SLAM local mapping.

---

## Phase 6 — Loop closing + full BA

**Engineer thought:** Seq 00 loops. Drift must reset.

**Checkpoint:** Full seq 00 (or first 1500 + a known loop). After a loop, ATE drops vs Phase 5. Essential-graph node count = keyframe count.

### Steps

1. DBoW-style inverted index over ORB (our own small vocabulary is enough; same interface as DBoW2).
2. Query: exclude covisible KFs; take top candidates.
3. Geometric verify: 3D–3D Horn RANSAC → `SE(3)` (stereo, scale known). `Sim(3)` only if we are in monocular mode.
4. ORB-SLAM3 recall trick: verify in **three covisible keyframes already in the map**, not three future frames.
5. Fuse duplicated points; add loop edges to covisibility / essential graph.
6. Essential graph = spanning tree + high-covisibility (`θ ≥ 100`) + loop edges.
7. Pose-graph on `SE(3)` (stereo) over that graph.
8. Full BA in a separate pass; if new keyframes arrived, propagate via spanning tree (ORB-SLAM2 merge rule).
9. Eval: RMS ATE with `SE(3)` Umeyama/Horn alignment to KITTI GT; RPE as in the KITTI devkit idea (relative pose over 100–800 m) if we want paper-comparable numbers.

**Coverage:** Long-term association; loop; global consistency.

---

## Phase 7 — Atlas (multi-map)

**Engineer thought:** Tracking loss should not kill the session.

**Checkpoint:** Scripted gap (drop 30 frames of seq 00) → new map starts → on revisit, maps weld. Two maps become one; tracking continues.

### Steps

1. Atlas = list of maps, one active; one BoW database across all.
2. On loss: reloc against **all** maps; switch active map if needed.
3. If reloc fails after a timeout: freeze active as inactive, bootstrap a new active map.
4. Place recognition vs whole Atlas: same map → loop (Phase 6); other map → weld.
5. Welding window: covisible neighborhoods, transform, fuse points, local BA, then essential-graph on the rest with weld window fixed.

**Coverage:** ORB-SLAM3 Atlas / multi-map data association.

---

## Phase 8 — System, visualization, KITTI runner

**Engineer thought:** I need one entry point and to see it.

### Steps

1. `SlamSystem.track(frame) -> T_cw`; sequential threads first (track → maybe local map → maybe loop).
2. `scripts/run_kitti.py --seq 00 --max-frames N` prints KF/point counts and ATE.
3. Minimal viz: current pose on GT XY, or write `trajectory.txt` (KITTI 3×4 T_wc lines).
4. Optional: run tracking / mapping / loop on three Python threads once single-thread is correct.

**Coverage:** Usable binary; eval hook.

---

## Phase 9 — nuScenes / Waymo as Frame adapters

**Engineer thought:** The SLAM core must not know dataset folders.

**Checkpoint:** One nuScenes camera stream and one Waymo camera stream produce `Frame`s with `K`, distortion, stamp, and body extrinsics. Front-camera VO runs. Surround cameras later as a rigid multi-camera body (ORB-SLAM3 non-rectified stereo idea).

### Steps

1. Keep `Frame` stable. New modules `data/nuscenes.py`, `data/waymo.py` only.
2. Body pose `T_WB` plus `T_C_i_B` extrinsics. Overlap → metric triangulation; non-overlap → monocular rays.
3. Do **not** assume KITTI rectified rows for matching.

**Coverage:** Dataset-agnostic core.

---

## Deferred on purpose (not forgotten)

| Item | When |
|---|---|
| IMU preintegration, VI BA, MAP IMU init | After visual Atlas is real; nuScenes IMU/CAN |
| Fisheye Kannala–Brandt | If a camera is not well-modeled as pinhole+undistort |
| Learned frontend (SuperPoint / LightGlue) | Swap behind the same `extract` / `match` interface after Phase 6 |
| DROID-style dense BA | Research fork, not v1 |
| Full 22-sequence KITTI gray zip | When we want 01 highway / 05–07 extra loops |
| Real-time C++ | Out of scope |

---

## ORB-SLAM3 analog — coverage checklist

Use this so a phase cannot be “done” while a box is empty.

**Data association**

- [ ] Short-term (last frame / motion model)
- [ ] Mid-term (covisibility local map in tracking and local BA)
- [ ] Long-term (BoW loop / reloc)
- [ ] Multi-map (Atlas weld)

**Tracking**

- [ ] ORB extract
- [ ] Stereo `u_R` + close/far
- [ ] Motion model + motion-only BA
- [ ] Track local map (FOV, 60° viewing angle, scale)
- [ ] Keyframe policy including close-point rule
- [ ] Relocalization (PnP)

**Mapping**

- [ ] Generous KF insert + later cull
- [ ] Point cull (25% / 3 KF rule)
- [ ] Triangulation + cheirality + parallax
- [ ] Local BA with fixed outer keyframes
- [ ] Covisibility + spanning tree + essential graph

**Loop / global**

- [ ] Place recognition with geometric verify
- [ ] Three-covisible verification (ORB-SLAM3 recall)
- [ ] Point fusion
- [ ] SE(3) pose-graph (Sim(3) if mono)
- [ ] Full BA + spanning-tree merge

**Robustness**

- [ ] Lost → reloc → else new Atlas map
- [ ] Map merge welding BA

**Eval / data**

- [ ] KITTI 00 ATE/RPE
- [ ] Frame adapter for at least one other dataset

---

## Session rhythm

| Session | Phase | You can run |
|---|---|---|
| A | 1 | Iterate KITTI frames |
| B | 2 | Project / stereo depth unit tests |
| C | 3 | ORB+stereo on frame 0 |
| D | 4 | 50-frame VO + dump XY |
| E | 5 | Local map + better RPE |
| F | 6 | Seq 00 with a loop, ATE |
| G | 7 | Survive a tracking gap |
| H | 8 | `run_kitti.py` |
| I | 9 | Second dataset adapter |

Do not start session N+1 until the checkpoint of N is green.

---

## What the assistant does vs what you do

- **You:** type code in this repo, run tests, say when a checkpoint is green (or paste the failure).
- **Assistant:** one thought + small snippet per message; compare silently to the hidden reference; never dump a whole module unless you ask to go faster.
- **Neither:** skip a checklist box because “we’ll add it later” without putting it on the deferred table.
