# Roadmap: stereo visual-inertial SLAM

Goal: extend slam2.0 from stereo-only SLAM on KITTI to stereo **visual-inertial** SLAM, evaluated on
EuRoC and then on drone/handheld data. This is the step that turns a driving-dataset project into a
robotics one: nearly every robot and drone carries an IMU, and visual-only tracking fails exactly where
those platforms operate (aggressive motion, motion blur, low texture).

## Why this, and why it fits what we already have

The hard part of VIO is optimisation on a manifold with robust costs, and that already exists here:
`optimize/solver.py` does batched Jacobians, Huber, chi2 outlier handling and a Schur complement.
Adding IMU means new residuals and a larger per-keyframe state — an extension, not a rewrite.

Stereo also buys a real simplification over the monocular case: **scale is already observable**, so our
inertial initialisation estimates gravity direction, biases and velocities, but not scale.

## Verified facts (checked against the real data, 2026-09-12)

| Item | Value |
|---|---|
| EuRoC source | HF mirrors, range-request capable: `pepijn223/euroc-mirror` (V1_01, 1.15 GB), `GlowBond/EuRoC_MAV_Dataset` (machine_hall 12.7 GB with nested per-sequence zips) |
| Sequence sizes | MH_01 1.57 GB · MH_02 1.29 · MH_03 1.11 · MH_04 0.73 · MH_05 0.83 · V1_01 1.15 |
| Layout | ASL: `mav0/cam0/data/*.png`, `mav0/imu0/data.csv`, `mav0/state_groundtruth_estimate0/data.csv`, `sensor.yaml` |
| Camera | pinhole + radial-tangential, 752x480 @ 20 Hz, intrinsics [458.654, 457.296, 367.215, 248.375], distortion [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05] — **not rectified** |
| IMU | ADIS16448 @ 200 Hz; `T_BS` = identity, so body frame == IMU frame |
| IMU noise | gyro 1.6968e-4 rad/s/sqrt(Hz), gyro RW 1.9393e-5; accel 2.0e-3 m/s^2/sqrt(Hz), accel RW 3.0e-3 |
| Ground truth | 200 Hz: position, quaternion, velocity, **gyro bias, accel bias** (so the initialiser can be checked against truth) |
| ASL's own host | `robotics.ethz.ch` times out from here; use the mirrors above |

## Targets (ORB-SLAM3 paper, EuRoC, median of 10 runs, SE(3) alignment)

| System | Avg ATE |
|---|---|
| ORB-SLAM3 stereo-inertial | 0.035 m |
| ORB-SLAM3 stereo (visual only) | 0.084 m |
| BASALT | 0.051 m |
| Kimera | 0.119 m |
| VINS-Fusion | 0.138 m |

Landing between VINS-Fusion and ORB-SLAM3 is a good outcome and will be reported as such.

## The maths we are implementing

**Preintegration** (Forster et al.). Between keyframes i and j, accumulate IMU-only quantities that do
not depend on the pose or velocity at i:

    dR_ij = prod Exp((w_k - b_g) dt)
    dv_ij = sum dR_ik (a_k - b_a) dt
    dp_ij = sum [ dv_ik dt + 0.5 dR_ik (a_k - b_a) dt^2 ]

Bias changes are handled with precomputed first-order Jacobians instead of re-integration, and the
9x9 covariance is propagated incrementally as measurements arrive.

**Residuals** connecting states i and j (gravity g in the world frame):

    r_R = Log( (dR_ij Exp(J_dR_bg db_g))^T R_i^T R_j )
    r_v = R_i^T (v_j - v_i - g dt)        - [ dv_ij + J_dv db ]
    r_p = R_i^T (p_j - p_i - v_i dt - 0.5 g dt^2) - [ dp_ij + J_dp db ]

plus a bias random-walk residual between consecutive keyframes.

**Initialisation** (ORB-SLAM3's structure, minus scale): run visual-only for ~2 s of keyframes, then an
inertial-only MAP step estimating gravity direction (2-parameter), biases and per-keyframe velocities
with a Gaussian bias prior, then a joint visual-inertial refinement.

## Work plan

### Phase 0 — EuRoC adapter and a visual-only baseline (about 1 week)
- `scripts/fetch_euroc.py`: range-extract one sequence from the HF mirrors (mirrors `fetch_kitti.py`).
- `data/euroc.py`: ASL reader — images by timestamp, IMU csv, ground truth csv, `sensor.yaml` parsing.
- Rectification: `cv2.stereoRectify` + `initUndistortRectifyMap` from cam0/cam1 intrinsics and `T_BS`,
  producing the rectified `StereoCalib` the rest of the pipeline already expects.
- Integrity test, as for KITTI: image-derived rotation must agree with ground truth.
- **Gate**: all 11 sequences run visual-only, no crashes; ATE reported per sequence against
  ORB-SLAM3's stereo numbers (0.084 m average).

**Phase 0 result** (visual-only, threaded, ATE RMSE on camera centres, SE(3)-aligned, one run each):

| Seq | ours | ORB-SLAM3 stereo | lost frames | fps |
|---|---|---|---|---|
| MH_01 | 0.032 | 0.029 | 0 | 7.7 |
| MH_02 | 0.031 | 0.019 | 0 | 8.4 |
| MH_03 | 0.026 | 0.024 | 0 | 7.7 |
| MH_04 | 0.172 | 0.085 | 0 | 10.9 |
| MH_05 | 0.060 | 0.052 | 0 | 10.4 |
| V1_01 | 0.035 | 0.035 | 0 | 12.9 |
| V1_02 | 0.021 | 0.025 | 0 | 11.7 |
| V1_03 | 0.085 | 0.061 | 61 | 12.1 |
| V2_01 | 0.029 | 0.041 | 0 | 13.6 |
| V2_02 | 0.041 | 0.028 | 0 | 9.0 |
| V2_03 | **2.719** | 0.521 | 906 | 22.5 |

Ten sequences land within reach of ORB-SLAM3 (sum 0.532 m vs 0.399 m). V2_03 fails: at frame ~1014 motion
blur during a fast turn drops keypoints from 1200 to 173; the images recover within ~8 frames but tracking
never does, because there is no BoW relocalisation — the tracker keeps searching around an extrapolated
pose. This is the Phase 4 "visual-only fails" sequence, found early. Two bugs fixed on the way: loop
consistency groups multiplied every keyframe when several candidates overlapped (MH_01 ran out of memory),
and the motion model assumed a fixed frame interval (V2_03 drops every other left frame from frame 1003).

### Phase 1 — preintegration (about 1 week)
- `geometry/imu.py`: `Preintegrated` accumulating dR, dv, dp, covariance and bias Jacobians.
- Tests: constant-acceleration and constant-rotation cases against closed form; bias Jacobians against
  finite differences; covariance growth monotone; re-integration vs first-order bias correction.
- **Gate**: unit tests pass; integrating EuRoC IMU between two ground-truth poses reproduces the
  ground-truth relative motion to within the expected noise.

### Phase 2 — initialisation (about 1 week)
- `vi_init.py`: vision-only window, then inertial-only MAP (gravity direction, biases, velocities),
  then joint refinement.
- **Gate**: on V1_01 and MH_01, gravity direction within ~1 degree of vertical, estimated biases within
  the ground-truth bias range from the GT csv, velocities matching GT to a few cm/s.

### Phase 3 — tight coupling (about 2 weeks)
- `optimize/vi_ba.py`: nav-state nodes of 15 dimensions (pose 6, velocity 3, biases 6), reusing
  `_project`, `_dxi` and `_irls` from the existing solver. The visual-only solver stays untouched so
  KITTI results remain reproducible and comparable.
- Tracking: IMU-predicted pose replaces the constant-velocity model once initialised; constant velocity
  remains the fallback before initialisation.
- Local mapping: inertial residuals between consecutive keyframes in the sliding window; older
  keyframes fixed.
- **Gate**: stereo-inertial beats our own stereo-only on the same sequences; target average ATE <= 0.06 m,
  aiming at 0.035 m. No regression on KITTI 00 (still 1.73 m).

### Phase 4 — robots and drones (about 2 weeks)
- TUM VI (handheld, fisheye — needs its own rectification), then UZH-FPV or M3ED drone sequences.
- **Gate**: a sequence where visual-only tracking fails and visual-inertial survives, with the plot.
- Optional: KITTI raw OXTS (100 Hz IMU) as a controlled A/B on sequence 00, which we have already
  validated end to end.

## Risks and how each is handled

| Risk | Mitigation |
|---|---|
| Camera/IMU time offset | EuRoC timestamps are synchronised; verify by checking gyro-integrated rotation against image-derived rotation before trusting anything |
| Extrinsics `T_BC` confusion | `T_BS` for the IMU is identity, so `T_BC` is cam0's `T_BS`; ground truth is in the body frame, so evaluation must transform poses, not compare frames directly |
| IMU noise parameters | Taken from `sensor.yaml` rather than guessed |
| Initialisation divergence | Validate against ground-truth biases/velocities in the GT csv; refuse to initialise when the window has insufficient excitation |
| 15-dimensional state breaking the working solver | New module; visual-only path untouched; KITTI result is the regression test |
| Fisheye (TUM VI) | Deferred to Phase 4; EuRoC is pinhole+radtan and needs only standard rectification |

## Non-goals for this roadmap

Gaussian splatting maps, semantic/open-vocabulary layers, event cameras, and learned front-ends. Each is
a reasonable follow-on, but none of them is the missing capability; the IMU is.
