# ORB-SLAM (Mur-Artal, Montiel, Tardós, TRO 2015)

Paper: [arxiv:1502.00956](https://arxiv.org/abs/1502.00956)

## What it is

The canonical **feature-based monocular keyframe SLAM**. Same ORB features are used for tracking, mapping, and place recognition. Three parallel threads: **Tracking**, **Local Mapping**, **Loop Closing**. This is the skeleton every later ORB-SLAM version keeps.

## Why keyframes (not a filter)

PTAM showed that splitting tracking (frame-rate pose) from mapping (keyframe BA) is more accurate than EKF SLAM at the same compute ([Strasdat et al., "Why filter?"](https://www.robots.ox.ac.uk/~mobile/Papers/2010ICRA_strasdat.pdf)). Intermediate frames are used only to localize; structure is estimated from a sparse set of keyframes.

## Data structures (must copy)

A **map point** stores: 3D position $X_{w,i}$, mean viewing direction $n_i$, a representative ORB descriptor $D_i$, and scale-invariant observation range $[d_{\min}, d_{\max}]$.

A **keyframe** stores: pose $T_{iw}\in\mathrm{SE}(3)$ (world $\to$ camera), intrinsics, and all undistorted ORB features (matched or not).

**Covisibility graph:** undirected weighted graph. Edge if two keyframes share $\ge 15$ map points; weight $\theta$ is the shared count.

**Spanning tree:** connected subgraph with a minimal number of edges, grown as keyframes are inserted/culled.

**Essential graph:** spanning tree + high-covisibility edges ($\theta_{\min}=100$) + loop-closure edges. Used for pose-graph optimization so we do not optimize the dense covisibility graph.

## Place recognition

DBoW2 bag-of-words over ORB. Offline vocabulary. Incremental inverted-index database of keyframes. Query groups scores by covisibility, not just time. Matching can be restricted to features in the same vocabulary-tree node (level 2 of 6).

## Map initialization (monocular)

Parallel RANSAC of a homography $H$ (planar) and fundamental matrix $F$ (general 3D), Hartley–Zisserman algorithms. Model selection:

$$R_H = \frac{S_H}{S_H+S_F}$$

Choose $H$ if $R_H > 0.45$, else $F$. Recover motion (8 hypotheses from $H$, 4 from $E=K^\top F K$), triangulate, keep the unique reconstruction with positive depth, parallax, and low reprojection. Then full BA. Refuse initialization if there is no clear winner (low parallax / planar twofold ambiguity).

## Tracking (every frame)

1. Extract ORB: 8-level pyramid, scale $1.2$, grid of FAST corners (1000 typical, **2000 on KITTI** $1241\times376$).
2. Predict pose with constant-velocity model; guided search of last-frame map points; **motion-only BA**.
3. If lost: DBoW2 reloc + PnP + RANSAC.
4. **Track local map:** project points from covisible keyframes $\mathcal{K}_1$ and their neighbors $\mathcal{K}_2$; reject by image bounds, viewing angle $60^\circ$, scale range; descriptor match at predicted pyramid level; motion-only BA again.
5. Keyframe if: $>20$ frames since reloc; mapping idle or $>20$ frames since last KF; $\ge 50$ tracked points; tracks $<90\%$ of the reference keyframe's points. Generous insertion, later culling.

## Local mapping (every new keyframe)

1. Update covisibility graph and spanning tree; compute BoW.
2. Cull recent points: must be found in $>25\%$ of predicted frames and seen in $\ge 3$ keyframes.
3. Triangulate unmatched ORB vs covisible keyframes (epipolar check, positive depth, parallax, reprojection, scale).
4. **Local BA:** optimize current KF + covisible KFs + their points; other KFs that see those points are **fixed**.
5. Cull redundant KFs: $90\%$ of points already seen in $\ge 3$ other KFs at the same or finer scale.

## Loop closing

1. Query DBoW2; require **three consecutive consistent** candidates (covisible).
2. Compute $\mathrm{Sim}(3)$ with Horn + RANSAC (7-DoF monocular drift: $R,t,s$).
3. Fuse duplicated points; align both sides of the loop.
4. Pose-graph optimization on the **essential graph** over $\mathrm{Sim}(3)$ (Strasdat scale-aware pose graph). Transform points with the correction of one observing keyframe.

## Optimization

All BA/pose-graph via Levenberg–Marquardt in **g2o**. Robust Huber kernel on reprojection.

Motion-only BA: pose only. Local BA: local poses + points. Full BA: everything except origin KF (gauge).

## Relevance to slam2.0

This paper is the **algorithm spec** for our visual core. ORB-SLAM2/3 add stereo/RGB-D, IMU, Atlas, and better place recognition, but tracking/local mapping/loop closing still follow these steps. Hartley & Zisserman is the geometry bible behind $H$/$F$/$E$, triangulation, and BA.
