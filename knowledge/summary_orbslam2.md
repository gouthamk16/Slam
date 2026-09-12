# ORB-SLAM2 (Mur-Artal, Tardós, TRO 2017)

Paper: [arxiv:1610.06475](https://arxiv.org/abs/1610.06475)

## What it adds over ORB-SLAM

Same three threads, plus a **fourth thread for full BA after loop closure**. Stereo and RGB-D make **scale observable**, so $\mathrm{Sim}(3)$ loop closing collapses to $\mathrm{SE}(3)$. Evaluated on **KITTI**, EuRoC, TUM RGB-D.

## Unified feature representation

After extraction, images are discarded. Everything is keypoints:

- **Stereo keypoint** $\mathbf{x}_s=(u_L,v_L,u_R)$. For rectified stereo: match left ORB on the same row in the right image, subpixel via patch correlation.
- **RGB-D** synthesizes a virtual right coordinate $u_R = u_L - f_x b / d$.
- **Monocular keypoint** $\mathbf{x}_m=(u_L,v_L)$ when no stereo/depth match exists. Triangulated later from multiple views; no scale.

**Close vs far stereo:** close if depth $< 40\times$ baseline. Close points give rotation, translation, and scale from one frame. Far points give mostly rotation; triangulate them only with multi-view support.

This close/far split is **critical for KITTI highways** (sequence 01): most of the scene is far, so the system must insert keyframes when close-point count drops ($\tau_t=100$ tracked close, $\tau_c=70$ creatable close).

## Bootstrapping

First stereo/RGB-D frame becomes a keyframe at the origin; all stereo keypoints become the initial map. No $H$/$F$ initialization.

## Bundle adjustment with stereo residuals

Motion-only BA (tracking):

$$\{\mathbf{R},\mathbf{t}\}=\arg\min_{\mathbf{R},\mathbf{t}}\sum_{i\in\mathcal{X}}\rho\Big(\big\|\mathbf{x}^i_{(\cdot)}-\pi_{(\cdot)}(\mathbf{R}\mathbf{X}^i+\mathbf{t})\big\|^2_\Sigma\Big)$$

Monocular projection $\pi_m$ is the pinhole model. Rectified stereo projection:

$$\pi_s\begin{bmatrix}X\\Y\\Z\end{bmatrix}=\begin{bmatrix}f_x X/Z+c_x\\ f_y Y/Z+c_y\\ f_x(X-b)/Z+c_x\end{bmatrix}$$

$\rho$ is Huber; $\Sigma$ is the covariance of the keypoint's pyramid scale.

Local BA: optimize covisible keyframes $\mathcal{K}_L$ and their points $\mathcal{P}_L$; other keyframes $\mathcal{K}_F$ that observe $\mathcal{P}_L$ contribute residuals but stay **fixed**.

Full BA: local BA over the whole map; origin keyframe fixed.

## Loop closing

Geometric validation and pose-graph use $\mathrm{SE}(3)$, not $\mathrm{Sim}(3)$. After pose-graph, launch **full BA in a separate thread**. If another loop arrives, abort GBA. When GBA finishes, merge by propagating pose corrections through the spanning tree to keyframes inserted during GBA.

## Localization mode

Mapping and loop closing off. Tracking uses map matches (drift-free in mapped areas) plus VO matches from the previous stereo frame (survives unmapped regions, but can drift).

## KITTI numbers (paper)

Stereo ORB-SLAM2 is a standard KITTI visual SLAM baseline. Sequence 00/05/07 have loops; 01 is the highway stress test for close points.

## Relevance to slam2.0

For **KITTI stereo first**, this paper is more operational than ORB-SLAM3: stereo keypoints, close/far policy, $\mathrm{SE}(3)$ loops, GBA merge via spanning tree. nuScenes/Waymo are **multi-camera without a classic stereo pair**, so we will later treat them more like ORB-SLAM3's non-rectified multi-camera body pose than like KITTI rectified stereo.
