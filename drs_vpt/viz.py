"""Plots and metrics for localization results (used by scripts/demo.py and scripts/demo_localization.py).

All poses are 4x4 world-to-camera matrices: x_cam = R @ x_cloud + t.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_VIZ_CAP = 200_000  # max cloud points drawn per panel


def project_points(xyz: np.ndarray, T_w2c: np.ndarray, K: np.ndarray):
    """Project (N, 3) cloud points into pixels. Returns uv (N, 2), depth (N,), in_front (N,)."""
    xyz_cam = xyz @ T_w2c[:3, :3].T + T_w2c[:3, 3]
    depth = xyz_cam[:, 2]
    uv_h = xyz_cam @ K.T
    uv = uv_h[:, :2] / (uv_h[:, 2:3] + 1e-8)
    return uv, depth, depth > 0.1


def pose_error(T_w2c_pred: np.ndarray, T_c2w_gt: np.ndarray) -> tuple:
    """Rotation error (deg) and camera-centre error (same unit as the cloud) against a GT camera-to-world pose."""
    R_pred, t_pred = T_w2c_pred[:3, :3], T_w2c_pred[:3, 3]
    R_gt, C_gt = T_c2w_gt[:3, :3].T, T_c2w_gt[:3, 3]
    rot_err = np.rad2deg(Rotation.from_matrix(R_pred @ R_gt.T).magnitude())
    trans_err = np.linalg.norm(-(R_pred.T @ t_pred) - C_gt)
    return float(rot_err), float(trans_err)


def save_overlay_figure(image: np.ndarray, cloud_xyz: np.ndarray, K: np.ndarray, poses: dict,
                        title: str, output_path: Path):
    """2x2 figure: the input image, then the cloud projected into it with each pose in `poses`
    (label -> 4x4 world-to-camera), coloured by depth. K must match the image resolution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = image.shape[:2]
    xyz = cloud_xyz
    if len(xyz) > _VIZ_CAP:
        xyz = xyz[np.random.default_rng(0).choice(len(xyz), _VIZ_CAP, replace=False)]

    fig, axes = plt.subplots(2, 2, figsize=(min(14, 1 + 9 * W / H), 9))  # narrower for portrait images
    fig.suptitle(title, fontsize=12)
    panels = [("Input image", None)] + list(poses.items())
    for ax, (label, T) in zip(axes.T.ravel(), panels):
        ax.imshow(image)
        if T is not None:
            uv, depth, in_front = project_points(xyz, T, K)
            mask = in_front & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if mask.any():
                d = depth[mask]
                ax.scatter(uv[mask, 0], uv[mask, 1], c=(d - d.min()) / (np.ptp(d) + 1e-6),
                           cmap="turbo", s=2.0, alpha=0.7, linewidths=0)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _frustum(T_w2c: np.ndarray, K_inv: np.ndarray, W: int, H: int, near: float, far: float):
    """Return (points, lines, camera_centre) of a camera frustum in cloud coordinates."""
    R, t = T_w2c[:3, :3], T_w2c[:3, 3]
    C = -(R.T @ t)

    def unproject(u, v, d):
        return R.T @ ((K_inv @ np.array([u, v, 1.0])) * d - t)

    corners = [(0, 0), (W, 0), (W, H), (0, H)]
    pts = np.vstack([C[None],
                     [unproject(u, v, near) for u, v in corners],
                     [unproject(u, v, far) for u, v in corners]]).astype(np.float32)
    segs = ([[0, 5 + i] for i in range(4)] +                # centre -> far corners
            [[1 + i, 1 + (i + 1) % 4] for i in range(4)] +  # near quad
            [[5 + i, 5 + (i + 1) % 4] for i in range(4)] +  # far quad
            [[1 + i, 5 + i] for i in range(4)])              # pillars
    return pts, np.array([[2, a, b] for a, b in segs], dtype=int).ravel(), C


# Camera axis that points up on screen after np.rot90(image, k), for each k.
_SCREEN_UP = {0: np.array([0.0, -1.0, 0.0]), 1: np.array([1.0, 0.0, 0.0]),
              2: np.array([0.0, 1.0, 0.0]), 3: np.array([-1.0, 0.0, 0.0])}


def upright_rotation(T_w2c: np.ndarray, up=None) -> int:
    """Return k for np.rot90(image, k) that shows the image upright, given the cloud's up axis.

    Photos taken with the phone held in portrait are often stored sideways (landscape, no EXIF tag), so the
    image shape alone cannot tell. With no `up`, the image is assumed upright (k = 0).
    """
    if up is None:
        return 0
    up_cam = T_w2c[:3, :3] @ np.asarray(up, dtype=float)
    return max(_SCREEN_UP, key=lambda k: _SCREEN_UP[k] @ up_cam)


def save_3d_view(cloud_xyz: np.ndarray, K: np.ndarray, image: np.ndarray, poses: dict,
                 title: str, output_path: Path, interactive: bool = False, up=None):
    """PyVista render of the camera image above a 3-D view of the cloud with a frustum per pose.

    K is the intrinsics of `image` (H, W, 3 uint8). The 3-D view looks along the last pose's forward axis from
    just behind it, rolled to match the displayed image. `up` (the cloud's up axis, from meta.json) rotates
    sideways photos upright; see upright_rotation.
    """
    import pyvista as pv

    H, W = image.shape[:2]
    K_inv = np.linalg.inv(K)
    far = float(np.percentile(np.linalg.norm(cloud_xyz, axis=1), 75))
    colors = ["#ff2020", "orange", "lime", "deepskyblue"]   # the input (identity) pose in bright red

    frusta = []
    for i, ((label, T), color) in enumerate(zip(poses.items(), colors)):
        s = 3.0 if i == 0 else 1.0                          # the input pose is usually far off: draw it larger
        pts, lines, _ = _frustum(T, K_inv, W, H, far * 0.002 * s, far * 0.1 * s)
        mesh = pv.PolyData()
        mesh.points, mesh.lines = pts, lines
        frusta.append((label, mesh, color))

    # View from the last (best) pose, with the same roll as the displayed image.
    T_ref = list(poses.values())[-1]
    k = upright_rotation(T_ref, up)
    shown = np.ascontiguousarray(np.rot90(image, k))
    sh, sw = shown.shape[:2]
    R, t = T_ref[:3, :3], T_ref[:3, 3]
    C, forward, view_up = -(R.T @ t), R.T @ np.array([0.0, 0.0, 1.0]), R.T @ _SCREEN_UP[k]
    f_vertical = K[1, 1] if k % 2 == 0 else K[0, 0]
    fov = np.degrees(2 * np.arctan(sh / 2 / f_vertical))

    cloud_pv = pv.PolyData(cloud_xyz)
    cloud_pv["height"] = cloud_xyz @ view_up

    # Image and 3-D view share the image's aspect: stacked for landscape, side by side for portrait.
    portrait = sh > sw
    panel_w, panel_h = (int(900 * sw / sh), 900) if portrait else (1600, int(1600 * sh / sw))
    shape = (1, 2) if portrait else (2, 1)
    window = (2 * panel_w, panel_h) if portrait else (panel_w, 2 * panel_h)
    pv.set_plot_theme("dark")
    pl = pv.Plotter(shape=shape, off_screen=not interactive, window_size=window)
    pl.subplot(0, 0)
    plane = pv.Plane(i_size=sw, j_size=sh)
    plane.texture_map_to_plane(inplace=True)
    pl.add_mesh(plane, texture=pv.numpy_to_texture(shown))
    pl.add_text("Camera image" + (f" (rotated {k * 90} deg to upright)" if k else ""),
                font_size=9, color="white", position="upper_left")
    pl.enable_parallel_projection()
    pl.view_xy()
    pl.reset_camera(bounds=plane.bounds)
    pl.camera.parallel_scale = sh / 2
    pl.add_text(title, font_size=8, color="lightgray", position="lower_left")

    pl.subplot(*((0, 1) if portrait else (1, 0)))
    pl.add_text(f"3-D view from behind the {list(poses)[-1].lower()} camera", font_size=9, color="white",
                position="upper_left")
    pl.add_points(cloud_pv, scalars="height", cmap="viridis", point_size=1.5, opacity=0.8, show_scalar_bar=False,
                  clim=np.percentile(cloud_pv["height"], [2, 98]))
    for i, (_, mesh, color) in enumerate(frusta):
        pl.add_mesh(mesh, color=color, line_width=5.0 if i == 0 else 2.5)
    for i, (label, _, color) in enumerate(frusta[::-1]):  # legend, bottom-up
        pl.add_text(f"— {label}", position=(10, 10 + 24 * i),
                    font_size=10, color=color)
    pl.camera.position = C - 0.15 * far * forward + 0.05 * far * view_up
    pl.camera.focal_point = C + far * forward
    pl.camera.up = view_up
    pl.camera.view_angle = min(fov * 1.4, 100.0)
    pl.reset_camera_clipping_range()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if interactive:
        pl.show()
    else:
        pl.screenshot(str(output_path), return_img=False)
    pl.close()
    print(f"Saved: {output_path}")


def rotate_view(image: np.ndarray, K: np.ndarray, poses: dict, k: int):
    """Rotate an image by np.rot90(image, k) and adjust K and the poses (label -> 4x4 world-to-camera) to match,
    so projections drawn on the rotated image stay consistent. Use k from upright_rotation."""
    K, poses = np.array(K, dtype=float), {label: np.array(T, dtype=float) for label, T in poses.items()}
    M = np.eye(4)
    M[:3, :3] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]  # camera axes after one 90-degree counter-clockwise turn
    for _ in range(k % 4):
        W = image.shape[1]
        image = np.rot90(image)
        K = np.array([[K[1, 1], 0, K[1, 2]], [0, K[0, 0], W - 1 - K[0, 2]], [0, 0, 1]])
        poses = {label: M @ T for label, T in poses.items()}
    return np.ascontiguousarray(image), K, poses


def save_trajectory_view(cloud_xyz: np.ndarray, frames: list, title: str, output_path: Path, up=None,
                         interactive: bool = False):
    """PyVista render of the map with the estimated (and ground-truth) camera trajectory.

    frames: dicts with "T_w2c" (estimate), optional "T_w2c_gt", "K" and "image_shape" (H, W) for the frusta.
    up: the cloud's up axis (meta.json); without it, the mean camera "up" is used.
    """
    import pyvista as pv

    est = [f["T_w2c"] for f in frames]
    centres = np.array([-(T[:3, :3].T @ T[:3, 3]) for T in est])
    forward = np.mean([T[:3, :3].T @ [0.0, 0.0, 1.0] for T in est], axis=0)
    up = np.asarray(up, dtype=float) if up is not None else -np.mean([T[:3, :3].T @ [0.0, 1.0, 0.0] for T in est], 0)
    up /= np.linalg.norm(up)
    centre = centres.mean(0)
    radius = float(np.percentile(np.linalg.norm(cloud_xyz - centre, axis=1), 90))

    cloud_pv = pv.PolyData(cloud_xyz)
    cloud_pv["height"] = cloud_xyz @ up
    pv.set_plot_theme("dark")
    pl = pv.Plotter(off_screen=not interactive, window_size=(1600, 1000))
    pl.add_points(cloud_pv, scalars="height", cmap="viridis", point_size=1.5, opacity=0.6, show_scalar_bar=False,
                  clim=np.percentile(cloud_pv["height"], [2, 98]))

    tracks = [("Estimate (fine)", "lime", "T_w2c")]
    if all(f.get("T_w2c_gt") is not None for f in frames):
        tracks.insert(0, ("Ground truth", "white", "T_w2c_gt"))
    depth = 0.04 * radius
    for label, color, key in tracks:
        for f in frames:
            H, W = f["image_shape"]
            pts, lines, _ = _frustum(f[key], np.linalg.inv(f["K"]), W, H, depth * 0.05, depth)
            mesh = pv.PolyData()
            mesh.points, mesh.lines = pts, lines
            pl.add_mesh(mesh, color=color, line_width=1.5)
        c = np.array([-(f[key][:3, :3].T @ f[key][:3, 3]) for f in frames])
        if len(c) > 1:
            pl.add_mesh(pv.lines_from_points(c), color=color, line_width=3)
    for i, (label, color, _) in enumerate(tracks[::-1]):
        pl.add_text(f"— {label}", position=(10, 10 + 24 * i), font_size=10, color=color)
    pl.add_text(f"{title} ({len(frames)} frames)", font_size=9, color="white", position="upper_left")

    # Look down at the trajectory from above and behind the cameras.
    behind = -(forward - (forward @ up) * up)
    behind = behind / np.linalg.norm(behind) if np.linalg.norm(behind) > 1e-6 else np.cross(up, [1.0, 0.0, 0.0])
    pl.camera.position = centre + radius * (1.0 * up + 1.2 * behind)
    pl.camera.focal_point = centre
    pl.camera.up = up
    pl.camera.view_angle = 50
    pl.reset_camera_clipping_range()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if interactive:
        pl.show()
    else:
        pl.screenshot(str(output_path), return_img=False)
    pl.close()
    print(f"Saved: {output_path}")
