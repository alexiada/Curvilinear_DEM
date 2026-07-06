"""
Minimal 3D pre-processing helpers for Case Study 4.

The container is a star-shaped surface written in spherical coordinates as

    x = r * B(theta, phi),     0 <= r <= 1,

where B is the bunny-ear boundary surface.  This file intentionally contains
only the routines needed to create and inspect initial particles.
"""

import math
import warnings
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


TWO_PI = 2.0 * np.pi


def bunny_ear_radius(theta, phi, R0=1.0, eps=0.55, phi0=0.35, sigma=0.20):
    """Star-shaped boundary radius R(theta, phi)."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    return R0 * (
        1.0
        + eps
        * np.exp(-((phi - phi0) ** 2) / sigma**2)
        * np.cos(2.0 * theta)
    )


def bunny_ear_surface(
    theta=None,
    phi=None,
    R0=1.0,
    eps=0.55,
    phi0=0.35,
    sigma=0.20,
    n_theta=240,
    n_phi=160,
):
    """Return the boundary mesh X, Y, Z and the angular grids."""
    if theta is None:
        theta = np.linspace(0.0, TWO_PI, int(n_theta), endpoint=True)
    if phi is None:
        phi = np.linspace(0.0, np.pi, int(n_phi), endpoint=True)

    Theta, Phi = np.meshgrid(theta, phi)
    R = bunny_ear_radius(Theta, Phi, R0=R0, eps=eps, phi0=phi0, sigma=sigma)

    X = R * np.sin(Phi) * np.cos(Theta)
    Y = R * np.sin(Phi) * np.sin(Theta)
    Z = R * np.cos(Phi)

    return X, Y, Z, Theta, Phi, R


def map_particles_3d(q, R0=1.0, eps=0.55, phi0=0.35, sigma=0.20):
    """Map q = (r, theta, phi) to physical x = (x, y, z)."""
    q = np.asarray(q, dtype=float)
    r = q[:, 0]
    theta = q[:, 1]
    phi = q[:, 2]

    R = bunny_ear_radius(theta, phi, R0=R0, eps=eps, phi0=phi0, sigma=sigma)
    rho = r * R

    x = np.empty((len(q), 3), dtype=float)
    x[:, 0] = rho * np.sin(phi) * np.cos(theta)
    x[:, 1] = rho * np.sin(phi) * np.sin(theta)
    x[:, 2] = rho * np.cos(phi)

    return x


def curvilinear_from_body_positions(x, R0=1.0, eps=0.55, phi0=0.35, sigma=0.20):
    """Convert body-frame Cartesian positions to q = (r, theta, phi)."""
    x = np.asarray(x, dtype=float)
    rho_phys = np.linalg.norm(x, axis=-1)

    r = np.zeros_like(rho_phys)
    theta = np.zeros_like(rho_phys)
    phi = np.zeros_like(rho_phys)

    nonzero = rho_phys > 0.0
    if np.any(nonzero):
        theta[nonzero] = np.mod(np.arctan2(x[..., 1][nonzero], x[..., 0][nonzero]), TWO_PI)
        cphi = x[..., 2][nonzero] / rho_phys[nonzero]
        bad = (cphi < -1.0 - 1.0e-12) | (cphi > 1.0 + 1.0e-12)
        if np.any(bad):
            raise ValueError("bad z/rho while converting body positions to curvilinear display")
        cphi = np.minimum(1.0, np.maximum(-1.0, cphi))
        phi[nonzero] = np.arccos(cphi)
        R = bunny_ear_radius(theta[nonzero], phi[nonzero], R0=R0, eps=eps, phi0=phi0, sigma=sigma)
        r[nonzero] = rho_phys[nonzero] / R

    return np.column_stack((r, theta, phi))


def make_angular_sampler(
    R0=1.0,
    eps=0.55,
    phi0=0.35,
    sigma=0.20,
    n_theta=480,
    n_phi=320,
):
    """
    Build a discrete angular sampler for approximately uniform volume samples.

    Direction weights are proportional to R(theta, phi)^3 sin(phi), the volume
    swept by each solid-angle ray of a star-shaped body.
    """
    theta = np.linspace(0.0, TWO_PI, int(n_theta), endpoint=False)
    phi = np.linspace(0.0, np.pi, int(n_phi), endpoint=True)
    Theta, Phi = np.meshgrid(theta, phi)
    R = bunny_ear_radius(Theta, Phi, R0=R0, eps=eps, phi0=phi0, sigma=sigma)

    weights = np.maximum(R, 0.0) ** 3 * np.sin(Phi)
    weights[0, :] = 0.0
    weights[-1, :] = 0.0

    flat_weights = weights.ravel()
    total = np.sum(flat_weights)
    if total <= 0.0:
        raise ValueError("angular sampler has zero total weight")

    cdf = np.cumsum(flat_weights / total)
    cdf[-1] = 1.0

    return theta, phi, cdf


def _sample_angles(rng, theta_table, phi_table, cdf_table, n):
    ids = np.searchsorted(cdf_table, rng.random(int(n)), side="left")
    n_theta = len(theta_table)
    phi_ids = ids // n_theta
    theta_ids = ids % n_theta

    dtheta = TWO_PI / n_theta
    dphi = np.pi / max(1, len(phi_table) - 1)

    theta = theta_table[theta_ids] + rng.random(int(n)) * dtheta
    theta = theta % TWO_PI
    phi = phi_table[phi_ids] + (rng.random(int(n)) - 0.5) * dphi
    phi = np.clip(phi, 0.0, np.pi)

    return theta, phi


def minimum_neighbour_distance_3d(x):
    if len(x) < 2:
        return np.inf, np.full(len(x), np.inf)

    if cKDTree is not None:
        tree = cKDTree(x)
        dist, _ = tree.query(x, k=2)
        return float(dist[:, 1].min()), dist[:, 1]

    nearest = np.full(len(x), np.inf)
    for i in range(len(x)):
        d2 = np.sum((x - x[i]) ** 2, axis=1)
        d2[i] = np.inf
        nearest[i] = np.sqrt(np.min(d2))
    return float(np.min(nearest)), nearest


def filling_3d(
    N,
    particle_radius,
    fraction=1.0,
    r_min=0.05,
    r_max=0.95,
    safety=1.05,
    max_trials=500000,
    seed=1,
    batch_size=None,
    R0=1.0,
    eps=0.55,
    phi0=0.35,
    sigma=0.20,
):
    """
    Generate non-overlapping particles inside the 3D star-shaped container.

    fraction = 1.0 fills the whole volume.
    fraction = 0.5 fills the lower physical half by z-coordinate.
    fraction = 0.75 fills the lower three quarters by z-coordinate.
    """
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError("fraction must satisfy 0 < fraction <= 1")
    if r_min < 0.0 or r_max > 1.0 or r_min >= r_max:
        raise ValueError("require 0 <= r_min < r_max <= 1")

    rng = np.random.default_rng(seed)
    min_dist = safety * 2.0 * particle_radius
    min_dist2 = min_dist * min_dist
    cell_size = min_dist

    if batch_size is None:
        batch_size = max(2048, 8 * N)
    batch_size = int(batch_size)

    theta_table, phi_table, cdf_table = make_angular_sampler(
        R0=R0,
        eps=eps,
        phi0=phi0,
        sigma=sigma,
    )

    X, Y, Z, _, _, _ = bunny_ear_surface(
        R0=R0,
        eps=eps,
        phi0=phi0,
        sigma=sigma,
    )
    z_min = float(np.min(Z))
    z_max = float(np.max(Z))
    z_limit = z_min + fraction * (z_max - z_min)

    q_acc = np.empty((N, 3), dtype=float)
    x_acc = np.empty((N, 3), dtype=float)
    cells = {}

    def cell_key(xp):
        return (
            int(math.floor(xp[0] / cell_size)),
            int(math.floor(xp[1] / cell_size)),
            int(math.floor(xp[2] / cell_size)),
        )

    def can_accept(xp):
        cx, cy, cz = cell_key(xp)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    ids = cells.get((cx + dx, cy + dy, cz + dz))
                    if ids is None:
                        continue
                    for j in ids:
                        d = xp - x_acc[j]
                        if np.dot(d, d) <= min_dist2:
                            return False
        return True

    count = 0
    trials = 0

    while count < N and trials < max_trials:
        n_try = min(batch_size, max_trials - trials)
        trials += n_try

        u = rng.random(n_try)
        r = (r_min**3 + u * (r_max**3 - r_min**3)) ** (1.0 / 3.0)
        theta, phi = _sample_angles(rng, theta_table, phi_table, cdf_table, n_try)

        q_batch = np.column_stack((r, theta, phi))
        x_batch = map_particles_3d(
            q_batch,
            R0=R0,
            eps=eps,
            phi0=phi0,
            sigma=sigma,
        )
        keep = np.nonzero(x_batch[:, 2] <= z_limit)[0]

        for idx in keep:
            if count >= N:
                break

            xp = x_batch[idx]
            if can_accept(xp):
                q_acc[count] = q_batch[idx]
                x_acc[count] = xp

                key = cell_key(xp)
                cells.setdefault(key, []).append(count)
                count += 1

    q_acc = q_acc[:count]
    x_acc = x_acc[:count]

    if count < N:
        warnings.warn(
            f"Only {count} particles were placed out of requested N={N}. "
            "Try reducing N, reducing particle_radius, increasing fraction, "
            "or increasing max_trials."
        )

    d_min, nearest = minimum_neighbour_distance_3d(x_acc)
    info = {
        "requested": N,
        "placed": count,
        "trials": trials,
        "particle_diameter": 2.0 * particle_radius,
        "minimum_allowed_distance": min_dist,
        "minimum_actual_distance": d_min,
        "fraction": fraction,
        "z_limit": z_limit,
        "region": f"physical lower fraction {fraction}",
        "method": "batch + physical 3D grid",
    }

    return q_acc, x_acc, nearest, info


def make_initial_conditions_3d(
    N,
    particle_radius,
    particle_mass=1.0,
    velocity=(0.0, 0.0, 0.0),
    fraction=1.0,
    r_min=0.05,
    r_max=0.95,
    safety=1.05,
    max_trials=500000,
    seed=1,
    R0=1.0,
    eps=0.55,
    phi0=0.35,
    sigma=0.20,
):
    """Create q0, p0, x0, v0, m, rad, group_mobile, nearest, and info."""
    q0, x0, nearest, info = filling_3d(
        N=N,
        particle_radius=particle_radius,
        fraction=fraction,
        r_min=r_min,
        r_max=r_max,
        safety=safety,
        max_trials=max_trials,
        seed=seed,
        R0=R0,
        eps=eps,
        phi0=phi0,
        sigma=sigma,
    )

    p0 = np.zeros_like(q0)
    p0[:, 0] = velocity[0]
    p0[:, 1] = velocity[1]
    p0[:, 2] = velocity[2]

    v0 = np.zeros_like(x0)
    m = np.full(len(q0), particle_mass, dtype=float)
    rad = np.full(len(q0), particle_radius, dtype=float)
    group_mobile = np.array([0, len(q0)], dtype=np.int64)

    return q0, p0, x0, v0, m, rad, group_mobile, nearest, info


def set_axes_equal_3d(ax, X, Y, Z, margin=0.05):
    """Set equal scale on a Matplotlib 3D axis."""
    mins = np.array([np.min(X), np.min(Y), np.min(Z)], dtype=float)
    maxs = np.array([np.max(X), np.max(Y), np.max(Z)], dtype=float)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius *= 1.0 + float(margin)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def curvilinear_display_points(q):
    """Map q = (r, theta, phi) to display coordinates (theta, phi, r)."""
    q = np.asarray(q, dtype=float)

    y = np.empty((len(q), 3), dtype=float)
    y[:, 0] = q[:, 1] % TWO_PI
    y[:, 1] = q[:, 2]
    y[:, 2] = q[:, 0]

    return y


def plot_initial_configuration_3d(
    q,
    x,
    R0=1.0,
    eps=0.55,
    phi0=0.35,
    sigma=0.20,
    particle_radius=None,
    s=8,
    surface_alpha=0.18,
):
    """Plot physical and curvilinear initial particles."""
    import matplotlib.pyplot as plt

    X, Y, Z, _, _, _ = bunny_ear_surface(
        R0=R0,
        eps=eps,
        phi0=phi0,
        sigma=sigma,
    )
    xq = curvilinear_display_points(q)
    theta_plane = np.linspace(0.0, TWO_PI, 80)
    phi_plane = np.linspace(0.0, np.pi, 50)
    Theta_q, Phi_q = np.meshgrid(theta_plane, phi_plane)
    R_q = np.ones_like(Theta_q)

    fig = plt.figure(figsize=(15, 4.8))
    ax_phys = fig.add_subplot(1, 3, 1, projection="3d")
    ax_proj = fig.add_subplot(1, 3, 2)
    ax_curv = fig.add_subplot(1, 3, 3, projection="3d")

    ax_phys.plot_surface(
        X,
        Y,
        Z,
        linewidth=0,
        antialiased=True,
        alpha=surface_alpha,
        color="0.75",
    )
    ax_phys.scatter(x[:, 0], x[:, 1], x[:, 2], s=s, color="tab:blue")
    if particle_radius is not None and len(x) > 0:
        ax_phys.set_title(f"physical space, radius = {particle_radius:g}")
    else:
        ax_phys.set_title("physical space")
    ax_phys.set_xlabel("x")
    ax_phys.set_ylabel("y")
    ax_phys.set_zlabel("z")
    set_axes_equal_3d(
        ax_phys,
        np.r_[X.ravel(), x[:, 0]],
        np.r_[Y.ravel(), x[:, 1]],
        np.r_[Z.ravel(), x[:, 2]],
    )

    ax_proj.plot(X[:, 0], Z[:, 0], linewidth=1.0, color="black")
    ax_proj.plot(X[:, X.shape[1] // 2], Z[:, X.shape[1] // 2], linewidth=1.0, color="black")
    ax_proj.scatter(x[:, 0], x[:, 2], s=s, color="tab:blue")
    ax_proj.set_aspect("equal", adjustable="box")
    ax_proj.set_xlabel("x")
    ax_proj.set_ylabel("z")
    ax_proj.grid(True)
    ax_proj.set_title("physical projection on y = 0")

    ax_curv.plot_surface(
        Theta_q,
        Phi_q,
        R_q,
        linewidth=0,
        antialiased=True,
        alpha=surface_alpha,
        color="0.75",
    )
    ax_curv.scatter(xq[:, 0], xq[:, 1], xq[:, 2], s=s, color="tab:red")
    ax_curv.set_title("curvilinear space")
    ax_curv.set_xlabel("theta")
    ax_curv.set_ylabel("phi")
    ax_curv.set_zlabel("r")
    ax_curv.set_xlim(0.0, TWO_PI)
    ax_curv.set_ylim(0.0, np.pi)
    ax_curv.set_zlim(0.0, 1.05)

    plt.tight_layout()
    return fig, (ax_phys, ax_proj, ax_curv)


def _resolve_snapshot_frame(t, frame=None, time=None):
    if frame is not None:
        return int(frame)
    if time is None:
        raise ValueError("provide either frame or time")
    return int(np.argmin(np.abs(np.asarray(t, dtype=float) - float(time))))


def _plot_physical_snapshot_3d(
    ax,
    ti,
    x_frame,
    surface,
    omega=0.0,
    s=8,
    surface_alpha=0.18,
    title=None,
):
    surface_lab = rotate_y_points(surface.reshape(-1, 3), omega * ti).reshape(surface.shape)
    ax.plot_surface(
        surface_lab[:, :, 0],
        surface_lab[:, :, 1],
        surface_lab[:, :, 2],
        linewidth=0,
        antialiased=True,
        alpha=surface_alpha,
        color="0.75",
    )
    ax.scatter(x_frame[:, 0], x_frame[:, 1], x_frame[:, 2], s=s, color="tab:blue")
    ax.set_title(title if title is not None else f"physical space, t = {ti:.3g}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    set_axes_equal_3d(
        ax,
        np.r_[surface_lab[:, :, 0].ravel(), x_frame[:, 0]],
        np.r_[surface_lab[:, :, 1].ravel(), x_frame[:, 1]],
        np.r_[surface_lab[:, :, 2].ravel(), x_frame[:, 2]],
    )


def _plot_curvilinear_snapshot_3d(
    ax,
    ti,
    q_frame,
    R0=1.0,
    eps=0.55,
    phi0=0.35,
    sigma=0.20,
    s=8,
    surface_alpha=0.18,
    title=None,
):
    theta_plane = np.linspace(0.0, TWO_PI, 80)
    phi_plane = np.linspace(0.0, np.pi, 50)
    Theta_q, Phi_q = np.meshgrid(theta_plane, phi_plane)
    R_q = np.ones_like(Theta_q)
    xq = curvilinear_display_points(q_frame)

    ax.plot_surface(
        Theta_q,
        Phi_q,
        R_q,
        linewidth=0,
        antialiased=True,
        alpha=surface_alpha,
        color="0.75",
    )
    ax.scatter(xq[:, 0], xq[:, 1], xq[:, 2], s=s, color="tab:red")
    ax.set_title(title if title is not None else f"curvilinear space, t = {ti:.3g}")
    ax.set_xlabel("theta")
    ax.set_ylabel("phi")
    ax.set_zlabel("r")
    ax.set_xlim(0.0, TWO_PI)
    ax.set_ylim(0.0, np.pi)
    ax.set_zlim(0.0, 1.05)


def plot_two_snapshot_comparison_3d(
    t,
    q,
    x_lab,
    frame_a=None,
    frame_b=None,
    time_a=None,
    time_b=None,
    x_body=None,
    space="both",
    R0=1.0,
    eps=0.55,
    phi0=0.35,
    sigma=0.20,
    omega=0.0,
    s=8,
    surface_alpha=0.18,
    figsize=None,
):
    """
    Plot two chosen saved timesteps and return fig, axes.

    Use either frame_a/frame_b or time_a/time_b.  This function does not save.
    Set space to "physical", "curvilinear", or "both".
    """
    import matplotlib.pyplot as plt

    frame_a = _resolve_snapshot_frame(t, frame=frame_a, time=time_a)
    frame_b = _resolve_snapshot_frame(t, frame=frame_b, time=time_b)
    frames = (frame_a, frame_b)
    times = (float(t[frame_a]), float(t[frame_b]))

    X, Y, Z, _, _, _ = bunny_ear_surface(
        R0=R0,
        eps=eps,
        phi0=phi0,
        sigma=sigma,
    )
    surface = np.stack((X, Y, Z), axis=-1)

    if space == "both":
        if figsize is None:
            figsize = (12.0, 9.0)
        fig = plt.figure(figsize=figsize)
        axes = np.empty((2, 2), dtype=object)
        for col, frame_id in enumerate(frames):
            ti = times[col]
            axes[0, col] = fig.add_subplot(2, 2, col + 1, projection="3d")
            _plot_physical_snapshot_3d(
                axes[0, col],
                ti,
                np.asarray(x_lab[frame_id], dtype=float),
                surface,
                omega=omega,
                s=s,
                surface_alpha=surface_alpha,
            )

            axes[1, col] = fig.add_subplot(2, 2, col + 3, projection="3d")
            if x_body is None:
                q_frame = np.asarray(q[frame_id], dtype=float)
            else:
                q_frame = curvilinear_from_body_positions(
                    x_body[frame_id],
                    R0=R0,
                    eps=eps,
                    phi0=phi0,
                    sigma=sigma,
                )
            _plot_curvilinear_snapshot_3d(
                axes[1, col],
                ti,
                q_frame,
                R0=R0,
                eps=eps,
                phi0=phi0,
                sigma=sigma,
                s=s,
                surface_alpha=surface_alpha,
            )
    elif space in ("physical", "curvilinear"):
        if figsize is None:
            figsize = (10.0, 4.8)
        fig = plt.figure(figsize=figsize)
        axes = np.empty(2, dtype=object)
        for col, frame_id in enumerate(frames):
            ti = times[col]
            axes[col] = fig.add_subplot(1, 2, col + 1, projection="3d")
            if space == "physical":
                _plot_physical_snapshot_3d(
                    axes[col],
                    ti,
                    np.asarray(x_lab[frame_id], dtype=float),
                    surface,
                    omega=omega,
                    s=s,
                    surface_alpha=surface_alpha,
                )
            else:
                if x_body is None:
                    q_frame = np.asarray(q[frame_id], dtype=float)
                else:
                    q_frame = curvilinear_from_body_positions(
                        x_body[frame_id],
                        R0=R0,
                        eps=eps,
                        phi0=phi0,
                        sigma=sigma,
                    )
                _plot_curvilinear_snapshot_3d(
                    axes[col],
                    ti,
                    q_frame,
                    R0=R0,
                    eps=eps,
                    phi0=phi0,
                    sigma=sigma,
                    s=s,
                    surface_alpha=surface_alpha,
                )
    else:
        raise ValueError("space must be 'physical', 'curvilinear', or 'both'")

    fig.tight_layout()
    return fig, axes


def rotation_y_matrix(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def rotate_y_points(x, angle):
    return np.asarray(x, dtype=float) @ rotation_y_matrix(angle).T


def _project_points(points, view_matrix, center, scale):
    p = np.asarray(points, dtype=float) @ view_matrix.T
    u = center[0] + scale * p[:, 0]
    v = center[1] - scale * p[:, 1]
    depth = p[:, 2]
    return np.column_stack((u, v, depth))


def _make_view_matrix(elev_deg=22.0, azim_deg=-55.0):
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    ce = np.cos(elev)
    se = np.sin(elev)
    ca = np.cos(azim)
    sa = np.sin(azim)
    return np.array(
        [
            [ca, -sa, 0.0],
            [se * sa, se * ca, ce],
            [ce * sa, ce * ca, -se],
        ],
        dtype=float,
    )


def _draw_projected_points(draw, projected, radius, fill):
    order = np.argsort(projected[:, 2])
    for idx in order:
        u, v, _ = projected[idx]
        draw.ellipse(
            [u - radius, v - radius, u + radius, v + radius],
            fill=fill,
        )


def _draw_projected_polyline(draw, points, view_matrix, center, scale, fill, width=1):
    projected = _project_points(points, view_matrix, center, scale)
    xy = [(float(u), float(v)) for u, v, _ in projected]
    if len(xy) >= 2:
        draw.line(xy, fill=fill, width=width)


_GIF_FONT_CACHE = {}


def _get_gif_font(size=26):
    from PIL import ImageFont

    size = int(size)
    if size in _GIF_FONT_CACHE:
        return _GIF_FONT_CACHE[size]

    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype(name, size)
            _GIF_FONT_CACHE[size] = font
            return font
        except Exception:
            pass

    font = ImageFont.load_default()
    _GIF_FONT_CACHE[size] = font
    return font


def _draw_projected_label(draw, point, view_matrix, center, scale, text, fill=(20, 20, 20), size=26):
    projected = _project_points(np.asarray(point, dtype=float).reshape(1, 3), view_matrix, center, scale)
    _draw_text(draw, (float(projected[0, 0]), float(projected[0, 1])), text, fill=fill, size=size)


def _draw_text(draw, xy, text, fill=(20, 20, 20), size=26):
    try:
        draw.text(xy, text, fill=fill, font=_get_gif_font(size))
    except Exception:
        pass


def save_rotating_3d_simulation_gif(
    path,
    t,
    q,
    x_lab,
    R0=1.0,
    eps=0.55,
    phi0=0.35,
    sigma=0.20,
    omega=0.0,
    frame_stride=1,
    fps=10,
    size=(1200, 560),
    particle_radius_px=3,
    n_theta=64,
    n_phi=40,
    x_body=None,
    title_font_size=30,
    label_font_size=26,
    time_font_size=28,
):
    """
    Save a quick GIF with physical rotating view and fixed curvilinear view.

    This intentionally uses PIL instead of Matplotlib writers so it can run in
    lightweight environments.
    """
    from PIL import Image, ImageDraw

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frame_ids = np.arange(0, len(t), max(1, int(frame_stride)))
    if frame_ids[-1] != len(t) - 1:
        frame_ids = np.append(frame_ids, len(t) - 1)

    X, Y, Z, _, _, _ = bunny_ear_surface(
        R0=R0,
        eps=eps,
        phi0=phi0,
        sigma=sigma,
        n_theta=n_theta,
        n_phi=n_phi,
    )
    surface = np.stack((X, Y, Z), axis=-1)

    theta = np.linspace(0.0, TWO_PI, n_theta)
    phi = np.linspace(0.0, np.pi, n_phi)
    Theta, Phi = np.meshgrid(theta, phi)
    curv_plane = np.stack((Theta, Phi, np.ones_like(Theta)), axis=-1)
    curv_center = np.array([0.5 * TWO_PI, 0.5 * np.pi, 0.5], dtype=float)
    curv_plane_plot = curv_plane - curv_center

    theta_edge = np.linspace(0.0, TWO_PI, n_theta)
    phi_edge = np.linspace(0.0, np.pi, n_phi)
    r_edge = np.linspace(0.0, 1.0, 8)
    curv_box_edges = []
    for phi_value in (0.0, np.pi):
        for r_value in (0.0, 1.0):
            curv_box_edges.append(
                np.column_stack((
                    theta_edge,
                    np.full_like(theta_edge, phi_value),
                    np.full_like(theta_edge, r_value),
                ))
            )
    for theta_value in (0.0, TWO_PI):
        for r_value in (0.0, 1.0):
            curv_box_edges.append(
                np.column_stack((
                    np.full_like(phi_edge, theta_value),
                    phi_edge,
                    np.full_like(phi_edge, r_value),
                ))
            )
    for theta_value in (0.0, TWO_PI):
        for phi_value in (0.0, np.pi):
            curv_box_edges.append(
                np.column_stack((
                    np.full_like(r_edge, theta_value),
                    np.full_like(r_edge, phi_value),
                    r_edge,
                ))
            )
    curv_box_edges_plot = [edge - curv_center for edge in curv_box_edges]
    curv_labels = (
        (np.array([0.5 * TWO_PI, -0.18, 0.0], dtype=float) - curv_center, "theta"),
        (np.array([TWO_PI + 0.20, 0.5 * np.pi, 0.0], dtype=float) - curv_center, "phi"),
        (np.array([TWO_PI + 0.18, np.pi + 0.12, 0.5], dtype=float) - curv_center, "r"),
    )

    view_phys = _make_view_matrix(elev_deg=20.0, azim_deg=-50.0)
    view_curv = _make_view_matrix(elev_deg=24.0, azim_deg=-58.0)
    left_center = (0.25 * size[0], 0.55 * size[1])
    right_center = (0.75 * size[0], 0.52 * size[1])
    left_scale = 0.18 * min(size)
    right_scale = 0.13 * min(size)

    frames = []
    duration_ms = int(round(1000.0 / fps))

    for frame_id in frame_ids:
        ti = float(t[frame_id])
        img = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(img, "RGBA")

        _draw_text(draw, (20, 16), "physical space", fill=(20, 20, 20), size=title_font_size)
        _draw_text(draw, (size[0] // 2 + 20, 16), "curvilinear space", fill=(20, 20, 20), size=title_font_size)
        _draw_text(draw, (size[0] - 160, size[1] - 38), f"t = {ti:.2f}", fill=(20, 20, 20), size=time_font_size)
        draw.line([(size[0] // 2, 0), (size[0] // 2, size[1])], fill=(210, 210, 210), width=1)

        surface_lab = rotate_y_points(surface.reshape(-1, 3), omega * ti).reshape(surface.shape)
        for row in surface_lab[::4]:
            _draw_projected_polyline(draw, row, view_phys, left_center, left_scale, fill=(150, 150, 150, 105))
        for col in np.swapaxes(surface_lab, 0, 1)[::6]:
            _draw_projected_polyline(draw, col, view_phys, left_center, left_scale, fill=(150, 150, 150, 105))

        projected_particles = _project_points(
            x_lab[frame_id],
            view_phys,
            left_center,
            left_scale,
        )
        _draw_projected_points(
            draw,
            projected_particles,
            particle_radius_px,
            fill=(31, 119, 180, 210),
        )

        for row in curv_plane_plot[::4]:
            _draw_projected_polyline(draw, row, view_curv, right_center, right_scale, fill=(150, 150, 150, 115))
        for col in np.swapaxes(curv_plane_plot, 0, 1)[::6]:
            _draw_projected_polyline(draw, col, view_curv, right_center, right_scale, fill=(150, 150, 150, 115))
        for edge in curv_box_edges_plot:
            _draw_projected_polyline(draw, edge, view_curv, right_center, right_scale, fill=(95, 95, 95, 150))
        for point, label in curv_labels:
            _draw_projected_label(draw, point, view_curv, right_center, right_scale, label, size=label_font_size)

        if x_body is None:
            q_frame = np.asarray(q[frame_id], dtype=float)
        else:
            q_frame = curvilinear_from_body_positions(
                x_body[frame_id],
                R0=R0,
                eps=eps,
                phi0=phi0,
                sigma=sigma,
            )
        curv_points = np.column_stack((q_frame[:, 1] % TWO_PI, q_frame[:, 2], q_frame[:, 0]))
        curv_points_plot = curv_points - curv_center
        projected_curv = _project_points(
            curv_points_plot,
            view_curv,
            right_center,
            right_scale,
        )
        _draw_projected_points(
            draw,
            projected_curv,
            particle_radius_px,
            fill=(214, 39, 40, 210),
        )

        frames.append(img)

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return path
