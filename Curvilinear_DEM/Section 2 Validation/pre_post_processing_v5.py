"""
Pre- and post-processing helpers for the Fourier bumpy-trapezium simulation.

This file keeps geometry construction, initial particle placement, plotting,
and animation outside the fast Numba solver.
"""

import warnings
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

try:
    from IPython.display import HTML
except Exception:  # pragma: no cover
    HTML = None


TWO_PI = 2.0 * np.pi


############################################################
# Fourier geometry
############################################################



def fourier_curve(theta, a, b, c, d):
    """
    Boundary curve B(theta) = (X(theta), Y(theta)).

    Vectorised version. For an array of theta values, all Fourier modes are
    evaluated in one NumPy block instead of a Python loop over modes.
    """
    theta = np.asarray(theta, dtype=float)
    scalar = theta.ndim == 0
    theta = np.atleast_1d(theta)

    K = len(a) - 1
    if K == 0:
        X = a[0] * np.ones_like(theta)
        Y = c[0] * np.ones_like(theta)
    else:
        k = np.arange(1, K + 1, dtype=float)[:, None]
        kt = k * theta[None, :]
        C = np.cos(kt)
        S = np.sin(kt)

        X = a[0] + a[1:] @ C + b[1:] @ S
        Y = c[0] + c[1:] @ C + d[1:] @ S

    if scalar:
        return float(X[0]), float(Y[0])

    return X, Y


def fourier_curve_derivatives(theta, a, b, c, d):
    """
    Return B, B_theta, and B_thetatheta.

    Vectorised version. It is faster for the boundary checks, plotting,
    inverse visualisation, and particle preprocessing.
    """
    theta = np.asarray(theta, dtype=float)
    scalar = theta.ndim == 0
    theta = np.atleast_1d(theta)

    K = len(a) - 1
    if K == 0:
        X = a[0] * np.ones_like(theta)
        Y = c[0] * np.ones_like(theta)
        Xt = np.zeros_like(theta)
        Yt = np.zeros_like(theta)
        Xtt = np.zeros_like(theta)
        Ytt = np.zeros_like(theta)
    else:
        k = np.arange(1, K + 1, dtype=float)[:, None]
        kt = k * theta[None, :]
        C = np.cos(kt)
        S = np.sin(kt)

        ak = a[1:, None]
        bk = b[1:, None]
        ck = c[1:, None]
        dk = d[1:, None]

        X = a[0] + np.sum(ak * C + bk * S, axis=0)
        Y = c[0] + np.sum(ck * C + dk * S, axis=0)

        Xt = np.sum(-k * ak * S + k * bk * C, axis=0)
        Yt = np.sum(-k * ck * S + k * dk * C, axis=0)

        k2 = k * k
        Xtt = np.sum(-k2 * ak * C - k2 * bk * S, axis=0)
        Ytt = np.sum(-k2 * ck * C - k2 * dk * S, axis=0)

    if scalar:
        return (
            float(X[0]), float(Y[0]),
            float(Xt[0]), float(Yt[0]),
            float(Xtt[0]), float(Ytt[0]),
        )

    return X, Y, Xt, Yt, Xtt, Ytt

def boundary_curve(a, b, c, d, n=2000):
    theta = np.linspace(0.0, TWO_PI, n, endpoint=True)
    X, Y = fourier_curve(theta, a, b, c, d)
    return np.column_stack((X, Y))


def is_valid_boundary(a, b, c, d, n_theta=5000, min_det=0.0):
    """
    Check the determinant condition for the radial map.

    For x = r B(theta), det(J) = r * (X Y_theta - Y X_theta).
    A valid orientation-preserving boundary has X Y_theta - Y X_theta > 0.
    """
    theta = np.linspace(0.0, TWO_PI, n_theta, endpoint=False)
    X, Y, Xt, Yt, _, _ = fourier_curve_derivatives(theta, a, b, c, d)
    det_theta = X * Yt - Y * Xt
    return bool(np.all(det_theta > min_det))


def boundary_det_min(a, b, c, d, n_theta=5000):
    theta = np.linspace(0.0, TWO_PI, n_theta, endpoint=False)
    X, Y, Xt, Yt, _, _ = fourier_curve_derivatives(theta, a, b, c, d)
    return float(np.min(X * Yt - Y * Xt))


############################################################
# Geometry construction
############################################################


def symmetric_bumpy_trapezium(
    H,
    W,
    K,
    bump_freq=None,
    bump_amp=0.0,
    n_sample=4000,
):
    """
    Build the y-axis symmetric bumpy trapezium used in the notebook.

    Returns Fourier coefficients a, b, c, d for

        X(theta) = a0 + sum_k a_k cos(k theta) + b_k sin(k theta)
        Y(theta) = c0 + sum_k c_k cos(k theta) + d_k sin(k theta)
    """
    Wb, Wt = W

    verts = np.array([
        [-Wb / 2.0, -H / 2.0],
        [ Wb / 2.0, -H / 2.0],
        [ Wt / 2.0,  H / 2.0],
        [-Wt / 2.0,  H / 2.0],
    ])

    def ray_intersect_polygon(theta, vertices):
        direction = np.array([np.cos(theta), np.sin(theta)])
        min_dist = np.inf

        n = len(vertices)
        for i in range(n):
            A = vertices[i]
            B = vertices[(i + 1) % n]
            edge = B - A

            M = np.column_stack((edge, -direction))
            rhs = -A

            if abs(np.linalg.det(M)) < 1.0e-12:
                continue

            t, s = np.linalg.solve(M, rhs)

            if 0.0 <= t <= 1.0 and s > 0.0:
                min_dist = min(min_dist, s)

        if min_dist == np.inf:
            raise RuntimeError("No ray-polygon intersection found.")

        return min_dist

    def enforce_y_axis_symmetry(a, b, c, d):
        a = a.copy()
        b = b.copy()
        c = c.copy()
        d = d.copy()

        a[0] = 0.0

        for k in range(1, len(a)):
            if k % 2 == 0:
                a[k] = 0.0
                d[k] = 0.0
            else:
                b[k] = 0.0
                c[k] = 0.0

        return a, b, c, d

    theta = np.linspace(0.0, TWO_PI, n_sample, endpoint=False)
    R = np.array([ray_intersect_polygon(th, verts) for th in theta])

    X = R * np.cos(theta)
    Y = R * np.sin(theta)

    a = np.zeros(K + 1)
    b = np.zeros(K + 1)
    c = np.zeros(K + 1)
    d = np.zeros(K + 1)

    d_theta = TWO_PI / n_sample

    a[0] = np.sum(X) * d_theta / TWO_PI
    c[0] = np.sum(Y) * d_theta / TWO_PI

    for k in range(1, K + 1):
        a[k] = (2.0 / TWO_PI) * np.sum(X * np.cos(k * theta)) * d_theta
        b[k] = (2.0 / TWO_PI) * np.sum(X * np.sin(k * theta)) * d_theta
        c[k] = (2.0 / TWO_PI) * np.sum(Y * np.cos(k * theta)) * d_theta
        d[k] = (2.0 / TWO_PI) * np.sum(Y * np.sin(k * theta)) * d_theta

    a, b, c, d = enforce_y_axis_symmetry(a, b, c, d)

    if bump_freq is not None and bump_amp != 0.0:
        k = int(bump_freq)

        if k > K:
            raise ValueError("bump_freq must be <= K")

        if k % 2 == 0:
            b[k] += bump_amp
            c[k] += bump_amp
        else:
            a[k] += bump_amp
            d[k] += bump_amp

        a, b, c, d = enforce_y_axis_symmetry(a, b, c, d)

    return a, b, c, d


############################################################
# Initial particles
############################################################


def area_weight(theta, a, b, c, d):
    X, Y, Xt, Yt, _, _ = fourier_curve_derivatives(theta, a, b, c, d)
    return np.abs(X * Yt - Y * Xt)


def make_theta_sampler(a, b, c, d, n=10000):
    theta = np.linspace(0.0, TWO_PI, n, endpoint=False)
    w = area_weight(theta, a, b, c, d)
    cdf = np.cumsum(np.maximum(w, 0.0))
    cdf = cdf / cdf[-1]
    return theta, cdf


def sample_theta(rng, theta_table, cdf_table):
    return np.interp(rng.random(), cdf_table, theta_table)


def map_particles(q, a, b, c, d):
    """
    Map q = (r, theta) to physical x = (x, y) in the container frame.
    """
    q = np.asarray(q, dtype=float)
    r = q[:, 0]
    theta = q[:, 1]

    X, Y = fourier_curve(theta, a, b, c, d)

    x = np.empty((len(q), 2))
    x[:, 0] = r * X
    x[:, 1] = r * Y

    return x


def curvilinear_to_physical_numpy(q, p, a, b, c, d, xA=(0.0, 0.0), omega=0.0, t=0.0):
    """
    Map q,p to laboratory-frame physical x,v.
    """
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    xA = np.asarray(xA, dtype=float)

    r = q[:, 0]
    theta = q[:, 1]
    rdot = p[:, 0]
    thetadot = p[:, 1]

    X, Y, Xt, Yt, _, _ = fourier_curve_derivatives(theta, a, b, c, d)

    shift = xA * np.sin(omega * t)
    shift_v = xA * omega * np.cos(omega * t)

    x = np.empty((len(q), 2))
    v = np.empty((len(q), 2))

    x[:, 0] = shift[0] + r * X
    x[:, 1] = shift[1] + r * Y

    v[:, 0] = shift_v[0] + X * rdot + r * Xt * thetadot
    v[:, 1] = shift_v[1] + Y * rdot + r * Yt * thetadot

    return x, v


def minimum_neighbour_distance(x):
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



def filling(
    N,
    a, b, c, d,
    particle_radius,
    fraction=0.5,
    r_min=0.05,
    r_max=0.95,
    safety=1.05,
    max_trials=500000,
    seed=1,
    batch_size=None,
):
    """
    Generate particles in the lower fraction of the physical domain.

    fraction = 1.0 fills the whole domain.
    fraction = 0.5 fills the lower physical half.
    fraction = 0.75 fills the lower three quarters.

    Fast version:
    - candidates are generated in batches;
    - Fourier mapping is vectorised;
    - overlap rejection uses a physical-space grid instead of checking every
      previously accepted particle.
    """
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError("fraction must satisfy 0 < fraction <= 1")

    rng = np.random.default_rng(seed)

    min_dist = safety * 2.0 * particle_radius
    min_dist2 = min_dist * min_dist
    cell_size = min_dist

    if batch_size is None:
        batch_size = max(1024, 8 * N)
    batch_size = int(batch_size)

    theta_table, cdf_table = make_theta_sampler(a, b, c, d)

    boundary = boundary_curve(a, b, c, d, n=2000)
    y_min = np.min(boundary[:, 1])
    y_max = np.max(boundary[:, 1])
    y_limit = y_min + fraction * (y_max - y_min)

    q_acc = np.empty((N, 2))
    x_acc = np.empty((N, 2))

    cells = {}

    def cell_key(xp):
        return (
            int(math.floor(xp[0] / cell_size)),
            int(math.floor(xp[1] / cell_size)),
        )

    def can_accept(xp):
        cx, cy = cell_key(xp)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                ids = cells.get((cx + dx, cy + dy))
                if ids is None:
                    continue
                for j in ids:
                    ddx = xp[0] - x_acc[j, 0]
                    ddy = xp[1] - x_acc[j, 1]
                    if ddx * ddx + ddy * ddy <= min_dist2:
                        return False
        return True

    count = 0
    trials = 0

    while count < N and trials < max_trials:
        n_try = min(batch_size, max_trials - trials)
        trials += n_try

        u = rng.random(n_try)
        r = np.sqrt(r_min * r_min + u * (r_max * r_max - r_min * r_min))
        theta = np.interp(rng.random(n_try), cdf_table, theta_table)

        q_batch = np.empty((n_try, 2))
        q_batch[:, 0] = r
        q_batch[:, 1] = theta

        x_batch = map_particles(q_batch, a, b, c, d)
        keep = np.nonzero(x_batch[:, 1] <= y_limit)[0]

        for idx in keep:
            if count >= N:
                break

            xp = x_batch[idx]
            if can_accept(xp):
                q_acc[count] = q_batch[idx]
                x_acc[count] = xp

                key = cell_key(xp)
                if key in cells:
                    cells[key].append(count)
                else:
                    cells[key] = [count]

                count += 1

    q_acc = q_acc[:count]
    x_acc = x_acc[:count]

    if count < N:
        warnings.warn(
            f"Only {count} particles were placed out of requested N={N}. "
            "Try reducing N, reducing particle_radius, increasing fraction, "
            "or increasing max_trials."
        )

    d_min, nearest = minimum_neighbour_distance(x_acc)

    info = {
        "requested": N,
        "placed": count,
        "trials": trials,
        "particle_diameter": 2.0 * particle_radius,
        "minimum_allowed_distance": min_dist,
        "minimum_actual_distance": d_min,
        "fraction": fraction,
        "y_limit": float(y_limit),
        "region": f"physical lower fraction {fraction}",
        "method": "batch + physical grid",
    }

    return q_acc, x_acc, nearest, info

def make_initial_conditions(
    N,
    a, b, c, d,
    particle_radius,
    particle_mass=1.0,
    velocity=(0.0, 0.0),
    fraction=0.5,
    r_min=0.05,
    r_max=0.95,
    safety=1.05,
    max_trials=500000,
    seed=1,
):
    """
    Create q0, p0, x0, v0, m, rad, and group_mobile.

    velocity is given directly as a coordinate velocity p0 = (rdot, thetadot).
    """
    q0, x0, nearest, info = filling(
        N=N,
        a=a,
        b=b,
        c=c,
        d=d,
        particle_radius=particle_radius,
        fraction=fraction,
        r_min=r_min,
        r_max=r_max,
        safety=safety,
        max_trials=max_trials,
        seed=seed,
    )

    p0 = np.zeros_like(q0)
    p0[:, 0] = velocity[0]
    p0[:, 1] = velocity[1]

    v0 = curvilinear_to_physical_numpy(
        q0, p0, a, b, c, d,
    )[1]

    m = np.full(len(q0), particle_mass, dtype=float)
    rad = np.full(len(q0), particle_radius, dtype=float)
    group_mobile = np.array([0, len(q0)], dtype=np.int64)

    return q0, p0, x0, v0, m, rad, group_mobile, nearest, info


############################################################
# Plots
############################################################


def plot_initial_configuration(q, x, a, b, c, d, s=0.5):
    boundary = boundary_curve(a, b, c, d)

    fig, (ax_phys, ax_q) = plt.subplots(1, 2, figsize=(10, 4))

    ax_phys.plot(boundary[:, 0], boundary[:, 1], linewidth=0.8, color="black")
    ax_phys.axhline(0.0, linewidth=0.8, linestyle="--", color="gray")
    ax_phys.scatter(x[:, 0], x[:, 1], s=s)
    ax_phys.set_aspect("equal", adjustable="box")
    ax_phys.set_xlabel("x")
    ax_phys.set_ylabel("y")
    ax_phys.grid(True)
    ax_phys.set_title("physical space")

    ax_q.scatter(q[:, 1], q[:, 0], s=s)
    ax_q.axhline(1.0, linewidth=0.8, linestyle="--")
    ax_q.set_xlim(0.0, TWO_PI)
    ax_q.set_ylim(0.0, 1.05)
    ax_q.set_xlabel(r"$\theta$")
    ax_q.set_ylabel(r"$r$")
    ax_q.grid(True)
    ax_q.set_title("curvilinear space")

    plt.tight_layout()
    return fig, (ax_phys, ax_q)


def plot_boundary_validity(a, b, c, d, n_theta=2000):
    theta = np.linspace(0.0, TWO_PI, n_theta, endpoint=False)
    X, Y, Xt, Yt, _, _ = fourier_curve_derivatives(theta, a, b, c, d)
    det_theta = X * Yt - Y * Xt

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(theta, det_theta)
    ax.axhline(0.0, linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$X Y_\theta - Y X_\theta$")
    ax.grid(True)
    ax.set_title("boundary determinant")
    return fig, ax


############################################################
# Animations
############################################################


def _frame_ids(t, stride):
    ids = np.arange(0, len(t), max(1, int(stride)))
    if ids[-1] != len(t) - 1:
        ids = np.append(ids, len(t) - 1)
    return ids


def animate_physical_simulation(
    t,
    x,
    a, b, c, d,
    xA=(0.0, 0.0),
    omega=0.0,
    stride=20,
    interval=30,
    s=8,
    show_trace0=True,
    title="physical simulation",
):
    """
    Animate particles in laboratory physical space with the oscillating boundary.
    """
    xA = np.asarray(xA, dtype=float)
    frame_ids = _frame_ids(t, stride)

    boundary0 = boundary_curve(a, b, c, d, n=1000)

    all_boundary = []
    for ti in t[frame_ids]:
        all_boundary.append(boundary0 + xA * np.sin(omega * ti))
    all_boundary = np.vstack(all_boundary)

    xmin = min(np.min(x[:, :, 0]), np.min(all_boundary[:, 0]))
    xmax = max(np.max(x[:, :, 0]), np.max(all_boundary[:, 0]))
    ymin = min(np.min(x[:, :, 1]), np.min(all_boundary[:, 1]))
    ymax = max(np.max(x[:, :, 1]), np.max(all_boundary[:, 1]))
    margin = 0.08 * max(xmax - xmin, ymax - ymin)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True)

    boundary_line, = ax.plot([], [], linewidth=1.0, label="boundary")
    scat = ax.scatter([], [], s=s, label="particles")
    trace0, = ax.plot([], [], linewidth=1.0, label="particle 0")
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    ax.legend(loc="upper left")

    def init():
        boundary_line.set_data([], [])
        scat.set_offsets(np.empty((0, 2)))
        trace0.set_data([], [])
        time_text.set_text("")
        return boundary_line, scat, trace0, time_text

    def update(frame):
        i = frame_ids[frame]
        ti = t[i]
        boundary = boundary0 + xA * np.sin(omega * ti)

        boundary_line.set_data(boundary[:, 0], boundary[:, 1])
        scat.set_offsets(x[i])

        if show_trace0:
            trace0.set_data(x[: i + 1, 0, 0], x[: i + 1, 0, 1])
        else:
            trace0.set_data([], [])

        time_text.set_text(f"t = {ti:.3f}")
        return boundary_line, scat, trace0, time_text

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        init_func=init,
        interval=interval,
        blit=True,
    )

    return ani



def make_inverse_by_angle(a, b, c, d, n_grid=8000):
    """
    Build a visual inverse for the radial map x = r B(theta).

    This is used only for plotting a translated physical boundary in the
    original curvilinear coordinates. It avoids a full nonlinear inverse.
    """
    theta_grid = np.linspace(0.0, TWO_PI, n_grid + 1)
    X, Y = fourier_curve(theta_grid, a, b, c, d)

    psi_grid = np.unwrap(np.arctan2(Y, X))
    psi0 = psi_grid[0]

    def inverse_points(x):
        x = np.asarray(x, dtype=float)

        psi = np.arctan2(x[:, 1], x[:, 0])
        psi = ((psi - psi0) % TWO_PI) + psi0

        theta = np.interp(psi, psi_grid, theta_grid) % TWO_PI

        Xb, Yb = fourier_curve(theta, a, b, c, d)
        den = Xb * Xb + Yb * Yb

        q = np.empty_like(x)
        q[:, 0] = (x[:, 0] * Xb + x[:, 1] * Yb) / den
        q[:, 1] = theta

        return q

    return inverse_points


def _lab_frame_q_from_q(q_frame, time_value, inverse_points, a, b, c, d, xA, omega):
    """
    Convert one saved q-frame to visual lab-frame curvilinear coordinates.
    """
    x_rel = map_particles(q_frame, a, b, c, d)
    shift = xA * np.sin(omega * time_value)
    return inverse_points(x_rel + shift)


def _insert_nan_at_theta_jumps(theta, r):
    """
    Break curves at the periodic seam so Matplotlib does not draw a line
    across theta = 0.
    """
    theta_plot = [theta[0]]
    r_plot = [r[0]]

    for k in range(1, len(theta)):
        if abs(theta[k] - theta[k - 1]) > np.pi:
            theta_plot.append(np.nan)
            r_plot.append(np.nan)
        theta_plot.append(theta[k])
        r_plot.append(r[k])

    return np.asarray(theta_plot), np.asarray(r_plot)



def animate_curvilinear_simulation(
    t,
    q,
    a, b, c, d,
    xA=(0.0, 0.0),
    omega=0.0,
    stride=20,
    interval=30,
    s=8,
    show_trace0=True,
    show_oscillating_boundary=True,
    show_solver_wall=True,
    show_legend=True,
    wall_color=None,
    title="curvilinear simulation",
):
    """
    Animate particles in (theta, r) space.

    Fast version. Expensive lab-frame inverse maps are precomputed once before
    the animation object is created instead of inside every update call.
    """
    xA = np.asarray(xA, dtype=float)
    frame_ids = _frame_ids(t, stride)
    n_frames = len(frame_ids)

    use_lab_view = bool(show_oscillating_boundary)

    if use_lab_view:
        inverse_points = make_inverse_by_angle(a, b, c, d)
        theta_wall0 = np.linspace(0.0, TWO_PI, 1200, endpoint=True)
        Xw, Yw = fourier_curve(theta_wall0, a, b, c, d)
        wall0 = np.column_stack((Xw, Yw))

        q_display_frames = np.empty((n_frames, q.shape[1], 2), dtype=float)
        wall_theta_frames = []
        wall_r_frames = []

        r_top = 1.05

        for kk, ii in enumerate(frame_ids):
            ti = t[ii]
            shift = xA * np.sin(omega * ti)

            x_rel = map_particles(q[ii], a, b, c, d)
            q_display_frames[kk] = inverse_points(x_rel + shift)

            wall_q = inverse_points(wall0 + shift)
            order = np.argsort(wall_q[:, 1])
            wall_theta = wall_q[order, 1] % TWO_PI
            wall_r = wall_q[order, 0]
            wall_theta, wall_r = _insert_nan_at_theta_jumps(wall_theta, wall_r)

            wall_theta_frames.append(wall_theta)
            wall_r_frames.append(wall_r)
            r_top = max(r_top, float(np.nanmax(wall_r)) + 0.05)
    else:
        q_display_frames = q[frame_ids]
        wall_theta_frames = None
        wall_r_frames = None
        r_top = 1.25

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(0.0, TWO_PI)
    ax.set_ylim(0.0, r_top)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$r$")
    ax.set_title(title)
    ax.grid(True)

    if show_solver_wall:
        ax.axhline(1.0, linewidth=0.8, linestyle="--", label="solver wall r = 1")

    wall_kwargs = {"linewidth": 1.0, "label": "moving wall"}
    if wall_color is not None:
        wall_kwargs["color"] = wall_color
    wall_line, = ax.plot([], [], **wall_kwargs)
    scat = ax.scatter([], [], s=s, label="particles")
    trace0, = ax.plot([], [], linewidth=1.0, label="particle 0")
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    if show_legend:
        ax.legend(loc="upper left")

    def init():
        wall_line.set_data([], [])
        scat.set_offsets(np.empty((0, 2)))
        trace0.set_data([], [])
        time_text.set_text("")
        return wall_line, scat, trace0, time_text

    def update(frame):
        ii = frame_ids[frame]
        qf = q_display_frames[frame]

        pts = np.column_stack((qf[:, 1] % TWO_PI, qf[:, 0]))
        scat.set_offsets(pts)

        if use_lab_view:
            wall_line.set_data(wall_theta_frames[frame], wall_r_frames[frame])
        else:
            wall_line.set_data([], [])

        if show_trace0:
            theta_trace = q_display_frames[:frame + 1, 0, 1] % TWO_PI
            r_trace = q_display_frames[:frame + 1, 0, 0]
            theta_trace, r_trace = _insert_nan_at_theta_jumps(theta_trace, r_trace)
            trace0.set_data(theta_trace, r_trace)
        else:
            trace0.set_data([], [])

        time_text.set_text(f"t = {t[ii]:.3f}")
        return wall_line, scat, trace0, time_text

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        interval=interval,
        blit=True,
    )

    return ani

def to_jshtml(animation):
    if HTML is None:
        return animation
    return HTML(animation.to_jshtml())
