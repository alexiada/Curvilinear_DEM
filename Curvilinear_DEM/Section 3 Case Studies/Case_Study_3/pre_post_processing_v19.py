"""
Pre/post-processing helpers for Case Study 3.

This version uses the explicit time-dependent map in maps.py.  The first
use is particle placement at t = 0, using the same logic as Case Study 1:
sample the curvilinear domain with area weight |det J|, map candidates to
physical space, and reject physical overlaps.
"""

import math
import warnings

import numpy as np

import maps

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


TWO_PI = 2.0 * np.pi


def boundary_curve(t=0.0, n=1200):
    theta = np.linspace(0.0, TWO_PI, int(n), endpoint=True)
    x = np.empty_like(theta)
    y = np.empty_like(theta)
    for i, th in enumerate(theta):
        x[i], y[i] = maps.eval_map(np.float32(1.0), np.float32(th), np.float32(t))
    return np.column_stack((x, y))


def cartesian_grid_box_from_boundary_times(times, n=900, margin=0.05, mode="intersection"):
    """Return a fixed Cartesian grid box based on boundary extents.

    The result has the same order as the solver box:

        [x_min, x_max, y_min, y_max]

    For videos, ``mode="intersection"`` chooses values that stay within the
    boundary bounding box for all sampled times.  This avoids drawing a
    Cartesian grid from a much larger simulation box, where most horizontal
    lines have no visible pre-image in the curvilinear domain.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim == 0:
        times = times[None]

    mins = []
    maxs = []
    for tt in times:
        boundary = boundary_curve(t=float(tt), n=n)
        mins.append(np.min(boundary, axis=0))
        maxs.append(np.max(boundary, axis=0))

    mins = np.asarray(mins)
    maxs = np.asarray(maxs)

    if mode == "intersection":
        low = np.max(mins, axis=0)
        high = np.min(maxs, axis=0)
    elif mode == "union":
        low = np.min(mins, axis=0)
        high = np.max(maxs, axis=0)
    else:
        raise ValueError("mode must be 'intersection' or 'union'")

    span = high - low
    low = low + margin * span
    high = high - margin * span

    return np.array([low[0], high[0], low[1], high[1]], dtype=float)


def map_particles(q, t=0.0):
    q = np.asarray(q, dtype=float)
    x = np.empty((len(q), 2), dtype=float)
    tt = np.float32(t)
    for i in range(len(q)):
        x[i, 0], x[i, 1] = maps.eval_map(
            np.float32(q[i, 0]), np.float32(q[i, 1]), tt
        )
    return x


def area_weight(theta, t=0.0):
    theta = np.asarray(theta, dtype=float)
    scalar = theta.ndim == 0
    theta = np.atleast_1d(theta)
    out = np.empty_like(theta)
    tt = np.float32(t)
    for i, th in enumerate(theta):
        out[i] = abs(float(maps.eval_cross(np.float32(th), tt)))
    if scalar:
        return float(out[0])
    return out


def make_theta_sampler(t=0.0, n=10000):
    theta = np.linspace(0.0, TWO_PI, int(n), endpoint=False)
    w = np.maximum(area_weight(theta, t=t), 0.0)
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return theta, cdf


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
    particle_radius,
    t=0.0,
    fraction=1.0,
    r_min=0.05,
    r_max=0.92,
    safety=1.05,
    max_trials=500000,
    seed=1,
    batch_size=None,
):
    """Generate non-overlapping particles inside the t-time boundary.

    fraction = 1.0 fills the whole mapped domain.  Values below one keep
    only the lower fraction in physical y, following the Case Study 1 helper.
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

    theta_table, cdf_table = make_theta_sampler(t=t)

    boundary = boundary_curve(t=t)
    y_min = float(np.min(boundary[:, 1]))
    y_max = float(np.max(boundary[:, 1]))
    y_limit = y_min + fraction * (y_max - y_min)

    q_acc = np.empty((N, 2), dtype=float)
    x_acc = np.empty((N, 2), dtype=float)
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

        q_batch = np.column_stack((r, theta))
        x_batch = map_particles(q_batch, t=t)
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
            "Try reducing N or particle_radius, increasing fraction, "
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
        "t": float(t),
        "y_limit": y_limit,
        "method": "maps.py + area-weighted q sampling + physical grid",
    }
    return q_acc, x_acc, nearest, info


def make_initial_conditions(
    N,
    particle_radius,
    particle_mass=1.0,
    velocity=(0.0, 0.0),
    t=0.0,
    fraction=1.0,
    r_min=0.05,
    r_max=0.92,
    safety=1.05,
    max_trials=500000,
    seed=1,
):
    q0, x0, nearest, info = filling(
        N=N,
        particle_radius=particle_radius,
        t=t,
        fraction=fraction,
        r_min=r_min,
        r_max=r_max,
        safety=safety,
        max_trials=max_trials,
        seed=seed,
    )

    p0 = np.zeros_like(q0, dtype=float)
    p0[:, 0] = velocity[0]
    p0[:, 1] = velocity[1]

    n_particles = len(q0)
    m = np.full(n_particles, particle_mass, dtype=np.float32)
    rad = np.full(n_particles, particle_radius, dtype=np.float32)
    group_mobile = np.array([0, n_particles], dtype=np.int64)

    return (
        q0.astype(np.float32),
        p0.astype(np.float32),
        x0.astype(np.float32),
        m,
        rad,
        group_mobile,
        nearest,
        info,
    )


def plot_initial_configuration(q, x, t=0.0, s=8):
    import matplotlib.pyplot as plt

    boundary = boundary_curve(t=t)

    fig, (ax_phys, ax_q) = plt.subplots(1, 2, figsize=(11, 4))

    ax_phys.plot(boundary[:, 0], boundary[:, 1], linewidth=1.2, color="black")
    ax_phys.scatter(x[:, 0], x[:, 1], s=s)
    ax_phys.set_aspect("equal", adjustable="box")
    ax_phys.set_xlabel("x")
    ax_phys.set_ylabel("y")
    ax_phys.set_title(f"physical space, t={t:g}")
    ax_phys.grid(True)

    ax_q.scatter(q[:, 1], q[:, 0], s=s)
    ax_q.axhline(1.0, linewidth=0.8, linestyle="--", color="black")
    ax_q.set_xlim(0.0, TWO_PI)
    ax_q.set_ylim(0.0, 1.05)
    ax_q.set_xlabel(r"$\theta$")
    ax_q.set_ylabel(r"$r$")
    ax_q.set_title("curvilinear space")
    ax_q.grid(True)

    plt.tight_layout()
    return fig, (ax_phys, ax_q)


def physical_to_curvilinear_points(points, t=0.0, newton_iters=8):
    """Map physical Cartesian points to q=(r, theta) for x = r B(theta,t).

    This inverse is used for diagnostics and visualisation.  It assumes the
    radial chart used in Case Study 3, where a physical point lies on the ray
    generated by B(theta,t).  The angle theta is obtained by Newton iteration
    on the cross-product alignment condition B(theta,t) x point = 0.
    """
    points = np.asarray(points, dtype=float)
    q = np.full((len(points), 2), np.nan, dtype=float)
    tt = np.float32(t)

    for i, (x0, x1) in enumerate(points):
        if not np.isfinite(x0) or not np.isfinite(x1):
            continue

        theta = float(np.arctan2(x1, x0) % TWO_PI)
        for _ in range(int(newton_iters)):
            (
                Bx, By,
                Bx_theta, By_theta,
                _, _, _, _, _, _, _, _,
            ) = maps.eval_boundary(np.float32(theta), tt)
            g = float(Bx) * x1 - float(By) * x0
            gp = float(Bx_theta) * x1 - float(By_theta) * x0
            if abs(gp) < 1.0e-12:
                break
            theta = (theta - g / gp) % TWO_PI

        Bx, By, *_ = maps.eval_boundary(np.float32(theta), tt)
        den = float(Bx) * float(Bx) + float(By) * float(By)
        if den <= 0.0:
            continue
        r = (x0 * float(Bx) + x1 * float(By)) / den
        q[i, 0] = r
        q[i, 1] = theta

    return q


def _split_wrapped_curve(theta, r, max_jump=np.pi):
    """Split a theta-r curve at invalid points and at theta wrap jumps."""
    theta = np.asarray(theta, dtype=float)
    r = np.asarray(r, dtype=float)
    valid = np.isfinite(theta) & np.isfinite(r)
    curves = []
    start = None

    for i, ok in enumerate(valid):
        if ok and start is None:
            start = i
        end_curve = False
        if start is not None:
            if not ok:
                end_curve = True
            elif i > start and abs(theta[i] - theta[i - 1]) > max_jump:
                end_curve = True
        if end_curve:
            end = i if not ok else i
            if end - start >= 2:
                curves.append(np.column_stack((theta[start:end], r[start:end])))
            start = i if ok else None
    if start is not None and len(theta) - start >= 2:
        curves.append(np.column_stack((theta[start:], r[start:])))

    return curves


def cartesian_grid_curves_in_curvilinear(
    t=0.0,
    box=None,
    n_x=8,
    n_y=5,
    n_points=400,
    r_min=0.02,
    r_max=1.0,
    margin=0.02,
):
    """Return pre-images of Cartesian grid lines in curvilinear coordinates.

    This version uses the radial map structure directly:

        Phi(t, r, theta) = r B(t, theta).

    A physical line x=x0 maps to r=x0/Bx(t,theta), while y=y0 maps to
    r=y0/By(t,theta).  This avoids pointwise inverse-map artefacts when the
    grid is drawn for videos.

    Returns a list of arrays with columns (theta, r), split at outside-domain
    gaps.
    """
    if box is None:
        boundary = boundary_curve(t=t, n=1200)
        x_min, y_min = np.min(boundary, axis=0)
        x_max, y_max = np.max(boundary, axis=0)
    else:
        x_min, x_max, y_min, y_max = [float(v) for v in box]

    dx = margin * (x_max - x_min)
    dy = margin * (y_max - y_min)
    xs = np.linspace(x_min + dx, x_max - dx, int(n_x))
    ys = np.linspace(y_min + dy, y_max - dy, int(n_y))

    theta = np.linspace(0.0, TWO_PI, int(n_points), endpoint=True)
    bx = np.empty_like(theta)
    by = np.empty_like(theta)
    tt = np.float32(t)
    for i, th in enumerate(theta):
        bx_i, by_i, *_ = maps.eval_boundary(np.float32(th), tt)
        bx[i] = float(bx_i)
        by[i] = float(by_i)

    curves = []

    for x0 in xs:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = float(x0) / bx
        inside = np.isfinite(r) & (r >= r_min) & (r <= r_max)
        r_plot = np.where(inside, r, np.nan)
        curves.extend(_split_wrapped_curve(theta, r_plot))

    for y0 in ys:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = float(y0) / by
        inside = np.isfinite(r) & (r >= r_min) & (r <= r_max)
        r_plot = np.where(inside, r, np.nan)
        curves.extend(_split_wrapped_curve(theta, r_plot))

    return curves


def draw_curvilinear_cartesian_grid(
    ax,
    t=0.0,
    box=None,
    n_x=8,
    n_y=5,
    n_points=400,
    r_min=0.02,
    r_max=1.0,
    **plot_kwargs,
):
    """Draw q-space pre-images of physical Cartesian straight grid lines."""
    kwargs = {
        "color": "0.72",
        "lw": 0.65,
        "alpha": 0.75,
        "zorder": 0,
    }
    kwargs.update(plot_kwargs)
    curves = cartesian_grid_curves_in_curvilinear(
        t=t,
        box=box,
        n_x=n_x,
        n_y=n_y,
        n_points=n_points,
        r_min=r_min,
        r_max=r_max,
    )
    lines = []
    for curve in curves:
        line, = ax.plot(curve[:, 0], curve[:, 1], **kwargs)
        lines.append(line)
    return lines, curves
