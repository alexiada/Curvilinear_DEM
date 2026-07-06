"""
Fast polar-coordinate DEM validation solver for a circular moving wall.

Version v2 keeps the public API of polar_solver_v1 where practical, but uses
the stability pattern from curvilinear_solver_v15:

    - particle contacts and neighbour lists are always evaluated in physical
      Euclidean coordinates;
    - physical forces are accumulated once in x-y and then pulled back to polar
      acceleration once per particle;
    - particles entering the polar singularity near r = 0 are temporarily
      integrated in Cartesian coordinates, then handed back to polar
      coordinates after they leave a small cap.

State variables
---------------
mode[i] == POLAR:
    q[i] = (r, theta)
    p[i] = (rdot, thetadot)
    acc[i] = (rddot, thetaddot)

mode[i] == CAP:
    q[i] = (x, y)
    p[i] = (vx, vy)
    acc[i] = (ax, ay)

The kick-drift-kick integrator is therefore the same in both modes.
"""

import math
import numpy as np
from numba import njit


f32 = np.float32
i64 = np.int64
TWO_PI = f32(2.0 * np.pi)

POLAR = i64(0)
CAP = i64(1)


def as_f32(a):
    return np.asarray(a, dtype=f32)


def as_i64(a):
    return np.asarray(a, dtype=i64)


@njit(cache=False, fastmath=True)
def wrap_theta(theta):
    return theta % TWO_PI


@njit(cache=False, fastmath=True)
def physical_to_polar_one(x0, x1, v0, v1, min_radius):
    r = math.sqrt(x0 * x0 + x1 * x1)
    theta = wrap_theta(math.atan2(x1, x0))

    r_eff = r
    if r_eff < min_radius:
        r_eff = min_radius

    c = math.cos(theta)
    s = math.sin(theta)

    rdot = v0 * c + v1 * s
    thetadot = (-v0 * s + v1 * c) / r_eff

    return r, theta, rdot, thetadot


@njit(cache=False, fastmath=True)
def build_cell_list(x, box, cell_size):
    x_min, x_max, y_min, y_max = box
    N = x.shape[0]
    nx = max(1, int(math.ceil((x_max - x_min) / cell_size)))
    ny = max(1, int(math.ceil((y_max - y_min) / cell_size)))

    head = np.full(nx * ny, -1, dtype=i64)
    nxt = np.full(N, -1, dtype=i64)
    cell_id = np.empty(N, dtype=i64)

    inv_cell = f32(1.0) / cell_size

    for i in range(N):
        xi = x[i, 0]
        yi = x[i, 1]

        if xi < x_min or xi > x_max or yi < y_min or yi > y_max:
            raise ValueError("particle outside box")

        cx = int((xi - x_min) * inv_cell)
        cy = int((yi - y_min) * inv_cell)

        if cx == nx:
            cx = nx - 1
        if cy == ny:
            cy = ny - 1

        c = cx + nx * cy
        cell_id[i] = c
        nxt[i] = head[c]
        head[c] = i

    return head, nxt, cell_id, nx, ny


@njit(cache=False, fastmath=True)
def build_verlet_csr(x, box, r_list, cell_size):
    N = x.shape[0]
    r2_list = r_list * r_list

    head, nxt, cell_id, nx, ny = build_cell_list(x, box, cell_size)
    cr = max(1, int(math.ceil(r_list / cell_size)))

    deg = np.zeros(N, dtype=i64)

    for i in range(N):
        xi0 = x[i, 0]
        xi1 = x[i, 1]
        cx = cell_id[i] % nx
        cy = cell_id[i] // nx

        for dy in range(-cr, cr + 1):
            cy2 = cy + dy
            if cy2 < 0 or cy2 >= ny:
                continue

            for dx in range(-cr, cr + 1):
                cx2 = cx + dx
                if cx2 < 0 or cx2 >= nx:
                    continue

                j = head[cx2 + nx * cy2]
                while j != -1:
                    if j > i:
                        dxij = xi0 - x[j, 0]
                        dyij = xi1 - x[j, 1]
                        if dxij * dxij + dyij * dyij <= r2_list:
                            deg[i] += 1
                            deg[j] += 1
                    j = nxt[j]

    offsets = np.empty(N + 1, dtype=i64)
    offsets[0] = 0
    for i in range(N):
        offsets[i + 1] = offsets[i] + deg[i]

    neigh = np.empty(offsets[N], dtype=i64)
    cursor = np.empty(N, dtype=i64)
    for i in range(N):
        cursor[i] = offsets[i]

    for i in range(N):
        xi0 = x[i, 0]
        xi1 = x[i, 1]
        cx = cell_id[i] % nx
        cy = cell_id[i] // nx

        for dy in range(-cr, cr + 1):
            cy2 = cy + dy
            if cy2 < 0 or cy2 >= ny:
                continue

            for dx in range(-cr, cr + 1):
                cx2 = cx + dx
                if cx2 < 0 or cx2 >= nx:
                    continue

                j = head[cx2 + nx * cy2]
                while j != -1:
                    if j > i:
                        dxij = xi0 - x[j, 0]
                        dyij = xi1 - x[j, 1]
                        if dxij * dxij + dyij * dyij <= r2_list:
                            neigh[cursor[i]] = j
                            cursor[i] += 1
                            neigh[cursor[j]] = i
                            cursor[j] += 1
                    j = nxt[j]

    return offsets, neigh


@njit(cache=False, fastmath=True)
def needs_verlet_rebuild(x_now, x_ref, skin):
    skin_half2 = (f32(0.5) * skin) ** f32(2.0)
    max_disp2 = f32(0.0)

    for i in range(x_now.shape[0]):
        dx = x_now[i, 0] - x_ref[i, 0]
        dy = x_now[i, 1] - x_ref[i, 1]
        d2 = dx * dx + dy * dy
        if d2 > max_disp2:
            max_disp2 = d2

    return max_disp2 > skin_half2


@njit(cache=False, fastmath=True)
def apply_safeguard(q, mode):
    for i in range(q.shape[0]):
        if mode[i] == POLAR:
            q[i, 1] = wrap_theta(q[i, 1])


@njit(cache=False, fastmath=True)
def handoff(q, p, mode, group, r_cap, r_exit, min_radius):
    start = group[0]
    end = group[1]
    r_exit2 = r_exit * r_exit

    for i in range(start, end):
        if mode[i] == POLAR:
            r = q[i, 0]
            if r < f32(0.0):
                r = -r
                q[i, 1] = q[i, 1] + f32(np.pi)
                p[i, 0] = -p[i, 0]
                q[i, 0] = r

            if r <= r_cap:
                theta = q[i, 1]
                rdot = p[i, 0]
                thetadot = p[i, 1]
                c = math.cos(theta)
                s = math.sin(theta)

                q[i, 0] = r * c
                q[i, 1] = r * s
                p[i, 0] = rdot * c - r * thetadot * s
                p[i, 1] = rdot * s + r * thetadot * c
                mode[i] = CAP
        else:
            x0 = q[i, 0]
            x1 = q[i, 1]
            rho2 = x0 * x0 + x1 * x1

            if rho2 < r_exit2:
                continue

            r, theta, rdot, thetadot = physical_to_polar_one(
                x0, x1, p[i, 0], p[i, 1], min_radius
            )

            if r >= r_exit:
                q[i, 0] = r
                q[i, 1] = theta
                p[i, 0] = rdot
                p[i, 1] = thetadot
                mode[i] = POLAR


@njit(cache=False, fastmath=True)
def map_to_physical(q, p, x, v, mode):
    for i in range(q.shape[0]):
        if mode[i] == POLAR:
            r = q[i, 0]
            theta = q[i, 1]
            rdot = p[i, 0]
            thetadot = p[i, 1]

            c = math.cos(theta)
            s = math.sin(theta)

            x[i, 0] = r * c
            x[i, 1] = r * s
            v[i, 0] = rdot * c - r * thetadot * s
            v[i, 1] = rdot * s + r * thetadot * c
        else:
            x[i, 0] = q[i, 0]
            x[i, 1] = q[i, 1]
            v[i, 0] = p[i, 0]
            v[i, 1] = p[i, 1]


@njit(cache=False, fastmath=True)
def accumulate_contact_forces(
    fext, x, v, rad, group, offsets, neigh, k_contact, gamma_contact
):
    start = group[0]
    end = group[1]

    for i in range(start, end):
        xi0 = x[i, 0]
        xi1 = x[i, 1]
        vi0 = v[i, 0]
        vi1 = v[i, 1]
        ri = rad[i]

        for pp in range(offsets[i], offsets[i + 1]):
            j = neigh[pp]

            if j <= i:
                continue
            if j < start or j >= end:
                continue

            dx = x[j, 0] - xi0
            dy = x[j, 1] - xi1
            r2 = dx * dx + dy * dy
            if r2 == f32(0.0):
                continue

            rr = math.sqrt(r2)
            overlap = ri + rad[j] - rr

            if overlap > f32(0.0):
                inv_rr = f32(1.0) / rr
                nx = dx * inv_rr
                ny = dy * inv_rr

                dvx = v[j, 0] - vi0
                dvy = v[j, 1] - vi1
                vn = dvx * nx + dvy * ny

                fm = k_contact * overlap - gamma_contact * vn
                if fm < f32(0.0):
                    fm = f32(0.0)

                fx = fm * nx
                fy = fm * ny

                fext[i, 0] -= fx
                fext[i, 1] -= fy
                fext[j, 0] += fx
                fext[j, 1] += fy


@njit(cache=False, fastmath=True)
def accumulate_wall_forces(
    fext, x, v, rad, group, radius, xA, omega, k_w, gamma_w, t
):
    start = group[0]
    end = group[1]

    cx = xA[0] * math.sin(omega * t)
    cy = xA[1] * math.sin(omega * t)
    cvx = xA[0] * omega * math.cos(omega * t)
    cvy = xA[1] * omega * math.cos(omega * t)

    for i in range(start, end):
        xr0 = x[i, 0] - cx
        xr1 = x[i, 1] - cy
        vr0 = v[i, 0] - cvx
        vr1 = v[i, 1] - cvy

        r2 = xr0 * xr0 + xr1 * xr1
        if r2 == f32(0.0):
            continue

        rr = math.sqrt(r2)
        delta = rr + rad[i] - radius

        if delta > f32(0.0):
            n0 = -xr0 / rr
            n1 = -xr1 / rr
            vn = vr0 * n0 + vr1 * n1

            f_mag = k_w * delta - gamma_w * vn
            if f_mag < f32(0.0):
                f_mag = f32(0.0)

            fext[i, 0] += f_mag * n0
            fext[i, 1] += f_mag * n1


@njit(cache=False, fastmath=True)
def finalize_accelerations(acc, q, p, fext, mode, group, m, g, min_radius):
    start = group[0]
    end = group[1]

    for i in range(start, end):
        ax = fext[i, 0] / m[i] + g[0]
        ay = fext[i, 1] / m[i] + g[1]

        if mode[i] == POLAR:
            r = q[i, 0]
            if r < min_radius:
                r = min_radius

            theta = q[i, 1]
            rdot = p[i, 0]
            thetadot = p[i, 1]
            c = math.cos(theta)
            s = math.sin(theta)

            acc[i, 0] = r * thetadot * thetadot + ax * c + ay * s
            acc[i, 1] = (
                -f32(2.0) * rdot * thetadot
                + (-ax * s + ay * c)
            ) / r
        else:
            acc[i, 0] = ax
            acc[i, 1] = ay


@njit(cache=False, fastmath=True)
def compute_accelerations(
    acc, q, p, x, v, fext, mode, m, rad, group_mobile,
    offsets, neigh,
    radius, xA, omega,
    k_contact, gamma_contact, k_w, gamma_w, g, t, min_radius,
):
    for i in range(fext.shape[0]):
        fext[i, 0] = f32(0.0)
        fext[i, 1] = f32(0.0)
        acc[i, 0] = f32(0.0)
        acc[i, 1] = f32(0.0)

    accumulate_contact_forces(
        fext, x, v, rad, group_mobile, offsets, neigh,
        k_contact, gamma_contact,
    )
    accumulate_wall_forces(
        fext, x, v, rad, group_mobile, radius, xA, omega,
        k_w, gamma_w, t,
    )
    finalize_accelerations(acc, q, p, fext, mode, group_mobile, m, g, min_radius)


@njit(cache=False, fastmath=True)
def snapshot_polar(q_save, p_save, q, p, mode, min_radius):
    for i in range(q.shape[0]):
        if mode[i] == POLAR:
            q_save[i, 0] = q[i, 0]
            q_save[i, 1] = q[i, 1]
            p_save[i, 0] = p[i, 0]
            p_save[i, 1] = p[i, 1]
        else:
            r, theta, rdot, thetadot = physical_to_polar_one(
                q[i, 0], q[i, 1], p[i, 0], p[i, 1], min_radius
            )
            q_save[i, 0] = r
            q_save[i, 1] = theta
            p_save[i, 0] = rdot
            p_save[i, 1] = thetadot


@njit(cache=False, fastmath=True)
def velocity_verlet_step(
    q, p, x, v, acc, fext, mode, dt, half_dt,
    m, rad, group_mobile,
    box, r_list, cell_size, skin,
    offsets, neigh, x_verlet_ref,
    radius, xA, omega,
    k_contact, gamma_contact, k_w, gamma_w, g, t,
    min_radius, r_cap, r_exit,
):
    N = q.shape[0]

    for i in range(N):
        p[i, 0] += half_dt * acc[i, 0]
        p[i, 1] += half_dt * acc[i, 1]

    for i in range(N):
        q[i, 0] += dt * p[i, 0]
        q[i, 1] += dt * p[i, 1]

    apply_safeguard(q, mode)
    handoff(q, p, mode, group_mobile, r_cap, r_exit, min_radius)
    apply_safeguard(q, mode)
    map_to_physical(q, p, x, v, mode)

    if needs_verlet_rebuild(x, x_verlet_ref, skin):
        offsets, neigh = build_verlet_csr(x, box, r_list, cell_size)
        x_verlet_ref[:, :] = x

    compute_accelerations(
        acc, q, p, x, v, fext, mode, m, rad, group_mobile,
        offsets, neigh,
        radius, xA, omega,
        k_contact, gamma_contact, k_w, gamma_w, g, t + dt, min_radius,
    )

    for i in range(N):
        p[i, 0] += half_dt * acc[i, 0]
        p[i, 1] += half_dt * acc[i, 1]

    map_to_physical(q, p, x, v, mode)

    return offsets, neigh


@njit(cache=False, fastmath=True)
def simulate_polar_particles_jit(
    box, q0, p0, m, rad, dt, T_max,
    group_mobile,
    k_contact, gamma_contact,
    radius, xA, omega,
    k_w, gamma_w, g,
    r_list, skin, cell_size, save_every,
    min_radius, r_cap, r_exit,
):
    N = q0.shape[0]
    num_steps = int(T_max / dt + f32(0.5))
    num_save = num_steps // save_every + 1

    t_out = np.empty(num_save, dtype=f32)
    q_out = np.empty((num_save, N, 2), dtype=f32)
    p_out = np.empty((num_save, N, 2), dtype=f32)
    x_out = np.empty((num_save, N, 2), dtype=f32)
    v_out = np.empty((num_save, N, 2), dtype=f32)

    q = q0.copy()
    p = p0.copy()
    mode = np.zeros(N, dtype=i64)

    for i in range(N):
        q[i, 1] = wrap_theta(q[i, 1])

    handoff(q, p, mode, group_mobile, r_cap, r_exit, min_radius)

    x = np.empty((N, 2), dtype=f32)
    v = np.empty((N, 2), dtype=f32)
    fext = np.empty((N, 2), dtype=f32)
    acc = np.empty((N, 2), dtype=f32)

    map_to_physical(q, p, x, v, mode)

    half_dt = f32(0.5) * dt
    x_verlet_ref = x.copy()
    offsets, neigh = build_verlet_csr(x, box, r_list, cell_size)

    compute_accelerations(
        acc, q, p, x, v, fext, mode, m, rad, group_mobile,
        offsets, neigh,
        radius, xA, omega,
        k_contact, gamma_contact, k_w, gamma_w, g, f32(0.0), min_radius,
    )

    save_id = 0
    t_out[save_id] = f32(0.0)
    snapshot_polar(q_out[save_id], p_out[save_id], q, p, mode, min_radius)
    x_out[save_id] = x
    v_out[save_id] = v
    save_id += 1

    for n in range(num_steps):
        t = f32(n * dt)

        offsets, neigh = velocity_verlet_step(
            q, p, x, v, acc, fext, mode, dt, half_dt,
            m, rad, group_mobile,
            box, r_list, cell_size, skin,
            offsets, neigh, x_verlet_ref,
            radius, xA, omega,
            k_contact, gamma_contact, k_w, gamma_w, g, t,
            min_radius, r_cap, r_exit,
        )

        if (n + 1) % save_every == 0:
            t_out[save_id] = f32((n + 1) * dt)
            snapshot_polar(q_out[save_id], p_out[save_id], q, p, mode, min_radius)
            x_out[save_id] = x
            v_out[save_id] = v
            save_id += 1

    return t_out, q_out, p_out, x_out, v_out


def simulate_polar_particles(
    box,
    q0,
    p0,
    m,
    rad,
    dt,
    T_max,
    group_mobile,
    k_contact,
    gamma_contact,
    radius,
    xA,
    omega,
    k_w,
    gamma_w,
    g,
    save_every=1,
    r_list=None,
    skin=None,
    cell_size=None,
    min_radius=1.0e-6,
    r_cap=0.05,
    r_exit=None,
):
    """
    Run the polar validation solver.

    min_radius is only a denominator guard for reconstructing thetadot. The
    dynamics near the origin are handled by r_cap/r_exit Cartesian handoff.
    """
    q0 = as_f32(q0)
    p0 = as_f32(p0)
    m = as_f32(m)
    rad = as_f32(rad)
    box = as_f32(box)
    group_mobile = as_i64(group_mobile)
    xA = as_f32(xA)
    g = as_f32(g)

    max_rad = f32(np.max(rad))
    skin_val = f32(skin if skin is not None else f32(0.25) * max_rad)
    r_contact = f32(2.0) * max_rad
    r_list_val = f32(r_list if r_list is not None else r_contact + skin_val)
    cell_size_val = f32(cell_size if cell_size is not None else r_list_val)

    r_cap_val = f32(r_cap)
    r_exit_val = f32(r_exit if r_exit is not None else f32(1.5) * r_cap_val)

    return simulate_polar_particles_jit(
        box,
        q0,
        p0,
        m,
        rad,
        f32(dt),
        f32(T_max),
        group_mobile,
        f32(k_contact),
        f32(gamma_contact),
        f32(radius),
        xA,
        f32(omega),
        f32(k_w),
        f32(gamma_w),
        g,
        r_list_val,
        skin_val,
        cell_size_val,
        int(save_every),
        f32(min_radius),
        r_cap_val,
        r_exit_val,
    )


@njit(cache=False, fastmath=True)
def euclidean_to_polar_particles_jit(x, v, q, p, min_radius):
    for i in range(x.shape[0]):
        r, theta, rdot, thetadot = physical_to_polar_one(
            x[i, 0], x[i, 1], v[i, 0], v[i, 1], min_radius
        )
        q[i, 0] = r
        q[i, 1] = theta
        p[i, 0] = rdot
        p[i, 1] = thetadot


def euclidean_to_polar_particles(x, v, min_radius=1.0e-6):
    x = as_f32(x)
    v = as_f32(v)
    q = np.empty_like(x, dtype=f32)
    p = np.empty_like(v, dtype=f32)

    euclidean_to_polar_particles_jit(x, v, q, p, f32(min_radius))

    return q, p


def polar_to_euclidean_particles(q, p, min_radius=1.0e-6):
    q = as_f32(q)
    p = as_f32(p)
    x = np.empty_like(q, dtype=f32)
    v = np.empty_like(p, dtype=f32)
    mode = np.zeros(q.shape[0], dtype=i64)

    map_to_physical(q, p, x, v, mode)

    return x, v


############################################################
# Visualisation
############################################################

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def animate_particles_in_polar_plane(
    t,
    q,
    radius,
    xA,
    omega,
    fps=60,
    stride=1,
    s=20,
    show_traj0=True,
    show_particle0=True,
    wall_color=None,
):
    """
    Animate particles in the polar coordinate plane.

    The moving circular wall is drawn with the same direct formula used in the
    original multiparticle polar notebook. This avoids using a general inverse
    mapping for the visual wall in the circular validation case.
    """
    idx = np.arange(0, len(t), max(1, int(stride)), dtype=np.int64)
    xA = np.asarray(xA, dtype=float)

    theta_grid = np.linspace(0.0, 2.0 * np.pi, 400)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_ylim(0.0, radius + np.linalg.norm(xA) + 0.5)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$r$")

    scat = ax.scatter(q[idx[0], :, 1], q[idx[0], :, 0], s=s)

    if show_particle0:
        p0_x = [q[idx[0], 0, 1]]
        p0_y = [q[idx[0], 0, 0]]
    else:
        p0_x = []
        p0_y = []

    p0_marker, = ax.plot(
        p0_x,
        p0_y,
        marker="o",
        linestyle="None",
        markersize=6,
    )

    traj0, = ax.plot([], [], linewidth=1.5)

    wall_kwargs = {"linewidth": 1.2}
    if wall_color is not None:
        wall_kwargs["color"] = wall_color
    wall, = ax.plot([], [], **wall_kwargs)

    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    def wall_curve(time_value):
        c = xA * np.sin(float(omega) * float(time_value))
        cx = c[0]
        cy = c[1]

        ct = cx * np.cos(theta_grid) + cy * np.sin(theta_grid)
        disc = ct * ct + radius * radius - (cx * cx + cy * cy)
        disc = np.maximum(disc, 0.0)

        return ct + np.sqrt(disc)

    def traj0_with_breaks(frame_i):
        th = q[idx[:frame_i + 1], 0, 1]
        rr = q[idx[:frame_i + 1], 0, 0]

        th_plot = [th[0]]
        r_plot = [rr[0]]

        for k in range(1, len(th)):
            if abs(th[k] - th[k - 1]) > np.pi:
                th_plot.append(np.nan)
                r_plot.append(np.nan)
            th_plot.append(th[k])
            r_plot.append(rr[k])

        return np.asarray(th_plot), np.asarray(r_plot)

    def init():
        n = idx[0]

        scat.set_offsets(np.column_stack((q[n, :, 1], q[n, :, 0])))
        if show_particle0:
            p0_marker.set_data([q[n, 0, 1]], [q[n, 0, 0]])
        else:
            p0_marker.set_data([], [])

        if show_traj0:
            traj0.set_data([q[n, 0, 1]], [q[n, 0, 0]])
        else:
            traj0.set_data([], [])

        wall.set_data(theta_grid, wall_curve(t[n]))
        time_text.set_text(f"t = {float(t[n]):.3f}")

        return scat, p0_marker, traj0, wall, time_text

    def update(frame_i):
        n = idx[frame_i]

        scat.set_offsets(np.column_stack((q[n, :, 1], q[n, :, 0])))
        if show_particle0:
            p0_marker.set_data([q[n, 0, 1]], [q[n, 0, 0]])
        else:
            p0_marker.set_data([], [])

        if show_traj0:
            th_traj, r_traj = traj0_with_breaks(frame_i)
            traj0.set_data(th_traj, r_traj)
        else:
            traj0.set_data([], [])

        wall.set_data(theta_grid, wall_curve(t[n]))
        time_text.set_text(f"t = {float(t[n]):.3f}")

        return scat, p0_marker, traj0, wall, time_text

    ani = FuncAnimation(
        fig,
        update,
        frames=len(idx),
        init_func=init,
        blit=True,
        interval=int(1000 / fps),
    )

    plt.close(fig)
    return HTML(ani.to_jshtml())
