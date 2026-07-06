"""
Fast curvilinear multiparticle solver for a Fourier bumpy trapezium.

Rotating-gravity version based on v15.

The radial chart

    x = r * B(theta)

is singular at r = 0. This version avoids the singular region instead of
regularising it. A particle that enters a small central cap is integrated in
physical Cartesian coordinates until it is safely outside the cap again. The
normal curvilinear equations are used everywhere else.

This variant removes the sinusoidal wall/frame translation used by v15. The
boundary is fixed at r = 1 and the imposed acceleration rotates in the body
frame as

    g(t) = g_magnitude * (cos(omega * t), sin(omega * t)).

State variables
---------------
mode[i] == 0, curvilinear mode:
    q[i] = (r, theta)
    p[i] = (rdot, thetadot)
    acc[i] = (rddot, thetaddot)

mode[i] == 1, Cartesian cap mode:
    q[i] = (x, y)
    p[i] = (vx, vy)
    acc[i] = (ax, ay)

The kick and drift are therefore the same for both modes:

    p += 0.5 * dt * acc
    q += dt * p

Only the geometry, force pullback, and cap handoff are mode-aware.
Particle contacts and neighbour lists always use physical x, v.
"""

import math
import numpy as np
from numba import njit


f32 = np.float32
i64 = np.int64
TWO_PI = f32(2.0 * np.pi)

CURV = i64(0)
CAP = i64(1)


def as_f32(a):
    return np.asarray(a, dtype=f32)


def as_i64(a):
    return np.asarray(a, dtype=i64)


@njit(cache=False, fastmath=True)
def wrap_theta(theta):
    return theta % TWO_PI


@njit(cache=False, fastmath=True)
def eval_boundary(theta, coef_a, coef_b, coef_c, coef_d):
    """Evaluate B(theta), B'(theta), and B''(theta)."""
    Bx = coef_a[0]
    By = coef_c[0]

    Bx_theta = f32(0.0)
    By_theta = f32(0.0)
    Bx_thetatheta = f32(0.0)
    By_thetatheta = f32(0.0)

    K = coef_a.shape[0] - 1
    for k in range(1, K + 1):
        kk = f32(k)
        ktheta = kk * theta
        ck = math.cos(ktheta)
        sk = math.sin(ktheta)

        ax = coef_a[k]
        bx = coef_b[k]
        ay = coef_c[k]
        by = coef_d[k]

        Bx += ax * ck + bx * sk
        By += ay * ck + by * sk

        Bx_theta += -kk * ax * sk + kk * bx * ck
        By_theta += -kk * ay * sk + kk * by * ck

        kk2 = kk * kk
        Bx_thetatheta += -kk2 * ax * ck - kk2 * bx * sk
        By_thetatheta += -kk2 * ay * ck - kk2 * by * sk

    return Bx, By, Bx_theta, By_theta, Bx_thetatheta, By_thetatheta


@njit(cache=False, fastmath=True)
def interp_theta_from_phi(phi, theta_table, phi_table):
    """Invert direction angle phi -> theta using a monotone lookup table."""
    phi0 = phi_table[0]
    phi1 = phi_table[phi_table.shape[0] - 1]

    while phi < phi0:
        phi += TWO_PI
    while phi > phi1:
        phi -= TWO_PI

    lo = 0
    hi = phi_table.shape[0] - 1

    while hi - lo > 1:
        mid = (lo + hi) // 2
        if phi_table[mid] <= phi:
            lo = mid
        else:
            hi = mid

    p0 = phi_table[lo]
    p1 = phi_table[hi]
    th0 = theta_table[lo]
    th1 = theta_table[hi]

    den = p1 - p0
    if den == f32(0.0):
        return wrap_theta(th0)

    w = (phi - p0) / den
    return wrap_theta(th0 + w * (th1 - th0))


@njit(cache=False, fastmath=True)
def physical_to_curvilinear_one(
    x0, x1, v0, v1,
    theta_table, phi_table,
    coef_a, coef_b, coef_c, coef_d,
    min_r,
):
    """Invert x = r B(theta) and convert physical velocity to qdot."""
    phi = math.atan2(x1, x0)
    theta = interp_theta_from_phi(phi, theta_table, phi_table)

    Bx, By, Bx_theta, By_theta, _, _ = eval_boundary(
        theta, coef_a, coef_b, coef_c, coef_d
    )

    den = Bx * Bx + By * By
    r = (x0 * Bx + x1 * By) / den

    r_eff = r
    if r_eff < min_r:
        r_eff = min_r

    cross = Bx * By_theta - By * Bx_theta
    inv_cross = f32(1.0) / cross

    rdot = (By_theta * v0 - Bx_theta * v1) * inv_cross
    thetadot = (-By * v0 + Bx * v1) * inv_cross / r_eff

    return r, theta, rdot, thetadot


@njit(cache=False, fastmath=True)
def build_cell_list(x, box, cell_size):
    # x is Euclidean position in the container frame.
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
    # Verlet neighbours use physical Euclidean distance in x.
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
    # Rebuild criterion based on Euclidean displacement in the container frame.
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
    # Only curvilinear particles have theta.
    for i in range(q.shape[0]):
        if mode[i] == CURV:
            q[i, 1] = wrap_theta(q[i, 1])


@njit(cache=False, fastmath=True)
def handoff(
    q, p, mode, group,
    coef_a, coef_b, coef_c, coef_d,
    r_cap, r_exit, exit_inner_abs2,
    theta_table, phi_table, min_r,
):
    """Switch particles between curvilinear mode and Cartesian cap mode."""
    start = group[0]
    end = group[1]

    for i in range(start, end):
        if mode[i] == CURV:
            r = q[i, 0]
            if r <= r_cap:
                theta = q[i, 1]
                rdot = p[i, 0]
                thetadot = p[i, 1]

                Bx, By, Bx_theta, By_theta, _, _ = eval_boundary(
                    theta, coef_a, coef_b, coef_c, coef_d
                )

                q[i, 0] = r * Bx
                q[i, 1] = r * By
                p[i, 0] = Bx * rdot + r * Bx_theta * thetadot
                p[i, 1] = By * rdot + r * By_theta * thetadot
                mode[i] = CAP
        else:
            x0 = q[i, 0]
            x1 = q[i, 1]
            rho2 = x0 * x0 + x1 * x1

            # Definitely inside the exit radius for every boundary direction.
            if rho2 < exit_inner_abs2:
                continue

            r_rec, theta_rec, rdot_rec, thetadot_rec = physical_to_curvilinear_one(
                x0, x1, p[i, 0], p[i, 1],
                theta_table, phi_table,
                coef_a, coef_b, coef_c, coef_d,
                min_r,
            )

            if r_rec >= r_exit:
                q[i, 0] = r_rec
                q[i, 1] = theta_rec
                p[i, 0] = rdot_rec
                p[i, 1] = thetadot_rec
                mode[i] = CURV


@njit(cache=False, fastmath=True)
def compute_geometry(q, geom, mode, coef_a, coef_b, coef_c, coef_d, min_r):
    # Cache B, B_theta, B_thetatheta, and J^{-1} for curvilinear particles.
    for i in range(q.shape[0]):
        if mode[i] != CURV:
            continue

        r = q[i, 0]
        if r < min_r:
            r = min_r

        theta = q[i, 1]
        Bx, By, Bx_theta, By_theta, Bx_thetatheta, By_thetatheta = eval_boundary(
            theta, coef_a, coef_b, coef_c, coef_d
        )

        detJ = r * (Bx * By_theta - By * Bx_theta)
        inv_det = f32(1.0) / detJ

        geom[i, 0] = Bx
        geom[i, 1] = By
        geom[i, 2] = Bx_theta
        geom[i, 3] = By_theta
        geom[i, 4] = Bx_thetatheta
        geom[i, 5] = By_thetatheta

        # J^{-1} for J = [[Bx, r*Bx_theta], [By, r*By_theta]]
        geom[i, 6] = r * By_theta * inv_det
        geom[i, 7] = -r * Bx_theta * inv_det
        geom[i, 8] = -By * inv_det
        geom[i, 9] = Bx * inv_det


@njit(cache=False, fastmath=True)
def map_to_physical(q, p, x, v, geom, mode):
    # Fill physical x, v for all particles.
    for i in range(q.shape[0]):
        if mode[i] == CURV:
            r = q[i, 0]
            rdot = p[i, 0]
            thetadot = p[i, 1]

            Bx = geom[i, 0]
            By = geom[i, 1]
            Bx_theta = geom[i, 2]
            By_theta = geom[i, 3]

            x[i, 0] = r * Bx
            x[i, 1] = r * By
            v[i, 0] = Bx * rdot + r * Bx_theta * thetadot
            v[i, 1] = By * rdot + r * By_theta * thetadot
        else:
            x[i, 0] = q[i, 0]
            x[i, 1] = q[i, 1]
            v[i, 0] = p[i, 0]
            v[i, 1] = p[i, 1]


@njit(cache=False, fastmath=True)
def accumulate_contact_forces(
    fext, x, v, rad, group, offsets, neigh, k_contact, gamma_contact
):
    # Physical contact forces. Pullback happens once per particle later.
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

            r_contact = ri + rad[j]
            if r2 >= r_contact * r_contact:
                continue

            rr = math.sqrt(r2)
            overlap = r_contact - rr

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
def accumulate_wall_forces(fext, q, v, geom, mode, group, k_w, gamma_w):
    # Centre-only outer wall. Cap particles are near the centre and skip this.
    start = group[0]
    end = group[1]

    for i in range(start, end):
        if mode[i] != CURV:
            continue

        delta = q[i, 0] - f32(1.0)
        if delta > f32(0.0):
            Bx_theta = geom[i, 2]
            By_theta = geom[i, 3]

            norm_t = math.sqrt(Bx_theta * Bx_theta + By_theta * By_theta)
            inv_norm_t = f32(1.0) / norm_t

            n0 = -By_theta * inv_norm_t
            n1 = Bx_theta * inv_norm_t

            vn = v[i, 0] * n0 + v[i, 1] * n1

            f_mag = k_w * delta - gamma_w * vn
            if f_mag < f32(0.0):
                f_mag = f32(0.0)

            fext[i, 0] += f_mag * n0
            fext[i, 1] += f_mag * n1


@njit(cache=False, fastmath=True)
def finalize_accelerations(acc, q, p, geom, fext, mode, group, m, ax_body, ay_body):
    # Convert physical acceleration to the active coordinate acceleration.
    start = group[0]
    end = group[1]

    for i in range(start, end):
        inv_m = f32(1.0) / m[i]
        ax = fext[i, 0] * inv_m + ax_body
        ay = fext[i, 1] * inv_m + ay_body

        if mode[i] == CURV:
            r = q[i, 0]
            rdot = p[i, 0]
            thetadot = p[i, 1]

            Bx_theta = geom[i, 2]
            By_theta = geom[i, 3]
            Bx_thetatheta = geom[i, 4]
            By_thetatheta = geom[i, 5]

            Hx = f32(2.0) * rdot * thetadot * Bx_theta + r * thetadot * thetadot * Bx_thetatheta
            Hy = f32(2.0) * rdot * thetadot * By_theta + r * thetadot * thetadot * By_thetatheta

            g6 = geom[i, 6]
            g7 = geom[i, 7]
            g8 = geom[i, 8]
            g9 = geom[i, 9]

            acc[i, 0] = -(g6 * Hx + g7 * Hy) + (g6 * ax + g7 * ay)
            acc[i, 1] = -(g8 * Hx + g9 * Hy) + (g8 * ax + g9 * ay)
        else:
            acc[i, 0] = ax
            acc[i, 1] = ay


@njit(cache=False, fastmath=True)
def compute_accelerations(
    acc, q, p, x, v, geom, fext, mode, m, rad, group_mobile,
    offsets, neigh,
    k_contact, gamma_contact, k_w, gamma_w, g_magnitude, omega, t,
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
        fext, q, v, geom, mode, group_mobile,
        k_w, gamma_w,
    )

    phase = omega * t
    gx = g_magnitude * math.cos(phase)
    gy = g_magnitude * math.sin(phase)

    finalize_accelerations(
        acc, q, p, geom, fext, mode, group_mobile, m,
        gx,
        gy,
    )


@njit(cache=False, fastmath=True)
def write_physical_output(x_save, v_save, x, v):
    for i in range(x.shape[0]):
        x_save[i, 0] = x[i, 0]
        x_save[i, 1] = x[i, 1]
        v_save[i, 0] = v[i, 0]
        v_save[i, 1] = v[i, 1]


@njit(cache=False, fastmath=True)
def snapshot_curvilinear(
    q_save, p_save, q, p, mode,
    theta_table, phi_table,
    coef_a, coef_b, coef_c, coef_d,
    min_r,
):
    # Output only. Dynamics of cap particles stay Cartesian.
    for i in range(q.shape[0]):
        if mode[i] == CURV:
            q_save[i, 0] = q[i, 0]
            q_save[i, 1] = q[i, 1]
            p_save[i, 0] = p[i, 0]
            p_save[i, 1] = p[i, 1]
        else:
            r, theta, rdot, thetadot = physical_to_curvilinear_one(
                q[i, 0], q[i, 1], p[i, 0], p[i, 1],
                theta_table, phi_table,
                coef_a, coef_b, coef_c, coef_d,
                min_r,
            )
            q_save[i, 0] = r
            q_save[i, 1] = theta
            p_save[i, 0] = rdot
            p_save[i, 1] = thetadot


@njit(cache=False, fastmath=True)
def velocity_verlet_step(
    q, p, x, v, geom, acc, fext, mode, dt, half_dt,
    m, rad, group_mobile,
    box, r_list, cell_size, skin,
    offsets, neigh, x_verlet_ref,
    k_contact, gamma_contact, k_w, gamma_w, g_magnitude, omega, t,
    coef_a, coef_b, coef_c, coef_d,
    theta_table, phi_table,
    min_r, r_cap, r_exit, exit_inner_abs2,
):
    N = q.shape[0]

    for i in range(N):
        p[i, 0] += half_dt * acc[i, 0]
        p[i, 1] += half_dt * acc[i, 1]

    for i in range(N):
        q[i, 0] += dt * p[i, 0]
        q[i, 1] += dt * p[i, 1]

    apply_safeguard(q, mode)
    handoff(
        q, p, mode, group_mobile,
        coef_a, coef_b, coef_c, coef_d,
        r_cap, r_exit, exit_inner_abs2,
        theta_table, phi_table, min_r,
    )

    compute_geometry(q, geom, mode, coef_a, coef_b, coef_c, coef_d, min_r)
    map_to_physical(q, p, x, v, geom, mode)

    if needs_verlet_rebuild(x, x_verlet_ref, skin):
        offsets, neigh = build_verlet_csr(x, box, r_list, cell_size)
        x_verlet_ref[:, :] = x

    compute_accelerations(
        acc, q, p, x, v, geom, fext, mode, m, rad, group_mobile,
        offsets, neigh,
        k_contact, gamma_contact, k_w, gamma_w, g_magnitude, omega, t + dt,
    )

    for i in range(N):
        p[i, 0] += half_dt * acc[i, 0]
        p[i, 1] += half_dt * acc[i, 1]

    map_to_physical(q, p, x, v, geom, mode)

    return offsets, neigh


@njit(cache=False, fastmath=True)
def simulate_curvilinear_particles_jit(
    box, q0, p0, m, rad, dt, T_max,
    group_mobile,
    k_contact, gamma_contact,
    k_w, gamma_w, g_magnitude, omega,
    coef_a, coef_b, coef_c, coef_d,
    theta_table, phi_table,
    r_list, skin, cell_size, save_every,
    min_r, r_cap, r_exit, exit_inner_abs2,
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
        if q[i, 0] <= r_cap:
            r = q[i, 0]
            theta = q[i, 1]
            rdot = p[i, 0]
            thetadot = p[i, 1]
            Bx, By, Bx_theta, By_theta, _, _ = eval_boundary(
                theta, coef_a, coef_b, coef_c, coef_d
            )
            q[i, 0] = r * Bx
            q[i, 1] = r * By
            p[i, 0] = Bx * rdot + r * Bx_theta * thetadot
            p[i, 1] = By * rdot + r * By_theta * thetadot
            mode[i] = CAP
        else:
            q[i, 1] = wrap_theta(q[i, 1])

    x = np.empty((N, 2), dtype=f32)
    v = np.empty((N, 2), dtype=f32)
    geom = np.empty((N, 10), dtype=f32)
    fext = np.empty((N, 2), dtype=f32)

    compute_geometry(q, geom, mode, coef_a, coef_b, coef_c, coef_d, min_r)
    map_to_physical(q, p, x, v, geom, mode)

    half_dt = f32(0.5) * dt
    x_verlet_ref = x.copy()
    acc = np.empty((N, 2), dtype=f32)

    offsets, neigh = build_verlet_csr(x, box, r_list, cell_size)

    t = f32(0.0)
    compute_accelerations(
        acc, q, p, x, v, geom, fext, mode, m, rad, group_mobile,
        offsets, neigh,
        k_contact, gamma_contact, k_w, gamma_w, g_magnitude, omega, t,
    )

    save_id = 0
    t_out[save_id] = f32(0.0)
    snapshot_curvilinear(
        q_out[save_id], p_out[save_id], q, p, mode,
        theta_table, phi_table,
        coef_a, coef_b, coef_c, coef_d,
        min_r,
    )
    write_physical_output(x_out[save_id], v_out[save_id], x, v)
    save_id += 1

    for n in range(num_steps):
        t = f32(n * dt)

        offsets, neigh = velocity_verlet_step(
            q, p, x, v, geom, acc, fext, mode, dt, half_dt,
            m, rad, group_mobile,
            box, r_list, cell_size, skin,
            offsets, neigh, x_verlet_ref,
            k_contact, gamma_contact, k_w, gamma_w, g_magnitude, omega, t,
            coef_a, coef_b, coef_c, coef_d,
            theta_table, phi_table,
            min_r, r_cap, r_exit, exit_inner_abs2,
        )

        if (n + 1) % save_every == 0:
            ts = f32((n + 1) * dt)
            t_out[save_id] = ts
            snapshot_curvilinear(
                q_out[save_id], p_out[save_id], q, p, mode,
                theta_table, phi_table,
                coef_a, coef_b, coef_c, coef_d,
                min_r,
            )
            write_physical_output(x_out[save_id], v_out[save_id], x, v)
            save_id += 1

    return t_out, q_out, p_out, x_out, v_out


def _eval_boundary_py(theta, a, b, c, d):
    Bx = float(a[0])
    By = float(c[0])
    K = len(a) - 1
    for k in range(1, K + 1):
        ck = math.cos(k * theta)
        sk = math.sin(k * theta)
        Bx += float(a[k]) * ck + float(b[k]) * sk
        By += float(c[k]) * ck + float(d[k]) * sk
    return Bx, By


def make_inverse_angle_table(a, b, c, d, n=4096):
    # Fixed-geometry lookup for x = r B(theta).
    theta = np.linspace(0.0, 2.0 * np.pi, int(n) + 1, dtype=np.float64)
    Bx = np.empty_like(theta)
    By = np.empty_like(theta)

    for i, th in enumerate(theta):
        Bx[i], By[i] = _eval_boundary_py(th, a, b, c, d)

    phi = np.unwrap(np.arctan2(By, Bx))
    norm = np.sqrt(Bx * Bx + By * By)

    if phi[-1] < phi[0]:
        phi = phi[::-1].copy()
        theta = theta[::-1].copy()

    return (
        theta.astype(f32),
        phi.astype(f32),
        f32(np.min(norm)),
    )


def simulate_curvilinear_particles(
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
    k_w,
    gamma_w,
    g_magnitude,
    omega,
    a,
    b,
    c,
    d,
    save_every=1,
    r_list=None,
    skin=None,
    cell_size=None,
    min_r=1.0e-6,
    r_cap=0.05,
    r_exit=None,
    inverse_n=4096,
):
    """
    min_r
        Tiny denominator guard used only when reconstructing qdot near r = 0.
        It does not clamp particle position.

    r_cap
        Enter the Cartesian cap when curvilinear r <= r_cap.

    r_exit
        Leave the Cartesian cap when reconstructed r >= r_exit.
        Defaults to 1.5 * r_cap for hysteresis.
    """
    q0 = as_f32(q0)
    p0 = as_f32(p0)
    m = as_f32(m)
    rad = as_f32(rad)
    box = as_f32(box)
    group_mobile = as_i64(group_mobile)
    coef_a = as_f32(a)
    coef_b = as_f32(b)
    coef_c = as_f32(c)
    coef_d = as_f32(d)

    theta_table, phi_table, b_norm_min = make_inverse_angle_table(
        coef_a, coef_b, coef_c, coef_d, n=inverse_n
    )

    r_cap_val = f32(r_cap)
    r_exit_val = f32(r_exit if r_exit is not None else f32(1.5) * r_cap_val)
    exit_inner_abs = r_exit_val * b_norm_min
    exit_inner_abs2 = exit_inner_abs * exit_inner_abs

    max_rad = f32(np.max(rad))
    skin_val = f32(skin if skin is not None else f32(0.25) * max_rad)
    r_contact = f32(2.0) * max_rad
    r_list_val = f32(r_list if r_list is not None else r_contact + skin_val)
    cell_size_val = f32(cell_size if cell_size is not None else r_list_val)

    return simulate_curvilinear_particles_jit(
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
        f32(k_w),
        f32(gamma_w),
        f32(g_magnitude),
        f32(omega),
        coef_a,
        coef_b,
        coef_c,
        coef_d,
        theta_table,
        phi_table,
        r_list_val,
        skin_val,
        cell_size_val,
        int(save_every),
        f32(min_r),
        r_cap_val,
        r_exit_val,
        exit_inner_abs2,
    )


def curvilinear_to_physical_particles(q, p, a, b, c, d, min_r=1.0e-6):
    q = as_f32(q)
    p = as_f32(p)
    coef_a = as_f32(a)
    coef_b = as_f32(b)
    coef_c = as_f32(c)
    coef_d = as_f32(d)

    N = q.shape[0]
    x_rel = np.empty_like(q, dtype=f32)
    v_rel = np.empty_like(p, dtype=f32)
    mode = np.zeros(N, dtype=i64)
    geom = np.empty((N, 10), dtype=f32)

    compute_geometry(q, geom, mode, coef_a, coef_b, coef_c, coef_d, f32(min_r))
    map_to_physical(q, p, x_rel, v_rel, geom, mode)

    x_abs = np.empty_like(x_rel, dtype=f32)
    v_abs = np.empty_like(v_rel, dtype=f32)
    write_physical_output(x_abs, v_abs, x_rel, v_rel)

    return x_abs, v_abs
