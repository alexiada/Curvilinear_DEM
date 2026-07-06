"""
Fast curvilinear multiparticle solver for the Case Study 3 time-varying map.

This is the radius-wall v16 solver adapted to the Section 4.3 map

    x = Phi(t, q),        q = (r, theta),

with the explicit peristaltic map implemented in maps.py.  The radial chart is
still singular at r = 0, so particles entering a small central cap are
integrated in physical Cartesian coordinates until they are safely outside the
cap again.  Everywhere else the time-dependent curvilinear equations are used.

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

import maps


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
def eval_boundary(theta, t):
    """Evaluate map data from maps.py."""
    return maps.eval_boundary(theta, t)


@njit(cache=False, fastmath=True)
def physical_to_curvilinear_one(
    x0, x1, v0, v1, t, min_r,
):
    """Invert x = r B(theta, t) and convert physical velocity to qdot.

    This is used only for Cartesian-cap handoff and output snapshots.
    """
    phi = math.atan2(x1, x0)
    theta = wrap_theta(phi)

    for _ in range(6):
        (
            Bx, By, Bx_theta, By_theta,
            _, _, _, _, _, _, _, _,
        ) = eval_boundary(theta, t)
        g_align = Bx * x1 - By * x0
        gp = Bx_theta * x1 - By_theta * x0
        if gp != f32(0.0):
            theta = theta - g_align / gp
    theta = wrap_theta(theta)

    (
        Bx, By, Bx_theta, By_theta,
        _, _, Bx_t, By_t, _, _, _, _,
    ) = eval_boundary(theta, t)

    den = Bx * Bx + By * By
    r = (x0 * Bx + x1 * By) / den

    r_eff = r
    if r_eff < min_r:
        r_eff = min_r

    detJ = r_eff * (Bx * By_theta - By * Bx_theta)
    inv_det = f32(1.0) / detJ
    inv00 = r_eff * By_theta * inv_det
    inv01 = -r_eff * Bx_theta * inv_det
    inv10 = -By * inv_det
    inv11 = Bx * inv_det

    ux = v0 - r * Bx_t
    uy = v1 - r * By_t

    rdot = inv00 * ux + inv01 * uy
    thetadot = inv10 * ux + inv11 * uy

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
    q, p, mode, group, r_cap, r_exit, exit_inner_abs2, min_r, t,
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

                x0, x1, v0, v1 = maps.curvilinear_to_physical_state(
                    r, theta, rdot, thetadot, t
                )
                q[i, 0] = x0
                q[i, 1] = x1
                p[i, 0] = v0
                p[i, 1] = v1
                mode[i] = CAP
        else:
            x0 = q[i, 0]
            x1 = q[i, 1]
            rho2 = x0 * x0 + x1 * x1

            # Definitely inside the exit radius for every boundary direction.
            if rho2 < exit_inner_abs2:
                continue

            r_rec, theta_rec, rdot_rec, thetadot_rec = physical_to_curvilinear_one(
                x0, x1, p[i, 0], p[i, 1], t, min_r,
            )

            if r_rec >= r_exit:
                q[i, 0] = r_rec
                q[i, 1] = theta_rec
                p[i, 0] = rdot_rec
                p[i, 1] = thetadot_rec
                mode[i] = CURV


@njit(cache=False, fastmath=True)
def compute_geometry(q, geom, mode, min_r, t):
    # Cache map derivatives and J^{-1} for curvilinear particles.
    for i in range(q.shape[0]):
        if mode[i] != CURV:
            continue

        r = q[i, 0]
        if r < min_r:
            r = min_r

        theta = q[i, 1]
        (
            Bx, By,
            Bx_theta, By_theta,
            Bx_thetatheta, By_thetatheta,
            Bx_t, By_t,
            Bx_thetat, By_thetat,
            Bx_tt, By_tt,
        ) = eval_boundary(theta, t)

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
        geom[i, 10] = Bx_t
        geom[i, 11] = By_t
        geom[i, 12] = Bx_thetat
        geom[i, 13] = By_thetat
        geom[i, 14] = Bx_tt
        geom[i, 15] = By_tt


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
            Bx_t = geom[i, 10]
            By_t = geom[i, 11]

            x[i, 0] = r * Bx
            x[i, 1] = r * By
            v[i, 0] = r * Bx_t + Bx * rdot + r * Bx_theta * thetadot
            v[i, 1] = r * By_t + By * rdot + r * By_theta * thetadot
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
def accumulate_wall_forces(fext, q, v, geom, mode, group, rad, k_w, gamma_w):
    # Radius-aware outer wall. Cap particles are near the centre and skip this.
    start = group[0]
    end = group[1]

    for i in range(start, end):
        if mode[i] != CURV:
            continue

        Bx = geom[i, 0]
        By = geom[i, 1]
        Bx_theta = geom[i, 2]
        By_theta = geom[i, 3]
        Bx_t = geom[i, 10]
        By_t = geom[i, 11]

        norm_t = math.sqrt(Bx_theta * Bx_theta + By_theta * By_theta)
        inv_norm_t = f32(1.0) / norm_t

        n0 = -By_theta * inv_norm_t
        n1 = Bx_theta * inv_norm_t

        centre_clearance = (q[i, 0] - f32(1.0)) * (Bx * n0 + By * n1)
        delta = rad[i] - centre_clearance

        if delta > f32(0.0):
            vn = (v[i, 0] - Bx_t) * n0 + (v[i, 1] - By_t) * n1

            f_mag = k_w * delta - gamma_w * vn
            if f_mag < f32(0.0):
                f_mag = f32(0.0)

            fext[i, 0] += f_mag * n0
            fext[i, 1] += f_mag * n1


@njit(cache=False, fastmath=True)
def finalize_accelerations(acc, q, p, geom, fext, mode, group, m, ax_body, ay_body):
    # Convert physical acceleration using Eq. 72 for x = Phi(t, q).
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
            Bx_t = geom[i, 10]
            By_t = geom[i, 11]
            Bx_thetat = geom[i, 12]
            By_thetat = geom[i, 13]
            Bx_tt = geom[i, 14]
            By_tt = geom[i, 15]

            Hx = f32(2.0) * rdot * thetadot * Bx_theta + r * thetadot * thetadot * Bx_thetatheta
            Hy = f32(2.0) * rdot * thetadot * By_theta + r * thetadot * thetadot * By_thetatheta

            Phi_tt_x = r * Bx_tt
            Phi_tt_y = r * By_tt
            Jt_qdot_x = Bx_t * rdot + r * Bx_thetat * thetadot
            Jt_qdot_y = By_t * rdot + r * By_thetat * thetadot

            rhs_x = ax - Phi_tt_x - f32(2.0) * Jt_qdot_x - Hx
            rhs_y = ay - Phi_tt_y - f32(2.0) * Jt_qdot_y - Hy

            g6 = geom[i, 6]
            g7 = geom[i, 7]
            g8 = geom[i, 8]
            g9 = geom[i, 9]

            acc[i, 0] = g6 * rhs_x + g7 * rhs_y
            acc[i, 1] = g8 * rhs_x + g9 * rhs_y
        else:
            acc[i, 0] = ax
            acc[i, 1] = ay


@njit(cache=False, fastmath=True)
def compute_accelerations(
    acc, q, p, x, v, geom, fext, mode, m, rad, group_mobile,
    offsets, neigh,
    k_contact, gamma_contact, k_w, gamma_w, g, t,
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
        rad, k_w, gamma_w,
    )

    finalize_accelerations(
        acc, q, p, geom, fext, mode, group_mobile, m,
        g[0],
        g[1],
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
    q_save, p_save, q, p, mode, min_r, t,
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
                q[i, 0], q[i, 1], p[i, 0], p[i, 1], t, min_r,
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
    k_contact, gamma_contact, k_w, gamma_w, g, t,
    min_r, r_cap, r_exit, exit_inner_abs2,
):
    N = q.shape[0]

    for i in range(N):
        p[i, 0] += half_dt * acc[i, 0]
        p[i, 1] += half_dt * acc[i, 1]

    for i in range(N):
        q[i, 0] += dt * p[i, 0]
        q[i, 1] += dt * p[i, 1]

    t_next = t + dt
    apply_safeguard(q, mode)
    handoff(
        q, p, mode, group_mobile, r_cap, r_exit, exit_inner_abs2, min_r, t_next,
    )

    compute_geometry(q, geom, mode, min_r, t_next)
    map_to_physical(q, p, x, v, geom, mode)

    if needs_verlet_rebuild(x, x_verlet_ref, skin):
        offsets, neigh = build_verlet_csr(x, box, r_list, cell_size)
        x_verlet_ref[:, :] = x

    compute_accelerations(
        acc, q, p, x, v, geom, fext, mode, m, rad, group_mobile,
        offsets, neigh,
        k_contact, gamma_contact, k_w, gamma_w, g, t_next,
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
    k_w, gamma_w, g,
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
            x0, x1, v0, v1 = maps.curvilinear_to_physical_state(
                r, theta, rdot, thetadot, f32(0.0)
            )
            q[i, 0] = x0
            q[i, 1] = x1
            p[i, 0] = v0
            p[i, 1] = v1
            mode[i] = CAP
        else:
            q[i, 1] = wrap_theta(q[i, 1])

    x = np.empty((N, 2), dtype=f32)
    v = np.empty((N, 2), dtype=f32)
    geom = np.empty((N, 16), dtype=f32)
    fext = np.empty((N, 2), dtype=f32)

    compute_geometry(q, geom, mode, min_r, f32(0.0))
    map_to_physical(q, p, x, v, geom, mode)

    half_dt = f32(0.5) * dt
    x_verlet_ref = x.copy()
    acc = np.empty((N, 2), dtype=f32)

    offsets, neigh = build_verlet_csr(x, box, r_list, cell_size)

    t = f32(0.0)
    compute_accelerations(
        acc, q, p, x, v, geom, fext, mode, m, rad, group_mobile,
        offsets, neigh,
        k_contact, gamma_contact, k_w, gamma_w, g, t,
    )

    save_id = 0
    t_out[save_id] = f32(0.0)
    snapshot_curvilinear(
        q_out[save_id], p_out[save_id], q, p, mode, min_r, f32(0.0),
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
            k_contact, gamma_contact, k_w, gamma_w, g, t,
            min_r, r_cap, r_exit, exit_inner_abs2,
        )

        if (n + 1) % save_every == 0:
            ts = f32((n + 1) * dt)
            t_out[save_id] = ts
            snapshot_curvilinear(
                q_out[save_id], p_out[save_id], q, p, mode, min_r, ts,
            )
            write_physical_output(x_out[save_id], v_out[save_id], x, v)
            save_id += 1

    return t_out, q_out, p_out, x_out, v_out


def boundary_norm_min(n_theta=4096, n_t=64):
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False)
    period = 2.0 * np.pi / float(maps.OMEGA)
    times = np.linspace(0.0, period, int(n_t), endpoint=False)
    min_norm = np.inf
    for tt in times:
        t32 = f32(tt)
        for th in theta:
            x, y = maps.eval_map(f32(1.0), f32(th), t32)
            norm = float(math.sqrt(x * x + y * y))
            if norm < min_norm:
                min_norm = norm
    return f32(min_norm)


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
    g,
    save_every=1,
    r_list=None,
    skin=None,
    cell_size=None,
    min_r=1.0e-6,
    r_cap=0.05,
    r_exit=None,
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
    g = as_f32(g)

    r_cap_val = f32(r_cap)
    r_exit_val = f32(r_exit if r_exit is not None else f32(1.5) * r_cap_val)
    b_norm_min = boundary_norm_min()
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
        g,
        r_list_val,
        skin_val,
        cell_size_val,
        int(save_every),
        f32(min_r),
        r_cap_val,
        r_exit_val,
        exit_inner_abs2,
    )


def simulate_time_varying_curvilinear_particles(*args, **kwargs):
    return simulate_curvilinear_particles(*args, **kwargs)


def curvilinear_to_physical_particles(q, p, t=0.0, min_r=1.0e-6):
    q = as_f32(q)
    p = as_f32(p)

    N = q.shape[0]
    x_abs = np.empty_like(q, dtype=f32)
    v_abs = np.empty_like(p, dtype=f32)
    mode = np.zeros(N, dtype=i64)
    geom = np.empty((N, 16), dtype=f32)

    compute_geometry(q, geom, mode, f32(min_r), f32(t))
    map_to_physical(q, p, x_abs, v_abs, geom, mode)

    return x_abs, v_abs
