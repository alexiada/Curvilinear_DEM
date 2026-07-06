"""
Fast Euclidean multiparticle solver with an explicit segment boundary.

Structure kept close to the Chapter 13/14 multiparticle code:
    - float32 arrays
    - int64 index arrays
    - explicit Numba kernels
    - CSR Verlet neighbour lists
    - one compiled simulation loop

This version keeps the particle-particle Verlet list unchanged. The wall is
passed as ordered boundary nodes. Consecutive nodes define wall segments:

    0-1, 1-2, 2-3, ...

If the first node is repeated at the end, the boundary is treated as closed.
Closed boundaries use the polygon orientation to choose inward normals and
apply a half-plane soft-wall force for every particle against every segment.
No particle-wall neighbour list is used.
"""

import math
import numpy as np
from numba import njit


f32 = np.float32
i64 = np.int64


def as_f32(a):
    return np.asarray(a, dtype=f32)


def as_i64(a):
    return np.asarray(a, dtype=i64)


def make_circle_boundary_nodes(radius, n_segments, closed=True):
    """
    Return ordered nodes approximating a circle.

    If closed is True, the first node is repeated at the end so the last
    segment closes the polygon.
    """
    n_segments = int(n_segments)
    if n_segments < 1:
        raise ValueError("n_segments must be positive")
    if closed:
        theta = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
        theta = np.concatenate((theta, theta[:1]))
    else:
        theta = np.linspace(0.0, 2.0 * np.pi, n_segments + 1, endpoint=True)
    nodes = np.empty((theta.shape[0], 2), dtype=f32)
    nodes[:, 0] = f32(radius) * np.cos(theta).astype(f32)
    nodes[:, 1] = f32(radius) * np.sin(theta).astype(f32)
    return nodes


def prepare_boundary_segments(boundary_nodes):
    """
    Convert ordered boundary nodes into segment arrays for the JIT solver.

    Closed boundaries are detected when the first and last nodes coincide.
    For a closed boundary, normals are oriented inward using the polygon
    signed area. A counter-clockwise polygon has its interior on the left of
    each edge.
    """
    nodes = as_f32(boundary_nodes)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("boundary_nodes must have shape (N, 2)")
    if nodes.shape[0] < 2:
        raise ValueError("boundary_nodes must contain at least two nodes")

    closed = bool(np.linalg.norm(nodes[0] - nodes[-1]) <= 1.0e-6)
    if closed and nodes.shape[0] < 4:
        raise ValueError("closed boundary_nodes must contain at least three segments")

    seg_a = nodes[:-1].copy()
    seg_b = nodes[1:].copy()
    nseg = seg_a.shape[0]

    inward = np.empty((nseg, 2), dtype=f32)
    outward = np.empty((nseg, 2), dtype=f32)
    edge_len2 = np.empty(nseg, dtype=f32)

    area2 = f32(0.0)
    if closed:
        for i in range(nseg):
            area2 += seg_a[i, 0] * seg_b[i, 1] - seg_b[i, 0] * seg_a[i, 1]

    for i in range(nseg):
        ex = seg_b[i, 0] - seg_a[i, 0]
        ey = seg_b[i, 1] - seg_a[i, 1]
        l2 = ex * ex + ey * ey
        if l2 <= 0.0:
            raise ValueError("boundary_nodes contain a zero-length segment")

        inv_l = f32(1.0) / f32(math.sqrt(float(l2)))
        edge_len2[i] = f32(l2)

        if closed:
            if area2 >= 0.0:
                # CCW polygon: interior is left, outward is right.
                ox = ey * inv_l
                oy = -ex * inv_l
            else:
                # CW polygon: interior is right, outward is left.
                ox = -ey * inv_l
                oy = ex * inv_l
            outward[i, 0] = ox
            outward[i, 1] = oy
            inward[i, 0] = -ox
            inward[i, 1] = -oy
        else:
            # Open boundaries do not have a unique inside. The JIT kernel uses
            # closest-point contact and this normal is not used for the sign.
            inward[i, 0] = -ey * inv_l
            inward[i, 1] = ex * inv_l
            outward[i, 0] = ey * inv_l
            outward[i, 1] = -ex * inv_l

    return seg_a, seg_b, inward, outward, edge_len2, i64(1 if closed else 0)


@njit(cache=False)
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


@njit(cache=False)
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


@njit(cache=False)
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


@njit(cache=False)
def zero_forces(f):
    for i in range(f.shape[0]):
        f[i, 0] = f32(0.0)
        f[i, 1] = f32(0.0)


@njit(cache=False)
def add_gravity(f, m, group, g):
    start = group[0]
    end = group[1]

    for i in range(start, end):
        f[i, 0] += m[i] * g[0]
        f[i, 1] += m[i] * g[1]


@njit(cache=False)
def add_tether_forces(f, x, x_ref, group, k_tether):
    start = group[0]
    end = group[1]

    for i in range(start, end):
        f[i, 0] -= k_tether * (x[i, 0] - x_ref[i, 0])
        f[i, 1] -= k_tether * (x[i, 1] - x_ref[i, 1])


@njit(cache=False)
def add_spring_forces(f, x, edge_i, edge_j, k_spring, l0):
    for e in range(edge_i.shape[0]):
        i = edge_i[e]
        j = edge_j[e]

        dx = x[j, 0] - x[i, 0]
        dy = x[j, 1] - x[i, 1]
        r2 = dx * dx + dy * dy
        if r2 == f32(0.0):
            continue

        r = math.sqrt(r2)
        delta = r - l0[e]
        fm = k_spring * delta / r

        fx = fm * dx
        fy = fm * dy

        f[i, 0] += fx
        f[i, 1] += fy
        f[j, 0] -= fx
        f[j, 1] -= fy


@njit(cache=False)
def add_hertzian_forces(f, x, rad, group, offsets, neigh, k_hertz):
    start = group[0]
    end = group[1]

    for i in range(start, end):
        xi0 = x[i, 0]
        xi1 = x[i, 1]
        ri = rad[i]

        for p in range(offsets[i], offsets[i + 1]):
            j = neigh[p]

            if j <= i:
                continue
            if j < start or j >= end:
                continue

            dx = x[j, 0] - xi0
            dy = x[j, 1] - xi1
            r2 = dx * dx + dy * dy
            if r2 == f32(0.0):
                continue

            r = math.sqrt(r2)
            overlap = ri + rad[j] - r

            if overlap > f32(0.0):
                fm = k_hertz * overlap * math.sqrt(overlap) / r
                fx = fm * dx
                fy = fm * dy

                f[i, 0] -= fx
                f[i, 1] -= fy
                f[j, 0] += fx
                f[j, 1] += fy


@njit(cache=False)
def add_linear_overlap_forces(f, x, rad, group_a, group_b, offsets, neigh, k_rep):
    start_a = group_a[0]
    end_a = group_a[1]
    start_b = group_b[0]
    end_b = group_b[1]

    for i in range(start_a, end_a):
        xi0 = x[i, 0]
        xi1 = x[i, 1]
        ri = rad[i]

        for p in range(offsets[i], offsets[i + 1]):
            j = neigh[p]

            if j < start_b or j >= end_b:
                continue

            dx = x[j, 0] - xi0
            dy = x[j, 1] - xi1
            r2 = dx * dx + dy * dy
            if r2 == f32(0.0):
                continue

            r = math.sqrt(r2)
            overlap = ri + rad[j] - r

            if overlap > f32(0.0):
                fm = k_rep * overlap / r
                fx = fm * dx
                fy = fm * dy

                f[i, 0] -= fx
                f[i, 1] -= fy
                f[j, 0] += fx
                f[j, 1] += fy


@njit(cache=False)
def add_soft_sphere_forces(f, x, v, rad, group, offsets, neigh, k_contact, gamma_contact):
    start = group[0]
    end = group[1]

    for i in range(start, end):
        xi0 = x[i, 0]
        xi1 = x[i, 1]
        vi0 = v[i, 0]
        vi1 = v[i, 1]
        ri = rad[i]

        for p in range(offsets[i], offsets[i + 1]):
            j = neigh[p]

            if j <= i:
                continue
            if j < start or j >= end:
                continue

            dx = x[j, 0] - xi0
            dy = x[j, 1] - xi1
            r2 = dx * dx + dy * dy
            if r2 == f32(0.0):
                continue

            r = math.sqrt(r2)
            overlap = ri + rad[j] - r

            if overlap > f32(0.0):
                nx = dx / r
                ny = dy / r

                dvx = v[j, 0] - vi0
                dvy = v[j, 1] - vi1
                vn = dvx * nx + dvy * ny

                fm = k_contact * overlap - gamma_contact * vn
                if fm < f32(0.0):
                    fm = f32(0.0)

                fx = fm * nx
                fy = fm * ny

                f[i, 0] -= fx
                f[i, 1] -= fy
                f[j, 0] += fx
                f[j, 1] += fy


@njit(cache=False)
def add_segment_wall_forces(
    f, x, v, rad, group,
    seg_a, seg_b, inward, outward, edge_len2, closed,
    xA, omega, k_w, gamma_w, t,
):
    start = group[0]
    end = group[1]

    dx_t0 = xA[0] * math.sin(omega * t)
    dx_t1 = xA[1] * math.sin(omega * t)

    dx_dot_t0 = xA[0] * omega * math.cos(omega * t)
    dx_dot_t1 = xA[1] * omega * math.cos(omega * t)

    nseg = seg_a.shape[0]

    for i in range(start, end):
        # Work in the translating boundary frame.
        px = x[i, 0] - dx_t0
        py = x[i, 1] - dx_t1
        vx = v[i, 0] - dx_dot_t0
        vy = v[i, 1] - dx_dot_t1
        ri = rad[i]

        if closed == 1:
            # Closed convex polygon: find the most violated/supporting
            # half-plane, then apply one wall contact for this particle.
            # Summing all nearby segment planes over-counts one physical wall
            # contact when the segment length is smaller than the particle.
            best_signed_out = f32(-3.4e38)
            best_n0 = f32(0.0)
            best_n1 = f32(0.0)

            for s in range(nseg):
                ax = seg_a[s, 0]
                ay = seg_a[s, 1]
                no0 = outward[s, 0]
                no1 = outward[s, 1]
                signed_out = (px - ax) * no0 + (py - ay) * no1

                if signed_out > best_signed_out:
                    best_signed_out = signed_out
                    best_n0 = inward[s, 0]
                    best_n1 = inward[s, 1]

            delta = best_signed_out + ri
            if delta > f32(0.0):
                vn = vx * best_n0 + vy * best_n1
                f_mag = k_w * delta - gamma_w * vn
                if f_mag < f32(0.0):
                    f_mag = f32(0.0)

                f[i, 0] += f_mag * best_n0
                f[i, 1] += f_mag * best_n1
        else:
            # Open polyline: use the closest finite segment, then apply one
            # closest-point contact if the particle overlaps the polyline.
            best_dist2 = f32(3.4e38)
            best_n0 = f32(0.0)
            best_n1 = f32(0.0)

            for s in range(nseg):
                ax = seg_a[s, 0]
                ay = seg_a[s, 1]
                bx = seg_b[s, 0]
                by = seg_b[s, 1]
                ex = bx - ax
                ey = by - ay
                tau = ((px - ax) * ex + (py - ay) * ey) / edge_len2[s]
                if tau < f32(0.0):
                    tau = f32(0.0)
                elif tau > f32(1.0):
                    tau = f32(1.0)

                cx = ax + tau * ex
                cy = ay + tau * ey
                dx = px - cx
                dy = py - cy
                d2 = dx * dx + dy * dy
                if d2 == f32(0.0):
                    continue

                dist = math.sqrt(d2)
                inv_dist = f32(1.0) / dist
                if d2 < best_dist2:
                    best_dist2 = d2
                    best_n0 = dx * inv_dist
                    best_n1 = dy * inv_dist

            if best_dist2 < f32(3.4e38):
                dist = math.sqrt(best_dist2)
                delta = ri - dist
                if delta > f32(0.0):
                    vn = vx * best_n0 + vy * best_n1
                    f_mag = k_w * delta - gamma_w * vn
                    if f_mag < f32(0.0):
                        f_mag = f32(0.0)

                    f[i, 0] += f_mag * best_n0
                    f[i, 1] += f_mag * best_n1


@njit(cache=False)
def compute_forces(
    f, x, v, m, rad, group_mobile,
    offsets, neigh,
    seg_a, seg_b, inward, outward, edge_len2, closed, xA, omega,
    k_contact, gamma_contact, k_w, gamma_w, g, t,
):
    zero_forces(f)

    add_soft_sphere_forces(f, x, v, rad, group_mobile, offsets, neigh, k_contact, gamma_contact)
    add_gravity(f, m, group_mobile, g)
    add_segment_wall_forces(
        f, x, v, rad, group_mobile,
        seg_a, seg_b, inward, outward, edge_len2, closed,
        xA, omega, k_w, gamma_w, t,
    )


@njit(cache=False)
def velocity_verlet_step(
    x, v, f, inv_m, dt, half_dt,
    m, rad, group_mobile,
    box, r_list, cell_size, skin,
    offsets, neigh, x_verlet_ref,
    seg_a, seg_b, inward, outward, edge_len2, closed, xA, omega,
    k_contact, gamma_contact, k_w, gamma_w, g, t,
):
    N = x.shape[0]

    for i in range(N):
        v[i, 0] += half_dt * f[i, 0] * inv_m[i]
        v[i, 1] += half_dt * f[i, 1] * inv_m[i]

    for i in range(N):
        x[i, 0] += dt * v[i, 0]
        x[i, 1] += dt * v[i, 1]

    if needs_verlet_rebuild(x, x_verlet_ref, skin):
        offsets, neigh = build_verlet_csr(x, box, r_list, cell_size)
        x_verlet_ref[:, :] = x

    compute_forces(
        f, x, v, m, rad, group_mobile,
        offsets, neigh,
        seg_a, seg_b, inward, outward, edge_len2, closed, xA, omega,
        k_contact, gamma_contact, k_w, gamma_w, g, t + dt,
    )

    for i in range(N):
        v[i, 0] += half_dt * f[i, 0] * inv_m[i]
        v[i, 1] += half_dt * f[i, 1] * inv_m[i]

    return offsets, neigh


@njit(cache=False)
def simulate_euclidean_particles_jit(
    box, x0, v0, m, rad, dt, T_max,
    group_mobile,
    k_contact, gamma_contact,
    seg_a, seg_b, inward, outward, edge_len2, closed, xA, omega,
    k_w, gamma_w, g,
    r_list, skin, cell_size, save_every,
):
    N = x0.shape[0]
    num_steps = int(T_max / dt + f32(0.5))
    num_save = num_steps // save_every + 1

    t_out = np.empty(num_save, dtype=f32)
    x_out = np.empty((num_save, N, 2), dtype=f32)
    v_out = np.empty((num_save, N, 2), dtype=f32)

    x = x0.copy()
    v = v0.copy()

    inv_m = np.empty(N, dtype=f32)
    for i in range(N):
        inv_m[i] = f32(1.0) / m[i]

    half_dt = f32(0.5) * dt
    x_verlet_ref = x0.copy()
    f = np.empty((N, 2), dtype=f32)

    offsets, neigh = build_verlet_csr(x, box, r_list, cell_size)

    t = f32(0.0)
    compute_forces(
        f, x, v, m, rad, group_mobile,
        offsets, neigh,
        seg_a, seg_b, inward, outward, edge_len2, closed, xA, omega,
        k_contact, gamma_contact, k_w, gamma_w, g, t,
    )

    save_id = 0
    t_out[save_id] = f32(0.0)
    x_out[save_id] = x
    v_out[save_id] = v
    save_id += 1

    for n in range(num_steps):
        t = f32(n * dt)

        offsets, neigh = velocity_verlet_step(
            x, v, f, inv_m, dt, half_dt,
            m, rad, group_mobile,
            box, r_list, cell_size, skin,
            offsets, neigh, x_verlet_ref,
            seg_a, seg_b, inward, outward, edge_len2, closed, xA, omega,
            k_contact, gamma_contact, k_w, gamma_w, g, t,
        )

        if (n + 1) % save_every == 0:
            t_out[save_id] = f32((n + 1) * dt)
            x_out[save_id] = x
            v_out[save_id] = v
            save_id += 1

    return t_out, x_out, v_out


def simulate_euclidean_particles(
    box,
    x0,
    v0,
    m,
    rad,
    dt,
    T_max,
    group_mobile,
    k_contact,
    gamma_contact,
    boundary_nodes,
    xA,
    omega,
    k_w,
    gamma_w,
    g,
    save_every=1,
    r_list=None,
    skin=None,
    cell_size=None,
):
    x0 = as_f32(x0)
    v0 = as_f32(v0)
    m = as_f32(m)
    rad = as_f32(rad)
    box = as_f32(box)
    group_mobile = as_i64(group_mobile)
    xA = as_f32(xA)
    g = as_f32(g)
    seg_a, seg_b, inward, outward, edge_len2, closed = prepare_boundary_segments(
        boundary_nodes
    )

    max_rad = f32(np.max(rad))
    skin_val = f32(skin if skin is not None else f32(0.25) * max_rad)
    r_contact = f32(2.0) * max_rad
    r_list_val = f32(r_list if r_list is not None else r_contact + skin_val)
    cell_size_val = f32(cell_size if cell_size is not None else r_list_val)

    return simulate_euclidean_particles_jit(
        box,
        x0,
        v0,
        m,
        rad,
        f32(dt),
        f32(T_max),
        group_mobile,
        f32(k_contact),
        f32(gamma_contact),
        seg_a,
        seg_b,
        inward,
        outward,
        edge_len2,
        closed,
        xA,
        f32(omega),
        f32(k_w),
        f32(gamma_w),
        g,
        r_list_val,
        skin_val,
        cell_size_val,
        int(save_every),
    )


############################################################
# Visualisation
############################################################

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def animate_particles_in_moving_circle(
    t,
    x,
    radius,
    xA,
    omega,
    box=None,
    fps=60,
    stride=1,
    s=20,
    show_box=True,
    show_traj0=True,
):
    idx = np.arange(0, len(t), max(1, int(stride)), dtype=np.int64)
    xA = np.asarray(xA, dtype=float)

    if box is None:
        pad = float(radius) + float(np.max(np.abs(xA))) + 0.5
        x_min = float(np.min(x[:, :, 0]) - pad)
        x_max = float(np.max(x[:, :, 0]) + pad)
        y_min = float(np.min(x[:, :, 1]) - pad)
        y_max = float(np.max(x[:, :, 1]) + pad)
    else:
        x_min, x_max, y_min, y_max = box

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if show_box and box is not None:
        ax.plot(
            [x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min],
            linewidth=1.0,
        )

    scat = ax.scatter(x[idx[0], :, 0], x[idx[0], :, 1], s=s)

    p0, = ax.plot(
        [x[idx[0], 0, 0]],
        [x[idx[0], 0, 1]],
        marker="o",
        linestyle="None",
        markersize=6,
    )

    traj0, = ax.plot([], [], linewidth=1.5)

    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    c0 = xA * np.sin(float(omega) * float(t[idx[0]]))
    wall, = ax.plot(
        c0[0] + radius * np.cos(theta),
        c0[1] + radius * np.sin(theta),
        linewidth=1.2,
    )

    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    def init():
        n = idx[0]

        scat.set_offsets(x[n])
        p0.set_data([x[n, 0, 0]], [x[n, 0, 1]])

        if show_traj0:
            traj0.set_data([x[n, 0, 0]], [x[n, 0, 1]])
        else:
            traj0.set_data([], [])

        c = xA * np.sin(float(omega) * float(t[n]))
        wall.set_data(
            c[0] + radius * np.cos(theta),
            c[1] + radius * np.sin(theta),
        )

        time_text.set_text(f"t = {float(t[n]):.3f}")
        return scat, p0, traj0, wall, time_text

    def update(frame_i):
        n = idx[frame_i]

        scat.set_offsets(x[n])
        p0.set_data([x[n, 0, 0]], [x[n, 0, 1]])

        if show_traj0:
            traj0.set_data(x[idx[:frame_i + 1], 0, 0], x[idx[:frame_i + 1], 0, 1])
        else:
            traj0.set_data([], [])

        c = xA * np.sin(float(omega) * float(t[n]))
        wall.set_data(
            c[0] + radius * np.cos(theta),
            c[1] + radius * np.sin(theta),
        )

        time_text.set_text(f"t = {float(t[n]):.3f}")
        return scat, p0, traj0, wall, time_text

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
