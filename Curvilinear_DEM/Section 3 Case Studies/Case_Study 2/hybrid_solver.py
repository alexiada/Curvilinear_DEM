"""
Hybrid Cartesian/curvilinear multiparticle solver for a rough trapezium.

The solver integrates particles in Cartesian coordinates in the bulk. Near the
wall it switches a particle into a local rough-wall patch coordinate system

    q = (u, v), edge_id in {0, 1, 2, 3},

where v = 0 is the rough wall and v increases inward. The straight trapezium
distance is used only as a cheap switching test. Wall forces in curvilinear
mode use the rough wall geometry.

State variables
---------------
mode[i] == CART:
    q[i] = (x, y)
    p[i] = (vx, vy)

mode[i] == CURV:
    q[i] = (u, v)
    p[i] = (udot, vdot)
    edge[i] stores the active wall patch.

Particle contacts and neighbour lists always use physical x, v.
"""

import math
import numpy as np

try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator


f32 = np.float32
i64 = np.int64

CART = i64(0)
CURV = i64(1)


def as_f32(a):
    return np.asarray(a, dtype=f32)


def as_i64(a):
    return np.asarray(a, dtype=i64)


@njit(cache=False, fastmath=True)
def _clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@njit(cache=False, fastmath=True)
def _dot2(x, y):
    return x * x + y * y


@njit(cache=False, fastmath=True)
def _side_length(top, bottom, height):
    r1 = f32(0.5) * bottom
    r2 = f32(0.5) * top
    return math.sqrt((r1 - r2) * (r1 - r2) + height * height)


@njit(cache=False, fastmath=True)
def _edge_data(edge_id, top, bottom, height):
    r1 = f32(0.5) * bottom
    r2 = f32(0.5) * top
    he = f32(0.5) * height
    side = _side_length(top, bottom, height)

    if edge_id == 0:
        vx = -r1
        vy = -he
        tx = f32(1.0)
        ty = f32(0.0)
        length = bottom
    elif edge_id == 1:
        vx = r1
        vy = -he
        tx = (r2 - r1) / side
        ty = height / side
        length = side
    elif edge_id == 2:
        vx = r2
        vy = he
        tx = f32(-1.0)
        ty = f32(0.0)
        length = top
    else:
        vx = -r2
        vy = he
        tx = (-r1 + r2) / side
        ty = -height / side
        length = side

    nx = -ty
    ny = tx
    return vx, vy, tx, ty, nx, ny, length


@njit(cache=False, fastmath=True)
def _straight_edge_u_and_clearance(x, y, edge_id, top, bottom, height):
    vx, vy, tx, ty, nx, ny, length = _edge_data(edge_id, top, bottom, height)
    rx = x - vx
    ry = y - vy
    sigma = rx * tx + ry * ty
    u = sigma / length
    clearance = rx * nx + ry * ny
    return u, clearance


@njit(cache=False, fastmath=True)
def straight_trapezium_nearest_edge(x, y, top, bottom, height):
    best_edge = 0
    best_u, best_d = _straight_edge_u_and_clearance(
        x, y, 0, top, bottom, height
    )

    for edge_id in range(1, 4):
        u, d = _straight_edge_u_and_clearance(x, y, edge_id, top, bottom, height)
        if d < best_d:
            best_d = d
            best_u = u
            best_edge = edge_id

    return best_edge, best_u, best_d


@njit(cache=False, fastmath=True)
def _edge_wave_count(length, target_frequency):
    n = int(math.floor(target_frequency * length + f32(0.5)))
    if n < 1:
        n = 1
    return n


@njit(cache=False, fastmath=True)
def _smoothstep_with_derivatives(z):
    if z <= f32(0.0):
        return f32(0.0), f32(0.0), f32(0.0)
    if z >= f32(1.0):
        return f32(1.0), f32(0.0), f32(0.0)

    f = z * z * (f32(3.0) - f32(2.0) * z)
    fp = f32(6.0) * z - f32(6.0) * z * z
    fpp = f32(6.0) - f32(12.0) * z
    return f, fp, fpp


@njit(cache=False, fastmath=True)
def _taper_with_derivatives(u, taper_fraction):
    taper = _clamp(taper_fraction, f32(1.0e-6), f32(0.499))

    left, left_z, left_zz = _smoothstep_with_derivatives(u / taper)
    dl = left_z / taper
    ddl = left_zz / (taper * taper)

    right, right_z, right_zz = _smoothstep_with_derivatives(
        (f32(1.0) - u) / taper
    )
    dr = -right_z / taper
    ddr = right_zz / (taper * taper)

    envelope = left * right
    envelope_u = dl * right + left * dr
    envelope_uu = ddl * right + f32(2.0) * dl * dr + left * ddr
    return envelope, envelope_u, envelope_uu


@njit(cache=False, fastmath=True)
def _roughness_profile(sigma, length, amplitude, target_frequency, taper_fraction):
    u = _clamp(sigma / length, f32(0.0), f32(1.0))
    n_waves = _edge_wave_count(length, target_frequency)
    phase = f32(2.0) * math.pi * f32(n_waves) * u

    wave = f32(0.5) * (f32(1.0) - math.cos(phase))
    wave_u = math.pi * f32(n_waves) * math.sin(phase)
    wave_uu = (
        f32(2.0) * math.pi * math.pi * f32(n_waves) * f32(n_waves)
        * math.cos(phase)
    )

    envelope, envelope_u, envelope_uu = _taper_with_derivatives(u, taper_fraction)

    h = amplitude * envelope * wave
    h_u = amplitude * (envelope_u * wave + envelope * wave_u)
    h_uu = amplitude * (
        envelope_uu * wave
        + f32(2.0) * envelope_u * wave_u
        + envelope * wave_uu
    )

    hp = h_u / length
    hpp = h_uu / (length * length)
    return h, hp, hpp


@njit(cache=False, fastmath=True)
def rough_patch_wall(edge_id, u, top, bottom, height, amplitude, target_frequency, taper_fraction):
    u = _clamp(u, f32(0.0), f32(1.0))
    vx, vy, tx, ty, nx, ny, length = _edge_data(edge_id, top, bottom, height)
    sigma = u * length
    h, _, _ = _roughness_profile(sigma, length, amplitude, target_frequency, taper_fraction)
    px = vx + sigma * tx
    py = vy + sigma * ty
    return px - h * nx, py - h * ny


@njit(cache=False, fastmath=True)
def rough_patch_wall_normal(edge_id, u, top, bottom, height, amplitude, target_frequency, taper_fraction):
    u = _clamp(u, f32(0.0), f32(1.0))
    _, _, tx, ty, nx, ny, length = _edge_data(edge_id, top, bottom, height)
    sigma = u * length
    _, hp, _ = _roughness_profile(sigma, length, amplitude, target_frequency, taper_fraction)

    n0 = nx + hp * tx
    n1 = ny + hp * ty
    inv = f32(1.0) / math.sqrt(n0 * n0 + n1 * n1)
    return n0 * inv, n1 * inv


@njit(cache=False, fastmath=True)
def rough_patch_inner(edge_id, u, top, bottom, height, inner_scale):
    u = _clamp(u, f32(0.0), f32(1.0))
    vx, vy, tx, ty, _, _, length = _edge_data(edge_id, top, bottom, height)
    sigma = u * length
    px = vx + sigma * tx
    py = vy + sigma * ty
    return inner_scale * px, inner_scale * py


@njit(cache=False, fastmath=True)
def rough_patch_map(edge_id, u, v, top, bottom, height, amplitude, target_frequency, taper_fraction, inner_scale):
    u = _clamp(u, f32(0.0), f32(1.0))
    bx, by = rough_patch_wall(
        edge_id, u, top, bottom, height, amplitude, target_frequency, taper_fraction
    )
    cx, cy = rough_patch_inner(edge_id, u, top, bottom, height, inner_scale)
    return (f32(1.0) - v) * bx + v * cx, (f32(1.0) - v) * by + v * cy


@njit(cache=False, fastmath=True)
def rough_patch_jacobian(edge_id, u, v, top, bottom, height, amplitude, target_frequency, taper_fraction, inner_scale):
    u = _clamp(u, f32(0.0), f32(1.0))
    _, _, tx, ty, nx, ny, length = _edge_data(edge_id, top, bottom, height)
    sigma = u * length
    _, hp, _ = _roughness_profile(sigma, length, amplitude, target_frequency, taper_fraction)

    bx, by = rough_patch_wall(
        edge_id, u, top, bottom, height, amplitude, target_frequency, taper_fraction
    )
    cx, cy = rough_patch_inner(edge_id, u, top, bottom, height, inner_scale)

    bux = length * (tx - hp * nx)
    buy = length * (ty - hp * ny)
    cux = inner_scale * length * tx
    cuy = inner_scale * length * ty

    j00 = (f32(1.0) - v) * bux + v * cux
    j10 = (f32(1.0) - v) * buy + v * cuy
    j01 = cx - bx
    j11 = cy - by
    return j00, j01, j10, j11


@njit(cache=False, fastmath=True)
def rough_patch_second_terms(edge_id, u, v, udot, vdot, top, bottom, height, amplitude, target_frequency, taper_fraction, inner_scale):
    u = _clamp(u, f32(0.0), f32(1.0))
    _, _, tx, ty, nx, ny, length = _edge_data(edge_id, top, bottom, height)
    sigma = u * length
    _, hp, hpp = _roughness_profile(sigma, length, amplitude, target_frequency, taper_fraction)

    buux = -length * length * hpp * nx
    buuy = -length * length * hpp * ny

    bux = length * (tx - hp * nx)
    buy = length * (ty - hp * ny)
    cux = inner_scale * length * tx
    cuy = inner_scale * length * ty

    juvx = cux - bux
    juvy = cuy - buy

    h0 = (f32(1.0) - v) * buux * udot * udot + f32(2.0) * juvx * udot * vdot
    h1 = (f32(1.0) - v) * buuy * udot * udot + f32(2.0) * juvy * udot * vdot
    return h0, h1


@njit(cache=False, fastmath=True)
def rough_patch_inverse_edge(x, y, edge_id, top, bottom, height, amplitude, target_frequency, taper_fraction, inner_scale):
    u, _ = _straight_edge_u_and_clearance(x, y, edge_id, top, bottom, height)
    u = _clamp(u, f32(0.0), f32(1.0))

    bx, by = rough_patch_wall(
        edge_id, u, top, bottom, height, amplitude, target_frequency, taper_fraction
    )
    cx, cy = rough_patch_inner(edge_id, u, top, bottom, height, inner_scale)
    vx = cx - bx
    vy = cy - by
    vv = vx * vx + vy * vy
    v = f32(0.0)
    if vv > f32(0.0):
        v = ((x - bx) * vx + (y - by) * vy) / vv

    for _ in range(12):
        mx, my = rough_patch_map(
            edge_id, u, v, top, bottom, height,
            amplitude, target_frequency, taper_fraction, inner_scale
        )
        fx = mx - x
        fy = my - y
        j00, j01, j10, j11 = rough_patch_jacobian(
            edge_id, u, v, top, bottom, height,
            amplitude, target_frequency, taper_fraction, inner_scale
        )
        det = j00 * j11 - j01 * j10
        if abs(det) < f32(1.0e-12):
            break
        du = (j11 * fx - j01 * fy) / det
        dv = (-j10 * fx + j00 * fy) / det
        u = _clamp(u - du, f32(0.0), f32(1.0))
        v = v - dv
        if du * du + dv * dv < f32(1.0e-18):
            break

    return u, v


@njit(cache=False, fastmath=True)
def physical_velocity_to_patch_velocity(vx, vy, edge_id, u, v, top, bottom, height, amplitude, target_frequency, taper_fraction, inner_scale):
    j00, j01, j10, j11 = rough_patch_jacobian(
        edge_id, u, v, top, bottom, height,
        amplitude, target_frequency, taper_fraction, inner_scale
    )
    det = j00 * j11 - j01 * j10
    inv00 = j11 / det
    inv01 = -j01 / det
    inv10 = -j10 / det
    inv11 = j00 / det
    return inv00 * vx + inv01 * vy, inv10 * vx + inv11 * vy


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
    cursor = offsets.copy()

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
                            pi = cursor[i]
                            neigh[pi] = j
                            cursor[i] += 1
                            pj = cursor[j]
                            neigh[pj] = i
                            cursor[j] += 1
                    j = nxt[j]

    return offsets, neigh


@njit(cache=False, fastmath=True)
def needs_verlet_rebuild(x_now, x_ref, skin):
    limit2 = (f32(0.5) * skin) * (f32(0.5) * skin)
    for i in range(x_now.shape[0]):
        dx = x_now[i, 0] - x_ref[i, 0]
        dy = x_now[i, 1] - x_ref[i, 1]
        if dx * dx + dy * dy > limit2:
            return True
    return False


@njit(cache=False, fastmath=True)
def map_to_physical(q, p, x, v_phys, mode, edge, top, bottom, height, amplitude, target_frequency, taper_fraction, inner_scale):
    for i in range(q.shape[0]):
        if mode[i] == CART:
            x[i, 0] = q[i, 0]
            x[i, 1] = q[i, 1]
            v_phys[i, 0] = p[i, 0]
            v_phys[i, 1] = p[i, 1]
        else:
            edge_id = edge[i]
            u = q[i, 0]
            vv = q[i, 1]
            udot = p[i, 0]
            vdot = p[i, 1]
            x0, x1 = rough_patch_map(
                edge_id, u, vv, top, bottom, height,
                amplitude, target_frequency, taper_fraction, inner_scale
            )
            j00, j01, j10, j11 = rough_patch_jacobian(
                edge_id, u, vv, top, bottom, height,
                amplitude, target_frequency, taper_fraction, inner_scale
            )
            x[i, 0] = x0
            x[i, 1] = x1
            v_phys[i, 0] = j00 * udot + j01 * vdot
            v_phys[i, 1] = j10 * udot + j11 * vdot


@njit(cache=False, fastmath=True)
def update_modes(q, p, x, v_phys, mode, edge, rad, group, top, bottom, height, amplitude, target_frequency, taper_fraction, inner_scale, enter_factor, exit_factor):
    start = group[0]
    end = group[1]

    for i in range(start, end):
        edge_id, _, clearance = straight_trapezium_nearest_edge(
            x[i, 0], x[i, 1], top, bottom, height
        )

        enter_d = enter_factor * rad[i]
        exit_d = exit_factor * rad[i]

        if mode[i] == CART:
            if clearance <= enter_d:
                u, vv = rough_patch_inverse_edge(
                    x[i, 0], x[i, 1], edge_id, top, bottom, height,
                    amplitude, target_frequency, taper_fraction, inner_scale
                )
                udot, vdot = physical_velocity_to_patch_velocity(
                    v_phys[i, 0], v_phys[i, 1], edge_id, u, vv,
                    top, bottom, height, amplitude, target_frequency,
                    taper_fraction, inner_scale
                )
                q[i, 0] = u
                q[i, 1] = vv
                p[i, 0] = udot
                p[i, 1] = vdot
                edge[i] = edge_id
                mode[i] = CURV
        else:
            if clearance > exit_d:
                q[i, 0] = x[i, 0]
                q[i, 1] = x[i, 1]
                p[i, 0] = v_phys[i, 0]
                p[i, 1] = v_phys[i, 1]
                edge[i] = i64(-1)
                mode[i] = CART
            elif edge_id != edge[i] or q[i, 0] < f32(-0.02) or q[i, 0] > f32(1.02):
                u, vv = rough_patch_inverse_edge(
                    x[i, 0], x[i, 1], edge_id, top, bottom, height,
                    amplitude, target_frequency, taper_fraction, inner_scale
                )
                udot, vdot = physical_velocity_to_patch_velocity(
                    v_phys[i, 0], v_phys[i, 1], edge_id, u, vv,
                    top, bottom, height, amplitude, target_frequency,
                    taper_fraction, inner_scale
                )
                q[i, 0] = u
                q[i, 1] = vv
                p[i, 0] = udot
                p[i, 1] = vdot
                edge[i] = edge_id


@njit(cache=False, fastmath=True)
def accumulate_contact_forces(fext, x, v_phys, rad, group, offsets, neigh, k_contact, gamma_contact):
    start = group[0]
    end = group[1]

    for i in range(start, end):
        xi0 = x[i, 0]
        xi1 = x[i, 1]
        vi0 = v_phys[i, 0]
        vi1 = v_phys[i, 1]
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
                dvx = v_phys[j, 0] - vi0
                dvy = v_phys[j, 1] - vi1
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
def _add_rough_wall_contact_for_edge(fext, i, x0, x1, vx, vy, rad_i, edge_id, u, top, bottom, height, amplitude, target_frequency, taper_fraction, k_w, gamma_w):
    u = _clamp(u, f32(0.0), f32(1.0))
    bx, by = rough_patch_wall(
        edge_id, u, top, bottom, height, amplitude, target_frequency, taper_fraction
    )
    n0, n1 = rough_patch_wall_normal(
        edge_id, u, top, bottom, height, amplitude, target_frequency, taper_fraction
    )
    clearance = (x0 - bx) * n0 + (x1 - by) * n1
    delta = rad_i - clearance

    if delta > f32(0.0):
        vn = vx * n0 + vy * n1
        fm = k_w * delta - gamma_w * vn
        if fm < f32(0.0):
            fm = f32(0.0)
        fext[i, 0] += fm * n0
        fext[i, 1] += fm * n1


@njit(cache=False, fastmath=True)
def accumulate_wall_forces(fext, q, x, v_phys, mode, edge, rad, group, top, bottom, height, amplitude, target_frequency, taper_fraction, k_w, gamma_w, corner_factor):
    start = group[0]
    end = group[1]

    for i in range(start, end):
        if mode[i] != CURV:
            continue

        x0 = x[i, 0]
        x1 = x[i, 1]
        vx = v_phys[i, 0]
        vy = v_phys[i, 1]
        edge_id = edge[i]
        u = q[i, 0]

        _add_rough_wall_contact_for_edge(
            fext, i, x0, x1, vx, vy, rad[i], edge_id, u,
            top, bottom, height, amplitude, target_frequency, taper_fraction,
            k_w, gamma_w
        )

        corner_d = corner_factor * rad[i]
        for e2 in range(4):
            if e2 == edge_id:
                continue
            u2, d2 = _straight_edge_u_and_clearance(x0, x1, e2, top, bottom, height)
            if d2 <= corner_d:
                _add_rough_wall_contact_for_edge(
                    fext, i, x0, x1, vx, vy, rad[i], e2, u2,
                    top, bottom, height, amplitude, target_frequency, taper_fraction,
                    k_w, gamma_w
                )


@njit(cache=False, fastmath=True)
def finalize_accelerations(acc, q, p, fext, mode, edge, group, m, ax_body, ay_body, top, bottom, height, amplitude, target_frequency, taper_fraction, inner_scale):
    start = group[0]
    end = group[1]

    for i in range(start, end):
        inv_m = f32(1.0) / m[i]
        ax = fext[i, 0] * inv_m + ax_body
        ay = fext[i, 1] * inv_m + ay_body

        if mode[i] == CART:
            acc[i, 0] = ax
            acc[i, 1] = ay
        else:
            edge_id = edge[i]
            u = q[i, 0]
            vv = q[i, 1]
            udot = p[i, 0]
            vdot = p[i, 1]

            h0, h1 = rough_patch_second_terms(
                edge_id, u, vv, udot, vdot, top, bottom, height,
                amplitude, target_frequency, taper_fraction, inner_scale
            )
            j00, j01, j10, j11 = rough_patch_jacobian(
                edge_id, u, vv, top, bottom, height,
                amplitude, target_frequency, taper_fraction, inner_scale
            )
            det = j00 * j11 - j01 * j10
            inv00 = j11 / det
            inv01 = -j01 / det
            inv10 = -j10 / det
            inv11 = j00 / det

            rx = ax - h0
            ry = ay - h1
            acc[i, 0] = inv00 * rx + inv01 * ry
            acc[i, 1] = inv10 * rx + inv11 * ry


@njit(cache=False, fastmath=True)
def compute_accelerations(acc, q, p, x, v_phys, fext, mode, edge, m, rad, group_mobile, offsets, neigh, xA, omega, k_contact, gamma_contact, k_w, gamma_w, g, t, top, bottom, height, amplitude, target_frequency, taper_fraction, inner_scale, corner_factor):
    for i in range(fext.shape[0]):
        fext[i, 0] = f32(0.0)
        fext[i, 1] = f32(0.0)
        acc[i, 0] = f32(0.0)
        acc[i, 1] = f32(0.0)

    accumulate_contact_forces(
        fext, x, v_phys, rad, group_mobile, offsets, neigh,
        k_contact, gamma_contact
    )

    accumulate_wall_forces(
        fext, q, x, v_phys, mode, edge, rad, group_mobile,
        top, bottom, height, amplitude, target_frequency, taper_fraction,
        k_w, gamma_w, corner_factor
    )

    wall_acc0 = -xA[0] * omega * omega * math.sin(omega * t)
    wall_acc1 = -xA[1] * omega * omega * math.sin(omega * t)

    finalize_accelerations(
        acc, q, p, fext, mode, edge, group_mobile, m,
        g[0] - wall_acc0,
        g[1] - wall_acc1,
        top, bottom, height, amplitude, target_frequency, taper_fraction, inner_scale
    )


@njit(cache=False, fastmath=True)
def write_lab_frame_output(x_save, v_save, x, v_phys, xA, omega, t):
    shift0 = xA[0] * math.sin(omega * t)
    shift1 = xA[1] * math.sin(omega * t)
    shift_v0 = xA[0] * omega * math.cos(omega * t)
    shift_v1 = xA[1] * omega * math.cos(omega * t)

    for i in range(x.shape[0]):
        x_save[i, 0] = x[i, 0] + shift0
        x_save[i, 1] = x[i, 1] + shift1
        v_save[i, 0] = v_phys[i, 0] + shift_v0
        v_save[i, 1] = v_phys[i, 1] + shift_v1


@njit(cache=False, fastmath=True)
def snapshot_hybrid(q_save, p_save, mode_save, q, p, mode, edge):
    for i in range(q.shape[0]):
        mode_save[i] = mode[i]
        if mode[i] == CURV:
            q_save[i, 0] = f32(edge[i])
            q_save[i, 1] = q[i, 0]
            q_save[i, 2] = q[i, 1]
            p_save[i, 0] = p[i, 0]
            p_save[i, 1] = p[i, 1]
        else:
            q_save[i, 0] = f32(-1.0)
            q_save[i, 1] = q[i, 0]
            q_save[i, 2] = q[i, 1]
            p_save[i, 0] = p[i, 0]
            p_save[i, 1] = p[i, 1]


@njit(cache=False, fastmath=True)
def velocity_verlet_step(q, p, x, v_phys, acc, fext, mode, edge, dt, half_dt, m, rad, group_mobile, box, r_list, cell_size, skin, offsets, neigh, x_verlet_ref, xA, omega, k_contact, gamma_contact, k_w, gamma_w, g, t, top, bottom, height, amplitude, target_frequency, taper_fraction, inner_scale, enter_factor, exit_factor, corner_factor):
    N = q.shape[0]

    for i in range(N):
        p[i, 0] += half_dt * acc[i, 0]
        p[i, 1] += half_dt * acc[i, 1]

    for i in range(N):
        q[i, 0] += dt * p[i, 0]
        q[i, 1] += dt * p[i, 1]

    map_to_physical(
        q, p, x, v_phys, mode, edge,
        top, bottom, height, amplitude, target_frequency, taper_fraction, inner_scale
    )

    update_modes(
        q, p, x, v_phys, mode, edge, rad, group_mobile,
        top, bottom, height, amplitude, target_frequency, taper_fraction,
        inner_scale, enter_factor, exit_factor
    )

    map_to_physical(
        q, p, x, v_phys, mode, edge,
        top, bottom, height, amplitude, target_frequency, taper_fraction, inner_scale
    )

    if needs_verlet_rebuild(x, x_verlet_ref, skin):
        offsets, neigh = build_verlet_csr(x, box, r_list, cell_size)
        x_verlet_ref[:, :] = x

    compute_accelerations(
        acc, q, p, x, v_phys, fext, mode, edge, m, rad, group_mobile,
        offsets, neigh, xA, omega, k_contact, gamma_contact, k_w, gamma_w,
        g, t + dt, top, bottom, height, amplitude, target_frequency,
        taper_fraction, inner_scale, corner_factor
    )

    for i in range(N):
        p[i, 0] += half_dt * acc[i, 0]
        p[i, 1] += half_dt * acc[i, 1]

    map_to_physical(
        q, p, x, v_phys, mode, edge,
        top, bottom, height, amplitude, target_frequency, taper_fraction, inner_scale
    )

    return offsets, neigh


@njit(cache=False, fastmath=True)
def simulate_hybrid_particles_jit(box, x0, v0, m, rad, dt, T_max, group_mobile, k_contact, gamma_contact, xA, omega, k_w, gamma_w, g, top, bottom, height, amplitude, target_frequency, taper_fraction, inner_scale, r_list, skin, cell_size, save_every, enter_factor, exit_factor, corner_factor):
    N = x0.shape[0]
    num_steps = int(T_max / dt + f32(0.5))
    num_save = num_steps // save_every + 1

    t_out = np.empty(num_save, dtype=f32)
    q_out = np.empty((num_save, N, 3), dtype=f32)
    p_out = np.empty((num_save, N, 2), dtype=f32)
    x_out = np.empty((num_save, N, 2), dtype=f32)
    v_out = np.empty((num_save, N, 2), dtype=f32)
    mode_out = np.empty((num_save, N), dtype=i64)

    q = x0.copy()
    p = v0.copy()
    x = x0.copy()
    v_phys = v0.copy()
    mode = np.zeros(N, dtype=i64)
    edge = np.full(N, -1, dtype=i64)
    fext = np.empty((N, 2), dtype=f32)
    acc = np.empty((N, 2), dtype=f32)

    update_modes(
        q, p, x, v_phys, mode, edge, rad, group_mobile,
        top, bottom, height, amplitude, target_frequency, taper_fraction,
        inner_scale, enter_factor, exit_factor
    )
    map_to_physical(
        q, p, x, v_phys, mode, edge,
        top, bottom, height, amplitude, target_frequency, taper_fraction, inner_scale
    )

    x_verlet_ref = x.copy()
    offsets, neigh = build_verlet_csr(x, box, r_list, cell_size)

    t = f32(0.0)
    compute_accelerations(
        acc, q, p, x, v_phys, fext, mode, edge, m, rad, group_mobile,
        offsets, neigh, xA, omega, k_contact, gamma_contact, k_w, gamma_w,
        g, t, top, bottom, height, amplitude, target_frequency, taper_fraction,
        inner_scale, corner_factor
    )

    save_id = 0
    t_out[save_id] = f32(0.0)
    snapshot_hybrid(q_out[save_id], p_out[save_id], mode_out[save_id], q, p, mode, edge)
    write_lab_frame_output(x_out[save_id], v_out[save_id], x, v_phys, xA, omega, f32(0.0))
    save_id += 1

    half_dt = f32(0.5) * dt
    for n in range(num_steps):
        t = f32(n) * dt
        offsets, neigh = velocity_verlet_step(
            q, p, x, v_phys, acc, fext, mode, edge, dt, half_dt,
            m, rad, group_mobile, box, r_list, cell_size, skin,
            offsets, neigh, x_verlet_ref, xA, omega,
            k_contact, gamma_contact, k_w, gamma_w, g, t,
            top, bottom, height, amplitude, target_frequency, taper_fraction,
            inner_scale, enter_factor, exit_factor, corner_factor
        )

        if (n + 1) % save_every == 0:
            ts = f32(n + 1) * dt
            t_out[save_id] = ts
            snapshot_hybrid(q_out[save_id], p_out[save_id], mode_out[save_id], q, p, mode, edge)
            write_lab_frame_output(x_out[save_id], v_out[save_id], x, v_phys, xA, omega, ts)
            save_id += 1

    return t_out, q_out, p_out, x_out, v_out, mode_out


def simulate_hybrid_particles(
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
    xA,
    omega,
    k_w,
    gamma_w,
    g,
    TOP,
    BOTTOM,
    HEIGHT,
    ROUGHNESS_AMPLITUDE,
    TARGET_FREQUENCY,
    TAPER_FRACTION,
    INNER_SCALE,
    save_every=1,
    r_list=None,
    skin=None,
    cell_size=None,
    enter_factor=2.0,
    exit_factor=2.5,
    corner_factor=2.0,
):
    """
    Run the hybrid solver.

    Returns
    -------
    t, q, p, x, v, mode

    q has shape (frames, N, 3). For curvilinear particles q[..., 0] is the
    wall patch id and q[..., 1:3] are (u, v). For Cartesian particles
    q[..., 0] = -1 and q[..., 1:3] are (x, y).
    """
    box = as_f32(box)
    x0 = as_f32(x0)
    v0 = as_f32(v0)
    m = as_f32(m)
    rad = as_f32(rad)
    group_mobile = as_i64(group_mobile)
    xA = as_f32(xA)
    g = as_f32(g)

    max_rad = f32(np.max(rad))
    skin_val = f32(skin if skin is not None else f32(0.25) * max_rad)
    r_contact = f32(2.0) * max_rad
    r_list_val = f32(r_list if r_list is not None else r_contact + skin_val)
    cell_size_val = f32(cell_size if cell_size is not None else r_list_val)

    return simulate_hybrid_particles_jit(
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
        xA,
        f32(omega),
        f32(k_w),
        f32(gamma_w),
        g,
        f32(TOP),
        f32(BOTTOM),
        f32(HEIGHT),
        f32(ROUGHNESS_AMPLITUDE),
        f32(TARGET_FREQUENCY),
        f32(TAPER_FRACTION),
        f32(INNER_SCALE),
        r_list_val,
        skin_val,
        cell_size_val,
        int(save_every),
        f32(enter_factor),
        f32(exit_factor),
        f32(corner_factor),
    )


def rough_patch_to_physical(q, p, TOP, BOTTOM, HEIGHT, ROUGHNESS_AMPLITUDE, TARGET_FREQUENCY, TAPER_FRACTION, INNER_SCALE):
    q = as_f32(q)
    p = as_f32(p)
    N = q.shape[0]
    x = np.empty((N, 2), dtype=f32)
    v = np.empty((N, 2), dtype=f32)
    mode = np.ones(N, dtype=i64)
    edge = q[:, 0].astype(np.int64)
    q2 = q[:, 1:3].copy()
    map_to_physical(
        q2,
        p,
        x,
        v,
        mode,
        edge,
        f32(TOP),
        f32(BOTTOM),
        f32(HEIGHT),
        f32(ROUGHNESS_AMPLITUDE),
        f32(TARGET_FREQUENCY),
        f32(TAPER_FRACTION),
        f32(INNER_SCALE),
    )
    return x, v
