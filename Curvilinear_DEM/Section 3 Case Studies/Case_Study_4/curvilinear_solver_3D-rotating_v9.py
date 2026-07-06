"""
Fast 3D curvilinear multiparticle solver for Case Study 4.

Numba rotating-y version of the bunny-ear star-shaped container

    x = r * B(theta, phi),       0 <= r <= 1.

The container is fixed in body coordinates.  The displayed laboratory boundary
rotates about the y axis through the centre.  Dynamics are integrated in the
body frame, so the laboratory gravity vector appears as a rotating body-frame
acceleration.

State variables
---------------
q[i] = (r, theta, phi)
p[i] = (rdot, thetadot, phidot)
acc[i] = (rddot, thetaddot, phiddot)

This file follows the same philosophy as the 2D solvers:
- Numba kernels with cache=False and fastmath=True;
- float32 physical arrays and int64 indices;
- Euclidean 3D Verlet neighbour list for particle contacts;
- radius-aware wall force using the local boundary tangent plane.
"""

import math
import numpy as np
from numba import njit


f32 = np.float32
i64 = np.int64
TWO_PI = f32(2.0 * np.pi)
PI = f32(np.pi)

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
def fold_phi(theta, phi, phidot):
    phi_mod = phi % TWO_PI
    if phi_mod > PI:
        return wrap_theta(theta + PI), TWO_PI - phi_mod, -phidot
    return wrap_theta(theta), phi_mod, phidot


@njit(cache=False, fastmath=True)
def eval_boundary(theta, phi, R0, eps, phi0, sigma):
    """Evaluate B, first derivatives, and second derivatives."""
    st = math.sin(theta)
    ct = math.cos(theta)
    sp = math.sin(phi)
    cp = math.cos(phi)

    u = phi - phi0
    sig2 = sigma * sigma
    A = math.exp(-(u * u) / sig2)
    c2 = math.cos(f32(2.0) * theta)
    s2 = math.sin(f32(2.0) * theta)

    R = R0 * (f32(1.0) + eps * A * c2)
    Rt = R0 * eps * A * (-f32(2.0) * s2)
    Rtt = R0 * eps * A * (-f32(4.0) * c2)

    Ap = A * (-f32(2.0) * u / sig2)
    App = A * ((f32(4.0) * u * u / (sig2 * sig2)) - (f32(2.0) / sig2))
    Rp = R0 * eps * Ap * c2
    Rpp = R0 * eps * App * c2
    Rtp = R0 * eps * Ap * (-f32(2.0) * s2)

    ex = sp * ct
    ey = sp * st
    ez = cp

    etx = -sp * st
    ety = sp * ct
    etz = f32(0.0)

    epx = cp * ct
    epy = cp * st
    epz = -sp

    ettx = -sp * ct
    etty = -sp * st
    ettz = f32(0.0)

    eppx = -ex
    eppy = -ey
    eppz = -ez

    etpx = -cp * st
    etpy = cp * ct
    etpz = f32(0.0)

    Bx = R * ex
    By = R * ey
    Bz = R * ez

    Btx = Rt * ex + R * etx
    Bty = Rt * ey + R * ety
    Btz = Rt * ez + R * etz

    Bpx = Rp * ex + R * epx
    Bpy = Rp * ey + R * epy
    Bpz = Rp * ez + R * epz

    Bttx = Rtt * ex + f32(2.0) * Rt * etx + R * ettx
    Btty = Rtt * ey + f32(2.0) * Rt * ety + R * etty
    Bttz = Rtt * ez + f32(2.0) * Rt * etz + R * ettz

    Btpx = Rtp * ex + Rt * epx + Rp * etx + R * etpx
    Btpy = Rtp * ey + Rt * epy + Rp * ety + R * etpy
    Btpz = Rtp * ez + Rt * epz + Rp * etz + R * etpz

    Bppx = Rpp * ex + f32(2.0) * Rp * epx + R * eppx
    Bppy = Rpp * ey + f32(2.0) * Rp * epy + R * eppy
    Bppz = Rpp * ez + f32(2.0) * Rp * epz + R * eppz

    return (
        Bx, By, Bz,
        Btx, Bty, Btz,
        Bpx, Bpy, Bpz,
        Bttx, Btty, Bttz,
        Btpx, Btpy, Btpz,
        Bppx, Bppy, Bppz,
    )


@njit(cache=False, fastmath=True)
def eval_boundary_first(theta, phi, R0, eps, phi0, sigma):
    """Evaluate B and first derivatives only."""
    st = math.sin(theta)
    ct = math.cos(theta)
    sp = math.sin(phi)
    cp = math.cos(phi)

    u = phi - phi0
    sig2 = sigma * sigma
    A = math.exp(-(u * u) / sig2)
    c2 = math.cos(f32(2.0) * theta)
    s2 = math.sin(f32(2.0) * theta)

    R = R0 * (f32(1.0) + eps * A * c2)
    Rt = R0 * eps * A * (-f32(2.0) * s2)

    Ap = A * (-f32(2.0) * u / sig2)
    Rp = R0 * eps * Ap * c2

    ex = sp * ct
    ey = sp * st
    ez = cp

    etx = -sp * st
    ety = sp * ct
    etz = f32(0.0)

    epx = cp * ct
    epy = cp * st
    epz = -sp

    Bx = R * ex
    By = R * ey
    Bz = R * ez

    Btx = Rt * ex + R * etx
    Bty = Rt * ey + R * ety
    Btz = Rt * ez + R * etz

    Bpx = Rp * ex + R * epx
    Bpy = Rp * ey + R * epy
    Bpz = Rp * ez + R * epz

    return (
        Bx, By, Bz,
        Btx, Bty, Btz,
        Bpx, Bpy, Bpz,
    )


@njit(cache=False, fastmath=True)
def inverse_3x3(
    a00, a01, a02,
    a10, a11, a12,
    a20, a21, a22,
):
    c00 = a11 * a22 - a12 * a21
    c01 = -(a10 * a22 - a12 * a20)
    c02 = a10 * a21 - a11 * a20

    c10 = -(a01 * a22 - a02 * a21)
    c11 = a00 * a22 - a02 * a20
    c12 = -(a00 * a21 - a01 * a20)

    c20 = a01 * a12 - a02 * a11
    c21 = -(a00 * a12 - a02 * a10)
    c22 = a00 * a11 - a01 * a10

    det = a00 * c00 + a01 * c01 + a02 * c02
    if math.fabs(det) <= f32(1.0e-12):
        print("bad inverse_3x3 determinant", det)
        raise ValueError("bad inverse_3x3 determinant")
    inv_det = f32(1.0) / det

    return (
        c00 * inv_det, c10 * inv_det, c20 * inv_det,
        c01 * inv_det, c11 * inv_det, c21 * inv_det,
        c02 * inv_det, c12 * inv_det, c22 * inv_det,
    )


@njit(cache=False, fastmath=True)
def apply_safeguard(q, p, mode, phi_eps):
    for i in range(q.shape[0]):
        if mode[i] == CURV:
            theta, phi, phidot = fold_phi(q[i, 1], q[i, 2], p[i, 2])
            q[i, 1] = theta
            q[i, 2] = phi
            p[i, 2] = phidot


@njit(cache=False, fastmath=True)
def compute_geometry(q, p, geom, R0, eps, phi0, sigma, phi_eps):
    """
    geom columns:
    0:3 x, 3:6 v, 6:15 J^{-1}, 15:18 H, 18:21 outward normal,
    21:24 boundary point B(theta, phi).
    """
    for i in range(q.shape[0]):
        r = q[i, 0]
        if r <= f32(0.0):
            print("bad rho in compute_geometry", i, r)
            raise ValueError("bad rho in compute_geometry")

        theta = wrap_theta(q[i, 1])
        phi = q[i, 2]
        if phi < f32(0.0) or phi > PI:
            print("bad phi in compute_geometry", i, phi)
            raise ValueError("bad phi in compute_geometry")

        rdot = p[i, 0]
        thetadot = p[i, 1]
        phidot = p[i, 2]

        (
            Bx, By, Bz,
            Btx, Bty, Btz,
            Bpx, Bpy, Bpz,
            Bttx, Btty, Bttz,
            Btpx, Btpy, Btpz,
            Bppx, Bppy, Bppz,
        ) = eval_boundary(theta, phi, R0, eps, phi0, sigma)

        j00 = Bx
        j10 = By
        j20 = Bz
        j01 = r * Btx
        j11 = r * Bty
        j21 = r * Btz
        j02 = r * Bpx
        j12 = r * Bpy
        j22 = r * Bpz

        (
            inv00, inv01, inv02,
            inv10, inv11, inv12,
            inv20, inv21, inv22,
        ) = inverse_3x3(
            j00, j01, j02,
            j10, j11, j12,
            j20, j21, j22,
        )

        x0 = r * Bx
        x1 = r * By
        x2 = r * Bz

        v0 = Bx * rdot + r * Btx * thetadot + r * Bpx * phidot
        v1 = By * rdot + r * Bty * thetadot + r * Bpy * phidot
        v2 = Bz * rdot + r * Btz * thetadot + r * Bpz * phidot

        H0 = (
            f32(2.0) * rdot * thetadot * Btx
            + f32(2.0) * rdot * phidot * Bpx
            + r * thetadot * thetadot * Bttx
            + f32(2.0) * r * thetadot * phidot * Btpx
            + r * phidot * phidot * Bppx
        )
        H1 = (
            f32(2.0) * rdot * thetadot * Bty
            + f32(2.0) * rdot * phidot * Bpy
            + r * thetadot * thetadot * Btty
            + f32(2.0) * r * thetadot * phidot * Btpy
            + r * phidot * phidot * Bppy
        )
        H2 = (
            f32(2.0) * rdot * thetadot * Btz
            + f32(2.0) * rdot * phidot * Bpz
            + r * thetadot * thetadot * Bttz
            + f32(2.0) * r * thetadot * phidot * Btpz
            + r * phidot * phidot * Bppz
        )

        # outward normal = normalized cross(B_phi, B_theta)
        nx = Bpy * Btz - Bpz * Bty
        ny = Bpz * Btx - Bpx * Btz
        nz = Bpx * Bty - Bpy * Btx
        nn = math.sqrt(nx * nx + ny * ny + nz * nz)
        inv_nn = f32(1.0) / nn
        nx *= inv_nn
        ny *= inv_nn
        nz *= inv_nn

        geom[i, 0] = x0
        geom[i, 1] = x1
        geom[i, 2] = x2
        geom[i, 3] = v0
        geom[i, 4] = v1
        geom[i, 5] = v2

        geom[i, 6] = inv00
        geom[i, 7] = inv01
        geom[i, 8] = inv02
        geom[i, 9] = inv10
        geom[i, 10] = inv11
        geom[i, 11] = inv12
        geom[i, 12] = inv20
        geom[i, 13] = inv21
        geom[i, 14] = inv22

        geom[i, 15] = H0
        geom[i, 16] = H1
        geom[i, 17] = H2

        geom[i, 18] = nx
        geom[i, 19] = ny
        geom[i, 20] = nz

        geom[i, 21] = Bx
        geom[i, 22] = By
        geom[i, 23] = Bz


@njit(cache=False, fastmath=True)
def compute_geometry_modes(q, p, geom, mode, R0, eps, phi0, sigma, phi_eps):
    for i in range(q.shape[0]):
        if mode[i] == CURV:
            # One-particle inline version of compute_geometry.
            r = q[i, 0]
            if r <= f32(0.0):
                print("bad rho in compute_geometry_modes", i, r)
                raise ValueError("bad rho in compute_geometry_modes")

            theta = wrap_theta(q[i, 1])
            phi = q[i, 2]
            if phi < f32(0.0) or phi > PI:
                print("bad phi in compute_geometry_modes", i, phi)
                raise ValueError("bad phi in compute_geometry_modes")

            rdot = p[i, 0]
            thetadot = p[i, 1]
            phidot = p[i, 2]

            (
                Bx, By, Bz,
                Btx, Bty, Btz,
                Bpx, Bpy, Bpz,
                Bttx, Btty, Bttz,
                Btpx, Btpy, Btpz,
                Bppx, Bppy, Bppz,
            ) = eval_boundary(theta, phi, R0, eps, phi0, sigma)

            j00 = Bx
            j10 = By
            j20 = Bz
            j01 = r * Btx
            j11 = r * Bty
            j21 = r * Btz
            j02 = r * Bpx
            j12 = r * Bpy
            j22 = r * Bpz

            (
                inv00, inv01, inv02,
                inv10, inv11, inv12,
                inv20, inv21, inv22,
            ) = inverse_3x3(
                j00, j01, j02,
                j10, j11, j12,
                j20, j21, j22,
            )

            geom[i, 0] = r * Bx
            geom[i, 1] = r * By
            geom[i, 2] = r * Bz
            geom[i, 3] = Bx * rdot + r * Btx * thetadot + r * Bpx * phidot
            geom[i, 4] = By * rdot + r * Bty * thetadot + r * Bpy * phidot
            geom[i, 5] = Bz * rdot + r * Btz * thetadot + r * Bpz * phidot

            geom[i, 6] = inv00
            geom[i, 7] = inv01
            geom[i, 8] = inv02
            geom[i, 9] = inv10
            geom[i, 10] = inv11
            geom[i, 11] = inv12
            geom[i, 12] = inv20
            geom[i, 13] = inv21
            geom[i, 14] = inv22

            geom[i, 15] = (
                f32(2.0) * rdot * thetadot * Btx
                + f32(2.0) * rdot * phidot * Bpx
                + r * thetadot * thetadot * Bttx
                + f32(2.0) * r * thetadot * phidot * Btpx
                + r * phidot * phidot * Bppx
            )
            geom[i, 16] = (
                f32(2.0) * rdot * thetadot * Bty
                + f32(2.0) * rdot * phidot * Bpy
                + r * thetadot * thetadot * Btty
                + f32(2.0) * r * thetadot * phidot * Btpy
                + r * phidot * phidot * Bppy
            )
            geom[i, 17] = (
                f32(2.0) * rdot * thetadot * Btz
                + f32(2.0) * rdot * phidot * Bpz
                + r * thetadot * thetadot * Bttz
                + f32(2.0) * r * thetadot * phidot * Btpz
                + r * phidot * phidot * Bppz
            )

            nx = Bpy * Btz - Bpz * Bty
            ny = Bpz * Btx - Bpx * Btz
            nz = Bpx * Bty - Bpy * Btx
            nn = math.sqrt(nx * nx + ny * ny + nz * nz)
            inv_nn = f32(1.0) / nn
            geom[i, 18] = nx * inv_nn
            geom[i, 19] = ny * inv_nn
            geom[i, 20] = nz * inv_nn
            geom[i, 21] = Bx
            geom[i, 22] = By
            geom[i, 23] = Bz


@njit(cache=False, fastmath=True)
def map_to_physical(q, p, x, v, geom, mode):
    for i in range(x.shape[0]):
        if mode[i] == CURV:
            x[i, 0] = geom[i, 0]
            x[i, 1] = geom[i, 1]
            x[i, 2] = geom[i, 2]
            v[i, 0] = geom[i, 3]
            v[i, 1] = geom[i, 4]
            v[i, 2] = geom[i, 5]
        else:
            x[i, 0] = q[i, 0]
            x[i, 1] = q[i, 1]
            x[i, 2] = q[i, 2]
            v[i, 0] = p[i, 0]
            v[i, 1] = p[i, 1]
            v[i, 2] = p[i, 2]


@njit(cache=False, fastmath=True)
def physical_to_curvilinear_one(x0, x1, x2, v0, v1, v2, R0, eps, phi0, sigma, phi_eps):
    rho = math.sqrt(x0 * x0 + x1 * x1 + x2 * x2)
    if rho < f32(1.0e-12):
        theta = f32(0.0)
        phi = f32(0.5) * PI
        r = f32(0.0)
    else:
        theta = wrap_theta(math.atan2(x1, x0))
        cphi = x2 / rho
        if cphi > f32(1.0) or cphi < -f32(1.0):
            print("bad z/rho in physical_to_curvilinear_one", cphi)
            raise ValueError("bad z/rho in physical_to_curvilinear_one")
        phi = math.acos(cphi)

        (
            Bx, By, Bz,
            Btx, Bty, Btz,
            Bpx, Bpy, Bpz,
        ) = eval_boundary_first(theta, phi, R0, eps, phi0, sigma)

        Bnorm = math.sqrt(Bx * Bx + By * By + Bz * Bz)
        if Bnorm <= f32(0.0):
            print("bad boundary norm in physical_to_curvilinear_one", Bnorm)
            raise ValueError("bad boundary norm in physical_to_curvilinear_one")
        r = rho / Bnorm

        if r < f32(1.0e-5):
            r_eff = f32(1.0e-5)
        else:
            r_eff = r

        j00 = Bx
        j10 = By
        j20 = Bz
        j01 = r_eff * Btx
        j11 = r_eff * Bty
        j21 = r_eff * Btz
        j02 = r_eff * Bpx
        j12 = r_eff * Bpy
        j22 = r_eff * Bpz

        (
            inv00, inv01, inv02,
            inv10, inv11, inv12,
            inv20, inv21, inv22,
        ) = inverse_3x3(
            j00, j01, j02,
            j10, j11, j12,
            j20, j21, j22,
        )

        rdot = inv00 * v0 + inv01 * v1 + inv02 * v2
        thetadot = inv10 * v0 + inv11 * v1 + inv12 * v2
        phidot = inv20 * v0 + inv21 * v1 + inv22 * v2

        return r, theta, phi, rdot, thetadot, phidot

    return r, theta, phi, f32(0.0), f32(0.0), f32(0.0)


@njit(cache=False, fastmath=True)
def physical_position_to_curvilinear_output(
    x0, x1, x2,
    theta_prev, phi_prev,
    R0, eps, phi0, sigma,
):
    rho_phys = math.sqrt(x0 * x0 + x1 * x1 + x2 * x2)
    if rho_phys <= f32(0.0):
        return f32(0.0), theta_prev, phi_prev

    r_cyl = math.sqrt(x0 * x0 + x1 * x1)
    if r_cyl <= f32(0.0):
        theta = theta_prev
    else:
        theta = wrap_theta(math.atan2(x1, x0))

    cphi = x2 / rho_phys
    if cphi > f32(1.0) or cphi < -f32(1.0):
        print("bad z/rho in output position inverse", cphi)
        raise ValueError("bad z/rho in output position inverse")
    phi = math.acos(cphi)

    Bx, By, Bz, _, _, _, _, _, _ = eval_boundary_first(
        theta, phi, R0, eps, phi0, sigma
    )
    Bnorm = math.sqrt(Bx * Bx + By * By + Bz * Bz)
    if Bnorm <= f32(0.0):
        print("bad boundary norm in output position inverse", Bnorm)
        raise ValueError("bad boundary norm in output position inverse")

    return rho_phys / Bnorm, theta, phi


@njit(cache=False, fastmath=True)
def handoff(q, p, mode, group, R0, eps, phi0, sigma, phi_eps, r_cap, r_exit, axis_cap, axis_exit):
    start = group[0]
    end = group[1]

    for i in range(start, end):
        if mode[i] == CURV:
            r = q[i, 0]
            theta = wrap_theta(q[i, 1])
            phi = q[i, 2]
            if phi < f32(0.0) or phi > PI:
                print("bad phi before handoff", i, phi)
                raise ValueError("bad phi before handoff")

            (
                Bx, By, Bz,
                Btx, Bty, Btz,
                Bpx, Bpy, Bpz,
            ) = eval_boundary_first(theta, phi, R0, eps, phi0, sigma)

            x0 = r * Bx
            x1 = r * By
            x2 = r * Bz
            r_cyl = math.sqrt(x0 * x0 + x1 * x1)

            if r <= r_cap or r_cyl <= axis_cap:
                rdot = p[i, 0]
                thetadot = p[i, 1]
                phidot = p[i, 2]

                q[i, 0] = x0
                q[i, 1] = x1
                q[i, 2] = x2

                p[i, 0] = Bx * rdot + r * Btx * thetadot + r * Bpx * phidot
                p[i, 1] = By * rdot + r * Bty * thetadot + r * Bpy * phidot
                p[i, 2] = Bz * rdot + r * Btz * thetadot + r * Bpz * phidot
                mode[i] = CAP
        else:
            x0 = q[i, 0]
            x1 = q[i, 1]
            x2 = q[i, 2]
            rho = math.sqrt(x0 * x0 + x1 * x1 + x2 * x2)
            r_cyl = math.sqrt(x0 * x0 + x1 * x1)
            if rho <= f32(0.0) or r_cyl < axis_exit:
                continue

            theta = wrap_theta(math.atan2(x1, x0))
            cphi = x2 / rho
            if cphi > f32(1.0) or cphi < -f32(1.0):
                print("bad z/rho before leaving cap", i, cphi)
                raise ValueError("bad z/rho before leaving cap")
            phi = math.acos(cphi)

            (
                Bx, By, Bz,
                Btx, Bty, Btz,
                Bpx, Bpy, Bpz,
            ) = eval_boundary_first(theta, phi, R0, eps, phi0, sigma)
            Bnorm = math.sqrt(Bx * Bx + By * By + Bz * Bz)
            if Bnorm <= f32(0.0):
                print("bad boundary norm before leaving cap", i, Bnorm)
                raise ValueError("bad boundary norm before leaving cap")
            r_rec = rho / Bnorm
            if r_rec < r_exit:
                continue

            if r_rec < f32(1.0e-5):
                r_eff = f32(1.0e-5)
            else:
                r_eff = r_rec

            (
                inv00, inv01, inv02,
                inv10, inv11, inv12,
                inv20, inv21, inv22,
            ) = inverse_3x3(
                Bx, r_eff * Btx, r_eff * Bpx,
                By, r_eff * Bty, r_eff * Bpy,
                Bz, r_eff * Btz, r_eff * Bpz,
            )

            v0 = p[i, 0]
            v1 = p[i, 1]
            v2 = p[i, 2]
            rdot_rec = inv00 * v0 + inv01 * v1 + inv02 * v2
            thetadot_rec = inv10 * v0 + inv11 * v1 + inv12 * v2
            phidot_rec = inv20 * v0 + inv21 * v1 + inv22 * v2

            q[i, 0] = r_rec
            q[i, 1] = theta
            q[i, 2] = phi
            p[i, 0] = rdot_rec
            p[i, 1] = thetadot_rec
            p[i, 2] = phidot_rec
            mode[i] = CURV


@njit(cache=False, fastmath=True)
def build_cell_list_3d(x, box, cell_size):
    x_min, x_max, y_min, y_max, z_min, z_max = box
    N = x.shape[0]

    nx = max(1, int(math.ceil((x_max - x_min) / cell_size)))
    ny = max(1, int(math.ceil((y_max - y_min) / cell_size)))
    nz = max(1, int(math.ceil((z_max - z_min) / cell_size)))

    head = np.full(nx * ny * nz, -1, dtype=i64)
    nxt = np.full(N, -1, dtype=i64)
    cell_id = np.empty(N, dtype=i64)
    inv_cell = f32(1.0) / cell_size

    for i in range(N):
        cx = int((x[i, 0] - x_min) * inv_cell)
        cy = int((x[i, 1] - y_min) * inv_cell)
        cz = int((x[i, 2] - z_min) * inv_cell)

        if cx < 0:
            cx = 0
        elif cx >= nx:
            cx = nx - 1
        if cy < 0:
            cy = 0
        elif cy >= ny:
            cy = ny - 1
        if cz < 0:
            cz = 0
        elif cz >= nz:
            cz = nz - 1

        c = cx + nx * (cy + ny * cz)
        cell_id[i] = c
        nxt[i] = head[c]
        head[c] = i

    return head, nxt, cell_id, nx, ny, nz


@njit(cache=False, fastmath=True)
def build_verlet_csr_3d(x, box, r_list, cell_size):
    N = x.shape[0]
    r2_list = r_list * r_list
    head, nxt, cell_id, nx, ny, nz = build_cell_list_3d(x, box, cell_size)
    cr = max(1, int(math.ceil(r_list / cell_size)))

    deg = np.zeros(N, dtype=i64)

    for i in range(N):
        xi0 = x[i, 0]
        xi1 = x[i, 1]
        xi2 = x[i, 2]
        c0 = cell_id[i]
        cx = c0 % nx
        tmp = c0 // nx
        cy = tmp % ny
        cz = tmp // ny

        for dz in range(-cr, cr + 1):
            cz2 = cz + dz
            if cz2 < 0 or cz2 >= nz:
                continue
            for dy in range(-cr, cr + 1):
                cy2 = cy + dy
                if cy2 < 0 or cy2 >= ny:
                    continue
                for dx in range(-cr, cr + 1):
                    cx2 = cx + dx
                    if cx2 < 0 or cx2 >= nx:
                        continue

                    j = head[cx2 + nx * (cy2 + ny * cz2)]
                    while j != -1:
                        if j > i:
                            dxij = xi0 - x[j, 0]
                            dyij = xi1 - x[j, 1]
                            dzij = xi2 - x[j, 2]
                            if dxij * dxij + dyij * dyij + dzij * dzij <= r2_list:
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
        xi2 = x[i, 2]
        c0 = cell_id[i]
        cx = c0 % nx
        tmp = c0 // nx
        cy = tmp % ny
        cz = tmp // ny

        for dz in range(-cr, cr + 1):
            cz2 = cz + dz
            if cz2 < 0 or cz2 >= nz:
                continue
            for dy in range(-cr, cr + 1):
                cy2 = cy + dy
                if cy2 < 0 or cy2 >= ny:
                    continue
                for dx in range(-cr, cr + 1):
                    cx2 = cx + dx
                    if cx2 < 0 or cx2 >= nx:
                        continue

                    j = head[cx2 + nx * (cy2 + ny * cz2)]
                    while j != -1:
                        if j > i:
                            dxij = xi0 - x[j, 0]
                            dyij = xi1 - x[j, 1]
                            dzij = xi2 - x[j, 2]
                            if dxij * dxij + dyij * dyij + dzij * dzij <= r2_list:
                                neigh[cursor[i]] = j
                                cursor[i] += 1
                                neigh[cursor[j]] = i
                                cursor[j] += 1
                        j = nxt[j]

    return offsets, neigh


@njit(cache=False, fastmath=True)
def needs_verlet_rebuild_3d(x_now, x_ref, skin):
    skin_half2 = (f32(0.5) * skin) ** f32(2.0)
    max_disp2 = f32(0.0)

    for i in range(x_now.shape[0]):
        dx = x_now[i, 0] - x_ref[i, 0]
        dy = x_now[i, 1] - x_ref[i, 1]
        dz = x_now[i, 2] - x_ref[i, 2]
        d2 = dx * dx + dy * dy + dz * dz
        if d2 > max_disp2:
            max_disp2 = d2

    return max_disp2 > skin_half2


@njit(cache=False, fastmath=True)
def accumulate_contact_forces(
    fext, x, v, rad, group, offsets, neigh, k_contact, gamma_contact
):
    start = group[0]
    end = group[1]

    for i in range(start, end):
        xi0 = x[i, 0]
        xi1 = x[i, 1]
        xi2 = x[i, 2]
        vi0 = v[i, 0]
        vi1 = v[i, 1]
        vi2 = v[i, 2]
        ri = rad[i]

        for pp in range(offsets[i], offsets[i + 1]):
            j = neigh[pp]
            if j <= i:
                continue
            if j < start or j >= end:
                continue

            dx = x[j, 0] - xi0
            dy = x[j, 1] - xi1
            dz = x[j, 2] - xi2
            r2 = dx * dx + dy * dy + dz * dz
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
            nz = dz * inv_rr

            dvx = v[j, 0] - vi0
            dvy = v[j, 1] - vi1
            dvz = v[j, 2] - vi2
            vn = dvx * nx + dvy * ny + dvz * nz

            fm = k_contact * overlap - gamma_contact * vn
            if fm < f32(0.0):
                fm = f32(0.0)

            fx = fm * nx
            fy = fm * ny
            fz = fm * nz

            fext[i, 0] -= fx
            fext[i, 1] -= fy
            fext[i, 2] -= fz
            fext[j, 0] += fx
            fext[j, 1] += fy
            fext[j, 2] += fz


@njit(cache=False, fastmath=True)
def accumulate_wall_forces(fext, x, v, geom, mode, rad, group, k_w, gamma_w, R0, eps, phi0, sigma, r_cap):
    start = group[0]
    end = group[1]

    for i in range(start, end):
        if mode[i] == CURV:
            nx_out = geom[i, 18]
            ny_out = geom[i, 19]
            nz_out = geom[i, 20]

            bx = geom[i, 21]
            by = geom[i, 22]
            bz = geom[i, 23]
        else:
            x0 = x[i, 0]
            x1 = x[i, 1]
            x2 = x[i, 2]
            rho = math.sqrt(x0 * x0 + x1 * x1 + x2 * x2)
            if rho <= r_cap:
                continue

            r_cyl = math.sqrt(x0 * x0 + x1 * x1)
            if r_cyl <= f32(0.0):
                print("cannot compute wall normal on polar axis", i, x0, x1, x2)
                raise ValueError("cannot compute wall normal on polar axis")

            theta = wrap_theta(math.atan2(x1, x0))
            cphi = x2 / rho
            if cphi > f32(1.0) or cphi < -f32(1.0):
                print("bad z/rho in cap wall force", i, cphi)
                raise ValueError("bad z/rho in cap wall force")
            phi = math.acos(cphi)

            (
                bx, by, bz,
                Btx, Bty, Btz,
                Bpx, Bpy, Bpz,
            ) = eval_boundary_first(theta, phi, R0, eps, phi0, sigma)

            nx_out = Bpy * Btz - Bpz * Bty
            ny_out = Bpz * Btx - Bpx * Btz
            nz_out = Bpx * Bty - Bpy * Btx
            nn = math.sqrt(nx_out * nx_out + ny_out * ny_out + nz_out * nz_out)
            if nn <= f32(0.0):
                print("zero wall normal in cap wall force", i)
                raise ValueError("zero wall normal in cap wall force")
            nx_out /= nn
            ny_out /= nn
            nz_out /= nn

        signed_outside = (
            (x[i, 0] - bx) * nx_out
            + (x[i, 1] - by) * ny_out
            + (x[i, 2] - bz) * nz_out
        )
        delta = rad[i] + signed_outside

        if delta > f32(0.0):
            # inward normal
            nx = -nx_out
            ny = -ny_out
            nz = -nz_out

            vn = v[i, 0] * nx + v[i, 1] * ny + v[i, 2] * nz
            f_mag = k_w * delta - gamma_w * vn
            if f_mag < f32(0.0):
                f_mag = f32(0.0)

            fext[i, 0] += f_mag * nx
            fext[i, 1] += f_mag * ny
            fext[i, 2] += f_mag * nz


@njit(cache=False, fastmath=True)
def finalize_accelerations(acc, q, p, geom, fext, mode, group, m, ax_body, ay_body, az_body):
    start = group[0]
    end = group[1]

    for i in range(start, end):
        inv_m = f32(1.0) / m[i]
        ax = fext[i, 0] * inv_m + ax_body
        ay = fext[i, 1] * inv_m + ay_body
        az = fext[i, 2] * inv_m + az_body

        if mode[i] == CURV:
            rx = ax - geom[i, 15]
            ry = ay - geom[i, 16]
            rz = az - geom[i, 17]

            acc[i, 0] = geom[i, 6] * rx + geom[i, 7] * ry + geom[i, 8] * rz
            acc[i, 1] = geom[i, 9] * rx + geom[i, 10] * ry + geom[i, 11] * rz
            acc[i, 2] = geom[i, 12] * rx + geom[i, 13] * ry + geom[i, 14] * rz
        else:
            acc[i, 0] = ax
            acc[i, 1] = ay
            acc[i, 2] = az


@njit(cache=False, fastmath=True)
def compute_accelerations(
    acc, q, p, x, v, geom, fext, mode, m, rad, group_mobile,
    offsets, neigh,
    k_contact, gamma_contact, k_w, gamma_w, g_magnitude, omega, t,
    R0, eps, phi0, sigma, phi_eps, r_cap,
):
    for i in range(fext.shape[0]):
        fext[i, 0] = f32(0.0)
        fext[i, 1] = f32(0.0)
        fext[i, 2] = f32(0.0)
        acc[i, 0] = f32(0.0)
        acc[i, 1] = f32(0.0)
        acc[i, 2] = f32(0.0)

    compute_geometry_modes(q, p, geom, mode, R0, eps, phi0, sigma, phi_eps)
    map_to_physical(q, p, x, v, geom, mode)

    accumulate_contact_forces(
        fext, x, v, rad, group_mobile, offsets, neigh,
        k_contact, gamma_contact,
    )
    accumulate_wall_forces(
        fext, x, v, geom, mode, rad, group_mobile, k_w, gamma_w,
        R0, eps, phi0, sigma, r_cap,
    )

    phase = omega * t
    ax_body = g_magnitude * math.sin(phase)
    ay_body = f32(0.0)
    az_body = -g_magnitude * math.cos(phase)

    finalize_accelerations(
        acc, q, p, geom, fext, mode, group_mobile, m,
        ax_body, ay_body, az_body,
    )


@njit(cache=False, fastmath=True)
def rotate_y_one(x0, x1, x2, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return c * x0 + s * x2, x1, -s * x0 + c * x2


@njit(cache=False, fastmath=True)
def write_outputs(
    q_save, p_save, x_body_save, v_body_save, x_lab_save,
    q, p, x, v, mode, q_display, p_display, omega, t,
    R0, eps, phi0, sigma, phi_eps,
):
    angle = omega * t
    for i in range(q.shape[0]):
        if mode[i] == CURV:
            q_display[i, 0] = q[i, 0]
            q_display[i, 1] = q[i, 1]
            q_display[i, 2] = q[i, 2]
            p_display[i, 0] = p[i, 0]
            p_display[i, 1] = p[i, 1]
            p_display[i, 2] = p[i, 2]
        else:
            r_out, theta_out, phi_out = physical_position_to_curvilinear_output(
                q[i, 0], q[i, 1], q[i, 2],
                q_display[i, 1], q_display[i, 2],
                R0, eps, phi0, sigma,
            )
            q_display[i, 0] = r_out
            q_display[i, 1] = theta_out
            q_display[i, 2] = phi_out

        q_save[i, 0] = q_display[i, 0]
        q_save[i, 1] = q_display[i, 1]
        q_save[i, 2] = q_display[i, 2]
        p_save[i, 0] = p_display[i, 0]
        p_save[i, 1] = p_display[i, 1]
        p_save[i, 2] = p_display[i, 2]

        x_body_save[i, 0] = x[i, 0]
        x_body_save[i, 1] = x[i, 1]
        x_body_save[i, 2] = x[i, 2]
        v_body_save[i, 0] = v[i, 0]
        v_body_save[i, 1] = v[i, 1]
        v_body_save[i, 2] = v[i, 2]

        xl0, xl1, xl2 = rotate_y_one(x[i, 0], x[i, 1], x[i, 2], angle)
        x_lab_save[i, 0] = xl0
        x_lab_save[i, 1] = xl1
        x_lab_save[i, 2] = xl2


@njit(cache=False, fastmath=True)
def velocity_verlet_step(
    q, p, x, v, geom, acc, fext, mode, dt, half_dt,
    m, rad, group_mobile,
    box, r_list, cell_size, skin,
    offsets, neigh, x_verlet_ref,
    k_contact, gamma_contact, k_w, gamma_w, g_magnitude, omega, t,
    R0, eps, phi0, sigma, phi_eps, r_cap, r_exit, axis_cap, axis_exit,
):
    for i in range(q.shape[0]):
        p[i, 0] += half_dt * acc[i, 0]
        p[i, 1] += half_dt * acc[i, 1]
        p[i, 2] += half_dt * acc[i, 2]

    for i in range(q.shape[0]):
        q[i, 0] += dt * p[i, 0]
        q[i, 1] += dt * p[i, 1]
        q[i, 2] += dt * p[i, 2]

    apply_safeguard(q, p, mode, phi_eps)
    handoff(
        q, p, mode, group_mobile, R0, eps, phi0, sigma, phi_eps,
        r_cap, r_exit, axis_cap, axis_exit,
    )
    compute_geometry_modes(q, p, geom, mode, R0, eps, phi0, sigma, phi_eps)
    map_to_physical(q, p, x, v, geom, mode)

    if needs_verlet_rebuild_3d(x, x_verlet_ref, skin):
        offsets, neigh = build_verlet_csr_3d(x, box, r_list, cell_size)
        x_verlet_ref[:, :] = x

    compute_accelerations(
        acc, q, p, x, v, geom, fext, mode, m, rad, group_mobile,
        offsets, neigh,
        k_contact, gamma_contact, k_w, gamma_w, g_magnitude, omega, t + dt,
        R0, eps, phi0, sigma, phi_eps, r_cap,
    )

    for i in range(q.shape[0]):
        p[i, 0] += half_dt * acc[i, 0]
        p[i, 1] += half_dt * acc[i, 1]
        p[i, 2] += half_dt * acc[i, 2]

    compute_geometry_modes(q, p, geom, mode, R0, eps, phi0, sigma, phi_eps)
    map_to_physical(q, p, x, v, geom, mode)

    return offsets, neigh


@njit(cache=False, fastmath=True)
def simulate_curvilinear_particles_jit(
    box, q0, p0, m, rad, dt, T_max,
    group_mobile,
    k_contact, gamma_contact, k_w, gamma_w, g_magnitude, omega,
    R0, eps, phi0, sigma,
    r_list, skin, cell_size, save_every, phi_eps, r_cap, r_exit, axis_cap, axis_exit,
):
    N = q0.shape[0]
    num_steps = int(T_max / dt + f32(0.5))
    num_save = num_steps // save_every + 1

    t_out = np.empty(num_save, dtype=f32)
    q_out = np.empty((num_save, N, 3), dtype=f32)
    p_out = np.empty((num_save, N, 3), dtype=f32)
    x_body_out = np.empty((num_save, N, 3), dtype=f32)
    v_body_out = np.empty((num_save, N, 3), dtype=f32)
    x_lab_out = np.empty((num_save, N, 3), dtype=f32)

    q = q0.copy()
    p = p0.copy()
    mode = np.zeros(N, dtype=i64)
    apply_safeguard(q, p, mode, phi_eps)
    q_display = q.copy()
    p_display = p.copy()
    handoff(
        q, p, mode, group_mobile, R0, eps, phi0, sigma, phi_eps,
        r_cap, r_exit, axis_cap, axis_exit,
    )

    x = np.empty((N, 3), dtype=f32)
    v = np.empty((N, 3), dtype=f32)
    geom = np.empty((N, 24), dtype=f32)
    fext = np.empty((N, 3), dtype=f32)
    acc = np.empty((N, 3), dtype=f32)

    compute_geometry_modes(q, p, geom, mode, R0, eps, phi0, sigma, phi_eps)
    map_to_physical(q, p, x, v, geom, mode)

    x_verlet_ref = x.copy()
    offsets, neigh = build_verlet_csr_3d(x, box, r_list, cell_size)

    t = f32(0.0)
    compute_accelerations(
        acc, q, p, x, v, geom, fext, mode, m, rad, group_mobile,
        offsets, neigh,
        k_contact, gamma_contact, k_w, gamma_w, g_magnitude, omega, t,
        R0, eps, phi0, sigma, phi_eps, r_cap,
    )

    save_id = 0
    t_out[save_id] = f32(0.0)
    write_outputs(
        q_out[save_id], p_out[save_id],
        x_body_out[save_id], v_body_out[save_id], x_lab_out[save_id],
        q, p, x, v, mode, q_display, p_display, omega, f32(0.0),
        R0, eps, phi0, sigma, phi_eps,
    )
    save_id += 1

    half_dt = f32(0.5) * dt
    for n in range(num_steps):
        t = f32(n) * dt
        offsets, neigh = velocity_verlet_step(
            q, p, x, v, geom, acc, fext, mode, dt, half_dt,
            m, rad, group_mobile,
            box, r_list, cell_size, skin,
            offsets, neigh, x_verlet_ref,
            k_contact, gamma_contact, k_w, gamma_w, g_magnitude, omega, t,
            R0, eps, phi0, sigma, phi_eps, r_cap, r_exit, axis_cap, axis_exit,
        )

        if (n + 1) % save_every == 0:
            ts = f32(n + 1) * dt
            t_out[save_id] = ts
            write_outputs(
                q_out[save_id], p_out[save_id],
                x_body_out[save_id], v_body_out[save_id], x_lab_out[save_id],
                q, p, x, v, mode, q_display, p_display, omega, ts,
                R0, eps, phi0, sigma, phi_eps,
            )
            save_id += 1

    return t_out, q_out, p_out, x_body_out, v_body_out, x_lab_out


def simulate_curvilinear_particles(
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
    R0=1.0,
    eps=0.55,
    phi0=0.35,
    sigma=0.20,
    save_every=1,
    box=None,
    r_list=None,
    skin=None,
    cell_size=None,
    box_padding=0.5,
    phi_eps=1.0e-3,
    r_cap=0.05,
    r_exit=None,
    axis_cap=0.10,
    axis_exit=None,
):
    q0 = as_f32(q0)
    p0 = as_f32(p0)
    m = as_f32(m)
    rad = as_f32(rad)
    group_mobile = as_i64(group_mobile)

    if box is None:
        # A conservative fixed box for body-frame Verlet cells.
        R_bound = float(R0) * (1.0 + abs(float(eps)))
        pad = float(box_padding) + float(np.max(rad))
        lim = R_bound + pad
        box = np.array([-lim, lim, -lim, lim, -lim, lim], dtype=f32)
    else:
        box = as_f32(box)

    max_rad = f32(np.max(rad))
    skin_val = f32(skin if skin is not None else f32(0.25) * max_rad)
    r_contact = f32(2.0) * max_rad
    r_list_val = f32(r_list if r_list is not None else r_contact + skin_val)
    cell_size_val = f32(cell_size if cell_size is not None else r_list_val)
    r_cap_val = f32(r_cap)
    r_exit_val = f32(r_exit if r_exit is not None else f32(1.5) * r_cap_val)
    axis_cap_val = f32(axis_cap)
    axis_exit_val = f32(axis_exit if axis_exit is not None else f32(1.5) * axis_cap_val)

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
        f32(R0),
        f32(eps),
        f32(phi0),
        f32(sigma),
        r_list_val,
        skin_val,
        cell_size_val,
        int(save_every),
        f32(phi_eps),
        r_cap_val,
        r_exit_val,
        axis_cap_val,
        axis_exit_val,
    )


def curvilinear_to_physical_particles(
    q,
    p,
    R0=1.0,
    eps=0.55,
    phi0=0.35,
    sigma=0.20,
    omega=0.0,
    t=0.0,
    phi_eps=1.0e-3,
):
    q = as_f32(q)
    p = as_f32(p)
    N = q.shape[0]
    geom = np.empty((N, 24), dtype=f32)
    x_body = np.empty((N, 3), dtype=f32)
    v_body = np.empty((N, 3), dtype=f32)
    mode = np.zeros(N, dtype=i64)

    compute_geometry_modes(
        q, p, geom, mode,
        f32(R0), f32(eps), f32(phi0), f32(sigma), f32(phi_eps),
    )
    map_to_physical(q, p, x_body, v_body, geom, mode)

    x_lab = np.empty((N, 3), dtype=f32)
    angle = f32(omega) * f32(t)
    for i in range(N):
        x_lab[i, 0], x_lab[i, 1], x_lab[i, 2] = rotate_y_one(
            x_body[i, 0], x_body[i, 1], x_body[i, 2], angle
        )

    return x_lab, v_body
