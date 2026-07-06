"""
Numba kernels for Map 2: peristaltic boundary with small x-deformation.

Parameters:

    a = 3.0, alpha = -0.2, b = 0.6
    eps_x = 0.08, eps_y = 0.25, M = 3, omega = 2.0

Boundary:

    Bx(theta,t) = a cos(theta) + alpha cos(3 theta)
                  + eps_x sin(theta) cos(M theta - omega t)

    By(theta,t) = sin(theta) * (b - eps_y cos(M theta - omega t))

Interior map:

    Phi(t,r,theta) = r * B(theta,t)
"""

import math
import numpy as np
from numba import njit


f32 = np.float32
TWO_PI = f32(2.0 * np.pi)

A = f32(3.0)
ALPHA = f32(-0.2)
B0 = f32(0.6)
EPS_X = f32(0.08)
EPS_Y = f32(0.25)
EPS = f32(0.25)
M = f32(3.0)
OMEGA = f32(2.0)


@njit(cache=False, fastmath=True)
def wrap_theta(theta):
    return theta % TWO_PI


@njit(cache=False, fastmath=True)
def eval_boundary(theta, t):
    """Return B, B_theta, B_thetatheta, B_t, B_thetat, B_tt."""
    phi = M * theta - OMEGA * t
    s = f32(math.sin(theta))
    c = f32(math.cos(theta))
    s3 = f32(math.sin(f32(3.0) * theta))
    c3 = f32(math.cos(f32(3.0) * theta))
    sp = f32(math.sin(phi))
    cp = f32(math.cos(phi))

    h = B0 - EPS_Y * cp

    bx = A * c + ALPHA * c3 + EPS_X * s * cp
    by = s * h

    bx_th = -A * s - f32(3.0) * ALPHA * s3 + EPS_X * (c * cp - M * s * sp)
    by_th = c * h + EPS_Y * M * s * sp

    bx_thth = (
        -A * c
        - f32(9.0) * ALPHA * c3
        + EPS_X * (-(f32(1.0) + M * M) * s * cp - f32(2.0) * M * c * sp)
    )
    by_thth = (
        -s * h
        + f32(2.0) * EPS_Y * M * c * sp
        + EPS_Y * M * M * s * cp
    )

    bx_t = EPS_X * OMEGA * s * sp
    by_t = -EPS_Y * OMEGA * s * sp

    bx_tht = EPS_X * OMEGA * (c * sp + M * s * cp)
    by_tht = -EPS_Y * OMEGA * c * sp - EPS_Y * M * OMEGA * s * cp

    bx_tt = -EPS_X * OMEGA * OMEGA * s * cp
    by_tt = EPS_Y * OMEGA * OMEGA * s * cp

    return (
        bx, by,
        bx_th, by_th,
        bx_thth, by_thth,
        bx_t, by_t,
        bx_tht, by_tht,
        bx_tt, by_tt,
    )


@njit(cache=False, fastmath=True)
def eval_map(r, theta, t):
    """Return physical position x = Phi(t,r,theta)."""
    bx, by, _, _, _, _, _, _, _, _, _, _ = eval_boundary(theta, t)
    return r * bx, r * by


@njit(cache=False, fastmath=True)
def eval_jacobian(r, theta, t):
    """Return J = d Phi / d(r,theta) as four scalars."""
    bx, by, bx_th, by_th, _, _, _, _, _, _, _, _ = eval_boundary(theta, t)
    return bx, r * bx_th, by, r * by_th


@njit(cache=False, fastmath=True)
def eval_cross(theta, t):
    """Return C = Bx * By_theta - By * Bx_theta."""
    bx, by, bx_th, by_th, _, _, _, _, _, _, _, _ = eval_boundary(theta, t)
    return bx * by_th - by * bx_th


@njit(cache=False, fastmath=True)
def eval_inverse_jacobian(r, theta, t, min_r=f32(1.0e-6)):
    """Return J^{-1} as four scalars."""
    r_eff = r
    if r_eff < min_r:
        r_eff = min_r

    bx, by, bx_th, by_th, _, _, _, _, _, _, _, _ = eval_boundary(theta, t)
    det_j = r_eff * (bx * by_th - by * bx_th)
    inv_det = f32(1.0) / det_j

    return (
        r_eff * by_th * inv_det,
        -r_eff * bx_th * inv_det,
        -by * inv_det,
        bx * inv_det,
    )


@njit(cache=False, fastmath=True)
def eval_metric(r, theta, t):
    """Return metric entries g_rr, g_rtheta, g_thetatheta."""
    bx, by, bx_th, by_th, _, _, _, _, _, _, _, _ = eval_boundary(theta, t)
    g_rr = bx * bx + by * by
    g_rt = r * (bx * bx_th + by * by_th)
    g_tt = r * r * (bx_th * bx_th + by_th * by_th)
    return g_rr, g_rt, g_tt


@njit(cache=False, fastmath=True)
def eval_time_terms(r, theta, t):
    """Return Phi_t, Phi_tt, and J_t as scalar components."""
    (
        _, _, _, _, _, _,
        bx_t, by_t,
        bx_tht, by_tht,
        bx_tt, by_tt,
    ) = eval_boundary(theta, t)

    phi_t_x = r * bx_t
    phi_t_y = r * by_t
    phi_tt_x = r * bx_tt
    phi_tt_y = r * by_tt

    jt00 = bx_t
    jt01 = r * bx_tht
    jt10 = by_t
    jt11 = r * by_tht

    return phi_t_x, phi_t_y, phi_tt_x, phi_tt_y, jt00, jt01, jt10, jt11


@njit(cache=False, fastmath=True)
def eval_hessian_terms(r, theta, t):
    """Return Hessian components of Phi for q=(r,theta)."""
    _, _, bx_th, by_th, bx_thth, by_thth, _, _, _, _, _, _ = eval_boundary(theta, t)
    return (
        f32(0.0), f32(0.0),
        bx_th, by_th,
        r * bx_thth, r * by_thth,
    )


@njit(cache=False, fastmath=True)
def eval_convective_term(r, theta, rdot, thetadot, t):
    """Return Hessian(Phi)[qdot,qdot]."""
    _, _, bx_th, by_th, bx_thth, by_thth, _, _, _, _, _, _ = eval_boundary(theta, t)
    c0 = f32(2.0) * rdot * thetadot * bx_th + r * thetadot * thetadot * bx_thth
    c1 = f32(2.0) * rdot * thetadot * by_th + r * thetadot * thetadot * by_thth
    return c0, c1


@njit(cache=False, fastmath=True)
def curvilinear_to_physical_state(r, theta, rdot, thetadot, t):
    """Return x,y,vx,vy using xdot = Phi_t + J qdot."""
    (
        bx, by,
        bx_th, by_th,
        _, _,
        bx_t, by_t,
        _, _,
        _, _,
    ) = eval_boundary(theta, t)

    x = r * bx
    y = r * by
    vx = r * bx_t + bx * rdot + r * bx_th * thetadot
    vy = r * by_t + by * rdot + r * by_th * thetadot

    return x, y, vx, vy


@njit(cache=False, fastmath=True)
def physical_acceleration_to_curvilinear(r, theta, rdot, thetadot, ax, ay, t):
    """Return qddot from Eq. 72 for the time-dependent map."""
    (
        _, _, phi_tt_x, phi_tt_y,
        jt00, jt01, jt10, jt11,
    ) = eval_time_terms(r, theta, t)
    h0, h1 = eval_convective_term(r, theta, rdot, thetadot, t)
    inv00, inv01, inv10, inv11 = eval_inverse_jacobian(r, theta, t)

    tx = ax - phi_tt_x - f32(2.0) * (jt00 * rdot + jt01 * thetadot) - h0
    ty = ay - phi_tt_y - f32(2.0) * (jt10 * rdot + jt11 * thetadot) - h1

    return inv00 * tx + inv01 * ty, inv10 * tx + inv11 * ty


@njit(cache=False, fastmath=True)
def eval_wall_kinematics(theta, t):
    """Return wall point, velocity, acceleration, inward normal, and C."""
    (
        bx, by,
        bx_th, by_th,
        _, _,
        bx_t, by_t,
        _, _,
        bx_tt, by_tt,
    ) = eval_boundary(theta, t)

    tangent_norm = f32(math.sqrt(bx_th * bx_th + by_th * by_th))
    inv_norm = f32(1.0) / tangent_norm
    nx_in = -by_th * inv_norm
    ny_in = bx_th * inv_norm
    cross = bx * by_th - by * bx_th

    return bx, by, bx_t, by_t, bx_tt, by_tt, nx_in, ny_in, tangent_norm, cross


@njit(cache=False, fastmath=True)
def relative_wall_normal_velocity(r, theta, rdot, thetadot, theta_wall, t):
    """Return (xdot_particle - xdot_wall) dot n_in(theta_wall,t)."""
    _, _, vx, vy = curvilinear_to_physical_state(r, theta, rdot, thetadot, t)
    (
        _, _, wx, wy, _, _,
        nx_in, ny_in, _, _,
    ) = eval_wall_kinematics(theta_wall, t)
    return (vx - wx) * nx_in + (vy - wy) * ny_in
