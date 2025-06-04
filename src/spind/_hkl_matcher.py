import numba
import numpy as np
from scipy.spatial import KDTree

from ._params import Params


@numba.jit(boundscheck=False, cache=True, nopython=True)
def get_alpha(al, bl, c, cl, d, dl, C1):  # pragma: no cover
    n = np.empty(3, np.float64)
    ans = np.empty((3, 2), np.float64)

    A = al * cl
    B = bl * dl

    cosC2 = c @ d / cl / dl
    if abs(cosC2) > 1:
        cosC2 /= cosC2
    C2 = np.arccos(cosC2)
    C = C1 - C2
    cosC = np.cos(C)
    sinC = np.sin(C)

    s = 0
    n[0] = c[1] * d[2] - c[2] * d[1]
    s += n[0] ** 2
    n[1] = c[2] * d[0] - c[0] * d[2]
    s += n[1] ** 2
    n[2] = c[0] * d[1] - c[1] * d[0]
    s += n[2] ** 2
    n /= s**0.5

    x, y = A + B * cosC, B * sinC
    rxy = (x**2 + y**2) ** 0.5
    cl = al / cl / rxy
    x *= cl
    y *= cl
    ans[0, 0] = (c[1] * n[2] - c[2] * n[1]) * y + x * c[0]
    ans[1, 0] = (c[2] * n[0] - c[0] * n[2]) * y + x * c[1]
    ans[2, 0] = (c[0] * n[1] - c[1] * n[0]) * y + x * c[2]

    dl = bl / dl / rxy
    x = (B + A * cosC) * dl
    y = -A * sinC * dl
    ans[0, 1] = (d[1] * n[2] - d[2] * n[1]) * y + x * d[0]
    ans[1, 1] = (d[2] * n[0] - d[0] * n[2]) * y + x * d[1]
    ans[2, 1] = (d[0] * n[1] - d[1] * n[0]) * y + x * d[2]

    return ans


@numba.jit(inline="always", cache=True, nopython=True)
def my_norm(a):  # pragma: no cover
    return (a[0] ** 2 + a[1] ** 2 + a[2] ** 2) ** 0.5


@numba.jit(boundscheck=False, cache=True, nopython=True)
def get_rot(a, ap):  # pragma: no cover
    d0 = ap[0] - a[0]
    d1 = ap[1] - a[1]
    l0 = my_norm(d0)
    l1 = my_norm(d1)
    if (l0 < 1e-4) and (l1 < 1e-4):
        return np.eye(3)

    if l0 < 1e-3:
        axis = a[0]
    elif l1 < 1e-3:
        axis = a[1]
    else:
        axis = np.cross(d0, d1)
        axis_l = my_norm(axis)
        if axis_l < 1e-3:
            if abs(l1 - l0) < 1e-3:
                axis = a[0] - a[1]
            else:
                l1 = (d1[0] * d0[0] + d1[1] * d0[1] + d1[2] * d0[2]) / l0
                dl = l1 - l0
                axis = l1 / dl * a[0] - l0 / dl * a[1]
                # k0, k1 = l1 / dl, l0 / dl
                # axis[0] = k0 * a[0, 0] + k1 * a[1, 0]
                # axis[1] = k0 * a[0, 1] + k1 * a[1, 1]
                # axis[2] = k0 * a[0, 2] + k1 * a[1, 2]

    if l0 < 1e-3:
        n0 = np.cross(axis, a[1])
        n1 = np.cross(axis, ap[1])
    else:
        n0 = np.cross(axis, a[0])
        n1 = np.cross(axis, ap[0])

    cosC = n0 @ n1 / my_norm(n0) / my_norm(n1)
    if abs(cosC) > 1:
        cosC /= cosC
    angle = np.arccos(cosC) / 2
    nn = np.cross(n0, n1)
    nl = my_norm(nn)
    axis = -nn / nl

    sinA = np.sin(angle)
    q0 = np.cos(angle)
    q1 = sinA * axis[0]
    q2 = sinA * axis[1]
    q3 = sinA * axis[2]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    return np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])


@numba.jit(boundscheck=False, nogil=True, cache=True, nopython=True)
def _hkl_match_kernel(
    q12, q1s, q1ns, q1l, q2s, q2ns, q2l, ang_s, ang_e, A0_inv, seed_hkl_tol
):  # pragma: no cover
    ref12 = np.empty((2, 3), np.float64)
    al = (q12[0, 0] ** 2 + q12[1, 0] ** 2 + q12[2, 0] ** 2) ** 0.5
    bl = (q12[0, 1] ** 2 + q12[1, 1] ** 2 + q12[2, 1] ** 2) ** 0.5
    C1 = (
        (q12[0, 0] * q12[0, 1] + q12[1, 0] * q12[1, 1] + q12[2, 0] * q12[2, 1])
        / al
        / bl
    )
    if abs(C1) > 1:
        C1 /= C1
    C1 = np.arccos(C1)
    Rs = []
    seed_errors = []
    # Rs = np.empty((0, 3, 3))
    # seed_errors = np.empty((0,))
    num_test = 0
    w = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], np.float64)
    for i1 in range(q1s.shape[0]):
        # na = 0
        for i2 in range(q2s.shape[0]):
            ang = (
                q1ns[i1, 0] * q2ns[i2, 0]
                + q1ns[i1, 1] * q2ns[i2, 1]
                + q1ns[i1, 2] * q2ns[i2, 2]
            )
            if (ang_s > ang) or (ang > ang_e):
                continue
            num_test += 1

            q12_rot = get_alpha(al, bl, q1s[i1], q1l[i1], q2s[i2], q2l[i2], C1)
            hkl = np.dot(A0_inv, q12_rot).ravel()
            seed_error = 0
            for i in range(6):
                se = abs(hkl[i] - round(hkl[i]))
                if se > seed_hkl_tol:
                    break
                if se > seed_error:
                    seed_error = se
            else:
                ref12[0] = q1s[i1]
                ref12[1] = q2s[i2]
                h = q12 @ ref12
                u, s, vh = np.linalg.svd(h)
                R = u @ vh
                if np.linalg.det(R) < 0:
                    R = u @ w @ vh
                Rs.append(R)
                seed_errors.append(seed_error)
    return Rs, seed_errors


class HKLMatcher:
    def __init__(self, hkls, qs, qls, seed_len_tol):
        self.hkls = hkls
        self.qs = qs
        self.qls = qls
        qls = qls.reshape(-1, 1)
        self.qns = np.divide(self.qs, qls, where=(np.abs(qls) > 1e-10))
        hklls = np.linalg.norm(self.hkls, axis=1)
        self.q_over_hkl = np.divide(self.qls, hklls, where=(np.abs(hklls) > 1e-10))
        self.seed_len_tol = seed_len_tol
        self.hkl_tree = KDTree(self.hkls)

    def as_dict(self):
        return {"hkls": self.hkls, "qs": self.qs}

    def __call__(self, q1, q2, A0_inv, seed_hkl_tol):
        q1l = np.linalg.norm(q1)
        q2l = np.linalg.norm(q2)
        ang12 = np.dot(q1, q2) / q1l / q2l
        if abs(ang12) > 1:
            ang12 = np.sign(ang12)
        ang12 = np.arccos(ang12)
        dq1 = np.abs(q1l - self.qls)
        idx1 = dq1 < self.seed_len_tol
        idx1 = idx1 & ((seed_hkl_tol * self.q_over_hkl) > dq1)

        dq2 = np.abs(q2l - self.qls)
        idx2 = dq2 < self.seed_len_tol
        idx2 = idx2 & ((seed_hkl_tol * self.q_over_hkl) > dq2)

        angle_tol = self.seed_len_tol * max(1 / q1l, 1 / q2l)
        q12 = np.array([q1, q2]).T.copy()
        Rs, es = _hkl_match_kernel(
            q12,
            self.qs[idx1],
            self.qns[idx1],
            self.qls[idx1],
            self.qs[idx2],
            self.qns[idx2],
            self.qls[idx2],
            np.cos(ang12 + angle_tol),
            np.cos(ang12 - angle_tol),
            A0_inv,
            seed_hkl_tol,
        )
        if len(Rs) == 0:
            return np.empty((0, 3, 3)), np.empty((0,))
        else:
            return np.array(Rs), np.array(es)


def gen_hkls(p: Params, ignore_miller_set=False):
    a_star, b_star, c_star = p.transform_matrix
    q_cutoff = 1.0 / p.res_cutoff
    if (p.miller_set is not None) and (not ignore_miller_set):
        hkls = np.load(p.miller_set)
    else:
        max_h = int(np.ceil(q_cutoff / np.linalg.norm(a_star)))
        max_k = int(np.ceil(q_cutoff / np.linalg.norm(b_star)))
        max_l = int(np.ceil(q_cutoff / np.linalg.norm(c_star)))
        # hkl grid
        hh = np.arange(-max_h, max_h + 1)
        kk = np.arange(-max_k, max_k + 1)
        ll = np.arange(-max_l, max_l + 1)

        hs, ks, ls = np.meshgrid(hh, kk, ll)
        hkls = np.ones((hs.size, 3))
        hkls[:, 0] = hs.reshape((-1))
        hkls[:, 1] = ks.reshape((-1))
        hkls[:, 2] = ls.reshape((-1))

    # remove high resolution hkls
    qs = hkls @ p.transform_matrix.T
    valid_idx = np.linalg.norm(qs, axis=1) < q_cutoff
    qs = qs[valid_idx]
    hkls = hkls[valid_idx]

    if (p.miller_set is None) and (not ignore_miller_set):
        # apply systematic absence
        if p.centering == "I":  # h+k+l == 2n
            valid_idx = hkls.sum(axis=1) % 2 == 0
        elif p.centering == "A":  # k+l == 2n
            valid_idx = (hkls[:, 1] + hkls[:, 2]) % 2 == 0
        elif p.centering == "B":  # h+l == 2n
            valid_idx = (hkls[:, 0] + hkls[:, 2]) % 2 == 0
        elif p.centering == "C":  # h+k == 2n
            valid_idx = (hkls[:, 0] + hkls[:, 1]) % 2 == 0
        elif p.centering == "P":
            valid_idx = np.ones(hkls.shape[0], bool)
        else:
            raise NotImplementedError("%s not implemented" % p.centering)
        hkls = hkls[valid_idx]
        qs = qs[valid_idx]
    return hkls, qs


def hkl_matcher(p: Params):
    hkls, qs = gen_hkls(p)
    la = np.linalg.norm(qs, axis=1)
    idx = np.argsort(la)
    return HKLMatcher(
        hkls[idx],
        qs[idx],
        la[idx],
        seed_len_tol=p.seed_len_tol,
    )
