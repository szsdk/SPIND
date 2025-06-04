from typing import Optional, Tuple

import numba
import numpy as np
import quaternion as quat

from ._hkl_matcher import gen_hkls
from ._utils import PEAKS_DTYPE


def _rand_quat() -> quat.quaternion:
    """
    Generate a random unit quaternion.

    Returns
    -------
    quat.quaternion
    """
    q = np.random.normal(size=4)
    q /= np.linalg.norm(q)
    return quat.quaternion(*q)


@numba.jit
def _simuate_kernel(qs, ewald_vec, peak_rad):
    ewald_rad = (ewald_vec[0] ** 2 + ewald_vec[1] ** 2 + ewald_vec[2] ** 2) ** 0.5
    ans = []
    for i in range(qs.shape[0]):
        d = (
            (qs[i, 0] - ewald_vec[0]) ** 2
            + (qs[i, 1] - ewald_vec[1]) ** 2
            + (qs[i, 2] - ewald_vec[2]) ** 2
        ) ** 0.5
        dr = abs(d - ewald_rad)
        if dr < peak_rad * 3:
            ql = (qs[i, 0] ** 2 + qs[i, 1] ** 2 + qs[i, 2] ** 2) ** 0.5
            if ql < 1e-20:
                continue
            ans.append((qs[i] / d * ewald_rad, np.exp(-((dr / peak_rad) ** 2)), 1 / ql))
    return ans


def simulate(
    qs: np.ndarray,
    ewald_rad: float,
    peak_rad: float,
    rot: Optional[quat.quaternion] = None,
) -> Tuple[quat.quaternion, np.ndarray]:
    """
    Simulate

    Parameters
    ----------
    qs : np.ndarray
       Bragg peaks

    ewald_rad : float
       Radius of Ewald's sphere in A^-1

    peak_rad : float
       Size of Bragg peaks in A^-1

    rot : Optional[quat.quaternion]
       If rot is not given, a random rotation quaternion is generated.


    Returns
    -------
    Tuple[quat.quaternion, np.ndarray[PEAKS_DTYPE]]
       Rotation quaternion and generated peaks
           ans['coor'] @ rot \\in qs
    """
    rot = _rand_quat() if rot is None else rot
    z = np.array([0, 0, -ewald_rad])
    ewald_vec = quat.rotate_vectors(rot.conj(), z)
    ans_raw = _simuate_kernel(qs, ewald_vec, peak_rad)
    ans = np.empty(len(ans_raw), PEAKS_DTYPE)
    for i, (coor, intensity, resolution) in enumerate(ans_raw):
        ans[i]["coor"] = coor
        ans[i]["intensity"] = intensity
        ans[i]["resolution"] = resolution
        ans[i]["snr"] = 1
    ans["coor"] = quat.rotate_vectors(rot, ans["coor"])
    return rot, ans
