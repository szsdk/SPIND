import tempfile
from dataclasses import fields

import h5py
import numpy as np
import pytest
import quaternion as quat

import spind


@pytest.fixture()
def param():
    return spind.params("config.yml")


def test_gen_hkls(param: spind.Params):
    spind.gen_hkls(param)


def test_hkl_matcher(param: spind.Params):
    spind.hkl_matcher(param)


def gen_peaks(param, r: quat.quaternion):
    _, qs = spind.gen_hkls(param)
    qs = qs[3::1000]
    qs = quat.rotate_vectors(r, qs)
    ans = np.empty(qs.shape[0], dtype=spind.PEAKS_DTYPE)
    ans["coor"] = qs
    ans["intensity"] = 1.0
    ans["snr"] = 1.0
    ans["resolution"] = 1 / np.linalg.norm(qs, axis=1)
    return ans


def test_index(param):
    r = quat.quaternion(0.5, 0.5, 0.5, 0.5)
    peaks = gen_peaks(param, r)
    hklmatcher = spind.hkl_matcher(param)
    solutions = spind.index(peaks, hklmatcher, param)
    assert len(solutions) == 1
    s = solutions[0]
    assert s.nb_peaks == peaks.shape[0]
    assert abs(s.pair_dist) < 1e-7


def test_multiple_index(param):
    r1 = quat.quaternion(0.5, 0.5, 0.5, 0.5)
    r2 = quat.quaternion(0.6, 0.0, 0.8, 0.0)
    peaks = np.concatenate([gen_peaks(param, r) for r in [r1, r2]])
    param = param._replace(multi_index=True, nb_try=10)

    hklmatcher = spind.hkl_matcher(param)
    solutions = spind.index(peaks, hklmatcher, param)
    assert len(solutions) == 2


def test_Solution_IO():
    with tempfile.TemporaryDirectory() as dir:
        fn = f"{dir}/t.h5"
        with h5py.File(fn, "w") as fp:
            sol = spind.Solution(nb_peaks=32)
            sol.write_h5(fp)
            sol_r = spind.Solution.read_h5(fp)
            for f in fields(sol):
                item_a, item_b = getattr(sol, f.name), getattr(sol_r, f.name)
                if isinstance(item_a, np.ndarray):
                    np.testing.assert_equal(item_a, item_b)
                else:
                    assert item_a == item_b


def test_calc_transform_matrix():
    np.testing.assert_almost_equal(
        spind.calc_transform_matrix([1, 2, 3, 90, 90, 90]),
        spind.calc_transform_matrix(np.diag([1, 2, 3])),
    )
