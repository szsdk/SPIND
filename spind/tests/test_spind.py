import tempfile
from dataclasses import fields

import h5py
import numpy as np
import pytest
import quaternion as quat
import yaml

import spind
from spind.simulator import simulate


@pytest.fixture()
def param():
    return spind.params("config.yml")


def test_gen_hkls(param: spind.Params):
    spind.gen_hkls(param)


def test_hkl_matcher(param: spind.Params):
    spind.hkl_matcher(param)


def gen_peaks(qs, r: quat.quaternion):
    qs = qs[3::1000]
    qs = quat.rotate_vectors(r, qs)
    ans = np.empty(qs.shape[0], dtype=spind.PEAKS_DTYPE)
    ans["coor"] = qs
    ans["intensity"] = 1.0
    ans["snr"] = 1.0
    ans["resolution"] = 1 / np.linalg.norm(qs, axis=1)
    return ans


def index_test_set():
    with open("config.yml") as fp:
        config = yaml.safe_load(fp)
    config["resolution cutoff"] = 20
    for cpara in [
        [103.45, 50.28, 69.380, 90.00, 109.67, 90.00],
        [[102, 1, 0], [0, 84, 40], [12, -32, 80]],
    ]:
        config["cell parameters"] = cpara
        p = spind.params(config)
        hklmatcher = spind.hkl_matcher(p)
        _, qs = spind.gen_hkls(p)
        for _ in range(10):
            r, peaks = simulate(qs, 0.2, 0.001)
            yield p, hklmatcher, peaks, r


@pytest.mark.parametrize("p, hklmatcher, peaks, r", index_test_set())
def test_index(p, hklmatcher, peaks, r):
    peaks = peaks[np.argsort(peaks["resolution"])]
    s_g = spind.eval_rot(peaks["coor"], hklmatcher, quat.as_rotation_matrix(r), p)
    solutions = spind.index(peaks, hklmatcher, p)
    assert len(solutions) == 1
    s = solutions[0]
    assert np.nanmean(s.ehkls) <= (s_g.ehkls.mean() * 3)
    # np.testing.assert_almost_equal(
    # np.abs(s.hkls),
    # np.abs(s_g.hkls)
    # )
    if not np.allclose(np.abs(s.hkls), np.abs(s_g.hkls)):
        from IPython import embed

        embed()


def test_multiple_index(param):
    r1 = quat.quaternion(0.5, 0.5, 0.5, 0.5)
    r2 = quat.quaternion(0.6, 0.0, 0.8, 0.0)
    _, qs = spind.gen_hkls(param)
    peaks = np.concatenate([gen_peaks(qs, r) for r in [r1, r2]])
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


def test_simulate(param):
    _, qs = spind.gen_hkls(param)
    simulate(qs, 0.2, 0.0001)
