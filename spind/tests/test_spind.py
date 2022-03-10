import numpy as np
import pytest
import spind
import quaternion as quat


@pytest.fixture()
def param():
    return spind.params('config.yml')


def test_gen_hkls(param: spind.Params):
    spind.gen_hkls(param)


def test_hkl_matcher(param: spind.Params):
    spind.hkl_matcher(param)


def gen_peaks(param, r:quat.quaternion):
    _, qs = spind.gen_hkls(param)
    qs = qs[3::1000] * 1e10
    qs = quat.rotate_vectors(r, qs)
    ans = np.empty(qs.shape[0], dtype=spind.PEAKS_DTYPE)
    ans['coor'] = qs
    ans['intensity'] = 1.0
    ans['snr'] = 1.0
    ans['resolution'] = 1 / np.linalg.norm(qs, axis=1)
    return ans


def test_index(param):
    r = quat.quaternion(0.5, 0.5, 0.5, 0.5)
    peaks = gen_peaks(param, r)
    hklmatcher = spind.hkl_matcher(param)
    solutions = spind.index(peaks, hklmatcher, param)
    assert len(solutions) == 1
    s = solutions[0]
    assert s.nb_peaks == peaks.shape[0]
    assert abs(s.pair_dist_refined) < 1e-7


def test_multiple_index(param):
    r1 = quat.quaternion(0.5, 0.5, 0.5, 0.5)
    r2 = quat.quaternion(0.6, 0.0, 0.8, 0.0)
    peaks = np.concatenate([gen_peaks(param, r) for r in [r1, r2]])
    param = param._replace(multi_index=True, nb_try=10)

    hklmatcher = spind.hkl_matcher(param)
    solutions = spind.index(peaks, hklmatcher, param)
    assert len(solutions) == 2
