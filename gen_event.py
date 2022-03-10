import itertools
import numpy as np
import yaml
from pathlib import Path


def calc_transform_matrix(cell_param, lattice_type=None):
    a, b, c = np.asarray(cell_param[0:3])
    al, be, ga = cell_param[3:]
    if lattice_type == "monoclinic":
        av = [a, 0.0, 0.0]
        bv = [0, b, 0.0]
        cv = [c * np.cos(np.deg2rad(be)), 0, c * np.sin(np.deg2rad(be))]
    elif lattice_type == "orthorhombic":
        av = [a, 0.0, 0.0]
        bv = [0.0, b, 0.0]
        cv = [0.0, 0.0, c]
        assert al == 90.0
        assert be == 90.0
        assert ga == 90.0
    else:
        raise NotImplementedError("%s not implemented yet" % lattice_type)
    a_star = (np.cross(bv, cv)) / ((np.cross(bv, cv).dot(av)))
    b_star = (np.cross(cv, av)) / ((np.cross(cv, av).dot(bv)))
    c_star = (np.cross(av, bv)) / ((np.cross(av, bv).dot(cv)))
    A = np.zeros((3, 3), dtype=np.float64)  # transform matrix
    A[:, 0] = a_star
    A[:, 1] = b_star
    A[:, 2] = c_star
    return A


class HKLMatcher:
    def __init__(self, hkls, qs, la, seed_len_tol, seed_angle_tol):
        self.hkls = hkls
        self.qs = qs
        self.la = la
        self.seed_len_tol = seed_len_tol
        self.seed_angle_tol = seed_angle_tol


def hkl_matcher(config):
    res_cutoff = config["resolution cutoff"]
    lattice_type = config["lattice type"]
    cell_param = np.asarray(config["cell parameters"])
    cell_param[:3] *= 1e-10  # convert to meters
    centering = config["centering"]
    if "hkl file" in config:
        hkl_file = config["hkl file"]
        print("Generating reference table from hkl file: %s" % hkl_file)
    else:
        hkl_file = None

    # a, b, c star
    A = calc_transform_matrix(cell_param, lattice_type=lattice_type)
    a_star, b_star, c_star = A[:, 0], A[:, 1], A[:, 2]

    q_cutoff = 1.0 / res_cutoff
    max_h = int(np.ceil(q_cutoff / np.linalg.norm(a_star)))
    max_k = int(np.ceil(q_cutoff / np.linalg.norm(b_star)))
    max_l = int(np.ceil(q_cutoff / np.linalg.norm(c_star)))
    print("max_h: %d, max_k: %d, max_l: %d" % (max_h, max_k, max_l))
    if hkl_file is None:
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
        qs = A.dot(hkls.T).T
        valid_idx = np.linalg.norm(qs, axis=1) < q_cutoff
        # valid_idx = []
        # for i in range(len(qs)):
            # if np.linalg.norm(qs[i]) <= q_cutoff:
                # valid_idx.append(i)
        hkls = hkls[valid_idx]

        # apply systematic absence
        if centering == "I":  # h+k+l == 2n
            valid_idx = hkls.sum(axis=1) % 2 == 0
        elif centering == "A":  # k+l == 2n
            valid_idx = (hkls[:, 1] + hkls[:, 2]) % 2 == 0
        elif centering == "B":  # h+l == 2n
            valid_idx = (hkls[:, 0] + hkls[:, 2]) % 2 == 0
        elif centering == "C":  # h+k == 2n
            valid_idx = (hkls[:, 0] + hkls[:, 1]) % 2 == 0
        elif centering == "P":
            valid_idx = np.ones(hkls.shape[0]) > 0  # all true
        else:
            raise NotImplementedError("%s not implemented" % centering)
        hkls = hkls[valid_idx]
    else:  # load 1/8 hkls from file
        hkl = np.loadtxt(hkl_file, dtype=np.int16)
        valid_idx = (hkl[:, 0] <= max_h) * (hkl[:, 1] <= max_k) * (hkl[:, 2] <= max_l)
        hkl = hkl[valid_idx]
        hk_l = hkl.copy()
        hk_l[:, 2] = -hkl[:, 2]  # h,k,-l
        h_kl = hkl.copy()
        h_kl[:, 1] = -hkl[:, 1]  # h,-k,l
        _hkl = hkl.copy()
        _hkl[:, 0] = -hkl[:, 0]  # -h,k,l
        h_k_l = -_hkl.copy()  # h,-k,-l
        _h_kl = -hk_l.copy()  # -h,-k,l
        _hk_l = -h_kl.copy()  # -h,k,-l
        _h_k_l = -hkl.copy()  # -h,-k,-l
        hkls = np.vstack((hkl, hk_l, h_kl, _hkl, h_k_l, _h_kl, _hk_l, _h_k_l))
        hkls = hkls.tolist()
        hkls.sort()
        hkls = np.asarray(list(hkl for hkl, _ in itertools.groupby(hkls)))

    # generate table

    qs = A.dot(hkls.T).T
    la = np.linalg.norm(qs, axis=1)
    idx = np.argsort(la)
    return HKLMatcher(
        hkls[idx],
        qs[idx],
        la[idx],
        seed_len_tol=float(config["seed length tolerance"]),
        seed_angle_tol=float(config["seed angle tolerance"])
    )

with open("config.yml") as fp:
    config = yaml.safe_load(fp)
hklmatcher = hkl_matcher(config)

for i in range(100):
    n = np.random.randint(35, 50)
    ans = np.zeros((n, 8))
    idx = np.random.choice(hklmatcher.qs.shape[0], n)
    ans[:, 4:7] = hklmatcher.qs[idx]
    ans[:, 7] = 1 / np.linalg.norm(ans[:, 4:7], axis=1)
    ans[:, :3] = np.random.rand(n, 3)
    print(ans)
    print(hklmatcher.hkls[idx])
    np.savetxt(f"events/event-{i}.txt", ans)
