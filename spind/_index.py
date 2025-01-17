#!/usr/bin/env python

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numba
import numpy as np
from scipy.optimize import fmin_cg, minimize

from ._params import Params
from ._solution import Solution


def correct_match_rate(sol, qs, miller_set=None, centering_weight=0.0):
    """
    Correct match rate and pair peaks with miller set constraint.
    :param qs: peak coordinates in fourier space, Nx3 array.
    :param miller_set: possible hkl indices, Nx3 array.
    :param centering_weight: weight of centering score.
    :return: None
    """
    dtype = "int8"
    sol_c = copy.deepcopy(sol)
    if miller_set is None:  # dummy case
        sol_c.match_rate = sol.match_rate
        sol_c.pair_ids = sol.pair_ids
        eXYZs = np.abs(sol.transform_matrix.dot(sol.rhkls.T) - qs.T).T
        sol_c.pair_dist = np.linalg.norm(eXYZs, axis=1)[sol_c.pair_ids].mean()
        sol_c.total_score = sol.total_score
    else:
        miller_set = miller_set.astype(dtype)
        true_pair_ids = []
        for pair_id in sol.pair_ids:
            if (
                np.linalg.norm(
                    miller_set - sol.rhkls[pair_id].astype(dtype), axis=1
                ).min()
                == 0
            ):
                true_pair_ids.append(pair_id)
        true_match_rate = float(len(true_pair_ids)) / sol.nb_peaks
        eXYZs = np.abs(sol.transform_matrix.dot(sol.rhkls.T) - qs.T).T
        true_pair_dist = np.linalg.norm(eXYZs, axis=1)[true_pair_ids].mean()
        true_pair_hkls = sol.rhkls[true_pair_ids].astype(dtype)
        centering_score = calc_centering_score(true_pair_hkls, centering=sol.centering)
        true_total_score = centering_score * centering_weight + true_match_rate
        sol_c.pair_ids = true_pair_ids
        sol_c.match_rate = true_match_rate
        sol_c.pair_dist = true_pair_dist
        sol_c.total_score = true_total_score
    return sol_c


def index(
    peaks, hklmatcher, p: Params, num_threads: int = 1, custom_check=lambda x: True
):
    """
    Perform indexing on given peaks.
    :param peaks: peak info, including pos_x, pos_y, total_intensity, snr,
                  qx, qy, qz, resolution.
    :param transform_matrix: transform matrix A = [a*, b*, c*] in per angstrom.
    :return: indexing solutions.
    """
    solutions = []
    unindexed_peak_ids = set(range(len(peaks)))

    nb_failed = 0
    for i in range(p.nb_try if p.multi_index else 1):
        if nb_failed >= p.nb_failed:
            break
        solution = index_once(
            peaks,
            hklmatcher,
            p,
            unindexed_peak_ids=sorted(list(unindexed_peak_ids)),
            num_threads=num_threads,
        )
        if (solution is None) or (not custom_check(solution)):
            nb_failed += 1
            continue
        nb_failed = 0
        solutions.append(solution)
        unindexed_peak_ids = unindexed_peak_ids - set(solution.pair_ids)
    return solutions


@numba.jit(boundscheck=False, nogil=True, cache=True, nopython=True)
def eval_solution_kernel(
    hklss, matched_idx, eval_hkl_tol, ref_hkls, centering="P", centering_weight=0.0
):  # pragma: no cover
    nb_peaks = hklss.shape[1]
    max_total_error = -np.inf
    for hkl_idx in range(hklss.shape[0]):
        nb_pairs = 0
        total_error = 0
        rhkls = np.empty((nb_peaks, 3), np.float64)
        ehkls = np.empty((nb_peaks, 3), np.float64)
        pair_ids = np.zeros(nb_peaks, np.bool_)
        nb_X_peaks = np.zeros(4, np.uint32)  # X: A B C I for centering
        for q_idx in range(nb_peaks):
            hkl = hklss[hkl_idx, q_idx]
            max_ehkl = 0
            if matched_idx[hkl_idx, q_idx] == len(ref_hkls):
                rhkls[q_idx, :] = np.nan
                ehkls[q_idx, :] = np.nan
                continue
            rhkls[q_idx, :] = ref_hkls[matched_idx[hkl_idx, q_idx]]
            sum_ehkl = 0
            for i in range(3):
                e = np.abs(hkl[i] - rhkls[q_idx, i])
                ehkls[q_idx, i] = e
                sum_ehkl += e
                if e > max_ehkl:
                    max_ehkl = e
            if max_ehkl > eval_hkl_tol:  # Not paired
                continue
            total_error += sum_ehkl
            nb_pairs += 1
            pair_ids[q_idx] = True
            nb_X_peaks[0] += 1 - (rhkls[q_idx, 1] + rhkls[q_idx, 2]) % 2
            nb_X_peaks[1] += 1 - (rhkls[q_idx, 2] + rhkls[q_idx, 0]) % 2
            nb_X_peaks[2] += 1 - (rhkls[q_idx, 0] + rhkls[q_idx, 1]) % 2
            nb_X_peaks[3] += (
                1 - (rhkls[q_idx, 0] + rhkls[q_idx, 1] + rhkls[q_idx, 2]) % 2
            )

        total_error = total_error / (3 * nb_peaks) if nb_pairs else 1.0
        # print(total_error)
        match_rate = nb_pairs / nb_peaks
        if nb_pairs == 0:
            centering_score = 0.0
        elif centering == "A":
            centering_score = 2 * nb_X_peaks[0] / nb_pairs - 1
        elif centering == "B":
            centering_score = 2 * nb_X_peaks[1] / nb_pairs - 1
        elif centering == "C":
            centering_score = 2 * nb_X_peaks[2] / nb_pairs - 1
        elif centering == "I":
            centering_score = 2 * nb_X_peaks[3] / nb_pairs - 1
        elif centering == "P":
            centering_score = 2 * (1 - np.max(nb_X_peaks) / nb_pairs)
        total_score = centering_weight * centering_score + match_rate - total_error
        # total_score = -total_error
        if total_score > max_total_error:
            ans = (
                hkl_idx,
                nb_pairs,
                total_error,
                centering_score,
                total_score,
                match_rate,
                pair_ids,
                rhkls,
                ehkls,
            )
            max_total_error = total_score
    if max_total_error > -np.inf:
        return ans
    else:
        return None


def eval_best_solution(
    hklmatcher,
    Rs,
    seed_errors,
    hklss,
    eval_hkl_tol=0.25,
    centering="P",
    centering_weight=0.0,
):
    d, matched_idx = hklmatcher.hkl_tree.query(
        hklss.reshape(-1, 3), distance_upper_bound=eval_hkl_tol * 2
    )
    matched_idx = matched_idx.reshape(hklss.shape[:-1])
    best_solution_raw = eval_solution_kernel(
        hklss,
        matched_idx,
        eval_hkl_tol,
        ref_hkls=hklmatcher.hkls,
        centering=centering,
        centering_weight=centering_weight,
    )
    if best_solution_raw is None:
        best_solution = None
    else:
        (
            hkl_idx,
            nb_pairs,
            total_error,
            centering_score,
            total_score,
            match_rate,
            pair_ids,
            rhkls,
            ehkls,
        ) = best_solution_raw
        pair_ids = np.where(pair_ids)[0]
        best_solution = Solution(
            nb_peaks=hklss.shape[1],
            centering=centering,
            match_rate=match_rate,
            total_score=total_score,
            seed_error=seed_errors[hkl_idx],
            centering_score=centering_score,
            pair_ids=pair_ids,
            rotation_matrix=Rs[hkl_idx],
            hkls=hklss[hkl_idx],
            ehkls=ehkls,
            rhkls=rhkls,
        )
    return best_solution


def eval_rot(peaks, hklmatcher, rot, p: Params):
    hkls = peaks @ rot @ p.inv_transform_matrix.T
    sol = eval_best_solution(
        hklmatcher,
        rot.reshape(1, 3, 3),
        np.array([np.nan]),
        hkls.reshape(1, -1, 3),
        eval_hkl_tol=p.eval_hkl_tol,
        centering=p.centering,
        centering_weight=p.centering_weight,
    )

    eXYZs = np.abs(sol.rhkls @ sol.transform_matrix.T - peaks)
    dists = np.linalg.norm(eXYZs, axis=1)
    sol.pair_dist = dists[
        sol.pair_ids
    ].mean()  # average distance between matched peaks and the correspoding predicted spots
    return sol


def index_once(
    peaks,
    hklmatcher,
    p: Params,
    unindexed_peak_ids,
    num_threads,
):
    """
    Perform index once.
    :param peaks: peak info, including pos_x, pos_y, total_intensity, snr,
                  qx, qy, qz, resolution.
    :param table: reference table dict, including hkl pairs and length, angle.
    :param transform_matrix: transform matrix A = [a*, b*, c*] in per angstrom.
    :param inv_transform_matrix: inverse of transform matrix A: inv(A).
    :param seed_pool_size: size of seed pool.
    :param seed_len_tol: length tolerance for seed.
    :param seed_angle_tol: angle tolerance for seed.
    :param seed_hkl_tol: hkl tolerance for seed.
    :param eval_hkl_tol: hkl tolerance for paired peaks.
    :param centering: centering type.
    :param centering_weight: weight of centering score.
    :param refine_mode: minimization mode, global or alternative.
    :param refine_cycles: number of refine cycles.
    :param miller_set: possible hkl indices.
    :param nb_top: number of top solutions.
    :param unindexed_peak_ids: indices of unindexed peaks.
    :return: best indexing solution.
    """

    qs = peaks["coor"]  # in angstrom^-1

    time_stat = {
        "evaluation": -1.0,
        "evaluation individual": [],
        "correction": -1.0,
        "refinement": -1.0,
        "total": -1.0,
    }

    t_total0 = time.time()
    if p.sort_by == "random":
        seed_pool = np.random.choice(
            unindexed_peak_ids,
            min(p.seed_pool_size, len(unindexed_peak_ids)),
            replace=False,
        )
    else:
        if p.sort_by in ["intensity", "snr"]:
            seed_pool = unindexed_peak_ids[
                : min(p.seed_pool_size, len(unindexed_peak_ids))
            ]
            seed_pool = np.array(seed_pool, dtype=int)
            seed_pool = seed_pool[np.argsort(peaks[seed_pool][p.sort_by])][::-1]
        else:
            raise ValueError()

    if len(seed_pool) <= 1:
        return None
    good_solutions = []
    # collect good solutions
    t0 = time.time()

    def _thread_worker(seed_pair):
        t_iter = time.time()
        q1, q2 = qs[seed_pair, :]
        Rs, seed_errors = hklmatcher(q1, q2, p.inv_transform_matrix, p.seed_hkl_tol)
        t_eval = time.time() - t_iter
        hklss = np.einsum("mp,npq,iq->nmi", qs, Rs, p.inv_transform_matrix)
        solution = eval_best_solution(
            hklmatcher,
            Rs,
            seed_errors,
            hklss,
            eval_hkl_tol=p.eval_hkl_tol,
            centering=p.centering,
            centering_weight=p.centering_weight,
        )
        if solution is not None:
            solution.seed_pair = tuple(seed_pair)
        return (seed_pair, t_eval, Rs.shape[0]), solution

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        gs = executor.map(_thread_worker, combinations(seed_pool, 2))
    good_solutions = []
    for seed_pair, s in gs:
        time_stat["evaluation individual"].append(seed_pair)
        if s is not None:
            good_solutions.append(s)

    if len(good_solutions) == 0:
        return None

    best_solution = max(good_solutions, key=lambda x: x.total_score)
    time_stat["evaluation"] = time.time() - t0

    # refine best solution
    t0 = time.time()
    best_solution.transform_matrix = best_solution.rotation_matrix.dot(
        p.transform_matrix
    )

    # Fourier space error between peaks and predicted spots
    eXYZs = np.abs(best_solution.rhkls @ best_solution.transform_matrix.T - qs)
    dists = np.linalg.norm(eXYZs, axis=1)
    best_solution.pair_dist = dists[
        best_solution.pair_ids
    ].mean()  # average distance between matched peaks and the correspoding predicted spots

    best_solution = refine_solution(
        best_solution, qs, mode=p.refine_mode, nb_cycles=p.refine_cycles
    )
    time_stat["refinement"] = time.time() - t0
    time_stat["total"] = time.time() - t_total0
    best_solution.time_stat = time_stat
    return best_solution


def refine_solution(sol, qs, mode="global", nb_cycles=10):
    """
    Refine indexing solution.
    :param sol: indexing solution.
    :param qs: q vectors of peaks.
    :param mode: minimization mode, global or alternative.
    :param nb_cycles: number of refine cycles.
    :return: indexing solution with refined results.
    """
    transform_matrix_refined = sol.transform_matrix.copy()
    rhkls = sol.rhkls
    pair_ids = sol.pair_ids

    if mode == "alternative":

        def _func(x, *argv):  # objective function
            asx, bsx, csx, asy, bsy, csy, asz, bsz, csz = x
            h, k, l, qx, qy, qz = argv
            r1 = asx * h + bsx * k + csx * l - qx
            r2 = asy * h + bsy * k + csy * l - qy
            r3 = asz * h + bsz * k + csz * l - qz
            return r1**2.0 + r2**2.0 + r3**2.0

        def _grad(x, *argv):  # gradient function
            asx, bsx, csx, asy, bsy, csy, asz, bsz, csz = x
            h, k, l, qx, qy, qz = argv
            r1 = asx * h + bsx * k + csx * l - qx
            r2 = asy * h + bsy * k + csy * l - qy
            r3 = asz * h + bsz * k + csz * l - qz
            g_asx, g_bsx, g_csx = 2.0 * h * r1, 2.0 * k * r1, 2.0 * l * r1
            g_asy, g_bsy, g_csy = 2.0 * h * r2, 2.0 * k * r2, 2.0 * l * r2
            g_asz, g_bsz, g_csz = 2.0 * h * r3, 2.0 * k * r3, 2.0 * l * r3
            return np.array(
                [g_asx, g_bsx, g_csx, g_asy, g_bsy, g_csy, g_asz, g_bsz, g_csz]
            )

        # alternative refinement
        for i in range(nb_cycles):
            for j in range(len(pair_ids)):
                x0 = transform_matrix_refined.reshape(-1)
                rhkl = rhkls[pair_ids[j]]
                q = qs[pair_ids[j]]
                args = (rhkl[0], rhkl[1], rhkl[2], q[0], q[1], q[2])
                res = fmin_cg(_func, x0, fprime=_grad, args=args, disp=0)
                transform_matrix_refined = res.reshape(3, 3)
    elif mode == "global":
        _rhkls = rhkls[sol.pair_ids]
        _qs = qs[sol.pair_ids]

        def _func(x):
            _A = x.reshape(3, 3)
            return np.linalg.norm(_A.dot(_rhkls.T).T - _qs, axis=1).mean()

        # global refinement
        x0 = transform_matrix_refined.reshape(-1)
        res = minimize(_func, x0, method="CG", options={"disp": False})
        transform_matrix_refined = res.x.reshape(3, 3)
    elif mode == "none":
        pass
    else:
        raise ValueError("Not supported refine mode: %s" % mode)

    # refinement results
    eXYZs = np.abs(transform_matrix_refined.dot(rhkls.T) - qs.T).T
    pair_dist = np.linalg.norm(eXYZs, axis=1)[pair_ids].mean()

    if pair_dist > sol.pair_dist:  # keep original sol
        return sol
    sol_r = copy.deepcopy(sol)
    sol_r.transform_matrix = transform_matrix_refined
    sol_r.pair_dist = pair_dist
    return sol_r


def calc_centering_score(pair_hkls, centering="P"):
    """
    Calculate centering score.
    :param pair_hkls: hkl indices of pair peaks, Nx3 array.
    :param centering: centering type.
    :return: centering score.
    """
    nb_pairs = len(pair_hkls)
    if nb_pairs == 0:
        centering_score = 0.0
    elif centering == "A":  # k+l=2n
        nb_A_peaks = ((pair_hkls[:, 1] + pair_hkls[:, 2]) % 2 == 0).sum()
        A_ratio = float(nb_A_peaks) / float(nb_pairs)
        centering_score = 2 * A_ratio - 1.0  # range from 0-1
    elif centering == "B":  # h+l=2n
        nb_B_peaks = ((pair_hkls[:, 0] + pair_hkls[:, 2]) % 2 == 0).sum()
        B_ratio = float(nb_B_peaks) / float(nb_pairs)
        centering_score = 2 * B_ratio - 1.0  # range from 0-1
    elif centering == "C":  # h+k=2n
        nb_C_peaks = ((pair_hkls[:, 0] + pair_hkls[:, 1]) % 2 == 0).sum()
        C_ratio = float(nb_C_peaks) / float(nb_pairs)
        centering_score = 2 * C_ratio - 1.0  # range from 0-1
    elif centering == "I":  # h+k+l=2n
        nb_I_peaks = (np.sum(pair_hkls, axis=1) % 2 == 0).sum()
        I_ratio = float(nb_I_peaks) / float(nb_pairs)
        centering_score = 2 * I_ratio - 1.0  # range from 0-1
    elif centering == "P":
        nb_A_peaks = ((pair_hkls[:, 1] + pair_hkls[:, 2]) % 2 == 0).sum()
        nb_B_peaks = ((pair_hkls[:, 0] + pair_hkls[:, 2]) % 2 == 0).sum()
        nb_C_peaks = ((pair_hkls[:, 0] + pair_hkls[:, 1]) % 2 == 0).sum()
        nb_I_peaks = (np.sum(pair_hkls, axis=1) % 2 == 0).sum()
        centering_score = 2.0 * (
            1
            - float(max(nb_A_peaks, nb_B_peaks, nb_C_peaks, nb_I_peaks))
            / float(nb_pairs)
        )
    else:
        raise ValueError("Not supported centering type: %s" % centering)
    return centering_score
