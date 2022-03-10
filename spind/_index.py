#!/usr/bin/env python

from itertools import combinations

import numpy as np
from numpy.linalg import norm
import numba
from scipy.optimize import fmin_cg, minimize
from math import pi, cos, sin
import time

from concurrent.futures import ThreadPoolExecutor
from ._utils import calc_angle, calc_transform_matrix
from ._params import Params


class Solution:
    """
    Indexing solution.
    """
    def __init__(self, nb_peaks=0, centering='P'):
        self.nb_peaks = nb_peaks
        self.centering=centering
        # raw results
        self.pair_ids = None
        self.match_rate = 0.
        self.total_score = 0.
        self.transform_matrix = None
        # corrected results
        self.true_pair_ids = None
        self.true_pair_dist = np.inf
        self.true_match_rate = 0.
        self.true_total_score = 0.
        # refined results
        self.transform_matrix_refined = None
        self.pair_dist_refined = np.inf
        self.time_stat = dict()
        self.rotation_matrix = None

    def correct_match_rate(self, qs, miller_set=None, centering_weight=0.):
        """
        Correct match rate and pair peaks with miller set constraint.
        :param qs: peak coordinates in fourier space, Nx3 array.
        :param miller_set: possible hkl indices, Nx3 array.
        :param centering_weight: weight of centering score.
        :return: None
        """
        dtype = 'int8'
        if miller_set is None:  # dummy case
            self.true_match_rate = self.match_rate
            self.true_pair_ids = self.pair_ids
            eXYZs = np.abs(self.transform_matrix.dot(self.rhkls.T) - qs.T).T
            self.true_pair_dist = norm(eXYZs, axis=1)[self.true_pair_ids].mean()
            self.true_total_score = self.total_score
        else:
            miller_set = miller_set.astype(dtype)
            true_pair_ids = []
            for pair_id in self.pair_ids:
                if norm(miller_set - self.rhkls[pair_id].astype(dtype),
                        axis=1).min() == 0:
                    true_pair_ids.append(pair_id)
            true_match_rate = float(len(true_pair_ids)) / self.nb_peaks
            eXYZs = np.abs(self.transform_matrix.dot(self.rhkls.T) - qs.T).T
            true_pair_dist = norm(eXYZs, axis=1)[true_pair_ids].mean()
            true_pair_hkls = self.rhkls[true_pair_ids].astype(dtype)
            centering_score = calc_centering_score(
                true_pair_hkls, centering=self.centering)
            true_total_score = centering_score * centering_weight \
                               + true_match_rate
            self.true_pair_ids = true_pair_ids
            self.true_match_rate = true_match_rate
            self.true_pair_dist = true_pair_dist
            self.true_total_score = true_total_score


def rad2deg(rad):
    return float(rad) / pi * 180.


def deg2rad(deg):
    return float(deg) / 180. * pi


def index(peaks, table, p: Params, num_threads: int=1, verbose=False):
    """
    Perform indexing on given peaks.
    :param peaks: peak info, including pos_x, pos_y, total_intensity, snr,
                  qx, qy, qz, resolution.
    :param transform_matrix: transform matrix A = [a*, b*, c*] in per angstrom.
    :return: indexing solutions.
    """
    solutions = []
    unindexed_peak_ids = list(range(len(peaks)))

    # transform_matrix = calc_transform_matrix(p.lattice_constants, p.lattice_type)
    # inv_transform_matrix = np.linalg.inv(transform_matrix)

    for i in range(p.nb_try if p.multi_index else 1):
        solution = index_once(peaks, table,
                              p.transform_matrix, p.inv_transform_matrix,
                              seed_pool_size=p.seed_pool_size,
                              seed_len_tol=p.seed_len_tol,
                              seed_angle_tol=p.seed_angle_tol,
                              seed_hkl_tol=p.seed_hkl_tol,
                              eval_hkl_tol=p.eval_hkl_tol,
                              centering=p.centering,
                              centering_weight=p.centering_weight,
                              refine_mode=p.refine_mode,
                              refine_cycles=p.refine_cycles,
                              miller_set=p.miller_set,
                              nb_top=p.nb_top,
                              unindexed_peak_ids=unindexed_peak_ids,
                              num_threads=num_threads)
        if solution.total_score == 0.:
            continue
        solutions.append(solution)
        unindexed_peak_ids = list(
            set(unindexed_peak_ids) - set(solution.pair_ids)
        )
        unindexed_peak_ids.sort()
    solutions_ = []
    for solution in solutions:
        print('%-20s: %-d' % ('peak num', solution.nb_peaks))
        print('%-20s: %-3f' % ('total score', solution.total_score))
        if verbose:
            print('%-20s: %-3f'
                  % ('match rate', solution.match_rate))
            print('%-20s: %-3d'
                  % ('matched peak num', len(solution.pair_ids)))
            print('%-20s: %-3e'
                  % ('matched dist', solution.pair_dist_refined))
            print('%-20s: %-3f'
                  % ('centering score', solution.centering_score))
            print(solution.time_stat)
            print(sum([i for _, i, _ in solution.time_stat['evaluation individual']]))
            # print('%-20s: %-3f sec'
                  # % ('t1/table lookup', solution.time_stat['table lookup']))
            # print('%-20s: %-3f sec'
                  # % ('t2/evaluation', solution.time_stat['evaluation']))
            # print('%-20s: %-3f sec'
                  # % ('t3/correction', solution.time_stat['correction']))
            # print('%-20s: %-3f sec'
                  # % ('t4/refinement', solution.time_stat['refinement']))
            # print('%-20s: %-3f sec'
                  # % ('t0/total', solution.time_stat['total']))
        print('=' * 50)

        solutions_.append(solution)
    return solutions_


@numba.jit(boundscheck=False, nogil=True, cache=True)
def eval_solution_kernel(hklss, eval_hkl_tol=0.25, centering="P", centering_weight=0.0):
    nb_peaks = hklss.shape[1]
    max_total_error = 0.0
    for hkl_idx in range(hklss.shape[0]):
        nb_pairs = 0
        total_error = 0
        rhkls = np.empty((nb_peaks, 3), np.int32)
        ehkls = np.empty((nb_peaks, 3), np.float64)
        pair_ids = np.zeros(nb_peaks, np.bool_)
        nb_X_peaks = np.zeros(4, np.uint32)  # X: A B C I for centering
        for q_idx in range(nb_peaks):
            hkl = hklss[hkl_idx, q_idx]
            max_ehkl = 0
            sum_ehkl = 0
            for i in range(3):
                rhkls[q_idx, i] = round(hkl[i])
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
        total_score = centering_weight * centering_score + match_rate
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
    if max_total_error > 0.0:
        return ans
    else:
        return None


def eval_best_solution(
    Rs, seed_errors, hklss, eval_hkl_tol=0.25, centering="P", centering_weight=0.0
):
    best_solution_raw = eval_solution_kernel(
        hklss, eval_hkl_tol=eval_hkl_tol, centering=centering, centering_weight=centering_weight
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
        best_solution = Solution()
        best_solution.rotation_matrix = Rs[hkl_idx]
        best_solution.nb_pairs = nb_pairs
        best_solution.match_rate = match_rate
        best_solution.seed_error = seed_errors[hkl_idx]
        best_solution.centering_score = centering_score
        best_solution.total_score = total_score
        best_solution.pair_ids = np.where(pair_ids)[0]
        best_solution.rhkls = rhkls
        best_solution.ehkls = ehkls
        best_solution.hkls = hklss[hkl_idx]
        best_solution.nb_peaks = hklss.shape[1]

    return best_solution


def index_once(peaks, hklmatcher, transform_matrix, inv_transform_matrix,
               seed_pool_size=5,
               seed_len_tol=0.003,
               seed_angle_tol=3.0,
               seed_hkl_tol=0.1,
               eval_hkl_tol=0.25,
               centering='P',
               centering_weight=0.,
               refine_mode='global',
               refine_cycles=10,
               miller_set=None,
               nb_top=5,
               unindexed_peak_ids=None,
               num_threads=1):
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

    qs = peaks['coor'] * 1e-10

    time_stat = {
        'evaluation': -1.,
        'evaluation individual': [],
        'correction': -1.,
        'refinement': -1.,
        'total': -1.,
    }

    t_total0 = time.time()
    if unindexed_peak_ids is None:
        unindexed_peak_ids = list(range(len(peaks)))
    seed_pool = unindexed_peak_ids[:min(seed_pool_size, len(unindexed_peak_ids))]
    good_solutions = []
    # collect good solutions
    t0 = time.time()
    def _thread_worker(seed_pair):
        t_iter = time.time()
        q1, q2 = qs[seed_pair, :]
        Rs, seed_errors = hklmatcher(q1, q2, inv_transform_matrix, seed_hkl_tol)
        t_eval = time.time() - t_iter
        hklss = np.einsum("mp,npq,iq->nmi", qs, Rs, inv_transform_matrix)
        solution = eval_best_solution(
            Rs,
            seed_errors,
            hklss,
            eval_hkl_tol=eval_hkl_tol,
            centering=centering,
            centering_weight=centering_weight,
        )
        return (seed_pair, t_eval, Rs.shape[0]), solution
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        gs = executor.map(_thread_worker, combinations(seed_pool, 2))
    good_solutions = []
    for seed_pair, s in gs:
        time_stat['evaluation individual'].append((seed_pair))
        if s is not None:
            good_solutions.append(s)

    if len(good_solutions) > 0:
        best_solution = max(good_solutions, key=lambda x: x.total_score)
    else:
        best_solution = None
    time_stat['evaluation'] = time.time() - t0

    # refine best solution if exists
    if best_solution is None:
        dummy_solution = Solution()
        dummy_solution.R = np.identity(3)
        dummy_solution.match_rate = 0.0
        return dummy_solution
    else:
        # refine best solution
        t0 = time.time()
        best_solution.transform_matrix = best_solution.rotation_matrix.dot(transform_matrix)
        eXYZs = np.abs(
            best_solution.transform_matrix.dot(best_solution.rhkls.T) - qs.T
        ).T  # Fourier space error between peaks and predicted spots
        dists = norm(eXYZs, axis=1)
        best_solution.pair_dist = dists[best_solution.pair_ids].mean()  # average distance between matched peaks and the correspoding predicted spots

        refine_solution(best_solution, qs,
                        mode=refine_mode,
                        nb_cycles=refine_cycles)
        time_stat['refinement'] = time.time() - t0
        time_stat['total'] = time.time() - t_total0
        best_solution.time_stat = time_stat
        return best_solution


def refine_solution(solution, qs, mode='global', nb_cycles=10):
    """
    Refine indexing solution.
    :param solution: indexing solution.
    :param qs: q vectors of peaks.
    :param mode: minimization mode, global or alternative.
    :param nb_cycles: number of refine cycles.
    :return: indexing solution with refined results.
    """
    transform_matrix_refined = solution.transform_matrix.copy()
    rhkls = solution.rhkls
    pair_ids = solution.pair_ids

    if mode == 'alternative':
        def _func(x, *argv):  # objective function
            asx, bsx, csx, asy, bsy, csy, asz, bsz, csz = x
            h, k, l, qx, qy, qz = argv
            r1 = (asx * h + bsx * k + csx * l - qx)
            r2 = (asy * h + bsy * k + csy * l - qy)
            r3 = (asz * h + bsz * k + csz * l - qz)
            return r1 ** 2. + r2 ** 2. + r3 ** 2.

        def _grad(x, *argv):  # gradient function
            asx, bsx, csx, asy, bsy, csy, asz, bsz, csz = x
            h, k, l, qx, qy, qz = argv
            r1 = (asx * h + bsx * k + csx * l - qx)
            r2 = (asy * h + bsy * k + csy * l - qy)
            r3 = (asz * h + bsz * k + csz * l - qz)
            g_asx, g_bsx, g_csx = 2. * h * r1, 2. * k * r1, 2. * l * r1
            g_asy, g_bsy, g_csy = 2. * h * r2, 2. * k * r2, 2. * l * r2
            g_asz, g_bsz, g_csz = 2. * h * r3, 2. * k * r3, 2. * l * r3
            return np.array([g_asx, g_bsx, g_csx,
                             g_asy, g_bsy, g_csy,
                             g_asz, g_bsz, g_csz])

        # alternative refinement
        for i in range(nb_cycles):
            for j in range(len(pair_ids)):
                x0 = transform_matrix_refined.reshape(-1)
                rhkl = rhkls[pair_ids[j]]
                q = qs[pair_ids[j]]
                args = (rhkl[0], rhkl[1], rhkl[2], q[0], q[1], q[2])
                try:
                    res = fmin_cg(_func, x0, fprime=_grad, args=args, disp=0)
                    transform_matrix_refined = res.reshape(3, 3)
                except:
                    pass
    elif mode == 'global':
        _rhkls = rhkls[solution.pair_ids]
        _qs = qs[solution.pair_ids]

        def _func(x):
            asx, bsx, csx, asy, bsy, csy, asz, bsz, csz = x
            _A = np.array([
                [asx, bsx, csx],
                [asy, bsy, csy],
                [asz, bsz, csz]
            ])
            return np.linalg.norm(_A.dot(_rhkls.T).T - _qs, axis=1).mean()

        # global refinement
        x0 = transform_matrix_refined.reshape(-1)
        try:
            res = minimize(_func, x0, method='CG', options={'disp': False})
            transform_matrix_refined = res.x.reshape(3, 3)
        except:
            pass
    else:
        raise ValueError('Not supported refine mode: %s' % mode)

    # refinement results
    eXYZs = np.abs(transform_matrix_refined.dot(rhkls.T) - qs.T).T
    pair_dist = norm(eXYZs, axis=1)[pair_ids].mean()

    if pair_dist > solution.pair_dist:  # keep original solution
        solution.transform_matrix_refined = solution.transform_matrix
        solution.pair_dist_refined = solution.pair_dist
    else:
        solution.transform_matrix_refined = transform_matrix_refined
        solution.pair_dist_refined = pair_dist
    return


def eval_solution(solution, qs, inv_transform_matrix, seed_pair,
                  eval_hkl_tol=0.25,
                  centering='P',
                  centering_weight=0.,
                  unindexed_peak_ids=None):
    """
    Evaluate indexing solution.
    :param solution: indexing solution.
    :param qs: q vectors of peaks.
    :param inv_transform_matrix: inverse of transform matrix A: inv(A).
    :param seed_pair: 2 indices of seed pair.
    :param eval_hkl_tol: hkl tolerance for paired peaks.
    :param centering: centering type.
    :param centering_weight: weight of centering score.
    :param unindexed_peak_ids: indices of unindexed peaks.
    :return:
    """
    # calculate hkl indices
    R = solution.rotation_matrix
    R_inv = np.linalg.inv(R)
    hkls = inv_transform_matrix.dot(R_inv.dot(qs.T)).T
    rhkls = np.round(hkls)
    ehkls = np.abs(hkls - rhkls)
    solution.hkls = hkls
    solution.rhkls = rhkls
    solution.ehkls = ehkls
    # find indices of pair peaks
    pair_ids = np.where(np.max(ehkls, axis=1) < eval_hkl_tol)[0].tolist()
    pair_ids = list(set(pair_ids) & (set(unindexed_peak_ids)))
    nb_pairs = len(pair_ids)
    match_rate = float(nb_pairs) / float(solution.nb_peaks)
    solution.pair_ids = np.where(pair_ids)[0]
    solution.nb_pairs = nb_pairs
    solution.match_rate = match_rate
    # centering score
    pair_hkls = rhkls.astype(int)[pair_ids]
    centering_score = calc_centering_score(pair_hkls, centering=centering)
    solution.centering_score = centering_score
    # error / score
    solution.seed_error = ehkls[seed_pair, :].max()
    solution.total_score = centering_score * centering_weight + match_rate
    return


def calc_rotation_matrix(mob_v1, mob_v2, ref_v1, ref_v2):
    """
    Calculate rotation matrix R, thus R.dot(ref_vx) ~ mob_vx.
    :param mob_v1: first mobile vector.
    :param mob_v2: second mobile vector.
    :param ref_v1: first reference vector.
    :param ref_v2: second reference vector.
    :return: rotation matrix R.
    """
    mob_v1, mob_v2 = np.float32(mob_v1), np.float32(mob_v2)
    ref_v1, ref_v2 = np.float32(ref_v1), np.float32(ref_v2)
    # rotate reference vector plane to  mobile plane
    mob_norm = np.cross(mob_v1, mob_v2)  # norm vector of mobile vectors
    ref_norm = np.cross(ref_v1, ref_v2)  # norm vector of reference vectors
    if min(norm(mob_norm), norm(ref_norm)) == 0.:
        return np.identity(3)  # return dummy matrix if co-linear
    axis = np.cross(ref_norm, mob_norm)
    angle = calc_angle(ref_norm, mob_norm, norm(ref_norm), norm(mob_norm))
    R1 = axis_angle_to_rotation_matrix(axis, angle)
    rot_ref_v1, rot_ref_v2 = R1.dot(ref_v1), R1.dot(ref_v2)
    # rotate reference vectors to mobile vectors approximately
    angle1 = calc_angle(rot_ref_v1, mob_v1, norm(rot_ref_v1), norm(mob_v1))
    angle2 = calc_angle(rot_ref_v2, mob_v2, norm(rot_ref_v2), norm(mob_v2))
    angle = (angle1 + angle2) * 0.5
    axis = np.cross(rot_ref_v1, mob_v1)  # important!!
    R2 = axis_angle_to_rotation_matrix(axis, angle)
    R = R2.dot(R1)
    return R


def axis_angle_to_rotation_matrix(axis, angle):
    """
    Calculate rotation matrix from axis/angle form.
    :param axis: axis vector with 3 elements.
    :param angle: angle in degree.
    :return: rotation matrix R.
    """
    x, y, z = axis / norm(axis)
    angle = deg2rad(angle)
    c, s = cos(angle), sin(angle)
    R = [[c+x**2.*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
         [y*x*(1-c)+z*s, c+y**2.*(1-c), y*z*(1-c)-x*s],
         [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z**2.*(1-c)]]
    return np.array(R)


def calc_centering_score(pair_hkls, centering='P'):
    """
    Calculate centering score.
    :param pair_hkls: hkl indices of pair peaks, Nx3 array.
    :param centering: centering type.
    :return: centering score.
    """
    nb_pairs = len(pair_hkls)
    if nb_pairs == 0:
        centering_score = 0.
    elif centering == 'A':  # k+l=2n
        nb_A_peaks = ((pair_hkls[:, 1] + pair_hkls[:, 2]) % 2 == 0).sum()
        A_ratio = float(nb_A_peaks) / float(nb_pairs)
        centering_score = 2 * A_ratio - 1.  # range from 0-1
    elif centering == 'B':  # h+l=2n
        nb_B_peaks = ((pair_hkls[:, 0] + pair_hkls[:, 2]) % 2 == 0).sum()
        B_ratio = float(nb_B_peaks) / float(nb_pairs)
        centering_score = 2 * B_ratio - 1.  # range from 0-1
    elif centering == 'C':  # h+k=2n
        nb_C_peaks = ((pair_hkls[:, 0] + pair_hkls[:, 1]) % 2 == 0).sum()
        C_ratio = float(nb_C_peaks) / float(nb_pairs)
        centering_score = 2 * C_ratio - 1.  # range from 0-1
    elif centering == 'I':  # h+k+l=2n
        nb_I_peaks = (np.sum(pair_hkls, axis=1) % 2 == 0).sum()
        I_ratio = float(nb_I_peaks) / float(nb_pairs)
        centering_score = 2 * I_ratio - 1.  # range from 0-1
    elif centering == 'P':
        nb_A_peaks = ((pair_hkls[:, 1] + pair_hkls[:, 2]) % 2 == 0).sum()
        nb_B_peaks = ((pair_hkls[:, 0] + pair_hkls[:, 2]) % 2 == 0).sum()
        nb_C_peaks = ((pair_hkls[:, 0] + pair_hkls[:, 1]) % 2 == 0).sum()
        nb_I_peaks = (np.sum(pair_hkls, axis=1) % 2 == 0).sum()
        centering_score = 2. * (1 - float(
            max(nb_A_peaks, nb_B_peaks, nb_C_peaks, nb_I_peaks)
        ) / float(nb_pairs))
    else:
        raise ValueError('Not supported centering type: %s' % centering)
    return centering_score
