#!/usr/bin/env python
from __future__ import print_function
from six import print_ as print

from itertools import combinations

import numpy as np
from numpy.linalg import norm
from scipy.optimize import fmin_cg, minimize
from math import pi, cos, sin
import time

from util import calc_angle


class Solution(object):
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

    def simplify(self):
        """
        Extract main info only.
        :return: simple solution with main info.
        """
        attrs = ['nb_peaks',
                 'total_score',
                 'match_rate',
                 'pair_ids',
                 'true_total_score',
                 'true_match_rate',
                 'true_pair_ids',
                 'transform_matrix',
                 'pair_dist',
                 'transform_matrix_refined',
                 'pair_dist_refined']
        simple_solution = Solution()
        for attr in attrs:
            setattr(simple_solution, attr, getattr(self, attr))
        return simple_solution

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


def index(peaks, table, transform_matrix, inv_transform_matrix,
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
          multi_index=False,
          nb_try=10,
          verbose=False):
    """
    Perform indexing on given peaks.
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
    :param multi_index: enable multiple index.
    :param nb_try: max try for multiple index.
    :return: indexing solutions.
    """
    solutions = []
    unindexed_peak_ids = list(range(len(peaks)))

    if multi_index:
        for i in range(nb_try):
            solution = index_once(peaks, table,
                                  transform_matrix, inv_transform_matrix,
                                  seed_pool_size=seed_pool_size,
                                  seed_len_tol=seed_len_tol,
                                  seed_angle_tol=seed_angle_tol,
                                  seed_hkl_tol=seed_hkl_tol,
                                  eval_hkl_tol=eval_hkl_tol,
                                  centering=centering,
                                  centering_weight=centering_weight,
                                  refine_mode=refine_mode,
                                  refine_cycles=refine_cycles,
                                  miller_set=miller_set,
                                  nb_top=nb_top,
                                  unindexed_peak_ids=unindexed_peak_ids)
            if solution.total_score == 0.:
                break
            solutions.append(solution)
            unindexed_peak_ids = list(
                set(unindexed_peak_ids) - set(solution.pair_ids)
            )
            unindexed_peak_ids.sort()
    else:
        solution = index_once(peaks, table,
                              transform_matrix, inv_transform_matrix,
                              seed_pool_size=seed_pool_size,
                              seed_len_tol=seed_len_tol,
                              seed_angle_tol=seed_angle_tol,
                              seed_hkl_tol=seed_hkl_tol,
                              eval_hkl_tol=eval_hkl_tol,
                              centering=centering,
                              centering_weight=centering_weight,
                              refine_mode=refine_mode,
                              refine_cycles=refine_cycles,
                              miller_set=miller_set,
                              nb_top=nb_top,
                              unindexed_peak_ids=unindexed_peak_ids)
        if solution.total_score > 0.:
            solutions.append(solution)
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
            print('%-20s: %-3f sec'
                  % ('t1/table lookup', solution.time_stat['table lookup']))
            print('%-20s: %-3f sec'
                  % ('t2/evaluation', solution.time_stat['evaluation']))
            print('%-20s: %-3f sec'
                  % ('t3/correction', solution.time_stat['correction']))
            print('%-20s: %-3f sec'
                  % ('t4/refinement', solution.time_stat['refinement']))
            print('%-20s: %-3f sec'
                  % ('t0/total', solution.time_stat['total']))
        print('=' * 50)

        solutions_.append(solution.simplify())
    return solutions_


def index_once(peaks, table, transform_matrix, inv_transform_matrix,
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
               unindexed_peak_ids=None):
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
    time_stat = {
        'table lookup': 0.,
        'evaluation': 0.,
        'correction': 0.,
        'refinement': 0.,
        'total': 0.,
    }
    time_start = time.time()
    qs = peaks[:, 4:7] * 1.E-10  # convert to per angstrom
    if unindexed_peak_ids is None:
        unindexed_peak_ids = list(range(len(peaks)))
    seed_pool = unindexed_peak_ids[:min(
        seed_pool_size, len(unindexed_peak_ids))]
    seed_pairs = list(combinations(seed_pool, 2))
    solutions = []
    nb_candidates = 0
    seed_len_tol = np.float32(seed_len_tol)
    seed_angle_tol = np.float32(seed_angle_tol)
    for seed_pair in seed_pairs:
        q1, q2 = qs[seed_pair, :]
        q1_len, q2_len = np.float32(norm(q1)), np.float32(norm(q2))
        if q1_len < q2_len:
            q1, q2 = q2, q1
            q1_len, q2_len = q2_len, q1_len
        angle = np.float32(calc_angle(q1, q2, q1_len, q2_len))
        t0 = time.time()
        candidates = np.where(
            (np.abs(q1_len - table['len_angle'][:, 0]) < seed_len_tol)
            * (np.abs(q2_len - table['len_angle'][:, 1]) < seed_len_tol)
            * (np.abs(angle - table['len_angle'][:, 2]) < seed_angle_tol)
        )[0]
        t1 = time.time()
        time_stat['table lookup'] += (t1 - t0)
        nb_candidates += len(candidates)

        t0 = time.time()
        for candidate in candidates:
            hkl1 = table['hkl1'][candidate]
            hkl2 = table['hkl2'][candidate]
            ref_q1 = transform_matrix.dot(hkl1)
            ref_q2 = transform_matrix.dot(hkl2)
            solution = Solution(nb_peaks=len(qs), centering=centering)
            solution.rotation_matrix = calc_rotation_matrix(
                q1, q2, ref_q1, ref_q2)
            solution.transform_matrix = solution.rotation_matrix.dot(
                transform_matrix)
            eval_solution(solution, qs, inv_transform_matrix, seed_pair,
                          eval_hkl_tol=eval_hkl_tol,
                          centering=centering,
                          centering_weight=centering_weight,
                          unindexed_peak_ids=unindexed_peak_ids)
            if solution.seed_error < seed_hkl_tol:
                solutions.append(solution)
        t1 = time.time()
        time_stat['evaluation'] += (t1 - t0)
    if len(solutions) > 0:
        solutions.sort(key=lambda x: x.total_score, reverse=True)
        total_scores = [solution.total_score for solution in solutions]
        if nb_top < 0:
            nb_top = len(solutions)
        else:
            nb_top = min(len(solutions), nb_top)
        top_solutions = solutions[:nb_top]
        # correct match rate for top solutions using miller set constraints
        t0 = time.time()
        for solution in top_solutions:
            solution.correct_match_rate(
                qs, miller_set=miller_set, centering_weight=centering_weight)
        top_solutions.sort(
            key=lambda x: (x.true_total_score, x.true_pair_dist))
        t1 = time.time()
        time_stat['correction'] = (t1 - t0)

        best_solution = solutions[0]
        best_solution.nb_candidates = nb_candidates
        best_solution.total_scores = total_scores

        # refine best solution
        t0 = time.time()
        best_solution.transform_matrix = best_solution.rotation_matrix.dot(
            transform_matrix)
        eXYZs = np.abs(best_solution.transform_matrix.dot(
            best_solution.rhkls.T) - qs.T).T
        dists = norm(eXYZs, axis=1)
        best_solution.pair_dist = dists[best_solution.pair_ids].mean()
        refine_solution(best_solution, qs,
                        mode=refine_mode,
                        nb_cycles=refine_cycles)
        t1 = time.time()
        time_stat['refinement'] = t1 - t0
    else:
        best_solution = Solution()
        best_solution.rotation_matrix = np.identity(3)
    time_stop = time.time()
    time_stat['total'] = time_stop - time_start

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
    solution.pair_ids = pair_ids
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
