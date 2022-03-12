#!/usr/bin/env python

"""
Usage:
    SPIND.py <config.yml> [options]

Options:
    -h --help                                       Show this screen.
    --peak-list-dir=peak_list_dir                   Peak list directory.
    --output-dir=output_dir                         Output directory.
"""

import numpy as np

PEAKS_DTYPE = np.dtype(
    [
        ("coor", np.float64, 3),
        ("snr", np.float64),
        ("intensity", np.float64),
        ("resolution", np.float64),
    ]
)


def presort_peaks(peaks, sort_by, res_cutoff):
    if sort_by == "snr":
        ind = np.argsort(peaks["snr"])
    elif sort_by == "intensity":
        ind = np.argsort(peaks["intensity"])
    else:
        raise ValueError('Please use "intensity" or "snr" sorting method!')
    peaks = peaks[ind[::-1]]  # reverse sort
    HP_ind = peaks["resolution"] > res_cutoff
    LP_ind = peaks["resolution"] <= res_cutoff
    peaks = np.concatenate((peaks[HP_ind], peaks[LP_ind]))
    return peaks


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
    A = np.empty((3, 3), dtype=np.float64)  # transform matrix
    A[:, 0] = a_star
    A[:, 1] = b_star
    A[:, 2] = c_star
    return A
