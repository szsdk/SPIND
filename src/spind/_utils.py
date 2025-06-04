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


_eijk = np.array(
    [
        [[0, 0, 0], [0, 0, 1], [0, -1, 0]],
        [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
        [[0, 1, 0], [-1, 0, 0], [0, 0, 0]],
    ]
)


def calc_transform_matrix(cell_param):
    if len(cell_param) == 6:
        a, b, c, al, be, ga = cell_param
        if not np.isclose(al, 90) or (not np.isclose(ga, 90)):
            raise NotImplementedError(
                "For cell_param having six elements, only monoclinic or orthorhombic crystal is supported."
            )
        A = np.array(
            [
                [a, 0.0, 0.0],
                [0, b, 0.0],
                [c * np.cos(np.deg2rad(be)), 0, c * np.sin(np.deg2rad(be))],
            ]
        )
    elif len(cell_param) == 3:
        A = np.array(cell_param, np.float64)
    else:
        raise ValueError()

    star = np.einsum("lmn,ijk,mj,nk->li", _eijk, _eijk, A, A)
    star /= np.einsum("ij,ij->i", star, A)
    return star.T
