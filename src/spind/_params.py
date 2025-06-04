from collections import namedtuple
from pathlib import Path

import numpy as np
import yaml

from ._utils import calc_transform_matrix

Params = namedtuple(
    "Params",
    [
        "res_cutoff",
        "lattice_constants",
        "centering",
        "sort_by",
        "seed_pool_size",
        "seed_len_tol",
        "seed_hkl_tol",
        "eval_hkl_tol",
        "centering_weight",
        "refine_mode",
        "refine_cycles",
        "nb_top",
        "multi_index",
        "nb_try",
        "nb_failed",
        "transform_matrix",
        "inv_transform_matrix",
        "miller_set",
    ],
)

"""
    :param seed_pool_size: size of seed pool.
    :param seed_len_tol: length tolerance for seed.
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
"""


def params(config):
    # with open(config_file) as fp:
    if isinstance(config, (str, Path)):
        with open(config) as fp:
            config = yaml.safe_load(fp)
    elif isinstance(config, dict):
        pass
    else:
        raise ValueError()

    transform_matrix = calc_transform_matrix(config["cell parameters"])
    inv_transform_matrix = np.linalg.inv(transform_matrix)

    return Params(
        res_cutoff=config["resolution cutoff"],
        lattice_constants=config["cell parameters"],
        centering=config["centering"],
        sort_by=config["sort by"],
        seed_pool_size=config["seed pool size"],
        seed_len_tol=config["seed length tolerance"],
        seed_hkl_tol=config["seed hkl tolerance"],
        eval_hkl_tol=config["eval tolerance"],
        centering_weight=config["centering factor"],
        refine_mode=config["refine mode"],
        refine_cycles=int(config["refine cycles"]),
        nb_top=int(config["top solutions"]),
        multi_index=config["multi index"],
        nb_try=int(config["number of try"]),
        nb_failed=int(config["number of failed try"]),
        transform_matrix=transform_matrix,
        inv_transform_matrix=inv_transform_matrix,
        miller_set=config["miller set"],
    )
