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
        "lattice_type",
        "centering",
        "seed_pool_size",
        "seed_len_tol",
        "seed_angle_tol",
        "seed_hkl_tol",
        "eval_hkl_tol",
        "centering_weight",
        "refine_mode",
        "refine_cycles",
        "miller_file",
        "nb_top",
        "multi_index",
        "nb_try",
        "transform_matrix",
        "inv_transform_matrix",
        "miller_set",
    ],
)

"""
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
"""


def param_from_args(args):
    return Params(
        min_res=float(args["--min-resolution"]),
        lattice_constants=eval(args["--lattice-constants"]),
        lattice_type=args["--lattice-type"],
        centering=args["--centering"],
        seed_pool_size=int(args["--seed-pool-size"]),
        seed_len_tol=float(args["--seed-len-tol"]),
        seed_angle_tol=float(args["--seed-angle-tol"]),
        seed_hkl_tol=float(args["--seed-hkl-tol"]),
        eval_hkl_tol=float(args["--eval-hkl-tol"]),
        centering_weight=float(args["--centering-weight"]),
        refine_mode=args["--refine-mode"],
        refine_cycles=int(args["--refine-cycles"]),
        miller_file=args["--miller-file"],
        nb_top=int(args["--top-solutions"]),
        verbose=args["--verbose"],
        multi_index=args["--multi-index"],
        nb_try=int(args["--nb-try"]),
    )


def params(config):
    # with open(config_file) as fp:
    if isinstance(config, (str, Path)):
        with open(config) as fp:
            config = yaml.safe_load(fp)
    elif isinstance(config, dict):
        pass
    else:
        raise ValueError()

    transform_matrix = calc_transform_matrix(
        config["cell parameters"], config["lattice type"]
    )
    inv_transform_matrix = np.linalg.inv(transform_matrix)

    return Params(
        res_cutoff=config["resolution cutoff"],
        lattice_constants=config["cell parameters"],
        lattice_type=config["lattice type"],
        centering=config["centering"],
        seed_pool_size=config["seed pool size"],
        seed_len_tol=config["seed length tolerance"],
        seed_angle_tol=config["seed angle tolerance"],
        seed_hkl_tol=config["seed hkl tolerance"],
        eval_hkl_tol=config["eval tolerance"],
        centering_weight=config["centering factor"],
        refine_mode=config["refine mode"],
        refine_cycles=int(config["refine cycles"]),
        # miller_file=config["miller file"], #TODO
        miller_file=None,
        nb_top=int(config["top solutions"]),
        multi_index=config["multi index"],
        nb_try=int(config["number of try"]),
        transform_matrix=transform_matrix,
        inv_transform_matrix=inv_transform_matrix,
        miller_set=None,
    )
