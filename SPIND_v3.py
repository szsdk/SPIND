#!/usr/bin/env python
"""
Usage:
    SPIND.py <peak-dir> [options]

Options:
    -h --help               Show this screen.
    -o --output DIRECTORY   Specify output directory [default: .].
    --num_proc NUM          Number of processes [default: 1]
    --config STR            Config file [default: none]
    --batch-size NUM        Specify batch size in a jobs [default: 10].
    --min-resolution NUM    resolution in A [default: 4.5].
    --max-resolution NUM    resolution in A [default: 1e10].
    --seed-pool-size NUM    Specify size of seed pool [default: 5].
    --seed-len-tol NUM      Specify length tolerance of seed in per angstrom
                            [default: 0.003].
    --seed-angle-tol NUM    Specify angle tolerance of seed in degrees
                            [default: 1].
    --seed-hkl-tol NUM      Specify hkl tolerance of seed [default: 1.0].
    --eval-hkl-tol NUM      Specify hkl tolerance of paired peaks
                            [default: 0.25].
    --centering-weight NUM  Specify weight of centering score [default: 0.].
    --refine-mode MODE      Specify refine mode [default: global].
    --refine-cycles NUM     Specify number of refinement cycles for
                            alternative mode [default: 10].
    --miller-file FILE      Specify miller file which contains all
                            possible hkl indices.
    --top-solutions NUM     Specify number of top solutions for match rate
                            correction [default: 5].
    --multi-index           Whether enable multiple indexing.
    --lattice-constants STR .
    --lattice-type STR      .
    --centering STR         .
    --nb-try NUM            Specify max trial for multiple indexing
                            [default: 3].
    --update-freq NUM       Specify update frequency [default: 10].
    -v --verbose            Whether enable verbose output.
"""

# from mpi4py import MPI
# from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import numpy as np
from collections import namedtuple
import operator
import sys
import os
import time
import yaml
from pathlib import Path
from glob import glob
from docopt import docopt

# from index import index
# from util import load_peaks, load_table, calc_transform_matrix
import spind


def write_summary(results, directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    summary = []
    for result in results:
        peak_path = result["peak path"]
        for crystal_id, solution in enumerate(result["solutions"]):
            event_id = int(os.path.basename(peak_path).split(".")[0].split("-")[1])
            match_rate = solution.match_rate
            nb_pairs = len(solution.pair_ids)
            pair_dist_refined = solution.pair_dist_refined * 1e10
            tm = solution.transform_matrix_refined * 1e10
            record = [
                event_id,
                crystal_id,
                match_rate,
                nb_pairs,
                pair_dist_refined,
                tm[0, 0],
                tm[1, 0],
                tm[2, 0],
                tm[0, 1],
                tm[1, 1],
                tm[2, 1],
                tm[0, 2],
                tm[1, 2],
                tm[2, 2],
            ]
            summary.append(record)
    summary.sort(key=operator.itemgetter(0, 1))
    summary = np.array(summary)
    np.savetxt(
        "%s/spind.txt" % directory,
        summary,
        fmt="%6d %2d %.2f %4d %.4E "
        "%.4E %.4E %.4E "
        "%.4E %.4E %.4E "
        "%.4E %.4E %.4E",
    )


def master_run(args):
    # parse parameters
    peak_dir = args["<peak-dir>"]
    batch_size = int(args["--batch-size"])
    output_dir = args["--output"]
    update_freq = int(args["--update-freq"])
    # collect and sort jobs
    peak_files = glob("%s/*.txt" % peak_dir)
    peak_files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
    jobs = []
    job = []
    for i in range(len(peak_files)):
        job.append({"peak path": peak_files[i]})
        if len(job) == batch_size:
            jobs.append(job)
            job = []
    if len(job) > 0:
        jobs.append(job)
    nb_jobs = len(jobs)

    # dispatch jobs
    job_id = 0
    reqs = {}
    results = []
    workers = set(range(1, size))
    finished_workers = set()
    for worker in workers:
        if job_id < nb_jobs:
            job = jobs[job_id]
        else:
            job = []  # dummy job
        comm.isend(job, dest=worker)
        print("%d/%d --> %d" % (job_id, nb_jobs, worker), flush=True)
        reqs[worker] = comm.irecv(source=worker)
        job_id += 1

    while job_id < nb_jobs:
        time.sleep(0.001)
        workers -= finished_workers
        for worker in workers:
            finished, result = reqs[worker].test()
            if finished:
                results += result
                if job_id < nb_jobs:
                    stop = False
                    comm.isend(stop, dest=worker)
                    comm.isend(jobs[job_id], dest=worker)
                    print("%d/%d --> %d" % (job_id, nb_jobs, worker), flush=True)
                    reqs[worker] = comm.irecv(source=worker)
                    job_id += 1
                else:
                    stop = True
                    comm.isend(stop, dest=worker)
                    finished_workers.add(worker)
                if job_id % update_freq == 0:
                    write_summary(results, output_dir)
                    print("indexing summary updated!", flush=True)

    all_done = False
    while not all_done:
        time.sleep(0.001)
        all_done = True
        workers -= finished_workers
        for worker in workers:
            finished, result = reqs[worker].test()
            if finished:
                results += result
                stop = True
                comm.isend(stop, dest=worker)
                finished_workers.add(worker)
            else:
                all_done = False

    write_summary(results, output_dir)
    print("all done")


Params = namedtuple(
    "Params",
    [
        "min_res",
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
        "verbose",
        "multi_index",
        "nb_try",
    ],
)


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


def param_from_yaml(config_file):
    with open(config_file) as fp:
        config = yaml.safe_load(fp)
    return Params(
        min_res=config["resolution cutoff"] * 1e10,
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
        verbose=config["verbose"],
        multi_index=config["multi index"],
        nb_try=int(config["number of try"]),
    )



def worker_run(param):
    # parse parameters
    # table = load_table(args["<table-file>"])
    # min_res = table["min_resolution"]
    # max_res = table["max_resolution"]
    # lattice_constants = table["lattice_constants"]
    # centering = table["centering"]
    transform_matrix = spind.calc_transform_matrix(param.lattice_constants, param.lattice_type)
    hklmatcher = spind.hkl_matcher(
        param.min_res, transform_matrix, param.centering,
        param.seed_len_tol, param.seed_angle_tol)
    inv_transform_matrix = np.linalg.inv(transform_matrix)
    if param.miller_file is not None:
        raise NotADirectoryError()
        miller_set = np.loadtxt(param.miller_file)
    else:
        miller_set = None
    stop = False
    while not stop:
        job = comm.recv(source=0)
        for i in range(len(job)):
            peaks = spind.load_peaks(
                job[i]["peak path"], min_res=param.min_res, max_res=param.max_res
            )
            np.savetxt("peaks.txt", peaks, fmt="%.4E")
            print("worker %d working on %s" % (rank, job[i]["peak path"]), flush=True)
            solutions = spind.index(
                peaks,
                hklmatcher,
                transform_matrix,
                inv_transform_matrix,
                **param._asdict(),
                miller_set=miller_set,
            )
            job[i]["solutions"] = solutions
        comm.send(job, dest=0)
        stop = comm.recv(source=0)
    print("worker %d is exiting" % rank)


def temp_worker_run(param, hklmatcher, peak_file):
    transform_matrix = spind.calc_transform_matrix(param.lattice_constants, param.lattice_type)
    inv_transform_matrix = np.linalg.inv(transform_matrix)
    peaks = spind.load_peaks(peak_file, 'snr', param.min_res)
    solutions = spind.index(
        peaks,
        hklmatcher,
        transform_matrix,
        inv_transform_matrix,
        miller_set=None,
        seed_pool_size=param.seed_pool_size,
        seed_len_tol=param.seed_len_tol,
        seed_angle_tol=param.seed_angle_tol,
        seed_hkl_tol=param.seed_hkl_tol,
        eval_hkl_tol=param.eval_hkl_tol,
        centering=param.centering,
        centering_weight=param.centering_weight,
        refine_mode=param.refine_mode,
        refine_cycles=param.refine_cycles,
        nb_top=param.nb_top,
        multi_index=param.multi_index,
        verbose=param.verbose,
    )
    return solutions

def main():
    args = docopt(__doc__)
    if args['--config'] == 'none':
        param = param_from_args(args)
    else:
        param = param_from_yaml(args['--config'])

    peak_dir = args["<peak-dir>"]
    output_dir = args["--output"]
    # collect and sort jobs
    peak_files = glob("%s/*.txt" % peak_dir)
    peak_files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))

    transform_matrix = spind.calc_transform_matrix(param.lattice_constants, param.lattice_type)
    hklmatcher = spind.hkl_matcher(
        param.min_res, transform_matrix, param.centering,
        param.seed_len_tol, param.seed_angle_tol)

    print(param)
    # for peak_file in peak_files:
    # with ThreadPoolExecutor(max_workers=8) as executor:
        # solutions = list(executor.map(lambda peak_file: temp_worker_run(param, hklmatcher, peak_file), peak_files))
    from itertools import repeat
    with Pool(processes=int(args['--num_proc'])) as pool:
        solutions = pool.starmap(temp_worker_run, zip(repeat(param), repeat(hklmatcher), peak_files))
    print(solutions)


main()
