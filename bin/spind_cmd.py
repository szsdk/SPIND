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

import operator
import os
from glob import glob
from itertools import repeat

# from mpi4py import MPI
# from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import click
import numpy as np
import rich

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


def worker_run(param, hklmatcher, peaks, num_threads):
    transform_matrix = spind.calc_transform_matrix(
        param.lattice_constants, param.lattice_type
    )
    inv_transform_matrix = np.linalg.inv(transform_matrix)
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
        num_threads=num_threads,
    )
    return solutions


def _peaks_from_stdin(inp):
    num_peaks = int.from_bytes(os.read(inp, 4), byteorder="little", signed=False)
    peaks = np.empty(num_peaks, dtype=spind.PEAKS_DTYPE)
    for i in range(num_peaks):
        a = np.frombuffer(os.read(inp, 32), "<f8")
        peaks[i]["coor"] = a[:3]
        peaks["intensity"] = a[3]
    return peaks


def _output_solutions(solutions, output):
    if isinstance(output, int):
        if len(solutions[0]) > 0:
            for s in solutions[0]:
                A = s.transform_matrix_refined * 1e10
                rich.print("A\n", A)
                os.write(output, A.tobytes("C"))
        else:
            os.write(output, np.full(9, np.nan, dtype=np.float64).tobytes("C"))

            # sys.stdout.buffer.write(solutions.transform_matrix_refined.tobytes("C"))
    else:
        print(solutions)


@click.command()
@click.argument("inp")
@click.option("--config", "-c")
@click.option("--output")
@click.option("--num-processes", type=int, default=1)
@click.option("--num-threads", type=int, default=1)
@click.option("--crystfel", is_flag=True, default=False, hidden=True)
def main(inp, config, output, num_processes, num_threads, crystfel):
    if crystfel:
        if num_processes != 1:
            raise ValueError("When stdin (-) is used, `num-processes` has to be 1.")
        inp = int(inp)
        output = int(output)

    param = spind.params(config)
    # collect and sort jobs

    hklmatcher = spind.hkl_matcher(param)

    rich.print(param)
    # peaks = spind.load_peaks(peaks, 'snr', param.min_res)
    if crystfel:
        solutions = [worker_run(param, hklmatcher, _peaks_from_stdin(inp), num_threads)]
    elif num_processes == 1:
        peak_files = glob("%s/*.txt" % inp)
        peak_files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
        solutions = [
            worker_run(
                param,
                hklmatcher,
                spind.load_peaks(pf, "snr", param.min_res),
                num_threads,
            )
            for pf in peak_files
        ]
    else:
        peak_files = glob("%s/*.txt" % inp)
        peak_files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
        with Pool(processes=num_processes) as pool:
            solutions = pool.starmap(
                worker_run,
                zip(
                    repeat(param),
                    repeat(hklmatcher),
                    map(
                        lambda pf: spind.load_peaks(pf, "snr", param.min_res),
                        peak_files,
                    ),
                    repeat(num_threads),
                ),
            )
    _output_solutions(solutions, output)


if __name__ == "__main__":
    main()
