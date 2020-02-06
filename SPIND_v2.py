#!/usr/bin/env python
"""
Usage:
    SPIND.py <peak-dir> <table-file> [options]

Options:
    -h --help               Show this screen.
    -o --output DIRECTORY   Specify output directory [default: .].
    --batch-size NUM        Specify batch size in a jobs [default: 10].
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
    --nb-try NUM            Specify max trial for multiple indexing
                            [default: 3].
    --update-freq NUM       Specify update frequency [default: 10].
    -v --verbose           Whether enable verbose output.
"""
from __future__ import print_function
from six import print_ as print

try:
    import mkl
    mkl.set_num_threads(1)  # disable numpy multi-thread parallel computation
except:
    pass

from mpi4py import MPI
import numpy as np
import operator
import sys
import os
import time
from glob import glob
from docopt import docopt

from index import index
from util import load_peaks, load_table, calc_transform_matrix


def write_summary(results, directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    summary = []
    for result in results:
        peak_path = result['peak path']
        for crystal_id, solution in enumerate(result['solutions']):
            event_id = int(os.path.basename(
                peak_path).split('.')[0].split('-')[1])
            match_rate = solution.match_rate
            nb_pairs = len(solution.pair_ids)
            pair_dist_refined = solution.pair_dist_refined * 1E10
            tm = solution.transform_matrix_refined * 1E10
            record = [
                event_id, crystal_id, match_rate, nb_pairs,
                pair_dist_refined,
                tm[0, 0], tm[1, 0], tm[2, 0],
                tm[0, 1], tm[1, 1], tm[2, 1],
                tm[0, 2], tm[1, 2], tm[2, 2]
            ]
            summary.append(record)
    summary.sort(key=operator.itemgetter(0, 1))
    summary = np.array(summary)
    np.savetxt('%s/spind.txt' % directory, summary,
               fmt='%6d %2d %.2f %4d %.4E '
                   '%.4E %.4E %.4E '
                   '%.4E %.4E %.4E '
                   '%.4E %.4E %.4E')


def master_run(args):
    # parse parameters
    peak_dir = args['<peak-dir>']
    batch_size = int(args['--batch-size'])
    output_dir = args['--output']
    update_freq = int(args['--update-freq'])
    # collect and sort jobs
    peak_files = glob('%s/*.txt' % peak_dir)
    peak_files.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
    jobs = []
    job = []
    for i in range(len(peak_files)):
        job.append({'peak path': peak_files[i]})
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
        print('%d/%d --> %d' % (job_id, nb_jobs, worker), flush=True)
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
                    print('%d/%d --> %d'
                          % (job_id, nb_jobs, worker), flush=True)
                    reqs[worker] = comm.irecv(source=worker)
                    job_id += 1
                else:
                    stop = True
                    comm.isend(stop, dest=worker)
                    finished_workers.add(worker)
                if job_id % update_freq == 0:
                    write_summary(results, output_dir)
                    print('indexing summary updated!', flush=True)

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
    print('all done')


def worker_run(args):
    # parse parameters
    table = load_table(args['<table-file>'])
    min_res = table['min_resolution']
    max_res = table['max_resolution']
    lattice_constants = table['lattice_constants']
    centering = table['centering']
    transform_matrix = calc_transform_matrix(lattice_constants)
    inv_transform_matrix = np.linalg.inv(transform_matrix)
    seed_pool_size = int(args['--seed-pool-size'])
    seed_len_tol = float(args['--seed-len-tol'])
    seed_angle_tol = float(args['--seed-angle-tol'])
    seed_hkl_tol = float(args['--seed-hkl-tol'])
    eval_hkl_tol = float(args['--eval-hkl-tol'])
    centering_weight = float(args['--centering-weight'])
    refine_mode = args['--refine-mode']
    refine_cycles = int(args['--refine-cycles'])
    miller_file = args['--miller-file']
    nb_top = int(args['--top-solutions'])
    verbose = args['--verbose']
    if miller_file is not None:
        miller_set = np.loadtxt(miller_file)
    else:
        miller_set = None
    multi_index = args['--multi-index']
    nb_try = int(args['--nb-try'])
    stop = False
    while not stop:
        job = comm.recv(source=0)
        for i in range(len(job)):
            peaks = load_peaks(job[i]['peak path'],
                               min_res=min_res, max_res=max_res)
            np.savetxt('peaks.txt', peaks, fmt='%.4E')
            print('worker %d working on %s' % (rank, job[i]['peak path']),
                  flush=True)
            solutions = index(peaks, table,
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
                              multi_index=multi_index,
                              nb_try=nb_try,
                              verbose=verbose)
            job[i]['solutions'] = solutions
        comm.send(job, dest=0)
        stop = comm.recv(source=0)
    print('worker %d is exiting' % rank)


if __name__ == '__main__':
    # save command
    command_list = sys.argv
    command = ' '.join(command_list)
    with open('spind.com', 'w') as f:
        f.write('%s\n' % command)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if size == 1:
        print('Run SPIND with at least 2 processes!')
        sys.exit()

    rank = comm.Get_rank()
    args = docopt(__doc__)
    if rank == 0:
        print(args, flush=True)
        master_run(args)
    else:
        worker_run(args)
