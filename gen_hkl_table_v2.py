#!/usr/bin/env python
"""
Generate hkl table.

Usage:
    gen_hkl_table.py <lattice-constants> [options]

Options:
    -h --help               Show this screen.
    -o FILE                 Specify output table file [default: output.h5].
    --min-res NUM           Min resolution in angstrom.
    --max-res NUM           Max resolution in angstrom [default: 10].
    --centering TYPE        Centering type [default: P].
    --hkl-file FILE         hkl file.
"""
from __future__ import print_function
from six import print_ as print
from mpi4py import MPI
import numpy as np
from numpy.linalg import norm
import h5py

from docopt import docopt
import sys
import os
import time

from util import calc_transform_matrix, calc_angle


def master_run(jobs, output='table.h5'):
    # distribute jobs
    nb_jobs = len(jobs)
    job_id = 0
    reqs = {}
    workers = set(range(1, size))
    finished_workers = set()
    tic = time.time()
    for worker in workers:
        if job_id < nb_jobs:
            job = jobs[job_id]
        else:
            job = []  # dummy job
        comm.isend(job, dest=worker)
        print('send %d/%d' % (job_id, nb_jobs), flush=True)
        reqs[worker] = comm.irecv(source=worker)
        job_id += 1
    while job_id < nb_jobs:
        stop = False
        time.sleep(0.001)  # take a break
        workers -= finished_workers
        for worker in workers:
            finished, result = reqs[worker].test()
            if finished:
                if job_id < nb_jobs:
                    comm.isend(stop, dest=worker)
                    comm.isend(jobs[job_id], dest=worker)
                    print('send %d/%d' % (job_id, nb_jobs), flush=True)
                    reqs[worker] = comm.irecv(source=worker)
                    job_id += 1
                else:
                    stop = True
                    comm.isend(stop, dest=worker)
                    finished_workers.add(worker)

    all_accepted = False
    while not all_accepted:
        time.sleep(0.001)
        all_accepted = True
        workers -= finished_workers
        for worker in workers:
            finished, result = reqs[worker].test()
            if finished:
                stop = True
                comm.isend(stop, dest=worker)
                finished_workers.add(worker)
            else:
                all_accepted = False
       
    all_done = False
    workers = set(range(1, size))
    finished_workers = set()
    for worker in workers:
        reqs[worker] = comm.irecv(source=worker)
    while not all_done:
        time.sleep(0.001)
        all_done = True
        workers -= finished_workers
        for worker in workers:
            finished, result = reqs[worker].test()
            if finished:
                stop = True
                comm.isend(stop, dest=worker)
                finished_workers.add(worker)
            else:
                all_done = False

    # merge tables
    table = h5py.File(output, 'w')
    workers = set(range(1, size))
    hkl1_list = []
    hkl2_list = []
    len_angle_list = []
    len_list = []
    for worker in workers:
        worker_file = 'table-%d.h5' % worker
        h5_obj = h5py.File(worker_file, 'r')
        hkl1_list.append(h5_obj['hkl1'].value)
        hkl2_list.append(h5_obj['hkl2'].value)
        len_angle_list.append(h5_obj['len_angle'].value[:, :3])
        len_list.append(h5_obj['len_angle'].value[:, 3])
        os.remove(worker_file)

    hkl1 = np.concatenate(hkl1_list)
    hkl2 = np.concatenate(hkl2_list)
    len_angle = np.concatenate(len_angle_list)
    len_mean = np.concatenate(len_list)
    sorted_ids = np.argsort(len_mean)

    table.create_dataset('hkl1', data=hkl1[sorted_ids].astype(np.int8))
    table.create_dataset('hkl2', data=hkl2[sorted_ids].astype(np.int8))
    table.create_dataset(
        'len_angle', data=len_angle[sorted_ids].astype(np.float32))
    table.create_dataset('lattice_constants', data=lattice_constants)
    table.create_dataset('min_resolution', data=min_res)
    table.create_dataset('max_resolution', data=max_res)
    table.create_dataset('centering', data=centering)

    toc = time.time()
    print('time elapsed: %.2f sec' % (toc - tic))
    MPI.Finalize()


def worker_run():
    chunk_size = 10000
    stop = False
    count = 0
    chunk = []
    # create table file
    table = h5py.File('table-%d.h5' % rank)
    while not stop:
        job = comm.recv(source=0)
        # work on job
        for i in job:
            for j in range(i+1, len(hkl)):
                q1, q2 = q_vectors[i], q_vectors[j]
                len1, len2 = q_lengths[i], q_lengths[j]
                len_mean = (len1 + len2) * 0.5
                hkl1, hkl2 = hkl[i], hkl[j]
                angle = calc_angle(q1, q2, len1, len2)
                if len1 >= len2:
                    row = [hkl1[0], hkl1[1], hkl1[2],
                           hkl2[0], hkl2[1], hkl2[2],
                           len1, len2, angle, len_mean]
                else:
                    row = [hkl2[0], hkl2[1], hkl2[2],
                           hkl1[0], hkl1[1], hkl1[2],
                           len2, len1, angle, len_mean]
                chunk.append(row)
                count += 1
                if count % chunk_size == 0:
                    save_chunk(chunk, table)
                    chunk = []
        comm.send(job, dest=0)
        stop = comm.recv(source=0)
    # last chunk
    if len(chunk) > 0:
        save_chunk(chunk, table)
    table.close()

    done = True
    comm.send(done, dest=0)


def save_chunk(chunk, h5_obj):
    chunk_size = len(chunk)
    chunk = np.array(chunk)
    if 'hkl1' in h5_obj.keys():  # dataset existed
        n = h5_obj['hkl1'].shape[0]
        h5_obj['hkl1'].resize(n+chunk_size, axis=0)
        h5_obj['hkl1'][n:n+chunk_size] = chunk[:, 0:3].astype(np.int8)
        h5_obj['hkl2'].resize(n+chunk_size, axis=0)
        h5_obj['hkl2'][n:n+chunk_size] = chunk[:, 3:6].astype(np.int8)
        h5_obj['len_angle'].resize(n+chunk_size, axis=0)
        h5_obj['len_angle'][n:n+chunk_size] = chunk[:, 6:10].astype(np.float32)
    else:  # dataset not existed, create it
        h5_obj.create_dataset(
            'hkl1', data=chunk[:, 0:3].astype(np.int8), maxshape=(None, 3))
        h5_obj.create_dataset(
            'hkl2', data=chunk[:, 3:6].astype(np.int8), maxshape=(None, 3))
        h5_obj.create_dataset(
            'len_angle', data=chunk[:, 6:10].astype(np.float32),
            maxshape=[None, 4])


def gen_hkl_list(transform_matrix, min_res=np.inf, max_res=10, centering='P'):
    a_star, b_star, c_star = transform_matrix.T
    min_q = 1. / min_res
    max_q = 1. / max_res
    max_h = int(np.ceil(max_q / norm(a_star)))
    max_k = int(np.ceil(max_q / norm(b_star)))
    max_l = int(np.ceil(max_q / norm(c_star)))
    hh = np.arange(-max_h, max_h + 1)
    kk = np.arange(-max_k, max_k + 1)
    ll = np.arange(-max_l, max_l + 1)
    h, k, l = np.meshgrid(hh, kk, ll)
    hkl = np.concatenate(
            [
                h.reshape(-1, 1),
                k.reshape(-1, 1),
                l.reshape(-1, 1)
            ],
            axis=1
        )
    q = norm(transform_matrix.dot(hkl.T), axis=0)
    # TODO: need double check centering type
    if centering == 'I':
        valid_idx = np.where(
            (q > min_q) * (q < max_q) * (hkl.sum(axis=1) % 2 == 0))[0]
    elif centering == 'A':
        valid_idx = np.where(
            (q > min_q) * (q < max_q) * ((hkl[:, 1] + hkl[:, 2]) % 2 == 0))[0]
    elif centering == 'B':
        valid_idx = np.where(
            (q > min_q) * (q < max_q) * ((hkl[:, 0] + hkl[:, 2]) % 2 == 0))[0]
    elif centering == 'C':
        valid_idx = np.where(
            (q > min_q) * (q < max_q) * ((hkl[:, 0] + hkl[:, 1]) % 2 == 0))[0]
    elif centering == 'I':
        valid_idx = np.where(
            (q > min_q) * (q < max_q) * (np.sum(hkl, axis=1) % 2 == 0))[0]
    elif centering == 'P':
        valid_idx = np.where((q > min_q) * (q < max_q))[0]
    else:
        raise ValueError('Not supported centering type: %s' % centering)
    if len(valid_idx) == 0:
        raise ValueError(
            'No hkl available in resolution between %s and %s angstrom'
            % (str(min_res), str(max_res)))
    hkl = hkl[valid_idx]
    return hkl


if __name__ == '__main__':
    # mpi setup
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if size == 1:
        print('Run gen_hkl_table script with at least 2 processes!')
        sys.exit()

    rank = comm.Get_rank()
    # parse command options
    args = docopt(__doc__)
    lattice_constants = list(
        map(float, args['<lattice-constants>'].split(',')))
    output = args['-o']
    min_res = args['--min-res']
    if min_res is not None:
        min_res = float(min_res)
    else:
        min_res = np.inf
    max_res = float(args['--max-res'])
    centering = args['--centering']
    hkl_file = args['--hkl-file']
    A = calc_transform_matrix(lattice_constants)

    if hkl_file is None:
        hkl = gen_hkl_list(
            A, min_res=min_res, max_res=max_res, centering=centering)
    else:
        hkl = np.loadtxt(hkl_file)

    if hkl.min() < -128 or hkl.max() > 127:
        print('hkl out of int8, please consider reduce resolution range.')
        sys.exit()

    # calculate length and angles
    q_vectors = A.dot(hkl.T).T
    q_lengths = norm(q_vectors, axis=1)

    if rank == 0:  # master
        print(args)
        print('hkl orders: %d' % hkl.shape[0])
        jobs_ = np.arange(hkl.shape[0] - 1)  # raw jobs
        jobs = []  # grouped jobs
        for i in range(len(jobs_) // 2):
            jobs.append([jobs_[i], jobs_[-i-1]])
        if len(jobs_) % 2 == 1:
            mid = len(jobs_) // 2
            jobs.append([jobs_[mid]])
        # collect big job batch
        job_batches = []
        batches = np.array_split(np.arange(len(jobs)), 200)
        for batch in batches:
            if len(batch) > 0:
                job_batch = []
                for i in range(len(batch)):
                    job_batch += jobs[batch[i]]
                job_batches.append(job_batch)
        master_run(job_batches, output=output)
    else:
        worker_run()
