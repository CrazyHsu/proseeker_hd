import pandas as pd
import numpy as np
# from sklearn.cluster import KMeans
# from numpy import asarray
from numpy import savetxt
from common_functions import *
import os

import numba as nb
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")

VALID_AA_LIST = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


def kmeans_for_subarray(subarray, n_init=1000, n_iter=1000000, random_state=31):
    kmeans = KMeans(n_clusters=50, n_init=n_init, max_iter=n_iter, algorithm="lloyd", random_state=random_state)
    kmeans.fit(subarray)
    return kmeans.cluster_centers_, kmeans.labels_


@nb.jit(nopython=True)
def kmeans_for_subarray_numba(lib_aa_mat_sliding_window):
    n_init = 1000
    n_iter = 1000000
    random_state = 31
    for i in range(lib_aa_mat_sliding_window.shape[0]):
        for j in range(lib_aa_mat_sliding_window.shape[1]):
            kmeans = KMeans(n_clusters=50, n_init=n_init, max_iter=n_iter, algorithm="lloyd", random_state=random_state)
            kmeans.fit(lib_aa_mat_sliding_window[i, j])
    # return kmeans.cluster_centers_, kmeans.labels_


def process(lib_aa_list, mut_index, aavals):
    sliding_window = [list(range(x - 6, x + 7)) for x in mut_index]
    lib_aa_list = np.vstack(lib_aa_list)
    lib_aa_index = np.vectorize(lambda x: VALID_AA_LIST.index(x) if x in VALID_AA_LIST else -1)(lib_aa_list)
    lib_aa_mat = aavals[:, lib_aa_index]
    lib_aa_mat = lib_aa_mat.transpose(1, 0, 2)
    lib_aa_mat = np.concatenate((lib_aa_mat, lib_aa_mat[:, :, list(range(6))]), axis=2)
    lib_aa_mat_sliding_window = lib_aa_mat[:, :, sliding_window]
    lib_aa_mat_sliding_window = lib_aa_mat_sliding_window.transpose(0, 2, 1, 3)

    cluster_centers_list, labels_list = zip(
        *Parallel(n_jobs=-1)(delayed(kmeans_for_subarray)(lib_aa_mat_sliding_window[i, j])
                             for i in range(lib_aa_mat_sliding_window.shape[0])
                             for j in range(lib_aa_mat_sliding_window.shape[1])))

    labels_list = np.array(labels_list)
    labels = labels_list.reshape(lib_aa_mat_sliding_window.shape[0], lib_aa_mat_sliding_window.shape[1], -1)
    lib_aa_mat_sliding_window = np.insert(lib_aa_mat_sliding_window, 13, labels, axis=3)
    return lib_aa_mat_sliding_window


def process1(lib_aa_list, mut_index, aavals):
    sliding_window = [list(range(x - 6, x + 7)) for x in mut_index]
    lib_aa_list = np.vstack(lib_aa_list)
    lib_aa_index = np.vectorize(lambda x: VALID_AA_LIST.index(x) if x in VALID_AA_LIST else -1)(lib_aa_list)
    lib_aa_mat = aavals[:, lib_aa_index]
    lib_aa_mat = lib_aa_mat.transpose(1, 0, 2)
    lib_aa_mat = np.concatenate((lib_aa_mat, lib_aa_mat[:, :, list(range(6))]), axis=2)
    lib_aa_mat_sliding_window = lib_aa_mat[:, :, sliding_window]
    lib_aa_mat_sliding_window = lib_aa_mat_sliding_window.transpose(0, 2, 1, 3)

    cluster_centers_list, labels_list = kmeans_for_subarray_numba(lib_aa_mat_sliding_window)

    labels_list = np.array(labels_list)
    labels = labels_list.reshape(lib_aa_mat_sliding_window.shape[0], lib_aa_mat_sliding_window.shape[1], -1)
    lib_aa_mat_sliding_window = np.insert(lib_aa_mat_sliding_window, 13, labels, axis=3)
    return lib_aa_mat_sliding_window


def main():
    assess_lib_file = "example_lib_bak.csv"
    ranking_file = "ranking.csv"
    output_dir = "test"
    resolveDir(output_dir)

    VALID_AA_LIST = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    aavals = pd.read_csv(ranking_file, usecols=VALID_AA_LIST, sep=',')
    aavals = normalize_aavals_numpy(aavals)

    library = pd.read_csv(assess_lib_file, header=None, sep=',')
    seqs = [list(x) for x in library[0].tolist()]
    sites = list(map(int, '1:2:3:4:46:49:50:53:54'.split(":")))

    lib_aa_mat_sliding_window = process(seqs, sites, aavals)
    for i in range(lib_aa_mat_sliding_window.shape[0]):
        for j in range(lib_aa_mat_sliding_window.shape[1]):
            out_file = os.path.join(output_dir, 'p{}.bres{}.csv'.format(i + 1, j + 1))
            savetxt(out_file, lib_aa_mat_sliding_window[i, j], delimiter=',', fmt='%f')


if __name__ == '__main__':
    main()
