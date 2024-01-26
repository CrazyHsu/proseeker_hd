#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: Proseeker_xufeng_numba_numpy_test.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2024-01-09 20:19:59
Last modified: 2024-01-09 20:19:59
'''

import sys, time, copy
# import subprocess
# import pkg_resources
import os, threading
from datetime import datetime
# import random
import warnings
import csv
# import shutil
# import math
from statistics import mean
import scipy

import numpy as np
import pandas as pd
import numba as nb
from numba import jit, types, prange, set_num_threads
from sklearn.metrics import mean_squared_error
from common_functions import *

# import taichi as ti
# ti.init(arch=ti.cpu)
# from sklearn.metrics import mean_squared_error
# from scipy.stats import shapiro
# from itertools import product
# from scipy.stats import kstest
# import logomaker as lm
# import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

VALID_AA_LIST = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
N_THREADINGS = 8
# nb.config.NUMBA_NUM_THREADS = N_THREADINGS
set_num_threads(N_THREADINGS)


def make_multithread(inner_func, numthreads):
    """
    Run the given function inside *numthreads* threads, splitting
    its arguments into equal-sized chunks.
    """
    def func_mt(*args):
        length = len(args[0])
        result = np.empty(length, dtype=np.float64)
        args = (result,) + args
        chunklen = (length + numthreads - 1) // numthreads
        # Create argument tuples for each input chunk
        chunks = [[arg[i * chunklen:(i + 1) * chunklen] for arg in
                   args] for i in range(numthreads)]
        # Spawn one thread per chunk
        threads = [threading.Thread(target=inner_func, args=chunk)
                   for chunk in chunks]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return result
    return func_mt


# def get_character_index(s, ch):
#     return [i for i, ltr in enumerate(s) if ltr == ch]


def sort_arr_by_axis(arr, axis_num=0):
    ndim = np.ndim(arr)
    new_axis = [*range(axis_num, ndim), *range(axis_num)]
    arr = arr.transpose(new_axis)
    arr = np.sort(arr, axis=0)
    transpose_index = np.argsort(new_axis)
    arr = arr.transpose(transpose_index)
    return arr

# def make_index_equal_length(bres_large_index, fill_value=-1):
#     max_length = max(len(sublist) for sublist in bres_large_index)
#     return np.array([sublist.tolist() + [fill_value] * (max_length - len(sublist)) for sublist in bres_large_index])


def pad_bres_large_df(bres_large_df):
    max_len = bres_large_df.groupby(by=["n_site", "n_bresnum", 13])[13].transform('size').max()
    pad_list = []
    for name, group in bres_large_df.groupby(by=["n_site", "n_bresnum", 13]):
        base_list = [-1] * 13 + [name[2], name[0], name[1]]
        pad_list.extend([base_list] * (max_len - group.shape[0]))

    pad_df = pd.DataFrame(pad_list, columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, "n_site", "n_bresnum"])
    bres_large_df_new = pd.concat([bres_large_df, pad_df])
    bres_large_df_new = bres_large_df_new.reset_index(drop=True)
    bres_large_df_new = bres_large_df_new.sort_values(by=[13, "n_site", "n_bresnum"])
    bres_large_df_new = bres_large_df_new.reset_index(drop=True)
    bres_large_df_new_zero = bres_large_df_new.replace(-1, 0)
    bres_large_df_new_na = bres_large_df_new.replace(-1, np.nan)
    # bres_large_df_new_zero = np.where(bres_large_df_new == -1, 0, bres_large_df_new)
    # bres_large_df_new_na = np.where(bres_large_df_new == -1, np.nan, bres_large_df_new)
    return bres_large_df_new_zero, bres_large_df_new_na


def pad_index(bres_large_index):
    max_len = max(len(tmp_index) for indices_group in bres_large_index for indices in indices_group for tmp_index in indices)
    bres_large_index_padded = np.full((bres_large_index.shape[0], bres_large_index.shape[1], 50, max_len), -1)
    for i, indices_group in enumerate(bres_large_index):
        for j, indices in enumerate(indices_group):
            for z, tmp_index in enumerate(indices):
                bres_large_index_padded[i, j, z, :len(tmp_index)] = np.array(tmp_index)
    return bres_large_index_padded


def normalize_aavals_numpy(aavals):
    aavals = aavals.to_numpy()
    min_values = aavals.min(axis=1, keepdims=True)
    max_values = aavals.max(axis=1, keepdims=True)
    normalize_aavals = (aavals - min_values) / (max_values - min_values)
    return normalize_aavals


def calculate_slope(group):
    group1 = group.iloc[:, :13]
    x_sorted = group1.transform(np.sort)
    return np.polyfit(range(1, x_sorted.shape[0]+1), x_sorted, 1)[0]


def get_assessment_info1(sites, bresnum, bresdir, aaprobset):
    # d = dict()
    # d['aaprobset'] = aaprobset
    # d1 = {}
    # bres_final_mean = []
    # bres_final_slope = []
    bres_list = []
    for bs in range(1, sites + 1):
        # bres_mean_list = []
        # bres_slope_list = []
        for x in range(1, bresnum + 1):
            bres = pd.read_csv(os.path.join(bresdir, 'p{}.bres{}.csv'.format(x, bs)), header=None, sep=',')
            bres["n_site"] = bs
            bres["n_bresnum"] = x
            bres_list.append(bres)
    bres_large_df = pd.concat(bres_list)
    bres_large_mean = bres_large_df.groupby(by=["n_site", "n_bresnum", 13]).mean().to_numpy()
    bres_large_mean = np.reshape(bres_large_mean, [sites, bresnum, 50, 13])

    bres_large_slope = bres_large_df.groupby(by=["n_site", "n_bresnum", 13]).apply(calculate_slope).to_list()
    bres_large_slope = np.stack(bres_large_slope, axis=0)
    bres_large_slope = np.reshape(bres_large_slope, [sites, bresnum, 50, 13])

    bres_large_index = bres_large_df.groupby(by=["n_site", "n_bresnum", 13]).apply(lambda g: g.index)
    bres_large_index = np.reshape(bres_large_index.values, [sites, bresnum, -1])
    return bres_large_mean, bres_large_slope, bres_large_index


def get_assessment_info_with_pad(sites, bresnum, bresdir):
    bres_list = []
    for bs in range(1, sites + 1):
        for x in range(1, bresnum + 1):
            bres = pd.read_csv(os.path.join(bresdir, 'p{}.bres{}.csv'.format(x, bs)), header=None, sep=',')
            bres["n_site"] = bs
            bres["n_bresnum"] = x
            bres_list.append(bres)
    bres_large_df = pd.concat(bres_list)
    bres_large_df_padded_zero, bres_large_df_padded_na = pad_bres_large_df(bres_large_df)

    bres_large_mean = bres_large_df_padded_na.groupby(by=["n_site", "n_bresnum", 13]).mean().to_numpy()
    bres_large_mean = np.reshape(bres_large_mean, [sites, bresnum, 50, 13])

    bres_large_slope = bres_large_df_padded_zero.groupby(by=["n_site", "n_bresnum", 13]).apply(calculate_slope).to_list()
    bres_large_slope = np.stack(bres_large_slope, axis=0)
    bres_large_slope = np.reshape(bres_large_slope, [sites, bresnum, 50, 13])

    bres_large_index = bres_large_df.groupby(by=["n_site", "n_bresnum", 13]).apply(lambda g: g.index)
    bres_large_index = np.reshape(bres_large_index.values, [sites, bresnum, -1])
    return bres_large_mean, bres_large_slope, bres_large_index


def get_mean_slope_with_loop(bres_large_index, g_generated_aa_mat1_in_window):
    count = 0
    for i, indices_group in enumerate(bres_large_index):
        # if i > 0: break
        for j, indices in enumerate(indices_group):
            # if j > 0: break
            for z, tmp_index in enumerate(indices):
                count += 1
                g_generated_aa_mat1_in_window_tmp = g_generated_aa_mat1_in_window[:, :, np.array(tmp_index), :]
                means = g_generated_aa_mat1_in_window_tmp.mean(axis=2)
                transposed_arr = g_generated_aa_mat1_in_window_tmp.transpose((2, 3, 0, 1))
                sorted_transposed_arr = np.sort(transposed_arr, axis=0)
                sorted_original_shape_arr = sorted_transposed_arr.transpose((2, 3, 0, 1))
                indices_ = np.arange(1, sorted_original_shape_arr.shape[2] + 1)
                slopes = np.apply_along_axis(lambda x: np.polyfit(indices_, x, 1)[0], axis=2, arr=sorted_original_shape_arr)


def get_mean_slope_with_vectorized(bres_large_index_padded, g_generated_aa_mat_padded_na_sliding_window, g_generated_aa_mat_padded_zero_sliding_window):
    tmp_na_arr = g_generated_aa_mat_padded_na_sliding_window[:, :, bres_large_index_padded, :]
    tmp_zero_arr = g_generated_aa_mat_padded_zero_sliding_window[:, :, bres_large_index_padded, :]
    tmp_zero_arr = sort_arr_by_axis(tmp_zero_arr, axis_num=5)
    mean_arr = np.nanmean(tmp_na_arr, axis=5)
    slope_arr = np.apply_along_axis(lambda x: np.polyfit(np.arange(1, bres_large_index_padded.shape[3] + 1), x, 1)[0], axis=5, arr=tmp_zero_arr)
    return mean_arr, slope_arr


def get_padded_mean_slope(bres_large_index_padded, g_generated_aa_mat_padded_na_sliding_window, g_generated_aa_mat_padded_zero_sliding_window):
    # a = g_generated_aa_mat_padded_na_sliding_window[:, :, bres_large_index_padded, :]
    # b = g_generated_aa_mat_padded_zero_sliding_window[:, :, bres_large_index_padded, :]
    # np.apply_along_axis(lambda x: np.polyfit(range(1, 38), x, 1)[0], axis=5, arr=b)

    for i in range(bres_large_index_padded.shape[0]):
        for j in range(bres_large_index_padded.shape[1]):
            tmp_na_arr = g_generated_aa_mat_padded_na_sliding_window[:, :, bres_large_index_padded[i, j, ...], :]
            tmp_mean_arr = np.nanmean(tmp_na_arr, axis=3)

            tmp_zero_arr = g_generated_aa_mat_padded_zero_sliding_window[:, :, bres_large_index_padded[i, j, ...], :]
            tmp_zero_arr = sort_arr_by_axis(tmp_zero_arr, axis_num=3)
            # tmp_zero_arr = tmp_zero_arr.transpose((2, 3, 0, 1))
            # tmp_zero_arr = np.sort(tmp_zero_arr, axis=0)
            # tmp_zero_arr = tmp_zero_arr.transpose((2, 3, 0, 1))
            tmp_slope_arr = np.apply_along_axis(lambda x: np.polyfit(range(1, 38), x, 1)[0], axis=3, arr=tmp_zero_arr)


@nb.jit
def _coeff_mat(x, deg):
    mat_ = np.zeros(shape=(x.shape[0], deg + 1))
    const = np.ones_like(x)
    mat_[:, 0] = const
    mat_[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            mat_[:, n] = x ** n
    return mat_


@nb.jit
def _fit_x(a, b):
    # linalg solves ax = b
    det_ = np.linalg.lstsq(a, b)[0]
    return det_


@nb.jit
def fit_poly(x: np.ndarray, y: np.ndarray, deg):
    a = _coeff_mat(x, deg)
    p = _fit_x(a, y)
    # Reverse order so p[0] is coefficient of highest order
    return p[::-1]


@nb.jit
def cal_rmse(y1: np.ndarray, y2: np.ndarray):
    """
    Root mean squared error (RMSE) calculation
    """
    return np.sqrt(((y1-y2)**2).mean())


@nb.jit
def cal_rsq(y1: np.ndarray, y2: np.ndarray):
    """
    R-square calculation
    """
    corr_matrixa = np.corrcoef(y1, y2)
    corra = corr_matrixa[0, 1]
    return 1 - (corra ** 2)


@jit(nopython=True, parallel=True)
def get_padded_mean_slope_numba(bres_large_index_padded, g_generated_aa_mat_padded_na_sliding_window, g_generated_aa_mat_padded_zero_sliding_window, bres_large_mean_padded, bres_large_slope_padded):
    # (4, 10, 50, 37) bres_large_index_padded
    # (100, 9, 545, 13) g_generated_aa_mat_padded_na_sliding_window
    # (4, 10, 100, 9, 50, 13) target dim
    my_shape1 = bres_large_index_padded.shape
    my_shape2 = g_generated_aa_mat_padded_na_sliding_window.shape
    mean_arr = np.zeros(shape=(my_shape1[0], my_shape1[1], my_shape2[0], my_shape2[1], my_shape1[2], my_shape2[3]))
    slope_arr = np.zeros(shape=(my_shape1[0], my_shape1[1], my_shape2[0], my_shape2[1], my_shape1[2], my_shape2[3]))
    rsq_arr = np.zeros(shape=(my_shape1[0], my_shape1[1], my_shape2[0], my_shape2[1], my_shape1[2]))
    rmse_arr = np.zeros(shape=(my_shape1[0], my_shape1[1], my_shape2[0], my_shape2[1], my_shape1[2]))
    # for j_1 in prange(my_shape1[1]):
    #     for i_1 in prange(my_shape1[0]):
    for i_1 in prange(my_shape1[0]):
        for j_1 in prange(my_shape1[1]):
            for z_1 in prange(my_shape1[2]):
                tmp_na_arr = np.zeros(shape=(my_shape2[0], my_shape2[1], my_shape1[3], my_shape2[3]))
                tmp_zero_arr = np.zeros(shape=(my_shape2[0], my_shape2[1], my_shape1[3], my_shape2[3]))
                for k_1 in prange(my_shape1[3]):
                    my_index = bres_large_index_padded[i_1, j_1, z_1, k_1]
                    tmp_na_arr[:, :, k_1, :] = g_generated_aa_mat_padded_na_sliding_window[:, :, my_index, :]
                    tmp_zero_arr[:, :, k_1, :] = g_generated_aa_mat_padded_zero_sliding_window[:, :, my_index, :]

                for i_2 in prange(tmp_na_arr.shape[0]):
                    for j_2 in prange(tmp_na_arr.shape[1]):
                        for k_2 in prange(tmp_na_arr.shape[3]):
                            mean_arr[i_1, j_1, i_2, j_2, z_1, k_2] = np.nanmean(tmp_na_arr[i_2, j_2, :, k_2])
                            tmp_zero_list = tmp_zero_arr[i_2, j_2, :, k_2]
                            tmp_zero_list = sorted(tmp_zero_list)
                            slope_arr[i_1, j_1, i_2, j_2, z_1, k_2] = fit_poly(np.arange(1, my_shape1[3] + 1), np.array(tmp_zero_list), 1)[0]

                        rsq_arr[i_1, j_1, i_2, j_2, z_1] = cal_rsq(mean_arr[i_1, j_1, i_2, j_2, z_1, :], bres_large_mean_padded[i_1, j_1, z_1, :])
                        # corr_matrixa = np.corrcoef(mean_arr[i_1, j_1, i_2, j_2, z_1, :], bres_large_mean_padded[i_1, j_1, z_1, :])
                        # corra = corr_matrixa[0, 1]
                        # rsq_arr[i_1, j_1, i_2, j_2, z_1] = 1 - (corra ** 2)

                        rmse_arr[i_1, j_1, i_2, j_2, z_1] = cal_rmse(mean_arr[i_1, j_1, i_2, j_2, z_1, :], mean_arr[i_1, j_1, i_2, j_2, z_1, :])
                        # rmse_arr[i_1, j_1, i_2, j_2, z_1] = euclidian_distance(mean_arr[i_1, j_1, i_2, j_2, z_1, :], mean_arr[i_1, j_1, i_2, j_2, z_1, :])
                        # rmse_arr[i_1, j_1, i_2, j_2, z_1] = mean_squared_error(mean_arr[i_1, j_1, i_2, j_2, z_1, :], mean_arr[i_1, j_1, i_2, j_2, z_1, :], squared=False)

    mean_arr = mean_arr.transpose(2, 3, 0, 1, 4, 5)
    slope_arr = slope_arr.transpose(2, 3, 0, 1, 4, 5)
    rsq_arr = rsq_arr.transpose(2, 3, 0, 1, 4)
    rmse_arr = rmse_arr.transpose(2, 3, 0, 1, 4)
    return mean_arr, slope_arr, rsq_arr, rmse_arr


@jit(nopython=True, parallel=True)
def get_padded_mean_slope_numba1(bres_large_index_padded, g_generated_aa_mat_padded_na_sliding_window, g_generated_aa_mat_padded_zero_sliding_window, bres_large_mean_padded, bres_large_slope_padded):
    # (4, 10, 50, 37) bres_large_index_padded
    # (100, 9, 545, 13) g_generated_aa_mat_padded_na_sliding_window
    # (4, 10, 100, 9, 50, 13) target dim
    my_shape1 = bres_large_index_padded.shape
    my_shape2 = g_generated_aa_mat_padded_na_sliding_window.shape
    mean_arr = np.zeros(shape=(my_shape1[0], my_shape2[0], my_shape2[1], my_shape1[1], my_shape2[3]))
    slope_arr = np.zeros_like(mean_arr)
    rsq_arr = np.zeros(shape=(my_shape1[0], my_shape2[0], my_shape2[1], my_shape1[1]))
    rmse_arr = np.zeros_like(rsq_arr)
    # for j_1 in prange(my_shape1[1]):
    #     for i_1 in prange(my_shape1[0]):
    for i_1 in prange(my_shape1[0]):
        for j_1 in range(my_shape1[1]):
            # for z_1 in prange(my_shape1[2]):
                tmp_na_arr = np.zeros(shape=(my_shape2[0], my_shape2[1], my_shape1[2], my_shape2[3]))
                tmp_zero_arr = np.zeros_like(tmp_na_arr)
                for k_1 in range(my_shape1[2]):
                    my_index = bres_large_index_padded[i_1, j_1, k_1]
                    tmp_na_arr[:, :, k_1, :] = g_generated_aa_mat_padded_na_sliding_window[:, :, my_index, :]
                    tmp_zero_arr[:, :, k_1, :] = g_generated_aa_mat_padded_zero_sliding_window[:, :, my_index, :]

                for i_2 in range(tmp_na_arr.shape[0]):
                    for j_2 in range(tmp_na_arr.shape[1]):
                        for k_2 in range(tmp_na_arr.shape[3]):
                            mean_arr[i_1, i_2, j_2, j_1, k_2] = np.nanmean(tmp_na_arr[i_2, j_2, :, k_2])
                            tmp_zero_list = tmp_zero_arr[i_2, j_2, :, k_2]
                            tmp_zero_list = sorted(tmp_zero_list)
                            slope_arr[i_1, i_2, j_2, j_1, k_2] = fit_poly(np.arange(1, my_shape1[2] + 1), np.array(tmp_zero_list), 1)[0]

                        rsq_arr[i_1, i_2, j_2, j_1] = cal_rsq(mean_arr[i_1, i_2, j_2, j_1, :], bres_large_mean_padded[i_1, j_1, :])
                        # corr_matrixa = np.corrcoef(mean_arr[i_1, j_1, i_2, j_2, z_1, :], bres_large_mean_padded[i_1, j_1, z_1, :])
                        # corra = corr_matrixa[0, 1]
                        # rsq_arr[i_1, j_1, i_2, j_2, z_1] = 1 - (corra ** 2)

                        rmse_arr[i_1, i_2, j_2, j_1] = cal_rmse(mean_arr[i_1, i_2, j_2, j_1, :], mean_arr[i_1, i_2, j_2, j_1, :])
                        # rmse_arr[i_1, j_1, i_2, j_2, z_1] = euclidian_distance(mean_arr[i_1, j_1, i_2, j_2, z_1, :], mean_arr[i_1, j_1, i_2, j_2, z_1, :])
                        # rmse_arr[i_1, j_1, i_2, j_2, z_1] = mean_squared_error(mean_arr[i_1, j_1, i_2, j_2, z_1, :], mean_arr[i_1, j_1, i_2, j_2, z_1, :], squared=False)

    # mean_arr = mean_arr.transpose(1, 2, 0, 3, 4)
    # slope_arr = slope_arr.transpose(1, 2, 0, 3, 4)
    # rsq_arr = rsq_arr.transpose(1, 2, 0, 3)
    # rmse_arr = rmse_arr.transpose(1, 2, 0, 3)

    # mean_arr = mean_arr.transpose(2, 3, 0, 1, 4, 5)
    # slope_arr = slope_arr.transpose(2, 3, 0, 1, 4, 5)
    # rsq_arr = rsq_arr.transpose(2, 3, 0, 1, 4)
    # rmse_arr = rmse_arr.transpose(2, 3, 0, 1, 4)
    # return mean_arr, slope_arr, rsq_arr, rmse_arr
    return mean_arr.transpose(1, 2, 0, 3, 4), slope_arr.transpose(1, 2, 0, 3, 4), rsq_arr.transpose(1, 2, 0, 3), rmse_arr.transpose(1, 2, 0, 3)


def get_padded_mean_slope_numba2(bres_large_index_padded, g_generated_aa_mat_padded_na_sliding_window, g_generated_aa_mat_padded_zero_sliding_window, bres_large_mean_padded, bres_large_slope_padded):
    # (4, 10, 50, 37) bres_large_index_padded
    # (100, 9, 545, 13) g_generated_aa_mat_padded_na_sliding_window
    # (4, 10, 100, 9, 50, 13) target dim
    my_shape1 = bres_large_index_padded.shape
    my_shape2 = g_generated_aa_mat_padded_na_sliding_window.shape
    mean_arr = np.zeros(shape=(my_shape1[0], my_shape2[0], my_shape2[1], my_shape1[1], my_shape2[3]))
    slope_arr = np.zeros(shape=(my_shape1[0], my_shape2[0], my_shape2[1], my_shape1[1], my_shape2[3]))
    rsq_arr = np.zeros(shape=(my_shape1[0], my_shape2[0], my_shape2[1], my_shape1[1]))
    rmse_arr = np.zeros(shape=(my_shape1[0], my_shape2[0], my_shape2[1], my_shape1[1]))
    # for j_1 in prange(my_shape1[1]):
    #     for i_1 in prange(my_shape1[0]):
    for i_1 in range(my_shape1[0]):
        for j_1 in range(my_shape1[1]):
            # for z_1 in prange(my_shape1[2]):
                tmp_na_arr = np.zeros(shape=(my_shape2[0], my_shape2[1], my_shape1[2], my_shape2[3]))
                tmp_zero_arr = np.zeros(shape=(my_shape2[0], my_shape2[1], my_shape1[2], my_shape2[3]))
                for k_1 in range(my_shape1[2]):
                    my_index = bres_large_index_padded[i_1, j_1, k_1]
                    tmp_na_arr[:, :, k_1, :] = g_generated_aa_mat_padded_na_sliding_window[:, :, my_index, :]
                    tmp_zero_arr[:, :, k_1, :] = g_generated_aa_mat_padded_zero_sliding_window[:, :, my_index, :]

                for i_2 in range(tmp_na_arr.shape[0]):
                    for j_2 in range(tmp_na_arr.shape[1]):
                        for k_2 in range(tmp_na_arr.shape[3]):
                            mean_arr[i_1, i_2, j_2, j_1, k_2] = np.nanmean(tmp_na_arr[i_2, j_2, :, k_2])
                            tmp_zero_list = tmp_zero_arr[i_2, j_2, :, k_2]
                            tmp_zero_list = sorted(tmp_zero_list)
                            slope_arr[i_1, i_2, j_2, j_1, k_2] = fit_poly(np.arange(1, my_shape1[2] + 1), np.array(tmp_zero_list), 1)[0]

                        rsq_arr[i_1, i_2, j_2, j_1] = cal_rsq(mean_arr[i_1, i_2, j_2, j_1, :], bres_large_mean_padded[i_1, j_1, :])
                        # corr_matrixa = np.corrcoef(mean_arr[i_1, j_1, i_2, j_2, z_1, :], bres_large_mean_padded[i_1, j_1, z_1, :])
                        # corra = corr_matrixa[0, 1]
                        # rsq_arr[i_1, j_1, i_2, j_2, z_1] = 1 - (corra ** 2)

                        rmse_arr[i_1, i_2, j_2, j_1] = cal_rmse(mean_arr[i_1, i_2, j_2, j_1, :], mean_arr[i_1, i_2, j_2, j_1, :])
                        # rmse_arr[i_1, j_1, i_2, j_2, z_1] = euclidian_distance(mean_arr[i_1, j_1, i_2, j_2, z_1, :], mean_arr[i_1, j_1, i_2, j_2, z_1, :])
                        # rmse_arr[i_1, j_1, i_2, j_2, z_1] = mean_squared_error(mean_arr[i_1, j_1, i_2, j_2, z_1, :], mean_arr[i_1, j_1, i_2, j_2, z_1, :], squared=False)

    mean_arr = mean_arr.transpose(1, 2, 0, 3, 4)
    slope_arr = slope_arr.transpose(1, 2, 0, 3, 4)
    rsq_arr = rsq_arr.transpose(1, 2, 0, 3)
    rmse_arr = rmse_arr.transpose(1, 2, 0, 3)

    # mean_arr = mean_arr.transpose(2, 3, 0, 1, 4, 5)
    # slope_arr = slope_arr.transpose(2, 3, 0, 1, 4, 5)
    # rsq_arr = rsq_arr.transpose(2, 3, 0, 1, 4)
    # rmse_arr = rmse_arr.transpose(2, 3, 0, 1, 4)
    return mean_arr, slope_arr, rsq_arr, rmse_arr


@jit(nopython=True, parallel=True)
def get_padded_mean_slope_numba3(bres_large_index_padded, g_generated_aa_mat_padded_na_sliding_window, g_generated_aa_mat_padded_zero_sliding_window, bres_large_mean_padded, bres_large_slope_padded):
    my_shape1 = bres_large_index_padded.shape
    my_shape2 = g_generated_aa_mat_padded_na_sliding_window.shape
    mean_arr = np.zeros((my_shape1[0], my_shape2[0], my_shape2[1], my_shape1[1], my_shape2[3]), dtype=np.float64)
    slope_arr = np.zeros_like(mean_arr)
    rsq_arr = np.zeros((my_shape1[0], my_shape2[0], my_shape2[1], my_shape1[1]), dtype=np.float64)
    rmse_arr = np.zeros_like(rsq_arr)
    for i_1 in nb.prange(my_shape1[0]):
        for j_1 in range(my_shape1[1]):
            tmp_na_arr = np.zeros((my_shape2[0], my_shape2[1], my_shape1[2], my_shape2[3]), dtype=np.float64)
            tmp_zero_arr = np.zeros_like(tmp_na_arr)

            for k_1 in range(my_shape1[2]):
                my_index = bres_large_index_padded[i_1, j_1, k_1]
                tmp_na_arr[:, :, k_1, :] = g_generated_aa_mat_padded_na_sliding_window[:, :, my_index, :]
                tmp_zero_arr[:, :, k_1, :] = g_generated_aa_mat_padded_zero_sliding_window[:, :, my_index, :]

            # 利用NumPy向量化操作计算mean_arr和slope_arr
            for i_2 in range(tmp_na_arr.shape[0]):
                for j_2 in range(tmp_na_arr.shape[1]):
                    mean_arr_slice = tmp_na_arr[i_2, j_2, :, :]
                    mean_arr[i_1, i_2, j_2, j_1, :] = np.nanmean(mean_arr_slice, axis=2)

                    # 避免在循环中多次排序，可以考虑一次性排序后分片使用
                    if not np.all(np.isnan(tmp_zero_arr[i_2, j_2, :, :])):
                        sorted_zero_list = np.sort(tmp_zero_arr[i_2, j_2, :, :])
                        x = np.arange(1, my_shape1[2] + 1)[..., np.newaxis]
                        slopes = fit_poly(x, sorted_zero_list, degree=1)[:, 0]
                        slope_arr[i_1, i_2, j_2, j_1, :] = slopes

            # 计算rsq_arr和rmse_arr，由于依赖于之前的计算结果，不适合并行化
            for i_2 in range(tmp_na_arr.shape[0]):
                for j_2 in range(tmp_na_arr.shape[1]):
                    rsq_arr[i_1, i_2, j_2, j_1] = cal_rsq(mean_arr[i_1, i_2, j_2, j_1, :], bres_large_mean_padded[i_1, j_1, :])
                    rmse_arr[i_1, i_2, j_2, j_1] = cal_rmse(mean_arr[i_1, i_2, j_2, j_1, :], bres_large_mean_padded[i_1, j_1, :])

    # 调整维度顺序
    mean_arr = mean_arr.transpose(1, 2, 0, 3, 4)
    slope_arr = slope_arr.transpose(1, 2, 0, 3, 4)
    rsq_arr = rsq_arr.transpose(1, 2, 0, 3)
    rmse_arr = rmse_arr.transpose(1, 2, 0, 3)
    return mean_arr, slope_arr, rsq_arr, rmse_arr


def mutation_pad(g1_seq, n_generations, n_sample_per_generation, n_mutation, mut_index, aaprobset, aavals, bres_large_mean_padded, bres_large_slope_padded, bres_large_index_padded, resdir, tpctfrc, asslib="NO"):
    aavals_padded_zero = np.vstack([aavals, [0.0] * 20])
    aavals_padded_na = np.vstack([aavals, [np.nan] * 20])

    g1_seq = np.array(list(g1_seq))
    generated_aa_dict = {}
    g = 2

    sliding_window = [list(range(x - 6, x + 7)) for x in mut_index]

    subject_list = []
    for k in range(0, n_sample_per_generation * 2):
        subject = copy.copy(g1_seq)

        mut_choice = np.random.choice(VALID_AA_LIST, size=n_mutation, p=aaprobset)
        mut_index_sub = np.random.choice(mut_index, size=n_mutation, replace=False)
        np.put(subject, mut_index_sub, mut_choice)
        # generated_aa_dict.update({"g{}_p{}".format(g, k): subject})
        subject = list(subject)
        if subject not in subject_list and len(subject_list) < n_sample_per_generation:
            subject_list.append(subject)

    # for k in range(0, n_sample_per_generation):
    #     generated_aa_dict.update({"g{}_p{}".format(g, k): subject_list[k]})
    #
    # g_generated_aa_list = []
    # for k in range(0, n_sample_per_generation):
    #     g_generated_aa_list.append(generated_aa_dict['g{}_p{}'.format(g, k)])
    # g_generated_aa_list = np.stack(g_generated_aa_list, axis=0)
    g_generated_aa_list = np.stack(subject_list, axis=0)
    g_generated_aa_index = np.vectorize(lambda x: VALID_AA_LIST.index(x) if x in VALID_AA_LIST else -1)(g_generated_aa_list)

    g_generated_aa_mat_padded_zero = aavals_padded_zero[:, g_generated_aa_index]
    g_generated_aa_mat_padded_zero = g_generated_aa_mat_padded_zero.transpose(1, 0, 2)
    g_generated_aa_mat_padded_zero = np.concatenate((g_generated_aa_mat_padded_zero, g_generated_aa_mat_padded_zero[:, :, list(range(6))]), axis=2)
    g_generated_aa_mat_padded_zero_sliding_window = g_generated_aa_mat_padded_zero[:, :, sliding_window]
    g_generated_aa_mat_padded_zero_sliding_window = g_generated_aa_mat_padded_zero_sliding_window.transpose(0, 2, 1, 3)

    g_generated_aa_mat_padded_na = aavals_padded_na[:, g_generated_aa_index]
    g_generated_aa_mat_padded_na = g_generated_aa_mat_padded_na.transpose(1, 0, 2)
    g_generated_aa_mat_padded_na = np.concatenate((g_generated_aa_mat_padded_na, g_generated_aa_mat_padded_na[:, :, list(range(6))]), axis=2)
    g_generated_aa_mat_padded_na_sliding_window = g_generated_aa_mat_padded_na[:, :, sliding_window]
    g_generated_aa_mat_padded_na_sliding_window = g_generated_aa_mat_padded_na_sliding_window.transpose(0, 2, 1, 3)

    bres_large_index_padded = pad_index(bres_large_index_padded)

    start = time.time()
    # get_padded_mean_slope(bres_large_index_padded, g_generated_aa_mat_padded_na_sliding_window, g_generated_aa_mat_padded_zero_sliding_window)
    # func_nb_mt = make_multithread(get_padded_mean_slope_numba, N_THREADINGS)
    g_generated_aa_mean, g_generated_aa_slope, g_generated_aa_rsq, g_generated_aa_rsme = get_padded_mean_slope_numba(bres_large_index_padded, g_generated_aa_mat_padded_na_sliding_window, g_generated_aa_mat_padded_zero_sliding_window, bres_large_mean_padded, bres_large_slope_padded)
    all_delta_mean = np.sum(np.abs(g_generated_aa_mean - bres_large_mean_padded), axis=5)
    all_delta_slope = np.sum(np.abs(g_generated_aa_slope - bres_large_slope_padded), axis=5)
    all_sse = np.sum(np.power(g_generated_aa_mean - bres_large_mean_padded, 2), axis=5)

    all_scores = all_delta_mean + all_delta_slope + all_sse + g_generated_aa_rsq + g_generated_aa_rsme
    all_scores = np.sum(all_scores, axis=(2, 4))
    all_scores = all_scores.min(axis=(1, 2))
    all_scores_sorted = np.sort(all_scores)
    all_scores_index = np.argsort(all_scores)

    bestmutset = all_scores_sorted

    aa_name = ["g{}_p{}".format(g, x) for x in all_scores_index]
    g_generated_aa_df = pd.DataFrame(g_generated_aa_list)
    g_generated_aa_df = g_generated_aa_df.apply(lambda x: ''.join(x), axis=1)
    g_generated_aa_df = g_generated_aa_df.to_frame(name="aa_seq")
    g_generated_aa_df = g_generated_aa_df.iloc[all_scores_index]
    g_generated_aa_df["aa_name"] = aa_name
    g_generated_aa_df["aa_score"] = all_scores_sorted

    g_generated_aa_df.iloc[:tpctfrc, ].to_csv(os.path.join(".", 'g{}variantscores.tsv'.format(g)), sep="\t", header=True, index=False, columns=["aa_name", "aa_seq", "aa_score"])

    end = time.time()
    print("numba time cost: {}s".format(end - start))

    # g_generated_aa_mean - bres_large_mean_padded
    start = time.time()
    # g_generated_aa_mean2, g_generated_aa_slope2 = get_mean_slope_with_vectorized(bres_large_index_padded, g_generated_aa_mat_padded_na_sliding_window, g_generated_aa_mat_padded_zero_sliding_window)
    end = time.time()
    print("numpy vectorized time cost: {}s".format(end - start))


    # print(g_generated_aa_mean - g_generated_aa_mean2)
    # print(g_generated_aa_slope - g_generated_aa_slope2)

    # start = time.time()
    # taichi_test(bres_large_index, g_generated_aa_mat1_in_window)
    # end = time.time()
    # print("taichi time cost: {}s".format(end - start))


def mutation_pad1(g1_seq, n_generations, n_sample_per_generation, n_mutation, mut_index, aaprobset, aavals, bres_large_mean_padded, bres_large_slope_padded, bres_large_index_padded, resdir, tpctfrc, asslib="NO"):
    aavals_padded_zero = np.vstack([aavals, [0.0] * 20])
    aavals_padded_na = np.vstack([aavals, [np.nan] * 20])

    g1_seq = np.array(list(g1_seq))
    generated_aa_dict = {}
    g = 2

    sliding_window = [list(range(x - 6, x + 7)) for x in mut_index]

    subject_list = []
    for k in range(0, n_sample_per_generation * 2):
        subject = copy.copy(g1_seq)

        mut_choice = np.random.choice(VALID_AA_LIST, size=n_mutation, p=aaprobset)
        mut_index_sub = np.random.choice(mut_index, size=n_mutation, replace=False)
        np.put(subject, mut_index_sub, mut_choice)
        # generated_aa_dict.update({"g{}_p{}".format(g, k): subject})
        subject = list(subject)
        if subject not in subject_list and len(subject_list) < n_sample_per_generation:
            subject_list.append(subject)

    # for k in range(0, n_sample_per_generation):
    #     generated_aa_dict.update({"g{}_p{}".format(g, k): subject_list[k]})
    #
    # g_generated_aa_list = []
    # for k in range(0, n_sample_per_generation):
    #     g_generated_aa_list.append(generated_aa_dict['g{}_p{}'.format(g, k)])
    # g_generated_aa_list = np.stack(g_generated_aa_list, axis=0)
    g_generated_aa_list = np.stack(subject_list, axis=0)
    g_generated_aa_index = np.vectorize(lambda x: VALID_AA_LIST.index(x) if x in VALID_AA_LIST else -1)(g_generated_aa_list)

    g_generated_aa_mat_padded_zero = aavals_padded_zero[:, g_generated_aa_index]
    g_generated_aa_mat_padded_zero = g_generated_aa_mat_padded_zero.transpose(1, 0, 2)
    g_generated_aa_mat_padded_zero = np.concatenate((g_generated_aa_mat_padded_zero, g_generated_aa_mat_padded_zero[:, :, list(range(6))]), axis=2)
    g_generated_aa_mat_padded_zero_sliding_window = g_generated_aa_mat_padded_zero[:, :, sliding_window]
    g_generated_aa_mat_padded_zero_sliding_window = g_generated_aa_mat_padded_zero_sliding_window.transpose(0, 2, 1, 3)

    g_generated_aa_mat_padded_na = aavals_padded_na[:, g_generated_aa_index]
    g_generated_aa_mat_padded_na = g_generated_aa_mat_padded_na.transpose(1, 0, 2)
    g_generated_aa_mat_padded_na = np.concatenate((g_generated_aa_mat_padded_na, g_generated_aa_mat_padded_na[:, :, list(range(6))]), axis=2)
    g_generated_aa_mat_padded_na_sliding_window = g_generated_aa_mat_padded_na[:, :, sliding_window]
    g_generated_aa_mat_padded_na_sliding_window = g_generated_aa_mat_padded_na_sliding_window.transpose(0, 2, 1, 3)

    bres_large_index_padded = pad_index(bres_large_index_padded)

    start = time.time()
    # get_padded_mean_slope(bres_large_index_padded, g_generated_aa_mat_padded_na_sliding_window, g_generated_aa_mat_padded_zero_sliding_window)
    # func_nb_mt = make_multithread(get_padded_mean_slope_numba, N_THREADINGS)
    bres_large_index_padded_reshape = bres_large_index_padded.reshape(-1, bres_large_index_padded.shape[2], bres_large_index_padded.shape[3])
    bres_large_mean_padded_reshape = bres_large_mean_padded.reshape(-1, bres_large_mean_padded.shape[2], bres_large_mean_padded.shape[3])
    bres_large_slope_padded_reshape = bres_large_slope_padded.reshape(-1, bres_large_slope_padded.shape[2], bres_large_slope_padded.shape[3])

    g_generated_aa_mean, g_generated_aa_slope, g_generated_aa_rsq, g_generated_aa_rsme = get_padded_mean_slope_numba1(bres_large_index_padded_reshape, g_generated_aa_mat_padded_na_sliding_window, g_generated_aa_mat_padded_zero_sliding_window, bres_large_mean_padded_reshape, bres_large_slope_padded_reshape)
    get_padded_mean_slope_numba1.parallel_diagnostics(level=4)
    g_generated_aa_mean = g_generated_aa_mean.reshape(g_generated_aa_mean.shape[0], g_generated_aa_mean.shape[1], bres_large_index_padded.shape[0], bres_large_index_padded.shape[1], g_generated_aa_mean.shape[3], g_generated_aa_mean.shape[4])
    g_generated_aa_slope = g_generated_aa_slope.reshape(g_generated_aa_slope.shape[0], g_generated_aa_slope.shape[1], bres_large_index_padded.shape[0], bres_large_index_padded.shape[1], g_generated_aa_slope.shape[3], g_generated_aa_slope.shape[4])
    g_generated_aa_rsq = g_generated_aa_rsq.reshape(g_generated_aa_rsq.shape[0], g_generated_aa_rsq.shape[1], bres_large_index_padded.shape[0], bres_large_index_padded.shape[1], g_generated_aa_rsq.shape[3])
    g_generated_aa_rsme = g_generated_aa_rsme.reshape(g_generated_aa_rsme.shape[0], g_generated_aa_rsme.shape[1], bres_large_index_padded.shape[0], bres_large_index_padded.shape[1], g_generated_aa_rsme.shape[3])

    all_delta_mean = np.sum(np.abs(g_generated_aa_mean - bres_large_mean_padded), axis=5)
    all_delta_slope = np.sum(np.abs(g_generated_aa_slope - bres_large_slope_padded), axis=5)
    all_sse = np.sum(np.power(g_generated_aa_mean - bres_large_mean_padded, 2), axis=5)

    all_scores = all_delta_mean + all_delta_slope + all_sse + g_generated_aa_rsq + g_generated_aa_rsme
    all_scores = np.sum(all_scores, axis=(2, 4))
    all_scores = all_scores.min(axis=(1, 2))
    all_scores_sorted = np.sort(all_scores)
    all_scores_index = np.argsort(all_scores)

    bestmutset = all_scores_sorted

    aa_name = ["g{}_p{}".format(g, x) for x in all_scores_index]
    g_generated_aa_df = pd.DataFrame(g_generated_aa_list)
    g_generated_aa_df = g_generated_aa_df.apply(lambda x: ''.join(x), axis=1)
    g_generated_aa_df = g_generated_aa_df.to_frame(name="aa_seq")
    g_generated_aa_df = g_generated_aa_df.iloc[all_scores_index]
    g_generated_aa_df["aa_name"] = aa_name
    g_generated_aa_df["aa_score"] = all_scores_sorted

    g_generated_aa_df.iloc[:tpctfrc, ].to_csv(os.path.join(".", 'g{}variantscores.tsv'.format(g)), sep="\t", header=True, index=False, columns=["aa_name", "aa_seq", "aa_score"])

    end = time.time()
    print("numba time cost: {}s".format(end - start))

    # g_generated_aa_mean - bres_large_mean_padded
    start = time.time()
    # g_generated_aa_mean2, g_generated_aa_slope2 = get_mean_slope_with_vectorized(bres_large_index_padded, g_generated_aa_mat_padded_na_sliding_window, g_generated_aa_mat_padded_zero_sliding_window)
    end = time.time()
    print("numpy vectorized time cost: {}s".format(end - start))


    # print(g_generated_aa_mean - g_generated_aa_mean2)
    # print(g_generated_aa_slope - g_generated_aa_slope2)

    # start = time.time()
    # taichi_test(bres_large_index, g_generated_aa_mat1_in_window)
    # end = time.time()
    # print("taichi time cost: {}s".format(end - start))


def main():
    # user_input_dir = str(sys.argv[1])
    user_input_dir = "./"
    # user_input_dir = "/home/xufeng/xufeng/software/Proseeker/proseeker_for_hd"
    # user_input_dir = "/data1/xufeng/Software/Proseeker/"
    # valid_aa_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    collist = ['g', 'k', 'MEG', 'block', 'g1', 'pchoice', *VALID_AA_LIST, 'truncres', 'cores', 'sites', 'bres', 'fmode',
               'fthresh', 'memorylimit', 'memorythresh', 'bresnum']

    pd.set_option("display.max_colwidth", 10000)
    np.random.seed(31)
    # jobstart = pd.read_csv(os.path.join(user_input_dir, 'jobstart_hd_test_1.csv'), usecols=collist)
    jobstart = pd.read_csv(os.path.join(user_input_dir, 'jobstart_hd_test.csv'), usecols=collist)
    # jobstart = pd.read_csv(os.path.join(user_input_dir, 'jobstart_hd.csv'), usecols=collist)
    # jobstart = pd.read_csv(os.path.join(user_input_dir, 'jobstart_test.csv'), usecols=collist)
    resdir = os.path.join(user_input_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    job_param = jobstart.loc[0,]
    bresstr = str(job_param.bres)
    bresdir = os.path.join(user_input_dir, bresstr)
    bresnum = int(job_param.bresnum)
    generations = int(job_param.g)
    wide = int(job_param.k)
    MEG = int(job_param.MEG)
    block = str(job_param.block)
    g1 = str(job_param.g1)
    pchoice = int(job_param.pchoice)
    trunc = int(job_param.truncres)
    cores = int(job_param.cores)
    sites = int(job_param.sites)
    asslib = str(job_param.fmode)
    asslibthresh = int(job_param.fthresh)
    if asslib == "YES":
        wide = asslibthresh
    memlots = str(job_param.memorylimit)
    memthresh = int(job_param.memorythresh)

    if pchoice == 1:
        aaprobset = jobstart[VALID_AA_LIST]
        aaprobset = aaprobset.values.tolist()
        aaprobset = [item for sublist in aaprobset for item in sublist]
    else:
        aaprobset = [0.05] * len(VALID_AA_LIST)

    # os.mkdir(resdir)
    # shutil.copyfile(os.path.join(user_input_dir, 'jobstart_hd.csv'), os.path.join(resdir, 'used_jobstart.csv'))

    # bres_large_mean, bres_large_slope, bres_large_index = get_assessment_info1(sites, bresnum, bresdir, aaprobset)
    bres_large_mean_padded, bres_large_slope_padded, bres_large_index_padded = get_assessment_info_with_pad(sites, bresnum, bresdir)

    aavals = pd.read_csv(os.path.join(user_input_dir, 'ranking.csv'), usecols=VALID_AA_LIST, sep=',')
    aavals = normalize_aavals_numpy(aavals)

    mut_index = get_character_index(block, "M")

    tpctfrc = 0
    if trunc > 0:
        tpctfrc = int(round(wide / trunc))
    start = time.time()
    # g1_seq, n_generations, n_sample_per_generation, n_mutation = g1, generations, wide, MEG
    # mutation_pad(g1, generations, wide, MEG, mut_index, aaprobset, aavals, bres_large_mean_padded, bres_large_slope_padded, bres_large_index_padded, resdir, tpctfrc, asslib="NO")
    mutation_pad1(g1, generations, wide, MEG, mut_index, aaprobset, aavals, bres_large_mean_padded, bres_large_slope_padded, bres_large_index_padded, resdir, tpctfrc, asslib="NO")
    end = time.time()
    print("mutation time cost: {}s".format(end - start))

    # start = time.time()
    # mutation1(g1, generations, wide, MEG, mut_index, aaprobset, asslib="NO")
    # end = time.time()
    # print("mutation1 time cost: {}s".format(end - start))


if __name__ == '__main__':
    main()