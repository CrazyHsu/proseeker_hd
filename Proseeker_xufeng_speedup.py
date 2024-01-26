#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: Proseeker_xufeng.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2024-01-06 12:54:49
Last modified: 2024-01-06 12:54:49
'''

import sys, time, copy
# import subprocess
# import pkg_resources
import os
from datetime import datetime
# import random
import warnings
# import csv
# import shutil
# import math
# import statistics
import numpy as np
import pandas as pd
import numba
from numba import jit, types
# from sklearn.metrics import mean_squared_error
# from scipy.stats import shapiro
# from itertools import product
# from scipy.stats import kstest
# import logomaker as lm
# import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

VALID_AA_LIST = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


def get_character_index(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def normalize_aavals_plain_python(aavals):
    for i in range(0, 544):
        for j in range(0, 20):
            rowmin = min(aavals.iloc[i])
            rowmax = max(aavals.iloc[i])
            val = aavals.iloc[i, j]
            aavals.replace([aavals.iloc[i, j]], (val - rowmin) / (rowmax - rowmin))
    return aavals


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


def get_assessment_info_cpu(sites, bresnum, bresdir, aaprobset):
    d = dict()
    d['aaprobset'] = aaprobset
    d1 = {}
    for bs in range(1, sites + 1):
        for x in range(1, bresnum + 1):
            bres = pd.read_csv(os.path.join(bresdir, 'p{}.bres{}.csv'.format(x, bs)), header=None, sep=',')
            d['b{}.bres{}.csv'.format(x, bs)] = bres
            d['b{}.bres{}.ind'.format(x, bs)] = list(bres.iloc[:, 13])

            for v in range(0, 13):
                d1['col{}'.format(v)] = list(bres.iloc[:, v])

            for y in range(0, 50):
                set1 = [index for index, element in enumerate(d['b{}.bres{}.ind'.format(x, bs)]) if element == y]
                meanset1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                slopeset1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                for z in range(0, 13):
                    colsel1 = d1['col{}'.format(z)]
                    sub1 = [colsel1[i] for i in set1]
                    meanset1[z] = sum(sub1) / len(sub1)
                    slopex1 = list.copy(sub1)
                    for q in range(1, len(slopex1) + 1):
                        slopex1[q - 1] = q
                    sub1.sort()
                    m, b = np.polyfit(slopex1, sub1, 1)
                    slopeset1[z] = m
                d['b{}.bres{}.c{}'.format(x, bs, y)] = meanset1
                d['b{}.cslopes{}.c{}'.format(x, bs, y)] = slopeset1


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


def get_assessment_info(sites, bresnum, bresdir, aaprobset):
    d = dict()
    d['aaprobset'] = aaprobset
    d1 = {}
    bres_final_mean = []
    bres_final_slope = []
    for bs in range(1, sites + 1):
        bres_mean_list = []
        bres_slope_list = []
        for x in range(1, bresnum + 1):
            bres = pd.read_csv(os.path.join(bresdir, 'p{}.bres{}.csv'.format(x, bs)), header=None, sep=',')
            bres_mean = bres.groupby(13).mean()
            bres_slope = bres.groupby(13).apply(calculate_slope)
            bres_mean_list.append(bres_mean.values)
            bres_slope_list.append(np.stack(bres_slope, axis=0))
        bres_final_mean.append(np.stack(bres_mean_list, axis=0))
        bres_final_slope.append(np.stack(bres_slope_list, axis=0))
    bres_final_mean = np.stack(bres_final_mean, axis=0)
    bres_final_slope = np.stack(bres_final_slope, axis=0)

            # d['b{}.bres{}.csv'.format(x, bs)] = bres
            # d['b{}.bres{}.ind'.format(x, bs)] = list(bres.iloc[:, 13])
            #
            # for v in range(0, 13):
            #     d1['col{}'.format(v)] = list(bres.iloc[:, v])
            #
            # for y in range(0, 50):
            #     set1 = [index for index, element in enumerate(d['b{}.bres{}.ind'.format(x, bs)]) if element == y]
            #     meanset1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #     slopeset1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #
            #     for z in range(0, 13):
            #         colsel1 = d1['col{}'.format(z)]
            #         sub1 = [colsel1[i] for i in set1]
            #         meanset1[z] = sum(sub1) / len(sub1)
            #         slopex1 = list.copy(sub1)
            #         for q in range(1, len(slopex1) + 1):
            #             slopex1[q - 1] = q
            #         sub1.sort()
            #         m, b = np.polyfit(slopex1, sub1, 1)
            #         slopeset1[z] = m
            #     d['b{}.bres{}.c{}'.format(x, bs, y)] = meanset1
            #     d['b{}.cslopes{}.c{}'.format(x, bs, y)] = slopeset1


def get_mean_slope(bres_large_index, g_generated_aa_mat1_in_window):
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


def mutation(g1_seq, n_generations, n_sample_per_generation, n_mutation, mut_index, aaprobset, aavals, bres_large_mean, bres_large_slope, bres_large_index, asslib="NO"):
    bestmutset = []
    g1_seq = np.array(list(g1_seq))
    # aavals = aavals.to_numpy()
    generated_aa_dict = {}
    for g in range(2, n_generations + 1):
        klstindx = 0

        if asslib != "YES":
            print('Gen {} mutagenesis commenced'.format(g))
            for k in range(0, n_sample_per_generation):
                if g == 2:
                    subject = copy.copy(g1_seq)
                else:
                    break
                    if k / 10 >= klstindx + 1:
                        klstindx += 1
                    subject = bestmutset[klstindx]

                mut_choice = np.random.choice(VALID_AA_LIST, size=n_mutation, p=aaprobset)
                np.put(subject, mut_index, mut_choice)
                generated_aa_dict.update({"g{}_p{}".format(g, k): subject})

        # VARIANT ASSESSMENT
        if asslib != "YES":
            print('Gen {} assessment commenced'.format(g))
        else:
            print("Library Assessment Commenced")
            wide = len(comblistmaster)

        g_generated_aa_list = []
        for k in range(0, n_sample_per_generation):
            # if asslib == "YES":
            #     print("Now assessing variant {} of {}".format(k + 1, n_sample_per_generation))
            g_generated_aa_list.append(generated_aa_dict['g{}_p{}'.format(g, k)])
        g_generated_aa_list = np.stack(g_generated_aa_list, axis=0)
        g_generated_aa_index = np.vectorize(lambda x: VALID_AA_LIST.index(x) if x in VALID_AA_LIST else -1)(g_generated_aa_list)
        g_generated_aa_mat = aavals[:, g_generated_aa_index]
        g_generated_aa_mat1 = g_generated_aa_mat.transpose(1, 0, 2)
        g_generated_aa_mat1 = np.concatenate((g_generated_aa_mat1, g_generated_aa_mat1[:, :, list(range(6))]), axis=2)

        windows = [list(range(x-6, x+7)) for x in mut_index]
        g_generated_aa_mat1_in_window = g_generated_aa_mat1[:, :, windows]
        g_generated_aa_mat1_in_window = g_generated_aa_mat1_in_window.transpose(0, 2, 1, 3)

        for i, indices_group in enumerate(bres_large_index):
            if i > 1: break
            for j, indices in enumerate(indices_group):
                if j > 1: break
                for z, tmp_index in enumerate(indices):
                    g_generated_aa_mat1_in_window_tmp = g_generated_aa_mat1_in_window[:, :, np.array(tmp_index), :]
                    mean = g_generated_aa_mat1_in_window_tmp.mean()
                    transposed_arr = g_generated_aa_mat1_in_window_tmp.transpose((2, 3, 0, 1))
                    sorted_transposed_arr = np.sort(transposed_arr, axis=0)
                    sorted_original_shape_arr = sorted_transposed_arr.transpose((2, 3, 0, 1))
                    indices_ = np.arange(1, sorted_original_shape_arr.shape[2] + 1)
                    slopes = np.apply_along_axis(lambda x: np.polyfit(indices_, x, 1)[0], axis=2, arr=sorted_original_shape_arr)

        # max(len(indices) for indices_group in bres_large_index for indices in indices_group)
        # [for indices_group in bres_large_index for indices in  indices_group]

        # np.vectorize(lambda indices: bres_large_index[:, :, indices])()
        # extract_func = np.frompyfunc(lambda indices: g_generated_aa_mat1_in_window[:2, :2, indices, :], 1, 1)
        # sub_arrays = extract_func(bres_large_index[:3, :3, ])

        # for x in mut_index:
        #     window = list(range(x-6, x+7))


        # var = list(generated_aa_dict['g{}_p{}'.format(g, k)])

    # return generated_aa_dict
                # print(subject)
                # var = list.copy(subject)
                #
                # for emeg in range(0, n_mutation):
                #     random.shuffle(mutinds)
                #
                #     mutchoice = np.random.choice(VALID_AA_LIST, size=3, p=aaprobset)
                #
                #     choice = int(mutinds[0])
                #     var[choice] = mutchoice
                #
                # d['g{}p{}'.format(g, k)] = var
                # d['g{}p{}'.format(g, k)] = [item for sublist in d['g{}p{}'.format(g, k)] for item in sublist]
                # del (var)


def mutation1(g1_seq, n_generations, n_sample_generation, n_mutation, mut_index, aaprobset, asslib="NO"):
    g1_seq = list(g1_seq)
    import random
    bestmutset = []
    d = {}
    for g in range(2, n_generations + 1):
        klstindx = 0

        if asslib != "YES":
            print('Gen {} mutagenesis commenced'.format(g))
            for k in range(0, n_sample_generation):
                if g == 2:
                    subject = list.copy(g1_seq)
                    # subject = subject[5:len(subject) - 24]
                else:
                    break
                    if k / 10 >= klstindx + 1:
                        klstindx += 1
                    subject = bestmutset[klstindx]

                var = list.copy(subject)

                for emeg in range(0, n_mutation):
                    random.shuffle(mut_index)

                    mutchoice = np.random.choice(
                        ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
                         'S', 'T', 'W', 'Y', 'V'], 1, p=aaprobset)

                    choice = int(mut_index[0])
                    var[choice] = mutchoice

                d['g{}p{}'.format(g, k)] = var
                d['g{}p{}'.format(g, k)] = [item for sublist in d['g{}p{}'.format(g, k)] for item in sublist]
                del (var)


def main():
    # user_input_dir = str(sys.argv[1])
    # user_input_dir = "/home/xufeng/xufeng/software/Proseeker/proseeker_for_hd"
    user_input_dir = "./"
    # valid_aa_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    collist = ['g', 'k', 'MEG', 'block', 'g1', 'pchoice', *VALID_AA_LIST, 'truncres', 'cores', 'sites', 'bres', 'fmode',
               'fthresh', 'memorylimit', 'memorythresh', 'bresnum']

    pd.set_option("display.max_colwidth", 10000)
    jobstart = pd.read_csv(os.path.join(user_input_dir, 'jobstart_hd_test.csv'), usecols=collist)
    # jobstart = pd.read_csv(os.path.join(user_input_dir, 'jobstart_hd.csv'), usecols=collist)
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

    # cpu_start = time.time()
    # get_assessment_info(sites, bresnum, bresdir, aaprobset)
    # cpu_end = time.time()
    # print("numpy1 time cost: {}s".format(cpu_end - cpu_start))
    #
    # cpu_start = time.time()
    bres_large_mean, bres_large_slope, bres_large_index = get_assessment_info1(sites, bresnum, bresdir, aaprobset)
    # cpu_end = time.time()
    # print("numpy2 time cost: {}s".format(cpu_end - cpu_start))
    #
    # cpu_start = time.time()
    # get_assessment_info_cpu(sites, bresnum, bresdir, aaprobset)
    # cpu_end = time.time()
    # print("cpu time cost: {}s".format(cpu_end - cpu_start))

    aavals = pd.read_csv(os.path.join(user_input_dir, 'ranking.csv'), usecols=VALID_AA_LIST, sep=',')

    # cpu_start = time.time()
    # normalize_aavals_plain_python(aavals)
    # cpu_end = time.time()
    # print("aavals normalization plain python time cost: {}s".format(cpu_end - cpu_start))

    # cpu_start = time.time()
    aavals = normalize_aavals_numpy(aavals)
    # cpu_end = time.time()
    # print("aavals normalization numpy time cost: {}s".format(cpu_end - cpu_start))

    mut_index = get_character_index(block, "M")

    start = time.time()
    g1_seq, n_generations, n_sample_per_generation, n_mutation = g1, generations, wide, MEG
    mutation(g1, generations, wide, MEG, mut_index, aaprobset, aavals, bres_large_mean, bres_large_slope, bres_large_index, asslib="NO")
    end = time.time()
    print("mutation time cost: {}s".format(end - start))

    # start = time.time()
    # mutation1(g1, generations, wide, MEG, mut_index, aaprobset, asslib="NO")
    # end = time.time()
    # print("mutation1 time cost: {}s".format(end - start))


if __name__ == '__main__':
    main()
