#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: common_functions.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2024-01-24 14:36:22
Last modified: 2024-01-24 14:36:22
'''
import os


def get_character_index(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def normalize_aavals_numpy(aavals):
    aavals = aavals.to_numpy()
    min_values = aavals.min(axis=1, keepdims=True)
    max_values = aavals.max(axis=1, keepdims=True)
    normalize_aavals = (aavals - min_values) / (max_values - min_values)
    return normalize_aavals


def read_file_as_list(input_file):
    lines_list = []
    with open(input_file, 'r') as file:
        for line in file:
            lines_list.append(line.strip())
    return lines_list


def resolveDir(dirName, chdir=False):
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    if chdir:
        os.chdir(dirName)
