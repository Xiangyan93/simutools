#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
from subprocess import Popen, PIPE
import os
from collections import Counter
import networkx as nx
import numpy as np
from scipy.optimize import curve_fit


def execute(cmd: str, input: str = None):
    if input is not None:
        print(input, ' | ', cmd)
    else:
        print(cmd)
    try:
        if input is not None:
            input_process = Popen(['echo']+input.split(), stdout=PIPE, text=True)
            cmd_process = Popen(cmd.split(), stdin=input_process.stdout, stdout=PIPE, stderr=PIPE, text=True)
            input_process.stdout.close()
        else:
            cmd_process = Popen(cmd.split(), stdout=PIPE, stderr=PIPE)
    except:
        raise RuntimeError(f'Error when execute command: {cmd}')
    return cmd_process.communicate()


def cd_and_mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)


def all_same(items, tol=1e-6):
    """
    Returns True if all the items in the list are the same up to the given tolerance level, otherwise False.
    """
    return all(abs(x - items[0]) < tol for x in items)


def find_index_of_max_unique_abs(numbers):
    counts = Counter(numbers)
    abs_sorted_numbers = sorted(numbers, key=abs, reverse=True)
    for val in abs_sorted_numbers:
        if counts[val] == min(counts.values()):
            v = val
            break
    return [i for i, x in enumerate(numbers) if x == v]


def merge_lists(list_of_lists):
    if not list_of_lists:
        return []

    merged_lists = []
    raw_merged_lists = []
    visited = set()

    for i in range(len(list_of_lists)):
        if i in visited:
            continue

        merged_list = set(list_of_lists[i])
        raw_merged_lists.append([])
        raw_merged_lists[-1].append(list_of_lists[i])
        visited.add(i)

        for j in range(i+1, len(list_of_lists)):
            if j in visited:
                continue

            if any(elem in merged_list for elem in list_of_lists[j]):
                merged_list.update(list_of_lists[j])
                raw_merged_lists[-1].append(list_of_lists[j])
                visited.add(j)

        merged_lists.append(list(merged_list))

    return merged_lists, raw_merged_lists


def mol_to_nx(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum(), atom_symbol=atom.GetSymbol())

    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType())
    return G


def curve_fit_rsq(f, xdata, ydata, guess=None, bounds=None, weights=None) -> (float, float):
    x_array = np.array(xdata)
    y_array = np.array(ydata)
    if weights is not None:
        weights = 1 / np.array(weights)
    if guess is None and bounds is not None:
        popt, pcov = curve_fit(f, x_array, y_array, bounds=bounds, sigma=weights)
    elif guess is not None and bounds is None:
        popt, pcov = curve_fit(f, x_array, y_array, p0=guess, sigma=weights)
    elif guess is None and bounds is None:
        popt, pcov = curve_fit(f, x_array, y_array, sigma=weights)
    else:
        popt, pcov = curve_fit(f, x_array, y_array, guess, bounds=bounds, sigma=weights)
    ss_tot = ((y_array - y_array.mean()) ** 2).sum()
    predict = np.array([f(x, *popt) for x in x_array])
    ss_res = ((y_array - predict) ** 2).sum()
    rsq = 1 - ss_res / ss_tot

    return popt, rsq
