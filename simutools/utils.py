#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
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

    g = nx.Graph()
    for i, l in enumerate(list_of_lists):
        g.add_node(i)

    for i, l1 in enumerate(list_of_lists):
        for j in range(i+1, len(list_of_lists)):
            l2 = list_of_lists[j]
            if set(l1).intersection(l2):
                g.add_edge(i, j)

    merged_lists = []
    raw_merged_lists = []
    for idx in list(nx.connected_components(g)):
        _ = []
        for i in idx:
            _ += list_of_lists[i]
        merged_lists.append(list(set(_)))
        raw_merged_lists.append([list_of_lists[i] for i in idx])
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


def generate_slurm(name: str, walltime: int = 168, nodes: str = 'cpu', gpu: int = 0, mem: int = None,
                   ntasks: int = None, commands: List[str] = None, exclude: str = None):
    file = open('%s.sh' % name, 'w')
    info = '#!/bin/bash\n'
    info += '#SBATCH -D %s\n' % os.getcwd()
    info += '#SBATCH -N 1\n'
    info += '#SBATCH --job-name=%s\n' % name
    if mem is not None:
        info += '#SBATCH --mem=%dG\n' % mem
    info += '#SBATCH -o %s.out\n' % name
    info += '#SBATCH -e %s.err\n' % name
    info += '#SBATCH --partition=%s\n' % nodes
    if ntasks is not None:
        info += '#SBATCH --ntasks=%i\n' % ntasks
    if gpu != 0:
        info += '#SBATCH --gres=gpu:%i\n' % gpu
        info += '#SBATCH --exclusive\n'
    if exclude is not None:
        info += '#SBATCH --exclude=%s\n' % exclude
    info += '#SBATCH --time=%i:0:0\n' % walltime
    # info += 'source ~/.zshrc\n'
    info += 'echo $SLURM_NODELIST > %s.time\n' % name
    info += 'echo $(date) >> %s.time\n' % name
    for cmd in commands:
        info += cmd + '\n'
    info += '\necho $(date) >> %s.time' % name
    file.write(info)
