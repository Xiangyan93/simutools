#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
import random
import string
from collections import Counter
from subprocess import Popen, PIPE

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


def random_string(length: int = 8):
    return ''.join(random.sample(string.ascii_letters, length))


def estimate_density_from_formula(f) -> float:
    # unit: g/mL
    from .formula import Formula
    formula = Formula.read(f)
    string = formula.to_str()
    density = {
        'H2': 0.07,
        'He': 0.15,
    }
    if string in density.keys():
        return density.get(string)

    nAtoms = formula.n_heavy + formula.n_h
    nC = formula.atomdict.get('C') or 0
    nH = formula.atomdict.get('H') or 0
    nO = formula.atomdict.get('O') or 0
    nN = formula.atomdict.get('N') or 0
    nS = formula.atomdict.get('S') or 0
    nF = formula.atomdict.get('F') or 0
    nCl = formula.atomdict.get('Cl') or 0
    nBr = formula.atomdict.get('Br') or 0
    nI = formula.atomdict.get('I') or 0
    nOther = nAtoms - nC - nH - nO - nN - nS - nF - nCl - nBr - nI
    return (1.175 * nC + 0.572 * nH + 1.774 * nO + 1.133 * nN + 2.184 * nS
            + 1.416 * nF + 2.199 * nCl + 5.558 * nBr + 7.460 * nI
            + 0.911 * nOther) / nAtoms


def random_res_name() -> str:
    """Get the random residue name."""
    return ''.join(random.choice(string.ascii_uppercase) for _ in range(3))


def random_res_name_list(n: int) -> List[str]:
    """ Get a list of random residue name.

    Parameters
    ----------
    n

    Returns
    -------

    """
    res_name_list = []
    while True:
        res_name = random_res_name()
        if res_name not in res_name_list:
            res_name_list.append(res_name)
        if len(res_name_list) == n:
            return res_name_list
