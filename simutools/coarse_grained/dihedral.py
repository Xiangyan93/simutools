#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import numpy as np
import pandas as pd
import warnings
from .bead import Bead
from ..utils import curve_fit_rsq


class Dihedral:
    def __init__(self, bead1: Bead, bead2: Bead, bead3: Bead, bead4: Bead, idx: int = None):
        self.bead1 = bead1
        self.bead2 = bead2
        self.bead3 = bead3
        self.bead4 = bead4
        self.idx = idx

    def __eq__(self, other):
        return sorted(self.bead_idx) == sorted(other.bead_idx)

    @property
    def bead_idx(self):
        return [self.bead1.idx, self.bead2.idx, self.bead3.idx, self.bead4.idx]

    @property
    def n_iter(self) -> int:
        for i in range(1000):
            if f'p_cg_{i}' not in self.df_dist:
                return i

    def get_aa_distribution(self, T: int = 298, tag: str = '', fitting: Literal['least_square', 'simple'] = 'simple',
                            f_idx=None):
        dihedrals_rad_traj = self.calculate_dihedral(self.bead1.com, self.bead2.com, self.bead3.com, self.bead4.com)
        dihedrals_degree_traj = np.degrees(dihedrals_rad_traj)
        hist, bin_edges = np.histogram(dihedrals_rad_traj, bins=90, range=[0, 2 * np.pi], density=True)
        bin_edges = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
        df = pd.DataFrame({'time': self.bead1.time, 'dihedral': dihedrals_degree_traj})
        df.to_csv(f'dihedral_{self.idx + 1}{tag}.xvg', header=False, index=False, sep='\t')
        self.df_dist = pd.DataFrame({'dihedral': np.degrees(bin_edges), 'p_aa': hist})
        if fitting == 'least_square':
            self.beta = 1000 / 8.314 / T
            self.k1_aa, self.s1_aa, self.k2_aa, self.s2_aa, self.k3_aa, self.s3_aa, best_func, best_popt, score, \
                self.f_idx = self.fit(bin_edges, hist, f_idx=f_idx)
            if score < 0.8:
                warnings.warn(f'Dihedral {self.idx + 1}: Bead{self.bead1.idx + 1}-Bead{self.bead2.idx + 1}-'
                              f'Bead{self.bead3.idx + 1}-Bead{self.bead4.idx + 1} fit not good.')
            self.df_dist['p_fit'] = best_func(bin_edges, *best_popt)
        else:
            return
        self.k1, self.s1, self.k2, self.s2, self.k3, self.s3 = \
            self.k1_aa, self.s1_aa, self.k2_aa, self.s2_aa, self.k3_aa, self.s3_aa
        self.df_dist.to_csv(f'dist_dihedral_{self.idx + 1}{tag}.xvg', header=False, index=False, sep='\t')

    def fit(self, x, y, f_idx=None):
        funcs = [
            lambda x, c, k1, s1: self.func(x, c, k1, s1, 0, 0, 0, 0),
            lambda x, c, k2, s2: self.func(x, c, 0, 0, k2, s2, 0, 0),
            lambda x, c, k3, s3: self.func(x, c, 0, 0, 0, 0, k3, s3),
            lambda x, c, k1, s1, k2, s2: self.func(x, c, k1, s1, k2, s2, 0, 0),
            lambda x, c, k2, s2, k3, s3: self.func(x, c, 0, 0, k2, s2, k3, s3),
            lambda x, c, k1, s1, k3, s3: self.func(x, c, k1, s1, 0, 0, k3, s3),
            lambda x, c, k1, s1, k2, s2, k3, s3: self.func(x, c, k1, s1, k2, s2, k3, s3),
        ]
        best_score = 0.
        for i, func in enumerate(funcs):
            if f_idx is not None and i != f_idx:
                continue
            if i < 3:
                n = 1
            elif i < 6:
                n = 2
            else:
                n = 3
            popt, score = curve_fit_rsq(f=func, xdata=x, ydata=y,
                                        bounds=[[-np.inf] + [0, -np.inf] * n, [np.inf] * (1 + n * 2)])
            if f_idx is not None or score > 0.95:
                best_func = func
                best_popt = popt
                best_score = score
                best_i = i
                break
            elif score > best_score and (score > 0.8 or i == 0):
                best_func = func
                best_popt = popt
                best_score = score
                best_i = i
        if best_score < 0.8:
            best_popt = [np.log(1 / 2 / np.pi) / - self.beta] + [0] * (len(best_popt) - 1)
        if best_i == 0:
            k1, s1, k2, s2, k3, s3 = *best_popt[1:], 0, 0, 0, 0
        elif best_i == 1:
            k1, s1, k2, s2, k3, s3 = 0, 0, *best_popt[1:], 0, 0
        elif best_i == 2:
            k1, s1, k2, s2, k3, s3 = 0, 0, 0, 0, *best_popt[1:]
        elif best_i == 3:
            k1, s1, k2, s2, k3, s3 = *best_popt[1:], 0, 0
        elif best_i == 4:
            k1, s1, k2, s2, k3, s3 = 0, 0, *best_popt[1:]
        elif best_i == 5:
            k1, s1, k2, s2, k3, s3 = *best_popt[1:3], 0, 0, *best_popt[3:]
        else:
            k1, s1, k2, s2, k3, s3 = best_popt[1:]
        return k1, self._pbc(np.degrees(s1), 0, 360), k2, self._pbc(np.degrees(s2), 0, 360), \
               k3, self._pbc(np.degrees(s3), 0, 360), best_func, best_popt, best_score, best_i

    @staticmethod
    def _pbc(v, p_min, p_max):
        assert p_max > p_min
        periodic = p_max - p_min
        if v >= p_max:
            v -= periodic * int((v - p_max) / periodic + 1)
        elif v < p_min:
            v += periodic * int((p_min - v) / periodic + 1)
        return v

    def update_cg_distribution(self, learning_rate=0.1):
        dihedrals_rad_traj_cg = self.calculate_dihedral(self.bead1.position, self.bead2.position, self.bead3.position,
                                                        self.bead4.position)
        hist, bin_edges = np.histogram(dihedrals_rad_traj_cg, bins=90, range=[0, 2 * np.pi], density=True)
        bin_edges = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
        self.df_dist[f'p_cg_{self.n_iter}'] = hist
        if not (self.k1 == 0 and self.k2 == 0 and self.k3 == 0):
            k1, s1, k2, s2, k3, s3, best_func, best_popt, score, _ = self.fit(bin_edges, hist, f_idx=self.f_idx)
            self.k1 += (self.k1_aa - k1) * learning_rate
            self.s1 += self._pbc(self.s1_aa - s1, -180., 180.) * learning_rate
            self.k2 += (self.k2_aa - k2) * learning_rate
            self.s2 += self._pbc(self.s2_aa - s2, -180., 180.) * learning_rate
            self.k3 += (self.k3_aa - k3) * learning_rate
            self.s3 += self._pbc(self.s3_aa - s3, -180., 180.) * learning_rate

    def func(self, x, c, k1, s1, k2, s2, k3, s3):
        V = k1 * (1 + np.cos(x - s1)) + k2 * (1 + np.cos(x * 2 - s2)) + k3 * (1 + np.cos(x * 3 - s3)) + c
        return np.exp(- V * self.beta)

    @staticmethod
    def calculate_dihedral(A, B, C, D):
        BA = A - B
        BC = C - B
        CD = D - C
        n1 = np.cross(BA, BC)
        angle_ref = np.einsum('ij,ij->i', n1, CD)
        n2 = np.cross(-BC, CD)
        # Normalize the normal vectors
        n1 /= np.linalg.norm(n1, axis=1).reshape(-1, 1)
        n2 /= np.linalg.norm(n2, axis=1).reshape(-1, 1)
        # Calculate the angle between the two normal vectors
        dihedral_rad = np.arccos(np.clip(np.einsum('ij,ij->i', n1, n2), -1.0, 1.0))
        dihedral_rad = dihedral_rad * np.where(angle_ref > 0, -1, 0) + dihedral_rad * np.where(angle_ref < 0, 1, 0) + np.where(angle_ref > 0, 2 * np.pi, 0)
        return dihedral_rad
