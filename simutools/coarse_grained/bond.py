#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import numpy as np
import pandas as pd
import warnings
from scipy.stats import wasserstein_distance
from ..utils import curve_fit_rsq
from .bead import Bead


class Bond:
    def __init__(self, bead1: Bead, bead2: Bead, idx: int = None):
        self.bead1 = bead1
        self.bead2 = bead2
        self.idx = idx

    def __eq__(self, other):
        return sorted(self.bead_idx) == sorted(other.bead_idx)

    @property
    def bead_idx(self):
        return [self.bead1.idx, self.bead2.idx]

    @property
    def IsConstraint(self) -> bool:
        if self.bead1.IsPartAromatic and self.bead2.IsPartAromatic:
            for ring_idx in self.bead1.raw_rings_idx:
                if set(ring_idx) in [set(g) for g in self.bead2.raw_rings_idx]:
                    assert self.kb_aa > 20000
                    return True
        if self.kb_aa > 20000:
            return True
        else:
            return False
        """
        if self.bead1.IsPartAromatic and self.bead2.IsPartAromatic:
            for ring_idx in self.bead1.raw_rings_idx:
                if set(ring_idx) in [set(g) for g in self.bead2.raw_rings_idx]:
                    return True
        elif self.bead1.IsSugar and self.bead2.IsSugar:
            return True
        return False
        """

    @property
    def n_iter(self) -> int:
        for i in range(1000):
            if f'p_cg_{i}' not in self.df_dist:
                return i

    def get_aa_distribution(self, T: int = 298, tag: str = '',
                            fitting: Literal['least_square', 'mean_variance'] = 'mean_variance'):
        bonds_traj = self.calculate_distance(self.bead1.com, self.bead2.com)
        self.min = bonds_traj.min() - 0.1 if bonds_traj.min() > 0.1 else 0.0
        self.max = bonds_traj.max() + 0.1
        hist, bin_edges = np.histogram(bonds_traj, bins=200, range=[self.min, self.max],
                                       density=True)
        bin_edges = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
        df = pd.DataFrame({'time': self.bead1.time, 'bond_length': bonds_traj})
        df.to_csv(f'bond_{self.idx + 1}{tag}.xvg', header=False, index=False, sep='\t')
        self.df_dist = pd.DataFrame({'bond_length': bin_edges, 'p_aa': hist})
        self.fitting = fitting
        if fitting == 'least_square':
            self.beta = 1000 / 8.314 / T
            self.b0_aa = bonds_traj.mean()
            func = lambda x, k: self.func(x, self.b0_aa, k)
            popt, score = curve_fit_rsq(f=func, xdata=bin_edges, ydata=hist)
            self.kb_aa = popt[0]
            if score < 0.9:
                warnings.warn(f'Bond {self.idx + 1}: Bead{self.bead1.idx + 1}-Bead{self.bead2.idx + 1} '
                              f'may have multiple equilibrium distance.')
            self.df_dist['p_fit'] = func(bin_edges, *popt)
        elif fitting == 'mean_variance':
            self.beta = 1000 / 8.314 / T
            self.b0_aa = bonds_traj.mean()
            self.kb_aa = 1 / (self.beta * bonds_traj.var())
            self.df_dist['p_fit'] = self.func(bin_edges, self.b0_aa, self.kb_aa)
        else:
            raise ValueError(f'fitting algorithm invalid: {fitting}.')
        self.df_dist.to_csv(f'dist_bond_{self.idx + 1}{tag}.xvg', header=False, index=False, sep='\t')
        self.b0 = self.b0_aa
        self.kb = self.kb_aa

    def emd(self, n_iter):
        """Earth Mover's distance. Wassertein's distance."""
        return wasserstein_distance(self.df_dist['bond_length'], self.df_dist['bond_length'],
                                    self.df_dist['p_aa'], self.df_dist[f'p_cg_{n_iter}'])
        # dx = self.df_dist['bond_length'][1] - self.df_dist['bond_length'][0]
        # return (self.df_dist['p_aa'] - self.df_dist[f'p_cg_{n_iter}']).abs().sum() * dx

    def update_cg_distribution(self, learning_rate=0.01, limit: float = 1.5):
        bonds_traj_cg = self.calculate_distance(self.bead1.position, self.bead2.position)
        hist, bin_edges = np.histogram(bonds_traj_cg, bins=200, range=[self.min, self.max],
                                       density=True)
        self.df_dist[f'p_cg_{self.n_iter}'] = hist
        if not self.IsConstraint:
            if self.n_iter > 1 and self.emd(self.n_iter - 1) > self.emd(self.n_iter - 2):
                self.b0 = self.b0_old
                self.kb = self.kb_old
            else:
                self.b0_old = self.b0
                self.kb_old = self.kb
                if self.fitting == 'mean_variance':
                    dev_b0 = self.b0_aa - bonds_traj_cg.mean()
                    self.b0 += dev_b0 * learning_rate
                    self.b0 = np.clip(self.b0, self.b0_aa / limit, self.b0_aa * limit)
                    kb_cg = 1 / (2 * self.beta * bonds_traj_cg.var())
                    mul_kb = self.kb_aa / kb_cg
                    self.kb += self.kb * (mul_kb - 1) * learning_rate
                    self.kb = np.clip(self.kb, self.kb_aa / limit, self.kb_aa * limit)
                elif self.fitting == 'least_square':
                    try:
                        bin_edges = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
                        b0 = bonds_traj_cg.mean()
                        func = lambda x, k: self.func(x, b0, k)
                        popt, score = curve_fit_rsq(f=func, xdata=bin_edges, ydata=hist)
                        kb = popt[0]
                        self.b0 += (self.b0_aa - b0) * learning_rate
                        self.kb += (self.kb_aa - kb) * learning_rate
                    except:
                        warnings.warn(f'square least fit error: no update for the parameters of bond {self.idx + 1}')

    def func(self, x, b0, k):
        V = k / 2 * (x - b0) ** 2
        return np.exp(- V * self.beta) / np.sqrt(2 * np.pi / self.beta / k)

    @staticmethod
    def calculate_distance(A, B):
        return np.sqrt(np.sum((A - B) ** 2, axis=1))
