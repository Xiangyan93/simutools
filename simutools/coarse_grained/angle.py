#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import numpy as np
import pandas as pd
import warnings
from scipy.integrate import trapz
from .bead import Bead
from ..utils import curve_fit_rsq


class Angle:
    def __init__(self, bead1: Bead, bead2: Bead, bead3: Bead, idx: int = None):
        self.bead1 = bead1
        self.bead2 = bead2
        self.bead3 = bead3
        self.idx = idx

    def __eq__(self, other):
        return sorted(self.bead_idx) == sorted(other.bead_idx)

    @property
    def bead_idx(self):
        return [self.bead1.idx, self.bead2.idx, self.bead3.idx]

    @property
    def n_iter(self) -> int:
        for i in range(1000):
            if f'p_cg_{i}' not in self.df_dist:
                return i

    def get_aa_distribution(self, T: int = 298, tag: str = '',
                            fitting: Literal['least_square', 'mean_variance'] = 'mean_variance'):
        angles_rad_traj = self.calculate_angle(self.bead1.com, self.bead2.com, self.bead3.com)
        angles_degree_traj = np.degrees(angles_rad_traj)
        hist, bin_edges = np.histogram(angles_rad_traj, bins=360, range=[0, np.pi], density=True)
        bin_edges = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
        df = pd.DataFrame({'time': self.bead1.time, 'angle': angles_degree_traj})
        df.to_csv(f'angle_{self.idx + 1}{tag}.xvg', header=False, index=False, sep='\t')
        self.df_dist = pd.DataFrame({'angle': np.degrees(bin_edges), 'p_aa': hist})
        if fitting == 'least_square':
            self.beta = 1000 / 8.314 / T
            self.a0_rad_aa = angles_rad_traj.mean()
            self.a0_degree_aa = np.degrees(self.a0_rad_aa)
            popt, score = curve_fit_rsq(f=self.func1, xdata=bin_edges, ydata=hist)
            self.ka_aa = popt[0]
            if score < 0.9:
                warnings.warn(f'Angle {self.idx + 1}: Bead{self.bead1.idx + 1}-Bead{self.bead2.idx + 1}-'
                              f'Bead{self.bead3.idx + 1} may have multiple equilibrium distance.')
            self.df_dist['p_fit'] = self.func1(bin_edges, *popt)
        elif fitting == 'mean_variance':
            self.beta = 1000 / 8.314 / T
            self.a0_rad_aa = angles_rad_traj.mean()
            self.a0_degree_aa = np.degrees(self.a0_rad_aa)
            self.ka_aa = 1 / (2 * self.beta * angles_rad_traj.var())
            self.df_dist['p_fit'] = self.func(bin_edges, self.a0_rad_aa, self.ka_aa)
        else:
            raise ValueError(f'fitting algorithm invalid: {fitting}.')
        self.df_dist.to_csv(f'dist_angle_{self.idx + 1}{tag}.xvg', header=False, index=False, sep='\t')
        self.a0 = self.a0_degree_aa
        self.ka = self.ka_aa

    def update_cg_distribution(self, learning_rate=0.01, limit: float = 1.5):
        angles_rad_traj_cg = self.calculate_angle(self.bead1.position, self.bead2.position, self.bead3.position)
        hist, bin_edges = np.histogram(angles_rad_traj_cg, bins=360, range=[0, np.pi], density=True)
        self.df_dist[f'p_cg_{self.n_iter}'] = hist
        # bin_edges = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
        dev_a0 = self.a0_rad_aa - angles_rad_traj_cg.mean()
        self.a0 += np.degrees(dev_a0) * learning_rate
        self.a0 = np.clip(self.a0, self.a0_degree_aa / limit, self.a0_degree_aa * limit)
        ka_cg = 1 / (2 * self.beta * angles_rad_traj_cg.var())
        mul_ka = self.ka_aa / ka_cg
        self.ka += self.ka * (mul_ka - 1) * learning_rate
        self.ka = np.clip(self.ka, self.ka_aa / limit, self.ka_aa * limit)

    def func1(self, x, k):
        V = k * (x - self.a0_rad_aa) ** 2
        return np.exp(- V * self.beta) / np.sqrt(np.pi / self.beta / k)

    def func(self, x, a0, k):
        V = k * (x - a0) ** 2
        return np.exp(- V * self.beta) / np.sqrt(np.pi / self.beta / k)

    @staticmethod
    def calculate_angle(A, B, C):
        BA = A - B
        BC = C - B
        a = np.einsum('ij,ij->i', BA, BC)
        b = np.linalg.norm(BA, axis=1)
        c = np.linalg.norm(BC, axis=1)
        angles_rad_traj = np.arccos(a / b / c)
        return angles_rad_traj
