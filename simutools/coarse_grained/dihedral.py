#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import numpy as np
import pandas as pd
import warnings
from scipy.stats import wasserstein_distance
from .bead import Bead
from .angle import Angle
from ..utils import curve_fit_rsq


class Dihedral:
    def __init__(self, bead1: Bead, bead2: Bead, bead3: Bead, bead4: Bead, idx: int = None):
        self.bead1 = bead1
        self.bead2 = bead2
        self.bead3 = bead3
        self.bead4 = bead4
        self.idx = idx
        self.fix_s1 = None

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

    def get_aa_distribution(self, T: int = 298, tag: str = '', f_idx=None):
        nbins = 90
        dihedrals_rad_traj = self.calculate_dihedral(self.bead1.com, self.bead2.com, self.bead3.com, self.bead4.com)
        dihedrals_degree_traj = np.degrees(dihedrals_rad_traj)
        hist, bin_edges = np.histogram(dihedrals_rad_traj, bins=nbins, range=[0, 2 * np.pi], density=True)
        bin_edges = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
        df = pd.DataFrame({'time': self.bead1.time, 'dihedral': dihedrals_degree_traj})
        df.to_csv(f'dihedral_{self.idx + 1}{tag}.xvg', header=False, index=False, sep='\t')
        self.df_dist = pd.DataFrame({'dihedral': np.degrees(bin_edges), 'p_aa': hist})
        if np.count_nonzero(hist < 1e-6) > nbins * 5 / 6:
            self.func_type = 2  # improper dihedral harmonic.
            # assert not self.CBT
            idx = np.where(hist > 1e-6)[0]
            if 0 in idx.tolist() and nbins - 1 in idx.tolist():
                idx0 = idx[idx < nbins / 2]
                assert idx0.max() == len(idx0) - 1
                idx1 = idx[idx > nbins / 2]
                assert idx1.min() == nbins - len(idx1)
            else:
                assert idx.max() - idx.min() + 1 == len(idx)
        # elif self.CBT:
        #     self.func_type = 11
        else:
            self.func_type = 13  # proper dihedral
        if self.func_type == 13:
            self.beta = 1000 / 8.314 / T
            a, b, c, d, e, f, best_func, best_popt, score, self.f_idx = self.fit1and3(bin_edges, hist)
            if self.func_type == 1:
                self.k1_aa, self.s1_aa, self.k2_aa, self.s2_aa, self.k3_aa, self.s3_aa = a, b, c, d, e, f
                self.k1, self.s1, self.k2, self.s2, self.k3, self.s3 = a, b, c, d, e, f
            else:
                assert self.func_type == 3
                self.C0_aa, self.C1_aa, self.C2_aa, self.C3_aa, self.C4_aa, self.C5_aa = 0, b, c, d, e, f
                self.C0, self.C1, self.C2, self.C3, self.C4, self.C5 = 0, b, c, d, e, f
            if score < 0.8:
                warnings.warn(f'Dihedral {self.idx + 1}: Bead{self.bead1.idx + 1}-Bead{self.bead2.idx + 1}-'
                              f'Bead{self.bead3.idx + 1}-Bead{self.bead4.idx + 1} fit not good.')
            self.df_dist['p_fit'] = best_func(bin_edges, *best_popt)
        elif self.func_type == 2:
            self.beta = 1000 / 8.314 / T
            d_rad = self._pbc(dihedrals_rad_traj, dihedrals_rad_traj[0] - np.pi, dihedrals_rad_traj[0] + np.pi)
            self.d0_rad_aa = self._pbc(d_rad.mean(), 0, 2 * np.pi)
            self.d0_degree_aa = np.degrees(self.d0_rad_aa)
            self.kd_aa = 1 / (2 * self.beta * d_rad.var())
            self.df_dist['p_fit'] = self.func2(bin_edges, self.d0_rad_aa, self.kd_aa)
            self.d0, self.kd = self.d0_degree_aa, self.kd_aa
        elif self.func_type == 11:
            self.beta = 1000 / 8.314 / T
            _, self.a1_aa, self.a2_aa, self.a3_aa, self.a4_aa, best_func, best_popt, score, self.f_idx = \
                self.fit11(bin_edges, hist, f_idx=f_idx)
            if score < 0.8:
                warnings.warn(f'Dihedral {self.idx + 1}: Bead{self.bead1.idx + 1}-Bead{self.bead2.idx + 1}-'
                              f'Bead{self.bead3.idx + 1}-Bead{self.bead4.idx + 1} fit not good.')
            self.df_dist['p_fit'] = best_func(bin_edges, *best_popt)
            self.a1, self.a2, self.a3, self.a4 = self.a1_aa, self.a2_aa, self.a3_aa, self.a4_aa
            self.a0 = 11.5 - self.a1 - self.a2 - self.a3 - self.a4
        else:
            raise ValueError
        self.df_dist.to_csv(f'dist_dihedral_{self.idx + 1}{tag}.xvg', header=False, index=False, sep='\t')

    def fit1(self, x, y, f_idx: int = None):
        if self.fix_s1 is not None:
            best_func = lambda x, c, k1: self.func1(x, c, k1, self.fix_s1, 0, 0, 0, 0)
            best_popt, best_score = curve_fit_rsq(f=best_func, xdata=x, ydata=y,
                                                  bounds=[[-np.inf, 0], [np.inf, np.inf]])
            k1, s1, k2, s2, k3, s3 = best_popt[1], self.fix_s1, 0, 0, 0, 0
            best_i = -1
        else:
            funcs = [
                lambda x, c, k1, s1: self.func1(x, c, k1, s1, 0, 0, 0, 0),
                lambda x, c, k2, s2: self.func1(x, c, 0, 0, k2, s2, 0, 0),
                lambda x, c, k3, s3: self.func1(x, c, 0, 0, 0, 0, k3, s3),
                lambda x, c, k1, s1, k2, s2: self.func1(x, c, k1, s1, k2, s2, 0, 0),
                lambda x, c, k2, s2, k3, s3: self.func1(x, c, 0, 0, k2, s2, k3, s3),
                lambda x, c, k1, s1, k3, s3: self.func1(x, c, k1, s1, 0, 0, k3, s3),
                lambda x, c, k1, s1, k2, s2, k3, s3: self.func1(x, c, k1, s1, k2, s2, k3, s3),
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
                try:
                    popt, score = curve_fit_rsq(f=func, xdata=x, ydata=y,
                                                bounds=[[-np.inf] + [0, -np.inf] * n, [np.inf] * (1 + n * 2)])
                except:
                    continue
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

    def fit3(self, x, y, f_idx: int = None):
        funcs = [
            lambda x, C0, C1: self.func3(x, C0, C1, 0, 0, 0, 0),
            lambda x, C0, C1, C2: self.func3(x, C0, C1, C2, 0, 0, 0),
            lambda x, C0, C1, C2, C3: self.func3(x, C0, C1, C2, C3, 0, 0),
            lambda x, C0, C1, C2, C3, C4: self.func3(x, C0, C1, C2, C3, C4, 0),
            self.func3
        ]
        best_score = 0.
        for i, func in enumerate(funcs):
            if f_idx is not None and i != f_idx:
                continue
            try:
                popt, score = curve_fit_rsq(f=func, xdata=x, ydata=y)
            except:
                continue
            if f_idx is not None or score > 0.95:
                best_func = func
                best_popt = popt
                best_score = score
                best_i = i
                break
            elif score > best_score:
                best_func = func
                best_popt = popt
                best_score = score
                best_i = i
        C0, C1, C2, C3, C4, C5 = *best_popt, *([0] * (6 - len(best_popt)))
        return C0, C1, C2, C3, C4, C5, best_func, best_popt, best_score, best_i

    def fit1and3(self, x, y):
        k1, s1, k2, s2, k3, s3, best_func1, best_popt1, best_score1, best_i1 = self.fit1(x, y)
        C0, C1, C2, C3, C4, C5, best_func3, best_popt3, best_score3, best_i3 = self.fit3(x, y)
        if best_score1 > best_score3:
            self.func_type = 1
            return k1, s1, k2, s2, k3, s3, best_func1, best_popt1, best_score1, best_i1
        else:
            self.func_type = 3
            return C0, C1, C2, C3, C4, C5, best_func3, best_popt3, best_score3, best_i3

    def fit11(self, x, y, f_idx: int = None):
        funcs = [
            lambda x, a0, a1: self.func11(x, a0, a1, 0, 0, 0),
            lambda x, a0, a1, a2: self.func11(x, a0, a1, a2, 0, 0),
            lambda x, a0, a1, a2, a3: self.func11(x, a0, a1, a2, a3, 0),
            self.func11
        ]
        best_score = 0.
        for i, func in enumerate(funcs):
            if f_idx is not None and i != f_idx:
                continue
            popt, score = curve_fit_rsq(f=func, xdata=x, ydata=y)
            if f_idx is not None or score > 0.95:
                best_func = func
                best_popt = popt
                best_score = score
                best_i = i
                break
            elif score > best_score:
                best_func = func
                best_popt = popt
                best_score = score
                best_i = i
        a0, a1, a2, a3, a4 = *best_popt, *([0] * (5 - len(best_popt)))
        return a0, a1, a2, a3, a4, best_func, best_popt, best_score, best_i

    @staticmethod
    def _pbc(v, p_min, p_max):
        assert p_max > p_min
        periodic = p_max - p_min
        while len(np.where(v < p_min)[0]) != 0:
            v = np.where(v < p_min, v + periodic, v)
        while len(np.where(v >= p_max)[0]) != 0:
            v = np.where(v >= p_max, v - periodic, v)
        return v

    def emd(self, n_iter):
        """Earth Mover's distance. Wassertein's distance."""
        return wasserstein_distance(self.df_dist['dihedral'], self.df_dist['dihedral'],
                                    self.df_dist['p_aa'], self.df_dist[f'p_cg_{n_iter}'])
        # dx = np.deg2rad(self.df_dist['dihedral'][1] - self.df_dist['dihedral'][0])
        # return (self.df_dist['p_aa'] - self.df_dist[f'p_cg_{n_iter}']).abs().sum() * dx

    def update_cg_distribution(self, learning_rate=0.1):
        dihedrals_rad_traj_cg = self.calculate_dihedral(self.bead1.position, self.bead2.position, self.bead3.position,
                                                        self.bead4.position)
        hist, bin_edges = np.histogram(dihedrals_rad_traj_cg, bins=90, range=[0, 2 * np.pi], density=True)
        bin_edges = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
        self.df_dist[f'p_cg_{self.n_iter}'] = hist
        if self.NoForce:
            return
        elif self.n_iter > 1 and self.emd(self.n_iter - 1) > self.emd(self.n_iter - 2):
            if self.func_type == 1:
                self.k1, self.s1, self.k2, self.s2, self.k3, self.s3 = \
                    self.k1_old, self.s1_old, self.k2_old, self.s2_old, self.k3_old, self.s3_old
            elif self.func_type == 2:
                self.d0, self.kd = self.d0_old, self.kd_old
            elif self.func_type == 3:
                self.C1, self.C2, self.C3, self.C4, self.C5 = \
                    self.C1_old, self.C2_old, self.C3_old, self.C4_old, self.C5_old
            elif self.func_type == 11:
                self.a0, self.a1, self.a2, self.a3, self.a4 = \
                    self.a0_old, self.a1_old, self.a2_old, self.a3_old, self.a4_old
        else:
            if self.func_type == 1:
                self.k1_old, self.s1_old, self.k2_old, self.s2_old, self.k3_old, self.s3_old = \
                    self.k1, self.s1, self.k2, self.s2, self.k3, self.s3
                if not (self.k1 == 0 and self.k2 == 0 and self.k3 == 0):
                    try:
                        k1, s1, k2, s2, k3, s3, best_func, best_popt, score, _ = self.fit1(bin_edges, hist, f_idx=self.f_idx)
                        self.k1 += (self.k1_aa - k1) * learning_rate
                        self.s1 += self._pbc(self.s1_aa - s1, -180., 180.) * learning_rate
                        self.k2 += (self.k2_aa - k2) * learning_rate
                        self.s2 += self._pbc(self.s2_aa - s2, -180., 180.) * learning_rate
                        self.k3 += (self.k3_aa - k3) * learning_rate
                        self.s3 += self._pbc(self.s3_aa - s3, -180., 180.) * learning_rate
                    except:
                        warnings.warn(f'square least fit error: no update for the parameters of dihedral {self.idx + 1}')
            elif self.func_type == 2:
                self.d0_old, self.kd_old = self.d0, self.kd
                dihedrals_rad_traj_cg = self._pbc(dihedrals_rad_traj_cg,
                                                  dihedrals_rad_traj_cg[0] - np.pi,
                                                  dihedrals_rad_traj_cg[0] + np.pi)
                #print('qq: ', self.d0)
                dev_d0 = self._pbc(self.d0_rad_aa - dihedrals_rad_traj_cg.mean(), - np.pi, np.pi)
                self.d0 = self._pbc(self.d0 + np.degrees(dev_d0) * learning_rate, 0, 360)
                # print(np.degrees(dev_d0), self.d0)
                # self.a0 = np.clip(self.a0, self.a0_degree_aa / limit, self.a0_degree_aa * limit)
                kd_cg = 1 / (2 * self.beta * dihedrals_rad_traj_cg.var())
                mul_kd = self.kd_aa / kd_cg
                self.kd += self.kd * (mul_kd - 1) * learning_rate
                # self.kd = np.clip(self.kd, self.kd_aa / limit, self.kd_aa * limit)
            elif self.func_type == 3:
                self.C1_old, self.C2_old, self.C3_old, self.C4_old, self.C5_old = \
                    self.C1, self.C2, self.C3, self.C4, self.C5
                try:
                    C0, C1, C2, C3, C4, C5, best_func, best_popt, score, _ = self.fit3(bin_edges, hist, f_idx=self.f_idx)
                    self.C1 += (self.C1_aa - C1) * learning_rate
                    self.C2 += (self.C2_aa - C2) * learning_rate
                    self.C3 += (self.C3_aa - C3) * learning_rate
                    self.C4 += (self.C4_aa - C4) * learning_rate
                    self.C5 += (self.C5_aa - C5) * learning_rate
                except:
                    warnings.warn(f'square least fit error: no update for the parameters of dihedral {self.idx + 1}')
            else:
                assert self.func_type == 11
                self.a0_old, self.a1_old, self.a2_old, self.a3_old, self.a4_old = \
                    self.a0, self.a1, self.a2, self.a3, self.a4
                try:
                    _, a1, a2, a3, a4, best_func, best_popt, score, _ = self.fit11(bin_edges, hist, f_idx=self.f_idx)
                    self.a1 += (self.a1_aa - a1) * learning_rate
                    self.a2 += (self.a2_aa - a2) * learning_rate
                    self.a3 += (self.a3_aa - a3) * learning_rate
                    self.a4 += (self.a4_aa - a4) * learning_rate
                    self.a0 = 11.5 - self.a1 - self.a2 - self.a3 - self.a4
                except:
                    warnings.warn(f'square least fit error: no update for the parameters of dihedral {self.idx + 1}')

    def func1(self, x, c, k1, s1, k2, s2, k3, s3):
        V = k1 * (1 + np.cos(x - s1)) + k2 * (1 + np.cos(x * 2 - s2)) + k3 * (1 + np.cos(x * 3 - s3)) + c
        return np.exp(- V * self.beta)

    def func2(self, x, d0, k):
        V = k * (x - d0) ** 2
        return np.exp(- V * self.beta) / np.sqrt(np.pi / self.beta / k)

    def func3(self, x, C0, C1, C2, C3, C4, C5):
        cosx = np.cos(x - np.pi)
        V = C0 + C1 * cosx + C2 * cosx ** 2 + C3 * cosx ** 3 + C4 * cosx ** 4 + C5 * cosx ** 5
        return np.exp(- V * self.beta)

    def func11(self, x, a0, a1, a2, a3, a4):
        cosx = np.cos(x)
        V = a0 + a1 * cosx + a2 * cosx ** 2 + a3 * cosx ** 3 + a4 * cosx ** 4
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
        dihedral_rad = dihedral_rad * np.where(angle_ref > 0, -1, 0) + dihedral_rad * np.where(angle_ref < 0, 1,
                                                                                               0) + np.where(
            angle_ref > 0, 2 * np.pi, 0)
        return dihedral_rad

    def set_angles(self, angles: List[Angle]):
        self.angles = []
        for angle in angles:
            if set(angle.bead_idx) in [{self.bead1.idx, self.bead2.idx, self.bead3.idx},
                                       {self.bead2.idx, self.bead3.idx, self.bead4.idx}]:
                self.angles.append(angle)
        # assert len(self.angles) == 2

    @property
    def CBT(self):
        for angle in self.angles:
            if angle.CBT:
                return True
        return False

    @property
    def NoForce(self) -> bool:
        if hasattr(self, 'no_force'):
            return self.no_force
        else:
            for b1 in self.bead2.neighbors:
                if b1 in self.bead3.neighbors:
                    return True
            return False
