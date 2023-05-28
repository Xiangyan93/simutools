#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
import pickle
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import pandas as pd
import warnings
from itertools import permutations
import matplotlib.pyplot as plt
from ..utils import mol_to_nx, merge_lists
from .bead import Bead, VirtualSite
from .bond import Bond
from .angle import Angle
from .dihedral import Dihedral
from .martini3bible import martini3bible, sugar_beadtype


class Mapping:
    def __init__(self, mol2):
        self.mol2 = mol2
        self.mol = Chem.rdmolfiles.MolFromMol2File(mol2, sanitize=True, removeHs=False)
        # assign all hydrogen to its connected heavy atom
        self.h_belong_map = {}
        for atom in self.mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                neighbors = atom.GetNeighbors()
                assert len(neighbors) == 1
                self.h_belong_map[atom.GetIdx()] = neighbors[0].GetIdx()
        self.graph = mol_to_nx(self.mol)
        self.graph_heavy = self.graph.copy()
        self.graph_heavy.remove_nodes_from(self.h_belong_map.keys())

    def mapping(self):
        """split the molecule into multiple substructures based on rings and functional groups"""
        # step1, assign atoms into subgraphs based on cyclic structures
        # # create groups for cyclic structures
        groups = []
        self.get_ring_info()
        for i, group_idx in enumerate(self.merge_rings_idx):
            group = Bead(mol=self.mol, graph_heavy=self.graph_heavy, atom_idx=group_idx)
            group.raw_rings_idx = self.raw_rings_idx[i]
            groups.append(group)
        # # create groups for acyclic structures
        graph = self.graph_heavy.copy()
        used_atoms = []
        for ring_idx in self.merge_rings_idx:
            used_atoms += ring_idx
        graph.remove_nodes_from(used_atoms)
        groups_idx = list(nx.connected_components(graph))
        for i, group_idx in enumerate(groups_idx):
            group = Bead(mol=self.mol, graph_heavy=self.graph_heavy, atom_idx=group_idx)
            groups.append(group)
        # print([group.atom_idx for group in groups])
        assert sum([len(group) for group in groups]) == self.mol.GetNumHeavyAtoms(), \
            f'{sum([len(group) for group in groups])} = {self.mol.GetNumHeavyAtoms()}'
        self.groups = groups
        # step2, finds functional groups and mapping
        for smarts, bead_type in martini3bible.items():
            matches = self.mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
            # print(smarts, matches)
            for match in matches:
                groups = self.find_matched_groups(match, self.groups)
                # if one of the atom has been used for mapping, skip
                if len(groups) != 1:
                    warnings.warn(f'Functional group {smarts}(atom id: {match}) cannot CG mapping, part in ring')
                    continue
                else:
                    if groups[0].IsAllInRing:
                        warnings.warn(f'Functional group {smarts}(atom id: {match}) cannot CG mapping, all in ring')
                        continue
                    elif groups[0].bead_type is not None:
                        warnings.warn(f'Functional group {smarts}(atom id: {match}) cannot CG mapping, already used')
                        continue
                # find the group that the substructure belongs to.
                assert len(groups) == 1
                group = groups[0]
                g = Bead(mol=self.mol, graph_heavy=self.graph_heavy, atom_idx=match)
                g.bead_type = bead_type
                self.groups.remove(group)
                self.groups.append(g)
                for g_ in group.substract_bead(g):
                    self.groups.append(g_)
        # step3, generate mapping for rings
        groups = []
        for group in self.groups:
            groups += group.ring_split()
        for group in groups:
            if group.bead_type is None:
                print(group.atom_idx)
                assert group.IsAlkane
                group.bead_type = 'C1'
        self.groups = groups
        # step4, merge single-connected single-heavy-atom groups into its neighbor
        self.update_connectivity()
        for i, group in enumerate(self.groups):
            for group_ in group.neighbors:
                if len(group_) == 1 and len(group_.neighbors) == 1 and len(self.groups[i]) != 1 and not group_.confirm:
                    self.groups[i] = self.groups[i].add_bead(group_)
                    group_.atom_idx.pop(0)
        self.groups = [group for group in self.groups if group.atom_idx]
        # step5, merge multi-connected single-heavy-atom groups into its neighbors
        while True:
            self.update_connectivity()
            for i, group in enumerate(self.groups):
                if len(group) == 1 and not group.confirm:
                    assert len(group.neighbors) != 1
                    merge_ranks = [group_.merge_rank() for group_ in group.neighbors]
                    merge_idx = merge_ranks.index(max(merge_ranks))
                    group_idx = self.groups.index(group.neighbors[merge_idx])
                    self.groups[group_idx] = self.groups[group_idx].add_bead(group)
                    group.atom_idx.pop(0)
                    self.groups = [group for group in self.groups if group.atom_idx]
                    break
            else:
                break
        # step6, assign beads type based on Martini Bible.
        for group in self.groups:
            assert group.bead_type is not None
            if group.bead_type.startswith('T') or group.bead_type.startswith('S'):
                continue
            if len(group) == 2:
                group.bead_type = 'T' + group.bead_type
            elif len(group) == 3:
                group.bead_type = 'S' + group.bead_type
            if group.IsSugar:
                group.bead_type = sugar_beadtype[group.bead_type]
        # step7,
        self.update_connectivity()
        for i, bead in enumerate(self.groups):
            bead.idx = i
        self.bonds = []
        for i, bead1 in enumerate(self.groups):
            for j in range(i + 1, len(self.groups)):
                bead2 = self.groups[j]
                if bead2 in bead1.neighbors:
                    bond = Bond(bead1, bead2, idx=len(self.bonds))
                    self.bonds.append(bond)

        self.virtual_sites = []
        for bond in self.bonds:
            if bond.bead1.IsSugar and bond.bead2.IsSugar:
                for vs in self.virtual_sites:
                    if bond.bead1 in vs.beads or bond.bead2 in vs.beads:
                        skip = True
                        break
                else:
                    skip = False
                if not skip:
                    for neighbor in bond.bead1.neighbors:
                        if neighbor.IsSugar and neighbor in bond.bead2.neighbors:
                            self.virtual_sites.append(VirtualSite(beads=[bond.bead1, bond.bead2, neighbor]))
        for i, bead in enumerate(self.virtual_sites):
            bead.idx = len(self.groups) + i

        self.angles = []
        for bead in self.groups:
            for n1 in bead.neighbors:
                for n2 in bead.neighbors:
                    if n1.idx >= n2.idx or n1 in n2.neighbors:
                        continue
                    angle = Angle(n1, bead, n2, idx=len(self.angles))
                    self.angles.append(angle)

        self.dihedrals = []
        for bond in self.bonds:
            b2 = bond.bead1
            b3 = bond.bead2
            for b1 in b2.neighbors:
                for b4 in b3.neighbors:
                    if b1 == b3 or b2 == b4 or b1 == b4:
                        continue
                    dihedral = Dihedral(b1, b2, b3, b4, idx=len(self.dihedrals))
                    if dihedral not in self.dihedrals:
                        dihedral.set_angles(self.angles)
                        self.dihedrals.append(dihedral)
        """
        self.cg_graph = nx.Graph()
        self.cg_graph.add_nodes_from(range(len(self.groups)))
        cg_edges = []
        for i, g1 in enumerate(self.groups):
            for j in range(i+1, len(self.groups)):
                g2 = self.groups[j]
                if g2 in g1.neighbors:
                    cg_edges.append((i, j))
        self.cg_graph.add_edges_from(cg_edges)
        self.cg_angles_idx = []
        for i in self.cg_graph.nodes:
            neighbors = list(self.cg_graph.neighbors(i))
            for j in neighbors:
                for k in neighbors:
                    if j >= k or self.cg_graph.has_edge(j, k):
                        continue
                    self.cg_angles_idx.append((j, i, k))
        self.cg_dihedral_idx = []
        for i, j in cg_edges:
            for k in list(self.cg_graph.neighbors(i)):
                for l in list(self.cg_graph.neighbors(j)):
                    if k == j or l == i or k == l:
                        continue
                    for perm in permutations([i, j, k, l]):
                        if tuple(perm) in self.cg_dihedral_idx:
                            break
                    else:
                        self.cg_dihedral_idx.append((k, i, j, l))
        #print(cg_edges)
        #print(self.cg_angles_idx)
        #print(self.cg_dihedral_idx, len(self.cg_dihedral_idx))
        self.generate_ndx_bond_angel_dihedral()
        """

    def update_connectivity(self):
        for group in self.groups:
            group.neighbors = []
        for i, group in enumerate(self.groups):
            for j in range(i + 1, len(self.groups)):
                group_ = self.groups[j]
                if group.is_connected(group_):
                    group.neighbors.append(group_)
                    group_.neighbors.append(group)

    def get_ring_info(self):
        sssr = Chem.GetSymmSSSR(self.mol)
        rings_idx = [list(ring) for ring in sssr]
        self.merge_rings_idx, self.raw_rings_idx = merge_lists(rings_idx)

    def find_matched_groups(self, idx: List[int], groups: List[Bead]):
        """Return the groups that containing the atoms provided in idx."""
        gs = []
        for i in idx:
            for g in groups:
                if i in g.atom_idx:
                    if g not in gs:
                        gs.append(g)
        return gs

    def generate_ndx_mapping(self, file: str):
        with open(file, 'w') as f:
            for i, group in enumerate(self.groups):
                idx = [j + 1 for j in group.atom_idx + group.atoms_idx_h]
                f.write(f'[ B{i + 1} ]\n')
                f.write(' '.join(list(map(str, idx))))
                f.write('\n\n')

    def generate_itp(self, file: str, resName: str, atom_only: bool = False, group: bool = False):
        with open(file, 'w') as f:
            f.write('[ moleculetype ]\n')
            f.write(f'   {resName}         1\n')
            f.write('\n[ atoms ]\n')
            for i, bead in enumerate(self.groups):
                f.write('%7d%7s   0%8s%5s%5s%5d%7d\n' % (i + 1, bead.bead_type, resName, 'B%d' % (i + 1), i + 1,
                                                         bead.charge, bead.mass))
                n_beads = i
            for i, bead in enumerate(self.virtual_sites):
                f.write('%7d%7s   0%8s%5s%5s%5d%7d\n' % (i + n_beads + 2, bead.bead_type, resName, 'VS%d' % (i + 1),
                                                         i + n_beads + 2, bead.charge, bead.mass))
            if atom_only:
                return
            f.write('\n[ constraints ]\n')
            for bond in self.constraints:
                assert bond.IsInRing
                f.write('%7d%7d     1%10.5f\n' % (bond.bead1.idx + 1, bond.bead2.idx + 1, bond.b0))
                if group:
                    f.write(';\n')
            f.write('\n[ bonds ]\n')
            for bond in self.bonds:
                assert not bond.IsInRing
                f.write('%7d%7d     1%10.5f%14.5f\n' % (bond.bead1.idx + 1, bond.bead2.idx + 1, bond.b0, bond.kb))
                if group:
                    f.write(';\n')
            f.write('\n[ angles ]\n')
            for angle in self.angles:
                f.write('%7d%7d%7d%7d%10.3f%10.3f\n' % (angle.bead1.idx + 1, angle.bead2.idx + 1,
                                                        angle.bead3.idx + 1, angle.func_type, angle.a0, angle.ka))
                if group:
                    f.write(';\n')
            f.write('\n[ dihedrals ]\n')
            for dihedral in self.dihedrals:
                if dihedral.NoForce:
                    continue
                if dihedral.func_type == 1:
                    if dihedral.k1 != 0:
                        f.write('%7d%7d%7d%7d     1%10.3f%10.3f%5d\n' % (dihedral.bead1.idx + 1, dihedral.bead2.idx + 1,
                                                                         dihedral.bead3.idx + 1, dihedral.bead4.idx + 1,
                                                                         dihedral.s1, dihedral.k1, 1))
                    if dihedral.k2 != 0:
                        f.write('%7d%7d%7d%7d     1%10.3f%10.3f%5d\n' % (dihedral.bead1.idx + 1, dihedral.bead2.idx + 1,
                                                                         dihedral.bead3.idx + 1, dihedral.bead4.idx + 1,
                                                                         dihedral.s2, dihedral.k2, 2))
                    if dihedral.k3 != 0:
                        f.write('%7d%7d%7d%7d     1%10.3f%10.3f%5d\n' % (dihedral.bead1.idx + 1, dihedral.bead2.idx + 1,
                                                                         dihedral.bead3.idx + 1, dihedral.bead4.idx + 1,
                                                                         dihedral.s3, dihedral.k3, 3))
                elif dihedral.func_type == 2:
                    f.write('%7d%7d%7d%7d     2%10.3f%10.3f\n' % (dihedral.bead1.idx + 1, dihedral.bead2.idx + 1,
                                                                  dihedral.bead3.idx + 1, dihedral.bead4.idx + 1,
                                                                  dihedral.d0, dihedral.kd))
                elif dihedral.func_type == 3:
                    f.write('%7d%7d%7d%7d     3%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f\n' % (
                        dihedral.bead1.idx + 1, dihedral.bead2.idx + 1,
                        dihedral.bead3.idx + 1, dihedral.bead4.idx + 1,
                        dihedral.C0, dihedral.C1,
                        dihedral.C2, dihedral.C3,
                        dihedral.C4, dihedral.C5))
                else:
                    assert dihedral.func_type == 11
                    sin_angles = np.sin(dihedral.angles[0].a0_rad_aa) ** 3 * np.sin(dihedral.angles[1].a0_rad_aa) ** 3
                    k = 10.0
                    f.write('%7d%7d%7d%7d    11%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f\n' % (
                        dihedral.bead1.idx + 1, dihedral.bead2.idx + 1,
                        dihedral.bead3.idx + 1, dihedral.bead4.idx + 1,
                        dihedral.a0 / k / sin_angles, dihedral.a1 / k / sin_angles,
                        dihedral.a2 / k / sin_angles, dihedral.a3 / k / sin_angles,
                        dihedral.a4 / k / sin_angles, k))
                if group:
                    f.write(';\n')
            if self.virtual_sites:
                f.write('\n[ virtual_sitesn ]\n')
                for vs in self.virtual_sites:
                    f.write('%7d%7d   ' % (vs.idx + 1, 1) + ' '.join([str(bead.idx + 1) for bead in vs.beads]) + '\n')
                f.write('\n[ exclusions ]\n')
                for vs in self.virtual_sites:
                    f.write('%7d   ' % (vs.idx + 1) + ' '.join([str(bead.idx + 1) for bead in vs.beads]) + '\n')

    def generate_gro(self, file: str, resName: str, box_length: float):
        with open(file, 'w') as f:
            f.write('frame t= 0.000\n')
            f.write('%5d\n' % (len(self.groups) + len(self.virtual_sites)))
            for i, g in enumerate(self.groups):
                f.write('    1%s%7s%5d%8.3f%8.3f%8.3f\n' % (resName, 'B%d' % (i + 1), i + 1,
                                                            g.com[-1][0], g.com[-1][1], g.com[-1][2]))
            for i, g in enumerate(self.virtual_sites):
                f.write('    1%s%7s%5d%8.3f%8.3f%8.3f\n' % (resName, 'VS%d' % (i + 1), i + 1 + len(self.groups),
                                                            g.com[-1][0], g.com[-1][1], g.com[-1][2]))
            f.write('%10.5f%10.5f%10.5f\n' % (box_length, box_length, box_length))

    def generate_ndx_bond_angel_dihedral(self):
        with open('bonds.ndx', 'w') as f:
            for i, (j, k) in enumerate(self.cg_graph.edges):
                f.write(f'[bond{i + 1}]\n')
                f.write(f'  {j + 1}  {k + 1}\n')
        with open('angles.ndx', 'w') as f:
            for i, (j, k, l) in enumerate(self.cg_angles_idx):
                f.write(f'[angle{i + 1}]\n')
                f.write(f'  {j + 1}  {k + 1}  {l + 1}\n')
        with open('dihedrals.ndx', 'w') as f:
            for i, (j, k, l, m) in enumerate(self.cg_dihedral_idx):
                f.write(f'[dihedral{i + 1}]\n')
                f.write(f'  {j + 1}  {k + 1}  {l + 1}  {m + 1}\n')

    def load_aa_traj(self, gro: str, tpr: str):
        U = mda.Universe(tpr, gro)
        for i, group in enumerate(self.groups):
            print(f'Calculating center of mass of bead {i}')
            atom_indices = group.atom_idx + group.atoms_idx_h
            mda_atoms = U.select_atoms('index {}'.format(' '.join(map(str, atom_indices))))
            group.com = np.array([mda_atoms.center_of_mass() for ts in U.trajectory]) / 10  # unit in nm
            group.time = [ts.time for ts in U.trajectory]
            # group.com = np.array([mda_atoms.center_of_mass() for j, ts in enumerate(U.trajectory) if j < 1000]) / 10  # unit in nm
            # group.time = [ts.time for j, ts in enumerate(U.trajectory) if j < 1000]

    def load_cg_traj(self, gro: str, tpr: str):
        U = mda.Universe(tpr, gro)
        for i, group in enumerate(self.groups):
            print(f'Loading the coordinates of bead {i}')
            group.position = np.array([U.atoms[i].position for ts in U.trajectory]) / 10  # unit in nm
            group.time = [ts.time for ts in U.trajectory]

    def get_aa_distribution(self, tag: str = '', dihedral_with_force=[], dihedral_no_force=[]):
        for i, bond in enumerate(self.bonds):
            bond.get_aa_distribution(tag=tag, fitting='mean_variance')
        bonds = []
        constraints = []
        for bond in self.bonds:
            if bond.IsInRing:
                constraints.append(bond)
            else:
                bonds.append(bond)
        self.bonds = bonds
        self.constraints = constraints
        for i, angle in enumerate(self.angles):
            angle.get_aa_distribution(tag=tag, fitting='mean_variance')
        for i, dihedral in enumerate(self.dihedrals):
            if i + 1 in dihedral_no_force:
                dihedral.no_force = True
            if i + 1 in dihedral_with_force:
                dihedral.no_force = False
            dihedral.get_aa_distribution(tag=tag)

    def update_parameter(self):
        for i, bond in enumerate(self.bonds + self.constraints):
            # print(f'Update parameters for bond {bond.idx + 1}: {bond.bead1.idx + 1}-{bond.bead2.idx + 1}')
            bond.update_cg_distribution(learning_rate=0.05)
        for i, angle in enumerate(self.angles):
            # print(f'Update parameters for angle {angle.idx + 1}: {angle.bead1.idx + 1}-{angle.bead2.idx + 1}-{angle.bead3.idx + 1}')
            angle.update_cg_distribution(learning_rate=0.05)
        for i, dihedral in enumerate(self.dihedrals):
            # print(f'Update parameters for dihedral {dihedral.idx + 1}: {dihedral.bead1.idx + 1}-{dihedral.bead2.idx + 1}-'
            #      f'{dihedral.bead3.idx + 1}-{dihedral.bead4.idx + 1}')
            dihedral.update_cg_distribution(learning_rate=0.05)

    def write_distribution(self, file: str = 'distribution.svg', CG: bool = True, fit: bool = False):
        nx = 4
        ny = max(len(self.constraints), len(self.bonds), len(self.angles), len(self.dihedrals))
        fig, axs = plt.subplots(nx, ny, figsize=(ny * 4, nx * 4))
        d = 0.05
        plt.subplots_adjust(left=d, right=1 - d, top=1 - d, bottom=d)
        for i, bond in enumerate(self.constraints):
            axs[0, i].set_title(f'constraint {i + 1}: {bond.bead1.idx + 1}-{bond.bead2.idx + 1}')
            axs[0, i].plot(bond.df_dist['bond_length'], bond.df_dist['p_aa'], color='red', label='AA')
            axs[0, i].fill_between(bond.df_dist['bond_length'], bond.df_dist['p_aa'], 0, color='red', alpha=0.5)
            if CG:
                axs[0, i].plot(bond.df_dist['bond_length'], bond.df_dist[f'p_cg_{bond.n_iter - 1}'],
                               color='blue', label='CG')
                axs[0, i].fill_between(bond.df_dist['bond_length'], bond.df_dist[f'p_cg_{bond.n_iter - 1}'], 0,
                                       color='blue', alpha=0.5)
            if fit:
                axs[0, i].plot(bond.df_dist['bond_length'], bond.df_dist[f'p_fit'],
                               color='green', label='fitting')
                axs[0, i].fill_between(bond.df_dist['bond_length'], bond.df_dist[f'p_fit'], 0,
                                       color='green', alpha=0.5)
        for i, bond in enumerate(self.bonds):
            title = f'bond {i + 1}: {bond.bead1.idx + 1}-{bond.bead2.idx + 1}'
            if CG:
                title += ';emd=%.2f' % bond.emd(bond.n_iter - 1)
            axs[1, i].set_title(title)
            axs[1, i].plot(bond.df_dist['bond_length'], bond.df_dist['p_aa'], color='red', label='AA')
            axs[1, i].fill_between(bond.df_dist['bond_length'], bond.df_dist['p_aa'], 0, color='red', alpha=0.5)
            if CG:
                axs[1, i].plot(bond.df_dist['bond_length'], bond.df_dist[f'p_cg_{bond.n_iter - 1}'],
                               color='blue', label='CG')
                axs[1, i].fill_between(bond.df_dist['bond_length'], bond.df_dist[f'p_cg_{bond.n_iter - 1}'], 0,
                                       color='blue', alpha=0.5)
            if fit:
                axs[1, i].plot(bond.df_dist['bond_length'], bond.df_dist[f'p_fit'],
                               color='green', label='fitting')
                axs[1, i].fill_between(bond.df_dist['bond_length'], bond.df_dist[f'p_fit'], 0,
                                       color='green', alpha=0.5)
        for i, angle in enumerate(self.angles):
            title = f'angle {i + 1}({angle.bead1.idx + 1}-{angle.bead2.idx + 1}-{angle.bead3.idx + 1})' \
                    f';funct={angle.func_type}'
            if CG:
                title += ';emd=%.2f' % angle.emd(angle.n_iter - 1)
            axs[2, i].set_title(title)
            axs[2, i].plot(angle.df_dist['angle'], angle.df_dist['p_aa'], color='red', label='AA')
            axs[2, i].fill_between(angle.df_dist['angle'], angle.df_dist['p_aa'], 0, color='red', alpha=0.5)
            if CG:
                axs[2, i].plot(angle.df_dist['angle'], angle.df_dist[f'p_cg_{angle.n_iter - 1}'],
                               color='blue', label='CG')
                axs[2, i].fill_between(angle.df_dist['angle'], angle.df_dist[f'p_cg_{angle.n_iter - 1}'], 0,
                                       color='blue', alpha=0.5)
            if fit:
                axs[2, i].plot(angle.df_dist['angle'], angle.df_dist[f'p_fit'],
                               color='green', label='fitting')
                axs[2, i].fill_between(angle.df_dist['angle'], angle.df_dist[f'p_fit'], 0,
                                       color='green', alpha=0.5)
        for i, dihedral in enumerate(self.dihedrals):
            if dihedral.NoForce:
                title = f'dihedral {i + 1}({dihedral.bead1.idx + 1}-{dihedral.bead2.idx + 1}-{dihedral.bead3.idx + 1}-{dihedral.bead4.idx + 1})' \
                        f';funct=none'
            else:
                title = f'dihedral {i + 1}({dihedral.bead1.idx + 1}-{dihedral.bead2.idx + 1}-{dihedral.bead3.idx + 1}-{dihedral.bead4.idx + 1})' \
                        f';funct={dihedral.func_type}'
            if CG:
                title += ';emd=%.2f' % dihedral.emd(dihedral.n_iter - 1)
            axs[3, i].set_title(title)
            axs[3, i].plot(dihedral.df_dist['dihedral'], dihedral.df_dist['p_aa'], color='red', label='AA')
            axs[3, i].fill_between(dihedral.df_dist['dihedral'], dihedral.df_dist['p_aa'], 0, color='red', alpha=0.5)
            if CG:
                axs[3, i].plot(dihedral.df_dist['dihedral'], dihedral.df_dist[f'p_cg_{dihedral.n_iter - 1}'],
                               color='blue', label='CG')
                axs[3, i].fill_between(dihedral.df_dist['dihedral'], dihedral.df_dist[f'p_cg_{dihedral.n_iter - 1}'], 0,
                                       color='blue', alpha=0.5)
            if fit:
                axs[3, i].plot(dihedral.df_dist['dihedral'], dihedral.df_dist[f'p_fit'],
                               color='green', label='fitting')
                axs[3, i].fill_between(dihedral.df_dist['dihedral'], dihedral.df_dist[f'p_fit'], 0,
                                       color='green', alpha=0.5)
        plt.savefig(file, format=file.split('.')[1])

    def write_emd(self, file: str = 'emd.log') -> float:
        with open(file, 'w') as f:
            f.write('Wasserstein distances for bonds:\n')
            bemds = []
            for bond in self.bonds:
                bemds.append(bond.emd(bond.n_iter - 1))
                f.write('%3d,%.3f\n' % (bond.idx + 1, bemds[-1]))
            f.write('Wasserstein distances for angles:\n')
            aemds = []
            for angle in self.angles:
                aemds.append(angle.emd(bond.n_iter - 1))
                f.write('%3d,%.3f\n' % (angle.idx + 1, aemds[-1]))
            f.write('Wasserstein distances for dihedrals:\n')
            demds = []
            for dihedral in self.dihedrals:
                demds.append(dihedral.emd(bond.n_iter - 1))
                f.write('%3d,%.3f\n' % (dihedral.idx + 1, demds[-1]))
            f.write('Average Wasserstein distances for bonds: %.3f\n' % np.mean(bemds))
            f.write('Average Wasserstein distances for angles: %.3f\n' % np.mean(aemds))
            f.write('Average Wasserstein distances for dihedrals: %.3f\n' % np.mean(demds))
        return np.mean(bemds) + np.mean(aemds) + np.mean(demds)

    def generate_mapping_img(self):
        from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
        mol = Chem.RemoveHs(self.mol)
        mol.RemoveAllConformers()
        highlightAtoms = []
        highlightBonds = []
        for bead in self.groups:
            for i, atom in enumerate(bead.atoms):
                idx = atom.GetIdx()
                highlightAtoms.append(idx)
                if i == 0:
                    mol.GetAtomWithIdx(idx).SetProp('atomNote', f'{bead.bead_type}')
            for bond in bead.bonds:
                highlightBonds.append(bond.GetIdx())
        opts = Draw.DrawingOptions()
        opts.bgColor = None
        d = rdMolDraw2D.MolDraw2DSVG(500, 500)
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=highlightAtoms, highlightBonds=highlightBonds)
        d.FinishDrawing()
        with open('mapping.svg', 'w') as f:
            # write the SVG text to the file
            f.write(d.GetDrawingText())

        for bead in self.groups:
            for atom in bead.atoms:
                idx = atom.GetIdx()
                mol.GetAtomWithIdx(idx).SetProp('atomNote', f'B{bead.idx + 1}:{atom.GetIdx() + 1}')
        Draw.MolToFile(mol, 'mapping_detail.svg', size=(500, 500), imageType='svg')

    def save(self, path='.', filename='mapping.pkl', overwrite=True):
        f_al = os.path.join(path, filename)
        if os.path.isfile(f_al) and not overwrite:
            raise RuntimeError(
                f'Path {f_al} already exists. To overwrite, set '
                '`overwrite=True`.'
            )
        store = self.__dict__.copy()
        pickle.dump(store, open(f_al, 'wb'), protocol=4)

    @classmethod
    def load(cls, path='.', filename='mapping.pkl'):
        f_al = os.path.join(path, filename)
        store = pickle.load(open(f_al, 'rb'))
        input = {}
        for key in ['mol2']:
            input[key] = store[key]
        dataset = cls(**input)
        dataset.__dict__.update(**store)
        return dataset
