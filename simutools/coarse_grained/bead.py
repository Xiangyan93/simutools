#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import networkx as nx
from rdkit import Chem
import numpy as np
import pandas as pd
import warnings
from itertools import permutations
from ..utils import mol_to_nx, merge_lists


class Bead:
    def __init__(self, mol: Chem.Mol,
                 graph_heavy: nx.Graph,
                 atom_idx: List[int]):
        # bead_type: str = None,
        # rings_idx: List[List[int]] = None):
        self.mol = mol
        self.graph_heavy = graph_heavy
        self.atom_idx = atom_idx if atom_idx.__class__ == list else list(atom_idx)
        self.graph = graph_heavy.subgraph(atom_idx)
        self._bead_type = None
        self.raw_rings_idx = []
        self.neighbors = []
        self.check()

    def __len__(self):
        return len(self.atom_idx)

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        else:
            return sorted(self.atom_idx) == sorted(other.atom_idx)

    def __str__(self):
        return ','.join(list(map(str, sorted(self.atom_idx))))

    def substract_bead(self, bead) -> List:
        """Return the left groups after group deletion."""
        groups = []
        for i in bead.atom_idx:
            assert i in self.atom_idx
        graph = self.graph.copy()
        graph.remove_nodes_from(bead.atom_idx)
        subgraphs = list(nx.connected_components(graph))
        for subgraph in subgraphs:
            g = Bead(mol=self.mol, graph_heavy=self.graph_heavy, atom_idx=subgraph)
            groups.append(g)
        return groups

    def add_bead(self, group):
        merge_bead_type = {
            'C3,P1': 'P1',
            'C3,N5a': 'N5a',
            'P1,P1': 'P4r',
            'C5,X3': 'X3',
            'C5,N3a': 'N2a',
        }
        merge_idx = self.atom_idx + group.atom_idx
        # print(1, self.bead_type)
        # print(self.atom_idx)
        # print(2, group.bead_type)
        # print(group.atom_idx)
        if group.bead_type == 'C1':
            bead_type = self.bead_type
        else:
            bead_type = merge_bead_type.get(f'{self.bead_type},{group.bead_type}')
        assert bead_type is not None
        # print(bead_type)
        g = Bead(mol=self.mol,
                 graph_heavy=self.graph_heavy,
                 atom_idx=merge_idx)
        g.bead_type = bead_type
        for ring_idx in group.raw_rings_idx:
            if set(ring_idx) not in [set(g) for g in self.raw_rings_idx]:
                self.raw_rings_idx.append(ring_idx)
        g.raw_rings_idx = self.raw_rings_idx
        assert len(g) < 5
        return g

    def get_raw_rings_idx(self, rings_idx, atoms_idx):
        raw_rings_idx = []
        for ring_idx in rings_idx:
            if set(atoms_idx).issubset(set(ring_idx)):
                raw_rings_idx.append(ring_idx)
        return raw_rings_idx

    def ring_split(self):
        groups = []
        if self.IsAllInRing:
            # create groups for all bridgehead atom pairs.
            for pair_idx in self.bridgehead_atom_pair_idx:
                assert len(pair_idx) == 2
                g = Bead(mol=self.mol,
                         graph_heavy=self.graph_heavy,
                         atom_idx=pair_idx)
                g.bead_type = self.get_bead_type_of_pair_atoms(pair_idx)
                g.raw_rings_idx = self.get_raw_rings_idx(self.raw_rings_idx, pair_idx)
                groups.append(g)
            # create operation for non-bridgehead atoms in ring.
            split_order = []
            for i, ring_idx in enumerate(self.raw_rings_idx):
                ring_idx_without_bridgehead_atoms = self.ring_idx_without_bridgehead_atoms[i]
                if len(ring_idx_without_bridgehead_atoms) < len(ring_idx):
                    if len(ring_idx_without_bridgehead_atoms) == 2:
                        split_order.append(1)
                    elif len(ring_idx_without_bridgehead_atoms) == 4 and len(ring_idx) == 6:
                        split_order.append(2)
                    else:
                        split_order.append(10)
                else:
                    if len(ring_idx) == 6:
                        split_order.append(3)
                    else:
                        raise ValueError
                if len(ring_idx) > 6:
                    warnings.warn(f'The molecule contain {len(ring_idx)}-member ring, be careful!')
            # create groups for non-bridgehead atoms in ring.
            for i in np.argsort(split_order):
                so = split_order[i]
                ring_idx = self.raw_rings_idx[i]
                ring_idx_wba = self.ring_idx_without_bridgehead_atoms[i]
                # so == 1 means only two atoms are non-bridged
                if so == 1:
                    assert len(ring_idx_wba) == 2
                    # if the two atoms are bonded, create one group.
                    if self.mol.GetBondBetweenAtoms(ring_idx_wba[0], ring_idx_wba[1]) is not None:
                        g = Bead(mol=self.mol,
                                 graph_heavy=self.graph_heavy,
                                 atom_idx=ring_idx_wba)
                        g.bead_type = self.get_bead_type_of_pair_atoms(ring_idx_wba)
                        g.raw_rings_idx = self.get_raw_rings_idx(self.raw_rings_idx, ring_idx_wba)
                        groups.append(g)
                    # if the two atoms are not bonded, create two groups.
                    else:
                        for j in range(2):
                            g = Bead(mol=self.mol,
                                     graph_heavy=self.graph_heavy,
                                     atom_idx=[ring_idx_wba[j]])
                            g.bead_type = self.get_bead_type_of_pair_atoms([ring_idx_wba[j]])
                            g.raw_rings_idx = self.get_raw_rings_idx(self.raw_rings_idx, [ring_idx_wba[j]])
                            groups.append(g)
                # four atoms are non-bridged in a 6-member ring
                elif so == 2:
                    for j in range(3):
                        assert self.mol.GetBondBetweenAtoms(ring_idx_wba[j], ring_idx_wba[j + 1]) is not None
                    for j in range(2):
                        g = Bead(mol=self.mol,
                                 graph_heavy=self.graph_heavy,
                                 atom_idx=ring_idx_wba[j * 2:(j + 1) * 2])
                        g.bead_type = self.get_bead_type_of_pair_atoms(ring_idx_wba[j * 2:(j + 1) * 2])
                        g.raw_rings_idx = self.get_raw_rings_idx(self.raw_rings_idx, ring_idx_wba[j * 2:(j + 1) * 2])
                        groups.append(g)
                # six atoms are non-bridged in a 6-member ring
                elif so == 3:
                    assert len(ring_idx) == len(ring_idx_wba)
                    sym1 = 0
                    sym2 = 0
                    for j, idx in enumerate(ring_idx):
                        idx_ = ring_idx[0] if j == len(ring_idx) - 1 else ring_idx[j + 1]
                        atom1 = self.mol.GetAtomWithIdx(idx)
                        atom2 = self.mol.GetAtomWithIdx(idx_)
                        if j % 2 == 0:
                            sym1 += int(self.cal_symmetry(atom1, atom2, ring_idx))
                        elif j % 2 == 1:
                            sym2 += int(self.cal_symmetry(atom1, atom2, ring_idx))
                        else:
                            raise ValueError
                    if sym1 < sym2:
                        j = ring_idx.pop(0)
                        ring_idx.append(j)
                    for j in range(int(len(ring_idx) / 2)):
                        g = Bead(mol=self.mol,
                                 graph_heavy=self.graph_heavy,
                                 atom_idx=ring_idx_wba[j * 2:(j + 1) * 2])
                        g.bead_type = self.get_bead_type_of_pair_atoms(ring_idx_wba[j * 2:(j + 1) * 2])
                        g.raw_rings_idx = self.get_raw_rings_idx(self.raw_rings_idx, ring_idx_wba[j * 2:(j + 1) * 2])
                        groups.append(g)
            # create groups for non-used atoms
            used_atoms = []
            for group in groups:
                used_atoms += group.atom_idx
            graph = self.graph.copy()
            graph.remove_nodes_from(used_atoms)
            groups_idx = list(nx.connected_components(graph))
            for i, group_idx in enumerate(groups_idx):
                group_idx = list(group_idx)
                group = Bead(mol=self.mol,
                             graph_heavy=self.graph_heavy,
                             atom_idx=group_idx)
                group.bead_type = self.get_bead_type_of_pair_atoms(group_idx)
                group.raw_rings_idx = self.get_raw_rings_idx(self.raw_rings_idx, group_idx)
                groups.append(group)
            return groups
        else:
            return [self]

    def get_bead_type_of_pair_atoms(self, pair_idx):
        if len(pair_idx) == 1:
            atom = self.mol.GetAtomWithIdx(pair_idx[0])
            if atom.GetAtomicNum() == 6:
                if atom.IsInRing():
                    if atom.GetIsAromatic():
                        return 'C5'
                    else:
                        return 'C3'
        elif len(pair_idx) == 2:
            atom1 = self.mol.GetAtomWithIdx(pair_idx[0])
            atom2 = self.mol.GetAtomWithIdx(pair_idx[1])
            bond = self.mol.GetBondBetweenAtoms(atom1.GetIdx(), atom2.GetIdx())
            bond_order = int(bond.GetBondType())
            if sorted([atom1.GetAtomicNum(), atom2.GetAtomicNum()]) == [6, 6]:
                if atom1.GetIsAromatic() and atom2.GetIsAromatic():
                    return 'C5'  # aromatic ring
                elif bond_order == 1:
                    return 'C3'  # cyclic alkane
                elif bond_order == 2:
                    return 'C4'  # alkene
                elif bond_order == 3:
                    return 'C6r'  # alkyne
                else:
                    raise ValueError
            elif sorted([atom1.GetAtomicNum(), atom2.GetAtomicNum()]) == [6, 8]:
                assert bond_order == 1
                assert not (atom1.GetIsAromatic() and atom2.GetIsAromatic())
                return 'N3r'  # ether
            elif sorted([atom1.GetAtomicNum(), atom2.GetAtomicNum()]) == [6, 7]:
                if atom1.GetIsAromatic() and atom2.GetIsAromatic():
                    return 'N6a'  # pyridine
                else:
                    raise ValueError
            else:
                raise ValueError
        else:
            raise ValueError

    def is_connected(self, group) -> bool:
        atom_idx = self.atom_idx + group.atom_idx
        return nx.is_connected(group.graph_heavy.subgraph(atom_idx))

    def check(self):
        if self.raw_rings_idx is not None:
            for ring_idx in self.raw_rings_idx:
                for i, idx in enumerate(ring_idx):
                    if i != len(ring_idx) - 1:
                        assert self.graph_heavy.has_edge(idx, ring_idx[i + 1])
                    else:
                        assert self.graph_heavy.has_edge(idx, ring_idx[0])

    def merge_rank(self):
        if len(self) >= 4:
            return 0
        elif len(self) == 3:
            if self.IsPartInRing:
                return 1
            else:
                return 2
        elif len(self) == 2:
            return 3
        elif len(self) == 1:
            return 4

    def cal_symmetry(self, atom1: Chem.Atom, atom2: Chem.Atom, discarded_idx: List[int]):
        an1 = sorted([n.GetAtomicNum() for n in atom1.GetNeighbors() if n.GetIdx() not in discarded_idx])
        an2 = sorted([n.GetAtomicNum() for n in atom2.GetNeighbors() if n.GetIdx() not in discarded_idx])
        if an1 == an2:
            return True
        else:
            return False

    @property
    def idx(self) -> int:
        try:
            return self._idx
        except AttributeError:
            raise AttributeError('Please set idx first.')

    @idx.setter
    def idx(self, idx):
        self._idx = idx

    @property
    def bead_type(self) -> str:
        return self._bead_type

    @bead_type.setter
    def bead_type(self, bead_type):
        self._bead_type = bead_type

    @property
    def mda(self) -> int:
        try:
            return self._mda
        except AttributeError:
            raise AttributeError('Please set mda first.')

    @mda.setter
    def mda(self, mda):
        self._mda = mda

    @property
    def atoms(self) -> List[Chem.Atom]:
        return [self.mol.GetAtomWithIdx(idx) for idx in self.atom_idx]

    @property
    def atoms_idx_h(self):
        idx_h = []
        for atom in self.atoms:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:
                    idx_h.append(neighbor.GetIdx())
        return idx_h

    @property
    def IsAllInRing(self) -> bool:
        v = list(set([a.IsInRing() for a in self.atoms]))
        if len(v) == 1:
            return v[0]
        else:
            return False

    @property
    def IsPartInRing(self) -> bool:
        v = list(set([a.IsInRing() for a in self.atoms]))
        if len(v) == 1:
            return v[0]
        else:
            return True

    @property
    def IsPlane(self) -> bool:
        v = list(set([a.GetIsAromatic() for a in self.atoms]))
        if len(v) == 1:
            return v[0]
        else:
            return False

    @property
    def IsPartAromatic(self) -> bool:
        v = list(set([a.GetIsAromatic() for a in self.atoms]))
        if len(v) == 1:
            return v[0]
        else:
            return True

    @property
    def IsAlkane(self) -> bool:
        v = list(set([a.GetAtomicNum() for a in self.atoms]))
        if len(v) == 1 and v[0] == 6:
            return True
        else:
            return False

    @property
    def bridgehead_atom_pair_idx(self) -> List[List[int]]:
        if hasattr(self, '_bridgehead_atom_pair_idx'):
            return self._bridgehead_atom_pair_idx
        bridgehead_pair_idx = []
        for i, r1_idx in enumerate(self.raw_rings_idx):
            for j in range(i + 1, len(self.raw_rings_idx)):
                r2_idx = self.raw_rings_idx[j]
                intersection = list(set(r1_idx).intersection(r2_idx))
                if len(intersection) == 2:
                    atom1 = self.mol.GetAtomWithIdx(intersection[0])
                    atom2 = self.mol.GetAtomWithIdx(intersection[1])
                    assert self.mol.GetBondBetweenAtoms(atom1.GetIdx(),
                                                        atom2.GetIdx()) is not None, 'Unconsidered structure'
                    bridgehead_pair_idx.append(intersection)
                else:
                    continue
        self._bridgehead_atom_pair_idx = bridgehead_pair_idx
        return bridgehead_pair_idx

    @property
    def bridgehead_atom_idx(self) -> List[int]:
        return np.array(self.bridgehead_atom_pair_idx).ravel().tolist()

    @property
    def ring_idx_without_bridgehead_atoms(self) -> List[List[int]]:
        if hasattr(self, '_ring_idx_without_bridgeatoms'):
            return self._ring_idx_without_bridgeatoms
        unused_ring_idx = []
        for ring_idx in self.raw_rings_idx:
            r_idx = ring_idx.copy()
            for idx in self.bridgehead_atom_idx:
                if idx in r_idx:
                    r_idx.remove(idx)
            if len(r_idx) < len(ring_idx) and len(r_idx) != 2:
                for i in range(100):
                    atom1 = self.mol.GetAtomWithIdx(r_idx[0])
                    atom2 = self.mol.GetAtomWithIdx(r_idx[-1])
                    if self.mol.GetBondBetweenAtoms(atom1.GetIdx(), atom2.GetIdx()) is not None:
                        j = r_idx.pop(0)
                        r_idx.append(j)
                    else:
                        break
            unused_ring_idx.append(r_idx)
        self._ring_idx_without_bridgeatoms = unused_ring_idx
        return unused_ring_idx

    @property
    def smarts(self) -> str:
        return Chem.MolFragmentToSmarts(self.mol, self.atom_idx)

    @property
    def mass(self):
        if len(self) == 2:
            return 36
        elif len(self) == 3:
            return 54
        elif len(self) == 4:
            return 72
        else:
            raise ValueError

    @property
    def charge(self):
        return sum([atom.GetFormalCharge() for atom in self.atoms])
