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


class VirtualSite:
    def __init__(self, beads=None):
        self.beads = beads or []
        self.bead_type = 'TC4'
        self.mass = 0
        self.charge = 0

    def merge(self, beads):
        overlap = False
        for bead in beads:
            if bead in self.beads:
                overlap = True
                break
        if overlap:
            for bead in beads:
                if bead not in self.beads:
                    self.beads.append(bead)
        return overlap

    @property
    def com(self):
        com = 0.
        mass = 0.
        for bead in self.beads:
            com += bead.com * bead.mass
            mass += bead.mass
        return com / mass
    @property
    def idx(self) -> int:
        try:
            return self._idx
        except AttributeError:
            raise AttributeError('Please set idx first.')

    @idx.setter
    def idx(self, idx):
        self._idx = idx


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
            'P1,P1': 'P4r',
            'N6,P1': 'P1',  # bi-Phenol
            'N4a,N4a': 'N4a',  # double ether OCCO
            'C5,C3': 'C4',  #
            'C5,C6': 'C6',  # Thiophenol
            'C5,N4a': 'N2a',  # ether is N4a, benzene-ether is N2a
            'C5,N5a': 'N4a',  # ketone is N5a, benzaldehyde is N4a
            'C5,N6d': 'N6d',  # primary amine
            'C5,P1': 'N6',  # Phenol
            'C5,X1': 'X1',  # Iodobenzene
            'C5,X2': 'X2',  # bromobenzene
            'C5,X3': 'X3',
            'C5,X4e': 'X4e',  # fluorobenzene
        }
        merge_idx = self.atom_idx + group.atom_idx
        print(1, self.bead_type)
        print(self.atom_idx)
        print(2, group.bead_type)
        print(group.atom_idx, group.confirm)
        if group.bead_type in ['C1', 'C3']:
            bead_type = self.bead_type
        elif self.bead_type in ['C1', 'C3']:
            bead_type = group.bead_type
        else:
            bead_type = merge_bead_type.get(f'{self.bead_type},{group.bead_type}')
        assert bead_type is not None
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
                g.bead_type = self.get_bead_type_of_pair_ring_atoms(pair_idx, bridgehead=True)
                g.raw_rings_idx = self.get_raw_rings_idx(self.raw_rings_idx, pair_idx)
                groups.append(g)
            # create operation order for non-bridgehead atoms in ring.
            split_order = []
            for i, ring_idx in enumerate(self.raw_rings_idx):
                ring_idx_without_bridgehead_atoms = self.ring_idx_without_bridgehead_atoms[i]
                # ring contains bridgehead atoms
                if len(ring_idx_without_bridgehead_atoms) < len(ring_idx):
                    if not self.IsLinearConnected(self.mol, ring_idx_without_bridgehead_atoms):
                        split_order.append(10)
                    elif len(ring_idx_without_bridgehead_atoms) % 2 == 0:
                        split_order.append(1)
                    elif len(ring_idx_without_bridgehead_atoms) % 2 == 1:
                        split_order.append(2)
                    else:
                        raise ValueError
                # ring do not contain bridgehead atoms
                else:
                    if len(ring_idx) == 6:
                        split_order.append(4)
                    elif len(ring_idx) == 5:
                        split_order.append(5)
                    else:
                        raise ValueError
                if len(ring_idx) > 6:
                    warnings.warn(f'The molecule contain {len(ring_idx)}-member ring, be careful!')
            # create groups for non-bridgehead atoms in ring.
            for i in np.argsort(split_order):
                so = split_order[i]
                ring_idx = self.raw_rings_idx[i]
                ring_idx_wba = self.ring_idx_without_bridgehead_atoms[i]
                # so == 1 means even number of atoms
                if so == 1:
                    n_groups = int(len(ring_idx_wba) / 2)
                    for j in range(n_groups):
                        g = Bead(mol=self.mol,
                                 graph_heavy=self.graph_heavy,
                                 atom_idx=ring_idx_wba[j * 2:(j + 1) * 2])
                        g.bead_type = self.get_bead_type_of_pair_ring_atoms(ring_idx_wba[j * 2:(j + 1) * 2])
                        g.raw_rings_idx = self.get_raw_rings_idx(self.raw_rings_idx, ring_idx_wba[j * 2:(j + 1) * 2])
                        groups.append(g)
                # so == 2 means odd number of atoms
                elif so == 2:
                    n_groups = int((len(ring_idx_wba) - 1) / 2)
                    n_neighbors1 = len(self.mol.GetAtomWithIdx(ring_idx_wba[0]).GetNeighbors())
                    n_neighbors2 = len(self.mol.GetAtomWithIdx(ring_idx_wba[-1]).GetNeighbors())
                    if n_neighbors1 > n_neighbors2:
                        g = Bead(mol=self.mol,
                                 graph_heavy=self.graph_heavy,
                                 atom_idx=[ring_idx_wba[0]])
                        g.bead_type = self.get_bead_type_of_pair_ring_atoms([ring_idx_wba[0]])
                        g.raw_rings_idx = self.get_raw_rings_idx(self.raw_rings_idx, [ring_idx_wba[0]])
                        groups.append(g)
                        for j in range(n_groups):
                            g = Bead(mol=self.mol,
                                     graph_heavy=self.graph_heavy,
                                     atom_idx=ring_idx_wba[j * 2 + 1:(j + 1) * 2 + 1])
                            g.bead_type = self.get_bead_type_of_pair_ring_atoms(ring_idx_wba[j * 2 + 1:(j + 1) * 2 + 1])
                            g.raw_rings_idx = self.get_raw_rings_idx(self.raw_rings_idx,
                                                                     ring_idx_wba[j * 2 + 1:(j + 1) * 2 + 1])
                            groups.append(g)
                    elif n_neighbors1 < n_neighbors2:
                        g = Bead(mol=self.mol,
                                 graph_heavy=self.graph_heavy,
                                 atom_idx=[ring_idx_wba[-1]])
                        g.bead_type = self.get_bead_type_of_pair_ring_atoms([ring_idx_wba[-1]])
                        g.raw_rings_idx = self.get_raw_rings_idx(self.raw_rings_idx, [ring_idx_wba[-1]])
                        groups.append(g)
                        for j in range(n_groups):
                            g = Bead(mol=self.mol,
                                     graph_heavy=self.graph_heavy,
                                     atom_idx=ring_idx_wba[j * 2:(j + 1) * 2])
                            g.bead_type = self.get_bead_type_of_pair_ring_atoms(ring_idx_wba[j * 2:(j + 1) * 2])
                            g.raw_rings_idx = self.get_raw_rings_idx(self.raw_rings_idx,
                                                                     ring_idx_wba[j * 2:(j + 1) * 2])
                            groups.append(g)
                # six atoms are non-bridged in a 6-member ring
                elif so == 4:
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
                                 atom_idx=ring_idx[j * 2:(j + 1) * 2])
                        g.bead_type = self.get_bead_type_of_pair_ring_atoms(ring_idx[j * 2:(j + 1) * 2])
                        g.raw_rings_idx = self.get_raw_rings_idx(self.raw_rings_idx, ring_idx[j * 2:(j + 1) * 2])
                        groups.append(g)
                # 5 atoms are non-bridged in a 5-member ring
                elif so == 5:
                    assert len(ring_idx) == len(ring_idx_wba)
                    atomic_numbers = [self.mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in ring_idx]
                    group_idx = []
                    if atomic_numbers.count(6) == 4:
                        _ = set(atomic_numbers)
                        _.remove(6)
                        non_carbon_atomic_number = list(_)[0]
                        non_carbon_idx = atomic_numbers.index(non_carbon_atomic_number)
                        group_idx.append([ring_idx.pop(non_carbon_idx)])
                        if non_carbon_idx % 2 == 1:
                            j = ring_idx.pop(0)
                            ring_idx.append(j)
                        group_idx.append(ring_idx[0:2])
                        group_idx.append(ring_idx[2:4])
                        for idx in group_idx:
                            g = Bead(mol=self.mol,
                                     graph_heavy=self.graph_heavy,
                                     atom_idx=idx)
                            g.bead_type = self.get_bead_type_of_pair_ring_atoms(idx)
                            g.raw_rings_idx = self.get_raw_rings_idx(self.raw_rings_idx, idx)
                            groups.append(g)
                    else:
                        raise ValueError
            # create groups for non-used atoms
            used_atoms = []
            for group in groups:
                used_atoms += group.atom_idx
            graph = self.graph.copy()
            graph.remove_nodes_from(used_atoms)
            groups_idx = list(nx.connected_components(graph))
            for i, group_idx in enumerate(groups_idx):
                print(group_idx)
                group_idx = list(group_idx)
                group = Bead(mol=self.mol,
                             graph_heavy=self.graph_heavy,
                             atom_idx=group_idx)
                group.bead_type = self.get_bead_type_of_pair_ring_atoms(group_idx)
                group.raw_rings_idx = self.get_raw_rings_idx(self.raw_rings_idx, group_idx)
                groups.append(group)
            return groups
        else:
            return [self]

    def get_bead_type_of_pair_ring_atoms(self, pair_idx, bridgehead: bool = False):
        for idx in pair_idx:
            atom = self.mol.GetAtomWithIdx(idx)
            assert atom.IsInRing()
        if len(pair_idx) == 1:
            atom = self.mol.GetAtomWithIdx(pair_idx[0])
            if atom.GetAtomicNum() == 6:
                if atom.IsInRing():
                    if atom.GetIsAromatic():
                        return 'C5'
                    else:
                        return 'C3'
            elif atom.GetAtomicNum() == 7:
                if atom.GetTotalNumHs() == 0:
                    return 'N3a'
                elif atom.GetTotalNumHs() == 1:
                    return 'N4'
                elif atom.GetTotalNumHs() == 2:
                    return 'N6d'  # pyrrolidine and pyrrole
            elif atom.GetAtomicNum() == 8:
                if set([a.GetIsAromatic() for a in atom.GetNeighbors()]) == {True}:
                    return 'TN2a'  # furan
                else:
                    return 'N4a'  # tetrahydrofuran
            elif atom.GetAtomicNum() == 16:
                return 'C6'  # sulfur
            else:
                raise ValueError
        elif len(pair_idx) == 2:
            atom1 = self.mol.GetAtomWithIdx(pair_idx[0])
            atom2 = self.mol.GetAtomWithIdx(pair_idx[1])
            bond = self.mol.GetBondBetweenAtoms(atom1.GetIdx(), atom2.GetIdx())
            bond_order = int(bond.GetBondType())
            if sorted([atom1.GetAtomicNum(), atom2.GetAtomicNum()]) == [6, 6]:
                if atom1.GetIsAromatic() and atom2.GetIsAromatic():
                    if bridgehead and set([a.GetIsAromatic() for a in atom1.GetNeighbors()] +
                                          [a.GetIsAromatic() for a in atom2.GetNeighbors()]) == {True}:
                        return 'C5e'  # two bridged atoms of two aromatic ring
                    else:
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
                return 'N4a'  # ether, N3a for C1COCCO1.
            elif sorted([atom1.GetAtomicNum(), atom2.GetAtomicNum()]) == [6, 7]:
                nitrogen_atom = atom1 if atom1.GetAtomicNum() == 7 else atom2
                if nitrogen_atom.GetFormalCharge() == 1:
                    assert not atom1.GetIsAromatic() and not atom2.GetIsAromatic()
                    if nitrogen_atom.GetTotalNumHs() == 0:
                        return 'Q2'
                    elif nitrogen_atom.GetTotalNumHs() == 1:
                        return 'Q2p'
                    elif nitrogen_atom.GetTotalNumHs() == 2:
                        return 'Q3p'
                    elif nitrogen_atom.GetTotalNumHs() == 3:
                        return 'Q4p'
                    else:
                        raise ValueError
                elif atom1.GetIsAromatic() and atom2.GetIsAromatic():
                    return 'N6a'  # pyridine
                elif nitrogen_atom.GetTotalNumHs() == 0:
                    return 'N3a'
                elif nitrogen_atom.GetTotalNumHs() == 1:
                    return 'N4'
                elif nitrogen_atom.GetTotalNumHs() == 2:
                    return 'N6d'  # pyrrolidine and pyrrole
                else:
                    raise ValueError
            else:
                raise ValueError
        elif len(pair_idx) == 3:
            atom1 = self.mol.GetAtomWithIdx(pair_idx[0])
            atom2 = self.mol.GetAtomWithIdx(pair_idx[1])
            atom3 = self.mol.GetAtomWithIdx(pair_idx[2])
            if sorted([atom1.GetAtomicNum(), atom2.GetAtomicNum(), atom3.GetAtomicNum()]) == [6, 6, 7]:
                if atom1.GetAtomicNum() == 7:
                    nitrogen_atom = atom1
                elif atom2.GetAtomicNum() == 7:
                    nitrogen_atom = atom2
                else:
                    nitrogen_atom = atom3
                if nitrogen_atom.FormalCharge() == 0:
                    if nitrogen_atom.GetTotalNumHs() == 0:
                        return
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
        if self.bead_type.startswith('C'):
            bead_rank = 25 - int(self.bead_type[1])
        elif self.bead_type.startswith('N'):
            bead_rank = 19 - int(self.bead_type[1])
        elif self.bead_type.startswith('P'):
            bead_rank = 13 - int(self.bead_type[1])
        elif self.bead_type.startswith('Q'):
            bead_rank = 7 - int(self.bead_type[1])
        elif self.bead_type.startswith('D'):
            bead_rank = 1
        else:
            raise ValueError
        n = 24
        if len(self) == 4:
            return 0
        elif len(self) == 3:
            if self.IsPartInRing:
                return bead_rank
            else:
                return n + bead_rank
        elif len(self) == 2:
            if self.IsPartInRing:
                return 2 * n + bead_rank
            else:
                return 3 * n + bead_rank
        elif len(self) == 1:
            if self.IsPartInRing:
                return 4 * n + bead_rank
            else:
                return 5 * n + bead_rank

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
    def bonds(self) -> List[Chem.Bond]:
        bonds = []
        for i, k in enumerate(self.atom_idx):
            for j in range(i+1, len(self)):
                l = self.atom_idx[j]
                bond = self.mol.GetBondBetweenAtoms(k, l)
                if bond is not None:
                    bonds.append(bond)
        return bonds

    @property
    def atoms_idx_h(self):
        idx_h = []
        for atom in self.atoms:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:
                    idx_h.append(neighbor.GetIdx())
        return idx_h

    @property
    def IsSugar(self) -> bool:
        sugar_smarts = '[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O'
        matches = self.mol.GetSubstructMatches(Chem.MolFromSmarts(sugar_smarts))
        for match in matches:
            _ = set([i in match for i in self.atom_idx])
            # assert _ != {True, False}
            if set(_) == {True} and 8 in [atom.GetAtomicNum() for atom in self.atoms]:
                return True
        else:
            return False

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

    @property
    def confirm(self) -> bool:
        if self.bead_type.startswith('T') or self.bead_type.startswith('S'):
            return True
        else:
            return False

    @staticmethod
    def IsLinearConnected(mol, atom_idx) -> bool:
        for i, idx in enumerate(atom_idx):
            if i != len(atom_idx) - 1:
                if mol.GetBondBetweenAtoms(idx, atom_idx[i + 1]) is None:
                    return False
        return True
