#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, List, Optional, Literal
from simutools.forcefields.base import BaseForceField
from simutools.simulator.program import *
from simutools.simulator.mol3d import Mol3D


def build(packmol: Packmol, simulator: BaseMDProgram, forcefield: Optional[BaseForceField],
          smiles_list: List[str], n_mol_list: List[int], res_name_list: List[str],
          density: float = None):
    # generate PDB and MOL2 files.
    assert len(smiles_list) == len(n_mol_list) == len(res_name_list)
    mols = []
    pdb_files = []
    for i, smiles in enumerate(smiles_list):
        name = res_name_list[i]
        pdb = '%s.pdb' % name
        mol = Mol3D(smiles)
        m = mol.optimize()
        m.write(format='pdb', filename=pdb)
        mols.append(mol)
        pdb_files.append(pdb)
    # get volume, cubic box.
    V = 0.
    for i, mol in enumerate(mols):
        density_ = mol.density * 0.9 if density is None else density  # g/L
        mass = n_mol_list[i] * mol.molwt  # g * mol
        V += 10 / 6.022 * mass / density_  # nm^3
    length = V ** (1 / 3)  # assume cubic box
    # Build initial box using packmol
    packmol.build_uniform(pdb_files=pdb_files, n_mol_list=n_mol_list, output='initial.pdb',
                          box_size=[length] * 3)
    # convert pdb into required format. e.g., gro for GROMACS.
    simulator.convert_pdb(pdb='initial.pdb', tag_out='initial', box_size=[length] * 3)
    # get all force field files.
    if forcefield is not None:
        forcefield.checkout(smiles_list=smiles_list, n_mol_list=n_mol_list,
                            name_list=res_name_list, res_name_list=res_name_list,
                            simulator=simulator)
