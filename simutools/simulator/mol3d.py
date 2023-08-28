#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import random
from openbabel import openbabel
import openbabel.pybel as pybel
from simutools.utils.utils import estimate_density_from_formula


class Mol3D:
    """Generate 3D molecular file (mol2, pdb, etc) from SMILES using openbabel."""
    def __init__(self, smiles: str, seed: int = 0):
        try:
            self.mol = pybel.readstring('smi', smiles)
            self.mol.addh()
        except:
            raise RuntimeError('Cannot create molecule from SMILES using openbabel.')

        # 3D coordinates generate.
        self.smiles = smiles
        self.seed = seed

    @property
    def charge(self) -> int:
        """formal charge"""
        return self.mol.charge

    @property
    def spin(self) -> int:
        return self.mol.spin

    @property
    def n_atoms(self) -> int:
        return len(self.mol.atoms)

    @property
    def molwt(self) -> float:
        return self.mol.molwt

    @property
    def formula(self) -> str:
        return self.mol.formula

    @property
    def density(self) -> float:
        """estimated density based on formula."""
        return estimate_density_from_formula(self.formula)

    def optimize(self, minimize: bool = True,
                 charge_model: Literal['Gasteiger', 'mmff94', 'eem', 'qeq', 'qtpie'] = 'mmff94',
                 force_field: Literal['mmff94', 'GAFF', 'uff'] = 'mmff94',
                 pH: float = None, N_search: int = 100) -> pybel.Molecule:
        """Generate the global minimized conformation.

        Parameters
        ----------
        minimize: whether to minimize the energy of the conformer.
        charge_model: charge method to assign partial charges.
        force_field: force field for energy minimization.
        pH: pH values.
        N_search: Number of steps for conformation rotor search.

        Returns
        -------
        Openbabel molecule object.
        """
        mol = pybel.readstring('smi', self.smiles)
        # add hydrogen
        mol.addh()
        # make 3D
        mol.make3D()
        openbabel.O
        if minimize:
            mol.localopt()
            # charge method
            charge_model = openbabel.OBChargeModel.FindType(charge_model)
            charge_model.ComputeCharges(mol.OBMol)
            # pH value
            if pH is not None:
                mol.OBMol.CorrectForPH(pH)
            # force field
            ff = openbabel.OBForceField.FindForceField(force_field)
            ff.Setup(mol.OBMol)
            #  systematic rotor search for lowest energy conformer
            ff.SystematicRotorSearch(N_search)
            # ff.WeightedRotorSearch(100, 100)
        return mol

    def conformers_optimize(self, n_select: int = 10, n_try: int = 10) -> List[pybel.Molecule]:
        if n_select == 0:
            return []
        random.seed(self.seed)
        ff = openbabel.OBForceField.FindForceField('mmff94')
        if n_try is None or n_try < n_select:
            n_try = n_select

        x_list = []
        for atom in self.mol.atoms:
            for x in atom.coords:
                x_list.append(x)
        xmin, xmax = min(x_list), max(x_list)
        xspan = xmax - xmin

        conformers = []
        for i in range(n_try):
            conformer = self.optimize(minimize=False)

            for atom in conformer.atoms:
                obatom = atom.OBAtom
                random_coord = [(random.random() * xspan + xmin) * k for k in [2, 1, 0.5]]
                obatom.SetVector(*random_coord)

            conformer.localopt()
            ff.Setup(conformer.OBMol)
            conformer.OBMol.SetEnergy(ff.Energy())
            conformers.append(conformer)
        conformers.sort(key=lambda x: x.energy)
        return conformers[:n_select]
