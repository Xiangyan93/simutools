#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List
import os
import re
import shutil
import parmed as pmd
from simutools.template import TEMPLATE_DIR
from simutools.simulator.mol3d import Mol3D
from simutools.simulator.program import *
from simutools.utils.utils import execute
from simutools.utils.rdkit import get_format_charge
from .base import BaseForceField


class AMBER(BaseForceField):
    def __init__(self, exe: str):
        self.exe = exe
        self.prefix = exe.replace('antechamber', '')
        self.check_version()
        self.tmp_dir = f'{TEMPLATE_DIR}/amber'

    def check_version(self):
        """Check the Antechamber executable file."""
        cmd = f'{self.exe}'
        out = execute(cmd)[0].decode()
        assert 'antechamber' in out, f'{self.exe} is not a valid antechamber executable file.'
        print(f'{self.exe} is valid.')

    def checkout(self, smiles_list: List[str], n_mol_list: List[int], name_list: List[str],
                 res_name_list: List[str], simulator: BaseMDProgram, outname: str = 'checkout'):
        """ Checkout AMBER force field for a list of molecules.

        Parameters
        ----------
        smiles_list: the SMILES of molecules.
        n_mol_list: the number of molecules.
        name_list: the name of molecules.
        res_name_list: the residue name of molecules.
        simulator: determine the format of output force field files.
        outname: output name of gro and top files.

        Returns
        -------

        """
        assert len(set(name_list)) == len(name_list)
        for i, smiles in enumerate(smiles_list):
            name = name_list[i]
            self.build(smiles=smiles, name=name, res_name=res_name_list[i])
            if i == 0:
                ff = pmd.load_file(f'{name}.prmtop', f'{name}.inpcrd')
            else:
                ff += pmd.load_file(f'{name}.prmtop', f'{name}.inpcrd')
        if isinstance(simulator, GROMACS):
            ff.save(f'{outname}.gro', overwrite=True)
            ff.save(f'{outname}.top', overwrite=True)
            for i, smiles in enumerate(smiles_list):
                simulator.modify_top_mol_numbers(top=f'{outname}.top', outtop=f'{outname}.top',
                                                 mol_name=res_name_list[i], n_mol=n_mol_list[i])
        else:
            raise ValueError('Only GROMACS is valid now.')

    def build(self, smiles: str, name: str, res_name, pH: float = None):
        """ Generate AMBER force field parameters for a single molecule.
        A series of AMBER force field files will be generated.

        Parameters
        ----------
        smiles: SMILES string of the molecule to be generated.
        name: name of the molecule.
        res_name: residue name.
        pH: pH value, not sure how it works in openbabel.

        Returns
        -------

        """
        # use tip3p water.
        if smiles in self.cache_molecules:
            ff = pmd.load_file(filename=f'{self.tmp_dir}/{self.cache_molecules[smiles]}.top',
                               xyz=f'{self.tmp_dir}/{self.cache_molecules[smiles]}.gro')
            ff.save(f'{name}.prmtop', overwrite=True)
            ff.save(f'{name}.inpcrd', overwrite=True)
            ff.write_pdb(f'{name}.pdb')
        else:
            charge = get_format_charge(smiles)
            # create tleap file
            shutil.copyfile(f'{self.tmp_dir}/tleap.in', 'tleap.in')
            with open('tleap.in') as f:
                contents = f.read()
            contents = contents.replace('template', name)
            with open('tleap.in', 'w') as f:
                f.write(contents)
            # generate random residue name if it is not given.
            assert len(res_name) == 3, 'The residue name must be 3 characters.'
            # generate mol2 file
            if not os.path.exists(f'{name}_ob.mol2'):
                mol3d = Mol3D(smiles=smiles)
                mol = mol3d.optimize(charge_model='eem', force_field='GAFF', pH=pH)
                mol.write('mol2', f'{name}_ob.mol2')
                mol.write('pdb', f'{name}_ob.pdb')
            # generate the AMBER force field files using antechamber
            cmds = []
            if not os.path.exists(f'{name}.mol2'):
                cmds.append(f'{self.exe} -i {name}_ob.mol2 -fi mol2 -o {name}.mol2 -rn {res_name} -fo mol2 -s 1 '
                            f'-nc {charge} -at gaff2 -dr no')
            cmds += [
                f'{self.prefix}parmchk2 -i {name}.mol2 -f mol2 -o {name}.frcmod -s gaff2 -a Y',
                f'{self.prefix}tleap -f tleap.in'
            ]
            print(f'Building AMBER files for molecule {smiles}')
            for cmd in cmds:
                execute(cmd)
            # output PDB file with correct residue name.
            with open(f'{name}_ob.pdb', 'r') as input_file, open(f'{name}.pdb', 'w') as output_file:
                for line in input_file:
                    modified_line = re.sub(r'UNL', res_name, line)
                    output_file.write(modified_line)

    @property
    def cache_molecules(self):
        return {
            'O': 'tip3p',
            '[Na+]': 'sodium',
            '[Cl-]': 'chloride'
        }
