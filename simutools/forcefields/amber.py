#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
import string
import shutil
import re
import parmed as pmd
import MDAnalysis as mda
from rdkit import Chem
from ..utils import execute
from ..template import TEMPLATE_DIR


class AMBER:
    def __init__(self):
        self.check_exe()

    def check_exe(self):
        cmds = ['obabel', 'antechamber']
        for cmd in cmds:
            execute(cmd)

    def build(self, smiles: str, name: str, pH: float = 7, charge: float = 0., resName: str = None,
              gromacs: bool = False, tip3p: bool = True):
        shutil.copyfile(f'{TEMPLATE_DIR}/tleap.in', 'tleap.in')
        with open('tleap.in') as f:
            contents = f.read()
        contents = contents.replace('template', name)
        with open('tleap.in', 'w') as f:
            f.write(contents)

        if resName is None:
            resName = ''.join(random.choice(string.ascii_uppercase) for _ in range(3))
        if not os.path.exists(f'{name}_ob.mol2'):
            cmd = f'obabel -:{smiles} {name} -omol2 -p {pH} -r --conformer --gen3d --weighted -h --ff GAFF ' \
                  f'--partialcharge eem -O {name}_ob.mol2'
            execute(cmd)
        # mol = Chem.MolFromSmiles(smiles)
        # self.mol2_fix(mol, f'{name}_ob.mol2')
        cmds = []
        if not os.path.exists(f'{name}.mol2'):
            cmds.append(f'antechamber -i {name}_ob.mol2 -fi mol2 -o {name}.mol2 -rn {resName} -fo mol2 -s 1 '
                        f'-nc {charge} -at gaff2 -dr no')
        cmds += [
            f'parmchk2 -i {name}.mol2 -f mol2 -o {name}.frcmod -s gaff2 -a Y',
            f'tleap -f tleap.in'
        ]
        print(f'Building AMBER files for molecule {smiles}')
        for cmd in cmds:
            execute(cmd)

        if gromacs:
            amber = pmd.load_file(f'{name}.prmtop', f'{name}.inpcrd')
            if charge > 0:
                for i in range(int(charge)):
                    amber += pmd.load_file(f'{TEMPLATE_DIR}/chloride.top', xyz=f'{TEMPLATE_DIR}/chloride.gro')
            elif charge < 0:
                for i in range(int(abs(charge))):
                    amber += pmd.load_file(f'{TEMPLATE_DIR}/sodium.top', xyz=f'{TEMPLATE_DIR}/sodium.gro')
            if tip3p:
                amber += pmd.load_file(f'{TEMPLATE_DIR}/tip3p.top', xyz=f'{TEMPLATE_DIR}/tip3p.gro')
            amber.save(f'{name}.top', overwrite=True)
            amber.strip(f':SOL')
            amber.save(f'{name}.gro', overwrite=True)
            return amber

    @staticmethod
    def get_bonds_modify(mol, smarts, single_bonds, double_bonds, single_bonds_idx, double_bonds_idx):
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        for match in matches:
            print(match)
            for i, j in single_bonds_idx:
                if match[i] < match[j]:
                    single_bonds.append((match[i], match[j]))
                else:
                    single_bonds.append((match[j], match[i]))
            for i, j in double_bonds_idx:
                if match[i] < match[j]:
                    double_bonds.append((match[i], match[j]))
                else:
                    double_bonds.append((match[j], match[i]))

    def mol2_fix(self, mol: Chem.Mol, mol2: str):
        """antechamber use -dr no """
        single_bonds = []
        double_bonds = []
        # Furan
        #smarts = 'c1ccco1'
        #single_bonds_idx = [(0, 4), (1, 2), (3, 4)]
        #double_bonds_idx = [(0, 1), (2, 3)]
        #self.get_bonds_modify(mol, smarts, single_bonds, double_bonds, single_bonds_idx, double_bonds_idx)
        # clofazimine
        smarts = 'N=c1cc2[n;D3]cc[n;D2]c-2cc1'
        single_bonds_idx = [(1, 2), (3, 4), (4, 5), (6, 7), (3, 8), (8, 9), (1, 10)]
        double_bonds_idx = [(2, 3), (7, 8), (9, 10)]
        self.get_bonds_modify(mol, smarts, single_bonds, double_bonds, single_bonds_idx, double_bonds_idx)
        # Pranlukast and moxifloxacin
        for smarts in ['c1cccc2c(=O)ccoc12', 'c1cccc2c(=O)cc[n;D3]c12']:
            single_bonds_idx = [(4, 5), (5, 7), (8, 9), (9, 10)]
            double_bonds_idx = [(5, 6), (7, 8)]
            self.get_bonds_modify(mol, smarts, single_bonds, double_bonds, single_bonds_idx, double_bonds_idx)
        # Caffeine
        smarts = 'n1c(=O)ccnc1=O'
        single_bonds_idx = [(0, 1), (1, 3), (4, 5), (5, 6), (0, 6)]
        double_bonds_idx = [(1, 2), (6, 7)]
        self.get_bonds_modify(mol, smarts, single_bonds, double_bonds, single_bonds_idx, double_bonds_idx)
        # rhodamine_B
        smarts = 'c1cc2ccc(=[N+](CC)CC)cc-2oc1'
        single_bonds_idx = [(5, 6), (4, 5), (5, 11), (2, 3), (12, 13), (13, 14)]
        double_bonds_idx = [(1, 2), (3, 4), (11, 12)]
        self.get_bonds_modify(mol, smarts, single_bonds, double_bonds, single_bonds_idx, double_bonds_idx)
        #
        smarts = 'c1nc(=O)[n;D3]cc1'
        single_bonds_idx = [(1, 2), (2, 4), (4, 5), (0, 6)]
        double_bonds_idx = [(0, 1), (5, 6)]
        self.get_bonds_modify(mol, smarts, single_bonds, double_bonds, single_bonds_idx, double_bonds_idx)
        # risperidone
        smarts = 'c1[n;H0;D2]c[n;H0;D3]c(=O)c1'
        single_bonds_idx = [(0, 1), (2, 3), (3, 4), (4, 6)]
        double_bonds_idx = [(1, 2), (0, 6)]
        self.get_bonds_modify(mol, smarts, single_bonds, double_bonds, single_bonds_idx, double_bonds_idx)
        smarts = 'c1[n;D2;H0]occ1'
        single_bonds_idx = [(1, 2), (2, 3), (4, 0)]
        double_bonds_idx = [(0, 1)]
        self.get_bonds_modify(mol, smarts, single_bonds, double_bonds, single_bonds_idx, double_bonds_idx)
        with open(mol2, 'r') as file:
            contents = file.read()
        # Perform the replacement using regular expressions
        for i, j in single_bonds:
            contents = re.sub(r"%d\s+%d\s+ar" % (i + 1, j + 1), "%d    %d    1" % (i + 1, j + 1), contents)
        for i, j in double_bonds:
            contents = re.sub(r"%d\s+%d\s+ar" % (i + 1, j + 1), "%d    %d    2" % (i + 1, j + 1), contents)

        # Write the updated contents to the output file
        with open(mol2, 'w') as file:
            file.write(contents)
