#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
import string
import shutil
import parmed as pmd
import MDAnalysis as mda
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
        cmds = []
        if not os.path.exists(f'{name}_ob.mol2'):
            cmds.append(f'obabel -:{smiles} {name} -omol2 -p {pH} -r --conformer --gen3d --weighted -h --ff GAFF --partialcharge eem -O {name}_ob.mol2')
        if not os.path.exists(f'{name}.mol2'):
            cmds.append(f'antechamber -i {name}_ob.mol2 -fi mol2 -o {name}.mol2 -rn {resName} -fo mol2 -c bcc -s 1 -nc {charge}')
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
            amber.save(f'{name}.top')
            amber.strip(f':SOL')
            amber.save(f'{name}.gro')
