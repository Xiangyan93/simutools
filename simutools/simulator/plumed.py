#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
import re
from MDAnalysis.topology.ITPParser import ITPParser
import numpy as np
import pandas as pd
from ..utils import execute, cd_and_mkdir, find_index_of_max_unique_abs
from ..template import TEMPLATE_DIR


class PLUMED:
    def __init__(self, plumed_exe: str):
        self.plumed_exe = plumed_exe

    def check_version(self):
        cmd = '%s -h' % self.plumed_exe
        stdout = execute(cmd)[0]
        for line in stdout.decode().splitlines():
            if line.startswith('Usage: plumed'):
                break
        else:
            raise ValueError(f'plumed not valid: {self.plumed_exe}')

    def generate_dat_from_template(self, template: str, output: str = 'plumed.dat', T: float = 298.0,
                                   group1: str = '1', group2: str = '2', biasfactor: int = 6,
                                   barrier: float = 20.0, upper_bound: float = 4.0):
        template = f'{TEMPLATE_DIR}/{template}'
        if not os.path.exists(template):
            raise ValueError(f'dat template not found: {template}')

        with open(template) as f_t:
            contents = f_t.read()
        contents = contents.replace('%T%', str(T)).replace('%group1%', group1).replace('%group2%', group2).\
            replace('%biasfactor%', str(biasfactor)).replace('%barrier%', str(barrier)).\
            replace('%upper_bound%', str(upper_bound))
        with open(output, 'w') as f_mdp:
            f_mdp.write(contents)

    def generate_dat_centripedal(self, groups: List[List[int]], output: str = 'plumed.dat'):
        with open(output, 'w') as f:
            for i, group in enumerate(groups):
                if group[-1] - group[0] == len(group) - 1:
                    f.write(f'COM{i+1}: COM ATOMS={group[0]}-{group[-1]}\n')
                else:
                    f.write('COM%d: COM ATOMS=%s\n' % (i + 1, ','.join([str(g) for g in group])))

            for i, group in enumerate(groups[1:]):
                f.write(f'd{i+1}: DISTANCE ATOMS=COM1,COM{i+2}\n')
            f.write('UPPER_WALLS ARG=%s AT=%s KAPPA=%s\n' % (','.join([f'd{i+1}' for i, _ in enumerate(groups[1:])]),
                                                             ','.join(['0.0'] * (len(groups) - 1)),
                                                             ','.join(['0.5'] * (len(groups) - 1))))

    def get_FES(self, algorithm: Literal['opes'], T: float = 298.0, colvar: str = None, kernels: str = None):
        if algorithm == 'opes':
            kbt = T * 0.0083144621  # kJ/mol
            assert colvar is not None
            assert kernels is not None
            df = pd.read_table(colvar, sep='\s+', comment='#', header=None)

        else:
            raise ValueError(f'unknown algorithm {algorithm}')
