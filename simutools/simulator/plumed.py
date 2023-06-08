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
                                   group1: str = '1', group2: str = '2', biasfactor: int = 6):
        template = f'{TEMPLATE_DIR}/{template}'
        if not os.path.exists(template):
            raise ValueError(f'dat template not found: {template}')

        with open(template) as f_t:
            contents = f_t.read()
        contents = contents.replace('%T%', str(T)).replace('%group1%', group1).replace('%group2%', group2).\
            replace('%biasfactor%', str(biasfactor))
        with open(output, 'w') as f_mdp:
            f_mdp.write(contents)

    def get_FES(self, algorithm: Literal['opes'], T: float = 298.0, colvar: str = None, kernels: str = None):
        if algorithm == 'opes':
            kbt = T * 0.0083144621  # kJ/mol
            assert colvar is not None
            assert kernels is not None
            df = pd.read_table(colvar, sep='\s+', comment='#', header=None)

        else:
            raise ValueError(f'unknown algorithm {algorithm}')
