#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple

import pandas as pd
from tap import Tap
import os
import numpy as np
from rdkit import Chem
import parmed as pmd
from copy import deepcopy
from rdkit.Chem import Descriptors
from simutools.forcefields.amber import AMBER
from simutools.simulator.gromacs.gromacs import GROMACS
from simutools.simulator.plumed import PLUMED
from simutools.coarse_grained.mapping import Mapping
from simutools.utils import cd_and_mkdir
from simutools.template import TEMPLATE_DIR
from simutools.utils import execute


class CommonArgs(Tap):
    """This script conduct OPES_METAD between two molecules. And output their binding free energy."""
    filename: str
    """csv file of SMILES."""
    @property
    def mols(self):
        return [Chem.MolFromSmiles(smiles) for smiles in self.smiles]

    @property
    def charge(self):
        return [sum([atom.GetFormalCharge() for atom in mol.GetAtoms()]) for mol in self.mols]

    def process_args(self) -> None:
        df = pd.read_csv(self.filename)
        self.smiles = df['smiles']


def main(args: CommonArgs):
    cd_and_mkdir('debug')
    amber = AMBER()
    for i, s in enumerate(args.smiles):
        print(f'Process {i+1}-th molecule: s')
        amber.build(s, f'mol{i}', charge=args.charge[0], gromacs=True, tip3p=False,
                    resName=f'MOL{i}')


if __name__ == '__main__':
    main(args=CommonArgs().parse_args())
