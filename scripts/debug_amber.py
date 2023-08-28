#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from tap import Tap
from rdkit import Chem
from simutools.forcefields.amber import AMBER
from simutools.utils import cd_and_mkdir


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
                    res_name=f'MOL{i}')


if __name__ == '__main__':
    main(args=CommonArgs().parse_args())
