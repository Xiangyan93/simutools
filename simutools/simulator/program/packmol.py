#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
from simutools.utils.utils import execute
from .base import BaseProgram, which_exe


class Packmol(BaseProgram):
    def __init__(self, exe: str):
        """ A Python object to use Packmol.

        Parameters
        ----------
        exe: Packmol executable file.
        """
        self.exe = exe
        self.check_version()

    def check_version(self):
        """Check the Packmol executable file."""
        which_exe(self.exe)

    def build_uniform(self, pdb_files: List[str], n_mol_list: List[int], output: str,
                      box_size: List[float], tolerance: float = 1.8, seed: int = 0,
                      inp_file='build.inp'):
        """ Build a simulation box that all molecules are uniform distributed.
        A inp file and a PDB file will be generated.

        Parameters
        ----------
        pdb_files: single-molecule PDB files.
        n_mol_list: number of molecules to be added.
        output: the output PDB file.
        box_size: x, y, and z length of the simulation box (nm).
        tolerance: tolerance parameter of Packmol: minimum distance between two atoms.
        seed: random seed.
        inp_file: generated inp file.

        Returns
        -------

        """
        assert len(pdb_files) > 0
        assert len(pdb_files) == len(n_mol_list)
        box_size = [v * 10 for v in box_size]  # nm to A, input unit is nm, but packmol need A.

        inp = (
            'filetype pdb\n'
            f'tolerance {tolerance}\n'
            f'output {output}\n'
            f'seed {seed}\n'
        )
        
        for i, filename in enumerate(pdb_files):
            number = n_mol_list[i]
            box = '0 0 0 %f %f %f' % tuple(box_size)
            inp += (
                f'structure {filename}\n'
                f'number {number}\n'
                f'inside box {box}\n'
                'end structure\n'
            )

        with open(inp_file, 'w') as f:
            f.write(inp)

        # execute(self.exe + ' < ' + inp_file)
        os.system(self.exe + ' < ' + inp_file)
