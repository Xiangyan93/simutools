#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import List
from simutools.utils.utils import execute


def which_exe(exe: str):
    """Check the Packmol executable file."""
    cmd = f'which {exe}'
    out = execute(cmd)[0].decode()
    assert len(out) != 0, f'{exe} not found.'
    print(f'{exe} found.')


class BaseProgram(ABC):
    @abstractmethod
    def check_version(self):
        pass


class BaseMDProgram(BaseProgram):
    @abstractmethod
    def convert_pdb(self, pdb: str, tag_out: str, box_size: List[float], **kwargs):
        """ Convert PDB into the required format for molecular simulation.

        Parameters
        ----------
        pdb: PDB file.
        tag_out: the tag of output file.
        box_size: the size of simulation box (nm).
        kwargs

        Returns
        -------

        """
        pass
    '''
    @abstractmethod
    def set_parameters(self, **kwargs):
        """set up simulation parameters."""
        pass

    @abstractmethod
    def energy_minimization(self, tag_in: str, tag_out: str, exe: bool = False) -> List[str]:
        """ Energy minimization.

        Parameters
        ----------
        tag_in: the tag of input file.
        tag_out: the tag of output file.
        exe: if true, execute the commands. Otherwise, only return the commands.

        Returns
        -------

        """
        pass

    @abstractmethod
    def annealing(self, tag_in: str, tag_out: str, T: int, T_annealing: int, exe: bool = False) -> List[str]:
        """ Annealing simulation. The temperature is increased from 0 to T_annealing and then decreased to T.

        Parameters
        ----------
        tag_in: the tag of input file.
        tag_out: the tag of output file.
        T: target simulation temperature (K).
        T_annealing: annealing temperature (K).
        exe: if true, execute the commands. Otherwise, only return the commands.

        Returns
        -------

        """
        pass

    @abstractmethod
    def equilibrium(self, tag_in: str, tag_out: str, exe: bool = False):
        pass

    @abstractmethod
    def production(self, tag_in: str, tag_out: str, exe: bool = False):
        pass
    '''