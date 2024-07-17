#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, List, Optional, Literal
import os
import math
import numpy as np
from simutools.simulator.mol3d import Mol3D
from .base import BaseProgram, which_exe


class Gaussian(BaseProgram):
    def __init__(self, exe: str):
        """ A Python object to use Gaussian for quantum chemistry calculation.

        Parameters
        ----------
        exe: Gaussian executable file.
        method: Gaussian input.
        basis: Gaussian input.
        n_jobs: number of CPU cores.
        mem: memory (GB)
        """
        self.exe = exe
        self.gauss_dir = os.path.dirname(exe)

    def check_version(self):
        which_exe(self.exe)

    @staticmethod
    def create_gjf(smiles: str, command: str, path: str, name: str = 'gaussian',
                   n_jobs: int = 1, mem: int = None, seed: int = 0, get_command: bool = False) -> str:
        """ Get the gjf script for GAUSSIAN.

        Parameters
        ----------
        smiles: the SMILES of a molecule.
        command: GAUSSIAN command. e.g.: # B3LYP
        path: absolute path of the job.
        name: job name.
        n_jobs: number of CPU cores.
        mem: memory (GB)
        seed: random seed for molecular 3D coordinates.
        get_command: if true, only return the gjf file in string format, but do not create the gjf file.

        Returns
        -------
        gjf script.
        """
        mol3d = Mol3D(smiles, seed=seed)
        contents = '%%nprocshared=%d\n' % n_jobs
        if mem is not None:
            contents += '%%mem=%dMB\n' % mem
        contents += f'%%chk={path}/{name}.chk\n' + \
                    f'# {command}\n' + \
                    '\n' + \
                    'Title\n' + \
                    '\n' + \
                    f'{mol3d.charge} {mol3d.spin}\n'
        for atom_line in mol3d.optimize().write('xyz').splitlines()[2:]:
            contents += atom_line + '\n'
        contents += '\n'
        if not get_command:
            with open(os.path.join(path, '%s.gjf' % name), 'w') as f:
                f.write(contents)
        return contents

    def create_gjf_cv(self, smiles: str, path: str, name: str = 'gaussian',
                      n_jobs: int = 1, mem: int = None, seed: int = 0, method: str = 'B3LYP', basis: str = '6-31G*',
                      scale: float = 0.9613, T_list: List[float] = None):
        """ Get the gjf script for GAUSSIAN to compute the intramolecular contribution of heat capacity.
            references:
            1. Ayala, P. Y.; Schlegel, H. B. Identification and Treatment of Internal Rotation in Normal Mode
               Vibrational Analysis. J. Chem. Phys. 1998, 108, 2314−2325.
            2. Kesharwani, M. K.; Brauer, B.; Martin, J. M. L. Frequency and Zero-Point Vibrational Energy Scale Factors
               for Double-Hybrid Density Functionals (and Other Selected Methods): Can Anharmonic Force Fields Be
               Avoided? J. Phys. Chem. A 2015, 119, 1701−1714.
            3. Gong Z, Wu Y, Wu L, et al. Predicting thermodynamic properties of alkanes by high-throughput force field
               simulation and machine learning[J]. J. Chem. Inf. Model. 2018, 58(12), 2502-2516.
        Parameters
        ----------
        smiles: the SMILES of a molecule.
        path: absolute path of the job.
        name: job name.
        n_jobs: number of CPU cores.
        mem: memory (GB)
        seed: random seed for molecular 3D coordinates.
        method: Gaussian input.
        basis: Gaussian input.
        scale: scale factor, 0.9613 is suggested by ref 2.
        T_list: A list of temperatures for the computations.

        Returns
        -------

        """
        command = f'opt freq=hindrot pop=full {method} {basis} scale=%.4f temperature=%.2f\n' % (scale, T_list[0])
        contents = self.create_gjf(smiles=smiles, command=command, path=path, name=name, n_jobs=n_jobs, mem=mem,
                                   seed=seed, get_command=True)
        for T in T_list[1:]:
            contents += '--Link1--\n' + \
                    f'%%chk=%(path)s/%(name)s.chk\n'+ \
                    '# freq=(readfc,hindrot) geom=allcheck scale=%.4f temperature=%.2f\n\n' % (scale, T)
        with open(os.path.join(path, '%s.gjf' % name), 'w') as f:
            f.write(contents)
        return contents

    def get_slurm_commands(self, file: str, tmp_dir: str):
        assert not os.path.exists(tmp_dir)
        commands = ['JOB_DIR=' + tmp_dir,
                    'mkdir -p ${JOB_DIR}',
                    'export GAUSS_EXEDIR=%s:%s/bsd' % (self.gauss_dir, self.gauss_dir),
                    'export GAUSS_SCRDIR=${JOB_DIR}', '%s %s' % (self.exe, file),
                    'rm -rf ${JOB_DIR}']
        return commands

    @staticmethod
    def analyze(log: str) -> Optional[Dict]:
        """ Analyze the Gaussian log file.

        Parameters
        ----------
        log: the Gaussian log file.

        Returns
        -------
        The results are returned by a dict.
        """
        if not os.path.exists(log):
            return None
        content = open(log).read()
        if content.find('Normal termination') == -1:
            return None
        if content.find('Error termination') > -1:
            return None
        if content.find('imaginary frequencies') > -1:
            return 'imaginary frequencies'

        result = {'EE': None, 'EE+ZPE': None, 'T': [], 'scale': [], 'cv': [], 'cv_corrected': [], 'FE': []}
        f = open(log)
        while True:
            line = f.readline()
            if line == '':
                break

            if line.strip().startswith('- Thermochemistry -'):
                line = f.readline()
                line = f.readline()
                T = float(line.strip().split()[1])
                result['T'].append(T)
                line = f.readline()
                if line.strip().startswith('Thermochemistry will use frequencies scaled by'):
                    scale = float(line.strip().split()[-1][:-1])
                else:
                    scale = 1
                result['scale'].append(scale)
            elif line.strip().startswith('E (Thermal)             CV                S'):
                line = f.readline()
                line = f.readline()
                Cv = float(line.strip().split()[2]) * 4.184
                result['cv'].append(Cv)
                line = f.readline()
                if line.strip().startswith('Corrected for'):
                    line = f.readline()
                    Cv_corr = float(line.strip().split()[3]) * 4.184
                    # Cv_corr might by NaN, this is a bug of gaussian
                    if math.isnan(Cv_corr):
                        Cv_corr = Cv
                    result['cv_corrected'].append(Cv_corr)
                else:
                    result['cv_corrected'].append(Cv)
            elif line.strip().startswith('Sum of electronic and thermal Free Energies='):
                fe = float(line.strip().split()[7])
                result['FE'].append(fe)
            elif result['EE+ZPE'] is None and line.strip().startswith('Sum of electronic and zero-point Energies='):
                ee_zpe = float(line.strip().split()[6])
                result['EE+ZPE'] = ee_zpe
            elif line.strip().startswith(' SCF Done:'):
                ee = float(line.strip().split()[3])
                result['EE'] = ee
        return result

    def prepare(self, smiles: str, path: str, name: str = 'gaussian',
                task: Literal['qm_cv'] = 'qm_cv',
                tmp_dir: str = '.', seed: int = 0) -> List[str]:
        mol3d = Mol3D(smiles, seed=seed)
        print('Build GAUSSIAN input file.')
        if task == 'qm_cv':
            self.create_gjf_cv(mol3d, path=path, name=name, T_list=list(range(100, 1400, 100)))
            file = os.path.join(path, '%s.gjf' % name)
            return self.get_slurm_commands(file, tmp_dir)
        else:
            return []