#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
import re
from MDAnalysis.topology.ITPParser import ITPParser
import numpy as np
from ...utils import execute, cd_and_mkdir, find_index_of_max_unique_abs
from ...template import TEMPLATE_DIR


class GROMACS:
    def __init__(self, gmx_exe_mdrun: str, gmx_exe_analysis: str = None):
        self.gmx_analysis = gmx_exe_analysis or gmx_exe_mdrun
        self.gmx_mdrun = gmx_exe_mdrun + ' mdrun'
        self.check_version()
        # TODO temporary hack for dielectric constant in mdp
        # self._DIELECTRIC = 1.0
        # TODO temporary hack for LJ96 function
        # self._LJ96 = False

    def check_version(self):
        for gmx in [self.gmx_analysis, self.gmx_mdrun]:
            cmd = '%s --version' % gmx
            stdout = execute(cmd)[0]
            for line in stdout.decode().splitlines():
                if line.startswith('GROMACS version'):
                    break
            else:
                raise ValueError(f'gmx not valid: {self.gmx_analysis}')

        # self.version = line.strip().split()[-1]
        # if not (self.version.startswith('2016') or self.version.startswith('2018') or self.version.startswith('2019')):
        #     raise GmxError('Supported GROMACS versions: 2016.x, 2018.x, 2019.x')
        # self.majorversion = self.version.split('.')[0]

    @staticmethod
    def generate_top(file: str, include_itps: List[str], mol_name: List[str], mol_number: List[int]):
        with open(file, 'w') as f:
            for include_itp in include_itps:
                f.write(f'#include "{include_itp}"\n')
            f.write('\n[ system ]\n')
            info = ', '.join([f'{name} {mol_number[i]}' for i, name in enumerate(mol_name)])
            f.write(f'{info}\n')
            f.write('\n[ molecules ]\n')
            for i, name in enumerate(mol_name):
                f.write(f'{name}         {mol_number[i]}\n')
        return

    def generate_mdp_from_template(self, template: str, mdp_out: str = 'grompp.mdp', T: float = 298, P: float = 1,
                                   nsteps: int = 10000, dt: float = 0.001, TANNEAL: int = 800, nstenergy: int = 100,
                                   nstxout: int = 0, nstvout: int = 0, nstxtcout: int = 10000, xtcgrps: str = 'System',
                                   restart: bool = False, tcoupl: str = 'langevin', pcoupl: str = 'parrinello-rahman',
                                   gen_seed: int = -1, constraints: str = 'h-bonds', ppm=0, dielectric=1.0, tau_t=None,
                                   tau_p=None, compressibility='4e-5', coulombtype='PME', rcoulomb='1.2', rvdw='1.2',
                                   nstlist=None):
        template = f'{TEMPLATE_DIR}/{template}'
        if not os.path.exists(template):
            raise ValueError(f'mdp template not found: {template}')

        if tcoupl.lower() == 'langevin':
            integrator = 'sd'
            tcoupl = 'no'
            tau_t = str(0.001 / dt) if tau_t is None else tau_t  # inverse friction coefficient
        elif tcoupl.lower() == 'nose-hoover':
            integrator = 'md'
            tcoupl = 'nose-hoover'
            tau_t = '0.5' if tau_t is None else tau_t
        elif tcoupl.lower() == 'v-rescale':
            integrator = 'md'
            tcoupl = 'v-rescale'
            tau_t = '0.1' if tau_t is None else tau_t
        else:
            raise Exception('Invalid tcoupl, should be one of langvein, nose-hoover, v-rescale')

        if pcoupl.lower() == 'berendsen':
            tau_p = '1' if tau_p is None else tau_p
        elif pcoupl.lower() == 'parrinello-rahman':
            tau_p = '5' if tau_p is None else tau_p
        elif pcoupl.lower() == 'mttk':
            tau_p = '5' if tau_p is None else tau_p
            constraints = 'none'
        else:
            raise Exception('Invalid pcoupl, should be one of berendsen, parrinello-rahman, mttk')

        if restart:
            genvel = 'no'
            continuation = 'yes'
        else:
            genvel = 'yes'
            continuation = 'no'

        nstlist = max(1, int(0.01 / dt)) if nstlist is None else nstlist

        with open(template) as f_t:
            contents = f_t.read()
        contents = contents.replace('%T%', str(T)).replace('%P%', str(P)).replace('%nsteps%', str(int(nsteps))) \
            .replace('%dt%', str(dt)).replace('%nstenergy%', str(nstenergy)) \
            .replace('%nstxout%', str(nstxout)).replace('%nstvout%', str(nstvout)) \
            .replace('%nstxtcout%', str(nstxtcout)).replace('%xtcgrps%', str(xtcgrps)) \
            .replace('%genvel%', genvel).replace('%seed%', str(gen_seed)).replace('%continuation%', continuation) \
            .replace('%integrator%', integrator).replace('%tcoupl%', tcoupl).replace('%tau-t%', tau_t) \
            .replace('%pcoupl%', pcoupl).replace('%tau-p%', tau_p) \
            .replace('%constraints%', constraints).replace('%TANNEAL%', str(TANNEAL)).replace('%ppm%', str(ppm)) \
            .replace('%nstlist%', str(nstlist)).replace('%dielectric%', str(dielectric)) \
            .replace('%compressibility%', compressibility).replace('%coulombtype%', coulombtype) \
            .replace('%rcoulomb%', rcoulomb).replace('%rvdw%', rvdw)

        with open(mdp_out, 'w') as f_mdp:
            f_mdp.write(contents)

    def insert_molecules(self, gro: str, ingro=None, outgro='output.gro', nmol: int = 1, box: str = '3.8 3.8 3.8', seed: int = 0):
        assert os.path.exists(gro), f'{gro} not exists.'
        cmd = f'{self.gmx_analysis} -quiet -nobackup insert-molecules -ci {gro} -nmol {nmol} -box {box} -rot xyz -seed {seed} -o {outgro}'
        if ingro is not None:
            cmd += f' -f {ingro}'
        execute(cmd)

    def solvate(self, gro: str, top: str = None, outgro='output.gro', solvent: str = 'spc216.gro'):
        assert os.path.exists(gro), f'{gro} not exists.'
        cmd = f'{self.gmx_analysis} -quiet -nobackup solvate -cp {gro} -cs {solvent} -o {outgro}'
        info = execute(cmd)[1]
        pattern = r'Generated solvent containing \d+ atoms in \d+ residues'
        matches = re.findall(pattern, str(info))
        assert len(matches) == 1
        matches = re.findall(r'\d+', matches[0])
        with open(top) as f:
            contents = f.read()
        if solvent.endswith('spc216.gro'):
            pattern = re.compile(r'\bSOL\s+\d+\b')
            contents = pattern.sub(f'SOL\t\t\t\t\t {matches[1]}', contents)
        elif solvent.endswith('box_martini3_water.gro'):
            contents += f'W\t\t\t\t\t {matches[1]}'
        with open(top, 'w') as f:
            f.write(contents)

    def grompp(self, gro: str, mdp: str, top: str, tpr: str, maxwarn: int = 0):
        for f in [gro, mdp, top]:
            assert os.path.exists(f), f'{f} not exists.'
        cmd = f'{self.gmx_analysis} -quiet -nobackup grompp -f {mdp} -c {gro} -p {top} -o {tpr} -maxwarn {maxwarn}'
        execute(cmd)

    def mdrun(self, tpr: str, ntmpi: int = None, ntomp: int = None, plumed: str = None, cpu: bool = False,
              mpi_format: str = 'mpi'):
        assert os.path.exists(tpr), f'{tpr} not exists.'
        name = tpr.split('.')[0]
        cmd = f'{self.gmx_analysis} -quiet -nobackup mdrun -v -deffnm {name}'
        if cpu:
            cmd += ' -nb cpu'
        if plumed is not None:
            cmd += f' -plumed {plumed}'
        if mpi_format == 'mpi':
            if ntmpi is not None:
                cmd = f'mpirun -np {ntmpi} ' + cmd
            if ntomp is not None:
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = str(ntomp)
        else:
            if ntmpi is not None:
                cmd += f' -ntmpi {ntmpi}'
            if ntomp is not None:
                cmd += f' -ntomp {ntomp}'
        if os.path.exists(f'{name}.cpt'):
            execute(cmd + f' -cpi {name}.cpt')
            if not os.path.exists(f'{name}.gro'):
                execute(cmd)
        else:
            execute(cmd)

    def fix_charge(self, itp: str):
        with open(itp) as f:
            contents = f.read()
        itp_parser = ITPParser(itp)
        itp_parser.parse()
        # assert len(itp_parser.molecules) == len(total_charge)
        for i, m in enumerate(itp_parser.molecules.values()):
            charges = np.array(m.charges).astype('float')
            int_charge = round(charges.sum())
            if abs(charges.sum() - int_charge) < 1e-10:
                continue
            else:
                idx = find_index_of_max_unique_abs(charges)
                charges[idx[0]] += (int_charge - charges.sum()) / len(idx)
                charge_old = m.charges[idx[0]]
                charge_new = '%.8f' % charges[idx[0]]
                assert contents.count(charge_old) == len(idx)
                contents = contents.replace(charge_old, charge_new)
        with open(itp, 'w') as f:
            f.write(contents)

    def trjconv(self, gro: str, out_gro: str, tpr: str, pbc_whole: bool = False, select='System'):
        assert os.path.exists(gro), f'{gro} not exists.'
        assert os.path.exists(tpr), f'{tpr} not exists.'
        cmd = f'{self.gmx_analysis} -quiet -nobackup trjconv -f {gro} -o {out_gro} -s {tpr}'
        if pbc_whole:
            cmd += ' -pbc whole'
        execute(cmd, input=f'{select}')

    def traj(self, gro: str, out_gro: str, tpr: str, ndx: str, ng: int, select='System',
             end_time: float = None):
        assert os.path.exists(gro), f'{gro} not exists.'
        assert os.path.exists(tpr), f'{tpr} not exists.'
        assert os.path.exists(ndx), f'{ndx} not exists.'
        cmd = f'{self.gmx_analysis} -quiet -nobackup traj -f {gro} -oxt {out_gro} -s {tpr} -n {ndx} -ng {ng} -com'
        if end_time is not None:
            cmd += f' -e {end_time}'
        execute(cmd, input=f'{select}')

    def distance(self, gro: str, tpr: str, ndx: str, select: str = '0'):
        assert os.path.exists(gro), f'{gro} not exists.'
        assert os.path.exists(tpr), f'{tpr} not exists.'
        assert os.path.exists(ndx), f'{ndx} not exists.'
        cmd = f'{self.gmx_analysis} -quiet -nobackup distance -f {gro} -s {tpr} -n {ndx} -oall bond_{select}.xvg --xvg none'
        execute(cmd, input=f'{select}')

    def angle(self, gro: str, ndx: str, select: str = '0', t: Literal['angle', 'dihedral'] = 'angle'):
        assert os.path.exists(gro), f'{gro} not exists.'
        assert os.path.exists(ndx), f'{ndx} not exists.'
        cmd = f'{self.gmx_analysis} -quiet -nobackup angle -type {t} -f {gro} -n {ndx}  -ov {t}_{select}.xvg -od temp.xvg'
        execute(cmd, input=f'{select}')

    def analyze(self, xvg: str, dist: str, bw: float = 0.001):
        assert os.path.exists(xvg), f'{xvg} not exists.'
        cmd = f'{self.gmx_analysis} -quiet -nobackup analyze -f {xvg} -dist {dist} --xvg none -bw {bw}'
        execute(cmd,)

    def energy(self, xvg: str, edr: str, select: str):
        assert os.path.exists(edr), f'{edr} not exists.'
        cmd = f'{self.gmx_analysis} -quiet -nobackup energy -f {edr} -o {xvg}'
        execute(cmd, input=f'{select}')

    def sasa(self, xvg: str, traj: str, tpr: str, select):
        assert os.path.exists(traj), f'{traj} not exists.'
        assert os.path.exists(tpr), f'{tpr} not exists.'
        cmd = f'{self.gmx_analysis} -quiet -nobackup sasa -f {traj} -s {tpr} -o {xvg}'
        execute(cmd, input=f'{select}')
