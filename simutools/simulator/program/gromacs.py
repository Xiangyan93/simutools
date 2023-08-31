#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
import re
import numpy as np
import MDAnalysis as mda
from MDAnalysis.topology.ITPParser import ITPParser
import parmed as pmd
from simutools.utils.utils import execute, cd_and_mkdir, find_index_of_max_unique_abs
from simutools.template import TEMPLATE_DIR
from .base import BaseMDProgram


class GROMACS(BaseMDProgram):
    def __init__(self, exe: str):
        self.gmx_analysis = exe
        self.gmx_mdrun = exe + ' mdrun'
        self.check_version()
        self.tmp_dir = f'{TEMPLATE_DIR}/gromacs'

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

    """
    def set_default_parameters(
            self, timestep: float = 0.001,
            tcoupl: Literal['sd', 'nose-hoover', 'v-rescale', 'berendsen', 'no'] = 'sd', tau_t: float = None,
            pcoupl: Literal['berendsen', 'parrinello-rahman'] = 'parrinello-rahman', tau_p: float = None,
            compressibility: float = 4e-5,
            coulombtype: Literal['Cut-off', 'Ewald', 'PME', 'Reaction-Field'] = 'PME', rcoulomb: float = 1.2,
            dielectric: float = 1.0,
            rvdw: float = 1.2
    ):
        self.timestep = timestep  # ps
        # thermostat
        self.tcoupl = tcoupl
        self.tau_t = tau_t
        # barostat
        self.pcoupl = pcoupl
        self.tau_t = tau_p
        self.compressibility = compressibility
        # electrostatic interation
        self.coulombtype = coulombtype
        self.rcoulomb = rcoulomb  # nm
        self.dielectric = dielectric
        # Van Der Waals interaction
        self.rvdw = rvdw  # nm
    """

    @staticmethod
    def generate_top(file: str, include_itps: List[str], mol_name: List[str], mol_number: List[int]):
        """ Generate a top file for GROMACS simulation.

        Parameters
        ----------
        file: the name of the top file.
        include_itps: the itp files need to be included.
        mol_name: the name of molecules.
        mol_number: the number of molecules.

        Returns
        -------
        the top file.
        """
        assert len(mol_name) == len(mol_number), 'the length of mol_name and mol_number must be the same.'
        contents = ''
        for include_itp in include_itps:
            contents += f'#include "{include_itp}"\n'
        contents += '\n[ system ]\n'
        contents += ', '.join([f'{name} {mol_number[i]}' for i, name in enumerate(mol_name)])
        contents += '\n\n[ molecules ]\n'
        for i, name in enumerate(mol_name):
            contents += f'{name}         {mol_number[i]}\n'
        with open(file, 'w') as f:
            f.write(contents)
        return contents

    def generate_mdp_from_template(
            self, template: str, mdp_out: str = 'grompp.mdp', T: float = 298., P: float = 1.,
            nsteps: int = 10000, dt: float = 0.001, T_annealing: float = 800.,
            nstenergy: int = 100, nstxout: int = 0, nstvout: int = 0, nstxtcout: int = 10000,
            xtcgrps: str = 'System', restart: bool = False,
            pcoupl: str = 'parrinello-rahman', tau_p: float = None, compressibility='4e-5',
            tcoupl: str = 'langevin', tau_t: float = None, gen_seed: int = -1,
            coulombtype: str = 'PME', rcoulomb: float = 1.2, dielectric: float = 1.0,
            rvdw: float = 1.2,
            constraints: str = 'h-bonds', ppm=0,
            nstlist=None):
        """generate mdp file for GROMACS simulation."""
        template = f'{self.tmp_dir}/{template}'
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
            .replace('%constraints%', constraints).replace('%TANNEAL%', str(T_annealing)).replace('%ppm%', str(ppm)) \
            .replace('%nstlist%', str(nstlist)).replace('%dielectric%', str(dielectric)) \
            .replace('%compressibility%', compressibility).replace('%coulombtype%', coulombtype) \
            .replace('%rcoulomb%', str(rcoulomb)).replace('%rvdw%', str(rvdw))

        with open(mdp_out, 'w') as f_mdp:
            f_mdp.write(contents)

    def fix_charge(self, itp: str):
        """ The total charge generated by Antechamber may not equal 0. Use this function to fix it.

        Parameters
        ----------
        itp: GROMACS itp or top file

        Returns
        -------
        A new itp file with correct charge.
        """
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

    @staticmethod
    def modify_top_mol_numbers(top: str, outtop: str, mol_name: str, n_mol: int):
        """ Modify the molecule number in a top file.

        Parameters
        ----------
        top: GROMACS top file.
        outtop: output GROMACS top file.
        mol_name: the name of the molecules to be modified.
        n_mol: the number of the molecules.

        Returns
        -------
        The top file will be modified.
        """
        with open(top) as f:
            contents = f.read()
            pattern = rf'\n{mol_name}\s+\d+\n'
            search_info = re.findall(pattern, contents)
            if len(search_info) == 0:
                contents += f'{mol_name}\t\t\t\t\t {n_mol}'
            elif len(search_info) in [1, 2]:
                # pattern = re.compile(pattern)
                # contents = pattern.sub(f'{mol_name}\t\t\t\t\t {n_mol}', contents)
                # replace the last occurance of pattern.
                contents = re.sub(rf'({pattern})(?!.*{pattern})', f'\n{mol_name}\t\t\t\t\t {n_mol}\n',
                                  contents, count=1, flags=re.DOTALL)
            else:
                raise ValueError()
        with open(outtop, 'w') as f:
            f.write(contents)

    @staticmethod
    def merge_top(dirs: List[str], outtop: str = 'topol.top'):
        ff = pmd.load_file(filename=f'{dirs[0]}.top', xyz=f'{dirs[0]}.gro')
        for d in dirs[1:]:
            ff += pmd.load_file(filename=f'{d}.top', xyz=f'{d}.gro')
        ff.save(outtop, overwrite=True)

    @staticmethod
    def generate_top_for_hvap(top, top_out):
        with open(top) as f:
            lines = f.read().splitlines()
        lines = [l for l in lines if not (l.startswith(';') or l == '')]

        f_out = open(top_out, 'w')

        line_number_molecule = []
        line_number_atom = []
        line_number_system = None
        for n, line in enumerate(lines):
            if line.find('[') != -1 and line.find('moleculetype') != -1:
                line_number_molecule.append(n)
            if line.find('[') != -1 and line.find('atoms') != -1:
                line_number_atom.append(n)
            if line.find('[') != -1 and line.find('system') != -1:
                line_number_system = n

        n_molecules = len(line_number_molecule)

        for n in range(line_number_molecule[0]):
            f_out.write(lines[n] + '\n')

        for i in range(n_molecules):
            for n in range(line_number_molecule[i], line_number_atom[i]):
                f_out.write(lines[n] + '\n')
            line_number_next_section = line_number_molecule[i + 1] if i < n_molecules - 1 else line_number_system
            if line_number_next_section is None:
                line_number_next_section = len(lines)
            n_atoms = 0
            f_out.write('[ atoms ]\n')
            for n in range(line_number_atom[i] + 1, line_number_next_section):
                line = lines[n]
                if line.find('[') != -1 or line.startswith('#'):
                    f_out.write('[ bonds ]\n[ exclusions ]\n')
                    for i in range(1, n_atoms):
                        exclusions = range(i, n_atoms + 1)
                        f_out.write(' '.join(list(map(str, exclusions))) + '\n')
                    break
                f_out.write(line + '\n')
                n_atoms += 1

        if line_number_system is not None:
            for n in range(line_number_system, len(lines)):
                f_out.write(lines[n] + '\n')

    def convert_pdb(self, pdb: str, tag_out: str, box_size: List[float],
                    algorithm: Literal['MDAnalysis', 'editconf'] = 'MDAnalysis') -> str:
        """ convert pdb into gro file.

        Parameters
        ----------
        pdb: the input pdb file.
        tag_out: the output file is {tag_out}.gro
        box_size: the X, Y, and Z length of simulation box. unit: nm.
        algorithm

        Returns
        -------
        The PDB file is converted into GRO file used in GROMACS.
        """
        gro = tag_out + '.gro'
        assert len(box_size) == 3
        if algorithm == 'editconf':
            cmd = f'{self.gmx_analysis} -quiet -nobackup editconf -f {pdb} -o {gro} -box {box_size}'
            execute(cmd)
        elif algorithm == 'MDAnalysis':
            box_size = [length * 10 for length in box_size]
            box_size += [90., 90., 90.]
            # Create a Universe object from the PDB file
            universe = mda.Universe(pdb)
            universe.dimensions = box_size
            # Write the Universe object to a GRO file
            universe.atoms.write(gro)
        else:
            raise ValueError
        return gro

    def energy_minimization(self, tag_in: str, tag_out: str, exe: bool = False, top: str = 'topol.top'):
        mdp = f'{tag_out}.mdp'
        gro = f'{tag_in}.gro'
        tpr = f'{tag_out}.tpr'
        self.generate_mdp_from_template(template='t_em.mdp', mdp_out=mdp)
        commands = [self.grompp(mdp=mdp, gro=gro, top=top, tpr=tpr, exe=exe),
                    self.mdrun(tpr=tpr, exe=exe)]
        return commands

    def annealing(self, tag_in: str, tag_out: str, T: float, T_annealing: float, exe: bool = False,
                  top: str = 'topol.top'):
        assert T < T_annealing
        gro = f'{tag_in}.gro'
        mdp = f'{tag_out}.mdp'
        tpr = f'{tag_out}.tpr'
        self.generate_mdp_from_template(template='t_nvt_anneal.mdp', mdp_out=mdp, T=T, T_annealing=T_annealing,
                                        nsteps=int(1E5), nstxtcout=0)
        commands = [self.grompp(mdp=mdp, gro=gro, top=top, tpr=tpr, exe=exe),
                    self.mdrun(tpr=tpr, exe=exe)]
        return commands

    def get_fluct_props(self, edr, begin=0, end=None) -> (float, float):
        '''
        Get thermal expansion and compressibility using fluctuation of enthalpy, volume
        Only works for NPT simulation
        :param edr:
        :param begin:
        :param end:
        :return:
        '''
        sp_out = self.energy(edr, properties=['temp', 'vol', 'enthalpy'], begin=begin, end=end, fluct_props=True)

        expansion = None
        compressibility = None
        for line in sp_out.decode().splitlines():
            if line.startswith('Coefficient of Thermal Expansion Alpha_P'):
                expansion = float(line.split()[-2])
            elif line.startswith('Isothermal Compressibility Kappa'):
                compressibility = float(line.split()[-2]) * 1e5
        return expansion, compressibility

    """GROMACS API are list below."""
    def analyze(self, xvg: str, dist: str, bw: float = 0.001):
        assert os.path.exists(xvg), f'{xvg} not exists.'
        cmd = f'{self.gmx_analysis} -quiet -nobackup analyze -f {xvg} -dist {dist} --xvg none -bw {bw}'
        execute(cmd, )

    def angle(self, gro: str, ndx: str, select: str = '0', t: Literal['angle', 'dihedral'] = 'angle'):
        assert os.path.exists(gro), f'{gro} not exists.'
        assert os.path.exists(ndx), f'{ndx} not exists.'
        cmd = f'{self.gmx_analysis} -quiet -nobackup angle -type {t} -f {gro} -n {ndx}  -ov {t}_{select}.xvg -od temp.xvg'
        execute(cmd, input=f'{select}')

    def convert_tpr(self, tpr: str, outtpr: str = None, extend: float = None, until: float = None, nsteps: int = None):
        """ Python API for "gmx convert-tpr".

        Parameters
        ----------
        tpr: GROMACS tpr file.
        outtpr: the output tpr file.
        extend: Extend runtime by this amount (ps)
        until: Extend runtime until this ending time (ps)
        nsteps: Change the number of steps remaining to be made.

        Returns
        -------
        A tpr file.
        """
        if outtpr is None:
            outtpr = tpr
        cmd = '%s -quiet convert-tpr -s %s -o %s' % (self.gmx_analysis, tpr, outtpr)
        if extend is not None:
            cmd += f' -extend {extend}'
        if until is not None:
            cmd += f' -until {until}'
        if nsteps is not None:
            cmd += f' -nsteps {nsteps}'
        execute(cmd)

    def msd(self, gro: str, tpr: str, xvg: str = 'msd.xvg', mol: bool = False, begin: float = 0., end: float = None,
                  beginfit: float = -1, endfit: float = -1, select: str = 'System', get_diffusion: bool = False):
        cmd = (f'{self.gmx_analysis} -quiet -nobackup msd -f {gro} -s {tpr} -o {xvg} '
               f'-b {begin} -beginfit {beginfit} -endfit {endfit}')
        if end is not None:
            cmd += f' -e {end}'
        if mol:
            # calculate the MSD of COM of molecules
            cmd += ' -mol'
        out, err = execute(cmd, input=f'{select}')
        if get_diffusion:
            for line in out.decode().splitlines():
                if line.startswith('D['):
                    words = line.strip().split(']')[-1].strip().split()
                    diffusion = float(words[0])
                    stderr = float(words[2][:-1])
                    unit = float(words[3])
                    diffusion *= unit  # cm^2/s
                    stderr *= unit  # cm^2/s
                    return diffusion, stderr

    def distance(self, gro: str, tpr: str, ndx: str, select: str = '0'):
        assert os.path.exists(gro), f'{gro} not exists.'
        assert os.path.exists(tpr), f'{tpr} not exists.'
        assert os.path.exists(ndx), f'{ndx} not exists.'
        cmd = f'{self.gmx_analysis} -quiet -nobackup distance -f {gro} -s {tpr} -n {ndx} -oall bond_{select}.xvg --xvg none'
        execute(cmd, input=f'{select}')

    def energy(self, xvg: str, edr: str, select: str):
        assert os.path.exists(edr), f'{edr} not exists.'
        cmd = f'{self.gmx_analysis} -quiet -nobackup energy -f {edr} -o {xvg}'
        execute(cmd, input=f'{select}')

    def grompp(self, gro: str, mdp: str, top: str, tpr: str, maxwarn: int = 0, exe: bool = True) -> str:
        """Python API for "gmx grompp"."""
        cmd = f'{self.gmx_analysis} -quiet -nobackup grompp -f {mdp} -c {gro} -p {top} -o {tpr} -maxwarn {maxwarn}'
        if exe:
            for f in [gro, mdp, top]:
                assert os.path.exists(f), f'{f} not exists.'
            execute(cmd)
        return cmd

    def insert_molecules(self, gro: str, ingro: str = None, outgro='output.gro', nmol: int = 1,
                         box_size: List[float] = None, seed: int = 0):
        """ Python API for "gmx insert-molecules".

        Parameters
        ----------
        gro: the gro file to be inserted.
        ingro: initial gro file. if None, create an empty box.
        outgro: output gro file.
        nmol: number of inserted molecules.
        box_size: the X, Y, and Z length of simulation box. unit: nm.
        seed: random seed.

        Returns
        -------

        """
        box = ' '.join([str(x) for x in box_size])
        assert os.path.exists(gro), f'{gro} not exists.'
        cmd = f'{self.gmx_analysis} -quiet -nobackup insert-molecules -ci {gro} -nmol {nmol} -box {box} -rot xyz ' \
              f'-seed {seed} -o {outgro}'
        if ingro is not None:
            cmd += f' -f {ingro}'
        execute(cmd)

    def mdrun(self, tpr: str, ntmpi: int = None, ntomp: int = None, plumed: str = None, cpu: bool = False,
              rerun: str = None, extend: bool = False, mpi_format: str = None, exe: bool = True) -> str:
        """Python API for "gmx mdrun"."""
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
        if extend:
            assert os.path.exists(f'{name}.cpt')
            cmd += f' -cpi {name}.cpt'
        if rerun:
            cmd += f' -rerun {rerun}'
        if exe:
            assert os.path.exists(tpr), f'{tpr} not exists.'
            execute(cmd)
        return cmd

    def sasa(self, xvg: str, traj: str, tpr: str, select):
        assert os.path.exists(traj), f'{traj} not exists.'
        assert os.path.exists(tpr), f'{tpr} not exists.'
        cmd = f'{self.gmx_analysis} -quiet -nobackup sasa -f {traj} -s {tpr} -o {xvg}'
        execute(cmd, input=f'{select}')

    def solvate(self, gro: str, top: str = None, outgro: str = 'output.gro', outtop: str = 'output.top',
                solvent: str = 'spc216.gro'):
        """ Python API for "gmx solvate".

        Parameters
        ----------
        gro: the gro file to be solvated.
        top: the top file to be solvated.
        outgro: the output gro file.
        outtop: the output top file.
        solvent: the gro file of solvent.

        Returns
        -------
        A new gro and top files are generated.
        """
        assert os.path.exists(gro), f'{gro} not exists.'
        cmd = f'{self.gmx_analysis} -quiet -nobackup solvate -cp {gro} -cs {solvent} -o {outgro}'
        info = execute(cmd)[1]
        if top is not None:
            # find the number of solvated molecules.
            pattern = r'Generated solvent containing \d+ atoms in \d+ residues'
            matches = re.findall(pattern, str(info))
            assert len(matches) == 1
            matches = re.findall(r'\d+', matches[0])
            n_solvated = matches[1]
            # find the residue name
            pattern = r'Found 1 molecule type:\\n\s+[0-9a-zA-Z]{1,3} \('
            matches = re.findall(pattern, str(info))
            assert len(matches) == 1
            split_s = matches[0].split()
            assert split_s[1] == '1'
            resname = split_s[4]
            self.modify_top_mol_numbers(top=top, outtop=outtop, mol_name=resname, n_mol=n_solvated)

    def traj(self, gro: str, out_gro: str, tpr: str, ndx: str, ng: int, select='System',
             end_time: float = None):
        assert os.path.exists(gro), f'{gro} not exists.'
        assert os.path.exists(tpr), f'{tpr} not exists.'
        assert os.path.exists(ndx), f'{ndx} not exists.'
        cmd = f'{self.gmx_analysis} -quiet -nobackup traj -f {gro} -oxt {out_gro} -s {tpr} -n {ndx} -ng {ng} -com'
        if end_time is not None:
            cmd += f' -e {end_time}'
        execute(cmd, input=f'{select}')

    def trjconv(self, gro: str, out_gro: str, tpr: str, pbc_whole: bool = False, select='System'):
        assert os.path.exists(gro), f'{gro} not exists.'
        assert os.path.exists(tpr), f'{tpr} not exists.'
        cmd = f'{self.gmx_analysis} -quiet -nobackup trjconv -f {gro} -o {out_gro} -s {tpr}'
        if pbc_whole:
            cmd += ' -pbc whole'
        execute(cmd, input=f'{select}')
