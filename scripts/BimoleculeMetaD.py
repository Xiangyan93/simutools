#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from tap import Tap
import os
import numpy as np
from rdkit import Chem
import parmed as pmd
from copy import deepcopy
from simutools.forcefields.amber import AMBER
from simutools.simulator.gromacs.gromacs import GROMACS
from simutools.simulator.plumed import PLUMED
from simutools.coarse_grained.mapping import Mapping
from simutools.utils import cd_and_mkdir
from simutools.template import TEMPLATE_DIR
from simutools.utils import execute


class CommonArgs(Tap):
    """This script conduct OPES_METAD between two molecules. And output their binding free energy."""
    save_dir: str
    """the output directory."""
    smiles: Tuple[str, str] = None
    """SMILES of the molecules."""
    res_name: List[str] = None
    """"""
    solvent: Literal['vaccum', 'water']
    """Simulation solvent."""
    top: List[str] = None
    """The top files of the two molecules of interested. (Alternative input of SMILES)"""
    gro: List[str] = None
    """The gro files of the two molecules of interested. (Alternative input of SMILES)"""
    ntmpi: int = None
    """number of MPI threads for gmx"""
    ntomp: int = None
    """number of OpenMP threads for gmx"""

    @property
    def charge(self):
        return [sum([atom.GetFormalCharge() for atom in Chem.MolFromSmiles(self.smiles[0]).GetAtoms()]),
                sum([atom.GetFormalCharge() for atom in Chem.MolFromSmiles(self.smiles[1]).GetAtoms()])]

    @property
    def SameMol(self):
        if self.smiles[0] == self.smiles[1]:
            return True
        else:
            return False

    def process_args(self) -> None:
        if self.smiles is not None:
            assert self.top is None
            assert self.top is None
            assert len(self.smiles) in [1, 2]
        else:
            assert len(self.gro) == 2
            assert len(self.top) == 2


def main(args: CommonArgs):
    cd_and_mkdir(args.save_dir)
    gmx = GROMACS(gmx_exe_mdrun='gmx')
    amber = AMBER()
    plumed = PLUMED(plumed_exe='plumed')
    # skip energy minimization if it is finished
    if not os.path.exists('em.gro'):
        if args.smiles is None:
            ff = pmd.load_file(f'../{args.top[0]}', xyz=f'../{args.gro[0]}')
            ff.strip(f':SOL')
            ff += pmd.load_file(f'../{args.top[1]}', xyz=f'../{args.gro[1]}')
            ff.save(f'bimolecule.top', overwrite=True)
            ff.strip(f':SOL')
            ff.save(f'bimolecule.gro', overwrite=True)
        else:
            amber.build(args.smiles[0], 'mol1', charge=args.charge[0], gromacs=True, tip3p=False,
                        resName=args.res_name[0])
            gmx.fix_charge(f'mol1.top')
            ff = pmd.load_file(f'mol1.top', xyz=f'mol1.gro')
            if not args.SameMol:
                amber.build(args.smiles[1], 'mol2', charge=args.charge[1], gromacs=True, tip3p=False,
                            resName=args.res_name[1])
                gmx.fix_charge(f'mol2.top')
                ff += pmd.load_file(f'mol2.top', xyz=f'mol2.gro')
            else:
                new_ff = deepcopy(ff)
                for atom in new_ff.atoms:
                    atom.xx += 25.0
                    atom.xy += 25.0
                    atom.xz += 25.0
                ff += new_ff
            ff.save(f'bimolecule.gro', overwrite=True)
            ff += pmd.load_file(f'{TEMPLATE_DIR}/tip3p.top', xyz=f'{TEMPLATE_DIR}/tip3p.gro')
            ff.save(f'bimolecule.top', overwrite=True)

        gmx.insert_molecules(f'bimolecule.gro', outgro='output.gro', box='5.0 5.0 5.0')
        if args.solvent == 'water':
            gmx.solvate('output.gro', top=f'bimolecule.top', outgro='initial.gro')
        else:
            shutil.copyfile('output.gro', 'initial.gro')
            ff.save(f'bimolecule.top', overwrite=True)

        if args.charge[0] != 0:
            assert ff.residues[1].name in ['NA', 'CL']
            n = abs(args.charge[0])
            plumed.generate_dat_from_template('bimolecule.dat', output='plumed.dat',
                                              group1=f'1-{len(ff.residues[0].atoms)}',
                                              group2=f'{len(ff.residues[0].atoms) + n + 1}-'
                                                     f'{len(ff.residues[0].atoms) + n + len(ff.residues[1 + n].atoms)}',
                                              biasfactor=100)
        else:
            plumed.generate_dat_from_template('bimolecule.dat', output='plumed.dat',
                                              group1=f'1-{len(ff.residues[0].atoms)}',
                                              group2=f'{len(ff.residues[0].atoms) + 1}-'
                                                     f'{len(ff.residues[0].atoms) + len(ff.residues[1].atoms)}',
                                              biasfactor=100)

        gmx.generate_mdp_from_template('t_em.mdp', mdp_out=f'em.mdp', dielectric=1.0)
        gmx.grompp(gro='initial.gro', mdp='em.mdp', top=f'bimolecule.top', tpr='em.tpr', maxwarn=1)
        gmx.mdrun(tpr='em.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp)

    if args.solvent == 'water':
        gmx.generate_mdp_from_template('t_nvt.mdp', mdp_out=f'eq_nvt.mdp', nsteps=200000, dt=0.002)
        gmx.grompp(gro='em.gro', mdp='eq_nvt.mdp', top=f'bimolecule.top', tpr='eq_nvt.tpr')
        gmx.mdrun(tpr='eq_nvt.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp)

        gmx.generate_mdp_from_template('t_npt.mdp', mdp_out=f'eq_npt.mdp', nsteps=200000, dt=0.002, pcoupl='berendsen')
        gmx.grompp(gro='eq_nvt.gro', mdp='eq_npt.mdp', top=f'bimolecule.top', tpr='eq_npt.tpr', maxwarn=1)
        gmx.mdrun(tpr='eq_npt.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp)

        gmx.generate_mdp_from_template('t_npt.mdp', mdp_out=f'run.mdp', nsteps=10000000, dt=0.002, nstxtcout=100,
                                       restart=True)
        gmx.grompp(gro='eq_nvt.gro', mdp='run.mdp', top=f'bimolecule.top', tpr='run.tpr')
        gmx.mdrun(tpr='run.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp, plumed='plumed.dat')
    else:  # vacuum
        gmx.generate_mdp_from_template('t_nvt.mdp', mdp_out=f'eq.mdp', nsteps=200000, dt=0.002)
        gmx.grompp(gro='em.gro', mdp='eq.mdp', top=f'bimolecule.top', tpr='eq.tpr')
        gmx.mdrun(tpr='eq.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp)

        gmx.generate_mdp_from_template('t_nvt.mdp', mdp_out=f'run.mdp', nsteps=10000000, dt=0.002, nstxtcout=100,
                                       restart=True)
        gmx.grompp(gro='eq.gro', mdp='run.mdp', top=f'bimolecule.top', tpr='run.tpr')
        gmx.mdrun(tpr='run.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp, plumed='plumed.dat')
    # gmx.trjconv(gro='run.xtc', out_gro='AA-traj.whole.xtc', tpr='run.tpr', pbc_whole=True)


if __name__ == '__main__':
    main(args=CommonArgs().parse_args())
