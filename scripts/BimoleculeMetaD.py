#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from tap import Tap
import os
import numpy as np
from rdkit import Chem
import parmed as pmd
from simutools.forcefields.amber import AMBER
from simutools.simulator.gromacs.gromacs import GROMACS
from simutools.simulator.plumed import PLUMED
from simutools.coarse_grained.mapping import Mapping
from simutools.utils import cd_and_mkdir
from simutools.template import TEMPLATE_DIR
from simutools.utils import execute


class CommonArgs(Tap):
    save_dir: str
    """the output directory."""
    smiles: str = None
    """SMILES of the molecule."""
    dihedral_no_force: List[int] = []
    """dihedral index without potential"""
    dihedral_with_force: List[int] = []
    """dihedral index with potential"""
    ntmpi: int = None
    """number of MPI threads for gmx"""
    ntomp: int = None
    """number of OpenMP threads for gmx"""
    n_iter: int = 1
    """"""
    solvent: Literal['vaccum', 'water']
    """"""
    top: List[str]
    """"""
    gro: List[str]
    """"""

    @property
    def charge(self):
        mol = Chem.MolFromSmiles(self.smiles)
        total_formal_charge = sum([atom.GetFormalCharge() for atom in mol.GetAtoms()])
        return total_formal_charge

    def process_args(self) -> None:
        assert len(self.gro) == 2
        assert len(self.top) == 2


def main(args: CommonArgs):
    cd_and_mkdir(args.save_dir)
    gmx = GROMACS(gmx_exe_mdrun='gmx')
    plumed = PLUMED(plumed_exe='plumed')

    if not os.path.exists('em.gro'):
        ff = pmd.load_file(f'../{args.top[0]}', xyz=f'../{args.gro[0]}')
        ff.strip(f':SOL')
        ff += pmd.load_file(f'../{args.top[1]}', xyz=f'../{args.gro[1]}')
        ff.save(f'bimolecule.top', overwrite=True)
        ff.strip(f':SOL')
        ff.save(f'bimolecule.gro', overwrite=True)
        gmx.insert_molecules(f'bimolecule.gro', outgro='output.gro', box='5.0 5.0 5.0')

        if args.solvent == 'water':
            gmx.solvate('output.gro', top=f'bimolecule.top', outgro='initial.gro')
        else:
            shutil.copyfile('output.gro', 'initial.gro')
            ff.save(f'bimolecule.top', overwrite=True)

        plumed.generate_dat_from_template('bimolecule.dat', output='plumed.dat', group1=f'1-{len(ff.residues[0].atoms)}',
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
