#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from tap import Tap
import MDAnalysis as mda
from simutools.utils import cd_and_mkdir
from simutools.builder.packmol import Packmol
from simutools.simulator.gromacs.gromacs import GROMACS
from simutools.template import TEMPLATE_DIR


def gro2pdb(gro_file, pdb_file):
    # Load the GRO file
    universe = mda.Universe(gro_file)

    # Save the PDB file
    universe.atoms.write(pdb_file)
    return universe


def pdb2gro(pdb_filename, gro_filename, box_dimensions):
    # Create a Universe object from the PDB file
    universe = mda.Universe(pdb_filename)
    universe.dimensions = box_dimensions
    # Write the Universe object to a GRO file
    universe.atoms.write(gro_filename)
    return universe


class CommonArgs(Tap):
    save_dir: str
    """the output directory."""
    gro_list: List[str]
    """gro files of the molecules to be simulated."""
    n_mol_list: List[int]
    """number of molecules."""
    box_size: List[float]
    """X, Y, Z box size. unit: nm."""
    itp_list: List[str]
    """"""
    ntmpi: int = None
    """number of MPI threads for gmx"""
    ntomp: int = None
    """number of OpenMP threads for gmx"""
    n_steps: int = 5000000

    def process_args(self) -> None:
        assert len(self.box_size) == 3
        valid_index= []
        for i, n in enumerate(self.n_mol_list):
            if n != 0:
                valid_index.append(i)
        self.gro_list = [self.gro_list[i] for i in valid_index]
        self.n_mol_list = [self.n_mol_list[i] for i in valid_index]


def main(args: CommonArgs):
    cd_and_mkdir(args.save_dir)
    pdb_files = []
    res_name = []
    for gro in args.gro_list:
        pdb = gro.split('/')[-1].split('.')[0] + '.pdb'
        universe = gro2pdb('../' + gro, pdb)
        assert len(universe.residues) == 1
        for residue in universe.residues:
            res_name.append(residue.resname)
        pdb_files.append(pdb)
    packmol = Packmol('packmol')
    packmol.build_box(pdb_files=pdb_files, n_mol_list=args.n_mol_list, output='bulk.pdb', box_size=args.box_size)
    pdb2gro('bulk.pdb', 'bulk.gro', [x * 10 for x in args.box_size] + [90., 90., 90.])
    gmx = GROMACS(gmx_exe_mdrun='gmx')
    itp_list = []
    for itp in args.itp_list:
        shutil.copy('../%s' % itp, '.')
        itp_list.append(itp.split('/')[-1])
    gmx.generate_top(f'CG.top', include_itps=[f'{TEMPLATE_DIR}/martini_v3.0.0.itp',
                                              f'{TEMPLATE_DIR}/martini_v3.0.0_solvents_v1.itp'] + itp_list,
                     mol_name=res_name, mol_number=args.n_mol_list)
    gmx.generate_mdp_from_template('t_CG_em.mdp', mdp_out=f'CG_em.mdp', dielectric=1.0)
    gmx.generate_mdp_from_template('t_npt.mdp', mdp_out=f'CG_eq.mdp', nsteps=200000, dt=0.002,
                                   tcoupl='v-rescale', tau_t='1.0',
                                   pcoupl='berendsen', tau_p='12.0', compressibility='3e-4',
                                   constraints='none', coulombtype='cutoff',
                                   rcoulomb='1.1', rvdw='1.1', dielectric=15, nstlist=20)
    gmx.generate_mdp_from_template('t_npt.mdp', mdp_out=f'CG_run.mdp', nsteps=args.n_steps, dt=0.005, nstxtcout=10000,
                                   restart=True,
                                   tcoupl='v-rescale', tau_t='1.0',
                                   pcoupl='parrinello-rahman', tau_p='12.0', compressibility='3e-4',
                                   constraints='none', coulombtype='cutoff', rcoulomb='1.1',
                                   rvdw='1.1', dielectric=15, nstlist=20)
    gmx.grompp(gro='bulk.gro', mdp='CG_em.mdp', top=f'CG.top', tpr=f'CG_em.tpr')
    gmx.mdrun(tpr=f'CG_em.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp)
    gmx.grompp(gro=f'CG_em.gro', mdp='CG_eq.mdp', top=f'CG.top', tpr=f'CG_eq.tpr',
               maxwarn=2)
    gmx.mdrun(tpr=f'CG_eq.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp)
    gmx.grompp(gro=f'CG_eq.gro', mdp='CG_run.mdp', top=f'CG.top', tpr=f'CG_run.tpr',
               maxwarn=1)
    gmx.mdrun(tpr=f'CG_run.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp)


if __name__ == '__main__':
    main(args=CommonArgs().parse_args())
