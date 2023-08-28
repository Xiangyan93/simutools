#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path
import shutil
from typing import List
from tap import Tap
import MDAnalysis as mda
from simutools.utils.utils import cd_and_mkdir
from simutools.simulator.program.packmol import Packmol
from simutools.simulator.program.gromacs import GROMACS
from simutools.simulator.program.plumed import PLUMED
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
    n_runs: int
    """Each simulation is 20,000,000 steps. The runs are simulated one by one."""
    mol_name: List[str]
    """"""
    PME: bool = False
    """Use PME for electrostatic interaction"""
    n_try: int = 10
    """"""
    centripedal: float = None
    """Apply a centripedal potential to enforce the system to form large nanoparticles. Suggest 0.1 for force 
    constant."""
    centripedal_mol: int = 2
    """Apply centripedal forces to the first N-type molecules."""

    def process_args(self) -> None:
        assert len(self.box_size) == 3
        valid_index = []
        for i, n in enumerate(self.n_mol_list):
            if n != 0:
                valid_index.append(i)
        self.gro_list = [self.gro_list[i] for i in valid_index]
        self.n_mol_list = [self.n_mol_list[i] for i in valid_index]
        self.mol_name = [self.mol_name[i] for i in valid_index]


def main(args: CommonArgs):
    cd_and_mkdir(args.save_dir)
    for i in range(args.n_try):
        try:
            if not os.path.exists('bulk.gro'):
                pdb_files = []
                for gro in args.gro_list:
                    pdb = gro.split('/')[-1].split('.')[0] + '.pdb'
                    universe = gro2pdb('../' + gro, pdb)
                    assert len(universe.residues) == 1
                    pdb_files.append(pdb)
                if args.centripedal is not None:
                    plumed = PLUMED(exe='plumed')
                    n_beads = [len(mda.Universe(pdb).atoms) for pdb in pdb_files]
                    groups = []
                    n_current = 1
                    for i, n_mol in enumerate(args.n_mol_list):
                        if i >= args.centripedal_mol:
                            break
                        for j in range(n_mol):
                            groups.append(list(range(n_current, n_current + n_beads[i])))
                            n_current += n_beads[i]
                    plumed.generate_dat_centripedal(groups=groups, k=args.centripedal)
                packmol = Packmol('packmol')
                packmol.build_uniform(pdb_files=pdb_files, n_mol_list=args.n_mol_list, output='bulk.pdb',
                                      box_size=args.box_size, tolerance=6.0)
                pdb2gro('bulk.pdb', 'bulk.gro', [x * 10 for x in args.box_size] + [90., 90., 90.])
            gmx = GROMACS(exe='gmx')
            itp_list = []
            for itp in args.itp_list:
                shutil.copy('../%s' % itp, '.')
                itp_list.append(itp.split('/')[-1])
            gmx.generate_top(f'CG.top', include_itps=[f'{TEMPLATE_DIR}/martini_v3.0.0.itp',
                                                      f'{TEMPLATE_DIR}/martini_v3.0.0_solvents_v1.itp',
                                                      f'{TEMPLATE_DIR}/martini_v3.0.0_ions_v1.itp'] + itp_list,
                             mol_name=args.mol_name, mol_number=args.n_mol_list)
            gmx.generate_mdp_from_template('t_CG_em.mdp', mdp_out=f'CG_em.mdp', dielectric=15.0)
            if args.PME:
                gmx.generate_mdp_from_template('t_npt.mdp', mdp_out=f'CG_eq.mdp', nsteps=5000000, dt=0.005,
                                               nstxtcout=10000,
                                               tcoupl='v-rescale', tau_t=1.0,
                                               pcoupl='berendsen', tau_p=1.0, compressibility='3e-4',
                                               constraints='none', coulombtype='PME',
                                               rcoulomb=1.1, rvdw=1.1, dielectric=15, nstlist=20)
                gmx.generate_mdp_from_template('t_npt.mdp', mdp_out=f'CG_run.mdp', nsteps=5000000, dt=0.005,
                                               nstxtcout=10000,
                                               restart=True,
                                               tcoupl='v-rescale', tau_t=1.0,
                                               pcoupl='parrinello-rahman', tau_p=12.0, compressibility='3e-4',
                                               constraints='none', coulombtype='PME', rcoulomb=1.1,
                                               rvdw=1.1, dielectric=15, nstlist=20)
            else:
                gmx.generate_mdp_from_template('t_npt.mdp', mdp_out=f'CG_eq.mdp', nsteps=500000, dt=0.005,
                                               nstxtcout=10000,
                                               tcoupl='v-rescale', tau_t=1.0,
                                               pcoupl='berendsen', tau_p=1.0, compressibility='3e-4',
                                               constraints='none', coulombtype='reaction-field',
                                               rcoulomb=1.1, rvdw=1.1, dielectric=15, nstlist=20)
                gmx.generate_mdp_from_template('t_npt.mdp', mdp_out=f'CG_run.mdp', nsteps=5000000, dt=0.005,
                                               nstxtcout=10000,
                                               restart=True,
                                               tcoupl='v-rescale', tau_t=1.0,
                                               pcoupl='parrinello-rahman', tau_p=12.0, compressibility='3e-4',
                                               constraints='none', coulombtype='reaction-field', rcoulomb=1.1,
                                               rvdw=1.1, dielectric=15, nstlist=20)
            if not os.path.exists('CG_em.gro'):
                gmx.grompp(gro='bulk.gro', mdp='CG_em.mdp', top=f'CG.top', tpr=f'CG_em.tpr')
                gmx.mdrun(tpr=f'CG_em.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp)

            if not os.path.exists('CG_eq.gro'):
                gmx.grompp(gro=f'CG_em.gro', mdp='CG_eq.mdp', top=f'CG.top', tpr=f'CG_eq.tpr',
                           maxwarn=2)
                gmx.mdrun(tpr=f'CG_eq.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp)
            for i in range(args.n_runs):
                if not os.path.exists(f'CG_run_{i}.gro'):
                    gro = 'CG_eq.gro' if i == 0 else f'CG_run_{i - 1}.gro'
                    gmx.grompp(gro=gro, mdp='CG_run.mdp', top=f'CG.top', tpr=f'CG_run_{i}.tpr',
                               maxwarn=1)
                    if args.centripedal is not None and i == 0:
                        gmx.mdrun(tpr=f'CG_run_{i}.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp, plumed='plumed.dat')
                    else:
                        gmx.mdrun(tpr=f'CG_run_{i}.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp)
        except:
            if os.path.exists('CG_eq.gro'):
                break
            else:
                break
                current_directory = os.getcwd()
                # Get a list of all files in the current directory
                files = os.listdir(current_directory)
                # Iterate over the files and delete them one by one
                for file_name in files:
                    file_path = os.path.join(current_directory, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        else:
            break


if __name__ == '__main__':
    main(args=CommonArgs().parse_args())
