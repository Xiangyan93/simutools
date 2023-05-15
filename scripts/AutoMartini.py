#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from tap import Tap
import os
from rdkit import Chem
from simutools.forcefields.amber import AMBER
from simutools.simulator.gromacs.gromacs import GROMACS
from simutools.coarse_grained.mapping import Mapping
from simutools.utils import cd_and_mkdir
from simutools.template import TEMPLATE_DIR
from simutools.utils import execute


class CommonArgs(Tap):
    save_dir: str
    """the output directory."""
    name: str = None
    """The name of the molecule. default = save_dir"""
    res_name: str = None
    """residual name"""
    smiles: str = None
    """SMILES of the molecule."""
    action: Literal['all-atom', 'cg-mapping', 'cg-sim', 'test', 'bond-opt', 'swarm-cg', 'stability']
    """action to be conducted."""
    dihedral_no_force: List[int] = []
    """dihedral index without potential"""
    dihedral_with_force: List[int] = []
    """dihedral index with potential"""
    ntmpi: int = None
    """number of MPI threads for gmx"""
    ntomp: int = None
    """number of OpenMP threads for gmx"""
    n_iter: int = 1

    @property
    def charge(self):
        mol = Chem.MolFromSmiles(self.smiles)
        total_formal_charge = sum([atom.GetFormalCharge() for atom in mol.GetAtoms()])
        return total_formal_charge

    def process_args(self) -> None:
        if self.name is None:
            self.name = self.save_dir


def main(args: CommonArgs):
    cd_and_mkdir(args.save_dir)
    gmx = GROMACS(gmx_exe_mdrun='gmx')
    if args.action == 'all-atom':
        assert args.smiles is not None
        cd_and_mkdir('1.all-atom')
        amber = AMBER()
        amber.build(args.smiles, args.name, charge=args.charge, gromacs=True, tip3p=True, resName=args.res_name)

        gmx.fix_charge(f'{args.name}.top')
        gmx.insert_molecules(f'{args.name}.gro', outgro='output.gro')
        gmx.solvate('output.gro', top=f'{args.name}.top', outgro='initial.gro')

        gmx.generate_mdp_from_template('t_em.mdp', mdp_out=f'em.mdp', dielectric=1.0)
        gmx.grompp(gro='initial.gro', mdp='em.mdp', top=f'{args.name}.top', tpr='em.tpr')
        gmx.mdrun(tpr='em.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp)

        gmx.generate_mdp_from_template('t_nvt.mdp', mdp_out=f'eq.mdp', nsteps=200000, dt=0.002)
        gmx.grompp(gro='em.gro', mdp='eq.mdp', top=f'{args.name}.top', tpr='eq.tpr')
        gmx.mdrun(tpr='eq.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp)

        gmx.generate_mdp_from_template('t_npt.mdp', mdp_out=f'run.mdp', nsteps=10000000, dt=0.002, nstxtcout=100,
                                       restart=True)
        gmx.grompp(gro='eq.gro', mdp='run.mdp', top=f'{args.name}.top', tpr='run.tpr')
        gmx.mdrun(tpr='run.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp)
        gmx.trjconv(gro='run.xtc', out_gro='AA-traj.whole.xtc', tpr='run.tpr', pbc_whole=True)
    elif args.action == 'cg-mapping':
        cd_and_mkdir('2.cg-mapping')
        # use MDAnalysis
        if os.path.exists('temp.pkl'):
            mapping = Mapping.load(filename='temp.pkl')
        else:
            mol2 = f'../1.all-atom/{args.name}_ob.mol2'
            mapping = Mapping(mol2=mol2)
            mapping.mapping()
            mapping.generate_mapping_img()
            mapping.generate_ndx_mapping(f'{args.name}.ndx')
            mapping.load_aa_traj('../1.all-atom/AA-traj.whole.xtc', '../1.all-atom/run.tpr')
            mapping.save(filename='temp.pkl')
        mapping.get_aa_distribution(dihedral_no_force=args.dihedral_no_force,
                                    dihedral_with_force=args.dihedral_with_force)
        mapping.write_distribution(file='aa-distribution.svg', CG=False, fit=True)
        # mapping.generate_itp(f'CG_{args.name}.itp', resName=args.res_name)
        # mapping.generate_gro(file=f'CG_{args.name}.gro', resName=args.res_name, box_length=3.8)
        mapping.save()
    elif args.action == 'bond-opt':
        cd_and_mkdir('3.bond-opt')
        mapping = Mapping.load(path='../2.cg-mapping')
        mapping.generate_gro(file=f'CG_{args.name}.gro', resName=args.res_name, box_length=3.8)
        gmx.generate_top(f'CG_{args.name}.top', include_itps=[f'{TEMPLATE_DIR}/martini_v3.0.0.itp',
                                                              f'{TEMPLATE_DIR}/martini_v3.0.0_solvents_v1.itp',
                                                              f'CG_{args.name}.itp'],
                         mol_name=[args.res_name], mol_number=[1])
        gmx.generate_mdp_from_template('t_CG_em.mdp', mdp_out=f'CG_em.mdp', dielectric=1.0)
        gmx.generate_mdp_from_template('t_npt.mdp', mdp_out=f'CG_eq.mdp', nsteps=100000, dt=0.005,
                                       tcoupl='v-rescale', tau_t='0.1',
                                       pcoupl='berendsen', tau_p='12.0', compressibility='3e-4',
                                       constraints='none', coulombtype='cutoff',
                                       rcoulomb='1.1', rvdw='1.1', dielectric=15, nstlist=20)
        gmx.generate_mdp_from_template('t_npt.mdp', mdp_out=f'CG_run.mdp', nsteps=1000000, dt=0.01, nstxtcout=10,
                                       restart=True,
                                       tcoupl='v-rescale', tau_t='0.1',
                                       pcoupl='parrinello-rahman', tau_p='12.0', compressibility='3e-4',
                                       constraints='none', coulombtype='reaction-field', rcoulomb='1.1',
                                       rvdw='1.1', dielectric=15, nstlist=20)
        gmx.solvate(f'CG_{args.name}.gro', top=f'CG_{args.name}.top', outgro='CG_initial.gro',
                    solvent=f'{TEMPLATE_DIR}/box_martini3_water.gro')
        for i in range(args.n_iter):
            print(f'Iteration: {i}\n')
            cd_and_mkdir(f'iteration_{i}')
            mapping.generate_itp(f'CG_{args.name}_{i}.itp', resName=args.res_name)
            shutil.copy(f'CG_{args.name}_{i}.itp', f'CG_{args.name}.itp')
            shutil.copy(f'../CG_{args.name}.top', f'.')
            # GROMACS simulation: energy minimization, equilibirum, and production.
            gmx.grompp(gro='../CG_initial.gro', mdp='../CG_em.mdp', top=f'CG_{args.name}.top', tpr=f'CG_em_{i}.tpr')
            gmx.mdrun(tpr=f'CG_em_{i}.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp)
            gmx.grompp(gro=f'CG_em_{i}.gro', mdp='../CG_eq.mdp', top=f'CG_{args.name}.top', tpr=f'CG_eq_{i}.tpr',
                       maxwarn=2)
            gmx.mdrun(tpr=f'CG_eq_{i}.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp)
            gmx.grompp(gro=f'CG_eq_{i}.gro', mdp='../CG_run.mdp', top=f'CG_{args.name}.top', tpr=f'CG_run_{i}.tpr',
                       maxwarn=1)
            gmx.mdrun(tpr=f'CG_run_{i}.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp)
            gmx.trjconv(gro=f'CG_run_{i}.xtc', out_gro=f'CG_run_{i}_pbc.xtc', tpr=f'CG_run_{i}.tpr', pbc_whole=True)

            mapping.load_cg_traj(f'CG_run_{i}_pbc.xtc', tpr=f'CG_run_{i}.tpr')
            mapping.update_parameter()
            mapping.write_distribution(CG=True)
            mapping.write_emd()
            for bead in mapping.groups:
                bead.position = None
            os.chdir('..')
    elif args.action == 'stability':
        cd_and_mkdir('4.stability')
        gmx.generate_mdp_from_template('t_CG_em.mdp', mdp_out=f'CG_em.mdp', dielectric=1.0)
        gmx.generate_mdp_from_template('t_npt.mdp', mdp_out=f'CG_eq.mdp', nsteps=100000, dt=0.005,
                                       tcoupl='v-rescale', tau_t='0.1',
                                       pcoupl='berendsen', tau_p='12.0', compressibility='3e-4',
                                       constraints='none', coulombtype='cutoff',
                                       rcoulomb='1.1', rvdw='1.1', dielectric=15, nstlist=20)
        gmx.generate_mdp_from_template('t_npt.mdp', mdp_out=f'CG_run.mdp', nsteps=10000000, dt=0.01, nstxtcout=20,
                                       restart=True,
                                       tcoupl='v-rescale', tau_t='0.1',
                                       pcoupl='parrinello-rahman', tau_p='12.0', compressibility='3e-4',
                                       constraints='none', coulombtype='cutoff', rcoulomb='1.1',
                                       rvdw='1.1', dielectric=15, nstlist=20)
        shutil.copy(f'../CG_{args.name}.itp', '.')
        shutil.copy(f'../CG_{args.name}.top', '.')
        shutil.copy(f'../CG_{args.name}.gro', '.')
        gmx.grompp(gro=f'CG_{args.name}.gro', mdp='CG_em.mdp', top=f'CG_{args.name}.top', tpr=f'CG_em.tpr')
        gmx.mdrun(tpr=f'CG_em.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp)
        gmx.grompp(gro=f'CG_em.gro', mdp='CG_eq.mdp', top=f'CG_{args.name}.top', tpr=f'CG_eq.tpr',
                   maxwarn=2)
        gmx.mdrun(tpr=f'CG_eq.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp)
        gmx.grompp(gro=f'CG_eq.gro', mdp='CG_run.mdp', top=f'CG_{args.name}.top', tpr=f'CG_run.tpr',
                   maxwarn=1)
        gmx.mdrun(tpr=f'CG_run.tpr', ntmpi=args.ntmpi, ntomp=args.ntomp)
    elif args.action == 'swarm-cg':
        cd_and_mkdir('3.swarm-cg')
        mapping = Mapping.load(path='../2.cg-mapping')
        mapping.generate_ndx_mapping(file=f'{args.name}.ndx')
        mapping.generate_itp(f'CG_{args.name}.itp', resName=args.res_name, group=True)
        gmx.generate_top(f'CG_{args.name}.top', include_itps=[f'../../simutools/template/martini_v3.0.0.itp',
                                                              f'../../simutools/template/martini_v3.0.0_solvents_v1.itp',
                                                              f'CG_{args.name}.itp'],
                         mol_name=[args.res_name], mol_number=[1])
        mapping.generate_gro(file=f'CG_{args.name}.gro', resName=args.res_name, box_length=3.8)
        gmx.solvate(f'CG_{args.name}.gro', top=f'CG_{args.name}.top', outgro='CG_initial.gro',
                    solvent=f'{TEMPLATE_DIR}/box_martini3_water.gro')
        for s in ['mini', 'equi', 'md']:
            shutil.copy(f'{TEMPLATE_DIR}/swarm_cg_{s}.mdp', f'..')
        cmd = f'scg_optimize -aa_tpr ../1.all-atom/run.tpr -aa_traj ../1.all-atom/AA-traj.whole.xtc ' \
              f'-cg_map {args.name}.ndx -cg_itp CG_{args.name}.itp -cg_gro CG_initial.gro -cg_top CG_{args.name}.top ' \
              f'-cg_mdp_mini swarm_cg_mini.mdp -cg_mdp_equi swarm_cg_equi.mdp -cg_mdp_md swarm_cg_md.mdp -gmx gmx'
        execute(cmd)

    if args.action == 'cg-sim':
        gmx.generate_top(f'CG_{args.name}.top', include_itps=[f'{TEMPLATE_DIR}/martini_v3.0.0.itp',
                                                              f'{TEMPLATE_DIR}/martini_v3.0.0_solvents_v1.itp',
                                                              f'CG_{args.name}.itp'],
                         mol_name=[args.res_name], mol_number=[1])
        gmx.solvate(f'CG_{args.name}.gro', top=f'CG_{args.name}.top', outgro='CG_initial.gro',
                    solvent=f'{TEMPLATE_DIR}/box_martini3_water.gro')
        gmx.generate_mdp_from_template('t_CG_em.mdp', mdp_out=f'CG_em.mdp', dielectric=1.0)
        gmx.grompp(gro='CG_initial.gro', mdp='CG_em.mdp', top=f'CG_{args.name}.top', tpr='CG_em.tpr')
        gmx.mdrun(tpr='CG_em.tpr')
        # use a small timestep when equilibrium, otherwise LINCS error.
        gmx.generate_mdp_from_template('t_npt.mdp', mdp_out=f'CG_eq.mdp', nsteps=200000, dt=0.02,
                                       tcoupl='v-rescale', tau_t='1.0',
                                       pcoupl='berendsen', tau_p='12.0', compressibility='3e-4',
                                       constraints='none', coulombtype='cutoff',
                                       rcoulomb='1.1', rvdw='1.1', dielectric=15, nstlist=20)
        gmx.grompp(gro='CG_em.gro', mdp='CG_eq.mdp', top=f'CG_{args.name}.top', tpr='CG_eq.tpr')
        gmx.mdrun(tpr='CG_eq.tpr')
        gmx.generate_mdp_from_template('t_npt.mdp', mdp_out=f'CG_run.mdp', nsteps=1000000, dt=0.02, nstxtcout=10,
                                       restart=True,
                                       tcoupl='v-rescale', tau_t='1.0',
                                       pcoupl='parrinello-rahman', tau_p='12.0', compressibility='3e-4',
                                       constraints='none', coulombtype='cutoff', rcoulomb='1.1',
                                       rvdw='1.1', dielectric=15, nstlist=20)
        gmx.grompp(gro='CG_eq.gro', mdp='CG_run.mdp', top=f'CG_{args.name}.top', tpr='CG_run.tpr')
        gmx.mdrun(tpr='CG_run.tpr')
        gmx.trjconv(gro='CG_run.xtc', out_gro='CG_run_pbc.xtc', tpr='CG_run.tpr', pbc_whole=True)
        """
        gmx.generate_mdp_from_template('t_nvt.mdp', mdp_out=f'eq.mdp', nsteps=200000, dt=0.002)
        gmx.grompp(gro='em.gro', mdp='eq.mdp', top=f'{args.name}.top', tpr='eq.tpr')
        gmx.mdrun(tpr='eq.tpr')

        gmx.generate_mdp_from_template('t_npt.mdp', mdp_out=f'run.mdp', nsteps=10000000, dt=0.002, nstxtcout=1000,
                                       restart=True)
        gmx.grompp(gro='eq.gro', mdp='run.mdp', top=f'{args.name}.top', tpr='run.tpr')
        gmx.mdrun(tpr='run.tpr')
        gmx.trjconv(gro='run.xtc', out_gro='AA-traj.whole.xtc', tpr='run.tpr', pbc_whole=True)
        """
    if args.action == 'test':
        cd_and_mkdir('../test')
        mapping = Mapping.load(path='../2.cg-mapping')
        print(mapping.groups[0].com)
        print(mapping.groups[1].com)


if __name__ == '__main__':
    main(args=CommonArgs().parse_args())
