#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
import time
import datetime
import pwd
import subprocess
from subprocess import Popen, PIPE
from simutools.utils.utils import execute


class SlurmJob:
    def __init__(self, id: int, name: str, state: Literal['pending', 'running', 'done'],
                 work_dir: str = None, user: str = None, partition: str = None):
        self.id = id
        self.name = name
        self.state = state
        self.work_dir = work_dir
        self.user = user
        self.partition = partition

    def __repr__(self):
        return '<PbsJob: %i %s %s %s>' % (self.id, self.name, self.state, self.user)

    def __eq__(self, other):
        return self.id == other.id


class Slurm:
    def __init__(self):
        self.username = pwd.getpwuid(os.getuid()).pw_name
        self.stored_jobs = []
        self.sh = '_job_slurm.sh'
        self.submit_cmd = 'sbatch'
        self.update_time = datetime.datetime.now()
        # self.update_stored_jobs()

    @property
    def is_working(self) -> bool:
        """The status of the Slurm."""
        cmd = 'sinfo --version'
        stdout, stderr = execute(cmd)
        return stdout.decode().startswith('slurm')

    @property
    def current_jobs(self) -> List[SlurmJob]:
        """Get the current jobs submitted to Slurm."""
        if (datetime.datetime.now() - self.update_time).total_seconds() >= 60:
            self.update_stored_jobs()
        return self.stored_jobs

    @property
    def n_current_jobs(self) -> int:
        return len(self.current_jobs)

    def generate_sh(self, name: str, commands: List[str], path: str = None, qos: str = None,
                    partition: str = 'cpu', nnodes: int = 1, ntasks: int = 1, n_gpu: int = None, memory: int = None,
                    walltime: int = 48, exclusive: bool = False, exclude: str = None,
                    save_running_time: bool = False, sh_index: bool = False) -> str:
        """ Generate a slurm script: <name>.sh.

        Parameters
        ----------
        name: the name of the job.
        commands: The commands to be executed by the job.
        path: The directory to save the sh file.
        qos: Slurm parameter.
        partition: Slurm parameter.
        ntasks: Slurm parameter.
        n_gpu: number of GPU.
        memory: allocated memory (GB).
        walltime: walltime (hour).
        exclusive: Slurm parameter.
        exclude: exclude bad nodes.
        save_running_time: generate a file that record the computational node
        sh_index: create <name>-<index>.sh as the slurm script.

        Returns
        -------

        """
        if path is None:
            path = os.getcwd()
        # n_mpi, srun_commands = self._replace_mpirun_srun(commands)

        info = '#!/bin/bash\n'
        info += '#SBATCH -D %s\n' % path
        info += '#SBATCH -N %s\n' % nnodes
        info += '#SBATCH --job-name=%s\n' % name
        info += '#SBATCH -o %s.out\n' % name
        info += '#SBATCH -e %s.err\n' % name
        info += '#SBATCH --partition=%s\n' % partition
        if memory is not None:
            info += '#SBATCH --mem=%dG\n' % memory
        if ntasks is not None:
            info += '#SBATCH --ntasks=%i\n' % ntasks
        if n_gpu is not None and n_gpu != 0:
            info += '#SBATCH --gres=gpu:%i\n' % n_gpu
        if exclude is not None:
            info += '#SBATCH --exclude=%s\n' % exclude
        if exclusive:
            info += '#SBATCH --exclusive\n'
        if qos is not None:
            info += '#SBATCH --qos=%s\n' % qos
        info += '#SBATCH --time=%i:0:0\n' % walltime
        if save_running_time:
            info += 'echo $SLURM_NODELIST > %s.time\n' % name
            info += 'echo $(date) >> %s.time\n' % name
        for cmd in commands:
            info += cmd + '\n'
        if save_running_time:
            info += '\necho $(date) >> %s.time' % name

        if sh_index:
            name = name + '-%i' % self._get_local_index_of_job(path=path, name=name)
        filename = os.path.join(path, name + '.sh')
        file = open(filename, 'w')
        file.write(info)
        return filename

    def submit(self, file: str) -> bool:
        cmd = self.submit_cmd + ' ' + file
        return subprocess.call(cmd.split()) == 0

    def is_running(self, name: str) -> bool:
        job = self._get_job_from_name(name)
        if job is None:
            return False
        return job.state in ['pending', 'running']

    def kill_job(self, name: str) -> bool:
        job = self._get_job_from_name(name)
        if job is None:
            return False
        cmd = f'scancel {job.id}'
        return subprocess.call(cmd.split()) == 0

    def update_stored_jobs(self):
        print('Update job information')
        self.stored_jobs = self._get_all_jobs()
        self.update_time = datetime.datetime.now()

    def _get_all_jobs(self) -> List[SlurmJob]:
        """get all jobs of current user.

        Returns
        -------
        A list of SlurmJob.
        """
        cmd = 'scontrol show job'
        sp = Popen(cmd.split(), stdout=PIPE, stderr=PIPE)
        stdout, stderr = sp.communicate()
        if sp.returncode != 0:
            print(stderr.decode())
            return []

        jobs = []
        for job_str in stdout.decode().split('\n\n'):  # split jobs
            if job_str.startswith('JobId'):
                job = self._get_job_from_str(job_str)
                # Show all jobs. Then check the user
                if job.user == self.username:
                    jobs.append(job)
        return jobs

    def _get_job_from_str(self, job_str) -> SlurmJob:
        """create job object from raw information get from 'scontrol show job'."""
        work_dir = None
        for line in job_str.split():  # split properties
            try:
                key, val = line.split('=')[0:2]
            except:
                continue
            if key == 'JobId':
                id = int(val)
            elif key == 'UserId':
                user = val.split('(')[0]  # UserId=username(uid)
            elif key == 'JobName' or key == 'Name':
                name = val
            elif key == 'Partition':
                partition = val
            elif key == 'JobState':
                state_str = val
                if val in ('PENDING', 'RESV_DEL_HOLD'):
                    state = 'pending'
                elif val in ('CONFIGURING', 'RUNNING', 'COMPLETING', 'STOPPED', 'SUSPENDED'):
                    state = 'running'
                else:
                    state = 'done'
            elif key == 'WorkDir':
                work_dir = val
        job = SlurmJob(id=id, name=name, state=state, work_dir=work_dir, user=user, partition=partition)
        return job

    def _get_job_from_name(self, name: str) -> Optional[SlurmJob]:
        """get job information from job name."""
        # if several job have same name, return the one with the largest id (most recent job)
        for job in sorted(self.current_jobs, key=lambda x: x.id, reverse=True):
            if job.name == name:
                return job
        else:
            return None

    def _replace_mpirun_srun(self, commands: List[str]) -> Tuple[int, List[str]]:
        n_mpi = 1
        cmds_replaced = []
        for cmd in commands:
            if cmd.startswith('mpirun'):
                n_mpi = int(cmd.split()[2])
                cmd_srun = 'srun -n %i ' % n_mpi + ' '.join(cmd.split()[3:])
                cmds_replaced.append(cmd_srun)
            else:
                cmds_replaced.append(cmd)
        return n_mpi, cmds_replaced

    @staticmethod
    def _get_local_index_of_job(path: str, name: str):
        """the index assure no slurm jobs overwrite the existed sh file."""
        i = 0
        while os.path.exists(os.path.join(path, '%s-%i.sh' % (name, i))):
            i += 1
        return i
