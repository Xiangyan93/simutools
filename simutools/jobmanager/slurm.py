#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Slurm Job Management System

This module provides a Python interface for interacting with the Slurm workload manager.
It allows for creating, submitting, and managing Slurm jobs.
"""

from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
import time
import datetime
import pwd
import subprocess
from subprocess import Popen, PIPE
from simutools.utils.utils import execute


class SlurmJob:
    """
    Represents a single Slurm job.

    Attributes:
        id (int): Unique identifier for the job.
        name (str): Name of the job.
        state (Literal['pending', 'running', 'done']): Current state of the job.
        work_dir (str, optional): Working directory for the job.
        user (str, optional): User who submitted the job.
        partition (str, optional): Slurm partition the job is running on.
    """

    def __init__(self, id: int, name: str, state: Literal['pending', 'running', 'done'],
                 work_dir: str = None, user: str = None, partition: str = None):
        self.id = id
        self.name = name
        self.state = state
        self.work_dir = work_dir
        self.user = user
        self.partition = partition

    def __repr__(self):
        """Returns a string representation of the job."""
        return '<PbsJob: %i %s %s %s>' % (self.id, self.name, self.state, self.user)

    def __eq__(self, other):
        """Compares two SlurmJob objects based on their ID."""
        return self.id == other.id


class Slurm:
    """
    Main class for interacting with the Slurm system.

    Attributes:
        username (str): Current user's username.
        stored_jobs (List[SlurmJob]): List of current jobs.
        sh (str): Default name for Slurm script files.
        submit_cmd (str): Command used to submit jobs (default: 'sbatch').
        update_time (datetime): Timestamp of the last job update.
    """

    def __init__(self):
        self.username = pwd.getpwuid(os.getuid()).pw_name
        self.stored_jobs = []
        self.sh = '_job_slurm.sh'
        self.submit_cmd = 'sbatch'
        self.update_time = datetime.datetime.now()
        # self.update_stored_jobs()

    @property
    def is_working(self) -> bool:
        """Checks if Slurm is operational."""
        cmd = 'sinfo --version'
        stdout, stderr = execute(cmd)
        return stdout.decode().startswith('slurm')

    @property
    def current_jobs(self) -> List[SlurmJob]:
        """Returns the list of current jobs, updating if necessary."""
        if (datetime.datetime.now() - self.update_time).total_seconds() >= 60:
            self.update_stored_jobs()
        return self.stored_jobs

    @property
    def n_current_jobs(self) -> int:
        """Returns the number of current jobs."""
        return len(self.current_jobs)

    def generate_sh(self, name: str, commands: List[str], path: str = None, qos: str = None,
                    partition: str = 'cpu', nnodes: int = 1, ntasks: int = 1, n_gpu: int = None, gpu: str = None,
                    memory: int = None, walltime: int = 48, exclusive: bool = False, 
                    nodelist: str = None, exclude: str = None,
                    save_running_time: bool = False, sh_index: bool = False) -> str:
        """
        Generate a slurm script: <name>.sh.

        Parameters:
            name (str): The name of the job.
            commands (List[str]): The commands to be executed by the job.
            path (str, optional): The directory to save the sh file.
            qos (str, optional): Slurm parameter.
            partition (str, default='cpu'): Slurm partition.
            nnodes (int, default=1): Number of nodes.
            ntasks (int, default=1): Number of tasks.
            n_gpu (int, optional): Number of GPUs.
            gpu (str, optional): GPU specification.
            memory (int, optional): Allocated memory in GB.
            walltime (int, default=48): Wall time in hours.
            exclusive (bool, default=False): Exclusive node usage.
            nodelist (str, optional): Nodes to submit.
            exclude (str, optional): Nodes to exclude.
            save_running_time (bool, default=False): Save job runtime information.
            sh_index (bool, default=False): Append index to script filename.

        Returns:
            str: Path to the generated script file.
        """
        if path is None:
            path = os.getcwd()
        if sh_index:
            name = name + '-%i' % self._get_local_index_of_job(path=path, name=name)
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
            if gpu is not None:
                info += '#SBATCH --gres=gpu:%s:%i\n' % (gpu, n_gpu)
            else:
                info += '#SBATCH --gres=gpu:%i\n' % n_gpu
        if exclusive:
            info += '#SBATCH --exclusive\n'
        if nodelist is not None:
            info += '#SBATCH --nodelist=%s\n' % exclude
        if exclude is not None:
            info += '#SBATCH --exclude=%s\n' % exclude
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

        filename = os.path.join(path, name + '.sh')
        file = open(filename, 'w')
        file.write(info)
        return filename

    def submit(self, file: str) -> bool:
        """
        Submits a Slurm script file to the job queue.

        Parameters:
            file (str): Path to the Slurm script file.

        Returns:
            bool: True if submission was successful, False otherwise.
        """
        cmd = self.submit_cmd + ' ' + file
        return subprocess.call(cmd.split()) == 0

    def is_running(self, name: str) -> bool:
        """
        Checks if a job with the given name is currently running or pending.

        Parameters:
            name (str): Name of the job to check.

        Returns:
            bool: True if the job is running or pending, False otherwise.
        """
        job = self._get_job_from_name(name)
        if job is None:
            return False
        return job.state in ['pending', 'running']

    def kill_job(self, name: str) -> bool:
        """
        Terminates a job with the given name.

        Parameters:
            name (str): Name of the job to terminate.

        Returns:
            bool: True if the job was successfully terminated, False otherwise.
        """
        job = self._get_job_from_name(name)
        if job is None:
            return False
        cmd = f'scancel {job.id}'
        return subprocess.call(cmd.split()) == 0

    def update_stored_jobs(self):
        """Updates the list of stored jobs with current information from Slurm."""
        print('Update job information')
        self.stored_jobs = self._get_all_jobs()
        self.update_time = datetime.datetime.now()

    def _get_all_jobs(self) -> List[SlurmJob]:
        """
        Retrieves all jobs for the current user from Slurm.

        Returns:
            List[SlurmJob]: List of SlurmJob objects representing current jobs.
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
        """
        Parses job information from a string returned by 'scontrol show job'.

        Parameters:
            job_str (str): String containing job information.

        Returns:
            SlurmJob: SlurmJob object with parsed information.
        """
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
        """
        Retrieves a job by its name from the list of current jobs.

        Parameters:
            name (str): Name of the job to retrieve.

        Returns:
            Optional[SlurmJob]: SlurmJob object if found, None otherwise.
        """
        # if several job have same name, return the one with the largest id (most recent job)
        for job in sorted(self.current_jobs, key=lambda x: x.id, reverse=True):
            if job.name == name:
                return job
        else:
            return None

    def _replace_mpirun_srun(self, commands: List[str]) -> Tuple[int, List[str]]:
        """
        Replaces mpirun commands with their srun equivalents.

        Parameters:
            commands (List[str]): List of commands to process.

        Returns:
            Tuple[int, List[str]]: Number of MPI processes and list of updated commands.
        """
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
        """
        Generates a unique index for job script files to avoid overwriting.

        Parameters:
            path (str): Directory path for the script file.
            name (str): Base name of the job.

        Returns:
            int: Unique index for the job script file.
        """
        i = 0
        while os.path.exists(os.path.join(path, '%s-%i.sh' % (name, i))):
            i += 1
        return i
