import shlex
import time
from subprocess import PIPE, Popen
from typing import Dict, List, Tuple

import psutil

from zppy_interfaces.multi_utils.logger import _setup_child_logger

logger = _setup_child_logger(__name__)

# Mapping from observational variable names to CMIP-standard
ALT_OBS_MAP: Dict[str, str] = {
    "pr": "PRECT",
    "sst": "ts",
    "sfcWind": "si10",
    "taux": "tauu",
    "tauy": "tauv",
    "rltcre": "toa_cre_lw_mon",
    "rstcre": "toa_cre_sw_mon",
    "rtmt": "toa_net_all_mon",
}


def count_child_processes(process=None):
    """
    Count the number of child processes for a given process.

    Parameters:
    - process (psutil.Process, optional): The process to check. If None, uses the current process.

    Returns:
    - int: Number of child processes.
    """
    if process is None:
        process = psutil.Process()

    children = process.children()
    return len(children)


def run_parallel_jobs(cmds: List[str], num_workers: int) -> List[Tuple[str, str, int]]:
    """
    Execute shell commands in parallel batches.

    Parameters:
    - cmds: List of command strings to run.
    - num_workers: Maximum number of subprocesses to run concurrently.

    Returns:
    - List of tuples: (stdout, stderr, return_code) for each command.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")

    results = []
    procs = []

    for i, cmd in enumerate(cmds):
        proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, text=True)
        procs.append((cmd, proc))

        # Run the batch if full or if it's the last command
        if len(procs) >= num_workers or i == len(cmds) - 1:
            logger.info(f"Running batch of {len(procs)} subprocesses...")
            for batch_cmd, batch_proc in procs:
                stdout, stderr = batch_proc.communicate()
                return_code = batch_proc.returncode

                if return_code != 0:
                    # Terminate any remaining running processes in the batch
                    for _, remaining_proc in procs:
                        if remaining_proc.poll() is None:
                            remaining_proc.terminate()
                    logger.error(
                        f"ERROR: Process failed: '{batch_cmd}'\nError: {stderr.strip()}"
                    )
                    raise RuntimeError(f"Subprocess failed: {batch_cmd}")

                results.append((stdout.strip(), stderr.strip(), return_code))

            time.sleep(0.25)  # Throttle before starting the next batch
            procs = []

    return results


def run_serial_jobs(cmds: List[str]) -> List[Tuple[str, str, int]]:
    """
    Execute shell commands one at a time (serially).

    Parameters:
    - cmds: List of command strings to run.

    Returns:
    - List of tuples: (stdout, stderr, return_code) for each command.
    """
    results = []

    for i, cmd in enumerate(cmds):
        logger.info(f"Running [{i + 1}/{len(cmds)}]: {cmd}")
        proc = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, shell=False, text=True)
        stdout, stderr = proc.communicate()
        return_code = proc.returncode

        if return_code != 0:
            logger.error(f"ERROR: Process failed: '{cmd}'\nError: {stderr.strip()}")
            raise RuntimeError(f"Subprocess failed: {cmd}")

        results.append((stdout.strip(), stderr.strip(), return_code))

    return results
