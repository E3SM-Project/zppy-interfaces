import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    results: List[Tuple[str, str, int]] = []
    procs = []

    for i, cmd in enumerate(cmds):
        proc = Popen(
            cmd,
            stdout=PIPE,
            stderr=PIPE,
            shell=True,
            text=True,
            start_new_session=os.name == "posix",
        )
        procs.append((cmd, proc))

        # Run the batch if full or if it's the last command
        if len(procs) >= num_workers or i == len(cmds) - 1:
            logger.info(f"Running batch of {len(procs)} subprocesses...")
            batch_results: Dict[int, Tuple[str, str, int]] = {}
            failed_command = None
            failed_stderr = ""

            with ThreadPoolExecutor(max_workers=len(procs)) as executor:
                futures = {
                    executor.submit(batch_proc.communicate): (
                        batch_index,
                        batch_cmd,
                        batch_proc,
                    )
                    for batch_index, (batch_cmd, batch_proc) in enumerate(procs)
                }

                for future in as_completed(futures):
                    batch_index, batch_cmd, batch_proc = futures[future]
                    stdout, stderr = future.result()
                    return_code = batch_proc.returncode
                    batch_results[batch_index] = (
                        stdout.strip(),
                        stderr.strip(),
                        return_code,
                    )

                    if return_code != 0 and failed_command is None:
                        failed_command = batch_cmd
                        failed_stderr = stderr.strip()

                        # Stop unfinished jobs immediately instead of waiting for
                        # earlier submissions to finish first.
                        running_procs = [
                            remaining_proc
                            for _, remaining_proc in procs
                            if remaining_proc.poll() is None
                        ]
                        for remaining_proc in running_procs:
                            _signal_process_group(remaining_proc)

                        # Bound termination and reap processes to avoid zombies.
                        for remaining_proc in running_procs:
                            if remaining_proc.poll() is None:
                                try:
                                    remaining_proc.wait(timeout=5)
                                except Exception:
                                    pass

                        # A shell may exit before one of its children. Signal each
                        # original process group again so no descendants survive.
                        for remaining_proc in running_procs:
                            _signal_process_group(remaining_proc, force=True)
                            if remaining_proc.poll() is None:
                                remaining_proc.wait()

            if failed_command is not None:
                logger.error(
                    f"ERROR: Process failed: '{failed_command}'\n"
                    f"Error: {failed_stderr}"
                )
                raise RuntimeError(f"Subprocess failed: {failed_command}")

            results.extend(batch_results[index] for index in range(len(procs)))

            time.sleep(0.25)  # Throttle before starting the next batch
            procs = []

    return results


def _signal_process_group(process, force=False):
    """Signal a process and, on POSIX, every descendant in its process group."""
    process_pid = getattr(process, "pid", None)
    if os.name == "posix" and process_pid is not None:
        sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            # start_new_session=True makes the child PID its process-group ID.
            os.killpg(process_pid, sig)
        except ProcessLookupError:
            pass
        return

    if process.poll() is not None:
        return
    if force:
        process.kill()
    else:
        process.terminate()


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

        proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, text=True)
        stdout, stderr = proc.communicate()
        return_code = proc.returncode

        stdout = stdout.strip()
        stderr = stderr.strip()

        if return_code != 0:
            logger.error(
                f"ERROR: Process failed [{i + 1}/{len(cmds)}]: '{cmd}'\n"
                f"Return code: {return_code}\n"
                f"STDOUT:\n{stdout}\n"
                f"STDERR:\n{stderr}"
            )
            raise RuntimeError(
                f"Subprocess failed [{i + 1}/{len(cmds)}] "
                f"with return code {return_code}: {cmd}"
            )

        results.append((stdout, stderr, return_code))

    return results
