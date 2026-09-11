import signal
import threading

import pytest

from zppy_interfaces.pcmdi_diags import utils


class FakeProcess:
    def __init__(self, stdout, stderr, return_code, ready=None):
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code
        self.returncode = None
        self.ready = ready or threading.Event()
        if ready is None:
            self.ready.set()
        self.terminated = False

    def communicate(self):
        self.ready.wait()
        self.returncode = -15 if self.terminated else self.return_code
        return self.stdout, self.stderr

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15
        self.ready.set()

    def wait(self, timeout=None):
        self.ready.wait(timeout)
        return self.returncode

    def kill(self):
        self.terminate()


def test_run_parallel_jobs_detects_later_failure_without_waiting_for_earlier_job(
    monkeypatch,
):
    blocked = FakeProcess("slow", "", 0, ready=threading.Event())
    failed = FakeProcess("", "failed", 1)
    processes = iter([blocked, failed])
    monkeypatch.setattr(utils, "Popen", lambda *args, **kwargs: next(processes))

    with pytest.raises(RuntimeError, match="Subprocess failed: fails"):
        utils.run_parallel_jobs(["blocks", "fails"], num_workers=2)

    assert blocked.terminated is True


def test_run_parallel_jobs_preserves_submission_order(monkeypatch):
    first = FakeProcess("first", "", 0)
    second = FakeProcess("second", "", 0)
    processes = iter([first, second])
    popen_kwargs = []

    def fake_popen(*args, **kwargs):
        popen_kwargs.append(kwargs)
        return next(processes)

    monkeypatch.setattr(utils, "Popen", fake_popen)

    assert utils.run_parallel_jobs(["first", "second"], num_workers=2) == [
        ("first", "", 0),
        ("second", "", 0),
    ]
    assert all(
        kwargs["start_new_session"] is (utils.os.name == "posix")
        for kwargs in popen_kwargs
    )


def test_signal_process_group_targets_entire_posix_group(monkeypatch):
    process = FakeProcess("", "", 0)
    process.pid = 12345
    signals = []
    monkeypatch.setattr(
        utils.os,
        "killpg",
        lambda process_group, sig: signals.append((process_group, sig)),
    )

    utils._signal_process_group(process)
    utils._signal_process_group(process, force=True)

    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]
