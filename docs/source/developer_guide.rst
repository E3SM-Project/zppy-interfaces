******************
Developer Guide
******************

Use these pages when you want to understand how the repository is organized,
where each CLI entry point starts, and how the internal processing flow is
implemented.

Repository layout
=================

The implementation is organized under ``zppy_interfaces``:

* ``zppy_interfaces/global_time_series`` contains the global time-series CLI,
  ocean time-series generation helpers, HTML templates, and the
  ``zppy_land_fields.csv`` variable catalogue.
* ``zppy_interfaces/pcmdi_diags`` contains CLI entry points for observation
  linking, mean-climate diagnostics, variability modes, ENSO, synthetic plots,
  and the shared setup logic used by those commands.
* ``zppy_interfaces/multi_utils`` contains shared logging and viewer helpers
  used by both interfaces.

Interface implementation guides
===============================

.. toctree::
   :maxdepth: 1

   developer_global_time_series
   developer_pcmdi_diags

Testing
=======

.. code-block:: bash

    # Set up branch
    cd zppy-interfaces
    git status # Check for uncommitted changes
    git fetch upstream main
    git checkout main
    git reset --hard upstream/main
    git log --oneline | head -n 1
    # Check that we're up to date with either:
    # 1. The latest commit on main: https://github.com/E3SM-Project/zppy-interfaces/commits/main/
    # 2. The commits of the pull request being tested.

    # Set up conda environment
    bash # Run bash so we're in an isolated subshell
    source ~/.bashrc
    # Make sure conda is activated
    rm -rf build
    conda clean --all --y
    conda env create -f conda/dev.yml -n env-name
    conda activate env-name
    pre-commit run --all-files
    python -m pip install .

    # Run unit tests
    pytest tests/unit/global_time_series/test_*.py
    pytest tests/unit/pcmdi_diags/test_*.py

    # Note that integration testing is done as part of zppy's testing.
