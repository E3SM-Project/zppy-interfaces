.. _index-label:

******************************
zppy-interfaces documentation
******************************

What is zppy-interfaces?
========================

``zppy-interfaces`` adds workflow-specific post-processing utilities on top of
external diagnostics packages used by E3SM post-processing. The repository
currently provides two main interfaces:

* ``global_time_series`` for generating coupled global time-series plots,
  including land-variable handling driven by a packaged CSV catalogue.
* ``pcmdi_diags`` for preparing inputs, launching PCMDI Metrics diagnostics,
  reorganizing outputs, and building viewer-ready summary products.

The package exposes command-line entry points that are meant to be called from
automation such as ``zppy``. Each entry point accepts explicit command-line
arguments, performs limited validation, and writes outputs into predictable
directory layouts.

How to use this documentation
=============================

This documentation is organized by interface:

* :doc:`global_time_series` documents the ``zi-global-time-series`` command,
  the supported plotting modes, and every available parameter.
* :doc:`pcmdi_diags` documents the PCMDI-related commands, their shared setup
  logic, and every available parameter across the CLI entry points.

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

Processing model
================

At a high level, the package follows the same pattern across its command-line
tools:

#. Parse explicit command-line arguments.
#. Normalize selected values such as booleans, comma-separated lists, and
   region names.
#. Assemble input and output paths from the provided workflow context.
#. Optionally derive or reorganize supporting files needed by downstream tools.
#. Invoke the plotting or diagnostics logic.
#. Write figures, metrics, or viewer assets into workflow-specific results
   directories.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   global_time_series
   pcmdi_diags
