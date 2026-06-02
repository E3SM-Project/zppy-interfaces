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

This documentation is organized by audience:

* :doc:`user_guide` groups the interface guides for people running the command
  line tools.
* :doc:`developer_guide` groups the implementation guides for people working on
  the repository internals.

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
   user_guide
   developer_guide
