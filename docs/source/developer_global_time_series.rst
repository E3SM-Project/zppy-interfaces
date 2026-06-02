***********************************
Developer Guide: global_time_series
***********************************

This page describes how ``zi-global-time-series`` is implemented. For the
runtime parameters and output-facing behavior, see :doc:`global_time_series`.

Entry point and main flow
=========================

The CLI entry point is ``zppy_interfaces.global_time_series.__main__:main``.
The implementation follows this sequence:

#. ``__main__._get_args()`` parses the command-line arguments and builds a
   ``zppy_interfaces.global_time_series.utils.Parameters`` instance.
#. ``Parameters`` normalizes booleans, region aliases, and comma-separated plot
   selections, then validates that the request is internally consistent.
#. ``main()`` optionally calls ``create_ocean_ts()`` when ocean-dependent
   classic plots are requested with ``use_ocn=True``.
#. ``main()`` then calls ``run_coupled_global()``.
#. ``run_coupled_global()`` builds a
   ``zppy_interfaces.global_time_series.coupled_global.utils.RequestedVariables``
   object, generates the plots, and optionally creates viewer HTML output.

Key modules
===========

* ``zppy_interfaces/global_time_series/__main__.py`` contains the CLI parser
  and the top-level control flow.
* ``zppy_interfaces/global_time_series/utils.py`` defines ``Parameters`` and
  the request normalization logic.
* ``zppy_interfaces/global_time_series/create_ocean_ts.py`` creates ocean
  support time series before plotting when needed.
* ``zppy_interfaces/global_time_series/coupled_global/driver.py`` coordinates
  original plots, component plots, and viewer generation.
* ``zppy_interfaces/global_time_series/coupled_global/mode_pdf.py`` assembles
  cumulative PDFs when ``make_viewer=False``.
* ``zppy_interfaces/global_time_series/coupled_global/mode_viewer.py`` and the
  ``mix_viewer_*`` helpers build the viewer pages when ``make_viewer=True``.

Developer notes
===============

* The classic plot names in ``plots_original`` are not the same thing as raw
  variable names, so the driver keeps them separate from the component-variable
  lists.
* Land plots depend on ``zppy_land_fields.csv`` for the accepted variable set,
  grouping metadata, units, and long names.
* The driver always writes output relative to ``results_dir``, and viewer mode
  adds HTML pages on top of the figure generation rather than replacing the
  underlying plot production.
