*****************************
Developer Guide: pcmdi_diags
*****************************

This page describes how the PCMDI-related CLI entry points are implemented. For
the runtime parameters and command behavior, see :doc:`pcmdi_diags`.

Implementation map
==================

The PCMDI interface is split across several CLI entry points:

* ``link_observation.py`` handles observation discovery, linking, and limited
  derived-variable generation.
* ``pcmdi_mean_cimate.py`` runs mean-climate diagnostics and reorganizes the
  resulting outputs.
* ``pcmdi_variability_modes.py`` runs variability-mode diagnostics and collects
  the resulting files.
* ``pcmdi_enso.py`` defines the intended ENSO interface, but ``main()`` still
  exits early because the command is not yet supported.
* ``pcmdi_synthetic_plots.py`` builds summary plots and the combined viewer
  pages from prior diagnostics output.

Shared setup path
=================

``zi-pcmdi-mean-climate``, ``zi-pcmdi-variability-modes``, and
``zi-pcmdi-enso`` all start by calling ``zppy_interfaces.pcmdi_diags.pcmdi_setup``:

#. Each CLI parses its command-specific arguments plus the shared core
   arguments.
#. ``CoreParameters`` stores the shared values needed by downstream commands.
#. ``set_up()`` resolves the reference data source, builds catalogue JSON files,
   optionally generates land-sea masks, and returns a ``CoreOutput`` object with
   reusable path templates and observation metadata.
#. The calling CLI then generates command lists or viewer inputs using that
   shared setup result.

Command-specific flows
======================

Mean climate
------------

#. Parse arguments and run the shared setup.
#. Save ``regions.json`` for variable-to-region mapping.
#. Build one ``mean_climate_driver.py`` command per supported variable.
#. Execute those commands in serial or parallel mode.
#. Use ``MeanClimateMetricsCollector`` to move figures, metrics JSON files, and
   NetCDF diagnostics into the ``results_dir`` structure expected by later
   stages.

Variability modes
-----------------

#. Parse arguments and run the shared setup.
#. Build one ``variability_modes_driver.py`` command per requested mode.
#. Execute those commands in serial or parallel mode.
#. Reorganize graphics, metrics, and diagnostics into the structured output
   tree used by the viewer layer.

Synthetic plots and viewer
--------------------------

#. Parse ``SyntheticPlotsParameters`` in ``pcmdi_synthetic_plots.py``.
#. Read ``synthetic_metrics_list.json`` and generate summary figures through
   ``SyntheticMetricsPlotter``.
#. Build the viewer configuration with helper functions from
   ``zppy_interfaces.pcmdi_diags.viewer``.
#. Write the methodology, data, and viewer HTML pages under
   ``web_dir/results_dir/viewer``.
