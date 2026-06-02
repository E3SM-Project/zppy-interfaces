***********
pcmdi_diags
***********

Overview
========

The ``pcmdi_diags`` package provides multiple command-line entry points that
prepare inputs for PCMDI Metrics, launch diagnostics, reorganize outputs, and
assemble summary viewers.

The entry points defined by this repository are:

* ``zi-pcmdi-link-observation``
* ``zi-pcmdi-mean-climate``
* ``zi-pcmdi-variability-modes``
* ``zi-pcmdi-enso``
* ``zi-pcmdi-synthetic-plots``

Shared workflow
===============

``zi-pcmdi-mean-climate``, ``zi-pcmdi-variability-modes``, and the unfinished
``zi-pcmdi-enso`` command all use ``zppy_interfaces.pcmdi_diags.pcmdi_setup`` to
perform the same initial preparation work.

Step-by-step shared setup
-------------------------

#. Parse the shared core parameters.
#. Decide whether multiprocessing is active. If ``num_workers < 2``, the code
   disables multiprocessing even if ``multiprocessing=True`` was provided.
#. Determine the reference data set:
   
   * ``run_type=model_vs_obs`` uses ``obs_sets``;
   * ``run_type=model_vs_model`` uses ``model_name_ref`` and ``tableID_ref``.
#. Check whether any requested derived variables need to be created. The setup
   logic can derive ``rstcre`` from ``rsutcs`` and ``rsut``, and ``rltcre``
   from ``rlutcs`` and ``rlut``.
#. Build test and reference catalogue JSON files inside a working
   ``pcmdi_diags`` directory.
#. Optionally generate land-sea mask files under ``fixed/`` when
   ``generate_sftlf`` is enabled.
#. Build path templates for the downstream diagnostics and return them to the
   calling command.

Shared core parameters
----------------------

These parameters are accepted by the mean-climate, variability-modes, and ENSO
commands.

.. list-table::
   :header-rows: 1
   :widths: 18 10 14 58

   * - Parameter
     - Required
     - Default
     - Description
   * - ``--num_workers``
     - Yes
     - -
     - Maximum number of workers used when multiprocessing is enabled.
   * - ``--multiprocessing``
     - Yes
     - -
     - Boolean-like string controlling parallel execution. The setup code still
       disables parallel execution when ``num_workers`` is less than 2.
   * - ``--subsection``
     - Yes
     - -
     - Label used in generated catalogue file names such as
       ``<path>_<subsection>_catalogue.json``.
   * - ``--climo_ts_dir_primary``
     - Yes
     - -
     - Directory containing the primary model climatology or time-series files.
   * - ``--climo_ts_dir_ref``
     - Yes
     - -
     - Directory containing the reference model or observational files.
   * - ``--model_name``
     - Yes
     - -
     - Primary model identifier. The code expects four dot-separated parts:
       ``mip.exp.model.realm``.
   * - ``--model_tableID``
     - Yes
     - -
     - Table identifier paired with ``model_name`` for output naming.
   * - ``--figure_format``
     - Yes
     - -
     - Graphics file format expected from the underlying diagnostics.
   * - ``--run_type``
     - Yes
     - -
     - Must be ``model_vs_obs`` or ``model_vs_model``.
   * - ``--obs_sets``
     - Required for ``model_vs_obs``
     - -
     - Comma-separated observation set identifiers used when the run compares a
       model against observations.
   * - ``--model_name_ref``
     - Required for ``model_vs_model``
     - -
     - Reference model name used when the run compares one model against
       another.
   * - ``--vars``
     - Yes
     - -
     - Comma-separated variable list. Base variable names are extracted by
       splitting on ``_`` and ``-`` where needed.
   * - ``--tableID_ref``
     - Required for ``model_vs_model``
     - -
     - Table identifier paired with ``model_name_ref``.
   * - ``--generate_sftlf``
     - Yes
     - -
     - Boolean-like string controlling land-sea mask generation. The code
       treats ``true``, ``y``, and ``yes`` as enabled.
   * - ``--case_id``
     - Yes
     - -
     - Case identifier appended to generated commands and output file names.
   * - ``--results_dir``
     - Yes
     - -
     - Base directory where organized outputs are written.
   * - ``--debug``
     - No
     - -
     - Enables debug logging when set to ``true``.

Boolean parsing notes
---------------------

The PCMDI-related commands do not all parse booleans the same way:

* ``multiprocessing`` and most ``debug`` flags treat only ``true``
  (case-insensitive) as true.
* ``generate_sftlf`` treats ``true``, ``y``, and ``yes`` as true.
* ``zi-pcmdi-synthetic-plots`` accepts a broader set of truthy and falsy values
  such as ``true``, ``1``, ``yes``, ``on``, ``false``, ``0``, and ``off`` for
  its viewer toggles and ``save_all_data``.

zi-pcmdi-link-observation
=========================

Purpose
-------

``zi-pcmdi-link-observation`` prepares observational time-series files for
PCMDI workflows. It looks up observation aliases, creates symlinks into a
temporary observation directory, and can derive ``rstcre`` or ``rltcre`` when
their source variables exist.

Step-by-step
------------

#. Parse the command-line arguments.
#. Resolve each requested variable to an observation alias using
   ``reference_alias.json``.
#. Search the observation source tree for the matching NetCDF file.
#. Symlink the matching file into ``obstmp_dir`` or write a renamed copy when
   the observational variable must be mapped from a CMIP-style alias.
#. Derive ``rstcre`` and ``rltcre`` if those variables were requested and are
   not already present.

Parameter reference
-------------------

.. list-table::
   :header-rows: 1
   :widths: 18 10 14 58

   * - Parameter
     - Required
     - Default
     - Description
   * - ``--model_name_ref``
     - Yes
     - -
     - Reference model name used to build linked observation file names.
   * - ``--tableID_ref``
     - Yes
     - -
     - Reference table identifier paired with ``model_name_ref``.
   * - ``--vars``
     - Yes
     - -
     - Comma-separated variables to link or derive.
   * - ``--obs_sets``
     - Yes
     - -
     - Observation set identifiers. When multiple sets are supplied and the
       count matches ``vars``, each variable uses its corresponding set;
       otherwise the first set is reused for all variables.
   * - ``--obs_ts``
     - Yes
     - -
     - Root directory containing the source observation time series.
   * - ``--obstmp_dir``
     - Yes
     - -
     - Output directory where symlinks or renamed files are written.
   * - ``--debug``
     - No
     - -
     - Enables debug logging when set to ``true``.

zi-pcmdi-mean-climate
=====================

Purpose
-------

``zi-pcmdi-mean-climate`` prepares PCMDI mean-climate runs, executes the
underlying ``mean_climate_driver.py`` commands, and reorganizes the resulting
figures, metrics JSON files, and NetCDF diagnostics into the ``results_dir``
layout expected by downstream viewers.

Step-by-step
------------

#. Run the shared setup described above.
#. Save a ``regions.json`` file that maps each requested variable key to the
   requested region list.
#. Generate one ``mean_climate_driver.py`` command per requested variable that
   is present in the reference catalogue.
#. Run those commands in parallel or serial mode.
#. Reorganize figures into grouped seasonal directories.
#. Reorganize JSON metrics and NetCDF diagnostics into ``metrics_data``.

Additional parameters
---------------------

.. list-table::
   :header-rows: 1
   :widths: 18 10 14 58

   * - Parameter
     - Required
     - Default
     - Description
   * - ``--regions``
     - Yes
     - -
     - Comma-separated list of regions used both for command preparation and
       for organizing the collected outputs.

zi-pcmdi-variability-modes
==========================

Purpose
-------

``zi-pcmdi-variability-modes`` prepares variability-mode diagnostics, launches
``variability_modes_driver.py`` commands, and then collects graphics, metrics,
and diagnostic files into a structured output tree.

Step-by-step
------------

#. Run the shared setup described above.
#. Read the observation metadata for the requested ``vars`` value.
#. Derive reference years and file paths from the observation catalogue.
#. Generate one ``variability_modes_driver.py`` command per requested mode.
#. Run those commands in parallel or serial mode.
#. Collect graphics into grouped seasonal directories.
#. Collect JSON metrics and NetCDF diagnostics into ``metrics_data``.

Additional parameters
---------------------

.. list-table::
   :header-rows: 1
   :widths: 18 10 14 58

   * - Parameter
     - Required
     - Default
     - Description
   * - ``--var_modes``
     - Yes
     - -
     - Comma-separated list of variability modes to process. Specific modes
       such as ``NPO``, ``NPGO``, ``PSA1``, and ``PSA2`` automatically select a
       higher EOF number than the default of 1.

zi-pcmdi-enso
=============

Purpose
-------

``zi-pcmdi-enso`` contains argument parsing, observation preparation helpers,
driver command generation, and output validation logic for ENSO diagnostics.
However, the current ``main()`` implementation exits immediately with an error
message stating that the command is not yet supported.

Current status
--------------

The CLI is present and documented here so callers can understand the intended
interface, but the command should currently be treated as unavailable.

Additional parameters
---------------------

.. list-table::
   :header-rows: 1
   :widths: 18 10 14 58

   * - Parameter
     - Required
     - Default
     - Description
   * - ``--enso_groups``
     - Yes
     - -
     - Comma-separated ENSO metric group names intended for
       ``enso_driver.py --metricsCollection``.

zi-pcmdi-synthetic-plots
========================

Purpose
-------

``zi-pcmdi-synthetic-plots`` generates summary metric plots and builds viewer
pages that combine mean-climate, variability, and ENSO outputs into a single
web presentation.

Step-by-step
------------

#. Parse the command-line arguments into ``SyntheticPlotsParameters``.
#. Build the expected input path under ``www/put_model_here/pcmdi_diags``.
#. Read ``synthetic_metrics_list.json``.
#. Generate the summary plots through ``SyntheticMetricsPlotter``.
#. Assemble viewer configuration and supporting assets.
#. Write methodology, data, and viewer HTML pages under
   ``web_dir/results_dir/viewer``.

Parameter reference
-------------------

.. list-table::
   :header-rows: 1
   :widths: 18 10 14 58

   * - Parameter
     - Required
     - Default
     - Description
   * - ``--synthetic_sets``
     - Yes
     - -
     - Synthetic plot set selection accepted by the CLI entry point.
   * - ``--figure_format``
     - Yes
     - -
     - Graphics format used for generated summary plots.
   * - ``--www``
     - Yes
     - -
     - Base web directory that contains ``put_model_here/pcmdi_diags`` inputs.
   * - ``--results_dir``
     - Yes
     - -
     - Results subdirectory below both the input and output web locations.
   * - ``--case``
     - Yes
     - -
     - Case name shown in generated plots and viewer pages.
   * - ``--model_name``
     - Yes
     - -
     - Model name shown in generated plots and viewer pages.
   * - ``--model_tableID``
     - Yes
     - -
     - Table identifier shown in generated plots and viewer pages.
   * - ``--web_dir``
     - Yes
     - -
     - Output web root where the viewer and summary products are written.
   * - ``--clim_viewer``
     - Yes
     - -
     - Boolean controlling whether climatology summary products are included.
   * - ``--clim_vars``
     - Yes
     - -
     - Comma-separated climatology variables used in the viewer.
   * - ``--clim_years``
     - Yes
     - -
     - Year range label for climatology products.
   * - ``--clim_regions``
     - Yes
     - -
     - Comma-separated climatology region list used in the viewer.
   * - ``--cmip_clim_dir``
     - Yes
     - -
     - Directory containing CMIP climatology comparison data.
   * - ``--cmip_clim_set``
     - Yes
     - -
     - Name of the CMIP climatology set used by the viewer.
   * - ``--mova_viewer``
     - Yes
     - -
     - Boolean controlling whether atmospheric variability summaries are
       included.
   * - ``--mova_modes``
     - Yes
     - -
     - Comma-separated atmospheric variability modes used in summary products.
   * - ``--mova_vars``
     - Yes
     - -
     - Comma-separated atmospheric variability variables accepted by the CLI.
   * - ``--mova_years``
     - Yes
     - -
     - Year range label for atmospheric variability products.
   * - ``--movc_viewer``
     - Yes
     - -
     - Boolean controlling whether coupled variability summaries are included.
   * - ``--movc_modes``
     - Yes
     - -
     - Comma-separated coupled variability modes used in summary products.
   * - ``--movc_vars``
     - Yes
     - -
     - Comma-separated coupled variability variables accepted by the CLI.
   * - ``--movc_years``
     - Yes
     - -
     - Year range label for coupled variability products.
   * - ``--cmip_movs_dir``
     - Yes
     - -
     - Directory containing CMIP variability comparison data.
   * - ``--cmip_movs_set``
     - Yes
     - -
     - Name of the CMIP variability set used by the viewer.
   * - ``--enso_viewer``
     - Yes
     - -
     - Boolean controlling whether ENSO summaries are included.
   * - ``--enso_vars``
     - Yes
     - -
     - Comma-separated ENSO variables accepted by the CLI.
   * - ``--enso_years``
     - Yes
     - -
     - Year range label for ENSO products.
   * - ``--cmip_enso_dir``
     - Yes
     - -
     - Directory containing CMIP ENSO comparison data.
   * - ``--cmip_enso_set``
     - Yes
     - -
     - Name of the CMIP ENSO set used by the viewer.
   * - ``--pcmdi_webtitle``
     - Yes
     - -
     - Title shown on generated viewer pages.
   * - ``--pcmdi_version``
     - Yes
     - -
     - Version label shown on generated viewer pages.
   * - ``--run_type``
     - Yes
     - -
     - Run-type label converted to a subtitle in the viewer.
   * - ``--pcmdi_external_prefix``
     - Yes
     - -
     - Base directory used to find shared PCMDI web assets and data.
   * - ``--pcmdi_viewer_template``
     - Yes
     - -
     - Template directory below ``pcmdi_external_prefix`` that contains viewer
       assets such as ``e3sm_pmp_logo.png``.
   * - ``--save_all_data``
     - Yes
     - -
     - Boolean controlling whether the plotter saves all underlying data.
   * - ``--debug``
     - No
     - -
     - Enables debug logging when set to ``true``.
