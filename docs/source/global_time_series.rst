******************
global_time_series
******************

Overview
========

The ``global_time_series`` interface is exposed through the
``zi-global-time-series`` command. It generates coupled global time-series
products from post-processed E3SM component outputs and can also prepare the
additional ocean time-series files needed by the classic eight-panel plots.

This page focuses on how to run the command and interpret its outputs. For the
implementation path through the Python modules, see
:doc:`developer_global_time_series`.

Step-by-step run flow
=====================

#. Parse command-line arguments.
#. Convert boolean-like strings such as ``True`` and ``False`` to Python
   booleans.
#. Convert ``regions`` to normalized region codes:
   ``glb``/``global`` -> ``glb``, ``n``/``north``/``northern`` -> ``n``, and
   ``s``/``south``/``southern`` -> ``s``.
#. Convert comma-separated plot selections into lists. Passing ``None`` means
   "no plots for this group".
#. Validate the request:
   
   * at least one plot must be requested;
   * if ocean plots are requested through the classic plot set and
     ``use_ocn=True``, ``moc_file`` must be provided.
#. If ``use_ocn=True`` and the classic plot selection includes ocean-dependent
   plots, create ocean time-series files in ``results_dir/ocn/glb/ts/monthly``.
#. Run the coupled global plotting driver, which writes figures and HTML output
   into ``results_dir``.

Output modes
============

The command supports two presentation modes controlled by ``make_viewer``.

* ``make_viewer=True`` generates viewer-oriented HTML output. Classic plots are
  still produced when ``plots_original`` is non-empty, and component plots are
  arranged into viewer pages by variable group and region.
* ``make_viewer=False`` generates a simpler file-list style output. Classic
  plots remain available, and component plots are emitted as cumulative PDFs
  instead of viewer pages.

Classic plot names
==================

``plots_original`` accepts the following named plot requests:

* ``net_toa_flux_restom`` -> ``RESTOM``
* ``net_atm_energy_imbalance`` -> ``RESTOM`` and ``RESSURF``
* ``global_surface_air_temperature`` -> ``TREFHT``
* ``toa_radiation`` -> ``FSNTOA`` and ``FLUT``
* ``net_atm_water_imbalance`` -> ``PRECT`` and ``QFLX``
* ``change_ohc`` -> ocean heat content plot
* ``max_moc`` -> maximum meridional overturning circulation plot
* ``change_sea_level`` -> sea-level change plot

The three ocean-related plot names (``change_ohc``, ``max_moc``,
``change_sea_level``) above require ocean support data.

Some of the variables above are derived rather than read directly. For EAM,
``RESTOM`` is ``FSNT - FLNT``, ``RESSURF`` is ``FSNS - FLNS - SHFLX - LHFLX``,
and ``PRECT`` is ``1e3 * (PRECC + PRECL)``.

EAMxx support
=============

EAMxx names these variables differently. This is detected automatically: if any
EAMxx variable is present in the atmosphere time-series directory, the classic
plots are read from the EAMxx fields instead. Nothing needs to be configured,
and the plot names, the plots themselves, and their units are unchanged; only
the variables read from the time-series files differ.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Variable
     - EAMxx source variables
   * - ``TREFHT``
     - ``T_2m``
   * - ``FLUT``
     - ``LW_flux_up_at_model_top``
   * - ``FSNTOA``
     - ``SW_flux_dn_at_model_top - SW_flux_up_at_model_top``
   * - ``RESTOM``
     - ``SW_flux_dn_at_model_top - SW_flux_up_at_model_top -``
       ``LW_flux_up_at_model_top``
   * - ``RESSURF``
     - ``(SW_flux_dn_at_model_bot - SW_flux_up_at_model_bot) -``
       ``(LW_flux_up_at_model_bot - LW_flux_dn_at_model_bot) -``
       ``surf_sens_flux - LHFLX``
   * - ``LHFLX``
     - ``(2.501e6+3.337e5)*surf_evap - 3.337e5*1e3*precip_liq_surf_mass_flux``
   * - ``PRECT``
     - ``1e3 * (precip_liq_surf_mass_flux + precip_ice_surf_mass_flux)``
   * - ``QFLX``
     - ``surf_evap``

These definitions match the EAMxx entries in e3sm_diags'
``DERIVED_VARIABLES``. Note that EAMxx's "model top" is the top of the
atmosphere, so there is no distinction between the model-top fluxes and the
TOA fluxes.

The requested variables must be present in
``post/atm/glb/ts/monthly/<ts_num_years>yr/`` under their EAMxx names, so the
corresponding zppy ``[ts]`` subtask must include them in its variable list.

Component plots (``plots_atm``) are not translated: for EAMxx, request EAMxx
variable names directly.

Component plot requests
=======================

``plots_atm``, ``plots_ice``, and ``plots_ocn`` accept comma-separated variable
names. These are treated as generic variable names and plotted directly.

``plots_lnd`` is special:

* If set to ``None``, no land plots are generated.
* If set to ``all``, every variable listed in
  `zppy_interfaces/global_time_series/zppy_land_fields.csv`_ is included. (NOTE: this shows the catalogue as of the latest ``main`` branch, which may differ from the version in the release you are using.)
* Otherwise, each requested land variable must match a row in
  ``zppy_interfaces/global_time_series/zppy_land_fields.csv``.

The CSV file also defines whether each land variable is treated as a land-area
average or a land total, along with the scale factor, units, plotting group,
and descriptive long name used by the generated pages.

Boolean parsing note
====================

``make_viewer`` and ``use_ocn`` are parsed with a strict string check: only
``true`` (case-insensitive) is treated as true. Any other value is treated as
false.

Parameter reference
===================

The following parameters are accepted by ``zi-global-time-series``.

.. list-table::
   :header-rows: 1
   :widths: 18 10 14 58

   * - Parameter
     - Required
     - Default
     - Description
   * - ``--make_viewer``
     - No
     - ``False``
     - Selects viewer-oriented output when true; otherwise the command writes a
       simpler results listing and component PDFs.
   * - ``--case_dir``
     - Yes
     - N/A
     - Case directory used to locate existing post-processing output, including
       cached MOC time-series files for ocean-enabled classic plots.
   * - ``--experiment_name``
     - Yes
     - N/A
     - Label used for the experiment in generated plot metadata.
   * - ``--figstr``
     - No
     - Empty string
     - Prefix used in generated figure file names.
   * - ``--color``
     - No
     - ``Blue``
     - Plot color assigned to the experiment in generated figures.
   * - ``--ts_num_years``
     - No
     - ``5``
     - Width, in years, of each time-series chunk expected in the input
       directory layout.
   * - ``--results_dir``
     - Yes
     - N/A
     - Base output directory for figures, HTML, and any generated ocean
       time-series files.
   * - ``--regions``
     - No
     - ``glb,n,s``
     - Comma-separated set of region codes or aliases. Valid aliases normalize
       to ``glb``, ``n``, and ``s``.
   * - ``--start_yr``
     - Yes
     - N/A
     - First simulation year to include.
   * - ``--end_yr``
     - Yes
     - N/A
     - Last simulation year to include.
   * - ``--use_ocn``
     - No
     - ``False``
     - Enables ocean support data generation for the classic ocean-related
       plots.
   * - ``--input``
     - Yes when ``use_ocn=True``
     - N/A
     - Base directory containing raw MPAS-O input used to construct ocean
       time-series files.
   * - ``--input_subdir``
     - No
     - ``archive/ocn/hist``
     - Subdirectory below ``input`` that contains raw MPAS-O monthly history
       files.
   * - ``--moc_file``
     - Yes when ocean-enabled classic plots are requested
     - ``None``
     - Name of the cached MOC file copied from ``case_dir`` into the generated
       ocean results directory.
   * - ``--plots_original``
     - No
     - ``net_toa_flux_restom,global_surface_air_temperature,toa_radiation,``
       ``net_atm_energy_imbalance,change_ohc,max_moc,change_sea_level,``
       ``net_atm_water_imbalance``
     - Comma-separated list of named classic plot requests. Use ``None`` to
       disable the classic plots entirely.
   * - ``--plots_atm``
     - No
     - ``None``
     - Comma-separated list of atmosphere variables to plot.
   * - ``--plots_ice``
     - No
     - ``None``
     - Comma-separated list of sea-ice variables to plot.
   * - ``--plots_lnd``
     - No
     - ``None``
     - Comma-separated list of land variables from ``zppy_land_fields.csv``, or
       ``all`` to include the full land catalogue.
   * - ``--plots_ocn``
     - No
     - ``None``
     - Comma-separated list of ocean variables to plot as component plots.
   * - ``--ncols``
     - No
     - ``2``
     - Number of columns used when building PDF output.
   * - ``--nrows``
     - No
     - ``4``
     - Number of rows used when building PDF output.

Important path conventions
==========================

The implementation expects the usual zppy post-processing layout inside
``case_dir``. For example, component time-series inputs are read from paths such
as ``post/atm/glb/ts/monthly/<ts_num_years>yr/`` and
``post/lnd/glb/ts/monthly/<ts_num_years>yr/``. Ocean support files are written
under ``results_dir/ocn/glb/ts/monthly/<ts_num_years>yr/``.

Land variable catalogue
=======================

The full land-variable list is packaged with the repository:

* `zppy_interfaces/global_time_series/zppy_land_fields.csv`_ (NOTE: this shows the catalogue as of the latest ``main`` branch, which may differ from the version in the release you are using.)

That file is the authoritative source for:

* the accepted ``plots_lnd`` variable names,
* whether each variable is treated as an average or total,
* unit conversions,
* variable grouping on viewer pages, and
* the long descriptions shown to users.

.. _zppy_interfaces/global_time_series/zppy_land_fields.csv: https://github.com/E3SM-Project/zppy-interfaces/blob/main/zppy_interfaces/global_time_series/zppy_land_fields.csv
