This directory contains notebooks to generate SLIIDERS-SLR; a dataset of gridded local sea-level Monte Carlo samples based on the LocalizeSL framework.

The final output is a Zarr store containing 10,000 Monte Carlo draws for each of the RCP scenarios and years (decadal), at each site ID (defined by LocalizeSL), for each corefile.

The steps to produce this output are as follows:
1. `download-ifile-to-gcs.ipynb`: define the corefiles (IFILES) that you'd like to use and download them on GCS
2. `convert-mat-version.ipynb`: Convert the downloaded corefiles (IFILES) to the Octave-readable MATLAB v5 format.
3. `generate-projected-lsl.ipynb`: Dask workers running Octave. For any corefile, call the LocalizeSL `LocalizeStoredProjections` function, followed by `WriteTableMC`, to get outputs as TSVs.
4. `retrieve-num-gcms.ipynb`: Calculate number of GCMs for each site-year-scenario, for later use in clipping some sites due to data quality issues.
5. `process-localizesl-output.ipynb`: combine all TSVs into a single Zarr store. Clip some sites based on data quality criteria.