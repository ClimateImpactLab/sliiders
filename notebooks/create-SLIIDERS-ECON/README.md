# Workflow for generating the SLIIDERS-ECON dataset

This directory contains notebooks to generate the **SLIIDERS-ECON** dataset. The final output for future projections is a Zarr store containing socioeconomic variables binned by coastal segment, elevation slice, and Shared Socioeconomic Pathway.

The steps to produce the final output are as follows.

1. Use `download-sliiders-econ-input-data.ipynb` to download necessary datasets, including various country-level datasets and datasets such as including World Bank Intercomparison Project 2017 and construction cost index by Lincke and Hinkel (2021, *Earth's Future*).
2. Go to the directory `country_level_ypk` and follow the instructions in the `README.md` in that directory. The workflow in `country_level_ypk` cleans (and when necessary, imputes) various country-level socioeconomic variables.
3. Go to the directory `exposure` and follow the instructions in the `README.md` in that directory. The workflow in `exposure` generates current-day global exposure data by coastal segment, elevation, and other variables.
4. Use `create-SLIIDERS-ECON.ipynb` to combine disparate data sources to generate the final output.
