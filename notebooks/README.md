# Instructions

This directory contains sub-directories to produce final SLIIDERS outputs. The order of execution is as follows. For further instructions, we refer to the `README.md` files in the respective sub-directories.

1. `create-SLIIDERS-SLR`: Workflow to generate **SLIIDERS-SLR**, a dataset of gridded local sea-level Monte Carlo samples for each RCP scenario, year (decadal), and site ID (defined by LocalizeSL).
2. `create-SLIIDERS-ECON`: Workflow to generate **SLIIDERS-ECON**, a dataset containing socioeconomic variables by coastal segment, elevation, Shared Socioeconomic Pathway scenario. Note that this workflow uses the SLIIDERS-SLR dataset to find nearest grid cells to match to coastal segments.
