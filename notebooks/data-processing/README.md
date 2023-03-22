# Data Processing for SLIIDERS

SLIIDERS is generated from two main intermediate data products. The first reflects historical and projected socioeconomic *temporal* trends at the country level. The second assimilates a variety of gridded and administrative-level data to reflect the present-day *spatial* distribution of exposued people and capital. The final SLIIDERS dataset reflects the product of these spatial and temporal trends.

To execute this workflow run the scripts in the following notebooks in the following order:

1. `1-country-level-temporal-trends`
2. `2-present-day-exposure`
3. `3-create-SLIIDERS.ipynb`: This notebook combines these data sources to create the final SLIIDERS dataset used as input to pyCIAM.