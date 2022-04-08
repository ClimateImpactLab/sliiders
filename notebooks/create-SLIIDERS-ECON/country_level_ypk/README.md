# Workflow for organizing and projecting GDP (`Y`), population (`P`), capital stock (`K`), and related variables for historical (1950-2020) and future (2010-2100) timelines

**This version: last updated on March 30, 2022**

## 1. Overview

This directory contains the data acquistion, clean-up, and projection notebook files to organize and project variables including GDP, GDP per capita (GDPpc), population, and capital stock for both historical (1950-2020) and future or projected (2010-2100) timelines. Many of the data sources used to generate historical and future panels have missing data, and therefore efforts were made to impute these missing data through either some form of extrapolation or other established methods. Also, efforts were made to keep the PPP and USD units consistent (e.g., constant 2019 PPP USD) across different sources having different vintages of PPP and USD units.

Below is a quick summary of what each file seeks to accomplish (where the header `ypk` stands for "GDP, population, and capital stock").
1. `ypk1_prep_clean.ipynb`: cleans up selected raw datasets requiring more attention than others to be consistent and workable with other datasets.
2. `ypk2_reorg_and_impute.ipynb`: reorganizes the raw and previously-cleaned historical datasets so that each variable considered has a single, consistent stream of values for each country. After this process, imputes missing GDPpc, GDP, and population values that might still be missing from the cleaned historical dataset.
3. `ypk3_demo_ratios_historical_reg.ipynb`: contains code to clean and extrapolate demographic (age-group) ratios and create the "demographic variables" necessary to conduct the "historical regression" (According to Higgins, 1998) of finding the relationship between investment-to-GDP ratio (I/Y ratio) and demographic variables, (relative) GDPpc, and GDPpc growth rate. Furthermore, the said historical regression is conducted to acquire estimates of investment-to-GDP ratios for missing country-years.
4. `ypk4_impute_hist_capital.ipynb`: contains code to use the historical and estimated investment-to-GDP ratios to create current-PPP investment values. These are used to replicate the initial-year capital stock estimation (country-by-country) as described in Inklaar, Woltjer, and Albarrán (2019). Also, the investment values are used in conjunction with GEG-15 and LitPop data sources to fill in missing values for the latter parts of the historical capital stock data. The end product is a filled (1950-2020) capital stock data for all relevant countries.
5. `ypk5_projected_yp.ipynb`: contains code to clean up GDP, GDPpc, and population for the future timeline, with some basic extrapolation conducted for countries with missing projections.
6. `ypk6_projected_capital.ipynb`: generates projections of capital stocks based on the Dellink et al. (2017) methodology.

For running these files, note that they have to be **run consecutively** (i.e., from `ypk1~` to `ypk7~`). Each notebook file contains basic descriptions on what each step does; in all cases, the cells must be run consecutively from top to bottom.

## 2. Basic description of key variables

We describe below some key variables produced by the above process. Note that our naming conventions largely follow Penn World Table 10.0.
- `cgdpo_19`: Current PPP (purchasing power parity) GDP in millions of 2017 and 2019 USD 
- `cgdpo_pc_19`: Current PPP GDP per capita in ones of 2017 and 2019 USD
- `rgdpna_19`: (National account-based) GDP in millions of constant 2019 PPP USD
- `rgdpna_pc_19`: (National account-based) GDP per capita in ones of constant 2019 PPP USD
- `cn_19`: Current PPP capital stock in millions of 2019 USD
- `rnna_19`: Capital stock in millions of constant PPP 2019 USD
- `pop`: Population in millions of people
- `k_movable_ratio`: ratio movable capital out of total physical capital (values in <img src="https://render.githubusercontent.com/render/math?math=[0, 1]">)
- `iy_ratio`: Investment-to-GDP ratio
- `delta`: Physical capital depreciation rate

Note that for GDP, GDP per capita, and capital stock variables, there are also versions with `_17` at the end instead of `_19`. For current PPP variables, this means using 2017 USD; for constant PPP variables, this means using constant 2017 PPP USD (i.e., constant PPP of 2017 and 2017 USD).

## 3. Output storage

We import the SLIIDERS `settings.py` as `sset`, which can be done as follows:
```
from sliiders import as settings as sset
```
For the aggregate long-panel format historical and future timeline variables, you may refer to the following:
1. Historical: `sset.DIR_YPK_FINAL / "gdp_gdppc_pop_capital_1950_2020.parquet"`
2. Future: `sset.DIR_YPK_FINAL / "gdp_gdppc_pop_capital_proj_2010_2100.parquet"`

where the metadata (e.g., units and sources) are also attached to the respective files.

## 4. Regression results for imputing missing historical investment-to-GDP ratios

We elaborate on the regression involving investment-to-GDP ratios mentioned in Section A3.2 in the notebook `ypk4_demo_ratios_historical_reg.ipynb`. The said notebook also contains information on how to derive each variable involved. We present the results below, where the dependent variable is investment-to-GDP ratio (denoted as <img src="https://render.githubusercontent.com/render/math?math=\frac{I}{Y}"> in the notebook).

| Variables | (1) | (2) | (3) | (4) |
| ------ | :------: | :------: | :------: | :------: |
| <img src="https://render.githubusercontent.com/render/math?math=\hat{g}"> | 0.405 <br/> (0.161) | 0.346<br/>(0.076) | 0.502<br/>(0.201) | 0.480<br/>(0.129) |
| <img src="https://render.githubusercontent.com/render/math?math=\hat{g}^2"> | 0.864<br/>(0.742) | 0.515<br/>(0.611) | 0.506<br/>(0.879) | 0.493<br/>(0.915) |
| <img src="https://render.githubusercontent.com/render/math?math=\hat{yhr}"> | -0.021<br/>(0.052) | -0.027<br/>(0.052) | 0.076<br/>(0.022) | 0.108<br/>(0.016) |
| <img src="https://render.githubusercontent.com/render/math?math=\hat{yhr}^2"> | 0.004<br/>(0.007) | 0.003<br/>(0.006) | -0.011<br/>(0.005) | -0.015<br/>(0.005) |
| <img src="https://render.githubusercontent.com/render/math?math=D_1"> | 0.184<br/>(0.186) |  | 0.348<br/>(0.190) |  |
| <img src="https://render.githubusercontent.com/render/math?math=D_2"> | -0.008<br/>(0.035) |  | -0.038<br/>(0.030) |  |
| <img src="https://render.githubusercontent.com/render/math?math=D_3"> | -0.000<br/>(0.002) |  | 0.001<br/>(0.001) |  |
| <img src="https://render.githubusercontent.com/render/math?math=D_1\times\hat{g}"> | 3.988<br/>(3.149) |  | 1.784<br/>(3.945) |  |
| <img src="https://render.githubusercontent.com/render/math?math=D_2\times\hat{g}"> | -0.797<br/>(0.570) |  | -0.465<br/>(0.597) |  |
| <img src="https://render.githubusercontent.com/render/math?math=D_3\times\hat{g}"> | 0.040<br/>(0.028) |  | 0.026<br/>(0.026) |  |
|  <img src="https://render.githubusercontent.com/render/math?math=N">  | 11145 | 11145 | 11145 | 11145 |
| Country fixed effects | Yes | Yes | No | No |
| Adjusted  <img src="https://render.githubusercontent.com/render/math?math=R^2">  | 0.325 | 0.315 | 0.068 | 0.054 |
| AIC | -12712 | -12557 | -9317 | -9157 |
| BIC | -11153 | -11042 | -9236 | -9120 |
