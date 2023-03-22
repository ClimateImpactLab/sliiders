# Workflow for organizing and projecting GDP (`Y`), population (`P`), capital stock (`K`), and related variables for historical (2000-2020) and future (2021-2100) timelines

**This version: last updated on November 25, 2022**

## 1. Overview

This directory contains the data acquistion, clean-up, and projection notebook files to organize and project variables including GDP, GDP per capita (GDPpc), population, and capital stock for both historical (2000-2020) and projected (2021-2100) timelines. Many of the data sources used to generate historical and future panels have missing data, and therefore efforts were made to impute these missing data through either some form of extrapolation or other established methods. Also, efforts were made to keep the PPP and USD units consistent (e.g., constant 2019 PPP USD) across different sources having different vintages of PPP and USD units.

Below is a quick summary of what each file seeks to accomplish (where the header `ypk` stands for "GDP, population, and capital stock").
1. `historical-income-pop`: Assimilates multiple data sources for historical GDP, GDP per capital and population to create a balanced panel for all countries reflected in SLIIDERS from 2000 to 2020.
2. `historical-capital`: Assimilates multiple data sources for capital stock and non-financial wealth per capita to create a balanced panel of capital stock estimates for all countries reflected in SLIIDERS from 2000 to 2020.
3. `projected-income-pop.ipynb`: contains code to clean up GDP, GDPpc, and population for the future timeline, with some basic extrapolation conducted for countries with missing projections.
6. `projected-capital.ipynb`: generates projections of capital stocks based on the Dellink et al. (2017) methodology.

## 2. Basic description of key variables

We describe below some key variables produced by the above process. Note that our naming conventions largely follow Penn World Table 10.0.
- `cgdpo_19`: Current PPP (purchasing power parity) GDP in millions of 2017 and 2019 USD 
- `cgdpo_pc_19`: Current PPP GDP per capita in ones of 2017 and 2019 USD
- `rgdpna_19`: (National account-based) GDP in millions of constant 2019 PPP USD
- `rgdpna_pc_19`: (National account-based) GDP per capita in ones of constant 2019 PPP USD
- `cn_19`: Current PPP capital stock in millions of 2019 USD
- `rnna_19`: Capital stock in millions of constant PPP 2019 USD
- `population`: Population in millions of people
- `iy_ratio`: Investment-to-GDP ratio
- `delta`: Physical capital depreciation rate

Note that for GDP, GDP per capita, and capital stock variables, there are also versions with `_17` at the end instead of `_19`. For current PPP variables, this means using 2017 USD; for constant PPP variables, this means using constant 2017 PPP USD (i.e., constant PPP of 2017 and 2017 USD).