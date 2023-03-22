"""
various functions used for the country-level information workflow in
`notebooks/data-processing/1-country-level-temporal-trends`
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from sliiders.settings import FRA_OVERSEAS_DEPT, HISTORICAL_YEARS

# group y, group x, year of independence if any, and whether it's disputed or not
GROUPS_X_Y = [
    # GBR cases, current territories and former territories (now independent)
    ["AIA", "GBR", np.nan, 0],
    ["BMU", "GBR", np.nan, 0],
    ["VGB", "GBR", np.nan, 0],
    ["CYM", "GBR", np.nan, 0],
    ["FLK", "GBR", np.nan, 1],  # disputed by Argentina
    ["IMN", "GBR", np.nan, 0],
    ["GGY", "GBR", np.nan, 0],
    ["GIB", "GBR", np.nan, 0],
    ["JEY", "GBR", np.nan, 0],
    ["MSR", "GBR", np.nan, 0],
    ["PCN", "GBR", np.nan, 0],
    ["SHN", "GBR", np.nan, 0],
    ["TCA", "GBR", np.nan, 0],
    ["ATG", "GBR", 1981, 0],
    ["BHS", "GBR", 1973, 0],
    ["BRB", "GBR", 1966, 0],
    ["CYP", "GBR", 1960, 0],
    ["DMA", "GBR", 1979, 0],
    ["FJI", "GBR", 1970, 0],
    ["GRD", "GBR", 1974, 0],
    ["JAM", "GBR", 1962, 0],
    ["KIR", "GBR", 1979, 0],
    ["MDV", "GBR", 1966, 0],
    ["MLT", "GBR", 1964, 0],
    ["MUS", "GBR", 1968, 0],
    ["SYC", "GBR", 1976, 0],
    ["SLB", "GBR", 1978, 0],
    ["KNA", "GBR", 1983, 0],
    ["LCA", "GBR", 1979, 0],
    ["VCT", "GBR", 1979, 0],
    ["TTO", "GBR", 1962, 0],
    ["TON", "GBR", 1970, 0],
    ["TUV", "GBR", 1978, 0],
    ["VUT", "GBR", 1980, 0],  # also formerly managed by France
    # FRA cases, current territories and former territories (now independent)
    ["GLP", "FRA", np.nan, 0],
    ["GUF", "FRA", np.nan, 0],
    ["MTQ", "FRA", np.nan, 0],
    ["MYT", "FRA", np.nan, 0],
    ["SPM", "FRA", np.nan, 0],
    ["REU", "FRA", np.nan, 0],
    ["MAF", "FRA", np.nan, 0],
    ["BLM", "FRA", np.nan, 0],
    ["PYF", "FRA", np.nan, 0],
    ["WLF", "FRA", np.nan, 0],
    ["NCL", "FRA", np.nan, 0],
    ["AND", "FRA", 1814, 0],
    ["COM", "FRA", 1975, 0],
    ["MCO", "FRA", 1918, 0],
    # NLD cases, current territory
    ["SXM", "NLD", np.nan, 0],
    ["ABW", "NLD", np.nan, 0],
    ["BES", "NLD", np.nan, 0],
    ["CUW", "NLD", np.nan, 0],
    # DNK cases, current territory
    ["FRO", "DNK", np.nan, 0],
    ["GRL", "DNK", np.nan, 0],
    # NZL cases, current territories and former territories (now independent)
    ["COK", "NZL", np.nan, 0],
    ["NIU", "NZL", np.nan, 0],
    ["TKL", "NZL", np.nan, 0],
    ["WSM", "NZL", 1962, 0],
    # USA cases, current territories and former territories (now independent)
    ["ASM", "USA", np.nan, 0],
    ["GUM", "USA", np.nan, 0],
    ["MNP", "USA", np.nan, 0],
    ["PRI", "USA", np.nan, 0],
    ["VIR", "USA", np.nan, 0],
    ["MHL", "USA", 1986, 0],  # under "free association"
    ["FSM", "USA", 1986, 0],  # under "free association"
    ["PLW", "USA", 1981, 0],  # under "free association"
    # FIN cases
    ["ALA", "FIN", np.nan, 0],
    # AUS cases, current territories and former territories (now independent)
    ["CCK", "AUS", np.nan, 0],
    ["CXR", "AUS", np.nan, 0],
    ["NFK", "AUS", np.nan, 0],
    ["NRU", "AUS", 1968, 0],  # technically also by GBR and NZL
    # Individual cases
    ["SSD", "SDN", 2011, 1],  # South Sudan and Sudan - regional conflicts
    ["ZNC", "CYP", 1983, 1],  # North Cyprus and Cyprus - disputed
    ["ESH", "MAR", np.nan, 1],  # Western Sahara with Morocco - disputed
    ["VAT", "ITA", 1929, 0],  # Vatican and Italy
    ["SMR", "ITA", 300, 0],  # San Marino and Italy
    ["LIE", "AUT", 1866, 0],  # Liech. was always independent but close ties to Austria
    ["XKO", "SRB", 2008, 1],  # Kosovo declared indep. in 2008
]

GROUPS_X_Y = pd.DataFrame(
    GROUPS_X_Y, columns=["ccode", "x_ccode", "indep_yr", "disputed"]
).set_index("ccode")


def yearly_growth(df, header="v_"):
    """Turn a horizontal (wide-panel) DataFrame with annual values (whose variable names
    start with `header` and have year designations) into containing annual growth rates.
    The initial-year growth rates are set to be 0.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing annual data, where for some 4-digit year `y`, the variable
        names are as follows: `{header}{y}`; should be in wide-panel format (i.e.,
        one row for each country)
    header : str
        header for the annual variable names

    Returns
    -------
    rtn_df : pandas DataFrame
        DataFrame containing annual growth rates

    """

    yrs = [int(v[-4:]) for v in df.columns if header in v]
    others = [v for v in df.columns if header not in v]
    yrs.sort()
    v_ = [header + str(yr) for yr in yrs]
    rtn_df = df[others + v_].copy()

    for i, v in enumerate(v_):
        if i == 0:
            rtn_df[v] = 0
            continue
        v_prev = v_[(i - 1)]
        rtn_df[v] = np.log(df[v]) - np.log(df[v_prev])

    return rtn_df


def helper_extrap_using_closest(
    prob_ctry, after, avail_yr, end_yr, tgt_df, sse_good_df, wgt_power, hdr="v_"
):
    """Helper function for the function `extrap_using_closest`, which is used in
    detecting similar-trajectory countries and using the said trajectories to impute
    the missing values of another country

    Parameters
    ----------
    prob_ctry : str
        country code needing extrapolation (projection)
    after : boolean
        for projecting forward in time, set as True; for projecting backwards in time,
        set as False
    avail_yr : int
        latest available year if projecting forward (`after`=True), and earliest
        available year if projecting backwards (`after`=False)
    end_yr : int
        latest year to project until if projecting forward (`after`=True), and earliest
        year to project until if projecting backwards (`after`=False)
    tgt_df : pandas DataFrame
        DataFrame to calculate the extrapolated projections from; should be in
        wide-panel format, containing
    sse_good_df : pandas DataFrame
        DataFrame containing the sum of squared errors (of known growth rates) with
        respect to the countries that have the closest trajectories to `prob_ctry`
    wgt_power : float
        by what exponent the weights (for creating extrapolations) should be applied at
    hdr : str
        header for the annual variables in `tgt_df`

    Returns
    -------
    extrapolated : numpy array
        containing extrapolated information, using similar-trajectory countries

    """

    if prob_ctry is None:
        prob_ctry = tgt_df.index.unique()[0]

    if after:
        v_s = [hdr + str(x) for x in range(avail_yr, end_yr + 1)]
    else:
        v_s = [hdr + str(x) for x in range(end_yr, avail_yr + 1)]

    gr_df_base_avail = sse_good_df[v_s + ["sse", "sse_rank"]].copy()
    avail_v = hdr + str(avail_yr)
    gr_df_base_avail[v_s] = gr_df_base_avail[v_s].div(gr_df_base_avail[avail_v], axis=0)

    # if there's a PERFERCTLY matching set of growth rates, then just take that
    # country (or those countries') growth rates
    if (gr_df_base_avail.sse == 0).any():
        idx = gr_df_base_avail.loc[gr_df_base_avail.sse == 0, :].index.unique()
        if len(idx) == 1:
            growth_rates = gr_df_base_avail.loc[idx[0], v_s]
        else:
            growth_rates = gr_df_base_avail.loc[idx, v_s].values.mean(axis=0)
    else:
        gr_df_base_avail["wgt_vals"] = (1 / gr_df_base_avail["sse"]).values ** wgt_power
        denom_values = np.sum(gr_df_base_avail["wgt_vals"].values)
        growth_rates = (
            np.sum(
                gr_df_base_avail[v_s].mul(gr_df_base_avail["wgt_vals"], axis=0), axis=0
            )
            / denom_values
        )

    avail_val = tgt_df.loc[prob_ctry, avail_v]
    extrapolated = np.array(growth_rates) * avail_val

    if after:
        extrapolated = extrapolated[1:]
    else:
        extrapolated = extrapolated[0:-1]

    return extrapolated


def extrap_using_closest(
    prob_lst,
    orig_df,
    n_det=5,
    wgt_power=1,
    begin_end=[1950, 2019],
    exclude_these=["MAF", "WLF", "ESH"],
    merge_orig=True,
    header="v_",
    fill_name="msng_fill",
    ctry_col="ccode",
):
    """Uses the "closest" countries (in terms of existing data's trajectory with
    respect to a given year, with the metric for determining "closeness" as the
    sum of squared errors [SSE]) whose data are non-missing to figure out the trajectory
    of "problematic" (i.e., with missing data) countries.

    Parameters
    ----------
    prob_lst : array-like
        List of countries whose data are partially missing
    n_det : int
        Number of "similar countries" to use
    wgt_power : float
        Whether higher weights should be given to those that are "closer" or not
        (higher positive number --> greater weights)
    begin_end : array-like of int
        The earliest and the last year that need extrapolation
    exclude_these : list of str
        list of countries to exclude for using as "closest" countries, or to extrapolate
        in general; for instance, if a country has only one year's worth of
        data (like MAF, WLF, ESH's GDP values) then it would be a good reason to
        exclude these countries.
    merge_orig : boolean
        whether the information from non-problematic countries should be merged
        when returning the data back
    header : str
        for the variables (e.g., "v_" for "v_1950" indicating 1950 values)
    fill_name : str
        column name for the missing value "fill" information (which years were filled,
        using which countries)
    ctry_col : str
        column name for the country-code variable, default being "ccode"

    Returns
    -------
    df_rtn : pandas DataFrame
        DataFrame containing extrapolated and existing information countries with
        missing values. If merge_orig is True, then it would also contain the countries
        without any extrapolated (i.e., the "non-problematic")

    """
    # indicing for operations below
    ctry_msg = "Needs have the country-code column / index `{}` in the dataset"
    ctry_msg = ctry_msg.format(ctry_col)
    assert (ctry_col in orig_df.index.names) or (ctry_col in orig_df.columns), ctry_msg

    if ctry_col not in orig_df.index.names and ctry_col in orig_df.columns:
        df_idxed = orig_df.set_index([ctry_col])
    else:
        df_idxed = pd.DataFrame(orig_df)

    # sorting the problematic (with missing-value) countries for consistency
    prob, exclude_these = list(np.sort(prob_lst)), list(exclude_these)

    # variable names and getting the dataframe of "good-to-go" country codes
    v_ = np.sort(
        [
            x
            for x in orig_df.columns
            if (header in x)
            and (int(x[-4:]) <= begin_end[1])
            and (int(x[-4:]) >= begin_end[0])
        ]
    )

    # good_ctries are only those that are absolutely filled
    # excluding those that should be excluded
    good_ctries = df_idxed[(~df_idxed[v_].isnull().any(axis=1))].index.unique()
    good_ctries = np.setdiff1d(good_ctries, prob + exclude_these)
    good_df = df_idxed.loc[good_ctries, :]
    good_gr = yearly_growth(good_df, header)

    # running for each of the problematic countries
    df_collection = []
    for i in tqdm(prob):
        # there could be missing values in between known yrs, so interpolate
        tgt_df = df_idxed.loc[[i], :].copy()
        row_vals = tgt_df.loc[i, v_].copy()
        row_vals = np.where(row_vals < 0, np.nan, row_vals)
        valid_where = np.where(~pd.isnull(row_vals))[0]
        mn_valid_loc, mx_valid_loc = min(valid_where), max(valid_where)
        v_valid = v_[mn_valid_loc : mx_valid_loc + 1]
        if len(valid_where) != (mx_valid_loc + 1 - mn_valid_loc):
            log_interp_vals = np.interp(
                range(mn_valid_loc, mx_valid_loc + 1),
                valid_where,
                np.log(
                    tgt_df.loc[i, np.array(v_)[valid_where]].values.astype("float64")
                ),
            )
            tgt_df[v_valid] = np.exp(log_interp_vals)
        row_valid = tgt_df.loc[i, v_valid]

        # as yearly growth rates, with missing values filled as 0
        tgt_gr = yearly_growth(tgt_df).fillna(0)

        # detecting which is the valid (or non-missing) growth rates
        gr_row_valid = tgt_gr.loc[i, v_valid].values

        # subtract the problematic growth rates from good-to-go growth rates,
        # and calculate the sum of squared errors to detect which is the closest
        sse_df = good_gr.copy()
        sse_df["sse"] = (sse_df[v_valid].sub(gr_row_valid, axis=1) ** 2).sum(axis=1)
        sse_df.sort_values(["sse"], inplace=True)
        sse_df["sse_rank"] = range(0, sse_df.shape[0])

        # top n closest in terms of trajectory
        necess_sse_df = sse_df[sse_df.sse_rank < n_det][["sse", "sse_rank"]]
        necess_sse_df = necess_sse_df.merge(
            good_df[v_],
            how="left",
            left_index=True,
            right_index=True,
        )

        # if need to project backwards in time
        rtn_row, past_fill, fut_fill = np.array(row_valid), None, None
        if v_valid[0] != v_[0]:
            avail_yr, earl_yr = int(v_valid[0][-4:]), begin_end[0]
            past_vals = helper_extrap_using_closest(
                i,
                False,
                avail_yr,
                earl_yr,
                tgt_df,
                necess_sse_df,
                wgt_power,
                hdr=header,
            )
            rtn_row = np.hstack([past_vals, rtn_row])
            past_fill = "{}-{}".format(earl_yr, avail_yr - 1)

        # if need to project forward in time
        if v_valid[-1] != v_[-1]:
            avail_yr, late_yr = int(v_valid[-1][-4:]), begin_end[-1]
            fut_vals = helper_extrap_using_closest(
                i,
                True,
                avail_yr,
                late_yr,
                tgt_df,
                necess_sse_df,
                wgt_power,
                hdr=header,
            )
            rtn_row = np.hstack([rtn_row, fut_vals])
            fut_fill = "{}-{}".format(avail_yr + 1, late_yr)

        # extrapolation information as "fill_info"
        used_ccodes, fill_info = ",".join(list(necess_sse_df.index.unique())), "-"
        if (past_fill is not None) and (fut_fill is not None):
            fill_info = past_fill + "," + fut_fill + ":" + used_ccodes
        elif past_fill is not None:
            fill_info = past_fill + ":" + used_ccodes
        elif fut_fill is not None:
            fill_info = fut_fill + ":" + used_ccodes

        tgt_df_extrap = tgt_df.copy()
        tgt_df_extrap[v_] = rtn_row
        tgt_df_extrap[fill_name] = fill_info
        df_collection.append(tgt_df_extrap)

    rtn_df = pd.concat(df_collection, axis=0)

    if merge_orig:
        unaltered = np.setdiff1d(
            orig_df.index.get_level_values("ccode").unique(),
            rtn_df.index.get_level_values("ccode").unique(),
        )
        orig_slice = orig_df.loc[unaltered, :].copy()
        orig_slice[fill_name] = "-"
        rtn_df = pd.concat([rtn_df, orig_slice], axis=0).sort_index()

    return rtn_df


def organize_hor_to_ver(
    df,
    main_cat,
    sub_cats,
    new_vname,
    hdr="v_",
    yrs=range(1950, 2020),
    timename="year",
):
    """Use for organizing wide-format panel data ("horizontal") to long-format panel
    data ("vertical"). Serves as a wrapper for the function `pandas.wide_to_long`, but
    repurposed for our workflow (mostly in terms of renaming the variables)

    Note: For every row of the "input" dataframe `df`, we assume that there is at most
        one combination of the categories in `catnames`; for instance, if `catname`
        is equal to ["ccode", "ssp", "iam"], we expect that there should be at most
        one account for each countrycode-SXSPIAM combination.

    Parameters
    ----------
    df : pandas DataFrame
        dataframe containing information, that is in a "wide-format"
    main_cat : str
        name of the main category we want organize by (e.g., "ccode" for country-codes)
    sub_cats : array-like of str or None
        list or array containing the names of additional categories we want to organize
        by (e.g., ["ssp", "iam"]); if equals to None, then is understood as an empty
        array
    new_vname : str
        name of the variable to be newly assigned
    hdr : str
        current "header" of the columns in wide-format (e.g., "v_" would
        mean that v_1950, v_1951, v_1952,... are the column names)
    yrs : array-like of int
        years to consider
    timename : str
        what to call the part of the index

    Returns
    -------
    long_df : pandas DataFrame
        containing the data in long-panel (or vertical) format

    """

    if sub_cats is None:
        sub_cats = []
    cats = np.hstack([[main_cat], sub_cats])
    reorder = np.hstack([[main_cat, timename], sub_cats])

    # resetting the index to be compliant with `pandas.wide_to_long`
    df_reind = df.reset_index()
    if df.index.names is None:
        df_reind.drop(["index"], axis=1, inplace=True)
    v_s = np.intersect1d([hdr + str(x) for x in yrs], df_reind.columns)
    df_reind = df_reind[np.hstack([[x for x in df_reind.columns if hdr not in x], v_s])]

    long_df = pd.wide_to_long(df_reind, hdr, cats, timename).reset_index()
    long_df.set_index(list(reorder), inplace=True)
    long_df.sort_index(axis=0, inplace=True)
    long_df.rename(columns={hdr: new_vname}, inplace=True)

    return long_df


def organize_ver_to_hor(
    df,
    varname,
    timename,
    ccodename,
    total_yrs=range(1950, 2020),
    impose_total=False,
):
    """For organizing a "vertical dataframe" (or long-panel form data) to "horizontal
    dataframe" (or wide-panel format data). Mainly works as a wrapper for pandas.pivot
    but repurposed for our purposes (including re-naming the columns)

    Parameters
    ----------
    df : pandas DataFrame
        dataframe containing information
    varname : str
        column name of the variable that we want the information about
    timename : str
        column name of the variable that indicates time or years
    ccodename : str
        column name of the variable indicating country-codes
    total_yrs : array-like
        range of the years that we want information about
    impose_total : boolean
        if True, all years in the `total_yrs` array are represented (even if missing
        entirely from the dataset); if False, then only necessary columns are reported
        (with at least some non-missing values)

    Returns
    -------
    df_rtn : pandas.DataFrame
        :py:class:`pandas.DataFrame` containing information specifically about the
        variable indicated by "varname", in a wide-panel format.

    """

    # necessary to reset the index to pass to pandas.pivot
    df_rtn = df.reset_index()
    names = np.array([varname, timename, ccodename])
    assert len(np.setdiff1d(names, df_rtn.columns)) == 0, "necessary columns missing."

    df_rtn.sort_values([ccodename, timename], inplace=True)
    df_rtn = df_rtn.pivot(index=[ccodename], columns=timename, values=varname)
    df_rtn.columns.name = None
    df_rtn.columns = ["v_" + str(x) for x in df_rtn.columns]
    total_v = ["v_" + str(x) for x in total_yrs]

    df_rtn = df_rtn[[v for v in total_v if v in df_rtn.columns]]
    if impose_total:
        leftovers = np.setdiff1d(total_v, df_rtn.columns)
        df_rtn[leftovers] = np.nan
        df_rtn = df_rtn[total_v]

    return df_rtn


def ppp_conversion_specific_year(
    yr,
    pwt_path,
    to=True,
    extrap_sim=True,
    fill_msng_ctries={},
    pwtvar="pl_gdpo",
):
    """Given a specified year (`yr`), creates a table of PPP conversion factors either
    to that year (to=True) or from that year (to=False). The range of years to
    convert from or to that year is fixed to 1950-2019, which is all the available
    years from Penn World Tables. We can specify the `pwtvar` variable to change
    whether we would like to use a different price level variable (e.g., `pl_n` for
    capital, `pl_gdpo` for output-side GDP).

    Parameters
    ----------
    yr : int
        specific year that we will calculate PPP conversion rates to or from
    to : boolean
        boolean for indicating if the target year is the year that one should calculate
        the years from (`to`=False) or to (`to`=True). e.g., if yr=2019 and to=True,
        this function will calculate the conversion rates from 2019 PPP to PPP of any
        year between 1950 and 2019 (but NOT change the base dollar terms)
    extrap_sim : boolean
        boolean for whether to extrapolate or not, for countries having partial
        information (i.e., not all conversion rates for 1950-2019).
    fill_msng_ctries : None or dict
        indicates if we should fill in for those countries that are either entirely
        missing from both WDI and PWT datasets or has too much unreliable / missing
        data
    pwtvar : str
        the name of the price level variable to calculate PPP conversion rates from
        for PWT.

    Outputs
    -------
    pl_ver : pandas DataFrame
        containing countrycode, year, and conversion rates (PPP); information organized
        in a vertical (long-panel) format, with extrapolation done for the specified
        variable when there are missing variables if `extrap_sim` is equal to True.

    """

    print("Fetching information from PWT...")
    # reading in the necessary PWT dataframe
    pwt = (
        pd.read_excel(pwt_path)
        .rename(columns={"countrycode": "ccode"})
        .set_index(["ccode", "year"])
    )
    pwt_years = pwt.index.get_level_values("year").unique()
    yr_range = range(pwt_years.min(), pwt_years.max() + 1)

    v_ = ["v_" + str(v) for v in yr_range]
    pl = organize_ver_to_hor(pwt, pwtvar, "year", "ccode", yr_range)
    pl_ccode = pl.index.get_level_values("ccode").unique()

    # replace with pl_gdpo information if specific pl values for a country are
    # missing entirely
    if pwtvar != "pl_gdpo":
        pl_gdpo = organize_ver_to_hor(pwt, "pl_gdpo", "year", "ccode", yr_range)
        for c in pl_ccode:
            row = pl.loc[c, v_].values
            if sum(pd.isnull(row)) == len(row):
                pl.loc[c, v_] = pl_gdpo.loc[c, v_].values

    if extrap_sim:
        prob = (
            pl.loc[pl[v_].isnull().any(axis=1), :]
            .index.get_level_values("ccode")
            .unique()
        )
        pl = extrap_using_closest(prob, pl, exclude_these=[])

        pl_ver = organize_hor_to_ver(pl, "ccode", None, pwtvar, yrs=yr_range)
        fill_name = "{}_fill".format(pwtvar)
        pl_ver.rename(columns={"msng_fill": fill_name}, inplace=True)
        pl_ver[fill_name] = [v.split(":")[-1] for v in pl_ver[fill_name]]

        # making sure that the fill-information is "-" if information was not missing
        pl_ver = pl_ver.merge(
            pwt[[pwtvar]].rename(columns={pwtvar: "temp"}),
            left_index=True,
            right_index=True,
            how="left",
        )
        pl_ver.loc[~pd.isnull(pl_ver["temp"]), fill_name] = "-"
        pl_ver.drop(["temp"], axis=1, inplace=True)

    else:
        pl_ver = organize_hor_to_ver(pl, "ccode", None, pwtvar, yrs=yr_range)

    # taking care of the case of Bermuda, since it is sometimes suffering
    # from negative price levels
    if (pwtvar == "pl_gdpo") and ("BMU" in pl_ccode):
        pl_ccode = np.setdiff1d(pl_ccode, ["BMU"])
        bmu_copy = pl_ver.loc[("GBR", slice(None)), :].reset_index().copy()
        bmu_copy[fill_name] = "copy_from_GBR"
        bmu_copy["ccode"] = "BMU"
        bmu_copy.set_index(["ccode", "year"], inplace=True)
        pl_ver = pd.concat([pl_ver.loc[(pl_ccode, slice(None)), :], bmu_copy], axis=0)

    # merging the "base" price level, which is that of the US
    pwt_ppp = pl_ver.merge(
        (
            pl_ver.loc[("USA", slice(None)), [pwtvar]]
            .reset_index()
            .drop(["ccode"], axis=1)
            .set_index(["year"])
            .rename(columns={pwtvar: "base"})
        ),
        left_index=True,
        right_index=True,
        how="left",
    )

    # note that according to Feenstra et al. (2015), PPP / XR = pl / pl_base
    # with the "base" again being the United States; `ppp` below is PPP / XR
    pwt_ppp["ppp"] = pwt_ppp[pwtvar] / pwt_ppp["base"]

    # multiplying `ppp` can be understood as turning PPP-adjusted value of a certain
    # year to nominal value; turning base-year-a PPP values to base-year-b PPP values
    # therefore requires multiplying `ppp`(a) / `ppp`(b)
    tgtyr_ppp = f"ppp_{yr}"
    pwt_ppp = pwt_ppp.merge(
        (
            pwt_ppp.loc[(slice(None), yr), ["ppp"]]
            .rename(columns={"ppp": tgtyr_ppp})
            .reset_index()
            .drop(["year"], axis=1)
            .set_index(["ccode"])
        ),
        left_index=True,
        right_index=True,
        how="left",
    )

    # conversion rates
    if to:
        pwt_ppp["conv"] = pwt_ppp["ppp"] / pwt_ppp[tgtyr_ppp]
    else:
        pwt_ppp["conv"] = pwt_ppp[tgtyr_ppp] / pwt_ppp["ppp"]
    pwt_ppp.drop([tgtyr_ppp], axis=1, inplace=True)

    to_keep = []
    for i in pwt_ppp.index.get_level_values("ccode").unique():
        i_case = pwt_ppp.loc[i, "conv"].isnull().all()
        if not i_case:
            to_keep.append(i)

    pwt_ppp = pwt_ppp.loc[(to_keep, slice(None)), :].sort_index()

    # filling in the missing countries with known values
    if fill_msng_ctries is not None:
        print("Filling in the missing countries...")
        pwt_ppp["conv_fill"] = "refer_to_other_cols"

        for key, replace_these in fill_msng_ctries.items():
            conv_fill_key = "copy_from_{}".format(key)
            no_replace_these_ccodes = np.setdiff1d(
                pwt_ppp.index.get_level_values("ccode").unique(), replace_these
            )
            pwt_ppp = pwt_ppp.loc[(no_replace_these_ccodes, slice(None)), :]

            copies = [pwt_ppp]
            for rep_ctry in replace_these:
                ctry_copied = pwt_ppp.loc[(key, slice(None)), :].copy().reset_index()
                ctry_copied["ccode"] = rep_ctry
                ctry_copied["conv_fill"] = conv_fill_key
                ctry_copied.set_index(["ccode", "year"], inplace=True)
                copies.append(ctry_copied)
            pwt_ppp = pd.concat(copies, axis=0)

    pwt_ppp = pwt_ppp[["conv", "conv_fill", fill_name]].copy()
    pwt_ppp.sort_index(inplace=True)
    print("...done")

    return pwt_ppp


def smooth_fill(
    da1_in,
    da2_in,
    fill_all_null=True,
    time_dim="time",
    other_dim="storm",
):
    """Fill values from 2D dataarray `da1_in` with values from 2D dataarray
    `da2_in`.

    For instance, one may use this with storm datasets. If filling the beginning or end
    of a storm, pin the `da2_in` value to the `da1_in` value at the first/last point of
    overlap and then use the `da2_in` values only to estimate the "change" in values
    over time, using a ratio of predicted value in the desired time to the reference
    time. This can also be used when, for example, `da1_in` refers to RMW and `da2_in`
    refers to ROCI. In this case, you want to define ``fill_all_null=False`` to avoid
    filling RMW with ROCI when no RMW values are available but some ROCI values are
    available.

    Parameters
    ----------
    da1_in, da2_in : xarray.DataArray
        DataArrays indexed by other dimension (defined by `other_dim`) and time
        dimension (defined by `time_dim`)
    fill_all_null : bool, optional
        If True, fills even when there are no known (or non-NA) values in `da1_in`
    time_dim : str, optional
        variable name to indicate the time dimension, default set to be "time"
    other_dim : str, optional
        variable name to indicate the other dimension, default set to be "storm" but
        can also indicate country or region names, for instance

    Returns
    -------
    :class:`xarray.DataArray`
        Same as ``da1`` but with NaN's filled by the described algorithm.

    Raises
    ------
    AssertionError :
        If there are "interior" NaN's in either dataset, i.e. if any storm has a NaN
        after the first non-NaN but before the last non-NaN. These should have
        previously been interpolated.

    Examples
    --------
    >>> import xarray as xr
    >>> da1 = xr.DataArray(
    ...     np.array(
    ...         [
    ...             [np.nan, 1, 2, 3],
    ...             [np.nan, np.nan, 4, 5],
    ...             [6, 7, np.nan, np.nan],
    ...             [8, 9, 10, np.nan],
    ...             [11, 12, 13, 14],
    ...             [np.nan, np.nan, np.nan, np.nan],
    ...         ]
    ...     ),
    ...     coords = {"storm": range(6), "time": range(4)},
    ...     dims = ["storm", "time"]
    ... )
    >>> da2 = xr.DataArray(
    ...     np.array(
    ...         [
    ...             [15, 16, 17, 18],
    ...             [19, 20, 21, 22],
    ...             [23, 24, 25, 26],
    ...             [27, 28, 29, 30],
    ...             [31, 32, 33, 34],
    ...             [35, 36, 37, 38],
    ...         ]
    ...     ),
    ...     coords = {"storm": range(6), "time": range(4)},
    ...     dims = ["storm", "time"]
    ... )
    >>> smooth_fill(da1, da2)
    <xarray.DataArray (storm: 6, time: 4)>
    array([[ 0.9375    ,  1.        ,  2.        ,  3.        ],
           [ 3.61904762,  3.80952381,  4.        ,  5.        ],
           [ 6.        ,  7.        ,  7.29166667,  7.58333333],
           [ 8.        ,  9.        , 10.        , 10.34482759],
           [11.        , 12.        , 13.        , 14.        ],
           [35.        , 36.        , 37.        , 38.        ]])
    Coordinates:
      * storm    (storm) int64 0 1 2 3 4 5
      * time     (time) int64 0 1 2 3
    """

    da1 = da1_in.copy()
    da2 = da2_in.copy()
    either_non_null = da1.notnull() | da2.notnull()

    da1 = da1.interpolate_na(dim=time_dim, use_coordinate=True)
    da2 = da2.interpolate_na(dim=time_dim, use_coordinate=True)
    for da in [da1, da2]:
        assert da.interpolate_na(dim=time_dim).notnull().sum() == da.notnull().sum()

    adjust = da1.reindex({other_dim: da2[other_dim]})
    first_valid_index = (adjust.notnull() & da2.notnull()).argmax(dim=time_dim)
    last_valid_index = (
        adjust.bfill(time_dim).isnull() | da2.bfill(time_dim).isnull()
    ).argmax(dim=time_dim) - 1

    all_null = adjust.isnull().all(dim=time_dim)
    if not fill_all_null:
        all_null *= False

    est_to_obs_rat_first = adjust.isel({time_dim: first_valid_index}) / da2.isel(
        {time_dim: first_valid_index}
    )

    est_val = da2.where(
        all_null | adjust.ffill(time_dim).notnull(),
        da2 * est_to_obs_rat_first,
    )

    est_to_obs_rat_last = adjust.isel({time_dim: last_valid_index}) / da2.isel(
        {time_dim: last_valid_index}
    )

    est_val = est_val.where(
        all_null | adjust.bfill(time_dim).notnull(),
        da2 * est_to_obs_rat_last,
    )

    # fill storms with da1 vals using the full da2 time series. For storms with some da1
    # vals, fill the tails using da2 scaled so that it matches at the first and last
    # points seen in both da1 and da2
    out = da1.fillna(est_val)

    # make sure we didn't add vals
    return out.where(either_non_null)


def smooth_fill_aux_source(
    spec_df,
    agg_df,
    use_col,
    msg_fill,
    msg_fill_gr,
    clean_col="rgdpna_pc",
    source_col="gdppc_source",
    years=HISTORICAL_YEARS,
):
    """Uses `smooth_fill` to fill in missing information for column `clean_col` in
    `spec_df` from `use_col` in `agg_df`. As with `smooth_fill`, copies unavailable
    information from `use_col` in `agg_df` when all information for a certain country
    is missing but applies growth-rate-based imputation when some information exists.

    Parameters
    ----------
    spec_df : pandas.DataFrame
        specified dataframe that contains `ccode` and `year` as multi-indices and
        the columns `clean_col` and `source_col`
    agg_df : pandas.DataFrame
        dataframe containing auxiliary information with `ccode` and `year` as multi-
        indices and the column `use_col`
    use_col : str
        column name in `agg_df` to be used to fill in missing information in column
        `clean_col` of `spec_df`
    msg_fill : str
        if `clean_col` has absolutely no information about a country (specified by
        index `ccode`) and we need to copy values from `use_col`, what the source
        message should be (e.g., if copying from World Bank WDI, put "WB")
    msg_fill_gr : str
        if `clean_Col` has some information about a country (`ccode`) and we use
        growth rates to fill in missing yearly information, what the source message
        should be (e.g., if original information is from PWT and using WB WDI growth
        rates to fill in missing yearly information, put "PWT + WB_gr")
    clean_col : str
        column in `spec_df` that contains primary information about information that
        one would like to fill in by `smooth_fill`
    source_col : str
        column in `spec_df` with details about where `clean_col` information is from;
        also used to fill in additional source information from `use_col`
    years : array-like of ints
        specifies the years that we need information for

    Returns
    -------
    pandas.DataFrame
        containing the columns `clean_col` and `source_col` with filled information
        from `agg_df` column `use_col`

    """

    join_df = spec_df.join(agg_df.loc[(slice(None), years), use_col], how="outer")
    all_ccode = join_df.index.get_level_values("ccode").unique()

    # detecting which to copy entirely from the new data source
    copy_all = []
    for i in all_ccode:
        cnd1 = pd.isnull(join_df.loc[i, clean_col]).all()
        cnd2 = pd.isnull(join_df.loc[i, use_col]).all()
        if cnd1 and (not cnd2):
            copy_all.append(i)
    to_clean = np.setdiff1d(all_ccode, copy_all)
    copied_df, to_clean = join_df.loc[copy_all, :], join_df.loc[to_clean, :]
    copied_df[clean_col] = copied_df[use_col].values
    copied_df.loc[~pd.isnull(copied_df[clean_col]), source_col] = msg_fill

    # using smooth-fill to fill in using growth rates
    fill = xr.Dataset.from_dataframe(to_clean[[use_col, clean_col]])
    fill = (
        smooth_fill(fill[clean_col], fill[use_col], other_dim="ccode", time_dim="year")
        .to_dataframe()
        .join(to_clean[[clean_col, source_col]].rename(columns={clean_col: "old"}))
    )
    fill.loc[
        pd.isnull(fill.old) & ~pd.isnull(fill[clean_col]), source_col
    ] = msg_fill_gr

    return pd.concat([fill, copied_df], axis=0).sort_index()[[clean_col, source_col]]


def fill_ratio_nominal_gdppc(
    y_ctries,
    x_ctries,
    agg_df,
    gdppc_df,
    y_nom_col="wb_gdppc_nom",
):
    """Calculates the nominal GDP per capita ratio between countries in `y_ctries` and
    those in `x_ctries` (in column "ratio"), then multiplies the `x_ctries` PPP GDP per
    capita (in column "x_rgdpna_pc") with `ratio` to impute PPP GDP per capita of
    `y_ctries` (in column "rgdpna_pc").

    Parameters
    ----------
    y_ctries : array-like of str
        list of countries that are considered to be current or former territories,
        disputed regions/countries, or smaller nations (e.g., Andorra) associated with
        nearby larger nations (e.g., France)
    x_ctries : array-like of str
        list of countries that are considered to be current or former patron/sovereign
        countries, disputed regions/countries, or larger nations associated with nearby
        smaller nations; each item of `x_ctries` should correspond to that of `y_ctries`
    agg_df : pandas.DataFrame
        containing the nominal GDP per capita information (in column `y_nom_col`)
    gdppc_df : pandas.DataFrame
        containing PPP GDP per capita information with columns "rgdpna_pc" and
        "gdppc_source"
    y_nom_col : str
        nominal GDP per capita column contained in `agg_df`

    Returns
    -------
    pandas.DataFrame
        containing the columns "x_ccode", "x_rgdpna_pc", "rgdpna_pc", "ratio", and
        "gdppc_source" with multi-indices "ccode" and "year"

    """

    # attaching Group X ctries with their rgdpna_pc and nominal gdppc values
    yiso_df = agg_df.loc[y_ctries, [y_nom_col]].dropna()
    yiso_df = yiso_df.join(
        pd.DataFrame(data={"x_ccode": x_ctries, "ccode": y_ctries}).set_index(["ccode"])
    )
    rename = {x: "x_" + x for x in ["rgdpna_pc", "ccode", y_nom_col]}
    yiso_df = yiso_df.reset_index().merge(
        agg_df[["rgdpna_pc", y_nom_col]].reset_index().rename(columns=rename),
        on=["x_ccode", "year"],
        how="left",
    )

    # 2020 values from `gdppc_df`
    if 2020 in yiso_df.year.unique():
        for iso in yiso_df.loc[(yiso_df.year == 2020), "x_ccode"].unique():
            yiso_df.loc[
                (yiso_df.year == 2020) & (yiso_df.x_ccode == iso), "x_rgdpna_pc"
            ] = gdppc_df.loc[(iso, 2020), "rgdpna_pc"].item()

    # creating rgdpna_pc equivalents
    yiso_df["ratio"] = yiso_df[y_nom_col].div(yiso_df[f"x_{y_nom_col}"])
    yiso_df["rgdpna_pc"] = yiso_df[["x_rgdpna_pc", "ratio"]].prod(axis=1)

    # GDPpc source
    yiso_df.set_index(["ccode", "year"], inplace=True)
    so = "WB(not_PPP)"
    if y_nom_col in ["un_gdppc_nom", "un_gdppc_const"]:
        so = "UN(not_PPP)"
    elif y_nom_col == "cia_rgdpna_pc":
        so = "CIA"
    yiso_df["gdppc_source"] = np.nan
    for i, iso in enumerate(y_ctries):
        xiso = x_ctries[i]
        if xiso == FRA_OVERSEAS_DEPT:
            xiso = "FRA+OV"
        so_fo = f"{so}:{iso}-{xiso}_ratio + PWT:{xiso}"
        if xiso == "FRA":
            so_fo = f"{so}:{iso}-{xiso}_ratio + PWT+OECD:{xiso}"
        yiso_df.loc[iso, "gdppc_source"] = so_fo
        if 2020 in yiso_df.loc[iso, :].reset_index()["year"].unique():
            if iso == "FRA":
                yiso_df.loc[(iso, 2020), "gdppc_source"] = so_fo.replace(
                    "OECD:", "OECD+IMF_gr:"
                )
            else:
                extra = "PWT+WB_gr:"
                if iso == "CYP":
                    extra = "PWT+IMF_gr:"
                yiso_df.loc[(iso, 2020), "gdppc_source"] = so_fo.replace("PWT:", extra)

    return yiso_df.loc[
        ~pd.isnull(yiso_df.ratio),
        ["x_ccode", "x_rgdpna_pc", "rgdpna_pc", "ratio", "gdppc_source"],
    ]


def bertram_sse_helper(Y_tr, Y_te, X_tr, X_te, spec, yvar):
    modelfit = sm.OLS(Y_tr[yvar], sm.add_constant(X_tr[spec])).fit()
    Y_pred = modelfit.predict(sm.add_constant(X_te[spec]))
    sse = np.sum((Y_te[yvar].values - Y_pred.values) ** 2)

    return sse


def bertram_k_fold(df, reg_specs, fold_k=10, yvar="rgdpna_pc", random_state=60637):

    cols = [x for x in df.columns if x != yvar]
    X_df, Y_df = df[cols], df[[yvar]]

    kf = KFold(n_splits=fold_k, shuffle=True, random_state=random_state)
    idx_split = list(kf.split(X_df))  # weird generator error if you don't list this
    min_sse, best_spec = np.inf, None
    for spec in tqdm(reg_specs):
        spec_sse = 0
        for train_idx, test_idx in idx_split:
            X_test, Y_test = X_df.iloc[test_idx], Y_df.iloc[test_idx]
            X_train, Y_train = X_df.iloc[train_idx], Y_df.iloc[train_idx]
            spec_sse += bertram_sse_helper(Y_train, Y_test, X_train, X_test, spec, yvar)
        if min_sse > spec_sse:
            best_spec = spec
            min_sse = spec_sse

    return best_spec
