# various functions used for the country-level information workflow
from itertools import product as lstprod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize as opt_min
from tqdm.auto import tqdm

from .settings import PATH_PWT_RAW, PPP_CCODE_IF_MSNG, SCENARIOS, SSP_PROJ_ORG_SER


def log_lin_interpolate(df, header="v_"):
    """Simple log-linear interpolation, to fit the horizontal (or wide-panel format)
    dataset that we use.

    Parameters
    ----------
    df : pandas DataFrame
        contains data that we may interpolate
    header : str
        header of the variable names; should be followed by year (e.g., "v_1950")

    Returns
    -------
    df_rtn : pandas DataFrame
        DataFrame containing interpolated data

    """

    v_ = np.sort([x for x in df.columns if header in x])
    yrs = [int(x.replace(header, "")) for x in v_]
    all_yrs = range(min(yrs), max(yrs) + 1)
    all_v = np.sort([header + str(x) for x in all_yrs])
    front_v = [x for x in df.columns if header not in x]

    df_missing_v = np.setdiff1d(all_v, v_)
    df_rtn = df.copy()
    if len(df_missing_v) > 0:
        df_rtn[df_missing_v] = np.nan

    ## re-ordering the columns, just in case
    df_rtn = df_rtn[np.hstack([front_v, all_v])]

    for i in df_rtn.index:
        fp = df_rtn.loc[i, :][all_v]
        ## in case there is any nonpositive values or no missing values,
        ## cannot log-linearly interpolate (former) and no need to interpolate (latter)
        if (fp <= 0).any() or (not fp.isnull().any()):
            continue

        where_nm = np.where(~pd.isnull(fp.values))[0]
        fp, i_yrs = np.log(fp[where_nm].astype("float64")), np.array(all_yrs)[where_nm]

        ## we only want to INTERpolate with this function, and not EXTRApolate
        want_interp_range = range(i_yrs.min(), i_yrs.max() + 1)
        case = np.exp(np.interp(want_interp_range, i_yrs, fp))
        want_interp_v = ["v_" + str(x) for x in want_interp_range]
        df_rtn.loc[i, want_interp_v] = case

    return df_rtn


def ssp_and_model_simplify(ssp_col, model_col, df, dic=SSP_PROJ_ORG_SER):
    """Simplifying the SSP and model (IAM) designations. For instance, "SSP2" has many
    corresponding versions of the same scenario including SSP2_v9_130219,
    SSP2_v9_130325, and so forth. This function simplifies those different names.

    Parameters
    ----------
    ssp_col : str
        column name for the SSP scenario
    model_col : str
        column name for the IAM scenario
    df : pandas DataFrame
        DataFrame to apply the function to
    dic : dict
        containing verbose names and simplified names for SSP-IAM scenarios

    Returns
    -------
    df_rtn : pandas DataFrame
        cleaned DataFrame containing simplified ssp and model names

    """

    ser = pd.Series(dic)

    df["ssp"] = ser.reindex(df[ssp_col]).values
    df["iam"] = ser.reindex(df[model_col]).values

    df_rtn = df.copy()
    df_rtn.rename(columns={"REGION": "ccode"}, inplace=True)

    csi = ["ccode", "ssp", "iam"]
    df_rtn_rest_col = [x for x in df_rtn.columns if x not in csi]
    df_rtn = df_rtn[csi + df_rtn_rest_col]

    return df_rtn


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

    ## if there's a PERFERCTLY matching set of growth rates, then just take that
    ## country (or those countries') growth rates
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
    ## indicing for operations below
    ctry_msg = "Needs have the country-code column / index `{}` in the dataset"
    ctry_msg = ctry_msg.format(ctry_col)
    assert (ctry_col in orig_df.index.names) or (ctry_col in orig_df.columns), ctry_msg

    if ctry_col not in orig_df.index.names and ctry_col in orig_df.columns:
        df_idxed = orig_df.set_index([ctry_col])
    else:
        df_idxed = pd.DataFrame(orig_df)

    ## sorting the problematic (with missing-value) countries for consistency
    prob, exclude_these = list(np.sort(prob_lst)), list(exclude_these)

    ## variable names and getting the dataframe of "good-to-go" country codes
    v_ = np.sort(
        [
            x
            for x in orig_df.columns
            if (header in x)
            and (int(x[-4:]) <= begin_end[1])
            and (int(x[-4:]) >= begin_end[0])
        ]
    )

    ## good_ctries are only those that are absolutely filled
    ## excluding those that should be excluded
    good_ctries = df_idxed[(~df_idxed[v_].isnull().any(axis=1))].index.unique()
    good_ctries = np.setdiff1d(good_ctries, prob + exclude_these)
    good_df = df_idxed.loc[good_ctries, :]
    good_gr = yearly_growth(good_df, header)

    ## running for each of the problematic countries
    df_collection = []
    for i in tqdm(prob):
        ## there could be missing values in between known yrs, so interpolate
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

        ## as yearly growth rates, with missing values filled as 0
        tgt_gr = yearly_growth(tgt_df).fillna(0)

        ## detecting which is the valid (or non-missing) growth rates
        gr_row_valid = tgt_gr.loc[i, v_valid].values

        ## subtract the problematic growth rates from good-to-go growth rates,
        ## and calculate the sum of squared errors to detect which is the closest
        sse_df = good_gr.copy()
        sse_df["sse"] = (sse_df[v_valid].sub(gr_row_valid, axis=1) ** 2).sum(axis=1)
        sse_df.sort_values(["sse"], inplace=True)
        sse_df["sse_rank"] = range(0, sse_df.shape[0])

        ## top n closest in terms of trajectory
        necess_sse_df = sse_df[sse_df.sse_rank < n_det][["sse", "sse_rank"]]
        necess_sse_df = necess_sse_df.merge(
            good_df[v_],
            how="left",
            left_index=True,
            right_index=True,
        )

        ## if need to project backwards in time
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

        ## if need to project forward in time
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

        ## extrapolation information as "fill_info"
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
        one account for each countrycode-SSP-IAM combination.

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

    ## resetting the index to be compliant with `pandas.wide_to_long`
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

    ## necessary to reset the index to pass to pandas.pivot
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
    to=True,
    extrap_sim=True,
    fill_msng_ctries=PPP_CCODE_IF_MSNG,
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
    ## reading in the necessary PWT dataframe
    pwt = (
        pd.read_excel(PATH_PWT_RAW)
        .rename(columns={"countrycode": "ccode"})
        .set_index(["ccode", "year"])
    )
    pwt_years = pwt.index.get_level_values("year").unique()
    yr_range = range(pwt_years.min(), pwt_years.max() + 1)

    v_ = ["v_" + str(v) for v in yr_range]
    pl = organize_ver_to_hor(pwt, pwtvar, "year", "ccode", yr_range)
    pl_ccode = pl.index.get_level_values("ccode").unique()

    ## replace with pl_gdpo information if specific pl values for a country are
    ## missing entirely
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

        ## making sure that the fill-information is "-" if information was not missing
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

    ## taking care of the case of Bermuda, since it is sometimes suffering
    ## from negative price levels
    if (pwtvar == "pl_gdpo") and ("BMU" in pl_ccode):
        pl_ccode = np.setdiff1d(pl_ccode, ["BMU"])
        bmu_copy = pl_ver.loc[("GBR", slice(None)), :].reset_index().copy()
        bmu_copy[fill_name] = "copy_from_GBR"
        bmu_copy["ccode"] = "BMU"
        bmu_copy.set_index(["ccode", "year"], inplace=True)
        pl_ver = pd.concat([pl_ver.loc[(pl_ccode, slice(None)), :], bmu_copy], axis=0)

    ## merging the "base" price level, which is that of the US
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

    ## note that according to Feenstra et al. (2015), PPP / XR = pl / pl_base
    ## with the "base" again being the United States; `ppp` below is PPP / XR
    pwt_ppp["ppp"] = pwt_ppp[pwtvar] / pwt_ppp["base"]

    ## multiplying `ppp` can be understood as turning PPP-adjusted value of a certain
    ## year to nominal value; turning base-year-a PPP values to base-year-b PPP values
    ## therefore requires multiplying `ppp`(a) / `ppp`(b)
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

    ## conversion rates
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

    ## filling in the missing countries with known values
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


def minimize_simple_production(x, K_values, Y_values):
    """Helper function for getting at the `A` (TFP) and `alpha` (GDP elasticity of
    capital). Returns the sum of squared errors with respect to actual GDP values.

    Parameters
    ----------
    x : array-like of floats
        divided into `A` (TFP) and `alpha` (GDP elasticity of capital)
    K_values : array-like of floats
        containing historical capital values (in the case of only-capital production
        functional form) or historical per-capita capital values (in the case of Cobb-
        Douglas functional form)
    Y_values : array-like of floats
        containing historical GDP values (in the case of only-capital production
        functional form) or historical per-capita GDP values (in the case of Cobb-
        Douglas functional form)

    Returns
    -------
    sse : float
        Sum of squared errors from netting the actual GDP values from estimated
        GDP values assuming a functional form

    """
    A, alpha = x

    diff = A * (K_values**alpha) - Y_values
    sse = np.sum(diff**2)

    return sse


def MPK_init_calc(
    ccode,
    hist_df,
    base2010_df,
    alpha_overall,
    hist_YKP=["rgdpna_19", "rnna_19", "pop"],
    base_YKP=["gdp", "capital", "pop"],
    init_A_alpha=[100, 1],
):
    """Function for calculating the value of MPK (marginal product of capital) of the
    country specified by `ccode` and in the year specified by `year`.
    Parameters
    ----------
    ccode : str
        country code of the country we need to calculate the MPK for
    hist_df : pandas.DataFrame
        dataframe containing historical (1950-2020) information on GDP, capital stock,
        and population; should contain the column names in `hist_YKP`. Note that its
        values are in millions (of dollars for GDP and capital, of people
        for population)
    base2010_df : pandas.DataFrame
        dataframe containing projected 2010 information on GDP and population and
        baseline 2010 historical capital stock information; should contain the column
        names in `base_YKP`
    alpha_overall : array-like of floats
        should contain elasticities of GDP w.r.t. capital values that have been pre-
        calculated; two elements, the former being our own calculation of the elasticity
        and the latter being the elasticity calculated in Crespo Cuaresma (2017)
    hist_YKP : array-like of str
        column names (of `hist_df`) in the following order: constant PPP GDP
        variable, constant PPP capital stock variable, and population variable
    base_YKP : array-like of str
        column names (of `base2010_df`) in the following order: constant PPP GDP
        variable, constant PPP capital stock variable, and population variable
    init_A_alpha : array-like of floats
        points of initialization for `A` (TFP) and `alpha` (elasticity of GDP w.r.t.
        capital)

    Returns
    -------
    MPK_overall_our, MPK_overall_iiasa, MPK_country_pc, MPK_country : tuple of floats
        calculated MPKs using different versions of the GDP elasticity w.r.t. capital,
        first - using our self-calculated elasticity,
        second - using the Crespo Cuaresma (2017) elasticity,
        third - using the country-specific elasticity assuming Cobb-Douglass function,
        fourth - using the country-specific elasticity assuming capital-only function

    """

    # Y, K, pop values of the years that we want to examine
    histccodes = hist_df.index.get_level_values("ccode").unique()
    baseccodes = base2010_df.index.get_level_values("ccode").unique()
    msg_error = "`ccode` must be in both `hist_df` and `base2010_df`."
    assert (ccode in histccodes) and (ccode in baseccodes), msg_error

    # multiplying 1,000,000 since they are in millions
    YKP = hist_df.loc[ccode, hist_YKP].dropna()
    Ys = YKP[hist_YKP[0]].values * 1000000
    Ks = YKP[hist_YKP[1]].values * 1000000
    Ps = YKP[hist_YKP[2]].values * 1000000

    # creating the capital intensity values using projected base 2010 data
    yk_df = base2010_df.loc[
        base2010_df.index.get_level_values("ccode") == ccode, :
    ].copy()
    yk_df = yk_df.reset_index().set_index(["ccode", "ssp", "iam"])[base_YKP]
    yk_df["yk"] = yk_df[base_YKP[0]] / yk_df[base_YKP[1]]

    # if all zeros for any of the three variables, no reason to calculate MPK
    if (Ys == 0).all() or (Ks == 0).all() or (Ps == 0).all():
        yk_df["mpk_our"], yk_df["mpk_iiasa"] = 0, 0
        yk_df["mpk_ctry_cd"], yk_df["mpk_ctry_co"] = 0, 0
        return yk_df

    # let us sort them in the order of Ks, just in case
    KYPs = np.array(sorted(zip(Ks, Ys, Ps)))
    Ks, Ys, Ps = KYPs[:, 0], KYPs[:, 1], KYPs[:, 2]

    # optimizing values for A (total factor productivity) and alpha (GDP elasticity
    # wrt. capital); capital-only
    A_alpha_getter = lambda x: minimize_simple_production(x, Ks, Ys)
    A, alpha = opt_min(
        A_alpha_getter, init_A_alpha, bounds=((0, np.inf), (0, np.inf))
    ).x

    # optimizing values for A and alpha; Cobb-Douglas
    A_alpha_getter = lambda x: minimize_simple_production(x, Ks / Ps, Ys / Ps)
    A_pc, alpha_pc = opt_min(
        A_alpha_getter, init_A_alpha, bounds=((0, np.inf), (0, np.inf))
    ).x

    # calculating MPK values based on the above A and alpha calculations
    yk_df["mpk_our"] = alpha_overall[0] * yk_df["yk"]
    yk_df["mpk_iiasa"] = alpha_overall[-1] * yk_df["yk"]
    yk_df["mpk_ctry_cd"] = alpha_pc * yk_df["yk"]
    yk_df["mpk_ctry_co"] = alpha * yk_df["yk"]

    return yk_df


def pim_single_ctry(
    ccode_df,
    MPK_init,
    alpha_overall,
    MPK_var="mpk_our",
    scenarios=SCENARIOS,
    yr_startend=[2010, 2100],
    MPK_bar=0.1,
    gamma_MPK=0.985,
    gamma_I=0.98,
    Yvar="gdp",
    Kvar="capital",
    iy_var="iy_ratio",
    depre_overall_var="delta",
    depre_ctry_var="delta_c",
):
    """Function for running the perpetual inventory method (PIM, as described in
    Dellink et al., 2017), for a specific country, for each SSP-IAM scenario.

    ----------
    ccode_df : pandas.DataFrame
        DataFrame containing country-specific information for conducting the by-country
        PIM process to acquire capital stock projections. Needs to contain `Yvar`,
        `Kvar`, with indices `ccode`, `year`, `ssp`, and `iam`.
    MPK_init : pandas.DataFrame
        DataFrame containing initial-year marginal product of capital. Also should
        contain depreciation rates; so should contain `MPK_var`, `depre_overall_var`,
        `iy_var`, and `depre_ctry_var` and with index `ccode`, `ssp`, and `iam`
    alpha_overall : float
        elasticity of GDP w.r.t. capital, global and not country-specific
    MPK_var : str
        column name for the initial-year marginal product of capital
    scenarios : array-like of tuples of str
        array-like of tuples containing SSP and IAM (in that order) scenarios
    yr_startend : array-like of ints
        starting year and end year of projection
    MPK_bar : float
        long-term elasticity of GDP w.r.t. capital, value 0.1 taken from Dellink et al.
        (2017)
    gamma_MPK : float
        velocity of converging to long-term elasticity of GDP w.r.t. capital, value
        0.985 taken from Dellink et al. (2017)
    gamma_I : float
        velocity of converging to long-term investment-to-GDP ratio, value 0.98 taken
        from Dellink et al. (2017)
    Yvar : str
        column name of the constant PPP GDP variable
    Kvar : str
        column name of the (initial-year) constant PPP capital stock variable
    iy_var : str
        column name of the (initial-year) investment-to-GDP variable
    depre_overall_var : str
        depreciation rate variable (over all countries)
    depre_ctry_var : str
        country-specific depreciation rate variable

    Returns
    -------
    ccode_df : pandas.DataFrame
        DataFrame containing the updated values of capital stock projection estimates

    """
    newvar = "{}_estim".format(Kvar)
    ccode = ccode_df.index.get_level_values("ccode").values[0]

    ## delta is same across all scenarios
    delta, delta_r = MPK_init.loc[
        (ccode, "SSP1", "OECD"), [depre_overall_var, depre_ctry_var]
    ].values

    ccode_df["MPK"], ccode_df["IY"] = np.nan, np.nan
    ccode_df[newvar], ccode_df["KY"] = np.nan, np.nan
    for yr in range(yr_startend[0], yr_startend[-1] + 1):
        for scen in scenarios:
            ## advancing MPK values annually
            ssp, iam = scen
            if yr == yr_startend[0]:
                MPK = MPK_init.loc[(ccode, ssp, iam), MPK_var]
            else:
                prev_MPK = ccode_df.loc[(ccode, yr - 1, ssp, iam), "MPK"]
                MPK = gamma_MPK * prev_MPK + (1 - gamma_MPK) * MPK_bar
            ccode_df.loc[(slice(None), yr, ssp, iam), "MPK"] = MPK
            ky_LT = alpha_overall / MPK

            ## year-to-year GDP growth rates
            if yr != yr_startend[-1]:
                g_Ys = ccode_df.loc[(ccode, [yr, yr + 1], ssp, iam), Yvar].values
                Y_yr = g_Ys[0]
            else:
                g_Ys = ccode_df.loc[(ccode, [yr - 1, yr], ssp, iam), Yvar].values
                Y_yr = g_Ys[1]
            g_Y = g_Ys[-1] / g_Ys[0] - 1

            ## long-run I-to-Y ratio
            iy_LT = (g_Y + delta) * ky_LT

            ## I-to-Y ratios time series
            if yr == yr_startend[0]:
                IY = MPK_init.loc[(ccode, ssp, iam), iy_var]
            else:
                prev_IY = ccode_df.loc[(ccode, yr - 1, ssp, iam), "IY"]
                IY = (gamma_I * prev_IY) + (1 - gamma_I) * iy_LT
            ccode_df.loc[(slice(None), yr, ssp, iam), "IY"] = IY

            ## Perpetual inventory method capital stock time series
            if yr == yr_startend[0]:
                K_yr = MPK_init.loc[(ccode, ssp, iam), Kvar]
            else:
                prev_K = ccode_df.loc[(ccode, yr - 1, ssp, iam), newvar]
                K_yr = (1 - delta_r) * prev_K + IY * Y_yr
            ccode_df.loc[(slice(None), yr, ssp, iam), newvar] = K_yr
            ccode_df.loc[(slice(None), yr, ssp, iam), "KY"] = K_yr / Y_yr

    return ccode_df


def examine_against_fig6(pim_df, intensity="KY", fig_size=(18, 9)):
    """
    Function to examine the estimated capital intensity (the variable `intensity` in the
    DataFrame `pim_df`) against the Dellink et al. (2017) output of the same variable
    for four countries Tanzania, India, China, and the United States (shown in Fig. 6 of
    the paper). Also calculates the SSE across own estimates and Dellink et al. (2017)'s
    numbers.

    Parameters
    ----------
    pim_df : pandas DataFrame
        containing the `intensity` variable; should have indices `ccode`, `year`,
        `ssp`, and `iam` (in that order)
    intensity : str
        capital intensity variable in `pim_df`
    fig_size : tuple of floats or ints
        to set the output figure size

    Returns
    -------
    sse : float
        SSE (w.r.t. Dellink et al. (2017) Figure 6) calculated
    also, presents the bar graphs (containing capital intensity values from data)
        plotted in comparison to Dellink et al. (2017) Figure 6

    """

    FOUR_CTRIES = ["TZA", "IND", "CHN", "USA"]
    SSP = ["SSP{}".format(x) for x in range(5, 0, -1)]
    FIG_YRS = [2100, 2050, 2020]

    ## preparing the figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    which = np.arange(0.1, 15 * 0.5 + 0.1, 0.5)
    which = which + np.array(range(0, len(which))) * 0.1

    ## from Dellink et al. (2017); had to measure the figure with a ruler
    ky_cm = 3 / 6.525
    dellink_case = pd.DataFrame(
        list(lstprod(*[FOUR_CTRIES, FIG_YRS, SSP])), columns=["ccode", "year", "ssp"]
    )
    dellink_case[intensity] = np.nan
    dellink_case["year"] = dellink_case["year"].astype("int64")
    dellink_case.set_index(["ccode", "year", "ssp"], inplace=True)

    ## Values from Figure 6, in the order of SSP5 -> SSP1 and 2100, 2050, 2020
    TZN = [
        np.array([6.2, 4.45, 4.4]) * ky_cm,
        np.array([6.95, 6.1, 4.475]) * ky_cm,
        np.array([6.25, 5.675, 4.5]) * ky_cm,
        np.array([5.95, 5.175, 4.45]) * ky_cm,
        np.array([6.25, 4.65, 4.45]) * ky_cm,
    ]
    IND = [
        np.array([7.4, 5.7, 5.75]) * ky_cm,
        np.array([7.575, 6.7, 5.8]) * ky_cm,
        np.array([7.65, 7.45, 5.85]) * ky_cm,
        np.array([7.3, 6.525, 5.775]) * ky_cm,
        np.array([7.6, 5.95, 5.75]) * ky_cm,
    ]
    CHN = [
        np.array([9.9, 8.45, 6.35]) * ky_cm,
        np.array([10.25, 9.55, 6.475]) * ky_cm,
        np.array([9.8, 10.55, 6.525]) * ky_cm,
        np.array([9.6, 9.55, 6.45]) * ky_cm,
        np.array([10.45, 8.8, 6.4]) * ky_cm,
    ]
    USA = [
        np.array([6.25, 5.4, 5.05]) * ky_cm,
        np.array([6.75, 5.9, 5.1]) * ky_cm,
        np.array([7.275, 6.325, 5.15]) * ky_cm,
        np.array([6.9, 6.05, 5.1]) * ky_cm,
        np.array([6.65, 5.8, 5.1]) * ky_cm,
    ]
    for i, ct in enumerate([TZN, IND, CHN, USA]):
        ctry = FOUR_CTRIES[i]
        for j, row in enumerate(ct):
            ssp = SSP[j]
            dellink_case.loc[(ctry, FIG_YRS, ssp), intensity] = row

    labs = []
    for j, ssp in enumerate(SSP):
        labs += ["2100", "{}   2050".format(ssp), "2020"]

    ax1.set_yticks(which + 0.15)
    ax1.set_yticklabels(labs)
    dellink_vals = []
    for l, ctry in enumerate(FOUR_CTRIES):
        ctry_vals = []
        for ssp in SSP:
            ctry_vals += list(dellink_case.loc[(ctry, FIG_YRS, ssp), intensity].values)
        ax1.barh(which + l * 0.1, ctry_vals, height=0.1, label=ctry)
        dellink_vals += ctry_vals
    ax1.legend()
    ax1.set_title("Capital intensities for selected countries, Dellink et al. (2017)")

    ax2.set_yticks(which + 0.15)
    ax2.set_yticklabels(labs)
    our_vals = []
    for l, ctry in enumerate(FOUR_CTRIES):
        ctry_vals = []
        for ssp in SSP:
            ctry_vals += list(
                pim_df.loc[(ctry, FIG_YRS, ssp, "OECD"), intensity].values
            )
        ax2.barh(which + l * 0.1, ctry_vals, height=0.1, label=ctry)
        our_vals += ctry_vals
    mx = max(our_vals + dellink_vals)
    ax_set = np.linspace(0, np.ceil(mx), 5)
    sse = ((np.array(our_vals) - np.array(dellink_vals)) ** 2).sum()
    sser = round(sse, 3)

    ax1.set_xticks(ax_set)
    ax2.set_xticks(ax_set)

    ax2.legend()
    ax2.set_title("Capital intensities, our own replication using the OECD method")

    fig.suptitle("Comparison of capital intensities; SSE={}".format(sser), fontsize=12)
    fig.show()

    return sse


def top_bottom_10(df, yr=2100, ssp="SSP3", capvar="capital_estim"):
    """Shows the top ten and bottom ten according to `capvar` in the DataFrame `df`
    in the year `yr` and the SSP `ssp`; figures for IIASA and OECD IAMs are drawn
    separately.

    Parameters
    ----------
    df : pandas DataFrame
        containing `capvar` variable, with indices `ccode`, `year`,
        `ssp`, and `iam` (in that order)
    yr : int
        year in which we would like to compare the `capvar` values across countries
    ssp : str
        SSP scenario that we would like to examine
    capvar : str
        the name of the variable to produce top 10 and bottom 10 countries from

    Returns
    -------
    None, but presents the top 10 and bottom 10 countries by IAMs in bar graphs

    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 14))

    iiasa_df = df.loc[(slice(None), yr, ssp, "IIASA"), [capvar]].copy()
    iiasa_df.sort_values([capvar], inplace=True)
    iiasa_sma = iiasa_df.index.get_level_values("ccode")[0:10]
    iiasa_sma_vals = np.log(iiasa_df[capvar].values[0:10])

    small = list(range(1, 11))
    ax1.barh(small, iiasa_sma_vals, label="Bottom 10", color="orange", height=0.8)
    ax1.set_yticks(small)
    ax1.set_yticklabels(iiasa_sma)

    iiasa_big = iiasa_df.index.get_level_values("ccode")[-10:]
    iiasa_big_vals = np.log(iiasa_df[capvar].values[-10:])

    big = list(range(11, 21))
    ax1.barh(big, iiasa_big_vals, label="Top 10", color="#87CEEB", height=0.8)
    ax1.set_yticks(small + big)
    ax1.set_yticklabels(np.hstack([iiasa_sma, iiasa_big]))
    fig.suptitle("Log of capital stock in the year {} and {} scenario".format(yr, ssp))
    ax1.set_title("Case for IIASA")
    ax1.set_xlabel("Log of millions of dollars")

    oecd_df = df.loc[(slice(None), yr, ssp, "OECD"), [capvar]].copy()
    oecd_df.sort_values([capvar], inplace=True)
    oecd_sma = oecd_df.index.get_level_values("ccode")[0:10]
    oecd_sma_vals = np.log(oecd_df[capvar].values[0:10])

    ax2.barh(small, oecd_sma_vals, label="Bottom 10", color="orange", height=0.8)
    ax2.set_yticks(small)
    ax2.set_yticklabels(oecd_sma)

    oecd_big = oecd_df.index.get_level_values("ccode")[-10:]
    oecd_big_vals = np.log(oecd_df[capvar].values[-10:])

    ax2.barh(big, oecd_big_vals, label="Top 10", color="#87CEEB", height=0.8)
    ax2.set_yticks(small + big)
    ax2.set_yticklabels(np.hstack([oecd_sma, oecd_big]))
    ax2.set_title("Case for OECD")
    ax2.set_xlabel("Log of millions of dollars")

    fig.show()

    return None
