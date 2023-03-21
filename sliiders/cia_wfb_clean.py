"""
Contains functions to clean CIA World Factbook (WFB). There are different versions
across the years, each with its own format -- this is why there are multiple functions
to organize different versions, some of which that can be grouped with one another due
to sharing similar formats.
"""

import re

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup as BSoup
from tqdm.auto import tqdm

REGIONS_TO_SKIP_CIA_WFB = [
    "Southern Ocean",
    "Indian Ocean",
    "Arctic Ocean",
    "Atlantic Ocean",
    "Pacific Ocean",
    "Baker Island",
]

CIA_WFB_CURRENT_BASE_URL = "https://www.cia.gov/the-world-factbook"
CIA_WFB_2021_BASE_URL = CIA_WFB_CURRENT_BASE_URL + "/about/archives/2021"
fields = ["population", "real-gdp-purchasing-power-parity", "real-gdp-per-capita"]
CIA_WFB_CURRENT_URLS = [CIA_WFB_CURRENT_BASE_URL + "/field/" + i for i in fields]
CIA_WFB_2021_URLS = [CIA_WFB_2021_BASE_URL + "/field/" + i for i in fields]


def helper_wfb_million_cleaner(string):
    """Helper function for cleaning CIA WFB GDP values in millions of USD.

    Parameters
    ----------
    string : str
        containing information about the GDP value (e.g., '$42 million')

    Returns
    -------
    numeric : float
        containing GDP information in millions of USD

    """
    numeric = float(re.sub(r"[a-zA-Z]|\$| |\,|-", "", string))
    if "trillion" in string:
        return numeric * 1e12
    elif "billion" in string:
        return numeric * 1e9
    elif "million" in string:
        return numeric * 1e6

    return numeric


def _gather_data(root_dir, codes=[2001, 2004, 2119], suffix=""):
    return [
        BSoup((root_dir / f"{i}{suffix}.html").open("r").read(), "html.parser")
        for i in codes
    ]


def _get_wfb_yr(direc, yr_st, yr_end):
    wfb_year = int(direc.stem[-4:])
    assert wfb_year in range(
        yr_st, yr_end + 1
    ), f"Cleans only {yr_st} to {yr_end} versions of CIA WFB."
    return wfb_year


def _ASM_fix(df):
    """Correct typo in american samoa. Use assertion to catch if this changes at some
    point."""
    out = df.copy()
    assert (
        out.loc[(out.country == "American Samoa") & (out.year == 2014), "gdp"].item()
        > 1e11
    )
    out.loc[(out.country == "American Samoa") & (out.year == 2014), "gdp"] /= 1e3
    return out


def helper_wfb_gather_soups(directory, subdirectory="geos", print_ver=False):
    """Helper function to go over each geographic location files (in `subdirectory`)
    and gather `bs4.BeautifulSoup` for each file.

    Parameters
    ----------
    directory : str or Path-like
        containing the overall directory containing CIA WFB information for a specific
        version
    subdirectory : str
        subdirectory (under) `directory` that contains all the geographic location files
    print_ver : bool
        if `True`, will gather `bs4.BeautifulSoup` for files with the header 'print_'
        (e.g., `print_us.html`); if `False`, will gather those for files without such
        headers

    Returns
    -------
    soups : list of `bs4.BeautifulSoup`
        for each of the geographic locations in the `subdirectory` under `directory`

    """
    glob = ""
    if print_ver:
        glob = "print_"
    return [
        BSoup(g.open("r").read(), "html.parser")
        for g in (directory / subdirectory).glob(f"{glob}??.html")
        if g.stem != "index"
    ]


def helper_fy_cleaner(list_of_years):
    """Helper function for cleaning a list of years (in string format) that may have
    financial year designations instead of YYYY format.

    Parameters
    ----------
    list_of_years : array-like of str or str
        containing years in string format

    Returns
    -------
    list of int or int
        of the year(s) cleaned in YYYY format

    """

    single = False
    if type(list_of_years) is str:
        list_of_years = [list_of_years]
        single = True

    if np.any(["FY" in x for x in list_of_years]):
        fix = []
        for yr in list_of_years:
            if "FY" in yr:
                yr = int(yr.split("/")[-1]) + 1900
                if yr < 1950:
                    yr += 100
            fix.append(str(yr))
        if single:
            return int(fix[0])
        return [int(x) for x in fix]

    if single:
        return int(list_of_years[0])
    return [int(x) for x in list_of_years]


def organize_cia_wfb_2000_2001(directory, no_info_names=REGIONS_TO_SKIP_CIA_WFB):
    """Organizes the population, GDP, and GDP per capita information from the CIA World
    Factbook (WFB) versions 2000 and 2001 into `pandas.DataFrame` formats. Note, Cyprus
    has a different format b/c it reports for Northern Cyprus as well, so is handled
    differently.

    Parameters
    ----------
    directory : pathlib.Path or str
        containing the relevant WFB information
    no_info_names : array-like of str
        containing country/region names to be excluded when cleaning the information,
        largely due to their pages containing no usable population and GDP information
        (e.g., Arctic Ocean)

    Returns
    -------
    pop_collect : pandas.DataFrame
        containing country/region-level population (in ones of people)
    gdp_collect : pandas.DataFrame
        containing country/region-level PPP GDP (in millions of USD) and PPP GDP per
        capita (in ones of USD)

    """

    wfb_year = _get_wfb_yr(directory, 2000, 2001)
    soups = helper_wfb_gather_soups(directory)

    # population
    pop_collect = []
    for soup in soups:
        name = soup.find("title").text.split(" -- ")[-1].strip()
        if name in no_info_names:
            continue

        popstr = soup.text[
            soup.text.find("Population:") : soup.text.find("Age structure:")
        ].replace("\n", "")
        if ("no indigenous" in popstr) or ("uninhabited" in popstr):
            continue

        popstr = [
            x
            for x in re.split(r"\(|\)", re.sub(r"(Population:)|\,|(est.)", "", popstr))
            if len(x.replace(" ", "")) > 0
        ]
        pop_val, pop_year = popstr[0], popstr[1]

        if name in ["South Africa", "Syria"]:
            pop_year = popstr[-1]

        if "note:" in pop_val:
            pop_val = pop_val.split("note:")[0]

        pop_collect.append(
            [name, int(round(float(pop_val.strip()))), int(pop_year.strip()[-4:])]
        )

    pop_collect = pd.DataFrame(pop_collect, columns=["country", "pop", "year"])
    pop_collect["wfb_year"] = wfb_year

    # GDP and GDPpc
    gdp_collect = []
    for soup in soups:
        name = soup.find("title").text.split(" -- ")[-1].strip()
        if name in no_info_names:
            continue

        # GDP
        gdp_txt = soup.text.replace("\n", " ")
        if name == "Cyprus":
            front_txt = "GDP: Greek Cypriot area:"
        elif wfb_year == 2000:
            front_txt = "GDP: purchasing power parity"
        elif wfb_year == 2001:
            front_txt = "GDP:  purchasing power parity"
        if front_txt not in gdp_txt:
            continue

        gdp_txt, gdppc_txt = gdp_txt[
            gdp_txt.find(front_txt) : gdp_txt.find("GDP - composition by sector")
        ].split("GDP - real growth rate:")
        gdp_txt = [
            x.strip()
            for x in re.split(
                r"\(|\)", re.sub(r"({} - \$)|( est.)".format(front_txt), "", gdp_txt)
            )
            if len(x.strip()) > 0
        ]
        if name == "Cyprus":
            gdp_txt = [i.split("$")[1] for i in gdp_txt[0].split(";")] + gdp_txt[1:2]
            names = ["Cyprus", "Northern Cyprus"]
        else:
            names = [name]
        if gdp_txt[0] == "NA":
            continue
        gdp_vals = [helper_wfb_million_cleaner(i) for i in gdp_txt[: len(names)]]
        gdp_year = helper_fy_cleaner([gdp_txt[len(names)]])[0]

        # GDPpc
        front_txt = "GDP - per capita:"
        gdppc_txt = re.sub(r" est.", "", gdppc_txt.split(front_txt)[-1]).strip()
        additional_condition = (name in ["Svalbard", "Norway"]) and (wfb_year == 2001)
        if (gdppc_txt[-2:] == "NA") or additional_condition:
            continue
        gdppc_val, gdppc_year = gdppc_txt.split("(")
        gdppc_year = gdppc_year.strip(" )")
        if "FY" in gdppc_year:
            possible_yrs = np.array([1900 + int(i) for i in gdppc_year[2:].split("/")])
            assert (possible_yrs == gdp_year).any()
            gdppc_year = gdp_year
        else:
            gdppc_year = int(gdppc_year)
        assert gdppc_year == gdp_year, (gdppc_year, gdp_year)
        gdppc_vals = [
            int(i.split("$")[-1].replace(",", "")) for i in gdppc_val.split(";")
        ]

        for nx, n in enumerate(names):
            gdp_collect.append([n, gdp_vals[nx], gdppc_vals[nx], gdp_year])

    gdp_collect = pd.DataFrame(gdp_collect, columns=["country", "gdp", "gdppc", "year"])
    gdp_collect["wfb_year"] = wfb_year

    return pop_collect, gdp_collect


def organize_cia_wfb_2002_2004(directory):
    """Organizes the population, GDP, and GDP per capita information from the CIA World
    Factbook (WFB) versions 2002-2004 into `pandas.DataFrame` formats.

    Parameters
    ----------
    directory : pathlib.Path or str
        containing the relevant WFB information

    Returns
    -------
    pop_df : pandas.DataFrame
        containing country/region-level population information (in ones of people)
    gdp_df : pandas.DataFrame
        containing country/region-level PPP GDP information (in millions of USD)
    gdppc_df : pandas.DataFrame
        containing country/region-level PPP GDP per capita information (in ones of USD)

    """

    wfb_year = _get_wfb_yr(directory, 2002, 2004)
    soups = _gather_data(directory / "fields")

    # GDP and GDP per capita
    gdp_case = True
    for soup in soups[:2]:
        gdp_lst = [
            re.sub(r"\n|\t", "", x.text)
            for x in soup.find_all("tr")
            if "power parity" in x.text
        ][1:]
        gdp_lst = [x.split("purchasing power parity - $") for x in gdp_lst]
        if not gdp_case:
            gdp_lst = [[x[0]] + [f.replace(",", "") for f in x[1:]] for x in gdp_lst]
        gdp_collect = []
        for i in gdp_lst:

            # skip if only containing NA
            if " - NA " in i[0]:
                continue

            gdp_vals = [val.strip().split(" (") for val in i[1:]]
            gdp_years = [v[1] if len(v) > 1 else None for v in gdp_vals]

            gdp_vals = [v[0].split("note")[0].strip() for v in gdp_vals]

            if any(["NA" in i for i in gdp_vals]):
                continue

            if gdp_case:
                gdp_vals = [helper_wfb_million_cleaner(gdp_val) for gdp_val in gdp_vals]
            else:
                gdp_vals = [int(gdp_val.strip()) for gdp_val in gdp_vals]

            gdp_years = [
                wfb_year
                if y is None
                else helper_fy_cleaner([y.split("est.")[0].replace(")", "").strip()])[0]
                for y in gdp_years
            ]

            if "World" in i[0]:
                gdp_collect.append(["World", gdp_vals[0], gdp_years[0]])
            else:
                if "Cyprus" in i[0]:
                    names = ["Cyprus", "Northern Cyprus"]
                else:
                    names = i[:1]
                for nx, n in enumerate(names):
                    gdp_collect.append([n, gdp_vals[nx], gdp_years[nx]])

        if gdp_case:
            gdp_df = gdp_collect.copy()
            gdp_case = False
        else:
            gdppc_df = gdp_collect.copy()

    gdppc_df = pd.DataFrame(gdppc_df, columns=["country", "gdppc", "year"])
    gdp_df = pd.DataFrame(gdp_df, columns=["country", "gdp", "year"])
    gdp_df["wfb_year"], gdppc_df["wfb_year"] = wfb_year, wfb_year

    # Population
    pop_df = []
    pop_lst = [
        re.sub(r"\n|\t", "", x.text)
        for x in soups[-1].find_all("tr")
        if "est.)" in x.text
    ][1:]
    for i in pop_lst:
        if ("no indigenous" in i) or ("uninhabited" in i):
            continue

        pop_idx = re.search(r"[0-9]", i).span()[0]
        name, pop_info = i[0:pop_idx], i[pop_idx:]
        pop_val = pop_info.split("(")
        pop_val, pop_year = pop_val[0], pop_val[-1]
        if "note" in pop_val:
            pop_val = pop_val.split("note")[0]
        if "million" in pop_val:
            pop_val = int(
                round(float(pop_val.strip().replace(" million", "")) * 1000000)
            )
        else:
            pop_val = int(pop_val.strip().replace(",", ""))
        pop_year = int(re.sub(r"[a-zA-Z]|\.|\)", "", pop_year))
        pop_df.append([name, pop_val, pop_year])

    pop_df = pd.DataFrame(pop_df, columns=["country", "pop", "year"])
    pop_df["wfb_year"] = wfb_year

    return pop_df, gdp_df, gdppc_df


def organize_cia_wfb_2005_2008(directory):
    """Organizes the population, GDP, and GDP per capita information from the CIA World
    Factbook (WFB) versions 2005 and 2007 into `pandas.DataFrame` formats.

    Parameters
    ----------
    directory : pathlib.Path or str
        containing the relevant WFB information

    Returns
    -------
    pop_df : pandas.DataFrame
        containing country/region-level population information (in ones of people)
    gdp_df : pandas.DataFrame
        containing country/region-level PPP GDP information (in millions of USD)
    gdppc_df : pandas.DataFrame
        containing country/region-level PPP GDP per capita information (in ones of USD)
    """

    wfb_year = _get_wfb_yr(directory, 2005, 2008)
    soups = _gather_data(directory / "fields")

    # GDP and GDP per capita
    for case, soup in enumerate(soups):
        collect = []
        lst = [
            re.sub(r"\n|\t", "", x.text)
            for x in soup.find_all("tr")
            if "est.)" in x.text
        ][1:]

        for i in lst:
            cnd_check = ("no indigenous" in i) or ("uninhabited" in i) or ("NA" in i)
            if cnd_check:
                continue

            searchby = r"\$"
            if case == 2:
                searchby = r"[0-9]"

            idx = re.search(searchby, i).span()[0]
            name, value = i[0:idx].replace("purchasing power parity - ", ""), i[idx:]

            if "World" in name:
                name = "World"

            if "Cyprus" in name:
                values = re.split(
                    "north Cyprus|area administered by Turkish Cypriots", value
                )
                year_ix = 1
            else:
                values = [value]
                year_ix = -1
            values = [value.split(" (") for value in values]

            years = [
                value[year_ix].strip("; ") if len(value) > 1 else None
                for value in values
            ]
            values = [value[0] for value in values]

            if ("- supplemented" in value) or ("note" in value):
                values = [
                    re.split(r"note|- supplemented", value)[0] for value in values
                ]
                value = [re.sub(r"\;", "", value) for value in values]

            if "Cyprus" in name:
                if len(values) == 2:
                    names = ["Cyprus", "Northern Cyprus"]
                    values = [v.split("$")[-1].strip() for v in values]
                elif len(values) < 2:
                    # in 2008, only GDP (not GDPpc) starts being combined between the 2
                    if case == 0 and wfb_year == 2008:
                        names = ["Cyprus/Northern Cyprus"]
                    else:
                        names = ["Cyprus"]
                else:
                    raise ValueError(value)
            else:
                names = [name]

            if case == 0:
                values = [
                    helper_wfb_million_cleaner(value.strip("; ")) for value in values
                ]
            else:
                values = [
                    re.sub(r"\$|\,| for Serbia", "", value).strip() for value in values
                ]
                values = [
                    int(round(float(value.replace("million", "").strip()) * 1000000))
                    if "million" in value
                    else int(value)
                    for value in values
                ]
            years = [
                int(
                    re.sub(
                        r"[a-zA-Z]",
                        "",
                        year.replace(" est.", "").replace(")", "").strip(),
                    )
                )
                for year in years
                if year is not None
            ]
            if len(years) > 1:
                assert len(years) == 2
            else:
                years *= len(names)

            for nx, n in enumerate(names):
                collect.append([n, values[nx], years[nx]])
        if case == 0:
            gdp_df = collect.copy()
        elif case == 1:
            gdppc_df = collect.copy()

    # GDP and GDPpc
    gdp_df = pd.DataFrame(gdp_df, columns=["country", "gdp", "year"])
    gdppc_df = pd.DataFrame(gdppc_df, columns=["country", "gdppc", "year"])
    gdp_df["wfb_year"], gdppc_df["wfb_year"] = wfb_year, wfb_year

    # population
    pop_df = pd.DataFrame(collect, columns=["country", "pop", "year"])
    pop_df["wfb_year"] = wfb_year

    return pop_df, gdp_df, gdppc_df


def organize_cia_wfb_2009_2012(directory):
    """Organizes the population, GDP, and GDP per capita information from the CIA World
    Factbook (WFB) versions 2009-2012 into `pandas.DataFrame` formats.

    Parameters
    ----------
    directory : pathlib.Path or str
        containing the relevant WFB information

    Returns
    -------
    pop_df : pandas.DataFrame
        containing country/region-level population information (in ones of people)
    gdp_df : pandas.DataFrame
        containing country/region-level PPP GDP information (in millions of USD)
    gdppc_df : pandas.DataFrame
        containing country/region-level PPP GDP per capita information (in ones of USD)
    """

    wfb_year = _get_wfb_yr(directory, 2009, 2012)
    soups = _gather_data(directory / "fields")

    cat_dict = {"class": "category_data"}
    for i, soup in enumerate(soups):
        # every displayed "row" is organized as a "table"
        souptable = [
            x
            for x in soup.find_all("table")
            if x.find("td", attrs=cat_dict) is not None
        ]
        souptable = [t.find("td", attrs={"class": "fl_region"}) for t in souptable]
        names = [t.find("a").text for t in souptable]
        values = [t.find("td", attrs=cat_dict) for t in souptable]

        # for reducing redundancies, as the tables are in a nested structure
        # same country information can be searched multiple times
        already_names = ["Akrotiri", "Dhekelia"]
        collect_df = []
        for j, value in enumerate(values):
            name, v = names[j], value.text
            if name in already_names:
                continue
            if i != 2:
                org = [
                    x.strip()
                    for x in v.split("\n")
                    if (len(x.strip()) > 0) and ("NA" not in x)
                ]
                numbers = [
                    x.replace("note:", "").strip()
                    for x in org
                    if (("(" in x) and (")" in x) and ("MADDISON" not in x))
                    or ("est." in x)
                ]
                note = [x for x in org if (x not in numbers) and ("data are in" in x)]
                num_orgs, years = [], []
                for num in numbers:
                    n, year = num.split("(")[0], num.split("(")[-1]
                    if i == 0:
                        n = helper_wfb_million_cleaner(n)
                    else:
                        n = int(re.sub("\$|\,", "", n).strip())
                    num_orgs.append(n)
                    years.append(
                        int(re.sub(r"[a-zA-Z]|\)|\.|\;|\$", "", year).strip()[0:4])
                    )

                usd_years = years.copy()
                if note:
                    nn = note[0].split(";")[0]
                    usd_years = [int(re.sub(r"[a-zA-Z]|\:|\.", "", nn).strip())] * len(
                        years
                    )

                # Cyprus GDPpc starts being combined in 2009
                if name == "Cyprus":
                    name = "Cyprus/Northern Cyprus"

                df = pd.DataFrame(
                    data=dict(
                        zip(
                            ["country", "year", "gdp", "usd_year"],
                            [[name] * len(years), years, num_orgs, usd_years],
                        )
                    )
                ).astype(
                    {
                        "country": str,
                        "year": "uint16",
                        "usd_year": "uint16",
                    },
                )
            else:
                if ("no indigenous" in v) or ("uninhabited" in v):
                    continue
                org = v.strip().replace("\n", "").replace(",", "").split("(")
                num = org[0].split("note")[0].strip()
                if "million" in num:
                    num = round(float(num.replace("million", "").strip()) * 1000000)
                else:
                    num = num.replace("total:", "").strip()
                num = int(num)

                if (name == "Curacao") and (wfb_year in [2010, 2011]):
                    year = re.sub(r"\)|\.", "", [x for x in org if "est." in x][0])
                elif (name == "South Sudan") and (wfb_year == 2011):
                    year = re.sub(r"\)", "", org[-1])
                else:
                    year = [x.split("est.)")[0] for x in org if "est.)" in x][0]

                # Cyprus pop starts being combined in 2010
                if name == "Cyprus" and wfb_year > 2009:
                    name = "Cyprus/Northern Cyprus"
                year = int(re.sub(r"[a-zA-Z]| ", "", year))
                df = [name, num, year]

            already_names.append(name)
            collect_df.append(df)
        if i == 0:
            gdp_df = pd.concat(collect_df, axis=0).reset_index(drop=True)
        elif i == 1:
            gdppc_df = (
                pd.concat(collect_df, axis=0)
                .reset_index(drop=True)
                .rename(columns={"gdp": "gdppc"})
            )
    pop_df = pd.DataFrame(collect_df, columns=["country", "pop", "year"])
    gdp_df["wfb_year"] = wfb_year
    gdppc_df["wfb_year"], pop_df["wfb_year"] = wfb_year, wfb_year

    # drop duplicates, due to multiple entries for North Korea in 2011
    gdppc_df.drop_duplicates(inplace=True)
    gdp_df.drop_duplicates(inplace=True)

    # manual cleaning to fix or drop unreliable data
    if wfb_year in [2011, 2012]:
        gdppc_df.loc[
            (gdppc_df.country == "Gibraltar") & (gdppc_df.gdppc == 43000),
            ["year", "usd_year"],
        ] = 2008
        if wfb_year == 2012:
            gdppc_df.loc[
                (gdppc_df.country == "Kosovo") & (gdppc_df.gdppc == 7400),
                ["year", "usd_year"],
            ] = 2012

    return pop_df, gdp_df, gdppc_df


def organize_cia_wfb_2013_2014(directory):
    """Organizes the population, GDP, and GDP per capita information from the CIA World
    Factbook (WFB) versions 2013-2014 into `pandas.DataFrame` formats.

    Parameters
    ----------
    directory : pathlib.Path or str
        containing the relevant WFB information

    Returns
    -------
    pop_df : pandas.DataFrame
        containing country/region-level population information (in ones of people)
    gdp_df : pandas.DataFrame
        containing country/region-level PPP GDP information (in millions of USD)
    gdppc_df : pandas.DataFrame
        containing country/region-level PPP GDP per capita information (in ones of USD)

    """

    wfb_year = _get_wfb_yr(directory, 2013, 2014)
    soups = _gather_data(directory / "fields")

    for i, soup in enumerate(soups):
        if (i == 2) and (wfb_year == 2015):
            continue
        soupfind = soup.find("div", attrs={"class": "text-holder-full"}).find_all(
            "td", attrs={"class": "fl_region"}
        )
        df_agg = []
        for j, case in enumerate(soupfind):
            name = case.text.split("\n\n")[0]
            values = case.find("td").text

            cnd_skip1 = name in ["Akrotiri", "Dhekelia"]
            cnd_skip2 = (i != 2) and (("NA" in values) or (name == "Gaza Strip"))
            cnd_skip3 = ("no indigenous" in values) or ("uninhabited" in values)
            if cnd_skip1 or cnd_skip2 or cnd_skip3:
                continue
            values = [x for x in values.split("\n") if len(x.strip()) > 0]
            if (i == 2) and (wfb_year == 2014):
                note = [x for x in values if ("note" in x)]
                values = [
                    x for x in values if ("note" not in x) and ("top ten" not in x)
                ]
            else:
                note = [x for x in values if ("note: data are in" in x)]
                if np.any(["est." in x for x in values]) and (i != 2):
                    values = [
                        x
                        for x in values
                        if ("est." in x) and ("note" not in x) and ("top ten" not in x)
                    ]
                else:
                    values = [
                        x for x in values if ("note" not in x) and ("top ten" not in x)
                    ]

            nums, years = [], []
            for val in values:
                if (name == "Bahrain") and (i == 2) and (wfb_year == 2013):
                    num, year = val.split("July")
                elif (i == 2) and (wfb_year == 2014) and (len(note) > 0):
                    num = val.strip()
                    if "(" in num:
                        num, year = num.split("(")
                    if "est." in note[0]:
                        year = note[0].split("(")[-1]
                elif "(" in val:
                    num, year = val.split("(")
                else:
                    num, year = val.strip(), str(wfb_year)
                year = re.sub(r"\(|\)|est.", "", year).strip()
                if "FY" in year:
                    year = helper_fy_cleaner(year)
                else:
                    year = int(re.sub(r"[a-zA-Z]", "", year).strip())
                if i == 0:
                    num = helper_wfb_million_cleaner(num.strip())
                else:
                    num = int(re.sub(r"\$|\,", "", num))
                years.append(year)
                nums.append(num)

            if len(nums) == 0:
                continue

            if i != 2:
                usd_years = years.copy()
                if len(note) > 0:
                    note = note[0].split("note: data are in")[-1].split("US dollars")[0]
                    usd_years = [int(note.strip())] * len(years)

                columns = ["country", "year", "gdp", "usd_year"]
                datavals = [[name] * len(years), years, nums, usd_years]

            else:
                columns = ["country", "year", "pop"]
                datavals = [[name] * len(years), years, nums]

            df_agg.append(pd.DataFrame(data=dict(zip(columns, datavals))))

        if i == 0:
            gdp_df = pd.concat(df_agg, axis=0).reset_index(drop=True)
            gdp_df["wfb_year"] = wfb_year
        elif i == 1:
            gdppc_df = pd.concat(df_agg, axis=0).reset_index(drop=True)
            gdppc_df.rename(columns={"gdp": "gdppc"}, inplace=True)

    pop_df = pd.concat(df_agg, axis=0).reset_index(drop=True)
    gdppc_df["wfb_year"], pop_df["wfb_year"] = wfb_year, wfb_year

    # manual cleaning to fix or drop unreliable data
    if wfb_year == 2013:
        gdp_df.loc[(gdp_df.country == "Macau") & (gdp_df.gdp > 47000), "year"] = 2012
        gdp_df = gdp_df.loc[
            ~((gdp_df.country == "Syria") & (gdp_df.year == 2010)), :
        ].copy()
        gdppc_df.loc[
            (gdppc_df.country == "Gibraltar") & (gdppc_df.gdppc == 43000), "year"
        ] = 2008
    else:
        gdp_df = gdp_df.loc[
            ~((gdp_df.country == "Croatia") & (gdp_df.year == 2012)), :
        ].copy()
        gdppc_df = gdppc_df.loc[
            ~((gdppc_df.country == "Kenya") & (gdppc_df.year == 2013)), :
        ].copy()
    gdppc_df = gdppc_df.loc[
        ~((gdppc_df.country == "Syria") & (gdppc_df.year == 2010)), :
    ].copy()

    return pop_df, gdp_df, gdppc_df


def organize_cia_wfb_2015(directory):
    """Organizes the population, GDP, and GDP per capita information from the CIA World
    Factbook (WFB) version 2015 into `pandas.DataFrame` formats.

    Parameters
    ----------
    directory : pathlib.Path or str
        containing the relevant WFB information

    Returns
    -------
    pop_df : pandas.DataFrame
        containing country/region-level population information (in ones of people)
    gdp_df : pandas.DataFrame
        containing country/region-level PPP GDP information (in millions of USD)
    gdppc_df : pandas.DataFrame
        containing country/region-level PPP GDP per capita information (in ones of USD)
    """

    soups = _gather_data(directory / "rankorder", suffix="rank")

    for i, soup in enumerate(soups):
        ranks = soup.find("table", attrs={"id": "rankOrder"})
        rows = ranks.find_all("tr")
        df = []
        for tr in rows:
            if "Date of Information" in tr.text:
                continue

            ranking, name, value, year = tr.find_all("td")
            if len(value.text.strip()) == 0:
                continue
            if len(year.text.strip()) == 0:
                year = 2014
            elif "FY" in year.text:
                front, back = year.text.split("/")
                back = re.sub(r"[a-zA-Z]|\.", "", back).strip()
                year = int(back) + 2000
                if year > 2050:
                    year -= 100
            else:
                year = int(re.sub(r"[a-zA-Z]|\.", "", year.text).strip())

            value = int(re.sub(r"\$|\,", "", value.text).strip())
            df.append([name.text, year, value])

        if i == 0:
            gdp_df = pd.DataFrame(df, columns=["country", "year", "gdp"])
        elif i == 1:
            gdppc_df = pd.DataFrame(df, columns=["country", "year", "gdppc"])

    gdppc_df["usd_year"], gdp_df["usd_year"] = gdppc_df["year"], gdp_df["year"]
    gdppc_df["wfb_year"], gdp_df["wfb_year"] = 2015, 2015
    pop_df = pd.DataFrame(df, columns=["country", "year", "pop"])
    pop_df["wfb_year"] = 2015

    return pop_df, gdp_df, gdppc_df


def organize_cia_wfb_2016_2017(directory):
    """Organizes the population, GDP, and GDP per capita information from the CIA World
    Factbook (WFB) versions 2016-2017 into `pandas.DataFrame` formats.

    Parameters
    ----------
    directory : pathlib.Path or str
        containing the relevant WFB information

    Returns
    -------
    pop_df : pandas.DataFrame
        containing country/region-level population information (in ones of people)
    gdp_df : pandas.DataFrame
        containing country/region-level PPP GDP information (in millions of USD)
    gdppc_df : pandas.DataFrame
        containing country/region-level PPP GDP per capita information (in ones of USD)

    """

    wfb_year = _get_wfb_yr(directory, 2016, 2017)
    soups = _gather_data(directory / "fields")

    for i, soup in enumerate(soups):
        names = [x.text for x in soup.find_all("td", attrs={"class": "country"})]
        cases = soup.find_all("td", attrs={"class": "fieldData"})
        if i == 2:
            cases = [x.text for x in cases]
        else:
            cases = [x.text.strip("\n").split("\n") for x in cases]
        df_agg = []
        for j, name in enumerate(names):
            this_case = cases[j]
            cnd_skip1 = name in ["Akrotiri", "Dhekelia"]
            cnd_skip2 = (i != 2) and np.any(["NA" in x for x in this_case])
            cnd_skip3 = (i == 2) and (
                ("no indigenous" in this_case) or ("uninhabited" in this_case)
            )
            if cnd_skip1 or cnd_skip2 or cnd_skip3:
                continue
            if i != 2:
                values = [
                    x.strip()
                    for x in this_case
                    if ("est." in x) and ("note" not in x) and (len(x.strip()) > 0)
                ]
                note = [
                    x.strip()
                    for x in this_case
                    if (len(x.strip()) > 0) and ("note" in x)
                ]

                nums, years = [], []
                for val in values:
                    num, year = val.split("(")
                    if i == 0:
                        num = helper_wfb_million_cleaner(num)
                    else:
                        num = int(re.sub(r"\,|\$", "", num).strip())
                    year = re.sub(r"\)|est.", "", year).strip()
                    if "FY" in year:
                        year = helper_fy_cleaner(year)
                    else:
                        year = int(year.strip())
                    nums.append(num), years.append(year)
                usd_years = years.copy()
                if len(note) > 0:
                    note_fix = note[0].replace(" US", "")
                    idx = note_fix.find(" dollars")
                    if idx != -1:
                        usd_years = [int(note_fix[(idx - 4) : idx])] * len(years)

            else:
                case = cases[j].replace("\n", "").split("(")
                if np.any(["top ten" in x for x in case]):
                    num, year = case[0], case[1].split("top ten")[0]
                else:
                    num = re.split(r"note|rank by population", case[0])[0]
                    year = case[-1]
                year = year.replace(")", "").split("note")[0]
                if "FY" in year:
                    year = helper_fy_cleaner(
                        "FY" + re.sub(r"[a-zA-Z]|\.", "", year).strip()
                    )
                else:
                    year = int(re.sub(r"[a-zA-Z]|\.", "", year).strip())
                if "million" in num:
                    num = round(float(num.replace("million", "").strip()) * 1000000)
                else:
                    num = re.sub(r"\,|[a-zA-Z]|\:", "", num).strip()
                num = int(num)

            if i != 2:
                columns = ["country", "year", "gdp", "usd_year"]
                datavals = [[name] * len(nums), years, nums, usd_years]
                df_agg.append(
                    pd.DataFrame(data=dict(zip(columns, datavals))).astype(
                        {"year": "uint16", "usd_year": "uint16"}
                    )
                )
            else:
                df_agg.append([name, year, num])

        if i == 0:
            gdp_df = pd.concat(df_agg, axis=0).reset_index(drop=True)
        elif i == 1:
            gdppc_df = pd.concat(df_agg, axis=0).reset_index(drop=True)
            gdppc_df.rename(columns={"gdp": "gdppc"}, inplace=True)

    pop_df = pd.DataFrame(df_agg, columns=["country", "year", "pop"])
    gdppc_df["wfb_year"], gdp_df["wfb_year"], pop_df["wfb_year"] = [wfb_year] * 3
    gdppc_df.drop_duplicates(inplace=True)

    # manual cleaning to fix or drop unreliable data
    gdp_df = gdp_df[~gdp_df[["country", "year"]].duplicated(keep=False)]

    return pop_df, gdp_df, gdppc_df


def organize_cia_wfb_2018_2019(directory):
    """Organizes the population, GDP, and GDP per capita information from the CIA World
    Factbook (WFB) versions 2018-2019 into `pandas.DataFrame` formats.

    Parameters
    ----------
    directory : pathlib.Path or str
        containing the relevant WFB information

    Returns
    -------
    pop_df : pandas.DataFrame
        containing country/region-level population information (in ones of people)
    gdp_df : pandas.DataFrame
        containing country/region-level PPP GDP information (in millions of USD)
    gdppc_df : pandas.DataFrame
        containing country/region-level PPP GDP per capita information (in ones of USD)

    """

    wfb_year = _get_wfb_yr(directory, 2018, 2019)
    soups = _gather_data(directory / "fields", codes=[208, 211, 335])

    find_val_fields = [
        "field-gdp-purchasing-power-parity",
        "field-gdp-per-capita-ppp",
        "field-population",
    ]
    find_category = "category_data subfield historic"
    df_cols_list = [
        ["country", "year", "gdp", "usd_year"],
        ["country", "year", "gdppc", "usd_year"],
        ["country", "year", "pop"],
    ]
    for i, soup in enumerate(soups):
        find_val_field, df_cols = find_val_fields[i], df_cols_list[i]
        souptable = soup.find("table", attrs={"id": "fieldListing"})
        countries = [
            x.text.replace("\n", "")
            for x in souptable.find_all("td", attrs={"class": "country"})
        ]
        if i == 2:
            find_category = "category_data subfield numeric"
        values = souptable.find_all("div", attrs={"id": find_val_field})
        notes = [v.find("div", attrs={"class": "category_data note"}) for v in values]
        values = [v.find_all("div", attrs={"class": find_category}) for v in values]

        df_collect = []
        for j, val in enumerate(values):
            # case when there are no information available
            if len(val) == 0:
                continue

            # getting the country name and note (note could be None)
            name, note = countries[j], notes[j]

            # multiple years and values available in versions 2017 and onwards
            numbers, years = [], []
            for v in val:
                year = None
                num = v.text.replace("\n", "").split("(")
                if len(num) > 1:
                    num, year = num[0], num[-1]
                    year = re.sub(r"[a-zA-Z]|\)| |\.", "", year)
                    if ("FY" in v.text) and ("/" in year):
                        year = "FY" + year
                    year = helper_fy_cleaner(year)

                if i == 0:
                    numbers.append(helper_wfb_million_cleaner(num))
                    years.append(year)
                else:
                    cnd_check = (
                        ("no indigenous" in num)
                        or ("uninhabited" in num)
                        or ("Akrotiri" in num)
                        or ("NA" in num)
                        or (year is None)
                    )
                    if cnd_check:
                        continue
                    if ("million" in num) and (i == 2):
                        num = round(float(num.replace("million", "").strip()) * 1000000)
                    else:
                        num = re.sub(r"\$|\,|[a-zA-Z]", "", num.strip())
                    numbers.append(int(num))
                    years.append(year)
            if len(numbers) == 0:
                continue

            name = [name] * len(years)

            # what year the GDP values are in
            if i != 2:
                usd_years = years.copy()
                if note is not None:
                    note = note.text
                    if not (("data are in" in note) and ("dollars" in note)):
                        continue
                    if (";" in note) or ("the war-driven" in note):
                        note = [
                            x
                            for x in re.split(r"\;|the war-driven", note)
                            if ("data are in" in x) and ("dollars" in x)
                        ][0]
                    noteyear = re.sub(r"[a-zA-Z]| |\n|\:", "", note)
                    usd_years = [int(noteyear)] * len(years)

                df_vals = [name, years, numbers, usd_years]
            else:
                df_vals = [name, years, numbers]
            df_collect.append(pd.DataFrame(data=dict(zip(df_cols, df_vals))))

        if i == 0:
            gdp_df = pd.concat(df_collect, axis=0).reset_index(drop=True)
        elif i == 1:
            gdppc_df = pd.concat(df_collect, axis=0).reset_index(drop=True)

    pop_df = pd.concat(df_collect, axis=0).reset_index(drop=True)
    gdp_df["wfb_year"], gdppc_df["wfb_year"], pop_df["wfb_year"] = [wfb_year] * 3

    return pop_df, gdp_df, gdppc_df


def helper_wfb_2020(soup):
    """Simple helper function for finding and cleaning the name of a country/region,
    used for organizing CIA World Factbook versions 2018 to 2020 (in conjunction with
    the function `organize_cia_wfb_2018_2020`).

    Parameters
    ----------
    soup : bs4.BeautifulSoup
        containing country/region information

    Returns
    -------
    name : str
        of the country/region being represented in `soup`

    """
    name = soup.find("title").text
    if " :: " in name:
        name = name.split(" :: ")[1].split(" — ")[0]
    else:
        name = name.split(" - ")[0]

    return name


def organize_cia_wfb_2020(
    directory,
    no_info_names=REGIONS_TO_SKIP_CIA_WFB,
):
    """Organizes the population, GDP, and GDP per capita information from the CIA World
    Factbook (WFB) version 2020 into `pandas.DataFrame` format.

    Parameters
    ----------
    directory : pathlib.Path or str
        containing the relevant WFB information
    no_info_names : array-like of str
        containing country/region names to be excluded when cleaning the information,
        largely due to their pages containing no usable population and GDP information
        (e.g., Arctic Ocean)

    Returns
    -------
    pop_collect : pandas.DataFrame
        containing population information (units in ones of people)
    gdp_collect : pandas.DataFrame
        containing PPP GDP information (units in millions of USD, USD year designated
        by the column `usd_year`)
    gdppc_collect : pandas.DataFrame
        containing PPP GDP per capita information (units in ones of USD, USD year
        designated by the column `usd_year`)
    """

    wfb_year = _get_wfb_yr(directory, 2020, 2020)

    # gathering soups
    soups = helper_wfb_gather_soups(directory, print_ver=True)

    # population
    pop_collect = []
    for soup in soups:
        name = helper_wfb_2020(soup)
        if name in no_info_names:
            continue

        pop_text = (
            soup.text[
                soup.text.find("People and Society ::") : soup.text.find("Nationality:")
            ]
            .split("Population:\n")[1]
            .replace("\n", " ")
        )
        if ("no indigenous" in pop_text) or ("uninhabited" in pop_text):
            pop_val, pop_year = 0, 2020
        else:
            if "note" in pop_text:
                pop_text = pop_text.split("note")[0]

            if name in ["Akrotiri", "Dhekelia"]:
                continue
            elif name == "European Union":
                pop_val = int(
                    pop_text.split("rank by population:")[0]
                    .split()[-1]
                    .replace(",", "")
                )
                pop_year = 2020
            else:
                pop_val = (
                    pop_text.replace(name, "")
                    .replace("(mostly African)", "")
                    .split(" (")[0]
                    .split()
                )
                if pop_val[-1] == "million":
                    pop_val = round(float(pop_val[-2].replace(",", "")) * 1e6)
                else:
                    pop_val = pop_val[-1].replace(",", "")
                pop_val = int(pop_val)
                pop_year = pop_text.replace("(mostly African)", "").split(" (")[1]
                split_by = ")"
                if "est. est.)" in pop_year:
                    split_by = "est. est.)"
                elif "est.)" in pop_year:
                    split_by = "est.)"

                pop_year = int(
                    [x for x in pop_year.split(split_by)[0].split(" ") if len(x) > 0][
                        -1
                    ]
                )

        pop_collect += [[name, pop_val, pop_year]]

    pop_collect = pd.DataFrame(pop_collect, columns=["country", "pop", "year"])
    pop_collect["wfb_year"] = wfb_year

    gdp_str_first = "GDP (purchasing power parity) - real:"

    # GDP and GDP per capita
    gdp_collect, gdppc_collect = [], []
    for soup in soups:
        name = helper_wfb_2020(soup)
        if name in no_info_names + ["Gaza Strip"]:
            continue

        # GDP (not GDPpc) information
        gdp_info_all = (
            soup.text[
                soup.text.find(gdp_str_first) : soup.text.find("Gross national saving:")
            ]
            .replace("\n", " ")
            .split("GDP (official exchange rate):")
        )

        gdp_info = gdp_info_all[0].replace(gdp_str_first, "")
        if "NA" in gdp_info:
            continue

        if len(gdp_info) > 0:
            note = None
            if ("note: " in gdp_info) and (name != "Saint Pierre and Miquelon"):
                gdp_info = gdp_info.split("note: ")
                gdp_info, note = gdp_info[0], gdp_info[1:]

            if "country comparison to" in gdp_info:
                gdp_info = gdp_info.split("country comparison to")[0]

            gdp_info = [
                r.strip()
                for r in re.split(r"\(|\)|more", gdp_info)
                if len(r.strip()) > 0
            ]

            if len(gdp_info) > 0:
                ix = np.array(
                    [fx for fx, f in enumerate(gdp_info) if f.startswith("$")],
                    dtype=int,
                )
                gdp_vals = np.array(gdp_info)[ix]
                gdp_dates = np.array(gdp_info)[ix + 1]
                gdp_vals = [helper_wfb_million_cleaner(x) for x in gdp_vals]
                gdp_yrs = helper_fy_cleaner(
                    [x.replace("est.", "").strip() for x in gdp_dates]
                )

                usd_year_assumed = "usd_year_assumed"
                if note is not None:
                    note = re.sub(r"[a-zA-Z]| ", "", note[0])
                    if note[0] == ";":
                        note = note[1:]
                    elif (";" in note) or ("-" in note):
                        note = re.split(r";|-", note)[0]

                    if (":" in note) and (wfb_year != 2020):
                        note = note.split(":")[0]

                    gdp_usd_yrs = np.array([int(note.replace(".", ""))] * len(gdp_yrs))
                    usd_year_assumed = "usd_year_original"
                else:
                    gdp_usd_yrs = np.array(gdp_yrs).astype(int)
                append_this = []
                for lx, yr in enumerate(gdp_yrs):
                    append_this.append(
                        [name, yr, gdp_usd_yrs[lx], gdp_vals[lx], usd_year_assumed]
                    )
                gdp_collect += append_this

        # GDPpc information
        gdppc_info = gdp_info_all[-1].split("GDP - per capita (PPP):")[-1]
        if len(gdppc_info.strip()) > 0:
            if "country comparison" not in gdppc_info:
                if "GDP - composition, by sector of origin" in gdppc_info:
                    gdppc_info = gdppc_info.split(
                        "GDP - composition, by sector of origin"
                    )[0]
            else:
                gdppc_info = gdppc_info.split("country comparison")[0]

            for string in ["Ease of Doing Business", "GDP - composition, by sector"]:
                if string in gdppc_info:
                    gdppc_info = gdppc_info.split(string)[0]

            if "NA" in gdppc_info:
                continue

            note = None
            if "note:" in gdppc_info:
                gdppc_info, note = gdppc_info.split("note:")

            gdppc_info = np.array(
                [
                    x
                    for x in re.split(r"\(|\)", gdppc_info)
                    if len(x.replace(" ", "")) > 0
                ]
            )
            ix = np.array(
                [fx for fx, f in enumerate(gdppc_info) if "$" in f], dtype=int
            )
            gdppc_vals, gdppc_years = gdppc_info[ix], gdppc_info[ix + 1]
            gdppc_vals = [int(re.sub(r"^(.*)\$|,", "", x.strip())) for x in gdppc_vals]
            gdppc_years = helper_fy_cleaner(
                [x.strip().replace(" est.", "") for x in gdppc_years]
            )

            usd_year_assumed = "usd_year_assumed"
            gdppc_usd_years = gdppc_years
            if (note is not None) and (name != "West Bank"):
                gdppc_usd_years = [int(re.sub(r"[a-zA-Z]|\.", "", note).strip())] * len(
                    gdppc_years
                )
                usd_year_assumed = "usd_year_orig"

            append_this = []
            for lx, yr in enumerate(gdppc_usd_years):
                append_this.append(
                    [name, gdppc_years[lx], yr, gdppc_vals[lx], usd_year_assumed]
                )
            gdppc_collect += append_this

    # organizing in pandas.DataFrame format
    gdp_columns = ["country", "year", "usd_year", "gdp", "usd_year_source"]
    gdp_collect = pd.DataFrame(gdp_collect, columns=gdp_columns)
    gdp_collect["wfb_year"] = wfb_year

    gdp_columns[3] = "gdppc"
    gdppc_collect = pd.DataFrame(gdppc_collect, columns=gdp_columns)
    gdppc_collect["wfb_year"] = wfb_year

    # fixing Cote d'Ivoire name
    gdp_collect.loc[
        gdp_collect.country == "Cote d&#39;Ivoire", "country"
    ] = "Cote d'Ivoire"
    gdppc_collect.loc[
        gdppc_collect.country == "Cote d&#39;Ivoire", "country"
    ] = "Cote d'Ivoire"
    pop_collect.loc[
        pop_collect.country == "Cote d&#39;Ivoire", "country"
    ] = "Cote d'Ivoire"

    # manual cleaning to fix or drop unreliable data
    gdppc_error_ctries = [
        "Togo",
        "Zimbabwe",
        "Turkmenistan",
        "Venezuela",
        "Sierra Leone",
        "Kosovo",
        "Guinea-Bissau",
        "Benin",
        "Cote d'Ivoire",
        "Kuwait",
        "Niger",
        "Taiwan",
        "Germany",
    ]
    gdppc_collect = gdppc_collect.loc[
        ~(
            (gdppc_collect.country.isin(gdppc_error_ctries))
            & (gdppc_collect.year == 2017)
        ),
        :,
    ].copy()

    gdp_error_ctries = [
        x for x in gdppc_error_ctries if x not in ["Kosovo", "Sierra Leone", "Taiwan"]
    ]
    gdp_error_ctries += ["Mozambique", "Mauritania", "Pakistan", "Jordan"]
    gdp_collect = gdp_collect.loc[
        ~((gdp_collect.country.isin(gdp_error_ctries)) & (gdp_collect.year == 2017)), :
    ].copy()

    gdp_collect = _ASM_fix(gdp_collect)

    return pop_collect, gdp_collect, gdppc_collect


def organize_cia_wfb_2021_2022(year):
    """Organizes the population, GDP, and GDP per capita information from the CIA World
    Factbook (WFB) version 2022 (latest as of 2022/04/12) into `pandas.DataFrame` format

    Parameters
    ----------
    urls : array-like of str
        containing URLs to population, GDP (PPP), and GDP per capita (PPP) information
        (in that order) from the latest 2022 version of CIA WFB

    Returns
    -------
    pop_collect : pandas.DataFrame
        containing population information (units in ones of people)
    gdp_collect : pandas.DataFrame
        containing PPP GDP information (units in millions of USD, USD year designated
        by the column `usd_year`)
    gdppc_collect : pandas.DataFrame
        containing PPP GDP per capita information (units in ones of USD, USD year
        designated by the column `usd_year`)
    """

    if year == 2021:
        urls = CIA_WFB_2021_URLS
    elif year == 2022:
        urls = CIA_WFB_CURRENT_URLS
    else:
        raise ValueError(year)

    soups = [BSoup(requests.get(url).content, "html.parser") for url in urls]

    # population
    lines = soups[0].find_all("li")
    pop_collect = []
    for li in lines:
        value = li.find_all("p")
        if len(value) == 0:
            continue
        country = li.find("a").text
        if (
            (country in ["Akrotiri", "Dhekelia", "European Union"])
            or ("uninhabited" in value[0].text)
            or ("no indige" in value[0].text)
        ):
            continue

        value = value[0].text.split("(")

        num = helper_wfb_million_cleaner(value[0])

        yr = int([x for x in value[1:] if "est." in x][0].split("est.")[0].strip()[-4:])
        pop_collect.append([country, num, yr])

    pop_collect = pd.DataFrame(pop_collect, columns=["country", "pop", "year"])
    pop_collect["wfb_year"] = year

    # GDP and GDP per capita
    swap_phrases = [
        "GDP estimate includes US subsidy;",
        "note: supplemented by annual payments from France of about \$60 million",
        "the war-driven deterioration of the economy resulted in a disappearance "
        "of quality national level statistics in the 2012-13 period",
    ]
    gdp_collect, gdppc_collect = [], []
    for i, soup in enumerate(soups[1:]):
        lines = soup.find_all("li")
        for li in lines:
            value = li.find_all("p")
            if len(value) == 0:
                continue
            country = li.find("a").text
            if country in ["Akrotiri", "Dhekelia"]:
                continue

            if ("NA" in value[0].text) or ("see entry for" in value[0].text):
                continue

            value = re.sub(r"|".join(swap_phrases), "", value[0].text)

            if country == "North Macedonia":
                value = re.split(r"\(|\)", value.split("; Macedonia has a large")[0])
            elif country == "Korea, North":
                value = re.split(r"\(|\)", value.split("North Korea")[0])
            else:
                value = re.split(r"\(|\)", value)

            nums, years = value[0::2], value[1::2]
            usd_year_from_note = [
                re.split(r"\$", x)[0] for x in nums if x.strip()[0:4].lower() == "note"
            ]

            if len(usd_year_from_note) > 0:
                usd_year_from_note = usd_year_from_note[0]
                nums = [
                    x.replace(usd_year_from_note, "").strip()
                    for x in nums
                    if x != usd_year_from_note
                ]
            nums = [x for x in nums if len(x.strip()) > 0]
            years = [x for x in years if len(x.strip()) > 0]

            note_in = 0
            numvals, years_cleaned = [], []
            usd_years = []
            for k, num in enumerate(nums):
                if "FY" not in years[k]:
                    year = int(re.sub(r"\(|\)|\.|[a-zA-Z]", "", years[k]))
                else:
                    year = helper_fy_cleaner(years[k].replace("est.", "").strip())
                if "note:" in num:
                    note_in += 1
                    numval, usd_year = num.split("note:")
                    if i == 0:
                        try:
                            numval = helper_wfb_million_cleaner(numval)
                        except:
                            raise ValueError(nums)
                    else:
                        numval = int(re.sub(r"\$|\,| ", "", numval))
                    usd_year = int(re.sub(r"\:|[a-zA-Z]", "", usd_year).strip())
                else:
                    if i == 0:
                        numval = helper_wfb_million_cleaner(num)
                    else:
                        numval = int(re.sub(r"\$|\,| ", "", num))
                    usd_year = year
                usd_years.append(usd_year)
                years_cleaned.append(year)
                numvals.append(numval)

            if (note_in == 0) and (len(usd_year_from_note) > 0):
                usd_year = int(
                    re.sub(r"[a-zA-Z]|\:|\.", "", usd_year_from_note).strip()
                )
                usd_years = [usd_year] * len(years_cleaned)

            if i == 0:
                gdp_df = pd.DataFrame(
                    {"year": years_cleaned, "usd_year": usd_years, "gdp": numvals}
                )
                gdp_df["country"] = country
                gdp_collect.append(gdp_df)
            else:
                gdppc_df = pd.DataFrame(
                    {"year": years_cleaned, "usd_year": usd_years, "gdppc": numvals}
                )
                gdppc_df["country"] = country
                gdppc_collect.append(gdppc_df)

    gdppc_collect = pd.concat(gdppc_collect, axis=0)
    gdp_collect = pd.concat(gdp_collect, axis=0)
    gdppc_collect["wfb_year"], gdp_collect["wfb_year"] = year, year

    gdp_collect = _ASM_fix(gdp_collect)

    return pop_collect, gdp_collect, gdppc_collect


def organize_gather_cia_wfb_2000_2022(
    home_dir, ccode_mapping, years=list(range(2000, 2023))
):
    """Cleaning all CIA WFB versions, from 2000 to 2022, and gathering them in list
    format (one list each for population, GDP, and GDP per capita). 2021 version is
    not available from the archive, and we skip this version.

    Parameters
    ----------
    years : array-like of int
        containing the version years to be cleaned; default runs from 2000 to 2020.

    Returns
    -------
    cia_pop_gather : list of pandas.DataFrame
        containing population data from the oldest version to the newest (data is in
        ones of people)
    cia_gdp_gather : list of pandas.DataFrame
        containing GDP data from the oldest version to the newest (data is in millions
        of USD)
    cia_gdppc_gather : list of pandas.DataFrame
        containing GDP per capita data from the oldest version to the newest (data is
        in ones of USD)

    """

    years = np.sort(years)
    assert (years.max() <= 2022) and (
        years.min() >= 2000
    ), "Only cleans versions 2000 to 2020."

    cia_gdp_gather = []
    cia_pop_gather = []
    cia_gdppc_gather = []
    for yr in tqdm(years):
        directory = home_dir / "factbook-{}".format(yr)
        if yr in [2000, 2001]:
            pop_df, gdp_df = organize_cia_wfb_2000_2001(directory)
            gdppc_df = gdp_df.copy()[["country", "year", "gdppc", "wfb_year"]]
            gdp_df = gdp_df[["country", "year", "gdp", "wfb_year"]]
        elif yr in [2002, 2003, 2004]:
            pop_df, gdp_df, gdppc_df = organize_cia_wfb_2002_2004(directory)
        elif yr in range(2005, 2009):
            pop_df, gdp_df, gdppc_df = organize_cia_wfb_2005_2008(directory)
        elif yr in range(2009, 2013):
            pop_df, gdp_df, gdppc_df = organize_cia_wfb_2009_2012(directory)
        elif yr in [2013, 2014]:
            pop_df, gdp_df, gdppc_df = organize_cia_wfb_2013_2014(directory)
        elif yr == 2015:
            pop_df, gdp_df, gdppc_df = organize_cia_wfb_2015(directory)
        elif yr in [2016, 2017]:
            pop_df, gdp_df, gdppc_df = organize_cia_wfb_2016_2017(directory)
        elif yr in [2018, 2019]:
            pop_df, gdp_df, gdppc_df = organize_cia_wfb_2018_2019(directory)
        elif yr == 2020:
            pop_df, gdp_df, gdppc_df = organize_cia_wfb_2020(directory)
        elif yr in [2021, 2022]:
            pop_df, gdp_df, gdppc_df = organize_cia_wfb_2021_2022(yr)
        else:
            raise ValueError(yr)

        if "usd_year" not in gdp_df.columns:
            gdp_df["usd_year"] = gdp_df["year"]
        if "usd_year" not in gdppc_df.columns:
            gdppc_df["usd_year"] = gdppc_df["year"]

        # fix france
        def _fix_france(df):
            return df.replace({"country": {"Francetotal: ": "France"}})

        pop_df, gdp_df, gdppc_df = list(map(_fix_france, [pop_df, gdp_df, gdppc_df]))

        # join
        pop_df = pop_df.join(ccode_mapping, on="country", how="left")
        gdp_df = gdp_df.join(ccode_mapping, on="country", how="left")
        gdppc_df = gdppc_df.join(ccode_mapping, on="country", how="left")

        def _check_no_missing(df):
            assert np.isin(
                df[df.ccode.isnull()].country.unique(), ["European Union", "World"]
            ).all()

        list(map(_check_no_missing, [pop_df, gdp_df, gdppc_df]))

        # manual cleaning after Palestine (West Bank + Gaza Strip)
        if "PSE" in pop_df.ccode.values:
            pse_df = pop_df.loc[pop_df.ccode == "PSE", :].reset_index(drop=True)
            pse_df = pse_df.groupby(["ccode", "year"]).sum()[["pop"]].reset_index()
            pse_df["country"] = "Palestine"
            pse_df["wfb_year"] = yr
            pop_df = pd.concat(
                [pse_df, pop_df.loc[pop_df.ccode != "PSE", :].copy()], axis=0
            ).reset_index(drop=True)
        if "PSE" in gdp_df.ccode.values:
            pse_df = gdp_df.loc[gdp_df.ccode == "PSE", :].reset_index(drop=True)
            pse_df = (
                pse_df.groupby(["ccode", "year", "usd_year"])
                .sum()[["gdp"]]
                .reset_index()
            )
            pse_df["country"] = "Palestine"
            pse_df["wfb_year"] = yr
            gdp_df = pd.concat(
                [pse_df, gdp_df.loc[gdp_df.ccode != "PSE", :].copy()], axis=0
            ).reset_index(drop=True)
        if "PSE" in gdppc_df.ccode.values:
            # getting those that do not have more than 1 ccode-year observations
            pse_df = gdppc_df.loc[gdppc_df.ccode == "PSE", :].reset_index(drop=True)
            pse_df["counter"] = 1
            pse_counter = (
                pse_df.groupby(["ccode", "year", "usd_year"])
                .sum()[["counter"]]
                .reset_index()
            )
            pse_df.drop(["counter"], axis=1, inplace=True)
            pse_df = pse_df.merge(
                pse_counter, on=["ccode", "year", "usd_year"], how="left"
            )
            pse_df = pse_df.loc[
                pse_df.counter == 1, ["ccode", "year", "gdppc", "usd_year", "wfb_year"]
            ]
            pse_df["country"] = "Palestine"
            gdppc_df = pd.concat(
                [pse_df, gdppc_df.loc[gdppc_df.ccode != "PSE", :].copy()], axis=0
            ).reset_index(drop=True)

        cia_pop_gather.append(pop_df)
        cia_gdp_gather.append(gdp_df)
        cia_gdppc_gather.append(gdppc_df)

    return list(
        map(_merge_all_years, (cia_pop_gather, cia_gdp_gather, cia_gdppc_gather))
    )


def _merge_all_years(dfs):
    dtypes = {"year": "uint16", "usd_year": "uint16", "wfb_year": "uint16"}
    all_dfs = pd.concat(dfs).dropna(subset="ccode")

    varname = [x for x in ["gdp", "pop", "gdppc"] if x in all_dfs.columns][0]

    # fix Cyprus (years that different data sets started referring to the combined
    # region)
    if varname == "pop":
        yr = 2010
    elif varname == "gdp":
        yr = 2008
    elif varname == "gdppc":
        yr = 2009

    all_dfs.loc[
        (all_dfs.ccode == "CYP") & (all_dfs.wfb_year >= yr), "ccode"
    ] = "CYP+ZNC"

    all_dfs = all_dfs.loc[all_dfs[varname] != 0, :].reset_index(drop=True)
    all_dfs = (
        all_dfs.astype({k: v for k, v in dtypes.items() if k in all_dfs.columns})
        .drop(columns=["usd_year_source", "country"], errors="ignore")
        .sort_values(["ccode", "year", "wfb_year"])
    )
    assert all_dfs.notnull().all().all(), all_dfs.notnull().all()

    return all_dfs.groupby(["ccode", "year"]).last()
