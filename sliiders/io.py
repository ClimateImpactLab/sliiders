import geopandas as gpd
import pandas as pd
from cartopy.io import shapereader


def load_adm0_shpfiles(vector_types, resolution_m=10):
    """
    Load and return dictionary of geopandas dataframes from the Natural Earth
    repository.

    Parameters
    ----------
    vector_types : list of strings
        each string is a natural earth admin0 vector type.
        Vector type options can be found here:
        https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
    resolution_m : int
        Resolution of file to obtain. Must match one of those available via
        Natural Earth

    Returns
    -------
    dict of :py:class:`geopandas.Dataframe`
        Keys are vector types associated with the geopandas dataframe and values are
        Dataframes from the Naturalearth API, loaded from cache if possible.
    """
    return_dict = {}
    for vector_type in vector_types:
        return_dict[vector_type] = gpd.read_file(
            shapereader.natural_earth(
                resolution=f"{resolution_m}m",
                category="cultural",
                name="admin_0_{}".format(vector_type),
            )
        )
    return return_dict


def read_gdf(fpath):
    """Reads in the `.gdf` file located at `fpath` into `pandas.DataFrame`, assigns
    columns `lon` for longitude, `lat` for latitude, and `z` for data (such as geoid),
    then returns a `xarray.DataArray` containing the values of `z` and coordinates
    `lon` and `lat`.

    Parameters
    ----------
    fpath : pathlib.Path-like
        path where the `.gdf` file of interest is located at

    Returns
    -------
    xarray.DataArray
        containing values of `z` with coordinates `lon` and `lat`

    """

    return (
        pd.read_table(
            fpath,
            skiprows=36,
            names=["lon", "lat", "z"],
            delim_whitespace=True,
        )
        .set_index(["lon", "lat"])
        .z.to_xarray()
    )
