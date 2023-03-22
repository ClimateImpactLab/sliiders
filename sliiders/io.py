from pathlib import Path

import dask.dataframe as ddf
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rio
import xarray as xr

from sliiders.settings import STORAGE_OPTIONS


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


def save(obj, path, *args, **kwargs):
    if path.suffix == ".zarr":
        meth = "to_zarr"
    elif path.suffix == ".parquet":
        meth = "to_parquet"
    else:
        raise ValueError(type(obj))
    getattr(obj, meth)(str(path), *args, storage_options=STORAGE_OPTIONS, **kwargs)


def open_zarr(path, **kwargs):
    return xr.open_zarr(str(path), storage_options=STORAGE_OPTIONS, **kwargs)


def read_parquet_dask(path, **kwargs):
    return ddf.read_parquet(str(path), storage_options=STORAGE_OPTIONS, **kwargs)


def _fuseify(path):
    return str(path).replace("gs://", "/gcs/")


def save_geoparquet(obj, path, **kwargs):
    _path = _fuseify(path)
    _generate_parent_fuse_dirs(_path)
    obj.to_parquet(_path, **kwargs)


def read_shapefile(path, **kwargs):
    _path = _fuseify(path)
    _generate_parent_fuse_dirs(_path)
    return gpd.read_file(_path, **kwargs)


def save_shapefile(obj, path, **kwargs):
    _path = _fuseify(path)
    _generate_parent_fuse_dirs(_path)
    return obj.to_file(_path, **kwargs)


def _generate_parent_fuse_dirs(path):
    return Path(path).parent.mkdir(exist_ok=True, parents=True)


def open_rasterio(path, **kwargs):
    _path = _fuseify(path)
    _generate_parent_fuse_dirs(_path)
    return rio.open_rasterio(_path, **kwargs)


def read_transform(path):
    return rasterio.open(_fuseify(path)).transform


def open_dataset(path, **kwargs):
    _path = _fuseify(path)
    _generate_parent_fuse_dirs(_path)
    return xr.open_dataset(_path, **kwargs)


def open_dataarray(path, **kwargs):
    _path = _fuseify(path)
    _generate_parent_fuse_dirs(_path)
    return xr.open_dataarray(_path, **kwargs)


def save_to_zarr_region(ds_in, store, already_aligned=False):
    ds_out = open_zarr(store, chunks=None)

    # convert dataarray to dataset if needed
    if isinstance(ds_in, xr.DataArray):
        assert len(ds_out.data_vars) == 1
        ds_in = ds_in.to_dataset(name=list(ds_out.data_vars)[0])

    # align
    for v in ds_in.data_vars:
        ds_in[v] = ds_in[v].transpose(*ds_out[v].dims).astype(ds_out[v].dtype)

    # find appropriate regions
    alignment_dims = {}
    regions = {}
    for r in ds_in.dims:
        if len(ds_in[r]) == len(ds_out[r]):
            alignment_dims[r] = ds_out[r].values
            continue
        valid_ixs = np.arange(len(ds_out[r]))[ds_out[r].isin(ds_in[r].values).values]
        n_valid = len(valid_ixs)
        st = valid_ixs[0]
        end = valid_ixs[-1]
        assert (
            end - st == n_valid - 1
        ), f"Indices are not continuous along dimension {r}"
        regions[r] = slice(st, end + 1)

    # align coords
    if not already_aligned:
        ds_in = ds_in.sel(alignment_dims)

    save(ds_in.drop_vars(ds_in.coords), store, region=regions)
