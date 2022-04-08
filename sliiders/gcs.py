from pathlib import Path

import gcsfs

# in CI, no access to creds so we need to handle this case
try:
    FS = gcsfs.GCSFileSystem(
        project="rhg-data", token="/opt/gcsfuse_tokens/rhg-data.json"
    )
except FileNotFoundError:
    FS = None

import gcsfs

# in CI, no access to creds so we need to handle this case
try:
    FS = gcsfs.GCSFileSystem(
        project="rhg-data", token="/opt/gcsfuse_tokens/rhg-data.json"
    )
except FileNotFoundError:
    FS = None


def fuse_to_gcsmap(path, fs=FS):
    """Convert a path using the gcs FUSE file system into a mapper that can be used
    for a zarr store.

    Parameters
    ----------
    path : str or :class:`pathlib.Path`
        Path on GCS FUSE (i.e. starts with ``/gcs/``)
    fs : `:class:`gcsfs.GCSFileSystem`
        If None, will just return the path on GCS
    Returns
    -------
    :class:`fsspec.mapping.FSMap`
        Mapper to object store
    """

    # handle when fs is null by just returning the GCSFUSE path
    if fs is None:
        return Path(path)

    return fs.get_mapper("/".join(Path(path).parts[2:]), check=False)


def gcsmap_to_fuse(gcsmap):
    return Path("/gcs", gcsmap.root)


def fuse_to_url(path):
    return str(path).replace("/gcs/", "gs://")


def fuse_to_gspath(path):
    return str(path).replace("/gcs/", "")
