import os
from pathlib import Path
from zipfile import ZipFile

from dask_gateway import GatewayCluster

import sliiders


def _zipdir(path, zip_filename):
    # ziph is zipfile handle

    # Create a ZIP file
    with ZipFile(zip_filename, "w") as ziph:
        for root, dirs, files in os.walk(path):
            for file in files:
                # Create a relative path for files to preserve the directory structure
                # within the ZIP archive. This relative path is based on the directory
                # being zipped, so files are stored in the same structure.
                relative_path = os.path.relpath(
                    os.path.join(root, file), os.path.join(path, "..")
                )
                ziph.write(os.path.join(root, file), arcname=relative_path)


def start_cluster(idle_timeout=3600, n_workers=None, profile="micro"):
    cluster = GatewayCluster(idle_timeout=idle_timeout, profile=profile)
    client = cluster.get_client()
    upload_sliiders(client)
    return client, cluster


def upload_sliiders(client, restart_client=True):
    """Upload a local package to Dask Workers. After calling this function, the package
    contained at ``pkg_dir`` will be available on all workers in your Dask cluster,
    including those that are instantiated afterward. This package will take priority
    over any existing packages of the same name.

    Parameters
    ----------
    client : :py:class:`distributed.Client`
        The client object associated with your Dask cluster's scheduler.
    pkg_dir : str or Path-like
        Path to the package you wish to zip and upload
    **kwargs
        Passed directly to :py:class:`distributed.diagnostics.plugin.UploadDirectory`
    """
    package_dir = Path(sliiders.__file__).parent
    zip_filename = "/tmp/sliiders.zip"  # Output ZIP file name
    _zipdir(package_dir, zip_filename)
    client.upload_file(zip_filename)
