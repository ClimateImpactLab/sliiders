from pathlib import Path

from distributed.diagnostics.plugin import UploadDirectory

from sliiders import __file__ as sliiders_path


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
    client.register_worker_plugin(
        UploadDirectory(
            Path(sliiders_path).parents[1],
            update_path=True,
            restart=restart_client,
            skip_words=(
                ".git",
                ".github",
                ".pytest_cache",
                "tests",
                "docs",
                "deploy",
                "notebooks",
                ".ipynb_checkpoints",
                "__pycache__",
                ".coverage",
                "dockerignore",
                ".gitignore",
                ".gitlab-ci.yml",
                ".gitmodules",
                "pyclaw.log",
            ),
        )
    )
