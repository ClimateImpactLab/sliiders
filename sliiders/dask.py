import os
import zipfile

from dask.utils import tmpfile


def upload_pkg(client, pkg_dir):
    with tmpfile(extension="zip") as f:
        zipf = zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED)
        for root, dirs, files in os.walk(pkg_dir):
            for file in files:
                zipf.write(
                    os.path.join(root, file),
                    os.path.relpath(
                        os.path.join(root, file), os.path.join(pkg_dir, "..")
                    ),
                )
        zipf.close()
        client.upload_file(f)
