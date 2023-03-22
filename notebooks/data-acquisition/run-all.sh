#!/usr/bin/env bash

set -e

script_dir=$(dirname "$0")

bash "${script_dir}/01-globcover.sh"
bash "${script_dir}/02-mangroves.sh"
papermill "${script_dir}/03-download-sliiders-econ-input-data.ipynb" -
