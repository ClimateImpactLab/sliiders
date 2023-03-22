#!/usr/bin/env bash

set -e

# GlobCover documentation available here: http://due.esrin.esa.int/page_globcover.php

GLOBCOVER_URL="http://due.esrin.esa.int/files/Globcover2009_V2.3_Global_.zip"
DOWNLOAD_DIR="/gcs/rhg-data/impactlab-rhg/coastal/sliiders/raw/wetlands_mangroves/Globcover2009_V2.3_Global"
DOCUMENTATION_PDF_URL="http://due.esrin.esa.int/files/GLOBCOVER2009_Validation_Report_2.2.pdf"

mkdir -p $DOWNLOAD_DIR

DOWNLOAD_PATH=${DOWNLOAD_DIR}/$(basename ${GLOBCOVER_URL})
PDF_DOWNLOAD_PATH=${DOWNLOAD_DIR}/$(basename ${DOCUMENTATION_PDF_URL})

if [ ! -f "${DOWNLOAD_PATH}" ];
then
    wget "${GLOBCOVER_URL}" -O "${DOWNLOAD_PATH}"
    unzip "${DOWNLOAD_PATH}" -d "${DOWNLOAD_DIR}"
fi

if [ ! -f "${PDF_DOWNLOAD_PATH}" ];
then
    wget "${DOCUMENTATION_PDF_URL}" -O "${PDF_DOWNLOAD_PATH}"
fi

echo "GlobCover download complete"
