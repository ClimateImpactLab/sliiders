#!/usr/bin/env bash

set -e

SOURCE_URL="https://wcmc.io/GMW_2016"
DOWNLOAD_DIR="/gcs/rhg-data/impactlab-rhg/coastal/sliiders/raw/wetlands_mangroves"
DOCUMENTATION_PDF_URL="https://data.unep-wcmc.org/pdfs/45/GMW_001_Metadata.pdf?1560444488"

mkdir -p $DOWNLOAD_DIR

DOWNLOAD_PATH=${DOWNLOAD_DIR}/$(basename ${SOURCE_URL})
PDF_DOWNLOAD_PATH=${DOWNLOAD_DIR}/$(basename ${DOCUMENTATION_PDF_URL})

if [ ! -f "${DOWNLOAD_PATH}" ];
then
    wget "${SOURCE_URL}" -O "${DOWNLOAD_PATH}"
    unzip "${DOWNLOAD_PATH}" -d "${DOWNLOAD_DIR}"
fi

if [ ! -f "${PDF_DOWNLOAD_PATH}" ];
then
    wget "${DOCUMENTATION_PDF_URL}" -O "${PDF_DOWNLOAD_PATH}"
fi

echo "Global Mangrove Watch download complete"
