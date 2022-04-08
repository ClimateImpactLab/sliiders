from pathlib import Path

import numpy as np
import pandas as pd

from .gcs import FS, fuse_to_gcsmap

# Versions
GLOBAL_PROTECTED_AREAS_VERS = "v0.2"
LEVEES_VERS = "v0.2"
GPW_VERS = "v4rev11"
LANDSCAN_YEAR = "2019"
LANDSCAN_VERS = f"LandScan Global {LANDSCAN_YEAR}"
GADM_VERS = "gadm36"
LITPOP_VERS = "LitPop_v1_2"
LITPOP_DATESTAMP = "20220118"
GEG15_VERS = "v0.1"
EXPOSURE_BLENDED_VERS = "v0.5"
EXPOSURE_BINNED_VERS = "v0.14"
COUNTRY_LEVEL_TABLE_VERS = "v0.10"
DATUM_CONVERSION_VERS = "v0.3"
SLIIDERS_VERS = "v1.0"
PWT_DATESTAMP = "20220328"
MPD_DATESTAMP = "20220329"
WB_WDI_DATESTAMP = "20220329"
ALAND_STATISTICS_DATESTAMP = "20220329"
GWDB_DATESTAMP = "20220321"
OECD_DATESTAMP = "20220329"
UN_AMA_DATESTAMP = "20220329"
IMF_WEO_VERS = "October_2021"
UN_WPP_VERS = "2019"
IIASA_PROJECTIONS_DOWNLOAD_VERS = "2018"

# Definitions
SPATIAL_WARNINGS_TO_IGNORE = [
    "CRS mismatch between the CRS",
    "Geometry is in a geographic CRS",
    "initial implementation of Parquet.",
    "Iteration over",
    "__len__ for multi-part geometries",
    "The array interface is deprecated",
    "Only Polygon objects have interior rings",
]

# SLIIDERS-SLR PARAMS
LOCALIZESL_COREFILES = {
    "SLRProjections190726core_SEJ_full": ["L", "H"],
    "SLRProjections170113GRIDDEDcore": [None],
    "SLRProjections200204GRIDDEDcore_D20": [None],
    "SLRProjections210628GRIDDEDcore_SROCC": [None],
}
LOCALIZESL_REV = "c9b020a0f9409cde3f6796ca936f229c90f7d5c6"

# Aland Islands, Western Sahara, Libya, Palestine, South Sudan, Syria, Kosovo
ISOS_IN_GEG_NOT_LITPOP = ["ALA", "ESH", "LBY", "PSE", "SSD", "SYR", "XKX"]

# for organizing scenarios
SSP_PROJ_ORG_SER = pd.Series(
    {
        "SSP1_v9_130219": "SSP1",
        "SSP1_v9_130325": "SSP1",
        "SSP1_v9_130424": "SSP1",
        "SSP1_v9_130115": "SSP1",
        "SSP2_v9_130219": "SSP2",
        "SSP2_v9_130325": "SSP2",
        "SSP2_v9_130424": "SSP2",
        "SSP2_v9_130115": "SSP2",
        "SSP3_v9_130219": "SSP3",
        "SSP3_v9_130325": "SSP3",
        "SSP3_v9_130424": "SSP3",
        "SSP3_v9_130115": "SSP3",
        "SSP4_v9_130219": "SSP4",
        "SSP4_v9_130325": "SSP4",
        "SSP4_v9_130424": "SSP4",
        "SSP4_v9_130115": "SSP4",
        "SSP4d_v9_130115": "SSP4",
        "SSP5_v9_130219": "SSP5",
        "SSP5_v9_130325": "SSP5",
        "SSP5_v9_130424": "SSP5",
        "SSP5_v9_130115": "SSP5",
        "IIASA GDP": "IIASA",
        "IIASA-WiC POP": "IIASA-WiC",
        "NCAR": "NCAR",
        "OECD Env-Growth": "OECD",
        "PIK GDP-32": "PIK",
    }
)
SCENARIOS = [
    ("SSP1", "OECD"),
    ("SSP1", "IIASA"),
    ("SSP2", "OECD"),
    ("SSP2", "IIASA"),
    ("SSP3", "OECD"),
    ("SSP3", "IIASA"),
    ("SSP4", "OECD"),
    ("SSP4", "IIASA"),
    ("SSP5", "OECD"),
    ("SSP5", "IIASA"),
]

# country ISO code groupings
EXCLUDED_ISOS = ["ATA", "XCA"]

FRA_MSNG = [
    "REU",
    "WLF",
    "ATF",
    "SPM",
    "AND",
    "BLM",
    "GLP",
    "GUF",
    "MAF",
    "MCO",
    "MTQ",
    "MYT",
    "NCL",
    "PYF",
]
USA_MSNG = [
    "ASM",
    "GUM",
    "LIE",
    "MNP",
    "PRK",
    "SOM",
    "MHL",
    "FSM",
    "ERI",
    "CUB",
    "UMI",
    "VIR",
]
PPP_CCODE_IF_MSNG = {
    "AUS": ["CCK", "CXR", "HMD", "NFK"],
    "DNK": ["GRL", "FRO"],
    "FRA": FRA_MSNG,
    "FIN": ["ALA"],
    "ITA": ["VAT", "SMR"],
    "USA": USA_MSNG,
    "MAR": ["ESH"],
    "CUW": ["BES"],
    "NZL": ["NIU", "COK", "TKL"],
    "NOR": ["BVT", "SJM"],
    "GBR": ["IMN", "FLK", "GGY+JEY", "GIB", "PCN", "SGS", "SHN", "GGY", "JEY"],
    "ESH": ["MAR"],
}

PWT_ISOS = [
    "ABW",
    "AGO",
    "AIA",
    "ALB",
    "ARE",
    "ARG",
    "ARM",
    "ATG",
    "AUS",
    "AUT",
    "AZE",
    "BDI",
    "BEL",
    "BEN",
    "BFA",
    "BGD",
    "BGR",
    "BHR",
    "BHS",
    "BIH",
    "BLR",
    "BLZ",
    "BMU",
    "BOL",
    "BRA",
    "BRB",
    "BRN",
    "BTN",
    "BWA",
    "CAF",
    "CAN",
    "CHE",
    "CHL",
    "CHN",
    "CIV",
    "CMR",
    "COD",
    "COG",
    "COL",
    "COM",
    "CPV",
    "CRI",
    "CUW",
    "CYM",
    "CYP",
    "CZE",
    "DEU",
    "DJI",
    "DMA",
    "DNK",
    "DOM",
    "DZA",
    "ECU",
    "EGY",
    "ESP",
    "EST",
    "ETH",
    "FIN",
    "FJI",
    "FRA",
    "GAB",
    "GBR",
    "GEO",
    "GHA",
    "GIN",
    "GMB",
    "GNB",
    "GNQ",
    "GRC",
    "GRD",
    "GTM",
    "GUY",
    "HKG",
    "HND",
    "HRV",
    "HTI",
    "HUN",
    "IDN",
    "IND",
    "IRL",
    "IRN",
    "IRQ",
    "ISL",
    "ISR",
    "ITA",
    "JAM",
    "JOR",
    "JPN",
    "KAZ",
    "KEN",
    "KGZ",
    "KHM",
    "KNA",
    "KOR",
    "KWT",
    "LAO",
    "LBN",
    "LBR",
    "LCA",
    "LKA",
    "LSO",
    "LTU",
    "LUX",
    "LVA",
    "MAC",
    "MAR",
    "MDA",
    "MDG",
    "MDV",
    "MEX",
    "MKD",
    "MLI",
    "MLT",
    "MMR",
    "MNE",
    "MNG",
    "MOZ",
    "MRT",
    "MSR",
    "MUS",
    "MWI",
    "MYS",
    "NAM",
    "NER",
    "NGA",
    "NIC",
    "NLD",
    "NOR",
    "NPL",
    "NZL",
    "OMN",
    "PAK",
    "PAN",
    "PER",
    "PHL",
    "POL",
    "PRT",
    "PRY",
    "PSE",
    "QAT",
    "ROU",
    "RUS",
    "RWA",
    "SAU",
    "SDN",
    "SEN",
    "SGP",
    "SLE",
    "SLV",
    "SRB",
    "STP",
    "SUR",
    "SVK",
    "SVN",
    "SWE",
    "SWZ",
    "SXM",
    "SYC",
    "SYR",
    "TCA",
    "TCD",
    "TGO",
    "THA",
    "TJK",
    "TKM",
    "TTO",
    "TUN",
    "TUR",
    "TWN",
    "TZA",
    "UGA",
    "UKR",
    "URY",
    "USA",
    "UZB",
    "VCT",
    "VEN",
    "VGB",
    "VNM",
    "YEM",
    "ZAF",
    "ZMB",
    "ZWE",
]

UNINHABITED_ISOS = ["ATF", "BVT", "CL-", "HMD", "IOT", "SGS"]
OTHER_ISOS = [
    "AFG",
    "ALA",
    "AND",
    "ASM",
    "BES",
    "BLM",
    "CCK",
    "COK",
    "CUB",
    "CXR",
    "ERI",
    "ESH",
    "FLK",
    "FRO",
    "FSM",
    "GGY",
    "GIB",
    "GLP",
    "GRL",
    "GUF",
    "GUM",
    "IMN",
    "JEY",
    "KIR",
    "KO-",
    "LBY",
    "LIE",
    "MAF",
    "MCO",
    "MHL",
    "MNP",
    "MTQ",
    "MYT",
    "NCL",
    "NFK",
    "NIU",
    "NRU",
    "PCN",
    "PLW",
    "PNG",
    "PRI",
    "PRK",
    "PYF",
    "REU",
    "SHN",
    "SJM",
    "SLB",
    "SMR",
    "SOM",
    "SPM",
    "SSD",
    "TKL",
    "TLS",
    "TON",
    "TUV",
    "UMI",
    "VAT",
    "VIR",
    "VUT",
    "WLF",
    "WSM",
]

ALL_ISOS = np.sort(np.union1d(PWT_ISOS, UNINHABITED_ISOS + OTHER_ISOS))
EXTENDED_ISOS = ["GGY+JEY", "CHI", "XKX"]
ALL_ISOS_EXTENDED = np.sort(np.union1d(ALL_ISOS, EXTENDED_ISOS))

# Dask image name
DASK_IMAGE = "gcr.io/rhg-project-1/pytc-image-devbase:latest"

# Constants
# Data
LITPOP_GRID_WIDTH = 1 / 120
GEG_GRID_WIDTH = 1 / 24
LANDSCAN_GRID_WIDTH = 1 / 120

EXPOSURE_BIN_WIDTH_V = 1 / 10  # meters
EXPOSURE_BIN_WIDTH_H = 1 / 10  # 10cm
HIGHEST_WITHELEV_EXPOSURE_METERS = 20
ELEV_CAP = HIGHEST_WITHELEV_EXPOSURE_METERS + 1  # "higher than coastal" value

## Spatial

# Area, in "square degrees", above which we will consider endorheic basins as protected areas
# N.B. this is an arbitrary choice (something more robust could use something like a bathtub model
# over a highly resolved elevation grid).
MIN_BASIN_TILE_DEGREE_AREA = 20.0

# minimum distance in degrees from the ocean to include an endorheic basin as
# a "protected area"
ENDORHEIC_BASIN_OCEAN_BUFFER = 0.2

MAX_VORONOI_COMPLEXITY = (
    40e6  # Maximum number of initial points in shapefile when generating Voronoi
)

# Width, in degrees, of squares in which to divide the shapes of administrative regions.
# The smaller shapes are more manageable and computationally efficient in many
# geometry-processing algorithms
DEFAULT_BOX_SIZE = 1.0

DENSIFY_TOLERANCE = 0.01
MARGIN_DIST = 0.001
ROUND_INPUT_POINTS = 6
SMALLEST_INTERIOR_RING = 1e-13

# What are the return periods (in years) we allow for retreat and protect standards
SVALS = np.array([10, 100, 1000, 10000])

# Paths and Directories
DIR_DATA = Path("/gcs/rhg-data/impactlab-rhg/coastal/sliiders")

DIR_DATA_RAW = DIR_DATA / "raw"
DIR_DATA_INT = DIR_DATA / "int"
DIR_RESULTS = DIR_DATA / "output"

DIR_EXPOSURE_RAW = DIR_DATA_RAW / "exposure"
DIR_EXPOSURE_INT = DIR_DATA_INT / "exposure"

DIR_LITPOP_RAW = DIR_EXPOSURE_RAW / "asset_value" / "litpop" / LITPOP_DATESTAMP
PATH_LITPOP_RAW = DIR_LITPOP_RAW / LITPOP_VERS / "LitPop_pc_30arcsec_*.csv"

DIR_GEG15_RAW = DIR_EXPOSURE_RAW / "asset_value" / "geg15"
DIR_GEG15_INT = DIR_EXPOSURE_INT / "asset_value" / "geg15" / GEG15_VERS
PATH_GEG15_INT = DIR_GEG15_INT / "gar_exp.parquet"

DIR_SLR_RAW = DIR_DATA_RAW / "slr"
DIR_SLR_INT = DIR_DATA_INT / "slr"

DIR_IFILES_RAW = DIR_SLR_RAW / "ifiles"
DIR_IFILES_INT = DIR_SLR_INT / "ifiles"
PATH_SLR_N_GCMS = fuse_to_gcsmap(DIR_SLR_INT / f"numGCMs_{SLIIDERS_VERS}.zarr", FS)

DIR_GEOG_RAW = DIR_DATA_RAW / "geography"
DIR_GEOG_INT = DIR_DATA_INT / "geography"

PATH_CIAM_2016 = fuse_to_gcsmap(
    DIR_DATA_RAW / "CIAM_2016" / "diaz2016_inputs_raw.zarr", FS
)

PATH_SLIIDERS_ECON = fuse_to_gcsmap(
    DIR_RESULTS / f"sliiders-econ-{SLIIDERS_VERS}.zarr", FS
)
PATH_SLIIDERS_SLR = fuse_to_gcsmap(
    DIR_RESULTS / f"sliiders-slr-{SLIIDERS_VERS}.zarr", FS
)

PATH_SEG_CENTROIDS = DIR_GEOG_INT / "gtsm_stations_thinned_ciam"

PATH_CIAM_COASTLINES = DIR_GEOG_INT / "ne_coastline_lines_CIAM_wexp_or_gtsm"

DIR_GTSM_STATIONS_TOTHIN = DIR_GEOG_RAW / "gtsm_stations_eur_tothin"

DIR_CIAM_VORONOI = DIR_GEOG_INT / "ciam_and_adm1_intersections" / EXPOSURE_BINNED_VERS
PATH_CIAM_ADM1_VORONOI_INTERSECTIONS = (
    DIR_CIAM_VORONOI / "ciam_and_adm1_intersections.parquet"
)

PATH_CIAM_ADM1_VORONOI_INTERSECTIONS_SHP = (
    DIR_CIAM_VORONOI / "ciam_and_adm1_intersections.shp"
)

DIR_SHAPEFILES = Path("/gcs/rhg-data/impactlab-rhg/spatial/shapefiles/source")

DIR_GADM = Path(DIR_SHAPEFILES / "gadm" / GADM_VERS)

PATH_GADM = DIR_GADM / f"{GADM_VERS}_levels" / f"{GADM_VERS}_levels.gpkg"
PATH_GADM_ADM1 = DIR_GADM / "adm1.parquet"
PATH_GADM_ADM0_VORONOI = DIR_GADM / "adm0_voronoi.parquet"
PATH_GADM_ADM1_VORONOI = DIR_GADM / "adm1_voronoi.parquet"

PATH_EXPOSURE_BLENDED = (
    DIR_EXPOSURE_INT
    / "asset_value"
    / "litpop"
    / EXPOSURE_BLENDED_VERS
    / "LitPop_pc_30arcsec.parquet"
)

PATH_NATURALEARTH_OCEAN = DIR_SHAPEFILES / "natural_earth" / "ne_10m_ocean"
DIR_HYDROBASINS_RAW = DIR_DATA_RAW / "hydrosheds" / "hydrobasins"

DIR_GLOBAL_PROTECTED_AREAS = (
    DIR_EXPOSURE_INT
    / "protected_locations"
    / "global"
    / "historical"
    / GLOBAL_PROTECTED_AREAS_VERS
)

PATH_US_MANUAL_PROTECTED_AREAS = (
    DIR_EXPOSURE_RAW
    / "protected_areas"
    / "usa"
    / "manual"
    / "us_manual_protected_areas.parquet"
)

PATH_MANUAL_PROTECTED_AREAS = (
    DIR_GLOBAL_PROTECTED_AREAS / "manual_global_basins.parquet"
)
PATH_GLOBAL_PROTECTED_AREAS = DIR_GLOBAL_PROTECTED_AREAS / "all_protected_areas.parquet"

DIR_WETLANDS_RAW = DIR_DATA_RAW / "wetlands_mangroves"
DIR_WETLANDS_INT = DIR_DATA_INT / "wetlands_mangroves"
PATH_GLOBCOVER_2009 = (
    DIR_WETLANDS_RAW
    / "Globcover2009_V2.3_Global"
    / "GLOBCOVER_L4_200901_200912_V2.3.tif"
)

PATH_GLOBAL_MANGROVES = (
    DIR_WETLANDS_RAW
    / "GMW_001_GlobalMangroveWatch_2016"
    / "01_Data"
    / "GMW_2016_v2.shp"
)

PATH_WETLANDS_INT = DIR_WETLANDS_INT / "wetlands.shp"

DIR_ELEVATION = Path("/gcs/rhg-data/impactlab-rhg/common_data/elevation")
DIR_ELEVATION_RAW = DIR_ELEVATION / "raw"
DIR_ELEVATION_INT = DIR_ELEVATION / "int"

PATH_SRTM15_PLUS = DIR_ELEVATION_RAW / "srtm15_plus" / "SRTM15_V2.3.nc"
DIR_MSS = DIR_ELEVATION_INT / "CoastalDEM_mss_corrected"
DIR_COASTALDEM = (
    DIR_ELEVATION_RAW / "climate_central" / "coastal_dem_30as" / "CoastalDEM_Global_30m"
)

DIR_LANDSCAN_RAW = DIR_EXPOSURE_RAW / "landscan"
DIR_LANDSCAN_INT = DIR_EXPOSURE_INT / "landscan" / f"ls{LANDSCAN_YEAR}"
PATH_LANDSCAN_INT = DIR_LANDSCAN_INT / "population.parquet"

DIR_EXPOSURE_BINNED = (
    DIR_EXPOSURE_INT / "asset_value" / "binned" / "global" / "historical"
)
DIR_EXPOSURE_BINNED_TMP = DIR_EXPOSURE_BINNED / "tmp"
DIR_EXPOSURE_BINNED_TMP_TILES = DIR_EXPOSURE_BINNED_TMP / "tiles"
DIR_EXPOSURE_BINNED_TMP_TILES_NOLAND = DIR_EXPOSURE_BINNED_TMP / "tiles_noland"
DIR_EXPOSURE_BINNED_TMP_TILES_SEGMENT_AREA = (
    DIR_EXPOSURE_BINNED_TMP / "tiles_segment_area"
)

PATH_EXPOSURE_TILE_LIST = DIR_EXPOSURE_BINNED / "tmp" / "meta" / "tile_list.parquet"

PATH_EXPOSURE_AREA_BY_CIAM_AND_ELEVATION = (
    DIR_EXPOSURE_BINNED / EXPOSURE_BINNED_VERS / "ciam_segs_area_by_elev.parquet"
)

PATH_EXPOSURE_BINNED_WITHOUTELEV = (
    DIR_EXPOSURE_BINNED
    / EXPOSURE_BINNED_VERS
    / "binned_exposure_withoutelev_base.parquet"
)

PATH_EXPOSURE_BINNED_WITHELEV = (
    DIR_EXPOSURE_BINNED / EXPOSURE_BINNED_VERS / "binned_exposure_withelev_base.parquet"
)

DIR_GEOG_DATUMS_RAW = DIR_GEOG_RAW / "datum_conversions"
DIR_GEOG_DATUMS_INT = DIR_GEOG_INT / "datum_conversions"

DIR_GEOG_DATUMS_EGM96_WGS84 = DIR_GEOG_DATUMS_RAW / "egm96"
DIR_GEOG_DATUMS_XGM2019e_WGS84 = DIR_GEOG_DATUMS_RAW / "xgm2019e"

PATH_GEOG_MDT_RAW = DIR_GEOG_RAW / "mdt" / "aviso_2018" / "mdt_cnes_cls18_global.nc"

PATH_GEOG_DATUMS_GRID = fuse_to_gcsmap(
    DIR_GEOG_DATUMS_INT / f"datum_conversions_gridded_{DATUM_CONVERSION_VERS}.zarr", FS
)

PATH_GTSM_SURGE = (
    DIR_DATA_RAW / "esl" / "CODEC_amax_ERA5_1979_2017_coor_mask_GUM_RPS.nc"
)

DIR_CCI_RAW = DIR_DATA_RAW / "cci"
PATH_EXPOSURE_WB_ICP = DIR_CCI_RAW / "world_bank_ICP_2017.csv"
PATH_EXPOSURE_LINCKE = DIR_CCI_RAW / "lincke_2021_country_input.csv"

# Various directories and paths for the country-level ("YPK") workflow
DIR_YPK_INT = DIR_EXPOSURE_INT / "ypk"
DIR_YPK_FINAL = DIR_YPK_INT / "finalized"
DIR_YPK_RAW = DIR_EXPOSURE_RAW / "ypk"
PATH_COUNTRY_LEVEL_EXPOSURE = DIR_YPK_FINAL / "gdp_gdppc_pop_capital_1950_2020.parquet"
PATH_COUNTRY_LEVEL_EXPOSURE_PROJ = (
    DIR_YPK_FINAL / "gdp_gdppc_pop_capital_proj_2010_2100.parquet"
)

DIR_UN_AMA_RAW = DIR_YPK_RAW / "un_ama" / UN_AMA_DATESTAMP
DIR_UN_WPP_RAW = DIR_YPK_RAW / "un_wpp" / UN_WPP_VERS
DIR_WB_WDI_RAW = DIR_YPK_RAW / "wb_wdi" / WB_WDI_DATESTAMP
DIR_OECD_REGIONS_RAW = DIR_YPK_RAW / "oecd_regions" / OECD_DATESTAMP
DIR_IIASA_PROJECTIONS = (
    DIR_YPK_RAW / "iiasa_projections" / IIASA_PROJECTIONS_DOWNLOAD_VERS
)
DIR_ALAND_STATISTICS_RAW = DIR_YPK_RAW / "asub" / ALAND_STATISTICS_DATESTAMP
PATH_GWDB2021_RAW = (
    DIR_YPK_RAW / "gwdb" / GWDB_DATESTAMP / "global-wealth-databook-2021.pdf"
)
PATH_PWT_RAW = DIR_YPK_RAW / "pwt" / PWT_DATESTAMP / "pwt_100.xlsx"
PATH_IMF_WEO_RAW = DIR_YPK_RAW / "imf_weo" / IMF_WEO_VERS / "WEO_iy_ratio_pop_gdp.xlsx"
PATH_MPD_RAW = DIR_YPK_RAW / "mpd" / MPD_DATESTAMP / "maddison_project.xlsx"
