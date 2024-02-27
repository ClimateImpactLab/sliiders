from pathlib import Path

import numpy as np
import pandas as pd
from cloudpathlib import AnyPath, GSClient
from fsspec import filesystem

###############
# Data Versions
###############

# Raw/Input Data
COASTALDEM_VERS = "v2.1"
LANDSCAN_YEAR = "2022"
LANDSCAN_VERS = f"LandScan Global {LANDSCAN_YEAR}"
GADM_VERS = "4.1"
GADM_VERS_STR = "gadm_410"
LITPOP_VERS = "LitPop_v1_2"
LITPOP_DATESTAMP = "20220118"
GEG15_VERS = "v0.1"
PWT_DATESTAMP = "20240222"
MPD_DATESTAMP = "20220329"
WB_WDI_DATESTAMP = "20240222"
ALAND_STATISTICS_DATESTAMP = "20240222"
GWDB_DATESTAMP = "20240222"
GWDB_YEAR = 2023
OECD_DATESTAMP = "20240222"
UN_AMA_DATESTAMP = "20240222"
IMF_WEO_VERS = "October_2023"
UN_WPP_VERS = "2022"
CIA_WFB_VERS = "20240222"
ADB_KI_VERS = "2023"
NE_DOWNLOAD_VERS = "20240222"
INC_POP_AUX_DATESTAMP = "20240222"
GMW_YEAR = 2020
NLDB_DATESTAMP = "20240223"

# intermediate data / output data versions
MSS_DEM_VERS = "v2.1.1"
INC_POP_CLEANED_VERS = "20240222"
YPK_FINALIZED_VERS = "20240222"
EXPOSURE_BLENDED_VERS = "20221201"
EXPOSURE_BINNED_VERS = "20240223"
COUNTRY_LEVEL_TABLE_VERS = "v0.10"
DATUM_CONVERSION_VERS = "v0.3"
ELEV_TILE_LIST_VERS = "v0.4"
COMBINED_PROTECTED_AREAS_VERS = "20240223"
SLIIDERS_VERS = "v1.2"
COASTLINES_VERS = "20240223"

############
# Parameters
############

# Spatial
ASSET_VALUE_GRID_WIDTH = 1 / 120
GEG_GRID_WIDTH = 1 / 24
POP_GRID_WIDTH = 1 / 120

EXPOSURE_BIN_WIDTH_V = 1 / 10  # meters
HIGHEST_WITHELEV_EXPOSURE_METERS = 20

# Temporal
HISTORICAL_YEARS = np.arange(2000, 2021)
PROJ_YEARS = np.arange(2000, 2101)
SOCIOECONOMIC_SCALE_YR = 2019

# Return periods (in years) we allow for retreat and protect standards
SVALS = np.array([10, 100, 1000, 10000])

#######
# Paths
#######

STORAGE_OPTIONS = {}
DIR_SCRATCH = Path("/tmp/sliiders-scratch")
FS = filesystem("file")
DIR_DATA = Path("/tmp/sliiders/data")

DIR_DATA_RAW = DIR_DATA / "raw"
DIR_DATA_INT = DIR_DATA / "int"
DIR_RESULTS = DIR_DATA / "output"

DIR_EXPOSURE_RAW = DIR_DATA_RAW / "exposure"
DIR_EXPOSURE_INT = DIR_DATA_INT / "exposure"

DIR_LITPOP_RAW = DIR_EXPOSURE_RAW / "asset_value" / "litpop" / LITPOP_DATESTAMP
PATH_LITPOP_RAW = DIR_LITPOP_RAW / LITPOP_VERS / "LitPop_pc_30arcsec_*.csv"

DIR_GEG15_RAW = DIR_EXPOSURE_RAW / "asset_value" / "geg15"
DIR_GEG15_INT = DIR_EXPOSURE_INT / "asset_value" / "geg15" / GEG15_VERS
PATH_GEG15_RAW = str(DIR_GEG15_RAW / "gar-exp" / "gar_exp.shp").replace("gs:/", "/gcs")
PATH_GEG15_INT = DIR_GEG15_INT / "gar_exp.parquet"

DIR_GEOG_RAW = DIR_DATA_RAW / "geography"
DIR_GEOG_INT = DIR_DATA_INT / "geography"

PATH_CIAM_2016 = DIR_DATA_RAW / "CIAM_2016" / "diaz2016_inputs_raw.zarr"

PATH_SLIIDERS = DIR_RESULTS / f"sliiders-{SLIIDERS_VERS}.zarr"

PATH_GEOG_GTSM_SNAPPED = (
    DIR_GEOG_INT / "gtsm_stations_ciam_ne_coastline_snapped.parquet"
)

PATH_SEG_PTS_MANUAL = DIR_GEOG_INT / "ciam_segment_pts_manual_adds.parquet"

PATH_SEG_CENTROIDS = DIR_GEOG_INT / "gtsm_stations_thinned_ciam.parquet"

PATH_GEOG_COASTLINES = DIR_GEOG_INT / f"ne_coastline_lines_CIAM_wexp_or_gtsm_10m_{COASTLINES_VERS}.parquet"

DIR_NATEARTH_RAW = DIR_GEOG_RAW / "natural_earth" / NE_DOWNLOAD_VERS
DIR_NATEARTH_INT = DIR_GEOG_INT / "natural_earth" / NE_DOWNLOAD_VERS
PATH_NATEARTH_LANDPOLYS = DIR_NATEARTH_RAW / "ne_10m_land.shp"
PATH_NATEARTH_COASTLINES_INT = DIR_NATEARTH_INT / "coastlines.parquet"
PATH_NATEARTH_OCEAN = DIR_NATEARTH_RAW / "ne_10m_ocean.shp"
PATH_NATEARTH_OCEAN_NOCASPIAN = DIR_NATEARTH_INT / "ne_10m_nocaspian.parquet"
PATH_NATEARTH_LAKES_INT = DIR_NATEARTH_INT / "inland_water.parquet"

PATH_GEOG_GTSM_STATIONS_TOTHIN = (
    DIR_GEOG_RAW / "gtsm_stations_eur_tothin" / "gtsm_stations_eur_tothin.parquet"
)

DIR_CIAM_VORONOI = DIR_GEOG_INT / "seg_region_intersections" / EXPOSURE_BINNED_VERS
PATH_SEG_REGION_VORONOI_INTERSECTIONS = (
    DIR_CIAM_VORONOI / "seg_region_intersections.parquet"
)
PATH_SEG_VORONOI = DIR_CIAM_VORONOI / "segment_voronoi.parquet"

PATH_SEG_REGION_VORONOI_INTERSECTIONS_SHP = (
    DIR_CIAM_VORONOI / "seg_region_intersections.shp"
)
PATH_SEGS = DIR_CIAM_VORONOI / "segment_linestrings.parquet"

DIR_GADM = DIR_DATA_RAW / "gadm" / GADM_VERS_STR

PATH_GADM = DIR_GADM / f"{GADM_VERS_STR}_levels" / f"{GADM_VERS_STR}-levels.gpkg"
PATH_GADM_ADM0_INT = DIR_GADM / "adm0.parquet"
PATH_GADM_ADM1_INT = DIR_GADM / "adm1.parquet"
PATH_GADM_ADM0_VORONOI = DIR_GADM / "adm0_voronoi.parquet"

PATH_EXPOSURE_ASSET_VALUE_BLENDED = (
    DIR_EXPOSURE_INT
    / "asset_value"
    / "litpop"
    / EXPOSURE_BLENDED_VERS
    / "LitPop_pc_30arcsec.parquet"
)

PATH_NLDB = DIR_EXPOSURE_RAW / "protected_areas" / "usa" / f"nldb-levee-areas_{NLDB_DATESTAMP}.parquet"
PATH_COMBINED_PROTECTED_AREAS = (
    DIR_EXPOSURE_INT
    / "protected_areas"
    / f"global-combined-protected-areas-{COMBINED_PROTECTED_AREAS_VERS}.parquet"
)

DIR_WETLANDS_RAW = DIR_DATA_RAW / "wetlands_mangroves"
DIR_WETLANDS_INT = DIR_DATA_INT / "wetlands_mangroves"
PATH_GLOBCOVER_2009 = (
    DIR_WETLANDS_RAW
    / "Globcover2009_V2.3_Global"
    / "GLOBCOVER_L4_200901_200912_V2.3.tif"
)

PATH_GLOBAL_MANGROVES = DIR_WETLANDS_RAW / f"GMW_{GMW_YEAR}" / f"gmw_v3_{GMW_YEAR}_vec.shp"

PATH_WETLANDS_INT = DIR_WETLANDS_INT / "wetlands.shp"

DIR_ELEVATION_RAW = DIR_DATA_RAW / "raw"
DIR_ELEVATION_INT = DIR_DATA_INT / "int"

PATH_GEBCO_RAW = DIR_ELEVATION_RAW / "gebco_2023" / "gebco_2023.zip"
DIR_ELEV_MSS = DIR_ELEVATION_INT / "coastalDEM_mss_corrected"
PATH_ELEV_MSS = DIR_ELEV_MSS / f"coastalDEM_mss_corrected_{MSS_DEM_VERS}.zarr"
PATH_HYDCON_TILE_CONNECTIONS = (
    DIR_ELEV_MSS / f"hydraulic_connectivity_tile_connections_{MSS_DEM_VERS}.parquet"
)

DIR_COASTALDEM = (
    DIR_ELEVATION_RAW
    / "climate_central"
    / "coastal_dem_30as"
    / f"coastaldem{COASTALDEM_VERS}_30m_egm96"
)

DIR_LANDSCAN_RAW = Path(
    str(DIR_EXPOSURE_RAW / "landscan" / LANDSCAN_YEAR).replace("gs:/", "/gcs")
)
PATH_LANDSCAN_RAW = DIR_LANDSCAN_RAW / f"landscan-global-{LANDSCAN_YEAR}.tif"
DIR_LANDSCAN_INT = DIR_EXPOSURE_INT / "landscan" / f"ls{LANDSCAN_YEAR}"
PATH_EXPOSURE_POP_INT = DIR_LANDSCAN_INT / "population.parquet"

DIR_EXPOSURE_BINNED = (
    DIR_EXPOSURE_INT / "asset_value" / "binned" / "global" / "historical"
)
DIR_EXPOSURE_BINNED_TMP = DIR_SCRATCH / "tmp_exposure"
DIR_EXPOSURE_BINNED_TMP_TILES = DIR_EXPOSURE_BINNED_TMP / "tiles"
DIR_EXPOSURE_BINNED_TMP_TILES_NOLAND = DIR_EXPOSURE_BINNED_TMP / "tiles_noland"
DIR_EXPOSURE_BINNED_TMP_TILES_SEGMENT_AREA = (
    DIR_EXPOSURE_BINNED_TMP / "tiles_segment_area"
)

PATH_EXPOSURE_TILE_LIST = (
    DIR_EXPOSURE_BINNED / "meta" / f"tile_list_{ELEV_TILE_LIST_VERS}.parquet"
)

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

PATH_GEOG_DATUMS_GRID = (
    DIR_GEOG_DATUMS_INT / f"datum_conversions_gridded_{DATUM_CONVERSION_VERS}.zarr"
)

PATH_GEOG_GTSM_SURGE = (
    DIR_DATA_RAW / "esl" / "CODEC_amax_ERA5_1979_2017_coor_mask_GUM_RPS.nc"
)

DIR_CCI_RAW = DIR_DATA_RAW / "cci"
PATH_EXPOSURE_WB_ICP = DIR_CCI_RAW / "world_bank_ICP_2017.csv"
PATH_EXPOSURE_LINCKE = DIR_CCI_RAW / "lincke_2021_country_input.csv"

# Country-code paths
PATH_ALL_VALID_HIST_CCODES = DIR_EXPOSURE_INT / "all_valid_hist_ccodes.parquet"
PATH_HIST_CCODE_MAPPING = DIR_EXPOSURE_INT / "hist_ccode_mapping.parquet"

# Various directories and paths for the country-level ("YPK") workflow
DIR_YPK_INT = DIR_EXPOSURE_INT / "ypk"
DIR_YPK_FINAL = DIR_YPK_INT / "finalized"
DIR_YPK_RAW = DIR_EXPOSURE_RAW / "ypk"

DIR_CIA_RAW = DIR_YPK_RAW / "cia_wfb"
PATH_CIA_INT = (
    DIR_YPK_INT / "cia_wfb" / CIA_WFB_VERS / "cia_wfb_constant_2017_ppp_usd.parquet"
)
DIR_UN_AMA_RAW = DIR_YPK_RAW / "un_ama" / UN_AMA_DATESTAMP
PATH_UN_REGION_DATA_RAW = DIR_UN_AMA_RAW / "UNSD — Methodology.csv"
DIR_UN_AMA_INT = DIR_YPK_INT / "un_ama" / UN_AMA_DATESTAMP
PATH_UN_AMA_INT = DIR_UN_AMA_INT / "un_ama_nominal_gdp_gdppc.parquet"
PATH_UN_REGION_DATA_INT = DIR_UN_AMA_INT / "un_region_mapping.parquet"

DIR_UN_WPP_RAW = DIR_YPK_RAW / "un_wpp" / UN_WPP_VERS
PATH_UN_WPP_INT = DIR_YPK_INT / "un_wpp" / f"un_wpp_{UN_WPP_VERS}.parquet"
DIR_WB_WDI_RAW = DIR_YPK_RAW / "wb_wdi" / WB_WDI_DATESTAMP
DIR_OECD_REGIONS_RAW = DIR_YPK_RAW / "oecd_regions" / OECD_DATESTAMP
PATH_IIASA_PROJECTIONS_RAW = (
    DIR_YPK_RAW / "iiasa_projections" / "SspDb_country_data_2013-06-12.csv"
)
DIR_ALAND_STATISTICS_RAW = DIR_YPK_RAW / "asub" / ALAND_STATISTICS_DATESTAMP

DIR_GLOBAL_WEALTH_INT = DIR_YPK_INT / "global_wealth_databook"
PATH_GWDB_RAW = (
    DIR_YPK_RAW / "gwdb" / GWDB_DATESTAMP / "global-wealth-databook-2023.pdf"
)
PATH_GWDB_INT = DIR_GLOBAL_WEALTH_INT / f"gwdb_{GWDB_YEAR}.parquet"

PATH_PWT_RAW = DIR_YPK_RAW / "pwt" / PWT_DATESTAMP / "pwt.xlsx"
PATH_IMF_WEO_RAW = DIR_YPK_RAW / "imf_weo" / IMF_WEO_VERS / "WEO_iy_ratio_pop_gdp.xls"
PATH_MPD_RAW = DIR_YPK_RAW / "mpd" / MPD_DATESTAMP / "maddison_project.xlsx"

PATH_ADB_RAW = (
    DIR_YPK_RAW / "adb" / ADB_KI_VERS / f"ki-{ADB_KI_VERS}-economy-tables_0.xlsx"
)

PATH_GW_TABLE = DIR_YPK_RAW / "gwstates.rda"
PATH_FARISS = DIR_YPK_RAW / "Fariss_JCR_2022.zip"
DIR_FARISS_INT = DIR_YPK_INT / "Fariss_JCR_2022"

PATH_INC_POP_AUX = DIR_YPK_INT / f"various_auxiliary_sources_yp_{INC_POP_AUX_DATESTAMP}.parquet"
PATH_INC_POP_AGG = DIR_YPK_INT / f"aggregated_sources_yp_{INC_POP_AUX_DATESTAMP}.parquet"
PATH_INC_POP_CLEANED = (
    DIR_YPK_INT
    / f"yp_{HISTORICAL_YEARS[0]}_{HISTORICAL_YEARS[-1]}_cleaned_{INC_POP_CLEANED_VERS}"
    ".parquet"
)
PATH_EXPOSURE_YPK_COUNTRY_HIST_INT = (
    DIR_YPK_FINAL / f"ypk_{HISTORICAL_YEARS[0]}_{HISTORICAL_YEARS[-1]}_"
    f"{YPK_FINALIZED_VERS}.parquet"
)
PATH_EXPOSURE_YPK_COUNTRY_PROJ_INT = (
    DIR_YPK_FINAL / f"ypk_{PROJ_YEARS[0]}_{PROJ_YEARS[-1]}_"
    f"{YPK_FINALIZED_VERS}.zarr"
)

##########################
# Country Data/Definitions
##########################

# Aland Islands, Western Sahara, Libya, Palestine, South Sudan, Syria, Kosovo
ISOS_IN_GEG_NOT_LITPOP = ["ALA", "ESH", "LBY", "PSE", "SSD", "SYR", "XKO"]

# uninhabited ISO codes to include in analysis (for land/wetland value)
UNINHABITED_ISOS = ["ATF", "BVT", "XCL", "HMD", "SGS", "UMI", "XSP"]

# country ISO code groupings to exclude entirely (antarctica and caspian sea)
EXCLUDED_ISOS = ["ATA", "XCA"]

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

FRA_OVERSEAS_DEPT = "FRA+GLP+GUF+MTQ+MYT+REU"
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
    FRA_OVERSEAS_DEPT,
]
PPP_CCODE_IF_MSNG = {
    "AUS": ["CCK", "CXR", "HMD", "NFK"],
    "DNK": ["GRL", "FRO"],
    "FRA": FRA_MSNG,
    "FIN": ["ALA"],
    "ITA": ["VAT", "SMR"],
    "USA": USA_MSNG,
    "MAR": ["ESH"],
    "CUW": ["BES", "BES+CUW+SXM"],
    "NZL": ["NIU", "COK", "TKL"],
    "NOR": ["BVT", "SJM"],
    "GBR": ["IMN", "FLK", "GGY+JEY", "GIB", "PCN", "SGS", "SHN", "GGY", "JEY"],
    "CYP": ["ZNC", "CYP+ZNC"],
    "MNE": ["SRB+MNE"],
}

# Extra country to ccode mappings
CCODE_MANUAL = (
    pd.DataFrame(
        [
            ["Akrotiri", "XAD"],
            ["Bahamas, The", "BHS"],
            ["Burma", "MMR"],
            ["Cape Verde", "CPV"],
            ["Channel Islands", "GGY+JEY"],
            ["Christmas Island", "CXR"],
            ["Cocos (Keeling) Islands", "CCK"],
            ["Congo, Democratic Republic of the", "COD"],
            ["Congo, Dem. Rep.", "COD"],
            ["Congo, Republic of the", "COG"],
            ["Congo, Rep.", "COG"],
            ["Coral Sea Islands", "AUS"],
            ["Cyprus/Northern Cyprus", "CYP+ZNC"],
            ["Czech Republic", "CZE"],
            ["Dhekelia", "XAD"],
            ["East Timor", "TLS"],
            ["Eswatini", "SWZ"],
            ["Falkland Islands (Islas Malvinas)", "FLK"],
            ["Former Sudan", "SDN+SSD"],
            ["Eritrea", "ERI"],
            ["Faroe Islands", "FRO"],
            ["French Southern and Antarctic Lands", "ATF"],
            ["Gambia, The", "GMB"],
            # Gaza Strip and West Bank will together constitute PSE
            ["Gaza Strip", "PSE"],
            ["West Bank", "PSE"],
            ["Heard Island and McDonald Islands", "HMD"],
            ["Holy See (Vatican City)", "VAT"],
            ["Hong Kong", "HKG"],
            ["Jan Mayen", "SJM"],  # Svalbard and Jan Mayen are grouped together
            ["Svalbard", "SJM"],
            ["Korea, North", "PRK"],
            ["Korea", "KOR"],
            ["Korea, South", "KOR"],
            ["Macao", "MAC"],
            ["Macau", "MAC"],
            ["Macedonia", "MKD"],
            ["Man, Isle of", "IMN"],
            ["Macedonia, The Former Yugoslav Republic of", "MKD"],
            ["Micronesia, Federated States of", "FSM"],
            ["Saint Barthelemy", "BLM"],
            ["Saint Helena", "SHN"],  # will use the below version whenever possible
            ["Saint Helena, Ascension, and Tristan da Cunha", "SHN"],
            ["St. Lucia", "LCA"],
            ["Saint Martin", "MAF"],
            ["Serbia and Montenegro", "SRB+MNE"],
            ["Yugoslavia", "SRB+MNE"],
            ["South Georgia and South Sandwich Islands", "SGS"],
            ["St. Vincent and the Grenadines", "VCT"],
            ["Virgin Islands", "VIR"],
            ["Navassa Island", "UMI"],
            ["Jarvis Island", "UMI"],
            ["Howland Island", "UMI"],
            ["Johnston Atoll", "UMI"],
            ["Kingman Reef", "UMI"],
            ["Palmyra Atoll", "UMI"],
            ["Midway Islands", "UMI"],
            ["United States Pacific Island Wildlife Refuges", "UMI"],
            ["Wake Island", "UMI"],
            ["Turkey (Turkiye)", "TUR"],
        ],
        columns=["name", "ccode"],
    )
    .set_index("name")
    .ccode
)


# not all countries/territories represented in GADM (which is our baseline definition)
# are present in the SSPs. Some that are not represented are independent territories, so
# we assign global average GDPpc and population growth rates. For others, they are
# territories, so we match their growth to that of the territory and/or claimant.
GADM_TO_SSP_ISO_MAPPING = pd.DataFrame(
    {
        "AIA": ("GBR", True),
        "ALA": ("FIN", True),
        "ASM": ("USA", True),
        "ATF": ("FRA", True),
        "BES": ("NLD", True),
        "BLM": ("FRA", True),
        "BMU": ("GBR", True),
        "BVT": ("NOR", True),
        "CCK": ("AUS", True),
        "CUW": ("NLD", True),
        "CXR": ("AUS", True),
        "CYM": ("GBR", True),
        "ESH": ("MAR", False),
        "FLK": ("GBR", True),
        "FRO": ("DNK", True),
        "GGY": ("GBR", True),
        "GIB": ("GBR", True),
        "GRL": ("DNK", True),
        "HMD": ("AUS", True),
        "IMN": ("GBR", True),
        "IOT": ("GBR", True),
        "JEY": ("GBR", True),
        "MAF": ("FRA", True),
        "MNP": ("USA", True),
        "MSR": ("GBR", True),
        "NFK": ("AUS", True),
        "PCN": ("GBR", True),
        "SGS": ("GBR", True),
        "SHN": ("GBR", True),
        "SJM": ("NOR", True),
        "SPM": ("FRA", True),
        "SSD": ("SDN", False),
        "SXM": ("NLD", True),
        "TCA": ("GBR", True),
        "TKL": ("NZL", True),
        "UMI": ("USA", True),
        "VGB": ("GBR", True),
        "WLF": ("FRA", True),
        "XAD": ("GBR", True),
        "XCL": ("FRA", True),
        "XKO": ("SRB", True),
        "XPI": ("CHN", False),
        "XSP": ("CHN", False),
        "ZNC": ("CYP", True),
    },
    index=["parent", "included_in_parent"],
).T.rename_axis(index="ccode")
