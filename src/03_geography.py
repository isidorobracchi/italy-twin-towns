from pathlib import Path
import zipfile
import requests
import unicodedata

import pandas as pd
import geopandas as gpd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw" / "boundaries"
PROCESSED_DIR = ROOT / "data" / "processed"

MASTER_FILE = PROCESSED_DIR / "municipalities_master.csv"
OUTPUT_FILE = PROCESSED_DIR / "municipalities_master.csv"

ZIP_FILE = RAW_DIR / "Limiti01012026_g.zip"
EXTRACT_DIR = RAW_DIR / "Limiti01012026_g"


# Confini amministrativi ISTAT 2026 - versione generalizzata
URL = (
    "https://www.istat.it/storage/cartografia/"
    "confini_amministrativi/generalizzati/2026/"
    "Limiti01012026_g.zip"
)


RAW_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# UTILITIES
# --------------------------------------------------

def normalize_name(value):
    """
    Normalizza i nomi dei comuni per effettuare join robusti.

    Gestisce:
    - spazi
    - maiuscole/minuscole
    - accenti e diacritici
    - apostrofi tipografici
    - mojibake tipo "NicolÃ²" -> "Nicolò"

    La funzione viene usata solo per il matching.
    I nomi originali non vengono modificati nel dataset finale.
    """

    if value is None:
        return ""

    value = str(value).strip()

    # Prova a correggere casi comuni di mojibake:
    # es. "NicolÃ²" -> "Nicolò"
    try:
        value = value.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # Uniforma gli apostrofi
    value = (
        value
        .replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
    )

    # Rimuove accenti / diacritici
    value = unicodedata.normalize("NFKD", value)

    value = "".join(
        char
        for char in value
        if not unicodedata.combining(char)
    )

    # Normalizza spazi e case
    value = " ".join(value.lower().split())

    return value


# --------------------------------------------------
# DOWNLOAD
# --------------------------------------------------

def download_boundaries():

    if ZIP_FILE.exists():
        print("Boundary ZIP already downloaded.")
        return

    print("Downloading ISTAT 2026 boundaries...")

    response = requests.get(URL, timeout=120)
    response.raise_for_status()

    ZIP_FILE.write_bytes(response.content)

    print(f"Saved: {ZIP_FILE}")


# --------------------------------------------------
# EXTRACT
# --------------------------------------------------

def extract_boundaries():

    shapefiles = list(EXTRACT_DIR.rglob("*.shp"))

    if shapefiles:
        print("Boundaries already extracted.")
        return

    print("Extracting boundaries...")

    with zipfile.ZipFile(ZIP_FILE, "r") as archive:
        archive.extractall(EXTRACT_DIR)

    print(f"Extracted to: {EXTRACT_DIR}")


# --------------------------------------------------
# FIND MUNICIPAL SHAPEFILE
# --------------------------------------------------

def find_municipality_shapefile():

    shapefiles = list(EXTRACT_DIR.rglob("*.shp"))

    print("\nShapefiles found:")

    for shp in shapefiles:
        print("-", shp.name)

    municipality_candidates = [
        shp
        for shp in shapefiles
        if shp.name.lower().startswith("com")
    ]

    if len(municipality_candidates) == 0:
        raise FileNotFoundError(
            "Could not automatically find municipality shapefile."
        )

    municipality_shp = municipality_candidates[0]

    print(
        f"\nMunicipality shapefile selected: "
        f"{municipality_shp.name}"
    )

    return municipality_shp


# --------------------------------------------------
# PREPARE GEOGRAPHY
# --------------------------------------------------

def prepare_geography(shapefile):

    print("\nReading municipality boundaries...")

    gdf = gpd.read_file(shapefile)

    print(f"Rows: {len(gdf):,}")
    print(f"CRS: {gdf.crs}")

    print("\nColumns:")
    for col in gdf.columns:
        print("-", col)

    # --------------------------------------------------
    # FIND ISTAT CODE
    # --------------------------------------------------

    possible_code_columns = [
        "PRO_COM_T",
        "PRO_COM",
        "COD_COM",
        "COM_CODE",
    ]

    code_column = None

    for candidate in possible_code_columns:
        if candidate in gdf.columns:
            code_column = candidate
            break

    if code_column is None:
        raise KeyError(
            "Could not identify municipality ISTAT code column."
        )

    print(f"\nISTAT code column: {code_column}")

    gdf["istat_code"] = (
        gdf[code_column]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.zfill(6)
    )

    # --------------------------------------------------
    # AREA
    # --------------------------------------------------

    # EPSG:3035 = ETRS89 / LAEA Europe
    # Proiezione equal-area adatta a confrontare superfici
    # tra comuni distribuiti su tutto il territorio italiano.
    area_gdf = gdf.to_crs(epsg=3035)

    gdf["area_km2"] = (
        area_gdf.geometry.area / 1_000_000
    )

    # --------------------------------------------------
    # REPRESENTATIVE POINT
    # --------------------------------------------------

    # Usiamo representative_point() invece del centroid:
    # garantisce che il punto sia interno al comune.
    points = area_gdf.geometry.representative_point()

    points = gpd.GeoSeries(
        points,
        crs=area_gdf.crs
    ).to_crs(epsg=4326)

    gdf["longitude"] = points.x
    gdf["latitude"] = points.y

    geography = gdf[
        [
            "istat_code",
            "area_km2",
            "latitude",
            "longitude",
        ]
    ].copy()

    geography["area_km2"] = geography["area_km2"].round(3)
    geography["latitude"] = geography["latitude"].round(6)
    geography["longitude"] = geography["longitude"].round(6)

    print("\nGeography preview:")
    print(
        geography
        .head()
        .to_string(index=False)
    )

    return geography, gdf


# --------------------------------------------------
# MERGE WITH MASTER
# --------------------------------------------------

def merge_with_master(geography, gdf):

    print("\nLoading master municipalities...")

    master = pd.read_csv(
        MASTER_FILE,
        dtype=str,
        keep_default_na=False,
    )

    master["istat_code"] = (
        master["istat_code"]
        .astype(str)
        .str.zfill(6)
    )

    master["region_code"] = (
        master["region_code"]
        .astype(str)
        .str.zfill(2)
    )

    # --------------------------------------------------
    # BUILD 2026 CODE CROSSWALK FROM SHAPEFILE
    # --------------------------------------------------

    geo_codes = gdf[
        [
            "PRO_COM_T",
            "COMUNE",
            "COD_REG",
            "COD_UTS",
        ]
    ].copy()

    geo_codes["istat_code_2026"] = (
        geo_codes["PRO_COM_T"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.zfill(6)
    )

    geo_codes["geo_name"] = (
        geo_codes["COMUNE"]
        .astype(str)
        .str.strip()
    )

    geo_codes["region_code"] = (
        geo_codes["COD_REG"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.zfill(2)
    )

    geo_codes["match_name"] = (
        geo_codes["geo_name"]
        .apply(normalize_name)
    )

    # --------------------------------------------------
    # FIRST TRY: EXISTING CODE
    # --------------------------------------------------

    current_2026_codes = set(
        geo_codes["istat_code_2026"]
    )

    # Conserviamo il codice precedente:
    # sarà utile per dataset storici.
    master["istat_code_old"] = master["istat_code"]

    master["code_matches_2026"] = (
        master["istat_code"]
        .isin(current_2026_codes)
    )

    print(
        "\nCodes already matching 2026 geography:",
        master["code_matches_2026"].sum()
    )

    print(
        "Codes requiring reconciliation:",
        (~master["code_matches_2026"]).sum()
    )

    # --------------------------------------------------
    # FALLBACK: NORMALIZED NAME + REGION
    # --------------------------------------------------

    unmatched = master[
        ~master["code_matches_2026"]
    ].copy()

    unmatched["match_name"] = (
        unmatched["name"]
        .apply(normalize_name)
    )

    crosswalk = unmatched.merge(
        geo_codes,
        on=[
            "match_name",
            "region_code",
        ],
        how="left",
        validate="one_to_one",
    )

    unresolved = (
        crosswalk["istat_code_2026"]
        .isna()
        .sum()
    )

    print(
        "Unresolved after normalized "
        "name + region reconciliation:",
        unresolved
    )

    if unresolved > 0:

        print("\nUnresolved municipalities:")

        print(
            crosswalk[
                crosswalk["istat_code_2026"].isna()
            ][
                [
                    "istat_code",
                    "name",
                    "region",
                ]
            ].to_string(index=False)
        )

        raise ValueError(
            "Some municipalities could not be reconciled."
        )

    # --------------------------------------------------
    # CREATE OLD -> 2026 CODE MAP
    # --------------------------------------------------

    code_map = dict(
        zip(
            crosswalk["istat_code"],
            crosswalk["istat_code_2026"],
        )
    )

    master["istat_code"] = (
        master["istat_code"]
        .replace(code_map)
    )

    # --------------------------------------------------
    # VALIDATE RECONCILIATION
    # --------------------------------------------------

    print("\n--- CODE RECONCILIATION ---")

    print(
        "Updated municipality codes:",
        len(code_map)
    )

    print(
        "Unique codes after reconciliation:",
        master["istat_code"].nunique()
    )

    duplicate_codes = master[
        master["istat_code"]
        .duplicated(keep=False)
    ]

    if len(duplicate_codes) > 0:

        print(
            "\nDuplicate codes after reconciliation:"
        )

        print(
            duplicate_codes[
                [
                    "istat_code",
                    "name",
                    "region",
                ]
            ].to_string(index=False)
        )

        raise ValueError(
            "Duplicate ISTAT codes after reconciliation."
        )

    # --------------------------------------------------
    # REMOVE PREVIOUS GEOGRAPHIC COLUMNS
    # --------------------------------------------------

    columns_to_replace = [
        "area_km2",
        "latitude",
        "longitude",
    ]

    master = master.drop(
        columns=[
            col
            for col in columns_to_replace
            if col in master.columns
        ]
    )

    # --------------------------------------------------
    # MERGE GEOGRAPHY
    # --------------------------------------------------

    merged = master.merge(
        geography,
        on="istat_code",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    print("\n--- GEOGRAPHY VALIDATION ---")

    print(
        f"Master municipalities: "
        f"{len(master):,}"
    )

    print(
        f"Geographic municipalities: "
        f"{len(geography):,}"
    )

    print(
        f"Merged rows: "
        f"{len(merged):,}"
    )

    print(
        "Missing areas:",
        merged["area_km2"].isna().sum()
    )

    print(
        "Missing coordinates:",
        merged["latitude"].isna().sum()
    )

    # --------------------------------------------------
    # TEST GIARRE
    # --------------------------------------------------

    print("\n--- GIARRE ---")

    print(
        merged[
            merged["name"] == "Giarre"
        ][
            [
                "istat_code",
                "name",
                "area_km2",
                "latitude",
                "longitude",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------
    # TEST SARDINIA
    # --------------------------------------------------

    print("\n--- SARDINIA TEST: ALGHERO ---")

    print(
        merged[
            merged["name"] == "Alghero"
        ][
            [
                "istat_code_old",
                "istat_code",
                "name",
                "province",
                "region",
                "area_km2",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------
    # CLEAN TEMPORARY COLUMNS
    # --------------------------------------------------

    merged = merged.drop(
        columns=[
            "code_matches_2026",
        ]
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print("\nUpdated master saved:")
    print(OUTPUT_FILE)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    download_boundaries()

    extract_boundaries()

    municipality_shapefile = (
        find_municipality_shapefile()
    )

    geography, gdf = prepare_geography(
        municipality_shapefile
    )

    merge_with_master(
        geography,
        gdf,
    )