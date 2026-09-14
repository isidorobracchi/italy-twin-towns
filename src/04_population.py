from pathlib import Path
import zipfile

import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw" / "population"
EXTRACT_DIR = RAW_DIR / "extracted"

MASTER_FILE = (
    ROOT
    / "data"
    / "processed"
    / "municipalities_master.csv"
)

OUTPUT_FILE = MASTER_FILE

RAW_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# FIND ZIP
# --------------------------------------------------

def find_zip():

    zip_files = list(RAW_DIR.glob("*.zip"))

    if len(zip_files) == 0:
        raise FileNotFoundError(
            "No population ZIP found in data/raw/population/"
        )

    if len(zip_files) > 1:
        print("Multiple ZIP files found. Using:")
        for file in zip_files:
            print("-", file.name)

    zip_file = zip_files[0]

    print(f"Population ZIP: {zip_file.name}")

    return zip_file


# --------------------------------------------------
# EXTRACT
# --------------------------------------------------

def extract_zip(zip_file):

    csv_files = list(EXTRACT_DIR.rglob("*.csv"))

    if csv_files:
        print("Population files already extracted.")
        return

    print("Extracting population files...")

    with zipfile.ZipFile(zip_file, "r") as archive:
        archive.extractall(EXTRACT_DIR)

    print(f"Extracted to: {EXTRACT_DIR}")


# --------------------------------------------------
# FIND CSV
# --------------------------------------------------

def find_csv():

    csv_files = list(EXTRACT_DIR.rglob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(
            "No CSV files found in extracted population folder."
        )

    if len(csv_files) > 1:
        print("Multiple CSV files found. Using the first one.")

    csv_file = csv_files[0]

    print(f"Population CSV: {csv_file.name}")

    return csv_file


# --------------------------------------------------
# READ POPULATION
# --------------------------------------------------

def read_population(csv_file):

    print("\nReading population file...")

    df = pd.read_csv(
        csv_file,
        sep=";",
        skiprows=1,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
        low_memory=False,
    )

    print(f"Raw rows: {len(df):,}")

    print("\nColumns:")
    for col in df.columns:
        print("-", col)

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    return df


# --------------------------------------------------
# PREPARE POPULATION
# --------------------------------------------------

def prepare_population(df):

    print("\nPreparing population totals...")

    required_columns = [
        "Codice comune",
        "Comune",
        "Età",
        "Totale",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing expected columns: {missing}"
        )

    # --------------------------------------------------
    # ISTAT CODE
    # --------------------------------------------------

    df["istat_code"] = (
        df["Codice comune"]
        .astype(str)
        .str.strip()
        .str.zfill(6)
    )

    # --------------------------------------------------
    # KEEP ONLY VALID MUNICIPALITY ROWS
    # --------------------------------------------------

    # Rimuove la nota finale presente nel CSV ISTAT.
    valid_code_mask = (
        df["istat_code"]
        .str.fullmatch(r"\d{6}")
    )

    invalid_code_rows = (~valid_code_mask).sum()

    print(
        "Rows without valid municipality code:",
        invalid_code_rows
    )

    df = df[valid_code_mask].copy()

    # --------------------------------------------------
    # CONVERT POPULATION TO NUMERIC
    # --------------------------------------------------

    df["Totale"] = pd.to_numeric(
        df["Totale"],
        errors="coerce"
    )

    invalid_population = (
        df["Totale"]
        .isna()
        .sum()
    )

    print(
        "Invalid population values after filtering:",
        invalid_population
    )

    if invalid_population > 0:

        print(
            df[df["Totale"].isna()]
            .head(20)
            .to_string(index=False)
        )

        raise ValueError(
            "Valid municipality rows contain "
            "non-numeric population values."
        )

    # --------------------------------------------------
    # USE ISTAT MUNICIPAL TOTAL ROW
    # --------------------------------------------------

    # Nel file ISTAT:
    # Età = 999 identifica il totale della popolazione
    # residente del comune.
    total_rows = df[
        df["Età"]
        .astype(str)
        .str.strip()
        .eq("999")
    ].copy()

    print(
        "Municipality total rows found:",
        len(total_rows)
    )

    if len(total_rows) == 0:
        raise ValueError(
            "No municipal total rows with Età = 999 were found."
        )

    population = (
        total_rows[
            [
                "istat_code",
                "Comune",
                "Totale",
            ]
        ]
        .rename(
            columns={
                "Comune": "population_name",
                "Totale": "population_2026",
            }
        )
        .copy()
    )

    population["population_2026"] = pd.to_numeric(
        population["population_2026"],
        errors="raise"
    )

    # --------------------------------------------------
    # VALIDATE AGGREGATED POPULATION
    # --------------------------------------------------

    print(
        f"\nMunicipalities in population dataset: "
        f"{len(population):,}"
    )

    duplicate_codes = (
        population["istat_code"]
        .duplicated(keep=False)
    )

    if duplicate_codes.any():

        print(
            "\nDuplicate municipality codes "
            "in population totals:"
        )

        print(
            population[
                duplicate_codes
            ].to_string(index=False)
        )

        raise ValueError(
            "Duplicate municipality totals found."
        )

    print("\nPopulation preview:")

    print(
        population
        .head()
        .to_string(index=False)
    )

    return population


# --------------------------------------------------
# MERGE WITH MASTER
# --------------------------------------------------

def merge_with_master(population):

    print("\nLoading municipality master...")

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

    if "istat_code_old" in master.columns:

        master["istat_code_old"] = (
            master["istat_code_old"]
            .astype(str)
            .str.zfill(6)
        )

    # --------------------------------------------------
    # CHECK DIRECT CODE MATCH
    # --------------------------------------------------

    master_codes = set(
        master["istat_code"]
    )

    population_codes = set(
        population["istat_code"]
    )

    direct_matches = len(
        master_codes.intersection(
            population_codes
        )
    )

    print(
        "\nDirect code matches:",
        direct_matches
    )

    print(
        "Master municipalities:",
        len(master_codes)
    )

    print(
        "Population municipalities:",
        len(population_codes)
    )

    # --------------------------------------------------
    # RECONCILE OLD -> 2026 CODES
    # --------------------------------------------------

    if "istat_code_old" in master.columns:

        old_to_new = dict(
            zip(
                master["istat_code_old"],
                master["istat_code"],
            )
        )

        population[
            "istat_code_original"
        ] = population["istat_code"]

        population["istat_code"] = (
            population["istat_code"]
            .replace(old_to_new)
        )

        changed_codes = (
            population["istat_code"]
            != population["istat_code_original"]
        ).sum()

        print(
            "Population codes reconciled "
            "via old→2026 crosswalk:",
            changed_codes
        )

    # --------------------------------------------------
    # VALIDATE POPULATION CODES
    # --------------------------------------------------

    duplicated_population_codes = (
        population["istat_code"]
        .duplicated(keep=False)
    )

    if duplicated_population_codes.any():

        print(
            "\nDuplicate population codes "
            "after reconciliation:"
        )

        print(
            population[
                duplicated_population_codes
            ].to_string(index=False)
        )

        raise ValueError(
            "Duplicate municipality codes "
            "in population dataset."
        )

    # --------------------------------------------------
    # REMOVE OLD FEATURES IF RERUN
    # --------------------------------------------------

    columns_to_replace = [
        "population_2026",
        "density_km2",
    ]

    master = master.drop(
        columns=[
            col
            for col in columns_to_replace
            if col in master.columns
        ]
    )

    # --------------------------------------------------
    # MERGE
    # --------------------------------------------------

    merged = master.merge(
        population[
            [
                "istat_code",
                "population_2026",
            ]
        ],
        on="istat_code",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------

    merged["area_km2"] = pd.to_numeric(
        merged["area_km2"],
        errors="coerce"
    )

    merged["population_2026"] = (
        pd.to_numeric(
            merged["population_2026"],
            errors="coerce"
        )
    )

    # --------------------------------------------------
    # DENSITY
    # --------------------------------------------------

    merged["density_km2"] = (
        merged["population_2026"]
        / merged["area_km2"]
    ).round(2)

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    print("\n--- POPULATION VALIDATION ---")

    print(
        "Merged municipalities:",
        len(merged)
    )

    print(
        "Missing population:",
        merged[
            "population_2026"
        ].isna().sum()
    )

    print(
        "Missing density:",
        merged[
            "density_km2"
        ].isna().sum()
    )

    # --------------------------------------------------
    # TEST MUNICIPALITIES
    # --------------------------------------------------

    for municipality in [
        "Giarre",
        "Roma",
        "Milano",
    ]:

        print(
            f"\n--- {municipality.upper()} ---"
        )

        print(
            merged[
                merged["name"]
                == municipality
            ][
                [
                    "istat_code",
                    "name",
                    "population_2026",
                    "area_km2",
                    "density_km2",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print(
        "\nUpdated master saved:"
    )

    print(
        OUTPUT_FILE
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    zip_file = find_zip()

    extract_zip(
        zip_file
    )

    csv_file = find_csv()

    population_raw = read_population(
        csv_file
    )

    population = prepare_population(
        population_raw
    )

    merge_with_master(
        population
    )