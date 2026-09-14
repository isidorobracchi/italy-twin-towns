from pathlib import Path

import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "territory"
)

MASTER_FILE = (
    ROOT
    / "data"
    / "processed"
    / "municipalities_master.csv"
)


# --------------------------------------------------
# FIND FILE
# --------------------------------------------------

def find_territory_file():

    files = [
        file
        for file in RAW_DIR.glob("*.csv")
        if file.is_file()
    ]

    if len(files) == 0:
        raise FileNotFoundError(
            "No territory CSV found in data/raw/territory/"
        )

    if len(files) > 1:
        print("Multiple territory files found:")
        for file in files:
            print("-", file.name)

    territory_file = files[0]

    print(
        f"Territory file: {territory_file.name}"
    )

    return territory_file


# --------------------------------------------------
# READ TERRITORY DATA
# --------------------------------------------------

def read_territory(file):

    print("\nReading ISTAT territory data...")

    df = pd.read_csv(
        file,
        sep=";",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns loaded: {len(df.columns)}"
    )

    return df


# --------------------------------------------------
# PREPARE TERRITORY FEATURES
# --------------------------------------------------

def prepare_territory(df):

    print("\nPreparing TERRITORY features...")

    required_columns = [
        "Codice Comune (alfanumerico)",
        "Comune",
        "Comune litoraneo",
        "Zona altimetrica",
        "Altitudine (Municipio)",
        "Zone costiere 2021",
        "Degurba 2021",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing expected territory columns: {missing}"
        )

    # --------------------------------------------------
    # ISTAT CODE
    # --------------------------------------------------

    df["istat_code"] = (
        df["Codice Comune (alfanumerico)"]
        .astype(str)
        .str.strip()
        .str.zfill(6)
    )

    valid_codes = (
        df["istat_code"]
        .str.fullmatch(r"\d{6}")
    )

    print(
        "Rows without valid ISTAT code:",
        (~valid_codes).sum()
    )

    df = df[
        valid_codes
    ].copy()

    # --------------------------------------------------
    # RENAME FIELDS
    # --------------------------------------------------

    territory = df[
        [
            "istat_code",
            "Comune",
            "Comune litoraneo",
            "Zona altimetrica",
            "Altitudine (Municipio)",
            "Zone costiere 2021",
            "Degurba 2021",
        ]
    ].copy()

    territory = territory.rename(
        columns={
            "Comune": "territory_name",
            "Comune litoraneo": "coastal",
            "Zona altimetrica": "altimetric_zone",
            "Altitudine (Municipio)": "altitude_m",
            "Zone costiere 2021": "coastal_zone",
            "Degurba 2021": "degurba",
        }
    )

    # --------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------

    numeric_columns = [
        "coastal",
        "altimetric_zone",
        "altitude_m",
        "coastal_zone",
        "degurba",
    ]

    for column in numeric_columns:

        territory[column] = pd.to_numeric(
            territory[column],
            errors="coerce"
        )

    print("\nInvalid numeric values:")

    print(
        territory[
            numeric_columns
        ].isna().sum()
    )

    # --------------------------------------------------
    # DUPLICATE CHECK
    # --------------------------------------------------

    duplicates = (
        territory["istat_code"]
        .duplicated(keep=False)
    )

    if duplicates.any():

        print(
            "\nDuplicate territory codes:"
        )

        print(
            territory[
                duplicates
            ].to_string(index=False)
        )

        raise ValueError(
            "Duplicate municipality codes "
            "in territory dataset."
        )

    print(
        f"\nMunicipalities in territory dataset: "
        f"{len(territory):,}"
    )

    print("\nTerritory preview:")

    print(
        territory
        .head()
        .to_string(index=False)
    )

    return territory


# --------------------------------------------------
# MERGE WITH MASTER
# --------------------------------------------------

def merge_with_master(territory):

    print(
        "\nLoading municipality master..."
    )

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
    # DIRECT MATCH
    # --------------------------------------------------

    master_codes = set(
        master["istat_code"]
    )

    territory_codes = set(
        territory["istat_code"]
    )

    direct_matches = len(
        master_codes.intersection(
            territory_codes
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
        "Territory municipalities:",
        len(territory_codes)
    )

    # --------------------------------------------------
    # OLD -> 2026 CROSSWALK
    # --------------------------------------------------

    if "istat_code_old" in master.columns:

        old_to_new = dict(
            zip(
                master["istat_code_old"],
                master["istat_code"],
            )
        )

        territory[
            "istat_code_original"
        ] = territory["istat_code"]

        territory["istat_code"] = (
            territory["istat_code"]
            .replace(old_to_new)
        )

        changed_codes = (
            territory["istat_code"]
            != territory[
                "istat_code_original"
            ]
        ).sum()

        print(
            "Territory codes reconciled "
            "via old→2026 crosswalk:",
            changed_codes
        )

    # --------------------------------------------------
    # DUPLICATES AFTER RECONCILIATION
    # --------------------------------------------------

    duplicates = (
        territory["istat_code"]
        .duplicated(keep=False)
    )

    if duplicates.any():

        print(
            "\nDuplicate codes after reconciliation:"
        )

        print(
            territory[
                duplicates
            ].to_string(index=False)
        )

        raise ValueError(
            "Duplicate territory codes "
            "after reconciliation."
        )

    # --------------------------------------------------
    # REMOVE OLD COLUMNS IF RERUN
    # --------------------------------------------------

    territory_columns = [
        "coastal",
        "altimetric_zone",
        "altitude_m",
        "coastal_zone",
        "degurba",
    ]

    master = master.drop(
        columns=[
            col
            for col in territory_columns
            if col in master.columns
        ]
    )

    # --------------------------------------------------
    # MERGE
    # --------------------------------------------------

    merged = master.merge(
        territory[
            [
                "istat_code",
                "coastal",
                "altimetric_zone",
                "altitude_m",
                "coastal_zone",
                "degurba",
            ]
        ],
        on="istat_code",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------
    # CONVERT NUMERIC
    # --------------------------------------------------

    for column in territory_columns:

        merged[column] = pd.to_numeric(
            merged[column],
            errors="coerce"
        )

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    print(
        "\n--- TERRITORY VALIDATION ---"
    )

    print(
        "Merged municipalities:",
        len(merged)
    )

    print(
        "\nMissing TERRITORY values:"
    )

    print(
        merged[
            territory_columns
        ]
        .isna()
        .sum()
    )

    # --------------------------------------------------
    # FIND UNMATCHED MUNICIPALITIES
    # --------------------------------------------------

    unmatched = merged[
        merged["altitude_m"].isna()
    ][
        [
            "istat_code",
            "istat_code_old",
            "name",
            "province",
            "region",
        ]
    ]

    if len(unmatched) > 0:

        print(
            "\nMunicipalities without territory match:"
        )

        print(
            unmatched.to_string(index=False)
        )

    # --------------------------------------------------
    # TEST MUNICIPALITIES
    # --------------------------------------------------

    for municipality in [
        "Giarre",
        "Acireale",
        "Bacoli",
        "Roma",
        "Milano",
        "Niscemi",
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
                    "name",
                    "area_km2",
                    "altitude_m",
                    "coastal",
                    "altimetric_zone",
                    "coastal_zone",
                    "degurba",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    merged.to_csv(
        MASTER_FILE,
        index=False,
        encoding="utf-8",
    )

    print(
        "\nUpdated master saved:"
    )

    print(
        MASTER_FILE
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    territory_file = (
        find_territory_file()
    )

    territory_raw = read_territory(
        territory_file
    )

    territory = prepare_territory(
        territory_raw
    )

    merge_with_master(
        territory
    )