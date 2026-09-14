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
    / "income"
)

MASTER_FILE = (
    ROOT
    / "data"
    / "processed"
    / "municipalities_master.csv"
)


# --------------------------------------------------
# COLUMNS
# --------------------------------------------------

COL_ISTAT = "Codice Istat Comune"
COL_NAME = "Denominazione Comune"

COL_TAXPAYERS = "Numero contribuenti"

COL_TOTAL_INCOME_FREQ = (
    "Reddito complessivo - Frequenza"
)

COL_TOTAL_INCOME_AMOUNT = (
    "Reddito complessivo - Ammontare in euro"
)

COL_55_75 = (
    "Reddito complessivo da 55000 a 75000 euro - Frequenza"
)

COL_75_120 = (
    "Reddito complessivo da 75000 a 120000 euro - Frequenza"
)

COL_OVER_120 = (
    "Reddito complessivo oltre 120000 euro - Frequenza"
)


# --------------------------------------------------
# FIND FILE
# --------------------------------------------------

def find_income_file():

    files = [
        file
        for file in RAW_DIR.glob("*.csv")
        if file.is_file()
    ]

    if len(files) == 0:
        raise FileNotFoundError(
            "No income CSV found in data/raw/income/"
        )

    if len(files) > 1:
        print("Multiple CSV files found:")
        for file in files:
            print("-", file.name)

    income_file = files[0]

    print(
        f"Income file: {income_file.name}"
    )

    return income_file


# --------------------------------------------------
# READ INCOME DATA
# --------------------------------------------------

def read_income(file):

    print("\nReading MEF income data...")

    df = pd.read_csv(
        file,
        sep=";",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    # Remove possible trailing empty column.
    df = df.loc[
        :,
        [
            col
            for col in df.columns
            if col.strip() != ""
        ]
    ]

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns loaded: {len(df.columns)}"
    )

    return df


# --------------------------------------------------
# PREPARE ECONOMIC FEATURES
# --------------------------------------------------

def prepare_income(df):

    print("\nPreparing ECONOMY features...")

    required_columns = [
        COL_ISTAT,
        COL_NAME,
        COL_TAXPAYERS,
        COL_TOTAL_INCOME_FREQ,
        COL_TOTAL_INCOME_AMOUNT,
        COL_55_75,
        COL_75_120,
        COL_OVER_120,
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing expected MEF columns: {missing}"
        )

    # --------------------------------------------------
    # ISTAT CODE
    # --------------------------------------------------

    df["istat_code"] = (
        df[COL_ISTAT]
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
    # NUMERIC CONVERSION
    # --------------------------------------------------

    numeric_columns = [
        COL_TAXPAYERS,
        COL_TOTAL_INCOME_FREQ,
        COL_TOTAL_INCOME_AMOUNT,
        COL_55_75,
        COL_75_120,
        COL_OVER_120,
    ]

    for column in numeric_columns:

        # Empty cells in MEF mean no observations.
        df[column] = (
            df[column]
            .replace("", "0")
        )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    print("\nInvalid numeric values:")

    print(
        df[numeric_columns]
        .isna()
        .sum()
    )

    if df[numeric_columns].isna().any().any():

        raise ValueError(
            "Some MEF numeric values could not be converted."
        )

    # --------------------------------------------------
    # AVERAGE INCOME
    # --------------------------------------------------

    # Mean comprehensive income among individuals
    # for whom comprehensive income is reported.
    df["average_income"] = (
        df[COL_TOTAL_INCOME_AMOUNT]
        / df[COL_TOTAL_INCOME_FREQ]
    )

    # --------------------------------------------------
    # HIGH-INCOME SHARE
    # --------------------------------------------------

    df["high_income_taxpayers"] = (
        df[COL_55_75]
        + df[COL_75_120]
        + df[COL_OVER_120]
    )

    df["high_income_share"] = (
        df["high_income_taxpayers"]
        / df[COL_TOTAL_INCOME_FREQ]
        * 100
    )

    income = df[
        [
            "istat_code",
            COL_NAME,
            COL_TAXPAYERS,
            "average_income",
            "high_income_share",
        ]
    ].copy()

    income = income.rename(
        columns={
            COL_NAME: "income_name",
            COL_TAXPAYERS: "taxpayers",
        }
    )

    income["average_income"] = (
        income["average_income"]
        .round(2)
    )

    income["high_income_share"] = (
        income["high_income_share"]
        .round(2)
    )

    print(
        f"\nMunicipalities in MEF dataset: "
        f"{len(income):,}"
    )

    duplicate_codes = (
        income["istat_code"]
        .duplicated(keep=False)
    )

    if duplicate_codes.any():

        print(
            "\nDuplicate municipality codes:"
        )

        print(
            income[
                duplicate_codes
            ].to_string(index=False)
        )

        raise ValueError(
            "Duplicate municipality codes in MEF data."
        )

    print("\nIncome preview:")

    print(
        income
        .head()
        .to_string(index=False)
    )

    return income


# --------------------------------------------------
# MERGE WITH MASTER
# --------------------------------------------------

def merge_with_master(income):

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
    # CHECK DIRECT MATCH
    # --------------------------------------------------

    master_codes = set(
        master["istat_code"]
    )

    income_codes = set(
        income["istat_code"]
    )

    direct_matches = len(
        master_codes.intersection(
            income_codes
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
        "MEF municipalities:",
        len(income_codes)
    )

    # --------------------------------------------------
    # OLD -> 2026 CODE CROSSWALK
    # --------------------------------------------------

    if "istat_code_old" in master.columns:

        old_to_new = dict(
            zip(
                master["istat_code_old"],
                master["istat_code"],
            )
        )

        income[
            "istat_code_original"
        ] = income["istat_code"]

        income["istat_code"] = (
            income["istat_code"]
            .replace(old_to_new)
        )

        changed_codes = (
            income["istat_code"]
            != income[
                "istat_code_original"
            ]
        ).sum()

        print(
            "MEF codes reconciled "
            "via old→2026 crosswalk:",
            changed_codes
        )

    # --------------------------------------------------
    # DUPLICATE CHECK
    # --------------------------------------------------

    duplicates = (
        income["istat_code"]
        .duplicated(keep=False)
    )

    if duplicates.any():

        print(
            "\nDuplicate codes after reconciliation:"
        )

        print(
            income[
                duplicates
            ].to_string(index=False)
        )

        raise ValueError(
            "Duplicate MEF municipality codes "
            "after reconciliation."
        )

    # --------------------------------------------------
    # REMOVE OLD COLUMNS IF SCRIPT IS RERUN
    # --------------------------------------------------

    economy_columns = [
        "taxpayers",
        "taxpayers_per_100_residents",
        "average_income",
        "high_income_share",
    ]

    master = master.drop(
        columns=[
            col
            for col in economy_columns
            if col in master.columns
        ]
    )

    # --------------------------------------------------
    # MERGE
    # --------------------------------------------------

    merged = master.merge(
        income[
            [
                "istat_code",
                "taxpayers",
                "average_income",
                "high_income_share",
            ]
        ],
        on="istat_code",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------

    numeric_columns = [
        "population_2026",
        "taxpayers",
        "average_income",
        "high_income_share",
    ]

    for column in numeric_columns:

        merged[column] = pd.to_numeric(
            merged[column],
            errors="coerce"
        )

    # --------------------------------------------------
    # TAXPAYERS PER 100 RESIDENTS
    # --------------------------------------------------

    merged[
        "taxpayers_per_100_residents"
    ] = (
        merged["taxpayers"]
        / merged["population_2026"]
        * 100
    ).round(2)

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    print(
        "\n--- ECONOMY VALIDATION ---"
    )

    print(
        "Merged municipalities:",
        len(merged)
    )

    print(
        "\nMissing ECONOMY values:"
    )

    print(
        merged[
            [
                "taxpayers",
                "average_income",
                "high_income_share",
                "taxpayers_per_100_residents",
            ]
        ]
        .isna()
        .sum()
    )

    # --------------------------------------------------
    # TEST MUNICIPALITIES
    # --------------------------------------------------

    for municipality in [
        "Giarre",
        "Magenta",
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
                    "population_2026",
                    "taxpayers",
                    "taxpayers_per_100_residents",
                    "average_income",
                    "high_income_share",
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

    income_file = (
        find_income_file()
    )

    income_raw = read_income(
        income_file
    )

    income = prepare_income(
        income_raw
    )

    merge_with_master(
        income
    )