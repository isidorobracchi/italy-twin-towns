from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

POPULATION_FILE = (
    ROOT
    / "data"
    / "raw"
    / "population"
    / "extracted"
    / "POSAS_2026_it_Comuni.csv"
)

MASTER_FILE = (
    ROOT
    / "data"
    / "processed"
    / "municipalities_master.csv"
)


# --------------------------------------------------
# READ POPULATION BY AGE
# --------------------------------------------------

def read_population():

    print("Reading population by age...")

    df = pd.read_csv(
        POPULATION_FILE,
        sep=";",
        skiprows=1,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
        low_memory=False,
    )

    # Remove ISTAT footer/note
    df["istat_code"] = (
        df["Codice comune"]
        .astype(str)
        .str.strip()
        .str.zfill(6)
    )

    df = df[
        df["istat_code"]
        .str.fullmatch(r"\d{6}")
    ].copy()

    # Convert age and population
    df["age"] = pd.to_numeric(
        df["Età"],
        errors="coerce"
    )

    df["population"] = pd.to_numeric(
        df["Totale"],
        errors="coerce"
    )

    # 999 = municipal total, not a real age.
    df = df[
        df["age"].between(0, 100)
    ].copy()

    print(
        f"Age rows loaded: {len(df):,}"
    )

    return df


# --------------------------------------------------
# CALCULATE DEMOGRAPHIC FEATURES
# --------------------------------------------------

def calculate_features(df):

    print("\nCalculating demographic features...")

    results = []

    for istat_code, group in df.groupby("istat_code"):

        total = group["population"].sum()

        pop_0_14 = group.loc[
            group["age"].between(0, 14),
            "population"
        ].sum()

        pop_15_64 = group.loc[
            group["age"].between(15, 64),
            "population"
        ].sum()

        pop_65_plus = group.loc[
            group["age"] >= 65,
            "population"
        ].sum()

        pop_80_plus = group.loc[
            group["age"] >= 80,
            "population"
        ].sum()

        # Weighted mean age
        mean_age = (
            group["age"]
            * group["population"]
        ).sum() / total

        share_0_14 = (
            pop_0_14 / total * 100
        )

        share_15_64 = (
            pop_15_64 / total * 100
        )

        share_65_plus = (
            pop_65_plus / total * 100
        )

        share_80_plus = (
            pop_80_plus / total * 100
        )

        ageing_index = (
            pop_65_plus / pop_0_14 * 100
            if pop_0_14 > 0
            else np.nan
        )

        results.append(
            {
                "istat_code": istat_code,
                "share_0_14": share_0_14,
                "share_15_64": share_15_64,
                "share_65_plus": share_65_plus,
                "share_80_plus": share_80_plus,
                "mean_age": mean_age,
                "ageing_index": ageing_index,
            }
        )

    demographics = pd.DataFrame(results)

    numeric_columns = [
        "share_0_14",
        "share_15_64",
        "share_65_plus",
        "share_80_plus",
        "mean_age",
        "ageing_index",
    ]

    demographics[numeric_columns] = (
        demographics[numeric_columns]
        .round(2)
    )

    print(
        f"Municipalities calculated: "
        f"{len(demographics):,}"
    )

    return demographics


# --------------------------------------------------
# MERGE WITH MASTER
# --------------------------------------------------

def merge_with_master(demographics):

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

    # Reconcile old Sardinia codes -> 2026 codes.
    if "istat_code_old" in master.columns:

        old_to_new = dict(
            zip(
                master["istat_code_old"],
                master["istat_code"],
            )
        )

        demographics["istat_code"] = (
            demographics["istat_code"]
            .replace(old_to_new)
        )

    feature_columns = [
        "share_0_14",
        "share_15_64",
        "share_65_plus",
        "share_80_plus",
        "mean_age",
        "ageing_index",
    ]

    # Allow rerunning the script.
    master = master.drop(
        columns=[
            col
            for col in feature_columns
            if col in master.columns
        ]
    )

    merged = master.merge(
        demographics,
        on="istat_code",
        how="left",
        validate="one_to_one",
    )

    print("\n--- VALIDATION ---")

    print(
        "Municipalities:",
        len(merged)
    )

    print(
        "Missing demographic values:"
    )

    print(
        merged[
            feature_columns
        ].isna().sum()
    )

    # --------------------------------------------------
    # TEST MUNICIPALITIES
    # --------------------------------------------------

    for municipality in [
        "Giarre",
        "Falconara Marittima",
        "Niscemi",
        "Milano",
        "Roma",
    ]:

        print(
            f"\n--- {municipality.upper()} ---"
        )

        print(
            merged[
                merged["name"] == municipality
            ][
                [
                    "name",
                    "population_2026",
                    "share_0_14",
                    "share_65_plus",
                    "share_80_plus",
                    "mean_age",
                    "ageing_index",
                ]
            ].to_string(index=False)
        )

    merged.to_csv(
        MASTER_FILE,
        index=False,
        encoding="utf-8",
    )

    print("\nUpdated master saved:")
    print(MASTER_FILE)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    population = read_population()

    demographics = calculate_features(
        population
    )

    merge_with_master(
        demographics
    )