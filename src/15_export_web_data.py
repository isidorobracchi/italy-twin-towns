from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "municipalities_2026.csv"
)

OUTPUT_FILE = (
    ROOT
    / "docs"
    / "data"
    / "municipalities.json"
)


# --------------------------------------------------
# MODEL FEATURES
# --------------------------------------------------

PEOPLE_FEATURES = [
    "population_2026",
    "density_km2",
    "share_0_14",
    "share_65_plus",
    "mean_age",
]

ECONOMY_FEATURES = [
    "average_income",
    "taxpayers_per_100_residents",
    "high_income_share",
]


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    print("Loading harmonised municipality dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        dtype={
            "istat_code": str,
            "istat_code_old": str,
        },
        keep_default_na=False,
    )

    df["istat_code"] = (
        df["istat_code"]
        .astype(str)
        .str.zfill(6)
    )

    numeric_columns = (
        PEOPLE_FEATURES
        + ECONOMY_FEATURES
        + [
            "area_km2",
            "altitude_m",
            "coastal",
            "degurba",
            "latitude",
            "longitude",
        ]
    )

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    if len(df) != 7894:
        raise ValueError(
            "Expected exactly 7,894 municipalities."
        )

    print(
        f"Municipalities loaded: {len(df):,}"
    )

    return df


# --------------------------------------------------
# PREPARE PEOPLE FEATURES
# --------------------------------------------------

def prepare_people(df):

    features = df[
        PEOPLE_FEATURES
    ].copy()

    features["population_2026"] = np.log1p(
        features["population_2026"]
    )

    features["density_km2"] = np.log1p(
        features["density_km2"]
    )

    scaler = RobustScaler()

    scaled = scaler.fit_transform(
        features
    )

    scaled_df = pd.DataFrame(
        scaled,
        columns=[
            "people_population",
            "people_density",
            "people_young",
            "people_elderly",
            "people_mean_age",
        ],
        index=df.index,
    )

    return scaled_df


# --------------------------------------------------
# PREPARE ECONOMY FEATURES
# --------------------------------------------------

def prepare_economy(df):

    features = df[
        ECONOMY_FEATURES
    ].copy()

    features["average_income"] = np.log1p(
        features["average_income"]
    )

    features["high_income_share"] = np.log1p(
        features["high_income_share"]
    )

    scaler = RobustScaler()

    scaled = scaler.fit_transform(
        features
    )

    scaled_df = pd.DataFrame(
        scaled,
        columns=[
            "economy_income",
            "economy_taxpayers",
            "economy_high_income",
        ],
        index=df.index,
    )

    return scaled_df


# --------------------------------------------------
# PREPARE TERRITORY FEATURES
# --------------------------------------------------

def prepare_territory(df):

    continuous = pd.DataFrame(
        {
            "territory_area": np.log1p(
                df["area_km2"]
            ),

            "territory_altitude":
                df["altitude_m"],
        },
        index=df.index,
    )

    scaler = RobustScaler()

    scaled = scaler.fit_transform(
        continuous
    )

    territory = pd.DataFrame(
        scaled,
        columns=[
            "territory_area",
            "territory_altitude",
        ],
        index=df.index,
    )

    # Keep binary/categorical values unscaled.
    territory["territory_coastal"] = (
        df["coastal"]
        .astype(int)
    )

    territory["territory_degurba"] = (
        df["degurba"]
        .astype(int)
    )

    return territory


# --------------------------------------------------
# BUILD WEB DATASET
# --------------------------------------------------

def build_web_dataset(df):

    print("\nPreparing model features...")

    people = prepare_people(
        df
    )

    economy = prepare_economy(
        df
    )

    territory = prepare_territory(
        df
    )

    web = pd.concat(
        [
            df.reset_index(drop=True),
            people.reset_index(drop=True),
            economy.reset_index(drop=True),
            territory.reset_index(drop=True),
        ],
        axis=1,
    )

    # --------------------------------------------------
    # KEEP ONLY FIELDS NEEDED BY FRONTEND
    # --------------------------------------------------

    columns = [

        # Identity
        "istat_code",
        "name",
        "province",
        "province_abbr",
        "region",

        # Location
        "latitude",
        "longitude",

        # Display values
        "population_2026",
        "density_km2",

        "share_0_14",
        "share_65_plus",
        "mean_age",

        "average_income",
        "taxpayers_per_100_residents",
        "high_income_share",

        "area_km2",
        "altitude_m",
        "coastal",
        "degurba",

        # PEOPLE model values
        "people_population",
        "people_density",
        "people_young",
        "people_elderly",
        "people_mean_age",

        # ECONOMY model values
        "economy_income",
        "economy_taxpayers",
        "economy_high_income",

        # TERRITORY model values
        "territory_area",
        "territory_altitude",
        "territory_coastal",
        "territory_degurba",
    ]

    missing = [
        column
        for column in columns
        if column not in web.columns
    ]

    if missing:
        raise KeyError(
            f"Missing web export columns: {missing}"
        )

    web = web[
        columns
    ].copy()

    # --------------------------------------------------
    # ROUND VALUES
    # --------------------------------------------------

    rounding = {

        "latitude": 6,
        "longitude": 6,

        "density_km2": 2,
        "share_0_14": 2,
        "share_65_plus": 2,
        "mean_age": 2,

        "average_income": 2,
        "taxpayers_per_100_residents": 2,
        "high_income_share": 2,

        "area_km2": 3,

        "people_population": 6,
        "people_density": 6,
        "people_young": 6,
        "people_elderly": 6,
        "people_mean_age": 6,

        "economy_income": 6,
        "economy_taxpayers": 6,
        "economy_high_income": 6,

        "territory_area": 6,
        "territory_altitude": 6,
    }

    for column, decimals in (
        rounding.items()
    ):

        web[column] = (
            web[column]
            .round(decimals)
        )

    return web


# --------------------------------------------------
# SAVE JSON
# --------------------------------------------------

def save_json(web):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = web.to_dict(
        orient="records"
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            records,
            file,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    size_mb = (
        OUTPUT_FILE.stat().st_size
        / 1024
        / 1024
    )

    print(
        "\n--- WEB EXPORT VALIDATION ---"
    )

    print(
        "Municipalities exported:",
        len(records)
    )

    print(
        f"JSON size: {size_mb:.2f} MB"
    )

    print(
        "Output:"
    )

    print(
        OUTPUT_FILE
    )


# --------------------------------------------------
# TEST MUNICIPALITIES
# --------------------------------------------------

def print_examples(web):

    print(
        "\n--- EXAMPLE RECORDS ---"
    )

    for municipality in [
        "Giarre",
        "Roma",
        "Milano",
        "Niscemi",
        "Castegnero Nanto",
    ]:

        row = web[
            web["name"]
            == municipality
        ]

        print(
            f"\n{municipality}:"
        )

        print(
            row.to_dict(
                orient="records"
            )[0]
        )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    df = load_data()

    web = build_web_dataset(
        df
    )

    print_examples(
        web
    )

    save_json(
        web
    )