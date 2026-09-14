from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import pairwise_distances


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MASTER_FILE = (
    ROOT
    / "data"
    / "processed"
    / "municipalities_master.csv"
)


# --------------------------------------------------
# FEATURES
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


# Default category weights
PEOPLE_WEIGHT = 0.50
ECONOMY_WEIGHT = 0.50


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    print("Loading municipality master...")

    df = pd.read_csv(
        MASTER_FILE,
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

    all_features = (
        PEOPLE_FEATURES
        + ECONOMY_FEATURES
    )

    for feature in all_features:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    print(
        f"Municipalities loaded: {len(df):,}"
    )

    print("\nMissing values:")

    print(
        df[all_features]
        .isna()
        .sum()
    )

    if df[all_features].isna().any().any():

        raise ValueError(
            "Some model features contain missing values."
        )

    return df


# --------------------------------------------------
# PREPARE PEOPLE
# --------------------------------------------------

def prepare_people(df):

    features = df[
        PEOPLE_FEATURES
    ].copy()

    # Highly skewed variables
    features["population_2026"] = np.log1p(
        features["population_2026"]
    )

    features["density_km2"] = np.log1p(
        features["density_km2"]
    )

    scaler = RobustScaler()

    return scaler.fit_transform(
        features
    )


# --------------------------------------------------
# PREPARE ECONOMY
# --------------------------------------------------

def prepare_economy(df):

    features = df[
        ECONOMY_FEATURES
    ].copy()

    # Income is mildly skewed.
    features["average_income"] = np.log1p(
        features["average_income"]
    )

    # high_income_share is also skewed,
    # especially in wealthier municipalities.
    features["high_income_share"] = np.log1p(
        features["high_income_share"]
    )

    scaler = RobustScaler()

    return scaler.fit_transform(
        features
    )


# --------------------------------------------------
# DISTANCES
# --------------------------------------------------

def calculate_distances(X):

    return pairwise_distances(
        X,
        metric="euclidean"
    )


# --------------------------------------------------
# NORMALISE CATEGORY DISTANCES
# --------------------------------------------------

def normalize_distance_matrix(matrix):

    # We use the median of non-zero distances
    # as a robust reference scale.
    non_zero = matrix[
        matrix > 0
    ]

    median_distance = np.median(
        non_zero
    )

    return matrix / median_distance


# --------------------------------------------------
# BUILD MODEL
# --------------------------------------------------

def build_model(df):

    print("\nPreparing PEOPLE category...")

    X_people = prepare_people(
        df
    )

    people_distance = (
        calculate_distances(
            X_people
        )
    )

    print(
        "PEOPLE distance matrix:",
        people_distance.shape
    )

    print("\nPreparing ECONOMY category...")

    X_economy = prepare_economy(
        df
    )

    economy_distance = (
        calculate_distances(
            X_economy
        )
    )

    print(
        "ECONOMY distance matrix:",
        economy_distance.shape
    )

    # Important:
    # categories have different numbers of features
    # and different natural distance scales.
    # Normalise them before weighting.
    people_norm = normalize_distance_matrix(
        people_distance
    )

    economy_norm = normalize_distance_matrix(
        economy_distance
    )

    combined_distance = (
        PEOPLE_WEIGHT
        * people_norm
        +
        ECONOMY_WEIGHT
        * economy_norm
    )

    return (
        people_norm,
        economy_norm,
        combined_distance,
    )


# --------------------------------------------------
# FIND MATCHES
# --------------------------------------------------

def find_matches(
    df,
    people_distance,
    economy_distance,
    combined_distance,
    municipality_name,
    n=10,
    exclude_same_region=False,
):

    candidates = df[
        df["name"]
        .str.casefold()
        == municipality_name.casefold()
    ]

    if len(candidates) == 0:

        print(
            f"\nMunicipality not found: "
            f"{municipality_name}"
        )

        return

    if len(candidates) > 1:

        print(
            f"\nMultiple municipalities named "
            f"{municipality_name}:"
        )

        print(
            candidates[
                [
                    "istat_code",
                    "name",
                    "province",
                    "region",
                ]
            ].to_string(index=False)
        )

        return

    selected_index = candidates.index[0]

    selected = df.loc[
        selected_index
    ]

    results = df.copy()

    results["people_distance"] = (
        people_distance[
            selected_index
        ]
    )

    results["economy_distance"] = (
        economy_distance[
            selected_index
        ]
    )

    results["combined_distance"] = (
        combined_distance[
            selected_index
        ]
    )

    # Remove selected municipality
    results = results[
        results.index != selected_index
    ].copy()

    if exclude_same_region:

        results = results[
            results["region"]
            != selected["region"]
        ].copy()

    results = (
        results
        .sort_values(
            "combined_distance"
        )
        .head(n)
    )

    print("\n" + "=" * 105)

    print(
        f"{selected['name'].upper()} "
        f"({selected['province']}, "
        f"{selected['region']})"
    )

    print(
        f"Population: "
        f"{int(selected['population_2026']):,}"
    )

    print(
        f"Density: "
        f"{selected['density_km2']:.1f}"
    )

    print(
        f"Mean age: "
        f"{selected['mean_age']:.2f}"
    )

    print(
        f"Average income: "
        f"€{selected['average_income']:,.0f}"
    )

    print(
        f"Taxpayers / 100 residents: "
        f"{selected['taxpayers_per_100_residents']:.2f}"
    )

    print(
        f"High income share: "
        f"{selected['high_income_share']:.2f}%"
    )

    if exclude_same_region:

        print(
            "Filter: same region excluded"
        )

    print(
        "\nClosest PEOPLE + ECONOMY profiles:\n"
    )

    columns = [
        "name",
        "province",
        "region",
        "population_2026",
        "average_income",
        "taxpayers_per_100_residents",
        "high_income_share",
        "people_distance",
        "economy_distance",
        "combined_distance",
    ]

    print(
        results[
            columns
        ].to_string(
            index=False,
            formatters={
                "population_2026":
                    lambda x: f"{int(x):,}",

                "average_income":
                    lambda x: f"€{x:,.0f}",

                "taxpayers_per_100_residents":
                    lambda x: f"{x:.2f}",

                "high_income_share":
                    lambda x: f"{x:.2f}",

                "people_distance":
                    lambda x: f"{x:.3f}",

                "economy_distance":
                    lambda x: f"{x:.3f}",

                "combined_distance":
                    lambda x: f"{x:.3f}",
            }
        )
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    df = load_data()

    (
        people_distance,
        economy_distance,
        combined_distance,
    ) = build_model(
        df
    )

    test_municipalities = [
        "Giarre",
        "Roma",
        "Milano",
        "Bologna",
        "Niscemi",
    ]

    for municipality in test_municipalities:

        find_matches(
            df,
            people_distance,
            economy_distance,
            combined_distance,
            municipality,
            n=10,
            exclude_same_region=False,
        )

    print(
        "\n\n--- GIARRE: EXCLUDING SICILY ---"
    )

    find_matches(
        df,
        people_distance,
        economy_distance,
        combined_distance,
        "Giarre",
        n=10,
        exclude_same_region=True,
    )