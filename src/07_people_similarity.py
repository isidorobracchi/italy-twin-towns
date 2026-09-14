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
# PEOPLE FEATURES
# --------------------------------------------------

PEOPLE_FEATURES = [
    "population_2026",
    "density_km2",
    "share_0_14",
    "share_65_plus",
    "mean_age",
]


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

    for feature in PEOPLE_FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    print(
        f"Municipalities loaded: {len(df):,}"
    )

    print(
        "\nMissing PEOPLE values:"
    )

    print(
        df[PEOPLE_FEATURES]
        .isna()
        .sum()
    )

    if df[PEOPLE_FEATURES].isna().any().any():

        raise ValueError(
            "Some PEOPLE features contain missing values."
        )

    return df


# --------------------------------------------------
# PREPARE PEOPLE FEATURES
# --------------------------------------------------

def prepare_people_features(df):

    print("\nPreparing PEOPLE features...")

    features = df[
        PEOPLE_FEATURES
    ].copy()

    # Population and density are highly skewed.
    # Log transformation makes comparisons
    # more meaningful across very different municipalities.
    features["population_2026"] = np.log1p(
        features["population_2026"]
    )

    features["density_km2"] = np.log1p(
        features["density_km2"]
    )

    # Robust scaling reduces the impact of outliers.
    scaler = RobustScaler()

    X = scaler.fit_transform(
        features
    )

    print(
        "PEOPLE feature matrix:",
        X.shape
    )

    return X


# --------------------------------------------------
# CALCULATE PEOPLE DISTANCE
# --------------------------------------------------

def calculate_people_distances(X):

    print("\nCalculating PEOPLE distances...")

    distances = pairwise_distances(
        X,
        metric="euclidean"
    )

    print(
        "Distance matrix:",
        distances.shape
    )

    return distances


# --------------------------------------------------
# FIND MATCHES
# --------------------------------------------------

def find_people_matches(
    df,
    distances,
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
        distances[selected_index]
    )

    # Remove selected municipality.
    results = results[
        results.index != selected_index
    ].copy()

    if exclude_same_region:

        results = results[
            results["region"]
            != selected["region"]
        ].copy()

    results = results.sort_values(
        "people_distance",
        ascending=True
    ).head(n)

    print("\n" + "=" * 90)

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
        f"{selected['density_km2']:.1f} inh./km²"
    )

    print(
        f"0–14: "
        f"{selected['share_0_14']:.2f}%"
    )

    print(
        f"65+: "
        f"{selected['share_65_plus']:.2f}%"
    )

    print(
        f"Mean age: "
        f"{selected['mean_age']:.2f}"
    )

    if exclude_same_region:

        print(
            "Filter: same region excluded"
        )

    print("\nClosest PEOPLE profiles:\n")

    columns = [
        "name",
        "province",
        "region",
        "population_2026",
        "density_km2",
        "share_0_14",
        "share_65_plus",
        "mean_age",
        "people_distance",
    ]

    print(
        results[
            columns
        ].to_string(
            index=False,
            formatters={
                "population_2026":
                    lambda x: f"{int(x):,}",
                "density_km2":
                    lambda x: f"{x:.1f}",
                "share_0_14":
                    lambda x: f"{x:.2f}",
                "share_65_plus":
                    lambda x: f"{x:.2f}",
                "mean_age":
                    lambda x: f"{x:.2f}",
                "people_distance":
                    lambda x: f"{x:.4f}",
            }
        )
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    df = load_data()

    X = prepare_people_features(
        df
    )

    distances = (
        calculate_people_distances(
            X
        )
    )

    test_municipalities = [
        "Giarre",
        "Roma",
        "Milano",
        "Bologna",
        "Niscemi",
    ]

    for municipality in test_municipalities:

        find_people_matches(
            df,
            distances,
            municipality,
            n=10,
            exclude_same_region=False,
        )

    print(
        "\n\n--- GIARRE: EXCLUDING SICILY ---"
    )

    find_people_matches(
        df,
        distances,
        "Giarre",
        n=10,
        exclude_same_region=True,
    )