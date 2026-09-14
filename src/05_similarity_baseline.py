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
# SETTINGS
# --------------------------------------------------

FEATURES = [
    "population_2026",
    "density_km2",
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

    for feature in FEATURES:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    print(
        f"Municipalities loaded: {len(df):,}"
    )

    print(
        "Missing feature values:",
        df[FEATURES].isna().sum().sum()
    )

    return df


# --------------------------------------------------
# PREPARE FEATURES
# --------------------------------------------------

def prepare_features(df):

    print("\nPreparing features...")

    feature_data = df[
        FEATURES
    ].copy()

    # Population and density are strongly skewed.
    # log1p compresses extreme values.
    for feature in FEATURES:

        feature_data[
            f"log_{feature}"
        ] = np.log1p(
            feature_data[feature]
        )

    model_features = [
        f"log_{feature}"
        for feature in FEATURES
    ]

    # RobustScaler reduces the influence of
    # extreme municipalities.
    scaler = RobustScaler()

    X = scaler.fit_transform(
        feature_data[model_features]
    )

    print(
        "Feature matrix shape:",
        X.shape
    )

    return X


# --------------------------------------------------
# CALCULATE DISTANCES
# --------------------------------------------------

def calculate_distances(X):

    print("\nCalculating municipality distances...")

    distances = pairwise_distances(
        X,
        metric="euclidean"
    )

    print(
        "Distance matrix shape:",
        distances.shape
    )

    return distances


# --------------------------------------------------
# FIND MATCHES
# --------------------------------------------------

def find_matches(
    df,
    distances,
    municipality_name,
    n=10,
    exclude_same_region=False,
):

    candidates = df[
        df["name"].str.casefold()
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

    selected = df.loc[selected_index]

    results = df.copy()

    results["distance"] = (
        distances[selected_index]
    )

    # Remove the selected municipality itself.
    results = results[
        results.index != selected_index
    ].copy()

    if exclude_same_region:

        results = results[
            results["region"]
            != selected["region"]
        ].copy()

    results = results.sort_values(
        "distance",
        ascending=True
    ).head(n)

    print("\n" + "=" * 70)

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
        f"{selected['density_km2']:.1f} "
        f"inh./km²"
    )

    if exclude_same_region:
        print(
            "Filter: same region excluded"
        )

    print("\nClosest municipalities:\n")

    output_columns = [
        "name",
        "province",
        "region",
        "population_2026",
        "density_km2",
        "distance",
    ]

    print(
        results[
            output_columns
        ].to_string(
            index=False,
            formatters={
                "population_2026":
                    lambda x: f"{int(x):,}",
                "density_km2":
                    lambda x: f"{x:.1f}",
                "distance":
                    lambda x: f"{x:.4f}",
            }
        )
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    df = load_data()

    X = prepare_features(
        df
    )

    distances = calculate_distances(
        X
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
            distances,
            municipality,
            n=10,
            exclude_same_region=False,
        )

    print(
        "\n\n--- GIARRE: EXCLUDING SICILY ---"
    )

    find_matches(
        df,
        distances,
        "Giarre",
        n=10,
        exclude_same_region=True,
    )