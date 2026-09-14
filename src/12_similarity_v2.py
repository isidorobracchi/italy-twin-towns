from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "municipalities_2026.csv"
)


# --------------------------------------------------
# CATEGORY FEATURES
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
# CATEGORY WEIGHTS
# --------------------------------------------------

CATEGORY_WEIGHTS = {
    "people": 1 / 3,
    "economy": 1 / 3,
    "territory": 1 / 3,
}


# --------------------------------------------------
# TERRITORY INTERNAL WEIGHTS
# --------------------------------------------------

TERRITORY_WEIGHTS = {
    "area": 0.30,
    "altitude": 0.30,
    "coastal": 0.20,
    "degurba": 0.20,
}


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    print("Loading harmonised 2026 dataset...")

    df = pd.read_csv(
        DATA_FILE,
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
        ]
    )

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    print(
        f"Municipalities loaded: {len(df):,}"
    )

    print("\nMissing model values:")

    print(
        df[numeric_columns]
        .isna()
        .sum()
    )

    if df[numeric_columns].isna().any().any():

        raise ValueError(
            "Model contains missing values."
        )

    if len(df) != 7894:

        raise ValueError(
            "Expected 7,894 municipalities."
        )

    return df


# --------------------------------------------------
# PREPARE PEOPLE
# --------------------------------------------------

def prepare_people(df):

    features = df[
        PEOPLE_FEATURES
    ].copy()

    # Strongly skewed variables
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

    features["average_income"] = np.log1p(
        features["average_income"]
    )

    features["high_income_share"] = np.log1p(
        features["high_income_share"]
    )

    scaler = RobustScaler()

    return scaler.fit_transform(
        features
    )


# --------------------------------------------------
# PREPARE TERRITORY
# --------------------------------------------------

def prepare_territory(df):

    territory = pd.DataFrame(
        index=df.index
    )

    # Area is heavily skewed.
    territory["log_area"] = np.log1p(
        df["area_km2"]
    )

    # Altitude is continuous.
    territory["altitude"] = (
        df["altitude_m"]
    )

    scaler = RobustScaler()

    territory[
        [
            "log_area",
            "altitude",
        ]
    ] = scaler.fit_transform(
        territory[
            [
                "log_area",
                "altitude",
            ]
        ]
    )

    # Binary variable:
    # 0 = not coastal
    # 1 = coastal
    territory["coastal"] = (
        df["coastal"]
    )

    # DEGURBA:
    # 1 = cities
    # 2 = towns/suburbs
    # 3 = rural areas
    territory["degurba"] = (
        df["degurba"]
    )

    return territory


# --------------------------------------------------
# EUCLIDEAN CATEGORY DISTANCE
# --------------------------------------------------

def euclidean_distance(
    matrix,
    selected_index,
):

    selected = matrix[
        selected_index
    ]

    differences = (
        matrix
        - selected
    )

    # RMS rather than raw Euclidean distance.
    # This prevents categories with more features
    # from automatically producing larger distances.
    distance = np.sqrt(
        np.mean(
            differences ** 2,
            axis=1,
        )
    )

    return distance


# --------------------------------------------------
# TERRITORY DISTANCE
# --------------------------------------------------

def territory_distance(
    territory,
    selected_index,
):

    selected = territory.loc[
        selected_index
    ]

    # Continuous components
    area_distance = np.abs(
        territory["log_area"]
        - selected["log_area"]
    )

    altitude_distance = np.abs(
        territory["altitude"]
        - selected["altitude"]
    )

    # Binary mismatch:
    # same = 0
    # different = 1
    coastal_distance = (
        territory["coastal"]
        != selected["coastal"]
    ).astype(float)

    # DEGURBA ranges from 1 to 3.
    # Divide by 2 so:
    #
    # same category       = 0
    # adjacent category   = 0.5
    # opposite category   = 1
    degurba_distance = (
        np.abs(
            territory["degurba"]
            - selected["degurba"]
        )
        / 2
    )

    distance = (
        TERRITORY_WEIGHTS["area"]
        * area_distance

        + TERRITORY_WEIGHTS["altitude"]
        * altitude_distance

        + TERRITORY_WEIGHTS["coastal"]
        * coastal_distance

        + TERRITORY_WEIGHTS["degurba"]
        * degurba_distance
    )

    return distance.to_numpy()


# --------------------------------------------------
# ROBUST CATEGORY NORMALISATION
# --------------------------------------------------

def normalize_distance(distance):

    positive = distance[
        distance > 0
    ]

    median = np.median(
        positive
    )

    if median == 0:
        raise ValueError(
            "Cannot normalize category distance."
        )

    return distance / median


# --------------------------------------------------
# FIND MUNICIPALITY
# --------------------------------------------------

def get_selected_index(
    df,
    municipality_name,
):

    candidates = df[
        df["name"]
        .str.casefold()
        == municipality_name.casefold()
    ]

    if len(candidates) == 0:

        raise ValueError(
            f"Municipality not found: "
            f"{municipality_name}"
        )

    if len(candidates) > 1:

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

        raise ValueError(
            f"Multiple municipalities named "
            f"{municipality_name}."
        )

    return candidates.index[0]


# --------------------------------------------------
# FIND MATCHES
# --------------------------------------------------

def find_matches(
    df,
    people_matrix,
    economy_matrix,
    territory_matrix,
    municipality_name,
    n=10,
    exclude_same_region=False,
):

    selected_index = (
        get_selected_index(
            df,
            municipality_name,
        )
    )

    selected = df.loc[
        selected_index
    ]

    # --------------------------------------------------
    # PEOPLE
    # --------------------------------------------------

    people = euclidean_distance(
        people_matrix,
        selected_index,
    )

    people = normalize_distance(
        people
    )

    # --------------------------------------------------
    # ECONOMY
    # --------------------------------------------------

    economy = euclidean_distance(
        economy_matrix,
        selected_index,
    )

    economy = normalize_distance(
        economy
    )

    # --------------------------------------------------
    # TERRITORY
    # --------------------------------------------------

    territory = territory_distance(
        territory_matrix,
        selected_index,
    )

    territory = normalize_distance(
        territory
    )

    # --------------------------------------------------
    # COMBINED DISTANCE
    # --------------------------------------------------

    combined = (
        CATEGORY_WEIGHTS["people"]
        * people

        + CATEGORY_WEIGHTS["economy"]
        * economy

        + CATEGORY_WEIGHTS["territory"]
        * territory
    )

    results = df.copy()

    results[
        "people_distance"
    ] = people

    results[
        "economy_distance"
    ] = economy

    results[
        "territory_distance"
    ] = territory

    results[
        "combined_distance"
    ] = combined

    # Remove itself
    results = results[
        results.index
        != selected_index
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

    # --------------------------------------------------
    # PRINT
    # --------------------------------------------------

    print(
        "\n"
        + "=" * 120
    )

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
        f"Income: "
        f"€{selected['average_income']:,.0f}"
    )

    print(
        f"Area: "
        f"{selected['area_km2']:.1f} km²"
    )

    print(
        f"Altitude: "
        f"{selected['altitude_m']:.0f} m"
    )

    print(
        f"Coastal: "
        f"{'yes' if selected['coastal'] == 1 else 'no'}"
    )

    print(
        f"DEGURBA: "
        f"{selected['degurba']:.0f}"
    )

    if exclude_same_region:

        print(
            "Filter: same region excluded"
        )

    print(
        "\nClosest PEOPLE + ECONOMY + TERRITORY profiles:\n"
    )

    columns = [
        "name",
        "province",
        "region",
        "population_2026",
        "average_income",
        "area_km2",
        "altitude_m",
        "coastal",
        "degurba",
        "people_distance",
        "economy_distance",
        "territory_distance",
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

                "area_km2":
                    lambda x: f"{x:.1f}",

                "altitude_m":
                    lambda x: f"{x:.0f}",

                "coastal":
                    lambda x: "Y" if x == 1 else "N",

                "degurba":
                    lambda x: f"{x:.0f}",

                "people_distance":
                    lambda x: f"{x:.3f}",

                "economy_distance":
                    lambda x: f"{x:.3f}",

                "territory_distance":
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

    print(
        "\nPreparing model features..."
    )

    people_matrix = (
        prepare_people(
            df
        )
    )

    economy_matrix = (
        prepare_economy(
            df
        )
    )

    territory_matrix = (
        prepare_territory(
            df
        )
    )

    test_municipalities = [
        "Giarre",
        "Roma",
        "Milano",
        "Bologna",
        "Niscemi",
        "Montalto Pavese",
        "Castegnero Nanto",
    ]

    for municipality in (
        test_municipalities
    ):

        find_matches(
            df,
            people_matrix,
            economy_matrix,
            territory_matrix,
            municipality,
            n=10,
            exclude_same_region=False,
        )

    print(
        "\n\n--- GIARRE: EXCLUDING SICILY ---"
    )

    find_matches(
        df,
        people_matrix,
        economy_matrix,
        territory_matrix,
        "Giarre",
        n=10,
        exclude_same_region=True,
    )