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


CATEGORY_WEIGHTS = {
    "people": 1 / 3,
    "economy": 1 / 3,
    "territory": 1 / 3,
}


TERRITORY_WEIGHTS = {
    "area": 0.30,
    "altitude": 0.30,
    "coastal": 0.20,
    "degurba": 0.20,
}


# --------------------------------------------------
# PEOPLE WEIGHT SCENARIOS
# --------------------------------------------------

PEOPLE_SCENARIOS = {

    "A_EQUAL": {
        "population_2026": 0.20,
        "density_km2": 0.20,
        "share_0_14": 0.20,
        "share_65_plus": 0.20,
        "mean_age": 0.20,
    },

    "B_SCALE_AWARE": {
        "population_2026": 0.35,
        "density_km2": 0.20,
        "share_0_14": 0.15,
        "share_65_plus": 0.15,
        "mean_age": 0.15,
    },

    "C_SCALE_STRONG": {
        "population_2026": 0.45,
        "density_km2": 0.20,
        "share_0_14": 0.10,
        "share_65_plus": 0.15,
        "mean_age": 0.10,
    },
}


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    df = pd.read_csv(
        DATA_FILE,
        dtype={
            "istat_code": str,
            "istat_code_old": str,
        },
        keep_default_na=False,
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
            errors="raise",
        )

    return df


# --------------------------------------------------
# PREPARE PEOPLE
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

    return pd.DataFrame(
        scaled,
        columns=PEOPLE_FEATURES,
        index=df.index,
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

    territory["log_area"] = np.log1p(
        df["area_km2"]
    )

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

    territory["coastal"] = (
        df["coastal"]
    )

    territory["degurba"] = (
        df["degurba"]
    )

    return territory


# --------------------------------------------------
# NORMALISATION
# --------------------------------------------------

def normalize_distance(distance):

    positive = distance[
        distance > 0
    ]

    median = np.median(
        positive
    )

    return distance / median


# --------------------------------------------------
# PEOPLE DISTANCE
# --------------------------------------------------

def weighted_people_distance(
    people,
    selected_index,
    weights,
):

    differences = (
        people
        - people.loc[selected_index]
    ) ** 2

    distance = np.zeros(
        len(people)
    )

    for feature, weight in weights.items():

        distance += (
            weight
            * differences[feature]
        )

    return np.sqrt(
        distance
    )


# --------------------------------------------------
# ECONOMY DISTANCE
# --------------------------------------------------

def economy_distance(
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

    return np.sqrt(
        np.mean(
            differences ** 2,
            axis=1,
        )
    )


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

    area = np.abs(
        territory["log_area"]
        - selected["log_area"]
    )

    altitude = np.abs(
        territory["altitude"]
        - selected["altitude"]
    )

    coastal = (
        territory["coastal"]
        != selected["coastal"]
    ).astype(float)

    degurba = (
        np.abs(
            territory["degurba"]
            - selected["degurba"]
        )
        / 2
    )

    return (
        TERRITORY_WEIGHTS["area"]
        * area

        + TERRITORY_WEIGHTS["altitude"]
        * altitude

        + TERRITORY_WEIGHTS["coastal"]
        * coastal

        + TERRITORY_WEIGHTS["degurba"]
        * degurba
    ).to_numpy()


# --------------------------------------------------
# FIND MUNICIPALITY
# --------------------------------------------------

def find_index(
    df,
    name,
):

    candidates = df[
        df["name"]
        .str.casefold()
        == name.casefold()
    ]

    if len(candidates) != 1:
        raise ValueError(
            f"Problem finding {name}"
        )

    return candidates.index[0]


# --------------------------------------------------
# TEST ONE MUNICIPALITY
# --------------------------------------------------

def test_municipality(
    df,
    people,
    economy,
    territory,
    name,
):

    selected_index = (
        find_index(
            df,
            name,
        )
    )

    economy_dist = normalize_distance(
        economy_distance(
            economy,
            selected_index,
        )
    )

    territory_dist = normalize_distance(
        territory_distance(
            territory,
            selected_index,
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        name.upper()
    )

    for scenario_name, weights in (
        PEOPLE_SCENARIOS.items()
    ):

        people_dist = normalize_distance(
            weighted_people_distance(
                people,
                selected_index,
                weights,
            )
        )

        combined = (
            CATEGORY_WEIGHTS["people"]
            * people_dist

            + CATEGORY_WEIGHTS["economy"]
            * economy_dist

            + CATEGORY_WEIGHTS["territory"]
            * territory_dist
        )

        results = df[
            [
                "name",
                "province",
                "region",
                "population_2026",
            ]
        ].copy()

        results["distance"] = (
            combined
        )

        results = results[
            results.index
            != selected_index
        ]

        results = (
            results
            .sort_values(
                "distance"
            )
            .head(5)
        )

        print(
            f"\n--- {scenario_name} ---"
        )

        print(
            results.to_string(
                index=False,
                formatters={
                    "population_2026":
                        lambda x: f"{int(x):,}",

                    "distance":
                        lambda x: f"{x:.3f}",
                }
            )
        )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    df = load_data()

    people = prepare_people(
        df
    )

    economy = prepare_economy(
        df
    )

    territory = prepare_territory(
        df
    )

    municipalities = [
        "Giarre",
        "Roma",
        "Milano",
        "Bologna",
        "Niscemi",
    ]

    for municipality in municipalities:

        test_municipality(
            df,
            people,
            economy,
            territory,
            municipality,
        )