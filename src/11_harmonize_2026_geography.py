from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


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

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "municipalities_2026.csv"
)

POPULATION_FILE = (
    ROOT
    / "data"
    / "raw"
    / "population"
    / "extracted"
    / "POSAS_2026_it_Comuni.csv"
)

INCOME_DIR = (
    ROOT
    / "data"
    / "raw"
    / "income"
)

TERRITORY_DIR = (
    ROOT
    / "data"
    / "raw"
    / "territory"
)

BOUNDARIES_DIR = (
    ROOT
    / "data"
    / "raw"
    / "boundaries"
)


# --------------------------------------------------
# ADMINISTRATIVE CHANGES
# --------------------------------------------------

MERGERS = {

    # Lirio incorporated into Montalto Pavese
    "018094": {
        "name": "Montalto Pavese",
        "sources": [
            "018094",
            "018082",
        ],
        "template": "018094",
        "cadastral_code": None,
    },

    # Castegnero + Nanto
    "024129": {
        "name": "Castegnero Nanto",
        "sources": [
            "024027",
            "024071",
        ],
        "template": "024027",
        "cadastral_code": "M439",
    },
}


SOURCE_CODES_TO_REMOVE = {
    "018082",
    "024027",
    "024071",
}


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def find_single_csv(directory):

    files = list(
        directory.glob("*.csv")
    )

    if len(files) == 0:
        raise FileNotFoundError(
            f"No CSV found in {directory}"
        )

    if len(files) > 1:
        print(
            f"Multiple CSV files found in {directory}:"
        )

        for file in files:
            print("-", file.name)

    return files[0]


def find_municipality_shapefile():

    shapefiles = list(
        BOUNDARIES_DIR.rglob(
            "Com*_WGS84.shp"
        )
    )

    if len(shapefiles) == 0:

        shapefiles = list(
            BOUNDARIES_DIR.rglob(
                "Com*.shp"
            )
        )

    if len(shapefiles) == 0:
        raise FileNotFoundError(
            "Municipality shapefile not found."
        )

    return shapefiles[0]


# --------------------------------------------------
# LOAD MASTER
# --------------------------------------------------

def load_master():

    print(
        "Loading 1 January 2026 master..."
    )

    df = pd.read_csv(
        MASTER_FILE,
        dtype=str,
        keep_default_na=False,
    )

    df["istat_code"] = (
        df["istat_code"]
        .astype(str)
        .str.zfill(6)
    )

    print(
        f"Master municipalities: {len(df):,}"
    )

    if len(df) != 7896:
        raise ValueError(
            "Expected 7,896 municipalities "
            "in municipalities_master.csv."
        )

    return df


# --------------------------------------------------
# POPULATION + DEMOGRAPHICS
# --------------------------------------------------

def load_population_raw():

    print(
        "\nLoading raw population by age..."
    )

    df = pd.read_csv(
        POPULATION_FILE,
        sep=";",
        skiprows=1,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
        low_memory=False,
    )

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

    df["age"] = pd.to_numeric(
        df["Età"],
        errors="coerce",
    )

    df["population"] = pd.to_numeric(
        df["Totale"],
        errors="coerce",
    )

    return df


def calculate_population_features(
    population_raw,
    source_codes,
):

    subset = population_raw[
        population_raw[
            "istat_code"
        ].isin(source_codes)
    ].copy()

    # Exact total from ISTAT total rows
    total_rows = subset[
        subset["age"] == 999
    ]

    population_total = (
        total_rows["population"]
        .sum()
    )

    # Age-specific rows
    ages = subset[
        subset["age"].between(
            0,
            100,
        )
    ].copy()

    age_totals = (
        ages
        .groupby(
            "age",
            as_index=False,
        )["population"]
        .sum()
    )

    age_population_total = (
        age_totals["population"]
        .sum()
    )

    if population_total != age_population_total:

        raise ValueError(
            "Population total and age-sum "
            "do not match."
        )

    pop_0_14 = (
        age_totals.loc[
            age_totals["age"].between(
                0,
                14,
            ),
            "population",
        ]
        .sum()
    )

    pop_15_64 = (
        age_totals.loc[
            age_totals["age"].between(
                15,
                64,
            ),
            "population",
        ]
        .sum()
    )

    pop_65_plus = (
        age_totals.loc[
            age_totals["age"] >= 65,
            "population",
        ]
        .sum()
    )

    pop_80_plus = (
        age_totals.loc[
            age_totals["age"] >= 80,
            "population",
        ]
        .sum()
    )

    mean_age = (
        (
            age_totals["age"]
            * age_totals["population"]
        ).sum()
        / population_total
    )

    share_0_14 = (
        pop_0_14
        / population_total
        * 100
    )

    share_15_64 = (
        pop_15_64
        / population_total
        * 100
    )

    share_65_plus = (
        pop_65_plus
        / population_total
        * 100
    )

    share_80_plus = (
        pop_80_plus
        / population_total
        * 100
    )

    ageing_index = (
        pop_65_plus
        / pop_0_14
        * 100
        if pop_0_14 > 0
        else np.nan
    )

    return {
        "population_2026":
            int(population_total),

        "share_0_14":
            round(share_0_14, 2),

        "share_15_64":
            round(share_15_64, 2),

        "share_65_plus":
            round(share_65_plus, 2),

        "share_80_plus":
            round(share_80_plus, 2),

        "mean_age":
            round(mean_age, 2),

        "ageing_index":
            round(ageing_index, 2),
    }


# --------------------------------------------------
# INCOME
# --------------------------------------------------

def load_income_raw():

    income_file = find_single_csv(
        INCOME_DIR
    )

    print(
        f"\nLoading raw income: "
        f"{income_file.name}"
    )

    df = pd.read_csv(
        income_file,
        sep=";",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    df["istat_code"] = (
        df["Codice Istat Comune"]
        .astype(str)
        .str.strip()
        .str.zfill(6)
    )

    return df


def calculate_income_features(
    income_raw,
    source_codes,
    population_total,
):

    columns = {

        "taxpayers":
            "Numero contribuenti",

        "total_income_freq":
            "Reddito complessivo - Frequenza",

        "total_income_amount":
            "Reddito complessivo - Ammontare in euro",

        "income_55_75":
            "Reddito complessivo da 55000 a 75000 euro - Frequenza",

        "income_75_120":
            "Reddito complessivo da 75000 a 120000 euro - Frequenza",

        "income_over_120":
            "Reddito complessivo oltre 120000 euro - Frequenza",
    }

    subset = income_raw[
        income_raw[
            "istat_code"
        ].isin(source_codes)
    ].copy()

    if len(subset) != len(source_codes):

        raise ValueError(
            "Not all source municipalities "
            "were found in income data."
        )

    for new_name, source_col in columns.items():

        subset[new_name] = (
            subset[source_col]
            .replace("", "0")
        )

        subset[new_name] = (
            pd.to_numeric(
                subset[new_name],
                errors="raise",
            )
        )

    taxpayers = (
        subset["taxpayers"]
        .sum()
    )

    total_income_freq = (
        subset["total_income_freq"]
        .sum()
    )

    total_income_amount = (
        subset["total_income_amount"]
        .sum()
    )

    high_income_taxpayers = (
        subset["income_55_75"].sum()
        + subset["income_75_120"].sum()
        + subset["income_over_120"].sum()
    )

    average_income = (
        total_income_amount
        / total_income_freq
    )

    high_income_share = (
        high_income_taxpayers
        / total_income_freq
        * 100
    )

    taxpayers_per_100_residents = (
        taxpayers
        / population_total
        * 100
    )

    return {
        "taxpayers":
            int(taxpayers),

        "average_income":
            round(
                average_income,
                2,
            ),

        "high_income_share":
            round(
                high_income_share,
                2,
            ),

        "taxpayers_per_100_residents":
            round(
                taxpayers_per_100_residents,
                2,
            ),
    }


# --------------------------------------------------
# TERRITORY
# --------------------------------------------------

def load_current_territory():

    territory_file = find_single_csv(
        TERRITORY_DIR
    )

    print(
        f"\nLoading current territory: "
        f"{territory_file.name}"
    )

    df = pd.read_csv(
        territory_file,
        sep=";",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    df["istat_code"] = (
        df[
            "Codice Comune (alfanumerico)"
        ]
        .astype(str)
        .str.strip()
        .str.zfill(6)
    )

    return df


def get_territory_features(
    territory,
    target_code,
):

    row = territory[
        territory["istat_code"]
        == target_code
    ]

    if len(row) != 1:

        raise ValueError(
            f"Expected exactly one territory "
            f"row for {target_code}, found "
            f"{len(row)}."
        )

    row = row.iloc[0]

    return {
        "coastal":
            int(row["Comune litoraneo"]),

        "altimetric_zone":
            int(row["Zona altimetrica"]),

        "altitude_m":
            int(row["Altitudine (Municipio)"]),

        "coastal_zone":
            int(row["Zone costiere 2021"]),

        "degurba":
            int(row["Degurba 2021"]),
    }


# --------------------------------------------------
# GEOGRAPHY
# --------------------------------------------------

def load_boundaries():

    shapefile = (
        find_municipality_shapefile()
    )

    print(
        f"\nLoading boundaries: "
        f"{shapefile.name}"
    )

    gdf = gpd.read_file(
        shapefile
    )

    if "PRO_COM_T" in gdf.columns:

        gdf["istat_code"] = (
            gdf["PRO_COM_T"]
            .astype(str)
            .str.zfill(6)
        )

    elif "PRO_COM" in gdf.columns:

        gdf["istat_code"] = (
            gdf["PRO_COM"]
            .astype(str)
            .str.zfill(6)
        )

    else:

        raise KeyError(
            "No municipality ISTAT code "
            "found in shapefile."
        )

    return gdf


def calculate_merged_geography(
    boundaries,
    source_codes,
):

    subset = boundaries[
        boundaries[
            "istat_code"
        ].isin(source_codes)
    ].copy()

    if len(subset) != len(source_codes):

        raise ValueError(
            "Not all source geometries "
            "were found."
        )

    metric = subset.to_crs(
        epsg=3035
    )

    merged_geometry = (
        metric.geometry
        .union_all()
    )

    area_km2 = (
        merged_geometry.area
        / 1_000_000
    )

    representative_point = (
        merged_geometry
        .representative_point()
    )

    point_gdf = gpd.GeoDataFrame(
        geometry=[
            representative_point
        ],
        crs="EPSG:3035",
    ).to_crs(
        epsg=4326
    )

    point = (
        point_gdf
        .geometry
        .iloc[0]
    )

    return {
        "area_km2":
            round(
                area_km2,
                3,
            ),

        "latitude":
            round(
                point.y,
                6,
            ),

        "longitude":
            round(
                point.x,
                6,
            ),
    }


# --------------------------------------------------
# BUILD MERGED ROW
# --------------------------------------------------

def build_target_row(
    master,
    target_code,
    config,
    population_raw,
    income_raw,
    territory,
    boundaries,
):

    print(
        f"\nBuilding {config['name']} "
        f"({target_code})..."
    )

    template_code = (
        config["template"]
    )

    template = master[
        master["istat_code"]
        == template_code
    ]

    if len(template) != 1:

        raise ValueError(
            f"Template {template_code} "
            "not found exactly once."
        )

    row = (
        template
        .iloc[0]
        .astype(object)
        .copy()
    )

    # --------------------------------------------------
    # ADMIN METADATA
    # --------------------------------------------------

    row["istat_code"] = (
        target_code
    )

    row["name"] = (
        config["name"]
    )

    if "istat_code_old" in row.index:

        if target_code == "024129":

            # No single predecessor code:
            # two municipalities merged.
            row["istat_code_old"] = ""

        else:

            row["istat_code_old"] = (
                target_code
            )

    if (
        "cadastral_code"
        in row.index
        and config[
            "cadastral_code"
        ] is not None
    ):

        row[
            "cadastral_code"
        ] = config[
            "cadastral_code"
        ]

    # --------------------------------------------------
    # POPULATION / DEMOGRAPHICS
    # --------------------------------------------------

    population_features = (
        calculate_population_features(
            population_raw,
            config["sources"],
        )
    )

    for key, value in (
        population_features.items()
    ):

        row[key] = value

    # --------------------------------------------------
    # ECONOMY
    # --------------------------------------------------

    income_features = (
        calculate_income_features(
            income_raw,
            config["sources"],
            population_features[
                "population_2026"
            ],
        )
    )

    for key, value in (
        income_features.items()
    ):

        row[key] = value

    # --------------------------------------------------
    # CURRENT TERRITORY
    # --------------------------------------------------

    territory_features = (
        get_territory_features(
            territory,
            target_code,
        )
    )

    for key, value in (
        territory_features.items()
    ):

        row[key] = value

    # --------------------------------------------------
    # GEOGRAPHY
    # --------------------------------------------------

    geo_features = (
        calculate_merged_geography(
            boundaries,
            config["sources"],
        )
    )

    for key, value in (
        geo_features.items()
    ):

        row[key] = value

    # Recalculate density with merged
    # population and merged area.
    row["density_km2"] = round(
        population_features[
            "population_2026"
        ]
        / geo_features[
            "area_km2"
        ],
        2,
    )

    return row


# --------------------------------------------------
# HARMONISE
# --------------------------------------------------

def harmonise():

    master = load_master()

    population_raw = (
        load_population_raw()
    )

    income_raw = (
        load_income_raw()
    )

    territory = (
        load_current_territory()
    )

    boundaries = (
        load_boundaries()
    )

    new_rows = []

    for target_code, config in (
        MERGERS.items()
    ):

        row = build_target_row(
            master=master,
            target_code=target_code,
            config=config,
            population_raw=population_raw,
            income_raw=income_raw,
            territory=territory,
            boundaries=boundaries,
        )

        new_rows.append(
            row
        )

    # --------------------------------------------------
    # REMOVE OLD MUNICIPALITIES
    # --------------------------------------------------

    harmonised = master[
        ~master["istat_code"]
        .isin(
            SOURCE_CODES_TO_REMOVE
        )
    ].copy()

    # Montalto Pavese already exists and
    # must be replaced by its enlarged version.
    harmonised = harmonised[
        harmonised["istat_code"]
        != "018094"
    ].copy()

    # --------------------------------------------------
    # APPEND CURRENT MUNICIPALITIES
    # --------------------------------------------------

    new_rows_df = pd.DataFrame(
        new_rows
    )

    harmonised = pd.concat(
        [
            harmonised,
            new_rows_df,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------

    print(
        "\n--- FINAL HARMONISATION VALIDATION ---"
    )

    print(
        "Final municipalities:",
        len(harmonised)
    )

    print(
        "Unique ISTAT codes:",
        harmonised[
            "istat_code"
        ].nunique()
    )

    duplicates = (
        harmonised[
            "istat_code"
        ]
        .duplicated()
        .sum()
    )

    print(
        "Duplicated ISTAT codes:",
        duplicates
    )

    old_codes_still_present = (
        harmonised[
            "istat_code"
        ]
        .isin(
            SOURCE_CODES_TO_REMOVE
        )
        .sum()
    )

    print(
        "Suppressed municipalities "
        "still present:",
        old_codes_still_present
    )

    expected_current_codes = [
        "018094",
        "024129",
    ]

    for code in (
        expected_current_codes
    ):

        print(
            f"\n--- {code} ---"
        )

        print(
            harmonised[
                harmonised[
                    "istat_code"
                ]
                == code
            ][
                [
                    "istat_code",
                    "name",
                    "population_2026",
                    "area_km2",
                    "density_km2",
                    "average_income",
                    "taxpayers_per_100_residents",
                    "high_income_share",
                    "altitude_m",
                    "coastal",
                    "degurba",
                    "latitude",
                    "longitude",
                ]
            ].to_string(
                index=False
            )
        )

    if len(harmonised) != 7894:
        raise ValueError(
            "Final dataset should contain "
            "exactly 7,894 municipalities."
        )

    if (
        harmonised[
            "istat_code"
        ].nunique()
        != 7894
    ):

        raise ValueError(
            "ISTAT municipality codes "
            "are not unique."
        )

    if duplicates != 0:

        raise ValueError(
            "Duplicate municipality codes "
            "found."
        )

    if old_codes_still_present != 0:

        raise ValueError(
            "Suppressed municipalities "
            "are still present."
        )

    # --------------------------------------------------
    # SORT + SAVE
    # --------------------------------------------------

    harmonised = (
        harmonised
        .sort_values(
            "istat_code"
        )
        .reset_index(
            drop=True
        )
    )

    harmonised.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print(
        "\nHarmonised dataset saved:"
    )

    print(
        OUTPUT_FILE
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    harmonise()