from pathlib import Path

import geopandas as gpd
import pandas as pd


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

BOUNDARIES_DIR = (
    ROOT
    / "data"
    / "raw"
    / "boundaries"
)


# --------------------------------------------------
# ADMINISTRATIVE MERGERS
# --------------------------------------------------

MERGED_GEOMETRIES = {

    # Montalto Pavese + Lirio
    "018094": [
        "018094",
        "018082",
    ],

    # Castegnero + Nanto
    "024129": [
        "024027",
        "024071",
    ],
}


# --------------------------------------------------
# FIND SHAPEFILE
# --------------------------------------------------

def find_shapefile():

    files = list(
        BOUNDARIES_DIR.rglob(
            "Com01012026_g_WGS84.shp"
        )
    )

    if not files:

        files = list(
            BOUNDARIES_DIR.rglob(
                "Com*.shp"
            )
        )

    if not files:

        raise FileNotFoundError(
            "Municipality shapefile not found."
        )

    return files[0]


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    df = pd.read_csv(
        DATA_FILE,
        dtype={
            "istat_code": str,
        },
        keep_default_na=False,
    )

    df["istat_code"] = (
        df["istat_code"]
        .astype(str)
        .str.zfill(6)
    )

    print(
        f"Municipalities: {len(df):,}"
    )

    return df


# --------------------------------------------------
# LOAD BOUNDARIES
# --------------------------------------------------

def load_boundaries():

    shapefile = find_shapefile()

    print(
        f"Boundary file: {shapefile.name}"
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

    else:

        gdf["istat_code"] = (
            gdf["PRO_COM"]
            .astype(str)
            .str.zfill(6)
        )

    return gdf


# --------------------------------------------------
# CALCULATE CENTROIDS
# --------------------------------------------------

def calculate_centroids(
    df,
    boundaries,
):

    print(
        "\nCalculating map centroids..."
    )

    # Correct metric projection for centroid calculation.
    metric = boundaries.to_crs(
        epsg=3035
    )

    coordinates = {}

    # --------------------------------------------------
    # NORMAL MUNICIPALITIES
    # --------------------------------------------------

    final_codes = set(
        df["istat_code"]
    )

    merged_targets = set(
        MERGED_GEOMETRIES.keys()
    )

    normal_codes = (
        final_codes
        - merged_targets
    )

    normal = metric[
        metric["istat_code"]
        .isin(normal_codes)
    ].copy()

    normal[
        "centroid_geometry"
    ] = normal.geometry.centroid

    centroids = gpd.GeoDataFrame(
        normal[
            [
                "istat_code",
            ]
        ].copy(),
        geometry=normal[
            "centroid_geometry"
        ],
        crs="EPSG:3035",
    ).to_crs(
        epsg=4326
    )

    for _, row in centroids.iterrows():

        coordinates[
            row["istat_code"]
        ] = (
            row.geometry.y,
            row.geometry.x,
        )

    # --------------------------------------------------
    # MERGED MUNICIPALITIES
    # --------------------------------------------------

    for target_code, source_codes in (
        MERGED_GEOMETRIES.items()
    ):

        subset = metric[
            metric["istat_code"]
            .isin(source_codes)
        ]

        if len(subset) != len(source_codes):

            raise ValueError(
                f"Missing source geometry for {target_code}"
            )

        merged_geometry = (
            subset.geometry
            .union_all()
        )

        centroid = (
            merged_geometry.centroid
        )

        centroid_gdf = (
            gpd.GeoDataFrame(
                geometry=[
                    centroid
                ],
                crs="EPSG:3035",
            )
            .to_crs(
                epsg=4326
            )
        )

        point = (
            centroid_gdf
            .geometry
            .iloc[0]
        )

        coordinates[
            target_code
        ] = (
            point.y,
            point.x,
        )

    # --------------------------------------------------
    # UPDATE DATASET
    # --------------------------------------------------

    df["latitude"] = (
        df["istat_code"]
        .map(
            lambda code:
                coordinates[code][0]
        )
        .round(6)
    )

    df["longitude"] = (
        df["istat_code"]
        .map(
            lambda code:
                coordinates[code][1]
        )
        .round(6)
    )

    print(
        "Missing centroids:",
        df[
            [
                "latitude",
                "longitude",
            ]
        ].isna().any(axis=1).sum()
    )

    return df


# --------------------------------------------------
# SAVE
# --------------------------------------------------

def save_data(df):

    df.to_csv(
        DATA_FILE,
        index=False,
        encoding="utf-8",
    )

    print(
        "\nUpdated:"
    )

    print(
        DATA_FILE
    )

    for name in [
        "Giarre",
        "Roma",
        "Milano",
        "Castegnero Nanto",
    ]:

        print(
            "\n",
            df[
                df["name"]
                == name
            ][
                [
                    "name",
                    "latitude",
                    "longitude",
                ]
            ].to_string(
                index=False
            )
        )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    df = load_data()

    boundaries = (
        load_boundaries()
    )

    df = calculate_centroids(
        df,
        boundaries,
    )

    save_data(
        df
    )