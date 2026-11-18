from pathlib import Path
import geopandas as gpd
import pandas as pd

ROOT = Path(
    r"D:\Adaptive_Multi_City_Flood_Prediction"
)

INPUT = (
    ROOT
    / "phase2_drainage"
    / "Chennai_NSAGII_Candidate_Drains.gpkg"
)

print("\n")
print("=" * 80)
print("PHASE 2 - STEP 37A")
print("DRAINAGE ATTRIBUTE AUDIT FOR NSGA-II")
print("=" * 80)

if not INPUT.exists():
    raise FileNotFoundError(
        f"Missing candidate file:\n{INPUT}"
    )

gdf = gpd.read_file(
    INPUT
)

print("\nShape:")
print(gdf.shape)

print("\nCRS:")
print(gdf.crs)

print("\n")
print("=" * 80)
print("ALL ATTRIBUTE COLUMNS")
print("=" * 80)

for i, col in enumerate(
    gdf.columns,
    start=1
):
    print(
        f"{i:02d}. {col}"
    )


print("\n")
print("=" * 80)
print("COLUMN DATA TYPES")
print("=" * 80)

print(
    gdf.dtypes
)


print("\n")
print("=" * 80)
print("FIRST 5 RECORDS - NON GEOMETRY ATTRIBUTES")
print("=" * 80)

non_geometry = [
    c
    for c in gdf.columns
    if c != "geometry"
]

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    250
)

print(
    gdf[
        non_geometry
    ]
    .head()
    .to_string(
        index=False
    )
)


print("\n")
print("=" * 80)
print("POSSIBLE ENGINEERING / COST ATTRIBUTES")
print("=" * 80)

keywords = [

    "width",
    "diameter",
    "dia",
    "depth",
    "height",
    "size",
    "capacity",
    "discharge",
    "flow",
    "type",
    "material",
    "cost",
    "rate",
    "ward",
    "zone",
    "length"

]

possible = []

for col in non_geometry:

    lower = col.lower()

    if any(
        keyword in lower
        for keyword in keywords
    ):

        possible.append(
            col
        )


if possible:

    print(
        possible
    )

    print("\nValues / statistics:")

    for col in possible:

        print("\n" + "-" * 80)
        print(col)
        print("-" * 80)

        if pd.api.types.is_numeric_dtype(
            gdf[col]
        ):

            print(
                gdf[col]
                .describe()
            )

        else:

            print(
                gdf[col]
                .value_counts(
                    dropna=False
                )
                .head(20)
            )

else:

    print(
        "No obvious engineering/cost fields detected."
    )


print("\n")
print("=" * 80)
print("STEP 37A STATUS: PASS")
print("=" * 80)