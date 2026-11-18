# ============================================================
# 02_harmonize_and_combine.py
# ============================================================
#
# PURPOSE:
# Harmonize the three rebuilt source-city datasets:
#
#   Jakarta 2020
#   Klang 2021
#   Tuticorin 2023
#
# OPERATIONS:
# 1. Load validated raw datasets
# 2. Rename POPULATION_1 -> POP_1
# 3. Harmonize ESA WorldCover LULC codes
# 4. Convert SOIL_1 to categorical string
# 5. Add City column
# 6. Preserve X / Y coordinates
# 7. Preserve Flood target
# 8. Save city-level processed datasets
# 9. Combine all source cities
#
# IMPORTANT:
# - X and Y are NOT ML predictors.
# - City is NOT an ML predictor.
# - Flood is the target.
# - No one-hot encoding is performed here.
# - No scaling is performed here.
# - No SMOTE is performed here.
# - Chennai is NOT included here.
#
# ============================================================


from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Adaptive_Multi_City_Flood_Prediction"
)

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "01_raw_training"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "02_processed_training"
)

MODEL_READY_DIR = (
    PROJECT_ROOT
    / "data"
    / "03_model_ready"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)


for folder in [
    PROCESSED_DIR,
    MODEL_READY_DIR,
    RESULTS_DIR
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 2. INPUT DATASETS
# ============================================================

DATASETS = {

    "Jakarta": (
        RAW_DIR
        / "Jakarta_2020_Final_Training_Dataset.csv"
    ),

    "Klang": (
        RAW_DIR
        / "Klang_2021_Final_Training_Dataset.csv"
    ),

    "Tuticorin": (
        RAW_DIR
        / "Tuticorin_2023_Final_Training_Dataset.csv"
    )

}


# ============================================================
# 3. ESA WORLDCOVER -> COMMON LULC GROUP
# ============================================================
#
# ESA WorldCover classes:
#
# 10  = Tree cover
# 20  = Shrubland
# 30  = Grassland
# 40  = Cropland
# 50  = Built-up
# 60  = Bare / sparse vegetation
# 70  = Snow / ice
# 80  = Permanent water bodies
# 90  = Herbaceous wetland
# 95  = Mangroves
# 100 = Moss and lichen
#
# For cross-city modelling we use common semantic groups.
#
# ============================================================

LULC_MAPPING = {

    10: "Trees",

    20: "Shrub_Scrub",

    30: "Grass",

    40: "Agriculture",

    50: "Built_Up",

    60: "Bare",

    70: "Other",

    80: "Water",

    90: "Wetland",

    95: "Wetland",

    100: "Other"

}


# ============================================================
# 4. EXPECTED FINAL COLUMN ORDER
# ============================================================

FINAL_COLUMNS = [

    "X",
    "Y",

    "DEM_1",
    "NDVI_1",
    "POP_1",
    "RAINFALL_1",
    "RIVER DISTANCE_1",
    "SLOPE_1",

    "LULC_GROUP",
    "SOIL_1",

    "City",
    "Flood"

]


# ============================================================
# 5. START
# ============================================================

print("\n")
print("=" * 80)
print("STEP 2 - SOURCE DATA HARMONIZATION")
print("=" * 80)


processed_datasets = []

summary_rows = []


# ============================================================
# 6. PROCESS EACH CITY
# ============================================================

for city, file_path in DATASETS.items():

    print("\n")
    print("=" * 80)
    print(f"PROCESSING: {city.upper()}")
    print("=" * 80)

    print("\nInput file:")
    print(file_path)


    # --------------------------------------------------------
    # FILE CHECK
    # --------------------------------------------------------

    if not file_path.exists():

        raise FileNotFoundError(
            f"Missing input file: {file_path}"
        )


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        file_path
    )


    print("\nOriginal shape:")
    print(df.shape)


    # --------------------------------------------------------
    # KEEP ORIGINAL RAW LULC TEMPORARILY
    # --------------------------------------------------------

    df["LULC_RAW"] = df[
        "LULC_1"
    ].copy()


    # --------------------------------------------------------
    # RENAME POPULATION
    # --------------------------------------------------------

    df = df.rename(
        columns={
            "POPULATION_1": "POP_1"
        }
    )


    # --------------------------------------------------------
    # LULC HARMONIZATION
    # --------------------------------------------------------

    df["LULC_GROUP"] = (
        df["LULC_RAW"]
        .map(
            LULC_MAPPING
        )
    )


    # --------------------------------------------------------
    # CHECK UNKNOWN LULC
    # --------------------------------------------------------

    unknown_lulc_mask = (
        df["LULC_GROUP"]
        .isna()
    )


    unknown_lulc_count = int(
        unknown_lulc_mask.sum()
    )


    if unknown_lulc_count > 0:

        unknown_codes = sorted(
            df.loc[
                unknown_lulc_mask,
                "LULC_RAW"
            ]
            .unique()
            .tolist()
        )

        print("\nWARNING:")
        print(
            "Unknown LULC codes found:"
        )
        print(
            unknown_codes
        )

        # Unrecognized classes are grouped as Other.
        df.loc[
            unknown_lulc_mask,
            "LULC_GROUP"
        ] = "Other"

    else:

        print(
            "\nUnknown LULC codes: None"
        )


    # --------------------------------------------------------
    # SOIL AS CATEGORICAL FEATURE
    # --------------------------------------------------------
    #
    # Important:
    # Soil code 6 is NOT numerically "greater" than soil 3.
    #
    # Therefore convert to string.
    #
    # Example:
    # 1 -> Soil_1
    # 3 -> Soil_3
    #
    # --------------------------------------------------------

    df["SOIL_1"] = (
        pd.to_numeric(
            df["SOIL_1"],
            errors="raise"
        )
        .astype(int)
        .astype(str)
        .map(
            lambda x: f"Soil_{x}"
        )
    )


    # --------------------------------------------------------
    # ADD CITY
    # --------------------------------------------------------

    df["City"] = city


    # --------------------------------------------------------
    # FLOOD LABEL TYPE
    # --------------------------------------------------------

    df["Flood"] = (
        pd.to_numeric(
            df["Flood"],
            errors="raise"
        )
        .astype(int)
    )


    # --------------------------------------------------------
    # COORDINATES AS NUMERIC
    # --------------------------------------------------------

    df["X"] = pd.to_numeric(
        df["X"],
        errors="raise"
    )

    df["Y"] = pd.to_numeric(
        df["Y"],
        errors="raise"
    )


    # --------------------------------------------------------
    # NUMERICAL PREDICTORS
    # --------------------------------------------------------

    numeric_features = [

        "DEM_1",
        "NDVI_1",
        "POP_1",
        "RAINFALL_1",
        "RIVER DISTANCE_1",
        "SLOPE_1"

    ]


    for column in numeric_features:

        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )


    # --------------------------------------------------------
    # FINAL COLUMN ORDER
    # --------------------------------------------------------

    processed = df[
        FINAL_COLUMNS
    ].copy()


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    missing = int(
        processed
        .isna()
        .sum()
        .sum()
    )


    duplicates = int(
        processed
        .duplicated()
        .sum()
    )


    duplicate_xy = int(

        processed
        .duplicated(
            subset=[
                "X",
                "Y"
            ]
        )
        .sum()

    )


    labels = sorted(
        processed[
            "Flood"
        ]
        .unique()
        .tolist()
    )


    if labels != [0, 1]:

        raise ValueError(
            f"{city}: invalid Flood labels {labels}"
        )


    print("\nProcessed shape:")
    print(
        processed.shape
    )


    print("\nMissing values:")
    print(
        missing
    )


    print("\nFull duplicates:")
    print(
        duplicates
    )


    print("\nDuplicate X/Y:")
    print(
        duplicate_xy
    )


    # --------------------------------------------------------
    # LULC GROUP DISTRIBUTION
    # --------------------------------------------------------

    print("\nLULC_GROUP distribution:")

    print(

        processed[
            "LULC_GROUP"
        ]
        .value_counts()
        .sort_index()

    )


    # --------------------------------------------------------
    # SOIL DISTRIBUTION
    # --------------------------------------------------------

    print("\nSOIL_1 distribution:")

    print(

        processed[
            "SOIL_1"
        ]
        .value_counts()
        .sort_index()

    )


    # --------------------------------------------------------
    # FLOOD DISTRIBUTION
    # --------------------------------------------------------

    print("\nFlood distribution:")

    print(

        processed[
            "Flood"
        ]
        .value_counts()
        .sort_index()

    )


    # --------------------------------------------------------
    # SAVE CITY PROCESSED CSV
    # --------------------------------------------------------

    output_path = (

        PROCESSED_DIR
        / f"{city}_Processed.csv"

    )


    processed.to_csv(
        output_path,
        index=False
    )


    print("\nSaved:")
    print(
        output_path
    )


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    processed_datasets.append(
        processed
    )


    summary_rows.append({

        "City":
            city,

        "Rows":
            len(processed),

        "Missing":
            missing,

        "Duplicates":
            duplicates,

        "Duplicate_XY":
            duplicate_xy,

        "Flood_0":
            int(
                (
                    processed["Flood"] == 0
                ).sum()
            ),

        "Flood_1":
            int(
                (
                    processed["Flood"] == 1
                ).sum()
            ),

        "LULC_Groups":
            processed[
                "LULC_GROUP"
            ].nunique(),

        "Soil_Classes":
            processed[
                "SOIL_1"
            ].nunique()

    })


# ============================================================
# 7. COMBINE SOURCE CITIES
# ============================================================

combined = pd.concat(
    processed_datasets,
    ignore_index=True
)


print("\n")
print("=" * 80)
print("COMBINED SOURCE DATA")
print("=" * 80)


print("\nCombined shape:")
print(
    combined.shape
)


print("\nRows by city:")

print(
    combined[
        "City"
    ].value_counts()
)


print("\nCombined Flood counts:")

print(
    combined[
        "Flood"
    ]
    .value_counts()
    .sort_index()
)


print("\nCombined Flood percentages:")

print(

    combined[
        "Flood"
    ]
    .value_counts(
        normalize=True
    )
    .sort_index()
    .mul(100)
    .round(4)

)


# ============================================================
# 8. COMBINED LULC DISTRIBUTION
# ============================================================

print("\n")
print("-" * 80)
print("COMBINED LULC GROUP DISTRIBUTION")
print("-" * 80)


print(

    combined[
        "LULC_GROUP"
    ]
    .value_counts()
    .sort_index()

)


# ============================================================
# 9. COMBINED SOIL DISTRIBUTION
# ============================================================

print("\n")
print("-" * 80)
print("COMBINED SOIL DISTRIBUTION")
print("-" * 80)


print(

    combined[
        "SOIL_1"
    ]
    .value_counts()
    .sort_index()

)


# ============================================================
# 10. CROSS-CITY FEATURE SUMMARY
# ============================================================

NUMERIC_MODEL_FEATURES = [

    "DEM_1",
    "NDVI_1",
    "POP_1",
    "RAINFALL_1",
    "RIVER DISTANCE_1",
    "SLOPE_1"

]


print("\n")
print("=" * 80)
print("NUMERICAL FEATURE MEANS BY CITY")
print("=" * 80)


city_means = (

    combined
    .groupby(
        "City"
    )[NUMERIC_MODEL_FEATURES]
    .mean()

)


print(
    city_means
)


print("\n")
print("=" * 80)
print("NUMERICAL FEATURE MEDIANS BY CITY")
print("=" * 80)


city_medians = (

    combined
    .groupby(
        "City"
    )[NUMERIC_MODEL_FEATURES]
    .median()

)


print(
    city_medians
)


# ============================================================
# 11. CLASS-WISE FEATURE MEANS BY CITY
# ============================================================

print("\n")
print("=" * 80)
print("FEATURE MEANS BY CITY AND FLOOD CLASS")
print("=" * 80)


city_class_means = (

    combined
    .groupby(
        [
            "City",
            "Flood"
        ]
    )[NUMERIC_MODEL_FEATURES]
    .mean()

)


print(
    city_class_means
)


# ============================================================
# 12. FINAL VALIDATION
# ============================================================

total_missing = int(
    combined
    .isna()
    .sum()
    .sum()
)


total_duplicates = int(
    combined
    .duplicated()
    .sum()
)


total_duplicate_xy_city = int(

    combined
    .duplicated(
        subset=[
            "City",
            "X",
            "Y"
        ]
    )
    .sum()

)


print("\n")
print("=" * 80)
print("FINAL COMBINED VALIDATION")
print("=" * 80)


print(
    "\nTotal missing values:",
    total_missing
)


print(
    "Total duplicate rows:",
    total_duplicates
)


print(
    "Duplicate coordinates within cities:",
    total_duplicate_xy_city
)


print(
    "Unique cities:",
    sorted(
        combined[
            "City"
        ]
        .unique()
        .tolist()
    )
)


print(
    "Unique LULC groups:",
    sorted(
        combined[
            "LULC_GROUP"
        ]
        .unique()
        .tolist()
    )
)


print(
    "Unique soil classes:",
    sorted(
        combined[
            "SOIL_1"
        ]
        .unique()
        .tolist()
    )
)


# ============================================================
# 13. SAVE COMBINED MODEL-READY SOURCE DATA
# ============================================================

combined_output = (

    MODEL_READY_DIR
    / "Combined_Model_Ready.csv"

)


combined.to_csv(
    combined_output,
    index=False
)


print("\nCombined dataset saved to:")

print(
    combined_output
)


# ============================================================
# 14. SAVE PROCESSING SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)


summary_output = (

    RESULTS_DIR
    / "02_harmonization_summary.csv"

)


summary_df.to_csv(
    summary_output,
    index=False
)


print("\nProcessing summary saved to:")

print(
    summary_output
)


# ============================================================
# 15. FINAL STATUS
# ============================================================

expected_total_rows = (
    5000
    + 4775
    + 4960
)


if (

    len(combined)
    == expected_total_rows

    and total_missing == 0

    and total_duplicates == 0

    and total_duplicate_xy_city == 0

):

    print("\n")
    print("=" * 80)
    print("STEP 2 STATUS: PASS")
    print("=" * 80)

    print(
        f"Combined source dataset contains "
        f"{len(combined)} valid samples."
    )

    print(
        "Source-city harmonization completed successfully."
    )

    print(
        "No rows were deleted."
    )

    print(
        "No SMOTE or artificial balancing was performed."
    )

    print(
        "No one-hot encoding was performed yet."
    )

    print(
        "No scaling was performed yet."
    )

    print(
        "Chennai was NOT included."
    )

    print(
        "Next step: build and evaluate generalized "
        "source-city ML models."
    )

else:

    print("\n")
    print("=" * 80)
    print("STEP 2 STATUS: CHECK REQUIRED")
    print("=" * 80)

    print(
        "Do not continue to model training "
        "until this issue is checked."
    )


print("\n")