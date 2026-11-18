# ============================================================
# 01_validate_new_datasets.py
# ============================================================
#
# PURPOSE:
# Validate the three newly rebuilt source-city datasets:
#
# 1. Jakarta 2020
# 2. Klang 2021
# 3. Tuticorin / Thoothukudi 2023
#
# This script DOES NOT modify the data.
#
# Checks:
# - File existence
# - Dataset shape
# - Column names
# - Missing values
# - Full-row duplicates
# - Duplicate coordinates
# - Flood class counts
# - Flood class percentages
# - Invalid Flood labels
# - Coordinate ranges
# - Numerical feature ranges
# - LULC categories
# - Soil categories
# - Class-wise feature means
#
# OUTPUT:
# results/01_new_dataset_validation_summary.csv
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

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "01_raw_training"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. DATASET FILES
# ============================================================

DATASETS = {

    "Jakarta": (
        RAW_DATA_DIR
        / "Jakarta_2020_Final_Training_Dataset.csv"
    ),

    "Klang": (
        RAW_DATA_DIR
        / "Klang_2021_Final_Training_Dataset.csv"
    ),

    "Tuticorin": (
        RAW_DATA_DIR
        / "Tuticorin_2023_Final_Training_Dataset.csv"
    )
}


# ============================================================
# 3. EXPECTED RAW COLUMNS
# ============================================================

EXPECTED_COLUMNS = [

    "X",
    "Y",
    "DEM_1",
    "LULC_1",
    "NDVI_1",
    "POPULATION_1",
    "RAINFALL_1",
    "RIVER DISTANCE_1",
    "SLOPE_1",
    "SOIL_1",
    "Flood"

]


# ============================================================
# 4. NUMERICAL FEATURES
# ============================================================

NUMERIC_FEATURES = [

    "DEM_1",
    "NDVI_1",
    "POPULATION_1",
    "RAINFALL_1",
    "RIVER DISTANCE_1",
    "SLOPE_1"

]


# ============================================================
# 5. PRINT HEADER
# ============================================================

print("\n")
print("=" * 80)
print("NEW SOURCE-CITY DATASET VALIDATION")
print("=" * 80)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nRaw training folder:")
print(RAW_DATA_DIR)


# ============================================================
# 6. VALIDATION STORAGE
# ============================================================

summary_rows = []

loaded_datasets = {}


# ============================================================
# 7. PROCESS EACH DATASET
# ============================================================

for city, file_path in DATASETS.items():

    print("\n")
    print("=" * 80)
    print(f"VALIDATING: {city.upper()}")
    print("=" * 80)

    print("\nFile:")
    print(file_path)


    # --------------------------------------------------------
    # FILE EXISTENCE
    # --------------------------------------------------------

    if not file_path.exists():

        print("\nERROR:")
        print("File does not exist.")

        summary_rows.append({

            "City": city,
            "File_Found": False

        })

        continue


    print("\nFile found: YES")


    # --------------------------------------------------------
    # LOAD CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        file_path
    )

    loaded_datasets[city] = df.copy()


    # --------------------------------------------------------
    # BASIC SHAPE
    # --------------------------------------------------------

    rows, columns = df.shape

    print("\nShape:")
    print(df.shape)

    print("\nRows:")
    print(rows)

    print("\nColumns:")
    print(columns)


    # --------------------------------------------------------
    # COLUMN CHECK
    # --------------------------------------------------------

    print("\nColumn names:")

    for col in df.columns:

        print(f"  - {col}")


    missing_columns = [
        col
        for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]


    extra_columns = [
        col
        for col in df.columns
        if col not in EXPECTED_COLUMNS
    ]


    print("\nMissing expected columns:")
    print(
        missing_columns
        if missing_columns
        else "None"
    )


    print("\nUnexpected extra columns:")
    print(
        extra_columns
        if extra_columns
        else "None"
    )


    columns_correct = (
        set(df.columns)
        ==
        set(EXPECTED_COLUMNS)
    )


    print("\nExpected column structure:")
    print(
        "PASS"
        if columns_correct
        else "CHECK REQUIRED"
    )


    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    missing_by_column = (
        df.isna()
        .sum()
    )

    total_missing = int(
        missing_by_column.sum()
    )


    print("\nMissing values by column:")

    print(
        missing_by_column
    )


    print("\nTotal missing values:")
    print(total_missing)


    # --------------------------------------------------------
    # INFINITE VALUES
    # --------------------------------------------------------

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    total_infinite = int(
        np.isinf(
            numeric_df
        )
        .sum()
        .sum()
    )


    print("\nInfinite numeric values:")
    print(total_infinite)


    # --------------------------------------------------------
    # FULL DUPLICATE ROWS
    # --------------------------------------------------------

    full_duplicates = int(
        df.duplicated().sum()
    )


    print("\nFull duplicate rows:")
    print(full_duplicates)


    # --------------------------------------------------------
    # DUPLICATE COORDINATES
    # --------------------------------------------------------

    if (
        "X" in df.columns
        and
        "Y" in df.columns
    ):

        duplicate_coordinates = int(

            df.duplicated(
                subset=[
                    "X",
                    "Y"
                ]
            ).sum()

        )

    else:

        duplicate_coordinates = -1


    print("\nDuplicate X/Y coordinates:")
    print(duplicate_coordinates)


    # --------------------------------------------------------
    # FLOOD LABEL CHECK
    # --------------------------------------------------------

    if "Flood" in df.columns:

        unique_labels = sorted(
            df["Flood"]
            .dropna()
            .unique()
            .tolist()
        )


        print("\nUnique Flood labels:")
        print(unique_labels)


        valid_labels = set(
            unique_labels
        ).issubset(
            {0, 1}
        )


        print("\nFlood labels valid:")
        print(
            "YES"
            if valid_labels
            else "NO"
        )


        class_counts = (
            df["Flood"]
            .value_counts()
            .sort_index()
        )


        class_percentages = (
            df["Flood"]
            .value_counts(
                normalize=True
            )
            .sort_index()
            .mul(100)
        )


        flood_0 = int(
            class_counts.get(
                0,
                0
            )
        )


        flood_1 = int(
            class_counts.get(
                1,
                0
            )
        )


        flood_0_pct = float(
            class_percentages.get(
                0,
                0
            )
        )


        flood_1_pct = float(
            class_percentages.get(
                1,
                0
            )
        )


        print("\nFlood class counts:")
        print(class_counts)


        print("\nFlood class percentages:")
        print(
            class_percentages.round(2)
        )


    else:

        valid_labels = False

        flood_0 = 0
        flood_1 = 0

        flood_0_pct = 0
        flood_1_pct = 0


    # --------------------------------------------------------
    # COORDINATE RANGE
    # --------------------------------------------------------

    if (
        "X" in df.columns
        and
        "Y" in df.columns
    ):

        print("\nCoordinate ranges:")

        print(
            f"X: {df['X'].min()} "
            f"to {df['X'].max()}"
        )

        print(
            f"Y: {df['Y'].min()} "
            f"to {df['Y'].max()}"
        )


    # --------------------------------------------------------
    # FEATURE RANGE CHECK
    # --------------------------------------------------------

    print("\n")
    print("-" * 80)
    print("NUMERICAL FEATURE RANGES")
    print("-" * 80)


    for feature in NUMERIC_FEATURES:

        if feature not in df.columns:

            print(
                f"\n{feature}: MISSING"
            )

            continue


        values = pd.to_numeric(
            df[feature],
            errors="coerce"
        )


        print(
            f"\n{feature}"
        )

        print(
            f"  Min    : {values.min()}"
        )

        print(
            f"  Max    : {values.max()}"
        )

        print(
            f"  Mean   : {values.mean()}"
        )

        print(
            f"  Median : {values.median()}"
        )

        print(
            f"  Std    : {values.std()}"
        )


    # --------------------------------------------------------
    # LULC DISTRIBUTION
    # --------------------------------------------------------

    if "LULC_1" in df.columns:

        print("\n")
        print("-" * 80)
        print("LULC DISTRIBUTION")
        print("-" * 80)

        lulc_counts = (
            df["LULC_1"]
            .value_counts()
            .sort_index()
        )

        print(lulc_counts)


    # --------------------------------------------------------
    # SOIL DISTRIBUTION
    # --------------------------------------------------------

    if "SOIL_1" in df.columns:

        print("\n")
        print("-" * 80)
        print("SOIL DISTRIBUTION")
        print("-" * 80)

        soil_counts = (
            df["SOIL_1"]
            .value_counts()
            .sort_index()
        )

        print(soil_counts)


    # --------------------------------------------------------
    # CLASS-WISE FEATURE MEANS
    # --------------------------------------------------------

    if (
        "Flood" in df.columns
        and
        valid_labels
    ):

        available_features = [

            feature

            for feature
            in NUMERIC_FEATURES

            if feature in df.columns

        ]


        print("\n")
        print("-" * 80)
        print("MEAN FEATURES BY FLOOD CLASS")
        print("-" * 80)


        class_means = (

            df.groupby(
                "Flood"
            )[available_features]
            .mean()
            .T

        )


        print(
            class_means
        )


    # --------------------------------------------------------
    # BASIC VALIDATION RESULT
    # --------------------------------------------------------

    validation_pass = (

        columns_correct
        and total_missing == 0
        and total_infinite == 0
        and full_duplicates == 0
        and duplicate_coordinates == 0
        and valid_labels

    )


    print("\n")
    print("-" * 80)

    print(
        f"{city} BASIC VALIDATION:"
    )

    print(
        "PASS"
        if validation_pass
        else "CHECK REQUIRED"
    )

    print("-" * 80)


    # --------------------------------------------------------
    # STORE SUMMARY
    # --------------------------------------------------------

    summary_rows.append({

        "City":
            city,

        "File_Found":
            True,

        "Rows":
            rows,

        "Columns":
            columns,

        "Expected_Columns":
            columns_correct,

        "Missing_Values":
            total_missing,

        "Infinite_Values":
            total_infinite,

        "Full_Duplicates":
            full_duplicates,

        "Duplicate_XY":
            duplicate_coordinates,

        "Flood_0":
            flood_0,

        "Flood_1":
            flood_1,

        "Flood_0_Percent":
            round(
                flood_0_pct,
                4
            ),

        "Flood_1_Percent":
            round(
                flood_1_pct,
                4
            ),

        "Valid_Flood_Labels":
            valid_labels,

        "Validation_Pass":
            validation_pass

    })


# ============================================================
# 8. COMBINED SOURCE-DATA SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("COMBINED SOURCE-DATA SUMMARY")
print("=" * 80)


if len(loaded_datasets) == 3:

    combined_raw = pd.concat(

        [

            df.assign(
                City=city
            )

            for city, df
            in loaded_datasets.items()

        ],

        ignore_index=True

    )


    print("\nTotal source rows:")
    print(
        len(combined_raw)
    )


    print("\nRows by city:")

    print(
        combined_raw[
            "City"
        ].value_counts()
    )


    print("\nCombined Flood counts:")

    print(
        combined_raw[
            "Flood"
        ]
        .value_counts()
        .sort_index()
    )


    print("\nCombined Flood percentages:")

    print(

        combined_raw[
            "Flood"
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
        .mul(100)
        .round(2)

    )


    print("\nTotal missing values:")
    print(
        combined_raw
        .isna()
        .sum()
        .sum()
    )


    print("\nTotal full duplicates:")
    print(
        combined_raw
        .duplicated()
        .sum()
    )


else:

    print(
        "\nNot all three datasets were loaded."
    )

    print(
        "Fix missing files before continuing."
    )


# ============================================================
# 9. SAVE VALIDATION SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)


summary_output = (

    RESULTS_DIR
    / "01_new_dataset_validation_summary.csv"

)


summary_df.to_csv(
    summary_output,
    index=False
)


print("\n")
print("=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

print(
    summary_df.to_string(
        index=False
    )
)


print("\nValidation summary saved to:")

print(
    summary_output
)


# ============================================================
# 10. FINAL STATUS
# ============================================================

if (
    len(summary_df) == 3
    and
    "Validation_Pass"
    in summary_df.columns
    and
    summary_df[
        "Validation_Pass"
    ].all()
):

    print("\n")
    print("=" * 80)
    print("STEP 1 STATUS: PASS")
    print("=" * 80)

    print(
        "All three rebuilt source-city datasets passed "
        "the basic validation checks."
    )

    print(
        "Next step: harmonize and preprocess the source datasets."
    )

else:

    print("\n")
    print("=" * 80)
    print("STEP 1 STATUS: CHECK REQUIRED")
    print("=" * 80)

    print(
        "At least one dataset requires checking "
        "before preprocessing."
    )


print("\n")