# ============================================================
# PHASE 2 - STEP 37B
# PREPARE DRAINAGE ENGINEERING ATTRIBUTES FOR NSGA-II
# ============================================================
#
# PURPOSE
# -------
# Clean and convert GCC drainage engineering attributes into
# numeric variables that can be used in the optimization phase.
#
# INPUT:
# Chennai_NSAGII_Candidate_Drains.gpkg
#
# OUTPUT:
# Chennai_NSGAII_Engineering_Ready.gpkg
# Chennai_NSGAII_Engineering_Ready.csv
# 37b_engineering_attribute_summary.csv
#
# IMPORTANT
# ---------
# - No NSGA-II optimization yet.
# - No cost assumptions yet.
# - No artificial capacity values are invented.
# - Missing width/depth values are imputed only for creating
#   a complete model-ready optimization table.
# - Imputed rows are explicitly flagged.
#
# ============================================================


from pathlib import Path

import re
import numpy as np
import pandas as pd
import geopandas as gpd


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(
    r"D:\Adaptive_Multi_City_Flood_Prediction"
)

INPUT_GPKG = (
    ROOT
    / "phase2_drainage"
    / "Chennai_NSAGII_Candidate_Drains.gpkg"
)

OUTPUT_GPKG = (
    ROOT
    / "phase2_drainage"
    / "Chennai_NSGAII_Engineering_Ready.gpkg"
)

OUTPUT_CSV = (
    ROOT
    / "phase2_drainage"
    / "Chennai_NSGAII_Engineering_Ready.csv"
)

SUMMARY_CSV = (
    ROOT
    / "results"
    / "37b_engineering_attribute_summary.csv"
)


# ============================================================
# 2. START
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 2 - STEP 37B")
print("ENGINEERING ATTRIBUTE PREPARATION")
print("=" * 80)


if not INPUT_GPKG.exists():

    raise FileNotFoundError(
        f"Missing candidate drainage file:\n{INPUT_GPKG}"
    )


# ============================================================
# 3. LOAD DATA
# ============================================================

gdf = gpd.read_file(
    INPUT_GPKG
)


print("\nInput candidate segments:")
print(
    len(gdf)
)


print("\nCRS:")
print(
    gdf.crs
)


if len(gdf) != 500:

    print(
        "\nWARNING:"
    )

    print(
        f"Expected 500 candidates but found {len(gdf)}."
    )


# ============================================================
# 4. REQUIRED ATTRIBUTES
# ============================================================

required = [

    "Candidate_ID",
    "Drain_ID",

    "Segment_Length_m",

    "Mean_Flood_Prob",
    "Max_Flood_Prob",

    "Flood_Exposure_Pct",
    "Priority_Score",

    "Risk_Exposure_Index",
    "Candidate_Priority",

    "drain_type",

    "drain_wid",
    "drain_dep",
    "drain_size",

    "typ_mat",
    "swd_mat",

    "status",

    "invert_sp",
    "invert_ep",

    "obstacles",

    "zone",
    "ward",
    "location"

]


missing = [

    col
    for col in required
    if col not in gdf.columns

]


if missing:

    raise ValueError(
        f"Missing required columns:\n{missing}"
    )


# ============================================================
# 5. TEXT CLEANING
# ============================================================

NULL_STRINGS = {

    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "-",
    "--",
    "nil"

}


def clean_text(value):

    if pd.isna(value):
        return None

    value = str(
        value
    ).strip()

    if value.lower() in NULL_STRINGS:
        return None

    return value


# ============================================================
# 6. NUMERIC PARSER
# ============================================================
#
# Extract the first valid number from fields such as:
#
# "1.32"
# "1.620000000000000"
# "1.2 m"
# "0.75"
#
# ============================================================

def extract_first_number(value):

    value = clean_text(
        value
    )

    if value is None:
        return np.nan

    match = re.search(
        r"[-+]?\d*\.?\d+",
        value
    )

    if match is None:
        return np.nan

    try:

        return float(
            match.group()
        )

    except ValueError:

        return np.nan


# ============================================================
# 7. DRAIN-SIZE PARSER
# ============================================================
#
# Examples:
#
# "0.75 x 0.75"
# "1.25 x 1.25"
# "1.20 x 1.20"
#
# Returns:
#
# width, depth
#
# ============================================================

def parse_drain_size(value):

    value = clean_text(
        value
    )

    if value is None:

        return (
            np.nan,
            np.nan
        )


    numbers = re.findall(
        r"\d*\.?\d+",
        value
    )


    if len(numbers) >= 2:

        try:

            width = float(
                numbers[0]
            )

            depth = float(
                numbers[1]
            )

            return (
                width,
                depth
            )

        except ValueError:

            pass


    return (
        np.nan,
        np.nan
    )


# ============================================================
# 8. PARSE DIRECT WIDTH / DEPTH
# ============================================================

gdf[
    "Width_Direct_m"
] = gdf[
    "drain_wid"
].apply(
    extract_first_number
)


gdf[
    "Depth_Direct_m"
] = gdf[
    "drain_dep"
].apply(
    extract_first_number
)


# ============================================================
# 9. PARSE DRAIN_SIZE FALLBACK
# ============================================================

size_values = gdf[
    "drain_size"
].apply(
    parse_drain_size
)


gdf[
    "Width_Size_m"
] = [

    value[0]
    for value in size_values

]


gdf[
    "Depth_Size_m"
] = [

    value[1]
    for value in size_values

]


# ============================================================
# 10. COMBINE DIRECT + FALLBACK DIMENSIONS
# ============================================================

gdf[
    "Width_Parsed_m"
] = gdf[
    "Width_Direct_m"
].fillna(
    gdf[
        "Width_Size_m"
    ]
)


gdf[
    "Depth_Parsed_m"
] = gdf[
    "Depth_Direct_m"
].fillna(
    gdf[
        "Depth_Size_m"
    ]
)


# ============================================================
# 11. FLAG DIMENSION SOURCE
# ============================================================

def dimension_source(row):

    width_direct = pd.notna(
        row[
            "Width_Direct_m"
        ]
    )

    depth_direct = pd.notna(
        row[
            "Depth_Direct_m"
        ]
    )

    width_fallback = pd.notna(
        row[
            "Width_Size_m"
        ]
    )

    depth_fallback = pd.notna(
        row[
            "Depth_Size_m"
        ]
    )


    if (
        width_direct
        and
        depth_direct
    ):

        return "DIRECT"


    if (
        (
            width_direct
            or
            width_fallback
        )
        and
        (
            depth_direct
            or
            depth_fallback
        )
    ):

        return "MIXED_OR_SIZE_FALLBACK"


    return "MISSING"


gdf[
    "Dimension_Source"
] = gdf.apply(
    dimension_source,
    axis=1
)


# ============================================================
# 12. BASIC PLAUSIBILITY FILTER
# ============================================================
#
# We do NOT silently accept clearly unrealistic values.
#
# Plausible broad engineering range used only for cleaning:
#
# 0.10 m <= width <= 10 m
# 0.10 m <= depth <= 10 m
#
# Out-of-range values become NaN and are flagged.
#
# ============================================================

width_bad = (

    (
        gdf[
            "Width_Parsed_m"
        ]
        <
        0.10
    )

    |

    (
        gdf[
            "Width_Parsed_m"
        ]
        >
        10.0
    )

)


depth_bad = (

    (
        gdf[
            "Depth_Parsed_m"
        ]
        <
        0.10
    )

    |

    (
        gdf[
            "Depth_Parsed_m"
        ]
        >
        10.0
    )

)


gdf[
    "Width_Outlier_Flag"
] = width_bad.astype(
    int
)


gdf[
    "Depth_Outlier_Flag"
] = depth_bad.astype(
    int
)


gdf.loc[
    width_bad,
    "Width_Parsed_m"
] = np.nan


gdf.loc[
    depth_bad,
    "Depth_Parsed_m"
] = np.nan


# ============================================================
# 13. PARSE INVERT LEVELS
# ============================================================

gdf[
    "Invert_Start_m"
] = gdf[
    "invert_sp"
].apply(
    extract_first_number
)


gdf[
    "Invert_End_m"
] = gdf[
    "invert_ep"
].apply(
    extract_first_number
)


# ============================================================
# 14. CALCULATE INVERT DIFFERENCE / SLOPE PROXY
# ============================================================
#
# Drain digitization direction may not necessarily equal
# hydraulic flow direction.
#
# Therefore:
#
# Signed slope:
#     (start invert - end invert) / length
#
# Absolute slope:
#     abs(start invert - end invert) / length
#
# We retain BOTH.
#
# ============================================================

valid_length = (

    gdf[
        "Segment_Length_m"
    ]
    >
    0
)


gdf[
    "Invert_Difference_m"
] = (

    gdf[
        "Invert_Start_m"
    ]

    -

    gdf[
        "Invert_End_m"
    ]

)


gdf[
    "Invert_Slope_Signed"
] = np.where(

    valid_length,

    gdf[
        "Invert_Difference_m"
    ]
    /
    gdf[
        "Segment_Length_m"
    ],

    np.nan

)


gdf[
    "Invert_Slope_Abs"
] = np.abs(

    gdf[
        "Invert_Slope_Signed"
    ]

)


# ============================================================
# 15. IMPUTE MISSING WIDTH / DEPTH
# ============================================================
#
# For the optimization model we need complete dimensions.
#
# Strategy:
#
# 1. Median within normalized drain_type
# 2. Overall candidate median
#
# Original / parsed values remain available.
#
# ============================================================

gdf[
    "Drain_Type_Clean"
] = (

    gdf[
        "drain_type"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
)


overall_width_median = gdf[
    "Width_Parsed_m"
].median()


overall_depth_median = gdf[
    "Depth_Parsed_m"
].median()


print("\n")
print("=" * 80)
print("DIMENSION CLEANING")
print("=" * 80)


print(
    "\nOverall parsed width median:",
    overall_width_median
)


print(
    "Overall parsed depth median:",
    overall_depth_median
)


gdf[
    "Width_Imputed_Flag"
] = gdf[
    "Width_Parsed_m"
].isna().astype(
    int
)


gdf[
    "Depth_Imputed_Flag"
] = gdf[
    "Depth_Parsed_m"
].isna().astype(
    int
)


type_width_median = (

    gdf.groupby(
        "Drain_Type_Clean"
    )[
        "Width_Parsed_m"
    ]
    .median()

)


type_depth_median = (

    gdf.groupby(
        "Drain_Type_Clean"
    )[
        "Depth_Parsed_m"
    ]
    .median()

)


def impute_width(row):

    value = row[
        "Width_Parsed_m"
    ]

    if pd.notna(
        value
    ):

        return value


    type_name = row[
        "Drain_Type_Clean"
    ]


    if type_name in type_width_median.index:

        median = type_width_median.loc[
            type_name
        ]

        if pd.notna(
            median
        ):

            return median


    return overall_width_median


def impute_depth(row):

    value = row[
        "Depth_Parsed_m"
    ]

    if pd.notna(
        value
    ):

        return value


    type_name = row[
        "Drain_Type_Clean"
    ]


    if type_name in type_depth_median.index:

        median = type_depth_median.loc[
            type_name
        ]

        if pd.notna(
            median
        ):

            return median


    return overall_depth_median


gdf[
    "Existing_Width_m"
] = gdf.apply(
    impute_width,
    axis=1
)


gdf[
    "Existing_Depth_m"
] = gdf.apply(
    impute_depth,
    axis=1
)


# ============================================================
# 16. CROSS-SECTIONAL AREA PROXY
# ============================================================
#
# We currently treat width × depth as a geometric section-area
# proxy.
#
# It is NOT yet claimed as actual hydraulic capacity.
#
# ============================================================

gdf[
    "Existing_Area_m2"
] = (

    gdf[
        "Existing_Width_m"
    ]

    *

    gdf[
        "Existing_Depth_m"
    ]

)


# ============================================================
# 17. GEOMETRIC VOLUME PROXY
# ============================================================
#
# Cross-sectional area × segment length.
#
# Useful for later upgrade-volume / cost calculations.
#
# ============================================================

gdf[
    "Existing_Volume_m3"
] = (

    gdf[
        "Existing_Area_m2"
    ]

    *

    gdf[
        "Segment_Length_m"
    ]

)


# ============================================================
# 18. MATERIAL / STATUS CLEANING
# ============================================================

def standardized_text(series):

    return (

        series

        .fillna(
            "Unknown"
        )

        .astype(str)

        .str.strip()

        .replace(
            {
                "":
                    "Unknown",

                "NA":
                    "Unknown",

                "N/A":
                    "Unknown",

                "nan":
                    "Unknown"
            }
        )

    )


gdf[
    "Material_Clean"
] = standardized_text(
    gdf[
        "swd_mat"
    ]
)


gdf[
    "Status_Clean"
] = standardized_text(
    gdf[
        "status"
    ]
)


gdf[
    "Obstacle_Clean"
] = standardized_text(
    gdf[
        "obstacles"
    ]
)


# ============================================================
# 19. ENGINEERING DATA QUALITY FLAG
# ============================================================

gdf[
    "Engineering_Imputed"
] = (

    (
        gdf[
            "Width_Imputed_Flag"
        ]
        ==
        1
    )

    |

    (
        gdf[
            "Depth_Imputed_Flag"
        ]
        ==
        1
    )

).astype(
    int
)


def quality_label(row):

    if row[
        "Engineering_Imputed"
    ] == 0:

        if (
            pd.notna(
                row[
                    "Invert_Start_m"
                ]
            )

            and

            pd.notna(
                row[
                    "Invert_End_m"
                ]
            )
        ):

            return "HIGH"

        return "MEDIUM"


    return "IMPUTED"


gdf[
    "Engineering_Data_Quality"
] = gdf.apply(
    quality_label,
    axis=1
)


# ============================================================
# 20. VALIDATE MODEL-READY ENGINEERING VARIABLES
# ============================================================

engineering_variables = [

    "Existing_Width_m",

    "Existing_Depth_m",

    "Existing_Area_m2",

    "Existing_Volume_m3"

]


print("\nMissing model-ready engineering values:")

print(
    gdf[
        engineering_variables
    ]
    .isna()
    .sum()
)


if gdf[
    engineering_variables
].isna().any().any():

    raise ValueError(
        "Missing values remain in model-ready "
        "engineering dimensions."
    )


# ============================================================
# 21. DIMENSION STATISTICS
# ============================================================

print("\n")
print("=" * 80)
print("FINAL ENGINEERING DIMENSION STATISTICS")
print("=" * 80)


print("\nExisting width (m):")

print(
    gdf[
        "Existing_Width_m"
    ]
    .describe()
)


print("\nExisting depth (m):")

print(
    gdf[
        "Existing_Depth_m"
    ]
    .describe()
)


print("\nExisting cross-sectional area proxy (m²):")

print(
    gdf[
        "Existing_Area_m2"
    ]
    .describe()
)


print("\nExisting geometric volume proxy (m³):")

print(
    gdf[
        "Existing_Volume_m3"
    ]
    .describe()
)


# ============================================================
# 22. INVERT DATA STATISTICS
# ============================================================

print("\n")
print("=" * 80)
print("INVERT / SLOPE DATA")
print("=" * 80)


print(
    "\nValid start invert values:",
    int(
        gdf[
            "Invert_Start_m"
        ]
        .notna()
        .sum()
    )
)


print(
    "Valid end invert values:",
    int(
        gdf[
            "Invert_End_m"
        ]
        .notna()
        .sum()
    )
)


print(
    "Valid absolute slope values:",
    int(
        gdf[
            "Invert_Slope_Abs"
        ]
        .notna()
        .sum()
    )
)


print("\nAbsolute invert slope statistics:")

print(
    gdf[
        "Invert_Slope_Abs"
    ]
    .describe()
)


# ============================================================
# 23. IMPUTATION SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("ENGINEERING DATA QUALITY")
print("=" * 80)


print(
    "\nDimension source:"
)

print(
    gdf[
        "Dimension_Source"
    ]
    .value_counts(
        dropna=False
    )
)


print(
    "\nWidth values imputed:",
    int(
        gdf[
            "Width_Imputed_Flag"
        ].sum()
    )
)


print(
    "Depth values imputed:",
    int(
        gdf[
            "Depth_Imputed_Flag"
        ].sum()
    )
)


print(
    "\nEngineering data quality:"
)

print(
    gdf[
        "Engineering_Data_Quality"
    ]
    .value_counts()
)


# ============================================================
# 24. MATERIAL / STATUS SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("MATERIAL / STATUS SUMMARY")
print("=" * 80)


print("\nMaterial:")

print(
    gdf[
        "Material_Clean"
    ]
    .value_counts()
    .head(
        20
    )
)


print("\nStatus:")

print(
    gdf[
        "Status_Clean"
    ]
    .value_counts()
    .head(
        20
    )
)


# ============================================================
# 25. TOP 20 CANDIDATES - ENGINEERING VIEW
# ============================================================

print("\n")
print("=" * 80)
print("TOP 20 CANDIDATES - ENGINEERING ATTRIBUTES")
print("=" * 80)


display_columns = [

    "Candidate_ID",

    "Drain_ID",

    "Segment_Length_m",

    "Mean_Flood_Prob",

    "Existing_Width_m",

    "Existing_Depth_m",

    "Existing_Area_m2",

    "Invert_Slope_Abs",

    "Material_Clean",

    "Status_Clean",

    "Engineering_Data_Quality"

]


print(

    gdf[
        display_columns
    ]

    .head(
        20
    )

    .to_string(
        index=False
    )

)


# ============================================================
# 26. SAVE GEOPACKAGE
# ============================================================

if OUTPUT_GPKG.exists():

    OUTPUT_GPKG.unlink()


gdf.to_file(

    OUTPUT_GPKG,

    layer="NSGAII_Engineering",

    driver="GPKG"

)


# ============================================================
# 27. SAVE CSV
# ============================================================

csv_output = gdf.drop(
    columns="geometry"
)


csv_output.to_csv(

    OUTPUT_CSV,

    index=False

)


# ============================================================
# 28. SUMMARY FILE
# ============================================================

summary = pd.DataFrame({

    "Metric": [

        "Candidate_Segments",

        "Parsed_Width_Available",

        "Parsed_Depth_Available",

        "Width_Imputed",

        "Depth_Imputed",

        "Valid_Start_Invert",

        "Valid_End_Invert",

        "Valid_Slope",

        "Median_Existing_Width_m",

        "Median_Existing_Depth_m",

        "Median_Existing_Area_m2",

        "Total_Existing_Volume_Proxy_m3"

    ],

    "Value": [

        len(
            gdf
        ),

        int(
            gdf[
                "Width_Parsed_m"
            ]
            .notna()
            .sum()
        ),

        int(
            gdf[
                "Depth_Parsed_m"
            ]
            .notna()
            .sum()
        ),

        int(
            gdf[
                "Width_Imputed_Flag"
            ]
            .sum()
        ),

        int(
            gdf[
                "Depth_Imputed_Flag"
            ]
            .sum()
        ),

        int(
            gdf[
                "Invert_Start_m"
            ]
            .notna()
            .sum()
        ),

        int(
            gdf[
                "Invert_End_m"
            ]
            .notna()
            .sum()
        ),

        int(
            gdf[
                "Invert_Slope_Abs"
            ]
            .notna()
            .sum()
        ),

        gdf[
            "Existing_Width_m"
        ].median(),

        gdf[
            "Existing_Depth_m"
        ].median(),

        gdf[
            "Existing_Area_m2"
        ].median(),

        gdf[
            "Existing_Volume_m3"
        ].sum()

    ]

})


SUMMARY_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)


summary.to_csv(

    SUMMARY_CSV,

    index=False

)


# ============================================================
# 29. OUTPUT
# ============================================================

print("\n")
print("=" * 80)
print("OUTPUT FILES")
print("=" * 80)


print(
    OUTPUT_GPKG
)


print(
    OUTPUT_CSV
)


print(
    SUMMARY_CSV
)


# ============================================================
# 30. FINAL STATUS
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 2 - STEP 37B STATUS: PASS")
print("=" * 80)


print(
    "Drainage engineering attributes cleaned and "
    "converted to NSGA-II-ready numeric variables."
)

print(
    "No upgrade assumptions or costs have been applied yet."
)

print(
    "Ready for Step 38: upgrade scenario and cost formulation."
)

print("\n")