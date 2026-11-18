# ============================================================
# PHASE 2 - STEP 36
# SELECT CRITICAL DRAINAGE SEGMENTS FOR NSGA-II
# ============================================================
#
# INPUT:
# Chennai_SWD_Risk_Attributed.gpkg
#
# PURPOSE:
# Reduce the 5,205 High / Very-High risk drainage segments
# to a practical candidate set for NSGA-II.
#
# Candidate selection is based on:
#
#   - Raster overlap
#   - High / Very High risk class
#   - Mean flood probability
#   - Flood exposure
#   - Segment length
#   - Priority score
#
# A risk-exposure index is calculated:
#
# Risk_Exposure_Index =
#       Priority_Score
#       × Segment_Length_m
#
# Top 500 critical segments are retained.
#
# ============================================================


from pathlib import Path

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
    / "Chennai_SWD_Risk_Attributed.gpkg"
)

OUTPUT_GPKG = (
    ROOT
    / "phase2_drainage"
    / "Chennai_NSAGII_Candidate_Drains.gpkg"
)

OUTPUT_CSV = (
    ROOT
    / "phase2_drainage"
    / "Chennai_NSAGII_Candidate_Drains.csv"
)

SUMMARY_CSV = (
    ROOT
    / "results"
    / "36_nsga2_candidate_summary.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

MAX_CANDIDATES = 500

MIN_MEAN_FLOOD_PROB = 0.60

MIN_FLOOD_EXPOSURE = 50.0

VALID_RISK_CLASSES = [
    "High",
    "Very_High"
]


# ============================================================
# 3. START
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 2 - STEP 36")
print("NSGA-II CRITICAL DRAINAGE CANDIDATE SELECTION")
print("=" * 80)


if not INPUT_GPKG.exists():

    raise FileNotFoundError(
        f"Missing Step 35 file:\n{INPUT_GPKG}"
    )


# ============================================================
# 4. LOAD STEP 35 NETWORK
# ============================================================

drains = gpd.read_file(
    INPUT_GPKG
)


print("\nInput drainage segments:")
print(
    len(drains)
)


print("\nCRS:")
print(
    drains.crs
)


# ============================================================
# 5. REQUIRED COLUMNS
# ============================================================

required = [

    "Drain_ID",

    "Segment_Length_m",

    "Mean_Flood_Prob",

    "Max_Flood_Prob",

    "Flood_Exposure_Pct",

    "Very_High_Exposure_Pct",

    "Priority_Score",

    "Risk_Class",

    "Raster_Overlap"

]


missing = [

    col
    for col in required
    if col not in drains.columns

]


if missing:

    raise ValueError(
        f"Missing required columns:\n{missing}"
    )


# ============================================================
# 6. BASE ELIGIBILITY FILTER
# ============================================================

eligible = drains[

    (
        drains[
            "Raster_Overlap"
        ]
        ==
        1
    )

    &

    (
        drains[
            "Risk_Class"
        ]
        .isin(
            VALID_RISK_CLASSES
        )
    )

    &

    (
        drains[
            "Mean_Flood_Prob"
        ]
        >=
        MIN_MEAN_FLOOD_PROB
    )

    &

    (
        drains[
            "Flood_Exposure_Pct"
        ]
        >=
        MIN_FLOOD_EXPOSURE
    )

].copy()


print("\n")
print("=" * 80)
print("ELIGIBLE HIGH-RISK SEGMENTS")
print("=" * 80)


print(
    "\nEligible segments:",
    len(eligible)
)


print(
    "\nEligible drainage length (km):",
    round(
        eligible[
            "Segment_Length_m"
        ].sum()
        /
        1000,
        3
    )
)


print(
    "\nRisk classes:"
)

print(
    eligible[
        "Risk_Class"
    ]
    .value_counts()
)


# ============================================================
# 7. RISK-EXPOSURE INDEX
# ============================================================
#
# Priority score by itself favors very high probability.
#
# For infrastructure optimization, we also care about how
# much drainage length is exposed.
#
# Risk_Exposure_Index =
#     Priority_Score × Segment_Length_m
#
# ============================================================

eligible[
    "Risk_Exposure_Index"
] = (

    eligible[
        "Priority_Score"
    ]

    *

    eligible[
        "Segment_Length_m"
    ]

)


# ============================================================
# 8. NORMALIZED COMPONENTS
# ============================================================

def minmax(series):

    minimum = series.min()

    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            np.ones(
                len(series)
            ),
            index=series.index
        )

    return (
        series - minimum
    ) / (
        maximum - minimum
    )


eligible[
    "Norm_Mean_Risk"
] = minmax(

    eligible[
        "Mean_Flood_Prob"
    ]

)


eligible[
    "Norm_Max_Risk"
] = minmax(

    eligible[
        "Max_Flood_Prob"
    ]

)


eligible[
    "Norm_Exposure"
] = (

    eligible[
        "Flood_Exposure_Pct"
    ]
    /
    100.0

)


eligible[
    "Norm_Length"
] = minmax(

    eligible[
        "Segment_Length_m"
    ]

)


# ============================================================
# 9. NSGA-II CANDIDATE PRIORITY SCORE
# ============================================================
#
# This score is ONLY used to choose the optimization
# candidate pool.
#
# It is NOT an NSGA-II objective.
#
# Weights:
#
# Mean flood risk       35%
# Maximum flood risk    20%
# Flood exposure        30%
# Segment length        15%
#
# ============================================================

eligible[
    "Candidate_Priority"
] = (

    0.35
    *
    eligible[
        "Norm_Mean_Risk"
    ]

    +

    0.20
    *
    eligible[
        "Norm_Max_Risk"
    ]

    +

    0.30
    *
    eligible[
        "Norm_Exposure"
    ]

    +

    0.15
    *
    eligible[
        "Norm_Length"
    ]

)


# ============================================================
# 10. SORT CRITICAL SEGMENTS
# ============================================================

eligible = eligible.sort_values(

    by=[

        "Candidate_Priority",

        "Risk_Exposure_Index"

    ],

    ascending=[

        False,

        False

    ]

).reset_index(
    drop=True
)


# ============================================================
# 11. SELECT TOP CANDIDATES
# ============================================================

candidate_count = min(

    MAX_CANDIDATES,

    len(eligible)

)


candidates = eligible.head(
    candidate_count
).copy()


candidates[
    "Candidate_ID"
] = np.arange(

    1,

    len(candidates) + 1

)


# ============================================================
# 12. REORDER IMPORTANT COLUMNS
# ============================================================

front_columns = [

    "Candidate_ID",

    "Drain_ID",

    "Segment_Length_m",

    "Mean_Flood_Prob",

    "Max_Flood_Prob",

    "Flood_Exposure_Pct",

    "Very_High_Exposure_Pct",

    "Priority_Score",

    "Risk_Exposure_Index",

    "Candidate_Priority",

    "Risk_Class"

]


other_columns = [

    col
    for col in candidates.columns

    if col not in front_columns
    and col != "geometry"

]


candidates = candidates[

    front_columns
    +
    other_columns
    +
    [
        "geometry"
    ]

]


# ============================================================
# 13. SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("FINAL NSGA-II CANDIDATE SET")
print("=" * 80)


print(
    "\nSelected candidates:",
    len(candidates)
)


print(
    "\nCandidate drainage length (km):",
    round(
        candidates[
            "Segment_Length_m"
        ].sum()
        /
        1000,
        3
    )
)


print(
    "\nMean flood probability:"
)

print(
    candidates[
        "Mean_Flood_Prob"
    ].describe()
)


print(
    "\nFlood exposure percentage:"
)

print(
    candidates[
        "Flood_Exposure_Pct"
    ].describe()
)


print(
    "\nRisk class distribution:"
)

print(
    candidates[
        "Risk_Class"
    ]
    .value_counts()
)


# ============================================================
# 14. COVERAGE OF HIGH-RISK NETWORK
# ============================================================

eligible_length = (

    eligible[
        "Segment_Length_m"
    ].sum()

)


candidate_length = (

    candidates[
        "Segment_Length_m"
    ].sum()

)


length_coverage = (

    100
    *
    candidate_length
    /
    eligible_length

)


risk_exposure_coverage = (

    100
    *
    candidates[
        "Risk_Exposure_Index"
    ].sum()
    /
    eligible[
        "Risk_Exposure_Index"
    ].sum()

)


print(
    "\nCandidate length coverage of eligible network:",
    round(
        length_coverage,
        2
    ),
    "%"
)


print(
    "Candidate risk-exposure coverage:",
    round(
        risk_exposure_coverage,
        2
    ),
    "%"
)


# ============================================================
# 15. TOP 25 CANDIDATES
# ============================================================

print("\n")
print("=" * 80)
print("TOP 25 NSGA-II CANDIDATE SEGMENTS")
print("=" * 80)


print(

    candidates[

        [

            "Candidate_ID",

            "Drain_ID",

            "Segment_Length_m",

            "Mean_Flood_Prob",

            "Max_Flood_Prob",

            "Flood_Exposure_Pct",

            "Priority_Score",

            "Candidate_Priority",

            "Risk_Class"

        ]

    ]

    .head(
        25
    )

    .to_string(
        index=False
    )

)


# ============================================================
# 16. SAVE GEOPACKAGE
# ============================================================

if OUTPUT_GPKG.exists():

    OUTPUT_GPKG.unlink()


candidates.to_file(

    OUTPUT_GPKG,

    layer="NSGAII_Candidates",

    driver="GPKG"

)


# ============================================================
# 17. SAVE CSV
# ============================================================

candidate_csv = candidates.drop(

    columns="geometry"

)


candidate_csv.to_csv(

    OUTPUT_CSV,

    index=False

)


# ============================================================
# 18. SAVE SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Metric": [

        "Original_Drainage_Segments",

        "Eligible_High_Risk_Segments",

        "Selected_NSAGII_Candidates",

        "Eligible_Drainage_Length_km",

        "Candidate_Drainage_Length_km",

        "Candidate_Length_Coverage_Percent",

        "Candidate_Risk_Exposure_Coverage_Percent",

        "Candidate_Mean_Flood_Probability",

        "Candidate_Mean_Exposure_Percent"

    ],

    "Value": [

        len(
            drains
        ),

        len(
            eligible
        ),

        len(
            candidates
        ),

        eligible_length
        /
        1000,

        candidate_length
        /
        1000,

        length_coverage,

        risk_exposure_coverage,

        candidates[
            "Mean_Flood_Prob"
        ].mean(),

        candidates[
            "Flood_Exposure_Pct"
        ].mean()

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
# 19. OUTPUT FILES
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
# 20. FINAL STATUS
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 2 - STEP 36 STATUS: PASS")
print("=" * 80)


print(
    f"{len(candidates)} critical drainage segments "
    "selected for NSGA-II optimization."
)

print(
    "No optimization has been performed yet."
)

print(
    "Candidate selection is based only on flood-risk "
    "and exposure characteristics."
)

print("\n")