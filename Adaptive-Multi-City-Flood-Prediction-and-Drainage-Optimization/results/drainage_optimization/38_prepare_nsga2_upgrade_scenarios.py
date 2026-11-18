# ============================================================
# PHASE 2 - STEP 38
# PREPARE NSGA-II UPGRADE SCENARIOS
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
    / "Chennai_NSGAII_Engineering_Ready.gpkg"
)

OUTPUT_CSV = (
    ROOT
    / "phase2_drainage"
    / "Chennai_NSGAII_Upgrade_Scenarios.csv"
)

SUMMARY_CSV = (
    ROOT
    / "results"
    / "38_nsga2_upgrade_scenario_summary.csv"
)


# ============================================================
# 2. UPGRADE ASSUMPTIONS
# ============================================================
#
# These are SCENARIO assumptions, not observed GCC values.
#
# Level 0:
#     No upgrade
#
# Level 1:
#     +15% width
#     +15% depth
#
# Level 2:
#     +30% width
#     +30% depth
#
# Cost is represented as an engineering proxy:
#
# added cross-sectional area × segment length
#
# This avoids inventing monetary unit rates.
#
# ============================================================

UPGRADE_LEVELS = {

    0: {
        "Name": "No_Upgrade",
        "Width_Factor": 1.00,
        "Depth_Factor": 1.00
    },

    1: {
        "Name": "Moderate",
        "Width_Factor": 1.15,
        "Depth_Factor": 1.15
    },

    2: {
        "Name": "Major",
        "Width_Factor": 1.30,
        "Depth_Factor": 1.30
    }

}


# ============================================================
# 3. START
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 2 - STEP 38")
print("NSGA-II UPGRADE SCENARIO PREPARATION")
print("=" * 80)


if not INPUT_GPKG.exists():

    raise FileNotFoundError(
        f"Missing Step 37B file:\n{INPUT_GPKG}"
    )


gdf = gpd.read_file(
    INPUT_GPKG
)


print("\nCandidate segments:")
print(
    len(gdf)
)


# ============================================================
# 4. REQUIRED VARIABLES
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

    "Existing_Width_m",
    "Existing_Depth_m",
    "Existing_Area_m2",

    "Invert_Slope_Abs"

]


missing = [

    col
    for col in required
    if col not in gdf.columns

]


if missing:

    raise ValueError(
        f"Missing variables:\n{missing}"
    )


# ============================================================
# 5. BASELINE RISK
# ============================================================
#
# Risk burden combines:
#
# flood probability
# × exposed drainage length
#
# ============================================================

gdf[
    "Baseline_Risk"
] = (

    gdf[
        "Mean_Flood_Prob"
    ]

    *

    gdf[
        "Segment_Length_m"
    ]

    *

    (
        gdf[
            "Flood_Exposure_Pct"
        ]
        /
        100.0
    )

)


# ============================================================
# 6. CREATE SCENARIO TABLE
# ============================================================

rows = []


for _, row in gdf.iterrows():

    existing_width = float(
        row[
            "Existing_Width_m"
        ]
    )

    existing_depth = float(
        row[
            "Existing_Depth_m"
        ]
    )

    existing_area = float(
        row[
            "Existing_Area_m2"
        ]
    )

    segment_length = float(
        row[
            "Segment_Length_m"
        ]
    )

    baseline_risk = float(
        row[
            "Baseline_Risk"
        ]
    )


    for level, params in UPGRADE_LEVELS.items():

        new_width = (
            existing_width
            *
            params[
                "Width_Factor"
            ]
        )


        new_depth = (
            existing_depth
            *
            params[
                "Depth_Factor"
            ]
        )


        new_area = (
            new_width
            *
            new_depth
        )


        area_gain = (
            new_area
            -
            existing_area
        )


        area_gain_percent = (

            100
            *
            area_gain
            /
            existing_area

            if existing_area > 0

            else 0.0
        )


        # ----------------------------------------------------
        # Relative capacity proxy
        # ----------------------------------------------------
        #
        # Without complete hydraulic calibration,
        # cross-sectional area increase is used as a
        # transparent relative capacity proxy.
        #
        # ----------------------------------------------------

        capacity_ratio = (

            new_area
            /
            existing_area

            if existing_area > 0

            else 1.0
        )


        capacity_gain = (
            capacity_ratio
            -
            1.0
        )


        # ----------------------------------------------------
        # COST PROXY
        # ----------------------------------------------------
        #
        # Additional geometric volume that would require
        # reconstruction / enlargement.
        #
        # Units are m³-equivalent, NOT rupees.
        #
        # ----------------------------------------------------

        added_volume = (

            max(
                area_gain,
                0.0
            )

            *
            segment_length

        )


        # ----------------------------------------------------
        # RISK REDUCTION PROXY
        # ----------------------------------------------------
        #
        # No upgrade:
        #     0 reduction
        #
        # Upgrade benefit increases with capacity gain.
        #
        # We deliberately cap the reduction because a drain
        # upgrade cannot be assumed to eliminate all flooding.
        #
        # ----------------------------------------------------

        if level == 0:

            reduction_fraction = 0.0

        else:

            reduction_fraction = min(

                capacity_gain
                /
                (
                    1.0
                    +
                    capacity_gain
                ),

                0.60

            )


        risk_reduction = (

            baseline_risk
            *
            reduction_fraction

        )


        residual_risk = (

            baseline_risk
            -
            risk_reduction

        )


        rows.append({

            "Candidate_ID":
                int(
                    row[
                        "Candidate_ID"
                    ]
                ),

            "Drain_ID":
                int(
                    row[
                        "Drain_ID"
                    ]
                ),

            "Upgrade_Level":
                level,

            "Upgrade_Name":
                params[
                    "Name"
                ],

            "Segment_Length_m":
                segment_length,

            "Mean_Flood_Prob":
                float(
                    row[
                        "Mean_Flood_Prob"
                    ]
                ),

            "Flood_Exposure_Pct":
                float(
                    row[
                        "Flood_Exposure_Pct"
                    ]
                ),

            "Priority_Score":
                float(
                    row[
                        "Priority_Score"
                    ]
                ),

            "Existing_Width_m":
                existing_width,

            "Existing_Depth_m":
                existing_depth,

            "Existing_Area_m2":
                existing_area,

            "Upgrade_Width_m":
                new_width,

            "Upgrade_Depth_m":
                new_depth,

            "Upgrade_Area_m2":
                new_area,

            "Area_Gain_m2":
                area_gain,

            "Area_Gain_Percent":
                area_gain_percent,

            "Capacity_Ratio_Proxy":
                capacity_ratio,

            "Capacity_Gain_Proxy":
                capacity_gain,

            "Added_Volume_Cost_Proxy":
                added_volume,

            "Baseline_Risk":
                baseline_risk,

            "Risk_Reduction_Fraction":
                reduction_fraction,

            "Risk_Reduction":
                risk_reduction,

            "Residual_Risk":
                residual_risk

        })


scenario = pd.DataFrame(
    rows
)


# ============================================================
# 7. VALIDATION
# ============================================================

expected_rows = (
    len(gdf)
    *
    len(UPGRADE_LEVELS)
)


print("\nExpected scenario rows:")
print(
    expected_rows
)


print("\nActual scenario rows:")
print(
    len(scenario)
)


if len(scenario) != expected_rows:

    raise RuntimeError(
        "Scenario row count mismatch."
    )


# ============================================================
# 8. SUMMARY BY UPGRADE LEVEL
# ============================================================

summary = (

    scenario

    .groupby(
        [
            "Upgrade_Level",
            "Upgrade_Name"
        ]
    )

    .agg(

        Segments=(
            "Candidate_ID",
            "count"
        ),

        Mean_Area_Gain_Percent=(
            "Area_Gain_Percent",
            "mean"
        ),

        Total_Cost_Proxy=(
            "Added_Volume_Cost_Proxy",
            "sum"
        ),

        Mean_Risk_Reduction_Fraction=(
            "Risk_Reduction_Fraction",
            "mean"
        ),

        Total_Risk_Reduction=(
            "Risk_Reduction",
            "sum"
        ),

        Total_Residual_Risk=(
            "Residual_Risk",
            "sum"
        )

    )

    .reset_index()

)


print("\n")
print("=" * 80)
print("UPGRADE SCENARIO SUMMARY")
print("=" * 80)


print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 9. EXAMPLE TOP CANDIDATES
# ============================================================

print("\n")
print("=" * 80)
print("FIRST 15 CANDIDATES - UPGRADE OPTIONS")
print("=" * 80)


example = scenario[

    scenario[
        "Candidate_ID"
    ]
    <=
    15

]


print(

    example[
        [

            "Candidate_ID",

            "Upgrade_Level",

            "Upgrade_Name",

            "Existing_Width_m",

            "Existing_Depth_m",

            "Upgrade_Width_m",

            "Upgrade_Depth_m",

            "Area_Gain_Percent",

            "Added_Volume_Cost_Proxy",

            "Risk_Reduction_Fraction",

            "Residual_Risk"

        ]
    ]

    .to_string(
        index=False
    )

)


# ============================================================
# 10. CHECK MONOTONICITY
# ============================================================
#
# For every segment:
#
# Higher upgrade level should:
# - increase cost proxy
# - reduce residual risk
#
# ============================================================

bad_segments = []


for candidate_id, group in scenario.groupby(
    "Candidate_ID"
):

    group = group.sort_values(
        "Upgrade_Level"
    )


    cost = group[
        "Added_Volume_Cost_Proxy"
    ].to_numpy()


    residual = group[
        "Residual_Risk"
    ].to_numpy()


    if not np.all(
        np.diff(
            cost
        )
        >=
        -1e-9
    ):

        bad_segments.append(
            candidate_id
        )

        continue


    if not np.all(
        np.diff(
            residual
        )
        <=
        1e-9
    ):

        bad_segments.append(
            candidate_id
        )


print("\n")
print("=" * 80)
print("SCENARIO VALIDATION")
print("=" * 80)


print(
    "Segments failing monotonicity check:",
    len(
        bad_segments
    )
)


if bad_segments:

    print(
        bad_segments[
            :20
        ]
    )

    raise RuntimeError(
        "Upgrade scenario validation failed."
    )


print(
    "Scenario monotonicity: PASS"
)


# ============================================================
# 11. SAVE
# ============================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)


scenario.to_csv(

    OUTPUT_CSV,

    index=False

)


summary.to_csv(

    SUMMARY_CSV,

    index=False

)


# ============================================================
# 12. OUTPUT
# ============================================================

print("\n")
print("=" * 80)
print("OUTPUT FILES")
print("=" * 80)


print(
    OUTPUT_CSV
)


print(
    SUMMARY_CSV
)


# ============================================================
# 13. FINAL STATUS
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 2 - STEP 38 STATUS: PASS")
print("=" * 80)


print(
    "Three upgrade scenarios prepared for all "
    "500 NSGA-II candidate segments."
)

print(
    "Cost is currently represented by an engineering "
    "volume proxy, not monetary currency."
)

print(
    "Ready for Step 39: NSGA-II optimization."
)

print("\n")