# ============================================================
# PHASE 2 - STEP 41
# EXPORT FINAL NSGA-II DRAINAGE PLANS FOR QGIS
# ============================================================

from pathlib import Path
import pandas as pd
import geopandas as gpd


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(
    r"D:\Adaptive_Multi_City_Flood_Prediction"
)

ENGINEERING_GPKG = (
    ROOT
    / "phase2_drainage"
    / "Chennai_NSGAII_Engineering_Ready.gpkg"
)

SELECTED_DECISIONS = (
    ROOT
    / "results"
    / "40_selected_pareto_decisions.csv"
)

SELECTED_SUMMARY = (
    ROOT
    / "results"
    / "40_selected_pareto_solutions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "phase2_drainage"
    / "final_qgis"
)

LOW_COST_GPKG = (
    OUTPUT_DIR
    / "Chennai_NSGAII_Low_Cost_Plan.gpkg"
)

BALANCED_GPKG = (
    OUTPUT_DIR
    / "Chennai_NSGAII_Balanced_Plan.gpkg"
)

MAX_RISK_GPKG = (
    OUTPUT_DIR
    / "Chennai_NSGAII_Max_Risk_Reduction_Plan.gpkg"
)

FINAL_SUMMARY = (
    ROOT
    / "results"
    / "41_final_phase2_summary.csv"
)


# ============================================================
# 2. START
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 2 - STEP 41")
print("EXPORT FINAL NSGA-II QGIS LAYERS")
print("=" * 80)


for path in [
    ENGINEERING_GPKG,
    SELECTED_DECISIONS,
    SELECTED_SUMMARY
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. LOAD BASE CANDIDATE GEOMETRIES
# ============================================================

base = gpd.read_file(
    ENGINEERING_GPKG
)

decisions = pd.read_csv(
    SELECTED_DECISIONS
)

summary = pd.read_csv(
    SELECTED_SUMMARY
)


print("\nCandidate geometries:")
print(
    len(base)
)


print("\nSelected decision rows:")
print(
    len(decisions)
)


# ============================================================
# 4. REQUIRED COLUMNS
# ============================================================

required_base = [

    "Candidate_ID",
    "Drain_ID",
    "Segment_Length_m",
    "Mean_Flood_Prob",
    "Priority_Score",
    "Risk_Class",
    "Existing_Width_m",
    "Existing_Depth_m"

]


missing_base = [

    col
    for col in required_base
    if col not in base.columns

]


if missing_base:

    raise ValueError(
        f"Missing base columns:\n{missing_base}"
    )


required_decisions = [

    "Plan",
    "Candidate_ID",
    "Drain_ID",
    "Upgrade_Level",
    "Upgrade_Name"

]


missing_decisions = [

    col
    for col in required_decisions
    if col not in decisions.columns

]


if missing_decisions:

    raise ValueError(
        f"Missing decision columns:\n{missing_decisions}"
    )


# ============================================================
# 5. PLAN EXPORT FUNCTION
# ============================================================

def export_plan(
    plan_name,
    output_file
):

    plan_decisions = decisions[

        decisions[
            "Plan"
        ]
        ==
        plan_name

    ].copy()


    if len(plan_decisions) != len(base):

        raise RuntimeError(
            f"{plan_name}: expected {len(base)} decisions, "
            f"found {len(plan_decisions)}"
        )


    merged = base.merge(

        plan_decisions[

            [
                "Candidate_ID",
                "Upgrade_Level",
                "Upgrade_Name"
            ]

        ],

        on="Candidate_ID",

        how="left",

        validate="one_to_one"

    )


    if merged[
        "Upgrade_Level"
    ].isna().any():

        raise RuntimeError(
            f"{plan_name}: missing upgrade decisions after merge."
        )


    # --------------------------------------------------------
    # Upgrade status
    # --------------------------------------------------------

    merged[
        "Upgrade_Selected"
    ] = (

        merged[
            "Upgrade_Level"
        ]
        >
        0

    ).astype(
        int
    )


    # --------------------------------------------------------
    # Proposed dimensions
    # --------------------------------------------------------

    merged[
        "Proposed_Width_m"
    ] = merged[
        "Existing_Width_m"
    ]


    merged[
        "Proposed_Depth_m"
    ] = merged[
        "Existing_Depth_m"
    ]


    moderate_mask = (

        merged[
            "Upgrade_Level"
        ]
        ==
        1

    )


    major_mask = (

        merged[
            "Upgrade_Level"
        ]
        ==
        2

    )


    merged.loc[
        moderate_mask,
        "Proposed_Width_m"
    ] = (

        merged.loc[
            moderate_mask,
            "Existing_Width_m"
        ]
        *
        1.15

    )


    merged.loc[
        moderate_mask,
        "Proposed_Depth_m"
    ] = (

        merged.loc[
            moderate_mask,
            "Existing_Depth_m"
        ]
        *
        1.15

    )


    merged.loc[
        major_mask,
        "Proposed_Width_m"
    ] = (

        merged.loc[
            major_mask,
            "Existing_Width_m"
        ]
        *
        1.30

    )


    merged.loc[
        major_mask,
        "Proposed_Depth_m"
    ] = (

        merged.loc[
            major_mask,
            "Existing_Depth_m"
        ]
        *
        1.30

    )


    # --------------------------------------------------------
    # Plan label
    # --------------------------------------------------------

    merged[
        "Plan"
    ] = plan_name


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if output_file.exists():

        output_file.unlink()


    merged.to_file(

        output_file,

        layer=plan_name,

        driver="GPKG"

    )


    return merged


# ============================================================
# 6. EXPORT ALL THREE PLANS
# ============================================================

low_cost = export_plan(

    "LOW_COST",

    LOW_COST_GPKG

)


balanced = export_plan(

    "BALANCED",

    BALANCED_GPKG

)


max_risk = export_plan(

    "MAX_RISK_REDUCTION",

    MAX_RISK_GPKG

)


# ============================================================
# 7. VALIDATION SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("FINAL PLAN VALIDATION")
print("=" * 80)


for plan_name, gdf in [

    (
        "LOW_COST",
        low_cost
    ),

    (
        "BALANCED",
        balanced
    ),

    (
        "MAX_RISK_REDUCTION",
        max_risk
    )

]:

    print("\n" + "-" * 80)

    print(
        plan_name
    )

    print("-" * 80)


    print(
        "Segments:",
        len(
            gdf
        )
    )


    print(
        "Upgraded segments:",
        int(
            gdf[
                "Upgrade_Selected"
            ].sum()
        )
    )


    print(
        "Moderate:",
        int(
            (
                gdf[
                    "Upgrade_Level"
                ]
                ==
                1
            ).sum()
        )
    )


    print(
        "Major:",
        int(
            (
                gdf[
                    "Upgrade_Level"
                ]
                ==
                2
            ).sum()
        )
    )


    upgraded_length_km = (

        gdf.loc[

            gdf[
                "Upgrade_Selected"
            ]
            ==
            1,

            "Segment_Length_m"

        ].sum()

        /
        1000

    )


    print(
        "Upgraded length (km):",
        round(
            upgraded_length_km,
            3
        )
    )


# ============================================================
# 8. CREATE FINAL PROJECT SUMMARY
# ============================================================

phase2_summary = summary[

    [

        "Plan",

        "Solution_ID",

        "Residual_Risk",

        "Risk_Reduction",

        "Risk_Reduction_Percent",

        "Cost_Proxy_m3",

        "Upgraded_Length_km",

        "Upgraded_Segments",

        "Moderate_Count",

        "Major_Count"

    ]

].copy()


phase2_summary.to_csv(

    FINAL_SUMMARY,

    index=False

)


# ============================================================
# 9. PRINT FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("FINAL PHASE-2 OPTIMIZATION SUMMARY")
print("=" * 80)


print(

    phase2_summary.to_string(
        index=False
    )

)


# ============================================================
# 10. OUTPUT FILES
# ============================================================

print("\n")
print("=" * 80)
print("FINAL QGIS FILES")
print("=" * 80)


print(
    LOW_COST_GPKG
)

print(
    BALANCED_GPKG
)

print(
    MAX_RISK_GPKG
)

print(
    FINAL_SUMMARY
)


# ============================================================
# 11. STATUS
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 2 - STEP 41 STATUS: PASS")
print("=" * 80)


print(
    "Final Chennai NSGA-II drainage upgrade plans "
    "exported for QGIS."
)

print(
    "Phase 2 core optimization workflow is complete."
)

print("\n")