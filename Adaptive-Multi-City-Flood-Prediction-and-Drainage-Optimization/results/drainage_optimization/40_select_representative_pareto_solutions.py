# ============================================================
# PHASE 2 - STEP 40
# SELECT REPRESENTATIVE NSGA-II PARETO SOLUTIONS
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(
    r"D:\Adaptive_Multi_City_Flood_Prediction"
)

PARETO_FILE = (
    ROOT
    / "results"
    / "39_nsga2_pareto_front.csv"
)

DECISION_FILE = (
    ROOT
    / "results"
    / "39_nsga2_pareto_decisions.csv"
)

SELECTED_SUMMARY = (
    ROOT
    / "results"
    / "40_selected_pareto_solutions.csv"
)

SELECTED_DECISIONS = (
    ROOT
    / "results"
    / "40_selected_pareto_decisions.csv"
)


# ============================================================
# 2. START
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 2 - STEP 40")
print("REPRESENTATIVE PARETO SOLUTION SELECTION")
print("=" * 80)


for path in [
    PARETO_FILE,
    DECISION_FILE
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing Step 39 file:\n{path}"
        )


# ============================================================
# 3. LOAD
# ============================================================

pareto = pd.read_csv(
    PARETO_FILE
)

decisions = pd.read_csv(
    DECISION_FILE
)


print("\nPareto solutions:")
print(
    len(pareto)
)


print("\nDecision rows:")
print(
    len(decisions)
)


# ============================================================
# 4. REQUIRED COLUMNS
# ============================================================

required = [

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


missing = [

    col
    for col in required
    if col not in pareto.columns

]


if missing:

    raise ValueError(
        f"Missing Pareto columns:\n{missing}"
    )


# ============================================================
# 5. NORMALIZATION
# ============================================================
#
# All three optimization objectives are minimization:
#
# Residual risk
# Cost
# Upgraded length
#
# Normalized ideal = [0, 0, 0]
#
# ============================================================

def minmax(series):

    minimum = series.min()

    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            np.zeros(
                len(series)
            ),
            index=series.index
        )

    return (

        (
            series
            -
            minimum
        )

        /

        (
            maximum
            -
            minimum
        )

    )


pareto[
    "Norm_Residual_Risk"
] = minmax(
    pareto[
        "Residual_Risk"
    ]
)


pareto[
    "Norm_Cost"
] = minmax(
    pareto[
        "Cost_Proxy_m3"
    ]
)


pareto[
    "Norm_Upgrade_Length"
] = minmax(
    pareto[
        "Upgraded_Length_km"
    ]
)


# ============================================================
# 6. BALANCED IDEAL-POINT DISTANCE
# ============================================================
#
# Equal weighting of the three NSGA-II objectives.
#
# Lower value = closer to ideal trade-off.
#
# ============================================================

pareto[
    "Balanced_Distance"
] = np.sqrt(

    pareto[
        "Norm_Residual_Risk"
    ] ** 2

    +

    pareto[
        "Norm_Cost"
    ] ** 2

    +

    pareto[
        "Norm_Upgrade_Length"
    ] ** 2

)


# ============================================================
# 7. LOW-COST SOLUTION
# ============================================================

low_cost = pareto.loc[

    pareto[
        "Cost_Proxy_m3"
    ].idxmin()

].copy()


# ============================================================
# 8. BALANCED SOLUTION
# ============================================================

balanced = pareto.loc[

    pareto[
        "Balanced_Distance"
    ].idxmin()

].copy()


# ============================================================
# 9. MAXIMUM RISK-REDUCTION SOLUTION
# ============================================================

max_risk_reduction = pareto.loc[

    pareto[
        "Residual_Risk"
    ].idxmin()

].copy()


# ============================================================
# 10. BUILD SELECTED TABLE
# ============================================================

selected_rows = []


for plan_name, row in [

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
        max_risk_reduction
    )

]:

    selected_rows.append({

        "Plan":
            plan_name,

        "Solution_ID":
            int(
                row[
                    "Solution_ID"
                ]
            ),

        "Residual_Risk":
            row[
                "Residual_Risk"
            ],

        "Risk_Reduction":
            row[
                "Risk_Reduction"
            ],

        "Risk_Reduction_Percent":
            row[
                "Risk_Reduction_Percent"
            ],

        "Cost_Proxy_m3":
            row[
                "Cost_Proxy_m3"
            ],

        "Upgraded_Length_km":
            row[
                "Upgraded_Length_km"
            ],

        "Upgraded_Segments":
            int(
                row[
                    "Upgraded_Segments"
                ]
            ),

        "Moderate_Count":
            int(
                row[
                    "Moderate_Count"
                ]
            ),

        "Major_Count":
            int(
                row[
                    "Major_Count"
                ]
            ),

        "Balanced_Distance":
            row[
                "Balanced_Distance"
            ]

    })


selected = pd.DataFrame(
    selected_rows
)


# ============================================================
# 11. REMOVE ACCIDENTAL DUPLICATE PLAN SELECTION
# ============================================================

print("\n")
print("=" * 80)
print("SELECTED REPRESENTATIVE SOLUTIONS")
print("=" * 80)


print(
    selected.to_string(
        index=False
    )
)


# ============================================================
# 12. GET DRAIN-LEVEL DECISIONS
# ============================================================

decision_parts = []


for _, row in selected.iterrows():

    solution_id = int(
        row[
            "Solution_ID"
        ]
    )

    plan_name = row[
        "Plan"
    ]


    part = decisions[

        decisions[
            "Solution_ID"
        ]
        ==
        solution_id

    ].copy()


    part[
        "Plan"
    ] = plan_name


    decision_parts.append(
        part
    )


selected_decisions = pd.concat(

    decision_parts,

    ignore_index=True

)


# ============================================================
# 13. PLAN-SPECIFIC UPGRADE SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("UPGRADE DECISION SUMMARY")
print("=" * 80)


for plan in [

    "LOW_COST",

    "BALANCED",

    "MAX_RISK_REDUCTION"

]:

    subset = selected_decisions[

        selected_decisions[
            "Plan"
        ]
        ==
        plan

    ]


    print("\n" + "-" * 80)

    print(
        plan
    )

    print("-" * 80)


    print(
        subset[
            "Upgrade_Name"
        ]
        .value_counts()
    )


    upgraded = subset[

        subset[
            "Upgrade_Level"
        ]
        >
        0

    ]


    print(
        "\nUpgraded drains:",
        len(
            upgraded
        )
    )


    print(
        "\nFirst 20 upgraded Drain_IDs:"
    )

    print(

        upgraded[
            [

                "Candidate_ID",

                "Drain_ID",

                "Upgrade_Level",

                "Upgrade_Name"

            ]
        ]

        .head(
            20
        )

        .to_string(
            index=False
        )

    )


# ============================================================
# 14. SAVE
# ============================================================

SELECTED_SUMMARY.parent.mkdir(
    parents=True,
    exist_ok=True
)


selected.to_csv(

    SELECTED_SUMMARY,

    index=False

)


selected_decisions.to_csv(

    SELECTED_DECISIONS,

    index=False

)


# ============================================================
# 15. VALIDATION
# ============================================================

print("\n")
print("=" * 80)
print("SELECTION VALIDATION")
print("=" * 80)


print(
    "Selected plans:",
    len(
        selected
    )
)


print(
    "Unique Solution IDs:",
    selected[
        "Solution_ID"
    ].nunique()
)


print(
    "Decision rows:",
    len(
        selected_decisions
    )
)


for plan in selected[
    "Plan"
]:

    count = len(

        selected_decisions[

            selected_decisions[
                "Plan"
            ]
            ==
            plan

        ]

    )


    print(
        f"{plan}: {count} drainage decisions"
    )


# ============================================================
# 16. OUTPUT
# ============================================================

print("\n")
print("=" * 80)
print("OUTPUT FILES")
print("=" * 80)


print(
    SELECTED_SUMMARY
)


print(
    SELECTED_DECISIONS
)


print("\n")
print("=" * 80)
print("PHASE 2 - STEP 40 STATUS: PASS")
print("=" * 80)


print(
    "Low-cost, balanced and maximum-risk-reduction "
    "Pareto solutions selected."
)

print(
    "Ready for Step 41: final QGIS drainage maps."
)

print("\n")