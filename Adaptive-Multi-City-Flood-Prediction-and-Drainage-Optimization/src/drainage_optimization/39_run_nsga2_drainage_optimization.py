# ============================================================
# PHASE 2 - STEP 39
# NSGA-II SMART DRAINAGE OPTIMIZATION
# ============================================================
#
# DECISION VARIABLE FOR EACH OF 500 DRAIN SEGMENTS:
#
# 0 = No Upgrade
# 1 = Moderate Upgrade
# 2 = Major Upgrade
#
# OBJECTIVES:
#
# F1 = Minimize Total Residual Flood Risk
#
# F2 = Minimize Total Upgrade Cost Proxy
#      (additional drainage volume, m3-equivalent)
#
# F3 = Minimize Total Upgraded Drainage Length
#
# OUTPUT:
#
# - Complete Pareto front
# - Decision vector for every Pareto solution
# - Objective values
# - Upgrade counts
#
# ============================================================


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. IMPORT PYMOO
# ============================================================

try:

    from pymoo.core.problem import Problem

    from pymoo.algorithms.moo.nsga2 import NSGA2

    from pymoo.optimize import minimize

    from pymoo.operators.sampling.rnd import IntegerRandomSampling

    from pymoo.operators.crossover.sbx import SBX

    from pymoo.operators.mutation.pm import PM

    from pymoo.core.repair import Repair

except ImportError:

    raise ImportError(
        "\n'pymoo' is not installed.\n\n"
        "Install it using:\n"
        "pip install pymoo\n\n"
        "Then run Step 39 again."
    )


# ============================================================
# 2. PATHS
# ============================================================

ROOT = Path(
    r"D:\Adaptive_Multi_City_Flood_Prediction"
)


SCENARIO_FILE = (
    ROOT
    / "phase2_drainage"
    / "Chennai_NSGAII_Upgrade_Scenarios.csv"
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


SUMMARY_FILE = (
    ROOT
    / "results"
    / "39_nsga2_optimization_summary.csv"
)


# ============================================================
# 3. SETTINGS
# ============================================================

POPULATION_SIZE = 200

GENERATIONS = 250

RANDOM_SEED = 42


# ============================================================
# 4. START
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 2 - STEP 39")
print("NSGA-II DRAINAGE OPTIMIZATION")
print("=" * 80)


if not SCENARIO_FILE.exists():

    raise FileNotFoundError(
        f"Missing Step 38 scenario file:\n{SCENARIO_FILE}"
    )


# ============================================================
# 5. LOAD SCENARIOS
# ============================================================

scenario = pd.read_csv(
    SCENARIO_FILE
)


print("\nScenario table shape:")
print(
    scenario.shape
)


required = [

    "Candidate_ID",

    "Drain_ID",

    "Upgrade_Level",

    "Upgrade_Name",

    "Segment_Length_m",

    "Added_Volume_Cost_Proxy",

    "Baseline_Risk",

    "Risk_Reduction",

    "Residual_Risk"

]


missing = [

    col
    for col in required
    if col not in scenario.columns

]


if missing:

    raise ValueError(
        f"Missing required columns:\n{missing}"
    )


# ============================================================
# 6. CHECK STRUCTURE
# ============================================================

candidate_ids = np.sort(
    scenario[
        "Candidate_ID"
    ]
    .unique()
)


n_candidates = len(
    candidate_ids
)


print("\nCandidate drains:")
print(
    n_candidates
)


if n_candidates != 500:

    print(
        f"WARNING: expected 500 candidates, "
        f"found {n_candidates}."
    )


levels_per_candidate = (

    scenario.groupby(
        "Candidate_ID"
    )[
        "Upgrade_Level"
    ]
    .nunique()

)


if not (
    levels_per_candidate
    ==
    3
).all():

    raise ValueError(
        "Each candidate must have exactly "
        "3 upgrade levels."
    )


print(
    "\nScenario structure: PASS"
)


# ============================================================
# 7. SORT SCENARIOS
# ============================================================

scenario = scenario.sort_values(

    [
        "Candidate_ID",
        "Upgrade_Level"
    ]

).reset_index(
    drop=True
)


# ============================================================
# 8. BUILD MATRICES
# ============================================================
#
# Each matrix:
#
# rows    = drain candidate
# columns = upgrade level 0,1,2
#
# ============================================================

cost_matrix = np.zeros(
    (
        n_candidates,
        3
    ),
    dtype=float
)


residual_matrix = np.zeros(
    (
        n_candidates,
        3
    ),
    dtype=float
)


reduction_matrix = np.zeros(
    (
        n_candidates,
        3
    ),
    dtype=float
)


segment_length = np.zeros(
    n_candidates,
    dtype=float
)


drain_ids = np.zeros(
    n_candidates,
    dtype=int
)


baseline_risk = np.zeros(
    n_candidates,
    dtype=float
)


for i, candidate_id in enumerate(
    candidate_ids
):

    group = scenario[

        scenario[
            "Candidate_ID"
        ]
        ==
        candidate_id

    ].sort_values(
        "Upgrade_Level"
    )


    cost_matrix[
        i,
        :
    ] = group[
        "Added_Volume_Cost_Proxy"
    ].to_numpy()


    residual_matrix[
        i,
        :
    ] = group[
        "Residual_Risk"
    ].to_numpy()


    reduction_matrix[
        i,
        :
    ] = group[
        "Risk_Reduction"
    ].to_numpy()


    segment_length[
        i
    ] = float(
        group[
            "Segment_Length_m"
        ].iloc[0]
    )


    drain_ids[
        i
    ] = int(
        group[
            "Drain_ID"
        ].iloc[0]
    )


    baseline_risk[
        i
    ] = float(
        group[
            "Baseline_Risk"
        ].iloc[0]
    )


# ============================================================
# 9. BASELINE TOTALS
# ============================================================

TOTAL_BASELINE_RISK = float(
    baseline_risk.sum()
)


TOTAL_NETWORK_LENGTH = float(
    segment_length.sum()
)


ALL_MODERATE_COST = float(
    cost_matrix[
        :,
        1
    ].sum()
)


ALL_MAJOR_COST = float(
    cost_matrix[
        :,
        2
    ].sum()
)


print("\n")
print("=" * 80)
print("BASELINE OPTIMIZATION VALUES")
print("=" * 80)


print(
    "Total baseline risk:",
    round(
        TOTAL_BASELINE_RISK,
        6
    )
)


print(
    "Candidate drainage length (km):",
    round(
        TOTAL_NETWORK_LENGTH
        /
        1000,
        3
    )
)


print(
    "All-moderate cost proxy:",
    round(
        ALL_MODERATE_COST,
        3
    )
)


print(
    "All-major cost proxy:",
    round(
        ALL_MAJOR_COST,
        3
    )
)


# ============================================================
# 10. INTEGER REPAIR
# ============================================================

class IntegerLevelRepair(
    Repair
):

    def _do(
        self,
        problem,
        X,
        **kwargs
    ):

        X = np.rint(
            X
        ).astype(
            int
        )


        X = np.clip(
            X,
            0,
            2
        )


        return X


# ============================================================
# 11. NSGA-II PROBLEM
# ============================================================

class DrainageOptimizationProblem(
    Problem
):

    def __init__(
        self
    ):

        super().__init__(

            n_var=n_candidates,

            n_obj=3,

            n_ieq_constr=0,

            xl=0,

            xu=2,

            vtype=int

        )


    def _evaluate(
        self,
        X,
        out,
        *args,
        **kwargs
    ):

        X = np.rint(
            X
        ).astype(
            int
        )


        X = np.clip(
            X,
            0,
            2
        )


        n_solutions = X.shape[
            0
        ]


        candidate_indices = np.arange(
            n_candidates
        )[None, :]


        # ----------------------------------------------------
        # F1 - TOTAL RESIDUAL FLOOD RISK
        # ----------------------------------------------------

        selected_residual = residual_matrix[

            candidate_indices,

            X

        ]


        total_residual_risk = (

            selected_residual
            .sum(
                axis=1
            )

        )


        # ----------------------------------------------------
        # F2 - TOTAL COST PROXY
        # ----------------------------------------------------

        selected_cost = cost_matrix[

            candidate_indices,

            X

        ]


        total_cost = (

            selected_cost
            .sum(
                axis=1
            )

        )


        # ----------------------------------------------------
        # F3 - TOTAL UPGRADED LENGTH
        # ----------------------------------------------------

        upgraded_mask = (

            X
            >
            0

        )


        upgraded_length = (

            upgraded_mask

            *

            segment_length[
                None,
                :
            ]

        ).sum(
            axis=1
        )


        out[
            "F"
        ] = np.column_stack([

            total_residual_risk,

            total_cost,

            upgraded_length

        ])


# ============================================================
# 12. ALGORITHM
# ============================================================

problem = DrainageOptimizationProblem()


algorithm = NSGA2(

    pop_size=POPULATION_SIZE,

    sampling=IntegerRandomSampling(),

    crossover=SBX(

        prob=0.90,

        eta=15,

        repair=IntegerLevelRepair()

    ),

    mutation=PM(

        eta=20,

        repair=IntegerLevelRepair()

    ),

    eliminate_duplicates=True

)


# ============================================================
# 13. RUN NSGA-II
# ============================================================

print("\n")
print("=" * 80)
print("RUNNING NSGA-II")
print("=" * 80)


print(
    "Population size:",
    POPULATION_SIZE
)


print(
    "Generations:",
    GENERATIONS
)


print(
    "Decision variables:",
    n_candidates
)


result = minimize(

    problem,

    algorithm,

    termination=(
        "n_gen",
        GENERATIONS
    ),

    seed=RANDOM_SEED,

    save_history=False,

    verbose=True

)


# ============================================================
# 14. RESULTS
# ============================================================

if result.X is None:

    raise RuntimeError(
        "NSGA-II returned no Pareto solutions."
    )


X = np.asarray(
    result.X
)


F = np.asarray(
    result.F
)


if X.ndim == 1:

    X = X[
        None,
        :
    ]


if F.ndim == 1:

    F = F[
        None,
        :
    ]


X = np.rint(
    X
).astype(
    int
)


n_solutions = len(
    X
)


print("\n")
print("=" * 80)
print("NSGA-II RESULT")
print("=" * 80)


print(
    "Pareto solutions:",
    n_solutions
)


# ============================================================
# 15. BUILD PARETO SUMMARY
# ============================================================

pareto_rows = []


decision_rows = []


for solution_index in range(
    n_solutions
):

    decision = X[
        solution_index
    ]


    residual_risk = float(
        F[
            solution_index,
            0
        ]
    )


    cost_proxy = float(
        F[
            solution_index,
            1
        ]
    )


    upgraded_length = float(
        F[
            solution_index,
            2
        ]
    )


    risk_reduction = (

        TOTAL_BASELINE_RISK
        -
        residual_risk

    )


    risk_reduction_pct = (

        100
        *
        risk_reduction
        /
        TOTAL_BASELINE_RISK

    )


    moderate_count = int(
        (
            decision
            ==
            1
        ).sum()
    )


    major_count = int(
        (
            decision
            ==
            2
        ).sum()
    )


    no_upgrade_count = int(
        (
            decision
            ==
            0
        ).sum()
    )


    upgraded_count = (

        moderate_count
        +
        major_count

    )


    solution_id = (
        solution_index
        +
        1
    )


    pareto_rows.append({

        "Solution_ID":
            solution_id,

        "Residual_Risk":
            residual_risk,

        "Risk_Reduction":
            risk_reduction,

        "Risk_Reduction_Percent":
            risk_reduction_pct,

        "Cost_Proxy_m3":
            cost_proxy,

        "Upgraded_Length_m":
            upgraded_length,

        "Upgraded_Length_km":
            upgraded_length
            /
            1000,

        "Upgraded_Segments":
            upgraded_count,

        "No_Upgrade_Count":
            no_upgrade_count,

        "Moderate_Count":
            moderate_count,

        "Major_Count":
            major_count

    })


    for i in range(
        n_candidates
    ):

        level = int(
            decision[
                i
            ]
        )


        decision_rows.append({

            "Solution_ID":
                solution_id,

            "Candidate_ID":
                int(
                    candidate_ids[
                        i
                    ]
                ),

            "Drain_ID":
                int(
                    drain_ids[
                        i
                    ]
                ),

            "Upgrade_Level":
                level,

            "Upgrade_Name":

                (
                    "No_Upgrade"
                    if level == 0

                    else

                    "Moderate"
                    if level == 1

                    else

                    "Major"
                )

        })


pareto = pd.DataFrame(
    pareto_rows
)


decisions = pd.DataFrame(
    decision_rows
)


# ============================================================
# 16. REMOVE EXACT DUPLICATE OBJECTIVE SOLUTIONS
# ============================================================

before_duplicates = len(
    pareto
)


keep_indices = (

    pareto

    .drop_duplicates(

        subset=[

            "Residual_Risk",

            "Cost_Proxy_m3",

            "Upgraded_Length_m"

        ]

    )

    .index

)


pareto = pareto.loc[
    keep_indices
].copy()


valid_solution_ids = set(
    pareto[
        "Solution_ID"
    ]
)


decisions = decisions[

    decisions[
        "Solution_ID"
    ]
    .isin(
        valid_solution_ids
    )

].copy()


print(
    "\nDuplicate objective solutions removed:",
    before_duplicates
    -
    len(
        pareto
    )
)


print(
    "Unique Pareto solutions:",
    len(
        pareto
    )
)


# ============================================================
# 17. SORT PARETO FRONT
# ============================================================

pareto = pareto.sort_values(

    "Cost_Proxy_m3"

).reset_index(
    drop=True
)


# ============================================================
# 18. PRINT PARETO RANGE
# ============================================================

print("\n")
print("=" * 80)
print("PARETO FRONT RANGE")
print("=" * 80)


print(
    "\nResidual risk:"
)

print(
    pareto[
        "Residual_Risk"
    ]
    .describe()
)


print(
    "\nRisk reduction percent:"
)

print(
    pareto[
        "Risk_Reduction_Percent"
    ]
    .describe()
)


print(
    "\nCost proxy:"
)

print(
    pareto[
        "Cost_Proxy_m3"
    ]
    .describe()
)


print(
    "\nUpgraded length (km):"
)

print(
    pareto[
        "Upgraded_Length_km"
    ]
    .describe()
)


print(
    "\nUpgraded segment count:"
)

print(
    pareto[
        "Upgraded_Segments"
    ]
    .describe()
)


# ============================================================
# 19. SHOW SAMPLE PARETO SOLUTIONS
# ============================================================

print("\n")
print("=" * 80)
print("SAMPLE PARETO SOLUTIONS")
print("=" * 80)


if len(
    pareto
) <= 20:

    sample = pareto.copy()

else:

    sample_indices = np.linspace(

        0,

        len(
            pareto
        )
        -
        1,

        20,

        dtype=int

    )


    sample = pareto.iloc[
        sample_indices
    ]


print(

    sample[
        [

            "Solution_ID",

            "Residual_Risk",

            "Risk_Reduction_Percent",

            "Cost_Proxy_m3",

            "Upgraded_Length_km",

            "Upgraded_Segments",

            "Moderate_Count",

            "Major_Count"

        ]
    ]

    .to_string(
        index=False
    )

)


# ============================================================
# 20. VALIDATION
# ============================================================

print("\n")
print("=" * 80)
print("OPTIMIZATION VALIDATION")
print("=" * 80)


invalid_levels = (

    ~decisions[
        "Upgrade_Level"
    ]
    .isin(
        [
            0,
            1,
            2
        ]
    )

).sum()


print(
    "Invalid decision levels:",
    invalid_levels
)


if invalid_levels > 0:

    raise RuntimeError(
        "Invalid NSGA-II decision levels found."
    )


if (

    pareto[
        "Residual_Risk"
    ]
    >
    TOTAL_BASELINE_RISK
).any():

    raise RuntimeError(
        "A solution has residual risk greater "
        "than baseline."
    )


print(
    "Decision validation: PASS"
)


# ============================================================
# 21. SAVE OUTPUTS
# ============================================================

PARETO_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


pareto.to_csv(

    PARETO_FILE,

    index=False

)


decisions.to_csv(

    DECISION_FILE,

    index=False

)


# ============================================================
# 22. OPTIMIZATION SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Metric": [

        "Candidate_Segments",

        "Decision_Variables",

        "Population_Size",

        "Generations",

        "Pareto_Solutions",

        "Baseline_Risk",

        "Min_Pareto_Residual_Risk",

        "Max_Pareto_Risk_Reduction_Percent",

        "Min_Pareto_Cost_Proxy",

        "Max_Pareto_Cost_Proxy",

        "Min_Pareto_Upgraded_Length_km",

        "Max_Pareto_Upgraded_Length_km"

    ],

    "Value": [

        n_candidates,

        n_candidates,

        POPULATION_SIZE,

        GENERATIONS,

        len(
            pareto
        ),

        TOTAL_BASELINE_RISK,

        pareto[
            "Residual_Risk"
        ].min(),

        pareto[
            "Risk_Reduction_Percent"
        ].max(),

        pareto[
            "Cost_Proxy_m3"
        ].min(),

        pareto[
            "Cost_Proxy_m3"
        ].max(),

        pareto[
            "Upgraded_Length_km"
        ].min(),

        pareto[
            "Upgraded_Length_km"
        ].max()

    ]

})


summary.to_csv(

    SUMMARY_FILE,

    index=False

)


# ============================================================
# 23. OUTPUT FILES
# ============================================================

print("\n")
print("=" * 80)
print("OUTPUT FILES")
print("=" * 80)


print(
    PARETO_FILE
)


print(
    DECISION_FILE
)


print(
    SUMMARY_FILE
)


# ============================================================
# 24. FINAL STATUS
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 2 - STEP 39 STATUS: PASS")
print("=" * 80)


print(
    "NSGA-II optimization completed."
)

print(
    "Pareto-optimal drainage upgrade plans generated."
)

print(
    "Ready for Step 40: select low-cost, balanced "
    "and maximum-risk-reduction solutions."
)

print("\n")