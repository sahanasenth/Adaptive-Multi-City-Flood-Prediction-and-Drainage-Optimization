# ============================================================
# STEP 43 - GENERIC FUTURE-CITY ADAPTIVE STRATEGY SELECTOR
# ============================================================
#
# PURPOSE:
# Compare three strategies for a new target city:
#
# 1. SOURCE TRANSFER
# 2. CUMULATIVE ADAPTATION
# 3. LOCAL TARGET-CITY MODEL
#
# Strategy selection is performed ONLY on the adaptation set
# using spatial GroupKFold.
#
# The untouched spatial holdout is used once at the end.
#
# ============================================================

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Adaptive_Multi_City_Flood_Prediction"
)

SOURCE_DATA = (
    PROJECT_ROOT
    / "data"
    / "03_model_ready"
    / "Combined_Model_Ready.csv"
)

ADAPTATION_DATA = (
    PROJECT_ROOT
    / "data"
    / "05_testing_spatial"
    / "Chennai_2015_Adaptation_Set.csv"
)

HOLDOUT_DATA = (
    PROJECT_ROOT
    / "data"
    / "05_testing_spatial"
    / "Chennai_2015_Spatial_Holdout.csv"
)

SOURCE_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "final"
    / "Final_Source_Random_Forest.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "43_strategy_selector"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "adaptive"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. FEATURE DEFINITIONS
# ============================================================

NUMERIC_FEATURES = [
    "DEM_1",
    "NDVI_1",
    "POP_1",
    "RAINFALL_1",
    "RIVER DISTANCE_1",
    "SLOPE_1"
]

CATEGORICAL_FEATURES = [
    "LULC_GROUP",
    "SOIL_1"
]

FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

TARGET = "Flood"


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n")
print("=" * 90)
print("STEP 43 - GENERIC FUTURE-CITY ADAPTIVE STRATEGY SELECTOR")
print("=" * 90)

source = pd.read_csv(
    SOURCE_DATA
)

adapt = pd.read_csv(
    ADAPTATION_DATA
)

holdout = pd.read_csv(
    HOLDOUT_DATA
)


print("\nSOURCE:")
print(source.shape)

print("\nADAPTATION:")
print(adapt.shape)

print("\nHOLDOUT:")
print(holdout.shape)


# ============================================================
# 4. VALIDATION
# ============================================================

required_columns = (
    FEATURES
    + [TARGET]
)

for name, df in [
    ("SOURCE", source),
    ("ADAPTATION", adapt),
    ("HOLDOUT", holdout)
]:

    missing = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{name} missing columns: {missing}"
        )


if "Spatial_Block" not in adapt.columns:

    raise ValueError(
        "Spatial_Block missing in adaptation set."
    )


print("\nValidation: PASS")


# ============================================================
# 5. MODEL BUILDER
# ============================================================

def build_rf():

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numerical",
                "passthrough",
                NUMERIC_FEATURES
            ),

            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES
            )

        ],

        remainder="drop"

    )


    rf = RandomForestClassifier(

        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        criterion="gini",
        random_state=42,
        n_jobs=-1

    )


    return Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "rf",
            rf
        )

    ])


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate(model, X, y):

    pred = model.predict(
        X
    )

    prob = model.predict_proba(
        X
    )[:, 1]


    return {

        "Accuracy":
            accuracy_score(
                y,
                pred
            ),

        "Precision":
            precision_score(
                y,
                pred,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y,
                pred,
                zero_division=0
            ),

        "F1":
            f1_score(
                y,
                pred,
                zero_division=0
            ),

        "ROC_AUC":
            roc_auc_score(
                y,
                prob
            )

    }


# ============================================================
# 7. PREPARE DATA
# ============================================================

X_adapt = adapt[
    FEATURES
]

y_adapt = adapt[
    TARGET
].astype(int)

groups = adapt[
    "Spatial_Block"
].values


group_cv = GroupKFold(
    n_splits=5
)


# ============================================================
# 8. STRATEGY A - SOURCE TRANSFER
# ============================================================

print("\n")
print("=" * 90)
print("STRATEGY A - SOURCE TRANSFER")
print("=" * 90)


source_model = joblib.load(
    SOURCE_MODEL_FILE
)


source_cv_results = []


for fold, (_, val_idx) in enumerate(

    group_cv.split(
        X_adapt,
        y_adapt,
        groups
    ),

    start=1

):

    X_val = X_adapt.iloc[
        val_idx
    ]

    y_val = y_adapt.iloc[
        val_idx
    ]


    metrics = evaluate(

        source_model,

        X_val,

        y_val

    )


    source_cv_results.append(
        metrics
    )


    print(
        f"Fold {fold}: "
        f"Acc={metrics['Accuracy']:.4f} | "
        f"F1={metrics['F1']:.4f} | "
        f"AUC={metrics['ROC_AUC']:.4f}"
    )


# ============================================================
# 9. STRATEGY B - LOCAL TARGET MODEL
# ============================================================

print("\n")
print("=" * 90)
print("STRATEGY B - LOCAL TARGET-CITY MODEL")
print("=" * 90)


local_cv_results = []


for fold, (
    train_idx,
    val_idx

) in enumerate(

    group_cv.split(
        X_adapt,
        y_adapt,
        groups
    ),

    start=1

):

    train = adapt.iloc[
        train_idx
    ]

    val = adapt.iloc[
        val_idx
    ]


    model = build_rf()


    model.fit(

        train[
            FEATURES
        ],

        train[
            TARGET
        ].astype(int)

    )


    metrics = evaluate(

        model,

        val[
            FEATURES
        ],

        val[
            TARGET
        ].astype(int)

    )


    local_cv_results.append(
        metrics
    )


    print(
        f"Fold {fold}: "
        f"Acc={metrics['Accuracy']:.4f} | "
        f"F1={metrics['F1']:.4f} | "
        f"AUC={metrics['ROC_AUC']:.4f}"
    )


# ============================================================
# 10. STRATEGY C - CUMULATIVE ADAPTATION
# ============================================================

print("\n")
print("=" * 90)
print("STRATEGY C - CUMULATIVE ADAPTATION")
print("=" * 90)


TARGET_WEIGHT = 1.5

cumulative_cv_results = []


for fold, (
    train_idx,
    val_idx

) in enumerate(

    group_cv.split(
        X_adapt,
        y_adapt,
        groups
    ),

    start=1

):

    target_train = adapt.iloc[
        train_idx
    ]

    target_val = adapt.iloc[
        val_idx
    ]


    combined = pd.concat(

        [

            source[
                FEATURES
                + [TARGET]
            ],

            target_train[
                FEATURES
                + [TARGET]
            ]

        ],

        ignore_index=True

    )


    sample_weights = np.concatenate(

        [

            np.ones(
                len(source)
            ),

            np.full(
                len(target_train),
                TARGET_WEIGHT
            )

        ]

    )


    model = build_rf()


    model.fit(

        combined[
            FEATURES
        ],

        combined[
            TARGET
        ].astype(int),

        rf__sample_weight=
            sample_weights

    )


    metrics = evaluate(

        model,

        target_val[
            FEATURES
        ],

        target_val[
            TARGET
        ].astype(int)

    )


    cumulative_cv_results.append(
        metrics
    )


    print(
        f"Fold {fold}: "
        f"Acc={metrics['Accuracy']:.4f} | "
        f"F1={metrics['F1']:.4f} | "
        f"AUC={metrics['ROC_AUC']:.4f}"
    )


# ============================================================
# 11. SUMMARIZE STRATEGIES
# ============================================================

def summarize(
    name,
    results
):

    df = pd.DataFrame(
        results
    )


    return {

        "Strategy":
            name,

        "Mean_Accuracy":
            df[
                "Accuracy"
            ].mean(),

        "Mean_Precision":
            df[
                "Precision"
            ].mean(),

        "Mean_Recall":
            df[
                "Recall"
            ].mean(),

        "Mean_F1":
            df[
                "F1"
            ].mean(),

        "Mean_ROC_AUC":
            df[
                "ROC_AUC"
            ].mean(),

        "F1_STD":
            df[
                "F1"
            ].std()

    }


strategy_table = pd.DataFrame([

    summarize(
        "Source_Transfer",
        source_cv_results
    ),

    summarize(
        "Local_Target_RF",
        local_cv_results
    ),

    summarize(
        "Cumulative_Adaptive_RF",
        cumulative_cv_results
    )

])


strategy_table = (

    strategy_table
    .sort_values(

        by=[
            "Mean_F1",
            "Mean_ROC_AUC"
        ],

        ascending=False

    )
    .reset_index(
        drop=True
    )

)


print("\n")
print("=" * 90)
print("STRATEGY COMPARISON")
print("=" * 90)

print(
    strategy_table.to_string(
        index=False
    )
)


BEST_STRATEGY = (
    strategy_table.iloc[
        0
    ][
        "Strategy"
    ]
)


print("\nBEST STRATEGY:")
print(
    BEST_STRATEGY
)


strategy_table.to_csv(

    OUTPUT_DIR
    / "strategy_comparison.csv",

    index=False

)


# ============================================================
# 12. TRAIN FINAL SELECTED MODEL
# ============================================================

print("\n")
print("=" * 90)
print("TRAIN FINAL SELECTED MODEL")
print("=" * 90)


if BEST_STRATEGY == "Source_Transfer":

    final_model = source_model


elif BEST_STRATEGY == "Local_Target_RF":

    final_model = build_rf()

    final_model.fit(

        adapt[
            FEATURES
        ],

        adapt[
            TARGET
        ].astype(int)

    )


elif BEST_STRATEGY == "Cumulative_Adaptive_RF":

    combined = pd.concat(

        [

            source[
                FEATURES
                + [TARGET]
            ],

            adapt[
                FEATURES
                + [TARGET]
            ]

        ],

        ignore_index=True

    )


    weights = np.concatenate(

        [

            np.ones(
                len(source)
            ),

            np.full(
                len(adapt),
                TARGET_WEIGHT
            )

        ]

    )


    final_model = build_rf()


    final_model.fit(

        combined[
            FEATURES
        ],

        combined[
            TARGET
        ].astype(int),

        rf__sample_weight=
            weights

    )


else:

    raise ValueError(
        "Unknown strategy."
    )


# ============================================================
# 13. FINAL UNTOUCHED HOLDOUT TEST
# ============================================================

print("\n")
print("=" * 90)
print("FINAL UNTOUCHED SPATIAL HOLDOUT")
print("=" * 90)


X_holdout = holdout[
    FEATURES
]

y_holdout = holdout[
    TARGET
].astype(int)


final_metrics = evaluate(

    final_model,

    X_holdout,

    y_holdout

)


for key, value in final_metrics.items():

    print(
        f"{key:12s}: {value:.6f}"
    )


# ============================================================
# 14. SAVE MODEL
# ============================================================

MODEL_FILE = (

    MODEL_DIR
    / "Chennai_2015_Selected_Adaptive_Strategy_RF.joblib"

)


joblib.dump(

    final_model,

    MODEL_FILE

)


# ============================================================
# 15. SAVE FINAL RESULTS
# ============================================================

final_result = pd.DataFrame([

    {

        "Selected_Strategy":
            BEST_STRATEGY,

        **final_metrics

    }

])


final_result.to_csv(

    OUTPUT_DIR
    / "final_selected_strategy_holdout.csv",

    index=False

)


# ============================================================
# 16. FINAL STATUS
# ============================================================

print("\n")
print("=" * 90)
print("STEP 43 STATUS: COMPLETED")
print("=" * 90)


print(
    "\nSelected adaptive strategy:"
)

print(
    BEST_STRATEGY
)


print(
    "\nSaved model:"
)

print(
    MODEL_FILE
)


print(
    "\nStrategy comparison:"
)

print(
    OUTPUT_DIR
    / "strategy_comparison.csv"
)


print(
    "\nFinal holdout result:"
)

print(
    OUTPUT_DIR
    / "final_selected_strategy_holdout.csv"
)


print("\nIMPORTANT:")

print(
    "Strategy selection used only the 3864-point "
    "adaptation set."
)

print(
    "The 975-point spatial holdout was used only "
    "for final evaluation."
)

print("=" * 90)