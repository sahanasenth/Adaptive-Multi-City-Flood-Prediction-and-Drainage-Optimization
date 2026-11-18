# ============================================================
# PHASE 2 - STEP 35
# EXTRACT FLOOD RISK FOR CHENNAI DRAINAGE SEGMENTS
# ============================================================
#
# INPUTS:
#
# 1. Chennai stormwater drainage network
# 2. Final Chennai 2015 Version-2 flood probability raster
#
# OUTPUT:
#
# A new drainage GeoPackage containing segment-level:
#
#   Segment_Length_m
#   Mean_Flood_Probability
#   Max_Flood_Probability
#   Median_Flood_Probability
#   High_Risk_Length_m
#   Very_High_Risk_Length_m
#   Flood_Exposure_Percent
#   Very_High_Exposure_Percent
#   Priority_Score
#   Risk_Class
#
# This dataset will later be used by NSGA-II.
#
# ============================================================


from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from shapely.geometry import (
    LineString,
    MultiLineString
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(
    r"D:\Adaptive_Multi_City_Flood_Prediction"
)


DRAINAGE_FILE = (
    ROOT
    / "phase2_drainage"
    / "Chennai_SWD_Network.gpkg"
)


PROBABILITY_RASTER = (
    ROOT
    / "results"
    / "final_maps"
    / "Chennai_2015_Flood_Probability_V2.tif"
)


OUTPUT_GPKG = (
    ROOT
    / "phase2_drainage"
    / "Chennai_SWD_Risk_Attributed.gpkg"
)


OUTPUT_CSV = (
    ROOT
    / "phase2_drainage"
    / "Chennai_SWD_Risk_Attributed.csv"
)


SUMMARY_CSV = (
    ROOT
    / "results"
    / "35_drainage_flood_risk_summary.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

# Sample points along drainage lines every 25 m.

SAMPLE_INTERVAL_M = 25.0


# Flood-risk thresholds used in the Phase-1 probability map.

HIGH_RISK_THRESHOLD = 0.60

VERY_HIGH_RISK_THRESHOLD = 0.80


# ============================================================
# 3. START
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 2 - STEP 35")
print("DRAINAGE SEGMENT FLOOD-RISK EXTRACTION")
print("=" * 80)


# ============================================================
# 4. CHECK FILES
# ============================================================

for path in [
    DRAINAGE_FILE,
    PROBABILITY_RASTER
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ============================================================
# 5. LOAD DRAINAGE NETWORK
# ============================================================

drains = gpd.read_file(
    DRAINAGE_FILE
)


print("\nDrainage network shape:")
print(
    drains.shape
)


print("\nDrainage CRS:")
print(
    drains.crs
)


print("\nGeometry types:")

print(
    drains.geometry.geom_type.value_counts()
)


# ============================================================
# 6. BASIC GEOMETRY CLEANING
# ============================================================

original_count = len(
    drains
)


drains = drains[
    drains.geometry.notna()
].copy()


drains = drains[
    ~drains.geometry.is_empty
].copy()


# Fix invalid geometries where possible.

invalid_count = int(
    (
        ~drains.geometry.is_valid
    ).sum()
)


print(
    "\nInvalid geometries before cleaning:",
    invalid_count
)


if invalid_count > 0:

    drains.geometry = (
        drains.geometry
        .make_valid()
    )


print(
    "\nDrainage features after geometry cleaning:",
    len(drains)
)


if len(drains) == 0:

    raise RuntimeError(
        "No valid drainage geometries remain."
    )


# ============================================================
# 7. OPEN PROBABILITY RASTER
# ============================================================

with rasterio.open(
    PROBABILITY_RASTER
) as src:

    raster_crs = src.crs

    raster_nodata = src.nodata

    raster_bounds = src.bounds

    raster_resolution = src.res


print("\nFlood probability raster:")

print(
    "CRS:",
    raster_crs
)

print(
    "Resolution:",
    raster_resolution
)

print(
    "Bounds:",
    raster_bounds
)

print(
    "NoData:",
    raster_nodata
)


# ============================================================
# 8. CRS VALIDATION
# ============================================================

if drains.crs is None:

    raise ValueError(
        "Drainage network has no CRS."
    )


if str(
    drains.crs
) != str(
    raster_crs
):

    print(
        "\nCRS mismatch detected."
    )

    print(
        "Reprojecting drainage network to raster CRS..."
    )

    drains = drains.to_crs(
        raster_crs
    )


print(
    "\nWorking CRS:",
    drains.crs
)


# ============================================================
# 9. CREATE UNIQUE DRAIN SEGMENT ID
# ============================================================

drains = drains.reset_index(
    drop=True
)


drains[
    "Drain_ID"
] = np.arange(
    1,
    len(drains) + 1
)


# ============================================================
# 10. CALCULATE SEGMENT LENGTH
# ============================================================

drains[
    "Segment_Length_m"
] = (
    drains.geometry.length
)


print("\n")
print("=" * 80)
print("DRAINAGE LENGTH STATISTICS")
print("=" * 80)


print(
    drains[
        "Segment_Length_m"
    ].describe()
)


print(
    "\nTotal drainage length (km):",
    round(
        drains[
            "Segment_Length_m"
        ].sum()
        /
        1000,
        3
    )
)


# ============================================================
# 11. FUNCTION TO SAMPLE POINTS ALONG A LINE
# ============================================================

def interpolate_line_points(
    geometry,
    interval
):

    """
    Generate regularly spaced sample points
    along LineString or MultiLineString geometry.
    """

    if geometry is None:

        return []


    if geometry.is_empty:

        return []


    points = []


    # --------------------------------------------------------
    # LineString
    # --------------------------------------------------------

    if isinstance(
        geometry,
        LineString
    ):

        length = geometry.length


        if length == 0:

            return [
                geometry.interpolate(
                    0
                )
            ]


        distances = np.arange(
            0,
            length,
            interval
        )


        # Always include end point.

        distances = np.append(
            distances,
            length
        )


        for distance in distances:

            points.append(
                geometry.interpolate(
                    float(
                        distance
                    )
                )
            )


    # --------------------------------------------------------
    # MultiLineString
    # --------------------------------------------------------

    elif isinstance(
        geometry,
        MultiLineString
    ):

        for part in geometry.geoms:

            part_points = (
                interpolate_line_points(
                    part,
                    interval
                )
            )

            points.extend(
                part_points
            )


    return points


# ============================================================
# 12. RASTER SAMPLING FUNCTION
# ============================================================

def sample_raster_values(
    points,
    raster
):

    if len(
        points
    ) == 0:

        return np.array(
            [],
            dtype=float
        )


    coordinates = [

        (
            point.x,
            point.y
        )

        for point in points

    ]


    sampled_values = []


    for value in raster.sample(
        coordinates
    ):

        raster_value = value[
            0
        ]


        if raster.nodata is not None:

            if np.isclose(
                raster_value,
                raster.nodata
            ):

                sampled_values.append(
                    np.nan
                )

                continue


        if not np.isfinite(
            raster_value
        ):

            sampled_values.append(
                np.nan
            )

            continue


        sampled_values.append(
            float(
                raster_value
            )
        )


    return np.asarray(
        sampled_values,
        dtype=float
    )


# ============================================================
# 13. EXTRACT RISK FOR EVERY DRAIN SEGMENT
# ============================================================

print("\n")
print("=" * 80)
print("EXTRACTING FLOOD RISK ALONG DRAINAGE SEGMENTS")
print("=" * 80)


mean_probability = []

max_probability = []

median_probability = []

sample_count_list = []

valid_sample_count_list = []

high_risk_length = []

very_high_risk_length = []

flood_exposure_percent = []

very_high_exposure_percent = []


with rasterio.open(
    PROBABILITY_RASTER
) as raster:


    for idx, row in drains.iterrows():

        geometry = row.geometry

        segment_length = float(
            row[
                "Segment_Length_m"
            ]
        )


        # ----------------------------------------------------
        # Generate sample points
        # ----------------------------------------------------

        points = interpolate_line_points(

            geometry,

            SAMPLE_INTERVAL_M

        )


        sample_count = len(
            points
        )


        sample_count_list.append(
            sample_count
        )


        # ----------------------------------------------------
        # Sample probability raster
        # ----------------------------------------------------

        values = sample_raster_values(

            points,

            raster

        )


        valid_values = values[
            np.isfinite(
                values
            )
        ]


        valid_count = len(
            valid_values
        )


        valid_sample_count_list.append(
            valid_count
        )


        # ----------------------------------------------------
        # No overlap with raster
        # ----------------------------------------------------

        if valid_count == 0:

            mean_probability.append(
                np.nan
            )

            max_probability.append(
                np.nan
            )

            median_probability.append(
                np.nan
            )

            high_risk_length.append(
                0.0
            )

            very_high_risk_length.append(
                0.0
            )

            flood_exposure_percent.append(
                0.0
            )

            very_high_exposure_percent.append(
                0.0
            )

            continue


        # ----------------------------------------------------
        # Probability statistics
        # ----------------------------------------------------

        mean_prob = float(
            np.mean(
                valid_values
            )
        )


        max_prob = float(
            np.max(
                valid_values
            )
        )


        median_prob = float(
            np.median(
                valid_values
            )
        )


        mean_probability.append(
            mean_prob
        )


        max_probability.append(
            max_prob
        )


        median_probability.append(
            median_prob
        )


        # ----------------------------------------------------
        # Risk exposure
        # ----------------------------------------------------

        high_fraction = float(

            np.mean(

                valid_values
                >=
                HIGH_RISK_THRESHOLD

            )

        )


        very_high_fraction = float(

            np.mean(

                valid_values
                >=
                VERY_HIGH_RISK_THRESHOLD

            )

        )


        # Estimate exposed segment length.

        high_length = (

            segment_length
            *
            high_fraction

        )


        very_high_length = (

            segment_length
            *
            very_high_fraction

        )


        high_risk_length.append(
            high_length
        )


        very_high_risk_length.append(
            very_high_length
        )


        flood_exposure_percent.append(

            high_fraction
            *
            100.0

        )


        very_high_exposure_percent.append(

            very_high_fraction
            *
            100.0

        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            idx + 1
        ) % 1000 == 0:

            print(
                f"Processed "
                f"{idx + 1} / {len(drains)} "
                f"drain segments"
            )


# ============================================================
# 14. ADD RISK COLUMNS
# ============================================================

drains[
    "Sample_Count"
] = sample_count_list


drains[
    "Valid_Risk_Samples"
] = valid_sample_count_list


drains[
    "Mean_Flood_Prob"
] = mean_probability


drains[
    "Max_Flood_Prob"
] = max_probability


drains[
    "Median_Flood_Prob"
] = median_probability


drains[
    "High_Risk_Length_m"
] = high_risk_length


drains[
    "Very_High_Risk_Length_m"
] = very_high_risk_length


drains[
    "Flood_Exposure_Pct"
] = flood_exposure_percent


drains[
    "Very_High_Exposure_Pct"
] = very_high_exposure_percent


# ============================================================
# 15. OVERLAP FLAG
# ============================================================

drains[
    "Raster_Overlap"
] = (

    drains[
        "Valid_Risk_Samples"
    ]
    >
    0

).astype(
    int
)


# ============================================================
# 16. PRIORITY SCORE
# ============================================================
#
# Priority score is ONLY a preliminary risk ranking.
#
# It is not the NSGA-II objective yet.
#
# Components:
#
# 45% Mean flood probability
# 25% Maximum flood probability
# 30% High-risk exposure fraction
#
# Score ranges approximately 0-1.
#
# ============================================================

exposure_fraction = (

    drains[
        "Flood_Exposure_Pct"
    ]
    /
    100.0

)


drains[
    "Priority_Score"
] = (

    0.45
    *
    drains[
        "Mean_Flood_Prob"
    ]

    +

    0.25
    *
    drains[
        "Max_Flood_Prob"
    ]

    +

    0.30
    *
    exposure_fraction

)


# ============================================================
# 17. ASSIGN DRAINAGE RISK CLASS
# ============================================================

def assign_risk_class(
    score
):

    if pd.isna(
        score
    ):

        return "Outside_Raster"


    if score < 0.20:

        return "Very_Low"


    elif score < 0.40:

        return "Low"


    elif score < 0.60:

        return "Moderate"


    elif score < 0.80:

        return "High"


    else:

        return "Very_High"


drains[
    "Risk_Class"
] = drains[
    "Priority_Score"
].apply(
    assign_risk_class
)


# ============================================================
# 18. VALIDATION
# ============================================================

print("\n")
print("=" * 80)
print("STEP 35 RISK EXTRACTION SUMMARY")
print("=" * 80)


print(
    "\nTotal drainage segments:",
    len(
        drains
    )
)


print(
    "\nSegments overlapping flood raster:",
    int(
        drains[
            "Raster_Overlap"
        ].sum()
    )
)


print(
    "\nSegments outside flood raster:",
    int(
        (
            drains[
                "Raster_Overlap"
            ]
            ==
            0
        ).sum()
    )
)


valid = drains[
    drains[
        "Raster_Overlap"
    ]
    ==
    1
].copy()


print("\nMean flood probability statistics:")

print(
    valid[
        "Mean_Flood_Prob"
    ]
    .describe()
)


print("\nMaximum flood probability statistics:")

print(
    valid[
        "Max_Flood_Prob"
    ]
    .describe()
)


print("\nFlood exposure percentage statistics:")

print(
    valid[
        "Flood_Exposure_Pct"
    ]
    .describe()
)


print("\nPriority score statistics:")

print(
    valid[
        "Priority_Score"
    ]
    .describe()
)


print("\nRisk class distribution:")

print(
    drains[
        "Risk_Class"
    ]
    .value_counts()
)


# ============================================================
# 19. HIGH-RISK SEGMENTS
# ============================================================

high_priority = drains[

    drains[
        "Risk_Class"
    ].isin(
        [
            "High",
            "Very_High"
        ]
    )

].copy()


print(
    "\nHigh + Very High risk segments:",
    len(
        high_priority
    )
)


print(
    "\nHigh + Very High risk drainage length (km):",
    round(
        high_priority[
            "Segment_Length_m"
        ].sum()
        /
        1000,
        3
    )
)


# ============================================================
# 20. TOP 20 CRITICAL SEGMENTS
# ============================================================

top20 = (

    valid

    .sort_values(
        "Priority_Score",
        ascending=False
    )

    .head(
        20
    )

)


print("\n")
print("=" * 80)
print("TOP 20 HIGHEST-RISK DRAINAGE SEGMENTS")
print("=" * 80)


print(

    top20[
        [

            "Drain_ID",

            "Segment_Length_m",

            "Mean_Flood_Prob",

            "Max_Flood_Prob",

            "Flood_Exposure_Pct",

            "Very_High_Exposure_Pct",

            "Priority_Score",

            "Risk_Class"

        ]
    ]

    .to_string(
        index=False
    )

)


# ============================================================
# 21. SAVE GEOPACKAGE
# ============================================================

if OUTPUT_GPKG.exists():

    OUTPUT_GPKG.unlink()


drains.to_file(

    OUTPUT_GPKG,

    driver="GPKG",

    layer="Chennai_SWD_Risk"

)


# ============================================================
# 22. SAVE CSV
# ============================================================

csv_output = drains.drop(
    columns="geometry"
)


csv_output.to_csv(

    OUTPUT_CSV,

    index=False

)


# ============================================================
# 23. SAVE SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Metric": [

        "Total_Drainage_Segments",

        "Raster_Overlap_Segments",

        "Outside_Raster_Segments",

        "Total_Drainage_Length_km",

        "Mean_Segment_Flood_Probability",

        "Mean_Maximum_Flood_Probability",

        "Mean_Flood_Exposure_Percent",

        "High_And_Very_High_Risk_Segments",

        "High_And_Very_High_Length_km"

    ],

    "Value": [

        len(
            drains
        ),

        int(
            drains[
                "Raster_Overlap"
            ].sum()
        ),

        int(
            (
                drains[
                    "Raster_Overlap"
                ]
                ==
                0
            ).sum()
        ),

        drains[
            "Segment_Length_m"
        ].sum()
        /
        1000,

        valid[
            "Mean_Flood_Prob"
        ].mean(),

        valid[
            "Max_Flood_Prob"
        ].mean(),

        valid[
            "Flood_Exposure_Pct"
        ].mean(),

        len(
            high_priority
        ),

        high_priority[
            "Segment_Length_m"
        ].sum()
        /
        1000

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
# 24. OUTPUT FILES
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
# 25. FINAL STATUS
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 2 - STEP 35 STATUS: PASS")
print("=" * 80)


print(
    "Flood risk successfully attributed "
    "to Chennai stormwater drainage segments."
)

print(
    "Risk-attributed drainage network is ready "
    "for critical-segment analysis and NSGA-II."
)

print("\n")