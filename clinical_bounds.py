"""
Clinical bounds for CTG features used in the Fetal Health Classifier.

Each entry provides:
    hard_min, hard_max : physiologically plausible input limits (enforce in UI).
    ref_low, ref_high  : clinical reference range. None where the feature is
                         signal-processing derived and has no published
                         reference range.
    default            : a clinically sensible starting value (NOT the dataset
                         median, which is contaminated by Suspect/Pathological
                         cases).
    unit               : display unit.
    note               : short clinical context shown to the user.

Sources for the directly clinical features:
    FIGO 2015 Intrapartum CTG guidelines
    NICE CG190 (2017) Intrapartum care
    ACOG Practice Bulletin 106 (Intrapartum Fetal Heart Rate Monitoring)
    Dawes/Redman criteria for computerised antenatal CTG

The "*_per_second" features and most histogram features are SisPorto-style
signal-processing outputs; their bounds are derived from physiological
plausibility and the expected support of those algorithms, not from clinical
reference ranges.

These bounds should be reviewed by an obstetric clinician before any
non-educational deployment.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class Bound(TypedDict):
    hard_min: float
    hard_max: float
    ref_low: Optional[float]
    ref_high: Optional[float]
    default: float
    unit: str
    note: str


CLINICAL_BOUNDS: dict[str, Bound] = {
    "baseline value": {
        "hard_min": 50, "hard_max": 220,
        "ref_low": 110, "ref_high": 160,
        "default": 140, "unit": "bpm",
        "note": "FIGO normal 110-160 bpm. <110 bradycardia, >160 tachycardia.",
    },
    "accelerations": {
        "hard_min": 0, "hard_max": 0.05,
        "ref_low": 0.001, "ref_high": None,
        "default": 0.003, "unit": "per second",
        "note": "Acceleration = >=15 bpm rise lasting >=15 s. Presence is reassuring.",
    },
    "fetal_movement": {
        "hard_min": 0, "hard_max": 0.5,
        "ref_low": None, "ref_high": None,
        "default": 0.0, "unit": "per second",
        "note": "Detected fetal movements per second (signal-processing derived).",
    },
    "uterine_contractions": {
        "hard_min": 0, "hard_max": 0.02,
        "ref_low": 0.003, "ref_high": 0.0083,
        "default": 0.004, "unit": "per second",
        "note": "Normal labour 3-5 per 10 min. Tachysystole >5 per 10 min.",
    },
    "light_decelerations": {
        "hard_min": 0, "hard_max": 0.02,
        "ref_low": None, "ref_high": 0.003,
        "default": 0.0, "unit": "per second",
        "note": "Early/variable decelerations. Repetitive ones can be concerning.",
    },
    "severe_decelerations": {
        "hard_min": 0, "hard_max": 0.005,
        "ref_low": None, "ref_high": 0.0,
        "default": 0.0, "unit": "per second",
        "note": "Rare. Any sustained presence is pathological per FIGO.",
    },
    "prolongued_decelerations": {
        "hard_min": 0, "hard_max": 0.005,
        "ref_low": None, "ref_high": 0.0,
        "default": 0.0, "unit": "per second",
        "note": "Deceleration >=2 min. >3 min requires escalation per FIGO.",
    },
    "abnormal_short_term_variability": {
        "hard_min": 0, "hard_max": 100,
        "ref_low": None, "ref_high": 30,
        "default": 25, "unit": "% of trace",
        "note": "% of time with abnormal beat-to-beat variability.",
    },
    "mean_value_of_short_term_variability": {
        "hard_min": 0, "hard_max": 10,
        "ref_low": 3, "ref_high": 8,
        "default": 4.5, "unit": "ms",
        "note": "Computerised STV. <3 ms is concerning (Dawes/Redman).",
    },
    "percentage_of_time_with_abnormal_long_term_variability": {
        "hard_min": 0, "hard_max": 100,
        "ref_low": None, "ref_high": 20,
        "default": 5, "unit": "% of trace",
        "note": "% of time with abnormal long-term variability.",
    },
    "mean_value_of_long_term_variability": {
        "hard_min": 0, "hard_max": 50,
        "ref_low": 5, "ref_high": 25,
        "default": 10, "unit": "bpm",
        "note": "FIGO: normal variability 5-25 bpm. <5 reduced, >25 saltatory.",
    },
    "histogram_width": {
        "hard_min": 0, "hard_max": 200,
        "ref_low": 30, "ref_high": 90,
        "default": 70, "unit": "bpm",
        "note": "Spread of FHR distribution. Narrow width = reduced variability.",
    },
    "histogram_min": {
        "hard_min": 50, "hard_max": 200,
        "ref_low": 100, "ref_high": 140,
        "default": 110, "unit": "bpm",
        "note": "Lowest FHR bin observed in the trace.",
    },
    "histogram_max": {
        "hard_min": 100, "hard_max": 240,
        "ref_low": 140, "ref_high": 180,
        "default": 160, "unit": "bpm",
        "note": "Highest FHR bin observed in the trace.",
    },
    "histogram_number_of_peaks": {
        "hard_min": 0, "hard_max": 30,
        "ref_low": 2, "ref_high": 8,
        "default": 4, "unit": "count",
        "note": "Local maxima in the FHR histogram.",
    },
    "histogram_number_of_zeroes": {
        "hard_min": 0, "hard_max": 15,
        "ref_low": None, "ref_high": 2,
        "default": 0, "unit": "count",
        "note": "Empty bins in the FHR histogram. Many zeroes => sparse trace.",
    },
    "histogram_mode": {
        "hard_min": 50, "hard_max": 220,
        "ref_low": 110, "ref_high": 160,
        "default": 140, "unit": "bpm",
        "note": "Most common FHR value in the trace.",
    },
    "histogram_mean": {
        "hard_min": 50, "hard_max": 220,
        "ref_low": 110, "ref_high": 160,
        "default": 140, "unit": "bpm",
        "note": "Mean FHR across the histogram.",
    },
    "histogram_median": {
        "hard_min": 50, "hard_max": 220,
        "ref_low": 110, "ref_high": 160,
        "default": 140, "unit": "bpm",
        "note": "Median FHR across the histogram.",
    },
    "histogram_variance": {
        "hard_min": 0, "hard_max": 300,
        "ref_low": 4, "ref_high": 50,
        "default": 15, "unit": "bpm^2",
        "note": "Variance of FHR distribution. Very low => loss of variability.",
    },
    "histogram_tendency": {
        "hard_min": -1, "hard_max": 1,
        "ref_low": 0, "ref_high": 0,
        "default": 0, "unit": "",
        "note": "Histogram skew: -1 left, 0 symmetric, 1 right. Use selectbox.",
    },
}

# Features that should be entered as integers.
INTEGER_FEATURES: set[str] = {
    "histogram_width",
    "histogram_min",
    "histogram_max",
    "histogram_number_of_peaks",
    "histogram_number_of_zeroes",
    "histogram_mode",
    "histogram_mean",
    "histogram_median",
}

# Features with a fixed discrete set of values.
DISCRETE_FEATURES: dict[str, list[int]] = {
    "histogram_tendency": [-1, 0, 1],
}


def is_outside_reference(feature: str, value: float) -> Optional[str]:
    """Return a short warning string if value is outside the clinical reference
    range, else None. Returns None when the feature has no reference range."""
    bound = CLINICAL_BOUNDS.get(feature)
    if bound is None:
        return None
    low, high = bound["ref_low"], bound["ref_high"]
    if low is not None and value < low:
        return f"Below reference range ({bound['unit']} >= {low})."
    if high is not None and value > high:
        return f"Above reference range ({bound['unit']} <= {high})."
    return None


# ---------------------------------------------------------------------------
# Usage sketch for app.py - replace numeric_input_for_feature with something
# like the following:
#
#   from clinical_bounds import (
#       CLINICAL_BOUNDS, DISCRETE_FEATURES, INTEGER_FEATURES, is_outside_reference,
#   )
#
#   def clinical_input(feature: str) -> float:
#       bound = CLINICAL_BOUNDS[feature]
#       description = FEATURE_DESCRIPTIONS.get(feature, feature)
#       if feature in DISCRETE_FEATURES:
#           value = st.selectbox(description, DISCRETE_FEATURES[feature],
#                                index=DISCRETE_FEATURES[feature].index(int(bound["default"])))
#           return float(value)
#       step = 1.0 if feature in INTEGER_FEATURES else 0.001
#       value = st.number_input(
#           description,
#           min_value=float(bound["hard_min"]),
#           max_value=float(bound["hard_max"]),
#           value=float(bound["default"]),
#           step=step,
#           help=bound["note"],
#       )
#       warning = is_outside_reference(feature, value)
#       if warning:
#           st.warning(warning)
#       return float(value)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Example CTG presets for each class. These are hand-crafted clinically-realistic
# values used by the "Load example" buttons in the assessment form. They let the
# user test that the model correctly classifies non-normal inputs without
# manually editing all 21 fields.
# ---------------------------------------------------------------------------

EXAMPLE_PRESETS: dict[str, dict[str, float]] = {
    "Normal": {
        "baseline value": 140,
        "accelerations": 0.005,
        "fetal_movement": 0.001,
        "uterine_contractions": 0.005,
        "light_decelerations": 0.0,
        "severe_decelerations": 0.0,
        "prolongued_decelerations": 0.0,
        "abnormal_short_term_variability": 25,
        "mean_value_of_short_term_variability": 4.5,
        "percentage_of_time_with_abnormal_long_term_variability": 5,
        "mean_value_of_long_term_variability": 12,
        "histogram_width": 70,
        "histogram_min": 110,
        "histogram_max": 160,
        "histogram_number_of_peaks": 4,
        "histogram_number_of_zeroes": 0,
        "histogram_mode": 140,
        "histogram_mean": 138,
        "histogram_median": 140,
        "histogram_variance": 12,
        "histogram_tendency": 0,
    },
    "Suspect": {
        "baseline value": 165,
        "accelerations": 0.001,
        "fetal_movement": 0.0,
        "uterine_contractions": 0.008,
        "light_decelerations": 0.005,
        "severe_decelerations": 0.0,
        "prolongued_decelerations": 0.0,
        "abnormal_short_term_variability": 55,
        "mean_value_of_short_term_variability": 2.5,
        "percentage_of_time_with_abnormal_long_term_variability": 30,
        "mean_value_of_long_term_variability": 4,
        "histogram_width": 40,
        "histogram_min": 120,
        "histogram_max": 170,
        "histogram_number_of_peaks": 2,
        "histogram_number_of_zeroes": 2,
        "histogram_mode": 160,
        "histogram_mean": 155,
        "histogram_median": 158,
        "histogram_variance": 6,
        "histogram_tendency": 1,
    },
    "Pathological": {
        "baseline value": 95,
        "accelerations": 0.0,
        "fetal_movement": 0.0,
        "uterine_contractions": 0.012,
        "light_decelerations": 0.008,
        "severe_decelerations": 0.002,
        "prolongued_decelerations": 0.002,
        "abnormal_short_term_variability": 85,
        "mean_value_of_short_term_variability": 1.0,
        "percentage_of_time_with_abnormal_long_term_variability": 70,
        "mean_value_of_long_term_variability": 2,
        "histogram_width": 15,
        "histogram_min": 90,
        "histogram_max": 110,
        "histogram_number_of_peaks": 1,
        "histogram_number_of_zeroes": 5,
        "histogram_mode": 95,
        "histogram_mean": 100,
        "histogram_median": 98,
        "histogram_variance": 2,
        "histogram_tendency": -1,
    },
}
