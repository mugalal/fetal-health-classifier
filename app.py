from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clinical_bounds import (
    CLINICAL_BOUNDS,
    DISCRETE_FEATURES,
    EXAMPLE_PRESETS,
    INTEGER_FEATURES,
    is_outside_reference,
)


DATA_URL = (
    "https://raw.githubusercontent.com/sfu-cmpt340/"
    "fetal-health-classification/main/TabulatedCTG/fetal_health.csv"
)

DATA_DIR = Path("data")
DATA_PATH = DATA_DIR / "fetal_health.csv"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.joblib"
FINGERPRINT_PATH = MODEL_DIR / "model.fingerprint"

CLASS_LABELS = {
    1: "Normal",
    2: "Suspect",
    3: "Pathological",
}

CLASS_COLORS = {
    "Normal": "#1f9d55",
    "Suspect": "#d97706",
    "Pathological": "#dc2626",
}

FEATURE_DESCRIPTIONS = {
    "baseline value": "Baseline fetal heart rate",
    "accelerations": "Accelerations per second",
    "fetal_movement": "Fetal movements per second",
    "uterine_contractions": "Uterine contractions per second",
    "light_decelerations": "Light decelerations per second",
    "severe_decelerations": "Severe decelerations per second",
    "prolongued_decelerations": "Prolonged decelerations per second",
    "abnormal_short_term_variability": "Abnormal short-term variability",
    "mean_value_of_short_term_variability": "Mean short-term variability",
    "percentage_of_time_with_abnormal_long_term_variability": "Abnormal long-term variability time",
    "mean_value_of_long_term_variability": "Mean long-term variability",
    "histogram_width": "FHR histogram width",
    "histogram_min": "FHR histogram minimum",
    "histogram_max": "FHR histogram maximum",
    "histogram_number_of_peaks": "FHR histogram peaks",
    "histogram_number_of_zeroes": "FHR histogram zeroes",
    "histogram_mode": "FHR histogram mode",
    "histogram_mean": "FHR histogram mean",
    "histogram_median": "FHR histogram median",
    "histogram_variance": "FHR histogram variance",
    "histogram_tendency": "FHR histogram tendency",
}


@dataclass(frozen=True)
class ClassifierEvaluation:
    name: str
    cv_accuracy: float
    cv_balanced_accuracy: float
    per_class: dict
    confusion_matrix: np.ndarray


@dataclass(frozen=True)
class ModelBundle:
    pipeline: Pipeline
    feature_names: list[str]
    test_accuracy: float
    balanced_accuracy: float
    report: str
    holdout: pd.DataFrame
    holdout_predictions: np.ndarray
    feature_stats: dict
    train_medians: pd.Series
    train_stds: pd.Series
    feature_importances: np.ndarray
    classes: np.ndarray
    evaluations: dict


st.set_page_config(
    page_title="Fetal Health Classifier",
    page_icon="FH",
    layout="wide",
    initial_sidebar_state="collapsed",
)


STYLES = """
<style>
:root {
    --ink: #18212f;
    --muted: #607086;
    --panel: #ffffff;
    --line: #d9e2ec;
    --soft: #f6f9fc;
    --accent: #0f766e;
}

.stApp {
    background: linear-gradient(180deg, #f7fbfc 0%, #eef5f7 100%);
    color: var(--ink);
}

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--line);
}

h1, h2, h3, h4, h5, h6, p, span, label, div {
    color: var(--ink);
}

button[data-baseweb="tab"] p,
button[data-baseweb="tab"] div,
button[data-baseweb="tab"] span {
    color: var(--ink) !important;
    opacity: 1 !important;
    font-weight: 650;
}

button[data-baseweb="tab"][aria-selected="true"] p,
button[data-baseweb="tab"][aria-selected="true"] div,
button[data-baseweb="tab"][aria-selected="true"] span {
    color: var(--accent) !important;
}

div[data-testid="stNumberInput"] label,
div[data-testid="stNumberInput"] label p {
    color: var(--ink) !important;
    opacity: 1 !important;
    font-weight: 700;
}

.hero {
    border-bottom: 1px solid var(--line);
    padding: 1rem 0 1.25rem;
    margin-bottom: 1rem;
}

.hero h1 {
    color: var(--ink);
    font-size: 2.15rem;
    line-height: 1.12;
    margin: 0 0 .45rem;
    letter-spacing: 0;
}

.hero p {
    color: var(--muted);
    font-size: 1rem;
    max-width: 820px;
    margin: 0;
}

.metric-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: .75rem;
    margin: .65rem 0 1.2rem;
}

.metric-box {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: .9rem 1rem;
    min-height: 96px;
}

.metric-box span {
    color: var(--muted);
    display: block;
    font-size: .82rem;
    margin-bottom: .35rem;
}

.metric-box strong {
    color: var(--ink);
    display: block;
    font-size: 1.45rem;
    line-height: 1.1;
}

.metric-box small {
    color: var(--muted);
    display: block;
    margin-top: .35rem;
}

.result {
    background: var(--panel);
    border: 1px solid var(--line);
    border-left: 6px solid var(--accent);
    border-radius: 8px;
    padding: 1rem 1.1rem;
    margin: .5rem 0 1rem;
}

.result h2 {
    font-size: 1.35rem;
    margin: 0 0 .35rem;
    letter-spacing: 0;
}

.result p {
    color: var(--muted);
    margin: 0;
}

.notice {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    color: #7c2d12;
    border-radius: 8px;
    padding: .8rem .9rem;
    margin-top: .75rem;
}

div[data-testid="stMetricValue"] {
    color: var(--ink);
}

.stButton button {
    background: #0f766e;
    color: white;
    border: 1px solid #0f766e;
    border-radius: 6px;
    min-height: 2.75rem;
    font-weight: 650;
}

.stButton button:hover {
    background: #115e59;
    color: white;
    border-color: #115e59;
}

.field-range {
    color: var(--muted);
    font-size: .78rem;
    margin: -.55rem 0 .7rem;
}

.field-label {
    background: #ffffff;
    border: 1px solid var(--line);
    border-bottom: 0;
    border-radius: 8px 8px 0 0;
    padding: .7rem .85rem .55rem;
}

.field-label strong {
    color: var(--ink);
    display: block;
    font-size: .95rem;
    margin-bottom: .2rem;
}

.field-label span {
    color: var(--muted);
    display: block;
    font-size: .8rem;
    line-height: 1.35;
}

@media (max-width: 900px) {
    .metric-strip {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .hero h1 {
        font-size: 1.7rem;
    }
}
</style>
"""


def inject_styles() -> None:
    if st.session_state.get("_styles_injected"):
        return
    st.markdown(STYLES, unsafe_allow_html=True)
    st.session_state["_styles_injected"] = True


@st.cache_data(show_spinner="Loading CTG dataset...")
def load_data() -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_csv(DATA_URL)
        df.to_csv(DATA_PATH, index=False)
    df.columns = [col.strip() for col in df.columns]
    df["fetal_health"] = df["fetal_health"].astype(int)
    return df


def _build_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def _build_classifier_suite() -> dict[str, Pipeline]:
    rf = Pipeline(steps=[("classifier", _build_classifier())])
    gb = Pipeline(
        steps=[
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                ),
            )
        ]
    )
    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    return {"Random Forest": rf, "Gradient Boosting": gb, "Logistic Regression": lr}


def _evaluate_classifier(name: str, pipeline: Pipeline, x: pd.DataFrame, y: pd.Series) -> ClassifierEvaluation:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_preds = cross_val_predict(pipeline, x, y, cv=cv, n_jobs=-1)
    cm = confusion_matrix(y, cv_preds, labels=[1, 2, 3])
    report = classification_report(y, cv_preds, output_dict=True, zero_division=0)
    per_class: dict[int, dict] = {}
    for label in (1, 2, 3):
        entry = report.get(str(label), {})
        per_class[label] = {
            "precision": float(entry.get("precision", 0.0)),
            "recall": float(entry.get("recall", 0.0)),
            "f1": float(entry.get("f1-score", 0.0)),
            "support": int(entry.get("support", 0)),
        }
    return ClassifierEvaluation(
        name=name,
        cv_accuracy=float(accuracy_score(y, cv_preds)),
        cv_balanced_accuracy=float(balanced_accuracy_score(y, cv_preds)),
        per_class=per_class,
        confusion_matrix=cm,
    )


def _evaluate_suite(x: pd.DataFrame, y: pd.Series) -> dict[str, ClassifierEvaluation]:
    return {name: _evaluate_classifier(name, pl, x, y) for name, pl in _build_classifier_suite().items()}


def _data_fingerprint(df: pd.DataFrame) -> str:
    """Stable short hash of the training data, used to invalidate the cached
    model whenever the underlying dataset changes."""
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update(",".join(sorted(df.columns)).encode())
    h.update(df.head(50).to_csv(index=False).encode())
    return h.hexdigest()[:16]


def _try_load_cached_model(expected_fingerprint: str) -> Pipeline | None:
    if not MODEL_PATH.exists() or not FINGERPRINT_PATH.exists():
        return None
    try:
        if FINGERPRINT_PATH.read_text().strip() != expected_fingerprint:
            return None
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


@st.cache_resource(show_spinner="Training Random Forest model...")
def train_model(df: pd.DataFrame) -> ModelBundle:
    feature_names = [col for col in df.columns if col != "fetal_health"]
    x = df[feature_names]
    y = df["fetal_health"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    MODEL_DIR.mkdir(exist_ok=True)
    fingerprint = _data_fingerprint(df)
    pipeline = _try_load_cached_model(fingerprint)
    if pipeline is None:
        pipeline = Pipeline(steps=[("classifier", _build_classifier())])
        pipeline.fit(x_train, y_train)
        joblib.dump(pipeline, MODEL_PATH)
        FINGERPRINT_PATH.write_text(fingerprint)

    predictions = pipeline.predict(x_test)

    feature_stats: dict[str, dict] = {}
    for feature in feature_names:
        col = df[feature]
        std = float(col.std()) or 1.0
        feature_stats[feature] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "median": float(col.median()),
            "std": std,
            "is_integer": bool(np.all(np.isclose(col, col.round()))),
        }

    holdout = x_test.copy()
    holdout["actual"] = y_test.to_numpy()

    classifier = pipeline.named_steps["classifier"]
    train_stds = x_train.std().replace(0, 1)

    evaluations = _evaluate_suite(x_train, y_train)

    return ModelBundle(
        pipeline=pipeline,
        feature_names=feature_names,
        test_accuracy=accuracy_score(y_test, predictions),
        balanced_accuracy=balanced_accuracy_score(y_test, predictions),
        report=classification_report(
            y_test,
            predictions,
            target_names=[CLASS_LABELS[i] for i in sorted(CLASS_LABELS)],
        ),
        holdout=holdout,
        holdout_predictions=predictions,
        feature_stats=feature_stats,
        train_medians=x_train.median(),
        train_stds=train_stds,
        feature_importances=classifier.feature_importances_,
        classes=classifier.classes_,
        evaluations=evaluations,
    )


def format_value(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _reference_text(bound: dict) -> str:
    parts = []
    if bound["ref_low"] is not None:
        parts.append(f">= {format_value(float(bound['ref_low']))}")
    if bound["ref_high"] is not None:
        parts.append(f"<= {format_value(float(bound['ref_high']))}")
    if not parts:
        return ""
    return f" | normal/reference: {' and '.join(parts)}"


def _training_range_text(feature: str, bundle: ModelBundle) -> str:
    stats = bundle.feature_stats[feature]
    return (
        f" | training data: {format_value(float(stats['min']))} to "
        f"{format_value(float(stats['max']))}"
    )


def clinical_input(feature: str, bundle: ModelBundle) -> float:
    bound = CLINICAL_BOUNDS.get(feature)
    description = FEATURE_DESCRIPTIONS.get(feature, feature.replace("_", " ").title())

    if bound is None:
        # Fallback when no clinical bound is defined: empirical bounds from the dataset.
        stats = bundle.feature_stats[feature]
        value = st.number_input(
            description,
            min_value=float(stats["min"]),
            max_value=float(stats["max"]),
            value=float(stats["median"]),
            step=1.0 if stats["is_integer"] else 0.001,
            key=f"input_{feature}",
        )
        return float(value)

    note = bound["note"]
    unit_label = bound["unit"] if bound["unit"] else "-"
    range_text = (
        f"Unit: {unit_label} | input range: "
        f"{format_value(float(bound['hard_min']))} to "
        f"{format_value(float(bound['hard_max']))}"
        f"{_reference_text(bound)}"
        f"{_training_range_text(feature, bundle)}"
    )

    st.markdown(
        f"""
        <div class="field-label">
            <strong>{description}</strong>
            <span>{note}</span>
            <span>{range_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if feature in DISCRETE_FEATURES:
        options = DISCRETE_FEATURES[feature]
        default = int(bound["default"])
        default_index = options.index(default) if default in options else 0
        value = st.selectbox(
            description,
            options,
            index=default_index,
            label_visibility="collapsed",
            help=note,
            key=f"input_{feature}",
        )
        warning = is_outside_reference(feature, float(value))
        if warning:
            st.warning(warning)
        return float(value)

    if feature in INTEGER_FEATURES:
        value = st.number_input(
            description,
            min_value=int(bound["hard_min"]),
            max_value=int(bound["hard_max"]),
            value=int(bound["default"]),
            step=1,
            label_visibility="collapsed",
            help=note,
            key=f"input_{feature}",
        )
    else:
        value = st.number_input(
            description,
            min_value=float(bound["hard_min"]),
            max_value=float(bound["hard_max"]),
            value=float(bound["default"]),
            step=0.001,
            format="%.4f",
            label_visibility="collapsed",
            help=note,
            key=f"input_{feature}",
        )

    warning = is_outside_reference(feature, float(value))
    if warning:
        st.warning(warning)

    return float(value)


def probability_chart(probabilities: np.ndarray, classes: np.ndarray) -> go.Figure:
    labels = [CLASS_LABELS[int(c)] for c in classes]
    fig = px.bar(
        x=labels,
        y=probabilities,
        color=labels,
        color_discrete_map=CLASS_COLORS,
        labels={"x": "Class", "y": "Model probability"},
        text=[f"{p:.1%}" for p in probabilities],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        showlegend=False,
        yaxis_tickformat=".0%",
        yaxis_range=[0, max(1.0, float(probabilities.max()) + 0.1)],
        margin=dict(l=10, r=10, t=20, b=10),
        height=320,
    )
    return fig


def contribution_table(bundle: ModelBundle, user_values: pd.DataFrame) -> pd.DataFrame:
    importances = bundle.feature_importances
    user_row = user_values.iloc[0]
    deltas = (user_row - bundle.train_medians).abs()
    normalized_deltas = deltas / bundle.train_stds
    influence = importances * normalized_deltas.to_numpy()

    table = pd.DataFrame(
        {
            "Feature": bundle.feature_names,
            "Clinical field": [
                FEATURE_DESCRIPTIONS.get(feature, feature) for feature in bundle.feature_names
            ],
            "User value": [user_row[feature] for feature in bundle.feature_names],
            "Training median": [bundle.train_medians[feature] for feature in bundle.feature_names],
            "Influence score": influence,
        }
    )
    return table.sort_values("Influence score", ascending=False).head(8)


def recommendation_for(label: str, confidence: float) -> str:
    low_conf_suffix = (
        " Confidence is low - treat the output cautiously and review with a clinician."
        if confidence < 0.6
        else ""
    )
    if label == "Normal":
        return (
            "Pattern is closest to normal CTG records in the training data. "
            "Continue routine monitoring and review alongside the full clinical picture."
            + low_conf_suffix
        )
    if label == "Suspect":
        return (
            "Pattern resembles suspect CTG records. Repeat assessment, review maternal/fetal context, "
            "and consider closer monitoring."
            + low_conf_suffix
        )
    return (
        "Pattern resembles pathological CTG records. Escalate for urgent clinician review and confirm "
        "with standard obstetric protocols."
        + low_conf_suffix
    )


def render_overview(df: pd.DataFrame, bundle: ModelBundle) -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Fetal Health CTG Classifier</h1>
            <p>
                A working medical analytics app that collects Cardiotocography measurements,
                analyzes the submitted values, and predicts whether fetal status is normal,
                suspect, or pathological.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    class_counts = df["fetal_health"].map(CLASS_LABELS).value_counts()
    st.markdown(
        f"""
        <div class="metric-strip">
            <div class="metric-box">
                <span>Dataset records</span>
                <strong>{len(df):,}</strong>
                <small>Public CTG exams</small>
            </div>
            <div class="metric-box">
                <span>Input features</span>
                <strong>{len(bundle.feature_names)}</strong>
                <small>Extracted from CTG signals</small>
            </div>
            <div class="metric-box">
                <span>Test accuracy</span>
                <strong>{bundle.test_accuracy:.1%}</strong>
                <small>Held-out stratified split</small>
            </div>
            <div class="metric-box">
                <span>Most common class</span>
                <strong>{class_counts.index[0]}</strong>
                <small>{class_counts.iloc[0]:,} records</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(df: pd.DataFrame, bundle: ModelBundle) -> None:
    st.subheader("Dataset Analytics")
    st.caption("These charts explain the public CTG dataset and what the trained model learned.")
    left, right = st.columns([1.1, 1])

    with left:
        class_df = (
            df["fetal_health"]
            .map(CLASS_LABELS)
            .value_counts()
            .rename_axis("Class")
            .reset_index(name="Records")
        )
        fig = px.bar(
            class_df,
            x="Class",
            y="Records",
            color="Class",
            color_discrete_map=CLASS_COLORS,
            title="Dataset Class Distribution",
        )
        fig.update_layout(showlegend=False, height=360, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        importance = (
            pd.DataFrame(
                {
                    "Feature": bundle.feature_names,
                    "Importance": bundle.feature_importances,
                }
            )
            .sort_values("Importance", ascending=False)
            .head(10)
        )
        importance["Feature"] = importance["Feature"].map(
            lambda item: FEATURE_DESCRIPTIONS.get(item, item)
        )
        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Model Drivers",
            color_discrete_sequence=["#0f766e"],
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            height=360,
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


def _default_value(feature: str, bundle: ModelBundle) -> float:
    if feature in CLINICAL_BOUNDS:
        return float(CLINICAL_BOUNDS[feature]["default"])
    return float(bundle.feature_stats[feature]["median"])


def _apply_preset(class_label: str, feature_names: list[str]) -> None:
    """Push EXAMPLE_PRESETS[class_label] into st.session_state for each input widget."""
    preset = EXAMPLE_PRESETS.get(class_label, {})
    for feature in feature_names:
        if feature not in preset:
            continue
        value = preset[feature]
        if feature in DISCRETE_FEATURES or feature in INTEGER_FEATURES:
            st.session_state[f"input_{feature}"] = int(value)
        else:
            st.session_state[f"input_{feature}"] = float(value)


def render_prediction_form(df: pd.DataFrame, bundle: ModelBundle) -> None:
    st.title("New Patient Assessment")
    st.caption(
        "Enter the main CTG values from the patient record. Advanced statistical fields are "
        "filled with clinically-typical defaults unless you choose to edit them. Use the "
        "example buttons below to populate the form with a representative CTG of each class."
    )

    preset_cols = st.columns(3)
    for col, class_label in zip(preset_cols, ["Normal", "Suspect", "Pathological"]):
        if col.button(f"Load {class_label} example", use_container_width=True, key=f"preset_{class_label}"):
            _apply_preset(class_label, bundle.feature_names)
            st.rerun()

    with st.form("ctg_assessment_form"):
        values: dict[str, float] = {
            feature: _default_value(feature, bundle) for feature in bundle.feature_names
        }

        st.markdown("**Quick CTG inputs**")
        st.caption("These are the easiest values to explain in your presentation and demo.")
        quick_features = [
            "baseline value",
            "accelerations",
            "fetal_movement",
            "uterine_contractions",
            "light_decelerations",
            "severe_decelerations",
            "prolongued_decelerations",
            "abnormal_short_term_variability",
            "mean_value_of_short_term_variability",
        ]
        cols = st.columns(2)
        for index, feature in enumerate(quick_features):
            with cols[index % 2]:
                values[feature] = clinical_input(feature, bundle)

        with st.expander("Advanced CTG statistics"):
            st.caption(
                "Optional fields used by the trained model. Defaults reflect clinically-typical values."
            )
            advanced_features = [
                feature for feature in bundle.feature_names if feature not in quick_features
            ]
            advanced_cols = st.columns(2)
            for index, feature in enumerate(advanced_features):
                with advanced_cols[index % 2]:
                    values[feature] = clinical_input(feature, bundle)

        submitted = st.form_submit_button("Assess fetal health", use_container_width=True)

    if not submitted:
        st.info("Submit the form to generate a prediction and analytics summary.")
        return

    user_df = pd.DataFrame([{feature: values[feature] for feature in bundle.feature_names}])
    probabilities = bundle.pipeline.predict_proba(user_df)[0]
    class_idx = int(np.argmax(probabilities))
    predicted_class = int(bundle.classes[class_idx])
    label = CLASS_LABELS[predicted_class]
    confidence = float(probabilities[class_idx])
    color = CLASS_COLORS[label]

    st.markdown(
        f"""
        <div class="result" style="border-left-color:{color}">
            <h2>Prediction: {label} fetal health</h2>
            <p>Model score: <strong>{confidence:.1%}</strong>. {recommendation_for(label, confidence)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_col, table_col = st.columns([1, 1.1])
    with chart_col:
        st.plotly_chart(
            probability_chart(probabilities, bundle.classes),
            use_container_width=True,
        )

    with table_col:
        st.markdown("**Most influential submitted values**")
        top_contributors = contribution_table(bundle, user_df)
        st.dataframe(
            top_contributors,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Influence score": st.column_config.ProgressColumn(
                    "Influence score",
                    min_value=0,
                    max_value=float(max(top_contributors["Influence score"].max(), 0.001)),
                )
            },
        )

    st.markdown(
        """
        <div class="notice">
            Educational disclaimer: this model supports project demonstration and clinical data analytics learning.
            It must not be used as a standalone diagnostic system.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _confusion_matrix_figure(name: str, cm: np.ndarray) -> go.Figure:
    labels = ["Normal", "Suspect", "Pathological"]
    text = [[str(int(v)) for v in row] for row in cm]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Pred {l}" for l in labels],
            y=[f"True {l}" for l in labels],
            text=text,
            texttemplate="%{text}",
            colorscale="Teal",
            showscale=False,
        )
    )
    fig.update_layout(
        title=name,
        height=320,
        margin=dict(l=10, r=10, t=45, b=10),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def render_model_details(bundle: ModelBundle) -> None:
    st.subheader("Model Evaluation")
    st.caption("This tab is for explaining how the model was trained and how it performed.")
    metric_col_1, metric_col_2 = st.columns(2)
    metric_col_1.metric("Test accuracy (Random Forest)", f"{bundle.test_accuracy:.1%}")
    metric_col_2.metric("Balanced accuracy (Random Forest)", f"{bundle.balanced_accuracy:.1%}")

    st.markdown(
        """
        The production classifier is a Random Forest. The table below compares it against
        Gradient Boosting and Logistic Regression using 5-fold cross-validation on the
        training set, so each row is an honest out-of-fold estimate. Per-class recall is
        the key metric for imbalanced clinical data - it tells you how often the model
        actually catches Suspect and Pathological cases.
        """
    )

    rows = []
    for ev in bundle.evaluations.values():
        rows.append(
            {
                "Classifier": ev.name,
                "Accuracy": f"{ev.cv_accuracy:.1%}",
                "Balanced acc.": f"{ev.cv_balanced_accuracy:.1%}",
                "Normal recall": f"{ev.per_class[1]['recall']:.1%}",
                "Suspect recall": f"{ev.per_class[2]['recall']:.1%}",
                "Pathological recall": f"{ev.per_class[3]['recall']:.1%}",
                "Normal F1": f"{ev.per_class[1]['f1']:.3f}",
                "Suspect F1": f"{ev.per_class[2]['f1']:.3f}",
                "Pathological F1": f"{ev.per_class[3]['f1']:.3f}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Confusion matrices (out-of-fold)")
    st.caption(
        "Rows are the true class, columns are the predicted class. Off-diagonal cells are "
        "mistakes. The Suspect row tells you how often Suspect cases get mislabeled as Normal."
    )
    cm_cols = st.columns(3)
    for col, ev in zip(cm_cols, bundle.evaluations.values()):
        with col:
            st.plotly_chart(
                _confusion_matrix_figure(ev.name, ev.confusion_matrix),
                use_container_width=True,
            )

    with st.expander("Held-out test set classification report (Random Forest)"):
        st.code(bundle.report, language="text")


def main() -> None:
    inject_styles()
    df = load_data()
    bundle = train_model(df)

    with st.sidebar:
        st.title("Project")
        st.markdown(
            """
            **Fetal Health Classification**

            Built for a Biomedical Data Analytics final project.
            """
        )
        st.caption("Use the sidebar arrow in the top-left corner to open or close this panel.")
        st.divider()
        st.markdown("**Dataset source**")
        st.link_button("UCI CTG Dataset", "https://archive.ics.uci.edu/dataset/193/cardiotocography")
        st.link_button("CSV Mirror", DATA_URL)
        st.divider()
        st.markdown("**Classes**")
        for class_id, label in CLASS_LABELS.items():
            st.markdown(f"- `{class_id}` {label}")

    assessment_tab, analytics_tab, model_tab = st.tabs(
        ["New Assessment", "Data Analytics", "Model Evaluation"]
    )
    with assessment_tab:
        render_prediction_form(df, bundle)
    with analytics_tab:
        render_overview(df, bundle)
        render_dashboard(df, bundle)
    with model_tab:
        render_model_details(bundle)


if __name__ == "__main__":
    main()
