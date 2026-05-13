from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_URL = (
    "https://raw.githubusercontent.com/sfu-cmpt340/"
    "fetal-health-classification/main/TabulatedCTG/fetal_health.csv"
)

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
class ModelBundle:
    pipeline: Pipeline
    feature_names: list[str]
    test_accuracy: float
    balanced_accuracy: float
    report: str
    holdout: pd.DataFrame
    holdout_predictions: np.ndarray


st.set_page_config(
    page_title="Fetal Health Classifier",
    page_icon="FH",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    st.markdown(
        """
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

        @media (max-width: 900px) {
            .metric-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .hero h1 {
                font-size: 1.7rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Downloading public CTG dataset...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL)
    df.columns = [col.strip() for col in df.columns]
    df["fetal_health"] = df["fetal_health"].astype(int)
    return df


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

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=350,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    holdout = x_test.copy()
    holdout["actual"] = y_test.to_numpy()

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
    )


def feature_bounds(df: pd.DataFrame, feature: str) -> tuple[float, float, float]:
    low = float(df[feature].min())
    high = float(df[feature].max())
    median = float(df[feature].median())
    if math.isclose(low, high):
        high = low + 1.0
    return low, high, median


def format_value(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def numeric_input_for_feature(df: pd.DataFrame, feature: str) -> float:
    low, high, median = feature_bounds(df, feature)
    description = FEATURE_DESCRIPTIONS.get(feature, feature.replace("_", " ").title())
    label = f"{description} | dataset column: {feature}"
    is_integer = np.all(np.isclose(df[feature], df[feature].round()))
    step = 1.0 if is_integer else 0.001
    value = st.number_input(
        label,
        min_value=low,
        max_value=high,
        value=median,
        step=step,
        help=(
            f"Allowed range from the training dataset: "
            f"{format_value(low)} to {format_value(high)}. "
            f"Default value is the dataset median: {format_value(median)}."
        ),
    )
    st.markdown(
        f"""
        <div class="field-range">
            Allowed range: <strong>{format_value(low)}</strong> to
            <strong>{format_value(high)}</strong> | median: {format_value(median)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    return float(value)


def probability_chart(probabilities: np.ndarray) -> go.Figure:
    labels = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS)]
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
    classifier = bundle.pipeline.named_steps["classifier"]
    importances = classifier.feature_importances_
    feature_medians = bundle.holdout[bundle.feature_names].median()
    user_row = user_values.iloc[0]
    deltas = (user_row - feature_medians).abs()
    normalized_deltas = deltas / (bundle.holdout[bundle.feature_names].std().replace(0, 1))
    influence = importances * normalized_deltas.to_numpy()

    table = pd.DataFrame(
        {
            "Feature": bundle.feature_names,
            "Clinical field": [
                FEATURE_DESCRIPTIONS.get(feature, feature) for feature in bundle.feature_names
            ],
            "User value": [user_row[feature] for feature in bundle.feature_names],
            "Dataset median": [feature_medians[feature] for feature in bundle.feature_names],
            "Influence score": influence,
        }
    )
    return table.sort_values("Influence score", ascending=False).head(8)


def recommendation_for(label: str, confidence: float) -> str:
    if label == "Normal":
        return (
            "Pattern is closest to normal CTG records in the training data. "
            "Continue routine monitoring and review alongside the full clinical picture."
        )
    if label == "Suspect":
        return (
            "Pattern resembles suspect CTG records. Repeat assessment, review maternal/fetal context, "
            "and consider closer monitoring."
        )
    return (
        "Pattern resembles pathological CTG records. Escalate for urgent clinician review and confirm "
        "with standard obstetric protocols."
    )


def render_header(df: pd.DataFrame, bundle: ModelBundle) -> None:
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
        importance = pd.DataFrame(
            {
                "Feature": bundle.feature_names,
                "Importance": bundle.pipeline.named_steps["classifier"].feature_importances_,
            }
        ).sort_values("Importance", ascending=False).head(10)
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


def render_prediction_form(df: pd.DataFrame, bundle: ModelBundle) -> None:
    st.subheader("New Patient CTG Assessment")
    st.caption(
        "Enter CTG-derived values from the patient record. Every field shows the dataset column name, "
        "allowed range, and median default."
    )

    with st.form("ctg_assessment_form"):
        tab_core, tab_variability, tab_histogram = st.tabs(
            ["Core CTG", "Variability", "Histogram"]
        )

        values: dict[str, float] = {}
        with tab_core:
            cols = st.columns(2)
            core_features = [
                "baseline value",
                "accelerations",
                "fetal_movement",
                "uterine_contractions",
                "light_decelerations",
                "severe_decelerations",
                "prolongued_decelerations",
            ]
            for index, feature in enumerate(core_features):
                with cols[index % 2]:
                    values[feature] = numeric_input_for_feature(df, feature)

        with tab_variability:
            cols = st.columns(2)
            variability_features = [
                "abnormal_short_term_variability",
                "mean_value_of_short_term_variability",
                "percentage_of_time_with_abnormal_long_term_variability",
                "mean_value_of_long_term_variability",
            ]
            for index, feature in enumerate(variability_features):
                with cols[index % 2]:
                    values[feature] = numeric_input_for_feature(df, feature)

        with tab_histogram:
            cols = st.columns(2)
            histogram_features = [
                "histogram_width",
                "histogram_min",
                "histogram_max",
                "histogram_number_of_peaks",
                "histogram_number_of_zeroes",
                "histogram_mode",
                "histogram_mean",
                "histogram_median",
                "histogram_variance",
                "histogram_tendency",
            ]
            for index, feature in enumerate(histogram_features):
                with cols[index % 2]:
                    values[feature] = numeric_input_for_feature(df, feature)

        submitted = st.form_submit_button("Assess fetal health", use_container_width=True)

    if not submitted:
        st.info("Submit the form to generate a prediction and analytics summary.")
        return

    user_df = pd.DataFrame([{feature: values[feature] for feature in bundle.feature_names}])
    predicted_class = int(bundle.pipeline.predict(user_df)[0])
    probabilities = bundle.pipeline.predict_proba(user_df)[0]
    label = CLASS_LABELS[predicted_class]
    confidence = float(probabilities[predicted_class - 1])
    color = CLASS_COLORS[label]

    st.markdown(
        f"""
        <div class="result" style="border-left-color:{color}">
            <h2>Prediction: {label} fetal health</h2>
            <p>Model confidence: <strong>{confidence:.1%}</strong>. {recommendation_for(label, confidence)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_col, table_col = st.columns([1, 1.1])
    with chart_col:
        st.plotly_chart(probability_chart(probabilities), use_container_width=True)

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


def render_model_details(bundle: ModelBundle) -> None:
    st.subheader("Model Training And Evaluation")
    st.caption("This tab is for explaining how the model was trained and how it performed.")
    metric_col_1, metric_col_2 = st.columns(2)
    metric_col_1.metric("Test accuracy", f"{bundle.test_accuracy:.1%}")
    metric_col_2.metric("Balanced accuracy", f"{bundle.balanced_accuracy:.1%}")

    st.markdown(
        """
        The model is a Random Forest classifier trained on CTG records. The dataset is split into
        training data and unseen test data, then the test set is used to estimate performance.
        """
    )
    with st.expander("Full classification report"):
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

    render_header(df, bundle)

    assessment_tab, analytics_tab, model_tab = st.tabs(
        ["New Assessment", "Dataset Analytics", "Model Training"]
    )
    with assessment_tab:
        render_prediction_form(df, bundle)
    with analytics_tab:
        render_dashboard(df, bundle)
    with model_tab:
        render_model_details(bundle)


if __name__ == "__main__":
    main()
