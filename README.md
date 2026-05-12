# Fetal Health Classification App

A Streamlit medical data analytics application for classifying fetal health from Cardiotocography (CTG) measurements.

The app satisfies the project requirements:

- Collects CTG measurements from the user through a real web interface.
- Analyzes user data with charts, risk drivers, and comparison against a public dataset.
- Predicts fetal health status using a machine learning model trained on a public medical dataset.

## Project Idea

**User:** Obstetrician, maternity clinic doctor, or trained healthcare staff.

**Input Data:** CTG-derived values such as baseline fetal heart rate, accelerations, fetal movement, uterine contractions, decelerations, short-term variability, long-term variability, and histogram features.

**Dataset:** Public fetal health CTG dataset based on 2,126 cardiotocogram records classified by expert obstetricians into:

- Normal
- Suspect
- Pathological

Primary source: [UCI Cardiotocography Dataset](https://archive.ics.uci.edu/dataset/193/cardiotocography)  
CSV mirror used by the app: [fetal_health.csv](https://raw.githubusercontent.com/sfu-cmpt340/fetal-health-classification/main/TabulatedCTG/fetal_health.csv)

**Machine Learning Model:** Random Forest classifier with class balancing.

**Prediction Output:** Fetal health class, confidence score, class probabilities, feature contribution estimate, and clinical-style recommendation.

> This application is for educational use only and is not a substitute for clinical judgment.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app downloads the public dataset automatically when it starts.

## Suggested Demo Flow

1. Open the app.
2. Review the dashboard metrics and class distribution.
3. Enter CTG values in the assessment form.
4. Click **Assess fetal health**.
5. Discuss the prediction, probability chart, and most influential features.
