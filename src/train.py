"""Loan Approval Prediction - end-to-end training pipeline.

This module implements Phases 1-7 of the project plan:

* Phase 1 - Understand the dataset
* Phase 2 - Exploratory Data Analysis (figures saved to ``plots/``)
* Phase 3 - Data preprocessing (``ColumnTransformer`` + ``Pipeline``)
* Phase 4 - Train multiple models
* Phase 5 - Evaluate the models
* Phase 6 - Select the best model
* Phase 7 - Save the model (``models/loan_model.pkl``)

Run it with::

    python -m src.train

The script is fully reproducible: it reads ``data/loan_data.csv``, writes the
EDA figures into ``plots/``, the trained pipeline into ``models/loan_model.pkl``
and the accompanying metadata / metrics as JSON into ``models/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend: the script must run without a display

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "loan_data.csv"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
MODEL_PATH = MODELS_DIR / "loan_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TARGET_COLUMN = "loan_status"
ID_COLUMN = "loan_id"

# Section 4 of the plan: 0 -> Rejected, 1 -> Approved
TARGET_MAPPING = {"Rejected": 0, "Approved": 1}
TARGET_LABELS = {0: "Rejected", 1: "Approved"}

RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC_FEATURES = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

CATEGORICAL_FEATURES = ["education", "self_employed"]

sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------- #
# Phase 1 - Understand the dataset
# --------------------------------------------------------------------------- #
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and apply the minimal cleaning needed for modelling.

    The Kaggle file stores its column names with a leading space
    (``" education"``); we strip them so the rest of the code can use the
    documented names from section 3 of the plan.
    """
    df = pd.read_csv(path)

    # The raw header contains a leading space on every column -> strip it.
    df.columns = [column.strip() for column in df.columns]
    # The categorical values are stored with a leading space as well
    # (``" Approved"``) -> strip them so they match the documented labels.
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].str.strip()


    # ``loan_id`` is a unique row identifier, it carries no predictive signal.
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    # Encode the target: Rejected -> 0, Approved -> 1.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(TARGET_MAPPING)
    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Unknown values found in the target column; expected "
            f"{sorted(TARGET_MAPPING)}."
        )

    return df


def dataset_overview(df: pd.DataFrame) -> None:
    """Phase 1: print a textual overview of the dataset.

    Answers the questions raised in section 11 of the plan: shape, dtypes,
    missing values, duplicates and the approved/rejected distribution.
    """
    print("=" * 72)
    print("PHASE 1 - UNDERSTAND THE DATASET")
    print("=" * 72)
    print(f"Rows      : {df.shape[0]}")
    print(f"Features  : {df.shape[1] - 1} (+1 target)")
    print(f"Columns   : {list(df.columns)}")
    print("\n-- dtypes / non-null counts --")
    print(df.info())
    print("\n-- statistical summary --")
    print(df.describe().T.to_string())
    print("\n-- missing values --")
    print(df.isnull().sum().to_string())
    print(f"\n-- duplicated rows: {df.duplicated().sum()} --")

    print("\n-- numerical features --")
    print(NUMERIC_FEATURES)
    print("-- categorical features --")
    print(CATEGORICAL_FEATURES)

    print("\n-- target distribution (loan_status) --")
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    for value, count in counts.items():
        share = count / len(df)
        print(f"{TARGET_LABELS[value]:>8}: {count:>5} ({share:6.2%})")

    # This dataset is only mildly imbalanced, but it is worth reporting.
    majority = counts.max() / counts.min()
    print(f"Imbalance ratio (majority/minority): {majority:.2f}")

# --------------------------------------------------------------------------- #
# Phase 2 - Exploratory Data Analysis
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, filename: str) -> None:
    """Persist a figure to the ``plots/`` directory and close it."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(BASE_DIR)}")


def run_eda(df: pd.DataFrame) -> None:
    """Phase 2: answer the EDA questions with figures written to ``plots/``."""
    print("\n" + "=" * 72)
    print("PHASE 2 - EXPLORATORY DATA ANALYSIS")
    print("=" * 72)

    status_labels = df[TARGET_COLUMN].map(TARGET_LABELS)

    # Q1 - Does the CIBIL score relate to loan approval?
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="cibil_score", hue=status_labels, bins=30, kde=True, ax=ax)
    ax.set_title("Q1  CIBIL score distribution by loan status")
    ax.set_xlabel("CIBIL score")
    _save(fig, "01_cibil_score_vs_status.png")

    # Q2 - Does income affect loan approval?
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x=status_labels, y="income_annum", ax=ax)
    ax.set_title("Q2  Annual income by loan status")
    ax.set_xlabel("Loan status")
    ax.set_ylabel("Annual income")
    _save(fig, "02_income_vs_status.png")

    # Q3 - Does the requested loan amount affect approval?
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x=status_labels, y="loan_amount", ax=ax)
    ax.set_title("Q3  Loan amount by loan status")
    ax.set_xlabel("Loan status")
    ax.set_ylabel("Loan amount")
    _save(fig, "03_loan_amount_vs_status.png")

    # Q4 - Do education / self-employment matter?
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(data=df, x="education", hue=status_labels, ax=axes[0])
    axes[0].set_title("Q4a  Education by loan status")
    axes[0].set_xlabel("Education")
    sns.countplot(data=df, x="self_employed", hue=status_labels, ax=axes[1])
    axes[1].set_title("Q4b  Self employment by loan status")
    axes[1].set_xlabel("Self employed")
    _save(fig, "04_categorical_vs_status.png")

    # Q5 - Is there a class imbalance problem?
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.countplot(x=status_labels, ax=ax)
    ax.set_title("Q5  Target class balance")
    ax.set_xlabel("Loan status")
    ax.set_ylabel("Count")
    for patch, value in zip(ax.patches, df[TARGET_COLUMN].value_counts().sort_index()):
        ax.annotate(
            str(value),
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
        )
    _save(fig, "05_target_balance.png")

    # Loan term is a useful additional numeric signal.
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="loan_term", y=status_labels, ax=ax)
    ax.set_title("Loan term vs loan status")
    ax.set_xlabel("Loan term (years)")
    ax.set_ylabel("Loan status")
    _save(fig, "06_loan_term_vs_status.png")

    # Correlation heat map (numeric features + target).
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation = df[NUMERIC_FEATURES + [TARGET_COLUMN]].corr()
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation heat map")
    _save(fig, "07_correlation_heatmap.png")

    # Scatter: CIBIL score vs loan amount, coloured by outcome.
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df, x="cibil_score", y="loan_amount",
        hue=status_labels, alpha=0.6, ax=ax,
    )
    ax.set_title("Q1/Q3  CIBIL score vs loan amount")
    _save(fig, "08_cibil_vs_loan_amount.png")

    strongest = (
        correlation[TARGET_COLUMN].drop(TARGET_COLUMN).abs().sort_values(ascending=False)
    )
    print("\nAbsolute correlation with the target (top 5):")
    print(strongest.head(5).to_string())


# --------------------------------------------------------------------------- #
# Phase 3 - Data preprocessing
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """Build the reusable preprocessing ``ColumnTransformer``.

    * numerical features -> median imputation + ``StandardScaler``
    * categorical features -> most-frequent imputation + ``OneHotEncoder``

    The exact same object is trained inside the model ``Pipeline`` and is
    therefore applied identically at prediction time (plan section 15). This
    keeps preprocessing consistent and avoids data leakage.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


# --------------------------------------------------------------------------- #
# Phase 4 - Train multiple models
# --------------------------------------------------------------------------- #
def build_models() -> dict:
    """Return the candidate classifiers to compare (plan section 16).

    ``class_weight="balanced"`` is used because the target is not perfectly
    balanced (about 62% Approved / 38% Rejected).
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def build_pipeline(estimator) -> Pipeline:
    """Chain the preprocessor and a classifier into a single ``Pipeline``."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


# --------------------------------------------------------------------------- #
# Phase 5 - Evaluate the models
# --------------------------------------------------------------------------- #
def evaluate(y_true, y_pred) -> dict:
    """Compute the classification metrics required by the plan (section 17)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str, filename: str) -> list:
    """Save a labelled confusion matrix and return it as a nested list."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[TARGET_LABELS[0], TARGET_LABELS[1]],
        yticklabels=[TARGET_LABELS[0], TARGET_LABELS[1]],
        ax=ax,
    )
    ax.set_title(f"Confusion matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    _save(fig, filename)
    return matrix.tolist()


# --------------------------------------------------------------------------- #
# Phases 6 & 7 - Select the best model and save it
# --------------------------------------------------------------------------- #
def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: understand the dataset ---------------------------------
    df = load_data()
    dataset_overview(df)

    # ---- Phase 2: exploratory data analysis ------------------------------
    run_eda(df)

    # ---- Phase 3: split + preprocessing ----------------------------------
    print("\n" + "=" * 72)
    print("PHASE 3 - DATA PREPROCESSING")
    print("=" * 72)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train rows: {len(X_train)}   Test rows: {len(X_test)}")
    print(f"Train target balance: {y_train.value_counts().sort_index().to_dict()}")
    print(f"Test  target balance: {y_test.value_counts().sort_index().to_dict()}")
    print("Preprocessing: median-impute + StandardScaler (numeric), "
          "most-frequent-impute + OneHotEncoder (categorical)")

    # ---- Phases 4 & 5: train and evaluate multiple models ----------------
    print("\n" + "=" * 72)
    print("PHASE 4/5 - TRAIN & EVALUATE MULTIPLE MODELS")
    print("=" * 72)
    results: dict = {}
    pipelines: dict = {}
    for name, estimator in build_models().items():
        print(f"\n-- {name} --")
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        results[name] = evaluate(y_test, y_pred)
        pipelines[name] = pipeline
        print(classification_report(
            y_test, y_pred, target_names=[TARGET_LABELS[0], TARGET_LABELS[1]]
        ))

    comparison = pd.DataFrame(results).T.sort_values("f1", ascending=False)

    # ---- Phase 6: select the best model ----------------------------------
    print("=" * 72)
    print("PHASE 6 - SELECT THE BEST MODEL")
    print("=" * 72)
    print(comparison.to_string(float_format=lambda value: f"{value:.4f}"))
    best_name = str(comparison.index[0])
    best_pipeline = pipelines[best_name]
    print(f"\nBest model (highest F1): {best_name}")

    for index, (name, pipeline) in enumerate(pipelines.items(), start=1):
        plot_confusion_matrix(
            y_test,
            pipeline.predict(X_test),
            name,
            f"09_confusion_matrix_{index}.png",
        )

    # ---- Phase 7: save the model -----------------------------------------
    print("\n" + "=" * 72)
    print("PHASE 7 - SAVE THE MODEL")
    print("=" * 72)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Saved pipeline -> {MODEL_PATH.relative_to(BASE_DIR)}")

    metadata = {
        "model_name": best_name,
        "target_column": TARGET_COLUMN,
        "target_mapping": TARGET_MAPPING,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "categorical_options": {
            feature: sorted(df[feature].unique().tolist())
            for feature in CATEGORICAL_FEATURES
        },
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "trained_on_rows": int(len(X_train)),
        "metrics": results[best_name],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    METRICS_PATH.write_text(json.dumps(
        {
            "comparison": comparison.to_dict(orient="index"),
            "best_model": best_name,
        },
        indent=2,
    ))
    print(f"Saved metadata -> {METADATA_PATH.relative_to(BASE_DIR)}")
    print(f"Saved metrics  -> {METRICS_PATH.relative_to(BASE_DIR)}")
    print("\nDone. Start the API with:  uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()


