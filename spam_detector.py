import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline


DEFAULT_DATA_PATH = "spam.csv"
DEFAULT_MODEL_PATH = "spam_model.joblib"


@dataclass
class TrainResult:
    model_path: str
    test_accuracy: float
    test_f1_spam: float
    test_auc: float


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load and normalize the SMS Spam dataset.

    The dataset variant commonly has columns: v1 (label), v2 (text), plus empty columns.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    # latin-1 handles the dataset's special characters (e.g., £)
    df = pd.read_csv(csv_path, encoding="latin-1")

    # Expect columns v1, v2, ... keep only needed
    if "v1" not in df.columns or "v2" not in df.columns:
        raise ValueError(
            "Expected columns 'v1' (label) and 'v2' (text) not found in CSV."
        )

    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})

    # Drop empties and duplicates
    df = df.dropna(subset=["label", "text"])  # type: ignore[arg-type]
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[(df["label"].isin(["ham", "spam"])) & (df["text"].str.len() > 0)]
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    X = df["text"]
    y = df["label"]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


def build_pipeline() -> Pipeline:
    # Tfidf with uni+bi-grams captures short SMS patterns; balanced LR handles class imbalance
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    n_jobs=None,
                    solver="liblinear",
                ),
            ),
        ]
    )
    return pipeline


def tune_hyperparameters(pipeline: Pipeline, X_train: pd.Series, y_train: pd.Series, seed: int = 42) -> Pipeline:
    param_grid = {
        "tfidf__min_df": [1, 2, 3],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.5, 1.0, 2.0, 4.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # Optimize for F1 on the spam class
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=make_scorer(f1_score, pos_label="spam"),
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_  # type: ignore[return-value]


def evaluate(model: Pipeline, X_test: pd.Series, y_test: pd.Series) -> Tuple[float, float, float]:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_spam = f1_score(y_test, y_pred, pos_label="spam")

    # AUC requires probability estimates; handle if not available
    auc = np.nan
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        classes = list(model.classes_)  # type: ignore[attr-defined]
        spam_index = classes.index("spam") if "spam" in classes else None
        if spam_index is not None:
            auc = roc_auc_score((y_test == "spam").astype(int), proba[:, spam_index])

    print("\n==== Evaluation on Hold-out Test ====")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {acc:.4f} | F1(spam): {f1_spam:.4f} | AUC: {auc:.4f}")

    return acc, f1_spam, auc


def train_and_save(
    data_path: str = DEFAULT_DATA_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    seed: int = 42,
) -> TrainResult:
    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, seed=seed)

    base_pipeline = build_pipeline()
    model = tune_hyperparameters(base_pipeline, X_train, y_train, seed=seed)
    # Fit best model on full training split
    model.fit(X_train, y_train)

    acc, f1_spam, auc = evaluate(model, X_test, y_test)

    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    return TrainResult(model_path=model_path, test_accuracy=acc, test_f1_spam=f1_spam, test_auc=auc)


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> Pipeline:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train it first with --train."
        )
    model = joblib.load(model_path)
    return model


def predict_text(message: str, model_path: str = DEFAULT_MODEL_PATH) -> Tuple[str, float]:
    model = load_model(model_path)
    pred = model.predict([message])[0]
    spam_prob = 0.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([message])[0]
        classes = list(model.classes_)  # type: ignore[attr-defined]
        if "spam" in classes:
            spam_prob = float(proba[classes.index("spam")])
    return str(pred), float(spam_prob)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SMS Spam Detector")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model on spam.csv and save to a file",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Predict a single message passed as a string",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to the spam CSV dataset",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to save/load the trained model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.train:
        train_and_save(data_path=args.data_path, model_path=args.model_path, seed=args.seed)

    if args.predict is not None:
        label, prob = predict_text(args.predict, model_path=args.model_path)
        print(f"Prediction: {label} | Spam probability: {prob:.4f}")

    if not args.train and args.predict is None:
        # If no action specified, show quick help hint
        parser.print_help()


if __name__ == "__main__":
    main()


