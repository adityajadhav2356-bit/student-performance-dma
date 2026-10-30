"""Decision Tree Classification Module for Student Performance.

Implements training, hyperparameter tuning, evaluation, confusion matrix generation,
and feature importance analysis for student performance categorization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.tree import DecisionTreeClassifier, export_text

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
CLASS_ORDER = ["Needs Improvement", "Average", "Excellent"]


class DecisionTreePerformanceClassifier:
    """Decision Tree Classifier with evaluation metrics and persistence."""

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int = 5,
        min_samples_split: int = 6,
        min_samples_leaf: int = 4,
        random_state: int = 42
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.model = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        self.feature_names: List[str] = []
        self.is_trained: bool = False
        self.metrics: Dict[str, Any] = {}

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the decision tree classifier."""
        self.feature_names = list(X_train.columns)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"[DecisionTree] Model fitted with {len(self.feature_names)} features and depth={self.model.get_depth()}")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate the model on test data and compute real, verified metrics."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")

        y_pred = self.model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
        cm_norm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER, normalize="true")

        # Per-class classification report
        report_dict = classification_report(
            y_test, y_pred,
            labels=CLASS_ORDER,
            output_dict=True,
            zero_division=0
        )

        # Feature importances
        importances = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        self.metrics = {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_macro": round(rec_macro, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": np.round(cm_norm, 4).tolist(),
            "class_labels": CLASS_ORDER,
            "classification_report": report_dict,
            "feature_importances": importances.to_dict(orient="records"),
            "tree_depth": int(self.model.get_depth()),
            "n_leaves": int(self.model.get_n_leaves())
        }

        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict student performance categories."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
        return self.model.predict_proba(X)

    def get_text_rules(self) -> str:
        """Export tree rules in human-readable plain text."""
        if not self.is_trained:
            return "Model not trained."
        return export_text(self.model, feature_names=self.feature_names)

    def save_artifacts(
        self,
        models_dir: Path | str = DEFAULT_MODELS_DIR,
        reports_dir: Path | str = DEFAULT_REPORTS_DIR
    ) -> Tuple[Path, Path]:
        """Save trained model and metrics artifacts."""
        m_dir = Path(models_dir)
        r_dir = Path(reports_dir)
        m_dir.mkdir(parents=True, exist_ok=True)
        r_dir.mkdir(parents=True, exist_ok=True)

        model_file = m_dir / "decision_tree_model.joblib"
        metrics_file = m_dir / "tree_metrics.json"

        # Save model and feature names
        bundle = {
            "model": self.model,
            "feature_names": self.feature_names,
            "class_labels": CLASS_ORDER,
            "hyperparameters": {
                "criterion": self.criterion,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf
            }
        }
        joblib.dump(bundle, model_file)

        # Save metrics JSON
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=4)

        # Save confusion matrix plot
        if "confusion_matrix" in self.metrics:
            cm = np.array(self.metrics["confusion_matrix"])
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=CLASS_ORDER,
                yticklabels=CLASS_ORDER
            )
            plt.title("Decision Tree Confusion Matrix")
            plt.xlabel("Predicted Class")
            plt.ylabel("Actual Class")
            plt.tight_layout()
            cm_plot_file = r_dir / "confusion_matrix.png"
            plt.savefig(cm_plot_file, dpi=150)
            plt.close()

        print(f"[DecisionTree] Model saved to {model_file} and metrics to {metrics_file}")
        return model_file, metrics_file

    @classmethod
    def load(cls, model_path: Path | str) -> "DecisionTreePerformanceClassifier":
        """Load a persisted classifier bundle."""
        bundle = joblib.load(model_path)
        clf = cls(
            criterion=bundle["hyperparameters"]["criterion"],
            max_depth=bundle["hyperparameters"]["max_depth"],
            min_samples_split=bundle["hyperparameters"]["min_samples_split"],
            min_samples_leaf=bundle["hyperparameters"]["min_samples_leaf"]
        )
        clf.model = bundle["model"]
        clf.feature_names = bundle["feature_names"]
        clf.is_trained = True
        return clf


if __name__ == "__main__":
    from src.data_loader import load_cleaned_dataset
    from src.preprocessor import StudentDataPreprocessor

    df = load_cleaned_dataset()
    prep = StudentDataPreprocessor()
    dataset = prep.prepare_data(df)

    dt = DecisionTreePerformanceClassifier(max_depth=5)
    dt.train(dataset.X_train, dataset.y_train)
    metrics = dt.evaluate(dataset.X_test, dataset.y_test)
    dt.save_artifacts()

    print("Test Accuracy:", metrics["accuracy"])
    print("F1-Score (Weighted):", metrics["f1_weighted"])
    print("Classification Report:\n", pd.DataFrame(metrics["classification_report"]).T)
