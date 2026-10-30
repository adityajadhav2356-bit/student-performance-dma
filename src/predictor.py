"""Inference & Prediction Pipeline for Student Academic Performance.

Supports single-student evaluation with real-time confidence scores and personalized
pedagogical advice, as well as batch CSV inference with enriched output.
Uses Decision Tree Classification only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

from src.model_tree import DecisionTreePerformanceClassifier

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class StudentPerformancePredictor:
    """Production inference engine for Decision Tree classification."""

    def __init__(self, models_dir: Path | str = DEFAULT_MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.tree_model_path = self.models_dir / "decision_tree_model.joblib"

        if not self.tree_model_path.exists():
            raise FileNotFoundError(
                f"Model artifacts missing in {self.models_dir}. Please run 'python train.py' first."
            )

        self.classifier = DecisionTreePerformanceClassifier.load(self.tree_model_path)
        self.feature_names = self.classifier.feature_names

    def generate_pedagogical_advice(
        self,
        attendance_rate: float,
        study_hours: float,
        previous_marks: float,
        assignment_scores: float,
        predicted_category: str
    ) -> List[str]:
        """Generate individualized, actionable academic recommendations."""
        advice = []

        if attendance_rate < 65.0:
            advice.append("⚠️ Attendance is critically low (<65%). Mandatory attendance counseling and weekly tracking advised.")
        elif attendance_rate < 75.0:
            advice.append("📅 Attendance is below recommended 75% college threshold. Increase lecture participation to avoid exam debarment.")
        else:
            advice.append("✅ Consistent attendance record (>=75%). Maintain regular classroom engagement.")

        if study_hours < 8.0:
            advice.append("⏰ Low self-study duration (<8 hrs/week). Dedicate at least 2 hours daily to structured revision.")
        elif study_hours < 18.0:
            advice.append("📖 Moderate study schedule. Formulate a focused weekend test-preparation routine.")
        else:
            advice.append("🌟 Strong self-study commitment (>18 hrs/week). Continue proactive problem-solving.")

        if assignment_scores < 50.0:
            advice.append("📝 Assignment performance indicates conceptual gaps. Schedule one-on-one TA/faculty doubt clearing.")
        elif assignment_scores >= 75.0:
            advice.append("🎯 High continuous evaluation score. Good mastery of assignment coursework.")

        if predicted_category == "Needs Improvement":
            advice.append("🚨 High Risk: Student requires targeted remedial tutorials before upcoming semester exams.")
        elif predicted_category == "Average":
            advice.append("💡 Moderate Range: Strengthen weak subjects and practice mock papers to elevate to 'Excellent'.")
        else:
            advice.append("🏆 Top Achiever: Nominate for advanced honors modules, hackathons, and student mentor roles.")

        return advice

    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run real-time inference on a single student record."""
        # Convert dictionary to DataFrame row
        row = pd.DataFrame([input_data])

        # Categorical encodings
        if "tutoring_encoded" not in row.columns:
            tutoring_val = row.get("tutoring", ["No"]).iloc[0]
            row["tutoring_encoded"] = 1 if str(tutoring_val).strip().lower() in ["yes", "1", "true"] else 0

        if "parent_edu_encoded" not in row.columns:
            edu_val = row.get("parent_education", ["Bachelor"]).iloc[0]
            edu_map = {"High School": 1, "Associate": 2, "Bachelor": 3, "Master": 4, "Doctorate": 5}
            row["parent_edu_encoded"] = edu_map.get(str(edu_val).strip(), 2)

        if "stress_encoded" not in row.columns:
            stress_val = row.get("stress_level", ["Moderate"]).iloc[0]
            stress_map = {"Low": 1, "Moderate": 2, "High": 3}
            row["stress_encoded"] = stress_map.get(str(stress_val).strip(), 2)

        # Fill missing features with reasonable defaults
        for feat in self.feature_names:
            if feat not in row.columns:
                row[feat] = 50.0

        X_input = row[self.feature_names].astype(float)

        # Classification prediction
        pred_class = self.classifier.predict(X_input)[0]
        probas = self.classifier.predict_proba(X_input)[0]
        class_labels = self.classifier.model.classes_

        proba_dict = {
            str(label): round(float(prob), 4)
            for label, prob in zip(class_labels, probas)
        }
        confidence = float(np.max(probas))

        # Risk level determination
        att = float(row["attendance_rate"].iloc[0])
        if pred_class == "Needs Improvement" or att < 60.0:
            risk_level = "High Risk"
            risk_color = "#EF4444"
        elif pred_class == "Average":
            risk_level = "Moderate"
            risk_color = "#F59E0B"
        else:
            risk_level = "Low Risk / High Achiever"
            risk_color = "#10B981"

        advice = self.generate_pedagogical_advice(
            attendance_rate=att,
            study_hours=float(row["study_hours_per_week"].iloc[0]),
            previous_marks=float(row["previous_marks"].iloc[0]),
            assignment_scores=float(row["assignment_scores"].iloc[0]),
            predicted_category=pred_class
        )

        return {
            "predicted_category": pred_class,
            "confidence": round(confidence, 4),
            "class_probabilities": proba_dict,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "personalized_advice": advice
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run batch predictions across an entire DataFrame."""
        out_df = df.copy()

        # Categorical encodings
        if "tutoring" in out_df.columns:
            out_df["tutoring_encoded"] = out_df["tutoring"].map(
                lambda x: 1 if str(x).strip().lower() in ["yes", "1", "true"] else 0
            )
        elif "tutoring_encoded" not in out_df.columns:
            out_df["tutoring_encoded"] = 0

        edu_map = {"High School": 1, "Associate": 2, "Bachelor": 3, "Master": 4, "Doctorate": 5}
        if "parent_education" in out_df.columns:
            out_df["parent_edu_encoded"] = out_df["parent_education"].map(edu_map).fillna(2).astype(int)
        elif "parent_edu_encoded" not in out_df.columns:
            out_df["parent_edu_encoded"] = 2

        stress_map = {"Low": 1, "Moderate": 2, "High": 3}
        if "stress_level" in out_df.columns:
            out_df["stress_encoded"] = out_df["stress_level"].map(stress_map).fillna(2).astype(int)
        elif "stress_encoded" not in out_df.columns:
            out_df["stress_encoded"] = 2

        # Validate required columns
        for col in self.feature_names:
            if col not in out_df.columns:
                out_df[col] = 50.0

        X = out_df[self.feature_names].astype(float)
        predictions = self.classifier.predict(X)
        probas = self.classifier.predict_proba(X)
        confidences = np.max(probas, axis=1)

        out_df["Predicted_Category"] = predictions
        out_df["Confidence"] = np.round(confidences, 3)

        # Risk flag
        out_df["Risk_Level"] = np.where(
            (out_df["Predicted_Category"] == "Needs Improvement") | (out_df.get("attendance_rate", 100.0) < 60.0),
            "High Risk",
            np.where(out_df["Predicted_Category"] == "Average", "Moderate Risk", "Low Risk")
        )

        return out_df


if __name__ == "__main__":
    predictor = StudentPerformancePredictor()
    sample_student = {
        "attendance_rate": 88.5,
        "study_hours_per_week": 22.0,
        "previous_marks": 82.0,
        "assignment_scores": 85.0,
        "midterm_marks": 80.0,
        "math_score": 84.0,
        "science_score": 86.0,
        "english_score": 78.0,
        "tutoring": "No",
        "parent_education": "Bachelor",
        "stress_level": "Low"
    }

    result = predictor.predict_single(sample_student)
    print("Single Student Inference Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
