"""Data Preprocessor Module for Student Performance Analysis & Prediction.

Handles missing values, duplicate removal, categorical encoding, feature scaling,
stratified train-test splitting, and data quality auditing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET_COLUMN = "performance_category"
CLASS_NAMES = ["Needs Improvement", "Average", "Excellent"]

# Core predictors used for classification
FEATURE_COLUMNS = [
    "attendance_rate",
    "study_hours_per_week",
    "previous_marks",
    "assignment_scores",
    "midterm_marks",
    "math_score",
    "science_score",
    "english_score",
    "tutoring_encoded",
    "parent_edu_encoded",
    "stress_encoded"
]

# Features used for K-Means unsupervised clustering
CLUSTERING_FEATURES = [
    "attendance_rate",
    "study_hours_per_week",
    "previous_marks",
    "assignment_scores",
    "final_marks"
]


@dataclass
class PreprocessedDataset:
    """Holds processed datasets and transformers."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    scaler: StandardScaler
    clustering_features: List[str]
    X_cluster_scaled: np.ndarray
    original_df: pd.DataFrame
    audit_report: Dict[str, any]


class StudentDataPreprocessor:
    """Comprehensive preprocessing and auditing engine for student data."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.cluster_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def audit_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """Perform automated data quality checks."""
        total_rows = len(df)
        missing_counts = df.isnull().sum().to_dict()
        duplicates_count = int(df.duplicated(subset=["student_id"]).sum()) if "student_id" in df.columns else int(df.duplicated().sum())

        numeric_summary = {}
        for col in ["attendance_rate", "study_hours_per_week", "previous_marks", "assignment_scores", "final_marks"]:
            if col in df.columns:
                numeric_summary[col] = {
                    "mean": float(np.round(df[col].mean(), 2)),
                    "median": float(np.round(df[col].median(), 2)),
                    "min": float(np.round(df[col].min(), 2)),
                    "max": float(np.round(df[col].max(), 2)),
                    "std": float(np.round(df[col].std(), 2)),
                }

        class_distribution = {}
        if TARGET_COLUMN in df.columns:
            class_distribution = df[TARGET_COLUMN].value_counts().to_dict()

        return {
            "total_rows": total_rows,
            "total_columns": len(df.columns),
            "duplicates_count": duplicates_count,
            "missing_values_total": int(sum(missing_counts.values())),
            "missing_per_column": {k: int(v) for k, v in missing_counts.items() if v > 0},
            "numeric_summary": numeric_summary,
            "class_distribution": class_distribution
        }

    def clean_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values, types, and remove duplicates."""
        df_clean = df.copy()

        # Remove duplicate records
        if "student_id" in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=["student_id"]).reset_index(drop=True)
        else:
            df_clean = df_clean.drop_duplicates().reset_index(drop=True)

        # Impute numeric columns with column median
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        # Impute categorical columns with column mode
        cat_cols = df_clean.select_dtypes(include=["object", "string"]).columns
        for col in cat_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

        return df_clean

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical factors for machine learning."""
        df_encoded = df.copy()

        # Tutoring: Yes -> 1, No -> 0
        if "tutoring" in df_encoded.columns:
            df_encoded["tutoring_encoded"] = df_encoded["tutoring"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
        else:
            df_encoded["tutoring_encoded"] = 0

        # Parent Education Ordinal Mapping
        edu_map = {
            "High School": 1,
            "Associate": 2,
            "Bachelor": 3,
            "Master": 4,
            "Doctorate": 5
        }
        if "parent_education" in df_encoded.columns:
            df_encoded["parent_edu_encoded"] = df_encoded["parent_education"].map(edu_map).fillna(2).astype(int)
        else:
            df_encoded["parent_edu_encoded"] = 2

        # Stress Level Ordinal Mapping
        stress_map = {"Low": 1, "Moderate": 2, "High": 3}
        if "stress_level" in df_encoded.columns:
            df_encoded["stress_encoded"] = df_encoded["stress_level"].map(stress_map).fillna(2).astype(int)
        else:
            df_encoded["stress_encoded"] = 2

        return df_encoded

    def prepare_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20
    ) -> PreprocessedDataset:
        """Run the complete end-to-end preprocessing pipeline."""
        audit_report = self.audit_data(df)
        df_clean = self.clean_and_impute(df)
        df_encoded = self.encode_features(df_clean)

        # Verify target column exists
        if TARGET_COLUMN not in df_encoded.columns:
            # Recreate category if missing
            conditions = [
                (df_encoded["final_marks"] >= 75.0),
                (df_encoded["final_marks"] >= 50.0) & (df_encoded["final_marks"] < 75.0),
                (df_encoded["final_marks"] < 50.0)
            ]
            choices = ["Excellent", "Average", "Needs Improvement"]
            df_encoded[TARGET_COLUMN] = np.select(conditions, choices, default="Average")

        # Feature matrix for classification
        X = df_encoded[FEATURE_COLUMNS].copy()
        y = df_encoded[TARGET_COLUMN].copy()

        # Stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )

        # Scale clustering features for K-Means
        cluster_data = df_clean[CLUSTERING_FEATURES].copy()
        X_cluster_scaled = self.cluster_scaler.fit_transform(cluster_data)

        return PreprocessedDataset(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=FEATURE_COLUMNS,
            scaler=self.cluster_scaler,
            clustering_features=CLUSTERING_FEATURES,
            X_cluster_scaled=X_cluster_scaled,
            original_df=df_clean,
            audit_report=audit_report
        )


if __name__ == "__main__":
    from src.data_loader import load_cleaned_dataset
    raw_df = load_cleaned_dataset()
    preprocessor = StudentDataPreprocessor()
    dataset = preprocessor.prepare_data(raw_df)
    print("Preprocessing completed!")
    print(f"X_train shape: {dataset.X_train.shape}, X_test shape: {dataset.X_test.shape}")
    print(f"Audit report: Total {dataset.audit_report['total_rows']} rows, duplicates: {dataset.audit_report['duplicates_count']}")
