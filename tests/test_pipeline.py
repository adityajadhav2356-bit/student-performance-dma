"""Automated Test Suite for Student Performance Analysis & Prediction Pipeline.

Verifies dataset integrity, preprocessing, Decision Tree classification,
K-Means clustering, and inference prediction contracts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_cleaned_dataset
from src.model_kmeans import KMeansStudentClusterer
from src.model_tree import DecisionTreePerformanceClassifier
from src.predictor import StudentPerformancePredictor
from src.preprocessor import StudentDataPreprocessor


@pytest.fixture(scope="session")
def dataset_df() -> pd.DataFrame:
    """Fixture providing clean student dataframe."""
    return load_cleaned_dataset()


def test_dataset_integrity(dataset_df: pd.DataFrame):
    """Test dataset structure, column presence, and non-emptiness."""
    assert len(dataset_df) >= 500, "Dataset should have at least 500 records"

    required_cols = [
        "student_id", "attendance_rate", "study_hours_per_week",
        "previous_marks", "assignment_scores", "midterm_marks",
        "final_marks", "performance_category", "passed", "needs_academic_support"
    ]
    for col in required_cols:
        assert col in dataset_df.columns, f"Missing required column: {col}"

    # Check for zero duplicates
    assert dataset_df["student_id"].duplicated().sum() == 0, "Duplicate student IDs detected"

    # Check realistic value ranges
    assert dataset_df["attendance_rate"].min() >= 30.0
    assert dataset_df["attendance_rate"].max() <= 100.0
    assert dataset_df["final_marks"].min() >= 0.0
    assert dataset_df["final_marks"].max() <= 100.0


def test_preprocessing_and_split(dataset_df: pd.DataFrame):
    """Test preprocessing, categorical encoding, and stratified split."""
    preprocessor = StudentDataPreprocessor(random_state=42)
    dataset = preprocessor.prepare_data(dataset_df, test_size=0.20)

    assert dataset.X_train.shape[0] == int(len(dataset_df) * 0.8)
    assert dataset.X_test.shape[0] == int(len(dataset_df) * 0.2)
    assert len(dataset.feature_names) == 11

    # Check that all target categories exist in both train and test sets
    train_classes = set(dataset.y_train.unique())
    test_classes = set(dataset.y_test.unique())
    expected_classes = {"Needs Improvement", "Average", "Excellent"}

    assert train_classes == expected_classes
    assert test_classes == expected_classes


def test_decision_tree_classifier(dataset_df: pd.DataFrame):
    """Test Decision Tree training, metrics calculation, and test accuracy."""
    preprocessor = StudentDataPreprocessor(random_state=42)
    dataset = preprocessor.prepare_data(dataset_df, test_size=0.20)

    clf = DecisionTreePerformanceClassifier(max_depth=5, random_state=42)
    clf.train(dataset.X_train, dataset.y_train)
    metrics = clf.evaluate(dataset.X_test, dataset.y_test)

    assert metrics["accuracy"] >= 0.75, f"Accuracy too low: {metrics['accuracy']}"
    assert metrics["f1_weighted"] >= 0.75, f"F1 too low: {metrics['f1_weighted']}"
    assert len(metrics["confusion_matrix"]) == 3
    assert len(metrics["feature_importances"]) > 0

    # Predictions
    sample = dataset.X_test.iloc[:5]
    preds = clf.predict(sample)
    assert len(preds) == 5
    for p in preds:
        assert p in ["Needs Improvement", "Average", "Excellent"]


def test_kmeans_clustering(dataset_df: pd.DataFrame):
    """Test K-Means clustering, elbow method curve, and persona generation."""
    clusterer = KMeansStudentClusterer(n_clusters=3, random_state=42)
    elbow_results = clusterer.compute_elbow_curve(dataset_df, k_min=2, k_max=5)

    assert len(elbow_results) == 4
    # Inertia must decrease monotonically as k increases
    inertias = [r["inertia"] for r in elbow_results]
    assert inertias == sorted(inertias, reverse=True), "Inertia should decrease as k increases"

    labels, clustered_df = clusterer.fit_predict(dataset_df)
    assert len(labels) == len(dataset_df)
    assert set(np.unique(labels)) == {0, 1, 2}
    assert "cluster_name" in clustered_df.columns
    assert len(clusterer.cluster_profiles) == 3


def test_inference_predictor():
    """Test real-time single student and batch CSV inference."""
    predictor = StudentPerformancePredictor()

    sample_student = {
        "attendance_rate": 88.5,
        "study_hours_per_week": 20.0,
        "sleep_hours": 7.5,
        "previous_marks": 82.0,
        "assignment_scores": 85.0,
        "midterm_marks": 82.0,
        "math_score": 85.0,
        "science_score": 86.0,
        "english_score": 80.0,
        "tutoring": "No",
        "parent_education": "Bachelor",
        "stress_level": "Low"
    }

    res = predictor.predict_single(sample_student)
    assert "predicted_category" in res
    assert res["predicted_category"] in ["Needs Improvement", "Average", "Excellent"]
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert "cluster_id" in res
    assert "personalized_advice" in res
    assert len(res["personalized_advice"]) > 0

    # Batch prediction test
    batch_df = pd.DataFrame([sample_student, sample_student])
    batch_out = predictor.predict_batch(batch_df)
    assert "Predicted_Category" in batch_out.columns
    assert "Cluster_Persona" in batch_out.columns
    assert "Risk_Level" in batch_out.columns
    assert len(batch_out) == 2
