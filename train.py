"""Master Training & Evaluation Pipeline for Student Performance DMA Project.

Runs end-to-end data acquisition, preprocessing, Decision Tree classification,
and K-Means clustering, generating all models and evaluation reports.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd

from src.data_loader import load_cleaned_dataset
from src.model_kmeans import KMeansStudentClusterer
from src.model_tree import DecisionTreePerformanceClassifier
from src.preprocessor import StudentDataPreprocessor

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"


def run_pipeline() -> None:
    """Execute complete end-to-end training and evaluation pipeline."""
    print("=" * 75)
    print("🎓 STUDENT PERFORMANCE ANALYSIS & PREDICTION - ML TRAINING PIPELINE")
    print("=" * 75)

    # Step 1: Load and Validate Dataset
    print("\n[Step 1/4] Loading and auditing student performance dataset...")
    df = load_cleaned_dataset()
    print(f"  ✓ Total Student Records: {len(df)}")
    print(f"  ✓ Features: {len(df.columns)}")
    print(f"  ✓ Target Class Distribution:")
    for cat, count in df["performance_category"].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"      - {cat}: {count} ({pct:.1f}%)")

    # Step 2: Data Preprocessing
    print("\n[Step 2/4] Preprocessing, encoding and splitting dataset (80/20 stratified)...")
    preprocessor = StudentDataPreprocessor(random_state=42)
    dataset = preprocessor.prepare_data(df, test_size=0.20)
    print(f"  ✓ Training Samples: {len(dataset.X_train)}")
    print(f"  ✓ Testing Samples: {len(dataset.X_test)}")
    print(f"  ✓ Classification Features ({len(dataset.feature_names)}): {', '.join(dataset.feature_names)}")

    # Step 3: Train & Evaluate Decision Tree Classifier
    print("\n[Step 3/4] Training Decision Tree Classifier...")
    tree_clf = DecisionTreePerformanceClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_split=6,
        min_samples_leaf=4,
        random_state=42
    )
    tree_clf.train(dataset.X_train, dataset.y_train)
    metrics = tree_clf.evaluate(dataset.X_test, dataset.y_test)
    tree_clf.save_artifacts(models_dir=MODELS_DIR, reports_dir=REPORTS_DIR)

    print("\n  --- Decision Tree Evaluation Metrics ---")
    print(f"  ✓ Test Accuracy:        {metrics['accuracy'] * 100:.2f}%")
    print(f"  ✓ Precision (Weighted): {metrics['precision_weighted'] * 100:.2f}%")
    print(f"  ✓ Recall (Weighted):    {metrics['recall_weighted'] * 100:.2f}%")
    print(f"  ✓ F1-Score (Weighted):  {metrics['f1_weighted'] * 100:.2f}%")
    print(f"  ✓ Tree Depth:           {metrics['tree_depth']}")

    print("\n  Classification Report per Performance Category:")
    report_df = pd.DataFrame(metrics["classification_report"]).T
    print(report_df.loc[["Needs Improvement", "Average", "Excellent"], ["precision", "recall", "f1-score", "support"]])

    print("\n  Top 5 Predictive Features:")
    for item in metrics["feature_importances"][:5]:
        print(f"      - {item['feature']}: {item['importance'] * 100:.2f}%")

    # Step 4: Train & Profile K-Means Clustering
    print("\n[Step 4/4] Performing K-Means Clustering & Elbow Method Analysis...")
    clusterer = KMeansStudentClusterer(n_clusters=3, random_state=42)
    elbow_data = clusterer.compute_elbow_curve(df, k_min=2, k_max=7)
    labels, df_clustered = clusterer.fit_predict(df)
    clusterer.save_artifacts(models_dir=MODELS_DIR, reports_dir=REPORTS_DIR)

    print("\n  --- Elbow Method & Silhouette Scores ---")
    elbow_df = pd.DataFrame(elbow_data)
    for _, row in elbow_df.iterrows():
        print(f"      k={int(row['k'])} -> Inertia: {row['inertia']:.1f} | Silhouette Score: {row['silhouette_score']:.4f}")

    print("\n  --- Identified Academic Cluster Personas ---")
    for prof in clusterer.cluster_profiles:
        print(f"\n  [Cluster {prof['cluster_id']}] {prof['persona_name']} ({prof['badge']})")
        print(f"      Student Count: {prof['student_count']} ({prof['student_percentage']}%)")
        print(f"      Avg Attendance: {prof['metrics']['avg_attendance']}% | Avg Marks: {prof['metrics']['avg_final_marks']}")
        print(f"      Key Action: {prof['recommended_intervention']}")

    print("\n" + "=" * 75)
    print("✅ MASTER TRAINING PIPELINE COMPLETE! ALL MODELS SAVED.")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()
