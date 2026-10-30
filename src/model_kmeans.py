"""K-Means Clustering Module for Student Academic Segmentation.

Implements the Elbow Method, Silhouette score analysis, K-Means clustering,
centroid interpretation, and actionable pedagogical cluster personas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

CLUSTERING_FEATURES = [
    "attendance_rate",
    "study_hours_per_week",
    "previous_marks",
    "assignment_scores",
    "final_marks"
]


class KMeansStudentClusterer:
    """K-Means clustering engine with Elbow method and cluster profiling."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: KMeans | None = None
        self.clustering_features = CLUSTERING_FEATURES
        self.elbow_results: List[Dict[str, float]] = []
        self.cluster_profiles: List[Dict[str, Any]] = []
        self.is_trained: bool = False

    def compute_elbow_curve(
        self,
        X_df: pd.DataFrame,
        k_min: int = 2,
        k_max: int = 7
    ) -> List[Dict[str, float]]:
        """Calculate Inertia and Silhouette scores across a range of k values."""
        X_scaled = self.scaler.fit_transform(X_df[self.clustering_features])
        self.elbow_results = []
        sample_n = min(600, len(X_scaled))

        print(f"[KMeans] Computing Elbow curve from k={k_min} to k={k_max}...", flush=True)
        for k in range(k_min, k_max + 1):
            km = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=self.random_state)
            km.fit(X_scaled)
            inertia = float(km.inertia_)
            sil = float(silhouette_score(X_scaled, km.labels_, sample_size=sample_n, random_state=self.random_state))
            self.elbow_results.append({
                "k": int(k),
                "inertia": round(inertia, 2),
                "silhouette_score": round(sil, 4)
            })
            print(f"  k={k}: Inertia={inertia:.1f}, Silhouette={sil:.4f}", flush=True)

        return self.elbow_results

    def fit_predict(
        self,
        df: pd.DataFrame,
        n_clusters: int | None = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Fit K-Means and assign cluster labels to students."""
        if n_clusters is not None:
            self.n_clusters = n_clusters

        X_data = df[self.clustering_features].copy()
        X_scaled = self.scaler.fit_transform(X_data)

        self.model = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=15,
            random_state=self.random_state
        )
        labels = self.model.fit_predict(X_scaled)
        self.is_trained = True

        df_clustered = df.copy()
        df_clustered["cluster"] = labels

        # Generate cluster personas based on average final_marks ranking
        self._generate_cluster_personas(df_clustered)

        # Map cluster IDs to human-readable names
        cluster_name_map = {p["cluster_id"]: p["persona_name"] for p in self.cluster_profiles}
        df_clustered["cluster_name"] = df_clustered["cluster"].map(cluster_name_map)

        return labels, df_clustered

    def _generate_cluster_personas(self, df_clustered: pd.DataFrame) -> None:
        """Derive educational personas and recommendations from cluster centroids."""
        self.cluster_profiles = []
        total_students = len(df_clustered)

        # Group by cluster to analyze unscaled metrics
        grouped = df_clustered.groupby("cluster")
        mean_marks = grouped["final_marks"].mean().sort_values(ascending=False)

        # Map highest average mark to High Achievers, lowest to At-Risk, middle to Moderate
        rank_order = list(mean_marks.index)

        persona_templates = {
            "top": {
                "name": "High Achievers & Disciplined Learners",
                "badge": "Top Tier",
                "color": "#10B981",  # Emerald
                "description": "Exemplary academic records characterized by consistent 85%+ attendance, diligent study habits, and top assignment and exam scores.",
                "intervention": "Advanced enrichment programs, honors research, leadership roles, and peer tutoring opportunities."
            },
            "middle": {
                "name": "Steady Performers with Untapped Potential",
                "badge": "Core Group",
                "color": "#3B82F6",  # Blue
                "description": "Consistent attendance (72-82%) and solid grasp of fundamental concepts, but room to improve in self-study discipline and exam confidence.",
                "intervention": "Targeted workshops in problem-solving, structured study time management, and mock test revisions."
            },
            "bottom": {
                "name": "At-Risk Students Requiring Immediate Support",
                "badge": "Urgent Attention",
                "color": "#EF4444",  # Red
                "description": "Struggling with attendance irregularities (<65%), limited study hours (<10 hrs/wk), and below-par assignment and test scores.",
                "intervention": "Mandatory counseling, personalized remedial tutorial classes, weekly progress tracking, and parent-faculty coordination."
            }
        }

        for rank_idx, cluster_id in enumerate(rank_order):
            c_df = df_clustered[df_clustered["cluster"] == cluster_id]
            count = len(c_df)
            pct = round((count / total_students) * 100, 1)

            if rank_idx == 0:
                template = persona_templates["top"]
            elif rank_idx == len(rank_order) - 1:
                template = persona_templates["bottom"]
            else:
                template = persona_templates["middle"]

            self.cluster_profiles.append({
                "cluster_id": int(cluster_id),
                "rank": rank_idx + 1,
                "persona_name": template["name"],
                "badge": template["badge"],
                "color": template["color"],
                "student_count": count,
                "student_percentage": pct,
                "description": template["description"],
                "recommended_intervention": template["intervention"],
                "metrics": {
                    "avg_attendance": round(float(c_df["attendance_rate"].mean()), 1),
                    "avg_study_hours": round(float(c_df["study_hours_per_week"].mean()), 1),
                    "avg_previous_marks": round(float(c_df["previous_marks"].mean()), 1),
                    "avg_assignment_scores": round(float(c_df["assignment_scores"].mean()), 1),
                    "avg_final_marks": round(float(c_df["final_marks"].mean()), 1),
                    "pass_rate": round(float((c_df["final_marks"] >= 50.0).mean() * 100), 1),
                }
            })

        # Re-sort profiles by cluster_id for consistent lookup
        self.cluster_profiles.sort(key=lambda x: x["cluster_id"])

    def predict_cluster(self, sample_features: pd.DataFrame | np.ndarray) -> int:
        """Assign an individual student to the closest K-Means cluster."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Clusterer not fitted.")
        if isinstance(sample_features, pd.DataFrame):
            sample_scaled = self.scaler.transform(sample_features[self.clustering_features])
        else:
            sample_scaled = self.scaler.transform(sample_features)
        return int(self.model.predict(sample_scaled)[0])

    def save_artifacts(
        self,
        models_dir: Path | str = DEFAULT_MODELS_DIR,
        reports_dir: Path | str = DEFAULT_REPORTS_DIR
    ) -> Tuple[Path, Path]:
        """Save fitted clusterer, scaler, and summary metadata."""
        m_dir = Path(models_dir)
        r_dir = Path(reports_dir)
        m_dir.mkdir(parents=True, exist_ok=True)
        r_dir.mkdir(parents=True, exist_ok=True)

        model_file = m_dir / "kmeans_model.joblib"
        summary_file = m_dir / "kmeans_summary.json"

        bundle = {
            "model": self.model,
            "scaler": self.scaler,
            "clustering_features": self.clustering_features,
            "n_clusters": self.n_clusters,
            "cluster_profiles": self.cluster_profiles,
            "elbow_results": self.elbow_results
        }
        joblib.dump(bundle, model_file)

        with open(summary_file, "w") as f:
            json.dump({
                "n_clusters": self.n_clusters,
                "features": self.clustering_features,
                "elbow_results": self.elbow_results,
                "cluster_profiles": self.cluster_profiles
            }, f, indent=4)

        print(f"[KMeans] Clusterer saved to {model_file} and summary to {summary_file}")
        return model_file, summary_file

    @classmethod
    def load(cls, model_path: Path | str) -> "KMeansStudentClusterer":
        """Load a persisted clusterer bundle."""
        bundle = joblib.load(model_path)
        clusterer = cls(n_clusters=bundle["n_clusters"])
        clusterer.model = bundle["model"]
        clusterer.scaler = bundle["scaler"]
        clusterer.clustering_features = bundle["clustering_features"]
        clusterer.cluster_profiles = bundle["cluster_profiles"]
        clusterer.elbow_results = bundle.get("elbow_results", [])
        clusterer.is_trained = True
        return clusterer


if __name__ == "__main__":
    from src.data_loader import load_cleaned_dataset

    df = load_cleaned_dataset()
    km = KMeansStudentClusterer(n_clusters=3)
    elbow_data = km.compute_elbow_curve(df, k_min=2, k_max=7)
    labels, df_clustered = km.fit_predict(df)
    km.save_artifacts()

    print("\nElbow Method Results:")
    print(pd.DataFrame(elbow_data))
    print("\nCluster Profiles:")
    for prof in km.cluster_profiles:
        print(f"\nCluster {prof['cluster_id']} [{prof['badge']}]: {prof['persona_name']}")
        print(f"  Students: {prof['student_count']} ({prof['student_percentage']}%)")
        print(f"  Avg Marks: {prof['metrics']['avg_final_marks']}, Avg Attendance: {prof['metrics']['avg_attendance']}%")
