"""
Behavior Prediction Module

Predicts user behavior including:
- Churn prediction (logistic regression / random forest)
- Next-action prediction (classification)
- Outputs prediction results and feature importance
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class BehaviorPredictor:
    """Predict user churn / behavior using ML models.

    Main entry: auto_predict() — auto-selects target, features, and best model.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.model = None
        self.model_name: str = ""
        self.feature_importance_: pd.DataFrame = None
        self.metrics_: dict = {}
        self._feature_cols: list = []
        self._target_col: str = ""

    # ── Main API ─────────────────────────────────────────────

    def auto_predict(
        self, target_column: str = None, segment_labels: np.ndarray = None
    ) -> "BehaviorPredictor":
        """
        Auto-select target, features, and best model.

        Priority for target:
          1. Explicit target_column if it exists in data
          2. segment_labels → treat smallest segment as "churn" class
          3. Auto-detect binary column in data
        """
        # --- Resolve target ---
        y = self._resolve_target(target_column, segment_labels)
        if y is None:
            print("  No target found; skipping prediction.")
            return self

        # --- Select features ---
        self._feature_cols = self.df.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        if self._target_col in self._feature_cols:
            self._feature_cols.remove(self._target_col)
        if not self._feature_cols:
            print("  No numeric features for prediction.")
            return self

        X = self.df[self._feature_cols].fillna(0).values
        print(f"  Predicting '{self._target_col}' from {len(self._feature_cols)} features")

        # --- Train/test split ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
        )

        # --- Auto-select best model ---
        self._auto_select_model(X_train, y_train, X_test, y_test)

        # --- Feature importance ---
        self._compute_importance()

        return self

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance sorted descending."""
        if self.feature_importance_ is None:
            return pd.DataFrame()
        return self.feature_importance_

    def get_metrics(self) -> dict:
        """Return evaluation metrics."""
        return self.metrics_

    # ── Internal ──────────────────────────────────────────────

    def _resolve_target(
        self, target_column: str = None, segment_labels: np.ndarray = None
    ) -> np.ndarray:
        """Resolve the target variable for prediction."""
        # 1. Explicit target column
        if target_column and target_column in self.df.columns:
            self._target_col = target_column
            return self.df[target_column].values

        # 2. From segmentation labels
        if segment_labels is not None and len(segment_labels) == len(self.df):
            self._target_col = "churn_risk"
            # Smallest segment → class 1 (churn), others → class 0
            unique, counts = np.unique(segment_labels, return_counts=True)
            smallest = unique[np.argmin(counts)]
            return (segment_labels == smallest).astype(int)

        # 3. Auto-detect binary column
        for col in self.df.columns:
            if self.df[col].nunique() == 2:
                self._target_col = col
                print(f"  Auto-detected target column: '{col}'")
                return self.df[col].astype(int).values

        return None

    def _auto_select_model(self, X_train, y_train, X_test, y_test):
        """Try RF and LR, pick the one with higher cross-val score."""
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_scores = cross_val_score(rf, X_train, y_train, cv=3, scoring="accuracy")
            rf_score = rf_scores.mean()
        except Exception:
            rf_score = 0

        try:
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr_scores = cross_val_score(lr, X_train, y_train, cv=3, scoring="accuracy")
            lr_score = lr_scores.mean()
        except Exception:
            lr_score = 0

        if rf_score >= lr_score:
            self.model = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
            self.model_name = "RandomForest"
        else:
            self.model = LogisticRegression(max_iter=1000, random_state=42)
            self.model_name = "LogisticRegression"

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        self.metrics_ = {
            "model": self.model_name,
            "accuracy": round(acc, 4),
            "cv_mean": round(max(rf_score, lr_score), 4),
            "test_samples": len(y_test),
        }
        print(f"  Best model: {self.model_name}  (accuracy={acc:.3f})")

    def _compute_importance(self):
        """Extract feature importance from the model."""
        if self.model is None or not self._feature_cols:
            return

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            return

        df = pd.DataFrame({
            "feature": self._feature_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        self.feature_importance_ = df
