"""
Feature Engineering Module

Generates derived features from raw data including:
- Categorical encoding (one-hot, label)
- Numerical scaling (standard, min-max)
- Time-based features (recency, frequency)
- Behavioral features (interaction counts, ratios)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


class FeatureEngineer:
    """Generate and transform features for analysis and modeling.

    Main entry: auto_process() — auto-detects column types and applies
    appropriate transformations.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._id_cols = []          # Columns dropped (e.g., user_id with unique values)
        self._categorical_cols = [] # Original categorical columns
        self._numeric_cols = []     # Numeric columns
        self._datetime_cols = []    # Datetime columns found

    # ── Public API ────────────────────────────────────────────

    def auto_process(self, id_column: str = None) -> "FeatureEngineer":
        """
        Automatically detect column types and apply processing:
          1. Drop ID-like columns (unique per row)
          2. Datetime → time features (year, month, day, dayofweek)
          3. Categorical → one-hot (≤10 cats) or label encoding (>10)
          4. Numeric → Standard scaling
        """
        self._detect_columns(id_column)

        if self._id_cols:
            print(f"  Dropped ID columns: {self._id_cols}")
            self.df = self.df.drop(columns=self._id_cols)

        for col in self._datetime_cols:
            self._create_time_features(col)

        if self._categorical_cols:
            self._encode_categorical(self._categorical_cols)

        if self._numeric_cols:
            self._scale_numerical(self._numeric_cols)

        print(f"  Features after engineering: {self.df.shape[1]} columns")
        return self

    def get_features(self) -> pd.DataFrame:
        """Return the engineered feature DataFrame."""
        return self.df

    # ── Internal: column detection ───────────────────────────

    def _detect_columns(self, id_column: str = None):
        """Classify each column as id / datetime / categorical / numeric."""
        for col in self.df.columns:
            series = self.df[col]
            n_unique = series.nunique()

            # Explicit ID column
            if id_column and col == id_column:
                self._id_cols.append(col)
                continue

            # Auto-detect ID: all unique and dtype is object/int
            if n_unique == len(series) and n_unique > 50:
                self._id_cols.append(col)
                continue

            # Datetime
            if pd.api.types.is_datetime64_any_dtype(series):
                self._datetime_cols.append(col)
                continue

            # Categorical (object or low-unique int)
            if series.dtype == "object" or series.dtype == "category":
                self._categorical_cols.append(col)
            elif pd.api.types.is_integer_dtype(series) and n_unique <= 20:
                self._categorical_cols.append(col)
            elif pd.api.types.is_numeric_dtype(series):
                self._numeric_cols.append(col)
            else:
                # Fallback: treat as categorical
                self._categorical_cols.append(col)

        print(f"  Detected: {len(self._datetime_cols)} datetime, "
              f"{len(self._categorical_cols)} categorical, "
              f"{len(self._numeric_cols)} numeric")

    # ── Internal: transformations ─────────────────────────────

    def _create_time_features(self, col: str, reference_date=None):
        """Extract year, month, day, dayofweek from a datetime column."""
        series = pd.to_datetime(self.df[col], errors="coerce")
        prefix = col
        self.df[f"{prefix}_year"] = series.dt.year.fillna(0).astype(int)
        self.df[f"{prefix}_month"] = series.dt.month.fillna(0).astype(int)
        self.df[f"{prefix}_day"] = series.dt.day.fillna(0).astype(int)
        self.df[f"{prefix}_dow"] = series.dt.dayofweek.fillna(0).astype(int)

        # Recency: days since most recent date
        if reference_date is None:
            reference_date = series.max()
        if pd.notna(reference_date):
            self.df[f"{prefix}_recency"] = (
                (reference_date - series).dt.days.fillna(0).astype(int)
            )

        self.df = self.df.drop(columns=[col])

    def _encode_categorical(self, columns: list):
        """One-hot for low-cardinality (≤10), label for high-cardinality."""
        for col in columns:
            if col not in self.df.columns:
                continue
            n_unique = self.df[col].nunique()

            if n_unique <= 10:
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df = self.df.drop(columns=[col])
            else:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))

    def _scale_numerical(self, columns: list):
        """Standard-scale numeric columns."""
        valid_cols = [c for c in columns if c in self.df.columns]
        if not valid_cols:
            return
        scaler = StandardScaler()
        self.df[valid_cols] = scaler.fit_transform(self.df[valid_cols].fillna(0))
