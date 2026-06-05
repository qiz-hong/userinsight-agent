"""
Data Cleaning Module

Handles missing values, outlier detection, data type correction,
and duplicate removal.
"""

import pandas as pd
import numpy as np
import warnings

# Suppress dateutil parser warning when auto-detecting datetime columns
warnings.filterwarnings("ignore", message=".*Could not infer format.*")


class DataCleaner:
    """Clean and preprocess raw data.

    Supports chained calls:
        cleaner.drop_duplicates().fill_missing().correct_dtypes()
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._stats = {"before": len(df), "duplicates_dropped": 0, "nulls_filled": 0}

    def drop_duplicates(self) -> "DataCleaner":
        """Remove duplicate rows."""
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        self._stats["duplicates_dropped"] = before - len(self.df)
        if self._stats["duplicates_dropped"]:
            print(f"  Dropped {self._stats['duplicates_dropped']} duplicates")
        return self

    def fill_missing(self, strategy: str = "auto") -> "DataCleaner":
        """
        Fill missing values with adaptive strategy per column.
        - 'auto': median for numeric, mode for categorical (default)
        - 'median' / 'mean' / 'mode': apply to all
        - 'drop': drop rows with any NaN
        """
        if strategy == "drop":
            self.df = self.df.dropna()
            return self

        for col in self.df.columns:
            if self.df[col].isnull().sum() == 0:
                continue

            if strategy == "auto":
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    fill_val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(fill_val)
                else:
                    # Categorical / object: use mode
                    mode_vals = self.df[col].mode()
                    fill_val = mode_vals[0] if len(mode_vals) > 0 else "Unknown"
                    self.df[col] = self.df[col].fillna(fill_val)
            elif strategy == "median":
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].median())
            elif strategy == "mean":
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif strategy == "mode":
                mode_vals = self.df[col].mode()
                fill_val = mode_vals[0] if len(mode_vals) > 0 else "Unknown"
                self.df[col] = self.df[col].fillna(fill_val)

            self._stats["nulls_filled"] += 1

        nulls_remaining = self.df.isnull().sum().sum()
        print(f"  Nulls remaining after fill: {nulls_remaining}")
        return self

    def remove_outliers(self, columns: list = None, method: str = "iqr") -> "DataCleaner":
        """
        Remove outliers using IQR method.
        columns: list of column names (None = all numeric columns)
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        before = len(self.df)
        mask = pd.Series(True, index=self.df.index)

        for col in columns:
            if col not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            mask &= (self.df[col] >= lower) & (self.df[col] <= upper)

        self.df = self.df[mask].reset_index(drop=True)
        removed = before - len(self.df)
        if removed:
            print(f"  Removed {removed} outlier rows (IQR)")
        return self

    def correct_dtypes(self) -> "DataCleaner":
        """Auto-detect and correct column data types (datetime, numeric)."""
        for col in self.df.columns:
            series = self.df[col]
            if series.dtype != "object":
                continue

            # Try datetime
            try:
                converted = pd.to_datetime(series, errors="coerce")
                if converted.notna().sum() > 0.8 * len(series):
                    self.df[col] = converted
                    continue
            except Exception:
                pass

            # Try numeric
            try:
                converted = pd.to_numeric(series, errors="coerce")
                if converted.notna().sum() > 0.8 * len(series):
                    self.df[col] = converted
            except Exception:
                pass

        return self

    def get_clean_data(self) -> pd.DataFrame:
        """Return the cleaned DataFrame."""
        print(f"  Final: {len(self.df)} rows × {len(self.df.columns)} cols")
        return self.df
