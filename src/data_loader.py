"""
Data Loading Module

Loads CSV/Excel data from the data/ directory and performs basic
schema validation and preview.
"""

import pandas as pd
from pathlib import Path


class DataLoader:
    """Load and validate input data files."""

    # Encodings to try for CSV files (most common first)
    _CSV_ENCODINGS = ["utf-8", "gbk", "gb2312", "latin-1", "iso-8859-1"]

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.df: pd.DataFrame = None
        self._loaded = False

    def load(self) -> pd.DataFrame:
        """Load data from file (CSV or Excel) with auto-encoding detection."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        suffix = self.file_path.suffix.lower()

        if suffix == ".csv":
            self.df = self._load_csv()
        elif suffix in (".xlsx", ".xls"):
            self.df = pd.read_excel(self.file_path)
        else:
            raise ValueError(
                f"Unsupported format: {suffix}. Use .csv or .xlsx"
            )

        if self.df.empty:
            raise ValueError("Loaded data is empty.")

        self._loaded = True
        print(f"  Loaded: {len(self.df)} rows × {len(self.df.columns)} columns")
        return self.df

    def _load_csv(self) -> pd.DataFrame:
        """Try multiple encodings to load CSV."""
        for enc in self._CSV_ENCODINGS:
            try:
                df = pd.read_csv(self.file_path, encoding=enc)
                print(f"  Encoding detected: {enc}")
                return df
            except (UnicodeDecodeError, UnicodeError):
                continue
        # Last resort
        raise ValueError(
            f"Could not decode CSV with any of: {self._CSV_ENCODINGS}"
        )

    def preview(self, n: int = 5) -> pd.DataFrame:
        """Return the first n rows for preview."""
        self._check_loaded()
        return self.df.head(n)

    def info(self) -> str:
        """Return schema and basic statistics of the dataset."""
        self._check_loaded()
        buf = []
        buf.append(f"Shape: {self.df.shape[0]} rows × {self.df.shape[1]} cols")
        buf.append(f"\nColumn dtypes:")
        for col, dtype in self.df.dtypes.items():
            null_count = self.df[col].isnull().sum()
            buf.append(f"  {col}: {dtype}  (nulls: {null_count})")
        return "\n".join(buf)

    def _check_loaded(self):
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load() first.")
