"""
Data Loading Module

Loads CSV/Excel data from the data/ directory and performs basic
schema validation, column-name sanitization, and preview.
"""

import re
import pandas as pd
from pathlib import Path


class DataLoader:
    """Load and validate input data files.

    Hardens against common dirty-data edge cases:
    - unknown encodings (multi-encoding fallback chain)
    - duplicate / blank / hostile column names (sanitized)
    - fully-null rows and columns (dropped with warning)
    - empty files (clear error)
    """

    # Encodings to try for CSV files (most common first)
    _CSV_ENCODINGS = ["utf-8-sig", "utf-8", "gbk", "gb2312",
                      "latin-1", "iso-8859-1"]

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

        # ── Edge cases ────────────────────────────────────
        if self.df is None or self.df.empty or len(self.df.columns) == 0:
            raise ValueError("Loaded data is empty (0 rows or 0 columns).")

        self.df = self._sanitize_columns(self.df)
        self.df = self._drop_all_null_axis(self.df)

        if self.df.empty or len(self.df.columns) == 0:
            raise ValueError("All data was null — nothing to analyze.")

        self._loaded = True
        print(f"  Loaded: {len(self.df)} rows × {len(self.df.columns)} columns")
        return self.df

    def _load_csv(self) -> pd.DataFrame:
        """Try multiple encodings to load CSV."""
        last_err = None
        for enc in self._CSV_ENCODINGS:
            try:
                df = pd.read_csv(self.file_path, encoding=enc)
                print(f"  Encoding detected: {enc}")
                return df
            except (UnicodeDecodeError, UnicodeError) as e:
                last_err = e
                continue
            except pd.errors.EmptyDataError:
                raise ValueError("CSV file is empty.")
        raise ValueError(
            f"Could not decode CSV with any of: {self._CSV_ENCODINGS} "
            f"(last error: {last_err})"
        )

    @staticmethod
    def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Dedupe, deblank, and strip hostile characters from column
        names so they are safe for plotting, Excel sheets, and LLM
        prompts (mitigates prompt-injection via column names)."""
        renamed = {}
        seen = {}
        for col in df.columns:
            name = str(col).strip()
            # Blank column name -> Unnamed_n
            if not name or name.lower() == "unnamed":
                name = f"Unnamed_{df.columns.get_loc(col)}"
            # Keep alphanumerics, CJK, and a few safe separators
            name = re.sub(r"[^\w\u4e00-\u9fff\.\-\s]", "_", name)
            name = re.sub(r"\s+", "_", name)
            # Deduplicate
            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[name]}"
            else:
                seen[name] = 0
            renamed[col] = name
        if renamed != {c: c for c in df.columns}:
            print(f"  Sanitized {sum(k != v for k, v in renamed.items())} "
                  f"column name(s)")
        return df.rename(columns=renamed)

    @staticmethod
    def _drop_all_null_axis(df: pd.DataFrame) -> pd.DataFrame:
        """Drop fully-null rows/columns with a warning (they carry no
        signal and break scaling / clustering)."""
        before = df.shape
        df = df.dropna(how="all")          # rows entirely null
        df = df.dropna(axis=1, how="all")  # columns entirely null
        if df.shape != before:
            print(f"  Dropped {before[0] - df.shape[0]} all-null rows, "
                  f"{before[1] - df.shape[1]} all-null columns")
        return df

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
