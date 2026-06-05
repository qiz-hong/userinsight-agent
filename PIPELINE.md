# UserInsight Agent — Pipeline & Usage Guide

## 1. What This Agent Does

This agent takes **any tabular user behavior data** (CSV/Excel) and automatically:

1. Cleans and normalizes it (missing values, outliers, type correction)
2. Engineers features (encoding, scaling, time-based features)
3. Segments users via clustering → **user profiling**
4. Trains a prediction model → **behavior forecasting**
5. Outputs charts + Excel/HTML reports

**No manual configuration required** — the pipeline auto-detects column types,
encodings, and selects the best ML model.

---

## 2. Pipeline Flow (7 Steps)

```
 ┌──────────┐    ┌──────────┐    ┌───────────────┐
 │ 1. LOAD  │───→│ 2. CLEAN │───→│ 3. FEATURE ENG │
 └──────────┘    └──────────┘    └───────┬───────┘
                                         │
      ┌──────────────────────────────────┘
      ▼
 ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
 │ 4. SEGMENT   │───→│ 5. PREDICT   │───→│ 6. VISUALIZE │
 └──────────────┘    └──────────────┘    └──────┬───────┘
                                                │
                                                ▼
                                         ┌──────────────┐
                                         │ 7. REPORT    │
                                         └──────────────┘
```

---

## 3. Step-by-Step Algorithm Details

### Step 1: Data Loading (`data_loader.py`)

| Sub-step | Algorithm |
|---|---|
| Format detection | `.csv` → CSV loader, `.xlsx/.xls` → Excel loader |
| Encoding detection | Tries `utf-8` → `gbk` → `gb2312` → `latin-1` → `iso-8859-1` in order, picks first successful |
| Validation | Rejects empty files; prints row × col count |

**Input requirement**: Any CSV or Excel file with a header row.

---

### Step 2: Data Cleaning (`data_cleaner.py`)

| Sub-step | Algorithm | Config override |
|---|---|---|
| **Drop duplicates** | `df.drop_duplicates()` — removes exact duplicate rows | `cleaning.drop_duplicates: false` |
| **Fill missing** | **auto** strategy: numeric cols → median; categorical cols → mode. Other options: `mean`, `drop` | `cleaning.fill_missing_strategy` |
| **Remove outliers** | **IQR method** (off by default): values beyond Q1-1.5×IQR or Q3+1.5×IQR are removed. Only applies to numeric columns. | `cleaning.outlier_columns` |
| **Type correction** | Object columns are tested: (1) try `pd.to_datetime` → if >80% succeed, convert to datetime; (2) try `pd.to_numeric` → if >80% succeed, convert to float | — |

---

### Step 3: Feature Engineering (`feature_engineering.py`)

This is the **core** of auto-adaptation. The method `auto_process()` classifies every column:

| Column type | Detection rule | Transformation |
|---|---|---|
| **ID column** | Explicit (from `config.data.user_id_column`) or auto: unique-per-row AND unique count > 50 | **Dropped** |
| **Datetime** | `datetime64` dtype (converted in Step 2) | Extract `year`, `month`, `day`, `dayofweek`, `recency` (days since latest date); drop original |
| **Categorical (low cardinality)** | `object`/`category` dtype, OR integer with ≤20 unique values, AND unique count ≤10 | **One-hot encoding** (drop_first=True to avoid multicollinearity) |
| **Categorical (high cardinality)** | Same detection, unique count >10 | **Label encoding** (0, 1, 2, ...) |
| **Numeric** | Any remaining numeric dtype | **Standard scaling** (zero mean, unit variance via `StandardScaler`) |

**Why this works**: The agent doesn't care what your columns are named — it infers from data types and value distributions. A column called `signup_date`, `created_at`, or `timestamp` will all be handled the same way as long as Step 2 converted them to datetime.

---

### Step 4: User Segmentation (`user_segmentation.py`)

| Sub-step | Algorithm |
|---|---|
| **Feature selection** | Picks numeric columns, preferring keywords: `recency`, `freq`, `count`, `amount`, `day`. Caps at 8 features. |
| **Cluster count (k)** | **Auto elbow method**: tries k=2..6, computes inertia for each, picks k where inertia improvement drops below 50% of the previous improvement. Falls back to k=3 if no clear elbow. |
| **Clustering** | `sklearn.cluster.KMeans` with `n_init='auto'`, `random_state=42` |
| **Segment profiling** | For each cluster: compute user count, percentage, and mean of each feature → `profiles_` DataFrame |

**Output**: Each user gets a segment label (0, 1, 2...); the smallest segment is treated as the "at-risk" group for prediction.

---

### Step 5: Behavior Prediction (`behavior_prediction.py`)

| Sub-step | Algorithm |
|---|---|
| **Target resolution** | Priority: (1) config `prediction.target_column` if it exists in data → (2) use segment labels (smallest segment = churn) → (3) auto-detect any binary column |
| **Feature selection** | All numeric columns except the target |
| **Model selection** | Trains both **RandomForest** (100 trees) and **LogisticRegression**, compares 3-fold cross-val accuracy, picks the winner |
| **Evaluation** | 80/20 train-test split (stratified), reports accuracy and CV score |

**Note**: If your data has a real `churn` or `is_active` column, the agent will detect it automatically. If not, it creates a synthetic target from the segmentation results — treating the smallest user cluster as the "positive" class.

---

### Step 6: Visualization (`visualization.py`)

| Chart | Description | When generated |
|---|---|---|
| `segment_distribution.png` | Bar chart: user count per segment | Always (if segmentation succeeded) |
| `segment_profiles.png` | Grouped bar: mean feature values per segment | Always (if segmentation succeeded) |
| `correlation_heatmap.png` | Upper-triangle correlation matrix of all numeric features | Always (if ≥2 numeric features) |
| `feature_importance.png` | Horizontal bar chart of top 15 features by importance | If prediction model was trained |

All charts use non-interactive `Agg` backend; saved at 150 DPI.

---

### Step 7: Report Generation (`report_generator.py`)

| Report | Contents |
|---|---|
| `analysis_report.xlsx` | 4 sheets: Segments (user labels), Profiles (cluster stats), Feature_Importance (feature ranking), Metrics (model performance) |
| `report.html` | Self-contained HTML page with summary statistics + embedded chart images |
| Console output | Formatted text summary printed to terminal |

---

## 4. How to Use with New Data

### 4.1 Minimal Setup (3 steps)

```bash
# Step 1: Install dependencies (once)
pip install -r requirements.txt

# Step 2: Place your data file in data/ directory
cp /path/to/your_user_data.csv data/

# Step 3: Update config.yaml → data.file_path
# Change:  file_path: "data/your_user_data.csv"

# Step 4: Run
python main.py
```

### 4.2 What Your Data Should Look Like

**Minimum**: Any CSV with a header row. The agent is **schema-flexible**.

**Recommended** columns for best results:

| Column type | Examples | Why useful |
|---|---|---|
| User ID | `user_id`, `customer_id` | Auto-dropped (not used as feature) |
| Demographics | `age`, `gender`, `city`, `membership_tier` | Categorical → one-hot encoded; numeric → scaled |
| Date/Time | `registration_date`, `last_login`, `order_date` | Auto-converted → recency, month, day-of-week features |
| Behavior counts | `login_count`, `purchase_count`, `page_views` | Core features for segmentation & prediction |
| Monetary | `total_spent`, `avg_order_value`, `ltv` | Key for high-value user identification |
| Label (optional) | `churn`, `is_active`, `converted` | If present, used as prediction target |

### 4.3 Configuration Reference (`config.yaml`)

```yaml
data:
  file_path: "data/my_data.csv"       # ← CHANGE THIS to your file
  user_id_column: "user_id"           # ← CHANGE to your ID column name
  date_column: "event_date"           # optional, for reference
  amount_column: "amount"             # optional

cleaning:
  drop_duplicates: true
  fill_missing_strategy: "auto"       # auto | median | mean | mode | drop
  outlier_columns: []                 # leave empty to skip outlier removal

features:
  encode_categorical: true
  scale_numerical: true

segmentation:
  n_clusters: null                    # null = auto-detect, or set 3/4/5
  method: "kmeans"

prediction:
  target_column: null                 # null = auto-detect or use segments
  model_type: "auto"                  # auto selects best of RF vs LR
  test_size: 0.2

output:
  dir: "output"
  excel_report: true
  html_report: true
  save_charts: true
```

### 4.4 Programmatic Usage (Python)

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import DataLoader
from data_cleaner import DataCleaner
from feature_engineering import FeatureEngineer
from user_segmentation import UserSegmentation
from behavior_prediction import BehaviorPredictor

# Load and clean
loader = DataLoader("data/my_data.csv")
df = loader.load()

cleaner = DataCleaner(df)
cleaner.drop_duplicates().fill_missing("auto").correct_dtypes()
df_clean = cleaner.get_clean_data()

# Feature engineering
engineer = FeatureEngineer(df_clean)
engineer.auto_process(id_column="user_id")
df_feat = engineer.get_features()

# Segmentation
segmenter = UserSegmentation(df_feat)
labels = segmenter.auto_segment()
print(segmenter.get_segment_profiles())
print(segmenter.get_segment_summary())

# Prediction
predictor = BehaviorPredictor(df_feat)
predictor.auto_predict(segment_labels=labels)
print(predictor.get_metrics())
print(predictor.get_feature_importance())
```

---

## 5. Error Handling Philosophy

Every step is wrapped in `try/except` in `main.py`. If any step fails:

- **Load fails** → pipeline stops (no data = nothing to analyze)
- **Clean/Feature/Segment/Predict/Visualize fails** → warning printed, pipeline continues with best-effort data
- **Report fails** → warning printed, charts still available

This means you can feed it messy real-world data and still get useful outputs.
