# UserInsight Agent

A lightweight data analysis agent for **user profiling** and **behavior analysis**, designed for business analytics scenarios such as user segmentation, churn prediction, and behavior forecasting.

## Features

- **Data Cleaning** — Handle missing values, outliers, and data type corrections
- **Feature Engineering** — Encoding, scaling, and derived feature generation
- **User Segmentation** — RFM analysis and K-Means clustering for user layering
- **Behavior Prediction** — Churn prediction and next-action forecasting
- **Visualization** — Generate analysis charts (distribution, clustering, trend)
- **Report Generation** — Output analysis tables and charts to HTML/Excel

## Project Structure

```
userinsight-agent/
├── main.py                    # Entry point, orchestrates the pipeline
├── config.yaml                # Configuration file
├── requirements.txt           # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading
│   ├── data_cleaner.py        # Data cleaning
│   ├── feature_engineering.py # Feature engineering
│   ├── user_segmentation.py   # User segmentation
│   ├── behavior_prediction.py # Behavior prediction
│   ├── visualization.py       # Chart generation
│   └── report_generator.py    # Report generation
├── data/                      # Input data directory
├── output/                    # Generated reports & charts
└── notebooks/                 # Jupyter notebook demos
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your CSV data in the data/ directory

# 3. Configure config.yaml (data path, parameters)

# 4. Run the analysis pipeline
python main.py
```

## Dependencies

- **pandas / numpy** — Data manipulation
- **scikit-learn** — Machine learning (clustering, classification)
- **matplotlib / seaborn** — Visualization
- **pyyaml** — Configuration parsing
- **openpyxl** — Excel report export
