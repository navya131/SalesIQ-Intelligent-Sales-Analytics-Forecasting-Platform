# 📊 SalesIQ — Intelligent Sales Analytics & Forecasting Platform

> **Technology:** Python 3.10 · Streamlit · Machine Learning · Plotly  
> **Type:** End-to-End Data Science & Business Intelligence Web Application  
> **Domain:** Sales Analytics · Demand Forecasting · Customer Intelligence

---

## 🎯 Problem Statement

Inaccurate sales forecasts lead to significant inventory management challenges — either
stockouts and lost sales, or overstocking and increased carrying costs. Traditional
forecasting methods fail to adapt to dynamic market conditions and external factors.

**SalesIQ** solves this by leveraging advanced statistical models, machine learning,
and BI tools to accurately predict sales demand, optimize inventory levels, and provide
actionable insights for supply chain efficiency — minimizing costs and maximizing profitability.

---

## ✨ Core Features (9 Modules)

| # | Module | Key Functionality |
|---|--------|-------------------|
| 1 | 📂 Data Upload | CSV/Excel upload, auto-validation, column detection, dataset summary |
| 2 | 🔧 Preprocessing & EDA | Missing value handling, outlier detection, distributions, correlation heatmap, time analysis |
| 3 | 📈 Sales Forecasting | Prophet, ARIMA, XGBoost, Moving Average — with confidence intervals |
| 4 | 🤖 Model Training | Random Forest, Gradient Boosting, Linear/Ridge Regression — MAE, RMSE, R², MAPE |
| 5 | 🔄 Churn Prediction | ML classifier with risk tiers, ROC curve, confusion matrix, probability scores |
| 6 | 💡 Insights & Filters | Interactive BI dashboard with dynamic filters by region, product, channel, revenue |
| 7 | 🚨 Anomaly Detection | Isolation Forest — flags unusual transactions with visual scatter plots |
| 8 | 📋 Reports | Auto-generate PDF + multi-sheet Excel reports with one click download |
| 9 | 🏠 Home Dashboard | KPI overview, navigation guide, quick stats |

---

## 🛠️ Complete Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.10 | Core development |
| **Web Framework** | Streamlit 1.35 | Interactive UI |
| **Data Processing** | Pandas, NumPy | Data wrangling |
| **Visualization** | Plotly Express, Plotly GO | Interactive charts |
| **Forecasting** | Prophet, Statsmodels (ARIMA) | Time series models |
| **ML Models** | Scikit-learn, XGBoost | Classification & Regression |
| **Anomaly Detection** | Scikit-learn IsolationForest | Outlier detection |
| **PDF Reports** | FPDF2 | PDF generation |
| **Excel Reports** | OpenPyXL | Multi-sheet Excel export |
| **Styling** | Custom CSS | Pastel baby-blue light theme |

---

## 📁 Project Structure

```
sales_intelligence_app/
│
├── app.py                       ← Main entry point, routing, top nav bar
├── config.py                    ← Colors, theme, page list
├── requirements.txt             ← All pip dependencies
│
├── modules/
│   ├── __init__.py
│   ├── home.py                  ← Home dashboard & KPI cards
│   ├── data_upload.py           ← File upload, validation, column info
│   ├── preprocessing.py         ← Cleaning, EDA, distributions, correlation
│   ├── forecasting.py           ← Prophet / ARIMA / XGBoost / Moving Avg
│   ├── model_training.py        ← Train & evaluate regression models
│   ├── churn_prediction.py      ← Churn classifier, ROC, confusion matrix
│   ├── insights.py              ← BI dashboard with sidebar filters
│   ├── anomaly_detection.py     ← Isolation Forest anomaly detection
│   └── reports.py               ← PDF + Excel report generator
│
├── utils/
│   ├── __init__.py
│   └── helpers.py               ← Shared utility functions
│
├── assets/
│   └── style.css                ← Custom pastel baby-blue CSS theme
│
└── sample_data/
    ├── generate.py              ← Auto-generate 1200-row sample dataset
    └── sample_sales.csv         ← Ready-to-use sample data
```

---

## 📊 ML Models & Algorithms

### Sales Forecasting Models
| Model | Algorithm | Best For |
|-------|-----------|---------|
| **XGBoost** | Gradient Boosted Trees with lag/time features | Short-medium term, complex patterns |
| **Prophet** | Additive time series (Facebook) | Seasonal trends, holiday effects |
| **ARIMA(2,1,2)** | AutoRegressive Integrated Moving Average | Stationary time series |
| **Moving Average** | Trend-adjusted rolling mean | Baseline, simple trends |

### Classification Models (Churn)
| Model | Algorithm | Metrics |
|-------|-----------|---------|
| **Random Forest** | Ensemble of decision trees | Accuracy, Precision, Recall, F1 |
| **Gradient Boosting** | Sequential boosted trees | AUC-ROC, Confusion Matrix |
| **Logistic Regression** | Linear probabilistic classifier | Probability scores, risk tiers |

### Anomaly Detection
| Model | Algorithm | Output |
|-------|-----------|--------|
| **Isolation Forest** | Random partitioning trees | Anomaly score, flagged records |

---

## 📋 Sample Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `date` | DateTime | Transaction date (2021–2024) |
| `customer_id` | String | Unique customer identifier |
| `product` | Category | Laptop, Phone, Tablet, Watch, Earbuds |
| `region` | Category | North, South, East, West |
| `channel` | Category | Online, Retail, Partner |
| `quantity` | Integer | Units sold per transaction |
| `unit_price` | Float | Price per unit (₹5,000–₹75,000) |
| `discount` | Float | Discount percentage (0–25%) |
| `revenue` | Float | Final revenue after discount |
| `csat_score` | Integer | Customer satisfaction (1–5) |
| `churn` | Binary | Churn label (0 = retained, 1 = churned) |

**Size:** 1,200 rows × 11 columns

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/navya131/SalesIQ-Intelligent-Sales-Analytics-Forecasting-Platform.git
cd SalesIQ-Intelligent-Sales-Analytics-Forecasting-Platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate sample data (optional)
python sample_data/generate.py

# 4. Launch the app
streamlit run app.py

# 5. Open browser at
# http://localhost:8501
```

---

## 📦 requirements.txt

```
streamlit==1.35.0
pandas==2.2.2
numpy==1.26.4
matplotlib==3.9.0
seaborn==0.13.2
plotly==5.22.0
scikit-learn==1.5.0
xgboost==2.0.3
prophet==1.1.5
statsmodels==0.14.2
openpyxl==3.1.2
fpdf2==2.7.9
imbalanced-learn==0.12.3
scipy==1.13.1
joblib==1.4.2
```

---

## 🎨 UI Design

- **Theme:** Light pastel baby-blue throughout
- **Sidebar:** Gradient blue navigation with dark readable text
- **Cards:** White metric cards with blue borders and soft shadows
- **Charts:** Light blue background (#F8FBFF) with dark axis labels
- **Buttons:** Pastel green-blue gradient with hover animation
- **Font:** Inter (Google Fonts) — clean and professional

---

## 📈 Evaluation Metrics Used

### Regression / Forecasting
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination
- **MAPE** — Mean Absolute Percentage Error

### Classification / Churn
- **Accuracy** — Overall correct predictions
- **Precision** — True positive rate
- **Recall** — Sensitivity
- **F1 Score** — Harmonic mean of precision & recall
- **AUC-ROC** — Area under ROC curve
- **Confusion Matrix** — Visual classification breakdown







> ⭐ **If this project helped you, please star the repository!**  
> Built with ❤️ using Python, Streamlit & Machine Learning
