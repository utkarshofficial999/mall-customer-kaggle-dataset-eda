# mall-customer-kaggle-dataset-eda

[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=flat&logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-KMeans-F7931E?style=flat&logo=scikitlearn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end Exploratory Data Analysis (EDA), Machine Learning segmentation (K-Means Clustering), and real-time FastAPI web deployment on the **Mall Customers** dataset from Kaggle.

---

## 📌 Project Overview

This repository provides:
1. **In-Depth Exploratory Data Analysis (EDA)**: Statistical profiling, IQR outlier detection, distribution analysis, two-sample $t$-test hypothesis checks on demographics, and bivariate behavioral dynamics.
2. **Interactive Jupyter Notebook** (`Mall_Customers_EDA.ipynb`): A self-contained, executed notebook following ML best practices where every code block is accompanied by statistical commentary.
3. **K-Means Clustering Model**: Feature scaling via `StandardScaler`, validation with the **Elbow Method (WCSS)** and **Silhouette Analysis** (optimal peak at $K=5$ with silhouette score **0.5547**).
4. **Real-Time Web Application & API**: A modern, dark-mode single-page dashboard built with **FastAPI**, featuring live inference sliders, real-time Chart.js scatter space with centroids, cluster playbooks, and a searchable dataset explorer.

---

## 📊 Dataset Description

The dataset consists of **200 customer records** with **zero missing values**:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `CustomerID` | `int64` | Unique numerical identifier for each customer |
| `Genre` (Gender) | `object` | Biological sex (`Male` / `Female`) |
| `Age` | `int64` | Shopper age in years (Range: 18 - 70) |
| `Annual Income (k$)` | `int64` | Customer annual earnings in thousands of USD ($15k - $137k) |
| `Spending Score (1-100)`| `int64` | Behavioral score assigned by the mall based on purchase habits |

---

## 🎯 Discovered Customer Personas (K = 5)

Through unsupervised learning on `Annual Income` and `Spending Score`, 5 distinct natural customer segments were identified:

| Persona | Centroid Income | Centroid Spend | Share | Targeted Marketing Strategy |
| :--- | :--- | :--- | :--- | :--- |
| 👑 **High-Value VIPs** | $86.5k | 82.1 / 100 | ~20% | Exclusive loyalty tier, personal concierge, luxury brand collaborations, private previews. |
| 💎 **Frugal Affluent** | $88.3k | 17.1 / 100 | ~18% | Quality-focused messaging, durable high-end products, investment value, warranties. |
| 🎯 **Mainstream Middle Class** | $55.3k | 49.7 / 100 | ~40% | Seasonal promotional campaigns, family weekend bundles, cashback reward points. |
| ⚡ **Carefree Spenders** | $25.7k | 79.4 / 100 | ~11% | Trend-driven apparel, limited edition drops, Buy Now Pay Later (BNPL), social media events. |
| 🛡️ **Sensible Budget Savers** | $26.3k | 20.9 / 100 | ~11% | Clearance deals, value packs, discount coupons, price-match guarantees. |

---

## 📈 Visualizations

All high-resolution figures are generated in the `plots/` folder:
- `01_univariate_distributions.png`: Histograms, KDE density curves, and boxplots for continuous features.
- `02_gender_breakdown.png`: Gender pie breakdown and statistical comparison.
- `03_income_vs_spending.png`: The classic 5-cluster behavioral plane.
- `04_age_vs_spending_income.png`: Generational spending gaps and the Age 40 threshold.
- `05_correlation_matrix.png`: Pearson and Spearman correlation heatmaps.
- `06_pairwise_relationships.png`: Multivariate pairplot matrix by gender.

---

## 🚀 Running the Web Application Locally

### 1. Install Dependencies
```bash
pip install fastapi uvicorn pandas numpy scikit-learn scipy matplotlib seaborn
```

### 2. Train the Model
```bash
python train_model.py
```

### 3. Launch the FastAPI Service
```bash
uvicorn app:app --host 127.0.0.1 --port 8080 --reload
```

- **Interactive Dashboard**: Open [http://127.0.0.1:8080](http://127.0.0.1:8080) in your browser.
- **Swagger REST API Documentation**: Open [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs).

---

## ⚡ API Quickstart

### Predict Customer Cluster
```bash
curl -X POST "http://127.0.0.1:8080/api/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "annual_income": 85,
       "spending_score": 82,
       "age": 30,
       "gender": "Female"
     }'
```

**Response:**
```json
{
  "cluster_id": 3,
  "persona_id": "vip",
  "cluster_name": "High-Value VIPs",
  "color": "#10b981",
  "icon": "👑",
  "confidence_score": 90.7,
  "marketing_strategy": "Invite to exclusive loyalty clubs, VIP concierge, private product previews, and luxury brand collaborations."
}
```

---

## 📁 Repository Structure

```text
├── Mall_Customers.csv             # Kaggle dataset
├── Mall_Customers_EDA.ipynb       # Fully executed exploratory analysis notebook
├── app.py                         # Production FastAPI server & interactive SPA dashboard
├── train_model.py                 # K-Means training & model parameter export
├── run_eda.py                     # Standalone statistical EDA analysis script
├── generate_full_notebook.py      # Automated notebook execution engine
├── model_metadata.json            # Model parameters, centroids, & evaluation metrics
├── clustered_customers.json       # Labeled customer records for API consumption
├── plots/                         # High-resolution generated charts
└── README.md                      # Project documentation
```
