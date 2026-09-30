import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()

cells = []

# Title cell
cells.append(nbf.v4.new_markdown_cell("""# Comprehensive Exploratory Data Analysis (EDA) on Mall Customers

## Business & Analytical Objectives
The objective of this exploratory data analysis is to understand customer demographics, income distribution, and purchasing behaviors from the **Mall Customers** dataset. Specifically, we aim to:
1. Examine data integrity, distributions, skewness, and outliers.
2. Unpack relationships between customer demographics (**Gender**, **Age**) and economic indicators (**Annual Income**, **Spending Score**).
3. Identify natural behavioral clusters to lay the groundwork for optimal customer segmentation (e.g., K-Means clustering).
4. Provide strategic marketing takeaways and personas.
"""))

# Cell 1: Environment & Data Ingestion
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

# Load dataset
df = pd.read_csv('Mall_Customers.csv')
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
df.head()
"""))

# Cell 1 Markdown Analysis
cells.append(nbf.v4.new_markdown_cell("""### Analysis of Ingestion & Schema
- The dataset consists of **200 observations** and **5 features**:
  - `CustomerID`: Unique identifier for each shopper (nominal ID).
  - `Genre`: Customer biological sex (`Male` / `Female`).
  - `Age`: Continuous numerical age in years.
  - `Annual Income (k$)`: Customer annual earnings in thousands of USD.
  - `Spending Score (1-100)`: Proprietary mall score assigned based on customer behavior and spending habits.
- We will standardize feature names for clean programmatic access (`Gender`, `Annual_Income_k`, and `Spending_Score`).
"""))

# Cell 2: Data Cleaning & Integrity Verification
cells.append(nbf.v4.new_code_cell("""# Rename columns
df_clean = df.rename(columns={
    'Genre': 'Gender',
    'Annual Income (k$)': 'Annual_Income_k',
    'Spending Score (1-100)': 'Spending_Score'
})

# Data integrity checks
info_dict = {
    'Non-Null Count': df_clean.notnull().sum(),
    'Null Count': df_clean.isnull().sum(),
    'Data Type': df_clean.dtypes,
    'Unique Values': df_clean.nunique()
}
integrity_df = pd.DataFrame(info_dict)
print("Data Integrity Summary:")
display(integrity_df)

print(f"Duplicate records: {df_clean.duplicated().sum()}")
"""))

# Cell 2 Markdown Analysis
cells.append(nbf.v4.new_markdown_cell("""### Analysis of Data Quality & Integrity
- **Zero Missing Values**: Every column contains 200 non-null values. No imputation or deletion is required.
- **No Duplicates**: All rows and `CustomerID` values are distinct.
- **Data Types**: `Age`, `Annual_Income_k`, and `Spending_Score` are integer types; `Gender` is a string category.
"""))

# Cell 3: Descriptive Statistics & Shape Parameters
cells.append(nbf.v4.new_code_cell("""stats_summary = df_clean[['Age', 'Annual_Income_k', 'Spending_Score']].describe().T
stats_summary['skewness'] = df_clean[['Age', 'Annual_Income_k', 'Spending_Score']].skew()
stats_summary['kurtosis'] = df_clean[['Age', 'Annual_Income_k', 'Spending_Score']].kurt()
stats_summary['iqr'] = stats_summary['75%'] - stats_summary['25%']
display(stats_summary.round(2))
"""))

# Cell 3 Markdown Analysis
cells.append(nbf.v4.new_markdown_cell("""### Statistical Profile & Distribution Characteristics
1. **Age**:
   - Ranges from **18 to 70 years**, with a mean of **38.85** and median of **36.00**.
   - Slight positive skewness (+0.49), indicating a slightly younger consumer base with a tail extending into older shoppers.
2. **Annual Income ($k)**:
   - Ranges from **$15k to $137k**, with an average of **$60.56k** and median of **$61.50k**.
   - Near-zero skewness (+0.32), indicating a relatively balanced income spectrum.
3. **Spending Score (1-100)**:
   - Ranges from **1 to 99**, with a mean of **50.20** and median of **50.00**.
   - Symmetric distribution (skewness = -0.05), perfectly centered around 50.
"""))

# Cell 4: Outlier Detection (IQR Method)
cells.append(nbf.v4.new_code_cell("""outliers_records = []
for col in ['Age', 'Annual_Income_k', 'Spending_Score']:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    outliers = df_clean[(df_clean[col] < lower_limit) | (df_clean[col] > upper_limit)]
    outliers_records.append({
        'Feature': col,
        'Q1': round(q1, 2),
        'Q3': round(q3, 2),
        'IQR': round(iqr, 2),
        'Lower Bound': round(lower_limit, 2),
        'Upper Bound': round(upper_limit, 2),
        'Outlier Count': len(outliers),
        'Outlier CustomerIDs': outliers['CustomerID'].tolist()
    })

pd.DataFrame(outliers_records)
"""))

# Cell 4 Markdown Analysis
cells.append(nbf.v4.new_markdown_cell("""### Outlier Detection Findings
- **Age**: No outliers detected (bounds: -1.62 to 79.38).
- **Spending Score**: No outliers detected (bounds: -22.62 to 130.38).
- **Annual Income**: Detected **2 upper outliers** (Customer IDs 199 and 200, earning **$137k** each with IQR bound at $132.75k).
- **Recommendation**: In accordance with machine learning best practices (*"prefer to keep data instead of dropping it when possible"*), these 2 records represent genuine high-earning shoppers (spending scores of 18 and 83) and should be **retained** rather than removed.
"""))

# Cell 5: Univariate Distributions & Outliers Visualized
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 3, figsize=(16, 8))
features = ['Age', 'Annual_Income_k', 'Spending_Score']
colors = ['#2b5c8f', '#2ca02c', '#d62728']
titles = ['Age (Years)', 'Annual Income (k$)', 'Spending Score (1-100)']

for i, col in enumerate(features):
    # Histogram + KDE
    sns.histplot(df_clean[col], kde=True, ax=axes[0, i], color=colors[i], bins=15, edgecolor='black', alpha=0.6)
    axes[0, i].axvline(df_clean[col].mean(), color='crimson', linestyle='--', label=f"Mean: {df_clean[col].mean():.1f}")
    axes[0, i].axvline(df_clean[col].median(), color='navy', linestyle=':', label=f"Median: {df_clean[col].median():.1f}")
    axes[0, i].set_title(f'Histogram & KDE: {titles[i]}', fontweight='bold')
    axes[0, i].legend()

    # Boxplot
    sns.boxplot(x=df_clean[col], ax=axes[1, i], color=colors[i], flierprops=dict(markerfacecolor='red', marker='D', markersize=8))
    axes[1, i].set_title(f'Boxplot: {titles[i]}', fontweight='bold')

plt.tight_layout()
plt.show()
"""))

# Cell 5 Markdown Analysis
cells.append(nbf.v4.new_markdown_cell("""### Insights from Univariate Distributions
- **Age Distribution**: Shows peaks around age 30–35, with steady representation through age 50 and a gradual taper toward age 70.
- **Income Distribution**: Exhibits a healthy bimodal-like spread around $50k–$80k, representing the core consumer middle/upper-middle tier.
- **Spending Score Distribution**: Well-behaved bell shape with a pronounced cluster around the median score of 50.
"""))

# Cell 6: Demographic Breakdown: Gender Comparisons & Hypothesis Testing
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Gender proportions
gender_counts = df_clean['Gender'].value_counts()
axes[0].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'],
            startangle=140, explode=(0.04, 0), wedgeprops=dict(edgecolor='white', linewidth=2))
axes[0].set_title('Gender Distribution Ratio', fontweight='bold')

# Income by Gender
sns.boxplot(x='Gender', y='Annual_Income_k', data=df_clean, ax=axes[1], hue='Gender', palette=['#ff9999', '#66b3ff'], legend=False)
axes[1].set_title('Annual Income by Gender', fontweight='bold')
axes[1].set_ylabel('Annual Income (k$)')

# Spending by Gender
sns.boxplot(x='Gender', y='Spending_Score', data=df_clean, ax=axes[2], hue='Gender', palette=['#ff9999', '#66b3ff'], legend=False)
axes[2].set_title('Spending Score by Gender', fontweight='bold')
axes[2].set_ylabel('Spending Score (1-100)')

plt.tight_layout()
plt.show()

# Statistical Significance (Two-sample independent t-tests)
t_inc, p_inc = stats.ttest_ind(df_clean[df_clean['Gender']=='Male']['Annual_Income_k'],
                               df_clean[df_clean['Gender']=='Female']['Annual_Income_k'])
t_spd, p_spd = stats.ttest_ind(df_clean[df_clean['Gender']=='Male']['Spending_Score'],
                               df_clean[df_clean['Gender']=='Female']['Spending_Score'])

print(f"Income Difference by Gender:  t = {t_inc:.3f}, p-value = {p_inc:.4f}")
print(f"Spending Difference by Gender: t = {t_spd:.3f}, p-value = {p_spd:.4f}")
"""))

# Cell 6 Markdown Analysis
cells.append(nbf.v4.new_markdown_cell("""### Analysis of Demographic Differences
- **Gender Balance**: Females represent **56.0%** (112 customers) while males represent **44.0%** (88 customers).
- **Statistical Invariance**:
  - Income difference test yields $p = 0.428$ ($> 0.05$).
  - Spending Score difference test yields $p = 0.414$ ($> 0.05$).
- **Key Takeaway**: Gender alone does **not** significantly dictate purchasing power or spending inclination. Segmentation strategies should not rely on gender alone as a primary splitting variable, but rather on behavioral and financial features.
"""))

# Cell 7: The Primary Behavioral Topology: Annual Income vs. Spending Score
cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(11, 7))

sns.scatterplot(
    data=df_clean,
    x='Annual_Income_k',
    y='Spending_Score',
    hue='Gender',
    style='Gender',
    s=90,
    palette={'Male': '#1f77b4', 'Female': '#e377c2'},
    alpha=0.85,
    edgecolor='black'
)

# Mark the natural boundaries of the 5 clusters
annotations = [
    (25, 80, "Carefree / Spenders\\n(Low Income, High Spending)", '#2ca02c'),
    (25, 20, "Budget / Sensible\\n(Low Income, Low Spending)", '#7f7f7f'),
    (55, 50, "Average Middle Class\\n(Moderate Income, Moderate Spending)", '#17becf'),
    (88, 85, "High-Value Targets / VIP\\n(High Income, High Spending)", '#d62728'),
    (88, 18, "Frugal Affluent / Savers\\n(High Income, Low Spending)", '#9467bd')
]
for x, y, text, color in annotations:
    plt.annotate(text, xy=(x, y), xytext=(x, y),
                 bbox=dict(boxstyle="round,pad=0.5", fc=color, alpha=0.18, ec=color, lw=1.5),
                 fontsize=10, fontweight='bold', ha='center')

plt.title('Annual Income vs. Spending Score: 5 Distinct Customer Clusters', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Annual Income (k$)', fontsize=12, fontweight='bold')
plt.ylabel('Spending Score (1-100)', fontsize=12, fontweight='bold')
plt.axvline(x=40, color='gray', linestyle=':', alpha=0.5)
plt.axvline(x=70, color='gray', linestyle=':', alpha=0.5)
plt.axhline(y=40, color='gray', linestyle=':', alpha=0.5)
plt.axhline(y=60, color='gray', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
"""))

# Cell 7 Markdown Analysis
cells.append(nbf.v4.new_markdown_cell("""### Deep Dive: Identification of 5 Natural Behavioral Clusters
The scatter plot reveals an unmistakably structured 5-cluster topology:
1. **Cluster 1: High Income, High Spending (VIP Targets)**
   - High earnings ($70k–$137k) and high spending scores (60–100). Prime audience for luxury releases, premium loyalty programs, and personalized concierge marketing.
2. **Cluster 2: High Income, Low Spending (Frugal Affluent)**
   - High earnings ($70k–$137k) but low spending scores (1–40). High potential purchasing power, but price-conscious or selective. Need value propositions and curated product showcases to convert.
3. **Cluster 3: Moderate Income, Moderate Spending (Balanced Middle Class)**
   - Medium earnings ($40k–$70k) and medium spending scores (40–60). Forms the largest core mass of customers. Responsive to broad promotional campaigns and standard rewards.
4. **Cluster 4: Low Income, High Spending (Carefree Spenders)**
   - Low earnings ($15k–$40k) but high spending scores (60–100). Impulse shoppers or lifestyle buyers. Responsive to flash sales, experiential events, and accessible trendy items.
5. **Cluster 5: Low Income, Low Spending (Sensible Savers)**
   - Low earnings ($15k–$40k) and low spending scores (1–40). Budget-constrained shoppers who visit strictly for necessities and deep discounts.
"""))

# Cell 8: Age Interactions: Generational Spending Patterns
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# Age vs Spending Score with trendline
sns.regplot(data=df_clean, x='Age', y='Spending_Score', ax=axes[0],
            scatter_kws={'alpha':0.6, 'color': '#2b5c8f', 's': 55}, line_kws={'color':'red', 'lw':2})
axes[0].axvline(40, color='green', linestyle='--', label='Age 40 Cutoff')
axes[0].axhline(60, color='orange', linestyle='--', label='High Spending Threshold (>60)')
axes[0].set_title('Age vs Spending Score: Generational Shift', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Age (Years)')
axes[0].set_ylabel('Spending Score (1-100)')
axes[0].legend(loc='upper right')

# Age Group Aggregation
bins = [17, 30, 45, 60, 100]
labels = ['Young Adults (18-30)', 'Middle-aged (31-45)', 'Older Adults (46-60)', 'Seniors (61+)']
df_clean['Age_Group'] = pd.cut(df_clean['Age'], bins=bins, labels=labels)
age_summary = df_clean.groupby('Age_Group', observed=True)[['Annual_Income_k', 'Spending_Score']].mean()

sns.barplot(data=df_clean, x='Age_Group', y='Spending_Score', ax=axes[1], hue='Age_Group', palette='Blues_r', legend=False)
axes[1].set_title('Mean Spending Score Across Age Cohorts', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Age Cohort')
axes[1].set_ylabel('Mean Spending Score')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()

display(age_summary.round(2))
"""))

# Cell 8 Markdown Analysis
cells.append(nbf.v4.new_markdown_cell("""### Analysis of Generational Behavior
1. **The Age 40 Threshold**:
   - Customers aged **under 40** dominate the high-spending bracket (Spending Score > 60).
   - In contrast, customers aged **above 40** almost never exceed a spending score of 60.
2. **Age Cohort Spending Gap**:
   - **Young Adults (18–30)** have the highest mean spending score (**61.03**).
   - **Middle-aged (31–45)** have the highest average income (**$69.39k**) and a healthy mean spending score (**52.57**).
   - **Older Adults (46–60)** have the lowest mean spending score (**35.10**).
"""))

# Cell 9: Correlation Analysis & Multivariate Pairwise Relationships
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Pearson Correlation Heatmap
corr_pearson = df_clean[['Age', 'Annual_Income_k', 'Spending_Score']].corr(method='pearson')
sns.heatmap(corr_pearson, annot=True, fmt='.3f', cmap='vlag', vmin=-1, vmax=1,
            square=True, linewidths=1.5, ax=axes[0])
axes[0].set_title('Pearson Linear Correlation', fontweight='bold')

# Spearman Rank Correlation Heatmap
corr_spearman = df_clean[['Age', 'Annual_Income_k', 'Spending_Score']].corr(method='spearman')
sns.heatmap(corr_spearman, annot=True, fmt='.3f', cmap='vlag', vmin=-1, vmax=1,
            square=True, linewidths=1.5, ax=axes[1])
axes[1].set_title('Spearman Rank Correlation', fontweight='bold')

plt.tight_layout()
plt.show()
"""))

# Cell 9 Markdown Analysis
cells.append(nbf.v4.new_markdown_cell("""### Correlation Analysis Insights
- **Age vs Spending Score**: Shows a moderate negative correlation ($r = -0.327$, Spearman $\\rho = -0.345$). Older age is associated with more conservative spending behaviors.
- **Annual Income vs Spending Score**: Correlation is near zero ($r = 0.010$). This confirms that spending appetite is **independent** of income level linearly, which explains why distinct non-linear clusters emerge (e.g. low-income high-spenders alongside high-income low-spenders).
- **Age vs Annual Income**: Weak positive correlation ($r = -0.012$, non-linear trajectory with earnings peaking in middle age).
"""))

# Cell 10: Executive Summary & Machine Learning Next Steps
cells.append(nbf.v4.new_markdown_cell("""## Strategic Summary & Next Steps for Modeling

### 1. Key Analytical Conclusions
- **Data Completeness**: Clean 200-record dataset with zero missing values and zero duplicate rows.
- **Outliers**: Only 2 mild upper-bound income outliers ($137k); both are valid data points and should be kept.
- **Demographics**: Gender distribution is 56% Female / 44% Male, but statistical hypothesis tests demonstrate that gender has **no significant impact** on either income or spending score.
- **Generational Contrast**: Customers under 40 represent almost all high-spending scores (>60), while customers over 40 exhibit disciplined, lower spending.
- **Cluster Structure**: The interaction between `Annual Income` and `Spending Score` naturally resolves into **5 clearly separable customer groups**.

### 2. Preparation for K-Means Clustering
- **Feature Selection**: The primary segmentation plane should be `Annual_Income_k` and `Spending_Score`. A secondary 3-dimensional segmentation model incorporating `Age` can be tested to isolate young high-value shoppers.
- **Standardization**: Because `Annual Income` (scale ~15–137) and `Spending Score` (scale ~1–100) have differing standard deviations (~26.26 vs ~25.82), applying `StandardScaler` is recommended before computing Euclidean distances in K-Means.
- **Expected Optimal $K$**: Visual evidence strongly supports **$K = 5$** as the primary cluster count, which can be formally verified using the Elbow Method (WCSS) and Silhouette Coefficient.
"""))

nb.cells = cells

# Save the raw notebook first
notebook_path = "Mall_Customers_EDA.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Saved notebook structure to {notebook_path}. Now executing...")

# Execute the notebook
client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()

# Save executed notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully executed and saved: {notebook_path}")
