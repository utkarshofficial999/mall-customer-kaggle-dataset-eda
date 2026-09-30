import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import nbformat as nbf
from nbclient import NotebookClient

# Configure aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
palette = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#1b9e77"]

dataset_path = "Mall_Customers.csv"
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
artifact_dir = r"C:\Users\Utkar\.gemini\antigravity-ide\brain\cb058f7b-b2d5-4a0b-985f-d379f08a6ad6"
os.makedirs(artifact_dir, exist_ok=True)

df = pd.read_csv(dataset_path)

print("--- DATASET HEAD ---")
print(df.head())
print("\n--- SHAPE & TYPES ---")
print(df.info())
print("\n--- DESCRIPTIVE STATS ---")
print(df.describe().T)
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())
print("\n--- DUPLICATES ---")
print(df.duplicated().sum())

# Rename columns for convenience
df_clean = df.rename(columns={
    'Genre': 'Gender',
    'Annual Income (k$)': 'Annual_Income_k',
    'Spending Score (1-100)': 'Spending_Score'
})

# 1. Outlier detection via IQR
outliers_summary = {}
for col in ['Age', 'Annual_Income_k', 'Spending_Score']:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
    skew = df_clean[col].skew()
    kurt = df_clean[col].kurtosis()
    outliers_summary[col] = {
        'Q1': q1, 'Q3': q3, 'IQR': iqr,
        'Lower_Bound': lower_bound, 'Upper_Bound': upper_bound,
        'Outlier_Count': len(outliers),
        'Outlier_IDs': outliers['CustomerID'].tolist() if len(outliers) > 0 else [],
        'Skewness': skew,
        'Kurtosis': kurt
    }
print("\n--- OUTLIER & SHAPE ANALYSIS ---")
for k, v in outliers_summary.items():
    print(f"{k}: {v}")

# 2. Gender stats
gender_counts = df_clean['Gender'].value_counts()
gender_income = df_clean.groupby('Gender')['Annual_Income_k'].agg(['count', 'mean', 'median', 'std'])
gender_spending = df_clean.groupby('Gender')['Spending_Score'].agg(['count', 'mean', 'median', 'std'])
print("\n--- GENDER STATS ---")
print("Counts:\n", gender_counts)
print("Income by Gender:\n", gender_income)
print("Spending by Gender:\n", gender_spending)

# T-tests
t_stat_inc, p_val_inc = stats.ttest_ind(df_clean[df_clean['Gender']=='Male']['Annual_Income_k'],
                                        df_clean[df_clean['Gender']=='Female']['Annual_Income_k'])
t_stat_spd, p_val_spd = stats.ttest_ind(df_clean[df_clean['Gender']=='Male']['Spending_Score'],
                                        df_clean[df_clean['Gender']=='Female']['Spending_Score'])
print(f"Income Gender T-test: t={t_stat_inc:.4f}, p={p_val_inc:.4f}")
print(f"Spending Gender T-test: t={t_stat_spd:.4f}, p={p_val_spd:.4f}")

# Age Group Segmentation
bins = [17, 30, 45, 60, 100]
labels = ['Young Adults (18-30)', 'Middle-aged (31-45)', 'Older Adults (46-60)', 'Seniors (61+)']
df_clean['Age_Group'] = pd.cut(df_clean['Age'], bins=bins, labels=labels)
age_group_summary = df_clean.groupby('Age_Group', observed=True)[['Annual_Income_k', 'Spending_Score']].mean()
print("\n--- AGE GROUP BEHAVIOR ---")
print(age_group_summary)

# ----------------- VISUALIZATIONS -----------------
# Plot 1: Univariate Distributions
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
num_cols = ['Age', 'Annual_Income_k', 'Spending_Score']
titles = ['Age Distribution (Years)', 'Annual Income (k$)', 'Spending Score (1-100)']
colors = ['#1f77b4', '#2ca02c', '#d62728']

for i, col in enumerate(num_cols):
    sns.histplot(df_clean[col], kde=True, ax=axes[0, i], color=colors[i], bins=15, edgecolor='black', alpha=0.65)
    axes[0, i].set_title(f'Histogram & KDE: {titles[i]}', fontsize=12, fontweight='bold')
    axes[0, i].axvline(df_clean[col].mean(), color='red', linestyle='--', label=f"Mean: {df_clean[col].mean():.1f}")
    axes[0, i].axvline(df_clean[col].median(), color='blue', linestyle=':', label=f"Median: {df_clean[col].median():.1f}")
    axes[0, i].legend(loc='upper right')
    
    sns.boxplot(x=df_clean[col], ax=axes[1, i], color=colors[i], flierprops=dict(markerfacecolor='red', marker='o'))
    axes[1, i].set_title(f'Boxplot: {titles[i]} (Outlier Detection)', fontsize=12, fontweight='bold')

plt.suptitle('Mall Customers: Univariate Distributions & Outlier Analysis', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
p1_path = os.path.join(output_dir, "01_univariate_distributions.png")
plt.savefig(p1_path, bbox_inches='tight')
plt.close()

# Plot 2: Gender Demographics & Spending Comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# Pie / Donut
gender_counts.plot.pie(ax=axes[0], autopct='%1.1f%%', startangle=140, colors=['#ff9999', '#66b3ff'],
                      wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
                      textprops={'fontsize': 12, 'weight': 'bold'})
axes[0].set_ylabel('')
axes[0].set_title('Customer Gender Distribution', fontsize=13, fontweight='bold')

# Boxplot Income by Gender
sns.boxplot(x='Gender', y='Annual_Income_k', data=df_clean, ax=axes[1], palette=['#66b3ff', '#ff9999'])
axes[1].set_title('Annual Income by Gender', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Annual Income (k$)')

# Boxplot Spending by Gender
sns.boxplot(x='Gender', y='Spending_Score', data=df_clean, ax=axes[2], palette=['#66b3ff', '#ff9999'])
axes[2].set_title('Spending Score by Gender', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Spending Score (1-100)')

plt.suptitle('Gender-Based Demographics and Behavioral Metrics', fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout()
p2_path = os.path.join(output_dir, "02_gender_breakdown.png")
plt.savefig(p2_path, bbox_inches='tight')
plt.close()

# Plot 3: The Classic 5 Clusters (Annual Income vs Spending Score)
plt.figure(figsize=(11, 7))
scatter = sns.scatterplot(
    data=df_clean,
    x='Annual_Income_k',
    y='Spending_Score',
    hue='Gender',
    style='Gender',
    s=90,
    palette={'Male': '#1f77b4', 'Female': '#e377c2'},
    alpha=0.9,
    edgecolor='black'
)

# Annotate the 5 clusters
annotations = [
    (25, 80, "Carefree Shoppers\n(Low Income, High Spending)", '#2ca02c'),
    (25, 20, "Sensible Shoppers\n(Low Income, Low Spending)", '#7f7f7f'),
    (55, 50, "Average Middle Class\n(Moderate Income, Moderate Spending)", '#17becf'),
    (88, 85, "High-Value Targets\n(High Income, High Spending)", '#d62728'),
    (88, 18, "Frugal Affluent\n(High Income, Low Spending)", '#9467bd')
]
for x, y, text, color in annotations:
    plt.annotate(text, xy=(x, y), xytext=(x, y),
                 bbox=dict(boxstyle="round,pad=0.5", fc=color, alpha=0.2, ec=color, lw=1.5),
                 fontsize=10, fontweight='bold', ha='center')

plt.title('Annual Income vs Spending Score: Clear 5-Cluster Topology', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Annual Income (k$)', fontsize=12, fontweight='bold')
plt.ylabel('Spending Score (1-100)', fontsize=12, fontweight='bold')
plt.axvline(x=40, color='gray', linestyle=':', alpha=0.5)
plt.axvline(x=70, color='gray', linestyle=':', alpha=0.5)
plt.axhline(y=40, color='gray', linestyle=':', alpha=0.5)
plt.axhline(y=60, color='gray', linestyle=':', alpha=0.5)
plt.tight_layout()
p3_path = os.path.join(output_dir, "03_income_vs_spending.png")
plt.savefig(p3_path, bbox_inches='tight')
plt.close()

# Plot 4: Age vs Spending Score & Age vs Annual Income
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# Age vs Spending Score
sns.regplot(data=df_clean, x='Age', y='Spending_Score', ax=axes[0],
            scatter_kws={'alpha':0.6, 'color': '#2b5c8f', 's': 50}, line_kws={'color':'red', 'lw':2})
axes[0].axvline(40, color='green', linestyle='--', label='Age 40 Cutoff')
axes[0].axhline(60, color='orange', linestyle='--', label='High Spending Ceiling (>60)')
axes[0].set_title('Age vs Spending Score: Generational Spending Gap', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Age (Years)', fontsize=11)
axes[0].set_ylabel('Spending Score (1-100)', fontsize=11)
axes[0].legend(loc='upper right')

# Age vs Annual Income
sns.scatterplot(data=df_clean, x='Age', y='Annual_Income_k', hue='Gender', ax=axes[1],
                palette={'Male': '#1f77b4', 'Female': '#e377c2'}, s=60, alpha=0.8)
sns.lineplot(data=df_clean, x='Age', y='Annual_Income_k', ax=axes[1], color='black', alpha=0.4, ci=None)
axes[1].set_title('Age vs Annual Income: Peak Earning Window (30-50)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Age (Years)', fontsize=11)
axes[1].set_ylabel('Annual Income (k$)', fontsize=11)

plt.suptitle('Bivariate Interactions with Customer Age', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
p4_path = os.path.join(output_dir, "04_age_vs_spending_income.png")
plt.savefig(p4_path, bbox_inches='tight')
plt.close()

# Plot 5: Correlation Heatmap and Pairplot
fig, ax = plt.subplots(figsize=(7, 5))
corr = df_clean[['Age', 'Annual_Income_k', 'Spending_Score']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', vmin=-1, vmax=1,
            square=True, linewidths=1.5, cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Feature Correlation Matrix (Pearson)', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
p5_path = os.path.join(output_dir, "05_correlation_matrix.png")
plt.savefig(p5_path, bbox_inches='tight')
plt.close()

# Pairplot
pairplot_fig = sns.pairplot(df_clean[['Gender', 'Age', 'Annual_Income_k', 'Spending_Score']],
                            hue='Gender', palette={'Male': '#1f77b4', 'Female': '#e377c2'},
                            diag_kind='kde', corner=True, plot_kws={'alpha': 0.7, 's': 40})
pairplot_fig.fig.suptitle('Pairwise Feature Matrix by Gender', fontsize=14, fontweight='bold', y=1.02)
p6_path = os.path.join(output_dir, "06_pairwise_relationships.png")
pairplot_fig.savefig(p6_path, bbox_inches='tight')
plt.close()

# Copy all generated plots to artifact directory
for fname in os.listdir(output_dir):
    src = os.path.join(output_dir, fname)
    dst = os.path.join(artifact_dir, fname)
    shutil.copy2(src, dst)

print("\nPlots successfully saved to:", output_dir, "and copied to artifact directory.")
