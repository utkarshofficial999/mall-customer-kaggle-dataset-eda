import json
import joblib
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Mall_Customers.csv")
df_clean = df.rename(columns={
    'Genre': 'Gender',
    'Annual Income (k$)': 'Annual_Income_k',
    'Spending Score (1-100)': 'Spending_Score'
})

# Feature selection
feature_cols = ['Annual_Income_k', 'Spending_Score']
X = df_clean[feature_cols].values

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optimal K-Means Training (K=5) with K-Means++ and multiple restarts
best_inertia = np.inf
best_centers = None
best_labels = None

np.random.seed(42)
for i in range(25):
    centers, labels = kmeans2(X_scaled, 5, minit='++', seed=42 + i)
    inertia = np.sum((X_scaled - centers[labels]) ** 2)
    if inertia < best_inertia:
        best_inertia = inertia
        best_centers = centers
        best_labels = labels

df_clean['Cluster'] = best_labels
sil_score = silhouette_score(X_scaled, best_labels)
db_score = davies_bouldin_score(X_scaled, best_labels)
ch_score = calinski_harabasz_score(X_scaled, best_labels)

# Compute unscaled centroids
raw_centroids = scaler.inverse_transform(best_centers)

# Map clusters to personas dynamically based on centroid coordinates
personas_meta = {}
cluster_map = {}

for idx, (inc, spd) in enumerate(raw_centroids):
    if inc > 65 and spd > 60:
        p_id = "vip"
        name = "High-Value VIPs"
        color = "#10b981" # Emerald Green
        desc = "High annual income ($70k-$137k) with high spending score (60-100). Prime targets for premium services."
        strategy = "Invite to exclusive loyalty clubs, VIP concierge, private product previews, and luxury brand collaborations."
        icon = "👑"
    elif inc > 65 and spd <= 40:
        p_id = "frugal_affluent"
        name = "Frugal Affluent (Savers)"
        color = "#8b5cf6" # Purple
        desc = "High annual income ($70k-$137k) with conservative spending (1-40). Wealthy but value-driven."
        strategy = "Target with quality assurance, durable products, premium warranties, investment-grade items, and prestige value."
        icon = "💎"
    elif inc < 45 and spd > 60:
        p_id = "carefree"
        name = "Carefree Spenders"
        color = "#f43f5e" # Rose Pink
        desc = "Modest annual income ($15k-$40k) but high spending score (60-100). Youthful trendsetters and impulse buyers."
        strategy = "Promote trendy apparel, limited edition pop-ups, buy-now-pay-later (BNPL) financing, and social media flash sales."
        icon = "⚡"
    elif inc < 45 and spd <= 40:
        p_id = "budget"
        name = "Sensible Budget Savers"
        color = "#f59e0b" # Amber
        desc = "Modest annual income ($15k-$40k) and conservative spending (1-40). Economically prudent shoppers."
        strategy = "Offer bundle savings, discount coupons, loyalty cashback, clearance promotions, and essential product lines."
        icon = "🛡️"
    else:
        p_id = "middle_class"
        name = "Mainstream Middle Class"
        color = "#0ea5e9" # Sky Blue
        desc = "Moderate annual income ($40k-$70k) with moderate spending score (40-60). The steady retail backbone."
        strategy = "Deploy family weekend specials, seasonal catalog launches, loyalty points rewards, and popular brand deals."
        icon = "🎯"
        
    cluster_map[idx] = {
        "cluster_id": int(idx),
        "persona_id": p_id,
        "name": name,
        "color": color,
        "icon": icon,
        "description": desc,
        "marketing_strategy": strategy,
        "centroid_income": round(float(inc), 2),
        "centroid_spending": round(float(spd), 2),
        "centroid_scaled": [float(c) for c in best_centers[idx]]
    }

df_clean['Cluster_Name'] = df_clean['Cluster'].apply(lambda x: cluster_map[x]['name'])
df_clean['Cluster_Color'] = df_clean['Cluster'].apply(lambda x: cluster_map[x]['color'])

# Summary per cluster
cluster_summaries = []
for idx in range(5):
    sub = df_clean[df_clean['Cluster'] == idx]
    c_info = cluster_map[idx]
    cluster_summaries.append({
        "cluster_id": idx,
        "name": c_info['name'],
        "persona_id": c_info['persona_id'],
        "color": c_info['color'],
        "icon": c_info['icon'],
        "description": c_info['description'],
        "marketing_strategy": c_info['marketing_strategy'],
        "centroid_income": c_info['centroid_income'],
        "centroid_spending": c_info['centroid_spending'],
        "customer_count": int(len(sub)),
        "percentage": round(float(len(sub) / len(df_clean) * 100), 1),
        "mean_age": round(float(sub['Age'].mean()), 1),
        "mean_income": round(float(sub['Annual_Income_k'].mean()), 1),
        "mean_spending": round(float(sub['Spending_Score'].mean()), 1),
        "female_ratio": round(float((sub['Gender'] == 'Female').mean() * 100), 1)
    })

# Export model artifacts
model_metadata = {
    "model_name": "KMeans_Customer_Segmentation",
    "version": "1.0.0",
    "k_clusters": 5,
    "features": feature_cols,
    "metrics": {
        "silhouette_score": round(float(sil_score), 4),
        "inertia_wcss": round(float(best_inertia), 4),
        "davies_bouldin_index": round(float(db_score), 4),
        "calinski_harabasz_index": round(float(ch_score), 4),
        "sample_size": len(df_clean)
    },
    "scaler": {
        "mean": [float(m) for m in scaler.mean_],
        "scale": [float(s) for s in scaler.scale_]
    },
    "clusters": cluster_summaries
}

with open("model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(model_metadata, f, indent=2)

# Save labeled dataset as JSON for API consumption
dataset_records = df_clean[['CustomerID', 'Gender', 'Age', 'Annual_Income_k', 'Spending_Score', 'Cluster', 'Cluster_Name', 'Cluster_Color']].to_dict(orient='records')
with open("clustered_customers.json", "w", encoding="utf-8") as f:
    json.dump(dataset_records, f, indent=2)

print("Model successfully trained & serialized:")
print(f"  - Silhouette Score: {sil_score:.4f}")
print(f"  - WCSS (Inertia): {best_inertia:.4f}")
print(f"  - Clusters mapped: {len(cluster_summaries)}")
