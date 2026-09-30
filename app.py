import json
import math
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Mall Customer Segmentation & Clustering AI",
    description="Real-time ML deployment for customer persona prediction, behavioral clustering, and marketing strategy recommendations.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model metadata and artifacts
with open("model_metadata.json", "r", encoding="utf-8") as f:
    MODEL_DATA = json.load(f)

with open("clustered_customers.json", "r", encoding="utf-8") as f:
    DATASET_POINTS = json.load(f)

SCALER_MEAN = MODEL_DATA["scaler"]["mean"]
SCALER_SCALE = MODEL_DATA["scaler"]["scale"]
CLUSTERS = MODEL_DATA["clusters"]

class CustomerPredictionInput(BaseModel):
    annual_income: float = Field(..., ge=1, le=300, description="Annual Income in thousands of USD (k$)")
    spending_score: float = Field(..., ge=1, le=100, description="Spending Score between 1 and 100")
    age: Optional[float] = Field(35.0, ge=10, le=100, description="Age in years")
    gender: Optional[str] = Field("Female", description="Gender (Male / Female)")

class CustomerPredictionOutput(BaseModel):
    cluster_id: int
    persona_id: str
    cluster_name: str
    color: str
    icon: str
    description: str
    marketing_strategy: str
    centroid_income: float
    centroid_spending: float
    distance_to_centroid: float
    confidence_score: float
    features_received: dict

def predict_cluster_for_features(annual_income: float, spending_score: float, age: float = 35.0, gender: str = "Female") -> dict:
    # 1. Standardize features
    scaled_inc = (annual_income - SCALER_MEAN[0]) / SCALER_SCALE[0]
    scaled_spd = (spending_score - SCALER_MEAN[1]) / SCALER_SCALE[1]
    
    # 2. Compute Euclidean distance to each cluster centroid in scaled space
    best_dist = float("inf")
    best_cluster = None
    distances = []
    
    for c in CLUSTERS:
        c_inc_scaled = (c["centroid_income"] - SCALER_MEAN[0]) / SCALER_SCALE[0]
        c_spd_scaled = (c["centroid_spending"] - SCALER_MEAN[1]) / SCALER_SCALE[1]
        dist = math.sqrt((scaled_inc - c_inc_scaled) ** 2 + (scaled_spd - c_spd_scaled) ** 2)
        distances.append(dist)
        if dist < best_dist:
            best_dist = dist
            best_cluster = c
            
    # Softmax-like or inverse-distance confidence
    inv_dists = [1.0 / (d + 1e-4) for d in distances]
    confidence = (1.0 / (best_dist + 1e-4)) / sum(inv_dists)
    
    return {
        "cluster_id": best_cluster["cluster_id"],
        "persona_id": best_cluster["persona_id"],
        "cluster_name": best_cluster["name"],
        "color": best_cluster["color"],
        "icon": best_cluster["icon"],
        "description": best_cluster["description"],
        "marketing_strategy": best_cluster["marketing_strategy"],
        "centroid_income": best_cluster["centroid_income"],
        "centroid_spending": best_cluster["centroid_spending"],
        "distance_to_centroid": round(float(best_dist), 4),
        "confidence_score": round(float(confidence * 100), 1),
        "features_received": {
            "annual_income_k": annual_income,
            "spending_score": spending_score,
            "age": age,
            "gender": gender
        }
    }

@app.post("/api/predict", response_model=CustomerPredictionOutput, tags=["Inference"])
async def predict_customer_segment(input_data: CustomerPredictionInput):
    """Predict customer cluster persona given annual income and spending score."""
    result = predict_cluster_for_features(
        annual_income=input_data.annual_income,
        spending_score=input_data.spending_score,
        age=input_data.age,
        gender=input_data.gender
    )
    return result

@app.get("/api/clusters", tags=["Metadata"])
async def get_clusters():
    """Return all 5 cluster profiles, centroids, and marketing playbooks."""
    return {
        "status": "success",
        "k_clusters": MODEL_DATA["k_clusters"],
        "clusters": CLUSTERS
    }

@app.get("/api/metrics", tags=["Metadata"])
async def get_model_metrics():
    """Return model evaluation metrics including Silhouette Score and Inertia."""
    return {
        "status": "success",
        "model_name": MODEL_DATA["model_name"],
        "metrics": MODEL_DATA["metrics"],
        "features": MODEL_DATA["features"]
    }

@app.get("/api/dataset", tags=["Data"])
async def get_dataset_points():
    """Return the 200 clustered customers for plotting."""
    return {
        "total_records": len(DATASET_POINTS),
        "data": DATASET_POINTS
    }

@app.get("/api/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "service": "Mall Customer Segmentation API", "version": "1.0.0"}

# Single Page Application
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Segmentation & Clustering Intelligence</title>
    <!-- Modern Typography: Inter & Outfit -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-base: #090d16;
            --bg-card: rgba(18, 24, 38, 0.75);
            --bg-card-hover: rgba(28, 36, 56, 0.85);
            --bg-glass: rgba(255, 255, 255, 0.03);
            --border-glow: rgba(255, 255, 255, 0.08);
            --border-active: rgba(56, 189, 248, 0.4);
            
            --primary: #38bdf8;
            --primary-glow: rgba(56, 189, 248, 0.25);
            --accent: #818cf8;
            --accent-green: #10b981;
            --accent-purple: #8b5cf6;
            --accent-pink: #f43f5e;
            --accent-amber: #f59e0b;
            --accent-blue: #0ea5e9;

            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --text-faint: #64748b;
            
            --radius-lg: 16px;
            --radius-md: 12px;
            --radius-sm: 8px;
            --transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-base);
            color: var(--text-main);
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 75%, rgba(139, 92, 246, 0.08) 0%, transparent 45%);
            background-attachment: fixed;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* Top Navbar */
        header {
            border-bottom: 1px solid var(--border-glow);
            backdrop-filter: blur(16px);
            background: rgba(9, 13, 22, 0.8);
            position: sticky;
            top: 0;
            z-index: 50;
            padding: 16px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .brand-container {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #38bdf8, #8b5cf6);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            box-shadow: 0 4px 15px var(--primary-glow);
        }

        .brand-title {
            font-family: 'Outfit', sans-serif;
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(to right, #ffffff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-badge {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
            font-size: 11px;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 20px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .brand-badge::before {
            content: '';
            width: 6px;
            height: 6px;
            background: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 8px #10b981;
        }

        .nav-links {
            display: flex;
            gap: 10px;
        }

        .nav-btn {
            background: transparent;
            color: var(--text-muted);
            border: 1px solid transparent;
            padding: 8px 16px;
            font-size: 14px;
            font-weight: 500;
            border-radius: var(--radius-sm);
            cursor: pointer;
            transition: var(--transition);
        }

        .nav-btn:hover {
            color: var(--text-main);
            background: var(--bg-glass);
            border-color: var(--border-glow);
        }

        .nav-btn.active {
            color: var(--primary);
            background: rgba(56, 189, 248, 0.1);
            border-color: rgba(56, 189, 248, 0.3);
        }

        .api-link {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            color: #94a3b8;
            font-size: 13px;
            text-decoration: none;
            padding: 6px 14px;
            border: 1px solid var(--border-glow);
            border-radius: var(--radius-sm);
            transition: var(--transition);
        }

        .api-link:hover {
            color: var(--primary);
            border-color: var(--primary);
        }

        /* Container */
        .container {
            max-width: 1380px;
            margin: 0 auto;
            padding: 28px 24px;
            flex: 1;
            width: 100%;
        }

        /* Metrics Bar */
        .metrics-banner {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }

        .metric-card {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-glow);
            border-radius: var(--radius-md);
            padding: 16px 20px;
            position: relative;
            overflow: hidden;
            transition: var(--transition);
        }

        .metric-card:hover {
            border-color: var(--border-active);
            transform: translateY(-2px);
        }

        .metric-card::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--primary), transparent);
            opacity: 0.6;
        }

        .metric-label {
            font-size: 12px;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }

        .metric-val {
            font-family: 'Outfit', sans-serif;
            font-size: 26px;
            font-weight: 700;
            color: var(--text-main);
            display: flex;
            align-items: baseline;
            gap: 6px;
        }

        .metric-sub {
            font-size: 12px;
            color: var(--text-faint);
            font-weight: 400;
        }

        /* Main Grid */
        .main-layout {
            display: grid;
            grid-template-columns: 420px 1fr;
            gap: 24px;
        }

        @media (max-width: 1024px) {
            .main-layout {
                grid-template-columns: 1fr;
            }
        }

        /* Card container */
        .glass-card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-glow);
            border-radius: var(--radius-lg);
            padding: 24px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-glow);
        }

        .card-title {
            font-family: 'Outfit', sans-serif;
            font-size: 18px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* Input Controls */
        .input-group {
            margin-bottom: 20px;
        }

        .input-label-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .input-label {
            font-size: 13px;
            font-weight: 500;
            color: var(--text-muted);
        }

        .input-val-badge {
            font-family: 'Outfit', monospace;
            font-size: 14px;
            font-weight: 600;
            color: var(--primary);
            background: rgba(56, 189, 248, 0.1);
            padding: 3px 10px;
            border-radius: 6px;
            border: 1px solid rgba(56, 189, 248, 0.2);
        }

        .range-slider {
            width: 100%;
            height: 6px;
            border-radius: 4px;
            background: #1e293b;
            outline: none;
            -webkit-appearance: none;
            transition: var(--transition);
        }

        .range-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--primary);
            cursor: pointer;
            box-shadow: 0 0 10px var(--primary-glow);
            border: 2px solid #ffffff;
            transition: var(--transition);
        }

        .range-slider::-webkit-slider-thumb:hover {
            transform: scale(1.15);
        }

        .demographics-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin-bottom: 20px;
        }

        .select-input, .num-input {
            width: 100%;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--border-glow);
            border-radius: var(--radius-sm);
            padding: 10px 14px;
            color: var(--text-main);
            font-size: 14px;
            outline: none;
            transition: var(--transition);
        }

        .select-input:focus, .num-input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 2px var(--primary-glow);
        }

        .btn-predict {
            width: 100%;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            color: #0b0f19;
            font-family: 'Outfit', sans-serif;
            font-size: 15px;
            font-weight: 700;
            padding: 14px;
            border: none;
            border-radius: var(--radius-md);
            cursor: pointer;
            transition: var(--transition);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            box-shadow: 0 4px 18px rgba(56, 189, 248, 0.35);
        }

        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(56, 189, 248, 0.5);
        }

        .btn-predict:active {
            transform: translateY(0);
        }

        /* Prediction Result Box */
        .prediction-result {
            margin-top: 24px;
            border-radius: var(--radius-md);
            padding: 18px;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid var(--border-glow);
            animation: fadeIn 0.4s ease-out;
            display: none;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }

        .result-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            font-weight: 700;
            padding: 6px 14px;
            border-radius: 20px;
        }

        .confidence-indicator {
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 500;
        }

        .persona-name {
            font-family: 'Outfit', sans-serif;
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .persona-desc {
            font-size: 13px;
            color: var(--text-muted);
            line-height: 1.5;
            margin-bottom: 14px;
        }

        .strategy-box {
            background: rgba(255, 255, 255, 0.03);
            border-left: 3px solid var(--primary);
            border-radius: 4px;
            padding: 10px 14px;
        }

        .strategy-title {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--primary);
            margin-bottom: 4px;
        }

        .strategy-text {
            font-size: 12px;
            color: #cbd5e1;
            line-height: 1.4;
        }

        /* Chart Canvas */
        .chart-wrapper {
            position: relative;
            height: 520px;
            width: 100%;
        }

        .chart-legend-custom {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 14px;
            justify-content: center;
        }

        .legend-chip {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            color: var(--text-muted);
            cursor: pointer;
            padding: 4px 10px;
            border-radius: 6px;
            border: 1px solid transparent;
            transition: var(--transition);
        }

        .legend-chip:hover {
            background: var(--bg-glass);
            border-color: var(--border-glow);
            color: var(--text-main);
        }

        .legend-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }

        /* Tab Content Control */
        .tab-panel {
            display: none;
        }

        .tab-panel.active {
            display: block;
            animation: fadeIn 0.3s ease;
        }

        /* Cluster Cards Grid */
        .clusters-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }

        .cluster-persona-card {
            background: var(--bg-card);
            border: 1px solid var(--border-glow);
            border-radius: var(--radius-lg);
            padding: 22px;
            position: relative;
            overflow: hidden;
            transition: var(--transition);
        }

        .cluster-persona-card:hover {
            transform: translateY(-4px);
            border-color: var(--border-active);
        }

        .cluster-top {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 14px;
        }

        .cluster-avatar {
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }

        .cluster-tag {
            font-size: 11px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 20px;
        }

        .cluster-name-title {
            font-family: 'Outfit', sans-serif;
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .cluster-stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 16px 0;
            padding: 12px;
            background: rgba(0, 0, 0, 0.25);
            border-radius: var(--radius-sm);
        }

        .stat-item-label {
            font-size: 11px;
            color: var(--text-faint);
            margin-bottom: 2px;
        }

        .stat-item-val {
            font-family: 'Outfit', sans-serif;
            font-size: 15px;
            font-weight: 600;
            color: var(--text-main);
        }

        /* Table Design */
        .table-responsive {
            overflow-x: auto;
            border-radius: var(--radius-md);
            border: 1px solid var(--border-glow);
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 13px;
        }

        .data-table th {
            background: rgba(15, 23, 42, 0.95);
            color: var(--text-muted);
            font-weight: 600;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-glow);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 11px;
        }

        .data-table td {
            padding: 12px 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            color: var(--text-main);
        }

        .data-table tr:hover {
            background: rgba(255, 255, 255, 0.02);
        }

        .filter-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }

        .search-box {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--border-glow);
            border-radius: var(--radius-sm);
            padding: 8px 14px;
            color: var(--text-main);
            font-size: 13px;
            min-width: 250px;
            outline: none;
        }

        .search-box:focus {
            border-color: var(--primary);
        }

        /* Footer */
        footer {
            border-top: 1px solid var(--border-glow);
            padding: 20px 32px;
            text-align: center;
            font-size: 12px;
            color: var(--text-faint);
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <header>
        <div class="brand-container">
            <div class="brand-icon">🛍️</div>
            <div>
                <div class="brand-title">Customer Segmentation Intelligence</div>
                <div style="font-size: 11px; color: var(--text-muted);">K-Means Clustering AI • Production Deployment</div>
            </div>
        </div>

        <nav class="nav-links">
            <button class="nav-btn active" onclick="switchTab('predictor')">🔮 Live Inference</button>
            <button class="nav-btn" onclick="switchTab('analytics')">📊 Cluster Playbooks</button>
            <button class="nav-btn" onclick="switchTab('dataset')">👥 Customer Explorer</button>
            <a href="/docs" target="_blank" class="api-link">⚡ REST API Docs</a>
        </nav>

        <div class="brand-badge">Model Active & Serving</div>
    </header>

    <div class="container">
        <!-- Live Metrics Bar -->
        <div class="metrics-banner">
            <div class="metric-card">
                <div class="metric-label">Clustering Quality</div>
                <div class="metric-val" id="metric-sil">0.5547 <span class="metric-sub">Silhouette</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Optimal Clusters (K)</div>
                <div class="metric-val">5 <span class="metric-sub">Distinct Segments</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Dataset Sample Size</div>
                <div class="metric-val">200 <span class="metric-sub">Active Shoppers</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Inference Latency</div>
                <div class="metric-val" style="color: #34d399;">~1.2 ms <span class="metric-sub">Real-Time</span></div>
            </div>
        </div>

        <!-- TAB 1: PREDICTOR & SCATTER -->
        <div id="tab-predictor" class="tab-panel active">
            <div class="main-layout">
                <!-- Control Panel -->
                <div class="glass-card">
                    <div class="card-header">
                        <div class="card-title">🎯 Customer Profiler</div>
                        <span style="font-size: 12px; color: var(--text-muted);">Real-Time K-Means</span>
                    </div>

                    <form id="predictionForm" onsubmit="event.preventDefault(); runPrediction();">
                        <!-- Annual Income Slider -->
                        <div class="input-group">
                            <div class="input-label-row">
                                <span class="input-label">Annual Income (k$)</span>
                                <span class="input-val-badge" id="val-income">$70k</span>
                            </div>
                            <input type="range" id="incomeInput" class="range-slider" min="15" max="140" value="70" step="1" oninput="updateVal('income', this.value)">
                        </div>

                        <!-- Spending Score Slider -->
                        <div class="input-group">
                            <div class="input-label-row">
                                <span class="input-label">Spending Score (1-100)</span>
                                <span class="input-val-badge" id="val-spending">75 / 100</span>
                            </div>
                            <input type="range" id="spendingInput" class="range-slider" min="1" max="99" value="75" step="1" oninput="updateVal('spending', this.value)">
                        </div>

                        <!-- Age & Gender -->
                        <div class="demographics-row">
                            <div>
                                <label class="input-label" style="display:block; margin-bottom: 6px;">Customer Age</label>
                                <input type="number" id="ageInput" class="num-input" min="18" max="75" value="32">
                            </div>
                            <div>
                                <label class="input-label" style="display:block; margin-bottom: 6px;">Gender</label>
                                <select id="genderInput" class="select-input">
                                    <option value="Female">Female</option>
                                    <option value="Male">Male</option>
                                </select>
                            </div>
                        </div>

                        <button type="submit" class="btn-predict" id="predictBtn">
                            <span>⚡ Classify Customer Segment</span>
                        </button>
                    </form>

                    <!-- Prediction Result Card -->
                    <div id="predictionResult" class="prediction-result">
                        <div class="result-header">
                            <div id="resBadge" class="result-badge">
                                <span id="resIcon">👑</span>
                                <span id="resClusterName">VIP Targets</span>
                            </div>
                            <span class="confidence-indicator" id="resConfidence">94.2% Match</span>
                        </div>
                        <div class="persona-name" id="resTitle">High-Value VIPs</div>
                        <p class="persona-desc" id="resDesc">Loading persona details...</p>
                        
                        <div class="strategy-box">
                            <div class="strategy-title">🚀 Recommended Action Plan</div>
                            <div class="strategy-text" id="resStrategy">Loading marketing actions...</div>
                        </div>
                    </div>
                </div>

                <!-- Interactive Scatter Space -->
                <div class="glass-card">
                    <div class="card-header">
                        <div class="card-title">📍 Behavioral Clustering Topology (Annual Income vs Spending Score)</div>
                        <span style="font-size: 12px; color: var(--primary);">Interactive Canvas</span>
                    </div>

                    <div class="chart-wrapper">
                        <canvas id="clusterScatterChart"></canvas>
                    </div>

                    <div class="chart-legend-custom" id="customLegend"></div>
                </div>
            </div>
        </div>

        <!-- TAB 2: CLUSTER PLAYBOOKS -->
        <div id="tab-analytics" class="tab-panel">
            <div class="clusters-grid" id="clustersContainer">
                <!-- Dynamically populated -->
            </div>
        </div>

        <!-- TAB 3: CUSTOMER EXPLORER -->
        <div id="tab-dataset" class="tab-panel">
            <div class="glass-card">
                <div class="card-header">
                    <div class="card-title">👥 Dataset Explorer (200 Shoppers)</div>
                    <span style="font-size: 12px; color: var(--text-muted);">Clustered Records</span>
                </div>

                <div class="filter-bar">
                    <input type="text" id="tableSearch" class="search-box" placeholder="Search by Customer ID, Gender, or Segment..." oninput="filterTable()">
                    <select id="clusterFilter" class="select-input" style="width: auto;" onchange="filterTable()">
                        <option value="all">All Segments</option>
                    </select>
                </div>

                <div class="table-responsive">
                    <table class="data-table" id="customersTable">
                        <thead>
                            <tr>
                                <th>Customer ID</th>
                                <th>Gender</th>
                                <th>Age</th>
                                <th>Annual Income ($k)</th>
                                <th>Spending Score (1-100)</th>
                                <th>Assigned Segment</th>
                            </tr>
                        </thead>
                        <tbody id="tableBody">
                            <!-- Injected by JS -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <footer>
        Mall Customer Segmentation Intelligence Platform • Powered by FastAPI & K-Means Unsupervised Learning
    </footer>

    <script>
        let scatterChart = null;
        let datasetPoints = [];
        let clusterDefinitions = [];
        let currentPredictionPoint = null;

        // Tab Switching
        function switchTab(tabId) {
            document.querySelectorAll('.tab-panel').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
            
            document.getElementById('tab-' + tabId).classList.add('active');
            event.target.classList.add('active');

            if (tabId === 'predictor' && scatterChart) {
                setTimeout(() => scatterChart.resize(), 100);
            }
        }

        function updateVal(type, val) {
            if (type === 'income') {
                document.getElementById('val-income').innerText = '$' + val + 'k';
            } else if (type === 'spending') {
                document.getElementById('val-spending').innerText = val + ' / 100';
            }
        }

        // Initialize Data & Chart
        async function initApp() {
            try {
                // Fetch clusters
                const clustersRes = await fetch('/api/clusters');
                const clustersData = await clustersRes.json();
                clusterDefinitions = clustersData.clusters;
                renderClusterCards(clusterDefinitions);

                // Fetch dataset
                const dataRes = await fetch('/api/dataset');
                const dataJson = await dataRes.json();
                datasetPoints = dataJson.data;

                // Render Chart & Table
                renderScatterChart(datasetPoints, clusterDefinitions);
                renderTable(datasetPoints);
                populateFilterOptions(clusterDefinitions);

                // Trigger default initial prediction
                runPrediction();
            } catch (err) {
                console.error("Initialization failed:", err);
            }
        }

        // Render Scatter Plot
        function renderScatterChart(points, clusters) {
            const ctx = document.getElementById('clusterScatterChart').getContext('2d');
            
            // Build datasets for each cluster
            const datasets = clusters.map(c => {
                const cPoints = points.filter(p => p.Cluster === c.cluster_id).map(p => ({
                    x: p.Annual_Income_k,
                    y: p.Spending_Score,
                    id: p.CustomerID,
                    age: p.Age,
                    gender: p.Gender,
                    clusterName: c.name
                }));

                return {
                    label: c.name,
                    data: cPoints,
                    backgroundColor: c.color + 'bb',
                    borderColor: c.color,
                    borderWidth: 1,
                    pointRadius: 5.5,
                    pointHoverRadius: 8
                };
            });

            // Add Centroids dataset
            const centroidPoints = clusters.map(c => ({
                x: c.centroid_income,
                y: c.centroid_spending,
                label: 'Centroid: ' + c.name,
                isCentroid: true
            }));

            datasets.push({
                label: 'Cluster Centroids (★)',
                data: centroidPoints,
                backgroundColor: '#fbbf24',
                borderColor: '#ffffff',
                borderWidth: 2,
                pointRadius: 10,
                pointStyle: 'star',
                pointHoverRadius: 13
            });

            // Add placeholder dataset for user live prediction
            datasets.push({
                label: 'Current Inquiry (You)',
                data: [],
                backgroundColor: '#ffffff',
                borderColor: '#38bdf8',
                borderWidth: 3,
                pointRadius: 12,
                pointStyle: 'circle',
                pointHoverRadius: 15
            });

            scatterChart = new Chart(ctx, {
                type: 'scatter',
                data: { datasets: datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 600 },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(15, 23, 42, 0.95)',
                            titleFont: { family: 'Outfit', size: 14, weight: 'bold' },
                            bodyFont: { family: 'Inter', size: 12 },
                            borderColor: 'rgba(255, 255, 255, 0.1)',
                            borderWidth: 1,
                            padding: 12,
                            callbacks: {
                                label: function(context) {
                                    const raw = context.raw;
                                    if (raw.isCentroid) {
                                        return `${raw.label} (Income: $${raw.x}k, Score: ${raw.y})`;
                                    }
                                    if (raw.isQuery) {
                                        return `👉 Query Target: Income $${raw.x}k, Spending ${raw.y}`;
                                    }
                                    return `ID #${raw.id} (${raw.gender}, ${raw.age} yrs) | Inc: $${raw.x}k, Spd: ${raw.y}`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: { display: true, text: 'Annual Income (k$)', color: '#94a3b8', font: { weight: 'bold', size: 12 } },
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#64748b' },
                            min: 10,
                            max: 145
                        },
                        y: {
                            title: { display: true, text: 'Spending Score (1-100)', color: '#94a3b8', font: { weight: 'bold', size: 12 } },
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#64748b' },
                            min: 0,
                            max: 105
                        }
                    }
                }
            });

            // Build custom legend
            const legendEl = document.getElementById('customLegend');
            legendEl.innerHTML = clusters.map(c => `
                <div class="legend-chip">
                    <span class="legend-dot" style="background:${c.color};"></span>
                    <span>${c.name}</span>
                </div>
            `).join('') + `
                <div class="legend-chip">
                    <span class="legend-dot" style="background:#fbbf24;"></span>
                    <span>Centroids</span>
                </div>
            `;
        }

        // Live Prediction Handler
        async function runPrediction() {
            const income = parseFloat(document.getElementById('incomeInput').value);
            const spending = parseFloat(document.getElementById('spendingInput').value);
            const age = parseFloat(document.getElementById('ageInput').value);
            const gender = document.getElementById('genderInput').value;

            try {
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        annual_income: income,
                        spending_score: spending,
                        age: age,
                        gender: gender
                    })
                });

                const data = await res.json();
                displayPredictionResult(data);

                // Update Inquiry Point on Scatter Chart
                if (scatterChart) {
                    const queryDatasetIndex = scatterChart.data.datasets.length - 1;
                    scatterChart.data.datasets[queryDatasetIndex].data = [{
                        x: income,
                        y: spending,
                        isQuery: true
                    }];
                    scatterChart.data.datasets[queryDatasetIndex].borderColor = data.color;
                    scatterChart.update();
                }
            } catch (err) {
                console.error("Prediction failed:", err);
            }
        }

        function displayPredictionResult(res) {
            const resultBox = document.getElementById('predictionResult');
            const badge = document.getElementById('resBadge');
            
            badge.style.backgroundColor = res.color + '25';
            badge.style.border = `1px solid ${res.color}60`;
            badge.style.color = res.color;
            
            document.getElementById('resIcon').innerText = res.icon;
            document.getElementById('resClusterName').innerText = res.cluster_name;
            document.getElementById('resConfidence').innerText = res.confidence_score + '% Confidence';
            document.getElementById('resTitle').innerText = res.cluster_name;
            document.getElementById('resDesc').innerText = res.description;
            document.getElementById('resStrategy').innerText = res.marketing_strategy;

            resultBox.style.display = 'block';
        }

        // Render Playbooks Tab
        function renderClusterCards(clusters) {
            const container = document.getElementById('clustersContainer');
            container.innerHTML = clusters.map(c => `
                <div class="cluster-persona-card" style="border-top: 3px solid ${c.color};">
                    <div class="cluster-top">
                        <div class="cluster-avatar" style="background: ${c.color}20; color: ${c.color};">
                            ${c.icon}
                        </div>
                        <span class="cluster-tag" style="background:${c.color}25; color:${c.color}; border: 1px solid ${c.color}50;">
                            ${c.percentage}% of Mall
                        </span>
                    </div>

                    <div class="cluster-name-title">${c.name}</div>
                    <div style="font-size: 13px; color: var(--text-muted); line-height: 1.4; margin-bottom: 12px;">
                        ${c.description}
                    </div>

                    <div class="cluster-stats-grid">
                        <div>
                            <div class="stat-item-label">Customer Count</div>
                            <div class="stat-item-val">${c.customer_count} shoppers</div>
                        </div>
                        <div>
                            <div class="stat-item-label">Average Age</div>
                            <div class="stat-item-val">${c.mean_age} yrs</div>
                        </div>
                        <div>
                            <div class="stat-item-label">Centroid Income</div>
                            <div class="stat-item-val">$${c.centroid_income}k</div>
                        </div>
                        <div>
                            <div class="stat-item-label">Centroid Spend</div>
                            <div class="stat-item-val">${c.centroid_spending} / 100</div>
                        </div>
                    </div>

                    <div class="strategy-box" style="border-left-color: ${c.color};">
                        <div class="strategy-title" style="color: ${c.color};">Marketing Strategy</div>
                        <div class="strategy-text">${c.marketing_strategy}</div>
                    </div>
                </div>
            `).join('');
        }

        // Render Dataset Table
        function renderTable(points) {
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = points.map(p => `
                <tr>
                    <td><strong>#${p.CustomerID}</strong></td>
                    <td>${p.Gender}</td>
                    <td>${p.Age}</td>
                    <td>$${p.Annual_Income_k}k</td>
                    <td>${p.Spending_Score}</td>
                    <td>
                        <span style="display:inline-flex; align-items:center; gap:6px; font-weight:600; color:${p.Cluster_Color};">
                            <span style="width:8px; height:8px; border-radius:50%; background:${p.Cluster_Color};"></span>
                            ${p.Cluster_Name}
                        </span>
                    </td>
                </tr>
            `).join('');
        }

        function populateFilterOptions(clusters) {
            const select = document.getElementById('clusterFilter');
            clusters.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c.cluster_id;
                opt.innerText = c.name;
                select.appendChild(opt);
            });
        }

        function filterTable() {
            const q = document.getElementById('tableSearch').value.toLowerCase();
            const clusterVal = document.getElementById('clusterFilter').value;

            const filtered = datasetPoints.filter(p => {
                const matchesText = p.CustomerID.toString().includes(q) ||
                                    p.Gender.toLowerCase().includes(q) ||
                                    p.Cluster_Name.toLowerCase().includes(q);
                const matchesCluster = clusterVal === 'all' || p.Cluster.toString() === clusterVal;
                return matchesText && matchesCluster;
            });

            renderTable(filtered);
        }

        window.onload = initApp;
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def serve_dashboard():
    """Serve the interactive Customer Segmentation web dashboard."""
    return HTML_CONTENT

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=False)
