import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Food Delivery ML Dashboard",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0f1117; }

.metric-card {
    background: linear-gradient(135deg, #1e2130 0%, #252836 100%);
    border: 1px solid #2e3348;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-label { font-size: 12px; color: #8892a4; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 4px; }
.metric-value { font-size: 28px; font-weight: 600; color: #f0f4ff; }
.metric-sub   { font-size: 12px; color: #5a6478; margin-top: 2px; }

.section-header {
    font-size: 18px; font-weight: 600; color: #c8d0e7;
    border-left: 3px solid #4f8ef7;
    padding-left: 12px; margin: 1.5rem 0 1rem 0;
}

.tag {
    display: inline-block;
    background: #1e2a42; color: #4f8ef7;
    border: 1px solid #2a3d5e; border-radius: 6px;
    font-size: 11px; padding: 2px 8px; margin: 2px;
    font-family: 'DM Mono', monospace;
}

.winner-badge {
    background: linear-gradient(90deg, #1e3a1e, #1a3a1a);
    border: 1px solid #2d6a2d; border-radius: 8px;
    padding: 0.5rem 1rem; color: #4caf50;
    font-size: 13px; font-weight: 500;
    display: inline-block; margin-top: 4px;
}

.predict-box {
    background: linear-gradient(135deg, #1a2035, #1e2540);
    border: 1px solid #2e3a5a; border-radius: 14px;
    padding: 1.5rem; margin-top: 1rem;
}
.predict-result {
    font-size: 48px; font-weight: 600;
    color: #4f8ef7; text-align: center; margin: 1rem 0 0.2rem 0;
}
.predict-label { text-align: center; color: #8892a4; font-size: 14px; }

.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #1e2130; border-radius: 8px; border: 1px solid #2e3348;
    color: #8892a4; padding: 6px 18px; font-size: 14px;
}
.stTabs [aria-selected="true"] {
    background: #1e3060 !important; border-color: #4f8ef7 !important; color: #4f8ef7 !important;
}
</style>
""", unsafe_allow_html=True)


# ── DATA LOADING & PIPELINE (CACHED) ─────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("22_food_delivery.csv")
    df_clean = df.drop(columns=["id"])
    df_clean = df_clean[(df_clean["delivery_time_min"] >= 0) & (df_clean["delivery_time_min"] <= 300)]

    X = df_clean.drop(columns=["delivery_time_min"])
    y = df_clean["delivery_time_min"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), ["distance_km", "prep_time_min"]),
        ("cat", OrdinalEncoder(categories=[["low", "medium", "high"]]), ["traffic"]),
        ("bin", "passthrough", ["raining"])
    ])

    models = {
        "Linear Regression":    LinearRegression(),
        "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting":    GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    pipes, results = {}, []
    for name, model in models.items():
        pipe = Pipeline([("pre", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        pipes[name] = pipe
        results.append({
            "Model": name,
            "RMSE":  round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
            "MAE":   round(mean_absolute_error(y_test, y_pred), 3),
            "R²":    round(r2_score(y_test, y_pred), 3),
            "y_pred": y_pred
        })

    return df_clean, X_train, X_test, y_train, y_test, pipes, results

df, X_train, X_test, y_train, y_test, pipes, results = load_and_train()
results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "y_pred"} for r in results])
best_model_name = results_df.loc[results_df["R²"].idxmax(), "Model"]
best_pipe = pipes[best_model_name]
best_result = next(r for r in results if r["Model"] == best_model_name)


# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍔 ML Dashboard")
    st.markdown("**Food Delivery Time Prediction**")
    st.markdown("---")
    st.markdown("#### Dataset Info")
    st.markdown(f"<span class='tag'>1000 rows</span> <span class='tag'>6 columns</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='tag'>Regression</span> <span class='tag'>4 features</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### Best Model")
    st.markdown(f"<div class='winner-badge'>🏆 {best_model_name}</div>", unsafe_allow_html=True)
    st.markdown(f"R² = **{best_result['R²']}** | RMSE = **{best_result['RMSE']}**")
    st.markdown("---")
    st.caption("ANN & ML · MSE-2 · Dataset 22")


# ── HEADER ───────────────────────────────────────────────────
st.markdown("# 🍔 Food Delivery Time — ML Pipeline")
st.markdown("End-to-end regression pipeline: EDA → Preprocessing → Training → Evaluation → Prediction")
st.markdown("---")

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA", "⚙️ Preprocessing", "🤖 Model Comparison", "📈 Feature Importance", "🎯 Predict"
])


# ════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)

    raw = pd.read_csv("22_food_delivery.csv")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown("<div class='metric-card'><div class='metric-label'>Total Rows</div><div class='metric-value'>1,000</div></div>", unsafe_allow_html=True)
    with c2: st.markdown("<div class='metric-card'><div class='metric-label'>Features</div><div class='metric-value'>4</div><div class='metric-sub'>after dropping id</div></div>", unsafe_allow_html=True)
    with c3: st.markdown("<div class='metric-card'><div class='metric-label'>Missing Values</div><div class='metric-value'>0</div></div>", unsafe_allow_html=True)
    with c4: st.markdown("<div class='metric-card'><div class='metric-label'>Task Type</div><div class='metric-value' style='font-size:18px;'>Regression</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Target Variable Distribution</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.histogram(raw, x="delivery_time_min", nbins=50, color_discrete_sequence=["#4f8ef7"],
                           title="Raw delivery_time_min (with outliers)")
        fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#161925",
                          font_color="#c8d0e7", title_font_size=14, margin=dict(t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.histogram(df, x="delivery_time_min", nbins=50, color_discrete_sequence=["#4caf50"],
                            title="Cleaned delivery_time_min (outliers removed)")
        fig2.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#161925",
                           font_color="#c8d0e7", title_font_size=14, margin=dict(t=40,b=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-header'>Feature Distributions</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = px.histogram(df, x="distance_km", nbins=30, color_discrete_sequence=["#f7984f"],
                           title="Distance (km)")
        fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#161925", font_color="#c8d0e7",
                          title_font_size=13, margin=dict(t=35,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x="prep_time_min", nbins=25, color_discrete_sequence=["#e07ef7"],
                           title="Prep Time (min)")
        fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#161925", font_color="#c8d0e7",
                          title_font_size=13, margin=dict(t=35,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        tc = df["traffic"].value_counts().reindex(["low","medium","high"])
        fig = px.bar(x=tc.index, y=tc.values, color=tc.index,
                     color_discrete_map={"low":"#4caf50","medium":"#f7984f","high":"#e05c5c"},
                     title="Traffic Level Counts")
        fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#161925", font_color="#c8d0e7",
                          title_font_size=13, showlegend=False, margin=dict(t=35,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>Correlation Heatmap</div>", unsafe_allow_html=True)
    num_df = df[["distance_km", "prep_time_min", "raining", "delivery_time_min"]]
    corr = num_df.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Blues",
                    title="Correlation Matrix (numeric features)")
    fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#161925", font_color="#c8d0e7",
                      title_font_size=14, margin=dict(t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — PREPROCESSING
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Outlier Detection & Removal</div>", unsafe_allow_html=True)

    raw2 = pd.read_csv("22_food_delivery.csv")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before Cleaning**")
        st.metric("Min delivery_time_min", "-2.1 min", delta="❌ Impossible", delta_color="inverse")
        st.metric("Max delivery_time_min", "765.3 min", delta="❌ Extreme outlier", delta_color="inverse")
        st.metric("Rows", "1000")
    with col2:
        st.markdown("**After Cleaning**")
        st.metric("Min delivery_time_min", f"{df['delivery_time_min'].min():.1f} min", delta="✅ Valid")
        st.metric("Max delivery_time_min", f"{df['delivery_time_min'].max():.1f} min", delta="✅ Reasonable")
        st.metric("Rows", len(df), delta=f"-{1000-len(df)} removed")

    st.markdown("---")
    st.markdown("<div class='section-header'>Encoding Strategy</div>", unsafe_allow_html=True)

    enc_data = {
        "Feature": ["distance_km", "prep_time_min", "traffic", "raining"],
        "Type": ["Numeric", "Numeric", "Categorical (Ordinal)", "Binary"],
        "Encoding": ["StandardScaler", "StandardScaler", "OrdinalEncoder (low=0, medium=1, high=2)", "Passthrough (already 0/1)"],
        "Reason": [
            "Different scale — normalise for Linear Regression",
            "Different scale — normalise for Linear Regression",
            "Has natural order: low < medium < high",
            "Already numeric binary — no encoding needed"
        ]
    }
    st.dataframe(pd.DataFrame(enc_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("<div class='section-header'>Train / Test Split</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown("<div class='metric-card'><div class='metric-label'>Train Set</div><div class='metric-value'>{}</div><div class='metric-sub'>80%</div></div>".format(len(X_train)), unsafe_allow_html=True)
    with col2: st.markdown("<div class='metric-card'><div class='metric-label'>Test Set</div><div class='metric-value'>{}</div><div class='metric-sub'>20%</div></div>".format(len(X_test)), unsafe_allow_html=True)
    with col3: st.markdown("<div class='metric-card'><div class='metric-label'>Random State</div><div class='metric-value'>42</div><div class='metric-sub'>reproducible</div></div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Model Performance Metrics</div>", unsafe_allow_html=True)

    cols = st.columns(3)
    colors_map = {"Linear Regression": "#f7984f", "Random Forest": "#4f8ef7", "Gradient Boosting": "#4caf50"}
    for i, row in results_df.iterrows():
        badge = " 🏆" if row["Model"] == best_model_name else ""
        mcolor = colors_map[row["Model"]]
        with cols[i]:
            st.markdown(f"**{row['Model']}{badge}**")
            st.markdown(f"<div class='metric-card'><div class='metric-label'>RMSE</div><div class='metric-value' style='color:{mcolor};font-size:22px'>{row['RMSE']}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><div class='metric-label'>MAE</div><div class='metric-value' style='font-size:22px'>{row['MAE']}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><div class='metric-label'>R²</div><div class='metric-value' style='color:{mcolor};font-size:22px'>{row['R²']}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Metric Comparison Charts</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    model_colors = [colors_map[m] for m in results_df["Model"]]

    for col, metric in zip([col1, col2, col3], ["RMSE", "MAE", "R²"]):
        with col:
            fig = px.bar(results_df, x="Model", y=metric, color="Model",
                         color_discrete_sequence=model_colors, title=metric)
            fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#161925", font_color="#c8d0e7",
                              showlegend=False, title_font_size=14, margin=dict(t=35,b=10))
            fig.update_xaxes(tickangle=-15)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>Actual vs Predicted — All Models</div>", unsafe_allow_html=True)
    fig = make_subplots(rows=1, cols=3, subplot_titles=[r["Model"] for r in results])
    for i, result in enumerate(results):
        fig.add_trace(go.Scatter(x=y_test, y=result["y_pred"], mode="markers",
                                 marker=dict(color=list(colors_map.values())[i], size=4, opacity=0.6),
                                 name=result["Model"]), row=1, col=i+1)
        lim = [min(y_test.min(), result["y_pred"].min()), max(y_test.max(), result["y_pred"].max())]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                 line=dict(color="#555", dash="dash", width=1),
                                 showlegend=False), row=1, col=i+1)

    fig.update_layout(height=380, paper_bgcolor="#0f1117", plot_bgcolor="#161925",
                      font_color="#c8d0e7", showlegend=False, margin=dict(t=40,b=20))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 4 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>Feature Importance — Gradient Boosting</div>", unsafe_allow_html=True)

    feature_names = ["distance_km", "prep_time_min", "traffic", "raining"]
    gb_pipe = pipes["Gradient Boosting"]
    importances = gb_pipe.named_steps["model"].feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=True)

    fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Blues",
                 title="Feature Importance (Gradient Boosting)")
    fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#161925", font_color="#c8d0e7",
                      title_font_size=14, coloraxis_showscale=False, margin=dict(t=40,b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>Feature Importance — Random Forest</div>", unsafe_allow_html=True)
    rf_pipe = pipes["Random Forest"]
    rf_imp = rf_pipe.named_steps["model"].feature_importances_
    rf_fi_df = pd.DataFrame({"Feature": feature_names, "Importance": rf_imp}).sort_values("Importance", ascending=True)

    fig2 = px.bar(rf_fi_df, x="Importance", y="Feature", orientation="h",
                  color="Importance", color_continuous_scale="Oranges",
                  title="Feature Importance (Random Forest)")
    fig2.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#161925", font_color="#c8d0e7",
                       title_font_size=14, coloraxis_showscale=False, margin=dict(t=40,b=20))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.info("💡 **Insight:** `distance_km` and `traffic` are consistently the top 2 features across both tree models. `raining` has the lowest importance — its effect may be partially captured by traffic patterns.")


# ════════════════════════════════════════════════════════════
# TAB 5 — PREDICT
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>Live Delivery Time Predictor</div>", unsafe_allow_html=True)
    st.markdown("Adjust the inputs below and get an instant prediction from all 3 models.")

    col1, col2 = st.columns([1, 1])

    with col1:
        distance   = st.slider("📍 Distance (km)", 0.5, 15.0, 5.0, 0.1)
        prep_time  = st.slider("🍳 Prep Time (min)", 5, 44, 20, 1)
        traffic    = st.selectbox("🚦 Traffic Level", ["low", "medium", "high"])
        raining    = st.radio("🌧️ Raining?", ["No", "Yes"], horizontal=True)
        rain_val   = 1 if raining == "Yes" else 0

    input_df = pd.DataFrame({
        "distance_km":   [distance],
        "prep_time_min": [prep_time],
        "traffic":       [traffic],
        "raining":       [rain_val]
    })

    with col2:
        st.markdown("<div class='predict-box'>", unsafe_allow_html=True)
        st.markdown("**Predictions from all models:**")
        st.markdown("")

        pred_colors = {"Linear Regression": "#f7984f", "Random Forest": "#4f8ef7", "Gradient Boosting": "#4caf50"}
        for name, pipe in pipes.items():
            pred = pipe.predict(input_df)[0]
            badge = " 🏆" if name == best_model_name else ""
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"background:#1a2035;border-radius:8px;padding:10px 16px;margin-bottom:8px;"
                f"border:1px solid #2e3a5a'>"
                f"<span style='font-size:13px;color:#8892a4'>{name}{badge}</span>"
                f"<span style='font-size:22px;font-weight:600;color:{pred_colors[name]}'>{pred:.1f} min</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        best_pred = pipes[best_model_name].predict(input_df)[0]
        st.markdown(f"<div class='predict-result'>{best_pred:.1f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='predict-label'>minutes estimated (best model: {best_model_name})</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Your input:**")
    st.dataframe(input_df, use_container_width=True, hide_index=True)
