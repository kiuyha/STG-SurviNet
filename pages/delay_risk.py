import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import streamlit.components.v1 as components

PRIMARY = "#395886"
ACCENT_GREEN = "#3f6735"
ACCENT_GOLD = "#d4a843"
WHITE = "#ffffff"


def render(df: pd.DataFrame, geojson: dict, gdf, base_dir: str):
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, {PRIMARY}, #2a4268);
        border-radius: 14px;
        padding: 28px 36px;
        margin-bottom: 28px;
    '>
        <h2 style='color: white !important; margin: 0 0 6px 0; font-size: 1.6rem;'>
            Delay Risk Analysis
        </h2>
        <p style='color: rgba(255,255,255,0.75); margin: 0; font-size: 0.88rem;'>
            Analyzing delay risk patterns learned by STG-SurviNet across complaint categories
            and community areas. Higher delay-risk score equals higher predicted risk of prolonged resolution.
        </p>
    </div>
    """, unsafe_allow_html=True)

    #  Delay Risk by Complaint Category 
    st.markdown("### Mean Delay-Risk Score by Complaint Category")

    category_risk = pd.DataFrame({
        "Category": [
            "Tree Debris Clean-Up Request",
            "Pothole in Street Complaint",
            "Street Light Out Complaint",
            "Traffic Signal Out Complaint",
        ],
        "Records": [60384, 243487, 188897, 48857],
        "Mean Duration": [13.85, 22.43, 15.68, 21.54],
        "Median Duration": [5, 5, 2, 0],
        "Mean Delay-Risk": [0.2005, 0.0186, -0.2680, -0.5163],
    })

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        # Positive risk score means slower resolution (red), negative means faster resolution (green)
        colors = ["#c75b5b" if v >= 0 else ACCENT_GREEN for v in category_risk["Mean Delay-Risk"]]

        fig_cat = go.Figure()
        fig_cat.add_trace(go.Bar(
            x=category_risk["Category"],
            y=category_risk["Mean Delay-Risk"],
            marker_color=colors,
            text=[f"{v:+.4f}" for v in category_risk["Mean Delay-Risk"]],
            textposition="outside",
            textfont=dict(size=12),
            hovertemplate="<b>%{x}</b><br>Delay-Risk: %{y:.4f}<extra></extra>",
        ))
        fig_cat.update_layout(
            title="Mean Delay-Risk Score by Complaint Category",
            yaxis_title="Mean Delay-Risk Score",
            font=dict(family="Inter"),
            title_font=dict(size=16),
            margin=dict(t=60, b=80),
            xaxis=dict(tickangle=-15),
            yaxis=dict(zeroline=True, zerolinewidth=1),
            height=450,
        )
        st.plotly_chart(fig_cat, width='stretch')

    with col_table:
        st.dataframe(
            category_risk.style.format({
                "Records": "{:,.0f}",
                "Mean Duration": "{:.2f} days",
                "Median Duration": "{:.0f} days",
                "Mean Delay-Risk": "{:+.4f}",
            }),
            width='stretch',
            hide_index=True,
        )
        
        st.markdown(f"""
        <div style='
            background: var(--background-color-secondary, #e8ebd4);
            color: var(--text-color, #1a1a1a);
            border-radius: 10px;
            padding: 16px;
            font-size: 0.85rem;
            line-height: 1.5;
            margin-top: 10px;
        '>
            <strong>Interpretation Framework:</strong><br>
            <strong>Positive delay-risk value</strong>: higher predicted delay context (lower hazard metrics indicating slower local resolutions).<br><br>
            <strong>Negative delay-risk value</strong>: lower predicted delay context (higher hazard metrics indicating faster local resolutions).<br><br>
            Tree Debris metrics reflect highest risk values (+0.2005) regardless of carrying a lower historical baseline mean duration profile.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    #  Community Area Delay Risk Map - Pre-rendered HTML Integration
    st.markdown("### Community Area Delay-Risk Map")
    
    html_path_map = os.path.join(base_dir, "community_area_delay_risk_map.html")
    if os.path.exists(html_path_map):
        with open(html_path_map, "r", encoding="utf-8") as f:
            html_map_content = f.read()
        components.html(html_map_content, height=660, scrolling=False)
    else:
        st.error("HTML representation file 'community_area_delay_risk_map.html' could not be located inside the execution directory.")

    st.divider()

    #  Workload vs Delay Risk Scatter - Pre-rendered HTML Integration
    st.markdown("### Community Area: Delay Risk vs 60-Day Workload")
    
    html_path_scatter = os.path.join(base_dir, "community_area_delay_risk_vs_workload.html")
    if os.path.exists(html_path_scatter):
        with open(html_path_scatter, "r", encoding="utf-8") as f:
            html_scatter_content = f.read()
        components.html(html_scatter_content, height=510, scrolling=False)
    else:
        st.error("HTML representation file 'community_area_delay_risk_vs_workload.html' could not be located inside the execution directory.")

    st.divider()

    #  Top Delay-Risk Areas DataFrame Processing Section
    st.markdown("### 🔝 Top 10 Highest Delay-Risk Community Areas")

    area_risk = df.groupby("community_area").agg(
        records=("duration_days", "size"),
        mean_duration=("duration_days", "mean"),
        median_duration=("duration_days", "median"),
        event_rate=("event", "mean"),
    ).reset_index()

    area_risk["delay_risk_score"] = (area_risk["mean_duration"] - area_risk["mean_duration"].mean()) / area_risk["mean_duration"].std()

    TEMPORAL_WINDOW = 60
    max_date = pd.to_datetime(df["created_date"]).max().normalize()
    start_date = max_date - pd.Timedelta(days=TEMPORAL_WINDOW - 1)
    window_df = df[pd.to_datetime(df["created_date"]) >= start_date]
    workload = window_df.groupby("community_area").size().reset_index(name="workload_60day")

    area_combined = area_risk.merge(workload, on="community_area", how="left")
    area_combined["workload_60day"] = area_combined["workload_60day"].fillna(0)
    area_combined = area_combined.merge(
        gdf[["area_numbe", "community"]],
        left_on="community_area", right_on="area_numbe", how="left",
    )

    top_risk = area_combined.nlargest(10, "delay_risk_score")[
        ["community", "community_area", "records", "mean_duration", "median_duration", "delay_risk_score", "workload_60day"]
    ].rename(columns={
        "community": "Community Area",
        "community_area": "Area #",
        "records": "Records",
        "mean_duration": "Mean Duration (days)",
        "median_duration": "Median Duration (days)",
        "delay_risk_score": "Delay-Risk Score",
        "workload_60day": "60-Day Workload",
    })

    st.dataframe(
        top_risk.style.format({
            "Records": "{:,.0f}",
            "Mean Duration (days)": "{:.1f}",
            "Median Duration (days)": "{:.0f}",
            "Delay-Risk Score": "{:+.4f}",
            "60-Day Workload": "{:,.0f}",
        }).background_gradient(subset=["Delay-Risk Score"], cmap="RdYlGn_r"),
        width='stretch',
        hide_index=True,
    )

    st.success("""
    Pattern Summary: The delay-risk analysis reveals that high delay-risk is not simply a function of
    high workload. Some areas with moderate complaint volume still exhibit high delay-risk, suggesting
    structural or resource-related factors beyond volume alone drive resolution delays.
    """)