import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

PRIMARY = "#395886"
ACCENT_GREEN = "#3f6735"
ACCENT_GOLD = "#d4a843"
WHITE = "#ffffff"


def render(base_dir: str):
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, {PRIMARY}, #2a4268);
        border-radius: 14px;
        padding: 28px 36px;
        margin-bottom: 28px;
    '>
        <h2 style='color: white !important; margin: 0 0 6px 0; font-size: 1.6rem;'>
            Model Evaluation & Comparison
        </h2>
        <p style='color: rgba(255,255,255,0.75); margin: 0; font-size: 0.88rem;'>
            Comprehensive evaluation of STG-SurviNet against classical survival baselines,
            plus ablation study to understand each component's contribution.
        </p>
    </div>
    """, unsafe_allow_html=True)

    #  Metric Explanations 
    st.markdown("### Evaluation Metrics Explained")

    metrics_info = [
        {
            "name": "C-Index (Concordance Index)",
            "icon": "track_changes",
            "color": PRIMARY,
            "desc": "Measures the model's ability to correctly rank pairs of observations by their predicted risk. A C-Index of 1.0 means perfect ranking; 0.5 is random.",
            "target": "≥ 0.75",
            "interpretation": "Higher is better. STG-SurviNet achieves 0.7075, moderate ranking ability, meaning ~70.75% of patient pairs are correctly ordered by predicted risk.",
        },
        {
            "name": "Integrated Brier Score (IBS)",
            "icon": "architecture",
            "color": ACCENT_GREEN,
            "desc": "Measures the average squared difference between predicted survival probabilities and actual outcomes across time. Integrates calibration and discrimination.",
            "target": "< 0.20",
            "interpretation": "Lower is better. STG-SurviNet achieves 0.1209, relatively low prediction error across all evaluated time points.",
        },
        {
            "name": "D-Calibration",
            "icon": "rule",
            "color": ACCENT_GOLD,
            "desc": "Tests whether predicted survival probabilities are uniformly distributed among subjects who experience the event. Uses a chi-squared goodness-of-fit test.",
            "target": "p > 0.05",
            "interpretation": "A p-value > 0.05 indicates well-calibrated probabilities. All models produce p = 0.0000, meaning predicted probabilities are NOT well calibrated.",
        },
        {
            "name": "Mean Absolute Error (MAE)",
            "icon": "straighten",
            "color": "#6b8cba",
            "desc": "Average absolute difference (in days) between predicted median survival time and actual resolution duration for completed complaints.",
            "target": "< 7 days",
            "interpretation": "Lower is better. STG-SurviNet achieves 13.68 days, predictions are off by ~2 weeks on average. Still above target.",
        },
    ]

    cols = st.columns(2)
    for idx, metric in enumerate(metrics_info):
        with cols[idx % 2]:
            st.markdown(f"""
            <div style='
                background: {WHITE};
                border-radius: 14px;
                padding: 24px;
                margin-bottom: 16px;
                box-shadow: 0 4px 20px rgba(57,88,134,0.08);
                border-left: 5px solid {metric["color"]};
                display: flex;
                flex-direction: column;
                align-items: center;
            '>
                <div style='display:flex; align-items:center; margin-bottom: 10px;'>
                    <span class='material-icons' style='font-size: 1.6rem; color: {metric["color"]}; margin-right: 10px;'>{metric["icon"]}</span>
                    <h4 style='color: {metric["color"]}; margin: 0;'>{metric["name"]}</h4>
                </div>
                <p style='color: #555; font-size: 0.84rem; line-height: 1.6; margin-bottom: 10px;'>
                    {metric["desc"]}
                </p>
                <div style='display:flex; gap: 16px; margin-top: 8px;'>
                    <div style='background: #f0f4e8; padding: 8px 14px; border-radius: 8px;'>
                        <span style='font-size: 0.75rem; color: #888;'>TARGET</span><br>
                        <span style='font-size: 0.95rem; font-weight: 700; color: {metric["color"]};'>{metric["target"]}</span>
                    </div>
                </div>
                <p style='color: #666; font-size: 0.8rem; margin-top: 10px; font-style: italic;'>
                    {metric["interpretation"]}
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    #  Model Comparison 
    st.markdown("### 🏆 Classical Model Comparison")

    comparison_data = pd.DataFrame({
        "Model": ["STG-SurviNet", "Cox PH", "Random Survival Forest", "Kaplan-Meier"],
        "C-Index": [0.7075, 0.6550, 0.5273, 0.5000],
        "IBS": [0.1209, 0.1417, 0.1163, 0.1581],
        "D-Cal p-value": [0.0000, 0.0000, 0.0000, 0.0000],
        "MAE (days)": [13.68, 15.01, 13.42, 15.55],
        "Type": ["Deep Learning Hybrid", "Classical Parametric", "Classical Non-parametric", "Non-parametric Baseline"],
    })

    st.dataframe(
        comparison_data.style
        .highlight_max(subset=["C-Index"], color="#d4edda")
        .highlight_min(subset=["IBS", "MAE (days)"], color="#d4edda")
        .format({
            "C-Index": "{:.4f}",
            "IBS": "{:.4f}",
            "D-Cal p-value": "{:.4f}",
            "MAE (days)": "{:.2f}",
        }),
        width='stretch',
        hide_index=True,
    )

    # Comparison chart
    tab_bar, tab_radar = st.tabs(["Bar Comparison", "🕸️ Radar Chart"])

    with tab_bar:
        fig_comp = make_subplots(
            rows=1, cols=3,
            subplot_titles=["C-Index (↑ better)", "Integrated Brier Score (↓ better)", "MAE in Days (↓ better)"],
            shared_yaxes=False,
        )
        model_colors = {
            "STG-SurviNet": PRIMARY,
            "Cox PH": ACCENT_GREEN,
            "Random Survival Forest": ACCENT_GOLD,
            "Kaplan-Meier": "#6b8cba",
        }

        for i, metric in enumerate(["C-Index", "IBS", "MAE (days)"]):
            for _, row in comparison_data.iterrows():
                fig_comp.add_trace(
                    go.Bar(
                        name=row["Model"],
                        x=[row["Model"]],
                        y=[row[metric]],
                        marker_color=model_colors[row["Model"]],
                        showlegend=(i == 0),
                        text=f"{row[metric]:.4f}" if metric != "MAE (days)" else f"{row[metric]:.2f}",
                        textposition="auto",
                        textfont=dict(size=10),
                    ),
                    row=1, col=i + 1,
                )

        fig_comp.update_layout(
            height=450,
            font=dict(family="Inter", size=11),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.20, xanchor="center", x=0.5),
            margin=dict(t=60, b=80),
        )
        for i in range(1, 4):
            fig_comp.update_xaxes(showticklabels=False, row=1, col=i)
            fig_comp.update_yaxes(row=1, col=i)
        st.plotly_chart(fig_comp, width='stretch')

    with tab_radar:
        # Normalize metrics to [0,1] for radar
        c_vals = comparison_data["C-Index"].values
        ibs_vals = 1 - comparison_data["IBS"].values / comparison_data["IBS"].max()  # Invert: lower is better
        mae_vals = 1 - comparison_data["MAE (days)"].values / comparison_data["MAE (days)"].max()

        categories = ["C-Index", "IBS (inverted)", "MAE (inverted)"]

        fig_radar = go.Figure()
        for idx, row in comparison_data.iterrows():
            values = [c_vals[idx], ibs_vals[idx], mae_vals[idx]]
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=row["Model"],
                line=dict(color=list(model_colors.values())[idx]),
                opacity=0.7,
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
                bgcolor="rgba(0,0,0,0)",
            ),
            font=dict(family="Inter"),
            showlegend=True,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_radar, width='stretch')

    st.info("""
    **Key Findings:**
    - **STG-SurviNet** achieves the highest C-Index (0.7075), outperforming all baselines in risk ranking.
    - **Random Survival Forest** has a low C-Index (0.5273) but achieves the lowest IBS (0.1163).
    - **All models** produce D-Calibration p = 0.0000, indicating no model's probability estimates are well calibrated.
    - STG-SurviNet is best suited as a **risk-ranking and pattern-discovery tool** rather than a calibrated probability predictor.
    """)

    st.divider()

    #  Ablation Study 
    st.markdown("### Ablation Study")
    st.markdown("""
    The ablation study removes one component at a time to measure its contribution.
    """)

    ablation_data = pd.DataFrame({
        "Variant": ["Full STG-SurviNet", "No Temporal", "No Spatial", "No Incident Features"],
        "C-Index": [0.7075, 0.7043, 0.7029, 0.5467],
        "IBS": [0.1209, 0.1223, 0.1229, 0.1474],
        "MAE (days)": [13.68, 13.80, 13.82, 15.14],
        "ΔC-Index": [0.0000, -0.0032, -0.0046, -0.1608],
    })


    st.dataframe(
        ablation_data.style
        .format({
            "C-Index": "{:.4f}", "IBS": "{:.4f}",
            "MAE (days)": "{:.2f}", "ΔC-Index": "{:+.4f}",
        })
        .background_gradient(subset=["ΔC-Index"], cmap="RdYlGn"),
        width='stretch',
        hide_index=True,
    )

    ablation_colors = [PRIMARY, ACCENT_GOLD, "#6b8cba", "#c75b5b"]

    fig_ablation = make_subplots(
        rows=1, cols=3,
        subplot_titles=["C-Index", "IBS", "MAE (days)"],
    )
    for i, metric in enumerate(["C-Index", "IBS", "MAE (days)"]):
        for j, variant in enumerate(ablation_data["Variant"]):
            val = ablation_data.loc[j, metric]
            fig_ablation.add_trace(
                go.Bar(
                    x=[variant], y=[val],
                    marker_color=ablation_colors[j],
                    showlegend=(i == 0),
                    name=variant,
                    text=f"{val:.4f}" if metric != "MAE (days)" else f"{val:.2f}",
                    textposition="auto",
                    textfont=dict(size=9),
                ),
                row=1, col=i + 1,
            )

    fig_ablation.update_layout(
        height=400,
        font=dict(family="Inter", size=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=9)),
        margin=dict(t=50, b=80),
    )
    for i in range(1, 4):
        fig_ablation.update_xaxes(showticklabels=False, row=1, col=i)
        fig_ablation.update_yaxes(row=1, col=i)
    st.plotly_chart(fig_ablation, width='stretch')

    #  Ablation Impact Bar 
    st.markdown("#### Component Contribution (ΔC-Index)")

    components = ["Incident Features", "Spatial (GCN)", "Temporal (TCN)"]
    deltas = [0.1608, 0.0046, 0.0032]
    component_colors = ["#c75b5b", "#6b8cba", ACCENT_GOLD]

    fig_delta = go.Figure()
    fig_delta.add_trace(go.Bar(
        x=components,
        y=deltas,
        marker_color=component_colors,
        text=[f"-{d:.4f}" for d in deltas],
        textposition="outside",
        textfont=dict(size=13),
    ))
    fig_delta.update_layout(
        title="C-Index Drop When Removing Each Component",
        yaxis_title="ΔC-Index (absolute drop)",
        font=dict(family="Inter"),
        title_font=dict(size=16),
        margin=dict(t=60, b=40),
        xaxis=dict(gridcolor="#e0e5d0"),
        yaxis=dict(gridcolor="#e0e5d0"),
        height=380,
    )
    st.plotly_chart(fig_delta, width='stretch')

    st.warning("""
    **Ablation Insight:** Incident-level features are the dominant contributor (ΔC-Index = -0.1608).
    The spatial (GCN) and temporal (TCN) components provide smaller but measurable improvements.
    This suggests the model primarily relies on complaint-specific attributes, with spatial/temporal context
    adding complementary value.
    """)
