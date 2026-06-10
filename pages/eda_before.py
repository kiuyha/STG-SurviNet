import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PRIMARY = "#395886"
ACCENT_GREEN = "#3f6735"
ACCENT_GOLD = "#d4a843"
WHITE = "#ffffff"


def render(df: pd.DataFrame, base_dir: str):
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, {PRIMARY}, #2a4268);
        border-radius: 14px;
        padding: 28px 36px;
        margin-bottom: 28px;
    '>
        <h2 style='color: white !important; margin: 0 0 6px 0; font-size: 1.6rem;'>
            Exploratory Data Analysis — Before Preprocessing
        </h2>
        <p style='color: rgba(255,255,255,0.75); margin: 0; font-size: 0.88rem;'>
            Analyzing the raw dataset to assess data quality and identify issues before cleaning.
            The original dataset contains 647,521 infrastructure complaint records.
        </p>
    </div>
    """, unsafe_allow_html=True)

    #  Missing Values 
    st.markdown("### 🔍 Missing Values Analysis")

    missing_data = pd.DataFrame({
        "Feature": ["closed_date", "community_area", "ward", "latitude", "longitude",
                     "parent_sr_number", "electricity_grid", "electrical_district"],
        "Missing_Count": [9271, 2950, 2792, 1062, 1062, 544249, 3569, 413376],
        "Missing_Percentage": [1.43, 0.45, 0.43, 0.16, 0.16, 83.72, 0.55, 63.59],
    })

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        fig_missing = px.bar(
            missing_data,
            x="Feature",
            y="Missing_Percentage",
            title="Missing Values Percentage Before Preprocessing",
            text_auto=".2f",
            color="Missing_Percentage",
            color_continuous_scale=[[0, "#f7d4b0"], [0.5, "#e07b5a"], [1.0, "#8b1a1a"]],
        )
        fig_missing.update_layout(
            yaxis_title="Missing Percentage (%)",
            xaxis_title="Columns",
            font=dict(family="Inter"),
            title_font=dict(size=16),
            margin=dict(t=60, b=60),
            coloraxis_colorbar=dict(title="% Missing"),
        )
        fig_missing.update_traces(
            textfont_size=11,
            textposition="outside",
            marker_line_width=0,
        )
        st.plotly_chart(fig_missing, width='stretch')

    with col_table:
        st.markdown(f"""
        <div style='
            background: {WHITE};
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 16px rgba(57,88,134,0.08);
            margin-top: 12px;
        '>
            <h4 style='color:{PRIMARY}; margin: 0 0 12px 0;'>📋 Missing Values Summary</h4>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(
            missing_data.style.format({"Missing_Percentage": "{:.2f}%", "Missing_Count": "{:,.0f}"}),
            width='stretch',
            hide_index=True,
        )

        st.info("""
        **Key Observations:**
        - `parent_sr_number` (83.72%) and `electrical_district` (63.59%) have the highest missing rates but are not primary model variables.
        - `closed_date` missing values (1.43%) indicate **still-open complaints** — treated as right-censored observations.
        - Spatial columns (`community_area`, `latitude`, `longitude`) have <0.5% missing.
        """)

    st.divider()

    #  Raw Ticket Status 
    st.markdown("### 🎫 Raw Ticket Status Distribution")

    status_data = pd.DataFrame({
        "Status": ["Completed", "Open", "Cancelled"],
        "Count": [629780, 9855, 9208],
        "Percentage": [97.1, 1.52, 1.42],
    })

    col_donut, col_info = st.columns([2, 2])

    with col_donut:
        fig_status = go.Figure(data=[go.Pie(
            labels=status_data["Status"],
            values=status_data["Count"],
            hole=0.5,
            textinfo="label+percent",
            textfont=dict(size=13, family="Inter"),
            marker=dict(
                colors=[ACCENT_GREEN, ACCENT_GOLD, "#c75b5b"],
                line=dict(color=WHITE, width=3),
            ),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,.0f}<br>Percentage: %{percent}<extra></extra>",
        )])
        fig_status.update_layout(
            title=dict(text="Raw Ticket Status Distribution", font=dict(size=16, family="Inter")),
            font=dict(family="Inter"),
            showlegend=True,
            legend=dict(font=dict(size=12)),
            margin=dict(t=60, b=20),
            annotations=[dict(
                text=f"<b>647,521</b><br><span style='font-size:11px'>Total</span>",
                x=0.5, y=0.5, font_size=18, showarrow=False,
                font=dict(family="Inter"),
            )],
        )
        st.plotly_chart(fig_status, width='stretch')

    with col_info:
        st.markdown(f"""
        <div style='
            background: {WHITE};
            border-radius: 14px;
            padding: 28px;
            box-shadow: 0 4px 20px rgba(57,88,134,0.08);
            margin-top: 20px;
        '>
            <h4 style='color: {PRIMARY}; margin: 0 0 16px 0;'>Interpretation</h4>
            <p style='color: #444; font-size: 0.88rem; line-height: 1.7;'>
                <strong>97.1%</strong> of complaints are marked <strong style='color:{ACCENT_GREEN}'>Completed</strong>,
                confirming that the research question is not <em>whether</em> complaints are resolved,
                but <em>how long</em> they take.
            </p>
            <div style='margin: 16px 0; padding: 12px 16px; background: #f0f4e8; border-radius: 10px; border-left: 4px solid {ACCENT_GREEN};'>
                <p style='font-size: 0.82rem; color: #444; margin: 0;'>
                    <strong style='color:{ACCENT_GREEN};'>Open (1.52%)</strong> — Retained as <em>right-censored</em> observations in survival analysis.
                    These contribute to hazard estimation even without a known completion time.
                </p>
            </div>
            <div style='margin: 12px 0; padding: 12px 16px; background: #fdf2f0; border-radius: 10px; border-left: 4px solid #c75b5b;'>
                <p style='font-size: 0.82rem; color: #444; margin: 0;'>
                    <strong style='color:#c75b5b;'>Cancelled (1.42%)</strong> — Excluded entirely because they don't
                    represent valid resolutions or survival outcomes.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
