import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PRIMARY = "#395886"
ACCENT_GREEN = "#3f6735"
ACCENT_GOLD = "#d4a843"
WHITE = "#ffffff"
COLORS = [PRIMARY, ACCENT_GREEN, ACCENT_GOLD, "#6b8cba"]
CATEGORY_COLORS = {
    "Pothole in Street Complaint": PRIMARY,
    "Street Light Out Complaint": ACCENT_GREEN,
    "Traffic Signal Out Complaint": ACCENT_GOLD,
    "Tree Debris Clean-Up Request": "#6b8cba",
}


def render(df: pd.DataFrame, geojson: dict, base_dir: str):
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, {PRIMARY}, #2a4268);
        border-radius: 14px;
        padding: 28px 36px;
        margin-bottom: 28px;
    '>
        <h2 style='color: white !important; margin: 0 0 6px 0; font-size: 1.6rem;'>
            Exploratory Data Analysis — After Preprocessing
        </h2>
        <p style='color: rgba(255,255,255,0.75); margin: 0; font-size: 0.88rem;'>
            After cleaning, the dataset contains <strong>541,625 valid records</strong>.
            Resolution duration is heavily right-skewed with a mean of ~22 days vs median of 4 days.
        </p>
    </div>
    """, unsafe_allow_html=True)

    #  Preprocessing Pipeline 
    with st.expander("Preprocessing Pipeline Steps", expanded=False):
        pipeline_data = pd.DataFrame({
            "Step": [
                "Drop missing spatial IDs",
                "Filter duplicates",
                "Convert timestamps",
                "Compute duration_days",
                "Fill missing closed_date",
                "Assign event indicator",
                "Remove negative durations",
            ],
            "Action": [
                "Drop rows missing community_area, latitude, or longitude",
                "Filter duplicate == False",
                "Convert created_date & closed_date to datetime",
                "end_date - created_date in calendar days",
                "Use max(created_date) as right-censoring boundary",
                "event = 1 if Completed, 0 otherwise",
                "Remove rows where duration_days < 0",
            ],
            "Purpose": [
                "Required for graph node assignment",
                "Prevent workload distortion",
                "Enable time arithmetic",
                "Primary survival response variable",
                "Right-censoring for open complaints",
                "Event indicator for Cox model",
                "Eliminate data entry errors",
            ],
        })
        st.dataframe(pipeline_data, width='stretch', hide_index=True)

    st.divider()

    #  Descriptive Statistics 
    st.markdown("### Descriptive Statistics by Complaint Category")

    stats = df.groupby("sr_type")["duration_days"].describe()
    stats = stats.rename(columns={
        "count": "Count", "mean": "Mean", "std": "Std",
        "min": "Min", "25%": "Q1", "50%": "Median", "75%": "Q3", "max": "Max"
    })

    st.dataframe(
        stats.style.format({
            "Count": "{:,.0f}", "Mean": "{:.2f}", "Std": "{:.2f}",
            "Min": "{:.0f}", "Q1": "{:.0f}", "Median": "{:.0f}",
            "Q3": "{:.0f}", "Max": "{:,.0f}",
        }),
        width='stretch',
    )

    st.divider()

    #  Boxplot and Density 
    st.markdown("### Resolution Duration Distribution")

    tab_box, tab_density = st.tabs(["📦 Boxplot", "📉 Density Distribution"])

    with tab_box:
        fig_box = px.box(
            df,
            x="duration_days",
            color="sr_type",
            title="Boxplot of Resolution Duration by Complaint Category",
            color_discrete_map=CATEGORY_COLORS,
            labels={"duration_days": "Duration (Days)", "sr_type": "Category"},
        )
        fig_box.update_layout(
            font=dict(family="Inter"),
            title_font=dict(size=16),
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.25,
                xanchor="center", x=0.5, title_text="",
                font=dict(size=11),
            ),
            margin=dict(t=60, b=80),
            xaxis=dict(gridcolor="#e0e5d0"),
        )
        st.plotly_chart(fig_box, width='stretch')

        st.info("""
        **Key Insight:** All categories show extreme right-skew with many outliers.
        Pothole complaints have the widest spread (max 1,486 days).
        Traffic Signal Out has a median of 0 days — many are resolved same-day.
        """)

    with tab_density:
        fig_density = go.Figure()
        for cat, color in CATEGORY_COLORS.items():
            subset = df[df["sr_type"] == cat]["duration_days"]
            subset = subset[subset <= 60]  # Cap at 60 days
            fig_density.add_trace(go.Histogram(
                x=subset,
                name=cat.replace(" Complaint", "").replace(" Request", ""),
                marker_color=color,
                opacity=0.6,
                nbinsx=60,
                histnorm="probability density",
            ))

        fig_density.update_layout(
            title="Density Distribution of Resolution Duration (Capped at 60 Days)",
            barmode="overlay",
            font=dict(family="Inter"),
            title_font=dict(size=16),
            xaxis_title="Duration (Days)",
            yaxis_title="Density",
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.25,
                xanchor="center", x=0.5,
                font=dict(size=11),
            ),
            margin=dict(t=60, b=80),
            xaxis=dict(range=[0, 60]),
            yaxis=dict(gridcolor="#e0e5d0"),
        )
        st.plotly_chart(fig_density, width='stretch')

        st.info("""
        **Key Insight:** The distribution is heavily right-skewed — most complaints are resolved within
        the first two weeks, but a long tail extends to 60+ days. This pattern supports survival analysis,
        which is specifically designed for right-skewed time-to-event data with censored observations.
        """)

    st.divider()

    #  Spatial Distribution Map 
    st.markdown("### Spatial Distribution of Complaints")

    df_sample = df.sample(n=min(5000, len(df)), random_state=42)

    fig_map = px.scatter_map(
        df_sample,
        lat="latitude",
        lon="longitude",
        color="sr_type",
        title="Infrastructure Complaints Across Chicago (5K Sample)",
        zoom=9.5,
        opacity=0.6,
        color_discrete_map=CATEGORY_COLORS,
        labels={"sr_type": "Complaint Type"},
    )
    fig_map.update_layout(
        map_center={"lat": 41.8372, "lon": -87.6860},
        map_zoom=9.72,
        margin={"r": 10, "t": 60, "l": 10, "b": 10},
        height=650,
        font=dict(family="Inter"),
        title_font=dict(size=16),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.05,
            xanchor="center", x=0.5, title_text="",
            font=dict(size=11),
        ),
        map_layers=[{
            "source": geojson,
            "type": "line",
            "color": PRIMARY,
            "line": {"width": 1.2},
        }],
    )
    st.plotly_chart(fig_map, width='stretch')

    st.success("""
    **Spatial Structure Confirmed:** The map shows clear geographic clustering of complaints across
    Chicago's 77 community areas. This validates the use of a graph-based spatial component (GCN)
    to capture how overload in one area influences its neighbors.
    """)
