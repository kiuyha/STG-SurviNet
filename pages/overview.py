import streamlit as st

#  Color constants 
PRIMARY = "#395886"
ACCENT_GREEN = "#3f6735"
ACCENT_GOLD = "#d4a843"
BG = "#f2f7e1"
CARD_BG = "#fafcf4"
CARD_BORDER = "#d7dec2"
TEXT_MUTED = "#4f5b4f"
LIGHT_GRAY = "#e8ebd4"

PLOTLY_COLORS = [PRIMARY, ACCENT_GREEN, ACCENT_GOLD, "#6b8cba", "#7db36e", "#e8c56d"]


def render(base_dir: str):
    #  Header 
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, {PRIMARY}, #2a4268);
        border-radius: 16px;
        padding: 40px 48px;
        margin-bottom: 32px;
        box-shadow: 0 8px 32px rgba(57, 88, 134, 0.25);
    '>
        <h1 style='color: #fff !important; font-size: 2.2rem; margin: 0 0 8px 0; letter-spacing: -1px;'>
            STG-SurviNet
        </h1>
        <p style='color: rgba(255,255,255,0.9); font-size: 1rem; margin: 0 0 4px 0; line-height: 1.6;'>
            Mining Spatial, Temporal, and Survival Patterns in Urban Infrastructure
            Resolution Times Across Chicago Community Areas
        </p>
        <p style='color: rgba(255,255,255,0.6); font-size: 0.8rem; margin: 16px 0 0 0;'>
            Syafira Najema P.A. · Ketut Shridhara · Naura Kanaya P.M. &nbsp;|&nbsp;
            Advisor: Moh. Khoridatul Huda, S.Pd., M.Si., Ph.D. &nbsp;|&nbsp; Data Science, UNESA 2024
        </p>
    </div>
    """, unsafe_allow_html=True)

    #  Key Metrics 
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", "541,625", help="After preprocessing")
    with col2:
        st.metric("Community Areas", "77", help="Chicago administrative areas")
    with col3:
        st.metric("Complaint Types", "4", help="Pothole, Street Light, Traffic Signal, Tree Debris")
    with col4:
        st.metric("Best C-Index", "0.7075", help="STG-SurviNet test performance")

    st.markdown("<br>", unsafe_allow_html=True)

    #  Architecture 
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown(f"### Architecture")
        st.markdown(f"""
        <div style='
            background: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 14px;
            padding: 28px;
            box-shadow: 0 4px 20px rgba(57,88,134,0.08);
        '>
            <p style='color: {PRIMARY}; font-size: 0.92rem; line-height: 1.8;'>
                <strong>STG-SurviNet</strong> is a hybrid deep learning architecture that integrates three components
                into a single end-to-end model for survival analysis:
            </p>
            <div style='margin: 16px 0;'>
                <div style='display:flex; align-items:center; margin: 12px 0;'>
                    <span style='background:{PRIMARY}; color:white; border-radius:8px; padding:4px 12px; font-size:0.8rem; font-weight:600; min-width:130px; text-align:center;'>
                        GCN (Spatial)
                    </span>
                    <span style='color:{TEXT_MUTED};; font-size:0.85rem; margin-left:14px;'>
                        Graph Convolutional Network — learns spatial relationships between 77 community areas via adjacency graph (394 edges)
                    </span>
                </div>
                <div style='display:flex; align-items:center; margin: 12px 0;'>
                    <span style='background:{ACCENT_GREEN}; color:white; border-radius:8px; padding:4px 12px; font-size:0.8rem; font-weight:600; min-width:130px; text-align:center;'>
                        TCN (Temporal)
                    </span>
                    <span style='color:{TEXT_MUTED};; font-size:0.85rem; margin-left:14px;'>
                        Temporal Convolutional Network — captures 60-day complaint workload patterns per area
                    </span>
                </div>
                <div style='display:flex; align-items:center; margin: 12px 0;'>
                    <span style='background:{ACCENT_GOLD}; color:white; border-radius:8px; padding:4px 12px; font-size:0.8rem; font-weight:600; min-width:130px; text-align:center;'>
                        Deep Cox PH
                    </span>
                    <span style='color:{TEXT_MUTED};; font-size:0.85rem; margin-left:14px;'>
                        Deep Cox Proportional Hazards — produces individual survival curves and handles right-censored data
                    </span>
                </div>
            </div>
            <p style='color:{TEXT_MUTED};; font-size: 0.82rem; margin-top: 16px; border-top: 1px solid #eee; padding-top: 12px;'>
                The spatial + temporal embeddings are concatenated with incident-level features (16 dims),
                then passed through a 3-layer survival head (256 → 128 → 64 → 1) to produce the log partial hazard.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("### Dataset Overview")

        import streamlit.components.v1 as components
        components.html(f"""
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
        <div style='
            font-family: Inter, sans-serif;
            background:{CARD_BG};
            border:1px solid {CARD_BORDER};
            border-radius:14px;
            padding:28px;
            box-shadow:0 4px 20px rgba(57,88,134,0.06);
        '>
            <div style='color:{PRIMARY}; font-size:0.95rem; font-weight:700; margin-bottom:14px;'>
                Dataset Statistics
            </div>

            <table style='width:100%; border-collapse:collapse; font-size:0.85rem;'>
                <tr style='border-bottom:1px solid #dde3cf;'>
                    <td style='padding:8px 4px; color:{PRIMARY}; font-weight:600;'>Source</td>
                    <td style='padding:8px 4px;'>Chicago 311 Open Data Portal</td>
                </tr>
                <tr style='border-bottom:1px solid #dde3cf;'>
                    <td style='padding:8px 4px; color:{PRIMARY}; font-weight:600;'>Period</td>
                    <td style='padding:8px 4px;'>2022-2026</td>
                </tr>
                <tr style='border-bottom:1px solid #dde3cf;'>
                    <td style='padding:8px 4px; color:{PRIMARY}; font-weight:600;'>Raw Records</td>
                    <td style='padding:8px 4px;'>650,094</td>
                </tr>
                <tr style='border-bottom:1px solid #dde3cf;'>
                    <td style='padding:8px 4px; color:{PRIMARY}; font-weight:600;'>Clean Records</td>
                    <td style='padding:8px 4px;'>541,625</td>
                </tr>
                <tr style='border-bottom:1px solid #dde3cf;'>
                    <td style='padding:8px 4px; color:{PRIMARY}; font-weight:600;'>Completion Rate</td>
                    <td style='padding:8px 4px;'>97.1%</td>
                </tr>
                <tr>
                    <td style='padding:8px 4px; color:{PRIMARY}; font-weight:600;'>Median Duration</td>
                    <td style='padding:8px 4px;'>4 days</td>
                </tr>
            </table>

            <div style='margin-top:24px; padding-top:18px; border-top:1px solid #dde3cf;'>
                <div style='color:{PRIMARY}; font-size:0.95rem; font-weight:700; margin-bottom:16px;'>
                    Data Split
                </div>

                <div style='margin-bottom:12px;'>
                    <div style='display:flex; justify-content:space-between; font-size:0.82rem;'>
                        <span>Training</span><span><b>60%</b></span>
                    </div>
                    <div style='height:8px; background:#e5ead8; border-radius:999px;'>
                        <div style='width:60%; height:100%; background:{PRIMARY}; border-radius:999px;'></div>
                    </div>
                </div>

                <div style='margin-bottom:12px;'>
                    <div style='display:flex; justify-content:space-between; font-size:0.82rem;'>
                        <span>Validation</span><span><b>20%</b></span>
                    </div>
                    <div style='height:8px; background:#e5ead8; border-radius:999px;'>
                        <div style='width:20%; height:100%; background:{ACCENT_GREEN}; border-radius:999px;'></div>
                    </div>
                </div>

                <div style='margin-bottom:12px;'>
                    <div style='display:flex; justify-content:space-between; font-size:0.82rem;'>
                        <span>Calibration</span><span><b>10%</b></span>
                    </div>
                    <div style='height:8px; background:#e5ead8; border-radius:999px;'>
                        <div style='width:10%; height:100%; background:{ACCENT_GOLD}; border-radius:999px;'></div>
                    </div>
                </div>

                <div>
                    <div style='display:flex; justify-content:space-between; font-size:0.82rem;'>
                        <span>Test</span><span><b>10%</b></span>
                    </div>
                    <div style='height:8px; background:#e5ead8; border-radius:999px;'>
                        <div style='width:10%; height:100%; background:#6b8cba; border-radius:999px;'></div>
                    </div>
                </div>
            </div>
        </div>
        """, height=520)

    st.markdown("<br>", unsafe_allow_html=True)

    #  Complaint Types 
    st.markdown("### Infrastructure Complaint Categories")
    
    cat_data = {
        "Category": [
            "Pothole in Street Complaint", 
            "Street Light Out Complaint",
            "Tree Debris Clean-Up Request", 
            "Traffic Signal Out Complaint"
        ],
        "Records": ["317,589", "222,722", "60,441", "49,342"],
        "Icon": ["report_problem", "lightbulb", "nature_people", "traffic"],
    }

    cols = st.columns(4)
    for i, col in enumerate(cols):
        with col:
            color_theme = [PRIMARY, ACCENT_GREEN, ACCENT_GOLD, "#6b8cba"][i]
            st.markdown(f"""
            <div style='
                background: var(--background-color-secondary, #ffffff);
                color: var(--text-color, #1a1a1a);
                border-radius: 14px;
                padding: 24px;
                text-align: center;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
                border-top: 4px solid {color_theme};
                height: 170px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                margin-bottom: 20px;
            '>
                <span class="material-icons" style="font-size: 2.4rem; color: {color_theme}; margin-bottom: 12px;">
                    {cat_data["Icon"][i]}
                </span>
                <div style='font-size: 0.82rem; font-weight: 600; line-height: 1.3; min-height: 38px; display: flex; align-items: center; justify-content: center;'>
                    {cat_data["Category"][i]}
                </div>
                <div style='font-size: 1.4rem; color: var(--text-color, #1a1a1a); font-weight: 800; margin-top: 8px;'>
                    {cat_data["Records"][i]}
                </div>
            </div>
            """, unsafe_allow_html=True)
