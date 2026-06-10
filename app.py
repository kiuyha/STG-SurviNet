import streamlit as st
import pandas as pd
import numpy as np
import json
import os

from pages import overview, eda_before, eda_after, gnn_spatial, model_evaluation, delay_risk, inference

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="STG-SurviNet Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS and Font Loading
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
 
    /* Hide sidebar collapse button */
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
            
    .stApp h1 {
        color: inherit !important;
    }

            
    /* Base font */
    .stApp, .main .block-container {
        font-family: 'Inter', sans-serif !important;
    }
 
    /* Hide the auto-generated Streamlit multipage file list */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
 
    /* Sidebar styling adjustment */
    section[data-testid="stSidebar"] * {
        font-family: 'Inter', sans-serif !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem !important;
        padding: 6px 12px !important;
        border-radius: 8px !important;
        transition: background-color 0.2s ease;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(128, 128, 128, 0.2) !important;
    }
 
    /* Theme-aware Metric cards */
    div[data-testid="stMetric"] {
        background:    var(--background-color-secondary, #ffffff);
        border-radius: 12px;
        padding:       16px 20px;
        box-shadow:    0 4px 20px rgba(0, 0, 0, 0.05);
        border-left:   4px solid #3f6735;
    }
    div[data-testid="stMetric"] label {
        font-weight: 600 !important;
        color: var(--text-color, #1a1a1a) !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color:       #3f6735 !important;
        font-weight: 700 !important;
    }
 
    /* Dataframe wrapper */
    .stDataFrame {
        border-radius: 12px;
        overflow:      hidden;
        box-shadow:    0 4px 20px rgba(0, 0, 0, 0.05);
    }
 
    /* Tabs adjustment */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: var(--background-color-secondary, #e8ebd4) !important;
        border-radius:    8px 8px 0 0 !important;
        font-weight:      500 !important;
        color: var(--text-color, #1a1a1a) !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #395886 !important;
        color:            #ffffff !important;
    }
 
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: var(--background-color-secondary, #e8ebd4) !important;
        border-radius:    10px !important;
        font-weight:      600 !important;
    }
 
    /* Interactive Buttons */
    .stButton > button {
        border:        none !important;
        border-radius: 10px !important;
        font-weight:   600 !important;
        padding:       0.6rem 2rem !important;
        transition:    all 0.3s ease !important;
        box-shadow:    0 2px 8px rgba(57, 88, 134, 0.2) !important;
    }
    .stButton > button:hover {
        transform:  translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(57, 88, 134, 0.3) !important;
    }
 
    /* Interfaces */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

#  Data loading (cached) 
@st.cache_data(show_spinner="Loading dataset…")
def load_clean_data():
    return pd.read_csv(os.path.join(BASE_DIR, "clean_data.csv"), parse_dates=["created_date", "closed_date"])


@st.cache_data(show_spinner="Loading GeoJSON…")
def load_geojson():
    with open(os.path.join(BASE_DIR, "Boundaries_-_Community_Areas_20260523.geojson"), "r") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading GeoJSON for geopandas…")
def load_gdf():
    import geopandas as gpd
    gdf = gpd.read_file(os.path.join(BASE_DIR, "Boundaries_-_Community_Areas_20260523.geojson"))
    gdf["area_numbe"] = gdf["area_numbe"].astype(int)
    gdf = gdf.sort_values("area_numbe").reset_index(drop=True)
    return gdf


#  Sidebar 
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <h2 style='margin:0; font-size:1.4rem; letter-spacing:-0.5px;'>STG-SurviNet</h2>
        <p style='margin:4px 0 0 0; font-size:0.75rem; opacity:0.8;'>Chicago 311 Survival Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "EDA (Before Preprocessing)",
            "EDA (After Preprocessing)",
            "GNN & Spatial Analysis",
            "Model Evaluation",
            "Delay Risk Analysis",
            "New Data Inference",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("""
    <div style='font-size: 0.7rem; opacity: 0.6; text-align: center; padding-top: 10px;'>
        <p>Syafira N. P. Anisa</p>
        <p>Ketut Shridhara</p>
        <p>Naura K. P. Masruri</p>
        <p style='margin-top:8px;'>INT24 - Data Science - UNESA</p>
    </div>
    """, unsafe_allow_html=True)


#  Page routing 
if page == "Overview":
    overview.render(BASE_DIR)

elif page == "EDA (Before Preprocessing)":
    df = load_clean_data()
    eda_before.render(df, BASE_DIR)

elif page == "EDA (After Preprocessing)":
    df = load_clean_data()
    geojson = load_geojson()
    eda_after.render(df, geojson, BASE_DIR)

elif page == "GNN & Spatial Analysis":
    df = load_clean_data()
    geojson = load_geojson()
    gdf = load_gdf()
    gnn_spatial.render(df, geojson, gdf, BASE_DIR)

elif page == "Model Evaluation":
    model_evaluation.render(BASE_DIR)

elif page == "Delay Risk Analysis":
    df = load_clean_data()
    geojson = load_geojson()
    gdf = load_gdf()
    delay_risk.render(df, geojson, gdf, BASE_DIR)

elif page == "New Data Inference":
    df = load_clean_data()
    geojson = load_geojson()
    gdf = load_gdf()
    inference.render(df, geojson, gdf, BASE_DIR)
