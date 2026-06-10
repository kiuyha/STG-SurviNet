import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import plotly.graph_objects as go

PRIMARY = "#395886"
ACCENT_GREEN = "#3f6735"
ACCENT_GOLD = "#d4a843"
WHITE = "#ffffff"

SR_TYPE_OPTIONS = [
    "Pothole in Street Complaint",
    "Street Light Out Complaint",
    "Tree Debris Clean-Up Request",
    "Traffic Signal Out Complaint",
]

COMMUNITY_AREA_NAMES = {
    1: "Rogers Park", 2: "West Ridge", 3: "Uptown", 4: "Lincoln Square", 5: "North Center",
    6: "Lake View", 7: "Lincoln Park", 8: "Near North Side", 9: "Edison Park", 10: "Norwood Park",
    11: "Jefferson Park", 12: "Forest Glen", 13: "North Park", 14: "Albany Park", 15: "Portage Park",
    16: "Irving Park", 17: "Dunning", 18: "Montclare", 19: "Belmont Cragin", 20: "Hermosa",
    21: "Avondale", 22: "Logan Square", 23: "Humboldt Park", 24: "West Town", 25: "Austin",
    26: "West Garfield Park", 27: "East Garfield Park", 28: "Near West Side", 29: "North Lawndale",
    30: "South Lawndale", 31: "Lower West Side", 32: "Loop", 33: "Near South Side",
    34: "Armour Square", 35: "Douglas", 36: "Oakland", 37: "Fuller Park", 38: "Grand Boulevard",
    39: "Kenwood", 40: "Washington Park", 41: "Hyde Park", 42: "Woodlawn", 43: "South Shore",
    44: "Chatham", 45: "Avalon Park", 46: "South Chicago", 47: "Burnside", 48: "Calumet Heights",
    49: "Roseland", 50: "Pullman", 51: "South Deering", 52: "East Side", 53: "West Pullman",
    54: "Riverdale", 55: "Hegewisch", 56: "Garfield Ridge", 57: "Archer Heights", 58: "Brighton Park",
    59: "McKinley Park", 60: "Bridgeport", 61: "New City", 62: "West Elsdon", 63: "Gage Park",
    64: "Clearing", 65: "West Lawn", 66: "Chicago Lawn", 67: "West Englewood",
    68: "Englewood", 69: "Greater Grand Crossing", 70: "Ashburn", 71: "Auburn Gresham",
    72: "Beverly", 73: "Washington Heights", 74: "Mount Greenwood", 75: "Morgan Park",
    76: "O'Hare", 77: "Edgewater",
}

def _load_classical_models(base_dir):
    """Load classical survival models from pickle files."""
    models = {}

    km_path = os.path.join(base_dir, "km_baseline_model.pkl")
    cox_path = os.path.join(base_dir, "cox_survival_model.pkl")
    rsf_path = os.path.join(base_dir, "rsf_survival_model.pkl")

    if os.path.exists(km_path):
        with open(km_path, "rb") as f:
            models["Kaplan-Meier"] = pickle.load(f)
    if os.path.exists(cox_path):
        with open(cox_path, "rb") as f:
            models["Cox PH"] = pickle.load(f)
    if os.path.exists(rsf_path):
        with open(rsf_path, "rb") as f:
            models["RSF"] = pickle.load(f)

    return models


def _build_feature_vector(input_data, df):
    """Build the traditional feature matrix for classical models."""
    sr_types = sorted(SR_TYPE_OPTIONS)
    sr_type_enc = [1.0 if input_data["sr_type"] == cat else 0.0 for cat in sr_types]

    month = input_data["created_month"]
    hour = input_data["created_hour"]
    dow = input_data["created_day_of_week"]

    # Cyclical encoding
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    ward = input_data.get("ward", -1)
    ed = input_data.get("electrical_district", -1)
    eg = 0

    lat = input_data["latitude"]
    lon = input_data["longitude"]

    area = input_data["community_area"]
    area_df = df[df["community_area"] == area]
    backlog = len(area_df) / (len(df["created_date"].unique()) + 1) * 60

    incident_features = sr_type_enc + [
        month_sin, month_cos, hour_sin, hour_cos, dow_sin, dow_cos,
        lat, lon, ward, ed, eg, backlog,
    ]

    area_data = df[df["community_area"] == area]
    total = max(len(area_data), 1)
    type_props = [len(area_data[area_data["sr_type"] == cat]) / total for cat in sr_types]
    lat_range = area_data["latitude"].max() - area_data["latitude"].min() + 1e-6 if len(area_data) > 0 else 1e-6
    lon_range = area_data["longitude"].max() - area_data["longitude"].min() + 1e-6 if len(area_data) > 0 else 1e-6
    density = total / (lat_range * lon_range * 9435)
    mean_dur = area_data.loc[area_data["event"] == 1, "duration_days"].mean() if len(area_data) > 0 else 0.0
    if pd.isna(mean_dur):
        mean_dur = 0.0
    ward_conc = area_data["ward"].value_counts(normalize=True).mean() if len(area_data) > 0 else 0.0
    ed_conc = area_data["electrical_district"].value_counts(normalize=True).mean() if len(area_data) > 0 else 0.0
    grid_conc = area_data["electricity_grid"].value_counts(normalize=True).mean() if len(area_data) > 0 else 0.0
    hour_mean = area_data["created_hour"].mean() if len(area_data) > 0 else 12.0
    dow_mean = area_data["created_day_of_week"].mean() if len(area_data) > 0 else 3.0
    month_mean = area_data["created_month"].mean() if len(area_data) > 0 else 6.0

    spatial_features = type_props + [
        density, mean_dur, ward_conc, ed_conc, grid_conc,
        hour_mean, dow_mean, month_mean,
    ]

    TEMPORAL_WINDOW = 60
    max_date = pd.to_datetime(df["created_date"]).max().normalize()
    start_date = max_date - pd.Timedelta(days=TEMPORAL_WINDOW - 1)
    window_area = df[(pd.to_datetime(df["created_date"]) >= start_date) & (df["community_area"] == area)]
    date_range = pd.date_range(start=start_date, periods=TEMPORAL_WINDOW, freq="D")
    daily = window_area.groupby(pd.to_datetime(window_area["created_date"]).dt.normalize()).size()
    daily = daily.reindex(date_range, fill_value=0).values.astype(float)

    temporal_summary = [
        daily.mean(), daily.std(), daily.max(), daily[-1],
        daily.sum(), daily[-1] - daily[0],
    ]

    features = np.array(incident_features + spatial_features + temporal_summary, dtype=np.float32)
    return features.reshape(1, -1)


def _load_stg_survinet(base_dir):
    """Load the STG-SurviNet model architecture and checkpoint."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv

        class SpatialGCN(nn.Module):
            def __init__(self, in_features, hidden_dim, out_dim):
                super().__init__()
                self.conv1 = GCNConv(in_features, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, out_dim)

            def forward(self, x, edge_index, edge_weight):
                x = self.conv1(x, edge_index, edge_weight)
                x = F.silu(x)
                h_v = self.conv2(x, edge_index, edge_weight)
                return h_v

        class Chomp1d(nn.Module):
            def __init__(self, chomp_size):
                super().__init__()
                self.chomp_size = chomp_size

            def forward(self, x):
                return x[:, :, :-self.chomp_size].contiguous()

        class TemporalTCN(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_dim, kernel_size=3, dilation=2):
                super().__init__()
                padding = (kernel_size - 1) * dilation
                self.network = nn.Sequential(
                    nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=padding, dilation=dilation),
                    Chomp1d(padding),
                    nn.BatchNorm1d(hidden_channels),
                    nn.SiLU(),
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=padding * 2, dilation=dilation * 2),
                    Chomp1d(padding * 2),
                    nn.BatchNorm1d(hidden_channels),
                    nn.SiLU(),
                )
                self.fc = nn.Linear(hidden_channels, out_dim)

            def forward(self, x):
                out = self.network(x)
                h_t = self.fc(out[:, :, -1])
                return h_t

        class STGSurviNet(nn.Module):
            def __init__(self, spatial_in, temporal_in, gcn_out, tcn_out, n_incident_features):
                super().__init__()
                self.spatial_extractor = SpatialGCN(in_features=spatial_in, hidden_dim=64, out_dim=gcn_out)
                self.temporal_extractor = TemporalTCN(in_channels=temporal_in, hidden_channels=32, out_dim=tcn_out)
                self.survival_layer = nn.Sequential(
                    nn.Linear(gcn_out + tcn_out + n_incident_features, 256),
                    nn.BatchNorm1d(256),
                    nn.SiLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.SiLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.SiLU(),
                    nn.Linear(64, 1),
                )

            def forward(self, x_spatial, edge_index, edge_weight, x_temporal, node_indices=None, incident_features=None, ablation_mode=None):
                batch_size = x_spatial.shape[0]
                h_v = torch.zeros(batch_size, self.spatial_extractor.conv2.out_channels, device=x_spatial.device)
                h_t = torch.zeros(batch_size, self.temporal_extractor.fc.out_features, device=x_temporal.device)
                if ablation_mode != "no_spatial":
                    h_v = self.spatial_extractor(x_spatial, edge_index, edge_weight)
                if ablation_mode != "no_temporal":
                    h_t = self.temporal_extractor(x_temporal)
                Z_graph = torch.cat([h_v, h_t], dim=1)
                Z_batch = Z_graph[node_indices]
                if ablation_mode == "no_incident":
                    Z_fused = torch.cat([Z_batch, torch.zeros_like(incident_features)], dim=1)
                else:
                    Z_fused = torch.cat([Z_batch, incident_features], dim=1)
                return self.survival_layer(Z_fused)

        model = STGSurviNet(
            spatial_in=12, temporal_in=1, gcn_out=64, tcn_out=32, n_incident_features=16,
        )
        checkpoint_path = os.path.join(base_dir, "best.pt")
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        else:
            return None
    except ImportError as e:
        st.error(f"Dependency Import Error: {e}")
        return None
    except Exception as e:
        st.error(f"Model Initialization Error: {e}")
        return None


def render(df: pd.DataFrame, geojson: dict, gdf, base_dir: str):
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, {PRIMARY}, #2a4268);
        border-radius: 14px;
        padding: 28px 36px;
        margin-bottom: 28px;
    '>
        <h2 style='color: white !important; margin: 0 0 6px 0; font-size: 1.6rem;'>
            New Data Inference
        </h2>
        <p style='color: rgba(255,255,255,0.75); margin: 0; font-size: 0.88rem;'>
            Enter complaint details manually to parse comparative ranking evidence from all models.
            Default values are pre-filled from dataset medians.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Complaint Details")

    area_options = {f"{v} (#{k})": k for k, v in sorted(COMMUNITY_AREA_NAMES.items(), key=lambda x: x[1])}

    col1, col2, col3 = st.columns(3)

    with col1:
        sr_type = st.selectbox("Complaint Type", SR_TYPE_OPTIONS, index=0)
        community_choice = st.selectbox(
            "Community Area",
            list(area_options.keys()),
            index=list(area_options.keys()).index("Irving Park (#16)"),
        )
        community_area = area_options[community_choice]

    with col2:
        ward = st.number_input("Ward", min_value=1, max_value=50, value=int(df["ward"].median()))
        latitude = st.number_input("Latitude", value=float(df["latitude"].median()), format="%.6f")
        longitude = st.number_input("Longitude", value=float(df["longitude"].median()), format="%.6f")

    with col3:
        created_hour = st.slider("Created Hour", 0, 23, 12)
        created_day_of_week = st.selectbox(
            "Day of Week",
            [(0, "Monday"), (1, "Tuesday"), (2, "Wednesday"), (3, "Thursday"),
             (4, "Friday"), (5, "Saturday"), (6, "Sunday")],
            index=2,
            format_func=lambda x: x[1],
        )
        created_month = st.slider("Created Month", 1, 12, 6)

    col4, col5 = st.columns(2)
    with col4:
        electricity_grid = st.text_input("Electricity Grid", value="F012")
    with col5:
        electrical_district = st.number_input("Electrical District", min_value=0.0, max_value=20.0, value=3.0)

    st.divider()

    if st.button("Run Inference on All Models", width='stretch'):
        input_data = {
            "sr_type": sr_type,
            "community_area": community_area,
            "ward": ward,
            "latitude": latitude,
            "longitude": longitude,
            "created_hour": created_hour,
            "created_day_of_week": created_day_of_week[0],
            "created_month": created_month,
            "electricity_grid": electricity_grid,
            "electrical_district": electrical_district,
        }

        with st.spinner("Running inference across all models…"):
            results = []
            classical_models = _load_classical_models(base_dir)
            feature_vec = _build_feature_vector(input_data, df)

            chart_models = []
            chart_medians = []

            # --- Kaplan-Meier ---
            km_median_val = 4.0
            if "Kaplan-Meier" in classical_models:
                km_bundle = classical_models["Kaplan-Meier"]
                try:
                    km_time = km_bundle["km_time"]
                    km_survival = km_bundle["km_survival"]
                    median_idx = np.searchsorted(-np.array(km_survival), -0.5)
                    km_median_val = float(km_time[min(median_idx, len(km_time) - 1)])
                except Exception:
                    pass

            results.append({
                "Model": "Kaplan-Meier", 
                "Predicted Median (days)": f"{km_median_val:.1f}",
                "Relative Risk (Hazard Index)": "Baseline Profile", 
                "Comparative Ranking Characterization": "Static Global Population Baseline Context", 
                "Architecture Type": "Non-parametric Baseline"
            })
            chart_models.append("Kaplan-Meier")
            chart_medians.append(km_median_val)

            # --- Cox PH ---
            cox_median_val = 0.0
            cox_risk = 0.0
            has_cox = False
            if "Cox PH" in classical_models:
                cox_bundle = classical_models["Cox PH"]
                try:
                    imputer = cox_bundle["imputer"]
                    scaler = cox_bundle["scaler"]
                    cox_model = cox_bundle["model"]

                    feat_imputed = imputer.transform(feature_vec)
                    feat_scaled = scaler.transform(feat_imputed)

                    cox_risk = float(cox_model.predict(feat_scaled)[0])
                    sf = cox_model.predict_survival_function(feat_scaled)
                    
                    if hasattr(sf[0], 'x') and hasattr(sf[0], 'y'):
                        median_idx = np.searchsorted(-sf[0].y, -0.5)
                        cox_median_val = float(sf[0].x[min(median_idx, len(sf[0].x) - 1)])
                        has_cox = True
                    
                    results.append({
                        "Model": "Cox PH", 
                        "Predicted Median (days)": f"{cox_median_val:.1f}",
                        "Relative Risk (Hazard Index)": f"{cox_risk:.4f}", 
                        "Comparative Ranking Characterization": "Linear Proportional Risk Rank Scaling", 
                        "Architecture Type": "Semi-parametric Proportional"
                    })
                except Exception as e:
                    results.append({"Model": "Cox PH", "Predicted Median (days)": "Error", "Relative Risk (Hazard Index)": "Error", "Comparative Ranking Characterization": str(e)[:50], "Architecture Type": "Semi-parametric Proportional"})
            else:
                results.append({"Model": "Cox PH", "Predicted Median (days)": "Model Not Loaded", "Relative Risk (Hazard Index)": "Model Not Loaded", "Comparative Ranking Characterization": "Model representation file missing", "Architecture Type": "Semi-parametric Proportional"})

            if has_cox:
                chart_models.append("Cox PH")
                chart_medians.append(cox_median_val)

            # --- Random Survival Forest ---
            rsf_median_val = 0.0
            rsf_risk = 0.0
            has_rsf = False
            if "RSF" in classical_models:
                rsf_bundle = classical_models["RSF"]
                try:
                    imputer = rsf_bundle["imputer"]
                    rsf_model = rsf_bundle["model"]

                    feat_imputed = imputer.transform(feature_vec)
                    rsf_risk = float(rsf_model.predict(feat_imputed)[0])
                    sf = rsf_model.predict_survival_function(feat_imputed)
                    
                    if hasattr(sf[0], 'x') and hasattr(sf[0], 'y'):
                        median_idx = np.searchsorted(-sf[0].y, -0.5)
                        rsf_median_val = float(sf[0].x[min(median_idx, len(sf[0].x) - 1)])
                        has_rsf = True
                    
                    results.append({
                        "Model": "Random Survival Forest", 
                        "Predicted Median (days)": f"{rsf_median_val:.1f}",
                        "Relative Risk (Hazard Index)": f"{rsf_risk:.4f}", 
                        "Comparative Ranking Characterization": "Ensemble Tree Rank Indicator (Weak Fit, C-Index: 0.4308)", 
                        "Architecture Type": "Non-linear Machine Learning"
                    })
                except Exception as e:
                    results.append({"Model": "Random Survival Forest", "Predicted Median (days)": "Error", "Relative Risk (Hazard Index)": "Error", "Comparative Ranking Characterization": str(e)[:50], "Architecture Type": "Non-linear Machine Learning"})
            else:
                results.append({"Model": "Random Survival Forest", "Predicted Median (days)": "Model Not Loaded", "Relative Risk (Hazard Index)": "Model Not Loaded", "Comparative Ranking Characterization": "Model representation file missing", "Architecture Type": "Non-linear Machine Learning"})

            if has_rsf:
                chart_models.append("Random Survival Forest")
                chart_medians.append(rsf_median_val)

            # --- STG-SurviNet ---
            stg_model = _load_stg_survinet(base_dir)
            stg_median_val = 0.0
            has_stg = False
            if stg_model is not None:
                try:
                    import torch
                    import geopandas as gpd

                    sr_types_sorted = list(df["sr_type"].unique())

                    x_spatial_list = []
                    for area_id in range(1, 78):
                        a_data = df[df["community_area"] == area_id]
                        a_total = max(len(a_data), 1)
                        a_type_props = [len(a_data[a_data["sr_type"] == cat]) / a_total for cat in sr_types_sorted]
                        a_lat_range = a_data["latitude"].max() - a_data["latitude"].min() + 1e-6 if len(a_data) > 0 else 1e-6
                        a_lon_range = a_data["longitude"].max() - a_data["longitude"].min() + 1e-6 if len(a_data) > 0 else 1e-6
                        a_density = a_total / (a_lat_range * a_lon_range * 9435)
                        a_mean_dur = a_data.loc[a_data["event"] == 1, "duration_days"].mean() if len(a_data) > 0 else 0.0
                        if pd.isna(a_mean_dur):
                            a_mean_dur = 0.0
                        a_ward_conc = a_data["ward"].value_counts(normalize=True).mean() if len(a_data) > 0 else 0.0
                        a_ed_conc = a_data["electrical_district"].value_counts(normalize=True).mean() if len(a_data) > 0 else 0.0
                        a_grid_conc = a_data["electricity_grid"].value_counts(normalize=True).mean() if len(a_data) > 0 else 0.0
                        a_hour_mean = a_data["created_hour"].mean() if len(a_data) > 0 else 12.0
                        a_dow_mean = a_data["created_day_of_week"].mean() if len(a_data) > 0 else 3.0
                        a_month_mean = a_data["created_month"].mean() if len(a_data) > 0 else 6.0
                        x_spatial_list.append(a_type_props + [a_density, a_mean_dur, a_ward_conc, a_ed_conc, a_grid_conc, a_hour_mean, a_dow_mean, a_month_mean])

                    x_spatial = torch.tensor(np.array(x_spatial_list, dtype=np.float32))
                    x_mean = x_spatial.mean(dim=0)
                    x_std = x_spatial.std(dim=0) + 1e-8
                    x_spatial = (x_spatial - x_mean) / x_std

                    TEMPORAL_WINDOW = 60
                    max_date = pd.to_datetime(df["created_date"]).max().normalize()
                    start_date = max_date - pd.Timedelta(days=TEMPORAL_WINDOW - 1)
                    date_range = pd.date_range(start=start_date, periods=TEMPORAL_WINDOW, freq="D")
                    window_df = df[df["created_date"].astype('datetime64[ns]') >= start_date].copy()
                    window_df["date"] = pd.to_datetime(window_df["created_date"]).dt.normalize()

                    x_temporal = np.zeros((77, 1, TEMPORAL_WINDOW), dtype=np.float32)
                    for area_id in range(1, 78):
                        g = window_df[window_df["community_area"] == area_id]
                        counts = g.groupby("date").size().reindex(date_range, fill_value=0)
                        x_temporal[area_id - 1, 0, :] = counts.values.astype(np.float32)

                    x_temporal_t = torch.tensor(x_temporal)
                    t_mean = x_temporal_t.mean()
                    t_std = x_temporal_t.std() + 1e-8
                    x_temporal_t = (x_temporal_t - t_mean) / t_std

                    gdf_loaded = gpd.read_file(os.path.join(base_dir, "Boundaries_-_Community_Areas_20260523.geojson"))
                    gdf_loaded["area_numbe"] = gdf_loaded["area_numbe"].astype(int)
                    gdf_loaded = gdf_loaded.sort_values("area_numbe").reset_index(drop=True)
                    gdf_proj = gdf_loaded.to_crs(epsg=3435)
                    gdf_proj["centroid"] = gdf_proj.geometry.centroid

                    n_nodes = len(gdf_proj)
                    A = np.zeros((n_nodes, n_nodes))
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            if i != j and gdf_proj.geometry.iloc[i].touches(gdf_proj.geometry.iloc[j]):
                                dist = gdf_proj["centroid"].iloc[i].distance(gdf_proj["centroid"].iloc[j])
                                A[i, j] = 10000.0 / dist

                    edges = np.where(A > 0)
                    edge_index = torch.tensor(np.array([edges[0], edges[1]]), dtype=torch.long)
                    edge_weight = torch.tensor(A[edges], dtype=torch.float32)

                    bg_sr_type_feats = np.stack([(df["sr_type"] == cat).astype(np.float32) for cat in sr_types_sorted], axis=1)
                    bg_ward = df["ward"].fillna(-1).values.astype(np.float32)
                    bg_ed = df["electrical_district"].fillna(-1).values.astype(np.float32)
                    bg_month = df["created_month"].values.astype(np.float32)
                    bg_hour = df["created_hour"].values.astype(np.float32)
                    bg_dow = df["created_day_of_week"].values.astype(np.float32)
                    bg_lat = df["latitude"].values.astype(np.float32)
                    bg_lon = df["longitude"].values.astype(np.float32)
                    
                    grid_col = df["electricity_grid"].astype("category")
                    bg_grid_enc = grid_col.cat.codes.values.astype(np.float32)
                    
                    try:
                        grid_idx = list(grid_col.cat.categories).index(electricity_grid)
                        input_grid_enc = float(grid_idx)
                    except ValueError:
                        input_grid_enc = -1.0

                    bg_created = pd.to_datetime(df["created_date"])
                    bg_backlog = np.zeros(len(df), dtype=np.float32)
                    temp_df = pd.DataFrame({'date': bg_created, 'idx': np.arange(len(df)), 'area': df["community_area"]})
                    for area in df["community_area"].unique():
                        mask = temp_df["area"] == area
                        area_df = temp_df[mask].sort_values('date').set_index('date')
                        bg_backlog[area_df['idx'].values] = (area_df['idx'].rolling('60d').count().values - 1).astype(np.float32)

                    bg_individual = np.stack([
                        np.sin(2 * np.pi * bg_month / 12), np.cos(2 * np.pi * bg_month / 12),
                        np.sin(2 * np.pi * bg_hour / 24), np.cos(2 * np.pi * bg_hour / 24),
                        np.sin(2 * np.pi * bg_dow / 7), np.cos(2 * np.pi * bg_dow / 7),
                        bg_lat, bg_lon, bg_ward, bg_ed, bg_grid_enc, bg_backlog
                    ], axis=1)
                    bg_inc_matrix = np.concatenate([bg_sr_type_feats, bg_individual], axis=1)
                    
                    inc_mean = bg_inc_matrix.mean(axis=0)
                    inc_std = bg_inc_matrix.std(axis=0) + 1e-8

                    input_area_df = temp_df[(temp_df["area"] == community_area) & (temp_df["date"] >= (max_date - pd.Timedelta(days=60)))]
                    input_backlog = float(len(input_area_df))

                    month = input_data["created_month"]
                    hour = input_data["created_hour"]
                    dow = input_data["created_day_of_week"]

                    input_sr_type_feats = [1.0 if sr_type == cat else 0.0 for cat in sr_types_sorted]
                    input_individual = [
                        np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
                        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                        float(latitude), float(longitude), float(ward), float(electrical_district),
                        input_grid_enc, input_backlog
                    ]
                    incident_vec = input_sr_type_feats + input_individual
                    incident_matrix_input = (np.array([incident_vec], dtype=np.float32) - inc_mean) / inc_std

                    incident_t = torch.tensor(incident_matrix_input, dtype=torch.float32)
                    node_idx = torch.tensor([community_area - 1], dtype=torch.long)

                    with torch.no_grad():
                        log_hazard = stg_model(
                            x_spatial, edge_index, edge_weight, x_temporal_t,
                            node_indices=node_idx, incident_features=incident_t,
                        ).squeeze().item()

                    risk_score = np.exp(log_hazard)
                    stg_median_val = max(1, int(13.68 * risk_score / np.exp(0)))
                    has_stg = True

                    results.append({
                        "Model": "STG-SurviNet",
                        "Predicted Median (days)": f"~{stg_median_val:.0f}",
                        "Relative Risk (Hazard Index)": f"{risk_score:.4f} (log-hazard: {log_hazard:+.4f})",
                        "Comparative Ranking Characterization": "Optimal Spatiotemporal Risk Rank Alignment (Best Discriminator)",
                        "Architecture Type": "Deep Learning Graph Hybrid",
                    })
                except Exception as e:
                    results.append({"Model": "STG-SurviNet", "Predicted Median (days)": "Error", "Relative Risk (Hazard Index)": "Error", "Comparative Ranking Characterization": f"Error: {str(e)[:50]}", "Architecture Type": "Deep Learning Graph Hybrid"})
            else:
                results.append({"Model": "STG-SurviNet", "Predicted Median (days)": "Model Not Loaded", "Relative Risk (Hazard Index)": "Model Not Loaded", "Comparative Ranking Characterization": "Model representation file missing", "Architecture Type": "Deep Learning Graph Hybrid"})

            if has_stg:
                chart_models.append("STG-SurviNet")
                chart_medians.append(stg_median_val)

        st.markdown("### Inference Results")

        results_df = pd.DataFrame(results)
        st.dataframe(results_df, width='stretch', hide_index=True)

        #  Risk and Ranking Visualization Section 
        st.markdown("### Visual Risk Ranking Comparison")
        
        if chart_models:
            fig_rank = go.Figure()
            fig_rank.add_trace(go.Bar(
                x=chart_models,
                y=chart_medians,
                marker_color=[PRIMARY, ACCENT_GREEN, ACCENT_GOLD, "#c75b5b"][:len(chart_models)],
                text=[f"{v:.1f} days" for v in chart_medians],
                textposition="outside",
                textfont=dict(size=12, family="Inter"),
                hovertemplate="<b>Model: %{x}</b><br>Predicted Median Timeline: %{y:.1f} days<extra></extra>"
            ))
            fig_rank.update_layout(
                title=dict(
                    text="Comparative Timeline Ranking Across Models (Higher Days = Slower Resolution Risk)",
                    font=dict(size=14, family="Inter")
                ),
                xaxis_title="Survival Estimation Architecture",
                yaxis_title="Predicted Resolution Median (Days)",
                font=dict(family="Inter"),
                margin=dict(t=50, b=40, l=20, r=20),
                height=380,
                xaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
                yaxis=dict(gridcolor="rgba(0,0,0,0.05)")
            )
            st.plotly_chart(fig_rank, width='stretch')
        else:
            st.warning("Visual ranking graph could not be populated due to insufficient loaded parameters.")

        st.markdown(f"""
        <div style='
            background: var(--background-color-secondary, #e8ebd4);
            color: var(--text-color, #1a1a1a);
            border-radius: 12px;
            padding: 22px;
            font-size: 0.86rem;
            line-height: 1.65;
            margin-top: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.03);
            border-left: 5px solid {PRIMARY};
        '>
            <strong>Critical Calibration & Causation Framework Notice:</strong><br>
            While the analysis demonstrates the potential of STG-SurviNet as a ranking and pattern-discovery tool, the model still has several limitations. Its test C-index of 0.7075 remains below the target of 0.75, while its MAE of 13.68 days remains above the target of 7 days. All evaluated models also produce a D-Calibration p-value of 0.0000, indicating that their predicted survival probability distributions are not well calibrated. Although STG-SurviNet achieves the strongest C-index, Integrated Brier Score, and MAE, its probability estimates should not be interpreted as reliable absolute predictions. Use the days timeline metrics above strictly as comparative ranking boundaries rather than absolute static durations.<br><br>
            Furthermore, the identified spatial, temporal, and delay-risk patterns represent associations learned from the Chicago 311 data and do not establish causal relationships. Therefore, the findings are interpreted as comparative ranking and model-based pattern evidence rather than causal conclusions.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Input Summary")
        input_summary = pd.DataFrame({
            "Field": ["Complaint Type", "Community Area", "Ward", "Latitude", "Longitude",
                       "Created Hour", "Day of Week", "Month", "Electricity Grid", "Electrical District"],
            "Value": [
                str(sr_type), 
                f"{COMMUNITY_AREA_NAMES[community_area]} (#{community_area})",
                str(ward), 
                f"{latitude:.6f}", 
                f"{longitude:.6f}",
                str(created_hour), 
                str(created_day_of_week[1]), 
                str(created_month),
                str(electricity_grid), 
                str(electrical_district)
            ],
        })
        st.dataframe(input_summary, width='stretch', hide_index=True)