import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import streamlit.components.v1 as components

PRIMARY = "#395886"
ACCENT_GREEN = "#3f6735"
ACCENT_GOLD = "#d4a843"
WHITE = "#ffffff"


def _build_adjacency(gdf):
    """Build adjacency from GeoJSON and return edge list with centroid coords."""
    gdf_proj = gdf.to_crs(epsg=3857)
    centroids_proj = gdf_proj.geometry.centroid  # EPSG:3857 centroids

    # Reproject centroids to WGS84 for plotting
    centroids_wgs = centroids_proj.to_crs(epsg=4326)
    centroid_lat = centroids_wgs.y
    centroid_lon = centroids_wgs.x

    n_nodes = len(gdf_proj)
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if gdf_proj.geometry.iloc[i].touches(gdf_proj.geometry.iloc[j]):
                dist = centroids_proj.iloc[i].distance(centroids_proj.iloc[j])
                weight = 10000.0 / dist
                edges.append((i, j, weight))

    return edges, centroid_lat.values, centroid_lon.values


@st.cache_data(show_spinner="Building spatial adjacency graph…")
def build_graph_data(_gdf):
    edges, lats, lons = _build_adjacency(_gdf)
    names = _gdf["community"].values
    area_numbers = _gdf["area_numbe"].values
    return edges, lats, lons, names, area_numbers


def render(df: pd.DataFrame, geojson: dict, gdf, base_dir: str):
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, {PRIMARY}, #2a4268);
        border-radius: 14px;
        padding: 28px 36px;
        margin-bottom: 28px;
    '>
        <h2 style='color: white !important; margin: 0 0 6px 0; font-size: 1.6rem;'>
            GNN & Spatial-Temporal Analysis
        </h2>
        <p style='color: rgba(255,255,255,0.75); margin: 0; font-size: 0.88rem;'>
            Visualizing the spatial graph structure, GCN/TCN embedding clusters,
            and temporal workload patterns across Chicago's 77 community areas.
        </p>
    </div>
    """, unsafe_allow_html=True)

    #  Community Area Adjacency Graph 
    st.markdown("### Community Area Adjacency Graph")
    st.markdown(f"""
    <p style='color: #555; font-size: 0.88rem; margin-bottom: 16px;'>
        The GCN operates on a spatial graph with <strong>77 nodes</strong> (community areas) and
        <strong>394 edges</strong> (shared geographic borders).
        <strong>Node color</strong> = GCN importance score (weighted degree: sum of adjacent edge weights).
        <strong>Edge width/color</strong> = GCN edge weight (10,000 / centroid distance).
    </p>
    """, unsafe_allow_html=True)

    edges, lats, lons, names, area_numbers = build_graph_data(gdf)

    n_nodes = len(lats)

    #  Compute node degrees (for hover) 
    degrees = np.zeros(n_nodes, dtype=int)
    for i, j, _ in edges:
        degrees[i] += 1
        degrees[j] += 1

    #  Compute weighted degree (GCN importance score) 
    weighted_degree = np.zeros(n_nodes, dtype=float)
    for i, j, w in edges:
        weighted_degree[i] += w
        weighted_degree[j] += w

    #  Normalise edge weights for visual scaling 
    weights      = np.array([w for _, _, w in edges])
    w_min, w_max = weights.min(), weights.max()
    w_norm       = (weights - w_min) / (w_max - w_min + 1e-9)  # 0–1

    #  Helper: interpolate between two RGB colours 
    def lerp_color(t, low=(200, 220, 240), high=(57, 88, 134)):
        r = int(low[0] + t * (high[0] - low[0]))
        g = int(low[1] + t * (high[1] - low[1]))
        b = int(low[2] + t * (high[2] - low[2]))
        return r, g, b

    fig_graph = go.Figure()

    #  Draw edges grouped into 5 weight quintiles 
    QUINTILES = 5
    for q in range(QUINTILES):
        lo, hi = q / QUINTILES, (q + 1) / QUINTILES
        mask    = (w_norm >= lo) & (w_norm < hi)
        t       = (lo + hi) / 2
        r, g, b = lerp_color(t)
        alpha   = 0.25 + 0.55 * t   # 0.25 → 0.80
        width   = 0.6 + 3.4 * t     # 0.6  → 4.0

        edge_lats, edge_lons = [], []
        for idx, (i, j, _) in enumerate(edges):
            if mask[idx]:
                edge_lats += [lats[i], lats[j], None]
                edge_lons += [lons[i], lons[j], None]

        if edge_lats:
            fig_graph.add_trace(go.Scattermap(
                lat=edge_lats,
                lon=edge_lons,
                mode="lines",
                line=dict(width=width, color=f"rgba({r},{g},{b},{alpha})"),
                hoverinfo="skip",
                showlegend=False,
                name=f"Edge weight Q{q + 1}",
            ))

    #  Draw nodes coloured by weighted degree (uniform size) 
    fig_graph.add_trace(go.Scattermap(
        lat=lats,
        lon=lons,
        mode="markers+text",
        marker=dict(
            size=12,
            color=weighted_degree.tolist(),
            colorscale=[
                [0.0,  "rgb(214,230,248)"],
                [0.35, "rgb(106,153,205)"],
                [0.70, "rgb(57, 88, 134)"],
                [1.0,  "rgb(26, 42, 74)" ],
            ],
            opacity=0.92,
            colorbar=dict(
                title=dict(text="GCN Node<br>Importance", font=dict(size=11, family="Inter")),
                thickness=14,
                len=0.55,
                x=1.01,
                tickfont=dict(size=10, family="Inter"),
            ),
        ),
        text=[str(n) for n in area_numbers],
        textposition="top center",
        textfont=dict(size=8, color="#222"),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Area #%{customdata[1]}<br>"
            "Degree: %{customdata[2]}<br>"
            "Weighted Importance: %{customdata[3]:.4f}<extra></extra>"
        ),
        customdata=list(zip(names, area_numbers, degrees.tolist(), weighted_degree.tolist())),
        showlegend=False,
    ))

    fig_graph.update_layout(
        title=dict(
            text="Chicago Community Area Adjacency Graph - Node Importance & Edge Weight",
            font=dict(size=16, family="Inter"),
        ),
        map=dict(
            center=dict(lat=41.8372, lon=-87.6860),
            zoom=9.5,
            style="carto-positron",
            layers=[{
                "source": geojson,
                "type": "line",
                "color": "rgba(57,88,134,0.25)",
                "line": {"width": 0.8},
            }],
        ),
        height=650,
        margin=dict(t=60, b=10, l=10, r=10),
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig_graph, width='stretch')

    #  Summary metrics 
    col_n, col_e, col_deg, col_w = st.columns(4)
    with col_n:
        st.metric("Total Nodes", "77", help="One per community area")
    with col_e:
        st.metric("Total Edges", str(len(edges)), help="Shared geographic borders")
    with col_deg:
        top_idx = int(weighted_degree.argmax())
        st.metric(
            "Most Important Node",
            str(area_numbers[top_idx]),
            help=f"{names[top_idx]} : weighted importance {weighted_degree[top_idx]:.4f}",
        )
    with col_w:
        st.metric(
            "Max Edge Weight",
            f"{weights.max():.4f}",
            help="10,000 / centroid distance (metres). Higher = closer neighbours.",
        )

    st.markdown(f"""
    <div style='
        background: {WHITE};
        border-radius: 12px;
        padding: 18px 24px;
        margin-top: 12px;
        box-shadow: 0 4px 16px rgba(57,88,134,0.08);
        font-size: 0.84rem;
        color: #444;
        line-height: 1.9;
    '>
        <strong style='color:{PRIMARY};'>Reading the graph</strong><br>
        <span style='display:inline-block; width:14px; height:14px;
              background:rgb(26,42,74); border-radius:50%; vertical-align:middle;
              margin-right:6px;'></span>
        <strong>Dark nodes</strong> : high GCN importance (large sum of adjacent edge weights;
        area receives stronger aggregated spatial signal during message-passing).<br>
        <span style='display:inline-block; width:14px; height:14px;
              background:rgb(214,230,248); border-radius:50%; vertical-align:middle;
              margin-right:6px;'></span>
        <strong>Light nodes</strong> : low importance (peripheral areas with weak or few connections).<br>
        <strong>Thick dark edges</strong> : high GCN weight (nearby centroids → stronger spatial message-passing).<br>
        <strong>Thin light edges</strong> : low weight (distant neighbours → weaker influence).
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    #  Spatial Embedding Clusters 
    st.markdown("### Spatial Embedding Clusters (GMM, k=2)")
    st.markdown(f"""
    <p style='color: #555; font-size: 0.88rem; margin-bottom: 16px;'>
        GCN embeddings are clustered using Gaussian Mixture Model. BIC selects k=2 as optimal.
    </p>
    """, unsafe_allow_html=True)

    # Dynamic loading from CSV file
    spatial_df = pd.read_csv(os.path.join(base_dir, "spatial_cluster.csv"))

    # Compute spatial baseline statistics from raw dataframe
    area_stats = df.groupby("community_area").agg(
        records=("duration_days", "size"),
        mean_duration=("duration_days", "mean"),
        median_duration=("duration_days", "median"),
        event_rate=("event", "mean"),
    ).reset_index()

    spatial_merged = spatial_df.merge(area_stats, on="community_area", how="left")
    cluster_vols = spatial_merged.groupby("spatial_cluster")["records"].sum()
    high_vol_idx = cluster_vols.idxmax()

    spatial_merged["cluster_name"] = spatial_merged["spatial_cluster"].apply(
        lambda x: "High-Volume Cluster" if x == high_vol_idx else "Moderate-Volume Cluster"
    )

    # Map to gdf for choropleth mapping component
    gdf_plot = gdf.merge(spatial_merged, left_on="area_numbe", right_on="community_area", how="left")
    gdf_plot = gdf_plot.reset_index(drop=True)

    fig_spatial_cluster = px.choropleth_map(
        gdf_plot,
        geojson=gdf_plot.geometry.__geo_interface__,
        locations=gdf_plot.index,
        color="cluster_name",
        hover_name="community",
        hover_data={"records": True, "mean_duration": ":.1f", "event_rate": ":.3f"},
        color_discrete_map={"High-Volume Cluster": PRIMARY, "Moderate-Volume Cluster": ACCENT_GREEN},
        title="Spatial GMM Clusters of Chicago Community Areas",
        labels={"cluster_name": "Cluster"},
    )
    fig_spatial_cluster.update_layout(
        map=dict(center=dict(lat=41.8372, lon=-87.6860), zoom=9.5, style="carto-positron"),
        height=600,
        margin=dict(t=60, b=10, l=10, r=10),
        font=dict(family="Inter"),
        title_font=dict(size=16),
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_spatial_cluster, width='stretch')

    # Dynamically generated Spatial cluster table
    st.markdown("#### Spatial Cluster Characteristics")
    spatial_summary = spatial_merged.groupby("cluster_name").agg(
        records_sum=("records", "sum"),
        mean_dur=("mean_duration", "mean"),
        median_dur=("median_duration", "median"),
        ev_rate=("event_rate", "mean")
    ).reset_index().sort_values("records_sum", ascending=False)

    cluster_stats = pd.DataFrame({
        "Cluster": spatial_summary["cluster_name"],
        "Records": spatial_summary["records_sum"].map("{:,.0f}".format),
        "Mean Duration": spatial_summary["mean_dur"].map("{:.2f} days".format),
        "Median Duration": spatial_summary["median_dur"].map("{:.0f} days".format),
        "Event Rate": spatial_summary["ev_rate"].map("{:.4f}".format),
        "Profile": [
            "Areas with higher relative complaint workloads across core districts" if "High" in c 
            else "Areas with suburban or peripheral workload densities" for c in spatial_summary["cluster_name"]
        ]
    })
    st.dataframe(cluster_stats, width='stretch', hide_index=True)

    st.divider()

    #  Temporal Workload Clusters 
    st.markdown("### Temporal Workload Pattern Analysis (GMM, k=3)")

    # Dynamic loading from CSV file
    temporal_df = pd.read_csv(os.path.join(base_dir, "temporal_cluster.csv"))
    
    # Sort automatically by total historical workload metrics to calculate names dynamically
    cluster_means = temporal_df.groupby("temporal_cluster")["total_60d_workload"].mean().sort_values()
    temp_map = {}
    if len(cluster_means) == 3:
        temp_map[cluster_means.index[0]] = "Lower Workload"
        temp_map[cluster_means.index[1]] = "Medium Workload"
        temp_map[cluster_means.index[2]] = "High Workload"
    else:
        for idx, c_id in enumerate(cluster_means.index):
            temp_map[c_id] = f"Cluster {c_id}"

    temporal_df["cluster_base_name"] = temporal_df["temporal_cluster"].map(temp_map)
    counts = temporal_df["cluster_base_name"].value_counts()
    temporal_df["cluster_name"] = temporal_df["cluster_base_name"].apply(lambda x: f"{x} ({counts[x]} areas)")

    # Merge with gdf for choropleth plotting component
    gdf_temp = gdf.merge(temporal_df, left_on="area_numbe", right_on="community_area", how="left")
    gdf_temp = gdf_temp.reset_index(drop=True)

    color_discrete_map = {}
    for k in temporal_df["cluster_name"].unique():
        if "High" in k:
            color_discrete_map[k] = "#c75b5b"
        elif "Medium" in k:
            color_discrete_map[k] = ACCENT_GOLD
        else:
            color_discrete_map[k] = ACCENT_GREEN

    fig_temp_cluster = px.choropleth_map(
        gdf_temp,
        geojson=gdf_temp.geometry.__geo_interface__,
        locations=gdf_temp.index,
        color="cluster_name",
        hover_name="community",
        hover_data={"total_60d_workload": True, "mean_daily_workload": ":.1f", "max_daily_workload": True},
        color_discrete_map=color_discrete_map,
        title="Temporal Workload Clusters Across Community Areas",
    )
    fig_temp_cluster.update_layout(
        map=dict(center=dict(lat=41.8372, lon=-87.6860), zoom=9.5, style="carto-positron"),
        height=600,
        margin=dict(t=60, b=10, l=10, r=10),
        font=dict(family="Inter"),
        title_font=dict(size=16),
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_temp_cluster, width='stretch')

    # Dynamically generated Temporal cluster summary table
    st.markdown("#### Temporal Cluster Characteristics")
    temporal_summary = temporal_df.groupby("cluster_name").agg(
        n_areas=("community_area", "count"),
        mean_60d=("total_60d_workload", "mean"),
        mean_daily=("mean_daily_workload", "mean"),
        mean_peak=("max_daily_workload", "mean")
    ).reset_index()

    # Dynamic example area extraction process
    examples = []
    for c_name in temporal_summary["cluster_name"]:
        ex_names = temporal_df[temporal_df["cluster_name"] == c_name]["community_name"].head(2).tolist()
        examples.append(", ".join([name.title() for name in ex_names]))

    temp_cluster_stats = pd.DataFrame({
        "Cluster": temporal_summary["cluster_name"],
        "Number of Areas": temporal_summary["n_areas"],
        "Mean 60-Day Workload": temporal_summary["mean_60d"].map("{:.2f}".format),
        "Mean Daily Workload": temporal_summary["mean_daily"].map("{:.2f}".format),
        "Mean Peak Daily": temporal_summary["mean_peak"].map("{:.2f}".format),
        "Example Areas": examples
    })
    st.dataframe(temp_cluster_stats, width='stretch', hide_index=True)

    st.divider()

    #  Citywide Workload Chart - Direct HTML Embedding
    st.markdown("### Citywide 60-Day Complaint Workload")
    html_path_1 = os.path.join(base_dir, "citywide_60_day_complaint_workload.html")
    if os.path.exists(html_path_1):
        with open(html_path_1, "r", encoding="utf-8") as f:
            html_content_1 = f.read()
        components.html(html_content_1, height=430, scrolling=False)
    else:
        st.error("HTML file 'citywide_60_day_complaint_workload.html' not found.")

    #  Workload by temporal cluster - Direct HTML Embedding
    st.markdown("### Average 60-Day Workload by Temporal Cluster")
    html_path_2 = os.path.join(base_dir, "average_workload_by_temporal_cluster.html")
    if os.path.exists(html_path_2):
        with open(html_path_2, "r", encoding="utf-8") as f:
            html_content_2 = f.read()
        components.html(html_content_2, height=430, scrolling=False)
    else:
        st.error("HTML file 'average_workload_by_temporal_cluster.html' not found.")