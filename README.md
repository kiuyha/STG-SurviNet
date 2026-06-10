# STG-SurviNet

A hybrid survival analysis model for predicting infrastructure complaint resolution times across Chicago's 77 community areas. Built by Syafira Najema Putri Anisa, Ketut Shridhara, and Naura Kanaya Putri Masruri for the Data Mining course at Universitas Negeri Surabaya (2024-INT).

Live dashboard: [https://stg-survinet.streamlit.app](https://stg-survinet.streamlit.app/)

---

## What This Does

Chicago's 311 system gets a lot of infrastructure complaints. Most resolve in a few days. A small number sit open for over four years. The interesting modeling problem is not the fast ones; it is what to do with the records that never close, or close after a year. Dropping them throws away information. Treating them like completed cases distorts the estimates.

This project frames complaint resolution as a time-to-event problem. How long until a complaint gets resolved, and how does that risk differ across community areas and complaint types? The model produces risk scores that can rank complaints by predicted delay, which is more useful for resource planning than a binary "will this close or not."

---

## Architecture

STG-SurviNet combines three components:

**Graph Convolutional Network (GCN)**
Chicago's 77 community areas become nodes in a spatial graph. Edges connect adjacent areas, weighted by inverse centroid distance. Two GCN layers produce 64-dimensional spatial embeddings that carry information from neighboring areas, not just the target area itself.

**Temporal Convolutional Network (TCN)**
For each community area, a 60-day daily complaint count sequence feeds into two dilated convolutional layers (kernel size 3, dilations 2 and 4). The output is a 32-dimensional embedding capturing recent workload levels.

**Deep Cox Proportional Hazards Layer**
The spatial embedding, temporal embedding, and incident-level features (16 dimensions) are concatenated into a 112-dimensional vector, then passed through fully connected layers of 256, 128, and 64 units. The final output is a log-hazard score per complaint. Survival curves are derived by combining this with the fitted baseline hazard.

The model is trained using a mini-batch approximation of the Cox Negative Partial Log-Likelihood, with small duration jitter to reduce ties.

---

## Data

Source: [Chicago 311 Service Requests](https://data.cityofchicago.org/Service-Requests/311-Service-Requests/v6vf-nfxy/about_data) (City of Chicago Open Data Portal)

Four complaint categories were selected:

| Category | Records |
|---|---|
| Pothole in Street Complaint | 317,589 |
| Street Light Out Complaint | 222,722 |
| Tree Debris Clean-Up Request | 60,441 |
| Traffic Signal Out Complaint | 49,342 |

Raw total: 650,094 records (2022-2026). After removing records with missing spatial identifiers, duplicates, and negative durations: 541,625 records.

**Survival variables:**
- `duration_days`: days from complaint creation to the processed completion boundary. Missing closed dates are filled with the dataset maximum creation date.
- `event`: 1 for Completed records, 0 for Open and Canceled.

**Data split:** 60/20/10/10 (training / validation / calibration / test), giving 324,975 training and 54,163 test records.

---

## Results

### Model Comparison

| Model | C-index | IBS | D-Cal p-value | MAE |
|---|---|---|---|---|
| STG-SurviNet | **0.7075** | **0.1209** | 0.0000 | **13.68 days** |
| Cox PH | 0.6527 | 0.1421 | 0.0000 | 15.01 days |
| Random Survival Forest | 0.4308 | 0.1290 | 0.0000 | 14.37 days |
| Kaplan-Meier | 0.5000 | 0.1579 | 0.0000 | 15.52 days |

STG-SurviNet leads on every metric. The MAE of 13.68 days is still well above the 7-day target, so the model is better understood as a risk-ranking tool than a precise duration predictor. The D-Calibration p-value is below 0.0001 for every model, meaning no model produces well-calibrated survival probabilities under the event-only chi-square test. Scores should be used for relative comparisons, not absolute predictions.

### Ablation Study

| Variant | C-index | IBS | MAE | Change in C-index |
|---|---|---|---|---|
| Full STG-SurviNet | 0.7075 | 0.1209 | 13.68 days | 0.0000 |
| No Temporal | 0.7043 | 0.1223 | 13.80 days | -0.0032 |
| No Spatial | 0.7029 | 0.1229 | 13.82 days | -0.0046 |
| No Incident Features | 0.5467 | 0.1474 | 15.14 days | -0.1608 |

Incident-level features carry most of the predictive weight. Removing them drops the C-index by 0.16. The spatial and temporal components each add smaller but consistent improvements.

### Delay Risk by Complaint Category

| Category | Mean Delay-Risk Score |
|---|---|
| Tree Debris Clean-Up Request | 0.1985 |
| Pothole in Street Complaint | 0.0167 |
| Street Light Out Complaint | -0.2699 |
| Traffic Signal Out Complaint | -0.5183 |

Negative scores do not mean no delay. They mean lower predicted delay risk relative to the categories above.

At the community-area level, O'Hare has the highest mean delay-risk score (0.1319), with a mean resolution duration of 29.95 days. The highest area-category combination is Roseland paired with Tree Debris Clean-Up Requests (mean delay-risk score: 0.5497, mean duration: 22.79 days).

---

## Spatial and Temporal Clustering

**Spatial (GMM on GCN embeddings):** Two clusters selected by BIC. Cluster 0 holds 389,266 records with a mean duration of 19.31 days. Cluster 1 holds 152,359 records, mean duration 18.35 days. Both clusters share a median duration of 3 days. The difference is mainly volume, with higher-volume areas concentrated in the north.

**Temporal (GMM on TCN embeddings):** Three clusters. Cluster 2 (10 areas) averages 326.90 complaints over 60 days, with Austin (509) and Near West Side (493) at the top. Cluster 0 (22 areas) averages 247.59. Cluster 1 (45 areas) averages 142.82. Workload varies enough across areas that the 60-day sequence adds real information.

---

## Limitations

The C-index of 0.7075 sits below the 0.75 target, and the MAE of 13.68 days is nearly twice the 7-day target. All models fail the D-Calibration test, so the survival probability estimates are not reliable as absolute values. Treat the outputs as relative rankings, not forecasts.

A few choices in the implementation also shape how results should be read. Canceled complaints are treated as censored (event = 0) alongside open ones, which is a reasonable choice but not the only one, and it affects the survival curves. The complaint density variable is computed from coordinate range rather than actual polygon area, so it is a rough proxy. The TCN window covers only March to May 2026, not the full four-year study period, so it reflects recent workload patterns rather than seasonal trends over time.

None of the patterns identified here establish causation. High workload correlating with longer resolution times is a model association, not proof that workload is the driver.

---

## Links

- GitHub: [https://github.com/kiuyha/STG-SurviNet](https://github.com/kiuyha/STG-SurviNet)
- Dashboard: [https://stg-survinet.streamlit.app](https://stg-survinet.streamlit.app/)
- Data source: [Chicago 311 Service Requests](https://data.cityofchicago.org/Service-Requests/311-Service-Requests/v6vf-nfxy/about_data)

---

## References

City of Chicago. (2026). *311 Service Requests*. Chicago Data Portal.

Defferrard, M., Bresson, X., & Vandergheynst, P. (2016). Convolutional neural networks on graphs with fast localized spectral filtering. *Advances in Neural Information Processing Systems*, 29, 3844-3852.

Katzman, J. L., Shaham, U., Cloninger, A., Bates, J., Jiang, T., & Kluger, Y. (2018). DeepSurv: Personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Medical Research Methodology*, 18(1), Article 24.

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *Proceedings of ICLR 2017*.

Lee, C., Zame, W. R., Yoon, J., & van der Schaar, M. (2018). DeepHit: A deep learning approach to survival analysis with competing risks. *Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1), 2314-2321.

---

## Course Info

Program Studi Sains Data, Universitas Negeri Surabaya
Data Mining Course, Class 2024-INT
Supervisor: Moh. Khoridatul Huda, S.Pd., M.Si., Ph.D.