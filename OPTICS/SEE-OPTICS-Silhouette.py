import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gaussian_kde, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, OPTICS
from sklearn.metrics import silhouette_score

# rpy2 (for AdhereR)
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

# -------------------------
# Load dataset – uses AdhereR's med.events
# -------------------------
pandas2ri.activate()
robjects.r('library(AdhereR)')
med_events_r = robjects.r('med.events')
med_events = pandas2ri.rpy2py(med_events_r)

ExamplePats = med_events.copy()
tidy = ExamplePats.copy()
tidy.columns = ["pnr", "eksd", "perday", "ATC", "dur_original"]
tidy['eksd'] = pd.to_datetime(tidy['eksd'], format='%m/%d/%Y')

# -------------------------
# SEE using KMeans with Silhouette Analysis
# -------------------------
def SEE_kmeans(arg1):
    # Filter rows where ATC equals arg1
    subset = tidy[tidy['ATC'] == arg1].copy()
    base_data = subset.copy()
    
    # Sort by patient and date, compute previous prescription date
    data = subset.sort_values(['pnr', 'eksd']).copy()
    data['prev_eksd'] = data.groupby('pnr')['eksd'].shift(1)
    data = data.dropna(subset=['prev_eksd']).copy()
    
    # For each patient, randomly sample one row
    data = data.groupby('pnr', group_keys=False).apply(lambda x: x.sample(1, random_state=1234)).reset_index(drop=True)
    data = data[['pnr', 'eksd', 'prev_eksd']]
    
    # Compute event interval
    data['event.interval'] = (data['eksd'] - data['prev_eksd']).dt.days.astype(float)
    
    # Compute ECDF and retain lower 80%
    ecdf_func = ECDF(data['event.interval'].values)
    x_vals = np.sort(data['event.interval'].values)
    y_vals = ecdf_func(x_vals)
    df_ecdf = pd.DataFrame({'x': x_vals, 'y': y_vals})
    df_ecdf_80 = df_ecdf[df_ecdf['y'] <= 0.8]
    ni = df_ecdf_80['x'].max()
    
    # Subset data up to the 80th percentile
    data_subset = data[data['event.interval'] <= ni].copy()
    
    # --- Density estimation on log(event.interval) ---
    log_intervals = np.log(data_subset['event.interval'])
    kde = gaussian_kde(log_intervals)
    x_grid = np.linspace(log_intervals.min(), log_intervals.max(), 100)
    # We won’t use KDE for clustering in this version; we cluster on ECDF x-values.
    
    # --- KMeans Clustering with Silhouette Analysis on ECDF x-values ---
    X = df_ecdf[['x']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    sil_scores = {}
    for k in range(2, min(11, len(X_scaled))):
        km = KMeans(n_clusters=k, random_state=1234)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        sil_scores[k] = score
    if sil_scores:
        optimal_k = max(sil_scores, key=sil_scores.get)
    else:
        optimal_k = 1
    
    plt.figure(figsize=(8,5))
    plt.plot(list(sil_scores.keys()), list(sil_scores.values()), marker='o')
    plt.title("KMeans Silhouette Analysis")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.show()
    
    km_final = KMeans(n_clusters=optimal_k, random_state=1234)
    labels_final = km_final.fit_predict(X_scaled)
    df_ecdf['cluster'] = labels_final
    
    # Compute cluster statistics (on original x)
    valid_clusters = df_ecdf.copy()  # In KMeans, all points are assigned
    cluster_stats = (valid_clusters.groupby('cluster')['x']
                     .agg(min_log=lambda x: np.log(x).min(),
                          max_log=lambda x: np.log(x).max(),
                          median_log=lambda x: np.log(x).median())
                     .reset_index())
    cluster_stats['Minimum'] = np.exp(cluster_stats['min_log'])
    cluster_stats['Maximum'] = np.exp(cluster_stats['max_log'])
    cluster_stats['Median'] = np.exp(cluster_stats['median_log'])
    cluster_stats = cluster_stats[cluster_stats['Median'] > 0]
    
    # Cross join to assign clusters based on whether event.interval falls in [Minimum, Maximum]
    data['_key'] = 1
    cluster_stats['_key'] = 1
    cross_df = pd.merge(data, cluster_stats, on='_key').drop('_key', axis=1)
    cross_df['Final_cluster'] = cross_df.apply(lambda row: row['cluster'] if (row['event.interval'] >= row['Minimum'] and row['event.interval'] <= row['Maximum']) else np.nan, axis=1)
    results = cross_df.dropna(subset=['Final_cluster']).copy()[['pnr','Median','Final_cluster']]
    
    most_common_cluster = results['Final_cluster'].value_counts().idxmax()
    default_median = cluster_stats.loc[cluster_stats['cluster'] == most_common_cluster, 'Median'].values[0]
    
    data = pd.merge(data, results, on='pnr', how='left')
    data['Median'] = data['Median'].fillna(default_median)
    data['Cluster'] = data['Final_cluster'].fillna(0)
    data['test'] = (data['event.interval'] - data['Median']).round(1)
    
    final_df = pd.merge(base_data, data[['pnr','Median','Cluster']], on='pnr', how='left')
    final_df['Median'] = final_df['Median'].fillna(default_median)
    final_df['Cluster'] = final_df['Cluster'].fillna(0)
    
    return final_df, data  # return the clustering details with the 'test' differences

# -------------------------
# SEE using OPTICS with Silhouette Analysis to find optimal min_samples
# -------------------------
def SEE_optics(arg1):
    # Filter rows where ATC equals arg1
    subset = tidy[tidy['ATC'] == arg1].copy()
    base_data = subset.copy()
    
    # Sort by patient and date, compute previous prescription date
    data = subset.sort_values(['pnr', 'eksd']).copy()
    data['prev_eksd'] = data.groupby('pnr')['eksd'].shift(1)
    data = data.dropna(subset=['prev_eksd']).copy()
    
    # For each patient, randomly sample one row
    data = data.groupby('pnr', group_keys=False).apply(lambda x: x.sample(1, random_state=1234)).reset_index(drop=True)
    data = data[['pnr', 'eksd', 'prev_eksd']]
    
    # Compute event interval
    data['event.interval'] = (data['eksd'] - data['prev_eksd']).dt.days.astype(float)
    
    # Compute ECDF and retain lower 80%
    ecdf_func = ECDF(data['event.interval'].values)
    x_vals = np.sort(data['event.interval'].values)
    y_vals = ecdf_func(x_vals)
    df_ecdf = pd.DataFrame({'x': x_vals, 'y': y_vals})
    df_ecdf_80 = df_ecdf[df_ecdf['y'] <= 0.8]
    ni = df_ecdf_80['x'].max()
    
    # Subset data up to the 80th percentile
    data_subset = data[data['event.interval'] <= ni].copy()
    
    # --- Density estimation on log(event.interval) ---
    log_intervals = np.log(data_subset['event.interval'])
    kde = gaussian_kde(log_intervals)
    x_grid = np.linspace(log_intervals.min(), log_intervals.max(), 100)
    y_kde = kde(x_grid)
    
    # (We again work with the ECDF x-values for clustering)
    X = df_ecdf[['x']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- Silhouette Analysis using OPTICS ---
    # Loop over candidate values for min_samples (e.g., 3 to 20)
    sil_scores = {}
    candidate_min_samples = range(2, 11)
    for ms in candidate_min_samples:
        optics_candidate = OPTICS(min_samples=ms, cluster_method='xi')
        candidate_labels = optics_candidate.fit_predict(X_scaled)
        unique_labels = set(candidate_labels)
        # Only compute silhouette score if more than one cluster exists (ignoring noise, labeled as -1)
        if len(unique_labels - {-1}) > 1:
            score = silhouette_score(X_scaled, candidate_labels)
            sil_scores[ms] = score
    if sil_scores:
        optimal_ms = max(sil_scores, key=sil_scores.get)
    else:
        optimal_ms = 2
    
    plt.figure(figsize=(8,5))
    plt.plot(list(sil_scores.keys()), list(sil_scores.values()), marker='o')
    plt.title("OPTICS Silhouette Analysis (min_samples)")
    plt.xlabel("min_samples")
    plt.ylabel("Silhouette Score")
    plt.show()
    
    optics_final = OPTICS(min_samples=optimal_ms, cluster_method='xi')
    labels_final = optics_final.fit_predict(X_scaled)
    df_ecdf['cluster'] = labels_final
    
    # Compute cluster statistics (ignoring noise - label -1)
    valid_clusters = df_ecdf[df_ecdf['cluster'] != -1]
    if not valid_clusters.empty:
        cluster_stats = (valid_clusters.groupby('cluster')['x']
                         .agg(min_log=lambda x: np.log(x).min(),
                              max_log=lambda x: np.log(x).max(),
                              median_log=lambda x: np.log(x).median())
                         .reset_index())
        cluster_stats['Minimum'] = np.exp(cluster_stats['min_log'])
        cluster_stats['Maximum'] = np.exp(cluster_stats['max_log'])
        cluster_stats['Median'] = np.exp(cluster_stats['median_log'])
        cluster_stats = cluster_stats[cluster_stats['Median'] > 0]
    else:
        cluster_stats = pd.DataFrame()
    
    # Cross join to assign clusters based on event.interval falling within [Minimum, Maximum]
    data['_key'] = 1
    if not cluster_stats.empty:
        cluster_stats['_key'] = 1
        cross_df = pd.merge(data, cluster_stats, on='_key').drop('_key', axis=1)
        cross_df['Final_cluster'] = cross_df.apply(lambda row: row['cluster'] if (row['event.interval'] >= row['Minimum'] and row['event.interval'] <= row['Maximum']) else np.nan, axis=1)
        results = cross_df.dropna(subset=['Final_cluster']).copy()[['pnr','Median','Final_cluster']]
        most_common_cluster = results['Final_cluster'].value_counts().idxmax()
        default_median = cluster_stats.loc[cluster_stats['cluster'] == most_common_cluster, 'Median'].values[0]
    else:
        results = pd.DataFrame(columns=['pnr','Median','Final_cluster'])
        default_median = data['event.interval'].median()
    
    data = pd.merge(data, results, on='pnr', how='left')
    data['Median'] = data['Median'].fillna(default_median)
    data['Cluster'] = data['Final_cluster'].fillna(0)
    data['test'] = (data['event.interval'] - data['Median']).round(1)
    
    final_df = pd.merge(base_data, data[['pnr','Median','Cluster']], on='pnr', how='left')
    final_df['Median'] = final_df['Median'].fillna(default_median)
    final_df['Cluster'] = final_df['Cluster'].fillna(0)
    
    return final_df, data

# -------------------------
# Perform SEE on a chosen medication (e.g., "medA") using both methods
# -------------------------
final_km, details_km = SEE_kmeans("medA")
final_optics, details_optics = SEE_optics("medA")

# Extract the 'test' differences from each clustering method
test_km = details_km['test'].dropna()
test_optics = details_optics['test'].dropna()

# Perform Mann–Whitney U test on the two distributions
u_stat, p_val = mannwhitneyu(test_km, test_optics, alternative='two-sided')
print("Mann–Whitney U Test comparing KMeans vs OPTICS (using silhouette score for min_samples):")
print(f"U statistic: {u_stat:.3f}")
print(f"p-value: {p_val:.3f}")

# Optional: visualize the two distributions side by side
plt.figure(figsize=(8,6))
plt.boxplot([test_km, test_optics], labels=['KMeans', 'OPTICS'])
plt.title("Comparison of 'test' differences between clustering methods")
plt.ylabel("Difference (event.interval - Median)")
plt.show()

plt.figure(figsize=(8,6))
combined = pd.DataFrame({
    'Difference': np.concatenate([test_km.values, test_optics.values]),
    'Method': ['KMeans'] * len(test_km) + ['OPTICS'] * len(test_optics)
})
sns.violinplot(x='Method', y='Difference', data=combined, inner="quartile")
plt.title("Violin Plot of 'test' Differences")
plt.ylabel("Difference (event.interval - Median)")
plt.show()
