"""Utilities for hierarchical clustering analysis with high-dimensional feature arrays.

This module provides a complete pipeline for hierarchical clustering, including
data standardization, distance matrix computation, cluster evaluation, and
interactive visualizations (dendrograms and heatmaps) using Plotly. It is
designed to work with high-dimensional data such as TSFEL features.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, cophenet
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as ff
import plotly.express as px

def pipeline_clustering_tsfel(features_array, labels, groups, metric="cosine", method="average"):
    """Complete hierarchical clustering pipeline optimized for high-dimensional data.

    Args:
        features_array:
            Feature array of shape ``(n_samples, n_features)``. For example,
            240 TSFEL features.
        labels:
            Name or identifier for each record/patient.
        groups:
            Class label for each record (e.g., ``["Healthy", "Severe"]``).
        metric:
            Distance metric. ``"cosine"`` or ``"correlation"`` are recommended
            for datasets with more than 100 features.
        method:
            Linkage method (e.g., ``"average"``, ``"ward"``, ``"complete"``).

    Returns:
        A tuple containing the linkage matrix and the square distance matrix.
    """
    
    features_array = np.array(features_array)
    labels = list(labels)
    groups = np.array(groups)
    
    # 1. Standardize features (crucial with 240 features to prevent one
    # variable from dominating the distance computation).
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
    
    # 2. Vectorized distance computation (replaces nested loops).
    # pdist computes pairwise distances between rows.
    D_condensed = pdist(features_scaled, metric=metric)
    
    # Build the square distance matrix for analysis and heatmap visualization.
    D_square = squareform(D_condensed)
    
    # 3. Hierarchical clustering.
    Z = linkage(D_condensed, method=method)
    
    # Evaluate tree quality via the cophenetic correlation coefficient.
    c, _ = cophenet(Z, D_condensed)
    print(f"Cophenetic index (closer to 1 is better): {c:.3f}\n")
    
    # 4. Compute mean intra-group distances.
    unique_groups = np.unique(groups)
    print("--- Mean distances (intra-group) ---")
    for g in unique_groups:
        idx_g = np.where(groups == g)[0]
        if len(idx_g) > 1:
            # Extract distances only between members of the same group.
            dists_intra = D_square[np.ix_(idx_g, idx_g)]
            # Compute mean excluding the zero diagonal.
            mean_dist = dists_intra[np.triu_indices_from(dists_intra, k=1)].mean()
            print(f"Group '{g}': {mean_dist:.3f}")

    # 5. Interactive visualization with Plotly.

    # --- A. Dendrogram ---
    # Plotly handles linkage internally when given a custom function.
    fig_dendro = ff.create_dendrogram(
        features_scaled, 
        orientation='bottom', 
        labels=labels, 
        linkagefun=lambda x: Z
    )
    fig_dendro.update_layout(
        title=f"Interactive Dendrogram (Distance: {metric}, Linkage: {method})",
        width=1200, height=700,
        xaxis_tickangle=-45
    )
    fig_dendro.show()
    
    # --- B. ClusterMap (reordered heatmap) ---
    # Get the element order as displayed in the dendrogram.
    leaves_order = fig_dendro['layout']['xaxis']['ticktext']
    leave_indices = [labels.index(l) for l in leaves_order]
    
    # Reorder the square distance matrix according to this order.
    D_reordered = D_square[np.ix_(leave_indices, leave_indices)]
    
    fig_heatmap = px.imshow(
        D_reordered, 
        x=leaves_order, 
        y=leaves_order,
        color_continuous_scale="Viridis",
        title="Distance Heatmap (Reordered by Clustering)"
    )
    fig_heatmap.update_layout(width=1000, height=1000)
    fig_heatmap.show()

    return Z, D_square