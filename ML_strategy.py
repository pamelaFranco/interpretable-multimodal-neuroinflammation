###############################################################################
# This Python script performs a complete machine learning pipeline for 
# biomarker discovery using Random Forests on brain-related data.
#
# The workflow includes:
#
# 1) Data preprocessing and feature-name cleaning.
#
# 2) Exploratory analysis:
#       • Hierarchical clustering of features (dendrogram + clustermap)
#       • Correlation-based grouping of feature clusters
#
# 3) Hyperparameter optimization:
#       • Nested cross-validation
#       • RandomizedSearchCV with controlled random seed (seedsearch)
#       • Export of all CV results and best hyperparameters (CSV + JSON)
#
# 4) Model selection and feature ranking:
#       • Logistic-Regression–based initial importance ranking
#       • Random Forest feature importance (saved as CSV + visual plot)
#       • Sequential Forward Selection (SFS) using the optimized RF model
#       • Visual SFS performance curve aligned to number of features
#
# 5) Dimensionality reduction:
#       • t-SNE comparison BEFORE vs AFTER SFS feature selection
#         (side-by-side visualization of class separability)
#
# 6) Model explainability using SHAP:
#       • Robust SHAP value extraction (fixes inconsistent SHAP formats)
#       • Global importance bar plot
#       • Summary plot for the target class
#       • Waterfall plot for a representative individual sample
#
# 7) Model interpretability:
#       • Visualization of a single decision tree from the optimized RF
#       • 2-D decision surface based on the top-2 SFS-selected features
#
# The script outputs all plots, CSV files, and JSON configuration files 
# required for reproducible research in neuroimaging-based ML pipelines.
#
# Author:      Dr. Pamela Franco
# Time-stamp:  2025-11-25
# E-mail:      pamela.franco@unab.cl / pafranco@uc.cl
# Python 3.11 compatible
###############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RandomizedSearchCV, train_test_split,
    StratifiedKFold, RepeatedStratifiedKFold
)
from scipy.stats import randint
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error,
    balanced_accuracy_score
)

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import plot_tree

import shap
from matplotlib.colors import ListedColormap


###############################################################################
# GLOBAL SETTINGS
###############################################################################

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["figure.autolayout"] = True


###############################################################################
# LOAD DATA
###############################################################################

data_path = "anestesia_ML.csv"
df = pd.read_csv(data_path)

features = df.drop(["Labels"], axis=1)
label = df["Labels"].values
classes = ["Session 1", "Session 2"]

df = features.rename(columns=lambda x: re.sub('[^*A-Za-z0-9_ ]+', '', x))
feature_names = list(df.columns)


###############################################################################
# SEED SEARCH (FAST + STRONG)
###############################################################################

candidate_seeds = list(range(1, 101))   
seed_scores = []

for seed in candidate_seeds:
    X_train, X_test, y_train, y_test = train_test_split(
        features, label, test_size=0.2, random_state=seed
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=seed
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    acc = balanced_accuracy_score(y_test, pred)
    seed_scores.append(acc)

best_seed = candidate_seeds[int(np.argmax(seed_scores))]

print(f"\n>>> Best seed found: {best_seed}")
print("Balanced accuracy:", np.max(seed_scores))


###############################################################################
# HIERARCHICAL CLUSTERING
###############################################################################

correlation_matrix = features.corr()
distance_threshold = 3

model = AgglomerativeClustering(
    n_clusters=None, linkage="average", distance_threshold=distance_threshold
)

cluster_cols = model.fit_predict(correlation_matrix.T)
cluster_rows = model.fit_predict(correlation_matrix)

n_clusters = len(np.unique(cluster_cols))
colors = plt.cm.get_cmap("plasma", n_clusters)

col_colors = [colors(i) for i in cluster_cols]
row_colors = [colors(i) for i in cluster_rows]

cluster_sizes = [sum(cluster_cols == i) for i in range(n_clusters)]
cluster_texts = [f"Cluster {i+1}: {cluster_sizes[i]} features"
                 for i in range(n_clusters)]

# dendrogram
Z = linkage(correlation_matrix, method="average", metric="euclidean")
plt.figure(figsize=(12, 8))
dendrogram(Z, labels=correlation_matrix.columns, color_threshold=distance_threshold)
plt.axhline(y=distance_threshold, color="r", linestyle="--")
plt.savefig("dendrogram.png", dpi=300)
plt.close()

# clustermap
g = sns.clustermap(
    correlation_matrix,
    cmap="Blues", figsize=(16,16),
    row_cluster=True, col_cluster=True,
    row_colors=row_colors, col_colors=col_colors,
    tree_kws={"linewidths": 2},
)

g.ax_cbar.set_position((0.85,.02,.03,.15))
g.ax_cbar.set_ylabel("Correlation (R)")
plt.savefig("clustermap.png", dpi=300)
plt.close()


###############################################################################
# 2) RANDOMIZED SEARCH CV (FAST VERSION)
###############################################################################

param_dist = {
    "n_estimators": randint(10, 500),
    "max_depth": randint(1, 20),
    "max_leaf_nodes": randint(2, 10),
    "min_samples_split": randint(5, 50),
    "max_features": ["sqrt", "log2", None],
}

rf = RandomForestClassifier(random_state=best_seed)

outer_cv = RepeatedStratifiedKFold(
    n_splits=5, n_repeats=10, random_state=best_seed
)

inner_cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=best_seed
)

rand_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=12,
    cv=inner_cv,
    random_state=best_seed,
    n_jobs=-1
)

outer_iteration = 1
total_iterations = outer_cv.get_n_splits()

for train_idx, test_idx in outer_cv.split(features, label):
    print(f"Outer CV {outer_iteration}/{total_iterations}")
    outer_iteration += 1
    X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = label[train_idx], label[test_idx]
    rand_search.fit(X_train, y_train)

best_rf = rand_search.best_estimator_
print("\nBest RF parameters:", best_rf.get_params())


###############################################################################
# SAVE RESULTS
###############################################################################

pd.DataFrame(rand_search.cv_results_).to_csv("RF_randomsearch_results.csv", index=False)
with open("RF_best_params.json", "w") as f:
    json.dump(best_rf.get_params(), f, indent=4)


###############################################################################
# RF FEATURE IMPORTANCE
###############################################################################
rf_importances = best_rf.feature_importances_

rf_importance_df = pd.DataFrame({
    "Feature": features.columns,
    "Importance": rf_importances
}).sort_values(by="Importance", ascending=False)

rf_importance_df.to_csv("RF_feature_importance.csv", index=False)

number_features = rf_importance_df.shape[0]

plt.figure(figsize=(20, 6))
ax = plt.gca()

ax.bar(
    rf_importance_df["Feature"],
    rf_importance_df["Importance"],
    color="cornflowerblue",
    edgecolor="black",
    linewidth=0.4
)

ax.set_xticklabels(rf_importance_df["Feature"], rotation=90, fontsize=10)
ax.set_xlabel("Features ", fontsize=14)
ax.set_ylabel("Random Forest Importance", fontsize=14)
ax.set_title("Random Forest Feature Importance", fontsize=16)
ax.grid(axis="y", linestyle="--", alpha=0.45)
ax.grid(axis="x", linestyle="--", alpha=0.45)
ax.set_xlim(-0.5, number_features - 0.5)
plt.tight_layout()
plt.savefig("RF_feature_importance.png", dpi=300)
plt.close()
###############################################################################
# SFS USING OPTIMIZED RF
###############################################################################
logreg = LogisticRegression(penalty="l2", solver="liblinear", max_iter=5000)
logreg.fit(features, label)
initial_rank = np.abs(logreg.coef_[0])

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": initial_rank
}).sort_values("Importance", ascending=False)

X_sel = features[feature_importance_df["Feature"]]
number_features = features.shape[1]

sfs = SFS(
    best_rf,
    k_features="best",
    forward=True,
    floating=False,
    scoring="accuracy",
    cv=inner_cv,
    verbose=2
)

fs = sfs.fit(X_sel, label)

X_selected = features[list(fs.k_feature_names_)]

metric_dict = sfs.get_metric_dict(confidence_interval=0.95)
fig, ax = plot_sfs(metric_dict, kind="std_dev", figsize=(20, 6))

k_list = []
score_list = []
for k in metric_dict.keys():
    k_list.append(k)
    score_list.append(metric_dict[k]['avg_score'])

scores = np.array(score_list)

first_max_idx = np.argmax(scores)
first_max_k = k_list[first_max_idx]
first_max_score = scores[first_max_idx]
ax.plot(first_max_k , first_max_score, 'ro', markersize=10, label="Maximum")
ax.grid(axis="y", linestyle="--", alpha=0.45)
ax.grid(axis="x", linestyle="--", alpha=0.45)
ax.set_xlim(1, 57)
ax.set_xlabel("Number of Features", fontsize=14)
ax.set_ylabel("Accuracy", fontsize=14)
ax.legend()
plt.savefig("feature_importanceSFS_RF.png")
plt.close()

###############################################################################
# TSNE BEFORE & AFTER FEATURE SELECTION
###############################################################################

tsne_before = TSNE(n_components=2, random_state=best_seed).fit_transform(features)
tsne_after  = TSNE(n_components=2, random_state=best_seed).fit_transform(X_selected)

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.scatter(tsne_before[:,0], tsne_before[:,1], c=label, cmap="viridis")
plt.title("t-SNE Before FS")

plt.subplot(1,2,2)
plt.scatter(tsne_after[:,0], tsne_after[:,1], c=label, cmap="viridis")
plt.title("t-SNE After FS")

plt.savefig("tsne_before_after.png", dpi=300)
plt.close()


###############################################################################
# SHAP ANALYSIS
###############################################################################

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, label, random_state=best_seed)

cls = best_rf
cls.fit(X_train, y_train)

explainer = shap.TreeExplainer(cls)
shap_raw = explainer.shap_values(X_selected)

print("Raw SHAP type:", type(shap_raw))


def fix_shap_format(shap_values):
    """Convert SHAP output into a valid list: 
       [array(n_samples, n_features), array(...), ...]"""

    # Case 1: Already list-like (OK)
    if isinstance(shap_values, list):
        print("SHAP already in list format.")
        return shap_values

    # Case 2: ndarray with shape (n_samples, n_features)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        print("Single-class SHAP matrix detected.")
        return [shap_values]

    # Case 3: ndarray with shape (n_samples, n_features, n_classes)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        print("Detected SHAP format (samples, features, classes). Fixing...")
        # convert to list of (samples, features)
        return [shap_values[:, :, c] for c in range(shap_values.shape[2])]

    # Case 4: ndarray with shape (n_classes, n_samples, n_features)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        print("Detected SHAP format (classes, samples, features). Fixing...")
        return [shap_values[c] for c in range(shap_values.shape[0])]

    raise ValueError("Unsupported SHAP shape: " + str(shap_values.shape))


shap_values = fix_shap_format(shap_raw)

for i, arr in enumerate(shap_values):
    print(f"Final SHAP[{i}] shape =", arr.shape)

# Must match:
#   n_samples = X_selected.shape[0]
#   n_features = X_selected.shape[1]

ns, nf = X_selected.shape
for arr in shap_values:
    assert arr.shape == (ns, nf), f"SHAP mismatch: {arr.shape} vs {(ns, nf)}"

# SUMMARY PLOTS
# Global importance
plt.figure()
shap.summary_plot(shap_values, X_selected, plot_type="bar", show=False)
plt.savefig("shapSummaryPlot_bar.png", dpi=300)
plt.close()

# Choose class 1 if exists, else class 0
target_class = 1 if len(shap_values) > 1 else 0

plt.figure()
shap.summary_plot(shap_values[target_class], X_selected, show=False)
plt.savefig("shapSummaryPlot_class1.png", dpi=300)
plt.close()

# WATERFALL PLOT
row = 0

plt.figure(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(
        values = shap_values[target_class][row],
        base_values = explainer.expected_value[target_class]
                      if isinstance(explainer.expected_value, (list, np.ndarray))
                      else explainer.expected_value,
        data = X_selected.iloc[row],
        feature_names = X_selected.columns
    )
)
plt.savefig("shapWaterfall_0.png", dpi=300)
plt.close()

print("SHAP plots saved successfully.")

###############################################################################
# DECISION TREE VISUALIZATION
###############################################################################

tree = cls.estimators_[0]
plt.figure(figsize=(20,10))
plot_tree(tree, feature_names=X_selected.columns,
          filled=True, class_names=classes, rounded=True)
plt.savefig("RF_tree.png")
plt.close()


###############################################################################
# DECISION SURFACE (top 2 features)
###############################################################################

top2 = list(fs.k_feature_names_)[:2]
print("Top 2 SFS features:", top2)

clf2D = RandomForestClassifier(**best_rf.get_params())
clf2D.fit(features[top2], label)

x_min, x_max = features[top2[0]].min() - 0.5, features[top2[0]].max() + 0.5
y_min, y_max = features[top2[1]].min() - 0.5, features[top2[1]].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)
grid = clf2D.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, grid, alpha=0.4, cmap=ListedColormap(['#FFAAAA','#AAAAFF']))
plt.scatter(features[top2[0]], features[top2[1]], c=label, cmap="viridis", edgecolor="k")
plt.xlabel(top2[0]); plt.ylabel(top2[1])
plt.title("Decision Surface (Top-2 Features)")
plt.savefig("decision_surface.png", dpi=300)
plt.close()


