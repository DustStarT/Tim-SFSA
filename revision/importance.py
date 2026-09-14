"""Active-region-block feature importance and correlation diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import uno_c_index


def _permute_features_by_group(
    x: np.ndarray, groups: np.ndarray, feature_indices: list[int], rng: np.random.Generator
) -> np.ndarray:
    permuted = np.asarray(x).copy()
    group_values = np.unique(groups.astype(str))
    donors = rng.permutation(group_values)
    index_map = {group: np.where(groups.astype(str) == group)[0] for group in group_values}
    for recipient, donor in zip(group_values, donors):
        recipient_indices = index_map[recipient]
        donor_indices = index_map[donor]
        for offset, row_index in enumerate(recipient_indices):
            donor_index = donor_indices[offset % len(donor_indices)]
            permuted[row_index, :, feature_indices] = x[donor_index, :, feature_indices]
    return permuted


def ar_block_permutation_importance(
    predict_risk,
    x: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    groups: np.ndarray,
    train_duration: np.ndarray,
    train_event: np.ndarray,
    feature_names: list[str],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    base_risk = predict_risk(x)
    baseline = uno_c_index(train_duration, train_event, duration, event, base_risk)
    rng = np.random.default_rng(seed)
    rows = []
    for feature_index, feature_name in enumerate(feature_names):
        changes = []
        for _ in range(repeats):
            permuted = _permute_features_by_group(x, groups, [feature_index], rng)
            score = uno_c_index(
                train_duration, train_event, duration, event, predict_risk(permuted)
            )
            changes.append(baseline - score)
        rows.append({
            "feature": feature_name,
            "baseline_uno_c_index": baseline,
            "importance_mean": float(np.mean(changes)),
            "importance_std": float(np.std(changes, ddof=1)) if len(changes) > 1 else 0.0,
            "importance_ci_lower": float(np.quantile(changes, 0.025)),
            "importance_ci_upper": float(np.quantile(changes, 0.975)),
            "repeats": repeats,
        })
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def correlated_group_permutation_importance(
    predict_risk,
    x: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    groups: np.ndarray,
    train_duration: np.ndarray,
    train_event: np.ndarray,
    feature_names: list[str],
    correlated_groups: list[list[str]],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    base_risk = predict_risk(x)
    baseline = uno_c_index(train_duration, train_event, duration, event, base_risk)
    rng = np.random.default_rng(seed)
    rows = []
    for cluster_index, cluster in enumerate(correlated_groups):
        feature_indices = [feature_names.index(name) for name in cluster]
        changes = []
        for _ in range(repeats):
            permuted = _permute_features_by_group(x, groups, feature_indices, rng)
            score = uno_c_index(
                train_duration, train_event, duration, event, predict_risk(permuted)
            )
            changes.append(baseline - score)
        rows.append({
            "group": cluster_index + 1,
            "features": ";".join(cluster),
            "feature_count": len(cluster),
            "baseline_uno_c_index": baseline,
            "importance_mean": float(np.mean(changes)),
            "importance_std": float(np.std(changes, ddof=1)) if len(changes) > 1 else 0.0,
            "importance_ci_lower": float(np.quantile(changes, 0.025)),
            "importance_ci_upper": float(np.quantile(changes, 0.975)),
        })
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False) if rows else pd.DataFrame()


def correlation_diagnostics(
    x_train: np.ndarray,
    feature_names: list[str],
    legacy_dropped: list[str],
    threshold: float,
) -> tuple[pd.DataFrame, list[list[str]]]:
    last = pd.DataFrame(np.asarray(x_train)[:, -1, :], columns=feature_names)
    pearson = last.corr(method="pearson")
    spearman = last.corr(method="spearman")
    rows = []
    retained_features = [
        name for name in feature_names if name not in set(legacy_dropped)
    ]
    for dropped in legacy_dropped:
        if dropped not in feature_names:
            continue
        ranked = sorted(
            retained_features,
            key=lambda name: (
                -abs(float(pearson.loc[dropped, name]))
                if np.isfinite(pearson.loc[dropped, name]) else float("inf")
            ),
        )
        for rank, retained in enumerate(ranked, start=1):
            rows.append({
                "formerly_dropped_feature": dropped,
                "retained_feature": retained,
                "absolute_pearson_rank": rank,
                "strongest_retained_pair": rank == 1,
                "pearson": float(pearson.loc[dropped, retained]),
                "spearman": float(spearman.loc[dropped, retained]),
                "old_removal_reason": "absolute Pearson correlation > 0.98 in the legacy pipeline",
                "physical_importance_inference": "none; redundancy is not lack of physical relevance",
            })

    adjacency = {name: set() for name in feature_names}
    for left_index, left in enumerate(feature_names):
        for right in feature_names[left_index + 1:]:
            if abs(float(spearman.loc[left, right])) >= threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)
    clusters = []
    unvisited = set(feature_names)
    while unvisited:
        root = unvisited.pop()
        component, stack = {root}, [root]
        while stack:
            current = stack.pop()
            neighbours = adjacency[current] & unvisited
            component.update(neighbours)
            unvisited.difference_update(neighbours)
            stack.extend(neighbours)
        if len(component) > 1:
            clusters.append(sorted(component))
    return pd.DataFrame(rows), sorted(clusters, key=lambda value: (-len(value), value))
