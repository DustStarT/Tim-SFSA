"""Revision figures built only from saved, reproducible experiment artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def create_revision_figures(run_dir: Path, seeds: list[int]) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []

    # A data-independent methods schematic keeps the sample definition explicit
    # in the revised manuscript without embedding any unverified case labels.
    figure, axis = plt.subplots(figsize=(12, 3.2))
    axis.set_xlim(-0.5, 18.5)
    axis.set_ylim(-1.0, 2.2)
    axis.axis("off")
    axis.plot([0, 18], [0, 0], color="0.25", linewidth=1.5)
    for position, label in ((0, "AR observation start\n(anchor 0)"), (6, "next M/X event\n(anchor 1)"),
                            (12, "next retained M/X event\n(anchor 2)"), (18, "actual observation end")):
        axis.plot([position, position], [-0.16, 0.16], color="0.15", linewidth=1.4)
        axis.text(position, -0.33, label, ha="center", va="top", fontsize=9)
    for start in (0, 6, 12):
        axis.add_patch(plt.Rectangle(
            (start, 0.38), 4, 0.45, facecolor="#92c5de", edgecolor="#2166ac"
        ))
        axis.text(start + 2, 0.605, "4 h / 20-step history", ha="center", va="center", fontsize=9)
        axis.annotate(
            "forecast origin", xy=(start + 4, 0.38), xytext=(start + 4, 1.18),
            ha="center", fontsize=8,
            arrowprops={"arrowstyle": "->", "color": "#2166ac"},
        )
    axis.text(2.0, 1.75, "event inside history: covered and skipped", ha="center", fontsize=9, color="#b2182b")
    axis.plot([2.2], [0.0], marker="*", markersize=11, color="#b2182b")
    axis.text(16.0, 1.75, "terminal interval: censored at observation end", ha="center", fontsize=9)
    figure.tight_layout()
    schematic_path = figure_dir / "event_anchored_sampling.png"
    figure.savefig(schematic_path, dpi=250, bbox_inches="tight")
    plt.close(figure)
    artifacts.append(schematic_path)

    calibration_frames = []
    importance_frames = []
    for seed in seeds:
        base = run_dir / "experiments" / "official" / "tim_sfsa" / f"seed_{seed}"
        calibration_path = base / "test_calibration.csv"
        if calibration_path.exists():
            frame = pd.read_csv(calibration_path)
            frame["seed"] = seed
            calibration_frames.append(frame)
        importance_path = base / "validation_ar_block_importance.csv"
        if importance_path.exists():
            frame = pd.read_csv(importance_path)
            frame["seed"] = seed
            importance_frames.append(frame)

    if calibration_frames:
        calibration = pd.concat(calibration_frames, ignore_index=True)
        selected = calibration[calibration["horizon_hours"].isin([12.0, 24.0, 48.0])]
        figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
        for axis, horizon in zip(axes, [12.0, 24.0, 48.0]):
            subset = selected[selected["horizon_hours"] == horizon]
            summary = subset.groupby("bin", as_index=False).agg(
                predicted=("predicted_event_probability", "mean"),
                observed=("observed_event_probability_ipcw", "mean"),
            )
            axis.plot([0, 1], [0, 1], "--", color="0.5", linewidth=1)
            axis.plot(summary["predicted"], summary["observed"], "o-", color="#2166ac")
            axis.set_title(f"{int(horizon)} h")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.set_xlabel("Predicted event probability")
        axes[0].set_ylabel("Observed event probability (IPCW)")
        figure.tight_layout()
        path = figure_dir / "calibration_12_24_48h.png"
        figure.savefig(path, dpi=250, bbox_inches="tight")
        plt.close(figure)
        artifacts.append(path)

    if importance_frames:
        importance = pd.concat(importance_frames, ignore_index=True)
        summary = importance.groupby("feature", as_index=False).agg(
            mean=("importance_mean", "mean"), std=("importance_mean", "std")
        ).sort_values("mean")
        figure, axis = plt.subplots(figsize=(7, 8))
        axis.barh(summary["feature"], summary["mean"], xerr=summary["std"].fillna(0), color="#4d9221")
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Decrease in validation Uno C-index after AR-block permutation")
        figure.tight_layout()
        path = figure_dir / "feature_importance.png"
        figure.savefig(path, dpi=250, bbox_inches="tight")
        plt.close(figure)
        artifacts.append(path)

    summary_path = run_dir / "aggregate" / "seed_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        subset = summary[
            (summary["split_mode"] == "official")
            & (summary["metric"].isin(["uno_c_index", "integrated_brier_score"]))
        ]
        if not subset.empty:
            pivot = subset.pivot(index="experiment", columns="metric", values="mean")
            pivot.to_csv(run_dir / "aggregate" / "primary_model_comparison.csv")
        primary = summary[
            (summary["split_mode"] == "official")
            & (summary["experiment"] == "tim_sfsa")
        ].set_index("metric")
        brier_rows = []
        for horizon in (12, 24, 48, 72, 144):
            model_key = f"{horizon}h_brier"
            reference_key = f"{horizon}h_km_reference_brier"
            bss_key = f"{horizon}h_bss"
            if model_key not in primary.index or reference_key not in primary.index:
                continue
            brier_rows.append({
                "horizon": horizon,
                "model": float(primary.loc[model_key, "mean"]),
                "km_reference": float(primary.loc[reference_key, "mean"]),
                "bss": float(primary.loc[bss_key, "mean"]) if bss_key in primary.index else np.nan,
            })
        if brier_rows:
            brier = pd.DataFrame(brier_rows)
            positions = np.arange(len(brier))
            figure, axis = plt.subplots(figsize=(8, 4.5))
            axis.bar(positions - 0.18, brier["model"], width=0.36, label="Tim-SFSA")
            axis.bar(positions + 0.18, brier["km_reference"], width=0.36, label="KM reference")
            axis.set_xticks(positions, [f"{value} h" for value in brier["horizon"]])
            axis.set_ylabel("IPCW Brier score (lower is better)")
            axis.legend(frameon=False)
            figure.tight_layout()
            path = figure_dir / "brier_vs_km_reference.png"
            figure.savefig(path, dpi=250, bbox_inches="tight")
            plt.close(figure)
            artifacts.append(path)
    return artifacts
