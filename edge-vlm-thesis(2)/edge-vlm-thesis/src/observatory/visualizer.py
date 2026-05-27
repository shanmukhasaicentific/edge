"""
src/observatory/visualizer.py

Thesis-Quality Visualization Engine.

Generates 5 publication-ready figures:

    1. Master Semantic Timeline  — the most important figure
       Multi-track aligned visualization of all semantic signals

    2. VLM Invocation Heatmap
       Where and how densely VLM was triggered

    3. Semantic Evolution Curve (SPF)
       Cumulative semantic progress with plateau/burst annotation

    4. Object Persistence Graph
       Track count over time with stability overlay

    5. Semantic State Transition Map
       Color-coded frame-level state classification

All saved as high-DPI PNG (300dpi) suitable for thesis.
"""

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from src.observatory.analytics import ObservatoryResult


# ─── Style constants ──────────────────────────────────────────────────────────

COLORS = {
    "drift":        "#2E86AB",   # blue
    "acceleration": "#E84855",   # red
    "volatility":   "#F4A261",   # orange
    "objects":      "#2D6A4F",   # green
    "vlm":          "#9B2226",   # dark red
    "tier3":        "#AE2012",
    "tier2":        "#EE9B00",
    "tier1":        "#94D2BD",
    "cache":        "#CAF0F8",
    "stable":       "#52B788",
    "drifting":     "#FFB703",
    "transition":   "#E63946",
    "novel":        "#7B2D8B",
    "progress":     "#5E60CE",
    "entropy":      "#48CAE4",
}

STYLE = {
    "figure.dpi":       300,
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
}

plt.rcParams.update(STYLE)


class SemanticVisualizer:
    """
    Generates all thesis-quality observatory figures.

    Args:
        output_dir:  where to save PNG files
        downsample:  plot every Nth point (performance for large logs)
    """

    def __init__(self, output_dir: str, downsample: int = 1):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.downsample = downsample

    def generate_all(self, result: ObservatoryResult) -> dict:
        """Generate all figures. Returns {name: filepath}."""
        df = result.enriched_df
        if self.downsample > 1:
            df = df.iloc[::self.downsample].copy()

        paths = {}
        print("\n[Visualizer] Generating thesis figures...")

        paths["master_timeline"]   = self._plot_master_timeline(df, result)
        paths["vlm_heatmap"]       = self._plot_vlm_heatmap(df, result)
        paths["evolution_curve"]   = self._plot_evolution_curve(df, result)
        paths["object_persistence"]= self._plot_object_persistence(df, result)
        paths["state_map"]         = self._plot_state_transition_map(df, result)

        for name, path in paths.items():
            size = os.path.getsize(path) / 1024
            print(f"  {name:<25} → {os.path.basename(path)}  ({size:.0f} KB)")

        return paths

    # ── Figure 1: Master Semantic Timeline ────────────────────────────────────

    def _plot_master_timeline(
        self, df: pd.DataFrame, result: ObservatoryResult
    ) -> str:
        """
        THE most important figure. 8 synchronized tracks:
            1. Semantic Drift
            2. Semantic Acceleration
            3. Semantic Volatility
            4. Object Count Timeline
            5. Scene Entropy
            6. VLM Invocation Density
            7. Compute Tier Allocation
            8. Semantic State Labels
        """
        x = df["frame_id"].values
        n_tracks = 8
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(
            "Master Semantic Timeline\n"
            "Semantic Monitoring for Adaptive Compute Allocation",
            fontsize=13, fontweight="bold", y=0.98
        )

        gs = gridspec.GridSpec(
            n_tracks, 1,
            hspace=0.08,
            top=0.93, bottom=0.05,
            left=0.08, right=0.97
        )

        axes = [fig.add_subplot(gs[i]) for i in range(n_tracks)]

        # ── Track 1: Semantic Drift ──
        ax = axes[0]
        ax.fill_between(x, df["drift_total"], alpha=0.3, color=COLORS["drift"])
        ax.plot(x, df["drift_total"], color=COLORS["drift"], lw=0.8, label="D_t")
        if "stability_tau" in dir(result):
            tau_low, tau_high = 0.10, 0.35
        else:
            tau_low, tau_high = 0.10, 0.35
        ax.axhline(tau_low, color=COLORS["stable"], lw=1.0, ls="--", alpha=0.7, label=f"τ_low={tau_low}")
        ax.axhline(tau_high, color=COLORS["transition"], lw=1.0, ls="--", alpha=0.7, label=f"τ_high={tau_high}")
        ax.set_ylabel("Drift D_t", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.set_title("Semantic Drift  D_t = α·D_embed + β·D_objects + γ·D_track", fontsize=9, loc="left")
        self._shade_regions(ax, result, x)

        # ── Track 2: Semantic Acceleration ──
        ax = axes[1]
        accel = df["semantic_acceleration_smooth"].values
        ax.plot(x, accel, color=COLORS["acceleration"], lw=0.8, label="A_t (smooth)")
        ax.fill_between(x, accel, 0, where=(accel > 0), color=COLORS["acceleration"], alpha=0.2, label="rising")
        ax.fill_between(x, accel, 0, where=(accel < 0), color=COLORS["drift"], alpha=0.2, label="falling")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_ylabel("Accel A_t", fontsize=8)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.set_title("Semantic Acceleration  A_t = D_t − D_{t−1}", fontsize=9, loc="left")

        # ── Track 3: Semantic Volatility ──
        ax = axes[2]
        ax.fill_between(x, df["semantic_volatility"], alpha=0.4, color=COLORS["volatility"])
        ax.plot(x, df["semantic_volatility"], color=COLORS["volatility"], lw=0.8, label="V_t")
        ax.set_ylabel("Volatility V_t", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title(f"Semantic Volatility  V_t = Var(D_{{t-k:t}})  window={result.enriched_df.shape[0] // 100}", fontsize=9, loc="left")

        # ── Track 4: Object Count ──
        ax = axes[3]
        if "n_tracks" in df.columns:
            ax.fill_between(x, df["n_tracks"], alpha=0.4, color=COLORS["objects"])
            ax.plot(x, df["n_tracks"], color=COLORS["objects"], lw=0.8, label="tracks")
        if "n_detections" in df.columns:
            ax.plot(x, df["n_detections"], color="#52B788", lw=0.6, ls="--", alpha=0.7, label="detections")
        ax.set_ylabel("Object Count", fontsize=8)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.set_title("Object Timeline (ByteTrack active tracks + YOLO detections)", fontsize=9, loc="left")

        # ── Track 5: Scene Entropy ──
        ax = axes[4]
        if "scene_entropy" in df.columns:
            ax.fill_between(x, df["scene_entropy"], alpha=0.4, color=COLORS["entropy"])
            ax.plot(x, df["scene_entropy"], color=COLORS["entropy"], lw=0.8, label="entropy")
        ax.set_ylabel("Entropy H_t", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("Scene Entropy  H_t (Shannon)", fontsize=9, loc="left")

        # ── Track 6: VLM Invocation Density ──
        ax = axes[5]
        if "vlm_density" in df.columns:
            ax.fill_between(x, df["vlm_density"], alpha=0.5, color=COLORS["vlm"])
            ax.plot(x, df["vlm_density"], color=COLORS["vlm"], lw=0.8, label="VLM density")
        # Mark individual VLM calls as ticks
        if "vlm_called" in df.columns:
            vlm_frames = df[df["vlm_called"] == True]["frame_id"]
            ax.scatter(vlm_frames, [0.01] * len(vlm_frames),
                      marker="|", color=COLORS["tier3"], s=30, alpha=0.5, zorder=3)
        ax.set_ylabel("VLM Rate", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("VLM Invocation Density (rolling rate + individual calls ↑)", fontsize=9, loc="left")

        # ── Track 7: Compute Tier ──
        ax = axes[6]
        if "tier" in df.columns:
            tier_colors = {1: COLORS["tier1"], 2: COLORS["tier2"], 3: COLORS["tier3"]}
            for tier_val, color in tier_colors.items():
                mask = df["tier"] == tier_val
                ax.fill_between(
                    x, 0, tier_val,
                    where=mask.values,
                    color=color, alpha=0.6,
                    label=f"Tier {tier_val}"
                )
        ax.set_ylabel("Compute Tier", fontsize=8)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["T1\nCache", "T2\nLight", "T3\nFull"], fontsize=7)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.set_title("Compute Tier Allocation (T1=cache, T2=lightweight, T3=full VLM)", fontsize=9, loc="left")

        # ── Track 8: Semantic State ──
        ax = axes[7]
        state_color_map = {
            "stable": COLORS["stable"],
            "drifting": COLORS["drifting"],
            "transition": COLORS["transition"],
        }
        if "frame_semantic_state" in df.columns:
            for state, color in state_color_map.items():
                mask = df["frame_semantic_state"] == state
                ax.fill_between(x, 0, 1, where=mask.values, color=color, alpha=0.6, label=state)
        ax.set_ylabel("State", fontsize=8)
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.set_xlabel("Frame ID", fontsize=9)
        ax.set_title("Semantic State Classification (Stable / Drifting / Transition)", fontsize=9, loc="left")

        # ── Shared x-axis formatting ──
        for ax in axes[:-1]:
            ax.set_xticklabels([])
            ax.grid(True, alpha=0.2)
        axes[-1].grid(True, alpha=0.2)

        # Align all x-limits
        xlim = (x[0], x[-1])
        for ax in axes:
            ax.set_xlim(xlim)

        path = os.path.join(self.output_dir, "fig1_master_semantic_timeline.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    def _shade_regions(self, ax, result: ObservatoryResult, x: np.ndarray):
        """Shade transition regions on the drift track."""
        for region in result.transition_regions[:20]:   # limit to 20 for readability
            ax.axvspan(
                region.start_frame, region.end_frame,
                alpha=0.08, color=COLORS["transition"], zorder=0
            )

    # ── Figure 2: VLM Invocation Heatmap ─────────────────────────────────────

    def _plot_vlm_heatmap(
        self, df: pd.DataFrame, result: ObservatoryResult
    ) -> str:
        fig, axes = plt.subplots(3, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [2, 1, 1]})
        fig.suptitle("VLM Invocation Analysis\nWhere and Why Expensive Reasoning Was Triggered",
                     fontsize=12, fontweight="bold")

        x = df["frame_id"].values

        # Heatmap row: VLM density as color intensity
        ax = axes[0]
        if "vlm_density" in df.columns:
            density = df["vlm_density"].values.reshape(1, -1)
            im = ax.imshow(
                density,
                aspect="auto",
                extent=[x[0], x[-1], 0, 1],
                cmap="YlOrRd",
                vmin=0, vmax=density.max()
            )
            plt.colorbar(im, ax=ax, label="VLM Invocation Rate", shrink=0.8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title("VLM Trigger Density Heatmap (brighter = more frequent reasoning)", fontsize=9)
        ax.set_yticks([])

        # Drift context
        ax = axes[1]
        ax.plot(x, df["drift_total"], color=COLORS["drift"], lw=0.8, label="Drift D_t")
        if "vlm_called" in df.columns:
            vlm_x = df[df["vlm_called"] == True]["frame_id"]
            vlm_y = df[df["vlm_called"] == True]["drift_total"]
            ax.scatter(vlm_x, vlm_y, color=COLORS["vlm"], s=20, zorder=5,
                      label="VLM call", marker="^")
        ax.set_ylabel("Semantic Drift", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.2)

        # Tier breakdown
        ax = axes[2]
        if "tier" in df.columns:
            tier3 = (df["tier"] == 3).astype(float)
            tier2 = (df["tier"] == 2).astype(float)
            ax.fill_between(x, tier3, color=COLORS["tier3"], alpha=0.7, label="Tier 3 (full VLM)")
            ax.fill_between(x, tier2, color=COLORS["tier2"], alpha=0.5, label="Tier 2 (lightweight)")
        ax.set_ylabel("Tier Active", fontsize=8)
        ax.set_xlabel("Frame ID", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.2)

        for ax in axes:
            ax.set_xlim(x[0], x[-1])

        plt.tight_layout()
        path = os.path.join(self.output_dir, "fig2_vlm_invocation_heatmap.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── Figure 3: Semantic Evolution Curve ───────────────────────────────────

    def _plot_evolution_curve(
        self, df: pd.DataFrame, result: ObservatoryResult
    ) -> str:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle("Semantic Evolution Curve (Semantic Progress Function)\n"
                     "Cumulative Semantic Distance Traveled Over Time",
                     fontsize=12, fontweight="bold")

        x = df["frame_id"].values

        # SPF curve
        ax = axes[0]
        if "semantic_progress_norm" in df.columns:
            spf = df["semantic_progress_norm"].values
            ax.plot(x, spf, color=COLORS["progress"], lw=1.5, label="SPF (normalized)")
            ax.fill_between(x, spf, alpha=0.2, color=COLORS["progress"])

            # Annotate plateaus — regions where SPF slope is very flat
            slope = np.gradient(spf)
            plateau_mask = np.abs(slope) < np.percentile(np.abs(slope), 20)
            ax.fill_between(x, 0, 1, where=plateau_mask,
                           color=COLORS["stable"], alpha=0.15, label="Plateau (slow progress)")

            # Annotate bursts — regions where SPF slope is very steep
            burst_mask = np.abs(slope) > np.percentile(np.abs(slope), 85)
            ax.fill_between(x, 0, 1, where=burst_mask,
                           color=COLORS["transition"], alpha=0.15, label="Burst (rapid progress)")

        ax.set_ylabel("Cumulative Semantic Progress (normalized)", fontsize=9)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title("Semantic Progress Function — flat=plateau, steep=semantic burst", fontsize=9)
        ax.grid(True, alpha=0.2)

        # Drift rate (derivative of SPF = D_t)
        ax = axes[1]
        ax.fill_between(x, df["drift_total"], alpha=0.3, color=COLORS["drift"])
        ax.plot(x, df["drift_total"], color=COLORS["drift"], lw=0.8, label="Instantaneous drift D_t")

        # Rolling mean for pacing
        rolling_mean = df["drift_total"].rolling(50, min_periods=1).mean()
        ax.plot(x, rolling_mean, color=COLORS["volatility"], lw=1.5, ls="--", label="Rolling mean (pacing)")

        ax.set_ylabel("Semantic Drift Rate", fontsize=9)
        ax.set_xlabel("Frame ID", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)

        for ax in axes:
            ax.set_xlim(x[0], x[-1])

        plt.tight_layout()
        path = os.path.join(self.output_dir, "fig3_semantic_evolution_curve.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── Figure 4: Object Persistence ─────────────────────────────────────────

    def _plot_object_persistence(
        self, df: pd.DataFrame, result: ObservatoryResult
    ) -> str:
        fig, axes = plt.subplots(3, 1, figsize=(14, 9))
        fig.suptitle("Object Persistence & Scene Composition\n"
                     "Temporal Object Continuity Analysis",
                     fontsize=12, fontweight="bold")

        x = df["frame_id"].values

        # Track count
        ax = axes[0]
        if "n_tracks" in df.columns:
            ax.fill_between(x, df["n_tracks"], alpha=0.4, color=COLORS["objects"])
            ax.plot(x, df["n_tracks"], color=COLORS["objects"], lw=1.0, label="Active tracks")
            # Shade stable regions
            for region in result.stable_regions[:15]:
                ax.axvspan(region.start_frame, region.end_frame,
                          alpha=0.1, color=COLORS["stable"])
        ax.set_ylabel("Active Tracks", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Active Object Tracks Over Time (green shading = stable semantic regions)", fontsize=9)
        ax.grid(True, alpha=0.2)

        # Detection count
        ax = axes[1]
        if "n_detections" in df.columns:
            ax.fill_between(x, df["n_detections"], alpha=0.4, color="#52B788")
            ax.plot(x, df["n_detections"], color="#52B788", lw=1.0, label="YOLO detections")
            rolling = df["n_detections"].rolling(30, min_periods=1).mean()
            ax.plot(x, rolling, color="#1B4332", lw=1.5, ls="--", label="Rolling mean")
        ax.set_ylabel("Detections", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("YOLO Object Detection Count", fontsize=9)
        ax.grid(True, alpha=0.2)

        # Track-detection delta (birth/death events)
        ax = axes[2]
        if "n_tracks" in df.columns:
            delta = df["n_tracks"].diff().fillna(0)
            ax.bar(x, delta.clip(lower=0), color=COLORS["stable"], alpha=0.7,
                  width=1.0, label="Track births")
            ax.bar(x, delta.clip(upper=0), color=COLORS["transition"], alpha=0.7,
                  width=1.0, label="Track deaths")
            ax.axhline(0, color="gray", lw=0.5)
        ax.set_ylabel("Δ Tracks", fontsize=9)
        ax.set_xlabel("Frame ID", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Object Birth (+) and Death (−) Events", fontsize=9)
        ax.grid(True, alpha=0.2)

        for ax in axes:
            ax.set_xlim(x[0], x[-1])

        plt.tight_layout()
        path = os.path.join(self.output_dir, "fig4_object_persistence.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── Figure 5: Semantic State Transition Map ───────────────────────────────

    def _plot_state_transition_map(
        self, df: pd.DataFrame, result: ObservatoryResult
    ) -> str:
        fig, axes = plt.subplots(2, 1, figsize=(16, 6),
                                  gridspec_kw={"height_ratios": [1, 2]})
        fig.suptitle("Semantic State Transition Map\n"
                     "Frame-Level Classification: Stable / Drifting / Transition",
                     fontsize=12, fontweight="bold")

        x = df["frame_id"].values

        # State color strip
        ax = axes[0]
        state_color_map = {
            "stable": COLORS["stable"],
            "drifting": COLORS["drifting"],
            "transition": COLORS["transition"],
        }
        if "frame_semantic_state" in df.columns:
            state_numeric = df["frame_semantic_state"].map(
                {"stable": 0, "drifting": 1, "transition": 2}
            ).fillna(1).values.reshape(1, -1)

            from matplotlib.colors import ListedColormap
            cmap = ListedColormap([COLORS["stable"], COLORS["drifting"], COLORS["transition"]])
            ax.imshow(state_numeric, aspect="auto",
                     extent=[x[0], x[-1], 0, 1],
                     cmap=cmap, vmin=0, vmax=2)

            # Legend patches
            patches = [
                mpatches.Patch(color=COLORS["stable"], label="Stable"),
                mpatches.Patch(color=COLORS["drifting"], label="Drifting"),
                mpatches.Patch(color=COLORS["transition"], label="Transition"),
            ]
            ax.legend(handles=patches, loc="upper right", fontsize=8, ncol=3)
        ax.set_yticks([])
        ax.set_title("Semantic State per Frame", fontsize=9)
        ax.set_xticklabels([])

        # Drift with state-colored fill
        ax = axes[1]
        drift = df["drift_total"].values
        ax.plot(x, drift, color="gray", lw=0.5, alpha=0.5)

        if "frame_semantic_state" in df.columns:
            for state, color in state_color_map.items():
                mask = df["frame_semantic_state"].values == state
                ax.fill_between(x, 0, drift, where=mask, color=color, alpha=0.5, label=state)

        # Region boundaries
        all_regions = (result.stable_regions + result.transition_regions + result.volatile_regions)
        all_regions.sort(key=lambda r: r.start_frame)
        for region in all_regions:
            ax.axvline(region.start_frame, color="white", lw=0.3, alpha=0.3)

        ax.set_ylabel("Semantic Drift D_t", fontsize=9)
        ax.set_xlabel("Frame ID", fontsize=9)
        ax.legend(loc="upper right", fontsize=8, ncol=3)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(x[0], x[-1])

        # Region count annotations
        total = result.total_frames
        stable_pct = sum(r.duration_frames for r in result.stable_regions) / total * 100
        trans_pct = sum(r.duration_frames for r in result.transition_regions) / total * 100
        ax.text(0.02, 0.95,
                f"Stable: {stable_pct:.1f}%  |  Transition: {trans_pct:.1f}%  |  "
                f"Other: {100-stable_pct-trans_pct:.1f}%",
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        plt.tight_layout()
        path = os.path.join(self.output_dir, "fig5_semantic_state_transition_map.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path
