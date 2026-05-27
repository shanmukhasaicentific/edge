"""
src/observatory/dashboard.py

Interactive Plotly Dashboard.

Generates a single self-contained HTML file with:
    - Interactive semantic timeline (zoom, pan, hover)
    - VLM trigger explorer
    - Semantic region browser
    - Drift distribution histogram

Open in any browser — no server needed.
"""

import os
from typing import Optional

import pandas as pd
import numpy as np

from src.observatory.analytics import ObservatoryResult


class ObservatoryDashboard:
    """
    Builds an interactive HTML dashboard from observatory results.

    Args:
        output_dir: where to save the HTML file
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def build(self, result: ObservatoryResult, filename: str = "semantic_dashboard.html") -> str:
        """
        Build and save the interactive dashboard.

        Args:
            result:   ObservatoryResult from analytics engine
            filename: output HTML filename

        Returns:
            Path to saved HTML file
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("[Dashboard] plotly not installed. Run: pip install plotly")
            print("[Dashboard] Skipping interactive dashboard generation.")
            return ""

        df = result.enriched_df

        # Downsample for browser performance (plotly gets slow above 50k points)
        max_points = 10000
        if len(df) > max_points:
            step = len(df) // max_points
            df_plot = df.iloc[::step].copy()
            print(f"[Dashboard] Downsampled to {len(df_plot)} points for browser performance.")
        else:
            df_plot = df.copy()

        x = df_plot["frame_id"].values

        fig = make_subplots(
            rows=6, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "Semantic Drift D_t",
                "Semantic Acceleration & Volatility",
                "Object Count (Tracks + Detections)",
                "VLM Invocation Density",
                "Compute Tier Allocation",
                "Semantic Progress Function (SPF)",
            ],
            vertical_spacing=0.04,
            row_heights=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15],
        )

        # ── Row 1: Drift ──
        fig.add_trace(go.Scatter(
            x=x, y=df_plot["drift_total"],
            mode="lines", name="Drift D_t",
            line=dict(color="#2E86AB", width=1),
            fill="tozeroy", fillcolor="rgba(46,134,171,0.15)",
        ), row=1, col=1)

        # Threshold lines
        fig.add_hline(y=0.10, line_dash="dash", line_color="#52B788",
                     annotation_text="τ_low=0.10", row=1, col=1)
        fig.add_hline(y=0.35, line_dash="dash", line_color="#E63946",
                     annotation_text="τ_high=0.35", row=1, col=1)

        # VLM trigger markers on drift
        if "vlm_called" in df_plot.columns:
            vlm_mask = df_plot["vlm_called"] == True
            if vlm_mask.any():
                fig.add_trace(go.Scatter(
                    x=df_plot.loc[vlm_mask, "frame_id"],
                    y=df_plot.loc[vlm_mask, "drift_total"],
                    mode="markers", name="VLM Triggered",
                    marker=dict(color="#9B2226", size=6, symbol="triangle-up"),
                ), row=1, col=1)

        # ── Row 2: Acceleration + Volatility ──
        if "semantic_acceleration_smooth" in df_plot.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df_plot["semantic_acceleration_smooth"],
                mode="lines", name="Acceleration A_t",
                line=dict(color="#E84855", width=1),
            ), row=2, col=1)

        if "semantic_volatility" in df_plot.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df_plot["semantic_volatility"],
                mode="lines", name="Volatility V_t",
                line=dict(color="#F4A261", width=1),
            ), row=2, col=1)

        # ── Row 3: Object count ──
        if "n_tracks" in df_plot.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df_plot["n_tracks"],
                mode="lines", name="Active Tracks",
                line=dict(color="#2D6A4F", width=1),
                fill="tozeroy", fillcolor="rgba(45,106,79,0.2)",
            ), row=3, col=1)

        if "n_detections" in df_plot.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df_plot["n_detections"],
                mode="lines", name="Detections",
                line=dict(color="#52B788", width=1, dash="dot"),
            ), row=3, col=1)

        # ── Row 4: VLM density ──
        if "vlm_density" in df_plot.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df_plot["vlm_density"],
                mode="lines", name="VLM Density",
                line=dict(color="#9B2226", width=1),
                fill="tozeroy", fillcolor="rgba(155,34,38,0.3)",
            ), row=4, col=1)

        # ── Row 5: Compute Tier ──
        if "tier" in df_plot.columns:
            tier_colors = {1: "rgba(148,210,189,0.6)", 2: "rgba(238,155,0,0.6)", 3: "rgba(174,32,18,0.6)"}
            for tier_val, color in tier_colors.items():
                mask = df_plot["tier"] == tier_val
                fig.add_trace(go.Scatter(
                    x=df_plot.loc[mask, "frame_id"],
                    y=[tier_val] * mask.sum(),
                    mode="markers", name=f"Tier {tier_val}",
                    marker=dict(color=color, size=3, symbol="square"),
                ), row=5, col=1)

        # ── Row 6: SPF ──
        if "semantic_progress_norm" in df_plot.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df_plot["semantic_progress_norm"],
                mode="lines", name="SPF (normalized)",
                line=dict(color="#5E60CE", width=1.5),
                fill="tozeroy", fillcolor="rgba(94,96,206,0.15)",
            ), row=6, col=1)

        # ── Layout ──
        fig.update_layout(
            title=dict(
                text="<b>Semantic Evolution Observatory Dashboard</b><br>"
                     "<sup>Semantic Monitoring for Adaptive Compute Allocation — Interactive Analysis</sup>",
                font=dict(size=16),
            ),
            height=900,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            template="plotly_white",
            font=dict(family="Arial", size=11),
        )

        fig.update_xaxes(title_text="Frame ID", row=6, col=1)

        # Add summary annotation
        summary_text = (
            f"Frames: {result.total_frames:,}  |  "
            f"VLM Rate: {result.vlm_call_rate:.1%}  |  "
            f"Mean Drift: {result.mean_drift:.3f}  |  "
            f"Stable Regions: {len(result.stable_regions)}  |  "
            f"Transition Regions: {len(result.transition_regions)}"
        )
        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.5, y=-0.03, showarrow=False,
            font=dict(size=10, color="gray"),
            xanchor="center",
        )

        # Save
        path = os.path.join(self.output_dir, filename)
        fig.write_html(path, include_plotlyjs="cdn")
        size_kb = os.path.getsize(path) / 1024
        print(f"[Dashboard] Interactive dashboard: {path} ({size_kb:.0f} KB)")
        return path
