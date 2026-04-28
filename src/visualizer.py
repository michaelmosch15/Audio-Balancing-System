"""Matplotlib visualizations for EQ recommendations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def visualize_eq_recommendations(
    results: dict[int, dict[str, Any]],
    song_name: str,
    show: bool = True,
    save_path: str | Path | None = None,
):
    """Create a grouped bar chart of bass and treble recommendations.

    Args:
        results: Recommendation dictionary returned by run_balancer().
        song_name: Song title to include in the chart title.
        show: Whether to display the chart window.
        save_path: Optional path where the chart image should be saved.

    Returns:
        The Matplotlib Figure object.
    """
    sorted_items = sorted(results.items())
    labels = [f"Zone {zone_id}\n{data.get('zone_name', '')}".strip() for zone_id, data in sorted_items]
    bass_values = [data.get("bass", 0.0) for _, data in sorted_items]
    treble_values = [data.get("treble", 0.0) for _, data in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 6))

    if sorted_items:
        x_positions = list(range(len(sorted_items)))
        width = 0.36
        bass_positions = [position - width / 2 for position in x_positions]
        treble_positions = [position + width / 2 for position in x_positions]

        ax.bar(bass_positions, bass_values, width, label="Bass", color="#2E86AB")
        ax.bar(treble_positions, treble_values, width, label="Treble", color="#F18F01")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No recommendations available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])

    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_ylim(-10, 10)
    ax.set_ylabel("Gain (dB)")
    ax.set_title(f"EQ Recommendations for {song_name}")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig
