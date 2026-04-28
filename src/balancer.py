"""Orchestration helpers for training and running zone EQ recommendations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

try:
    from src.knn_model import predict_zone_eq, train_zone_knn
except ImportError:  # Lets tests monkeypatch these names before knn_model exists.
    predict_zone_eq = None
    train_zone_knn = None


REQUIRED_ACTUAL_COLUMNS = {"zone_id", "song_title", "artist", "bass", "treble", "clarity"}
REQUIRED_MODEL_COLUMNS = {"zone_id", "loudness", "energy", "ideal_bass", "ideal_treble"}


def _zone_items(zones: dict[int, Any] | Iterable[Any]) -> list[tuple[int, Any]]:
    """Return zones as sorted ``(zone_id, zone)`` pairs."""
    if isinstance(zones, dict):
        return sorted(zones.items())

    items = []
    for zone in zones:
        zone_id = getattr(zone, "zone_id", None)
        if zone_id is None:
            raise ValueError("Every zone object must have a zone_id attribute.")
        items.append((int(zone_id), zone))
    return sorted(items)


def _zone_name(zone_id: int, zone: Any) -> str:
    """Get a display name from a zone object with a stable fallback."""
    return getattr(zone, "zone_name", None) or getattr(zone, "name", None) or f"Zone {zone_id}"


def prepare_training_data(
    training_df: pd.DataFrame,
    spotify_client: Any | None = None,
    feature_cache: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Convert raw listening-test rows into model-ready training rows.

    The architecture document describes a model-ready CSV with loudness, energy,
    ideal_bass, and ideal_treble. The current project data instead stores many
    tried bass/treble settings per song with a clarity score. This helper picks
    the highest-clarity setting per zone/song/artist. Loudness and energy must
    already be present in the CSV because Spotify's Audio Features endpoint is
    deprecated for new apps.

    Args:
        training_df: Raw or model-ready training data.
        spotify_client: Deprecated compatibility parameter. It is not used.
        feature_cache: Deprecated compatibility parameter. It is not used.

    Returns:
        A DataFrame with at least zone_id, loudness, energy, ideal_bass, and
        ideal_treble columns.

    Raises:
        ValueError: If required columns are missing or Spotify features are
            missing from the CSV.
    """
    if REQUIRED_MODEL_COLUMNS.issubset(training_df.columns):
        return training_df.copy()

    missing = REQUIRED_ACTUAL_COLUMNS - set(training_df.columns)
    if missing:
        raise ValueError(f"Training data is missing required columns: {sorted(missing)}")

    best_rows = (
        training_df.sort_values("clarity", ascending=False)
        .drop_duplicates(subset=["zone_id", "song_title", "artist"], keep="first")
        .copy()
    )
    best_rows = best_rows.rename(columns={"bass": "ideal_bass", "treble": "ideal_treble"})

    if {"loudness", "energy"}.issubset(best_rows.columns):
        return best_rows

    raise ValueError(
        "Training data must include loudness and energy columns. Spotify search can identify tracks, "
        "but the Spotify Audio Features endpoint is deprecated for new apps."
    )


def train_all_zones(
    zones: dict[int, Any] | Iterable[Any],
    training_df: pd.DataFrame,
    n_neighbors: int = 3,
    spotify_client: Any | None = None,
) -> dict[int, Any]:
    """Train one KNN model per zone and attach it to each zone object.

    Args:
        zones: Mapping or iterable of SpeakerZone objects.
        training_df: Raw or model-ready training data.
        n_neighbors: K value for the KNN model.
        spotify_client: Optional SpotifyClient for converting raw CSV rows.

    Returns:
        A dictionary mapping zone IDs to trained KNN model objects.

    Raises:
        ImportError: If the KNN module has not implemented train_zone_knn yet.
    """
    if train_zone_knn is None:
        raise ImportError("src.knn_model.train_zone_knn must be implemented before training zones.")

    model_df = prepare_training_data(training_df, spotify_client=spotify_client)
    models = {}

    for zone_id, zone in _zone_items(zones):
        model = train_zone_knn(zone_id, model_df, n_neighbors=n_neighbors)
        zone.knn_model = model
        models[zone_id] = model

    return models


def run_balancer(zones: dict[int, Any] | Iterable[Any], loudness: float, energy: float) -> dict[int, dict[str, Any]]:
    """Predict bass and treble recommendations for every speaker zone.

    Args:
        zones: Mapping or iterable of trained SpeakerZone objects.
        loudness: Spotify integrated loudness in dB.
        energy: Spotify energy value from 0.0 to 1.0.

    Returns:
        A nested dictionary keyed by zone ID.

    Raises:
        ImportError: If the KNN module has not implemented predict_zone_eq yet.
    """
    if predict_zone_eq is None:
        raise ImportError("src.knn_model.predict_zone_eq must be implemented before balancing zones.")

    results = {}
    for zone_id, zone in _zone_items(zones):
        bass, treble = predict_zone_eq(zone, loudness, energy)
        results[zone_id] = {
            "zone_name": _zone_name(zone_id, zone),
            "bass": float(bass),
            "treble": float(treble),
        }

    return results


def format_recommendations(song_info: dict[str, Any], results: dict[int, dict[str, Any]]) -> str:
    """Format song features and per-zone EQ results for console output."""
    song_name = song_info.get("name", "Unknown Song")
    artist = song_info.get("artist", "Unknown Artist")
    loudness = song_info.get("loudness", "n/a")
    energy = song_info.get("energy", "n/a")

    lines = [
        f'Song: "{song_name}" by {artist}',
        f"Spotify Features - Loudness: {loudness} dB | Energy: {energy}",
        "-" * 58,
    ]

    for zone_id, rec in sorted(results.items()):
        zone_label = rec.get("zone_name", f"Zone {zone_id}")
        bass = rec.get("bass", 0.0)
        treble = rec.get("treble", 0.0)
        lines.append(f"Zone {zone_id} ({zone_label}): Bass = {bass:+.1f} dB | Treble = {treble:+.1f} dB")

    return "\n".join(lines)
