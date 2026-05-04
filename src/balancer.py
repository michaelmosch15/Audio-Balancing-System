"""Orchestration helpers for training and running zone EQ recommendations."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from src.knn_model import SONG_FEATURE_COLUMNS, predict_zone_eq, train_zone_knn
except ImportError:  # Lets tests monkeypatch these names before knn_model exists.
    SONG_FEATURE_COLUMNS = (
        "energy", "danceability", "tempo", "acousticness",
        "instrumentalness", "valence", "speechiness",
    )
    predict_zone_eq = None
    train_zone_knn = None


SONG_KEY_COLUMNS = ("song_title", "artist")
REQUIRED_ACTUAL_COLUMNS = {"zone_id", "song_title", "artist", "bass", "treble", "clarity"}


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


def _normalized_text(value: Any) -> str:
    return str(value).strip().casefold()


def _song_key(row: pd.Series | dict[str, Any]) -> tuple[str, str]:
    return (_normalized_text(row["song_title"]), _normalized_text(row["artist"]))


def validate_song_features(song_features_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a normalized copy of song feature rows."""
    if not isinstance(song_features_df, pd.DataFrame):
        raise TypeError(
            f"song_features_df must be a pd.DataFrame, got {type(song_features_df).__name__}"
        )
    if song_features_df.empty:
        raise ValueError("song features data contains no rows")

    required = list(SONG_KEY_COLUMNS) + list(SONG_FEATURE_COLUMNS)
    missing = [c for c in required if c not in song_features_df.columns]
    if missing:
        raise ValueError(f"song features data is missing required columns: {missing}")

    features = song_features_df.loc[:, required].copy()
    for col in SONG_KEY_COLUMNS:
        if features[col].isna().any() or features[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"song features data has blank values in column {col!r}")
        features[col] = features[col].astype(str).str.strip()

    normalized_keys = features.apply(_song_key, axis=1)
    if normalized_keys.duplicated().any():
        duplicates = sorted(set(normalized_keys[normalized_keys.duplicated()].tolist()))
        raise ValueError(f"song features data has duplicate song rows: {duplicates}")

    invalid = []
    for col in SONG_FEATURE_COLUMNS:
        values = pd.to_numeric(features[col], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            invalid.append(col)
        else:
            features[col] = values.astype(float)

    if invalid:
        raise ValueError(f"song features data has blank or invalid values in: {invalid}")

    return features


def load_song_features(filepath: str | Path) -> pd.DataFrame:
    """Load and validate data/song_features.csv."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Song features CSV not found: {path}")
    return validate_song_features(pd.read_csv(path))


def prepare_training_data(
    training_df: pd.DataFrame,
    song_features_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Collapse tuning rows and merge song features for KNN training."""
    if not isinstance(training_df, pd.DataFrame):
        raise TypeError(f"training_df must be a pd.DataFrame, got {type(training_df).__name__}")

    missing = REQUIRED_ACTUAL_COLUMNS - set(training_df.columns)
    if missing:
        raise ValueError(f"Training data is missing required columns: {sorted(missing)}")
    if training_df.empty:
        raise ValueError("Training data contains no rows")

    best_rows = (
        training_df.sort_values("clarity", ascending=False)
        .drop_duplicates(subset=list(SONG_KEY_COLUMNS) + ["zone_id"], keep="first")
        .copy()
    )

    has_all_features = all(col in best_rows.columns for col in SONG_FEATURE_COLUMNS)
    if has_all_features and song_features_df is None:
        return validate_training_feature_rows(best_rows)

    if song_features_df is None:
        raise ValueError("Training data must be merged with song_features.csv before KNN training")

    # training_data.csv may contain stale or blank feature columns; song_features.csv is authoritative.
    best_rows = best_rows.drop(columns=[c for c in SONG_FEATURE_COLUMNS if c in best_rows.columns])
    features = validate_song_features(song_features_df)
    merged = best_rows.merge(features, on=list(SONG_KEY_COLUMNS), how="left", indicator=True)
    missing_rows = merged[merged["_merge"] == "left_only"]
    if not missing_rows.empty:
        songs = sorted(
            f"{row.song_title} by {row.artist}"
            for row in missing_rows.loc[:, list(SONG_KEY_COLUMNS)].drop_duplicates().itertuples(index=False)
        )
        raise ValueError(f"song_features.csv is missing feature rows for: {songs}")

    merged = merged.drop(columns=["_merge"])
    return validate_training_feature_rows(merged)


def validate_training_feature_rows(training_df: pd.DataFrame) -> pd.DataFrame:
    """Validate feature columns on merged training rows."""
    invalid = []
    for col in SONG_FEATURE_COLUMNS:
        if col not in training_df.columns:
            invalid.append(col)
            continue
        values = pd.to_numeric(training_df[col], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            invalid.append(col)
        else:
            training_df[col] = values.astype(float)

    if invalid:
        raise ValueError(f"Training data has missing or invalid song feature values in: {invalid}")
    return training_df.reset_index(drop=True)


def lookup_song_features(
    song_features_df: pd.DataFrame,
    song_title: str,
    artist: str | None = None,
) -> dict[str, float]:
    """Return the feature dictionary for a song, using case-insensitive matching."""
    features = validate_song_features(song_features_df)
    title_key = _normalized_text(song_title)
    matches = features[features["song_title"].map(_normalized_text) == title_key]

    if artist:
        artist_key = _normalized_text(artist)
        matches = matches[matches["artist"].map(_normalized_text) == artist_key]

    if matches.empty:
        label = f"{song_title} by {artist}" if artist else song_title
        raise ValueError(f"No song features found for {label}. Add it to data/song_features.csv.")
    if len(matches) > 1:
        raise ValueError(f"Multiple feature rows found for {song_title}; specify the artist.")

    row = matches.iloc[0]
    return {col: float(row[col]) for col in SONG_FEATURE_COLUMNS}


def train_all_zones(
    zones: dict[int, Any] | Iterable[Any],
    training_df: pd.DataFrame,
    song_features_df: pd.DataFrame | None = None,
    n_neighbors: int = 3,
) -> dict[int, Any]:
    """Train one KNN model per zone and attach it to each zone object."""
    if train_zone_knn is None:
        raise ImportError("src.knn_model.train_zone_knn must be implemented before training zones.")

    model_df = prepare_training_data(training_df, song_features_df=song_features_df)
    models = {}

    for zone_id, zone in _zone_items(zones):
        model = train_zone_knn(zone_id, model_df, n_neighbors=n_neighbors)
        zone.knn_model = model
        models[zone_id] = model

    return models


def run_balancer(zones: dict[int, Any] | Iterable[Any], song_features: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Predict bass and treble recommendations for every speaker zone."""
    if predict_zone_eq is None:
        raise ImportError("src.knn_model.predict_zone_eq must be implemented before balancing zones.")

    results = {}
    for zone_id, zone in _zone_items(zones):
        bass, treble = predict_zone_eq(zone, song_features)
        results[zone_id] = {
            "zone_name": _zone_name(zone_id, zone),
            "bass": float(bass),
            "treble": float(treble),
        }

    return results


def format_recommendations(song_info: dict[str, Any], results: dict[int, dict[str, Any]]) -> str:
    """Format song metadata and per-zone EQ results for console output."""
    song_name = song_info.get("name", "Unknown Song")
    artist = song_info.get("artist", "Unknown Artist")

    lines = [
        f'Song: "{song_name}" by {artist}',
        "Song features: loaded from data/song_features.csv",
        "-" * 58,
    ]

    for zone_id, rec in sorted(results.items()):
        zone_label = rec.get("zone_name", f"Zone {zone_id}")
        bass = rec.get("bass", 0.0)
        treble = rec.get("treble", 0.0)
        lines.append(f"Zone {zone_id} ({zone_label}): Bass = {bass:+.1f} dB | Treble = {treble:+.1f} dB")

    return "\n".join(lines)

