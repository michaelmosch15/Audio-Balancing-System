"""Per-zone KNN training and prediction.

Two scikit-learn KNeighborsRegressor models are fit per zone: one for
bass and one for treble. Training data supplies the user's preferred
bass/treble labels, while song features come from data/song_features.csv
and are merged into the training rows before model fitting.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


# Hardware limit of the digital audio controller (dB). Predictions are
# clipped into [-GAIN_LIMIT_DB, +GAIN_LIMIT_DB] before being returned.
GAIN_LIMIT_DB = 10.0

# Default k for the regressors. Three is a sensible starting point given
# the modest size of the hand-labeled training set.
DEFAULT_K = 3

# Zone 7 is the subwoofer; treble predictions for it are forced to 0.
SUBWOOFER_ZONE_ID = 7

# Song features consumed by each model. Order is fixed so the matching
# numpy column order in the feature matrix is predictable.
# "acousticness_inverted" is synthetic: 1.0 - acousticness.
BASS_FEATURES = (
    "energy", "danceability", "tempo", "acousticness_inverted", "instrumentalness",
)
TREBLE_FEATURES = ("energy", "valence", "acousticness", "speechiness")
SONG_FEATURE_COLUMNS = (
    "energy", "danceability", "tempo", "acousticness",
    "instrumentalness", "valence", "speechiness",
)

REQUIRED_TRAINING_COLUMNS = (
    "zone_id", "song_title", "artist", "bass", "treble", "clarity",
)


def load_training_data(filepath):
    """Read training_data.csv and reduce each (zone, song) sweep to its best row.

    The raw CSV records a grid of bass/treble combos with subjective
    clarity scores. For each (zone_id, song_title, artist) group we
    keep the single row with the highest clarity. If song feature
    columns are already present, they are preserved for training.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_TRAINING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"training CSV is missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"training CSV contains no rows: {path}")

    # idxmax breaks ties by first occurrence, which is deterministic given CSV order.
    keep = df.groupby(["zone_id", "song_title", "artist"], sort=False)["clarity"].idxmax()
    columns = list(REQUIRED_TRAINING_COLUMNS)
    columns.extend(c for c in SONG_FEATURE_COLUMNS if c in df.columns)
    return df.loc[keep, columns].reset_index(drop=True)


def validate_training_features(training_df):
    """Ensure merged training data has usable numeric song feature columns."""
    missing = [c for c in SONG_FEATURE_COLUMNS if c not in training_df.columns]
    if missing:
        raise ValueError(f"training data is missing song feature columns: {missing}")

    invalid = []
    for col in SONG_FEATURE_COLUMNS:
        values = pd.to_numeric(training_df[col], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            invalid.append(col)

    if invalid:
        raise ValueError(f"training data has blank or invalid song feature values in: {invalid}")


def _feature_vector(features, names):
    """Pull `names` out of `features` as a list of floats, in order.

    The synthetic "acousticness_inverted" key is derived from the raw
    "acousticness" value so callers only need to provide actual feature
    names from song_features.csv.
    """
    if not hasattr(features, "__getitem__"):
        raise TypeError(f"features must be a mapping, got {type(features).__name__}")

    vec = []
    for name in names:
        if name == "acousticness_inverted":
            if "acousticness" not in features:
                raise ValueError("features missing required key 'acousticness' "
                                 "(needed to derive 'acousticness_inverted')")
            value = 1.0 - float(features["acousticness"])
        else:
            if name not in features:
                raise ValueError(f"features missing required key {name!r}")
            value = float(features[name])

        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"feature {name!r} must be finite, got {value!r}")
        vec.append(value)
    return vec


def train_zone_knn(zone_id, training_df, n_neighbors=DEFAULT_K):
    """Fit and return (bass_model, treble_model) for one zone.

    training_df must already contain song feature columns from
    data/song_features.csv, merged by (song_title, artist).
    """
    if not isinstance(training_df, pd.DataFrame):
        raise TypeError(f"training_df must be a pd.DataFrame, got {type(training_df).__name__}")
    if not isinstance(n_neighbors, int) or isinstance(n_neighbors, bool) or n_neighbors < 1:
        raise ValueError(f"n_neighbors must be a positive int, got {n_neighbors!r}")
    validate_training_features(training_df)

    zone_rows = training_df[training_df["zone_id"] == zone_id]
    if len(zone_rows) < n_neighbors:
        raise ValueError(
            f"zone {zone_id} has only {len(zone_rows)} training sample(s); "
            f"need at least n_neighbors={n_neighbors}"
        )

    bass_rows, treble_rows = [], []
    bass_y, treble_y = [], []
    for _, row in zone_rows.iterrows():
        features = row.to_dict()
        bass_rows.append(_feature_vector(features, BASS_FEATURES))
        treble_rows.append(_feature_vector(features, TREBLE_FEATURES))
        bass_y.append(float(row["bass"]))
        treble_y.append(float(row["treble"]))

    bass_model = KNeighborsRegressor(n_neighbors=n_neighbors).fit(
        np.asarray(bass_rows, dtype=float), np.asarray(bass_y, dtype=float)
    )
    treble_model = KNeighborsRegressor(n_neighbors=n_neighbors).fit(
        np.asarray(treble_rows, dtype=float), np.asarray(treble_y, dtype=float)
    )
    return bass_model, treble_model


def train_all_zones(zones, training_df, n_neighbors=DEFAULT_K):
    """Train and attach (bass_model, treble_model) tuples to every zone."""
    validate_training_features(training_df)
    labeled_zones = set(training_df["zone_id"].unique().tolist())
    requested = set(zones.keys())
    missing = requested - labeled_zones
    if missing:
        raise ValueError(f"no training data for zone(s) {sorted(missing)}")

    for zone_id, zone in zones.items():
        zone.knn_model = train_zone_knn(
            zone_id=zone_id,
            training_df=training_df,
            n_neighbors=n_neighbors,
        )


def predict_zone_eq(zone, song_features):
    """Predict (bass_gain, treble_gain) in dB for one zone and one song."""
    if zone.knn_model is None:
        raise RuntimeError(
            f"zone {getattr(zone, 'zone_id', '?')} has no trained KNN model; "
            "call train_zone_knn or train_all_zones first"
        )
    bass_model, treble_model = zone.knn_model

    x_bass = np.asarray([_feature_vector(song_features, BASS_FEATURES)], dtype=float)
    x_treble = np.asarray([_feature_vector(song_features, TREBLE_FEATURES)], dtype=float)

    bass = float(np.clip(bass_model.predict(x_bass)[0], -GAIN_LIMIT_DB, GAIN_LIMIT_DB))
    treble = float(np.clip(treble_model.predict(x_treble)[0], -GAIN_LIMIT_DB, GAIN_LIMIT_DB))

    # The subwoofer driver physically can't reproduce treble; force 0 regardless of model output.
    if zone.zone_id == SUBWOOFER_ZONE_ID:
        treble = 0.0

    zone.set_eq(bass, treble)
    return bass, treble
