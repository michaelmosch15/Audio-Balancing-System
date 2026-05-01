"""Per-zone KNN training and prediction.

Two scikit-learn KNeighborsRegressor models are fit per zone — one for
bass and one for treble — each using a different subset of Spotify
audio features chosen for the part of the spectrum it predicts. See
the proposal (Section 1) for why bass and treble use different features.

Training data lives in ``data/training_data.csv`` as a swept grid of
bass/treble settings per (zone, song) annotated with a subjective
clarity score; ``load_training_data`` collapses each grid down to the
single highest-clarity row, which becomes the "ideal" label.
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

# Spotify features consumed by each model. Order is fixed so the matching
# numpy column order in the feature matrix is predictable.
# "acousticness_inverted" is synthetic: 1.0 - acousticness.
BASS_FEATURES = (
    "energy", "danceability", "tempo", "acousticness_inverted", "instrumentalness",
)
TREBLE_FEATURES = ("energy", "valence", "acousticness", "speechiness")

REQUIRED_TRAINING_COLUMNS = (
    "zone_id", "song_title", "artist", "bass", "treble", "clarity",
)


def load_training_data(filepath):
    """Read training_data.csv and reduce each (zone, song) sweep to its best row.

    The raw CSV records a grid of bass/treble combos with subjective
    clarity scores. For each (zone_id, song_title, artist) group we
    keep the single row with the highest clarity — that row's bass and
    treble are the user's ideal labels for the KNN to learn.
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
    return df.loc[keep, list(REQUIRED_TRAINING_COLUMNS)].reset_index(drop=True)


def _feature_vector(features, names):
    """Pull `names` out of `features` as a list of floats, in order.

    The synthetic "acousticness_inverted" key is derived from the raw
    "acousticness" value so callers only need to provide Spotify's
    actual feature names.
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


def train_zone_knn(zone_id, training_df, feature_lookup, n_neighbors=DEFAULT_K):
    """Fit and return (bass_model, treble_model) for one zone.

    feature_lookup is a callable (song_title, artist) -> dict-of-floats
    that resolves Spotify audio features for a song. We accept a
    callable instead of hard-wiring SpotifyClient so tests can inject
    a deterministic stub.
    """
    if not isinstance(training_df, pd.DataFrame):
        raise TypeError(f"training_df must be a pd.DataFrame, got {type(training_df).__name__}")
    if not callable(feature_lookup):
        raise TypeError("feature_lookup must be callable")
    if not isinstance(n_neighbors, int) or isinstance(n_neighbors, bool) or n_neighbors < 1:
        raise ValueError(f"n_neighbors must be a positive int, got {n_neighbors!r}")

    zone_rows = training_df[training_df["zone_id"] == zone_id]
    if len(zone_rows) < n_neighbors:
        raise ValueError(
            f"zone {zone_id} has only {len(zone_rows)} training sample(s); "
            f"need at least n_neighbors={n_neighbors}"
        )

    # Resolve Spotify features once per row, then split into the two
    # disjoint feature matrices the bass and treble models need.
    bass_rows, treble_rows = [], []
    bass_y, treble_y = [], []
    for i, (_, row) in enumerate(zone_rows.iterrows()):
        try:
            features = feature_lookup(str(row["song_title"]), str(row["artist"]))
        except Exception as exc:
            # Wrap so the caller knows which row failed without losing the cause.
            raise RuntimeError(
                f"feature_lookup failed on row {i} "
                f"({row['song_title']!r} by {row['artist']!r}): {exc}"
            ) from exc
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


def train_all_zones(zones, training_df, feature_lookup, n_neighbors=DEFAULT_K):
    """Train and attach (bass_model, treble_model) tuples to every zone.

    Also sanity-checks that every zone has labels in the training set —
    a missing zone usually means an off-by-one in the CSV.
    """
    # Set difference flags zones that have no training rows at all.
    labeled_zones = set(training_df["zone_id"].unique().tolist())
    requested = set(zones.keys())
    missing = requested - labeled_zones
    if missing:
        raise ValueError(f"no training data for zone(s) {sorted(missing)}")

    for zone_id, zone in zones.items():
        zone.knn_model = train_zone_knn(
            zone_id=zone_id,
            training_df=training_df,
            feature_lookup=feature_lookup,
            n_neighbors=n_neighbors,
        )


def predict_zone_eq(zone, spotify_features):
    """Predict (bass_gain, treble_gain) in dB for one zone and one song.

    Reads the (bass_model, treble_model) tuple from zone.knn_model,
    runs each regressor on its own feature vector, clips to the
    controller's [-10, +10] dB range, forces treble to 0 for the
    subwoofer zone, and writes the result back via zone.set_eq.
    """
    if zone.knn_model is None:
        raise RuntimeError(
            f"zone {getattr(zone, 'zone_id', '?')} has no trained KNN model; "
            "call train_zone_knn or train_all_zones first"
        )
    bass_model, treble_model = zone.knn_model

    x_bass = np.asarray([_feature_vector(spotify_features, BASS_FEATURES)], dtype=float)
    x_treble = np.asarray([_feature_vector(spotify_features, TREBLE_FEATURES)], dtype=float)

    bass = float(np.clip(bass_model.predict(x_bass)[0], -GAIN_LIMIT_DB, GAIN_LIMIT_DB))
    treble = float(np.clip(treble_model.predict(x_treble)[0], -GAIN_LIMIT_DB, GAIN_LIMIT_DB))

    # The subwoofer driver physically can't reproduce treble — force 0 regardless of model output.
    if zone.zone_id == SUBWOOFER_ZONE_ID:
        treble = 0.0

    zone.set_eq(bass, treble)
    return bass, treble
