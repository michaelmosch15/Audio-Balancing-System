"""Tests for the data + ML pipeline (Michael M's components).

Covers the speaker.csv -> Speaker/Subwoofer -> SpeakerZone -> KNN
train -> predict chain using song feature columns from song_features.csv.
"""

from pathlib import Path

import pytest

from src.balancer import load_song_features, prepare_training_data
from src.knn_model import (
    BASS_FEATURES, TREBLE_FEATURES, SONG_FEATURE_COLUMNS,
    load_training_data, predict_zone_eq, train_all_zones, train_zone_knn,
)
from src.speaker import Speaker, Subwoofer, load_speakers_from_csv
from src.zone import SpeakerZone, build_zones


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEAKERS_CSV = REPO_ROOT / "data" / "speakers.csv"
TRAINING_CSV = REPO_ROOT / "data" / "training_data.csv"
SONG_FEATURES_CSV = REPO_ROOT / "data" / "song_features.csv"


@pytest.fixture(scope="module")
def speakers():
    return load_speakers_from_csv(SPEAKERS_CSV)


@pytest.fixture(scope="module")
def zones(speakers):
    return build_zones(speakers)


@pytest.fixture(scope="module")
def raw_training_df():
    return load_training_data(TRAINING_CSV)


@pytest.fixture(scope="module")
def song_features_df():
    return load_song_features(SONG_FEATURES_CSV)


@pytest.fixture(scope="module")
def training_df(raw_training_df, song_features_df):
    return prepare_training_data(raw_training_df, song_features_df)


@pytest.fixture(scope="module")
def sample_features(song_features_df):
    row = song_features_df.iloc[0]
    return {col: float(row[col]) for col in SONG_FEATURE_COLUMNS}


@pytest.fixture(scope="module")
def trained_zones(zones, training_df):
    train_all_zones(zones, training_df, n_neighbors=3)
    return zones


# Speaker / Subwoofer (classes, inheritance, operator overloads)

class TestSpeakerClass:

    def test_construct_and_str(self):
        spk = Speaker("Polk RC80i", 35, 20000, 89, 8, 100, zone_id=2)
        assert spk.name == "Polk RC80i"
        assert spk.freq_range == (35.0, 20000.0)
        assert "Polk RC80i" in str(spk)

    def test_supports_frequency(self):
        spk = Speaker("X", 40, 20000, 89, 8, 100)
        assert spk.supports_frequency(1000)
        assert not spk.supports_frequency(20)

    def test_capability_in_unit_range(self):
        spk = Speaker("X", 40, 20000, 89, 8, 100)
        assert 0.0 <= spk.bass_capability() <= 1.0
        assert 0.0 <= spk.treble_capability() <= 1.0

    def test_eq_and_hash(self):
        # Equality only looks at name + freq_range, not electrical specs.
        a = Speaker("X", 40, 20000, 89, 8, 100, zone_id=2)
        b = Speaker("X", 40, 20000, 90, 4, 200, zone_id=5)
        assert a == b
        assert hash(a) == hash(b)

    def test_subwoofer_inherits_and_overrides(self):
        sub = Subwoofer("Dayton SUB-1200", 25, 200, 87, 4, 120, 120, zone_id=7)
        assert isinstance(sub, Speaker)
        assert sub.bass_capability() == 1.0
        assert sub.treble_capability() == 0.0
        assert "Subwoofer" in str(sub)


# Exception scenario #1: bad construction inputs

class TestSpeakerExceptions:

    def test_invalid_freq_order_raises(self):
        with pytest.raises(ValueError, match="freq_max"):
            Speaker("Bad", 20000, 35, 89, 8, 100)

    def test_negative_sensitivity_raises(self):
        with pytest.raises(ValueError, match="sensitivity"):
            Speaker("Bad", 35, 20000, -5, 8, 100)

    def test_subwoofer_crossover_outside_range_raises(self):
        with pytest.raises(ValueError, match="crossover_freq"):
            Subwoofer("Bad", 25, 200, 87, 4, 120, crossover_freq=500)


# CSV loading + zone composition

class TestSpeakerLoadAndZoneComposition:

    def test_load_speakers_csv(self, speakers):
        assert len(speakers) == 11
        # Set comprehension over zone_ids: avoids duplicates from the four
        # Polk units in zone 2.
        assert {s.zone_id for s in speakers} == {2, 3, 4, 5, 7}
        # Subwoofer row must come back as a Subwoofer instance.
        subs = [s for s in speakers if isinstance(s, Subwoofer)]
        assert len(subs) == 1 and subs[0].zone_id == 7

    def test_build_zones_groups_by_zone_id(self, zones):
        assert set(zones.keys()) == {2, 3, 4, 5, 7}
        assert len(zones[2]) == 4
        assert len(zones[7]) == 1

    def test_zone_str_len_and_add(self, zones):
        text = str(zones[3])
        assert "Zone 3" in text
        assert len(zones[2]) == 4
        merged = zones[3] + zones[4]
        assert len(merged) == len(zones[3]) + len(zones[4])
        assert merged.zone_id > 0
        assert "Bookshelf" in merged.zone_name

    def test_zone_capability_aggregation(self, zones):
        # Subwoofer-only zone should average to (1.0, 0.0).
        assert zones[7].avg_bass_capability() == 1.0
        assert zones[7].avg_treble_capability() == 0.0


# Training data loader

class TestTrainingDataLoader:

    def test_load_collapses_grid(self, raw_training_df):
        # After collapsing each (zone, song) sweep, every key is unique.
        dup_count = raw_training_df.duplicated(
            subset=["zone_id", "song_title", "artist"]
        ).sum()
        assert dup_count == 0
        assert (raw_training_df["clarity"] >= 0).all()

    def test_prepare_training_data_merges_features(self, training_df):
        assert all(col in training_df.columns for col in SONG_FEATURE_COLUMNS)
        assert not training_df.loc[:, list(SONG_FEATURE_COLUMNS)].isna().any().any()

    def test_missing_file_raises(self):
        missing = REPO_ROOT / "data" / "definitely_missing_training.csv"
        assert not missing.exists()
        with pytest.raises(FileNotFoundError):
            load_training_data(missing)


# KNN training + prediction

class TestKnnPipeline:

    def test_train_returns_two_models(self, training_df):
        bass_model, treble_model = train_zone_knn(
            zone_id=3, training_df=training_df, n_neighbors=3,
        )
        assert bass_model.n_features_in_ == len(BASS_FEATURES)
        assert treble_model.n_features_in_ == len(TREBLE_FEATURES)

    def test_predict_in_controller_range(self, trained_zones, sample_features):
        for zid, zone in trained_zones.items():
            bass, treble = predict_zone_eq(zone, sample_features)
            assert -10.0 <= bass <= 10.0, f"bass out of range in zone {zid}"
            assert -10.0 <= treble <= 10.0, f"treble out of range in zone {zid}"
            # set_eq side effect should leave the zone holding the latest values.
            assert zone.eq_settings["bass"] == bass
            assert zone.eq_settings["treble"] == treble

    def test_subwoofer_treble_forced_to_zero(self, trained_zones, sample_features):
        _, treble = predict_zone_eq(trained_zones[7], sample_features)
        assert treble == 0.0


# Exception scenario #2: KNN-side validation

class TestKnnExceptions:

    def test_insufficient_samples(self, training_df):
        n = len(training_df[training_df["zone_id"] == 7])
        with pytest.raises(ValueError, match="need at least n_neighbors"):
            train_zone_knn(
                zone_id=7, training_df=training_df, n_neighbors=n + 1,
            )

    def test_training_with_missing_feature_column(self, training_df):
        with pytest.raises(ValueError, match="missing song feature columns"):
            train_zone_knn(
                zone_id=3,
                training_df=training_df.drop(columns=["energy"]),
                n_neighbors=3,
            )

    def test_training_with_nan_feature_value(self, training_df):
        broken = training_df.copy()
        broken.loc[0, "energy"] = None

        with pytest.raises(ValueError, match="blank or invalid"):
            train_zone_knn(zone_id=3, training_df=broken, n_neighbors=3)

    def test_predict_without_trained_model(self, sample_features):
        zone = SpeakerZone(
            zone_id=99, zone_name="Test Zone",
            speakers=[Speaker("X", 40, 20000, 89, 8, 100, zone_id=99)],
        )
        with pytest.raises(RuntimeError, match="no trained KNN model"):
            predict_zone_eq(zone, sample_features)

    def test_predict_with_missing_feature(self, trained_zones):
        with pytest.raises(ValueError, match="missing required key"):
            predict_zone_eq(trained_zones[3], {"energy": 0.5})
