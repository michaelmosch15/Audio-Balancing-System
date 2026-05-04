import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from src import balancer
import src.spotify_client as spotify_module
from src.spotify_client import SpotifyClient
from src.visualizer import visualize_eq_recommendations


FEATURE_ROW = {
    "song_title": "Song A",
    "artist": "Artist",
    "energy": 0.75,
    "danceability": 0.6,
    "tempo": 120.0,
    "acousticness": 0.2,
    "instrumentalness": 0.0,
    "valence": 0.5,
    "speechiness": 0.05,
}


class FakeZone:
    """Small test double that acts like a SpeakerZone."""

    def __init__(self, zone_id, zone_name):
        """Store only the attributes balancer.py needs."""
        self.zone_id = zone_id
        self.zone_name = zone_name
        self.knn_model = None


def raw_training_rows():
    return pd.DataFrame(
        [
            {
                "zone_id": 2,
                "song_title": "Song A",
                "artist": "Artist",
                "bass": -3,
                "treble": 1,
                "clarity": 2.0,
            },
            {
                "zone_id": 2,
                "song_title": "Song A",
                "artist": "Artist",
                "bass": 4,
                "treble": -2,
                "clarity": 5.0,
            },
        ]
    )


def song_feature_rows():
    return pd.DataFrame([FEATURE_ROW])


def test_prepare_training_data_merges_song_features_and_selects_highest_clarity_row():
    """Raw clarity-grid data should merge features into model-ready KNN rows."""
    prepared = balancer.prepare_training_data(raw_training_rows(), song_feature_rows())

    assert len(prepared) == 1
    row = prepared.iloc[0]
    assert row["bass"] == 4
    assert row["treble"] == -2
    assert row["energy"] == 0.75
    assert row["tempo"] == 120.0


def test_prepare_training_data_requires_song_feature_rows():
    """Training songs must have matching rows in song_features.csv."""
    with pytest.raises(ValueError, match="missing feature rows"):
        balancer.prepare_training_data(raw_training_rows(), pd.DataFrame([{
            **FEATURE_ROW,
            "song_title": "Different Song",
        }]))


def test_validate_song_features_requires_feature_columns():
    """song_features.csv must contain every KNN feature column."""
    with pytest.raises(ValueError, match="missing required columns"):
        balancer.validate_song_features(song_feature_rows().drop(columns=["energy"]))


def test_validate_song_features_rejects_blank_values():
    """Blank feature values should fail before model training."""
    features = song_feature_rows().astype({"tempo": object})
    features.loc[0, "tempo"] = ""

    with pytest.raises(ValueError, match="blank or invalid"):
        balancer.validate_song_features(features)


def test_lookup_song_features_is_case_insensitive():
    """Spotify metadata casing does not need to exactly match the CSV casing."""
    features = balancer.lookup_song_features(song_feature_rows(), "song a", "artist")

    assert features["energy"] == 0.75
    assert features["tempo"] == 120.0


def test_run_balancer_formats_zone_results(monkeypatch):
    """run_balancer should call predict_zone_eq once per zone with a feature dict."""
    zones = {2: FakeZone(2, "Overhead"), 7: FakeZone(7, "Subwoofer")}

    def fake_predict(zone, song_features):
        assert song_features["energy"] == 0.75
        return zone.zone_id, -zone.zone_id

    monkeypatch.setattr(balancer, "predict_zone_eq", fake_predict)

    results = balancer.run_balancer(zones, {"energy": 0.75})

    assert results[2] == {"zone_name": "Overhead", "bass": 2.0, "treble": -2.0}
    assert results[7] == {"zone_name": "Subwoofer", "bass": 7.0, "treble": -7.0}


def test_visualizer_returns_figure_without_showing():
    """Visualizer should create an inspectable Matplotlib figure."""
    fig = visualize_eq_recommendations(
        {2: {"zone_name": "Overhead", "bass": 1, "treble": -2}},
        "Test Song",
        show=False,
    )

    assert fig.axes
    assert fig.axes[0].get_title() == "EQ Recommendations for Test Song"


def test_spotify_client_missing_credentials(monkeypatch):
    """SpotifyClient should reject startup without credentials."""
    class FakeCredentials:
        def __init__(self, client_id, client_secret):
            self.client_id = client_id
            self.client_secret = client_secret

    class FakeSpotipyModule:
        class Spotify:
            def __init__(self, client_credentials_manager):
                self.client_credentials_manager = client_credentials_manager

    monkeypatch.setattr(spotify_module, "spotipy", FakeSpotipyModule)
    monkeypatch.setattr(spotify_module, "SpotifyClientCredentials", FakeCredentials)
    monkeypatch.delenv("SPOTIPY_CLIENT_ID", raising=False)
    monkeypatch.delenv("SPOTIPY_CLIENT_SECRET", raising=False)

    with pytest.raises(ValueError):
        SpotifyClient()


def test_spotify_client_get_song_metadata_with_fake_spotify():
    """SpotifyClient should return metadata without audio_features calls."""

    class FakeSpotipyClient:
        """Fake spotipy client with only the non-deprecated search method."""

        def search(self, q, type, limit):
            """Return one fake Spotify search result."""
            return {
                "tracks": {
                    "items": [
                        {
                            "id": "track-123",
                            "name": "Test Song",
                            "artists": [{"name": "Test Artist"}],
                            "album": {"name": "Test Album", "release_date": "2024-01-01"},
                            "duration_ms": 180000,
                            "popularity": 80,
                            "external_urls": {"spotify": "https://open.spotify.com/track/track-123"},
                        }
                    ]
                }
            }

    client = SpotifyClient.__new__(SpotifyClient)
    client.client_id = "fake"
    client.client_secret = "fake"
    client.sp = FakeSpotipyClient()

    features = client.get_song_metadata("Test Song", "Test Artist")

    assert features["id"] == "track-123"
    assert features["name"] == "Test Song"
    assert features["artist"] == "Test Artist"
    assert features["album"] == "Test Album"
    assert features["release_date"] == "2024-01-01"
    assert features["duration_ms"] == 180000
    assert features["popularity"] == 80
    assert features["spotify_url"] == "https://open.spotify.com/track/track-123"
    assert "loudness" not in features
    assert "energy" not in features


