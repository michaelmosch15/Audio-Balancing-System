import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from src import balancer
from src.spotify_client import SpotifyClient
from src.visualizer import visualize_eq_recommendations


class FakeZone:
    """Small test double that acts like a SpeakerZone."""

    def __init__(self, zone_id, zone_name):
        """Store only the attributes balancer.py needs."""
        self.zone_id = zone_id
        self.zone_name = zone_name
        self.knn_model = None


def test_prepare_training_data_selects_highest_clarity_row():
    """Raw clarity-grid data should become model-ready KNN labels."""
    raw_data = pd.DataFrame(
        [
            {
                "zone_id": 2,
                "song_title": "Song A",
                "artist": "Artist",
                "bass": -3,
                "treble": 1,
                "clarity": 2.0,
                "loudness": -5.5,
                "energy": 0.75,
            },
            {
                "zone_id": 2,
                "song_title": "Song A",
                "artist": "Artist",
                "bass": 4,
                "treble": -2,
                "clarity": 5.0,
                "loudness": -5.5,
                "energy": 0.75,
            },
        ]
    )

    prepared = balancer.prepare_training_data(raw_data)

    assert len(prepared) == 1
    row = prepared.iloc[0]
    assert row["ideal_bass"] == 4
    assert row["ideal_treble"] == -2
    assert row["loudness"] == -5.5
    assert row["energy"] == 0.75


def test_prepare_training_data_requires_local_song_features():
    """Spotify should not be used to fill deprecated audio feature fields."""
    raw_data = pd.DataFrame(
        [
            {"zone_id": 2, "song_title": "Song A", "artist": "Artist", "bass": 4, "treble": -2, "clarity": 5.0},
        ]
    )

    with pytest.raises(ValueError, match="loudness and energy"):
        balancer.prepare_training_data(raw_data)


def test_run_balancer_formats_zone_results(monkeypatch):
    """run_balancer should call predict_zone_eq once per zone."""
    zones = {2: FakeZone(2, "Overhead"), 7: FakeZone(7, "Subwoofer")}

    def fake_predict(zone, loudness, energy):
        return zone.zone_id, -zone.zone_id

    monkeypatch.setattr(balancer, "predict_zone_eq", fake_predict)

    results = balancer.run_balancer(zones, loudness=-6.0, energy=0.8)

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
