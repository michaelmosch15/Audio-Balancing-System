"""Console entry point for the Audio Balancing System."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional because environment variables also work.
    load_dotenv = None

from src.spotify_client import SpotifyClient


PROJECT_ROOT = Path(__file__).resolve().parent
SPEAKERS_CSV = PROJECT_ROOT / "data" / "speakers.csv"
TRAINING_CSV = PROJECT_ROOT / "data" / "training_data.csv"
SONG_FEATURES_CSV = PROJECT_ROOT / "data" / "song_features.csv"


def load_training_data(filepath: str | Path = TRAINING_CSV) -> pd.DataFrame:
    """Load the raw tuning dataset from CSV."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Training data file not found: {path}")
    return pd.read_csv(path)


def initialize_system(n_neighbors: int = 3):
    """Load data, build zones, authenticate Spotify, load features, and train models."""
    from src.balancer import load_song_features, train_all_zones
    from src.speaker import load_speakers_from_csv
    from src.zone import build_zones

    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")

    speakers = load_speakers_from_csv(SPEAKERS_CSV)
    zones = build_zones(speakers)
    training_df = load_training_data(TRAINING_CSV)
    song_features = load_song_features(SONG_FEATURES_CSV)
    spotify_client = SpotifyClient()
    train_all_zones(zones, training_df, song_features_df=song_features, n_neighbors=n_neighbors)
    return zones, spotify_client, song_features


def prompt_for_song() -> tuple[str, str | None] | None:
    """Prompt the user for a song and optional artist."""
    song_name = input("Enter song name (or 'quit'): ").strip()
    if song_name.lower() in {"q", "quit"}:
        return None
    if not song_name:
        print("Please enter a song name.")
        return ("", None)

    artist = input("Artist (optional, press Enter to skip): ").strip()
    return song_name, artist or None


def interactive_loop(zones, spotify_client: SpotifyClient, song_features: pd.DataFrame) -> None:
    """Run the repeated song lookup, prediction, print, and chart workflow."""
    from src.balancer import format_recommendations, lookup_song_features, run_balancer
    from src.visualizer import visualize_eq_recommendations

    while True:
        requested = prompt_for_song()
        if requested is None:
            print("Goodbye.")
            break

        song_name, artist = requested
        if not song_name:
            continue

        try:
            song_info = spotify_client.get_song_metadata(song_name, artist)
        except (ConnectionError, ValueError) as exc:
            print(f"Error: {exc}")
            continue

        print(f"Matched Spotify track: {song_info['name']} by {song_info['artist']}")
        try:
            feature_values = lookup_song_features(
                song_features,
                song_info["name"],
                song_info["artist"],
            )
        except ValueError as exc:
            print(f"Error: {exc}")
            continue

        results = run_balancer(zones, feature_values)

        print()
        print(format_recommendations(song_info, results))
        print()
        visualize_eq_recommendations(results, song_info.get("name", song_name))


def main() -> None:
    """Initialize the system and start the interactive console loop."""
    try:
        zones, spotify_client, song_features = initialize_system()
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"Startup error: {exc}")
        return

    interactive_loop(zones, spotify_client, song_features)


if __name__ == "__main__":
    main()
