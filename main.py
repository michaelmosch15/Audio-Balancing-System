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


def load_training_data(filepath: str | Path = TRAINING_CSV) -> pd.DataFrame:
    """Load the raw tuning dataset from CSV.

    Args:
        filepath: Path to the training CSV.

    Returns:
        A Pandas DataFrame containing the training rows.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Training data file not found: {path}")
    return pd.read_csv(path)


def initialize_system(n_neighbors: int = 3):
    """Load data, build zones, authenticate Spotify, and train models."""
    from src.balancer import train_all_zones
    from src.speaker import load_speakers_from_csv
    from src.zone import build_zones

    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")

    speakers = load_speakers_from_csv(SPEAKERS_CSV)
    zones = build_zones(speakers)
    training_df = load_training_data(TRAINING_CSV)
    spotify_client = SpotifyClient()
    train_all_zones(zones, training_df, n_neighbors=n_neighbors)
    return zones, spotify_client


def prompt_for_song() -> tuple[str, str | None] | None:
    """Prompt the user for a song and optional artist.

    Returns:
        A (song_name, artist) tuple, or None when the user wants to quit.
    """
    song_name = input("Enter song name (or 'quit'): ").strip()
    if song_name.lower() in {"q", "quit"}:
        return None
    if not song_name:
        print("Please enter a song name.")
        return ("", None)

    artist = input("Artist (optional, press Enter to skip): ").strip()
    return song_name, artist or None


def prompt_for_song_features() -> tuple[float, float] | None:
    """Prompt for local song features used by the KNN model.

    Spotify search is still used to identify the track, but loudness and energy
    must come from local/manual data because Spotify's Audio Features endpoint
    is deprecated for new apps.
    """
    try:
        loudness = float(input("Loudness in dB from your CSV/manual notes: ").strip())
        energy = float(input("Energy from 0.0 to 1.0 from your CSV/manual notes: ").strip())
    except ValueError:
        print("Please enter numeric values for loudness and energy.")
        return None

    if not 0.0 <= energy <= 1.0:
        print("Energy must be between 0.0 and 1.0.")
        return None

    return loudness, energy


def interactive_loop(zones, spotify_client: SpotifyClient) -> None:
    """Run the repeated song lookup, prediction, print, and chart workflow."""
    from src.balancer import format_recommendations, run_balancer
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
        feature_values = prompt_for_song_features()
        if feature_values is None:
            continue

        loudness, energy = feature_values
        song_info["loudness"] = loudness
        song_info["energy"] = energy
        results = run_balancer(zones, loudness, energy)

        print()
        print(format_recommendations(song_info, results))
        print()
        visualize_eq_recommendations(results, song_info.get("name", song_name))


def main() -> None:
    """Initialize the system and start the interactive console loop."""
    try:
        zones, spotify_client = initialize_system()
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"Startup error: {exc}")
        return

    interactive_loop(zones, spotify_client)


if __name__ == "__main__":
    main()
