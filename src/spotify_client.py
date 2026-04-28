"""Spotify API wrapper used to retrieve non-deprecated track metadata."""

from __future__ import annotations

import os
from typing import Any

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    spotipy = None
    SpotifyClientCredentials = None


class SpotifyClient:
    """Authenticate with Spotify and search public track metadata."""

    def __init__(self, client_id: str | None = None, client_secret: str | None = None):
        """Create an authenticated Spotify client using client credentials.

        Args:
            client_id: Spotify application client ID. Falls back to the
                SPOTIPY_CLIENT_ID environment variable.
            client_secret: Spotify application secret. Falls back to the
                SPOTIPY_CLIENT_SECRET environment variable.

        Raises:
            ImportError: If spotipy is not installed.
            ValueError: If credentials are missing.
        """
        if spotipy is None or SpotifyClientCredentials is None:
            raise ImportError("spotipy is required. Install dependencies with pip install -r requirements.txt")

        self.client_id = client_id or os.getenv("SPOTIPY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SPOTIPY_CLIENT_SECRET")

        if not self.client_id or not self.client_secret:
            raise ValueError("Spotify credentials not set. Check SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET.")

        credentials = SpotifyClientCredentials(
            client_id=self.client_id,
            client_secret=self.client_secret,
        )
        self.sp = spotipy.Spotify(client_credentials_manager=credentials)

    def search_track(self, song_name: str, artist: str | None = None) -> dict[str, Any]:
        """Search Spotify for a track and return non-deprecated metadata.

        Args:
            song_name: Track title to search for.
            artist: Optional artist name to narrow the search.

        Returns:
            A dictionary with track id, name, artist, album, release date,
            duration, popularity, and Spotify URL.

        Raises:
            ValueError: If the song name is empty or no result is found.
            ConnectionError: If Spotify rejects or fails the request.
        """
        if not song_name or not song_name.strip():
            raise ValueError("Song name cannot be empty.")

        query = f'track:"{song_name.strip()}"'
        if artist and artist.strip():
            query += f' artist:"{artist.strip()}"'

        try:
            results = self.sp.search(q=query, type="track", limit=1)
        except Exception as exc:  # spotipy raises SpotifyException, plus network errors
            raise ConnectionError(f"Spotify API search failed: {exc}") from exc

        items = results.get("tracks", {}).get("items", [])
        if not items:
            requested = f"{song_name} by {artist}" if artist else song_name
            raise ValueError(f"No track found on Spotify for: {requested}")

        track = items[0]
        artists = track.get("artists", [])
        primary_artist = artists[0].get("name", "Unknown Artist") if artists else "Unknown Artist"
        album = track.get("album", {})
        external_urls = track.get("external_urls", {})
        return {
            "id": track["id"],
            "name": track.get("name", song_name),
            "artist": primary_artist,
            "album": album.get("name"),
            "release_date": album.get("release_date"),
            "duration_ms": track.get("duration_ms"),
            "popularity": track.get("popularity"),
            "spotify_url": external_urls.get("spotify"),
        }

    def get_song_metadata(self, song_name: str, artist: str | None = None) -> dict[str, Any]:
        """Search Spotify and return non-deprecated track metadata.

        Args:
            song_name: Track title to search for.
            artist: Optional artist name to narrow the search.

        Returns:
            The same metadata dictionary returned by search_track().
        """
        return self.search_track(song_name, artist)

    def get_song_features(self, song_name: str, artist: str | None = None) -> dict[str, Any]:
        """Backward-compatible alias for get_song_metadata().

        Spotify removed access to the Audio Features endpoint for new Web API
        apps, so this method intentionally returns metadata only. Loudness,
        energy, or replacement song descriptors must come from local data or
        user input.
        """
        return self.get_song_metadata(song_name, artist)

    def __str__(self) -> str:
        """Return a readable authentication summary."""
        return "SpotifyClient(authenticated=True)"
