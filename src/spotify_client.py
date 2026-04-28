
"stuff i was trying out with the spotify client stuff"

from __future__ import annotations

import os
from typing import Optional

import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials


REQUIRED_FEATURE_KEYS: tuple[str, ...] = (
    "loudness",
    "energy",
    "danceability",
    "tempo",
    "acousticness",
    "instrumentalness",
    "valence",
    "speechiness",
)


class SpotifyClient:
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> None:
        load_dotenv()

        self.client_id = client_id or os.environ.get("SPOTIPY_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get(
            "SPOTIPY_CLIENT_SECRET", ""
        )

        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Spotify credentials not set. Define SPOTIPY_CLIENT_ID and "
                "SPOTIPY_CLIENT_SECRET in your environment or .env file."
            )

        try:
            auth_manager = SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret,
            )
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
        except spotipy.SpotifyOauthError as exc:
            raise ConnectionError(
                f"Failed to authenticate with Spotify: {exc}"
            ) from exc

    def search_track(
        self,
        song_name: str,
        artist: Optional[str] = None,
    ) -> dict:
        if not song_name or not song_name.strip():
            raise ValueError("song_name must be a non-empty string.")

        query = f'track:"{song_name.strip()}"'
        if artist and artist.strip():
            query += f' artist:"{artist.strip()}"'

        try:
            results = self.sp.search(q=query, type="track", limit=1)
        except spotipy.SpotifyException as exc:
            raise ConnectionError(f"Spotify API error during search: {exc}") from exc

        items = results.get("tracks", {}).get("items", [])
        if not items:
            raise ValueError(
                f"No track found on Spotify for: '{song_name}'"
                + (f" by '{artist}'" if artist else "")
            )

        track = items[0]
        return {
            "track_id": track["id"],
            "name": track["name"],
            "artist": track["artists"][0]["name"] if track["artists"] else "",
        }

    def get_audio_features(self, track_id: str) -> dict:
        if not track_id:
            raise ValueError("track_id must be a non-empty string.")

        try:
            payload = self.sp.audio_features([track_id])
        except spotipy.SpotifyException as exc:
            raise ConnectionError(
                f"Spotify API error fetching audio features: {exc}"
            ) from exc

        if not payload or payload[0] is None:
            raise ValueError(
                f"Spotify returned no audio features for track_id '{track_id}'."
            )

        raw = payload[0]
        missing = [k for k in REQUIRED_FEATURE_KEYS if k not in raw]
        if missing:
            raise ValueError(
                f"Spotify audio-features payload is missing keys: {missing}"
            )

        return {key: float(raw[key]) for key in REQUIRED_FEATURE_KEYS}

    def get_song_features(
        self,
        song_name: str,
        artist: Optional[str] = None,
    ) -> dict:
        track = self.search_track(song_name, artist=artist)
        features = self.get_audio_features(track["track_id"])
        return {**features, **track}

    def __str__(self) -> str:
        return f"SpotifyClient(authenticated={self.sp is not None})"
