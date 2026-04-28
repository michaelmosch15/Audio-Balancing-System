from src.spotify_client import SpotifyClient


def main() -> None:
    song_name = "Despacito"

    client = SpotifyClient()
    features = client.get_song_features(song_name)

    print(f'Song:   "{features["name"]}" by {features["artist"]}')
    print(f'Track ID: {features["track_id"]}')
    print("Audio features:")
    for key in (
        "loudness",
        "energy",
        "danceability",
        "tempo",
        "acousticness",
        "instrumentalness",
        "valence",
        "speechiness",
    ):
        print(f"  {key:<17} {features[key]}")


if __name__ == "__main__":
    main()
