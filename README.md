**Michael M** — data + ML side
- `src/speaker.py`, `src/zone.py`, `src/knn_model.py`
- `data/speakers.csv`, `data/training_data.csv` (built from listening sessions on the basement testbed)
- KNN + speaker model tests

**Michael S** — Spotify, orchestration, UI
- `src/spotify_client.py`, `src/balancer.py`, `src/visualizer.py`
- `main.py` interactive loop
- Spotify dev app + `.env` setup
- Spotify / I/O tests

## Setup

Install the project dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

## Spotify Web API Credentials

This project uses Spotify's Web API search endpoint to identify songs and return
track metadata such as title, artist, album, release date, duration, and Spotify
URL. It does not use Spotify's deprecated Audio Features endpoint.

To run the Spotify portion of the project, create your own Spotify developer app:

1. Go to https://developer.spotify.com/dashboard.
2. Log in with a Spotify account.
3. Click **Create app**.
4. Enter any app name and description, such as `Audio Balancing System`.
5. For API/SDK selection, choose **Web API**.
6. If Spotify asks for a redirect URI, enter:

```text
http://127.0.0.1:3000
```

7. After creating the app, open its settings.
8. Copy the **Client ID** and **Client Secret**.
9. In the project root, create a local `.env` file:

```env
SPOTIPY_CLIENT_ID=your_client_id_here
SPOTIPY_CLIENT_SECRET=your_client_secret_here
```

Keep `.env` private and do not commit it to GitHub.

## Running

From the repository root:

```powershell
python main.py
```

To quickly test Spotify search credentials:

```powershell
python -c "from dotenv import load_dotenv; load_dotenv(); from src.spotify_client import SpotifyClient; c=SpotifyClient(); print(c); print(c.get_song_metadata('Blinding Lights', 'The Weeknd'))"
```

If the credentials are valid, the command prints `SpotifyClient(authenticated=True)`
and a dictionary of Spotify track metadata.
