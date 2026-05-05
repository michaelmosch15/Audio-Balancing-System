# Audio Balancing System for Diverse Speaker Sets

CPE 551 Final Project — Stevens Institute of Technology

- Michael Moschello (mmoschel@stevens.edu, 10479727)
- Michael Savo (msavo@stevens.edu, 20026290)

## What this project does

One of us has a multi-zone home audio setup in the basement: five speaker zones plus a subwoofer, all running off the same amplifier. The problem is that no single bass/treble setting sounds right for every song. A track like "Blinding Lights" that lives mostly in the mids needs very different EQ than something bass-heavy like a Daft Punk track, and the overhead speakers behave nothing like the subwoofer.

So we wrote a program that does the tuning for us. You type in a song name, it pulls the track's metadata from the Spotify Web API, looks up that song's audio features (loudness and energy, plus a few we precomputed), and feeds those features into a per-zone K-Nearest Neighbors model. Each zone gets its own predicted bass and treble gain based on training data we collected by hand while listening to reference tracks through that zone. The console prints the recommended EQ per zone and a matplotlib chart shows the result visually.

The point is: the speakers in each zone have different physical capabilities (frequency range, sensitivity, power handling), so it doesn't make sense to apply one global EQ. KNN lets us learn what "good" sounds like for each zone separately, without having to hand-write rules.

## Project layout

```
SpeakerBalence/
├── main.py                    # interactive console entry point
├── requirements.txt
├── pytest.ini
├── README.md
├── .env                       # your Spotify creds (you create this)
├── data/
│   ├── speakers.csv           # hardware specs per speaker
│   ├── training_data.csv      # hand-tuned (loudness, energy) -> (bass, treble) rows
│   └── song_features.csv      # precomputed audio features for known tracks
├── src/
│   ├── speaker.py             # Speaker class + Subwoofer subclass (inheritance)
│   ├── zone.py                # SpeakerZone (composition of Speakers)
│   ├── knn_model.py           # KNN training + prediction wrapper
│   ├── spotify_client.py      # spotipy wrapper for track metadata
│   ├── balancer.py            # ties zones + KNN + features together
│   └── visualizer.py          # matplotlib bar chart of EQ recommendations
├── tests/
│   ├── test_balancer.py
│   └── unit/test_pipeline.py
└── planning/
    ├── PROPOSAL.txt
    └── ARCHITECTURE.txt
```

## Who did what

**Michael M** — data + ML side
- `src/speaker.py`, `src/zone.py`, `src/knn_model.py`
- `data/speakers.csv`, `data/training_data.csv` (built from listening sessions on the basement testbed)
- KNN + speaker/zone tests

**Michael S** — Spotify, orchestration, UI
- `src/spotify_client.py`, `src/balancer.py`, `src/visualizer.py`
- `main.py` interactive loop
- Spotify dev app + `.env` setup
- Spotify / I/O tests

## Setup

From the repo root:

```powershell
python -m pip install -r requirements.txt
```

That installs pandas, numpy, scikit-learn, matplotlib, spotipy, python-dotenv, and pytest.

### Spotify credentials

The Spotify portion uses the Web API search endpoint to look up track metadata (title, artist, album, release date, duration, Spotify URL). We do not use the deprecated Audio Features endpoint — instead, the audio features used for KNN come from `data/song_features.csv`, which we precomputed for the songs we trained on.

To get the Spotify part working you need your own developer credentials:

1. Go to https://developer.spotify.com/dashboard and log in.
2. Click Create app. Name it whatever (e.g. `Audio Balancing System`).
3. For API/SDK, pick Web API.
4. If it asks for a redirect URI, enter `http://127.0.0.1:3000`.
5. Open the app's settings and copy the Client ID and Client Secret.
6. In the project root, make a file called `.env`:

```env
SPOTIPY_CLIENT_ID=your_client_id_here
SPOTIPY_CLIENT_SECRET=your_client_secret_here
```

`.env` is in `.gitignore` so it won't get pushed.

### Adding more songs

To make the program recommend EQ for a new song, add one row to `data/song_features.csv`. You do not need to add that song to `data/training_data.csv` unless you also want the model to learn from your own hand-tuned EQ ratings for that song. In other words, `song_features.csv` controls which songs can be predicted, while `training_data.csv` controls what the KNN model learns from.

Use Chosic's song analyzer as the preferred source for feature values: https://www.chosic.com/music-genre-finder/. Chosic is useful because it reports Spotify-style audio features such as energy, danceability, acousticness, instrumentalness, valence, speechiness, and tempo. These values are based on Spotify's audio analysis data, which normal Spotify Web API developers no longer have direct access to through the old Audio Features endpoint.

Keep the CSV song title and artist as clean, canonical names. The program can handle common Spotify metadata variants like radio edits, fuller artist names, and classical catalog numbers.

Example row format:

```csv
song_title,artist,energy,danceability,tempo,acousticness,instrumentalness,valence,speechiness
Black Hole Sun,Soundgarden,0.83,0.35,105.0,0.00,0.00,0.15,0.04
```

## Running

```powershell
python main.py
```

You'll get prompted for a song name (and optionally an artist). The program:

1. Searches Spotify for the track and prints what it matched.
2. Looks up that song's loudness/energy/etc. from `song_features.csv`.
3. Runs each zone's trained KNN to predict bass and treble gains.
4. Prints the recommendations and pops up a matplotlib bar chart.

Type `quit` (or `q`) when you're done.

To sanity-check Spotify auth without running the whole thing:

```powershell
python -c "from dotenv import load_dotenv; load_dotenv(); from src.spotify_client import SpotifyClient; c=SpotifyClient(); print(c); print(c.get_song_metadata('Blinding Lights', 'The Weeknd'))"
```

If your creds are good, you'll see `SpotifyClient(authenticated=True)` and a dict of track metadata.

## Tests

We use pytest. From the repo root:

```powershell
python -m pytest
```

Test files live in `tests/`. They cover the Speaker/Zone classes, the KNN training/prediction wrapper, the balancer orchestration, and the Spotify client (with mocked responses so the tests don't hit the network).

## How the rubric requirements are met

**Part 1 — fundamentals**
- *Classes:* `Speaker`, `Subwoofer(Speaker)` (inheritance), `SpeakerZone` (composition of Speakers), `SpotifyClient`.
- *Functions:* every module is broken into small functions with docstrings — see `src/balancer.py` and `main.py` for the orchestration ones.
- *Exception handling:* `main.py` catches `FileNotFoundError`, `ImportError`, `ValueError`, and `ConnectionError` around startup and the user loop. `src/spotify_client.py` raises `ConnectionError` / `ValueError` for auth and lookup failures. `src/knn_model.py` validates inputs and raises `ValueError` on bad training data.
- *Data I/O:* CSV loading via pandas in `main.py`, `src/balancer.py`, and `src/speaker.py`. Environment variables loaded from `.env` via python-dotenv.
- *Loops:* the interactive `while True` loop in `main.py`, plus zone-iteration loops in `src/balancer.py`.
- *Libraries:* pandas, numpy, scikit-learn (`KNeighborsRegressor`), matplotlib, spotipy, python-dotenv.
- *Docstrings:* every public class, function, and module has one.
- *README:* this file.

**Part 2 — advanced features (need at least four; we have more)**
1. *List/dict comprehensions* — used throughout `src/balancer.py` and `src/zone.py` for building zone lists and recommendation dicts.
2. *Operator overloading* — `Speaker.__str__` and `Speaker.__eq__` (and `Subwoofer` overrides) in `src/speaker.py`.
3. *Inheritance* — `Subwoofer` extends `Speaker` and overrides `bass_capability` / `treble_capability` / `__str__`.
4. *Built-in modules* — `pathlib.Path` for cross-platform file paths, `__future__` annotations, `typing` hints.
5. *Third-party advanced libraries* — scikit-learn for ML, matplotlib for visualization, spotipy for the Spotify Web API.
6. *Unit testing with pytest* including mocking for the Spotify client.

## Troubleshooting

- **`Startup error: Training data file not found`** — make sure you're running `python main.py` from the project root, not from inside `src/`.
- **Spotify auth fails** — double-check `.env` is in the project root and the variable names are exactly `SPOTIPY_CLIENT_ID` / `SPOTIPY_CLIENT_SECRET`. No quotes around the values.
- **`Song not found in features CSV`** — `song_features.csv` only contains the songs we precomputed features for. Try one of those (see the file for the full list) or add your own row.
- **matplotlib window doesn't appear** — on some systems you may need a GUI backend. `pip install pyqt5` usually fixes it on Windows.
