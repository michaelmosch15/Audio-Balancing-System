"""
Project Architecture Tree — Audio Balancing System for Diverse Speaker Sets

Renders the full project architecture as a clean tree diagram and exports
it to a PNG image suitable for embedding in a Word document.

Run:  python project_tree.py
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def render_tree_png(output_path="project_tree.png"):
    """Render the project architecture tree to a Word-friendly PNG file."""

    tree = (
        "Audio Balancing System for Diverse Speaker Sets\n"
        "Moschello & Savo  -  CPE 551\n"
        "\n"
        "FILE STRUCTURE\n"
        "\n"
        "    SpeakerBalance/\n"
        "    |\n"
        "    +-- data/\n"
        "    |   +-- speakers.csv              Speaker hardware specs per zone\n"
        "    |   +-- training_data.csv         User-labeled training data per zone\n"
        "    |\n"
        "    +-- src/\n"
        "    |   +-- speaker.py                Speaker + Subwoofer classes\n"
        "    |   +-- zone.py                   SpeakerZone class (composition)\n"
        "    |   +-- spotify_client.py         Spotify API wrapper\n"
        "    |   +-- knn_model.py              Dual KNN training + prediction\n"
        "    |   +-- balancer.py               Orchestration across all zones\n"
        "    |   +-- visualizer.py             Matplotlib chart output\n"
        "    |\n"
        "    +-- tests/\n"
        "    |   +-- test_balancer.py          Pytest test cases\n"
        "    |\n"
        "    +-- main.py                       Entry point\n"
        "    +-- requirements.txt              Dependencies\n"
        "    +-- .env                          Spotify API credentials\n"
        "    +-- README.md                     Project documentation\n"
        "\n"
        "\n"
        "CLASSES & RELATIONSHIPS\n"
        "\n"
        "    Speaker  (src/speaker.py)\n"
        "    |   Attributes:  name, freq_range, sensitivity, impedance, power_handling\n"
        "    |   Methods:     bass_capability(), treble_capability()\n"
        "    |   Overloads:   __str__(), __eq__()\n"
        "    |\n"
        "    +-- Subwoofer  [inherits from Speaker]  (src/speaker.py)\n"
        "    |       Additional:  crossover_freq\n"
        "    |       Overrides:   bass_capability() -> 1.0,  treble_capability() -> 0.0\n"
        "    |\n"
        "    SpeakerZone  [contains list of Speakers]  (src/zone.py)\n"
        "    |   Attributes:  zone_id, zone_name, speakers[], eq_settings{},\n"
        "    |                bass_knn_model, treble_knn_model\n"
        "    |   Methods:     avg_bass_capability(), avg_treble_capability(), set_eq()\n"
        "    |   Overloads:   __str__(), __len__(), __add__()\n"
        "    |\n"
        "    SpotifyClient  (src/spotify_client.py)\n"
        "        Attributes:  client_id, client_secret, sp\n"
        "        Methods:     search_track(), get_audio_features(), get_song_features()\n"
        "        Overloads:   __str__()\n"
        "\n"
        "\n"
        "CORE FUNCTIONS\n"
        "\n"
        "    train_zone_knn(zone_id, training_df, n_neighbors)     src/knn_model.py\n"
        "        Fits TWO KNeighborsRegressor models per zone:\n"
        "          Bass KNN:   [energy, danceability, tempo,\n"
        "                       1-acousticness, instrumentalness] -> ideal_bass\n"
        "          Treble KNN: [energy, valence, acousticness,\n"
        "                       speechiness] -> ideal_treble\n"
        "\n"
        "    predict_zone_eq(zone, features_dict)                  src/knn_model.py\n"
        "        Builds bass + treble feature vectors from Spotify data\n"
        "        Runs each KNN, clamps output to [-10, +10]\n"
        "\n"
        "    load_speakers_from_csv(filepath)                      src/speaker.py\n"
        "        Reads CSV, builds Speaker/Subwoofer objects per row\n"
        "\n"
        "    load_training_data(filepath)                          src/knn_model.py\n"
        "        Reads CSV, validates columns, returns DataFrame\n"
        "\n"
        "    run_balancer(zones, features_dict)                    src/balancer.py\n"
        "        Loops all zones, calls predict_zone_eq() for each\n"
        "\n"
        "    visualize_eq_recommendations(zones, song_name)        src/visualizer.py\n"
        "        Grouped bar chart showing bass/treble gain per zone\n"
        "\n"
        "\n"
        "LIBRARIES\n"
        "\n"
        "    scikit-learn       KNeighborsRegressor (prediction engine)\n"
        "    matplotlib         Grouped bar charts for EQ visualization\n"
        "    spotify            Spotify Web API (search + audio features)\n"
        "    numpy              Feature arrays, gain clamping\n"
        "    pandas             CSV reading, DataFrame filtering\n"
        "\n"
        "\n"
        "ERROR HANDLING & TESTS\n"
        "\n"
        "    Exception:  Spotify API errors (song not found, network failure)\n"
        "    Exception:  Training data errors (file missing, too few samples)\n"
        "    Pytest:     test_knn_prediction_range (gains within [-10, +10])\n"
        "    Pytest:     test_subwoofer_treble_zero (zone 7 treble = 0)\n"
        "\n"
        "\n"
        "END-TO-END FLOW\n"
        "\n"
        "    1. Load speakers.csv            Build Speaker/Subwoofer objects\n"
        "    2. Load training_data.csv       Pandas DataFrame\n"
        "    3. Train 2 KNNs per zone       Bass KNN + Treble KNN per SpeakerZone\n"
        "    4. User enters song name        While loop for continuous input\n"
        "    5. Spotify API lookup            Pull perceptual + spectral features\n"
        "    6. Bass KNN predicts per zone   energy, danceability, tempo, etc.\n"
        "    7. Treble KNN predicts per zone energy, valence, acousticness, etc.\n"
        "    8. Clamp gains to +/-10 dB      Apply controller range limits\n"
        "    9. Output results               Console print + Matplotlib chart\n"
        "\n"
        "\n"
        "PART 2 FEATURES\n"
        "\n"
        "    List comprehension              Zone ID extraction, speaker name lists\n"
        "    math module                     Logarithmic frequency-to-capability scoring\n"
        "    enumerate                       Chart bar positioning, result printing\n"
        "    if __name__ == \"__main__\"        main.py entry guard\n"
    )

    # -- figure sizing --
    lines = tree.split("\n")
    n_lines = len(lines)
    max_width = max(len(line) for line in lines)

    fig_w = max(8.5, max_width * 0.072)
    fig_h = max(6, n_lines * 0.138)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # pick a monospace font
    mono_fonts = ["Consolas", "Cascadia Code", "Courier New", "DejaVu Sans Mono", "monospace"]
    chosen_font = "monospace"
    available = {f.name for f in fm.fontManager.ttflist}
    for f in mono_fonts:
        if f in available:
            chosen_font = f
            break

    # draw section headers bold, rest normal
    y_pos = 0.98
    line_height = 1.0 / (n_lines + 2)

    headers = {
        "Audio Balancing System for Diverse Speaker Sets",
        "FILE STRUCTURE",
        "CLASSES & RELATIONSHIPS",
        "CORE FUNCTIONS",
        "LIBRARIES",
        "ERROR HANDLING & TESTS",
        "END-TO-END FLOW",
        "PART 2 FEATURES",
    }

    for line_text in lines:
        stripped = line_text.strip()
        if stripped in headers:
            weight = "bold"
            size = 9 if stripped == "Audio Balancing System for Diverse Speaker Sets" else 8.5
        elif stripped == "Moschello & Savo  -  CPE 551":
            weight = "normal"
            size = 8
        else:
            weight = "normal"
            size = 7.5

        ax.text(
            0.03, y_pos, line_text,
            transform=ax.transAxes,
            fontsize=size,
            fontfamily=chosen_font,
            fontweight=weight,
            color="black",
            verticalalignment="top",
            horizontalalignment="left",
        )
        y_pos -= line_height

    plt.tight_layout(pad=0.3)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    render_tree_png()

