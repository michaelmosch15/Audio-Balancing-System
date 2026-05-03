"""Speaker hardware models.

Defines `Speaker` and its `Subwoofer` subclass, plus a CSV loader that
turns ``data/speakers.csv`` into a list of those objects. These are the
hardware-level pieces the rest of the system reasons about.
"""

import math
from pathlib import Path

import pandas as pd


# Frequency bands used for capability scoring (Hz).
BASS_BAND_HZ = 250.0
TREBLE_BAND_HZ = 4_000.0

# Anchors for normalizing capability scores into [0, 1]. 20 Hz / 20 kHz
# are the conventional bottom and top of human hearing; 95 dB and 200 W
# are reasonable upper references for sensitivity and power handling.
BASS_REF_HZ = 20.0
TREBLE_REF_HZ = 20_000.0
SENSITIVITY_REF_DB = 95.0
POWER_REF_W = 200.0

REQUIRED_COLUMNS = (
    "zone_id", "speaker_name", "type", "freq_min", "freq_max",
    "sensitivity", "impedance", "power_handling", "crossover_freq",
)


def _positive(value, label):
    """Return float(value) if it is a positive, finite real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number, got {type(value).__name__}")
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        raise ValueError(f"{label} must be finite, got {value!r}")
    if f <= 0:
        raise ValueError(f"{label} must be positive, got {f}")
    return f


def _clamp_unit(x):
    """Clamp x into [0.0, 1.0]."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


class Speaker:
    """A single passive speaker with its hardware specs."""

    def __init__(self, name, freq_min, freq_max, sensitivity, impedance,
                 power_handling, zone_id=0):
        """Build a Speaker. All numeric specs must be positive and finite."""
        if not isinstance(name, str):
            raise TypeError(f"name must be a str, got {type(name).__name__}")
        if not name.strip():
            raise ValueError("name must be a non-empty string")

        fmin = _positive(freq_min, "freq_min")
        fmax = _positive(freq_max, "freq_max")
        if fmax <= fmin:
            raise ValueError(f"freq_max ({fmax}) must be greater than freq_min ({fmin})")

        self.name = name.strip()
        # Tuple so the freq response is effectively immutable after construction.
        self.freq_range = (fmin, fmax)
        self.sensitivity = _positive(sensitivity, "sensitivity")
        self.impedance = _positive(impedance, "impedance")
        self.power_handling = _positive(power_handling, "power_handling")
        self.zone_id = int(zone_id)

    def supports_frequency(self, freq):
        """True if freq (Hz) lies inside this speaker's response."""
        f = _positive(freq, "freq")
        return self.freq_range[0] <= f <= self.freq_range[1]

    def bass_capability(self):
        """Score in [0, 1] for how well this speaker handles bass (<250 Hz).

        Mixes two factors: how far below 250 Hz the driver reaches on a
        log scale (so 20 Hz scores ~1.0, 250 Hz scores 0.0), and the
        power handling normalized against a 200 W reference. Log scaling
        is used because it matches how humans perceive frequency.
        """
        fmin = self.freq_range[0]
        if fmin >= BASS_BAND_HZ:
            reach = 0.0
        else:
            span = math.log10(BASS_BAND_HZ) - math.log10(BASS_REF_HZ)
            reach = (math.log10(BASS_BAND_HZ) - math.log10(fmin)) / span

        power = self.power_handling / POWER_REF_W
        return _clamp_unit((reach + power) / 2.0)

    def treble_capability(self):
        """Score in [0, 1] for how well this speaker handles treble (>4 kHz).

        Same idea as bass_capability but using freq_max against a 20 kHz
        anchor and sensitivity (dB) against a 95 dB anchor.
        """
        fmax = self.freq_range[1]
        if fmax <= TREBLE_BAND_HZ:
            reach = 0.0
        else:
            span = math.log10(TREBLE_REF_HZ) - math.log10(TREBLE_BAND_HZ)
            reach = (math.log10(fmax) - math.log10(TREBLE_BAND_HZ)) / span

        sens = self.sensitivity / SENSITIVITY_REF_DB
        return _clamp_unit((reach + sens) / 2.0)

    def __str__(self):
        """Human-readable one-liner."""
        fmin, fmax = self.freq_range
        return (f"Speaker({self.name}, {fmin:g}Hz-{fmax:g}Hz, "
                f"{self.sensitivity:g}dB, {self.impedance:g}ohm)")

    def __repr__(self):
        return (f"Speaker(name={self.name!r}, freq_range={self.freq_range!r}, "
                f"sensitivity={self.sensitivity!r}, impedance={self.impedance!r}, "
                f"power_handling={self.power_handling!r}, zone_id={self.zone_id!r})")

    def __eq__(self, other):
        """Two speakers are equal if name and freq_range match.

        Electrical specs and zone_id are intentionally ignored so duplicate
        physical units in the same zone (e.g. four Polk RC80i's in zone 2)
        compare equal.
        """
        if not isinstance(other, Speaker):
            return NotImplemented
        return self.name == other.name and self.freq_range == other.freq_range

    def __hash__(self):
        return hash((self.name, self.freq_range))


class Subwoofer(Speaker):
    """A dedicated subwoofer. Always full bass, never any treble."""

    def __init__(self, name, freq_min, freq_max, sensitivity, impedance,
                 power_handling, crossover_freq, zone_id=0):
        """Same as Speaker, plus a crossover_freq that must lie in the response."""
        super().__init__(name, freq_min, freq_max, sensitivity, impedance,
                         power_handling, zone_id=zone_id)
        xover = _positive(crossover_freq, "crossover_freq")
        fmin, fmax = self.freq_range
        if not (fmin <= xover <= fmax):
            raise ValueError(
                f"crossover_freq ({xover}) must lie within freq_range [{fmin}, {fmax}]"
            )
        self.crossover_freq = xover

    def bass_capability(self):
        """Subwoofers are dedicated bass drivers — always 1.0."""
        return 1.0

    def treble_capability(self):
        """Subwoofers can't reproduce treble — always 0.0."""
        return 0.0

    def __str__(self):
        fmin, fmax = self.freq_range
        return (f"Subwoofer({self.name}, {fmin:g}Hz-{fmax:g}Hz, "
                f"crossover={self.crossover_freq:g}Hz)")

    def __repr__(self):
        base = super().__repr__()
        return base.replace("Speaker(", "Subwoofer(", 1).rstrip(")") + (
            f", crossover_freq={self.crossover_freq!r})"
        )


def load_speakers_from_csv(filepath):
    """Read speakers.csv and return a list of Speaker / Subwoofer objects.

    Rows whose ``type`` column is "subwoofer" (case-insensitive) become
    Subwoofer instances and require a crossover_freq value; every other
    row becomes a regular Speaker. The zone_id column is copied onto each
    instance so the list can be passed straight to ``build_zones``.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Speaker CSV not found: {path}")

    df = pd.read_csv(path)

    # Validate columns up front so we don't crash deep inside the row loop.
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"speakers CSV is missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"speakers CSV contains no rows: {path}")

    speakers = []
    for index, row in df.iterrows():
        kind = str(row["type"]).strip().lower()
        common = dict(
            name=str(row["speaker_name"]),
            freq_min=float(row["freq_min"]),
            freq_max=float(row["freq_max"]),
            sensitivity=float(row["sensitivity"]),
            impedance=float(row["impedance"]),
            power_handling=float(row["power_handling"]),
            zone_id=int(row["zone_id"]),
        )
        if kind == "subwoofer":
            xover = row["crossover_freq"]
            if pd.isna(xover):
                raise ValueError(
                    f"Row {index}: subwoofer '{row['speaker_name']}' is missing crossover_freq"
                )
            speakers.append(Subwoofer(crossover_freq=float(xover), **common))
        else:
            speakers.append(Speaker(**common))
    return speakers
