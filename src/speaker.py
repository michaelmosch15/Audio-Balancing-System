"""
Speaker module.

Defines the hardware-level domain objects of the audio-balancing system:

* :class:`Speaker` — a single passive speaker unit characterized by its
  frequency response, sensitivity, impedance, and power handling.
* :class:`Subwoofer` — a specialization of ``Speaker`` for low-frequency
  drivers, which adds a crossover frequency and overrides the
  bass/treble capability scoring.

The two classes form the **inheritance** half of the project's required
class relationships (``Subwoofer`` *is-a* ``Speaker``); the
**composition** half lives in :mod:`src.zone`, where ``SpeakerZone``
owns a list of these objects.

Capability scoring uses :mod:`math.log10` to compress the very wide
frequency range a speaker can cover (tens of Hz to tens of kHz) into a
normalized ``[0.0, 1.0]`` score, which better matches the way humans
perceive frequency than a linear mapping would.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Upper edge of the "bass" band in Hz. Frequencies at or below this are
#: considered low-end content for capability scoring.
_BASS_BAND_HZ: float = 250.0

#: Lower edge of the "treble" band in Hz. Frequencies at or above this are
#: considered high-end content for capability scoring.
_TREBLE_BAND_HZ: float = 4_000.0

#: Reference low end (Hz) at which a non-subwoofer speaker would score a
#: perfect 1.0 for bass capability. 20 Hz is the conventional bottom of
#: human hearing.
_BASS_REFERENCE_HZ: float = 20.0

#: Reference high end (Hz) at which a speaker would score a perfect 1.0
#: for treble capability. 20 kHz is the conventional top of human hearing.
_TREBLE_REFERENCE_HZ: float = 20_000.0

#: Reference sensitivity (dB) used to normalize the sensitivity component
#: of treble capability. Typical bookshelf speakers fall in the 82–92 dB
#: range, so 95 dB is a generous upper anchor.
_SENSITIVITY_REFERENCE_DB: float = 95.0

#: Reference power handling (W RMS) used to normalize the power component
#: of bass capability. Larger drivers with higher power handling typically
#: produce stronger bass.
_POWER_REFERENCE_W: float = 200.0

#: Required columns for ``speakers.csv``. Validated at load time.
_REQUIRED_SPEAKER_COLUMNS: Tuple[str, ...] = (
    "zone_id",
    "speaker_name",
    "type",
    "freq_min",
    "freq_max",
    "sensitivity",
    "impedance",
    "power_handling",
    "crossover_freq",
)


# ---------------------------------------------------------------------------
# Speaker
# ---------------------------------------------------------------------------

class Speaker:
    """A single passive speaker unit.

    A ``Speaker`` carries the static hardware specifications needed to
    reason about how well a particular driver can reproduce a given part
    of the audio spectrum. Two scoring methods —
    :meth:`bass_capability` and :meth:`treble_capability` — collapse
    these specs into ``[0.0, 1.0]`` scalars that downstream code (the
    zone-level aggregator and, indirectly, the visualizer) can compare
    across heterogeneous hardware.

    Attributes:
        name: Manufacturer / model identifier (e.g. ``"Polk RC80i"``).
        freq_range: ``(freq_min, freq_max)`` tuple in Hz. Stored as a
            tuple so the frequency response is treated as immutable.
        sensitivity: Sensitivity in dB SPL @ 1 W / 1 m.
        impedance: Nominal impedance in ohms.
        power_handling: Maximum continuous (RMS) power in watts.
        zone_id: Zone identifier this speaker belongs to. Set by the
            CSV loader; used by :func:`src.zone.build_zones` to group
            speakers into zones.
    """

    def __init__(
        self,
        name: str,
        freq_min: float,
        freq_max: float,
        sensitivity: float,
        impedance: float,
        power_handling: float,
        zone_id: int = 0,
    ) -> None:
        """Construct a ``Speaker``.

        Args:
            name: Non-empty manufacturer / model identifier.
            freq_min: Lower bound of the speaker's frequency response,
                in Hz. Must be positive and strictly less than
                ``freq_max``.
            freq_max: Upper bound of the speaker's frequency response,
                in Hz. Must be strictly greater than ``freq_min``.
            sensitivity: Sensitivity in dB. Must be positive (typical
                values fall between ~80 and ~95 dB).
            impedance: Nominal impedance in ohms. Must be positive.
            power_handling: Continuous power handling in watts. Must be
                positive.
            zone_id: Optional zone identifier. Defaults to ``0`` for
                free-standing speakers that aren't attached to a zone
                yet (e.g. when constructed directly in tests).

        Raises:
            TypeError: If ``name`` is not a ``str`` or any numeric
                argument is non-numeric.
            ValueError: If ``name`` is empty/whitespace, ``freq_min`` is
                not positive, ``freq_max`` is not greater than
                ``freq_min``, or any of ``sensitivity``, ``impedance``,
                or ``power_handling`` is non-positive.
        """
        if not isinstance(name, str):
            raise TypeError(f"name must be a str, got {type(name).__name__}")
        if not name.strip():
            raise ValueError("name must be a non-empty string")

        fmin = self._coerce_positive(freq_min, "freq_min")
        fmax = self._coerce_positive(freq_max, "freq_max")
        if fmax <= fmin:
            raise ValueError(
                f"freq_max ({fmax}) must be greater than freq_min ({fmin})"
            )

        self.name: str = name.strip()
        self.freq_range: Tuple[float, float] = (fmin, fmax)
        self.sensitivity: float = self._coerce_positive(sensitivity, "sensitivity")
        self.impedance: float = self._coerce_positive(impedance, "impedance")
        self.power_handling: float = self._coerce_positive(
            power_handling, "power_handling"
        )
        self.zone_id: int = int(zone_id)

    # ------------------------------------------------------------------
    # Capability methods
    # ------------------------------------------------------------------

    def supports_frequency(self, freq: float) -> bool:
        """Return ``True`` if ``freq`` falls within the speaker's response.

        Args:
            freq: Frequency to test, in Hz.

        Returns:
            ``True`` iff ``freq_min <= freq <= freq_max``.

        Raises:
            TypeError: If ``freq`` is not numeric.
            ValueError: If ``freq`` is non-positive.
        """
        f = self._coerce_positive(freq, "freq")
        return self.freq_range[0] <= f <= self.freq_range[1]

    def bass_capability(self) -> float:
        """Score how well this speaker reproduces bass content.

        The score combines two factors:

        1. How far below the bass-band ceiling (``250 Hz``) the
           speaker's ``freq_min`` reaches, on a logarithmic scale —
           speakers that extend down to ~20 Hz score full marks, while
           speakers that roll off above 80 Hz score progressively less.
        2. The speaker's power handling, normalized against a 200 W
           reference. Higher-powered drivers generally move more air at
           low frequencies.

        The two factors are averaged equally and the result is clamped
        to ``[0.0, 1.0]``.

        Returns:
            Bass capability as a float in ``[0.0, 1.0]``.
        """
        fmin = self.freq_range[0]
        # Logarithmic frequency reach: 1.0 at 20 Hz, 0.0 at the bass-band
        # ceiling (250 Hz). Speakers that don't reach into the bass band
        # at all are floored at zero.
        if fmin >= _BASS_BAND_HZ:
            reach_score = 0.0
        else:
            span = math.log10(_BASS_BAND_HZ) - math.log10(_BASS_REFERENCE_HZ)
            reach_score = (math.log10(_BASS_BAND_HZ) - math.log10(fmin)) / span

        power_score = self.power_handling / _POWER_REFERENCE_W
        return _clamp_unit((reach_score + power_score) / 2.0)

    def treble_capability(self) -> float:
        """Score how well this speaker reproduces treble content.

        Combines two factors:

        1. How far above the treble-band floor (``4 kHz``) the speaker's
           ``freq_max`` reaches, on a logarithmic scale.
        2. The speaker's sensitivity, normalized against a 95 dB
           reference. More sensitive drivers tend to render high-end
           detail more cleanly at moderate listening levels.

        The two factors are averaged equally and clamped to
        ``[0.0, 1.0]``.

        Returns:
            Treble capability as a float in ``[0.0, 1.0]``.
        """
        fmax = self.freq_range[1]
        if fmax <= _TREBLE_BAND_HZ:
            reach_score = 0.0
        else:
            span = math.log10(_TREBLE_REFERENCE_HZ) - math.log10(_TREBLE_BAND_HZ)
            reach_score = (math.log10(fmax) - math.log10(_TREBLE_BAND_HZ)) / span

        sensitivity_score = self.sensitivity / _SENSITIVITY_REFERENCE_DB
        return _clamp_unit((reach_score + sensitivity_score) / 2.0)

    # ------------------------------------------------------------------
    # Operator overloads
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """Return a one-line human-readable summary."""
        fmin, fmax = self.freq_range
        return (
            f"Speaker({self.name}, {fmin:g}Hz-{fmax:g}Hz, "
            f"{self.sensitivity:g}dB, {self.impedance:g}ohm)"
        )

    def __repr__(self) -> str:
        return (
            f"Speaker(name={self.name!r}, freq_range={self.freq_range!r}, "
            f"sensitivity={self.sensitivity!r}, impedance={self.impedance!r}, "
            f"power_handling={self.power_handling!r}, zone_id={self.zone_id!r})"
        )

    def __eq__(self, other: object) -> bool:
        """Two speakers are equal when ``name`` and ``freq_range`` match.

        Equality intentionally ignores ``zone_id`` and the electrical
        specs so duplicate physical units in the same zone (e.g. four
        Polk RC80i speakers in zone 2) compare equal — which is what
        the deduplication logic in tests and in any future inventory
        view will want.
        """
        if not isinstance(other, Speaker):
            return NotImplemented
        return self.name == other.name and self.freq_range == other.freq_range

    def __hash__(self) -> int:
        return hash((self.name, self.freq_range))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_positive(value: object, label: str) -> float:
        """Validate that ``value`` is a positive, finite real number."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"{label} must be a real number, got {type(value).__name__}"
            )
        f = float(value)
        if f != f or f in (float("inf"), float("-inf")):
            raise ValueError(f"{label} must be finite, got {value!r}")
        if f <= 0.0:
            raise ValueError(f"{label} must be positive, got {f}")
        return f


# ---------------------------------------------------------------------------
# Subwoofer (inheritance)
# ---------------------------------------------------------------------------

class Subwoofer(Speaker):
    """A subwoofer driver — a specialization of :class:`Speaker`.

    Subwoofers behave fundamentally differently from full-range drivers:
    they are dedicated to the low end and intentionally do not reproduce
    treble content above their crossover. Capability scoring is
    therefore overridden to a fixed ``1.0`` for bass and ``0.0`` for
    treble, regardless of nominal frequency response.

    Attributes:
        crossover_freq: The crossover frequency in Hz above which the
            subwoofer's output is rolled off. Stored alongside the base
            ``Speaker`` attributes.
    """

    def __init__(
        self,
        name: str,
        freq_min: float,
        freq_max: float,
        sensitivity: float,
        impedance: float,
        power_handling: float,
        crossover_freq: float,
        zone_id: int = 0,
    ) -> None:
        """Construct a ``Subwoofer``.

        Args:
            name: Manufacturer / model identifier.
            freq_min: Lower bound of frequency response in Hz.
            freq_max: Upper bound of frequency response in Hz.
            sensitivity: Sensitivity in dB.
            impedance: Nominal impedance in ohms.
            power_handling: Continuous power handling in watts.
            crossover_freq: Crossover frequency in Hz. Must be positive
                and lie within ``[freq_min, freq_max]`` so it represents
                an actual point in the driver's response.
            zone_id: Optional zone identifier; see :class:`Speaker`.

        Raises:
            ValueError: If ``crossover_freq`` is non-positive or falls
                outside the speaker's frequency range. All other
                validation is inherited from :class:`Speaker`.
        """
        super().__init__(
            name=name,
            freq_min=freq_min,
            freq_max=freq_max,
            sensitivity=sensitivity,
            impedance=impedance,
            power_handling=power_handling,
            zone_id=zone_id,
        )
        xover = self._coerce_positive(crossover_freq, "crossover_freq")
        fmin, fmax = self.freq_range
        if not (fmin <= xover <= fmax):
            raise ValueError(
                f"crossover_freq ({xover}) must lie within freq_range "
                f"[{fmin}, {fmax}]"
            )
        self.crossover_freq: float = xover

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def bass_capability(self) -> float:
        """Subwoofers are dedicated bass drivers — always return ``1.0``."""
        return 1.0

    def treble_capability(self) -> float:
        """Subwoofers do not reproduce treble — always return ``0.0``."""
        return 0.0

    def __str__(self) -> str:
        fmin, fmax = self.freq_range
        return (
            f"Subwoofer({self.name}, {fmin:g}Hz-{fmax:g}Hz, "
            f"crossover={self.crossover_freq:g}Hz)"
        )

    def __repr__(self) -> str:
        base = super().__repr__()
        # Insert the crossover field into the parent's repr for symmetry.
        return base.replace("Speaker(", "Subwoofer(", 1).rstrip(")") + (
            f", crossover_freq={self.crossover_freq!r})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp_unit(value: float) -> float:
    """Clamp ``value`` to the closed interval ``[0.0, 1.0]``."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_speakers_from_csv(filepath: Union[str, Path]) -> List[Speaker]:
    """Load speakers from a CSV file into ``Speaker`` / ``Subwoofer`` objects.

    The CSV must have the columns listed in
    :data:`_REQUIRED_SPEAKER_COLUMNS`. Each row becomes one ``Speaker``;
    rows whose ``type`` column equals ``"subwoofer"`` (case-insensitive)
    become ``Subwoofer`` instances and require a non-empty
    ``crossover_freq`` value.

    Args:
        filepath: Path to ``speakers.csv``.

    Returns:
        A list of ``Speaker`` (and/or ``Subwoofer``) objects in the
        same order they appear in the CSV. The ``zone_id`` attribute is
        populated on each instance from the corresponding column so the
        list can be passed directly to :func:`src.zone.build_zones`.

    Raises:
        FileNotFoundError: If ``filepath`` does not exist.
        ValueError: If the CSV is missing required columns, contains no
            rows, or a subwoofer row is missing its ``crossover_freq``.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Speaker CSV not found: {path}")

    df = pd.read_csv(path)

    missing = [col for col in _REQUIRED_SPEAKER_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"speakers CSV is missing required columns: {missing}"
        )
    if df.empty:
        raise ValueError(f"speakers CSV contains no rows: {path}")

    speakers: List[Speaker] = []
    for index, row in df.iterrows():
        kind = str(row["type"]).strip().lower()
        # Branch on the type column — subwoofers need an extra crossover
        # field; everything else is a generic full-range Speaker.
        if kind == "subwoofer":
            xover_raw = row["crossover_freq"]
            if pd.isna(xover_raw):
                raise ValueError(
                    f"Row {index}: subwoofer '{row['speaker_name']}' is "
                    "missing crossover_freq"
                )
            speakers.append(
                Subwoofer(
                    name=str(row["speaker_name"]),
                    freq_min=float(row["freq_min"]),
                    freq_max=float(row["freq_max"]),
                    sensitivity=float(row["sensitivity"]),
                    impedance=float(row["impedance"]),
                    power_handling=float(row["power_handling"]),
                    crossover_freq=float(xover_raw),
                    zone_id=int(row["zone_id"]),
                )
            )
        else:
            speakers.append(
                Speaker(
                    name=str(row["speaker_name"]),
                    freq_min=float(row["freq_min"]),
                    freq_max=float(row["freq_max"]),
                    sensitivity=float(row["sensitivity"]),
                    impedance=float(row["impedance"]),
                    power_handling=float(row["power_handling"]),
                    zone_id=int(row["zone_id"]),
                )
            )
    return speakers
