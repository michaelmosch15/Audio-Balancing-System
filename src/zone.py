"""
Zone module.

Defines the :class:`SpeakerZone` aggregate that groups one or more
:class:`~src.speaker.Speaker` objects into a single physical zone of the
audio testbed. Each zone owns its recommended EQ settings and (after
training) its per-zone KNN regression model.

This module also provides :func:`build_zones`, which groups a flat list of
``Speaker`` objects into a ``{zone_id: SpeakerZone}`` mapping using the
canonical zone-name table defined by :data:`DEFAULT_ZONE_NAMES`.

The relationship between ``SpeakerZone`` and ``Speaker`` is *composition*:
a zone owns its speakers, but speakers exist independently and are passed
in by the caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - import only used for static typing
    from src.speaker import Speaker


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Canonical human-readable names for each in-scope zone of the testbed.
#: Sourced from the project architecture document (zones 1, 6, and 8 are
#: explicitly out of scope).
DEFAULT_ZONE_NAMES: Dict[int, str] = {
    2: "Overhead Speakers",
    3: "Bookshelf Speakers",
    4: "Horizontal Firing (Zone 4)",
    5: "Horizontal Firing (Zone 5)",
    7: "Subwoofer",
}

#: Hardware limit of the digital audio controller, in dB. Predicted EQ
#: values are clamped to this range elsewhere; the zone stores raw values.
_EQ_GAIN_LIMIT_DB: float = 10.0


# ---------------------------------------------------------------------------
# SpeakerZone
# ---------------------------------------------------------------------------

class SpeakerZone:
    """A physical zone composed of one or more :class:`Speaker` objects.

    A ``SpeakerZone`` aggregates the capabilities of its member speakers
    and acts as the unit of prediction for the recommendation system: each
    zone has its own KNN model trained on its own slice of the labeled
    data, and produces its own ``(bass, treble)`` gain recommendation.

    Attributes:
        zone_id: Stable integer identifier matching the ``zone_id`` column
            in ``speakers.csv`` and ``training_data.csv``.
        zone_name: Human-readable name shown in console output and charts.
        speakers: List of ``Speaker`` objects belonging to the zone. The
            list is owned by this instance; mutating it after construction
            is permitted (e.g. for tests) but is the caller's
            responsibility.
        eq_settings: Current recommended EQ for the zone, expressed as
            ``{"bass": float, "treble": float}`` in dB. Defaults to
            ``{"bass": 0.0, "treble": 0.0}`` and is updated by
            :meth:`set_eq`.
        knn_model: The fitted scikit-learn regressor produced by
            ``train_zone_knn``. ``None`` until training is performed.
    """

    __slots__ = (
        "zone_id",
        "zone_name",
        "speakers",
        "eq_settings",
        "knn_model",
    )

    def __init__(
        self,
        zone_id: int,
        zone_name: str,
        speakers: Iterable["Speaker"],
    ) -> None:
        """Initialize a zone.

        Args:
            zone_id: Integer zone identifier. Must be a positive ``int``.
            zone_name: Non-empty human-readable name.
            speakers: Iterable of ``Speaker`` instances to assign to the
                zone. May be empty at construction time but most
                operations (capability averages, frequency range) require
                at least one speaker.

        Raises:
            TypeError: If ``zone_id`` is not an ``int`` or ``zone_name``
                is not a ``str``.
            ValueError: If ``zone_id`` is not positive or ``zone_name``
                is empty / whitespace-only.
        """
        if not isinstance(zone_id, int) or isinstance(zone_id, bool):
            raise TypeError(
                f"zone_id must be an int, got {type(zone_id).__name__}"
            )
        if zone_id <= 0:
            raise ValueError(f"zone_id must be positive, got {zone_id}")

        if not isinstance(zone_name, str):
            raise TypeError(
                f"zone_name must be a str, got {type(zone_name).__name__}"
            )
        if not zone_name.strip():
            raise ValueError("zone_name must be a non-empty string")

        self.zone_id: int = zone_id
        self.zone_name: str = zone_name
        self.speakers: List["Speaker"] = list(speakers)
        self.eq_settings: Dict[str, float] = {"bass": 0.0, "treble": 0.0}
        self.knn_model: Optional[object] = None

    # ------------------------------------------------------------------
    # Capability aggregation
    # ------------------------------------------------------------------

    def avg_bass_capability(self) -> float:
        """Return the mean bass-capability score across member speakers.

        Returns:
            Float in ``[0.0, 1.0]`` representing the average of every
            member speaker's :meth:`Speaker.bass_capability` score.

        Raises:
            ValueError: If the zone contains no speakers (the average is
                undefined).
        """
        self._require_non_empty("avg_bass_capability")
        return sum(s.bass_capability() for s in self.speakers) / len(self.speakers)

    def avg_treble_capability(self) -> float:
        """Return the mean treble-capability score across member speakers.

        Returns:
            Float in ``[0.0, 1.0]`` representing the average of every
            member speaker's :meth:`Speaker.treble_capability` score.

        Raises:
            ValueError: If the zone contains no speakers.
        """
        self._require_non_empty("avg_treble_capability")
        return sum(s.treble_capability() for s in self.speakers) / len(self.speakers)

    def get_freq_range(self) -> Tuple[float, float]:
        """Return the combined frequency range covered by the zone.

        The lower bound is the minimum ``freq_min`` across all member
        speakers; the upper bound is the maximum ``freq_max``. The result
        therefore describes the *envelope* of frequencies that at least
        one speaker in the zone can reproduce.

        Returns:
            ``(lowest_min, highest_max)`` as a tuple of floats in Hz.

        Raises:
            ValueError: If the zone contains no speakers.
        """
        self._require_non_empty("get_freq_range")
        lows = [s.freq_range[0] for s in self.speakers]
        highs = [s.freq_range[1] for s in self.speakers]
        return (float(min(lows)), float(max(highs)))

    # ------------------------------------------------------------------
    # EQ state
    # ------------------------------------------------------------------

    def set_eq(self, bass: float, treble: float) -> None:
        """Record the recommended EQ gain for this zone.

        Inputs are *not* clamped here; the calling prediction code is
        responsible for clamping into the controller's hardware range.
        This method validates the inputs are real, finite numbers and
        stores them on the instance.

        Args:
            bass: Recommended bass gain in dB.
            treble: Recommended treble gain in dB.

        Raises:
            TypeError: If either argument is not numeric.
            ValueError: If either argument is NaN or infinite.
        """
        bass_f = self._coerce_gain(bass, "bass")
        treble_f = self._coerce_gain(treble, "treble")
        self.eq_settings["bass"] = bass_f
        self.eq_settings["treble"] = treble_f

    # ------------------------------------------------------------------
    # Operator overloads
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """Return a one-line human-readable summary of the zone."""
        return (
            f"Zone {self.zone_id} ({self.zone_name}): "
            f"{len(self.speakers)} speakers | "
            f"Bass={self.eq_settings['bass']:+g} dB, "
            f"Treble={self.eq_settings['treble']:+g} dB"
        )

    def __repr__(self) -> str:
        return (
            f"SpeakerZone(zone_id={self.zone_id!r}, "
            f"zone_name={self.zone_name!r}, "
            f"speakers=<{len(self.speakers)} items>)"
        )

    def __len__(self) -> int:
        """Return the number of speakers in the zone."""
        return len(self.speakers)

    def __add__(self, other: "SpeakerZone") -> "SpeakerZone":
        """Return a new zone whose speakers are the union of ``self`` and ``other``.

        The returned zone uses ``self.zone_id`` (negated to flag it as a
        synthetic merged zone) and a combined name. This operator is
        primarily intended to make tests that exercise multi-zone
        capability scoring easier to write; it is not used in the main
        prediction pipeline.

        Args:
            other: Another ``SpeakerZone`` to merge with.

        Returns:
            A brand-new ``SpeakerZone`` containing the concatenated
            speaker lists. EQ settings and the trained model are *not*
            copied.

        Raises:
            TypeError: If ``other`` is not a ``SpeakerZone``.
        """
        if not isinstance(other, SpeakerZone):
            return NotImplemented
        merged_id = -abs(self.zone_id) * 100 - abs(other.zone_id)
        merged_name = f"{self.zone_name} + {other.zone_name}"
        return SpeakerZone(
            zone_id=merged_id if merged_id < 0 else -1,
            zone_name=merged_name,
            speakers=self.speakers + other.speakers,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_non_empty(self, op: str) -> None:
        """Raise ``ValueError`` if the zone has no speakers."""
        if not self.speakers:
            raise ValueError(
                f"Cannot compute {op} on zone {self.zone_id}: no speakers assigned."
            )

    @staticmethod
    def _coerce_gain(value: object, label: str) -> float:
        """Validate and coerce a gain argument to ``float``."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"{label} gain must be a real number, got "
                f"{type(value).__name__}"
            )
        gain = float(value)
        # Reject NaN / infinity so downstream consumers (numpy, matplotlib)
        # don't propagate undefined values into charts or model state.
        if gain != gain or gain in (float("inf"), float("-inf")):
            raise ValueError(f"{label} gain must be finite, got {value!r}")
        return gain


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_zones(
    speakers: Iterable["Speaker"],
    zone_names: Optional[Dict[int, str]] = None,
) -> Dict[int, SpeakerZone]:
    """Group a flat iterable of speakers into ``SpeakerZone`` instances.

    Each speaker is expected to expose a ``zone_id`` attribute (set by
    :func:`load_speakers_from_csv`). Speakers that share a ``zone_id`` are
    placed in the same ``SpeakerZone``. The returned mapping is keyed by
    ``zone_id`` and ordered by insertion (i.e. by first appearance in
    ``speakers``).

    Args:
        speakers: Iterable of ``Speaker`` (or ``Subwoofer``) instances to
            partition.
        zone_names: Optional override mapping ``zone_id -> display name``.
            Falls back to :data:`DEFAULT_ZONE_NAMES`. Unknown zone IDs
            receive a generic ``"Zone <id>"`` name.

    Returns:
        ``dict`` of ``zone_id -> SpeakerZone``.

    Raises:
        AttributeError: If a speaker is missing the ``zone_id`` attribute.
        ValueError: If ``speakers`` is empty.
    """
    speakers_list = list(speakers)
    if not speakers_list:
        raise ValueError("build_zones() requires at least one speaker.")

    name_table: Dict[int, str] = dict(DEFAULT_ZONE_NAMES)
    if zone_names:
        name_table.update(zone_names)

    grouped: Dict[int, List["Speaker"]] = {}
    for spk in speakers_list:
        try:
            zid = int(spk.zone_id)
        except AttributeError as exc:
            raise AttributeError(
                "Speaker objects passed to build_zones() must expose a "
                "'zone_id' attribute."
            ) from exc
        grouped.setdefault(zid, []).append(spk)

    zones: Dict[int, SpeakerZone] = {}
    for zid, members in grouped.items():
        zones[zid] = SpeakerZone(
            zone_id=zid,
            zone_name=name_table.get(zid, f"Zone {zid}"),
            speakers=members,
        )
    return zones
