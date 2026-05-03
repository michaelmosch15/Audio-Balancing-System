"""SpeakerZone: composition of Speaker objects into a physical zone.

A zone owns a list of Speaker / Subwoofer instances and stores the EQ
recommendation produced by the KNN pipeline. ``build_zones`` is a
factory that groups a flat speaker list by zone_id.
"""

from src.speaker import Speaker


# Canonical names for each in-scope zone of the testbed (zones 1, 6, 8
# are wireless / out of scope per the proposal).
DEFAULT_ZONE_NAMES = {
    2: "Overhead Speakers",
    3: "Bookshelf Speakers",
    4: "Horizontal Firing (Zone 4)",
    5: "Horizontal Firing (Zone 5)",
    7: "Subwoofer",
}


def _finite_gain(value, label):
    """Return float(value) if it is finite; otherwise raise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} gain must be a real number, got {type(value).__name__}")
    g = float(value)
    if g != g or g in (float("inf"), float("-inf")):
        raise ValueError(f"{label} gain must be finite, got {value!r}")
    return g


class SpeakerZone:
    """A physical zone made up of one or more Speaker objects."""

    def __init__(self, zone_id, zone_name, speakers):
        """Create a zone. zone_id must be a positive int and zone_name non-empty."""
        if not isinstance(zone_id, int) or isinstance(zone_id, bool):
            raise TypeError(f"zone_id must be an int, got {type(zone_id).__name__}")
        if zone_id <= 0:
            raise ValueError(f"zone_id must be positive, got {zone_id}")
        if not isinstance(zone_name, str):
            raise TypeError(f"zone_name must be a str, got {type(zone_name).__name__}")
        if not zone_name.strip():
            raise ValueError("zone_name must be a non-empty string")

        self.zone_id = zone_id
        self.zone_name = zone_name
        # Mutable list — the zone owns these speakers (composition).
        self.speakers = list(speakers)
        # Mutable dict updated by set_eq once a prediction is made.
        self.eq_settings = {"bass": 0.0, "treble": 0.0}
        # Filled in by knn_model.train_zone_knn / train_all_zones.
        self.knn_model = None

    def avg_bass_capability(self):
        """Mean bass_capability across the zone's speakers."""
        self._require_speakers("avg_bass_capability")
        # Generator expression keeps things lazy — no intermediate list.
        return sum(s.bass_capability() for s in self.speakers) / len(self.speakers)

    def avg_treble_capability(self):
        """Mean treble_capability across the zone's speakers."""
        self._require_speakers("avg_treble_capability")
        return sum(s.treble_capability() for s in self.speakers) / len(self.speakers)

    def get_freq_range(self):
        """Return (min_freq_min, max_freq_max) across all member speakers."""
        self._require_speakers("get_freq_range")
        # List comprehensions: more idiomatic here than .append() loops.
        lows = [s.freq_range[0] for s in self.speakers]
        highs = [s.freq_range[1] for s in self.speakers]
        return (float(min(lows)), float(max(highs)))

    def set_eq(self, bass, treble):
        """Store the recommended EQ for this zone (no clamping done here)."""
        self.eq_settings["bass"] = _finite_gain(bass, "bass")
        self.eq_settings["treble"] = _finite_gain(treble, "treble")

    def __str__(self):
        """Single-line summary used by main.py for console output."""
        return (f"Zone {self.zone_id} ({self.zone_name}): "
                f"{len(self.speakers)} speakers | "
                f"Bass={self.eq_settings['bass']:+g} dB, "
                f"Treble={self.eq_settings['treble']:+g} dB")

    def __repr__(self):
        return (f"SpeakerZone(zone_id={self.zone_id!r}, "
                f"zone_name={self.zone_name!r}, "
                f"speakers=<{len(self.speakers)} items>)")

    def __len__(self):
        """Number of speakers in the zone."""
        return len(self.speakers)

    def __add__(self, other):
        """Merge two zones into a synthetic combined zone (for tests)."""
        if not isinstance(other, SpeakerZone):
            return NotImplemented
        # Encode the two source IDs into one positive int (e.g. 3,4 -> 30004).
        # Using a >=10_000 multiplier keeps merged IDs from colliding with real ones.
        merged_id = abs(self.zone_id) * 10_000 + abs(other.zone_id)
        return SpeakerZone(
            zone_id=merged_id,
            zone_name=f"{self.zone_name} + {other.zone_name}",
            speakers=self.speakers + other.speakers,
        )

    def _require_speakers(self, op):
        if not self.speakers:
            raise ValueError(f"Cannot compute {op} on zone {self.zone_id}: no speakers assigned.")


def build_zones(speakers, zone_names=None):
    """Group a flat speaker list into a {zone_id: SpeakerZone} dict.

    Each speaker is expected to expose a zone_id attribute (set by
    ``load_speakers_from_csv``). zone_names overrides the default
    display names from ``DEFAULT_ZONE_NAMES``; unknown ids fall back
    to "Zone <id>".
    """
    speakers_list = list(speakers)
    if not speakers_list:
        raise ValueError("build_zones() requires at least one speaker.")

    names = dict(DEFAULT_ZONE_NAMES)
    if zone_names:
        names.update(zone_names)

    grouped = {}
    for spk in speakers_list:
        if not isinstance(spk, Speaker):
            raise TypeError(f"build_zones() got non-Speaker object: {type(spk).__name__}")
        zid = int(spk.zone_id)
        grouped.setdefault(zid, []).append(spk)

    zones = {}
    for zid, members in grouped.items():
        zones[zid] = SpeakerZone(
            zone_id=zid,
            zone_name=names.get(zid, f"Zone {zid}"),
            speakers=members,
        )
    return zones
