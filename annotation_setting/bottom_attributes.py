"""Lower-body attribute enums (English) for labeling and code."""

from __future__ import annotations

from enum import Enum


class BottomWaistRise(str, Enum):
    """bottom_waist_rise — 하의 허리선 위치"""

    HIGH_RISE = "high_rise"
    MID_RISE = "mid_rise"
    LOW_RISE = "low_rise"
    UNKNOWN = "unknown"


class BottomLength(str, Enum):
    """bottom_length — 하의 기장"""

    MICRO_SHORT = "micro_short"
    SHORT = "short"
    BERMUDA = "bermuda"
    CAPRI = "capri"
    MIDI = "midi"
    ANKLE = "ankle"
    FLOOR_MAXI = "floor_maxi"
    UNKNOWN = "unknown"


BOTTOM_WAIST_RISE_VALUES: tuple[str, ...] = tuple(v.value for v in BottomWaistRise)
BOTTOM_LENGTH_VALUES: tuple[str, ...] = tuple(v.value for v in BottomLength)
