from __future__ import annotations

import bisect
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class DonorPoint:
    """A trusted GPS reference point from a geotagged catalog photo or GPX track."""

    media_id: int | None
    timestamp: int
    lat: float
    lon: float
    location_name: str = ""
    camera_model: str = ""
    filename: str = ""
    path: str = ""
    source: str = "catalog"  # "catalog" | "gpx"


@dataclass(slots=True)
class TargetPhoto:
    """An untagged catalog photo with a capture timestamp."""

    media_id: int
    path: str
    filename: str
    timestamp: int
    camera_model: str = ""


@dataclass(slots=True)
class ProposedMatch:
    """A proposed GPS match for an untagged target photo."""

    media_id: int
    path: str
    filename: str
    camera_model: str
    target_ts: int
    lat: float
    lon: float
    location_name: str
    time_gap_sec: int
    confidence: str  # "exact" | "nearby" | "interpolated"
    donor_desc: str
    donor_media_id: int | None = None
    donor_filename: str = ""
    donor_path: str = ""
    donor_camera_model: str = ""
    donor_timestamp: int = 0
    donor_lat: float | None = None
    donor_lon: float | None = None
    donor_location_name: str = ""
    donor_source: str = "catalog"
    checked: bool = True


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two WGS84 coordinates."""
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _parse_iso_timestamp(text: str) -> int | None:
    s = (text or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(round(dt.timestamp()))
    except Exception:
        return None


def parse_gpx_file(gpx_path: str | Path) -> list[DonorPoint]:
    """Parses trackpoints (<trkpt>) and waypoints (<wpt>) with <time> from a GPX file."""
    path = Path(gpx_path)
    tree = ET.parse(path)
    root = tree.getroot()
    points: list[DonorPoint] = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        if tag not in ("trkpt", "wpt"):
            continue
        try:
            lat = float(elem.attrib["lat"])
            lon = float(elem.attrib["lon"])
        except (KeyError, ValueError, TypeError):
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        ts: int | None = None
        name = ""
        for child in elem:
            ctag = child.tag.split("}")[-1].lower()
            if ctag == "time" and child.text:
                ts = _parse_iso_timestamp(child.text)
            elif ctag == "name" and child.text:
                name = child.text.strip()
        if ts is not None and ts > 0:
            points.append(
                DonorPoint(
                    media_id=None,
                    timestamp=ts,
                    lat=lat,
                    lon=lon,
                    location_name=name,
                    camera_model="GPX Track",
                    filename=path.name,
                    source="gpx",
                )
            )
    points.sort(key=lambda p: p.timestamp)
    return points


def format_time_gap(seconds: int) -> str:
    """Formats a non-negative time gap in seconds cleanly (e.g. '2m', '1h 15m')."""
    s = max(0, int(abs(seconds)))
    if s < 60:
        return f"{s}s"
    mins = s // 60
    if mins < 60:
        return f"{mins}m"
    hrs = mins // 60
    rem_m = mins % 60
    return f"{hrs}h {rem_m}m" if rem_m else f"{hrs}h"


def match_timed_photos(
    targets: Iterable[TargetPhoto],
    donors: Iterable[DonorPoint],
    *,
    clock_offset_sec: int = 0,
    max_gap_sec: int = 7200,
    same_spot_sec: int = 900,
    same_spot_meters: float = 100.0,
    max_speed_kmh: float = 250.0,
) -> list[ProposedMatch]:
    """
    Matches untagged target photos against sorted GPS donor points using
    camera clock-drift offset (Δt), same-spot clamping, and speed-guarded
    linear interpolation.
    """
    sorted_donors = sorted(
        (d for d in donors if d.timestamp > 0 and -90.0 <= d.lat <= 90.0 and -180.0 <= d.lon <= 180.0),
        key=lambda d: d.timestamp,
    )
    if not sorted_donors:
        return []

    donor_ts = [d.timestamp for d in sorted_donors]
    matches: list[ProposedMatch] = []

    for tgt in targets:
        if tgt.timestamp <= 0:
            continue
        t_adj = int(tgt.timestamp) + int(clock_offset_sec)
        idx = bisect.bisect_left(donor_ts, t_adj)
        d1 = sorted_donors[idx - 1] if idx > 0 else None
        d2 = sorted_donors[idx] if idx < len(sorted_donors) else None

        gap1 = abs(t_adj - d1.timestamp) if d1 is not None else 10**12
        gap2 = abs(d2.timestamp - t_adj) if d2 is not None else 10**12
        nearest = d1 if gap1 <= gap2 else d2
        nearest_gap = min(gap1, gap2)

        if nearest is None or nearest_gap > max_gap_sec:
            continue

        # 1. Same-spot clamping: close in time (<= 15 min) or both donors at same spot (< 100m)
        both_exist = d1 is not None and d2 is not None
        same_spot = False
        if nearest_gap <= same_spot_sec:
            same_spot = True
        elif both_exist and (d2.timestamp - d1.timestamp) <= max_gap_sec:
            if haversine_meters(d1.lat, d1.lon, d2.lat, d2.lon) <= same_spot_meters:
                same_spot = True

        if same_spot:
            conf = "exact" if nearest_gap <= 300 else "nearby"
            desc = f"{'Exact' if conf == 'exact' else 'Nearby'} ({format_time_gap(nearest_gap)})"
            matches.append(
                ProposedMatch(
                    media_id=tgt.media_id,
                    path=tgt.path,
                    filename=tgt.filename,
                    camera_model=tgt.camera_model,
                    target_ts=tgt.timestamp,
                    lat=nearest.lat,
                    lon=nearest.lon,
                    location_name=nearest.location_name,
                    time_gap_sec=nearest_gap,
                    confidence=conf,
                    donor_desc=desc,
                    donor_media_id=nearest.media_id,
                    donor_filename=nearest.filename,
                    donor_path=nearest.path,
                    donor_camera_model=nearest.camera_model,
                    donor_timestamp=nearest.timestamp,
                    donor_lat=nearest.lat,
                    donor_lon=nearest.lon,
                    donor_location_name=nearest.location_name,
                    donor_source=nearest.source,
                )
            )
            continue

        # 2. Between-points linear interpolation (requires bracketing donors within max_gap_sec)
        if both_exist and gap1 <= max_gap_sec and gap2 <= max_gap_sec and d2.timestamp > d1.timestamp:
            dt_sec = d2.timestamp - d1.timestamp
            dist_m = haversine_meters(d1.lat, d1.lon, d2.lat, d2.lon)
            if dt_sec > 60:
                speed_kmh = (dist_m / 1000.0) / (dt_sec / 3600.0)
                if speed_kmh > max_speed_kmh:
                    # Travel speed implausibly high (e.g., flight); skip unsafe interpolation
                    continue

            alpha = (t_adj - d1.timestamp) / float(dt_sec)
            lat = d1.lat + alpha * (d2.lat - d1.lat)
            lon = d1.lon + alpha * (d2.lon - d1.lon)
            loc_name = ""
            if haversine_meters(lat, lon, nearest.lat, nearest.lon) <= 150.0 and nearest.location_name:
                loc_name = nearest.location_name
            matches.append(
                ProposedMatch(
                    media_id=tgt.media_id,
                    path=tgt.path,
                    filename=tgt.filename,
                    camera_model=tgt.camera_model,
                    target_ts=tgt.timestamp,
                    lat=lat,
                    lon=lon,
                    location_name=loc_name,
                    time_gap_sec=nearest_gap,
                    confidence="interpolated",
                    donor_desc=f"Interpolated ({format_time_gap(dt_sec)} span)",
                    donor_media_id=nearest.media_id,
                    donor_filename=nearest.filename,
                    donor_path=nearest.path,
                    donor_camera_model=nearest.camera_model,
                    donor_timestamp=nearest.timestamp,
                    donor_lat=nearest.lat,
                    donor_lon=nearest.lon,
                    donor_location_name=nearest.location_name,
                    donor_source=nearest.source,
                )
            )
            continue

        # 3. Single-sided fallback only within 30 min
        if nearest_gap <= min(max_gap_sec, 1800):
            matches.append(
                ProposedMatch(
                    media_id=tgt.media_id,
                    path=tgt.path,
                    filename=tgt.filename,
                    camera_model=tgt.camera_model,
                    target_ts=tgt.timestamp,
                    lat=nearest.lat,
                    lon=nearest.lon,
                    location_name=nearest.location_name,
                    time_gap_sec=nearest_gap,
                    confidence="nearby",
                    donor_desc=f"Nearby ({format_time_gap(nearest_gap)})",
                    donor_media_id=nearest.media_id,
                    donor_filename=nearest.filename,
                    donor_path=nearest.path,
                    donor_camera_model=nearest.camera_model,
                    donor_timestamp=nearest.timestamp,
                    donor_lat=nearest.lat,
                    donor_lon=nearest.lon,
                    donor_location_name=nearest.location_name,
                    donor_source=nearest.source,
                )
            )

    return matches


def resolve_missing_place_names(matches: list[ProposedMatch], db_engine=None) -> None:
    """Populates blank location_name fields on ProposedMatch items using offline reverse geocoding."""
    if not matches:
        return
    try:
        from src.core.geotag_resolver import reverse_geocode
    except Exception:
        return

    cache: dict[tuple[float, float], str] = {}
    for m in matches:
        if m.location_name:
            continue
        key = (round(m.lat, 3), round(m.lon, 3))
        if key not in cache:
            try:
                rev = reverse_geocode(m.lat, m.lon, db_engine=db_engine)
                if rev:
                    parts = [p for p in (rev.get("city"), rev.get("state"), rev.get("country")) if p]
                    cache[key] = ", ".join(parts) if parts else str(rev.get("name") or "")
                else:
                    cache[key] = f"{m.lat:.4f}, {m.lon:.4f}"
            except Exception:
                cache[key] = f"{m.lat:.4f}, {m.lon:.4f}"
        m.location_name = cache[key]
