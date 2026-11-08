"""
Comprehensive Offline Gazetteer and Online Nominatim Geotag Resolver for PhotoGallery64.
Provides sub-2ms instant prefix/substring gazetteer search across 3,500+ world cities
and detailed UK towns/villages, KD-Tree reverse geocoding, and OpenStreetMap Nominatim fallback
with automatic SQLite caching in saved_places.
"""

from __future__ import annotations

import json
import logging
import math
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence

from src.core.gazetteer_data import BUILTIN_PLACES

logger = logging.getLogger(__name__)

# Try importing reverse_geocoder and scipy
_HAS_REVERSE_GEOCODER = False
try:
    import reverse_geocoder as _rg  # type: ignore
    _HAS_REVERSE_GEOCODER = True
except Exception:
    _rg = None

_HAS_SCIPY = False
try:
    import numpy as np
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except Exception:
    np = None
    cKDTree = None


def _normalize_admin(
    name: str, admin2: str, admin1: str, country: str, cc: str
) -> tuple[str, str]:
    """
    Cleans raw FIPS/GeoNames administrative codes into human-readable regional names:
    - Normalizes UK subdivision codes: ENG -> England, SCT -> Scotland, WLS -> Wales, NIR -> Northern Ireland.
    - Resolves known administrative division codes (Canada, Australia, etc.).
    - Strips internal numeric codes (e.g. '01', '07', '39', '051') so raw database numbers never leak to UI.
    """
    a1 = (admin1 or "").strip()
    a2 = (admin2 or "").strip()
    cc_up = (cc or "").strip().upper()
    c_norm = (country or "").strip().lower()

    # 1. Normalize UK subdivision codes (ENG -> England)
    if cc_up == "GB" or c_norm in ("united kingdom", "great britain", "uk"):
        uk_map = {
            "ENG": "England",
            "SCT": "Scotland",
            "WLS": "Wales",
            "NIR": "Northern Ireland",
        }
        if a1.upper() in uk_map:
            a1 = uk_map[a1.upper()]
        if a2.upper() in uk_map:
            a2 = ""

    # 2. Normalize Canadian province codes
    elif cc_up == "CA" or c_norm == "canada":
        ca_map = {
            "01": "Alberta", "02": "British Columbia", "03": "Manitoba",
            "04": "New Brunswick", "05": "Newfoundland and Labrador", "07": "Nova Scotia",
            "08": "Ontario", "09": "Prince Edward Island", "10": "Quebec",
            "11": "Saskatchewan", "12": "Yukon", "13": "Northwest Territories", "14": "Nunavut",
        }
        if a1 in ca_map:
            a1 = ca_map[a1]

    # 3. Normalize Australian state codes
    elif cc_up == "AU" or c_norm == "australia":
        au_map = {
            "01": "Australian Capital Territory", "02": "New South Wales",
            "03": "Northern Territory", "04": "Queensland", "05": "South Australia",
            "06": "Tasmania", "07": "Victoria", "08": "Western Australia",
        }
        if a1 in au_map:
            a1 = au_map[a1]

    # 4. Strip pure numeric codes (e.g. '01', '07', '39', '051') so numbers never leak to UI
    if a1.isdigit():
        a1 = ""
    if a2.isdigit():
        a2 = ""
    if a2.lower() == a1.lower():
        a2 = ""

    return a2, a1


@dataclass(slots=True)
class PlaceRecord:
    name: str
    admin2: str
    admin1: str
    country: str
    cc: str
    lat: float
    lon: float

    def __post_init__(self) -> None:
        a2, a1 = _normalize_admin(self.name, self.admin2, self.admin1, self.country, self.cc)
        self.admin2 = a2
        self.admin1 = a1

    @property
    def formatted_address(self) -> str:
        parts = [self.name]
        if self.admin1 and self.admin1.lower() != self.name.lower() and self.admin1.lower() != self.country.lower():
            parts.append(self.admin1)
        if self.country:
            parts.append(self.country)
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "city": self.name,
            "admin2": self.admin2,
            "admin1": self.admin1,
            "state": self.admin1,
            "country": self.country,
            "cc": self.cc,
            "lat": self.lat,
            "lon": self.lon,
            "formatted_address": self.formatted_address,
        }


# In-memory index of all built-in places
_BUILTIN_RECORDS: list[PlaceRecord] = [
    PlaceRecord(
        name=p[0],
        admin2=p[1],
        admin1=p[2],
        country=p[3],
        cc=p[4],
        lat=float(p[5]),
        lon=float(p[6]),
    )
    for p in BUILTIN_PLACES
]

# Fast spatial KDTree index for reverse geocoding
_KDTREE_INSTANCE: Any = None
_KDTREE_RECORDS: list[PlaceRecord] = []


def _ensure_kdtree() -> tuple[Any, list[PlaceRecord]]:
    global _KDTREE_INSTANCE, _KDTREE_RECORDS
    if _KDTREE_INSTANCE is not None:
        return _KDTREE_INSTANCE, _KDTREE_RECORDS

    _KDTREE_RECORDS = list(_BUILTIN_RECORDS)
    if _HAS_SCIPY:
        coords = np.array([[p.lat, p.lon] for p in _KDTREE_RECORDS], dtype=np.float64)
        _KDTREE_INSTANCE = cKDTree(coords)
    return _KDTREE_INSTANCE, _KDTREE_RECORDS


def parse_coordinates(text: str) -> tuple[float, float] | None:
    """Parses decimal lat/lon or DMS coordinates into a float tuple (lat, lon)."""
    clean = text.strip()
    if not clean:
        return None

    # 1. Standard decimal lat, lon: "51.3631, -0.3461"
    m_dec = re.match(
        r"^([+-]?\d{1,2}(?:\.\d+)?)\s*°?\s*,\s*([+-]?\d{1,3}(?:\.\d+)?)\s*°?$",
        clean,
    )
    if m_dec:
        try:
            lat = float(m_dec.group(1))
            lon = float(m_dec.group(2))
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return lat, lon
        except ValueError:
            pass

    # 2. Decimal with compass indicators: "51.3631 N, 0.3461 W"
    m_compass = re.match(
        r"^(\d{1,2}(?:\.\d+)?)\s*°?\s*([NSns])\s*[,; ]\s*(\d{1,3}(?:\.\d+)?)\s*°?\s*([EWew])$",
        clean,
    )
    if m_compass:
        try:
            lat = float(m_compass.group(1)) * (-1.0 if m_compass.group(2).upper() == "S" else 1.0)
            lon = float(m_compass.group(3)) * (-1.0 if m_compass.group(4).upper() == "W" else 1.0)
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return lat, lon
        except ValueError:
            pass

    # 3. DMS: 51° 21' 47" N, 0° 20' 46" W
    dms_regex = (
        r"^(\d{1,2})[°\s]+(\d{1,2})['\s]+(\d{1,2}(?:\.\d+)?)\"?\s*([NSns])"
        r"[,;\s]+"
        r"(\d{1,3})[°\s]+(\d{1,2})['\s]+(\d{1,2}(?:\.\d+)?)\"?\s*([EWew])$"
    )
    m_dms = re.match(dms_regex, clean)
    if m_dms:
        try:
            d_lat = float(m_dms.group(1)) + float(m_dms.group(2)) / 60.0 + float(m_dms.group(3)) / 3600.0
            if m_dms.group(4).upper() == "S":
                d_lat = -d_lat
            d_lon = float(m_dms.group(5)) + float(m_dms.group(6)) / 60.0 + float(m_dms.group(7)) / 3600.0
            if m_dms.group(8).upper() == "W":
                d_lon = -d_lon
            if -90.0 <= d_lat <= 90.0 and -180.0 <= d_lon <= 180.0:
                return d_lat, d_lon
        except ValueError:
            pass

    return None


def search_places_offline(
    query: str,
    limit: int = 12,
    db_engine: Any = None,
) -> list[dict[str, Any]]:
    """
    Sub-2ms prefix & substring matching across place name, county/admin2, state/admin1, and country.
    Ranks exact city prefix matches highest.
    Also queries persistent 'saved_places' in catalog.db if available.
    """
    clean = query.strip()
    if not clean:
        return []

    q_lower = clean.lower()
    matches: list[tuple[int, PlaceRecord]] = []
    seen_addresses: set[str] = set()

    # 1. Search saved_places in catalog.db first (user-cached and online discoveries)
    if db_engine is not None:
        try:
            conn = db_engine.catalog_conn()
            like_pat = f"%{clean}%"
            rows = conn.execute(
                """
                SELECT name, admin2, admin1, country, cc, lat, lon, formatted_address
                FROM saved_places
                WHERE name LIKE ? OR formatted_address LIKE ?
                LIMIT 15;
                """,
                (like_pat, like_pat),
            ).fetchall()
            for r in rows:
                rec = PlaceRecord(
                    name=str(r["name"]),
                    admin2=str(r["admin2"] or ""),
                    admin1=str(r["admin1"] or ""),
                    country=str(r["country"] or ""),
                    cc=str(r["cc"] or ""),
                    lat=float(r["lat"]),
                    lon=float(r["lon"]),
                )
                fmt = rec.formatted_address
                if fmt not in seen_addresses:
                    seen_addresses.add(fmt)
                    score = 0 if rec.name.lower().startswith(q_lower) else 10
                    matches.append((score, rec))
        except Exception:
            pass

    # 2. Search comprehensive built-in gazetteer
    for rec in _BUILTIN_RECORDS:
        fmt = rec.formatted_address
        if fmt in seen_addresses:
            continue

        name_lower = rec.name.lower()
        if name_lower == q_lower:
            score = 0
        elif name_lower.startswith(q_lower):
            score = 1
        elif q_lower in name_lower:
            score = 3
        elif rec.admin2 and q_lower in rec.admin2.lower():
            score = 5
        elif rec.admin1 and q_lower in rec.admin1.lower():
            score = 6
        elif rec.country and q_lower in rec.country.lower():
            score = 8
        elif q_lower in fmt.lower():
            score = 9
        else:
            continue

        seen_addresses.add(fmt)
        matches.append((score, rec))
        if len(matches) >= limit * 3:
            break

    # Sort by score ascending, then by name length
    matches.sort(key=lambda item: (item[0], len(item[1].name), item[1].name))
    return [rec.to_dict() for _, rec in matches[:limit]]


def save_place_to_db(db_engine: Any, place_dict: dict[str, Any]) -> None:
    """Caches a newly discovered place into catalog.db's saved_places table."""
    if db_engine is None:
        return
    name = str(place_dict.get("name") or "").strip()
    if not name:
        return
    lat = float(place_dict.get("lat") or 0.0)
    lon = float(place_dict.get("lon") or 0.0)
    fmt = str(place_dict.get("formatted_address") or f"{name}")

    try:
        import time
        with db_engine.write_lock:
            conn = db_engine.catalog_conn()
            conn.execute(
                """
                INSERT OR IGNORE INTO saved_places (
                    name, admin2, admin1, country, cc, lat, lon, formatted_address, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'nominatim', ?);
                """,
                (
                    name,
                    str(place_dict.get("admin2") or ""),
                    str(place_dict.get("admin1") or ""),
                    str(place_dict.get("country") or ""),
                    str(place_dict.get("cc") or ""),
                    lat,
                    lon,
                    fmt,
                    time.time(),
                ),
            )
            conn.commit()
    except Exception as exc:
        logger.debug("Could not cache saved_place: %s", exc)


def search_places_online(
    query: str,
    db_engine: Any = None,
    timeout: float = 2.0,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """
    Queries OpenStreetMap Nominatim API for places/postcodes/towns not in the local gazetteer,
    and caches discovered places permanently into catalog.db's saved_places table.
    """
    clean = query.strip()
    if not clean or len(clean) < 2:
        return []

    encoded = urllib.parse.quote(clean)
    url = f"https://nominatim.openstreetmap.org/search?format=jsonv2&addressdetails=1&limit={limit}&q={encoded}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PhotoGallery64/1.0 (Desktop Gallery Geotagger)"},
    )

    results: list[dict[str, Any]] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                payload = json.loads(response.read().decode("utf-8"))
                if isinstance(payload, list):
                    for item in payload:
                        addr = item.get("address", {})
                        city = (
                            addr.get("city")
                            or addr.get("town")
                            or addr.get("village")
                            or addr.get("hamlet")
                            or addr.get("suburb")
                            or item.get("name")
                            or clean
                        )
                        admin2 = addr.get("county", "")
                        admin1 = addr.get("state", "")
                        country = addr.get("country", "")
                        cc = str(addr.get("country_code") or "").upper()
                        lat = float(item["lat"])
                        lon = float(item["lon"])

                        rec = PlaceRecord(
                            name=city,
                            admin2=admin2,
                            admin1=admin1,
                            country=country,
                            cc=cc,
                            lat=lat,
                            lon=lon,
                        )
                        p_dict = rec.to_dict()
                        results.append(p_dict)
                        if db_engine is not None:
                            save_place_to_db(db_engine, p_dict)
    except Exception as exc:
        logger.debug("Nominatim online search failed for %s: %s", clean, exc)

    return results


def reverse_geocode(
    lat: float,
    lon: float,
    db_engine: Any = None,
) -> dict[str, Any]:
    """
    Reverse geocodes (lat, lon) coordinates to structured administrative fields:
    city/name, state/admin1, admin2/county, country, cc, and formatted_address.
    Uses reverse_geocoder if installed, or fast KDTree / distance search across 3,500+ built-in places.
    """
    f_lat = float(lat)
    f_lon = float(lon)

    # 1. If reverse_geocoder library is installed, use it as priority
    if _HAS_REVERSE_GEOCODER and _rg is not None:
        try:
            results = _rg.search([(f_lat, f_lon)])
            if results:
                r = results[0]
                name = r.get("name") or "Unknown"
                admin1 = r.get("admin1") or ""
                admin2 = r.get("admin2") or ""
                cc = r.get("cc") or ""
                # Determine country name if available
                country = cc
                for p in _BUILTIN_RECORDS:
                    if p.cc == cc:
                        country = p.country
                        break
                rec = PlaceRecord(
                    name=name,
                    admin2=admin2,
                    admin1=admin1,
                    country=country,
                    cc=cc,
                    lat=f_lat,
                    lon=f_lon,
                )
                return rec.to_dict()
        except Exception as exc:
            logger.debug("reverse_geocoder error: %s", exc)

    # 2. Check saved_places table for closest cached discovery
    if db_engine is not None:
        try:
            conn = db_engine.catalog_conn()
            rows = conn.execute(
                """
                SELECT name, admin2, admin1, country, cc, lat, lon
                FROM saved_places
                ORDER BY ((lat - ?) * (lat - ?) + (lon - ?) * (lon - ?)) ASC
                LIMIT 1;
                """,
                (f_lat, f_lat, f_lon, f_lon),
            ).fetchall()
            if rows:
                r = rows[0]
                d_lat = float(r["lat"]) - f_lat
                d_lon = float(r["lon"]) - f_lon
                # Within ~3 km
                if (d_lat * d_lat + d_lon * d_lon) < 0.001:
                    rec = PlaceRecord(
                        name=str(r["name"]),
                        admin2=str(r["admin2"] or ""),
                        admin1=str(r["admin1"] or ""),
                        country=str(r["country"] or ""),
                        cc=str(r["cc"] or ""),
                        lat=float(r["lat"]),
                        lon=float(r["lon"]),
                    )
                    return rec.to_dict()
        except Exception:
            pass

    # 3. Fast KDTree nearest neighbor on built-in places
    tree, records = _ensure_kdtree()
    if tree is not None and _HAS_SCIPY:
        dist, idx = tree.query([f_lat, f_lon], k=1)
        if 0 <= idx < len(records):
            nearest = records[idx]
            return nearest.to_dict()

    # 4. Pure Python distance fallback
    best_rec = _BUILTIN_RECORDS[0]
    best_dist = 1e9
    for rec in _BUILTIN_RECORDS:
        d = (rec.lat - f_lat) ** 2 + (rec.lon - f_lon) ** 2
        if d < best_dist:
            best_dist = d
            best_rec = rec

    return best_rec.to_dict()


def resolve_location_query(
    query: str,
    allow_network: bool = True,
    db_engine: Any = None,
) -> tuple[float | None, float | None, str]:
    """
    Main entry point for resolving user geotag queries across PhotoGallery64.
    Returns (lat, lon, formatted_address).
    """
    clean = query.strip()
    if not clean:
        return None, None, ""

    # 1. Explicit coordinate check
    coords = parse_coordinates(clean)
    if coords is not None:
        lat, lon = coords
        # Try to resolve reverse name for coordinates
        rev = reverse_geocode(lat, lon, db_engine=db_engine)
        label = rev["formatted_address"] if rev else f"{lat:.4f}°, {lon:.4f}°"
        return lat, lon, label

    # 2. Offline gazetteer match
    offline_matches = search_places_offline(clean, limit=1, db_engine=db_engine)
    if offline_matches:
        top = offline_matches[0]
        return top["lat"], top["lon"], top["formatted_address"]

    # 3. Optional online Nominatim fallback
    if allow_network:
        online_matches = search_places_online(clean, db_engine=db_engine, timeout=2.0, limit=1)
        if online_matches:
            top = online_matches[0]
            return top["lat"], top["lon"], top["formatted_address"]

    # 4. Fallback: preserve location name string without coordinates
    return None, None, clean
