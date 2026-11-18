"""
Offline and Online Geotag Coordinate and Location Resolver for PhotoGallery64.
Parses latitude/longitude coordinates, matches offline world gazetteer cities,
and falls back to optional OpenStreetMap Nominatim queries.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Optional, Tuple

# Comprehensive offline gazetteer of major world cities, capitals, and landmarks
# Format: key -> (lat, lon, canonical_display_name)
OFFLINE_GAZETTEER: dict[str, Tuple[float, float, str]] = {
    # United Kingdom & Ireland
    "london": (51.5074, -0.1278, "London, United Kingdom"),
    "edinburgh": (55.9533, -3.1883, "Edinburgh, United Kingdom"),
    "manchester": (53.4808, -2.2426, "Manchester, United Kingdom"),
    "birmingham": (52.4862, -1.8904, "Birmingham, United Kingdom"),
    "glasgow": (55.8642, -4.2518, "Glasgow, United Kingdom"),
    "liverpool": (53.4084, -2.9916, "Liverpool, United Kingdom"),
    "bristol": (51.4545, -2.5879, "Bristol, United Kingdom"),
    "belfast": (54.5973, -5.9301, "Belfast, United Kingdom"),
    "cardiff": (51.4816, -3.1791, "Cardiff, United Kingdom"),
    "oxford": (51.7520, -1.2577, "Oxford, United Kingdom"),
    "cambridge": (52.2053, 0.1218, "Cambridge, United Kingdom"),
    "dublin": (53.3498, -6.2603, "Dublin, Ireland"),
    "cork": (51.8985, -8.4756, "Cork, Ireland"),
    "galway": (53.2707, -9.0568, "Galway, Ireland"),

    # Western & Central Europe
    "paris": (48.8566, 2.3522, "Paris, France"),
    "nice": (43.7102, 7.2620, "Nice, France"),
    "lyon": (45.7640, 4.8357, "Lyon, France"),
    "marseille": (43.2965, 5.3698, "Marseille, France"),
    "berlin": (52.5200, 13.4050, "Berlin, Germany"),
    "munich": (48.1351, 11.5820, "Munich, Germany"),
    "frankfurt": (50.1109, 8.6821, "Frankfurt, Germany"),
    "hamburg": (53.5511, 9.9937, "Hamburg, Germany"),
    "cologne": (50.9375, 6.9603, "Cologne, Germany"),
    "amsterdam": (52.3676, 4.9041, "Amsterdam, Netherlands"),
    "rotterdam": (51.9244, 4.4777, "Rotterdam, Netherlands"),
    "the hague": (52.0705, 4.3007, "The Hague, Netherlands"),
    "brussels": (50.8503, 4.3517, "Brussels, Belgium"),
    "antwerp": (51.2194, 4.4025, "Antwerp, Belgium"),
    "vienna": (48.2082, 16.3738, "Vienna, Austria"),
    "salzburg": (47.8095, 13.0550, "Salzburg, Austria"),
    "zurich": (47.3769, 8.5417, "Zurich, Switzerland"),
    "geneva": (46.2044, 6.1432, "Geneva, Switzerland"),
    "bern": (46.9480, 7.4474, "Bern, Switzerland"),

    # Southern Europe
    "rome": (41.9028, 12.4964, "Rome, Italy"),
    "milan": (45.4642, 9.1900, "Milan, Italy"),
    "florence": (43.7696, 11.2558, "Florence, Italy"),
    "venice": (45.4408, 12.3155, "Venice, Italy"),
    "naples": (40.8518, 14.2681, "Naples, Italy"),
    "madrid": (40.4168, -3.7038, "Madrid, Spain"),
    "barcelona": (41.3879, 2.1699, "Barcelona, Spain"),
    "seville": (37.3891, -5.9845, "Seville, Spain"),
    "valencia": (39.4699, -0.3763, "Valencia, Spain"),
    "lisbon": (38.7223, -9.1393, "Lisbon, Portugal"),
    "porto": (41.1579, -8.6291, "Porto, Portugal"),
    "athens": (37.9838, 23.7275, "Athens, Greece"),
    "santorini": (36.3932, 25.4615, "Santorini, Greece"),

    # Northern & Eastern Europe
    "stockholm": (59.3293, 18.0686, "Stockholm, Sweden"),
    "oslo": (59.9139, 10.7522, "Oslo, Norway"),
    "copenhagen": (55.6761, 12.5683, "Copenhagen, Denmark"),
    "helsinki": (60.1699, 24.9384, "Helsinki, Finland"),
    "reykjavik": (64.1466, -21.9426, "Reykjavik, Iceland"),
    "prague": (50.0755, 14.4378, "Prague, Czech Republic"),
    "warsaw": (52.2297, 21.0122, "Warsaw, Poland"),
    "krakow": (50.0647, 19.9450, "Krakow, Poland"),
    "budapest": (47.4979, 19.0402, "Budapest, Hungary"),
    "bucharest": (44.4268, 26.1025, "Bucharest, Romania"),
    "istanbul": (41.0082, 28.9784, "Istanbul, Turkey"),

    # North America - United States
    "new york": (40.7128, -74.0060, "New York, NY, United States"),
    "nyc": (40.7128, -74.0060, "New York, NY, United States"),
    "manhattan": (40.7831, -73.9712, "Manhattan, NY, United States"),
    "brooklyn": (40.6782, -73.9442, "Brooklyn, NY, United States"),
    "los angeles": (34.0522, -118.2437, "Los Angeles, CA, United States"),
    "la": (34.0522, -118.2437, "Los Angeles, CA, United States"),
    "san francisco": (37.7749, -122.4194, "San Francisco, CA, United States"),
    "sf": (37.7749, -122.4194, "San Francisco, CA, United States"),
    "chicago": (41.8781, -87.6298, "Chicago, IL, United States"),
    "seattle": (47.6062, -122.3321, "Seattle, WA, United States"),
    "washington dc": (38.9072, -77.0369, "Washington, D.C., United States"),
    "washington": (38.9072, -77.0369, "Washington, D.C., United States"),
    "boston": (42.3601, -71.0589, "Boston, MA, United States"),
    "miami": (25.7617, -80.1918, "Miami, FL, United States"),
    "las vegas": (36.1699, -115.1398, "Las Vegas, NV, United States"),
    "san diego": (32.7157, -117.1611, "San Diego, CA, United States"),
    "austin": (30.2672, -97.7431, "Austin, TX, United States"),
    "dallas": (32.7767, -96.7970, "Dallas, TX, United States"),
    "houston": (29.7604, -95.3698, "Houston, TX, United States"),
    "denver": (39.7392, -104.9903, "Denver, CO, United States"),
    "atlanta": (33.7490, -84.3880, "Atlanta, GA, United States"),
    "philadelphia": (39.9526, -75.1652, "Philadelphia, PA, United States"),
    "phoenix": (33.4484, -112.0740, "Phoenix, AZ, United States"),
    "portland": (45.5152, -122.6784, "Portland, OR, United States"),
    "honolulu": (21.3069, -157.8583, "Honolulu, HI, United States"),
    "hawaii": (19.8968, -155.5828, "Hawaii, United States"),

    # North America - Canada & Mexico
    "toronto": (43.6532, -79.3832, "Toronto, ON, Canada"),
    "vancouver": (49.2827, -123.1207, "Vancouver, BC, Canada"),
    "montreal": (45.5017, -73.5673, "Montreal, QC, Canada"),
    "ottawa": (45.4215, -75.6972, "Ottawa, ON, Canada"),
    "calgary": (51.0447, -114.0719, "Calgary, AB, Canada"),
    "mexico city": (19.4326, -99.1332, "Mexico City, Mexico"),
    "cancun": (21.1619, -86.8515, "Cancun, Mexico"),

    # Asia & Middle East
    "tokyo": (35.6762, 139.6503, "Tokyo, Japan"),
    "kyoto": (35.0116, 135.7681, "Kyoto, Japan"),
    "osaka": (34.6937, 135.5023, "Osaka, Japan"),
    "beijing": (39.9042, 116.4074, "Beijing, China"),
    "shanghai": (31.2304, 121.4737, "Shanghai, China"),
    "hong kong": (22.3193, 114.1694, "Hong Kong"),
    "taipei": (25.0330, 121.5654, "Taipei, Taiwan"),
    "seoul": (37.5665, 126.9780, "Seoul, South Korea"),
    "singapore": (1.3521, 103.8198, "Singapore"),
    "bangkok": (13.7563, 100.5018, "Bangkok, Thailand"),
    "kuala lumpur": (3.1390, 101.6869, "Kuala Lumpur, Malaysia"),
    "jakarta": (-6.2088, 106.8456, "Jakarta, Indonesia"),
    "bali": (-8.3405, 115.0920, "Bali, Indonesia"),
    "manila": (14.5995, 120.9842, "Manila, Philippines"),
    "hanoi": (21.0285, 105.8542, "Hanoi, Vietnam"),
    "mumbai": (19.0760, 72.8777, "Mumbai, India"),
    "delhi": (28.6139, 77.2090, "New Delhi, India"),
    "new delhi": (28.6139, 77.2090, "New Delhi, India"),
    "bengaluru": (12.9716, 77.5946, "Bengaluru, India"),
    "bangalore": (12.9716, 77.5946, "Bengaluru, India"),
    "dubai": (25.2048, 55.2708, "Dubai, United Arab Emirates"),
    "abu dhabi": (24.4539, 54.3773, "Abu Dhabi, United Arab Emirates"),
    "doha": (25.2854, 51.5310, "Doha, Qatar"),
    "riyadh": (24.7136, 46.6753, "Riyadh, Saudi Arabia"),
    "jerusalem": (31.7683, 35.2137, "Jerusalem"),
    "tel aviv": (32.0853, 34.7818, "Tel Aviv, Israel"),

    # Oceania
    "sydney": (-33.8688, 151.2093, "Sydney, Australia"),
    "melbourne": (-37.8136, 144.9631, "Melbourne, Australia"),
    "brisbane": (-27.4698, 153.0251, "Brisbane, Australia"),
    "perth": (-31.9505, 115.8605, "Perth, Australia"),
    "adelaide": (-34.9285, 138.6007, "Adelaide, Australia"),
    "auckland": (-36.8485, 174.7633, "Auckland, New Zealand"),
    "wellington": (-41.2865, 174.7762, "Wellington, New Zealand"),
    "christchurch": (-43.5321, 172.6362, "Christchurch, New Zealand"),

    # South America & Africa
    "sao paulo": (-23.5505, -46.6333, "São Paulo, Brazil"),
    "rio de janeiro": (-22.9068, -43.1729, "Rio de Janeiro, Brazil"),
    "rio": (-22.9068, -43.1729, "Rio de Janeiro, Brazil"),
    "buenos aires": (-34.6037, -58.3816, "Buenos Aires, Argentina"),
    "santiago": (-33.4489, -70.6693, "Santiago, Chile"),
    "lima": (-12.0464, -77.0428, "Lima, Peru"),
    "bogota": (4.7110, -74.0721, "Bogotá, Colombia"),
    "cairo": (30.0444, 31.2357, "Cairo, Egypt"),
    "cape town": (-33.9249, 18.4241, "Cape Town, South Africa"),
    "johannesburg": (-26.2041, 28.0473, "Johannesburg, South Africa"),
    "nairobi": (-1.2921, 36.8219, "Nairobi, Kenya"),
    "casablanca": (33.5731, -7.5898, "Casablanca, Morocco"),

    # World Famous Landmarks
    "eiffel tower": (48.8584, 2.2945, "Eiffel Tower, Paris"),
    "statue of liberty": (40.6892, -74.0445, "Statue of Liberty, New York"),
    "colosseum": (41.8902, 12.4922, "Colosseum, Rome"),
    "big ben": (51.5007, -0.1246, "Big Ben, London"),
    "taj mahal": (27.1751, 78.0421, "Taj Mahal, Agra"),
    "grand canyon": (36.1069, -112.1129, "Grand Canyon, Arizona"),
    "machu picchu": (-13.1631, -72.5450, "Machu Picchu, Peru"),
    "sydney opera house": (-33.8568, 151.2153, "Sydney Opera House, Sydney"),
    "mount everest": (27.9881, 86.9250, "Mount Everest, Nepal/Tibet"),
    "central park": (40.7851, -73.9683, "Central Park, New York"),
}


def parse_coordinates(text: str) -> Optional[Tuple[float, float]]:
    """
    Parses latitude and longitude from a coordinate string.
    Supports formats:
    - Decimal: "51.5074, -0.1278" or "51.5074 -0.1278"
    - Cardinal: "51.5074 N, 0.1278 W" or "33.8688° S, 151.2093° E"
    Returns (lat, lon) within valid Earth bounds, or None.
    """
    clean = text.strip()
    if not clean:
        return None

    # Decimal: 51.5074, -0.1278
    m_dec = re.match(
        r"^([+-]?\d+(?:\.\d+)?)\s*[,;\s]\s*([+-]?\d+(?:\.\d+)?)$", clean
    )
    if m_dec:
        try:
            lat = float(m_dec.group(1))
            lon = float(m_dec.group(2))
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return lat, lon
        except ValueError:
            pass

    # Cardinal: 51.5074 N, 0.1278 W or 51.5074°N, 0.1278°W
    m_card = re.match(
        r"^(\d+(?:\.\d+)?)\s*°?\s*([NSns])\s*[,;\s]\s*(\d+(?:\.\d+)?)\s*°?\s*([EWew])$",
        clean,
    )
    if m_card:
        try:
            lat = float(m_card.group(1)) * (-1.0 if m_card.group(2).upper() == "S" else 1.0)
            lon = float(m_card.group(3)) * (-1.0 if m_card.group(4).upper() == "W" else 1.0)
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return lat, lon
        except ValueError:
            pass

    return None


def query_nominatim(
    place_name: str, timeout: float = 1.5
) -> Optional[Tuple[float, float, str]]:
    """
    Performs optional online OpenStreetMap Nominatim geocoding lookup.
    Safe with short timeout and exception handling so offline usage never blocks or fails.
    """
    place_clean = place_name.strip()
    if not place_clean or len(place_clean) < 2:
        return None

    encoded = urllib.parse.quote(place_clean)
    url = (
        f"https://nominatim.openstreetmap.org/search"
        f"?q={encoded}&format=jsonv2&addressdetails=1&namedetails=1&accept-language=en&limit=1"
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "PhotoGallery64/1.0 (Desktop Gallery Geotagger)",
            "Accept-Language": "en",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                payload = json.loads(response.read().decode("utf-8"))
                if isinstance(payload, list) and payload:
                    from src.core.geotag_resolver import _extract_english_nominatim_place

                    top = payload[0]
                    rec = _extract_english_nominatim_place(top, place_clean)
                    display_name = rec.formatted_address or str(top.get("display_name") or place_clean)
                    return rec.lat, rec.lon, display_name
    except Exception:
        # Gracefully swallow timeout, network errors, or rate limits
        return None

    return None


def resolve_location_query(
    query: str, allow_network: bool = True
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Main entry point for resolving user geotag queries.
    1. Checks if string contains explicit coordinates -> (lat, lon, "lat°, lon°").
    2. Checks offline gazetteer for major cities/landmarks -> (lat, lon, canonical_name).
    3. Checks online OpenStreetMap Nominatim lookup if allow_network=True.
    4. Fallback: returns (None, None, query) preserving place name even without GPS.
    """
    clean = query.strip()
    if not clean:
        return None, None, ""

    # 1. Coordinate check
    coords = parse_coordinates(clean)
    if coords is not None:
        lat, lon = coords
        label = f"{lat:.4f}°, {lon:.4f}°"
        return lat, lon, label

    # 2. Offline gazetteer check
    normalized_key = re.sub(r"[^\w\s]", "", clean.lower()).strip()
    if normalized_key in OFFLINE_GAZETTEER:
        lat, lon, display = OFFLINE_GAZETTEER[normalized_key]
        return lat, lon, display

    # Also check if query starts with or contains city name
    for key, (lat, lon, display) in OFFLINE_GAZETTEER.items():
        if normalized_key == key or normalized_key.startswith(key + " "):
            return lat, lon, display

    # 3. Optional online nominatim lookup
    if allow_network:
        res = query_nominatim(clean, timeout=1.5)
        if res is not None:
            return res[0], res[1], res[2]

    # 4. Fallback: preserve location name without coordinates
    return None, None, clean
