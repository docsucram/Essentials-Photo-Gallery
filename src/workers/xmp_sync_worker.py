"""
Background XMP & EXIF Metadata Synchronization Daemon (src/workers/xmp_sync_worker.py).
Drains `metadata_dirty = 1` rows from catalog.db and writes WLPG-compliant XMP
(lr:hierarchicalSubject, MicrosoftPhoto:LastKeywordXMP, dc:subject, xmp:Rating,
dc:description, exif:DateTimeOriginal, tiff:Orientation, and static-image MP:RegionInfo)
losslessly without re-encoding image pixel data.
"""

from __future__ import annotations

import datetime
import os
import stat
import struct
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, _epoch_to_datetime_utc

XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"
EXIF_HEADER = b"Exif\x00\x00"

# XML Namespaces used by WLPG / Adobe Lightroom / MWG / Microsoft Photo / digiKam
NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "lr": "http://ns.adobe.com/lightroom/1.0/",
    "MicrosoftPhoto": "http://ns.microsoft.com/photo/1.0/",
    "MP": "http://ns.microsoft.com/photo/1.2/",
    "MPRI": "http://ns.microsoft.com/photo/1.2/t/RegionInfo#",
    "MPReg": "http://ns.microsoft.com/photo/1.2/t/Region#",
    "exif": "http://ns.adobe.com/exif/1.0/",
    "tiff": "http://ns.adobe.com/tiff/1.0/",
    "Iptc4xmpCore": "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/",
    "mwg-rs": "http://www.metadataworkinggroup.com/schemas/regions/",
    "stDim": "http://ns.adobe.com/xap/1.0/sType/Dimensions#",
    "stArea": "http://ns.adobe.com/xmp/sType/Area#",
    "digiKam": "http://www.digikam.org/ns/1.0/",
}

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)

from src.imaging.image_loader import RAW_DOT_EXT_SET as RAW_EXT_SET


def _rating_to_microsoft_percent(rating: int) -> int:
    mapping = {0: 0, 1: 1, 2: 25, 3: 50, 4: 75, 5: 99}
    return mapping.get(int(rating), 0)


_RECENTLY_WRITTEN: dict[str, tuple[int, int, float]] = {}  # normpath -> (size, mtime, write_timestamp)
_RECENTLY_WRITTEN_LOCK = threading.Lock()


def record_recently_written(file_path: Path | str, size: int, mtime: int) -> None:
    """Records that a file was written by the metadata sync daemon to prevent watcher loops."""
    norm = os.path.normcase(os.path.normpath(str(file_path)))
    now = time.time()
    with _RECENTLY_WRITTEN_LOCK:
        _RECENTLY_WRITTEN[norm] = (size, mtime, now)
        for k, (_, _, ts) in list(_RECENTLY_WRITTEN.items()):
            if now - ts > 30.0:
                _RECENTLY_WRITTEN.pop(k, None)


def is_recently_written(
    file_path: Path | str,
    size: int | None = None,
    mtime: int | None = None,
    window_secs: float = 5.0,
) -> bool:
    """Checks whether a file was written by the sync daemon within the last window_secs."""
    norm = os.path.normcase(os.path.normpath(str(file_path)))
    with _RECENTLY_WRITTEN_LOCK:
        entry = _RECENTLY_WRITTEN.get(norm)
        if entry is None:
            return False
        rec_size, rec_mtime, ts = entry
        if time.time() - ts > window_secs:
            return False
        if size is not None and rec_size != size:
            return False
        if mtime is not None and rec_mtime != mtime:
            return False
        return True


def build_wlpg_xmp_packet(media_detail: dict[str, Any]) -> bytes:
    """
    Constructs a standards-compliant WLPG / Lightroom / digiKam / MWG XMP packet containing:
    - xmp:Rating & MicrosoftPhoto:Rating (0..5 & 0, 1, 25, 50, 75, 99)
    - tiff:Orientation & exif:DateTimeOriginal
    - dc:description & dc:title (caption), dc:creator (author)
    - Iptc4xmpCore:Location (location_name) & exif:GPSLatitude/Longitude
    - dc:subject (leaf names), lr:hierarchicalSubject (|), MicrosoftPhoto:LastKeywordXMP (/), digiKam:TagsList (/)
    - MP:RegionInfo (top-left) & mwg-rs:Regions (center-point) strictly for is_video == 0
    """
    rating = int(media_detail.get("rating") or 0)
    ms_rating = _rating_to_microsoft_percent(rating)
    orientation = int(media_detail.get("orientation") or 1)
    caption = str(media_detail.get("caption") or "")
    author = str(media_detail.get("author") or "").strip()
    location_name = str(media_detail.get("location_name") or "").strip()
    lat = media_detail.get("lat")
    lon = media_detail.get("lon")
    is_video = bool(media_detail.get("is_video"))
    date_taken = media_detail.get("date_taken")

    if date_taken is not None:
        dt = _epoch_to_datetime_utc(int(date_taken))
        iso_dt = dt.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        iso_dt = ""

    raw_tags = media_detail.get("tags")
    if isinstance(raw_tags, list) and raw_tags and isinstance(raw_tags[0], str):
        full_paths = [str(x) for x in raw_tags]
        leaf_names = [str(x).split("/")[-1] for x in raw_tags]
    elif isinstance(raw_tags, list) and raw_tags and isinstance(raw_tags[0], dict):
        full_paths = [str(t["full_path"]) for t in raw_tags if t.get("full_path")]
        leaf_names = [str(t["name"]) for t in raw_tags if t.get("name")]
    elif "hierarchical_tags" in media_detail:
        ht = media_detail["hierarchical_tags"] or []
        full_paths = [str(x) for x in ht]
        leaf_names = [str(x).split("/")[-1] for x in ht]
    else:
        full_paths = []
        leaf_names = []

    # Only confirmed (state = 1) faces are written to permanent file metadata
    people: list[dict[str, Any]] = (
        [p for p in media_detail.get("people", []) if not p.get("is_unconfirmed")]
        if not is_video
        else []
    )

    # Build RDF Bag / Seq sections
    dc_subject_items = "\n".join(
        f"            <rdf:li>{xml_escape(name)}</rdf:li>" for name in leaf_names
    )
    lr_hier_items = "\n".join(
        f"            <rdf:li>{xml_escape(fp.replace('/', '|'))}</rdf:li>"
        for fp in full_paths
    )
    ms_kw_items = "\n".join(
        f"            <rdf:li>{xml_escape(fp)}</rdf:li>" for fp in full_paths
    )
    digikam_items = "\n".join(
        f"            <rdf:li>{xml_escape(fp)}</rdf:li>" for fp in full_paths
    )

    creator_xml = (
        f"""\n      <dc:creator>
        <rdf:Seq>
          <rdf:li>{xml_escape(author)}</rdf:li>
        </rdf:Seq>
      </dc:creator>"""
        if author
        else ""
    )

    location_xml = (
        f"\n      <Iptc4xmpCore:Location>{xml_escape(location_name)}</Iptc4xmpCore:Location>"
        if location_name
        else ""
    )

    gps_xml = ""
    if lat is not None and lon is not None:
        try:
            f_lat = float(lat)
            f_lon = float(lon)
            gps_xml = (
                f"\n      <exif:GPSLatitude>{f_lat:.6f}</exif:GPSLatitude>"
                f"\n      <exif:GPSLongitude>{f_lon:.6f}</exif:GPSLongitude>"
            )
        except (ValueError, TypeError):
            pass

    regions_xml = ""
    if not is_video and people:
        seen_people: set[tuple[str, float, float, float, float]] = set()
        deduped_people: list[dict[str, Any]] = []
        for p in people:
            pname = str(p.get("name") or "").strip()
            if not pname:
                continue
            if "rect" in p and isinstance(p["rect"], (tuple, list)) and len(p["rect"]) == 4:
                rx, ry, rw, rh = (float(v) for v in p["rect"])
            else:
                rx = float(p.get("rect_x", 0.0))
                ry = float(p.get("rect_y", 0.0))
                rw = float(p.get("rect_w", 0.0))
                rh = float(p.get("rect_h", 0.0))
            key = (pname.lower(), round(rx, 3), round(ry, 3), round(rw, 3), round(rh, 3))
            if key not in seen_people:
                seen_people.add(key)
                deduped_people.append(p)
        people = deduped_people

        mp_region_li_list = []
        mwg_region_li_list = []
        for p in people:
            pname = xml_escape(str(p["name"]))
            if "rect" in p and isinstance(p["rect"], (tuple, list)) and len(p["rect"]) == 4:
                rx, ry, rw, rh = (float(v) for v in p["rect"])
            else:
                rx = float(p.get("rect_x", 0.0))
                ry = float(p.get("rect_y", 0.0))
                rw = float(p.get("rect_w", 0.0))
                rh = float(p.get("rect_h", 0.0))

            # Microsoft Photo Region: top-left (x, y, w, h)
            rect_str = f"{rx:.6f}, {ry:.6f}, {rw:.6f}, {rh:.6f}"
            mp_region_li_list.append(
                f"""              <rdf:li rdf:parseType="Resource">
                <MPReg:PersonDisplayName>{pname}</MPReg:PersonDisplayName>
                <MPReg:Rectangle>{rect_str}</MPReg:Rectangle>
              </rdf:li>"""
            )

            # MWG Region: center-point (cx, cy, w, h)
            cx = rx + (rw / 2.0)
            cy = ry + (rh / 2.0)
            mwg_region_li_list.append(
                f"""              <rdf:li rdf:parseType="Resource">
                <mwg-rs:Name>{pname}</mwg-rs:Name>
                <mwg-rs:Type>Face</mwg-rs:Type>
                <mwg-rs:Area stArea:x="{cx:.6f}" stArea:y="{cy:.6f}" stArea:w="{rw:.6f}" stArea:h="{rh:.6f}" stArea:unit="normalized"/>
              </rdf:li>"""
            )

        disp_w = int(media_detail.get("width") or 0)
        disp_h = int(media_detail.get("height") or 0)
        applied_dim_xml = ""
        if disp_w > 0 and disp_h > 0:
            applied_dim_xml = f"""\n        <mwg-rs:AppliedToDimensions stDim:w="{disp_w}" stDim:h="{disp_h}" stDim:unit="pixel"/>"""

        mp_regions_joined = "\n".join(mp_region_li_list)
        mwg_regions_joined = "\n".join(mwg_region_li_list)
        regions_xml = f"""
      <MP:RegionInfo rdf:parseType="Resource">
        <MPRI:Regions>
          <rdf:Bag>
{mp_regions_joined}
          </rdf:Bag>
        </MPRI:Regions>
      </MP:RegionInfo>
      <mwg-rs:Regions rdf:parseType="Resource">{applied_dim_xml}
        <mwg-rs:RegionList>
          <rdf:Bag>
{mwg_regions_joined}
          </rdf:Bag>
        </mwg-rs:RegionList>
      </mwg-rs:Regions>"""

    date_attr = f' exif:DateTimeOriginal="{iso_dt}"' if iso_dt else ""

    xmp_str = f"""<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:dc="http://purl.org/dc/elements/1.1/"
        xmlns:xmp="http://ns.adobe.com/xap/1.0/"
        xmlns:lr="http://ns.adobe.com/lightroom/1.0/"
        xmlns:MicrosoftPhoto="http://ns.microsoft.com/photo/1.0/"
        xmlns:MP="http://ns.microsoft.com/photo/1.2/"
        xmlns:MPRI="http://ns.microsoft.com/photo/1.2/t/RegionInfo#"
        xmlns:MPReg="http://ns.microsoft.com/photo/1.2/t/Region#"
        xmlns:exif="http://ns.adobe.com/exif/1.0/"
        xmlns:tiff="http://ns.adobe.com/tiff/1.0/"
        xmlns:Iptc4xmpCore="http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/"
        xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/"
        xmlns:stDim="http://ns.adobe.com/xap/1.0/sType/Dimensions#"
        xmlns:stArea="http://ns.adobe.com/xmp/sType/Area#"
        xmlns:digiKam="http://www.digikam.org/ns/1.0/"
        xmp:Rating="{rating}"
        MicrosoftPhoto:Rating="{ms_rating}"
        tiff:Orientation="{orientation}"{date_attr}>
      <dc:description>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">{xml_escape(caption)}</rdf:li>
        </rdf:Alt>
      </dc:description>
      <dc:title>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">{xml_escape(caption)}</rdf:li>
        </rdf:Alt>
      </dc:title>{creator_xml}{location_xml}{gps_xml}
      <dc:subject>
        <rdf:Bag>
{dc_subject_items}
        </rdf:Bag>
      </dc:subject>
      <lr:hierarchicalSubject>
        <rdf:Bag>
{lr_hier_items}
        </rdf:Bag>
      </lr:hierarchicalSubject>
      <MicrosoftPhoto:LastKeywordXMP>
        <rdf:Bag>
{ms_kw_items}
        </rdf:Bag>
      </MicrosoftPhoto:LastKeywordXMP>
      <digiKam:TagsList>
        <rdf:Seq>
{digikam_items}
        </rdf:Seq>
      </digiKam:TagsList>{regions_xml}
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
    return xmp_str.encode("utf-8")


def transform_box_raw_to_display(
    rect: tuple[float, float, float, float],
    orientation: int,
) -> tuple[float, float, float, float]:
    """
    Transforms normalized top-left (x, y, w, h) from raw unrotated sensor space
    to display space according to EXIF Orientation:
      - Orientation == 6 (90° CW): (1.0 - (y + h), x, h, w)
      - Orientation == 8 (270° CW / 90° CCW): (y, 1.0 - (x + w), h, w)
      - Orientation == 3 (180°): (1.0 - (x + w), 1.0 - (y + h), w, h)
      - Orientation == 5: (y, x, h, w)
      - Orientation == 7: (1.0 - (y + h), 1.0 - (x + w), h, w)
      - Orientation == 2: (1.0 - (x + w), y, w, h)
      - Orientation == 4: (x, 1.0 - (y + h), w, h)
    """
    x, y, w, h = rect
    if orientation == 6:  # 90° CW
        nx = 1.0 - (y + h)
        ny = x
        nw = h
        nh = w
    elif orientation == 8:  # 270° CW / 90° CCW
        nx = y
        ny = 1.0 - (x + w)
        nw = h
        nh = w
    elif orientation == 3:  # 180°
        nx = 1.0 - (x + w)
        ny = 1.0 - (y + h)
        nw = w
        nh = h
    elif orientation == 5:  # Transpose (flip horizontal + 90° CCW)
        nx = y
        ny = x
        nw = h
        nh = w
    elif orientation == 7:  # Transverse (flip horizontal + 90° CW)
        nx = 1.0 - (y + h)
        ny = 1.0 - (x + w)
        nw = h
        nh = w
    elif orientation == 2:  # Flip horizontal
        nx = 1.0 - (x + w)
        ny = y
        nw = w
        nh = h
    elif orientation == 4:  # Flip vertical
        nx = x
        ny = 1.0 - (y + h)
        nw = w
        nh = h
    else:
        return (x, y, w, h)

    return (
        max(0.0, min(1.0, round(nx, 6))),
        max(0.0, min(1.0, round(ny, 6))),
        max(0.0, min(1.0 - nx, round(nw, 6))),
        max(0.0, min(1.0 - ny, round(nh, 6))),
    )


def parse_xmp_packet_bytes(
    xmp_bytes: bytes,
    image_width: int | None = None,
    image_height: int | None = None,
    raw_width: int | None = None,
    raw_height: int | None = None,
    orientation: int | None = None,
) -> dict[str, Any]:
    """Parses WLPG / Lightroom / digiKam / MWG XMP XML bytes into structured metadata fields."""
    result: dict[str, Any] = {
        "rating": None,
        "orientation": None,
        "date_taken": None,
        "caption": None,
        "author": None,
        "location_name": None,
        "lat": None,
        "lon": None,
        "hierarchical_tags": [],
        "people": [],
        "applied_to_dimensions": None,
    }
    try:
        start_idx = xmp_bytes.find(b"<x:xmpmeta")
        end_idx = xmp_bytes.find(b"</x:xmpmeta>")
        if start_idx == -1 or end_idx == -1:
            return result
        xml_slice = xmp_bytes[start_idx : end_idx + len(b"</x:xmpmeta>")]
        root = ET.fromstring(xml_slice)

        raw_ms_rating = None
        for desc in root.findall(".//rdf:Description", NS):
            rating_attr = desc.attrib.get(f"{{{NS['xmp']}}}Rating")
            if rating_attr is not None:
                try:
                    result["rating"] = max(0, min(5, int(rating_attr)))
                except ValueError:
                    pass

            if raw_ms_rating is None:
                ms_attr = desc.attrib.get(f"{{{NS['MicrosoftPhoto']}}}Rating")
                if ms_attr is not None:
                    raw_ms_rating = ms_attr

            orient_attr = desc.attrib.get(f"{{{NS['tiff']}}}Orientation")
            if orient_attr is not None:
                try:
                    result["orientation"] = int(orient_attr)
                except ValueError:
                    pass

            dt_attr = desc.attrib.get(f"{{{NS['exif']}}}DateTimeOriginal")
            if dt_attr:
                result["date_taken"] = _parse_iso_or_exif_datetime(dt_attr)

        # Fallback element lookup for rating if not present on rdf:Description attributes
        if result["rating"] is None:
            r_el = root.find(".//xmp:Rating", NS)
            if r_el is not None and r_el.text:
                try:
                    result["rating"] = max(0, min(5, int(r_el.text.strip())))
                except ValueError:
                    pass

        if result["rating"] is None:
            if raw_ms_rating is None:
                ms_el = root.find(".//MicrosoftPhoto:Rating", NS)
                if ms_el is not None and ms_el.text:
                    raw_ms_rating = ms_el.text.strip()
            if raw_ms_rating is not None:
                try:
                    pct = int(raw_ms_rating)
                    if pct >= 90:
                        result["rating"] = 5
                    elif pct >= 65:
                        result["rating"] = 4
                    elif pct >= 40:
                        result["rating"] = 3
                    elif pct >= 15:
                        result["rating"] = 2
                    elif pct >= 1:
                        result["rating"] = 1
                    else:
                        result["rating"] = 0
                except ValueError:
                    pass

        # Caption (prefer dc:description, fallback to dc:title)
        cap_el = root.find(".//dc:description/rdf:Alt/rdf:li", NS)
        if cap_el is not None and cap_el.text is not None:
            result["caption"] = cap_el.text
        if not result["caption"]:
            title_el = root.find(".//dc:title/rdf:Alt/rdf:li", NS)
            if title_el is not None and title_el.text is not None:
                result["caption"] = title_el.text

        # Author / Creator
        author_el = root.find(".//dc:creator/rdf:Seq/rdf:li", NS)
        if author_el is not None and author_el.text:
            result["author"] = author_el.text.strip()

        # Location name
        loc_el = root.find(".//Iptc4xmpCore:Location", NS)
        if loc_el is not None and loc_el.text:
            result["location_name"] = loc_el.text.strip()

        # GPS lat / lon
        lat_el = root.find(".//exif:GPSLatitude", NS)
        lon_el = root.find(".//exif:GPSLongitude", NS)
        if lat_el is not None and lat_el.text:
            try:
                result["lat"] = float(lat_el.text.strip())
            except ValueError:
                pass
        if lon_el is not None and lon_el.text:
            try:
                result["lon"] = float(lon_el.text.strip())
            except ValueError:
                pass

        # Hierarchical tags (prefer MicrosoftPhoto:LastKeywordXMP or lr:hierarchicalSubject or digiKam:TagsList)
        tags_found: list[str] = []
        for li in root.findall(".//MicrosoftPhoto:LastKeywordXMP/rdf:Bag/rdf:li", NS):
            if li.text and li.text.strip():
                tags_found.append(li.text.strip().replace("|", "/"))
        if not tags_found:
            for li in root.findall(".//lr:hierarchicalSubject/rdf:Bag/rdf:li", NS):
                if li.text and li.text.strip():
                    tags_found.append(li.text.strip().replace("|", "/"))
        if not tags_found:
            for seq_or_bag in ("Seq", "Bag"):
                for li in root.findall(f".//digiKam:TagsList/rdf:{seq_or_bag}/rdf:li", NS):
                    if li.text and li.text.strip():
                        tags_found.append(li.text.strip().replace("|", "/"))
        if not tags_found:
            for li in root.findall(".//dc:subject/rdf:Bag/rdf:li", NS):
                if li.text and li.text.strip():
                    tags_found.append(li.text.strip())
        result["hierarchical_tags"] = tags_found

        # Face regions (MP:RegionInfo - Top-Left coordinates)
        mp_people: list[dict[str, Any]] = []
        for reg_li in root.findall(".//MP:RegionInfo/MPRI:Regions/rdf:Bag/rdf:li", NS):
            name_el = reg_li.find("MPReg:PersonDisplayName", NS)
            rect_el = reg_li.find("MPReg:Rectangle", NS)
            pname = (
                name_el.text.strip()
                if (name_el is not None and name_el.text)
                else reg_li.attrib.get(f"{{{NS['MPReg']}}}PersonDisplayName", "").strip()
            )
            rect_txt = (
                rect_el.text.strip()
                if (rect_el is not None and rect_el.text)
                else reg_li.attrib.get(f"{{{NS['MPReg']}}}Rectangle", "").strip()
            )
            if pname:
                coords = (0.0, 0.0, 0.0, 0.0)
                if rect_txt:
                    try:
                        parts = [float(x.strip()) for x in rect_txt.split(",")]
                        if len(parts) == 4:
                            coords = (parts[0], parts[1], parts[2], parts[3])
                    except ValueError:
                        pass
                mp_people.append({"name": pname, "rect": coords})

        # mwg-rs:AppliedToDimensions check
        applied_w: int | None = None
        applied_h: int | None = None
        dim_el = root.find(".//mwg-rs:Regions/mwg-rs:AppliedToDimensions", NS)
        if dim_el is not None:
            w_val = dim_el.attrib.get(f"{{{NS['stDim']}}}w") or dim_el.attrib.get("stDim:w") or dim_el.attrib.get("w")
            h_val = dim_el.attrib.get(f"{{{NS['stDim']}}}h") or dim_el.attrib.get("stDim:h") or dim_el.attrib.get("h")
            if w_val is None:
                sub_w = dim_el.find("stDim:w", NS)
                if sub_w is not None and sub_w.text:
                    w_val = sub_w.text.strip()
            if h_val is None:
                sub_h = dim_el.find("stDim:h", NS)
                if sub_h is not None and sub_h.text:
                    h_val = sub_h.text.strip()
            if w_val and h_val:
                try:
                    applied_w = int(float(w_val))
                    applied_h = int(float(h_val))
                except ValueError:
                    pass
        if applied_w and applied_h:
            result["applied_to_dimensions"] = (applied_w, applied_h)

        # mwg-rs:Regions (Center-Point coordinates)
        mwg_people: list[dict[str, Any]] = []
        for seq_or_bag in ("Bag", "Seq"):
            for mwg_li in root.findall(f".//mwg-rs:Regions/mwg-rs:RegionList/rdf:{seq_or_bag}/rdf:li", NS):
                name_el = mwg_li.find("mwg-rs:Name", NS)
                pname = (
                    name_el.text.strip()
                    if (name_el is not None and name_el.text)
                    else mwg_li.attrib.get(f"{{{NS['mwg-rs']}}}Name", "").strip()
                )
                area_el = mwg_li.find("mwg-rs:Area", NS)
                if area_el is None:
                    area_el = mwg_li

                cx = area_el.attrib.get(f"{{{NS['stArea']}}}x") or area_el.attrib.get("stArea:x") or area_el.attrib.get("x")
                cy = area_el.attrib.get(f"{{{NS['stArea']}}}y") or area_el.attrib.get("stArea:y") or area_el.attrib.get("y")
                w = area_el.attrib.get(f"{{{NS['stArea']}}}w") or area_el.attrib.get("stArea:w") or area_el.attrib.get("w")
                h = area_el.attrib.get(f"{{{NS['stArea']}}}h") or area_el.attrib.get("stArea:h") or area_el.attrib.get("h")

                if cx is None:
                    el = area_el.find("stArea:x", NS)
                    if el is not None and el.text:
                        cx = el.text.strip()
                if cy is None:
                    el = area_el.find("stArea:y", NS)
                    if el is not None and el.text:
                        cy = el.text.strip()
                if w is None:
                    el = area_el.find("stArea:w", NS)
                    if el is not None and el.text:
                        w = el.text.strip()
                if h is None:
                    el = area_el.find("stArea:h", NS)
                    if el is not None and el.text:
                        h = el.text.strip()

                if pname:
                    coords = (0.0, 0.0, 0.0, 0.0)
                    if cx is not None and cy is not None and w is not None and h is not None:
                        try:
                            f_cx = float(cx)
                            f_cy = float(cy)
                            f_w = float(w)
                            f_h = float(h)
                            rx = max(0.0, f_cx - (f_w / 2.0))
                            ry = max(0.0, f_cy - (f_h / 2.0))
                            rw = min(1.0 - rx, f_w)
                            rh = min(1.0 - ry, f_h)
                            coords = (rx, ry, rw, rh)
                        except ValueError:
                            pass
                    mwg_people.append({"name": pname, "rect": coords})

        # Decision rule: Prefer mwg-rs:Regions over MP:RegionInfo.
        # Fall back to MP:RegionInfo only if mwg-rs:Regions is absent or empty.
        effective_orient = orientation or result.get("orientation") or 1
        disp_w = image_width
        disp_h = image_height

        if mwg_people:
            # Check if AppliedToDimensions indicates raw unrotated sensor space
            is_raw_space = False
            has_applied_dim = bool(applied_w and applied_h)
            is_display_space = False
            if has_applied_dim and disp_w and disp_h:
                if (applied_w, applied_h) == (disp_w, disp_h):
                    is_display_space = True

            if not is_display_space:
                if applied_w and applied_h and raw_width and raw_height:
                    if (applied_w, applied_h) == (raw_width, raw_height) and (
                        disp_w != raw_width or disp_h != raw_height or effective_orient in (3, 5, 6, 7, 8)
                    ):
                        if (applied_w, applied_h) != (disp_w, disp_h):
                            is_raw_space = True
                elif (
                    applied_w and applied_h and disp_w and disp_h
                    and (applied_w, applied_h) == (disp_h, disp_w)
                    and effective_orient in (5, 6, 7, 8)
                ):
                    is_raw_space = True
                elif not has_applied_dim and mp_people and effective_orient in (2, 3, 4, 5, 6, 7, 8):
                    # Only if AppliedToDimensions is completely absent, check if unrotated MP:RegionInfo was copied
                    for p_mwg in mwg_people:
                        for p_mp in mp_people:
                            if (
                                abs(p_mwg["rect"][0] - p_mp["rect"][0]) < 0.02
                                and abs(p_mwg["rect"][1] - p_mp["rect"][1]) < 0.02
                                and abs(p_mwg["rect"][2] - p_mp["rect"][2]) < 0.02
                                and abs(p_mwg["rect"][3] - p_mp["rect"][3]) < 0.02
                            ):
                                is_raw_space = True
                                break
                        if is_raw_space:
                            break

            if is_raw_space and effective_orient in (2, 3, 4, 5, 6, 7, 8):
                transformed_mwg = []
                for p in mwg_people:
                    transformed_mwg.append(
                        {
                            "name": p["name"],
                            "rect": transform_box_raw_to_display(p["rect"], effective_orient),
                        }
                    )
                result["people"] = transformed_mwg
            else:
                result["people"] = mwg_people
        elif mp_people:
            # MP:RegionInfo is written in raw unrotated sensor coordinates by WLPG.
            # If the image has EXIF orientation, transform it to display space.
            if effective_orient in (2, 3, 4, 5, 6, 7, 8):
                transformed_mp = []
                for p in mp_people:
                    transformed_mp.append(
                        {
                            "name": p["name"],
                            "rect": transform_box_raw_to_display(p["rect"], effective_orient),
                        }
                    )
                result["people"] = transformed_mp
            else:
                result["people"] = mp_people
        else:
            result["people"] = []
    except Exception:
        pass
    return result


def _parse_iso_or_exif_datetime(dt_str: str) -> int | None:
    s = dt_str.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.datetime.strptime(s[:19], fmt).replace(
                tzinfo=datetime.timezone.utc
            )
            if dt.year < 1826 or dt.year > 2100:
                return None
            epoch_zero = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
            ts = int((dt - epoch_zero).total_seconds())
            # Reject unset QuickTime/MP4 epoch zero (1904-01-01, -2082844800) and Unix epoch zero (1970-01-01 / 1969-12-31)
            if abs(ts - (-2082844800)) <= 172800 or abs(ts) <= 86400:
                return None
            return ts
        except ValueError:
            continue
    return None


def read_xmp_from_file(
    file_path: Path | str,
    image_width: int | None = None,
    image_height: int | None = None,
    raw_width: int | None = None,
    raw_height: int | None = None,
    orientation: int | None = None,
) -> dict[str, Any]:
    """Reads XMP metadata from a JPEG, RAW sidecar, or header slice without loading full pixel data."""
    p = Path(file_path)
    # Check sidecars (.ext.xmp or .xmp)
    sidecar_candidates = [
        p.with_suffix(p.suffix + ".xmp"),
        p.with_suffix(".xmp"),
    ]
    for sc in sidecar_candidates:
        if sc.is_file():
            try:
                data = sc.read_bytes()
                if b"<x:xmpmeta" in data:
                    res = parse_xmp_packet_bytes(
                        data,
                        image_width=image_width,
                        image_height=image_height,
                        raw_width=raw_width,
                        raw_height=raw_height,
                        orientation=orientation,
                    )
                    if res.get("people") or res.get("hierarchical_tags") or res.get("rating") is not None:
                        return res
            except Exception:
                pass

    try:
        with p.open("rb") as f:
            head = f.read(262144)  # First 256 KB covers APP1 XMP segments
        if b"<x:xmpmeta" in head:
            return parse_xmp_packet_bytes(
                head,
                image_width=image_width,
                image_height=image_height,
                raw_width=raw_width,
                raw_height=raw_height,
                orientation=orientation,
            )
    except Exception:
        pass
    return {
        "rating": None,
        "orientation": None,
        "date_taken": None,
        "caption": None,
        "author": None,
        "location_name": None,
        "lat": None,
        "lon": None,
        "hierarchical_tags": [],
        "people": [],
    }


def _patch_exif_orientation_in_tiff_header(
    exif_payload: bytes, new_orientation: int
) -> bytes:
    """
    Losslessly patches the IFD0 Orientation tag (0x0112) inside an existing JPEG EXIF APP1 payload
    if present, without altering any other EXIF/MakerNote tag or offset.
    """
    if not exif_payload.startswith(EXIF_HEADER) or len(exif_payload) < 16:
        return exif_payload
    tiff = bytearray(exif_payload[6:])
    endian_mark = bytes(tiff[:2])
    if endian_mark == b"II":
        endian = "<"
    elif endian_mark == b"MM":
        endian = ">"
    else:
        return exif_payload

    try:
        ifd0_offset = struct.unpack(f"{endian}I", tiff[4:8])[0]
        if ifd0_offset + 2 > len(tiff):
            return exif_payload
        num_entries = struct.unpack(f"{endian}H", tiff[ifd0_offset : ifd0_offset + 2])[0]
        pos = ifd0_offset + 2
        for _ in range(num_entries):
            if pos + 12 > len(tiff):
                break
            tag_id, field_type, count = struct.unpack(f"{endian}HHI", tiff[pos : pos + 8])
            if tag_id == 0x0112 and field_type == 3 and count == 1:
                struct.pack_into(f"{endian}H", tiff, pos + 8, int(new_orientation))
                return EXIF_HEADER + bytes(tiff)
            pos += 12
    except Exception:
        pass
    return exif_payload


def _inject_xmp_and_exif_into_jpeg_lossless(
    file_path: Path, xmp_packet: bytes, orientation: int
) -> bool:
    """
    Losslessly injects/replaces the XMP APP1 segment (FF E1) and patches EXIF Orientation
    in a JPEG file while keeping the compressed image scan (FF DA .. FF D9) 100% byte-identical.
    """
    raw = file_path.read_bytes()
    if len(raw) < 4 or raw[:2] != b"\xff\xd8":
        return False

    xmp_app1_payload = XMP_HEADER + xmp_packet
    if len(xmp_app1_payload) + 2 > 65535:
        return False
    xmp_segment = (
        b"\xff\xe1"
        + struct.pack(">H", len(xmp_app1_payload) + 2)
        + xmp_app1_payload
    )

    out_chunks: list[bytes] = [b"\xff\xd8"]
    pos = 2
    xmp_inserted = False
    n = len(raw)

    while pos + 1 < n:
        if raw[pos] != 0xFF:
            # Reached scan data unexpectedly
            out_chunks.append(raw[pos:])
            break

        marker = raw[pos + 1]
        # Standalone markers without length
        if marker in (0xD8, 0x01) or (0xD0 <= marker <= 0xD7):
            out_chunks.append(raw[pos : pos + 2])
            pos += 2
            continue

        # SOS (Start of Scan) or EOI: everything from here to EOF is lossless compressed image data
        if marker in (0xDA, 0xD9):
            if not xmp_inserted:
                out_chunks.append(xmp_segment)
                xmp_inserted = True
            out_chunks.append(raw[pos:])
            break

        if pos + 4 > n:
            out_chunks.append(raw[pos:])
            break

        seg_len = struct.unpack(">H", raw[pos + 2 : pos + 4])[0]
        seg_end = pos + 2 + seg_len
        if seg_end > n:
            out_chunks.append(raw[pos:])
            break

        payload = raw[pos + 4 : seg_end]

        if marker == 0xE1 and payload.startswith(XMP_HEADER):
            # Replace existing XMP APP1 segment
            if not xmp_inserted:
                out_chunks.append(xmp_segment)
                xmp_inserted = True
        elif marker == 0xE1 and payload.startswith(EXIF_HEADER):
            # Patch EXIF orientation in-place if present
            patched_payload = _patch_exif_orientation_in_tiff_header(payload, orientation)
            out_chunks.append(
                b"\xff\xe1" + struct.pack(">H", len(patched_payload) + 2) + patched_payload
            )
        else:
            out_chunks.append(raw[pos:seg_end])

        pos = seg_end

    if not xmp_inserted:
        out_chunks.insert(1, xmp_segment)

    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp_xmp")
    tmp_path.write_bytes(b"".join(out_chunks))
    return _safe_atomic_replace(tmp_path, file_path)


def _safe_atomic_replace(tmp_path: Path, target_path: Path, max_retries: int = 5) -> bool:
    """
    Safely replaces target_path with tmp_path on Windows without failing due to
    antivirus/indexer file locks (WinError 5 Access is denied) or Read-Only attributes.
    """
    if target_path.exists():
        try:
            os.chmod(target_path, stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            pass

    for attempt in range(max_retries):
        try:
            os.replace(tmp_path, target_path)
            return True
        except (PermissionError, OSError) as exc:
            if attempt == max_retries - 1:
                # Fallback: copy bytes directly into target_path if atomic rename is blocked
                try:
                    target_path.write_bytes(tmp_path.read_bytes())
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    return True
                except Exception:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise exc
            time.sleep(0.04 * (2 ** attempt))
    return False


def _sync_single_media_item(
    media_detail: dict[str, Any], engine: DualDatabaseEngine | None = None
) -> int | None:
    """
    Synchronizes metadata for a single media item losslessly while preserving
    original filesystem mtime via os.utime and updating catalog.db.
    Returns preserved mtime (int epoch seconds) on success, or None if file missing.
    """
    folder_path = media_detail.get("folder_path")
    filename = media_detail.get("filename")
    if not folder_path or not filename:
        return None

    file_path = Path(folder_path) / filename
    if not file_path.exists() or not file_path.is_file():
        return None

    try:
        stat_res = file_path.stat()
        orig_atime = stat_res.st_atime
        orig_mtime = stat_res.st_mtime
    except OSError:
        return None

    xmp_packet = build_wlpg_xmp_packet(media_detail)
    orientation = int(media_detail.get("orientation") or 1)
    suffix = file_path.suffix.lower()

    ok = False
    if suffix in RAW_EXT_SET or suffix not in (".jpg", ".jpeg"):
        # Camera RAW or non-JPEG container: ALWAYS write to .xmp sidecar, NEVER modify RAW container directly
        sidecar = file_path.with_suffix(file_path.suffix + ".xmp")
        sidecar_tmp = sidecar.with_suffix(sidecar.suffix + ".tmp_xmp")
        sidecar_tmp.write_bytes(xmp_packet)
        ok = _safe_atomic_replace(sidecar_tmp, sidecar)
        record_recently_written(sidecar, len(xmp_packet), int(time.time()))
    else:
        ok = _inject_xmp_and_exif_into_jpeg_lossless(file_path, xmp_packet, orientation)

    if ok:
        try:
            os.utime(file_path, (orig_atime, orig_mtime))
        except OSError:
            pass
        try:
            cur_st = os.stat(file_path)
            new_size = int(cur_st.st_size)
            orig_mtime_int = int(cur_st.st_mtime)
        except OSError:
            orig_mtime_int = int(orig_mtime)
            new_size = int(media_detail.get("size") or 0)

        record_recently_written(file_path, new_size, orig_mtime_int)

        if engine is not None and "id" in media_detail:
            repo = CatalogRepository(engine)
            repo.mark_media_clean([int(media_detail["id"])], {int(media_detail["id"]): orig_mtime_int})
            with engine.write_lock:
                conn = engine.catalog_conn()
                conn.execute(
                    "UPDATE media SET size = ?, mtime = ? WHERE id = ?;",
                    (new_size, orig_mtime_int, int(media_detail["id"])),
                )
                conn.commit()
        return orig_mtime_int
    return None


def write_metadata_lossless(media_detail: dict[str, Any]) -> int | None:
    """
    Writes XMP and EXIF metadata for `media_detail` back to disk losslessly.
    Preserves original filesystem mtime via os.utime.
    Returns the file mtime (int epoch seconds) on success, or None if file missing.
    """
    return _sync_single_media_item(media_detail)


class _AsyncFlushSignals(QObject):
    batch_synced = Signal(list)
    sync_error = Signal(int, str)
    finished = Signal()


class _AsyncFlushWorker(QRunnable):
    """
    Background worker that runs a single metadata flush batch on a worker thread.
    """

    def __init__(
        self,
        engine: DualDatabaseEngine,
        batch_limit: int,
        signals: _AsyncFlushSignals,
        stop_requested: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.batch_limit = batch_limit
        self.signals = signals
        self.stop_requested = stop_requested

    def run(self) -> None:
        try:
            repo = CatalogRepository(self.engine)
            dirty_items = repo.fetch_dirty_media_batch(limit=self.batch_limit)
            if not dirty_items:
                return

            synced_ids: list[int] = []
            updated_mtimes: dict[int, int] = {}

            for item in dirty_items:
                if self.stop_requested and self.stop_requested.is_set():
                    break
                mid = int(item["id"])
                try:
                    new_mtime = write_metadata_lossless(item)
                    synced_ids.append(mid)
                    if new_mtime is not None:
                        updated_mtimes[mid] = new_mtime
                except Exception as exc:
                    self.signals.sync_error.emit(mid, str(exc))
                    synced_ids.append(mid)

            if synced_ids:
                repo.mark_media_clean(synced_ids, updated_mtimes)
                self.signals.batch_synced.emit(synced_ids)
        except Exception as exc:
            self.signals.sync_error.emit(0, str(exc))
        finally:
            self.engine.close_thread_connections()
            self.signals.finished.emit()


class MetadataSyncDaemon(QObject):
    """
    Background worker daemon that polls catalog.db for metadata_dirty = 1 rows
    and writes XMP/EXIF back to disk without blocking the UI thread.
    """

    batch_synced = Signal(list)       # list[int] of synced media_ids
    sync_error = Signal(int, str)     # media_id, error_message

    def __init__(
        self,
        engine: DualDatabaseEngine,
        poll_interval_ms: int = 1500,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.repo = CatalogRepository(engine)
        self.poll_interval_ms = poll_interval_ms
        self._running = False
        self._paused = False
        self.stop_requested = threading.Event()
        self._lock = threading.Lock()
        self._is_flushing = False
        self._timer: QTimer | None = None
        self._thread_pool = QThreadPool(self)
        self._thread_pool.setMaxThreadCount(1)

    def is_paused(self) -> bool:
        """Returns whether background file write-back is currently paused (Safe Mode)."""
        return self._paused

    def set_paused(self, paused: bool) -> None:
        """Pauses or resumes background file write-back."""
        self._paused = bool(paused)

    def notify_dirty(self) -> None:
        """Wakes the daemon to sync dirty metadata rows immediately if running and not paused."""
        if not self._running or self._paused or self._is_flushing or self.stop_requested.is_set():
            return
        QTimer.singleShot(0, self._on_timer_tick)

    def start(self) -> None:
        self._running = True
        self.stop_requested.clear()
        if self._timer is None:
            self._timer = QTimer(self)
            self._timer.setInterval(self.poll_interval_ms)
            self._timer.timeout.connect(self._on_timer_tick)
        self._timer.start()

    def stop(self) -> None:
        self._running = False
        self.stop_requested.set()
        if self._timer is not None and self._timer.isActive():
            self._timer.stop()
        self._thread_pool.waitForDone(3000)

    def _on_timer_tick(self) -> None:
        if not self._running or self._is_flushing or self._paused or self.stop_requested.is_set():
            return
        self._is_flushing = True
        signals = _AsyncFlushSignals()
        signals.batch_synced.connect(self.batch_synced.emit)
        signals.sync_error.connect(self.sync_error.emit)
        signals.finished.connect(self._on_flush_finished)
        worker = _AsyncFlushWorker(self.engine, batch_limit=25, signals=signals, stop_requested=self.stop_requested)
        self._thread_pool.start(worker)

    def _on_flush_finished(self) -> None:
        self._is_flushing = False

    def flush_now(self, batch_limit: int = 100, force: bool = False) -> int:
        """
        Synchronously drains dirty metadata items (up to batch_limit per pass until clean).
        If paused and not force, returns 0 immediately without modifying files.
        Returns total number of media items synced to disk.
        """
        if self._paused and not force:
            return 0
        total_synced = 0
        with self._lock:
            while True:
                dirty_items = self.repo.fetch_dirty_media_batch(limit=batch_limit)
                if not dirty_items:
                    break

                synced_ids: list[int] = []
                updated_mtimes: dict[int, int] = {}

                for item in dirty_items:
                    mid = int(item["id"])
                    try:
                        new_mtime = write_metadata_lossless(item)
                        synced_ids.append(mid)
                        if new_mtime is not None:
                            updated_mtimes[mid] = new_mtime
                    except Exception as exc:
                        self.sync_error.emit(mid, str(exc))
                        # Still mark clean or skip to avoid infinite tight loop on read-only file
                        synced_ids.append(mid)

                if synced_ids:
                    self.repo.mark_media_clean(synced_ids, updated_mtimes)
                    total_synced += len(synced_ids)
                    self.batch_synced.emit(synced_ids)

                if len(dirty_items) < batch_limit:
                    break
        return total_synced

