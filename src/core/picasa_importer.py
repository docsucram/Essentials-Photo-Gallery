"""
Non-destructive Google Picasa 3 Heritage Importer (.picasa.ini & contacts.xml).
Supports dry-run scanning and selective import of:
- Starred photos (`star=yes` -> `media.is_favourite = 1`)
- Captions (`caption=` / `description=`)
- Keywords / Tags (`keywords=`)
- Face regions (`faces=rect64(...),contact_id` + `[Contacts2]` / `contacts.xml`)
- Virtual albums (`[.album:...]` + `albums=`)
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.db.repository import CatalogRepository

_RECT64_FACE_RE = re.compile(r"rect64\(([0-9a-fA-F]{1,16})\),([^;]+)")


@dataclass
class PicasaFaceEntry:
    person_name: str
    rect_x: float
    rect_y: float
    rect_w: float
    rect_h: float


@dataclass
class PicasaMediaEntry:
    folder_path: Path
    filename: str
    starred: bool = False
    caption: str = ""
    keywords: list[str] = field(default_factory=list)
    faces: list[PicasaFaceEntry] = field(default_factory=list)
    album_tokens: list[str] = field(default_factory=list)


@dataclass
class PicasaAlbumEntry:
    token: str
    name: str
    description: str = ""


@dataclass
class PicasaImportPlan:
    root_dir: Path
    ini_files: int = 0
    entries: list[PicasaMediaEntry] = field(default_factory=list)
    albums: dict[str, PicasaAlbumEntry] = field(default_factory=dict)

    @property
    def stars_count(self) -> int:
        return sum(1 for e in self.entries if e.starred)

    @property
    def captions_count(self) -> int:
        return sum(1 for e in self.entries if e.caption)

    @property
    def tags_count(self) -> int:
        return sum(len(e.keywords) for e in self.entries)

    @property
    def faces_count(self) -> int:
        return sum(len(e.faces) for e in self.entries)


def decode_rect64(hex_str: str) -> tuple[float, float, float, float] | None:
    """Decodes Picasa 64-bit hex `rect64(L, T, R, B)` into normalized `(x, y, w, h)`."""
    try:
        val = int(hex_str.strip(), 16)
    except ValueError:
        return None
    left = ((val >> 48) & 0xFFFF) / 65535.0
    top = ((val >> 32) & 0xFFFF) / 65535.0
    right = ((val >> 16) & 0xFFFF) / 65535.0
    bottom = (val & 0xFFFF) / 65535.0
    w = max(0.0, min(1.0, right) - max(0.0, left))
    h = max(0.0, min(1.0, bottom) - max(0.0, top))
    if w <= 0.001 or h <= 0.001:
        return None
    return (max(0.0, left), max(0.0, top), w, h)


def parse_picasa_contacts_xml(contacts_xml_path: Path | str | None = None) -> dict[str, str]:
    """Loads `{contact_id: display_name}` from Picasa `contacts.xml` if present."""
    if contacts_xml_path is None:
        local_app = os.environ.get("LOCALAPPDATA", "")
        if not local_app:
            return {}
        p = Path(local_app) / "Google" / "Picasa2" / "contacts" / "contacts.xml"
    else:
        p = Path(contacts_xml_path)
    if not p.is_file():
        return {}
    mapping: dict[str, str] = {}
    try:
        tree = ET.parse(p)
        for el in tree.getroot().iter("contact"):
            cid = str(el.attrib.get("id") or "").strip()
            name = str(el.attrib.get("name") or el.attrib.get("display") or "").strip()
            if cid and name:
                mapping[cid] = name
    except Exception:
        return {}
    return mapping


def _parse_ini_sections(ini_path: Path) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    cur_sec: str | None = None
    try:
        text = ini_path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return sections
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            cur_sec = line[1:-1].strip()
            sections.setdefault(cur_sec, {})
        elif "=" in line and cur_sec is not None:
            k, v = line.split("=", 1)
            sections[cur_sec][k.strip()] = v.strip()
    return sections


def scan_picasa_tree(
    root_dir: Path | str,
    contacts_xml_path: Path | str | None = None,
) -> PicasaImportPlan:
    """Recursively scans `root_dir` for `.picasa.ini` / `Picasa.ini` metadata without modifying the DB."""
    root = Path(root_dir)
    plan = PicasaImportPlan(root_dir=root)
    if not root.is_dir():
        return plan

    global_contacts = parse_picasa_contacts_xml(contacts_xml_path)
    ini_names = {".picasa.ini", "picasa.ini"}

    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower() not in ini_names:
                continue
            ini_path = Path(dirpath) / fn
            plan.ini_files += 1
            sections = _parse_ini_sections(ini_path)

            # Local [Contacts2] overrides/supplements global contacts.xml
            local_contacts = dict(global_contacts)
            for sec_name, kv in sections.items():
                if sec_name.lower() == "contacts2":
                    for cid, raw_val in kv.items():
                        person = raw_val.split(";", 1)[0].strip()
                        if person:
                            local_contacts[cid.strip()] = person

            for sec_name, kv in sections.items():
                low = sec_name.lower()
                if low in ("picasa", "contacts2", "encoding"):
                    continue
                if low.startswith(".album:"):
                    token = sec_name.split(":", 1)[1].strip()
                    aname = kv.get("name", "").strip()
                    if token and aname:
                        plan.albums[token] = PicasaAlbumEntry(
                            token=token,
                            name=aname,
                            description=kv.get("description", "").strip(),
                        )
                    continue

                # File section
                starred = kv.get("star", "").strip().lower() in ("yes", "1", "true")
                caption = kv.get("caption", "") or kv.get("description", "")
                raw_kw = kv.get("keywords", "")
                keywords = [k.strip() for k in raw_kw.split(",") if k.strip()] if raw_kw else []
                raw_alb = kv.get("albums", "")
                album_tokens = [a.strip() for a in raw_alb.split(",") if a.strip()] if raw_alb else []

                faces: list[PicasaFaceEntry] = []
                raw_faces = kv.get("faces", "")
                if raw_faces:
                    for rect_hex, cid in _RECT64_FACE_RE.findall(raw_faces):
                        box = decode_rect64(rect_hex)
                        if box is None:
                            continue
                        pname = local_contacts.get(cid.strip(), "")
                        if not pname or pname == "ffffffffffffffff":
                            continue
                        faces.append(PicasaFaceEntry(pname, *box))

                if starred or caption or keywords or faces or album_tokens:
                    plan.entries.append(
                        PicasaMediaEntry(
                            folder_path=Path(dirpath),
                            filename=sec_name,
                            starred=starred,
                            caption=caption,
                            keywords=keywords,
                            faces=faces,
                            album_tokens=album_tokens,
                        )
                    )
    return plan


def apply_picasa_import(
    repo: CatalogRepository,
    plan: PicasaImportPlan,
    *,
    import_stars: bool = True,
    import_captions: bool = True,
    import_tags: bool = True,
    import_faces: bool = True,
    import_albums: bool = True,
) -> dict[str, int]:
    """Applies a `PicasaImportPlan` to `CatalogRepository`, returning counts of updated items."""
    stats = {"matched": 0, "stars": 0, "captions": 0, "tags": 0, "faces": 0, "albums": 0}
    if not plan.entries:
        return stats

    conn = repo.engine.catalog_conn()
    folder_map = {
        os.path.normcase(str(f["path"])): int(f["id"]) for f in repo.get_folders()
    }
    star_ids: list[int] = []
    album_media: dict[str, list[int]] = {}

    for entry in plan.entries:
        fid = folder_map.get(os.path.normcase(str(entry.folder_path)))
        if fid is None:
            continue
        row = conn.execute(
            "SELECT id FROM media WHERE folder_id = ? AND LOWER(filename) = ?;",
            (fid, entry.filename.lower()),
        ).fetchone()
        if not row:
            continue
        mid = int(row[0])
        stats["matched"] += 1

        if import_stars and entry.starred:
            star_ids.append(mid)

        if import_captions and entry.caption:
            repo.set_caption([mid], entry.caption)
            stats["captions"] += 1

        if import_tags and entry.keywords:
            for kw in entry.keywords:
                tid = repo.create_tag(kw)
                repo.assign_tag([mid], tid)
                stats["tags"] += 1

        if import_faces and entry.faces:
            with repo.engine.catalog_transaction() as tx:
                for face in entry.faces:
                    prow = tx.execute(
                        "SELECT id FROM people WHERE LOWER(name) = ?;",
                        (face.person_name.lower(),),
                    ).fetchone()
                    if prow:
                        pid = int(prow[0])
                    else:
                        cur = tx.execute(
                            "INSERT INTO people (name) VALUES (?);",
                            (face.person_name,),
                        )
                        pid = int(cur.lastrowid)
                    tx.execute(
                        """
                        INSERT INTO media_people (media_id, person_id, state, rect_x, rect_y, rect_w, rect_h, is_manual)
                        VALUES (?, ?, 1, ?, ?, ?, ?, 1);
                        """,
                        (mid, pid, face.rect_x, face.rect_y, face.rect_w, face.rect_h),
                    )
                    stats["faces"] += 1

        if import_albums and entry.album_tokens:
            for tok in entry.album_tokens:
                album_media.setdefault(tok, []).append(mid)

    if star_ids:
        repo.batch_update_favourite(star_ids, True)
        stats["stars"] = len(star_ids)

    if import_albums and album_media:
        existing_albums = {a["name"].lower(): int(a["id"]) for a in repo.get_albums_with_counts()}
        for tok, mids in album_media.items():
            alb_info = plan.albums.get(tok)
            aname = alb_info.name if alb_info else f"Picasa Album ({tok[:8]})"
            aid = existing_albums.get(aname.lower())
            if aid is None:
                aid = repo.create_album(aname)
                existing_albums[aname.lower()] = aid
            repo.add_media_to_album(aid, mids)
            stats["albums"] += 1

    return stats
