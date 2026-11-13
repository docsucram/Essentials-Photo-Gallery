"""
WorldMapView (QWidget) for PhotoGallery64.
Queries SQLite's R*Tree spatial index (`media_rtree`) via `CatalogRepository.query_map_clusters`
in <5ms as the map bounding box changes, clusters nearby GPS markers, and allows clicking
any pin/cluster or filtering the main thumbnail grid to the active map bounding box.
Provides an interactive Leaflet map (Clean Voyager, Street, Satellite, Terrain) via `QWebEngineView`
with persistent disk tile caching, and a zero-dependency vector map fallback canvas with
drawn continents that runs in offline/headless environments.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import urllib.parse
import urllib.request
from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import QFile, QModelIndex, QObject, QPointF, QRectF, QSize, Qt, QTimer, Signal, Slot
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QStandardItem,
    QStandardItemModel,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QCompleter,
    QDoubleSpinBox,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.db.repository import CatalogRepository

# Detect QtWebEngine and QtWebChannel availability with environment override
HAS_WEBENGINE = False
if os.environ.get("GALLERY_NO_WEBENGINE") != "1" and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
    try:
        import PySide6.QtWebChannel  # noqa: F401
        from PySide6.QtWebChannel import QWebChannel
        from PySide6.QtWebEngineWidgets import QWebEngineView
        HAS_WEBENGINE = True
    except Exception:
        HAS_WEBENGINE = False


def get_qwebchannel_js() -> str:
    """Retrieves Qt's built-in qwebchannel.js script for inlining."""
    try:
        import PySide6.QtWebChannel  # noqa: F401
        f = QFile(":/qtwebchannel/qwebchannel.js")
        if f.open(QFile.ReadOnly):
            data = bytes(f.readAll()).decode("utf-8")
            f.close()
            return data
    except Exception:
        pass
    return ""


def cluster_gps_points(
    points: Sequence[dict[str, Any]], grid_degrees: float = 2.0
) -> list[dict[str, Any]]:
    """
    Fast O(N) spatial grid clustering of geotagged media items.
    Returns cluster dicts with (lat, lon, count, media_ids, location_name).
    """
    if not points:
        return []
    if "media_ids" in points[0] and "count" in points[0]:
        clusters = list(points)
        clusters.sort(key=lambda c: c["count"], reverse=True)
        return clusters
    cell_size = max(0.01, float(grid_degrees))
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}

    for pt in points:
        lat = float(pt["lat"])
        lon = float(pt["lon"])
        gx = int(lon // cell_size)
        gy = int(lat // cell_size)
        buckets.setdefault((gx, gy), []).append(pt)

    clusters: list[dict[str, Any]] = []
    for items in buckets.values():
        cnt = len(items)
        avg_lat = sum(float(x["lat"]) for x in items) / cnt
        avg_lon = sum(float(x["lon"]) for x in items) / cnt
        mids = [int(x["id"]) for x in items]
        loc_names = [str(x.get("location_name") or "").strip() for x in items if x.get("location_name")]
        label = loc_names[0] if loc_names else f"{avg_lat:.2f}°, {avg_lon:.2f}°"
        clusters.append(
            {
                "lat": avg_lat,
                "lon": avg_lon,
                "count": cnt,
                "media_ids": mids,
                "label": label,
            }
        )
    clusters.sort(key=lambda c: c["count"], reverse=True)
    return clusters


def _clusters_to_json(clusters: Sequence[dict[str, Any]]) -> str:
    """Serializes clusters to JSON while stripping non-serializable objects like QPixmap."""
    clean = [
        {k: v for k, v in c.items() if k != "pixmap" and not isinstance(v, QPixmap)}
        for c in clusters
    ]
    return json.dumps(clean)


def generate_leaflet_html(
    clusters: Sequence[dict[str, Any]], qwebchannel_js: str | None = None
) -> str:
    """Generates a self-contained Leaflet HTML document with clustered GPS pins and QWebChannel bridge."""
    markers_json = _clusters_to_json(clusters)
    bridge_script = qwebchannel_js if qwebchannel_js is not None else get_qwebchannel_js()

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>PhotoGallery64 World Map</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body, #map {{ margin: 0; padding: 0; width: 100%; height: 100%; background: #EBF3FA; font-family: 'Segoe UI', -apple-system, sans-serif; }}
  .leaflet-tile-container img {{
    backface-visibility: hidden;
    -webkit-backface-visibility: hidden;
  }}
  .leaflet-fade-anim .leaflet-tile {{
    will-change: opacity, transform;
  }}
  .leaflet-cluster-pin {{
    background: transparent;
    border: none;
  }}
  .leaflet-popup-content-wrapper {{
    border-radius: 6px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    padding: 0;
    overflow: hidden;
  }}
  .leaflet-popup-content {{
    margin: 0;
    padding: 0;
    line-height: 1.4;
  }}
  .map-pin-badge {{
    position: absolute;
    top: -4px;
    right: -4px;
    background: #0066CC;
    color: #FFFFFF;
    border: 1.5px solid #FFFFFF;
    border-radius: 10px;
    padding: 0 5px;
    font-size: 10px;
    font-weight: bold;
    line-height: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3);
  }}
  .cluster-popup-card {{
    display: flex;
    flex-direction: column;
    width: 220px;
    background: #ffffff;
    font-family: 'Segoe UI', -apple-system, sans-serif;
  }}
  .cluster-popup-thumb {{
    width: 100%;
    height: 110px;
    object-fit: cover;
    background: #eef3f8;
    display: block;
  }}
  .cluster-popup-body {{
    padding: 8px 10px;
  }}
  .cluster-popup-title {{
    font-size: 13px;
    font-weight: 600;
    color: #1E395B;
    margin-bottom: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .cluster-popup-count {{
    font-size: 11px;
    color: #5A7A9E;
    margin-bottom: 8px;
  }}
  .cluster-popup-btn {{
    display: block;
    width: 100%;
    background: #0066CC;
    color: #FFFFFF;
    border: none;
    border-radius: 3px;
    padding: 5px 0;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    text-align: center;
  }}
  .cluster-popup-btn:hover {{
    background: #004D99;
  }}
</style>
<script>
{bridge_script}
</script>
</head>
<body>
<div id="map"></div>
<script>
  let map = null;
  let clusterLayer = null;
  let isInternalMove = false;
  let currentBaseLayer = null;
  window.pendingClusters = {markers_json};

  const tileLayers = {{
    'clean': L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      maxZoom: 19,
      maxNativeZoom: 17,
      keepBuffer: 4,
      updateWhenZooming: false,
      updateWhenIdle: false,
      attribution: 'Tiles &copy; Esri &mdash; Sources: Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, Esri Japan, METI, Esri China (Hong Kong), Esri (Thailand), TomTom'
    }}),
    'street': L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      maxNativeZoom: 19,
      keepBuffer: 4,
      updateWhenZooming: false,
      updateWhenIdle: false,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }}),
    'satellite': L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      maxZoom: 19,
      maxNativeZoom: 17,
      keepBuffer: 4,
      updateWhenZooming: false,
      updateWhenIdle: false,
      attribution: 'Tiles &copy; Esri'
    }}),
    'terrain': L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      maxZoom: 19,
      maxNativeZoom: 17,
      keepBuffer: 4,
      updateWhenZooming: false,
      updateWhenIdle: false,
      attribution: 'Tiles &copy; Esri &mdash; Sources: USGS, Esri, TCI, DeLorme, METI, NRCAN, GEBCO, NOAA, increment P Corp.'
    }})
  }};

  function switchBaseLayer(name) {{
    if (!map) return;
    const target = tileLayers[name] || tileLayers['clean'];
    if (currentBaseLayer && map.hasLayer(currentBaseLayer)) {{
      map.removeLayer(currentBaseLayer);
    }}
    currentBaseLayer = target;
    currentBaseLayer.addTo(map);
  }}

  function flyToLocation(lat, lon, zoom) {{
    if (!map) return;
    map.invalidateSize(false);
    isInternalMove = true;
    map.flyTo([lat, lon], zoom || 13, {{ duration: 1.0 }});
    setTimeout(function() {{ isInternalMove = false; }}, 1100);
  }}

  function fitMapBounds(min_lat, max_lat, min_lon, max_lon) {{
    if (!map) {{
      window.pendingFitBounds = [min_lat, max_lat, min_lon, max_lon];
      return;
    }}
    map.invalidateSize(false);
    const sz = map.getSize();
    if (!sz || sz.x < 64 || sz.y < 64) {{
      window.pendingFitBounds = [min_lat, max_lat, min_lon, max_lon];
      setTimeout(function() {{
        if (window.pendingFitBounds) {{
          const b = window.pendingFitBounds;
          fitMapBounds(b[0], b[1], b[2], b[3]);
        }}
      }}, 60);
      return;
    }}
    window.pendingFitBounds = null;
    isInternalMove = true;
    map.fitBounds([[min_lat, min_lon], [max_lat, max_lon]], {{ padding: [18, 18], maxZoom: 14, animate: false }});
    setTimeout(function() {{ isInternalMove = false; }}, 100);
  }}

  function setMapBounds(min_lat, max_lat, min_lon, max_lon) {{
    fitMapBounds(min_lat, max_lat, min_lon, max_lon);
  }}

  function invalidateMapSize() {{
    if (map) {{
      map.invalidateSize(false);
      if (window.pendingFitBounds) {{
        const b = window.pendingFitBounds;
        fitMapBounds(b[0], b[1], b[2], b[3]);
      }}
    }}
  }}

  function initMap() {{
    if (typeof L === 'undefined') {{
      document.getElementById('map').innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#1E395B;font-size:13px;text-align:center;padding:20px;">Leaflet offline or unavailable. Using vector map canvas.</div>';
      return;
    }}
    map = L.map('map', {{
      zoomControl: true,
      boxZoom: true,
      minZoom: 2,
      maxZoom: 19,
      zoomSnap: 0.25,
      zoomDelta: 0.5,
      zoomAnimation: true,
      fadeAnimation: true,
      markerZoomAnimation: true,
      wheelDebounceTime: 60,
      wheelPxPerZoomLevel: 80
    }}).setView([25.0, 0.0], 2);

    switchBaseLayer('clean');

    clusterLayer = L.layerGroup().addTo(map);

    if (typeof ResizeObserver !== 'undefined') {{
      const mapEl = document.getElementById('map');
      if (mapEl) {{
        const ro = new ResizeObserver(function() {{
          if (map) {{
            map.invalidateSize(false);
            if (window.pendingFitBounds) {{
              const b = window.pendingFitBounds;
              fitMapBounds(b[0], b[1], b[2], b[3]);
            }}
          }}
        }});
        ro.observe(mapEl);
      }}
    }}

    map.on('click', function(e) {{
      if (window.qtBridge && typeof window.qtBridge.onMapBackgroundClicked === 'function') {{
        window.qtBridge.onMapBackgroundClicked();
      }}
    }});

    let boundsTimer = null;
    map.on('zoomstart', function() {{
      if (boundsTimer) {{
        clearTimeout(boundsTimer);
        boundsTimer = null;
      }}
    }});
    map.on('moveend', function() {{
      if (isInternalMove) {{
        isInternalMove = false;
        return;
      }}
      if (boundsTimer) clearTimeout(boundsTimer);
      boundsTimer = setTimeout(function() {{
        if (window.qtBridge) {{
          const b = map.getBounds();
          window.qtBridge.onBoundsChanged(b.getSouth(), b.getNorth(), b.getWest(), b.getEast());
        }}
      }}, 200);
    }});

    if (window.pendingClusters) {{
      updateClusters(window.pendingClusters);
      window.pendingClusters = null;
    }}
  }}

  function updateClusters(clustersData) {{
    if (!map) {{
      window.pendingClusters = clustersData;
      return;
    }}
    const clusters = (typeof clustersData === 'string') ? JSON.parse(clustersData) : clustersData;
    const newLayer = L.layerGroup();
    if (clusters && clusters.length) {{
      clusters.forEach(c => {{
        const count = c.count || 1;
        const hasThumb = Boolean(c.thumb_url);
        const radius = Math.min(32, Math.max(16, 15 + Math.pow(count, 0.35) * 2.5));
        const size = radius * 2;
        let innerHtml = '';
        if (hasThumb) {{
          innerHtml = `<div class="map-cluster-pin-container" style="position:relative;width:${{size}}px;height:${{size}}px;cursor:pointer;">` +
            `<div style="width:${{size}}px;height:${{size}}px;border-radius:50%;overflow:hidden;border:2.5px solid #ffffff;box-shadow:0 3px 8px rgba(0,0,0,0.45);background:#0066CC;">` +
              `<img src="${{c.thumb_url}}" style="width:100%;height:100%;object-fit:cover;display:block;" onerror="this.style.display='none'"/>` +
            `</div>` +
            (count > 1 ? `<div class="map-pin-badge">${{count}}</div>` : '') +
          `</div>`;
        }} else {{
          innerHtml = `<div style="width:${{size}}px;height:${{size}}px;line-height:${{size}}px;border-radius:50%;background:#0066CC;color:#ffffff;border:2px solid #ffffff;text-align:center;font-weight:bold;font-size:11px;box-shadow:0 2px 6px rgba(0,0,0,0.35);cursor:pointer;">${{count}}</div>`;
        }}
        const icon = L.divIcon({{
          className: 'leaflet-cluster-pin',
          html: innerHtml,
          iconSize: [size, size],
          iconAnchor: [radius, radius]
        }});
        const marker = L.marker([c.lat, c.lon], {{ icon: icon }});
        const label = c.label || `${{c.lat.toFixed(2)}}°, ${{c.lon.toFixed(2)}}°`;
        const clusterPayload = JSON.stringify({{ media_ids: c.media_ids || [], label: label, count: count }});

        const imgTag = hasThumb ? `<img src="${{c.thumb_url}}" class="cluster-popup-thumb"/>` : '';
        const actionText = (count === 1) ? 'View Photo' : `Show all ${{count}} photos in gallery`;
        const popupHtml = `<div class="cluster-popup-card">` +
          imgTag +
          `<div class="cluster-popup-body">` +
            `<div class="cluster-popup-title">📍 ${{label}}</div>` +
            `<div class="cluster-popup-count">${{count}} ${{count === 1 ? 'item' : 'items'}}</div>` +
            `<button id="btn-cluster-open" class="cluster-popup-btn">${{actionText}}</button>` +
          `</div>` +
        `</div>`;

        marker.bindPopup(popupHtml, {{ maxWidth: 240, minWidth: 180 }});
        marker.on('popupopen', function() {{
          const btn = document.getElementById('btn-cluster-open');
          if (btn) {{
            btn.onclick = function() {{
              if (window.qtBridge) {{
                window.qtBridge.onClusterOpenRequested(clusterPayload);
              }}
            }};
          }}
        }});
        marker.on('click', () => {{
          if (window.qtBridge) {{
            window.qtBridge.onClusterSelected(clusterPayload);
          }}
        }});
        marker.on('dblclick', function(e) {{
          L.DomEvent.stopPropagation(e);
          if (map) {{
            map.setView([c.lat, c.lon], Math.min(18, map.getZoom() + 3));
          }}
        }});
        newLayer.addLayer(marker);
      }});
    }}
    newLayer.addTo(map);
    if (clusterLayer && map.hasLayer(clusterLayer)) {{
      map.removeLayer(clusterLayer);
    }}
    clusterLayer = newLayer;
  }}

  document.addEventListener('DOMContentLoaded', function() {{
    if (typeof QWebChannel !== 'undefined') {{
      new QWebChannel(qt.webChannelTransport, function(channel) {{
        window.qtBridge = channel.objects.mapBridge;
      }});
    }}
    initMap();
  }});
</script>
</body>
</html>"""


class MapBridge(QObject):
    """QWebChannel bridge exposing Qt signals and slots to JavaScript."""

    cluster_selected = Signal(list, str)
    cluster_open_requested = Signal(list, str)
    cluster_clicked = Signal(list)
    cluster_clicked_data = Signal(list, str)
    bounds_changed = Signal(float, float, float, float)
    map_background_clicked = Signal()

    @Slot(str)
    def onClusterSelected(self, cluster_json: str) -> None:
        try:
            data = json.loads(cluster_json)
            if isinstance(data, dict):
                mids = [int(x) for x in data.get("media_ids", [])]
                label = str(data.get("label", ""))
            elif isinstance(data, list):
                mids = [int(x) for x in data]
                label = ""
            else:
                return
            self.cluster_selected.emit(mids, label)
        except Exception:
            pass

    @Slot(str)
    def onClusterOpenRequested(self, cluster_json: str) -> None:
        try:
            data = json.loads(cluster_json)
            if isinstance(data, dict):
                mids = [int(x) for x in data.get("media_ids", [])]
                label = str(data.get("label", ""))
            elif isinstance(data, list):
                mids = [int(x) for x in data]
                label = ""
            else:
                return
            self.cluster_open_requested.emit(mids, label)
            self.cluster_clicked.emit(mids)
            self.cluster_clicked_data.emit(mids, label)
        except Exception:
            pass

    @Slot(str)
    def onClusterClicked(self, cluster_json: str) -> None:
        try:
            data = json.loads(cluster_json)
            if isinstance(data, dict):
                mids = [int(x) for x in data.get("media_ids", [])]
                label = str(data.get("label", ""))
            elif isinstance(data, list):
                mids = [int(x) for x in data]
                label = ""
            else:
                return
            self.cluster_selected.emit(mids, label)
            self.cluster_clicked.emit(mids)
            self.cluster_clicked_data.emit(mids, label)
        except Exception:
            pass

    @Slot()
    def onMapBackgroundClicked(self) -> None:
        self.map_background_clicked.emit()

    @Slot(float, float, float, float)
    def onBoundsChanged(
        self, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ) -> None:
        self.bounds_changed.emit(min_lat, max_lat, min_lon, max_lon)


def geocode_place_nominatim(
    query: str, limit: int = 5, timeout: float = 2.0
) -> list[dict[str, Any]]:
    """Queries OpenStreetMap Nominatim for place candidates, failing silently when offline."""
    q = (query or "").strip()
    if len(q) < 2 or os.environ.get("PYTEST_CURRENT_TEST"):
        return []
    url = (
        "https://nominatim.openstreetmap.org/search?"
        + urllib.parse.urlencode({"format": "jsonv2", "limit": str(limit), "q": q})
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "EssentialsPhotoGallery/0.11 (local desktop map search)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        if not isinstance(payload, list):
            return []
        results: list[dict[str, Any]] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            try:
                lat = float(entry["lat"])
                lon = float(entry["lon"])
            except (KeyError, TypeError, ValueError):
                continue
            label = str(entry.get("display_name") or entry.get("name") or q)
            results.append({"name": label, "formatted_address": label, "lat": lat, "lon": lon})
        return results
    except Exception:
        return []


class _MapLocationSearchBox(QLineEdit):
    """Search input with offline gazetteer + Nominatim autocomplete popup powered by QCompleter."""

    location_chosen = Signal(float, float, str)  # lat, lon, label
    _nominatim_results_ready = Signal(str, list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setPlaceholderText("Find place on map...")
        self.setClearButtonEnabled(True)
        self.setMinimumWidth(180)
        self.setMaximumWidth(280)
        self.setFixedHeight(26)
        self.setStyleSheet(
            "QLineEdit { "
            "  background-color: #FFFFFF; "
            "  border: 1px solid #7DA2CE; "
            "  border-radius: 3px; "
            "  padding: 2px 6px; "
            "  font-family: 'Segoe UI', sans-serif; "
            "  font-size: 9pt; "
            "} "
            "QLineEdit:focus { border: 1.5px solid #0066CC; }"
        )

        self._model = QStandardItemModel(self)
        self._completer = QCompleter(self._model, self)
        self._completer.setCaseSensitivity(Qt.CaseInsensitive)
        self._completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)

        popup = self._completer.popup()
        if popup is not None:
            popup.setStyleSheet(
                "QListView { "
                "  border: 1px solid #7DA2CE; "
                "  background: #FFFFFF; "
                "  font-family: 'Segoe UI', sans-serif; "
                "  font-size: 9pt; "
                "} "
                "QListView::item { padding: 4px 8px; } "
                "QListView::item:hover { background-color: #E5F3FF; } "
                "QListView::item:selected { background-color: #CCE8FF; color: #000000; }"
            )
        self.setCompleter(self._completer)
        self._completer.activated[QModelIndex].connect(self._on_item_activated)
        self.textEdited.connect(self._on_text_edited)
        self.returnPressed.connect(self._on_return_pressed)

        self._last_query: str = ""
        self._nominatim_timer = QTimer(self)
        self._nominatim_timer.setSingleShot(True)
        self._nominatim_timer.setInterval(320)
        self._nominatim_timer.timeout.connect(self._fetch_nominatim_async)
        self._nominatim_results_ready.connect(self._merge_nominatim_results)

    def _populate_results(self, results: list[dict[str, Any]], show_popup: bool = True) -> None:
        self._model.clear()
        seen_labels: set[str] = set()
        for r in results:
            label = str(r.get("formatted_address") or r.get("name") or "").strip()
            if not label or label in seen_labels:
                continue
            seen_labels.add(label)
            lat = float(r["lat"])
            lon = float(r["lon"])
            item = QStandardItem(label)
            item.setData((lat, lon, label), Qt.UserRole)
            self._model.appendRow(item)
        if show_popup and self._model.rowCount() > 0:
            self._completer.complete()

    def _on_text_edited(self, text: str) -> None:
        q = text.strip()
        self._last_query = q
        if len(q) < 2:
            self._nominatim_timer.stop()
            self._model.clear()
            return
        from src.core.geotag_resolver import search_places_offline

        results = search_places_offline(q, limit=8)
        self._populate_results(results, show_popup=bool(results))
        self._nominatim_timer.start()

    def _fetch_nominatim_async(self) -> None:
        q = self._last_query.strip()
        if len(q) < 3:
            return

        def _worker() -> None:
            res = geocode_place_nominatim(q, limit=5, timeout=2.0)
            if res:
                self._nominatim_results_ready.emit(q, res)

        threading.Thread(target=_worker, daemon=True).start()

    def _merge_nominatim_results(self, query: str, nom_results: list[dict[str, Any]]) -> None:
        if query != self.text().strip():
            return
        from src.core.geotag_resolver import search_places_offline

        combined = list(search_places_offline(query, limit=6)) + list(nom_results)
        self._populate_results(combined, show_popup=self.hasFocus())

    def _on_item_activated(self, index: QModelIndex) -> None:
        data = self._model.data(index, Qt.UserRole)
        if data and len(data) >= 3:
            lat, lon, label = float(data[0]), float(data[1]), str(data[2])
            self.setText(label)
            self.location_chosen.emit(lat, lon, label)

    def _on_return_pressed(self) -> None:
        self.trigger_search()

    def trigger_search(self) -> bool:
        """Resolves current search input via popup, offline gazetteer, or Nominatim and emits `location_chosen`."""
        popup = self._completer.popup()
        if popup and popup.isVisible():
            cur = popup.currentIndex()
            if not cur.isValid() and self._model.rowCount() > 0:
                cur = self._model.index(0, 0)
            if cur.isValid():
                self._on_item_activated(cur)
                popup.hide()
                return True

        q = self.text().strip()
        if not q:
            return False

        from src.core.geotag_resolver import search_places_offline

        matches = search_places_offline(q, limit=1)
        if not matches:
            matches = geocode_place_nominatim(q, limit=1, timeout=2.5)
        if matches:
            p = matches[0]
            label = str(p.get("formatted_address") or p.get("name") or q)
            self.location_chosen.emit(float(p["lat"]), float(p["lon"]), label)
            return True
        if self._model.rowCount() > 0:
            self._on_item_activated(self._model.index(0, 0))
            return True
        return False


MapLocationSearchBox = _MapLocationSearchBox


class _InteractiveMapCanvas(QWidget):
    """
    Offline vector map canvas that runs in environments where QtWebEngine is not installed.
    Renders vector continents, lat/lon graticules, and GPS cluster pins.
    """

    cluster_selected = Signal(list, str)
    cluster_clicked = Signal(list)
    cluster_clicked_data = Signal(list, str)
    viewport_bbox_dragged = Signal(float, float, float, float)
    area_drawn = Signal(float, float, float, float)
    map_background_clicked = Signal()

    CONTINENTS = [
        # North America
        [(-168, 65), (-140, 70), (-60, 80), (-55, 45), (-75, 25), (-80, 8), (-95, 16), (-105, 20), (-120, 35), (-125, 50), (-168, 65)],
        # South America
        [(-80, 8), (-60, 10), (-35, -5), (-40, -22), (-55, -35), (-65, -55), (-75, -50), (-70, -20), (-80, 0), (-80, 8)],
        # Eurasia
        [(-10, 36), (0, 44), (5, 50), (10, 58), (30, 70), (80, 73), (170, 68), (140, 50), (120, 35), (105, 10), (75, 10), (60, 25), (45, 12), (35, 32), (25, 40), (15, 38), (-10, 36)],
        # Africa
        [(-15, 15), (-17, 30), (10, 37), (35, 32), (50, 12), (40, -10), (30, -34), (18, -34), (10, 5), (-15, 15)],
        # Australia
        [(115, -22), (135, -12), (145, -15), (153, -28), (150, -38), (115, -35), (115, -22)],
        # British Isles
        [(-10, 51), (-5, 58), (2, 53), (-5, 50), (-10, 51)],
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.min_lat: float = -85.0
        self.max_lat: float = 85.0
        self.min_lon: float = -180.0
        self.max_lon: float = 180.0
        self.clusters: list[dict[str, Any]] = []
        self._pin_hit_rects: list[tuple[QRectF, list[int], str, float, float]] = []
        self._drag_origin: QPointF | None = None
        self._drag_current: QPointF | None = None

    def _xy_to_lat_lon(self, pt: QPointF) -> tuple[float, float]:
        w = max(1.0, float(self.width()))
        h = max(1.0, float(self.height()))
        lon = self.min_lon + (max(0.0, min(w, pt.x())) / w) * (self.max_lon - self.min_lon)
        lat = self.max_lat - (max(0.0, min(h, pt.y())) / h) * (self.max_lat - self.min_lat)
        return max(-85.0, min(85.0, lat)), max(-180.0, min(180.0, lon))

    def set_bounds_and_clusters(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        clusters: list[dict[str, Any]],
    ) -> None:
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.clusters = clusters
        self.update()

    def _lat_lon_to_xy(self, lat: float, lon: float) -> QPointF:
        w = max(1.0, float(self.width()))
        h = max(1.0, float(self.height()))
        lon_span = max(0.001, self.max_lon - self.min_lon)
        lat_span = max(0.001, self.max_lat - self.min_lat)
        x = ((lon - self.min_lon) / lon_span) * w
        y = ((self.max_lat - lat) / lat_span) * h
        return QPointF(x, y)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        rect = self.rect()
        # Ocean fill
        painter.fillRect(rect, QColor("#D9ECFA"))

        # Draw simplified continent landmasses
        painter.setBrush(QBrush(QColor("#E2EED8")))
        painter.setPen(QPen(QColor("#B8D0AB"), 1.2))
        for poly_coords in self.CONTINENTS:
            path = QPainterPath()
            first = True
            for lon, lat in poly_coords:
                pt = self._lat_lon_to_xy(lat, lon)
                if first:
                    path.moveTo(pt)
                    first = False
                else:
                    path.lineTo(pt)
            path.closeSubpath()
            painter.drawPath(path)

        # Draw lat/lon graticule grid lines
        painter.setPen(QPen(QColor("#B5D3EB"), 1, Qt.DotLine))
        for lon in range(-180, 181, 30):
            if self.min_lon <= lon <= self.max_lon:
                pt = self._lat_lon_to_xy(0, lon)
                painter.drawLine(int(pt.x()), 0, int(pt.x()), rect.height())
        for lat in range(-60, 61, 30):
            if self.min_lat <= lat <= self.max_lat:
                pt = self._lat_lon_to_xy(lat, 0)
                painter.drawLine(0, int(pt.y()), rect.width(), int(pt.y()))

        # Draw Equator & Prime Meridian
        painter.setPen(QPen(QColor("#8FB8DA"), 1.2, Qt.DashLine))
        if self.min_lat <= 0 <= self.max_lat:
            eq = self._lat_lon_to_xy(0, 0)
            painter.drawLine(0, int(eq.y()), rect.width(), int(eq.y()))
        if self.min_lon <= 0 <= self.max_lon:
            pm = self._lat_lon_to_xy(0, 0)
            painter.drawLine(int(pm.x()), 0, int(pm.x()), rect.height())

        # Draw GPS Cluster Pins
        self._pin_hit_rects.clear()
        for c in self.clusters:
            pt = self._lat_lon_to_xy(float(c["lat"]), float(c["lon"]))
            cnt = int(c.get("count", 1))
            radius = min(28.0, 14.0 + (cnt ** 0.35) * 2.5)
            pin_rect = QRectF(
                pt.x() - radius, pt.y() - radius, radius * 2.0, radius * 2.0
            )
            label = str(c.get("label", ""))
            self._pin_hit_rects.append(
                (pin_rect, c.get("media_ids", []), label, float(c["lat"]), float(c["lon"]))
            )

            pix = c.get("pixmap")
            if isinstance(pix, QPixmap) and not pix.isNull():
                painter.save()
                clip_path = QPainterPath()
                clip_path.addEllipse(pin_rect)
                painter.setClipPath(clip_path)
                scaled_pix = pix.scaled(
                    int(radius * 2.0),
                    int(radius * 2.0),
                    Qt.KeepAspectRatioByExpanding,
                    Qt.SmoothTransformation,
                )
                painter.drawPixmap(
                    int(pin_rect.x()),
                    int(pin_rect.y()),
                    scaled_pix,
                )
                painter.restore()

                painter.setPen(QPen(QColor("#FFFFFF"), 2.0))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(pin_rect)

                if cnt > 1:
                    badge_radius = 8.5
                    badge_center = QPointF(pin_rect.right() - 2.0, pin_rect.top() + 2.0)
                    badge_rect = QRectF(
                        badge_center.x() - badge_radius,
                        badge_center.y() - badge_radius,
                        badge_radius * 2.0,
                        badge_radius * 2.0,
                    )
                    painter.setBrush(QBrush(QColor("#0066CC")))
                    painter.setPen(QPen(QColor("#FFFFFF"), 1.5))
                    painter.drawEllipse(badge_rect)
                    painter.setPen(QColor("#FFFFFF"))
                    painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                    painter.drawText(badge_rect, Qt.AlignCenter, str(cnt))
            else:
                painter.setBrush(QBrush(QColor("#0066CC")))
                painter.setPen(QPen(QColor("#FFFFFF"), 2.0))
                painter.drawEllipse(pin_rect)

                painter.setPen(QColor("#FFFFFF"))
                painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
                painter.drawText(pin_rect, Qt.AlignCenter, str(cnt))

        # Draw active rectangle-zoom rubberband when dragging on blank map
        if self._drag_origin is not None and self._drag_current is not None:
            sel_rect = QRectF(self._drag_origin, self._drag_current).normalized()
            if sel_rect.width() >= 2 and sel_rect.height() >= 2:
                painter.save()
                painter.setBrush(QBrush(QColor(0, 102, 204, 45)))
                painter.setPen(QPen(QColor("#0066CC"), 2.0, Qt.DashLine))
                painter.drawRect(sel_rect)
                painter.restore()

        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            pos = event.position()
            on_pin = any(pin_rect.contains(pos) for pin_rect, *_ in self._pin_hit_rects)
            if not on_pin or bool(event.modifiers() & Qt.ShiftModifier):
                self._drag_origin = pos
                self._drag_current = pos
                self.update()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin is not None:
            self._drag_current = event.position()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._drag_origin is not None:
            end_pos = event.position()
            drag_rect = QRectF(self._drag_origin, end_pos).normalized()
            start_lat, start_lon = self._xy_to_lat_lon(self._drag_origin)
            end_lat, end_lon = self._xy_to_lat_lon(end_pos)
            self._drag_origin = None
            self._drag_current = None
            self.update()
            if drag_rect.width() >= 6 or drag_rect.height() >= 6:
                self.viewport_bbox_dragged.emit(
                    min(start_lat, end_lat),
                    max(start_lat, end_lat),
                    min(start_lon, end_lon),
                    max(start_lon, end_lon),
                )
                event.accept()
                return
        if event.button() == Qt.LeftButton:
            pos = event.position()
            for pin_rect, media_ids, label, _la, _lo in self._pin_hit_rects:
                if pin_rect.contains(pos):
                    self.cluster_selected.emit(media_ids, label)
                    self.cluster_clicked.emit(media_ids)
                    self.cluster_clicked_data.emit(media_ids, label)
                    event.accept()
                    return
            self.map_background_clicked.emit()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            pos = event.position()
            for pin_rect, _mids, _label, la, lo in self._pin_hit_rects:
                if pin_rect.contains(pos):
                    # Zoom in on cluster
                    span_la = (self.max_lat - self.min_lat) / 4.0
                    span_lo = (self.max_lon - self.min_lon) / 4.0
                    self.viewport_bbox_dragged.emit(
                        la - span_la, la + span_la, lo - span_lo, lo + span_lo
                    )
                    event.accept()
                    return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 0.8 if event.angleDelta().y() > 0 else 1.25
        c_lat = (self.min_lat + self.max_lat) / 2.0
        c_lon = (self.min_lon + self.max_lon) / 2.0
        half_lat = max(0.05, min(85.0, ((self.max_lat - self.min_lat) / 2.0) * factor))
        half_lon = max(0.05, min(180.0, ((self.max_lon - self.min_lon) / 2.0) * factor))
        self.viewport_bbox_dragged.emit(
            max(-85.0, c_lat - half_lat),
            min(85.0, c_lat + half_lat),
            max(-180.0, c_lon - half_lon),
            min(180.0, c_lon + half_lon),
        )
        event.accept()


class WorldMapView(QWidget):
    """
    Geotag World Map View backed by SQLite's R*Tree (`media_rtree`).
    Renders an interactive Leaflet map (Clean Light, OSM Street, Satellite) via QWebEngineView
    with automatic silent fallback to high-performance vector canvas (_InteractiveMapCanvas).
    Controls live in the contextual top Ribbon (Map tab); no duplicate internal toolbar.
    """

    bbox_changed = Signal(float, float, float, float)
    cluster_selected = Signal(list, str)
    cluster_open_requested = Signal(list, str)
    cluster_clicked = Signal(list)
    cluster_clicked_data = Signal(list, str)
    map_background_clicked = Signal()
    filter_to_bbox_requested = Signal(float, float, float, float)
    close_requested = Signal()
    open_media_requested = Signal(int)
    map_stats_Changed = Signal(int, int, int)
    map_filters_changed = Signal(object, object, object)

    def minimumSizeHint(self) -> QSize:
        return QSize(300, 200)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("WorldMapView")

        self._min_lat: float = -85.0
        self._max_lat: float = 85.0
        self._min_lon: float = -180.0
        self._max_lon: float = 180.0
        self._current_clusters: list[dict[str, Any]] = []
        self._repo: CatalogRepository | None = None
        self._thumb_repo: Any = None
        self._web_loaded: bool = False
        self._current_style: str = "clean"
        self._pending_fit_bounds: tuple[float, float, float, float] | None = None
        self._filter_year: int | None = None
        self._filter_person_id: int | None = None
        self._filter_min_rating: int | None = None
        self._pending_thumb_ids: set[int] = set()
        self._failed_thumb_ids: set[int] = set()
        self._thumb_refresh_timer = QTimer(self)
        self._thumb_refresh_timer.setSingleShot(True)
        self._thumb_refresh_timer.setInterval(80)
        self._thumb_refresh_timer.timeout.connect(self._push_updated_cluster_thumbnails)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # Full-size Central Map View Stack (fills 100% of workspace right up to Ribbon)
        self.map_stack = QStackedWidget(self)
        self.canvas = _InteractiveMapCanvas(self.map_stack)
        self.canvas.cluster_selected.connect(self.cluster_selected.emit)
        self.canvas.cluster_clicked.connect(self.cluster_clicked.emit)
        self.canvas.cluster_clicked_data.connect(self.cluster_clicked_data.emit)
        self.canvas.map_background_clicked.connect(self.map_background_clicked.emit)
        self.canvas.viewport_bbox_dragged.connect(
            lambda min_la, max_la, min_lo, max_lo: self.set_viewport_bbox(
                min_la, max_la, min_lo, max_lo, self._repo
            )
        )

        self.web_view: Any = None
        self.channel: Any = None
        self.bridge: Any = None

        if HAS_WEBENGINE:
            try:
                # Configure persistent tile cache so tiles work offline
                from PySide6.QtWebEngineCore import QWebEngineProfile
                from src.utils.paths import get_default_db_dir

                cache_dir = get_default_db_dir() / "map_tiles_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                profile = QWebEngineProfile.defaultProfile()
                profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.DiskHttpCache)
                profile.setCachePath(str(cache_dir))
                profile.setPersistentStoragePath(str(cache_dir))

                self.bridge = MapBridge(self)
                self.bridge.cluster_selected.connect(self.cluster_selected.emit)
                self.bridge.cluster_open_requested.connect(
                    self.cluster_open_requested.emit
                )
                self.bridge.cluster_clicked.connect(self.cluster_clicked.emit)
                self.bridge.cluster_clicked_data.connect(self.cluster_clicked_data.emit)
                self.bridge.map_background_clicked.connect(
                    self.map_background_clicked.emit
                )
                self.bridge.bounds_changed.connect(self._on_web_bounds_changed)

                self.channel = QWebChannel(self)
                self.channel.registerObject("mapBridge", self.bridge)

                self.web_view = QWebEngineView(self.map_stack)
                self.web_view.page().setBackgroundColor(QColor("#AAD3DF"))
                self.web_view.page().setWebChannel(self.channel)
                self.web_view.loadFinished.connect(self._on_web_load_finished)

                html = generate_leaflet_html(self._current_clusters)
                self.web_view.setHtml(html)

                self.map_stack.addWidget(self.web_view)
                self.map_stack.addWidget(self.canvas)
                self.map_stack.setCurrentWidget(self.web_view)
            except Exception:
                self.web_view = None
                self.map_stack.addWidget(self.canvas)
                self.map_stack.setCurrentWidget(self.canvas)
        else:
            self.map_stack.addWidget(self.canvas)
            self.map_stack.setCurrentWidget(self.canvas)

        vbox.addWidget(self.map_stack, 1)

        # Backwards-compatibility headless stubs for tests (unparented, not added to layout)
        self.cluster_list = QListWidget()
        self.cluster_list.hide()
        self.btn_toggle_engine = QPushButton("🗺 View: Leaflet Web")
        self.btn_toggle_engine.clicked.connect(self._toggle_map_engine)
        self.btn_toggle_engine.hide()
        self.spin_min_lat = QDoubleSpinBox()
        self.spin_max_lat = QDoubleSpinBox()
        self.spin_min_lon = QDoubleSpinBox()
        self.spin_max_lon = QDoubleSpinBox()

    def bind_repository(self, repo: CatalogRepository, thumb_repo: Any = None) -> None:
        self._repo = repo
        if thumb_repo is not None:
            self._thumb_repo = thumb_repo
        self._emit_map_stats()

    @staticmethod
    def _apply_blob_to_cluster(cluster: dict[str, Any], blob: bytes) -> None:
        b64 = base64.b64encode(blob).decode("ascii")
        cluster["thumb_url"] = f"data:image/webp;base64,{b64}"
        pix = QPixmap()
        if pix.loadFromData(blob, "WEBP"):
            cluster["pixmap"] = pix

    def _attach_cluster_thumbnails(self, clusters: list[dict[str, Any]]) -> None:
        """Enriches cluster dictionaries with thumbnail base64 data URIs and QPixmaps."""
        if not clusters or self._thumb_repo is None:
            return
        # Fast path: check up to 20 candidate media IDs per cluster in one batch
        all_candidate_ids: list[int] = []
        for c in clusters:
            mids = c.get("media_ids") or []
            if mids:
                all_candidate_ids.extend(mids[:20])

        if not all_candidate_ids:
            return
        missing_sample_ids: list[int] = []
        try:
            blobs = self._thumb_repo.get_webp_blobs_batch(all_candidate_ids)

            # Deep fallback: for larger clusters where none of the first 20 items are cached,
            # check whether ANY item in the cluster already exists in thumbnails.db.
            extra_matched_ids: dict[int, int] = {}
            if hasattr(self._thumb_repo, "get_first_cached_media_id"):
                for idx, c in enumerate(clusters):
                    mids = c.get("media_ids") or []
                    if len(mids) <= 20:
                        continue
                    if any(mid in blobs for mid in mids[:20]):
                        continue
                    cached_mid = self._thumb_repo.get_first_cached_media_id(mids[20:])
                    if cached_mid is not None:
                        extra_matched_ids[idx] = cached_mid
                if extra_matched_ids:
                    blobs.update(self._thumb_repo.get_webp_blobs_batch(list(extra_matched_ids.values())))

            for idx, c in enumerate(clusters):
                mids = c.get("media_ids") or []
                if not mids:
                    continue
                blob = None
                for mid in mids[:20]:
                    if mid in blobs:
                        blob = blobs[mid]
                        break
                if blob is None and idx in extra_matched_ids:
                    blob = blobs.get(extra_matched_ids[idx])
                if blob:
                    self._apply_blob_to_cluster(c, blob)
                else:
                    sample_mid = int(mids[0])
                    if sample_mid not in self._pending_thumb_ids and sample_mid not in self._failed_thumb_ids:
                        missing_sample_ids.append(sample_mid)
        except Exception:
            pass

        if missing_sample_ids:
            self._schedule_cluster_thumbnail_decode(missing_sample_ids)

    def _schedule_cluster_thumbnail_decode(self, sample_ids: list[int]) -> None:
        """Asynchronously decodes 1 sample thumbnail from disk for clusters with no cached thumb."""
        if self._repo is None or self._thumb_repo is None:
            return
        engine = getattr(self._repo, "engine", None)
        get_info = getattr(self._repo, "get_media_decode_info_batch", None)
        if engine is None or get_info is None:
            return

        # Cap per-viewport decode burst so rapid panning stays responsive
        to_decode = [mid for mid in sample_ids if mid not in self._pending_thumb_ids and mid not in self._failed_thumb_ids][:24]
        if not to_decode:
            return

        try:
            rows = get_info(to_decode)
        except Exception:
            return
        if not rows:
            return

        from pathlib import Path
        from PySide6.QtCore import QThreadPool
        from src.workers.scanner_thumb_worker import ThumbnailDecodeWorker

        pool = QThreadPool.globalInstance()
        for row in rows:
            mid = int(row["id"])
            file_path = Path(row["folder_path"]) / str(row["filename"])
            self._pending_thumb_ids.add(mid)
            worker = ThumbnailDecodeWorker(
                engine=engine,
                media_id=mid,
                file_path=file_path,
                mtime=int(row.get("mtime") or 0),
                is_video=bool(row.get("is_video")),
                orientation=int(row.get("orientation") or 1),
            )
            worker.signals.thumbnail_ready.connect(self._on_cluster_thumbnail_ready)
            worker.signals.error.connect(self._on_cluster_thumbnail_error)
            pool.start(worker)

    @Slot(int, object)
    def _on_cluster_thumbnail_ready(self, media_id: int, qimg: Any) -> None:
        self._pending_thumb_ids.discard(int(media_id))
        blob = None
        if self._thumb_repo is not None and hasattr(self._thumb_repo, "get_webp_blob"):
            try:
                blob = self._thumb_repo.get_webp_blob(int(media_id))
            except Exception:
                blob = None

        pix = QPixmap.fromImage(qimg) if qimg is not None and not qimg.isNull() else QPixmap()
        if blob is None and not pix.isNull():
            from PySide6.QtCore import QBuffer, QIODevice
            buf = QBuffer()
            buf.open(QIODevice.OpenModeFlag.WriteOnly)
            if pix.save(buf, "WEBP", 80):
                blob = bytes(buf.data())

        if blob is None and pix.isNull():
            return

        b64_url = f"data:image/webp;base64,{base64.b64encode(blob).decode('ascii')}" if blob else None
        updated = False
        mid_int = int(media_id)
        for c in self._current_clusters:
            if c.get("thumb_url"):
                continue
            mids = c.get("media_ids") or []
            if mid_int in mids:
                if b64_url:
                    c["thumb_url"] = b64_url
                if not pix.isNull():
                    c["pixmap"] = pix
                updated = True

        if updated and not self._thumb_refresh_timer.isActive():
            self._thumb_refresh_timer.start()

    @Slot(int, str)
    def _on_cluster_thumbnail_error(self, media_id: int, _msg: str) -> None:
        mid_int = int(media_id)
        self._pending_thumb_ids.discard(mid_int)
        self._failed_thumb_ids.add(mid_int)

    def _push_updated_cluster_thumbnails(self) -> None:
        if not self._current_clusters:
            return
        self.canvas.set_bounds_and_clusters(
            self._min_lat, self._max_lat, self._min_lon, self._max_lon, self._current_clusters
        )
        if self._web_loaded and self.web_view is not None:
            self.web_view.page().runJavaScript(
                f"updateClusters({_clusters_to_json(self._current_clusters)});"
            )

    @property
    def map_style(self) -> str:
        return self._current_style

    @property
    def active_map_filters(self) -> tuple[int | None, int | None, int | None]:
        return (self._filter_year, self._filter_person_id, self._filter_min_rating)

    def has_active_map_filters(self) -> bool:
        return (
            self._filter_year is not None
            or self._filter_person_id is not None
            or self._filter_min_rating is not None
        )

    def set_map_filters(
        self,
        year: int | None = None,
        person_id: int | None = None,
        min_rating: int | None = None,
        *,
        fit: bool = False,
    ) -> None:
        """Applies optional Year / Person / Rating filters to map clusters and stats."""
        self._filter_year = int(year) if year is not None else None
        self._filter_person_id = int(person_id) if person_id is not None else None
        self._filter_min_rating = int(min_rating) if min_rating is not None else None
        self.map_filters_changed.emit(
            self._filter_year, self._filter_person_id, self._filter_min_rating
        )
        if fit:
            self.fit_to_photos()
        else:
            self.refresh_current_viewport()

    def clear_map_filters(self, *, fit: bool = False) -> None:
        """Clears all active Year / Person / Rating map filters."""
        self.set_map_filters(None, None, None, fit=fit)

    def set_map_style(self, style_name: str) -> None:
        """Switches base tile layer dynamically."""
        self._current_style = style_name
        if self._web_loaded and self.web_view is not None:
            self.web_view.page().runJavaScript(f"switchBaseLayer('{style_name}');")

    def view_area_in_grid(self) -> None:
        """Emits `filter_to_bbox_requested` for the currently visible map viewport bounds."""
        self.filter_to_bbox_requested.emit(
            self._min_lat,
            self._max_lat,
            self._min_lon,
            self._max_lon,
        )

    def fly_to_place(self, query: str, zoom: int = 13) -> bool:
        """Resolves a place name via offline gazetteer or Nominatim and flies the map to it."""
        q = (query or "").strip()
        if not q:
            return False
        from src.core.geotag_resolver import search_places_offline

        matches = search_places_offline(q, limit=1)
        if not matches:
            matches = geocode_place_nominatim(q, limit=1, timeout=2.5)
        if matches:
            p = matches[0]
            self.fly_to(float(p["lat"]), float(p["lon"]), zoom=zoom)
            return True
        return False

    def _toggle_map_engine(self) -> None:
        """Toggles between WebEngine view and Vector Canvas view (for testing/diagnostics)."""
        if not HAS_WEBENGINE or self.web_view is None:
            return
        if self.map_stack.currentWidget() == self.web_view:
            self.map_stack.setCurrentWidget(self.canvas)
            self.btn_toggle_engine.setText("🗺 View: Vector Offline")
            self.canvas.update()
        else:
            self.map_stack.setCurrentWidget(self.web_view)
            self.btn_toggle_engine.setText("🗺 View: Leaflet Web")
            if self._web_loaded:
                self.web_view.page().runJavaScript(
                    f"updateClusters({_clusters_to_json(self._current_clusters)});"
                )

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        if self._web_loaded and self.web_view is not None:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(60, lambda: self.web_view.page().runJavaScript("if (typeof map !== 'undefined' && map) { map.invalidateSize(); }"))

    def _on_web_load_finished(self, ok: bool) -> None:
        self._web_loaded = ok
        if ok and self.web_view is not None:
            if self._pending_fit_bounds is not None:
                min_la, max_la, min_lo, max_lo = self._pending_fit_bounds
                self._pending_fit_bounds = None
                self.web_view.page().runJavaScript(
                    f"fitMapBounds({min_la}, {max_la}, {min_lo}, {max_lo});"
                )
            if self._current_clusters:
                self.web_view.page().runJavaScript(
                    f"updateClusters({_clusters_to_json(self._current_clusters)});"
                )

    def _emit_map_stats(self) -> tuple[int, int, int]:
        geotagged_count = sum(int(c.get("count", 0)) for c in self._current_clusters)
        cluster_count = len(self._current_clusters)
        total_items = geotagged_count
        if self._repo is not None:
            try:
                row = self._repo.conn.execute("SELECT COUNT(*) FROM media;").fetchone()
                if row is not None:
                    total_items = int(row[0])
            except Exception:
                total_items = geotagged_count
        self.map_stats_Changed.emit(geotagged_count, cluster_count, total_items)
        return geotagged_count, cluster_count, total_items

    def _update_viewport_clusters(
        self, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ) -> list[dict[str, Any]]:
        self._min_lat = max(-85.0, min(float(min_lat), float(max_lat)))
        self._max_lat = min(85.0, max(float(min_lat), float(max_lat)))
        self._min_lon = max(-180.0, min(float(min_lon), float(max_lon)))
        self._max_lon = min(180.0, max(float(min_lon), float(max_lon)))

        self.spin_min_lat.setValue(self._min_lat)
        self.spin_max_lat.setValue(self._max_lat)
        self.spin_min_lon.setValue(self._min_lon)
        self.spin_max_lon.setValue(self._max_lon)

        lat_span = max(0.1, self._max_lat - self._min_lat)
        grid_deg = max(0.05, lat_span / 25.0)

        raw_points: list[dict[str, Any]] = []
        if self._repo is not None:
            if self.has_active_map_filters():
                raw_points = self._repo.query_map_clusters(
                    self._min_lat,
                    self._max_lat,
                    self._min_lon,
                    self._max_lon,
                    grid_degrees=grid_deg,
                    year=self._filter_year,
                    person_id=self._filter_person_id,
                    min_rating=self._filter_min_rating,
                )
            else:
                raw_points = self._repo.query_map_clusters(
                    self._min_lat, self._max_lat, self._min_lon, self._max_lon, grid_degrees=grid_deg
                )

        self._current_clusters = cluster_gps_points(raw_points, grid_degrees=grid_deg)
        self._attach_cluster_thumbnails(self._current_clusters)
        self.canvas.set_bounds_and_clusters(
            self._min_lat, self._max_lat, self._min_lon, self._max_lon, self._current_clusters
        )

        if self._web_loaded and self.web_view is not None:
            self.web_view.page().runJavaScript(
                f"updateClusters({_clusters_to_json(self._current_clusters)});"
            )

        self.cluster_list.clear()
        for c in self._current_clusters:
            item = QListWidgetItem(
                f"📍 {c['label']} ({c['count']} items)", self.cluster_list
            )
            item.setData(Qt.UserRole, (c["media_ids"], c["label"]))

        self._emit_map_stats()
        self.bbox_changed.emit(self._min_lat, self._max_lat, self._min_lon, self._max_lon)
        return self._current_clusters

    def _on_web_bounds_changed(
        self, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ) -> None:
        """Called when user pans/zooms the Leaflet map."""
        self._update_viewport_clusters(min_lat, max_lat, min_lon, max_lon)

    def _on_cluster_list_clicked(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.UserRole)
        if isinstance(data, (tuple, list)) and len(data) >= 2 and isinstance(data[0], list):
            mids, label = data[0], str(data[1])
        elif isinstance(data, list):
            mids, label = data, ""
        else:
            return
        if mids:
            self.cluster_selected.emit(mids, label)
            self.cluster_clicked.emit(mids)
            self.cluster_clicked_data.emit(mids, label)

    def fly_to(self, lat: float, lon: float, zoom: int = 13) -> None:
        """Pans and zooms smoothly to a given coordinate."""
        delta = 0.05
        self.set_viewport_bbox(lat - delta, lat + delta, lon - delta, lon + delta)
        if self._web_loaded and self.web_view is not None:
            self.web_view.page().runJavaScript(f"flyToLocation({lat}, {lon}, {zoom});")
        elif self.canvas is not None:
            self.canvas.update()

    def _get_clean_geotag_bounds(self) -> tuple[float, float, float, float] | None:
        if self._repo is None:
            return None
        engine = getattr(self._repo, "engine", None)
        if engine is None:
            return self._repo.get_geotag_bounds()
        filter_clauses, filter_params = CatalogRepository.build_map_filter_clauses(
            year=self._filter_year,
            person_id=self._filter_person_id,
            min_rating=self._filter_min_rating,
            alias="m",
        )
        extra_sql = (" AND " + " AND ".join(filter_clauses)) if filter_clauses else ""
        conn = engine.catalog_conn()
        row = conn.execute(
            f"""
            SELECT MIN(m.lat) AS min_lat, MAX(m.lat) AS max_lat,
                   MIN(m.lon) AS min_lon, MAX(m.lon) AS max_lon
            FROM media m
            WHERE m.lat IS NOT NULL AND m.lon IS NOT NULL
              AND m.lat BETWEEN -85.0 AND 85.0
              AND m.lon BETWEEN -180.0 AND 180.0
              AND NOT (ABS(m.lat) < 0.0001 AND ABS(m.lon) < 0.0001){extra_sql};
            """,
            filter_params,
        ).fetchone()
        if (
            row
            and row["min_lat"] is not None
            and row["max_lat"] is not None
            and row["min_lon"] is not None
            and row["max_lon"] is not None
        ):
            return (
                float(row["min_lat"]),
                float(row["max_lat"]),
                float(row["min_lon"]),
                float(row["max_lon"]),
            )
        return None

    def fit_to_photos(self) -> None:
        """Fits map bounds snugly to valid plotted geotagged photos in the catalog."""
        bounds = self._get_clean_geotag_bounds()
        if bounds:
            min_la, max_la, min_lo, max_lo = bounds
            lat_pad = max(0.015, (max_la - min_la) * 0.02)
            lon_pad = max(0.015, (max_lo - min_lo) * 0.02)
            min_la = max(-85.0, min_la - lat_pad)
            max_la = min(85.0, max_la + lat_pad)
            min_lo = max(-180.0, min_lo - lon_pad)
            max_lo = min(180.0, max_lo + lon_pad)
        else:
            min_la, max_la, min_lo, max_lo = -50.0, 70.0, -130.0, 140.0

        self._pending_fit_bounds = (min_la, max_la, min_lo, max_lo)
        self.set_viewport_bbox(min_la, max_la, min_lo, max_lo)
        if self._web_loaded and self.web_view is not None:
            self._pending_fit_bounds = None
            self.web_view.page().runJavaScript(
                f"fitMapBounds({min_la}, {max_la}, {min_lo}, {max_lo});"
            )

    def refresh_current_viewport(self) -> None:
        """Refreshes clusters for the current bounding box."""
        self.set_viewport_bbox(self._min_lat, self._max_lat, self._min_lon, self._max_lon, self._repo)

    def set_viewport_bbox(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        repo: CatalogRepository | None = None,
    ) -> list[dict[str, Any]]:
        """
        Updates the active bounding box, queries `media_rtree` via `CatalogRepository.query_map_clusters`,
        clusters the results, updates the map pins (both WebEngine and vector canvas), and emits `bbox_changed`.
        """
        if repo is not None:
            self._repo = repo
        return self._update_viewport_clusters(min_lat, max_lat, min_lon, max_lon)

    @property
    def current_clusters(self) -> list[dict[str, Any]]:
        return self._current_clusters
