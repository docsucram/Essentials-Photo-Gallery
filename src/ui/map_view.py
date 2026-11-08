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
from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import QFile, QModelIndex, QObject, QPointF, QRectF, QSize, Qt, Signal, Slot
from PySide6.QtGui import (
    QBrush,
    QColor,
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
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.db.repository import CatalogRepository
from src.ui.icon_provider import get_icon

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
      attribution: 'Tiles &copy; Esri &mdash; Sources: Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, Esri Japan, METI, Esri China (Hong Kong), Esri (Thailand), TomTom'
    }}),
    'street': L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }}),
    'satellite': L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      maxZoom: 19,
      attribution: 'Tiles &copy; Esri'
    }}),
    'terrain': L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      maxZoom: 19,
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
    isInternalMove = true;
    map.flyTo([lat, lon], zoom || 13, {{ duration: 1.0 }});
  }}

  function fitMapBounds(min_lat, max_lat, min_lon, max_lon) {{
    if (!map) return;
    isInternalMove = true;
    map.fitBounds([[min_lat, min_lon], [max_lat, max_lon]], {{ padding: [30, 30], maxZoom: 15 }});
  }}

  function setMapBounds(min_lat, max_lat, min_lon, max_lon) {{
    fitMapBounds(min_lat, max_lat, min_lon, max_lon);
  }}

  function initMap() {{
    if (typeof L === 'undefined') {{
      document.getElementById('map').innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#1E395B;font-size:13px;text-align:center;padding:20px;">Leaflet offline or unavailable. Using vector map canvas.</div>';
      return;
    }}
    map = L.map('map', {{
      zoomControl: true,
      minZoom: 2,
      maxZoom: 19
    }}).setView([25.0, 0.0], 2);

    switchBaseLayer('clean');

    clusterLayer = L.layerGroup().addTo(map);

    map.on('click', function(e) {{
      if (window.qtBridge && typeof window.qtBridge.onMapBackgroundClicked === 'function') {{
        window.qtBridge.onMapBackgroundClicked();
      }}
    }});

    let boundsTimer = null;
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
      }}, 150);
    }});

    if (window.pendingClusters) {{
      updateClusters(window.pendingClusters);
      window.pendingClusters = null;
    }}
  }}

  function updateClusters(clustersData) {{
    if (!clusterLayer) {{
      window.pendingClusters = clustersData;
      return;
    }}
    clusterLayer.clearLayers();
    const clusters = (typeof clustersData === 'string') ? JSON.parse(clustersData) : clustersData;
    if (!clusters || !clusters.length) return;

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
      const marker = L.marker([c.lat, c.lon], {{ icon: icon }}).addTo(clusterLayer);
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
    }});
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
            self.cluster_clicked.emit(mids)
            self.cluster_clicked_data.emit(mids, label)
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
        self.onClusterSelected(cluster_json)

    @Slot()
    def onMapBackgroundClicked(self) -> None:
        self.map_background_clicked.emit()

    @Slot(float, float, float, float)
    def onBoundsChanged(
        self, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ) -> None:
        self.bounds_changed.emit(min_lat, max_lat, min_lon, max_lon)


class _MapLocationSearchBox(QLineEdit):
    """Search input with an instant gazetteer dropdown popup powered by QCompleter."""

    location_chosen = Signal(float, float, str)  # lat, lon, label

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setPlaceholderText("Find place on map...")
        self.setClearButtonEnabled(True)
        self.setMinimumWidth(200)
        self.setMaximumWidth(320)
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

    def _on_text_edited(self, text: str) -> None:
        q = text.strip()
        if len(q) < 2:
            self._model.clear()
            return
        from src.core.geotag_resolver import search_places_offline

        results = search_places_offline(q, limit=10)
        if not results:
            self._model.clear()
            return
        self._model.clear()
        for r in results:
            label = r.get("formatted_address") or r.get("name") or ""
            lat = float(r["lat"])
            lon = float(r["lon"])
            item = QStandardItem(label)
            item.setData((lat, lon, label), Qt.UserRole)
            self._model.appendRow(item)
        self._completer.complete()

    def _on_item_activated(self, index: QModelIndex) -> None:
        data = self._model.data(index, Qt.UserRole)
        if data and len(data) >= 3:
            lat, lon, label = data[0], data[1], data[2]
            self.setText(label)
            self.location_chosen.emit(lat, lon, label)

    def _on_return_pressed(self) -> None:
        popup = self._completer.popup()
        if popup and popup.isVisible():
            cur = popup.currentIndex()
            if not cur.isValid() and self._model.rowCount() > 0:
                cur = self._model.index(0, 0)
            if cur.isValid():
                self._on_item_activated(cur)
                popup.hide()


class _InteractiveMapCanvas(QWidget):
    """
    Offline vector map canvas that runs in environments where QtWebEngine is not installed.
    Renders vector continents, lat/lon graticules, and GPS cluster pins.
    """

    cluster_selected = Signal(list, str)
    cluster_clicked = Signal(list)
    cluster_clicked_data = Signal(list, str)
    viewport_bbox_dragged = Signal(float, float, float, float)
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

        painter.end()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
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
    Renders an interactive Leaflet map (Voyager, OSM, Satellite, Terrain) via QWebEngineView
    with automatic silent fallback to high-performance vector canvas (_InteractiveMapCanvas).
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

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # Top Map Control Bar
        top_bar = QFrame(self)
        top_bar.setFixedHeight(40)
        top_bar.setStyleSheet(
            "background-color: #EEF3F8; border-bottom: 1px solid #C8D2E0;"
        )
        t_hbox = QHBoxLayout(top_bar)
        t_hbox.setContentsMargins(10, 4, 10, 4)
        t_hbox.setSpacing(8)

        lbl_title = QLabel("🌍 World Map", top_bar)
        lbl_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        lbl_title.setStyleSheet("color: #1E395B;")
        t_hbox.addWidget(lbl_title)

        # Instant Offline Gazetteer Place Search
        self.search_box = _MapLocationSearchBox(top_bar)
        self.search_box.location_chosen.connect(
            lambda lat, lon, _label: self.fly_to(lat, lon, 13)
        )
        t_hbox.addWidget(self.search_box)

        # Map Style Selector (Clean Voyager, Street OSM, Satellite Esri, Terrain Topo)
        lbl_style = QLabel("Style:", top_bar)
        lbl_style.setStyleSheet("color: #1E395B; font-size: 9pt;")
        t_hbox.addWidget(lbl_style)

        self.combo_style = QComboBox(top_bar)
        self.combo_style.addItems(
            ["Clean (Esri)", "Street (OSM)", "Satellite (Esri)", "Terrain (Topo)"]
        )
        self.combo_style.setFixedHeight(26)
        self.combo_style.setStyleSheet(
            "QComboBox { background-color: #FFFFFF; border: 1px solid #7DA2CE; border-radius: 3px; padding: 2px 8px; font-family: 'Segoe UI', sans-serif; font-size: 9pt; } "
            "QComboBox::drop-down { border: none; width: 16px; }"
        )
        self.combo_style.currentIndexChanged.connect(self._on_style_changed)
        t_hbox.addWidget(self.combo_style)

        # Fit to Photos button
        self.btn_fit_photos = QPushButton("Fit to photos", top_bar)
        self.btn_fit_photos.setIcon(get_icon("fit_photos", 16))
        self.btn_fit_photos.setIconSize(QSize(16, 16))
        self.btn_fit_photos.setCursor(Qt.PointingHandCursor)
        self.btn_fit_photos.setFixedHeight(26)
        self.btn_fit_photos.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; border: 1px solid #7DA2CE; border-radius: 3px; padding: 2px 10px; font-family: 'Segoe UI', sans-serif; font-size: 9pt; } "
            "QPushButton:hover { background-color: #E5F3FF; border-color: #0066CC; }"
        )
        self.btn_fit_photos.clicked.connect(self.fit_to_photos)
        t_hbox.addWidget(self.btn_fit_photos)

        t_hbox.addStretch(1)

        self.lbl_cluster_summary = QLabel("0 photos in view", top_bar)
        self.lbl_cluster_summary.setStyleSheet(
            "color: #29588C; font-weight: bold; font-family: 'Segoe UI', sans-serif; font-size: 9pt;"
        )
        t_hbox.addWidget(self.lbl_cluster_summary)

        self.btn_filter_grid = QPushButton("View area in grid", top_bar)
        self.btn_filter_grid.setIcon(get_icon("all_media", 16))
        self.btn_filter_grid.setIconSize(QSize(16, 16))
        self.btn_filter_grid.setCursor(Qt.PointingHandCursor)
        self.btn_filter_grid.setFixedHeight(26)
        self.btn_filter_grid.setStyleSheet(
            "QPushButton { background-color: #0066CC; color: #FFFFFF; border: 1px solid #0055AA; border-radius: 3px; padding: 2px 12px; font-family: 'Segoe UI', sans-serif; font-size: 9pt; font-weight: 600; } "
            "QPushButton:hover { background-color: #004D99; }"
        )
        self.btn_filter_grid.clicked.connect(
            lambda: self.filter_to_bbox_requested.emit(
                self._min_lat, self._max_lat, self._min_lon, self._max_lon
            )
        )
        t_hbox.addWidget(self.btn_filter_grid)

        self.btn_close_map = QPushButton("Return to grid", top_bar)
        self.btn_close_map.setIcon(get_icon("thumbnail_grid", 16))
        self.btn_close_map.setIconSize(QSize(16, 16))
        self.btn_close_map.setCursor(Qt.PointingHandCursor)
        self.btn_close_map.setFixedHeight(26)
        self.btn_close_map.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; border: 1px solid #7DA2CE; border-radius: 3px; padding: 2px 10px; font-family: 'Segoe UI', sans-serif; font-size: 9pt; } "
            "QPushButton:hover { background-color: #FFEFEF; border-color: #CC0000; color: #CC0000; }"
        )
        self.btn_close_map.clicked.connect(self.close_requested.emit)
        t_hbox.addWidget(self.btn_close_map)

        vbox.addWidget(top_bar)

        # Full-size Central Map View Stack (fills 100% of workspace, no duplicate right splitter column!)
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

        # Backwards-compatibility stubs for tests
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

    def _attach_cluster_thumbnails(self, clusters: list[dict[str, Any]]) -> None:
        """Enriches cluster dictionaries with thumbnail base64 data URIs and QPixmaps."""
        if not clusters or self._thumb_repo is None:
            return
        lead_ids = [c["media_ids"][0] for c in clusters if c.get("media_ids")]
        if not lead_ids:
            return
        try:
            blobs = self._thumb_repo.get_webp_blobs_batch(lead_ids)
            for c in clusters:
                if not c.get("media_ids"):
                    continue
                lead_id = c["media_ids"][0]
                blob = blobs.get(lead_id)
                if blob:
                    b64 = base64.b64encode(blob).decode("ascii")
                    c["thumb_url"] = f"data:image/webp;base64,{b64}"
                    pix = QPixmap()
                    if pix.loadFromData(blob, "WEBP"):
                        c["pixmap"] = pix
        except Exception:
            pass

    def _on_style_changed(self, idx: int) -> None:
        styles = ["clean", "street", "satellite", "terrain"]
        if 0 <= idx < len(styles):
            self.set_map_style(styles[idx])

    def set_map_style(self, style_name: str) -> None:
        """Switches base tile layer dynamically."""
        self._current_style = style_name
        if self._web_loaded and self.web_view is not None:
            self.web_view.page().runJavaScript(f"switchBaseLayer('{style_name}');")

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

    def _on_web_load_finished(self, ok: bool) -> None:
        self._web_loaded = ok
        if ok and self._current_clusters and self.web_view is not None:
            self.web_view.page().runJavaScript(
                f"updateClusters({_clusters_to_json(self._current_clusters)});"
            )

    def _on_web_bounds_changed(
        self, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ) -> None:
        """Called when user pans/zooms the Leaflet map."""
        self._min_lat = max(-85.0, min(85.0, float(min_lat)))
        self._max_lat = max(-85.0, min(85.0, float(max_lat)))
        self._min_lon = max(-180.0, min(180.0, float(min_lon)))
        self._max_lon = max(-180.0, min(180.0, float(max_lon)))

        self.spin_min_lat.setValue(self._min_lat)
        self.spin_max_lat.setValue(self._max_lat)
        self.spin_min_lon.setValue(self._min_lon)
        self.spin_max_lon.setValue(self._max_lon)

        lat_span = max(0.1, self._max_lat - self._min_lat)
        grid_deg = max(0.05, lat_span / 25.0)

        raw_points: list[dict[str, Any]] = []
        if self._repo is not None:
            raw_points = self._repo.query_map_clusters(
                self._min_lat, self._max_lat, self._min_lon, self._max_lon, grid_degrees=grid_deg
            )

        self._current_clusters = cluster_gps_points(raw_points, grid_degrees=grid_deg)
        self._attach_cluster_thumbnails(self._current_clusters)
        self.canvas.set_bounds_and_clusters(
            self._min_lat, self._max_lat, self._min_lon, self._max_lon, self._current_clusters
        )

        total_items = sum(c["count"] for c in self._current_clusters)
        self.lbl_cluster_summary.setText(
            f"{total_items} geotagged items in {len(self._current_clusters)} clusters"
        )

        self.cluster_list.clear()
        for c in self._current_clusters:
            item = QListWidgetItem(
                f"📍 {c['label']} ({c['count']} items)", self.cluster_list
            )
            item.setData(Qt.UserRole, (c["media_ids"], c["label"]))

        if self._web_loaded and self.web_view is not None:
            self.web_view.page().runJavaScript(
                f"updateClusters({_clusters_to_json(self._current_clusters)});"
            )

        self.bbox_changed.emit(self._min_lat, self._max_lat, self._min_lon, self._max_lon)

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

    def fit_to_photos(self) -> None:
        """Fits map bounds to encompass all geotagged photos in the catalog."""
        if self._repo is not None:
            bounds = self._repo.get_geotag_bounds()
            if bounds:
                min_la, max_la, min_lo, max_lo = bounds
                lat_pad = max(0.04, (max_la - min_la) * 0.15)
                lon_pad = max(0.04, (max_lo - min_lo) * 0.15)
                min_la = max(-85.0, min_la - lat_pad)
                max_la = min(85.0, max_la + lat_pad)
                min_lo = max(-180.0, min_lo - lon_pad)
                max_lo = min(180.0, max_lo + lon_pad)
                self.set_viewport_bbox(min_la, max_la, min_lo, max_lo)
                if self._web_loaded and self.web_view is not None:
                    self.web_view.page().runJavaScript(
                        f"fitMapBounds({min_la}, {max_la}, {min_lo}, {max_lo});"
                    )
                return
        self.set_viewport_bbox(-60.0, 75.0, -140.0, 140.0)

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

        self._min_lat = min(float(min_lat), float(max_lat))
        self._max_lat = max(float(min_lat), float(max_lat))
        self._min_lon = min(float(min_lon), float(max_lon))
        self._max_lon = max(float(min_lon), float(max_lon))

        self.spin_min_lat.setValue(self._min_lat)
        self.spin_max_lat.setValue(self._max_lat)
        self.spin_min_lon.setValue(self._min_lon)
        self.spin_max_lon.setValue(self._max_lon)

        lat_span = max(0.1, self._max_lat - self._min_lat)
        grid_deg = max(0.05, lat_span / 25.0)

        raw_points: list[dict[str, Any]] = []
        if self._repo is not None:
            raw_points = self._repo.query_map_clusters(
                self._min_lat, self._max_lat, self._min_lon, self._max_lon, grid_degrees=grid_deg
            )

        self._current_clusters = cluster_gps_points(raw_points, grid_degrees=grid_deg)
        self._attach_cluster_thumbnails(self._current_clusters)

        self.canvas.set_bounds_and_clusters(
            self._min_lat,
            self._max_lat,
            self._min_lon,
            self._max_lon,
            self._current_clusters,
        )

        if self._web_loaded and self.web_view is not None:
            self.web_view.page().runJavaScript(
                f"updateClusters({_clusters_to_json(self._current_clusters)});"
            )

        total_items = sum(c["count"] for c in self._current_clusters)
        self.lbl_cluster_summary.setText(
            f"{total_items} geotagged items in {len(self._current_clusters)} clusters"
        )

        self.cluster_list.clear()
        for c in self._current_clusters:
            item = QListWidgetItem(
                f"📍 {c['label']} ({c['count']} items)", self.cluster_list
            )
            item.setData(Qt.UserRole, (c["media_ids"], c["label"]))

        self.bbox_changed.emit(
            self._min_lat, self._max_lat, self._min_lon, self._max_lon
        )
        return self._current_clusters

    @property
    def current_clusters(self) -> list[dict[str, Any]]:
        return self._current_clusters
