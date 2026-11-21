/**
 * Hierarchical pixel-distance spatial clustering Web Worker for Essentials Photo Gallery.
 * Builds a Supercluster-style pyramid across zoom levels [maxZoom..0] once per dataset version
 * so map panning and zooming perform zero SQLite round-trips and zero main-thread clustering.
 */

function projectPoint(lat, lon, zoom) {
  const scale = 256 * Math.pow(2, zoom);
  const clampedLat = Math.max(-85.0511287798, Math.min(85.0511287798, lat));
  const sin = Math.sin((clampedLat * Math.PI) / 180);
  return {
    x: scale * (lon / 360 + 0.5),
    y: scale * (0.5 - Math.log((1 + sin) / (1 - sin)) / (4 * Math.PI)),
  };
}

function clusterAtZoom(points, zoom, spacing) {
  const cells = new Map();
  const clusters = [];
  const spacingSq = spacing * spacing;
  const sorted = points.slice().sort(
    (a, b) => (b.count - a.count) || (a.rep_media_id - b.rep_media_id)
  );

  for (let i = 0; i < sorted.length; i++) {
    const pt = sorted[i];
    const px = projectPoint(pt.lat, pt.lon, zoom);
    const gx = Math.floor(px.x / spacing);
    const gy = Math.floor(px.y / spacing);

    let nearest = null;
    let bestDistSq = Infinity;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const bucket = cells.get((gx + dx) + ":" + (gy + dy));
        if (!bucket) continue;
        for (let j = 0; j < bucket.length; j++) {
          const cand = bucket[j];
          const ddx = cand.px.x - px.x;
          const ddy = cand.px.y - px.y;
          const d2 = ddx * ddx + ddy * ddy;
          if (d2 < spacingSq && d2 < bestDistSq) {
            nearest = cand.cluster;
            bestDistSq = d2;
          }
        }
      }
    }

    const ptMinLat = pt.min_lat !== undefined ? pt.min_lat : pt.lat - 0.005;
    const ptMaxLat = pt.max_lat !== undefined ? pt.max_lat : pt.lat + 0.005;
    const ptMinLon = pt.min_lon !== undefined ? pt.min_lon : pt.lon - 0.005;
    const ptMaxLon = pt.max_lon !== undefined ? pt.max_lon : pt.lon + 0.005;

    if (nearest) {
      nearest.count += pt.count;
      if (!nearest.label && pt.label) {
        nearest.label = pt.label;
      }
      if (ptMinLat < nearest.min_lat) nearest.min_lat = ptMinLat;
      if (ptMaxLat > nearest.max_lat) nearest.max_lat = ptMaxLat;
      if (ptMinLon < nearest.min_lon) nearest.min_lon = ptMinLon;
      if (ptMaxLon > nearest.max_lon) nearest.max_lon = ptMaxLon;
    } else {
      const cluster = {
        lat: pt.lat,
        lon: pt.lon,
        count: pt.count,
        rep_media_id: pt.rep_media_id,
        rep_mtime: pt.rep_mtime || 0,
        label: pt.label || "",
        min_lat: ptMinLat,
        max_lat: ptMaxLat,
        min_lon: ptMinLon,
        max_lon: ptMaxLon,
      };
      clusters.push(cluster);
      const key = gx + ":" + gy;
      let bucket = cells.get(key);
      if (!bucket) {
        bucket = [];
        cells.set(key, bucket);
      }
      bucket.push({ px: px, cluster: cluster });
    }
  }
  return clusters;
}

function buildPyramid(points, maxZoom, spacing) {
  const levels = new Map();
  let current = points;
  for (let z = maxZoom; z >= 0; z--) {
    current = clusterAtZoom(current, z, spacing);
    levels.set(z, current);
  }
  return levels;
}

function isInPixelBounds(cluster, zoom, bounds) {
  if (!bounds) return true;
  const px = projectPoint(cluster.lat, cluster.lon, zoom);
  const buffer = 160;
  return (
    px.x >= bounds.minX - buffer &&
    px.x <= bounds.maxX + buffer &&
    px.y >= bounds.minY - buffer &&
    px.y <= bounds.maxY + buffer
  );
}

let cachedState = null;

self.onmessage = function (ev) {
  const data = ev.data || {};
  try {
    const version = data.version || 0;
    const maxZoom = data.maxZoom || 18;
    const spacing = data.spacing || 52;
    if (
      !cachedState ||
      cachedState.version !== version ||
      cachedState.maxZoom !== maxZoom ||
      cachedState.spacing !== spacing
    ) {
      const pts = Array.isArray(data.points) ? data.points : [];
      cachedState = {
        version: version,
        maxZoom: maxZoom,
        spacing: spacing,
        levels: buildPyramid(pts, maxZoom, spacing),
      };
    }
    const z = Math.max(0, Math.min(cachedState.maxZoom, Math.floor(data.zoom || 2)));
    const levelClusters = cachedState.levels.get(z) || [];
    const visible = levelClusters.filter(function (c) {
      return isInPixelBounds(c, data.zoom || z, data.bounds);
    });
    self.postMessage({
      requestId: data.requestId,
      clusters: visible,
    });
  } catch (err) {
    self.postMessage({
      requestId: data.requestId,
      error: String(err),
      clusters: [],
    });
  }
};
