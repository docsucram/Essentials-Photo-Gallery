"""
Optional On-Device Semantic AI Search & Subjects Engine (MobileCLIP-S1 ONNX Add-on).

Architecture highlights:
- Dedicated `semantic_vectors.db` (separate from `catalog.db`) storing L2-normalized
  float16 embeddings (512-d -> 1 KB/photo) indexed strictly from cached thumbnails.
- Built-in `SUBJECT_PRESETS` with per-subject calibrated similarity thresholds so the
  Navigation Pane shows real matched counts instead of raw total library counts.
- Optional ~80 MB ONNX add-on download via Options -> Add-ons with HTTP Range resume
  and atomic `.part` replacement.
- Reciprocal Rank Fusion (`fuse_semantic_and_fts_scores`) blending natural-language
  semantic matches with SQLite FTS5 keyword/tag/OCR matches.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from src.utils.paths import get_app_data_dir

logger = logging.getLogger(__name__)

MODEL_VERSION_ID = "mobileclip_s1"
EMBED_DIM = 512
SEMANTIC_ADDON_SIZE_MB = 82

# (fname, url, approx_bytes)
SEMANTIC_MODEL_FILES: tuple[tuple[str, str, int], ...] = (
    (
        "mobileclip_s1_image.onnx",
        "https://huggingface.co/Xenova/mobileclip_s1/resolve/main/onnx/vision_model_int8.onnx",
        24 * 1024 * 1024,
    ),
    (
        "mobileclip_s1_text.onnx",
        "https://huggingface.co/Xenova/mobileclip_s1/resolve/main/onnx/text_model_int8.onnx",
        58 * 1024 * 1024,
    ),
)

# 12 curated built-in Subject presets: (key, label, prompts, calibrated_threshold)
SUBJECT_PRESETS: tuple[tuple[str, str, tuple[str, ...], float], ...] = (
    ("family", "Family", ("a family photo together", "relatives gathered together"), 0.25),
    ("kids", "Kids", ("a photo of children playing or smiling", "toddlers and kids"), 0.25),
    ("pets", "Pets", ("a pet dog or cat animal photo", "cute dog or cat indoors or outdoors"), 0.24),
    ("portraits", "Portraits", ("a close-up portrait photograph of a person", "headshot portrait"), 0.24),
    ("food", "Food", ("a photograph of plated food, meal or dessert", "restaurant dish"), 0.24),
    ("sports", "Sports", ("people playing sports or athletic action", "running cycling soccer"), 0.25),
    ("landscape", "Landscape", ("scenic landscape mountains hills countryside view", "panoramic nature scenery"), 0.23),
    ("night", "Night", ("nighttime photograph city lights stars evening dark sky", "night scene"), 0.24),
    ("architecture", "Architecture", ("building architecture cityscape bridge facade", "historic or modern building"), 0.24),
    ("beach", "Beach & Coast", ("beach ocean coastline sea waves sand", "seaside vacation photo"), 0.24),
    ("nature", "Nature & Flowers", ("flowers plants trees wildlife nature closeup", "botanical garden"), 0.24),
    ("documents", "Documents & Notes", ("document page receipt text screenshot", "printed paper or whiteboards"), 0.25),
)


def get_semantic_model_dir() -> Path:
    """Return the AppData directory storing optional MobileCLIP-S1 ONNX models."""
    return get_app_data_dir() / "models" / "mobileclip_s1"


def get_semantic_addon_status() -> dict[str, Any]:
    """Probe installation status of optional MobileCLIP-S1 ONNX models without loading ONNX."""
    model_dir = get_semantic_model_dir()
    img_model = model_dir / "mobileclip_s1_image.onnx"
    txt_model = model_dir / "mobileclip_s1_text.onnx"
    installed = img_model.is_file() and img_model.stat().st_size > 1024
    size_mb = SEMANTIC_ADDON_SIZE_MB
    if installed:
        total = img_model.stat().st_size + (txt_model.stat().st_size if txt_model.is_file() else 0)
        size_mb = max(1.0, round(total / (1024 * 1024), 1))
    return {
        "installed": installed,
        "text_encoder_installed": txt_model.is_file() and txt_model.stat().st_size > 1024,
        "dir": model_dir,
        "size_mb": size_mb,
        "image_model_path": img_model if installed else None,
        "text_model_path": txt_model if txt_model.is_file() else None,
    }


def download_semantic_models(
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    timeout: float = 30.0,
) -> Path | None:
    """
    Download MobileCLIP-S1 ONNX models (~82 MB) to AppData with HTTP Range resume
    and atomic `.part` swap.
    """
    dest_dir = get_semantic_model_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    total_expected = sum(sz for _, _, sz in SEMANTIC_MODEL_FILES)
    cumulative_done = 0

    try:
        from src.version import APP_VERSION

        for fname, url, _approx_sz in SEMANTIC_MODEL_FILES:
            dest_file = dest_dir / fname
            if dest_file.is_file() and dest_file.stat().st_size > 1024:
                cumulative_done += dest_file.stat().st_size
                if progress_cb is not None:
                    progress_cb(cumulative_done, max(total_expected, cumulative_done))
                continue

            tmp_file = dest_dir / f"{fname}.part"
            existing_bytes = tmp_file.stat().st_size if tmp_file.exists() else 0
            headers = {"User-Agent": f"EssentialsPhotoGallery/{APP_VERSION}"}
            if existing_bytes > 0:
                headers["Range"] = f"bytes={existing_bytes}-"

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status_code = getattr(resp, "status", 200)
                append_mode = (status_code == 206 and existing_bytes > 0)
                file_done = existing_bytes if append_mode else 0
                open_mode = "ab" if append_mode else "wb"
                with open(tmp_file, open_mode) as fh:
                    while True:
                        if cancel_cb is not None and cancel_cb():
                            raise InterruptedError("Download cancelled")
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
                        file_done += len(chunk)
                        if progress_cb is not None:
                            progress_cb(cumulative_done + file_done, max(total_expected, cumulative_done + file_done))

            if tmp_file.stat().st_size < 1024:
                raise ValueError(f"Downloaded semantic model {fname} is unexpectedly small")
            os.replace(tmp_file, dest_file)
            cumulative_done += dest_file.stat().st_size
        return dest_dir
    except Exception as exc:
        logger.warning("MobileCLIP-S1 add-on download failed: %s", exc)
        return None


def uninstall_semantic_models() -> bool:
    """Remove downloaded MobileCLIP-S1 ONNX models from AppData."""
    dest_dir = get_semantic_model_dir()
    removed = False
    for fname, _, _ in SEMANTIC_MODEL_FILES:
        for cand in (dest_dir / fname, dest_dir / f"{fname}.part"):
            if cand.exists():
                try:
                    cand.unlink()
                    removed = True
                except OSError:
                    pass
    return removed


def _l2_normalize_1d(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(v))
    if norm > 1e-8:
        v = v / norm
    return v


def fuse_semantic_and_fts_scores(
    fts_ids: Sequence[int],
    semantic_scored: Sequence[tuple[int, float]],
    *,
    k: int = 60,
    semantic_weight: float = 1.0,
    fts_weight: float = 1.15,
) -> list[int]:
    """
    Combine keyword/tag/OCR FTS5 results (`fts_ids` in rank order) and semantic
    cosine matches (`semantic_scored`) using Reciprocal Rank Fusion (RRF).
    """
    scores: dict[int, float] = {}
    kk = max(1, int(k))
    for rank, mid in enumerate(fts_ids):
        imid = int(mid)
        scores[imid] = scores.get(imid, 0.0) + float(fts_weight) / float(kk + rank + 1)

    ordered_sem = sorted(semantic_scored, key=lambda x: x[1], reverse=True)
    for rank, (mid, _sim) in enumerate(ordered_sem):
        imid = int(mid)
        scores[imid] = scores.get(imid, 0.0) + float(semantic_weight) / float(kk + rank + 1)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [mid for mid, _sc in ranked]


class SemanticVectorRepository:
    """
    Thread-safe repository managing `semantic_vectors.db` separately from `catalog.db`.
    Stores compact L2-normalized float16 embeddings (~1 KB/photo) and keeps a lazy
    in-memory float32 matrix cache for < 5 ms vector similarity queries across 100k+ photos.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._local = threading.local()
        self._matrix_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(str(self.db_path), timeout=5.0, check_same_thread=False)
            c.execute("PRAGMA journal_mode = WAL;")
            c.execute("PRAGMA synchronous = NORMAL;")
            c.execute("PRAGMA busy_timeout = 5000;")
            self._local.conn = c
        return c

    def _init_schema(self) -> None:
        with self._lock:
            c = self._conn()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_vectors (
                    media_id INTEGER PRIMARY KEY,
                    mtime INTEGER NOT NULL,
                    model_ver TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vec_fp16 BLOB NOT NULL
                );
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS subject_vectors (
                    subject_key TEXT NOT NULL,
                    model_ver TEXT NOT NULL,
                    vec_fp16 BLOB NOT NULL,
                    PRIMARY KEY (subject_key, model_ver)
                );
                """
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_sem_model ON semantic_vectors(model_ver);"
            )
            c.commit()

    def upsert_vectors_batch(
        self,
        rows: Sequence[tuple[int, int, str, np.ndarray]],
    ) -> None:
        """Persist `[(media_id, mtime, model_ver, vec)]` as normalized float16 BLOBs."""
        if not rows:
            return
        payload: list[tuple[int, int, str, int, bytes]] = []
        touched_models: set[str] = set()
        for mid, mtime, mver, vec in rows:
            normed = _l2_normalize_1d(vec)
            fp16 = normed.astype(np.float16).tobytes()
            payload.append((int(mid), int(mtime), str(mver), int(normed.size), fp16))
            touched_models.add(str(mver))

        with self._lock:
            c = self._conn()
            c.executemany(
                """
                INSERT INTO semantic_vectors(media_id, mtime, model_ver, dim, vec_fp16)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(media_id) DO UPDATE SET
                    mtime = excluded.mtime,
                    model_ver = excluded.model_ver,
                    dim = excluded.dim,
                    vec_fp16 = excluded.vec_fp16;
                """,
                payload,
            )
            c.commit()
            for mv in touched_models:
                self._matrix_cache.pop(mv, None)

    def get_indexed_mtimes(
        self,
        media_ids: Sequence[int] | None = None,
        *,
        model_ver: str = MODEL_VERSION_ID,
    ) -> dict[int, int]:
        """Return `{media_id: mtime}` already indexed under `model_ver`."""
        with self._lock:
            c = self._conn()
            out: dict[int, int] = {}
            if media_ids is None:
                rows = c.execute(
                    "SELECT media_id, mtime FROM semantic_vectors WHERE model_ver = ?;",
                    (model_ver,),
                ).fetchall()
                for mid, mt in rows:
                    out[int(mid)] = int(mt)
                return out
            ids = [int(m) for m in media_ids]
            for i in range(0, len(ids), 900):
                chunk = ids[i : i + 900]
                ph = ",".join("?" for _ in chunk)
                rows = c.execute(
                    f"SELECT media_id, mtime FROM semantic_vectors WHERE model_ver = ? AND media_id IN ({ph});",
                    [model_ver, *chunk],
                ).fetchall()
                for mid, mt in rows:
                    out[int(mid)] = int(mt)
            return out

    def delete_vectors(self, media_ids: Sequence[int]) -> None:
        if not media_ids:
            return
        ids = [int(m) for m in media_ids]
        with self._lock:
            c = self._conn()
            for i in range(0, len(ids), 900):
                chunk = ids[i : i + 900]
                ph = ",".join("?" for _ in chunk)
                c.execute(f"DELETE FROM semantic_vectors WHERE media_id IN ({ph});", chunk)
            c.commit()
            self._matrix_cache.clear()

    def count_indexed(self, *, model_ver: str = MODEL_VERSION_ID) -> int:
        with self._lock:
            c = self._conn()
            row = c.execute(
                "SELECT COUNT(*) FROM semantic_vectors WHERE model_ver = ?;",
                (model_ver,),
            ).fetchone()
            return int(row[0]) if row else 0

    def put_subject_vectors(
        self,
        mapping: dict[str, np.ndarray],
        *,
        model_ver: str = MODEL_VERSION_ID,
    ) -> None:
        if not mapping:
            return
        payload = [
            (str(key), str(model_ver), _l2_normalize_1d(vec).astype(np.float16).tobytes())
            for key, vec in mapping.items()
        ]
        with self._lock:
            c = self._conn()
            c.executemany(
                """
                INSERT INTO subject_vectors(subject_key, model_ver, vec_fp16)
                VALUES (?, ?, ?)
                ON CONFLICT(subject_key, model_ver) DO UPDATE SET
                    vec_fp16 = excluded.vec_fp16;
                """,
                payload,
            )
            c.commit()

    def get_subject_vectors(
        self, *, model_ver: str = MODEL_VERSION_ID
    ) -> dict[str, np.ndarray]:
        with self._lock:
            c = self._conn()
            rows = c.execute(
                "SELECT subject_key, vec_fp16 FROM subject_vectors WHERE model_ver = ?;",
                (model_ver,),
            ).fetchall()
        out: dict[str, np.ndarray] = {}
        for key, blob in rows:
            vec = np.frombuffer(bytes(blob), dtype=np.float16).astype(np.float32)
            out[str(key)] = _l2_normalize_1d(vec)
        return out

    def load_matrix(
        self, *, model_ver: str = MODEL_VERSION_ID
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(media_ids_int64, l2_normalized_float32_matrix)` from cache or DB."""
        with self._lock:
            cached = self._matrix_cache.get(model_ver)
            if cached is not None:
                return cached
            c = self._conn()
            rows = c.execute(
                "SELECT media_id, dim, vec_fp16 FROM semantic_vectors WHERE model_ver = ? ORDER BY media_id ASC;",
                (model_ver,),
            ).fetchall()
            if not rows:
                empty = (np.empty((0,), dtype=np.int64), np.empty((0, EMBED_DIM), dtype=np.float32))
                self._matrix_cache[model_ver] = empty
                return empty
            dim = int(rows[0][1] or EMBED_DIM)
            valid_ids: list[int] = []
            vecs: list[np.ndarray] = []
            for mid, rdim, blob in rows:
                if int(rdim) != dim:
                    continue
                arr = np.frombuffer(bytes(blob), dtype=np.float16).astype(np.float32)
                if arr.size == dim:
                    valid_ids.append(int(mid))
                    vecs.append(arr)
            if not vecs:
                empty = (np.empty((0,), dtype=np.int64), np.empty((0, dim), dtype=np.float32))
                self._matrix_cache[model_ver] = empty
                return empty
            mat = np.vstack(vecs)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.where(norms > 1e-8, norms, 1.0)
            mat = mat / norms
            pair = (np.array(valid_ids, dtype=np.int64), mat)
            self._matrix_cache[model_ver] = pair
            return pair

    def close(self) -> None:
        with self._lock:
            c = getattr(self._local, "conn", None)
            if c is not None:
                try:
                    c.close()
                except Exception:
                    pass
                self._local.conn = None


class SemanticEngine:
    """
    Lazy ONNX Runtime wrapper + fast NumPy vector matcher for MobileCLIP-S1 embeddings.
    Works cleanly offline; `is_available()` returns False unless optional ONNX models exist.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._img_sess: Any = None
        self._txt_sess: Any = None

    @staticmethod
    def is_available() -> bool:
        return bool(get_semantic_addon_status().get("installed", False))

    def _ensure_image_session(self) -> Any:
        if self._img_sess is not None:
            return self._img_sess
        with self._lock:
            if self._img_sess is not None:
                return self._img_sess
            st = get_semantic_addon_status()
            img_p = st.get("image_model_path")
            if not img_p:
                return None
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 2))
                self._img_sess = ort.InferenceSession(
                    str(img_p), sess_options=opts, providers=["CPUExecutionProvider"]
                )
            except Exception as exc:
                logger.warning("Failed to load MobileCLIP image ONNX model: %s", exc)
                self._img_sess = None
            return self._img_sess

    def _ensure_text_session(self) -> Any:
        if self._txt_sess is not None:
            return self._txt_sess
        with self._lock:
            if self._txt_sess is not None:
                return self._txt_sess
            st = get_semantic_addon_status()
            txt_p = st.get("text_model_path")
            if not txt_p:
                return None
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 2))
                self._txt_sess = ort.InferenceSession(
                    str(txt_p), sess_options=opts, providers=["CPUExecutionProvider"]
                )
            except Exception as exc:
                logger.warning("Failed to load MobileCLIP text ONNX model: %s", exc)
                self._txt_sess = None
            return self._txt_sess

    def encode_image_bytes(self, webp_or_img_bytes: bytes) -> np.ndarray | None:
        """Encode cached thumbnail bytes into an L2-normalized 512-d float32 embedding."""
        if not webp_or_img_bytes:
            return None
        sess = self._ensure_image_session()
        if sess is None:
            return None
        try:
            import cv2

            arr = cv2.imdecode(np.frombuffer(webp_or_img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if arr is None or arr.size == 0:
                return None
            inp = sess.get_inputs()[0]
            shape = getattr(inp, "shape", [1, 3, 256, 256])
            target_h = int(shape[2]) if isinstance(shape[2], int) and shape[2] > 0 else 256
            target_w = int(shape[3]) if isinstance(shape[3], int) and shape[3] > 0 else 256
            resized = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_AREA)
            rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
            chw = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)
            out = sess.run(None, {inp.name: chw})[0]
            return _l2_normalize_1d(out[0])
        except Exception as exc:
            logger.debug("Semantic image encoding failed: %s", exc)
            return None

    @staticmethod
    def _simple_clip_tokenize(prompt: str, context_len: int = 77) -> np.ndarray:
        """Deterministic lightweight token ID encoding when ONNX text session takes int64 tokens."""
        import zlib

        words = [w for w in (prompt or "").strip().lower().split() if w]
        tokens = np.zeros((1, context_len), dtype=np.int64)
        tokens[0, 0] = 49406  # SOT
        idx = 1
        for w in words[: max(0, context_len - 2)]:
            tokens[0, idx] = 1 + (zlib.crc32(w.encode("utf-8")) % 49000)
            idx += 1
        tokens[0, min(idx, context_len - 1)] = 49407  # EOT
        return tokens

    def encode_text(self, prompt: str) -> np.ndarray | None:
        """Encode a natural-language query into a 512-d embedding when text ONNX is available."""
        clean = (prompt or "").strip()
        if not clean:
            return None
        sess = self._ensure_text_session()
        if sess is None:
            return None
        try:
            inp = sess.get_inputs()[0]
            ctx = int(inp.shape[1]) if len(inp.shape) >= 2 and isinstance(inp.shape[1], int) else 77
            toks = self._simple_clip_tokenize(clean, context_len=ctx)
            out = sess.run(None, {inp.name: toks})[0]
            return _l2_normalize_1d(out[0])
        except Exception as exc:
            logger.debug("Semantic text encoding failed: %s", exc)
            return None

    @staticmethod
    def query_by_vector(
        repo: SemanticVectorRepository,
        query_vec: np.ndarray,
        *,
        threshold: float = 0.22,
        limit: int = 500,
        scope_ids: set[int] | None = None,
        model_ver: str = MODEL_VERSION_ID,
    ) -> list[tuple[int, float]]:
        """Score all indexed embeddings against `query_vec` in < 5 ms via matrix multiply."""
        media_ids, mat = repo.load_matrix(model_ver=model_ver)
        if mat.shape[0] == 0:
            return []
        q = _l2_normalize_1d(query_vec)
        if q.size != mat.shape[1]:
            return []
        sims = mat @ q
        idx_mask = np.where(sims >= float(threshold))[0]
        if idx_mask.size == 0:
            return []
        cand_scores = sims[idx_mask]
        order = np.argsort(-cand_scores)
        results: list[tuple[int, float]] = []
        for pos in order:
            orig_i = int(idx_mask[int(pos)])
            mid = int(media_ids[orig_i])
            if scope_ids is not None and mid not in scope_ids:
                continue
            results.append((mid, float(sims[orig_i])))
            if len(results) >= int(limit):
                break
        return results

    @staticmethod
    def compute_subject_counts(
        repo: SemanticVectorRepository,
        subject_vectors: dict[str, np.ndarray],
        *,
        thresholds: dict[str, float] | None = None,
        default_threshold: float = 0.24,
        model_ver: str = MODEL_VERSION_ID,
    ) -> dict[str, int]:
        """Compute calibrated per-subject counts in one batched matrix multiply."""
        media_ids, mat = repo.load_matrix(model_ver=model_ver)
        counts: dict[str, int] = {k: 0 for k in subject_vectors}
        if mat.shape[0] == 0 or not subject_vectors:
            return counts
        keys = [k for k, v in subject_vectors.items() if v.size == mat.shape[1]]
        if not keys:
            return counts
        subj_mat = np.vstack([_l2_normalize_1d(subject_vectors[k]) for k in keys])
        # sims: (N_photos, N_subjects)
        sims = mat @ subj_mat.T
        thr_map = thresholds or {k: t for k, _lbl, _pr, t in SUBJECT_PRESETS}
        for j, k in enumerate(keys):
            thr = float(thr_map.get(k, default_threshold))
            counts[k] = int(np.count_nonzero(sims[:, j] >= thr))
        return counts
