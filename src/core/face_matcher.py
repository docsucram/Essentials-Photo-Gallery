"""
High-Performance In-Memory Vector Face Matcher for PhotoGallery64.
Provides zero-latency face similarity search, centroid tracking, and rejection filtering:
- Vectorized NumPy matrix multiplication S = U @ C.T for high-throughput batch matching
- Fast rejection masking using face_rejected_suggestions pairs
- L2-normalized running centroid maintenance in <1ms
- Serialization helpers for SQLite BLOB storage
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np


class FaceMatchSuggestion(NamedTuple):
    """Container for a candidate face suggestion."""

    face_id: int
    person_id: int
    score: float
    is_high_confidence: bool = False


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """L2-normalizes a 1D vector (512,) or 2D array of vectors (N, 512)."""
    if v.ndim == 1:
        norm = float(np.linalg.norm(v))
        if norm > 1e-9:
            return (v / norm).astype(np.float32)
        dim = max(1, v.shape[0])
        return (np.ones((dim,), dtype=np.float32) / np.sqrt(float(dim))).astype(np.float32)

    norms = np.linalg.norm(v, axis=1, keepdims=True)
    zero_mask = (norms < 1e-9).squeeze()
    norms[norms < 1e-9] = 1.0
    res = (v / norms).astype(np.float32)
    if np.any(zero_mask):
        res[zero_mask] = 1.0 / np.sqrt(float(v.shape[1]))
    return res


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Converts a (512,) or (K, 512) float32 embedding into raw bytes for SQLite BLOB storage."""
    arr = np.ascontiguousarray(embedding, dtype=np.float32)
    return arr.tobytes()


def deserialize_embedding(blob: bytes, allow_multi: bool = False) -> np.ndarray:
    """
    Converts raw bytes from SQLite BLOB into an L2-normalized float32 NumPy array.
    If blob contains multiple packed centroids (K*2048 bytes), returns:
    - (K, 512) array if allow_multi=True
    - (512,) unit vector (primary centroid) if allow_multi=False
    """
    if not blob or len(blob) < 4:
        return np.zeros((512,), dtype=np.float32)
    arr = np.frombuffer(blob, dtype=np.float32).copy()
    if arr.size % 512 == 0 and arr.size > 512:
        mat = normalize_vector(arr.reshape(-1, 512))
        if allow_multi:
            return mat
        return mat[0]
    return normalize_vector(arr)


def deserialize_centroids(blob: bytes) -> np.ndarray:
    """Converts raw K*2048 bytes from SQLite BLOB into a (K, 512) float32 NumPy array (K >= 1)."""
    if not blob or len(blob) < 4:
        return np.zeros((1, 512), dtype=np.float32)
    arr = np.frombuffer(blob, dtype=np.float32).copy()
    if arr.size % 512 != 0 or arr.size == 0:
        return np.zeros((1, 512), dtype=np.float32)
    mat = arr.reshape(-1, 512)
    return normalize_vector(mat)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two 512-D vectors in [-1.0, 1.0]."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class FaceVectorMatcher:
    """
    Tier 2 In-Memory Vector Matcher.
    Computes cosine similarities between unassigned faces and confirmed person centroids
    at speeds exceeding 1,000,000 comparisons / second using BLAS-accelerated NumPy.
    Supports multi-era sub-centroids (e.g. child to adult aging) and unknown face clustering.
    """

    @staticmethod
    def compute_sub_centroids(
        embeddings: np.ndarray | Sequence[np.ndarray],
        k: int | None = None,
        max_k: int = 5,
        min_samples_for_multi: int = 10,
    ) -> np.ndarray:
        """
        Computes up to K sub-centroids for multi-era matching (e.g. child to adult).
        Falls back to 1x512 single mean centroid when confirmed count < min_samples_for_multi (default 10).

        Args:
            embeddings: (N, 512) array or sequence of face embeddings.
            k: Exact number of sub-centroids to produce (if None, automatically chooses min(max_k, max(1, N // 3))).
            max_k: Maximum number of sub-centroids (default: 5).
            min_samples_for_multi: Minimum number of samples required to produce >1 sub-centroids (default: 10).

        Returns:
            (K, 512) float32 L2-normalized array of sub-centroids.
        """
        arr = np.ascontiguousarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.size == 0:
            return np.zeros((1, 512), dtype=np.float32)

        arr = normalize_vector(arr)
        n_samples = arr.shape[0]

        if k is not None:
            target_k = max(1, min(int(k), n_samples))
        elif n_samples < min_samples_for_multi:
            target_k = 1
        else:
            target_k = min(int(max_k), max(2, n_samples // 3))

        if target_k <= 1 or n_samples == 1:
            mean_vec = np.mean(arr, axis=0, keepdims=True)
            return normalize_vector(mean_vec).astype(np.float32)

        # Multi-centroid clustering using cv2.kmeans (K-Means++)
        try:
            import cv2
            # Uniformly stride very large sets so k-means++ stays fast (<30ms) while preserving diversity
            arr_kmeans = arr[::max(1, n_samples // 1000)] if n_samples > 1000 else arr
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            flags = cv2.KMEANS_PP_CENTERS
            _compactness, labels, centers = cv2.kmeans(
                arr_kmeans, target_k, None, criteria, 10, flags
            )
            counts = np.bincount(labels.flatten(), minlength=target_k)
            order = np.argsort(-counts)
            sorted_centers = centers[order]
            return normalize_vector(sorted_centers).astype(np.float32)
        except Exception:
            mean_vec = np.mean(arr, axis=0, keepdims=True)
            return normalize_vector(mean_vec).astype(np.float32)

    @staticmethod
    def cluster_unnamed_faces(
        face_ids: Sequence[int],
        embeddings: np.ndarray | Sequence[np.ndarray],
        sim_threshold: float = 0.58,
        min_cluster_size: int = 3,
    ) -> dict[str, list[int]]:
        """
        Clusters unnamed face embeddings by cosine similarity.
        Groups dense clusters into 'Unknown Person #1', 'Unknown Person #2', etc.
        Leaves singleton noise or clusters < min_cluster_size in 'Other Unnamed Faces'.

        Args:
            face_ids: Sequence of face IDs.
            embeddings: (N, 512) array of face embeddings.
            sim_threshold: Minimum cosine similarity to link faces (default: 0.58).
            min_cluster_size: Minimum number of faces to constitute an identified cluster (default: 3).

        Returns:
            dict mapping group label (e.g. 'Unknown Person #1', 'Other Unnamed Faces')
            to list of face_ids.
        """
        n_faces = len(face_ids)
        if n_faces == 0:
            return {}

        face_ids_list = list(face_ids)
        if n_faces < min_cluster_size:
            return {"Other Unnamed Faces": face_ids_list}

        E = np.ascontiguousarray(embeddings, dtype=np.float32)
        if E.ndim == 1:
            E = E.reshape(1, -1)
        E = normalize_vector(E)

        # Compute NxN cosine similarity matrix
        sim_matrix = E @ E.T
        adj = (sim_matrix >= sim_threshold)
        np.fill_diagonal(adj, True)

        # Vectorized greedy leader clustering (O(N) updates, avoiding N 2D matrix copies)
        active = np.ones(n_faces, dtype=bool)
        deg = adj.sum(axis=1)
        clusters: list[list[int]] = []

        while True:
            deg_masked = np.where(active, deg, 0)
            best_idx = int(np.argmax(deg_masked))
            max_deg = deg_masked[best_idx]
            if max_deg < min_cluster_size:
                break

            cluster_mask = adj[best_idx] & active
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) < min_cluster_size:
                active[best_idx] = False
                continue

            # Sort cluster members by similarity to cluster mean
            cluster_emb = E[cluster_indices]
            cluster_mean = normalize_vector(np.mean(cluster_emb, axis=0, keepdims=True))
            sims_to_mean = (cluster_emb @ cluster_mean.T).flatten()
            sorted_members = cluster_indices[np.argsort(-sims_to_mean)]

            clusters.append([face_ids_list[idx] for idx in sorted_members])
            active[cluster_indices] = False
            deg -= adj[:, cluster_indices].sum(axis=1)

        result: dict[str, list[int]] = {}
        for idx, cluster_fids in enumerate(clusters, 1):
            result[f"Unknown Person #{idx}"] = cluster_fids

        other_indices = np.where(active)[0]
        if len(other_indices) > 0:
            result["Other Unnamed Faces"] = [face_ids_list[idx] for idx in other_indices]

        return result

    @staticmethod
    def match_unassigned_faces(
        face_ids: Sequence[int] | np.ndarray,
        face_matrix: np.ndarray,
        person_ids: Sequence[int] | np.ndarray,
        centroid_matrix: np.ndarray | Sequence[np.ndarray] | dict[int, np.ndarray],
        centroid_person_ids: Sequence[int] | None = None,
        rejected_pairs: set[tuple[int, int]] | None = None,
        rejections: dict[int, set[int]] | None = None,
        high_thresh: float = 0.55,
        low_thresh: float = 0.42,
        top_k: int = 1,
        assume_normalized: bool = True,
    ) -> list[FaceMatchSuggestion]:
        """
        Matches unassigned face embeddings against person centroids (including multi-era sub-centroids).

        Args:
            face_ids: List or array of N unassigned media_people IDs.
            face_matrix: (N, 512) float32 array of face embeddings.
            person_ids: List or array of M person IDs.
            centroid_matrix: (M, 512) float32 array, list of arrays, or dict of person centroid embeddings.
            centroid_person_ids: Optional sequence of person IDs corresponding to each row of centroid_matrix.
            rejected_pairs: Set of (face_id, person_id) tuples to permanently exclude.
            rejections: Optional mapping {face_id: set_of_rejected_person_ids} (merged into rejected_pairs).
            high_thresh: Similarity threshold for high-confidence match badge.
            low_thresh: Minimum similarity threshold for candidate suggestions.
            top_k: Number of candidate person matches to return per face (default: 1).
            assume_normalized: Set True when inputs are pre-normalized float32 vectors.

        Returns:
            List of FaceMatchSuggestion instances for matches with score >= low_thresh.
        """
        n_faces = len(face_ids)
        if n_faces == 0:
            return []

        # Parse centroid_matrix and map to unique person_ids
        unique_persons: list[int] = []
        row_person_ids: list[int] = []
        c_flat_list: list[np.ndarray] = []

        if isinstance(centroid_matrix, dict):
            for pid, val in centroid_matrix.items():
                unique_persons.append(int(pid))
                v_arr = np.ascontiguousarray(val, dtype=np.float32)
                if v_arr.ndim == 1:
                    v_arr = v_arr.reshape(1, -1)
                c_flat_list.append(v_arr)
                row_person_ids.extend([int(pid)] * v_arr.shape[0])
            if c_flat_list:
                C_flat = np.vstack(c_flat_list)
            else:
                return []
        elif isinstance(centroid_matrix, (list, tuple)) and centroid_matrix and isinstance(centroid_matrix[0], np.ndarray):
            unique_persons = [int(p) for p in person_ids]
            for pid, v_arr in zip(unique_persons, centroid_matrix):
                v_arr = np.ascontiguousarray(v_arr, dtype=np.float32)
                if v_arr.ndim == 1:
                    v_arr = v_arr.reshape(1, -1)
                c_flat_list.append(v_arr)
                row_person_ids.extend([pid] * v_arr.shape[0])
            C_flat = np.vstack(c_flat_list)
        else:
            c_arr = np.ascontiguousarray(centroid_matrix, dtype=np.float32)
            if c_arr.ndim == 3:
                m_dim, k_dim, d_dim = c_arr.shape
                C_flat = c_arr.reshape(m_dim * k_dim, d_dim)
                unique_persons = [int(p) for p in person_ids]
                row_person_ids = [pid for pid in unique_persons for _ in range(k_dim)]
            else:
                C_flat = c_arr
                if centroid_person_ids is not None:
                    row_person_ids = [int(p) for p in centroid_person_ids]
                    seen = set()
                    unique_persons = [p for p in row_person_ids if not (p in seen or seen.add(p))]
                elif len(person_ids) == 1 and C_flat.shape[0] > 1:
                    unique_persons = [int(person_ids[0])]
                    row_person_ids = [unique_persons[0]] * C_flat.shape[0]
                else:
                    unique_persons = [int(p) for p in person_ids]
                    row_person_ids = unique_persons

        n_persons = len(unique_persons)
        if n_persons == 0 or C_flat.shape[0] == 0:
            return []

        # 1. Ensure float32 contiguous arrays & normalize
        U = np.ascontiguousarray(face_matrix, dtype=np.float32)
        C = np.ascontiguousarray(C_flat, dtype=np.float32)

        if not assume_normalized:
            U = normalize_vector(U)
            C = normalize_vector(C)

        # 2. Vectorized Cosine Similarity Matrix: S_all = U @ C.T (shape: N, Total_K)
        S_all = U @ C.T

        # 3. Collapse sub-centroids per person: score = max(cosine similarity across person's sub-centroids)
        if len(row_person_ids) == n_persons and row_person_ids == unique_persons:
            S = S_all
        else:
            S = np.zeros((n_faces, n_persons), dtype=np.float32)
            pid_to_cols: dict[int, list[int]] = {}
            for col_idx, pid in enumerate(row_person_ids):
                pid_to_cols.setdefault(pid, []).append(col_idx)

            for col_p, pid in enumerate(unique_persons):
                cols = pid_to_cols.get(pid, [])
                if not cols:
                    S[:, col_p] = -1.0
                elif len(cols) == 1:
                    S[:, col_p] = S_all[:, cols[0]]
                else:
                    S[:, col_p] = np.max(S_all[:, cols], axis=1)

        # 4. Mask out permanently rejected (face_id, person_id) pairs to -1.0
        effective_rejections = set(rejected_pairs) if rejected_pairs else set()
        if rejections:
            for fid, pids in rejections.items():
                for pid in pids:
                    effective_rejections.add((fid, pid))

        if effective_rejections:
            face_to_row = {fid: idx for idx, fid in enumerate(face_ids)}
            person_to_col = {pid: idx for idx, pid in enumerate(unique_persons)}

            mask_rows = []
            mask_cols = []
            for fid, pid in effective_rejections:
                r = face_to_row.get(fid)
                c = person_to_col.get(pid)
                if r is not None and c is not None:
                    mask_rows.append(r)
                    mask_cols.append(c)

            if mask_rows:
                S[mask_rows, mask_cols] = -1.0

        # 5. Extract Top-K Matches per Face
        results: list[FaceMatchSuggestion] = []
        face_ids_arr = np.asarray(face_ids)
        person_ids_arr = np.asarray(unique_persons)

        if top_k <= 1:
            best_col_indices = np.argmax(S, axis=1)
            best_scores = S[np.arange(n_faces), best_col_indices]

            valid_mask = best_scores >= low_thresh
            valid_indices = np.where(valid_mask)[0]

            for i in valid_indices:
                score = float(best_scores[i])
                results.append(
                    FaceMatchSuggestion(
                        face_id=int(face_ids_arr[i]),
                        person_id=int(person_ids_arr[best_col_indices[i]]),
                        score=score,
                        is_high_confidence=bool(score >= high_thresh),
                    )
                )
        else:
            k = min(top_k, n_persons)
            top_k_indices = np.argpartition(-S, k - 1, axis=1)[:, :k]

            for i in range(n_faces):
                row_indices = top_k_indices[i]
                sorted_cols = row_indices[np.argsort(-S[i, row_indices])]
                for c in sorted_cols:
                    score = float(S[i, c])
                    if score >= low_thresh:
                        results.append(
                            FaceMatchSuggestion(
                                face_id=int(face_ids_arr[i]),
                                person_id=int(person_ids_arr[c]),
                                score=score,
                                is_high_confidence=bool(score >= high_thresh),
                            )
                        )

        return results

    @staticmethod
    def update_centroid_incremental(
        old_centroid: np.ndarray | None,
        confirmed_count: int,
        new_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Computes L2-normalized running mean centroid:
        mu_new = normalize(N * mu_old + sum(e_i)) in <1ms.

        Args:
            old_centroid: (512,) float32 current centroid vector (or None if first confirmation).
            confirmed_count: Number of faces previously confirmed for this person.
            new_embeddings: (512,) or (K, 512) float32 newly confirmed face embedding(s).

        Returns:
            (512,) L2-normalized float32 centroid vector.
        """
        arr = np.ascontiguousarray(new_embeddings, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]

        if arr.size == 0:
            if old_centroid is not None:
                return normalize_vector(old_centroid)
            return np.zeros((512,), dtype=np.float32)

        # Sum new vectors along axis 0
        sum_new = np.sum(arr, axis=0)

        if old_centroid is None or confirmed_count <= 0:
            combined = sum_new
        else:
            old_vec = np.ascontiguousarray(old_centroid, dtype=np.float64)
            old_norm = np.linalg.norm(old_vec)
            if old_norm > 1e-9:
                old_vec = old_vec / old_norm
            combined = float(confirmed_count) * old_vec + sum_new

        norm = float(np.linalg.norm(combined))
        if norm > 1e-9:
            return (combined / norm).astype(np.float32)
        return combined.astype(np.float32)
