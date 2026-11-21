"""
Cross-platform offline OCR Engine with tiered zero-crash backends:
  1. rapidocr_onnxruntime (Cross-platform Windows & Linux ONNX)
  2. Windows 10/11 Native WinRT OCR (winocr or Windows.Media.Ocr via PowerShell)
  3. pytesseract (Linux / general fallback when tesseract binary is on PATH)
  4. Deterministic OpenCV MSER text-region stub fallback (never crashes in headless/test envs)
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

RAPIDOCR_SIZE_MB = 15
RAPIDOCR_FILES: list[tuple[str, str, int]] = [
    (
        "ch_PP-OCRv4_det_infer.onnx",
        "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_det_infer.onnx",
        4_745_517,
    ),
    (
        "ch_PP-OCRv4_rec_infer.onnx",
        "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx",
        10_857_958,
    ),
    (
        "ppocr_keys_v1.txt",
        "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/ppocr_keys_v1.txt",
        26_250,
    ),
]


def _get_rapidocr_appdata_dir() -> Path:
    try:
        from src.utils.paths import get_app_data_dir
        return get_app_data_dir() / "models" / "ocr"
    except Exception:
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            return Path(local_app) / "PhotoGallery" / "models" / "ocr"
        return Path("models/ocr").resolve()


def _find_rapidocr_files(directory: Path | None) -> tuple[Path | None, Path | None, Path | None]:
    if directory is None or not directory.is_dir():
        return None, None, None
    det = next(directory.glob("*det*.onnx"), None)
    rec = next(directory.glob("*rec*.onnx"), None)
    keys = next(directory.glob("*keys*.txt"), None) or next(directory.glob("*dict*.txt"), None)
    return det, rec, keys


def get_rapidocr_model_status() -> dict[str, Any]:
    """Return installation status of the optional RapidOCR (~15 MB) ONNX add-on."""
    custom_dir, _ = OcrEngine.get_saved_ocr_paths()
    search_dirs = [Path(custom_dir)] if custom_dir else []
    search_dirs.append(_get_rapidocr_appdata_dir())

    for d in search_dirs:
        det, rec, keys = _find_rapidocr_files(d)
        if det and rec and det.is_file() and rec.is_file():
            total_bytes = det.stat().st_size + rec.stat().st_size
            if keys and keys.is_file():
                total_bytes += keys.stat().st_size
            return {
                "installed": True,
                "dir": d,
                "size_mb": round(total_bytes / (1024 * 1024), 1),
                "det_path": det,
                "rec_path": rec,
                "keys_path": keys,
            }
    return {
        "installed": False,
        "dir": _get_rapidocr_appdata_dir(),
        "size_mb": RAPIDOCR_SIZE_MB,
        "det_path": None,
        "rec_path": None,
        "keys_path": None,
    }


def download_rapidocr_models(
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    timeout: float = 25.0,
) -> Path | None:
    """Download RapidOCR PP-OCRv4 ONNX models (~15 MB) atomically to AppData."""
    import urllib.request

    dest_dir = _get_rapidocr_appdata_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    total_expected = sum(sz for _, _, sz in RAPIDOCR_FILES)
    cumulative_done = 0

    try:
        for fname, url, approx_sz in RAPIDOCR_FILES:
            dest_file = dest_dir / fname
            tmp_file = dest_dir / f"{fname}.part"
            from src.version import APP_VERSION
            req = urllib.request.Request(url, headers={"User-Agent": f"EssentialsPhotoGallery/{APP_VERSION}"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                file_done = 0
                with open(tmp_file, "wb") as fh:
                    while True:
                        if cancel_cb is not None and cancel_cb():
                            raise InterruptedError("Download cancelled")
                        chunk = resp.read(128 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
                        file_done += len(chunk)
                        if progress_cb is not None:
                            progress_cb(cumulative_done + file_done, total_expected)
            if tmp_file.stat().st_size < 1024:
                raise ValueError(f"Downloaded file {fname} is unexpectedly small")
            os.replace(tmp_file, dest_file)
            cumulative_done += dest_file.stat().st_size
        return dest_dir
    except Exception as exc:
        logger.warning("RapidOCR model download failed: %s", exc)
        for fname, _, _ in RAPIDOCR_FILES:
            part = dest_dir / f"{fname}.part"
            if part.exists():
                try:
                    part.unlink()
                except OSError:
                    pass
        return None


def uninstall_rapidocr_models() -> bool:
    """Remove downloaded RapidOCR ONNX models from AppData."""
    dest_dir = _get_rapidocr_appdata_dir()
    removed = False
    for fname, _, _ in RAPIDOCR_FILES:
        p = dest_dir / fname
        if p.exists():
            try:
                p.unlink()
                removed = True
            except OSError:
                pass
    return removed


class _OnnxRapidOcrRunner:
    """Lightweight pure-ONNXRuntime runner for PP-OCRv4 det + rec models."""

    def __init__(self, det_path: Path, rec_path: Path, keys_path: Path | None) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.det_sess = ort.InferenceSession(str(det_path), opts, providers=["CPUExecutionProvider"])
        self.rec_sess = ort.InferenceSession(str(rec_path), opts, providers=["CPUExecutionProvider"])
        self.det_in = self.det_sess.get_inputs()[0].name
        self.rec_in = self.rec_sess.get_inputs()[0].name

        self.chars: list[str] = ["blank"]
        if keys_path and keys_path.is_file():
            with open(keys_path, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    self.chars.append(line.rstrip("\r\n"))
        self.chars.append(" ")

    def __call__(self, bgr: np.ndarray) -> tuple[list[tuple[Any, str, float]], None]:
        h, w = bgr.shape[:2]
        scale = min(1.0, 960.0 / max(h, w))
        nh = max(32, int(round(h * scale / 32.0)) * 32)
        nw = max(32, int(round(w * scale / 32.0)) * 32)
        resized = cv2.resize(bgr, (nw, nh))
        inp = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        inp = ((inp - mean) / std).transpose(2, 0, 1)[None, ...]

        prob = self.det_sess.run(None, {self.det_in: inp})[0][0, 0]
        mask = (prob > 0.3).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        sx, sy = w / float(nw), h / float(nh)
        boxes: list[tuple[int, int, int, int]] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 16:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            pad_x = max(2, int(bh * 0.25))
            pad_y = max(2, int(bh * 0.25))
            x1 = max(0, int((bx - pad_x) * sx))
            y1 = max(0, int((by - pad_y) * sy))
            x2 = min(w, int((bx + bw + pad_x) * sx))
            y2 = min(h, int((by + bh + pad_y) * sy))
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                boxes.append((x1, y1, x2, y2))

        boxes.sort(key=lambda b: (b[1] // 20, b[0]))
        results: list[tuple[Any, str, float]] = []
        for x1, y1, x2, y2 in boxes[:64]:
            crop = bgr[y1:y2, x1:x2]
            ch, cw = crop.shape[:2]
            rw = max(16, min(640, int(round(cw * (48.0 / max(1, ch))))))
            rc = cv2.resize(crop, (rw, 48))
            rin = ((rc.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None, ...]
            preds = self.rec_sess.run(None, {self.rec_in: rin})[0][0]
            idxs = np.argmax(preds, axis=1)
            probs = np.max(preds, axis=1)
            chars_out: list[str] = []
            confs: list[float] = []
            prev = 0
            for idx_val, conf in zip(idxs, probs):
                iv = int(idx_val)
                if iv != 0 and iv != prev and iv < len(self.chars):
                    chars_out.append(self.chars[iv])
                    confs.append(float(conf))
                prev = iv
            txt = "".join(chars_out).strip()
            if txt:
                results.append(([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], txt, sum(confs) / max(1, len(confs))))
        return results, None


def normalize_ocr_text(text: str) -> str:
    """Collapse excessive whitespace and strip control characters from OCR output."""
    if not text:
        return ""
    cleaned = re.sub(r"[\r\n\t]+", " ", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def resolve_tesseract_executable(hint: str | Path | None = None) -> str | None:
    """
    Resolve a path or directory hint (e.g. tesseract.exe, Tesseract-OCR/, or tessdata/)
    to an existing tesseract executable path.
    """
    if not hint:
        return None
    p = Path(str(hint).strip().strip('"'))
    if p.is_file():
        return str(p)
    if p.is_dir():
        for cand in (
            p / "tesseract.exe",
            p / "tesseract",
            p.parent / "tesseract.exe",
            p.parent / "tesseract",
        ):
            if cand.is_file():
                return str(cand)
    return None


def detect_tesseract_executable() -> str | None:
    """Auto-detect an installed Tesseract executable on PATH or standard OS install folders."""
    which_tess = shutil.which("tesseract")
    if which_tess and Path(which_tess).is_file():
        return str(which_tess)

    if sys.platform == "win32":
        candidates: list[Path] = [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ]
        for env_key, sub in (
            ("ProgramW6432", Path("Tesseract-OCR") / "tesseract.exe"),
            ("ProgramFiles", Path("Tesseract-OCR") / "tesseract.exe"),
            ("ProgramFiles(x86)", Path("Tesseract-OCR") / "tesseract.exe"),
            ("LOCALAPPDATA", Path("Programs") / "Tesseract-OCR" / "tesseract.exe"),
            ("USERPROFILE", Path("scoop") / "apps" / "tesseract" / "current" / "tesseract.exe"),
        ):
            base = os.environ.get(env_key)
            if base:
                candidates.append(Path(base) / sub)
        for cand in candidates:
            if cand.is_file():
                return str(cand)
    return None


class OcrEngine:
    """
    Offline text-in-photo extractor with automatic cross-platform backend selection.
    Thread-safe for single-worker sequential use; lazy-initializes backend on first call.
    """

    def __init__(
        self,
        backend: str = "auto",
        *,
        custom_model_dir: str | Path | None = None,
        tesseract_cmd: str | Path | None = None,
    ) -> None:
        self._requested_backend = backend
        self._backend_name: str | None = None
        self._rapid_ocr: Any = None
        self._winocr_mod: Any = None
        self._pytesseract_mod: Any = None
        self._tesseract_exe: str | None = None
        self._custom_model_dir = str(custom_model_dir) if custom_model_dir else None
        self._tesseract_cmd = str(tesseract_cmd) if tesseract_cmd else None

    @staticmethod
    def get_preferred_backend() -> str:
        """Return user-preferred OCR backend ('auto', 'pytesseract', 'rapidocr', 'windows_media_ocr')."""
        try:
            from PySide6.QtCore import QSettings

            s = QSettings("PhotoGallery64", "PhotoGallery64")
            val = str(s.value("ocr/preferred_backend", "auto", type=str) or "auto").strip().lower()
            return val or "auto"
        except Exception:
            return "auto"

    @staticmethod
    def set_preferred_backend(backend: str) -> None:
        """Persist user-preferred OCR backend in QSettings."""
        try:
            from PySide6.QtCore import QSettings

            s = QSettings("PhotoGallery64", "PhotoGallery64")
            s.setValue("ocr/preferred_backend", str(backend or "auto").strip().lower())
        except Exception:
            pass

    @staticmethod
    def get_saved_ocr_paths() -> tuple[str | None, str | None]:
        """Return (custom_model_dir, tesseract_path) saved in QSettings, if any."""
        try:
            from PySide6.QtCore import QSettings

            s = QSettings("PhotoGallery64", "PhotoGallery64")
            m_dir = s.value("ocr/custom_model_dir", "", type=str) or None
            t_cmd = s.value("ocr/tesseract_path", "", type=str) or None
            return m_dir, t_cmd
        except Exception:
            return None, None

    @staticmethod
    def save_ocr_paths(
        *,
        custom_model_dir: str | None = None,
        tesseract_path: str | None = None,
    ) -> None:
        """Persist custom RapidOCR model folder or Tesseract binary path in QSettings."""
        try:
            from PySide6.QtCore import QSettings

            s = QSettings("PhotoGallery64", "PhotoGallery64")
            if custom_model_dir is not None:
                s.setValue("ocr/custom_model_dir", custom_model_dir)
            if tesseract_path is not None:
                s.setValue("ocr/tesseract_path", tesseract_path)
        except Exception:
            pass

    @property
    def backend_name(self) -> str:
        if self._backend_name is None:
            self._select_backend()
        return self._backend_name or "stub"

    def _try_init_rapidocr(self) -> bool:
        saved_dir, _ = self.get_saved_ocr_paths()
        model_dir = self._custom_model_dir or saved_dir
        try:
            from rapidocr_onnxruntime import RapidOCR

            kwargs: dict[str, Any] = {}
            if model_dir and Path(model_dir).is_dir():
                det, rec, _ = _find_rapidocr_files(Path(model_dir))
                cls = next(Path(model_dir).glob("*cls*.onnx"), None)
                if det:
                    kwargs["det_model_path"] = str(det)
                if rec:
                    kwargs["rec_model_path"] = str(rec)
                if cls:
                    kwargs["cls_model_path"] = str(cls)

            self._rapid_ocr = RapidOCR(**kwargs)
            self._backend_name = "rapidocr"
            logger.info("OcrEngine initialized with rapidocr_onnxruntime")
            return True
        except Exception as exc:
            logger.debug("rapidocr_onnxruntime package unavailable: %s", exc)

        # Fallback: pure onnxruntime runner using downloaded AppData or custom RapidOCR ONNX models
        try:
            st = get_rapidocr_model_status()
            if model_dir and Path(model_dir).is_dir():
                det, rec, keys = _find_rapidocr_files(Path(model_dir))
            else:
                det, rec, keys = st.get("det_path"), st.get("rec_path"), st.get("keys_path")
            if det and rec and Path(det).is_file() and Path(rec).is_file():
                self._rapid_ocr = _OnnxRapidOcrRunner(Path(det), Path(rec), Path(keys) if keys else None)
                self._backend_name = "rapidocr"
                logger.info("OcrEngine initialized with AppData RapidOCR ONNX models (%s)", Path(det).parent)
                return True
        except Exception as exc:
            logger.debug("RapidOCR ONNX runner unavailable: %s", exc)
        return False

    def _try_init_winocr(self) -> bool:
        if sys.platform != "win32":
            return False
        try:
            import winocr  # type: ignore[import-not-found]

            self._winocr_mod = winocr
            self._backend_name = "winocr"
            logger.info("OcrEngine initialized with winocr")
            return True
        except Exception:
            pass

        # Verify PowerShell + Windows.Media.Ocr availability on Windows 10/11
        if shutil.which("powershell.exe") or shutil.which("pwsh"):
            self._backend_name = "windows_media_ocr"
            logger.info("OcrEngine initialized with Windows.Media.Ocr PowerShell bridge")
            return True
        return False

    def _try_init_pytesseract(self) -> bool:
        _, saved_tess = self.get_saved_ocr_paths()
        tess_cmd = (
            resolve_tesseract_executable(self._tesseract_cmd)
            or resolve_tesseract_executable(saved_tess)
            or detect_tesseract_executable()
        )
        if not tess_cmd or not Path(tess_cmd).is_file():
            return False

        self._tesseract_exe = str(tess_cmd)
        try:
            import pytesseract  # type: ignore[import-not-found]

            pytesseract.pytesseract.tesseract_cmd = str(tess_cmd)
            self._pytesseract_mod = pytesseract
        except Exception:
            # pytesseract wrapper is optional; we can invoke tesseract.exe directly via subprocess
            self._pytesseract_mod = None

        self._backend_name = "pytesseract"
        logger.info("OcrEngine initialized with Tesseract (%s)", tess_cmd)
        return True

    def _select_backend(self) -> str:
        if self._backend_name is not None:
            return self._backend_name

        req = (self._requested_backend or "auto").strip().lower()
        if req == "auto":
            pref = self.get_preferred_backend()
            if pref != "auto":
                req = pref
        if req == "rapidocr":
            if self._try_init_rapidocr():
                return "rapidocr"
        elif req in ("winocr", "windows_media_ocr"):
            if self._try_init_winocr():
                return self._backend_name or "stub"
        elif req in ("pytesseract", "tesseract"):
            if self._try_init_pytesseract():
                return "pytesseract"
        elif req == "stub":
            self._backend_name = "stub"
            return "stub"
        else:
            if self._try_init_winocr():
                return self._backend_name or "stub"
            if self._try_init_rapidocr():
                return "rapidocr"
            if self._try_init_pytesseract():
                return "pytesseract"

        self._backend_name = "stub"
        logger.debug("OcrEngine using deterministic MSER stub fallback")
        return "stub"

    def detect_text_regions(self, rgb_array: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Deterministic OpenCV MSER text-region detector used as a lightweight
        fallback/diagnostic when no full OCR backend is installed.
        Returns bounding boxes (x, y, w, h).
        """
        if rgb_array is None or rgb_array.size == 0:
            return []
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY) if rgb_array.ndim == 3 else rgb_array
        mser = cv2.MSER_create(delta=5, min_area=40, max_area=4000)
        regions, _ = mser.detectRegions(gray)
        boxes: list[tuple[int, int, int, int]] = []
        for pts in regions[:64]:
            x, y, w, h = cv2.boundingRect(pts)
            if 4 <= w <= gray.shape[1] * 0.8 and 6 <= h <= gray.shape[0] * 0.4:
                boxes.append((int(x), int(y), int(w), int(h)))
        return boxes

    def _extract_rapidocr(self, rgb_array: np.ndarray) -> str:
        bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        result, _ = self._rapid_ocr(bgr)
        if not result:
            return ""
        lines = [str(item[1]) for item in result if len(item) >= 2 and item[1]]
        return normalize_ocr_text(" ".join(lines))

    def _extract_winocr(self, rgb_array: np.ndarray) -> str:
        from PIL import Image

        pil_img = Image.fromarray(rgb_array)
        res = self._winocr_mod.recognize_pil_sync(pil_img, "en")
        text = getattr(res, "text", None) or (res.get("text", "") if isinstance(res, dict) else str(res or ""))
        return normalize_ocr_text(text)

    def _extract_windows_media_ocr(self, rgb_array: np.ndarray) -> str:
        if sys.platform != "win32":
            return ""
        ps_exe = shutil.which("powershell.exe") or shutil.which("pwsh")
        if not ps_exe:
            return ""

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tmp_path = tf.name
            bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(tmp_path, bgr)

            escaped = tmp_path.replace("'", "''")
            ps_script = (
                "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
                "Add-Type -AssemblyName System.Runtime.WindowsRuntime; "
                "$null = [Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime]; "
                "$null = [Windows.Graphics.Imaging.BitmapDecoder,Windows.Graphics,ContentType=WindowsRuntime]; "
                "$null = [Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]; "
                "function Await($op, $t) { "
                "  $m = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.IsGenericMethod } | Select-Object -First 1; "
                "  $gm = $m.MakeGenericMethod($t); $task = $gm.Invoke($null, @($op)); $task.Wait(); return $task.Result "
                "}; "
                f"$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync('{escaped}')) ([Windows.Storage.StorageFile]); "
                "$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream]); "
                "$dec = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder]); "
                "$bmp = Await ($dec.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap]); "
                "$eng = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages(); "
                "if ($null -ne $eng) { $res = Await ($eng.RecognizeAsync($bmp)) ([Windows.Media.Ocr.OcrResult]); Write-Output $res.Text }"
            )
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.run(
                [ps_exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=12,
                creationflags=creation_flags,
            )
            if proc.returncode == 0 and proc.stdout:
                return normalize_ocr_text(proc.stdout)
            return ""
        except Exception as exc:
            logger.debug("Windows.Media.Ocr failed: %s", exc)
            return ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _extract_pytesseract(self, rgb_array: np.ndarray) -> str:
        if self._pytesseract_mod is not None:
            from PIL import Image

            pil_img = Image.fromarray(rgb_array)
            text = self._pytesseract_mod.image_to_string(pil_img)
            return normalize_ocr_text(text)

        if not self._tesseract_exe:
            return ""
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tmp_path = tf.name
            bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(tmp_path, bgr)
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.run(
                [self._tesseract_exe, tmp_path, "stdout", "--psm", "3"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                creationflags=creation_flags,
            )
            if proc.returncode == 0 and proc.stdout:
                return normalize_ocr_text(proc.stdout)
            return ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def extract_text(self, rgb_array: np.ndarray) -> str:
        """
        Extract normalized text from an RGB uint8 numpy array.
        Downscales images larger than 1024px long edge for fast, consistent OCR.
        """
        if rgb_array is None or rgb_array.size == 0:
            return ""

        if rgb_array.ndim == 2:
            rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_GRAY2RGB)
        elif rgb_array.shape[2] == 4:
            rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2RGB)

        h, w = rgb_array.shape[:2]
        max_dim = max(h, w)
        if max_dim > 1024:
            scale = 1024.0 / float(max_dim)
            rgb_array = cv2.resize(
                rgb_array,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )

        backend = self._select_backend()
        try:
            if backend == "rapidocr":
                return self._extract_rapidocr(rgb_array)
            if backend == "winocr":
                return self._extract_winocr(rgb_array)
            if backend == "windows_media_ocr":
                return self._extract_windows_media_ocr(rgb_array)
            if backend == "pytesseract":
                return self._extract_pytesseract(rgb_array)
        except Exception as exc:
            logger.debug("OCR extraction error on backend %s: %s", backend, exc)
            return ""

        # Deterministic stub fallback: exercise MSER detector without inventing fake text
        _ = self.detect_text_regions(rgb_array)
        return ""


def has_configured_ocr_engine() -> bool:
    """Return True if RapidOCR (package or downloaded ONNX models), Tesseract, or winocr is available."""
    if get_rapidocr_model_status().get("installed"):
        return True
    _, saved_tess = OcrEngine.get_saved_ocr_paths()
    if resolve_tesseract_executable(saved_tess) or detect_tesseract_executable():
        return True
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except Exception:
        pass
    try:
        import winocr  # noqa: F401
        return True
    except Exception:
        pass
    return False


def get_ocr_engine_status() -> dict[str, Any]:
    """Return a summary dict of active OCR backend and detected/configured paths."""
    custom_dir, saved_tess = OcrEngine.get_saved_ocr_paths()
    detected_tess = resolve_tesseract_executable(saved_tess) or detect_tesseract_executable()
    eng = OcrEngine()
    backend = eng.backend_name
    rapid_st = get_rapidocr_model_status()
    rapid_pkg = bool(rapid_st.get("installed"))
    if not rapid_pkg:
        try:
            import rapidocr_onnxruntime  # noqa: F401
            rapid_pkg = True
        except Exception:
            pass
    return {
        "backend": backend,
        "rapidocr_available": rapid_pkg,
        "rapidocr_installed": bool(rapid_st.get("installed")),
        "rapidocr_size_mb": rapid_st.get("size_mb", RAPIDOCR_SIZE_MB),
        "custom_model_dir": custom_dir or "",
        "tesseract_configured": saved_tess or "",
        "tesseract_detected": detected_tess or "",
    }
