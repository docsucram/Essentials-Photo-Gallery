"""
Windows Portable Devices (WPD) MTP / PTP Camera & Smartphone Detection and Transfer.

Uses native Windows COM interfaces via ctypes (PortableDeviceApi.dll / ole32.dll)
with zero external Python dependencies:
- IPortableDeviceManager: Enumerates connected WPD devices & friendly names.
- IPortableDevice / IPortableDeviceContent / IPortableDeviceProperties:
  Recursively discovers photos & videos on MTP/PTP cameras and smartphones.
- IPortableDeviceResources / IStream:
  Streams thumbnails and full-resolution media files directly to disk.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

# Device types from PortableDevice.h (WPD_DEVICE_TYPES)
WPD_DEVICE_TYPE_GENERIC = 0
WPD_DEVICE_TYPE_CAMERA = 1
WPD_DEVICE_TYPE_MEDIA_PLAYER = 2
WPD_DEVICE_TYPE_PHONE = 3
WPD_DEVICE_TYPE_VIDEO = 4

# Folder names to skip on MTP/PTP devices (app caches, OS thumbnail caches)
_SKIP_FOLDER_NAMES = {
    ".thumbnails",
    ".thumbdata",
    ".trash",
    ".trashed",
    "android",
    "$recycle.bin",
    "system volume information",
}

if IS_WINDOWS:
    from ctypes import wintypes

    _ole32 = ctypes.windll.ole32
    _ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    _ole32.CoInitializeEx.restype = ctypes.c_long
    _ole32.CoUninitialize.argtypes = []
    _ole32.CoUninitialize.restype = None
    _ole32.CoCreateInstance.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _ole32.CoCreateInstance.restype = ctypes.c_long
    _ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    _ole32.CoTaskMemFree.restype = None
    _ole32.PropVariantClear.argtypes = [ctypes.c_void_p]
    _ole32.PropVariantClear.restype = ctypes.c_long

    COINIT_MULTITHREADED = 0x0
    CLSCTX_INPROC_SERVER = 0x1
    STGM_READ = 0x0
    PORTABLE_DEVICE_DELETE_NO_RECURSION = 0

    VT_EMPTY = 0
    VT_UI4 = 19
    VT_UI8 = 21
    VT_DATE = 7
    VT_LPWSTR = 31
    VT_FILETIME = 64
    VT_CLSID = 72

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", ctypes.c_ubyte * 8),
        ]

        def to_uuid(self) -> uuid.UUID:
            b = (
                self.Data1.to_bytes(4, "big")
                + self.Data2.to_bytes(2, "big")
                + self.Data3.to_bytes(2, "big")
                + bytes(self.Data4)
            )
            return uuid.UUID(bytes=b)

    class PROPERTYKEY(ctypes.Structure):
        _fields_ = [
            ("fmtid", GUID),
            ("pid", wintypes.DWORD),
        ]

    class _PROPVARIANT_UNION(ctypes.Union):
        _fields_ = [
            ("ulVal", wintypes.ULONG),
            ("uhVal", ctypes.c_uint64),
            ("dblVal", ctypes.c_double),
            ("pwszVal", ctypes.c_void_p),
            ("puuid", ctypes.POINTER(GUID)),
            ("filetime", wintypes.FILETIME),
            ("_pad", ctypes.c_ubyte * 16),
        ]

    class PROPVARIANT(ctypes.Structure):
        _fields_ = [
            ("vt", ctypes.c_ushort),
            ("wReserved1", ctypes.c_ushort),
            ("wReserved2", ctypes.c_ushort),
            ("wReserved3", ctypes.c_ushort),
            ("val", _PROPVARIANT_UNION),
        ]

    def _guid(s: str) -> GUID:
        u = uuid.UUID(s)
        return GUID(
            u.time_low,
            u.time_mid,
            u.time_hi_version,
            (ctypes.c_ubyte * 8)(*u.bytes[8:]),
        )

    def _pkey(guid_str: str, pid: int) -> PROPERTYKEY:
        return PROPERTYKEY(_guid(guid_str), pid)

    # COM CLSIDs & IIDs
    CLSID_PortableDeviceManager = _guid("0af10cec-2ecd-4b92-9581-34f6ae0637f3")
    IID_IPortableDeviceManager = _guid("a1567595-4c2f-4574-a6fa-ecef917b9a40")

    CLSID_PortableDeviceFTM = _guid("f7c0039a-4762-488a-b4b3-760ef9a1ba9b")
    IID_IPortableDevice = _guid("625e2df8-6392-4cf0-9ad1-3cfa5f17775c")

    CLSID_PortableDeviceValues = _guid("0c15d503-d017-47ce-9016-7b3f978721cc")
    IID_IPortableDeviceValues = _guid("6848f6f2-3155-4f86-b6f5-263eeeab3143")

    CLSID_PortableDeviceKeyCollection = _guid("de2d022d-2480-43be-97f0-d1fa2cf98f4f")
    IID_IPortableDeviceKeyCollection = _guid("dada2357-e0ad-492e-98db-dd61c53ba353")

    CLSID_PortableDevicePropVariantCollection = _guid("08a99e2f-6d6d-4b80-af5a-baf2bcbe4cb9")
    IID_IPortableDevicePropVariantCollection = _guid("89b2e422-4f1b-4316-bcef-a44afea83eb3")

    # WPD Property Keys
    WPD_CLIENT_NAME = _pkey("204D9F0C-2292-4080-9F42-40664E70F859", 2)
    WPD_CLIENT_MAJOR_VERSION = _pkey("204D9F0C-2292-4080-9F42-40664E70F859", 3)
    WPD_CLIENT_MINOR_VERSION = _pkey("204D9F0C-2292-4080-9F42-40664E70F859", 4)
    WPD_CLIENT_REVISION = _pkey("204D9F0C-2292-4080-9F42-40664E70F859", 5)
    WPD_CLIENT_SECURITY_QUALITY_OF_SERVICE = _pkey("204D9F0C-2292-4080-9F42-40664E70F859", 8)

    WPD_OBJECT_NAME = _pkey("EF6B490D-5CD8-437A-AFFC-DA8B60EE4A3C", 4)
    WPD_OBJECT_CONTENT_TYPE = _pkey("EF6B490D-5CD8-437A-AFFC-DA8B60EE4A3C", 7)
    WPD_OBJECT_SIZE = _pkey("EF6B490D-5CD8-437A-AFFC-DA8B60EE4A3C", 11)
    WPD_OBJECT_ORIGINAL_FILE_NAME = _pkey("EF6B490D-5CD8-437A-AFFC-DA8B60EE4A3C", 12)
    WPD_OBJECT_DATE_CREATED = _pkey("EF6B490D-5CD8-437A-AFFC-DA8B60EE4A3C", 18)
    WPD_OBJECT_DATE_MODIFIED = _pkey("EF6B490D-5CD8-437A-AFFC-DA8B60EE4A3C", 19)

    WPD_DEVICE_TYPE = _pkey("26D4979A-E643-4626-9E2B-736DC0C92FDC", 15)
    WPD_DEVICE_PROTOCOL = _pkey("26D4979A-E643-4626-9E2B-736DC0C92FDC", 19)

    WPD_RESOURCE_DEFAULT = _pkey("E81E79BE-34F0-41BF-B53F-F1A06AE87842", 0)
    WPD_RESOURCE_THUMBNAIL = _pkey("E81E79BE-34F0-41BF-B53F-F1A06AE87842", 17)

    # Content-Type GUIDs
    WPD_CONTENT_TYPE_FUNCTIONAL_OBJECT = uuid.UUID("99ED0160-17FF-4C44-9D98-1D7A6F941921")
    WPD_CONTENT_TYPE_FOLDER = uuid.UUID("27E2E392-A111-48E0-AB0C-E17705A05F85")

    def _vcall(ptr: ctypes.c_void_p, idx: int, restype: type, *argtypes: type):
        vtable = ctypes.cast(
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))[0],
            ctypes.POINTER(ctypes.c_void_p),
        )
        return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtable[idx])

    def _release(ptr: ctypes.c_void_p | None) -> None:
        if ptr and ptr.value:
            try:
                _vcall(ptr, 2, wintypes.ULONG)(ptr)
            except Exception:
                pass

    def _co_create(clsid: GUID, iid: GUID) -> ctypes.c_void_p | None:
        ptr = ctypes.c_void_p()
        hr = _ole32.CoCreateInstance(
            ctypes.byref(clsid),
            None,
            CLSCTX_INPROC_SERVER,
            ctypes.byref(iid),
            ctypes.byref(ptr),
        )
        if hr < 0 or not ptr.value:
            return None
        return ptr

    class _ComScope:
        """Initializes COM on the current thread and uninitializes if needed."""

        def __init__(self) -> None:
            self._should_uninit = False

        def __enter__(self) -> "_ComScope":
            hr = _ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
            # S_OK (0) or S_FALSE (1) -> we own a CoInitialize count
            if hr in (0, 1):
                self._should_uninit = True
            return self

        def __exit__(self, *_: object) -> None:
            if self._should_uninit:
                try:
                    _ole32.CoUninitialize()
                except Exception:
                    pass


@dataclass
class MtpDeviceInfo:
    pnp_id: str
    friendly_name: str
    manufacturer: str = ""
    description: str = ""
    device_type: int = 0
    protocol: str = ""

    @property
    def label(self) -> str:
        name = self.friendly_name or self.description or "Portable Device"
        if self.manufacturer and self.manufacturer.lower() not in name.lower():
            display = f"{self.manufacturer} {name}".strip()
        else:
            display = name

        proto_tag = "PTP" if "PTP" in self.protocol.upper() and "MTP" not in self.protocol.upper() else "MTP"
        if self.device_type == WPD_DEVICE_TYPE_CAMERA:
            return f"Camera ({display}) [{proto_tag}]"
        if self.device_type == WPD_DEVICE_TYPE_PHONE:
            return f"Smartphone ({display}) [{proto_tag}]"
        if self.device_type == WPD_DEVICE_TYPE_VIDEO:
            return f"Video Camera ({display}) [{proto_tag}]"
        return f"Portable Device ({display}) [{proto_tag}]"


@dataclass
class MtpMediaObject:
    device_pnp_id: str
    object_id: str
    filename: str
    file_size: int
    date_taken: datetime
    is_video: bool
    relative_folder: str = ""
    thumbnail_bytes: bytes | None = None


def is_wpd_available() -> bool:
    return IS_WINDOWS


def is_mtp_ptp_device(pnp_id: str, device_type: int = 0, protocol: str = "") -> bool:
    """
    Filters out standard Windows drive letters wrapped by WPD (wpdbusenum / storage#volume),
    keeping true MTP/PTP cameras, smartphones, and portable media devices.
    """
    low = (pnp_id or "").lower()
    if "wpdbusenum" in low or "storage#volume" in low:
        return False
    if device_type in (
        WPD_DEVICE_TYPE_CAMERA,
        WPD_DEVICE_TYPE_MEDIA_PLAYER,
        WPD_DEVICE_TYPE_PHONE,
        WPD_DEVICE_TYPE_VIDEO,
    ):
        return True
    proto_up = (protocol or "").upper()
    if "MTP" in proto_up or "PTP" in proto_up:
        return True
    # USB/Bluetooth WPD devices that are not filesystem volumes
    if low.startswith("\\\\?\\usb#") or low.startswith("\\\\?\\bth"):
        return True
    return False


def _ole_date_to_datetime(ole_date: float) -> datetime:
    # OLE Automation DATE: days since 1899-12-30
    return datetime(1899, 12, 30) + timedelta(days=ole_date)


def _filetime_to_datetime(ft) -> datetime:
    ticks = (int(ft.dwHighDateTime) << 32) | int(ft.dwLowDateTime)
    # FILETIME: 100-ns intervals since 1601-01-01
    us = ticks // 10
    return datetime(1601, 1, 1) + timedelta(microseconds=us)


def _parse_propvariant_datetime(pv: PROPVARIANT) -> datetime | None:
    try:
        if pv.vt == VT_DATE:
            return _ole_date_to_datetime(float(pv.val.dblVal))
        if pv.vt == VT_FILETIME:
            return _filetime_to_datetime(pv.val.filetime)
        if pv.vt == VT_LPWSTR and pv.val.pwszVal:
            raw = ctypes.wstring_at(pv.val.pwszVal).strip()
            for fmt in ("%Y/%m/%d:%H:%M:%S.%f", "%Y/%m/%d:%H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y:%m:%d %H:%M:%S"):
                try:
                    return datetime.strptime(raw[:23], fmt)
                except ValueError:
                    continue
    except Exception:
        pass
    return None


def _read_manager_string(mgr: ctypes.c_void_p, vtable_idx: int, pnp_id: str) -> str:
    fn = _vcall(mgr, vtable_idx, ctypes.c_long, wintypes.LPCWSTR, ctypes.c_wchar_p, ctypes.POINTER(wintypes.DWORD))
    cch = wintypes.DWORD(0)
    hr = fn(mgr, pnp_id, None, ctypes.byref(cch))
    if hr < 0 or cch.value == 0:
        return ""
    buf = ctypes.create_unicode_buffer(cch.value)
    hr = fn(mgr, pnp_id, buf, ctypes.byref(cch))
    if hr < 0:
        return ""
    return buf.value.strip()


def _get_values_string(vals: ctypes.c_void_p, key: PROPERTYKEY) -> str:
    fn = _vcall(vals, 8, ctypes.c_long, ctypes.POINTER(PROPERTYKEY), ctypes.POINTER(ctypes.c_void_p))
    p_str = ctypes.c_void_p()
    hr = fn(vals, ctypes.byref(key), ctypes.byref(p_str))
    if hr < 0 or not p_str.value:
        return ""
    try:
        return ctypes.wstring_at(p_str.value).strip()
    finally:
        _ole32.CoTaskMemFree(p_str.value)


def _get_values_uint32(vals: ctypes.c_void_p, key: PROPERTYKEY, default: int = 0) -> int:
    fn = _vcall(vals, 10, ctypes.c_long, ctypes.POINTER(PROPERTYKEY), ctypes.POINTER(wintypes.ULONG))
    out = wintypes.ULONG(default)
    hr = fn(vals, ctypes.byref(key), ctypes.byref(out))
    return int(out.value) if hr >= 0 else default


def _get_values_uint64(vals: ctypes.c_void_p, key: PROPERTYKEY, default: int = 0) -> int:
    fn = _vcall(vals, 14, ctypes.c_long, ctypes.POINTER(PROPERTYKEY), ctypes.POINTER(ctypes.c_uint64))
    out = ctypes.c_uint64(default)
    hr = fn(vals, ctypes.byref(key), ctypes.byref(out))
    return int(out.value) if hr >= 0 else default


def _get_values_guid(vals: ctypes.c_void_p, key: PROPERTYKEY) -> uuid.UUID | None:
    fn = _vcall(vals, 22, ctypes.c_long, ctypes.POINTER(PROPERTYKEY), ctypes.POINTER(GUID))
    out = GUID()
    hr = fn(vals, ctypes.byref(key), ctypes.byref(out))
    if hr < 0:
        return None
    return out.to_uuid()


def _get_values_propvariant(vals: ctypes.c_void_p, key: PROPERTYKEY) -> PROPVARIANT | None:
    fn = _vcall(vals, 6, ctypes.c_long, ctypes.POINTER(PROPERTYKEY), ctypes.POINTER(PROPVARIANT))
    pv = PROPVARIANT()
    hr = fn(vals, ctypes.byref(key), ctypes.byref(pv))
    if hr < 0:
        return None
    return pv


class MtpDeviceSession:
    """Context-managed WPD IPortableDevice session for scanning, reading, and deleting objects."""

    def __init__(self, pnp_id: str) -> None:
        self.pnp_id = pnp_id
        self._com_scope: _ComScope | None = None
        self._device: ctypes.c_void_p | None = None
        self._content: ctypes.c_void_p | None = None
        self._properties: ctypes.c_void_p | None = None
        self._resources: ctypes.c_void_p | None = None

    def __enter__(self) -> "MtpDeviceSession":
        if not IS_WINDOWS:
            raise RuntimeError("WPD MTP/PTP is only supported on Windows.")
        self._com_scope = _ComScope().__enter__()
        client_info = _co_create(CLSID_PortableDeviceValues, IID_IPortableDeviceValues)
        if not client_info:
            raise RuntimeError("Failed to create IPortableDeviceValues.")
        try:
            set_str = _vcall(client_info, 7, ctypes.c_long, ctypes.POINTER(PROPERTYKEY), wintypes.LPCWSTR)
            set_u32 = _vcall(client_info, 9, ctypes.c_long, ctypes.POINTER(PROPERTYKEY), wintypes.ULONG)
            set_str(client_info, ctypes.byref(WPD_CLIENT_NAME), "Photo Gallery")
            set_u32(client_info, ctypes.byref(WPD_CLIENT_MAJOR_VERSION), 1)
            set_u32(client_info, ctypes.byref(WPD_CLIENT_MINOR_VERSION), 0)
            set_u32(client_info, ctypes.byref(WPD_CLIENT_REVISION), 0)
            set_u32(client_info, ctypes.byref(WPD_CLIENT_SECURITY_QUALITY_OF_SERVICE), 0x00020000)

            dev = _co_create(CLSID_PortableDeviceFTM, IID_IPortableDevice)
            if not dev:
                raise RuntimeError("Failed to create IPortableDevice.")
            open_fn = _vcall(dev, 3, ctypes.c_long, wintypes.LPCWSTR, ctypes.c_void_p)
            hr = open_fn(dev, self.pnp_id, client_info)
            if hr < 0:
                _release(dev)
                raise OSError(f"Failed to open WPD device (hr=0x{hr & 0xFFFFFFFF:08X})")
            self._device = dev

            content_ptr = ctypes.c_void_p()
            content_fn = _vcall(dev, 5, ctypes.c_long, ctypes.POINTER(ctypes.c_void_p))
            if content_fn(dev, ctypes.byref(content_ptr)) >= 0 and content_ptr.value:
                self._content = content_ptr
                props_ptr = ctypes.c_void_p()
                props_fn = _vcall(content_ptr, 4, ctypes.c_long, ctypes.POINTER(ctypes.c_void_p))
                if props_fn(content_ptr, ctypes.byref(props_ptr)) >= 0 and props_ptr.value:
                    self._properties = props_ptr
                res_ptr = ctypes.c_void_p()
                res_fn = _vcall(content_ptr, 5, ctypes.c_long, ctypes.POINTER(ctypes.c_void_p))
                if res_fn(content_ptr, ctypes.byref(res_ptr)) >= 0 and res_ptr.value:
                    self._resources = res_ptr
        finally:
            _release(client_info)
        return self

    def __exit__(self, *_: object) -> None:
        _release(self._resources)
        _release(self._properties)
        _release(self._content)
        if self._device and self._device.value:
            try:
                _vcall(self._device, 8, ctypes.c_long)(self._device)  # Close()
            except Exception:
                pass
            _release(self._device)
        self._resources = None
        self._properties = None
        self._content = None
        self._device = None
        if self._com_scope is not None:
            self._com_scope.__exit__(None, None, None)
            self._com_scope = None

    def get_device_metadata(self) -> tuple[int, str]:
        """Returns (WPD_DEVICE_TYPE, WPD_DEVICE_PROTOCOL) for 'DEVICE'."""
        if not self._properties:
            return (0, "")
        keys = _co_create(CLSID_PortableDeviceKeyCollection, IID_IPortableDeviceKeyCollection)
        if not keys:
            return (0, "")
        vals = ctypes.c_void_p()
        try:
            add_key = _vcall(keys, 5, ctypes.c_long, ctypes.POINTER(PROPERTYKEY))
            add_key(keys, ctypes.byref(WPD_DEVICE_TYPE))
            add_key(keys, ctypes.byref(WPD_DEVICE_PROTOCOL))
            get_vals = _vcall(
                self._properties,
                5,
                ctypes.c_long,
                wintypes.LPCWSTR,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            )
            hr = get_vals(self._properties, "DEVICE", keys, ctypes.byref(vals))
            if hr < 0 or not vals.value:
                return (0, "")
            dev_type = _get_values_uint32(vals, WPD_DEVICE_TYPE, 0)
            proto = _get_values_string(vals, WPD_DEVICE_PROTOCOL)
            return (dev_type, proto)
        finally:
            _release(vals)
            _release(keys)

    def _create_media_key_collection(self) -> ctypes.c_void_p | None:
        keys = _co_create(CLSID_PortableDeviceKeyCollection, IID_IPortableDeviceKeyCollection)
        if not keys:
            return None
        add_key = _vcall(keys, 5, ctypes.c_long, ctypes.POINTER(PROPERTYKEY))
        for k in (
            WPD_OBJECT_NAME,
            WPD_OBJECT_ORIGINAL_FILE_NAME,
            WPD_OBJECT_CONTENT_TYPE,
            WPD_OBJECT_SIZE,
            WPD_OBJECT_DATE_CREATED,
            WPD_OBJECT_DATE_MODIFIED,
        ):
            add_key(keys, ctypes.byref(k))
        return keys

    def _enum_children(self, parent_id: str) -> list[str]:
        if not self._content:
            return []
        enum_ptr = ctypes.c_void_p()
        enum_fn = _vcall(
            self._content,
            3,
            ctypes.c_long,
            wintypes.DWORD,
            wintypes.LPCWSTR,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        hr = enum_fn(self._content, 0, parent_id, None, ctypes.byref(enum_ptr))
        if hr < 0 or not enum_ptr.value:
            return []
        children: list[str] = []
        try:
            next_fn = _vcall(
                enum_ptr,
                3,
                ctypes.c_long,
                wintypes.ULONG,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(wintypes.ULONG),
            )
            batch_size = 32
            buf = (ctypes.c_void_p * batch_size)()
            fetched = wintypes.ULONG(0)
            while True:
                fetched.value = 0
                hr = next_fn(enum_ptr, batch_size, buf, ctypes.byref(fetched))
                if hr < 0 or fetched.value == 0:
                    break
                for i in range(fetched.value):
                    if buf[i]:
                        children.append(ctypes.wstring_at(buf[i]))
                        _ole32.CoTaskMemFree(buf[i])
                if hr != 0:  # S_FALSE (1) means end of enumeration
                    break
        finally:
            _release(enum_ptr)
        return children

    def read_resource_bytes(self, object_id: str, resource_key: PROPERTYKEY, max_bytes: int = 512 * 1024) -> bytes | None:
        if not self._resources:
            return None
        stream_ptr = ctypes.c_void_p()
        opt_buf = wintypes.DWORD(0)
        get_stream = _vcall(
            self._resources,
            5,
            ctypes.c_long,
            wintypes.LPCWSTR,
            ctypes.POINTER(PROPERTYKEY),
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(ctypes.c_void_p),
        )
        hr = get_stream(self._resources, object_id, ctypes.byref(resource_key), STGM_READ, ctypes.byref(opt_buf), ctypes.byref(stream_ptr))
        if hr < 0 or not stream_ptr.value:
            return None
        try:
            read_fn = _vcall(stream_ptr, 3, ctypes.c_long, ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG))
            chunk_size = min(max(opt_buf.value, 16384), 65536)
            buf = ctypes.create_string_buffer(chunk_size)
            out = bytearray()
            read_len = wintypes.ULONG(0)
            while len(out) < max_bytes:
                read_len.value = 0
                hr = read_fn(stream_ptr, buf, chunk_size, ctypes.byref(read_len))
                if hr < 0 or read_len.value == 0:
                    break
                out.extend(buf.raw[: read_len.value])
                if hr != 0:
                    break
            return bytes(out) if out else None
        finally:
            _release(stream_ptr)

    def copy_object_to_file(self, object_id: str, dest_path: Path, expected_size: int = 0) -> int:
        if not self._resources:
            raise RuntimeError("WPD resources interface unavailable.")
        stream_ptr = ctypes.c_void_p()
        opt_buf = wintypes.DWORD(0)
        get_stream = _vcall(
            self._resources,
            5,
            ctypes.c_long,
            wintypes.LPCWSTR,
            ctypes.POINTER(PROPERTYKEY),
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(ctypes.c_void_p),
        )
        hr = get_stream(
            self._resources,
            object_id,
            ctypes.byref(WPD_RESOURCE_DEFAULT),
            STGM_READ,
            ctypes.byref(opt_buf),
            ctypes.byref(stream_ptr),
        )
        if hr < 0 or not stream_ptr.value:
            raise OSError(f"Failed to open MTP stream for {object_id} (hr=0x{hr & 0xFFFFFFFF:08X})")

        total_written = 0
        try:
            read_fn = _vcall(stream_ptr, 3, ctypes.c_long, ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG))
            chunk_size = max(opt_buf.value, 262144)  # 256 KB
            buf = ctypes.create_string_buffer(chunk_size)
            read_len = wintypes.ULONG(0)
            with open(dest_path, "wb") as f:
                while True:
                    read_len.value = 0
                    hr = read_fn(stream_ptr, buf, chunk_size, ctypes.byref(read_len))
                    if hr < 0:
                        raise OSError(f"Error reading MTP stream (hr=0x{hr & 0xFFFFFFFF:08X})")
                    if read_len.value == 0:
                        break
                    f.write(buf.raw[: read_len.value])
                    total_written += read_len.value
                    if hr != 0:
                        break
        finally:
            _release(stream_ptr)

        if expected_size > 0 and total_written != expected_size:
            raise OSError(f"Byte size mismatch transferring {dest_path.name}: got {total_written}, expected {expected_size}")
        return total_written

    def delete_object(self, object_id: str) -> bool:
        if not self._content:
            return False
        coll = _co_create(CLSID_PortableDevicePropVariantCollection, IID_IPortableDevicePropVariantCollection)
        if not coll:
            return False
        try:
            pv = PROPVARIANT()
            pv.vt = VT_LPWSTR
            wbuf = ctypes.create_unicode_buffer(object_id)
            pv.val.pwszVal = ctypes.cast(wbuf, ctypes.c_void_p).value
            add_fn = _vcall(coll, 5, ctypes.c_long, ctypes.POINTER(PROPVARIANT))
            if add_fn(coll, ctypes.byref(pv)) < 0:
                return False
            del_fn = _vcall(
                self._content,
                8,
                ctypes.c_long,
                wintypes.DWORD,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            )
            hr = del_fn(self._content, PORTABLE_DEVICE_DELETE_NO_RECURSION, coll, None)
            return hr >= 0
        finally:
            _release(coll)

    def scan_media(
        self,
        supported_exts: set[str],
        video_exts: set[str],
        progress_cb: Callable[[int], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[MtpMediaObject]:
        if not self._content or not self._properties:
            return []
        keys = self._create_media_key_collection()
        if not keys:
            return []

        results: list[MtpMediaObject] = []
        examined = 0
        get_vals = _vcall(
            self._properties,
            5,
            ctypes.c_long,
            wintypes.LPCWSTR,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )

        # Stack of (object_id, relative_folder_path)
        queue: list[tuple[str, str]] = [("DEVICE", "")]
        try:
            while queue:
                if cancel_check and cancel_check():
                    break
                parent_id, rel_dir = queue.pop(0)
                for child_id in self._enum_children(parent_id):
                    if cancel_check and cancel_check():
                        break
                    vals = ctypes.c_void_p()
                    try:
                        hr = get_vals(self._properties, child_id, keys, ctypes.byref(vals))
                        if hr < 0 or not vals.value:
                            continue

                        ctype = _get_values_guid(vals, WPD_OBJECT_CONTENT_TYPE)
                        orig_name = _get_values_string(vals, WPD_OBJECT_ORIGINAL_FILE_NAME)
                        obj_name = _get_values_string(vals, WPD_OBJECT_NAME)
                        filename = orig_name if (orig_name and Path(orig_name).suffix) else (obj_name or orig_name)

                        if ctype in (WPD_CONTENT_TYPE_FUNCTIONAL_OBJECT, WPD_CONTENT_TYPE_FOLDER):
                            folder_label = obj_name or orig_name or child_id
                            low_folder = folder_label.strip().lower()
                            if low_folder in _SKIP_FOLDER_NAMES or low_folder.startswith("."):
                                continue
                            next_rel = f"{rel_dir}/{folder_label}".strip("/") if rel_dir else folder_label
                            queue.append((child_id, next_rel))
                            continue

                        if not filename:
                            continue
                        ext = Path(filename).suffix.lower()
                        if ext not in supported_exts:
                            continue

                        examined += 1
                        if progress_cb and examined % 15 == 0:
                            progress_cb(examined)

                        file_size = _get_values_uint64(vals, WPD_OBJECT_SIZE, 0)
                        dt: datetime | None = None
                        for dt_key in (WPD_OBJECT_DATE_CREATED, WPD_OBJECT_DATE_MODIFIED):
                            pv = _get_values_propvariant(vals, dt_key)
                            if pv is not None:
                                dt = _parse_propvariant_datetime(pv)
                                _ole32.PropVariantClear(ctypes.byref(pv))
                                if dt is not None:
                                    break
                        if dt is None:
                            dt = datetime.now()

                        is_vid = ext in video_exts
                        thumb_bytes: bytes | None = None
                        if not is_vid:
                            thumb_bytes = self.read_resource_bytes(child_id, WPD_RESOURCE_THUMBNAIL, max_bytes=128 * 1024)

                        results.append(
                            MtpMediaObject(
                                device_pnp_id=self.pnp_id,
                                object_id=child_id,
                                filename=filename,
                                file_size=file_size,
                                date_taken=dt,
                                is_video=is_vid,
                                relative_folder=rel_dir,
                                thumbnail_bytes=thumb_bytes,
                            )
                        )
                    finally:
                        _release(vals)
        finally:
            _release(keys)
        return results


def enumerate_mtp_devices() -> list[MtpDeviceInfo]:
    """Enumerates connected MTP/PTP cameras and smartphones via Windows Portable Devices."""
    if not IS_WINDOWS:
        return []

    devices: list[MtpDeviceInfo] = []
    with _ComScope():
        mgr = _co_create(CLSID_PortableDeviceManager, IID_IPortableDeviceManager)
        if not mgr:
            return []
        try:
            # Refresh device list first so newly plugged devices are seen
            try:
                _vcall(mgr, 4, ctypes.c_long)(mgr)
            except Exception:
                pass

            get_devices = _vcall(
                mgr,
                3,
                ctypes.c_long,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(wintypes.DWORD),
            )
            count = wintypes.DWORD(0)
            hr = get_devices(mgr, None, ctypes.byref(count))
            if hr < 0 or count.value == 0:
                return []

            arr = (ctypes.c_void_p * count.value)()
            hr = get_devices(mgr, arr, ctypes.byref(count))
            if hr < 0:
                return []

            pnp_ids: list[str] = []
            for i in range(count.value):
                if arr[i]:
                    pnp_ids.append(ctypes.wstring_at(arr[i]))
                    _ole32.CoTaskMemFree(arr[i])

            for pnp_id in pnp_ids:
                # Fast pre-filter for obvious mounted volume PnP IDs
                low = pnp_id.lower()
                if "wpdbusenum" in low or "storage#volume" in low:
                    continue

                friendly = _read_manager_string(mgr, 5, pnp_id)
                desc = _read_manager_string(mgr, 6, pnp_id)
                mfg = _read_manager_string(mgr, 7, pnp_id)

                dev_type = 0
                proto = ""
                try:
                    with MtpDeviceSession(pnp_id) as sess:
                        dev_type, proto = sess.get_device_metadata()
                except Exception:
                    # Locked phones may refuse Open() until unlocked, but still show up in GetDevices
                    pass

                if is_mtp_ptp_device(pnp_id, dev_type, proto):
                    devices.append(
                        MtpDeviceInfo(
                            pnp_id=pnp_id,
                            friendly_name=friendly,
                            manufacturer=mfg,
                            description=desc,
                            device_type=dev_type,
                            protocol=proto,
                        )
                    )
        finally:
            _release(mgr)
    return devices
