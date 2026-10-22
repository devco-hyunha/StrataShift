"""Process RSS: Linux `/proc/<pid>/status` VmRSS, Windows Working Set (p6-10).

`None` means "cannot measure on this OS" — never 0, so a `--max-rss-gb` ceiling and the
RSS flatness gate cannot pass silently where nothing is measured.
"""
from __future__ import annotations

import ctypes
import sys

# Win32 values (Microsoft Learn: Process Security and Access Rights; psapi.h
# PROCESS_MEMORY_COUNTERS) — OS ABI, not an engine format value.
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_VM_READ = 0x0010


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _linux_rss(pid: int | None) -> int | None:
    path = f"/proc/{'self' if pid is None else pid}/status"
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _load_win32() -> tuple | None:
    """Bind the Win32 calls once at import: per-sample WinDLL/argtypes setup allocates
    Python objects that land inside the tracemalloc slice-heap window (p6-10 measured
    +50..90 KB on the small emit fixtures before this)."""
    if sys.platform != "win32":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_ProcessMemoryCounters), ctypes.c_ulong]
    return kernel32, psapi, _ProcessMemoryCounters()


_WIN32 = _load_win32()


def _windows_rss(pid: int | None) -> int | None:
    if _WIN32 is None:
        return None
    kernel32, psapi, counters = _WIN32
    if pid is None:
        handle = kernel32.GetCurrentProcess()  # pseudo handle, no CloseHandle needed
    else:
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ,
                                      False, pid)
        if not handle:
            return None
    try:
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize)
    finally:
        if pid is not None:
            kernel32.CloseHandle(handle)


def process_rss_bytes(pid: int | None = None) -> int | None:
    """Resident memory of `pid` (this process when None) in bytes, or None if unmeasurable."""
    if sys.platform == "win32":
        return _windows_rss(pid)
    if sys.platform.startswith("linux"):
        return _linux_rss(pid)
    return None


def current_rss_bytes() -> int | None:
    return process_rss_bytes()
