# System memory watchdog (macOS). Keeps the whole machine under a RAM ceiling:
# above ~90 % used a 16 GB Mac starts swapping hard and the UI freezes.
import ctypes
import ctypes.util
import os

_libc = ctypes.CDLL(ctypes.util.find_library("c"))
_PAGE = os.sysconf("SC_PAGE_SIZE")
TOTAL = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def _sysctl_int(name):
    val = ctypes.c_uint64(0)
    size = ctypes.c_size_t(ctypes.sizeof(val))
    if _libc.sysctlbyname(name.encode(), ctypes.byref(val), ctypes.byref(size), None, 0) != 0:
        return None
    return val.value


def available_bytes():
    """Free + inactive + speculative + purgeable pages: what can be handed out without swapping."""
    free = _sysctl_int("vm.page_free_count") or 0
    spec = _sysctl_int("vm.page_speculative_count") or 0
    purg = _sysctl_int("vm.page_purgeable_count") or 0
    inactive = _sysctl_int("vm.page_pageable_internal_count")  # anonymous pageable, not directly freeable
    ext = _sysctl_int("vm.page_pageable_external_count") or 0     # file cache: freeable
    del inactive
    return (free + spec + purg + ext) * _PAGE


def used_fraction():
    return 1.0 - available_bytes() / TOTAL


def pressure_level():
    """1 = normal, 2 = warning, 4 = critical (kern.memorystatus_vm_pressure_level)."""
    return _sysctl_int("kern.memorystatus_vm_pressure_level") or 1


def headroom_bytes(limit):
    """Bytes that may still be allocated before the system crosses `limit` (fraction of RAM)."""
    return int(available_bytes() - (1.0 - limit) * TOTAL)
