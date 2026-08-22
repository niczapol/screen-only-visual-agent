from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass


TOKEN_QUERY = 0x0008
TOKEN_INTEGRITY_LEVEL = 25
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


@dataclass(frozen=True)
class IntegrityLevel:
    name: str
    rid: int


class SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", wintypes.LPVOID), ("Attributes", wintypes.DWORD)]


class TOKEN_MANDATORY_LABEL(ctypes.Structure):
    _fields_ = [("Label", SID_AND_ATTRIBUTES)]


def get_current_integrity_level() -> IntegrityLevel | None:
    return get_process_integrity_level(ctypes.windll.kernel32.GetCurrentProcessId())


def get_process_integrity_level(pid: int) -> IntegrityLevel | None:
    advapi = ctypes.windll.advapi32
    kernel = ctypes.windll.kernel32
    _configure_ctypes(advapi, kernel)

    process = kernel.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not process:
        return None

    token = wintypes.HANDLE()
    if not advapi.OpenProcessToken(process, TOKEN_QUERY, ctypes.byref(token)):
        kernel.CloseHandle(process)
        return None

    try:
        size = wintypes.DWORD(0)
        advapi.GetTokenInformation(token, TOKEN_INTEGRITY_LEVEL, None, 0, ctypes.byref(size))
        buffer = ctypes.create_string_buffer(size.value)
        if not advapi.GetTokenInformation(token, TOKEN_INTEGRITY_LEVEL, buffer, size, ctypes.byref(size)):
            return None

        label = ctypes.cast(buffer, ctypes.POINTER(TOKEN_MANDATORY_LABEL)).contents
        sub_authority_count = advapi.GetSidSubAuthorityCount(label.Label.Sid).contents.value
        rid = advapi.GetSidSubAuthority(label.Label.Sid, sub_authority_count - 1).contents.value
        return IntegrityLevel(_integrity_name(rid), int(rid))
    finally:
        kernel.CloseHandle(token)
        kernel.CloseHandle(process)


def target_requires_elevation(target_pid: int) -> bool:
    current = get_current_integrity_level()
    target = get_process_integrity_level(target_pid)
    if current is None or target is None:
        return False
    return target.rid > current.rid


def _integrity_name(rid: int) -> str:
    if rid >= 0x3000:
        return "High"
    if rid >= 0x2000:
        return "Medium"
    if rid >= 0x1000:
        return "Low"
    return "Unknown"


def _configure_ctypes(advapi: ctypes.WinDLL, kernel: ctypes.WinDLL) -> None:
    advapi.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi.OpenProcessToken.restype = wintypes.BOOL
    advapi.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi.GetTokenInformation.restype = wintypes.BOOL
    advapi.GetSidSubAuthorityCount.argtypes = [wintypes.LPVOID]
    advapi.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
    advapi.GetSidSubAuthority.argtypes = [wintypes.LPVOID, wintypes.DWORD]
    advapi.GetSidSubAuthority.restype = ctypes.POINTER(wintypes.DWORD)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GetCurrentProcessId.restype = wintypes.DWORD
