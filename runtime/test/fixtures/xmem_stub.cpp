// Link-time stand-in for xmem.dll. The bundled DLL imports XMemAlloc/XMemFree and the startup guest imports
// XMemVirtualAlloc. The mapped guests need the shim registry to answer these imports for allocation to succeed.
#include <windows.h>
#include <cstddef>
#include <cstdint>

extern "C" __declspec(dllexport) void* __stdcall XMemAlloc(size_t, uint64_t) { return nullptr; }
extern "C" __declspec(dllexport) void __stdcall XMemFree(void*, uint64_t) {}
extern "C" __declspec(dllexport) void* WINAPI XMemVirtualAlloc(void*, SIZE_T, DWORD, ULONGLONG, DWORD) { return nullptr; }
