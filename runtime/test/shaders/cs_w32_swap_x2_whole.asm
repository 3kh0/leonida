; SHADER-868: contended buffer and global swap_x2 must swap the 64-bit element whole. Every invocation of 64
; workgroups of 32 swaps its own pair into four shared 64-bit elements, one per form, and stores the two pairs the glc
; forms got back. Workgroup id in s4 (the default header: 4 user SGPRs, then TGID_X), thread id in v0, so invocation
; i = s4 * 32 + v0. The heap starts as the 0xEE fill, and s[0:3] is the raw V# over it (base = heap address 0).
;   form f (0 global glc, 1 buffer glc, 2 global without glc, 3 buffer without glc) swaps in
;     low = (0x51 + f) << 24 | i, high = ~low (the halves of a pair belong together)
;   the elements: bytes 0x8000 (f = 0), 0x8008 (1), 0x8010 (2), 0x8018 (3): words 8192..8199
;   out[2 * i], out[2 * i + 1]: the pair global_atomic_swap_x2 ... glc gave invocation i (words 0..4095)
;   out[4096 + 2 * i], out[4097 + 2 * i]: the pair buffer_atomic_swap_x2 ... glc gave it (words 4096..8191)
; What must hold whatever order the invocations ran in (the console swaps the element whole): for each glc form, the
; 2,048 returned pairs plus the element's final pair are exactly the fill pair plus the 2,048 replacement pairs, each
; once, so no pair is torn (A's low dword with B's high one) and none is lost or handed out twice; each element
; without glc ends as one whole replacement pair. One 32-bit exchange per dword (rdna2spirv.py --no-int64-atomics)
; tears most returned pairs on NVIDIA (#4964). check.py's swap_x2_whole_words has the check.
; Run: VKC_OUT_WORDS=8208 VKC_DUMP=<file> vk_compute_test cs_w32_swap_x2_whole.spv 64 dump
; group: 32,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
v_lshl_add_u32 v1, s4, 5, v0
v_or_b32_e32 v30, 0x51000000, v1
v_not_b32_e32 v31, v30
v_or_b32_e32 v32, 0x52000000, v1
v_not_b32_e32 v33, v32
v_or_b32_e32 v34, 0x53000000, v1
v_not_b32_e32 v35, v34
v_or_b32_e32 v38, 0x54000000, v1
v_not_b32_e32 v39, v38
v_mov_b32_e32 v10, 0x8000
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v12, 0x8010
v_mov_b32_e32 v13, 0
v_mov_b32_e32 v14, 0x8008
v_mov_b32_e32 v15, 0x8018
v_mov_b32_e32 v36, 0
v_mov_b32_e32 v37, 0
global_atomic_swap_x2 v[36:37], v[10:11], v[30:31], off glc
buffer_atomic_swap_x2 v[32:33], v14, s[0:3], 0 offen glc
global_atomic_swap_x2 v[12:13], v[34:35], off
buffer_atomic_swap_x2 v[38:39], v15, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_lshlrev_b32_e32 v5, 3, v1
buffer_store_dwordx2 v[36:37], v5, s[0:3], 0 offen
v_add_nc_u32_e32 v6, 0x4000, v5
buffer_store_dwordx2 v[32:33], v6, s[0:3], 0 offen
s_endpgm
