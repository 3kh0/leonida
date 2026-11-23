; v_fmamk_f16 / v_fmaak_f16 (translator support #1989; docs/design.md row "v_fmamk_f16, v_fmaak_f16"), run on a GPU.
; fmamk: D = S0 * K + S1 (printed vdst, src0, K, vsrc1); fmaak: D = S0 * S1 + K (printed vdst, src0, vsrc1, K).
; One rounding to f16 (fused); the result goes to the low half of D and the high half is kept.
; Every lane computes the same values; EXEC is cut to lane 0 before the stores, so the 32 lanes do not race on one address (as in cs_w32_new_ops). Lane 0 stores eight dwords:
;   out[0] = fmamk(2.0, K 3.0, 1.0) over 0xDEAD0000   = 0xDEAD4700 (7.0, high half kept; S0 * S1 + K would give 5.0)
;   out[1] = fmaak(2.0, 3.0, K 0.5) over 0xBEEF0000   = 0xBEEF4680 (6.5, high half kept)
;   out[2] = fmamk(0x3C01, K 0x3C01, 0x1000)          = 0x00003C03 (exact 1 + 2^-9 + 2^-11 + 2^-20 rounds up; rounding
;            the product first gives 1 + 2^-9, and the tie 1 + 2^-9 + 2^-11 then rounds to even: 0x3C02)
;   out[3] = fmaak(0x3C01, 0x3C01, K 0x1000)          = 0x00003C03 (the same case in the fmaak order)
;   out[4] = fmamk(2.0, K 0.1, 0.0)                   = 0x00003266 (the float literal 0.1 becomes f16 0x2E66, times 2)
;   out[5] = fmaak(3.0, -2.0, K 10.0)                 = 0x00004400 (4.0; S0 * K + S1 would give 28.0)
;   out[6] = fmaak(s6 = 3.0, 2.0, K 1.0)              = 0x00004700 (7.0, SGPR src0)
;   out[7] = fmamk(256.0, K 256.0, 0.0)               = 0x00007C00 (65536 rounds to +inf)
; group: 32,1,1
; Run: vk_compute_test cs_w32_fmamk_f16.spv 1 dump -> out[0..7] = 3735897856 3203352192 15363 15363 12902 17408 18176 31744
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0x4000
v_mov_b32_e32 v2, 0x3c00
v_mov_b32_e32 v3, 0xdead0000
v_fmamk_f16 v3, v1, 0x4200, v2
v_mov_b32_e32 v2, 0x4200
v_mov_b32_e32 v4, 0xbeef0000
v_fmaak_f16 v4, v1, v2, 0x3800
v_mov_b32_e32 v5, 0x3c01
v_mov_b32_e32 v6, 0x1000
v_mov_b32_e32 v7, 0
v_fmamk_f16 v7, v5, 0x3c01, v6
v_mov_b32_e32 v8, 0
v_fmaak_f16 v8, v5, v5, 0x1000
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v9, 0
v_fmamk_f16 v9, v1, 0.1, v6
v_mov_b32_e32 v11, 0xc000
v_mov_b32_e32 v10, 0
v_fmaak_f16 v10, v2, v11, 0x4900
s_mov_b32 s6, 0x4200
v_mov_b32_e32 v13, 0x4000
v_mov_b32_e32 v12, 0
v_fmaak_f16 v12, s6, v13, 0x3c00
v_mov_b32_e32 v15, 0x5c00
v_mov_b32_e32 v14, 0
v_fmamk_f16 v14, v15, 0x5c00, v6
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v25, 0
buffer_store_dword v3, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 4
buffer_store_dword v4, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 8
buffer_store_dword v7, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 12
buffer_store_dword v8, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 16
buffer_store_dword v9, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 20
buffer_store_dword v10, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 24
buffer_store_dword v12, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 28
buffer_store_dword v14, v25, s[0:3], 0 offen
s_endpgm