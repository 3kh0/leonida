; Synthetic BUILD-873 probe. At U = 1.34375 in the eight-texel red ramp (36*x),
; wrap/linear gives 81, mirror/point gives 180, and border/point gives 0.
; Export the three red values in RGB with opaque alpha.
; fssm-at: 0x10
; fssm-data: 0x873 0x5053
s_load_dwordx8 s[8:15], s[0:1], 0
v_mov_b32_e32 v0, 0x3fac0000
v_mov_b32_e32 v1, 0.5
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x90
s_mov_b32 s17, 0x500000
s_mov_b32 s18, 0x500000
s_mov_b32 s19, 0
image_sample_lz v4, v[0:1], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
s_mov_b32 s20, 0x91
s_mov_b32 s21, 0x500000
s_mov_b32 s22, 0
s_mov_b32 s23, 0
image_sample_lz v5, v[0:1], s[8:15], s[20:23] dmask:0x1 dim:SQ_RSRC_IMG_2D
s_mov_b32 s24, 0x96
s_mov_b32 s25, 0x500000
s_mov_b32 s26, 0
s_mov_b32 s27, 0
image_sample_lz v6, v[0:1], s[8:15], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mov_b32_e32 v7, 1.0
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
