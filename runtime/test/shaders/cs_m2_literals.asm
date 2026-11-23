; Synthetic BUILD-873 probe. At U = 1.34375 in the eight-texel red ramp (36*x),
; wrap/linear gives 81, mirror/point gives 180, and mirror/linear gives 171.
; Store these normalized red values and a completion marker through a structured UAV.
; group: 1,1,1
; rsrc2: 0x8088
; fssm-at: 0x150
; fssm-data: 0x873 0x4353
s_load_dwordx8 s[8:15], s[0:1], 0
s_load_dwordx4 s[28:31], s[2:3], 0
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
s_mov_b32 s24, 0x91
s_mov_b32 s25, 0x500000
s_mov_b32 s26, 0x500000
s_mov_b32 s27, 0
image_sample_lz v6, v[0:1], s[8:15], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mov_b32_e32 v7, 1.0
v_mov_b32_e32 v0, 0
buffer_store_dwordx4 v[4:7], v0, s[28:31], 0 idxen
s_endpgm
