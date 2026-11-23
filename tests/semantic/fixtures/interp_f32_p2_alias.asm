; SHADER-826, synthetic fragment input attr0.z = (1, 3, 5) at the three vertices.
; Hardware I = 0.25. Initial v1 = 0.5 must be replaced by P1: 1 + (3 - 1) * 0.25 = 1.5.
; P2 reads that 1.5 as both J and accumulator: 1.5 + (5 - 1) * 1.5 = 7.5.
; Expected mrt0 words: 0x40f00000 0x40f00000 0x40f00000 0x40f00000.
s_mov_b32 m0, s2
v_mov_b32_e32 v1, 0.5
v_interp_p1_f32_e32 v1, v0, attr0.z
v_interp_p2_f32_e32 v1, v1, attr0.z
exp mrt0 v1, v1, v1, v1 done vm
s_endpgm
