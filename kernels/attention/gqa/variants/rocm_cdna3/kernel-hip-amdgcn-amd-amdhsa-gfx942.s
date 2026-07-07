	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.section	.text._Z10attend_kerILi128EEv12attn_globalsIXT_EE,"axG",@progbits,_Z10attend_kerILi128EEv12attn_globalsIXT_EE,comdat
	.protected	_Z10attend_kerILi128EEv12attn_globalsIXT_EE ; -- Begin function _Z10attend_kerILi128EEv12attn_globalsIXT_EE
	.globl	_Z10attend_kerILi128EEv12attn_globalsIXT_EE
	.p2align	8
	.type	_Z10attend_kerILi128EEv12attn_globalsIXT_EE,@function
_Z10attend_kerILi128EEv12attn_globalsIXT_EE: ; @_Z10attend_kerILi128EEv12attn_globalsIXT_EE
; %bb.0:                                ; %.preheader247.preheader
	v_and_b32_e32 v60, 15, v0
	s_lshl_b32 s16, s4, 11
	s_lshl_b32 s2, s2, 4
	v_lshrrev_b32_e32 v1, 4, v0
	v_or_b32_e32 v2, s16, v60
	v_lshlrev_b32_e32 v59, 2, v1
	v_add_lshl_u32 v2, v2, s2, 13
	s_lshl_b32 s5, s3, 7
	s_load_dwordx2 s[6:7], s[0:1], 0x0
	s_load_dwordx2 s[10:11], s[0:1], 0x30
	s_load_dwordx2 s[12:13], s[0:1], 0x60
	s_load_dwordx2 s[8:9], s[0:1], 0x90
	v_add3_u32 v2, v59, s5, v2
	v_ashrrev_i32_e32 v3, 31, v2
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[6:7]
	global_load_dwordx2 v[40:41], v[2:3], off
	global_load_dwordx2 v[42:43], v[2:3], off offset:32
	global_load_dwordx2 v[44:45], v[2:3], off offset:64
	global_load_dwordx2 v[46:47], v[2:3], off offset:96
	global_load_dwordx2 v[48:49], v[2:3], off offset:128
	global_load_dwordx2 v[50:51], v[2:3], off offset:160
	global_load_dwordx2 v[52:53], v[2:3], off offset:192
	global_load_dwordx2 v[54:55], v[2:3], off offset:224
	s_load_dwordx2 s[6:7], s[0:1], 0xc0
	v_cmp_gt_u32_e32 vcc, 16, v0
	v_lshlrev_b32_e32 v56, 2, v0
	s_and_saveexec_b64 s[0:1], vcc
; %bb.1:
	v_mov_b32_e32 v2, 0xff61b1e6
	v_mov_b32_e32 v3, 0
	v_add_u32_e32 v4, 0x400, v56
	ds_write2_b32 v4, v3, v2 offset0:144 offset1:160
; %bb.2:
	s_or_b64 exec, exec, s[0:1]
	v_lshlrev_b32_e32 v2, 2, v60
	s_ashr_i32 s0, s3, 31
	v_lshl_or_b32 v61, v1, 8, v2
	v_mov_b32_e32 v2, 0x400
	s_lshr_b32 s0, s0, 29
	s_lshl_b32 s1, s4, 21
	v_and_b32_e32 v62, 0x3f0, v0
	v_lshlrev_b32_e32 v63, 6, v0
	v_lshl_add_u32 v64, v0, 5, v2
	v_lshlrev_b32_e32 v0, 10, v60
	s_add_i32 s0, s3, s0
	v_or3_b32 v66, s1, v0, v59
	v_lshlrev_b32_e32 v0, 12, v1
	s_lshl_b32 s0, s0, 4
	v_lshl_or_b32 v3, v60, 5, v2
	v_lshlrev_b32_e32 v4, 3, v1
	v_or3_b32 v67, s1, v0, v60
	v_mov_b32_e32 v0, 0
	s_and_b32 s17, s0, 0xffffff80
	v_add_u32_e32 v57, 0x680, v56
	v_add_u32_e32 v58, 0x640, v56
	v_add_u32_e32 v65, 0x600, v56
	s_mov_b32 s18, -16
	s_movk_i32 s19, 0x7fff
	s_mov_b32 s20, 0x7060302
	v_add_u32_e32 v68, v3, v4
	s_mov_b32 s21, 0x5040100
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v6, v0
	v_mov_b32_e32 v7, v0
	v_mov_b32_e32 v8, v0
	v_mov_b32_e32 v9, v0
	v_mov_b32_e32 v10, v0
	v_mov_b32_e32 v11, v0
	v_mov_b32_e32 v16, v0
	v_mov_b32_e32 v17, v0
	v_mov_b32_e32 v18, v0
	v_mov_b32_e32 v19, v0
	v_mov_b32_e32 v12, v0
	v_mov_b32_e32 v13, v0
	v_mov_b32_e32 v14, v0
	v_mov_b32_e32 v15, v0
	v_mov_b32_e32 v24, v0
	v_mov_b32_e32 v25, v0
	v_mov_b32_e32 v26, v0
	v_mov_b32_e32 v27, v0
	v_mov_b32_e32 v20, v0
	v_mov_b32_e32 v21, v0
	v_mov_b32_e32 v22, v0
	v_mov_b32_e32 v23, v0
	v_mov_b32_e32 v28, v0
	v_mov_b32_e32 v29, v0
	v_mov_b32_e32 v30, v0
	v_mov_b32_e32 v31, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_branch .LBB0_4
.LBB0_3:                                ; %.preheader244
                                        ;   in Loop: Header=BB0_4 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[32:35], v62 offset:1536
	s_add_i32 s18, s18, 16
	v_add_u32_e32 v66, 0x4000, v66
	s_cmpk_gt_u32 s18, 0x7ef
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[2:3], v[34:35], v[2:3]
	v_pk_mul_f32 v[6:7], v[34:35], v[6:7]
	v_pk_mul_f32 v[10:11], v[34:35], v[10:11]
	v_pk_mul_f32 v[18:19], v[34:35], v[18:19]
	v_pk_mul_f32 v[14:15], v[34:35], v[14:15]
	v_pk_mul_f32 v[26:27], v[34:35], v[26:27]
	v_pk_mul_f32 v[22:23], v[34:35], v[22:23]
	v_pk_mul_f32 v[30:31], v[34:35], v[30:31]
	v_add_u32_e32 v34, s17, v67
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[36:37], v[34:35], 1, s[12:13]
	global_load_ushort v35, v[36:37], off
	v_add_u32_e32 v36, 0x400, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v38, v[36:37], off
	v_add_u32_e32 v36, 0x800, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v39, v[36:37], off
	v_add_u32_e32 v36, 0xc00, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v36, v[36:37], off
	v_pk_mul_f32 v[0:1], v[32:33], v[0:1]
	v_pk_mul_f32 v[4:5], v[32:33], v[4:5]
	v_pk_mul_f32 v[8:9], v[32:33], v[8:9]
	v_pk_mul_f32 v[16:17], v[32:33], v[16:17]
	v_pk_mul_f32 v[12:13], v[32:33], v[12:13]
	v_pk_mul_f32 v[24:25], v[32:33], v[24:25]
	v_pk_mul_f32 v[20:21], v[32:33], v[20:21]
	v_pk_mul_f32 v[28:29], v[32:33], v[28:29]
	ds_read_b64 v[32:33], v68
	v_add_u32_e32 v67, 0x4000, v67
	s_waitcnt vmcnt(0)
	v_perm_b32 v37, v36, v39, s21
	v_perm_b32 v36, v38, v35, s21
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_f32_16x16x16_bf16 v[0:3], v[32:33], v[36:37], v[0:3]
	v_add_u32_e32 v36, 16, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v35, v[36:37], off
	v_add_u32_e32 v36, 0x410, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v38, v[36:37], off
	v_add_u32_e32 v36, 0x810, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v39, v[36:37], off
	v_add_u32_e32 v36, 0xc10, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v36, v[36:37], off
	s_waitcnt vmcnt(0)
	v_perm_b32 v37, v36, v39, s21
	v_perm_b32 v36, v38, v35, s21
	s_nop 1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[32:33], v[36:37], v[4:7]
	v_add_u32_e32 v36, 32, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v35, v[36:37], off
	v_add_u32_e32 v36, 0x420, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v38, v[36:37], off
	v_add_u32_e32 v36, 0x820, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v39, v[36:37], off
	v_add_u32_e32 v36, 0xc20, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v36, v[36:37], off
	s_waitcnt vmcnt(0)
	v_perm_b32 v37, v36, v39, s21
	v_perm_b32 v36, v38, v35, s21
	s_nop 1
	v_mfma_f32_16x16x16_bf16 v[8:11], v[32:33], v[36:37], v[8:11]
	v_add_u32_e32 v36, 48, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v35, v[36:37], off
	v_add_u32_e32 v36, 0x430, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v38, v[36:37], off
	v_add_u32_e32 v36, 0x830, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v39, v[36:37], off
	v_add_u32_e32 v36, 0xc30, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v36, v[36:37], off
	s_waitcnt vmcnt(0)
	v_perm_b32 v37, v36, v39, s21
	v_perm_b32 v36, v38, v35, s21
	s_nop 1
	v_mfma_f32_16x16x16_bf16 v[16:19], v[32:33], v[36:37], v[16:19]
	v_add_u32_e32 v36, 64, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v35, v[36:37], off
	v_add_u32_e32 v36, 0x440, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v38, v[36:37], off
	v_add_u32_e32 v36, 0x840, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v39, v[36:37], off
	v_add_u32_e32 v36, 0xc40, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v36, v[36:37], off
	s_waitcnt vmcnt(0)
	v_perm_b32 v37, v36, v39, s21
	v_perm_b32 v36, v38, v35, s21
	s_nop 1
	v_mfma_f32_16x16x16_bf16 v[12:15], v[32:33], v[36:37], v[12:15]
	v_add_u32_e32 v36, 0x50, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v35, v[36:37], off
	v_add_u32_e32 v36, 0x450, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v38, v[36:37], off
	v_add_u32_e32 v36, 0x850, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v39, v[36:37], off
	v_add_u32_e32 v36, 0xc50, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v36, v[36:37], off
	s_waitcnt vmcnt(0)
	v_perm_b32 v37, v36, v39, s21
	v_perm_b32 v36, v38, v35, s21
	s_nop 1
	v_mfma_f32_16x16x16_bf16 v[24:27], v[32:33], v[36:37], v[24:27]
	v_add_u32_e32 v36, 0x60, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v35, v[36:37], off
	v_add_u32_e32 v36, 0x460, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v38, v[36:37], off
	v_add_u32_e32 v36, 0x860, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v39, v[36:37], off
	v_add_u32_e32 v36, 0xc60, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v36, v[36:37], off
	s_waitcnt vmcnt(0)
	v_perm_b32 v37, v36, v39, s21
	v_perm_b32 v36, v38, v35, s21
	s_nop 1
	v_mfma_f32_16x16x16_bf16 v[20:23], v[32:33], v[36:37], v[20:23]
	v_add_u32_e32 v36, 0x70, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v38, v[36:37], off
	v_add_u32_e32 v36, 0x470, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	global_load_ushort v39, v[36:37], off
	v_add_u32_e32 v36, 0x870, v34
	v_add_u32_e32 v34, 0xc70, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	v_lshl_add_u64 v[34:35], v[34:35], 1, s[12:13]
	global_load_ushort v36, v[36:37], off
	s_nop 0
	global_load_ushort v34, v[34:35], off
	s_barrier
	s_waitcnt vmcnt(0)
	v_perm_b32 v35, v34, v36, s21
	v_perm_b32 v34, v39, v38, s21
	s_nop 1
	v_mfma_f32_16x16x16_bf16 v[28:31], v[32:33], v[34:35], v[28:31]
	s_cbranch_scc1 .LBB0_6
.LBB0_4:                                ; %.preheader246
                                        ; =>This Inner Loop Header: Depth=1
	v_add_u32_e32 v32, s17, v66
	v_ashrrev_i32_e32 v33, 31, v32
	v_lshl_add_u64 v[36:37], v[32:33], 1, s[10:11]
	global_load_dwordx2 v[32:33], v[36:37], off
	global_load_dwordx2 v[38:39], v[36:37], off offset:32
	global_load_dwordx2 v[70:71], v[36:37], off offset:64
	global_load_dwordx2 v[72:73], v[36:37], off offset:96
	global_load_dwordx2 v[74:75], v[36:37], off offset:128
	s_waitcnt vmcnt(4)
	v_mfma_f32_16x16x16_bf16 v[32:35], v[40:41], v[32:33], 0
	s_waitcnt vmcnt(3)
	v_mfma_f32_16x16x16_bf16 v[32:35], v[42:43], v[38:39], v[32:35]
	global_load_dwordx2 v[38:39], v[36:37], off offset:160
	s_waitcnt vmcnt(3)
	v_mfma_f32_16x16x16_bf16 v[32:35], v[44:45], v[70:71], v[32:35]
	global_load_dwordx2 v[70:71], v[36:37], off offset:192
	s_nop 0
	global_load_dwordx2 v[36:37], v[36:37], off offset:224
	s_waitcnt vmcnt(4)
	v_mfma_f32_16x16x16_bf16 v[32:35], v[46:47], v[72:73], v[32:35]
	s_waitcnt vmcnt(3)
	v_mfma_f32_16x16x16_bf16 v[32:35], v[48:49], v[74:75], v[32:35]
	s_waitcnt vmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[32:35], v[50:51], v[38:39], v[32:35]
	s_waitcnt vmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[32:35], v[52:53], v[70:71], v[32:35]
	s_waitcnt vmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[32:35], v[54:55], v[36:37], v[32:35]
	s_nop 6
	v_mul_f32_e32 v32, 0x3db504f3, v32
	v_mul_f32_e32 v33, 0x3db504f3, v33
	v_mul_f32_e32 v34, 0x3db504f3, v34
	v_mul_f32_e32 v35, 0x3db504f3, v35
	ds_write2_b32 v61, v32, v33 offset1:16
	ds_write2_b32 v61, v34, v35 offset0:32 offset1:48
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB0_3
; %bb.5:                                ;   in Loop: Header=BB0_4 Depth=1
	ds_read_b32 v69, v57
	ds_read_b128 v[70:73], v63
	ds_read_b128 v[74:77], v63 offset:16
	ds_read_b128 v[36:39], v63 offset:32
	ds_read_b128 v[32:35], v63 offset:48
	ds_read_b32 v79, v58
	s_waitcnt lgkmcnt(4)
	v_max3_f32 v78, v69, v70, v71
	v_max3_f32 v78, v78, v72, v73
	s_waitcnt lgkmcnt(3)
	v_max3_f32 v78, v78, v74, v75
	v_max3_f32 v78, v78, v76, v77
	s_waitcnt lgkmcnt(2)
	v_max3_f32 v78, v78, v36, v37
	v_max3_f32 v78, v78, v38, v39
	s_waitcnt lgkmcnt(1)
	v_max3_f32 v78, v78, v32, v33
	v_max3_f32 v78, v78, v34, v35
	v_sub_f32_e32 v70, v70, v78
	v_mul_f32_e32 v70, 0x3fb8aa3b, v70
	v_sub_f32_e32 v69, v69, v78
	v_sub_f32_e32 v71, v71, v78
	v_exp_f32_e32 v70, v70
	v_mul_f32_e32 v69, 0x3fb8aa3b, v69
	v_mul_f32_e32 v71, 0x3fb8aa3b, v71
	v_sub_f32_e32 v72, v72, v78
	v_exp_f32_e32 v69, v69
	v_exp_f32_e32 v71, v71
	v_mul_f32_e32 v72, 0x3fb8aa3b, v72
	v_sub_f32_e32 v73, v73, v78
	v_exp_f32_e32 v72, v72
	v_mul_f32_e32 v73, 0x3fb8aa3b, v73
	v_exp_f32_e32 v73, v73
	v_sub_f32_e32 v74, v74, v78
	v_bfe_u32 v80, v70, 16, 1
	v_mul_f32_e32 v74, 0x3fb8aa3b, v74
	v_add3_u32 v80, v80, v70, s19
	v_or_b32_e32 v81, 0x400000, v70
	v_cmp_u_f32_e64 s[0:1], v70, v70
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v70, v79, v69
	v_bfe_u32 v79, v71, 16, 1
	v_exp_f32_e32 v74, v74
	v_sub_f32_e32 v75, v75, v78
	v_cndmask_b32_e64 v80, v80, v81, s[0:1]
	v_add3_u32 v79, v79, v71, s19
	v_or_b32_e32 v81, 0x400000, v71
	v_cmp_u_f32_e64 s[0:1], v71, v71
	v_add_f32_e32 v70, v71, v70
	v_bfe_u32 v71, v72, 16, 1
	v_mul_f32_e32 v75, 0x3fb8aa3b, v75
	v_cndmask_b32_e64 v79, v79, v81, s[0:1]
	v_add3_u32 v71, v71, v72, s19
	v_or_b32_e32 v81, 0x400000, v72
	v_cmp_u_f32_e64 s[0:1], v72, v72
	v_add_f32_e32 v70, v72, v70
	v_bfe_u32 v72, v73, 16, 1
	v_exp_f32_e32 v75, v75
	v_sub_f32_e32 v76, v76, v78
	v_cndmask_b32_e64 v71, v71, v81, s[0:1]
	v_add3_u32 v72, v72, v73, s19
	v_or_b32_e32 v81, 0x400000, v73
	v_cmp_u_f32_e64 s[0:1], v73, v73
	v_mul_f32_e32 v76, 0x3fb8aa3b, v76
	v_exp_f32_e32 v76, v76
	v_cndmask_b32_e64 v81, v72, v81, s[0:1]
	v_bfe_u32 v72, v74, 16, 1
	v_sub_f32_e32 v77, v77, v78
	v_add_f32_e32 v70, v73, v70
	v_add3_u32 v72, v72, v74, s19
	v_or_b32_e32 v73, 0x400000, v74
	v_cmp_u_f32_e64 s[0:1], v74, v74
	v_mul_f32_e32 v77, 0x3fb8aa3b, v77
	v_exp_f32_e32 v77, v77
	v_cndmask_b32_e64 v72, v72, v73, s[0:1]
	v_bfe_u32 v73, v75, 16, 1
	v_add_f32_e32 v70, v74, v70
	v_add3_u32 v73, v73, v75, s19
	v_or_b32_e32 v74, 0x400000, v75
	v_cmp_u_f32_e64 s[0:1], v75, v75
	v_sub_f32_e32 v36, v36, v78
	v_add_f32_e32 v70, v75, v70
	v_cndmask_b32_e64 v74, v73, v74, s[0:1]
	v_bfe_u32 v73, v76, 16, 1
	v_add3_u32 v73, v73, v76, s19
	v_or_b32_e32 v75, 0x400000, v76
	v_cmp_u_f32_e64 s[0:1], v76, v76
	v_mul_f32_e32 v36, 0x3fb8aa3b, v36
	v_sub_f32_e32 v37, v37, v78
	v_cndmask_b32_e64 v73, v73, v75, s[0:1]
	v_add_f32_e32 v75, v76, v70
	v_bfe_u32 v70, v77, 16, 1
	v_exp_f32_e32 v36, v36
	v_mul_f32_e32 v37, 0x3fb8aa3b, v37
	v_sub_f32_e32 v38, v38, v78
	v_add3_u32 v70, v70, v77, s19
	v_or_b32_e32 v76, 0x400000, v77
	v_cmp_u_f32_e64 s[0:1], v77, v77
	v_exp_f32_e32 v37, v37
	v_mul_f32_e32 v38, 0x3fb8aa3b, v38
	v_sub_f32_e32 v39, v39, v78
	v_cndmask_b32_e64 v70, v70, v76, s[0:1]
	v_exp_f32_e32 v38, v38
	v_mul_f32_e32 v39, 0x3fb8aa3b, v39
	v_sub_f32_e32 v32, v32, v78
	v_perm_b32 v73, v70, v73, s20
	v_perm_b32 v72, v74, v72, s20
	v_perm_b32 v71, v81, v71, s20
	v_perm_b32 v70, v79, v80, s20
	v_exp_f32_e32 v39, v39
	v_mul_f32_e32 v32, 0x3fb8aa3b, v32
	v_sub_f32_e32 v33, v33, v78
	ds_write_b128 v64, v[70:73]
	v_add_f32_e32 v70, v77, v75
	v_bfe_u32 v71, v36, 16, 1
	v_exp_f32_e32 v32, v32
	v_mul_f32_e32 v33, 0x3fb8aa3b, v33
	v_sub_f32_e32 v34, v34, v78
	v_add3_u32 v71, v71, v36, s19
	v_or_b32_e32 v72, 0x400000, v36
	v_cmp_u_f32_e64 s[0:1], v36, v36
	v_add_f32_e32 v36, v36, v70
	v_bfe_u32 v70, v37, 16, 1
	v_exp_f32_e32 v33, v33
	v_mul_f32_e32 v34, 0x3fb8aa3b, v34
	v_cndmask_b32_e64 v71, v71, v72, s[0:1]
	v_add3_u32 v70, v70, v37, s19
	v_or_b32_e32 v72, 0x400000, v37
	v_cmp_u_f32_e64 s[0:1], v37, v37
	v_add_f32_e32 v36, v37, v36
	v_bfe_u32 v37, v38, 16, 1
	v_exp_f32_e32 v34, v34
	v_sub_f32_e32 v35, v35, v78
	v_cndmask_b32_e64 v70, v70, v72, s[0:1]
	v_add3_u32 v37, v37, v38, s19
	v_or_b32_e32 v72, 0x400000, v38
	v_cmp_u_f32_e64 s[0:1], v38, v38
	v_add_f32_e32 v36, v38, v36
	v_bfe_u32 v38, v39, 16, 1
	v_mul_f32_e32 v35, 0x3fb8aa3b, v35
	v_cndmask_b32_e64 v37, v37, v72, s[0:1]
	v_add3_u32 v38, v38, v39, s19
	v_or_b32_e32 v72, 0x400000, v39
	v_cmp_u_f32_e64 s[0:1], v39, v39
	v_add_f32_e32 v36, v39, v36
	v_bfe_u32 v39, v32, 16, 1
	v_exp_f32_e32 v73, v35
	v_cndmask_b32_e64 v38, v38, v72, s[0:1]
	v_add3_u32 v39, v39, v32, s19
	v_or_b32_e32 v72, 0x400000, v32
	v_cmp_u_f32_e64 s[0:1], v32, v32
	v_add_f32_e32 v32, v32, v36
	v_bfe_u32 v36, v33, 16, 1
	v_cndmask_b32_e64 v39, v39, v72, s[0:1]
	v_add3_u32 v36, v36, v33, s19
	v_or_b32_e32 v72, 0x400000, v33
	v_cmp_u_f32_e64 s[0:1], v33, v33
	v_add_f32_e32 v32, v33, v32
	v_bfe_u32 v33, v34, 16, 1
	v_cndmask_b32_e64 v36, v36, v72, s[0:1]
	v_add3_u32 v33, v33, v34, s19
	v_or_b32_e32 v72, 0x400000, v34
	v_cmp_u_f32_e64 s[0:1], v34, v34
	s_nop 1
	v_cndmask_b32_e64 v33, v33, v72, s[0:1]
	v_add_f32_e32 v72, v34, v32
	v_bfe_u32 v32, v73, 16, 1
	v_add3_u32 v32, v32, v73, s19
	v_or_b32_e32 v34, 0x400000, v73
	v_cmp_u_f32_e64 s[0:1], v73, v73
	s_nop 1
	v_cndmask_b32_e64 v32, v32, v34, s[0:1]
	v_perm_b32 v35, v32, v33, s20
	v_perm_b32 v34, v36, v39, s20
	v_perm_b32 v33, v38, v37, s20
	v_perm_b32 v32, v70, v71, s20
	ds_write_b128 v64, v[32:35] offset:16
	v_add_f32_e32 v32, v73, v72
	ds_write_b32 v57, v78
	ds_write_b32 v58, v32
	ds_write_b32 v65, v69
	s_branch .LBB0_3
.LBB0_6:                                ; %.preheader
	v_lshlrev_b32_e32 v41, 2, v59
	ds_read_b32 v37, v41 offset:1600
	v_or_b32_e32 v36, s5, v60
	s_movk_i32 s5, 0x7fff
	s_add_i32 s10, s2, s16
	ds_read_b64 v[34:35], v41 offset:1608
	ds_read_b64 v[32:33], v41 offset:1600
	ds_read_b32 v44, v41 offset:1604
	s_waitcnt lgkmcnt(3)
	v_rcp_f32_e32 v38, v37
	v_cmp_lt_f32_e64 s[0:1], 0, v37
	s_waitcnt lgkmcnt(2)
	v_rcp_f32_e32 v48, v34
	v_cndmask_b32_e64 v40, 0, v38, s[0:1]
	v_mul_f32_e32 v0, v40, v0
	v_bfe_u32 v37, v0, 16, 1
	v_add3_u32 v37, v37, v0, s5
	v_or_b32_e32 v38, 0x400000, v0
	v_cmp_u_f32_e64 s[0:1], v0, v0
	v_mul_f32_e32 v4, v40, v4
	s_nop 0
	v_cndmask_b32_e64 v0, v37, v38, s[0:1]
	v_add_lshl_u32 v37, s10, v59, 13
	v_add_u32_e32 v38, v36, v37
	v_ashrrev_i32_e32 v39, 31, v38
	v_lshl_add_u64 v[38:39], v[38:39], 1, s[8:9]
	global_store_short_d16_hi v[38:39], v0, off
	s_waitcnt lgkmcnt(1)
	v_rcp_f32_e32 v0, v33
	v_cmp_lt_f32_e64 s[0:1], 0, v33
	v_add_u32_e32 v39, 0x644, v41
	ds_read_b32 v41, v41 offset:1612
	v_cndmask_b32_e64 v45, 0, v0, s[0:1]
	v_mul_f32_e32 v0, v45, v1
	v_bfe_u32 v1, v0, 16, 1
	v_add3_u32 v1, v1, v0, s5
	v_or_b32_e32 v33, 0x400000, v0
	v_cmp_u_f32_e64 s[0:1], v0, v0
	v_rcp_f32_e32 v38, v32
	s_nop 0
	v_cndmask_b32_e64 v46, v1, v33, s[0:1]
	ds_read2_b32 v[0:1], v39 offset1:1
	v_add_u32_e32 v33, 0x2000, v37
	v_add_u32_e32 v42, v36, v33
	v_ashrrev_i32_e32 v43, 31, v42
	v_lshl_add_u64 v[42:43], v[42:43], 1, s[8:9]
	s_waitcnt lgkmcnt(0)
	v_rcp_f32_e32 v47, v1
	v_cmp_lt_f32_e64 s[0:1], 0, v1
	global_store_short_d16_hi v[42:43], v46, off
	v_rcp_f32_e32 v39, v0
	v_cndmask_b32_e64 v46, 0, v47, s[0:1]
	v_mul_f32_e32 v1, v46, v2
	v_bfe_u32 v2, v1, 16, 1
	v_rcp_f32_e32 v47, v35
	v_add3_u32 v2, v2, v1, s5
	v_or_b32_e32 v42, 0x400000, v1
	v_cmp_u_f32_e64 s[0:1], v1, v1
	v_add_u32_e32 v1, 0x4000, v37
	s_nop 0
	v_cndmask_b32_e64 v2, v2, v42, s[0:1]
	v_add_u32_e32 v42, v36, v1
	v_ashrrev_i32_e32 v43, 31, v42
	v_cmp_lt_f32_e64 s[0:1], 0, v35
	v_lshl_add_u64 v[42:43], v[42:43], 1, s[8:9]
	global_store_short_d16_hi v[42:43], v2, off
	v_cndmask_b32_e64 v35, 0, v47, s[0:1]
	v_mul_f32_e32 v2, v35, v3
	v_bfe_u32 v3, v2, 16, 1
	v_add3_u32 v3, v3, v2, s5
	v_or_b32_e32 v42, 0x400000, v2
	v_cmp_u_f32_e64 s[0:1], v2, v2
	v_add_u32_e32 v2, 0x6000, v37
	s_nop 0
	v_cndmask_b32_e64 v3, v3, v42, s[0:1]
	v_add_u32_e32 v42, v36, v2
	v_ashrrev_i32_e32 v43, 31, v42
	v_lshl_add_u64 v[42:43], v[42:43], 1, s[8:9]
	global_store_short_d16_hi v[42:43], v3, off
	v_bfe_u32 v42, v4, 16, 1
	v_or_b32_e32 v3, 16, v36
	v_add3_u32 v42, v42, v4, s5
	v_or_b32_e32 v43, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	s_nop 1
	v_cndmask_b32_e64 v4, v42, v43, s[0:1]
	v_add_u32_e32 v42, v3, v37
	v_ashrrev_i32_e32 v43, 31, v42
	v_lshl_add_u64 v[42:43], v[42:43], 1, s[8:9]
	global_store_short_d16_hi v[42:43], v4, off
	v_mul_f32_e32 v4, v45, v5
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v42, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v33
	s_nop 0
	v_cndmask_b32_e64 v42, v5, v42, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	global_store_short_d16_hi v[4:5], v42, off
	v_mul_f32_e32 v4, v46, v6
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v6, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v1
	s_nop 0
	v_cndmask_b32_e64 v6, v5, v6, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	global_store_short_d16_hi v[4:5], v6, off
	v_mul_f32_e32 v4, v35, v7
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v6, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v2
	v_or_b32_e32 v3, 32, v36
	v_cndmask_b32_e64 v6, v5, v6, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	global_store_short_d16_hi v[4:5], v6, off
	v_mul_f32_e32 v4, v40, v8
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v6, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v37
	v_rcp_f32_e32 v7, v44
	v_cndmask_b32_e64 v6, v5, v6, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	global_store_short_d16_hi v[4:5], v6, off
	v_mul_f32_e32 v4, v45, v9
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v6, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v33
	v_rcp_f32_e32 v9, v41
	v_cndmask_b32_e64 v6, v5, v6, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	global_store_short_d16_hi v[4:5], v6, off
	v_mul_f32_e32 v4, v46, v10
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v6, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v1
	s_nop 0
	v_cndmask_b32_e64 v6, v5, v6, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	global_store_short_d16_hi v[4:5], v6, off
	v_mul_f32_e32 v4, v35, v11
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v6, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v2
	v_or_b32_e32 v3, 48, v36
	v_cndmask_b32_e64 v6, v5, v6, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	global_store_short_d16_hi v[4:5], v6, off
	v_mul_f32_e32 v4, v40, v16
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v6, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v37
	s_nop 0
	v_cndmask_b32_e64 v6, v5, v6, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	v_cmp_lt_f32_e64 s[0:1], 0, v44
	global_store_short_d16_hi v[4:5], v6, off
	s_nop 0
	v_cndmask_b32_e64 v6, 0, v7, s[0:1]
	v_mul_f32_e32 v4, v6, v17
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v7, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v33
	s_nop 0
	v_cndmask_b32_e64 v7, v5, v7, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	global_store_short_d16_hi v[4:5], v7, off
	v_mul_f32_e32 v4, v46, v18
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v7, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v1
	s_nop 0
	v_cndmask_b32_e64 v7, v5, v7, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	global_store_short_d16_hi v[4:5], v7, off
	v_mul_f32_e32 v4, v35, v19
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v7, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v2
	v_or_b32_e32 v3, 64, v36
	v_cndmask_b32_e64 v7, v5, v7, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	global_store_short_d16_hi v[4:5], v7, off
	v_mul_f32_e32 v4, v40, v12
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v7, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v37
	s_nop 0
	v_cndmask_b32_e64 v7, v5, v7, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	global_store_short_d16_hi v[4:5], v7, off
	v_mul_f32_e32 v4, v6, v13
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v7, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v33
	s_nop 0
	v_cndmask_b32_e64 v7, v5, v7, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	v_cmp_lt_f32_e64 s[0:1], 0, v34
	global_store_short_d16_hi v[4:5], v7, off
	s_nop 0
	v_cndmask_b32_e64 v7, 0, v48, s[0:1]
	v_mul_f32_e32 v4, v7, v14
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v8, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v1
	s_nop 0
	v_cndmask_b32_e64 v8, v5, v8, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	v_cmp_lt_f32_e64 s[0:1], 0, v41
	global_store_short_d16_hi v[4:5], v8, off
	s_nop 0
	v_cndmask_b32_e64 v8, 0, v9, s[0:1]
	v_mul_f32_e32 v4, v8, v15
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v9, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v2
	v_or_b32_e32 v3, 0x50, v36
	v_cndmask_b32_e64 v9, v5, v9, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	global_store_short_d16_hi v[4:5], v9, off
	v_mul_f32_e32 v4, v40, v24
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v9, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v37
	s_nop 0
	v_cndmask_b32_e64 v9, v5, v9, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	global_store_short_d16_hi v[4:5], v9, off
	v_mul_f32_e32 v4, v6, v25
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v9, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v33
	s_nop 0
	v_cndmask_b32_e64 v9, v5, v9, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	global_store_short_d16_hi v[4:5], v9, off
	v_mul_f32_e32 v4, v7, v26
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v9, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v1
	s_nop 0
	v_cndmask_b32_e64 v9, v5, v9, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	global_store_short_d16_hi v[4:5], v9, off
	v_mul_f32_e32 v4, v8, v27
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v9, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v2
	v_or_b32_e32 v3, 0x60, v36
	v_cndmask_b32_e64 v9, v5, v9, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	global_store_short_d16_hi v[4:5], v9, off
	v_mul_f32_e32 v4, v40, v20
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v9, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v37
	s_nop 0
	v_cndmask_b32_e64 v9, v5, v9, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	global_store_short_d16_hi v[4:5], v9, off
	v_mul_f32_e32 v4, v6, v21
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v6, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v33
	s_nop 0
	v_cndmask_b32_e64 v6, v5, v6, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	global_store_short_d16_hi v[4:5], v6, off
	v_mul_f32_e32 v4, v7, v22
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v6, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v1
	s_nop 0
	v_cndmask_b32_e64 v6, v5, v6, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	global_store_short_d16_hi v[4:5], v6, off
	v_mul_f32_e32 v4, v8, v23
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v6, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v2
	v_or_b32_e32 v3, 0x70, v36
	v_cndmask_b32_e64 v6, v5, v6, s[0:1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	v_cmp_lt_f32_e64 s[0:1], 0, v32
	global_store_short_d16_hi v[4:5], v6, off
	s_nop 0
	v_cndmask_b32_e64 v4, 0, v38, s[0:1]
	v_mul_f32_e32 v4, v4, v28
	v_bfe_u32 v5, v4, 16, 1
	v_add3_u32 v5, v5, v4, s5
	v_or_b32_e32 v6, 0x400000, v4
	v_cmp_u_f32_e64 s[0:1], v4, v4
	v_add_u32_e32 v4, v3, v37
	s_nop 0
	v_cndmask_b32_e64 v6, v5, v6, s[0:1]
	v_cmp_lt_f32_e64 s[0:1], 0, v0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	v_cndmask_b32_e64 v0, 0, v39, s[0:1]
	v_mul_f32_e32 v0, v0, v29
	global_store_short_d16_hi v[4:5], v6, off
	v_bfe_u32 v4, v0, 16, 1
	v_add3_u32 v4, v4, v0, s5
	v_or_b32_e32 v5, 0x400000, v0
	v_cmp_u_f32_e64 s[0:1], v0, v0
	s_nop 1
	v_cndmask_b32_e64 v0, v4, v5, s[0:1]
	v_add_u32_e32 v4, v3, v33
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]
	global_store_short_d16_hi v[4:5], v0, off
	v_mul_f32_e32 v0, v7, v30
	v_bfe_u32 v4, v0, 16, 1
	v_add3_u32 v4, v4, v0, s5
	v_or_b32_e32 v5, 0x400000, v0
	v_cmp_u_f32_e64 s[0:1], v0, v0
	v_add_u32_e32 v0, v3, v1
	v_ashrrev_i32_e32 v1, 31, v0
	v_cndmask_b32_e64 v4, v4, v5, s[0:1]
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[8:9]
	global_store_short_d16_hi v[0:1], v4, off
	v_mul_f32_e32 v0, v8, v31
	v_bfe_u32 v1, v0, 16, 1
	v_add3_u32 v1, v1, v0, s5
	v_or_b32_e32 v4, 0x400000, v0
	v_cmp_u_f32_e64 s[0:1], v0, v0
	v_add_u32_e32 v0, v3, v2
	s_nop 0
	v_cndmask_b32_e64 v4, v1, v4, s[0:1]
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[8:9]
	global_store_short_d16_hi v[0:1], v4, off
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_8
; %bb.7:
	ds_read_b32 v0, v58
	ds_read_b32 v3, v57
	s_mov_b32 s0, 0x800000
	v_mov_b32_e32 v1, 0xc1b17218
	s_waitcnt lgkmcnt(1)
	v_cmp_gt_f32_e32 vcc, s0, v0
	s_lshl_b32 s0, s4, 6
	s_nop 0
	v_cndmask_b32_e64 v2, 0, 32, vcc
	v_ldexp_f32 v2, v0, v2
	v_log_f32_e32 v2, v2
	s_add_i32 s0, s0, s3
	s_ashr_i32 s1, s0, 31
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[0:1], s[0:1], 13
	v_cndmask_b32_e32 v1, 0, v1, vcc
	s_add_u32 s4, s6, s0
	v_fmamk_f32 v1, v2, 0x3f317218, v1
	v_cmp_lt_f32_e32 vcc, 0, v0
	s_addc_u32 s5, s7, s1
	s_lshl_b64 s[0:1], s[2:3], 2
	v_cndmask_b32_e32 v0, 0, v1, vcc
	s_add_u32 s0, s4, s0
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v0, v0, v3
	s_addc_u32 s1, s5, s1
	global_store_dword v56, v0, s[0:1]
.LBB0_8:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z10attend_kerILi128EEv12attn_globalsIXT_EE
		.amdhsa_group_segment_fixed_size 1728
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 248
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 82
		.amdhsa_next_free_sgpr 22
		.amdhsa_accum_offset 84
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z10attend_kerILi128EEv12attn_globalsIXT_EE,"axG",@progbits,_Z10attend_kerILi128EEv12attn_globalsIXT_EE,comdat
.Lfunc_end0:
	.size	_Z10attend_kerILi128EEv12attn_globalsIXT_EE, .Lfunc_end0-_Z10attend_kerILi128EEv12attn_globalsIXT_EE
                                        ; -- End function
	.set _Z10attend_kerILi128EEv12attn_globalsIXT_EE.num_vgpr, 82
	.set _Z10attend_kerILi128EEv12attn_globalsIXT_EE.num_agpr, 0
	.set _Z10attend_kerILi128EEv12attn_globalsIXT_EE.numbered_sgpr, 22
	.set _Z10attend_kerILi128EEv12attn_globalsIXT_EE.num_named_barrier, 0
	.set _Z10attend_kerILi128EEv12attn_globalsIXT_EE.private_seg_size, 0
	.set _Z10attend_kerILi128EEv12attn_globalsIXT_EE.uses_vcc, 1
	.set _Z10attend_kerILi128EEv12attn_globalsIXT_EE.uses_flat_scratch, 0
	.set _Z10attend_kerILi128EEv12attn_globalsIXT_EE.has_dyn_sized_stack, 0
	.set _Z10attend_kerILi128EEv12attn_globalsIXT_EE.has_recursion, 0
	.set _Z10attend_kerILi128EEv12attn_globalsIXT_EE.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6084
; TotalNumSgprs: 28
; NumVgprs: 82
; NumAgprs: 0
; TotalNumVgprs: 82
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 1728 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 10
; NumSGPRsForWavesPerEU: 28
; NumVGPRsForWavesPerEU: 82
; AccumOffset: 84
; Occupancy: 5
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 20
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_7003bf60c5321fc8,@object ; @__hip_cuid_7003bf60c5321fc8
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_7003bf60c5321fc8
__hip_cuid_7003bf60c5321fc8:
	.byte	0                               ; 0x0
	.size	__hip_cuid_7003bf60c5321fc8, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_7003bf60c5321fc8
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           248
        .value_kind:     by_value
    .group_segment_fixed_size: 1728
    .kernarg_segment_align: 8
    .kernarg_segment_size: 248
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z10attend_kerILi128EEv12attn_globalsIXT_EE
    .private_segment_fixed_size: 0
    .sgpr_count:     28
    .sgpr_spill_count: 0
    .symbol:         _Z10attend_kerILi128EEv12attn_globalsIXT_EE.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     82
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
