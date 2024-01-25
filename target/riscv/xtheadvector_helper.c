/*
 * RISC-V XTheadVector Extension Helpers for QEMU.
 *
 * Copyright (c) 2024 Alibaba Group. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2 or later, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "qemu/osdep.h"
#include "qemu/host-utils.h"
#include "qemu/bitops.h"
#include "cpu.h"
#include "exec/memop.h"
#include "exec/exec-all.h"
#include "exec/cpu_ldst.h"
#include "exec/helper-proto.h"
#include "fpu/softfloat.h"
#include "tcg/tcg-gvec-desc.h"
#include "internals.h"
#include "vector_internals.h"
#include <math.h>

/* Different desc encoding need different parse functions */
static inline uint32_t th_nf(uint32_t desc)
{
    return FIELD_EX32(simd_data(desc), VDATA_TH, NF);
}

static inline uint32_t th_mlen(uint32_t desc)
{
    return FIELD_EX32(simd_data(desc), VDATA_TH, MLEN);
}

static inline uint32_t th_vm(uint32_t desc)
{
    return FIELD_EX32(simd_data(desc), VDATA_TH, VM);
}

static inline uint32_t th_lmul(uint32_t desc)
{
    return FIELD_EX32(simd_data(desc), VDATA_TH, LMUL);
}

static uint32_t th_wd(uint32_t desc)
{
    return (simd_data(desc) >> 11) & 0x1;
}

/*
 * Get vector group length in bytes. Its range is [64, 2048].
 *
 * As simd_desc support at most 256, the max vlen is 512 bits.
 * So vlen in bytes is encoded as maxsz.
 *
 * XtheadVector diff from RVV1.0 is that TH not have fractional lmul and emul.
 */
static inline uint32_t th_maxsz(uint32_t desc)
{
    return simd_maxsz(desc) << th_lmul(desc);
}

static inline target_ulong adjust_addr(CPURISCVState *env, target_ulong addr)
{
    return (addr & ~env->cur_pmmask) | env->cur_pmbase;
}

static void probe_pages_th(CPURISCVState *env, target_ulong addr,
                           target_ulong len, uintptr_t ra,
                           MMUAccessType access_type)
{
    probe_pages(env, addr, len, ra, access_type);
}


/* XTheadVector need to clear the tail elements */
#if HOST_BIG_ENDIAN
static void th_clear(void *tail, uint32_t cnt, uint32_t tot)
{
    /*
     * Split the remaining range to two parts.
     * The first part is in the last uint64_t unit.
     * The second part start from the next uint64_t unit.
     */
    int part1 = 0, part2 = tot - cnt;
    if (cnt % 8) {
        part1 = 8 - (cnt % 8);
        part2 = tot - cnt - part1;
        memset(QEMU_ALIGN_PTR_DOWN(tail, 8), 0, part1);
        memset(QEMU_ALIGN_PTR_UP(tail, 8), 0, part2);
    } else {
        memset(tail, 0, part2);
    }
}
#else
static void th_clear(void *tail, uint32_t cnt, uint32_t tot)
{
    memset(tail, 0, tot - cnt);
}
#endif

static void clearb_th(void *vd, uint32_t idx, uint32_t cnt, uint32_t tot)
{
    int8_t *cur = ((int8_t *)vd + H1(idx));
    th_clear(cur, cnt, tot);
}

static void clearh_th(void *vd, uint32_t idx, uint32_t cnt, uint32_t tot)
{
    int16_t *cur = ((int16_t *)vd + H2(idx));
    th_clear(cur, cnt, tot);
}

static void clearl_th(void *vd, uint32_t idx, uint32_t cnt, uint32_t tot)
{
    int32_t *cur = ((int32_t *)vd + H4(idx));
    th_clear(cur, cnt, tot);
}

static void clearq_th(void *vd, uint32_t idx, uint32_t cnt, uint32_t tot)
{
    int64_t *cur = (int64_t *)vd + idx;
    th_clear(cur, cnt, tot);
}

/*
 * XTheadVector has different mask layout.
 *
 * The mask bits for element i are located in
 * bits [MLEN*i+(MLEN-1) : MLEN*i] of the mask register.
 *
 * MLEN = SEW/LMUL
 */
static inline int th_elem_mask(void *v0, int mlen, int index)
{
    int idx = (index * mlen) / 64;
    int pos = (index * mlen) % 64;
    return (((uint64_t *)v0)[idx] >> pos) & 1;
}

static inline void th_set_elem_mask(void *v0, int mlen, int index,
                                    uint8_t value)
{
    int idx = (index * mlen) / 64;
    int pos = (index * mlen) % 64;
    uint64_t old = ((uint64_t *)v0)[idx];
    ((uint64_t *)v0)[idx] = deposit64(old, pos, mlen, value);
}

/* elements operations for load and store */
typedef void th_ldst_elem_fn(CPURISCVState *env, target_ulong addr,
                             uint32_t idx, void *vd, uintptr_t retaddr);

/*
 * GEN_TH_LD_ELEM is almost the copy of to GEN_VEXT_LD_ELEM, except that
 * we add "MTYPE data" to deal with zero/sign-extended.
 *
 * For XTheadVector, mem access width is determined by the instruction,
 * while the reg element size equals SEW, therefore mem access width may
 * not equal reg element size. For example, ldb_d means load 8-bit data
 * and sign-extended to 64-bit vector element.
 * As for RVV1.0, mem access width always equals reg element size.
 *
 * So we need to deal with zero/sign-extended in addition.
 */

#define GEN_TH_LD_ELEM(NAME, MTYPE, ETYPE, H, LDSUF)       \
static void NAME(CPURISCVState *env, abi_ptr addr,         \
                 uint32_t idx, void *vd, uintptr_t retaddr)\
{                                                          \
    MTYPE data;                                            \
    ETYPE *cur = ((ETYPE *)vd + H(idx));                   \
    data = cpu_##LDSUF##_data_ra(env, addr, retaddr);      \
    *cur = data;                                           \
}                                                          \

GEN_TH_LD_ELEM(ldb_b, int8_t,  int8_t,  H1, ldsb)
GEN_TH_LD_ELEM(ldb_h, int8_t,  int16_t, H2, ldsb)
GEN_TH_LD_ELEM(ldb_w, int8_t,  int32_t, H4, ldsb)
GEN_TH_LD_ELEM(ldb_d, int8_t,  int64_t, H8, ldsb)
GEN_TH_LD_ELEM(ldh_h, int16_t, int16_t, H2, ldsw)
GEN_TH_LD_ELEM(ldh_w, int16_t, int32_t, H4, ldsw)
GEN_TH_LD_ELEM(ldh_d, int16_t, int64_t, H8, ldsw)
GEN_TH_LD_ELEM(ldw_w, int32_t, int32_t, H4, ldl)
GEN_TH_LD_ELEM(ldw_d, int32_t, int64_t, H8, ldl)
GEN_TH_LD_ELEM(lde_b, int8_t,  int8_t,  H1, ldsb)
GEN_TH_LD_ELEM(lde_h, int16_t, int16_t, H2, ldsw)
GEN_TH_LD_ELEM(lde_w, int32_t, int32_t, H4, ldl)
GEN_TH_LD_ELEM(lde_d, int64_t, int64_t, H8, ldq)
GEN_TH_LD_ELEM(ldbu_b, uint8_t,  uint8_t,  H1, ldub)
GEN_TH_LD_ELEM(ldbu_h, uint8_t,  uint16_t, H2, ldub)
GEN_TH_LD_ELEM(ldbu_w, uint8_t,  uint32_t, H4, ldub)
GEN_TH_LD_ELEM(ldbu_d, uint8_t,  uint64_t, H8, ldub)
GEN_TH_LD_ELEM(ldhu_h, uint16_t, uint16_t, H2, lduw)
GEN_TH_LD_ELEM(ldhu_w, uint16_t, uint32_t, H4, lduw)
GEN_TH_LD_ELEM(ldhu_d, uint16_t, uint64_t, H8, lduw)
GEN_TH_LD_ELEM(ldwu_w, uint32_t, uint32_t, H4, ldl)
GEN_TH_LD_ELEM(ldwu_d, uint32_t, uint64_t, H8, ldl)

#define GEN_TH_ST_ELEM(NAME, ETYPE, H, STSUF)            \
static void NAME(CPURISCVState *env, abi_ptr addr,         \
                 uint32_t idx, void *vd, uintptr_t retaddr)\
{                                                          \
    ETYPE data = *((ETYPE *)vd + H(idx));                  \
    cpu_##STSUF##_data_ra(env, addr, data, retaddr);       \
}

GEN_TH_ST_ELEM(stb_b, int8_t,  H1, stb)
GEN_TH_ST_ELEM(stb_h, int16_t, H2, stb)
GEN_TH_ST_ELEM(stb_w, int32_t, H4, stb)
GEN_TH_ST_ELEM(stb_d, int64_t, H8, stb)
GEN_TH_ST_ELEM(sth_h, int16_t, H2, stw)
GEN_TH_ST_ELEM(sth_w, int32_t, H4, stw)
GEN_TH_ST_ELEM(sth_d, int64_t, H8, stw)
GEN_TH_ST_ELEM(stw_w, int32_t, H4, stl)
GEN_TH_ST_ELEM(stw_d, int64_t, H8, stl)
GEN_TH_ST_ELEM(ste_b, int8_t,  H1, stb)
GEN_TH_ST_ELEM(ste_h, int16_t, H2, stw)
GEN_TH_ST_ELEM(ste_w, int32_t, H4, stl)
GEN_TH_ST_ELEM(ste_d, int64_t, H8, stq)

/*
 *** stride: access vector element from strided memory
 */
typedef void clear_fn(void *vd, uint32_t idx, uint32_t cnt, uint32_t tot);

/*
 * This function is almost the copy of vext_ldst_stride, except:
 * 1) XTheadVector has different mask layout, using th_elem_mask
 *    to get [MLEN*i] bit
 * 2) XTheadVector using different data encoding, using th_ functions
 *    to parse.
 * 3) XTheadVector keep the masked elements value, while RVV1.0 policy is
 *    determined by vma.
 * 4) XTheadVector clear the tail elements, while RVV1.0 policy is to rather
 *    set all bits 1s or keep it, determined by vta.
 */
static void
th_ldst_stride(void *vd, void *v0, target_ulong base,
               target_ulong stride, CPURISCVState *env,
               uint32_t desc, uint32_t vm,
               th_ldst_elem_fn *ldst_elem, clear_fn *clear_elem,
               uint32_t esz, uint32_t msz, uintptr_t ra)
{
    uint32_t i, k;
    uint32_t nf = th_nf(desc);
    uint32_t mlen = th_mlen(desc);
    uint32_t vlmax = th_maxsz(desc) / esz;

    /* do real access */
    for (i = env->vstart; i < env->vl; i++, env->vstart++) {
        k = 0;
        if (!vm && !th_elem_mask(v0, mlen, i)) {
            continue;
        }
        while (k < nf) {
            target_ulong addr = base + stride * i + k * msz;
            ldst_elem(env, adjust_addr(env, addr), i + k * vlmax, vd, ra);
            k++;
        }
    }
    env->vstart = 0;
    /* clear tail elements */
    /* clear_elem is Null when store */
    if (clear_elem) {
        for (k = 0; k < nf; k++) {
            clear_elem(vd, env->vl + k * vlmax, env->vl * esz, vlmax * esz);
        }
    }
}

/*
 * GEN_TH_LD_STRIDE is similar to GEN_VEXT_LD_STRIDE
 * just change the function args
 */
#define GEN_TH_LD_STRIDE(NAME, MTYPE, ETYPE, LOAD_FN, CLEAR_FN)         \
void HELPER(NAME)(void *vd, void * v0, target_ulong base,               \
                  target_ulong stride, CPURISCVState *env,              \
                  uint32_t desc)                                        \
{                                                                       \
    uint32_t vm = th_vm(desc);                                          \
    th_ldst_stride(vd, v0, base, stride, env, desc, vm, LOAD_FN,        \
                   CLEAR_FN, sizeof(ETYPE), sizeof(MTYPE), GETPC());    \
}

GEN_TH_LD_STRIDE(th_vlsb_v_b,  int8_t,   int8_t,   ldb_b,  clearb_th)
GEN_TH_LD_STRIDE(th_vlsb_v_h,  int8_t,   int16_t,  ldb_h,  clearh_th)
GEN_TH_LD_STRIDE(th_vlsb_v_w,  int8_t,   int32_t,  ldb_w,  clearl_th)
GEN_TH_LD_STRIDE(th_vlsb_v_d,  int8_t,   int64_t,  ldb_d,  clearq_th)
GEN_TH_LD_STRIDE(th_vlsh_v_h,  int16_t,  int16_t,  ldh_h,  clearh_th)
GEN_TH_LD_STRIDE(th_vlsh_v_w,  int16_t,  int32_t,  ldh_w,  clearl_th)
GEN_TH_LD_STRIDE(th_vlsh_v_d,  int16_t,  int64_t,  ldh_d,  clearq_th)
GEN_TH_LD_STRIDE(th_vlsw_v_w,  int32_t,  int32_t,  ldw_w,  clearl_th)
GEN_TH_LD_STRIDE(th_vlsw_v_d,  int32_t,  int64_t,  ldw_d,  clearq_th)
GEN_TH_LD_STRIDE(th_vlse_v_b,  int8_t,   int8_t,   lde_b,  clearb_th)
GEN_TH_LD_STRIDE(th_vlse_v_h,  int16_t,  int16_t,  lde_h,  clearh_th)
GEN_TH_LD_STRIDE(th_vlse_v_w,  int32_t,  int32_t,  lde_w,  clearl_th)
GEN_TH_LD_STRIDE(th_vlse_v_d,  int64_t,  int64_t,  lde_d,  clearq_th)
GEN_TH_LD_STRIDE(th_vlsbu_v_b, uint8_t,  uint8_t,  ldbu_b, clearb_th)
GEN_TH_LD_STRIDE(th_vlsbu_v_h, uint8_t,  uint16_t, ldbu_h, clearh_th)
GEN_TH_LD_STRIDE(th_vlsbu_v_w, uint8_t,  uint32_t, ldbu_w, clearl_th)
GEN_TH_LD_STRIDE(th_vlsbu_v_d, uint8_t,  uint64_t, ldbu_d, clearq_th)
GEN_TH_LD_STRIDE(th_vlshu_v_h, uint16_t, uint16_t, ldhu_h, clearh_th)
GEN_TH_LD_STRIDE(th_vlshu_v_w, uint16_t, uint32_t, ldhu_w, clearl_th)
GEN_TH_LD_STRIDE(th_vlshu_v_d, uint16_t, uint64_t, ldhu_d, clearq_th)
GEN_TH_LD_STRIDE(th_vlswu_v_w, uint32_t, uint32_t, ldwu_w, clearl_th)
GEN_TH_LD_STRIDE(th_vlswu_v_d, uint32_t, uint64_t, ldwu_d, clearq_th)

/*
 * GEN_TH_ST_STRIDE is similar to GEN_VEXT_ST_STRIDE
 * just change the function name and args
 */
#define GEN_TH_ST_STRIDE(NAME, MTYPE, ETYPE, STORE_FN)                  \
void HELPER(NAME)(void *vd, void *v0, target_ulong base,                \
                  target_ulong stride, CPURISCVState *env,              \
                  uint32_t desc)                                        \
{                                                                       \
    uint32_t vm = th_vm(desc);                                          \
    th_ldst_stride(vd, v0, base, stride, env, desc, vm, STORE_FN,       \
                   NULL, sizeof(ETYPE), sizeof(MTYPE), GETPC());        \
}

GEN_TH_ST_STRIDE(th_vssb_v_b, int8_t,  int8_t,  stb_b)
GEN_TH_ST_STRIDE(th_vssb_v_h, int8_t,  int16_t, stb_h)
GEN_TH_ST_STRIDE(th_vssb_v_w, int8_t,  int32_t, stb_w)
GEN_TH_ST_STRIDE(th_vssb_v_d, int8_t,  int64_t, stb_d)
GEN_TH_ST_STRIDE(th_vssh_v_h, int16_t, int16_t, sth_h)
GEN_TH_ST_STRIDE(th_vssh_v_w, int16_t, int32_t, sth_w)
GEN_TH_ST_STRIDE(th_vssh_v_d, int16_t, int64_t, sth_d)
GEN_TH_ST_STRIDE(th_vssw_v_w, int32_t, int32_t, stw_w)
GEN_TH_ST_STRIDE(th_vssw_v_d, int32_t, int64_t, stw_d)
GEN_TH_ST_STRIDE(th_vsse_v_b, int8_t,  int8_t,  ste_b)
GEN_TH_ST_STRIDE(th_vsse_v_h, int16_t, int16_t, ste_h)
GEN_TH_ST_STRIDE(th_vsse_v_w, int32_t, int32_t, ste_w)
GEN_TH_ST_STRIDE(th_vsse_v_d, int64_t, int64_t, ste_d)

/*
 *** unit-stride: access elements stored contiguously in memory
 */

/* unmasked unit-stride load and store operation*/
/*
 * This function is almost the copy of vext_ldst_us, except:
 * 1) different mask layout
 * 2) different data encoding
 * 3) different the tail elements process policy
 */
static void
th_ldst_us(void *vd, target_ulong base, CPURISCVState *env, uint32_t desc,
           th_ldst_elem_fn *ldst_elem, clear_fn *clear_elem,
           uint32_t esz, uint32_t msz, uintptr_t ra)
{
    uint32_t i, k;
    uint32_t nf = th_nf(desc);
    uint32_t vlmax = th_maxsz(desc) / esz;

    /* load bytes from guest memory */
    for (i = env->vstart; i < env->vl; i++, env->vstart++) {
        k = 0;
        while (k < nf) {
            target_ulong addr = base + (i * nf + k) * msz;
            ldst_elem(env, adjust_addr(env, addr), i + k * vlmax, vd, ra);
            k++;
        }
    }
    env->vstart = 0;
    /* clear tail elements */
    if (clear_elem) {
        for (k = 0; k < nf; k++) {
            clear_elem(vd, env->vl + k * vlmax, env->vl * esz, vlmax * esz);
        }
    }
}

/*
 * masked unit-stride load and store operation will be a special case of stride,
 * stride = NF * sizeof (MTYPE)
 */
/* similar to GEN_GEN_VEXT_LD_US, change the function */
#define GEN_TH_LD_US(NAME, MTYPE, ETYPE, LOAD_FN, CLEAR_FN)             \
void HELPER(NAME##_mask)(void *vd, void *v0, target_ulong base,         \
                         CPURISCVState *env, uint32_t desc)             \
{                                                                       \
    uint32_t stride = th_nf(desc) * sizeof(MTYPE);                      \
    th_ldst_stride(vd, v0, base, stride, env, desc, false, LOAD_FN,     \
                   CLEAR_FN, sizeof(ETYPE), sizeof(MTYPE), GETPC());    \
}                                                                       \
                                                                        \
void HELPER(NAME)(void *vd, void *v0, target_ulong base,                \
                  CPURISCVState *env, uint32_t desc)                    \
{                                                                       \
    th_ldst_us(vd, base, env, desc, LOAD_FN, CLEAR_FN,                  \
               sizeof(ETYPE), sizeof(MTYPE), GETPC());                  \
}

GEN_TH_LD_US(th_vlb_v_b,  int8_t,   int8_t,   ldb_b,  clearb_th)
GEN_TH_LD_US(th_vlb_v_h,  int8_t,   int16_t,  ldb_h,  clearh_th)
GEN_TH_LD_US(th_vlb_v_w,  int8_t,   int32_t,  ldb_w,  clearl_th)
GEN_TH_LD_US(th_vlb_v_d,  int8_t,   int64_t,  ldb_d,  clearq_th)
GEN_TH_LD_US(th_vlh_v_h,  int16_t,  int16_t,  ldh_h,  clearh_th)
GEN_TH_LD_US(th_vlh_v_w,  int16_t,  int32_t,  ldh_w,  clearl_th)
GEN_TH_LD_US(th_vlh_v_d,  int16_t,  int64_t,  ldh_d,  clearq_th)
GEN_TH_LD_US(th_vlw_v_w,  int32_t,  int32_t,  ldw_w,  clearl_th)
GEN_TH_LD_US(th_vlw_v_d,  int32_t,  int64_t,  ldw_d,  clearq_th)
GEN_TH_LD_US(th_vle_v_b,  int8_t,   int8_t,   lde_b,  clearb_th)
GEN_TH_LD_US(th_vle_v_h,  int16_t,  int16_t,  lde_h,  clearh_th)
GEN_TH_LD_US(th_vle_v_w,  int32_t,  int32_t,  lde_w,  clearl_th)
GEN_TH_LD_US(th_vle_v_d,  int64_t,  int64_t,  lde_d,  clearq_th)
GEN_TH_LD_US(th_vlbu_v_b, uint8_t,  uint8_t,  ldbu_b, clearb_th)
GEN_TH_LD_US(th_vlbu_v_h, uint8_t,  uint16_t, ldbu_h, clearh_th)
GEN_TH_LD_US(th_vlbu_v_w, uint8_t,  uint32_t, ldbu_w, clearl_th)
GEN_TH_LD_US(th_vlbu_v_d, uint8_t,  uint64_t, ldbu_d, clearq_th)
GEN_TH_LD_US(th_vlhu_v_h, uint16_t, uint16_t, ldhu_h, clearh_th)
GEN_TH_LD_US(th_vlhu_v_w, uint16_t, uint32_t, ldhu_w, clearl_th)
GEN_TH_LD_US(th_vlhu_v_d, uint16_t, uint64_t, ldhu_d, clearq_th)
GEN_TH_LD_US(th_vlwu_v_w, uint32_t, uint32_t, ldwu_w, clearl_th)
GEN_TH_LD_US(th_vlwu_v_d, uint32_t, uint64_t, ldwu_d, clearq_th)

/* similar to GEN_GEN_VEXT_ST_US, change the function */
#define GEN_TH_ST_US(NAME, MTYPE, ETYPE, STORE_FN)                      \
void HELPER(NAME##_mask)(void *vd, void *v0, target_ulong base,         \
                         CPURISCVState *env, uint32_t desc)             \
{                                                                       \
    uint32_t stride = th_nf(desc) * sizeof(MTYPE);                      \
    th_ldst_stride(vd, v0, base, stride, env, desc, false, STORE_FN,    \
                   NULL, sizeof(ETYPE), sizeof(MTYPE), GETPC());        \
}                                                                       \
                                                                        \
void HELPER(NAME)(void *vd, void *v0, target_ulong base,                \
                  CPURISCVState *env, uint32_t desc)                    \
{                                                                       \
    th_ldst_us(vd, base, env, desc, STORE_FN, NULL,                     \
               sizeof(ETYPE), sizeof(MTYPE), GETPC());                  \
}

GEN_TH_ST_US(th_vsb_v_b, int8_t,  int8_t , stb_b)
GEN_TH_ST_US(th_vsb_v_h, int8_t,  int16_t, stb_h)
GEN_TH_ST_US(th_vsb_v_w, int8_t,  int32_t, stb_w)
GEN_TH_ST_US(th_vsb_v_d, int8_t,  int64_t, stb_d)
GEN_TH_ST_US(th_vsh_v_h, int16_t, int16_t, sth_h)
GEN_TH_ST_US(th_vsh_v_w, int16_t, int32_t, sth_w)
GEN_TH_ST_US(th_vsh_v_d, int16_t, int64_t, sth_d)
GEN_TH_ST_US(th_vsw_v_w, int32_t, int32_t, stw_w)
GEN_TH_ST_US(th_vsw_v_d, int32_t, int64_t, stw_d)
GEN_TH_ST_US(th_vse_v_b, int8_t,  int8_t , ste_b)
GEN_TH_ST_US(th_vse_v_h, int16_t, int16_t, ste_h)
GEN_TH_ST_US(th_vse_v_w, int32_t, int32_t, ste_w)
GEN_TH_ST_US(th_vse_v_d, int64_t, int64_t, ste_d)

/*
 *** index: access vector element from indexed memory
 */
typedef target_ulong th_get_index_addr(target_ulong base,
        uint32_t idx, void *vs2);

/* The copy of GEN_VEXT_GET_INDEX_ADDR */
#define GEN_TH_GET_INDEX_ADDR(NAME, ETYPE, H)          \
static target_ulong NAME(target_ulong base,            \
                         uint32_t idx, void *vs2)      \
{                                                      \
    return (base + *((ETYPE *)vs2 + H(idx)));          \
}

GEN_TH_GET_INDEX_ADDR(idx_b, int8_t,  H1)
GEN_TH_GET_INDEX_ADDR(idx_h, int16_t, H2)
GEN_TH_GET_INDEX_ADDR(idx_w, int32_t, H4)
GEN_TH_GET_INDEX_ADDR(idx_d, int64_t, H8)

/*
 * This function is almost the copy of vext_ldst_index, except:
 * 1) different mask layout
 * 2) different data encoding
 * 3) different mask/tail elements process policy
 */
static inline void
th_ldst_index(void *vd, void *v0, target_ulong base,
              void *vs2, CPURISCVState *env, uint32_t desc,
              th_get_index_addr get_index_addr,
              th_ldst_elem_fn *ldst_elem,
              clear_fn *clear_elem,
              uint32_t esz, uint32_t msz, uintptr_t ra)
{
    uint32_t i, k;
    uint32_t nf = th_nf(desc);
    uint32_t vm = th_vm(desc);
    uint32_t mlen = th_mlen(desc);
    uint32_t vlmax = th_maxsz(desc) / esz;

    /* load bytes from guest memory */
    for (i = env->vstart; i < env->vl; i++, env->vstart++) {
        k = 0;
        if (!vm && !th_elem_mask(v0, mlen, i)) {
            continue;
        }
        while (k < nf) {
            abi_ptr addr = get_index_addr(base, i, vs2) + k * msz;
            ldst_elem(env, adjust_addr(env, addr), i + k * vlmax, vd, ra);
            k++;
        }
    }
    env->vstart = 0;
    /* clear tail elements */
    if (clear_elem) {
        for (k = 0; k < nf; k++) {
            clear_elem(vd, env->vl + k * vlmax, env->vl * esz, vlmax * esz);
        }
    }
}

/* Similar to GEN_VEXT_LD_INDEX */
#define GEN_TH_LD_INDEX(NAME, MTYPE, ETYPE, INDEX_FN, LOAD_FN, CLEAR_FN)   \
void HELPER(NAME)(void *vd, void *v0, target_ulong base,                   \
                  void *vs2, CPURISCVState *env, uint32_t desc)            \
{                                                                          \
    th_ldst_index(vd, v0, base, vs2, env, desc, INDEX_FN,                  \
                  LOAD_FN, CLEAR_FN, sizeof(ETYPE), sizeof(MTYPE),         \
                  GETPC());                                                \
}

GEN_TH_LD_INDEX(th_vlxb_v_b,  int8_t,   int8_t,   idx_b, ldb_b,  clearb_th)
GEN_TH_LD_INDEX(th_vlxb_v_h,  int8_t,   int16_t,  idx_h, ldb_h,  clearh_th)
GEN_TH_LD_INDEX(th_vlxb_v_w,  int8_t,   int32_t,  idx_w, ldb_w,  clearl_th)
GEN_TH_LD_INDEX(th_vlxb_v_d,  int8_t,   int64_t,  idx_d, ldb_d,  clearq_th)
GEN_TH_LD_INDEX(th_vlxh_v_h,  int16_t,  int16_t,  idx_h, ldh_h,  clearh_th)
GEN_TH_LD_INDEX(th_vlxh_v_w,  int16_t,  int32_t,  idx_w, ldh_w,  clearl_th)
GEN_TH_LD_INDEX(th_vlxh_v_d,  int16_t,  int64_t,  idx_d, ldh_d,  clearq_th)
GEN_TH_LD_INDEX(th_vlxw_v_w,  int32_t,  int32_t,  idx_w, ldw_w,  clearl_th)
GEN_TH_LD_INDEX(th_vlxw_v_d,  int32_t,  int64_t,  idx_d, ldw_d,  clearq_th)
GEN_TH_LD_INDEX(th_vlxe_v_b,  int8_t,   int8_t,   idx_b, lde_b,  clearb_th)
GEN_TH_LD_INDEX(th_vlxe_v_h,  int16_t,  int16_t,  idx_h, lde_h,  clearh_th)
GEN_TH_LD_INDEX(th_vlxe_v_w,  int32_t,  int32_t,  idx_w, lde_w,  clearl_th)
GEN_TH_LD_INDEX(th_vlxe_v_d,  int64_t,  int64_t,  idx_d, lde_d,  clearq_th)
GEN_TH_LD_INDEX(th_vlxbu_v_b, uint8_t,  uint8_t,  idx_b, ldbu_b, clearb_th)
GEN_TH_LD_INDEX(th_vlxbu_v_h, uint8_t,  uint16_t, idx_h, ldbu_h, clearh_th)
GEN_TH_LD_INDEX(th_vlxbu_v_w, uint8_t,  uint32_t, idx_w, ldbu_w, clearl_th)
GEN_TH_LD_INDEX(th_vlxbu_v_d, uint8_t,  uint64_t, idx_d, ldbu_d, clearq_th)
GEN_TH_LD_INDEX(th_vlxhu_v_h, uint16_t, uint16_t, idx_h, ldhu_h, clearh_th)
GEN_TH_LD_INDEX(th_vlxhu_v_w, uint16_t, uint32_t, idx_w, ldhu_w, clearl_th)
GEN_TH_LD_INDEX(th_vlxhu_v_d, uint16_t, uint64_t, idx_d, ldhu_d, clearq_th)
GEN_TH_LD_INDEX(th_vlxwu_v_w, uint32_t, uint32_t, idx_w, ldwu_w, clearl_th)
GEN_TH_LD_INDEX(th_vlxwu_v_d, uint32_t, uint64_t, idx_d, ldwu_d, clearq_th)

/* Similar to GEN_VEXT_ST_INDEX */
#define GEN_TH_ST_INDEX(NAME, MTYPE, ETYPE, INDEX_FN, STORE_FN)  \
void HELPER(NAME)(void *vd, void *v0, target_ulong base,         \
                  void *vs2, CPURISCVState *env, uint32_t desc)  \
{                                                                \
    th_ldst_index(vd, v0, base, vs2, env, desc, INDEX_FN,        \
                  STORE_FN, NULL, sizeof(ETYPE), sizeof(MTYPE),  \
                  GETPC());                                      \
}

GEN_TH_ST_INDEX(th_vsxb_v_b, int8_t,  int8_t,  idx_b, stb_b)
GEN_TH_ST_INDEX(th_vsxb_v_h, int8_t,  int16_t, idx_h, stb_h)
GEN_TH_ST_INDEX(th_vsxb_v_w, int8_t,  int32_t, idx_w, stb_w)
GEN_TH_ST_INDEX(th_vsxb_v_d, int8_t,  int64_t, idx_d, stb_d)
GEN_TH_ST_INDEX(th_vsxh_v_h, int16_t, int16_t, idx_h, sth_h)
GEN_TH_ST_INDEX(th_vsxh_v_w, int16_t, int32_t, idx_w, sth_w)
GEN_TH_ST_INDEX(th_vsxh_v_d, int16_t, int64_t, idx_d, sth_d)
GEN_TH_ST_INDEX(th_vsxw_v_w, int32_t, int32_t, idx_w, stw_w)
GEN_TH_ST_INDEX(th_vsxw_v_d, int32_t, int64_t, idx_d, stw_d)
GEN_TH_ST_INDEX(th_vsxe_v_b, int8_t,  int8_t,  idx_b, ste_b)
GEN_TH_ST_INDEX(th_vsxe_v_h, int16_t, int16_t, idx_h, ste_h)
GEN_TH_ST_INDEX(th_vsxe_v_w, int32_t, int32_t, idx_w, ste_w)
GEN_TH_ST_INDEX(th_vsxe_v_d, int64_t, int64_t, idx_d, ste_d)

/*
 *** unit-stride fault-only-fisrt load instructions
 */
/*
 * This function is almost the copy of vext_ldff, except:
 * 1) different mask layout
 * 2) different data encoding
 * 3) different mask/tail elements process policy
 */
static inline void
th_ldff(void *vd, void *v0, target_ulong base,
        CPURISCVState *env, uint32_t desc,
        th_ldst_elem_fn *ldst_elem,
        clear_fn *clear_elem,
        uint32_t esz, uint32_t msz, uintptr_t ra)
{
    void *host;
    uint32_t i, k, vl = 0;
    uint32_t mlen = th_mlen(desc);
    uint32_t nf = th_nf(desc);
    uint32_t vm = th_vm(desc);
    uint32_t vlmax = th_maxsz(desc) / esz;
    target_ulong addr, offset, remain;

    /* probe every access*/
    for (i = env->vstart; i < env->vl; i++) {
        if (!vm && !th_elem_mask(v0, mlen, i)) {
            continue;
        }
        addr = adjust_addr(env, base + nf * i * msz);
        if (i == 0) {
            probe_pages_th(env, addr, nf * msz, ra, MMU_DATA_LOAD);
        } else {
            /* if it triggers an exception, no need to check watchpoint */
            remain = nf * msz;
            while (remain > 0) {
                offset = -(addr | TARGET_PAGE_MASK);
                host = tlb_vaddr_to_host(env, addr, MMU_DATA_LOAD,
                                         cpu_mmu_index(env, false));
                if (host) {
#ifdef CONFIG_USER_ONLY
                    if (!page_check_range(addr, offset, PAGE_READ)) {
                        vl = i;
                        goto ProbeSuccess;
                    }
#else
                    probe_pages_th(env, addr, offset, ra, MMU_DATA_LOAD);
#endif
                } else {
                    vl = i;
                    goto ProbeSuccess;
                }
                if (remain <=  offset) {
                    break;
                }
                remain -= offset;
                addr = adjust_addr(env, addr + offset);
            }
        }
    }
ProbeSuccess:
    /* load bytes from guest memory */
    if (vl != 0) {
        env->vl = vl;
    }
    for (i = env->vstart; i < env->vl; i++) {
        k = 0;
        if (!vm && !th_elem_mask(v0, mlen, i)) {
            continue;
        }
        while (k < nf) {
            addr = base + (i * nf + k) * msz;
            ldst_elem(env, adjust_addr(env, addr), i + k * vlmax, vd, ra);
            k++;
        }
    }
    env->vstart = 0;
    /* clear tail elements */
    if (vl != 0) {
        return;
    }
    for (k = 0; k < nf; k++) {
        clear_elem(vd, env->vl + k * vlmax, env->vl * esz, vlmax * esz);
    }
}

#define GEN_TH_LDFF(NAME, MTYPE, ETYPE, LOAD_FN, CLEAR_FN)       \
void HELPER(NAME)(void *vd, void *v0, target_ulong base,         \
                  CPURISCVState *env, uint32_t desc)             \
{                                                                \
    th_ldff(vd, v0, base, env, desc, LOAD_FN, CLEAR_FN,          \
            sizeof(ETYPE), sizeof(MTYPE), GETPC());              \
}

GEN_TH_LDFF(th_vlbff_v_b,  int8_t,   int8_t,   ldb_b,  clearb_th)
GEN_TH_LDFF(th_vlbff_v_h,  int8_t,   int16_t,  ldb_h,  clearh_th)
GEN_TH_LDFF(th_vlbff_v_w,  int8_t,   int32_t,  ldb_w,  clearl_th)
GEN_TH_LDFF(th_vlbff_v_d,  int8_t,   int64_t,  ldb_d,  clearq_th)
GEN_TH_LDFF(th_vlhff_v_h,  int16_t,  int16_t,  ldh_h,  clearh_th)
GEN_TH_LDFF(th_vlhff_v_w,  int16_t,  int32_t,  ldh_w,  clearl_th)
GEN_TH_LDFF(th_vlhff_v_d,  int16_t,  int64_t,  ldh_d,  clearq_th)
GEN_TH_LDFF(th_vlwff_v_w,  int32_t,  int32_t,  ldw_w,  clearl_th)
GEN_TH_LDFF(th_vlwff_v_d,  int32_t,  int64_t,  ldw_d,  clearq_th)
GEN_TH_LDFF(th_vleff_v_b,  int8_t,   int8_t,   lde_b,  clearb_th)
GEN_TH_LDFF(th_vleff_v_h,  int16_t,  int16_t,  lde_h,  clearh_th)
GEN_TH_LDFF(th_vleff_v_w,  int32_t,  int32_t,  lde_w,  clearl_th)
GEN_TH_LDFF(th_vleff_v_d,  int64_t,  int64_t,  lde_d,  clearq_th)
GEN_TH_LDFF(th_vlbuff_v_b, uint8_t,  uint8_t,  ldbu_b, clearb_th)
GEN_TH_LDFF(th_vlbuff_v_h, uint8_t,  uint16_t, ldbu_h, clearh_th)
GEN_TH_LDFF(th_vlbuff_v_w, uint8_t,  uint32_t, ldbu_w, clearl_th)
GEN_TH_LDFF(th_vlbuff_v_d, uint8_t,  uint64_t, ldbu_d, clearq_th)
GEN_TH_LDFF(th_vlhuff_v_h, uint16_t, uint16_t, ldhu_h, clearh_th)
GEN_TH_LDFF(th_vlhuff_v_w, uint16_t, uint32_t, ldhu_w, clearl_th)
GEN_TH_LDFF(th_vlhuff_v_d, uint16_t, uint64_t, ldhu_d, clearq_th)
GEN_TH_LDFF(th_vlwuff_v_w, uint32_t, uint32_t, ldwu_w, clearl_th)
GEN_TH_LDFF(th_vlwuff_v_d, uint32_t, uint64_t, ldwu_d, clearq_th)

/*
 *** Vector AMO Operations (Zvamo)
 */
typedef void th_amo_noatomic_fn(void *vs3, target_ulong addr,
                                uint32_t wd, uint32_t idx, CPURISCVState *env,
                                uintptr_t retaddr);

#define TH_SWAP(N, M) (M)
#define TH_XOR(N, M)  (N ^ M)
#define TH_OR(N, M)   (N | M)
#define TH_AND(N, M)  (N & M)
#define TH_ADD(N, M)  (N + M)

#define GEN_TH_AMO_NOATOMIC_OP(NAME, ESZ, MSZ, H, DO_OP, SUF)   \
static void                                                     \
th_##NAME##_noatomic_op(void *vs3, target_ulong addr,           \
                        uint32_t wd, uint32_t idx,              \
                        CPURISCVState *env, uintptr_t retaddr)  \
{                                                               \
    typedef int##ESZ##_t ETYPE;                                 \
    typedef int##MSZ##_t MTYPE;                                 \
    typedef uint##MSZ##_t UMTYPE __attribute__((unused));       \
    ETYPE *pe3 = (ETYPE *)vs3 + H(idx);                         \
    MTYPE  a = cpu_ld##SUF##_data(env, addr), b = *pe3;         \
                                                                \
    cpu_st##SUF##_data(env, addr, DO_OP(a, b));                 \
    if (wd) {                                                   \
        *pe3 = a;                                               \
    }                                                           \
}

#define TH_MAX(N, M)  ((N) >= (M) ? (N) : (M))
#define TH_MIN(N, M)  ((N) >= (M) ? (M) : (N))
#define TH_MAXU(N, M) TH_MAX((UMTYPE)N, (UMTYPE)M)
#define TH_MINU(N, M) TH_MIN((UMTYPE)N, (UMTYPE)M)

GEN_TH_AMO_NOATOMIC_OP(th_vamoswapw_v_w, 32, 32, H4, TH_SWAP, l)
GEN_TH_AMO_NOATOMIC_OP(th_vamoaddw_v_w,  32, 32, H4, TH_ADD,  l)
GEN_TH_AMO_NOATOMIC_OP(th_vamoxorw_v_w,  32, 32, H4, TH_XOR,  l)
GEN_TH_AMO_NOATOMIC_OP(th_vamoandw_v_w,  32, 32, H4, TH_AND,  l)
GEN_TH_AMO_NOATOMIC_OP(th_vamoorw_v_w,   32, 32, H4, TH_OR,   l)
GEN_TH_AMO_NOATOMIC_OP(th_vamominw_v_w,  32, 32, H4, TH_MIN,  l)
GEN_TH_AMO_NOATOMIC_OP(th_vamomaxw_v_w,  32, 32, H4, TH_MAX,  l)
GEN_TH_AMO_NOATOMIC_OP(th_vamominuw_v_w, 32, 32, H4, TH_MINU, l)
GEN_TH_AMO_NOATOMIC_OP(th_vamomaxuw_v_w, 32, 32, H4, TH_MAXU, l)
GEN_TH_AMO_NOATOMIC_OP(th_vamoswapw_v_d, 64, 32, H8, TH_SWAP, l)
GEN_TH_AMO_NOATOMIC_OP(th_vamoswapd_v_d, 64, 64, H8, TH_SWAP, q)
GEN_TH_AMO_NOATOMIC_OP(th_vamoaddw_v_d,  64, 32, H8, TH_ADD,  l)
GEN_TH_AMO_NOATOMIC_OP(th_vamoaddd_v_d,  64, 64, H8, TH_ADD,  q)
GEN_TH_AMO_NOATOMIC_OP(th_vamoxorw_v_d,  64, 32, H8, TH_XOR,  l)
GEN_TH_AMO_NOATOMIC_OP(th_vamoxord_v_d,  64, 64, H8, TH_XOR,  q)
GEN_TH_AMO_NOATOMIC_OP(th_vamoandw_v_d,  64, 32, H8, TH_AND,  l)
GEN_TH_AMO_NOATOMIC_OP(th_vamoandd_v_d,  64, 64, H8, TH_AND,  q)
GEN_TH_AMO_NOATOMIC_OP(th_vamoorw_v_d,   64, 32, H8, TH_OR,   l)
GEN_TH_AMO_NOATOMIC_OP(th_vamoord_v_d,   64, 64, H8, TH_OR,   q)
GEN_TH_AMO_NOATOMIC_OP(th_vamominw_v_d,  64, 32, H8, TH_MIN,  l)
GEN_TH_AMO_NOATOMIC_OP(th_vamomind_v_d,  64, 64, H8, TH_MIN,  q)
GEN_TH_AMO_NOATOMIC_OP(th_vamomaxw_v_d,  64, 32, H8, TH_MAX,  l)
GEN_TH_AMO_NOATOMIC_OP(th_vamomaxd_v_d,  64, 64, H8, TH_MAX,  q)
GEN_TH_AMO_NOATOMIC_OP(th_vamominuw_v_d, 64, 32, H8, TH_MINU, l)
GEN_TH_AMO_NOATOMIC_OP(th_vamominud_v_d, 64, 64, H8, TH_MINU, q)
GEN_TH_AMO_NOATOMIC_OP(th_vamomaxuw_v_d, 64, 32, H8, TH_MAXU, l)
GEN_TH_AMO_NOATOMIC_OP(th_vamomaxud_v_d, 64, 64, H8, TH_MAXU, q)

static inline void
th_amo_noatomic(void *vs3, void *v0, target_ulong base,
                void *vs2, CPURISCVState *env, uint32_t desc,
                th_get_index_addr get_index_addr,
                th_amo_noatomic_fn * noatomic_op,
                clear_fn * clear_elem,
                uint32_t esz, uint32_t msz, uintptr_t ra)
{
    uint32_t i;
    target_long addr;
    uint32_t wd = th_wd(desc);
    uint32_t vm = th_vm(desc);
    uint32_t mlen = th_mlen(desc);
    uint32_t vlmax = th_maxsz(desc) / esz;
    uint32_t vl = env->vl;

    for (i = env->vstart; i < vl; i++, env->vstart++) {
        if (!vm && !th_elem_mask(v0, mlen, i)) {
            continue;
        }
        addr = get_index_addr(base, i, vs2);
        noatomic_op(vs3, adjust_addr(env, addr), wd, i, env, ra);
    }
    env->vstart = 0;
    clear_elem(vs3, env->vl, env->vl * esz, vlmax * esz);
}

#define GEN_TH_AMO(NAME, MTYPE, ETYPE, INDEX_FN, CLEAR_FN)      \
void HELPER(NAME)(void *vs3, void *v0, target_ulong base,       \
                  void *vs2, CPURISCVState *env, uint32_t desc) \
{                                                               \
    th_amo_noatomic(vs3, v0, base, vs2, env, desc,              \
                    INDEX_FN, th_##NAME##_noatomic_op,          \
                    CLEAR_FN, sizeof(ETYPE), sizeof(MTYPE),     \
                    GETPC());                                   \
}

GEN_TH_AMO(th_vamoswapw_v_d, int32_t,  int64_t,  idx_d, clearq_th)
GEN_TH_AMO(th_vamoswapd_v_d, int64_t,  int64_t,  idx_d, clearq_th)
GEN_TH_AMO(th_vamoaddw_v_d,  int32_t,  int64_t,  idx_d, clearq_th)
GEN_TH_AMO(th_vamoaddd_v_d,  int64_t,  int64_t,  idx_d, clearq_th)
GEN_TH_AMO(th_vamoxorw_v_d,  int32_t,  int64_t,  idx_d, clearq_th)
GEN_TH_AMO(th_vamoxord_v_d,  int64_t,  int64_t,  idx_d, clearq_th)
GEN_TH_AMO(th_vamoandw_v_d,  int32_t,  int64_t,  idx_d, clearq_th)
GEN_TH_AMO(th_vamoandd_v_d,  int64_t,  int64_t,  idx_d, clearq_th)
GEN_TH_AMO(th_vamoorw_v_d,   int32_t,  int64_t,  idx_d, clearq_th)
GEN_TH_AMO(th_vamoord_v_d,   int64_t,  int64_t,  idx_d, clearq_th)
GEN_TH_AMO(th_vamominw_v_d,  int32_t,  int64_t,  idx_d, clearq_th)
GEN_TH_AMO(th_vamomind_v_d,  int64_t,  int64_t,  idx_d, clearq_th)
GEN_TH_AMO(th_vamomaxw_v_d,  int32_t,  int64_t,  idx_d, clearq_th)
GEN_TH_AMO(th_vamomaxd_v_d,  int64_t,  int64_t,  idx_d, clearq_th)
GEN_TH_AMO(th_vamominuw_v_d, uint32_t, uint64_t, idx_d, clearq_th)
GEN_TH_AMO(th_vamominud_v_d, uint64_t, uint64_t, idx_d, clearq_th)
GEN_TH_AMO(th_vamomaxuw_v_d, uint32_t, uint64_t, idx_d, clearq_th)
GEN_TH_AMO(th_vamomaxud_v_d, uint64_t, uint64_t, idx_d, clearq_th)
GEN_TH_AMO(th_vamoswapw_v_w, int32_t,  int32_t,  idx_w, clearl_th)
GEN_TH_AMO(th_vamoaddw_v_w,  int32_t,  int32_t,  idx_w, clearl_th)
GEN_TH_AMO(th_vamoxorw_v_w,  int32_t,  int32_t,  idx_w, clearl_th)
GEN_TH_AMO(th_vamoandw_v_w,  int32_t,  int32_t,  idx_w, clearl_th)
GEN_TH_AMO(th_vamoorw_v_w,   int32_t,  int32_t,  idx_w, clearl_th)
GEN_TH_AMO(th_vamominw_v_w,  int32_t,  int32_t,  idx_w, clearl_th)
GEN_TH_AMO(th_vamomaxw_v_w,  int32_t,  int32_t,  idx_w, clearl_th)
GEN_TH_AMO(th_vamominuw_v_w, uint32_t, uint32_t, idx_w, clearl_th)
GEN_TH_AMO(th_vamomaxuw_v_w, uint32_t, uint32_t, idx_w, clearl_th)

/*
 *** Vector Integer Arithmetic Instructions
 */
/* redefine macro to decouple */
/* expand macro args before macro */
#define THCALL(macro, ...)  macro(__VA_ARGS__)

/* (TD, T1, T2, TX1, TX2) */
#define TH_OP_SSS_B int8_t, int8_t, int8_t, int8_t, int8_t
#define TH_OP_SSS_H int16_t, int16_t, int16_t, int16_t, int16_t
#define TH_OP_SSS_W int32_t, int32_t, int32_t, int32_t, int32_t
#define TH_OP_SSS_D int64_t, int64_t, int64_t, int64_t, int64_t

#define TH_OP_UUU_B uint8_t, uint8_t, uint8_t, uint8_t, uint8_t
#define TH_OP_UUU_H uint16_t, uint16_t, uint16_t, uint16_t, uint16_t
#define TH_OP_UUU_W uint32_t, uint32_t, uint32_t, uint32_t, uint32_t
#define TH_OP_UUU_D uint64_t, uint64_t, uint64_t, uint64_t, uint64_t

#define TH_OP_SUS_B int8_t, uint8_t, int8_t, uint8_t, int8_t
#define TH_OP_SUS_H int16_t, uint16_t, int16_t, uint16_t, int16_t
#define TH_OP_SUS_W int32_t, uint32_t, int32_t, uint32_t, int32_t
#define TH_OP_SUS_D int64_t, uint64_t, int64_t, uint64_t, int64_t

/* operation of two vector elements */
#define opivv2_fn_th opivv2_fn

#define TH_OPIVV2(NAME, TD, T1, T2, TX1, TX2, HD, HS1, HS2, OP) \
        OPIVV2(NAME, TD, T1, T2, TX1, TX2, HD, HS1, HS2, OP)

#define TH_SUB(N, M) (N - M)
#define TH_RSUB(N, M) (M - N)

THCALL(TH_OPIVV2, th_vadd_vv_b, TH_OP_SSS_B, H1, H1, H1, TH_ADD)
THCALL(TH_OPIVV2, th_vadd_vv_h, TH_OP_SSS_H, H2, H2, H2, TH_ADD)
THCALL(TH_OPIVV2, th_vadd_vv_w, TH_OP_SSS_W, H4, H4, H4, TH_ADD)
THCALL(TH_OPIVV2, th_vadd_vv_d, TH_OP_SSS_D, H8, H8, H8, TH_ADD)
THCALL(TH_OPIVV2, th_vsub_vv_b, TH_OP_SSS_B, H1, H1, H1, TH_SUB)
THCALL(TH_OPIVV2, th_vsub_vv_h, TH_OP_SSS_H, H2, H2, H2, TH_SUB)
THCALL(TH_OPIVV2, th_vsub_vv_w, TH_OP_SSS_W, H4, H4, H4, TH_SUB)
THCALL(TH_OPIVV2, th_vsub_vv_d, TH_OP_SSS_D, H8, H8, H8, TH_SUB)

/*
 * This function is almost the copy of do_vext_vv, except:
 * 1) XTheadVector has different mask layout, using th_elem_mask
 *    to get [MLEN*i] bit
 * 2) XTheadVector using different data encoding, using th_ functions
 *    to parse.
 * 3) XTheadVector keep the masked elements value, while RVV1.0 policy is
 *    determined by vma.
 * 4) XTheadVector clear the tail elements, while RVV1.0 policy is to rather
 *    set all bits 1s or keep it, determined by vta.
 */
static void do_vext_vv_th(void *vd, void *v0, void *vs1, void *vs2,
                          CPURISCVState *env, uint32_t desc,
                          uint32_t esz, uint32_t dsz,
                          opivv2_fn_th *fn, clear_fn *clearfn)
{
    uint32_t vlmax = th_maxsz(desc) / esz;
    uint32_t mlen = th_mlen(desc);
    uint32_t vm = th_vm(desc);
    uint32_t vl = env->vl;
    uint32_t i;

    for (i = env->vstart; i < vl; i++) {
        if (!vm && !th_elem_mask(v0, mlen, i)) {
            continue;
        }
        fn(vd, vs1, vs2, i);
    }
    env->vstart = 0;
    clearfn(vd, vl, vl * dsz,  vlmax * dsz);
}

/* generate the helpers for OPIVV */
#define GEN_TH_VV(NAME, ESZ, DSZ, CLEAR_FN)               \
void HELPER(NAME)(void *vd, void *v0, void *vs1,          \
                  void *vs2, CPURISCVState *env,          \
                  uint32_t desc)                          \
{                                                         \
    do_vext_vv_th(vd, v0, vs1, vs2, env, desc, ESZ, DSZ,  \
                  do_##NAME, CLEAR_FN);                   \
}

GEN_TH_VV(th_vadd_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vadd_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vadd_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vadd_vv_d, 8, 8, clearq_th)
GEN_TH_VV(th_vsub_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vsub_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vsub_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vsub_vv_d, 8, 8, clearq_th)

#define opivx2_fn_th opivx2_fn

/*
 * (T1)s1 gives the real operator type.
 * (TX1)(T1)s1 expands the operator type of widen or narrow operations.
 */
#define TH_OPIVX2(NAME, TD, T1, T2, TX1, TX2, HD, HS2, OP)    \
        OPIVX2(NAME, TD, T1, T2, TX1, TX2, HD, HS2, OP)

THCALL(TH_OPIVX2, th_vadd_vx_b, TH_OP_SSS_B, H1, H1, TH_ADD)
THCALL(TH_OPIVX2, th_vadd_vx_h, TH_OP_SSS_H, H2, H2, TH_ADD)
THCALL(TH_OPIVX2, th_vadd_vx_w, TH_OP_SSS_W, H4, H4, TH_ADD)
THCALL(TH_OPIVX2, th_vadd_vx_d, TH_OP_SSS_D, H8, H8, TH_ADD)
THCALL(TH_OPIVX2, th_vsub_vx_b, TH_OP_SSS_B, H1, H1, TH_SUB)
THCALL(TH_OPIVX2, th_vsub_vx_h, TH_OP_SSS_H, H2, H2, TH_SUB)
THCALL(TH_OPIVX2, th_vsub_vx_w, TH_OP_SSS_W, H4, H4, TH_SUB)
THCALL(TH_OPIVX2, th_vsub_vx_d, TH_OP_SSS_D, H8, H8, TH_SUB)
THCALL(TH_OPIVX2, th_vrsub_vx_b, TH_OP_SSS_B, H1, H1, TH_RSUB)
THCALL(TH_OPIVX2, th_vrsub_vx_h, TH_OP_SSS_H, H2, H2, TH_RSUB)
THCALL(TH_OPIVX2, th_vrsub_vx_w, TH_OP_SSS_W, H4, H4, TH_RSUB)
THCALL(TH_OPIVX2, th_vrsub_vx_d, TH_OP_SSS_D, H8, H8, TH_RSUB)

/*
 * This function is almost the copy of do_vext_vx, except:
 * 1) different mask layout
 * 2) different data encoding
 * 3) different mask/tail elements process policy
 */
static void do_vext_vx_th(void *vd, void *v0, target_long s1, void *vs2,
                          CPURISCVState *env, uint32_t desc,
                          uint32_t esz, uint32_t dsz,
                          opivx2_fn_th fn, clear_fn *clearfn)
{
    uint32_t vlmax = th_maxsz(desc) / esz;
    uint32_t mlen = th_mlen(desc);
    uint32_t vm = th_vm(desc);
    uint32_t vl = env->vl;
    uint32_t i;

    for (i = env->vstart; i < vl; i++) {
        if (!vm && !th_elem_mask(v0, mlen, i)) {
            continue;
        }
        fn(vd, s1, vs2, i);
    }
    env->vstart = 0;
    clearfn(vd, vl, vl * dsz,  vlmax * dsz);
}

/* generate the helpers for OPIVX */
#define GEN_TH_VX(NAME, ESZ, DSZ, CLEAR_FN)               \
void HELPER(NAME)(void *vd, void *v0, target_ulong s1,    \
                  void *vs2, CPURISCVState *env,          \
                  uint32_t desc)                          \
{                                                         \
    do_vext_vx_th(vd, v0, s1, vs2, env, desc, ESZ, DSZ,   \
                  do_##NAME, CLEAR_FN);                   \
}

GEN_TH_VX(th_vadd_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vadd_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vadd_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vadd_vx_d, 8, 8, clearq_th)
GEN_TH_VX(th_vsub_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vsub_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vsub_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vsub_vx_d, 8, 8, clearq_th)
GEN_TH_VX(th_vrsub_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vrsub_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vrsub_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vrsub_vx_d, 8, 8, clearq_th)

/* Vector Widening Integer Add/Subtract */

#define TH_WOP_UUU_B uint16_t, uint8_t, uint8_t, uint16_t, uint16_t
#define TH_WOP_UUU_H uint32_t, uint16_t, uint16_t, uint32_t, uint32_t
#define TH_WOP_UUU_W uint64_t, uint32_t, uint32_t, uint64_t, uint64_t
#define TH_WOP_SSS_B int16_t, int8_t, int8_t, int16_t, int16_t
#define TH_WOP_SSS_H int32_t, int16_t, int16_t, int32_t, int32_t
#define TH_WOP_SSS_W int64_t, int32_t, int32_t, int64_t, int64_t
#define TH_WOP_WUUU_B  uint16_t, uint8_t, uint16_t, uint16_t, uint16_t
#define TH_WOP_WUUU_H  uint32_t, uint16_t, uint32_t, uint32_t, uint32_t
#define TH_WOP_WUUU_W  uint64_t, uint32_t, uint64_t, uint64_t, uint64_t
#define TH_WOP_WSSS_B  int16_t, int8_t, int16_t, int16_t, int16_t
#define TH_WOP_WSSS_H  int32_t, int16_t, int32_t, int32_t, int32_t
#define TH_WOP_WSSS_W  int64_t, int32_t, int64_t, int64_t, int64_t
THCALL(TH_OPIVV2, th_vwaddu_vv_b, TH_WOP_UUU_B, H2, H1, H1, TH_ADD)
THCALL(TH_OPIVV2, th_vwaddu_vv_h, TH_WOP_UUU_H, H4, H2, H2, TH_ADD)
THCALL(TH_OPIVV2, th_vwaddu_vv_w, TH_WOP_UUU_W, H8, H4, H4, TH_ADD)
THCALL(TH_OPIVV2, th_vwsubu_vv_b, TH_WOP_UUU_B, H2, H1, H1, TH_SUB)
THCALL(TH_OPIVV2, th_vwsubu_vv_h, TH_WOP_UUU_H, H4, H2, H2, TH_SUB)
THCALL(TH_OPIVV2, th_vwsubu_vv_w, TH_WOP_UUU_W, H8, H4, H4, TH_SUB)
THCALL(TH_OPIVV2, th_vwadd_vv_b, TH_WOP_SSS_B, H2, H1, H1, TH_ADD)
THCALL(TH_OPIVV2, th_vwadd_vv_h, TH_WOP_SSS_H, H4, H2, H2, TH_ADD)
THCALL(TH_OPIVV2, th_vwadd_vv_w, TH_WOP_SSS_W, H8, H4, H4, TH_ADD)
THCALL(TH_OPIVV2, th_vwsub_vv_b, TH_WOP_SSS_B, H2, H1, H1, TH_SUB)
THCALL(TH_OPIVV2, th_vwsub_vv_h, TH_WOP_SSS_H, H4, H2, H2, TH_SUB)
THCALL(TH_OPIVV2, th_vwsub_vv_w, TH_WOP_SSS_W, H8, H4, H4, TH_SUB)
THCALL(TH_OPIVV2, th_vwaddu_wv_b, TH_WOP_WUUU_B, H2, H1, H1, TH_ADD)
THCALL(TH_OPIVV2, th_vwaddu_wv_h, TH_WOP_WUUU_H, H4, H2, H2, TH_ADD)
THCALL(TH_OPIVV2, th_vwaddu_wv_w, TH_WOP_WUUU_W, H8, H4, H4, TH_ADD)
THCALL(TH_OPIVV2, th_vwsubu_wv_b, TH_WOP_WUUU_B, H2, H1, H1, TH_SUB)
THCALL(TH_OPIVV2, th_vwsubu_wv_h, TH_WOP_WUUU_H, H4, H2, H2, TH_SUB)
THCALL(TH_OPIVV2, th_vwsubu_wv_w, TH_WOP_WUUU_W, H8, H4, H4, TH_SUB)
THCALL(TH_OPIVV2, th_vwadd_wv_b, TH_WOP_WSSS_B, H2, H1, H1, TH_ADD)
THCALL(TH_OPIVV2, th_vwadd_wv_h, TH_WOP_WSSS_H, H4, H2, H2, TH_ADD)
THCALL(TH_OPIVV2, th_vwadd_wv_w, TH_WOP_WSSS_W, H8, H4, H4, TH_ADD)
THCALL(TH_OPIVV2, th_vwsub_wv_b, TH_WOP_WSSS_B, H2, H1, H1, TH_SUB)
THCALL(TH_OPIVV2, th_vwsub_wv_h, TH_WOP_WSSS_H, H4, H2, H2, TH_SUB)
THCALL(TH_OPIVV2, th_vwsub_wv_w, TH_WOP_WSSS_W, H8, H4, H4, TH_SUB)
GEN_TH_VV(th_vwaddu_vv_b, 1, 2, clearh_th)
GEN_TH_VV(th_vwaddu_vv_h, 2, 4, clearl_th)
GEN_TH_VV(th_vwaddu_vv_w, 4, 8, clearq_th)
GEN_TH_VV(th_vwsubu_vv_b, 1, 2, clearh_th)
GEN_TH_VV(th_vwsubu_vv_h, 2, 4, clearl_th)
GEN_TH_VV(th_vwsubu_vv_w, 4, 8, clearq_th)
GEN_TH_VV(th_vwadd_vv_b, 1, 2, clearh_th)
GEN_TH_VV(th_vwadd_vv_h, 2, 4, clearl_th)
GEN_TH_VV(th_vwadd_vv_w, 4, 8, clearq_th)
GEN_TH_VV(th_vwsub_vv_b, 1, 2, clearh_th)
GEN_TH_VV(th_vwsub_vv_h, 2, 4, clearl_th)
GEN_TH_VV(th_vwsub_vv_w, 4, 8, clearq_th)
GEN_TH_VV(th_vwaddu_wv_b, 1, 2, clearh_th)
GEN_TH_VV(th_vwaddu_wv_h, 2, 4, clearl_th)
GEN_TH_VV(th_vwaddu_wv_w, 4, 8, clearq_th)
GEN_TH_VV(th_vwsubu_wv_b, 1, 2, clearh_th)
GEN_TH_VV(th_vwsubu_wv_h, 2, 4, clearl_th)
GEN_TH_VV(th_vwsubu_wv_w, 4, 8, clearq_th)
GEN_TH_VV(th_vwadd_wv_b, 1, 2, clearh_th)
GEN_TH_VV(th_vwadd_wv_h, 2, 4, clearl_th)
GEN_TH_VV(th_vwadd_wv_w, 4, 8, clearq_th)
GEN_TH_VV(th_vwsub_wv_b, 1, 2, clearh_th)
GEN_TH_VV(th_vwsub_wv_h, 2, 4, clearl_th)
GEN_TH_VV(th_vwsub_wv_w, 4, 8, clearq_th)

THCALL(TH_OPIVX2, th_vwaddu_vx_b, TH_WOP_UUU_B, H2, H1, TH_ADD)
THCALL(TH_OPIVX2, th_vwaddu_vx_h, TH_WOP_UUU_H, H4, H2, TH_ADD)
THCALL(TH_OPIVX2, th_vwaddu_vx_w, TH_WOP_UUU_W, H8, H4, TH_ADD)
THCALL(TH_OPIVX2, th_vwsubu_vx_b, TH_WOP_UUU_B, H2, H1, TH_SUB)
THCALL(TH_OPIVX2, th_vwsubu_vx_h, TH_WOP_UUU_H, H4, H2, TH_SUB)
THCALL(TH_OPIVX2, th_vwsubu_vx_w, TH_WOP_UUU_W, H8, H4, TH_SUB)
THCALL(TH_OPIVX2, th_vwadd_vx_b, TH_WOP_SSS_B, H2, H1, TH_ADD)
THCALL(TH_OPIVX2, th_vwadd_vx_h, TH_WOP_SSS_H, H4, H2, TH_ADD)
THCALL(TH_OPIVX2, th_vwadd_vx_w, TH_WOP_SSS_W, H8, H4, TH_ADD)
THCALL(TH_OPIVX2, th_vwsub_vx_b, TH_WOP_SSS_B, H2, H1, TH_SUB)
THCALL(TH_OPIVX2, th_vwsub_vx_h, TH_WOP_SSS_H, H4, H2, TH_SUB)
THCALL(TH_OPIVX2, th_vwsub_vx_w, TH_WOP_SSS_W, H8, H4, TH_SUB)
THCALL(TH_OPIVX2, th_vwaddu_wx_b, TH_WOP_WUUU_B, H2, H1, TH_ADD)
THCALL(TH_OPIVX2, th_vwaddu_wx_h, TH_WOP_WUUU_H, H4, H2, TH_ADD)
THCALL(TH_OPIVX2, th_vwaddu_wx_w, TH_WOP_WUUU_W, H8, H4, TH_ADD)
THCALL(TH_OPIVX2, th_vwsubu_wx_b, TH_WOP_WUUU_B, H2, H1, TH_SUB)
THCALL(TH_OPIVX2, th_vwsubu_wx_h, TH_WOP_WUUU_H, H4, H2, TH_SUB)
THCALL(TH_OPIVX2, th_vwsubu_wx_w, TH_WOP_WUUU_W, H8, H4, TH_SUB)
THCALL(TH_OPIVX2, th_vwadd_wx_b, TH_WOP_WSSS_B, H2, H1, TH_ADD)
THCALL(TH_OPIVX2, th_vwadd_wx_h, TH_WOP_WSSS_H, H4, H2, TH_ADD)
THCALL(TH_OPIVX2, th_vwadd_wx_w, TH_WOP_WSSS_W, H8, H4, TH_ADD)
THCALL(TH_OPIVX2, th_vwsub_wx_b, TH_WOP_WSSS_B, H2, H1, TH_SUB)
THCALL(TH_OPIVX2, th_vwsub_wx_h, TH_WOP_WSSS_H, H4, H2, TH_SUB)
THCALL(TH_OPIVX2, th_vwsub_wx_w, TH_WOP_WSSS_W, H8, H4, TH_SUB)
GEN_TH_VX(th_vwaddu_vx_b, 1, 2, clearh_th)
GEN_TH_VX(th_vwaddu_vx_h, 2, 4, clearl_th)
GEN_TH_VX(th_vwaddu_vx_w, 4, 8, clearq_th)
GEN_TH_VX(th_vwsubu_vx_b, 1, 2, clearh_th)
GEN_TH_VX(th_vwsubu_vx_h, 2, 4, clearl_th)
GEN_TH_VX(th_vwsubu_vx_w, 4, 8, clearq_th)
GEN_TH_VX(th_vwadd_vx_b, 1, 2, clearh_th)
GEN_TH_VX(th_vwadd_vx_h, 2, 4, clearl_th)
GEN_TH_VX(th_vwadd_vx_w, 4, 8, clearq_th)
GEN_TH_VX(th_vwsub_vx_b, 1, 2, clearh_th)
GEN_TH_VX(th_vwsub_vx_h, 2, 4, clearl_th)
GEN_TH_VX(th_vwsub_vx_w, 4, 8, clearq_th)
GEN_TH_VX(th_vwaddu_wx_b, 1, 2, clearh_th)
GEN_TH_VX(th_vwaddu_wx_h, 2, 4, clearl_th)
GEN_TH_VX(th_vwaddu_wx_w, 4, 8, clearq_th)
GEN_TH_VX(th_vwsubu_wx_b, 1, 2, clearh_th)
GEN_TH_VX(th_vwsubu_wx_h, 2, 4, clearl_th)
GEN_TH_VX(th_vwsubu_wx_w, 4, 8, clearq_th)
GEN_TH_VX(th_vwadd_wx_b, 1, 2, clearh_th)
GEN_TH_VX(th_vwadd_wx_h, 2, 4, clearl_th)
GEN_TH_VX(th_vwadd_wx_w, 4, 8, clearq_th)
GEN_TH_VX(th_vwsub_wx_b, 1, 2, clearh_th)
GEN_TH_VX(th_vwsub_wx_h, 2, 4, clearl_th)
GEN_TH_VX(th_vwsub_wx_w, 4, 8, clearq_th)

/* Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions */
#define TH_VADC(N, M, C) (N + M + C)
#define TH_VSBC(N, M, C) (N - M - C)
/*
 * This function is almost the copy of GEN_VEXT_VADC_VVM, except:
 * 1) different mask layout
 * 2) different data encoding
 * 3) different tail elements process policy
 */
#define GEN_TH_VADC_VVM(NAME, ETYPE, H, DO_OP, CLEAR_FN)      \
void HELPER(NAME)(void *vd, void *v0, void *vs1, void *vs2,   \
                  CPURISCVState *env, uint32_t desc)          \
{                                                             \
    uint32_t mlen = th_mlen(desc);                            \
    uint32_t vl = env->vl;                                    \
    uint32_t esz = sizeof(ETYPE);                             \
    uint32_t vlmax = th_maxsz(desc) / esz;                    \
    uint32_t i;                                               \
                                                              \
    for (i = env->vstart; i < vl; i++) {                      \
        ETYPE s1 = *((ETYPE *)vs1 + H(i));                    \
        ETYPE s2 = *((ETYPE *)vs2 + H(i));                    \
        uint8_t carry = th_elem_mask(v0, mlen, i);            \
                                                              \
        *((ETYPE *)vd + H(i)) = DO_OP(s2, s1, carry);         \
    }                                                         \
    env->vstart = 0;                                          \
    CLEAR_FN(vd, vl, vl * esz, vlmax * esz);                  \
}

GEN_TH_VADC_VVM(th_vadc_vvm_b, uint8_t,  H1, TH_VADC, clearb_th)
GEN_TH_VADC_VVM(th_vadc_vvm_h, uint16_t, H2, TH_VADC, clearh_th)
GEN_TH_VADC_VVM(th_vadc_vvm_w, uint32_t, H4, TH_VADC, clearl_th)
GEN_TH_VADC_VVM(th_vadc_vvm_d, uint64_t, H8, TH_VADC, clearq_th)

GEN_TH_VADC_VVM(th_vsbc_vvm_b, uint8_t,  H1, TH_VSBC, clearb_th)
GEN_TH_VADC_VVM(th_vsbc_vvm_h, uint16_t, H2, TH_VSBC, clearh_th)
GEN_TH_VADC_VVM(th_vsbc_vvm_w, uint32_t, H4, TH_VSBC, clearl_th)
GEN_TH_VADC_VVM(th_vsbc_vvm_d, uint64_t, H8, TH_VSBC, clearq_th)
/*
 * This function is almost the copy of GEN_VEXT_VADC_VXM, except:
 * 1) different mask layout
 * 2) different data encoding
 * 3) different tail elements process policy
 */
#define GEN_TH_VADC_VXM(NAME, ETYPE, H, DO_OP, CLEAR_FN)                 \
void HELPER(NAME)(void *vd, void *v0, target_ulong s1, void *vs2,        \
                  CPURISCVState *env, uint32_t desc)                     \
{                                                                        \
    uint32_t mlen = th_mlen(desc);                                       \
    uint32_t vl = env->vl;                                               \
    uint32_t esz = sizeof(ETYPE);                                        \
    uint32_t vlmax = th_maxsz(desc) / esz;                               \
    uint32_t i;                                                          \
                                                                         \
    for (i = env->vstart; i < vl; i++) {                                 \
        ETYPE s2 = *((ETYPE *)vs2 + H(i));                               \
        uint8_t carry = th_elem_mask(v0, mlen, i);                       \
                                                                         \
        *((ETYPE *)vd + H(i)) = DO_OP(s2, (ETYPE)(target_long)s1, carry);\
    }                                                                    \
    env->vstart = 0;                                                     \
    CLEAR_FN(vd, vl, vl * esz, vlmax * esz);                             \
}

GEN_TH_VADC_VXM(th_vadc_vxm_b, uint8_t,  H1, TH_VADC, clearb_th)
GEN_TH_VADC_VXM(th_vadc_vxm_h, uint16_t, H2, TH_VADC, clearh_th)
GEN_TH_VADC_VXM(th_vadc_vxm_w, uint32_t, H4, TH_VADC, clearl_th)
GEN_TH_VADC_VXM(th_vadc_vxm_d, uint64_t, H8, TH_VADC, clearq_th)

GEN_TH_VADC_VXM(th_vsbc_vxm_b, uint8_t,  H1, TH_VSBC, clearb_th)
GEN_TH_VADC_VXM(th_vsbc_vxm_h, uint16_t, H2, TH_VSBC, clearh_th)
GEN_TH_VADC_VXM(th_vsbc_vxm_w, uint32_t, H4, TH_VSBC, clearl_th)
GEN_TH_VADC_VXM(th_vsbc_vxm_d, uint64_t, H8, TH_VSBC, clearq_th)

#define TH_MADC(N, M, C) (C ? (__typeof(N))(N + M + 1) <= N :           \
                          (__typeof(N))(N + M) < N)
#define TH_MSBC(N, M, C) (C ? N <= M : N < M)
/*
 * This function is almost the copy of GEN_VEXT_VMADC_VVM, except:
 * 1) different mask layout
 * 2) different data encoding
 * 3) different tail elements process policy
 * 4) When vm = 1, RVV1.0 vmadc and vmsbc perform the computation
 *    without carry-in/borrow-in. While XTheadVector does not have
 *    this kind of situation.
 */
#define GEN_TH_VMADC_VVM(NAME, ETYPE, H, DO_OP)               \
void HELPER(NAME)(void *vd, void *v0, void *vs1, void *vs2,   \
                  CPURISCVState *env, uint32_t desc)          \
{                                                             \
    uint32_t mlen = th_mlen(desc);                            \
    uint32_t vl = env->vl;                                    \
    uint32_t vlmax = th_maxsz(desc) / sizeof(ETYPE);          \
    uint32_t i;                                               \
                                                              \
    for (i = env->vstart; i < vl; i++) {                      \
        ETYPE s1 = *((ETYPE *)vs1 + H(i));                    \
        ETYPE s2 = *((ETYPE *)vs2 + H(i));                    \
        uint8_t carry = th_elem_mask(v0, mlen, i);            \
                                                              \
        th_set_elem_mask(vd, mlen, i, DO_OP(s2, s1, carry));  \
    }                                                         \
    env->vstart = 0;                                          \
    for (; i < vlmax; i++) {                                  \
        th_set_elem_mask(vd, mlen, i, 0);                     \
    }                                                         \
}

GEN_TH_VMADC_VVM(th_vmadc_vvm_b, uint8_t,  H1, TH_MADC)
GEN_TH_VMADC_VVM(th_vmadc_vvm_h, uint16_t, H2, TH_MADC)
GEN_TH_VMADC_VVM(th_vmadc_vvm_w, uint32_t, H4, TH_MADC)
GEN_TH_VMADC_VVM(th_vmadc_vvm_d, uint64_t, H8, TH_MADC)

GEN_TH_VMADC_VVM(th_vmsbc_vvm_b, uint8_t,  H1, TH_MSBC)
GEN_TH_VMADC_VVM(th_vmsbc_vvm_h, uint16_t, H2, TH_MSBC)
GEN_TH_VMADC_VVM(th_vmsbc_vvm_w, uint32_t, H4, TH_MSBC)
GEN_TH_VMADC_VVM(th_vmsbc_vvm_d, uint64_t, H8, TH_MSBC)
/*
 * This function is almost the copy of GEN_VEXT_VMADC_VXM, except:
 * 1) different mask layout
 * 2) different data encoding
 * 3) different tail elements process policy
 * 4) When vm = 1, RVV1.0 vmadc and vmsbc perform the computation
 *    without carry-in/borrow-in. While XTheadVector does not have
 *    this kind of situation.
 */
#define GEN_TH_VMADC_VXM(NAME, ETYPE, H, DO_OP)                 \
void HELPER(NAME)(void *vd, void *v0, target_ulong s1,          \
                  void *vs2, CPURISCVState *env, uint32_t desc) \
{                                                               \
    uint32_t mlen = th_mlen(desc);                              \
    uint32_t vl = env->vl;                                      \
    uint32_t vlmax = th_maxsz(desc) / sizeof(ETYPE);            \
    uint32_t i;                                                 \
                                                                \
    for (i = env->vstart; i < vl; i++) {                        \
        ETYPE s2 = *((ETYPE *)vs2 + H(i));                      \
        uint8_t carry = th_elem_mask(v0, mlen, i);              \
                                                                \
        th_set_elem_mask(vd, mlen, i,                           \
                DO_OP(s2, (ETYPE)(target_long)s1, carry));      \
    }                                                           \
    env->vstart = 0;                                            \
    for (; i < vlmax; i++) {                                    \
        th_set_elem_mask(vd, mlen, i, 0);                       \
    }                                                           \
}

GEN_TH_VMADC_VXM(th_vmadc_vxm_b, uint8_t,  H1, TH_MADC)
GEN_TH_VMADC_VXM(th_vmadc_vxm_h, uint16_t, H2, TH_MADC)
GEN_TH_VMADC_VXM(th_vmadc_vxm_w, uint32_t, H4, TH_MADC)
GEN_TH_VMADC_VXM(th_vmadc_vxm_d, uint64_t, H8, TH_MADC)

GEN_TH_VMADC_VXM(th_vmsbc_vxm_b, uint8_t,  H1, TH_MSBC)
GEN_TH_VMADC_VXM(th_vmsbc_vxm_h, uint16_t, H2, TH_MSBC)
GEN_TH_VMADC_VXM(th_vmsbc_vxm_w, uint32_t, H4, TH_MSBC)
GEN_TH_VMADC_VXM(th_vmsbc_vxm_d, uint64_t, H8, TH_MSBC)

/* Vector Bitwise Logical Instructions */
THCALL(TH_OPIVV2, th_vand_vv_b, TH_OP_SSS_B, H1, H1, H1, TH_AND)
THCALL(TH_OPIVV2, th_vand_vv_h, TH_OP_SSS_H, H2, H2, H2, TH_AND)
THCALL(TH_OPIVV2, th_vand_vv_w, TH_OP_SSS_W, H4, H4, H4, TH_AND)
THCALL(TH_OPIVV2, th_vand_vv_d, TH_OP_SSS_D, H8, H8, H8, TH_AND)
THCALL(TH_OPIVV2, th_vor_vv_b, TH_OP_SSS_B, H1, H1, H1, TH_OR)
THCALL(TH_OPIVV2, th_vor_vv_h, TH_OP_SSS_H, H2, H2, H2, TH_OR)
THCALL(TH_OPIVV2, th_vor_vv_w, TH_OP_SSS_W, H4, H4, H4, TH_OR)
THCALL(TH_OPIVV2, th_vor_vv_d, TH_OP_SSS_D, H8, H8, H8, TH_OR)
THCALL(TH_OPIVV2, th_vxor_vv_b, TH_OP_SSS_B, H1, H1, H1, TH_XOR)
THCALL(TH_OPIVV2, th_vxor_vv_h, TH_OP_SSS_H, H2, H2, H2, TH_XOR)
THCALL(TH_OPIVV2, th_vxor_vv_w, TH_OP_SSS_W, H4, H4, H4, TH_XOR)
THCALL(TH_OPIVV2, th_vxor_vv_d, TH_OP_SSS_D, H8, H8, H8, TH_XOR)
GEN_TH_VV(th_vand_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vand_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vand_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vand_vv_d, 8, 8, clearq_th)
GEN_TH_VV(th_vor_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vor_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vor_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vor_vv_d, 8, 8, clearq_th)
GEN_TH_VV(th_vxor_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vxor_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vxor_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vxor_vv_d, 8, 8, clearq_th)

THCALL(TH_OPIVX2, th_vand_vx_b, TH_OP_SSS_B, H1, H1, TH_AND)
THCALL(TH_OPIVX2, th_vand_vx_h, TH_OP_SSS_H, H2, H2, TH_AND)
THCALL(TH_OPIVX2, th_vand_vx_w, TH_OP_SSS_W, H4, H4, TH_AND)
THCALL(TH_OPIVX2, th_vand_vx_d, TH_OP_SSS_D, H8, H8, TH_AND)
THCALL(TH_OPIVX2, th_vor_vx_b, TH_OP_SSS_B, H1, H1, TH_OR)
THCALL(TH_OPIVX2, th_vor_vx_h, TH_OP_SSS_H, H2, H2, TH_OR)
THCALL(TH_OPIVX2, th_vor_vx_w, TH_OP_SSS_W, H4, H4, TH_OR)
THCALL(TH_OPIVX2, th_vor_vx_d, TH_OP_SSS_D, H8, H8, TH_OR)
THCALL(TH_OPIVX2, th_vxor_vx_b, TH_OP_SSS_B, H1, H1, TH_XOR)
THCALL(TH_OPIVX2, th_vxor_vx_h, TH_OP_SSS_H, H2, H2, TH_XOR)
THCALL(TH_OPIVX2, th_vxor_vx_w, TH_OP_SSS_W, H4, H4, TH_XOR)
THCALL(TH_OPIVX2, th_vxor_vx_d, TH_OP_SSS_D, H8, H8, TH_XOR)
GEN_TH_VX(th_vand_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vand_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vand_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vand_vx_d, 8, 8, clearq_th)
GEN_TH_VX(th_vor_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vor_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vor_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vor_vx_d, 8, 8, clearq_th)
GEN_TH_VX(th_vxor_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vxor_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vxor_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vxor_vx_d, 8, 8, clearq_th)

/* Vector Single-Width Bit Shift Instructions */
#define TH_SLL(N, M)  (N << (M))
#define TH_SRL(N, M)  (N >> (M))
/*
 * GEN_TH_SHIFT_VV and GEN_TH_SHIFT_VX are almost the copy of
 * GEN_VEXT_SHIFT_VV and GEN_VEXT_SHIFT_VX, except:
 * 1) different mask layout
 * 2) different data encoding
 * 3) different masked/tail elements process policy
 */
/* generate the helpers for shift instructions with two vector operators */
#define GEN_TH_SHIFT_VV(NAME, TS1, TS2, HS1, HS2, OP, MASK, CLEAR_FN)     \
void HELPER(NAME)(void *vd, void *v0, void *vs1,                          \
                  void *vs2, CPURISCVState *env, uint32_t desc)           \
{                                                                         \
    uint32_t mlen = th_mlen(desc);                                        \
    uint32_t vm = th_vm(desc);                                            \
    uint32_t vl = env->vl;                                                \
    uint32_t esz = sizeof(TS1);                                           \
    uint32_t vlmax = th_maxsz(desc) / esz;                                \
    uint32_t i;                                                           \
                                                                          \
    for (i = env->vstart; i < vl; i++) {                                  \
        if (!vm && !th_elem_mask(v0, mlen, i)) {                          \
            continue;                                                     \
        }                                                                 \
        TS1 s1 = *((TS1 *)vs1 + HS1(i));                                  \
        TS2 s2 = *((TS2 *)vs2 + HS2(i));                                  \
        *((TS1 *)vd + HS1(i)) = OP(s2, s1 & MASK);                        \
    }                                                                     \
    env->vstart = 0;                                                      \
    CLEAR_FN(vd, vl, vl * esz, vlmax * esz);                              \
}

GEN_TH_SHIFT_VV(th_vsll_vv_b, uint8_t,  uint8_t, H1, H1, TH_SLL,
                0x7, clearb_th)
GEN_TH_SHIFT_VV(th_vsll_vv_h, uint16_t, uint16_t, H2, H2, TH_SLL,
                0xf, clearh_th)
GEN_TH_SHIFT_VV(th_vsll_vv_w, uint32_t, uint32_t, H4, H4, TH_SLL,
                0x1f, clearl_th)
GEN_TH_SHIFT_VV(th_vsll_vv_d, uint64_t, uint64_t, H8, H8, TH_SLL,
                0x3f, clearq_th)

GEN_TH_SHIFT_VV(th_vsrl_vv_b, uint8_t, uint8_t, H1, H1, TH_SRL,
                0x7, clearb_th)
GEN_TH_SHIFT_VV(th_vsrl_vv_h, uint16_t, uint16_t, H2, H2, TH_SRL,
                0xf, clearh_th)
GEN_TH_SHIFT_VV(th_vsrl_vv_w, uint32_t, uint32_t, H4, H4, TH_SRL,
                0x1f, clearl_th)
GEN_TH_SHIFT_VV(th_vsrl_vv_d, uint64_t, uint64_t, H8, H8, TH_SRL,
                0x3f, clearq_th)

GEN_TH_SHIFT_VV(th_vsra_vv_b, uint8_t,  int8_t, H1, H1, TH_SRL,
                0x7, clearb_th)
GEN_TH_SHIFT_VV(th_vsra_vv_h, uint16_t, int16_t, H2, H2, TH_SRL,
                0xf, clearh_th)
GEN_TH_SHIFT_VV(th_vsra_vv_w, uint32_t, int32_t, H4, H4, TH_SRL,
                0x1f, clearl_th)
GEN_TH_SHIFT_VV(th_vsra_vv_d, uint64_t, int64_t, H8, H8, TH_SRL,
                0x3f, clearq_th)

/* generate the helpers for shift instructions with one vector and one scalar */
#define GEN_TH_SHIFT_VX(NAME, TD, TS2, HD, HS2, OP, MASK, CLEAR_FN)   \
void HELPER(NAME)(void *vd, void *v0, target_ulong s1,                \
        void *vs2, CPURISCVState *env, uint32_t desc)                 \
{                                                                     \
    uint32_t mlen = th_mlen(desc);                                    \
    uint32_t vm = th_vm(desc);                                        \
    uint32_t vl = env->vl;                                            \
    uint32_t esz = sizeof(TD);                                        \
    uint32_t vlmax = th_maxsz(desc) / esz;                            \
    uint32_t i;                                                       \
                                                                      \
    for (i = env->vstart; i < vl; i++) {                              \
        if (!vm && !th_elem_mask(v0, mlen, i)) {                      \
            continue;                                                 \
        }                                                             \
        TS2 s2 = *((TS2 *)vs2 + HS2(i));                              \
        *((TD *)vd + HD(i)) = OP(s2, s1 & MASK);                      \
    }                                                                 \
    env->vstart = 0;                                                  \
    CLEAR_FN(vd, vl, vl * esz, vlmax * esz);                          \
}

GEN_TH_SHIFT_VX(th_vsll_vx_b, uint8_t, int8_t, H1, H1, TH_SLL,
                0x7, clearb_th)
GEN_TH_SHIFT_VX(th_vsll_vx_h, uint16_t, int16_t, H2, H2, TH_SLL,
                0xf, clearh_th)
GEN_TH_SHIFT_VX(th_vsll_vx_w, uint32_t, int32_t, H4, H4, TH_SLL,
                0x1f, clearl_th)
GEN_TH_SHIFT_VX(th_vsll_vx_d, uint64_t, int64_t, H8, H8, TH_SLL,
                0x3f, clearq_th)

GEN_TH_SHIFT_VX(th_vsrl_vx_b, uint8_t, uint8_t, H1, H1, TH_SRL,
                0x7, clearb_th)
GEN_TH_SHIFT_VX(th_vsrl_vx_h, uint16_t, uint16_t, H2, H2, TH_SRL,
                0xf, clearh_th)
GEN_TH_SHIFT_VX(th_vsrl_vx_w, uint32_t, uint32_t, H4, H4, TH_SRL,
                0x1f, clearl_th)
GEN_TH_SHIFT_VX(th_vsrl_vx_d, uint64_t, uint64_t, H8, H8, TH_SRL,
                0x3f, clearq_th)

GEN_TH_SHIFT_VX(th_vsra_vx_b, int8_t, int8_t, H1, H1, TH_SRL,
                0x7, clearb_th)
GEN_TH_SHIFT_VX(th_vsra_vx_h, int16_t, int16_t, H2, H2, TH_SRL,
                0xf, clearh_th)
GEN_TH_SHIFT_VX(th_vsra_vx_w, int32_t, int32_t, H4, H4, TH_SRL,
                0x1f, clearl_th)
GEN_TH_SHIFT_VX(th_vsra_vx_d, int64_t, int64_t, H8, H8, TH_SRL,
                0x3f, clearq_th)

/* Vector Narrowing Integer Right Shift Instructions */
GEN_TH_SHIFT_VV(th_vnsrl_vv_b, uint8_t,  uint16_t, H1, H2, TH_SRL,
                0xf, clearb_th)
GEN_TH_SHIFT_VV(th_vnsrl_vv_h, uint16_t, uint32_t, H2, H4, TH_SRL,
                0x1f, clearh_th)
GEN_TH_SHIFT_VV(th_vnsrl_vv_w, uint32_t, uint64_t, H4, H8, TH_SRL,
                0x3f, clearl_th)
GEN_TH_SHIFT_VV(th_vnsra_vv_b, uint8_t,  int16_t, H1, H2, TH_SRL,
                0xf, clearb_th)
GEN_TH_SHIFT_VV(th_vnsra_vv_h, uint16_t, int32_t, H2, H4, TH_SRL,
                0x1f, clearh_th)
GEN_TH_SHIFT_VV(th_vnsra_vv_w, uint32_t, int64_t, H4, H8, TH_SRL,
                0x3f, clearl_th)
GEN_TH_SHIFT_VX(th_vnsrl_vx_b, uint8_t, uint16_t, H1, H2, TH_SRL,
                0xf, clearb_th)
GEN_TH_SHIFT_VX(th_vnsrl_vx_h, uint16_t, uint32_t, H2, H4, TH_SRL,
                0x1f, clearh_th)
GEN_TH_SHIFT_VX(th_vnsrl_vx_w, uint32_t, uint64_t, H4, H8, TH_SRL,
                0x3f, clearl_th)
GEN_TH_SHIFT_VX(th_vnsra_vx_b, int8_t, int16_t, H1, H2, TH_SRL,
                0xf, clearb_th)
GEN_TH_SHIFT_VX(th_vnsra_vx_h, int16_t, int32_t, H2, H4, TH_SRL,
                0x1f, clearh_th)
GEN_TH_SHIFT_VX(th_vnsra_vx_w, int32_t, int64_t, H4, H8, TH_SRL,
                0x3f, clearl_th)

/* Vector Integer Comparison Instructions */
#define TH_MSEQ(N, M) (N == M)
#define TH_MSNE(N, M) (N != M)
#define TH_MSLT(N, M) (N < M)
#define TH_MSLE(N, M) (N <= M)
#define TH_MSGT(N, M) (N > M)

#define GEN_TH_CMP_VV(NAME, ETYPE, H, DO_OP)                  \
void HELPER(NAME)(void *vd, void *v0, void *vs1, void *vs2,   \
                  CPURISCVState *env, uint32_t desc)          \
{                                                             \
    uint32_t mlen = th_mlen(desc);                            \
    uint32_t vm = th_vm(desc);                                \
    uint32_t vl = env->vl;                                    \
    uint32_t vlmax = th_maxsz(desc) / sizeof(ETYPE);          \
    uint32_t i;                                               \
                                                              \
    for (i = env->vstart; i < vl; i++) {                      \
        ETYPE s1 = *((ETYPE *)vs1 + H(i));                    \
        ETYPE s2 = *((ETYPE *)vs2 + H(i));                    \
        if (!vm && !th_elem_mask(v0, mlen, i)) {              \
            continue;                                         \
        }                                                     \
        th_set_elem_mask(vd, mlen, i, DO_OP(s2, s1));         \
    }                                                         \
    env->vstart = 0;                                          \
    for (; i < vlmax; i++) {                                  \
        th_set_elem_mask(vd, mlen, i, 0);                     \
    }                                                         \
}

GEN_TH_CMP_VV(th_vmseq_vv_b, uint8_t,  H1, TH_MSEQ)
GEN_TH_CMP_VV(th_vmseq_vv_h, uint16_t, H2, TH_MSEQ)
GEN_TH_CMP_VV(th_vmseq_vv_w, uint32_t, H4, TH_MSEQ)
GEN_TH_CMP_VV(th_vmseq_vv_d, uint64_t, H8, TH_MSEQ)

GEN_TH_CMP_VV(th_vmsne_vv_b, uint8_t,  H1, TH_MSNE)
GEN_TH_CMP_VV(th_vmsne_vv_h, uint16_t, H2, TH_MSNE)
GEN_TH_CMP_VV(th_vmsne_vv_w, uint32_t, H4, TH_MSNE)
GEN_TH_CMP_VV(th_vmsne_vv_d, uint64_t, H8, TH_MSNE)

GEN_TH_CMP_VV(th_vmsltu_vv_b, uint8_t,  H1, TH_MSLT)
GEN_TH_CMP_VV(th_vmsltu_vv_h, uint16_t, H2, TH_MSLT)
GEN_TH_CMP_VV(th_vmsltu_vv_w, uint32_t, H4, TH_MSLT)
GEN_TH_CMP_VV(th_vmsltu_vv_d, uint64_t, H8, TH_MSLT)

GEN_TH_CMP_VV(th_vmslt_vv_b, int8_t,  H1, TH_MSLT)
GEN_TH_CMP_VV(th_vmslt_vv_h, int16_t, H2, TH_MSLT)
GEN_TH_CMP_VV(th_vmslt_vv_w, int32_t, H4, TH_MSLT)
GEN_TH_CMP_VV(th_vmslt_vv_d, int64_t, H8, TH_MSLT)

GEN_TH_CMP_VV(th_vmsleu_vv_b, uint8_t,  H1, TH_MSLE)
GEN_TH_CMP_VV(th_vmsleu_vv_h, uint16_t, H2, TH_MSLE)
GEN_TH_CMP_VV(th_vmsleu_vv_w, uint32_t, H4, TH_MSLE)
GEN_TH_CMP_VV(th_vmsleu_vv_d, uint64_t, H8, TH_MSLE)

GEN_TH_CMP_VV(th_vmsle_vv_b, int8_t,  H1, TH_MSLE)
GEN_TH_CMP_VV(th_vmsle_vv_h, int16_t, H2, TH_MSLE)
GEN_TH_CMP_VV(th_vmsle_vv_w, int32_t, H4, TH_MSLE)
GEN_TH_CMP_VV(th_vmsle_vv_d, int64_t, H8, TH_MSLE)

#define GEN_TH_CMP_VX(NAME, ETYPE, H, DO_OP)                        \
void HELPER(NAME)(void *vd, void *v0, target_ulong s1, void *vs2,   \
                  CPURISCVState *env, uint32_t desc)                \
{                                                                   \
    uint32_t mlen = th_mlen(desc);                                  \
    uint32_t vm = th_vm(desc);                                      \
    uint32_t vl = env->vl;                                          \
    uint32_t vlmax = th_maxsz(desc) / sizeof(ETYPE);                \
    uint32_t i;                                                     \
                                                                    \
    for (i = env->vstart; i < vl; i++) {                            \
        ETYPE s2 = *((ETYPE *)vs2 + H(i));                          \
        if (!vm && !th_elem_mask(v0, mlen, i)) {                    \
            continue;                                               \
        }                                                           \
        th_set_elem_mask(vd, mlen, i,                               \
                DO_OP(s2, (ETYPE)(target_long)s1));                 \
    }                                                               \
    env->vstart = 0;                                                \
    for (; i < vlmax; i++) {                                        \
        th_set_elem_mask(vd, mlen, i, 0);                           \
    }                                                               \
}

GEN_TH_CMP_VX(th_vmseq_vx_b, uint8_t,  H1, TH_MSEQ)
GEN_TH_CMP_VX(th_vmseq_vx_h, uint16_t, H2, TH_MSEQ)
GEN_TH_CMP_VX(th_vmseq_vx_w, uint32_t, H4, TH_MSEQ)
GEN_TH_CMP_VX(th_vmseq_vx_d, uint64_t, H8, TH_MSEQ)

GEN_TH_CMP_VX(th_vmsne_vx_b, uint8_t,  H1, TH_MSNE)
GEN_TH_CMP_VX(th_vmsne_vx_h, uint16_t, H2, TH_MSNE)
GEN_TH_CMP_VX(th_vmsne_vx_w, uint32_t, H4, TH_MSNE)
GEN_TH_CMP_VX(th_vmsne_vx_d, uint64_t, H8, TH_MSNE)

GEN_TH_CMP_VX(th_vmsltu_vx_b, uint8_t,  H1, TH_MSLT)
GEN_TH_CMP_VX(th_vmsltu_vx_h, uint16_t, H2, TH_MSLT)
GEN_TH_CMP_VX(th_vmsltu_vx_w, uint32_t, H4, TH_MSLT)
GEN_TH_CMP_VX(th_vmsltu_vx_d, uint64_t, H8, TH_MSLT)

GEN_TH_CMP_VX(th_vmslt_vx_b, int8_t,  H1, TH_MSLT)
GEN_TH_CMP_VX(th_vmslt_vx_h, int16_t, H2, TH_MSLT)
GEN_TH_CMP_VX(th_vmslt_vx_w, int32_t, H4, TH_MSLT)
GEN_TH_CMP_VX(th_vmslt_vx_d, int64_t, H8, TH_MSLT)

GEN_TH_CMP_VX(th_vmsleu_vx_b, uint8_t,  H1, TH_MSLE)
GEN_TH_CMP_VX(th_vmsleu_vx_h, uint16_t, H2, TH_MSLE)
GEN_TH_CMP_VX(th_vmsleu_vx_w, uint32_t, H4, TH_MSLE)
GEN_TH_CMP_VX(th_vmsleu_vx_d, uint64_t, H8, TH_MSLE)

GEN_TH_CMP_VX(th_vmsle_vx_b, int8_t,  H1, TH_MSLE)
GEN_TH_CMP_VX(th_vmsle_vx_h, int16_t, H2, TH_MSLE)
GEN_TH_CMP_VX(th_vmsle_vx_w, int32_t, H4, TH_MSLE)
GEN_TH_CMP_VX(th_vmsle_vx_d, int64_t, H8, TH_MSLE)

GEN_TH_CMP_VX(th_vmsgtu_vx_b, uint8_t,  H1, TH_MSGT)
GEN_TH_CMP_VX(th_vmsgtu_vx_h, uint16_t, H2, TH_MSGT)
GEN_TH_CMP_VX(th_vmsgtu_vx_w, uint32_t, H4, TH_MSGT)
GEN_TH_CMP_VX(th_vmsgtu_vx_d, uint64_t, H8, TH_MSGT)

GEN_TH_CMP_VX(th_vmsgt_vx_b, int8_t,  H1, TH_MSGT)
GEN_TH_CMP_VX(th_vmsgt_vx_h, int16_t, H2, TH_MSGT)
GEN_TH_CMP_VX(th_vmsgt_vx_w, int32_t, H4, TH_MSGT)
GEN_TH_CMP_VX(th_vmsgt_vx_d, int64_t, H8, TH_MSGT)

/* Vector Integer Min/Max Instructions */
THCALL(TH_OPIVV2, th_vminu_vv_b, TH_OP_UUU_B, H1, H1, H1, TH_MIN)
THCALL(TH_OPIVV2, th_vminu_vv_h, TH_OP_UUU_H, H2, H2, H2, TH_MIN)
THCALL(TH_OPIVV2, th_vminu_vv_w, TH_OP_UUU_W, H4, H4, H4, TH_MIN)
THCALL(TH_OPIVV2, th_vminu_vv_d, TH_OP_UUU_D, H8, H8, H8, TH_MIN)
THCALL(TH_OPIVV2, th_vmin_vv_b, TH_OP_SSS_B, H1, H1, H1, TH_MIN)
THCALL(TH_OPIVV2, th_vmin_vv_h, TH_OP_SSS_H, H2, H2, H2, TH_MIN)
THCALL(TH_OPIVV2, th_vmin_vv_w, TH_OP_SSS_W, H4, H4, H4, TH_MIN)
THCALL(TH_OPIVV2, th_vmin_vv_d, TH_OP_SSS_D, H8, H8, H8, TH_MIN)
THCALL(TH_OPIVV2, th_vmaxu_vv_b, TH_OP_UUU_B, H1, H1, H1, TH_MAX)
THCALL(TH_OPIVV2, th_vmaxu_vv_h, TH_OP_UUU_H, H2, H2, H2, TH_MAX)
THCALL(TH_OPIVV2, th_vmaxu_vv_w, TH_OP_UUU_W, H4, H4, H4, TH_MAX)
THCALL(TH_OPIVV2, th_vmaxu_vv_d, TH_OP_UUU_D, H8, H8, H8, TH_MAX)
THCALL(TH_OPIVV2, th_vmax_vv_b, TH_OP_SSS_B, H1, H1, H1, TH_MAX)
THCALL(TH_OPIVV2, th_vmax_vv_h, TH_OP_SSS_H, H2, H2, H2, TH_MAX)
THCALL(TH_OPIVV2, th_vmax_vv_w, TH_OP_SSS_W, H4, H4, H4, TH_MAX)
THCALL(TH_OPIVV2, th_vmax_vv_d, TH_OP_SSS_D, H8, H8, H8, TH_MAX)
GEN_TH_VV(th_vminu_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vminu_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vminu_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vminu_vv_d, 8, 8, clearq_th)
GEN_TH_VV(th_vmin_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vmin_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vmin_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vmin_vv_d, 8, 8, clearq_th)
GEN_TH_VV(th_vmaxu_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vmaxu_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vmaxu_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vmaxu_vv_d, 8, 8, clearq_th)
GEN_TH_VV(th_vmax_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vmax_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vmax_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vmax_vv_d, 8, 8, clearq_th)

THCALL(TH_OPIVX2, th_vminu_vx_b, TH_OP_UUU_B, H1, H1, TH_MIN)
THCALL(TH_OPIVX2, th_vminu_vx_h, TH_OP_UUU_H, H2, H2, TH_MIN)
THCALL(TH_OPIVX2, th_vminu_vx_w, TH_OP_UUU_W, H4, H4, TH_MIN)
THCALL(TH_OPIVX2, th_vminu_vx_d, TH_OP_UUU_D, H8, H8, TH_MIN)
THCALL(TH_OPIVX2, th_vmin_vx_b, TH_OP_SSS_B, H1, H1, TH_MIN)
THCALL(TH_OPIVX2, th_vmin_vx_h, TH_OP_SSS_H, H2, H2, TH_MIN)
THCALL(TH_OPIVX2, th_vmin_vx_w, TH_OP_SSS_W, H4, H4, TH_MIN)
THCALL(TH_OPIVX2, th_vmin_vx_d, TH_OP_SSS_D, H8, H8, TH_MIN)
THCALL(TH_OPIVX2, th_vmaxu_vx_b, TH_OP_UUU_B, H1, H1, TH_MAX)
THCALL(TH_OPIVX2, th_vmaxu_vx_h, TH_OP_UUU_H, H2, H2, TH_MAX)
THCALL(TH_OPIVX2, th_vmaxu_vx_w, TH_OP_UUU_W, H4, H4, TH_MAX)
THCALL(TH_OPIVX2, th_vmaxu_vx_d, TH_OP_UUU_D, H8, H8, TH_MAX)
THCALL(TH_OPIVX2, th_vmax_vx_b, TH_OP_SSS_B, H1, H1, TH_MAX)
THCALL(TH_OPIVX2, th_vmax_vx_h, TH_OP_SSS_H, H2, H2, TH_MAX)
THCALL(TH_OPIVX2, th_vmax_vx_w, TH_OP_SSS_W, H4, H4, TH_MAX)
THCALL(TH_OPIVX2, th_vmax_vx_d, TH_OP_SSS_D, H8, H8, TH_MAX)
GEN_TH_VX(th_vminu_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vminu_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vminu_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vminu_vx_d, 8, 8, clearq_th)
GEN_TH_VX(th_vmin_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vmin_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vmin_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vmin_vx_d, 8, 8, clearq_th)
GEN_TH_VX(th_vmaxu_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vmaxu_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vmaxu_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vmaxu_vx_d, 8, 8,  clearq_th)
GEN_TH_VX(th_vmax_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vmax_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vmax_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vmax_vx_d, 8, 8, clearq_th)

/* Vector Single-Width Integer Multiply Instructions */
#define TH_MUL(N, M) (N * M)
THCALL(TH_OPIVV2, th_vmul_vv_b, TH_OP_SSS_B, H1, H1, H1, TH_MUL)
THCALL(TH_OPIVV2, th_vmul_vv_h, TH_OP_SSS_H, H2, H2, H2, TH_MUL)
THCALL(TH_OPIVV2, th_vmul_vv_w, TH_OP_SSS_W, H4, H4, H4, TH_MUL)
THCALL(TH_OPIVV2, th_vmul_vv_d, TH_OP_SSS_D, H8, H8, H8, TH_MUL)
GEN_TH_VV(th_vmul_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vmul_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vmul_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vmul_vv_d, 8, 8, clearq_th)

#define GEN_TH_MUL_FUNC(NAME, TYPE)             \
static TYPE th_##NAME(TYPE s2, TYPE s1)         \
{                                               \
    return do_##NAME(s2, s1);                   \
}

GEN_TH_MUL_FUNC(mulh_b, int8_t)
GEN_TH_MUL_FUNC(mulh_h, int16_t)
GEN_TH_MUL_FUNC(mulh_w, int32_t)
GEN_TH_MUL_FUNC(mulh_d, int64_t)
GEN_TH_MUL_FUNC(mulhu_b, uint8_t)
GEN_TH_MUL_FUNC(mulhu_h, uint16_t)
GEN_TH_MUL_FUNC(mulhu_w, uint32_t)
GEN_TH_MUL_FUNC(mulhu_d, uint64_t)
GEN_TH_MUL_FUNC(mulhsu_b, uint8_t)
GEN_TH_MUL_FUNC(mulhsu_h, uint16_t)
GEN_TH_MUL_FUNC(mulhsu_w, uint32_t)
GEN_TH_MUL_FUNC(mulhsu_d, uint64_t)

THCALL(TH_OPIVV2, th_vmulh_vv_b, TH_OP_SSS_B, H1, H1, H1, th_mulh_b)
THCALL(TH_OPIVV2, th_vmulh_vv_h, TH_OP_SSS_H, H2, H2, H2, th_mulh_h)
THCALL(TH_OPIVV2, th_vmulh_vv_w, TH_OP_SSS_W, H4, H4, H4, th_mulh_w)
THCALL(TH_OPIVV2, th_vmulh_vv_d, TH_OP_SSS_D, H8, H8, H8, th_mulh_d)
THCALL(TH_OPIVV2, th_vmulhu_vv_b, TH_OP_UUU_B, H1, H1, H1, th_mulhu_b)
THCALL(TH_OPIVV2, th_vmulhu_vv_h, TH_OP_UUU_H, H2, H2, H2, th_mulhu_h)
THCALL(TH_OPIVV2, th_vmulhu_vv_w, TH_OP_UUU_W, H4, H4, H4, th_mulhu_w)
THCALL(TH_OPIVV2, th_vmulhu_vv_d, TH_OP_UUU_D, H8, H8, H8, th_mulhu_d)
THCALL(TH_OPIVV2, th_vmulhsu_vv_b, TH_OP_SUS_B, H1, H1, H1, th_mulhsu_b)
THCALL(TH_OPIVV2, th_vmulhsu_vv_h, TH_OP_SUS_H, H2, H2, H2, th_mulhsu_h)
THCALL(TH_OPIVV2, th_vmulhsu_vv_w, TH_OP_SUS_W, H4, H4, H4, th_mulhsu_w)
THCALL(TH_OPIVV2, th_vmulhsu_vv_d, TH_OP_SUS_D, H8, H8, H8, th_mulhsu_d)
GEN_TH_VV(th_vmulh_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vmulh_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vmulh_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vmulh_vv_d, 8, 8, clearq_th)
GEN_TH_VV(th_vmulhu_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vmulhu_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vmulhu_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vmulhu_vv_d, 8, 8, clearq_th)
GEN_TH_VV(th_vmulhsu_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vmulhsu_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vmulhsu_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vmulhsu_vv_d, 8, 8, clearq_th)

THCALL(TH_OPIVX2, th_vmul_vx_b, TH_OP_SSS_B, H1, H1, TH_MUL)
THCALL(TH_OPIVX2, th_vmul_vx_h, TH_OP_SSS_H, H2, H2, TH_MUL)
THCALL(TH_OPIVX2, th_vmul_vx_w, TH_OP_SSS_W, H4, H4, TH_MUL)
THCALL(TH_OPIVX2, th_vmul_vx_d, TH_OP_SSS_D, H8, H8, TH_MUL)
THCALL(TH_OPIVX2, th_vmulh_vx_b, TH_OP_SSS_B, H1, H1, th_mulh_b)
THCALL(TH_OPIVX2, th_vmulh_vx_h, TH_OP_SSS_H, H2, H2, th_mulh_h)
THCALL(TH_OPIVX2, th_vmulh_vx_w, TH_OP_SSS_W, H4, H4, th_mulh_w)
THCALL(TH_OPIVX2, th_vmulh_vx_d, TH_OP_SSS_D, H8, H8, th_mulh_d)
THCALL(TH_OPIVX2, th_vmulhu_vx_b, TH_OP_UUU_B, H1, H1, th_mulhu_b)
THCALL(TH_OPIVX2, th_vmulhu_vx_h, TH_OP_UUU_H, H2, H2, th_mulhu_h)
THCALL(TH_OPIVX2, th_vmulhu_vx_w, TH_OP_UUU_W, H4, H4, th_mulhu_w)
THCALL(TH_OPIVX2, th_vmulhu_vx_d, TH_OP_UUU_D, H8, H8, th_mulhu_d)
THCALL(TH_OPIVX2, th_vmulhsu_vx_b, TH_OP_SUS_B, H1, H1, th_mulhsu_b)
THCALL(TH_OPIVX2, th_vmulhsu_vx_h, TH_OP_SUS_H, H2, H2, th_mulhsu_h)
THCALL(TH_OPIVX2, th_vmulhsu_vx_w, TH_OP_SUS_W, H4, H4, th_mulhsu_w)
THCALL(TH_OPIVX2, th_vmulhsu_vx_d, TH_OP_SUS_D, H8, H8, th_mulhsu_d)
GEN_TH_VX(th_vmul_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vmul_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vmul_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vmul_vx_d, 8, 8, clearq_th)
GEN_TH_VX(th_vmulh_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vmulh_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vmulh_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vmulh_vx_d, 8, 8, clearq_th)
GEN_TH_VX(th_vmulhu_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vmulhu_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vmulhu_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vmulhu_vx_d, 8, 8, clearq_th)
GEN_TH_VX(th_vmulhsu_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vmulhsu_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vmulhsu_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vmulhsu_vx_d, 8, 8, clearq_th)

/* Vector Integer Divide Instructions */
#define TH_DIVU(N, M) (unlikely(M == 0) ? (__typeof(N))(-1) : N / M)
#define TH_REMU(N, M) (unlikely(M == 0) ? N : N % M)
#define TH_DIV(N, M)  (unlikely(M == 0) ? (__typeof(N))(-1) :\
        unlikely((N == -N) && (M == (__typeof(N))(-1))) ? N : N / M)
#define TH_REM(N, M)  (unlikely(M == 0) ? N :\
        unlikely((N == -N) && (M == (__typeof(N))(-1))) ? 0 : N % M)

THCALL(TH_OPIVV2, th_vdivu_vv_b, TH_OP_UUU_B, H1, H1, H1, TH_DIVU)
THCALL(TH_OPIVV2, th_vdivu_vv_h, TH_OP_UUU_H, H2, H2, H2, TH_DIVU)
THCALL(TH_OPIVV2, th_vdivu_vv_w, TH_OP_UUU_W, H4, H4, H4, TH_DIVU)
THCALL(TH_OPIVV2, th_vdivu_vv_d, TH_OP_UUU_D, H8, H8, H8, TH_DIVU)
THCALL(TH_OPIVV2, th_vdiv_vv_b, TH_OP_SSS_B, H1, H1, H1, TH_DIV)
THCALL(TH_OPIVV2, th_vdiv_vv_h, TH_OP_SSS_H, H2, H2, H2, TH_DIV)
THCALL(TH_OPIVV2, th_vdiv_vv_w, TH_OP_SSS_W, H4, H4, H4, TH_DIV)
THCALL(TH_OPIVV2, th_vdiv_vv_d, TH_OP_SSS_D, H8, H8, H8, TH_DIV)
THCALL(TH_OPIVV2, th_vremu_vv_b, TH_OP_UUU_B, H1, H1, H1, TH_REMU)
THCALL(TH_OPIVV2, th_vremu_vv_h, TH_OP_UUU_H, H2, H2, H2, TH_REMU)
THCALL(TH_OPIVV2, th_vremu_vv_w, TH_OP_UUU_W, H4, H4, H4, TH_REMU)
THCALL(TH_OPIVV2, th_vremu_vv_d, TH_OP_UUU_D, H8, H8, H8, TH_REMU)
THCALL(TH_OPIVV2, th_vrem_vv_b, TH_OP_SSS_B, H1, H1, H1, TH_REM)
THCALL(TH_OPIVV2, th_vrem_vv_h, TH_OP_SSS_H, H2, H2, H2, TH_REM)
THCALL(TH_OPIVV2, th_vrem_vv_w, TH_OP_SSS_W, H4, H4, H4, TH_REM)
THCALL(TH_OPIVV2, th_vrem_vv_d, TH_OP_SSS_D, H8, H8, H8, TH_REM)
GEN_TH_VV(th_vdivu_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vdivu_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vdivu_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vdivu_vv_d, 8, 8, clearq_th)
GEN_TH_VV(th_vdiv_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vdiv_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vdiv_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vdiv_vv_d, 8, 8, clearq_th)
GEN_TH_VV(th_vremu_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vremu_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vremu_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vremu_vv_d, 8, 8, clearq_th)
GEN_TH_VV(th_vrem_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vrem_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vrem_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vrem_vv_d, 8, 8, clearq_th)

THCALL(TH_OPIVX2, th_vdivu_vx_b, TH_OP_UUU_B, H1, H1, TH_DIVU)
THCALL(TH_OPIVX2, th_vdivu_vx_h, TH_OP_UUU_H, H2, H2, TH_DIVU)
THCALL(TH_OPIVX2, th_vdivu_vx_w, TH_OP_UUU_W, H4, H4, TH_DIVU)
THCALL(TH_OPIVX2, th_vdivu_vx_d, TH_OP_UUU_D, H8, H8, TH_DIVU)
THCALL(TH_OPIVX2, th_vdiv_vx_b, TH_OP_SSS_B, H1, H1, TH_DIV)
THCALL(TH_OPIVX2, th_vdiv_vx_h, TH_OP_SSS_H, H2, H2, TH_DIV)
THCALL(TH_OPIVX2, th_vdiv_vx_w, TH_OP_SSS_W, H4, H4, TH_DIV)
THCALL(TH_OPIVX2, th_vdiv_vx_d, TH_OP_SSS_D, H8, H8, TH_DIV)
THCALL(TH_OPIVX2, th_vremu_vx_b, TH_OP_UUU_B, H1, H1, TH_REMU)
THCALL(TH_OPIVX2, th_vremu_vx_h, TH_OP_UUU_H, H2, H2, TH_REMU)
THCALL(TH_OPIVX2, th_vremu_vx_w, TH_OP_UUU_W, H4, H4, TH_REMU)
THCALL(TH_OPIVX2, th_vremu_vx_d, TH_OP_UUU_D, H8, H8, TH_REMU)
THCALL(TH_OPIVX2, th_vrem_vx_b, TH_OP_SSS_B, H1, H1, TH_REM)
THCALL(TH_OPIVX2, th_vrem_vx_h, TH_OP_SSS_H, H2, H2, TH_REM)
THCALL(TH_OPIVX2, th_vrem_vx_w, TH_OP_SSS_W, H4, H4, TH_REM)
THCALL(TH_OPIVX2, th_vrem_vx_d, TH_OP_SSS_D, H8, H8, TH_REM)
GEN_TH_VX(th_vdivu_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vdivu_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vdivu_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vdivu_vx_d, 8, 8, clearq_th)
GEN_TH_VX(th_vdiv_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vdiv_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vdiv_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vdiv_vx_d, 8, 8, clearq_th)
GEN_TH_VX(th_vremu_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vremu_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vremu_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vremu_vx_d, 8, 8, clearq_th)
GEN_TH_VX(th_vrem_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vrem_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vrem_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vrem_vx_d, 8, 8, clearq_th)

#define TH_WOP_SUS_B int16_t, uint8_t, int8_t, uint16_t, int16_t
#define TH_WOP_SUS_H int32_t, uint16_t, int16_t, uint32_t, int32_t
#define TH_WOP_SUS_W int64_t, uint32_t, int32_t, uint64_t, int64_t
/* Vector Widening Integer Multiply Instructions */
THCALL(TH_OPIVV2, th_vwmul_vv_b, TH_WOP_SSS_B, H2, H1, H1, TH_MUL)
THCALL(TH_OPIVV2, th_vwmul_vv_h, TH_WOP_SSS_H, H4, H2, H2, TH_MUL)
THCALL(TH_OPIVV2, th_vwmul_vv_w, TH_WOP_SSS_W, H8, H4, H4, TH_MUL)
THCALL(TH_OPIVV2, th_vwmulu_vv_b, TH_WOP_UUU_B, H2, H1, H1, TH_MUL)
THCALL(TH_OPIVV2, th_vwmulu_vv_h, TH_WOP_UUU_H, H4, H2, H2, TH_MUL)
THCALL(TH_OPIVV2, th_vwmulu_vv_w, TH_WOP_UUU_W, H8, H4, H4, TH_MUL)
THCALL(TH_OPIVV2, th_vwmulsu_vv_b, TH_WOP_SUS_B, H2, H1, H1, TH_MUL)
THCALL(TH_OPIVV2, th_vwmulsu_vv_h, TH_WOP_SUS_H, H4, H2, H2, TH_MUL)
THCALL(TH_OPIVV2, th_vwmulsu_vv_w, TH_WOP_SUS_W, H8, H4, H4, TH_MUL)
GEN_TH_VV(th_vwmul_vv_b, 1, 2, clearh_th)
GEN_TH_VV(th_vwmul_vv_h, 2, 4, clearl_th)
GEN_TH_VV(th_vwmul_vv_w, 4, 8, clearq_th)
GEN_TH_VV(th_vwmulu_vv_b, 1, 2, clearh_th)
GEN_TH_VV(th_vwmulu_vv_h, 2, 4, clearl_th)
GEN_TH_VV(th_vwmulu_vv_w, 4, 8, clearq_th)
GEN_TH_VV(th_vwmulsu_vv_b, 1, 2, clearh_th)
GEN_TH_VV(th_vwmulsu_vv_h, 2, 4, clearl_th)
GEN_TH_VV(th_vwmulsu_vv_w, 4, 8, clearq_th)

THCALL(TH_OPIVX2, th_vwmul_vx_b, TH_WOP_SSS_B, H2, H1, TH_MUL)
THCALL(TH_OPIVX2, th_vwmul_vx_h, TH_WOP_SSS_H, H4, H2, TH_MUL)
THCALL(TH_OPIVX2, th_vwmul_vx_w, TH_WOP_SSS_W, H8, H4, TH_MUL)
THCALL(TH_OPIVX2, th_vwmulu_vx_b, TH_WOP_UUU_B, H2, H1, TH_MUL)
THCALL(TH_OPIVX2, th_vwmulu_vx_h, TH_WOP_UUU_H, H4, H2, TH_MUL)
THCALL(TH_OPIVX2, th_vwmulu_vx_w, TH_WOP_UUU_W, H8, H4, TH_MUL)
THCALL(TH_OPIVX2, th_vwmulsu_vx_b, TH_WOP_SUS_B, H2, H1, TH_MUL)
THCALL(TH_OPIVX2, th_vwmulsu_vx_h, TH_WOP_SUS_H, H4, H2, TH_MUL)
THCALL(TH_OPIVX2, th_vwmulsu_vx_w, TH_WOP_SUS_W, H8, H4, TH_MUL)
GEN_TH_VX(th_vwmul_vx_b, 1, 2, clearh_th)
GEN_TH_VX(th_vwmul_vx_h, 2, 4, clearl_th)
GEN_TH_VX(th_vwmul_vx_w, 4, 8, clearq_th)
GEN_TH_VX(th_vwmulu_vx_b, 1, 2, clearh_th)
GEN_TH_VX(th_vwmulu_vx_h, 2, 4, clearl_th)
GEN_TH_VX(th_vwmulu_vx_w, 4, 8, clearq_th)
GEN_TH_VX(th_vwmulsu_vx_b, 1, 2, clearh_th)
GEN_TH_VX(th_vwmulsu_vx_h, 2, 4, clearl_th)
GEN_TH_VX(th_vwmulsu_vx_w, 4, 8, clearq_th)

/* Vector Single-Width Integer Multiply-Add Instructions */
#define TH_OPIVV3(NAME, TD, T1, T2, TX1, TX2, HD, HS1, HS2, OP)   \
static void do_##NAME(void *vd, void *vs1, void *vs2, int i)       \
{                                                                  \
    TX1 s1 = *((T1 *)vs1 + HS1(i));                                \
    TX2 s2 = *((T2 *)vs2 + HS2(i));                                \
    TD d = *((TD *)vd + HD(i));                                    \
    *((TD *)vd + HD(i)) = OP(s2, s1, d);                           \
}
#define TH_MACC(N, M, D) (M * N + D)
#define TH_NMSAC(N, M, D) (-(M * N) + D)
#define TH_MADD(N, M, D) (M * D + N)
#define TH_NMSUB(N, M, D) (-(M * D) + N)
THCALL(TH_OPIVV3, th_vmacc_vv_b, TH_OP_SSS_B, H1, H1, H1, TH_MACC)
THCALL(TH_OPIVV3, th_vmacc_vv_h, TH_OP_SSS_H, H2, H2, H2, TH_MACC)
THCALL(TH_OPIVV3, th_vmacc_vv_w, TH_OP_SSS_W, H4, H4, H4, TH_MACC)
THCALL(TH_OPIVV3, th_vmacc_vv_d, TH_OP_SSS_D, H8, H8, H8, TH_MACC)
THCALL(TH_OPIVV3, th_vnmsac_vv_b, TH_OP_SSS_B, H1, H1, H1, TH_NMSAC)
THCALL(TH_OPIVV3, th_vnmsac_vv_h, TH_OP_SSS_H, H2, H2, H2, TH_NMSAC)
THCALL(TH_OPIVV3, th_vnmsac_vv_w, TH_OP_SSS_W, H4, H4, H4, TH_NMSAC)
THCALL(TH_OPIVV3, th_vnmsac_vv_d, TH_OP_SSS_D, H8, H8, H8, TH_NMSAC)
THCALL(TH_OPIVV3, th_vmadd_vv_b, TH_OP_SSS_B, H1, H1, H1, TH_MADD)
THCALL(TH_OPIVV3, th_vmadd_vv_h, TH_OP_SSS_H, H2, H2, H2, TH_MADD)
THCALL(TH_OPIVV3, th_vmadd_vv_w, TH_OP_SSS_W, H4, H4, H4, TH_MADD)
THCALL(TH_OPIVV3, th_vmadd_vv_d, TH_OP_SSS_D, H8, H8, H8, TH_MADD)
THCALL(TH_OPIVV3, th_vnmsub_vv_b, TH_OP_SSS_B, H1, H1, H1, TH_NMSUB)
THCALL(TH_OPIVV3, th_vnmsub_vv_h, TH_OP_SSS_H, H2, H2, H2, TH_NMSUB)
THCALL(TH_OPIVV3, th_vnmsub_vv_w, TH_OP_SSS_W, H4, H4, H4, TH_NMSUB)
THCALL(TH_OPIVV3, th_vnmsub_vv_d, TH_OP_SSS_D, H8, H8, H8, TH_NMSUB)
GEN_TH_VV(th_vmacc_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vmacc_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vmacc_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vmacc_vv_d, 8, 8, clearq_th)
GEN_TH_VV(th_vnmsac_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vnmsac_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vnmsac_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vnmsac_vv_d, 8, 8, clearq_th)
GEN_TH_VV(th_vmadd_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vmadd_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vmadd_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vmadd_vv_d, 8, 8, clearq_th)
GEN_TH_VV(th_vnmsub_vv_b, 1, 1, clearb_th)
GEN_TH_VV(th_vnmsub_vv_h, 2, 2, clearh_th)
GEN_TH_VV(th_vnmsub_vv_w, 4, 4, clearl_th)
GEN_TH_VV(th_vnmsub_vv_d, 8, 8, clearq_th)

#define TH_OPIVX3(NAME, TD, T1, T2, TX1, TX2, HD, HS2, OP)             \
static void do_##NAME(void *vd, target_long s1, void *vs2, int i)   \
{                                                                   \
    TX2 s2 = *((T2 *)vs2 + HS2(i));                                 \
    TD d = *((TD *)vd + HD(i));                                     \
    *((TD *)vd + HD(i)) = OP(s2, (TX1)(T1)s1, d);                   \
}

THCALL(TH_OPIVX3, th_vmacc_vx_b, TH_OP_SSS_B, H1, H1, TH_MACC)
THCALL(TH_OPIVX3, th_vmacc_vx_h, TH_OP_SSS_H, H2, H2, TH_MACC)
THCALL(TH_OPIVX3, th_vmacc_vx_w, TH_OP_SSS_W, H4, H4, TH_MACC)
THCALL(TH_OPIVX3, th_vmacc_vx_d, TH_OP_SSS_D, H8, H8, TH_MACC)
THCALL(TH_OPIVX3, th_vnmsac_vx_b, TH_OP_SSS_B, H1, H1, TH_NMSAC)
THCALL(TH_OPIVX3, th_vnmsac_vx_h, TH_OP_SSS_H, H2, H2, TH_NMSAC)
THCALL(TH_OPIVX3, th_vnmsac_vx_w, TH_OP_SSS_W, H4, H4, TH_NMSAC)
THCALL(TH_OPIVX3, th_vnmsac_vx_d, TH_OP_SSS_D, H8, H8, TH_NMSAC)
THCALL(TH_OPIVX3, th_vmadd_vx_b, TH_OP_SSS_B, H1, H1, TH_MADD)
THCALL(TH_OPIVX3, th_vmadd_vx_h, TH_OP_SSS_H, H2, H2, TH_MADD)
THCALL(TH_OPIVX3, th_vmadd_vx_w, TH_OP_SSS_W, H4, H4, TH_MADD)
THCALL(TH_OPIVX3, th_vmadd_vx_d, TH_OP_SSS_D, H8, H8, TH_MADD)
THCALL(TH_OPIVX3, th_vnmsub_vx_b, TH_OP_SSS_B, H1, H1, TH_NMSUB)
THCALL(TH_OPIVX3, th_vnmsub_vx_h, TH_OP_SSS_H, H2, H2, TH_NMSUB)
THCALL(TH_OPIVX3, th_vnmsub_vx_w, TH_OP_SSS_W, H4, H4, TH_NMSUB)
THCALL(TH_OPIVX3, th_vnmsub_vx_d, TH_OP_SSS_D, H8, H8, TH_NMSUB)
GEN_TH_VX(th_vmacc_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vmacc_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vmacc_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vmacc_vx_d, 8, 8, clearq_th)
GEN_TH_VX(th_vnmsac_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vnmsac_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vnmsac_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vnmsac_vx_d, 8, 8, clearq_th)
GEN_TH_VX(th_vmadd_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vmadd_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vmadd_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vmadd_vx_d, 8, 8, clearq_th)
GEN_TH_VX(th_vnmsub_vx_b, 1, 1, clearb_th)
GEN_TH_VX(th_vnmsub_vx_h, 2, 2, clearh_th)
GEN_TH_VX(th_vnmsub_vx_w, 4, 4, clearl_th)
GEN_TH_VX(th_vnmsub_vx_d, 8, 8, clearq_th)

#define TH_WOP_SSU_B int16_t, int8_t, uint8_t, int16_t, uint16_t
#define TH_WOP_SSU_H int32_t, int16_t, uint16_t, int32_t, uint32_t
#define TH_WOP_SSU_W int64_t, int32_t, uint32_t, int64_t, uint64_t
/* Vector Widening Integer Multiply-Add Instructions */
THCALL(TH_OPIVV3, th_vwmaccu_vv_b, TH_WOP_UUU_B, H2, H1, H1, TH_MACC)
THCALL(TH_OPIVV3, th_vwmaccu_vv_h, TH_WOP_UUU_H, H4, H2, H2, TH_MACC)
THCALL(TH_OPIVV3, th_vwmaccu_vv_w, TH_WOP_UUU_W, H8, H4, H4, TH_MACC)
THCALL(TH_OPIVV3, th_vwmacc_vv_b, TH_WOP_SSS_B, H2, H1, H1, TH_MACC)
THCALL(TH_OPIVV3, th_vwmacc_vv_h, TH_WOP_SSS_H, H4, H2, H2, TH_MACC)
THCALL(TH_OPIVV3, th_vwmacc_vv_w, TH_WOP_SSS_W, H8, H4, H4, TH_MACC)
THCALL(TH_OPIVV3, th_vwmaccsu_vv_b, TH_WOP_SSU_B, H2, H1, H1, TH_MACC)
THCALL(TH_OPIVV3, th_vwmaccsu_vv_h, TH_WOP_SSU_H, H4, H2, H2, TH_MACC)
THCALL(TH_OPIVV3, th_vwmaccsu_vv_w, TH_WOP_SSU_W, H8, H4, H4, TH_MACC)
GEN_TH_VV(th_vwmaccu_vv_b, 1, 2, clearh_th)
GEN_TH_VV(th_vwmaccu_vv_h, 2, 4, clearl_th)
GEN_TH_VV(th_vwmaccu_vv_w, 4, 8, clearq_th)
GEN_TH_VV(th_vwmacc_vv_b, 1, 2, clearh_th)
GEN_TH_VV(th_vwmacc_vv_h, 2, 4, clearl_th)
GEN_TH_VV(th_vwmacc_vv_w, 4, 8, clearq_th)
GEN_TH_VV(th_vwmaccsu_vv_b, 1, 2, clearh_th)
GEN_TH_VV(th_vwmaccsu_vv_h, 2, 4, clearl_th)
GEN_TH_VV(th_vwmaccsu_vv_w, 4, 8, clearq_th)

THCALL(TH_OPIVX3, th_vwmaccu_vx_b, TH_WOP_UUU_B, H2, H1, TH_MACC)
THCALL(TH_OPIVX3, th_vwmaccu_vx_h, TH_WOP_UUU_H, H4, H2, TH_MACC)
THCALL(TH_OPIVX3, th_vwmaccu_vx_w, TH_WOP_UUU_W, H8, H4, TH_MACC)
THCALL(TH_OPIVX3, th_vwmacc_vx_b, TH_WOP_SSS_B, H2, H1, TH_MACC)
THCALL(TH_OPIVX3, th_vwmacc_vx_h, TH_WOP_SSS_H, H4, H2, TH_MACC)
THCALL(TH_OPIVX3, th_vwmacc_vx_w, TH_WOP_SSS_W, H8, H4, TH_MACC)
THCALL(TH_OPIVX3, th_vwmaccsu_vx_b, TH_WOP_SSU_B, H2, H1, TH_MACC)
THCALL(TH_OPIVX3, th_vwmaccsu_vx_h, TH_WOP_SSU_H, H4, H2, TH_MACC)
THCALL(TH_OPIVX3, th_vwmaccsu_vx_w, TH_WOP_SSU_W, H8, H4, TH_MACC)
THCALL(TH_OPIVX3, th_vwmaccus_vx_b, TH_WOP_SUS_B, H2, H1, TH_MACC)
THCALL(TH_OPIVX3, th_vwmaccus_vx_h, TH_WOP_SUS_H, H4, H2, TH_MACC)
THCALL(TH_OPIVX3, th_vwmaccus_vx_w, TH_WOP_SUS_W, H8, H4, TH_MACC)
GEN_TH_VX(th_vwmaccu_vx_b, 1, 2, clearh_th)
GEN_TH_VX(th_vwmaccu_vx_h, 2, 4, clearl_th)
GEN_TH_VX(th_vwmaccu_vx_w, 4, 8, clearq_th)
GEN_TH_VX(th_vwmacc_vx_b, 1, 2, clearh_th)
GEN_TH_VX(th_vwmacc_vx_h, 2, 4, clearl_th)
GEN_TH_VX(th_vwmacc_vx_w, 4, 8, clearq_th)
GEN_TH_VX(th_vwmaccsu_vx_b, 1, 2, clearh_th)
GEN_TH_VX(th_vwmaccsu_vx_h, 2, 4, clearl_th)
GEN_TH_VX(th_vwmaccsu_vx_w, 4, 8, clearq_th)
GEN_TH_VX(th_vwmaccus_vx_b, 1, 2, clearh_th)
GEN_TH_VX(th_vwmaccus_vx_h, 2, 4, clearl_th)
GEN_TH_VX(th_vwmaccus_vx_w, 4, 8, clearq_th)

/* Vector Integer Merge and Move Instructions */
/*
 * The funtions below of VMV and vmerge are all the copy of RVV1.0 functions,
 * except:
 * 1) different desc encoding
 * 2) different tail/masked element process policy
 * 3) different mask layout
 */
#define GEN_TH_VMV_VV(NAME, ETYPE, H, CLEAR_FN)                      \
void HELPER(NAME)(void *vd, void *vs1, CPURISCVState *env,           \
                  uint32_t desc)                                     \
{                                                                    \
    uint32_t vl = env->vl;                                           \
    uint32_t esz = sizeof(ETYPE);                                    \
    uint32_t vlmax = th_maxsz(desc) / esz;                           \
    uint32_t i;                                                      \
                                                                     \
    for (i = env->vstart; i < vl; i++) {                             \
        ETYPE s1 = *((ETYPE *)vs1 + H(i));                           \
        *((ETYPE *)vd + H(i)) = s1;                                  \
    }                                                                \
    env->vstart = 0;                                                 \
    CLEAR_FN(vd, vl, vl * esz, vlmax * esz);                         \
}

GEN_TH_VMV_VV(th_vmv_v_v_b, int8_t,  H1, clearb_th)
GEN_TH_VMV_VV(th_vmv_v_v_h, int16_t, H2, clearh_th)
GEN_TH_VMV_VV(th_vmv_v_v_w, int32_t, H4, clearl_th)
GEN_TH_VMV_VV(th_vmv_v_v_d, int64_t, H8, clearq_th)

#define GEN_TH_VMV_VX(NAME, ETYPE, H, CLEAR_FN)                      \
void HELPER(NAME)(void *vd, uint64_t s1, CPURISCVState *env,         \
                  uint32_t desc)                                     \
{                                                                    \
    uint32_t vl = env->vl;                                           \
    uint32_t esz = sizeof(ETYPE);                                    \
    uint32_t vlmax = th_maxsz(desc) / esz;                           \
    uint32_t i;                                                      \
                                                                     \
    for (i = env->vstart; i < vl; i++) {                             \
        *((ETYPE *)vd + H(i)) = (ETYPE)s1;                           \
    }                                                                \
    env->vstart = 0;                                                 \
    CLEAR_FN(vd, vl, vl * esz, vlmax * esz);                         \
}

GEN_TH_VMV_VX(th_vmv_v_x_b, int8_t,  H1, clearb_th)
GEN_TH_VMV_VX(th_vmv_v_x_h, int16_t, H2, clearh_th)
GEN_TH_VMV_VX(th_vmv_v_x_w, int32_t, H4, clearl_th)
GEN_TH_VMV_VX(th_vmv_v_x_d, int64_t, H8, clearq_th)

#define GEN_TH_VMERGE_VV(NAME, ETYPE, H, CLEAR_FN)                   \
void HELPER(NAME)(void *vd, void *v0, void *vs1, void *vs2,          \
                  CPURISCVState *env, uint32_t desc)                 \
{                                                                    \
    uint32_t mlen = th_mlen(desc);                                   \
    uint32_t vl = env->vl;                                           \
    uint32_t esz = sizeof(ETYPE);                                    \
    uint32_t vlmax = th_maxsz(desc) / esz;                           \
    uint32_t i;                                                      \
                                                                     \
    for (i = env->vstart; i < vl; i++) {                             \
        ETYPE *vt = (!th_elem_mask(v0, mlen, i) ? vs2 : vs1);        \
        *((ETYPE *)vd + H(i)) = *(vt + H(i));                        \
    }                                                                \
    env->vstart = 0;                                                 \
    CLEAR_FN(vd, vl, vl * esz, vlmax * esz);                         \
}

GEN_TH_VMERGE_VV(th_vmerge_vvm_b, int8_t,  H1, clearb_th)
GEN_TH_VMERGE_VV(th_vmerge_vvm_h, int16_t, H2, clearh_th)
GEN_TH_VMERGE_VV(th_vmerge_vvm_w, int32_t, H4, clearl_th)
GEN_TH_VMERGE_VV(th_vmerge_vvm_d, int64_t, H8, clearq_th)

#define GEN_TH_VMERGE_VX(NAME, ETYPE, H, CLEAR_FN)                   \
void HELPER(NAME)(void *vd, void *v0, target_ulong s1,               \
                  void *vs2, CPURISCVState *env, uint32_t desc)      \
{                                                                    \
    uint32_t mlen = th_mlen(desc);                                   \
    uint32_t vl = env->vl;                                           \
    uint32_t esz = sizeof(ETYPE);                                    \
    uint32_t vlmax = th_maxsz(desc) / esz;                           \
    uint32_t i;                                                      \
                                                                     \
    for (i = env->vstart; i < vl; i++) {                             \
        ETYPE s2 = *((ETYPE *)vs2 + H(i));                           \
        ETYPE d = (!th_elem_mask(v0, mlen, i) ? s2 :                 \
                   (ETYPE)(target_long)s1);                          \
        *((ETYPE *)vd + H(i)) = d;                                   \
    }                                                                \
    env->vstart = 0;                                                 \
    CLEAR_FN(vd, vl, vl * esz, vlmax * esz);                         \
}

GEN_TH_VMERGE_VX(th_vmerge_vxm_b, int8_t,  H1, clearb_th)
GEN_TH_VMERGE_VX(th_vmerge_vxm_h, int16_t, H2, clearh_th)
GEN_TH_VMERGE_VX(th_vmerge_vxm_w, int32_t, H4, clearl_th)
GEN_TH_VMERGE_VX(th_vmerge_vxm_d, int64_t, H8, clearq_th)

/*
 *** Vector Fixed-Point Arithmetic Instructions
 */

/* Vector Single-Width Saturating Add and Subtract */

/*
 * As fixed point instructions probably have round mode and saturation,
 * define common macros for fixed point here.
 */
typedef void opivv2_rm_fn_th(void *vd, void *vs1, void *vs2, int i,
                             CPURISCVState *env, int vxrm);

/*
 * The functions of fix-point operations below are just the copies of
 * functions in RVV1.0.
 * The changes in these functions are:
 * 1) different desc encoding
 * 2) different tail/masked element process policy
 * 3) different mask layout
 */
#define TH_OPIVV2_RM(NAME, TD, T1, T2, TX1, TX2, HD, HS1, HS2, OP)  \
static inline void                                                  \
do_##NAME(void *vd, void *vs1, void *vs2, int i,                    \
          CPURISCVState *env, int vxrm)                             \
{                                                                   \
    TX1 s1 = *((T1 *)vs1 + HS1(i));                                 \
    TX2 s2 = *((T2 *)vs2 + HS2(i));                                 \
    *((TD *)vd + HD(i)) = OP(env, vxrm, s2, s1);                    \
}

static inline void
th_vv_rm_1(void *vd, void *v0, void *vs1, void *vs2,
           CPURISCVState *env,
           uint32_t vl, uint32_t vm, uint32_t mlen, int vxrm,
           opivv2_rm_fn_th *fn)
{
    for (uint32_t i = env->vstart; i < vl; i++) {
        if (!vm && !th_elem_mask(v0, mlen, i)) {
            continue;
        }
        fn(vd, vs1, vs2, i, env, vxrm);
    }
    env->vstart = 0;
}

static inline void
th_vv_rm_2(void *vd, void *v0, void *vs1, void *vs2,
           CPURISCVState *env,
           uint32_t desc, uint32_t esz, uint32_t dsz,
           opivv2_rm_fn_th *fn, clear_fn *clearfn)
{
    uint32_t vlmax = th_maxsz(desc) / esz;
    uint32_t mlen = th_mlen(desc);
    uint32_t vm = th_vm(desc);
    uint32_t vl = env->vl;

    switch (env->vxrm) {
    case 0: /* rnu */
        th_vv_rm_1(vd, v0, vs1, vs2,
                   env, vl, vm, mlen, 0, fn);
        break;
    case 1: /* rne */
        th_vv_rm_1(vd, v0, vs1, vs2,
                   env, vl, vm, mlen, 1, fn);
        break;
    case 2: /* rdn */
        th_vv_rm_1(vd, v0, vs1, vs2,
                   env, vl, vm, mlen, 2, fn);
        break;
    default: /* rod */
        th_vv_rm_1(vd, v0, vs1, vs2,
                   env, vl, vm, mlen, 3, fn);
        break;
    }

    clearfn(vd, vl, vl * dsz,  vlmax * dsz);
}

#define GEN_TH_SADD_FUNC(NAME, TYPE)                \
static TYPE th_##NAME(CPURISCVState *env, int vxrm, \
                      TYPE s2, TYPE s1)             \
{                                                   \
    return NAME(env, vxrm, s2, s1);                 \
}

GEN_TH_SADD_FUNC(saddu8, uint8_t)
GEN_TH_SADD_FUNC(saddu16, uint16_t)
GEN_TH_SADD_FUNC(saddu32, uint32_t)
GEN_TH_SADD_FUNC(saddu64, uint64_t)
GEN_TH_SADD_FUNC(sadd8, int8_t)
GEN_TH_SADD_FUNC(sadd16, int16_t)
GEN_TH_SADD_FUNC(sadd32, int32_t)
GEN_TH_SADD_FUNC(sadd64, int64_t)
GEN_TH_SADD_FUNC(ssub8, int8_t)
GEN_TH_SADD_FUNC(ssub16, int16_t)
GEN_TH_SADD_FUNC(ssub32, int32_t)
GEN_TH_SADD_FUNC(ssub64, int64_t)
GEN_TH_SADD_FUNC(ssubu8, uint8_t)
GEN_TH_SADD_FUNC(ssubu16, uint16_t)
GEN_TH_SADD_FUNC(ssubu32, uint32_t)
GEN_TH_SADD_FUNC(ssubu64, uint64_t)

/* generate helpers for fixed point instructions with OPIVV format */
#define GEN_TH_VV_RM(NAME, ESZ, DSZ, CLEAR_FN)                  \
void HELPER(NAME)(void *vd, void *v0, void *vs1, void *vs2,     \
                  CPURISCVState *env, uint32_t desc)            \
{                                                               \
    th_vv_rm_2(vd, v0, vs1, vs2, env, desc, ESZ, DSZ,           \
               do_##NAME, CLEAR_FN);                            \
}

THCALL(TH_OPIVV2_RM, th_vsaddu_vv_b, TH_OP_UUU_B, H1, H1, H1, th_saddu8)
THCALL(TH_OPIVV2_RM, th_vsaddu_vv_h, TH_OP_UUU_H, H2, H2, H2, th_saddu16)
THCALL(TH_OPIVV2_RM, th_vsaddu_vv_w, TH_OP_UUU_W, H4, H4, H4, th_saddu32)
THCALL(TH_OPIVV2_RM, th_vsaddu_vv_d, TH_OP_UUU_D, H8, H8, H8, th_saddu64)
GEN_TH_VV_RM(th_vsaddu_vv_b, 1, 1, clearb_th)
GEN_TH_VV_RM(th_vsaddu_vv_h, 2, 2, clearh_th)
GEN_TH_VV_RM(th_vsaddu_vv_w, 4, 4, clearl_th)
GEN_TH_VV_RM(th_vsaddu_vv_d, 8, 8, clearq_th)

typedef void opivx2_rm_fn_th(void *vd, target_long s1, void *vs2, int i,
                             CPURISCVState *env, int vxrm);

#define TH_OPIVX2_RM(NAME, TD, T1, T2, TX1, TX2, HD, HS2, OP)       \
static inline void                                                  \
do_##NAME(void *vd, target_long s1, void *vs2, int i,               \
          CPURISCVState *env, int vxrm)                             \
{                                                                   \
    TX2 s2 = *((T2 *)vs2 + HS2(i));                                 \
    *((TD *)vd + HD(i)) = OP(env, vxrm, s2, (TX1)(T1)s1);           \
}

static inline void
th_vx_rm_1(void *vd, void *v0, target_long s1, void *vs2,
           CPURISCVState *env,
           uint32_t vl, uint32_t vm, uint32_t mlen, int vxrm,
           opivx2_rm_fn_th *fn)
{
    for (uint32_t i = env->vstart; i < vl; i++) {
        if (!vm && !th_elem_mask(v0, mlen, i)) {
            continue;
        }
        fn(vd, s1, vs2, i, env, vxrm);
    }
    env->vstart = 0;
}

static inline void
th_vx_rm_2(void *vd, void *v0, target_long s1, void *vs2,
           CPURISCVState *env,
           uint32_t desc, uint32_t esz, uint32_t dsz,
           opivx2_rm_fn_th *fn, clear_fn *clearfn)
{
    uint32_t vlmax = th_maxsz(desc) / esz;
    uint32_t mlen = th_mlen(desc);
    uint32_t vm = th_vm(desc);
    uint32_t vl = env->vl;

    switch (env->vxrm) {
    case 0: /* rnu */
        th_vx_rm_1(vd, v0, s1, vs2,
                   env, vl, vm, mlen, 0, fn);
        break;
    case 1: /* rne */
        th_vx_rm_1(vd, v0, s1, vs2,
                   env, vl, vm, mlen, 1, fn);
        break;
    case 2: /* rdn */
        th_vx_rm_1(vd, v0, s1, vs2,
                   env, vl, vm, mlen, 2, fn);
        break;
    default: /* rod */
        th_vx_rm_1(vd, v0, s1, vs2,
                   env, vl, vm, mlen, 3, fn);
        break;
    }

    clearfn(vd, vl, vl * dsz,  vlmax * dsz);
}

/* generate helpers for fixed point instructions with OPIVX format */
#define GEN_TH_VX_RM(NAME, ESZ, DSZ, CLEAR_FN)            \
void HELPER(NAME)(void *vd, void *v0, target_ulong s1,    \
        void *vs2, CPURISCVState *env, uint32_t desc)     \
{                                                         \
    th_vx_rm_2(vd, v0, s1, vs2, env, desc, ESZ, DSZ,      \
               do_##NAME, CLEAR_FN);                      \
}

THCALL(TH_OPIVX2_RM, th_vsaddu_vx_b, TH_OP_UUU_B, H1, H1, th_saddu8)
THCALL(TH_OPIVX2_RM, th_vsaddu_vx_h, TH_OP_UUU_H, H2, H2, th_saddu16)
THCALL(TH_OPIVX2_RM, th_vsaddu_vx_w, TH_OP_UUU_W, H4, H4, th_saddu32)
THCALL(TH_OPIVX2_RM, th_vsaddu_vx_d, TH_OP_UUU_D, H8, H8, th_saddu64)
GEN_TH_VX_RM(th_vsaddu_vx_b, 1, 1, clearb_th)
GEN_TH_VX_RM(th_vsaddu_vx_h, 2, 2, clearh_th)
GEN_TH_VX_RM(th_vsaddu_vx_w, 4, 4, clearl_th)
GEN_TH_VX_RM(th_vsaddu_vx_d, 8, 8, clearq_th)

THCALL(TH_OPIVV2_RM, th_vsadd_vv_b, TH_OP_SSS_B, H1, H1, H1, th_sadd8)
THCALL(TH_OPIVV2_RM, th_vsadd_vv_h, TH_OP_SSS_H, H2, H2, H2, th_sadd16)
THCALL(TH_OPIVV2_RM, th_vsadd_vv_w, TH_OP_SSS_W, H4, H4, H4, th_sadd32)
THCALL(TH_OPIVV2_RM, th_vsadd_vv_d, TH_OP_SSS_D, H8, H8, H8, th_sadd64)
GEN_TH_VV_RM(th_vsadd_vv_b, 1, 1, clearb_th)
GEN_TH_VV_RM(th_vsadd_vv_h, 2, 2, clearh_th)
GEN_TH_VV_RM(th_vsadd_vv_w, 4, 4, clearl_th)
GEN_TH_VV_RM(th_vsadd_vv_d, 8, 8, clearq_th)

THCALL(TH_OPIVX2_RM, th_vsadd_vx_b, TH_OP_SSS_B, H1, H1, th_sadd8)
THCALL(TH_OPIVX2_RM, th_vsadd_vx_h, TH_OP_SSS_H, H2, H2, th_sadd16)
THCALL(TH_OPIVX2_RM, th_vsadd_vx_w, TH_OP_SSS_W, H4, H4, th_sadd32)
THCALL(TH_OPIVX2_RM, th_vsadd_vx_d, TH_OP_SSS_D, H8, H8, th_sadd64)
GEN_TH_VX_RM(th_vsadd_vx_b, 1, 1, clearb_th)
GEN_TH_VX_RM(th_vsadd_vx_h, 2, 2, clearh_th)
GEN_TH_VX_RM(th_vsadd_vx_w, 4, 4, clearl_th)
GEN_TH_VX_RM(th_vsadd_vx_d, 8, 8, clearq_th)

THCALL(TH_OPIVV2_RM, th_vssubu_vv_b, TH_OP_UUU_B, H1, H1, H1, th_ssubu8)
THCALL(TH_OPIVV2_RM, th_vssubu_vv_h, TH_OP_UUU_H, H2, H2, H2, th_ssubu16)
THCALL(TH_OPIVV2_RM, th_vssubu_vv_w, TH_OP_UUU_W, H4, H4, H4, th_ssubu32)
THCALL(TH_OPIVV2_RM, th_vssubu_vv_d, TH_OP_UUU_D, H8, H8, H8, th_ssubu64)
GEN_TH_VV_RM(th_vssubu_vv_b, 1, 1, clearb_th)
GEN_TH_VV_RM(th_vssubu_vv_h, 2, 2, clearh_th)
GEN_TH_VV_RM(th_vssubu_vv_w, 4, 4, clearl_th)
GEN_TH_VV_RM(th_vssubu_vv_d, 8, 8, clearq_th)

THCALL(TH_OPIVX2_RM, th_vssubu_vx_b, TH_OP_UUU_B, H1, H1, th_ssubu8)
THCALL(TH_OPIVX2_RM, th_vssubu_vx_h, TH_OP_UUU_H, H2, H2, th_ssubu16)
THCALL(TH_OPIVX2_RM, th_vssubu_vx_w, TH_OP_UUU_W, H4, H4, th_ssubu32)
THCALL(TH_OPIVX2_RM, th_vssubu_vx_d, TH_OP_UUU_D, H8, H8, th_ssubu64)
GEN_TH_VX_RM(th_vssubu_vx_b, 1, 1, clearb_th)
GEN_TH_VX_RM(th_vssubu_vx_h, 2, 2, clearh_th)
GEN_TH_VX_RM(th_vssubu_vx_w, 4, 4, clearl_th)
GEN_TH_VX_RM(th_vssubu_vx_d, 8, 8, clearq_th)

THCALL(TH_OPIVV2_RM, th_vssub_vv_b, TH_OP_SSS_B, H1, H1, H1, th_ssub8)
THCALL(TH_OPIVV2_RM, th_vssub_vv_h, TH_OP_SSS_H, H2, H2, H2, th_ssub16)
THCALL(TH_OPIVV2_RM, th_vssub_vv_w, TH_OP_SSS_W, H4, H4, H4, th_ssub32)
THCALL(TH_OPIVV2_RM, th_vssub_vv_d, TH_OP_SSS_D, H8, H8, H8, th_ssub64)
GEN_TH_VV_RM(th_vssub_vv_b, 1, 1, clearb_th)
GEN_TH_VV_RM(th_vssub_vv_h, 2, 2, clearh_th)
GEN_TH_VV_RM(th_vssub_vv_w, 4, 4, clearl_th)
GEN_TH_VV_RM(th_vssub_vv_d, 8, 8, clearq_th)

THCALL(TH_OPIVX2_RM, th_vssub_vx_b, TH_OP_SSS_B, H1, H1, th_ssub8)
THCALL(TH_OPIVX2_RM, th_vssub_vx_h, TH_OP_SSS_H, H2, H2, th_ssub16)
THCALL(TH_OPIVX2_RM, th_vssub_vx_w, TH_OP_SSS_W, H4, H4, th_ssub32)
THCALL(TH_OPIVX2_RM, th_vssub_vx_d, TH_OP_SSS_D, H8, H8, th_ssub64)
GEN_TH_VX_RM(th_vssub_vx_b, 1, 1, clearb_th)
GEN_TH_VX_RM(th_vssub_vx_h, 2, 2, clearh_th)
GEN_TH_VX_RM(th_vssub_vx_w, 4, 4, clearl_th)
GEN_TH_VX_RM(th_vssub_vx_d, 8, 8, clearq_th)


GEN_TH_SADD_FUNC(aadd32, int32_t)
GEN_TH_SADD_FUNC(aadd64, int64_t)
GEN_TH_SADD_FUNC(asub32, int32_t)
GEN_TH_SADD_FUNC(asub64, int64_t)

THCALL(TH_OPIVV2_RM, th_vaadd_vv_b, TH_OP_SSS_B, H1, H1, H1, th_aadd32)
THCALL(TH_OPIVV2_RM, th_vaadd_vv_h, TH_OP_SSS_H, H2, H2, H2, th_aadd32)
THCALL(TH_OPIVV2_RM, th_vaadd_vv_w, TH_OP_SSS_W, H4, H4, H4, th_aadd32)
THCALL(TH_OPIVV2_RM, th_vaadd_vv_d, TH_OP_SSS_D, H8, H8, H8, th_aadd64)
GEN_TH_VV_RM(th_vaadd_vv_b, 1, 1, clearb_th)
GEN_TH_VV_RM(th_vaadd_vv_h, 2, 2, clearh_th)
GEN_TH_VV_RM(th_vaadd_vv_w, 4, 4, clearl_th)
GEN_TH_VV_RM(th_vaadd_vv_d, 8, 8, clearq_th)

THCALL(TH_OPIVX2_RM, th_vaadd_vx_b, TH_OP_SSS_B, H1, H1, th_aadd32)
THCALL(TH_OPIVX2_RM, th_vaadd_vx_h, TH_OP_SSS_H, H2, H2, th_aadd32)
THCALL(TH_OPIVX2_RM, th_vaadd_vx_w, TH_OP_SSS_W, H4, H4, th_aadd32)
THCALL(TH_OPIVX2_RM, th_vaadd_vx_d, TH_OP_SSS_D, H8, H8, th_aadd64)
GEN_TH_VX_RM(th_vaadd_vx_b, 1, 1, clearb_th)
GEN_TH_VX_RM(th_vaadd_vx_h, 2, 2, clearh_th)
GEN_TH_VX_RM(th_vaadd_vx_w, 4, 4, clearl_th)
GEN_TH_VX_RM(th_vaadd_vx_d, 8, 8, clearq_th)

THCALL(TH_OPIVV2_RM, th_vasub_vv_b, TH_OP_SSS_B, H1, H1, H1, th_asub32)
THCALL(TH_OPIVV2_RM, th_vasub_vv_h, TH_OP_SSS_H, H2, H2, H2, th_asub32)
THCALL(TH_OPIVV2_RM, th_vasub_vv_w, TH_OP_SSS_W, H4, H4, H4, th_asub32)
THCALL(TH_OPIVV2_RM, th_vasub_vv_d, TH_OP_SSS_D, H8, H8, H8, th_asub64)
GEN_TH_VV_RM(th_vasub_vv_b, 1, 1, clearb_th)
GEN_TH_VV_RM(th_vasub_vv_h, 2, 2, clearh_th)
GEN_TH_VV_RM(th_vasub_vv_w, 4, 4, clearl_th)
GEN_TH_VV_RM(th_vasub_vv_d, 8, 8, clearq_th)

THCALL(TH_OPIVX2_RM, th_vasub_vx_b, TH_OP_SSS_B, H1, H1, th_asub32)
THCALL(TH_OPIVX2_RM, th_vasub_vx_h, TH_OP_SSS_H, H2, H2, th_asub32)
THCALL(TH_OPIVX2_RM, th_vasub_vx_w, TH_OP_SSS_W, H4, H4, th_asub32)
THCALL(TH_OPIVX2_RM, th_vasub_vx_d, TH_OP_SSS_D, H8, H8, th_asub64)
GEN_TH_VX_RM(th_vasub_vx_b, 1, 1, clearb_th)
GEN_TH_VX_RM(th_vasub_vx_h, 2, 2, clearh_th)
GEN_TH_VX_RM(th_vasub_vx_w, 4, 4, clearl_th)
GEN_TH_VX_RM(th_vasub_vx_d, 8, 8, clearq_th)

/* Vector Single-Width Fractional Multiply with Rounding and Saturation */
GEN_TH_SADD_FUNC(vsmul8, int8_t)
GEN_TH_SADD_FUNC(vsmul16, int16_t)
GEN_TH_SADD_FUNC(vsmul32, int32_t)
GEN_TH_SADD_FUNC(vsmul64, int64_t)
THCALL(TH_OPIVV2_RM, th_vsmul_vv_b, TH_OP_SSS_B, H1, H1, H1, th_vsmul8)
THCALL(TH_OPIVV2_RM, th_vsmul_vv_h, TH_OP_SSS_H, H2, H2, H2, th_vsmul16)
THCALL(TH_OPIVV2_RM, th_vsmul_vv_w, TH_OP_SSS_W, H4, H4, H4, th_vsmul32)
THCALL(TH_OPIVV2_RM, th_vsmul_vv_d, TH_OP_SSS_D, H8, H8, H8, th_vsmul64)
GEN_TH_VV_RM(th_vsmul_vv_b, 1, 1, clearb_th)
GEN_TH_VV_RM(th_vsmul_vv_h, 2, 2, clearh_th)
GEN_TH_VV_RM(th_vsmul_vv_w, 4, 4, clearl_th)
GEN_TH_VV_RM(th_vsmul_vv_d, 8, 8, clearq_th)

THCALL(TH_OPIVX2_RM, th_vsmul_vx_b, TH_OP_SSS_B, H1, H1, th_vsmul8)
THCALL(TH_OPIVX2_RM, th_vsmul_vx_h, TH_OP_SSS_H, H2, H2, th_vsmul16)
THCALL(TH_OPIVX2_RM, th_vsmul_vx_w, TH_OP_SSS_W, H4, H4, th_vsmul32)
THCALL(TH_OPIVX2_RM, th_vsmul_vx_d, TH_OP_SSS_D, H8, H8, th_vsmul64)
GEN_TH_VX_RM(th_vsmul_vx_b, 1, 1, clearb_th)
GEN_TH_VX_RM(th_vsmul_vx_h, 2, 2, clearh_th)
GEN_TH_VX_RM(th_vsmul_vx_w, 4, 4, clearl_th)
GEN_TH_VX_RM(th_vsmul_vx_d, 8, 8, clearq_th)

/* Vector Widening Saturating Scaled Multiply-Add */

static uint8_t th_get_round(int vxrm, uint64_t v, uint8_t shift)
{
    return get_round(vxrm, v, shift);
}

static inline uint16_t
vwsmaccu8(CPURISCVState *env, int vxrm, uint8_t a, uint8_t b,
          uint16_t c)
{
    uint8_t round;
    uint16_t res = (uint16_t)a * b;

    round = th_get_round(vxrm, res, 4);
    res   = (res >> 4) + round;
    return saddu16(env, vxrm, c, res);
}

static inline uint32_t
vwsmaccu16(CPURISCVState *env, int vxrm, uint16_t a, uint16_t b,
           uint32_t c)
{
    uint8_t round;
    uint32_t res = (uint32_t)a * b;

    round = th_get_round(vxrm, res, 8);
    res   = (res >> 8) + round;
    return saddu32(env, vxrm, c, res);
}

static inline uint64_t
vwsmaccu32(CPURISCVState *env, int vxrm, uint32_t a, uint32_t b,
           uint64_t c)
{
    uint8_t round;
    uint64_t res = (uint64_t)a * b;

    round = th_get_round(vxrm, res, 16);
    res   = (res >> 16) + round;
    return saddu64(env, vxrm, c, res);
}

#define TH_OPIVV3_RM(NAME, TD, T1, T2, TX1, TX2, HD, HS1, HS2, OP) \
static inline void                                                 \
do_##NAME(void *vd, void *vs1, void *vs2, int i,                   \
          CPURISCVState *env, int vxrm)                            \
{                                                                  \
    TX1 s1 = *((T1 *)vs1 + HS1(i));                                \
    TX2 s2 = *((T2 *)vs2 + HS2(i));                                \
    TD d = *((TD *)vd + HD(i));                                    \
    *((TD *)vd + HD(i)) = OP(env, vxrm, s2, s1, d);                \
}

THCALL(TH_OPIVV3_RM, th_vwsmaccu_vv_b, TH_WOP_UUU_B, H2, H1, H1, vwsmaccu8)
THCALL(TH_OPIVV3_RM, th_vwsmaccu_vv_h, TH_WOP_UUU_H, H4, H2, H2, vwsmaccu16)
THCALL(TH_OPIVV3_RM, th_vwsmaccu_vv_w, TH_WOP_UUU_W, H8, H4, H4, vwsmaccu32)
GEN_TH_VV_RM(th_vwsmaccu_vv_b, 1, 2, clearh_th)
GEN_TH_VV_RM(th_vwsmaccu_vv_h, 2, 4, clearl_th)
GEN_TH_VV_RM(th_vwsmaccu_vv_w, 4, 8, clearq_th)

#define TH_OPIVX3_RM(NAME, TD, T1, T2, TX1, TX2, HD, HS2, OP)      \
static inline void                                                 \
do_##NAME(void *vd, target_long s1, void *vs2, int i,              \
          CPURISCVState *env, int vxrm)                            \
{                                                                  \
    TX2 s2 = *((T2 *)vs2 + HS2(i));                                \
    TD d = *((TD *)vd + HD(i));                                    \
    *((TD *)vd + HD(i)) = OP(env, vxrm, s2, (TX1)(T1)s1, d);       \
}

THCALL(TH_OPIVX3_RM, th_vwsmaccu_vx_b, TH_WOP_UUU_B, H2, H1, vwsmaccu8)
THCALL(TH_OPIVX3_RM, th_vwsmaccu_vx_h, TH_WOP_UUU_H, H4, H2, vwsmaccu16)
THCALL(TH_OPIVX3_RM, th_vwsmaccu_vx_w, TH_WOP_UUU_W, H8, H4, vwsmaccu32)
GEN_TH_VX_RM(th_vwsmaccu_vx_b, 1, 2, clearh_th)
GEN_TH_VX_RM(th_vwsmaccu_vx_h, 2, 4, clearl_th)
GEN_TH_VX_RM(th_vwsmaccu_vx_w, 4, 8, clearq_th)

static inline int16_t
vwsmacc8(CPURISCVState *env, int vxrm, int8_t a, int8_t b, int16_t c)
{
    uint8_t round;
    int16_t res = (int16_t)a * b;

    round = th_get_round(vxrm, res, 4);
    res   = (res >> 4) + round;
    return sadd16(env, vxrm, c, res);
}

static inline int32_t
vwsmacc16(CPURISCVState *env, int vxrm, int16_t a, int16_t b, int32_t c)
{
    uint8_t round;
    int32_t res = (int32_t)a * b;

    round = th_get_round(vxrm, res, 8);
    res   = (res >> 8) + round;
    return sadd32(env, vxrm, c, res);

}

static inline int64_t
vwsmacc32(CPURISCVState *env, int vxrm, int32_t a, int32_t b, int64_t c)
{
    uint8_t round;
    int64_t res = (int64_t)a * b;

    round = th_get_round(vxrm, res, 16);
    res   = (res >> 16) + round;
    return sadd64(env, vxrm, c, res);
}

THCALL(TH_OPIVV3_RM, th_vwsmacc_vv_b, TH_WOP_SSS_B, H2, H1, H1, vwsmacc8)
THCALL(TH_OPIVV3_RM, th_vwsmacc_vv_h, TH_WOP_SSS_H, H4, H2, H2, vwsmacc16)
THCALL(TH_OPIVV3_RM, th_vwsmacc_vv_w, TH_WOP_SSS_W, H8, H4, H4, vwsmacc32)
GEN_TH_VV_RM(th_vwsmacc_vv_b, 1, 2, clearh_th)
GEN_TH_VV_RM(th_vwsmacc_vv_h, 2, 4, clearl_th)
GEN_TH_VV_RM(th_vwsmacc_vv_w, 4, 8, clearq_th)
THCALL(TH_OPIVX3_RM, th_vwsmacc_vx_b, TH_WOP_SSS_B, H2, H1, vwsmacc8)
THCALL(TH_OPIVX3_RM, th_vwsmacc_vx_h, TH_WOP_SSS_H, H4, H2, vwsmacc16)
THCALL(TH_OPIVX3_RM, th_vwsmacc_vx_w, TH_WOP_SSS_W, H8, H4, vwsmacc32)
GEN_TH_VX_RM(th_vwsmacc_vx_b, 1, 2, clearh_th)
GEN_TH_VX_RM(th_vwsmacc_vx_h, 2, 4, clearl_th)
GEN_TH_VX_RM(th_vwsmacc_vx_w, 4, 8, clearq_th)

static inline int16_t
vwsmaccsu8(CPURISCVState *env, int vxrm, uint8_t a, int8_t b, int16_t c)
{
    uint8_t round;
    int16_t res = a * (int16_t)b;

    round = th_get_round(vxrm, res, 4);
    res   = (res >> 4) + round;
    return ssub16(env, vxrm, c, res);
}

static inline int32_t
vwsmaccsu16(CPURISCVState *env, int vxrm, uint16_t a, int16_t b, uint32_t c)
{
    uint8_t round;
    int32_t res = a * (int32_t)b;

    round = th_get_round(vxrm, res, 8);
    res   = (res >> 8) + round;
    return ssub32(env, vxrm, c, res);
}

static inline int64_t
vwsmaccsu32(CPURISCVState *env, int vxrm, uint32_t a, int32_t b, int64_t c)
{
    uint8_t round;
    int64_t res = a * (int64_t)b;

    round = th_get_round(vxrm, res, 16);
    res   = (res >> 16) + round;
    return ssub64(env, vxrm, c, res);
}

THCALL(TH_OPIVV3_RM, th_vwsmaccsu_vv_b, TH_WOP_SSU_B, H2, H1, H1, vwsmaccsu8)
THCALL(TH_OPIVV3_RM, th_vwsmaccsu_vv_h, TH_WOP_SSU_H, H4, H2, H2, vwsmaccsu16)
THCALL(TH_OPIVV3_RM, th_vwsmaccsu_vv_w, TH_WOP_SSU_W, H8, H4, H4, vwsmaccsu32)
GEN_TH_VV_RM(th_vwsmaccsu_vv_b, 1, 2, clearh_th)
GEN_TH_VV_RM(th_vwsmaccsu_vv_h, 2, 4, clearl_th)
GEN_TH_VV_RM(th_vwsmaccsu_vv_w, 4, 8, clearq_th)
THCALL(TH_OPIVX3_RM, th_vwsmaccsu_vx_b, TH_WOP_SSU_B, H2, H1, vwsmaccsu8)
THCALL(TH_OPIVX3_RM, th_vwsmaccsu_vx_h, TH_WOP_SSU_H, H4, H2, vwsmaccsu16)
THCALL(TH_OPIVX3_RM, th_vwsmaccsu_vx_w, TH_WOP_SSU_W, H8, H4, vwsmaccsu32)
GEN_TH_VX_RM(th_vwsmaccsu_vx_b, 1, 2, clearh_th)
GEN_TH_VX_RM(th_vwsmaccsu_vx_h, 2, 4, clearl_th)
GEN_TH_VX_RM(th_vwsmaccsu_vx_w, 4, 8, clearq_th)

static inline int16_t
vwsmaccus8(CPURISCVState *env, int vxrm, int8_t a, uint8_t b, int16_t c)
{
    uint8_t round;
    int16_t res = (int16_t)a * b;

    round = th_get_round(vxrm, res, 4);
    res   = (res >> 4) + round;
    return ssub16(env, vxrm, c, res);
}

static inline int32_t
vwsmaccus16(CPURISCVState *env, int vxrm, int16_t a, uint16_t b, int32_t c)
{
    uint8_t round;
    int32_t res = (int32_t)a * b;

    round = th_get_round(vxrm, res, 8);
    res   = (res >> 8) + round;
    return ssub32(env, vxrm, c, res);
}

static inline int64_t
vwsmaccus32(CPURISCVState *env, int vxrm, int32_t a, uint32_t b, int64_t c)
{
    uint8_t round;
    int64_t res = (int64_t)a * b;

    round = th_get_round(vxrm, res, 16);
    res   = (res >> 16) + round;
    return ssub64(env, vxrm, c, res);
}

THCALL(TH_OPIVX3_RM, th_vwsmaccus_vx_b, TH_WOP_SUS_B, H2, H1, vwsmaccus8)
THCALL(TH_OPIVX3_RM, th_vwsmaccus_vx_h, TH_WOP_SUS_H, H4, H2, vwsmaccus16)
THCALL(TH_OPIVX3_RM, th_vwsmaccus_vx_w, TH_WOP_SUS_W, H8, H4, vwsmaccus32)
GEN_TH_VX_RM(th_vwsmaccus_vx_b, 1, 2, clearh_th)
GEN_TH_VX_RM(th_vwsmaccus_vx_h, 2, 4, clearl_th)
GEN_TH_VX_RM(th_vwsmaccus_vx_w, 4, 8, clearq_th)

/* Vector Single-Width Scaling Shift Instructions */
GEN_TH_SADD_FUNC(vssrl8, uint8_t)
GEN_TH_SADD_FUNC(vssrl16, uint16_t)
GEN_TH_SADD_FUNC(vssrl32, uint32_t)
GEN_TH_SADD_FUNC(vssrl64, uint64_t)
GEN_TH_SADD_FUNC(vssra8, int8_t)
GEN_TH_SADD_FUNC(vssra16, int16_t)
GEN_TH_SADD_FUNC(vssra32, int32_t)
GEN_TH_SADD_FUNC(vssra64, int64_t)

THCALL(TH_OPIVV2_RM, th_vssrl_vv_b, TH_OP_UUU_B, H1, H1, H1, th_vssrl8)
THCALL(TH_OPIVV2_RM, th_vssrl_vv_h, TH_OP_UUU_H, H2, H2, H2, th_vssrl16)
THCALL(TH_OPIVV2_RM, th_vssrl_vv_w, TH_OP_UUU_W, H4, H4, H4, th_vssrl32)
THCALL(TH_OPIVV2_RM, th_vssrl_vv_d, TH_OP_UUU_D, H8, H8, H8, th_vssrl64)
GEN_TH_VV_RM(th_vssrl_vv_b, 1, 1, clearb_th)
GEN_TH_VV_RM(th_vssrl_vv_h, 2, 2, clearh_th)
GEN_TH_VV_RM(th_vssrl_vv_w, 4, 4, clearl_th)
GEN_TH_VV_RM(th_vssrl_vv_d, 8, 8, clearq_th)

THCALL(TH_OPIVX2_RM, th_vssrl_vx_b, TH_OP_UUU_B, H1, H1, th_vssrl8)
THCALL(TH_OPIVX2_RM, th_vssrl_vx_h, TH_OP_UUU_H, H2, H2, th_vssrl16)
THCALL(TH_OPIVX2_RM, th_vssrl_vx_w, TH_OP_UUU_W, H4, H4, th_vssrl32)
THCALL(TH_OPIVX2_RM, th_vssrl_vx_d, TH_OP_UUU_D, H8, H8, th_vssrl64)
GEN_TH_VX_RM(th_vssrl_vx_b, 1, 1, clearb_th)
GEN_TH_VX_RM(th_vssrl_vx_h, 2, 2, clearh_th)
GEN_TH_VX_RM(th_vssrl_vx_w, 4, 4, clearl_th)
GEN_TH_VX_RM(th_vssrl_vx_d, 8, 8, clearq_th)

THCALL(TH_OPIVV2_RM, th_vssra_vv_b, TH_OP_SSS_B, H1, H1, H1, th_vssra8)
THCALL(TH_OPIVV2_RM, th_vssra_vv_h, TH_OP_SSS_H, H2, H2, H2, th_vssra16)
THCALL(TH_OPIVV2_RM, th_vssra_vv_w, TH_OP_SSS_W, H4, H4, H4, th_vssra32)
THCALL(TH_OPIVV2_RM, th_vssra_vv_d, TH_OP_SSS_D, H8, H8, H8, th_vssra64)
GEN_TH_VV_RM(th_vssra_vv_b, 1, 1, clearb_th)
GEN_TH_VV_RM(th_vssra_vv_h, 2, 2, clearh_th)
GEN_TH_VV_RM(th_vssra_vv_w, 4, 4, clearl_th)
GEN_TH_VV_RM(th_vssra_vv_d, 8, 8, clearq_th)

THCALL(TH_OPIVX2_RM, th_vssra_vx_b, TH_OP_SSS_B, H1, H1, th_vssra8)
THCALL(TH_OPIVX2_RM, th_vssra_vx_h, TH_OP_SSS_H, H2, H2, th_vssra16)
THCALL(TH_OPIVX2_RM, th_vssra_vx_w, TH_OP_SSS_W, H4, H4, th_vssra32)
THCALL(TH_OPIVX2_RM, th_vssra_vx_d, TH_OP_SSS_D, H8, H8, th_vssra64)
GEN_TH_VX_RM(th_vssra_vx_b, 1, 1, clearb_th)
GEN_TH_VX_RM(th_vssra_vx_h, 2, 2, clearh_th)
GEN_TH_VX_RM(th_vssra_vx_w, 4, 4, clearl_th)
GEN_TH_VX_RM(th_vssra_vx_d, 8, 8, clearq_th)

#define TH_NOP_SSS_B int8_t, int8_t, int16_t, int8_t, int16_t
#define TH_NOP_SSS_H int16_t, int16_t, int32_t, int16_t, int32_t
#define TH_NOP_SSS_W int32_t, int32_t, int64_t, int32_t, int64_t
#define TH_NOP_UUU_B uint8_t, uint8_t, uint16_t, uint8_t, uint16_t
#define TH_NOP_UUU_H uint16_t, uint16_t, uint32_t, uint16_t, uint32_t
#define TH_NOP_UUU_W uint32_t, uint32_t, uint64_t, uint32_t, uint64_t

/* Vector Narrowing Fixed-Point Clip Instructions */
#define GEN_TH_VNCLIP_FUNC(NAME, ATYPE, BTYPE)      \
static BTYPE th_##NAME(CPURISCVState *env, int vxrm,\
                      ATYPE s2, BTYPE s1)           \
{                                                   \
    return NAME(env, vxrm, s2, s1);                 \
}
GEN_TH_VNCLIP_FUNC(vnclip8, int16_t, int8_t)
GEN_TH_VNCLIP_FUNC(vnclip16, int32_t, int16_t)
GEN_TH_VNCLIP_FUNC(vnclip32, int64_t, int32_t)
GEN_TH_VNCLIP_FUNC(vnclipu8, uint16_t, uint8_t)
GEN_TH_VNCLIP_FUNC(vnclipu16, uint32_t, uint16_t)
GEN_TH_VNCLIP_FUNC(vnclipu32, uint64_t, uint32_t)

THCALL(TH_OPIVV2_RM, th_vnclip_vv_b, TH_NOP_SSS_B, H1, H2, H1, th_vnclip8)
THCALL(TH_OPIVV2_RM, th_vnclip_vv_h, TH_NOP_SSS_H, H2, H4, H2, th_vnclip16)
THCALL(TH_OPIVV2_RM, th_vnclip_vv_w, TH_NOP_SSS_W, H4, H8, H4, th_vnclip32)
GEN_TH_VV_RM(th_vnclip_vv_b, 1, 1, clearb_th)
GEN_TH_VV_RM(th_vnclip_vv_h, 2, 2, clearh_th)
GEN_TH_VV_RM(th_vnclip_vv_w, 4, 4, clearl_th)

THCALL(TH_OPIVX2_RM, th_vnclip_vx_b, TH_NOP_SSS_B, H1, H2, th_vnclip8)
THCALL(TH_OPIVX2_RM, th_vnclip_vx_h, TH_NOP_SSS_H, H2, H4, th_vnclip16)
THCALL(TH_OPIVX2_RM, th_vnclip_vx_w, TH_NOP_SSS_W, H4, H8, th_vnclip32)
GEN_TH_VX_RM(th_vnclip_vx_b, 1, 1, clearb_th)
GEN_TH_VX_RM(th_vnclip_vx_h, 2, 2, clearh_th)
GEN_TH_VX_RM(th_vnclip_vx_w, 4, 4, clearl_th)

THCALL(TH_OPIVV2_RM, th_vnclipu_vv_b, TH_NOP_UUU_B, H1, H2, H1, th_vnclipu8)
THCALL(TH_OPIVV2_RM, th_vnclipu_vv_h, TH_NOP_UUU_H, H2, H4, H2, th_vnclipu16)
THCALL(TH_OPIVV2_RM, th_vnclipu_vv_w, TH_NOP_UUU_W, H4, H8, H4, th_vnclipu32)
GEN_TH_VV_RM(th_vnclipu_vv_b, 1, 1, clearb_th)
GEN_TH_VV_RM(th_vnclipu_vv_h, 2, 2, clearh_th)
GEN_TH_VV_RM(th_vnclipu_vv_w, 4, 4, clearl_th)

THCALL(TH_OPIVX2_RM, th_vnclipu_vx_b, TH_NOP_UUU_B, H1, H2, th_vnclipu8)
THCALL(TH_OPIVX2_RM, th_vnclipu_vx_h, TH_NOP_UUU_H, H2, H4, th_vnclipu16)
THCALL(TH_OPIVX2_RM, th_vnclipu_vx_w, TH_NOP_UUU_W, H4, H8, th_vnclipu32)
GEN_TH_VX_RM(th_vnclipu_vx_b, 1, 1, clearb_th)
GEN_TH_VX_RM(th_vnclipu_vx_h, 2, 2, clearh_th)
GEN_TH_VX_RM(th_vnclipu_vx_w, 4, 4, clearl_th)
