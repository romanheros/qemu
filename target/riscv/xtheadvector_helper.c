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
