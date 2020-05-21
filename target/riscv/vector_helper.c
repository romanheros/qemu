/*
 * RISC-V Vector Extension Helpers for QEMU.
 *
 * Copyright (c) 2020 T-Head Semiconductor Co., Ltd. All rights reserved.
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
#include "cpu.h"
#include "exec/memop.h"
#include "exec/exec-all.h"
#include "exec/helper-proto.h"
#include "tcg/tcg-gvec-desc.h"
#include "internals.h"
#include <math.h>

target_ulong HELPER(vsetvl)(CPURISCVState *env, target_ulong s1,
                            target_ulong s2)
{
    int vlmax, vl;
    RISCVCPU *cpu = env_archcpu(env);
    uint16_t sew = 8 << FIELD_EX64(s2, VTYPE, VSEW);
    uint8_t ediv = FIELD_EX64(s2, VTYPE, VEDIV);
    bool vill = FIELD_EX64(s2, VTYPE, VILL);
    target_ulong reserved = FIELD_EX64(s2, VTYPE, RESERVED);

    if ((sew > cpu->cfg.elen) || vill || (ediv != 0) || (reserved != 0)) {
        /* only set vill bit. */
        env->vtype = FIELD_DP64(0, VTYPE, VILL, 1);
        env->vl = 0;
        env->vstart = 0;
        return 0;
    }

    vlmax = vext_get_vlmax(cpu, s2);
    if (s1 <= vlmax) {
        vl = s1;
    } else {
        vl = vlmax;
    }
    env->vl = vl;
    env->vtype = s2;
    env->vstart = 0;
    return vl;
}

/*
 * Note that vector data is stored in host-endian 64-bit chunks,
 * so addressing units smaller than that needs a host-endian fixup.
 */
#ifdef HOST_WORDS_BIGENDIAN
#define H1(x)   ((x) ^ 7)
#define H1_2(x) ((x) ^ 6)
#define H1_4(x) ((x) ^ 4)
#define H2(x)   ((x) ^ 3)
#define H4(x)   ((x) ^ 1)
#define H8(x)   ((x))
#else
#define H1(x)   (x)
#define H1_2(x) (x)
#define H1_4(x) (x)
#define H2(x)   (x)
#define H4(x)   (x)
#define H8(x)   (x)
#endif

static inline uint32_t vext_nf(uint32_t desc)
{
    return FIELD_EX32(simd_data(desc), VDATA, NF);
}

static inline uint32_t vext_mlen(uint32_t desc)
{
    return FIELD_EX32(simd_data(desc), VDATA, MLEN);
}

static inline uint32_t vext_vm(uint32_t desc)
{
    return FIELD_EX32(simd_data(desc), VDATA, VM);
}

static inline uint32_t vext_lmul(uint32_t desc)
{
    return FIELD_EX32(simd_data(desc), VDATA, LMUL);
}

static uint32_t vext_wd(uint32_t desc)
{
    return (simd_data(desc) >> 11) & 0x1;
}

/*
 * Get vector group length in bytes. Its range is [64, 2048].
 *
 * As simd_desc support at most 256, the max vlen is 512 bits.
 * So vlen in bytes is encoded as maxsz.
 */
static inline uint32_t vext_maxsz(uint32_t desc)
{
    return simd_maxsz(desc) << vext_lmul(desc);
}

/*
 * This function checks watchpoint before real load operation.
 *
 * In softmmu mode, the TLB API probe_access is enough for watchpoint check.
 * In user mode, there is no watchpoint support now.
 *
 * It will trigger an exception if there is no mapping in TLB
 * and page table walk can't fill the TLB entry. Then the guest
 * software can return here after process the exception or never return.
 */
static void probe_pages(CPURISCVState *env, target_ulong addr,
                        target_ulong len, uintptr_t ra,
                        MMUAccessType access_type)
{
    target_ulong pagelen = -(addr | TARGET_PAGE_MASK);
    target_ulong curlen = MIN(pagelen, len);

    probe_access(env, addr, curlen, access_type,
                 cpu_mmu_index(env, false), ra);
    if (len > curlen) {
        addr += curlen;
        curlen = len - curlen;
        probe_access(env, addr, curlen, access_type,
                     cpu_mmu_index(env, false), ra);
    }
}

#ifdef HOST_WORDS_BIGENDIAN
static void vext_clear(void *tail, uint32_t cnt, uint32_t tot)
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
        memset(tail & ~(7ULL), 0, part1);
        memset((tail + 8) & ~(7ULL), 0, part2);
    } else {
        memset(tail, 0, part2);
    }
}
#else
static void vext_clear(void *tail, uint32_t cnt, uint32_t tot)
{
    memset(tail, 0, tot - cnt);
}
#endif

static void clearb(void *vd, uint32_t idx, uint32_t cnt, uint32_t tot)
{
    int8_t *cur = ((int8_t *)vd + H1(idx));
    vext_clear(cur, cnt, tot);
}

static void clearh(void *vd, uint32_t idx, uint32_t cnt, uint32_t tot)
{
    int16_t *cur = ((int16_t *)vd + H2(idx));
    vext_clear(cur, cnt, tot);
}

static void clearl(void *vd, uint32_t idx, uint32_t cnt, uint32_t tot)
{
    int32_t *cur = ((int32_t *)vd + H4(idx));
    vext_clear(cur, cnt, tot);
}

static void clearq(void *vd, uint32_t idx, uint32_t cnt, uint32_t tot)
{
    int64_t *cur = (int64_t *)vd + idx;
    vext_clear(cur, cnt, tot);
}

static inline void vext_set_elem_mask(void *v0, int mlen, int index,
        uint8_t value)
{
    int idx = (index * mlen) / 64;
    int pos = (index * mlen) % 64;
    uint64_t old = ((uint64_t *)v0)[idx];
    ((uint64_t *)v0)[idx] = deposit64(old, pos, mlen, value);
}

static inline int vext_elem_mask(void *v0, int mlen, int index)
{
    int idx = (index * mlen) / 64;
    int pos = (index * mlen) % 64;
    return (((uint64_t *)v0)[idx] >> pos) & 1;
}

/* elements operations for load and store */
typedef void vext_ldst_elem_fn(CPURISCVState *env, target_ulong addr,
                               uint32_t idx, void *vd, uintptr_t retaddr);
typedef void clear_fn(void *vd, uint32_t idx, uint32_t cnt, uint32_t tot);

#define GEN_VEXT_LD_ELEM(NAME, MTYPE, ETYPE, H, LDSUF)     \
static void NAME(CPURISCVState *env, abi_ptr addr,         \
                 uint32_t idx, void *vd, uintptr_t retaddr)\
{                                                          \
    MTYPE data;                                            \
    ETYPE *cur = ((ETYPE *)vd + H(idx));                   \
    data = cpu_##LDSUF##_data_ra(env, addr, retaddr);      \
    *cur = data;                                           \
}                                                          \

GEN_VEXT_LD_ELEM(ldb_b, int8_t,  int8_t,  H1, ldsb)
GEN_VEXT_LD_ELEM(ldb_h, int8_t,  int16_t, H2, ldsb)
GEN_VEXT_LD_ELEM(ldb_w, int8_t,  int32_t, H4, ldsb)
GEN_VEXT_LD_ELEM(ldb_d, int8_t,  int64_t, H8, ldsb)
GEN_VEXT_LD_ELEM(ldh_h, int16_t, int16_t, H2, ldsw)
GEN_VEXT_LD_ELEM(ldh_w, int16_t, int32_t, H4, ldsw)
GEN_VEXT_LD_ELEM(ldh_d, int16_t, int64_t, H8, ldsw)
GEN_VEXT_LD_ELEM(ldw_w, int32_t, int32_t, H4, ldl)
GEN_VEXT_LD_ELEM(ldw_d, int32_t, int64_t, H8, ldl)
GEN_VEXT_LD_ELEM(lde_b, int8_t,  int8_t,  H1, ldsb)
GEN_VEXT_LD_ELEM(lde_h, int16_t, int16_t, H2, ldsw)
GEN_VEXT_LD_ELEM(lde_w, int32_t, int32_t, H4, ldl)
GEN_VEXT_LD_ELEM(lde_d, int64_t, int64_t, H8, ldq)
GEN_VEXT_LD_ELEM(ldbu_b, uint8_t,  uint8_t,  H1, ldub)
GEN_VEXT_LD_ELEM(ldbu_h, uint8_t,  uint16_t, H2, ldub)
GEN_VEXT_LD_ELEM(ldbu_w, uint8_t,  uint32_t, H4, ldub)
GEN_VEXT_LD_ELEM(ldbu_d, uint8_t,  uint64_t, H8, ldub)
GEN_VEXT_LD_ELEM(ldhu_h, uint16_t, uint16_t, H2, lduw)
GEN_VEXT_LD_ELEM(ldhu_w, uint16_t, uint32_t, H4, lduw)
GEN_VEXT_LD_ELEM(ldhu_d, uint16_t, uint64_t, H8, lduw)
GEN_VEXT_LD_ELEM(ldwu_w, uint32_t, uint32_t, H4, ldl)
GEN_VEXT_LD_ELEM(ldwu_d, uint32_t, uint64_t, H8, ldl)

#define GEN_VEXT_ST_ELEM(NAME, ETYPE, H, STSUF)            \
static void NAME(CPURISCVState *env, abi_ptr addr,         \
                 uint32_t idx, void *vd, uintptr_t retaddr)\
{                                                          \
    ETYPE data = *((ETYPE *)vd + H(idx));                  \
    cpu_##STSUF##_data_ra(env, addr, data, retaddr);       \
}

GEN_VEXT_ST_ELEM(stb_b, int8_t,  H1, stb)
GEN_VEXT_ST_ELEM(stb_h, int16_t, H2, stb)
GEN_VEXT_ST_ELEM(stb_w, int32_t, H4, stb)
GEN_VEXT_ST_ELEM(stb_d, int64_t, H8, stb)
GEN_VEXT_ST_ELEM(sth_h, int16_t, H2, stw)
GEN_VEXT_ST_ELEM(sth_w, int32_t, H4, stw)
GEN_VEXT_ST_ELEM(sth_d, int64_t, H8, stw)
GEN_VEXT_ST_ELEM(stw_w, int32_t, H4, stl)
GEN_VEXT_ST_ELEM(stw_d, int64_t, H8, stl)
GEN_VEXT_ST_ELEM(ste_b, int8_t,  H1, stb)
GEN_VEXT_ST_ELEM(ste_h, int16_t, H2, stw)
GEN_VEXT_ST_ELEM(ste_w, int32_t, H4, stl)
GEN_VEXT_ST_ELEM(ste_d, int64_t, H8, stq)

/*
 *** stride: access vector element from strided memory
 */
static void
vext_ldst_stride(void *vd, void *v0, target_ulong base,
                 target_ulong stride, CPURISCVState *env,
                 uint32_t desc, uint32_t vm,
                 vext_ldst_elem_fn *ldst_elem, clear_fn *clear_elem,
                 uint32_t esz, uint32_t msz, uintptr_t ra,
                 MMUAccessType access_type)
{
    uint32_t i, k;
    uint32_t nf = vext_nf(desc);
    uint32_t mlen = vext_mlen(desc);
    uint32_t vlmax = vext_maxsz(desc) / esz;

    /* probe every access*/
    for (i = 0; i < env->vl; i++) {
        if (!vm && !vext_elem_mask(v0, mlen, i)) {
            continue;
        }
        probe_pages(env, base + stride * i, nf * msz, ra, access_type);
    }
    /* do real access */
    for (i = 0; i < env->vl; i++) {
        k = 0;
        if (!vm && !vext_elem_mask(v0, mlen, i)) {
            continue;
        }
        while (k < nf) {
            target_ulong addr = base + stride * i + k * msz;
            ldst_elem(env, addr, i + k * vlmax, vd, ra);
            k++;
        }
    }
    /* clear tail elements */
    if (clear_elem) {
        for (k = 0; k < nf; k++) {
            clear_elem(vd, env->vl + k * vlmax, env->vl * esz, vlmax * esz);
        }
    }
}

#define GEN_VEXT_LD_STRIDE(NAME, MTYPE, ETYPE, LOAD_FN, CLEAR_FN)       \
void HELPER(NAME)(void *vd, void * v0, target_ulong base,               \
                  target_ulong stride, CPURISCVState *env,              \
                  uint32_t desc)                                        \
{                                                                       \
    uint32_t vm = vext_vm(desc);                                        \
    vext_ldst_stride(vd, v0, base, stride, env, desc, vm, LOAD_FN,      \
                     CLEAR_FN, sizeof(ETYPE), sizeof(MTYPE),            \
                     GETPC(), MMU_DATA_LOAD);                           \
}

GEN_VEXT_LD_STRIDE(vlsb_v_b,  int8_t,   int8_t,   ldb_b,  clearb)
GEN_VEXT_LD_STRIDE(vlsb_v_h,  int8_t,   int16_t,  ldb_h,  clearh)
GEN_VEXT_LD_STRIDE(vlsb_v_w,  int8_t,   int32_t,  ldb_w,  clearl)
GEN_VEXT_LD_STRIDE(vlsb_v_d,  int8_t,   int64_t,  ldb_d,  clearq)
GEN_VEXT_LD_STRIDE(vlsh_v_h,  int16_t,  int16_t,  ldh_h,  clearh)
GEN_VEXT_LD_STRIDE(vlsh_v_w,  int16_t,  int32_t,  ldh_w,  clearl)
GEN_VEXT_LD_STRIDE(vlsh_v_d,  int16_t,  int64_t,  ldh_d,  clearq)
GEN_VEXT_LD_STRIDE(vlsw_v_w,  int32_t,  int32_t,  ldw_w,  clearl)
GEN_VEXT_LD_STRIDE(vlsw_v_d,  int32_t,  int64_t,  ldw_d,  clearq)
GEN_VEXT_LD_STRIDE(vlse_v_b,  int8_t,   int8_t,   lde_b,  clearb)
GEN_VEXT_LD_STRIDE(vlse_v_h,  int16_t,  int16_t,  lde_h,  clearh)
GEN_VEXT_LD_STRIDE(vlse_v_w,  int32_t,  int32_t,  lde_w,  clearl)
GEN_VEXT_LD_STRIDE(vlse_v_d,  int64_t,  int64_t,  lde_d,  clearq)
GEN_VEXT_LD_STRIDE(vlsbu_v_b, uint8_t,  uint8_t,  ldbu_b, clearb)
GEN_VEXT_LD_STRIDE(vlsbu_v_h, uint8_t,  uint16_t, ldbu_h, clearh)
GEN_VEXT_LD_STRIDE(vlsbu_v_w, uint8_t,  uint32_t, ldbu_w, clearl)
GEN_VEXT_LD_STRIDE(vlsbu_v_d, uint8_t,  uint64_t, ldbu_d, clearq)
GEN_VEXT_LD_STRIDE(vlshu_v_h, uint16_t, uint16_t, ldhu_h, clearh)
GEN_VEXT_LD_STRIDE(vlshu_v_w, uint16_t, uint32_t, ldhu_w, clearl)
GEN_VEXT_LD_STRIDE(vlshu_v_d, uint16_t, uint64_t, ldhu_d, clearq)
GEN_VEXT_LD_STRIDE(vlswu_v_w, uint32_t, uint32_t, ldwu_w, clearl)
GEN_VEXT_LD_STRIDE(vlswu_v_d, uint32_t, uint64_t, ldwu_d, clearq)

#define GEN_VEXT_ST_STRIDE(NAME, MTYPE, ETYPE, STORE_FN)                \
void HELPER(NAME)(void *vd, void *v0, target_ulong base,                \
                  target_ulong stride, CPURISCVState *env,              \
                  uint32_t desc)                                        \
{                                                                       \
    uint32_t vm = vext_vm(desc);                                        \
    vext_ldst_stride(vd, v0, base, stride, env, desc, vm, STORE_FN,     \
                     NULL, sizeof(ETYPE), sizeof(MTYPE),                \
                     GETPC(), MMU_DATA_STORE);                          \
}

GEN_VEXT_ST_STRIDE(vssb_v_b, int8_t,  int8_t,  stb_b)
GEN_VEXT_ST_STRIDE(vssb_v_h, int8_t,  int16_t, stb_h)
GEN_VEXT_ST_STRIDE(vssb_v_w, int8_t,  int32_t, stb_w)
GEN_VEXT_ST_STRIDE(vssb_v_d, int8_t,  int64_t, stb_d)
GEN_VEXT_ST_STRIDE(vssh_v_h, int16_t, int16_t, sth_h)
GEN_VEXT_ST_STRIDE(vssh_v_w, int16_t, int32_t, sth_w)
GEN_VEXT_ST_STRIDE(vssh_v_d, int16_t, int64_t, sth_d)
GEN_VEXT_ST_STRIDE(vssw_v_w, int32_t, int32_t, stw_w)
GEN_VEXT_ST_STRIDE(vssw_v_d, int32_t, int64_t, stw_d)
GEN_VEXT_ST_STRIDE(vsse_v_b, int8_t,  int8_t,  ste_b)
GEN_VEXT_ST_STRIDE(vsse_v_h, int16_t, int16_t, ste_h)
GEN_VEXT_ST_STRIDE(vsse_v_w, int32_t, int32_t, ste_w)
GEN_VEXT_ST_STRIDE(vsse_v_d, int64_t, int64_t, ste_d)

/*
 *** unit-stride: access elements stored contiguously in memory
 */

/* unmasked unit-stride load and store operation*/
static void
vext_ldst_us(void *vd, target_ulong base, CPURISCVState *env, uint32_t desc,
             vext_ldst_elem_fn *ldst_elem, clear_fn *clear_elem,
             uint32_t esz, uint32_t msz, uintptr_t ra,
             MMUAccessType access_type)
{
    uint32_t i, k;
    uint32_t nf = vext_nf(desc);
    uint32_t vlmax = vext_maxsz(desc) / esz;

    /* probe every access */
    probe_pages(env, base, env->vl * nf * msz, ra, access_type);
    /* load bytes from guest memory */
    for (i = 0; i < env->vl; i++) {
        k = 0;
        while (k < nf) {
            target_ulong addr = base + (i * nf + k) * msz;
            ldst_elem(env, addr, i + k * vlmax, vd, ra);
            k++;
        }
    }
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

#define GEN_VEXT_LD_US(NAME, MTYPE, ETYPE, LOAD_FN, CLEAR_FN)           \
void HELPER(NAME##_mask)(void *vd, void *v0, target_ulong base,         \
                         CPURISCVState *env, uint32_t desc)             \
{                                                                       \
    uint32_t stride = vext_nf(desc) * sizeof(MTYPE);                    \
    vext_ldst_stride(vd, v0, base, stride, env, desc, false, LOAD_FN,   \
                     CLEAR_FN, sizeof(ETYPE), sizeof(MTYPE),            \
                     GETPC(), MMU_DATA_LOAD);                           \
}                                                                       \
                                                                        \
void HELPER(NAME)(void *vd, void *v0, target_ulong base,                \
                  CPURISCVState *env, uint32_t desc)                    \
{                                                                       \
    vext_ldst_us(vd, base, env, desc, LOAD_FN, CLEAR_FN,                \
                 sizeof(ETYPE), sizeof(MTYPE), GETPC(), MMU_DATA_LOAD); \
}

GEN_VEXT_LD_US(vlb_v_b,  int8_t,   int8_t,   ldb_b,  clearb)
GEN_VEXT_LD_US(vlb_v_h,  int8_t,   int16_t,  ldb_h,  clearh)
GEN_VEXT_LD_US(vlb_v_w,  int8_t,   int32_t,  ldb_w,  clearl)
GEN_VEXT_LD_US(vlb_v_d,  int8_t,   int64_t,  ldb_d,  clearq)
GEN_VEXT_LD_US(vlh_v_h,  int16_t,  int16_t,  ldh_h,  clearh)
GEN_VEXT_LD_US(vlh_v_w,  int16_t,  int32_t,  ldh_w,  clearl)
GEN_VEXT_LD_US(vlh_v_d,  int16_t,  int64_t,  ldh_d,  clearq)
GEN_VEXT_LD_US(vlw_v_w,  int32_t,  int32_t,  ldw_w,  clearl)
GEN_VEXT_LD_US(vlw_v_d,  int32_t,  int64_t,  ldw_d,  clearq)
GEN_VEXT_LD_US(vle_v_b,  int8_t,   int8_t,   lde_b,  clearb)
GEN_VEXT_LD_US(vle_v_h,  int16_t,  int16_t,  lde_h,  clearh)
GEN_VEXT_LD_US(vle_v_w,  int32_t,  int32_t,  lde_w,  clearl)
GEN_VEXT_LD_US(vle_v_d,  int64_t,  int64_t,  lde_d,  clearq)
GEN_VEXT_LD_US(vlbu_v_b, uint8_t,  uint8_t,  ldbu_b, clearb)
GEN_VEXT_LD_US(vlbu_v_h, uint8_t,  uint16_t, ldbu_h, clearh)
GEN_VEXT_LD_US(vlbu_v_w, uint8_t,  uint32_t, ldbu_w, clearl)
GEN_VEXT_LD_US(vlbu_v_d, uint8_t,  uint64_t, ldbu_d, clearq)
GEN_VEXT_LD_US(vlhu_v_h, uint16_t, uint16_t, ldhu_h, clearh)
GEN_VEXT_LD_US(vlhu_v_w, uint16_t, uint32_t, ldhu_w, clearl)
GEN_VEXT_LD_US(vlhu_v_d, uint16_t, uint64_t, ldhu_d, clearq)
GEN_VEXT_LD_US(vlwu_v_w, uint32_t, uint32_t, ldwu_w, clearl)
GEN_VEXT_LD_US(vlwu_v_d, uint32_t, uint64_t, ldwu_d, clearq)

#define GEN_VEXT_ST_US(NAME, MTYPE, ETYPE, STORE_FN)                    \
void HELPER(NAME##_mask)(void *vd, void *v0, target_ulong base,         \
                         CPURISCVState *env, uint32_t desc)             \
{                                                                       \
    uint32_t stride = vext_nf(desc) * sizeof(MTYPE);                    \
    vext_ldst_stride(vd, v0, base, stride, env, desc, false, STORE_FN,  \
                     NULL, sizeof(ETYPE), sizeof(MTYPE),                \
                     GETPC(), MMU_DATA_STORE);                          \
}                                                                       \
                                                                        \
void HELPER(NAME)(void *vd, void *v0, target_ulong base,                \
                  CPURISCVState *env, uint32_t desc)                    \
{                                                                       \
    vext_ldst_us(vd, base, env, desc, STORE_FN, NULL,                   \
                 sizeof(ETYPE), sizeof(MTYPE), GETPC(), MMU_DATA_STORE);\
}

GEN_VEXT_ST_US(vsb_v_b, int8_t,  int8_t , stb_b)
GEN_VEXT_ST_US(vsb_v_h, int8_t,  int16_t, stb_h)
GEN_VEXT_ST_US(vsb_v_w, int8_t,  int32_t, stb_w)
GEN_VEXT_ST_US(vsb_v_d, int8_t,  int64_t, stb_d)
GEN_VEXT_ST_US(vsh_v_h, int16_t, int16_t, sth_h)
GEN_VEXT_ST_US(vsh_v_w, int16_t, int32_t, sth_w)
GEN_VEXT_ST_US(vsh_v_d, int16_t, int64_t, sth_d)
GEN_VEXT_ST_US(vsw_v_w, int32_t, int32_t, stw_w)
GEN_VEXT_ST_US(vsw_v_d, int32_t, int64_t, stw_d)
GEN_VEXT_ST_US(vse_v_b, int8_t,  int8_t , ste_b)
GEN_VEXT_ST_US(vse_v_h, int16_t, int16_t, ste_h)
GEN_VEXT_ST_US(vse_v_w, int32_t, int32_t, ste_w)
GEN_VEXT_ST_US(vse_v_d, int64_t, int64_t, ste_d)

/*
 *** index: access vector element from indexed memory
 */
typedef target_ulong vext_get_index_addr(target_ulong base,
        uint32_t idx, void *vs2);

#define GEN_VEXT_GET_INDEX_ADDR(NAME, ETYPE, H)        \
static target_ulong NAME(target_ulong base,            \
                         uint32_t idx, void *vs2)      \
{                                                      \
    return (base + *((ETYPE *)vs2 + H(idx)));          \
}

GEN_VEXT_GET_INDEX_ADDR(idx_b, int8_t,  H1)
GEN_VEXT_GET_INDEX_ADDR(idx_h, int16_t, H2)
GEN_VEXT_GET_INDEX_ADDR(idx_w, int32_t, H4)
GEN_VEXT_GET_INDEX_ADDR(idx_d, int64_t, H8)

static inline void
vext_ldst_index(void *vd, void *v0, target_ulong base,
                void *vs2, CPURISCVState *env, uint32_t desc,
                vext_get_index_addr get_index_addr,
                vext_ldst_elem_fn *ldst_elem,
                clear_fn *clear_elem,
                uint32_t esz, uint32_t msz, uintptr_t ra,
                MMUAccessType access_type)
{
    uint32_t i, k;
    uint32_t nf = vext_nf(desc);
    uint32_t vm = vext_vm(desc);
    uint32_t mlen = vext_mlen(desc);
    uint32_t vlmax = vext_maxsz(desc) / esz;

    /* probe every access*/
    for (i = 0; i < env->vl; i++) {
        if (!vm && !vext_elem_mask(v0, mlen, i)) {
            continue;
        }
        probe_pages(env, get_index_addr(base, i, vs2), nf * msz, ra,
                    access_type);
    }
    /* load bytes from guest memory */
    for (i = 0; i < env->vl; i++) {
        k = 0;
        if (!vm && !vext_elem_mask(v0, mlen, i)) {
            continue;
        }
        while (k < nf) {
            abi_ptr addr = get_index_addr(base, i, vs2) + k * msz;
            ldst_elem(env, addr, i + k * vlmax, vd, ra);
            k++;
        }
    }
    /* clear tail elements */
    if (clear_elem) {
        for (k = 0; k < nf; k++) {
            clear_elem(vd, env->vl + k * vlmax, env->vl * esz, vlmax * esz);
        }
    }
}

#define GEN_VEXT_LD_INDEX(NAME, MTYPE, ETYPE, INDEX_FN, LOAD_FN, CLEAR_FN) \
void HELPER(NAME)(void *vd, void *v0, target_ulong base,                   \
                  void *vs2, CPURISCVState *env, uint32_t desc)            \
{                                                                          \
    vext_ldst_index(vd, v0, base, vs2, env, desc, INDEX_FN,                \
                    LOAD_FN, CLEAR_FN, sizeof(ETYPE), sizeof(MTYPE),       \
                    GETPC(), MMU_DATA_LOAD);                               \
}

GEN_VEXT_LD_INDEX(vlxb_v_b,  int8_t,   int8_t,   idx_b, ldb_b,  clearb)
GEN_VEXT_LD_INDEX(vlxb_v_h,  int8_t,   int16_t,  idx_h, ldb_h,  clearh)
GEN_VEXT_LD_INDEX(vlxb_v_w,  int8_t,   int32_t,  idx_w, ldb_w,  clearl)
GEN_VEXT_LD_INDEX(vlxb_v_d,  int8_t,   int64_t,  idx_d, ldb_d,  clearq)
GEN_VEXT_LD_INDEX(vlxh_v_h,  int16_t,  int16_t,  idx_h, ldh_h,  clearh)
GEN_VEXT_LD_INDEX(vlxh_v_w,  int16_t,  int32_t,  idx_w, ldh_w,  clearl)
GEN_VEXT_LD_INDEX(vlxh_v_d,  int16_t,  int64_t,  idx_d, ldh_d,  clearq)
GEN_VEXT_LD_INDEX(vlxw_v_w,  int32_t,  int32_t,  idx_w, ldw_w,  clearl)
GEN_VEXT_LD_INDEX(vlxw_v_d,  int32_t,  int64_t,  idx_d, ldw_d,  clearq)
GEN_VEXT_LD_INDEX(vlxe_v_b,  int8_t,   int8_t,   idx_b, lde_b,  clearb)
GEN_VEXT_LD_INDEX(vlxe_v_h,  int16_t,  int16_t,  idx_h, lde_h,  clearh)
GEN_VEXT_LD_INDEX(vlxe_v_w,  int32_t,  int32_t,  idx_w, lde_w,  clearl)
GEN_VEXT_LD_INDEX(vlxe_v_d,  int64_t,  int64_t,  idx_d, lde_d,  clearq)
GEN_VEXT_LD_INDEX(vlxbu_v_b, uint8_t,  uint8_t,  idx_b, ldbu_b, clearb)
GEN_VEXT_LD_INDEX(vlxbu_v_h, uint8_t,  uint16_t, idx_h, ldbu_h, clearh)
GEN_VEXT_LD_INDEX(vlxbu_v_w, uint8_t,  uint32_t, idx_w, ldbu_w, clearl)
GEN_VEXT_LD_INDEX(vlxbu_v_d, uint8_t,  uint64_t, idx_d, ldbu_d, clearq)
GEN_VEXT_LD_INDEX(vlxhu_v_h, uint16_t, uint16_t, idx_h, ldhu_h, clearh)
GEN_VEXT_LD_INDEX(vlxhu_v_w, uint16_t, uint32_t, idx_w, ldhu_w, clearl)
GEN_VEXT_LD_INDEX(vlxhu_v_d, uint16_t, uint64_t, idx_d, ldhu_d, clearq)
GEN_VEXT_LD_INDEX(vlxwu_v_w, uint32_t, uint32_t, idx_w, ldwu_w, clearl)
GEN_VEXT_LD_INDEX(vlxwu_v_d, uint32_t, uint64_t, idx_d, ldwu_d, clearq)

#define GEN_VEXT_ST_INDEX(NAME, MTYPE, ETYPE, INDEX_FN, STORE_FN)\
void HELPER(NAME)(void *vd, void *v0, target_ulong base,         \
                  void *vs2, CPURISCVState *env, uint32_t desc)  \
{                                                                \
    vext_ldst_index(vd, v0, base, vs2, env, desc, INDEX_FN,      \
                    STORE_FN, NULL, sizeof(ETYPE), sizeof(MTYPE),\
                    GETPC(), MMU_DATA_STORE);                    \
}

GEN_VEXT_ST_INDEX(vsxb_v_b, int8_t,  int8_t,  idx_b, stb_b)
GEN_VEXT_ST_INDEX(vsxb_v_h, int8_t,  int16_t, idx_h, stb_h)
GEN_VEXT_ST_INDEX(vsxb_v_w, int8_t,  int32_t, idx_w, stb_w)
GEN_VEXT_ST_INDEX(vsxb_v_d, int8_t,  int64_t, idx_d, stb_d)
GEN_VEXT_ST_INDEX(vsxh_v_h, int16_t, int16_t, idx_h, sth_h)
GEN_VEXT_ST_INDEX(vsxh_v_w, int16_t, int32_t, idx_w, sth_w)
GEN_VEXT_ST_INDEX(vsxh_v_d, int16_t, int64_t, idx_d, sth_d)
GEN_VEXT_ST_INDEX(vsxw_v_w, int32_t, int32_t, idx_w, stw_w)
GEN_VEXT_ST_INDEX(vsxw_v_d, int32_t, int64_t, idx_d, stw_d)
GEN_VEXT_ST_INDEX(vsxe_v_b, int8_t,  int8_t,  idx_b, ste_b)
GEN_VEXT_ST_INDEX(vsxe_v_h, int16_t, int16_t, idx_h, ste_h)
GEN_VEXT_ST_INDEX(vsxe_v_w, int32_t, int32_t, idx_w, ste_w)
GEN_VEXT_ST_INDEX(vsxe_v_d, int64_t, int64_t, idx_d, ste_d)

/*
 *** unit-stride fault-only-fisrt load instructions
 */
static inline void
vext_ldff(void *vd, void *v0, target_ulong base,
          CPURISCVState *env, uint32_t desc,
          vext_ldst_elem_fn *ldst_elem,
          clear_fn *clear_elem,
          int mmuidx, uint32_t esz, uint32_t msz, uintptr_t ra)
{
    void *host;
    uint32_t i, k, vl = 0;
    uint32_t mlen = vext_mlen(desc);
    uint32_t nf = vext_nf(desc);
    uint32_t vm = vext_vm(desc);
    uint32_t vlmax = vext_maxsz(desc) / esz;
    target_ulong addr, offset, remain;

    /* probe every access*/
    for (i = 0; i < env->vl; i++) {
        if (!vm && !vext_elem_mask(v0, mlen, i)) {
            continue;
        }
        addr = base + nf * i * msz;
        if (i == 0) {
            probe_pages(env, addr, nf * msz, ra, MMU_DATA_LOAD);
        } else {
            /* if it triggers an exception, no need to check watchpoint */
            remain = nf * msz;
            while (remain > 0) {
                offset = -(addr | TARGET_PAGE_MASK);
                host = tlb_vaddr_to_host(env, addr, MMU_DATA_LOAD, mmuidx);
                if (host) {
#ifdef CONFIG_USER_ONLY
                    if (page_check_range(addr, nf * msz, PAGE_READ) < 0) {
                        vl = i;
                        goto ProbeSuccess;
                    }
#else
                    probe_pages(env, addr, nf * msz, ra, MMU_DATA_LOAD);
#endif
                } else {
                    vl = i;
                    goto ProbeSuccess;
                }
                if (remain <=  offset) {
                    break;
                }
                remain -= offset;
                addr += offset;
            }
        }
    }
ProbeSuccess:
    /* load bytes from guest memory */
    if (vl != 0) {
        env->vl = vl;
    }
    for (i = 0; i < env->vl; i++) {
        k = 0;
        if (!vm && !vext_elem_mask(v0, mlen, i)) {
            continue;
        }
        while (k < nf) {
            target_ulong addr = base + (i * nf + k) * msz;
            ldst_elem(env, addr, i + k * vlmax, vd, ra);
            k++;
        }
    }
    /* clear tail elements */
    if (vl != 0) {
        return;
    }
    for (k = 0; k < nf; k++) {
        clear_elem(vd, env->vl + k * vlmax, env->vl * esz, vlmax * esz);
    }
}

#define GEN_VEXT_LDFF(NAME, MTYPE, ETYPE, MMUIDX, LOAD_FN, CLEAR_FN)  \
void HELPER(NAME)(void *vd, void *v0, target_ulong base,              \
                  CPURISCVState *env, uint32_t desc)                  \
{                                                                     \
    vext_ldff(vd, v0, base, env, desc, LOAD_FN, CLEAR_FN, MMUIDX,     \
              sizeof(ETYPE), sizeof(MTYPE), GETPC());                 \
}

GEN_VEXT_LDFF(vlbff_v_b,  int8_t,   int8_t,   MO_SB,   ldb_b,  clearb)
GEN_VEXT_LDFF(vlbff_v_h,  int8_t,   int16_t,  MO_SB,   ldb_h,  clearh)
GEN_VEXT_LDFF(vlbff_v_w,  int8_t,   int32_t,  MO_SB,   ldb_w,  clearl)
GEN_VEXT_LDFF(vlbff_v_d,  int8_t,   int64_t,  MO_SB,   ldb_d,  clearq)
GEN_VEXT_LDFF(vlhff_v_h,  int16_t,  int16_t,  MO_LESW, ldh_h,  clearh)
GEN_VEXT_LDFF(vlhff_v_w,  int16_t,  int32_t,  MO_LESW, ldh_w,  clearl)
GEN_VEXT_LDFF(vlhff_v_d,  int16_t,  int64_t,  MO_LESW, ldh_d,  clearq)
GEN_VEXT_LDFF(vlwff_v_w,  int32_t,  int32_t,  MO_LESL, ldw_w,  clearl)
GEN_VEXT_LDFF(vlwff_v_d,  int32_t,  int64_t,  MO_LESL, ldw_d,  clearq)
GEN_VEXT_LDFF(vleff_v_b,  int8_t,   int8_t,   MO_SB,   lde_b,  clearb)
GEN_VEXT_LDFF(vleff_v_h,  int16_t,  int16_t,  MO_LESW, lde_h,  clearh)
GEN_VEXT_LDFF(vleff_v_w,  int32_t,  int32_t,  MO_LESL, lde_w,  clearl)
GEN_VEXT_LDFF(vleff_v_d,  int64_t,  int64_t,  MO_LEQ,  lde_d,  clearq)
GEN_VEXT_LDFF(vlbuff_v_b, uint8_t,  uint8_t,  MO_UB,   ldbu_b, clearb)
GEN_VEXT_LDFF(vlbuff_v_h, uint8_t,  uint16_t, MO_UB,   ldbu_h, clearh)
GEN_VEXT_LDFF(vlbuff_v_w, uint8_t,  uint32_t, MO_UB,   ldbu_w, clearl)
GEN_VEXT_LDFF(vlbuff_v_d, uint8_t,  uint64_t, MO_UB,   ldbu_d, clearq)
GEN_VEXT_LDFF(vlhuff_v_h, uint16_t, uint16_t, MO_LEUW, ldhu_h, clearh)
GEN_VEXT_LDFF(vlhuff_v_w, uint16_t, uint32_t, MO_LEUW, ldhu_w, clearl)
GEN_VEXT_LDFF(vlhuff_v_d, uint16_t, uint64_t, MO_LEUW, ldhu_d, clearq)
GEN_VEXT_LDFF(vlwuff_v_w, uint32_t, uint32_t, MO_LEUL, ldwu_w, clearl)
GEN_VEXT_LDFF(vlwuff_v_d, uint32_t, uint64_t, MO_LEUL, ldwu_d, clearq)

/*
 *** Vector AMO Operations (Zvamo)
 */
typedef void vext_amo_noatomic_fn(void *vs3, target_ulong addr,
                                  uint32_t wd, uint32_t idx, CPURISCVState *env,
                                  uintptr_t retaddr);

/* no atomic opreation for vector atomic insructions */
#define DO_SWAP(N, M) (M)
#define DO_AND(N, M)  (N & M)
#define DO_XOR(N, M)  (N ^ M)
#define DO_OR(N, M)   (N | M)
#define DO_ADD(N, M)  (N + M)

#define GEN_VEXT_AMO_NOATOMIC_OP(NAME, ESZ, MSZ, H, DO_OP, SUF) \
static void                                                     \
vext_##NAME##_noatomic_op(void *vs3, target_ulong addr,         \
                          uint32_t wd, uint32_t idx,            \
                          CPURISCVState *env, uintptr_t retaddr)\
{                                                               \
    typedef int##ESZ##_t ETYPE;                                 \
    typedef int##MSZ##_t MTYPE;                                 \
    typedef uint##MSZ##_t UMTYPE __attribute__((unused));       \
    ETYPE *pe3 = (ETYPE *)vs3 + H(idx);                         \
    MTYPE a = *pe3, b = cpu_ld##SUF##_data(env, addr);          \
    a = DO_OP(a, b);                                            \
    cpu_st##SUF##_data(env, addr, a);                           \
    if (wd) {                                                   \
        *pe3 = a;                                               \
    }                                                           \
}

/* Signed min/max */
#define DO_MAX(N, M)  ((N) >= (M) ? (N) : (M))
#define DO_MIN(N, M)  ((N) >= (M) ? (M) : (N))

/* Unsigned min/max */
#define DO_MAXU(N, M) DO_MAX((UMTYPE)N, (UMTYPE)M)
#define DO_MINU(N, M) DO_MIN((UMTYPE)N, (UMTYPE)M)

GEN_VEXT_AMO_NOATOMIC_OP(vamoswapw_v_w, 32, 32, H4, DO_SWAP, l)
GEN_VEXT_AMO_NOATOMIC_OP(vamoaddw_v_w,  32, 32, H4, DO_ADD,  l)
GEN_VEXT_AMO_NOATOMIC_OP(vamoxorw_v_w,  32, 32, H4, DO_XOR,  l)
GEN_VEXT_AMO_NOATOMIC_OP(vamoandw_v_w,  32, 32, H4, DO_AND,  l)
GEN_VEXT_AMO_NOATOMIC_OP(vamoorw_v_w,   32, 32, H4, DO_OR,   l)
GEN_VEXT_AMO_NOATOMIC_OP(vamominw_v_w,  32, 32, H4, DO_MIN,  l)
GEN_VEXT_AMO_NOATOMIC_OP(vamomaxw_v_w,  32, 32, H4, DO_MAX,  l)
GEN_VEXT_AMO_NOATOMIC_OP(vamominuw_v_w, 32, 32, H4, DO_MINU, l)
GEN_VEXT_AMO_NOATOMIC_OP(vamomaxuw_v_w, 32, 32, H4, DO_MAXU, l)
#ifdef TARGET_RISCV64
GEN_VEXT_AMO_NOATOMIC_OP(vamoswapw_v_d, 64, 32, H8, DO_SWAP, l)
GEN_VEXT_AMO_NOATOMIC_OP(vamoswapd_v_d, 64, 64, H8, DO_SWAP, q)
GEN_VEXT_AMO_NOATOMIC_OP(vamoaddw_v_d,  64, 32, H8, DO_ADD,  l)
GEN_VEXT_AMO_NOATOMIC_OP(vamoaddd_v_d,  64, 64, H8, DO_ADD,  q)
GEN_VEXT_AMO_NOATOMIC_OP(vamoxorw_v_d,  64, 32, H8, DO_XOR,  l)
GEN_VEXT_AMO_NOATOMIC_OP(vamoxord_v_d,  64, 64, H8, DO_XOR,  q)
GEN_VEXT_AMO_NOATOMIC_OP(vamoandw_v_d,  64, 32, H8, DO_AND,  l)
GEN_VEXT_AMO_NOATOMIC_OP(vamoandd_v_d,  64, 64, H8, DO_AND,  q)
GEN_VEXT_AMO_NOATOMIC_OP(vamoorw_v_d,   64, 32, H8, DO_OR,   l)
GEN_VEXT_AMO_NOATOMIC_OP(vamoord_v_d,   64, 64, H8, DO_OR,   q)
GEN_VEXT_AMO_NOATOMIC_OP(vamominw_v_d,  64, 32, H8, DO_MIN,  l)
GEN_VEXT_AMO_NOATOMIC_OP(vamomind_v_d,  64, 64, H8, DO_MIN,  q)
GEN_VEXT_AMO_NOATOMIC_OP(vamomaxw_v_d,  64, 32, H8, DO_MAX,  l)
GEN_VEXT_AMO_NOATOMIC_OP(vamomaxd_v_d,  64, 64, H8, DO_MAX,  q)
GEN_VEXT_AMO_NOATOMIC_OP(vamominuw_v_d, 64, 32, H8, DO_MINU, l)
GEN_VEXT_AMO_NOATOMIC_OP(vamominud_v_d, 64, 64, H8, DO_MINU, q)
GEN_VEXT_AMO_NOATOMIC_OP(vamomaxuw_v_d, 64, 32, H8, DO_MAXU, l)
GEN_VEXT_AMO_NOATOMIC_OP(vamomaxud_v_d, 64, 64, H8, DO_MAXU, q)
#endif

static inline void
vext_amo_noatomic(void *vs3, void *v0, target_ulong base,
                  void *vs2, CPURISCVState *env, uint32_t desc,
                  vext_get_index_addr get_index_addr,
                  vext_amo_noatomic_fn *noatomic_op,
                  clear_fn *clear_elem,
                  uint32_t esz, uint32_t msz, uintptr_t ra)
{
    uint32_t i;
    target_long addr;
    uint32_t wd = vext_wd(desc);
    uint32_t vm = vext_vm(desc);
    uint32_t mlen = vext_mlen(desc);
    uint32_t vlmax = vext_maxsz(desc) / esz;

    for (i = 0; i < env->vl; i++) {
        if (!vm && !vext_elem_mask(v0, mlen, i)) {
            continue;
        }
        probe_pages(env, get_index_addr(base, i, vs2), msz, ra, MMU_DATA_LOAD);
        probe_pages(env, get_index_addr(base, i, vs2), msz, ra, MMU_DATA_STORE);
    }
    for (i = 0; i < env->vl; i++) {
        if (!vm && !vext_elem_mask(v0, mlen, i)) {
            continue;
        }
        addr = get_index_addr(base, i, vs2);
        noatomic_op(vs3, addr, wd, i, env, ra);
    }
    clear_elem(vs3, env->vl, env->vl * esz, vlmax * esz);
}

#define GEN_VEXT_AMO(NAME, MTYPE, ETYPE, INDEX_FN, CLEAR_FN)    \
void HELPER(NAME)(void *vs3, void *v0, target_ulong base,       \
                  void *vs2, CPURISCVState *env, uint32_t desc) \
{                                                               \
    vext_amo_noatomic(vs3, v0, base, vs2, env, desc,            \
                      INDEX_FN, vext_##NAME##_noatomic_op,      \
                      CLEAR_FN, sizeof(ETYPE), sizeof(MTYPE),   \
                      GETPC());                                 \
}

#ifdef TARGET_RISCV64
GEN_VEXT_AMO(vamoswapw_v_d, int32_t,  int64_t,  idx_d, clearq)
GEN_VEXT_AMO(vamoswapd_v_d, int64_t,  int64_t,  idx_d, clearq)
GEN_VEXT_AMO(vamoaddw_v_d,  int32_t,  int64_t,  idx_d, clearq)
GEN_VEXT_AMO(vamoaddd_v_d,  int64_t,  int64_t,  idx_d, clearq)
GEN_VEXT_AMO(vamoxorw_v_d,  int32_t,  int64_t,  idx_d, clearq)
GEN_VEXT_AMO(vamoxord_v_d,  int64_t,  int64_t,  idx_d, clearq)
GEN_VEXT_AMO(vamoandw_v_d,  int32_t,  int64_t,  idx_d, clearq)
GEN_VEXT_AMO(vamoandd_v_d,  int64_t,  int64_t,  idx_d, clearq)
GEN_VEXT_AMO(vamoorw_v_d,   int32_t,  int64_t,  idx_d, clearq)
GEN_VEXT_AMO(vamoord_v_d,   int64_t,  int64_t,  idx_d, clearq)
GEN_VEXT_AMO(vamominw_v_d,  int32_t,  int64_t,  idx_d, clearq)
GEN_VEXT_AMO(vamomind_v_d,  int64_t,  int64_t,  idx_d, clearq)
GEN_VEXT_AMO(vamomaxw_v_d,  int32_t,  int64_t,  idx_d, clearq)
GEN_VEXT_AMO(vamomaxd_v_d,  int64_t,  int64_t,  idx_d, clearq)
GEN_VEXT_AMO(vamominuw_v_d, uint32_t, uint64_t, idx_d, clearq)
GEN_VEXT_AMO(vamominud_v_d, uint64_t, uint64_t, idx_d, clearq)
GEN_VEXT_AMO(vamomaxuw_v_d, uint32_t, uint64_t, idx_d, clearq)
GEN_VEXT_AMO(vamomaxud_v_d, uint64_t, uint64_t, idx_d, clearq)
#endif
GEN_VEXT_AMO(vamoswapw_v_w, int32_t,  int32_t,  idx_w, clearl)
GEN_VEXT_AMO(vamoaddw_v_w,  int32_t,  int32_t,  idx_w, clearl)
GEN_VEXT_AMO(vamoxorw_v_w,  int32_t,  int32_t,  idx_w, clearl)
GEN_VEXT_AMO(vamoandw_v_w,  int32_t,  int32_t,  idx_w, clearl)
GEN_VEXT_AMO(vamoorw_v_w,   int32_t,  int32_t,  idx_w, clearl)
GEN_VEXT_AMO(vamominw_v_w,  int32_t,  int32_t,  idx_w, clearl)
GEN_VEXT_AMO(vamomaxw_v_w,  int32_t,  int32_t,  idx_w, clearl)
GEN_VEXT_AMO(vamominuw_v_w, uint32_t, uint32_t, idx_w, clearl)
GEN_VEXT_AMO(vamomaxuw_v_w, uint32_t, uint32_t, idx_w, clearl)

/*
 *** Vector Integer Arithmetic Instructions
 */

/* expand macro args before macro */
#define RVVCALL(macro, ...)  macro(__VA_ARGS__)

/* (TD, T1, T2, TX1, TX2) */
#define OP_SSS_B int8_t, int8_t, int8_t, int8_t, int8_t
#define OP_SSS_H int16_t, int16_t, int16_t, int16_t, int16_t
#define OP_SSS_W int32_t, int32_t, int32_t, int32_t, int32_t
#define OP_SSS_D int64_t, int64_t, int64_t, int64_t, int64_t

/* operation of two vector elements */
typedef void opivv2_fn(void *vd, void *vs1, void *vs2, int i);

#define OPIVV2(NAME, TD, T1, T2, TX1, TX2, HD, HS1, HS2, OP)    \
static void do_##NAME(void *vd, void *vs1, void *vs2, int i)    \
{                                                               \
    TX1 s1 = *((T1 *)vs1 + HS1(i));                             \
    TX2 s2 = *((T2 *)vs2 + HS2(i));                             \
    *((TD *)vd + HD(i)) = OP(s2, s1);                           \
}
#define DO_SUB(N, M) (N - M)
#define DO_RSUB(N, M) (M - N)

RVVCALL(OPIVV2, vadd_vv_b, OP_SSS_B, H1, H1, H1, DO_ADD)
RVVCALL(OPIVV2, vadd_vv_h, OP_SSS_H, H2, H2, H2, DO_ADD)
RVVCALL(OPIVV2, vadd_vv_w, OP_SSS_W, H4, H4, H4, DO_ADD)
RVVCALL(OPIVV2, vadd_vv_d, OP_SSS_D, H8, H8, H8, DO_ADD)
RVVCALL(OPIVV2, vsub_vv_b, OP_SSS_B, H1, H1, H1, DO_SUB)
RVVCALL(OPIVV2, vsub_vv_h, OP_SSS_H, H2, H2, H2, DO_SUB)
RVVCALL(OPIVV2, vsub_vv_w, OP_SSS_W, H4, H4, H4, DO_SUB)
RVVCALL(OPIVV2, vsub_vv_d, OP_SSS_D, H8, H8, H8, DO_SUB)

static void do_vext_vv(void *vd, void *v0, void *vs1, void *vs2,
                       CPURISCVState *env, uint32_t desc,
                       uint32_t esz, uint32_t dsz,
                       opivv2_fn *fn, clear_fn *clearfn)
{
    uint32_t vlmax = vext_maxsz(desc) / esz;
    uint32_t mlen = vext_mlen(desc);
    uint32_t vm = vext_vm(desc);
    uint32_t vl = env->vl;
    uint32_t i;

    for (i = 0; i < vl; i++) {
        if (!vm && !vext_elem_mask(v0, mlen, i)) {
            continue;
        }
        fn(vd, vs1, vs2, i);
    }
    clearfn(vd, vl, vl * dsz,  vlmax * dsz);
}

/* generate the helpers for OPIVV */
#define GEN_VEXT_VV(NAME, ESZ, DSZ, CLEAR_FN)             \
void HELPER(NAME)(void *vd, void *v0, void *vs1,          \
                  void *vs2, CPURISCVState *env,          \
                  uint32_t desc)                          \
{                                                         \
    do_vext_vv(vd, v0, vs1, vs2, env, desc, ESZ, DSZ,     \
               do_##NAME, CLEAR_FN);                      \
}

GEN_VEXT_VV(vadd_vv_b, 1, 1, clearb)
GEN_VEXT_VV(vadd_vv_h, 2, 2, clearh)
GEN_VEXT_VV(vadd_vv_w, 4, 4, clearl)
GEN_VEXT_VV(vadd_vv_d, 8, 8, clearq)
GEN_VEXT_VV(vsub_vv_b, 1, 1, clearb)
GEN_VEXT_VV(vsub_vv_h, 2, 2, clearh)
GEN_VEXT_VV(vsub_vv_w, 4, 4, clearl)
GEN_VEXT_VV(vsub_vv_d, 8, 8, clearq)

typedef void opivx2_fn(void *vd, target_long s1, void *vs2, int i);

/*
 * (T1)s1 gives the real operator type.
 * (TX1)(T1)s1 expands the operator type of widen or narrow operations.
 */
#define OPIVX2(NAME, TD, T1, T2, TX1, TX2, HD, HS2, OP)             \
static void do_##NAME(void *vd, target_long s1, void *vs2, int i)   \
{                                                                   \
    TX2 s2 = *((T2 *)vs2 + HS2(i));                                 \
    *((TD *)vd + HD(i)) = OP(s2, (TX1)(T1)s1);                      \
}

RVVCALL(OPIVX2, vadd_vx_b, OP_SSS_B, H1, H1, DO_ADD)
RVVCALL(OPIVX2, vadd_vx_h, OP_SSS_H, H2, H2, DO_ADD)
RVVCALL(OPIVX2, vadd_vx_w, OP_SSS_W, H4, H4, DO_ADD)
RVVCALL(OPIVX2, vadd_vx_d, OP_SSS_D, H8, H8, DO_ADD)
RVVCALL(OPIVX2, vsub_vx_b, OP_SSS_B, H1, H1, DO_SUB)
RVVCALL(OPIVX2, vsub_vx_h, OP_SSS_H, H2, H2, DO_SUB)
RVVCALL(OPIVX2, vsub_vx_w, OP_SSS_W, H4, H4, DO_SUB)
RVVCALL(OPIVX2, vsub_vx_d, OP_SSS_D, H8, H8, DO_SUB)
RVVCALL(OPIVX2, vrsub_vx_b, OP_SSS_B, H1, H1, DO_RSUB)
RVVCALL(OPIVX2, vrsub_vx_h, OP_SSS_H, H2, H2, DO_RSUB)
RVVCALL(OPIVX2, vrsub_vx_w, OP_SSS_W, H4, H4, DO_RSUB)
RVVCALL(OPIVX2, vrsub_vx_d, OP_SSS_D, H8, H8, DO_RSUB)

static void do_vext_vx(void *vd, void *v0, target_long s1, void *vs2,
                       CPURISCVState *env, uint32_t desc,
                       uint32_t esz, uint32_t dsz,
                       opivx2_fn fn, clear_fn *clearfn)
{
    uint32_t vlmax = vext_maxsz(desc) / esz;
    uint32_t mlen = vext_mlen(desc);
    uint32_t vm = vext_vm(desc);
    uint32_t vl = env->vl;
    uint32_t i;

    for (i = 0; i < vl; i++) {
        if (!vm && !vext_elem_mask(v0, mlen, i)) {
            continue;
        }
        fn(vd, s1, vs2, i);
    }
    clearfn(vd, vl, vl * dsz,  vlmax * dsz);
}

/* generate the helpers for OPIVX */
#define GEN_VEXT_VX(NAME, ESZ, DSZ, CLEAR_FN)             \
void HELPER(NAME)(void *vd, void *v0, target_ulong s1,    \
                  void *vs2, CPURISCVState *env,          \
                  uint32_t desc)                          \
{                                                         \
    do_vext_vx(vd, v0, s1, vs2, env, desc, ESZ, DSZ,      \
               do_##NAME, CLEAR_FN);                      \
}

GEN_VEXT_VX(vadd_vx_b, 1, 1, clearb)
GEN_VEXT_VX(vadd_vx_h, 2, 2, clearh)
GEN_VEXT_VX(vadd_vx_w, 4, 4, clearl)
GEN_VEXT_VX(vadd_vx_d, 8, 8, clearq)
GEN_VEXT_VX(vsub_vx_b, 1, 1, clearb)
GEN_VEXT_VX(vsub_vx_h, 2, 2, clearh)
GEN_VEXT_VX(vsub_vx_w, 4, 4, clearl)
GEN_VEXT_VX(vsub_vx_d, 8, 8, clearq)
GEN_VEXT_VX(vrsub_vx_b, 1, 1, clearb)
GEN_VEXT_VX(vrsub_vx_h, 2, 2, clearh)
GEN_VEXT_VX(vrsub_vx_w, 4, 4, clearl)
GEN_VEXT_VX(vrsub_vx_d, 8, 8, clearq)

void HELPER(vec_rsubs8)(void *d, void *a, uint64_t b, uint32_t desc)
{
    intptr_t oprsz = simd_oprsz(desc);
    intptr_t i;

    for (i = 0; i < oprsz; i += sizeof(uint8_t)) {
        *(uint8_t *)(d + i) = (uint8_t)b - *(uint8_t *)(a + i);
    }
}

void HELPER(vec_rsubs16)(void *d, void *a, uint64_t b, uint32_t desc)
{
    intptr_t oprsz = simd_oprsz(desc);
    intptr_t i;

    for (i = 0; i < oprsz; i += sizeof(uint16_t)) {
        *(uint16_t *)(d + i) = (uint16_t)b - *(uint16_t *)(a + i);
    }
}

void HELPER(vec_rsubs32)(void *d, void *a, uint64_t b, uint32_t desc)
{
    intptr_t oprsz = simd_oprsz(desc);
    intptr_t i;

    for (i = 0; i < oprsz; i += sizeof(uint32_t)) {
        *(uint32_t *)(d + i) = (uint32_t)b - *(uint32_t *)(a + i);
    }
}

void HELPER(vec_rsubs64)(void *d, void *a, uint64_t b, uint32_t desc)
{
    intptr_t oprsz = simd_oprsz(desc);
    intptr_t i;

    for (i = 0; i < oprsz; i += sizeof(uint64_t)) {
        *(uint64_t *)(d + i) = b - *(uint64_t *)(a + i);
    }
}

/* Vector Widening Integer Add/Subtract */
#define WOP_UUU_B uint16_t, uint8_t, uint8_t, uint16_t, uint16_t
#define WOP_UUU_H uint32_t, uint16_t, uint16_t, uint32_t, uint32_t
#define WOP_UUU_W uint64_t, uint32_t, uint32_t, uint64_t, uint64_t
#define WOP_SSS_B int16_t, int8_t, int8_t, int16_t, int16_t
#define WOP_SSS_H int32_t, int16_t, int16_t, int32_t, int32_t
#define WOP_SSS_W int64_t, int32_t, int32_t, int64_t, int64_t
#define WOP_WUUU_B  uint16_t, uint8_t, uint16_t, uint16_t, uint16_t
#define WOP_WUUU_H  uint32_t, uint16_t, uint32_t, uint32_t, uint32_t
#define WOP_WUUU_W  uint64_t, uint32_t, uint64_t, uint64_t, uint64_t
#define WOP_WSSS_B  int16_t, int8_t, int16_t, int16_t, int16_t
#define WOP_WSSS_H  int32_t, int16_t, int32_t, int32_t, int32_t
#define WOP_WSSS_W  int64_t, int32_t, int64_t, int64_t, int64_t
RVVCALL(OPIVV2, vwaddu_vv_b, WOP_UUU_B, H2, H1, H1, DO_ADD)
RVVCALL(OPIVV2, vwaddu_vv_h, WOP_UUU_H, H4, H2, H2, DO_ADD)
RVVCALL(OPIVV2, vwaddu_vv_w, WOP_UUU_W, H8, H4, H4, DO_ADD)
RVVCALL(OPIVV2, vwsubu_vv_b, WOP_UUU_B, H2, H1, H1, DO_SUB)
RVVCALL(OPIVV2, vwsubu_vv_h, WOP_UUU_H, H4, H2, H2, DO_SUB)
RVVCALL(OPIVV2, vwsubu_vv_w, WOP_UUU_W, H8, H4, H4, DO_SUB)
RVVCALL(OPIVV2, vwadd_vv_b, WOP_SSS_B, H2, H1, H1, DO_ADD)
RVVCALL(OPIVV2, vwadd_vv_h, WOP_SSS_H, H4, H2, H2, DO_ADD)
RVVCALL(OPIVV2, vwadd_vv_w, WOP_SSS_W, H8, H4, H4, DO_ADD)
RVVCALL(OPIVV2, vwsub_vv_b, WOP_SSS_B, H2, H1, H1, DO_SUB)
RVVCALL(OPIVV2, vwsub_vv_h, WOP_SSS_H, H4, H2, H2, DO_SUB)
RVVCALL(OPIVV2, vwsub_vv_w, WOP_SSS_W, H8, H4, H4, DO_SUB)
RVVCALL(OPIVV2, vwaddu_wv_b, WOP_WUUU_B, H2, H1, H1, DO_ADD)
RVVCALL(OPIVV2, vwaddu_wv_h, WOP_WUUU_H, H4, H2, H2, DO_ADD)
RVVCALL(OPIVV2, vwaddu_wv_w, WOP_WUUU_W, H8, H4, H4, DO_ADD)
RVVCALL(OPIVV2, vwsubu_wv_b, WOP_WUUU_B, H2, H1, H1, DO_SUB)
RVVCALL(OPIVV2, vwsubu_wv_h, WOP_WUUU_H, H4, H2, H2, DO_SUB)
RVVCALL(OPIVV2, vwsubu_wv_w, WOP_WUUU_W, H8, H4, H4, DO_SUB)
RVVCALL(OPIVV2, vwadd_wv_b, WOP_WSSS_B, H2, H1, H1, DO_ADD)
RVVCALL(OPIVV2, vwadd_wv_h, WOP_WSSS_H, H4, H2, H2, DO_ADD)
RVVCALL(OPIVV2, vwadd_wv_w, WOP_WSSS_W, H8, H4, H4, DO_ADD)
RVVCALL(OPIVV2, vwsub_wv_b, WOP_WSSS_B, H2, H1, H1, DO_SUB)
RVVCALL(OPIVV2, vwsub_wv_h, WOP_WSSS_H, H4, H2, H2, DO_SUB)
RVVCALL(OPIVV2, vwsub_wv_w, WOP_WSSS_W, H8, H4, H4, DO_SUB)
GEN_VEXT_VV(vwaddu_vv_b, 1, 2, clearh)
GEN_VEXT_VV(vwaddu_vv_h, 2, 4, clearl)
GEN_VEXT_VV(vwaddu_vv_w, 4, 8, clearq)
GEN_VEXT_VV(vwsubu_vv_b, 1, 2, clearh)
GEN_VEXT_VV(vwsubu_vv_h, 2, 4, clearl)
GEN_VEXT_VV(vwsubu_vv_w, 4, 8, clearq)
GEN_VEXT_VV(vwadd_vv_b, 1, 2, clearh)
GEN_VEXT_VV(vwadd_vv_h, 2, 4, clearl)
GEN_VEXT_VV(vwadd_vv_w, 4, 8, clearq)
GEN_VEXT_VV(vwsub_vv_b, 1, 2, clearh)
GEN_VEXT_VV(vwsub_vv_h, 2, 4, clearl)
GEN_VEXT_VV(vwsub_vv_w, 4, 8, clearq)
GEN_VEXT_VV(vwaddu_wv_b, 1, 2, clearh)
GEN_VEXT_VV(vwaddu_wv_h, 2, 4, clearl)
GEN_VEXT_VV(vwaddu_wv_w, 4, 8, clearq)
GEN_VEXT_VV(vwsubu_wv_b, 1, 2, clearh)
GEN_VEXT_VV(vwsubu_wv_h, 2, 4, clearl)
GEN_VEXT_VV(vwsubu_wv_w, 4, 8, clearq)
GEN_VEXT_VV(vwadd_wv_b, 1, 2, clearh)
GEN_VEXT_VV(vwadd_wv_h, 2, 4, clearl)
GEN_VEXT_VV(vwadd_wv_w, 4, 8, clearq)
GEN_VEXT_VV(vwsub_wv_b, 1, 2, clearh)
GEN_VEXT_VV(vwsub_wv_h, 2, 4, clearl)
GEN_VEXT_VV(vwsub_wv_w, 4, 8, clearq)

RVVCALL(OPIVX2, vwaddu_vx_b, WOP_UUU_B, H2, H1, DO_ADD)
RVVCALL(OPIVX2, vwaddu_vx_h, WOP_UUU_H, H4, H2, DO_ADD)
RVVCALL(OPIVX2, vwaddu_vx_w, WOP_UUU_W, H8, H4, DO_ADD)
RVVCALL(OPIVX2, vwsubu_vx_b, WOP_UUU_B, H2, H1, DO_SUB)
RVVCALL(OPIVX2, vwsubu_vx_h, WOP_UUU_H, H4, H2, DO_SUB)
RVVCALL(OPIVX2, vwsubu_vx_w, WOP_UUU_W, H8, H4, DO_SUB)
RVVCALL(OPIVX2, vwadd_vx_b, WOP_SSS_B, H2, H1, DO_ADD)
RVVCALL(OPIVX2, vwadd_vx_h, WOP_SSS_H, H4, H2, DO_ADD)
RVVCALL(OPIVX2, vwadd_vx_w, WOP_SSS_W, H8, H4, DO_ADD)
RVVCALL(OPIVX2, vwsub_vx_b, WOP_SSS_B, H2, H1, DO_SUB)
RVVCALL(OPIVX2, vwsub_vx_h, WOP_SSS_H, H4, H2, DO_SUB)
RVVCALL(OPIVX2, vwsub_vx_w, WOP_SSS_W, H8, H4, DO_SUB)
RVVCALL(OPIVX2, vwaddu_wx_b, WOP_WUUU_B, H2, H1, DO_ADD)
RVVCALL(OPIVX2, vwaddu_wx_h, WOP_WUUU_H, H4, H2, DO_ADD)
RVVCALL(OPIVX2, vwaddu_wx_w, WOP_WUUU_W, H8, H4, DO_ADD)
RVVCALL(OPIVX2, vwsubu_wx_b, WOP_WUUU_B, H2, H1, DO_SUB)
RVVCALL(OPIVX2, vwsubu_wx_h, WOP_WUUU_H, H4, H2, DO_SUB)
RVVCALL(OPIVX2, vwsubu_wx_w, WOP_WUUU_W, H8, H4, DO_SUB)
RVVCALL(OPIVX2, vwadd_wx_b, WOP_WSSS_B, H2, H1, DO_ADD)
RVVCALL(OPIVX2, vwadd_wx_h, WOP_WSSS_H, H4, H2, DO_ADD)
RVVCALL(OPIVX2, vwadd_wx_w, WOP_WSSS_W, H8, H4, DO_ADD)
RVVCALL(OPIVX2, vwsub_wx_b, WOP_WSSS_B, H2, H1, DO_SUB)
RVVCALL(OPIVX2, vwsub_wx_h, WOP_WSSS_H, H4, H2, DO_SUB)
RVVCALL(OPIVX2, vwsub_wx_w, WOP_WSSS_W, H8, H4, DO_SUB)
GEN_VEXT_VX(vwaddu_vx_b, 1, 2, clearh)
GEN_VEXT_VX(vwaddu_vx_h, 2, 4, clearl)
GEN_VEXT_VX(vwaddu_vx_w, 4, 8, clearq)
GEN_VEXT_VX(vwsubu_vx_b, 1, 2, clearh)
GEN_VEXT_VX(vwsubu_vx_h, 2, 4, clearl)
GEN_VEXT_VX(vwsubu_vx_w, 4, 8, clearq)
GEN_VEXT_VX(vwadd_vx_b, 1, 2, clearh)
GEN_VEXT_VX(vwadd_vx_h, 2, 4, clearl)
GEN_VEXT_VX(vwadd_vx_w, 4, 8, clearq)
GEN_VEXT_VX(vwsub_vx_b, 1, 2, clearh)
GEN_VEXT_VX(vwsub_vx_h, 2, 4, clearl)
GEN_VEXT_VX(vwsub_vx_w, 4, 8, clearq)
GEN_VEXT_VX(vwaddu_wx_b, 1, 2, clearh)
GEN_VEXT_VX(vwaddu_wx_h, 2, 4, clearl)
GEN_VEXT_VX(vwaddu_wx_w, 4, 8, clearq)
GEN_VEXT_VX(vwsubu_wx_b, 1, 2, clearh)
GEN_VEXT_VX(vwsubu_wx_h, 2, 4, clearl)
GEN_VEXT_VX(vwsubu_wx_w, 4, 8, clearq)
GEN_VEXT_VX(vwadd_wx_b, 1, 2, clearh)
GEN_VEXT_VX(vwadd_wx_h, 2, 4, clearl)
GEN_VEXT_VX(vwadd_wx_w, 4, 8, clearq)
GEN_VEXT_VX(vwsub_wx_b, 1, 2, clearh)
GEN_VEXT_VX(vwsub_wx_h, 2, 4, clearl)
GEN_VEXT_VX(vwsub_wx_w, 4, 8, clearq)

/* Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions */
#define DO_VADC(N, M, C) (N + M + C)
#define DO_VSBC(N, M, C) (N - M - C)

#define GEN_VEXT_VADC_VVM(NAME, ETYPE, H, DO_OP, CLEAR_FN)    \
void HELPER(NAME)(void *vd, void *v0, void *vs1, void *vs2,   \
                  CPURISCVState *env, uint32_t desc)          \
{                                                             \
    uint32_t mlen = vext_mlen(desc);                          \
    uint32_t vl = env->vl;                                    \
    uint32_t esz = sizeof(ETYPE);                             \
    uint32_t vlmax = vext_maxsz(desc) / esz;                  \
    uint32_t i;                                               \
                                                              \
    for (i = 0; i < vl; i++) {                                \
        ETYPE s1 = *((ETYPE *)vs1 + H(i));                    \
        ETYPE s2 = *((ETYPE *)vs2 + H(i));                    \
        uint8_t carry = vext_elem_mask(v0, mlen, i);          \
                                                              \
        *((ETYPE *)vd + H(i)) = DO_OP(s2, s1, carry);         \
    }                                                         \
    CLEAR_FN(vd, vl, vl * esz, vlmax * esz);                  \
}

GEN_VEXT_VADC_VVM(vadc_vvm_b, uint8_t,  H1, DO_VADC, clearb)
GEN_VEXT_VADC_VVM(vadc_vvm_h, uint16_t, H2, DO_VADC, clearh)
GEN_VEXT_VADC_VVM(vadc_vvm_w, uint32_t, H4, DO_VADC, clearl)
GEN_VEXT_VADC_VVM(vadc_vvm_d, uint64_t, H8, DO_VADC, clearq)

GEN_VEXT_VADC_VVM(vsbc_vvm_b, uint8_t,  H1, DO_VSBC, clearb)
GEN_VEXT_VADC_VVM(vsbc_vvm_h, uint16_t, H2, DO_VSBC, clearh)
GEN_VEXT_VADC_VVM(vsbc_vvm_w, uint32_t, H4, DO_VSBC, clearl)
GEN_VEXT_VADC_VVM(vsbc_vvm_d, uint64_t, H8, DO_VSBC, clearq)

#define GEN_VEXT_VADC_VXM(NAME, ETYPE, H, DO_OP, CLEAR_FN)               \
void HELPER(NAME)(void *vd, void *v0, target_ulong s1, void *vs2,        \
                  CPURISCVState *env, uint32_t desc)                     \
{                                                                        \
    uint32_t mlen = vext_mlen(desc);                                     \
    uint32_t vl = env->vl;                                               \
    uint32_t esz = sizeof(ETYPE);                                        \
    uint32_t vlmax = vext_maxsz(desc) / esz;                             \
    uint32_t i;                                                          \
                                                                         \
    for (i = 0; i < vl; i++) {                                           \
        ETYPE s2 = *((ETYPE *)vs2 + H(i));                               \
        uint8_t carry = vext_elem_mask(v0, mlen, i);                     \
                                                                         \
        *((ETYPE *)vd + H(i)) = DO_OP(s2, (ETYPE)(target_long)s1, carry);\
    }                                                                    \
    CLEAR_FN(vd, vl, vl * esz, vlmax * esz);                             \
}

GEN_VEXT_VADC_VXM(vadc_vxm_b, uint8_t,  H1, DO_VADC, clearb)
GEN_VEXT_VADC_VXM(vadc_vxm_h, uint16_t, H2, DO_VADC, clearh)
GEN_VEXT_VADC_VXM(vadc_vxm_w, uint32_t, H4, DO_VADC, clearl)
GEN_VEXT_VADC_VXM(vadc_vxm_d, uint64_t, H8, DO_VADC, clearq)

GEN_VEXT_VADC_VXM(vsbc_vxm_b, uint8_t,  H1, DO_VSBC, clearb)
GEN_VEXT_VADC_VXM(vsbc_vxm_h, uint16_t, H2, DO_VSBC, clearh)
GEN_VEXT_VADC_VXM(vsbc_vxm_w, uint32_t, H4, DO_VSBC, clearl)
GEN_VEXT_VADC_VXM(vsbc_vxm_d, uint64_t, H8, DO_VSBC, clearq)

#define DO_MADC(N, M, C) (C ? (__typeof(N))(N + M + 1) <= N :           \
                          (__typeof(N))(N + M) < N)
#define DO_MSBC(N, M, C) (C ? N <= M : N < M)

#define GEN_VEXT_VMADC_VVM(NAME, ETYPE, H, DO_OP)             \
void HELPER(NAME)(void *vd, void *v0, void *vs1, void *vs2,   \
                  CPURISCVState *env, uint32_t desc)          \
{                                                             \
    uint32_t mlen = vext_mlen(desc);                          \
    uint32_t vl = env->vl;                                    \
    uint32_t vlmax = vext_maxsz(desc) / sizeof(ETYPE);        \
    uint32_t i;                                               \
                                                              \
    for (i = 0; i < vl; i++) {                                \
        ETYPE s1 = *((ETYPE *)vs1 + H(i));                    \
        ETYPE s2 = *((ETYPE *)vs2 + H(i));                    \
        uint8_t carry = vext_elem_mask(v0, mlen, i);          \
                                                              \
        vext_set_elem_mask(vd, mlen, i, DO_OP(s2, s1, carry));\
    }                                                         \
    for (; i < vlmax; i++) {                                  \
        vext_set_elem_mask(vd, mlen, i, 0);                   \
    }                                                         \
}

GEN_VEXT_VMADC_VVM(vmadc_vvm_b, uint8_t,  H1, DO_MADC)
GEN_VEXT_VMADC_VVM(vmadc_vvm_h, uint16_t, H2, DO_MADC)
GEN_VEXT_VMADC_VVM(vmadc_vvm_w, uint32_t, H4, DO_MADC)
GEN_VEXT_VMADC_VVM(vmadc_vvm_d, uint64_t, H8, DO_MADC)

GEN_VEXT_VMADC_VVM(vmsbc_vvm_b, uint8_t,  H1, DO_MSBC)
GEN_VEXT_VMADC_VVM(vmsbc_vvm_h, uint16_t, H2, DO_MSBC)
GEN_VEXT_VMADC_VVM(vmsbc_vvm_w, uint32_t, H4, DO_MSBC)
GEN_VEXT_VMADC_VVM(vmsbc_vvm_d, uint64_t, H8, DO_MSBC)

#define GEN_VEXT_VMADC_VXM(NAME, ETYPE, H, DO_OP)               \
void HELPER(NAME)(void *vd, void *v0, target_ulong s1,          \
                  void *vs2, CPURISCVState *env, uint32_t desc) \
{                                                               \
    uint32_t mlen = vext_mlen(desc);                            \
    uint32_t vl = env->vl;                                      \
    uint32_t vlmax = vext_maxsz(desc) / sizeof(ETYPE);          \
    uint32_t i;                                                 \
                                                                \
    for (i = 0; i < vl; i++) {                                  \
        ETYPE s2 = *((ETYPE *)vs2 + H(i));                      \
        uint8_t carry = vext_elem_mask(v0, mlen, i);            \
                                                                \
        vext_set_elem_mask(vd, mlen, i,                         \
                DO_OP(s2, (ETYPE)(target_long)s1, carry));      \
    }                                                           \
    for (; i < vlmax; i++) {                                    \
        vext_set_elem_mask(vd, mlen, i, 0);                     \
    }                                                           \
}

GEN_VEXT_VMADC_VXM(vmadc_vxm_b, uint8_t,  H1, DO_MADC)
GEN_VEXT_VMADC_VXM(vmadc_vxm_h, uint16_t, H2, DO_MADC)
GEN_VEXT_VMADC_VXM(vmadc_vxm_w, uint32_t, H4, DO_MADC)
GEN_VEXT_VMADC_VXM(vmadc_vxm_d, uint64_t, H8, DO_MADC)

GEN_VEXT_VMADC_VXM(vmsbc_vxm_b, uint8_t,  H1, DO_MSBC)
GEN_VEXT_VMADC_VXM(vmsbc_vxm_h, uint16_t, H2, DO_MSBC)
GEN_VEXT_VMADC_VXM(vmsbc_vxm_w, uint32_t, H4, DO_MSBC)
GEN_VEXT_VMADC_VXM(vmsbc_vxm_d, uint64_t, H8, DO_MSBC)

/* Vector Bitwise Logical Instructions */
RVVCALL(OPIVV2, vand_vv_b, OP_SSS_B, H1, H1, H1, DO_AND)
RVVCALL(OPIVV2, vand_vv_h, OP_SSS_H, H2, H2, H2, DO_AND)
RVVCALL(OPIVV2, vand_vv_w, OP_SSS_W, H4, H4, H4, DO_AND)
RVVCALL(OPIVV2, vand_vv_d, OP_SSS_D, H8, H8, H8, DO_AND)
RVVCALL(OPIVV2, vor_vv_b, OP_SSS_B, H1, H1, H1, DO_OR)
RVVCALL(OPIVV2, vor_vv_h, OP_SSS_H, H2, H2, H2, DO_OR)
RVVCALL(OPIVV2, vor_vv_w, OP_SSS_W, H4, H4, H4, DO_OR)
RVVCALL(OPIVV2, vor_vv_d, OP_SSS_D, H8, H8, H8, DO_OR)
RVVCALL(OPIVV2, vxor_vv_b, OP_SSS_B, H1, H1, H1, DO_XOR)
RVVCALL(OPIVV2, vxor_vv_h, OP_SSS_H, H2, H2, H2, DO_XOR)
RVVCALL(OPIVV2, vxor_vv_w, OP_SSS_W, H4, H4, H4, DO_XOR)
RVVCALL(OPIVV2, vxor_vv_d, OP_SSS_D, H8, H8, H8, DO_XOR)
GEN_VEXT_VV(vand_vv_b, 1, 1, clearb)
GEN_VEXT_VV(vand_vv_h, 2, 2, clearh)
GEN_VEXT_VV(vand_vv_w, 4, 4, clearl)
GEN_VEXT_VV(vand_vv_d, 8, 8, clearq)
GEN_VEXT_VV(vor_vv_b, 1, 1, clearb)
GEN_VEXT_VV(vor_vv_h, 2, 2, clearh)
GEN_VEXT_VV(vor_vv_w, 4, 4, clearl)
GEN_VEXT_VV(vor_vv_d, 8, 8, clearq)
GEN_VEXT_VV(vxor_vv_b, 1, 1, clearb)
GEN_VEXT_VV(vxor_vv_h, 2, 2, clearh)
GEN_VEXT_VV(vxor_vv_w, 4, 4, clearl)
GEN_VEXT_VV(vxor_vv_d, 8, 8, clearq)

RVVCALL(OPIVX2, vand_vx_b, OP_SSS_B, H1, H1, DO_AND)
RVVCALL(OPIVX2, vand_vx_h, OP_SSS_H, H2, H2, DO_AND)
RVVCALL(OPIVX2, vand_vx_w, OP_SSS_W, H4, H4, DO_AND)
RVVCALL(OPIVX2, vand_vx_d, OP_SSS_D, H8, H8, DO_AND)
RVVCALL(OPIVX2, vor_vx_b, OP_SSS_B, H1, H1, DO_OR)
RVVCALL(OPIVX2, vor_vx_h, OP_SSS_H, H2, H2, DO_OR)
RVVCALL(OPIVX2, vor_vx_w, OP_SSS_W, H4, H4, DO_OR)
RVVCALL(OPIVX2, vor_vx_d, OP_SSS_D, H8, H8, DO_OR)
RVVCALL(OPIVX2, vxor_vx_b, OP_SSS_B, H1, H1, DO_XOR)
RVVCALL(OPIVX2, vxor_vx_h, OP_SSS_H, H2, H2, DO_XOR)
RVVCALL(OPIVX2, vxor_vx_w, OP_SSS_W, H4, H4, DO_XOR)
RVVCALL(OPIVX2, vxor_vx_d, OP_SSS_D, H8, H8, DO_XOR)
GEN_VEXT_VX(vand_vx_b, 1, 1, clearb)
GEN_VEXT_VX(vand_vx_h, 2, 2, clearh)
GEN_VEXT_VX(vand_vx_w, 4, 4, clearl)
GEN_VEXT_VX(vand_vx_d, 8, 8, clearq)
GEN_VEXT_VX(vor_vx_b, 1, 1, clearb)
GEN_VEXT_VX(vor_vx_h, 2, 2, clearh)
GEN_VEXT_VX(vor_vx_w, 4, 4, clearl)
GEN_VEXT_VX(vor_vx_d, 8, 8, clearq)
GEN_VEXT_VX(vxor_vx_b, 1, 1, clearb)
GEN_VEXT_VX(vxor_vx_h, 2, 2, clearh)
GEN_VEXT_VX(vxor_vx_w, 4, 4, clearl)
GEN_VEXT_VX(vxor_vx_d, 8, 8, clearq)

/* Vector Single-Width Bit Shift Instructions */
#define DO_SLL(N, M)  (N << (M))
#define DO_SRL(N, M)  (N >> (M))

/* generate the helpers for shift instructions with two vector operators */
#define GEN_VEXT_SHIFT_VV(NAME, TS1, TS2, HS1, HS2, OP, MASK, CLEAR_FN)   \
void HELPER(NAME)(void *vd, void *v0, void *vs1,                          \
                  void *vs2, CPURISCVState *env, uint32_t desc)           \
{                                                                         \
    uint32_t mlen = vext_mlen(desc);                                      \
    uint32_t vm = vext_vm(desc);                                          \
    uint32_t vl = env->vl;                                                \
    uint32_t esz = sizeof(TS1);                                           \
    uint32_t vlmax = vext_maxsz(desc) / esz;                              \
    uint32_t i;                                                           \
                                                                          \
    for (i = 0; i < vl; i++) {                                            \
        if (!vm && !vext_elem_mask(v0, mlen, i)) {                        \
            continue;                                                     \
        }                                                                 \
        TS1 s1 = *((TS1 *)vs1 + HS1(i));                                  \
        TS2 s2 = *((TS2 *)vs2 + HS2(i));                                  \
        *((TS1 *)vd + HS1(i)) = OP(s2, s1 & MASK);                        \
    }                                                                     \
    CLEAR_FN(vd, vl, vl * esz, vlmax * esz);                              \
}

GEN_VEXT_SHIFT_VV(vsll_vv_b, uint8_t,  uint8_t, H1, H1, DO_SLL, 0x7, clearb)
GEN_VEXT_SHIFT_VV(vsll_vv_h, uint16_t, uint16_t, H2, H2, DO_SLL, 0xf, clearh)
GEN_VEXT_SHIFT_VV(vsll_vv_w, uint32_t, uint32_t, H4, H4, DO_SLL, 0x1f, clearl)
GEN_VEXT_SHIFT_VV(vsll_vv_d, uint64_t, uint64_t, H8, H8, DO_SLL, 0x3f, clearq)

GEN_VEXT_SHIFT_VV(vsrl_vv_b, uint8_t, uint8_t, H1, H1, DO_SRL, 0x7, clearb)
GEN_VEXT_SHIFT_VV(vsrl_vv_h, uint16_t, uint16_t, H2, H2, DO_SRL, 0xf, clearh)
GEN_VEXT_SHIFT_VV(vsrl_vv_w, uint32_t, uint32_t, H4, H4, DO_SRL, 0x1f, clearl)
GEN_VEXT_SHIFT_VV(vsrl_vv_d, uint64_t, uint64_t, H8, H8, DO_SRL, 0x3f, clearq)

GEN_VEXT_SHIFT_VV(vsra_vv_b, uint8_t,  int8_t, H1, H1, DO_SRL, 0x7, clearb)
GEN_VEXT_SHIFT_VV(vsra_vv_h, uint16_t, int16_t, H2, H2, DO_SRL, 0xf, clearh)
GEN_VEXT_SHIFT_VV(vsra_vv_w, uint32_t, int32_t, H4, H4, DO_SRL, 0x1f, clearl)
GEN_VEXT_SHIFT_VV(vsra_vv_d, uint64_t, int64_t, H8, H8, DO_SRL, 0x3f, clearq)

/* generate the helpers for shift instructions with one vector and one scalar */
#define GEN_VEXT_SHIFT_VX(NAME, TD, TS2, HD, HS2, OP, MASK, CLEAR_FN) \
void HELPER(NAME)(void *vd, void *v0, target_ulong s1,                \
        void *vs2, CPURISCVState *env, uint32_t desc)                 \
{                                                                     \
    uint32_t mlen = vext_mlen(desc);                                  \
    uint32_t vm = vext_vm(desc);                                      \
    uint32_t vl = env->vl;                                            \
    uint32_t esz = sizeof(TD);                                        \
    uint32_t vlmax = vext_maxsz(desc) / esz;                          \
    uint32_t i;                                                       \
                                                                      \
    for (i = 0; i < vl; i++) {                                        \
        if (!vm && !vext_elem_mask(v0, mlen, i)) {                    \
            continue;                                                 \
        }                                                             \
        TS2 s2 = *((TS2 *)vs2 + HS2(i));                              \
        *((TD *)vd + HD(i)) = OP(s2, s1 & MASK);                      \
    }                                                                 \
    CLEAR_FN(vd, vl, vl * esz, vlmax * esz);                          \
}

GEN_VEXT_SHIFT_VX(vsll_vx_b, uint8_t, int8_t, H1, H1, DO_SLL, 0x7, clearb)
GEN_VEXT_SHIFT_VX(vsll_vx_h, uint16_t, int16_t, H2, H2, DO_SLL, 0xf, clearh)
GEN_VEXT_SHIFT_VX(vsll_vx_w, uint32_t, int32_t, H4, H4, DO_SLL, 0x1f, clearl)
GEN_VEXT_SHIFT_VX(vsll_vx_d, uint64_t, int64_t, H8, H8, DO_SLL, 0x3f, clearq)

GEN_VEXT_SHIFT_VX(vsrl_vx_b, uint8_t, uint8_t, H1, H1, DO_SRL, 0x7, clearb)
GEN_VEXT_SHIFT_VX(vsrl_vx_h, uint16_t, uint16_t, H2, H2, DO_SRL, 0xf, clearh)
GEN_VEXT_SHIFT_VX(vsrl_vx_w, uint32_t, uint32_t, H4, H4, DO_SRL, 0x1f, clearl)
GEN_VEXT_SHIFT_VX(vsrl_vx_d, uint64_t, uint64_t, H8, H8, DO_SRL, 0x3f, clearq)

GEN_VEXT_SHIFT_VX(vsra_vx_b, int8_t, int8_t, H1, H1, DO_SRL, 0x7, clearb)
GEN_VEXT_SHIFT_VX(vsra_vx_h, int16_t, int16_t, H2, H2, DO_SRL, 0xf, clearh)
GEN_VEXT_SHIFT_VX(vsra_vx_w, int32_t, int32_t, H4, H4, DO_SRL, 0x1f, clearl)
GEN_VEXT_SHIFT_VX(vsra_vx_d, int64_t, int64_t, H8, H8, DO_SRL, 0x3f, clearq)
