/*
 * QEMU RISC-V CPU -- internal functions and types
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

#ifndef RISCV_CPU_INTERNALS_H
#define RISCV_CPU_INTERNALS_H

#include "hw/registerfields.h"

/* share data between vector helpers and decode code */
FIELD(VDATA, MLEN, 0, 8)
FIELD(VDATA, VM, 8, 1)
FIELD(VDATA, LMUL, 9, 2)
FIELD(VDATA, NF, 11, 4)
FIELD(VDATA, WD, 11, 1)

/* float point classify helpers */
target_ulong fclass_h(uint64_t frs1);
target_ulong fclass_s(uint64_t frs1);
target_ulong fclass_d(uint64_t frs1);

#define SEW8  0
#define SEW16 1
#define SEW32 2
#define SEW64 3

#ifndef CONFIG_USER_ONLY
extern const VMStateDescription vmstate_riscv_cpu;
#endif

static inline uint64_t nanbox_s(float32 f)
{
    return f | MAKE_64BIT_MASK(32, 32);
}

static inline float32 check_nanbox_s(uint64_t f)
{
    uint64_t mask = MAKE_64BIT_MASK(32, 32);

    if (likely((f & mask) == mask)) {
        return (uint32_t)f;
    } else {
        return 0x7fc00000u; /* default qnan */
    }
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

/* share functions about saturation */
int8_t sadd8(CPURISCVState *, int vxrm, int8_t, int8_t);
int16_t sadd16(CPURISCVState *, int vxrm, int16_t, int16_t);
int32_t sadd32(CPURISCVState *, int vxrm, int32_t, int32_t);
int64_t sadd64(CPURISCVState *, int vxrm, int64_t, int64_t);

uint8_t saddu8(CPURISCVState *, int vxrm, uint8_t, uint8_t);
uint16_t saddu16(CPURISCVState *, int vxrm, uint16_t, uint16_t);
uint32_t saddu32(CPURISCVState *, int vxrm, uint32_t, uint32_t);
uint64_t saddu64(CPURISCVState *, int vxrm, uint64_t, uint64_t);

int8_t ssub8(CPURISCVState *, int vxrm, int8_t, int8_t);
int16_t ssub16(CPURISCVState *, int vxrm, int16_t, int16_t);
int32_t ssub32(CPURISCVState *, int vxrm, int32_t, int32_t);
int64_t ssub64(CPURISCVState *, int vxrm, int64_t, int64_t);

uint8_t ssubu8(CPURISCVState *, int vxrm, uint8_t, uint8_t);
uint16_t ssubu16(CPURISCVState *, int vxrm, uint16_t, uint16_t);
uint32_t ssubu32(CPURISCVState *, int vxrm, uint32_t, uint32_t);
uint64_t ssubu64(CPURISCVState *, int vxrm, uint64_t, uint64_t);

/* share shift functions */
int8_t vssra8(CPURISCVState *env, int vxrm, int8_t a, int8_t b);
int16_t vssra16(CPURISCVState *env, int vxrm, int16_t a, int16_t b);
int32_t vssra32(CPURISCVState *env, int vxrm, int32_t a, int32_t b);
int64_t vssra64(CPURISCVState *env, int vxrm, int64_t a, int64_t b);
uint8_t vssrl8(CPURISCVState *env, int vxrm, uint8_t a, uint8_t b);
uint16_t vssrl16(CPURISCVState *env, int vxrm, uint16_t a, uint16_t b);
uint32_t vssrl32(CPURISCVState *env, int vxrm, uint32_t a, uint32_t b);
uint64_t vssrl64(CPURISCVState *env, int vxrm, uint64_t a, uint64_t b);
#endif
