/*
 * QEMU RISC-V CPU -- internal functions and types
 *
 * Copyright (c) 2020 PingTouGe Semiconductor Co., Ltd. All rights reserved.
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
FIELD(VDATA, VM, 0, 1)
FIELD(VDATA, LMUL, 1, 3)
FIELD(VDATA, SEW, 4, 3)
FIELD(VDATA, VTA, 7, 1)
FIELD(VDATA, VMA, 8, 1)
/* NF is 4-bits wide as its range is [1..8] */
FIELD(VDATA, NF, 9, 4)
FIELD(VDATA, WD, 9, 1)

/* float point classify helpers */
target_ulong fclass_h(uint64_t frs1);
target_ulong fclass_s(uint64_t frs1);
target_ulong fclass_d(uint64_t frs1);

/* table to convert fractional LMUL value */
static const float flmul_table[8] = {
    1, 2, 4, 8,      /* LMUL */
    -1,              /* reserved */
    0.125, 0.25, 0.5 /* fractional LMUL */
};
#endif
