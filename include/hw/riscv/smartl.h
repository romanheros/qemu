/*
 * RISCV SMARTL System emulation.
 *
 * Copyright (c) 2021 T-Head Semiconductor Co., Ltd. All rights reserved.
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

#ifndef HW_SMARTL_H
#define HW_SMARTL_H

#define SMARTL_CLIC_IRQ_NUMS 256
#define SMARTL_CLIC_VERSION "v0.8"
#define SMARTL_CLIC_HARTS 1
#define SMARTL_CLIC_INTCTLBITS 3

enum {
    SMARTL_SRAM0,
    SMARTL_SRAM1,
    SMARTL_SRAM2,
    SMARTL_SRAM3,
    SMARTL_TIMER,
    SMARTL_UART,
    SMARTL_CLINT,
    SMARTL_CLIC
};

#endif
