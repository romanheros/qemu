/*
 * THEAD UART interface.
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

#ifndef THEAD_UART_H
#define THEAD_UART_H

#include "chardev/char-fe.h"
#include "exec/memory.h"
#include "chardev/char.h"
#include "hw/sysbus.h"
#include "qom/object.h"

#define TYPE_THEAD_UART  "thead_uart"
#define THEAD_UART(obj)  OBJECT_CHECK(thead_uart_state, (obj), TYPE_THEAD_UART)

typedef struct thead_uart_state {
    SysBusDevice parent_obj;

    MemoryRegion iomem;
    uint32_t dll;   /* Divisor Latch Low */
    uint32_t dlh;   /* Divisor Latch High */
    uint32_t ier;   /* Interrupt Enable Register */
    uint32_t iir;   /* Interrupt Identity Register */
    uint32_t fcr;   /* FIFO control register */
    uint32_t lcr;   /* line control register */
    uint32_t mcr;   /* modem control register */
    uint32_t lsr;   /* line status register */
    uint32_t msr;   /* modem status register */
    uint32_t usr;   /* uart status register */
    uint32_t rx_fifo[16];
    int rx_pos;
    int rx_count;
    int rx_trigger;
    CharBackend chr;
    qemu_irq irq;
} thead_uart_state;

DeviceState *thead_uart_create(hwaddr, qemu_irq, Chardev *);
#endif

