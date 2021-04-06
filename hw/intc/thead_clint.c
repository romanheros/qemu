/*
 * T-HEAD CLINT (Core Local Interruptor)
 *
 * Copyright (c) 2021 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * This provides real-time clock, timer and interprocessor interrupts.
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
#include "qemu/log.h"
#include "qapi/error.h"
#include "qemu/module.h"
#include "hw/sysbus.h"
#include "target/riscv/cpu.h"
#include "hw/intc/thead_clint.h"
#include "qemu/timer.h"

static uint64_t cpu_riscv_read_rtc(void)
{
    return muldiv64(qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL),
                    10000000, NANOSECONDS_PER_SECOND);
}

static void thead_clint_mtimecmp_cb(void *opaque)
{
    THEADCLINTState *s = (THEADCLINTState *)opaque;
    qemu_irq_pulse(s->irq[1]);
}

/*
 * Called when timecmp is written to update the QEMU timer or immediately
 * trigger timer interrupt if mtimecmp <= current timer value.
 */
static void thead_clint_write_timecmp(THEADCLINTState *s, RISCVCPU *cpu,
                                      uint64_t value)
{
    uint64_t rtc = cpu_riscv_read_rtc();
    uint64_t cmp = s->mtimecmp = value;
    uint64_t diff = cmp - rtc;
    uint64_t next_ns = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) +
                                         muldiv64(diff, NANOSECONDS_PER_SECOND,
                                                  10000000);

    if (cmp <= rtc) {
        /*
         * if we're setting a timecmp value in the "past",
         * immediately raise the timer interrupt
         */
        qemu_irq_pulse(s->irq[1]);
    } else {
        /* otherwise, set up the future timer interrupt */
        timer_mod(s->timer, next_ns);
    }
}

/* CPU wants to read rtc or timecmp register */
static uint64_t thead_clint_read(void *opaque, hwaddr addr, unsigned size)
{
    THEADCLINTState *clint = opaque;

    /* reads must be 4 byte aligned words */
    if ((addr & 0x3) != 0 || size != 4) {
        qemu_log_mask(LOG_GUEST_ERROR,
            "clint: invalid read size %u: 0x%" HWADDR_PRIx "\n", size, addr);
        return 0;
    }

    if (addr == 0) {
        return clint->msip;
    } else if (addr == 0x4000) {
        /* timecmp_lo */
        uint64_t timecmp = clint->mtimecmp;
        return timecmp & 0xFFFFFFFF;
    } else if (addr == 0x4004) {
        /* timecmp_hi */
        uint64_t timecmp = clint->mtimecmp;
        return (timecmp >> 32) & 0xFFFFFFFF;
    } else if (addr == 0xbff8) {
        /* time_lo */
        return cpu_riscv_read_rtc() & 0xFFFFFFFF;
    } else if (addr == 0xbffc) {
        /* time_hi */
        return (cpu_riscv_read_rtc() >> 32) & 0xFFFFFFFF;
    } else {
        qemu_log_mask(LOG_GUEST_ERROR,
            "clint: invalid read: 0x%" HWADDR_PRIx "\n", addr);
    }

    return 0;
}

/* CPU wrote to rtc or timecmp register */
static void thead_clint_write(void *opaque, hwaddr addr, uint64_t value,
                              unsigned size)
{
    THEADCLINTState *clint = opaque;

    /* writes must be 4 byte aligned words */
    if ((addr & 0x3) != 0 || size != 4) {
        qemu_log_mask(LOG_GUEST_ERROR,
            "clint: invalid write size %u: 0x%" HWADDR_PRIx "\n", size, addr);
        return;
    }

    if (addr == 0x0) {
        qemu_irq_pulse(clint->irq[0]);
        clint->msip = 0x1;
    } else if (addr == 0x4000) {
        /* timecmp_lo */
        uint64_t timecmp_hi = clint->mtimecmp >> 32;
        thead_clint_write_timecmp(clint, RISCV_CPU(current_cpu),
                                  timecmp_hi << 32 | (value & 0xFFFFFFFF));
    } else if (addr == 0x4004) {
        /* timecmp_hi */
        uint64_t timecmp_lo = clint->mtimecmp;
        thead_clint_write_timecmp(clint, RISCV_CPU(current_cpu),
                                  value << 32 | (timecmp_lo & 0xFFFFFFFF));
    } else if (addr == 0xbff8) {
        /* time_lo */
        qemu_log_mask(LOG_UNIMP, "clint: time_lo write not implemented\n");
    } else if (addr == 0xbffc) {
        /* time_hi */
        qemu_log_mask(LOG_UNIMP, "clint: time_hi write not implemented");
    } else {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "clint: invalid write: 0x%" HWADDR_PRIx "\n", addr);
    }
}

static const MemoryRegionOps thead_clint_ops = {
    .read = thead_clint_read,
    .write = thead_clint_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 4,
        .max_access_size = 4
    }
};

static void thead_clint_init(Object *obj)
{
    THEADCLINTState *s = THEAD_CLINT(obj);
    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);

    s->timer = timer_new_ns(QEMU_CLOCK_VIRTUAL,
                            &thead_clint_mtimecmp_cb, s);

    qdev_init_gpio_out(DEVICE(obj), s->irq, 2);

    memory_region_init_io(&s->mmio, obj, &thead_clint_ops, s,
                          TYPE_THEAD_CLINT, 0x10000);
    sysbus_init_mmio(sbd, &s->mmio);
}

static const TypeInfo thead_clint_info = {
    .name          = TYPE_THEAD_CLINT,
    .parent        = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(THEADCLINTState),
    .instance_init = thead_clint_init
};

static void thead_clint_register_types(void)
{
    type_register_static(&thead_clint_info);
}

type_init(thead_clint_register_types)

/*
 * Create CLINT device.
 */
DeviceState *thead_clint_create(hwaddr addr, qemu_irq msip,
                                qemu_irq mtip)
{
    DeviceState *dev = qdev_new(TYPE_THEAD_CLINT);
    sysbus_realize_and_unref(SYS_BUS_DEVICE(dev), &error_fatal);
    sysbus_mmio_map(SYS_BUS_DEVICE(dev), 0, addr);
    qdev_connect_gpio_out(dev, 0, msip);
    qdev_connect_gpio_out(dev, 1, mtip);

    return dev;
}
