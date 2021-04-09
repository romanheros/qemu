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

#include "qemu/osdep.h"
#include "qemu/units.h"
#include "qapi/error.h"
#include "qemu-common.h"
#include "target/riscv/cpu.h"
#include "hw/sysbus.h"
#include "sysemu/sysemu.h"
#include "hw/boards.h"
#include "hw/loader.h"
#include "exec/address-spaces.h"
#include "qemu/error-report.h"
#include "elf.h"
#include "hw/riscv/riscv_hart.h"
#include "hw/riscv/smartl.h"
#include "hw/intc/riscv_clic.h"
#include "hw/intc/thead_clint.h"
#include "hw/char/thead_uart.h"
#include "hw/timer/thead_timer.h"

static const MemMapEntry smartl_memmap[] = {
    [SMARTL_SRAM0] =            {  0x00000000, 16 * MiB },
    [SMARTL_SRAM1] =            {  0x20000000, 16 * MiB },
    [SMARTL_TIMER] =            {  0x40011000, 16 * KiB },
    [SMARTL_UART] =             {  0x40015000, 4  * MiB },
    [SMARTL_SRAM2] =            {  0x50000000, 16 * MiB },
    [SMARTL_SRAM3] =            {  0x60000000, 16 * MiB },
    [SMARTL_CLINT] =            {  0xe0000000, 64 * KiB },
    [SMARTL_CLIC] =             {  0xe0800000, 20 * KiB },
};

static uint64_t load_kernel(CPURISCVState* env, const char *kernel_filename)
{
    uint64_t kernel_entry = 0, kernel_low = 0, kernel_high = 0;

    if (load_elf(kernel_filename, NULL, NULL, NULL,
                 &kernel_entry, &kernel_low, &kernel_high,
                 0, 0, EM_RISCV, 1, 0) < 0) {
        error_report("qemu: could not load kernel '%s'", kernel_filename);
        exit(1);
    }
    env->pc = (uint32_t)kernel_entry;
    return kernel_entry;
}

static void
smartl_add_memory_subregion(MemoryRegion *sysmem, hwaddr base, hwaddr size,
                            const char *name)
{
    MemoryRegion *ram = g_new(MemoryRegion, 1);
    memory_region_init_ram(ram, NULL, name, size, &error_fatal);
    memory_region_add_subregion(sysmem, base, ram);
}

static void smartl_init(MachineState *machine)
{
    CPURISCVState *env;
    DeviceState *dev;
    MemoryRegion *sysmem = get_system_memory();
    qemu_irq *irqs = g_malloc0(sizeof(qemu_irq) * SMARTL_CLIC_IRQ_NUMS);
    int i;

    /* Create cpu */
    Object *cpuobj = object_new(machine->cpu_type);;
    object_property_set_bool(cpuobj, "realized", true, &error_fatal);
    env = &RISCV_CPU(cpuobj)->env;

    /* Add memory region */
    for (i = 0; i < 4; i++) {
        MemMapEntry map = smartl_memmap[SMARTL_SRAM0 + i];
        char *name = g_strdup_printf("riscv.thead.smartl.ram.%d", i);
        smartl_add_memory_subregion(sysmem, map.base, map.size, name);
        g_free(name);
    }

    /* Create CLIC */
    dev = riscv_clic_create(smartl_memmap[SMARTL_CLIC].base,
                            false, false,
                            SMARTL_CLIC_HARTS,
                            SMARTL_CLIC_IRQ_NUMS,
                            SMARTL_CLIC_INTCTLBITS,
                            SMARTL_CLIC_VERSION);
    for (i = 0; i < SMARTL_CLIC_IRQ_NUMS; i++) {
        irqs[i] = qdev_get_gpio_in(dev, i);
    }

    /* Create CLINT */
    thead_clint_create(smartl_memmap[SMARTL_CLINT].base, irqs[3], irqs[7]);

    /* Create THEAD UART */
    thead_uart_create(smartl_memmap[SMARTL_UART].base, irqs[0x10], serial_hd(0));

    /* Create THEAD timer */
    thead_timer_set_freq(1000000000ll);
    sysbus_create_varargs("thead_timer",
                          smartl_memmap[SMARTL_TIMER].base, irqs[0x12], irqs[0x13],
                          irqs[0x14], irqs[0x15], NULL);

    if (machine->kernel_filename) {
        load_kernel(env, machine->kernel_filename);
    }
}

static void smartl_class_init(ObjectClass *oc, void *data)
{
    MachineClass *mc = MACHINE_CLASS(oc);

    mc->desc = "RISC-V smartl";
    mc->init = smartl_init;
    mc->default_cpu_type = RISCV_CPU_TYPE_NAME("e902");
}

static const TypeInfo smartl_type = {
    .name = MACHINE_TYPE_NAME("smartl"),
    .parent = TYPE_MACHINE,
    .class_init = smartl_class_init,
};

static void smartl_machine_init(void)
{
    type_register_static(&smartl_type);
}

type_init(smartl_machine_init)
