#ifndef THEAD_CLINT_H
#define THEAD_CLINT_H

#include "hw/irq.h"
#include "qemu/timer.h"

#define TYPE_THEAD_CLINT "thead_clint"
#define THEAD_CLINT(obj) \
    OBJECT_CHECK(THEADCLINTState, (obj), TYPE_THEAD_CLINT)

typedef struct THEADCLINTCState {
    /*< private >*/
    SysBusDevice parent_obj;

    uint32_t msip;
    uint64_t mtimecmp;
    QEMUTimer *timer;
    MemoryRegion mmio;
    qemu_irq irq[2];
} THEADCLINTState;
DeviceState *thead_clint_create(hwaddr addr, qemu_irq msip,
                                qemu_irq mtip);
#endif
