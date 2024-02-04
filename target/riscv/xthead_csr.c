/*
 * Xuantie implementation for RISC-V Control and Status Registers.
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
#include "qemu/log.h"
#include "cpu.h"
#include "tcg/tcg-cpu.h"
#include "exec/exec-all.h"
#include "exec/tb-flush.h"
#include "qapi/error.h"

/* In XTheadVector, vcsr is inaccessible */
static RISCVException th_vcsr_check(CPURISCVState *env, int csrno)
{
    RISCVCPU *cpu = env_archcpu(env);
    if (cpu->cfg.ext_xtheadvector) {
        return false;
    }
    return vs(env, csrno);
}

static RISCVException
th_read_fcsr(CPURISCVState *env, int csrno, target_ulong *val)
{
    RISCVCPU *cpu = env_archcpu(env);
    RISCVException ret = read_fcsr(env, csrno, val);
    if (cpu->cfg.ext_xtheadvector) {
        *val = set_field(*val, TH_FSR_VXRM,  env->vxrm);
        *val = set_field(*val, TH_FSR_VXSAT,  env->vxsat);
    }
    return ret;
}

static RISCVException
th_write_fcsr(CPURISCVState *env, int csrno, target_ulong val)
{
    RISCVCPU *cpu = env_archcpu(env);
    if (cpu->cfg.ext_xtheadvector) {
        env->vxrm = get_field(val, TH_FSR_VXRM);
        env->vxsat = get_field(val, TH_FSR_VXSAT);
    }
    return write_fcsr(env, csrno, val);
}

/*
 * We use the RVV1.0 format for env->vtype
 * When reading vtype, we need to change the format.
 * In RVV1.0:
 *   vtype[7] -> vma
 *   vtype[6] -> vta
 *   vtype[5:3] -> vsew
 *   vtype[2:0] -> vlmul
 * In XTheadVector:
 *   vtype[6:5] -> vediv
 *   vtype[4:2] -> vsew
 *   vtype[1:0] -> vlmul
 * Although vlmul size is different between RVV1.0 and XTheadVector,
 * the lower 2 bits have the same meaning.
 * vma, vta and vediv are useless in XTheadVector, So we need to clear
 * vtype[7:5] for XTheadVector
 */
static RISCVException
th_read_vtype(CPURISCVState *env, int csrno, target_ulong *val)
{
    RISCVCPU *cpu = env_archcpu(env);
    RISCVException ret = read_vtype(env, csrno, val);
    if (cpu->cfg.ext_xtheadvector) {
        *val = set_field(*val, TH_VTYPE_LMUL,
                          FIELD_EX64(*val, VTYPE, VLMUL));
        *val = set_field(*val, TH_VTYPE_SEW,
                          FIELD_EX64(*val, VTYPE, VSEW));
        *val = set_field(*val, TH_VTYPE_CLEAR, 0);
    }
    return ret;
}

#if !defined(CONFIG_USER_ONLY)
static RISCVException
th_read_mstatus(CPURISCVState *env, int csrno, target_ulong *val)
{
    RISCVCPU *cpu = env_archcpu(env);
    RISCVException ret = read_mstatus(env, csrno, val);
    if (cpu->cfg.ext_xtheadvector) {
        *val = set_field(*val, TH_MSTATUS_VS,
                         get_field(*val, MSTATUS_VS));
    }
    return ret;
}

static RISCVException
th_write_mstatus(CPURISCVState *env, int csrno, target_ulong val)
{
    RISCVCPU *cpu = env_archcpu(env);
    if (cpu->cfg.ext_xtheadvector) {
        val = set_field(val, MSTATUS_VS,
                        get_field(val, TH_MSTATUS_VS));
    }
    return write_mstatus(env, csrno, val);
}

static RISCVException
th_read_sstatus(CPURISCVState *env, int csrno, target_ulong *val)
{
    RISCVCPU *cpu = env_archcpu(env);
    RISCVException ret = read_sstatus(env, csrno, val);
    if (cpu->cfg.ext_xtheadvector) {
        *val = set_field(*val, TH_MSTATUS_VS,
                        get_field(*val, MSTATUS_VS));
    }
    return ret;
}

static RISCVException
th_write_sstatus(CPURISCVState *env, int csrno, target_ulong val)
{
    RISCVCPU *cpu = env_archcpu(env);
    if (cpu->cfg.ext_xtheadvector) {
        val = set_field(val, MSTATUS_VS,
                        get_field(val, TH_MSTATUS_VS));
    }
    return write_sstatus(env, csrno, val);
}
static RISCVException th_maee_check(CPURISCVState *env, int csrno)
{
    if (riscv_cpu_cfg(env)->ext_xtheadmaee) {
        return RISCV_EXCP_ILLEGAL_INST;
    }
    return RISCV_EXCP_NONE;
}

static RISCVException
read_th_mxstatus(CPURISCVState *env, int csrno, target_ulong *val)
{
    *val = env->th_mxstatus;
    return RISCV_EXCP_NONE;
}

static RISCVException
write_th_mxstatus(CPURISCVState *env, int csrno, target_ulong val)
{
    uint64_t mxstatus = env->th_mxstatus;
    uint64_t mask = TH_MXSTATUS_MAEE;

    if ((val ^ mxstatus) & TH_MXSTATUS_MAEE) {
        tlb_flush(env_cpu(env));
    }

    mxstatus = (mxstatus & ~mask) | (val & mask);
    env->th_mxstatus = mxstatus;
    return RISCV_EXCP_NONE;
}
#endif

riscv_csr_operations th_csr_ops[CSR_TABLE_SIZE] = {
    [CSR_FCSR]     = { "fcsr",     fs, th_read_fcsr,    th_write_fcsr},
    [CSR_VCSR]     = { "vcsr",     th_vcsr_check, read_vcsr, write_vcsr},
    [CSR_VTYPE]    = { "vtype",    vs, th_read_vtype                 },
#if !defined(CONFIG_USER_ONLY)
    [CSR_MSTATUS]  = { "mstatus",  any,   th_read_mstatus, th_write_mstatus},
    [CSR_SSTATUS]  = { "sstatus",  smode, th_read_sstatus, th_write_sstatus},
    [CSR_TH_MXSTATUS]     = { "th_mxstatus", th_maee_check, read_th_mxstatus,
                                                            write_th_mxstatus},
#endif /* !CONFIG_USER_ONLY */
};
