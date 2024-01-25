/*
 * RISC-V translation routines for the XTheadVector Extension.
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
#include "tcg/tcg-op-gvec.h"
#include "tcg/tcg-gvec-desc.h"
#include "internals.h"

static bool require_thv(DisasContext *s)
{
    return s->cfg_ptr->ext_xtheadvector;
}

static bool th_vsetvl(DisasContext *s, int rd, int rs1, TCGv s2)
{
    return do_vsetvl(s, rd, rs1, s2);
}

/*
 * XTheadVector has different vtype encoding from RVV1.0.
 * We recode the value in RVV1.0 vtype format to reuse the RVV1.0 functions.
 * In RVV1.0:
 *   vtype[7] -> vma
 *   vtype[6] -> vta
 *   vtype[5:3] -> vsew
 *   vtype[2:0] -> vlmul
 * In XTheadVector:
 *   vtype[6:5] -> vediv (reserved)
 *   vtype[4:2] -> vsew
 *   vtype[1:0] -> vlmul
 *
 * Also th_vsetvl does not have feature of keeping existing vl when
 * (rd == 0 && rs1 == 0), So we need to set cpu_vl to RV_VLEN_MAX to make
 * it work as (rs1 == 0) branch.
 */
static bool trans_th_vsetvl(DisasContext *s, arg_th_vsetvl *a)
{
    if (!require_thv(s)) {
        return false;
    }

    TCGv s2 = get_gpr(s, a->rs2, EXT_ZERO);
    TCGv temp = tcg_temp_new();
    TCGv dst = tcg_temp_new();
    /* illegal value check*/
    TCGLabel *legal = gen_new_label();
    tcg_gen_shri_tl(temp, s2, 5);
    tcg_gen_brcondi_tl(TCG_COND_EQ, temp, 0, legal);
    /*
     * if illegal, set unsupported value
     * s2[8:5] == 0b111, which is reserved field in XTheadVector
     */
    tcg_gen_movi_tl(s2, 0xff);
    gen_set_label(legal);
    /* get vlmul, s2[1:0] -> dst[2:0] */
    tcg_gen_andi_tl(dst, s2, 0x3);
    /* get vsew, s2[4:2] -> dst[5:3] */
    tcg_gen_andi_tl(temp, s2, 0x1c);
    tcg_gen_shli_tl(temp, temp, 1);
    tcg_gen_or_tl(dst, dst, temp);
    /*
     * get reserved field when illegal, s2[7:5] -> dst[10:8]
     * avoid dst[7:6], because dst[7:6] are vma and vta.
     *
     * Make the dst an illegal value for RVV1.0, leads to the illegal
     * operation processing flow.
     */
    tcg_gen_andi_tl(temp, s2, 0xe0);
    tcg_gen_shli_tl(temp, temp, 3);
    tcg_gen_or_tl(dst, dst, temp);
    if (a->rd == 0 && a->rs1 == 0) {
        cpu_vl = tcg_constant_tl(RV_VLEN_MAX);
    }
    return th_vsetvl(s, a->rd, a->rs1, dst);
}

static bool trans_th_vsetvli(DisasContext *s, arg_th_vsetvli *a)
{
    if (!require_thv(s)) {
        return false;
    }

    TCGv s2 = tcg_constant_tl(a->zimm);
    TCGv temp = tcg_temp_new();
    TCGv dst = tcg_temp_new();
    /* illegal value check*/
    TCGLabel *legal = gen_new_label();
    tcg_gen_shri_tl(temp, s2, 5);
    tcg_gen_brcondi_tl(TCG_COND_EQ, temp, 0, legal);
    /* if illegal, set unsupported value */
    tcg_gen_movi_tl(s2, 0xff);
    gen_set_label(legal);
    /* get vlmul */
    tcg_gen_andi_tl(dst, s2, 0x3);
    /* get vsew */
    tcg_gen_andi_tl(temp, s2, 0x1c);
    tcg_gen_shli_tl(temp, temp, 1);
    tcg_gen_or_tl(dst, dst, temp);
    /* get reserved field when illegal*/
    tcg_gen_andi_tl(temp, s2, 0xe0);
    tcg_gen_shli_tl(temp, temp, 3);
    tcg_gen_or_tl(dst, dst, temp);
    if (a->rd == 0 && a->rs1 == 0) {
        cpu_vl = tcg_constant_tl(RV_VLEN_MAX);
    }
    return th_vsetvl(s, a->rd, a->rs1, dst);
}

static uint32_t th_vreg_ofs(DisasContext *s, int reg)
{
    return vreg_ofs(s, reg);
}

/* check functions */

/*
 * There are two rules check here.
 * 1. Vector register numbers are multiples of LMUL.
 * 2. For all widening instructions, the destination LMUL value must also be
 *    a supported LMUL value.
 *
 * This function is the combination of require_align and vext_wide_check_common,
 * except:
 * 1) In require_align, if LMUL < 0, i.e. fractional LMUL, any vector register
 *    is allowed, we do not need to check this situation.
 * 2) In vext_wide_check_common, RVV check all the constraints of widen
 *    instruction, including SEW < 64, 2*SEW <ELEN and overlap constraints.
 *    But in th_check_reg, we only check the reg constraint, including lmul < 8
 *    and destination vector register number is multiples of 2 * LMUL.
 *    Other widen constraints are checked in other functions.
 */
static bool th_check_reg(DisasContext *s, uint32_t reg, bool widen)
{
    /*
     * The destination vector register group results are arranged as if both
     * SEW and LMUL were at twice their current settings.
     */
    int legal = widen ? 2 << s->lmul : 1 << s->lmul;

    return !((s->lmul == 0x3 && widen) || (reg % legal));
}

/*
 * There are two rules check here.
 * 1. The destination vector register group for a masked vector instruction can
 *    only overlap the source mask register (v0) when LMUL=1.
 * 2. In widen instructions and some other insturctions, like vslideup.vx,
 *    there is no need to check whether LMUL=1.
 *
 * This function is almost the copy of require_vm, except:
 * 1) In RVV1.0, destination vector register group cannot overlap source mask
 *    register even when LMUL=1. So we have to add a check of "(s->lmul == 0)".
 * 2) When the instruction forbid the mask overlap in all situation, we add
 *    a arg of "force" to flag the situation.
 */
static bool th_check_overlap_mask(DisasContext *s, uint32_t vd, bool vm,
                                  bool force)
{
    return (vm != 0 || vd != 0) || (!force && (s->lmul == 0));
}

/*
 * The LMUL setting must be such that LMUL * NFIELDS <= 8.
 *
 * This function is almost the copy of require_nf, except that
 * XTheadVectot does not have fractional LMUL, so we do not need to
 * max(lmul, 0)
 * RVV use "size = nf << MAX(lmul, 0)" to let the one segment be loaded
 * to at least one vector register.
 */
static bool th_check_nf(DisasContext *s, uint32_t vd, uint32_t nf)
{
    int size = nf << s->lmul;
    return size <= 8 && vd + size <= 32;
}

/*
 * The destination vector register group cannot overlap a source vector register
 * group of a different element width.
 *
 * The overlap check rule is different from RVV1.0. The function
 * "require_noover" describes the check rule in RVV1.0. In general, in some
 * situations, the destination vector register group can overlap the source
 * vector register group of a different element width. While XTheadVector not.
 */
static inline bool th_check_overlap_group(int rd, int dlen, int rs, int slen)
{
    return ((rd >= rs + slen) || (rs >= rd + dlen));
}

static bool th_check_isa_ill(DisasContext *s)
{
    return vext_check_isa_ill(s);
}

/* common translation macro */
/*
 * GEN_TH_TRANS is similar to GEN_VEXT_TRANS
 * just change the function args
 */
#define GEN_TH_TRANS(NAME, SEQ, ARGTYPE, OP, CHECK)        \
static bool trans_##NAME(DisasContext *s, arg_##ARGTYPE *a)\
{                                                          \
    if (CHECK(s, a)) {                                     \
        return OP(s, a, SEQ);                              \
    }                                                      \
    return false;                                          \
}

/*
 *** stride load and store
 */

#define gen_helper_ldst_stride_th gen_helper_ldst_stride

static bool ldst_stride_trans_th(uint32_t vd, uint32_t rs1, uint32_t rs2,
                                 uint32_t data, gen_helper_ldst_stride_th *fn,
                                 DisasContext *s, bool is_store)
{
    return ldst_stride_trans(vd, rs1, rs2, data, fn, s, is_store);
}
/*
 * This function is almost the copy of ld_stride_op, except:
 * 1) XTheadVector has more insns to handle zero/sign-extended.
 * 2) XTheadVector using different data encoding, add MLEN,
 *    delete VTA and VMA.
 *
 *    MLEN = SEW/LMUL. to indicate the mask bit.
 */
static bool ld_stride_op_th(DisasContext *s, arg_rnfvm *a, uint8_t seq)
{
    uint32_t data = 0;
    gen_helper_ldst_stride_th *fn;
    static gen_helper_ldst_stride_th * const fns[7][4] = {
        { gen_helper_th_vlsb_v_b,  gen_helper_th_vlsb_v_h,
          gen_helper_th_vlsb_v_w,  gen_helper_th_vlsb_v_d },
        { NULL,                    gen_helper_th_vlsh_v_h,
          gen_helper_th_vlsh_v_w,  gen_helper_th_vlsh_v_d },
        { NULL,                    NULL,
          gen_helper_th_vlsw_v_w,  gen_helper_th_vlsw_v_d },
        { gen_helper_th_vlse_v_b,  gen_helper_th_vlse_v_h,
          gen_helper_th_vlse_v_w,  gen_helper_th_vlse_v_d },
        { gen_helper_th_vlsbu_v_b, gen_helper_th_vlsbu_v_h,
          gen_helper_th_vlsbu_v_w, gen_helper_th_vlsbu_v_d },
        { NULL,                    gen_helper_th_vlshu_v_h,
          gen_helper_th_vlshu_v_w, gen_helper_th_vlshu_v_d },
        { NULL,                    NULL,
          gen_helper_th_vlswu_v_w, gen_helper_th_vlswu_v_d },
    };

    fn =  fns[seq][s->sew];
    if (fn == NULL) {
        return false;
    }
    /* Need extra mlen to find the mask bit */
    data = FIELD_DP32(data, VDATA_TH, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_TH, VM, a->vm);
    data = FIELD_DP32(data, VDATA_TH, LMUL, s->lmul);
    data = FIELD_DP32(data, VDATA_TH, NF, a->nf);
    return ldst_stride_trans_th(a->rd, a->rs1, a->rs2, data, fn, s, false);
}

/*
 * check function
 * 1) check overlap mask, XTheadVector can overlap mask reg v0 when
 *    lmul == 1, while RVV1.0 can not.
 * 2) check reg, XTheadVector Vector register numbers are multiples
 *    of integral LMUL, while RVV1.0 has fractional LMUL, which allows
 *    any vector register.
 * 3) check nf, the LMUL setting must be such that LMUL * NFIELDS <= 8,
 *    which is the same as RVV1.0. But we do not need to check fractional
 *    LMUL.
 */
static bool ld_stride_check_th(DisasContext *s, arg_rnfvm* a)
{
    return (require_thv(s) &&
            th_check_isa_ill(s) &&
            th_check_overlap_mask(s, a->rd, a->vm, false) &&
            th_check_reg(s, a->rd, false) &&
            th_check_nf(s, a->rd, a->nf));
}

GEN_TH_TRANS(th_vlsb_v, 0, rnfvm, ld_stride_op_th, ld_stride_check_th)
GEN_TH_TRANS(th_vlsh_v, 1, rnfvm, ld_stride_op_th, ld_stride_check_th)
GEN_TH_TRANS(th_vlsw_v, 2, rnfvm, ld_stride_op_th, ld_stride_check_th)
GEN_TH_TRANS(th_vlse_v, 3, rnfvm, ld_stride_op_th, ld_stride_check_th)
GEN_TH_TRANS(th_vlsbu_v, 4, rnfvm, ld_stride_op_th, ld_stride_check_th)
GEN_TH_TRANS(th_vlshu_v, 5, rnfvm, ld_stride_op_th, ld_stride_check_th)
GEN_TH_TRANS(th_vlswu_v, 6, rnfvm, ld_stride_op_th, ld_stride_check_th)

/*
 * This function is almost the copy of st_stride_op, except:
 * 1) XTheadVector using different data encoding, add MLEN,
 *    delete VTA and VMA.
 * 2) XTheadVector has more situations. vss{8,16,32,64}.v decide the
 *    reg and mem width both equal 8/16/32/64. As for th.vss{b,h,w}.v, the
 *    reg width equals SEW, and the mem width equals 8/16/32. The reg and
 *    mem width of th.vsse.v both equal SEW. Therefore, we need to add more
 *    helper functions depending on SEW.
 */
static bool st_stride_op_th(DisasContext *s, arg_rnfvm *a, uint8_t seq)
{
    uint32_t data = 0;
    gen_helper_ldst_stride_th *fn;
    static gen_helper_ldst_stride_th * const fns[4][4] = {
        /* masked stride store */
        { gen_helper_th_vssb_v_b,  gen_helper_th_vssb_v_h,
          gen_helper_th_vssb_v_w,  gen_helper_th_vssb_v_d },
        { NULL,                    gen_helper_th_vssh_v_h,
          gen_helper_th_vssh_v_w,  gen_helper_th_vssh_v_d },
        { NULL,                    NULL,
          gen_helper_th_vssw_v_w,  gen_helper_th_vssw_v_d },
        { gen_helper_th_vsse_v_b,  gen_helper_th_vsse_v_h,
          gen_helper_th_vsse_v_w,  gen_helper_th_vsse_v_d }
    };

    data = FIELD_DP32(data, VDATA_TH, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_TH, VM, a->vm);
    data = FIELD_DP32(data, VDATA_TH, LMUL, s->lmul);
    data = FIELD_DP32(data, VDATA_TH, NF, a->nf);
    fn =  fns[seq][s->sew];
    if (fn == NULL) {
        return false;
    }

    return ldst_stride_trans_th(a->rd, a->rs1, a->rs2, data, fn, s, true);
}

/* store does not need to check overlap */
static bool st_stride_check_th(DisasContext *s, arg_rnfvm* a)
{
    return (require_thv(s) &&
            th_check_isa_ill(s) &&
            th_check_reg(s, a->rd, false) &&
            th_check_nf(s, a->rd, a->nf));
}

GEN_TH_TRANS(th_vssb_v, 0, rnfvm, st_stride_op_th, st_stride_check_th)
GEN_TH_TRANS(th_vssh_v, 1, rnfvm, st_stride_op_th, st_stride_check_th)
GEN_TH_TRANS(th_vssw_v, 2, rnfvm, st_stride_op_th, st_stride_check_th)
GEN_TH_TRANS(th_vsse_v, 3, rnfvm, st_stride_op_th, st_stride_check_th)

/*
 *** unit stride load and store
 */

#define gen_helper_ldst_us_th gen_helper_ldst_us

static bool ldst_us_trans_th(uint32_t vd, uint32_t rs1, uint32_t data,
                             gen_helper_ldst_us_th *fn, DisasContext *s,
                             bool is_store)
{
    return ldst_us_trans(vd, rs1, data, fn, s, is_store);
}

/*
 * This function is almost the copy of ld_us_op, except:
 * 1) different data encoding
 * 2) XTheadVector has more insns to handle zero/sign-extended.
 */
static bool ld_us_op_th(DisasContext *s, arg_r2nfvm *a, uint8_t seq)
{
    uint32_t data = 0;
    gen_helper_ldst_us_th *fn;
    static gen_helper_ldst_us_th * const fns[2][7][4] = {
        /* masked unit stride load */
        { { gen_helper_th_vlb_v_b_mask,  gen_helper_th_vlb_v_h_mask,
            gen_helper_th_vlb_v_w_mask,  gen_helper_th_vlb_v_d_mask },
          { NULL,                        gen_helper_th_vlh_v_h_mask,
            gen_helper_th_vlh_v_w_mask,  gen_helper_th_vlh_v_d_mask },
          { NULL,                        NULL,
            gen_helper_th_vlw_v_w_mask,  gen_helper_th_vlw_v_d_mask },
          { gen_helper_th_vle_v_b_mask,  gen_helper_th_vle_v_h_mask,
            gen_helper_th_vle_v_w_mask,  gen_helper_th_vle_v_d_mask },
          { gen_helper_th_vlbu_v_b_mask, gen_helper_th_vlbu_v_h_mask,
            gen_helper_th_vlbu_v_w_mask, gen_helper_th_vlbu_v_d_mask },
          { NULL,                        gen_helper_th_vlhu_v_h_mask,
            gen_helper_th_vlhu_v_w_mask, gen_helper_th_vlhu_v_d_mask },
          { NULL,                        NULL,
            gen_helper_th_vlwu_v_w_mask, gen_helper_th_vlwu_v_d_mask } },
        /* unmasked unit stride load */
        { { gen_helper_th_vlb_v_b,  gen_helper_th_vlb_v_h,
            gen_helper_th_vlb_v_w,  gen_helper_th_vlb_v_d },
          { NULL,                   gen_helper_th_vlh_v_h,
            gen_helper_th_vlh_v_w,  gen_helper_th_vlh_v_d },
          { NULL,                   NULL,
            gen_helper_th_vlw_v_w,  gen_helper_th_vlw_v_d },
          { gen_helper_th_vle_v_b,  gen_helper_th_vle_v_h,
            gen_helper_th_vle_v_w,  gen_helper_th_vle_v_d },
          { gen_helper_th_vlbu_v_b, gen_helper_th_vlbu_v_h,
            gen_helper_th_vlbu_v_w, gen_helper_th_vlbu_v_d },
          { NULL,                   gen_helper_th_vlhu_v_h,
            gen_helper_th_vlhu_v_w, gen_helper_th_vlhu_v_d },
          { NULL,                   NULL,
            gen_helper_th_vlwu_v_w, gen_helper_th_vlwu_v_d } }
    };

    fn =  fns[a->vm][seq][s->sew];
    if (fn == NULL) {
        return false;
    }

    data = FIELD_DP32(data, VDATA_TH, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_TH, VM, a->vm);
    data = FIELD_DP32(data, VDATA_TH, LMUL, s->lmul);
    data = FIELD_DP32(data, VDATA_TH, NF, a->nf);
    return ldst_us_trans_th(a->rd, a->rs1, data, fn, s, false);
}

static bool ld_us_check_th(DisasContext *s, arg_r2nfvm* a)
{
    return (require_thv(s) &&
            th_check_isa_ill(s) &&
            th_check_overlap_mask(s, a->rd, a->vm, false) &&
            th_check_reg(s, a->rd, false) &&
            th_check_nf(s, a->rd, a->nf));
}

GEN_TH_TRANS(th_vlb_v, 0, r2nfvm, ld_us_op_th, ld_us_check_th)
GEN_TH_TRANS(th_vlh_v, 1, r2nfvm, ld_us_op_th, ld_us_check_th)
GEN_TH_TRANS(th_vlw_v, 2, r2nfvm, ld_us_op_th, ld_us_check_th)
GEN_TH_TRANS(th_vle_v, 3, r2nfvm, ld_us_op_th, ld_us_check_th)
GEN_TH_TRANS(th_vlbu_v, 4, r2nfvm, ld_us_op_th, ld_us_check_th)
GEN_TH_TRANS(th_vlhu_v, 5, r2nfvm, ld_us_op_th, ld_us_check_th)
GEN_TH_TRANS(th_vlwu_v, 6, r2nfvm, ld_us_op_th, ld_us_check_th)

/*
 * This function is almost the copy of st_us_op, except:
 * 1) different data encoding.
 * 2) XTheadVector has more situations, depending on SEW.
 */
static bool st_us_op_th(DisasContext *s, arg_r2nfvm *a, uint8_t seq)
{
    uint32_t data = 0;
    gen_helper_ldst_us_th *fn;
    static gen_helper_ldst_us_th * const fns[2][4][4] = {
        /* masked unit stride load and store */
        { { gen_helper_th_vsb_v_b_mask,  gen_helper_th_vsb_v_h_mask,
            gen_helper_th_vsb_v_w_mask,  gen_helper_th_vsb_v_d_mask },
          { NULL,                        gen_helper_th_vsh_v_h_mask,
            gen_helper_th_vsh_v_w_mask,  gen_helper_th_vsh_v_d_mask },
          { NULL,                        NULL,
            gen_helper_th_vsw_v_w_mask,  gen_helper_th_vsw_v_d_mask },
          { gen_helper_th_vse_v_b_mask,  gen_helper_th_vse_v_h_mask,
            gen_helper_th_vse_v_w_mask,  gen_helper_th_vse_v_d_mask } },
        /* unmasked unit stride store */
        { { gen_helper_th_vsb_v_b,  gen_helper_th_vsb_v_h,
            gen_helper_th_vsb_v_w,  gen_helper_th_vsb_v_d },
          { NULL,                   gen_helper_th_vsh_v_h,
            gen_helper_th_vsh_v_w,  gen_helper_th_vsh_v_d },
          { NULL,                   NULL,
            gen_helper_th_vsw_v_w,  gen_helper_th_vsw_v_d },
          { gen_helper_th_vse_v_b,  gen_helper_th_vse_v_h,
            gen_helper_th_vse_v_w,  gen_helper_th_vse_v_d } }
    };

    fn =  fns[a->vm][seq][s->sew];
    if (fn == NULL) {
        return false;
    }

    data = FIELD_DP32(data, VDATA_TH, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_TH, VM, a->vm);
    data = FIELD_DP32(data, VDATA_TH, LMUL, s->lmul);
    data = FIELD_DP32(data, VDATA_TH, NF, a->nf);
    return ldst_us_trans_th(a->rd, a->rs1, data, fn, s, true);
}

static bool st_us_check_th(DisasContext *s, arg_r2nfvm* a)
{
    return (require_thv(s) &&
            th_check_isa_ill(s) &&
            th_check_reg(s, a->rd, false) &&
            th_check_nf(s, a->rd, a->nf));
}

GEN_TH_TRANS(th_vsb_v, 0, r2nfvm, st_us_op_th, st_us_check_th)
GEN_TH_TRANS(th_vsh_v, 1, r2nfvm, st_us_op_th, st_us_check_th)
GEN_TH_TRANS(th_vsw_v, 2, r2nfvm, st_us_op_th, st_us_check_th)
GEN_TH_TRANS(th_vse_v, 3, r2nfvm, st_us_op_th, st_us_check_th)

/*
 *** index load and store
 */

#define gen_helper_ldst_index_th gen_helper_ldst_index

static bool ldst_index_trans_th(uint32_t vd, uint32_t rs1, uint32_t vs2,
                                uint32_t data, gen_helper_ldst_index_th *fn,
                                DisasContext *s, bool is_store)
{
    return ldst_index_trans(vd, rs1, vs2, data, fn, s, is_store);
}

/*
 * This function is almost the copy of ld_index_op, except:
 * 1) different data encoding
 * 2) XTheadVector has more insns to handle zero/sign-extended.
 */
static bool ld_index_op_th(DisasContext *s, arg_rnfvm *a, uint8_t seq)
{
    uint32_t data = 0;
    gen_helper_ldst_index_th *fn;
    static gen_helper_ldst_index_th * const fns[7][4] = {
        { gen_helper_th_vlxb_v_b,  gen_helper_th_vlxb_v_h,
          gen_helper_th_vlxb_v_w,  gen_helper_th_vlxb_v_d },
        { NULL,                    gen_helper_th_vlxh_v_h,
          gen_helper_th_vlxh_v_w,  gen_helper_th_vlxh_v_d },
        { NULL,                    NULL,
          gen_helper_th_vlxw_v_w,  gen_helper_th_vlxw_v_d },
        { gen_helper_th_vlxe_v_b,  gen_helper_th_vlxe_v_h,
          gen_helper_th_vlxe_v_w,  gen_helper_th_vlxe_v_d },
        { gen_helper_th_vlxbu_v_b, gen_helper_th_vlxbu_v_h,
          gen_helper_th_vlxbu_v_w, gen_helper_th_vlxbu_v_d },
        { NULL,                    gen_helper_th_vlxhu_v_h,
          gen_helper_th_vlxhu_v_w, gen_helper_th_vlxhu_v_d },
        { NULL,                    NULL,
          gen_helper_th_vlxwu_v_w, gen_helper_th_vlxwu_v_d },
    };

    fn =  fns[seq][s->sew];
    if (fn == NULL) {
        return false;
    }

    data = FIELD_DP32(data, VDATA_TH, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_TH, VM, a->vm);
    data = FIELD_DP32(data, VDATA_TH, LMUL, s->lmul);
    data = FIELD_DP32(data, VDATA_TH, NF, a->nf);
    return ldst_index_trans_th(a->rd, a->rs1, a->rs2, data, fn, s, false);
}

/*
 * For vector indexed segment loads, the destination vector register
 * groups cannot overlap the source vector register group (specified by
 * `vs2`), else an illegal instruction exception is raised.
 */
static bool ld_index_check_th(DisasContext *s, arg_rnfvm* a)
{
    return (require_thv(s) &&
            th_check_isa_ill(s) &&
            th_check_overlap_mask(s, a->rd, a->vm, false) &&
            th_check_reg(s, a->rd, false) &&
            th_check_reg(s, a->rs2, false) &&
            th_check_nf(s, a->rd, a->nf) &&
            ((a->nf == 1) ||
            th_check_overlap_group(a->rd, a->nf << s->lmul,
                                   a->rs2, 1 << s->lmul)));
}

GEN_TH_TRANS(th_vlxb_v, 0, rnfvm, ld_index_op_th, ld_index_check_th)
GEN_TH_TRANS(th_vlxh_v, 1, rnfvm, ld_index_op_th, ld_index_check_th)
GEN_TH_TRANS(th_vlxw_v, 2, rnfvm, ld_index_op_th, ld_index_check_th)
GEN_TH_TRANS(th_vlxe_v, 3, rnfvm, ld_index_op_th, ld_index_check_th)
GEN_TH_TRANS(th_vlxbu_v, 4, rnfvm, ld_index_op_th, ld_index_check_th)
GEN_TH_TRANS(th_vlxhu_v, 5, rnfvm, ld_index_op_th, ld_index_check_th)
GEN_TH_TRANS(th_vlxwu_v, 6, rnfvm, ld_index_op_th, ld_index_check_th)

/*
 * This function is almost the copy of st_index_op, except:
 * 1) different data encoding.
 */
static bool st_index_op_th(DisasContext *s, arg_rnfvm *a, uint8_t seq)
{
    uint32_t data = 0;
    gen_helper_ldst_index_th *fn;
    static gen_helper_ldst_index_th * const fns[4][4] = {
        { gen_helper_th_vsxb_v_b,  gen_helper_th_vsxb_v_h,
          gen_helper_th_vsxb_v_w,  gen_helper_th_vsxb_v_d },
        { NULL,                    gen_helper_th_vsxh_v_h,
          gen_helper_th_vsxh_v_w,  gen_helper_th_vsxh_v_d },
        { NULL,                    NULL,
          gen_helper_th_vsxw_v_w,  gen_helper_th_vsxw_v_d },
        { gen_helper_th_vsxe_v_b,  gen_helper_th_vsxe_v_h,
          gen_helper_th_vsxe_v_w,  gen_helper_th_vsxe_v_d }
    };

    fn =  fns[seq][s->sew];
    if (fn == NULL) {
        return false;
    }

    data = FIELD_DP32(data, VDATA_TH, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_TH, VM, a->vm);
    data = FIELD_DP32(data, VDATA_TH, LMUL, s->lmul);
    data = FIELD_DP32(data, VDATA_TH, NF, a->nf);
    return ldst_index_trans_th(a->rd, a->rs1, a->rs2, data, fn, s, true);
}

static bool st_index_check_th(DisasContext *s, arg_rnfvm* a)
{
    return (require_thv(s) &&
            th_check_isa_ill(s) &&
            th_check_reg(s, a->rd, false) &&
            th_check_reg(s, a->rs2, false) &&
            th_check_nf(s, a->rd, a->nf));
}

GEN_TH_TRANS(th_vsxb_v, 0, rnfvm, st_index_op_th, st_index_check_th)
GEN_TH_TRANS(th_vsxh_v, 1, rnfvm, st_index_op_th, st_index_check_th)
GEN_TH_TRANS(th_vsxw_v, 2, rnfvm, st_index_op_th, st_index_check_th)
GEN_TH_TRANS(th_vsxe_v, 3, rnfvm, st_index_op_th, st_index_check_th)

/*
 *** unit stride fault-only-first load
 */
static bool ldff_trans_th(uint32_t vd, uint32_t rs1, uint32_t data,
                          gen_helper_ldst_us_th *fn, DisasContext *s)
{
    return ldff_trans(vd, rs1, data, fn, s);
}
/*
 * This function is almost the copy of ldff_op, except:
 * 1) different data encoding.
 * 2) XTheadVector has more insns to handle zero/sign-extended.
 */
static bool ldff_op_th(DisasContext *s, arg_r2nfvm *a, uint8_t seq)
{
    uint32_t data = 0;
    gen_helper_ldst_us_th *fn;
    static gen_helper_ldst_us_th * const fns[7][4] = {
        { gen_helper_th_vlbff_v_b,  gen_helper_th_vlbff_v_h,
          gen_helper_th_vlbff_v_w,  gen_helper_th_vlbff_v_d },
        { NULL,                     gen_helper_th_vlhff_v_h,
          gen_helper_th_vlhff_v_w,  gen_helper_th_vlhff_v_d },
        { NULL,                     NULL,
          gen_helper_th_vlwff_v_w,  gen_helper_th_vlwff_v_d },
        { gen_helper_th_vleff_v_b,  gen_helper_th_vleff_v_h,
          gen_helper_th_vleff_v_w,  gen_helper_th_vleff_v_d },
        { gen_helper_th_vlbuff_v_b, gen_helper_th_vlbuff_v_h,
          gen_helper_th_vlbuff_v_w, gen_helper_th_vlbuff_v_d },
        { NULL,                     gen_helper_th_vlhuff_v_h,
          gen_helper_th_vlhuff_v_w, gen_helper_th_vlhuff_v_d },
        { NULL,                     NULL,
          gen_helper_th_vlwuff_v_w, gen_helper_th_vlwuff_v_d }
    };

    fn =  fns[seq][s->sew];
    if (fn == NULL) {
        return false;
    }

    data = FIELD_DP32(data, VDATA_TH, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_TH, VM, a->vm);
    data = FIELD_DP32(data, VDATA_TH, LMUL, s->lmul);
    data = FIELD_DP32(data, VDATA_TH, NF, a->nf);
    return ldff_trans_th(a->rd, a->rs1, data, fn, s);
}

GEN_TH_TRANS(th_vlbff_v, 0, r2nfvm, ldff_op_th, ld_us_check_th)
GEN_TH_TRANS(th_vlhff_v, 1, r2nfvm, ldff_op_th, ld_us_check_th)
GEN_TH_TRANS(th_vlwff_v, 2, r2nfvm, ldff_op_th, ld_us_check_th)
GEN_TH_TRANS(th_vleff_v, 3, r2nfvm, ldff_op_th, ld_us_check_th)
GEN_TH_TRANS(th_vlbuff_v, 4, r2nfvm, ldff_op_th, ld_us_check_th)
GEN_TH_TRANS(th_vlhuff_v, 5, r2nfvm, ldff_op_th, ld_us_check_th)
GEN_TH_TRANS(th_vlwuff_v, 6, r2nfvm, ldff_op_th, ld_us_check_th)


/*
 *** vector atomic operation
 */
typedef void gen_helper_amo(TCGv_ptr, TCGv_ptr, TCGv, TCGv_ptr,
                            TCGv_env, TCGv_i32);
static bool amo_trans_th(uint32_t vd, uint32_t rs1, uint32_t vs2,
                         uint32_t data, gen_helper_amo *fn, DisasContext *s)
{
    TCGv_ptr dest, mask, index;
    TCGv base;
    TCGv_i32 desc;

    TCGLabel *over = gen_new_label();
    tcg_gen_brcond_tl(TCG_COND_GEU, cpu_vstart, cpu_vl, over);

    dest = tcg_temp_new_ptr();
    mask = tcg_temp_new_ptr();
    index = tcg_temp_new_ptr();
    base = get_gpr(s, rs1, EXT_NONE);
    desc = tcg_constant_i32(simd_desc(s->cfg_ptr->vlen / 8,
                                      s->cfg_ptr->vlen / 8, data));

    tcg_gen_addi_ptr(dest, tcg_env, th_vreg_ofs(s, vd));
    tcg_gen_addi_ptr(index, tcg_env, th_vreg_ofs(s, vs2));
    tcg_gen_addi_ptr(mask, tcg_env, th_vreg_ofs(s, 0));

    fn(dest, mask, base, index, tcg_env, desc);

    gen_set_label(over);
    return true;
}

static bool amo_op_th(DisasContext *s, arg_rwdvm *a, uint8_t seq)
{
    uint32_t data = 0;
    gen_helper_amo *fn;
    static gen_helper_amo *const fnsw[9] = {
        /* no atomic operation */
        gen_helper_th_vamoswapw_v_w,
        gen_helper_th_vamoaddw_v_w,
        gen_helper_th_vamoxorw_v_w,
        gen_helper_th_vamoandw_v_w,
        gen_helper_th_vamoorw_v_w,
        gen_helper_th_vamominw_v_w,
        gen_helper_th_vamomaxw_v_w,
        gen_helper_th_vamominuw_v_w,
        gen_helper_th_vamomaxuw_v_w
    };
    static gen_helper_amo *const fnsd[18] = {
        gen_helper_th_vamoswapw_v_d,
        gen_helper_th_vamoaddw_v_d,
        gen_helper_th_vamoxorw_v_d,
        gen_helper_th_vamoandw_v_d,
        gen_helper_th_vamoorw_v_d,
        gen_helper_th_vamominw_v_d,
        gen_helper_th_vamomaxw_v_d,
        gen_helper_th_vamominuw_v_d,
        gen_helper_th_vamomaxuw_v_d,
        gen_helper_th_vamoswapd_v_d,
        gen_helper_th_vamoaddd_v_d,
        gen_helper_th_vamoxord_v_d,
        gen_helper_th_vamoandd_v_d,
        gen_helper_th_vamoord_v_d,
        gen_helper_th_vamomind_v_d,
        gen_helper_th_vamomaxd_v_d,
        gen_helper_th_vamominud_v_d,
        gen_helper_th_vamomaxud_v_d
    };

    if (tb_cflags(s->base.tb) & CF_PARALLEL) {
        gen_helper_exit_atomic(tcg_env);
        s->base.is_jmp = DISAS_NORETURN;
        return true;
    }
    switch (s->sew) {
    case 0 ... 2:
        assert(seq < ARRAY_SIZE(fnsw));
        fn = fnsw[seq];
        break;
    case 3:
        /* XLEN check done in amo_check(). */
        assert(seq < ARRAY_SIZE(fnsd));
        fn = fnsd[seq];
        break;
    default:
        g_assert_not_reached();
    }

    data = FIELD_DP32(data, VDATA_TH, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_TH, VM, a->vm);
    data = FIELD_DP32(data, VDATA_TH, LMUL, s->lmul);
    data = FIELD_DP32(data, VDATA_TH, WD, a->wd);
    return amo_trans_th(a->rd, a->rs1, a->rs2, data, fn, s);
}
/*
 * There are two rules check here.
 *
 * 1. SEW must be at least as wide as the AMO memory element size.
 *
 * 2. If SEW is greater than XLEN, an illegal instruction exception is raised.
 */
static bool amo_check_th(DisasContext *s, arg_rwdvm* a)
{
    return (require_thv(s) &&
            !s->vill && has_ext(s, RVA) &&
            (!a->wd || th_check_overlap_mask(s, a->rd, a->vm, false)) &&
            th_check_reg(s, a->rd, false) &&
            th_check_reg(s, a->rs2, false) &&
            ((1 << s->sew) <= (get_xlen(s) / 8)) &&
            ((1 << s->sew) >= 4));
}

static bool amo_check64_th(DisasContext *s, arg_rwdvm* a)
{
    REQUIRE_64BIT(s);
    return amo_check_th(s, a);
}

GEN_TH_TRANS(th_vamoswapw_v, 0, rwdvm, amo_op_th, amo_check_th)
GEN_TH_TRANS(th_vamoaddw_v, 1, rwdvm, amo_op_th, amo_check_th)
GEN_TH_TRANS(th_vamoxorw_v, 2, rwdvm, amo_op_th, amo_check_th)
GEN_TH_TRANS(th_vamoandw_v, 3, rwdvm, amo_op_th, amo_check_th)
GEN_TH_TRANS(th_vamoorw_v, 4, rwdvm, amo_op_th, amo_check_th)
GEN_TH_TRANS(th_vamominw_v, 5, rwdvm, amo_op_th, amo_check_th)
GEN_TH_TRANS(th_vamomaxw_v, 6, rwdvm, amo_op_th, amo_check_th)
GEN_TH_TRANS(th_vamominuw_v, 7, rwdvm, amo_op_th, amo_check_th)
GEN_TH_TRANS(th_vamomaxuw_v, 8, rwdvm, amo_op_th, amo_check_th)
GEN_TH_TRANS(th_vamoswapd_v, 9, rwdvm, amo_op_th, amo_check64_th)
GEN_TH_TRANS(th_vamoaddd_v, 10, rwdvm, amo_op_th, amo_check64_th)
GEN_TH_TRANS(th_vamoxord_v, 11, rwdvm, amo_op_th, amo_check64_th)
GEN_TH_TRANS(th_vamoandd_v, 12, rwdvm, amo_op_th, amo_check64_th)
GEN_TH_TRANS(th_vamoord_v, 13, rwdvm, amo_op_th, amo_check64_th)
GEN_TH_TRANS(th_vamomind_v, 14, rwdvm, amo_op_th, amo_check64_th)
GEN_TH_TRANS(th_vamomaxd_v, 15, rwdvm, amo_op_th, amo_check64_th)
GEN_TH_TRANS(th_vamominud_v, 16, rwdvm, amo_op_th, amo_check64_th)
GEN_TH_TRANS(th_vamomaxud_v, 17, rwdvm, amo_op_th, amo_check64_th)

/*
 *** Vector Integer Arithmetic Instructions
 */
static inline uint32_t MAXSZ_TH(DisasContext *s)
{
    return MAXSZ(s);
}

/*
 * check function
 * 1) check overlap mask, XTheadVector can overlap mask reg v0 when
 *    lmul == 1, while RVV1.0 can not.
 * 2) check reg, XTheadVector Vector register numbers are multiples
 *    of integral LMUL, while RVV1.0 has fractional LMUL, which allows
 *    any vector register.
 */
static bool opivv_check_th(DisasContext *s, arg_rmrr *a)
{
    return (require_thv(s) &&
            th_check_isa_ill(s) &&
            th_check_overlap_mask(s, a->rd, a->vm, false) &&
            th_check_reg(s, a->rd, false) &&
            th_check_reg(s, a->rs2, false) &&
            th_check_reg(s, a->rs1, false));
}

#define GVecGen3Fn_Th GVecGen3Fn
/*
 * This function is almost the copy of do_opivv_gvec, except:
 * 1) XTheadVector using different data encoding, add MLEN,
 *    delete VTA and VMA.
 * 2) XTheadVector simplifies the judgment logic of whether
 *    to accelerate or not for its lack of fractional LMUL and
 *    VTA.
 */
static inline bool
do_opivv_gvec_th(DisasContext *s, arg_rmrr *a, GVecGen3Fn_Th *gvec_fn,
                 gen_helper_gvec_4_ptr *fn)
{
    TCGLabel *over = gen_new_label();

    tcg_gen_brcond_tl(TCG_COND_GEU, cpu_vstart, cpu_vl, over);

    if (a->vm && s->vl_eq_vlmax) {
        gvec_fn(s->sew, th_vreg_ofs(s, a->rd),
                th_vreg_ofs(s, a->rs2), th_vreg_ofs(s, a->rs1),
                MAXSZ_TH(s), MAXSZ_TH(s));
    } else {
        uint32_t data = 0;
        /* Need extra mlen to find the mask bit */
        data = FIELD_DP32(data, VDATA_TH, MLEN, s->mlen);
        data = FIELD_DP32(data, VDATA_TH, VM, a->vm);
        data = FIELD_DP32(data, VDATA_TH, LMUL, s->lmul);
        tcg_gen_gvec_4_ptr(th_vreg_ofs(s, a->rd), th_vreg_ofs(s, 0),
                           th_vreg_ofs(s, a->rs1), th_vreg_ofs(s, a->rs2),
                           tcg_env, s->cfg_ptr->vlen / 8,
                           s->cfg_ptr->vlen / 8, data, fn);
    }
    mark_vs_dirty(s);
    gen_set_label(over);
    return true;
}

/* OPIVV with GVEC IR */
/*
 * GEN_OPIVV_GVEC_TRANS_TH is similar to GEN_OPIVV_GVEC_TRANS
 * just change the check and do_ functions.
 */
#define GEN_OPIVV_GVEC_TRANS_TH(NAME, SUF)                         \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)             \
{                                                                  \
    static gen_helper_gvec_4_ptr * const fns[4] = {                \
        gen_helper_##NAME##_b, gen_helper_##NAME##_h,              \
        gen_helper_##NAME##_w, gen_helper_##NAME##_d,              \
    };                                                             \
    if (!opivv_check_th(s, a)) {                                   \
        return false;                                              \
    }                                                              \
    return do_opivv_gvec_th(s, a, tcg_gen_gvec_##SUF, fns[s->sew]);\
}

GEN_OPIVV_GVEC_TRANS_TH(th_vadd_vv, add)
GEN_OPIVV_GVEC_TRANS_TH(th_vsub_vv, sub)

/*
 * This function is almost the copy of opivx_trans, except:
 * 1) XTheadVector using different data encoding, add MLEN,
 *    delete VTA and VMA.
 */
#define gen_helper_opivx_th gen_helper_opivx
static bool opivx_trans_th(uint32_t vd, uint32_t rs1, uint32_t vs2, uint32_t vm,
                           gen_helper_opivx_th *fn, DisasContext *s)
{
    TCGv_ptr dest, src2, mask;
    TCGv src1;
    TCGv_i32 desc;
    uint32_t data = 0;

    TCGLabel *over = gen_new_label();
    tcg_gen_brcond_tl(TCG_COND_GEU, cpu_vstart, cpu_vl, over);

    dest = tcg_temp_new_ptr();
    mask = tcg_temp_new_ptr();
    src2 = tcg_temp_new_ptr();
    src1 = get_gpr(s, rs1, EXT_SIGN);

    data = FIELD_DP32(data, VDATA_TH, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_TH, VM, vm);
    data = FIELD_DP32(data, VDATA_TH, LMUL, s->lmul);
    desc = tcg_constant_i32(simd_desc(s->cfg_ptr->vlen / 8,
                                      s->cfg_ptr->vlen / 8, data));

    tcg_gen_addi_ptr(dest, tcg_env, th_vreg_ofs(s, vd));
    tcg_gen_addi_ptr(src2, tcg_env, th_vreg_ofs(s, vs2));
    tcg_gen_addi_ptr(mask, tcg_env, th_vreg_ofs(s, 0));

    fn(dest, mask, src1, src2, tcg_env, desc);

    mark_vs_dirty(s);
    gen_set_label(over);
    return true;
}

static bool opivx_check_th(DisasContext *s, arg_rmrr *a)
{
    return (require_thv(s) &&
            th_check_isa_ill(s) &&
            th_check_overlap_mask(s, a->rd, a->vm, false) &&
            th_check_reg(s, a->rd, false) &&
            th_check_reg(s, a->rs2, false));
}

#define GVecGen2sFn_Th GVecGen2sFn
/*
 * This function is almost the copy of do_opivx_gvec, except:
 * 1) XTheadVector simplifies the judgment logic of whether
 *    to accelerate or not for its lack of fractional LMUL and
 *    VTA.
 */
static inline bool
do_opivx_gvec_th(DisasContext *s, arg_rmrr *a, GVecGen2sFn_Th *gvec_fn,
                 gen_helper_opivx_th *fn)
{
    if (a->vm && s->vl_eq_vlmax) {
        TCGv_i64 src1 = tcg_temp_new_i64();

        tcg_gen_ext_tl_i64(src1, get_gpr(s, a->rs1, EXT_SIGN));
        gvec_fn(s->sew, th_vreg_ofs(s, a->rd), th_vreg_ofs(s, a->rs2),
                src1, MAXSZ_TH(s), MAXSZ_TH(s));
        mark_vs_dirty(s);
        return true;
    }
    return opivx_trans_th(a->rd, a->rs1, a->rs2, a->vm, fn, s);
}

/* OPIVX with GVEC IR */
#define GEN_OPIVX_GVEC_TRANS_TH(NAME, SUF)                         \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)             \
{                                                                  \
    static gen_helper_opivx_th * const fns[4] = {                  \
        gen_helper_##NAME##_b, gen_helper_##NAME##_h,              \
        gen_helper_##NAME##_w, gen_helper_##NAME##_d,              \
    };                                                             \
    if (!opivx_check_th(s, a)) {                                   \
        return false;                                              \
    }                                                              \
    return do_opivx_gvec_th(s, a, tcg_gen_gvec_##SUF, fns[s->sew]);\
}

GEN_OPIVX_GVEC_TRANS_TH(th_vadd_vx, adds)
GEN_OPIVX_GVEC_TRANS_TH(th_vsub_vx, subs)
GEN_OPIVX_GVEC_TRANS_TH(th_vrsub_vx, rsubs)

#define imm_mode_t_th imm_mode_t

static int64_t extract_imm_th(DisasContext *s, uint32_t imm,
                              imm_mode_t_th imm_mode)
{
    return extract_imm(s, imm, imm_mode);
}

/*
 * This function is almost the copy of opivi_trans, except:
 * 1) XTheadVector using different data encoding, add MLEN,
 *    delete VTA and VMA.
 */
static bool opivi_trans_th(uint32_t vd, uint32_t imm, uint32_t vs2, uint32_t vm,
                           gen_helper_opivx_th *fn, DisasContext *s,
                           imm_mode_t_th imm_mode)
{
    TCGv_ptr dest, src2, mask;
    TCGv src1;
    TCGv_i32 desc;
    uint32_t data = 0;

    TCGLabel *over = gen_new_label();
    tcg_gen_brcond_tl(TCG_COND_GEU, cpu_vstart, cpu_vl, over);

    dest = tcg_temp_new_ptr();
    mask = tcg_temp_new_ptr();
    src2 = tcg_temp_new_ptr();
    src1 = tcg_constant_tl(extract_imm_th(s, imm, imm_mode));

    data = FIELD_DP32(data, VDATA_TH, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_TH, VM, vm);
    data = FIELD_DP32(data, VDATA_TH, LMUL, s->lmul);
    desc = tcg_constant_i32(simd_desc(s->cfg_ptr->vlen / 8,
                                      s->cfg_ptr->vlen / 8, data));

    tcg_gen_addi_ptr(dest, tcg_env, th_vreg_ofs(s, vd));
    tcg_gen_addi_ptr(src2, tcg_env, th_vreg_ofs(s, vs2));
    tcg_gen_addi_ptr(mask, tcg_env, th_vreg_ofs(s, 0));

    fn(dest, mask, src1, src2, tcg_env, desc);

    mark_vs_dirty(s);
    gen_set_label(over);
    return true;
}
#define GVecGen2iFn_Th GVecGen2iFn
/*
 * This function is almost the copy of do_opivi_gvec, except:
 * 1) XTheadVector simplifies the judgment logic of whether
 *    to accelerate or not for its lack of fractional LMUL and
 *    VTA.
 */
static inline bool
do_opivi_gvec_th(DisasContext *s, arg_rmrr *a, GVecGen2iFn_Th *gvec_fn,
                 gen_helper_opivx_th *fn, imm_mode_t_th imm_mode)
{
    if (a->vm && s->vl_eq_vlmax) {
        gvec_fn(s->sew, th_vreg_ofs(s, a->rd), th_vreg_ofs(s, a->rs2),
                extract_imm_th(s, a->rs1, imm_mode), MAXSZ_TH(s), MAXSZ_TH(s));
        mark_vs_dirty(s);
        return true;
    }
    return opivi_trans_th(a->rd, a->rs1, a->rs2, a->vm, fn, s, imm_mode);
}

/* OPIVI with GVEC IR */
#define GEN_OPIVI_GVEC_TRANS_TH(NAME, IMM_MODE, OPIVX, SUF)        \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)             \
{                                                                  \
    static gen_helper_opivx_th * const fns[4] = {                  \
        gen_helper_##OPIVX##_b, gen_helper_##OPIVX##_h,            \
        gen_helper_##OPIVX##_w, gen_helper_##OPIVX##_d,            \
    };                                                             \
    if (!opivx_check_th(s, a)) {                                   \
        return false;                                              \
    }                                                              \
    return do_opivi_gvec_th(s, a, tcg_gen_gvec_##SUF,              \
                            fns[s->sew], IMM_MODE);                \
}

GEN_OPIVI_GVEC_TRANS_TH(th_vadd_vi, IMM_SX, th_vadd_vx, addi)
GEN_OPIVI_GVEC_TRANS_TH(th_vrsub_vi, IMM_SX, th_vrsub_vx, rsubi)

/* Vector Widening Integer Add/Subtract */

/* OPIVV with WIDEN */
static bool opivv_widen_check_th(DisasContext *s, arg_rmrr *a)
{
    return (require_thv(s) &&
            th_check_isa_ill(s) &&
            th_check_overlap_mask(s, a->rd, a->vm, true) &&
            th_check_reg(s, a->rd, true) &&
            th_check_reg(s, a->rs2, false) &&
            th_check_reg(s, a->rs1, false) &&
            th_check_overlap_group(a->rd, 2 << s->lmul, a->rs2,
                                   1 << s->lmul) &&
            th_check_overlap_group(a->rd, 2 << s->lmul, a->rs1,
                                   1 << s->lmul) &&
            (s->lmul < 0x3) && (s->sew < 0x3));
}

/*
 * This function is almost the copy of do_opivv_widen, except:
 * 1) XTheadVector using different data encoding, add MLEN,
 *    delete VTA and VMA.
 */
static bool do_opivv_widen_th(DisasContext *s, arg_rmrr *a,
                              gen_helper_gvec_4_ptr *fn,
                              bool (*checkfn)(DisasContext *, arg_rmrr *))
{
    if (checkfn(s, a)) {
        uint32_t data = 0;
        TCGLabel *over = gen_new_label();
        tcg_gen_brcond_tl(TCG_COND_GEU, cpu_vstart, cpu_vl, over);

        data = FIELD_DP32(data, VDATA_TH, MLEN, s->mlen);
        data = FIELD_DP32(data, VDATA_TH, VM, a->vm);
        data = FIELD_DP32(data, VDATA_TH, LMUL, s->lmul);
        tcg_gen_gvec_4_ptr(th_vreg_ofs(s, a->rd), th_vreg_ofs(s, 0),
                           th_vreg_ofs(s, a->rs1),
                           th_vreg_ofs(s, a->rs2),
                           tcg_env, s->cfg_ptr->vlen / 8,
                           s->cfg_ptr->vlen / 8,
                           data, fn);
        mark_vs_dirty(s);
        gen_set_label(over);
        return true;
    }
    return false;
}

#define GEN_OPIVV_WIDEN_TRANS_TH(NAME, CHECK)                \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)       \
{                                                            \
    static gen_helper_gvec_4_ptr * const fns[3] = {          \
        gen_helper_##NAME##_b,                               \
        gen_helper_##NAME##_h,                               \
        gen_helper_##NAME##_w                                \
    };                                                       \
    return do_opivv_widen_th(s, a, fns[s->sew], CHECK);      \
}

GEN_OPIVV_WIDEN_TRANS_TH(th_vwaddu_vv, opivv_widen_check_th)
GEN_OPIVV_WIDEN_TRANS_TH(th_vwadd_vv, opivv_widen_check_th)
GEN_OPIVV_WIDEN_TRANS_TH(th_vwsubu_vv, opivv_widen_check_th)
GEN_OPIVV_WIDEN_TRANS_TH(th_vwsub_vv, opivv_widen_check_th)

/* OPIVX with WIDEN */
static bool opivx_widen_check_th(DisasContext *s, arg_rmrr *a)
{
    return (require_thv(s) &&
            th_check_isa_ill(s) &&
            th_check_overlap_mask(s, a->rd, a->vm, true) &&
            th_check_reg(s, a->rd, true) &&
            th_check_reg(s, a->rs2, false) &&
            th_check_overlap_group(a->rd, 2 << s->lmul, a->rs2,
                                     1 << s->lmul) &&
            (s->lmul < 0x3) && (s->sew < 0x3));
}

#define GEN_OPIVX_WIDEN_TRANS_TH(NAME, CHECK)                \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)       \
{                                                            \
    if (CHECK(s, a)) {                                       \
        static gen_helper_opivx_th * const fns[3] = {        \
            gen_helper_##NAME##_b,                           \
            gen_helper_##NAME##_h,                           \
            gen_helper_##NAME##_w                            \
        };                                                   \
        return opivx_trans_th(a->rd, a->rs1, a->rs2, a->vm,  \
                              fns[s->sew], s);               \
    }                                                        \
    return false;                                            \
}

GEN_OPIVX_WIDEN_TRANS_TH(th_vwaddu_vx, opivx_widen_check_th)
GEN_OPIVX_WIDEN_TRANS_TH(th_vwadd_vx, opivx_widen_check_th)
GEN_OPIVX_WIDEN_TRANS_TH(th_vwsubu_vx, opivx_widen_check_th)
GEN_OPIVX_WIDEN_TRANS_TH(th_vwsub_vx, opivx_widen_check_th)

/* WIDEN OPIVV with WIDEN */
static bool opiwv_widen_check_th(DisasContext *s, arg_rmrr *a)
{
    return (require_thv(s) &&
            th_check_isa_ill(s) &&
            th_check_overlap_mask(s, a->rd, a->vm, true) &&
            th_check_reg(s, a->rd, true) &&
            th_check_reg(s, a->rs2, true) &&
            th_check_reg(s, a->rs1, false) &&
            th_check_overlap_group(a->rd, 2 << s->lmul, a->rs1,
                                     1 << s->lmul) &&
            (s->lmul < 0x3) && (s->sew < 0x3));
}

static bool do_opiwv_widen_th(DisasContext *s, arg_rmrr *a,
                              gen_helper_gvec_4_ptr *fn)
{
    if (opiwv_widen_check_th(s, a)) {
        uint32_t data = 0;
        TCGLabel *over = gen_new_label();
        tcg_gen_brcond_tl(TCG_COND_GEU, cpu_vstart, cpu_vl, over);

        data = FIELD_DP32(data, VDATA_TH, MLEN, s->mlen);
        data = FIELD_DP32(data, VDATA_TH, VM, a->vm);
        data = FIELD_DP32(data, VDATA_TH, LMUL, s->lmul);
        tcg_gen_gvec_4_ptr(th_vreg_ofs(s, a->rd), th_vreg_ofs(s, 0),
                           th_vreg_ofs(s, a->rs1),
                           th_vreg_ofs(s, a->rs2),
                           tcg_env, s->cfg_ptr->vlen / 8,
                           s->cfg_ptr->vlen / 8, data, fn);
        mark_vs_dirty(s);
        gen_set_label(over);
        return true;
    }
    return false;
}

#define GEN_OPIWV_WIDEN_TRANS_TH(NAME)                       \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)       \
{                                                            \
    static gen_helper_gvec_4_ptr * const fns[3] = {          \
        gen_helper_##NAME##_b,                               \
        gen_helper_##NAME##_h,                               \
        gen_helper_##NAME##_w                                \
    };                                                       \
    return do_opiwv_widen_th(s, a, fns[s->sew]);             \
}

GEN_OPIWV_WIDEN_TRANS_TH(th_vwaddu_wv)
GEN_OPIWV_WIDEN_TRANS_TH(th_vwadd_wv)
GEN_OPIWV_WIDEN_TRANS_TH(th_vwsubu_wv)
GEN_OPIWV_WIDEN_TRANS_TH(th_vwsub_wv)

/* WIDEN OPIVX with WIDEN */
static bool opiwx_widen_check_th(DisasContext *s, arg_rmrr *a)
{
    return (require_thv(s) &&
            th_check_isa_ill(s) &&
            th_check_overlap_mask(s, a->rd, a->vm, true) &&
            th_check_reg(s, a->rd, true) &&
            th_check_reg(s, a->rs2, true) &&
            (s->lmul < 0x3) && (s->sew < 0x3));
}


static bool do_opiwx_widen_th(DisasContext *s, arg_rmrr *a,
                              gen_helper_opivx_th *fn)
{
    if (opiwx_widen_check_th(s, a)) {
        return opivx_trans_th(a->rd, a->rs1, a->rs2, a->vm, fn, s);
    }
    return false;
}

#define GEN_OPIWX_WIDEN_TRANS_TH(NAME)                       \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)       \
{                                                            \
    static gen_helper_opivx_th * const fns[3] = {            \
        gen_helper_##NAME##_b,                               \
        gen_helper_##NAME##_h,                               \
        gen_helper_##NAME##_w                                \
    };                                                       \
    return do_opiwx_widen_th(s, a, fns[s->sew]);             \
}

GEN_OPIWX_WIDEN_TRANS_TH(th_vwaddu_wx)
GEN_OPIWX_WIDEN_TRANS_TH(th_vwadd_wx)
GEN_OPIWX_WIDEN_TRANS_TH(th_vwsubu_wx)
GEN_OPIWX_WIDEN_TRANS_TH(th_vwsub_wx)

/* Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions */
/*
 * This function is almost the copy of opivv_trans, except:
 * 1) XTheadVector using different data encoding, add MLEN,
 *    delete VTA and VMA.
 */
static bool opivv_trans_th(uint32_t vd, uint32_t vs1, uint32_t vs2, uint32_t vm,
                           gen_helper_gvec_4_ptr *fn, DisasContext *s)
{
    uint32_t data = 0;
    TCGLabel *over = gen_new_label();
    tcg_gen_brcond_tl(TCG_COND_GEU, cpu_vstart, cpu_vl, over);

    data = FIELD_DP32(data, VDATA_TH, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_TH, VM, vm);
    data = FIELD_DP32(data, VDATA_TH, LMUL, s->lmul);
    tcg_gen_gvec_4_ptr(th_vreg_ofs(s, vd), th_vreg_ofs(s, 0),
                       th_vreg_ofs(s, vs1), th_vreg_ofs(s, vs2),
                       tcg_env, s->cfg_ptr->vlen / 8,
                       s->cfg_ptr->vlen / 8, data, fn);
    mark_vs_dirty(s);
    gen_set_label(over);
    return true;
}

/* OPIVV without GVEC IR */
#define GEN_OPIVV_TRANS_TH(NAME, CHECK)                            \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)             \
{                                                                  \
    if (CHECK(s, a)) {                                             \
        static gen_helper_gvec_4_ptr * const fns[4] = {            \
            gen_helper_##NAME##_b, gen_helper_##NAME##_h,          \
            gen_helper_##NAME##_w, gen_helper_##NAME##_d,          \
        };                                                         \
        return opivv_trans_th(a->rd, a->rs1, a->rs2, a->vm,        \
                              fns[s->sew], s);                     \
    }                                                              \
    return false;                                                  \
}

/*
 * For vadc and vsbc, an illegal instruction exception is raised if the
 * destination vector register is v0 and LMUL > 1.
 */
static bool opivv_vadc_check_th(DisasContext *s, arg_rmrr *a)
{
    return (require_thv(s) &&
            th_check_isa_ill(s) &&
            th_check_reg(s, a->rd, false) &&
            th_check_reg(s, a->rs2, false) &&
            th_check_reg(s, a->rs1, false) &&
            ((a->rd != 0) || (s->lmul == 0)));
}

GEN_OPIVV_TRANS_TH(th_vadc_vvm, opivv_vadc_check_th)
GEN_OPIVV_TRANS_TH(th_vsbc_vvm, opivv_vadc_check_th)

/*
 * For vmadc and vmsbc, an illegal instruction exception is raised if the
 * destination vector register overlaps a source vector register group.
 */
static bool opivv_vmadc_check_th(DisasContext *s, arg_rmrr *a)
{
    return (require_thv(s) &&
            th_check_isa_ill(s) &&
            th_check_reg(s, a->rs2, false) &&
            th_check_reg(s, a->rs1, false) &&
            th_check_overlap_group(a->rd, 1, a->rs1, 1 << s->lmul) &&
            th_check_overlap_group(a->rd, 1, a->rs2, 1 << s->lmul));
}

GEN_OPIVV_TRANS_TH(th_vmadc_vvm, opivv_vmadc_check_th)
GEN_OPIVV_TRANS_TH(th_vmsbc_vvm, opivv_vmadc_check_th)

static bool opivx_vadc_check_th(DisasContext *s, arg_rmrr *a)
{
    return (require_thv(s) &&
            th_check_isa_ill(s) &&
            th_check_reg(s, a->rd, false) &&
            th_check_reg(s, a->rs2, false) &&
            ((a->rd != 0) || (s->lmul == 0)));
}

/* OPIVX without GVEC IR */
#define GEN_OPIVX_TRANS_TH(NAME, CHECK)                                  \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)                   \
{                                                                        \
    if (CHECK(s, a)) {                                                   \
        static gen_helper_opivx * const fns[4] = {                       \
            gen_helper_##NAME##_b, gen_helper_##NAME##_h,                \
            gen_helper_##NAME##_w, gen_helper_##NAME##_d,                \
        };                                                               \
                                                                         \
        return opivx_trans_th(a->rd, a->rs1, a->rs2, a->vm,              \
                              fns[s->sew], s);                           \
    }                                                                    \
    return false;                                                        \
}

GEN_OPIVX_TRANS_TH(th_vadc_vxm, opivx_vadc_check_th)
GEN_OPIVX_TRANS_TH(th_vsbc_vxm, opivx_vadc_check_th)

static bool opivx_vmadc_check_th(DisasContext *s, arg_rmrr *a)
{
    return (require_thv(s) &&
            th_check_isa_ill(s) &&
            th_check_reg(s, a->rs2, false) &&
            th_check_overlap_group(a->rd, 1, a->rs2, 1 << s->lmul));
}

GEN_OPIVX_TRANS_TH(th_vmadc_vxm, opivx_vmadc_check_th)
GEN_OPIVX_TRANS_TH(th_vmsbc_vxm, opivx_vmadc_check_th)

/* OPIVI without GVEC IR */
#define GEN_OPIVI_TRANS_TH(NAME, ZX, OPIVX, CHECK)                       \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)                   \
{                                                                        \
    if (CHECK(s, a)) {                                                   \
        static gen_helper_opivx * const fns[4] = {                       \
            gen_helper_##OPIVX##_b, gen_helper_##OPIVX##_h,              \
            gen_helper_##OPIVX##_w, gen_helper_##OPIVX##_d,              \
        };                                                               \
        return opivi_trans_th(a->rd, a->rs1, a->rs2, a->vm,              \
                              fns[s->sew], s, ZX);                       \
    }                                                                    \
    return false;                                                        \
}

GEN_OPIVI_TRANS_TH(th_vadc_vim, IMM_SX, th_vadc_vxm, opivx_vadc_check_th)
GEN_OPIVI_TRANS_TH(th_vmadc_vim, IMM_SX, th_vmadc_vxm, opivx_vmadc_check_th)

/* Vector Bitwise Logical Instructions */
GEN_OPIVV_GVEC_TRANS_TH(th_vand_vv, and)
GEN_OPIVV_GVEC_TRANS_TH(th_vor_vv,  or)
GEN_OPIVV_GVEC_TRANS_TH(th_vxor_vv, xor)
GEN_OPIVX_GVEC_TRANS_TH(th_vand_vx, ands)
GEN_OPIVX_GVEC_TRANS_TH(th_vor_vx,  ors)
GEN_OPIVX_GVEC_TRANS_TH(th_vxor_vx, xors)
GEN_OPIVI_GVEC_TRANS_TH(th_vand_vi, IMM_SX, th_vand_vx, andi)
GEN_OPIVI_GVEC_TRANS_TH(th_vor_vi, IMM_SX, th_vor_vx,  ori)
GEN_OPIVI_GVEC_TRANS_TH(th_vxor_vi, IMM_SX, th_vxor_vx, xori)

/* Vector Single-Width Bit Shift Instructions */
GEN_OPIVV_GVEC_TRANS_TH(th_vsll_vv,  shlv)
GEN_OPIVV_GVEC_TRANS_TH(th_vsrl_vv,  shrv)
GEN_OPIVV_GVEC_TRANS_TH(th_vsra_vv,  sarv)

#define GVecGen2sFn32_Th GVecGen2sFn32
/*
 * This function is almost the copy of do_opivx_gvec_shift, except:
 * 1) XTheadVector simplifies the judgment logic of whether
 *    to accelerate or not for its lack of fractional LMUL and
 *    VTA.
 */
static inline bool
do_opivx_gvec_shift_th(DisasContext *s, arg_rmrr *a, GVecGen2sFn32_Th *gvec_fn,
                       gen_helper_opivx_th *fn)
{
    if (a->vm && s->vl_eq_vlmax) {
        TCGv_i32 src1 = tcg_temp_new_i32();

        tcg_gen_trunc_tl_i32(src1, get_gpr(s, a->rs1, EXT_NONE));
        tcg_gen_extract_i32(src1, src1, 0, s->sew + 3);
        gvec_fn(s->sew, th_vreg_ofs(s, a->rd), th_vreg_ofs(s, a->rs2),
                src1, MAXSZ_TH(s), MAXSZ_TH(s));

        mark_vs_dirty(s);
        return true;
    }
    return opivx_trans_th(a->rd, a->rs1, a->rs2, a->vm, fn, s);
}

#define GEN_OPIVX_GVEC_SHIFT_TRANS_TH(NAME, SUF)                          \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)                    \
{                                                                         \
    static gen_helper_opivx * const fns[4] = {                            \
        gen_helper_##NAME##_b, gen_helper_##NAME##_h,                     \
        gen_helper_##NAME##_w, gen_helper_##NAME##_d,                     \
    };                                                                    \
    if (!opivx_check_th(s, a)) {                                          \
        return false;                                                     \
    }                                                                     \
    return do_opivx_gvec_shift_th(s, a, tcg_gen_gvec_##SUF, fns[s->sew]); \
}

GEN_OPIVX_GVEC_SHIFT_TRANS_TH(th_vsll_vx,  shls)
GEN_OPIVX_GVEC_SHIFT_TRANS_TH(th_vsrl_vx,  shrs)
GEN_OPIVX_GVEC_SHIFT_TRANS_TH(th_vsra_vx,  sars)

GEN_OPIVI_GVEC_TRANS_TH(th_vsll_vi, IMM_TRUNC_SEW, th_vsll_vx,  shli)
GEN_OPIVI_GVEC_TRANS_TH(th_vsrl_vi, IMM_TRUNC_SEW, th_vsrl_vx,  shri)
GEN_OPIVI_GVEC_TRANS_TH(th_vsra_vi, IMM_TRUNC_SEW, th_vsra_vx,  sari)

/* Vector Narrowing Integer Right Shift Instructions */
static bool opivv_narrow_check_th(DisasContext *s, arg_rmrr *a)
{
    return (require_thv(s) &&
            th_check_isa_ill(s) &&
            th_check_overlap_mask(s, a->rd, a->vm, false) &&
            th_check_reg(s, a->rd, false) &&
            th_check_reg(s, a->rs2, true) &&
            th_check_reg(s, a->rs1, false) &&
            th_check_overlap_group(a->rd, 1 << s->lmul, a->rs2,
                2 << s->lmul) &&
            (s->lmul < 0x3) && (s->sew < 0x3));
}

/* OPIVV with NARROW */
/* different data encoding */
#define GEN_OPIVV_NARROW_TRANS_TH(NAME)                            \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)             \
{                                                                  \
    if (opivv_narrow_check_th(s, a)) {                             \
        uint32_t data = 0;                                         \
        static gen_helper_gvec_4_ptr * const fns[3] = {            \
            gen_helper_##NAME##_b,                                 \
            gen_helper_##NAME##_h,                                 \
            gen_helper_##NAME##_w,                                 \
        };                                                         \
        TCGLabel *over = gen_new_label();                          \
        tcg_gen_brcond_tl(TCG_COND_GEU, cpu_vstart, cpu_vl, over); \
                                                                   \
        data = FIELD_DP32(data, VDATA_TH, MLEN, s->mlen);          \
        data = FIELD_DP32(data, VDATA_TH, VM, a->vm);              \
        data = FIELD_DP32(data, VDATA_TH, LMUL, s->lmul);          \
        tcg_gen_gvec_4_ptr(th_vreg_ofs(s, a->rd),                  \
                           th_vreg_ofs(s, 0),                      \
                           th_vreg_ofs(s, a->rs1),                 \
                           th_vreg_ofs(s, a->rs2), tcg_env,        \
                           s->cfg_ptr->vlen / 8,                   \
                           s->cfg_ptr->vlen / 8, data,             \
                           fns[s->sew]);                           \
        mark_vs_dirty(s);                                          \
        gen_set_label(over);                                       \
        return true;                                               \
    }                                                              \
    return false;                                                  \
}
GEN_OPIVV_NARROW_TRANS_TH(th_vnsra_vv)
GEN_OPIVV_NARROW_TRANS_TH(th_vnsrl_vv)

static bool opivx_narrow_check_th(DisasContext *s, arg_rmrr *a)
{
    return (require_thv(s) &&
            th_check_isa_ill(s) &&
            th_check_overlap_mask(s, a->rd, a->vm, false) &&
            th_check_reg(s, a->rd, false) &&
            th_check_reg(s, a->rs2, true) &&
            th_check_overlap_group(a->rd, 1 << s->lmul, a->rs2,
                2 << s->lmul) &&
            (s->lmul < 0x3) && (s->sew < 0x3));
}

/* OPIVX with NARROW */
#define GEN_OPIVX_NARROW_TRANS_TH(NAME)                                     \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)                      \
{                                                                           \
    if (opivx_narrow_check_th(s, a)) {                                      \
        static gen_helper_opivx * const fns[3] = {                          \
            gen_helper_##NAME##_b,                                          \
            gen_helper_##NAME##_h,                                          \
            gen_helper_##NAME##_w,                                          \
        };                                                                  \
        return opivx_trans_th(a->rd, a->rs1, a->rs2, a->vm, fns[s->sew], s);\
    }                                                                       \
    return false;                                                           \
}

GEN_OPIVX_NARROW_TRANS_TH(th_vnsra_vx)
GEN_OPIVX_NARROW_TRANS_TH(th_vnsrl_vx)

/* OPIVI with NARROW */
#define GEN_OPIVI_NARROW_TRANS_TH(NAME, ZX, OPIVX)                       \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)                   \
{                                                                        \
    if (opivx_narrow_check_th(s, a)) {                                   \
        static gen_helper_opivx * const fns[3] = {                       \
            gen_helper_##OPIVX##_b,                                      \
            gen_helper_##OPIVX##_h,                                      \
            gen_helper_##OPIVX##_w,                                      \
        };                                                               \
        return opivi_trans_th(a->rd, a->rs1, a->rs2, a->vm,              \
                              fns[s->sew], s, ZX);                       \
    }                                                                    \
    return false;                                                        \
}

GEN_OPIVI_NARROW_TRANS_TH(th_vnsra_vi, IMM_ZX, th_vnsra_vx)
GEN_OPIVI_NARROW_TRANS_TH(th_vnsrl_vi, IMM_ZX, th_vnsrl_vx)

/* Vector Integer Comparison Instructions */
/*
 * For all comparison instructions, an illegal instruction exception is raised
 * if the destination vector register overlaps a source vector register group
 * and LMUL > 1.
 */
static bool opivv_cmp_check_th(DisasContext *s, arg_rmrr *a)
{
    return (require_thv(s) &&
            th_check_isa_ill(s) &&
            th_check_reg(s, a->rs2, false) &&
            th_check_reg(s, a->rs1, false) &&
            ((th_check_overlap_group(a->rd, 1, a->rs1, 1 << s->lmul) &&
              th_check_overlap_group(a->rd, 1, a->rs2, 1 << s->lmul)) ||
             (s->lmul == 0)));
}
GEN_OPIVV_TRANS_TH(th_vmseq_vv, opivv_cmp_check_th)
GEN_OPIVV_TRANS_TH(th_vmsne_vv, opivv_cmp_check_th)
GEN_OPIVV_TRANS_TH(th_vmsltu_vv, opivv_cmp_check_th)
GEN_OPIVV_TRANS_TH(th_vmslt_vv, opivv_cmp_check_th)
GEN_OPIVV_TRANS_TH(th_vmsleu_vv, opivv_cmp_check_th)
GEN_OPIVV_TRANS_TH(th_vmsle_vv, opivv_cmp_check_th)

static bool opivx_cmp_check_th(DisasContext *s, arg_rmrr *a)
{
    return (require_thv(s) &&
            th_check_isa_ill(s) &&
            th_check_reg(s, a->rs2, false) &&
            (th_check_overlap_group(a->rd, 1, a->rs2, 1 << s->lmul) ||
             (s->lmul == 0)));
}

GEN_OPIVX_TRANS_TH(th_vmseq_vx, opivx_cmp_check_th)
GEN_OPIVX_TRANS_TH(th_vmsne_vx, opivx_cmp_check_th)
GEN_OPIVX_TRANS_TH(th_vmsltu_vx, opivx_cmp_check_th)
GEN_OPIVX_TRANS_TH(th_vmslt_vx, opivx_cmp_check_th)
GEN_OPIVX_TRANS_TH(th_vmsleu_vx, opivx_cmp_check_th)
GEN_OPIVX_TRANS_TH(th_vmsle_vx, opivx_cmp_check_th)
GEN_OPIVX_TRANS_TH(th_vmsgtu_vx, opivx_cmp_check_th)
GEN_OPIVX_TRANS_TH(th_vmsgt_vx, opivx_cmp_check_th)

GEN_OPIVI_TRANS_TH(th_vmseq_vi, IMM_SX, th_vmseq_vx, opivx_cmp_check_th)
GEN_OPIVI_TRANS_TH(th_vmsne_vi, IMM_SX, th_vmsne_vx, opivx_cmp_check_th)
GEN_OPIVI_TRANS_TH(th_vmsleu_vi, IMM_ZX, th_vmsleu_vx, opivx_cmp_check_th)
GEN_OPIVI_TRANS_TH(th_vmsle_vi, IMM_SX, th_vmsle_vx, opivx_cmp_check_th)
GEN_OPIVI_TRANS_TH(th_vmsgtu_vi, IMM_ZX, th_vmsgtu_vx, opivx_cmp_check_th)
GEN_OPIVI_TRANS_TH(th_vmsgt_vi, IMM_SX, th_vmsgt_vx, opivx_cmp_check_th)

/* Vector Integer Min/Max Instructions */
GEN_OPIVV_GVEC_TRANS_TH(th_vminu_vv, umin)
GEN_OPIVV_GVEC_TRANS_TH(th_vmin_vv,  smin)
GEN_OPIVV_GVEC_TRANS_TH(th_vmaxu_vv, umax)
GEN_OPIVV_GVEC_TRANS_TH(th_vmax_vv,  smax)
GEN_OPIVX_TRANS_TH(th_vminu_vx, opivx_check_th)
GEN_OPIVX_TRANS_TH(th_vmin_vx,  opivx_check_th)
GEN_OPIVX_TRANS_TH(th_vmaxu_vx, opivx_check_th)
GEN_OPIVX_TRANS_TH(th_vmax_vx,  opivx_check_th)

/* Vector Single-Width Integer Multiply Instructions */
GEN_OPIVV_GVEC_TRANS_TH(th_vmul_vv,  mul)
GEN_OPIVV_TRANS_TH(th_vmulh_vv, opivv_check_th)
GEN_OPIVV_TRANS_TH(th_vmulhu_vv, opivv_check_th)
GEN_OPIVV_TRANS_TH(th_vmulhsu_vv, opivv_check_th)
GEN_OPIVX_GVEC_TRANS_TH(th_vmul_vx,  muls)
GEN_OPIVX_TRANS_TH(th_vmulh_vx, opivx_check_th)
GEN_OPIVX_TRANS_TH(th_vmulhu_vx, opivx_check_th)
GEN_OPIVX_TRANS_TH(th_vmulhsu_vx, opivx_check_th)

/* Vector Integer Divide Instructions */
GEN_OPIVV_TRANS_TH(th_vdivu_vv, opivv_check_th)
GEN_OPIVV_TRANS_TH(th_vdiv_vv, opivv_check_th)
GEN_OPIVV_TRANS_TH(th_vremu_vv, opivv_check_th)
GEN_OPIVV_TRANS_TH(th_vrem_vv, opivv_check_th)
GEN_OPIVX_TRANS_TH(th_vdivu_vx, opivx_check_th)
GEN_OPIVX_TRANS_TH(th_vdiv_vx, opivx_check_th)
GEN_OPIVX_TRANS_TH(th_vremu_vx, opivx_check_th)
GEN_OPIVX_TRANS_TH(th_vrem_vx, opivx_check_th)

/* Vector Widening Integer Multiply Instructions */
GEN_OPIVV_WIDEN_TRANS_TH(th_vwmul_vv, opivv_widen_check_th)
GEN_OPIVV_WIDEN_TRANS_TH(th_vwmulu_vv, opivv_widen_check_th)
GEN_OPIVV_WIDEN_TRANS_TH(th_vwmulsu_vv, opivv_widen_check_th)
GEN_OPIVX_WIDEN_TRANS_TH(th_vwmul_vx, opivx_widen_check_th)
GEN_OPIVX_WIDEN_TRANS_TH(th_vwmulu_vx, opivx_widen_check_th)
GEN_OPIVX_WIDEN_TRANS_TH(th_vwmulsu_vx, opivx_widen_check_th)

/* Vector Single-Width Integer Multiply-Add Instructions */
GEN_OPIVV_TRANS_TH(th_vmacc_vv, opivv_check_th)
GEN_OPIVV_TRANS_TH(th_vnmsac_vv, opivv_check_th)
GEN_OPIVV_TRANS_TH(th_vmadd_vv, opivv_check_th)
GEN_OPIVV_TRANS_TH(th_vnmsub_vv, opivv_check_th)
GEN_OPIVX_TRANS_TH(th_vmacc_vx, opivx_check_th)
GEN_OPIVX_TRANS_TH(th_vnmsac_vx, opivx_check_th)
GEN_OPIVX_TRANS_TH(th_vmadd_vx, opivx_check_th)
GEN_OPIVX_TRANS_TH(th_vnmsub_vx, opivx_check_th)

/* Vector Widening Integer Multiply-Add Instructions */
GEN_OPIVV_WIDEN_TRANS_TH(th_vwmaccu_vv, opivx_widen_check_th)
GEN_OPIVV_WIDEN_TRANS_TH(th_vwmacc_vv, opivx_widen_check_th)
GEN_OPIVV_WIDEN_TRANS_TH(th_vwmaccsu_vv, opivx_widen_check_th)
GEN_OPIVX_WIDEN_TRANS_TH(th_vwmaccu_vx, opivx_widen_check_th)
GEN_OPIVX_WIDEN_TRANS_TH(th_vwmacc_vx, opivx_widen_check_th)
GEN_OPIVX_WIDEN_TRANS_TH(th_vwmaccsu_vx, opivx_widen_check_th)
GEN_OPIVX_WIDEN_TRANS_TH(th_vwmaccus_vx, opivx_widen_check_th)

/* Vector Integer Merge and Move Instructions */
/*
 * This function is almost the copy of trans_vmv_v_v, except:
 * 1) XTheadVector simplifies the judgment logic of whether
 *    to accelerate or not for its lack of fractional LMUL and
 *    VTA.
 */
static bool trans_th_vmv_v_v(DisasContext *s, arg_th_vmv_v_v *a)
{
    if (require_thv(s) &&
        th_check_isa_ill(s) &&
        th_check_reg(s, a->rd, false) &&
        th_check_reg(s, a->rs1, false)) {

        if (s->vl_eq_vlmax) {
            tcg_gen_gvec_mov(s->sew, th_vreg_ofs(s, a->rd),
                             th_vreg_ofs(s, a->rs1),
                             MAXSZ_TH(s), MAXSZ_TH(s));
        } else {
            uint32_t data = FIELD_DP32(0, VDATA_TH, LMUL, s->lmul);
            static gen_helper_gvec_2_ptr * const fns[4] = {
                gen_helper_th_vmv_v_v_b, gen_helper_th_vmv_v_v_h,
                gen_helper_th_vmv_v_v_w, gen_helper_th_vmv_v_v_d,
            };
            TCGLabel *over = gen_new_label();
            tcg_gen_brcond_tl(TCG_COND_GEU, cpu_vstart, cpu_vl, over);

            tcg_gen_gvec_2_ptr(th_vreg_ofs(s, a->rd), th_vreg_ofs(s, a->rs1),
                               tcg_env, s->cfg_ptr->vlen / 8,
                               s->cfg_ptr->vlen / 8, data,
                               fns[s->sew]);
            gen_set_label(over);
        }
        mark_vs_dirty(s);
        return true;
    }
    return false;
}


#define gen_helper_vmv_vx_th gen_helper_vmv_vx
/*
 * This function is almost the copy of trans_vmv_v_x, except:
 * 1) Simplier judgment logic of acceleration
 * 2) XTheadVector has no limit of SEW of 8 to 64, Therefore, it is not
 *    suitable to use acceleration when xlen < SEW.
 */
static bool trans_th_vmv_v_x(DisasContext *s, arg_th_vmv_v_x *a)
{
    if (require_thv(s) &&
        th_check_isa_ill(s) &&
        th_check_reg(s, a->rd, false)) {

        TCGv s1;
        TCGLabel *over = gen_new_label();
        tcg_gen_brcond_tl(TCG_COND_GEU, cpu_vstart, cpu_vl, over);

        s1 = get_gpr(s, a->rs1, EXT_SIGN);

        if (s->vl_eq_vlmax && (8 << s->sew) <= get_xlen(s)) {
            tcg_gen_gvec_dup_tl(s->sew, th_vreg_ofs(s, a->rd),
                                MAXSZ_TH(s), MAXSZ_TH(s), s1);
        } else {
            TCGv_i32 desc;
            TCGv_i64 s1_i64 = tcg_temp_new_i64();
            TCGv_ptr dest = tcg_temp_new_ptr();
            uint32_t data = FIELD_DP32(0, VDATA_TH, LMUL, s->lmul);
            static gen_helper_vmv_vx_th * const fns[4] = {
                gen_helper_th_vmv_v_x_b, gen_helper_th_vmv_v_x_h,
                gen_helper_th_vmv_v_x_w, gen_helper_th_vmv_v_x_d,
            };

            tcg_gen_ext_tl_i64(s1_i64, s1);
            desc = tcg_constant_i32(simd_desc(s->cfg_ptr->vlen / 8,
                                              s->cfg_ptr->vlen / 8, data));
            tcg_gen_addi_ptr(dest, tcg_env, th_vreg_ofs(s, a->rd));
            fns[s->sew](dest, s1_i64, tcg_env, desc);
        }

        mark_vs_dirty(s);
        gen_set_label(over);
        return true;
    }
    return false;
}

/* The difference is same as trans_th_vmv_v_v */
static bool trans_th_vmv_v_i(DisasContext *s, arg_th_vmv_v_i *a)
{
    if (require_thv(s) &&
        th_check_isa_ill(s) &&
        th_check_reg(s, a->rd, false)) {

        int64_t simm = sextract64(a->rs1, 0, 5);
        if (s->vl_eq_vlmax) {
            tcg_gen_gvec_dup_imm(s->sew, th_vreg_ofs(s, a->rd),
                                 MAXSZ_TH(s), MAXSZ_TH(s), simm);
            mark_vs_dirty(s);
        } else {
            TCGv_i32 desc;
            TCGv_i64 s1;
            TCGv_ptr dest;
            uint32_t data = FIELD_DP32(0, VDATA_TH, LMUL, s->lmul);
            static gen_helper_vmv_vx_th * const fns[4] = {
                gen_helper_th_vmv_v_x_b, gen_helper_th_vmv_v_x_h,
                gen_helper_th_vmv_v_x_w, gen_helper_th_vmv_v_x_d,
            };

            TCGLabel *over = gen_new_label();
            tcg_gen_brcond_tl(TCG_COND_GEU, cpu_vstart, cpu_vl, over);

            s1 = tcg_constant_i64(simm);
            dest = tcg_temp_new_ptr();
            desc = tcg_constant_i32(simd_desc(s->cfg_ptr->vlen / 8,
                                              s->cfg_ptr->vlen / 8, data));
            tcg_gen_addi_ptr(dest, tcg_env, th_vreg_ofs(s, a->rd));
            fns[s->sew](dest, s1, tcg_env, desc);

            mark_vs_dirty(s);
            gen_set_label(over);
        }
        return true;
    }
    return false;
}

GEN_OPIVV_TRANS_TH(th_vmerge_vvm, opivv_vadc_check_th)
GEN_OPIVX_TRANS_TH(th_vmerge_vxm, opivx_vadc_check_th)
GEN_OPIVI_TRANS_TH(th_vmerge_vim, IMM_SX, th_vmerge_vxm, opivx_vadc_check_th)

/*
 *** Vector Fixed-Point Arithmetic Instructions
 */

/* Vector Single-Width Saturating Add and Subtract */
GEN_OPIVV_TRANS_TH(th_vsaddu_vv, opivv_check_th)
GEN_OPIVV_TRANS_TH(th_vsadd_vv,  opivv_check_th)
GEN_OPIVV_TRANS_TH(th_vssubu_vv, opivv_check_th)
GEN_OPIVV_TRANS_TH(th_vssub_vv,  opivv_check_th)
GEN_OPIVX_TRANS_TH(th_vsaddu_vx,  opivx_check_th)
GEN_OPIVX_TRANS_TH(th_vsadd_vx,  opivx_check_th)
GEN_OPIVX_TRANS_TH(th_vssubu_vx,  opivx_check_th)
GEN_OPIVX_TRANS_TH(th_vssub_vx,  opivx_check_th)
GEN_OPIVI_TRANS_TH(th_vsaddu_vi, IMM_ZX, th_vsaddu_vx, opivx_check_th)
GEN_OPIVI_TRANS_TH(th_vsadd_vi, IMM_SX, th_vsadd_vx, opivx_check_th)

/* Vector Single-Width Averaging Add and Subtract */
GEN_OPIVV_TRANS_TH(th_vaadd_vv, opivv_check_th)
GEN_OPIVV_TRANS_TH(th_vasub_vv, opivv_check_th)
GEN_OPIVX_TRANS_TH(th_vaadd_vx,  opivx_check_th)
GEN_OPIVX_TRANS_TH(th_vasub_vx,  opivx_check_th)
GEN_OPIVI_TRANS_TH(th_vaadd_vi, IMM_SX, th_vaadd_vx, opivx_check_th)

/* Vector Single-Width Fractional Multiply with Rounding and Saturation */
GEN_OPIVV_TRANS_TH(th_vsmul_vv, opivv_check_th)
GEN_OPIVX_TRANS_TH(th_vsmul_vx, opivx_check_th)

/* Vector Widening Saturating Scaled Multiply-Add */
GEN_OPIVV_WIDEN_TRANS_TH(th_vwsmaccu_vv, opivv_widen_check_th)
GEN_OPIVV_WIDEN_TRANS_TH(th_vwsmacc_vv, opivv_widen_check_th)
GEN_OPIVV_WIDEN_TRANS_TH(th_vwsmaccsu_vv, opivv_widen_check_th)
GEN_OPIVX_WIDEN_TRANS_TH(th_vwsmaccu_vx, opivx_widen_check_th)
GEN_OPIVX_WIDEN_TRANS_TH(th_vwsmacc_vx, opivx_widen_check_th)
GEN_OPIVX_WIDEN_TRANS_TH(th_vwsmaccsu_vx, opivx_widen_check_th)
GEN_OPIVX_WIDEN_TRANS_TH(th_vwsmaccus_vx, opivx_widen_check_th)

/* Vector Single-Width Scaling Shift Instructions */
GEN_OPIVV_TRANS_TH(th_vssrl_vv, opivv_check_th)
GEN_OPIVV_TRANS_TH(th_vssra_vv, opivv_check_th)
GEN_OPIVX_TRANS_TH(th_vssrl_vx,  opivx_check_th)
GEN_OPIVX_TRANS_TH(th_vssra_vx,  opivx_check_th)
GEN_OPIVI_TRANS_TH(th_vssrl_vi, IMM_TRUNC_SEW, th_vssrl_vx, opivx_check_th)
GEN_OPIVI_TRANS_TH(th_vssra_vi, IMM_TRUNC_SEW, th_vssra_vx, opivx_check_th)

#define TH_TRANS_STUB(NAME)                                \
static bool trans_##NAME(DisasContext *s, arg_##NAME *a)   \
{                                                          \
    return require_thv(s);                                 \
}

TH_TRANS_STUB(th_vnclipu_vv)
TH_TRANS_STUB(th_vnclipu_vx)
TH_TRANS_STUB(th_vnclipu_vi)
TH_TRANS_STUB(th_vnclip_vv)
TH_TRANS_STUB(th_vnclip_vx)
TH_TRANS_STUB(th_vnclip_vi)
TH_TRANS_STUB(th_vfadd_vv)
TH_TRANS_STUB(th_vfadd_vf)
TH_TRANS_STUB(th_vfsub_vv)
TH_TRANS_STUB(th_vfsub_vf)
TH_TRANS_STUB(th_vfrsub_vf)
TH_TRANS_STUB(th_vfwadd_vv)
TH_TRANS_STUB(th_vfwadd_vf)
TH_TRANS_STUB(th_vfwadd_wv)
TH_TRANS_STUB(th_vfwadd_wf)
TH_TRANS_STUB(th_vfwsub_vv)
TH_TRANS_STUB(th_vfwsub_vf)
TH_TRANS_STUB(th_vfwsub_wv)
TH_TRANS_STUB(th_vfwsub_wf)
TH_TRANS_STUB(th_vfmul_vv)
TH_TRANS_STUB(th_vfmul_vf)
TH_TRANS_STUB(th_vfdiv_vv)
TH_TRANS_STUB(th_vfdiv_vf)
TH_TRANS_STUB(th_vfrdiv_vf)
TH_TRANS_STUB(th_vfwmul_vv)
TH_TRANS_STUB(th_vfwmul_vf)
TH_TRANS_STUB(th_vfmacc_vv)
TH_TRANS_STUB(th_vfnmacc_vv)
TH_TRANS_STUB(th_vfnmacc_vf)
TH_TRANS_STUB(th_vfmacc_vf)
TH_TRANS_STUB(th_vfmsac_vv)
TH_TRANS_STUB(th_vfmsac_vf)
TH_TRANS_STUB(th_vfnmsac_vv)
TH_TRANS_STUB(th_vfnmsac_vf)
TH_TRANS_STUB(th_vfmadd_vv)
TH_TRANS_STUB(th_vfmadd_vf)
TH_TRANS_STUB(th_vfnmadd_vv)
TH_TRANS_STUB(th_vfnmadd_vf)
TH_TRANS_STUB(th_vfmsub_vv)
TH_TRANS_STUB(th_vfmsub_vf)
TH_TRANS_STUB(th_vfnmsub_vv)
TH_TRANS_STUB(th_vfnmsub_vf)
TH_TRANS_STUB(th_vfwmacc_vv)
TH_TRANS_STUB(th_vfwmacc_vf)
TH_TRANS_STUB(th_vfwnmacc_vv)
TH_TRANS_STUB(th_vfwnmacc_vf)
TH_TRANS_STUB(th_vfwmsac_vv)
TH_TRANS_STUB(th_vfwmsac_vf)
TH_TRANS_STUB(th_vfwnmsac_vv)
TH_TRANS_STUB(th_vfwnmsac_vf)
TH_TRANS_STUB(th_vfsqrt_v)
TH_TRANS_STUB(th_vfmin_vv)
TH_TRANS_STUB(th_vfmin_vf)
TH_TRANS_STUB(th_vfmax_vv)
TH_TRANS_STUB(th_vfmax_vf)
TH_TRANS_STUB(th_vfsgnj_vv)
TH_TRANS_STUB(th_vfsgnj_vf)
TH_TRANS_STUB(th_vfsgnjn_vv)
TH_TRANS_STUB(th_vfsgnjn_vf)
TH_TRANS_STUB(th_vfsgnjx_vv)
TH_TRANS_STUB(th_vfsgnjx_vf)
TH_TRANS_STUB(th_vmfeq_vv)
TH_TRANS_STUB(th_vmfeq_vf)
TH_TRANS_STUB(th_vmfne_vv)
TH_TRANS_STUB(th_vmfne_vf)
TH_TRANS_STUB(th_vmflt_vv)
TH_TRANS_STUB(th_vmflt_vf)
TH_TRANS_STUB(th_vmfle_vv)
TH_TRANS_STUB(th_vmfle_vf)
TH_TRANS_STUB(th_vmfgt_vf)
TH_TRANS_STUB(th_vmfge_vf)
TH_TRANS_STUB(th_vmford_vv)
TH_TRANS_STUB(th_vmford_vf)
TH_TRANS_STUB(th_vfclass_v)
TH_TRANS_STUB(th_vfmerge_vfm)
TH_TRANS_STUB(th_vfmv_v_f)
TH_TRANS_STUB(th_vfcvt_xu_f_v)
TH_TRANS_STUB(th_vfcvt_x_f_v)
TH_TRANS_STUB(th_vfcvt_f_xu_v)
TH_TRANS_STUB(th_vfcvt_f_x_v)
TH_TRANS_STUB(th_vfwcvt_xu_f_v)
TH_TRANS_STUB(th_vfwcvt_x_f_v)
TH_TRANS_STUB(th_vfwcvt_f_xu_v)
TH_TRANS_STUB(th_vfwcvt_f_x_v)
TH_TRANS_STUB(th_vfwcvt_f_f_v)
TH_TRANS_STUB(th_vfncvt_xu_f_v)
TH_TRANS_STUB(th_vfncvt_x_f_v)
TH_TRANS_STUB(th_vfncvt_f_xu_v)
TH_TRANS_STUB(th_vfncvt_f_x_v)
TH_TRANS_STUB(th_vfncvt_f_f_v)
TH_TRANS_STUB(th_vredsum_vs)
TH_TRANS_STUB(th_vredand_vs)
TH_TRANS_STUB(th_vredor_vs)
TH_TRANS_STUB(th_vredxor_vs)
TH_TRANS_STUB(th_vredminu_vs)
TH_TRANS_STUB(th_vredmin_vs)
TH_TRANS_STUB(th_vredmaxu_vs)
TH_TRANS_STUB(th_vredmax_vs)
TH_TRANS_STUB(th_vwredsumu_vs)
TH_TRANS_STUB(th_vwredsum_vs)
TH_TRANS_STUB(th_vfredsum_vs)
TH_TRANS_STUB(th_vfredmin_vs)
TH_TRANS_STUB(th_vfredmax_vs)
TH_TRANS_STUB(th_vfwredsum_vs)
TH_TRANS_STUB(th_vmand_mm)
TH_TRANS_STUB(th_vmnand_mm)
TH_TRANS_STUB(th_vmandnot_mm)
TH_TRANS_STUB(th_vmxor_mm)
TH_TRANS_STUB(th_vmor_mm)
TH_TRANS_STUB(th_vmnor_mm)
TH_TRANS_STUB(th_vmornot_mm)
TH_TRANS_STUB(th_vmxnor_mm)
TH_TRANS_STUB(th_vmpopc_m)
TH_TRANS_STUB(th_vmfirst_m)
TH_TRANS_STUB(th_vmsbf_m)
TH_TRANS_STUB(th_vmsif_m)
TH_TRANS_STUB(th_vmsof_m)
TH_TRANS_STUB(th_viota_m)
TH_TRANS_STUB(th_vid_v)
TH_TRANS_STUB(th_vext_x_v)
TH_TRANS_STUB(th_vmv_s_x)
TH_TRANS_STUB(th_vfmv_f_s)
TH_TRANS_STUB(th_vfmv_s_f)
TH_TRANS_STUB(th_vslideup_vx)
TH_TRANS_STUB(th_vslideup_vi)
TH_TRANS_STUB(th_vslide1up_vx)
TH_TRANS_STUB(th_vslidedown_vx)
TH_TRANS_STUB(th_vslidedown_vi)
TH_TRANS_STUB(th_vslide1down_vx)
TH_TRANS_STUB(th_vrgather_vv)
TH_TRANS_STUB(th_vrgather_vx)
TH_TRANS_STUB(th_vrgather_vi)
TH_TRANS_STUB(th_vcompress_vm)
