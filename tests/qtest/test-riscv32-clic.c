/*
 * QTest testcase for the E906 CLIC (Core Local Interrupt Controller)
 *
 * Copyright (c) 2021 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, see <http://www.gnu.org/licenses/>.
 */

#include "qemu/osdep.h"
#include "libqtest-single.h"

/* clic reg addr */
#define SMARTL_CLIC_MMODE_BASE 0xe0800000
#define CLICCFG_ADDR (SMARTL_CLIC_MMODE_BASE + 0)
#define CLICINFO_ADDR (SMARTL_CLIC_MMODE_BASE + 4)

/* generate reg addr */
#define GEN_CLIC_IRQ_REG(irq_num) \
    uint64_t clicintip##irq_num##_addr =                                \
    SMARTL_CLIC_MMODE_BASE + 0x1000 + 4 * irq_num;                      \
    uint64_t clicintie##irq_num##_addr =                                \
    SMARTL_CLIC_MMODE_BASE + 0x1001 + 4 * irq_num;                      \
    uint64_t clicintattr##irq_num##_addr =                              \
    SMARTL_CLIC_MMODE_BASE + 0x1002 + 4 * irq_num;                      \
    uint64_t clicintctl##irq_num##_addr =                               \
    SMARTL_CLIC_MMODE_BASE + 0x1003 + 4 * irq_num;

/* test variable for configure case and we ues 12 irq to test */
GEN_CLIC_IRQ_REG(12)

/* test variable for interrupt case we ues irq 25 and irq 26 to test */
GEN_CLIC_IRQ_REG(25)
GEN_CLIC_IRQ_REG(26)

/* generate configure function */
#define GEN_CHECK_REG_MMIO(CASE_NAME, WIDTH, reg_addr, set_value, expected) \
static void test_configure_##CASE_NAME(void)                                \
{                                                                           \
    uint8_t _set_value = set_value;                                         \
    uint8_t _expected = expected;                                           \
    write##WIDTH(reg_addr, _set_value);                       \
    uint8_t result = read##WIDTH(reg_addr);                   \
    g_assert_cmpuint(result, ==, _expected);                                \
}

/* test case defination */
GEN_CHECK_REG_MMIO(cliccfg_min_nlbits, b, CLICCFG_ADDR, 0x1, 0x1)
/* set nlbits = 0, nmbits = 0, nvbits = 1 and compare */

GEN_CHECK_REG_MMIO(cliccfg_supported_max_nlbits, b, CLICCFG_ADDR, 0x11, 0x11)
/* set nlbits = 8, nmbits = 0, nvbits = 1 and compare */

GEN_CHECK_REG_MMIO(cliccfg_unsupported_nlbits, b, CLICCFG_ADDR, 0x15, 0x11)
/* set nlbits = 10, nmbits = 0, nvbits = 1 and compare */

GEN_CHECK_REG_MMIO(cliccfg_unsupported_nmbits, b, CLICCFG_ADDR, 0x51, 0x11)
/* set nmbits = 2, nlbits = 8, nvbits = 1 and compare */

GEN_CHECK_REG_MMIO(clicintie_enable, b, clicintie12_addr, 0x1, 0x1)
/* set clicintie[i] = 0x1 and compare */

GEN_CHECK_REG_MMIO(clicintie_disable, b, clicintie12_addr, 0, 0)
/* set clicintie[i] = 0x0 and compare */

GEN_CHECK_REG_MMIO(clicintattr_mode_warl, b, clicintattr12_addr,
                   0x41, 0xc1)
/*
 * set
 * cliccfg = 0x11
 * mode = b01, tri = b00, shv = b1
 * clicintattr[i] = 0x61
 * expected
 * mode = b11, tri = b00, shv = b1
 * clicintattr[i] = 0xc1
 */

GEN_CHECK_REG_MMIO(clicintattr_unsupported_mode, b, clicintattr12_addr,
                   0x81, 0xc1)
/*
 * set
 * cliccfg = 0x11
 * mode = b10, tri = b00, shv = b1
 * clicintattr[i] = 0x81
 * expected
 * mode = b11, tri = b00, shv = b1
 * clicintattr[i] = 0xc1
 */

GEN_CHECK_REG_MMIO(clicintattr_positive_edge_triggered, b, clicintattr12_addr,
                   0xc1, 0xc1)
/*
 * set
 * cliccfg = 0x11
 * mode = b11, tri = b10, shv = b1
 * clicintattr[i] = 0xc4
 * expected
 * mode = b11, tri = b10, shv = b1
 * clicintattr[i] = 0xc4
 */

GEN_CHECK_REG_MMIO(clicintattr_negative_edge_triggered, b, clicintattr12_addr,
                   0xc3, 0xc3)
/*
 * set
 * cliccfg = 0x11
 * mode = b11, tri = b11, shv = b1
 * clicintattr[i] = 0xc6
 * expected
 * mode = b11, tri = b11, shv = b1
 * clicintattr[i] = 0xc6
 */

GEN_CHECK_REG_MMIO(clicintattr_positive_level_triggered, b, clicintattr12_addr,
                   0xc5, 0xc5)
/*
 * set
 * cliccfg = 0x11
 * mode = b11, tri = b00, shv = b1
 * clicintattr[i] = 0xc1
 * expected
 * mode = b11, tri = b00, shv = b1
 * clicintattr[i] = 0xc1
 */

GEN_CHECK_REG_MMIO(clicintattr_negative_level_triggered, b, clicintattr12_addr,
                   0xc7, 0xc7)
/*
 * set
 * cliccfg = 0x11
 * mode = b11, tri = b10, shv = b1
 * clicintattr[i] = 0xc5
 * expected
 * mode = b11, tri = b10, shv = b1
 * clicintattr[i] = 0xc5
 */

GEN_CHECK_REG_MMIO(clicintattr_none_vectored, b, clicintattr12_addr,
                   0xc6, 0xc6)
/*
 * set
 * cliccfg = 0x11
 * mode = b11, tri = b11, shv = b0
 * clicintattr[i] = 0xc6
 * expected
 * mode = b11, tri = b11, shv = b0
 * clicintattr[i] = 0xc6
 */

GEN_CHECK_REG_MMIO(clicintctl_warl, b, clicintctl12_addr,
                   64, (64 | 0x1F))
/*
 * set
 * cliccfg = 0x11
 * 64 is invalid value, so it won't be written
 */

GEN_CHECK_REG_MMIO(clicintctl_set_interrupt_level_63, b, clicintctl12_addr,
                   (64 | 0x1F), (64 | 0x1F))
/*
 * set level = 63
 * clicfg = 0x11
 * clicintctl[i] = 64 | 0x3F
 */

static void test_configure_clicinfo_read_only(void)
{
/*
 * read clicinfo
 * wirite a value to clicinfo
 * check the value of clicinfo that is changed ord not
 */

    uint32_t orig_value = qtest_readl(global_qtest, CLICINFO_ADDR);
    uint32_t set_value = 5;
    writel(CLICINFO_ADDR, set_value);
    uint32_t result = readl(CLICINFO_ADDR);

    g_assert_cmpuint(orig_value, ==, result);
}

static void test_configure_clicintip_level_triggered_read_only(void)
{
/*
 * read clicintip[i] to result
 * set cliccfg = 0x11, clicintattr[i] = 0x31, clicintip[i] = 0x1
 * check the value of clicintip[i] that is changed ord not
 */

    /* configure level-triggered mode */
    writeb(clicintattr12_addr, 0xc1);
    g_assert_cmpuint(readb(clicintattr12_addr), ==, 0xc1);

    uint8_t orig_value = readb(clicintip12_addr);
    writeb(clicintip12_addr, 0x1);
    uint32_t result = readb(clicintip12_addr);

    g_assert_cmpuint(orig_value, ==, result);
}

static void boot_qemu(void)
{
    global_qtest = qtest_start("-M smartl -cpu rv32");
}

static void shut_down_qemu(void)
{
    qtest_quit(global_qtest);
}

static void test_vectored_positive_level_triggered_interrupt(void)
{
/*
 * test vectored positive level triggered interrupt
 * test points:
 * 1. we ues interrupt 25 and 26 to test arbitration in two situation:
 *    same level, different level.
 * 2. within level triggered mdoe, we can only ues device to clear pending.
 * 3. within positive level triggered mode, set gpio-in rise
 *    to trigger interrupt.
 */

    QTestState *qts = qtest_start("-M smartl -cpu rv32");
    /* intercept in and out irq */
    qtest_irq_intercept_out(qts, "/machine/unattached/device[1]");
    qtest_irq_intercept_in(qts, "/machine/unattached/device[1]");

    /* configure */
    qtest_writeb(qts, CLICCFG_ADDR, 0x3);
    qtest_writeb(qts, clicintattr25_addr, 0xc1);
    qtest_writeb(qts, clicintattr26_addr, 0xc1);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 0);

    /* set pending */
    /* arbitration will be made and 26 will be delivered */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 1);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 26, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 1);
    qtest_writeb(qts, clicintie25_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie25_addr), ==, 1);
    qtest_writeb(qts, clicintie26_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie26_addr), ==, 1);
    /* trigger arbitration */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 1);
    g_assert_true(qtest_irq_delivered(qts, 26));
    g_assert_true(qtest_get_irq(qts, 0));

    /*
     * level trigger wouldn't auto clear clear pending,
     * so we need to manually do it.
     */
    qtest_writeb(qts, clicintie25_addr, 0);
    qtest_writeb(qts, clicintie26_addr, 0);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 0);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 26, 0);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 0, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 0);

    /* set interrupt 25 level 255, interrupt 26 level 127 */
    /* arbitration will be made and 25 will rise */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 1);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 26, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 1);
    qtest_writeb(qts, clicintctl25_addr, 0xbf);
    qtest_writeb(qts, clicintctl26_addr, 0x3f);
    qtest_writeb(qts, clicintie25_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie25_addr), ==, 1);
    qtest_writeb(qts, clicintie26_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie26_addr), ==, 1);
    /* trigger arbitration */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 1);
    g_assert_true(qtest_irq_delivered(qts, 25));
    g_assert_true(qtest_get_irq(qts, 0));

    qtest_quit(qts);
}

static void test_vectored_negative_level_triggered_interrupt(void)
{
/*
 * test vectored negative level triggered interrupt
 * test points:
 * 1. we ues interrupt 25 and 26 to test arbitration in two situation:
 *    same level, different level.
 * 2. within level triggered mdoe, we can only ues device to clear pending.
 * 3. within negative level triggered mode,
 *    set gpio-in lower to trigger interrupt.
 */

    QTestState *qts = qtest_start("-M smartl -cpu rv32");
    /* intercept in and out irq */
    qtest_irq_intercept_out(qts, "/machine/unattached/device[1]");
    qtest_irq_intercept_in(qts, "/machine/unattached/device[1]");

    /* configure */
    qtest_writeb(qts, CLICCFG_ADDR, 0x3);
    qtest_writeb(qts, clicintattr25_addr, 0xc5);
    qtest_readb(qts, clicintattr25_addr);
    qtest_writeb(qts, clicintattr26_addr, 0xc5);
    qtest_readb(qts, clicintattr26_addr);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 1);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 26, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 0);

    /* set pending */
    /* arbitration will be made and 26 will be delivered */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 1);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 26, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 1);
    qtest_writeb(qts, clicintie25_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie25_addr), ==, 1);
    qtest_writeb(qts, clicintie26_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie26_addr), ==, 1);
    /* trigger arbitration */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 0);
    g_assert_true(qtest_irq_delivered(qts, 26));
    g_assert_true(qtest_get_irq(qts, 0));

    /*
     * level trigger wouldn't auto clear clear pending,
     * so we need to manually do it.
     */
    qtest_writeb(qts, clicintie25_addr, 0);
    qtest_writeb(qts, clicintie26_addr, 0);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 1);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 26, 1);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 0, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 0);

    /* set interrupt 25 level 255, interrupt 26 level 127 */
    /* arbitration will be made and 25 will rise */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 1);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 26, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 1);
    qtest_writeb(qts, clicintctl25_addr, 0xbf);
    qtest_writeb(qts, clicintctl26_addr, 0x3f);
    qtest_writeb(qts, clicintie25_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie25_addr), ==, 1);
    qtest_writeb(qts, clicintie26_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie26_addr), ==, 1);
    /* trigger arbitration */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 0);
    g_assert_true(qtest_irq_delivered(qts, 25));
    g_assert_true(qtest_get_irq(qts, 0));

    qtest_quit(qts);
}

static void test_vectored_positive_edge_triggered_interrupt(void)
{
/*
 * test vectored positive edge triggered interrupt
 * test points:
 * 1. we ues interrupt 25 and 26 to test arbitration in two situation:
 *    same level, different level.
 * 2. within vectored edge triggered mdoe, pending bit will be
 *    automatically cleared.
 * 3. within positive edge triggered mode, set gpio-in from
 *    lower to rise to trigger interrupt.
 */

    QTestState *qts = qtest_start("-M smartl -cpu rv32");
    /* intercept in and out irq */
    qtest_irq_intercept_out(qts, "/machine/unattached/device[1]");
    qtest_irq_intercept_in(qts, "/machine/unattached/device[1]");

    /* configure */
    qtest_writeb(qts, CLICCFG_ADDR, 0x3);
    qtest_writeb(qts, clicintattr25_addr, 0xc3);
    qtest_writeb(qts, clicintattr26_addr, 0xc3);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 0);

    /* set pending */
    /* arbitration will be made and 26 will be delivered */
    qtest_writeb(qts, clicintip25_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 1);
    qtest_writeb(qts, clicintip26_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 1);
    qtest_writeb(qts, clicintie25_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie25_addr), ==, 1);
    qtest_writeb(qts, clicintie26_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie26_addr), ==, 1);
    /* trigger arbitration */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 1);
    g_assert_true(qtest_irq_delivered(qts, 26));
    g_assert_true(qtest_get_irq(qts, 0));

    /* vectored edge trigger will auto clear clear pending */
    qtest_writeb(qts, clicintie25_addr, 0);
    qtest_writeb(qts, clicintie26_addr, 0);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 0);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 0, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 0);

    /* set interrupt 25 level 255, interrupt 26 level 127 */
    /* arbitration will be made and 25 will rise */
    qtest_writeb(qts, clicintip25_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 1);
    qtest_writeb(qts, clicintip26_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 1);
    qtest_writeb(qts, clicintctl25_addr, 0xbf);
    qtest_writeb(qts, clicintctl26_addr, 0x3f);
    qtest_writeb(qts, clicintie25_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie25_addr), ==, 1);
    qtest_writeb(qts, clicintie26_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie26_addr), ==, 1);
    /* trigger arbitration */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 1);
    g_assert_true(qtest_irq_delivered(qts, 25));
    g_assert_true(qtest_get_irq(qts, 0));

    qtest_quit(qts);
}

static void test_vectored_negative_edge_triggered_interrupt(void)
{
/*
 * test vectored negative edge triggered interrupt
 * test points:
 * 1. we ues interrupt 25 and 26 to test arbitration in two situation:
 *    same level, different level.
 * 2. within vectored edge triggered mdoe, pending bit will be
 *    automatically cleared.
 * 3. within negative edge triggered mode, set gpio-in from
 *    rise to lower to trigger interrupt.
 */

    QTestState *qts = qtest_start("-M smartl -cpu rv32");
    /* intercept in and out irq */
    qtest_irq_intercept_out(qts, "/machine/unattached/device[1]");
    qtest_irq_intercept_in(qts, "/machine/unattached/device[1]");

    /* configure */
    qtest_writeb(qts, CLICCFG_ADDR, 0x3);
    qtest_writeb(qts, clicintattr25_addr, 0xc7);
    qtest_writeb(qts, clicintattr26_addr, 0xc7);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 1);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 26, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 0);

    /* set pending */
    /* arbitration will be made and 26 will be delivered */
    qtest_writeb(qts, clicintip25_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 1);
    qtest_writeb(qts, clicintip26_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 1);
    qtest_writeb(qts, clicintie25_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie25_addr), ==, 1);
    qtest_writeb(qts, clicintie26_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie26_addr), ==, 1);
    /* trigger arbitration */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 0);
    g_assert_true(qtest_irq_delivered(qts, 26));
    g_assert_true(qtest_get_irq(qts, 0));

    /* vectored edge trigger will auto clear clear pending */
    qtest_writeb(qts, clicintie25_addr, 0);
    qtest_writeb(qts, clicintie26_addr, 0);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 1);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 0, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 0);

    /* set interrupt 25 level 255, interrupt 26 level 127 */
    /* arbitration will be made and 25 will rise */
    qtest_writeb(qts, clicintip25_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 1);
    qtest_writeb(qts, clicintip26_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 1);
    qtest_writeb(qts, clicintctl25_addr, 0xbf);
    qtest_writeb(qts, clicintctl26_addr, 0x3f);
    qtest_writeb(qts, clicintie25_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie25_addr), ==, 1);
    qtest_writeb(qts, clicintie26_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie26_addr), ==, 1);
    /* trigger arbitration */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 0);
    g_assert_true(qtest_irq_delivered(qts, 25));
    g_assert_true(qtest_get_irq(qts, 0));

    qtest_quit(qts);
}

static void test_unvectored_positive_level_triggered_interrupt(void)
{
/*
 * test unvectored positive level triggered interrupt
 * test points:
 * 1. we ues interrupt 25 and 26 to test arbitration in two situation:
 *    same level, different level.
 * 2. within level triggered mdoe, we can only ues device to clear pending.
 * 3. within positive level triggered mode, set gpio-in rise to
 *    trigger interrupt.
 */

    QTestState *qts = qtest_start("-M smartl -cpu rv32");
    /* intercept in and out irq */
    qtest_irq_intercept_out(qts, "/machine/unattached/device[1]");
    qtest_irq_intercept_in(qts, "/machine/unattached/device[1]");

    /* configure */
    qtest_writeb(qts, CLICCFG_ADDR, 0x3);
    qtest_writeb(qts, clicintattr25_addr, 0xc0);
    qtest_writeb(qts, clicintattr26_addr, 0xc0);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 0);

    /* set pending */
    /* arbitration will be made and 26 will be delivered */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 1);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 26, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 1);
    qtest_writeb(qts, clicintie25_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie25_addr), ==, 1);
    qtest_writeb(qts, clicintie26_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie26_addr), ==, 1);
    /* trigger arbitration */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 1);
    g_assert_true(qtest_irq_delivered(qts, 26));
    g_assert_true(qtest_get_irq(qts, 0));

    /*
     * level trigger wouldn't auto clear clear pending,
     * so we need to manually do it.
     */
    qtest_writeb(qts, clicintie25_addr, 0);
    qtest_writeb(qts, clicintie26_addr, 0);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 0);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 26, 0);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 0, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 0);

    /* set interrupt 25 level 255, interrupt 26 level 127 */
    /* arbitration will be made and 25 will rise */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 1);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 26, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 1);
    qtest_writeb(qts, clicintctl25_addr, 0xbf);
    qtest_writeb(qts, clicintctl26_addr, 0x3f);
    qtest_writeb(qts, clicintie25_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie25_addr), ==, 1);
    qtest_writeb(qts, clicintie26_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie26_addr), ==, 1);
    /* trigger arbitration */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 1);
    g_assert_true(qtest_irq_delivered(qts, 25));
    g_assert_true(qtest_get_irq(qts, 0));

    qtest_quit(qts);
}

static void test_unvectored_negative_level_triggered_interrupt(void)
{
/*
 * test unvectored negative level triggered interrupt
 * test points:
 * 1. we ues interrupt 25 and 26 to test arbitration in two situation:
 *    same level, different level.
 * 2. within level triggered mdoe, we can only ues device to clear pending.
 * 3. within negative level triggered mode, set gpio-in lower
 *    to trigger interrupt.
 */

    QTestState *qts = qtest_start("-M smartl -cpu rv32");
    /* intercept in and out irq */
    qtest_irq_intercept_out(qts, "/machine/unattached/device[1]");
    qtest_irq_intercept_in(qts, "/machine/unattached/device[1]");

    /* configure */
    qtest_writeb(qts, CLICCFG_ADDR, 0x3);
    qtest_writeb(qts, clicintattr25_addr, 0xc4);
    qtest_writeb(qts, clicintattr26_addr, 0xc4);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 1);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 26, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 0);

    /* set pending */
    /* arbitration will be made and 26 will be delivered */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 1);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 26, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 1);
    qtest_writeb(qts, clicintie25_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie25_addr), ==, 1);
    qtest_writeb(qts, clicintie26_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie26_addr), ==, 1);
    /* trigger arbitration */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 0);
    g_assert_true(qtest_irq_delivered(qts, 26));
    g_assert_true(qtest_get_irq(qts, 0));

    /*
     * level trigger wouldn't auto clear clear pending,
     * so we need to manually do it.
     */
    qtest_writeb(qts, clicintie25_addr, 0);
    qtest_writeb(qts, clicintie26_addr, 0);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 1);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 26, 1);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 0, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 0);

    /* set interrupt 25 level 255, interrupt 26 level 127 */
    /* arbitration will be made and 25 will rise */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 1);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 26, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 1);
    qtest_writeb(qts, clicintctl25_addr, 0xbf);
    qtest_writeb(qts, clicintctl26_addr, 0x3f);
    qtest_writeb(qts, clicintie25_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie25_addr), ==, 1);
    qtest_writeb(qts, clicintie26_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie26_addr), ==, 1);
    /* trigger arbitration */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 0);
    g_assert_true(qtest_irq_delivered(qts, 25));
    g_assert_true(qtest_get_irq(qts, 0));

    qtest_quit(qts);
}

static void test_unvectored_positive_edge_triggered_interrupt(void)
{
/*
 * test unvectored positive edge triggered interrupt
 * test points:
 * 1. we ues interrupt 25 and 26 to test arbitration in same level
 * 2. within unvectored edge triggered mdoe, pending bit can be
 *    cleared by using nxti instruction which can't be tested in qtest.
 * 3. within positive edge triggered mode, set gpio-in
 *    from lower to rise to trigger interrupt.
 */

   QTestState *qts = qtest_start("-M smartl -cpu rv32");
    /* intercept in and out irq */
    qtest_irq_intercept_out(qts, "/machine/unattached/device[1]");
    qtest_irq_intercept_in(qts, "/machine/unattached/device[1]");

    /* configure */
    qtest_writeb(qts, CLICCFG_ADDR, 0x3);
    qtest_writeb(qts, clicintattr25_addr, 0xc2);
    qtest_writeb(qts, clicintattr26_addr, 0xc2);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 0);

    /* set pending */
    /* arbitration will be made and 26 will be delivered */
    qtest_writeb(qts, clicintip25_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 1);
    qtest_writeb(qts, clicintip26_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 1);
    qtest_writeb(qts, clicintie25_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie25_addr), ==, 1);
    qtest_writeb(qts, clicintie26_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie26_addr), ==, 1);
    /* trigger arbitration */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 0);
    g_assert_true(qtest_irq_delivered(qts, 26));
    g_assert_true(qtest_get_irq(qts, 0));

    qtest_quit(qts);
}

static void test_unvectored_negative_edge_triggered_interrupt(void)
{
/*
 * test unvectored negative edge triggered interrupt
 * test points:
 * 1. we ues interrupt 25 and 26 to test arbitration in same level
 * 2. within unvectored edge triggered mdoe, pending bit can be cleared
 *    by using nxti instruction which can't be tested in qtest.
 * 3. within positive edge triggered mode, set gpio-in from
 *    rise to lower to trigger interrupt.
 */

    QTestState *qts = qtest_start("-M smartl -cpu rv32");
    /* intercept in and out irq */
    qtest_irq_intercept_out(qts, "/machine/unattached/device[1]");
    qtest_irq_intercept_in(qts, "/machine/unattached/device[1]");

    /* configure */
    qtest_writeb(qts, CLICCFG_ADDR, 0x3);
    qtest_writeb(qts, clicintattr25_addr, 0xc6);
    qtest_writeb(qts, clicintattr26_addr, 0xc6);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 1);
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 26, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 0);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 0);

    /* set pending */
    /* arbitration will be made and 26 will be delivered */
    qtest_writeb(qts, clicintip25_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip25_addr), ==, 1);
    qtest_writeb(qts, clicintip26_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintip26_addr), ==, 1);
    qtest_writeb(qts, clicintie25_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie25_addr), ==, 1);
    qtest_writeb(qts, clicintie26_addr, 1);
    g_assert_cmpuint(qtest_readb(qts, clicintie26_addr), ==, 1);
    /* trigger arbitration */
    qtest_set_irq_in(qts, "/machine/unattached/device[1]", NULL, 25, 0);
    g_assert_true(qtest_irq_delivered(qts, 26));
    g_assert_true(qtest_get_irq(qts, 0));

    qtest_quit(qts);
}

static void clic_configure_reg_mmio_test_case(void)
{
    qtest_add_func("smartl/clic/boot_qemu",
                   boot_qemu);

    /* cliccfg configure case */
    qtest_add_func("smartl/clic/cliccfg_min_nlbits",
                   test_configure_cliccfg_min_nlbits);
    qtest_add_func("smartl/clic/cliccfg_supported_max_nlbits",
                   test_configure_cliccfg_supported_max_nlbits);
    qtest_add_func("smartl/clic/cliccfg_unsupported_nlbits",
                   test_configure_cliccfg_unsupported_nlbits);
    qtest_add_func("smartl/clic/cliccfg_unsupported_nmbits",
                   test_configure_cliccfg_unsupported_nmbits);

    /* clicinfo RO case */
    qtest_add_func("smartl/clic/clicinfo_ro",
                   test_configure_clicinfo_read_only);

    /* clicintip configure case */
    qtest_add_func("smartl/clic/clicintip_level_triggered_readonly",
                   test_configure_clicintip_level_triggered_read_only);

    /* clicintie configure case */
    qtest_add_func("smartl/clic/clicintie_enbale",
                   test_configure_clicintie_enable);
    qtest_add_func("smartl/clic/clicintie_disbale",
                   test_configure_clicintie_disable);

    /* clicintattr configure case */
    qtest_add_func("smartl/clic/clicintattr_mode_warl",
                   test_configure_clicintattr_mode_warl);
    qtest_add_func("smartl/clic/clicintattr_unsupported_mode",
                   test_configure_clicintattr_unsupported_mode);
    qtest_add_func("smartl/clic/clicintattr_positive_edge_triggered",
                   test_configure_clicintattr_positive_edge_triggered);
    qtest_add_func("smartl/clic/clicintattr_negative_edge_triggered",
                   test_configure_clicintattr_negative_edge_triggered);
    qtest_add_func("smartl/clic/clicintattr_positive_level_triggered",
                   test_configure_clicintattr_positive_level_triggered);
    qtest_add_func("smartl/clic/clicintattr_negative_level_triggered",
                   test_configure_clicintattr_negative_level_triggered);
    qtest_add_func("smartl/clic/clicintattr_none_vectored",
                   test_configure_clicintattr_none_vectored);

    /* clicintctl configure case */
    qtest_add_func("smartl/clic/clicintctl_WARL",
                   test_configure_clicintctl_warl);
    qtest_add_func("smartl/clic/clicintctl_set_interrupt_level_63",
                   test_configure_clicintctl_set_interrupt_level_63);

    qtest_add_func("smartl/clic/shout_down_qemu",
                   shut_down_qemu);
}

static void clic_irq_test_case(void)
{
    /* interrupt test case */
    qtest_add_func("smartl/clic/vectored_positive_level_triggered_interrupt",
                   test_vectored_positive_level_triggered_interrupt);
    qtest_add_func("smartl/clic/vectored_negative_level_triggered_interrupt",
                   test_vectored_negative_level_triggered_interrupt);
    qtest_add_func("smartl/clic/vectored_positive_edge_triggered_interrupt",
                   test_vectored_positive_edge_triggered_interrupt);
    qtest_add_func("smartl/clic/vectored_negative_edge_triggered_interrupt",
                   test_vectored_negative_edge_triggered_interrupt);
    qtest_add_func("smartl/clic/unvectored_positive_level_triggered_interrupt",
                   test_unvectored_positive_level_triggered_interrupt);
    qtest_add_func("smartl/clic/unvectored_negative_level_triggered_interrupt",
                   test_unvectored_negative_level_triggered_interrupt);
    qtest_add_func("smartl/clic/unvectored_positive_edge_triggered_interrupt",
                   test_unvectored_positive_edge_triggered_interrupt);
    qtest_add_func("smartl/clic/unvectored_negative_edge_triggered_interrupt",
                   test_unvectored_negative_edge_triggered_interrupt);

}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);
    g_test_set_nonfatal_assertions();

    /* test cases */
    clic_configure_reg_mmio_test_case();
    clic_irq_test_case();

    /* Run the tests */
    int ret = g_test_run();

    return ret;
}
