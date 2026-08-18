// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Parallel FPU matmul (TRISC0-2 -> buffer_C) + isolated SrcS SFPU exp (TRISC3 -> buffer_Res).
// FPU path uses Dest dvalid {FPU, PACK}. SFPU path uses UNP_S/SrcS/PACK1 only (no Dest).

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr std::uint32_t buf_desc_id_src_a = 29;
constexpr std::uint32_t buf_desc_id_src_b = 30;
constexpr std::uint32_t buf_desc_id_dst   = 31;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_matmul.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
    set_ttsync_enables<TRACK_ALL>(ckernel::TRISC_ID);

    buffer_descriptor_u bd_src_a = {0};
    bd_src_a.f.l1_addr_16B       = L1_ADDRESS(params.buffer_A[0]);
    bd_src_a.f.format            = static_cast<std::uint8_t>(formats.unpack_A_src);
    bd_src_a.f.lmt_addr_16B      = 0;
    bd_src_a.f.x_dim             = FACE_C_DIM;
    bd_src_a.f.y_dim             = FACE_R_DIM;
    bd_src_a.f.z_dim             = params.num_faces_A;

    buffer_descriptor_u bd_src_b = {0};
    bd_src_b.f.l1_addr_16B       = L1_ADDRESS(params.buffer_B[0]);
    bd_src_b.f.format            = static_cast<std::uint8_t>(formats.unpack_B_src);
    bd_src_b.f.lmt_addr_16B      = 0;
    bd_src_b.f.x_dim             = FACE_C_DIM;
    bd_src_b.f.y_dim             = FACE_R_DIM;
    bd_src_b.f.z_dim             = params.num_faces_B;

    _configure_buf_desc_table_(buf_desc_id_src_a, bd_src_a);
    _configure_buf_desc_table_(buf_desc_id_src_b, bd_src_b);
    _llk_unpack_configure_binary_<p_unpacr::UNP_B, p_unpacr::UNP_A>(
        static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_B_dst));

    _llk_unpack_matmul_init_<UNPACK_TRANSPOSE_FACES>(buf_desc_id_src_a, buf_desc_id_src_b, params.CT_DIM, params.RT_DIM, params.KT_DIM);

    for (std::uint32_t j = 0; j < params.KT_DIM; j++)
    {
        _llk_unpack_matmul_(params.CT_DIM, params.RT_DIM, params.KT_DIM, j, j * params.CT_DIM);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false>(
        static_cast<DataFormat>(formats.math), static_cast<DataFormat>(formats.math));
    _llk_math_matmul_init_<(ckernel::MathFidelity)MATH_FIDELITY, ENABLE_DIRECT_INDEXING, ENABLE_2X_FORMAT>(params.CT_DIM, params.RT_DIM);

    for (std::uint32_t i = 0; i < params.KT_DIM; i++)
    {
        _llk_math_matmul_block_(params.CT_DIM, params.RT_DIM);
    }
    _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
}

#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_srcs.h"
#include "params.h"
#include "sfpu/ckernel_sfpu_exp.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t num_tiles = params.TILE_CNT;

    const bool PARAM_SRCS_32BIT_MODE                = _is_srcs_32bit_mode_(static_cast<DataFormat>(formats.unpack_S_dst));
    constexpr std::uint32_t PARAM_SRCS_XDIM         = srcs_dims::XDIM;
    constexpr std::uint32_t PARAM_SRCS_ZDIM         = srcs_dims::ZDIM;
    const std::uint32_t PARAM_SRCS_YDIM             = srcs_dims::ydim(PARAM_SRCS_32BIT_MODE);
    const std::uint32_t PARAM_SRCS_SLICE_COUNT      = srcs_dims::slice_count(PARAM_SRCS_32BIT_MODE);
    constexpr std::uint32_t PARAM_SRCS_INSTRN_COUNT = 1;

    constexpr std::uint32_t buf_desc_id_unpack = 0;
    constexpr std::uint32_t buf_desc_id_pack   = 8;

    buffer_descriptor_u bd_unpack = {0};
    buffer_descriptor_u bd_pack   = {0};

    bd_unpack.f.l1_addr_16B = L1_ADDRESS(params.buffer_S[0]);
    bd_unpack.f.format      = static_cast<std::uint8_t>(formats.unpack_S_src);
    bd_unpack.f.x_dim       = PARAM_SRCS_XDIM;
    bd_unpack.f.y_dim       = PARAM_SRCS_YDIM;
    bd_unpack.f.z_dim       = PARAM_SRCS_ZDIM;
    _configure_buf_desc_table_(buf_desc_id_unpack, bd_unpack);
    _llk_unpack_configure_unary_<p_unpacr::UNP_S>(static_cast<DataFormat>(formats.unpack_S_dst));

    bd_pack.f.l1_addr_16B = L1_ADDRESS(params.buffer_Res[0]);
    bd_pack.f.format      = static_cast<std::uint8_t>(formats.pack_S_dst);
    bd_pack.f.x_dim       = PARAM_SRCS_XDIM;
    bd_pack.f.y_dim       = PARAM_SRCS_YDIM;
    bd_pack.f.z_dim       = PARAM_SRCS_ZDIM;
    _configure_buf_desc_table_(buf_desc_id_pack, bd_pack);
    _llk_pack_hw_configure_<p_pacr::PACK1, false>(static_cast<DataFormat>(formats.pack_S_src), ckernel::ReluConfig::none());

    _llk_unpack_srcs_config_for_tile_<PARAM_SRCS_INSTRN_COUNT>(PARAM_SRCS_32BIT_MODE);
    _llk_pack_srcs_config_for_tile_<PARAM_SRCS_INSTRN_COUNT>(PARAM_SRCS_32BIT_MODE);
    _llk_math_eltwise_sfpu_init_();

    const int num_sfpu_iterations = PARAM_SRCS_YDIM >> 1;
    const int load_base_addr      = ckernel::math::SFPU_SRCS_BASE_ADDR;

    // The SFPU load reads what UNP_S wrote (unpack_S_dst); the folded store writes what PACK1
    // will read (pack_S_src).
    const std::uint32_t load_sfpmem  = _sfpu_sfpmem_type_(static_cast<DataFormat>(formats.unpack_S_dst));
    const std::uint32_t store_sfpmem = _sfpu_sfpmem_type_(static_cast<DataFormat>(formats.pack_S_src));

    // One-time setup of the exp LOADMACRO replay. The store offset (2 * YDIM = slice size) must
    // be a compile-time constant, so branch on the runtime 32-bit mode into constexpr variants.
    if (PARAM_SRCS_32BIT_MODE)
    {
        _exp_init_loadmacro_<2 * srcs_dims::ydim(true)>(load_base_addr, num_sfpu_iterations, load_sfpmem, store_sfpmem);
    }
    else
    {
        _exp_init_loadmacro_<2 * srcs_dims::ydim(false)>(load_base_addr, num_sfpu_iterations, load_sfpmem, store_sfpmem);
    }
    // One MOP run issues one REPLAY per SrcS slice of a tile. The `done` bit on the final
    // LOADMACRO of each replay swaps the SrcS banks and resets the dvalids in hardware, so the
    // SFPU is paced purely by the dvalid handshake with the unpacker and packer.
    ckernel_template mop(PARAM_SRCS_SLICE_COUNT, 1, _exp_loadmacro_op_(num_sfpu_iterations));
    mop.program(instrn_buffer);

    // Full TRISC3 path: UNP_S -> SFPU exp (self-contained SFPLOADMACRO replay) -> PACK1.
    // Pack is kicked before the MOP so PACK1 is already waiting on the output dvalids.
    for (std::uint32_t i = 0; i < num_tiles; ++i)
    {
        _llk_unpack_srcs_<PARAM_SRCS_INSTRN_COUNT>(buf_desc_id_unpack, i * PARAM_SRCS_SLICE_COUNT);
        _llk_pack_srcs_<PARAM_SRCS_INSTRN_COUNT>(buf_desc_id_pack, i * PARAM_SRCS_SLICE_COUNT);
        ckernel_template::run(instrn_buffer);
    }

    wait_mop_idle();
    wait_pack_idle();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_matmul.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    buffer_descriptor_u bd_dst = {0};
    bd_dst.f.l1_addr_16B       = L1_ADDRESS(params.buffer_C[0]);
    bd_dst.f.lmt_addr_16B      = 0;
    bd_dst.f.format            = static_cast<std::uint8_t>(formats.pack_dst);
    bd_dst.f.x_dim             = FACE_C_DIM;
    bd_dst.f.y_dim             = FACE_R_DIM;
    bd_dst.f.z_dim             = params.num_faces;

    _configure_buf_desc_table_(buf_desc_id_dst, bd_dst);
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_matmul_init_(buf_desc_id_dst, params.RT_DIM, params.CT_DIM, 1);

    _llk_pack_matmul_(0, 0);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
