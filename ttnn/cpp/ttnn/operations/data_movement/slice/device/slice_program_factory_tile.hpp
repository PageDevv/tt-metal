// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim {

struct SliceTileProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

// Reader per-core [start_id, num_tiles, id_per_dim...] and writer per-core scalars [num_pages, start_id]
// are hash-excluded (work-split-derived), so patch_slice_program_addresses must re-emit them on every
// cache hit — otherwise a same-hash hit on a program built from a differently-partitioned prior dispatch
// leaves writer num_pages stale (often 0) and produces an all-zero output (issue #52651).
std::vector<tt::tt_metal::DynamicRuntimeArg> slice_tile_dynamic_args(
    const SliceParams& args,
    const SliceInputs& tensor_args,
    const Tensor& output,
    uint32_t start_offset,
    uint32_t reader_kernel_idx,
    uint32_t writer_kernel_idx);

}  // namespace ttnn::prim
