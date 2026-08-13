# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Coverage for https://github.com/tenstorrent/tt-metal/issues/43050:
# generic reduce (sum/mean/max/min/std/var) now supports DRAM-sharded input/output for natural-axis
# reduces (single-H, single-W, full-HW), not just L1-sharded. The fix touched:
#   - common.cpp::build_reduce_output_tensor_spec - stop borrowing a shard grid across buffer types
#     (DRAM shard grids are bank ids, L1 shard grids are worker-core coordinates - disjoint spaces).
#   - common.cpp/.hpp::validate_reduce_sharded_buffer_types - accept DRAM alongside L1.
#   - reduce_op_device_operation.cpp - the Tensix-grid-containment checks are meaningless for a DRAM
#     bank grid, so they're gated to L1 only (DRAM's own bank-grid legality is enforced separately,
#     in tt_metal's validate_buffer_parameters).
#   - reduce_op_multi_core_h_program_factory.cpp - the L1-only width-sharded fast path
#     (use_width_sharding, which binds CBs directly to the tensor's own buffer - impossible for
#     DRAM) is gated to L1, so a DRAM WIDTH_SHARDED tensor falls through to the already-generic,
#     TensorAccessor-based branch used by every other reduce/dim combination.
# std/var dispatch through a separate WelfordReduceDeviceOperation, but share the same
# validate_reduce_sharded_buffer_types (op_name="Std/Var reduction") and were already fully generic
# (no L1-only fast path of their own).
#
# Explicitly out of scope (see plan): non-natural-dim/multi-axis reduce, which internally calls
# ttnn::transpose (sum/mean/max/min) or ttnn::permute (std/var) mid-reduction - neither op's own
# DRAM-sharded handling has been audited. And DRAM BLOCK_SHARDED, which is physically impossible for
# any op: DRAM banks are a 1D, row-y=0 address space (tt_metal/impl/buffers/buffer.cpp), incompatible
# with BLOCK_SHARDED's inherently 2D shard grid - confirmed below via plain tensor construction.

import pytest

pytestmark = pytest.mark.use_module_device

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics

TEST_PADDING_VALUE = -42

REDUCE_OPS = {
    "sum": (ttnn.sum, lambda t, dim, keepdim: torch.sum(t, dim=dim, keepdim=keepdim)),
    "mean": (ttnn.mean, lambda t, dim, keepdim: torch.mean(t, dim=dim, keepdim=keepdim)),
    # amax/amin (not max/min) since they accept a tuple dim, needed for the full-HW reduce tests.
    "max": (ttnn.max, lambda t, dim, keepdim: torch.amax(t, dim=dim, keepdim=keepdim)),
    "min": (ttnn.min, lambda t, dim, keepdim: torch.amin(t, dim=dim, keepdim=keepdim)),
    # correction defaults to True (Bessel's correction) on both ttnn.std/var and torch.std/var.
    "std": (ttnn.std, lambda t, dim, keepdim: torch.std(t, dim=dim, keepdim=keepdim)),
    "var": (ttnn.var, lambda t, dim, keepdim: torch.var(t, dim=dim, keepdim=keepdim)),
}


_L1_SHARD_CORE_GRIDS = {
    ttnn.ShardStrategy.HEIGHT: ttnn.CoreGrid(x=1, y=4),
    ttnn.ShardStrategy.WIDTH: ttnn.CoreGrid(x=5, y=1),  # 160 / 5 = 32, tile-aligned
    ttnn.ShardStrategy.BLOCK: ttnn.CoreGrid(x=5, y=8),
}


@pytest.mark.parametrize("op_name", list(REDUCE_OPS.keys()))
@pytest.mark.parametrize(
    "shard_strategy", [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK]
)
def test_reduce_l1_sharded(device, op_name, shard_strategy):
    """L1-sharded input/output already works for generic reduce; guard against regressions."""
    torch.manual_seed(0)
    ttnn_op, torch_op = REDUCE_OPS[op_name]

    shape = (1, 1024, 160)
    core_grid = _L1_SHARD_CORE_GRIDS[shard_strategy]

    torch_input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_op(torch_input_tensor, dim=-1, keepdim=True)

    sharded_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=core_grid,
        strategy=shard_strategy,
        use_height_and_width_as_shard_shape=False,
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_config,
    )

    output_tensor = ttnn_op(input_tensor, dim=-1, keepdim=True, memory_config=sharded_config)

    output_mem_config = output_tensor.memory_config()
    assert output_mem_config.buffer_type == ttnn.BufferType.L1
    assert output_mem_config.is_sharded()

    output_tensor = ttnn.to_torch(output_tensor)
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.05,
        atol=0.05,
        frobenius_threshold=0.01,
    )


def test_reduce_dram_block_sharded_construction_is_impossible(device, expect_error):
    """DRAM BLOCK_SHARDED can never be constructed, for any op: DRAM banks are a 1D, row-y=0
    address space (tt_metal/impl/buffers/buffer.cpp's validate_buffer_parameters), incompatible
    with BLOCK_SHARDED's inherently 2D shard grid. Confirmed via plain tensor construction -
    this is not a reduce-specific gap, so there is no positive DRAM BLOCK_SHARDED case to test."""
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    shard_spec = ttnn.ShardSpec(shard_grid, (32, 32), ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    torch_input_tensor = torch.randn((1, 1, 128, 128), dtype=torch.bfloat16)
    with expect_error(RuntimeError, "DRAM banks are 1D"):
        ttnn.from_torch(
            torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
        )


# (tensor_shape, shard_shape, shard_grid) satisfying each layout's physical shard-geometry
# constraints (e.g. WIDTH_SHARDED requires shard height == full physical height) for a DRAM shard
# grid (bank ids, single row y=0). BLOCK_SHARDED is absent: see
# test_reduce_dram_block_sharded_construction_is_impossible above.
_DRAM_SHARD_GEOMETRY = {
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: {
        "tensor_shape": (1, 1, 416, 32),
        "shard_shape": (128, 32),
        "shard_grid": ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
    },
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: {
        "tensor_shape": (1, 1, 32, 128),
        "shard_shape": (32, 32),
        "shard_grid": ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
    },
}


def _dram_sharded_memory_config(shard_layout, shard_grid=None, shard_shape=None):
    geometry = _DRAM_SHARD_GEOMETRY[shard_layout]
    shard_spec = ttnn.ShardSpec(
        shard_grid or geometry["shard_grid"],
        shard_shape or geometry["shard_shape"],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(shard_layout, ttnn.BufferType.DRAM, shard_spec)


def _dram_sharded_input(device, shard_layout, dtype=ttnn.bfloat16, torch_input_tensor=None):
    tensor_shape = _DRAM_SHARD_GEOMETRY[shard_layout]["tensor_shape"]
    if torch_input_tensor is None:
        torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    interleaved_input = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    dram_sharded_input = ttnn.interleaved_to_sharded(interleaved_input, _dram_sharded_memory_config(shard_layout))
    return torch_input_tensor, dram_sharded_input


@pytest.mark.parametrize("op_name", list(REDUCE_OPS.keys()))
@pytest.mark.parametrize("shard_layout", list(_DRAM_SHARD_GEOMETRY.keys()))
@pytest.mark.parametrize("dim", [-1, -2])
def test_reduce_dram_sharded(device, op_name, shard_layout, dim):
    """
    DRAM-sharded input+output, covering both reduce-dim program factories, for every op:
    - dim=-1 (W-reduce) was already fully generic (TensorAccessor-based) pre-fix, so this is mainly
      a regression check for the buffer-type relaxation + TensorSpec-grid-construction fix.
    - dim=-2 (H-reduce) on a WIDTH_SHARDED tensor is the one path that used to hard-fail: H-reduce's
      L1-only fast path is now gated off for DRAM, falling through to the same generic branch.
    Includes reducing along the tensor's own sharded dimension (e.g. WIDTH_SHARDED + dim=-1): this
    collapses the shard count to one physical tile-of-shards, which is a pre-existing TensorSpec
    convention (also true for L1) rather than anything this fix touches, but is included since it's
    numerically well-defined and already covered by the geometry below.
    """
    torch.manual_seed(0)
    ttnn_op, torch_op = REDUCE_OPS[op_name]

    torch_input_tensor, dram_sharded_input = _dram_sharded_input(device, shard_layout)
    torch_output_tensor = torch_op(torch_input_tensor, dim, True)

    dram_sharded_config = _dram_sharded_memory_config(shard_layout)
    output_tensor = ttnn_op(dram_sharded_input, dim=dim, keepdim=True, memory_config=dram_sharded_config)

    output_mem_config = output_tensor.memory_config()
    assert output_mem_config.buffer_type == ttnn.BufferType.DRAM
    assert output_mem_config.is_sharded()

    output_tensor = ttnn.to_torch(output_tensor)
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.05,
        atol=0.05,
        frobenius_threshold=0.01,
    )


@pytest.mark.parametrize("op_name", list(REDUCE_OPS.keys()))
@pytest.mark.parametrize("shard_layout", list(_DRAM_SHARD_GEOMETRY.keys()))
def test_reduce_dram_sharded_full_hw_reduce(device, op_name, shard_layout):
    """
    Full-HW reduce (dim=(-2,-1)) from a DRAM-sharded input to a plain DRAM-interleaved output.
    A collapsed 1x1(-tile) result can't meaningfully stay sharded across multiple cores, so the
    output is interleaved here (a realistic "reduce a huge sharded tensor to a scalar" request);
    the point is exercising the real DRAM-sharded input through the multi-core-HW dispatch, which
    for sum/mean/max/min decomposes host-side into an internal W-reduce then H-reduce
    (reduce_op.cpp) - each internal step reads the actual sharded/intermediate tensor, so this is a
    distinct code path from the single-axis cases in test_reduce_dram_sharded above. std/var use a
    single unified Welford call for the HW case instead of this two-step decomposition.
    """
    torch.manual_seed(0)
    ttnn_op, torch_op = REDUCE_OPS[op_name]

    torch_input_tensor, dram_sharded_input = _dram_sharded_input(device, shard_layout)
    torch_output_tensor = torch_op(torch_input_tensor, (-2, -1), True)

    output_tensor = ttnn_op(dram_sharded_input, dim=(-2, -1), keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    output_mem_config = output_tensor.memory_config()
    assert output_mem_config.buffer_type == ttnn.BufferType.DRAM
    assert not output_mem_config.is_sharded()

    output_tensor = ttnn.to_torch(output_tensor)
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.05,
        atol=0.05,
        frobenius_threshold=0.01,
    )


@pytest.mark.parametrize("op_name", ["sum", "max"])
def test_reduce_dram_sharded_full_bank_width_h_reduce(device, op_name):
    """Regression test for the device_grid.contains(shard_grid)/program_grid.contains(shard_grid)
    checks that used to be meaningless for a DRAM bank grid (reduce_op_device_operation.cpp): a
    WIDTH_SHARDED grid spanning every DRAM bank is wider (12 on Wormhole) than the Tensix compute
    grid (8x8), so pre-fix this could fail even after the buffer-type relaxation alone."""
    torch.manual_seed(0)
    ttnn_op, torch_op = REDUCE_OPS[op_name]

    num_banks = device.dram_grid_size().x
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})
    tensor_shape = (1, 1, 32, 32 * num_banks)
    shard_shape = (32, 32)

    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_op(torch_input_tensor, -2, True)

    interleaved_input = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    dram_sharded_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )
    dram_sharded_input = ttnn.interleaved_to_sharded(interleaved_input, dram_sharded_config)

    output_tensor = ttnn_op(dram_sharded_input, dim=-2, keepdim=True, memory_config=dram_sharded_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.05,
        atol=0.05,
        frobenius_threshold=0.01,
    )


@pytest.mark.parametrize("op_name", ["sum", "max"])
@pytest.mark.parametrize("dram_side", ["input", "output"])
def test_reduce_h_width_sharded_mixed_l1_dram(device, op_name, dram_side):
    """One side DRAM-WIDTH_SHARDED, the other L1-WIDTH_SHARDED, on dim=-2 (H-reduce): the
    use_width_sharding gate must key off buffer type per-side, not just layout, so this is the
    case most likely to catch a wrong '&&' in that condition."""
    torch.manual_seed(0)
    ttnn_op, torch_op = REDUCE_OPS[op_name]

    tensor_shape = _DRAM_SHARD_GEOMETRY[ttnn.TensorMemoryLayout.WIDTH_SHARDED]["tensor_shape"]
    shard_shape = _DRAM_SHARD_GEOMETRY[ttnn.TensorMemoryLayout.WIDTH_SHARDED]["shard_shape"]
    shard_grid = _DRAM_SHARD_GEOMETRY[ttnn.TensorMemoryLayout.WIDTH_SHARDED]["shard_grid"]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    dram_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)
    l1_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)

    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_op(torch_input_tensor, -2, True)

    interleaved_input = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_config = dram_config if dram_side == "input" else l1_config
    output_config = l1_config if dram_side == "input" else dram_config
    input_tensor = ttnn.interleaved_to_sharded(interleaved_input, input_config)

    output_tensor = ttnn_op(input_tensor, dim=-2, keepdim=True, memory_config=output_config)

    output_mem_config = output_tensor.memory_config()
    assert output_mem_config.buffer_type == (ttnn.BufferType.L1 if dram_side == "input" else ttnn.BufferType.DRAM)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.05,
        atol=0.05,
        frobenius_threshold=0.01,
    )


@pytest.mark.parametrize("op_name", list(REDUCE_OPS.keys()))
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
def test_reduce_dram_sharded_dtypes(device, op_name, dtype):
    """DRAM-sharded WIDTH_SHARDED + dim=-2 (the fast-path-fallback case) across every dtype
    generic reduce claims to support, at one fixed geometry."""
    torch.manual_seed(0)
    ttnn_op, torch_op = REDUCE_OPS[op_name]
    shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED

    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    tensor_shape = _DRAM_SHARD_GEOMETRY[shard_layout]["tensor_shape"]
    torch_input_tensor = torch.randn(tensor_shape, dtype=torch_dtype)
    torch_output_tensor = torch_op(torch_input_tensor, -2, True)

    _, dram_sharded_input = _dram_sharded_input(
        device, shard_layout, dtype=dtype, torch_input_tensor=torch_input_tensor
    )
    dram_sharded_config = _dram_sharded_memory_config(shard_layout)
    output_tensor = ttnn_op(dram_sharded_input, dim=-2, keepdim=True, memory_config=dram_sharded_config)
    output_tensor = ttnn.to_torch(output_tensor)

    # bfloat8_b's block-float quantization pushes near-zero sums to a large relative error even
    # though the absolute error and PCC/Frobenius norm are all comfortably within tolerance.
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.1,
        atol=0.2,
        frobenius_threshold=0.02,
    )


def test_reduce_dram_sharded_non_natural_dim(device, expect_error):
    """Non-natural-dim reduce (dim=1, not H/W) internally calls ttnn::transpose (sum/mean/max/min)
    or ttnn::permute (std/var) to move the reduced axis into H/W position - neither op's own
    DRAM-sharded handling has been audited, so this is out of this fix's scope. Verified behavior
    is currently inconsistent across ops: ttnn.sum succeeds on this exact input, but
    mean/max/min/std/var all hit the same shard-grid-fit error while building the transposed
    intermediate's TensorSpec. Pinned here via ttnn.mean so a future transpose/permute fix has a
    test to flip; this is not claiming every op behaves identically today.
    """
    torch.manual_seed(0)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, (416, 32), ttnn.ShardOrientation.ROW_MAJOR)  # one shard per (n, c) slice
    dram_sharded_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    torch_input_tensor = torch.randn(1, 4, 416, 32, dtype=torch.bfloat16)
    interleaved_input = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    dram_sharded_input = ttnn.interleaved_to_sharded(interleaved_input, dram_sharded_config)

    with expect_error(RuntimeError, "Number of shards along height"):
        ttnn.mean(dram_sharded_input, dim=1, keepdim=True)


def test_reduce_h_width_sharded_l1_and_dram_use_distinct_programs(device):
    """L1-WIDTH_SHARDED and DRAM-WIDTH_SHARDED H-reduce must compile to different cached programs
    (the fast path vs. the generic fallback), proving the buffer-type gate actually participates
    in the program-cache key rather than being erased by it."""
    shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    tensor_shape = _DRAM_SHARD_GEOMETRY[shard_layout]["tensor_shape"]
    shard_shape = _DRAM_SHARD_GEOMETRY[shard_layout]["shard_shape"]
    shard_grid = _DRAM_SHARD_GEOMETRY[shard_layout]["shard_grid"]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    l1_config = ttnn.MemoryConfig(shard_layout, ttnn.BufferType.L1, shard_spec)
    dram_config = ttnn.MemoryConfig(shard_layout, ttnn.BufferType.DRAM, shard_spec)

    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    interleaved_input = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Start from a clean cache regardless of what earlier tests in this session already compiled.
    device.enable_program_cache()
    device.clear_program_cache()
    l1_input = ttnn.interleaved_to_sharded(interleaved_input, l1_config)
    ttnn.sum(l1_input, dim=-2, keepdim=True, memory_config=l1_config)
    entries_after_l1 = device.num_program_cache_entries()

    dram_input = ttnn.interleaved_to_sharded(interleaved_input, dram_config)
    ttnn.sum(dram_input, dim=-2, keepdim=True, memory_config=dram_config)
    entries_after_dram = device.num_program_cache_entries()

    assert entries_after_dram > entries_after_l1, "expected a new program cache entry for the DRAM-sharded case"


def test_reduce_dram_sharded_requires_explicit_output_shard_spec_across_buffer_types(device, expect_error):
    """Regression test for the build_reduce_output_tensor_spec fix: a sharded output config with no
    shard_spec of its own may only fall back to the input's shard grid when both share a buffer
    type. A DRAM-sharded input paired with an L1-sharded-but-spec-less output must be rejected
    rather than silently building an invalid TensorSpec from the wrong coordinate space."""
    shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    torch_input_tensor, dram_sharded_input = _dram_sharded_input(device, shard_layout)
    output_config_no_spec = ttnn.MemoryConfig(shard_layout, ttnn.BufferType.L1)

    with expect_error(RuntimeError, "requires an explicit shard_spec"):
        ttnn.sum(dram_sharded_input, dim=-1, keepdim=True, memory_config=output_config_no_spec)
