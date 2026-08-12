# Context switch: `perf_math_matmul` scaling experiment

## Goal

Convert `perf_math_matmul` into a focused output-tile scaling experiment:

- keep `KT_DIM = 1`;
- test output tile counts `8, 16, 32, 64, 128`;
- use balanced full-grid geometries `2x4, 4x4, 4x8, 8x8, 8x16`;
- test input-0 tile sizes `16x32` and `32x32`;
- include Half/Full destination synchronization;
- include destination accumulation disabled/enabled;
- include Float16/Float32 input-output combinations;
- retain LoFi, HiFi2, HiFi3, and HiFi4.

The resulting suite contains 640 pytest variants.

## Modified source files

### `metal/tt-metal/tt_metal/tt-llk/tests/python_tests/perf_math_matmul.py`

- Replaced the original broad matmul/tiny-tile sweeps with a focused generator.
- Current output grids:

  ```python
  OUTPUT_TILE_GRIDS = [(2, 4), (4, 4), (4, 8), (8, 8), (8, 16)]
  ```

- Current input-0 tile sizes:

  ```python
  IN0_TILE_DIMENSIONS = [(16, 32), (32, 32)]
  ```

- `KT_DIM` is fixed at 1.
- Stimuli counts align with the full matrix:
  - A tiles = `RT_DIM`;
  - B tiles = `CT_DIM`;
  - result tiles = `RT_DIM * CT_DIM`.
- `TILE_COUNT` remains `RT_DIM * CT_DIM * KT_DIM`, therefore it is the full output-tile count.

### `metal/tt-metal/tt_metal/tt-llk/tests/sources/math_matmul_perf.cpp`

- Added destination-capacity-aware output blocking for unpack, math, pack, mock, congestion, and isolate paths.
- Destination capacity is:

  ```cpp
  (dest_sync == ckernel::DstSync::SyncFull ? 16 : 8) /
      (is_fp32_dest_acc_en ? 2 : 1)
  ```

- Added `select_matmul_block_dimensions()`.
  - It considers exact divisors of full RT/CT.
  - It first maximizes tiles per destination block.
  - Among equally full blocks, it minimizes `block_rt + block_ct`, approximating source loads and improving operand reuse.
  - It prefers the wider CT dimension only as the final tie-break.

- Expected selected block shapes:
  - capacity 4: `2x2`;
  - capacity 8: `2x4`;
  - capacity 16: `4x4` where possible; the 8-tile grid remains `2x4`.

- Unpack uses global source indices:
  - A: `block_row * KT_DIM + j`;
  - B: `j * CT_DIM + block_col`.
- Pack maps each local destination tile back to the full row-major output index.
- End-to-end synchronization performs one wait/done handshake per destination block.
- Added capacity and exact-divisibility assertions.

## Important performance result

The original CT-first blocking policy changed from `2x4` blocks to `1x8` blocks at 32 output tiles. That increased approximate source loads per output and caused unpack isolate to regress from about 25.1 to 39.5 cycles/tile.

The balanced selector removed that regression. For this representative path:

- Float16 input/register/output;
- HiFi2;
- Half sync;
- no destination accumulation;
- 32x32 tiles;

the latest measurements were:

- 8 tiles: unpack 25.2206 cycles/tile;
- 16 tiles: 25.1454;
- 32 tiles: 25.1373;
- 64 tiles: 25.1332;
- 128 tiles: 25.1311.

Per-tile L1_TO_L1 cost is also effectively flat from 8 through 128 tiles.

Other findings from the latest non-Speed-of-Light report:

- pack is the largest isolated stage in 440/640 configurations (68.8%);
- 16x32 tiles are 1.17x faster end-to-end than 32x32 across 320/320 exact pairs;
- Float16 output is 1.22x faster end-to-end than Float32 on common support, mostly from pack;
- destination accumulation averages 0.77x end-to-end performance on common register support;
- Full sync averages 0.70x end-to-end versus Half despite improving isolated unpack/math, indicating synchronization or pipeline overhead.

## Speed-of-Light issue and fix

The first full `--speed-of-light` attempt showed many failures and was later stopped with `Ctrl+Z` and killed via:

```text
pkill -9 -f "test"
```

Therefore the shell's `Killed` message was not an OOM diagnosis.

The actual compile failure was that `BLOCK_CT_DIM` and `BLOCK_RT_DIM` were declared inside `#ifndef SPEED_OF_LIGHT`. In Speed-of-Light mode, CT/RT are generated as compile-time globals, but the block variables were absent.

The declarations were moved after the preprocessor guard so both runtime and Speed-of-Light modes use the same block policy.

## Verification completed

- Python formatting and syntax checks passed.
- IDE lint checks passed.
- C++ formatting and `git diff --check` passed.
- Geometry checks passed for capacities 4, 8, and 16.
- Focused normal-mode 128-tile compile passed.
- Focused Speed-of-Light compile passed.
- Focused 128-tile Speed-of-Light hardware run passed.
- Full Speed-of-Light run passed:

  ```text
  640 passed in 1530.63s (0:25:30)
  ```

Recommended split workflow:

```bash
pytest --compile-producer --speed-of-light -n 10 -x python_tests/perf_math_matmul.py
pytest --compile-consumer --speed-of-light -n 10 -x python_tests/perf_math_matmul.py > test_output.txt 2>&1
```

## Reports and analysis

- Performance report:

  `metal/tt-metal/tt_metal/tt-llk/perf_data/perf_math_matmul/perf_math_matmul.post.csv`

- Interactive Cursor Canvas:

  `/home/ndivnic/.cursor/projects/localdev-ndivnic/canvases/perf-math-matmul-impact.canvas.tsx`

The canvas was updated for the 640-row non-Speed-of-Light report and includes:

- bottleneck ownership;
- absolute cycles with observed min-max ranges;
- scaling relative to 8 tiles;
- exact paired effects using concrete baseline → treatment cycles and cycle deltas;
- destination accumulation, sync, tile-shape, format-path, congestion, and code-size analysis.

If the CSV has since been overwritten by the full Speed-of-Light run, redo the canvas analysis rather than mixing Speed-of-Light and normal rows.

## Current git/worktree state

Target source modifications are uncommitted:

- `tt_metal/tt-llk/tests/python_tests/perf_math_matmul.py`
- `tt_metal/tt-llk/tests/sources/math_matmul_perf.cpp`

Untracked generated/user files currently include:

- `tt_metal/tt-llk/tests/python_tests/compare_output.txt`
- `tt_metal/tt-llk/tests/test_output.txt`

Do not commit generated output files unless explicitly requested.

No git commit has been created.

## Suggested next steps

1. Inspect whether the current `.post.csv` contains only `speed_of_light=True` rows.
2. Re-run parameter-impact analysis for the Speed-of-Light report if that is the intended comparison.
3. Review the two source diffs.
4. Commit only the Python test and C++ kernel when requested.
