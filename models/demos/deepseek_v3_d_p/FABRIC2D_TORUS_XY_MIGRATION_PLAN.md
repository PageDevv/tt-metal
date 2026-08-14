# Existing prefill tests: Fabric2D and TorusXY migration plan

Status: Stage 9.4 `LB-F2D-main` cleanup in progress; 4x1/1x4 QuietBox and 8x1 LoudBox proxies are
restored on their original shapes and migrated to TorusY/TorusX. All four tracked local hang reports
are closed on LoudBox: H1 is not reproducible, and the H2-H4 unchanged reproductions now complete.
H2's migrated 8x1 TorusY performance bound has been recalibrated from two stable post-rebase runs and
passes locally. Galaxy CI remains required for the rebased four-commit stack.

Prepared: 2026-08-13

## Scope and goal

This work migrates the existing DeepSeek-family prefill test inventory used by Kimi K2.6/K2.7,
already-present Kimi K3 cases, and GLM-5.2. It does not develop any model.

The required end state is:

1. Existing communicating local tests use `ttnn.FabricConfig.FABRIC_2D` on non-ring local meshes.
   Existing Nx1 proxies (4x1 QuietBox and 8x1 LoudBox) use `FABRIC_2D_TORUS_Y`; existing 1xN proxies
   (1x4 QuietBox and any retained 1x8 row) use `FABRIC_2D_TORUS_X`.
2. Existing production-shaped 8x4 Galaxy tests use `ttnn.FabricConfig.FABRIC_2D_TORUS_XY`.
3. No scoped test uses `FABRIC_1D` or `FABRIC_1D_RING`.
4. Tests select FabricConfig, not an independent topology flavor. Collectives derive per-axis topology
   from the active FabricConfig: plain Fabric2D gives `(Linear, Linear)`, TorusX gives `(Linear, Ring)`,
   TorusY gives `(Ring, Linear)`, and TorusXY gives `(Ring, Ring)`.
5. Existing topology siblings that become redundant are removed. Unique semantic coverage is retained
   and migrated; a test is not deleted merely because it is op-level.
6. Production release coverage moves toward existing higher-level Galaxy tests. This milestone does not
   add test files, test functions, workload configurations, mesh shapes, model configurations, or CI jobs;
   a Fabric2D parameter may replace the corresponding existing Fabric1d row one-for-one.

## CI baselines used for the inventory

- [Blaze prefill run 31461379870](https://github.com/tenstorrent/tt-metal/actions/runs/31461379870): use the existing Kimi/GLM/DeepSeek MLA, MoE, prefill-block, chunked-transformer, cache-table,
  runner, and performance jobs as the Galaxy inventory. Reassign existing production rows to TorusXY;
  do not add a K3 integration job or any other configuration.
- [Blackhole run 31472546803](https://github.com/tenstorrent/tt-metal/actions/runs/31472546803): the local milestone is based on the existing
  `bh_lb_DeepSeek_PREFILL_OP_TESTS`, `bh_lb_DeepSeek_PREFILL`, `bh_lb_DeepSeek_PREFILL_PERF`, and
  `bh_lb_DeepSeek_DSA` selections. Preserve their semantic coverage while removing Fabric1d topology
  variants and redundant op-level production matrices.

## Explicit exclusions

- No Kimi K3 model development. In particular, do not implement or modify K3 MLA/KDA/AttnRes/MoE,
  blocks, transformers, caches, runners, adapters, manifests, or checkpoint support.
- No K2.x, GLM, or DeepSeek model/runtime development except a separately approved minimal compatibility
  fix proven necessary to run an existing migrated test. Such a fix is not part of this milestone by default.
- No new tests or broader parameter matrices. A migration is one-for-one or reduces the existing matrix.
- No reduced surrogate for an existing 8x4 production test. Preserve its 8x4 shape for Galaxy CI.
- No local TorusXY claim. This eight-chip LoudBox has no certified XY-wrap descriptor.
- No use of `scripts/run_safe_pytest.sh` in CI scheduling; it is for local launches only.
- Do not trigger or schedule CI as part of the current local milestone. Galaxy CI execution is a later,
  separately authorized stage; current work may only prepare selectors and collect them locally.

## Topology policy

FabricConfig decides topology. Test parametrization may retain mesh shape, link count, trace mode, dtype,
and existing model/workload choices, but must not cross-product a separate `Linear`/`Ring` parameter.

| Active fabric | Local/production role | Derived SP topology | Derived TP topology |
|---|---|---:|---:|
| `FABRIC_2D` | local LoudBox/QuietBox non-ring diagnostics | Linear | Linear |
| `FABRIC_2D_TORUS_X` | existing 1x4/1x8 local ring proxies | Linear | Ring |
| `FABRIC_2D_TORUS_Y` | existing 4x1 QuietBox and 8x1 LoudBox ring proxies | Ring | Linear |
| `FABRIC_2D_TORUS_XY` | production-shaped 8x4 Galaxy | Ring | Ring |

If an operation API still requires a topology argument, the test obtains it from
`models.demos.deepseek_v3_d_p.tt.tt_ccl.per_axis_topology()` after device open. It must not carry a
separate topology parameter. Fabric1d/Fabric1d-ring siblings are migrated in place. Redundant independent
topology siblings are removed only when the same existing workload has a Fabric2D/TorusX/TorusY/TorusXY owner;
an existing single-axis proxy is not removed merely because its migrated Torus execution currently hangs.

## Migration inventory

### Shared fixtures and profiles

| Existing area | Migration |
|---|---|
| `tests/fabric_profiles.py` | Centralize Fabric2D, TorusX, TorusY, and Galaxy TorusXY device profiles. Do not encode a second collective-topology parameter. |
| `tests/conftest.py` | Keep existing 2D local Fabric2D rows; map 4x1/8x1 to TorusY and 1x4/1x8 to TorusX; retain production 8x4 TorusXY and the existing explicitly described 4x4 TorusX/TorusY/TorusXY subtorus diagnostics. Remove Fabric1d and duplicate unwrapped 8x4 siblings. Preserve original shapes. |
| `tests/pcc/mesh_configs.py` | Preserve the existing 2x1 one-/two-link rows under Fabric2D. Migrate existing 4x1 and 8x1 one-/two-link linear/ring rows to TorusY; because FabricConfig determines topology, collapse each former Linear/Ring pair to one TorusY row per link count without creating a replacement shape. |

### Cache tests

The following existing cache tests migrate their communicating rows in place from Fabric1d to Fabric2D:

- `tests/cache/test_embedding_cache.py`
- `tests/cache/test_ffn_cache.py`
- `tests/cache/test_gate_cache.py`
- `tests/cache/test_lm_head_cache.py`
- `tests/cache/test_mla_cache.py`
- `tests/cache/test_moe_cache.py`
- `tests/cache/test_rms_norm_cache.py`
- `tests/cache/test_routed_expert_cache.py`
- `tests/cache/test_shared_expert_cache.py`

Retain cold/warm cache semantics and the original mesh shape. Existing K3 cache rows are only topology
migrations; no K3 cache implementation work is allowed.

### Op and diagnostic tests

| Existing test | Migration/disposition |
|---|---|
| `op_unit_tests/test_combine_subdevices.py` | Migrate the retained communicating row to Fabric2D. |
| `op_unit_tests/test_dispatch_combine_l1_small_semaphores.py` | Keep the four existing dispatch/combine × L1-default/L1-small cases on their original 4x1 shape, migrated to TorusY. Derive topology from FabricConfig. |
| `op_unit_tests/test_fp8_kv_cache_gather.py` | Fabric2D only; derive the collective topology from active FabricConfig. |
| `op_unit_tests/test_masked_bincount.py` | Keep the existing 1x1/1x2 fabric-irrelevant diagnostics with fabric disabled, preserve the original QuietBox 1x4 execution under TorusX, and migrate the existing 2x4 row to Fabric2D. |
| `op_unit_tests/test_mla_matmuls.py` | Retain the unique production-shaped 8x4 program-config diagnostic, migrate it one-for-one to TorusXY, and give it an explicit ID. |
| `op_unit_tests/test_offset_cumsum.py` | Preserve all four existing shapes: 2x1, 4x2, and 2x4 use Fabric2D; 4x1 uses TorusY. Derive topology from FabricConfig. |
| `op_unit_tests/test_prefill_combine.py` | Existing non-ring local rows use Fabric2D; preserve the existing 4x1 and 8x1 proxies one-for-one as TorusY. |
| `op_unit_tests/test_prefill_dispatch.py` | Existing local rows use Fabric2D. |
| `op_unit_tests/test_reduce.py` | Keep the existing 4x2 Fabric2D diagnostic and restore the original 4x1 row under TorusY. Retain its top-k=1 test and derive topology from FabricConfig. |
| `op_unit_tests/test_ring_joint_mla.py` | Keep the local Fabric2D diagnostics, retain the existing 32x4 scale-out rows as unwrapped Fabric2D, and migrate the existing 8x4 perf row one-for-one to TorusXY. Lack of a 32x4 TorusXY CI owner is not redundancy and does not justify removing the row. |
| `op_unit_tests/test_rope_prefill.py` | Migrate the retained row to Fabric2D. |
| `op_unit_tests/test_sub_device_load_clear_timing.py` | Migrate the retained communicating row to Fabric2D. |
| `op_unit_tests/test_ttnn_dispatch_combine.py` | Keep the file and existing workload matrix. The shared mesh matrix is Fabric2D/TorusY/TorusXY; preserve the overflow/top-4 regressions on their original 8x1 shape as TorusY and derive topology from FabricConfig. |
| `op_unit_tests/test_deepseek_prefill_rotary_embedding_indexed.py` | Retain scalar/metadata equivalence and program-cache semantics. Its existing production-shaped 8x4 row moves one-for-one to TorusXY and remains Galaxy-only. |
| `op_unit_tests/test_deepseek_prefill_update_padded_kv_cache.py` | Retain multi-chip scalar/metadata equivalence. Preserve the single-iteration 1x4 row as TorusX, 2x4 as Fabric2D, and production 8x4 as TorusXY. |
| `op_unit_tests/test_zero_padded_kv_cache.py` | Retain the existing 2x4 cross-chip/program-cache diagnostic under Fabric2D, preserve the existing SP=8 8x1 LoudBox proxy as TorusY, and migrate the original production 8x4 row to TorusXY. Do not remove either execution environment during this milestone. |
| `op_unit_tests/test_moe_padding_config.py` | Keep both existing mesh rows. Run the 12 cases addressable on 2x4 locally with Fabric2D. Preserve the two existing long-position cases, which exceed the 2x4 addressable range, and execute those through their original production-sized 8x4 row in the existing Galaxy functional job under TorusXY. |

### PCC/module tests

Existing two-dimensional rows move to Fabric2D locally; existing 1x4 rows move to TorusX, existing
4x1/8x1 rows move to TorusY, and existing 8x4 production rows use TorusXY:

- `tests/pcc/test_lm_head.py`
- `tests/pcc/test_moe_gate_prefill2d.py`
- `tests/pcc/test_moe_routing_setup.py`
- `tests/pcc/test_parallel_embedding.py`
- `tests/pcc/test_shared_expert.py`
- `tests/pcc/test_ttnn_moe.py`

Topology arguments are derived from FabricConfig. Existing model/workload/dtype/trace cases are not
expanded.

The original 1x4 LM-head, shared-expert, parallel-embedding, FFN, and RMSNorm rows are not removed or
reshaped; they migrate to TorusX. The original 4x1 routing-setup and 8x1 routing/MoE proxy rows migrate
to TorusY. Old linear/ring duplicates collapse to one row because FabricConfig determines topology.

Keep standalone `pcc/test_ffn.py` and `pcc/test_rmsnorm.py`. Collapse each former 1x4 line/ring pair
into one 1x4 TorusX row, preserving shape, TP=4, and workload parameters. Derive TP topology from
FabricConfig.

### Dense MLA, blocks, transformers, and DFlash

| Existing test | Migration/disposition |
|---|---|
| `tests/test_mla.py` | Local 2x4/2x2 rows use Fabric2D. Existing 8x4 Kimi K2.x, K3, GLM, and DeepSeek rows use TorusXY. K3 changes are test-parameter changes only. |
| `tests/test_prefill_block.py` | Preserve local Fabric2D rows and migrate existing 8x4 production rows to TorusXY; remove redundant topology siblings. |
| `tests/test_prefill_block_chunked.py` | Existing 8x4 production cases use TorusXY; do not create 2x4 surrogates. |
| `tests/test_prefill_block_loop.py` | Existing local rows use Fabric2D and production cases remain their original shape. Retain the two 4x4-subtorus-only ISLs and their 128-expert host-gate accommodation with the existing explicit 4x4 descriptors. They are unique diagnostics, not replacements for 8x4 production coverage. |
| `tests/test_prefill_transformer.py` | Existing production 8x4 rows use TorusXY and redundant topology variants are removed. |
| `tests/test_prefill_transformer_chunked.py` | Existing trace/no-trace and model rows use TorusXY at 8x4. Preserve the existing matrix size or reduce it. |
| `dflash_prefill/test_dflash.py` | Existing 8x4 row uses TorusXY; Galaxy-only. |
| `dflash_prefill/test_dflash_prefill_integration.py` | Existing 8x4 row uses TorusXY; Galaxy-only. |

### Sparse MLA / GLM-5.2

| Existing test | Migration/disposition |
|---|---|
| `sparse_mla/sparse_mla_mesh.py` | Remove topology as an independent selection input; derive it from FabricConfig. Preserve the existing single-chip diagnostic with fabric disabled, 1x4 with TorusX, 8x1 with TorusY, 2D local shapes with Fabric2D, and 8x4 with TorusXY. |
| `sparse_mla/test_sparse_mla.py` | Preserve the existing QuietBox 1x4 row as TorusX and 2x2 as Fabric2D; LoudBox 2D rows use Fabric2D; existing 8x4 Galaxy rows use TorusXY. |
| `sparse_mla/test_sparse_mla_cache.py` | Existing local cache rows use Fabric2D and the original 8x4 shared-layer row uses TorusXY. Do not change 8x4 to 2x4. |
| `sparse_mla/test_sparse_mla_ccl_perf.py` | Keep the file. Preserve the existing SP-path 8x1 LoudBox proxy as TorusY and the existing TP-path 2x4 proxy as Fabric2D, without adding scenarios. On Galaxy, retain the production 8x4 shape and move it to TorusXY. Remove the separate linear/ring parameter and derive each collective-axis topology from FabricConfig. |
| `sparse_mla/test_sparse_mla_perf.py` | Keep its existing dynamic QuietBox/LoudBox/Galaxy shapes: 1x4 uses TorusX, 2x4 uses Fabric2D, and production 8x4 uses TorusXY. Recalibrate rather than reuse a threshold from another fabric. |
| `sparse_mla/test_sparse_mla_vs_trace.py` | Existing production trace comparison uses TorusXY. |

### Socket, disaggregation, and KV-table tests

| Existing test | Migration/disposition |
|---|---|
| `tests/test_d2d_socket_sync.py` | Retain its existing local Fabric2D row and migrate its existing production 8x4 row to TorusXY. |
| `tests/test_embedding_socket.py` | Migrate the existing 8x4 production row one-for-one to TorusXY. |
| `tests/test_h2d_socket_sync.py` | Migrate the existing 8x4 production row one-for-one to TorusXY. |
| `tests/test_disaggregation.py` | Remove Fabric1d and disable fabric for address/FNID diagnostics that do not communicate. Do not create degenerate Fabric2D or a new 32x4 TorusXY profile. |
| `tests/test_kv_cache_table.py` | Preserve every original mesh shape, especially the 8x4 rows. Local rows use Fabric2D and production 8x4 rows use TorusXY. |

### Performance wrappers

| Existing test | Migration/disposition |
|---|---|
| `perf/test_mla_perf.py` | Keep the existing local 2x4 proxy wrapper; two exact Fabric2D reruns completed, so the earlier all-gather timeout is historical rather than an active hang. Retain the production 8x4 worker and migrate it to TorusXY. Historical values are migration starting points and must be recalibrated from the exact active fabric before acceptance. |
| `perf/test_moe_perf.py` | Keep both existing LoudBox proxy legs: 8x1 migrates to TorusY and 2x4 migrates to Fabric2D. Preserve their existing approximation utility and schedule. Migrate existing 8x4 workers to TorusXY and execute them rather than green-by-skip. |
| `perf/test_prefill_block_perf.py` | Keep the measured local 2x4 Fabric2D two-link row (`38_638_478`), production 8x4 TorusXY rows (`18_157_603`, `60_634_662`), and the existing 4x4 subtorus rows with their explicit carve descriptors and existing skip/calibration state. Remove redundant full-8x4 TorusX/Y, Fabric1d, and unwrapped 8x4 workers. |
| `perf/test_prefill_dispatch_combine.py` and `perf/test_dispatch_combine_perf.py` | Keep both files and their existing DeepSeek/Kimi/GLM captured workloads on the original LoudBox 8x1 proxy, migrated to TorusY. Collapse only the former independent linear/ring topology siblings and derive topology from FabricConfig. Keep captured-column slicing and the single dispatch group; do not replace the proxy with a new 8x4 workload. The retained measurements are migration starting points until TorusY can execute locally. |

### Explicit consolidation and ownership map

| Existing row/file | Disposition | Retained owner |
|---|---|---|
| 4x1 `test_prefill_combine.py` row | Not removed; migrate one-for-one to TorusY. | Same existing row. |
| 4x1 `test_reduce.py` row and its single-expert function | Not removed; migrate one-for-one to TorusY. | Same existing row and function, alongside the 4x2 Fabric2D cases. |
| `test_deepseek_v3_mla_perf_loudbox` wrapper | Not removed. Its 2x4 Fabric2D all-gather completes in repeated exact safe-wrapper runs; retain it as local coverage. | Existing local wrapper plus `test_deepseek_v3_mla_perf_galaxy` on TorusXY. |
| 8x1/local approximation helpers in `utils/perf_utils.py` | Not removed. They are existing proxy infrastructure and remain referenced by the restored local wrappers. | Existing 8x1 TorusY and 2x4 Fabric2D proxy measurements plus direct production 8x4 TorusXY workers. |
| 1x4 line/ring siblings in `pcc/test_ffn.py` and `pcc/test_rmsnorm.py` | Topology must not remain an independent test axis. | One 1x4 TorusX row per file preserves the existing workload and shape. |
| 8x1 `test_ds_moe` / `test_kimi_moe` parameter rows | Not removed. The prior Fabric1d SP proxies migrate one-for-one to TorusY. | The same 8x1 rows; production ownership also remains 8x4 TorusXY. |
| Sparse Galaxy 8x2 diagnostics | Not removed. The 8x2 sub-plane cannot carry the full TP wrap edge, so it remains unwrapped Fabric2D while the complete production 8x4 view uses TorusXY. | Same existing 8x2 diagnostic plus existing local proxies and 8x4 TorusXY production row. |
| 32x4 `test_ring_joint_mla.py` rows | Not removed. There is no certified 32x4 TorusXY descriptor, so these scale-out diagnostics remain unwrapped Fabric2D. | Same existing 32x4 workload, plus 8x4 TorusXY production perf ownership. |
| Dispatch/combine 8x1 baseline assertions | Not removed. The separate linear/ring threshold axis is redundant because TorusY determines Ring. | Both files and all existing model/layer/column scenarios remain on the original 8x1 proxy shape under TorusY; exact-fabric recalibration remains blocked on local bring-up. |
| K3 realtime perf test | Existing numeric band was measured on unwrapped Fabric2D, not TorusXY. | Preserve the existing job and test one-for-one and migrate its selector to TorusXY. It executes in its certified Galaxy job; no K3 case or model work is added. |
| Degenerate 2x1 rows | Not removed when they own an existing workload. They cannot form a useful ring, so migrate them one-for-one to Fabric2D. | The same existing row and workload. |
| Fabric1d-only 1x4/4x1 PCC rows | Not removed; migrate one-for-one to TorusX/TorusY. | Same existing rows, with independent topology siblings collapsed. |
| 8x4 unwrapped Fabric2D/TorusY mesh-config siblings | Production-shaped rows must use TorusXY; fabric/topology siblings are redundant, but link count is not. | One TorusXY 8x4 row for each existing one-/two-link count; local 4x2 Fabric2D one-/two-link diagnostics are also preserved. |
| Sparse single-chip/1x1 shape entries | Fabric is irrelevant on one chip and these entries cannot participate in the Fabric2D migration. | Host/reference semantics and retained 2x2/2x4/4x2 Fabric2D sparse tests. |
| 4x4 TorusX/TorusY/TorusXY block parameter siblings | Not removed. They use a distinct 16-chip shape and explicit experimental descriptors, so they are not redundant with 8x4 production TorusXY. | Same existing 4x4 parameter rows and descriptor-driven local perf wrappers; full production ownership remains 8x4 TorusXY. |
| `isl_2k56` / `isl_12k8` rows and the 128-expert host-gate path in `test_prefill_block_loop.py` | Not removed. These are unique 4x4-subtorus workloads even though they are not currently CI-owned. | Same ISLs, expert-halving branch, and host-gate behavior under the restored 4x4 rows. |
| Four DeepSeek release `non_balanced` block invocations | They remain by-design CI skips, but are unique existing rows rather than redundant topology siblings. | Preserve the four commands and migrate their selectors one-for-one from unwrapped Fabric2D to TorusXY; do not retarget them to balanced workloads. |

The unscheduled developer utility `utils/sanity_test_32x4_device.py` is not a collected test or a
production prefill schedule and remains outside this migration. Its standalone `FABRIC_1D` setting is
therefore not evidence of a retained Fabric1d test configuration.

The runtime README still contains a `linear-2` example and command whose mesh-config row no longer
exists after this migration. Updating runtime documentation is outside this test/CI migration; track
that stale example as a separate documentation follow-up rather than expanding this branch into model
or runtime changes.

### CI selectors

These existing schedules may be edited, but no job or configuration is added:

- `tests/pipeline_reorg/blackhole_e2e_tests.yaml`: select only retained local Fabric2D cases.
- `tests/pipeline_reorg/blaze_models_prefill_tests.yaml`: select existing `torus-xy-8x4` production cases and provide the certified TorusXY descriptor.
- `tests/pipeline_reorg/demo_sp_release_tests.yaml`: remove scoped Fabric1d selectors; preserve original model/shape ownership.
- `scripts/common.sh`: keep the existing production soak selection pointed at the retained
  `torus-xy-8x4` transformer row.

CI commands use direct `pytest`. `scripts/run_safe_pytest.sh` must not appear in these schedules.

## Stage 9.4 local LoudBox milestone: `LB-F2D-main`

The local milestone is complete when all of the following are true:

1. The branch diff contains test code/assets, test scheduling (including the existing production soak
   selector in `scripts/common.sh`), and this plan only; no model or runtime implementation changes.
2. Collected test files contain no executable `FABRIC_1D` or `FABRIC_1D_RING` configuration; the
   unscheduled `utils/sanity_test_32x4_device.py` developer utility is explicitly out of scope.
3. No test function, workload configuration, mesh shape, model configuration, or CI job was added. Any
   added Fabric2D parameter entry replaces an existing Fabric1d entry for the same workload and shape.
4. Existing local communicating rows collect as Fabric2D on non-ring 2D meshes, as TorusX on the original
   1x4/1x8 proxies, and as TorusY on the original 4x1/8x1 proxies. All retained rows must remain present
   even when a hardware path is an open hang.
5. Existing 8x4 TorusXY rows collect with stable, explicit IDs but are not executed locally.
6. Unique scalar/metadata, cache-format, zero-window, program-cache, and program-config diagnostics remain.
7. `./build_metal.sh --release` passes from this branch.
8. Every local pytest launch uses `scripts/run_safe_pytest.sh` without `--dev`.
9. The relevant locally supported test selection passes and closes the device cleanly. Collection-only,
   hardware-inapplicable, and known-hanging rows are reported separately rather than counted
   as passing execution evidence or deleted from the inventory.
10. Claude reviews the complete stage repeatedly; every finding is fixed and rebuilt/retested until Claude
    returns exactly `OK`.
11. No migrated performance threshold is accepted without measurement provenance for the exact active
    FabricConfig. A watchdog hang is a bring-up result, not a performance baseline and not a passing test.

### Local validation commands

Activate the requested environment first:

```bash
source /localdev/pjosipovic/tt-metal/python_env/bin/activate
```

Build exactly:

```bash
./build_metal.sh --release
```

All local pytest collection and execution uses:

```bash
scripts/run_safe_pytest.sh <pytest arguments>
```

Do not pass `--dev`. Locally validate:

- import/collection for every changed test file;
- the exact existing LoudBox selector from `blackhole_e2e_tests.yaml` with its unchanged expected count;
- retained Fabric2D cache, dispatch/combine, MLA, padding, and program-cache diagnostics;
- collection-only for each 8x4 TorusXY Galaxy selector;
- `git diff --check`, Python compilation, YAML parsing, and pre-commit checks.

If a local test hangs, preserve state and use `tt-triage` before reset or process termination. Normal long
runtime, compilation, and an intentional stop are not hangs.

Each review iteration is started with the requested CLI form:

```bash
claude --dangerously-skip-permissions -p '<stage-specific review prompt; return exactly OK only with no findings>'
```

Ten minutes without output is normal. Continue independent builds, collection, and local tests while the
review runs; do not cancel a healthy review merely because it is quiet.

### Current local cleanup evidence

- A base-to-branch shape-preservation audit keeps every existing literal 4x1, 1x4, and 8x1 test site on
  the same mesh shape. No existing literal 1x8 test row was found, so this migration does not invent one.
- The retained ring-shaped proxies map by mesh axis: 4x1/8x1 use TorusY and 1x4 uses TorusX. Existing
  non-ring 2D local rows use Fabric2D; existing production-shaped 8x4 rows use TorusXY.
- Existing proxy functions and approximation utilities remain. The 4x1 top-k=1 reduce test, 1x4
  masked-bincount test, 1x4 PCC tests, 4x1 routing tests, 8x1 MoE/dispatch tests, and sparse-MLA proxy
  tests are retained. Only redundant independent Linear/Ring siblings are collapsed.
- No test file, test function, workload configuration, mesh shape, model configuration, or CI job is
  added by this migration. No CI is launched during the local cleanup milestone.
- The exact release build `./build_metal.sh --release` passes on the cleaned migration tree and is repeated
  after any subsequent code change.
- Python compilation, YAML parsing, and `git diff --check` pass on the current cleanup tree. The latest
  safe-wrapper collection over the restored 4x1/1x4/8x1 files selected 664 of 1,743 collected tests with
  `-k '(torus-x or torus-y) and not torus-xy'` and returned `SAFE_PYTEST_RESULT: PASS`; this is collection
  evidence only.
- QuietBox 4x1/1x4 rows can be collected on this LoudBox but require a four-device QuietBox allocation for
  execution evidence. Production 8x4 TorusXY rows can be collected locally but require Galaxy CI to run.
- The restored QuietBox schedules collect 43 PCC cases and 426 op cases through their exact local
  `-k` expressions, including the existing TorusX 1x4 and TorusY 4x1 rows. The existing LoudBox MLA
  performance wrapper also remains in its original perf job and passes locally on 2x4 Fabric2D.

### Local hang inventory and resolution

These existing tests remain in place. None of the four tracked reports is an active local hang after the
runtime fixes below. Any recurrence must be run with `scripts/run_safe_pytest.sh` without `--dev`; preserve
the live stall and use `tt-triage` before the safe wrapper resets the devices.

| ID | Existing test/workload | Mesh and fabric | Observed stall and evidence | Status |
|---|---|---|---|---|
| H1 (historical) | `test_deepseek_v3_mla_perf_loudbox` | 2x4 Fabric2D | An earlier run was reported to time out in `ttnn.experimental.high_bw_all_gather`, but its triage snapshot was overwritten by a later 8x1 repro. On 2026-08-14, two exact safe-wrapper reruns passed end-to-end; all-gather measured 0.409 ms and 0.408 ms. No dispatch timeout fired and no new triage artifact was produced. A post-pass snapshot found healthy ARC heartbeats on all eight devices and passing Ethernet status. | Retain the test. Treat the old event as transient/stale device state unless it recurs with a fresh live triage snapshot; it is not an active reproducible hang. |
| H2 (resolved locally) | `test_deepseek_v3_moe_perf_loudbox` 8x1 proxy leg | 8x1 TorusY | The earlier run timed out in DeepSeek prefill dispatch and was captured by `tt-triage`. After the H3 dispatch/combine fix, the unchanged wrapper completed both its 8x1 TorusY host-profile leg and 2x4 Fabric2D device-profile leg. Three completed TorusY samples measured 15.382 ms, 15.390 ms, and 15.398 ms. The two fresh post-rebase samples differ by 0.05%, so the migrated 8x1 bound is recalibrated to their 15.394 ms mean with the existing 3% margin. The paired 2x4 Fabric2D samples measured 17.112 ms and 17.167 ms against the unchanged 17.217 ms bound. | Retain the proxy. The hang and the topology-migration performance gate are both resolved locally; validate the 8x4 TorusXY production path in Galaxy CI. |
| H3 (resolved locally) | `test_ttnn_dispatch_combine_overflow` diagnostic | 8x1 TorusY | Triage isolated the stall to the custom `DispatchDeviceOperation`: standard all-gather and offset-cumsum had completed, while dispatch senders and workers were blocked in the Fabric2D multicast handshake. A logical 8x1 group turns through the physical 2x4 LoudBox graph, so its peer set cannot be represented by one straight multicast range. The Fabric2D dispatch and combine paths now exchange init/exit semaphores with explicit per-peer hybrid-route unicasts, and the host opens every physical first-hop direction required by the logical group. The unchanged `cut_short_last` and `omit_last` nodes both pass. The existing top-4 hang regression also passes in both tile and row-major layouts with exact combine and round-trip validation. | Retain the tests and the paired host/kernel routing fix. Validate physical 8x4 TorusXY in Galaxy CI when separately authorized. |
| H4 (resolved locally) | `test_prep_dispatch_combine` with `torus-y-8x1`, predictable routing, and zero padding | 8x1 TorusY | The original run timed out in `AllGatherDeviceOperation`; triage showed multicast writer CB waits and multicast reader semaphore waits. Effective-1D Fabric2D ring/torus standard all-gather now selects the direct-neighbor unicast factory instead of the mesh multicast factory. After an exact release build, the unchanged reproducer passed with all 24 routing outputs exact. The existing offset-cumsum matrix also passed its applicable 4x2 and 2x4 Fabric2D rows; hardware-inapplicable rows skipped. | Retain the test and the narrow all-gather factory-selection fix. Galaxy TorusXY validation remains required. |

The current local hang list is empty. Reopen an item only if its unchanged reproducer times out again and
captures a fresh live snapshot. Record the exact node ID, timeout, triage artifact, reset result, and
device-health result for every recurrence. A correctness or performance assertion after device execution
completes is not a hang and is tracked separately.

## Galaxy-only validation

The following cannot be validated on this LoudBox and remain mandatory Galaxy CI gates:

- physical 8x4 XY-wrap routing and the certified Ring/Ring mesh descriptor;
- every existing K2.6/K2.7/K3/GLM-5.2 production-shaped 8x4 row;
- TorusXY `(Ring, Ring)` collective execution across both wrap edges;
- production checkpoint/cache mounts, full-size memory pressure, trace replay, and device performance;
- 32-chip and multi-host socket/disaggregation behavior;
- performance recalibration after moving from Fabric1d/Fabric2D to TorusXY.

The existing Galaxy perf schedules remain active and must execute rather than green-by-skip. Their
historical thresholds are migration starting points, not accepted TorusXY calibrations; an explicitly
authorized certified-Galaxy run must measure and update them before final acceptance. This branch does
not launch that calibration or any other CI.

Galaxy commands use direct `pytest`, a scheduler-certified allocation, and the explicit TorusXY descriptor.
Pre-existing expected-count annotations remain only on their original release commands; they are not execution
evidence. The job's pytest result remains the pass/fail authority, and skips are reported separately rather than
counted as passes in the validation evidence.

## Phasing and review gates

### Stage 1: test-only branch and inventory

- Start from current `origin/main`.
- Carry only existing test/CI changes.
- Establish one-for-one migration or explicit redundancy owner for every removed row.
- Run static audits and collection.
- Iterate Claude review to exact `OK`.

### Stage 2: local Fabric2D migration

- Migrate retained LoudBox/QuietBox communicating rows by shape: non-ring 2D meshes use Fabric2D,
  1x4/1x8 rings use TorusX, and 4x1/8x1 rings use TorusY.
- Remove Fabric1d/Fabric1d-ring topology siblings without expanding another matrix.
- Build release and run the exact local suite twice through the safe wrapper.
- Use `tt-triage` for actual hangs.
- Keep functional smoke/PCC coverage that passes; do not substitute Fabric1d baselines or add smaller
  workload configurations. The tracked local hangs are closed, and H2's stable topology-migration
  performance shift is captured by its locally validated TorusY baseline.
- Iterate Claude review to exact `OK` while overlapping review wait with independent local validation.

### Stage 3: existing production rows to TorusXY

- Reassign existing 8x4 production params and selectors to TorusXY one-for-one.
- Remove redundant unwrapped/TorusX/TorusY production siblings.
- Collect locally only; execute in Galaxy CI with direct pytest.
- Recalibrate existing perf thresholds where needed on a certified Galaxy. Do not schedule or trigger that
  work without separate authorization.
- Iterate Claude review with CI evidence until exact `OK`.

### Stage 4: move release ownership upward

- Make existing block/transformer/runner Galaxy tests the production topology gate.
- Remove an op/perf row only after the higher-level existing row has executed and is named as its owner.
- Retain unique local diagnostics even when they are op-level.
- Iterate Claude review with the removal map and CI evidence until exact `OK`.

## Acceptance checklist

- [ ] Diff is test/CI/plan and existing test-runner selector-only; no K3 or other model development.
- [ ] No new test, configuration, shape, parameter row, or job.
- [ ] No scoped test uses Fabric1d or Fabric1d-ring.
- [ ] FabricConfig is the sole topology-selection input.
- [ ] Local communicating rows use Fabric2D/TorusX/TorusY according to shape; supported LoudBox rows
      pass, and any future hang recurrence is separately documented with triage evidence.
- [ ] Existing production 8x4 rows use TorusXY and preserve their shape.
- [ ] Every removed topology sibling has a named existing replacement.
- [ ] Unique semantic diagnostics remain.
- [ ] Release build passes.
- [ ] Local pytest uses the safe wrapper without `--dev`; CI uses direct pytest.
- [ ] Actual hangs have `tt-triage` evidence.
- [ ] Each stage has iterative Claude review ending in exact `OK`.
