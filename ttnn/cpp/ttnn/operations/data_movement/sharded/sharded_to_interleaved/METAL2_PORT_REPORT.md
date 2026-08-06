# Metal 2.0 Port Report — `sharded_to_interleaved`

## Outcome

**`PORTED`** — the op's single factory (`ShardedToInterleavedProgramFactory`) is converted to
`ProgramSpecFactoryConcept` across all three of its reachable configs (TILE, TILE+convert, ROW_MAJOR),
together with all four kernel entry points it can bind. No factories remain on the legacy concept.

**One caveat the reader must not miss: this port has not been compiled or run.** The invoker reserved
building and testing, and this checkout has no build tree (a cold `./build_metal.sh --build-tests` would
be required). So the recipe's Build and Run-tests verification steps were **not** executed; what *was*
executed is the anti-pattern self-audit (all items below) plus a line-by-line legacy-vs-ported diff of
every argument, binding, and hardware-config value. Treat the first build + test run as the remaining
verification gate. Suggested commands are in [Test commands](#test-commands).

## Provenance

- **Recipe docs (this port):** *cannot be pinned.*
  `git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`
  printed nothing — the recipe was supplied to this port as a standalone file
  (`/localdev/edwinlee/metal2_port.md`), not from a tracked doc-branch checkout, and
  `docs/.../metal_2.0/` does not exist in this tree. See the Friction entry
  [Shared reference docs were absent](#gaps).
- **Audit docs (inherited):** `4386dc456a1 2026-07-29 docs(metal_2.0): require an explicit opt_level when porting compute kernels`
- **Code baseline:** `origin/main` @ `2b7bf3396eb` — identical to the audit's recorded baseline, so no
  rebase was needed and every gate the audit cleared applies to exactly the code that was ported.

## TTNN ProgramFactory

- **Concept realized:** `ProgramSpecFactoryConcept`, as the audit chose.
  `create_descriptor(...) -> tt::tt_metal::ProgramDescriptor` became
  `create_program_artifacts(...) -> ttnn::device_operation::ProgramArtifacts`. Same three parameters,
  unchanged. The header's `<tt-metalium/program_descriptors.hpp>` became `"ttnn/metal_v2_artifacts.hpp"`.
- **Custom `compute_program_hash` deletion:** none — the op was already on the default reflection-based
  hash. Nothing deleted.
- **Pybind entry points removed:** none. `sharded_to_interleaved_nanobind.cpp` binds only the
  `sharded_to_interleaved` free function; no `create_descriptor` was exposed, so no user-visible surface
  changed.
- **Device-operation-class edits forced:** **none.** `sharded_to_interleaved_device_operation.{hpp,cpp}`
  and `..._device_operation_types.hpp` are byte-identical to `origin/main`. The class declares no
  `select_program_factory`, no `cached_program_t`, and nothing descriptor-shaped, so the framework's
  per-factory concept dispatch picked the new path up with no other change. This is the success case the
  integration doc describes.
- **Open items with the concept:** none. The concept fit this op with no friction.

## Handoff points

- **Boundary-rule assumption violations:** none. No call site required a `sem::` or `tensor::` handle
  across the op boundary. The op has no semaphores at all, and both `tensor::dst` uses are inside the
  op's own (forked) kernels.
- **Kernel-lib gaps:** none. Every out-of-op callee the ported kernels touch is an LLK or dataflow
  primitive that either takes a `uint32_t` buffer id (bridged by the documented
  `DFBBindingToken → uint32_t` conversion: `unary_op_init_common`, `copy_tile_init`, `copy_tile`,
  `pack_tile`) or takes the DFB object itself (`Noc::async_write`). Nothing outside the op directory
  needed editing.
- **Framework gaps:** none bit during the port. No UNSUPPORTED feature was reached: no
  `GlobalCircularBuffer`, no `CrossNodeDataflowBuffer`, no compute-kernel Case-2 binding, no
  offset-folded base pointer, no `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr`.
- **Removed pybind surface:** none.

## Successes

- **[Shared-kernel Caution] / the invoker escalation on the reader fork paid off.** The audit's
  Questions #1 (rung 1's fork check is *locational*, and a real non-quasar fork of
  `reader_unary_sharded.cpp` sits in **typecast's** tree, not beside the original) was routed to the
  invoker instead of guessed. The answer — bind typecast's existing fork — avoided creating a **second**
  Metal 2.0 fork of one kernel, which is precisely the duplication the fork convention exists to
  prevent. Had the porter followed rung 2 literally, the tree would now carry two forks of
  `reader_unary_sharded.cpp` with independently-chosen binding vocabularies.
  Applies at `device/sharded_to_interleaved_program_factory.cpp:149`.

- **[Hardware configuration → Compute kernels] Style A vs Style B caught a real trap.** The legacy op
  builds its compute config as a bare `ComputeConfigDescriptor{}` — Style B, every field defaulted. The
  natural instinct was to reach for `to_compute_hardware_config(device->arch(), ...)` "because that's the
  TTNN way." The recipe's warning that the helper's defaults are the *high-performance* ones (so any
  field not explicitly copied would flip) fired correctly: the right answer is a bare
  `ComputeGen1Config{}`, whose defaults match the legacy descriptor's field for field. Applies at
  `device/sharded_to_interleaved_program_factory.cpp:228`.

- **[Compiler options] rule 2 caught a silent perf loss that nothing else would have.** The legacy
  compute `KernelDescriptor` sets no `opt_level` — which reads as "nothing to carry over," and no
  validator, build, or test would have complained. But an absent level resolves to **O3** on a legacy
  `ComputeConfigDescriptor` while Metal 2.0's `CompilerOptions` defaults to **O2**, so leaving it alone
  would have quietly dropped a level on the compute kernel's compile *and* link. Set explicitly at
  `device/sharded_to_interleaved_program_factory.cpp:210`.

- **[Kernel-side whitelist rule 7] The tiled writer's `get_tile_size(cb_id_out)` was not portable as
  written.** Once the buffer id is gone there is no argument to pass, so the free function had to become
  the member getter `dfb_out.get_tile_size()`. Because the getter reads the same JIT
  `unpack_tile_size[]` array behind the same `chlkc_descriptors.h` gate the free function used
  (`DFB_DESCRIPTORS_DEFINED` vs `DATA_FORMATS_DEFINED`, both `__has_include("chlkc_descriptors.h")`),
  the value is identical — but this also means the ported DFB **must** declare
  `data_format_metadata`, or the descriptor header is not generated and the getter silently vanishes.
  Both DFB specs declare it. Applies at
  `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id_metal2.cpp:38`.

- **[Anti-pattern self-audit → "No `cb` survives in a DFB's name"] found real leftovers.** The first
  draft passed every structural check but left seven `cb`-flavoured names: `input_cb_data_format` /
  `output_cb_data_format` in the factory and `cb_read_offset` in the row-major writer, plus four
  legacy-cross-reference comments (`// legacy CBIndex::c_0` and friends). Notably the checklist's own
  reasoning — *post-port the op has no CBs, so every hit is a real leftover* — is what made the comment
  hits worth resolving rather than waving through: a comment is where a retired concept survives
  longest. All renamed / reworded; the sweep now returns zero.

## Friction

### Gaps

- **The shared reference docs the recipe leans on are not in this checkout.** `metal2_port.md` links
  `../shared/port_patterns.md`, `../shared/migration_guide.md`, `../shared/ttnn_factory.md`,
  `../shared/cb_dfb_api_whitelist.md`, and `../shared/workspace_setup.md`; none exist here (`find` over
  the repo returns nothing, and `docs/.../metal_2.0/` is absent). Several recipe steps route the real
  answer to those files — the endpoint-assignment procedure, the CB→DFB method table, the
  `create_program_artifacts` signature and cache lifecycle, the shared-kernel rungs, the fixed form of
  the pointer comment a fork leaves in its legacy original. The port proceeded on the recipe's own
  instruction to treat the headers as ground truth, which worked well (see the next bullet), but three
  things had to be reconstructed rather than looked up: (1) the exact `create_program_artifacts`
  signature and `ProgramArtifacts` shape — recovered from `ttnn/api/ttnn/metal_v2_artifacts.hpp` and the
  concept dispatch in `ttnn/api/ttnn/device_operation.hpp:238`; (2) the CB→DFB spelling for
  `get_tile_size` — recovered from `dataflow_buffer.h:167`; (3) the **wording** of the fork pointer
  comment, which has no recoverable canonical form and was therefore written freehand in all three legacy
  originals. *Suggested fix:* either ship the `shared/` set alongside `metal2_port.md` when the recipe is
  handed to a porter as a standalone file, or have the recipe state up front which of its cross-references
  are load-bearing enough to block on. The pointer-comment form in particular should be inlined into the
  recipe, since it is the one artifact a porter writes into a file they are otherwise forbidden to touch.

- **"Go to the headers first" is stronger advice than the recipe lets on — say so more forcefully.** With
  `port_patterns.md` unavailable, `tt_metal/api/tt-metalium/experimental/metal2_host_api/*.hpp`
  answered essentially every structural question, and answered several *better* than a paraphrase could:
  the per-node "exactly one producer, exactly one consumer" invariant and its legal multi-binding
  relaxation are stated precisely at `dataflow_buffer_spec.hpp:41-50`; the required-`unpack_modes` rule
  at `compute_hardware_config.hpp:119-121`; the reader/writer default triples *with their rationale* at
  `data_movement_hardware_config.hpp:58-100`; the `AddRuntimeArgsForNode` call shapes at
  `program_run_args.hpp:173-186`. The recipe frames header-reading as the better reflex than
  precedent-hunting; on this port it was sufficient on its own. Worth promoting from "the reflex to hunt
  for a precedent is the weaker one" to something closer to "the headers are the primary source; the
  recipe is the procedure around them."

### Confusion

- **The anti-pattern checklist's `cb`-sweep expects zero hits, but does not say whether *comments*
  count.** The item says "Expect **zero** hits: post-port the op has no CBs, so every hit is a real
  leftover," and separately the whitelist says a `CircularBuffer` / `CBDescriptor` grep "should return
  zero hits **in code** (only legacy-comparison artifacts in the port report, if any)." Those two pull in
  opposite directions for a deliberate migration-aid comment such as `// legacy CBIndex::c_0` — which is
  neither stale nor a live CB reference, and which the landed reference port on `main`
  (`typecast_sharded_program_factory.cpp:84-85`) writes verbatim. This port resolved it by keeping the
  *information* and dropping the CB *spelling* (`// the resident input shard`), so the checklist grep is
  clean and a reviewer running it gets no hits to adjudicate. *Suggested fix:* state the intended rule
  explicitly — either "comments too: express the mapping without the legacy spelling" (what was done
  here) or "legacy-cross-reference comments are exempt; the grep is about live vocabulary."

- **Near-miss on the C1 DFB aliasing.** When `!convert_df`, legacy sets `out_cb_index = src0_cb_index`,
  so reader and writer touch the *same* buffer. Read quickly, "one buffer, two touchers, and the writer's
  accessor name differs from the reader's" invites either the multi-binding flag or a self-loop. Both are
  wrong: it is a textbook two-toucher **1P+1C** (reader PRODUCER, writer CONSUMER), and it needs no
  `advanced_options` at all — `accessor_name` is a per-binding-site label, so one DFB can legally be
  reached as `dfb::in` by one kernel and `dfb::out` by another. It also is **not** the recipe's "aliased
  DFB" case, which means a legacy `CBDescriptor` with multiple `format_descriptors` — a different
  construct that happens to share the word "aliasing." The re-derive-from-the-census instruction is what
  kept this straight. *Suggested fix:* a sentence in the aliased-DFB pattern distinguishing *index*
  aliasing across configs (not aliasing; just point the binding at the other spec) from
  *format-descriptor* aliasing (the real case).

- **`Table` vs `Group` is easy to get backwards under time pressure, and the failure is a compile error
  in a 300-line initializer.** `runtime_arg_names` is a `Group` (vector — `push_back` works),
  `runtime_arg_values` / `compile_time_args` / `defines` are `Table`s (no `push_back`). The recipe warns
  about this, and the warning was correct and needed. One thing it does not mention: assigning a *nested*
  designated initializer to a `RuntimeArgSchema` member after the fact
  (`writer.runtime_arg_schema = {.runtime_arg_names = {...}}`) reads oddly next to the surrounding
  designated-initializer style; assigning the inner field directly
  (`writer.runtime_arg_schema.runtime_arg_names = {...}`) is clearer for the runtime-selected-source case
  where two branches set different schemas on one `KernelSpec`.

## Open items for downstream

### Shared kernel touches

Four borrowed kernels; **one reused fork, three new forks, zero in-place modifications** of any legacy
kernel's logic. The only edit to a legacy original is the pointer comment above its includes.

| kernel (legacy original) | rung taken | fork path | remaining unmigrated consumers |
|---|---|---|---|
| `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | **Reused an existing `_metal2` fork.** No new file created; no pointer comment added (rung 1 forbids touching the original). | `copy/typecast/device/kernels/dataflow/reader_unary_sharded_metal2.cpp` (pre-existing, PR #51397) | `sharded_to_interleaved_partial`, `tilize` (×2), `transpose_wh_sharded`, `untilize` (×3), `untilize_with_unpadding`, `slice_write` (×2) |
| `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` | **Created the fork.** Pointer comment landed in the legacy original. | `…/writer_unary_sharded_blocks_interleaved_start_id_metal2.cpp` | `sharded_to_interleaved_partial` |
| `data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | **Created the fork.** Pointer comment landed in the legacy original. | `…/writer_unary_stick_layout_sharded_blocks_interleaved_start_id_metal2.cpp` | `sharded_to_interleaved_partial` |
| `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | **Created the fork.** Pointer comment landed in the legacy original. | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy_metal2.cpp` | `copy` (×2), `interleaved_to_sharded`, `sharded_to_interleaved_partial`, `interleaved_to_sharded_partial`, `untilize_with_unpadding` |

**Binding vocabulary the three new forks establish** — these names are now every later consumer's
interface, so a sibling port should bind against them rather than renaming:

| fork | DFB accessors | tensor accessors | named args |
|---|---|---|---|
| tiled writer | `out` (CONSUMER) | `dst` | RTAs `block_height_tiles`, `block_width_tiles`, `unpadded_block_height_tiles`, `unpadded_block_width_tiles`, `output_width_tiles`, `block_num_tiles`, `start_id_offset`, `start_id_base` |
| RM writer | `out` (CONSUMER) | `dst` | RTAs `block_height`, `block_width_bytes`, `padded_block_width_bytes`, `input_width_offset_bytes`, `start_id` |
| `eltwise_copy_metal2` | `in` (CONSUMER), `out` (PRODUCER) | none | CTA `per_core_tile_cnt` |

The output tensor accessor is named **`dst`**, not `out`, deliberately: `dfb::out` and `tensor::out`
would both be legal (different namespaces) but read as the same resource in a file that binds both.
`dst` matches the kernels' existing `dst_addr` / `dst_args` vocabulary.

**Sunset checklist:** each of the three legacy originals can be deleted once every op in its
"remaining consumers" column has ported. `sharded_to_interleaved_partial` appears in **all four** rows —
it is the single largest co-borrower and the natural next port (see below).

### Next port in the family

`sharded_to_interleaved_partial` binds **all four** of these kernels and has the same
`descriptor` → `ProgramSpecFactoryConcept` shape. Sequencing it next would let it reuse all four forks
at rung 1 with zero new kernel files, and would retire three of the four legacy originals in one step.
It is also the op that actually *uses* the `num_slices` / `slice_index` generality this op carries
vestigially. (This was the audit's Questions #3; recording it here so it is not lost.)

### Pre-existing anomalies carried over verbatim — for the ops team, not this port

These are the audit's "Misc anomalies," re-verified against the ported code. None was touched; each is
behaviour-identical before and after.

1. **Dead row-major writer runtime arg.** Legacy pushed **7** writer args on the RM path, but the kernel
   reads indices 0, 2, 3, 4, 5, 6 — index **1** (`num_units_per_row`, legacy
   `program_factory.cpp:294`) is never read. The brief asked the porter to leave the host-side push
   alone. **Note carefully what that means after the port:** Metal 2.0 has no positional push to leave
   in place — a kernel's runtime args *are* its named schema, and the kernel reads no arg corresponding
   to the dead slot, so there is no name to give it. The dead arg therefore stops being emitted, not
   because the port removed a cleanup-owned line but because the named-arg model has no way to express
   an unnamed unread slot. This is a **zero-behaviour-change** consequence (the kernel never read the
   value), but it does mean the ops team's pending cleanup is already effectively done *on the ported
   path* while remaining outstanding on the legacy original. Flagging explicitly so nobody reads the
   port diff as having quietly absorbed that cleanup.
2. **`is_l1_aligned` is a hardcoded `true`** (factory `:30` post-port), which makes the row-major guard
   `if (is_blackhole or is_l1_aligned) { if (!dst_is_dram or is_l1_aligned) { … } }` unconditionally
   taken. Consequences preserved as-is: `is_blackhole` and `dst_is_dram` are computed but dead in that
   branch (`dst_is_dram` has no other use), and the first `padded_shard_width` assignment is always
   overwritten by the second. A forced constant hiding an unreachable branch — worth a deliberate
   decision, but not the porter's.
3. **`num_slices` / `slice_index` are vestigial for this op.** The launch site hardcodes `1` / `0`, and
   `calculate_starting_idx_h` early-returns `0` when `num_slices <= 1`, so the tiled writer's
   `start_id_base` is **always 0** here. Preserved (still computed through
   `calculate_starting_idx_h`); real generality for the `_partial` sibling, but dead attributes on this
   op's hash.
4. **The TILE/ROW_MAJOR decision is taken off two different tensors.** The unit-size and core-count
   blocks branch on `output.layout()`; kernel-source selection and the per-core runtime-arg loop branch
   on `input.layout()`. They agree in practice (the output is built with the input's layout, and a
   preallocated output must match), but the split reads as accidental and would diverge silently if
   either invariant were relaxed. Preserved exactly, including which tensor each site reads.
5. **Stray debug include in the borrowed reader.** `reader_unary_sharded.cpp:9` includes
   `api/debug/dprint.h` with no `DPRINT` use — and the typecast `_metal2` fork this port now binds
   carries it forward (`:18`). Cosmetic; belongs to whoever owns the fork.

### RTA that is really a CRTA

The reader's `num_tiles_per_core` has the **same value on every node** (`num_units_per_shard`), so it is
a common runtime arg in disguise; the legacy factory set it per-core and the port faithfully kept it
per-core, since RTA→CRTA changes dispatch semantics and is explicitly not port work. Worth picking up in
a later dispatch-efficiency pass — `program_run_args.hpp:66-67` makes the same point at the field. Note
the reused typecast fork has the identical shape, so the two should change together.

### Test coverage notes

- **`test_sharded.py` exercises `sharded_to_interleaved_partial` at six sites** (lines 401, 528, 674,
  773, …). Those hit the **legacy, unforked** kernels, which makes the file a useful incidental check
  that the three new forks did not disturb their legacy originals — the pointer comments are the only
  edits there, but the check is free.
- **The convert_df (C2) path has exactly one obvious test:**
  `tests/ttnn/unit_tests/base_functionality/test_copy.py::test_copy_tilized_nd_sharded_to_interleaved_dtype_conversion`.
  Since C2 is the only config that instantiates the compute kernel and the only user of `OUT_DFB`, that
  single test is the sole guard on the `eltwise_copy_metal2` fork, the second DFB spec, and the
  `ComputeGen1Config` / `O3` decisions. Thin for the config carrying the most port-specific structure —
  worth a deliberate widening (more dtype pairs, block/width/height sharding under conversion) in a
  follow-up, independent of this port.

## Test commands

N150 (single card). Run from the checkout root with the venv active
(`source python_env/bin/activate`).

```bash
# 1. Fastest useful signal — the op's own OOB test plus the convert_df (compute-kernel) path.
#    These two cover C1/C3 shapes and the only C2 coverage that exists.
pytest tests/ttnn/unit_tests/operations/data_movement/test_sharded_to_interleaved_oob.py -x -v
pytest tests/ttnn/unit_tests/base_functionality/test_copy.py -x -v -k sharded_to_interleaved

# 2. Primary sharded coverage. Also exercises sharded_to_interleaved_partial, which still binds the
#    legacy (unforked) kernels — a free check that the forks left their originals intact.
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_sharded.py -x -v

# 3. The remaining two confirmed-baseline files (both reach the op via to_memory_config / ttnn.to_torch).
pytest tests/ttnn/unit_tests/base_functionality/test_to_memory_config.py -x -v
pytest tests/ttnn/unit_tests/operations/data_movement/test_core.py -x -v

# Or the whole confirmed baseline in one go:
pytest tests/ttnn/unit_tests/operations/data_movement/test_sharded_to_interleaved_oob.py \
       tests/ttnn/unit_tests/base_functionality/test_copy.py \
       tests/tt_eager/python_api_testing/unit_testing/misc/test_sharded.py \
       tests/ttnn/unit_tests/base_functionality/test_to_memory_config.py \
       tests/ttnn/unit_tests/operations/data_movement/test_core.py -v
```

There are **no C++ gtests** for this op — a `--gtest_filter='*ShardedToInterleaved*'` over
`unit_tests_ttnn` and its siblings matches nothing, so the pytests above are the whole baseline.

Two notes on reading failures:

- Every test above **selects a converted path**, so the recipe's "unconverted kernel path segfaults the
  pytest session (exit 139)" hazard does not apply here — all four entry points flipped together. An
  exit 139 would be a real problem, not a partial-conversion artifact.
- A `TT_FATAL` from `program_spec.cpp` / `program_run_args.cpp` on the **first** invocation points at
  spec construction (most likely a DFB endpoint or a runtime-arg-name mismatch against the generated
  header). A failure that appears only on the **second and later** invocations would point at the
  program cache — but this op has no custom hash to have survived, so that shape is unexpected here.

## Anti-pattern self-audit — results

| check | result |
|---|---|
| No `tensor.buffer()->address()` survived | **pass** — zero hits. Only `->alignment()` and `->buffer_type()` remain, neither an address. |
| No magic-number CB indices in CTAs | **pass** — the reader and both writers now carry **no** compile-time args at all; the compute kernel carries one named CTA (`per_core_tile_cnt`). |
| No `TensorAccessorArgs<N>()` in any ported kernel | **pass** — zero hits; both writers use `TensorAccessor(tensor::dst)`. |
| No `cb` survives in a DFB's name | **pass** — `grep -rnE '[Cc][Bb]_\|_[Cc][Bb]\b\|\b[Cc][Bb]\b\|\bCB[A-Z]\|CircularBuffer\|CBDescriptor'` over the factory + three forks returns zero. Required renaming `input_cb_data_format`→`input_data_format`, `output_cb_data_format`→`output_data_format`, `cb_read_offset`→`dfb_read_offset`, and rewording four comments. |
| Conditional DFB bindings follow the pattern | **n/a, and deliberately so** — `OUT_DFB` is conditional on `convert_df`, but its only binders (`COMPUTE`, and the writer which points at `IN_DFB` instead when `!convert_df`) mean no kernel source ever references a token its own config does not bind. So no `defines` / `#ifdef` coordination is needed, and none was added. |
| No `.id` extraction at LLK call sites | **pass** — `dfb::in` / `dfb::out` are passed directly into `unary_op_init_common`, `copy_tile_init`, `copy_tile`, `pack_tile`. |
| No CTA→RTA demotion in compute kernels | **pass** — `per_core_tile_cnt` was a CTA in legacy and is a named CTA now. |
| No unnecessary multi-binding flag; never stacked with a self-loop | **pass** — `allow_instance_multi_binding` appears nowhere; no DFB is self-looped. Every DFB is a clean 1P+1C in every config (census in the port plan). |
| All CTAs are named | **pass** — the one surviving CTA is `{{"per_core_tile_cnt", …}}`. |
| No nameable argument smuggled into varargs | **pass** — `get_vararg` appears nowhere; every argument in all four kernels is a distinct field read once, hence named. |
| No ephemeral doc cited from code | **pass** — grepping the diff's `.cpp`/`.hpp` files for `*.md` returns zero. The three pointer comments name the sibling `_metal2` **source** files (which exist on `main` after this PR), not any doc. |
| Every legacy `TT_FATAL` accounted for | **pass** — per-file counts identical before and after (the op's only one is in the untouched device-operation class; the factory had zero and still has zero). |
| Every `hw_config` reproduces the legacy resolved values | **pass** — reader `ReaderConfigDescriptor{}` → `create_reader_datamovement_config` (`RISCV_1`/`NOC_0`/`DEDICATED`); writer `WriterConfigDescriptor{}` → `create_writer_datamovement_config` (`RISCV_0`/`NOC_1`/`DEDICATED`) — distinct cores and distinct NOCs, so the Gen1 node invariant holds; compute all-default `ComputeGen1Config{}` matching an all-default legacy `ComputeConfigDescriptor{}` field for field. `bfp_pack_precision_mode` left default (legacy `bfp8_pack_precise=false`); `unpack_modes` left empty (legacy vector empty, and `enable_32_bit_dest=false` so no explicit entry is required). No Gen2 config authored, no `arch == QUASAR` branch added. |
| Every `KernelSpec`'s `opt_level` matches its legacy kernel's | **pass** — both DM kernels set none (legacy `O2` = Metal 2.0 default `O2`); the compute kernel sets `O3` explicitly (legacy compute default `O3` ≠ Metal 2.0 default `O2`). |
