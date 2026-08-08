# Sparse MLA All-to-All Optimization Plan

## Objective

Reduce the two TP4 redistributions around GLM-5.2 sparse SDPA without changing model semantics or the
production fabric setup:

| Transition | Local input | Local output | Original | Direct retained | Linear mux retained |
|---|---:|---:|---:|---:|---:|
| Head to sequence | `[1,16,640,576]` BF16 TILE | `[1,64,160,576]` | 805.6 us | 347.8 us | 282.6 us |
| Sequence to head | `[1,64,160,512]` BF16 TILE | `[1,16,640,512]` | 720.0 us | 318.5 us | 251.4 us |
| Combined | | | 1.526 ms | 0.666 ms | 0.534 ms |

The retained Ring path uses the same bank-owned packets through three mux workers per direction, stripes each
destination across workers by DRAM bank, and splits an even Ring's antipodal destination evenly across both
arcs. On the native LoudBox 1x8 Ring proxy this changes the three-run medians from 737.8 to 324.9 us for
head-to-sequence and from 568.3 to 249.7 us for sequence-to-head: 1.306 ms to 574.6 us combined. The balanced
schedule sustains 36.3 and 42.0 GB/s on each 50 GB/s directed cut (72.6 and 84.0 GB/s bidirectional aggregate).
The best observed samples were 322.0 and 246.6 us (568.6 us combined).

The 1x8 proxy keeps the production per-chip payload but has TP8 hop geometry. For the actual Galaxy TP4
Torus-X ring, balanced shortest-path traffic is 5.898 MB/direction for head-to-sequence and 5.243
MB/direction for sequence-to-head, giving ideal 118.0 and 104.9 us bounds (222.8 us combined). Exact Galaxy
latency must be measured on Galaxy; the local result validates the schedule and achievable link utilization,
not that latency projection.

The operation is semantically correct: it moves each head/token element to the one chip that consumes it.
The optimization target is `all_to_all_async_generic`, not a replacement with an all-gather that replicates
unused data.

Primary metric: real-time-profiler device duration for the two exact production shapes. The bandwidth report
uses 25 GB/s per link per direction. With two links to each neighbor and traffic in both directions, the
aggregate physical ingress roof is 100 GB/s; 50 GB/s is the roof of one directed cut. Report both the busiest
directed edge and the aggregate traffic on the busiest physical cut so routing imbalance is not hidden by a
network average.

## Bottleneck decomposition and candidate audit

Treat the device program as three independently testable stages. A change is not attributed to a stage merely
because it touches that stage's kernel; use the discriminating proxy or sweep in this table.

| Stage | Current work | Discriminating evidence | Candidate changes |
|---|---|---|---|
| Reader side | Interleaved DRAM pages are read one page at a time; one read barrier publishes each retained seven-page CB entry | Compare DRAM input with an otherwise identical L1-input proxy; sweep pages/packet; count source-bank-contiguous runs | Bank-owned block order, coalesced source reads, more reader workers, deeper/multi-transaction read pipeline after coalescing |
| Fabric | Three workers per direction feed a mux and send up to seven 2 KiB pages in a 14 KiB packet; every packet is flushed before its CB entry is released | Payload sweep, latency-vs-bytes intercept, Linear/Ring comparison, per-direction/cut traffic model, and same-platform high-bandwidth-all-gather roof | End-to-end coalesced packets, safe pairwise/directional schedule, multiple in-flight payloads, connection/setup amortization |
| Receiving side | Each packet becomes up to four independent NoC writes to interleaved output DRAM, followed by a completion flush; local output is copied by separate workers | Count actual physical destination runs; profile address/header/write time; compare concurrent versus deferred local copy and the high-bandwidth-all-gather receive roof | Bank-owned destination order, contiguous or fewer-entry scatter writes, fused final write/completion, more receive-side injection, local-copy scheduling |

The same bank-owned ordering can improve all three stages. Blackhole interleaved DRAM has a small fixed bank
set; walking logical pages by the bank count turns same-bank pages into physically contiguous runs. For these
transpose shapes, a run normally remains contiguous in both the source and destination until a tensor-group
boundary. Coalesce each run into one NoC read and one destination write, then combine up to a full 14 KiB
fabric payload as long as the packet needs at most four runs. This removes the current four-*page* limit: the
hardware limit is four destination runs, not four tensor pages.

Do not conflate the shared ordering with one physically fused transaction at every stage. End-to-end
coalescing can help all three stages: it can reduce sender DRAM transactions and barriers, fabric packets and
headers, and receiver writes and completion work. However, each benefit must be demonstrated independently.
In the isolated source-only sweep, seven pipelined 2 KiB source reads were faster than one 14 KiB source read
(about 405-407 versus 426-427 us); that variant preserved the same full 14 KiB CB entries, packet count, and
destination schedule, so it could not improve the fabric or receiver stages. One 14 KiB fabric packet and a
validated contiguous destination write were beneficial. Coalescing is an end-to-end scheduling property here;
the best physical transaction granularity can differ at each stage.

Use `high_bw_all_gather` as the reference implementation for two mechanisms, adapting rather than copying its
all-gather schedule:

- bank-owned slices (`derive_bank_owned_slice`, `slice_step=num_dram_banks`) and contiguous CB batches;
- several workers per direction feeding one dedicated fabric mux core per link/direction.

Candidate order, chosen to separate causes before combining them:

1. Measure an otherwise identical L1-input proxy for both production directions, and use writer-zone,
   destination-run, and high-bandwidth-all-gather evidence to bound the receive side.
2. Add bank-owned/coalesced packets with the current worker topology; sweep 8/10/12/14 KiB effective payload.
3. Add mux-fed worker-count tiers and measure them with coalescing. Linear and Ring retain three workers per
   direction for the target shapes. Ring stripes a destination's banks across workers, with bank-0 ownership
   defining its single initialization signal and per-stream completion counts defining the final barrier exactly.
4. Revisit directional or pairwise scheduling only after packet/worker limits are known; retain only schedules
   that are deadlock-free in Linear and Ring tests.
5. Re-test writer in-flight depth and local-copy timing after coalescing, because the dominant stage may move.
6. Keep topology and adjacent layout fusion as separate experiments; neither is required by the production
   fabric setup in this branch.

## Iteration loop

For every experiment:

1. Change one scheduling or buffering variable at a time.
2. Build with `./build_metal.sh --release`.
3. Run the two isolated production-shape tests:

   ```bash
   scripts/run_safe_pytest.sh \
     models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_ccl_perf.py \
     -m perf -k 'reshard' -s
   ```

4. Record both individual times and their sum. Repeat promising results enough times to rule out noise.
5. Keep only changes that remain bit-exact and improve both directions, or materially improve the pair
   without significantly regressing either direction.

Do not combine packet size, worker count, and scheduling changes until their individual effects are known.

## Experiment 1: use both fabric directions concurrently

Baseline state:

- `num_senders_per_link = 1`.
- One worker for each of the two link lanes services local, positive-direction, and negative-direction
  destinations sequentially.
- TP4 A2A has symmetric eastbound and westbound traffic, so serializing the directions can leave half of the
  full-duplex opportunity idle during parts of the schedule.

Try:

- Set up two sender streams per link.
- Give one stream positive offsets and the other negative offsets.
- Assign the local destination to one stream initially; later isolate it if it causes imbalance.
- Preserve the common destination ordering required to avoid cyclic schedules.

The factory already contains two-element `device_offsets`, `block_starts`, and `block_ends` structures, but
only populates stream zero. Complete this path rather than adding a separate implementation.

Questions to answer:

- Do eastbound and westbound transfers overlap in the device profiler?
- Does four total worker cores improve DRAM/NoC issue rate as well as fabric utilization?
- Is one stream delayed by handling the local quarter?

Keep if the combined time improves by at least 10% without a correctness failure.

## Experiment 2: increase actual tensor payload per packet

Original state:

- Fabric supports a maximum payload near 14 KiB.
- A BF16 TILE page is 2 KiB.
- The generic A2A explicitly flushes after two pages, so its normal tensor payload is at most 4 KiB.
- The approximately 28 KiB circular buffer provides seven 4 KiB slots; it does not create 14 KiB packets.

Sweep:

| Pages | Payload | Notes |
|---:|---:|---|
| 2 | 4 KiB | Baseline |
| 3 | 6 KiB | Fits existing four-entry scatter header |
| 4 | 8 KiB | Largest direct sweep with four independent scatter destinations |

For 12-14 KiB packets, first determine when adjacent destination TILE writes can be coalesced. The writer has
only four scatter address entries, so increasing the byte cap alone is not enough for six or seven unrelated
tiles.

Inspection confirmed that `NOC_SCATTER_WRITE_MAX_CHUNKS` is four. The GLM interleaved-DRAM TILE addresses are
not generally contiguous because consecutive pages rotate through DRAM banks, so 8 KiB is the direct packet
limit without changing the output layout or fabric command format.

Update the reader and writer limits together. The retained implementation derives the page count from the
configured fabric maximum payload and is not keyed to topology or model identity. For bank-owned DRAM-output
schedules, the writer coalesces physically contiguous destination pages into at most four address runs, so the
DeepSeek/GLM 14 KiB configuration selects seven 2 KiB pages. Other layouts, tails, and the generic 4352-byte
fabric default retain the direct scatter path, which safely selects at most four pages and in the default case
two. Verify that CB page size, CB depth, tail handling, and the fused final write-plus-atomic packet remain
consistent.

Questions to answer:

- Is performance monotonic from 4 to 8 KiB?
- Does a larger packet reduce `async_read_barrier` and fabric-send overhead?
- Does it reduce pipeline depth enough to hurt overlap?

## Experiment 3: sweep packet size and sender count together

After Experiments 1 and 2 establish useful independent settings, measure the small cross-product:

| Senders/link | Payload | Head to sequence | Sequence to head | Pair |
|---:|---:|---:|---:|---:|
| 1 | 4 KiB | baseline | baseline | baseline |
| 1 | best larger payload | | | |
| 2 | 4 KiB | | | |
| 2 | best larger payload | | | |

The best isolated values may interact: more workers increase outstanding traffic, while larger packets reduce
the number of opportunities to interleave workers.

## Experiment 4: remove the local-copy stall

Each source retains one quarter of its tensor locally. The same workers currently process that local copy and
remote destinations, including a local NoC write barrier.

Try, in order:

1. Schedule the local destination before or after remote traffic and measure both orders.
2. Assign the local copy to a dedicated worker so eastbound and westbound streams remain active.
3. If address ranges are contiguous enough, coalesce local writes independently of fabric packetization.

Keep this separate from the direction split so the value of local-copy isolation is visible.

## Experiment 5: improve reader/writer overlap

Profile and inspect the remaining serialization after the worker and packet sweeps:

- Reader performs an `async_read_barrier` before publishing every packet.
- Writer calls `async_writes_flushed()` after every local or remote packet.
- Destinations are processed one at a time.
- Connection initialization and global semaphores are paid once per invocation.

Try only changes justified by the profile:

- Increase CB buffering depth if the writer starves.
- Pipeline reads for the next packet while the current packet is being sent.
- Avoid redundant write flushes while preserving packet-header and source-buffer lifetime.
- Interleave destination work only if it does not introduce a fabric cycle or semaphore ambiguity.

Use smaller synthetic shapes to estimate fixed startup cost from the latency intercept, but judge changes on
the two production shapes.

## Experiment 6: layout fusion after fabric utilization improves

The first A2A is immediately followed by TILE-to-row-major conversion. The reverse path converts row-major
SDPA output to TILE immediately before the second A2A.

Investigate a specialized output/input layout path only after the collective itself is substantially faster:

- Head-to-sequence A2A writes row-major output consumed by sparse SDPA.
- Sequence-to-head A2A reads row-major SDPA output and writes TILE output for `wkv_b2`.

This could remove the adjacent untilize and tilize operations, but it is a larger correctness and address
generation change and should not obscure basic fabric tuning.

Measured result: the exact adjacent GLM-5.2 BF16 warm calls are 61.2 us for TILE-to-row-major after the first
A2A and 50.6 us for row-major-to-TILE before the reverse A2A (111.8 us total, 2.3% of a 4.831 ms forward).
TILE storage groups the last two tensor axes `(sequence, width)`, whereas sparse SDPA consumes one token as a
`(heads, width)` matrix. Direct TILE consumption would therefore require 32-token slab ownership and would
collapse the current token-parallel core distribution; direct TILE production has the symmetric whole-slab
write requirement. Fusing conversion into A2A would instead require compute participation and a new mixed-layout
packet/page contract. The measured upper bound does not justify either correctness/scheduling expansion in this
all-gather/A2A branch, so the current explicit conversions are retained.

## Deferred: physical Torus-X

Do not change the production fabric configuration during line-kernel tuning.

The isolated test now has a native `FABRIC_1D_RING` 1x8 LoudBox proxy in addition to real
`FABRIC_2D_TORUS_X` Galaxy-only cases. The proxy keeps the per-chip BF16 payload equal to the target TP4
shapes, but it is still TP8: dimension factorization, hop count, and the even-ring antipodal routing pattern
differ. It is suitable for iterating on ring correctness and link utilization locally, not for claiming the
exact TP4 Torus-X latency.

Once the generic A2A opens wrap-neighbor connections correctly, run the physical topology case on Galaxy:

```bash
scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_ccl_perf.py \
  -m perf -k 'reshard and torus' -s
```

Only X affects these TP-axis A2As. Torus-XY should provide the same benefit as Torus-X for these operations;
Torus-Y does not shorten their routes. With the current TP4 routing schedule, the expected topology-only gain
is bounded and should be evaluated separately from packet and worker improvements.

## Validation gates

After each retained kernel change:

- Exact A2A microbenchmarks pass for both directions.
- Generic A2A coverage passes for supported dimensions, half-TILE tails, one/two links, DRAM output, trace
  reuse, and FABRIC_1D Linear/Ring behavior.
- `git diff --check` is clean.

After a meaningful combined improvement:

- Run `test_sparse_mla.py` for correctness.
- Run `test_sparse_mla_perf.py` for GLM-5.2 sparse warm and long cases.
- Confirm the model still uses exactly two A2As and that the pair improvement appears on the model critical
  path.
- Run `test_sparse_mla_ccl_perf.py` in full on available hardware.
- Validate the physical Torus-X cases on Galaxy before claiming a topology result.

## Result log

| Revision / experiment | Head to sequence | Sequence to head | Pair | Correct | Notes |
|---|---:|---:|---:|:---:|---|
| Baseline | 805.6 us | 720.0 us | 1.526 ms | Yes | LoudBox SP2xTP4, FABRIC_2D Linear |
| Direction split | Hang | Hang | Hang | No | Independent positive/negative streams violate the global destination schedule; rejected |
| Two workers/link | Hang | Hang | Hang | No | Same destination order with block-only partitioning still deadlocks on shared fabric channels; rejected |
| 6 KiB packets | 620.7 us | 553.3 us | 1.174 ms | Yes | Better than baseline, slower than 8 KiB |
| Packet sweep winner | 566.5 us | 503.1 us | 1.070 ms | Yes | 8 KiB packets; 29.9% pair improvement |
| Combined winner | 554.5 us | 493.1 us | 1.048 ms | Yes | 8 KiB packets plus pipelined local writes; 31.3% below baseline |
| Local copy last | 567.7 us | 501.2 us | 1.069 ms | Yes | Within run-to-run noise of 8 KiB baseline; rejected |
| Local write flush | 554.5 us | 493.1 us | 1.048 ms | Yes | Removed redundant per-packet completion barrier; repeat pair was 1.054 ms |
| Fresh packet-batching baseline | 562.0 us | 496.9 us | 1.059 ms | Yes | Same build and machine immediately before local-copy isolation |
| Dedicated local-copy worker | 463.4 us | 413.2 us | 0.877 ms | Yes | Retained; repeats are within about 2 us; 17.2% below fresh baseline and 42.5% below original pair |
| CB depth 2 / 3 / 4 | 464.3 / 463.6 / 464.5 us | 414.6 / 414.7 / 414.4 us | 0.879 / 0.878 / 0.879 ms | Yes | Flat; retained depth 3 |
| Two-transaction read pipeline | 466.3 us | 416.2 us | 0.883 ms | Yes | Correct but slower; rejected |
| Bank-owned 14 KiB + paired target order | 347.8 us | 318.5 us | 0.666 ms | Yes | Direct remote worker; 67.8/65.9 GB/s aggregate |
| Direct worker per direction | 378.7 us | 343.9 us | 0.723 ms | Yes | Opposite bank phase recovered about 10 us but still regressed; rejected |
| Two mux workers/direction | 346.5 us | 307.8 us | 0.654 ms | Yes | Head path flat, reverse 3.4% faster; stable three-run averages |
| Three mux workers/direction | 282.8 us | 251.9 us | 0.535 ms | Yes | Stable three-run averages; 83.4%/83.2% aggregate roofline utilization |
| Per-chip/worker bank phase staggering | 283.5 us | 252.2 us | 0.536 ms | Yes | Neutral/slightly slower than the unrotated three-worker schedule; rejected |
| Three mux buffers/channel | 283.7 us | 253.3 us | 0.537 ms | Yes | Extra mux in-flight depth was flat/slightly slower; retained two buffers |
| Four mux workers, every target split four ways | 385.6 us | 343.6 us | 0.729 ms | Yes | Synchronized all clients on one destination at a time and lost path/receiver spreading; rejected |
| Four mux workers, exclusive target owners | 294.9 us | 256.3 us | 0.551 ms | Yes | Preserved global ordering, but seven remote targets do not keep four clients balanced enough to offset extra mux arbitration; rejected |
| Four mux workers, target owners plus helper | Hang | Hang | Hang | No | Helper visits multiple destinations and reintroduced a cyclic fabric schedule; rejected |
| Address-verified 4 KiB source read runs | 290.4 us | 256.3 us | 0.547 ms | Yes | Single paired run regressed both directions versus the stable per-page baseline; rejected |
| Address-verified 8 KiB source read runs | 282.6 us | 251.8 us | 0.534 ms | Yes | Three-run average is within 0.2 us of per-page reads; rejected as neutral complexity |
| L1-input proxy, same mux/output path | 287.9 us | 254.1 us | N/A | Yes | Paired DRAM-input measurements were 287.7 us and about 254 us; removing source DRAM work is neutral in both directions |
| Assume one contiguous destination run per bank batch | 287.2 us | Incorrect | N/A | No | Head-to-sequence was flat, while sequence-to-head has physical run breaks within some same-bank batches; rejected |
| Incremental bank-owned destination iterator | 289.1 us | 254.0 us | 0.543 ms | Yes | Rollover state/branches cost more than compiler-lowered constant division and were not hidden better by the mux; rejected |
| Per-source mux-channel target rotation | 289.7 us | 254.7 us | 0.544 ms | Yes | Desynchronized channel ownership from the common destination schedule and regressed both directions; rejected |
| Remove mux-path initialization barrier | Hang | Hang | N/A | No | Removes both the all-connections-ready guarantee and the cross-invocation semaphore epoch; rejected because the current protocol requires those invariants |
| Retained Linear mux, final warm-cache run | 282.6 us | 251.4 us | 0.534 ms | Yes | 83.5%/83.4% aggregate ingress-roof utilization |
| Original native 1x8 Ring direct path | 737.8 us | 568.3 us | 1.306 ms | Yes | Same production per-chip payload; TP8 hop geometry |
| Ring mux, two workers/direction | 345.4 us | 262.2 us | 0.608 ms | Yes | Three-run medians; insufficient sender concurrency |
| Retained native 1x8 Ring mux | 324.9 us | 249.7 us | 0.575 ms | Yes | Three-run medians with three workers/direction; best samples 322.0/246.6 us; 72.6%/84.0% utilization |
| Ring mux, four workers/direction | 328.4 us | 258.1 us | 0.586 ms | Yes | Three-run medians; extra mux arbitration regresses both directions |
| Ring mux, five workers/direction | 331.4 us | 261.3 us | 0.593 ms | Yes | Three-run medians; only four bank streams exist per destination |

Device sum profiling on the critical head-to-sequence chip separates a 258--262 us remote-worker data phase
into 202--206 us in fabric payload sends, about 19--20 us in the surrounding destination write/header path,
only about 2 us waiting for reader CB data, and roughly 34 us in destination address generation and packet
scheduling. The reader is therefore ahead of the writer: its long outer zone includes backpressure while the
CB is full, not a DRAM-read bottleneck. The tested source-only transaction merge did not raise fabric
utilization because the writer already received the same full 14 KiB CB entries. A future integrated schedule
that also changes packet count or destination work could still help all three stages. Receive coalescing remains
valuable, but must be based on actual physical destination addresses rather than DRAM-bank identity alone.

The corrected L1-input proxy retains the same bank-owned block assignment, three-worker mux, 14 KiB packetization,
and DRAM destination path; only the source buffer moves from DRAM to interleaved L1. Head-to-sequence measured
287.67 us from DRAM and 287.91 us from L1 in the same run. Sequence-to-head measured about 254 us from either
source. This independently confirms the zone result: neither production direction is source-reader limited.

The qualified high-bandwidth all-gather Fabric2D line test sustains 48.38 GB/s of its 50 GB/s two-link
receive roof with 2 KiB pages on the same LoudBox. Head-to-sequence A2A sustains about 41.4 GB/s over the full
operation, but about 45.1 GB/s when measured over its data phase alone. The comparison bounds the remaining
steady-state cost of multi-destination routing, address scheduling, and arbitrary destination writes to roughly
7% versus the one-hop store-and-forward reference; most of the larger full-operation gap is the approximately
23 us initialization/epoch and teardown cost.

The direction-owned schedule adapts to the subdevice: the target path uses three workers per direction through
one mux, smaller subdevices step down to two muxed workers, and the generic scatter order can use one directly
connected worker per direction. Bank-owned ordering requires a mux tier; if a restricted subdevice cannot fit
one, it retains the proven legacy remote/local schedule. A one-worker schedule on wrap-enabled FABRIC_2D also
retains the legacy path because an equal-cost antipodal route can select either physical direction, regardless of
the operation's logical topology. Small shapes retain that legacy schedule until the message contains enough
full-payload packet work to amortize the additional direction workers and mux startup. The selector normalizes the
physical output allocation by link count, pages per packet, and the three preferred direction-worker lanes. It uses
16 packet equivalents per lane on Linear and four on Ring, where the legacy remote stream otherwise serializes both
physical directions. For 2 KiB pages these correspond to 672 KiB/link for the default seven-page Linear packets,
288 KiB/link for the generic three-page Linear packets, and 168 KiB/link for seven-page Ring packets. These are not
independent byte constants: they move automatically with payload capacity and page size. Two-worker muxing remains
a restricted-subdevice capacity fallback rather than a separate message-size tier, since it did not improve the
full-core crossover sweep.
The dedicated local-copy worker opens no fabric connections and cannot participate in a fabric cycle. Bank
ownership is independent of worker and mux selection: link `l` owns banks `l`,
`l + num_links`, and so on, including uneven bank/link geometry. Missing final bank phases become empty ranges
and do not contribute completion signals.

On the native LoudBox 1x8 Ring proxy, the retained balanced schedule measured three-run medians of 324.9 us
head-to-sequence and 249.7 us sequence-to-head. Splitting the even-ring antipodal destination by DRAM bank
makes every physical cut carry the same 23.593 MB and 20.972 MB respectively. That yields 72.6 and 84.0 GB/s
of the 100 GB/s bidirectional ingress roof, or 36.3 and 42.0 GB/s on each 50 GB/s directed cut. Sweeping two
through five mux clients per direction established three as the optimum on both the pair latency and each
individual direction.

The capacity-derived packet limit and local-worker path passed the generic A2A matrix (20/20): FABRIC_1D
Linear/Ring, FABRIC_2D Linear/Ring, one/two links, DRAM/L1 sharding, explicit reader and writer half-TILE tails,
interleaved L1 input to DRAM output, fewer blocks than links, trace reuse, FABRIC_1D Linear muxing, and adaptive
two-mux-worker/one-direct-worker subdevices. Dedicated cases cover the bank-owned 1D Ring, the small-message
bank-owned legacy schedule, and the restricted-subdevice bank fallback. Empty generic work ranges are skipped by
the writer and excluded from the completion count, matching the reader's zero-CB-entry behavior. A hard-coded
8 KiB limit was rejected because it exceeds the generic fabric default's 4352-byte payload; that failure was a
payload-capacity violation, not an observed topology rule. The full Sparse MLA correctness matrix passed 32/32
cases across 2x4 and 4x2 meshes, including GLM-5.2 BF16 and scaled-FP8 long-cache coverage.
Uneven bank/link ownership is compile-time checked with an 8-bank/3-link geometry; available LoudBox link counts
divide its DRAM-bank count, so physical uneven-ownership coverage remains a Galaxy/BH follow-up.

The retained A2A improvement is visible on the GLM-5.2 model critical path. The final validation run measured:

| Implementation | Cache format | Warm 50K | Long 500K | A2A pair warm | A2A pair long | Programs |
|---|---|---:|---:|---:|---:|---:|
| Sparse | BF16 | 4.830 ms | 9.436 ms | 0.535 ms | 0.536 ms | 60 |
| Sparse | scaled FP8 | 4.645 ms | 9.135 ms | 0.537 ms | 0.536 ms | 62 |
| Dense | BF16 | 3.670 ms | 19.730 ms | N/A | N/A | 26 |

These are equal-workload model comparisons: the sparse and dense proxies use the same GLM-5.2 local chunk,
cache length, head geometry, mesh, and output shape. Sparse has extra index selection and redistribution work,
so it remains slower for the warm cache; at the long cache its reduced attention work dominates and it is
2.09x faster in BF16 (2.16x with the scaled-FP8 cache). The model contains exactly two A2As, and their stable
roughly 0.536 ms combined duration matches the isolated retained result rather than the older
dedicated-local-worker revision's 0.877 ms pair.

## Stop criteria

Stop an experiment when it:

- Regresses both production directions.
- Requires model- or GLM-specific behavior inside the generic collective.
- Adds a program-cache key that changes with runtime tensor contents.
- Changes fabric setup as a hidden requirement.
- Improves the isolated op but does not improve the sparse-MLA critical path.

The practical milestone is complete at a stable 0.534 ms combined Linear pair and a 0.575 ms native 1x8
Ring-proxy pair. Reader CB depth,
transaction-ID pipelining, and source-read coalescing did not help. Experiment 6's adjacent layout fusion was
measured and deferred: its 111.8 us ceiling does not justify adding a mixed-layout collective or restructuring
sparse SDPA's token/core ownership in this all-gather/A2A branch.
