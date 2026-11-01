# Scope: return KV physical pages when a sequence ends (`kv_shrink_to`)

Status: **scoped, not implemented.** Recommendation at the bottom is to defer.

## What is being scoped

`VirtualSpan::decommit_to` (`src/gpu/vmem.hip:132`) exists, is chunk-granular, and is called only by
the destructor. The KV grower (`DenseGpu::kv_autogrow_start`, `dense_gpu.hip:2627`) commits forward
and never back, so the 0.66 GiB a maximum-length request spends is held for the life of the process.

The change is to release that memory when a sequence ends, retaining the load-time `--kv-commit`
watermark.

## Measured starting point (2026-11-01)

| | value |
|---|---|
| KV committed at load (`--kv-commit 262144`) | 0.66 GiB (350 MiB/card) |
| growth to `--kv-size 524288` | +662 MiB, linear at 0.66 GiB per 262144 positions |
| free GPU0 after a 479,603-token request, before the sysfs sizing fix | **66 MiB** |
| free GPU0 after the same request, with `hip_free_vram_checked` (shipped) | **920 MiB** |
| cost of reaching 512k positions | the entire load-time reserve |

## Design

### API

```cpp
// DenseGpu: release rules device pages for positions >= `keep`, in every span of the CURRENT slot.
// Requires: no kernel from the finished sequence is in flight; the prefix cache has drained.
// Publishes `kv_backed_ = keep`, which is the watermark the dispatch trusts.
bool kv_shrink_to(uint64_t keep, std::string* err);
```

### Sequence

1. `dense_gpu.sync_devices()` — every card's stream idle. Not optional: the unmap follows kernels
   that may still be reading rows the previous sequence wrote.
2. `prefix_cache.drain()` — the publish writer queue copies rows out of the spans asynchronously.
   Unmapping under a queued copy is the same hazard in a different thread.
3. Per span, per card: `commit_to`'s inverse. Target bytes per layer is
   `comp_rows_for(l, keep) * kv_row_bytes(...)` — the SAME formula the commit uses, so the two
   cannot disagree about where the boundary is.
4. `kv_backed_.store(keep, release)` — after the unmap, never before. A stale-high watermark is the
   one failure mode that corrupts instead of refusing.
5. `kv_note_position(0)` before step 1, so the grower cannot race back up; a mutex shared with the
   grower's commit loop closes the window.

### Call site

`tools/affinity.cpp:2896-2899`, per request, immediately after `model.init_state(&st, ...)` zeroes
the position and after `dense_gpu.set_slot(slot)` selects the slot whose spans are being shrunk
(`:2919`). Guards:

- `dense_gpu.n_slots() == 1` — with `--slots > 1` the engine already requires
  `--kv-commit == --kv-size` (`affinity.cpp:1020`), the grower does not run, and there is nothing to
  release. Two slots would also need per-slot watermarks, which is a wider change.
- skip when `kv_backed_positions() <= keep` (nothing was ever grown).
- retain `--kv-commit`, not 0: the slab was sized against exactly that much KV, and a restore that
  resumes at any point below it must not have to grow first.

## Invariants that must hold

1. **Nothing reads above `keep`.** True at position 0 for the new sequence; the compressor's
   cross-chunk state is a ring indexed by position, so it is overwritten, not read (`affinity.cpp:2495`).
2. **`kv_backed_` never exceeds what is mapped.** The dispatch checks
   `kv_backed_positions()` before every write (`affinity.cpp:2387`). Shrink must lower it atomically
   with the unmap, and a concurrent `kv_commit_positions` must not interleave — hence the grower
   mutex.
3. **Growth after a shrink re-zeroes.** `commit_to` memsets the newly mapped range
   (`vmem.hip:119`); attention masks a partially filled cache by row count, not by content, so a
   re-mapped page holding an old sequence's bytes is a wrong number, not a crash. This is already
   handled and must stay handled.
4. **Restore after a shrink grows first.** `pc_load` → `kv_part_load` writes rows without growing;
   the safety is `kv_reserve_for(hit + 2)` at `affinity.cpp:2921`. That call is what makes a
   shrunk-then-restored prefix correct, and it is the first regression a test must cover.
5. **Chunk granularity.** `decommit_to` refuses to split a chunk, so the retain point rounds up to
   the last whole mapping. Fine at 16k-position margins; it means `keep` is a floor, not an exact
   target.

## Risk register

| risk | severity | mitigation |
|---|---|---|
| unmap under an in-flight kernel | corrupt output, GPU fault | `sync_devices()` + `drain()` before, asserted not assumed |
| publish queue reading rows being freed | fault | `prefix_cache.drain()`; and publish only reads rows below the sequence position |
| grower races the shrink upward | mapping churn or a stale watermark | one mutex; shrink sets `kv_note_position(0)` first |
| restore into unmapped rows | fault | covered by invariant 4; add an explicit test, do not rely on it implicitly |
| driver pool fragmentation from map/unmap cycles | slower growth over hours | soak: 50 × (480k prompt → short request), watch `free` and growth time |
| `--slots > 1` | wrong span shrunk | refused by the guard |

## Test plan

1. **Watermark test (no GPU logic):** after `kv_shrink_to(262144)`, `kv_backed_positions() == 262144`
   and `kv_committed_bytes()` equals the load-time figure.
2. **Round trip:** 480k-token request → free VRAM sampled; assert the reserve returns to within
   32 MiB of idle. Then a 480k request again; assert it succeeds and cost is identical (no
   fragmentation drift over 3 cycles).
3. **Restore after shrink:** publish a 300k-token prefix, shrink, then send the same prompt again.
   Assert `prefix cache: N/N prompt tokens restored` with N > 262144 and correct output hash.
4. **Determinism:** grep the reply hash against the pre-change build on a fixed prompt
   (`--placement-strategy static` makes this a hard check).
5. **Soak:** 50 alternating long/short requests, watchdog on `HIPMALLOC-FAIL` and `cannot back`.

## Effort

| item | size |
|---|---|
| `kv_shrink_to` + grower mutex in `dense_gpu.{h,hip}` | ~80 LOC |
| call site + guards in `tools/affinity.cpp` | ~15 LOC |
| tests (2–4) | ~90 LOC |
| soak + A/B | ~2 h wall clock |
| **total** | **~0.5–1 day**, one build, one restart per arm |

## Recommendation: defer

The shipped `hip_free_vram_checked` fix already removed the danger: after a 479,603-token request
the box holds **920 MiB** of real headroom, not 66. Shrinking would return a further 662 MiB, but:

- **it does not buy residency.** The slab is sized once at load, so the reserve must still cover the
  peak *during* a maximum-length request. Shrinking cannot make the peak smaller, so
  `--gpu-headroom-mib` cannot drop and the expert count cannot rise.
- **the benefit is slack between requests**, and there is already 920 MiB of it.
- **the risk is unmapping GPU memory under a live engine**, plus a coupling to the prefix-cache
  writer queue that today does not exist.

Build it only if one of these becomes true: `--kv-size` goes above 524288 (the peak reserve stops
fitting), max-length requests start arriving often enough that 662 MiB of standing loss matters for
the placement engine's shadow slots, or a second workload needs that VRAM between turns.

Cheaper alternatives that need no code:

- lower `--kv-size` to 327680 — refuses predictably, no growth, no standing loss, context capped;
- raise `--gpu-headroom-mib` by the peak growth (1024 → 1686) — costs ~109 experts, which is the
  same trade the shrink would make at peak, but permanent.

## References

- `src/gpu/vmem.hip:73-148` — `commit_to` / `decommit_to`, chunk list, zero-on-commit
- `src/gpu/dense_gpu.hip:2588-2671` — `comp_rows_for`, `kv_commit_positions`, grower
- `src/gpu/dense_gpu.hip:2935` — `kv_part_load` writes without growing
- `tools/affinity.cpp:1707` — `kv_reserve_for`, `:2896` request reset, `:2921` restore growth
- `tools/affinity.cpp:1020` — `--slots > 1` requires `--kv-commit == --kv-size`
