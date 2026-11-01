// Sampling on the device, for the speculative verify block.
//
// WHY IT IS HERE AND NOT ON THE HOST. The block head leaves a [vocab][nb] plane in VRAM, and the
// vocabulary is Row-split, so a host sampler would need every card's band brought back — and then a
// softmax and a nucleus cut over the whole vocabulary per position, on the dispatch thread, in the
// middle of the decode loop. Sampling where the logits already are costs one gather of the other
// cards' bands and nothing else.
//
// WHY IT RETURNS THREE THINGS. Greedy speculation accepts a drafted token when it equals the
// target's argmax. That rule is only correct at temperature 0. With the target sampling, the correct
// rule is rejection sampling — and with a greedy draft (`draft_sample_method: greedy`, which is what
// the model card's vLLM line specifies) the draft's distribution is a point mass, so a proposal is
// accepted with probability p(x) under the target's own filtered distribution. On rejection the
// emitted token must come from the residual, which for a point-mass proposal is the same
// distribution with the proposed token removed and the rest renormalised.
//
// So one pass produces everything a position can need: the accept probability, a draw for the
// bonus/all-accepted case, and a draw for the rejection case. The host then walks the positions with
// its own uniforms and never has to come back.
//
// Consequence worth stating plainly: acceptance FALLS relative to greedy-against-greedy, because a
// proposal the target agrees with is still only accepted with probability p(x) rather than always.
// That is the cost of emitting the target's true distribution, not a regression.

#pragma once

#include <cstdint>

namespace aff {

// The model card's recommendation is the default: temperature 1.0, top_p 0.95 for agentic work
// (1.0 otherwise). top_k and min_p are off, as `generation_config.json` leaves them.
struct SampleParams {
  float    temperature = 1.0f;
  float    top_p       = 0.95f;
  uint32_t top_k       = 0;      // 0 disables
  float    min_p       = 0.0f;   // keep p >= min_p * p_max
  float    repeat_penalty = 1.0f;  // 1.0 disables; negative-free divide/multiply rule
  uint32_t repeat_last_n  = 0;     // 0 disables the window
};

struct SpecDraw {
  uint32_t tok;       // a draw from the filtered distribution
  uint32_t tok_excl;  // a draw with `query` removed and the remainder renormalised
  float    p_query;   // filtered probability of `query` — the accept probability
};

// One position a block. `logits` is the head's plane; `row_stride`/`col_stride` describe it exactly
// as batch_argmax_hip's do, so either layout the head produces is readable without a transpose.
//
// `d_query[i]` is the token the draft proposed for position i, or 0xFFFFFFFF where there is none
// (the last position of a block, which has no proposal to accept). `d_u` holds two uniforms a
// position, [0,1), used for `tok` and `tok_excl`; the accept draw stays on the host, which is where
// the seed lives.
//
// `vocab` is the FULL vocabulary — the caller gathers the other bands first, because a nucleus over
// half a distribution is not a nucleus.
bool batch_sample_hip(const float* logits, uint32_t vocab, uint32_t n,
                      uint32_t row_stride, uint32_t col_stride,
                      const uint32_t* d_query, const float* d_u, SampleParams p,
                      SpecDraw* d_out, void* stream);

// Repetition penalty, applied to the logit plane BEFORE `batch_sample_hip` reads it.
//
// It is a separate launch for one reason: the plane comes back from the head's GEMM, and the
// sample kernel makes several passes over it (max, softmax, top_k radix, nucleus). Penalising
// inside any of those passes would mean either a membership scan per vocabulary entry or a
// second copy of the same rule in each pass. Touching only the historic ids costs one thread per
// id and leaves every pass below reading one consistent plane.
//
// `ids` is a DEVICE array of already-deduplicated token ids in [0, vocab); the caller owns the
// dedup because a window that repeats a token would otherwise compound the penalty per occurrence.
// The rule matches sampler.cpp: a positive logit is divided by `penalty`, a negative one
// multiplied, so no token's score is ever raised. `n` columns (one per position in the block) are
// penalised, `col_stride` apart.
bool apply_repeat_penalty_hip(float* logits, uint32_t vocab, uint32_t n,
                              uint32_t row_stride, uint32_t col_stride,
                              const uint32_t* d_ids, uint32_t n_ids, float penalty, void* stream);

// DRY: subtract `delta` from the named ids, for tokens the host identified as the ones that
// continued a repeated n-gram the last time that n-gram appeared. A constant subtraction, not a
// ratio — see the kernel's comment. `ids` is device-resident and deduplicated, same contract as
// above, and a `delta` of 0 disables the pass.
bool apply_dry_penalty_hip(float* logits, uint32_t vocab, uint32_t n,
                           uint32_t row_stride, uint32_t col_stride,
                           const uint32_t* d_ids, uint32_t n_ids, float delta, void* stream);

} // namespace aff
