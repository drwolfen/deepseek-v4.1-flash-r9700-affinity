// OpenAI-compatible HTTP server (requirement R4).
//
// Deliberately NOT a vLLM-style continuous-batching throughput engine: per decision D1 the
// scheduler is latency-first at B=1-4, because the MoE union law makes wide batching a bad trade
// on this hardware: batching buys a little aggregate throughput for much worse per-request
// latency.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace aff {

struct ToolCallIO {
  std::string id;         // OpenAI's call id; only used to order tool results
  std::string name;
  std::string arguments;  // a JSON object, as a string, which is what the schema wants
  std::string ns;         // optional 1017 tool namespace
};

struct ChatMessage {
  std::string role, content;
  std::string reasoning_content;          // assistant turns, echoed back on the next request
  std::string tool_call_id;               // role == "tool"
  std::vector<ToolCallIO> tool_calls;     // role == "assistant"
  std::string image_data;                 // base64 payload if message contains image
  bool has_image = false;
};

struct CompletionRequest {
  std::string model;
  std::vector<ChatMessage> messages;   // chat endpoint
  std::string prompt;                  // legacy completions endpoint
  // Absent means "as much as the context allows", which is what OpenAI and llama.cpp both do; a
  // fixed default here would silently truncate a long answer at a number the caller never chose.
  uint32_t max_tokens = 0;
  bool has_max_tokens = false;
  uint32_t n = 1;                          // completions to return; only 1 is served
  float temperature = 1.0f;
  float top_p = 1.0f;
  uint32_t top_k = 0;
  float min_p = 0.0f;
  // Repetition penalty, applied on the device before the sample kernel's passes. Same rule as
  // sampler.cpp (positive logits divide, negative multiply). 1.0 disables; `repeat_last_n` 0
  // disables the window. `repetition_penalty` is accepted as an alias because that is the name
  // llama.cpp and most clients send.
  float repeat_penalty = 1.0f;
  uint32_t repeat_last_n = 0;
  // DRY: subtract `dry_delta` from the logits of tokens that continued a repeated suffix n-gram.
  // 0 leaves the engine default (`--dry-delta`) in charge.
  float dry_delta = 0.0f;
  bool raw_prompt = false;
  bool ignore_eos = false;
  bool stream = false;
  // stream_options.include_usage: OpenAI appends one extra chunk carrying usage and no choices.
  bool include_usage = false;
  std::vector<std::string> stop;
  uint64_t seed = 0;
  // The `function` object of each entry of `tools`, as raw JSON. Kept as text rather than parsed
  // because the encoder re-emits it verbatim into the prompt — see chat_encode.h.
  std::vector<std::string> tools;
  std::string tool_choice;                // "auto" | "none" | "required"; advisory
  std::string response_format;            // the raw JSON object, rendered into the system message
  std::string reasoning_effort;           // "low" | "high" | "max"
  std::string thinking_mode;              // "thinking" | "chat"
  bool suppress_final_think = false;      // reasoning_effort="none" while keeping thinking prefix cache
  bool has_temperature = false;           // so a server default is distinguishable from a request
  bool has_top_p = false;
  // Populated by preflight_check on text-only requests so gen_ avoids re-encoding/tokenizing.
  mutable std::vector<uint32_t> preencoded_ids;
  mutable bool has_preencoded_ids = false;
};

// What one generation produced, once its markup has been parsed.
struct GenResult {
  std::string content;
  std::string reasoning_content;
  std::vector<ToolCallIO> tool_calls;
  std::string finish_reason = "stop";     // "stop" | "tool_calls" | "length"
  // `usage` is not optional in the OpenAI schema, and a client that meters context or cost reads
  // it on every reply. Counted in tokens, so reasoning counts toward completion as it does upstream.
  uint32_t prompt_tokens = 0;
  uint32_t completion_tokens = 0;
  uint32_t cached_tokens = 0;
  // Optional work (e.g. prefix cache publish) executed after HTTP reply / SSE [DONE] is sent and
  // the client socket is closed, while the request's slot is still held.
  std::function<void()> post_send;
  // Set when the request cannot be served as asked. The server answers 400 rather than an empty
  // completion, so the caller sees a refusal instead of a model that had nothing to say.
  std::string error;
};

// Which channel a streamed piece belongs to. Reasoning is not the answer, and a client that shows
// the two the same way shows the model thinking out loud as if it had replied — so the stream says
// which is which and lets the client decide, exactly as `reasoning_content` does on the completed
// message.
enum class Delta { Content, Reasoning };

#ifndef AFF_VERSION
#define AFF_VERSION "v5.2.0"
#endif

#ifndef AFF_GIT_SHA
#define AFF_GIT_SHA "unknown"
#endif

#ifndef AFF_BUILD_STAMP
#define AFF_BUILD_STAMP ""
#endif

struct ServerContext {
  std::string status = "ok";
  std::string version = AFF_VERSION;
  std::string git_sha = AFF_GIT_SHA;
  std::string build_stamp = AFF_BUILD_STAMP;
  std::function<int64_t()> get_resident_experts;
  std::function<std::vector<uint64_t>()> get_free_vram_mib;
  // Optional pre-flight validator invoked after parse_completion_request and BEFORE sending
  // HTTP 200 OK headers or acquiring a slot. Non-empty return string produces HTTP 400.
  std::function<std::string(const CompletionRequest&)> preflight_check;
};

// Emits one piece of output. Returning false asks generation to stop (client disconnected).
using TokenSink = std::function<bool(const std::string& piece, Delta chan, bool done)>;

// Supplied by the engine; the server owns no model state.
// The sink streams text as it appears; the result carries what the markup meant, which is only
// known once the turn is complete. A tool call is not text and cannot be streamed as content.
// `slot` is the sequence slot the engine must run this request in: the generator owns one set of
// per-sequence device state per slot, so two requests in different slots can interleave their
// STEPS without touching each other's KV. Always < the value passed to set_slots().
using GenerateFn =
    std::function<void(const CompletionRequest&, const TokenSink&, GenResult*, uint32_t slot)>;

class Server {
public:
  ~Server() { stop(); }
  bool start(const std::string& host, uint16_t port, GenerateFn gen, std::string* err);
  void stop();
  void set_model_name(const std::string& n) { model_name_ = n; }
  // How many requests may be in flight. One slot is the historical behaviour: strict one at a
  // time. Must match the engine's --slots, because it is an index into the engine's per-sequence
  // state and not just a count.
  void set_slots(uint32_t n) { n_slots_ = n ? n : 1; }
  void set_context(ServerContext ctx) { ctx_ = std::move(ctx); }
  const ServerContext& context() const { return ctx_; }
  uint16_t port() const { return port_; }

private:
  void serve_one(int fd);
  int listen_fd_ = -1;
  uint16_t port_ = 0;
  std::string model_name_ = "DeepSeek-V4-Flash";
  ServerContext ctx_;
  // ---- admission ------------------------------------------------------------------------------
  //
  // The generator closes over one engine. Its KV cache is now per SLOT, so two requests in
  // different slots can be in flight at once; what they still share is the per-step scratch, and
  // the generator serializes individual steps on its own lock. This pool is therefore admission
  // control, not mutual exclusion: it hands out slots in strict FIFO ticket order and drops dead
  // sockets before assigning a slot.
  int32_t acquire_slot(int fd, bool stream, const std::string& envelope, const std::string& obj,
                       bool is_chat, std::chrono::steady_clock::time_point* last_send);
  void release_slot(uint32_t s);
  std::mutex slot_mu_;
  std::condition_variable slot_cv_;
  uint64_t slot_busy_ = 0;              // bit s set == slot s is taken
  uint32_t n_slots_ = 1;
  std::deque<uint64_t> wait_queue_;
  uint64_t next_ticket_ = 0;
  GenerateFn gen_;
  std::atomic<bool> running_{false};
  std::thread accept_thread_;
  std::mutex active_mu_;
  std::condition_variable active_cv_;
  uint32_t active_workers_ = 0;
};

// Exposed for testing.
bool parse_completion_request(const std::string& body, bool chat, CompletionRequest* out);
bool fd_alive(int fd);
std::string make_sse_keepalive_chunk(const std::string& envelope, const std::string& obj,
                                     const std::string& model_name, bool is_chat);
bool dechunk_http_body(const std::string& in, std::string* out);
size_t utf8_safe_end(const std::string& s, size_t limit);
std::string sanitize_utf8(const std::string& s);
std::string json_escape(const std::string& s);

} // namespace aff
