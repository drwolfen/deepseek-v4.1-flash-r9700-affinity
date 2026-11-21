#include "server/server.h"
#include "engine/chat_encode.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define CHECK(cond)                                                               \
  do {                                                                            \
    if (!(cond)) {                                                                \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
      std::abort();                                                               \
    }                                                                             \
  } while (0)

namespace {

int connect_loopback(uint16_t port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  CHECK(fd >= 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  CHECK(::connect(fd, (sockaddr*)&a, sizeof(a)) == 0);
  return fd;
}

void write_all(int fd, const std::string& s) {
  size_t off = 0;
  while (off < s.size()) {
    ssize_t w = ::send(fd, s.data() + off, s.size() - off, 0);
    CHECK(w > 0);
    off += (size_t)w;
  }
}

std::string read_until_eof(int fd) {
  std::string out;
  char buf[4096];
  for (;;) {
    ssize_t r = ::recv(fd, buf, sizeof(buf), 0);
    if (r <= 0) break;
    out.append(buf, (size_t)r);
  }
  return out;
}

std::string make_chunked(const std::string& body, size_t split_at) {
  std::string out;
  auto emit_chunk = [&](const std::string& piece) {
    if (piece.empty()) return;
    char hex[32];
    std::snprintf(hex, sizeof(hex), "%zx\r\n", piece.size());
    out += hex;
    out += piece;
    out += "\r\n";
  };
  if (split_at > 0 && split_at < body.size()) {
    emit_chunk(body.substr(0, split_at));
    emit_chunk(body.substr(split_at));
  } else {
    emit_chunk(body);
  }
  out += "0\r\n\r\n";
  return out;
}

void test_messages_inside_tools_and_json_validation() {
  // "messages" inside a tool schema parameter must NOT hijack top-level "messages" lookup!
  const std::string body = R"({
    "model": "DeepSeek-V4.1-Flash",
    "temperature": 0.7,
    "stream": true,
    "tools": [{
      "type": "function",
      "function": {
        "name": "send_batch",
        "parameters": {
          "type": "object",
          "properties": {
            "messages": {"type": "array", "description": "nested messages field"}
          }
        }
      }
    }],
    "messages": [
      {"role": "system", "content": "Be concise."},
      {"role": "user", "content": "Real user prompt"}
    ]
  })";
  aff::CompletionRequest req;
  CHECK(aff::parse_completion_request(body, true, &req));
  CHECK(req.stream == true);
  CHECK(req.tools.size() == 1);
  CHECK(req.messages.size() == 2);
  CHECK(req.messages[0].role == "system");
  CHECK(req.messages[1].role == "user");
  CHECK(req.messages[1].content == "Real user prompt");

  // Truncated / malformed JSON must fail parse_completion_request
  CHECK(!aff::parse_completion_request(R"({"messages":[{"role":"user","content":"hi"}] )", true, &req));
  CHECK(!aff::parse_completion_request("not json", true, &req));
  CHECK(!aff::parse_completion_request(R"({"model":"x"})", true, &req));

  std::printf("PASS: test_messages_inside_tools_and_json_validation\n");
}

void test_chunked_with_embedded_zero_crlf() {
  const std::string payload =
      R"({"model":"m","messages":[{"role":"user","content":"line1)" "\r\n0\r\n" R"(line2"}]})";
  const std::string chunked = make_chunked(payload, payload.size() / 2);
  std::string decoded;
  CHECK(aff::dechunk_http_body(chunked, &decoded));
  CHECK(decoded == payload);

  // Incomplete chunked body returns false
  std::string partial;
  CHECK(!aff::dechunk_http_body(chunked.substr(0, chunked.size() - 5), &partial));

  std::printf("PASS: test_chunked_with_embedded_zero_crlf\n");
}

void test_sse_keepalive_and_fd_alive() {
  const std::string ka = aff::make_sse_keepalive_chunk(
      "\"id\":\"chatcmpl-1\",\"created\":12345,\"object\":\"",
      "chat.completion.chunk", "DeepSeek-V4.1-Flash", true);
  CHECK(ka ==
        "data: {\"id\":\"chatcmpl-1\",\"created\":12345,\"object\":\"chat.completion.chunk\","
        "\"model\":\"DeepSeek-V4.1-Flash\",\"choices\":[{\"index\":0,\"delta\":{}}]}\n\n");

  int sv[2] = {-1, -1};
  CHECK(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  CHECK(aff::fd_alive(sv[0]));
  ::close(sv[1]);
  CHECK(!aff::fd_alive(sv[0]));
  ::close(sv[0]);

  std::printf("PASS: test_sse_keepalive_and_fd_alive\n");
}

void test_server_http_preflight_chunked_and_content_length() {
  aff::Server srv;
  srv.set_model_name("DeepSeek-V4.1-Flash");
  srv.set_slots(1);
  aff::ServerContext ctx;
  ctx.preflight_check = [](const aff::CompletionRequest& r) -> std::string {
    if (r.n > 1) return "n > 1 not allowed";
    return {};
  };
  srv.set_context(ctx);

  std::string err;
  CHECK(srv.start("127.0.0.1", 0,
                  [](const aff::CompletionRequest& r, const aff::TokenSink&, aff::GenResult* res,
                     uint32_t) {
                    res->content = "echo:" + r.messages.back().content;
                    res->finish_reason = "stop";
                  },
                  &err));
  const uint16_t port = srv.port();

  // 1. Pre-flight HTTP 400 on invalid JSON even when stream=true (no 200 OK header!)
  {
    int fd = connect_loopback(port);
    const std::string bad = "{\"stream\":true,\"messages\":";
    const std::string http = "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                             "Content-Length: " + std::to_string(bad.size()) + "\r\n\r\n" + bad;
    write_all(fd, http);
    const std::string resp = read_until_eof(fd);
    ::close(fd);
    CHECK(resp.rfind("HTTP/1.1 400 Bad Request", 0) == 0);
  }

  // 2. Pre-flight validator failure returns HTTP 400 even when stream=true
  {
    int fd = connect_loopback(port);
    const std::string body = R"({"stream":true,"n":3,"messages":[{"role":"user","content":"hi"}]})";
    const std::string http = "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                             "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    write_all(fd, http);
    const std::string resp = read_until_eof(fd);
    ::close(fd);
    CHECK(resp.rfind("HTTP/1.1 400 Bad Request", 0) == 0);
    CHECK(resp.find("n > 1 not allowed") != std::string::npos);
  }

  // 3. X-Content-Length: 0 must NOT hijack real Content-Length
  {
    int fd = connect_loopback(port);
    const std::string body = R"({"messages":[{"role":"user","content":"header-match"}]})";
    const std::string http = "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                             "X-Content-Length: 0\r\n"
                             "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    write_all(fd, http);
    const std::string resp = read_until_eof(fd);
    ::close(fd);
    CHECK(resp.rfind("HTTP/1.1 200 OK", 0) == 0);
    CHECK(resp.find("echo:header-match") != std::string::npos);
  }

  // 4. Chunked HTTP request containing literal \r\n0\r\n inside chunk payload
  {
    int fd = connect_loopback(port);
    const std::string body =
        R"({"messages":[{"role":"user","content":"before\r\n0\r\nafter"}]})";
    // Also test literal raw \r\n0\r\n bytes inside chunk (using escaped or literal):
    const std::string raw_body =
        std::string("{\"messages\":[{\"role\":\"user\",\"content\":\"before") +
        "\r\n0\r\n" + "after\"}]}";
    (void)body;
    const std::string chunked = make_chunked(raw_body, raw_body.size() / 2);
    const std::string http = "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                             "Transfer-Encoding: chunked\r\n\r\n" + chunked;
    write_all(fd, http);
    const std::string resp = read_until_eof(fd);
    ::close(fd);
    CHECK(resp.rfind("HTTP/1.1 200 OK", 0) == 0);
    CHECK(resp.find("after") != std::string::npos);
  }

  srv.stop();
  std::printf("PASS: test_server_http_preflight_chunked_and_content_length\n");
}

void test_server_fifo_and_disconnect_abort() {
  aff::Server srv;
  srv.set_model_name("DeepSeek-V4.1-Flash");
  srv.set_slots(1);

  std::mutex order_mu;
  std::vector<std::string> served_order;
  std::atomic<bool> req0_entered{false};
  std::atomic<bool> release_req0{false};
  std::atomic<bool> dead_client_ran{false};
  std::atomic<bool> nonstream_aborted{false};

  std::string err;
  CHECK(srv.start("127.0.0.1", 0,
                  [&](const aff::CompletionRequest& r, const aff::TokenSink& sink,
                      aff::GenResult* res, uint32_t) {
                    const std::string tag = r.messages.back().content;
                    if (tag == "dead-queued") {
                      dead_client_ran = true;
                      return;
                    }
                    if (tag == "req0") {
                      req0_entered = true;
                      while (!release_req0.load()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(5));
                      }
                    }
                    if (tag == "abort-nonstream") {
                      for (int i = 0; i < 100; ++i) {
                        if (!sink("", aff::Delta::Content, false)) {
                          nonstream_aborted = true;
                          return;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                      }
                    }
                    {
                      std::lock_guard<std::mutex> lk(order_mu);
                      served_order.push_back(tag);
                    }
                    res->content = tag;
                    res->finish_reason = "stop";
                  },
                  &err));
  const uint16_t port = srv.port();

  auto send_req = [&](const std::string& tag) -> int {
    int fd = connect_loopback(port);
    const std::string body = R"({"messages":[{"role":"user","content":")" + tag + R"("}]})";
    const std::string http = "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                             "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    write_all(fd, http);
    return fd;
  };

  // Occupy slot 0 with req0
  int fd0 = send_req("req0");
  while (!req0_entered.load()) std::this_thread::sleep_for(std::chrono::milliseconds(5));

  // Queue req1, dead-queued, req2, req3 in order; sleep briefly between connects so
  // accept_thread_ preserves queue entry order deterministically.
  int fd1 = send_req("req1");
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  int fd_dead = send_req("dead-queued");
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  int fd2 = send_req("req2");
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  int fd3 = send_req("req3");
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  // Disconnect dead-queued while it is queued behind req0!
  ::close(fd_dead);
  std::this_thread::sleep_for(std::chrono::milliseconds(30));

  // Release req0 -> req1, req2, req3 should run in strict FIFO order, and dead-queued must NOT run!
  release_req0 = true;

  CHECK(read_until_eof(fd0).find("\"req0\"") != std::string::npos);
  CHECK(read_until_eof(fd1).find("\"req1\"") != std::string::npos);
  CHECK(read_until_eof(fd2).find("\"req2\"") != std::string::npos);
  CHECK(read_until_eof(fd3).find("\"req3\"") != std::string::npos);
  ::close(fd0); ::close(fd1); ::close(fd2); ::close(fd3);

  CHECK(!dead_client_ran.load());
  {
    std::lock_guard<std::mutex> lk(order_mu);
    CHECK((served_order == std::vector<std::string>{"req0", "req1", "req2", "req3"}));
  }

  // Test non-streaming disconnect abort via fd_alive(fd)
  int fd_abort = send_req("abort-nonstream");
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  ::close(fd_abort);

  // Server::stop() must join the active abort-nonstream worker cleanly before returning
  srv.stop();
  CHECK(nonstream_aborted.load());

  std::printf("PASS: test_server_fifo_and_disconnect_abort\n");
}

void test_post_send_ordering_and_slot_hold() {
  for (bool stream : {false, true}) {
    aff::Server srv;
    srv.set_model_name("DeepSeek-V4.1-Flash");
    srv.set_slots(1);

    std::atomic<bool> post_entered{false};
    std::atomic<bool> allow_post_finish{false};
    std::atomic<bool> post_finished{false};
    std::atomic<bool> req2_entered{false};
    std::atomic<bool> err_post_ran{false};

    std::string err;
    CHECK(srv.start("127.0.0.1", 0,
                    [&](const aff::CompletionRequest& r, const aff::TokenSink& sink,
                        aff::GenResult* res, uint32_t) {
                      const std::string tag = r.messages.back().content;
                      if (tag == "req1") {
                        if (stream) sink("hello", aff::Delta::Content, false);
                        res->content = "hello";
                        res->finish_reason = "stop";
                        res->post_send = [&]() {
                          post_entered = true;
                          while (!allow_post_finish.load()) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(5));
                          }
                          post_finished = true;
                        };
                      } else if (tag == "req2") {
                        req2_entered = true;
                        if (stream) sink("world", aff::Delta::Content, false);
                        res->content = "world";
                        res->finish_reason = "stop";
                      } else if (tag == "err") {
                        res->error = "boom";
                        res->post_send = [&]() { err_post_ran = true; };
                      }
                    },
                    &err));
    const uint16_t port = srv.port();

    auto send_req = [&](const std::string& tag) -> int {
      int fd = connect_loopback(port);
      const std::string body = std::string("{\"stream\":") + (stream ? "true" : "false") +
                               ",\"messages\":[{\"role\":\"user\",\"content\":\"" + tag + "\"}]}";
      const std::string http = "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                               "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
      write_all(fd, http);
      return fd;
    };

    // 1. Client gets complete reply + EOF BEFORE post_send unblocks!
    int fd1 = send_req("req1");
    const std::string resp1 = read_until_eof(fd1);
    ::close(fd1);
    CHECK(resp1.find("hello") != std::string::npos);
    if (stream) CHECK(resp1.find("data: [DONE]") != std::string::npos);
    while (!post_entered.load()) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CHECK(!post_finished.load());

    // 2. While slot 0 is held in post_send, req2 cannot enter gen_!
    int fd2 = send_req("req2");
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    CHECK(!req2_entered.load());

    // 3. Unblock post_send -> slot releases -> req2 runs and completes.
    allow_post_finish = true;
    const std::string resp2 = read_until_eof(fd2);
    ::close(fd2);
    CHECK(post_finished.load());
    CHECK(req2_entered.load());
    CHECK(resp2.find("world") != std::string::npos);

    // 4. Error responses must NOT invoke post_send.
    int fd_err = send_req("err");
    (void)read_until_eof(fd_err);
    ::close(fd_err);
    CHECK(!err_post_ran.load());

    srv.stop();
  }
  std::printf("PASS: test_post_send_ordering_and_slot_hold\n");
}

void test_preencoded_ids_and_json_string_edges() {
  aff::CompletionRequest def_req;
  CHECK(!def_req.has_preencoded_ids);
  CHECK(def_req.preencoded_ids.empty());

  // Large unescaped string (> 64 KiB) + escapes + BMP + surrogate pair
  std::string big(70000, 'x');
  const std::string body =
      std::string(R"({"model":"m","messages":[{"role":"user","content":")") + big +
      R"(\n\t\r\"\\\u0041\uD83D\uDE00\a"}]})";
  aff::CompletionRequest req;
  CHECK(aff::parse_completion_request(body, true, &req));
  CHECK(req.messages.size() == 1);
  const std::string expected = big + "\n\t\r\"\\A\xF0\x9F\x98\x80" "a";
  CHECK(req.messages[0].content == expected);

  // encode_messages reserve invariance on tool conversation
  std::vector<aff::ChatMsg> msgs;
  aff::ChatMsg sys;
  sys.role = "system";
  sys.content = "System instructions";
  sys.tools.push_back({R"({"name":"calc","description":"Compute"})", "math", "Math tools"});
  msgs.push_back(sys);
  aff::ChatMsg usr;
  usr.role = "user";
  usr.content = std::string(4096, 'y');
  msgs.push_back(usr);
  aff::ChatMsg asst;
  asst.role = "assistant";
  asst.reasoning_content = "Thinking...";
  asst.tool_calls.push_back({"call_1", "calc", R"({"expr":"1+1"})", "math"});
  msgs.push_back(asst);
  aff::EncodeOpts eo;
  eo.v41 = true;
  const std::string encoded = aff::encode_messages(msgs, eo);
  CHECK(encoded.find("System instructions") != std::string::npos);
  CHECK(encoded.find(std::string(4096, 'y')) != std::string::npos);
  CHECK(encoded.find("Thinking...") != std::string::npos);

  std::printf("PASS: test_preencoded_ids_and_json_string_edges\n");
}

} // namespace

int main() {
  test_messages_inside_tools_and_json_validation();
  test_chunked_with_embedded_zero_crlf();
  test_sse_keepalive_and_fd_alive();
  test_server_http_preflight_chunked_and_content_length();
  test_server_fifo_and_disconnect_abort();
  test_post_send_ordering_and_slot_hold();
  test_preencoded_ids_and_json_string_edges();
  std::printf("PASS: all server tests passed.\n");
  return 0;
}
