"""Network-free tests for the Anthropic<->OpenAI translation and the handler.

Upstream is mocked; no real API calls. Run: python3 -m unittest discover -s tests
"""
import json
import os
import sys
import threading
import unittest
import urllib.request
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import proxy  # noqa: E402


def parse_sse(byte_chunks):
    """Turn a list of SSE byte chunks into [(event, data_dict), ...]."""
    text = b"".join(byte_chunks).decode()
    events = []
    cur_event = None
    for line in text.splitlines():
        if line.startswith("event:"):
            cur_event = line[6:].strip()
        elif line.startswith("data:"):
            events.append((cur_event, json.loads(line[5:].strip())))
    return events


class TestRequestTranslation(unittest.TestCase):
    def test_string_content_passthrough(self):
        out = proxy.to_openai({"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(out["messages"][-1], {"role": "user", "content": "hi"})

    def test_system_list_becomes_system_message(self):
        out = proxy.to_openai({
            "system": [{"type": "text", "text": "you are X"}],
            "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(out["messages"][0], {"role": "system", "content": "you are X"})

    def test_assistant_tool_use_injects_reasoning_content(self):
        out = proxy.to_openai({"messages": [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "run", "input": {"x": 1}}]}]})
        am = out["messages"][-1]
        self.assertEqual(am["reasoning_content"], "")           # the fix
        self.assertEqual(am["tool_calls"][0]["id"], "t1")
        self.assertEqual(json.loads(am["tool_calls"][0]["function"]["arguments"]), {"x": 1})
        self.assertIsNone(am["content"])

    def test_tool_messages_precede_user_text(self):
        """Regression: tool messages must come BEFORE user text in the same turn."""
        out = proxy.to_openai({"messages": [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "run", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                {"type": "text", "text": "<reminder>continue</reminder>"}]}]})
        roles = [m["role"] for m in out["messages"]]
        # ... assistant(tool_calls) -> tool -> user(text)
        self.assertEqual(roles, ["assistant", "tool", "user"])
        self.assertEqual(out["messages"][1]["tool_call_id"], "t1")

    def test_assistant_text_only_never_null(self):
        out = proxy.to_openai({"messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}]})
        self.assertEqual(out["messages"][-1], {"role": "assistant", "content": "hello"})

    def test_assistant_only_dropped_thinking_is_empty_not_null(self):
        out = proxy.to_openai({"messages": [
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "x"}]}]})
        self.assertEqual(out["messages"][-1], {"role": "assistant", "content": ""})

    def test_image_block_becomes_multimodal_parts(self):
        out = proxy.to_openai({"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]})
        content = out["messages"][-1]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0], {"type": "text", "text": "what is this"})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,AAAA"))

    def test_forced_tool_choice_downgraded_to_auto(self):
        out = proxy.to_openai({"messages": [{"role": "user", "content": "x"}],
                               "tool_choice": {"type": "any"}})
        self.assertEqual(out["tool_choice"], "auto")

    def test_tools_schema_mapped(self):
        out = proxy.to_openai({"messages": [{"role": "user", "content": "x"}],
                               "tools": [{"name": "run", "description": "d",
                                          "input_schema": {"type": "object"}}]})
        fn = out["tools"][0]["function"]
        self.assertEqual(fn["name"], "run")
        self.assertEqual(fn["parameters"], {"type": "object"})

    def test_provider_pin_and_model_forced(self):
        out = proxy.to_openai({"messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(out["model"], proxy.MODEL)
        self.assertEqual(out["provider"], {"only": [proxy.ONLY], "allow_fallbacks": False})


class TestResponseTranslation(unittest.TestCase):
    def test_anthropic_json_text(self):
        oai = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 3}}
        out = proxy.anthropic_json(oai, "m")
        self.assertEqual(out["content"], [{"type": "text", "text": "hello"}])
        self.assertEqual(out["stop_reason"], "end_turn")
        self.assertEqual(out["usage"], {"input_tokens": 10, "output_tokens": 3})

    def test_anthropic_json_tool_use(self):
        oai = {"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "c1", "function": {"name": "run", "arguments": '{"a":1}'}}]},
            "finish_reason": "tool_calls"}], "usage": {}}
        out = proxy.anthropic_json(oai, "m")
        self.assertEqual(out["stop_reason"], "tool_use")
        self.assertEqual(out["content"][0], {"type": "tool_use", "id": "c1", "name": "run", "input": {"a": 1}})

    def test_synth_sse_sequence(self):
        oai = {"choices": [{"message": {"content": "hi", "tool_calls": [
            {"id": "c1", "function": {"name": "run", "arguments": '{"a":1}'}}]},
            "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 3}}}
        events = parse_sse(list(proxy.synth_sse(oai, "m")))
        names = [e for e, _ in events]
        self.assertEqual(names[0], "message_start")
        self.assertEqual(names[-1], "message_stop")
        self.assertIn("content_block_start", names)
        # cache surfaced in message_start
        self.assertEqual(events[0][1]["message"]["usage"]["cache_read_input_tokens"], 3)
        # two blocks: text (0) + tool_use (1)
        starts = [d for e, d in events if e == "content_block_start"]
        self.assertEqual(starts[0]["content_block"]["type"], "text")
        self.assertEqual(starts[1]["content_block"]["type"], "tool_use")
        delta = [d for e, d in events if e == "message_delta"][0]
        self.assertEqual(delta["delta"]["stop_reason"], "tool_use")

    def test_stream_sse_reconstructs_text_and_tool(self):
        chunks = [
            b'data: {"choices":[{"delta":{"role":"assistant","content":"He"}}]}',
            b'data: {"choices":[{"delta":{"content":"llo"}}]}',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"run","arguments":"{\\"a\\":"}}]}}]}',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}',
            b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"completion_tokens":7}}',
            b'data: [DONE]',
        ]
        events = parse_sse(list(proxy.stream_sse(iter(chunks), "m")))
        names = [e for e, _ in events]
        self.assertEqual(names[0], "message_start")
        self.assertEqual(names[-1], "message_stop")
        # text reconstructs to "Hello"
        text = "".join(d["delta"]["text"] for e, d in events
                        if e == "content_block_delta" and d["delta"]["type"] == "text_delta")
        self.assertEqual(text, "Hello")
        # tool args reconstruct to {"a":1}
        args = "".join(d["delta"]["partial_json"] for e, d in events
                        if e == "content_block_delta" and d["delta"]["type"] == "input_json_delta")
        self.assertEqual(json.loads(args), {"a": 1})
        delta = [d for e, d in events if e == "message_delta"][0]
        self.assertEqual(delta["delta"]["stop_reason"], "tool_use")
        self.assertEqual(delta["usage"]["output_tokens"], 7)


class FakeResp:
    def __init__(self, payload):
        self._b = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    def read(self):
        return self._b
    def close(self):
        pass


class TestHandlerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), proxy.Handler)
        cls.port = cls.srv.server_address[1]
        cls.t = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.t.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def _post(self, body):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/v1/messages", json.dumps(body),
                     {"Content-Type": "application/json"})
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, data

    def test_count_tokens(self):
        status, data = self._post_path("/v1/messages/count_tokens",
                                       {"messages": [{"role": "user", "content": "hi there"}]})
        self.assertEqual(status, 200)
        self.assertIn("input_tokens", json.loads(data))

    def _post_path(self, path, body):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, data

    def test_buffered_non_stream(self):
        fake = FakeResp({"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        with mock.patch("urllib.request.urlopen", return_value=fake):
            status, data = self._post({"messages": [{"role": "user", "content": "hi"}], "stream": False})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["content"][0]["text"], "hi")

    def test_buffered_stream_returns_sse(self):
        fake = FakeResp({"choices": [{"message": {"content": "yo"}, "finish_reason": "stop"}], "usage": {}})
        with mock.patch("urllib.request.urlopen", return_value=fake):
            status, data = self._post({"messages": [{"role": "user", "content": "hi"}], "stream": True})
        self.assertEqual(status, 200)
        self.assertIn(b"event: message_start", data)
        self.assertIn(b"event: message_stop", data)

    def test_upstream_error_propagates(self):
        err = urllib.error.HTTPError("u", 400, "Bad", {}, FakeResp(b'{"error":{"message":"nope"}}'))
        with mock.patch("urllib.request.urlopen", side_effect=err):
            status, data = self._post({"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 400)
        self.assertIn("nope", json.loads(data)["error"]["message"])

    def test_bad_json_400(self):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/v1/messages", "not json", {"Content-Type": "application/json"})
        r = conn.getresponse()
        self.assertEqual(r.status, 400)
        conn.close()


if __name__ == "__main__":
    unittest.main()
