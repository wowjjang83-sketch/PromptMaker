import os
import tempfile
import unittest

from promptmaker import llm


class FakeLLM:
    name = "fake"

    def __init__(self, reply):
        self.reply = reply
        self.messages = []
        self.usage = {"input_tokens": 100, "output_tokens": 3}

    def chat(self, messages, max_new_tokens=64):
        self.messages.append(messages)
        return self.reply


class LLMVerifyTestCase(unittest.TestCase):
    ITEMS = [
        {"key": "hair", "text": "long wavy black hair"},
        {"key": "top", "text": "a red string bikini"},
        {"key": "location", "text": "she looks happy and the photo is great"},
    ]

    def test_ok_keeps_everything(self):
        res = llm.verify_items(self.ITEMS, llm=FakeLLM("ok"), cfg=llm.LLMConfig())
        self.assertEqual([r["verdict"] for r in res["items"]], ["ok", "ok", "ok"])

    def test_move_and_drop(self):
        res = llm.verify_items(self.ITEMS, llm=FakeLLM("2=swimwear\n3=x"), cfg=llm.LLMConfig())
        rows = res["items"]
        self.assertEqual(rows[1]["key"], "swimwear")
        self.assertEqual(rows[1]["orig_key"], "top")
        self.assertEqual(rows[1]["verdict"], "move")
        self.assertEqual(rows[2]["verdict"], "drop")
        self.assertEqual(res["stats"]["moved"], 1)
        self.assertEqual(res["stats"]["dropped"], 1)

    def test_unknown_key_and_thinking_ignored(self):
        reply = llm.strip_thinking("<think>hmm</think>1=hairstyle\n2: `swimwear`")
        res = llm.verify_items(self.ITEMS, llm=FakeLLM(reply), cfg=llm.LLMConfig())
        self.assertEqual(res["items"][0]["verdict"], "ok")  # 없는 요소 이름은 무시
        self.assertEqual(res["items"][1]["key"], "swimwear")

    def test_drop_unfit_off(self):
        res = llm.verify_items(self.ITEMS, llm=FakeLLM("3=x"), cfg=llm.LLMConfig(drop_unfit=False))
        self.assertEqual(res["items"][2]["verdict"], "ok")

    def test_prompt_allows_adult_content_and_is_short(self):
        fake = FakeLLM("ok")
        llm.verify_items(self.ITEMS, llm=fake, cfg=llm.LLMConfig())
        system = fake.messages[0][0]["content"]
        self.assertIn("Explicit", system)
        self.assertLess(len(fake.messages[0][1]["content"]), 2500)

    def test_batches(self):
        items = [{"key": "hair", "text": f"hair {i}"} for i in range(5)]
        fake = FakeLLM("ok")
        llm.verify_items(items, llm=fake, cfg=llm.LLMConfig(batch_size=2))
        self.assertEqual(len(fake.messages), 3)

    def test_grouped(self):
        res = llm.verify_grouped({"top": ["a red string bikini"], "hair": ["a bob"]}, llm=FakeLLM("1=swimwear"),
                                 cfg=llm.LLMConfig())
        self.assertEqual(res["found"], {"swimwear": ["a red string bikini"], "hair": ["a bob"]})

    def test_detect_model(self):
        tmp = tempfile.mkdtemp()
        self.assertFalse(llm.detect_model(tmp)["ok"])
        with open(os.path.join(tmp, "config.json"), "w") as f:
            f.write('{"architectures": ["Qwen3ForCausalLM"]}')
        with open(os.path.join(tmp, "model.safetensors"), "wb") as f:
            f.write(b"0" * 10)
        info = llm.detect_model(tmp)
        self.assertTrue(info["ok"])
        self.assertEqual(info["kind"], "transformers")
        gg = tempfile.mkdtemp()
        open(os.path.join(gg, "qwen3-4b-q4_k_m.gguf"), "wb").close()
        self.assertEqual(llm.detect_model(gg)["kind"], "gguf")
        self.assertEqual(llm.LLMConfig(model_path=gg).resolved_backend(), "llama_cpp")

    def test_config_roundtrip(self):
        path = os.path.join(tempfile.mkdtemp(), "llm.json")
        llm.save_config(llm.LLMConfig(model_path="X", batch_size=7), path)
        cfg = llm.load_config(path)
        self.assertEqual((cfg.model_path, cfg.batch_size), ("X", 7))


if __name__ == "__main__":
    unittest.main()


class ScopeGuardTestCase(unittest.TestCase):
    def test_scope_checker_parses_numbers(self):
        chk = llm.ScopeChecker(llm=FakeLLM("2, 3"))
        self.assertEqual(chk.check("침실에서", [("hair", "a"), ("location", "on a beach"), ("mood", "x")]), [1, 2])
        self.assertEqual(llm.ScopeChecker(llm=FakeLLM("ok")).check("x", [("hair", "a")]), [])

    def test_generate_repicks_out_of_scope(self):
        from promptmaker import NoTranslator, PromptMaker

        class BanBeach:
            calls = 0

            def check(self, request, picks):
                BanBeach.calls += 1
                return [i for i, (k, t) in enumerate(picks) if "beach" in t.lower() or "sand" in t.lower()]

        pm = PromptMaker(translator=NoTranslator(), scope_checker=BanBeach())
        for seed in range(15):
            res = pm.generate("a woman reading a book", seed=seed, scope_check=True)
            scope = res.lexicon["scope"]
            if scope.get("flagged"):
                self.assertNotIn("beach", res.picks.get("location", "").lower())
        self.assertGreater(BanBeach.calls, 0)

    def test_scope_off_by_default(self):
        from promptmaker import NoTranslator, PromptMaker

        class Boom:
            def check(self, request, picks):
                raise AssertionError("should not be called")

        res = PromptMaker(translator=NoTranslator(), scope_checker=Boom()).generate("a woman", seed=1)
        self.assertEqual(res.lexicon["scope"], {})


class GgufRoutingTestCase(unittest.TestCase):
    def _gguf_dir(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "Qwen3-VL-8B-Instruct-Q8_0.gguf"), "wb") as f:
            f.write(b"GGUF")
        open(os.path.join(d, "mmproj-F16.gguf"), "wb").close()
        return d

    def test_gguf_never_goes_to_transformers(self):
        d = self._gguf_dir()
        for backend in ("auto", "transformers"):
            self.assertIn(llm.LLMConfig(backend=backend, model_path=d).resolved_backend(), ("llama_cpp", "llama_server"))
        self.assertEqual(llm.detect_model(d)["name"], "Qwen3-VL-8B-Instruct-Q8_0.gguf")  # mmproj 제외

    def test_transformers_backend_refuses_gguf(self):
        d = self._gguf_dir()
        with self.assertRaises(RuntimeError) as ctx:
            llm.TransformersLLM(llm.LLMConfig(backend="transformers", model_path=d))
        self.assertIn("GGUF", str(ctx.exception))

    @unittest.skipIf(os.name == "nt", "shell script server")
    def test_llama_server_backend(self):
        import stat
        import sys

        d = self._gguf_dir()
        tmp = tempfile.mkdtemp()
        script = os.path.join(tmp, "fake_server.py")
        with open(script, "w") as f:
            f.write('''import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj):
        b = json.dumps(obj).encode(); self.send_response(200)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._send({"status": "ok"})
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self._send({"choices": [{"message": {"content": "1=swimwear"}}], "usage": {"prompt_tokens": 50, "completion_tokens": 3}})
HTTPServer(("127.0.0.1", port), H).serve_forever()
''')
        exe = os.path.join(tmp, "llama-server")
        with open(exe, "w") as f:
            f.write(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        os.chmod(exe, os.stat(exe).st_mode | stat.S_IEXEC)
        cfg = llm.LLMConfig(backend="llama_server", model_path=d, server_path=tmp, server_port=18931)
        self.assertEqual(llm.find_llama_server(tmp), exe)
        try:
            res = llm.verify_items([{"key": "top", "text": "a red bikini"}], cfg=cfg)
            self.assertEqual(res["items"][0]["key"], "swimwear")
            self.assertEqual(res["stats"]["output_tokens"], 3)
        finally:
            llm.unload_llm()


class VisionShareTestCase(unittest.TestCase):
    def test_empty_path_uses_vision_model(self):
        self.assertEqual(llm.LLMConfig().resolved_backend(), "vision")
        self.assertEqual(llm.LLMConfig(model_id="qwen3").resolved_backend(), "openai")

    def test_vision_backend_reuses_loaded_model(self):
        from promptmaker import vision

        calls = []

        class FakeVisionBackend:
            device = "cuda"

        orig = (vision.load_config, vision.get_backend, vision.is_loaded)
        vision.load_config = lambda path=None: vision.VisionConfig(backend="openai", model_id="vl", api_base="http://x/v1")
        vision.get_backend = lambda cfg, progress=None: calls.append(cfg) or FakeVisionBackend()
        vision.is_loaded = lambda cfg: True
        orig_chat = llm.OpenAILLM.chat
        llm.OpenAILLM.chat = lambda self, messages, max_new_tokens=64: "1=swimwear"
        try:
            res = llm.verify_items([{"key": "top", "text": "a red bikini"}], cfg=llm.LLMConfig(backend="vision"))
            self.assertEqual(res["items"][0]["key"], "swimwear")
            self.assertEqual(len(calls), 1)
        finally:
            vision.load_config, vision.get_backend, vision.is_loaded = orig
            llm.OpenAILLM.chat = orig_chat
            llm.unload_llm()
