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
