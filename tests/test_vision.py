import base64
import io
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
import urllib.request

from promptmaker import NoTranslator, PromptMaker, vision
from promptmaker.data_loader import PACKAGE_DATA_DIR


class FakeBackend:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def chat(self, image, prompt):
        self.prompts.append(prompt)
        return self.reply


def _png_bytes():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 120, 90)).save(buf, format="PNG")
    return buf.getvalue()


class VisionTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data = os.path.join(self.tmp, "data")
        shutil.copytree(PACKAGE_DATA_DIR, self.data)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def read(self, name):
        with open(os.path.join(self.data, "elements", name), encoding="utf-8") as f:
            return f.read()


class ParseTests(unittest.TestCase):
    def test_parse_fenced_and_trailing_comma(self):
        text = 'Here:\n```json\n{"hair": ["long red hair", "long red hair"], "mood": "calm", "x": [], "pose": [1, "none"],}\n```'
        self.assertEqual(vision.parse_response(text), {"hair": ["long red hair"], "mood": ["calm"]})

    def test_parse_drops_placeholders(self):
        text = ('{"hair": ["Not visible"], "attire": "N/A", "pose": ["no person visible"], "mood": ["not visible", "calm"], '
                '"motion": ["no motion detected"], "makeup": ["natural no-makeup makeup"]}')
        self.assertEqual(vision.parse_response(text), {"mood": ["calm"], "makeup": ["natural no-makeup makeup"]})

    def test_parse_without_json(self):
        with self.assertRaises(ValueError):
            vision.parse_response("no json here")

    def test_normalize(self):
        n = vision.normalize
        self.assertEqual(n("3::A White Shirt #summer +beach !winter"), n("white  shirt."))
        self.assertEqual(n("casting soft shadows across {her} face"), n("Casting soft shadows across her face"))
        self.assertNotEqual(n("long red hair"), n("short red hair"))


class AppendTests(VisionTestCase):
    def test_duplicates_unsafe_and_unknown_skipped(self):
        before = self.read("hair.txt")
        rep = vision.append_items(self.data, {
            "hair": ["Long straight black hair", "silver hair in a braided crown", "silver hair in a braided crown"],
            "location": ["in a schoolgirl dormitory", "inside a mustard-yellow vintage diner"],
            "no_such_key": ["x"],
        })
        self.assertEqual(rep["added"], {"hair": ["silver hair in a braided crown"],
                                        "location": ["inside a mustard-yellow vintage diner"]})
        reasons = {k: [r["reason"] for r in v] for k, v in rep["skipped"].items()}
        self.assertEqual(reasons["hair"], ["duplicate", "duplicate"])
        self.assertEqual(reasons["location"], ["unsafe"])
        self.assertEqual(reasons["no_such_key"], ["unknown_key"])
        after = self.read("hair.txt")
        self.assertTrue(after.startswith(before))  # only appended
        self.assertIn("# 이미지 분석 자동 추가 (", after)
        self.assertTrue(after.endswith("silver hair in a braided crown\n"))
        # second run: everything is a duplicate, header not repeated
        rep2 = vision.append_items(self.data, {"hair": ["silver hair in a braided crown"]})
        self.assertEqual(rep2["added_count"], 0)
        self.assertEqual(self.read("hair.txt").count("이미지 분석 자동 추가"), 1)

    def test_rating_file_and_cross_rating_duplicates(self):
        rep = vision.append_items(self.data, {"attire_state": ["with a deep plunging neckline", "with a low open back"]},
                                  rating="suggestive")
        self.assertEqual(rep["added"], {"attire_state": ["with a low open back"]})
        self.assertIn("with a low open back", self.read("attire_state.suggestive.txt"))

    def test_appended_items_load(self):
        vision.append_items(self.data, {"location": ["inside a sunlit greenhouse full of ferns"]})
        maker = PromptMaker(data_dir=self.data, translator=NoTranslator())
        texts = [i.text for i in maker.elements["location"].items]
        self.assertIn("inside a sunlit greenhouse full of ferns", texts)


class AnalyzerTests(VisionTestCase):
    def test_analyze_and_apply(self):
        reply = json.dumps({"hair": ["long straight black hair", "a sleek silver bob"],
                            "age": ["25"], "location": ["on a windswept cliff above the sea"],
                            "made_up": ["foo"]})
        maker = PromptMaker(data_dir=self.data, translator=NoTranslator())
        backend = FakeBackend(reply)
        an = vision.ImageAnalyzer(maker, vision.VisionConfig(), backend=backend)
        res = an.analyze(_png_bytes())
        prompt = backend.prompts[-1]
        self.assertIn("- hair:", prompt)
        self.assertNotIn("- age:", prompt)
        self.assertNotIn("- casting:", prompt)
        status = {k: [r["status"] for r in v] for k, v in res["elements"].items()}
        self.assertEqual(status, {"hair": ["duplicate", "new"], "location": ["new"], "made_up": ["unknown_key"]})
        rep = an.apply(res["new"])
        self.assertEqual(rep["added_count"], 2)
        self.assertIn("a sleek silver bob", [i.text for i in maker.elements["hair"].items])  # reloaded

    def test_progress_and_cancel(self):
        class StreamingFake:
            usage = {}

            def chat(self, image, prompt, progress=None, expected_tokens=0):
                if progress is None:  # presence pre-check
                    return '{"person": false, "man": false, "couple": false}'
                progress.generation_started(321, expected_tokens)
                for _ in range(5):
                    progress.token(1)
                self.usage = {"input_tokens": 321, "output_tokens": 5}
                return '{"mood": ["wistful"]}'

        maker = PromptMaker(data_dir=self.data, translator=NoTranslator())
        events = []
        an = vision.ImageAnalyzer(maker, vision.VisionConfig(max_new_tokens=50), backend=StreamingFake())
        res = an.analyze(_png_bytes(), progress=events.append)
        stages = [e.stage for e in events]
        self.assertEqual(stages[0], "prepare")
        self.assertIn("generate", stages)
        self.assertEqual(stages[-1], "done")
        self.assertEqual(events[-1].percent, 100)
        self.assertTrue(all(a.percent <= b.percent for a, b in zip(events, events[1:])))
        self.assertEqual((res["stats"]["input_tokens"], res["stats"]["output_tokens"]), (321, 5))
        self.assertIn("입력 321 토큰", str(events[-1]))
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(vision.AnalysisCancelled):
            an.analyze(_png_bytes(), cancel=cancel)

    def test_presence_precheck_skips_person_keys(self):
        class TwoStep(FakeBackend):
            def chat(self, image, prompt):
                self.prompts.append(prompt)
                if '"person"' in prompt and "- hair:" not in prompt:
                    return '{"person": false, "man": false, "couple": false}'
                return '{"location": ["on a windswept cliff above the sea"], "hair": ["red hair"]}'

        maker = PromptMaker(data_dir=self.data, translator=NoTranslator())
        backend = TwoStep("")
        res = vision.ImageAnalyzer(maker, vision.VisionConfig(), backend=backend).analyze(_png_bytes())
        self.assertEqual(len(backend.prompts), 2)
        main_prompt = backend.prompts[-1]
        self.assertIn("- location:", main_prompt)
        for k in ("- hair:", "- pose:", "- male_look:", "- interaction:"):
            self.assertNotIn(k, main_prompt)
        self.assertEqual(res["presence"], {"person": False, "man": False, "couple": False})
        self.assertEqual(res["elements"]["hair"][0]["status"], "unknown_key")  # answered anyway: not added
        # a woman alone: person keys asked, man/couple keys not
        self.assertEqual(vision.excluded_keys({"person": True, "man": False, "couple": False}, [self.data]),
                         ["male_attire", "male_look", "interaction"])
        self.assertEqual(vision.excluded_keys(None), [])
        off = TwoStep("")
        vision.ImageAnalyzer(maker, vision.VisionConfig(prefilter=False), backend=off).analyze(_png_bytes())
        self.assertEqual(len(off.prompts), 1)

    def test_config_roundtrip(self):
        path = os.path.join(self.tmp, "cfg", "vision.json")
        cfg = vision.VisionConfig.from_dict({"backend": "openai", "skip_keys": "age, casting", "bogus": 1})
        vision.save_config(cfg, path)
        again = vision.load_config(path)
        self.assertEqual(again.backend, "openai")
        self.assertEqual(again.skip_keys, ["age", "casting"])
        self.assertEqual(vision.load_config(os.path.join(self.tmp, "missing.json")).model_id, vision.DEFAULT_MODEL_ID)

    def test_single_file_fp8_dequant(self):
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("torch/safetensors 없음")
        path = os.path.join(self.tmp, "m.safetensors")
        w = torch.tensor([[1.0, -2.0], [0.5, 4.0]])
        save_file({"a.weight": (w / 0.5).to(torch.float8_e4m3fn), "a.weight_scale": torch.tensor(0.5),
                   "a.comfy_quant": torch.tensor(list(b'{"format": "float8_e4m3fn"}'), dtype=torch.uint8),
                   "a.bias": torch.ones(2)}, path)
        self.assertTrue(vision.check_model_path(path)["ok"])
        sd = vision.load_single_file(path, torch.float32)
        self.assertEqual(sorted(sd), ["a.bias", "a.weight"])
        self.assertTrue(torch.allclose(sd["a.weight"], w))

    def test_check_model_path(self):
        self.assertTrue(vision.check_model_path("")["ok"])
        self.assertFalse(vision.check_model_path(os.path.join(self.tmp, "nope"))["ok"])
        self.assertFalse(vision.check_model_path(self.tmp)["ok"])
        open(os.path.join(self.tmp, "config.json"), "w").close()
        open(os.path.join(self.tmp, "model.safetensors"), "w").close()
        self.assertTrue(vision.check_model_path(self.tmp)["ok"])


FULL_REPLY = json.dumps({"location": ["inside a glass conservatory full of orchids"],
                         "full_prompt": "A woman in her late twenties stands in a glass conservatory.\nSoft light."})


class QueueAndFullPromptTests(VisionTestCase):
    def analyzer(self, reply=FULL_REPLY, full=True):
        maker = PromptMaker(data_dir=self.data, translator=NoTranslator())
        cfg = vision.VisionConfig(full_prompt=full, full_prompt_file=os.path.join(self.tmp, "out", "fp.txt"))
        return vision.ImageAnalyzer(maker, cfg, backend=FakeBackend(reply))

    def test_full_prompt_requested_and_split(self):
        an = self.analyzer()
        res = an.analyze(_png_bytes())
        self.assertIn('"full_prompt"', an._backend.prompts[-1])
        self.assertEqual(res["full_prompt"], "A woman in her late twenties stands in a glass conservatory. Soft light.")
        self.assertNotIn("full_prompt", res["elements"])
        off = self.analyzer(full=False)
        self.assertEqual(off.analyze(_png_bytes())["full_prompt"], "")
        self.assertNotIn('"full_prompt"', off._backend.prompts[-1])

    def test_save_full_prompts_dedup_and_safety(self):
        path = os.path.join(self.tmp, "out", "fp.txt")
        r1 = vision.save_full_prompts(["A calm portrait of an adult woman.", "a calm portrait of an adult woman",
                                       "A schoolgirl in a classroom."], path)
        self.assertEqual(r1["saved"], ["A calm portrait of an adult woman."])
        self.assertEqual([s["reason"] for s in r1["skipped"]], ["duplicate", "unsafe"])
        r2 = vision.save_full_prompts(["A calm portrait of an adult woman.", "A second prompt."], path)
        self.assertEqual(r2["saved"], ["A second prompt."])
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.assertEqual(lines[0][0], "#")
        self.assertEqual(lines[1:], ["A calm portrait of an adult woman.", "A second prompt."])

    def test_queue_runs_in_order_with_auto_options(self):
        an = self.analyzer()
        q = vision.AnalysisQueue(an, auto_apply=True, auto_save_full_prompt=True)
        q.add_many([_png_bytes(), _png_bytes()])
        folder = os.path.join(self.tmp, "imgs")
        os.makedirs(folder)
        for n in ("b.png", "a.png", "skip.txt"):
            with open(os.path.join(folder, n), "wb") as f:
                f.write(_png_bytes())
        self.assertEqual([i.name for i in q.add_folder(folder)], ["a.png", "b.png"])
        seen = []
        items = q.run(progress=lambda item, p: seen.append(item.id))
        self.assertEqual([i.status for i in items], ["done"] * 4)
        self.assertEqual(sorted(set(seen)), [i.id for i in items])
        self.assertEqual(items[0].report["added_count"], 1)   # first image adds the location
        self.assertIsNone(items[1].report)   # same answer again: all duplicates, nothing to add
        self.assertEqual(items[0].full_prompt_report["saved_count"], 1)
        self.assertEqual(items[1].full_prompt_report["saved_count"], 0)
        self.assertEqual(q.summary()["percent"], 100.0)

    def test_queue_errors_and_stop(self):
        an = self.analyzer(reply="no json")
        q = vision.AnalysisQueue(an)
        q.add_many([_png_bytes(), _png_bytes()])
        items = q.run()
        self.assertEqual([i.status for i in items], ["error", "error"])  # keeps going after an error
        q2 = vision.AnalysisQueue(self.analyzer())
        q2.add_many([_png_bytes(), _png_bytes()])
        q2.run(progress=lambda item, p: q2.stop())
        self.assertEqual([i.status for i in q2.items], ["cancelled", "cancelled"])


class UnloadTests(VisionTestCase):
    def setUp(self):
        super().setUp()
        self.calls = []
        self._old = vision.get_backend, vision.unload_backends
        vision.get_backend = lambda cfg, progress=None: FakeBackend('{"mood": ["serene"]}')
        vision.unload_backends = lambda: self.calls.append(1) or {"models": 1, "gpu_freed_mb": 0}

    def tearDown(self):
        vision.get_backend, vision.unload_backends = self._old
        super().tearDown()

    def analyzer(self, **cfg):
        maker = PromptMaker(data_dir=self.data, translator=NoTranslator())
        return vision.ImageAnalyzer(maker, vision.VisionConfig(**cfg))

    def test_single_analysis_unloads_by_default(self):
        res = self.analyzer().analyze(_png_bytes())
        self.assertEqual(self.calls, [1])
        self.assertEqual(res["unloaded"]["models"], 1)
        self.analyzer().analyze(_png_bytes(), unload=False)
        self.analyzer(unload_after=False).analyze(_png_bytes())
        self.assertEqual(self.calls, [1])

    def test_unload_even_on_error(self):
        vision.get_backend = lambda cfg, progress=None: FakeBackend("no json")
        with self.assertRaises(ValueError):
            self.analyzer().analyze(_png_bytes())
        self.assertEqual(self.calls, [1])

    def test_queue_unloads_once_at_the_end(self):
        q = vision.AnalysisQueue(self.analyzer())
        q.add_many([_png_bytes()] * 3)
        q.run()
        self.assertEqual(self.calls, [1])
        self.assertEqual(q.summary()["last_unload"]["models"], 1)
        q2 = vision.AnalysisQueue(self.analyzer(), unload_when_done=False)
        q2.add(_png_bytes())
        q2.run()
        self.assertEqual(self.calls, [1])

    def test_queue_picks_up_images_added_while_unloading(self):
        q = vision.AnalysisQueue(self.analyzer())
        q.add(_png_bytes())
        added = []

        def unload_and_add():
            self.calls.append(1)
            if not added:  # someone drops another image just as the queue empties
                added.append(q.add(_png_bytes()))
            return {"models": 1, "gpu_freed_mb": 0}

        vision.unload_backends = unload_and_add
        q.run()
        self.assertEqual([i.status for i in q.items], ["done", "done"])
        self.assertEqual(len(self.calls), 2)

    def test_real_unload_clears_cache(self):
        vision.unload_backends = self._old[1]
        dummy = type("B", (), {"model": object(), "processor": object()})()
        vision._backend_cache[("x",)] = dummy
        info = vision.unload_backends()
        self.assertEqual(info["models"], 1)
        self.assertEqual(vision._backend_cache, {})
        self.assertIsNone(dummy.model)


class VisionServerTests(VisionTestCase):
    def test_api(self):
        from http.server import ThreadingHTTPServer
        from promptmaker.server import make_handler

        maker = PromptMaker(data_dir=self.data, translator=NoTranslator())
        old_cfg, old_get = vision.CONFIG_PATH, vision.get_backend
        from promptmaker import vision_api

        vision_api._queue["q"] = None
        vision.CONFIG_PATH = os.path.join(self.tmp, "vision.json")
        vision.get_backend = lambda cfg, progress=None: FakeBackend('{"mood": ["romantic"], "pose": ["kneeling on a picnic blanket"]}')
        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(maker))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}/api/vision/"

        def call(path, body=None):
            data = json.dumps(body).encode() if body is not None else None
            req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as r:
                return json.loads(r.read())
        try:
            self.assertEqual(call("config")["config"]["model_id"], vision.DEFAULT_MODEL_ID)
            saved = call("config", {"config": {"api_key": "secret", "max_items_per_key": 2,
                                               "full_prompt_file": os.path.join(self.tmp, "fp.txt")}})
            self.assertEqual(saved["config"]["api_key"], "***")
            self.assertEqual(vision.load_config().api_key, "secret")
            img = "data:image/png;base64," + base64.b64encode(_png_bytes()).decode()
            res = call("analyze", {"image": img})
            self.assertEqual(res["new"], {"pose": ["kneeling on a picnic blanket"]})  # "romantic" exists
            rep = call("apply", {"items": res["new"], "rating": "sfw"})
            self.assertEqual(rep["added_count"], 1)
            self.assertIn("kneeling on a picnic blanket", self.read("pose.txt"))
            self.assertIn("dirs", call("browse?path=" + urllib.request.quote(self.tmp)))
            # picking a file shows its folder instead of failing
            listing = call("browse?path=" + urllib.request.quote(os.path.join(self.data, "elements", "pose.txt")))
            self.assertIn("pose.txt", listing["files"])
            job = call("jobs", {"image": img})
            for _ in range(100):
                job = call("jobs/" + job["id"])
                if job["status"] not in ("queued", "running"):
                    break
                time.sleep(0.05)
            self.assertEqual(job["status"], "done")
            self.assertEqual(job["progress"]["stage"], "done")
            self.assertIn("runtime", call("config"))
            # queue: two images in one request, analysed one after another
            state = call("queue", {"images": [{"name": "one.png", "image": img}, {"name": "two.png", "image": img}],
                                   "auto_apply": False})
            self.assertEqual([i["name"] for i in state["items"]][-2:], ["one.png", "two.png"])
            for _ in range(100):
                state = call("queue")
                if not state["summary"]["running"] and not state["summary"]["counts"].get("pending"):
                    break
                time.sleep(0.05)
            self.assertEqual([i["status"] for i in state["items"]], ["done", "done"])
            self.assertNotIn("result", state["items"][0])
            one = call("queue/" + state["items"][0]["id"])
            self.assertIn("elements", one["result"])
            state = call("queue/clear", {})
            self.assertEqual(state["items"], [])
            saved = call("full_prompts", {"prompts": ["An adult woman reading by a window."]})
            self.assertEqual(saved["saved_count"], 1)
            self.assertTrue(os.path.isfile(saved["path"]))
        finally:
            srv.shutdown()
            srv.server_close()
            vision.CONFIG_PATH, vision.get_backend = old_cfg, old_get


if __name__ == "__main__":
    unittest.main()
