import base64
import io
import json
import os
import shutil
import tempfile
import threading
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
        text = '{"hair": ["Not visible"], "attire": "N/A", "pose": ["no person visible"], "mood": ["not visible", "calm"]}'
        self.assertEqual(vision.parse_response(text), {"mood": ["calm"]})

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
        prompt = backend.prompts[0]
        self.assertIn("- hair:", prompt)
        self.assertNotIn("- age:", prompt)
        self.assertNotIn("- casting:", prompt)
        status = {k: [r["status"] for r in v] for k, v in res["elements"].items()}
        self.assertEqual(status, {"hair": ["duplicate", "new"], "location": ["new"], "made_up": ["unknown_key"]})
        rep = an.apply(res["new"])
        self.assertEqual(rep["added_count"], 2)
        self.assertIn("a sleek silver bob", [i.text for i in maker.elements["hair"].items])  # reloaded

    def test_config_roundtrip(self):
        path = os.path.join(self.tmp, "cfg", "vision.json")
        cfg = vision.VisionConfig.from_dict({"backend": "openai", "skip_keys": "age, casting", "bogus": 1})
        vision.save_config(cfg, path)
        again = vision.load_config(path)
        self.assertEqual(again.backend, "openai")
        self.assertEqual(again.skip_keys, ["age", "casting"])
        self.assertEqual(vision.load_config(os.path.join(self.tmp, "missing.json")).model_id, vision.DEFAULT_MODEL_ID)

    def test_check_model_path(self):
        self.assertTrue(vision.check_model_path("")["ok"])
        self.assertFalse(vision.check_model_path(os.path.join(self.tmp, "nope"))["ok"])
        self.assertFalse(vision.check_model_path(self.tmp)["ok"])
        open(os.path.join(self.tmp, "config.json"), "w").close()
        open(os.path.join(self.tmp, "model.safetensors"), "w").close()
        self.assertTrue(vision.check_model_path(self.tmp)["ok"])


class VisionServerTests(VisionTestCase):
    def test_api(self):
        from http.server import ThreadingHTTPServer
        from promptmaker.server import make_handler

        maker = PromptMaker(data_dir=self.data, translator=NoTranslator())
        old_cfg, old_get = vision.CONFIG_PATH, vision.get_backend
        vision.CONFIG_PATH = os.path.join(self.tmp, "vision.json")
        vision.get_backend = lambda cfg: FakeBackend('{"mood": ["romantic"], "pose": ["kneeling on a picnic blanket"]}')
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
            saved = call("config", {"config": {"api_key": "secret", "max_items_per_key": 2}})
            self.assertEqual(saved["config"]["api_key"], "***")
            self.assertEqual(vision.load_config().api_key, "secret")
            img = "data:image/png;base64," + base64.b64encode(_png_bytes()).decode()
            res = call("analyze", {"image": img})
            self.assertEqual(res["new"], {"pose": ["kneeling on a picnic blanket"]})  # "romantic" exists
            rep = call("apply", {"items": res["new"], "rating": "sfw"})
            self.assertEqual(rep["added_count"], 1)
            self.assertIn("kneeling on a picnic blanket", self.read("pose.txt"))
            self.assertIn("dirs", call("browse?path=" + urllib.request.quote(self.tmp)))
        finally:
            srv.shutdown()
            srv.server_close()
            vision.CONFIG_PATH, vision.get_backend = old_cfg, old_get


if __name__ == "__main__":
    unittest.main()
