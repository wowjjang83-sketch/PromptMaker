"""신체 부위별 클로즈업 + 이미지 분석 구도 문구 분리 테스트."""
import os
import re
import shutil
import tempfile
import unittest

from promptmaker import NoTranslator, PromptMaker
from promptmaker import body_parts
from promptmaker.element_check import check_item
from promptmaker.translator import DictionaryTranslator


class DetectTests(unittest.TestCase):
    def test_korean(self):
        cases = {"다리 클로즈업": "legs", "입술 접사 사진": "lips", "손만 나오게": "hands", "발가락 클로즈업": "feet",
                 "쇄골 클로즈업": "neck", "허리 클로즈업 비키니": "waist", "눈 클로즈업": "eyes", "등 클로즈업": "back",
                 "클로즈업된 허벅지": "thighs"}
        for text, part in cases.items():
            self.assertEqual(body_parts.detect(text)[0], part, text)
        for text in ["다리 위에 서 있는 여자", "클로즈업 사진", "목걸이를 한 여자", "얼빡샷", "옷을 입은 클로즈업"]:
            self.assertIsNone(body_parts.detect(text)[0], text)

    def test_english(self):
        self.assertEqual(body_parts.detect("", "a close-up of her legs")[0], "legs")
        self.assertEqual(body_parts.detect("", "lips close-up")[0], "lips")
        self.assertEqual(body_parts.detect("", "macro shot of her eyes")[0], "eyes")
        self.assertIsNone(body_parts.detect("", "close-up portrait of a woman")[0])

    def test_mentions(self):
        self.assertTrue(body_parts.mentions("sitting with {her} legs crossed", "legs"))
        self.assertFalse(body_parts.mentions("lying on {her} back", "back"))
        self.assertTrue(body_parts.mentions("showing {her} bare back", "back"))


class RouteTests(unittest.TestCase):
    """이미지 분석에서 나온 구도 문구 분리."""

    def test_route_phrase(self):
        self.assertEqual(body_parts.route_phrase("close-up of her legs"), ("body_closeup", "close-up of her legs #legs"))
        self.assertEqual(body_parts.route_phrase("medium shot from the waist up"),
                         ("framing", "medium shot from the waist up #medium"))
        self.assertIsNone(body_parts.route_phrase("shot on an 85mm lens"))
        self.assertIsNone(body_parts.route_phrase("low-angle shot"))

    def test_check_item_moves_camera_framing(self):
        keys = ["camera", "framing", "body_closeup", "hair"]
        r = check_item("camera", "close-up of her feet", keys)
        self.assertEqual((r["status"], r["key"], r["text"]), ("moved", "body_closeup", "close-up of her feet #feet"))
        r = check_item("camera", "full-body shot", keys)
        self.assertEqual((r["key"], r["text"]), ("framing", "full-body shot #full_body"))
        self.assertEqual(check_item("framing", "soft lighting", keys)["status"], "wrong_key")
        self.assertEqual(check_item("camera", "full-body shot")["status"], "ok")  # 옮길 곳 목록이 없으면 그대로

    def test_append_items_routes(self):
        from promptmaker.vision import append_items
        tmp = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmp, "elements"))
            for k in ("camera", "framing", "body_closeup"):
                open(os.path.join(tmp, "elements", f"{k}.txt"), "w", encoding="utf-8").close()
            res = append_items(tmp, {"camera": ["close-up of her lips", "cowboy shot framed from the thighs up"]})
            body = open(os.path.join(tmp, "elements", "body_closeup.txt"), encoding="utf-8").read()
            framing = open(os.path.join(tmp, "elements", "framing.txt"), encoding="utf-8").read()
            self.assertIn("close-up of her lips #lips", body)
            self.assertIn("#cowboy", framing)
            self.assertEqual(len(res["moved"]), 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_template_keys_not_analysed(self):
        from promptmaker.vision import NO_ANALYSIS_KEYS
        self.assertIn("undressing", NO_ANALYSIS_KEYS)


class GenerateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pm = PromptMaker(translator=NoTranslator())
        cls.pm.translator = DictionaryTranslator(cls.pm.ko_en)

    def gen(self, req, seed, **kw):
        return self.pm.generate(req, mode="random", seed=seed, **kw)

    def test_part_closeups(self):
        other = {"legs": r"\b(lips|eyes|lipstick|smil\w*)\b", "lips": r"\b(heels|sandals|legs|feet|knees?)\b",
                 "feet": r"\b(lips|eyes|smil\w*|lipstick|blouse|shirt|top)\b", "hands": r"\b(heels|feet|legs|dress|skirt)\b"}
        for req, part in [("다리 클로즈업", "legs"), ("입술 클로즈업", "lips"), ("발 클로즈업", "feet"), ("손 클로즈업", "hands")]:
            for seed in range(15):
                r = self.gen(req, seed)
                p = r.base_prompt
                self.assertEqual(r.lexicon["intent"]["body_part"], part)
                self.assertEqual(r.template_id, "body_part_closeup", p)
                self.assertIn(f"focus is on her {part}", p)
                self.assertNotRegex(p, other[part], f"{req}/{seed}: {p}")
                self.assertNotIn("{", p)
                self.assertNotIn("also features", p.lower())

    def test_body_template_not_used_without_part(self):
        for seed in range(30):
            self.assertNotEqual(self.gen("해변의 여자", seed).template_id, "body_part_closeup")

    def test_option_and_other_templates(self):
        r = self.gen("여자", 2, body_part="feet")
        self.assertIn("feet", r.base_prompt)
        r = self.gen("다리 클로즈업", 3, template="portrait_default")
        self.assertRegex(r.base_prompt, r"[Cc]lose-up of her (long )?legs")
        with self.assertRaises(ValueError):
            self.gen("여자", 1, body_part="elbow")

    def test_back_closeup_avoids_lying_on_back(self):
        for seed in range(20):
            self.assertNotIn("lying on her back", self.gen("등 클로즈업", seed).base_prompt)

    def test_info(self):
        ids = [p["id"] for p in self.pm.info()["body_parts"]]
        self.assertIn("legs", ids)


if __name__ == "__main__":
    unittest.main()
