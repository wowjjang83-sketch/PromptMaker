"""요소 검사 (promptmaker.element_check) 테스트 — 문장 조각 · 잘못된 요소 파일 · 저장/불러오기 적용."""
import os
import re
import shutil
import tempfile
import unittest

from promptmaker import PromptMaker
from promptmaker.data_loader import PACKAGE_DATA_DIR, load_elements
from promptmaker.element_check import (CHECK_KEYS, check_item, clean_element_files, fits, fragment_reason,
                                      suggest_key, trim_fragment)
from promptmaker.vision import append_items, classify


def _base_lines():
    edir = os.path.join(PACKAGE_DATA_DIR, "elements")
    for fname in sorted(os.listdir(edir)):
        key = fname.split(".")[0]
        if not fname.endswith(".txt") or key not in CHECK_KEYS:
            continue
        with open(os.path.join(edir, fname), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    line = re.sub(r"^\s*\d+(?:\.\d+)?::", "", line)
                    line = re.sub(r"(?:\s+[#+!][A-Za-z0-9_]+)+\s*$", "", line)
                    yield key, line


class BaseElementsTest(unittest.TestCase):
    def test_every_base_line_is_ok(self):
        bad = [(k, t, check_item(k, t)) for k, t in _base_lines() if check_item(k, t)["status"] != "ok"]
        self.assertEqual(bad, [], "기본 요소는 모두 그대로 써야 합니다")


class FragmentTest(unittest.TestCase):
    def test_fragments_are_detected(self):
        for text in ("19age is visible in a close-up portrait shot", "conveying a serene",
                     "suggesting an indoor setting with ambient lighting", "the camera angle is low",
                     "looking directly at the camera with a gentle", "no visible makeup",
                     "standing at approximately 5'5", "a sits cross-legged on a bright white background",
                     "diffused, likely studio lighting", "creates gentle shadows", "sunlight streams down from above",
                     "strands framing her face; her makeup is minimal"):
            self.assertTrue(fragment_reason(text), text)

    def test_short_phrases_are_not_fragments(self):
        for text in ("smiling slightly", "serene", "soft lighting", "a relaxed, contented smile",
                     "lying on {her} side propped up on one elbow", "natural daylight behind {her}",
                     "glistening, slightly sweaty", "tilting {her} head slightly"):
            self.assertEqual(fragment_reason(text), "", text)

    def test_trim(self):
        self.assertEqual(trim_fragment("bathed in warm, golden backlighting that"), "warm, golden backlighting")
        self.assertEqual(trim_fragment("the lighting is soft"), "soft lighting")
        self.assertEqual(trim_fragment("19age with long straight black hair"), "long straight black hair")
        self.assertEqual(trim_fragment("20 years old, with long dark brown hair"), "long dark brown hair")
        self.assertEqual(trim_fragment("{her} long, straight brown hair falls over {her} shoulders"),
                         "{her} long, straight brown hair")


class KeyTest(unittest.TestCase):
    def test_misfiled_lines_move(self):
        cases = [
            ("camera", "her dark black hair styled with straight bangs framing her face", "hair"),
            ("camera", "dark hair framing her face", "hair"),
            ("color_grade", "a muted pink lipstick", "makeup"),
            ("lighting", "light pink lips", "makeup"),
            ("lighting", "cheeks subtly contoured with soft lighting", "makeup"),
            ("top", "a teal strapless bikini top", "swimwear"),
            ("action", "sparkling emerald teardrop earrings", "accessory"),
            ("action", "inviting smile", "expression"),
        ]
        for key, text, want in cases:
            r = check_item(key, text)
            self.assertEqual((r["status"], r["key"]), ("moved", want), (key, text, r))

    def test_light_as_colour_is_not_lighting(self):
        self.assertFalse(fits("lighting", "a light blue flower pinned near {her} neck"))
        self.assertFalse(fits("lighting", "red and yellow roses on a light background"))
        self.assertTrue(fits("lighting", "soft natural light"))
        self.assertTrue(fits("lighting", "warm tungsten lamp light"))

    def test_fixed_and_moved_fragment(self):
        r = check_item("lighting", "bathed in soft natural light that")
        self.assertEqual((r["status"], r["text"]), ("fixed", "soft natural light"))
        r = check_item("camera", "a neutral expression gazes directly at the camera in a close-up")
        self.assertEqual((r["status"], r["key"], r["text"]), ("moved", "expression", "a neutral expression"))

    def test_suggest(self):
        self.assertEqual(suggest_key("smiling warmly at the camera"), "action")
        self.assertEqual(suggest_key("seated on a white chair"), "pose")
        self.assertEqual(suggest_key("in a cozy cafe"), "location")


class ApplyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        edir = os.path.join(self.tmp, "elements")
        os.makedirs(edir)
        for key in ("camera", "hair", "lighting", "makeup", "action"):
            with open(os.path.join(edir, f"{key}.txt"), "w", encoding="utf-8") as f:
                f.write(f"# {key}\n")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def read(self, key):
        with open(os.path.join(self.tmp, "elements", f"{key}.txt"), encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip() and not l.startswith("#")]

    def test_append_checks_items(self):
        rep = append_items(self.tmp, {
            "camera": ["her dark black hair styled with straight bangs framing her face",
                       "19age is visible in a close-up shot", "the composition is a tight", "low-angle shot"],
            "lighting": ["a light blue flower pinned near {her} neck", "bathed in warm sunlight that"],
        }, search_dirs=[self.tmp])
        self.assertIn("low-angle shot", self.read("camera"))
        self.assertIn("a close-up shot", self.read("camera"))
        self.assertIn("her dark black hair styled with straight bangs framing her face", self.read("hair"))
        self.assertIn("warm sunlight", self.read("lighting"))
        self.assertNotIn("a light blue flower pinned near {her} neck", self.read("lighting"))
        self.assertEqual(rep["moved"][0]["to"], "hair")
        reasons = {s["reason"] for v in rep["skipped"].values() for s in v}
        self.assertTrue({"fragment", "wrong_key"} <= reasons)

    def test_append_without_check(self):
        append_items(self.tmp, {"camera": ["the composition is a tight"]}, search_dirs=[self.tmp], check=False)
        self.assertIn("the composition is a tight", self.read("camera"))

    def test_classify_marks_rows(self):
        rows = classify([PACKAGE_DATA_DIR, self.tmp], {"camera": ["dark hair framing her face", "conveying a serene"]},
                        ["camera", "hair"])
        st = {r["text"]: (r["status"], r.get("key")) for r in rows["camera"]}
        self.assertEqual(st["dark hair framing her face"], ("new", "hair"))
        self.assertEqual(st["conveying a serene"][0], "fragment")

    def test_generation_skips_fragments(self):
        with open(os.path.join(self.tmp, "elements", "camera.txt"), "a", encoding="utf-8") as f:
            f.write("the camera captures her from a low angle\nher dark hair framing her face\n")
        els = load_elements([PACKAGE_DATA_DIR, self.tmp])
        cam = {i.text for i in els["camera"].items}
        self.assertNotIn("the camera captures her from a low angle", cam)
        self.assertNotIn("her dark hair framing her face", cam)
        self.assertIn("her dark hair framing her face", {i.text for i in els["hair"].items})
        raw = load_elements([PACKAGE_DATA_DIR, self.tmp], check=False)
        self.assertIn("the camera captures her from a low angle", {i.text for i in raw["camera"].items})
        pm = PromptMaker(user_data_dir=self.tmp)
        for seed in range(5):
            self.assertNotIn("captures her", pm.generate("클로즈업 인물 사진", prompt_type="image", seed=seed).prompt)

    def test_clean_files_with_backup(self):
        with open(os.path.join(self.tmp, "elements", "camera.txt"), "a", encoding="utf-8") as f:
            f.write("2::dark hair framing her face #x\nthe composition is a tight\nlow-angle shot\n"
                    "captured in a close-up that reveals her face fully\n")
        preview = clean_element_files(self.tmp)
        self.assertFalse(preview["applied"])
        self.assertIn("the composition is a tight", self.read("camera"))
        res = clean_element_files(self.tmp, apply=True)
        self.assertTrue(os.path.isfile(os.path.join(res["backup"], "camera.txt")))
        self.assertTrue(os.path.isfile(os.path.join(res["backup"], "_manifest.json")))
        self.assertEqual(self.read("camera"), ["low-angle shot", "a close-up"])
        self.assertIn("2::dark hair framing her face #x", self.read("hair"))
        self.assertEqual(clean_element_files(self.tmp)["items"], [])


if __name__ == "__main__":
    unittest.main()


class WildcardJunkTest(unittest.TestCase):
    """0.10.0 — 와일드카드 분석에서 나온 잘못된 줄은 걸러야 함."""

    def test_rejected(self):
        for key, text in (
                ("action", "embodying a mix of innocence"), ("action", "rocking a mesh-overlay lingerie set"),
                ("action", "holding a bouquet of pink"), ("action", "keeping it cool"),
                ("pose", "looking down at it"), ("location", "on a palm trees lining sidewalk"),
                ("location", "on a kneels on a bed"), ("location", "on a transparent mesh thong set"),
                ("lighting", "pedestrians under bright daylight sky"), ("lighting", "intimate atmosphere with minimal shadows"),
                ("camera", "upper body without harsh shadows"), ("camera", "technical details : 24mm wide-angle lens"),
                ("color_grade", "toned thighs"), ("color_grade", "wood grain"), ("color_grade", "a muted lip color"),
                ("skin", "natural matte penis"), ("skin", "forearm"), ("skin", "realistic"), ("mood", "form"),
                ("makeup", "C-cup bust size, full make-up"), ("makeup", "{her} cheeks"),
                ("expression", "a so her facial expression"), ("legwear", "her bare legs"),
                ("accessory", "bangs wearing round-framed glasses"), ("hair", "vellus hair along upper lip"),
                ("hair", "young woman with long dark hair")):
            self.assertNotEqual(check_item(key, text)["status"], "ok", (key, text))

    def test_garment_in_hair_moves(self):
        self.assertEqual(check_item("hair", "a fringe chemise")["key"], "lingerie")
        self.assertEqual(check_item("hair", "fringe micro mini-skirt")["key"], "bottom")

    def test_kept(self):
        for key, text in (("location", "on a charming Paris street lined with cafés"), ("legwear", "bare legs"),
                          ("color_grade", "high-contrast black and white"), ("action", "balancing on a rooftop ledge"),
                          ("hair", "a neat fringe"), ("lighting", "soft natural daylight"), ("camera", "upper body shot"),
                          ("mood", "carefree"), ("skin", "smooth porcelain"), ("footwear", "white chunky sneakers")):
            self.assertEqual(check_item(key, text)["status"], "ok", (key, text))


class WildcardPackTest(unittest.TestCase):
    """0.12.0 — 와일드카드 확장 요소 묶음 (기본은 꺼짐)."""

    def test_pack_lines_are_ok_and_not_in_base(self):
        from promptmaker.core import PACKS_DIR
        edir = os.path.join(PACKS_DIR, "wildcard", "elements")
        base = {(k, t.lower()) for k, t in _base_lines()}
        n = 0
        for fname in sorted(os.listdir(edir)):
            key = fname[:-4]
            with open(os.path.join(edir, fname), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    n += 1
                    self.assertEqual(check_item(key, line)["status"], "ok", (key, line))
                    self.assertNotIn((key, line.lower()), base)
        self.assertGreater(n, 10000)

    def test_pack_is_off_by_default_and_can_be_enabled(self):
        from promptmaker.core import available_packs
        self.assertIn("wildcard", available_packs())
        off = PromptMaker()
        on = PromptMaker(packs=["wildcard"])
        self.assertEqual(off.packs, [])
        self.assertEqual(on.packs, ["wildcard"])
        self.assertGreater(len(on.elements["top"].items), len(off.elements["top"].items) + 1000)
        self.assertTrue(on.generate("bikini on the beach", seed=1).prompt)
