"""샷 구도(얼빡샷·클로즈업·미디엄·카우보이·전신)와 탈의 동작(옷을 벗는 중/장면) 조합 테스트."""
import re
import unittest

from promptmaker import NoTranslator, PromptMaker
from promptmaker import framing, undressing
from promptmaker.translator import DictionaryTranslator

LEGS = re.compile(r"\b(heels|sandals|boots|sneakers|loafers|barefoot|wedges|shoes|pumps|flats|feet|knees?|"
                  r"kneeling|anklets?)\b", re.I)


class FramingDetectTests(unittest.TestCase):
    def test_korean_terms(self):
        self.assertEqual(framing.detect("얼빡샷")[0], "extreme_closeup")
        self.assertEqual(framing.detect("얼굴만 나오게")[0], "extreme_closeup")
        self.assertEqual(framing.detect("클로즈업 사진")[0], "closeup")
        self.assertEqual(framing.detect("미디엄샷 여자")[0], "medium")
        self.assertEqual(framing.detect("카우보이샷")[0], "cowboy")
        self.assertEqual(framing.detect("전신샷")[0], "full_body")
        self.assertEqual(framing.detect("비키니샷")[0], None)

    def test_english_terms(self):
        self.assertEqual(framing.detect("", "an extreme close-up of a woman")[0], "extreme_closeup")
        self.assertEqual(framing.detect("", "close-up portrait")[0], "closeup")
        self.assertEqual(framing.detect("", "cowboy shot")[0], "cowboy")
        self.assertEqual(framing.detect("", "full body shot")[0], "full_body")
        self.assertEqual(framing.detect("", "upper body")[0], "medium")

    def test_conflicts_pick_one(self):
        fr, found, _ = framing.detect("클로즈업 전신샷")
        self.assertEqual(fr, "full_body")
        self.assertEqual(found, ["closeup", "full_body"])
        self.assertEqual(framing.detect("얼빡 클로즈업")[0], "extreme_closeup")  # 둘 다 가까우면 더 가까운 쪽

    def test_ko_to_english_keeps_framing(self):
        self.assertIn("extreme close-up", framing.ko_to_english("얼빡샷 비키니 여자"))

    def test_region_rank(self):
        self.assertEqual(framing.region_rank("smiling softly at the camera"), 0)
        self.assertEqual(framing.region_rank("sitting with {her} chin in {her} hand"), 1)
        self.assertEqual(framing.region_rank("sitting cross-legged on the floor"), 5)
        self.assertTrue(framing.fits("biting {her} lower lip", "extreme_closeup"))
        self.assertFalse(framing.fits("standing confidently with one hand on {her} hip", "medium"))
        self.assertFalse(framing.fits("full-body shot", "closeup"))


class UndressingDetectTests(unittest.TestCase):
    def test_detect(self):
        for s in ["옷을 벗는 중인 여자", "옷을 벗는 장면", "블라우스 단추를 푸는 여자", "옷을 벗고 있는 여자"]:
            self.assertTrue(undressing.detect(s), s)
        for s in ["머리를 풀고 있는 여자", "탈의실의 여자", "다 벗은 여자"]:
            self.assertFalse(undressing.detect(s), s)
        self.assertTrue(undressing.detect("", "a woman undressing"))
        self.assertTrue(undressing.detect("", "taking off her white blouse"))
        self.assertFalse(undressing.detect("", "a striped dress"))
        self.assertFalse(undressing.detect("", "undressed"))

    def test_garment_noun(self):
        self.assertEqual(undressing.garment_noun("a School-style tennis micro mini pleated skirt", "bottom"),
                         "pleated skirt")
        self.assertEqual(undressing.garment_noun("a pastel ruffled bikini", "swimwear"), "ruffled bikini")
        self.assertIsNone(undressing.fill_garments("unzipping {her} {dress}", {"top": "blouse"}))


class GenerateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pm = PromptMaker(translator=NoTranslator())
        cls.pm.translator = DictionaryTranslator(cls.pm.ko_en)  # 네트워크 없이

    def gen(self, req, seed, **kw):
        return self.pm.generate(req, mode="random", seed=seed, **kw)

    def test_eolppak_is_face_only(self):
        for seed in range(25):
            r = self.gen("얼빡샷", seed)
            p = r.base_prompt.lower()
            self.assertEqual(r.lexicon["intent"]["framing"], "extreme_closeup")
            self.assertIn("extreme close-up", p)
            self.assertNotIn("quick", p)
            self.assertFalse(r.picks.get("attire"), p)
            self.assertIsNone(LEGS.search(p), p)
            self.assertNotIn("full-body", p)

    def test_requested_framing_has_no_conflicting_framing(self):
        for req, want in [("클로즈업 사진의 여자", "closeup"), ("미디엄샷 여자", "medium"), ("카우보이샷 여자", "cowboy"),
                          ("전신샷 여자", "full_body"), ("클로즈업 전신샷 여자", "full_body")]:
            for seed in range(20):
                r = self.gen(req, seed)
                for sentence in re.split(r"(?<=\.)\s", r.base_prompt):
                    self.assertTrue(framing.compatible(framing.framing_in_text(sentence), want),
                                    f"{req} / {seed}: {r.base_prompt}")
                if want in ("closeup", "medium"):
                    self.assertIsNone(LEGS.search(r.base_prompt), f"{req} / {seed}: {r.base_prompt}")
        r = self.gen("클로즈업 전신샷 여자", 1)
        self.assertTrue(any("샷 구도" in w for w in r.warnings))

    def test_framing_option_overrides_request(self):
        r = self.gen("전신샷 여자", 3, framing="extreme_closeup")
        self.assertIn("extreme close-up", r.base_prompt.lower())
        r = self.gen("얼빡샷", 3, framing="none")
        self.assertNotIn("extreme close-up", r.base_prompt.lower())
        with self.assertRaises(ValueError):
            self.gen("여자", 1, framing="wide")

    def test_undressing_action_names_worn_garment(self):
        for seed in range(25):
            r = self.gen("옷을 벗는 중인 여자", seed)
            self.assertTrue(r.lexicon["intent"]["undressing"])
            self.assertFalse(r.lexicon["intent"]["nude"])
            p = r.base_prompt
            self.assertTrue(r.picks.get("attire"), p)
            self.assertRegex(p, r"(unbutton|taking off|untying|slipping|sliding|unzip|pulling|peeling|loosening|"
                                r"shrugging|halfway out|unhooking|lifting|letting|reaching)", p)
            self.assertNotIn("{", p)
            self.assertNotIn("also features", p.lower())

    def test_undressing_requested_garment(self):
        for seed in range(15):
            r = self.gen("woman taking off her blouse", seed)
            self.assertIn("blouse", r.base_prompt.lower())

    def test_undressing_with_medium_shot_skips_lower_body(self):
        for seed in range(20):
            r = self.gen("미디엄샷 옷을 벗는 여자", seed)
            p = r.base_prompt
            self.assertNotRegex(p, r"down over (?:her|their) hips", p)
            self.assertIsNone(LEGS.search(p), p)

    def test_undressing_in_eolppak_is_dropped_with_warning(self):
        r = self.gen("얼빡샷 옷을 벗는 여자", 2)
        self.assertTrue(any("탈의 동작" in w for w in r.warnings))
        self.assertIn("extreme close-up", r.base_prompt.lower())

    def test_undressing_video(self):
        r = self.gen("옷을 벗는 중인 여자 영상", 4, prompt_type="video")
        self.assertEqual(r.prompt_type, "video")
        self.assertRegex(r.base_prompt, r"(unbutton|taking off|untying|slipping|sliding|unzip|pulling|peeling|"
                                        r"loosening|shrugging|halfway out|unhooking|lifting|letting|reaching)")

    def test_moss_tags_have_framing(self):
        r = self.gen("얼빡샷", 1, model_profile="moss")
        self.assertIn("extreme close-up", r.prompt)

    def test_info_lists_framings(self):
        ids = [f["id"] for f in self.pm.info()["framings"]]
        self.assertEqual(ids[0], "auto")
        self.assertIn("extreme_closeup", ids)


if __name__ == "__main__":
    unittest.main()
